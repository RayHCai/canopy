"""PhotonGeocoder: filtering, labels, dedup and the short-query circuit breaker."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from canopy.config import SiteCfg, load_config
from canopy.errors import GeodataError
from canopy.worldgen.geo.photon import PhotonGeocoder


def _feature(
    housenumber: str | None = "12",
    street: str = "Oak Street",
    city: str | None = "Springfield",
    state: str | None = "Illinois",
    postcode: str | None = "62704",
    country: str | None = "United States",
    osm_type: str = "W",
    osm_id: int = 123,
    lon: float = -75.0,
    lat: float = 40.0,
) -> dict[str, Any]:
    optional = {
        "housenumber": housenumber,
        "city": city,
        "state": state,
        "postcode": postcode,
        "country": country,
    }
    props: dict[str, Any] = {
        "osm_type": osm_type,
        "osm_id": osm_id,
        "street": street,
        **{key: value for key, value in optional.items() if value is not None},
    }
    return {"geometry": {"coordinates": [lon, lat]}, "properties": props}


class _FakeTransport:
    """Records the query and answers with a canned GeoJSON document."""

    def __init__(self, doc: Any) -> None:
        self.doc = doc
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get_json(self, url: str, params: Mapping[str, str]) -> Any:
        self.calls.append((url, dict(params)))
        return self.doc

    def post_form(self, url: str, form: Mapping[str, str]) -> Any:
        del url, form
        raise AssertionError("PhotonGeocoder never POSTs")


@pytest.fixture
def site_cfg() -> SiteCfg:
    """Return the shipped ``worldgen.site`` settings."""
    return load_config().worldgen.site


def test_a_query_shorter_than_three_characters_makes_no_request(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"features": []})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    assert geocoder.suggest(" 1", limit=5) == []
    assert geocoder.suggest("", limit=5) == []
    assert transport.calls == []


def test_suggest_sends_the_stripped_expanded_query_and_triple_the_limit(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"features": []})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    geocoder.suggest("  12 N Oak  ", limit=5)
    [(url, params)] = transport.calls
    assert url == site_cfg.geocoder_url
    assert params["q"] == "12 North Oak"
    assert params["limit"] == "15"


def test_suggest_builds_the_full_label_and_ref(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"features": [_feature()]})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    [address] = geocoder.suggest("12 oak", limit=5)
    assert address.label == "12 Oak Street, Springfield, Illinois 62704, United States"
    assert address.ref == "osm:way/123"
    assert address.provider == "photon"
    assert address.lat_deg == pytest.approx(40.0)
    assert address.lon_deg == pytest.approx(-75.0)


def test_suggest_omits_missing_label_parts(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"features": [_feature(state=None, postcode=None, country=None, city=None)]}
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    [address] = geocoder.suggest("12 oak", limit=5)
    assert address.label == "12 Oak Street"


def test_suggest_drops_features_without_a_housenumber(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"features": [_feature(housenumber=None), _feature(housenumber="14")]}
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/123"]
    assert results[0].label.startswith("14 ")


def test_suggest_dedupes_identical_labels(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"features": [_feature(osm_id=1), _feature(osm_id=2)]})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert len(results) == 1


def test_suggest_caps_results_at_the_limit(site_cfg: SiteCfg) -> None:
    features = [_feature(osm_id=i, housenumber=str(i)) for i in range(10)]
    transport = _FakeTransport({"features": features})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    assert len(geocoder.suggest("oak", limit=3)) == 3


def test_suggest_skips_a_malformed_feature_but_keeps_the_rest(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"features": [{"properties": {"housenumber": "1"}}, _feature(housenumber="2")]}
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/123"]


def test_suggest_raises_geodata_error_on_a_malformed_document(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"not_features": []})
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    with pytest.raises(GeodataError):
        geocoder.suggest("oak street", limit=5)


def test_suggest_filters_out_a_typed_house_number_mismatch(site_cfg: SiteCfg) -> None:
    """A typed number, once complete, only matches a non-digit extension: "120" is out."""
    transport = _FakeTransport(
        {
            "features": [
                _feature(housenumber="12", osm_id=1),
                _feature(housenumber="3392", osm_id=2),
                _feature(housenumber="120", osm_id=3),
            ]
        }
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("12 Oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/1"]


def test_suggest_reranks_a_feature_photon_listed_second(site_cfg: SiteCfg) -> None:
    """A feature matching more typed words is promoted, whatever order Photon answered in."""
    transport = _FakeTransport(
        {
            "features": [
                _feature(street="Oak Road", city="Springfield", osm_id=1),
                _feature(street="Oak Park Avenue", city="Springfield", osm_id=2),
            ]
        }
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("12 Oak Park", limit=5)
    assert results[0].ref == "osm:way/2"


def test_suggest_countries_drops_a_feature_outside_the_configured_countries(
    site_cfg: SiteCfg,
) -> None:
    cfg = dataclasses.replace(site_cfg, suggest_countries=("us",))
    us_feature = _feature(osm_id=1)
    us_feature["properties"]["countrycode"] = "US"
    ca_feature = _feature(osm_id=2)
    ca_feature["properties"]["countrycode"] = "CA"
    transport = _FakeTransport({"features": [us_feature, ca_feature]})
    geocoder = PhotonGeocoder(cfg, transport=transport)
    results = geocoder.suggest("12 oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/1"]


def test_suggest_skips_a_feature_whose_properties_is_not_a_dict(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"features": [{"properties": ["not", "a", "dict"]}, _feature(housenumber="2")]}
    )
    geocoder = PhotonGeocoder(site_cfg, transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/123"]
