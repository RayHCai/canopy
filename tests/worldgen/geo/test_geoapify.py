"""GeoapifyGeocoder: request shape, label/ref/attribution mapping, filtering and dedup."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from canopy.config import SiteCfg, load_config
from canopy.errors import GeodataError
from canopy.worldgen.geo.geoapify import GeoapifyGeocoder


def _result(
    housenumber: str | None = "12",
    street: str = "Oak Street",
    formatted: str = "12 Oak Street, Springfield, Illinois, United States",
    lat: float = 40.0,
    lon: float = -75.0,
    place_id: str = "abc123",
    sourcename: str | None = "openstreetmap",
    attribution: str | None = "(c) OpenStreetMap contributors",
    osm_type: str = "w",
    osm_id: int = 123,
) -> dict[str, Any]:
    datasource: dict[str, Any] | None = None
    if sourcename is not None:
        datasource = {
            "sourcename": sourcename,
            "attribution": attribution,
            "raw": {"osm_type": osm_type, "osm_id": osm_id},
        }
    result: dict[str, Any] = {
        "street": street,
        "formatted": formatted,
        "lat": lat,
        "lon": lon,
        "place_id": place_id,
    }
    if housenumber is not None:
        result["housenumber"] = housenumber
    if datasource is not None:
        result["datasource"] = datasource
    return result


class _FakeTransport:
    """Records the query and answers with a canned document."""

    def __init__(self, doc: Any) -> None:
        self.doc = doc
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get_json(self, url: str, params: Mapping[str, str]) -> Any:
        self.calls.append((url, dict(params)))
        return self.doc

    def post_form(self, url: str, form: Mapping[str, str]) -> Any:
        del url, form
        raise AssertionError("GeoapifyGeocoder never POSTs")


@pytest.fixture
def site_cfg() -> SiteCfg:
    """Return the shipped ``worldgen.site`` settings."""
    return load_config().worldgen.site


def test_a_query_shorter_than_three_characters_makes_no_request(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"results": []})
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    assert geocoder.suggest(" 1", limit=5) == []
    assert geocoder.suggest("", limit=5) == []
    assert transport.calls == []


def test_suggest_sends_expanded_text_double_the_limit_and_the_key(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"results": []})
    geocoder = GeoapifyGeocoder(site_cfg, "secret-key", transport=transport)
    geocoder.suggest("  12 N Oak  ", limit=5)
    [(url, params)] = transport.calls
    assert url == site_cfg.geoapify_url
    assert params["text"] == "12 North Oak"
    assert params["limit"] == "10"
    assert params["format"] == "json"
    assert params["apiKey"] == "secret-key"
    assert "filter" not in params


def test_suggest_countries_adds_a_countrycode_filter(site_cfg: SiteCfg) -> None:
    cfg = dataclasses.replace(site_cfg, suggest_countries=("us", "ca"))
    transport = _FakeTransport({"results": []})
    geocoder = GeoapifyGeocoder(cfg, "key", transport=transport)
    geocoder.suggest("12 oak", limit=5)
    [(_url, params)] = transport.calls
    assert params["filter"] == "countrycode:us,ca"


def test_suggest_builds_label_ref_provider_and_attribution(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"results": [_result()]})
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    [address] = geocoder.suggest("12 oak", limit=5)
    assert address.label == "12 Oak Street, Springfield, Illinois, United States"
    assert address.provider == "geoapify"
    assert address.ref == "osm:way/123"
    assert address.attribution == "Powered by Geoapify; (c) OpenStreetMap contributors"
    assert address.lat_deg == pytest.approx(40.0)
    assert address.lon_deg == pytest.approx(-75.0)


def test_ref_recognizes_both_osm_type_spellings(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"results": [_result(place_id="p1", osm_type="w"), _result(place_id="p2", osm_type="way")]}
    )
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert {a.ref for a in results} == {"osm:way/123"}  # both collapse to the same osm ref


def test_ref_falls_back_to_geoapify_place_id_for_a_non_osm_source(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"results": [_result(sourcename="openaddresses", place_id="place-99")]}
    )
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    [address] = geocoder.suggest("oak", limit=5)
    assert address.ref == "geoapify:place-99"


def test_attribution_is_bare_geoapify_credit_when_datasource_has_none(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"results": [_result(sourcename=None)]})
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    [address] = geocoder.suggest("oak", limit=5)
    assert address.attribution == "Powered by Geoapify"


def test_suggest_drops_results_without_a_housenumber(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"results": [_result(housenumber=None, place_id="a"), _result(place_id="b")]}
    )
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert [a.ref for a in results] == ["osm:way/123"]


def test_suggest_dedupes_identical_labels(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"results": [_result(place_id="a"), _result(place_id="b")]})
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert len(results) == 1


def test_suggest_filters_out_a_typed_house_number_mismatch(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {
            "results": [
                _result(housenumber="12", place_id="a"),
                _result(housenumber="3392", place_id="b"),
                _result(housenumber="120", place_id="c"),
            ]
        }
    )
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    results = geocoder.suggest("12 Oak", limit=5)
    assert [a.label for a in results] == ["12 Oak Street, Springfield, Illinois, United States"]


def test_suggest_skips_a_malformed_result_but_keeps_the_rest(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport(
        {"results": [{"housenumber": "1"}, _result(housenumber="2", place_id="b")]}
    )
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    results = geocoder.suggest("oak", limit=5)
    assert len(results) == 1


def test_suggest_raises_geodata_error_on_a_malformed_document(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"not_results": []})
    geocoder = GeoapifyGeocoder(site_cfg, "key", transport=transport)
    with pytest.raises(GeodataError):
        geocoder.suggest("oak street", limit=5)


def test_empty_api_key_raises_geodata_error_naming_the_env_var(site_cfg: SiteCfg) -> None:
    with pytest.raises(GeodataError, match=site_cfg.geoapify_key_env):
        GeoapifyGeocoder(site_cfg, "   ")


def test_blank_api_key_raises_before_any_key_reaches_a_params_dict(site_cfg: SiteCfg) -> None:
    """Empty means never getting far enough to build a request, so no key value is ever logged."""
    with pytest.raises(GeodataError):
        GeoapifyGeocoder(site_cfg, "")
