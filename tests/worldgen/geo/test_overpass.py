"""OverpassSource: the query text it sends, and how it parses the answer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from canopy.config import SiteCfg, load_config
from canopy.errors import GeodataError
from canopy.worldgen.geo.overpass import OverpassSource


class _FakeTransport:
    """Records the Overpass query and answers with a canned document."""

    def __init__(self, doc: Any) -> None:
        self.doc = doc
        self.forms: list[Mapping[str, str]] = []

    def get_json(self, url: str, params: Mapping[str, str]) -> Any:
        del url, params
        raise AssertionError("OverpassSource never GETs")

    def post_form(self, url: str, form: Mapping[str, str]) -> Any:
        del url
        self.forms.append(dict(form))
        return self.doc


def _member(points: list[tuple[float, float]], *, role: str = "outer") -> dict[str, Any]:
    """One ``way`` member of a multipolygon relation, geometry given as ``(lat, lon)`` pairs."""
    return {
        "type": "way",
        "role": role,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in points],
    }


def _relation(rel_id: int, tags: dict[str, str], members: list[dict[str, Any]]) -> dict[str, Any]:
    """One Overpass ``relation`` element carrying the given ``members``."""
    return {"type": "relation", "id": rel_id, "tags": tags, "members": members}


@pytest.fixture
def site_cfg() -> SiteCfg:
    """Return the shipped ``worldgen.site`` settings."""
    return load_config().worldgen.site


def test_features_sends_one_query_with_the_bbox_substituted_everywhere(
    site_cfg: SiteCfg,
) -> None:
    transport = _FakeTransport({"elements": []})
    source = OverpassSource(site_cfg, transport=transport)
    source.features(40.0, -75.0, radius_m=100.0)

    [form] = transport.forms
    query = form["data"]
    assert query.startswith(f"[out:json][timeout:{int(site_cfg.overpass_query_timeout_s)}];")
    # No area-database lookup: it is what made the public server time out.
    assert "is_in" not in query
    assert "area" not in query
    assert query.endswith("out geom;")

    # Extract the bbox Overpass QL puts after "building", then check every
    # other tag filter was substituted with that exact same bounding box.
    start = query.index('way["building"](') + len('way["building"](')
    bbox_text = query[start : query.index(")", start)]
    assert bbox_text.count(",") == 3  # south,west,north,east
    for clause in (
        'way["building"](',
        'way["highway"](',
        'node["natural"="tree"](',
        'node["power"~"^(pole|tower)$"](',
        'node["shop"](',
        'node["office"](',
        'node["amenity"](',
        'way["landuse"](',
    ):
        assert f"{clause}{bbox_text})" in query


def test_features_bbox_is_a_square_half_width_radius_m_about_the_point(
    site_cfg: SiteCfg,
) -> None:
    transport = _FakeTransport({"elements": []})
    source = OverpassSource(site_cfg, transport=transport)
    source.features(0.0, 0.0, radius_m=1000.0)
    query = transport.forms[0]["data"]
    # South-west and north-east corners of a 1000 m half-width square at the
    # equator: about 0.009 degrees in both directions.
    first_clause = query.split(";", 2)[1]
    numbers = [float(v) for v in first_clause[first_clause.rindex("(") + 1 : -1].split(",")]
    south, west, north, east = numbers
    assert south == pytest.approx(-north)
    assert west == pytest.approx(-east)
    assert north == pytest.approx(0.009043, abs=1e-4)


def _closed_way(
    way_id: int, tags: dict[str, str], lat0: float, lon0: float, size: float
) -> dict[str, Any]:
    """Build a closed square way with its south-west corner at ``(lat0, lon0)``."""
    corners = [(lat0, lon0), (lat0 + size, lon0), (lat0 + size, lon0 + size), (lat0, lon0 + size)]
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in [*corners, corners[0]]],
    }


def test_features_parses_ways_nodes_landuse_and_timestamp(site_cfg: SiteCfg) -> None:
    doc = {
        "elements": [
            {
                "type": "way",
                "id": 111,
                "tags": {"building": "house"},
                "geometry": [
                    {"lat": 40.001, "lon": -75.001},
                    {"lat": 40.002, "lon": -75.001},
                    {"lat": 40.002, "lon": -75.002},
                ],
            },
            {
                "type": "node",
                "id": 222,
                "tags": {"natural": "tree"},
                "lat": 40.0015,
                "lon": -75.0015,
            },
            # Contains the query point (40, -75): its landuse counts.
            _closed_way(333, {"landuse": "residential"}, 39.999, -75.001, 0.002),
            # Inside the fetched area but not under the pin: ignored.
            _closed_way(444, {"landuse": "retail"}, 40.0005, -74.9995, 0.0003),
        ],
        "osm3s": {"timestamp_osm_base": "2026-09-25T12:00:00Z"},
    }
    transport = _FakeTransport(doc)
    source = OverpassSource(site_cfg, transport=transport)
    features = source.features(40.0, -75.0, radius_m=200.0)

    # Landuse ways are evidence about the pin, never scene geometry.
    assert len(features.ways) == 1
    way = features.ways[0]
    assert way.id == 111
    assert way.tags == {"building": "house"}
    assert np.allclose(way.lat, [40.001, 40.002, 40.002])
    assert np.allclose(way.lon, [-75.001, -75.001, -75.002])

    assert len(features.nodes) == 1
    node = features.nodes[0]
    assert (node.id, node.tags, node.lat, node.lon) == (222, {"natural": "tree"}, 40.0015, -75.0015)

    assert features.landuse == ("residential",)
    assert features.timestamp == "2026-09-25T12:00:00Z"


def test_an_open_landuse_way_is_never_taken_to_contain_the_pin(site_cfg: SiteCfg) -> None:
    open_way = _closed_way(1, {"landuse": "residential"}, 39.999, -75.001, 0.002)
    open_way["geometry"] = open_way["geometry"][:-1]  # drop the closing point
    source = OverpassSource(site_cfg, transport=_FakeTransport({"elements": [open_way]}))
    assert source.features(40.0, -75.0, radius_m=200.0).landuse == ()


def test_a_way_both_landuse_and_building_stays_in_the_scene(site_cfg: SiteCfg) -> None:
    both = _closed_way(1, {"landuse": "residential", "building": "house"}, 39.999, -75.001, 0.002)
    source = OverpassSource(site_cfg, transport=_FakeTransport({"elements": [both]}))
    features = source.features(40.0, -75.0, radius_m=200.0)
    assert [w.id for w in features.ways] == [1]
    assert features.landuse == ()


def test_a_query_overpass_stopped_early_raises_rather_than_building_from_part(
    site_cfg: SiteCfg,
) -> None:
    doc = {
        "elements": [_closed_way(1, {"building": "house"}, 39.999, -75.001, 0.0002)],
        "remark": 'runtime error: Query timed out in "query" at line 1 after 25 seconds.',
    }
    source = OverpassSource(site_cfg, transport=_FakeTransport(doc))
    with pytest.raises(GeodataError, match="stopped the query"):
        source.features(40.0, -75.0, radius_m=200.0)


def test_features_timestamp_is_none_when_the_answer_omits_it(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"elements": []})
    source = OverpassSource(site_cfg, transport=transport)
    assert source.features(40.0, -75.0, radius_m=50.0).timestamp is None


def test_features_skips_a_malformed_element_but_keeps_the_rest(site_cfg: SiteCfg) -> None:
    doc = {
        "elements": [
            {"type": "way", "id": 1},  # missing "geometry": malformed, skipped
            {"type": "node", "id": 2, "tags": {}, "lat": 1.0, "lon": 2.0},
        ]
    }
    transport = _FakeTransport(doc)
    source = OverpassSource(site_cfg, transport=transport)
    features = source.features(40.0, -75.0, radius_m=50.0)
    assert features.ways == ()
    assert [n.id for n in features.nodes] == [2]


def test_features_raises_geodata_error_on_a_malformed_document(site_cfg: SiteCfg) -> None:
    transport = _FakeTransport({"no_elements_here": True})
    source = OverpassSource(site_cfg, transport=transport)
    with pytest.raises(GeodataError):
        source.features(40.0, -75.0, radius_m=50.0)


#: Corners of a small square: the same ring a multipolygon relation's outer
#: member ways are split across in the tests below.
_P0, _P1, _P2, _P3 = (50.0, 10.0), (50.001, 10.0), (50.001, 10.001), (50.0, 10.001)


def test_query_includes_the_relation_clause_with_bbox_substituted(site_cfg: SiteCfg) -> None:
    """A house mapped only as a multipolygon relation must still be fetched."""
    transport = _FakeTransport({"elements": []})
    source = OverpassSource(site_cfg, transport=transport)
    source.features(40.0, -75.0, radius_m=100.0)
    query = transport.forms[0]["data"]

    start = query.index('way["building"](') + len('way["building"](')
    bbox_text = query[start : query.index(")", start)]
    assert f'relation["building"]["type"="multipolygon"]({bbox_text})' in query


def test_outer_ring_across_two_member_ways_with_one_reversed(site_cfg: SiteCfg) -> None:
    """A ring split into two ways, one of them stored running the other way, still closes."""
    members = [
        _member([_P0, _P1, _P2]),
        _member([_P0, _P3, _P2]),  # the p2->p3->p0 edge, written back to front
    ]
    doc = {"elements": [_relation(555, {"building": "yes", "type": "multipolygon"}, members)]}
    transport = _FakeTransport(doc)
    source = OverpassSource(site_cfg, transport=transport)
    features = source.features(40.0, -75.0, radius_m=100.0)

    assert len(features.ways) == 1
    way = features.ways[0]
    assert way.kind == "relation"
    assert way.id == 555
    assert np.allclose(way.lat, [_P0[0], _P1[0], _P2[0], _P3[0], _P0[0]])
    assert np.allclose(way.lon, [_P0[1], _P1[1], _P2[1], _P3[1], _P0[1]])


def test_outer_ring_across_three_member_ways_with_one_reversed(site_cfg: SiteCfg) -> None:
    """A ring split into three ways, the middle one reversed, still closes in order."""
    members = [
        _member([_P0, _P1]),
        _member([_P2, _P1]),  # the p1->p2 edge, written back to front
        _member([_P2, _P3, _P0]),
    ]
    doc = {"elements": [_relation(556, {"building": "yes", "type": "multipolygon"}, members)]}
    transport = _FakeTransport(doc)
    source = OverpassSource(site_cfg, transport=transport)
    features = source.features(40.0, -75.0, radius_m=100.0)

    assert len(features.ways) == 1
    way = features.ways[0]
    assert way.kind == "relation"
    assert np.allclose(way.lat, [_P0[0], _P1[0], _P2[0], _P3[0], _P0[0]])
    assert np.allclose(way.lon, [_P0[1], _P1[1], _P2[1], _P3[1], _P0[1]])


def test_relation_whose_outer_chain_never_closes_is_dropped(site_cfg: SiteCfg) -> None:
    """Two member ways sharing no endpoint cannot form a ring, so the relation is skipped."""
    members = [_member([_P0, _P1]), _member([_P2, _P3])]  # opposite, disjoint edges
    doc = {"elements": [_relation(557, {"building": "yes", "type": "multipolygon"}, members)]}
    transport = _FakeTransport(doc)
    source = OverpassSource(site_cfg, transport=transport)
    features = source.features(40.0, -75.0, radius_m=100.0)
    assert features.ways == ()
