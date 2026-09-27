"""fetch_site end to end, against a fake FeatureSource: no network, ever.

Fixtures are metric layouts (plain ENU offsets from the pin), turned into
lat/lon with :func:`canopy.worldgen.geo.frames.geodetic_from_enu` so the
numbers in each test read as the metres they are.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Polygon

from canopy.config import Config
from canopy.contracts import ResidentialDecision, ResolvedAddress
from canopy.errors import GeodataError, SiteRejectedError
from canopy.worldgen.geo import frames
from canopy.worldgen.geo.build import default_geocoder, fetch_site
from canopy.worldgen.geo.geoapify import GeoapifyGeocoder
from canopy.worldgen.geo.overpass import OsmFeatures, OsmNode, OsmWay
from canopy.worldgen.geo.photon import PhotonGeocoder
from canopy.worldgen.snapshot import load_snapshot, save_snapshot

#: The address pin. ``frames`` always projects about the pin, so this is ENU (0, 0).
_ANCHOR_LAT, _ANCHOR_LON = 40.0, -75.0


def _way(way_id: int, tags: dict[str, str], outline_enu: npt.NDArray[np.float64]) -> OsmWay:
    """One way from a closed ENU ring (first point repeated, like a real OSM way)."""
    closed = np.vstack([outline_enu, outline_enu[:1]])
    lat, lon = frames.geodetic_from_enu(closed, _ANCHOR_LAT, _ANCHOR_LON)
    return OsmWay(id=way_id, tags=tags, lat=lat, lon=lon)


def _relation_way(
    way_id: int, tags: dict[str, str], outline_enu: npt.NDArray[np.float64]
) -> OsmWay:
    """One building *relation*'s already-assembled outer ring, else identical to :func:`_way`."""
    closed = np.vstack([outline_enu, outline_enu[:1]])
    lat, lon = frames.geodetic_from_enu(closed, _ANCHOR_LAT, _ANCHOR_LON)
    return OsmWay(id=way_id, tags=tags, lat=lat, lon=lon, kind="relation")


def _node(node_id: int, tags: dict[str, str], xy: tuple[float, float]) -> OsmNode:
    lat, lon = frames.geodetic_from_enu(np.array(xy), _ANCHOR_LAT, _ANCHOR_LON)
    return OsmNode(id=node_id, tags=tags, lat=float(lat), lon=float(lon))


def _rect(x0: float, y0: float, x1: float, y1: float) -> npt.NDArray[np.float64]:
    """Counter-clockwise rectangle outline, shape ``(4, 2)``."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


class _FakeSource:
    """Answers every query with the same canned :class:`OsmFeatures`."""

    def __init__(self, osm: OsmFeatures) -> None:
        self.osm = osm

    def features(self, lat_deg: float, lon_deg: float, radius_m: float) -> OsmFeatures:
        del lat_deg, lon_deg, radius_m
        return self.osm


#: A house tag that, alone, comfortably clears accept_p in the shipped config.
_HOUSE_TAGS = {"building": "house", "building:levels": "2"}
#: 10 x 8 m footprint, centred on the pin.
_HOUSE_ENU = _rect(-5.0, -4.0, 5.0, 4.0)

_ADDRESS = ResolvedAddress(
    label="1 Test Lane, Testville",
    provider="fixture",
    ref="osm:way/1",
    lat_deg=_ANCHOR_LAT,
    lon_deg=_ANCHOR_LON,
)


def _lot_corners_enu(
    origin_enu_m: npt.NDArray[np.float64], north_rad: float, lot_m: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Return the lot's own two extreme corners, in ENU, reconstructed from the snapshot fields.

    The lot always spans exactly ``+/- lot_m / 2`` in the world frame (the
    world origin *is* the lot centre), so projecting those two corners back
    through :func:`~canopy.worldgen.geo.frames.enu_from_world` recovers the
    lot boundary in the same ENU metres the fixtures are written in, without
    the test re-deriving the fetch step's own rotation-and-shrink algebra.
    """
    half = np.array([lot_m[0] / 2.0, lot_m[1] / 2.0])
    corners_world = np.array([-half, half])
    return frames.enu_from_world(corners_world, origin_enu_m, north_rad)


def test_target_by_ref(cfg: Config) -> None:
    """A way id matching ``address.ref`` is the target, even if it does not contain the pin."""
    offset_house = _HOUSE_ENU + np.array([100.0, 100.0])  # pin no longer inside it
    decoy = _rect(-3.0, -3.0, 3.0, 3.0)  # this one *does* contain the pin
    osm = OsmFeatures(
        ways=(
            _way(1, _HOUSE_TAGS, offset_house),
            _way(2, {"building": "house"}, decoy),
        ),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:way/1"


def test_target_by_containment(cfg: Config) -> None:
    """With no matching ref, the building containing the pin is the target."""
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:way/999",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    decoy = _rect(20.0, 20.0, 26.0, 26.0)
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU), _way(2, {"building": "house"}, decoy)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:way/1"


def test_target_by_nearest_within_snap_radius(cfg: Config) -> None:
    """With no ref match and no building over the pin, the nearest one within range wins."""
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:way/999",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    near = _HOUSE_ENU + np.array([10.0, 0.0])  # just past the footprint, not containing the pin
    far = _HOUSE_ENU + np.array([200.0, 0.0])
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, near), _way(2, {"building": "house"}, far)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:way/1"
    assert cfg.worldgen.site.snap_radius_m < 200.0  # sanity: "far" really is out of range


def test_nothing_within_snap_radius_raises_geodata_error(cfg: Config) -> None:
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:way/999",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    far = _HOUSE_ENU + np.array([1000.0, 0.0])
    osm = OsmFeatures(ways=(_way(1, _HOUSE_TAGS, far),), nodes=(), landuse=(), timestamp=None)
    with pytest.raises(GeodataError):
        fetch_site(address, cfg, source=_FakeSource(osm))


def test_street_ends_up_along_minus_y(cfg: Config) -> None:
    """The street faces ``-y``, and an unrotated map's north stays at ``pi / 2``."""
    street = np.array([[-50.0, -30.0], [50.0, -30.0]])
    osm = OsmFeatures(
        ways=(
            _way(1, _HOUSE_TAGS, _HOUSE_ENU),
            _way(2, {"highway": "residential", "name": "Oak Street"}, street),
        ),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))

    assert snapshot.street is not None
    assert float(snapshot.street[:, 1].max()) < float(snapshot.house.footprint[:, 1].min())
    # The house's footprint (in ENU) is already axis-aligned with the street
    # due south, so no quarter turn was needed: north stays where a seed-only
    # property's does.
    assert snapshot.north_rad == pytest.approx(math.pi / 2.0)


def test_lot_boundary_sits_at_the_midpoint_to_a_side_neighbour(cfg: Config) -> None:
    site = cfg.worldgen.site
    gap = 3.0
    # Shifting the whole 10 m-wide house shape by (house width + gap) puts its
    # new left edge exactly `gap` past the original house's right edge (5 m).
    neighbour = _HOUSE_ENU + np.array([10.0 + gap, 0.0])
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU), _way(2, {"building": "house"}, neighbour)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))

    lo, hi = _lot_corners_enu(snapshot.origin_enu_m, snapshot.north_rad, snapshot.lot_m)
    assert gap > 2.0 * site.party_wall_gap_m  # sanity: this is a shrink, not a party wall
    # snapshot.py rounds coordinates to a micrometre, which moves a lot corner
    # a few micrometres at this scale; 1e-4 m (0.1 mm) absorbs that without
    # hiding a real placement bug.
    assert float(hi[0]) == pytest.approx(5.0 + gap / 2.0, abs=1e-4)
    # The untouched side keeps the default yard.
    assert float(lo[0]) == pytest.approx(-5.0 - site.default_side_yard_m, abs=1e-4)


def test_party_wall_neighbour_gets_no_side_yard(cfg: Config) -> None:
    site = cfg.worldgen.site
    # A tiny gap (half the party-wall threshold) past the house's own right
    # edge (5 m): close enough to read as a shared wall, not a side yard.
    touching = _HOUSE_ENU + np.array([10.0 + site.party_wall_gap_m / 2.0, 0.0])
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU), _way(2, {"building": "house"}, touching)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    _lo, hi = _lot_corners_enu(snapshot.origin_enu_m, snapshot.north_rad, snapshot.lot_m)
    # snapshot.py rounds coordinates to a micrometre, which moves a lot corner
    # a few micrometres at this scale; 1e-4 m (0.1 mm) absorbs that without
    # hiding a real placement bug.
    assert float(hi[0]) == pytest.approx(5.0, abs=1e-4)


def test_min_front_yard_is_enforced_against_a_close_street(cfg: Config) -> None:
    site = cfg.worldgen.site
    close_street = np.array([[-50.0, -6.0], [50.0, -6.0]])  # 2 m from the house's front wall
    osm = OsmFeatures(
        ways=(
            _way(1, _HOUSE_TAGS, _HOUSE_ENU),
            _way(2, {"highway": "residential"}, close_street),
        ),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    lo, _hi = _lot_corners_enu(snapshot.origin_enu_m, snapshot.north_rad, snapshot.lot_m)
    # snapshot.py rounds coordinates to a micrometre, which moves a lot corner
    # a few micrometres at this scale; 1e-4 m (0.1 mm) absorbs that without
    # hiding a real placement bug.
    assert float(lo[1]) == pytest.approx(-4.0 - site.min_front_yard_m, abs=1e-4)


def test_envelope_rejection_for_an_oversized_footprint(cfg: Config) -> None:
    huge = _rect(-15.0, -15.0, 15.0, 15.0)  # 30 x 30 m, over max_footprint_m2
    osm = OsmFeatures(ways=(_way(1, _HOUSE_TAGS, huge),), nodes=(), landuse=(), timestamp=None)
    with pytest.raises(SiteRejectedError):
        fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))


def test_residential_reject_raises_site_rejected_error(cfg: Config) -> None:
    office = _way(1, {"building": "office"}, _HOUSE_ENU)
    osm = OsmFeatures(ways=(office,), nodes=(), landuse=(), timestamp=None)
    with pytest.raises(SiteRejectedError, match="p="):
        fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))


def test_residential_ask_returns_a_snapshot(cfg: Config) -> None:
    generic = _way(1, {"building": "yes", "building:levels": "2"}, _HOUSE_ENU)
    osm = OsmFeatures(ways=(generic,), nodes=(), landuse=(), timestamp=None)
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    assert snapshot.verdict.decision is ResidentialDecision.ASK


def test_trees_and_poles_are_projected_into_the_world_frame(cfg: Config) -> None:
    tree = _node(10, {"natural": "tree", "diameter_crown": "6"}, (8.0, 10.0))
    pole = _node(11, {"power": "pole"}, (14.0, -2.0))
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU),), nodes=(tree, pole), landuse=(), timestamp=None
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    assert snapshot.trees.shape == (1, 3)
    assert snapshot.trees[0, 2] == pytest.approx(3.0)  # half of diameter_crown
    assert snapshot.poles.shape == (1, 2)


def test_snapshot_round_trips_through_save_and_load(cfg: Config, tmp_path: Path) -> None:
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU),), nodes=(), landuse=(), timestamp=None
    )
    snapshot = fetch_site(
        _ADDRESS, cfg, source=_FakeSource(osm), now=lambda: "2026-09-26T00:00:00Z"
    )
    path = save_snapshot(snapshot, out_dir=tmp_path)
    loaded = load_snapshot(path)
    assert loaded.site_id == snapshot.site_id
    assert loaded.site_id != ""


def test_target_by_ref_matches_a_relation_not_a_way_of_the_same_id(cfg: Config) -> None:
    """``osm:relation/<id>`` matches a relation; a way with the same numeric id is not it."""
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:relation/1",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    offset_house = _HOUSE_ENU + np.array([100.0, 100.0])  # pin not inside it
    decoy_way = _rect(-3.0, -3.0, 3.0, 3.0)  # a *way*, id=1 too, and it *does* hold the pin
    osm = OsmFeatures(
        ways=(
            _way(1, {"building": "house"}, decoy_way),
            _relation_way(1, _HOUSE_TAGS, offset_house),
        ),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:relation/1"


def test_pin_inside_a_garage_prefers_the_house_a_few_metres_away(cfg: Config) -> None:
    """Past an exact ref match, an outbuilding is skipped for containment, even holding the pin."""
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:way/999",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    garage = _rect(-2.0, -2.0, 2.0, 2.0)  # contains the pin
    house = _HOUSE_ENU + np.array([10.0, 0.0])  # a few metres off; does not contain the pin
    osm = OsmFeatures(
        ways=(_way(1, {"building": "garage"}, garage), _way(2, _HOUSE_TAGS, house)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:way/2"


def test_only_a_garage_mapped_is_still_chosen_rather_than_rejected(cfg: Config) -> None:
    """With nothing else mapped, the outbuilding exclusion falls back instead of raising."""
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="fixture",
        ref="osm:way/999",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
    )
    garage = _rect(-2.0, -2.0, 2.0, 2.0)  # contains the pin; nothing else is mapped at all
    osm = OsmFeatures(
        ways=(_way(1, {"building": "garage"}, garage),), nodes=(), landuse=(), timestamp=None
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    assert snapshot.house.source == "osm:way/1"


def test_bowtie_neighbour_is_repaired_to_a_valid_ccw_polygon(cfg: Config) -> None:
    """A self-intersecting mapped outline (a real if rare mapping error) is repaired, not fatal."""
    # A 16 x 16 square visited corner, opposite corner, corner, opposite
    # corner, so the outline crosses itself at the centre -- a bowtie, the
    # rare but real mapping error _ccw repairs via buffer(0). One 64 m^2
    # triangle survives the repair, comfortably over _MIN_NEIGHBOUR_AREA_M2;
    # shifted well clear of the house so it does not also touch the
    # lot-shrink logic under test elsewhere.
    bowtie = np.array([[30.0, 30.0], [46.0, 46.0], [46.0, 30.0], [30.0, 46.0]])
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU), _way(2, {"building": "house"}, bowtie)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))  # must not raise
    assert len(snapshot.neighbours) == 1
    footprint = snapshot.neighbours[0].footprint
    polygon = Polygon(footprint)
    assert polygon.is_valid
    assert polygon.exterior.is_ccw
    assert len(footprint) == 3  # the repaired triangle, not all 4 of the bowtie's raw vertices


def test_diagonal_neighbour_off_a_corner_leaves_the_default_side_yard(cfg: Config) -> None:
    """A neighbour overlapping the house along neither x nor y shares no wall, shrinks nothing."""
    site = cfg.worldgen.site
    offset = 0.3  # well inside party_wall_gap_m, but purely diagonal off the corner
    assert offset < site.party_wall_gap_m  # sanity: this is the close-but-no-overlap case
    # A 6 x 6 m building whose nearest corner sits `offset` diagonally past the
    # house's rear-right corner (5, 4): no overlap with the house along x or y.
    neighbour = _rect(5.0 + offset, 4.0 + offset, 11.0 + offset, 10.0 + offset)
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU), _way(2, {"building": "house"}, neighbour)),
        nodes=(),
        landuse=(),
        timestamp=None,
    )
    snapshot = fetch_site(_ADDRESS, cfg, source=_FakeSource(osm))
    _lo, hi = _lot_corners_enu(snapshot.origin_enu_m, snapshot.north_rad, snapshot.lot_m)
    # snapshot.py rounds coordinates to a micrometre, which moves a lot corner
    # a few micrometres at this scale; 1e-4 m (0.1 mm) absorbs that without
    # hiding a real placement bug.
    assert float(hi[0]) == pytest.approx(5.0 + site.default_side_yard_m, abs=1e-4)


# ---------------------------------------------------------------------------
# default_geocoder
# ---------------------------------------------------------------------------
def test_default_geocoder_auto_with_key_present_picks_geoapify(cfg: Config) -> None:
    site = cfg.worldgen.site
    geocoder = default_geocoder(site, environ={site.geoapify_key_env: "a-key"})
    assert isinstance(geocoder, GeoapifyGeocoder)


@pytest.mark.parametrize("environ", [{}, {"geoapify_key_env": "   "}])
def test_default_geocoder_auto_with_no_or_blank_key_falls_back_to_photon(
    cfg: Config, environ: dict[str, str]
) -> None:
    site = cfg.worldgen.site
    env = {site.geoapify_key_env: environ["geoapify_key_env"]} if environ else {}
    geocoder = default_geocoder(site, environ=env)
    assert isinstance(geocoder, PhotonGeocoder)


def test_default_geocoder_photon_with_a_key_still_returns_photon(cfg: Config) -> None:
    site = dataclasses.replace(cfg.worldgen.site, geocoder="photon")
    geocoder = default_geocoder(site, environ={site.geoapify_key_env: "a-key"})
    assert isinstance(geocoder, PhotonGeocoder)


def test_default_geocoder_geoapify_with_no_key_raises_naming_the_env_var(cfg: Config) -> None:
    site = dataclasses.replace(cfg.worldgen.site, geocoder="geoapify")
    with pytest.raises(GeodataError, match=site.geoapify_key_env):
        default_geocoder(site, environ={})


# ---------------------------------------------------------------------------
# Geocoder source record attribution
# ---------------------------------------------------------------------------
def test_geoapify_source_record_keeps_the_resolved_per_match_attribution(cfg: Config) -> None:
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="geoapify",
        ref="",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
        attribution="Powered by Geoapify; © OpenAddresses",
    )
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU),), nodes=(), landuse=(), timestamp=None
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    geo_source = next(s for s in snapshot.sources if s.provider == "geoapify")
    assert geo_source.attribution == "Powered by Geoapify; © OpenAddresses"


def test_photon_source_record_falls_back_to_the_standing_credit(cfg: Config) -> None:
    address = ResolvedAddress(
        label=_ADDRESS.label,
        provider="photon",
        ref="",
        lat_deg=_ANCHOR_LAT,
        lon_deg=_ANCHOR_LON,
        attribution="",
    )
    osm = OsmFeatures(
        ways=(_way(1, _HOUSE_TAGS, _HOUSE_ENU),), nodes=(), landuse=(), timestamp=None
    )
    snapshot = fetch_site(address, cfg, source=_FakeSource(osm))
    geo_source = next(s for s in snapshot.sources if s.provider == "photon")
    assert geo_source.attribution == "(c) OpenStreetMap contributors, geocoding by Photon (komoot)"
