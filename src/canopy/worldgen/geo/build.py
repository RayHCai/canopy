"""Address to :class:`~canopy.contracts.SiteSnapshot`: the fetch step, end to end.

:func:`suggest_addresses` and :func:`fetch_site` are the only two names this
module exports, and the only place network I/O happens in Canopy. Everything
in between -- picking the target building out of everyone's mess of mapped
polygons, turning the street to face ``-y``, inferring a lot with no parcel
data to read one from -- lives in this module because none of it is reusable
outside a fetch: :mod:`canopy.worldgen.evidence` is what the offline build
reads back out of the frozen snapshot this produces.
"""

from __future__ import annotations

import math
import os
import re
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient

from canopy.contracts import (
    Points2,
    ResidentialDecision,
    ResidentialVerdict,
    ResolvedAddress,
    SiteBuilding,
    SiteSnapshot,
    SourceRecord,
)
from canopy.errors import GeodataError, SiteRejectedError
from canopy.mathutil import wrap_to_pi
from canopy.worldgen.evidence import check_buildable, dominant_orientation, levels_for
from canopy.worldgen.geo import frames
from canopy.worldgen.geo.geoapify import GeoapifyGeocoder
from canopy.worldgen.geo.overpass import OsmFeatures, OsmWay, OverpassSource
from canopy.worldgen.geo.photon import PhotonGeocoder
from canopy.worldgen.geo.residential import ResidentialFeatures, mentions_suite, score_residential
from canopy.worldgen.snapshot import finalize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from canopy.config import Config, SiteCfg

__all__ = ["FeatureSource", "Geocoder", "default_geocoder", "fetch_site", "suggest_addresses"]

#: Building tags that stand but are never worth keeping as background scenery.
_EXCLUDED_NEIGHBOUR_TAGS = frozenset({"shed", "carport", "roof", "hut"})
#: Buildings that are never the house an address names, whatever the pin
#: lands on: preferring anything else keeps a pin dropped in a back yard
#: from snapping to the garage.
_OUTBUILDING_TAGS = frozenset(
    {"shed", "garage", "garages", "carport", "hut", "roof", "greenhouse", "outbuilding"}
)
#: Below this footprint a "neighbour" is an outbuilding, not another house.
_MIN_NEIGHBOUR_AREA_M2 = 25.0
#: Highway classes that count as "the street", in preference order; ``service``
#: is used only when nothing of a better class was mapped.
_STREET_CLASSES = (
    "residential",
    "living_street",
    "unclassified",
    "tertiary",
    "secondary",
    "primary",
)
_SERVICE_HIGHWAY = "service"
#: A bare number, or a number with a trailing "m": OSM's metric convention for
#: ``width`` and ``height``. Anything else (feet, inches, no unit at all
#: assumed otherwise) is not parsed.
_METRES_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*m?\s*$")
#: Metres per lane, when a street has no mapped ``width``.
_METRES_PER_LANE = 3.5
#: A point-of-interest tag on the house itself counts toward the residential score.
_POI_KEYS = ("shop", "office", "amenity")
#: A way with fewer distinct vertices is too degenerate to be a polygon or polyline.
_MIN_DISTINCT_VERTICES = 3
#: A polyline needs at least two points to have a direction at all.
_MIN_POLYLINE_VERTICES = 2
#: Below this, a candidate rotation's reference direction is too close to zero to score.
_ROTATION_EPS = 1e-9
#: Each geocoder's dataset, licence and standing attribution, for its SourceRecord.
_GEOCODER_SOURCES = {
    "photon": (
        "OpenStreetMap",
        "ODbL-1.0",
        "(c) OpenStreetMap contributors, geocoding by Photon (komoot)",
    ),
    "geoapify": (
        "Geoapify (OpenStreetMap, OpenAddresses and other open address data)",
        "Geoapify terms; ODbL-1.0 and each address source's own licence",
        "Powered by Geoapify; (c) OpenStreetMap contributors",
    ),
}


class FeatureSource(Protocol):
    """What :func:`fetch_site` needs from an OpenStreetMap feature provider."""

    def features(self, lat_deg: float, lon_deg: float, radius_m: float) -> OsmFeatures:
        """Return every mapped feature within ``radius_m`` of a point."""
        ...


class Geocoder(Protocol):
    """What :func:`suggest_addresses` needs from an address search provider."""

    @property
    def name(self) -> str:
        """The provider's name, recorded as :attr:`ResolvedAddress.provider`."""
        ...

    def suggest(self, text: str, *, limit: int) -> list[ResolvedAddress]:
        """Return addresses matching partly typed ``text``, best first."""
        ...


def suggest_addresses(
    text: str, cfg: Config, *, geocoder: Geocoder | None = None
) -> list[ResolvedAddress]:
    """Return addresses matching partly typed ``text``, best first.

    The provider is ``geocoder`` if given, else the one
    ``cfg.worldgen.site.geocoder`` selects (see :func:`default_geocoder`).

    Raises
    ------
    GeodataError
        If the geocoder cannot be reached or answers with something unreadable,
        or the configured one needs an API key that is not set.
    """
    site_cfg = cfg.worldgen.site
    provider = geocoder if geocoder is not None else default_geocoder(site_cfg)
    return provider.suggest(text, limit=site_cfg.suggest_limit)


def _windows_user_env(name: str) -> str:
    """Read ``name`` from the Windows user environment as saved, not as inherited.

    A process's environment is a copy taken when its parent started, and
    terminals (VS Code's especially, which revives old sessions) outlive the
    moment a key is saved, so a viewer launched from one never sees it and
    address search silently degrades to Photon. The registry holds the value
    as it is now. Empty off Windows, or when the value is not set.
    """
    if sys.platform != "win32":
        return ""
    import winreg  # noqa: PLC0415 -- exists only on Windows; a top-level import breaks elsewhere

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env_key:
            value, _ = winreg.QueryValueEx(env_key, name)
    except OSError:
        return ""
    return str(value).strip()


def default_geocoder(cfg: SiteCfg, environ: Mapping[str, str] | None = None) -> Geocoder:
    """Build the address search provider ``cfg.geocoder`` selects.

    ``"auto"`` is Geoapify when its key is in the environment -- near-complete
    address coverage -- and Photon otherwise, so the field works with no
    account at all and gets better the moment a key is set.

    Parameters
    ----------
    cfg
        ``worldgen.site`` settings.
    environ
        Where to look up ``cfg.geoapify_key_env``. Defaults to
        :data:`os.environ`, then the saved Windows user environment (see
        :func:`_windows_user_env`); a test passes a plain dict, which is read alone.

    Raises
    ------
    GeodataError
        If ``cfg.geocoder`` is ``"geoapify"`` and the key is not set.
    """
    if environ is None:
        key = os.environ.get(cfg.geoapify_key_env, "").strip() or _windows_user_env(
            cfg.geoapify_key_env
        )
    else:
        key = environ.get(cfg.geoapify_key_env, "").strip()
    if cfg.geocoder == "photon" or (cfg.geocoder == "auto" and not key):
        return PhotonGeocoder(cfg)
    if not key:
        msg = (
            "worldgen.site.geocoder is 'geoapify' but no API key is set; "
            f"put one in ${cfg.geoapify_key_env}, or set geocoder to 'auto' to fall back to Photon"
        )
        raise GeodataError(msg)
    return GeoapifyGeocoder(cfg, key)


def fetch_site(
    address: ResolvedAddress,
    cfg: Config,
    *,
    query: str = "",
    source: FeatureSource | None = None,
    now: Callable[[], str] | None = None,
    on_retry: Callable[[int, int, float], None] | None = None,
) -> SiteSnapshot:
    """Fetch, check and freeze the open map data around ``address``.

    Every geometry field of the returned snapshot is in the world frame:
    metres, Z-up, origin at the inferred lot centre, turned so the house's
    street side faces ``-y``. See ``docs/adr/0015-address-seeded-sites.md``.

    Parameters
    ----------
    address
        A pin a geocoder resolved, e.g. from :func:`suggest_addresses`.
    cfg
        Full configuration; ``cfg.worldgen.site`` is read.
    query
        The raw text the user searched, for the suite/unit check
        (:func:`~canopy.worldgen.geo.residential.mentions_suite`) that the
        resolved label alone may not carry.
    source
        Feature provider. Defaults to a real
        :class:`~canopy.worldgen.geo.overpass.OverpassSource`.
    now
        Returns an ISO-8601 UTC timestamp. Defaults to the real clock.
    on_retry
        Told ``(next_attempt, attempts, wait_s)`` before each retry of a busy
        or timed-out Overpass request, for a progress line. Only used by
        the default ``source``.

    Raises
    ------
    GeodataError
        If the data cannot be fetched, or no building stands near the address.
    SiteRejectedError
        If the address is not a home, or not one the generator can rebuild.
    """
    site_cfg = cfg.worldgen.site
    feature_source = source if source is not None else OverpassSource(site_cfg, on_retry=on_retry)
    clock = now if now is not None else _utc_now

    osm = feature_source.features(address.lat_deg, address.lon_deg, site_cfg.aoi_radius_m)
    anchor_lat, anchor_lon = address.lat_deg, address.lon_deg
    pin = np.zeros(2, dtype=np.float64)

    building_ways = [
        (way, outline)
        for way in osm.ways
        if "building" in way.tags
        for outline in (_way_outline(way, anchor_lat, anchor_lon),)
        if outline is not None
    ]
    target_way, target_outline = _target_building(building_ways, address, site_cfg)
    house_centroid = _centroid(target_outline)

    neighbour_entries = _neighbour_buildings(building_ways, target_way, house_centroid, site_cfg)

    highways = [
        (way, line)
        for way in osm.ways
        if "highway" in way.tags
        for line in (_way_polyline(way, anchor_lat, anchor_lon),)
        if line is not None
    ]
    street_way, street_line, street_point, street_note = _find_street(
        highways, address, house_centroid
    )

    # -- frame: turn the house so its street faces -y -----------------------
    theta = dominant_orientation(target_outline)
    if street_point is not None:
        front_ref = street_point
    else:
        to_pin = pin - house_centroid
        front_ref = (
            house_centroid + to_pin
            if float(np.linalg.norm(to_pin)) > 1.0
            else house_centroid + np.array([0.0, -1.0])
        )
    phi = _choose_rotation(theta, house_centroid, front_ref)
    north_rad = wrap_to_pi(phi + math.pi / 2.0)
    rot = _rotation(phi)

    def to_temp(points: Points2) -> Points2:
        """Rotate ENU points by ``phi`` about the house centroid (step e's scratch frame)."""
        return (np.atleast_2d(points) - house_centroid) @ rot.T

    house_temp = to_temp(target_outline)
    neighbour_temp = [to_temp(outline) for _way, outline in neighbour_entries]
    street_point_temp = to_temp(street_point)[0] if street_point is not None else None

    left, right, front, rear = _infer_lot(house_temp, neighbour_temp, street_point_temp, site_cfg)
    lot_centre_temp = np.array([(left + right) / 2.0, (front + rear) / 2.0], dtype=np.float64)
    lot_m = (right - left, rear - front)
    origin_enu_m = house_centroid + rot.T @ lot_centre_temp

    def to_world(points: Points2) -> Points2:
        """Finish the transform: shift the temporary frame so the lot centre is the origin."""
        return to_temp(points) - lot_centre_temp

    notes: list[str] = [
        "the lot boundary is inferred from mapped neighbours and the street; "
        "no parcel data was used"
    ]
    if street_note is not None:
        notes.append(street_note)

    house_building = _site_building(target_way, to_world(target_outline))
    check_buildable(house_building, site_cfg)  # a gate: SiteRejectedError propagates
    verdict = _residential_verdict(
        osm,
        target_way,
        target_outline,
        house_building,
        address,
        query,
        anchor_lat,
        anchor_lon,
        site_cfg,
    )

    neighbour_buildings = tuple(
        _site_building(way, to_world(outline)) for way, outline in neighbour_entries
    )

    street_world = to_world(street_line) if street_line is not None else None
    street_name = street_way.tags.get("name", "") if street_way is not None else ""
    street_width_m = (
        _street_width(street_way, site_cfg)
        if street_way is not None
        else site_cfg.default_street_width_m
    )

    trees_world = _tree_positions(osm, anchor_lat, anchor_lon, to_world)
    poles_world = _pole_positions(osm, anchor_lat, anchor_lon, to_world)
    aoi = _aoi_bounds(house_centroid, site_cfg.aoi_radius_m, to_world)
    sources = _source_records(osm, address, clock())

    snapshot = SiteSnapshot(
        site_id="",
        address=address,
        verdict=verdict,
        anchor_lat_deg=anchor_lat,
        anchor_lon_deg=anchor_lon,
        origin_enu_m=origin_enu_m,
        north_rad=north_rad,
        lot_m=lot_m,
        house=house_building,
        neighbours=neighbour_buildings,
        trees=trees_world,
        poles=poles_world,
        street=street_world,
        street_name=street_name,
        street_width_m=street_width_m,
        aoi=aoi,
        sources=sources,
        notes=tuple(notes),
    )
    return finalize(snapshot)


# ---------------------------------------------------------------------------
# Geometry helpers shared by the steps above
# ---------------------------------------------------------------------------
def _rotation(angle_rad: float) -> npt.NDArray[np.float64]:
    """2D rotation matrix, anticlockwise by ``angle_rad``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _centroid(polygon: Points2) -> npt.NDArray[np.float64]:
    """Area centroid of a simple polygon (not the vertex mean)."""
    centroid = Polygon(polygon).buffer(0.0).centroid
    return np.array([centroid.x, centroid.y], dtype=np.float64)


def _bounds(points: Points2) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of a point set, as ``(x0, y0, x1, y1)``."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _ccw(points: Points2) -> Points2:
    """Return a clean outline: counter-clockwise, first point not repeated (contracts.py's).

    A self-intersecting way -- a bowtie, a real if rare mapping error -- is
    repaired first, keeping its largest piece, so no consumer downstream
    has to remember to.

    Raises
    ------
    GeodataError
        If the outline encloses no area at all.
    """
    shape = Polygon(points).buffer(0.0)
    if shape.is_empty or shape.area <= 0.0:
        msg = f"a mapped building outline of {len(points)} vertices encloses no area"
        raise GeodataError(msg)
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda part: part.area)
    ring = orient(Polygon(shape.exterior), sign=1.0)
    return np.array(ring.exterior.coords[:-1], dtype=np.float64)


def _way_outline(way: OsmWay, anchor_lat_deg: float, anchor_lon_deg: float) -> Points2 | None:
    """Project one way's vertices to ENU, drop a repeated closing vertex.

    Returns
    -------
    Points2 | None
        ``None`` for a way with fewer than 3 distinct vertices: too
        degenerate to be a polygon or a usable polyline.
    """
    pts = frames.enu_from_geodetic(way.lat, way.lon, anchor_lat_deg, anchor_lon_deg)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(np.unique(np.round(pts, 9), axis=0)) < _MIN_DISTINCT_VERTICES:
        return None
    return pts


def _way_polyline(way: OsmWay, anchor_lat_deg: float, anchor_lon_deg: float) -> Points2 | None:
    """Project one way's vertices to ENU as an open polyline, such as a street.

    Unlike :func:`_way_outline`, two vertices are enough: OpenStreetMap splits
    streets into short ways at every junction, and a straight segment between
    two of them is often all the street a house fronts.

    Returns
    -------
    Points2 | None
        ``None`` for a way with fewer than two distinct vertices.
    """
    pts = frames.enu_from_geodetic(way.lat, way.lon, anchor_lat_deg, anchor_lon_deg)
    if len(np.unique(np.round(pts, 9), axis=0)) < _MIN_POLYLINE_VERTICES:
        return None
    return pts


def _parse_ref(ref: str) -> tuple[str, int] | None:
    """Split an ``"osm:way/<id>"`` or ``"osm:relation/<id>"`` reference; ``None`` otherwise."""
    match = re.fullmatch(r"osm:(way|relation)/(\d+)", ref)
    return (match.group(1), int(match.group(2))) if match else None


def _parse_metres(text: str) -> float | None:
    """Parse a bare or ``"N m"`` OSM measurement; ``None`` for feet or another unit."""
    match = _METRES_RE.match(text)
    return float(match.group(1)) if match else None


def _crown_radius(diameter_crown: str | None) -> float:
    """Tree crown radius in metres from a mapped ``diameter_crown``; ``0.0`` if unmapped."""
    if diameter_crown is None:
        return 0.0
    parsed = _parse_metres(diameter_crown)
    return parsed / 2.0 if parsed is not None else 0.0


def _levels_tag(tags: dict[str, str]) -> int | None:
    """``building:levels``, rounded; ``None`` if absent or unparseable."""
    raw = tags.get("building:levels")
    if raw is None:
        return None
    try:
        return round(float(raw))
    except ValueError:
        return None


def _height_tag(tags: dict[str, str]) -> float | None:
    """``height`` in metres, as mapped; ``None`` if absent or given in another unit."""
    raw = tags.get("height")
    return _parse_metres(raw) if raw is not None else None


def _roof_shape_tag(tags: dict[str, str]) -> str | None:
    """``roof:shape``, lower-cased; ``None`` if absent."""
    raw = tags.get("roof:shape")
    return raw.lower() if raw is not None else None


def _site_building(way: OsmWay, outline_world: Points2) -> SiteBuilding:
    """One mapped way, with its tags read, as a world-frame :class:`SiteBuilding`."""
    return SiteBuilding(
        footprint=_ccw(outline_world),
        levels=_levels_tag(way.tags),
        height_m=_height_tag(way.tags),
        roof_shape=_roof_shape_tag(way.tags),
        source=f"osm:{way.kind}/{way.id}",
    )


def _utc_now() -> str:
    """Real clock: an ISO-8601 UTC timestamp, e.g. ``"2026-09-26T12:00:00Z"``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tree_positions(
    osm: OsmFeatures,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    to_world: Callable[[Points2], Points2],
) -> npt.NDArray[np.float64]:
    """Return mapped trees in the world frame, as ``(x, y, crown_radius_m)`` rows."""
    nodes = [n for n in osm.nodes if n.tags.get("natural") == "tree"]
    enu = frames.enu_from_geodetic(
        np.array([n.lat for n in nodes], dtype=np.float64),
        np.array([n.lon for n in nodes], dtype=np.float64),
        anchor_lat_deg,
        anchor_lon_deg,
    )
    radii = np.array([_crown_radius(n.tags.get("diameter_crown")) for n in nodes], dtype=np.float64)
    return np.column_stack([to_world(enu), radii])


def _pole_positions(
    osm: OsmFeatures,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    to_world: Callable[[Points2], Points2],
) -> Points2:
    """Return mapped utility poles and towers, in the world frame."""
    nodes = [n for n in osm.nodes if n.tags.get("power") in ("pole", "tower")]
    enu = frames.enu_from_geodetic(
        np.array([n.lat for n in nodes], dtype=np.float64),
        np.array([n.lon for n in nodes], dtype=np.float64),
        anchor_lat_deg,
        anchor_lon_deg,
    )
    return to_world(enu)


def _aoi_bounds(
    house_centroid: npt.NDArray[np.float64],
    radius_m: float,
    to_world: Callable[[Points2], Points2],
) -> npt.NDArray[np.float64]:
    """World-frame bounds of the square area the data was fetched for, about the house."""
    corners_enu = house_centroid + np.array(
        [
            [-radius_m, -radius_m],
            [radius_m, -radius_m],
            [radius_m, radius_m],
            [-radius_m, radius_m],
        ],
        dtype=np.float64,
    )
    corners_world = to_world(corners_enu)
    return np.array([corners_world.min(axis=0), corners_world.max(axis=0)], dtype=np.float64)


def _source_records(
    osm: OsmFeatures, address: ResolvedAddress, retrieved_at: str
) -> tuple[SourceRecord, SourceRecord]:
    """Provenance: one record for Overpass/OpenStreetMap, one for the geocoder.

    The geocoder's record takes the match's own attribution when the provider
    gave one (Geoapify names the record's source per result, and its terms
    ask that credit to be kept with anything stored), else the provider's
    standing credit from :data:`_GEOCODER_SOURCES`.
    """
    dataset = "OpenStreetMap" if osm.timestamp is None else f"OpenStreetMap ({osm.timestamp})"
    geo_dataset, geo_licence, geo_attribution = _GEOCODER_SOURCES.get(
        address.provider, (address.provider, "see provider terms", address.provider)
    )
    return (
        SourceRecord(
            provider="overpass",
            dataset=dataset,
            licence="ODbL-1.0",
            attribution="(c) OpenStreetMap contributors",
            retrieved_at=retrieved_at,
        ),
        SourceRecord(
            provider=address.provider,
            dataset=geo_dataset,
            licence=geo_licence,
            attribution=address.attribution or geo_attribution,
            retrieved_at=retrieved_at,
        ),
    )


# ---------------------------------------------------------------------------
# b. Target building
# ---------------------------------------------------------------------------
def _target_building(
    candidates: Sequence[tuple[OsmWay, Points2]], address: ResolvedAddress, cfg: SiteCfg
) -> tuple[OsmWay, Points2]:
    """Pick the mapped building the address names: by id, else containment, else nearest.

    Raises
    ------
    GeodataError
        If nothing candidate stands within ``cfg.snap_radius_m`` of the pin.
    """
    ref = _parse_ref(address.ref)
    if ref is not None:
        for way, outline in candidates:
            if (way.kind, way.id) == ref:
                return way, outline

    # Past an exact id match, a shed or a garage is never the house the
    # address names, unless nothing else was mapped at all.
    houses = [c for c in candidates if c[0].tags.get("building") not in _OUTBUILDING_TAGS]
    candidates = houses or list(candidates)
    pin = Point(0.0, 0.0)
    containing = [
        (way, outline) for way, outline in candidates if Polygon(outline).buffer(0.0).contains(pin)
    ]
    if containing:
        return containing[0]

    if candidates:
        nearest = min(candidates, key=lambda item: Polygon(item[1]).buffer(0.0).distance(pin))
        if Polygon(nearest[1]).buffer(0.0).distance(pin) <= cfg.snap_radius_m:
            return nearest

    msg = f"no building was found within {cfg.snap_radius_m:.0f} m of {address.label!r}"
    raise GeodataError(msg)


def _poi_evidence(
    osm: OsmFeatures,
    target: OsmWay,
    target_outline: Points2,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
) -> tuple[int, bool]:
    """Points of interest inside the house footprint, and whether the house itself is one."""
    has_poi_tags = any(key in target.tags for key in _POI_KEYS)
    poi_nodes = [n for n in osm.nodes if any(key in n.tags for key in _POI_KEYS)]
    if not poi_nodes:
        return 0, has_poi_tags
    enu = frames.enu_from_geodetic(
        np.array([n.lat for n in poi_nodes], dtype=np.float64),
        np.array([n.lon for n in poi_nodes], dtype=np.float64),
        anchor_lat_deg,
        anchor_lon_deg,
    )
    house = Polygon(target_outline).buffer(0.0)
    inside = shapely.contains_xy(house, enu[:, 0], enu[:, 1])
    return int(np.count_nonzero(inside)), has_poi_tags


# ---------------------------------------------------------------------------
# h. Residential
# ---------------------------------------------------------------------------
def _residential_verdict(
    osm: OsmFeatures,
    target: OsmWay,
    target_outline: Points2,
    house_building: SiteBuilding,
    address: ResolvedAddress,
    query: str,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    cfg: SiteCfg,
) -> ResidentialVerdict:
    """Score the address and raise if the residential check rejects it.

    Raises
    ------
    SiteRejectedError
        If the score is at or below ``cfg.residential.reject_p``.
    """
    pois_inside, has_poi_tags = _poi_evidence(
        osm, target, target_outline, anchor_lat_deg, anchor_lon_deg
    )
    features = ResidentialFeatures(
        building_tag=target.tags.get("building"),
        building_has_poi_tags=has_poi_tags,
        pois_inside=pois_inside,
        landuse_here=osm.landuse,
        area_m2=float(Polygon(target_outline).buffer(0.0).area),
        levels=levels_for(house_building, cfg),
        suite=mentions_suite(query) or mentions_suite(address.label),
    )
    verdict = score_residential(features, cfg.residential)
    if verdict.decision is ResidentialDecision.REJECT:
        top = "; ".join(verdict.reasons[:3])
        msg = (
            f"{address.label!r} scores p={verdict.p_residential:.2f} as a home, at or below "
            f"worldgen.site.residential.reject_p = {cfg.residential.reject_p}: {top}"
        )
        raise SiteRejectedError(msg)
    return verdict


# ---------------------------------------------------------------------------
# c. Neighbours
# ---------------------------------------------------------------------------
def _neighbour_buildings(
    candidates: Sequence[tuple[OsmWay, Points2]],
    target: OsmWay,
    house_centroid: npt.NDArray[np.float64],
    cfg: SiteCfg,
) -> list[tuple[OsmWay, Points2]]:
    """Other mapped buildings kept as background scenery, nearest first."""
    scored: list[tuple[float, OsmWay, Points2]] = []
    for way, outline in candidates:
        if way is target or way.tags.get("building") in _EXCLUDED_NEIGHBOUR_TAGS:
            continue
        polygon = Polygon(outline).buffer(0.0)
        if polygon.area < _MIN_NEIGHBOUR_AREA_M2:
            continue
        centroid = np.array([polygon.centroid.x, polygon.centroid.y], dtype=np.float64)
        scored.append((float(np.linalg.norm(centroid - house_centroid)), way, outline))
    scored.sort(key=lambda item: item[0])
    return [(way, outline) for _dist, way, outline in scored[: cfg.max_neighbours]]


# ---------------------------------------------------------------------------
# d. Street
# ---------------------------------------------------------------------------
def _street_name_from_label(label: str) -> str:
    """Text after the leading house number in a label's first comma-separated part."""
    first = label.split(",", 1)[0].strip()
    match = re.match(r"^\S+\s+(.*)$", first)
    return match.group(1).strip() if match else first


def _find_street(
    highways: Sequence[tuple[OsmWay, Points2]],
    address: ResolvedAddress,
    house_centroid: npt.NDArray[np.float64],
) -> tuple[OsmWay | None, Points2 | None, npt.NDArray[np.float64] | None, str | None]:
    """Choose the street the address fronts, and the closest point on it to the house.

    Returns
    -------
    tuple
        ``(way, polyline, nearest_point, note)``. All ``None`` (with a note)
        when nothing was mapped nearby.
    """
    candidates = [
        (way, line) for way, line in highways if way.tags.get("highway") in _STREET_CLASSES
    ]
    if not candidates:
        candidates = [
            (way, line) for way, line in highways if way.tags.get("highway") == _SERVICE_HIGHWAY
        ]
    if not candidates:
        note = "no street was mapped near the address; the front was inferred from the pin"
        return None, None, None, note

    named_street = _street_name_from_label(address.label).casefold()
    named = (
        [
            (way, line)
            for way, line in candidates
            if way.tags.get("name", "").casefold() == named_street
        ]
        if named_street
        else []
    )
    pool = named or candidates

    def house_point(xy: npt.NDArray[np.float64]) -> Point:
        """``xy`` as a shapely point."""
        return Point(float(xy[0]), float(xy[1]))

    best_way, best_line = min(
        pool, key=lambda item: LineString(item[1]).distance(house_point(house_centroid))
    )
    projected = LineString(best_line).interpolate(
        LineString(best_line).project(house_point(house_centroid))
    )
    nearest_point = np.array([projected.x, projected.y], dtype=np.float64)
    return best_way, best_line, nearest_point, None


def _street_width(way: OsmWay, cfg: SiteCfg) -> float:
    """Carriageway width: mapped ``width``, else lanes, else the configured default."""
    width_tag = way.tags.get("width")
    if width_tag is not None:
        parsed = _parse_metres(width_tag)
        if parsed is not None:
            return parsed
    lanes_tag = way.tags.get("lanes")
    if lanes_tag is not None:
        try:
            return float(lanes_tag) * _METRES_PER_LANE
        except ValueError:
            pass
    return cfg.default_street_width_m


# ---------------------------------------------------------------------------
# e. Frame
# ---------------------------------------------------------------------------
def _choose_rotation(
    theta: float, house_centroid: npt.NDArray[np.float64], front_ref: npt.NDArray[np.float64]
) -> float:
    """Pick the quarter turn of ``-theta`` that puts the street closest to due south.

    ``theta`` (from :func:`~canopy.worldgen.evidence.dominant_orientation`) is
    only known modulo a quarter turn, so all four square-up rotations are
    candidates; the one that best faces the street toward ``-y`` wins.
    """
    to_front = front_ref - house_centroid
    norm = float(np.linalg.norm(to_front))
    if norm < _ROTATION_EPS:
        return -theta  # no direction to face: any square-up turn will do
    best_phi = -theta
    best_score = -math.inf
    for k in range(4):
        phi = -theta + k * math.pi / 2.0
        rotated = _rotation(phi) @ to_front
        score = float(np.dot(rotated / norm, np.array([0.0, -1.0])))
        if score > best_score:
            best_score = score
            best_phi = phi
    return best_phi


# ---------------------------------------------------------------------------
# f. Lot inference
# ---------------------------------------------------------------------------
def _shrink_proportional(
    lo: float,
    hi: float,
    house_lo: float,
    house_hi: float,
    max_span: float,
    *,
    lo_reserve: float = 0.0,
) -> tuple[float, float]:
    """Pull ``lo``/``hi`` in toward the house, proportionally to each side's slack.

    Never crosses ``house_lo``/``house_hi`` (or ``lo_reserve`` beyond it):
    the caller has already checked the house itself fits within ``max_span``,
    so the two sides' slack always covers the required reduction.
    """
    span = hi - lo
    if span <= max_span:
        return lo, hi
    lo_slack = max(0.0, (house_lo - lo_reserve) - lo)
    hi_slack = max(0.0, hi - house_hi)
    total_slack = lo_slack + hi_slack
    if total_slack <= 0.0:
        return lo, hi
    excess = span - max_span
    return lo + excess * (lo_slack / total_slack), hi - excess * (hi_slack / total_slack)


def _shrink_for_neighbour(
    left: float,
    right: float,
    rear: float,
    house: tuple[float, float, float, float],
    neighbour: tuple[float, float, float, float],
    gap: float,
) -> tuple[float, float, float]:
    """Pull one yard boundary toward one neighbour, or put it on the house edge.

    A neighbour beside or behind the house shrinks that yard to the midpoint
    of the gap. One that also runs alongside a wall -- overlapping the house
    along it -- and stands within ``gap`` of it shares that wall outright, so
    that side gets no yard at all. A neighbour only diagonally off a corner
    shares no wall and pulls in no boundary: whatever part of it pokes into
    the lot is kept clear by the generator instead (``SiteEvidence.lot_obstacles``).
    """
    hx0, hy0, hx1, hy1 = house
    nx0, ny0, nx1, ny1 = neighbour
    y_overlaps = ny0 < hy1 and ny1 > hy0
    x_overlaps = nx0 < hx1 and nx1 > hx0
    if nx0 >= hx1 and y_overlaps:
        right = min(right, (hx1 + nx0) / 2.0)
    if nx1 <= hx0 and y_overlaps:
        left = max(left, (hx0 + nx1) / 2.0)
    if ny0 >= hy1 and x_overlaps:
        rear = min(rear, (hy1 + ny0) / 2.0)

    if y_overlaps and nx1 > hx1 and nx0 <= hx1 + gap:
        right = min(right, hx1)
    if y_overlaps and nx0 < hx0 and nx1 >= hx0 - gap:
        left = max(left, hx0)
    if x_overlaps and ny1 > hy1 and ny0 <= hy1 + gap:
        rear = min(rear, hy1)
    return left, right, rear


def _infer_lot(
    house_temp: Points2,
    neighbour_temp: Sequence[Points2],
    street_point_temp: npt.NDArray[np.float64] | None,
    cfg: SiteCfg,
) -> tuple[float, float, float, float]:
    """Infer lot boundaries in the rotated, house-centred temporary frame.

    Returns
    -------
    tuple[float, float, float, float]
        ``(left, right, front, rear)``, the lot's ``x`` and ``y`` bounds in
        the same frame ``house_temp`` is given in.

    Raises
    ------
    SiteRejectedError
        If the house cannot fit inside ``cfg.max_lot_m`` even with only the
        minimum front yard.
    """
    hx0, hy0, hx1, hy1 = _bounds(house_temp)
    neighbour_bounds = [_bounds(n) for n in neighbour_temp]

    if street_point_temp is not None:
        front = float(street_point_temp[1]) + cfg.street_centre_offset_m
    else:
        front = hy0 - cfg.default_front_yard_m
    front = min(front, hy0 - cfg.min_front_yard_m)

    left = hx0 - cfg.default_side_yard_m
    right = hx1 + cfg.default_side_yard_m
    rear = hy1 + cfg.default_back_yard_m
    house_bounds = (hx0, hy0, hx1, hy1)
    for neighbour in neighbour_bounds:
        left, right, rear = _shrink_for_neighbour(
            left, right, rear, house_bounds, neighbour, cfg.party_wall_gap_m
        )

    min_w, min_d = cfg.min_lot_m
    house_cx, house_cy = (hx0 + hx1) / 2.0, (hy0 + hy1) / 2.0
    if (right - left) < min_w:
        left, right = house_cx - min_w / 2.0, house_cx + min_w / 2.0
    if (rear - front) < min_d:
        front, rear = house_cy - min_d / 2.0, house_cy + min_d / 2.0

    max_w, max_h = cfg.max_lot_m
    house_w = hx1 - hx0
    house_d_with_front = (hy1 - hy0) + cfg.min_front_yard_m
    if house_w > max_w or house_d_with_front > max_h:
        msg = (
            f"the house is {house_w:.0f} x {(hy1 - hy0):.0f} m, too large to fit inside "
            f"worldgen.site.max_lot_m = {list(cfg.max_lot_m)} even with the minimum front yard"
        )
        raise SiteRejectedError(msg)
    left, right = _shrink_proportional(left, right, hx0, hx1, max_w)
    front, rear = _shrink_proportional(
        front, rear, hy0, hy1, max_h, lo_reserve=cfg.min_front_yard_m
    )
    return left, right, front, rear
