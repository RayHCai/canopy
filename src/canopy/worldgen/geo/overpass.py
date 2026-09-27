"""Overpass: every mapped feature in one area of interest, in one query.

One call fetches everything :mod:`canopy.worldgen.geo.build` needs to rebuild
a property -- buildings, streets, trees, poles, points of interest and the
landuse at the pin -- because a second round trip per feature kind would be a
second chance for the public server to rate-limit a fetch that already has to
succeed in one shot (a snapshot is fetched once and built from forever after).

The query stays on the server's cheap path: plain bounding-box filters and
nothing else. Overpass's ``is_in`` (which area contains this point?) looked
like the natural way to ask for the landuse at the pin, but it runs against a
separate area database that the public instance keeps busy: measured on
2026-09-26, the query with it failed half the time (504 after about 12 s)
where the same query without it answered every time in 2-4 s -- and when it
did answer, it found no landuse area at all. So landuse polygons come back in
the same bounding-box query, and the point-in-polygon test is done here.

The returned types (:class:`OsmWay`, :class:`OsmNode`, :class:`OsmFeatures`)
stay in this module rather than in ``contracts.py``: they are Overpass's own
document shape, not something another stage ever reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point, Polygon

from canopy.errors import GeodataError
from canopy.worldgen.geo import frames
from canopy.worldgen.geo.http import Transport, UrllibTransport

if TYPE_CHECKING:
    from collections.abc import Callable

    from canopy.config import SiteCfg

__all__ = ["OsmFeatures", "OsmNode", "OsmWay", "OverpassSource"]

#: Tag filters fetched around the address, in query order.
_ELEMENT_CLAUSES = (
    'way["building"](bbox)',
    # A house mapped as a multipolygon (a courtyard, or just a mapper's habit)
    # has no building way of its own; without this it would be invisible, and
    # the fetch would snap to the nearest shed instead.
    'relation["building"]["type"="multipolygon"](bbox)',
    'way["highway"](bbox)',
    'node["natural"="tree"](bbox)',
    'node["power"~"^(pole|tower)$"](bbox)',
    'node["shop"](bbox)',
    'node["office"](bbox)',
    'node["amenity"](bbox)',
    # Landuse polygons crossing the area, tested against the pin locally. One
    # that wholly contains the area (a neighbourhood-sized polygon whose edge
    # is further than aoi_radius_m away) is not returned; the residential
    # score then just lacks that one piece of evidence. Multipolygon landuse
    # is not asked for: a town-sized relation's full geometry can be larger
    # than everything else here together.
    'way["landuse"](bbox)',
)

#: Tags that make a way part of the scene. A way with ``landuse`` but none of
#: these is only evidence about the pin, and never reaches ``OsmFeatures.ways``.
_SCENE_KEYS = ("building", "highway")

#: How Overpass reports a query it stopped early (out of time or memory):
#: HTTP 200, whatever it had gathered so far, and this in the ``remark``.
_RUNTIME_ERROR = "runtime error"

#: Fewest points in a closed ring, the first repeated as the last.
_MIN_RING_POINTS = 4

#: Decimal places in a query's embedded coordinates: sub-millimetre, so
#: rounding never moves a query's answer.
_QUERY_PRECISION = 7


@dataclass(frozen=True, slots=True, eq=False)
class OsmWay:
    """One Overpass ``way`` with its geometry resolved (``out geom``), or a building relation.

    A multipolygon building relation arrives here too, as its largest outer
    ring: everything downstream wants one outline per building, and a
    house's inner rings (a courtyard) are dropped by the house frame anyway.
    """

    id: int
    tags: dict[str, str]
    lat: npt.NDArray[np.float64]
    """Vertex latitudes, shape ``(N,)``, in the way's node order."""
    lon: npt.NDArray[np.float64]
    """Vertex longitudes, likewise."""
    kind: Literal["way", "relation"] = "way"
    """Which OpenStreetMap element this is; ids are only unique within a kind."""


@dataclass(frozen=True, slots=True)
class OsmNode:
    """One Overpass ``node``."""

    id: int
    tags: dict[str, str]
    lat: float
    lon: float


@dataclass(frozen=True, slots=True, eq=False)
class OsmFeatures:
    """Everything :meth:`OverpassSource.features` fetched in one area of interest."""

    ways: tuple[OsmWay, ...]
    nodes: tuple[OsmNode, ...]
    landuse: tuple[str, ...]
    """``landuse`` values of any mapped landuse polygon containing the query
    point (closed ways only; see ``_ELEMENT_CLAUSES``)."""
    timestamp: str | None
    """The data's ``osm3s.timestamp_osm_base``; ``None`` if the answer omitted it."""


def _bbox(lat_deg: float, lon_deg: float, radius_m: float) -> tuple[float, float, float, float]:
    """South, west, north, east bounds of a ``radius_m``-half-width square about a point."""
    corners_enu = np.array([[-radius_m, -radius_m], [radius_m, radius_m]], dtype=np.float64)
    lat, lon = frames.geodetic_from_enu(corners_enu, lat_deg, lon_deg)
    south, north = float(lat[0]), float(lat[1])
    west, east = float(lon[0]), float(lon[1])
    return south, west, north, east


def _query(bbox: tuple[float, float, float, float], timeout_s: float) -> str:
    """Build the one Overpass QL query :meth:`OverpassSource.features` sends."""
    south, west, north, east = bbox
    bbox_text = ",".join(f"{v:.{_QUERY_PRECISION}f}" for v in (south, west, north, east))
    elements = "".join(
        clause.replace("(bbox)", f"({bbox_text})") + ";" for clause in _ELEMENT_CLAUSES
    )
    return f"[out:json][timeout:{int(timeout_s)}];({elements});out geom;"


class OverpassSource:
    """Fetch OpenStreetMap features around a point, via one Overpass query."""

    def __init__(
        self,
        cfg: SiteCfg,
        transport: Transport | None = None,
        *,
        on_retry: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """Configure the source.

        Parameters
        ----------
        cfg
            ``worldgen.site`` settings: the endpoint, request identity and
            retry policy.
        transport
            HTTP client. Defaults to a real
            :class:`~canopy.worldgen.geo.http.UrllibTransport` using
            ``cfg.overpass_timeout_s``, ``cfg.overpass_retries`` and
            ``cfg.overpass_backoff_s``.
        on_retry
            Passed to the default transport: told of each retry before its
            wait. Ignored when ``transport`` is given.
        """
        self.cfg = cfg
        self.transport: Transport = (
            transport
            if transport is not None
            else UrllibTransport(
                cfg.user_agent,
                cfg.overpass_timeout_s,
                cfg.overpass_retries,
                cfg.overpass_backoff_s,
                on_retry=on_retry,
            )
        )

    def features(self, lat_deg: float, lon_deg: float, radius_m: float) -> OsmFeatures:
        """Fetch every mapped building, street, tree, pole, POI and the landuse near a point.

        Parameters
        ----------
        lat_deg, lon_deg
            Centre of the area of interest, e.g. the address pin.
        radius_m
            Half-width of the (square) area fetched.

        Raises
        ------
        GeodataError
            If Overpass cannot be reached or answers with an unreadable document.
        """
        query = _query(_bbox(lat_deg, lon_deg, radius_m), self.cfg.overpass_query_timeout_s)
        doc = self.transport.post_form(self.cfg.overpass_url, {"data": query})
        if not isinstance(doc, dict) or not isinstance(doc.get("elements"), list):
            msg = (
                "Overpass returned a document with no usable 'elements' list "
                f"(got {type(doc).__name__})"
            )
            raise GeodataError(msg)
        remark = str(doc.get("remark") or "")
        if _RUNTIME_ERROR in remark.casefold():
            # Partial data would build a street with houses missing from it,
            # silently; better to fail and let the user fetch again.
            msg = f"Overpass stopped the query before it finished: {remark.strip()}"
            raise GeodataError(msg)

        ways: list[OsmWay] = []
        nodes: list[OsmNode] = []
        landuse: list[str] = []
        for element in doc["elements"]:
            try:
                _add_element(element, ways, nodes)
            except (KeyError, TypeError, ValueError):
                continue  # one malformed element does not spoil the rest
        landuse.extend(
            w.tags["landuse"]
            for w in ways
            if "landuse" in w.tags and not _is_scene_way(w) and _ring_contains(w, lat_deg, lon_deg)
        )
        ways = [w for w in ways if _is_scene_way(w)]

        timestamp = None
        osm3s = doc.get("osm3s")
        if isinstance(osm3s, dict) and osm3s.get("timestamp_osm_base") is not None:
            timestamp = str(osm3s["timestamp_osm_base"])
        return OsmFeatures(
            ways=tuple(ways), nodes=tuple(nodes), landuse=tuple(landuse), timestamp=timestamp
        )


def _add_element(element: Any, ways: list[OsmWay], nodes: list[OsmNode]) -> None:
    """Parse one Overpass element into ``ways`` or ``nodes``, in place."""
    kind = element["type"]
    if kind == "way":
        geometry = element["geometry"]
        ways.append(
            OsmWay(
                id=int(element["id"]),
                tags=dict(element.get("tags", {})),
                lat=np.array([pt["lat"] for pt in geometry], dtype=np.float64),
                lon=np.array([pt["lon"] for pt in geometry], dtype=np.float64),
            )
        )
    elif kind == "relation":
        ring = _outer_ring(element.get("members", []))
        if ring is not None:
            ways.append(
                OsmWay(
                    id=int(element["id"]),
                    tags=dict(element.get("tags", {})),
                    lat=np.array([pt[0] for pt in ring], dtype=np.float64),
                    lon=np.array([pt[1] for pt in ring], dtype=np.float64),
                    kind="relation",
                )
            )
    elif kind == "node":
        nodes.append(
            OsmNode(
                id=int(element["id"]),
                tags=dict(element.get("tags", {})),
                lat=float(element["lat"]),
                lon=float(element["lon"]),
            )
        )


def _outer_ring(members: Any) -> list[tuple[float, float]] | None:
    """Join a multipolygon's outer member ways into rings; return the largest.

    Outer rings are often drawn as several ways that meet end to end, so the
    pieces are chained by shared end points -- the same node, so the same
    coordinates exactly -- flipping a piece when it runs the other way. A
    chain that never closes is dropped rather than guessed at.

    Returns
    -------
    list[tuple[float, float]] | None
        ``(lat, lon)`` points, closed (first repeated as last), or ``None``
        when no outer ring closes.
    """
    pieces = [
        [(float(pt["lat"]), float(pt["lon"])) for pt in member["geometry"]]
        for member in members
        if member.get("type") == "way"
        and member.get("role", "outer") in ("outer", "")
        and member.get("geometry")
    ]
    rings: list[list[tuple[float, float]]] = []
    while pieces:
        ring = pieces.pop(0)
        while ring[0] != ring[-1]:
            for i, piece in enumerate(pieces):
                if piece[0] == ring[-1]:
                    ring = ring + piece[1:]
                elif piece[-1] == ring[-1]:
                    ring = ring + piece[-2::-1]
                elif piece[-1] == ring[0]:
                    ring = piece[:-1] + ring
                elif piece[0] == ring[0]:
                    ring = piece[:0:-1] + ring
                else:
                    continue
                del pieces[i]
                break
            else:
                break  # nothing continues this chain: it cannot close
        if ring[0] == ring[-1] and len(ring) >= _MIN_RING_POINTS:
            rings.append(ring)
    if not rings:
        return None
    return max(rings, key=_ring_area)


def _is_scene_way(way: OsmWay) -> bool:
    """Whether a way is a building or a street, rather than landuse evidence only."""
    return any(key in way.tags for key in _SCENE_KEYS)


def _ring_contains(way: OsmWay, lat_deg: float, lon_deg: float) -> bool:
    """Whether a closed way's ring contains a point, tested in raw degrees.

    Degrees are fine here: containment does not depend on the map scale, and
    a landuse polygon is small enough that its edges are straight in either.
    """
    closed = (
        way.lat.size >= _MIN_RING_POINTS and way.lat[0] == way.lat[-1] and way.lon[0] == way.lon[-1]
    )
    if not closed:
        return False
    ring = Polygon(np.column_stack([way.lon, way.lat]))
    return bool(ring.is_valid and ring.contains(Point(lon_deg, lat_deg)))


def _ring_area(ring: list[tuple[float, float]]) -> float:
    """Shoelace area in square degrees: only ever compared between rings a few metres apart."""
    lat = np.array([pt[0] for pt in ring], dtype=np.float64)
    lon = np.array([pt[1] for pt in ring], dtype=np.float64)
    return abs(float(np.dot(lon, np.roll(lat, -1)) - np.dot(lat, np.roll(lon, -1)))) / 2.0
