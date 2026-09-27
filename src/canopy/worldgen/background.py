"""Ground, neighbours and street: everything outside the surveyed lot's house.

The ground plane, the neighbouring lots and houses, the street's sidewalks
and lane markings, the yard scatter (mostly trees) and the fence are all
scenery rather than the swarm's job -- real geometry, in the raycasting
scene, but exempt from coverage and drawn in true colour from the start (see
:attr:`~canopy.contracts.SceneObject.background`). On an address-built
property most of these rules read directly from the map data instead of
drawing: :func:`_lawn_bands`, :func:`_mapped_neighbours`, :func:`_mapped_trees`.

A neighbour house is deliberately not registered with ``defines_house``: the
surveyed house frame is the one :mod:`canopy.worldgen.house`'s ``house_lot``
rule built, and background never enters it.

See :mod:`canopy.worldgen.placement` for the shared kernel this module builds
on, and :mod:`canopy.worldgen.house` and :mod:`canopy.worldgen.equipment` for
the other two rule families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon

from canopy.contracts import Provenance
from canopy.errors import WorldgenError
from canopy.worldgen.placement import (
    MAX_ATTEMPTS,
    HouseFrame,
    Placement,
    PlacementContext,
    as3,
    choose_among,
    choose_wall,
    clear,
    crosses,
    register_rule,
    rotated_plan,
    seen,
    yaw_onto,
)

if TYPE_CHECKING:
    from canopy.worldgen.evidence import SiteEvidence


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
@register_rule("lot_plane")
def _lot_plane(ctx: PlacementContext) -> list[Placement]:
    """Place the ground: one plane spanning the whole lot, centred on the origin.

    Its size comes from ``worldgen.lot_m`` rather than from the model, so the
    lot stays a config tunable and the ground model stays reusable.
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    return [
        Placement(
            spec=ctx.spec,
            pos=np.zeros(3, dtype=np.float64),
            size=np.array([lot_x, lot_y, 0.0], dtype=np.float64),
        )
    ]


#: Row of a neighbour lot that shares the surveyed lot's street frontage.
#: Positive rows are behind it; negative rows are across the street, which is
#: why the street's own width has to be skipped for them.
_SAME_STREET_ROW = 0


@dataclass(frozen=True, slots=True, eq=False)
class _NeighbourLot:
    """One neighbouring lot: its grid cell, its centre and its street frontage."""

    column: int
    row: int
    centre: npt.NDArray[np.float64]
    """Lot centre ``(x, y)`` in world metres."""
    fronts_minus_y: bool
    """Whether the lot's street frontage is its ``-y`` edge.

    Row 0 fronts the surveyed street on its ``-y`` side. Every other row
    fronts ``+y``: the row across the street faces back across it, and the
    back row faces a street of its own beyond the scenery, which puts it back
    to back with the surveyed lot the way real subdivisions are laid out."""


def _neighbour_lots(ctx: PlacementContext) -> list[_NeighbourLot]:
    """Read the role's ``lots`` grid into lot centres.

    ``lots`` lists ``[column, row]`` cells in units of the surveyed lot's size,
    with the surveyed lot at ``[0, 0]``; ``street_gap_m`` is how far the row
    across the street starts beyond the surveyed front lot line. Every
    background role that needs the grid shares one YAML anchor for it, so the
    lawns, houses and street cannot disagree about where the neighbourhood is.

    Raises
    ------
    WorldgenError
        If the grid names the surveyed lot itself, or names a cell twice.
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    gap = float(ctx.param("street_gap_m"))
    cells = [(int(c), int(r)) for c, r in ctx.param("lots")]
    if (0, _SAME_STREET_ROW) in cells or len(set(cells)) != len(cells):
        msg = (
            f"role {ctx.role.name!r}: lots must be distinct and must not include the "
            f"surveyed lot [0, 0], got {cells}"
        )
        raise WorldgenError(msg)

    return [
        _NeighbourLot(
            column=column,
            row=row,
            centre=np.array(
                [column * lot_x, row * lot_y - (gap if row < _SAME_STREET_ROW else 0.0)],
                dtype=np.float64,
            ),
            fronts_minus_y=row == _SAME_STREET_ROW,
        )
        for column, row in cells
    ]


@register_rule("neighbour_lot")
def _neighbour_lot(ctx: PlacementContext) -> list[Placement]:
    """Place a lawn plane on each neighbouring lot -- background scenery.

    One lawn per cell of the ``lots`` grid, the same size as the surveyed lot.
    On an address-built property the neighbours' lots are not known, so the
    lawn is laid as bands around the lot instead (:func:`_lawn_bands`).
    """
    if ctx.site is not None:
        return _lawn_bands(ctx, ctx.site)
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    return [
        Placement(
            spec=ctx.spec,
            pos=as3(lot.centre),
            size=np.array([lot_x, lot_y, 0.0], dtype=np.float64),
        )
        for lot in _neighbour_lots(ctx)
    ]


#: Storeys from which a mapped neighbour gets a two-storey model.
_TWO_STOREYS = 2


def _lawn_bands(ctx: PlacementContext, site: SiteEvidence) -> list[Placement]:
    """Lawn over the fetched area outside the surveyed lot and the street.

    Four bands: beside the lot on either side and behind it, each from the
    front lot line back to the area's edge, and one across the street beyond
    its far kerb. They meet the lot, the street strips and each other edge to
    edge, so no two ground planes overlap and z-fight.
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    gap = float(ctx.param("street_gap_m"))
    ax0, ay0 = (float(v) for v in site.snapshot.aoi[0])
    ax1, ay1 = (float(v) for v in site.snapshot.aoi[1])
    half_x, half_y = lot_x / 2.0, lot_y / 2.0
    front = -half_y
    bands = (
        (ax0, front, -half_x, ay1),
        (half_x, front, ax1, ay1),
        (-half_x, half_y, half_x, ay1),
        (ax0, ay0, ax1, front - gap),
    )
    return [
        Placement(
            spec=ctx.spec,
            pos=np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0], dtype=np.float64),
            size=np.array([x1 - x0, y1 - y0, 0.0], dtype=np.float64),
        )
        for x0, y0, x1, y1 in bands
        if x1 > x0 and y1 > y0
    ]


def _mapped_neighbours(ctx: PlacementContext, site: SiteEvidence) -> list[Placement]:
    """Stand one neighbour model on each mapped neighbouring footprint.

    The model is the one-storey or two-storey kind the mapped storey count
    asks for -- the neighbour models' heights include their roofs, so
    ``two_storey_min_height_m`` of total height separates the two -- or any
    kind when the count is unmapped. It is stretched over the footprint's
    minimum rotated rectangle and turned to that rectangle's real heading, so
    the gap between the houses is the real one. Background never enters the
    house frame, so it is exempt from the quarter-turn rule; the quarter turn
    that matters is the one that puts the model's ``-y`` front toward the
    street.
    """
    snapshot = site.snapshot
    tall_m = float(ctx.param("two_storey_min_height_m"))
    found = ctx.library.candidates(ctx.role.tag)
    street_y = (
        float(np.mean(snapshot.street[:, 1]))
        if snapshot.street is not None
        else -ctx.cfg.worldgen.lot_m[1] / 2.0 - ctx.cfg.worldgen.site.street_centre_offset_m
    )
    (ax0, ay0), (ax1, ay1) = snapshot.aoi

    placements: list[Placement] = []
    for building in snapshot.neighbours:
        outline = np.asarray(building.footprint, dtype=np.float64)
        centroid = outline.mean(axis=0)
        if not (ax0 <= centroid[0] <= ax1 and ay0 <= centroid[1] <= ay1):
            continue
        if building.levels is None:
            spec = ctx.pick()
        else:
            want_tall = building.levels >= _TWO_STOREYS
            pool = [
                s
                for s in found
                if (s.size_z is not None and sum(s.size_z) / 2.0 >= tall_m) == want_tall
            ]
            spec = choose_among(ctx, pool or found)

        # The rectangle's first edge gives the heading; of its four quarter
        # turns, keep the one whose -y front faces the street.
        rect = np.asarray(
            Polygon(outline).minimum_rotated_rectangle.exterior.coords, dtype=np.float64
        )
        edge = rect[1] - rect[0]
        heading = math.atan2(float(edge[1]), float(edge[0]))
        toward_street = -1.0 if centroid[1] > street_y else 1.0
        yaw = max(
            (heading + k * math.pi / 2.0 for k in range(4)),
            key=lambda a: toward_street * -math.cos(a),
        )
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        local = (outline - centroid) @ np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        lo, hi = local.min(axis=0), local.max(axis=0)
        mid = (lo + hi) / 2.0
        centre = centroid + np.array(
            [cos_y * mid[0] - sin_y * mid[1], sin_y * mid[0] + cos_y * mid[1]]
        )
        height = building.height_m if building.height_m is not None else float(ctx.extents(spec)[2])
        placements.append(
            Placement(
                spec=spec,
                pos=as3(centre),
                size=np.array([hi[0] - lo[0], hi[1] - lo[1], height], dtype=np.float64),
                yaw=yaw,
                provenance=Provenance.OBSERVED,
            )
        )
    return placements


@register_rule("neighbour_house")
def _neighbour_house(ctx: PlacementContext) -> list[Placement]:
    """Place one house on each neighbouring lot, facing its street.

    A row of houses reads as a street only if they all face it at a similar
    setback, so the yaw is not the model's own: the lot's row fixes it, turning
    the model's ``-y`` front towards the frontage. The setback is drawn from
    ``front_yard_m`` and the house slides freely along the frontage within
    ``lot_margin_m`` of the side lot lines. An edge shared with the surveyed
    lot gets ``min_gap_m`` instead where that is the larger clearance, so a
    neighbour's eaves never loom over the surveyed lot line. This rule is
    deliberately not registered with ``defines_house``: the surveyed house
    frame stays the one ``house_lot`` built.

    Raises
    ------
    WorldgenError
        If a sampled footprint does not fit its neighbour lot at all.
    """
    if ctx.site is not None:
        return _mapped_neighbours(ctx, ctx.site)
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    lot_margin = float(ctx.param("lot_margin_m"))
    min_gap = float(ctx.param("min_gap_m"))
    yard_lo, yard_hi = (float(v) for v in ctx.param("front_yard_m"))
    near_margin = max(lot_margin, min_gap)
    half_x, half_y = lot_x / 2.0, lot_y / 2.0

    placements: list[Placement] = []
    for lot in _neighbour_lots(ctx):
        cx, cy = float(lot.centre[0]), float(lot.centre[1])
        # Drawn per house so the role can mix models; one shape repeated on
        # every lot reads as a tiling artefact rather than a street.
        spec = ctx.pick()
        extents = ctx.extents(spec)
        yaw = 0.0 if lot.fronts_minus_y else math.pi
        half_plan = rotated_plan(extents, yaw)[:2] / 2.0
        depth = 2.0 * float(half_plan[1])

        # A side neighbour on the same street shares an x edge with the
        # surveyed lot; the lot directly behind it shares its rear edge.
        beside = lot.row == _SAME_STREET_ROW and abs(lot.column) == 1
        west = near_margin if beside and lot.column > 0 else lot_margin
        east = near_margin if beside and lot.column < 0 else lot_margin
        rear = near_margin if (lot.column, lot.row) == (0, 1) else lot_margin
        x_lo = cx - half_x + west + float(half_plan[0])
        x_hi = cx + half_x - east - float(half_plan[0])
        setback = float(ctx.rng.uniform(yard_lo, yard_hi))

        if x_lo > x_hi or setback + depth > lot_y - rear:
            msg = (
                f"model {spec.asset_id!r} sampled a {2 * half_plan[0]:.1f} x {depth:.1f} m "
                f"footprint, which does not fit a {lot_x} x {lot_y} m neighbour lot with "
                f"{lot_margin} m margins, a {setback:.1f} m front yard and a {min_gap} m "
                "gap from the surveyed lot"
            )
            raise WorldgenError(msg)

        frontage, inward = (cy - half_y, 1.0) if lot.fronts_minus_y else (cy + half_y, -1.0)
        centre = np.array(
            [ctx.rng.uniform(x_lo, x_hi), frontage + inward * (setback + depth / 2.0)],
            dtype=np.float64,
        )
        placements.append(Placement(spec=spec, pos=as3(centre), size=extents, yaw=yaw))
    return placements


@register_rule("frontage_strip")
def _frontage_strip(ctx: PlacementContext) -> list[Placement]:
    """Lay flat strips parallel to the front lot line -- sidewalks, street, road lines.

    Each entry of ``setback_m`` is one strip, its near edge that many metres in
    front of the surveyed lot and ``depth_m`` deep. The strips run the full
    width of the ``lots`` grid, so the street passes every house on it rather
    than stopping at the survey's lot lines. ``z_m`` lifts a strip off the
    ground: a road marking laid exactly on the asphalt would z-fight with it.
    Nothing here is sampled: a strip's place is fixed by the lot and the
    params.

    Raises
    ------
    WorldgenError
        If the role's count disagrees with the number of setbacks, which means
        the library was edited on one side only.
    """
    setbacks = [float(v) for v in ctx.param("setback_m")]
    depth = float(ctx.param("depth_m"))
    lift = float(ctx.param("z_m", default=0.0))
    if ctx.n != len(setbacks):
        msg = (
            f"role {ctx.role.name!r} places {ctx.n} strips but lists {len(setbacks)} "
            f"setbacks ({setbacks}); set count to {{fixed: {len(setbacks)}}}"
        )
        raise WorldgenError(msg)

    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    provenance = Provenance.INFERRED
    if ctx.site is not None:
        # The real street runs the width of the fetched area, not of a grid of
        # neighbour lots; it is observed when the data mapped one.
        lo_x, hi_x = float(ctx.site.snapshot.aoi[0, 0]), float(ctx.site.snapshot.aoi[1, 0])
        if ctx.site.snapshot.street is not None:
            provenance = Provenance.OBSERVED
    else:
        columns = [0, *(int(c) for c, _ in ctx.param("lots"))]
        lo_x = (min(columns) - 0.5) * lot_x
        hi_x = (max(columns) + 0.5) * lot_x
    front = -lot_y / 2.0
    return [
        Placement(
            spec=ctx.spec,
            pos=np.array(
                [(lo_x + hi_x) / 2.0, front - setback - depth / 2.0, lift], dtype=np.float64
            ),
            size=np.array([hi_x - lo_x, depth, 0.0], dtype=np.float64),
            provenance=provenance,
        )
        for setback in setbacks
    ]


@register_rule("yard_scatter")
def _yard_scatter(ctx: PlacementContext) -> list[Placement]:
    """Scatter free-standing objects out in the yard -- the trees.

    When ``first_near_house`` is set the first one is placed at exactly its
    minimum clearance from a wall, so at least one crown overhangs the roof
    edge. That single tree is what makes the orbit rings a real planning problem
    rather than a circle in open air.
    """
    clearance = float(ctx.param("house_clearance_m"))
    lot_margin = float(ctx.param("lot_margin_m"))
    first_near_house = bool(ctx.param("first_near_house", default=False))
    # Sheds belong behind the house, not on the street elevation the survey
    # photographs; trees may go anywhere.
    back_only = bool(ctx.param("back_yard_only", default=False))

    house = ctx.require_house()
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    if (
        ctx.site is not None
        and str(ctx.param("site_evidence", default="")) == "trees"
        and len(ctx.site.snapshot.trees) > 0
    ):
        return _mapped_trees(ctx, ctx.site, house, lot_margin)

    placements: list[Placement] = []
    for index in range(ctx.n):
        for _attempt in range(MAX_ATTEMPTS):
            spec = ctx.pick()
            size = ctx.extents(spec)
            radius = float(np.max(size[:2])) / 2.0
            # The limit is on the object's edge, not its centre: a tree crown
            # hanging past the lot line would sit outside `lot_bounds`, and so
            # outside the voxel grid the mapper sizes from them.
            x_lim = lot_x / 2.0 - lot_margin - radius
            y_lim = lot_y / 2.0 - lot_margin - radius
            if x_lim <= 0.0 or y_lim <= 0.0:
                continue
            if index == 0 and first_near_house:
                wall, t = choose_wall(ctx, clearance_m=0.0)
                xy = wall.at(t) + wall.normal * (clearance + radius)
                if back_only and xy[1] < house.centre[1]:
                    continue
            else:
                y_floor = float(house.centre[1]) if back_only else -y_lim
                if y_floor >= y_lim:
                    continue
                xy = np.array(
                    [ctx.rng.uniform(-x_lim, x_lim), ctx.rng.uniform(y_floor, y_lim)],
                    dtype=np.float64,
                )
            if abs(xy[0]) > x_lim or abs(xy[1]) > y_lim:
                continue
            if house.contains(xy, margin_m=clearance + radius):
                continue
            if not clear(xy, radius, seen(ctx, placements)):
                continue
            placements.append(
                Placement(spec=spec, pos=as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
            )
            break
    return placements


#: A mapped tree whose trunk stands this close to the house is a mapping
#: error or a tree the house was built around; either way it is left out.
_MAPPED_TREE_HOUSE_CLEARANCE_M = 0.5


#: How far a mapped crown radius may stretch or shrink a tree model. Beyond
#: this the art's proportions read wrong, so the tree keeps the nearer limit.
_TREE_SCALE_RANGE = (0.5, 2.0)


def _mapped_trees(
    ctx: PlacementContext, site: SiteEvidence, house: HouseFrame, lot_margin: float
) -> list[Placement]:
    """Place the trees the map shows on the lot, and only those.

    Trees are mapped unevenly: where anyone mapped one in the fetched area the
    mapping is taken as the truth and nothing is added, while an area with no
    trees mapped at all falls back to drawing them (the caller's job). A tree
    is scaled so its crown matches the mapped radius when there is one, shrunk
    if its crown would hang past the lot line (outside the mapper's grid), and
    left out if it stands in the house or the launch area -- or if its trunk is
    so near the lot line that no crown at least half its model's size fits.
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    lo_scale, hi_scale = _TREE_SCALE_RANGE
    placements: list[Placement] = []
    for x, y, radius_m in np.asarray(site.snapshot.trees, dtype=np.float64):
        xy = np.array([x, y], dtype=np.float64)
        spec = ctx.pick()
        size = ctx.extents(spec)
        native = float(np.max(size[:2])) / 2.0
        scale = float(np.clip(radius_m / native, lo_scale, hi_scale)) if radius_m > 0.0 else 1.0
        room = min(lot_x / 2.0 - lot_margin - abs(x), lot_y / 2.0 - lot_margin - abs(y))
        scale = min(scale, room / native)
        if scale < lo_scale or house.contains(xy, margin_m=_MAPPED_TREE_HOUSE_CLEARANCE_M):
            continue
        size = size * scale
        crown = native * scale
        if not clear(xy, crown, (*ctx.launch_area, *(p.obstacle() for p in placements))):
            continue
        placements.append(
            Placement(
                spec=spec,
                pos=as3(xy),
                size=size,
                yaw=spec.sample_yaw(ctx.rng),
                provenance=Provenance.OBSERVED,
            )
        )
    return placements


#: Least clearance between a fence line and the house it encloses.
_FENCE_HOUSE_CLEARANCE_M = 0.5


def _fence_inset_limit(
    ctx: PlacementContext, lot_x: float, lot_y: float, *, closed_front: bool
) -> float:
    """Largest fence inset that keeps every fenced side clear of the house."""
    outline = np.asarray(ctx.require_house().footprint, dtype=np.float64)
    (min_x, min_y), (max_x, max_y) = outline.min(axis=0), outline.max(axis=0)
    limits = [float(min_x) + lot_x / 2.0, lot_x / 2.0 - float(max_x), lot_y / 2.0 - float(max_y)]
    if closed_front:
        limits.append(float(min_y) + lot_y / 2.0)
    return min(limits) - _FENCE_HOUSE_CLEARANCE_M


@register_rule("fence_perimeter")
def _fence_perimeter(ctx: PlacementContext) -> list[Placement]:
    """Tile fence panels around the lot, usually leaving the street side open.

    ``n`` is whether the property is fenced at all, not a panel count: the rule
    works out how many panels a side needs. The inset is sampled per property,
    so the yard the fence encloses varies in size rather than always hugging the
    lot line.

    A closed front fence can run through the launch area, so any panel that
    would cross it is left out: a gate where the operator launches. Leaving
    panels out draws nothing from the generator, so fences that miss the
    launch area come out exactly as before.
    """
    if ctx.n <= 0:
        return []

    inset_lo, inset_hi = (float(v) for v in ctx.param("inset_m"))
    inset = float(ctx.rng.uniform(inset_lo, inset_hi))
    front_open = ctx.rng.random() < float(ctx.param("p_front_open"))

    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    if ctx.site is not None:
        # A real house can stand closer to its lot lines than any inset the
        # role draws; pull the fence in no further than keeps it clear.
        fits = _fence_inset_limit(ctx, lot_x, lot_y, closed_front=not front_open)
        if fits < inset_lo:
            return []
        inset = min(inset, fits)
    hx, hy = lot_x / 2.0 - inset, lot_y / 2.0 - inset
    if hx <= 0.0 or hy <= 0.0:
        return []

    corners = {
        "back": (np.array([-hx, hy]), np.array([hx, hy]), np.array([0.0, 1.0])),
        "right": (np.array([hx, hy]), np.array([hx, -hy]), np.array([1.0, 0.0])),
        "front": (np.array([hx, -hy]), np.array([-hx, -hy]), np.array([0.0, -1.0])),
        "left": (np.array([-hx, -hy]), np.array([-hx, hy]), np.array([-1.0, 0.0])),
    }
    if front_open:
        del corners["front"]

    placements: list[Placement] = []
    for start, end, normal in corners.values():
        span = end - start
        length = float(np.linalg.norm(span))
        unit = span / length
        spec = ctx.library.choose(str(ctx.param("panel_tag")), ctx.rng)
        panel_w = float(ctx.library.native_extents(spec)[0])
        n_panels = max(round(length / panel_w), 1)
        # Stretch the panels a hair rather than leaving a gap at the corner.
        width = length / n_panels
        for i in range(n_panels):
            centre = start + unit * (width * (i + 0.5))
            size = ctx.library.native_extents(spec).copy()
            size[0] = width
            if crosses(centre, unit, size, ctx.launch_area):
                continue
            placements.append(
                Placement(
                    spec=spec,
                    pos=as3(centre),
                    size=size,
                    yaw=yaw_onto(spec.facing_xy(), np.asarray(normal, dtype=np.float64)),
                )
            )
    return placements
