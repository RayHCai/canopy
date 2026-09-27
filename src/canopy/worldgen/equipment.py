"""Wall-mounted equipment and the openings fitted around it.

The electrical service is the mission -- :func:`_service_assembly` reproduces
what the twelve authored homes encode as one connected group on a single
wall, meter first -- and :func:`_wall_mount` and :func:`_wall_adjacent` place
the other things that stand against or near a wall: the gas meter, the AC
condenser, the foundation planting that may occlude the electric meter on
purpose.

:func:`_facade_openings` runs last of all of it, fitting windows, a front
door and perhaps a garage door around whatever equipment already claimed a
stretch of wall -- a window behind the meter would be both wrong and an
occluder the survey never asked for. It shares
:func:`~canopy.worldgen.house.wall_height_at` with :mod:`canopy.worldgen.house`,
since a wall's height at a given point depends on which of the house's massed
blocks it belongs to.

See :mod:`canopy.worldgen.placement` for the shared kernel this module builds
on, and :mod:`canopy.worldgen.house` and :mod:`canopy.worldgen.background`
for the other two rule families.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.errors import WorldgenError
from canopy.worldgen.house import wall_height_at
from canopy.worldgen.placement import (
    EPS_M,
    MAX_ATTEMPTS,
    SAME_WALL_DOT,
    HouseFrame,
    Placement,
    PlacementContext,
    WallSegment,
    as3,
    choose_wall,
    clear,
    clear_of_openings,
    inside_lot,
    is_free,
    register_rule,
    seen,
    walls_for,
    yaw_onto,
)

if TYPE_CHECKING:
    from canopy.worldgen.evidence import SiteEvidence


#: Kept clear either side of an opening by anything fixed to or set against a
#: wall, when the role gives no ``opening_clearance_m`` of its own.
_OPENING_CLEARANCE_M = 0.3


@register_rule("wall_mount")
def _wall_mount(ctx: PlacementContext) -> list[Placement]:
    """Fix a unit flat to an exterior wall at a given height -- the meters.

    The outward normal is recorded on the placement because the whole mission
    turns on it: the inspection viewpoints are offsets along this normal, and
    the site solver needs it to know which way the conduit leaves the wall.

    Models for this rule are authored with their origin on the wall face, so the
    placement point is the wall contact point and only ``standoff_m`` separates
    them -- there is no half-depth to add.

    A unit goes only on bare wall, ``opening_clearance_m`` clear of any window
    or door the house model carries. One that finds no such spot is left out:
    the gas meter is a distractor, not the mission.
    """
    bottom = float(ctx.param("bottom_height_m", default=0.0))
    clearance = float(ctx.param("corner_clearance_m"))
    standoff = float(ctx.param("standoff_m"))
    gap = float(ctx.param("opening_clearance_m", default=_OPENING_CLEARANCE_M))
    walls = walls_for(ctx, avoid_meter_wall=bool(ctx.param("avoid_meter_wall", default=False)))

    placements: list[Placement] = []
    for _ in range(ctx.n):
        for _attempt in range(MAX_ATTEMPTS):
            wall, t = choose_wall(ctx, clearance, walls)
            spec = ctx.pick()
            xy = wall.at(t) + wall.normal * standoff
            candidate = Placement(
                spec=spec,
                pos=as3(xy, bottom),
                size=ctx.extents(spec),
                yaw=yaw_onto(spec.facing_xy(), wall.normal),
                wall_normal=as3(wall.normal),
            )
            if clear_of_openings(wall, [candidate], gap):
                placements.append(candidate)
                break
    return placements


@register_rule("wall_adjacent")
def _wall_adjacent(ctx: PlacementContext) -> list[Placement]:
    """Stand a unit on the ground, set back from an exterior wall -- the AC condenser.

    The unit may stand under a window it does not reach up to, as real
    condensers do, but never in front of a door or a garage door.
    """
    standoff = float(ctx.param("standoff_m"))
    clearance = float(ctx.param("corner_clearance_m"))
    gap = float(ctx.param("opening_clearance_m", default=_OPENING_CLEARANCE_M))

    placements: list[Placement] = []
    for _ in range(ctx.n):
        for _attempt in range(MAX_ATTEMPTS):
            wall, t = choose_wall(ctx, clearance)
            spec = ctx.pick()
            size = ctx.extents(spec)
            xy = wall.at(t) + wall.normal * (standoff + float(np.max(size[:2])) / 2.0)
            candidate = Placement(
                spec=spec,
                pos=as3(xy),
                size=size,
                yaw=yaw_onto(spec.facing_xy(), wall.normal),
                wall_normal=as3(wall.normal),
            )
            if (
                clear(xy, candidate.radius_m, seen(ctx, placements))
                and clear_of_openings(wall, [candidate], gap, below_m=float(size[2]) + gap)
                and (ctx.site is None or inside_lot(ctx, xy, candidate.radius_m))
            ):
                placements.append(candidate)
                break
    return placements


@register_rule("foundation_band")
def _foundation_band(ctx: PlacementContext) -> list[Placement]:
    """Plant along the foundation -- the bushes.

    With probability ``worldgen.p_meter_occluded`` the first bush is forced
    directly in front of the meter. That is the point of the bushes: an
    unoccluded meter makes the inspection viewpoint search trivial and leaves
    the bush-removal list in the SSR packet empty.
    """
    standoff_lo, standoff_hi = (float(v) for v in ctx.param("standoff_m"))
    occl_lo, occl_hi = (float(v) for v in ctx.param("occluder_standoff_m"))
    ctx.require_house()

    placements: list[Placement] = []
    meter = ctx.meter
    if (
        ctx.n > 0
        and meter is not None
        and meter.wall_normal is not None
        and ctx.rng.random() < ctx.cfg.worldgen.p_meter_occluded
    ):
        spec = ctx.pick()
        size = ctx.extents(spec)
        distance = float(ctx.rng.uniform(occl_lo, occl_hi))
        xy = meter.pos[:2] + meter.wall_normal[:2] * distance
        placements.append(
            Placement(spec=spec, pos=as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
        )

    while len(placements) < ctx.n:
        placed = False
        for _attempt in range(MAX_ATTEMPTS):
            wall, t = choose_wall(ctx, clearance_m=0.0)
            spec = ctx.pick()
            size = ctx.extents(spec)
            standoff = float(ctx.rng.uniform(standoff_lo, standoff_hi))
            xy = wall.at(t) + wall.normal * (standoff + float(np.max(size[:2])) / 2.0)
            candidate = Placement(spec=spec, pos=as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
            if clear(xy, candidate.radius_m, seen(ctx, placements)) and (
                ctx.site is None or inside_lot(ctx, xy, candidate.radius_m)
            ):
                placements.append(candidate)
                placed = True
                break
        if not placed:
            # The foundation band is full. Thinning the hedge is the right
            # failure: the count is a density target, not a requirement.
            break
    return placements


#: How far out from a wall a placement's disc may reach and still count as in
#: the way of a window: wall-mounted equipment sits a couple of centimetres
#: proud, while an AC unit or a bush stands well clear and may sit under one.
_FLUSH_REACH_M = 0.1


#: How far in front of the front door must be clear of anything placed.
_DOOR_APPROACH_M = 1.5


#: Minimum wall above a window's head, up to the eaves or the floor above.
_WINDOW_HEAD_M = 0.3


#: A wall this much short of a whole number of storeys still counts as having
#: them: a 5.6 m two-storey block is two 2.8 m storeys, not one.
_STOREY_SLACK_M = 0.3


def _blocked_spans(
    ctx: PlacementContext,
    wall: WallSegment,
    reach_m: float,
    gap_m: float,
    *,
    equipment_only: bool,
) -> list[tuple[float, float]]:
    """Stretches of ``wall``, in metres from :attr:`WallSegment.a`, that something occupies.

    Anything placed outside the house whose disc comes within ``reach_m`` of the
    wall's face blocks its own width along the wall, plus ``gap_m`` either side.
    The house's own blocks and roofs are excluded by position: their centres lie
    inside the footprint. ``equipment_only`` narrows it to placements mounted
    on or against a wall -- the meter, its conduit, the AC unit -- leaving out
    planting, which may stand in front of a window.
    """
    house = ctx.require_house()
    unit = (wall.b - wall.a) / max(wall.length, EPS_M)
    spans = []
    for item in ctx.placed:
        xy = item.pos[:2]
        out = float(np.dot(xy - wall.a, wall.normal))
        if out <= 0.0 or out - item.radius_m > reach_m or house.contains(xy):
            continue
        if equipment_only and item.wall_normal is None:
            continue
        along = float(np.dot(xy - wall.a, unit))
        half = item.radius_m + gap_m
        spans.append((along - half, along + half))
    return spans


@register_rule("facade_openings")
def _facade_openings(ctx: PlacementContext) -> list[Placement]:
    """Put windows, a front door and perhaps a garage door on a procedural house.

    The authored shell already has openings, so this places nothing there; it
    is for procedural massing, which is otherwise bare boxes. Each opening is a
    thin panel centred on the wall plane, standing a couple of centimetres proud
    so it neither z-fights the wall nor hides wall area from the coverage
    metric (the voxel it shares with the wall behind it is seen either way).

    It runs after everything else on the lot on purpose. The openings fit
    around the meter, panel and conduit already on the walls -- a window behind
    the meter would be both wrong and an occluder the survey never asked for --
    and running last means adding them leaves every other object's random draw,
    and so every existing seed's layout, exactly as it was.

    The front door goes on the tallest front-facing (``-y``) wall, which is the
    main block's, with nothing standing in its approach. A garage door, with
    probability ``p_garage``, goes on a single-storey front wall. Windows are
    then spaced evenly along every wall, one per storey in each column, skipping
    any column that would overlap an opening or wall-mounted equipment.
    """
    house = ctx.require_house()
    if ctx.n == 0 or house.has_openings:
        return []

    storey = float(ctx.param("storey_m"))
    corner = float(ctx.param("corner_clearance_m"))
    gap = float(ctx.param("equipment_clearance_m"))
    depth = float(ctx.param("depth_m"))
    sill = float(ctx.param("sill_m"))
    window_h = float(ctx.param("window_height_m"))
    width_lo, width_hi = (float(v) for v in ctx.param("window_width_m"))
    pitch_lo, pitch_hi = (float(v) for v in ctx.param("window_spacing_m"))

    walls = house.walls
    flush = [_blocked_spans(ctx, w, _FLUSH_REACH_M, gap, equipment_only=True) for w in walls]
    # A door would rather not open onto a bush, but the foundation planting is
    # dense enough that some front walls have no gap; then only equipment, which
    # would physically block the door, rules a spot out.
    approach = [
        (
            _blocked_spans(ctx, w, _DOOR_APPROACH_M, gap, equipment_only=False),
            _blocked_spans(ctx, w, _DOOR_APPROACH_M, gap, equipment_only=True),
        )
        for w in walls
    ]
    placements: list[Placement] = []

    def panel(
        tag: str, index: int, along_m: float, bottom_m: float, w_m: float, h_m: float
    ) -> None:
        wall = walls[index]
        edge = wall.b - wall.a
        placements.append(
            Placement(
                spec=ctx.library.choose(tag, ctx.rng),
                pos=as3(wall.at(along_m / wall.length), bottom_m),
                size=np.array([w_m, depth, h_m], dtype=np.float64),
                yaw=math.atan2(float(edge[1]), float(edge[0])),
                wall_normal=as3(wall.normal),
            )
        )

    def door(tag: str, candidates: Sequence[int], w_m: float, h_m: float) -> None:
        """Place one door on the first candidate wall with a clear stretch for it."""
        for index in candidates:
            length = walls[index].length
            if length - 2.0 * corner < w_m:
                continue
            # The approach spans reach further out than the flush ones, so they
            # cover wall-mounted equipment too.
            for spans in approach[index]:
                for _attempt in range(MAX_ATTEMPTS):
                    along = float(ctx.rng.uniform(corner + w_m / 2.0, length - corner - w_m / 2.0))
                    lo, hi = along - w_m / 2.0, along + w_m / 2.0
                    if is_free(lo, hi, spans):
                        panel(tag, index, along, 0.0, w_m, h_m)
                        taken = (lo - gap, hi + gap)
                        flush[index].append(taken)
                        for others in approach[index]:
                            others.append(taken)
                        return

    def height(index: int) -> float:
        return wall_height_at(house, walls[index].at(0.5))

    front = [i for i, w in enumerate(walls) if float(w.normal[1]) < -SAME_WALL_DOT]
    door_w, door_h = (float(v) for v in ctx.param("door_size_m"))
    door(
        str(ctx.param("door_tag")),
        sorted(front, key=lambda i: (height(i), walls[i].length), reverse=True),
        door_w,
        door_h,
    )
    if ctx.rng.random() < float(ctx.param("p_garage")):
        garage_w, garage_h = (float(v) for v in ctx.param("garage_size_m"))
        single = [i for i in front if height(i) < 1.5 * storey]
        order = [single[int(k)] for k in ctx.rng.permutation(len(single))]
        door(str(ctx.param("garage_tag")), order, garage_w, garage_h)

    window_tag = str(ctx.param("window_tag"))
    for index, wall in enumerate(walls):
        # One width and pitch per wall, so each elevation reads as designed
        # rather than as a scatter of mismatched panes.
        width = float(ctx.rng.uniform(width_lo, width_hi))
        pitch = float(ctx.rng.uniform(pitch_lo, pitch_hi))
        grid = _window_grid(
            house, wall, flush[index], (width, window_h), pitch, corner, storey, sill
        )
        for along, bottom in grid:
            panel(window_tag, index, along, bottom, width, window_h)
    return placements


def _window_grid(
    house: HouseFrame,
    wall: WallSegment,
    blocked: Sequence[tuple[float, float]],
    size_m: tuple[float, float],
    pitch_m: float,
    corner_m: float,
    storey_m: float,
    sill_m: float,
) -> list[tuple[float, float]]:
    """Lay out one wall's windows: ``(along, bottom)`` in metres for each.

    Columns are centred on the wall at ``pitch_m`` and kept ``corner_m`` from
    both ends; a column overlapping any ``blocked`` span is dropped whole, so a
    service drop running up the wall never has a window stacked over it. Each
    column gets one window per storey the wall is tall enough for at that point,
    with ``_WINDOW_HEAD_M`` of wall left above each.
    """
    width, height = size_m
    usable = wall.length - 2.0 * corner_m - width
    if usable < 0.0:
        return []
    columns = int(usable // pitch_m) + 1
    first = wall.length / 2.0 - (columns - 1) * pitch_m / 2.0

    grid = []
    for k in range(columns):
        along = first + k * pitch_m
        if not is_free(along - width / 2.0, along + width / 2.0, blocked):
            continue
        wall_h = wall_height_at(house, wall.at(along / wall.length))
        for level in range(max(1, int((wall_h + _STOREY_SLACK_M) // storey_m))):
            bottom = level * storey_m + sill_m
            if bottom + height > min(wall_h, (level + 1) * storey_m) - _WINDOW_HEAD_M:
                break
            grid.append((along, bottom))
    return grid


#: A wall whose outward normal is within about 45 degrees of the street
#: direction (-y) is the front; of +y, the back; anything else is a side.
_FACING_DOT = 0.7


def _meter_wall_weights(ctx: PlacementContext, site: SiteEvidence) -> npt.NDArray[np.float64]:
    """Odds, per house wall, that a real house's meter is on it.

    Two pieces of evidence: which side of the house the wall is on, read from
    the role's ``site_side_weights``, and how close it is to a mapped utility
    pole, since an overhead service drop usually lands on the nearest wall.
    Both are placeholder priors until calibrated against surveyed homes
    (docs/adr/0015-address-seeded-sites.md).

    A wall facing a lot line nearer than the meter's working space
    (``worldgen.invariants.meter_working_space_m``) gets no odds at all: no
    real meter is fitted where nobody can stand to read it. That is the prior
    preventing what the invariant would otherwise have to repair -- and a
    repair redraws at random, so it can miss the one good wall by bad luck.
    If no wall has the room, the odds are left alone so that the invariant,
    not a misleading wall-length error, reports why.
    """
    walls = ctx.require_house().walls
    sides = ctx.param("site_side_weights")
    normals = np.array([w.normal for w in walls], dtype=np.float64).reshape(-1, 2)
    toward_street = normals @ np.array([0.0, -1.0])
    weights = np.where(
        toward_street > _FACING_DOT,
        float(sides["front"]),
        np.where(toward_street < -_FACING_DOT, float(sides["back"]), float(sides["side"])),
    )
    poles = np.asarray(site.snapshot.poles, dtype=np.float64).reshape(-1, 2)
    if len(poles) > 0:
        mids = np.array([(w.a + w.b) / 2.0 for w in walls], dtype=np.float64).reshape(-1, 2)
        nearest = np.min(np.linalg.norm(mids[:, None, :] - poles[None, :, :], axis=2), axis=1)
        pull = float(ctx.param("site_pole_weight"))
        falloff = float(ctx.param("site_pole_falloff_m"))
        weights = weights * (1.0 + pull * np.exp(-nearest / falloff))
    roomy = _frontage_depth(ctx, walls) >= ctx.cfg.worldgen.invariants.meter_working_space_m
    if np.any(roomy):
        weights = np.where(roomy, weights, 0.0)
    return np.asarray(weights, dtype=np.float64)


def _frontage_depth(ctx: PlacementContext, walls: Sequence[WallSegment]) -> npt.NDArray[np.float64]:
    """Distance from each wall's midpoint, straight out along its normal, to the lot line.

    The ray-box exit distance. Other blocks of the house are ignored: a wall
    facing back into its own house would be a courtyard, which the house
    frame never has.
    """
    lo, hi = ctx.lot_bounds[0, :2], ctx.lot_bounds[1, :2]
    depths = []
    for wall in walls:
        mid = (wall.a + wall.b) / 2.0
        exits = [
            ((hi[k] if wall.normal[k] > 0.0 else lo[k]) - mid[k]) / wall.normal[k]
            for k in range(2)
            if abs(float(wall.normal[k])) > EPS_M
        ]
        depths.append(min(exits) if exits else math.inf)
    return np.asarray(depths, dtype=np.float64)


@register_rule("service_assembly")
def _service_assembly(ctx: PlacementContext) -> list[Placement]:
    """Place the electrical service as one connected group on a single wall.

    This reproduces what the twelve authored homes encode, because they are the
    only statement of what a physically sensible arrangement looks like. In all
    twelve the meter's underside sits at z = 1.25 on one wall, and one of two
    topologies follows:

    ``panel``
        Ten of twelve. A breaker panel 0.6 to 1.05 m along the same wall with
        its underside at z = 0.75, and a short conduit nipple bridging the gap
        between them at about z = 1.5.
    ``riser``
        The other two. No exterior panel; an LB riser about 0.85 m along the
        wall carries the service through the wall instead.

    Either way an optional vertical run drops from the meter toward the ground.
    The pieces are emitted as separate objects so the detector, the coverage
    metric and the site solver see a meter, a panel and conduit rather than one
    undifferentiated lump -- and because the conduit route is what the SSR
    packet is ultimately about.

    The whole group goes on bare wall, ``opening_clearance_m`` clear of any
    window or door the house model carries, so the assembly is redrawn until
    it fits between them. Procedural massing has no openings yet at this
    point, so there the first draw always stands.

    Raises
    ------
    WorldgenError
        If no draw finds a stretch of bare wall long enough. Every property
        needs its meter, so this is not a count to thin out.
    """
    clearance = float(ctx.param("corner_clearance_m"))
    gap = float(ctx.param("opening_clearance_m", default=_OPENING_CLEARANCE_M))
    weights = _meter_wall_weights(ctx, ctx.site) if ctx.site is not None else None

    for _attempt in range(MAX_ATTEMPTS):
        wall, t = choose_wall(ctx, clearance, weights=weights)
        placements = _service_on(ctx, wall, t, clearance)
        if clear_of_openings(wall, placements, gap):
            return placements

    house = ctx.require_house()
    msg = (
        f"role {ctx.role.name!r}: found no bare wall for the service assembly in "
        f"{MAX_ATTEMPTS} draws; the house's {len(house.walls)} walls carry "
        f"{sum(len(w.openings) for w in house.walls)} openings, and each draw needs "
        f"{gap} m clear of them and {clearance} m clear of the corners"
    )
    raise WorldgenError(msg)


def _service_on(
    ctx: PlacementContext, wall: WallSegment, t: float, clearance_m: float
) -> list[Placement]:
    """Draw one service assembly with its meter a fraction ``t`` along ``wall``."""
    standoff = float(ctx.param("standoff_m"))
    meter_bottom = float(ctx.param("meter_bottom_m"))

    along = wall.b - wall.a
    unit = along / max(float(np.linalg.norm(along)), EPS_M)
    anchor = wall.at(t)

    def mounted(tag: str, at: npt.NDArray[np.float64], bottom_m: float) -> Placement:
        spec = ctx.library.choose(tag, ctx.rng)
        return Placement(
            spec=spec,
            pos=as3(at + wall.normal * standoff, bottom_m),
            size=ctx.extents(spec),
            yaw=yaw_onto(spec.facing_xy(), wall.normal),
            wall_normal=as3(wall.normal),
        )

    placements = [mounted(str(ctx.param("meter_tag")), anchor, meter_bottom)]

    # Which way along the wall the rest of the assembly runs. Pick the side with
    # more room so a long assembly never overhangs a corner.
    forward = float(np.linalg.norm(wall.b - anchor))
    backward = float(np.linalg.norm(anchor - wall.a))
    direction = unit if forward >= backward else -unit
    room = max(forward, backward) - clearance_m

    panel_lo, panel_hi = (float(v) for v in ctx.param("panel_offset_m"))
    offset = float(ctx.rng.uniform(panel_lo, panel_hi))
    wants_panel = ctx.rng.random() < float(ctx.param("p_panel"))

    if wants_panel and room >= offset:
        panel_at = anchor + direction * offset
        placements.append(
            mounted(str(ctx.param("panel_tag")), panel_at, float(ctx.param("panel_bottom_m")))
        )
        # The nipple bridges the gap, so it belongs at the midpoint.
        placements.append(
            mounted(
                str(ctx.param("nipple_tag")),
                anchor + direction * (offset / 2.0),
                float(ctx.param("nipple_height_m")),
            )
        )
    else:
        riser_offset = min(float(ctx.param("riser_offset_m")), max(room, 0.0))
        placements.append(
            mounted(
                str(ctx.param("riser_tag")),
                anchor + direction * riser_offset,
                float(ctx.param("riser_bottom_m")),
            )
        )

    if ctx.rng.random() < float(ctx.param("p_service_drop")):
        # A vertical run from grade up to the meter. Offset a little along the
        # wall so it does not sit inside the meter body.
        placements.append(
            mounted(
                str(ctx.param("straight_tag")),
                anchor + direction * float(ctx.param("drop_offset_m")),
                0.0,
            )
        )
    return placements
