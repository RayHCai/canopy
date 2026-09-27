"""House massing: the ``house_lot`` rule and the two body styles it draws from.

A property has exactly one house, and which style it gets is per-seed rather
than per-property-kind: ``p_authored`` of properties get the one authored
brick shell (:func:`_authored_house`), which already carries its own roof,
windows, doors and garage; the rest get one to three plain blocks unioned
into an L or a T, each roofed separately (:func:`_massed_house`). The
procedural path is what gives genuinely varying dimensions, which a single
authored shell cannot.

On an address-built property neither draw runs: the house is the mapped
footprint instead (:func:`_observed_house`), fitted where it stands rather
than sited on the lot. :func:`wall_height_at` is the one thing this module
exposes beyond the registry, because :mod:`canopy.worldgen.equipment` needs
it too -- a straight exterior wall can run along a two-storey block and then
on along a flush single-storey wing, so a wall's height is a property of the
point being measured, not of the wall as a whole.

See :mod:`canopy.worldgen.placement` for the shared kernel (:class:`HouseFrame`,
:class:`Placement`, :class:`PlacementContext`, the rule registry) this module
builds on, and :mod:`canopy.worldgen.equipment` and
:mod:`canopy.worldgen.background` for the other two rule families.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import Provenance
from canopy.errors import WorldgenError
from canopy.worldgen.placement import (
    HouseFrame,
    Placement,
    PlacementContext,
    as3,
    choose_among,
    register_rule,
    rotated_plan,
)

if TYPE_CHECKING:
    from canopy.worldgen.assets import AssetSpec
    from canopy.worldgen.evidence import SiteEvidence

__all__ = ["wall_height_at"]


#: Default overlap between a wing and the main mass, so the union connects.
_DEFAULT_WING_OVERLAP_M = 0.3


#: p_authored default: with no value in the library, never use the shell.
_ALWAYS_PROCEDURAL = 0.0


@register_rule("house_lot", defines_house=True)
def _house_lot(ctx: PlacementContext) -> list[Placement]:
    """Place the house, either as an authored shell or as procedural massing.

    Two house styles share one rule because a property has exactly one house and
    the choice is per-seed. ``p_authored`` of properties get the authored brick
    shell, which brings its own roof, windows, doors and garage; the rest get
    one to three plain blocks unioned into an L or a T, each with its own roof.
    The procedural path is what gives genuinely varying dimensions, which a
    single authored shell cannot.

    The setback is not cosmetic. Drones launch from the front of the lot and the
    orbit rings need room in front of the house for the approach; a house
    centred on the lot leaves the front elevation unreachable at low altitude.

    On an address-built property the house is the mapped one instead: see
    :func:`_observed_house`.
    """
    if ctx.site is not None:
        return _observed_house(ctx, ctx.site)
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    margin = float(ctx.param("lot_margin_m"))
    front_yard = float(ctx.param("front_yard_m"))

    if ctx.rng.random() < float(ctx.param("p_authored", default=_ALWAYS_PROCEDURAL)):
        return _authored_house(ctx, lot_x, lot_y, margin, front_yard)
    return _massed_house(ctx, lot_x, lot_y, margin, front_yard)


def _fit_offset(
    ctx: PlacementContext,
    bounds: tuple[float, float, float, float],
    lot_x: float,
    lot_y: float,
    margin: float,
    front_yard: float,
) -> npt.NDArray[np.float64]:
    """Draw a translation putting a plan's bounding box inside the buildable lot.

    Raises
    ------
    WorldgenError
        If the plan cannot fit the lot at all, which means the sampled
        dimensions and the configured margins disagree.
    """
    minx, miny, maxx, maxy = bounds
    width, depth = maxx - minx, maxy - miny

    x_span = lot_x / 2.0 - margin - width / 2.0
    y_lo = -lot_y / 2.0 + front_yard + depth / 2.0
    y_hi = lot_y / 2.0 - margin - depth / 2.0
    if x_span < 0.0 or y_lo > y_hi:
        msg = (
            f"role {ctx.role.name!r} sampled a {width:.1f} x {depth:.1f} m plan, which does "
            f"not fit a {lot_x} x {lot_y} m lot with {margin} m margins and a "
            f"{front_yard} m front yard"
        )
        raise WorldgenError(msg)

    target = np.array(
        [ctx.rng.uniform(-x_span, x_span), ctx.rng.uniform(y_lo, y_hi)], dtype=np.float64
    )
    return target - np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0], dtype=np.float64)


def _authored_house(
    ctx: PlacementContext, lot_x: float, lot_y: float, margin: float, front_yard: float
) -> list[Placement]:
    """Place the one authored shell, which already carries its own roof and openings."""
    spec = ctx.library.choose(str(ctx.param("authored_tag")), ctx.rng)
    extents = ctx.extents(spec)
    yaw = spec.sample_yaw(ctx.rng)

    plan = rotated_plan(extents, yaw)

    offset = _fit_offset(
        ctx,
        (-plan[0] / 2.0, -plan[1] / 2.0, plan[0] / 2.0, plan[1] / 2.0),
        lot_x,
        lot_y,
        margin,
        front_yard,
    )
    return [Placement(spec=spec, pos=as3(offset), size=extents, yaw=yaw)]


def _massed_house(
    ctx: PlacementContext, lot_x: float, lot_y: float, margin: float, front_yard: float
) -> list[Placement]:
    """Compose the house from blocks: a main mass plus abutting wings, each roofed."""
    mass_tag = str(ctx.param("mass_tag"))
    wing_tag = str(ctx.param("wing_tag"))
    roof_tag = str(ctx.param("roof_tag"))
    overlap = float(ctx.param("wing_overlap_m", default=_DEFAULT_WING_OVERLAP_M))
    wing_lo, wing_hi = (int(v) for v in ctx.param("wings"))

    main_spec = ctx.library.choose(mass_tag, ctx.rng)
    main_size = ctx.extents(main_spec)
    blocks: list[tuple[AssetSpec, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = [
        (main_spec, np.zeros(2, dtype=np.float64), main_size)
    ]

    for _ in range(int(ctx.rng.integers(wing_lo, wing_hi + 1))):
        wing_spec = ctx.library.choose(wing_tag, ctx.rng)
        wing_size = ctx.extents(wing_spec)
        # Abut a face of the main mass, overlapping slightly so the union is one
        # connected polygon rather than two blocks touching at a hairline.
        axis = int(ctx.rng.integers(0, 2))
        sign = float(ctx.rng.choice(np.array([-1.0, 1.0])))
        centre = np.zeros(2, dtype=np.float64)
        centre[axis] = sign * ((float(main_size[axis]) + float(wing_size[axis])) / 2.0 - overlap)
        # Slide along the shared face, keeping the wing within the main mass's
        # span so the plan stays an L or a T rather than growing a spur.
        other = 1 - axis
        slack = max(float(main_size[other]) - float(wing_size[other]), 0.0) / 2.0
        centre[other] = float(ctx.rng.uniform(-slack, slack))
        blocks.append((wing_spec, centre, wing_size))

    minx = min(float(c[0] - s[0] / 2.0) for _, c, s in blocks)
    maxx = max(float(c[0] + s[0] / 2.0) for _, c, s in blocks)
    miny = min(float(c[1] - s[1] / 2.0) for _, c, s in blocks)
    maxy = max(float(c[1] + s[1] / 2.0) for _, c, s in blocks)
    offset = _fit_offset(ctx, (minx, miny, maxx, maxy), lot_x, lot_y, margin, front_yard)

    placements: list[Placement] = []
    for spec, centre, size in blocks:
        at = centre + offset
        placements.append(Placement(spec=spec, pos=as3(at), size=size))
        placements.append(_roof_for(ctx, roof_tag, at, size))
    return placements


def _observed_house(ctx: PlacementContext, site: SiteEvidence) -> list[Placement]:
    """Rebuild the mapped house from its footprint blocks.

    The blocks are the footprint fitted in the world frame, so they are placed
    where they stand rather than fitted to the lot. The massing model is the
    one whose storey count matches the mapped one, and a single wall height is
    drawn for the whole house so every block shares its eaves -- a real house
    does not step its eaves line between wings of the same storey count. The
    roof is the mapped shape when there is one, and drawn per block otherwise.
    The authored shell is never used here: it is one fixed shape.
    """
    mass_tag = str(ctx.param("mass_tag"))
    roof_tag = str(ctx.param("roof_tag"))
    spec = _mass_for_storeys(ctx, mass_tag, site.levels)
    height = float(ctx.extents(spec)[2])
    roof_spec = _roof_of_kind(ctx, roof_tag, site.roof) if site.roof is not None else None
    roof_provenance = Provenance.OBSERVED if site.roof is not None else Provenance.INFERRED

    placements: list[Placement] = []
    for block in site.blocks:
        size = np.array([float(block.size[0]), float(block.size[1]), height], dtype=np.float64)
        placements.append(
            Placement(spec=spec, pos=as3(block.centre), size=size, provenance=Provenance.OBSERVED)
        )
        roof = _roof_for(ctx, roof_tag, block.centre, size, spec=roof_spec)
        placements.append(dataclasses.replace(roof, provenance=roof_provenance))
    return placements


def _storeys(spec: AssetSpec, storey_m: float) -> int:
    """How many storeys a massing model stands, from the middle of its height range."""
    if spec.size_z is None:
        return 1
    return max(round((spec.size_z[0] + spec.size_z[1]) / 2.0 / storey_m), 1)


def _mass_for_storeys(ctx: PlacementContext, tag: str, levels: int | None) -> AssetSpec:
    """Draw the massing model for a mapped storey count, or any one when it is unmapped."""
    if levels is None:
        return ctx.library.choose(tag, ctx.rng)
    storey_m = ctx.cfg.worldgen.site.storey_m
    found = ctx.library.candidates(tag)
    gaps = [abs(_storeys(spec, storey_m) - levels) for spec in found]
    return choose_among(
        ctx, [spec for spec, gap in zip(found, gaps, strict=True) if gap == min(gaps)]
    )


def _roof_of_kind(ctx: PlacementContext, tag: str, kind: str) -> AssetSpec:
    """Draw a roof model of the mapped kind.

    A model declaring its own ``size_z`` is the flat roof (see :func:`_roof_for`);
    one without is a slope. If the library has no model of the kind, any roof
    beats no roof, so the draw falls back to the whole tag.
    """
    found = [
        spec for spec in ctx.library.candidates(tag) if (spec.size_z is None) == (kind == "gable")
    ]
    return choose_among(ctx, found) if found else ctx.library.choose(tag, ctx.rng)


def _roof_for(
    ctx: PlacementContext,
    roof_tag: str,
    centre: npt.NDArray[np.float64],
    mass_size: npt.NDArray[np.float64],
    *,
    spec: AssetSpec | None = None,
) -> Placement:
    """Cap one block with a roof, overhanging by the eaves.

    A roof model that declares its own ``size_z`` is taken at its word -- that
    is a flat roof's parapet. One that does not is a slope, and its height falls
    out of the pitch and the span perpendicular to the ridge. The ridge runs
    along the block's long axis, which for art authored ridge-along-X means a
    quarter turn when the block is deeper than it is wide. ``spec`` fixes the
    model, as a mapped roof shape does; otherwise it is drawn from the tag.
    """
    eave = float(ctx.param("eave_overhang_m"))
    pitch_deg = float(ctx.param("pitch_deg"))
    if spec is None:
        spec = ctx.library.choose(roof_tag, ctx.rng)

    span_x = float(mass_size[0]) + 2.0 * eave
    span_y = float(mass_size[1]) + 2.0 * eave
    ridge_along_x = span_x >= span_y
    # Extents are measured before yaw, so a quarter turn swaps them back.
    plan = (span_x, span_y) if ridge_along_x else (span_y, span_x)
    yaw = 0.0 if ridge_along_x else math.pi / 2.0

    if spec.size_z is not None:
        height = float(ctx.rng.uniform(*spec.size_z))
    else:
        cross_span = span_y if ridge_along_x else span_x
        height = math.tan(math.radians(pitch_deg)) * cross_span / 2.0

    return Placement(
        spec=spec,
        pos=as3(centre, float(mass_size[2])),
        size=np.array([plan[0], plan[1], height], dtype=np.float64),
        yaw=yaw,
    )


#: Tolerance for a point on the outline to count as on a block's boundary.
_ON_WALL_TOL_M = 1e-6


def wall_height_at(house: HouseFrame, xy: npt.NDArray[np.float64]) -> float:
    """Height of the wall at a point on the outline: its own block's height.

    A straight exterior wall can run along a two-storey block and then on along
    a flush single-storey wing, so height is a property of the point, not of the
    segment. Public because :mod:`canopy.worldgen.equipment` needs it too, to
    fit windows and doors around the eaves line of a mixed-storey house.
    """
    heights = [float(m.size[2]) for m in house.masses if m.contains(xy, _ON_WALL_TOL_M)]
    return max(heights, default=house.height_m)
