"""Placement kernel: the geometry, the rule registry, and the helpers every rule shares.

A rule answers one question -- "given the property so far, where do ``n`` of
this role's models belong?" -- and answers it in world coordinates. Rules are
looked up by name from the ``roles:`` table in ``assets/models/index.yaml``, so
adding a prop that belongs somewhere an existing rule already covers takes no
code at all. Writing a new rule is the one thing that does:

.. code-block:: python

    @register_rule("under_the_eaves")
    def _under_the_eaves(ctx: PlacementContext) -> list[Placement]:
        house = ctx.require_house()
        ...

Every rule draws from ``ctx.rng``, the single generator threaded through
generation, and only from it: the whole property must be reproducible from its
seed. Rules run in role order, so a later rule can measure against anything an
earlier one placed -- which is why the meter is sited before the bushes that may
be asked to occlude it.

Coordinates are metres in the Z-up world frame, origin at lot centre, ground at
``z = 0``. A model's placement point is its own origin, which the library
authors at the ground or wall contact point.

A property built from a real address (``PlacementContext.site``) runs these
same rules. Where the map data observed something -- the house's footprint,
the neighbours, the street, mapped trees -- a rule places it there instead of
drawing it; everything else is drawn exactly as for a seed-only property. The
world is turned so the real street runs along ``-y``, which is the layout every
rule here already assumes. Every address-only branch is guarded by
``ctx.site is not None``, so a seed-only property draws, and writes, exactly
what it always did.

This module is the kernel three rule families build on rather than a family
of its own: the geometry a rule reasons about (:class:`Opening`,
:class:`WallSegment`, :class:`Mass`, :class:`HouseFrame`, :class:`Obstacle`,
:class:`Placement`), the interface a rule is written against
(:class:`PlacementContext`, :func:`register_rule`, :func:`get_rule`), and the
generic helpers -- clearance tests, wall selection, weighted draws -- used by
more than one family. The families themselves live beside it:
:mod:`canopy.worldgen.house` (the house_lot rule and its massing),
:mod:`canopy.worldgen.equipment` (wall-mounted equipment and the openings
fitted around it) and :mod:`canopy.worldgen.background` (the ground,
neighbours and street). Each of those modules imports this one; this module
imports none of them, so registering a new rule never risks a cycle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from canopy.contracts import Cls, Provenance
from canopy.errors import WorldgenError

if TYPE_CHECKING:
    from canopy.config import Config
    from canopy.contracts import Vec3
    from canopy.worldgen.assets import AssetLibrary, AssetSpec, RoleSpec
    from canopy.worldgen.evidence import SiteEvidence


__all__ = [
    "OPENING_CLASSES",
    "HouseFrame",
    "Mass",
    "Obstacle",
    "Opening",
    "Placement",
    "PlacementContext",
    "PlacementRule",
    "WallSegment",
    "defines_house",
    "get_rule",
    "house_frame",
    "register_rule",
    "rotated_plan",
    "rule_names",
]


#: Attempts a rejection sampler gets per object before giving up. Generous: a
#: crowded lot should thin out, not abort the mission.
MAX_ATTEMPTS = 60


#: Yaw within this many radians of a quarter turn counts as one, for deciding
#: whether a footprint's x and y extents swap.
_QUARTER_TURN_TOL = 1e-6


#: Two outward normals whose dot product exceeds this are the same wall.
SAME_WALL_DOT = 0.99


#: Outline vertices closer to collinear than this are merged, so a wall split
#: by an abutting wing reads as one segment.
_COLLINEAR_TOL_M = 1e-9


#: Exterior edges shorter than this are slivers, not mountable walls.
_MIN_WALL_M = 0.35


#: Guards a division by the length of a degenerate wall segment.
EPS_M = 1e-9


#: Semantic classes that are holes in a wall rather than wall.
OPENING_CLASSES = frozenset({Cls.WINDOW, Cls.DOOR, Cls.GARAGE_DOOR})


# ---------------------------------------------------------------------------
# Geometry a rule reasons about
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Opening:
    """A window or door the house model carries, as the stretch of wall it fills.

    Only the authored shell has these at the time equipment is sited;
    procedural massing gets its openings last, fitted around the equipment,
    so its walls carry none.
    """

    lo_m: float
    """Start, in metres along the wall from :attr:`WallSegment.a`."""
    hi_m: float
    """End, in metres along the wall from :attr:`WallSegment.a`."""
    bottom_m: float
    """Height of the opening's lowest point above the ground."""
    cls: Cls
    """``WINDOW``, ``DOOR`` or ``GARAGE_DOOR``."""


@dataclass(frozen=True, slots=True, eq=False)
class WallSegment:
    """One exterior wall of the house, as a ground-level line segment."""

    a: npt.NDArray[np.float64]
    """Start point ``(x, y)``."""
    b: npt.NDArray[np.float64]
    """End point ``(x, y)``."""
    normal: npt.NDArray[np.float64]
    """Outward unit normal ``(x, y)``, pointing away from the house."""
    openings: tuple[Opening, ...] = ()
    """Windows and doors already in this wall, which nothing may be mounted over."""

    @property
    def length(self) -> float:
        """Segment length in metres."""
        return float(np.linalg.norm(self.b - self.a))

    def at(self, t: float) -> npt.NDArray[np.float64]:
        """Point a fraction ``t`` of the way from :attr:`a` to :attr:`b`."""
        return self.a + (self.b - self.a) * t

    def along(self, xy: npt.NDArray[np.float64]) -> float:
        """Distance of ``xy``'s projection along the wall from :attr:`a`, in metres."""
        unit = (self.b - self.a) / max(self.length, EPS_M)
        return float(np.dot(xy[:2] - self.a, unit))


@dataclass(frozen=True, slots=True, eq=False)
class Mass:
    """One axis-aligned rectangular block of the house, in world coordinates.

    A house is a union of these. Keeping each one axis-aligned is what lets the
    union stay a simple polygon that the site solver's world-axis cost map can
    consume, while still producing L and T plans rather than a single box.
    """

    centre: npt.NDArray[np.float64]
    """``(x, y)`` of this block's footprint centre."""
    size: npt.NDArray[np.float64]
    """``(x, y, z)`` extents: footprint and wall height."""

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)``."""
        cx, cy = float(self.centre[0]), float(self.centre[1])
        hx, hy = float(self.size[0]) / 2.0, float(self.size[1]) / 2.0
        return (cx - hx, cy - hy, cx + hx, cy + hy)

    def contains(self, xy: npt.NDArray[np.float64], margin_m: float = 0.0) -> bool:
        """Whether ``xy`` lies inside this block grown by ``margin_m``."""
        gap = np.abs(xy - self.centre) - (self.size[:2] / 2.0 + margin_m)
        return bool(np.all(gap <= 0.0))


@dataclass(frozen=True, slots=True, eq=False)
class HouseFrame:
    """The sited house, as everything downstream needs to see it.

    The house is one or more :class:`Mass` blocks, and this frame is their
    union: ``footprint`` is the outline, ``walls`` are its exterior edges with
    outward normals, and everything that measures from "the house" -- the
    meter's wall, the foundation planting band, the tree clearance -- measures
    from those.

    Holes are dropped. A courtyard would give the outline an interior ring that
    no consumer here handles, and no residential plan this generator produces
    should have one.
    """

    masses: tuple[Mass, ...]
    footprint: tuple[tuple[float, float], ...]
    """Union outline, counter-clockwise, first point not repeated."""
    walls: tuple[WallSegment, ...] = ()
    height_m: float = 0.0
    """Tallest block's wall height. What the orbit rings have to clear."""
    centre: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    """Centroid of the union."""
    size: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    """``(x, y, z)`` extents of the union's bounding box, and the height."""
    has_openings: bool = False
    """Whether the house models already carry windows and doors.

    The authored shell does; procedural massing does not, and gets its openings
    from the ``facade_openings`` rule instead."""

    @property
    def perimeter_m(self) -> float:
        """Total exterior wall length, which is what sets the bush count."""
        return float(sum(w.length for w in self.walls))

    def footprint_ccw(self) -> list[tuple[float, float]]:
        """Return the outline as :class:`~canopy.contracts.SceneManifest` wants it."""
        return [(float(x), float(y)) for x, y in self.footprint]

    def contains(self, xy: npt.NDArray[np.float64], margin_m: float = 0.0) -> bool:
        """Whether ``xy`` lies inside any block grown by ``margin_m``.

        Tested per block rather than against the buffered union: the two differ
        only just outside a reflex corner, and this stays a handful of array
        comparisons instead of a shapely call per candidate placement.
        """
        return any(m.contains(xy, margin_m) for m in self.masses)


def house_frame(masses: Sequence[Mass]) -> HouseFrame:
    """Union one or more blocks into a :class:`HouseFrame`.

    Parameters
    ----------
    masses
        The house's blocks. Must overlap or touch enough to form one connected
        plan; two detached blocks are a bug in the rule that produced them, not
        a house.

    Raises
    ------
    WorldgenError
        If no masses are given, or if they do not union into a single polygon.
    """
    blocks = tuple(masses)
    if not blocks:
        msg = "a house needs at least one mass"
        raise WorldgenError(msg)

    union = unary_union([box(*m.bounds_xy) for m in blocks])
    if union.geom_type != "Polygon":
        msg = (
            f"the house's {len(blocks)} masses do not touch, so they union into a "
            f"{union.geom_type} rather than one connected plan"
        )
        raise WorldgenError(msg)

    # Drop any interior ring, force counter-clockwise, and merge the collinear
    # vertices that a shared block edge leaves behind so one straight wall is
    # one segment.
    ring = orient(Polygon(union.exterior), sign=1.0).simplify(_COLLINEAR_TOL_M)
    coords = [(float(x), float(y)) for x, y in ring.exterior.coords[:-1]]

    walls = []
    for i, a in enumerate(coords):
        b = coords[(i + 1) % len(coords)]
        edge = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64)
        length = float(np.linalg.norm(edge))
        if length < _MIN_WALL_M:
            # A sliver left by a small overlap. It stays in the outline but is
            # not somewhere a meter or a bush can go.
            continue
        # For a counter-clockwise ring the outward normal of a->b is (dy, -dx).
        normal = np.array([edge[1], -edge[0]], dtype=np.float64) / length
        walls.append(
            WallSegment(
                a=np.array(a, dtype=np.float64), b=np.array(b, dtype=np.float64), normal=normal
            )
        )

    minx, miny, maxx, maxy = ring.bounds
    height = max(float(m.size[2]) for m in blocks)
    return HouseFrame(
        masses=blocks,
        footprint=tuple(coords),
        walls=tuple(walls),
        height_m=height,
        centre=np.array([ring.centroid.x, ring.centroid.y], dtype=np.float64),
        size=np.array([maxx - minx, maxy - miny, height], dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False)
class Obstacle:
    """A placed object's 2D footprint, as a disc, for clearance tests.

    A disc rather than the real polygon on purpose: bushes and tree crowns are
    round, and exact mesh intersection per candidate would cost far more than
    the occasional overlapping leaf is worth.
    """

    xy: npt.NDArray[np.float64]
    radius_m: float


@dataclass(frozen=True, slots=True, eq=False)
class Placement:
    """One object's model, pose and chosen size, before its mesh is built.

    The model travels with the placement rather than being fixed per role, so
    one role can mix models: a hedge of two bush shapes is one
    ``foundation_band`` call, not two.
    """

    spec: AssetSpec
    pos: npt.NDArray[np.float64]
    """World position of the model's origin ``(x, y, z)``."""
    size: npt.NDArray[np.float64]
    """Target extents ``(x, y, z)`` in metres, measured *before* ``yaw``."""
    yaw: float = 0.0
    wall_normal: npt.NDArray[np.float64] | None = None
    """Outward wall normal ``(x, y, z)`` for wall-mounted objects; ``None`` otherwise."""
    background: bool = False
    """Whether this is neighbouring scenery rather than the surveyed lot.

    Set from :attr:`~canopy.worldgen.assets.RoleSpec.background` by the
    generator after a rule returns, not by the rule itself: a rule places
    geometry, it does not know which lot it is placing it for."""
    provenance: Provenance = Provenance.INFERRED
    """Where the rule got this placement: map data, its own draw, or a repair.
    Copied onto every :class:`~canopy.contracts.SceneObject` it becomes."""

    @property
    def radius_m(self) -> float:
        """Clearance radius for :class:`Obstacle` bookkeeping."""
        return float(np.max(self.size[:2])) / 2.0

    def obstacle(self) -> Obstacle:
        """Return this placement as an :class:`Obstacle`."""
        return Obstacle(xy=self.pos[:2].copy(), radius_m=self.radius_m)


# ---------------------------------------------------------------------------
# Rule interface
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class PlacementContext:
    """Everything a rule may read. A rule mutates nothing here."""

    rng: np.random.Generator
    cfg: Config
    library: AssetLibrary
    spec: AssetSpec
    """Default model for the role, already weighted-sampled. Use it where one
    model governs the whole role; call :meth:`pick` per object otherwise."""
    role: RoleSpec
    n: int
    """How many objects to place. May be zero."""
    lot_bounds: npt.NDArray[np.float64]
    """``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``, shape ``(2, 3)``."""
    house: HouseFrame | None = None
    """``None`` only while the ground and the house itself are being placed."""
    obstacles: tuple[Obstacle, ...] = ()
    """Footprints of everything placed so far, for clearance tests. Starts with
    :attr:`launch_area`, so every rule that tests clearance keeps out of it."""
    launch_area: tuple[Obstacle, ...] = ()
    """One disc per launch pad column, which nothing may stand in. Also in
    :attr:`obstacles`; repeated here for rules whose objects are not discs."""
    placed: tuple[Placement, ...] = ()
    """Every surveyed-lot placement so far, for rules that need more than a disc.

    A rule fitting something between wall-mounted equipment needs to know what
    is flush against which wall, which an :class:`Obstacle` cannot say."""
    meter: Placement | None = None
    """The sited electric meter, once the meter role has run."""
    site: SiteEvidence | None = None
    """What a real address's map data pins down; ``None`` for a seed-only property."""

    def pick(self) -> AssetSpec:
        """Draw another model for this role, weighted. Lets one role mix models."""
        return self.library.choose(self.role.tag, self.rng)

    def extents(self, spec: AssetSpec) -> Vec3:
        """Draw target world extents for one instance of ``spec``."""
        return self.library.sample_extents(spec, self.rng)

    def param(self, name: str, default: Any = None) -> Any:
        """Read one of the role's ``params``.

        Raises
        ------
        WorldgenError
            If the parameter is absent and no default was supplied, which means
            the library and the rule disagree about the rule's interface.
        """
        if name in self.role.params:
            return self.role.params[name]
        if default is None:
            msg = (
                f"role {self.role.name!r} uses rule {self.role.rule!r}, which requires "
                f"params.{name}"
            )
            raise WorldgenError(msg)
        return default

    def require_house(self) -> HouseFrame:
        """Return the sited house, or raise a clear error about role ordering.

        Raises
        ------
        WorldgenError
            If no house has been placed yet.
        """
        if self.house is None:
            msg = (
                f"role {self.role.name!r} uses rule {self.role.rule!r}, which measures from "
                "the house, so it must be listed after the house role in the model index"
            )
            raise WorldgenError(msg)
        return self.house


#: What every generation rule looks like.
PlacementRule = Callable[["PlacementContext"], list["Placement"]]


_RULES: dict[str, PlacementRule] = {}

_HOUSE_RULES: set[str] = set()


def register_rule(
    name: str, *, defines_house: bool = False
) -> Callable[[PlacementRule], PlacementRule]:
    """Register a placement rule under the name the model index uses.

    Parameters
    ----------
    name
        Name the ``roles:`` table refers to.
    defines_house
        Whether this rule's placement establishes the :class:`HouseFrame` that
        later rules measure against. Declaring it here rather than hard-coding
        a rule name in the generator means a future alternative house rule --
        an L-shaped footprint, say -- needs no change outside this module.

    Raises
    ------
    WorldgenError
        On a duplicate name, which would otherwise silently shadow a rule.
    """

    def decorate(fn: PlacementRule) -> PlacementRule:
        if name in _RULES:
            msg = f"placement rule {name!r} is already registered by {_RULES[name].__name__}"
            raise WorldgenError(msg)
        _RULES[name] = fn
        if defines_house:
            _HOUSE_RULES.add(name)
        return fn

    return decorate


def defines_house(name: str) -> bool:
    """Whether the named rule establishes the house frame."""
    return name in _HOUSE_RULES


def get_rule(name: str) -> PlacementRule:
    """Look up a registered rule.

    Raises
    ------
    WorldgenError
        If no rule of that name exists, listing the ones that do.
    """
    try:
        return _RULES[name]
    except KeyError as exc:
        msg = f"unknown placement rule {name!r}; registered rules are {rule_names()}"
        raise WorldgenError(msg) from exc


def rule_names() -> list[str]:
    """Every registered rule name, sorted."""
    return sorted(_RULES)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def rotated_plan(extents: npt.NDArray[np.float64], yaw: float) -> npt.NDArray[np.float64]:
    """Extents after a quarter-turn yaw, which simply swaps x and y.

    Raises
    ------
    WorldgenError
        If ``yaw`` is not a multiple of 90 degrees. Houses turn in quarter turns
        only, so that their footprints stay axis-aligned for the site solver.
    """
    turns = yaw / (math.pi / 2.0)
    if abs(turns - round(turns)) > _QUARTER_TURN_TOL:
        msg = (
            "a house model's yaw must be a multiple of 90 degrees so the footprint stays "
            f"axis-aligned, got {math.degrees(yaw):.3f} degrees"
        )
        raise WorldgenError(msg)
    ext = np.asarray(extents, dtype=np.float64)
    return ext if round(turns) % 2 == 0 else np.array([ext[1], ext[0], ext[2]])


def yaw_onto(facing_xy: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]) -> float:
    """Yaw that turns a model's authored facing direction onto ``normal``.

    The library records which way each wall-mounted model was authored to face
    (:meth:`AssetSpec.facing_xy`), so this is what makes "flush against that
    wall, reading outward" a single number instead of a per-model special case.
    """
    return float(
        math.atan2(float(normal[1]), float(normal[0]))
        - math.atan2(float(facing_xy[1]), float(facing_xy[0]))
    )


def walls_for(ctx: PlacementContext, *, avoid_meter_wall: bool) -> tuple[WallSegment, ...]:
    """Return the house walls a rule may use, optionally excluding the meter's own wall."""
    walls = ctx.require_house().walls
    if not avoid_meter_wall or ctx.meter is None or ctx.meter.wall_normal is None:
        return walls
    meter_n = ctx.meter.wall_normal[:2]
    kept = tuple(w for w in walls if float(np.dot(w.normal, meter_n)) < SAME_WALL_DOT)
    # Falling back to every wall beats failing: a house could in principle have
    # only one usable wall, and a second meter on the same wall is a cosmetic
    # problem, not a broken property.
    return kept or walls


def choose_wall(
    ctx: PlacementContext,
    clearance_m: float,
    walls: Sequence[WallSegment] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
) -> tuple[WallSegment, float]:
    """Pick a wall in proportion to its usable length, and a point along it.

    Returns the segment and a fraction ``t`` at least ``clearance_m`` from both
    corners, so nothing lands on an outside corner where it would be half
    hidden from every viewpoint. ``weights``, one per wall, scale each wall's
    odds on top of its length: how an address-built property says the meter
    is likelier on some sides of a real house than others.

    Raises
    ------
    WorldgenError
        If no wall is long enough to satisfy the clearance.
    """
    usable_walls = tuple(walls) if walls is not None else ctx.require_house().walls
    usable = np.array(
        [max(w.length - 2.0 * clearance_m, 0.0) for w in usable_walls], dtype=np.float64
    )
    if weights is not None:
        usable = usable * np.asarray(weights, dtype=np.float64)
    if not np.any(usable > 0.0):
        msg = (
            f"role {ctx.role.name!r}: no wall is longer than 2 x {clearance_m} m of corner "
            f"clearance (walls: {[round(w.length, 2) for w in usable_walls]})"
        )
        raise WorldgenError(msg)
    wall = usable_walls[int(ctx.rng.choice(len(usable_walls), p=usable / usable.sum()))]
    along = float(ctx.rng.uniform(clearance_m, wall.length - clearance_m))
    return wall, along / wall.length


def clear(xy: npt.NDArray[np.float64], radius_m: float, obstacles: Sequence[Obstacle]) -> bool:
    """Whether a disc at ``xy`` avoids every obstacle. Vectorised over obstacles."""
    if not obstacles:
        return True
    centres = np.array([o.xy for o in obstacles], dtype=np.float64)
    radii = np.array([o.radius_m for o in obstacles], dtype=np.float64)
    return bool(np.all(np.linalg.norm(centres - xy, axis=1) >= radii + radius_m))


def crosses(
    centre: npt.NDArray[np.float64],
    unit: npt.NDArray[np.float64],
    size: npt.NDArray[np.float64],
    obstacles: Sequence[Obstacle],
) -> bool:
    """Whether a straight panel overlaps any obstacle's disc.

    The panel runs ``size[0]`` along ``unit`` and is ``size[1]`` thick, centred
    on ``centre``. A long panel is a poor fit for :func:`clear`'s disc, which
    would either miss its ends or claim far too much beside it.
    """
    if not obstacles:
        return False
    centres = np.array([o.xy for o in obstacles], dtype=np.float64)
    radii = np.array([o.radius_m for o in obstacles], dtype=np.float64)
    rel = centres - centre[:2]
    along = np.clip(rel @ unit, -size[0] / 2.0, size[0] / 2.0)
    gap = np.linalg.norm(rel - along[:, None] * unit, axis=1) - size[1] / 2.0
    return bool(np.any(gap < radii))


def as3(xy: npt.NDArray[np.float64], z: float = 0.0) -> npt.NDArray[np.float64]:
    """Lift a 2D point to 3D."""
    return np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)


def seen(ctx: PlacementContext, placements: Sequence[Placement]) -> tuple[Obstacle, ...]:
    """Everything placed before this rule, plus what it has placed so far."""
    return (*ctx.obstacles, *(p.obstacle() for p in placements))


def inside_lot(ctx: PlacementContext, xy: npt.NDArray[np.float64], radius_m: float) -> bool:
    """Whether a disc at ``xy`` stays within the lot.

    Only address-built properties need asking: a real side yard can be narrower
    than a bush's standoff, and anything past the lot line lies outside the
    voxel grid the mapper sizes from the lot.
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    return (
        abs(float(xy[0])) + radius_m <= lot_x / 2.0 and abs(float(xy[1])) + radius_m <= lot_y / 2.0
    )


def choose_among(ctx: PlacementContext, specs: Sequence[AssetSpec]) -> AssetSpec:
    """Weighted draw from an already-filtered list of models, as ``library.choose`` draws."""
    weights = np.array([spec.weight for spec in specs], dtype=np.float64)
    return specs[int(ctx.rng.choice(len(specs), p=weights / weights.sum()))]


def clear_of_openings(
    wall: WallSegment,
    group: Sequence[Placement],
    gap_m: float,
    *,
    below_m: float = math.inf,
) -> bool:
    """Whether a group of placements against ``wall`` covers none of its openings.

    The group is taken as one stretch of wall, from the near edge of its first
    piece to the far edge of its last, so a conduit bridging a meter and a
    panel cannot straddle a window between them. Each piece is as wide as its
    clearance disc, as in :func:`~canopy.worldgen.equipment._blocked_spans`. That
    stretch must miss every opening by ``gap_m``, whatever the heights involved: a meter has conduit
    running down to grade, so a window below it or above it is equally in the
    way.

    ``below_m`` relaxes that for a unit standing on the ground: only openings
    whose bottom is lower than it count, so an AC condenser may sit under a
    window but never in front of a door.
    """
    if not wall.openings or not group:
        return True
    alongs = [wall.along(p.pos) for p in group]
    lo = min(a - p.radius_m for a, p in zip(alongs, group, strict=True))
    hi = max(a + p.radius_m for a, p in zip(alongs, group, strict=True))
    blocked = [(o.lo_m - gap_m, o.hi_m + gap_m) for o in wall.openings if o.bottom_m < below_m]
    return is_free(lo, hi, blocked)


def is_free(lo: float, hi: float, spans: Sequence[tuple[float, float]]) -> bool:
    """Whether ``[lo, hi]`` overlaps none of ``spans``."""
    return all(hi <= s_lo or lo >= s_hi for s_lo, s_hi in spans)
