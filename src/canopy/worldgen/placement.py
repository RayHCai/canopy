"""Generation rules: where each model goes.

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

from canopy.contracts import Cls
from canopy.errors import WorldgenError

if TYPE_CHECKING:
    from canopy.config import Config
    from canopy.contracts import Vec3
    from canopy.worldgen.assets import AssetLibrary, AssetSpec, RoleSpec

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
_MAX_ATTEMPTS = 60

#: Yaw within this many radians of a quarter turn counts as one, for deciding
#: whether a footprint's x and y extents swap.
_QUARTER_TURN_TOL = 1e-6

#: Two outward normals whose dot product exceeds this are the same wall.
_SAME_WALL_DOT = 0.99

#: Outline vertices closer to collinear than this are merged, so a wall split
#: by an abutting wing reads as one segment.
_COLLINEAR_TOL_M = 1e-9

#: Exterior edges shorter than this are slivers, not mountable walls.
_MIN_WALL_M = 0.35

#: Guards a division by the length of a degenerate wall segment.
_EPS_M = 1e-9

#: Default overlap between a wing and the main mass, so the union connects.
_DEFAULT_WING_OVERLAP_M = 0.3

#: p_authored default: with no value in the library, never use the shell.
_ALWAYS_PROCEDURAL = 0.0


#: Semantic classes that are holes in a wall rather than wall.
OPENING_CLASSES = frozenset({Cls.WINDOW, Cls.DOOR, Cls.GARAGE_DOOR})

#: Kept clear either side of an opening by anything fixed to or set against a
#: wall, when the role gives no ``opening_clearance_m`` of its own.
_OPENING_CLEARANCE_M = 0.3


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
        unit = (self.b - self.a) / max(self.length, _EPS_M)
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


def _yaw_onto(facing_xy: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]) -> float:
    """Yaw that turns a model's authored facing direction onto ``normal``.

    The library records which way each wall-mounted model was authored to face
    (:meth:`AssetSpec.facing_xy`), so this is what makes "flush against that
    wall, reading outward" a single number instead of a per-model special case.
    """
    return float(
        math.atan2(float(normal[1]), float(normal[0]))
        - math.atan2(float(facing_xy[1]), float(facing_xy[0]))
    )


def _walls_for(ctx: PlacementContext, *, avoid_meter_wall: bool) -> tuple[WallSegment, ...]:
    """Return the house walls a rule may use, optionally excluding the meter's own wall."""
    walls = ctx.require_house().walls
    if not avoid_meter_wall or ctx.meter is None or ctx.meter.wall_normal is None:
        return walls
    meter_n = ctx.meter.wall_normal[:2]
    kept = tuple(w for w in walls if float(np.dot(w.normal, meter_n)) < _SAME_WALL_DOT)
    # Falling back to every wall beats failing: a house could in principle have
    # only one usable wall, and a second meter on the same wall is a cosmetic
    # problem, not a broken property.
    return kept or walls


def _choose_wall(
    ctx: PlacementContext,
    clearance_m: float,
    walls: Sequence[WallSegment] | None = None,
) -> tuple[WallSegment, float]:
    """Pick a wall in proportion to its usable length, and a point along it.

    Returns the segment and a fraction ``t`` at least ``clearance_m`` from both
    corners, so nothing lands on an outside corner where it would be half
    hidden from every viewpoint.

    Raises
    ------
    WorldgenError
        If no wall is long enough to satisfy the clearance.
    """
    usable_walls = tuple(walls) if walls is not None else ctx.require_house().walls
    usable = np.array(
        [max(w.length - 2.0 * clearance_m, 0.0) for w in usable_walls], dtype=np.float64
    )
    if not np.any(usable > 0.0):
        msg = (
            f"role {ctx.role.name!r}: no wall is longer than 2 x {clearance_m} m of corner "
            f"clearance (walls: {[round(w.length, 2) for w in usable_walls]})"
        )
        raise WorldgenError(msg)
    wall = usable_walls[int(ctx.rng.choice(len(usable_walls), p=usable / usable.sum()))]
    along = float(ctx.rng.uniform(clearance_m, wall.length - clearance_m))
    return wall, along / wall.length


def _clear(xy: npt.NDArray[np.float64], radius_m: float, obstacles: Sequence[Obstacle]) -> bool:
    """Whether a disc at ``xy`` avoids every obstacle. Vectorised over obstacles."""
    if not obstacles:
        return True
    centres = np.array([o.xy for o in obstacles], dtype=np.float64)
    radii = np.array([o.radius_m for o in obstacles], dtype=np.float64)
    return bool(np.all(np.linalg.norm(centres - xy, axis=1) >= radii + radius_m))


def _crosses(
    centre: npt.NDArray[np.float64],
    unit: npt.NDArray[np.float64],
    size: npt.NDArray[np.float64],
    obstacles: Sequence[Obstacle],
) -> bool:
    """Whether a straight panel overlaps any obstacle's disc.

    The panel runs ``size[0]`` along ``unit`` and is ``size[1]`` thick, centred
    on ``centre``. A long panel is a poor fit for :func:`_clear`'s disc, which
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


def _as3(xy: npt.NDArray[np.float64], z: float = 0.0) -> npt.NDArray[np.float64]:
    """Lift a 2D point to 3D."""
    return np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)


def _seen(ctx: PlacementContext, placements: Sequence[Placement]) -> tuple[Obstacle, ...]:
    """Everything placed before this rule, plus what it has placed so far."""
    return (*ctx.obstacles, *(p.obstacle() for p in placements))


def _clear_of_openings(
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
    clearance disc, as in :func:`_blocked_spans`. That stretch must miss every
    opening by ``gap_m``, whatever the heights involved: a meter has conduit
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
    return _is_free(lo, hi, blocked)


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
    """
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
    return [Placement(spec=spec, pos=_as3(offset), size=extents, yaw=yaw)]


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
        placements.append(Placement(spec=spec, pos=_as3(at), size=size))
        placements.append(_roof_for(ctx, roof_tag, at, size))
    return placements


def _roof_for(
    ctx: PlacementContext,
    roof_tag: str,
    centre: npt.NDArray[np.float64],
    mass_size: npt.NDArray[np.float64],
) -> Placement:
    """Cap one block with a roof, overhanging by the eaves.

    A roof model that declares its own ``size_z`` is taken at its word -- that
    is a flat roof's parapet. One that does not is a slope, and its height falls
    out of the pitch and the span perpendicular to the ridge. The ridge runs
    along the block's long axis, which for art authored ridge-along-X means a
    quarter turn when the block is deeper than it is wide.
    """
    eave = float(ctx.param("eave_overhang_m"))
    pitch_deg = float(ctx.param("pitch_deg"))
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
        pos=_as3(centre, float(mass_size[2])),
        size=np.array([plan[0], plan[1], height], dtype=np.float64),
        yaw=yaw,
    )


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
    walls = _walls_for(ctx, avoid_meter_wall=bool(ctx.param("avoid_meter_wall", default=False)))

    placements: list[Placement] = []
    for _ in range(ctx.n):
        for _attempt in range(_MAX_ATTEMPTS):
            wall, t = _choose_wall(ctx, clearance, walls)
            spec = ctx.pick()
            xy = wall.at(t) + wall.normal * standoff
            candidate = Placement(
                spec=spec,
                pos=_as3(xy, bottom),
                size=ctx.extents(spec),
                yaw=_yaw_onto(spec.facing_xy(), wall.normal),
                wall_normal=_as3(wall.normal),
            )
            if _clear_of_openings(wall, [candidate], gap):
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
        for _attempt in range(_MAX_ATTEMPTS):
            wall, t = _choose_wall(ctx, clearance)
            spec = ctx.pick()
            size = ctx.extents(spec)
            xy = wall.at(t) + wall.normal * (standoff + float(np.max(size[:2])) / 2.0)
            candidate = Placement(
                spec=spec,
                pos=_as3(xy),
                size=size,
                yaw=_yaw_onto(spec.facing_xy(), wall.normal),
                wall_normal=_as3(wall.normal),
            )
            if _clear(xy, candidate.radius_m, _seen(ctx, placements)) and _clear_of_openings(
                wall, [candidate], gap, below_m=float(size[2]) + gap
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
            Placement(spec=spec, pos=_as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
        )

    while len(placements) < ctx.n:
        placed = False
        for _attempt in range(_MAX_ATTEMPTS):
            wall, t = _choose_wall(ctx, clearance_m=0.0)
            spec = ctx.pick()
            size = ctx.extents(spec)
            standoff = float(ctx.rng.uniform(standoff_lo, standoff_hi))
            xy = wall.at(t) + wall.normal * (standoff + float(np.max(size[:2])) / 2.0)
            candidate = Placement(spec=spec, pos=_as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
            if _clear(xy, candidate.radius_m, _seen(ctx, placements)):
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

#: Tolerance for a point on the outline to count as on a block's boundary.
_ON_WALL_TOL_M = 1e-6

#: A wall this much short of a whole number of storeys still counts as having
#: them: a 5.6 m two-storey block is two 2.8 m storeys, not one.
_STOREY_SLACK_M = 0.3


def _wall_height_at(house: HouseFrame, xy: npt.NDArray[np.float64]) -> float:
    """Height of the wall at a point on the outline: its own block's height.

    A straight exterior wall can run along a two-storey block and then on along
    a flush single-storey wing, so height is a property of the point, not of the
    segment.
    """
    heights = [float(m.size[2]) for m in house.masses if m.contains(xy, _ON_WALL_TOL_M)]
    return max(heights, default=house.height_m)


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
    unit = (wall.b - wall.a) / max(wall.length, _EPS_M)
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


def _is_free(lo: float, hi: float, spans: Sequence[tuple[float, float]]) -> bool:
    """Whether ``[lo, hi]`` overlaps none of ``spans``."""
    return all(hi <= s_lo or lo >= s_hi for s_lo, s_hi in spans)


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
                pos=_as3(wall.at(along_m / wall.length), bottom_m),
                size=np.array([w_m, depth, h_m], dtype=np.float64),
                yaw=math.atan2(float(edge[1]), float(edge[0])),
                wall_normal=_as3(wall.normal),
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
                for _attempt in range(_MAX_ATTEMPTS):
                    along = float(ctx.rng.uniform(corner + w_m / 2.0, length - corner - w_m / 2.0))
                    lo, hi = along - w_m / 2.0, along + w_m / 2.0
                    if _is_free(lo, hi, spans):
                        panel(tag, index, along, 0.0, w_m, h_m)
                        taken = (lo - gap, hi + gap)
                        flush[index].append(taken)
                        for others in approach[index]:
                            others.append(taken)
                        return

    def height(index: int) -> float:
        return _wall_height_at(house, walls[index].at(0.5))

    front = [i for i, w in enumerate(walls) if float(w.normal[1]) < -_SAME_WALL_DOT]
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
        if not _is_free(along - width / 2.0, along + width / 2.0, blocked):
            continue
        wall_h = _wall_height_at(house, wall.at(along / wall.length))
        for level in range(max(1, int((wall_h + _STOREY_SLACK_M) // storey_m))):
            bottom = level * storey_m + sill_m
            if bottom + height > min(wall_h, (level + 1) * storey_m) - _WINDOW_HEAD_M:
                break
            grid.append((along, bottom))
    return grid


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
    """
    lot_x, lot_y = ctx.cfg.worldgen.lot_m
    return [
        Placement(
            spec=ctx.spec,
            pos=_as3(lot.centre),
            size=np.array([lot_x, lot_y, 0.0], dtype=np.float64),
        )
        for lot in _neighbour_lots(ctx)
    ]


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
        placements.append(Placement(spec=spec, pos=_as3(centre), size=extents, yaw=yaw))
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

    placements: list[Placement] = []
    for index in range(ctx.n):
        for _attempt in range(_MAX_ATTEMPTS):
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
                wall, t = _choose_wall(ctx, clearance_m=0.0)
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
            if not _clear(xy, radius, _seen(ctx, placements)):
                continue
            placements.append(
                Placement(spec=spec, pos=_as3(xy), size=size, yaw=spec.sample_yaw(ctx.rng))
            )
            break
    return placements


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

    for _attempt in range(_MAX_ATTEMPTS):
        wall, t = _choose_wall(ctx, clearance)
        placements = _service_on(ctx, wall, t, clearance)
        if _clear_of_openings(wall, placements, gap):
            return placements

    house = ctx.require_house()
    msg = (
        f"role {ctx.role.name!r}: found no bare wall for the service assembly in "
        f"{_MAX_ATTEMPTS} draws; the house's {len(house.walls)} walls carry "
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
    unit = along / max(float(np.linalg.norm(along)), _EPS_M)
    anchor = wall.at(t)

    def mounted(tag: str, at: npt.NDArray[np.float64], bottom_m: float) -> Placement:
        spec = ctx.library.choose(tag, ctx.rng)
        return Placement(
            spec=spec,
            pos=_as3(at + wall.normal * standoff, bottom_m),
            size=ctx.extents(spec),
            yaw=_yaw_onto(spec.facing_xy(), wall.normal),
            wall_normal=_as3(wall.normal),
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
            if _crosses(centre, unit, size, ctx.launch_area):
                continue
            placements.append(
                Placement(
                    spec=spec,
                    pos=_as3(centre),
                    size=size,
                    yaw=_yaw_onto(spec.facing_xy(), np.asarray(normal, dtype=np.float64)),
                )
            )
    return placements
