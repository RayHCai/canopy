"""Placement rules: what makes a spot on a wall a good battery site, as data.

``config/rules.yaml`` encodes the product's "Battery Space" photo-review
checklist (ADR 0013) as a list of rule entries, so the checklist can grow or
change without a solver change. Each entry names a registered :class:`Rule`
type and gives its parameters::

    rules:
      - rule: harness_run         # the registered name
        pass_max_m: 4.88          # that rule's own parameters
      - rule: clear_of
        cls: BUSH
        min_m: 0.5

Every rule does two jobs at once, over every candidate site in one call:

* **Constraint.** ``on_fail`` says what a failure means: ``"fail"`` (the
  default) rejects the site outright unless the failure is *marginal* --
  close enough that a measurement or another photo could clear it -- in
  which case the site goes to manual review instead; ``"review"`` always
  sends a failure to review, never to reject, because the checklist itself
  says "member sign-off" rather than "no"; ``"ignore"`` makes the rule a pure
  cost preference with no bearing on the verdict.
* **Factor.** Each rule also prices every site, and ``weight`` (default 1)
  scales that price into the site's cost. Sites are ranked by verdict, then
  by how many non-ignored rules they broke, then by cost.

Adding a rule takes three steps: subclass :class:`Rule` as a frozen,
keyword-only dataclass whose fields are its parameters; give it a ``name``
and implement :meth:`Rule.evaluate` and :meth:`Rule.explain`; decorate it with
:func:`register_rule`. It is then usable from ``rules.yaml`` by name, with
its parameters type-checked like the rest of the config.

Rules see only what the swarm found -- the traced house and the discovered
objects in a :class:`SiteContext` -- never the scene manifest.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt

from canopy.config import build_section, load_rules
from canopy.contracts import Cls, DiscoveredObject, MapState, Occ, Points
from canopy.errors import ConfigError
from canopy.site.geometry import Boxes, box_gap, point_gap
from canopy.site.outline import HouseOutline

__all__ = [
    "BatterySpec",
    "Candidates",
    "ClearOf",
    "Facing",
    "FreeSpace",
    "HarnessRoute",
    "HarnessRun",
    "Headroom",
    "OutlineSpec",
    "PlacementSpec",
    "Rule",
    "RuleOutcome",
    "SiteContext",
    "SiteRules",
    "build_rule",
    "load_site_rules",
    "parse_site_rules",
    "register_rule",
    "rule_names",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_R = TypeVar("_R", bound="type[Rule]")

#: Metres per foot, for warnings a US installer reads.
_M_PER_FT = 0.3048

#: Half a turn, in degrees.
_HALF_TURN_DEG = 180.0

#: Top-level keys of ``rules.yaml``.
_SECTIONS = ("battery", "placement", "outline", "rules")

#: Valid values of :attr:`Rule.on_fail`.
_ON_FAIL = ("fail", "review", "ignore")


# ---------------------------------------------------------------------------
# The fixed sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BatterySpec:
    """The battery's bounding box, which is what every clearance is measured from."""

    width_m: float
    """Along the wall."""
    depth_m: float
    """Out from the wall."""
    height_m: float

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless every extent is positive."""
        for name in ("width_m", "depth_m", "height_m"):
            if not getattr(self, name) > 0.0:
                msg = f"rules.yaml battery.{name} must be positive, got {getattr(self, name)}"
                raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    """How candidate sites are laid out along the walls and how many are kept."""

    step_m: float
    """Spacing of candidate sites along each wall."""
    corner_margin_m: float
    """Gap kept between a battery's side and the end of its wall."""
    top_k: int
    """Most sites offered."""
    min_separation_m: float
    """Least distance between two offered sites, so the three are three options."""
    fill_with_flagged: bool
    """Top up to ``top_k`` with non-PASS sites when too few sites pass cleanly.
    Non-PASS sites are always offered when *none* pass."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` on a non-positive step or count."""
        if not self.step_m > 0.0:
            msg = f"rules.yaml placement.step_m must be positive, got {self.step_m}"
            raise ConfigError(msg)
        if self.top_k < 1:
            msg = f"rules.yaml placement.top_k must be at least 1, got {self.top_k}"
            raise ConfigError(msg)
        if self.corner_margin_m < 0.0 or self.min_separation_m < 0.0:
            msg = (
                "rules.yaml placement.corner_margin_m and min_separation_m must not be "
                f"negative, got {self.corner_margin_m} and {self.min_separation_m}"
            )
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class OutlineSpec:
    """How the house's walls are traced from occupancy (see :mod:`canopy.site.outline`)."""

    band_z_m: tuple[float, float]
    """Height band a wall must fill. Above what stands against a wall, below the eaves."""
    min_fill: float
    """Fraction of the band's voxels that must be occupied for a column to be wall."""
    close_m: float
    """Gaps in the traced walls up to about twice this wide are bridged."""
    simplify_m: float
    """Largest deviation of the wall polygon from the traced boundary."""
    max_meter_gap_m: float
    """The meter must be within this of a mapped wall."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` on an empty band or an out-of-range fraction."""
        lo, hi = self.band_z_m
        if not lo < hi:
            msg = f"rules.yaml outline.band_z_m must be [low, high], got {list(self.band_z_m)}"
            raise ConfigError(msg)
        if not 0.0 < self.min_fill <= 1.0:
            msg = f"rules.yaml outline.min_fill must be in (0, 1], got {self.min_fill}"
            raise ConfigError(msg)


# ---------------------------------------------------------------------------
# What a rule is handed, and what it returns
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class Candidates:
    """Every candidate battery site, as parallel arrays of length ``N``.

    A site is a battery standing on the ground with its back against a wall:
    ``anchor`` is the middle of its back edge, on the wall.
    """

    anchor: Points
    """Plan position on the wall, shape ``(N, 2)``."""
    normal: Points
    """The wall's outward unit normal, shape ``(N, 2)``; the battery faces this way."""
    wall: npt.NDArray[np.int64]
    """Index into :attr:`HouseOutline.walls`, shape ``(N,)``."""
    battery: BatterySpec

    def __len__(self) -> int:
        """Return the number of candidates."""
        return len(self.anchor)

    @property
    def tangent(self) -> Points:
        """Unit direction along the wall, anticlockwise around the house, ``(N, 2)``."""
        return np.stack([-self.normal[:, 1], self.normal[:, 0]], axis=1)

    def boxes(self) -> Boxes:
        """Return the battery's bounding box at every site."""
        b = self.battery
        n = len(self)
        return Boxes(
            centre=self.anchor + self.normal * (b.depth_m / 2.0),
            axis=self.tangent,
            half=np.tile([b.width_m / 2.0, b.depth_m / 2.0], (n, 1)),
            z=np.tile([0.0, b.height_m], (n, 1)),
        )


@dataclass(frozen=True, slots=True, eq=False)
class SiteContext:
    """What the swarm found, as the rules may see it."""

    meter: DiscoveredObject
    house: HouseOutline
    objects: tuple[DiscoveredObject, ...]
    """Every discovered object, the meter included."""
    state: MapState
    """The whole map, for rules that read occupancy rather than objects."""

    def of_class(self, cls: Cls) -> tuple[DiscoveredObject, ...]:
        """Return the discovered objects of one class."""
        return tuple(o for o in self.objects if o.cls is cls)


@dataclass(frozen=True, slots=True, eq=False)
class RuleOutcome:
    """One rule's verdict on every candidate, as arrays of shape ``(N,)``."""

    measure: FloatArray
    """The quantity the rule judges, in its own unit (usually metres). Reported
    in each site's breakdown and handed to :meth:`Rule.explain`."""
    passed: BoolArray
    cost: FloatArray
    """Unweighted price, non-negative; lower is better."""
    marginal: BoolArray | None = None
    """Sites that are not a clean pass but could be cleared by a measurement or
    another photo. Always a subset of ``~passed`` -- :func:`canopy.site.solver.assess_site`
    enforces that by masking with it, so a rule need not get the edge cases
    exactly right itself. ``None`` (the default) means no site of this rule's
    ever qualifies as marginal."""


@dataclass(frozen=True, kw_only=True)
class Rule(ABC):
    """One placement rule. Subclasses add their parameters as fields.

    Not slotted: a slotted dataclass is rebuilt as a new class, which breaks
    the zero-argument ``super()`` a subclass's ``validate`` wants to call.
    """

    name: ClassVar[str]
    """Registered name, used as ``rule:`` in ``rules.yaml``."""

    on_fail: str = "fail"
    """What a failure of this rule means for the site's verdict:
    ``"fail"`` rejects unless the failure is marginal (then it goes to
    review); ``"review"`` always goes to review, never reject; ``"ignore"``
    never affects the verdict, only the cost."""
    weight: float = 1.0
    """Multiplier on :attr:`RuleOutcome.cost` in a site's total cost."""

    @property
    def key(self) -> str:
        """Unique label for this entry in a site's breakdown."""
        return self.name

    def validate(self, where: str) -> None:
        """Raise :class:`ConfigError` if a parameter is out of range.

        Overrides call this first, via ``super().validate(where)``.
        """
        if self.on_fail not in _ON_FAIL:
            msg = f"{where}: on_fail must be one of {_ON_FAIL}, got {self.on_fail!r}"
            raise ConfigError(msg)
        if not math.isfinite(self.weight) or self.weight < 0.0:
            msg = f"{where}: weight must be finite and not negative, got {self.weight}"
            raise ConfigError(msg)

    @abstractmethod
    def evaluate(self, candidates: Candidates, context: SiteContext) -> RuleOutcome:
        """Judge every candidate at once."""

    @abstractmethod
    def explain(self, measure: float, *, marginal: bool) -> str:
        """Say, in one sentence for a reviewer, how a site with ``measure`` fails.

        ``marginal`` says whether the call needs confirming (a measurement or
        another photo could still clear it) rather than being a clean break.
        """


_REGISTRY: dict[str, type[Rule]] = {}


def register_rule(cls: _R) -> _R:
    """Make a :class:`Rule` subclass usable from ``rules.yaml`` by its ``name``.

    Raises
    ------
    ConfigError
        If another class already holds the name.
    """
    name = cls.name
    if (held := _REGISTRY.get(name)) is not None and held is not cls:
        msg = f"rule name {name!r} is already registered to {held.__qualname__}"
        raise ConfigError(msg)
    _REGISTRY[name] = cls
    return cls


def rule_names() -> list[str]:
    """Every registered rule name, sorted."""
    return sorted(_REGISTRY)


def build_rule(entry: Any, where: str) -> Rule:
    """Build and validate one ``rules:`` entry.

    Raises
    ------
    ConfigError
        If the entry names no registered rule, or its parameters do not fit it.
    """
    if not isinstance(entry, dict):
        msg = f"{where}: expected a mapping, got {type(entry).__name__}"
        raise ConfigError(msg)
    name = entry.get("rule")
    if name not in _REGISTRY:
        msg = f"{where}: rule {name!r} is not registered; known rules are {rule_names()}"
        raise ConfigError(msg)
    params = {k: v for k, v in entry.items() if k != "rule"}
    rule = build_section(_REGISTRY[name], params, f"{where} ({name})")
    rule.validate(f"{where} ({name})")
    return rule


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _length(m: float) -> str:
    """Format a distance in metres and feet."""
    return f"{m:.2f} m ({m / _M_PER_FT:.1f} ft)"


def _grid(lo: float, hi: float, step: float) -> FloatArray:
    """Evenly spaced samples covering ``[lo, hi]`` no further apart than ``step``."""
    n = max(int(np.ceil((hi - lo) / step)), 0) + 1
    return np.linspace(lo, hi, n)


def _occ_at(state: MapState, plan: FloatArray, z: FloatArray) -> npt.NDArray[np.int64]:
    """The occupancy state at each point; points off the mapped grid read FREE."""
    points = np.concatenate([plan, z[..., None]], axis=-1)
    idx = np.floor((points - np.asarray(state.origin)) / state.voxel).astype(np.int64)
    shape = np.array(state.occ.shape)
    inside = ((idx >= 0) & (idx < shape)).all(axis=-1)
    clipped = np.clip(idx, 0, shape - 1)
    value = state.occ[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
    result: npt.NDArray[np.int64] = np.where(inside, value, int(Occ.FREE))
    return result


def _sample_region(
    candidates: Candidates, state: MapState, across: FloatArray, out: FloatArray, up: FloatArray
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64], Points]:
    """Sample occupancy on a grid in every site's own (across, out, up) frame.

    Shared by :class:`FreeSpace` and :class:`Headroom`, which differ only in
    the extent of the region they sample -- the sampling and the occupancy
    lookup are otherwise identical.

    Returns
    -------
    tuple
        ``(o, z, occ, plan)``: the out-from-wall and height coordinate of
        every sample (``(S,)`` each, ``S = len(across) * len(out) * len(up)``),
        the occupancy at every site's every sample (``(N, S)``), and the plan
        position of every site's every sample (``(N, S, 2)``), site-major.
    """
    a, o, z = (g.ravel() for g in np.meshgrid(across, out, up, indexing="ij"))
    plan = (
        candidates.anchor[:, None, :]
        + a[None, :, None] * candidates.tangent[:, None, :]
        + o[None, :, None] * candidates.normal[:, None, :]
    )
    occ = _occ_at(state, plan, np.broadcast_to(z, plan.shape[:2]))
    return o, z, occ, plan


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class HarnessRoute:
    """Which way round the walls the harness runs from the meter to each candidate.

    Computed once, by :meth:`HarnessRun.route`, and used both to judge the
    rule and (:mod:`canopy.site.solver`) to draw the conduit -- so the two
    never disagree about which wall the cable follows.
    """

    length: FloatArray
    """Metres from the meter to the battery's nearer side, plus the vertical
    drop to the battery's height; ``inf`` where both directions are blocked."""
    forward: BoolArray
    """Whether the chosen direction is increasing arc coordinate (anticlockwise)."""
    crosses_review: BoolArray
    """Whether the chosen route crosses a ``review_block`` object."""
    both_blocked: BoolArray
    """Whether every direction is blocked by a ``hard_block`` object."""
    s_meter: float
    """The meter's arc coordinate on the outline."""
    s_candidate: FloatArray
    """Each candidate's arc coordinate, shape ``(N,)``."""


def _crosses_path(
    house: HouseOutline,
    objects: Sequence[DiscoveredObject],
    s_from: float,
    path_len: FloatArray,
    *,
    forward: bool,
    wall_tol_m: float,
) -> BoolArray:
    """Whether any of ``objects`` sits on the wall between ``s_from`` and the far end of each path.

    An object only counts when its footprint is within ``wall_tol_m`` of the
    outline (mounted on this wall, not merely standing near it) and its own
    arc interval overlaps the travelled stretch.
    """
    n = len(path_len)
    if not objects:
        return np.zeros(n, dtype=bool)
    boxes = Boxes.from_oriented([o.box for o in objects])
    _, _, centre_d = house.arc_of(boxes.centre)
    near_wall = centre_d <= wall_tol_m
    corners = boxes.corners().reshape(-1, 2)
    _, corner_s, _ = house.arc_of(corners)
    corner_s = corner_s.reshape(len(objects), 4)
    lo, hi = corner_s.min(axis=1), corner_s.max(axis=1)
    perimeter = house.perimeter
    if forward:
        off_lo = np.mod(lo - s_from, perimeter)
        off_hi = np.mod(hi - s_from, perimeter)
    else:
        off_lo = np.mod(s_from - hi, perimeter)
        off_hi = np.mod(s_from - lo, perimeter)
    # Normally off_lo <= off_hi; when it isn't, the interval wraps the cut
    # point at s_from, i.e. the object surrounds the reference point itself.
    wrapped = off_hi < off_lo
    on_path = (wrapped | (np.minimum(off_lo, off_hi) <= path_len[:, None])) & near_wall[None, :]
    result: BoolArray = on_path.any(axis=1)
    return result


@register_rule
@dataclass(frozen=True, kw_only=True)
class HarnessRun(Rule):
    """Keep the harness -- the cable fastened to the wall from meter to battery -- short.

    Measured along the traced outline, not straight-line, because that is
    the cable's actual path. Both directions round the house are considered;
    the shorter one that avoids a ``hard_block`` class (a garage door: no
    route exists across its track) wins, with a preference for one that also
    avoids a ``review_block`` class (a door: occasionally acceptable, but
    always worth a second look) when that is possible without exceeding
    ``max_m``.
    """

    name: ClassVar[str] = "harness_run"
    pass_max_m: float
    """Clean-pass limit."""
    max_m: float
    """Beyond this, no photo or measurement saves the site."""
    hard_block: tuple[str, ...] = ()
    """Classes that make a route impassable, e.g. ``GARAGE_DOOR``."""
    review_block: tuple[str, ...] = ()
    """Classes that make a route passable but worth a second look, e.g. ``DOOR``."""
    wall_tol_m: float = 0.3
    """How close a blocking object's footprint must be to the outline to count
    as sitting on the wall the harness runs along."""

    def validate(self, where: str) -> None:
        """Raise on a non-positive or inverted limit, or an unknown block class."""
        super().validate(where)
        if not 0.0 < self.pass_max_m <= self.max_m:
            msg = f"{where}: need 0 < pass_max_m <= max_m, got {self.pass_max_m} and {self.max_m}"
            raise ConfigError(msg)
        if self.wall_tol_m < 0.0:
            msg = f"{where}: wall_tol_m must not be negative, got {self.wall_tol_m}"
            raise ConfigError(msg)
        for field_name, classes in (("hard_block", self.hard_block), ("review_block", self.review_block)):
            if bad := [c for c in classes if c not in Cls.__members__]:
                msg = f"{where}: {field_name} {bad} are not classes; expected {list(Cls.__members__)}"
                raise ConfigError(msg)

    def route(self, candidates: Candidates, context: SiteContext) -> HarnessRoute:
        """Choose a direction round the walls for every candidate, and its length."""
        house = context.house
        perimeter = house.perimeter
        s_meter = float(house.arc_of(context.meter.box.center[None, :2])[1][0])
        _, s_candidate, _ = house.arc_of(candidates.anchor)
        width = candidates.battery.width_m
        drop = max(float(context.meter.box.center[2]) - candidates.battery.height_m, 0.0)

        raw_fwd = np.mod(s_candidate - s_meter, perimeter)
        raw_back = np.mod(s_meter - s_candidate, perimeter)
        len_fwd = np.maximum(raw_fwd - width / 2.0, 0.0) + drop
        len_back = np.maximum(raw_back - width / 2.0, 0.0) + drop

        hard = [o for c in self.hard_block for o in context.of_class(Cls[c])]
        review = [o for c in self.review_block for o in context.of_class(Cls[c])]
        blocked_fwd = _crosses_path(
            house, hard, s_meter, raw_fwd, forward=True, wall_tol_m=self.wall_tol_m
        )
        blocked_back = _crosses_path(
            house, hard, s_meter, raw_back, forward=False, wall_tol_m=self.wall_tol_m
        )
        review_fwd = _crosses_path(
            house, review, s_meter, raw_fwd, forward=True, wall_tol_m=self.wall_tol_m
        )
        review_back = _crosses_path(
            house, review, s_meter, raw_back, forward=False, wall_tol_m=self.wall_tol_m
        )

        length = np.stack([len_fwd, len_back], axis=1)
        usable = ~np.stack([blocked_fwd, blocked_back], axis=1)
        review_cross = np.stack([review_fwd, review_back], axis=1)

        clean = usable & ~review_cross & (length <= self.max_m)
        use_clean = clean.any(axis=1)
        idx = np.where(
            use_clean,
            np.argmin(np.where(clean, length, np.inf), axis=1),
            np.argmin(np.where(usable, length, np.inf), axis=1),
        )

        rows = np.arange(len(candidates))
        both_blocked = ~usable.any(axis=1)
        chosen_length = np.where(both_blocked, np.inf, length[rows, idx])
        chosen_review = review_cross[rows, idx] & ~both_blocked
        return HarnessRoute(
            length=chosen_length,
            forward=idx == 0,
            crosses_review=chosen_review,
            both_blocked=both_blocked,
            s_meter=s_meter,
            s_candidate=s_candidate,
        )

    def evaluate(self, candidates: Candidates, context: SiteContext) -> RuleOutcome:
        """Judge every candidate by its chosen route's length."""
        route = self.route(candidates, context)
        passed = (route.length <= self.pass_max_m) & ~route.crosses_review
        marginal = ~passed & (route.length <= self.max_m) & ~route.both_blocked
        return RuleOutcome(measure=route.length, passed=passed, marginal=marginal, cost=route.length)

    def explain(self, measure: float, *, marginal: bool) -> str:
        """Distinguish an impassable route from a long one from a door crossing."""
        if not math.isfinite(measure):
            blockers = " or a ".join(c.lower().replace("_", " ") for c in self.hard_block)
            return f"No wall route to it that avoids a {blockers}" if blockers else "No wall route to it"
        if not marginal:
            return f"{_length(measure)} of harness run exceeds the {_length(self.max_m)} limit"
        if measure > self.pass_max_m:
            return (
                f"{_length(measure)} of harness run is over the preferred {_length(self.pass_max_m)}; "
                "confirm the run before approving"
            )
        return "The shortest run crosses a door; confirm it is an acceptable path for the harness"


@register_rule
@dataclass(frozen=True, kw_only=True)
class ClearOf(Rule):
    """Keep clearance from every discovered object of one class.

    By default the gap is measured in three dimensions, so an object above
    or below the battery does not count against it; ``plan`` drops the
    height term for checks that are really about footprint, such as not
    sitting under a window. ``below_m`` further restricts which objects of
    the class count, by the height of their own box's bottom (a ground-floor
    window, ignoring the ones on an upper storey). ``swing`` grows each
    object's box into the arc a door or gate swings through -- and the
    doorway standing in front of it -- along its short horizontal axis, by
    its long horizontal extent, both sides. A class the swarm found none of
    passes everywhere. Below ``prefer_m`` (default ``min_m``) the shortfall
    is the cost, which pushes compliant sites further off than the bare
    minimum.
    """

    name: ClassVar[str] = "clear_of"
    cls: str
    """Name of a :class:`~canopy.contracts.Cls` member, e.g. ``BUSH``."""
    min_m: float
    prefer_m: float | None = None
    plan: bool = False
    review_m: float | None = None
    """Gaps in ``[min_m, review_m)`` are marginal rather than a clean pass."""
    below_m: float | None = None
    swing: bool = False

    @property
    def key(self) -> str:
        """``clear_of_<class>``, so one entry per class stays distinguishable."""
        return f"{self.name}_{self.cls.lower()}"

    def validate(self, where: str) -> None:
        """Raise on an unknown class or a negative, inverted or unordered clearance."""
        super().validate(where)
        if self.cls not in Cls.__members__:
            msg = (
                f"{where}: cls {self.cls!r} is not a class; expected one of {list(Cls.__members__)}"
            )
            raise ConfigError(msg)
        if self.min_m < 0.0 or (self.prefer_m is not None and self.prefer_m < self.min_m):
            msg = f"{where}: need 0 <= min_m <= prefer_m, got {self.min_m} and {self.prefer_m}"
            raise ConfigError(msg)
        if self.review_m is not None and self.review_m < self.min_m:
            msg = f"{where}: need min_m <= review_m, got {self.min_m} and {self.review_m}"
            raise ConfigError(msg)

    def _objects(self, context: SiteContext) -> tuple[DiscoveredObject, ...]:
        """Discovered objects of this rule's class, filtered by ``below_m`` if set."""
        found = context.of_class(Cls[self.cls])
        if self.below_m is None:
            return found
        return tuple(o for o in found if o.box.center[2] - o.box.size[2] / 2.0 < self.below_m)

    def _boxes(self, found: tuple[DiscoveredObject, ...]) -> Boxes:
        """Batch ``found`` into boxes, grown for ``swing`` if set."""
        boxes = Boxes.from_oriented([o.box for o in found])
        if not self.swing:
            return boxes
        long_extent = 2.0 * boxes.half.max(axis=1)
        short_is_across = boxes.half[:, 0] >= boxes.half[:, 1]
        grown = boxes.half.copy()
        grown[short_is_across, 1] += long_extent[short_is_across]
        grown[~short_is_across, 0] += long_extent[~short_is_across]
        return Boxes(centre=boxes.centre, axis=boxes.axis, half=grown, z=boxes.z)

    def evaluate(self, candidates: Candidates, context: SiteContext) -> RuleOutcome:
        """Gap from each site to the nearest object of the class."""
        found = self._objects(context)
        if not found:
            gap = np.full(len(candidates), np.inf)
        else:
            gap = box_gap(candidates.boxes(), self._boxes(found), plan=self.plan).min(axis=1)
        threshold = self.min_m if self.review_m is None else self.review_m
        passed = gap >= threshold
        marginal = None if self.review_m is None else (~passed & (gap >= self.min_m))
        prefer = self.min_m if self.prefer_m is None else self.prefer_m
        return RuleOutcome(
            measure=gap, passed=passed, marginal=marginal, cost=np.maximum(prefer - gap, 0.0)
        )

    def explain(self, measure: float, *, marginal: bool) -> str:
        """Name the gap and the clearance it needs."""
        noun = self.cls.lower().replace("_", " ")
        if marginal:
            review = self.review_m if self.review_m is not None else self.min_m
            return (
                f"Only {_length(measure)} from a {noun}; {_length(review)} is preferred, "
                "confirm this is workable"
            )
        if measure <= 0.0:
            return f"Touches a {noun}; it needs {_length(self.min_m)} clear"
        return f"Only {_length(measure)} from a {noun}; it needs {_length(self.min_m)} clear"


@register_rule
@dataclass(frozen=True, kw_only=True)
class Facing(Rule):
    """Avoid walls that face a compass bearing, e.g. south, into the afternoon sun.

    A site on a wall whose outward normal lies within ``within_deg`` of
    ``avoid_deg`` (degrees clockwise from north, +Y) fails and costs 1; any
    other site costs 0. The measure is the angle between the two.
    """

    name: ClassVar[str] = "facing"
    avoid_deg: float
    within_deg: float = 45.0

    def validate(self, where: str) -> None:
        """Raise unless ``within_deg`` is in ``[0, 180]``."""
        super().validate(where)
        if not 0.0 <= self.within_deg <= _HALF_TURN_DEG:
            msg = f"{where}: within_deg must be in [0, 180], got {self.within_deg}"
            raise ConfigError(msg)

    def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
        """Angle between each site's facing and the avoided bearing."""
        bearing = np.degrees(np.arctan2(candidates.normal[:, 0], candidates.normal[:, 1]))
        off = np.abs(
            (bearing - self.avoid_deg + _HALF_TURN_DEG) % (2 * _HALF_TURN_DEG) - _HALF_TURN_DEG
        )
        facing = off <= self.within_deg
        return RuleOutcome(measure=off, passed=~facing, cost=facing.astype(np.float64))

    def explain(self, measure: float, *, marginal: bool) -> str:
        """Name the bearing the wall faces.

        ``facing`` has no marginal band of its own -- it is ``on_fail:
        review`` in the shipped rules, so every failure already goes to
        review regardless.
        """
        del marginal
        return f"Faces within {measure:.0f} degrees of bearing {self.avoid_deg:.0f}"


@register_rule
@dataclass(frozen=True, kw_only=True)
class FreeSpace(Rule):
    """Keep the working alley in front of the battery free, and free of surprises.

    Reads occupancy rather than perception, so it sees what no detector class
    covers -- an AC unit, a gas meter, a fence, a shed -- at the price of not
    knowing what it saw. Points on a ``step_m`` grid fill the region: across
    the battery's width plus ``side_m`` of margin on each side, from
    ``wall_gap_m`` off the wall (closer in are the wall's own voxels, and
    anything flush on it) out to ``review_reach_m``, and from ``ground_m``
    (above the ground's own voxels) up to the battery's height. Points inside
    a discovered object of an ``except_cls`` class are skipped, leaving that
    class to a ``clear_of`` rule that can name it.

    The measure is how far out from the wall the nearest occupied point is
    (infinite when there is none) -- the prompt's front clearance. A clean
    pass needs the whole alley free out to ``review_reach_m`` *and* fully
    mapped: an alley nobody has flown down is not one you can approve.
    """

    name: ClassVar[str] = "free_space"
    min_reach_m: float
    """Below this, something is standing where the technician needs to stand."""
    review_reach_m: float
    """The full working alley; short of this by a little is a borderline call."""
    side_m: float = 0.06
    wall_gap_m: float = 0.3
    ground_m: float = 0.3
    step_m: float = 0.1
    except_cls: tuple[str, ...] = ()

    def validate(self, where: str) -> None:
        """Raise on a negative extent, an unordered reach, a non-positive step, or an unknown class."""
        super().validate(where)
        if min(self.side_m, self.wall_gap_m, self.ground_m) < 0.0 or not self.step_m > 0.0:
            msg = (
                f"{where}: side_m, wall_gap_m and ground_m must not be negative and step_m "
                f"must be positive, got {self.side_m}, {self.wall_gap_m}, {self.ground_m}, "
                f"{self.step_m}"
            )
            raise ConfigError(msg)
        if not 0.0 < self.min_reach_m <= self.review_reach_m:
            msg = (
                f"{where}: need 0 < min_reach_m <= review_reach_m, got {self.min_reach_m} "
                f"and {self.review_reach_m}"
            )
            raise ConfigError(msg)
        if bad := [c for c in self.except_cls if c not in Cls.__members__]:
            msg = f"{where}: except_cls {bad} are not classes; expected {list(Cls.__members__)}"
            raise ConfigError(msg)

    def evaluate(self, candidates: Candidates, context: SiteContext) -> RuleOutcome:
        """Distance out from the wall to the nearest occupied point in each site's alley."""
        b = candidates.battery
        half_across = b.width_m / 2.0 + self.side_m
        across = _grid(-half_across, half_across, self.step_m)
        out = _grid(self.wall_gap_m, self.review_reach_m, self.step_m)
        up = _grid(self.ground_m, b.height_m, self.step_m)
        o, z, occ, plan = _sample_region(candidates, context.state, across, out, up)
        occupied = occ == Occ.OCC
        unknown = (occ == Occ.UNKNOWN).any(axis=1)

        excepted = [obj for c in self.except_cls for obj in context.of_class(Cls[c])]
        if excepted:
            boxes = Boxes.from_oriented([obj.box for obj in excepted])
            # (B, S): which samples' heights each box spans; tiled per site to
            # match the site-major order of the flattened plan points.
            level = (z[None, :] >= boxes.z[:, :1]) & (z[None, :] <= boxes.z[:, 1:])
            inside = (point_gap(plan.reshape(-1, 2), boxes) <= 0.0) & np.tile(
                level, (1, len(candidates))
            )
            occupied &= ~inside.any(axis=0).reshape(occupied.shape)

        nearest = np.where(occupied, np.broadcast_to(o[None, :], occupied.shape), np.inf).min(
            axis=1
        )
        passed = (nearest >= self.review_reach_m) & ~unknown
        marginal = ~passed & (nearest >= self.min_reach_m)
        return RuleOutcome(
            measure=nearest,
            passed=passed,
            marginal=marginal,
            cost=np.maximum(self.review_reach_m - nearest, 0.0),
        )

    def explain(self, measure: float, *, marginal: bool) -> str:
        """Say whether the alley is unmapped, borderline, or blocked."""
        if marginal and not math.isfinite(measure):
            return "The space in front of it hasn't been mapped; confirm it's clear before approving"
        if marginal:
            return (
                f"Only {_length(measure)} clear in front of it, short of the preferred "
                f"{_length(self.review_reach_m)}; confirm it's workable"
            )
        if measure <= self.wall_gap_m + self.step_m:
            return (
                f"Something is mapped against the wall where it would stand "
                f"({_length(self.review_reach_m)} clear needed)"
            )
        return (
            f"Something is mapped {_length(measure)} out from the wall; it needs "
            f"{_length(self.review_reach_m)} clear"
        )


@register_rule
@dataclass(frozen=True, kw_only=True)
class Headroom(Rule):
    """Keep the air above the battery clear up to service height.

    A technician needs room to lift the battery clear of its mount; the
    checklist wants headroom to ``clear_m``. Samples the battery's own
    footprint (no side allowance -- that is :class:`FreeSpace`'s job) from
    its height up to ``clear_m``. Something actually mapped there fails
    outright; unmapped space is only marginal, since it might be clear but
    nobody has looked.
    """

    name: ClassVar[str] = "headroom"
    clear_m: float
    wall_gap_m: float = 0.3
    step_m: float = 0.1

    def validate(self, where: str) -> None:
        """Raise on a non-positive step or a negative wall gap."""
        super().validate(where)
        if not self.step_m > 0.0:
            msg = f"{where}: step_m must be positive, got {self.step_m}"
            raise ConfigError(msg)
        if self.wall_gap_m < 0.0:
            msg = f"{where}: wall_gap_m must not be negative, got {self.wall_gap_m}"
            raise ConfigError(msg)

    def evaluate(self, candidates: Candidates, context: SiteContext) -> RuleOutcome:
        """Height of the lowest thing (mapped or unmapped) above the battery."""
        b = candidates.battery
        if self.clear_m <= b.height_m:
            msg = f"headroom.clear_m ({self.clear_m}) must exceed the battery height ({b.height_m})"
            raise ConfigError(msg)
        across = _grid(-b.width_m / 2.0, b.width_m / 2.0, self.step_m)
        out = _grid(self.wall_gap_m, b.depth_m, self.step_m)
        up = _grid(b.height_m, self.clear_m, self.step_m)
        _, z, occ, _ = _sample_region(candidates, context.state, across, out, up)
        z_broadcast = np.broadcast_to(z[None, :], occ.shape)
        blocked_z = np.where(occ == Occ.OCC, z_broadcast, np.inf).min(axis=1)
        unknown_z = np.where(occ == Occ.UNKNOWN, z_broadcast, np.inf).min(axis=1)
        fail = np.isfinite(blocked_z)
        marginal = ~fail & np.isfinite(unknown_z)
        measure = np.minimum(blocked_z, unknown_z)
        return RuleOutcome(
            measure=measure,
            passed=~fail & ~marginal,
            marginal=marginal,
            cost=np.maximum(self.clear_m - np.minimum(measure, self.clear_m), 0.0),
        )

    def explain(self, measure: float, *, marginal: bool) -> str:
        """Say whether the headroom is unmapped or actually blocked."""
        del measure
        if marginal:
            return (
                f"Space above the battery up to {_length(self.clear_m)} hasn't been mapped; "
                "confirm it's clear before approving"
            )
        return f"Something is mapped above the battery, blocking headroom to {_length(self.clear_m)}"


# ---------------------------------------------------------------------------
# rules.yaml
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class SiteRules:
    """The whole of ``rules.yaml``, validated."""

    battery: BatterySpec
    placement: PlacementSpec
    outline: OutlineSpec
    rules: tuple[Rule, ...]


def parse_site_rules(raw: Any) -> SiteRules:
    """Validate a parsed ``rules.yaml`` mapping.

    Raises
    ------
    ConfigError
        On a missing or unknown section, an unregistered rule, a parameter
        that does not fit its rule, or two entries with the same breakdown key.
    """
    if not isinstance(raw, dict):
        msg = f"rules.yaml must hold a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    if unknown := sorted(set(raw) - set(_SECTIONS)):
        msg = f"rules.yaml: unknown key(s) {unknown}; known keys are {list(_SECTIONS)}"
        raise ConfigError(msg)
    if missing := [s for s in _SECTIONS if s not in raw]:
        msg = f"rules.yaml: missing key(s) {missing}"
        raise ConfigError(msg)

    battery = build_section(BatterySpec, raw["battery"], "rules.yaml battery")
    placement = build_section(PlacementSpec, raw["placement"], "rules.yaml placement")
    outline = build_section(OutlineSpec, raw["outline"], "rules.yaml outline")
    for section in (battery, placement, outline):
        section.validate()

    entries = raw["rules"]
    if not isinstance(entries, list):
        msg = f"rules.yaml rules: expected a list, got {type(entries).__name__}"
        raise ConfigError(msg)
    rules = tuple(build_rule(e, f"rules.yaml rules[{i}]") for i, e in enumerate(entries))
    keys = [r.key for r in rules]
    if dupes := sorted({k for k in keys if keys.count(k) > 1}):
        msg = f"rules.yaml rules: {dupes} appear more than once"
        raise ConfigError(msg)
    return SiteRules(battery=battery, placement=placement, outline=outline, rules=rules)


def load_site_rules(path: Path | str | None = None) -> SiteRules:
    """Load and validate ``rules.yaml`` (the shipped one by default)."""
    return parse_site_rules(load_rules(path))
