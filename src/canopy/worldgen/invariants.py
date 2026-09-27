"""Hard rules every generated property must meet, checked as each role is placed.

A placement rule draws a plausible layout; an invariant says what no layout may
get wrong. The electrical service is the case that matters: a property without
a meter on a real exterior wall has no mission, and on an address-built
property the evidence can make that harder than on a random one -- a wall
shared with a neighbour is not exterior, and a side yard narrower than a
meter's working space is no place for one.

Invariants register against a *placement rule* by name, the way rules register
against roles, and :func:`place` runs them the moment that rule returns rather
than once at the end. Order is load-bearing here: bushes are placed to occlude
the meter and openings are fitted around it, so a meter moved after them would
leave both wrong. A violation is repaired by redrawing the same rule, from a
generator spawned off the property's (spawning does not advance the parent
stream, so every later role draws exactly what it would have), down a ladder
of relaxed corner clearances from ``worldgen.invariants``. The draw that
passes is marked :attr:`~canopy.contracts.Provenance.REPAIRED`.

Some rules are about the mission rather than the house -- whether INSPECT's
close-up has room in front of the meter. :func:`review` reports those into the
manifest's notes and never repairs them: real meters do sit in narrow side
yards, and redrawing every such property would quietly bias the simulation
toward easy houses.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import Cls, Provenance
from canopy.errors import SiteRejectedError, WorldgenError
from canopy.worldgen.placement import get_rule

if TYPE_CHECKING:
    from canopy.config import Config
    from canopy.worldgen.placement import HouseFrame, Placement, PlacementContext, WallSegment

__all__ = [
    "Invariant",
    "InvariantCheck",
    "Violation",
    "invariant_names",
    "place",
    "register_invariant",
    "review",
]

#: How far a wall-mounted object's contact point may sit off its wall line
#: and still be on it: a couple of centimetres of standoff plus float slack.
_ON_WALL_TOL_M = 0.05

#: Two unit normals closer than this are the same wall direction.
_SAME_WALL_DOT = 0.99

#: Step when walking a ray out from the meter to measure clear depth.
_PROBE_STEP_M = 0.05

#: Failures quoted in a final error message; the rest are the same story.
_QUOTED_FAILURES = 3


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken hard rule, with the measurement that broke it."""

    invariant: str
    detail: str
    """One sentence naming the measured value and what was required."""


#: What an invariant looks like: the rule's context and its output, in; a
#: violation, or ``None`` when the rule holds.
InvariantCheck = Callable[["PlacementContext", Sequence["Placement"]], "Violation | None"]


@dataclass(frozen=True, slots=True)
class Invariant:
    """A registered hard rule and the placement rule whose output it checks."""

    name: str
    after: str
    check: InvariantCheck


_INVARIANTS: dict[str, Invariant] = {}


def register_invariant(name: str, *, after: str) -> Callable[[InvariantCheck], InvariantCheck]:
    """Register a hard rule, checked whenever the placement rule ``after`` returns.

    Parameters
    ----------
    name
        Names the rule in violations and error messages.
    after
        Placement rule name, as registered in :mod:`canopy.worldgen.placement`.
        Keyed on the rule rather than the role so renaming a role in the model
        index cannot silently switch a check off.

    Raises
    ------
    WorldgenError
        On a duplicate name, which would otherwise silently shadow a rule.
    """

    def decorate(fn: InvariantCheck) -> InvariantCheck:
        if name in _INVARIANTS:
            msg = f"invariant {name!r} is already registered"
            raise WorldgenError(msg)
        _INVARIANTS[name] = Invariant(name=name, after=after, check=fn)
        return fn

    return decorate


def invariant_names() -> list[str]:
    """Every registered invariant name, sorted."""
    return sorted(_INVARIANTS)


def _checks_for(rule: str) -> list[Invariant]:
    return [inv for inv in _INVARIANTS.values() if inv.after == rule]


def _violations(
    ctx: PlacementContext, found: Sequence[Placement], checks: Sequence[Invariant]
) -> list[Violation]:
    return [v for inv in checks if (v := inv.check(ctx, found)) is not None]


def place(ctx: PlacementContext) -> list[Placement]:
    """Run the role's placement rule and hold its output to the rule's invariants.

    A rule with no invariants is simply called, so it draws exactly what it
    always did. Otherwise a rule that raises :class:`WorldgenError`, or whose
    output breaks an invariant, is redrawn as the module docstring describes.

    Raises
    ------
    SiteRejectedError
        On an address-built property whose house no redraw can satisfy: the
        address, not the generator, is the problem.
    WorldgenError
        The same on a seed-only property.
    """
    rule = get_rule(ctx.role.rule)
    checks = _checks_for(ctx.role.rule)
    if not checks:
        return rule(ctx)

    failures: list[str] = []
    try:
        found = rule(ctx)
    except WorldgenError as exc:
        failures.append(f"{ctx.role.rule}: {exc}")
    else:
        violations = _violations(ctx, found, checks)
        if not violations:
            return found
        failures.extend(f"{v.invariant}: {v.detail}" for v in violations)
    return _repair(ctx, checks, failures)


def _repair(
    ctx: PlacementContext, checks: Sequence[Invariant], failures: list[str]
) -> list[Placement]:
    """Redraw down the corner-clearance ladder until a draw keeps every invariant."""
    rule = get_rule(ctx.role.rule)
    ladder = ctx.cfg.worldgen.invariants.corner_clearance_ladder_m
    per_rung = ctx.cfg.worldgen.invariants.repairs_per_rung
    for clearance in ladder:
        role = dataclasses.replace(
            ctx.role, params={**ctx.role.params, "corner_clearance_m": clearance}
        )
        for _ in range(per_rung):
            attempt = dataclasses.replace(ctx, rng=ctx.rng.spawn(1)[0], role=role)
            try:
                found = rule(attempt)
            except WorldgenError as exc:
                failures.append(f"{ctx.role.rule}: {exc}")
                continue
            violations = _violations(attempt, found, checks)
            if not violations:
                return [dataclasses.replace(p, provenance=Provenance.REPAIRED) for p in found]
            failures.extend(f"{v.invariant}: {v.detail}" for v in violations)

    walls = [round(w.length, 2) for w in ctx.house.walls] if ctx.house is not None else []
    distinct = list(dict.fromkeys(failures))
    msg = (
        f"role {ctx.role.name!r} could not meet its hard rules after {per_rung} redraws at each "
        f"corner clearance in {list(ladder)} m; the house's exterior walls are {walls} m; "
        f"last failures: {'; '.join(distinct[-_QUOTED_FAILURES:])}"
    )
    if ctx.site is not None:
        raise SiteRejectedError(msg)
    raise WorldgenError(msg)


# ---------------------------------------------------------------------------
# Geometry the checks share
# ---------------------------------------------------------------------------
def _meters(found: Sequence[Placement]) -> list[Placement]:
    return [p for p in found if p.spec.cls is Cls.METER]


def _meter_wall(ctx: PlacementContext, meter: Placement) -> tuple[WallSegment, float] | None:
    """Return the exterior wall a meter hangs on and how far along it, or ``None``.

    Measured from the meter's contact point -- its position less the standoff
    along its normal -- against the house frame's walls, which on an
    address-built property no longer include walls shared with a neighbour.
    """
    if ctx.house is None or meter.wall_normal is None:
        return None
    normal = np.asarray(meter.wall_normal[:2], dtype=np.float64)
    contact = meter.pos[:2] - normal * float(ctx.param("standoff_m", default=0.0))
    for wall in ctx.house.walls:
        if float(np.dot(wall.normal, normal)) < _SAME_WALL_DOT:
            continue
        off = abs(float(np.dot(contact - wall.a, wall.normal)))
        along = wall.along(contact)
        if off <= _ON_WALL_TOL_M and -_ON_WALL_TOL_M <= along <= wall.length + _ON_WALL_TOL_M:
            return wall, along
    return None


def _clear_depth(
    origin: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
    house: HouseFrame | None,
    max_depth_m: float,
) -> float:
    """How far a ray from ``origin`` runs before it leaves ``[lo, hi]`` or enters the house.

    Walked in :data:`_PROBE_STEP_M` steps, capped at ``max_depth_m``: the
    answer only has to say whether the space in front of a meter is deep
    enough, not measure open ground to the horizon.
    """
    depths = np.arange(_PROBE_STEP_M, max_depth_m + _PROBE_STEP_M / 2.0, _PROBE_STEP_M)
    points = origin[None, :] + depths[:, None] * direction[None, :]
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    if house is not None:
        inside &= np.array([not house.contains(p) for p in points], dtype=np.bool_)
    blocked = np.flatnonzero(~inside)
    if blocked.size == 0:
        return max_depth_m
    return float(depths[blocked[0]] - _PROBE_STEP_M)


# ---------------------------------------------------------------------------
# The meter's hard rules
# ---------------------------------------------------------------------------
@register_invariant("one_meter", after="service_assembly")
def _one_meter(ctx: PlacementContext, found: Sequence[Placement]) -> Violation | None:
    """Check for exactly one electric meter: none leaves no mission, two leave no answer."""
    del ctx  # the count needs nothing but the placements
    n = len(_meters(found))
    if n == 1:
        return None
    return Violation("one_meter", f"placed {n} electric meters; every property needs exactly one")


@register_invariant("meter_on_exterior_wall", after="service_assembly")
def _meter_on_exterior_wall(ctx: PlacementContext, found: Sequence[Placement]) -> Violation | None:
    """Check the meter hangs flush on an exterior wall, not in the air or on a shared one."""
    for meter in _meters(found):
        if _meter_wall(ctx, meter) is None:
            x, y = float(meter.pos[0]), float(meter.pos[1])
            return Violation(
                "meter_on_exterior_wall",
                f"the meter at ({x:.2f}, {y:.2f}) is on no exterior wall of the house "
                "(a wall shared with a neighbour does not count)",
            )
    return None


@register_invariant("meter_height", after="service_assembly")
def _meter_height(ctx: PlacementContext, found: Sequence[Placement]) -> Violation | None:
    """Check the meter's centre sits at reading height (spec.md, ``test_worldgen``)."""
    lo, hi = ctx.cfg.worldgen.invariants.meter_centre_height_m
    for meter in _meters(found):
        centre = float(meter.pos[2]) + float(meter.size[2]) / 2.0
        if not lo <= centre <= hi:
            return Violation(
                "meter_height",
                f"the meter's centre is {centre:.2f} m above grade, outside "
                f"worldgen.invariants.meter_centre_height_m = [{lo}, {hi}]",
            )
    return None


@register_invariant("meter_corner_clearance", after="service_assembly")
def _meter_corner_clearance(ctx: PlacementContext, found: Sequence[Placement]) -> Violation | None:
    """Check the meter keeps off its wall's ends, where it would be half hidden from every side."""
    floor = ctx.cfg.worldgen.invariants.meter_min_corner_clearance_m
    for meter in _meters(found):
        if (hit := _meter_wall(ctx, meter)) is None:
            continue  # meter_on_exterior_wall reports it
        wall, along = hit
        nearest = min(along, wall.length - along)
        if nearest < floor:
            return Violation(
                "meter_corner_clearance",
                f"the meter is {nearest:.2f} m from the end of its {wall.length:.2f} m wall, "
                f"under worldgen.invariants.meter_min_corner_clearance_m = {floor}",
            )
    return None


@register_invariant("meter_working_space", after="service_assembly")
def _meter_working_space(ctx: PlacementContext, found: Sequence[Placement]) -> Violation | None:
    """Check someone can stand in front of the meter: clear depth in the lot, outside the house."""
    need = ctx.cfg.worldgen.invariants.meter_working_space_m
    lo = np.asarray(ctx.lot_bounds[0, :2], dtype=np.float64)
    hi = np.asarray(ctx.lot_bounds[1, :2], dtype=np.float64)
    for meter in _meters(found):
        if meter.wall_normal is None:
            continue
        clear = _clear_depth(meter.pos[:2], meter.wall_normal[:2], lo, hi, ctx.house, need)
        if clear < need:
            return Violation(
                "meter_working_space",
                f"only {clear:.2f} m is clear in front of the meter inside the lot, under "
                f"worldgen.invariants.meter_working_space_m = {need}",
            )
    return None


# ---------------------------------------------------------------------------
# Reported, never repaired
# ---------------------------------------------------------------------------
def review(
    house: HouseFrame,
    placed: Sequence[Placement],
    lot_bounds: npt.NDArray[np.float64],
    cfg: Config,
) -> tuple[str, ...]:
    """Check the rules about the mission rather than the house, as manifest notes.

    Today that is one: whether INSPECT's close-up has room in front of the
    meter inside the geofence (``worldgen.invariants.meter_approach_m``). A
    failure is a property the survey will struggle with, which is a finding,
    not a generation bug.
    """
    meter = next((p for p in placed if p.spec.cls is Cls.METER and not p.background), None)
    if meter is None or meter.wall_normal is None:
        return ()
    need = cfg.worldgen.invariants.meter_approach_m
    inset = cfg.safety.geofence_inset_m
    lo = np.asarray(lot_bounds[0, :2], dtype=np.float64) + inset
    hi = np.asarray(lot_bounds[1, :2], dtype=np.float64) - inset
    clear = _clear_depth(meter.pos[:2], meter.wall_normal[:2], lo, hi, house, need)
    if clear < need:
        return (
            f"meter approach: {clear:.1f} m is clear in front of the meter inside the geofence; "
            f"INSPECT's close-up needs {need:.1f} m",
        )
    return ()
