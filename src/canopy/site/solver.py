"""Battery siting: lay candidate sites along the walls, judge them, keep the best few.

The search space is small and exact rather than a cost raster: a battery
hangs on a wall, so the only freedom is *where along which wall*. Candidates
are laid every ``placement.step_m`` along each traced wall long enough to take
the battery with ``placement.corner_margin_m`` to spare at both ends -- which
is what "along the edge of the house" means, by construction -- and every
rule in ``rules.yaml`` judges all of them in one vectorised call.

Each site gets its own verdict: REJECT if any ``on_fail: fail`` rule fails
outright, MANUAL_REVIEW if any non-ignored rule fails only marginally (or is
an ``on_fail: review`` rule), PASS otherwise.
Sites are ranked by that verdict first, then by how many non-ignored rules
they broke, then by cost. The top ``placement.top_k`` are picked greedily,
skipping any site within ``placement.min_separation_m`` of one already
picked, so three suggestions are three different options rather than one
spot nudged sideways. The overall call (:func:`assess_site`) looks at every
candidate ever scored, not just the ones offered: PASS if any of them
passes, REJECT only if every one of them does, MANUAL_REVIEW otherwise -- a
single marginal spot anywhere is enough to keep a REJECT off the table,
because rejecting a customer outright is the expensive mistake to make.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from canopy.contracts import (
    Cls,
    DiscoveredObject,
    MapState,
    Points,
    SiteAssessment,
    SiteCandidate,
    SiteVerdict,
)
from canopy.errors import SiteError
from canopy.site.outline import HouseOutline, trace_house
from canopy.site.rules import (
    BatterySpec,
    Candidates,
    FreeSpace,
    HarnessRoute,
    HarnessRun,
    PlacementSpec,
    RuleOutcome,
    SiteContext,
    SiteRules,
)

__all__ = ["assess_site", "find_meter", "wall_candidates"]

#: Metres per foot, for the justification's ft figures.
_M_PER_FT = 0.3048

#: Ranking order for a verdict: PASS sorts first, REJECT last.
_VERDICT_RANK = {SiteVerdict.PASS: 0, SiteVerdict.MANUAL_REVIEW: 1, SiteVerdict.REJECT: 2}

#: Most failing-rule reasons named in a REJECT justification.
_TOP_REASONS = 2


def find_meter(state: MapState) -> DiscoveredObject:
    """Return the discovered meter the battery serves: the most confident, if several.

    Raises
    ------
    SiteError
        If no meter was discovered.
    """
    meters = [o for o in state.discovered.values() if o.cls is Cls.METER]
    if not meters:
        msg = "no meter was discovered, so there is nothing to site a battery against"
        raise SiteError(msg)
    return max(meters, key=lambda o: (o.confidence, o.n_hits))


def wall_candidates(
    house: HouseOutline, battery: BatterySpec, placement: PlacementSpec
) -> Candidates:
    """Lay candidate sites evenly along every wall with room for the battery.

    Each wall's candidates are centred on it, so a wall with room for exactly
    one battery gets it in the middle.
    """
    anchors, normals, walls = [], [], []
    for i, wall in enumerate(house.walls):
        room = wall.length - battery.width_m - 2.0 * placement.corner_margin_m
        if room < 0.0:
            continue
        n = int(room // placement.step_m) + 1
        along = (wall.length - (n - 1) * placement.step_m) / 2.0 + np.arange(n) * placement.step_m
        anchors.append(wall.start + along[:, None] * wall.tangent)
        normals.append(np.tile(wall.normal, (n, 1)))
        walls.append(np.full(n, i, dtype=np.int64))
    if not anchors:
        return Candidates(
            anchor=np.zeros((0, 2)),
            normal=np.zeros((0, 2)),
            wall=np.zeros(0, dtype=np.int64),
            battery=battery,
        )
    return Candidates(
        anchor=np.concatenate(anchors),
        normal=np.concatenate(normals),
        wall=np.concatenate(walls),
        battery=battery,
    )


def assess_site(state: MapState, rules: SiteRules) -> SiteAssessment:
    """Judge the finished map against the checklist and offer the best sites.

    Parameters
    ----------
    state
        The swarm's map: its occupancy (for the walls) and discovered objects
        (for the meter and everything a rule measures against).
    rules
        The parsed ``rules.yaml``.

    Returns
    -------
    SiteAssessment
        The overall verdict, the offered sites (best first, at most
        ``placement.top_k``), a short justification, and how many candidates
        were scored in total.

    Raises
    ------
    SiteError
        If no meter was discovered, no wall was traced beside it, or no wall
        is long enough for the battery.
    """
    meter = find_meter(state)
    house = trace_house(state, meter.box.center, rules.outline)
    candidates = wall_candidates(house, rules.battery, rules.placement)
    if not len(candidates):
        msg = (
            f"none of the {len(house.walls)} traced walls is long enough for a "
            f"{rules.battery.width_m} m battery"
        )
        raise SiteError(msg)

    context = SiteContext(
        meter=meter, house=house, objects=tuple(state.discovered.values()), state=state
    )
    outcomes = [rule.evaluate(candidates, context) for rule in rules.rules]

    n = len(candidates)
    cost = np.zeros(n)
    reject = np.zeros(n, dtype=bool)
    review = np.zeros(n, dtype=bool)
    broken = np.zeros(n, dtype=np.int64)
    fail_counts: dict[str, int] = {}
    for rule, outcome in zip(rules.rules, outcomes, strict=True):
        cost += rule.weight * outcome.cost
        failing = ~outcome.passed
        marginal = outcome.marginal if outcome.marginal is not None else np.zeros(n, dtype=bool)
        marginal = marginal & failing
        if rule.on_fail != "ignore":
            broken += failing
            fail_counts[rule.key] = int(failing.sum())
        if rule.on_fail == "fail":
            reject |= failing & ~marginal
            review |= failing & marginal
        elif rule.on_fail == "review":
            review |= failing

    # A plain list, not a NumPy object array: NumPy silently narrows a
    # StrEnum (it is also a str) to a bare str the moment it touches an
    # object-dtype array, which would leave every site's ``verdict`` failing
    # `is SiteVerdict.PASS` even though `== SiteVerdict.PASS` still holds.
    verdict = [
        SiteVerdict.REJECT if r else SiteVerdict.MANUAL_REVIEW if v else SiteVerdict.PASS
        for r, v in zip(reject.tolist(), review.tolist(), strict=True)
    ]
    rank = np.where(
        reject,
        _VERDICT_RANK[SiteVerdict.REJECT],
        np.where(review, _VERDICT_RANK[SiteVerdict.MANUAL_REVIEW], _VERDICT_RANK[SiteVerdict.PASS]),
    )

    placement = rules.placement
    order = np.lexsort((cost, broken, rank))
    if not placement.fill_with_flagged and (rank == 0).any():
        order = order[rank[order] == 0]

    picked: list[int] = []
    for i in order:
        near = np.linalg.norm(candidates.anchor[picked] - candidates.anchor[i], axis=1)
        if (near >= placement.min_separation_m).all():
            picked.append(int(i))
            if len(picked) == placement.top_k:
                break

    harness = next((r for r in rules.rules if isinstance(r, HarnessRun)), None)
    route = harness.route(candidates, context) if harness is not None else None

    sites = [
        _site(i, candidates, rules, outcomes, cost, verdict, meter, house, route) for i in picked
    ]
    overall = _overall_verdict(verdict)
    justification = _justify(overall, sites, fail_counts, n, len(house.walls))
    return SiteAssessment(
        verdict=overall, sites=tuple(sites), justification=justification, n_candidates=n
    )


def _overall_verdict(verdict: list[SiteVerdict]) -> SiteVerdict:
    """PASS if any candidate passes; REJECT only if every one does; else review.

    A single marginal site anywhere blocks REJECT on purpose -- turning a
    customer away outright is the expensive mistake in this product, worth
    avoiding even at the cost of a wider MANUAL_REVIEW net.
    """
    if any(v is SiteVerdict.PASS for v in verdict):
        return SiteVerdict.PASS
    if all(v is SiteVerdict.REJECT for v in verdict):
        return SiteVerdict.REJECT
    return SiteVerdict.MANUAL_REVIEW


def _conduit(
    i: int,
    candidates: Candidates,
    house: HouseOutline,
    battery_height_m: float,
    meter: DiscoveredObject,
    route: HarnessRoute | None,
) -> Points:
    """Build the harness's polyline from the meter to the top of the battery's back.

    Follows :attr:`HarnessRoute.forward` -- the same direction the
    ``harness_run`` rule scored -- through every corner the route passes, so
    the drawn conduit is never a route the rule would have rejected. Falls
    back to a straight line when ``rules.yaml`` has no ``harness_run`` entry
    (a caller's own, e.g. a test rule list), matching the old behaviour.
    """
    anchor = candidates.anchor[i]
    top = np.array([anchor[0], anchor[1], battery_height_m])
    if route is None:
        return np.array([meter.box.center, top])
    if route.forward[i]:
        plan = house.walk(route.s_meter, float(route.s_candidate[i]))
    else:
        plan = house.walk(float(route.s_candidate[i]), route.s_meter)[::-1]
    z = np.full(len(plan), battery_height_m)
    wall_path: Points = np.concatenate([plan, z[:, None]], axis=1)
    return np.vstack([meter.box.center[None, :], wall_path])


def _site(
    i: int,
    candidates: Candidates,
    rules: SiteRules,
    outcomes: list[RuleOutcome],
    cost: npt.NDArray[np.float64],
    verdict: list[SiteVerdict],
    meter: DiscoveredObject,
    house: HouseOutline,
    route: HarnessRoute | None,
) -> SiteCandidate:
    """Package candidate ``i`` as the contract type."""
    anchor = candidates.anchor[i]
    warnings = tuple(
        rule.explain(
            float(outcome.measure[i]),
            marginal=bool(outcome.marginal[i]) if outcome.marginal is not None else False,
        )
        for rule, outcome in zip(rules.rules, outcomes, strict=True)
        if not outcome.passed[i]
    )
    return SiteCandidate(
        pos=np.array([anchor[0], anchor[1], 0.0]),
        wall_normal=np.array([candidates.normal[i, 0], candidates.normal[i, 1], 0.0]),
        cost=float(cost[i]),
        breakdown={
            rule.key: float(outcome.measure[i])
            for rule, outcome in zip(rules.rules, outcomes, strict=True)
        },
        conduit=_conduit(i, candidates, house, rules.battery.height_m, meter, route),
        bushes_to_remove=[],
        verdict=verdict[i],
        warnings=warnings,
    )


def _justify(
    overall: SiteVerdict,
    sites: list[SiteCandidate],
    fail_counts: dict[str, int],
    n_candidates: int,
    n_walls: int,
) -> str:
    """One to three sentences a reviewer can act on without opening the map."""
    if overall is SiteVerdict.PASS:
        # The order sites are ranked in puts every PASS ahead of every
        # MANUAL_REVIEW and REJECT, so the first offered site is a PASS
        # whenever the overall call is.
        best = sites[0]
        parts = []
        harness = best.breakdown.get(HarnessRun.name)
        if harness is not None and math.isfinite(harness):
            parts.append(f"a {harness / _M_PER_FT:.0f} ft harness run")
        clearance = best.breakdown.get(FreeSpace.name)
        if clearance is not None and math.isfinite(clearance):
            parts.append(f"{clearance / _M_PER_FT:.1f} ft of front clearance")
        detail = " and ".join(parts) if parts else "every rule satisfied"
        return f"The best site clears every rule, with {detail}."
    if overall is SiteVerdict.MANUAL_REVIEW:
        best = sites[0]
        if not best.warnings:
            return "The best candidate needs a member's sign-off before approval."
        return "The best candidate needs a member's sign-off: " + "; ".join(best.warnings[:2]) + "."
    top = sorted(fail_counts.items(), key=lambda kv: -kv[1])[:_TOP_REASONS]
    reasons = " and ".join(name.replace("_", " ") for name, _ in top)
    tail = f", most often on {reasons}" if reasons else ""
    return f"All {n_candidates} candidate spots along the {n_walls} traced walls fail{tail}."
