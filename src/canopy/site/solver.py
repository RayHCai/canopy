"""Battery siting: lay candidate sites along the walls, judge them, keep the best few.

The search space is small and exact rather than a cost raster: a battery
hangs on a wall, so the only freedom is *where along which wall*. Candidates
are laid every ``placement.step_m`` along each traced wall long enough to take
the battery with ``placement.corner_margin_m`` to spare at both ends -- which
is what "along the edge of the house" means, by construction -- and every
rule in ``rules.yaml`` judges all of them in one vectorised call.

Ranking is two-tiered. Sites that pass every required rule come first, by
cost; flagged sites follow, fewest broken rules first, then by cost. The
top ``placement.top_k`` are then picked greedily, skipping any site within
``placement.min_separation_m`` of one already picked, so three suggestions
are three different options rather than one spot nudged sideways.
"""

from __future__ import annotations

import numpy as np

from canopy.contracts import Cls, DiscoveredObject, MapState, SiteCandidate
from canopy.errors import SiteError
from canopy.site.outline import HouseOutline, trace_house
from canopy.site.rules import (
    BatterySpec,
    Candidates,
    PlacementSpec,
    RuleOutcome,
    SiteContext,
    SiteRules,
)

__all__ = ["find_meter", "suggest_sites", "wall_candidates"]


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


def suggest_sites(state: MapState, rules: SiteRules) -> list[SiteCandidate]:
    """Suggest up to ``placement.top_k`` battery sites from the finished map.

    Parameters
    ----------
    state
        The swarm's map: its occupancy (for the walls) and discovered objects
        (for the meter and everything a rule measures against).
    rules
        The parsed ``rules.yaml``.

    Returns
    -------
    list of SiteCandidate
        Best first. Sites that break a required rule carry a warning per rule
        broken, and are offered only after every site that passes -- or, with
        ``placement.fill_with_flagged`` off, only when none passes.

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
    cost = np.zeros(len(candidates))
    broken = np.zeros((len(rules.rules), len(candidates)), dtype=bool)
    for k, (rule, outcome) in enumerate(zip(rules.rules, outcomes, strict=True)):
        cost += rule.weight * outcome.cost
        broken[k] = rule.required & ~outcome.passed
    n_broken = broken.sum(axis=0)

    placement = rules.placement
    order = np.lexsort((cost, n_broken))
    if not placement.fill_with_flagged and (n_broken == 0).any():
        order = order[n_broken[order] == 0]
    picked: list[int] = []
    for i in order:
        near = np.linalg.norm(candidates.anchor[picked] - candidates.anchor[i], axis=1)
        if (near >= placement.min_separation_m).all():
            picked.append(int(i))
            if len(picked) == placement.top_k:
                break

    return [_site(i, candidates, rules, outcomes, cost, broken, meter) for i in picked]


def _site(
    i: int,
    candidates: Candidates,
    rules: SiteRules,
    outcomes: list[RuleOutcome],
    cost: np.ndarray,
    broken: np.ndarray,
    meter: DiscoveredObject,
) -> SiteCandidate:
    """Package candidate ``i`` as the contract type."""
    anchor = candidates.anchor[i]
    return SiteCandidate(
        pos=np.array([anchor[0], anchor[1], 0.0]),
        wall_normal=np.array([candidates.normal[i, 0], candidates.normal[i, 1], 0.0]),
        cost=float(cost[i]),
        breakdown={
            rule.key: float(outcome.measure[i])
            for rule, outcome in zip(rules.rules, outcomes, strict=True)
        },
        # From the meter to where the conduit enters the battery: the top of its back.
        conduit=np.array([meter.box.center, [anchor[0], anchor[1], rules.battery.height_m]]),
        bushes_to_remove=[],
        warnings=tuple(
            rule.explain(float(outcome.measure[i]))
            for k, (rule, outcome) in enumerate(zip(rules.rules, outcomes, strict=True))
            if broken[k, i]
        ),
    )
