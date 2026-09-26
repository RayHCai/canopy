"""Assessing battery sites (:func:`canopy.site.assess_site` and friends).

Built on the same synthetic 10 x 8 m rectangular-ring house as
``tests/site/test_outline.py``, with a meter discovered on the outside of the
east wall.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from canopy.config import load_rules
from canopy.contracts import Cls, DiscoveredObject, MapState, OrientedBox, SiteVerdict
from canopy.errors import SiteError
from canopy.site import (
    Candidates,
    ClearOf,
    FreeSpace,
    Rule,
    RuleOutcome,
    SiteContext,
    SiteRules,
    assess_site,
    load_site_rules,
    parse_site_rules,
    register_rule,
    trace_house,
    wall_candidates,
)

Box = tuple[tuple[float, float, float], tuple[float, float, float]]

_RECT_WALLS: list[Box] = [
    ((-5.0, 3.75, 0.0), (5.0, 4.0, 3.0)),  # north
    ((-5.0, -4.0, 0.0), (5.0, -3.75, 3.0)),  # south
    ((4.75, -4.0, 0.0), (5.0, 4.0, 3.0)),  # east
    ((-5.0, -4.0, 0.0), (-4.75, 4.0, 3.0)),  # west
]

_METER_CENTER = np.array([5.15, 0.0, 1.5])
_METER_SIZE = np.array([0.3, 0.15, 0.45])


def _meter() -> DiscoveredObject:
    return DiscoveredObject(
        track_id=1,
        cls=Cls.METER,
        pos=_METER_CENTER,
        box=OrientedBox(center=_METER_CENTER, size=_METER_SIZE, yaw=0.0),
        n_hits=50,
        confidence=0.9,
        first_seen_t=0.0,
        first_seen_by=0,
    )


def _bush(track_id: int, center: tuple[float, float, float]) -> DiscoveredObject:
    size = np.array([0.6, 0.6, 0.8])
    return DiscoveredObject(
        track_id=track_id,
        cls=Cls.BUSH,
        pos=np.array(center),
        box=OrientedBox(center=np.array(center), size=size, yaw=0.0),
        n_hits=30,
        confidence=0.8,
        first_seen_t=0.0,
        first_seen_by=0,
    )


def _house(make_map: Callable[..., MapState], *bushes: DiscoveredObject) -> MapState:
    state = make_map(boxes=_RECT_WALLS)
    state.discovered = {1: _meter(), **dict(enumerate(bushes, start=2))}
    return state


@pytest.fixture
def rules() -> SiteRules:
    return load_site_rules()


def test_top_sites_are_on_the_house_face_outward_and_pass(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = _house(make_map)
    house_center = np.array([0.0, 0.0])
    assessment = assess_site(state, rules)

    assert assessment.verdict is SiteVerdict.PASS
    assert 0 < len(assessment.sites) <= rules.placement.top_k
    for site in assessment.sites:
        assert site.verdict is SiteVerdict.PASS
        assert not site.warnings
        assert np.dot(site.pos[:2] - house_center, site.wall_normal[:2]) > 0.0

    costs = [s.cost for s in assessment.sites]
    assert costs == sorted(costs)

    positions = np.array([s.pos[:2] for s in assessment.sites])
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert (
                np.linalg.norm(positions[i] - positions[j])
                >= rules.placement.min_separation_m - 1e-6
            )
    assert len(assessment.sites) == rules.placement.top_k


def test_top_sites_all_pass_harness_run_when_unobstructed(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = _house(make_map)
    assessment = assess_site(state, rules)
    for site in assessment.sites:
        assert site.breakdown["harness_run"] <= rules.placement.step_m + 10.0  # sane, finite


def test_a_bush_beside_the_meter_pushes_sites_away_from_it(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    bush = _bush(2, (4.6, 0.4, 0.4))
    state = _house(make_map, bush)
    min_m = next(r.min_m for r in rules.rules if isinstance(r, ClearOf) and r.cls == "BUSH")
    assessment = assess_site(state, rules)
    for site in assessment.sites:
        if site.verdict is not SiteVerdict.PASS:
            continue
        gap = np.linalg.norm(site.pos[:2] - np.array([4.6, 0.4]))
        assert gap >= min_m - 1e-6


def test_bushes_never_block_a_pass_but_still_cost_more(
    make_map: Callable[..., MapState],
) -> None:
    """``clear_of BUSH`` is ``on_fail: ignore`` -- movable, so a preference, not a gate.

    ``free_space`` also excepts ``BUSH`` explicitly (``rules.yaml``), so a
    wall of bushes must never turn an otherwise-clean east-wall site into a
    review or a reject; it only makes that site cost more than a bush-free one.
    """
    rules = load_site_rules()
    clear_state = _house(make_map)
    bushes = [_bush(i, (5.3, y, 0.4)) for i, y in enumerate(np.arange(-3.5, 3.51, 0.4), start=2)]
    bushy_state = _house(make_map, *bushes)

    clear = assess_site(clear_state, rules)
    bushy = assess_site(bushy_state, rules)
    assert clear.verdict is SiteVerdict.PASS
    assert bushy.verdict is SiteVerdict.PASS

    east_bushy = [s for s in bushy.sites if s.wall_normal[0] > 0.5]
    assert east_bushy, "expected at least one offered site on the bush-covered east wall"
    for site in east_bushy:
        assert site.verdict is SiteVerdict.PASS
        assert site.cost > 0.0


def test_fill_with_flagged_orders_passing_sites_before_non_pass(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["placement"]["fill_with_flagged"] = True
    raw["placement"]["top_k"] = 6
    rules = parse_site_rules(raw)
    bushes = [_bush(i, (5.3, y, 0.4)) for i, y in enumerate(np.arange(-1.0, 1.01, 0.4), start=2)]
    state = _house(make_map, *bushes)

    assessment = assess_site(state, rules)
    ranks = [0 if s.verdict is SiteVerdict.PASS else 1 for s in assessment.sites]
    assert ranks == sorted(ranks)


def test_fill_with_flagged_false_returns_only_passing_sites_when_any_pass(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["placement"]["fill_with_flagged"] = False
    rules = parse_site_rules(raw)
    state = _house(make_map)  # nothing blocking: every site should pass

    assessment = assess_site(state, rules)
    assert assessment.sites
    for site in assessment.sites:
        assert site.verdict is SiteVerdict.PASS


def test_no_meter_discovered_raises_site_error(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = make_map(boxes=_RECT_WALLS)
    with pytest.raises(SiteError):
        assess_site(state, rules)


def test_wall_candidates_skips_walls_too_short_for_the_battery(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = _house(make_map)
    house = trace_house(state, _METER_CENTER, rules.outline)
    min_length = rules.battery.width_m + 2.0 * rules.placement.corner_margin_m

    candidates = wall_candidates(house, rules.battery, rules.placement)
    used_walls = set(candidates.wall.tolist())
    for i, wall in enumerate(house.walls):
        if wall.length < min_length:
            assert i not in used_walls


def _east_wall_context(
    make_map: Callable[..., MapState], *extra_occ: Box, bushes: tuple[DiscoveredObject, ...] = ()
) -> tuple[SiteContext, Candidates]:
    rules = load_site_rules()
    state = make_map(boxes=[*_RECT_WALLS, *extra_occ])
    meter = _meter()
    discovered: dict[int, DiscoveredObject] = {1: meter, **dict(enumerate(bushes, start=2))}
    state.discovered = discovered

    house = trace_house(state, _METER_CENTER, rules.outline)
    candidates: Candidates = wall_candidates(house, rules.battery, rules.placement)
    context = SiteContext(meter=meter, house=house, objects=tuple(discovered.values()), state=state)
    return context, candidates


def _east_wall_indices(candidates: Candidates) -> np.ndarray:
    """Return indices of candidates whose normal points east (+X), the wall the meter sits on."""
    return np.where(candidates.normal[:, 0] > 0.5)[0]


def test_free_space_flags_sites_blocked_by_an_ac_unit_sized_box(
    make_map: Callable[..., MapState],
) -> None:
    # An AC-unit-sized 0.8 m cube, 0.5 m off the east wall, dead centre on it --
    # squarely in front of the only near-meter spots.
    ac_unit: Box = ((5.5, -0.4, 0.0), (6.3, 0.4, 0.8))
    context, candidates = _east_wall_context(make_map, ac_unit)
    rule = FreeSpace(min_reach_m=1.37, review_reach_m=1.68, except_cls=("BUSH",))
    outcome = rule.evaluate(candidates, context)

    east = _east_wall_indices(candidates)
    assert len(east) > 0
    near_center = east[np.argmin(np.abs(candidates.anchor[east, 1]))]
    assert not outcome.passed[near_center]
    assert np.isfinite(outcome.measure[near_center])


def test_free_space_ignores_a_box_flush_on_the_wall_thinner_than_wall_gap_m(
    make_map: Callable[..., MapState],
) -> None:
    # Sits inside the wall's own outer voxel (voxel_m is 0.25), so once mapped
    # occupancy it never extends past the wall's outer face -- well within
    # wall_gap_m (0.3) of it -- and the free_space sample grid never reaches it.
    thin_box: Box = ((4.8, -0.4, 0.0), (5.0, 0.4, 0.8))
    context, candidates = _east_wall_context(make_map, thin_box)
    rule = FreeSpace(min_reach_m=1.37, review_reach_m=1.68, except_cls=("BUSH",))
    outcome = rule.evaluate(candidates, context)

    east = _east_wall_indices(candidates)
    assert len(east) > 0
    assert outcome.passed[east].all()
    assert np.all(np.isinf(outcome.measure[east]))


def test_free_space_does_not_trip_on_a_bush_named_in_except_cls(
    make_map: Callable[..., MapState],
) -> None:
    # A bush-shaped obstruction, mapped as occupancy and also discovered as a
    # BUSH: except_cls=("BUSH",) must let it through even though it fills the
    # working-clearance region.
    bush_box: Box = ((5.4, -0.2, 0.0), (5.8, 0.2, 0.6))
    # The discovered box must cover the OCC region once voxel-rounded (voxel_m
    # is 0.25), not just the box's nominal extent, or free_space would still
    # trip on the sliver the rounding adds outside the except box.
    bush = DiscoveredObject(
        track_id=2,
        cls=Cls.BUSH,
        pos=np.array([5.6, 0.0, 0.5]),
        box=OrientedBox(center=np.array([5.6, 0.0, 0.5]), size=np.array([1.2, 1.0, 1.0]), yaw=0.0),
        n_hits=30,
        confidence=0.8,
        first_seen_t=0.0,
        first_seen_by=0,
    )
    context, candidates = _east_wall_context(make_map, bush_box, bushes=(bush,))
    rule = FreeSpace(min_reach_m=1.37, review_reach_m=1.68, except_cls=("BUSH",))
    outcome = rule.evaluate(candidates, context)

    east = _east_wall_indices(candidates)
    assert len(east) > 0
    near_center = east[np.argmin(np.abs(candidates.anchor[east, 1]))]
    assert outcome.passed[near_center]
    assert np.isinf(outcome.measure[near_center])


# ---------------------------------------------------------------------------
# Overall verdict: PASS if any candidate passes, REJECT only if every one
# does, MANUAL_REVIEW otherwise. Custom always-broken rules make each branch
# reachable without depending on the shipped checklist's own thresholds.
# ---------------------------------------------------------------------------
@register_rule
@dataclass(frozen=True, kw_only=True)
class _AlwaysBroken(Rule):
    """Test-only rule: fails every candidate, clean by construction."""

    name: ClassVar[str] = "test_always_broken_zz9"

    def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
        n = len(candidates)
        return RuleOutcome(measure=np.zeros(n), passed=np.zeros(n, dtype=bool), cost=np.zeros(n))

    def explain(self, _measure: float, *, marginal: bool) -> str:
        del marginal
        return "always broken, for testing"


def test_overall_verdict_is_reject_only_when_every_candidate_fails(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["rules"] = [{"rule": "test_always_broken_zz9"}]  # on_fail defaults to "fail"
    rules = parse_site_rules(raw)
    state = _house(make_map)

    assessment = assess_site(state, rules)
    assert assessment.verdict is SiteVerdict.REJECT
    assert all(s.verdict is SiteVerdict.REJECT for s in assessment.sites)
    assert str(assessment.n_candidates) in assessment.justification


def test_overall_verdict_is_manual_review_when_nothing_passes_but_nothing_is_rejected(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["rules"] = [{"rule": "test_always_broken_zz9", "on_fail": "review"}]
    rules = parse_site_rules(raw)
    state = _house(make_map)

    assessment = assess_site(state, rules)
    assert assessment.verdict is SiteVerdict.MANUAL_REVIEW
    assert all(s.verdict is SiteVerdict.MANUAL_REVIEW for s in assessment.sites)
