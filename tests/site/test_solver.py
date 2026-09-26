"""Suggesting battery sites (:func:`canopy.site.suggest_sites` and friends).

Built on the same synthetic 10 x 8 m rectangular-ring house as
``tests/site/test_outline.py``, with a meter discovered on the outside of the
east wall.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from canopy.config import load_rules
from canopy.contracts import Cls, DiscoveredObject, MapState, OrientedBox
from canopy.errors import SiteError
from canopy.site import (
    Candidates,
    ClearOf,
    FreeSpace,
    MeterDistance,
    SiteContext,
    SiteRules,
    load_site_rules,
    parse_site_rules,
    suggest_sites,
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


def test_top_sites_are_on_the_house_face_outward_and_meet_the_meter_rule(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = _house(make_map)
    house_center = np.array([0.0, 0.0])
    sites = suggest_sites(state, rules)

    assert 0 < len(sites) <= rules.placement.top_k
    for site in sites:
        assert not site.warnings
        assert np.dot(site.pos[:2] - house_center, site.wall_normal[:2]) > 0.0

    costs = [s.cost for s in sites]
    assert costs == sorted(costs)

    positions = np.array([s.pos[:2] for s in sites])
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert (
                np.linalg.norm(positions[i] - positions[j])
                >= rules.placement.min_separation_m - 1e-6
            )
    assert len(sites) == rules.placement.top_k


def test_top_sites_all_pass_meter_distance_when_unobstructed(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = _house(make_map)
    sites = suggest_sites(state, rules)
    meter_limit = next(r.max_m for r in rules.rules if isinstance(r, MeterDistance))
    for site in sites:
        gap = np.linalg.norm(site.pos[:2] - _METER_CENTER[:2])
        assert gap <= meter_limit + 1e-6


def test_a_bush_beside_the_meter_pushes_sites_away_from_it(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    bush = _bush(2, (4.6, 0.4, 0.4))
    state = _house(make_map, bush)
    min_m = next(r.min_m for r in rules.rules if isinstance(r, ClearOf) and r.cls == "BUSH")
    sites = suggest_sites(state, rules)
    for site in sites:
        if site.warnings:
            continue
        gap = np.linalg.norm(site.pos[:2] - np.array([4.6, 0.4]))
        assert gap >= min_m - 1e-6


def test_bushes_blocking_every_spot_near_the_meter_still_return_flagged_sites(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    rules = parse_site_rules(raw)
    # A wall of bushes covering every candidate within meter_distance.max_m of
    # the meter, on both walls it could reach.
    bushes = [_bush(i, (5.3, y, 0.4)) for i, y in enumerate(np.arange(-2.0, 2.01, 0.4), start=2)]
    state = _house(make_map, *bushes)

    sites = suggest_sites(state, rules)
    assert sites, (
        "the solver must never return an empty list when the fill_with_flagged path applies"
    )
    for site in sites:
        assert site.warnings
        assert any("bush" in w.lower() for w in site.warnings)


def test_fill_with_flagged_orders_passing_sites_before_flagged(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["placement"]["fill_with_flagged"] = True
    raw["placement"]["top_k"] = 6
    rules = parse_site_rules(raw)
    # Bushes block the near half of the east wall (close to the meter) but leave
    # the far side, and the other walls, clear -- so some sites pass and some
    # (near the meter, but too close to a bush) are flagged.
    bushes = [_bush(i, (5.3, y, 0.4)) for i, y in enumerate(np.arange(-1.0, 1.01, 0.4), start=2)]
    state = _house(make_map, *bushes)

    sites = suggest_sites(state, rules)
    warned = [bool(s.warnings) for s in sites]
    # Every passing site (False) must precede every flagged one (True).
    assert warned == sorted(warned)


def test_fill_with_flagged_false_returns_only_passing_sites_when_any_pass(
    make_map: Callable[..., MapState],
) -> None:
    raw = copy.deepcopy(load_rules())
    raw["placement"]["fill_with_flagged"] = False
    rules = parse_site_rules(raw)
    state = _house(make_map)  # nothing blocking: every site should pass

    sites = suggest_sites(state, rules)
    assert sites
    for site in sites:
        assert not site.warnings


def test_no_meter_discovered_raises_site_error(
    make_map: Callable[..., MapState], rules: SiteRules
) -> None:
    state = make_map(boxes=_RECT_WALLS)
    with pytest.raises(SiteError):
        suggest_sites(state, rules)


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
    rule = FreeSpace(front_m=0.9, except_cls=("BUSH",))
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
    rule = FreeSpace(front_m=0.9, except_cls=("BUSH",))
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
    rule = FreeSpace(front_m=0.9, except_cls=("BUSH",))
    outcome = rule.evaluate(candidates, context)

    east = _east_wall_indices(candidates)
    assert len(east) > 0
    near_center = east[np.argmin(np.abs(candidates.anchor[east, 1]))]
    assert outcome.passed[near_center]
    assert np.isinf(outcome.measure[near_center])
