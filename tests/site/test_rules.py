"""``rules.yaml`` parsing and the built-in rules (:mod:`canopy.site.rules`).

``rules.yaml`` parsing is exercised at the public boundary
(:func:`canopy.site.load_site_rules`, :func:`canopy.site.parse_site_rules`).
Individual rules are exercised directly, against hand-built
:class:`~canopy.site.Candidates` and :class:`~canopy.site.SiteContext`, the
same pattern :mod:`tests.site.test_solver` uses for the solver as a whole --
a rectangular ring house built straight from :class:`~canopy.site.HouseOutline`
and :class:`~canopy.site.Wall`, without going through occupancy tracing, keeps
the arc-coordinate arithmetic exact and easy to check by hand.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from canopy.config import load_rules
from canopy.contracts import Cls, DiscoveredObject, MapState, OrientedBox
from canopy.errors import ConfigError
from canopy.site import (
    BatterySpec,
    Candidates,
    ClearOf,
    FreeSpace,
    HarnessRun,
    Headroom,
    HouseOutline,
    Rule,
    RuleOutcome,
    SiteContext,
    Wall,
    load_site_rules,
    parse_site_rules,
    register_rule,
    rule_names,
)

Box = tuple[tuple[float, float, float], tuple[float, float, float]]

_METER_NEAR = np.array([5.15, 0.0, 1.5])


@pytest.fixture
def raw_rules() -> dict[str, Any]:
    """Return a fresh, mutable deep copy of the shipped ``rules.yaml``."""
    return copy.deepcopy(load_rules())


# ---------------------------------------------------------------------------
# rules.yaml parsing
# ---------------------------------------------------------------------------
def test_shipped_rules_parse_and_hold_the_expected_rules() -> None:
    rules = load_site_rules()
    by_key = {r.key: r for r in rules.rules}

    harness = by_key["harness_run"]
    assert isinstance(harness, HarnessRun)
    assert harness.pass_max_m == pytest.approx(4.88, abs=0.01)
    assert harness.max_m == pytest.approx(7.92, abs=0.01)
    assert harness.hard_block == ("GARAGE_DOOR",)

    bush = by_key["clear_of_bush"]
    assert isinstance(bush, ClearOf)
    assert bush.cls == "BUSH"
    assert bush.on_fail == "ignore"

    conduit = by_key["clear_of_conduit"]
    assert isinstance(conduit, ClearOf)
    assert conduit.cls == "CONDUIT"


def test_unregistered_rule_name_lists_known_rules(raw_rules: dict[str, Any]) -> None:
    raw_rules["rules"][0]["rule"] = "not_a_real_rule"
    with pytest.raises(ConfigError, match="not_a_real_rule") as exc_info:
        parse_site_rules(raw_rules)
    message = str(exc_info.value)
    for name in rule_names():
        assert name in message


def test_unknown_parameter_key_raises(raw_rules: dict[str, Any]) -> None:
    raw_rules["rules"][0]["not_a_real_param"] = 1.0
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_wrong_parameter_type_raises(raw_rules: dict[str, Any]) -> None:
    raw_rules["rules"][0]["pass_max_m"] = "far"  # noqa: S105 -- a config key, not a credential
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_missing_required_parameter_raises(raw_rules: dict[str, Any]) -> None:
    del raw_rules["rules"][0]["pass_max_m"]
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_bad_cls_for_clear_of_raises(raw_rules: dict[str, Any]) -> None:
    for entry in raw_rules["rules"]:
        if entry["rule"] == "clear_of":
            entry["cls"] = "NOT_A_CLASS"
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_missing_top_level_section_raises(raw_rules: dict[str, Any]) -> None:
    del raw_rules["outline"]
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_unknown_top_level_section_raises(raw_rules: dict[str, Any]) -> None:
    raw_rules["bogus_section"] = {}
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_duplicate_breakdown_keys_raise(raw_rules: dict[str, Any]) -> None:
    raw_rules["rules"].append({"rule": "clear_of", "cls": "BUSH", "min_m": 0.3})
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_bad_on_fail_value_raises(raw_rules: dict[str, Any]) -> None:
    raw_rules["rules"][0]["on_fail"] = "sometimes"
    with pytest.raises(ConfigError, match="on_fail"):
        parse_site_rules(raw_rules)


@register_rule
@dataclass(frozen=True, kw_only=True)
class _PreferEast(Rule):
    """Test-only rule: cheapest on a wall facing +X, dearest on a wall facing -X."""

    name: ClassVar[str] = "test_prefer_east_zz9"

    def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
        facing_east = np.clip(candidates.normal[:, 0], 0.0, 1.0)
        cost = 1.0 - facing_east
        return RuleOutcome(measure=cost, passed=np.ones(len(candidates), dtype=bool), cost=cost)

    def explain(self, measure: float, *, marginal: bool) -> str:
        del marginal
        return f"faces east by {1.0 - measure:.2f}"


def test_registering_a_different_class_under_an_existing_name_raises() -> None:
    @dataclass(frozen=True, kw_only=True)
    class _Impostor(Rule):
        name: ClassVar[str] = "test_prefer_east_zz9"

        def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
            raise NotImplementedError

        def explain(self, _measure: float, *, marginal: bool) -> str:
            del marginal
            return ""

    with pytest.raises(ConfigError):
        register_rule(_Impostor)


# ---------------------------------------------------------------------------
# A hand-built rectangular ring house, for exact arc-coordinate arithmetic.
# 10 x 8 m, walls anticlockwise starting at the south-east corner:
# east (length 8, s in [0, 8]), north (length 10, s in [8, 18]),
# west (length 8, s in [18, 26]), south (length 10, s in [26, 36]).
# ---------------------------------------------------------------------------
def _square_house(half_x: float = 5.0, half_y: float = 4.0) -> HouseOutline:
    corners = np.array([[half_x, -half_y], [half_x, half_y], [-half_x, half_y], [-half_x, -half_y]])
    walls = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        tangent = (end - start) / np.linalg.norm(end - start)
        normal = np.array([tangent[1], -tangent[0]])
        walls.append(Wall(start=start, end=end, normal=normal))
    return HouseOutline(polygon=corners, walls=tuple(walls))


def _meter(pos: tuple[float, float, float]) -> DiscoveredObject:
    center = np.array(pos)
    return DiscoveredObject(
        track_id=1,
        cls=Cls.METER,
        pos=center,
        box=OrientedBox(center=center, size=np.array([0.3, 0.15, 0.45]), yaw=0.0),
        n_hits=50,
        confidence=0.9,
        first_seen_t=0.0,
        first_seen_by=0,
    )


def _object(
    cls: Cls,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float = 0.0,
) -> DiscoveredObject:
    center = np.array(pos)
    return DiscoveredObject(
        track_id=2,
        cls=cls,
        pos=center,
        box=OrientedBox(center=center, size=np.array(size), yaw=yaw),
        n_hits=30,
        confidence=0.8,
        first_seen_t=0.0,
        first_seen_by=0,
    )


def _candidates(
    anchors: list[tuple[float, float]],
    normals: list[tuple[float, float]],
    battery: BatterySpec,
) -> Candidates:
    n = len(anchors)
    return Candidates(
        anchor=np.array(anchors, dtype=np.float64),
        normal=np.array(normals, dtype=np.float64),
        wall=np.zeros(n, dtype=np.int64),
        battery=battery,
    )


def _context(
    house: HouseOutline,
    meter: DiscoveredObject,
    state: MapState,
    *extra: DiscoveredObject,
) -> SiteContext:
    return SiteContext(meter=meter, house=house, objects=(meter, *extra), state=state)


def _marginal(outcome: RuleOutcome, i: int) -> bool:
    """Candidate ``i``'s marginal flag; ``RuleOutcome.marginal`` may be ``None``."""
    return outcome.marginal is not None and bool(outcome.marginal[i])


# ---------------------------------------------------------------------------
# HarnessRun
# ---------------------------------------------------------------------------
#: Battery whose width is small enough that width/2 barely perturbs the arc math.
_THIN_BATTERY = BatterySpec(width_m=0.5, depth_m=0.5, height_m=1.0)


def test_harness_run_follows_the_walls_around_a_corner(
    make_map: Callable[..., MapState],
) -> None:
    """A candidate around the corner from the meter routes via the corner, not as the crow flies."""
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))  # s = 4, on the east wall
    candidates = _candidates([(0.0, 4.0)], [(0.0, 1.0)], _THIN_BATTERY)  # s = 13, on the north wall
    context = _context(house, meter, make_map())
    rule = HarnessRun(pass_max_m=100.0, max_m=200.0, hard_block=("GARAGE_DOOR",))

    outcome = rule.evaluate(candidates, context)

    straight_line = float(np.linalg.norm(candidates.anchor[0] - meter.box.center[:2]))
    # 4 m along the east wall to the corner, then 5 m along the north wall,
    # less half the battery's width -- well over the straight-line distance.
    assert outcome.measure[0] == pytest.approx(9.0 - _THIN_BATTERY.width_m / 2.0, abs=1e-6)
    assert outcome.measure[0] > straight_line


def test_harness_run_length_bands(make_map: Callable[..., MapState]) -> None:
    """16 ft (pass_max_m) and 26 ft (max_m) split a clean pass, a review and a reject."""
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))  # s = 4, drop = 0 (meter at battery height)
    # s = 8, 10 and 13 on the north wall -> forward runs of 4, 6 and 9 m.
    candidates = _candidates([(5.0, 4.0), (3.0, 4.0), (0.0, 4.0)], [(0.0, 1.0)] * 3, _THIN_BATTERY)
    context = _context(house, meter, make_map())
    rule = HarnessRun(pass_max_m=4.88, max_m=7.92, hard_block=("GARAGE_DOOR",))

    outcome = rule.evaluate(candidates, context)

    assert outcome.passed[0]
    assert not _marginal(outcome, 0)
    assert not outcome.passed[1]
    assert _marginal(outcome, 1)
    assert not outcome.passed[2]
    assert not _marginal(outcome, 2)


def test_harness_run_garage_door_on_the_short_route_forces_the_long_way_and_fails(
    make_map: Callable[..., MapState],
) -> None:
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))  # s = 4
    candidates = _candidates([(0.0, 4.0)], [(0.0, 1.0)], _THIN_BATTERY)  # s = 13
    garage_door = _object(
        Cls.GARAGE_DOOR, (5.0, 2.0, 1.0), (1.0, 0.2, 2.0)
    )  # s = 6, on the east wall
    context = _context(house, meter, make_map(), garage_door)
    rule = HarnessRun(pass_max_m=4.88, max_m=7.92, hard_block=("GARAGE_DOOR",))

    outcome = rule.evaluate(candidates, context)

    assert not outcome.passed[0]
    assert not _marginal(outcome, 0)
    # Forced the long way round (27 m plus change), well past the limit.
    assert outcome.measure[0] > rule.max_m
    assert "harness run" in rule.explain(float(outcome.measure[0]), marginal=False).lower()


def test_harness_run_garage_doors_on_every_route_are_unroutable(
    make_map: Callable[..., MapState],
) -> None:
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))  # s = 4
    candidates = _candidates([(0.0, 4.0)], [(0.0, 1.0)], _THIN_BATTERY)  # s = 13
    # One on the forward path (s = 6, east wall), one on the backward path
    # (s = 30, south wall) -- neither direction has a route left.
    forward_blocker = _object(Cls.GARAGE_DOOR, (5.0, 2.0, 1.0), (1.0, 0.2, 2.0))
    backward_blocker = _object(Cls.GARAGE_DOOR, (-1.0, -4.0, 1.0), (1.0, 0.2, 2.0))
    context = _context(house, meter, make_map(), forward_blocker, backward_blocker)
    rule = HarnessRun(pass_max_m=4.88, max_m=7.92, hard_block=("GARAGE_DOOR",))

    outcome = rule.evaluate(candidates, context)

    assert not outcome.passed[0]
    assert not _marginal(outcome, 0)
    assert not np.isfinite(outcome.measure[0])
    assert "garage door" in rule.explain(float(outcome.measure[0]), marginal=False).lower()


def test_harness_run_sees_a_garage_door_straddling_the_outline_start(
    make_map: Callable[..., MapState],
) -> None:
    """A blocker across the arc coordinate's seam (s = 0 = perimeter) still blocks.

    Where the outline starts is an artefact of contour tracing, so a garage
    door sitting across it must block exactly as it would anywhere else. Its
    corners land near both 0 and the perimeter; read as a plain interval they
    would span the whole house and put its near edge in the wrong place.
    """
    house = _square_house()
    meter = _meter((3.0, -4.0, 1.0))  # s = 34, on the south wall
    # s = 35.65, 1.4 m on from the meter, past the south-east corner at s = 0.
    candidates = _candidates([(4.65, -4.0)], [(0.0, -1.0)], _THIN_BATTERY)
    # Turned 45 degrees across the corner: near edge at s = 35.09, on the way.
    garage_door = _object(Cls.GARAGE_DOOR, (4.8, -3.8, 1.0), (1.0, 1.0, 2.0), yaw=np.pi / 4)
    context = _context(house, meter, make_map(), garage_door)
    rule = HarnessRun(pass_max_m=4.88, max_m=7.92, hard_block=("GARAGE_DOOR",))

    outcome = rule.evaluate(candidates, context)

    assert not outcome.passed[0]
    assert outcome.measure[0] > rule.max_m  # sent the long way round


def test_harness_run_door_on_the_route_is_a_marginal_crossing(
    make_map: Callable[..., MapState],
) -> None:
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))  # s = 4
    candidates = _candidates([(4.5, 4.0)], [(0.0, 1.0)], _THIN_BATTERY)  # s = 8.5, run = 4.5 m
    door = _object(Cls.DOOR, (5.0, 2.0, 1.0), (1.0, 0.1, 2.0))  # s = 6, on the run
    context = _context(house, meter, make_map(), door)
    rule = HarnessRun(pass_max_m=4.88, max_m=7.92, review_block=("DOOR",))

    outcome = rule.evaluate(candidates, context)

    assert not outcome.passed[0]
    assert _marginal(outcome, 0)
    assert outcome.measure[0] <= rule.pass_max_m
    assert "door" in rule.explain(float(outcome.measure[0]), marginal=True).lower()


def test_harness_run_hard_block_requires_a_known_class() -> None:
    with pytest.raises(ConfigError):
        HarnessRun(pass_max_m=1.0, max_m=2.0, hard_block=("NOT_A_CLASS",)).validate("test")


# ---------------------------------------------------------------------------
# FreeSpace
# ---------------------------------------------------------------------------
_ALLEY_BATTERY = BatterySpec(width_m=0.6, depth_m=0.4, height_m=0.8)
_ALLEY_CANDIDATE = _candidates([(0.0, 0.0)], [(1.0, 0.0)], _ALLEY_BATTERY)
_ALLEY_RULE = FreeSpace(
    min_reach_m=1.0, review_reach_m=1.5, side_m=0.1, wall_gap_m=0.2, ground_m=0.3
)


def _alley_context(
    make_map: Callable[..., MapState],
    boxes: list[Box] | None = None,
    unknown: list[Box] | None = None,
) -> SiteContext:
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))
    return _context(house, meter, make_map(boxes=boxes or [], unknown=unknown or []))


def test_free_space_marginal_alley_sends_to_review(make_map: Callable[..., MapState]) -> None:
    # Something at ~1.3 m out: inside the working alley, short of review_reach_m.
    context = _alley_context(make_map, boxes=[((1.2, -0.5, 0.0), (1.4, 0.5, 1.0))])
    outcome = _ALLEY_RULE.evaluate(_ALLEY_CANDIDATE, context)
    assert not outcome.passed[0]
    assert _marginal(outcome, 0)
    assert _ALLEY_RULE.min_reach_m <= outcome.measure[0] < _ALLEY_RULE.review_reach_m


def test_free_space_unmapped_alley_sends_to_review(make_map: Callable[..., MapState]) -> None:
    # Nothing occupied, but part of the alley was never flown -- not approvable outright.
    context = _alley_context(make_map, unknown=[((1.0, -0.5, 0.2), (1.1, 0.5, 0.6))])
    outcome = _ALLEY_RULE.evaluate(_ALLEY_CANDIDATE, context)
    assert not outcome.passed[0]
    assert _marginal(outcome, 0)
    assert not np.isfinite(outcome.measure[0])


def test_free_space_off_the_mapped_grid_sends_to_review(
    make_map: Callable[..., MapState],
) -> None:
    """Space beyond the map's extent was never seen, so it defers rather than passing."""
    far = _candidates([(1000.0, 0.0)], [(1.0, 0.0)], _ALLEY_BATTERY)
    outcome = _ALLEY_RULE.evaluate(far, _alley_context(make_map))
    assert not outcome.passed[0]
    assert _marginal(outcome, 0)


def test_free_space_close_obstruction_fails(make_map: Callable[..., MapState]) -> None:
    context = _alley_context(make_map, boxes=[((0.3, -0.5, 0.0), (0.5, 0.5, 1.0))])
    outcome = _ALLEY_RULE.evaluate(_ALLEY_CANDIDATE, context)
    assert not outcome.passed[0]
    assert not _marginal(outcome, 0)


def test_free_space_clear_alley_passes(make_map: Callable[..., MapState]) -> None:
    context = _alley_context(make_map)
    outcome = _ALLEY_RULE.evaluate(_ALLEY_CANDIDATE, context)
    assert outcome.passed[0]


# ---------------------------------------------------------------------------
# Headroom
# ---------------------------------------------------------------------------
def test_headroom_occupied_over_the_footprint_fails(make_map: Callable[..., MapState]) -> None:
    context = _alley_context(make_map, boxes=[((0.2, -0.3, 0.9), (0.4, 0.3, 1.2))])
    rule = Headroom(clear_m=1.5, wall_gap_m=0.2)
    outcome = rule.evaluate(_ALLEY_CANDIDATE, context)
    assert not outcome.passed[0]
    assert not _marginal(outcome, 0)


def test_headroom_unmapped_over_the_footprint_sends_to_review(
    make_map: Callable[..., MapState],
) -> None:
    context = _alley_context(make_map, unknown=[((0.2, -0.3, 0.9), (0.4, 0.3, 1.2))])
    rule = Headroom(clear_m=1.5, wall_gap_m=0.2)
    outcome = rule.evaluate(_ALLEY_CANDIDATE, context)
    assert not outcome.passed[0]
    assert _marginal(outcome, 0)


def test_headroom_clear_footprint_passes(make_map: Callable[..., MapState]) -> None:
    context = _alley_context(make_map)
    rule = Headroom(clear_m=1.5, wall_gap_m=0.2)
    outcome = rule.evaluate(_ALLEY_CANDIDATE, context)
    assert outcome.passed[0]


# ---------------------------------------------------------------------------
# ClearOf
# ---------------------------------------------------------------------------
_PLAN_BATTERY = BatterySpec(width_m=0.6, depth_m=0.4, height_m=0.8)
_PLAN_CANDIDATE = _candidates([(0.0, 0.0)], [(1.0, 0.0)], _PLAN_BATTERY)


def _plan_context(make_map: Callable[..., MapState], *objects: DiscoveredObject) -> SiteContext:
    house = _square_house()
    meter = _meter((5.0, 0.0, 1.0))
    return _context(house, meter, make_map(), *objects)


def test_clear_of_plan_ignores_height_separation(make_map: Callable[..., MapState]) -> None:
    """A window straight above the battery's footprint fails a plan check but not a 3D one."""
    window = _object(Cls.WINDOW, (0.2, 0.0, 3.0), (0.6, 0.1, 0.6))
    context = _plan_context(make_map, window)

    plan_rule = ClearOf(cls="WINDOW", min_m=0.5, plan=True)
    plan_outcome = plan_rule.evaluate(_PLAN_CANDIDATE, context)
    assert not plan_outcome.passed[0]

    solid_rule = ClearOf(cls="WINDOW", min_m=0.5, plan=False)
    solid_outcome = solid_rule.evaluate(_PLAN_CANDIDATE, context)
    assert solid_outcome.passed[0]


def test_clear_of_below_m_ignores_upper_floor_windows(make_map: Callable[..., MapState]) -> None:
    ground_window = _object(Cls.WINDOW, (0.2, 0.0, 1.0), (0.6, 0.1, 0.6))
    context = _plan_context(make_map, ground_window)
    rule = ClearOf(cls="WINDOW", min_m=0.5, plan=True, below_m=2.0)
    assert not rule.evaluate(_PLAN_CANDIDATE, context).passed[0]

    upper_window = _object(Cls.WINDOW, (0.2, 0.0, 3.0), (0.6, 0.1, 0.6))
    context = _plan_context(make_map, upper_window)
    assert rule.evaluate(_PLAN_CANDIDATE, context).passed[0]


def test_clear_of_swing_grows_the_zone_the_door_opens_into(
    make_map: Callable[..., MapState],
) -> None:
    # A 1 m wide, thin door footprint out from the wall and beside the
    # battery's footprint (x in [0, 0.4], y in [-0.3, 0.3]): 0.15 m clear of
    # it in plan, which passes on its own but not once the door's own width
    # grows its thin axis into the battery's footprint.
    door = _object(Cls.DOOR, (0.6, 0.0, 1.0), (0.1, 1.0, 2.0))
    context = _plan_context(make_map, door)

    plain = ClearOf(cls="DOOR", min_m=0.1, plan=True)
    assert plain.evaluate(_PLAN_CANDIDATE, context).passed[0]

    swinging = ClearOf(cls="DOOR", min_m=0.1, plan=True, swing=True)
    assert not swinging.evaluate(_PLAN_CANDIDATE, context).passed[0]


def test_clear_of_review_m_makes_a_short_gap_marginal(make_map: Callable[..., MapState]) -> None:
    gas_meter = _object(Cls.GAS_METER, (1.0, 0.0, 1.0), (0.3, 0.3, 0.4))
    context = _plan_context(make_map, gas_meter)
    rule = ClearOf(cls="GAS_METER", plan=True, min_m=0.3, review_m=1.0)

    outcome = rule.evaluate(_PLAN_CANDIDATE, context)
    assert not outcome.passed[0]
    assert _marginal(outcome, 0)


def test_clear_of_on_fail_and_review_m_validation() -> None:
    with pytest.raises(ConfigError):
        ClearOf(cls="BUSH", min_m=1.0, review_m=0.5).validate("test")
    with pytest.raises(ConfigError):
        ClearOf(cls="BUSH", min_m=0.5, on_fail="sometimes").validate("test")
