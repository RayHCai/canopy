"""``rules.yaml`` parsing and the ``Rule`` extension point (:mod:`canopy.site.rules`).

Exercises the public boundary only: :func:`canopy.site.load_site_rules` and
:func:`canopy.site.parse_site_rules` on copies of the shipped ``rules.yaml``,
and :func:`canopy.site.register_rule` for a custom rule added from a test.
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
    Candidates,
    ClearOf,
    MeterDistance,
    Rule,
    RuleOutcome,
    SiteContext,
    load_site_rules,
    parse_site_rules,
    register_rule,
    rule_names,
    suggest_sites,
)

#: 6 feet, the shipped meter_distance limit.
_SIX_FEET_M = 1.83

Box = tuple[tuple[float, float, float], tuple[float, float, float]]

#: A 10 x 8 m rectangular ring, centred at the origin, walls 0.25 m thick.
_RECT_WALLS: list[Box] = [
    ((-5.0, 3.75, 0.0), (5.0, 4.0, 3.0)),  # north
    ((-5.0, -4.0, 0.0), (5.0, -3.75, 3.0)),  # south
    ((4.75, -4.0, 0.0), (5.0, 4.0, 3.0)),  # east
    ((-5.0, -4.0, 0.0), (-4.75, 4.0, 3.0)),  # west
]

_METER_NEAR = np.array([5.15, 0.0, 1.5])


@pytest.fixture
def raw_rules() -> dict[str, Any]:
    """Return a fresh, mutable deep copy of the shipped ``rules.yaml``."""
    return copy.deepcopy(load_rules())


def test_shipped_rules_parse_and_hold_the_expected_rules() -> None:
    rules = load_site_rules()
    by_key = {r.key: r for r in rules.rules}

    meter = by_key["meter_distance"]
    assert isinstance(meter, MeterDistance)
    assert meter.max_m == pytest.approx(_SIX_FEET_M, abs=0.01)

    bush = by_key["clear_of_bush"]
    assert isinstance(bush, ClearOf)
    assert bush.cls == "BUSH"

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
    raw_rules["rules"][0]["max_m"] = "far"
    with pytest.raises(ConfigError):
        parse_site_rules(raw_rules)


def test_missing_required_parameter_raises(raw_rules: dict[str, Any]) -> None:
    del raw_rules["rules"][0]["max_m"]
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


@register_rule
@dataclass(frozen=True, kw_only=True)
class _PreferEast(Rule):
    """Test-only rule: cheapest on a wall facing +X, dearest on a wall facing -X."""

    name: ClassVar[str] = "test_prefer_east_zz9"

    def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
        facing_east = np.clip(candidates.normal[:, 0], 0.0, 1.0)
        cost = 1.0 - facing_east
        return RuleOutcome(measure=cost, passed=np.ones(len(candidates), dtype=bool), cost=cost)

    def explain(self, measure: float) -> str:
        return f"faces east by {1.0 - measure:.2f}"


def test_custom_rule_ranks_sites_by_its_own_cost(
    make_map: Callable[..., MapState], raw_rules: dict[str, Any]
) -> None:
    raw_rules["rules"] = [{"rule": "test_prefer_east_zz9"}]
    rules = parse_site_rules(raw_rules)

    state = make_map(boxes=_RECT_WALLS)
    meter = DiscoveredObject(
        track_id=1,
        cls=Cls.METER,
        pos=_METER_NEAR,
        box=OrientedBox(center=_METER_NEAR, size=np.array([0.3, 0.15, 0.45]), yaw=0.0),
        n_hits=50,
        confidence=0.9,
        first_seen_t=0.0,
        first_seen_by=0,
    )
    state.discovered = {1: meter}

    sites = suggest_sites(state, rules)
    assert sites, "expected at least one suggested site"
    for site in sites:
        assert site.wall_normal[0] == pytest.approx(1.0, abs=0.01), (
            "every top site should be on the east-facing wall, the cheapest under the custom rule"
        )
    costs = [site.cost for site in sites]
    assert costs == sorted(costs)


def test_registering_a_different_class_under_an_existing_name_raises() -> None:
    @dataclass(frozen=True, kw_only=True)
    class _Impostor(Rule):
        name: ClassVar[str] = "test_prefer_east_zz9"

        def evaluate(self, candidates: Candidates, _context: SiteContext) -> RuleOutcome:
            raise NotImplementedError

        def explain(self, _measure: float) -> str:
            return ""

    with pytest.raises(ConfigError):
        register_rule(_Impostor)
