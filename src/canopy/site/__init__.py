"""Battery siting: the top few places on the house's walls to mount a battery.

:mod:`canopy.site.outline` traces the house's walls from the swarm's occupancy
map (:func:`trace_house`). :mod:`canopy.site.rules` turns ``config/rules.yaml``
into a list of placement :class:`Rule` objects -- each both a constraint and a
cost factor, and new ones added by :func:`register_rule` -- and
:mod:`canopy.site.geometry` measures the clearances they need.
:mod:`canopy.site.solver` lays candidates along the walls, has every rule judge
them and keeps the best (:func:`suggest_sites`). Only what the swarm found is
read, never the scene manifest.

Public API: :func:`suggest_sites`, :func:`load_site_rules`, and what a new rule
needs -- :class:`Rule`, :func:`register_rule`, :class:`Candidates`,
:class:`SiteContext`, :class:`RuleOutcome`, :class:`Boxes`, :func:`box_gap`,
:func:`point_gap`.
"""

from __future__ import annotations

from canopy.site.geometry import Boxes, box_gap, point_gap
from canopy.site.outline import HouseOutline, Wall, trace_house
from canopy.site.rules import (
    BatterySpec,
    Candidates,
    ClearOf,
    Facing,
    FreeSpace,
    MeterDistance,
    OutlineSpec,
    PlacementSpec,
    Rule,
    RuleOutcome,
    SiteContext,
    SiteRules,
    build_rule,
    load_site_rules,
    parse_site_rules,
    register_rule,
    rule_names,
)
from canopy.site.solver import find_meter, suggest_sites, wall_candidates

__all__ = [
    "BatterySpec",
    "Boxes",
    "Candidates",
    "ClearOf",
    "Facing",
    "FreeSpace",
    "HouseOutline",
    "MeterDistance",
    "OutlineSpec",
    "PlacementSpec",
    "Rule",
    "RuleOutcome",
    "SiteContext",
    "SiteRules",
    "Wall",
    "box_gap",
    "build_rule",
    "find_meter",
    "load_site_rules",
    "parse_site_rules",
    "point_gap",
    "register_rule",
    "rule_names",
    "suggest_sites",
    "trace_house",
    "wall_candidates",
]
