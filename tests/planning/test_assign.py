"""Drone-to-frontier assignment: feasibility, conflict penalty, hysteresis."""

from __future__ import annotations

import numpy as np

from canopy.config import Config
from canopy.planning.assign import assign_frontiers
from canopy.planning.frontier import Frontier


def _frontier(
    viewpoint: tuple[float, float, float], gain: float = 10.0, component: int = 1
) -> Frontier:
    return Frontier(
        centroid=np.asarray(viewpoint, dtype=np.float64),
        n_voxels=10,
        gain=gain,
        viewpoint=np.asarray(viewpoint, dtype=np.float64),
        component=component,
    )


def test_three_drones_three_frontiers_is_one_to_one(cfg: Config) -> None:
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0]),
        2: np.array([0.0, 10.0, 0.0]),
    }
    frontiers = [
        _frontier((0.0, 0.5, 0.0)),
        _frontier((10.0, 0.5, 0.0)),
        _frontier((0.0, 10.5, 0.0)),
    ]
    components = {0: 1, 1: 1, 2: 1}
    result = assign_frontiers(positions, frontiers, {}, cfg.planner, components=components)
    assert len(result) == 3
    assert set(result.values()) == {0, 1, 2}


def test_conflict_penalty_steers_a_drone_away_from_another_drones_goal(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.5, 0.0, 0.0])}
    # Two frontiers close together; drone 1 already has a goal near frontier 0.
    frontiers = [_frontier((1.0, 0.0, 0.0)), _frontier((1.0, 5.0, 0.0))]
    current_goals = {1: np.array([1.0, 0.1, 0.0])}
    components = {0: 1, 1: 1}
    result = assign_frontiers(
        positions, frontiers, current_goals, cfg.planner, components=components
    )
    # Drone 0 should be steered to frontier 1, leaving frontier 0 to drone 1's held goal region.
    assert result[0] == 1


def test_hysteresis_keeps_current_goal_against_a_marginally_better_one(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0])}
    frontiers = [_frontier((5.0, 0.0, 0.0), gain=10.0), _frontier((5.1, 0.0, 0.0), gain=10.0)]
    current_goals = {0: np.array([5.0, 0.0, 0.0])}
    components = {0: 1}
    result = assign_frontiers(
        positions, frontiers, current_goals, cfg.planner, components=components
    )
    assert result[0] == 0


def test_hysteresis_switches_for_a_much_better_frontier(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0])}
    frontiers = [_frontier((5.0, 0.0, 0.0), gain=0.0), _frontier((0.5, 0.0, 0.0), gain=1000.0)]
    current_goals = {0: np.array([5.0, 0.0, 0.0])}
    components = {0: 1}
    result = assign_frontiers(
        positions, frontiers, current_goals, cfg.planner, components=components
    )
    assert result[0] == 1


def test_more_drones_than_frontiers_leaves_extras_unassigned(cfg: Config) -> None:
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0]),
    }
    frontiers = [_frontier((0.0, 0.5, 0.0))]
    components = {0: 1, 1: 1, 2: 1}
    result = assign_frontiers(positions, frontiers, {}, cfg.planner, components=components)
    assert len(result) == 1


def test_single_drone_needs_no_special_case(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0])}
    frontiers = [_frontier((1.0, 0.0, 0.0))]
    result = assign_frontiers(positions, frontiers, {}, cfg.planner, components={0: 1})
    assert result == {0: 0}


def test_infeasible_component_is_never_assigned(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0])}
    frontiers = [_frontier((1.0, 0.0, 0.0), component=2)]
    result = assign_frontiers(positions, frontiers, {}, cfg.planner, components={0: 1})
    assert result == {}


def test_drone_missing_from_components_is_infeasible_everywhere(cfg: Config) -> None:
    positions = {0: np.array([0.0, 0.0, 0.0])}
    frontiers = [_frontier((1.0, 0.0, 0.0), component=1)]
    result = assign_frontiers(positions, frontiers, {}, cfg.planner, components={})
    assert result == {}


def test_no_drones_or_no_frontiers_returns_empty(cfg: Config) -> None:
    assert assign_frontiers({}, [_frontier((0.0, 0.0, 0.0))], {}, cfg.planner, components={}) == {}
    assert assign_frontiers({0: np.zeros(3)}, [], {}, cfg.planner, components={0: 1}) == {}
