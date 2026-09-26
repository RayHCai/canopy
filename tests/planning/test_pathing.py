"""plan_path and the shared PlanningGrid reachability helper."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest

from canopy.config import Config
from canopy.errors import PlanningError
from canopy.planning.pathing import plan_path, planning_grid
from canopy.planning.safety import ClearanceMap


def test_plan_path_straight_line_in_open_air(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    cm = clearance()
    start = np.array([-5.0, -5.0, 3.0])
    goal = np.array([5.0, 5.0, 3.0])
    path = plan_path(
        start, goal, cm, plan_factor=cfg.map.plan_factor, required_m=cfg.safety.inflation_m
    )
    assert len(path) >= 2
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)


def test_plan_path_leaves_the_padding_around_a_drone_it_starts_inside(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    """A drone stopped just outside ``min_separation_m`` of another can still plan out.

    The shield parks a standoff pair about that far apart, which is inside the
    avoid radius's half-cell padding. Here a wall shuts the one cell leading
    straight away from the other drone, so a planner that blocked the whole
    padded ball would find no first step at all.
    """
    cm = clearance(boxes=[((-15.0, -20.0, 0.0), (-0.5, 20.0, 12.0))])
    start = np.array([0.75, 0.25, 3.25])  # a cell centre, west neighbour in the wall
    other = start + np.array([1.3, -0.8, 0.0])
    goal = start + np.array([0.0, 5.0, 0.0])
    separation = cfg.safety.min_separation_m
    start_gap = float(np.linalg.norm(other - start))
    cell = cm.voxel * cfg.map.plan_factor
    assert separation < start_gap < separation + 0.5 * np.sqrt(3.0) * cell

    path = plan_path(
        start,
        goal,
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        avoid=other[None, :],
        avoid_radius_m=separation,
    )

    np.testing.assert_allclose(path[-1], goal)
    samples = np.vstack([cm.sample_segment(a, b) for a, b in pairwise(path)])
    assert np.linalg.norm(samples - other, axis=1).min() >= separation


def test_planning_grid_labels_open_space_as_one_component(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    cm = clearance()
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    passable_labels = grid.labels[grid.passable]
    assert np.all(passable_labels == passable_labels[0])
    assert passable_labels[0] != 0


def test_label_at_is_zero_outside_the_grid_and_in_impassable_cells(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    # A wall spanning the full height splits the lot into two components.
    cm = clearance(boxes=[((-0.5, -20.0, 0.0), (0.5, 20.0, 12.0))])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    far_outside = np.array([[1000.0, 1000.0, 1000.0]])
    assert grid.label_at(far_outside)[0] == 0

    inside_wall = np.array([[0.0, 0.0, 3.0]])
    assert grid.label_at(inside_wall)[0] == 0

    left = np.array([[-5.0, 0.0, 3.0]])
    right = np.array([[5.0, 0.0, 3.0]])
    left_label, right_label = grid.label_at(np.vstack([left, right]))
    assert left_label != 0
    assert right_label != 0
    assert left_label != right_label


def test_start_inside_a_margin_is_forced_passable(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    # On the ground, clearance is zero: without the start override, no cell
    # is passable there and the start would get label 0.
    start = np.array([0.0, 0.0, 0.0 + cfg.map.voxel_m / 2])
    grid = planning_grid(
        cm := clearance(),
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=start.reshape(1, 3),
    )
    assert grid.label_at(start.reshape(1, 3))[0] != 0

    without_override = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    assert without_override.label_at(start.reshape(1, 3))[0] == 0


def test_planning_grid_ignores_starts_outside_the_map(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    grid = planning_grid(
        clearance(),
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.array([[1000.0, 1000.0, 1000.0]]),
    )
    assert grid.passable.any()


def test_centres_returns_cell_midpoints(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    grid = planning_grid(
        clearance(),
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    mask = np.zeros_like(grid.passable)
    mask[0, 0, 0] = True
    centres = grid.centres(mask)
    assert centres.shape == (1, 3)
    np.testing.assert_allclose(centres[0], grid.origin + 0.5 * grid.cell)


@pytest.mark.parametrize("plan_factor", [0, -1])
def test_planning_grid_rejects_bad_plan_factor(
    plan_factor: int, clearance: Callable[..., ClearanceMap]
) -> None:
    with pytest.raises(PlanningError):
        planning_grid(clearance(), plan_factor=plan_factor, required_m=0.5, starts=np.zeros((0, 3)))
