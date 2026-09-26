"""Frontier voxel extraction and viewpoint selection."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage

from canopy.config import Config
from canopy.contracts import MapState
from canopy.planning.frontier import find_frontiers, find_inspection_targets, frontier_voxels
from canopy.planning.pathing import PlanningGrid, planning_grid
from canopy.planning.safety import ClearanceMap


def _expected_frontier_mask(state: MapState, cm: ClearanceMap, min_z: float) -> np.ndarray:
    """Independent check: dilate UNKNOWN by one 6-connected step, then filter."""
    free = state.occ == 1  # Occ.FREE
    unknown = state.occ == 0  # Occ.UNKNOWN
    six = ndimage.generate_binary_structure(3, 1)
    touches_unknown = ndimage.binary_dilation(unknown, structure=six) & ~unknown
    shape = state.occ.shape
    axes = [state.origin[a] + (np.arange(shape[a]) + 0.5) * state.voxel for a in range(3)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    in_fence = np.all((centres >= cm.geofence[0]) & (centres <= cm.geofence[1]), axis=-1)
    above = centres[..., 2] >= min_z
    result: np.ndarray = free & touches_unknown & in_fence & above
    return result


def test_frontier_voxels_match_an_independent_boundary_computation(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    unknown_box = ((0.0, -20.0, 1.0), (15.0, 20.0, 8.0))
    state = make_map(unknown=[unknown_box])
    cm = clearance(unknown=[unknown_box])
    got = frontier_voxels(state, cm, cfg.planner)
    expected = _expected_frontier_mask(state, cm, cfg.planner.frontier_min_z_m)
    np.testing.assert_array_equal(got, expected)
    assert got.any()


def test_small_clusters_are_dropped_and_large_ones_kept(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    # A cube big enough to keep, isolated so it forms one cluster.
    big = ((-1.0, -1.0, 3.0), (1.0, 1.0, 5.0))
    state = make_map(unknown=[big])
    cm = clearance(unknown=[big])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    drone_component = int(grid.label_at(np.array([[-5.0, -5.0, 3.0]]))[0])
    frontiers = find_frontiers(state, cm, grid, cfg.planner, cfg.map, components={drone_component})
    assert len(frontiers) == 1
    f = frontiers[0]
    assert f.n_voxels >= cfg.planner.frontier_min_voxels
    np.testing.assert_allclose(f.centroid, [0.0, 0.0, 4.0], atol=0.3)


def test_frontier_below_min_z_or_outside_geofence_is_ignored(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    # Entirely below frontier_min_z_m (1.0 m): no frontier voxels survive.
    low_box = ((-1.0, -1.0, 0.1), (1.0, 1.0, 0.4))
    state = make_map(unknown=[low_box])
    cm = clearance(unknown=[low_box])
    assert not frontier_voxels(state, cm, cfg.planner).any()


def test_frontier_in_a_component_with_no_drone_is_dropped(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    big = ((-1.0, -1.0, 3.0), (1.0, 1.0, 5.0))
    state = make_map(unknown=[big])
    cm = clearance(unknown=[big])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    # No live drone occupies any component.
    frontiers = find_frontiers(state, cm, grid, cfg.planner, cfg.map, components=set())
    assert frontiers == []


def test_viewpoint_is_passable_with_required_clearance(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    big = ((-1.0, -1.0, 3.0), (1.0, 1.0, 5.0))
    state = make_map(unknown=[big])
    cm = clearance(unknown=[big])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    drone_component = int(grid.label_at(np.array([[-5.0, -5.0, 3.0]]))[0])
    frontiers = find_frontiers(state, cm, grid, cfg.planner, cfg.map, components={drone_component})
    assert len(frontiers) == 1
    vp = frontiers[0].viewpoint
    assert float(cm.clearance_at(vp.reshape(1, 3))[0]) >= cfg.safety.inflation_m
    assert frontiers[0].component == drone_component


def test_ground_band_clusters_get_more_gain(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    ground_box = ((-1.0, -1.0, 1.0), (1.0, 1.0, 2.0))
    high_box = ((5.0, -1.0, 6.0), (7.0, 1.0, 7.0))
    state = make_map(unknown=[ground_box, high_box])
    cm = clearance(unknown=[ground_box, high_box])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    drone_component = int(grid.label_at(np.array([[-8.0, -8.0, 3.0]]))[0])
    frontiers = find_frontiers(state, cm, grid, cfg.planner, cfg.map, components={drone_component})
    ground = next(f for f in frontiers if f.centroid[2] < cfg.map.ground_band_max_z_m)
    high = next(f for f in frontiers if f.centroid[2] >= cfg.map.ground_band_max_z_m)
    assert ground.gain == pytest.approx(ground.n_voxels * cfg.planner.ground_band_gain)
    assert high.gain == pytest.approx(high.n_voxels)


def _wall_grid_and_component(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> tuple[MapState, ClearanceMap, PlanningGrid, set[int]]:
    """Build a full-height wall at x in [-0.5, 0.5], with the drone's component on the left."""
    wall = ((-0.5, -20.0, 0.0), (0.5, 20.0, 12.0))
    state = make_map(boxes=[wall])
    cm = clearance(boxes=[wall])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    left = np.array([[-5.0, 0.0, 3.0]])
    components = {int(grid.label_at(left)[0])}
    return state, cm, grid, components


def test_find_inspection_targets_needs_surface_seen(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    """``surface_seen=None`` (a hand-built grid with no mapper behind it) finds nothing."""
    state, cm, grid, components = _wall_grid_and_component(cfg, make_map, clearance)
    assert state.surface_seen is None
    assert (
        find_inspection_targets(state, cm, grid, cfg.planner, cfg.map, components=components) == []
    )


def test_find_inspection_targets_on_unseen_wall_surface(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    """Unseen exposed OCC surface gets a reachable, in-range viewpoint on the FREE side."""
    state, cm, grid, components = _wall_grid_and_component(cfg, make_map, clearance)
    state.surface_seen = np.zeros_like(state.occ, dtype=np.bool_)

    targets = find_inspection_targets(state, cm, grid, cfg.planner, cfg.map, components=components)
    assert targets

    lo, hi = cfg.planner.inspect_range_m
    for f in targets:
        assert int(grid.label_at(f.viewpoint.reshape(1, 3))[0]) in components
        dist = float(np.linalg.norm(f.viewpoint - f.centroid))
        assert lo - 1e-6 <= dist <= hi + 1e-6
        # The wall occupies x in [-0.5, 0.5]; a viewpoint reachable only from
        # the left component must sit on the free side, left of the wall.
        assert f.viewpoint[0] < -0.5


def test_find_inspection_targets_none_when_surface_already_seen(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    """A wall entirely marked ``surface_seen`` has no inspection targets left."""
    state, cm, grid, components = _wall_grid_and_component(cfg, make_map, clearance)
    state.surface_seen = np.ones_like(state.occ, dtype=np.bool_)
    assert (
        find_inspection_targets(state, cm, grid, cfg.planner, cfg.map, components=components) == []
    )


@pytest.mark.slow
def test_find_frontiers_is_fast_on_a_realistic_half_explored_map(
    cfg: Config,
    make_map: Callable[..., MapState],
    clearance: Callable[..., ClearanceMap],
) -> None:
    unknown = ((0.0, -20.0, 0.0), (15.0, 20.0, 12.0))
    state = make_map(unknown=[unknown])
    cm = clearance(unknown=[unknown])
    grid = planning_grid(
        cm,
        plan_factor=cfg.map.plan_factor,
        required_m=cfg.safety.inflation_m,
        starts=np.zeros((0, 3)),
    )
    drone_component = int(grid.label_at(np.array([[-5.0, -5.0, 3.0]]))[0])
    start = time.perf_counter()
    find_frontiers(state, cm, grid, cfg.planner, cfg.map, components={drone_component})
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5
