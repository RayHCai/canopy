"""Tests for :mod:`canopy.mapping.occupancy`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from canopy.config import Config
from canopy.contracts import MapState, Observation, Occ
from canopy.mapping.occupancy import integrate_occupancy, mark_free_box, new_map_state


def _obs(
    origin: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
) -> Observation:
    """Build a minimal :class:`Observation`; occupancy never reads colour or grid layout."""
    return Observation(drone_id=0, t=0.0, origin=origin, dirs=dirs, dist=dist)


def _voxel(state: MapState, point: tuple[float, float, float]) -> tuple[int, int, int]:
    """Index of the voxel containing ``point``, for assertions."""
    idx = np.floor((np.asarray(point) - state.origin) / state.voxel).astype(int)
    return int(idx[0]), int(idx[1]), int(idx[2])


def test_carve_hit_and_unknown_beyond(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """One ray in an empty room carves FREE, marks the wall OCC, leaves beyond UNKNOWN."""
    state = new_map_state(lot, cfg.map)
    origin = np.array([-10.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    dist = np.array([5.0])
    changed = integrate_occupancy(state, _obs(origin, dirs, dist), cfg.map, cfg.sensor.max_range_m)
    assert changed

    near = _voxel(state, (origin[0] + 1.0, origin[1], origin[2]))
    assert state.occ[near] == Occ.FREE

    hit = _voxel(state, (origin[0] + 5.0, origin[1], origin[2]))
    assert state.occ[hit] == Occ.OCC

    beyond = _voxel(state, (origin[0] + 5.5, origin[1], origin[2]))
    assert state.occ[beyond] == Occ.UNKNOWN


def test_occ_never_downgraded_to_free(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """A later miss ray carving through a mapped wall must not erase it."""
    state = new_map_state(lot, cfg.map)
    origin = np.array([-10.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    hit_dist = np.array([5.0])
    integrate_occupancy(state, _obs(origin, dirs, hit_dist), cfg.map, cfg.sensor.max_range_m)
    hit = _voxel(state, (origin[0] + 5.0, origin[1], origin[2]))
    assert state.occ[hit] == Occ.OCC

    miss = np.array([np.inf])
    integrate_occupancy(state, _obs(origin, dirs, miss), cfg.map, cfg.sensor.max_range_m)
    assert state.occ[hit] == Occ.OCC


def test_points_outside_grid_are_ignored(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """A hit far outside the lot is dropped without error and touches nothing."""
    state = new_map_state(lot, cfg.map)
    before = state.occ.copy()
    origin = np.array([lot[0, 0] - 100.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    dist = np.array([1.0])
    changed = integrate_occupancy(state, _obs(origin, dirs, dist), cfg.map, cfg.sensor.max_range_m)
    assert not changed
    assert np.array_equal(state.occ, before)


def test_mark_free_box_never_touches_occ(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """``mark_free_box`` only promotes UNKNOWN, and only inside the box."""
    state = new_map_state(lot, cfg.map)
    corner = state.origin
    occ_voxel = _voxel(state, (corner[0] + 0.1, corner[1] + 0.1, corner[2] + 0.1))
    state.occ[occ_voxel] = Occ.OCC

    lo = corner
    hi = corner + np.array([1.0, 1.0, 1.0])
    mark_free_box(state, lo, hi)

    assert state.occ[occ_voxel] == Occ.OCC
    free_voxel = _voxel(state, (corner[0] + 0.5, corner[1] + 0.5, corner[2] + 0.5))
    assert state.occ[free_voxel] == Occ.FREE

    outside_voxel = _voxel(state, (corner[0] + 5.0, corner[1] + 5.0, corner[2] + 5.0))
    assert state.occ[outside_voxel] == Occ.UNKNOWN
