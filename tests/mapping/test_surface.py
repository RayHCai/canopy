"""Tests for :mod:`canopy.mapping.surface`.

Every observation here is a synthetic ray/plane intersection against a flat
wall, not a mesh: exactly the azimuth-major range-image layout a real lidar
reports (:attr:`~canopy.contracts.Observation.grid_shape`), so the test can
say precisely which normal a return *should* get.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from canopy.config import Config
from canopy.contracts import Observation
from canopy.mapping.surface import SurfaceTracker, estimate_normals


def _grid_dirs(
    n_az: int, n_el: int, el_lo_deg: float, el_hi_deg: float, az_centre_deg: float = 0.0
) -> npt.NDArray[np.float64]:
    """Azimuth-major unit direction grid, matching a real scan's ray layout."""
    az = np.deg2rad(az_centre_deg) + np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    el = np.deg2rad(np.linspace(el_lo_deg, el_hi_deg, n_el))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    return np.stack(
        [
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)


def _wall_observation(
    *,
    origin: npt.NDArray[np.float64] | None = None,
    wall_y: float = 5.0,
    n_az: int = 24,
    n_el: int = 9,
    max_range_m: float = 12.0,
    forward_dy: float = 0.2,
) -> tuple[Observation, tuple[int, int]]:
    """Build a full-circle sweep of an infinite flat wall at ``y = wall_y``; most rays miss.

    Only azimuths whose ray points usefully toward the wall (``dy`` past
    ``forward_dy``) hit; the rest read ``inf``, exactly like a real sweep of a
    wall that fills only part of the view.
    """
    origin = np.array([0.0, 0.0, 1.5]) if origin is None else origin
    dirs = _grid_dirs(n_az, n_el, -45.0, 45.0)
    dy = dirs[:, 1]
    forward = dy > forward_dy
    dist = np.where(forward, (wall_y - origin[1]) / np.where(forward, dy, 1.0), np.inf)
    dist = np.where(dist <= max_range_m, dist, np.inf)
    obs = Observation(
        drone_id=0, t=0.0, origin=origin, dirs=dirs, dist=dist, grid_shape=(n_az, n_el)
    )
    return obs, (n_az, n_el)


# ---------------------------------------------------------------------------
# estimate_normals
# ---------------------------------------------------------------------------
def test_estimate_normals_of_a_flat_wall_match_the_true_normal_within_a_few_degrees() -> None:
    """Every valid return's normal is within a few degrees of the wall's true ``+-y``."""
    obs, _ = _wall_observation()
    normals, valid = estimate_normals(obs)
    assert valid.any()

    cos_to_true_normal = np.abs(normals[valid, 1])
    angle_deg = np.degrees(np.arccos(np.clip(cos_to_true_normal, 0.0, 1.0)))
    assert np.all(angle_deg < 5.0)


def test_estimate_normals_of_a_ground_plane_match_the_true_normal_within_a_few_degrees() -> None:
    """The same check against a horizontal ground plane, true normal ``+-z``."""
    origin = np.array([0.0, 0.0, 3.0])
    n_az, n_el = 24, 9
    dirs = _grid_dirs(n_az, n_el, -80.0, -10.0)
    dz = dirs[:, 2]
    dist = np.where(dz < 0.0, -origin[2] / dz, np.inf)
    obs = Observation(
        drone_id=0, t=0.0, origin=origin, dirs=dirs, dist=dist, grid_shape=(n_az, n_el)
    )

    normals, valid = estimate_normals(obs)
    assert valid.any()
    cos_to_true_normal = np.abs(normals[valid, 2])
    angle_deg = np.degrees(np.arccos(np.clip(cos_to_true_normal, 0.0, 1.0)))
    assert np.all(angle_deg < 5.0)


def test_estimate_normals_without_grid_shape_finds_nothing_valid() -> None:
    """A hand-built scan with no ray layout has no neighbours, so no normal is valid."""
    obs = Observation(
        drone_id=0,
        t=0.0,
        origin=np.zeros(3),
        dirs=np.array([[1.0, 0.0, 0.0]]),
        dist=np.array([2.0]),
    )
    normals, valid = estimate_normals(obs)
    assert normals.shape == (1, 3)
    assert not valid.any()


def test_estimate_normals_top_and_bottom_elevation_rows_are_invalid() -> None:
    """The first and last elevation row have no neighbour on one side."""
    obs, (n_az, n_el) = _wall_observation()
    _, valid = estimate_normals(obs)
    valid_grid = valid.reshape(n_az, n_el)
    assert not valid_grid[:, 0].any()
    assert not valid_grid[:, -1].any()


def test_estimate_normals_a_return_next_to_a_miss_is_invalid() -> None:
    """Azimuth wraps, but a return whose azimuth neighbour missed still gets no normal."""
    obs, (n_az, n_el) = _wall_observation()
    hit = np.isfinite(obs.dist).reshape(n_az, n_el)
    _, valid = estimate_normals(obs)
    valid_grid = valid.reshape(n_az, n_el)

    left = np.roll(hit, 1, axis=0)
    right = np.roll(hit, -1, axis=0)
    # Every valid return's azimuth neighbours (and itself) must be hits too.
    assert np.all(~valid_grid | (left & right & hit))
    # And the miss region contributes at least one such invalid boundary return,
    # so this isn't vacuously true because the whole grid hit.
    assert not hit.all()


# ---------------------------------------------------------------------------
# SurfaceTracker
# ---------------------------------------------------------------------------
def _tracker(cfg: Config) -> SurfaceTracker:
    """Build a tracker over a 20 x 20 x 12 m grid, ample for the patches below."""
    origin = np.array([-10.0, -10.0, 0.0])
    voxel = cfg.map.voxel_m
    nx, ny, nz = np.round(np.array([20.0, 20.0, 12.0]) / voxel).astype(int)
    return SurfaceTracker(cfg.map, origin, voxel, (int(nx), int(ny), int(nz)))


def _small_wall_patch(
    origin: npt.NDArray[np.float64], wall_y: float, az_centre_deg: float
) -> Observation:
    """Build a tiny 3 x 3 azimuth/elevation patch whose centre ray is aimed at a flat wall."""
    az = np.deg2rad(az_centre_deg + np.array([-5.0, 0.0, 5.0]))
    el = np.deg2rad(np.array([-5.0, 0.0, 5.0]))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    dirs = np.stack(
        [
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    dy = dirs[:, 1]
    forward = dy > 1e-9
    dist = np.where(forward, (wall_y - origin[1]) / np.where(forward, dy, 1.0), np.inf)
    return Observation(drone_id=0, t=0.0, origin=origin, dirs=dirs, dist=dist, grid_shape=(3, 3))


def test_surface_tracker_marks_close_square_on_returns(cfg: Config) -> None:
    """A wall seen close and near square-on gets its voxel marked."""
    obs = _small_wall_patch(np.array([0.0, 0.0, 1.5]), wall_y=5.0, az_centre_deg=90.0)
    tracker = _tracker(cfg)
    tracker.integrate(obs)
    assert tracker.seen.any()


def test_surface_tracker_ignores_grazing_incidence(cfg: Config) -> None:
    """The same wall, seen at a shallow angle from 1 m away, never counts.

    Close range alone (well under ``coverage_max_range_m``) is not enough: the
    ray direction here is within about 10 degrees of the wall plane itself.
    """
    obs = _small_wall_patch(np.array([0.0, 4.0, 1.5]), wall_y=5.0, az_centre_deg=10.0)
    tracker = _tracker(cfg)
    tracker.integrate(obs)
    assert not tracker.seen.any()


def test_surface_tracker_ignores_returns_beyond_coverage_max_range(cfg: Config) -> None:
    """The same square-on wall, seen from beyond ``coverage_max_range_m``, never counts."""
    obs = _small_wall_patch(np.array([0.0, -4.0, 1.5]), wall_y=5.0, az_centre_deg=90.0)
    tracker = _tracker(cfg)
    tracker.integrate(obs)
    assert not tracker.seen.any()
