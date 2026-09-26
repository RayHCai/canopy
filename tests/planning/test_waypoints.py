"""Orbit geometry and the local waypoint follower."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from canopy.config import Config, DemoCfg
from canopy.errors import CanopyError
from canopy.planning.waypoints import WaypointFollower, demo_path, orbit_ring, path_length


def test_orbit_ring_shape_and_geometry() -> None:
    center = np.array([1.0, -2.0, 0.0])
    ring = orbit_ring(center, radius=5.0, altitude=2.0, n_waypoints=24)
    assert ring.shape == (24, 3)
    assert np.allclose(ring[:, 2], 2.0)
    radii = np.linalg.norm(ring[:, :2] - center[:2], axis=1)
    assert radii == pytest.approx(5.0)


def test_orbit_ring_is_open_so_laps_concatenate() -> None:
    """Repeating the first point would make the drone stop mid-lap."""
    ring = orbit_ring(np.zeros(3), radius=3.0, altitude=1.0, n_waypoints=8)
    assert not np.allclose(ring[0], ring[-1])


def test_orbit_ring_is_counter_clockwise() -> None:
    ring = orbit_ring(np.zeros(3), radius=1.0, altitude=0.0, n_waypoints=4)
    assert ring[0] == pytest.approx([1.0, 0.0, 0.0])
    assert ring[1] == pytest.approx([0.0, 1.0, 0.0])


def test_orbit_ring_start_angle_offsets_the_ring() -> None:
    ring = orbit_ring(np.zeros(3), 1.0, 0.0, 4, start_angle=np.pi / 2)
    assert ring[0] == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.parametrize(("n", "radius"), [(2, 5.0), (24, 0.0), (24, -1.0)])
def test_orbit_ring_rejects_degenerate_input(n: int, radius: float) -> None:
    with pytest.raises(CanopyError):
        orbit_ring(np.zeros(3), radius, 2.0, n)


def test_demo_path_climbs_then_orbits(cfg: Config) -> None:
    path = demo_path(cfg.demo)
    assert path.shape == (1 + cfg.demo.laps * cfg.demo.orbit_waypoints, 3)
    # First waypoint is straight up from home.
    assert path[0] == pytest.approx([*cfg.demo.home[:2], cfg.demo.takeoff_altitude_m])
    assert np.allclose(path[:, 2], cfg.demo.takeoff_altitude_m)


def test_demo_path_rejects_zero_laps(cfg: Config) -> None:
    with pytest.raises(CanopyError, match=re.escape("demo.laps")):
        demo_path(dataclasses.replace(cfg.demo, laps=0))


def test_path_length() -> None:
    path = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 5.0]])
    assert path_length(path) == pytest.approx(10.0)
    assert path_length(path[:1]) == pytest.approx(0.0)


def _line(n: int) -> np.ndarray:
    return np.array([[float(i), 0.0, 0.0] for i in range(n)])


def test_follower_advances_on_arrival() -> None:
    follower = WaypointFollower(_line(3), tolerance_m=0.1)
    assert follower.target() == pytest.approx([0.0, 0.0, 0.0])
    target = follower.update(np.array([0.0, 0.0, 0.0]))
    assert target == pytest.approx([1.0, 0.0, 0.0])
    assert follower.reached_count == 1


def test_follower_skips_several_waypoints_in_one_tick() -> None:
    """A fast drone can overfly closely spaced waypoints inside one control tick."""
    follower = WaypointFollower(_line(5), tolerance_m=2.5)
    target = follower.update(np.array([0.0, 0.0, 0.0]))
    assert follower.reached_count == 3
    assert target == pytest.approx([3.0, 0.0, 0.0])


def test_follower_reports_done_and_returns_no_target() -> None:
    follower = WaypointFollower(_line(2), tolerance_m=0.5)
    follower.update(np.array([0.0, 0.0, 0.0]))
    # Checked via reached_count rather than `not follower.done`: mypy narrows a
    # bool property to Literal[False] and does not invalidate that on the
    # intervening update(), which would make the rest of the test unreachable.
    assert follower.reached_count == 1
    follower.update(np.array([1.0, 0.0, 0.0]))
    assert follower.reached_count == 2
    assert follower.done
    assert follower.target() is None


def test_follower_does_not_advance_when_far_away() -> None:
    follower = WaypointFollower(_line(3), tolerance_m=0.1)
    follower.update(np.array([0.5, 0.0, 0.0]))
    assert follower.reached_count == 0


@pytest.mark.parametrize("tolerance", [0.0, -1.0])
def test_follower_rejects_bad_tolerance(tolerance: float) -> None:
    with pytest.raises(CanopyError):
        WaypointFollower(_line(2), tolerance_m=tolerance)


def test_follower_rejects_empty_path() -> None:
    with pytest.raises(CanopyError, match="empty path"):
        WaypointFollower(np.zeros((0, 3)), tolerance_m=0.5)


def test_demo_cfg_home_xyz_is_float64(cfg: Config) -> None:
    demo: DemoCfg = cfg.demo
    assert demo.home_xyz.dtype == np.float64
