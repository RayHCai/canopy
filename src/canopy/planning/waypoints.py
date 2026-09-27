"""Waypoint geometry and the local follower.

:func:`orbit_ring` and :func:`demo_path` are the obstacle-free flight check
(``canopy-fly``): one drone climbs to altitude and flies rings around the
launch pad, with no scene, sensing or mission logic involved. The mission
state machine (:mod:`canopy.planning.mission`) uses :class:`WaypointFollower`
directly for its own paths, not these ring helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from canopy.contracts import Points, Vec3
from canopy.errors import CanopyError
from canopy.mathutil import norm

if TYPE_CHECKING:
    from canopy.config import DemoCfg

__all__ = ["WaypointFollower", "demo_path", "orbit_ring", "path_length"]

#: Fewer than this and a "ring" is not a closed loop worth flying.
_MIN_ORBIT_WAYPOINTS = 3


def orbit_ring(
    center: Vec3,
    radius: float,
    altitude: float,
    n_waypoints: int,
    start_angle: float = 0.0,
) -> Points:
    """Waypoints for one counter-clockwise circle at a fixed altitude.

    Parameters
    ----------
    center
        Circle centre; only its X and Y are used.
    radius
        Circle radius in metres.
    altitude
        Constant Z for every waypoint.
    n_waypoints
        Waypoints per lap. The ring is open: the first point is not repeated at
        the end, so consecutive laps concatenate cleanly.
    start_angle
        Angular offset in radians, used to fan a swarm around the ring.

    Returns
    -------
    Points
        Shape ``(n_waypoints, 3)``.
    """
    if n_waypoints < _MIN_ORBIT_WAYPOINTS:
        msg = f"an orbit needs at least {_MIN_ORBIT_WAYPOINTS} waypoints, got {n_waypoints}"
        raise CanopyError(msg)
    if radius <= 0.0:
        msg = f"orbit radius must be positive, got {radius}"
        raise CanopyError(msg)

    angles = start_angle + np.linspace(0.0, 2.0 * np.pi, n_waypoints, endpoint=False)
    ring = np.empty((n_waypoints, 3), dtype=np.float64)
    ring[:, 0] = center[0] + radius * np.cos(angles)
    ring[:, 1] = center[1] + radius * np.sin(angles)
    ring[:, 2] = altitude
    return ring


def demo_path(cfg: DemoCfg) -> Points:
    """Climb from home to the orbit altitude, then fly ``cfg.laps`` laps.

    Returns
    -------
    Points
        Shape ``(1 + laps * orbit_waypoints, 3)``.
    """
    if cfg.laps < 1:
        msg = f"demo.laps must be at least 1, got {cfg.laps}"
        raise CanopyError(msg)

    home = cfg.home_xyz
    climb = np.array([[home[0], home[1], cfg.takeoff_altitude_m]], dtype=np.float64)
    ring = orbit_ring(home, cfg.orbit_radius_m, cfg.takeoff_altitude_m, cfg.orbit_waypoints)
    return np.vstack([climb, np.tile(ring, (cfg.laps, 1))])


def path_length(path: Points) -> float:
    """Total polyline length of ``path`` in metres.

    A path of fewer than two points has no segments, and ``np.diff`` yields an
    empty array for it, so no special case is needed.
    """
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


class WaypointFollower:
    """Walks a drone through a polyline, advancing on arrival.

    This is the classical local follower the optional RL policy would replace.
    It owns no motion model: it only answers "which point am I heading for?".
    """

    def __init__(self, path: Points, tolerance_m: float) -> None:
        if len(path) == 0:
            msg = "cannot follow an empty path"
            raise CanopyError(msg)
        if tolerance_m <= 0.0:
            msg = f"waypoint tolerance must be positive, got {tolerance_m}"
            raise CanopyError(msg)
        self._path = np.asarray(path, dtype=np.float64)
        self._tolerance = tolerance_m
        self._index = 0

    @property
    def index(self) -> int:
        """Index of the waypoint currently being flown to."""
        return self._index

    @property
    def n_waypoints(self) -> int:
        """Total waypoints in the path."""
        return len(self._path)

    @property
    def reached_count(self) -> int:
        """How many waypoints have been reached so far."""
        return self._index

    @property
    def done(self) -> bool:
        """Whether every waypoint has been reached."""
        return self._index >= len(self._path)

    def target(self) -> Vec3 | None:
        """Return the current target, or ``None`` once the path is complete."""
        if self.done:
            return None
        result: Vec3 = self._path[self._index]
        return result

    def update(self, pos: Vec3) -> Vec3 | None:
        """Advance past every waypoint within tolerance of ``pos``, then return the target.

        Advancing in a loop matters: a fast drone can overfly more than one
        closely spaced waypoint inside a single control tick.
        """
        while not self.done and norm(self._path[self._index] - pos) <= self._tolerance:
            self._index += 1
        return self.target()
