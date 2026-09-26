"""Swarm planning: frontiers, assignment, pathing, the mission state machine and safety shield.

:mod:`canopy.planning.waypoints` is the local waypoint geometry and follower
(:class:`WaypointFollower`, :func:`demo_path`, :func:`path_length`), used both
by the ORBIT phase of the mission and by the obstacle-free ``canopy-fly``
check. :mod:`canopy.planning.frontier`, :mod:`canopy.planning.assign` and
:mod:`canopy.planning.pathing` find unexplored space, assign it to drones and
route paths through it; :mod:`canopy.planning.safety` shields those paths from
collisions and geofence violations. :mod:`canopy.planning.mission` composes
all of that into the mission state machine, and :mod:`canopy.planning.run`
wires it to :class:`~canopy.sim.SimWorld` and :class:`~canopy.mapping.Mapper`
for one mapping mission end to end (:class:`MissionRun`).

Public API: :class:`MissionRun`, :class:`WaypointFollower`, :func:`demo_path`,
:func:`path_length`.
"""

from __future__ import annotations

from canopy.planning.run import MissionRun
from canopy.planning.waypoints import WaypointFollower, demo_path, path_length

__all__ = ["MissionRun", "WaypointFollower", "demo_path", "path_length"]
