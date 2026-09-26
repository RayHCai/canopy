"""Mission controller: one central planner for a swarm of any size.

The controller turns the shared :class:`~canopy.contracts.MapState` into one
:class:`~canopy.contracts.DroneTask` per live drone, and every control tick
turns those tasks into shield-vetted targets for the dynamics. It is the only
place that decides where a drone goes.

Phases are global and linear::

    TAKEOFF -> EXPLORE -> RETURN -> DONE

* **TAKEOFF**: each drone climbs straight up from its pad. The pads' columns
  are operator-surveyed free space (:meth:`MissionController.launch_boxes`);
  without that seed the known-free-only rule would pin a drone on the ground
  forever, since no scan from the pad looks straight up.
* **EXPLORE**: at ``sim.replan_hz``, extract targets -- voxel frontiers and
  mapped-but-unseen surfaces -- match drones to them with the Hungarian
  algorithm and plan paths to their viewpoints. Ends when no reachable target
  is left, at ``sim.timeout_s``, or, if ``planner.done_ground_coverage`` is
  set, when ground-band coverage reaches it. The shipped config leaves it
  unset, so the whole lot is mapped. A drone the assignment leaves without a
  target for ``planner.idle_return_s`` has finished its share: it flies home
  and lands rather than hovering until the others are done. Until it touches
  down it is still in the assignment, so a frontier that opens up in the
  meantime can recall it; once landed it stays down.
* **RETURN**: every drone flies to the hover point above its own pad and lands.
  If something beside the pads leaves that point too tight to plan to, it
  returns to the lowest clear point higher up the same column instead.
  A drone whose battery falls to ``sim.rth_battery`` does this on its own,
  whatever the phase; its frontiers are simply absorbed by the others at the
  next assignment.

Scaling is the size of the cost matrix. Nothing below is special-cased for one
drone: a swarm of one is a one-row assignment, a shield with nobody to yield
to, and a stuck check like any other.

Deconfliction has two layers. The :class:`~canopy.planning.safety.Shield` keeps
drones apart tick by tick, deciding in drone-id order, so a lower id has right
of way. What it cannot do is resolve a standoff -- two drones meeting in a gap
both hold, safely, forever. After ``safety.hold_replan_s`` the controller
breaks the tie the same way the shield ranks drones: the higher id gives way,
first by re-routing around the others and, failing that, by stepping aside off
the right-of-way drone's path.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from canopy import mathutil
from canopy.contracts import DroneState, DroneTask, MapState, Points, Vec3
from canopy.errors import PlanningError
from canopy.log import get_logger
from canopy.planning.assign import assign_frontiers
from canopy.planning.frontier import Frontier, find_frontiers, find_inspection_targets
from canopy.planning.pathing import PlanningGrid, plan_path, planning_grid
from canopy.planning.safety import ClearanceMap, Shield, geofence_box
from canopy.planning.waypoints import WaypointFollower

if TYPE_CHECKING:
    import numpy.typing as npt

    from canopy.config import Config

__all__ = ["MissionController", "Phase"]

_log = get_logger(__name__)

#: Below this speed (m/s) a drone at pad height counts as touched down.
_LANDED_SPEED = 0.05

#: Nearest genuinely passable cells tried for an escape hop. The nearest few
#: are almost always visible; beyond that a straight hop is the wrong tool.
_ESCAPE_TRIES = 12

#: An escape hop counts as arrived within this fraction of a planning cell.
_ESCAPE_SETTLE = 0.2


class Phase(StrEnum):
    """Global mission phase."""

    TAKEOFF = "takeoff"
    EXPLORE = "explore"
    RETURN = "return"
    DONE = "done"


@dataclass(slots=True, eq=False)
class _Drone:
    """The controller's bookkeeping for one drone."""

    pad: Vec3
    task: DroneTask
    follower: WaypointFollower | None = None
    #: ``(t, pos)`` samples over the last ``stuck_window_s``, for stuck detection.
    history: deque[tuple[float, Vec3]] = field(default_factory=deque)
    #: Not eligible for a new frontier until then; set after stepping aside so
    #: it does not fly straight back into the gap it just cleared.
    resting_until: float = -np.inf
    #: Last time a shield standoff was resolved for this drone.
    last_yield_t: float = -np.inf
    #: When the assignment first left this drone without a target; ``None``
    #: while it has one. Idle for ``idle_return_s`` sends it home.
    idle_since: float | None = None
    landed: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class _Mark:
    """A point frontiers are kept away from, until ``expires`` (``inf``: forever)."""

    point: Vec3
    radius: float
    expires: float


class MissionController:
    """Frontier exploration for a swarm, from take-off to touchdown.

    Parameters
    ----------
    cfg
        Validated configuration. Reads ``sim``, ``map``, ``planner`` and
        ``safety``.
    lot_bounds
        ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` of the surveyed lot; the
        geofence is derived from it.
    pads
        Launch pad per drone, shape ``(N, 3)``; drone ``i`` launches from
        ``pads[i]`` and returns there.
    """

    def __init__(self, cfg: Config, lot_bounds: npt.NDArray[np.float64], pads: Points) -> None:
        self._cfg = cfg
        self._geofence = geofence_box(lot_bounds, cfg.safety)
        self._shield = Shield(cfg.safety, cfg.sim)
        self._phase = Phase.TAKEOFF
        self._drones: dict[int, _Drone] = {}
        for i, pad in enumerate(np.asarray(pads, dtype=np.float64)):
            hover = pad + np.array([0.0, 0.0, cfg.planner.takeoff_altitude_m])
            task = DroneTask(i, "takeoff", hover, None, np.vstack([pad, hover]))
            self._drones[i] = _Drone(
                pad=pad.copy(),
                task=task,
                follower=WaypointFollower(task.path, cfg.planner.waypoint_tolerance_m),
            )
        self._clearance: ClearanceMap | None = None
        self._clearance_version = -1
        self._grid: PlanningGrid | None = None
        self._frontiers: list[Frontier] = []
        self._marks: list[_Mark] = []
        self._next_replan = 0.0

    # -- introspection ------------------------------------------------------
    @property
    def phase(self) -> Phase:
        """Current mission phase."""
        return self._phase

    @property
    def done(self) -> bool:
        """Whether every live drone is back on its pad."""
        return self._phase is Phase.DONE

    @property
    def tasks(self) -> dict[int, DroneTask]:
        """Each drone's current task."""
        return {i: d.task for i, d in self._drones.items()}

    @property
    def frontiers(self) -> list[Frontier]:
        """Frontier clusters from the latest replan, after blacklisting."""
        return list(self._frontiers)

    @property
    def landed(self) -> frozenset[int]:
        """Drones that have flown home and touched down on their pads."""
        return frozenset(i for i, d in self._drones.items() if d.landed)

    def launch_boxes(self) -> list[tuple[Vec3, Vec3]]:
        """World boxes the operator has surveyed clear before launch.

        One column per pad, ``launch_clear_radius_m`` wide on each side and
        reaching ``launch_clear_radius_m`` above take-off altitude. Seed the map
        with these (:meth:`canopy.mapping.mapper.Mapper.mark_free_box`) before
        the first :meth:`step`; the ground layer is left for the sensor to mark.
        """
        r = self._cfg.planner.launch_clear_radius_m
        top = self._cfg.planner.takeoff_altitude_m + r
        floor = self._cfg.map.voxel_m
        return [
            (d.pad + np.array([-r, -r, floor]), d.pad + np.array([r, r, top]))
            for d in self._drones.values()
        ]

    # -- the tick -------------------------------------------------------------
    def step(
        self,
        t: float,
        states: Sequence[DroneState],
        map_state: MapState,
        *,
        map_version: int,
        coverage_ground_band: float,
    ) -> dict[int, Vec3]:
        """Advance the mission one control tick and return vetted targets.

        Parameters
        ----------
        t
            Simulated time, seconds.
        states
            Every drone, dead ones included (the shield still avoids them).
        map_state
            The shared map.
        map_version
            Bumped by the mapper whenever ``map_state.occ`` changes; the
            clearance field is rebuilt only when it has.
        coverage_ground_band
            Current ground-band coverage in ``[0, 1]``.

        Returns
        -------
        dict
            Approved target per ``drone_id``, for
            :meth:`~canopy.sim.dynamics.KinematicDynamics.step`. A drone absent
            from it brakes and holds station.
        """
        live = [s for s in states if s.alive and s.drone_id in self._drones]
        for s in states:
            if not s.alive and s.drone_id in self._drones:
                self._retire(s.drone_id)

        if t >= self._next_replan or self._clearance is None:
            self._refresh_clearance(map_state, map_version)
            self._replan(t, live, map_state, coverage_ground_band)
            self._next_replan = t + 1.0 / self._cfg.sim.replan_hz
        clearance = self._clearance
        if clearance is None:  # pragma: no cover - set by _refresh_clearance above
            msg = "clearance map missing after refresh"
            raise PlanningError(msg)

        targets = self._follow(t, live)
        landing = [i for i, d in self._drones.items() if d.task.kind == "land"]
        verdict = self._shield.filter(states, targets, clearance, landing=landing)
        self._resolve_holds(t, live, verdict.held)
        self._check_stuck(t, live)
        self._check_phase(live)
        return verdict.targets

    # -- replanning -----------------------------------------------------------
    def _refresh_clearance(self, map_state: MapState, map_version: int) -> None:
        if self._clearance is None or map_version != self._clearance_version:
            self._clearance = ClearanceMap.from_map(map_state, self._geofence)
            self._clearance_version = map_version

    def _replan(
        self, t: float, live: list[DroneState], map_state: MapState, coverage: float
    ) -> None:
        """Make the 1 Hz decision: escapes, battery, termination, assignment, paths."""
        clearance = self._clearance
        if clearance is None:  # pragma: no cover - set by _refresh_clearance
            return
        starts = np.array([s.pos for s in live], dtype=np.float64).reshape(-1, 3)
        self._grid = planning_grid(
            clearance,
            plan_factor=self._cfg.map.plan_factor,
            required_m=self._cfg.safety.inflation_m,
            starts=starts,
        )
        self._free_trapped(t, live)
        self._send_home_if_due(live)
        if self._phase is not Phase.EXPLORE:
            return

        # A drone flying home on its own (not on low battery, which the battery
        # test below excludes) is still an explorer: new work can recall it.
        explorers = [
            s
            for s in live
            if self._drones[s.drone_id].task.kind in ("frontier", "hold", "rth")
            and not self._drones[s.drone_id].landed
            and self._drones[s.drone_id].resting_until <= t
            and s.battery > self._cfg.sim.rth_battery
        ]
        self._frontiers = self._find_frontiers(t, live, map_state)
        reason = self._termination(t, coverage)
        if reason is not None:
            _log.info("exploration done at t=%.1f s: %s", t, reason)
            self._phase = Phase.RETURN
            self._frontiers = []
            for s in live:
                if not self._drones[s.drone_id].landed:
                    self._start_rth(s)
            return
        if not explorers:
            return

        grid = self._grid
        positions = {s.drone_id: s.pos for s in explorers}
        components = {i: int(grid.label_at(p[None, :])[0]) for i, p in positions.items()}
        current = {
            i: self._drones[i].task.goal
            for i in positions
            if self._drones[i].task.kind == "frontier"
        }
        chosen = assign_frontiers(
            positions, self._frontiers, current, self._cfg.planner, components=components
        )
        for s in explorers:
            idx = chosen.get(s.drone_id)
            if idx is None:
                self._idle(t, s)
            else:
                self._drones[s.drone_id].idle_since = None
                self._pursue(t, s, self._frontiers[idx])

    def _idle(self, t: float, state: DroneState) -> None:
        """Hold a drone the assignment has no target for, and send it home once it is done.

        The grace period is what keeps a momentary shortage -- more drones than
        targets for one replan, typically just after take-off -- from grounding
        a drone for the rest of the mission.
        """
        d = self._drones[state.drone_id]
        if d.task.kind == "rth":
            return  # already on its way; re-planning would only restart the path
        if d.idle_since is None:
            d.idle_since = t
        if t - d.idle_since >= self._cfg.planner.idle_return_s:
            _log.info("drone %d has no work left: returning home", state.drone_id)
            self._start_rth(state)
        else:
            self._hold(state.drone_id)

    def _free_trapped(self, t: float, live: list[DroneState]) -> None:
        """Hop drones out of planning cells that are passable only because they are in them.

        A drone can end up clear at its exact position but inside a margin at
        planning resolution: parked under the ceiling, or beside a roof or
        wall that the map filled in after it arrived. Its cell is then forced
        passable, so it forms a component of one cell that no target is in,
        and the first leg of any planned path is unchecked, which the shield
        then refuses. Nothing downstream can fix that, so it is fixed first.
        The drone takes the shortest straight hop the shield's own map rule
        accepts to a cell that is passable in its own right.
        """
        grid, clearance = self._grid, self._clearance
        if grid is None or grid.forced is None or clearance is None:
            return
        if self._phase is Phase.TAKEOFF:
            return  # the launch column is seeded free; the pads are meant to be forced
        forced_at = grid.forced
        genuine = grid.centres(grid.passable & ~forced_at)
        if len(genuine) == 0:
            return
        required = self._cfg.safety.inflation_m
        for s in live:
            d = self._drones[s.drone_id]
            if d.landed or d.task.kind in ("land", "escape", "takeoff"):
                continue
            idx = np.floor((s.pos - grid.origin) / grid.cell).astype(np.intp)
            if np.any(idx < 0) or np.any(idx >= np.asarray(forced_at.shape)):
                continue
            if not forced_at[tuple(idx)]:
                continue
            order = np.argsort(np.linalg.norm(genuine - s.pos, axis=1))
            for j in order[:_ESCAPE_TRIES]:
                if clearance.segment_clear(s.pos, genuine[j], required):
                    _log.info(
                        "drone %d is boxed in at planning resolution; stepping out", s.drone_id
                    )
                    if d.task.kind == "frontier":
                        self._blacklist(t, d.task.goal)
                    path = np.vstack([s.pos, genuine[j]])
                    self._assign(
                        s.drone_id,
                        DroneTask(s.drone_id, "escape", genuine[j], None, path),
                        # Settle well inside the target cell: the usual waypoint
                        # tolerance is most of a cell wide, and "arriving" in the
                        # forced neighbour would just trigger another escape.
                        tolerance_m=_ESCAPE_SETTLE * grid.cell,
                    )
                    d.history.clear()
                    break

    def _send_home_if_due(self, live: list[DroneState]) -> None:
        """Start a return for low-battery drones, and retry returns that failed to plan."""
        for s in live:
            d = self._drones[s.drone_id]
            if d.landed or d.task.kind in ("rth", "land"):
                continue
            if s.battery <= self._cfg.sim.rth_battery:
                _log.info("drone %d battery %.2f: returning home", s.drone_id, s.battery)
                self._start_rth(s)
            elif d.task.kind == "hold" and self._phase is Phase.RETURN:
                self._start_rth(s)

    def _find_frontiers(
        self, t: float, live: list[DroneState], map_state: MapState
    ) -> list[Frontier]:
        """Reachable targets, minus spent and blacklisted ones.

        Targets are voxel frontiers (unknown space to explore) and inspection
        targets (known surface not yet seen at photo quality) in one pool. The
        assignment weighs them on the same distance-versus-gain scale, so
        drones naturally inspect whatever is near while exploration continues
        elsewhere.
        """
        clearance, grid = self._clearance, self._grid
        if clearance is None or grid is None:  # pragma: no cover - set by _replan
            return []
        starts = np.array([s.pos for s in live], dtype=np.float64).reshape(-1, 3)
        components = {int(c) for c in grid.label_at(starts)} - {0}
        args = (map_state, clearance, grid, self._cfg.planner, self._cfg.map)
        found = find_frontiers(*args, components=components)
        found += find_inspection_targets(*args, components=components)
        self._marks = [m for m in self._marks if m.expires > t]
        if not self._marks or not found:
            return found
        points = np.array([m.point for m in self._marks])
        radii = np.array([m.radius for m in self._marks])
        # A target is spent if either where it is looked from or what it looks
        # at has been marked; the second stops a surface that cannot be seen
        # well from anywhere reachable from cycling through viewpoints forever.
        blocked = np.zeros(len(found), dtype=np.bool_)
        for attr in ("viewpoint", "centroid"):
            where = np.array([getattr(f, attr) for f in found])
            gaps = np.linalg.norm(where[:, None, :] - points[None, :, :], axis=-1)
            blocked |= (gaps < radii[None, :]).any(axis=1)
        return [f for f, spent in zip(found, blocked, strict=True) if not spent]

    def _termination(self, t: float, coverage: float) -> str | None:
        """Why exploration should stop now, or ``None`` to carry on."""
        threshold = self._cfg.planner.done_ground_coverage
        if threshold is not None and coverage >= threshold:
            return f"ground-band coverage {coverage:.3f}"
        if not self._frontiers:
            return "no reachable frontiers"
        if t >= self._cfg.sim.timeout_s:
            return f"timeout after {self._cfg.sim.timeout_s:.0f} s"
        return None

    def _pursue(self, t: float, state: DroneState, frontier: Frontier) -> None:
        """Send a drone to ``frontier``, re-planning only if the goal moved or the path broke."""
        d = self._drones[state.drone_id]
        goal = frontier.viewpoint
        same_goal = (
            d.task.kind == "frontier"
            and mathutil.norm(d.task.goal - goal) < self._cfg.map.plan_voxel_m
        )
        if same_goal and d.follower is not None and self._path_still_clear(state, d.follower):
            d.task = DroneTask(
                state.drone_id, "frontier", d.task.goal, frontier.centroid, d.task.path
            )
            return
        path = self._plan(state.pos, goal)
        if path is None:
            self._blacklist(t, goal)
            self._hold(state.drone_id)
            return
        if not same_goal:
            # Re-planning toward the same goal is not progress, so it must not
            # reset the stuck window; a drone the shield keeps refusing would
            # otherwise re-plan forever and never be declared stuck.
            d.history.clear()
        self._assign(
            state.drone_id, DroneTask(state.drone_id, "frontier", goal, frontier.centroid, path)
        )

    def _path_still_clear(self, state: DroneState, follower: WaypointFollower) -> bool:
        """Whether the rest of a drone's path is still known-free on the new map."""
        clearance = self._clearance
        if clearance is None or follower.done:
            return False
        path = self._drones[state.drone_id].task.path
        rest = np.vstack([state.pos, path[follower.index :]])
        required = self._cfg.safety.inflation_m
        return all(clearance.segment_clear(a, b, required) for a, b in pairwise(rest))

    def _plan(self, start: Vec3, goal: Vec3, avoid: Points | None = None) -> Points | None:
        """``plan_path`` with this mission's settings; ``None`` when no path exists."""
        clearance = self._clearance
        if clearance is None:  # pragma: no cover - set by _refresh_clearance
            return None
        try:
            return plan_path(
                start,
                goal,
                clearance,
                plan_factor=self._cfg.map.plan_factor,
                required_m=self._cfg.safety.inflation_m,
                avoid=avoid,
                avoid_radius_m=self._cfg.safety.min_separation_m,
            )
        except PlanningError as exc:
            _log.debug("no path: %s", exc)
            return None

    # -- task bookkeeping -----------------------------------------------------
    def _assign(self, drone_id: int, task: DroneTask, *, tolerance_m: float | None = None) -> None:
        d = self._drones[drone_id]
        d.task = task
        tolerance = self._cfg.planner.waypoint_tolerance_m if tolerance_m is None else tolerance_m
        d.follower = WaypointFollower(task.path, tolerance)

    def _hold(self, drone_id: int) -> None:
        d = self._drones[drone_id]
        pos = d.task.goal if d.follower is None else d.task.path[-1]
        d.task = DroneTask(drone_id, "hold", pos, None, np.empty((0, 3), dtype=np.float64))
        d.follower = None

    def _retire(self, drone_id: int) -> None:
        """Forget a dead drone's task; the next assignment absorbs its frontier."""
        d = self._drones[drone_id]
        if d.task.kind != "hold" or d.follower is not None:
            _log.info("drone %d lost; its task is released", drone_id)
            self._hold(drone_id)

    def _start_rth(self, state: DroneState) -> None:
        """Fly to a hover point above this drone's pad (then land, on arrival)."""
        d = self._drones[state.drone_id]
        tolerance = self._cfg.planner.waypoint_tolerance_m
        hover = self._landing_hover(state)
        if hover is None:
            _log.debug("drone %d: no reachable hover above its pad yet", state.drone_id)
            self._hold(state.drone_id)  # retried at the next replan
            return
        # Already in the column below the hover: the take-off column, or one
        # just checked clear, so there is nothing left to plan.
        in_column = mathutil.norm(state.pos[:2] - d.pad[:2]) <= tolerance
        if in_column and state.pos[2] <= hover[2] + tolerance:
            self._start_landing(state.drone_id, state.pos)
            return
        path = self._plan(state.pos, hover)
        if path is None:
            self._hold(state.drone_id)  # retried at the next replan
            return
        self._assign(state.drone_id, DroneTask(state.drone_id, "rth", hover, None, path))
        d.history.clear()

    def _landing_hover(self, state: DroneState) -> Vec3 | None:
        """Find the lowest point above this drone's pad it can plan to and land from.

        Normally the take-off hover. But anything standing beside the pads --
        a front fence, a hedge -- leaves that point inside the planning margin,
        and a return aimed at it can never be planned: the drone would hold in
        RETURN forever. Take-off never meets this, because it climbs straight
        up rather than planning. So look higher up the same column, one planning
        cell at a time, for a point with full clearance in the drone's own
        component, and require the drop from it back to the take-off hover to
        keep the drone's body clear: the landing is exempt from the shield's map
        check on the understanding that it is a vertical drop through space
        known to be clear.
        """
        clearance, grid = self._clearance, self._grid
        if clearance is None or grid is None:  # pragma: no cover - set by _replan
            return None
        label = int(grid.label_at(state.pos[None, :])[0])
        if label == 0:
            return None
        pad = self._drones[state.drone_id].pad
        base = pad + np.array([0.0, 0.0, self._cfg.planner.takeoff_altitude_m])
        heights = np.arange(base[2], clearance.geofence[1, 2], grid.cell)
        column = np.repeat(base[None, :], len(heights), axis=0)
        column[:, 2] = heights
        ok = (clearance.clearance_at(column) >= self._cfg.safety.inflation_m) & (
            grid.label_at(column) == label
        )
        body = self._cfg.safety.drone_radius_m
        for hover in column[ok]:
            if clearance.segment_clear(hover, base, body, require_gain=False):
                result: Vec3 = hover
                return result
        return None

    def _start_landing(self, drone_id: int, hover: Vec3) -> None:
        """Descend vertically from ``hover``, directly above the pad, to touchdown."""
        d = self._drones[drone_id]
        # A landing is not followed waypoint by waypoint: the pad is commanded
        # until touchdown, so the follower's arrival tolerance cannot leave the
        # drone hovering a few decimetres up.
        d.task = DroneTask(drone_id, "land", d.pad.copy(), None, np.vstack([hover, d.pad]))
        d.follower = None

    def _blacklist(self, t: float, point: Vec3) -> None:
        planner = self._cfg.planner
        self._marks.append(
            _Mark(point.copy(), planner.goal_conflict_radius_m, t + planner.blacklist_s)
        )

    def _mark_visited(self, point: Vec3) -> None:
        """Spend a reached viewpoint: what could be seen from there has been."""
        self._marks.append(_Mark(point.copy(), self._cfg.planner.visited_radius_m, np.inf))

    # -- per tick -----------------------------------------------------------
    def _follow(self, t: float, live: list[DroneState]) -> dict[int, Vec3]:
        """Each drone's current waypoint, handling arrivals."""
        targets: dict[int, Vec3] = {}
        for s in live:
            d = self._drones[s.drone_id]
            if d.task.kind == "land":
                if s.pos[2] <= self._cfg.planner.landed_altitude_m and (
                    mathutil.norm(s.vel) < _LANDED_SPEED
                ):
                    d.landed = True
                    d.task = DroneTask(s.drone_id, "hold", d.pad.copy(), None, np.empty((0, 3)))
                else:
                    targets[s.drone_id] = d.task.goal
                continue
            if d.follower is None:
                continue
            target = d.follower.update(s.pos)
            if target is None:
                self._arrive(t, s)
                continue
            targets[s.drone_id] = target
        return targets

    def _arrive(self, t: float, state: DroneState) -> None:
        d = self._drones[state.drone_id]
        kind = d.task.kind
        if kind == "frontier":
            self._mark_visited(d.task.goal)
            if d.task.look_at is not None:
                # The target itself is spent too, but only its own bucket: half
                # a bucket keeps the neighbouring surface patches in play.
                radius = 0.5 * self._cfg.planner.inspect_bucket_m
                self._marks.append(_Mark(d.task.look_at.copy(), radius, np.inf))
            self._hold(state.drone_id)
            self._next_replan = t  # a free drone should not idle until the next tick
        elif kind == "rth":
            self._start_landing(state.drone_id, d.task.goal)
        elif kind == "yield":
            d.resting_until = t + self._cfg.safety.hold_replan_s
            self._hold(state.drone_id)
        elif kind == "escape":
            self._hold(state.drone_id)
        else:  # takeoff
            self._hold(state.drone_id)

    def _check_phase(self, live: list[DroneState]) -> None:
        if self._phase is Phase.TAKEOFF and all(
            self._drones[s.drone_id].task.kind != "takeoff" for s in live
        ):
            _log.info("all drones airborne: exploring")
            self._phase = Phase.EXPLORE
            self._next_replan = 0.0
        elif self._phase is Phase.RETURN and all(self._drones[s.drone_id].landed for s in live):
            _log.info("all drones home")
            self._phase = Phase.DONE

    def _check_stuck(self, t: float, live: list[DroneState]) -> None:
        """Blacklist a goal a drone has made no progress toward for a whole window."""
        window = self._cfg.safety.stuck_window_s
        for s in live:
            d = self._drones[s.drone_id]
            if d.task.kind not in ("frontier", "rth", "yield", "escape"):
                d.history.clear()
                continue
            d.history.append((t, s.pos.copy()))
            while d.history and d.history[0][0] < t - window:
                d.history.popleft()
            t0, p0 = d.history[0]
            if t - t0 < window - self._cfg.sim.dt:
                continue
            if mathutil.norm(s.pos - p0) >= self._cfg.safety.stuck_min_progress_m:
                continue
            _log.info("drone %d stuck on %s task; replanning", s.drone_id, d.task.kind)
            d.history.clear()
            if d.task.kind == "frontier":
                self._blacklist(t, d.task.goal)
            # A held drone is re-tasked at the next replan: a new frontier, or
            # (in RETURN, or on low battery) a fresh attempt at the way home.
            self._hold(s.drone_id)
            self._next_replan = t

    # -- standoffs ------------------------------------------------------------
    def _resolve_holds(self, t: float, live: list[DroneState], held: dict[int, str]) -> None:
        """Break standoffs the shield has held for ``hold_replan_s``.

        A ``"map"`` hold means the drone's path now crosses space the map no
        longer calls free: re-plan it. A ``"drone N"`` hold is a standoff, and
        the higher id of the two gives way -- the same ranking the shield uses,
        so the drone that yields is always the one without right of way.
        """
        wait = self._cfg.safety.hold_replan_s
        by_id = {s.drone_id: s for s in live}
        for drone_id, reason in sorted(held.items()):
            if self._shield.held_for_s(drone_id) < wait:
                continue
            if reason == "map":
                if t - self._drones[drone_id].last_yield_t >= wait:
                    self._drones[drone_id].last_yield_t = t
                    self._reroute(t, by_id[drone_id], avoid=None)
                continue
            other = int(reason.split()[-1])
            # A dead drone cannot move out of anyone's way.
            yielder = drone_id if other not in by_id else max(drone_id, other)
            blocker = other if yielder == drone_id else drone_id
            if t - self._drones[yielder].last_yield_t < wait:
                continue
            self._drones[yielder].last_yield_t = t
            _log.info("drone %d gives way to drone %d", yielder, blocker)
            others = [s.pos for s in live if s.drone_id != yielder]
            avoid = np.array(others, dtype=np.float64).reshape(-1, 3) if others else None
            if not self._reroute(t, by_id[yielder], avoid=avoid):
                self._step_aside(by_id[yielder], by_id.get(blocker))

    def _reroute(self, t: float, state: DroneState, avoid: Points | None) -> bool:
        """Re-plan a drone to its current goal, around ``avoid``; ``True`` on success."""
        d = self._drones[state.drone_id]
        if d.task.kind not in ("frontier", "rth", "yield"):
            return False
        path = self._plan(state.pos, d.task.goal, avoid)
        if path is None:
            if d.task.kind == "frontier":
                self._blacklist(t, d.task.goal)
            return False
        self._assign(
            state.drone_id,
            DroneTask(state.drone_id, d.task.kind, d.task.goal, d.task.look_at, path),
        )
        return True

    def _step_aside(self, state: DroneState, blocker: DroneState | None) -> None:
        """Move off the right-of-way drone's path to the nearest spot that clears it.

        The spot must be reachable, at least ``min_separation_m`` plus the
        inflation margin from the blocker and from every point of its remaining
        path, so the blocker can pass without the shield stopping it again.
        """
        grid = self._grid
        if grid is None or blocker is None:
            self._hold(state.drone_id)
            return
        label = int(grid.label_at(state.pos[None, :])[0])
        if label == 0:
            self._hold(state.drone_id)
            return
        cells = grid.centres(grid.labels == label)
        keep_clear = self._keep_clear_points(blocker)
        need = self._cfg.safety.min_separation_m + self._cfg.safety.inflation_m
        gaps = np.linalg.norm(cells[:, None, :] - keep_clear[None, :, :], axis=-1).min(axis=1)
        ok = cells[gaps >= need]
        if len(ok) == 0:
            self._hold(state.drone_id)
            return
        order = np.argsort(np.linalg.norm(ok - state.pos, axis=1))
        avoid = blocker.pos[None, :]
        # Nearest first; a handful of tries covers a spot behind a wall.
        for idx in order[:8]:
            path = self._plan(state.pos, ok[idx], avoid)
            if path is not None:
                self._assign(
                    state.drone_id, DroneTask(state.drone_id, "yield", ok[idx], None, path)
                )
                self._drones[state.drone_id].history.clear()
                return
        self._hold(state.drone_id)

    def _keep_clear_points(self, blocker: DroneState) -> Points:
        """Sample the blocker's position and remaining path at planning resolution."""
        d = self._drones[blocker.drone_id]
        rest: list[Vec3] = [blocker.pos]
        if d.follower is not None and not d.follower.done:
            rest.extend(d.task.path[d.follower.index :])
        elif d.task.kind == "land":
            rest.append(d.task.goal)
        step = self._cfg.map.plan_voxel_m
        samples = [rest[0][None, :]]
        for a, b in pairwise(rest):
            n = max(2, int(np.ceil(mathutil.norm(b - a) / step)) + 1)
            samples.append(np.linspace(a, b, n))
        return np.vstack(samples)
