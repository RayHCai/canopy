"""What each simulated control tick leaves behind: a snapshot and a run log.

Two consumers read the ticks :class:`~canopy.viz.session.ViewerSession`'s
simulation thread produces, at different times:

:class:`Snapshot`
    Everything one frame shows about one tick. Playback trails the simulation
    on purpose (see :mod:`canopy.viz.session`), so a frame can never read the
    live mission; it reads the snapshot taken when that tick ran.
:class:`RunRecorder`
    The whole mission, recorded for the dashboard (``docs/adr/0017``): drone
    tracks and fleet events, built up in lists
    :meth:`~canopy.viz.session.ViewerSession.run_record` can hand over
    verbatim. It is fed on the simulation thread, right after the tick it
    describes -- not from :meth:`~canopy.viz.session.ViewerSession.advance`,
    which only ever sees the tick playback has reached so far. Recording from
    ``advance`` would leave the record missing whatever tail of the mission
    the page had not caught up to yet; recording where the tick itself
    happens means it is already complete once the mission reaches ``done``,
    however far behind that the playback clock still is when the page
    notices and uploads it.

Neither takes a lock of its own: the session calls both under ``_sim_lock``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from canopy.contracts import DiscoveredObject, DroneState, Points, Vec3
from canopy.planning import MappingEta, MissionRun

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["T_EPS", "RunRecorder", "Snapshot", "take_snapshot"]

#: Snapshot times are sums of ``dt``; this absorbs their rounding.
T_EPS = 1e-9

#: How often a drone's pose lands in its recorded track (ADR 0017), fine enough
#: for smooth playback scrubbing without a row per control tick.
_TRACK_PERIOD_S = 0.5

#: Battery fraction below which a drone's track gets one ``low_battery`` event.
#: Distinct from ``sim.rth_battery``: that is when a drone actually turns for
#: home, this is only when the dashboard should start warning about it.
_LOW_BATTERY_FRACTION = 0.2

#: Verb phrase for a recorded fleet event, per task kind. ``frontier`` is
#: handled separately in :meth:`RunRecorder._task_label`, since the same
#: kind covers both exploring new space and inspecting a mapped one up close
#: (see ``MissionController._replan``'s docstring) and the two read very
#: differently to a reviewer.
_TASK_VERBS: dict[str, str] = {
    "takeoff": "taking off",
    "hold": "holding",
    "yield": "yielding",
    "escape": "rerouting",
    "rth": "returning to pad",
    "land": "landing",
}

#: Task labels too brief to log. Holding, yielding and rerouting are the
#: controller settling its own traffic -- one seed-7 run logged 71 of them in
#: 135 events -- and a reviewer reading the fleet log needs where each drone
#: went, not every pause on the way. They neither emit an event nor count as
#: the drone's last reported task, so exploring -> rerouting -> exploring
#: logs nothing.
_UNLOGGED_TASKS = frozenset({"taking off", "holding", "yielding", "rerouting"})


@dataclass(frozen=True, slots=True, eq=False)
class Snapshot:
    """Everything a frame shows about one control tick, captured when it ran.

    Playback trails the simulation, so a frame cannot read the live mission:
    by the time a tick is on screen the controller may be seconds further on.
    """

    t: float
    states: list[DroneState]
    phase: str
    ground_band: float
    total: float
    #: Per drone id: task kind and goal.
    tasks: dict[int, tuple[str, Vec3]]
    landed: frozenset[int]
    #: Target centroids, shape ``(k, 3)``.
    frontiers: Points
    #: Triangle ids first seen on this tick.
    revealed: npt.NDArray[np.int64]
    #: Objects classified so far. The mapper replaces, never mutates, the
    #: objects it publishes, so holding them here needs no copy.
    detections: tuple[DiscoveredObject, ...]
    #: Suggested battery sites as the page draws them; ``None`` while mapping.
    site: dict[str, Any] | None
    #: Estimated simulated seconds of exploration left; ``None`` while there
    #: is no trend to project yet, ``0.0`` once mapping has ended.
    eta_s: float | None
    #: Simulated time mapping ended; ``None`` while it has not.
    mapped_at_s: float | None


def take_snapshot(run: MissionRun, eta: MappingEta, site: dict[str, Any] | None = None) -> Snapshot:
    """Capture ``run`` as it stands after its latest tick, ``eta`` already updated."""
    frontiers = run.controller.frontiers
    return Snapshot(
        t=run.t,
        states=run.drones,
        phase=str(run.phase),
        ground_band=run.coverage_ground_band,
        total=run.coverage_total,
        tasks={i: (task.kind, task.goal.copy()) for i, task in run.controller.tasks.items()},
        landed=run.controller.landed,
        frontiers=np.array([f.centroid for f in frontiers], dtype=np.float64).reshape(-1, 3),
        revealed=run.pop_revealed(),
        detections=tuple(run.mapper.state.discovered.values()),
        site=site,
        eta_s=eta.remaining_s,
        mapped_at_s=eta.done_at_s,
    )


class RunRecorder:
    """The recorded mission (ADR 0017) of one :class:`~canopy.planning.MissionRun`.

    One recorder per mission: the session builds a fresh one on every reset,
    so nothing here needs clearing. Its trackers start empty rather than at
    some sentinel "no task yet" value. The first :meth:`record` -- for the
    pre-tick state ``MissionController`` seeds every drone with -- logs
    samples but no fleet events, because every drone starts on its takeoff
    task, which is never logged; its first real task then logs as that
    drone's launch.

    Parameters
    ----------
    run
        The mission being recorded. Read, never driven: task labels need the
        live controller's ``look_at``, which no snapshot carries.
    cfg
        Validated configuration. Reads ``planner.inspect_bucket_m``.
    """

    def __init__(self, run: MissionRun, cfg: Config) -> None:
        self._run = run
        self._cfg = cfg
        ids = [state.drone_id for state in run.drones]
        self._tracks: dict[int, list[list[float]]] = {i: [] for i in ids}
        self._track_next_t: dict[int, float] = dict.fromkeys(ids, -math.inf)
        self._events: list[dict[str, Any]] = []
        self._event_prev_label: dict[int, str] = {}
        self._event_low_battery: set[int] = set()
        self._event_failed: set[int] = set()
        self._recording_done = False

    def tracks(self) -> list[dict[str, Any]]:
        """Per drone its ``id``, recorded ``samples``, and current ``alive`` and ``battery``.

        ``samples`` are ``[t, x, y, z, yaw, battery]`` rows roughly
        ``_TRACK_PERIOD_S`` apart, plus the first tick; copies, so the caller
        may hold them after the lock is released.
        """
        return [
            {
                "id": state.drone_id,
                "samples": [list(row) for row in self._tracks[state.drone_id]],
                "alive": state.alive,
                "battery": state.battery,
            }
            for state in self._run.drones
        ]

    def events(self) -> list[dict[str, Any]]:
        """Every fleet event so far, copied, in the shape :meth:`_event` builds."""
        return [dict(event) for event in self._events]

    def record(self, snap: Snapshot) -> None:
        """Append this tick's track samples and any fleet events. Caller holds ``_sim_lock``.

        Called once by the session's reset for the pre-tick state -- seeding the
        per-drone trackers below so nothing about it reads as a change -- and
        once per tick from the simulation thread after. A no-op once the
        mission has reached ``done``: that tick's ``complete`` event is the
        last thing worth recording, and every later tick just holds station.
        """
        if self._recording_done:
            return
        for state in snap.states:
            self._sample(snap.t, state)
            self._track_task_and_health(snap, state)
        if snap.phase == "done":
            self._recording_done = True
            n_objects = len(snap.detections)
            coverage_pct = round(snap.total * 100)
            battery = snap.states[0].battery if snap.states else 0.0
            self._events.append(
                {
                    "t": snap.t,
                    "drone_id": 0,
                    "type": "complete",
                    "message": (
                        f"Survey complete: {n_objects} objects detected, coverage {coverage_pct}%"
                    ),
                    "status": "Idle",
                    "battery_pct": round(battery * 100),
                    "task": "landed",
                }
            )

    def _sample(self, t: float, state: DroneState) -> None:
        """Append ``state`` to its drone's track if ``_TRACK_PERIOD_S`` has passed."""
        if t + T_EPS < self._track_next_t[state.drone_id]:
            return
        self._tracks[state.drone_id].append(
            [
                t,
                float(state.pos[0]),
                float(state.pos[1]),
                float(state.pos[2]),
                state.yaw,
                state.battery,
            ]
        )
        self._track_next_t[state.drone_id] = t + _TRACK_PERIOD_S

    def _track_task_and_health(self, snap: Snapshot, state: DroneState) -> None:
        """Emit a fleet event for ``state`` if its task changed or its health crossed a line."""
        drone_id = state.drone_id
        label = self._task_label(drone_id)
        if label not in _UNLOGGED_TASKS and self._event_prev_label.get(drone_id) != label:
            self._events.append(
                self._event(snap, state, "capture", f"Drone-{drone_id + 1} {label}", label)
            )
            self._event_prev_label[drone_id] = label
        if state.battery < _LOW_BATTERY_FRACTION and drone_id not in self._event_low_battery:
            self._event_low_battery.add(drone_id)
            pct = round(state.battery * 100)
            message = f"Drone-{drone_id + 1} battery low ({pct}%)"
            self._events.append(self._event(snap, state, "low_battery", message, label))
        if not state.alive and drone_id not in self._event_failed:
            self._event_failed.add(drone_id)
            message = f"Drone-{drone_id + 1} failed"
            self._events.append(self._event(snap, state, "failure", message, label))

    def _event(
        self, snap: Snapshot, state: DroneState, kind: str, message: str, task: str
    ) -> dict[str, Any]:
        """One fleet event dict, in the shape ``ViewerSession.run_record`` documents."""
        if not state.alive:
            status = "Failed"
        elif state.drone_id in snap.landed:
            status = "Idle"
        else:
            status = "Flying"
        return {
            "t": snap.t,
            "drone_id": state.drone_id,
            "type": kind,
            "message": message,
            "status": status,
            "battery_pct": round(state.battery * 100),
            "task": task,
        }

    def _task_label(self, drone_id: int) -> str:
        """Human text for one drone's current task. Caller holds ``_sim_lock``.

        Reads the controller directly rather than a snapshot's simplified
        ``(kind, goal)`` pair, because telling exploring from inspecting needs
        ``look_at`` -- what the task is looking *at*, not where it flies to --
        which no snapshot carries.
        """
        if drone_id in self._run.controller.landed:
            return "landed"
        task = self._run.controller.tasks.get(drone_id)
        if task is None:  # pragma: no cover - every live drone always has one
            return "idle"
        if task.kind == "frontier":
            if task.look_at is not None:
                near = self._nearest_discovered(task.look_at)
                if near is not None:
                    return f"inspecting {near}"
            return "exploring"
        return _TASK_VERBS.get(task.kind, task.kind)

    def _nearest_discovered(self, point: Vec3) -> str | None:
        """Class name of the discovered object nearest ``point``, within half an inspection bucket.

        The radius matches :meth:`~canopy.planning.mission.MissionController._arrive`'s
        own half-bucket mark: a task whose look-at point is that close to an
        object is, in effect, standing at it.
        """
        discovered = list(self._run.mapper.state.discovered.values())
        if not discovered:
            return None
        centers = np.array([obj.box.center for obj in discovered], dtype=np.float64)
        dists = np.linalg.norm(centers - point, axis=1)
        idx = int(np.argmin(dists))
        radius = 0.5 * self._cfg.planner.inspect_bucket_m
        if dists[idx] > radius:
            return None
        return discovered[idx].cls.name.lower().replace("_", " ")
