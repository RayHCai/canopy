"""``canopy-fly``: fly one drone in an empty world.

This is the M0/M1 stack check, not a mission. There is no scene, no sensing and
no obstacles: one drone climbs to altitude and orbits the launch pad, under
either dynamics backend. If this runs, the local stack is good.

Examples
--------
Kinematic (works everywhere, no PyBullet needed)::

    canopy-fly --laps 2

Physics, with the PyBullet window (Linux or WSL2)::

    canopy-fly --dynamics pybullet --gui
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from canopy import log
from canopy.config import Config, load_config
from canopy.contracts import DroneState
from canopy.errors import CanopyError
from canopy.planning.waypoints import WaypointFollower, demo_path, path_length
from canopy.sim.dynamics import make_dynamics

if TYPE_CHECKING:
    from canopy.contracts import Points

__all__ = ["FlightSummary", "build_parser", "fly", "main"]

_log = log.get_logger(__name__)

#: Telemetry cadence, in seconds of simulated time.
_TELEMETRY_PERIOD_S = 1.0

#: Safety factor on the analytically estimated flight time before we give up.
_TIMEOUT_SLACK = 3.0


@dataclasses.dataclass(frozen=True, slots=True)
class FlightSummary:
    """What the flight actually did, for humans and for tests."""

    dynamics: str
    waypoints_total: int
    waypoints_reached: int
    completed: bool
    sim_time_s: float
    wall_time_s: float
    path_length_m: float
    distance_flown_m: float
    max_speed_ms: float
    final_pos: tuple[float, float, float]
    final_battery: float

    def as_dict(self) -> dict[str, object]:
        """Serialise for ``--json``."""
        return dataclasses.asdict(self)

    def render(self) -> str:
        """Format as an aligned human-readable block."""
        rows = [
            ("dynamics", self.dynamics),
            ("waypoints", f"{self.waypoints_reached}/{self.waypoints_total}"),
            ("completed", "yes" if self.completed else "NO (timed out)"),
            ("sim time", f"{self.sim_time_s:.2f} s"),
            ("wall time", f"{self.wall_time_s:.2f} s"),
            ("realtime factor", f"{self._realtime_factor():.1f}x"),
            ("path length", f"{self.path_length_m:.2f} m"),
            ("distance flown", f"{self.distance_flown_m:.2f} m"),
            ("max speed", f"{self.max_speed_ms:.2f} m/s"),
            ("final position", "({:.2f}, {:.2f}, {:.2f}) m".format(*self.final_pos)),
            ("battery left", f"{self.final_battery * 100:.1f} %"),
        ]
        width = max(len(label) for label, _ in rows)
        body = "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)
        return f"flight summary\n{body}"

    def _realtime_factor(self) -> float:
        return self.sim_time_s / self.wall_time_s if self.wall_time_s > 0 else math.inf


def _initial_state(cfg: Config) -> DroneState:
    """One drone, parked on the pad."""
    return DroneState(
        drone_id=0,
        pos=cfg.demo.home_xyz.copy(),
        vel=np.zeros(3, dtype=np.float64),
        yaw=0.0,
    )


def _estimate_timeout_s(path: Points, cfg: Config) -> float:
    """Generous upper bound on how long the path should take to fly."""
    cruise = path_length(path) / cfg.sim.v_max
    return max(cruise * _TIMEOUT_SLACK, cfg.sim.timeout_s)


def fly(cfg: Config) -> FlightSummary:
    """Fly the demo path once and report what happened.

    Parameters
    ----------
    cfg
        Validated configuration. ``cfg.sim.dynamics`` selects the backend and
        the ``demo`` section defines the path.

    Returns
    -------
    FlightSummary
        Outcome of the flight, including whether it completed.
    """
    path = demo_path(cfg.demo)
    follower = WaypointFollower(path, cfg.demo.waypoint_tolerance_m)
    dynamics = make_dynamics(cfg)

    dt = cfg.sim.dt
    timeout_s = _estimate_timeout_s(path, cfg)
    _log.info(
        "flying %d waypoint(s), %.1f m, dynamics=%s, dt=%.3f s, timeout=%.0f s",
        follower.n_waypoints,
        path_length(path),
        cfg.sim.dynamics,
        dt,
        timeout_s,
    )

    state = _initial_state(cfg)
    dynamics.reset([state])

    sim_t = 0.0
    distance = 0.0
    max_speed = 0.0
    next_telemetry = 0.0
    wall_start = time.perf_counter()
    try:
        while not follower.done and sim_t < timeout_s:
            target = follower.update(state.pos)
            if target is None:
                break
            previous = state.pos.copy()
            state = dynamics.step({0: target}, dt)[0]
            sim_t += dt

            distance += float(np.linalg.norm(state.pos - previous))
            max_speed = max(max_speed, float(np.linalg.norm(state.vel)))

            if sim_t >= next_telemetry:
                next_telemetry += _TELEMETRY_PERIOD_S
                _log.info(
                    "t=%6.2fs  wp %3d/%-3d  pos=(%6.2f,%6.2f,%5.2f)  |v|=%4.2f m/s  "
                    "yaw=%6.1f deg  batt=%5.1f%%",
                    sim_t,
                    follower.reached_count,
                    follower.n_waypoints,
                    *state.pos,
                    float(np.linalg.norm(state.vel)),
                    math.degrees(state.yaw),
                    state.battery * 100.0,
                )
        # One last check so a drone that arrives on the final tick counts as done.
        follower.update(state.pos)
    finally:
        dynamics.close()

    wall_elapsed = time.perf_counter() - wall_start
    return FlightSummary(
        dynamics=cfg.sim.dynamics,
        waypoints_total=follower.n_waypoints,
        waypoints_reached=follower.reached_count,
        completed=follower.done,
        sim_time_s=sim_t,
        wall_time_s=wall_elapsed,
        path_length_m=path_length(path),
        distance_flown_m=distance,
        max_speed_ms=max_speed,
        final_pos=(float(state.pos[0]), float(state.pos[1]), float(state.pos[2])),
        final_battery=state.battery,
    )


def build_parser() -> argparse.ArgumentParser:
    """Command line for ``canopy-fly``."""
    parser = argparse.ArgumentParser(
        prog="canopy-fly",
        description="Fly a single drone in an empty world to check the local stack.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="config YAML (default: the shipped config/default.yaml)",
    )
    parser.add_argument(
        "--dynamics",
        choices=("kinematic", "pybullet"),
        default=None,
        help="motion model (default: from config)",
    )
    gui = parser.add_mutually_exclusive_group()
    gui.add_argument(
        "--gui",
        dest="gui",
        action="store_true",
        default=None,
        help="open the PyBullet window (physics mode only)",
    )
    gui.add_argument("--no-gui", dest="gui", action="store_false", help="run headless")
    parser.add_argument("--laps", type=int, default=None, help="orbit laps to fly")
    parser.add_argument("--radius", type=float, default=None, metavar="M", help="orbit radius")
    parser.add_argument("--altitude", type=float, default=None, metavar="M", help="orbit altitude")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="repeat for DEBUG")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold CLI flags into the loaded config, then re-validate."""
    sim = cfg.sim
    physics = cfg.physics
    demo = cfg.demo
    if args.dynamics is not None:
        sim = dataclasses.replace(sim, dynamics=args.dynamics)
    if args.gui is not None:
        physics = dataclasses.replace(physics, gui=args.gui)
    if args.laps is not None:
        demo = dataclasses.replace(demo, laps=args.laps)
    if args.radius is not None:
        demo = dataclasses.replace(demo, orbit_radius_m=args.radius)
    if args.altitude is not None:
        demo = dataclasses.replace(demo, takeoff_altitude_m=args.altitude)

    updated = dataclasses.replace(cfg, sim=sim, physics=physics, demo=demo)
    updated.validate()
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    log.configure(-1 if args.quiet else args.verbose)

    try:
        cfg = _apply_overrides(load_config(args.config), args)
        summary = fly(cfg)
    except CanopyError as exc:
        _log.error("%s", exc)
        return 2

    if args.json:
        print(json.dumps(summary.as_dict(), indent=2))
    else:
        print(summary.render())
    return 0 if summary.completed else 1


if __name__ == "__main__":  # Windows spawns rather than forks; always guard.
    sys.exit(main())
