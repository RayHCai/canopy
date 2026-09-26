"""Motion model.

:class:`KinematicDynamics` is the only one: a velocity-limited waypoint follower
with no forces and no attitude. Fast and deterministic, which is what a
per-frame mapping loop needs. A PyBullet backend used to sit beside it; it was
removed to keep one code path (docs/adr/0006-kinematic-only-dynamics.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from canopy import mathutil
from canopy.contracts import DroneState, Vec3
from canopy.errors import SimulationError

if TYPE_CHECKING:
    from canopy.config import SimCfg

__all__ = ["KinematicDynamics"]

#: Descent speed, as a fraction of ``v_max``, for a drone that has been killed.
_LANDING_SPEED_FRACTION = 0.33


def _copy_state(state: DroneState) -> DroneState:
    """Return an independent copy of ``state``."""
    return DroneState(
        drone_id=state.drone_id,
        pos=np.array(state.pos, dtype=np.float64),
        vel=np.array(state.vel, dtype=np.float64),
        yaw=float(state.yaw),
        alive=state.alive,
        battery=float(state.battery),
    )


def _drain_battery(state: DroneState, dt: float, drain_per_10s: float) -> None:
    """Apply the linear battery model in place."""
    state.battery = max(0.0, state.battery - drain_per_10s * dt / 10.0)


def _slew_yaw(state: DroneState, dt: float, yaw_rate_deg_s: float) -> None:
    """Turn ``state.yaw`` toward the horizontal velocity, rate-limited."""
    horizontal = state.vel[:2]
    if float(np.linalg.norm(horizontal)) < mathutil.EPS:
        return
    desired = float(np.arctan2(horizontal[1], horizontal[0]))
    error = mathutil.wrap_to_pi(desired - state.yaw)
    max_step = np.deg2rad(yaw_rate_deg_s) * dt
    state.yaw = mathutil.wrap_to_pi(state.yaw + float(np.clip(error, -max_step, max_step)))


class KinematicDynamics:
    """Velocity-limited waypoint follower. No forces, no attitude, no physics.

    Each tick the drone accelerates toward its target at up to ``a_max``, never
    exceeding ``v_max``, and starts braking early enough to settle on the
    waypoint rather than orbit it.
    """

    def __init__(self, cfg: SimCfg) -> None:
        self._cfg = cfg
        self._states: list[DroneState] = []

    def reset(self, states: list[DroneState]) -> None:
        """Adopt ``states`` as the initial condition (copied, not aliased)."""
        self._states = [_copy_state(s) for s in sorted(states, key=lambda s: s.drone_id)]

    def step(self, targets: Mapping[int, Vec3], dt: float) -> list[DroneState]:
        """Advance every drone one tick toward its target position.

        Parameters
        ----------
        targets
            Target position per ``drone_id``. A drone absent from the mapping
            brakes to a stop and holds station.
        dt
            Control timestep in seconds.

        Returns
        -------
        list of DroneState
            The new state of every drone, in ``drone_id`` order.
        """
        if dt <= 0.0:
            msg = f"dt must be positive, got {dt}"
            raise SimulationError(msg)

        for state in self._states:
            if not state.alive:
                self._land(state, dt)
                continue
            target = targets.get(state.drone_id)
            self._track(state, target, dt)
            _slew_yaw(state, dt, self._cfg.yaw_rate_deg_s)
            _drain_battery(state, dt, self._cfg.battery_drain_per_10s)

        # Snapshots, not the live objects: a caller that accumulates the returned
        # states would otherwise end up holding N aliases of one mutating object.
        return [_copy_state(s) for s in self._states]

    # -- internals ----------------------------------------------------------
    def _track(self, state: DroneState, target: Vec3 | None, dt: float) -> None:
        """Accelerate toward ``target``, or brake to a stop if there is none."""
        if target is None:
            # Decelerate at a_max along the current heading. This must not share
            # the arrival clamp below: with no target that clamp would zero the
            # velocity in one tick, an infinite deceleration the safety shield's
            # stopping-distance model would then silently depend on.
            state.vel = state.vel - mathutil.clamp_norm(state.vel, self._cfg.a_max * dt)
            state.pos = state.pos + state.vel * dt
            return

        to_target = np.asarray(target, dtype=np.float64) - state.pos
        distance = mathutil.norm(to_target)
        # The `distance / dt` term is what stops the follower chattering. The
        # continuous braking profile alone still commands more speed than the
        # remaining distance once inside a_max * dt^2, so the drone overshoots,
        # turns around, overshoots again, and settles into a limit cycle
        # instead of arriving -- which also makes the yaw (and therefore the
        # camera look-at) jitter.
        speed = min(
            mathutil.braking_speed(distance, self._cfg.v_max, self._cfg.a_max),
            distance / dt,
        )
        desired_vel = mathutil.unit(to_target) * speed

        delta_v = mathutil.clamp_norm(desired_vel - state.vel, self._cfg.a_max * dt)
        state.vel = state.vel + delta_v

        step = state.vel * dt
        if mathutil.norm(step) > distance:
            # Land exactly on the target rather than past it, and keep velocity
            # consistent with the displacement actually taken.
            step = to_target
            state.vel = step / dt
        state.pos = state.pos + step

    def _land(self, state: DroneState, dt: float) -> None:
        """Drop a dead drone's horizontal velocity and descend it to the ground."""
        descent = self._cfg.v_max * _LANDING_SPEED_FRACTION
        state.vel = np.array([0.0, 0.0, -descent if state.pos[2] > 0.0 else 0.0])
        state.pos = state.pos + state.vel * dt
        if state.pos[2] <= 0.0:
            state.pos[2] = 0.0
            state.vel = np.zeros(3, dtype=np.float64)
