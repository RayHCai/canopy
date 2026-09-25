"""Motion models.

Two interchangeable backends sit behind :class:`Dynamics`:

``KinematicDynamics``
    The default. A velocity-limited waypoint follower with no physics at all.
    Fast, deterministic, and the only backend on the critical path.
``PyBulletDynamics``
    Quadcopter rigid-body physics with cascaded PID attitude control, borrowed
    from our gym-pybullet-drones fork. Selected with ``--dynamics pybullet``.

Keeping the interface this narrow is what makes physics optional: nothing
upstream of here knows which backend it is driving.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from canopy import mathutil
from canopy.contracts import DroneState, Vec3
from canopy.errors import DependencyMissingError, SimulationError
from canopy.log import get_logger

if TYPE_CHECKING:
    from canopy.config import Config, PhysicsCfg, SimCfg

__all__ = ["Dynamics", "KinematicDynamics", "PyBulletDynamics", "make_dynamics"]

_log = get_logger(__name__)

#: Descent speed, as a fraction of ``v_max``, for a drone that has been killed.
_LANDING_SPEED_FRACTION = 0.33


@runtime_checkable
class Dynamics(Protocol):
    """How drone states evolve given per-drone waypoint targets.

    ``close`` is an addition to the interface in the spec: the PyBullet backend
    owns an OS-level resource and the caller must be able to release it without
    knowing which backend it holds.
    """

    def reset(self, states: list[DroneState]) -> None:
        """Adopt ``states`` as the initial condition."""
        ...

    def step(self, targets: Mapping[int, Vec3], dt: float) -> list[DroneState]:
        """Advance by ``dt`` seconds toward each drone's target position.

        Parameters
        ----------
        targets
            Target position per ``drone_id``. A drone absent from the mapping
            holds station.
        dt
            Control timestep in seconds.

        Returns
        -------
        list of DroneState
            The new state of every drone, in ``drone_id`` order.
        """
        ...

    def close(self) -> None:
        """Release any backend resources. Safe to call more than once."""
        ...


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
        self._states = [
            DroneState(
                drone_id=s.drone_id,
                pos=np.array(s.pos, dtype=np.float64),
                vel=np.array(s.vel, dtype=np.float64),
                yaw=float(s.yaw),
                alive=s.alive,
                battery=float(s.battery),
            )
            for s in sorted(states, key=lambda s: s.drone_id)
        ]

    def step(self, targets: Mapping[int, Vec3], dt: float) -> list[DroneState]:
        """Advance every drone one tick. See :meth:`Dynamics.step`."""
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

        return self._states

    def close(self) -> None:
        """No resources to release."""

    # -- internals ----------------------------------------------------------
    def _track(self, state: DroneState, target: Vec3 | None, dt: float) -> None:
        """Accelerate toward ``target``, or brake to a stop if there is none."""
        if target is None:
            desired_vel = np.zeros(3, dtype=np.float64)
        else:
            to_target = np.asarray(target, dtype=np.float64) - state.pos
            speed = mathutil.braking_speed(
                mathutil.norm(to_target), self._cfg.v_max, self._cfg.a_max
            )
            desired_vel = mathutil.unit(to_target) * speed

        delta_v = mathutil.clamp_norm(desired_vel - state.vel, self._cfg.a_max * dt)
        state.vel = state.vel + delta_v
        state.pos = state.pos + state.vel * dt

    def _land(self, state: DroneState, dt: float) -> None:
        """A dead drone drops its horizontal velocity and descends to the ground."""
        descent = self._cfg.v_max * _LANDING_SPEED_FRACTION
        state.vel = np.array([0.0, 0.0, -descent if state.pos[2] > 0.0 else 0.0])
        state.pos = state.pos + state.vel * dt
        if state.pos[2] <= 0.0:
            state.pos[2] = 0.0
            state.vel = np.zeros(3, dtype=np.float64)


class PyBulletDynamics:
    """Rigid-body quadcopter physics via our gym-pybullet-drones fork.

    One Canopy control tick is executed as :attr:`Config.physics_steps_per_tick`
    inner PID steps, which is why the config validates that the three rates nest
    exactly. Targets are fed to ``DSLPIDControl`` as position setpoints with the
    yaw aimed along the direction of travel.
    """

    def __init__(self, sim: SimCfg, physics: PhysicsCfg, steps_per_tick: int) -> None:
        self._sim = sim
        self._physics = physics
        self._steps_per_tick = steps_per_tick
        self._env: Any = None
        self._controllers: list[Any] = []
        self._obs: Any = None
        self._ids: list[int] = []
        self._alive: dict[int, bool] = {}
        self._battery: dict[int, float] = {}
        self._modules = _import_pybullet_modules()

    def reset(self, states: list[DroneState]) -> None:
        """Build a fresh aviary seeded with ``states``."""
        self.close()
        ordered = sorted(states, key=lambda s: s.drone_id)
        self._ids = [s.drone_id for s in ordered]
        self._alive = {s.drone_id: s.alive for s in ordered}
        self._battery = {s.drone_id: float(s.battery) for s in ordered}

        mods = self._modules
        initial_xyzs = np.array([s.pos for s in ordered], dtype=np.float64)
        initial_rpys = np.array([[0.0, 0.0, s.yaw] for s in ordered], dtype=np.float64)
        drone_model = mods["DroneModel"](self._physics.drone_model)

        self._env = mods["CtrlAviary"](
            drone_model=drone_model,
            num_drones=len(ordered),
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=mods["Physics"]("pyb"),
            pyb_freq=self._physics.pyb_freq,
            ctrl_freq=self._physics.ctrl_freq,
            gui=self._physics.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self._controllers = [mods["DSLPIDControl"](drone_model=drone_model) for _ in ordered]
        self._obs, _ = self._env.reset()
        _log.debug(
            "pybullet aviary ready: %d drone(s), pyb=%dHz ctrl=%dHz gui=%s",
            len(ordered),
            self._physics.pyb_freq,
            self._physics.ctrl_freq,
            self._physics.gui,
        )

    def step(self, targets: Mapping[int, Vec3], dt: float) -> list[DroneState]:
        """Advance the physics by ``dt``. See :meth:`Dynamics.step`."""
        if self._env is None:
            msg = "PyBulletDynamics.step() called before reset()"
            raise SimulationError(msg)
        expected = 1.0 / self._sim.control_hz
        if not np.isclose(dt, expected):
            msg = (
                f"PyBulletDynamics is wired for dt={expected:.6f}s "
                f"(sim.control_hz={self._sim.control_hz}), got dt={dt:.6f}s"
            )
            raise SimulationError(msg)

        ctrl_timestep = self._env.CTRL_TIMESTEP
        action = np.zeros((len(self._ids), 4), dtype=np.float64)
        for _ in range(self._steps_per_tick):
            for row, drone_id in enumerate(self._ids):
                state_vec = self._obs[row]
                target = targets.get(drone_id)
                if target is None or not self._alive[drone_id]:
                    target = self._hold_target(state_vec, alive=self._alive[drone_id])
                rpm, _, _ = self._controllers[row].computeControlFromState(
                    control_timestep=ctrl_timestep,
                    state=state_vec,
                    target_pos=np.asarray(target, dtype=np.float64),
                    target_rpy=np.array([0.0, 0.0, self._target_yaw(state_vec, target)]),
                )
                action[row, :] = rpm
            self._obs, _, _, _, _ = self._env.step(action)

        for drone_id in self._ids:
            if self._alive[drone_id]:
                self._battery[drone_id] = max(
                    0.0,
                    self._battery[drone_id] - self._sim.battery_drain_per_10s * dt / 10.0,
                )
        return self._read_states()

    def kill(self, drone_id: int) -> None:
        """Mark a drone dead; it will be commanded to descend from now on."""
        self._alive[drone_id] = False

    def close(self) -> None:
        """Disconnect the PyBullet client. Safe to call more than once."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._controllers = []
            self._obs = None

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _hold_target(state_vec: Any, *, alive: bool) -> Vec3:
        """Setpoint for a drone with no task: hover, or sink to the ground."""
        pos = np.asarray(state_vec[0:3], dtype=np.float64)
        if alive:
            return pos
        return np.array([pos[0], pos[1], 0.0], dtype=np.float64)

    @staticmethod
    def _target_yaw(state_vec: Any, target: Vec3) -> float:
        """Face the direction of travel; hold current yaw when nearly there."""
        delta = np.asarray(target, dtype=np.float64)[:2] - np.asarray(
            state_vec[0:2], dtype=np.float64
        )
        if float(np.linalg.norm(delta)) < mathutil.EPS:
            return float(state_vec[9])
        return float(np.arctan2(delta[1], delta[0]))

    def _read_states(self) -> list[DroneState]:
        """Project the aviary observation onto :class:`DroneState`.

        The observation row layout is fixed by ``CtrlAviary._computeObs``:
        ``[x y z, q1..q4, roll pitch yaw, vx vy vz, wx wy wz, rpm0..3]``.
        """
        states: list[DroneState] = []
        for row, drone_id in enumerate(self._ids):
            obs = self._obs[row]
            states.append(
                DroneState(
                    drone_id=drone_id,
                    pos=np.asarray(obs[0:3], dtype=np.float64),
                    vel=np.asarray(obs[10:13], dtype=np.float64),
                    yaw=float(obs[9]),
                    alive=self._alive[drone_id],
                    battery=self._battery[drone_id],
                )
            )
        return states


def _import_pybullet_modules() -> dict[str, Any]:
    """Import the physics backend, or explain how to install it."""
    try:
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = (
            "physics mode needs the `physics` extra, which PyBullet only ships as a "
            "Linux wheel. On Windows run the physics track under WSL2:\n"
            "    wsl -d Ubuntu\n"
            "    make -C /mnt/c/.../canopy setup-physics\n"
            "or install it directly with `uv sync --extra dev --extra physics`."
        )
        raise DependencyMissingError(msg) from exc
    return {
        "CtrlAviary": CtrlAviary,
        "DSLPIDControl": DSLPIDControl,
        "DroneModel": DroneModel,
        "Physics": Physics,
    }


def make_dynamics(cfg: Config) -> Dynamics:
    """Build the dynamics backend named by ``cfg.sim.dynamics``."""
    if cfg.sim.dynamics == "kinematic":
        return KinematicDynamics(cfg.sim)
    if cfg.sim.dynamics == "pybullet":
        return PyBulletDynamics(cfg.sim, cfg.physics, cfg.physics_steps_per_tick)
    msg = f"unknown dynamics backend {cfg.sim.dynamics!r}"  # pragma: no cover - config validates
    raise SimulationError(msg)
