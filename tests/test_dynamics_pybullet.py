"""The optional physics backend.

Skipped wherever the `physics` extra is not installed, which includes native
Windows -- PyBullet publishes no Windows wheel. Run these under WSL2 or Linux.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import DroneState
from canopy.errors import SimulationError
from canopy.sim.dynamics import Dynamics, PyBulletDynamics, make_dynamics

pybullet = pytest.importorskip("pybullet", reason="needs the `physics` extra (Linux/WSL only)")
pytest.importorskip("gym_pybullet_drones", reason="needs the `physics` extra (Linux/WSL only)")

pytestmark = [pytest.mark.physics, pytest.mark.slow]


@pytest.fixture
def physics_cfg(cfg: Config) -> Config:
    """Default config switched to headless physics."""
    return dataclasses.replace(
        cfg,
        sim=dataclasses.replace(cfg.sim, dynamics="pybullet"),
        physics=dataclasses.replace(cfg.physics, gui=False),
    )


@pytest.fixture
def dyn(physics_cfg: Config) -> Iterator[PyBulletDynamics]:
    """Return a one-drone aviary hovering at 1 m, closed on teardown."""
    backend = make_dynamics(physics_cfg)
    assert isinstance(backend, PyBulletDynamics)
    backend.reset(
        [
            DroneState(
                drone_id=0,
                pos=np.array([0.0, 0.0, 1.0]),
                vel=np.zeros(3),
                yaw=0.0,
            )
        ]
    )
    yield backend
    backend.close()


def _settle(dyn: PyBulletDynamics, target: np.ndarray, cfg: Config, seconds: float) -> DroneState:
    """Hold ``target`` for ``seconds`` of simulated time and return the last state."""
    ticks = round(seconds / cfg.sim.dt)
    states = [dyn.step({0: target}, cfg.sim.dt)[0] for _ in range(ticks)]
    return states[-1]


def test_make_dynamics_selects_pybullet(physics_cfg: Config) -> None:
    backend = make_dynamics(physics_cfg)
    try:
        assert isinstance(backend, PyBulletDynamics)
        assert isinstance(backend, Dynamics)
    finally:
        backend.close()


def test_holds_a_hover(dyn: PyBulletDynamics, physics_cfg: Config) -> None:
    """Commanded to its own position, the PID should keep the drone near it."""
    target = np.array([0.0, 0.0, 1.0])
    state = _settle(dyn, target, physics_cfg, seconds=2.0)
    assert state.pos == pytest.approx(target, abs=0.25)


def test_climbs_toward_a_higher_setpoint(dyn: PyBulletDynamics, physics_cfg: Config) -> None:
    target = np.array([0.0, 0.0, 1.5])
    state = _settle(dyn, target, physics_cfg, seconds=4.0)
    assert state.pos[2] > 1.2


def test_state_projection_is_finite(dyn: PyBulletDynamics, physics_cfg: Config) -> None:
    state = dyn.step({0: np.array([0.0, 0.0, 1.0])}, physics_cfg.sim.dt)[0]
    assert np.all(np.isfinite(state.pos))
    assert np.all(np.isfinite(state.vel))
    assert np.isfinite(state.yaw)
    assert state.drone_id == 0
    assert state.alive


def test_battery_drains(dyn: PyBulletDynamics, physics_cfg: Config) -> None:
    state = _settle(dyn, np.array([0.0, 0.0, 1.0]), physics_cfg, seconds=10.0)
    assert state.battery == pytest.approx(1.0 - physics_cfg.sim.battery_drain_per_10s, abs=1e-9)


def test_step_before_reset_is_rejected(physics_cfg: Config) -> None:
    backend = make_dynamics(physics_cfg)
    with pytest.raises(SimulationError, match="before reset"):
        backend.step({0: np.zeros(3)}, physics_cfg.sim.dt)


def test_mismatched_dt_is_rejected(dyn: PyBulletDynamics, physics_cfg: Config) -> None:
    """The backend is wired for one dt; a different one would silently desync."""
    with pytest.raises(SimulationError, match="wired for dt"):
        dyn.step({0: np.zeros(3)}, physics_cfg.sim.dt * 1.5)


def test_multiple_drones_keep_their_ids(physics_cfg: Config) -> None:
    backend = make_dynamics(physics_cfg)
    try:
        backend.reset(
            [
                DroneState(
                    drone_id=i,
                    pos=np.array([float(i), 0.0, 1.0]),
                    vel=np.zeros(3),
                    yaw=0.0,
                )
                for i in (0, 1, 2)
            ]
        )
        states = backend.step({}, physics_cfg.sim.dt)
        assert [s.drone_id for s in states] == [0, 1, 2]
    finally:
        backend.close()


def test_close_is_idempotent(physics_cfg: Config) -> None:
    backend = make_dynamics(physics_cfg)
    backend.close()
    backend.close()
