"""The default motion model: velocity- and acceleration-limited tracking."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import DroneState
from canopy.errors import SimulationError
from canopy.sim.dynamics import KinematicDynamics


def _state(pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> DroneState:
    return DroneState(
        drone_id=0,
        pos=np.array(pos, dtype=np.float64),
        vel=np.zeros(3, dtype=np.float64),
        yaw=0.0,
    )


def _run(
    dyn: KinematicDynamics, target: np.ndarray | None, dt: float, ticks: int
) -> list[DroneState]:
    """Step ``ticks`` times toward ``target``; ``None`` means "no task assigned"."""
    targets = {} if target is None else {0: target}
    return [dyn.step(targets, dt)[0] for _ in range(ticks)]


def test_reaches_its_target(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    target = np.array([5.0, 0.0, 2.0])
    final = _run(dyn, target, cfg.sim.dt, ticks=400)[-1]
    assert final.pos == pytest.approx(target, abs=1e-2)


def test_settles_exactly_on_target_without_overshooting(cfg: Config) -> None:
    """Regression: the follower used to chatter instead of arriving.

    A pure braking profile still commands more speed than the remaining distance
    once inside ``a_max * dt**2``, so the drone overshot by ~7 cm, turned around,
    and settled into a limit cycle -- which also made the yaw, and therefore the
    camera look-at, jitter forever.
    """
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    target = np.array([5.0, 0.0, 0.0])
    history = _run(dyn, target, cfg.sim.dt, ticks=400)

    assert max(float(s.pos[0]) for s in history) <= 5.0
    assert history[-1].pos == pytest.approx(target, abs=0.0)
    assert float(np.linalg.norm(history[-1].vel)) == 0.0


def test_yaw_holds_after_arrival(cfg: Config) -> None:
    """Regression: chattering at the target made the settled yaw drift off."""
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    history = _run(dyn, np.array([0.0, 20.0, 0.0]), cfg.sim.dt, ticks=400)
    assert np.degrees(history[-1].yaw) == pytest.approx(90.0, abs=1e-9)


def test_step_returns_snapshots_not_live_state(cfg: Config) -> None:
    """Regression: step() used to hand back its own mutable DroneState objects.

    A caller accumulating them ended up with N aliases of one object, so any
    history it kept silently collapsed to the latest tick.
    """
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    target = np.array([10.0, 0.0, 0.0])
    first = dyn.step({0: target}, cfg.sim.dt)[0]
    first_pos = first.pos.copy()
    second = dyn.step({0: target}, cfg.sim.dt)[0]

    assert first is not second
    assert first.pos == pytest.approx(first_pos)  # unchanged by the second step
    assert float(second.pos[0]) > float(first.pos[0])


def test_never_exceeds_v_max(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    history = _run(dyn, np.array([100.0, 0.0, 0.0]), cfg.sim.dt, ticks=200)
    assert max(float(np.linalg.norm(s.vel)) for s in history) <= cfg.sim.v_max + 1e-9


def test_never_exceeds_a_max(cfg: Config) -> None:
    dt = cfg.sim.dt
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    history = _run(dyn, np.array([100.0, 0.0, 0.0]), dt, ticks=200)
    velocities = [np.zeros(3), *[s.vel for s in history]]
    for before, after in itertools.pairwise(velocities):
        accel = float(np.linalg.norm(after - before)) / dt
        assert accel <= cfg.sim.a_max + 1e-6


def test_yaw_turns_toward_velocity_at_the_configured_rate(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    # Target is straight along +Y, so the drone should end up facing +90 deg.
    history = _run(dyn, np.array([0.0, 20.0, 0.0]), cfg.sim.dt, ticks=200)
    assert np.degrees(history[-1].yaw) == pytest.approx(90.0, abs=1e-6)

    max_rate = max(
        abs(np.degrees(b.yaw - a.yaw)) / cfg.sim.dt for a, b in itertools.pairwise(history)
    )
    assert max_rate <= cfg.sim.yaw_rate_deg_s + 1e-6


def test_battery_drains_linearly(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    # 10 s of flight should cost exactly one drain quantum.
    ticks = round(10.0 / cfg.sim.dt)
    final = _run(dyn, np.array([1.0, 0.0, 0.0]), cfg.sim.dt, ticks)[-1]
    assert final.battery == pytest.approx(1.0 - cfg.sim.battery_drain_per_10s, abs=1e-9)


def test_battery_never_goes_negative(cfg: Config) -> None:
    sim = dataclasses.replace(cfg.sim, battery_drain_per_10s=5.0)
    dyn = KinematicDynamics(sim)
    dyn.reset([_state()])
    final = _run(dyn, np.array([1.0, 0.0, 0.0]), sim.dt, ticks=200)[-1]
    assert final.battery == 0.0


def test_a_drone_with_no_target_brakes_to_a_stop(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    _run(dyn, np.array([50.0, 0.0, 0.0]), cfg.sim.dt, ticks=40)
    state = _run(dyn, None, cfg.sim.dt, ticks=200)[-1]
    assert float(np.linalg.norm(state.vel)) == pytest.approx(0.0, abs=1e-9)


def test_a_dead_drone_lands_in_place(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    start = _state((3.0, -4.0, 6.0))
    start.alive = False
    dyn.reset([start])
    state = _run(dyn, np.array([50.0, 50.0, 10.0]), cfg.sim.dt, ticks=400)[-1]
    assert state.pos[2] == pytest.approx(0.0)
    assert state.pos[:2] == pytest.approx([3.0, -4.0])
    assert float(np.linalg.norm(state.vel)) == pytest.approx(0.0)


def test_reset_copies_rather_than_aliases(cfg: Config) -> None:
    """The caller's DroneState must not mutate underneath it."""
    dyn = KinematicDynamics(cfg.sim)
    caller_state = _state()
    dyn.reset([caller_state])
    dyn.step({0: np.array([5.0, 0.0, 0.0])}, cfg.sim.dt)
    assert caller_state.pos == pytest.approx([0.0, 0.0, 0.0])


def test_states_are_returned_in_drone_id_order(cfg: Config) -> None:
    dyn = KinematicDynamics(cfg.sim)
    states = [DroneState(drone_id=i, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0) for i in (2, 0, 1)]
    dyn.reset(states)
    assert [s.drone_id for s in dyn.step({}, cfg.sim.dt)] == [0, 1, 2]


@pytest.mark.parametrize("dt", [0.0, -0.05])
def test_non_positive_dt_is_rejected(cfg: Config, dt: float) -> None:
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    with pytest.raises(SimulationError, match="dt must be positive"):
        dyn.step({0: np.zeros(3)}, dt)


def test_braking_respects_a_max(cfg: Config) -> None:
    """Regression: a drone with no target used to stop dead in a single tick.

    The arrival clamp zeroed the velocity whenever there was no target, an
    infinite deceleration that would let the safety shield's stopping-distance
    model pass tests the real vehicle would fail.
    """
    dt = cfg.sim.dt
    dyn = KinematicDynamics(cfg.sim)
    dyn.reset([_state()])
    cruising = _run(dyn, np.array([100.0, 0.0, 0.0]), dt, ticks=40)[-1]
    speed = float(np.linalg.norm(cruising.vel))
    braking = _run(dyn, None, dt, ticks=200)

    velocities = [cruising.vel, *[s.vel for s in braking]]
    for before, after in itertools.pairwise(velocities):
        assert float(np.linalg.norm(after - before)) / dt <= cfg.sim.a_max + 1e-6
    travelled = float(braking[-1].pos[0] - cruising.pos[0])
    assert travelled == pytest.approx(speed**2 / (2 * cfg.sim.a_max), rel=0.1)
