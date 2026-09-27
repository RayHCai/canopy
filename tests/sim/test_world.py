"""Tests for :class:`canopy.sim.SimWorld`.

A tiny scene -- a flat ground quad far below every drone the tests fly -- is
enough here: :class:`SimWorld` only needs a loadable
:class:`~canopy.contracts.SceneGeometry` to build its sensor, and none of these
tests care what a scan hits.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import Cls, SceneGeometry, SceneManifest, SceneObject
from canopy.errors import SimulationError
from canopy.sim import RaySensor, SimWorld, load_geometry

#: Ground quad, well clear of the drones flown in these tests.
_LOT = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]])


def _write_ground(path: Path) -> None:
    verts = "\n".join(
        [
            "v -10.000000 -10.000000 0.000000",
            "v 10.000000 -10.000000 0.000000",
            "v 10.000000 10.000000 0.000000",
            "v -10.000000 10.000000 0.000000",
        ]
    )
    faces = "f 1 2 3\nf 1 3 4\n"
    path.write_text(f"# canopy test\no ground\n{verts}\n{faces}", encoding="utf-8")


@pytest.fixture
def geometry(tmp_path: Path) -> SceneGeometry:
    """Build a one-object flat-ground scene, loaded once per test."""
    ground_path = tmp_path / "0_ground.obj"
    _write_ground(ground_path)
    manifest = SceneManifest(
        seed=0,
        lot_bounds=_LOT.copy(),
        footprint=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        objects=[
            SceneObject(obj_id=0, cls=Cls.GROUND, mesh_path=str(ground_path), color=(120, 120, 120))
        ],
        home=np.array([0.0, 0.0, 0.0]),
        gt_meter_id=-1,
    )
    return load_geometry(manifest)


def _pads(n: int) -> np.ndarray:
    """``n`` pads on a line along x at z=0, well separated and clear of the ground edges."""
    xs = np.linspace(-2.0, 2.0, n)
    return np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)


def _world(geometry: SceneGeometry, cfg: Config, n: int = 1, seed: int = 0) -> SimWorld:
    return SimWorld(geometry, _pads(n), cfg, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3])
def test_construction_places_each_drone_at_rest_on_its_pad(
    geometry: SceneGeometry, cfg: Config, n: int
) -> None:
    pads = _pads(n)
    world = SimWorld(geometry, pads, cfg, np.random.default_rng(0))

    assert world.t == 0.0
    drones = world.drones
    assert [d.drone_id for d in drones] == list(range(n))
    for pad, state in zip(pads, drones, strict=True):
        np.testing.assert_allclose(state.pos, pad)
        np.testing.assert_allclose(state.vel, np.zeros(3))
        assert state.alive
        assert state.battery == pytest.approx(1.0)


def test_drones_are_read_only_snapshots(geometry: SceneGeometry, cfg: Config) -> None:
    """`.drones` returns copies whose arrays reject in-place writes.

    A caller that mutated a returned state's ``pos`` in place used to silently
    corrupt the live simulator state; it must now raise instead, both before
    and after the first `step`.
    """
    world = _world(geometry, cfg)
    with pytest.raises(ValueError, match="read-only"):
        world.drones[0].pos[:] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        world.drones[0].vel[:] = 0.0

    world.step({0: world.drones[0].pos.copy()})
    with pytest.raises(ValueError, match="read-only"):
        world.drones[0].pos[:] = 0.0


# ---------------------------------------------------------------------------
# Stepping: time, motion and sensor cadence
# ---------------------------------------------------------------------------
def test_step_advances_time_by_exactly_one_dt(geometry: SceneGeometry, cfg: Config) -> None:
    world = _world(geometry, cfg)
    world.step({0: world.drones[0].pos.copy()})
    assert world.t == pytest.approx(cfg.sim.dt)
    world.step({0: world.drones[0].pos.copy()})
    assert world.t == pytest.approx(2 * cfg.sim.dt)


def test_step_moves_a_drone_toward_its_target(geometry: SceneGeometry, cfg: Config) -> None:
    world = _world(geometry, cfg)
    start = world.drones[0].pos.copy()
    target = start + np.array([0.0, 0.0, 3.0])
    for _ in range(50):
        world.step({0: target})
    final = world.drones[0].pos
    assert final[2] > start[2]
    np.testing.assert_allclose(final, target, atol=0.05)


def test_step_brakes_a_drone_with_no_target(geometry: SceneGeometry, cfg: Config) -> None:
    """A drone at rest with no target this tick has nothing to brake off of."""
    world = _world(geometry, cfg)
    world.step({})
    state = world.drones[0]
    np.testing.assert_allclose(state.vel, np.zeros(3))


def test_sensor_ticks_land_every_control_hz_over_sensor_hz_ticks(
    geometry: SceneGeometry, cfg: Config
) -> None:
    """Scans come back only every ``control_hz // sensor_hz`` control ticks."""
    world = _world(geometry, cfg)
    every = cfg.sim.control_hz // cfg.sim.sensor_hz
    target = world.drones[0].pos.copy()

    for i in range(1, every + 1):
        scans = world.step({0: target})
        if i % every == 0:
            assert len(scans) == 1
            assert scans[0].drone_id == 0
        else:
            assert scans == []


def test_scan_all_reports_every_live_drone_without_advancing_time(
    geometry: SceneGeometry, cfg: Config
) -> None:
    world = _world(geometry, cfg, n=2)
    t_before = world.t
    scans = world.scan_all()
    assert world.t == pytest.approx(t_before)
    assert {s.drone_id for s in scans} == {0, 1}


def test_scan_all_excludes_a_killed_drone(geometry: SceneGeometry, cfg: Config) -> None:
    world = _world(geometry, cfg, n=2)
    world.kill_drone(0)
    scans = world.scan_all()
    assert {s.drone_id for s in scans} == {1}


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
def test_battery_drains_linearly_with_time_while_flying(
    geometry: SceneGeometry, cfg: Config
) -> None:
    fast_cfg = replace(cfg, sim=replace(cfg.sim, battery_drain_per_10s=1.0))
    world = _world(geometry, fast_cfg)
    target = world.drones[0].pos.copy() + np.array([0.0, 0.0, 0.5])

    for _ in range(5):
        world.step({0: target})
    expected = max(0.0, 1.0 - 1.0 * (5 * fast_cfg.sim.dt) / 10.0)
    assert world.drones[0].battery == pytest.approx(expected, abs=1e-9)


def test_battery_clamps_at_zero_and_never_goes_negative(
    geometry: SceneGeometry, cfg: Config
) -> None:
    fast_cfg = replace(cfg, sim=replace(cfg.sim, battery_drain_per_10s=100.0))
    world = _world(geometry, fast_cfg)
    target = world.drones[0].pos.copy() + np.array([0.0, 0.0, 0.5])

    for _ in range(30):
        world.step({0: target})
    assert world.drones[0].battery == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# kill_drone
# ---------------------------------------------------------------------------
def test_kill_drone_marks_it_dead_and_lands_it(geometry: SceneGeometry, cfg: Config) -> None:
    world = _world(geometry, cfg)
    # Fly it up first so there is somewhere to descend from.
    up = world.drones[0].pos.copy() + np.array([0.0, 0.0, 3.0])
    for _ in range(50):
        world.step({0: up})
    airborne = world.drones[0].pos.copy()
    assert airborne[2] > 1.0

    world.kill_drone(0)
    assert world.drones[0].alive is False

    for _ in range(200):
        world.step({0: up})  # a target is ignored for a dead drone
    assert world.drones[0].pos[2] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bad_id", [-1, 1, 5])
def test_kill_drone_raises_for_an_unknown_id(
    geometry: SceneGeometry, cfg: Config, bad_id: int
) -> None:
    world = _world(geometry, cfg, n=1)
    with pytest.raises(SimulationError):
        world.kill_drone(bad_id)


# ---------------------------------------------------------------------------
# Determinism and sensor reuse
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_trajectories(geometry: SceneGeometry, cfg: Config) -> None:
    world_a = _world(geometry, cfg, n=2, seed=7)
    world_b = _world(geometry, cfg, n=2, seed=7)
    target = {0: np.array([1.0, 0.5, 2.0]), 1: np.array([-1.0, -0.5, 1.0])}

    for _ in range(20):
        world_a.step(target)
        world_b.step(target)

    for state_a, state_b in zip(world_a.drones, world_b.drones, strict=True):
        np.testing.assert_array_equal(state_a.pos, state_b.pos)
        np.testing.assert_array_equal(state_a.vel, state_b.vel)


def test_prebuilt_sensor_is_reused_rather_than_rebuilt(
    geometry: SceneGeometry, cfg: Config
) -> None:
    sensor = RaySensor(geometry, cfg.sensor, np.random.default_rng(0))
    world = SimWorld(geometry, _pads(1), cfg, np.random.default_rng(0), sensor=sensor)
    assert world.sensor is sensor
