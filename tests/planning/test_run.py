"""Tests for :class:`canopy.planning.MissionRun`.

The scene is a ground plane plus one closed box "house", built as raw OBJ text
(the same tiny-lot idiom as ``test_mission.py``) so a whole mission finishes in
a few seconds of sim time and this file runs quickly without touching worldgen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config
from canopy.contracts import Cls, SceneManifest, SceneObject
from canopy.errors import PlanningError
from canopy.planning import MissionRun
from canopy.planning.mission import Phase
from canopy.sim import load_geometry

#: A small lot: big enough for a house and launch pads, small enough that a
#: full mission explores it in seconds.
_LOT = np.array([[-6.0, -6.0, 0.0], [6.0, 6.0, 6.0]])
_HOUSE_LO = np.array([-3.0, -2.0, 0.0])
_HOUSE_HI = np.array([3.0, 2.0, 3.0])
#: Launch point, well clear of the house footprint (y in [-2, 2]).
_HOME = np.array([0.0, -5.0, 0.0])

#: Standard 12-triangle box, one triangle pair per face, outward winding.
_BOX_FACES = [
    (1, 4, 3),
    (1, 3, 2),  # bottom, z0
    (5, 6, 7),
    (5, 7, 8),  # top, z1
    (1, 2, 6),
    (1, 6, 5),  # front, y0
    (4, 8, 7),
    (4, 7, 3),  # back, y1
    (1, 5, 8),
    (1, 8, 4),  # left, x0
    (2, 3, 7),
    (2, 7, 6),  # right, x1
]


def _write_obj(path: Path, o_name: str, verts: str, faces: str) -> None:
    path.write_text(f"# canopy test\no {o_name}\n{verts}\n{faces}", encoding="utf-8")


def _box_mesh(lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]) -> tuple[str, str]:
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    verts_text = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners)
    faces_text = "\n".join(f"f {a} {b} {c}" for a, b, c in _BOX_FACES) + "\n"
    return verts_text, faces_text


def _ground_mesh(lo_xy: npt.NDArray[np.float64], hi_xy: npt.NDArray[np.float64]) -> tuple[str, str]:
    verts = [
        (lo_xy[0], lo_xy[1], 0.0),
        (hi_xy[0], lo_xy[1], 0.0),
        (hi_xy[0], hi_xy[1], 0.0),
        (lo_xy[0], hi_xy[1], 0.0),
    ]
    verts_text = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts)
    faces_text = "f 1 2 3\nf 1 3 4\n"
    return verts_text, faces_text


@pytest.fixture
def manifest(tmp_path: Path) -> SceneManifest:
    """Ground plane plus one closed box house, as a hand-built manifest."""
    ground_path = tmp_path / "0_ground.obj"
    house_path = tmp_path / "1_house.obj"
    g_verts, g_faces = _ground_mesh(_LOT[0, :2], _LOT[1, :2])
    h_verts, h_faces = _box_mesh(_HOUSE_LO, _HOUSE_HI)
    _write_obj(ground_path, "0_ground", g_verts, g_faces)
    _write_obj(house_path, "1_house", h_verts, h_faces)

    return SceneManifest(
        seed=0,
        lot_bounds=_LOT.copy(),
        footprint=[
            (_HOUSE_LO[0], _HOUSE_LO[1]),
            (_HOUSE_HI[0], _HOUSE_LO[1]),
            (_HOUSE_HI[0], _HOUSE_HI[1]),
            (_HOUSE_LO[0], _HOUSE_HI[1]),
        ],
        objects=[
            SceneObject(
                obj_id=0, cls=Cls.GROUND, mesh_path=str(ground_path), color=(120, 120, 120)
            ),
            SceneObject(obj_id=1, cls=Cls.WALL, mesh_path=str(house_path), color=(200, 200, 200)),
        ],
        home=_HOME.copy(),
        gt_meter_id=-1,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_drones", [1, 2])
def test_construction_places_pads_on_the_ground_at_home(
    cfg: Config, manifest: SceneManifest, n_drones: int
) -> None:
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, n_drones, cfg, seed=0)

    assert run.pads.shape == (n_drones, 3)
    np.testing.assert_allclose(run.pads.mean(axis=0)[:2], _HOME[:2])
    np.testing.assert_allclose(run.pads[:, 2], 0.0)
    assert run.t == 0.0
    assert run.phase is Phase.TAKEOFF
    assert not run.done
    for pad, state in zip(run.pads, run.drones, strict=True):
        np.testing.assert_allclose(state.pos, pad)


@pytest.mark.parametrize("n_drones", [0, -1])
def test_construction_rejects_an_empty_swarm(
    cfg: Config, manifest: SceneManifest, n_drones: int
) -> None:
    geometry = load_geometry(manifest)
    with pytest.raises(PlanningError, match=f"n_drones={n_drones}"):
        MissionRun(manifest, geometry, n_drones, cfg)


def test_construction_integrates_the_initial_scan_before_any_tick(
    cfg: Config, manifest: SceneManifest
) -> None:
    """The constructor scans from the pads once, so ground truth is already revealed."""
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)

    revealed = run.pop_revealed()
    assert revealed.size > 0
    assert list(revealed) == sorted(revealed.tolist())
    # A second call with nothing new since must be empty.
    assert run.pop_revealed().size == 0


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------
def test_tick_advances_time_and_moves_drones(cfg: Config, manifest: SceneManifest) -> None:
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)
    start = run.drones[0].pos.copy()

    for _ in range(20):
        run.tick()

    assert run.t == pytest.approx(20 * cfg.sim.dt)
    assert not np.allclose(run.drones[0].pos, start), "drone never left the pad during takeoff"


@pytest.mark.slow
def test_run_finishes_within_the_timeout_and_lands_every_drone(
    cfg: Config, manifest: SceneManifest
) -> None:
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 2, cfg, seed=0)

    run.run(max_t_s=90.0)

    assert run.done
    assert run.phase is Phase.DONE
    assert run.controller.landed == frozenset({0, 1})
    for pad, state in zip(run.pads, run.drones, strict=True):
        np.testing.assert_allclose(state.pos[:2], pad[:2], atol=0.1)
        assert state.pos[2] == pytest.approx(0.0, abs=0.05)
    assert run.coverage_ground_band > 0.0
    assert run.coverage_total > 0.0


def test_run_stops_at_the_timeout_even_if_not_done(cfg: Config, manifest: SceneManifest) -> None:
    """A max_t_s far too short to finish still returns, having ticked up to it."""
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)

    run.run(max_t_s=0.2)

    assert run.t >= 0.2 - cfg.sim.dt
    assert run.t < 0.2 + cfg.sim.dt + 1e-9


@pytest.mark.slow
def test_finished_mission_keeps_its_drones_parked_on_further_ticks(
    cfg: Config, manifest: SceneManifest
) -> None:
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)
    run.run(max_t_s=90.0)
    assert run.done
    resting = [d.pos.copy() for d in run.drones]

    for _ in range(10):
        run.tick()

    for before, state in zip(resting, run.drones, strict=True):
        np.testing.assert_allclose(state.pos, before, atol=1e-9)


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------
def test_coverage_starts_at_zero_and_never_decreases_over_a_few_ticks(
    cfg: Config, manifest: SceneManifest
) -> None:
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)
    prev_ground, prev_total = run.coverage_ground_band, run.coverage_total
    assert 0.0 <= prev_ground <= 1.0
    assert 0.0 <= prev_total <= 1.0

    for _ in range(30):
        run.tick()
        ground, total = run.coverage_ground_band, run.coverage_total
        assert ground >= prev_ground - 1e-9
        assert total >= prev_total - 1e-9
        prev_ground, prev_total = ground, total


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_trajectories_over_a_partial_run(
    cfg: Config, manifest: SceneManifest
) -> None:
    """Two runs built from the same seed on the same scene tick identically."""
    geometry = load_geometry(manifest)
    run_a = MissionRun(manifest, geometry, 2, cfg, seed=3)
    run_b = MissionRun(manifest, geometry, 2, cfg, seed=3)

    for _ in range(40):
        run_a.tick()
        run_b.tick()

    for state_a, state_b in zip(run_a.drones, run_b.drones, strict=True):
        np.testing.assert_array_equal(state_a.pos, state_b.pos)
        np.testing.assert_array_equal(state_a.vel, state_b.vel)
        assert state_a.battery == pytest.approx(state_b.battery)
    assert run_a.phase is run_b.phase
    assert run_a.coverage_ground_band == pytest.approx(run_b.coverage_ground_band)


@pytest.mark.slow
def test_different_seeds_can_diverge_in_sensor_noise_but_not_in_phase_progress(
    cfg: Config, manifest: SceneManifest
) -> None:
    """Different seeds still reach DONE; the seed only perturbs sensor noise, not the outcome."""
    geometry = load_geometry(manifest)
    run_a = MissionRun(manifest, geometry, 1, cfg, seed=1)
    run_b = MissionRun(manifest, geometry, 1, cfg, seed=2)

    run_a.run(max_t_s=90.0)
    run_b.run(max_t_s=90.0)

    assert run_a.done
    assert run_b.done
