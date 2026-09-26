"""End-to-end mission tests: a swarm mapping a small hand-built house.

The scene is a ground plane plus one closed box "house", built as raw OBJ text
rather than through :func:`canopy.worldgen.generate.generate_field` (worldgen
is mid-refactor elsewhere). It is deliberately tiny -- a 12 x 8 x 6 m lot -- so
a full mission (takeoff, explore, return) finishes in a few seconds of sim
time and the test itself runs quickly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config
from canopy.contracts import Cls, DroneState, MapState, SceneManifest, SceneObject
from canopy.planning.mission import MissionController, Phase
from canopy.planning.run import MissionRun
from canopy.planning.safety import geofence_box
from canopy.sim.scene import load_geometry
from canopy.worldgen import launch_pads

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
    """Build a closed box from ``lo`` to ``hi``, as OBJ vertex/face text."""
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
    """Build a flat quad (two triangles) covering ``lo_xy`` to ``hi_xy`` at z=0."""
    verts = [
        (lo_xy[0], lo_xy[1], 0.0),
        (hi_xy[0], lo_xy[1], 0.0),
        (hi_xy[0], hi_xy[1], 0.0),
        (lo_xy[0], hi_xy[1], 0.0),
    ]
    verts_text = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts)
    faces_text = "f 1 2 3\nf 1 3 4\n"
    return verts_text, faces_text


def _build_scene(tmp_path: Path) -> SceneManifest:
    """Build a ground plane plus one closed box house, as a hand-built manifest."""
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


def _build_scene_with_fence_beside_pad(tmp_path: Path, cfg: Config) -> SceneManifest:
    """Ground plane, house and a fence beside the launch pad's column.

    The fence stands ``0.4`` m from home along the pad line's own axis, ``1.9``
    m tall: close enough that ``pad + [0, 0, takeoff_altitude_m]`` -- the old
    RTH's only target -- reads below ``inflation_m`` clearance, but with
    ``drone_radius_m`` clearance still, so a higher hover point in the same
    column can drop straight down to it. ``launch_clear_radius_m`` is widened
    so the drone's post-takeoff position is not itself isolated from the rest
    of the map at planning resolution by the fence's own halo -- a scene
    artefact of a fence this close to a 1-cell-wide column, not something the
    fix is responsible for.
    """
    ground_path = tmp_path / "0_ground.obj"
    house_path = tmp_path / "1_house.obj"
    fence_path = tmp_path / "2_fence.obj"
    g_verts, g_faces = _ground_mesh(_LOT[0, :2], _LOT[1, :2])
    h_verts, h_faces = _box_mesh(_HOUSE_LO, _HOUSE_HI)
    fence_lo = np.array([-1.5, _HOME[1] + 0.4, 0.0])
    fence_hi = np.array([1.5, _HOME[1] + 0.6, 1.9])
    f_verts, f_faces = _box_mesh(fence_lo, fence_hi)
    _write_obj(ground_path, "0_ground", g_verts, g_faces)
    _write_obj(house_path, "1_house", h_verts, h_faces)
    _write_obj(fence_path, "2_fence", f_verts, f_faces)

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
            SceneObject(obj_id=2, cls=Cls.WALL, mesh_path=str(fence_path), color=(180, 180, 180)),
        ],
        home=_HOME.copy(),
        gt_meter_id=-1,
    )


@pytest.mark.slow
def test_mission_lands_when_the_pad_hover_is_blocked_beside_the_pad(
    cfg: Config, tmp_path: Path
) -> None:
    """RTH must fall back to a clear point higher in the pad column.

    ``drone_radius_m`` and ``margin_m`` are traded off (``inflation_m`` held
    fixed) only to land the naive hover point's clearance -- fixed by the
    scene's geometry and the map's voxel size, not tunable directly -- between
    ``drone_radius_m`` and ``inflation_m``. Every other planning decision reads
    ``inflation_m``, so this leaves the mission's behaviour otherwise
    unchanged from the ``test_mission_explores_and_returns_safely`` scene.
    """
    cfg = replace(
        cfg,
        safety=replace(cfg.safety, drone_radius_m=0.2, margin_m=0.4),
        planner=replace(cfg.planner, launch_clear_radius_m=2.5),
    )
    manifest = _build_scene_with_fence_beside_pad(tmp_path, cfg)
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 1, cfg, seed=0)

    naive_hover_checked = False
    while not run.done and run.t < 90.0:
        run.tick()
        if not naive_hover_checked and run.phase is Phase.RETURN:
            naive_hover_checked = True
            pad = run.pads[0]
            naive_hover = pad + np.array([0.0, 0.0, cfg.planner.takeoff_altitude_m])
            clearance = run.controller._clearance
            assert clearance is not None
            naive_clearance = float(clearance.clearance_at(naive_hover[None, :])[0])
            # This is the old RTH's only target, and it must be too tight for
            # normal planning (below inflation_m) but safe enough to drop
            # straight down onto (at or above drone_radius_m) -- otherwise
            # this scene is not exercising the fixed code path at all.
            assert naive_clearance < cfg.safety.inflation_m
            assert naive_clearance >= cfg.safety.drone_radius_m

    assert naive_hover_checked, "mission never reached RETURN"
    assert run.done, f"mission did not finish by t={run.t:.0f}s"
    assert run.controller.landed == frozenset({0})
    np.testing.assert_allclose(run.drones[0].pos[:2], run.pads[0][:2], atol=0.1)
    assert run.drones[0].pos[2] == pytest.approx(0.0, abs=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("n_drones", [1, 3])
def test_mission_explores_and_returns_safely(cfg: Config, tmp_path: Path, n_drones: int) -> None:
    """A full mission on a tiny house: safe throughout, and finishes explored and home."""
    manifest = _build_scene(tmp_path)
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, n_drones, cfg, seed=0)

    fence = geofence_box(manifest.lot_bounds, cfg.safety)
    xy_lo, xy_hi = fence[0, :2], fence[1, :2]
    #: Above this height a drone counts as "airborne": clear of the ground
    #: margin and of the launch column's recovery allowance.
    airborne_z = 0.3
    #: Float round-off on exact clearance/separation reads.
    tol = 0.05

    max_t_s = 90.0
    prev_ground = -1.0
    prev_total = -1.0
    seen_explore = False
    seen_frontier_task = False

    while not run.done and run.t < max_t_s:
        run.tick()

        if run.phase is Phase.EXPLORE:
            seen_explore = True
        if any(task.kind == "frontier" for task in run.controller.tasks.values()):
            seen_frontier_task = True

        airborne = [s for s in run.drones if s.alive and s.pos[2] > airborne_z]
        if airborne:
            positions = np.array([s.pos for s in airborne])
            clearances = run.world.sensor.surface_distance(positions)
            assert np.all(clearances >= cfg.safety.drone_radius_m - 1e-6), (
                f"t={run.t:.2f}: ground-truth collision, clearances={clearances}"
            )
            assert np.all(positions[:, 0] >= xy_lo[0] - tol)
            assert np.all(positions[:, 0] <= xy_hi[0] + tol)
            assert np.all(positions[:, 1] >= xy_lo[1] - tol)
            assert np.all(positions[:, 1] <= xy_hi[1] + tol)

            if len(positions) > 1:
                deltas = positions[:, None, :] - positions[None, :, :]
                dists = np.linalg.norm(deltas, axis=-1)
                iu = np.triu_indices(len(positions), k=1)
                assert np.all(dists[iu] >= cfg.safety.min_separation_m - tol), (
                    f"t={run.t:.2f}: separation violated, min={dists[iu].min():.3f}"
                )

        ground_now = run.mapper.coverage_ground_band
        total_now = run.mapper.coverage_total
        assert ground_now >= prev_ground - 1e-9
        assert total_now >= prev_total - 1e-9
        prev_ground, prev_total = ground_now, total_now

    assert run.phase is Phase.DONE, f"mission did not finish by t={max_t_s:.0f}s"
    assert seen_explore, "mission never entered EXPLORE"
    assert seen_frontier_task, "no drone was ever assigned a frontier task"
    assert prev_ground > 0.0

    for pad, state in zip(run.pads, run.drones, strict=True):
        if not state.alive:
            continue
        np.testing.assert_allclose(state.pos[:2], pad[:2], atol=0.1)
        assert state.pos[2] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_launch_pads_spaced_on_a_line_through_home(cfg: Config, n: int) -> None:
    """``n`` pads on world x, centred on home, spaced ``min_separation_m + margin_m``."""
    home = np.array([1.0, 2.0, 0.0])
    pads = launch_pads(home, n, cfg)

    assert pads.shape == (n, 3)
    np.testing.assert_allclose(pads[:, 2], 0.0)
    np.testing.assert_allclose(pads.mean(axis=0)[:2], home[:2])
    np.testing.assert_allclose(pads[:, 1], home[1])

    spacing = cfg.safety.min_separation_m + cfg.safety.margin_m
    order = np.argsort(pads[:, 0])
    diffs = np.diff(pads[order, 0])
    np.testing.assert_allclose(diffs, spacing)


def test_launch_pads_of_one_is_home(cfg: Config) -> None:
    home = np.array([3.0, -4.0, 0.0])
    pads = launch_pads(home, 1, cfg)
    np.testing.assert_allclose(pads, home.reshape(1, 3))


@pytest.mark.parametrize("n", [1, 3])
def test_launch_boxes_one_per_pad_around_its_takeoff_column(cfg: Config, n: int) -> None:
    """One box per pad, centred on it, spanning the takeoff column."""
    home = np.array([0.0, 0.0, 0.0])
    pads = launch_pads(home, n, cfg)
    controller = MissionController(cfg, _LOT, pads)

    boxes = controller.launch_boxes()
    assert len(boxes) == n

    r = cfg.planner.launch_clear_radius_m
    top = cfg.planner.takeoff_altitude_m + r
    floor = cfg.map.voxel_m
    for pad, (lo, hi) in zip(pads, boxes, strict=True):
        np.testing.assert_allclose(lo, pad + np.array([-r, -r, floor]))
        np.testing.assert_allclose(hi, pad + np.array([r, r, top]))
        # The column contains the vertical takeoff line from pad to hover.
        assert lo[0] <= pad[0] <= hi[0]
        assert lo[1] <= pad[1] <= hi[1]
        assert lo[2] <= floor
        assert hi[2] >= cfg.planner.takeoff_altitude_m


def test_landed_is_empty_on_a_freshly_built_controller(
    cfg: Config, lot: npt.NDArray[np.float64]
) -> None:
    """No drone has flown anywhere yet, so ``landed`` starts empty."""
    pads = launch_pads(np.array([0.0, 0.0, 0.0]), 2, cfg)
    controller = MissionController(cfg, lot, pads)
    assert controller.landed == frozenset()


def _fly_toward_targets(
    states: dict[int, DroneState],
    targets: dict[int, npt.NDArray[np.float64]],
    dt: float,
    v_max: float,
) -> dict[int, DroneState]:
    """Fly each drone straight at ``v_max`` toward its vetted target, as a stand-in dynamics.

    Good enough to drive :meth:`MissionController.step` deterministically without
    pulling in the dynamics/shield-interaction machinery a real
    :class:`~canopy.sim.dynamics.KinematicDynamics` provides -- the controller's
    own shield already vetted ``targets``, so nothing here needs to re-check
    clearance or separation.
    """
    updated: dict[int, DroneState] = {}
    for drone_id, state in states.items():
        target = targets.get(drone_id)
        if target is None:
            updated[drone_id] = replace(state, vel=np.zeros(3))
            continue
        delta = target - state.pos
        dist = float(np.linalg.norm(delta))
        step = min(dist, v_max * dt)
        if dist < 1e-9:
            updated[drone_id] = replace(state, vel=np.zeros(3))
        else:
            direction = delta / dist
            updated[drone_id] = replace(
                state, pos=state.pos + direction * step, vel=direction * (step / dt)
            )
    return updated


@pytest.mark.slow
def test_surplus_drone_idles_then_returns_home_mid_explore(
    cfg: Config,
    lot: npt.NDArray[np.float64],
    make_map: Callable[..., MapState],
) -> None:
    """3 drones, 2 persistent frontier pockets: the odd one out idles, then goes home.

    A short ``idle_return_s`` is used so the whole trip -- idle, then home, then
    touchdown -- fits comfortably before the two remaining pockets are worked
    through and the mission moves to RETURN on its own. The map never changes
    (no mapper here), so both pockets keep offering fresh viewpoints for several
    replans, which is what keeps the mission in EXPLORE long enough to observe
    the surplus drone landing while the other two are still working.
    """
    cfg = replace(cfg, planner=replace(cfg.planner, idle_return_s=2.5))
    pads = np.array([[-9.0, -14.0, 0.0], [9.0, 14.0, 0.0], [0.0, 0.0, 0.0]])
    map_state = make_map(
        unknown=[
            ((-6.0, 3.0, 1.0), (-1.0, 9.0, 5.0)),
            ((1.0, -9.0, 1.0), (6.0, -3.0, 5.0)),
        ]
    )
    controller = MissionController(cfg, lot, pads)
    states = {i: DroneState(i, pads[i].copy(), np.zeros(3), 0.0) for i in range(3)}

    dt = 1.0 / cfg.sim.control_hz
    v_max = cfg.sim.v_max
    t = 0.0
    idle_since: float | None = None
    landed_during_explore = False
    surplus_id: int | None = None
    for _ in range(6000):
        live = list(states.values())
        targets = controller.step(t, live, map_state, map_version=1, coverage_ground_band=0.0)
        states = _fly_toward_targets(states, targets, dt, v_max)
        t += dt

        tasks = controller.tasks
        if surplus_id is None:
            # The first drone the assignment leaves without a frontier.
            idle_now = [i for i, task in tasks.items() if task.kind == "hold"]
            if len(idle_now) == 1 and controller.phase is Phase.EXPLORE:
                surplus_id = idle_now[0]
                idle_since = t
        elif idle_since is not None and t - idle_since < cfg.planner.idle_return_s:
            # Still within the grace period: held, not sent home yet.
            assert tasks[surplus_id].kind == "hold"
            assert surplus_id not in controller.landed
        if controller.phase is Phase.EXPLORE and surplus_id in controller.landed:
            landed_during_explore = True
        if controller.done:
            break

    assert surplus_id is not None, "no drone was ever left idle"
    assert landed_during_explore, "the surplus drone never landed while the others kept exploring"
    assert controller.done
    assert controller.landed == {0, 1, 2}


@pytest.mark.slow
def test_landed_covers_every_drone_once_a_full_mission_finishes(
    cfg: Config, tmp_path: Path
) -> None:
    """After a real mission on the tiny house finishes, every drone shows up as landed."""
    manifest = _build_scene(tmp_path)
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 2, cfg, seed=0)
    assert run.controller.landed == frozenset()

    while not run.done and run.t < 90.0:
        run.tick()

    assert run.done, f"mission did not finish by t={run.t:.0f}s"
    assert run.controller.landed == frozenset(range(2))
