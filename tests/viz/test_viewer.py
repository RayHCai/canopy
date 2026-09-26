"""The desktop viewer session: a real mapping mission, headless.

Generating a property and building its raycasting scene costs roughly a
second, so one seed is shared across every test in this module (module-scoped
fixture) and the tests that advance simulated time are marked ``slow``.
"""

from __future__ import annotations

import base64
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import Cls
from canopy.errors import ConfigError, SimulationError
from canopy.site import load_site_rules
from canopy.viz.viewer import MAX_TIME_SCALE, MIN_TIME_SCALE, ViewerSession, battery_model


@pytest.fixture(scope="module")
def session(cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> ViewerSession:
    """One viewer session, seeded for a reproducible, small(ish) property."""
    scene_dir = tmp_path_factory.mktemp("viewer_scenes")
    return ViewerSession(cfg, drones=3, seed=1, scene_dir=scene_dir)


def _decode(b64: str, dtype: np.dtype[np.generic]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def test_world_decodes_to_the_right_triangle_counts(session: ViewerSession) -> None:
    world = session.world()
    assert world["seed"] == session.seed

    offsets = [obj["tri_offset"] for obj in world["objects"]]
    assert offsets == sorted(offsets)

    total_tris = 0
    for obj in world["objects"]:
        positions = _decode(obj["positions"], np.dtype(np.float32))
        indices = _decode(obj["indices"], np.dtype(np.uint32))
        assert positions.size % 3 == 0
        assert indices.size % 3 == 0
        n_verts = positions.size // 3
        assert indices.max(initial=0) < n_verts
        total_tris += indices.size // 3

    assert offsets[0] == 0
    # Each object's own triangle count matches the gap to the next offset (or
    # the running total, for the last object).
    running = 0
    for obj, next_offset in zip(world["objects"], [*offsets[1:], total_tris], strict=True):
        n_tris = _decode(obj["indices"], np.dtype(np.uint32)).size // 3
        running += n_tris
        assert running == next_offset


@pytest.mark.slow
def test_advancing_reveals_valid_never_background_triangles(session: ViewerSession) -> None:
    world = session.world()
    background_ranges = [
        (
            obj["tri_offset"],
            obj["tri_offset"] + _decode(obj["indices"], np.dtype(np.uint32)).size // 3,
        )
        for obj in world["objects"]
        if obj["background"]
    ]
    total_tris = sum(
        _decode(obj["indices"], np.dtype(np.uint32)).size // 3 for obj in world["objects"]
    )

    seen: set[int] = set()
    coverage_history: list[float] = []
    phases: list[str] = []
    for _ in range(30 * 60):  # 30 simulated seconds at 60 "page" fps
        frame = session.advance(1.0 / 60.0)
        revealed = _decode(frame["revealed"], np.dtype(np.int32))
        assert revealed.tolist() == sorted(set(revealed.tolist())), "ids must be unique per frame"
        for tri_id in revealed.tolist():
            assert 0 <= tri_id < total_tris
            assert not any(lo <= tri_id < hi for lo, hi in background_ranges)
            assert tri_id not in seen, "an id must not be revealed twice"
            seen.add(tri_id)
        coverage_history.append(frame["coverage"]["ground_band"])
        phases.append(frame["phase"])

    # Coverage never decreases.
    for a, b in itertools.pairwise(coverage_history):
        assert b >= a - 1e-9
    assert phases[0] in {"takeoff", "explore"}


def test_set_drone_count_reuses_the_property(session: ViewerSession) -> None:
    before = session.scene()["epoch"]
    geometry_before = session.world()

    scene = session.set_drone_count(2)
    assert scene["epoch"] == before + 1
    assert scene["drones"] == 2
    assert session.sim_time_s == 0.0

    # Same seed: the cached property must not have been regenerated.
    world = session.world()
    assert world is geometry_before
    assert world["seed"] == scene["seed"]

    session.set_drone_count(3)  # restore for later tests in this module


def test_restart_rewinds_without_regenerating(session: ViewerSession) -> None:
    session.advance(4.1 * session.config.sim.dt)
    before = session.scene()
    world_before = session.world()

    scene = session.restart()
    assert scene["epoch"] == before["epoch"] + 1
    assert (scene["seed"], scene["drones"]) == (before["seed"], before["drones"])
    assert session.sim_time_s == 0.0
    assert session.world() is world_before


def test_new_scene_changes_the_seed(session: ViewerSession) -> None:
    old_seed = session.seed
    scene = session.new_scene(seed=old_seed + 12345)
    assert scene["seed"] != old_seed
    assert session.world()["seed"] == scene["seed"]


def test_invalid_drone_count_raises_config_error(session: ViewerSession) -> None:
    with pytest.raises(ConfigError):
        session.set_drone_count(0)
    with pytest.raises(ConfigError):
        session.set_drone_count(session.config.viewer.max_drones + 1)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_bad_wall_dt_raises_simulation_error(session: ViewerSession, bad: float) -> None:
    with pytest.raises(SimulationError):
        session.advance(bad)


def test_time_scale_multiplies_simulated_time(session: ViewerSession) -> None:
    session.set_drone_count(session.n_drones)  # reset: sim time back to zero
    dt = session.config.sim.dt
    try:
        scene = session.set_time_scale(2.0)
        assert scene["time_scale"] == 2.0
        # Four ticks of wall time become eight of simulated time. The spare
        # fraction keeps float drift off a tick boundary, and the whole stays
        # under the per-call wall-clock clamp.
        session.advance(4.1 * dt)
        assert session.sim_time_s == pytest.approx(8 * dt)
    finally:
        session.set_time_scale(1.0)


def test_time_scale_survives_a_reset(session: ViewerSession) -> None:
    try:
        session.set_time_scale(MAX_TIME_SCALE)
        scene = session.set_drone_count(session.n_drones)
        assert scene["time_scale"] == MAX_TIME_SCALE
        assert (scene["min_time_scale"], scene["max_time_scale"]) == (
            MIN_TIME_SCALE,
            MAX_TIME_SCALE,
        )
    finally:
        session.set_time_scale(1.0)


@pytest.mark.parametrize(
    "bad", [MIN_TIME_SCALE / 2, MAX_TIME_SCALE * 2, 0.0, -1.0, float("nan"), float("inf")]
)
def test_invalid_time_scale_raises_config_error(session: ViewerSession, bad: float) -> None:
    with pytest.raises(ConfigError):
        session.set_time_scale(bad)
    assert session.time_scale == 1.0


def test_scene_dir_defaults_and_is_configurable(cfg: Config, tmp_path: Path) -> None:
    scene_dir = tmp_path / "custom"
    ViewerSession(cfg, drones=1, seed=99, scene_dir=scene_dir)
    assert (scene_dir / "99").is_dir()


def test_landed_is_a_false_bool_on_every_drone_at_launch(cfg: Config, tmp_path: Path) -> None:
    """Every drone dict carries a ``landed`` bool, false right after take-off."""
    session = ViewerSession(cfg, drones=2, seed=321, scene_dir=tmp_path)
    frame = session.advance(0.01)
    assert len(frame["drones"]) == 2
    for drone in frame["drones"]:
        assert isinstance(drone["landed"], bool)
        assert drone["landed"] is False


@pytest.mark.slow
def test_inline_sim_time_never_outruns_accumulated_playback(cfg: Config, tmp_path: Path) -> None:
    """Without :meth:`ViewerSession.start`, ``t`` tracks wall time and never jumps ahead of it."""
    session = ViewerSession(cfg, drones=1, seed=555, scene_dir=tmp_path)
    dt = 0.01
    accumulated = 0.0
    prev_t = -1.0
    for _ in range(400):  # 4 s simulated, well over one control tick each call
        accumulated += dt  # time_scale is 1.0 by default
        frame = session.advance(dt)
        assert frame["t"] <= accumulated + 1e-6
        assert frame["t"] >= prev_t - 1e-9
        prev_t = frame["t"]


@pytest.mark.slow
def test_started_session_advances_fast_and_moves_time_forward(cfg: Config, tmp_path: Path) -> None:
    """With the simulation thread running, ``advance`` never waits on a replan.

    A replan (the 1 Hz clearance/frontier/assignment pass) takes the better
    part of a second; if a frame call ever blocked on one, this would be the
    test to catch it.
    """
    session = ViewerSession(cfg, drones=1, seed=777, scene_dir=tmp_path)
    try:
        session.start()
        deadline = time.monotonic() + 5.0
        moved_t = -1.0
        while time.monotonic() < deadline:
            started = time.perf_counter()
            frame = session.advance(0.05)
            elapsed = time.perf_counter() - started
            assert elapsed < 0.05, f"advance() took {elapsed:.3f}s with the thread running"
            if frame["t"] > 0.0:
                moved_t = frame["t"]
                break
            time.sleep(0.02)
        assert moved_t > 0.0, "sim time never advanced after start() within the deadline"

        times = [moved_t]
        seen: set[int] = set()
        for _ in range(20):
            frame = session.advance(0.05)
            revealed = np.frombuffer(base64.b64decode(frame["revealed"]), dtype=np.int32)
            assert revealed.tolist() == sorted(set(revealed.tolist())), "ids must be sorted, unique"
            assert seen.isdisjoint(revealed.tolist()), "an id must not be revealed twice"
            seen.update(revealed.tolist())
            times.append(frame["t"])
        assert times[-1] > moved_t
        for a, b in itertools.pairwise(times):
            assert b >= a - 1e-9
    finally:
        session.close()


def test_close_stops_the_thread_and_is_idempotent(cfg: Config, tmp_path: Path) -> None:
    session = ViewerSession(cfg, drones=1, seed=888, scene_dir=tmp_path)
    try:
        session.start()
        session.advance(0.01)
        started = time.perf_counter()
        session.close()
        assert time.perf_counter() - started < 2.0, "close() should join the thread promptly"
        # A second close(), and further advances, must not raise.
        session.close()
        session.advance(0.01)
    finally:
        session.close()


def test_frame_always_has_a_detections_list(cfg: Config, tmp_path: Path) -> None:
    """advance() always includes a detections list, even before anything is classified."""
    session = ViewerSession(cfg, drones=1, seed=42, scene_dir=tmp_path)
    frame = session.advance(0.01)
    assert isinstance(frame["detections"], list)


@pytest.mark.slow
def test_detections_match_documented_shape_and_configured_colours(
    cfg: Config, tmp_path: Path
) -> None:
    """Every frame's detections match the documented shape, and colours match the viewer config.

    Seed 1 with 3 drones (the same scene the module ``session`` fixture uses)
    reliably gets a drone close enough to the house that a METER is classified
    within a few seconds of simulated time; running to 25 s leaves ample
    margin and matches what a real, un-started viewer session would show.
    """
    session = ViewerSession(cfg, drones=3, seed=1, scene_dir=tmp_path)
    expected_keys = {"id", "cls", "color", "center", "size", "yaw", "confidence"}
    color_by_cls = {c.cls: [v / 255.0 for v in c.rgb] for c in cfg.viewer.detection_colors}
    default_color = [v / 255.0 for v in cfg.viewer.detection_default_rgb]

    dt = cfg.sim.dt
    seen_classes: set[str] = set()
    frame = session.advance(dt)
    for _ in range(round(25.0 / dt) - 1):
        frame = session.advance(dt)
        detections = frame["detections"]
        assert isinstance(detections, list)
        for det in detections:
            assert set(det) == expected_keys
            assert isinstance(det["id"], int)
            assert isinstance(det["cls"], str)
            assert det["cls"] in Cls.__members__
            assert isinstance(det["color"], list)
            assert isinstance(det["center"], list)
            assert isinstance(det["size"], list)
            assert len(det["color"]) == len(det["center"]) == len(det["size"]) == 3
            assert isinstance(det["yaw"], float)
            assert isinstance(det["confidence"], float)
            # Documented as [0, 1), but confidence_hits underflow can round it
            # to exactly 1.0 -- see the accompanying report.
            assert 0.0 <= det["confidence"] <= 1.0
            expected_color = color_by_cls.get(det["cls"], default_color)
            assert det["color"] == pytest.approx(expected_color)
            seen_classes.add(det["cls"])

    assert seen_classes, "no object was ever classified in 25 s of simulated time"
    assert "METER" in seen_classes
    final_meters = [d for d in frame["detections"] if d["cls"] == "METER"]
    assert final_meters, "no METER detection survived to the final frame"
    for det in final_meters:
        assert det["color"] == pytest.approx(color_by_cls["METER"])


def test_frame_site_is_none_right_after_construction(cfg: Config, tmp_path: Path) -> None:
    """A fresh session has not finished mapping, so there is nothing to site yet."""
    session = ViewerSession(cfg, drones=1, seed=1, scene_dir=tmp_path)
    frame = session.advance(0.05)
    assert "site" in frame
    assert frame["site"] is None


def test_battery_model_bounding_box_matches_rules_yaml() -> None:
    """The battery mesh's box must match ``rules.yaml`` battery, in the site frame.

    +X out of the wall (front), the back on the wall (x = 0), width centred on
    ``y = 0`` -- see :func:`canopy.viz.viewer.battery_model`.
    """
    battery = load_site_rules().battery
    model = battery_model()
    vertices = np.array(model["vertices"], dtype=np.float64).reshape(-1, 3)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    size = hi - lo

    assert size[0] == pytest.approx(battery.depth_m, abs=0.02)
    assert size[1] == pytest.approx(battery.width_m, abs=0.02)
    assert size[2] == pytest.approx(battery.height_m, abs=0.02)
    assert (lo[1] + hi[1]) / 2.0 == pytest.approx(0.0, abs=0.02)
    assert lo[0] == pytest.approx(0.0, abs=0.02)
    assert hi[0] > lo[0]


@pytest.mark.slow
def test_a_completed_mission_yields_json_safe_battery_sites(cfg: Config, tmp_path: Path) -> None:
    """Run a real mission inline to completion and check the ``site`` payload's shape.

    Ticked with :meth:`ViewerSession.advance` alone (no ``start()``), so this
    runs inline rather than against a background sim thread; ~100 s of
    simulated time at 0.25 s per call takes on the order of a minute of wall
    time on a modest machine.
    """
    session = ViewerSession(cfg, drones=3, seed=42, scene_dir=tmp_path)
    frame = session.advance(0.25)
    for _ in range(2000):  # generous cap; the mission should finish well within this
        if frame["site"] is not None:
            break
        frame = session.advance(0.25)
    else:
        pytest.fail("mission did not finish siting within the iteration cap")

    site_payload = frame["site"]
    sites = site_payload["sites"]
    assert 1 <= len(sites) <= 3

    viewer = cfg.viewer
    passing_rgb = [c / 255.0 for c in viewer.site_rgb]
    warning_rgb = [c / 255.0 for c in viewer.site_warning_rgb]
    for site in sites:
        assert isinstance(site["rank"], int)
        assert isinstance(site["pos"], list)
        assert len(site["pos"]) == 3
        assert isinstance(site["yaw"], float)
        assert isinstance(site["meter"], list)
        assert len(site["meter"]) == 3
        assert isinstance(site["warnings"], list)
        expected = warning_rgb if site["warnings"] else passing_rgb
        assert site["color"] == pytest.approx(expected)
        for value in site["breakdown"].values():
            assert value is None or (isinstance(value, float) and np.isfinite(value))

    json.dumps(frame)  # the whole frame must round-trip through JSON
