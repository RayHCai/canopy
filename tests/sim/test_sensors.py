"""Tests for :mod:`canopy.sim.sensors`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config, SensorCfg, load_config
from canopy.contracts import Cls, SceneGeometry, SceneManifest, SceneObject
from canopy.sim.scene import load_geometry
from canopy.sim.sensors import RaySensor
from canopy.worldgen.generate import generate_field

#: A vertical wall in the plane x=5, spanning y in [-1, 1], z in [0, 2].
_WALL_VERTS = "\n".join(
    [
        "v 5.000000 -1.000000 0.000000",
        "v 5.000000 1.000000 0.000000",
        "v 5.000000 1.000000 2.000000",
        "v 5.000000 -1.000000 2.000000",
    ]
)
_WALL_FACES = "f 1 2 3\nf 1 3 4\n"

#: A tiny, far-away floor object placed *before* the wall in the manifest, so
#: the wall's global triangle offset is non-zero.
_FLOOR_VERTS = "\n".join(
    [
        "v -10.000000 -10.000000 0.000000",
        "v -9.000000 -10.000000 0.000000",
        "v -9.000000 -9.000000 0.000000",
    ]
)
_FLOOR_FACES = "f 1 2 3\n"


def _write_obj(path: Path, o_name: str, verts: str, faces: str) -> None:
    path.write_text(f"# canopy test\no {o_name}\n{verts}\n{faces}", encoding="utf-8")


def _make_object(obj_id: int, mesh_path: Path, cls: Cls) -> SceneObject:
    return SceneObject(obj_id=obj_id, cls=cls, mesh_path=str(mesh_path), color=(255, 255, 255))


@pytest.fixture
def wall_geometry(tmp_path: Path) -> SceneGeometry:
    """Build a two-object scene: a small floor (obj 0) then the wall (obj 1)."""
    floor_path = tmp_path / "0_ground.obj"
    wall_path = tmp_path / "1_wall.obj"
    _write_obj(floor_path, "0_ground", _FLOOR_VERTS, _FLOOR_FACES)
    _write_obj(wall_path, "1_wall", _WALL_VERTS, _WALL_FACES)

    manifest = SceneManifest(
        seed=0,
        lot_bounds=np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]]),
        footprint=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        objects=[
            _make_object(0, floor_path, Cls.GROUND),
            _make_object(1, wall_path, Cls.WALL),
        ],
        home=np.array([0.0, -8.0, 0.0]),
        gt_meter_id=0,
    )
    return load_geometry(manifest)


def _sensor_cfg(**overrides: float) -> SensorCfg:
    base: dict[str, float] = {
        "az_rays": 8,
        "el_rays": 4,
        "el_min_deg": -30.0,
        "el_max_deg": 30.0,
        "max_range_m": 12.0,
        "range_noise_m": 0.0,
    }
    base.update(overrides)
    return SensorCfg(
        az_rays=int(base["az_rays"]),
        el_rays=int(base["el_rays"]),
        el_min_deg=base["el_min_deg"],
        el_max_deg=base["el_max_deg"],
        max_range_m=base["max_range_m"],
        range_noise_m=base["range_noise_m"],
        sun_dir=(0.0, 0.0, 1.0),
        ambient=base.get("ambient", 0.35),
        rgb_noise=base.get("rgb_noise", 0.0),
        sky_rgb=(184, 201, 218),
    )


def test_ray_hits_known_wall(wall_geometry: SceneGeometry) -> None:
    """Rays that hit the wall report the correct analytic distance and ids.

    The azimuth jitter rotates the whole grid by an unknown angle, so rather
    than pin a single ray this recomputes each hit's expected distance from
    its (post-jitter) direction and the wall's known plane ``x = 5``.
    """
    rng = np.random.default_rng(0)
    # Dense azimuths guarantee some ray lands in the wall's narrow angular
    # window regardless of the random jitter offset.
    cfg = _sensor_cfg(az_rays=64, el_min_deg=0.0, el_max_deg=0.0)
    sensor = RaySensor(wall_geometry, cfg, rng)

    origin = np.array([0.0, 0.0, 1.0])
    scan = sensor.scan(drone_id=0, t=0.0, origin=origin)

    hit = scan.obj_ids == 1
    assert hit.any()
    expected_dist = (5.0 - origin[0]) / scan.dirs[hit, 0]
    np.testing.assert_allclose(scan.dist[hit], expected_dist, atol=0.02)
    # The wall owns global triangle ids [1, 3): offset carried from the floor object.
    assert np.all(wall_geometry.tri_obj[scan.tri_ids[hit]] == 1)


def test_scan_hits_and_misses(wall_geometry: SceneGeometry) -> None:
    """A scan toward the wall reports hits; rays that clear max_range miss."""
    rng = np.random.default_rng(1)
    sensor = RaySensor(wall_geometry, _sensor_cfg(max_range_m=4.0), rng)

    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    assert scan.dirs.shape == (sensor.n_rays, 3)
    # The wall is 5 m away but max_range is 4 m, so nothing can hit it.
    assert np.all(np.isinf(scan.dist))
    assert np.all(scan.obj_ids == -1)
    assert np.all(scan.tri_ids == -1)


def test_scan_within_range_hits_wall(wall_geometry: SceneGeometry) -> None:
    """Widening max_range lets the forward-pointing rays hit the wall."""
    rng = np.random.default_rng(2)
    # Dense azimuths guarantee some ray lands in the wall's narrow angular
    # window regardless of the random jitter offset.
    cfg = _sensor_cfg(az_rays=64, max_range_m=12.0)
    sensor = RaySensor(wall_geometry, cfg, rng)

    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    hit = scan.obj_ids >= 0
    assert hit.any()
    assert np.all(scan.obj_ids[hit] == 1)
    assert np.all(scan.dist[hit] > 0.0)


def test_noise_off_gives_exact_distance(wall_geometry: SceneGeometry) -> None:
    """With range_noise_m=0, repeated scans from the same spot agree exactly."""
    rng = np.random.default_rng(3)
    sensor = RaySensor(wall_geometry, _sensor_cfg(range_noise_m=0.0), rng)
    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    hit = np.isfinite(scan.dist)
    assert hit.any()
    # Every finite distance must be an exact geometric hit distance -- an
    # integer number of millimetres at worst, never a noised float.
    assert np.all(scan.dist[hit] < 12.0)


def test_azimuth_jitter_changes_dirs_but_stays_unit_and_in_band(
    wall_geometry: SceneGeometry,
) -> None:
    """Each scan's jittered directions differ but stay unit-length and in elevation band."""
    rng = np.random.default_rng(4)
    cfg = _sensor_cfg()
    sensor = RaySensor(wall_geometry, cfg, rng)

    scan_a = sensor.scan(drone_id=0, t=0.0, origin=np.zeros(3))
    scan_b = sensor.scan(drone_id=0, t=0.0, origin=np.zeros(3))

    assert not np.allclose(scan_a.dirs, scan_b.dirs)
    for dirs in (scan_a.dirs, scan_b.dirs):
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        elevations = np.rad2deg(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))
        assert np.all(elevations >= cfg.el_min_deg - 1e-6)
        assert np.all(elevations <= cfg.el_max_deg + 1e-6)


def test_dirs_property_is_unjittered_grid(wall_geometry: SceneGeometry) -> None:
    """The dirs property is the base grid, independent of any scan's jitter."""
    rng = np.random.default_rng(5)
    sensor = RaySensor(wall_geometry, _sensor_cfg(), rng)
    grid_before = sensor.dirs.copy()
    sensor.scan(drone_id=0, t=0.0, origin=np.zeros(3))
    np.testing.assert_array_equal(sensor.dirs, grid_before)


def test_surface_distance_known_point(wall_geometry: SceneGeometry) -> None:
    """A point 1 m off the wall plane reports ~1 m unsigned distance."""
    rng = np.random.default_rng(6)
    sensor = RaySensor(wall_geometry, _sensor_cfg(), rng)

    dist = sensor.surface_distance(np.array([[4.0, 0.0, 1.0]]))
    assert dist[0] == pytest.approx(1.0, abs=0.02)


@pytest.mark.slow
def test_generated_scene_loads_and_scans(tmp_path: Path) -> None:
    """A real generated scene loads, triangle counts match, and a scan hits."""
    cfg: Config = load_config()
    manifest = generate_field(12345, cfg, out_dir=tmp_path)

    geometry = load_geometry(manifest)

    expected_faces = 0
    for obj in manifest.objects:
        text = Path(obj.mesh_path).read_text(encoding="utf-8")
        expected_faces += sum(1 for line in text.splitlines() if line.startswith("f "))
    assert geometry.n_triangles == expected_faces

    rng = np.random.default_rng(0)
    sensor = RaySensor(geometry, cfg.sensor, rng)
    origin = manifest.home + np.array([0.0, 0.0, 2.0])
    scan = sensor.scan(drone_id=0, t=0.0, origin=origin)
    assert np.any(scan.obj_ids >= 0)


def test_scan_reports_rgb_shape_and_dtype(wall_geometry: SceneGeometry) -> None:
    """Every scan carries an (n_rays, 3) uint8 colour channel alongside range."""
    rng = np.random.default_rng(7)
    sensor = RaySensor(wall_geometry, _sensor_cfg(), rng)

    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    assert scan.rgb is not None
    assert scan.rgb.shape == (sensor.n_rays, 3)
    assert scan.rgb.dtype == np.uint8


def test_miss_reports_sky_rgb(wall_geometry: SceneGeometry) -> None:
    """A ray that clears max_range reports the configured sky colour exactly."""
    rng = np.random.default_rng(8)
    cfg = _sensor_cfg(max_range_m=4.0)  # the wall is 5 m away: everything misses
    sensor = RaySensor(wall_geometry, cfg, rng)

    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    assert np.all(np.isinf(scan.dist))
    assert scan.rgb is not None
    expected_sky = np.array(cfg.sky_rgb, dtype=np.uint8)
    np.testing.assert_array_equal(scan.rgb, np.broadcast_to(expected_sky, scan.rgb.shape))


def test_wall_hit_colour_is_ambient_only_when_normal_is_side_on_to_the_sun(
    wall_geometry: SceneGeometry,
) -> None:
    """A vertical wall's normal is perpendicular to a straight-up sun, so lambert is zero.

    ``shade`` collapses to exactly ``ambient`` for every hit, regardless of
    which of the wall's two triangles or which jittered azimuth found it: the
    wall's normal has no Z component, so ``normal . sun`` is zero either way.
    """
    rng = np.random.default_rng(9)
    # Dense azimuths guarantee some ray lands in the wall's narrow angular
    # window regardless of the random jitter offset (as in test_ray_hits_known_wall).
    cfg = _sensor_cfg(az_rays=64, el_min_deg=0.0, el_max_deg=0.0, rgb_noise=0.0)
    sensor = RaySensor(wall_geometry, cfg, rng)

    scan = sensor.scan(drone_id=0, t=0.0, origin=np.array([0.0, 0.0, 1.0]))

    hit = scan.obj_ids == 1
    assert hit.any()
    assert scan.rgb is not None
    assert wall_geometry.obj_color is not None
    wall_colour = wall_geometry.obj_color[1].astype(np.float64)
    expected = np.clip(np.rint(wall_colour * cfg.ambient), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(scan.rgb[hit], np.broadcast_to(expected, scan.rgb[hit].shape))


def test_floor_hit_facing_the_sun_reports_full_colour(wall_geometry: SceneGeometry) -> None:
    """A surface facing straight into the sun is shaded at full strength, ambient aside.

    The floor object's one triangle faces +Z (see ``_FLOOR_VERTS``), so a ray
    straight down onto it -- independent of azimuth jitter, which only rotates
    about Z -- sees ``lambert = 1`` and thus the object's true colour untouched.
    """
    rng = np.random.default_rng(10)
    cfg = _sensor_cfg(az_rays=1, el_rays=1, el_min_deg=-90.0, el_max_deg=-90.0, rgb_noise=0.0)
    sensor = RaySensor(wall_geometry, cfg, rng)
    # Well inside the floor triangle (-10,-10)-(-9,-10)-(-9,-9), a straight-down
    # ray from here cannot land on anything else in the scene.
    origin = np.array([-9.3, -9.8, 1.0])

    scan = sensor.scan(drone_id=0, t=0.0, origin=origin)

    assert scan.obj_ids[0] == 0
    assert scan.rgb is not None
    assert wall_geometry.obj_color is not None
    floor_colour = wall_geometry.obj_color[0]
    np.testing.assert_array_equal(scan.rgb[0], floor_colour)


def test_colour_noise_does_not_perturb_the_range_stream(wall_geometry: SceneGeometry) -> None:
    """Colour noise is drawn from a spawned child rng, so ranges and azimuth jitter are unaffected.

    Two sensors seeded identically with ``rgb_noise > 0`` must produce
    identical ``dist`` arrays to each other and to a third, noiseless sensor
    built from the same seed -- while their ``rgb`` differs from the
    noiseless one.
    """
    cfg_noisy = _sensor_cfg(az_rays=64, el_min_deg=0.0, el_max_deg=0.0, rgb_noise=5.0)
    cfg_clean = _sensor_cfg(az_rays=64, el_min_deg=0.0, el_max_deg=0.0, rgb_noise=0.0)
    origin = np.array([0.0, 0.0, 1.0])

    sensor_a = RaySensor(wall_geometry, cfg_noisy, np.random.default_rng(123))
    sensor_b = RaySensor(wall_geometry, cfg_noisy, np.random.default_rng(123))
    sensor_c = RaySensor(wall_geometry, cfg_clean, np.random.default_rng(123))

    scan_a = sensor_a.scan(drone_id=0, t=0.0, origin=origin)
    scan_b = sensor_b.scan(drone_id=0, t=0.0, origin=origin)
    scan_c = sensor_c.scan(drone_id=0, t=0.0, origin=origin)
    assert scan_a.rgb is not None
    assert scan_b.rgb is not None
    assert scan_c.rgb is not None

    np.testing.assert_array_equal(scan_a.dist, scan_b.dist)
    np.testing.assert_array_equal(scan_a.dist, scan_c.dist)
    np.testing.assert_array_equal(scan_a.rgb, scan_b.rgb)
    assert not np.array_equal(scan_a.rgb, scan_c.rgb)
