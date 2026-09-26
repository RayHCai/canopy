"""Tests for :mod:`canopy.mapping.mapper`."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config
from canopy.contracts import Cls, Scan, SceneGeometry, SceneManifest, SceneObject
from canopy.mapping.mapper import Mapper


def _manifest(lot: npt.NDArray[np.float64], objects: list[SceneObject]) -> SceneManifest:
    return SceneManifest(
        seed=0,
        lot_bounds=lot,
        footprint=[],
        objects=objects,
        home=np.zeros(3),
        gt_meter_id=-1,
    )


def _geometry() -> SceneGeometry:
    """One wall triangle, object 0."""
    return SceneGeometry(
        vertices=[np.zeros((0, 3))],
        faces=[np.zeros((0, 3), dtype=np.int32)],
        obj_tri_offset=np.array([0, 1], dtype=np.int64),
        tri_obj=np.array([0], dtype=np.int32),
        tri_normal=np.array([[-1.0, 0.0, 0.0]]),
        tri_area=np.array([1.0]),
        tri_centroid=np.array([[2.0, 0.0, 1.0]]),
    )


def _scan(
    origin: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    tri_ids: list[int],
) -> Scan:
    n = dist.shape[0]
    return Scan(
        drone_id=0,
        t=0.0,
        origin=origin,
        dirs=dirs,
        dist=dist,
        obj_ids=np.zeros(n, dtype=np.int32),
        tri_ids=np.array(tri_ids, dtype=np.int32),
    )


def test_version_bumps_only_on_change(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """Repeating the same scan (or nothing new) must not bump ``version``."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    geometry = _geometry()
    mapper = Mapper(manifest, geometry, cfg)
    assert mapper.version == 0

    origin = np.array([0.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    scan = _scan(origin, dirs, np.array([2.0]), [0])

    mapper.integrate(scan)
    v1 = mapper.version
    assert v1 > 0

    mapper.integrate(scan)
    assert mapper.version == v1


def test_mark_free_box_bumps_version(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """Seeding the launch column is itself an occupancy change."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    mapper = Mapper(manifest, _geometry(), cfg)

    lo = mapper.state.origin
    hi = lo + np.array([1.0, 1.0, 1.0])
    mapper.mark_free_box(lo, hi)
    assert mapper.version == 1
    mapper.mark_free_box(lo, hi)
    assert mapper.version == 1


def test_pop_revealed_and_coverage_properties(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """Integrating a good hit reveals the triangle and moves the coverage metrics."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    mapper = Mapper(manifest, _geometry(), cfg)

    origin = np.array([0.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    scan = _scan(origin, dirs, np.array([2.0]), [0])
    mapper.integrate(scan)

    assert mapper.pop_revealed().tolist() == [0]
    assert mapper.pop_revealed().size == 0
    assert mapper.coverage_total == 1.0
    assert mapper.coverage_ground_band == 1.0
    assert mapper.state.tri_seen[0]


def _geometry_with_hidden_triangle() -> SceneGeometry:
    """Two triangles, object 0: one exterior, one never visible from outside."""
    return SceneGeometry(
        vertices=[np.zeros((0, 3))],
        faces=[np.zeros((0, 3), dtype=np.int32)],
        obj_tri_offset=np.array([0, 2], dtype=np.int64),
        tri_obj=np.array([0, 0], dtype=np.int32),
        tri_normal=np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        tri_area=np.array([1.0, 1.0]),
        tri_centroid=np.array([[2.0, 0.0, 1.0], [2.0, 1.0, 1.0]]),
        tri_exterior=np.array([True, False]),
    )


def test_hidden_triangle_revealed_but_never_counted(
    cfg: Config, lot: npt.NDArray[np.float64]
) -> None:
    """A good hit reveals a ``tri_exterior=False`` triangle but never counts it.

    ``tri_exterior`` is a one-point heuristic that misfires on faces straddling
    a wall or buried in a canopy; a hit proves the face visible, so it colours
    it, while the metrics keep excluding it.
    """
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    mapper = Mapper(manifest, _geometry_with_hidden_triangle(), cfg)

    origin = np.array([0.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scan = _scan(origin, dirs, np.array([2.0, 2.0]), [0, 1])
    mapper.integrate(scan)

    assert mapper.pop_revealed().tolist() == [0, 1]
    np.testing.assert_array_equal(mapper.state.tri_seen, [True, True])
    # The hidden triangle's area never enters either denominator, so a fully
    # seen exterior triangle alone reads 100%.
    assert mapper.coverage_total == pytest.approx(1.0)
    assert mapper.coverage_ground_band == pytest.approx(1.0)


def test_surface_seen_is_a_live_view_updated_by_integrate(
    cfg: Config, lot: npt.NDArray[np.float64]
) -> None:
    """``mapper.state.surface_seen`` shares memory with the tracker: no re-fetch needed."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    mapper = Mapper(manifest, _geometry(), cfg)

    surface_seen = mapper.state.surface_seen
    assert surface_seen is not None
    assert surface_seen.shape == mapper.state.occ.shape
    assert not surface_seen.any()

    origin = np.array([0.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    scan = _scan(origin, dirs, np.array([2.0]), [0])
    mapper.integrate(scan)

    assert surface_seen.any()


def test_integrate_is_fast(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """Performance target: one 10,800-ray scan integrates in well under 20 ms."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest(lot, [wall])
    mapper = Mapper(manifest, _geometry(), cfg)

    n_rays = cfg.sensor.n_rays
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(n_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dist = rng.uniform(0.5, cfg.sensor.max_range_m, size=n_rays)
    tri_ids = np.full(n_rays, -1, dtype=np.int32)
    scan = _scan(np.zeros(3), dirs, dist, tri_ids.tolist())

    # Best-of-N: a shared test box under load has enough scheduling noise to
    # spike any one run well past its true cost, so take the fastest of a
    # handful rather than a single sample.
    best_ms = min(_timed_integrate(mapper, scan) for _ in range(5))
    assert best_ms < 20.0, f"integrate took {best_ms:.2f} ms"


def _timed_integrate(mapper: Mapper, scan: Scan) -> float:
    """Milliseconds for one :meth:`Mapper.integrate` call."""
    start = time.perf_counter()
    mapper.integrate(scan)
    return (time.perf_counter() - start) * 1000.0


def _meter_on_wall_scan(t: float, *, coloured: bool) -> Scan:
    """One scan of a yellow meter-sized box on a beige wall at ``y = 5``, from 3 m away.

    The rays are aimed straight at sampled surface points, so the test says
    exactly what the detector sees; ``coloured=False`` is the same scan
    without its colour channel.
    """
    rng = np.random.default_rng(int(t * 10))
    wall = np.column_stack(
        [rng.uniform(-2.0, 2.0, 2000), np.full(2000, 5.0), rng.uniform(0.3, 3.0, 2000)]
    )
    lo, hi = np.array([-0.15, 4.67, 1.25]), np.array([0.15, 5.0, 1.96])
    meter = lo + rng.uniform(size=(2000, 3)) * (hi - lo)
    face = rng.integers(0, 3, 2000)
    meter[np.arange(2000), face] = np.where(rng.random(2000) < 0.5, lo[face], hi[face])
    points = np.concatenate([wall, meter])
    colour = np.concatenate(
        [np.tile([225, 205, 170], (2000, 1)), np.tile([230, 200, 30], (2000, 1))]
    )
    origin = np.array([0.0, 2.0, 1.5])
    ray = points - origin
    dist = np.linalg.norm(ray, axis=1)
    n = len(points)
    return Scan(
        drone_id=0,
        t=t,
        origin=origin,
        dirs=ray / dist[:, None],
        dist=dist,
        obj_ids=np.zeros(n, dtype=np.int32),
        tri_ids=np.full(n, -1, dtype=np.int32),
        rgb=colour.astype(np.uint8) if coloured else None,
    )


def test_discovered_comes_from_coloured_scans(cfg: Config, lot: npt.NDArray[np.float64]) -> None:
    """The mapper publishes what its detector finds, replacing ``discovered``, never mutating it.

    Colourless scans leave nothing to classify. Objects are re-extracted on
    the first scan due each ``1 / perception.extract_hz`` seconds of scan time.
    """
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    mapper = Mapper(_manifest(lot, [wall]), _geometry(), cfg)

    mapper.integrate(_meter_on_wall_scan(0.0, coloured=False))
    assert mapper.state.discovered == {}
    empty = mapper.state.discovered

    period = 1.0 / cfg.perception.extract_hz
    for step in range(1, 6):
        mapper.integrate(_meter_on_wall_scan(step * period / 5.0, coloured=True))
    found = mapper.state.discovered
    assert found is not empty
    assert empty == {}
    assert [o.cls for o in found.values()] == [Cls.METER]
    (meter,) = found.values()
    np.testing.assert_allclose(meter.box.center, [0.0, 4.835, 1.605], atol=0.03)
