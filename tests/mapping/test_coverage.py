"""Tests for :mod:`canopy.mapping.coverage`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from canopy.config import Config
from canopy.contracts import Cls, Scan, SceneGeometry, SceneManifest, SceneObject
from canopy.mapping.coverage import CoverageTracker

_VOXEL = 0.25
_ORIGIN = np.array([0.0, 0.0, 0.0])
_SHAPE = (40, 40, 40)  # 10 m cube, comfortably larger than any test geometry


def _manifest(objects: list[SceneObject]) -> SceneManifest:
    """Build a manifest with just enough to satisfy the coverage tracker."""
    return SceneManifest(
        seed=0,
        lot_bounds=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]),
        footprint=[],
        objects=objects,
        home=np.zeros(3),
        gt_meter_id=-1,
    )


def _geometry(
    tri_obj: list[int],
    tri_normal: npt.NDArray[np.float64],
    tri_area: npt.NDArray[np.float64],
    tri_centroid: npt.NDArray[np.float64],
) -> SceneGeometry:
    """Build a geometry with one dummy object per distinct ``tri_obj`` value."""
    n_objects = max(tri_obj) + 1
    return SceneGeometry(
        vertices=[np.zeros((0, 3)) for _ in range(n_objects)],
        faces=[np.zeros((0, 3), dtype=np.int32) for _ in range(n_objects)],
        obj_tri_offset=np.array([0, len(tri_obj)], dtype=np.int64),
        tri_obj=np.array(tri_obj, dtype=np.int32),
        tri_normal=tri_normal,
        tri_area=tri_area,
        tri_centroid=tri_centroid,
    )


def _scan(
    origin: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    tri_ids: list[int],
) -> Scan:
    """Build a minimal scan; occupancy fields are unused by coverage."""
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


def test_grazing_and_far_hits_do_not_count(cfg: Config) -> None:
    """Beyond coverage range, or at grazing incidence, a hit does not reveal."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest([wall])
    geometry = _geometry(
        tri_obj=[0, 0],
        tri_normal=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        tri_area=np.array([1.0, 1.0]),
        tri_centroid=np.array([[5.0, 5.0, 1.0], [5.0, 5.0, 1.0]]),
    )
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)

    # Triangle 0: straight-on but beyond coverage_max_range_m.
    far_dist = cfg.map.coverage_max_range_m + 1.0
    tracker.integrate(
        _scan(
            np.array([5.0 - far_dist, 5.0, 1.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([far_dist]),
            [0],
        ),
        geometry,
    )
    assert not tracker.tri_seen[0]

    # Triangle 1: in range but grazing (normal perpendicular to the ray).
    tracker.integrate(
        _scan(
            np.array([5.0, 5.0 - 1.0, 1.0]),
            np.array([[0.0, 1.0, 0.0]]),
            np.array([1.0]),
            [1],
        ),
        geometry,
    )
    assert not tracker.tri_seen[1]


def test_good_hit_reveals_voxelmates(cfg: Config) -> None:
    """A photo-quality hit reveals the hit triangle and its centroid-voxel neighbours."""
    bush = SceneObject(obj_id=0, cls=Cls.BUSH, mesh_path="", color=(0, 0, 0))
    manifest = _manifest([bush])
    # Triangle 0 is the one actually hit; triangles 1-2 are tiny leaves sharing
    # its voxel; triangle 3 sits in a different voxel and must stay unseen.
    geometry = _geometry(
        tri_obj=[0, 0, 0, 0],
        tri_normal=np.array([[-1.0, 0.0, 0.0]] * 4),
        tri_area=np.array([1.0, 0.01, 0.01, 1.0]),
        tri_centroid=np.array(
            [
                [5.0, 5.0, 1.0],
                [5.05, 5.05, 1.05],
                [5.1, 5.1, 1.1],
                [8.0, 8.0, 1.0],
            ]
        ),
    )
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)

    tracker.integrate(
        _scan(
            np.array([4.0, 5.0, 1.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
            [0],
        ),
        geometry,
    )
    assert tracker.tri_seen[0]
    assert tracker.tri_seen[1]
    assert tracker.tri_seen[2]
    assert not tracker.tri_seen[3]


def test_background_triangles_never_count(cfg: Config) -> None:
    """A background object's triangles never count, direct hit or not."""
    neighbour = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0), background=True)
    manifest = _manifest([neighbour])
    geometry = _geometry(
        tri_obj=[0],
        tri_normal=np.array([[-1.0, 0.0, 0.0]]),
        tri_area=np.array([1.0]),
        tri_centroid=np.array([[5.0, 5.0, 1.0]]),
    )
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)

    tracker.integrate(
        _scan(
            np.array([4.0, 5.0, 1.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
            [0],
        ),
        geometry,
    )
    assert not tracker.tri_seen[0]
    assert tracker.coverage_total == 0.0
    assert tracker.coverage_ground_band == 0.0


def test_non_exterior_hit_reveals_but_does_not_count(cfg: Config) -> None:
    """A face ``tri_exterior`` wrongly calls sealed still reveals when photographed.

    ``sky_exposed`` probes from one point per face, so a ground triangle whose
    centroid sits under a wall, or a leaf inside a canopy, is flagged sealed
    even though drones see it. A photo-quality hit is proof of visibility, so
    it must colour the face (and its voxelmates), while the metrics keep the
    face out of both numerator and denominator.
    """
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest([wall])
    # 0: exterior, unseen. 1: flagged non-exterior, hit directly. 2: flagged
    # non-exterior, shares triangle 1's voxel.
    geometry = _geometry(
        tri_obj=[0, 0, 0],
        tri_normal=np.array([[-1.0, 0.0, 0.0]] * 3),
        tri_area=np.array([1.0, 1.0, 1.0]),
        tri_centroid=np.array([[8.0, 8.0, 1.0], [5.0, 5.0, 1.0], [5.05, 5.05, 1.05]]),
    )
    geometry.tri_exterior = np.array([True, False, False])
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)

    tracker.integrate(
        _scan(
            np.array([4.0, 5.0, 1.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
            [1],
        ),
        geometry,
    )
    assert tracker.tri_seen.tolist() == [False, True, True]
    assert sorted(tracker.pop_revealed().tolist()) == [1, 2]
    assert tracker.coverage_total == 0.0
    assert tracker.coverage_ground_band == 0.0


def test_ground_band_and_total_metrics(cfg: Config) -> None:
    """Area-weighted metrics over a hand-built mix of classes."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    ground = SceneObject(obj_id=1, cls=Cls.GROUND, mesh_path="", color=(0, 0, 0))
    tree = SceneObject(obj_id=2, cls=Cls.TREE, mesh_path="", color=(0, 0, 0))
    manifest = _manifest([wall, ground, tree])
    # 0: wall, in band, seen. 1: wall, in band, unseen. 2: ground, excluded from
    # both metrics. 3: tree, above the band, counts toward total only.
    geometry = _geometry(
        tri_obj=[0, 0, 1, 2],
        tri_normal=np.array([[-1.0, 0.0, 0.0]] * 4),
        tri_area=np.array([1.0, 3.0, 5.0, 2.0]),
        tri_centroid=np.array(
            [
                [5.0, 5.0, 1.0],
                [6.0, 5.0, 1.0],
                [5.0, 5.0, 0.5],
                [5.0, 5.0, 5.0],
            ]
        ),
    )
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)
    tracker.integrate(
        _scan(
            np.array([4.0, 5.0, 1.0]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
            [0],
        ),
        geometry,
    )
    # total area of non-ground, non-background triangles: 1 (wall,seen) + 3
    # (wall, unseen) + 2 (tree) = 6; seen area = 1.
    assert tracker.coverage_total == 1.0 / 6.0
    # ground-band area (wall triangles only, tree is above the band): 1 + 3 = 4;
    # seen area = 1.
    assert tracker.coverage_ground_band == 1.0 / 4.0


def test_pop_revealed_returns_each_id_once(cfg: Config) -> None:
    """Every newly seen id is reported exactly once, then cleared."""
    wall = SceneObject(obj_id=0, cls=Cls.WALL, mesh_path="", color=(0, 0, 0))
    manifest = _manifest([wall])
    geometry = _geometry(
        tri_obj=[0, 0],
        tri_normal=np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        tri_area=np.array([1.0, 1.0]),
        tri_centroid=np.array([[5.0, 5.0, 1.0], [5.0, 5.0, 1.0]]),
    )
    tracker = CoverageTracker(manifest, geometry, cfg.map, _ORIGIN, _VOXEL, _SHAPE)
    scan = _scan(
        np.array([4.0, 5.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        [0, 0],
    )
    tracker.integrate(scan, geometry)
    first = tracker.pop_revealed()
    assert sorted(first.tolist()) == [0, 1]
    assert tracker.pop_revealed().size == 0

    tracker.integrate(scan, geometry)
    assert tracker.pop_revealed().size == 0
