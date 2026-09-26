"""Tests for :mod:`canopy.sim.scene`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canopy.contracts import Cls, SceneManifest, SceneObject
from canopy.errors import WorldgenError
from canopy.sim.scene import load_geometry

#: Two triangles forming a unit square in the XY plane, z=0 -- a stand-in floor.
_FLOOR_VERTS = "\n".join(
    [
        "v 0.000000 0.000000 0.000000",
        "v 1.000000 0.000000 0.000000",
        "v 1.000000 1.000000 0.000000",
        "v 0.000000 1.000000 0.000000",
    ]
)
_FLOOR_FACES = "f 1 2 3\nf 1 3 4\n"

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


def _write_obj(path: Path, o_name: str, verts: str, faces: str) -> None:
    """Write an OBJ in the exact format ``generate.py`` produces."""
    path.write_text(f"# canopy test\no {o_name}\n{verts}\n{faces}", encoding="utf-8")


def _make_object(obj_id: int, mesh_path: Path, cls: Cls = Cls.WALL) -> SceneObject:
    return SceneObject(
        obj_id=obj_id,
        cls=cls,
        mesh_path=str(mesh_path),
        color=(255, 255, 255),
    )


def _manifest(objects: list[SceneObject]) -> SceneManifest:
    return SceneManifest(
        seed=0,
        lot_bounds=np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]]),
        footprint=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        objects=objects,
        home=np.array([0.0, -8.0, 0.0]),
        gt_meter_id=0,
    )


def test_load_geometry_offsets_and_tri_obj(tmp_path: Path) -> None:
    """Two objects concatenate in manifest order with correct global ids."""
    floor_path = tmp_path / "0_ground.obj"
    wall_path = tmp_path / "1_wall.obj"
    _write_obj(floor_path, "0_ground", _FLOOR_VERTS, _FLOOR_FACES)
    _write_obj(wall_path, "1_wall", _WALL_VERTS, _WALL_FACES)

    manifest = _manifest(
        [_make_object(0, floor_path, Cls.GROUND), _make_object(1, wall_path, Cls.WALL)]
    )
    geometry = load_geometry(manifest)

    assert geometry.n_triangles == 4
    np.testing.assert_array_equal(geometry.obj_tri_offset, [0, 2, 4])
    np.testing.assert_array_equal(geometry.tri_obj, [0, 0, 1, 1])


def test_load_geometry_known_triangle(tmp_path: Path) -> None:
    """Area, centroid and normal of the floor's first triangle are as expected."""
    floor_path = tmp_path / "0_ground.obj"
    _write_obj(floor_path, "0_ground", _FLOOR_VERTS, _FLOOR_FACES)
    manifest = _manifest([_make_object(0, floor_path, Cls.GROUND)])

    geometry = load_geometry(manifest)

    # Triangle (0,0,0), (1,0,0), (1,1,0): right triangle, legs of length 1.
    assert geometry.tri_area[0] == pytest.approx(0.5)
    np.testing.assert_allclose(geometry.tri_centroid[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
    # Winding (0,0,0)->(1,0,0)->(1,1,0) gives +Z by the right-hand rule.
    np.testing.assert_allclose(geometry.tri_normal[0], [0.0, 0.0, 1.0])


def test_load_geometry_missing_mesh_raises(tmp_path: Path) -> None:
    """A mesh path that does not exist raises WorldgenError naming the path."""
    missing = tmp_path / "does_not_exist.obj"
    manifest = _manifest([_make_object(0, missing)])

    with pytest.raises(WorldgenError, match="does_not_exist"):
        load_geometry(manifest)


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


def _box_mesh(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[str, str]:
    """Build a closed box from ``lo`` to ``hi``, as OBJ vertex/face text."""
    corners = [
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]),
        (lo[0], hi[1], hi[2]),
    ]
    verts_text = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners)
    faces_text = "\n".join(f"f {a} {b} {c}" for a, b, c in _BOX_FACES) + "\n"
    return verts_text, faces_text


def _grid_ground(nx: int, ny: int) -> tuple[str, str]:
    """Build a ground mesh over ``[0, nx] x [0, ny]`` at z=0, two triangles per unit cell."""

    def vid(i: int, j: int) -> int:
        return i * (ny + 1) + j + 1  # 1-based

    verts = [(float(i), float(j), 0.0) for i in range(nx + 1) for j in range(ny + 1)]
    faces = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    verts_text = "\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts)
    faces_text = "\n".join(f"f {a} {b} {c}" for a, b, c in faces) + "\n"
    return verts_text, faces_text


def test_sky_exposed_box_on_ground(tmp_path: Path) -> None:
    """A closed box's shell and open ground are exterior.

    A nested box, and the ground triangles hidden beneath the outer box, are not.
    """
    ground_path = tmp_path / "0_ground.obj"
    outer_path = tmp_path / "1_outer.obj"
    inner_path = tmp_path / "2_inner.obj"

    g_verts, g_faces = _grid_ground(4, 4)
    _write_obj(ground_path, "0_ground", g_verts, g_faces)
    o_verts, o_faces = _box_mesh((1.0, 1.0, 0.0), (2.0, 2.0, 1.0))
    _write_obj(outer_path, "1_outer", o_verts, o_faces)
    # Well inside the outer shell: 0.35 m lateral margin, 0.3 m vertical margin
    # to the roof, enough that no upward sky probe can slip past it.
    i_verts, i_faces = _box_mesh((1.35, 1.35, 0.3), (1.65, 1.65, 0.7))
    _write_obj(inner_path, "2_inner", i_verts, i_faces)

    manifest = _manifest(
        [
            _make_object(0, ground_path, Cls.GROUND),
            _make_object(1, outer_path, Cls.WALL),
            _make_object(2, inner_path, Cls.WALL),
        ]
    )
    geometry = load_geometry(manifest)
    assert geometry.tri_exterior is not None

    offsets = geometry.obj_tri_offset
    ground = slice(offsets[0], offsets[1])
    outer = slice(offsets[1], offsets[2])
    inner = slice(offsets[2], offsets[3])

    ground_centroids = geometry.tri_centroid[ground]
    under_box = (
        (ground_centroids[:, 0] >= 1.0)
        & (ground_centroids[:, 0] <= 2.0)
        & (ground_centroids[:, 1] >= 1.0)
        & (ground_centroids[:, 1] <= 2.0)
    )
    ground_exterior = geometry.tri_exterior[ground]
    assert np.all(ground_exterior[~under_box]), "open ground should be exterior"
    assert np.all(~ground_exterior[under_box]), "ground hidden under the box should not be"

    # The bottom two triangles sit flush on the ground (both at z=0): every
    # upward probe from either side hits the coincident ground or the box's
    # own roof, so -- correctly -- the underside of a box resting on the
    # ground is never exterior, same as a house floor slab.
    outer_exterior = geometry.tri_exterior[outer]
    assert np.all(outer_exterior[2:]), "the box's above-ground shell should be exterior"
    assert np.all(~outer_exterior[:2]), "the box's floor, flush with the ground, should not be"
    assert np.all(~geometry.tri_exterior[inner]), "a box fully enclosed by another should not be"


def test_load_geometry_malformed_mesh_raises(tmp_path: Path) -> None:
    """A face line naming more than 3 vertices (not pre-triangulated) raises."""
    bad_path = tmp_path / "bad.obj"
    bad_path.write_text(
        "# canopy test\no bad\n"
        "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nv 0.0 1.0 0.0\n"
        "f 1 2 3 4\n",
        encoding="utf-8",
    )
    manifest = _manifest([_make_object(0, bad_path)])

    with pytest.raises(WorldgenError):
        load_geometry(manifest)
