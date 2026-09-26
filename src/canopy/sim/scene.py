"""Load a scene manifest's meshes into one :class:`~canopy.contracts.SceneGeometry`.

The OBJ files here are the ones :func:`canopy.worldgen.generate._write_obj`
wrote: already world-space and Z-up, one object, triangles only, 1-based face
indices. That is *not* what :func:`canopy.worldgen.objio.read_obj` assumes --
it exists to read Y-up authored art and applies an up-axis rotation this
module must not repeat -- so scenes get their own small, vectorized parser
instead of reusing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import SceneGeometry
from canopy.errors import WorldgenError
from canopy.sim.sensors import build_raycasting_scene, sky_exposed

if TYPE_CHECKING:
    from canopy.contracts import SceneManifest

__all__ = ["load_geometry"]

#: Columns after the ``v``/``f`` tag: x, y, z or a, b, c.
_XYZ = 3
#: Every parsed vertex/face row is a flat 2D array: rows x 3 columns.
_ROW_NDIM = 2


def _parse_mesh(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Read one world-space OBJ into vertex and (0-based) face arrays.

    Vectorized: the file is split into ``v``/``f`` lines once, then handed to
    ``np.array`` in bulk rather than parsed line by line in Python, which is
    what keeps loading a quarter-million triangles well under a second.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read mesh {path}: {exc}"
        raise WorldgenError(msg) from exc

    v_rows: list[str] = []
    f_rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            v_rows.append(line[2:])
        elif line.startswith("f "):
            f_rows.append(line[2:])

    if not v_rows or not f_rows:
        msg = f"{path}: no vertices or no faces found"
        raise WorldgenError(msg)

    try:
        vertices = np.array([row.split() for row in v_rows], dtype=np.float64)
    except ValueError as exc:
        msg = f"{path}: malformed vertex line: {exc}"
        raise WorldgenError(msg) from exc
    if vertices.ndim != _ROW_NDIM or vertices.shape[1] != _XYZ:
        msg = f"{path}: vertex lines must have exactly 3 coordinates"
        raise WorldgenError(msg)

    try:
        faces_1based = np.array([row.split() for row in f_rows], dtype=np.int64)
    except ValueError as exc:
        msg = f"{path}: malformed face line (not a triangle mesh?): {exc}"
        raise WorldgenError(msg) from exc
    if faces_1based.ndim != _ROW_NDIM or faces_1based.shape[1] != _XYZ:
        msg = f"{path}: face lines must reference exactly 3 vertices (triangles only)"
        raise WorldgenError(msg)

    faces = (faces_1based - 1).astype(np.int32)
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
        msg = f"{path}: face index out of range"
        raise WorldgenError(msg)

    return vertices, faces


def _face_geometry(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized per-triangle normal, area and centroid.

    A degenerate triangle (zero cross product) gets a zero normal rather than
    a NaN: dividing by its zero length would otherwise poison every downstream
    reduction that touches ``tri_normal``.
    """
    tri = vertices[faces]  # (F, 3, 3)
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge1, edge2)
    lengths = np.linalg.norm(cross, axis=1)
    area = lengths / 2.0
    safe_lengths = np.where(lengths > 0.0, lengths, 1.0)
    normal = cross / safe_lengths[:, None]
    normal[lengths == 0.0] = 0.0
    centroid = tri.mean(axis=1)
    return normal, area, centroid


def load_geometry(manifest: SceneManifest) -> SceneGeometry:
    """Load every object's mesh and build the global triangle indices.

    Parameters
    ----------
    manifest
        Scene to load. Objects are concatenated in manifest order, so object
        ``k`` owns global triangle ids ``[obj_tri_offset[k], obj_tri_offset[k + 1])``
        -- the same convention :class:`~canopy.sim.sensors.RaySensor` uses to
        map an Open3D geometry/primitive id pair back to a global triangle id.

    Returns
    -------
    SceneGeometry
        Per-object mesh data plus the global per-triangle arrays.

    Raises
    ------
    WorldgenError
        If a mesh file is missing, unreadable or malformed.
    """
    all_vertices: list[npt.NDArray[np.float64]] = []
    all_faces: list[npt.NDArray[np.int32]] = []
    tri_counts: list[int] = []
    normals: list[npt.NDArray[np.float64]] = []
    areas: list[npt.NDArray[np.float64]] = []
    centroids: list[npt.NDArray[np.float64]] = []
    tri_obj_parts: list[npt.NDArray[np.int32]] = []

    for obj in manifest.objects:
        vertices, faces = _parse_mesh(Path(obj.mesh_path))
        normal, area, centroid = _face_geometry(vertices, faces)

        all_vertices.append(vertices)
        all_faces.append(faces)
        tri_counts.append(len(faces))
        normals.append(normal)
        areas.append(area)
        centroids.append(centroid)
        tri_obj_parts.append(np.full(len(faces), obj.obj_id, dtype=np.int32))

    obj_tri_offset = np.zeros(len(manifest.objects) + 1, dtype=np.int64)
    np.cumsum(tri_counts, out=obj_tri_offset[1:])

    tri_normal = np.concatenate(normals) if normals else np.zeros((0, 3), dtype=np.float64)
    tri_centroid = np.concatenate(centroids) if centroids else np.zeros((0, 3), dtype=np.float64)
    # One-off ground-truth visibility pass, so coverage never waits on faces
    # sealed inside the house (see SceneGeometry.tri_exterior).
    scene = build_raycasting_scene(all_vertices, all_faces)
    tri_exterior = sky_exposed(scene, tri_centroid, tri_normal)

    return SceneGeometry(
        vertices=all_vertices,
        faces=all_faces,
        obj_tri_offset=obj_tri_offset,
        tri_obj=np.concatenate(tri_obj_parts) if tri_obj_parts else np.zeros(0, dtype=np.int32),
        tri_normal=tri_normal,
        tri_area=np.concatenate(areas) if areas else np.zeros(0, dtype=np.float64),
        tri_centroid=tri_centroid,
        tri_exterior=tri_exterior,
        obj_color=np.array([obj.color for obj in manifest.objects], dtype=np.uint8).reshape(-1, 3),
    )
