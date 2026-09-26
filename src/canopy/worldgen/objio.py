"""A purpose-built reader for the model library's Wavefront OBJ files.

The library at ``assets/obj_export/`` holds files authored Y-up in metres,
each containing several named objects (``o house``, ``o yard``, ``o tree_1``,
...) whose faces are further split into runs by ``usemtl``. Downstream code
needs *both* groupings at once: the object name is how a placement rule pulls
one prop out of a multi-object file, and the material name is what later maps
to a semantic class (``brick`` -> ``WALL``, ``roof_shingle`` -> ``ROOF``, ...).

``trimesh.load`` cannot give us this. It groups faces by material and merges
across ``o`` objects, and ``split_object=True`` shatters a file into thousands
of per-run fragments -- neither preserves the object/material product. Hence
this module: it is the file-format boundary of the worldgen stage, reading
just enough of the OBJ grammar to keep both groupings and nothing else.

Up-axis conversion. Models are authored Y-up; the world frame is Z-up (see
``contracts.py``). :func:`read_obj` rotates +90 degrees about X to convert,
``(x, y, z) -> (x, -z, y)``, which keeps the frame right-handed: model up +Y
becomes world +Z. A consequence worth stating plainly, because a downstream
wall-mounting rule depends on it: a model's native +Z "outward/front" axis
becomes **-Y** in the world frame, not +Y.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from canopy.errors import AssetError

__all__ = ["MaterialGroup", "MtlMaterial", "ObjObject", "read_mtl", "read_obj"]

#: Y-up (model space) or Z-up (already world space); see the module docstring.
UpAxis = Literal["y", "z"]

_XYZ = 3
_MIN_FACE_VERTICES = 3


@dataclass(frozen=True, slots=True)
class MtlMaterial:
    """The part of one ``newmtl`` block that survives into a flat-shaded render.

    Attributes
    ----------
    name
        The ``newmtl`` name, as referenced by ``usemtl`` in the OBJ.
    diffuse
        ``Kd`` as RGB in ``[0, 1]``, as authored (conventionally sRGB).
    opacity
        ``d``, where ``1.0`` is opaque.
    """

    name: str
    diffuse: tuple[float, float, float]
    opacity: float


@dataclass(frozen=True, slots=True, eq=False)
class MaterialGroup:
    """One contiguous run of faces sharing a ``usemtl`` name.

    Attributes
    ----------
    material
        The ``usemtl`` name, or ``""`` for faces declared before any
        ``usemtl`` line in their object.
    faces
        Triangle indices, shape ``(F, 3)``, into the owning
        :attr:`ObjObject.vertices` -- local to that object, not global to the
        file.
    """

    material: str
    faces: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True, eq=False)
class ObjObject:
    """One ``o``-delimited object, re-indexed to its own local vertex array.

    Attributes
    ----------
    name
        The ``o`` name, or ``""`` for faces declared before any ``o`` line.
    vertices
        Shape ``(V, 3)`` float64, Z-up metres, indexed local to this object.
    groups
        Material runs in first-appearance order. Never empty: an object with
        no faces is dropped by :func:`read_obj` rather than appearing here.
    """

    name: str
    vertices: npt.NDArray[np.float64]
    groups: tuple[MaterialGroup, ...]

    @property
    def faces(self) -> npt.NDArray[np.int32]:
        """All groups' triangles concatenated, in group order, shape ``(F, 3)``."""
        return np.concatenate([g.faces for g in self.groups], axis=0)

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """Axis-aligned bounding box, shape ``(2, 3)``: ``[[min xyz], [max xyz]]``."""
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def extents(self) -> npt.NDArray[np.float64]:
        """Bounding-box size along each axis, shape ``(3,)``."""
        bounds = self.bounds
        return np.asarray(bounds[1] - bounds[0], dtype=np.float64)


def _parse_vertex(rest: str, path: Path, lineno: int, line: str) -> list[float]:
    """Parse the ``x y z`` (plus any ignored trailing values) of a ``v`` line."""
    try:
        xyz = [float(tok) for tok in rest.split()[:_XYZ]]
    except ValueError as exc:
        msg = f"{path}:{lineno}: malformed vertex {line!r}"
        raise AssetError(msg) from exc
    if len(xyz) < _XYZ:
        msg = f"{path}:{lineno}: malformed vertex {line!r}"
        raise AssetError(msg)
    return xyz


def _parse_face_index(token: str, vertex_count: int, path: Path, lineno: int, line: str) -> int:
    """Resolve one ``f`` vertex reference to a 0-based index, global to the file.

    References come as ``v``, ``v/vt``, ``v//vn`` or ``v/vt/vn``; only the
    part before the first ``/`` matters here. Indices are 1-based; negative
    indices count back from ``vertex_count``, the number of ``v`` lines seen
    so far, per the OBJ spec.
    """
    raw = token.split("/", 1)[0]
    try:
        i = int(raw)
    except ValueError as exc:
        msg = f"{path}:{lineno}: non-integer face index {raw!r} in {line!r}"
        raise AssetError(msg) from exc
    idx = i - 1 if i > 0 else vertex_count + i
    if not 0 <= idx < vertex_count:
        msg = (
            f"{path}:{lineno}: face index {i} out of range "
            f"({vertex_count} vertices seen so far) in {line!r}"
        )
        raise AssetError(msg)
    return idx


def read_obj(path: Path, *, up_axis: UpAxis = "y") -> dict[str, ObjObject]:
    """Read a multi-object OBJ file, grouping faces by both ``o`` and ``usemtl``.

    Parameters
    ----------
    path
        File to read.
    up_axis
        ``"y"`` (the default, matching how these models are authored) rotates
        vertices into the project's Z-up world frame; ``"z"`` leaves them
        untouched. See the module docstring for what this means for a
        model's forward axis.

    Returns
    -------
    dict[str, ObjObject]
        One entry per non-empty ``o`` object, in first-appearance order --
        callers rely on that order (e.g. to report "objects found" in the
        same order a human reading the file would list them), so a plain
        ``dict`` is used deliberately rather than being sorted.

    Raises
    ------
    AssetError
        If the file cannot be read, contains a malformed ``v`` line or face
        index, contains a face with fewer than three vertices, or yields no
        faces at all.

    Notes
    -----
    Only ``v``, ``o``, ``usemtl`` and ``f`` lines are interpreted; ``vt``,
    ``vn``, ``mtllib``, ``s``, ``g`` and comments are legitimately present in
    these files and are ignored. Faces are triangulated with a fan
    (``(v0, v1, v2), (v0, v2, v3), ...``), which preserves winding -- the ray
    sensor reports primitive normals, so getting this wrong would flip them.
    A repeated ``o`` name, or a material used in two separate runs of one
    object, accumulates into the existing object/group rather than
    replacing it, so no geometry is lost.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read OBJ file {path}: {exc}"
        raise AssetError(msg) from exc

    global_verts: list[list[float]] = []
    # object name -> material name -> triangles (each a 3-int list of global
    # 0-based vertex indices). Both dict levels are insertion-ordered, which
    # is how first-appearance order survives into the result below.
    objects: dict[str, dict[str, list[list[int]]]] = {}
    current_obj = ""
    current_mat = ""

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, rest = line.partition(" ")
        rest = rest.strip()

        if tag == "v":
            global_verts.append(_parse_vertex(rest, path, lineno, line))
        elif tag == "o":
            current_obj = rest
        elif tag == "usemtl":
            current_mat = rest
        elif tag == "f":
            tokens = rest.split()
            if len(tokens) < _MIN_FACE_VERTICES:
                msg = f"{path}:{lineno}: face has fewer than 3 vertices: {line!r}"
                raise AssetError(msg)
            verts = [
                _parse_face_index(tok, len(global_verts), path, lineno, line) for tok in tokens
            ]
            triangles = objects.setdefault(current_obj, {}).setdefault(current_mat, [])
            v0 = verts[0]
            for k in range(1, len(verts) - 1):
                triangles.append([v0, verts[k], verts[k + 1]])
        # else: vt, vn, mtllib, s, g -- present in these files, irrelevant here.

    if not objects:
        msg = f"{path}: no faces found in any object"
        raise AssetError(msg)

    verts_arr = np.array(global_verts, dtype=np.float64)
    if up_axis == "y":
        verts_arr = np.stack([verts_arr[:, 0], -verts_arr[:, 2], verts_arr[:, 1]], axis=1)

    result: dict[str, ObjObject] = {}
    for name, mat_groups in objects.items():
        result[name] = _build_object(name, mat_groups, verts_arr)
    return result


def read_mtl(path: Path) -> dict[str, MtlMaterial]:
    """Read the diffuse colour and opacity of every material in an MTL file.

    Worldgen never needs this -- it maps material *names* to classes -- but a
    renderer drawing an authored model needs its colours, and the MTL beside
    the OBJ is where the art keeps them.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    dict[str, MtlMaterial]
        Keyed by ``newmtl`` name, in file order. A material with no ``Kd`` is
        mid-grey, one with no ``d`` is opaque -- the MTL spec's defaults.

    Raises
    ------
    AssetError
        If the file cannot be read, or a ``Kd`` or ``d`` line is malformed or
        appears before any ``newmtl``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read MTL file {path}: {exc}"
        raise AssetError(msg) from exc

    diffuse: dict[str, tuple[float, float, float]] = {}
    opacity: dict[str, float] = {}
    current: str | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        tag, _, rest = line.partition(" ")
        if tag == "newmtl":
            current = rest.strip()
            diffuse[current] = (0.5, 0.5, 0.5)
            opacity[current] = 1.0
        elif tag in ("Kd", "d"):
            if current is None:
                msg = f"{path}:{lineno}: {tag!r} before any newmtl: {line!r}"
                raise AssetError(msg)
            try:
                values = [float(tok) for tok in rest.split()]
            except ValueError as exc:
                msg = f"{path}:{lineno}: malformed {tag!r} line {line!r}"
                raise AssetError(msg) from exc
            want = _XYZ if tag == "Kd" else 1
            if len(values) != want:
                msg = f"{path}:{lineno}: {tag!r} takes {want} value(s), got {line!r}"
                raise AssetError(msg)
            if tag == "Kd":
                diffuse[current] = (values[0], values[1], values[2])
            else:
                opacity[current] = values[0]
        # else: Ka, Ks, Ns, illum, maps, comments -- not needed to draw it.

    return {
        name: MtlMaterial(name=name, diffuse=rgb, opacity=opacity[name])
        for name, rgb in diffuse.items()
    }


def _build_object(
    name: str,
    mat_groups: dict[str, list[list[int]]],
    verts_arr: npt.NDArray[np.float64],
) -> ObjObject:
    """Re-index one object's faces to a local vertex array.

    Gathers the unique global vertex indices this object's faces reference
    and remaps to local indices with a single ``np.unique(return_inverse=True)``
    call rather than looping over faces in Python.
    """
    group_names = list(mat_groups)
    face_lists = [mat_groups[m] for m in group_names]
    flat = np.array([tri for faces in face_lists for tri in faces], dtype=np.int64)
    unique_idx, inverse = np.unique(flat.ravel(), return_inverse=True)
    local_faces_all = inverse.reshape(-1, 3).astype(np.int32)
    local_verts = verts_arr[unique_idx]

    groups = []
    offset = 0
    for mat_name, faces in zip(group_names, face_lists, strict=True):
        count = len(faces)
        groups.append(
            MaterialGroup(material=mat_name, faces=local_faces_all[offset : offset + count])
        )
        offset += count

    return ObjObject(name=name, vertices=local_verts, groups=tuple(groups))
