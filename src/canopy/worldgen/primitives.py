"""Primitive-mesh builders for ``parts:`` models.

Split out of :mod:`canopy.worldgen.assets` because this is pure trimesh
geometry construction -- normalised unit-box shapes with no notion of a scene,
a material, or a YAML document -- while the rest of that module is either
runtime library plumbing or index parsing. Keeping it separate means a new
:data:`~canopy.worldgen.assets.PartKind` only ever touches this file's
dispatch in :func:`_primitive`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import trimesh

from canopy.errors import AssetError
from canopy.worldgen.objio import MaterialGroup, ObjObject

if TYPE_CHECKING:
    from canopy.worldgen.assets import AssetSpec, PartSpec

__all__ = ["build_parts"]

#: Icosphere subdivision level for ``ellipsoid`` parts, per spec.md Module 1.
_ELLIPSOID_SUBDIVISIONS = 2
#: Radial segments for ``cylinder`` and ``cone`` parts.
_RADIAL_SECTIONS = 16


def _plane() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Build a flat unit quad in the XY plane at ``z = 0``, wound +Z up."""
    verts = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    return verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _gable_prism() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Build a unit triangular prism: ridge along +X, apex on top, centred on the origin.

    Written out by hand rather than via ``extrude_polygon`` because the cross
    section is three points and the winding has to come out outward-facing --
    the ray sensor reports primitive normals and the photo shader uses them.

    Centred like every other primitive, eaves at ``z = -0.5``, so a part's ``at``
    means its centre whatever its kind. With the eave line at ``z = 0`` instead,
    the stand-on-the-base default lifted a lone roof by half its height, and
    every procedural house's roof floated clear of its walls.
    """
    verts = np.array(
        [
            [-0.5, -0.5, -0.5],  # 0  -X back eave
            [-0.5, 0.5, -0.5],  # 1  -X front eave
            [-0.5, 0.0, 0.5],  # 2  -X ridge
            [0.5, -0.5, -0.5],  # 3  +X back eave
            [0.5, 0.5, -0.5],  # 4  +X front eave
            [0.5, 0.0, 0.5],  # 5  +X ridge
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],  # -X gable end
            [3, 4, 5],  # +X gable end
            [0, 1, 4],
            [0, 4, 3],  # underside
            [1, 2, 5],
            [1, 5, 4],  # +Y pitch
            [0, 3, 5],
            [0, 5, 2],  # -Y pitch
        ],
        dtype=np.int32,
    )
    return verts, faces


def _primitive(part: PartSpec) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Build one part with unit extents, centred on the origin."""
    if part.kind == "plane":
        return _plane()
    if part.kind == "gable_prism":
        return _gable_prism()
    if part.kind == "box":
        prim = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    elif part.kind == "ellipsoid":
        prim = trimesh.creation.icosphere(subdivisions=_ELLIPSOID_SUBDIVISIONS, radius=0.5)
    elif part.kind == "cylinder":
        prim = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=_RADIAL_SECTIONS)
    else:
        # trimesh's cone sits on z=0; recentre so `at` means the same thing for
        # every kind. This is the last arm because the dispatch is exhaustive
        # over PartKind: adding a kind without a builder is a mypy error here,
        # which is a better guard than a runtime branch that can never run.
        prim = trimesh.creation.cone(radius=0.5, height=1.0, sections=_RADIAL_SECTIONS)
        prim.vertices = np.asarray(prim.vertices, dtype=np.float64) - np.array([0.0, 0.0, 0.5])
    return (
        np.asarray(prim.vertices, dtype=np.float64),
        np.asarray(prim.faces, dtype=np.int32),
    )


def build_parts(spec: AssetSpec) -> ObjObject:
    """Synthesise a ``parts`` model in the normalised unit box."""
    if not spec.parts:
        msg = f"asset {spec.asset_id!r} declares neither `mesh` nor `parts`"
        raise AssetError(msg)

    chunks_v: list[npt.NDArray[np.float64]] = []
    by_material: dict[str, list[npt.NDArray[np.int32]]] = {}
    offset = 0
    for part in spec.parts:
        verts, faces = _primitive(part)
        if part.axis != "z":
            # Cylinders and cones are built along +Z; swing them onto their
            # axis so `size` keeps meaning extents in the model's x, y and z.
            verts = verts[:, {"x": [2, 1, 0], "y": [0, 2, 1]}[part.axis]]
        verts = verts * np.asarray(part.size, dtype=np.float64)
        verts = verts + np.asarray(part.centre, dtype=np.float64)
        chunks_v.append(verts)
        by_material.setdefault(part.material, []).append(faces + offset)
        offset += len(verts)

    return ObjObject(
        name=spec.asset_id,
        vertices=np.concatenate(chunks_v),
        # One group per material, in first-use order, so `build` splits a
        # parts model by class exactly as it splits authored art.
        groups=tuple(
            MaterialGroup(material=m, faces=np.concatenate(f)) for m, f in by_material.items()
        ),
    )
