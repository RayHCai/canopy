"""The model library: a database of 3D models plus the roles that place them.

Why a library at all. The spec's Module 1 describes each object as a hand-coded
trimesh primitive. That does not survive contact with the goal of *adding* new
residential props: every new bush would mean a new function, a new placement
branch and a new colour lookup. Instead one YAML index
(``assets/models/index.yaml``) declares every model, every material-to-class
mapping and every generation role, and this module turns it into typed specs.
Adding a prop is a data change; only a genuinely new *kind* of placement needs
code (see :mod:`canopy.worldgen.placement`).

Two model sources are supported deliberately. ``mesh:`` names an authored OBJ,
optionally narrowed to one ``o`` group with ``object:``; this is the real
artwork, used at the size it was authored, with ``scale`` giving size variation.
``parts:`` describes a model as primitives in a normalised unit box -- x and y
over ``[-0.5, 0.5]``, z over ``[0, 1]`` -- scaled to the target extents in
``size_x/y/z``; this is how the generated ground plane exists without a file,
and how a placeholder prop can be flown against before any art for it exists.
Both paths end at the same thing, a world-space mesh scaled to chosen extents,
so a placement rule cannot tell them apart.

One authored mesh becomes *several* scene objects. The library's ``materials``
map turns each material name into a :class:`~canopy.contracts.Cls`, so the house
shell's ``brick``, ``roof_shingle``, ``window_glass`` and ``garage_door`` faces
come out as separately classified objects. That is what makes semantics free at
sensing time, per the spec's ray-casting decision, and it is why the art's
material naming is load-bearing.

Canopy is an application, so the library lives at the repository root next to
``config/`` and is located by walking up from this module. Set
``CANOPY_ASSETS_DIR`` to override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import trimesh
import yaml

from canopy.contracts import CLASS_COLORS, Cls, Rgb, Vec3
from canopy.errors import AssetError
from canopy.worldgen.objio import MaterialGroup, ObjObject, read_obj

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "AssetLibrary",
    "AssetSpec",
    "BuiltMesh",
    "CountSpec",
    "PartSpec",
    "RoleSpec",
    "default_assets_dir",
    "default_index_path",
    "load_library",
]

#: Supported primitive part kinds. Extending this tuple is the only reason a new
#: *shape* needs code; a new size or arrangement of existing shapes does not.
PartKind = Literal["plane", "box", "ellipsoid", "cylinder", "cone", "gable_prism"]
_PART_KINDS: tuple[str, ...] = ("plane", "box", "ellipsoid", "cylinder", "cone", "gable_prism")

#: Which way a model faces before it is rotated, in its own authored frame.
FaceAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
_FACE_AXES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")

#: Icosphere subdivision level for ``ellipsoid`` parts, per spec.md Module 1.
_ELLIPSOID_SUBDIVISIONS = 2
#: Radial segments for ``cylinder`` and ``cone`` parts.
_RADIAL_SECTIONS = 16

_SUPPORTED_INDEX_VERSION = 1
_XYZ = 3
_PAIR = 2
#: Extents below this are treated as degenerate and left unscaled, which keeps
#: the flat ground plane (zero z extent) from dividing by zero.
_MIN_EXTENT_M = 1e-9


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PartSpec:
    """One primitive inside a ``parts`` model, in the normalised unit box.

    Attributes
    ----------
    kind
        Primitive shape.
    size
        Extents as fractions of the model's final bounding box.
    at
        Centre of the part in normalised coordinates. Defaults to
        ``(0, 0, size_z / 2)``, i.e. the part stands on the model's base, which
        is what a single-part model always wants.
    axis
        Axis of revolution for ``cylinder`` and ``cone``. Ignored otherwise.
    material
        Material name, resolved to a class exactly as an authored mesh's is.
        Empty means the model's own ``cls``. This is what lets a primitive
        house carry a ``ROOF`` roof on ``WALL`` walls instead of being one
        class throughout.
    """

    kind: PartKind
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    at: tuple[float, float, float] | None = None
    axis: Literal["x", "y", "z"] = "z"
    material: str = ""

    @property
    def centre(self) -> tuple[float, float, float]:
        """``at``, or the standing-on-the-base default.

        A plane has no thickness to stand on, so its default is the base
        itself. Lifting it by half its nominal ``size_z`` put the generated lawn
        half a metre above ground level -- above the launch pads, which then
        sat inside the lawn with no way up.
        """
        if self.at is not None:
            return self.at
        return (0.0, 0.0, 0.0 if self.kind == "plane" else self.size[2] / 2.0)


@dataclass(frozen=True, slots=True, eq=False)
class AssetSpec:
    """One entry in the model library."""

    asset_id: str
    cls: Cls
    """Fallback class for any face whose material the library does not map."""
    tags: tuple[str, ...]
    weight: float = 1.0
    mesh: str | None = None
    """OBJ file, relative to the library directory. Mutually exclusive with ``parts``."""
    object: str | None = None
    """One ``o`` group to take from ``mesh``. ``None`` takes the whole file."""
    up_axis: Literal["y", "z"] = "y"
    """Up axis of the authored file. The shipped library is Y-up."""
    parts: tuple[PartSpec, ...] = ()
    scale: tuple[float, float] = (1.0, 1.0)
    """Uniform multiplier on the authored size. For ``mesh`` models."""
    size_x: tuple[float, float] | None = None
    size_y: tuple[float, float] | None = None
    size_z: tuple[float, float] | None = None
    """Target world extents in metres. For ``parts`` models, which have no authored size."""
    face_axis: FaceAxis = "+z"
    """Which authored axis points away from the wall, for wall-mounted models."""
    yaw_deg: tuple[float, float] = (0.0, 360.0)
    yaw_choices_deg: tuple[float, ...] | None = None
    """Discrete yaw options, drawn uniformly. Overrides ``yaw_deg`` when set."""
    max_edge_m: float | None = None
    """Subdivide to this edge length after scaling. ``None`` leaves the mesh alone."""
    color: Rgb | None = None
    """Overrides :data:`~canopy.contracts.CLASS_COLORS` for this model only."""
    materials: dict[str, Cls] = field(default_factory=dict)
    """Material-to-class overrides for this model, consulted before the library's.

    The library-wide map is keyed on material name alone, and the authored art
    reuses names across very different props: a shed's ``concrete`` pad is not a
    driveway and its ``wood_dark`` trim is not a fence. Overriding per model is
    how a prop keeps its own semantics without renaming anyone else's."""

    @property
    def has_declared_size(self) -> bool:
        """Whether this model declares explicit target extents."""
        return self.size_x is not None and self.size_y is not None and self.size_z is not None

    def sample_yaw(self, rng: np.random.Generator) -> float:
        """Draw a rotation about +Z, in radians."""
        if self.yaw_choices_deg is not None:
            return float(np.deg2rad(rng.choice(np.asarray(self.yaw_choices_deg))))
        return float(np.deg2rad(rng.uniform(*self.yaw_deg)))

    def facing_xy(self) -> npt.NDArray[np.float64]:
        """Return the unit horizontal direction the model faces, in the world frame.

        Wall mounting needs to know which way the art points before rotation.
        For the shipped Y-up library the authored front is ``+z``, which the
        Y-up-to-Z-up conversion turns into world ``-y``; getting this wrong
        mounts the meter facing into the brick.

        Raises
        ------
        AssetError
            If the declared face axis is vertical in the world frame, which
            cannot describe a wall mounting.
        """
        sign = -1.0 if self.face_axis[0] == "-" else 1.0
        native = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}.get(
            self.face_axis[1], np.array([0.0, 0.0, 1.0])
        )
        vec = _to_z_up(sign * native.reshape(1, 3), self.up_axis)[0]
        if abs(vec[2]) > abs(vec[0]) + abs(vec[1]):
            msg = (
                f"asset {self.asset_id!r}: face_axis {self.face_axis!r} is vertical once "
                "converted to the Z-up world frame, so it cannot face a wall"
            )
            raise AssetError(msg)
        horizontal = vec[:2]
        return cast("npt.NDArray[np.float64]", horizontal / np.linalg.norm(horizontal))

    def rgb_for(self, cls: Cls) -> Rgb:
        """Return the true colour for one of this model's classes.

        A per-model ``color`` override applies only to the model's own fallback
        class: overriding it for, say, the roof faces of a house would silently
        break the reveal palette that the viewer and the photo renderer share.
        """
        if self.color is not None and cls is self.cls:
            return self.color
        try:
            return CLASS_COLORS[cls]
        except KeyError as exc:  # pragma: no cover - guards a future Cls addition
            msg = f"asset {self.asset_id!r}: no colour for class {cls.name}"
            raise AssetError(msg) from exc


@dataclass(frozen=True, slots=True)
class CountSpec:
    """How many objects a role places.

    Exactly one source is set. ``cfg_range`` and ``per_10m_wall_cfg`` name a
    field of :class:`~canopy.config.WorldgenCfg`, so a tunable the spec puts in
    ``config/default.yaml`` stays there instead of being duplicated here.
    """

    fixed: int | None = None
    range: tuple[int, int] | None = None
    p: float | None = None
    """With ``range``: probability of the high value. Absent means uniform."""
    cfg_range: str | None = None
    per_10m_wall_cfg: str | None = None

    @property
    def cfg_field(self) -> str | None:
        """Name of the ``WorldgenCfg`` field this count reads, if any."""
        return self.cfg_range or self.per_10m_wall_cfg

    def resolve(
        self,
        rng: np.random.Generator,
        cfg_value: Any = None,
        wall_perimeter_m: float = 0.0,
    ) -> int:
        """Draw the object count for one property.

        Parameters
        ----------
        rng
            The single generator threaded through generation.
        cfg_value
            Value of the field named by :attr:`cfg_field`, or ``None`` for the
            literal forms.
        wall_perimeter_m
            House wall length. Read only by ``per_10m_wall_cfg``.
        """
        if self.fixed is not None:
            return self.fixed
        if self.range is not None:
            lo, hi = self.range
            if self.p is not None:
                return hi if rng.random() < self.p else lo
            return int(rng.integers(lo, hi + 1))
        if self.cfg_range is not None:
            lo, hi = (int(v) for v in cfg_value)
            return int(rng.integers(lo, hi + 1))
        return round(float(cfg_value) * wall_perimeter_m / 10.0)


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """One slot in a property: what to place, where, and how many."""

    name: str
    tag: str
    """Models carrying this tag are the candidates for the role."""
    rule: str
    """Placement rule name, resolved against the registry in ``placement.py``."""
    count: CountSpec
    params: dict[str, Any] = field(default_factory=dict)
    """Rule-specific tunables, passed through verbatim."""
    background: bool = False
    """Whether this role's objects are neighbouring scenery, not the surveyed lot.

    Carried through to every :class:`~canopy.contracts.SceneObject` the role
    produces, which is what lets the mapper ignore it and the viewer draw it in
    full colour from the start.
    """


@dataclass(frozen=True, slots=True, eq=False)
class BuiltMesh:
    """One class's worth of a placed model's geometry, in world coordinates.

    A placement yields several of these when the model's materials map to
    several classes -- the house shell becomes wall, roof, window, door, garage
    door and slab. Each becomes one :class:`~canopy.contracts.SceneObject`.
    """

    cls: Cls
    color: Rgb
    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class AssetLibrary:
    """Every model, material mapping and role, with selection and mesh building."""

    root: Path
    """Directory holding ``index.yaml`` and the mesh files it references."""
    models: dict[str, AssetSpec]
    roles: tuple[RoleSpec, ...]
    """In document order, which is also placement order."""
    materials: dict[str, Cls] = field(default_factory=dict)
    """Material name to semantic class. Unlisted materials use the model's ``cls``."""

    _cache: dict[str, ObjObject] = field(default_factory=dict, repr=False)

    # -- selection ----------------------------------------------------------
    def candidates(self, tag: str) -> list[AssetSpec]:
        """Models carrying ``tag``, in index order.

        Raises
        ------
        AssetError
            If no model carries the tag, which means a role can never be filled.
        """
        found = [spec for spec in self.models.values() if tag in spec.tags]
        if not found:
            known = sorted({t for spec in self.models.values() for t in spec.tags})
            msg = f"no model in {self.root} is tagged {tag!r}; known tags are {known}"
            raise AssetError(msg)
        return found

    def choose(self, tag: str, rng: np.random.Generator) -> AssetSpec:
        """Pick one model for ``tag``, weighted by ``weight``."""
        found = self.candidates(tag)
        weights = np.array([spec.weight for spec in found], dtype=np.float64)
        if not np.all(weights > 0.0):
            msg = f"models tagged {tag!r} must all have weight > 0, got {weights.tolist()}"
            raise AssetError(msg)
        return found[int(rng.choice(len(found), p=weights / weights.sum()))]

    def cls_for(self, spec: AssetSpec, material: str) -> Cls:
        """Semantic class of one material run.

        Most specific wins: the model's own override, then the library-wide map,
        then the model's fallback class.
        """
        if (own := spec.materials.get(material)) is not None:
            return own
        return self.materials.get(material, spec.cls)

    # -- geometry -----------------------------------------------------------
    def canonical(self, spec: AssetSpec) -> ObjObject:
        """Return the model as authored, converted to Z-up metres, cached per asset.

        A ``parts`` model is synthesised in the normalised unit box; a ``mesh``
        model keeps its authored dimensions, because artwork drawn to real-world
        size should be flown against at that size.
        """
        if (cached := self._cache.get(spec.asset_id)) is not None:
            return cached
        obj = _build_parts(spec) if spec.mesh is None else self._read(spec)
        self._cache[spec.asset_id] = obj
        return obj

    def _read(self, spec: AssetSpec) -> ObjObject:
        """Read a mesh file and pick out the requested object."""
        mesh_name = spec.mesh
        if mesh_name is None:  # pragma: no cover - guarded by the caller
            msg = f"asset {spec.asset_id!r} has no mesh file"
            raise AssetError(msg)
        objects = read_obj(self.root / mesh_name, up_axis=spec.up_axis)
        if spec.object is None:
            return _merge(objects.values(), name=spec.asset_id)
        try:
            return objects[spec.object]
        except KeyError as exc:
            msg = (
                f"asset {spec.asset_id!r}: {self.root / mesh_name} has no object "
                f"{spec.object!r}; it contains {sorted(objects)}"
            )
            raise AssetError(msg) from exc

    def native_extents(self, spec: AssetSpec) -> Vec3:
        """Bounding-box extents of the model as authored, in metres."""
        return self.canonical(spec).extents

    def scaled_class_bounds(
        self, spec: AssetSpec, cls: Cls, extents: Vec3
    ) -> npt.NDArray[np.float64]:
        """Bounding box of one class's faces, scaled as if the model were placed.

        Needed because a model's overall bounding box is not its wall. The house
        shell's box includes the gutters and the roof overhang, so mounting a
        meter on it leaves the meter floating half a metre off the brick. Asking
        for the ``WALL`` box instead gives the plane things actually attach to.

        Parameters
        ----------
        spec
            Model to measure.
        cls
            Semantic class to measure. Faces of other classes are ignored.
        extents
            Target extents for the whole model, as passed to :meth:`build`.

        Returns
        -------
        npt.NDArray[np.float64]
            ``[[min xyz], [max xyz]]``, shape ``(2, 3)``, in the model's own
            frame before rotation and translation.

        Raises
        ------
        AssetError
            If the model has no faces of that class.
        """
        obj = self.canonical(spec)
        groups = [g.faces for g in obj.groups if self.cls_for(spec, g.material) is cls]
        if not groups:
            present = sorted({self.cls_for(spec, g.material).name for g in obj.groups})
            msg = (
                f"asset {spec.asset_id!r} has no {cls.name} geometry to measure; "
                f"it carries {present}"
            )
            raise AssetError(msg)
        verts = obj.vertices[np.unique(np.concatenate(groups))]
        factor = _scale_factor(obj.extents, np.asarray(extents, dtype=np.float64))
        return np.array([verts.min(axis=0) * factor, verts.max(axis=0) * factor])

    def sample_extents(self, spec: AssetSpec, rng: np.random.Generator) -> Vec3:
        """Draw the target world extents for one instance.

        A ``mesh`` model is scaled uniformly off its authored size, so its
        proportions survive; a ``parts`` model has no authored size and takes
        its extents per axis from ``size_x/y/z``.

        Raises
        ------
        AssetError
            If a ``parts`` model declares no size ranges, leaving nothing to
            scale it to.
        """
        if spec.has_declared_size:
            ranges = (spec.size_x, spec.size_y, spec.size_z)
            return np.array([rng.uniform(*r) for r in ranges if r is not None], dtype=np.float64)
        if spec.mesh is None:
            msg = (
                f"asset {spec.asset_id!r} is built from `parts`, which have no authored "
                "size, so it must declare size_x, size_y and size_z"
            )
            raise AssetError(msg)
        return self.native_extents(spec) * float(rng.uniform(*spec.scale))

    def build(
        self,
        spec: AssetSpec,
        *,
        pos: Vec3,
        extents: Vec3,
        yaw: float,
        default_max_edge_m: float,
    ) -> list[BuiltMesh]:
        """Instantiate one placed model, split by semantic class.

        The order of operations matters. Scaling happens before subdivision so
        ``max_edge_m`` is a real length; rotation before translation so the yaw
        is about the model's own origin rather than the lot centre; translation
        last so the subdivided coordinates stay small, which keeps the OBJ
        export reproducible.

        Parameters
        ----------
        spec
            Model to instantiate.
        pos
            World position of the model's origin, which the library authors at
            the ground or wall contact point.
        extents
            Target bounding-box extents ``(x, y, z)`` in metres, before yaw.
        yaw
            Rotation about +Z, in radians.
        default_max_edge_m
            Used when the model declares no ``max_edge_m`` of its own and is
            built from ``parts``; authored art is left at its own tessellation.

        Returns
        -------
        list[BuiltMesh]
            One entry per semantic class present in the model, in the order the
            materials first appear.
        """
        obj = self.canonical(spec)
        verts = obj.vertices * _scale_factor(obj.extents, np.asarray(extents, dtype=np.float64))

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        verts = verts @ rot.T
        verts = verts + np.asarray(pos, dtype=np.float64)

        max_edge = spec.max_edge_m
        if max_edge is None and spec.mesh is None:
            max_edge = default_max_edge_m

        by_cls: dict[Cls, list[npt.NDArray[np.int32]]] = {}
        for group in obj.groups:
            by_cls.setdefault(self.cls_for(spec, group.material), []).append(group.faces)

        built = []
        for cls, face_lists in by_cls.items():
            faces = np.concatenate(face_lists) if len(face_lists) > 1 else face_lists[0]
            sub_v, sub_f = _compact(verts, faces)
            if max_edge is not None and max_edge > 0.0:
                sub_v, sub_f = _subdivide(sub_v, sub_f, max_edge)
            built.append(BuiltMesh(cls=cls, color=spec.rgb_for(cls), vertices=sub_v, faces=sub_f))
        return built


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _to_z_up(verts: npt.NDArray[np.float64], up_axis: Literal["y", "z"]) -> npt.NDArray[np.float64]:
    """Rotate ``(x, y, z)`` from a Y-up frame into the project's Z-up frame."""
    if up_axis == "z":
        return verts
    return np.column_stack((verts[:, 0], -verts[:, 2], verts[:, 1]))


def _scale_factor(native: npt.NDArray[np.float64], target: Vec3) -> npt.NDArray[np.float64]:
    """Per-axis multiplier taking ``native`` extents to ``target`` extents.

    A degenerate axis -- the ground plane has no thickness -- is left at 1.0
    rather than dividing by zero.
    """
    safe = np.where(np.abs(native) > _MIN_EXTENT_M, native, 1.0)
    factor = np.asarray(target, dtype=np.float64) / safe
    return np.where(np.abs(native) > _MIN_EXTENT_M, factor, 1.0)


def _compact(
    verts: npt.NDArray[np.float64], faces: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Slice out only the vertices ``faces`` references, and reindex."""
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return verts[used], inverse.reshape(faces.shape).astype(np.int32)


def _subdivide(
    verts: npt.NDArray[np.float64], faces: npt.NDArray[np.int32], max_edge_m: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Split triangles until no edge is longer than ``max_edge_m``.

    Fine triangles are what make the grayscale-to-colour reveal read as a wipe
    across a wall rather than whole objects popping into colour.
    """
    out_v, out_f = trimesh.remesh.subdivide_to_size(verts, faces, max_edge=max_edge_m)
    return np.asarray(out_v, dtype=np.float64), np.asarray(out_f, dtype=np.int32)


def _merge(objects: Iterable[ObjObject], name: str) -> ObjObject:
    """Concatenate several OBJ objects into one, keeping material groups."""
    items = list(objects)
    if not items:
        msg = f"asset {name!r}: nothing to merge"
        raise AssetError(msg)
    if len(items) == 1:
        return items[0]

    verts = np.concatenate([o.vertices for o in items])
    groups: dict[str, list[npt.NDArray[np.int32]]] = {}
    offset = 0
    for obj in items:
        for group in obj.groups:
            groups.setdefault(group.material, []).append(group.faces + offset)
        offset += len(obj.vertices)
    return ObjObject(
        name=name,
        vertices=verts,
        groups=tuple(MaterialGroup(material=m, faces=np.concatenate(f)) for m, f in groups.items()),
    )


# ---------------------------------------------------------------------------
# Primitive construction
# ---------------------------------------------------------------------------
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


def _build_parts(spec: AssetSpec) -> ObjObject:
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def default_assets_dir() -> Path:
    """Locate the model library directory.

    Honours ``CANOPY_ASSETS_DIR``, then walks up from this module looking for
    ``assets/models/index.yaml`` -- the same strategy :mod:`canopy.config` uses
    to find ``config/``, and for the same reason: both are edited mid-run.

    Raises
    ------
    AssetError
        If the override is not a directory, or no library is found.
    """
    if (env := os.environ.get("CANOPY_ASSETS_DIR")) is not None:
        path = Path(env).expanduser()
        if not path.is_dir():
            msg = f"CANOPY_ASSETS_DIR={env!r} is not a directory"
            raise AssetError(msg)
        return path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets" / "models" / "index.yaml"
        if candidate.is_file():
            return candidate.parent

    msg = (
        "could not locate assets/models/index.yaml by walking up from "
        f"{Path(__file__).resolve()}; set CANOPY_ASSETS_DIR"
    )
    raise AssetError(msg)


def default_index_path() -> Path:
    """Path to the shipped model index."""
    return default_assets_dir() / "index.yaml"


def _mapping(value: Any, where: str) -> dict[str, Any]:
    """Return ``value`` as a mapping, or raise."""
    if not isinstance(value, dict):
        msg = f"{where}: expected a mapping, got {type(value).__name__}"
        raise AssetError(msg)
    return value


def _reject_unknown(data: dict[str, Any], known: tuple[str, ...], where: str) -> None:
    """Fail loudly on a misspelled key rather than silently ignoring it."""
    if unknown := sorted(set(data) - set(known)):
        msg = f"{where}: unknown key(s) {unknown}; known keys are {sorted(known)}"
        raise AssetError(msg)


def _triple(value: Any, where: str) -> tuple[float, float, float]:
    """Coerce a 3-sequence of numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != _XYZ:
        msg = f"{where}: expected 3 numbers, got {value!r}"
        raise AssetError(msg)
    return (float(value[0]), float(value[1]), float(value[2]))


def _pair(value: Any, where: str) -> tuple[float, float]:
    """Coerce an inclusive ``[lo, hi]`` range, rejecting an inverted one."""
    if not isinstance(value, (list, tuple)) or len(value) != _PAIR:
        msg = f"{where}: expected [lo, hi], got {value!r}"
        raise AssetError(msg)
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        msg = f"{where}: lo ({lo}) must not exceed hi ({hi})"
        raise AssetError(msg)
    return (lo, hi)


def _int_pair(value: Any, where: str) -> tuple[int, int]:
    """Coerce an inclusive integer ``[lo, hi]`` range."""
    lo, hi = _pair(value, where)
    return (int(lo), int(hi))


def _cls(name: Any, where: str) -> Cls:
    """Coerce a :class:`~canopy.contracts.Cls` member name."""
    if not isinstance(name, str) or name not in Cls.__members__:
        msg = f"{where}: expected one of {list(Cls.__members__)}, got {name!r}"
        raise AssetError(msg)
    return Cls[name]


def _parse_part(data: Any, where: str) -> PartSpec:
    """Build one :class:`PartSpec`."""
    raw = _mapping(data, where)
    _reject_unknown(raw, ("kind", "size", "at", "axis", "material"), where)
    kind = raw.get("kind")
    if kind not in _PART_KINDS:
        msg = f"{where}: kind must be one of {_PART_KINDS}, got {kind!r}"
        raise AssetError(msg)
    axis = raw.get("axis", "z")
    if axis not in ("x", "y", "z"):
        msg = f"{where}: axis must be 'x', 'y' or 'z', got {axis!r}"
        raise AssetError(msg)
    return PartSpec(
        kind=cast("PartKind", kind),
        size=_triple(raw.get("size", [1.0, 1.0, 1.0]), f"{where}.size"),
        at=None if raw.get("at") is None else _triple(raw["at"], f"{where}.at"),
        axis=cast("Literal['x', 'y', 'z']", axis),
        material=str(raw.get("material", "")),
    )


_MODEL_KEYS = (
    "cls",
    "tags",
    "weight",
    "mesh",
    "object",
    "up_axis",
    "parts",
    "scale",
    "size_x",
    "size_y",
    "size_z",
    "face_axis",
    "yaw_deg",
    "yaw_choices_deg",
    "max_edge_m",
    "color",
    "materials",
)


def _parse_model(asset_id: str, data: Any) -> AssetSpec:
    """Build one :class:`AssetSpec`."""
    where = f"models.{asset_id}"
    raw = _mapping(data, where)
    _reject_unknown(raw, _MODEL_KEYS, where)

    tags = raw.get("tags", [])
    if not isinstance(tags, (list, tuple)) or not all(isinstance(t, str) for t in tags):
        msg = f"{where}.tags: expected a list of strings, got {tags!r}"
        raise AssetError(msg)
    if not tags:
        msg = f"{where}.tags: a model with no tags can never fill a role"
        raise AssetError(msg)

    has_mesh, has_parts = raw.get("mesh") is not None, bool(raw.get("parts"))
    if has_mesh == has_parts:
        msg = f"{where}: set exactly one of `mesh` (a file) or `parts` (primitives)"
        raise AssetError(msg)
    if raw.get("object") is not None and not has_mesh:
        msg = f"{where}.object: names an `o` group in a `mesh` file, so it needs `mesh`"
        raise AssetError(msg)

    up_axis = raw.get("up_axis", "y")
    if up_axis not in ("y", "z"):
        msg = f"{where}.up_axis: expected 'y' or 'z', got {up_axis!r}"
        raise AssetError(msg)

    face_axis = raw.get("face_axis", "+z")
    if face_axis not in _FACE_AXES:
        msg = f"{where}.face_axis: expected one of {_FACE_AXES}, got {face_axis!r}"
        raise AssetError(msg)

    color = raw.get("color")
    if color is not None:
        rgb = _triple(color, f"{where}.color")
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    choices = raw.get("yaw_choices_deg")
    if choices is not None and (not isinstance(choices, (list, tuple)) or not choices):
        msg = f"{where}.yaw_choices_deg: expected a non-empty list, got {choices!r}"
        raise AssetError(msg)

    max_edge = raw.get("max_edge_m")
    if max_edge is not None and float(max_edge) <= 0.0:
        msg = f"{where}.max_edge_m: must be positive, or omitted to skip subdivision"
        raise AssetError(msg)

    return AssetSpec(
        asset_id=asset_id,
        cls=_cls(raw.get("cls"), f"{where}.cls"),
        tags=tuple(tags),
        weight=float(raw.get("weight", 1.0)),
        mesh=raw.get("mesh"),
        object=raw.get("object"),
        up_axis=cast("Literal['y', 'z']", up_axis),
        parts=tuple(
            _parse_part(p, f"{where}.parts[{i}]") for i, p in enumerate(raw.get("parts", []))
        ),
        scale=_pair(raw.get("scale", [1.0, 1.0]), f"{where}.scale"),
        size_x=None if raw.get("size_x") is None else _pair(raw["size_x"], f"{where}.size_x"),
        size_y=None if raw.get("size_y") is None else _pair(raw["size_y"], f"{where}.size_y"),
        size_z=None if raw.get("size_z") is None else _pair(raw["size_z"], f"{where}.size_z"),
        face_axis=cast("FaceAxis", face_axis),
        yaw_deg=_pair(raw.get("yaw_deg", [0.0, 360.0]), f"{where}.yaw_deg"),
        yaw_choices_deg=None if choices is None else tuple(float(c) for c in choices),
        max_edge_m=None if max_edge is None else float(max_edge),
        color=color,
        materials={
            name: _cls(value, f"{where}.materials.{name}")
            for name, value in _mapping(raw.get("materials", {}), f"{where}.materials").items()
        },
    )


_COUNT_SOURCES = ("fixed", "range", "cfg_range", "per_10m_wall_cfg")


def _parse_count(data: Any, where: str) -> CountSpec:
    """Build one :class:`CountSpec`, insisting on exactly one source."""
    raw = _mapping(data, where)
    _reject_unknown(raw, (*_COUNT_SOURCES, "p"), where)
    present = [k for k in _COUNT_SOURCES if k in raw]
    if len(present) != 1:
        msg = f"{where}: set exactly one of {list(_COUNT_SOURCES)}, got {present}"
        raise AssetError(msg)
    if "p" in raw and "range" not in raw:
        msg = f"{where}: `p` weights a `range`, so it needs one"
        raise AssetError(msg)
    return CountSpec(
        fixed=None if raw.get("fixed") is None else int(raw["fixed"]),
        range=None if raw.get("range") is None else _int_pair(raw["range"], f"{where}.range"),
        p=None if raw.get("p") is None else float(raw["p"]),
        cfg_range=raw.get("cfg_range"),
        per_10m_wall_cfg=raw.get("per_10m_wall_cfg"),
    )


def _parse_role(name: str, data: Any) -> RoleSpec:
    """Build one :class:`RoleSpec`."""
    where = f"roles.{name}"
    raw = _mapping(data, where)
    _reject_unknown(raw, ("tag", "rule", "count", "params", "background"), where)
    for required in ("tag", "rule", "count"):
        if required not in raw:
            msg = f"{where}: missing key {required!r}"
            raise AssetError(msg)
    return RoleSpec(
        name=name,
        tag=str(raw["tag"]),
        rule=str(raw["rule"]),
        count=_parse_count(raw["count"], f"{where}.count"),
        params=_mapping(raw.get("params", {}), f"{where}.params"),
        background=bool(raw.get("background", False)),
    )


def load_library(path: Path | str | None = None) -> AssetLibrary:
    """Load and validate the model index.

    Parameters
    ----------
    path
        ``index.yaml`` to read. Defaults to the shipped library.

    Returns
    -------
    AssetLibrary
        Validated models, material mappings and roles. Role order follows the
        document, and that order is placement order.

    Raises
    ------
    AssetError
        If the file is unreadable, of an unsupported version, or holds a
        malformed model, material or role. Nothing here is a warning: a
        misspelled key in the library would otherwise surface as a
        mysteriously empty yard.
    """
    resolved = Path(path) if path is not None else default_index_path()
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read model index {resolved}: {exc}"
        raise AssetError(msg) from exc

    doc = _mapping(yaml.safe_load(text), str(resolved))
    _reject_unknown(doc, ("version", "roles", "models", "materials"), str(resolved))

    version = doc.get("version")
    if version != _SUPPORTED_INDEX_VERSION:
        msg = f"{resolved}: version must be {_SUPPORTED_INDEX_VERSION}, got {version!r}"
        raise AssetError(msg)

    models = {
        asset_id: _parse_model(asset_id, data)
        for asset_id, data in _mapping(doc.get("models", {}), "models").items()
    }
    if not models:
        msg = f"{resolved}: the library declares no models"
        raise AssetError(msg)

    roles = tuple(
        _parse_role(name, data) for name, data in _mapping(doc.get("roles", {}), "roles").items()
    )
    if not roles:
        msg = f"{resolved}: the library declares no roles, so nothing would be placed"
        raise AssetError(msg)

    materials = {
        name: _cls(value, f"materials.{name}")
        for name, value in _mapping(doc.get("materials", {}), "materials").items()
    }

    library = AssetLibrary(
        root=resolved.parent.parent, models=models, roles=roles, materials=materials
    )
    for role in roles:
        library.candidates(role.tag)  # fail now, not mid-generation
    return library
