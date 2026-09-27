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

Two neighbouring modules pull their own weight out of what used to live here:
:mod:`canopy.worldgen.index_schema` turns the YAML document into the specs
below (``load_library``), and :mod:`canopy.worldgen.primitives` builds the
unit-box meshes a ``parts:`` model is made of (``build_parts``). This module
keeps the specs themselves and the runtime :class:`AssetLibrary` that selects
and instantiates them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import trimesh

from canopy.contracts import CLASS_COLORS, Cls, MaterialRun, Rgb, Vec3
from canopy.errors import AssetError
from canopy.worldgen.objio import MaterialGroup, MtlMaterial, ObjObject, read_mtl, read_obj
from canopy.worldgen.primitives import build_parts

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
]

#: Supported primitive part kinds. Extending this tuple is the only reason a new
#: *shape* needs code; a new size or arrangement of existing shapes does not.
#: The list of kinds itself is validated in :mod:`canopy.worldgen.index_schema`.
PartKind = Literal["plane", "box", "ellipsoid", "cylinder", "cone", "gable_prism"]

#: Which way a model faces before it is rotated, in its own authored frame.
#: The list of axes is validated in :mod:`canopy.worldgen.index_schema`.
FaceAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

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
    runs: tuple[MaterialRun, ...] = ()
    """Display colours of ``faces``, as runs in face order; see
    :attr:`~canopy.contracts.SceneObject.materials`. Run ``k`` covers the next
    ``runs[k].n_faces`` faces after the ones before it, and the counts sum to
    ``len(faces)``."""


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
    palette: dict[str, Rgb] = field(default_factory=dict)
    """Material name to display colour (sRGB 0..255), for the viewer only.

    A ``parts:`` model has no authored MTL to draw its per-material colour
    from, so this is how one of its materials -- ``roof_shingle``, say -- gets
    a display colour of its own instead of the flat class colour every other
    material of that class would share. Unlisted materials fall through to
    :meth:`display_rgb`'s other sources; sensing never reads this.
    """

    _cache: dict[str, ObjObject] = field(default_factory=dict, repr=False)
    _mtl_cache: dict[str, dict[str, MtlMaterial]] = field(default_factory=dict, repr=False)
    """Parsed MTL files, keyed by the ``mesh`` path that named them.

    A missing MTL is recorded as ``{}`` rather than left unread, so a model
    with no authored materials (most ``parts:`` models, and some ``mesh:``
    ones) is not re-stat'd on every :meth:`display_rgb` call.
    """

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

    def display_rgb(self, spec: AssetSpec, material: str) -> Rgb:
        """Return the colour the viewer should paint one material run in.

        Most specific wins, same principle as :meth:`cls_for`, but over a
        different ladder: the authored MTL's own ``Kd`` beside a ``mesh:``
        model (only that kind can have one), then the library-wide ``palette``
        for a ``parts:`` prop with no art of its own, then the class's one flat
        sensed colour -- which is also what :attr:`~canopy.contracts.SceneObject.materials`
        falls back to when it is empty, so an unmapped material is never
        actually wrong, just less specific.
        """
        if spec.mesh is not None:
            mtl = self._mtl_for(spec.mesh)
            if (found := mtl.get(material)) is not None:
                return _rgb_from_kd(found.diffuse)
        if (rgb := self.palette.get(material)) is not None:
            return rgb
        return spec.rgb_for(self.cls_for(spec, material))

    def _mtl_for(self, mesh: str) -> dict[str, MtlMaterial]:
        """Return the MTL beside ``mesh``, cached by mesh name; ``{}`` if none exists.

        A missing MTL is not an error -- ``parts:`` models and some authored
        ones have none -- so a miss is cached too, or every uncoloured
        material would re-``stat`` the same absent file on every run.
        """
        if (cached := self._mtl_cache.get(mesh)) is not None:
            return cached
        path = (self.root / mesh).with_suffix(".mtl")
        found = read_mtl(path) if path.is_file() else {}
        self._mtl_cache[mesh] = found
        return found

    # -- geometry -----------------------------------------------------------
    def canonical(self, spec: AssetSpec) -> ObjObject:
        """Return the model as authored, converted to Z-up metres, cached per asset.

        A ``parts`` model is synthesised in the normalised unit box; a ``mesh``
        model keeps its authored dimensions, because artwork drawn to real-world
        size should be flown against at that size.
        """
        if (cached := self._cache.get(spec.asset_id)) is not None:
            return cached
        obj = build_parts(spec) if spec.mesh is None else self._read(spec)
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
            materials first appear. Each carries :attr:`BuiltMesh.runs`, the
            same class's faces further split by authored material, purely for
            the viewer -- :attr:`BuiltMesh.color` is still the one flat colour
            every one of the class's faces reports to the ranger, unchanged by
            any of this.
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

        by_cls: dict[Cls, list[MaterialGroup]] = {}
        for group in obj.groups:
            by_cls.setdefault(self.cls_for(spec, group.material), []).append(group)

        built = []
        for cls, groups in by_cls.items():
            faces = (
                np.concatenate([g.faces for g in groups]) if len(groups) > 1 else groups[0].faces
            )
            # Which of `groups` each face of `faces` came from, in the same
            # first-appearance order: the groups are already concatenated in
            # that order, so this starts out sorted ascending.
            face_material = np.repeat(np.arange(len(groups)), [len(g.faces) for g in groups])
            sub_v, sub_f = _compact(verts, faces)
            if max_edge is not None and max_edge > 0.0:
                sub_v, sub_f, source_face = _subdivide(sub_v, sub_f, max_edge)
                face_material = face_material[source_face]
                # Subdivision does not preserve the material's contiguous run,
                # since a split face's children are interleaved with its
                # neighbours'; sort them back into runs. Stable, so faces that
                # share a material keep their relative order (cosmetic, but
                # deterministic output is the point of this whole module).
                order = np.argsort(face_material, kind="stable")
                sub_f = sub_f[order]
                face_material = face_material[order]
            counts = np.bincount(face_material, minlength=len(groups))
            runs = tuple(
                MaterialRun(
                    material=group.material,
                    rgb=self.display_rgb(spec, group.material),
                    n_faces=int(n),
                )
                for group, n in zip(groups, counts, strict=True)
            )
            built.append(
                BuiltMesh(cls=cls, color=spec.rgb_for(cls), vertices=sub_v, faces=sub_f, runs=runs)
            )
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Split triangles until no edge is longer than ``max_edge_m``.

    Fine triangles are what make the grayscale-to-colour reveal read as a wipe
    across a wall rather than whole objects popping into colour.

    Returns
    -------
    tuple
        ``(vertices, faces, source_face)``: ``source_face[i]`` is the index,
        into the *input* ``faces``, of the original triangle output face ``i``
        was split from. :meth:`AssetLibrary.build` uses it to carry a face's
        material through subdivision, since a split face is otherwise
        indistinguishable from any other.
    """
    out_v, out_f, source = trimesh.remesh.subdivide_to_size(
        verts, faces, max_edge=max_edge_m, return_index=True
    )
    return (
        np.asarray(out_v, dtype=np.float64),
        np.asarray(out_f, dtype=np.int32),
        np.asarray(source, dtype=np.int64),
    )


def _rgb_from_kd(diffuse: tuple[float, float, float]) -> Rgb:
    """Convert an authored ``Kd`` (0..1 float, as authored) to display sRGB 0..255.

    Clipped rather than trusted: art occasionally authors a ``Kd`` a hair
    outside ``[0, 1]`` (a bloom-adjacent highlight colour, say), and an 8-bit
    channel outside ``[0, 255]`` is meaningless to three.js.
    """
    r, g, b = (max(0, min(255, round(c * 255))) for c in diffuse)
    return (r, g, b)


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
