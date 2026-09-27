"""Parsing and validation of ``assets/models/index.yaml`` into typed specs.

Split out of :mod:`canopy.worldgen.assets` because this is pure document
validation -- coercing YAML's ``Any`` into :class:`~canopy.worldgen.assets.PartSpec`,
:class:`~canopy.worldgen.assets.AssetSpec`, :class:`~canopy.worldgen.assets.CountSpec`
and :class:`~canopy.worldgen.assets.RoleSpec`, rejecting anything malformed --
with no geometry and no runtime library behaviour in it. Isolating it here
means the "is this key spelled right" logic is one file to audit, distinct
from mesh building or asset selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml

from canopy.contracts import Cls, Rgb
from canopy.errors import AssetError
from canopy.worldgen.assets import (
    AssetLibrary,
    AssetSpec,
    CountSpec,
    FaceAxis,
    PartKind,
    PartSpec,
    RoleSpec,
    default_index_path,
)

__all__ = ["load_library"]

_SUPPORTED_INDEX_VERSION = 1
_XYZ = 3
_PAIR = 2
#: Upper bound of one 8-bit display colour channel.
_RGB_MAX = 255

#: Supported primitive part kinds, mirroring :data:`~canopy.worldgen.assets.PartKind`.
_PART_KINDS: tuple[str, ...] = ("plane", "box", "ellipsoid", "cylinder", "cone", "gable_prism")
#: Mirrors :data:`~canopy.worldgen.assets.FaceAxis`.
_FACE_AXES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")


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


def _rgb_value(value: Any, where: str) -> Rgb:
    """Coerce a display colour: 3 integers in ``[0, 255]``."""
    triple = _triple(value, where)
    ints = tuple(int(c) for c in triple)
    if any(c < 0 or c > _RGB_MAX for c in ints):
        msg = f"{where}: expected 3 integers in [0, {_RGB_MAX}], got {value!r}"
        raise AssetError(msg)
    return (ints[0], ints[1], ints[2])


def _cls(name: Any, where: str) -> Cls:
    """Coerce a :class:`~canopy.contracts.Cls` member name."""
    if not isinstance(name, str) or name not in Cls.__members__:
        msg = f"{where}: expected one of {list(Cls.__members__)}, got {name!r}"
        raise AssetError(msg)
    return Cls[name]


def _parse_part(data: Any, where: str) -> PartSpec:
    """Build one :class:`~canopy.worldgen.assets.PartSpec`."""
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
    """Build one :class:`~canopy.worldgen.assets.AssetSpec`."""
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
    """Build one :class:`~canopy.worldgen.assets.CountSpec`, insisting on exactly one source."""
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
    """Build one :class:`~canopy.worldgen.assets.RoleSpec`."""
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
    _reject_unknown(doc, ("version", "roles", "models", "materials", "palette"), str(resolved))

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
    palette = {
        name: _rgb_value(value, f"palette.{name}")
        for name, value in _mapping(doc.get("palette", {}), "palette").items()
    }

    library = AssetLibrary(
        root=resolved.parent.parent,
        models=models,
        roles=roles,
        materials=materials,
        palette=palette,
    )
    for role in roles:
        library.candidates(role.tag)  # fail now, not mid-generation
    return library
