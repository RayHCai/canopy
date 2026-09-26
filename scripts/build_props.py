"""Author the model library's props as low-poly, flat-shaded OBJ files.

Run it to (re)build every placeable prop under ``assets/obj_export/assets/``
except the drone::

    python scripts/build_props.py [--out DIR] [--only NAME ...] [--catalog]

Why a script rather than a modelling package. The library's art is data the
generator places (``assets/models/index.yaml``), and every downstream stage
measures it: placement rules read each model's bounding box, the site solver's
``rules.yaml`` records the battery's, and the detectors are tuned against the
extents of the meter, the conduit and the bushes. Authoring the props in code
keeps those numbers pinned -- each builder ends with an ``expect`` on the box
it must fill -- and makes a restyle a diff rather than a re-export.

Style. The props are "POLYGON" low-poly: chunky silhouettes, chamfered edges,
faceted domes and blobs, and one flat colour per material with no textures.
Flat shading is the renderer's job (the viewer un-indexes and computes face
normals), so the files carry positions and faces only. Every solid is convex
or a union of convex pieces, so the parts are built as convex hulls of a few
points, which also guarantees outward winding -- the viewer culls back faces,
and a wall drawn inside-out disappears from outside.

Conventions, shared with the rest of the library: Y-up, metres, origin at the
model's ground or wall contact point, front (the face that turns away from a
wall) on +Z. Material names matter: ``index.yaml`` maps them to semantic
classes, so the names here are chosen to land in the right class -- a shed's
pad is ``shed_pad``, not ``concrete``, which the library reads as driveway.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh

from canopy.worldgen.objio import read_mtl, read_obj

Vec = tuple[float, float, float]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# ---------------------------------------------------------------------------
# Palette: sRGB in [0, 1]. Saturated, clean flats; the darker variants are for
# recesses and trim, so a part reads even when the light is flat.
# ---------------------------------------------------------------------------
PALETTE: dict[str, tuple[float, float, float]] = {
    # Painted and galvanised steel: the service equipment family.
    "steel_paint": (0.58, 0.67, 0.76),
    "steel_paint_dark": (0.40, 0.48, 0.57),
    "panel_paint": (0.62, 0.70, 0.78),
    "panel_seam": (0.30, 0.34, 0.38),
    "galvanized": (0.66, 0.71, 0.75),
    "conduit_paint": (0.63, 0.71, 0.82),
    "meter_bezel": (0.22, 0.24, 0.27),
    "meter_ring": (0.74, 0.76, 0.78),
    "meter_face": (0.95, 0.96, 0.97),
    "meter_lcd": (0.45, 0.56, 0.42),
    "meter_glass": (0.78, 0.86, 0.95),
    "label_yellow": (0.95, 0.76, 0.12),
    "label_black": (0.14, 0.14, 0.16),
    "label_white": (0.95, 0.95, 0.93),
    "label_red": (0.80, 0.16, 0.14),
    # Gas service.
    "gas_meter_body": (0.72, 0.73, 0.71),
    "gas_meter_dark": (0.36, 0.37, 0.39),
    "dial_dark": (0.15, 0.15, 0.17),
    "gas_pipe": (0.85, 0.66, 0.16),
    "gas_iron": (0.47, 0.49, 0.51),
    # Air conditioner.
    "ac_pad": (0.74, 0.73, 0.70),
    "ac_cabinet": (0.83, 0.80, 0.73),
    "ac_cabinet_dark": (0.66, 0.63, 0.57),
    "ac_grille": (0.20, 0.20, 0.22),
    "ac_coil": (0.30, 0.28, 0.27),
    "ac_fan": (0.12, 0.12, 0.13),
    "foam_black": (0.14, 0.14, 0.15),
    "copper": (0.76, 0.46, 0.20),
    # Battery.
    "core_white": (0.94, 0.94, 0.92),
    "core_panel": (0.86, 0.87, 0.86),
    "core_trim": (0.22, 0.23, 0.25),
    "core_led": (0.31, 0.75, 1.00),
    "core_vent": (0.35, 0.36, 0.38),
    "disconnect_gray": (0.60, 0.64, 0.68),
    # Shed.
    "shed_pad": (0.70, 0.69, 0.66),
    "shed_siding": (0.79, 0.71, 0.58),
    "shed_siding_dark": (0.66, 0.58, 0.46),
    "shed_trim": (0.95, 0.94, 0.90),
    "shed_door": (0.47, 0.35, 0.25),
    "shed_roof": (0.32, 0.30, 0.29),
    "shed_roof_light": (0.42, 0.40, 0.39),
    "shed_vent": (0.10, 0.10, 0.11),
    "shed_glass": (0.60, 0.76, 0.86),
    "shed_ramp": (0.55, 0.41, 0.28),
    # Fence.
    "wood_dark": (0.55, 0.41, 0.27),
    "wood": (0.68, 0.51, 0.33),
    "wood_gray": (0.60, 0.56, 0.49),
    "wood_light": (0.76, 0.61, 0.43),
    # House shell. Names the library maps to a class are load-bearing: `slab`
    # and `concrete` are DRIVEWAY, `window_glass` WINDOW, `door_red` DOOR,
    # `garage_door` GARAGE_DOOR, `roof_shingle`/`shingle_light`/`gutter` ROOF;
    # the rest fall to WALL unless index.yaml overrides them for the house.
    "slab": (0.68, 0.67, 0.64),
    "concrete": (0.76, 0.75, 0.72),
    "brick": (0.75, 0.48, 0.38),
    "gable_siding": (0.93, 0.90, 0.82),
    "trim_white": (0.96, 0.96, 0.94),
    "shutter": (0.18, 0.32, 0.30),
    "roof_shingle": (0.33, 0.31, 0.30),
    "shingle_light": (0.44, 0.42, 0.41),
    "gutter": (0.92, 0.92, 0.90),
    "barge_board": (0.96, 0.96, 0.94),
    "downspout": (0.86, 0.86, 0.84),
    "chimney": (0.60, 0.40, 0.33),
    "chimney_cap": (0.72, 0.71, 0.68),
    "door_red": (0.62, 0.16, 0.14),
    "door_glass": (0.62, 0.78, 0.88),
    "door_knob": (0.85, 0.72, 0.35),
    "porch_roof": (0.33, 0.31, 0.30),
    "garage_door": (0.92, 0.91, 0.88),
    "garage_line": (0.78, 0.77, 0.74),
    "window_glass": (0.56, 0.74, 0.86),
    "gable_vent": (0.30, 0.30, 0.32),
    "lamp": (0.20, 0.20, 0.22),
    # Vegetation.
    "bark": (0.38, 0.30, 0.23),
    "bark_dark": (0.27, 0.22, 0.16),
    "leaf_light": (0.64, 0.82, 0.40),
    "leaf": (0.50, 0.72, 0.31),
    "leaf_dark": (0.37, 0.58, 0.24),
}

#: Materials drawn see-through, as ``d`` in the MTL.
OPACITY: dict[str, float] = {"meter_glass": 0.45, "shed_glass": 0.7}

#: Rounded parts use this many sides: few enough to read as facets.
SIDES = 12


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def rot(axis: str, deg: float) -> FloatArray:
    """Rotation matrix about one authored axis."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _ring(n: int, r: float, y: float, phase: float = 0.0) -> list[Vec]:
    """``n`` points on a horizontal circle of radius ``r`` at height ``y``."""
    return [
        (r * math.cos(2 * math.pi * k / n + phase), y, r * math.sin(2 * math.pi * k / n + phase))
        for k in range(n)
    ]


def _apply(m: FloatArray, points: list[Vec]) -> list[Vec]:
    """Rotate every point by ``m``."""
    arr = np.asarray(points, dtype=np.float64) @ m.T
    return [(float(p[0]), float(p[1]), float(p[2])) for p in arr]


def _swing(points: list[Vec], axis: str) -> list[Vec]:
    """Turn a Y-axis solid so its axis lies along ``axis``."""
    if axis == "y":
        return points
    return _apply(rot("z", -90.0) if axis == "x" else rot("x", 90.0), points)


@dataclass
class Model:
    """Parts accumulated for one OBJ, each a convex hull or an explicit mesh."""

    name: str
    verts: list[FloatArray] = field(default_factory=list)
    faces: list[IntArray] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)

    def add(self, verts: FloatArray, faces: IntArray, material: str) -> None:
        """Append one part."""
        if material not in PALETTE:
            msg = f"{self.name}: unknown material {material!r}"
            raise KeyError(msg)
        self.verts.append(np.asarray(verts, dtype=np.float64))
        self.faces.append(np.asarray(faces, dtype=np.int64))
        self.materials.append(material)

    def hull(self, points: list[Vec], material: str, at: Vec = (0.0, 0.0, 0.0)) -> None:
        """Add the convex hull of ``points``, translated by ``at``."""
        pts = np.asarray(points, dtype=np.float64) + np.asarray(at, dtype=np.float64)
        h = trimesh.convex.convex_hull(pts)
        self.add(np.asarray(h.vertices, dtype=np.float64), np.asarray(h.faces), material)

    def box(
        self,
        size: Vec,
        at: Vec,
        material: str,
        bevel: float = 0.0,
        m: FloatArray | None = None,
    ) -> None:
        """Add a box centred on ``at``; ``bevel`` chamfers every edge by that much."""
        w, h, d = (s / 2.0 for s in size)
        b = min(bevel, w, h, d)
        pts: list[Vec] = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    if b > 0:
                        pts.append((sx * (w - b), sy * h, sz * d))
                        pts.append((sx * w, sy * (h - b), sz * d))
                        pts.append((sx * w, sy * h, sz * (d - b)))
                    else:
                        pts.append((sx * w, sy * h, sz * d))
        if m is not None:
            pts = _apply(m, pts)
        self.hull(pts, material, at)

    def prism(
        self,
        r: float,
        h: float,
        at: Vec,
        material: str,
        n: int = SIDES,
        axis: str = "y",
        r_top: float | None = None,
        phase: float = 0.0,
        m: FloatArray | None = None,
    ) -> None:
        """Add an ``n``-gon cylinder (or frustum) centred on ``at``, along ``axis``."""
        rt = r if r_top is None else r_top
        pts = _ring(n, r, -h / 2.0, phase) + _ring(n, rt, h / 2.0, phase)
        pts = _swing(pts, axis)
        if m is not None:
            pts = _apply(m, pts)
        self.hull(pts, material, at)

    def dome(
        self, r: float, h: float, at: Vec, material: str, n: int = SIDES, axis: str = "y"
    ) -> None:
        """Add a faceted half-dome of radius ``r`` and height ``h`` rising from ``at``."""
        pts: list[Vec] = _ring(n, r, 0.0)
        for t in (0.55, 0.85):
            pts += _ring(n, r * math.sqrt(1 - t * t), h * t, math.pi / n)
        pts.append((0.0, h, 0.0))
        self.hull(_swing(pts, axis), material, at)

    def lobe(
        self,
        r: float,
        at: Vec,
        material: str,
        rng: np.random.Generator,
        jitter: float = 0.12,
        squash: float = 0.85,
    ) -> None:
        """Add a faceted blob: a coarse icosphere with its vertices pushed in and out.

        The one part that is not a convex hull, so its winding is checked:
        every face must still look away from the blob's centre, which a
        radial jitter this small cannot break but a larger one could.
        """
        ico = trimesh.creation.icosphere(subdivisions=1, radius=r)
        v = np.asarray(ico.vertices, dtype=np.float64)
        v = v * (1.0 + rng.uniform(-jitter, jitter, size=(len(v), 1)))
        v = v * np.array([1.0, squash, 1.0])
        f = np.asarray(ico.faces, dtype=np.int64)
        normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        if np.any(np.einsum("ij,ij->i", normals, v[f].mean(axis=1)) <= 0.0):
            msg = f"{self.name}: a lobe face turned inward (jitter {jitter} is too large)"
            raise ValueError(msg)
        self.add(v + np.asarray(at, dtype=np.float64), f, material)

    def tube(
        self,
        r_in: float,
        r_out: float,
        h: float,
        at: Vec,
        material: str,
        n: int = SIDES,
        axis: str = "y",
    ) -> None:
        """Add a ring with a hole: the one non-convex part, so it is wound by hand."""
        outer_b, outer_t = _ring(n, r_out, -h / 2), _ring(n, r_out, h / 2)
        inner_b, inner_t = _ring(n, r_in, -h / 2), _ring(n, r_in, h / 2)
        pts = _swing(outer_b + outer_t + inner_b + inner_t, axis)
        v = np.asarray(pts, dtype=np.float64) + np.asarray(at, dtype=np.float64)
        f: list[tuple[int, int, int]] = []
        for k in range(n):
            j = (k + 1) % n
            ob, ot, ib, it = k, n + k, 2 * n + k, 3 * n + k
            jb, jt, jib, jit = j, n + j, 2 * n + j, 3 * n + j
            f += [(ob, jt, jb), (ob, ot, jt)]  # outer wall, outward
            f += [(ib, jib, jit), (ib, jit, it)]  # inner wall, facing the hole
            f += [(ot, jit, jt), (ot, it, jit)]  # top
            f += [(ob, jb, jib), (ob, jib, ib)]  # bottom
        self.add(v, np.asarray(f, dtype=np.int64), material)

    def bounds(self) -> FloatArray:
        """``[[min xyz], [max xyz]]`` over every part."""
        v = np.concatenate(self.verts)
        return np.stack([v.min(axis=0), v.max(axis=0)])

    def n_faces(self) -> int:
        """Triangle count."""
        return sum(len(f) for f in self.faces)


def expect(model: Model, lo: Vec, hi: Vec, tol: float = 0.02) -> None:
    """Fail loudly if the model's box drifted from what the library was tuned to."""
    b = model.bounds()
    want = np.array([lo, hi], dtype=np.float64)
    if np.abs(b - want).max() > tol:
        msg = f"{model.name}: bounds {np.round(b, 3).tolist()} != expected {want.tolist()}"
        raise ValueError(msg)


def fit(model: Model, lo: Vec, hi: Vec) -> None:
    """Map the model's box onto the target box, axis by axis.

    For the organic props -- a bush, a tree -- the silhouette comes from a
    handful of jittered blobs, and exact extents are easier to impose after
    the fact than to author. The scale is a few percent, never a reshape.
    """
    b = model.bounds()
    want = np.array([lo, hi], dtype=np.float64)
    scale = (want[1] - want[0]) / (b[1] - b[0])
    for v in model.verts:
        v[:] = (v - b[0]) * scale + want[0]


def _obj_lines(model: Model, first_index: int, shift: Vec = (0.0, 0.0, 0.0)) -> list[str]:
    """Serialise one model's vertices and faces, one ``usemtl`` run per material.

    ``first_index`` is how many vertices precede it in the file (OBJ indices
    are 1-based and global), and ``shift`` moves it, for the catalog sheet.
    """
    order = list(dict.fromkeys(model.materials))
    lines: list[str] = []
    offsets: list[int] = []
    total = first_index
    for v in model.verts:
        offsets.append(total)
        moved = v + np.asarray(shift, dtype=np.float64)
        lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in moved)
        total += len(v)
    for material in order:
        lines.append(f"usemtl {material}")
        for f, m, off in zip(model.faces, model.materials, offsets, strict=True):
            if m != material:
                continue
            lines.extend(f"f {a + off + 1} {b + off + 1} {c + off + 1}" for a, b, c in f)
    return lines


def _mtl_lines(materials: dict[str, tuple[tuple[float, float, float], float]]) -> list[str]:
    """Serialise ``{name: (rgb, opacity)}`` as MTL blocks."""
    mtl: list[str] = []
    for material, ((r, g, b), d) in materials.items():
        mtl += [
            f"newmtl {material}",
            f"Kd {r:.3f} {g:.3f} {b:.3f}",
            f"Ka {r * 0.35:.3f} {g * 0.35:.3f} {b * 0.35:.3f}",
            "Ks 0.06 0.06 0.06",
            "Ns 12",
            "illum 2",
            f"d {d}",
            "",
        ]
    return mtl


def _palette_for(model: Model) -> dict[str, tuple[tuple[float, float, float], float]]:
    return {m: (PALETTE[m], OPACITY.get(m, 1.0)) for m in dict.fromkeys(model.materials)}


def write(model: Model, out_dir: Path) -> None:
    """Write ``<name>.obj`` and ``<name>.mtl``."""
    lines = [
        f"# {model.name}.obj  Y-up, meters, origin at ground/wall contact",
        f"mtllib {model.name}.mtl",
        *_obj_lines(model, 0),
    ]
    (out_dir / f"{model.name}.obj").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / f"{model.name}.mtl").write_text(
        "\n".join(_mtl_lines(_palette_for(model))), encoding="utf-8", newline="\n"
    )


def write_catalog(models: list[Model], drone: Path | None, out_dir: Path) -> None:
    """Write ``base_power_scene_assets.obj``: every prop in a row, one ``o`` each.

    A catalog sheet for eyeballing the library in a modelling package; nothing
    in Canopy reads it. The drone is copied in from its own file so the sheet
    stays complete without this script authoring it.
    """
    name = "base_power_scene_assets"
    lines = [f"# {name}.obj  every library prop laid out along +X", f"mtllib {name}.mtl"]
    materials: dict[str, tuple[tuple[float, float, float], float]] = {}
    x = 0.0
    total = 0
    for model in models:
        lo, hi = model.bounds()
        lines.append(f"o {model.name}")
        lines.extend(_obj_lines(model, total, (x - float(lo[0]), 0.0, 0.0)))
        total += sum(len(v) for v in model.verts)
        materials.update(_palette_for(model))
        x += float(hi[0] - lo[0]) + 1.0
    if drone is not None:
        for mat in read_mtl(drone.with_suffix(".mtl")).values():
            materials.setdefault(mat.name, (mat.diffuse, mat.opacity))
        for obj in read_obj(drone, up_axis="z").values():
            copy = Model(drone.stem)
            for group in obj.groups:
                PALETTE.setdefault(group.material, materials[group.material][0])
                copy.add(obj.vertices, group.faces.astype(np.int64), group.material)
            lines.append(f"o {drone.stem}")
            lines.extend(_obj_lines(copy, total, (x - float(obj.bounds[0][0]), 0.0, 0.0)))
            total += sum(len(v) for v in copy.verts)
    (out_dir / f"{name}.obj").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{name}.mtl").write_text(
        "\n".join(_mtl_lines(materials)), encoding="utf-8", newline="\n"
    )


# ---------------------------------------------------------------------------
# Props. Each builder fills the same bounding box the library was tuned to,
# stated in the `expect` at its end (native Y-up frame).
# ---------------------------------------------------------------------------
Builder = Callable[[], Model]
BUILDERS: dict[str, Builder] = {}


def prop(fn: Builder) -> Builder:
    """Register a builder under its function name."""
    BUILDERS[fn.__name__] = fn
    return fn


@prop
def electric_meter() -> Model:
    """Build the socket meter: a painted base, a ringed glass dome, a hub to the service above."""
    m = Model("electric_meter")
    m.box((0.30, 0.50, 0.01), (0.0, 0.26, 0.0), "steel_paint_dark")  # back plate
    m.box((0.30, 0.52, 0.11), (0.0, 0.26, 0.055), "steel_paint", bevel=0.012)
    m.box((0.012, 0.44, 0.03), (-0.15, 0.26, 0.06), "steel_paint_dark")  # hinge strip
    m.prism(0.018, 0.05, (0.15, 0.40, 0.055), "galvanized", n=8, axis="x")  # side hub
    # The socket: housing, bezel, locking ring, then the dial under its dome.
    m.prism(0.105, 0.06, (0.0, 0.30, 0.14), "steel_paint_dark", axis="z")
    m.tube(0.085, 0.112, 0.03, (0.0, 0.30, 0.185), "meter_bezel", axis="z")
    m.tube(0.10, 0.118, 0.025, (0.0, 0.30, 0.20), "meter_ring", axis="z")
    m.prism(0.085, 0.01, (0.0, 0.30, 0.20), "meter_face", axis="z")
    m.box((0.07, 0.025, 0.006), (0.0, 0.31, 0.208), "meter_lcd")
    m.box((0.06, 0.012, 0.006), (0.0, 0.28, 0.208), "label_black")
    m.dome(0.095, 0.10, (0.0, 0.30, 0.205), "meter_glass", axis="z")
    # Tags on the base, and the hub carrying the service up.
    m.box((0.05, 0.03, 0.006), (0.09, 0.08, 0.113), "label_yellow")
    m.box((0.10, 0.04, 0.006), (-0.06, 0.08, 0.113), "label_white")
    m.box((0.08, 0.008, 0.007), (-0.06, 0.08, 0.113), "label_black")
    m.prism(0.03, 0.19, (0.0, 0.615, 0.055), "galvanized", n=10)
    m.prism(0.04, 0.05, (0.0, 0.56, 0.055), "galvanized", n=10)
    expect(m, (-0.155, 0.0, -0.005), (0.175, 0.71, 0.306))
    return m


@prop
def gas_meter() -> Model:
    """Build the gas meter: a loaf body on a riser, a regulator, unions, a pipe into the wall."""
    m = Model("gas_meter")
    m.box((0.20, 0.04, 0.05), (0.0, 0.30, 0.025), "gas_meter_dark")  # wall bracket
    m.box((0.30, 0.28, 0.20), (0.0, 0.42, 0.15), "gas_meter_body", bevel=0.02)
    m.box((0.31, 0.06, 0.21), (0.0, 0.575, 0.15), "gas_meter_dark", bevel=0.015)
    # Index box and its four dials on the front.
    m.box((0.14, 0.10, 0.02), (0.0, 0.44, 0.255), "meter_face")
    for x in (-0.045, -0.015, 0.015, 0.045):
        m.prism(0.012, 0.008, (x, 0.44, 0.266), "dial_dark", n=8, axis="z")
    m.box((0.05, 0.03, 0.006), (0.09, 0.33, 0.253), "label_yellow")
    # Riser from the ground with the regulator can on it, feeding the left union.
    m.prism(0.03, 0.04, (-0.24, 0.02, 0.15), "gas_iron", n=10)
    m.prism(0.022, 0.77, (-0.24, 0.385, 0.15), "gas_iron", n=10)
    m.prism(0.058, 0.06, (-0.27, 0.56, 0.15), "gas_meter_dark", n=10, axis="x")
    m.prism(0.012, 0.05, (-0.27, 0.60, 0.15), "gas_iron", n=8)
    m.prism(0.02, 0.15, (-0.165, 0.77, 0.15), "gas_iron", n=10, axis="x")
    # Inlet and outlet: brass unions on the pipes, elbows at the top, and the
    # outlet turning into the wall.
    for x in (-0.09, 0.09):
        m.prism(0.02, 0.17, (x, 0.69, 0.15), "gas_iron", n=10)
        m.prism(0.035, 0.05, (x, 0.66, 0.15), "gas_pipe", n=8)
        m.box((0.06, 0.06, 0.06), (x, 0.77, 0.15), "gas_iron", bevel=0.015)
    m.prism(0.02, 0.17, (0.09, 0.77, 0.065), "gas_iron", n=10, axis="z")
    expect(m, (-0.302, 0.0, -0.02), (0.155, 0.80, 0.264))
    return m


@prop
def breaker_panel() -> Model:
    """Build the load centre: a hooded box with a hinged door, a handle, labels and knockouts."""
    m = Model("breaker_panel")
    m.box((0.46, 1.10, 0.01), (0.0, 0.55, 0.0), "panel_seam")
    m.box((0.46, 1.10, 0.12), (0.0, 0.55, 0.06), "panel_paint", bevel=0.01)
    m.box((0.44, 0.92, 0.006), (0.01, 0.62, 0.121), "panel_seam")  # door shadow line
    m.box((0.42, 0.90, 0.03), (0.01, 0.62, 0.135), "panel_paint", bevel=0.008)
    m.box((0.012, 0.86, 0.035), (-0.215, 0.62, 0.135), "galvanized")  # hinge
    m.box((0.02, 0.09, 0.025), (0.19, 0.62, 0.16), "galvanized")  # handle
    m.box((0.50, 0.03, 0.17), (-0.005, 1.116, 0.085), "steel_paint_dark")  # drip hood
    m.box((0.42, 0.10, 0.02), (0.01, 0.12, 0.13), "steel_paint_dark")  # dead front
    for x in (-0.10, 0.10):
        m.prism(0.025, 0.05, (x, -0.005, 0.06), "galvanized", n=10)  # knockouts
    m.box((0.16, 0.10, 0.006), (0.0, 0.95, 0.153), "label_white")
    for y in (0.965, 0.945, 0.925):
        m.box((0.12, 0.008, 0.007), (0.0, y, 0.153), "label_black")
    m.box((0.10, 0.05, 0.006), (0.0, 0.76, 0.153), "label_yellow")
    m.box((0.06, 0.04, 0.006), (0.0, 0.40, 0.153), "label_red")
    m.box((0.10, 0.03, 0.006), (0.0, 0.30, 0.153), "label_black")
    expect(m, (-0.254, -0.03, -0.005), (0.245, 1.131, 0.17))
    return m


@prop
def conduit_nipple() -> Model:
    """Build the short horizontal run between meter and panel, couplings and bushings."""
    m = Model("conduit_nipple")
    m.prism(0.028, 0.254, (0.0, 0.0, 0.05), "galvanized", n=10, axis="x")
    for sx in (-1.0, 1.0):
        m.prism(0.036, 0.035, (sx * 0.108, 0.0, 0.05), "galvanized", n=10, axis="x")
        m.prism(0.032, 0.02, (sx * 0.117, 0.0, 0.05), "label_black", n=10, axis="x")
    expect(m, (-0.127, -0.04, 0.014), (0.127, 0.04, 0.086))
    return m


@prop
def conduit_lb_riser() -> Model:
    """Build the riser up the wall, ending in an LB body that turns through it."""
    m = Model("conduit_lb_riser")
    m.prism(0.026, 1.27, (0.0, 0.683, 0.05), "conduit_paint", n=10)
    m.prism(0.034, 0.06, (0.0, 0.078, 0.05), "galvanized", n=10)
    for y in (0.45, 1.0):
        m.box((0.10, 0.03, 0.09), (0.0, y, 0.045), "galvanized", bevel=0.006)  # straps
    m.box((0.10, 0.17, 0.085), (0.0, 1.40, 0.0425), "steel_paint_dark", bevel=0.01)
    m.prism(0.04, 0.03, (0.0, 1.485, 0.05), "galvanized", n=10)
    m.box((0.07, 0.13, 0.006), (0.0, 1.40, 0.088), "conduit_paint")  # cover plate
    for y in (1.345, 1.455):
        m.prism(0.008, 0.006, (0.0, y, 0.091), "galvanized", n=8, axis="z")  # screws
    m.prism(0.026, 0.08, (0.0, 1.40, 0.01), "galvanized", n=10, axis="z")  # into the wall
    expect(m, (-0.05, 0.048, -0.03), (0.05, 1.5, 0.092))
    return m


@prop
def conduit_straight() -> Model:
    """Build the vertical drop from grade to the meter, strapped to the wall."""
    m = Model("conduit_straight")
    m.prism(0.026, 1.5, (0.0, 0.75, 0.03), "conduit_paint", n=10)
    m.prism(0.03, 0.06, (0.0, 0.75, 0.03), "galvanized", n=10)  # coupling
    m.prism(0.03, 0.04, (0.0, 0.02, 0.03), "galvanized", n=10)  # foot
    for y in (0.35, 1.15):
        m.box((0.10, 0.03, 0.058), (0.0, y, 0.029), "galvanized", bevel=0.006)
    expect(m, (-0.05, 0.0, 0.0), (0.05, 1.5, 0.059))
    return m


@prop
def base_core_battery() -> Model:
    """Build the floor-standing battery cabinet with its side disconnect.

    Its box is what ``rules.yaml`` ``battery`` records and every clearance was
    measured from, disconnect included, so the extents here are not free.
    """
    m = Model("base_core_battery")
    m.box((0.60, 0.06, 0.02), (0.0, 0.85, 0.0), "core_trim")  # wall bracket
    m.box((0.78, 0.06, 0.55), (0.0, 0.03, 0.28), "core_trim")  # plinth
    m.box((0.78, 0.90, 0.567), (0.0, 0.51, 0.2875), "core_white", bevel=0.02)
    m.box((0.75, 0.043, 0.54), (0.0, 0.9815, 0.29), "core_panel", bevel=0.015)  # cap
    m.box((0.62, 0.70, 0.016), (0.0, 0.52, 0.571), "core_panel")  # front panel
    m.box((0.40, 0.02, 0.008), (0.0, 0.80, 0.579), "core_led")
    m.box((0.12, 0.05, 0.006), (0.15, 0.25, 0.579), "label_white")
    m.box((0.09, 0.008, 0.007), (0.15, 0.25, 0.579), "label_black")
    for i in range(6):
        m.box((0.012, 0.03, 0.30), (0.392, 0.30 + 0.08 * i, 0.30), "core_vent")
    # The disconnect beside it, fed from the cabinet and dropping into the wall.
    m.box((0.13, 0.30, 0.18), (-0.469, 0.62, 0.18), "disconnect_gray", bevel=0.01)
    m.box((0.03, 0.08, 0.02), (-0.469, 0.62, 0.28), "label_red")
    m.prism(0.02, 0.03, (-0.397, 0.62, 0.18), "conduit_paint", n=10, axis="x")
    m.prism(0.02, 0.40, (-0.469, 0.27, 0.05), "conduit_paint", n=10)
    m.prism(0.02, 0.05, (-0.469, 0.10, 0.025), "conduit_paint", n=10, axis="z")
    m.box((0.08, 0.025, 0.06), (-0.469, 0.30, 0.03), "galvanized", bevel=0.005)
    expect(m, (-0.534, 0.0, -0.01), (0.396, 1.003, 0.579))
    return m


@prop
def shed() -> Model:
    """Build the garden shed: a gabled box with a braced door, a side window and a ramp."""
    m = Model("shed")
    m.box((2.54, 0.10, 3.15), (0.0, 0.05, 0.0), "shed_pad", bevel=0.01)
    m.box((2.44, 2.05, 3.05), (0.0, 1.125, 0.0), "shed_siding")
    # Gable attic as one prism, two-tone against the walls.
    gable: list[Vec] = []
    for z in (-1.525, 1.525):
        gable += [(-1.22, 2.15, z), (1.22, 2.15, z), (0.0, 2.85, z)]
    m.hull(gable, "shed_siding_dark")
    # Roof slabs: through the eave line and the ridge, overhanging all round.
    for sx in (-1.0, 1.0):
        slab: list[Vec] = []
        for z in (-1.75, 1.75):
            slab += [
                (sx * 1.434, 2.147, z),
                (0.0, 2.95, z),
                (sx * 1.434, 2.027, z),
                (0.0, 2.83, z),
            ]
        m.hull(slab, "shed_roof")
        m.box((0.05, 0.15, 3.50), (sx * 1.41, 2.10, 0.0), "shed_trim")  # fascia
    m.box((0.18, 0.06, 3.50), (0.0, 2.945, 0.0), "shed_roof_light")  # ridge cap
    for x in (-1.22, 1.22):
        for z in (-1.525, 1.525):
            m.box((0.08, 2.05, 0.08), (x, 1.125, z), "shed_trim")  # corner boards
    # Door on the +Z gable end.
    m.box((0.96, 1.95, 0.06), (0.0, 1.075, 1.525), "shed_trim")
    m.box((0.84, 1.85, 0.04), (0.0, 1.025, 1.545), "shed_door")
    for deg in (40.0, -40.0):
        m.box((0.10, 1.10, 0.02), (0.0, 1.05, 1.57), "wood_dark", m=rot("z", deg))
    for y in (0.40, 1.05, 1.70):
        m.box((0.14, 0.05, 0.02), (-0.35, y, 1.575), "galvanized")  # hinges
    m.box((0.04, 0.08, 0.03), (0.30, 1.0, 1.58), "galvanized")  # handle
    m.box((0.36, 0.26, 0.02), (0.0, 2.45, 1.53), "shed_trim")
    m.box((0.30, 0.20, 0.03), (0.0, 2.45, 1.535), "shed_vent")
    ramp: list[Vec] = []
    for x in (-0.55, 0.55):
        ramp += [(x, 0.10, 1.575), (x, 0.0, 1.575), (x, 0.0, 1.75)]
    m.hull(ramp, "shed_ramp")
    # Window on the +X side, with a cross mullion.
    m.box((0.06, 0.66, 0.66), (1.22, 1.45, -0.3), "shed_trim")
    m.box((0.03, 0.56, 0.56), (1.24, 1.45, -0.3), "shed_glass")
    m.box((0.035, 0.56, 0.04), (1.245, 1.45, -0.3), "shed_trim")
    m.box((0.035, 0.04, 0.56), (1.245, 1.45, -0.3), "shed_trim")
    expect(m, (-1.434, 0.0, -1.75), (1.434, 2.975, 1.75))
    return m


@prop
def fence_panel() -> Model:
    """Build one bay of picket fence: two capped posts, three rails, pointed pickets."""
    m = Model("fence_panel")
    rng = np.random.default_rng(11)
    for x in (-1.235, 1.235):
        m.box((0.09, 1.90, 0.09), (x, 0.95, -0.075), "wood_dark")
        cap: list[Vec] = [
            (x + sx * 0.045, 1.90, -0.075 + sz * 0.045) for sx in (-1, 1) for sz in (-1, 1)
        ]
        cap.append((x, 1.95, -0.075))
        m.hull(cap, "wood_dark")
    for y in (0.35, 1.0, 1.65):
        m.box((2.38, 0.09, 0.115), (0.0, y, -0.0775), "wood")
    pitch = 2.36 / 15
    for i in range(15):
        x = -1.18 + pitch * (i + 0.5)
        h = float(rng.uniform(1.78, 1.86))
        tone = str(rng.choice(["wood_light", "wood", "wood_gray"], p=[0.5, 0.35, 0.15]))
        picket: list[Vec] = [
            (x + sx * 0.065, y, z) for sx in (-1, 1) for y in (0.05, h) for z in (-0.02, 0.024)
        ]
        picket += [(x, h + 0.06, -0.02), (x, h + 0.06, 0.024)]
        m.hull(picket, tone)
    expect(m, (-1.28, 0.0, -0.135), (1.28, 1.95, 0.024))
    return m


@prop
def bush_small() -> Model:
    """Build the foundation shrub: a cluster of faceted lobes, lighter towards the top."""
    m = Model("bush_small")
    rng = np.random.default_rng(7)
    m.prism(0.03, 0.14, (0.01, 0.07, 0.0), "bark", n=7)
    m.lobe(0.30, (0.01, 0.34, 0.005), "leaf", rng)
    for k in range(6):
        a = 2 * math.pi * k / 6 + 0.3
        tone = "leaf_dark" if k % 2 else "leaf"
        m.lobe(0.22, (0.27 * math.cos(a), 0.26, 0.27 * math.sin(a)), tone, rng)
    m.lobe(0.20, (0.06, 0.47, 0.03), "leaf_light", rng)
    m.lobe(0.17, (-0.13, 0.44, -0.10), "leaf_light", rng)
    fit(m, (-0.489, 0.0, -0.472), (0.509, 0.658, 0.482))
    return m


@prop
def tree() -> Model:
    """Build the broadleaf tree: flared trunk, three limbs, a crown of faceted lobes."""
    m = Model("tree")
    rng = np.random.default_rng(3)
    flare: list[Vec] = _ring(7, 0.48, 0.0) + _ring(7, 0.30, 0.40, math.pi / 7)
    m.hull(flare, "bark_dark")
    m.prism(0.32, 2.7, (0.0, 1.35, 0.0), "bark", n=7, r_top=0.17)
    for k, (tilt, turn) in enumerate(((38.0, 20.0), (34.0, 150.0), (40.0, 265.0))):
        limb = rot("y", turn) @ rot("z", tilt)
        top = limb @ np.array([0.0, 0.9, 0.0])
        at = (float(top[0]), 2.5 + float(top[1]), float(top[2]))
        m.prism(0.15, 1.8, at, "bark" if k else "bark_dark", n=7, r_top=0.07, m=limb)
    m.lobe(1.5, (0.1, 4.0, 0.3), "leaf", rng, squash=0.8)
    for k in range(5):
        a = 2 * math.pi * k / 5 + 0.5
        tone = "leaf_dark" if k % 2 else "leaf"
        m.lobe(1.1, (1.45 * math.cos(a), 3.6 + 0.3 * (k % 3), 1.45 * math.sin(a)), tone, rng)
    for x, z in ((0.6, -0.2), (-0.7, 0.5), (0.2, 0.9)):
        m.lobe(0.9, (x, 4.75, z), "leaf_light", rng)
    m.lobe(0.8, (-1.2, 3.0, -0.9), "leaf_dark", rng)
    m.lobe(0.75, (1.3, 3.05, 0.8), "leaf_dark", rng)
    fit(m, (-2.605, -0.009, -2.555), (2.885, 5.484, 3.164))
    return m


@prop
def ac_condenser_unit() -> Model:
    """Build the condenser on its pad: louvred sides, a fan in a ring, line set to the wall."""
    m = Model("ac_condenser_unit")
    m.box((1.0, 0.06, 1.0), (0.0, 0.03, 0.0), "ac_pad", bevel=0.01)
    m.box((0.86, 0.72, 0.86), (0.0, 0.42, 0.0), "ac_cabinet", bevel=0.02)
    for x in (-0.40, 0.40):
        for z in (-0.40, 0.40):
            m.box((0.10, 0.72, 0.10), (x, 0.42, z), "ac_cabinet_dark")
    # Louvres: a dark coil panel with chunky slats over it, on three and a
    # half sides; the front's right-hand part is the service panel.
    for side, (cx, cz, w) in enumerate(((-0.15, 0.435, 0.46), (0.0, -0.435, 0.66))):
        m.box((w, 0.58, 0.02), (cx, 0.40, cz), "ac_grille")
        for i in range(7):
            m.box((w, 0.035, 0.03), (cx, 0.16 + 0.08 * i, cz * 1.012), "ac_cabinet")
        del side
    for cx in (-0.435, 0.435):
        m.box((0.02, 0.58, 0.66), (cx, 0.40, 0.0), "ac_grille")
        for i in range(7):
            m.box((0.03, 0.035, 0.66), (cx * 1.012, 0.16 + 0.08 * i, 0.0), "ac_cabinet")
    m.box((0.26, 0.60, 0.015), (0.28, 0.40, 0.437), "ac_cabinet_dark")
    m.box((0.12, 0.06, 0.006), (0.28, 0.55, 0.447), "label_white")
    m.box((0.09, 0.008, 0.007), (0.28, 0.55, 0.447), "label_black")
    m.box((0.08, 0.04, 0.006), (0.28, 0.30, 0.447), "label_yellow")
    # Top: lip, guard disc, fan hub and four pitched blades, and the ring.
    m.box((0.90, 0.05, 0.90), (0.0, 0.80, 0.0), "ac_cabinet_dark", bevel=0.01)
    m.prism(0.30, 0.01, (0.0, 0.83, 0.0), "ac_grille", n=16)
    m.prism(0.06, 0.05, (0.0, 0.855, 0.0), "ac_fan", n=8)
    for k in range(4):
        turn = rot("y", 90.0 * k)
        at = turn @ np.array([0.17, 0.85, 0.0])
        m.box(
            (0.20, 0.01, 0.07),
            (float(at[0]), float(at[1]), float(at[2])),
            "ac_fan",
            m=turn @ rot("x", 25.0),
        )
    m.tube(0.30, 0.36, 0.05, (0.0, 0.855, 0.0), "galvanized", n=16)
    # Refrigerant lines and the whip, out of the back toward the wall.
    m.prism(0.03, 0.07, (0.36, 0.25, -0.465), "foam_black", n=8, axis="z")
    m.prism(0.02, 0.07, (0.30, 0.18, -0.465), "copper", n=8, axis="z")
    m.prism(0.015, 0.07, (-0.30, 0.30, -0.465), "galvanized", n=8, axis="z")
    expect(m, (-0.5, 0.0, -0.5), (0.5, 0.881, 0.5))
    return m


def _window(m: Model, x: float, y: float, w: float, h: float, side: str, sign: float) -> None:
    """Add a framed pane with a sill to the wall named by ``side`` (x or z) and ``sign``."""
    wall = 7.0 if side == "x" else 5.0
    depth = 0.05

    def put(size: tuple[float, float], du: float, dy: float, thick: float, mat: str) -> None:
        # ``size`` is (width along the wall, height); ``du`` shifts along the wall.
        along, tall = size
        if side == "z":
            m.box((along, tall, thick), (x + du, y + dy, sign * (wall + thick / 2.0)), mat)
        else:
            m.box((thick, tall, along), (sign * (wall + thick / 2.0), y + dy, x + du), mat)

    put((w, h), 0.0, 0.0, 0.03, "window_glass")
    put((w + 0.16, 0.08), 0.0, h / 2.0 + 0.04, depth, "trim_white")
    put((0.08, h), -(w / 2.0 + 0.04), 0.0, depth, "trim_white")
    put((0.08, h), w / 2.0 + 0.04, 0.0, depth, "trim_white")
    put((0.05, h), 0.0, 0.0, 0.04, "trim_white")  # mullion
    put((w + 0.24, 0.08), 0.0, -(h / 2.0 + 0.04), 0.06, "trim_white")  # sill


@prop
def house_ranch() -> Model:
    """Build the authored house shell: a brick ranch under a gable, with its own openings.

    Replaces the box-with-a-roof shell from the home scenes at exactly its
    envelope (14.90 x 5.36 x 11.02 m, front on +Z), so every seed's layout is
    unchanged. Nothing of class WALL stands more than 6 cm proud of the brick:
    the wall-mounting rules read the WALL bounding box as the wall plane, so a
    porch on posts would leave every meter floating in front of the house.
    Anything that must project further is mapped to ROOF or DRIVEWAY by the
    house's ``materials`` overrides in ``index.yaml``.
    """
    m = Model("house_ranch")
    m.box((14.2, 0.16, 10.2), (0.0, 0.08, 0.0), "slab", bevel=0.01)
    m.box((14.0, 2.70, 10.0), (0.0, 1.51, 0.0), "brick")
    gable: list[Vec] = []
    for x in (-7.0, 7.0):
        gable += [(x, 2.84, -5.0), (x, 2.84, 5.0), (x, 5.34, 0.0)]
    m.hull(gable, "gable_siding")
    for x in (-6.98, 6.98):
        for z in (-4.98, 4.98):
            m.box((0.12, 2.70, 0.12), (x, 1.51, z), "trim_white")  # corner boards
    _house_roof(m)
    _house_front(m)
    _house_back(m)
    _house_ends(m)
    expect(m, (-7.45, 0.0, -5.51), (7.45, 5.36, 5.51))
    return m


def _house_roof(m: Model) -> None:
    """Add the roof: two slabs, ridge cap, barge boards, gutters, downspouts, chimney."""
    for sz in (-1.0, 1.0):
        slab: list[Vec] = []
        # The slab stops 2 cm short of the gutter's face and the barge boards'
        # outer edge, so no two faces share a plane and flicker.
        for x in (-7.43, 7.43):
            slab += [
                (x, 2.637, sz * 5.49),
                (x, 5.36, 0.0),
                (x, 2.477, sz * 5.49),
                (x, 5.20, 0.0),
            ]
        m.hull(slab, "roof_shingle")
        m.box((14.9, 0.16, 0.12), (0.0, 2.55, sz * 5.45), "gutter")
        for sx in (-1.0, 1.0):
            barge: list[Vec] = []
            for z, y in ((sz * 5.51, 2.627), (0.0, 5.36)):
                barge += [(sx * 7.45, y - 0.22, z), (sx * 7.45, y - 0.02, z)]
                barge += [(sx * 7.30, y - 0.22, z), (sx * 7.30, y - 0.02, z)]
            m.hull(barge, "barge_board")
    m.box((14.9, 0.06, 0.30), (0.0, 5.33, 0.0), "shingle_light")
    for x in (-6.9, 6.9):
        for z in (-5.08, 5.08):
            m.prism(0.05, 2.3, (x, 1.35, z), "downspout", n=8)
    m.box((0.9, 1.6, 0.7), (4.5, 4.56, -1.5), "chimney", bevel=0.02)
    m.box((1.0, 0.08, 0.8), (4.5, 5.32, -1.5), "chimney_cap")


def _house_front(m: Model) -> None:
    """Add the +Z elevation: two shuttered windows, the door under a canopy, the garage."""
    for x in (-5.3, -3.2):
        _window(m, x, 1.75, 1.4, 1.3, "z", 1.0)
        for sx in (-1.0, 1.0):
            m.box((0.32, 1.42, 0.04), (x + sx * 0.99, 1.75, 5.02), "shutter")
    m.box((1.15, 2.25, 0.05), (-1.0, 0.16 + 1.125, 5.025), "trim_white")
    m.box((0.95, 2.10, 0.06), (-1.0, 0.16 + 1.05, 5.03), "door_red")
    m.box((0.45, 0.9, 0.02), (-1.0, 1.75, 5.07), "door_glass")
    m.prism(0.04, 0.03, (-0.62, 1.15, 5.075), "door_knob", n=8, axis="z")
    m.box((1.4, 0.14, 0.5), (-1.0, 0.07, 5.25), "concrete")
    canopy: list[Vec] = []
    for x in (-1.9, -0.1):
        canopy += [(x, 2.55, 5.0), (x, 2.95, 5.0), (x, 2.45, 5.5), (x, 2.60, 5.5)]
    m.hull(canopy, "porch_roof")
    m.box((5.2, 2.4, 0.05), (3.6, 0.16 + 1.2, 5.025), "trim_white")
    m.box((5.0, 2.2, 0.05), (3.6, 0.16 + 1.1, 5.03), "garage_door")
    for k in range(3):
        m.box((5.0, 0.04, 0.06), (3.6, 0.16 + 0.55 * (k + 1), 5.03), "garage_line")
    m.box((0.9, 0.25, 0.06), (3.6, 2.15, 5.035), "door_glass")
    m.box((0.5, 0.2, 0.03), (-5.6, 2.6, 5.02), "lamp")


def _house_back(m: Model) -> None:
    """Add the -Z elevation: three windows and a patio door onto a step."""
    for x in (-4.6, 0.0, 4.6):
        _window(m, x, 1.75, 1.6, 1.3, "z", -1.0)
    m.box((1.9, 2.15, 0.03), (-2.3, 0.16 + 1.075, -5.015), "window_glass")
    m.box((2.06, 0.08, 0.05), (-2.3, 2.35, -5.025), "trim_white")
    for sx in (-1.0, 0.0, 1.0):
        m.box((0.08, 2.15, 0.05), (-2.3 + sx * 0.99, 0.16 + 1.075, -5.025), "trim_white")
    m.box((2.2, 0.14, 0.5), (-2.3, 0.07, -5.25), "concrete")


def _house_ends(m: Model) -> None:
    """Add the gable ends: two windows each and a framed attic vent."""
    for sign in (-1.0, 1.0):
        for z in (-2.2, 2.2):
            _window(m, z, 1.75, 1.2, 1.3, "x", sign)
        m.box((0.05, 0.55, 0.55), (sign * 7.02, 3.75, 0.0), "gable_vent")
        m.box((0.06, 0.65, 0.08), (sign * 7.03, 3.75, -0.28), "trim_white")
        m.box((0.06, 0.65, 0.08), (sign * 7.03, 3.75, 0.28), "trim_white")
        m.box((0.06, 0.08, 0.65), (sign * 7.03, 4.05, 0.0), "trim_white")
        m.box((0.06, 0.08, 0.65), (sign * 7.03, 3.45, 0.0), "trim_white")


def main() -> None:
    """Build every registered prop, or those named with ``--only``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "obj_export" / "assets",
    )
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(BUILDERS))
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="also write base_power_scene_assets.obj, every prop in a row (needs a full build)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    built: list[Model] = []
    for name in args.only or sorted(BUILDERS):
        model = BUILDERS[name]()
        write(model, args.out)
        built.append(model)
        lo, hi = model.bounds()
        print(
            f"{name:20s} {model.n_faces():6d} tris  "
            f"{np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}"
        )
    if args.catalog:
        drone = args.out / "canopy_scout.obj"
        write_catalog(built, drone if drone.is_file() else None, args.out)


if __name__ == "__main__":
    main()
