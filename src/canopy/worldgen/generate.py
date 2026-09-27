"""Seed to property: the one entry point that builds a random residential field.

:func:`generate_field` is the whole of Module 1's public surface. It walks the
model library's roles in order, asks each role's placement rule where its
objects go, bakes each placed model's pose into a world-space OBJ, and returns
the :class:`~canopy.contracts.SceneManifest` that the sim, the mapper and the
viewer all read.

Everything random comes from one ``numpy.random.default_rng(seed)`` threaded
through the roles in a fixed order, so a seed reproduces a property exactly --
including the OBJ files, which are written by this module rather than by trimesh
precisely so the formatting is ours and the bytes are stable.

Background roles -- the neighbouring lots, houses and street -- are the one
exception. They draw from their own generator with a constant seed, so the
neighbourhood is identical for every property and cannot shift the surveyed
lot's random stream wherever it sits in the role order.

The design decision worth knowing before reading on: *what* gets placed and
*where* are both data. The roles, the models, the counts and the
material-to-class mapping live in ``assets/models/index.yaml``; only a new kind
of placement is code. See :mod:`canopy.worldgen.assets` and
:mod:`canopy.worldgen.placement`.

Given a :class:`~canopy.contracts.SiteSnapshot`, the same walk rebuilds a real
property: the snapshot's lot replaces the configured one, the rules place what
the map data observed where it stands, and they draw the rest as usual. Hard
rules (:mod:`canopy.worldgen.invariants`) are checked as each role is placed,
on either kind of property. See ``docs/adr/0015-address-seeded-sites.md``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import canopy.worldgen.background as _background_rules
import canopy.worldgen.equipment as _equipment_rules
import canopy.worldgen.house as _house_rules
from canopy.config import load_config
from canopy.contracts import (
    Cls,
    MaterialRun,
    Points,
    Provenance,
    SceneManifest,
    SceneObject,
    Vec3,
)
from canopy.errors import AssetError, WorldgenError
from canopy.log import get_logger
from canopy.worldgen import invariants
from canopy.worldgen import placement as rules
from canopy.worldgen.assets import AssetLibrary
from canopy.worldgen.evidence import site_evidence
from canopy.worldgen.index_schema import load_library

if TYPE_CHECKING:
    from canopy.config import Config
    from canopy.contracts import SiteSnapshot
    from canopy.worldgen.assets import BuiltMesh, RoleSpec
    from canopy.worldgen.evidence import SiteEvidence
    from canopy.worldgen.placement import HouseFrame, Placement

__all__ = ["generate_field", "launch_pads", "load_manifest", "save_manifest"]

_log = get_logger(__name__)

#: The placement rule families. A rule exists only once its module has run its
#: ``@register_rule`` decorators, and generation looks rules up by name, so
#: holding the modules here is what guarantees they are imported. Aliased
#: because a placed house is called ``house`` throughout this module.
_RULE_FAMILIES = (_background_rules, _equipment_rules, _house_rules)

#: Triangle budget from spec.md Module 1. Exceeding it is logged, not fatal:
#: a heavy scene still flies, it just costs frame time.
_TRIANGLE_BUDGET = 250_000

#: Metres from the front lot edge to the launch pad.
_HOME_INSET_M = 1.5

#: Decimal places in exported OBJ coordinates. Six is sub-millimetre at lot
#: scale and gives a fixed-width representation, which is what makes a reseeded
#: run byte-identical.
_OBJ_PRECISION = 6

#: Seed for every background role. Constant, not derived from the property
#: seed: the neighbourhood is a fixed backdrop, and varying it per seed would
#: only add frame-to-frame noise to comparisons between properties.
_BACKGROUND_SEED = 0

#: How far a window or door triangle may sit from its wall line and still be
#: in that wall. Glazing is recessed a few centimetres and a door surround
#: stands proud by as much again; half a metre takes both with room to spare
#: while still falling short of the opposite wall of any room.
_OPENING_REACH_M = 0.5

#: Two stretches of opening closer than this along a wall are one opening: the
#: pieces of one window frame touch, or overlap, rather than leave a gap.
_OPENING_MERGE_M = 0.01

#: North on a seed-only property, which has no geography: +y, the contract's
#: default for :attr:`~canopy.contracts.SceneManifest.north_rad`.
_SEED_NORTH_RAD = math.pi / 2.0

#: Clearance kept around a mapped tree's trunk by every role placed before the
#: trees are: the authored trunks are about 0.3 m across, plus a margin.
_MAPPED_TRUNK_RADIUS_M = 0.3


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def launch_pads(home: Vec3, n: int, cfg: Config) -> Points:
    """Place ``n`` launch pads on a line through ``home``, along world x.

    Spaced ``min_separation_m + margin_m`` apart rather than exactly the minimum:
    drones land in parallel, and on pads exactly one separation apart the
    shield's "may keep, never shrink the gap" rule would stall any descent that
    drifts a centimetre inward.

    It lives here, not with the mission that flies from the pads, because
    :func:`generate_field` has to keep them clear before any drone exists.

    Returns
    -------
    Points
        Shape ``(n, 3)``, on the ground (``z = 0``).
    """
    spacing = cfg.safety.min_separation_m + cfg.safety.margin_m
    pads = np.repeat(np.asarray(home, dtype=np.float64)[None, :], n, axis=0)
    pads[:, 0] += (np.arange(n) - (n - 1) / 2.0) * spacing
    pads[:, 2] = 0.0
    return pads


def generate_field(
    seed: int,
    cfg: Config | None = None,
    *,
    out_dir: Path | str | None = None,
    library: AssetLibrary | None = None,
    site: SiteSnapshot | None = None,
) -> SceneManifest:
    """Build one residential property -- random, or a real one -- and write its meshes.

    Parameters
    ----------
    seed
        Everything random on the surveyed lot derives from this; background
        scenery does not vary with it. The same seed and the same library
        produce byte-identical output. With a ``site``, the seed draws only
        what the map data did not observe.
    cfg
        Configuration. Defaults to the shipped ``config/default.yaml``.
    out_dir
        Directory for ``manifest.json`` and ``meshes/``. Defaults to
        ``out/scenes/<seed>``. Created if absent; existing mesh files for the
        same object ids are overwritten.
    library
        Model library. Defaults to the shipped ``assets/models/index.yaml``.
    site
        A real address's frozen map data, from :func:`~canopy.worldgen.fetch_site`
        or :func:`~canopy.worldgen.load_snapshot`. Its lot replaces
        ``worldgen.lot_m``, and ``out_dir`` defaults to
        ``out/scenes/site-<site_id>/<seed>`` instead.

    Returns
    -------
    SceneManifest
        Objects in placement order, so ``obj_id`` equals the index at which the
        sim will add the mesh to the Open3D raycasting scene.

    Raises
    ------
    WorldgenError
        If a role names an unknown rule or config field, if the house does not
        fit the lot, or if no electric meter was placed -- a property without
        one has no mission.
    SiteRejectedError
        If ``site`` holds a house the generator cannot rebuild, or one no
        redraw can fit a meter to.
    AssetError
        If the library or one of its mesh files is malformed.
    """
    cfg = cfg if cfg is not None else load_config()
    evidence: SiteEvidence | None = None
    if site is not None:
        evidence = site_evidence(site, cfg.worldgen.site)
        # Every rule that sizes anything from the lot reads worldgen.lot_m, so
        # the real lot replaces the configured one rather than being threaded
        # through each rule on its own.
        cfg = dataclasses.replace(cfg, worldgen=dataclasses.replace(cfg.worldgen, lot_m=site.lot_m))
    library = library if library is not None else load_library()
    rng = np.random.default_rng(seed)
    background_rng = np.random.default_rng(_BACKGROUND_SEED)

    lot_x, lot_y = cfg.worldgen.lot_m
    lot_bounds = np.array(
        [[-lot_x / 2.0, -lot_y / 2.0, 0.0], [lot_x / 2.0, lot_y / 2.0, cfg.map.height_m]],
        dtype=np.float64,
    )

    home = np.array([0.0, -lot_y / 2.0 + _HOME_INSET_M, 0.0], dtype=np.float64)
    # The mission seeds each pad's column as known free space without scanning
    # it, so nothing may actually stand there: a fence through the column lets
    # the drones take off but leaves no clear point above the pads to land from.
    launch_area = tuple(
        rules.Obstacle(xy=pad[:2].copy(), radius_m=cfg.planner.launch_clear_radius_m)
        for pad in launch_pads(home, cfg.worldgen.launch_area_pads, cfg)
    )

    house: HouseFrame | None = None
    meter: Placement | None = None
    obstacles: list[rules.Obstacle] = list(launch_area)
    if evidence is not None:
        obstacles.extend(evidence.lot_obstacles(lot_bounds, _MAPPED_TRUNK_RADIUS_M))
    placed: list[Placement] = []

    for role in library.roles:
        role_rng = background_rng if role.background else rng
        spec = library.choose(role.tag, role_rng)
        n = _resolve_count(role, cfg, role_rng, house)
        ctx = rules.PlacementContext(
            rng=role_rng,
            cfg=cfg,
            library=library,
            spec=spec,
            role=role,
            n=n,
            lot_bounds=lot_bounds,
            house=house,
            obstacles=tuple(obstacles),
            launch_area=launch_area,
            placed=tuple(p for p in placed if not p.background),
            meter=meter,
            site=evidence,
        )
        found = invariants.place(ctx)
        if role.background:
            found = [dataclasses.replace(item, background=True) for item in found]

        if rules.defines_house(role.rule):
            house = _wall_frame(found, library)
            if evidence is not None:
                # A wall shared with a neighbour is not exterior: nothing may be
                # mounted on it, planted against it or glazed.
                house = evidence.without_party_walls(house, cfg.worldgen.site.party_wall_gap_m)

        for item in found:
            if meter is None and item.spec.cls is Cls.METER:
                meter = item
            # The ground plane is not an obstacle, and neither is background:
            # a neighbour house's clearance disc can reach over the lot line,
            # and the surveyed lot must not thin out because of scenery.
            if item.spec.cls is not Cls.GROUND and not item.background:
                obstacles.append(item.obstacle())
        placed.extend(found)

    if house is None:
        msg = (
            "no role established a house frame; exactly one role must use a rule "
            f"registered with defines_house=True (rules: {rules.rule_names()})"
        )
        raise WorldgenError(msg)
    if meter is None:
        msg = (
            "no electric meter was placed, so the property has no mission; the library "
            f"needs a model with cls METER on a role (roles: "
            f"{[r.name for r in library.roles]})"
        )
        raise WorldgenError(msg)

    resolved_out = _scene_dir(out_dir, seed, site)
    objects = _bake(placed, library, cfg, resolved_out)

    notes = (
        *(evidence.notes if evidence is not None else ()),
        *(site.notes if site is not None else ()),
        *invariants.review(house, placed, lot_bounds, cfg),
    )
    gt_meter_id = next(o.obj_id for o in objects if o.cls is Cls.METER)
    manifest = SceneManifest(
        seed=seed,
        lot_bounds=lot_bounds,
        footprint=house.footprint_ccw(),
        objects=objects,
        home=home,
        gt_meter_id=gt_meter_id,
        site_id=site.site_id if site is not None else "",
        north_rad=site.north_rad if site is not None else _SEED_NORTH_RAD,
        notes=notes,
    )
    save_manifest(manifest, resolved_out)
    return manifest


def _scene_dir(out_dir: Path | str | None, seed: int, site: SiteSnapshot | None) -> Path:
    """Where a property's meshes go: the caller's choice, else one directory per seed.

    Address-built properties get a directory per site above the seeds, so two
    addresses built with the same seed never overwrite each other's meshes.
    """
    if out_dir is not None:
        return Path(out_dir)
    scenes = Path("out") / "scenes"
    return scenes / f"site-{site.site_id}" / str(seed) if site is not None else scenes / str(seed)


def _wall_frame(placed: Sequence[Placement], library: AssetLibrary) -> HouseFrame:
    """Build the house frame from wall geometry, not from bounding boxes.

    A house arrives as several placements -- one per block, plus a roof for
    each -- and only some of them carry wall faces. Each that does contributes
    one :class:`~canopy.worldgen.placement.Mass`, and their union is the
    footprint.

    Measuring the ``WALL`` class rather than the model's overall box matters for
    the authored shell in particular: its gutters and eaves overhang the brick
    by nearly half a metre, and mounting against that box leaves the meter
    floating in mid air. Height stays the model's full height, because the
    frame's ``height_m`` is what the orbit planner has to clear.

    Raises
    ------
    WorldgenError
        If no placement carries wall geometry, so there is no house to measure.
    """
    masses = []
    for item in placed:
        try:
            box = library.scaled_class_bounds(item.spec, Cls.WALL, item.size)
        except AssetError:
            continue  # a roof or an opening panel: part of the house, not its walls
        lo, hi = box[0], box[1]
        extents = np.array([hi[0] - lo[0], hi[1] - lo[1], float(item.size[2])], dtype=np.float64)
        # The wall box need not be centred on the model origin (a garage bay
        # pushes it off), so its offset has to turn with the block.
        offset = (lo[:2] + hi[:2]) / 2.0
        cos_y, sin_y = np.cos(item.yaw), np.sin(item.yaw)
        rotated = np.array(
            [cos_y * offset[0] - sin_y * offset[1], sin_y * offset[0] + cos_y * offset[1]]
        )
        plan = rules.rotated_plan(extents, item.yaw)
        masses.append(rules.Mass(centre=item.pos[:2] + rotated, size=plan))

    if not masses:
        msg = (
            "the house role placed nothing with WALL geometry, so there is no footprint "
            "to measure; check the models its rule chose"
        )
        raise WorldgenError(msg)
    # The authored shell brings its own windows; procedural massing gets them
    # from the facade_openings rule, which needs to know which it is facing.
    glazed = any(
        library.cls_for(item.spec, group.material) is Cls.WINDOW
        for item in placed
        for group in library.canonical(item.spec).groups
    )
    frame = rules.house_frame(masses)
    return dataclasses.replace(
        frame, walls=_wall_openings(frame.walls, placed, library), has_openings=glazed
    )


def _wall_openings(
    walls: Sequence[rules.WallSegment], placed: Sequence[Placement], library: AssetLibrary
) -> tuple[rules.WallSegment, ...]:
    """Record the house models' own windows and doors on the walls they are in.

    The authored shell's openings are materials in its mesh, not placements, so
    without this every wall looks bare to the rules that mount equipment on it
    and a meter can land on a window. Each triangle of an opening class goes to
    the wall nearest its centroid, and its extent along that wall is merged with
    its neighbours' into one :class:`~canopy.worldgen.placement.Opening` per
    window or door. Windows stacked on two storeys merge into one, which is
    what the rules want: equipment avoids the whole column.

    Built from the same :meth:`AssetLibrary.build` the baker uses, so the
    openings are exactly where the exported mesh puts them.
    """
    if not walls:
        return tuple(walls)
    starts = np.array([w.a for w in walls], dtype=np.float64)
    edges = np.array([w.b - w.a for w in walls], dtype=np.float64)
    lengths = np.linalg.norm(edges, axis=1)
    units = edges / lengths[:, None]

    found: list[list[rules.Opening]] = [[] for _ in walls]
    for item in placed:
        groups = library.canonical(item.spec).groups
        if not any(library.cls_for(item.spec, g.material) in rules.OPENING_CLASSES for g in groups):
            continue  # a procedural block or a roof: nothing to build
        meshes = library.build(
            item.spec, pos=item.pos, extents=item.size, yaw=item.yaw, default_max_edge_m=0.0
        )
        for mesh in meshes:
            if mesh.cls not in rules.OPENING_CLASSES:
                continue
            tris = mesh.vertices[mesh.faces]
            # Every triangle against every wall: (faces, walls, corners, xy).
            rel = tris[:, None, :, :2] - starts[None, :, None, :]
            along = np.einsum("fwvk,wk->fwv", rel, units)
            centre = np.clip(along.mean(axis=2), 0.0, lengths)
            offset = rel.mean(axis=2) - centre[..., None] * units[None]
            nearest = np.argmin(np.linalg.norm(offset, axis=2), axis=1)
            rows = np.arange(len(tris))
            reach = np.linalg.norm(offset[rows, nearest], axis=1)
            lo = along[rows, nearest].min(axis=1)
            hi = along[rows, nearest].max(axis=1)
            bottom = tris[:, :, 2].min(axis=1)
            for f in np.flatnonzero(reach <= _OPENING_REACH_M):
                found[int(nearest[f])].append(
                    rules.Opening(
                        lo_m=float(lo[f]),
                        hi_m=float(hi[f]),
                        bottom_m=float(bottom[f]),
                        cls=mesh.cls,
                    )
                )

    return tuple(
        dataclasses.replace(wall, openings=_merge_openings(pieces)) if pieces else wall
        for wall, pieces in zip(walls, found, strict=True)
    )


def _merge_openings(pieces: Sequence[rules.Opening]) -> tuple[rules.Opening, ...]:
    """Merge overlapping stretches of one wall into whole openings, sorted along it.

    A merged opening takes the class of its lowest piece, so a glazed door
    stays a door rather than becoming a window because of its glass.
    """
    merged: list[rules.Opening] = []
    for piece in sorted(pieces, key=lambda o: o.lo_m):
        if merged and piece.lo_m <= merged[-1].hi_m + _OPENING_MERGE_M:
            last = merged[-1]
            low = piece if piece.bottom_m < last.bottom_m else last
            merged[-1] = rules.Opening(
                lo_m=last.lo_m,
                hi_m=max(last.hi_m, piece.hi_m),
                bottom_m=low.bottom_m,
                cls=low.cls,
            )
        else:
            merged.append(piece)
    return tuple(merged)


def _resolve_count(
    role: RoleSpec, cfg: Config, rng: np.random.Generator, house: HouseFrame | None
) -> int:
    """Draw how many objects a role places, reading any config field it names."""
    cfg_value: Any = None
    if (name := role.count.cfg_field) is not None:
        if not hasattr(cfg.worldgen, name):
            msg = (
                f"role {role.name!r} count reads worldgen.{name}, which does not exist; "
                f"known fields are {sorted(vars(type(cfg.worldgen))['__slots__'])}"
            )
            raise WorldgenError(msg)
        cfg_value = getattr(cfg.worldgen, name)
    perimeter = house.perimeter_m if house is not None else 0.0
    return max(0, role.count.resolve(rng, cfg_value, perimeter))


def _bake(
    placed: list[Placement], library: AssetLibrary, cfg: Config, out_dir: Path
) -> list[SceneObject]:
    """Build every placed model's world-space mesh and write one OBJ per class.

    One placement can yield several objects: the house shell's materials map to
    wall, roof, window, door, garage door and slab, and each becomes its own
    :class:`~canopy.contracts.SceneObject` so that sensing returns a class
    directly from the geometry id.
    """
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    objects: list[SceneObject] = []
    triangles = 0
    background_triangles = 0
    for item in placed:
        built = library.build(
            item.spec,
            pos=item.pos,
            extents=item.size,
            yaw=item.yaw,
            default_max_edge_m=cfg.worldgen.max_edge_m,
        )
        for mesh in built:
            obj_id = len(objects)
            name = f"{obj_id}_{mesh.cls.name.lower()}.obj"
            _write_obj(mesh_dir / name, mesh, asset_id=item.spec.asset_id)
            # The triangle budget exists for the reveal renderer, which never
            # touches background scenery, so background triangles do not count
            # toward it -- they are logged separately instead.
            if item.background:
                background_triangles += len(mesh.faces)
            else:
                triangles += len(mesh.faces)
            objects.append(
                SceneObject(
                    obj_id=obj_id,
                    cls=mesh.cls,
                    mesh_path=str((mesh_dir / name).resolve()),
                    color=mesh.color,
                    # Only wall-mounted objects carry a normal, and only the
                    # meter's is ever read; the rest are recorded because the
                    # contract has a slot for them.
                    wall_normal=item.wall_normal,
                    asset_id=item.spec.asset_id,
                    background=item.background,
                    materials=mesh.runs,
                    provenance=item.provenance,
                )
            )

    if triangles > _TRIANGLE_BUDGET:
        _log.warning(
            "scene has %d surveyed-lot triangles, over the %d budget; consider raising "
            "max_edge_m on the heaviest models",
            triangles,
            _TRIANGLE_BUDGET,
        )
    else:
        _log.info("generated %d objects, %d surveyed-lot triangles", len(objects), triangles)
    if background_triangles:
        _log.info("plus %d background triangles across the neighbouring lots", background_triangles)
    return objects


def _write_obj(path: Path, mesh: BuiltMesh, asset_id: str) -> None:
    """Write one world-space OBJ.

    Hand-rolled rather than delegated to trimesh so the byte layout is ours:
    fixed-precision coordinates and no library version banner are what let a
    reseeded run produce identical files, which is the determinism guarantee in
    spec.md Module 1.

    A ``usemtl`` line precedes each of :attr:`~canopy.worldgen.assets.BuiltMesh.runs`
    that names a material, since ``mesh.faces`` is already ordered to match --
    this is how the viewer recovers each face's authored material straight from
    the file it already reads. The sim's own OBJ parser only looks at ``v`` and
    ``f`` lines (:func:`canopy.sim.scene.load_geometry`), so this is invisible
    to sensing.
    """
    lines = [f"# canopy {mesh.cls.name} from asset {asset_id}", f"o {path.stem}"]
    lines.extend(
        f"v {x:.{_OBJ_PRECISION}f} {y:.{_OBJ_PRECISION}f} {z:.{_OBJ_PRECISION}f}"
        for x, y, z in mesh.vertices
    )
    # Runs that under-cover the faces would silently drop triangles from the
    # file; fail here, as the viewer does, rather than ship a holed mesh.
    covered = sum(run.n_faces for run in mesh.runs)
    if covered != len(mesh.faces):
        msg = (
            f"asset {asset_id!r} ({mesh.cls.name}): material runs cover {covered} faces "
            f"but the mesh has {len(mesh.faces)}"
        )
        raise WorldgenError(msg)
    # OBJ face indices are 1-based.
    offset = 0
    for run in mesh.runs:
        if run.material:
            lines.append(f"usemtl {run.material}")
        chunk = mesh.faces[offset : offset + run.n_faces]
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in chunk)
        offset += run.n_faces
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------
def save_manifest(manifest: SceneManifest, out_dir: Path | str) -> Path:
    """Write ``manifest.json`` beside the meshes.

    Mesh paths are stored relative to the manifest so a scene directory can be
    moved or copied between machines; :func:`load_manifest` resolves them back
    to absolute paths.

    Returns
    -------
    Path
        The file written.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"

    doc = {
        "seed": manifest.seed,
        "lot_bounds": manifest.lot_bounds.tolist(),
        "footprint": [[float(x), float(y)] for x, y in manifest.footprint],
        "home": manifest.home.tolist(),
        "gt_meter_id": manifest.gt_meter_id,
        "site_id": manifest.site_id,
        "north_rad": manifest.north_rad,
        "notes": list(manifest.notes),
        "objects": [
            {
                "obj_id": obj.obj_id,
                # The integer value, not the name: Cls values are the stable
                # part of the contract.
                "cls": int(obj.cls),
                "mesh_path": _relative(obj.mesh_path, directory),
                "color": list(obj.color),
                "wall_normal": (
                    None if obj.wall_normal is None else [float(v) for v in obj.wall_normal]
                ),
                "asset_id": obj.asset_id,
                "background": obj.background,
                "materials": [[run.material, list(run.rgb), run.n_faces] for run in obj.materials],
                "provenance": str(obj.provenance),
            }
            for obj in manifest.objects
        ],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _relative(mesh_path: str, directory: Path) -> str:
    """``mesh_path`` relative to the scene directory, with forward slashes."""
    try:
        rel = Path(mesh_path).resolve().relative_to(directory.resolve())
    except ValueError:
        # A mesh outside the scene directory stays absolute rather than
        # sprouting a chain of `..` that would not survive a move anyway.
        return str(Path(mesh_path))
    return rel.as_posix()


def load_manifest(path: Path | str) -> SceneManifest:
    """Read a manifest written by :func:`save_manifest`.

    Parameters
    ----------
    path
        The ``manifest.json`` file, or the scene directory holding it.

    Returns
    -------
    SceneManifest
        With every ``mesh_path`` resolved to an absolute path.

    Raises
    ------
    WorldgenError
        If the file is unreadable or missing a required field.
    """
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    try:
        doc = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read manifest {resolved}: {exc}"
        raise WorldgenError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{resolved} is not valid JSON: {exc}"
        raise WorldgenError(msg) from exc

    try:
        objects = [
            SceneObject(
                obj_id=int(o["obj_id"]),
                cls=Cls(int(o["cls"])),
                mesh_path=str((resolved.parent / o["mesh_path"]).resolve()),
                color=(int(o["color"][0]), int(o["color"][1]), int(o["color"][2])),
                wall_normal=(
                    None
                    if o.get("wall_normal") is None
                    else np.asarray(o["wall_normal"], dtype=np.float64)
                ),
                asset_id=str(o.get("asset_id", "")),
                background=bool(o.get("background", False)),
                materials=tuple(
                    MaterialRun(
                        material=str(m[0]),
                        rgb=(int(m[1][0]), int(m[1][1]), int(m[1][2])),
                        n_faces=int(m[2]),
                    )
                    for m in o.get("materials", [])
                ),
                provenance=Provenance(o.get("provenance", Provenance.INFERRED)),
            )
            for o in doc["objects"]
        ]
        return SceneManifest(
            seed=int(doc["seed"]),
            lot_bounds=np.asarray(doc["lot_bounds"], dtype=np.float64),
            footprint=[(float(x), float(y)) for x, y in doc["footprint"]],
            objects=objects,
            home=np.asarray(doc["home"], dtype=np.float64),
            gt_meter_id=int(doc["gt_meter_id"]),
            site_id=str(doc.get("site_id", "")),
            north_rad=float(doc.get("north_rad", _SEED_NORTH_RAD)),
            notes=tuple(str(n) for n in doc.get("notes", [])),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        msg = f"{resolved} is missing or malformed: {exc}"
        raise WorldgenError(msg) from exc
