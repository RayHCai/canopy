"""generate_field: seed to property, including background neighbour scenery."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import Cls, SceneManifest
from canopy.sim import load_geometry
from canopy.worldgen import launch_pads
from canopy.worldgen.assets import load_library
from canopy.worldgen.generate import generate_field, load_manifest, save_manifest

_SEEDS = (1, 7, 42)
#: Cells in the neighbour ``lots`` grid in ``assets/models/index.yaml``.
_NEIGHBOUR_LOTS = 12
#: Sidewalk, street and sidewalk between the surveyed lot and the row opposite.
_STREET_GAP_M = 10.5


def _bbox_xy(vertices_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return the combined ``(min, max)`` xy bounding box across a set of OBJ files."""
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for path in vertices_paths:
        verts = [
            [float(v) for v in line.split()[1:4]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
        arr = np.asarray(verts, dtype=np.float64)
        lo = np.minimum(lo, arr[:, :2].min(axis=0))
        hi = np.maximum(hi, arr[:, :2].max(axis=0))
    return lo, hi


@pytest.mark.parametrize("seed", _SEEDS)
def test_background_objects_lie_outside_the_surveyed_lot(
    seed: int, cfg: Config, tmp_path: Path
) -> None:
    manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed))
    lot_x, lot_y = cfg.worldgen.lot_m
    half_x, half_y = lot_x / 2.0, lot_y / 2.0

    background = [o for o in manifest.objects if o.background]
    foreground = [o for o in manifest.objects if not o.background]
    assert background, "expected neighbour scenery to be generated"
    assert foreground, "expected the surveyed lot's own objects"

    tol = 1e-6
    for obj in background:
        lo, hi = _bbox_xy([Path(obj.mesh_path)])
        outside = (
            lo[0] >= half_x - tol
            or hi[0] <= -half_x + tol
            or lo[1] >= half_y - tol
            or hi[1] <= -half_y + tol
        )
        assert outside, f"background object {obj.obj_id} overlaps the surveyed lot"

    for obj in foreground:
        lo, hi = _bbox_xy([Path(obj.mesh_path)])
        assert lo[0] >= -half_x - tol
        assert hi[0] <= half_x + tol
        assert lo[1] >= -half_y - tol
        assert hi[1] <= half_y + tol


@pytest.mark.parametrize("seed", _SEEDS)
def test_neighbour_houses_clear_the_surveyed_lot_by_min_gap(
    seed: int, cfg: Config, tmp_path: Path
) -> None:
    manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed))
    lot_x, lot_y = cfg.worldgen.lot_m
    half_x, half_y = lot_x / 2.0, lot_y / 2.0
    min_gap_m = 4.0

    houses = [o for o in manifest.objects if o.background and o.cls is Cls.WALL]
    assert len(houses) == _NEIGHBOUR_LOTS
    for obj in houses:
        lo, hi = _bbox_xy([Path(obj.mesh_path)])
        clears = (
            lo[0] >= half_x + min_gap_m - 1e-6
            or hi[0] <= -half_x - min_gap_m + 1e-6
            or lo[1] >= half_y + min_gap_m - 1e-6
            # Across the street, beyond both sidewalks.
            or hi[1] <= -half_y - _STREET_GAP_M + 1e-6
        )
        assert clears, f"neighbour house {obj.obj_id} is closer than {min_gap_m} m to the lot"


def test_neighbour_houses_face_their_street(cfg: Config, tmp_path: Path) -> None:
    """Each front door is on the side of its house nearest that lot's street."""
    manifest = generate_field(1, cfg, out_dir=tmp_path)
    lot_y = cfg.worldgen.lot_m[1]
    street_y = -lot_y / 2.0 - _STREET_GAP_M / 2.0

    walls = [o for o in manifest.objects if o.background and o.cls is Cls.WALL]
    doors = [o for o in manifest.objects if o.background and o.cls is Cls.DOOR]
    assert len(doors) == len(walls) == _NEIGHBOUR_LOTS
    for wall, door in zip(walls, doors, strict=True):
        wall_lo, wall_hi = _bbox_xy([Path(wall.mesh_path)])
        door_lo, door_hi = _bbox_xy([Path(door.mesh_path)])
        door_y = (door_lo[1] + door_hi[1]) / 2.0
        wall_y = (wall_lo[1] + wall_hi[1]) / 2.0
        if wall_lo[1] >= -lot_y / 2.0:
            # Same side of the street as the surveyed lot, the door faces -y.
            # The back row is the exception: it faces its own street beyond
            # the scenery, so it is back to back with the surveyed lot.
            faces_minus_y = wall_hi[1] <= lot_y / 2.0
        else:
            # Across the street: the door looks back across it, at +y.
            assert wall_hi[1] < street_y
            faces_minus_y = False
        assert (door_y < wall_y) == faces_minus_y, f"house {wall.obj_id} faces away"


def _faces(path: Path) -> int:
    """Count the triangles in one exported OBJ."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("f "))


def test_background_is_identical_for_every_seed(cfg: Config, tmp_path: Path) -> None:
    a = generate_field(1, cfg, out_dir=tmp_path / "a")
    b = generate_field(42, cfg, out_dir=tmp_path / "b")

    bg_a = [o for o in a.objects if o.background]
    bg_b = [o for o in b.objects if o.background]
    assert [(o.cls, o.asset_id) for o in bg_a] == [(o.cls, o.asset_id) for o in bg_b]
    for obj_a, obj_b in zip(bg_a, bg_b, strict=True):
        # Line 2 names the file after its obj_id, which legitimately differs
        # when the surveyed lots hold different numbers of objects.
        lines_a = Path(obj_a.mesh_path).read_text(encoding="utf-8").splitlines()
        lines_b = Path(obj_b.mesh_path).read_text(encoding="utf-8").splitlines()
        assert lines_a[:1] + lines_a[2:] == lines_b[:1] + lines_b[2:]


def test_background_does_not_perturb_the_surveyed_lot(cfg: Config, tmp_path: Path) -> None:
    library = load_library()
    bare = dataclasses.replace(library, roles=tuple(r for r in library.roles if not r.background))

    full = generate_field(7, cfg, out_dir=tmp_path / "full", library=library)
    alone = generate_field(7, cfg, out_dir=tmp_path / "alone", library=bare)

    surveyed = [o for o in full.objects if not o.background]
    assert len(surveyed) == len(alone.objects)
    for with_bg, without in zip(surveyed, alone.objects, strict=True):
        assert (with_bg.obj_id, with_bg.asset_id) == (without.obj_id, without.asset_id)
        assert Path(with_bg.mesh_path).read_bytes() == Path(without.mesh_path).read_bytes()


def test_neighbour_houses_have_openings_but_stay_cheap(cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(1, cfg, out_dir=tmp_path)

    houses = [
        o
        for o in manifest.objects
        if o.background and o.asset_id.startswith("neighbour_") and o.cls is not Cls.GROUND
    ]
    for cls in (Cls.WALL, Cls.ROOF, Cls.WINDOW, Cls.DOOR):
        assert sum(o.cls is cls for o in houses) == _NEIGHBOUR_LOTS, cls.name
    # Primitives only, never subdivided: a few hundred triangles a house.
    assert sum(_faces(Path(o.mesh_path)) for o in houses) <= 400 * _NEIGHBOUR_LOTS


def test_street_runs_the_length_of_the_neighbourhood(cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(1, cfg, out_dir=tmp_path)
    lot_x, lot_y = cfg.worldgen.lot_m

    strips = [
        o
        for o in manifest.objects
        if o.asset_id in {"street_asphalt", "sidewalk_concrete", "street_line"}
    ]
    assert {o.asset_id for o in strips} == {"street_asphalt", "sidewalk_concrete", "street_line"}
    for obj in strips:
        assert obj.background
        lo, hi = _bbox_xy([Path(obj.mesh_path)])
        assert hi[1] <= -lot_y / 2.0 + 1e-6, f"strip {obj.asset_id} intrudes on the lot"
        assert lo[0] == pytest.approx(-2.5 * lot_x)
        assert hi[0] == pytest.approx(2.5 * lot_x)


def test_street_lines_sit_on_the_street_centreline(cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(1, cfg, out_dir=tmp_path)

    street = next(o for o in manifest.objects if o.asset_id == "street_asphalt")
    lines = [o for o in manifest.objects if o.asset_id == "street_line"]
    assert len(lines) == 2
    street_lo, street_hi = _bbox_xy([Path(street.mesh_path)])
    centre_y = (street_lo[1] + street_hi[1]) / 2.0
    for line in lines:
        assert line.color != street.color
        lo, hi = _bbox_xy([Path(line.mesh_path)])
        assert abs((lo[1] + hi[1]) / 2.0 - centre_y) < 0.25
        # Lifted clear of the asphalt so the two never z-fight.
        zs = {
            float(row.split()[3])
            for row in Path(line.mesh_path).read_text(encoding="utf-8").splitlines()
            if row.startswith("v ")
        }
        assert min(zs) > 0.0


def test_manifest_background_round_trips(cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(1, cfg, out_dir=tmp_path / "scene")
    path = save_manifest(manifest, tmp_path / "scene")
    reloaded = load_manifest(path)

    flags = [o.background for o in manifest.objects]
    reloaded_flags = [o.background for o in reloaded.objects]
    assert flags == reloaded_flags
    assert any(flags)
    assert not all(flags)


def test_manifest_without_background_key_loads_as_false(cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(1, cfg, out_dir=tmp_path / "scene")
    path = save_manifest(manifest, tmp_path / "scene")

    doc = json.loads(path.read_text(encoding="utf-8"))
    for obj in doc["objects"]:
        del obj["background"]
    path.write_text(json.dumps(doc), encoding="utf-8")

    reloaded = load_manifest(path)
    assert all(o.background is False for o in reloaded.objects)


def test_generation_is_deterministic(cfg: Config, tmp_path: Path) -> None:
    """A reseeded run must reproduce byte-identical OBJ files, background included."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = generate_field(7, cfg, out_dir=out_a)
    manifest_b = generate_field(7, cfg, out_dir=out_b)

    assert len(manifest_a.objects) == len(manifest_b.objects)
    for obj_a, obj_b in zip(manifest_a.objects, manifest_b.objects, strict=True):
        assert obj_a.background == obj_b.background
        assert Path(obj_a.mesh_path).read_bytes() == Path(obj_b.mesh_path).read_bytes()


# ---------------------------------------------------------------------------
# Openings on procedural houses
# ---------------------------------------------------------------------------
#: Seed 1 draws the authored shell; seeds 2 and 3 draw procedural massing.
_AUTHORED_SEED = 1
_MASSED_SEEDS = (2, 3)
_OPENING_IDS = frozenset({"window_pane", "entry_door", "garage_door_panel"})
_MOUNTED = frozenset({Cls.METER, Cls.GAS_METER, Cls.PANEL, Cls.CONDUIT})


@pytest.fixture(scope="module")
def houses(cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> dict[int, SceneManifest]:
    """One authored-shell property and two procedural ones, generated once."""
    root = tmp_path_factory.mktemp("houses")
    return {
        seed: generate_field(seed, cfg, out_dir=root / str(seed))
        for seed in (_AUTHORED_SEED, *_MASSED_SEEDS)
    }


def _house_model(manifest: SceneManifest) -> str:
    return next(o.asset_id for o in manifest.objects if o.cls is Cls.WALL and not o.background)


def test_procedural_houses_get_a_front_door_and_windows(
    houses: dict[int, SceneManifest],
) -> None:
    for seed in _MASSED_SEEDS:
        manifest = houses[seed]
        assert _house_model(manifest).startswith("house_mass"), f"seed {seed} changed style"
        doors = [o for o in manifest.objects if o.asset_id == "entry_door"]
        windows = [o for o in manifest.objects if o.asset_id == "window_pane"]
        assert len(doors) == 1
        assert all(o.cls is Cls.DOOR for o in doors)
        assert len(windows) >= 4
        assert all(o.cls is Cls.WINDOW for o in windows)
        # The front door faces the street, which is -y.
        normal = doors[0].wall_normal
        assert normal is not None
        np.testing.assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-9)


def test_authored_house_gets_no_extra_openings(houses: dict[int, SceneManifest]) -> None:
    manifest = houses[_AUTHORED_SEED]
    assert _house_model(manifest) == "house_brick_two_story"
    assert not [o for o in manifest.objects if o.asset_id in _OPENING_IDS]
    # It still has windows: its own.
    assert any(o.cls is Cls.WINDOW for o in manifest.objects if not o.background)


def test_openings_stay_clear_of_wall_mounted_equipment(houses: dict[int, SceneManifest]) -> None:
    for seed in _MASSED_SEEDS:
        manifest = houses[seed]
        openings = [o for o in manifest.objects if o.asset_id in _OPENING_IDS]
        mounted = [o for o in manifest.objects if o.cls in _MOUNTED]
        assert mounted, f"seed {seed} has no service equipment to test against"
        for opening in openings:
            assert opening.wall_normal is not None
            normal = opening.wall_normal[:2]
            along = np.array([-normal[1], normal[0]])
            o_lo, o_hi = _bbox_xy([Path(opening.mesh_path)])
            o_mid = (o_lo + o_hi) / 2.0
            o_span = sorted((float(o_lo @ along), float(o_hi @ along)))
            for item in mounted:
                m_lo, m_hi = _bbox_xy([Path(item.mesh_path)])
                if abs(float(((m_lo + m_hi) / 2.0 - o_mid) @ normal)) > 0.5:
                    continue  # on another wall
                m_span = sorted((float(m_lo @ along), float(m_hi @ along)))
                overlap = min(o_span[1], m_span[1]) - max(o_span[0], m_span[0])
                assert overlap <= 0.0, (
                    f"seed {seed}: {opening.asset_id} {opening.obj_id} overlaps "
                    f"{item.cls.name} {item.obj_id} by {overlap:.2f} m along the wall"
                )


def _bbox_3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(min, max)`` xyz bounding box of one OBJ file."""
    verts = np.asarray(
        [
            [float(v) for v in line.split()[1:4]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )
    return verts.min(axis=0), verts.max(axis=0)


@pytest.mark.parametrize("seed", range(2, 12))
def test_procedural_roofs_sit_on_their_walls(seed: int, cfg: Config, tmp_path: Path) -> None:
    manifest = generate_field(seed, cfg, out_dir=tmp_path)
    blocks = [o for o in manifest.objects if o.asset_id.startswith(("house_mass", "house_wing"))]
    roofs = [o for o in manifest.objects if o.cls is Cls.ROOF and not o.background]
    if not blocks:
        pytest.skip(f"seed {seed} drew the authored shell")
    assert len(roofs) == len(blocks), "every block carries exactly one roof"

    block_boxes = [_bbox_3d(Path(b.mesh_path)) for b in blocks]
    for roof in roofs:
        r_lo, r_hi = _bbox_3d(Path(roof.mesh_path))
        # The roof's eaves rest on the top of exactly the block it caps, which
        # it covers in plan with the eave overhang to spare.
        capped = [
            (lo, hi)
            for lo, hi in block_boxes
            if np.all(r_lo[:2] <= lo[:2]) and np.all(r_hi[:2] >= hi[:2])
        ]
        assert capped, f"seed {seed}: roof {roof.obj_id} covers no block in plan"
        tops = [float(hi[2]) for _, hi in capped]
        assert min(abs(float(r_lo[2]) - t) for t in tops) < 1e-6, (
            f"seed {seed}: roof {roof.obj_id} starts at z={r_lo[2]:.3f} m, "
            f"but the walls under it end at {tops}"
        )


def test_openings_leave_the_rest_of_the_lot_unchanged(cfg: Config, tmp_path: Path) -> None:
    library = load_library()
    bare = dataclasses.replace(
        library, roles=tuple(r for r in library.roles if r.rule != "facade_openings")
    )
    seed = _MASSED_SEEDS[0]
    full = generate_field(seed, cfg, out_dir=tmp_path / "full", library=library)
    alone = generate_field(seed, cfg, out_dir=tmp_path / "alone", library=bare)

    kept = [o for o in full.objects if o.asset_id not in _OPENING_IDS and not o.background]
    before = [o for o in alone.objects if not o.background]
    assert len(kept) == len(before)
    for with_openings, without in zip(kept, before, strict=True):
        assert with_openings.asset_id == without.asset_id
        assert Path(with_openings.mesh_path).read_bytes() == Path(without.mesh_path).read_bytes()


#: Seeds that, before the launch area was reserved, put something inside a pad
#: column: a closed front fence 1 cm from the pads, one straight through them,
#: and a tree crown 0.6 m from a pad.
_CROWDED_LAUNCH_SEEDS = (126647939, 1054, 1120)


@pytest.mark.slow
@pytest.mark.parametrize("seed", _CROWDED_LAUNCH_SEEDS)
def test_launch_columns_are_clear(seed: int, cfg: Config, tmp_path: Path) -> None:
    """Nothing on the lot stands within ``launch_clear_radius_m`` of any pad.

    The mission seeds these columns as free without scanning them, and lands
    down them; something standing there strands the swarm in the air.
    """
    manifest = generate_field(seed, cfg, out_dir=tmp_path)
    geometry = load_geometry(manifest)
    pads = launch_pads(manifest.home, cfg.worldgen.launch_area_pads, cfg)
    radius = cfg.planner.launch_clear_radius_m
    top = cfg.planner.takeoff_altitude_m + radius
    # Barycentric samples on each triangle: a long fence panel can cross a
    # column with every vertex well outside it.
    steps = 8
    bary = (
        np.array(
            [(i, j, steps - i - j) for i in range(steps + 1) for j in range(steps + 1 - i)],
            dtype=np.float64,
        )
        / steps
    )
    for obj, vertices, faces in zip(
        manifest.objects, geometry.vertices, geometry.faces, strict=True
    ):
        if obj.background or obj.cls is Cls.GROUND:
            continue
        points = np.einsum("bk,fkd->fbd", bary, vertices[faces]).reshape(-1, 3)
        points = points[(points[:, 2] > 0.0) & (points[:, 2] <= top)]
        if len(points) == 0:
            continue
        gap = np.linalg.norm(points[:, None, :2] - pads[None, :, :2], axis=-1).min()
        assert gap >= radius - 1e-6, (
            f"seed {seed}: {obj.cls.name} object {obj.obj_id} ({obj.asset_id}) is "
            f"{gap:.2f} m from a launch pad, inside the {radius} m launch column"
        )
