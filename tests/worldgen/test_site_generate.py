"""generate_field(site=...): rebuilding a real address rather than a seed-only lot."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Polygon

from canopy.config import Config
from canopy.contracts import Cls, Provenance, SceneManifest, SiteBuilding, SiteSnapshot
from canopy.worldgen.generate import generate_field, load_manifest, save_manifest


def _vertices(path: str) -> npt.NDArray[np.float64]:
    """Parse an OBJ file's ``v`` lines into an ``(N, 3)`` array of world coordinates."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in lines if line.startswith("v ")],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def built(
    cfg: Config, site_snapshot: SiteSnapshot, tmp_path_factory: pytest.TempPathFactory
) -> SceneManifest:
    """Build the fixture site once from seed 1, and share it read-only across this module."""
    out_dir = tmp_path_factory.mktemp("site_generate_default")
    return generate_field(1, cfg, out_dir=out_dir, site=site_snapshot)


def test_generate_field_is_byte_identical_across_two_builds(
    cfg: Config, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    a = generate_field(1, cfg, out_dir=tmp_path / "a", site=site_snapshot)
    b = generate_field(1, cfg, out_dir=tmp_path / "b", site=site_snapshot)
    assert len(a.objects) == len(b.objects)
    for obj_a, obj_b in zip(a.objects, b.objects, strict=True):
        assert Path(obj_a.mesh_path).read_bytes() == Path(obj_b.mesh_path).read_bytes()


def test_manifest_footprint_matches_the_fixture_houses_area_and_bounds(
    built: SceneManifest,
) -> None:
    assert Polygon(built.footprint).area == pytest.approx(126.0, abs=0.5)
    xs = [x for x, _ in built.footprint]
    ys = [y for _, y in built.footprint]
    assert (min(xs), max(xs)) == pytest.approx((-6.0, 6.0))
    assert (min(ys), max(ys)) == pytest.approx((-7.0, 7.0))


def test_lot_bounds_are_plus_or_minus_half_the_snapshot_lot(
    built: SceneManifest, site_snapshot: SiteSnapshot
) -> None:
    half_x, half_y = site_snapshot.lot_m[0] / 2.0, site_snapshot.lot_m[1] / 2.0
    assert np.allclose(built.lot_bounds[0, :2], [-half_x, -half_y])
    assert np.allclose(built.lot_bounds[1, :2], [half_x, half_y])


def test_site_id_and_north_rad_come_from_the_snapshot(
    built: SceneManifest, site_snapshot: SiteSnapshot
) -> None:
    assert built.site_id == site_snapshot.site_id
    assert built.site_id != ""
    assert built.north_rad == pytest.approx(site_snapshot.north_rad)


# ---------------------------------------------------------------------------
# Provenance: the mapped house is observed, everything sampled is not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls_", [Cls.WALL, Cls.ROOF])
def test_the_observed_houses_walls_and_roof_are_marked_observed(
    built: SceneManifest, cls_: Cls
) -> None:
    matches = [o for o in built.objects if not o.background and o.cls is cls_]
    assert matches, f"expected some surveyed-lot {cls_.name} objects"
    assert all(o.provenance is Provenance.OBSERVED for o in matches)


@pytest.mark.parametrize("cls_", [Cls.WINDOW, Cls.DOOR])
def test_openings_the_data_never_saw_are_marked_inferred(built: SceneManifest, cls_: Cls) -> None:
    matches = [o for o in built.objects if not o.background and o.cls is cls_]
    assert matches, f"expected some surveyed-lot {cls_.name} objects"
    assert all(o.provenance is Provenance.INFERRED for o in matches)


def test_the_meter_is_never_presented_as_observed(built: SceneManifest) -> None:
    """The meter is always a hypothesis on an address-built property (ADR 0015)."""
    meters = [o for o in built.objects if not o.background and o.cls is Cls.METER]
    assert len(meters) == 1
    assert meters[0].provenance in (Provenance.INFERRED, Provenance.REPAIRED)


# ---------------------------------------------------------------------------
# Everything on the surveyed lot stays inside its lot_bounds
# ---------------------------------------------------------------------------
#: Slack for the OBJ export's 6-decimal-place rounding.
_LOT_VERTEX_TOL_M = 1e-3


def test_no_surveyed_lot_vertex_falls_outside_lot_bounds(built: SceneManifest) -> None:
    lo, hi = built.lot_bounds[0, :2], built.lot_bounds[1, :2]
    for obj in built.objects:
        if obj.background:
            continue
        xy = _vertices(obj.mesh_path)[:, :2]
        assert np.all(xy >= lo - _LOT_VERTEX_TOL_M), f"{obj.cls.name} {obj.obj_id} left the lot"
        assert np.all(xy <= hi + _LOT_VERTEX_TOL_M), f"{obj.cls.name} {obj.obj_id} left the lot"


def test_background_objects_exist_and_background_walls_stand_outside_the_lot(
    built: SceneManifest,
) -> None:
    background = [o for o in built.objects if o.background]
    assert background, "expected neighbour scenery to be generated"

    lo, hi = built.lot_bounds[0, :2], built.lot_bounds[1, :2]
    walls = [o for o in background if o.cls is Cls.WALL]
    assert walls, "expected the mapped neighbours to carry WALL geometry"
    for obj in walls:
        xy = _vertices(obj.mesh_path)[:, :2]
        outside = (xy[:, 0] < lo[0]) | (xy[:, 0] > hi[0]) | (xy[:, 1] < lo[1]) | (xy[:, 1] > hi[1])
        assert np.all(outside), f"background wall {obj.obj_id} has a vertex inside the lot"


# ---------------------------------------------------------------------------
# The footprint is pinned by the address; the meter is still a per-seed draw
# ---------------------------------------------------------------------------
def test_the_footprint_is_pinned_across_seeds_while_the_meter_is_not(
    cfg: Config, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    footprints = []
    meter_positions = []
    for seed in range(1, 5):
        manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed), site=site_snapshot)
        footprints.append(manifest.footprint)
        meter = next(o for o in manifest.objects if o.obj_id == manifest.gt_meter_id)
        meter_positions.append(tuple(_vertices(meter.mesh_path).mean(axis=0)))

    assert all(fp == footprints[0] for fp in footprints[1:]), "the mapped footprint moved"
    distinct = {tuple(round(v, 6) for v in p) for p in meter_positions}
    assert len(distinct) > 1, "the meter should not land in the same place on every seed"


#: A neighbour flush against the main block's +x wall (x in [-6, 6], y in
#: [-7, 1]), still clear of the 26 x 36 m lot line: exactly the case
#: `without_party_walls` exists for.
_SEMI_DETACHED_NEIGHBOUR = SiteBuilding(
    footprint=np.array([[6.0, -7.0], [12.9, -7.0], [12.9, 1.0], [6.0, 1.0]]),
    levels=1,
    height_m=None,
    roof_shape=None,
    source="fixture:semi",
)


@pytest.mark.slow
def test_a_semidetached_partys_wall_never_takes_the_meter(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    """The shared +x wall is dropped from the house frame, so nothing mounts on it."""
    snapshot = make_site(neighbours=(_SEMI_DETACHED_NEIGHBOUR,), lot_m=(26.0, 36.0))
    for seed in range(6):
        manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed), site=snapshot)
        meter = next(o for o in manifest.objects if o.obj_id == manifest.gt_meter_id)
        assert meter.wall_normal is not None
        assert float(meter.wall_normal[0]) <= 0.5, (
            f"seed {seed}: the meter faces +x, the wall shared with the neighbour"
        )


# ---------------------------------------------------------------------------
# Mapped trees are taken as truth; an area with none falls back to drawing them
# ---------------------------------------------------------------------------
def test_a_snapshot_with_no_mapped_trees_still_draws_inferred_trees(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    snapshot = make_site(trees=np.zeros((0, 3), dtype=np.float64))
    manifest = generate_field(3, cfg, out_dir=tmp_path, site=snapshot)
    trees = [o for o in manifest.objects if o.cls is Cls.TREE and not o.background]
    assert trees, "an area with no mapped trees should still fall back to drawing some"
    assert all(o.provenance is Provenance.INFERRED for o in trees)


def test_the_default_site_places_exactly_its_one_mapped_tree(built: SceneManifest) -> None:
    trees = [o for o in built.objects if o.cls is Cls.TREE and not o.background]
    assert len(trees) == 1
    assert trees[0].provenance is Provenance.OBSERVED


# ---------------------------------------------------------------------------
# manifest.json round trip: site identity, notes and provenance survive it
# ---------------------------------------------------------------------------
def test_save_and_load_manifest_round_trips_site_identity_and_provenance(
    built: SceneManifest, tmp_path: Path
) -> None:
    path = save_manifest(built, tmp_path / "roundtrip")
    reloaded = load_manifest(path)
    assert reloaded.site_id == built.site_id
    assert reloaded.north_rad == pytest.approx(built.north_rad)
    assert reloaded.notes == built.notes
    assert [o.provenance for o in reloaded.objects] == [o.provenance for o in built.objects]


def test_save_and_load_manifest_round_trips_a_reported_note(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    """The tight-but-legal lot from test_invariants.py always reports one note."""
    snapshot = make_site(lot_m=(14.6, 36.0))
    manifest = generate_field(0, cfg, out_dir=tmp_path / "build", site=snapshot)
    assert len(manifest.notes) == 1

    reloaded = load_manifest(save_manifest(manifest, tmp_path / "roundtrip"))
    assert reloaded.notes == manifest.notes


def test_save_and_load_manifest_round_trips_a_repaired_provenance(
    cfg: Config, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    """A REPAIRED mark survives the file. Which draws get repaired is test_invariants.py's job."""
    manifest = generate_field(0, cfg, out_dir=tmp_path / "build", site=site_snapshot)
    meter = next(o for o in manifest.objects if o.obj_id == manifest.gt_meter_id)
    meter.provenance = Provenance.REPAIRED

    reloaded = load_manifest(save_manifest(manifest, tmp_path / "roundtrip"))
    reloaded_meter = next(o for o in reloaded.objects if o.obj_id == reloaded.gt_meter_id)
    assert reloaded_meter.provenance is Provenance.REPAIRED


def test_load_manifest_defaults_site_fields_when_absent(
    built: SceneManifest, tmp_path: Path
) -> None:
    """A manifest written before ADR 0015 lacks these keys entirely; loading it must not fail."""
    path = save_manifest(built, tmp_path / "full")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("site_id", "north_rad", "notes"):
        del doc[key]
    for obj_doc in doc["objects"]:
        obj_doc.pop("provenance", None)
    stripped_dir = tmp_path / "stripped"
    stripped_dir.mkdir()
    (stripped_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")

    reloaded = load_manifest(stripped_dir)
    assert reloaded.site_id == ""
    assert reloaded.north_rad == pytest.approx(np.pi / 2.0)
    assert reloaded.notes == ()
    assert all(o.provenance is Provenance.INFERRED for o in reloaded.objects)
