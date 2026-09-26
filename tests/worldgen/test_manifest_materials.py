"""SceneManifest.materials: the JSON round-trip and the authored OBJ's usemtl runs.

The hand-built manifests below stay off disk for the geometry (mesh files are
never read by :func:`save_manifest`/:func:`load_manifest`, only their paths
are recorded), so most of this module is fast; the one test that needs a real
generated scene -- to check the ``usemtl`` lines :mod:`canopy.worldgen.generate`
actually writes -- is the exception.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canopy.config import Config
from canopy.contracts import Cls, MaterialRun, SceneManifest, SceneObject
from canopy.worldgen.generate import generate_field, load_manifest, save_manifest
from canopy.worldgen.objio import read_obj


def _object(obj_id: int, materials: tuple[MaterialRun, ...]) -> SceneObject:
    return SceneObject(
        obj_id=obj_id,
        cls=Cls.BUSH,
        mesh_path=f"meshes/{obj_id}_bush.obj",
        color=(10, 20, 30),
        materials=materials,
    )


def _manifest(objects: list[SceneObject]) -> SceneManifest:
    return SceneManifest(
        seed=0,
        lot_bounds=np.zeros((2, 3)),
        footprint=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        objects=objects,
        home=np.zeros(3),
        gt_meter_id=objects[0].obj_id,
    )


def test_material_runs_round_trip_through_save_and_load(tmp_path: Path) -> None:
    runs = (
        MaterialRun(material="leaf", rgb=(50, 120, 40), n_faces=12),
        MaterialRun(material="bark", rgb=(90, 60, 30), n_faces=3),
    )
    with_materials = _object(0, runs)
    without_materials = _object(1, ())

    manifest = _manifest([with_materials, without_materials])
    path = save_manifest(manifest, tmp_path)
    reloaded = load_manifest(path)

    got_with, got_without = reloaded.objects
    # MaterialRun holds plain values, so the generated __eq__ compares by value.
    assert got_with.materials == runs
    for run in got_with.materials:
        assert isinstance(run.material, str)
        assert isinstance(run.rgb, tuple)
        assert all(isinstance(c, int) for c in run.rgb)
        assert isinstance(run.n_faces, int)
    assert got_without.materials == ()


def test_manifest_without_materials_key_loads_with_empty_tuple(tmp_path: Path) -> None:
    obj = _object(0, (MaterialRun(material="leaf", rgb=(1, 2, 3), n_faces=5),))
    manifest = _manifest([obj])
    path = save_manifest(manifest, tmp_path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["objects"]:
        del entry["materials"]
    path.write_text(json.dumps(doc), encoding="utf-8")

    reloaded = load_manifest(path)
    assert all(o.materials == () for o in reloaded.objects)


def test_generated_meter_obj_usemtl_runs_match_the_manifest(cfg: Config, tmp_path: Path) -> None:
    """The meter's OBJ file groups its faces by ``usemtl`` exactly as the manifest records.

    ``_write_obj`` writes world-space coordinates (already Z-up), so this must
    read the file back with ``up_axis="z"`` -- the default ``"y"`` would rotate
    it a second time.
    """
    manifest = generate_field(1, cfg, out_dir=tmp_path)
    meter = next(o for o in manifest.objects if o.cls is Cls.METER)
    assert meter.materials, "the meter model always carries authored material runs"

    text = Path(meter.mesh_path).read_text(encoding="utf-8")
    assert "usemtl" in text

    objects = read_obj(Path(meter.mesh_path), up_axis="z")
    (obj,) = objects.values()
    assert len(obj.groups) == len(meter.materials)
    for run, group in zip(meter.materials, obj.groups, strict=True):
        assert run.material == group.material
        assert run.n_faces == len(group.faces)
