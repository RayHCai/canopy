"""ViewerSession.world(): per-face material colours, and the run/face-count contract.

Generating a property and building its raycasting scene costs roughly a
second, so one seed is shared across the tests that need a running session
(module-scoped fixture, matching ``tests/viz/test_viewer.py``); the payload
contract is checked separately with a hand-built manifest/geometry pair, since
it needs a face count that deliberately does not match its runs.
"""

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config
from canopy.contracts import Cls, MaterialRun, SceneGeometry, SceneManifest, SceneObject
from canopy.errors import WorldgenError
from canopy.viz.viewer import ViewerSession, _world_payload
from canopy.worldgen.generate import generate_field


@pytest.fixture(scope="module")
def session(cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> ViewerSession:
    """One viewer session, seeded for a reproducible, small(ish) property."""
    scene_dir = tmp_path_factory.mktemp("world_colors_scenes")
    return ViewerSession(cfg, drones=3, seed=1, scene_dir=scene_dir)


@pytest.fixture(scope="module")
def manifest(session: ViewerSession, tmp_path_factory: pytest.TempPathFactory) -> SceneManifest:
    """Regenerate the property ``session`` built, to read its ``materials`` runs.

    ``generate_field`` is deterministic in seed and config (see
    ``canopy.worldgen.generate``), so this reproduces ``session``'s manifest
    without reaching into the session's private state for it.
    """
    out_dir = tmp_path_factory.mktemp("world_colors_manifest")
    return generate_field(session.seed, session.config, out_dir=out_dir)


def _decode(b64: str, dtype: npt.DTypeLike) -> npt.NDArray[np.generic]:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def test_objects_with_material_runs_carry_colors_of_exactly_f_times_3_bytes(
    session: ViewerSession, manifest: SceneManifest
) -> None:
    world = session.world()
    by_id = {obj.obj_id: obj for obj in manifest.objects}
    checked_any = False
    for entry in world["objects"]:
        obj = by_id[entry["id"]]
        indices = _decode(entry["indices"], np.dtype(np.uint32))
        if obj.materials:
            checked_any = True
            assert "colors" in entry
            colors = _decode(entry["colors"], np.dtype(np.uint8))
            # F*3 uint8 bytes, one per triangle's rgb -- the same F*3 count as
            # the (uint32) index array, since both are 3 values per face.
            assert colors.size == indices.size
            first_row = tuple(int(c) for c in colors[:3])
            assert first_row == obj.materials[0].rgb
        else:
            assert "colors" not in entry
    assert checked_any, "expected at least one object with authored material runs"


def test_meter_object_shows_more_than_one_distinct_colour(session: ViewerSession) -> None:
    world = session.world()
    meters = [o for o in world["objects"] if o["cls"] == "METER"]
    assert meters, "expected an electric meter in the generated property"
    for entry in meters:
        colors = _decode(entry["colors"], np.dtype(np.uint8)).reshape(-1, 3)
        distinct = {tuple(row) for row in colors.tolist()}
        assert len(distinct) > 1


def test_world_payload_raises_when_material_runs_miscount_faces() -> None:
    vertices: npt.NDArray[np.float64] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces: npt.NDArray[np.int32] = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)  # 2 triangles
    obj = SceneObject(
        obj_id=0,
        cls=Cls.METER,
        mesh_path="unused.obj",
        color=(1, 2, 3),
        # Covers only 1 of the mesh's 2 faces: a run count that cannot come
        # out of AssetLibrary.build, only a hand-built or hand-edited manifest.
        materials=(MaterialRun(material="m", rgb=(4, 5, 6), n_faces=1),),
    )
    bad_manifest = SceneManifest(
        seed=0,
        lot_bounds=np.zeros((2, 3)),
        footprint=[],
        objects=[obj],
        home=np.zeros(3),
        gt_meter_id=0,
    )
    geometry = SceneGeometry(
        vertices=[vertices],
        faces=[faces],
        obj_tri_offset=np.array([0, 2], dtype=np.int64),
        tri_obj=np.array([0, 0], dtype=np.int32),
        tri_normal=np.zeros((2, 3)),
        tri_area=np.ones(2),
        tri_centroid=np.zeros((2, 3)),
    )
    with pytest.raises(WorldgenError):
        _world_payload(0, bad_manifest, geometry)
