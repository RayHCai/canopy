"""End to end: a swarm mapping a generated property finds its meter, conduit and bushes.

This is the one test that runs the whole chain -- worldgen, the coloured
ray sensor, the mission and the mapper's detector -- on a real property.
Ground truth appears here only to score the result, never on its way to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from canopy.config import Config
from canopy.contracts import Cls, DiscoveredObject, SceneGeometry, SceneManifest, SceneObject, Vec3
from canopy.planning import MissionRun
from canopy.sim import load_geometry
from canopy.worldgen import generate_field


@pytest.mark.slow
def test_swarm_finds_the_meter_conduit_and_bushes(cfg: Config, tmp_path: Path) -> None:
    """After a full mission on seed 42 the meter is boxed within 10 cm, with conduit and bushes."""
    manifest = generate_field(42, cfg, out_dir=tmp_path)
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 3, cfg, seed=42)
    run.run(cfg.sim.timeout_s)

    found = list(run.mapper.state.discovered.values())
    meters = [o for o in found if o.cls is Cls.METER]
    assert len(meters) == 1
    true_meter = geometry.vertices[manifest.gt_meter_id]
    true_centre = (true_meter.min(axis=0) + true_meter.max(axis=0)) / 2.0
    np.testing.assert_allclose(meters[0].box.center, true_centre, atol=0.1)
    assert any(o.cls is Cls.CONDUIT for o in found)
    assert sum(o.cls is Cls.BUSH for o in found) >= 5


# ---------------------------------------------------------------------------
# DOOR, GARAGE_DOOR, WINDOW, GAS_METER, AC_UNIT and PANEL against ground truth.
#
# Ground truth here only scores the swarm's detections; it never feeds them
# (ADR 0011). A manifest object is not always one physical instance: an
# authored house draws every window pane as one mesh, so instances are
# recovered from the mesh's own face connectivity, not by object id.
# ---------------------------------------------------------------------------
_SCORED_CLASSES = (Cls.DOOR, Cls.GARAGE_DOOR, Cls.WINDOW, Cls.GAS_METER, Cls.AC_UNIT, Cls.PANEL)
#: A discovered box and a true instance closer than this, centre to centre,
#: are the same object. Comfortably under the gap between any two real
#: instances of these classes (windows are spaced >= 1.3 m apart edge to edge).
_MATCH_DIST_M = 1.0


def _instance_centres(geometry: SceneGeometry, obj: SceneObject) -> list[Vec3]:
    """Centroid of each disjoint mesh region inside one scene object.

    Faces are joined by shared vertices: an authored quad shares none with a
    neighbouring one, so this recovers one centre per physical window, door or
    unit even when worldgen drew them all as a single mesh.
    """
    vertices = geometry.vertices[obj.obj_id]
    faces = geometry.faces[obj.obj_id]
    if faces.size == 0:
        return [(vertices.min(axis=0) + vertices.max(axis=0)) / 2.0]
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(vertices.shape[0], vertices.shape[0])
    )
    n_components, labels = connected_components(graph, directed=False)
    centres = []
    for label in range(n_components):
        pts = vertices[labels == label]
        centres.append((pts.min(axis=0) + pts.max(axis=0)) / 2.0)
    return centres


def _ground_truth(manifest: SceneManifest, geometry: SceneGeometry, cls: Cls) -> list[Vec3]:
    """Every real, on-lot instance of ``cls``, as one centre point each."""
    return [
        centre
        for obj in manifest.objects
        if obj.cls is cls and not obj.background
        for centre in _instance_centres(geometry, obj)
    ]


def _nearest(point: Vec3, others: list[Vec3]) -> float:
    """Distance from ``point`` to the closest of ``others``, or infinity if there are none."""
    if not others:
        return float("inf")
    return float(min(np.linalg.norm(point - o) for o in others))


@pytest.mark.slow
def test_new_classes_against_ground_truth(cfg: Config, tmp_path: Path) -> None:
    """Over three seeds, none of the six new classes reports a box with no real instance nearby.

    Recall is reported, not asserted, per class and seed: a class the swarm
    never orbits close enough to see (an upper-storey window, say) can have low
    recall for reasons outside the detector's colour and shape rules, and
    loosening those rules to compensate would just trade misses for false
    positives elsewhere.
    """
    report: dict[tuple[int, str], tuple[int, int, int]] = {}
    for seed in (42, 7, 123):
        manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed))
        geometry = load_geometry(manifest)
        run = MissionRun(manifest, geometry, 3, cfg, seed=seed)
        run.run(cfg.sim.timeout_s)
        found = list(run.mapper.state.discovered.values())

        for cls in _SCORED_CLASSES:
            truth = _ground_truth(manifest, geometry, cls)
            detections: list[DiscoveredObject] = [o for o in found if o.cls is cls]
            false_positives = [
                d for d in detections if _nearest(d.box.center, truth) > _MATCH_DIST_M
            ]
            assert not false_positives, (
                f"seed {seed} {cls.name}: false positive(s) at "
                f"{[d.box.center.tolist() for d in false_positives]}, "
                f"nearest true instance(s) at {[t.tolist() for t in truth]}"
            )
            hits = sum(
                1 for t in truth if _nearest(t, [d.box.center for d in detections]) <= _MATCH_DIST_M
            )
            report[(seed, cls.name)] = (hits, len(truth), len(detections))

    lines = [
        f"seed {seed} {name}: {hits}/{n_truth} true instances found "
        f"({n_found} reported, 0 false positives)"
        for (seed, name), (hits, n_truth, n_found) in sorted(report.items())
    ]
    print("\n" + "\n".join(lines))  # the per-class, per-seed report; run with `-s` to see it
