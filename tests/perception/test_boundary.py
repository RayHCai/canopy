"""The information-boundary test: perception must never see simulator ground truth.

:class:`~canopy.contracts.Observation` is what a drone's ranger senses --
range and colour, never an object or triangle identity. The simulator knows
more than that (which object and triangle a ray hit, an object's true colour,
the ground-truth meter id, and the scene types and loaders that carry that
truth end to end), and none of it is passed to :mod:`canopy.perception`.
Nothing at import time or at runtime enforces that on its own, so this test
walks every perception source file with the stdlib ``ast`` module and fails,
listing file and line, the moment one of those names appears as an
identifier. A mention inside a docstring or a comment is fine: those are
string constants (``ast.Constant``), not one of the node kinds checked here.
"""

from __future__ import annotations

import ast
import dataclasses
import typing
from pathlib import Path

import numpy as np

from canopy.contracts import Observation, Scan
from canopy.perception import ObjectDetector

# Per-ray ground truth: which object and triangle a ray actually hit. This is
# exactly the label perception exists to infer from range and colour alone.
_RAY_IDENTITY = ("obj_ids", "tri_ids")

# Per-triangle ground-truth geometry, kept for the coverage metric and for
# rendering; no pipeline stage that only senses should need it.
_TRIANGLE_GEOMETRY = ("tri_obj", "obj_tri_offset", "tri_exterior", "tri_seen")

# An object's true appearance and the true-colour lookup table. Perception
# must classify from sensed, noisy colour via its own HSV rules, never by
# matching a hit against the answer key.
_OBJECT_APPEARANCE = ("obj_color", "CLASS_COLORS")

# The evaluation-only ground-truth meter id, and the scene types that carry
# ground truth (ids, true colour, exteriority) from generation through to the
# sensor.
_SCENE_TRUTH = ("gt_meter_id", "Scan", "SceneManifest", "SceneObject", "SceneGeometry")

# Loaders that hand back the ground-truth scene types above.
_LOADERS = ("load_manifest", "load_geometry")

#: Every identifier perception may never name; see the grouped comments above.
_FORBIDDEN: frozenset[str] = frozenset(
    _RAY_IDENTITY + _TRIANGLE_GEOMETRY + _OBJECT_APPEARANCE + _SCENE_TRUTH + _LOADERS
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PERCEPTION_ROOT = _REPO_ROOT / "src" / "canopy" / "perception"


def _forbidden_identifiers_in(path: Path) -> list[tuple[int, str]]:
    """Every forbidden Name/Attribute/alias/arg identifier in one file, with its line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.arg) and node.arg in _FORBIDDEN:
            found.append((node.lineno, node.arg))
        elif isinstance(node, ast.alias):
            if node.name in _FORBIDDEN:
                found.append((node.lineno, node.name))
            if node.asname is not None and node.asname in _FORBIDDEN:
                found.append((node.lineno, node.asname))
    return found


def test_perception_source_never_names_simulator_ground_truth() -> None:
    """No file under src/canopy/perception references a forbidden ground-truth identifier."""
    violations = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}: {name}"
        for path in sorted(_PERCEPTION_ROOT.rglob("*.py"))
        for lineno, name in _forbidden_identifiers_in(path)
    ]
    assert violations == [], "ground-truth identifiers leaked into perception:\n" + "\n".join(
        violations
    )


def test_observation_has_no_object_or_triangle_id_fields() -> None:
    """Observation, unlike Scan, has no id field a classifier could read instead of inferring."""
    names = {f.name for f in dataclasses.fields(Observation)}
    assert "obj_ids" not in names
    assert "tri_ids" not in names


def test_object_detector_integrate_is_annotated_to_take_an_observation() -> None:
    """integrate()'s obs parameter is typed as Observation, not the ground-truth-bearing Scan."""
    hints = typing.get_type_hints(ObjectDetector.integrate)
    assert hints["obs"] is Observation


def test_scan_observation_strips_ids_and_shares_the_sensed_arrays() -> None:
    """A hand-built Scan's observation() drops obj_ids/tri_ids and shares dist/dirs/rgb (is)."""
    scan = Scan(
        drone_id=0,
        t=0.0,
        origin=np.zeros(3),
        dirs=np.array([[1.0, 0.0, 0.0]]),
        dist=np.array([2.0]),
        obj_ids=np.array([3], dtype=np.int32),
        tri_ids=np.array([7], dtype=np.int32),
        rgb=np.array([[10, 20, 30]], dtype=np.uint8),
    )

    obs = scan.observation()

    assert isinstance(obs, Observation)
    assert not hasattr(obs, "obj_ids")
    assert not hasattr(obs, "tri_ids")
    assert obs.dist is scan.dist
    assert obs.dirs is scan.dirs
    assert obs.rgb is scan.rgb
