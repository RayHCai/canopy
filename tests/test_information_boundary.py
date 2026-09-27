"""ADR 0016: the drone-side stages see only their sensors, never simulator ground truth.

Two checks. First, a source scan: no module under
``src/canopy/{mapping,perception,planning,site}`` may name a ground-truth
identifier, except the two modules whose whole job is to score the swarm
against the scene (:mod:`canopy.mapping.coverage`,
:mod:`canopy.planning.run`). Second, :class:`~canopy.mapping.Mapper`'s own
signatures: it takes an :class:`~canopy.contracts.Observation`, never a
manifest or geometry.
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path

from canopy.contracts import Observation
from canopy.mapping.mapper import Mapper

# Ground-truth scene types, the ids and geometry only they carry, and the
# fields that used to leak the lot and the answer key into drone-side code
# (ADR 0016). A mention inside a docstring or comment is fine: those are
# string constants (``ast.Constant``), not one of the node kinds checked here.
_FORBIDDEN: frozenset[str] = frozenset(
    {
        "SceneManifest",
        "SceneGeometry",
        "tri_ids",
        "obj_ids",
        "lot_bounds",
        "gt_meter_id",
        "tri_normal",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "canopy"

#: The two modules allowed to hold the scene: they score the swarm, not plan it.
_SCORING_MODULES = frozenset(
    {
        _SRC / "mapping" / "coverage.py",
        _SRC / "planning" / "run.py",
    }
)

#: Every other module in these drone-side stages must stay off ground truth.
_DRONE_SIDE_STAGES = ("mapping", "perception", "planning", "site")


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


def test_drone_side_source_never_names_simulator_ground_truth() -> None:
    """No drone-side module, but the two scoring ones, names a ground-truth identifier."""
    files = [
        path
        for stage in _DRONE_SIDE_STAGES
        for path in sorted((_SRC / stage).rglob("*.py"))
        if path not in _SCORING_MODULES
    ]
    assert files, "expected to find source files under the drone-side stages"
    violations = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}: {name}"
        for path in files
        for lineno, name in _forbidden_identifiers_in(path)
    ]
    assert violations == [], "ground-truth identifiers leaked into drone-side code:\n" + "\n".join(
        violations
    )


def test_scoring_modules_are_exactly_the_declared_exceptions() -> None:
    """Sanity check on the exception list itself: both scoring modules must exist."""
    for path in _SCORING_MODULES:
        assert path.is_file(), f"expected a scoring module at {path}"


def test_mapper_integrate_is_annotated_to_take_an_observation() -> None:
    """``integrate``'s ``obs`` parameter is typed as ``Observation``, never a ``Scan``."""
    signature = inspect.signature(Mapper.integrate)
    assert "obs" in signature.parameters
    hints = typing.get_type_hints(Mapper.integrate)
    assert hints["obs"] is Observation


def test_mapper_init_takes_no_manifest_or_geometry() -> None:
    """``Mapper(cfg, envelope, launch_xy)``: no scene object reaches its constructor."""
    params = set(inspect.signature(Mapper.__init__).parameters)
    assert "manifest" not in params
    assert "geometry" not in params
    assert params == {"self", "cfg", "envelope", "launch_xy"}
