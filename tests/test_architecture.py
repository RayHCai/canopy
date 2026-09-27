"""Enforce the package layering that discipline alone used to hold together.

Every cross-stage import in ``src/canopy`` should point from a later pipeline
stage to an earlier one, and go through the earlier stage's facade rather than
reaching into its submodules. Nothing at runtime stops a new import from
breaking either rule, so this test walks the whole source tree with the stdlib
``ast`` module and fails listing every violation. See
``docs/adr/0009-package-facades-and-enforced-layering.md`` for the decision
this test enforces.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
from pathlib import Path

import pytest

#: Pipeline stage order. A stage may depend on any strictly earlier stage.
_STAGES = [
    "worldgen",
    "sim",
    "perception",
    "mapping",
    "planning",
    "site",
    "viz",
    "cli",
]

#: Top-level modules with no stage of their own; importable from anywhere.
_CORE_LEAVES = {"contracts", "errors", "log", "mathutil", "config"}

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_PKG_ROOT = _SRC_ROOT / "canopy"


@dataclasses.dataclass(frozen=True, slots=True)
class _Import:
    """One resolved canopy-internal import, for reporting violations."""

    file: Path
    lineno: int
    importer: str
    target: str


def _module_name_for(path: Path) -> str:
    """Return the dotted module name a source file under ``src`` defines."""
    rel = path.relative_to(_SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_submodule(candidate: str) -> bool:
    """Whether ``candidate`` (a dotted name) names a real file under ``src``.

    Compares on-disk names case-sensitively even on a case-insensitive
    filesystem (Windows): ``canopy.mapping.Mapper`` must not match
    ``mapping/mapper.py`` just because a class, not a submodule, is named
    ``Mapper``.
    """
    rel = Path(*candidate.split("."))
    parent = _SRC_ROOT / rel.parent
    if not parent.is_dir():
        return False
    entries = {entry.name for entry in parent.iterdir()}
    return f"{rel.name}.py" in entries or (
        rel.name in entries and (parent / rel.name / "__init__.py").is_file()
    )


def _resolve_from_import(node: ast.ImportFrom, importer: str, *, is_package: bool) -> str | None:
    """Return the absolute dotted module a ``from`` import's target module is.

    ``level`` handles ``from . import x`` / ``from .. import x``. Level 1 is
    anchored at the importer's own package: the importer itself when it is a
    package's ``__init__.py``, its parent otherwise.
    """
    if node.level == 0:
        base = node.module or ""
    else:
        importer_parts = importer.split(".")
        package = importer_parts if is_package else importer_parts[:-1]
        anchor = package[: len(package) - (node.level - 1)]
        base = ".".join([*anchor, node.module] if node.module else anchor)
    return base or None


def _targets_in_file(path: Path, importer: str) -> list[_Import]:
    """Every canopy-internal import target in one source file, with line numbers."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    found: list[_Import] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                _Import(path, node.lineno, importer, alias.name)
                for alias in node.names
                if alias.name.startswith("canopy")
            )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(node, importer, is_package=is_package)
            if base is None or not base.startswith("canopy"):
                continue
            for alias in node.names:
                # ``from canopy.worldgen import placement`` names a real
                # submodule: the target is that submodule, not the package's
                # facade. ``from canopy.sim import RaySensor`` does not: the
                # target is the package itself, i.e. its facade.
                candidate = f"{base}.{alias.name}"
                target = candidate if _is_submodule(candidate) else base
                found.append(_Import(path, node.lineno, importer, target))
    return found


def _all_source_files() -> list[Path]:
    return sorted(_PKG_ROOT.rglob("*.py"))


def _own_stage(module: str) -> str | None:
    """Return the pipeline stage ``module`` belongs to, or ``None`` for a core module."""
    parts = module.split(".")
    if len(parts) == 1:
        return None
    if len(parts) == 2 and parts[1] in _CORE_LEAVES:
        return None
    return parts[1]


@functools.cache
def _all_imports() -> tuple[_Import, ...]:
    """Every canopy-internal import in ``src/canopy``, parsed once per session."""
    return tuple(
        imp
        for path in _all_source_files()
        for imp in _targets_in_file(path, _module_name_for(path))
    )


def _format(violations: list[_Import], why: str) -> str:
    lines = [
        f"{imp.file.relative_to(_SRC_ROOT)}:{imp.lineno} -> {imp.target}" for imp in violations
    ]
    return f"{why}:\n" + "\n".join(lines)


def test_core_modules_import_only_core() -> None:
    """Core modules (contracts, errors, log, mathutil, config) import no pipeline stage."""
    violations = [
        imp
        for imp in _all_imports()
        if _own_stage(imp.importer) is None and _own_stage(imp.target) is not None
    ]
    assert violations == [], _format(violations, "core module importing a stage")


def test_stage_modules_import_only_earlier_stages_or_core() -> None:
    """A stage may import core, its own package, and strictly earlier stages."""
    violations = []
    for imp in _all_imports():
        own = _own_stage(imp.importer)
        if own is None:
            continue  # covered by test_core_modules_import_only_core
        target_stage = _own_stage(imp.target)
        if target_stage is None or target_stage == own:
            continue  # core, or same package: always fine
        if _STAGES.index(target_stage) >= _STAGES.index(own):
            violations.append(imp)
    assert violations == [], _format(violations, "stage importing a same-or-later stage")


def test_cross_stage_imports_go_through_the_facade() -> None:
    """Importing a different stage package must name the package itself, not a submodule."""
    violations = []
    for imp in _all_imports():
        own = _own_stage(imp.importer)
        target_stage = _own_stage(imp.target)
        if target_stage is None or target_stage == own:
            continue  # core, or same package: not a cross-stage import
        if imp.target != f"canopy.{target_stage}":
            violations.append(imp)
    assert violations == [], _format(violations, "cross-stage import bypassing the facade")


def test_no_module_imports_its_own_facade() -> None:
    """A submodule importing its own package's facade is how import cycles start."""
    violations = []
    for imp in _all_imports():
        own = _own_stage(imp.importer)
        if own is None:
            continue
        if imp.target != f"canopy.{own}":
            continue
        if imp.importer == f"canopy.{own}":
            continue  # the facade (__init__.py) itself is exempt
        violations.append(imp)
    assert violations == [], _format(violations, "module importing its own package's facade")


#: Stages with code behind them, so their facade must declare a public API.
#: ``cli`` is excluded: it holds entry points, and nothing imports it.
_IMPLEMENTED_STAGES = [
    stage
    for stage in _STAGES
    if stage != "cli" and any(p.name != "__init__.py" for p in (_PKG_ROOT / stage).glob("*.py"))
]


@pytest.mark.parametrize("stage", _IMPLEMENTED_STAGES)
def test_facade_all_is_sorted_and_importable(stage: str) -> None:
    """Every implemented stage declares a sorted ``__all__`` whose names all resolve."""
    module = importlib.import_module(f"canopy.{stage}")
    all_names = getattr(module, "__all__", None)
    assert all_names is not None, (
        f"canopy.{stage} has submodules but its facade declares no __all__"
    )
    assert all_names == sorted(all_names), f"canopy.{stage}.__all__ is not sorted: {all_names}"
    missing = [name for name in all_names if not hasattr(module, name)]
    assert missing == [], f"canopy.{stage}.__all__ names not importable from the facade: {missing}"


#: Standard-library and third-party network clients: the network boundary itself.
_NETWORK_MODULES = {"urllib", "http", "socket", "ssl", "requests", "httpx"}


def _network_import_names(node: ast.AST) -> list[str]:
    """Top-level module names a single import statement names, if any."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        return [node.module]
    return []


def test_only_geo_imports_network_modules() -> None:
    """No module outside ``canopy.worldgen.geo`` may import an HTTP or socket library.

    That package is the one place in Canopy allowed to touch the network (see
    its module docstring); everything else reaches real-world site data only
    through the :class:`~canopy.contracts.SiteSnapshot` that package produces.
    """
    violations: list[str] = []
    for path in _all_source_files():
        module = _module_name_for(path)
        if module == "canopy.worldgen.geo" or module.startswith("canopy.worldgen.geo."):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # Narrowing to the two statement types _network_import_names
            # actually handles (rather than the bare ast.AST ast.walk yields)
            # is what gives ``node.lineno`` a non-Optional int: every ``stmt``
            # subclass carries it, but ``AST`` itself does not.
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            violations.extend(
                f"{path.relative_to(_SRC_ROOT)}:{node.lineno} -> {name}"
                for name in _network_import_names(node)
                if name.split(".")[0] in _NETWORK_MODULES
            )
    assert violations == [], "network import outside canopy.worldgen.geo:\n" + "\n".join(violations)
