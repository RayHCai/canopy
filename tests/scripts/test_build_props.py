"""scripts/build_props.py: every registered prop builds cleanly against the shipped library.

The script is a repo-root tool, not a package under ``src/canopy``, so it is
loaded here with ``importlib`` rather than a normal import -- ``BUILDERS`` and
``Model`` are its runtime, dynamically-typed API as far as this test is
concerned, hence the ``Any`` throughout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from canopy.worldgen.index_schema import load_library
from canopy.worldgen.objio import read_mtl, read_obj

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_build_props() -> Any:
    """Import ``scripts/build_props.py`` as a module, without running its ``main()``."""
    path = _REPO_ROOT / "scripts" / "build_props.py"
    spec = importlib.util.spec_from_file_location("canopy_build_props", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses' postponed-annotation resolution (its own @dataclass on
    # Model) looks the defining module up in sys.modules by name; a module
    # exec'd without being registered there first fails with an obscure
    # AttributeError deep in the stdlib.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_props = _load_build_props()


@pytest.mark.parametrize("name", sorted(_build_props.BUILDERS))
def test_every_builder_produces_a_nonempty_model(name: str) -> None:
    """Each builder ends with its own ``expect``/``fit`` box check; a drift raises there."""
    model = _build_props.BUILDERS[name]()
    assert model.n_faces() > 0


def _mesh_model_ids() -> list[str]:
    library = load_library()
    return sorted(asset_id for asset_id, spec in library.models.items() if spec.mesh is not None)


@pytest.mark.parametrize("asset_id", _mesh_model_ids())
def test_mesh_model_mtl_names_every_material_the_obj_uses(asset_id: str) -> None:
    library = load_library()
    spec = library.models[asset_id]
    mesh = spec.mesh
    assert mesh is not None

    mtl_path = (library.root / mesh).with_suffix(".mtl")
    assert mtl_path.is_file(), f"{asset_id}: no {mtl_path.name} beside {mesh}"

    objects = read_obj(library.root / mesh, up_axis=spec.up_axis)
    used = {group.material for obj in objects.values() for group in obj.groups}
    named = set(read_mtl(mtl_path))
    missing = used - named
    assert not missing, f"{asset_id}: {mtl_path.name} does not name {sorted(missing)}"
