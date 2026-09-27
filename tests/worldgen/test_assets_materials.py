"""AssetLibrary.build and display_rgb: material runs, palette, and the class-colour fallback.

Runs are the viewer-only per-material split of a placed model's faces (see
``contracts.MaterialRun``); every check here pins the two invariants
:meth:`~canopy.worldgen.assets.AssetLibrary.build` documents -- a mesh's runs
always cover every one of its faces, exactly once, even after ``max_edge_m``
subdivision -- and the colour ladder :meth:`~canopy.worldgen.assets.AssetLibrary.display_rgb`
climbs: authored MTL, then the library palette, then the flat class colour.
"""

from __future__ import annotations

import numpy as np
import pytest

from canopy.contracts import CLASS_COLORS, Cls
from canopy.worldgen.assets import BuiltMesh
from canopy.worldgen.index_schema import load_library
from canopy.worldgen.objio import read_mtl


def _assert_runs_cover_every_face(built: list[BuiltMesh]) -> None:
    for mesh in built:
        assert sum(run.n_faces for run in mesh.runs) == len(mesh.faces)


def test_shipped_library_has_a_nonempty_palette_with_roof_shingle() -> None:
    library = load_library()
    assert library.palette
    assert "roof_shingle" in library.palette


def test_electric_meter_build_runs_cover_faces_and_match_mtl_materials() -> None:
    library = load_library()
    spec = library.models["electric_meter"]
    mesh = spec.mesh
    assert mesh is not None, "electric_meter is authored to test the mesh: branch of display_rgb"
    mtl = read_mtl((library.root / mesh).with_suffix(".mtl"))
    extents = library.native_extents(spec)

    built = library.build(spec, pos=np.zeros(3), extents=extents, yaw=0.0, default_max_edge_m=0.2)
    _assert_runs_cover_every_face(built)

    materials = {run.material for m in built for run in m.runs}
    assert materials == set(mtl)

    rgbs = {run.rgb for m in built for run in m.runs}
    assert len(rgbs) > 1, "the meter's runs should carry more than one authored colour"


def test_display_rgb_falls_back_to_class_colour_for_an_unmapped_material() -> None:
    library = load_library()
    spec = library.models["electric_meter"]
    mesh = spec.mesh
    assert mesh is not None
    material = "not_a_real_material"
    assert material not in read_mtl((library.root / mesh).with_suffix(".mtl"))
    assert material not in library.palette

    rgb = library.display_rgb(spec, material)
    assert rgb == CLASS_COLORS[Cls.METER]
    # Same ladder display_rgb documents falling through to: the model's own
    # rgb_for its resolved class.
    assert rgb == spec.rgb_for(library.cls_for(spec, material))


def test_parts_model_roof_shingle_faces_get_the_palette_colour() -> None:
    library = load_library()
    spec = library.models["neighbour_ranch"]
    assert spec.mesh is None, "neighbour_ranch is authored to test the parts: (no MTL) path"

    built = library.build(
        spec,
        pos=np.zeros(3),
        extents=np.array([14.0, 9.5, 5.2]),
        yaw=0.0,
        default_max_edge_m=0.2,
    )
    _assert_runs_cover_every_face(built)

    roof = next(m for m in built if m.cls is Cls.ROOF)
    assert {run.material for run in roof.runs} == {"roof_shingle"}
    assert roof.runs[0].rgb == library.palette["roof_shingle"]


def test_subdivision_leaves_runs_covering_every_face_exactly() -> None:
    """``house_mass_one_story`` declares ``max_edge_m: 0.5``, well under its own extents.

    A box is 12 triangles unsubdivided; asserting well over that confirms
    subdivision actually ran, not just that the (trivially true) empty case
    passed.
    """
    library = load_library()
    spec = library.models["house_mass_one_story"]
    assert spec.max_edge_m == pytest.approx(0.5)

    built = library.build(
        spec,
        pos=np.zeros(3),
        extents=np.array([10.0, 9.0, 3.1]),
        yaw=0.3,
        default_max_edge_m=0.2,
    )
    _assert_runs_cover_every_face(built)
    assert len(built) == 1
    assert len(built[0].faces) > 12
