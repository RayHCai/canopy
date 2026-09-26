"""Procedural property generation: seed -> SceneManifest plus OBJ meshes.

:func:`generate_field` is the entry point. What it places, and how many of
each, is data in ``assets/models/index.yaml``; see :mod:`canopy.worldgen.assets`
for the model library (:class:`AssetLibrary`, :func:`load_library`,
:func:`default_assets_dir`) and :mod:`canopy.worldgen.placement` for the
generation rules. :mod:`canopy.worldgen.objio` is the OBJ/MTL file-format
boundary (:func:`read_obj`, :func:`read_mtl`) that both the asset library and
the desktop viewer read authored art through. :func:`save_manifest` and
:func:`load_manifest` round-trip a generated property to and from disk.

Public API: :class:`AssetLibrary`, :func:`default_assets_dir`,
:func:`generate_field`, :func:`launch_pads`, :func:`load_library`, :func:`load_manifest`,
:func:`read_mtl`, :func:`read_obj`, :func:`save_manifest`.
"""

from __future__ import annotations

from canopy.worldgen.assets import AssetLibrary, default_assets_dir, load_library
from canopy.worldgen.generate import generate_field, launch_pads, load_manifest, save_manifest
from canopy.worldgen.objio import read_mtl, read_obj

__all__ = [
    "AssetLibrary",
    "default_assets_dir",
    "generate_field",
    "launch_pads",
    "load_library",
    "load_manifest",
    "read_mtl",
    "read_obj",
    "save_manifest",
]
