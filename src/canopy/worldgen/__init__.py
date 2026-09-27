"""Property generation: seed (and optionally a real address) -> SceneManifest plus OBJ meshes.

:func:`generate_field` is the entry point. What it places, and how many of
each, is data in ``assets/models/index.yaml``; see :mod:`canopy.worldgen.assets`
for the model library (:class:`AssetLibrary`, :func:`load_library`,
:func:`default_assets_dir`) and :mod:`canopy.worldgen.placement` for the
generation rules. :mod:`canopy.worldgen.objio` is the OBJ/MTL file-format
boundary (:func:`read_obj`, :func:`read_mtl`) that both the asset library and
the desktop viewer read authored art through. :func:`save_manifest` and
:func:`load_manifest` round-trip a generated property to and from disk.

A property can also be rebuilt from a real street address.
:func:`suggest_addresses` and :func:`fetch_site` are the one step that touches
the network: they turn an address into a :class:`~canopy.contracts.SiteSnapshot`,
which :func:`save_snapshot` and :func:`load_snapshot` keep under
:func:`sites_root`, and which ``generate_field(seed, site=snapshot)`` then
builds from offline.

Public API: :class:`AssetLibrary`, :func:`default_assets_dir`, :func:`fetch_site`,
:func:`generate_field`, :func:`launch_pads`, :func:`load_library`, :func:`load_manifest`,
:func:`load_snapshot`, :func:`read_mtl`, :func:`read_obj`, :func:`save_manifest`,
:func:`save_snapshot`, :func:`sites_root`, :func:`suggest_addresses`.
"""

from __future__ import annotations

from canopy.worldgen.assets import AssetLibrary, default_assets_dir
from canopy.worldgen.generate import generate_field, launch_pads, load_manifest, save_manifest
from canopy.worldgen.geo.build import fetch_site, suggest_addresses
from canopy.worldgen.index_schema import load_library
from canopy.worldgen.objio import read_mtl, read_obj
from canopy.worldgen.snapshot import load_snapshot, save_snapshot, sites_root

__all__ = [
    "AssetLibrary",
    "default_assets_dir",
    "fetch_site",
    "generate_field",
    "launch_pads",
    "load_library",
    "load_manifest",
    "load_snapshot",
    "read_mtl",
    "read_obj",
    "save_manifest",
    "save_snapshot",
    "sites_root",
    "suggest_addresses",
]
