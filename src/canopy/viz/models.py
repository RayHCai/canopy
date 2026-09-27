"""The authored meshes the viewer page draws: the drone and the battery.

The drones are drawn from the authored ``canopy_scout`` model in the asset
library, and each suggested battery site from ``base_core_battery``, both read
here with the same OBJ reader worldgen uses and handed to the page as plain
arrays. The page ships only three.js core, not its loaders, and cannot see
``assets/`` from the web directory pywebview serves -- and keeping the file
format on the Python side means one OBJ reader, not two.

Nothing here touches a session: a model is a pure function of its file, which
is why both loaders are cached for the life of the process.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np

from canopy.contracts import Points
from canopy.worldgen import default_assets_dir, read_mtl, read_obj

__all__ = ["BATTERY_MODEL", "DRONE_MODEL", "battery_model", "drone_model"]

#: The drone model, relative to the asset root (the directory holding
#: ``models/`` and ``obj_export/``). Its MTL sits beside it with the same stem.
DRONE_MODEL = Path("obj_export/assets/canopy_scout.obj")

#: The battery model drawn at each suggested site, relative to the asset root.
BATTERY_MODEL = Path("obj_export/assets/base_core_battery.obj")


@functools.cache
def drone_model(path: Path | None = None) -> dict[str, Any]:
    """Load the drone mesh the page draws, in the body frame.

    The body frame is the one :class:`~canopy.contracts.DroneState` yaw is
    measured in: +X forward at yaw 0, +Z up, origin at the landing-gear contact
    point, metres. The authored scout faces its native +Z, which
    :func:`~canopy.worldgen.objio.read_obj` turns into world -Y, so it is turned
    a further +90 degrees about Z here to put the camera on +X.

    Cached: the file does not change under a running viewer, and the page asks
    once per load.

    Parameters
    ----------
    path
        OBJ to read. Defaults to :data:`DRONE_MODEL` under the asset root. The
        MTL is the file of the same stem beside it.

    Returns
    -------
    dict
        ``vertices``: flat ``[x, y, z, ...]`` in the body frame. ``groups``: per
        material its ``material`` name, ``color`` ``[r, g, b]`` (sRGB, from
        ``Kd``), ``opacity`` and flat triangle ``indices`` into ``vertices``.

    Raises
    ------
    AssetError
        If the OBJ or MTL is missing or malformed.
    """
    obj_path = default_assets_dir().parent / DRONE_MODEL if path is None else path
    return _front_on_x(obj_path)


@functools.cache
def battery_model(path: Path | None = None) -> dict[str, Any]:
    """Load the battery mesh the page draws at each suggested site, in the site frame.

    The site frame is the one a :class:`~canopy.contracts.SiteCandidate`
    implies: +X out of the wall along its normal, +Z up, origin on the wall at
    ground level. The authored battery faces its native +Z with its back on
    the wall, so the drone's +90 degree turn puts its front on +X too. It is
    then slid along the wall until its bounding box -- which takes in a
    side-mounted disconnect -- is centred on the site, because that box is
    what ``rules.yaml``'s ``battery`` describes and every clearance was
    measured from.

    Parameters
    ----------
    path
        OBJ to read. Defaults to :data:`BATTERY_MODEL` under the asset root.

    Returns
    -------
    dict
        As :func:`drone_model`.

    Raises
    ------
    AssetError
        If the OBJ or MTL is missing or malformed.
    """
    obj_path = default_assets_dir().parent / BATTERY_MODEL if path is None else path
    return _front_on_x(obj_path, centre_across=True)


def _front_on_x(obj_path: Path, *, centre_across: bool = False) -> dict[str, Any]:
    """Read an authored model facing native +Z and turn it to face +X.

    With ``centre_across``, also centre its bounding box on ``y = 0``.
    """
    objects = read_obj(obj_path)
    materials = read_mtl(obj_path.with_suffix(".mtl"))

    # Every group of every object shares one vertex array, so a multi-object
    # file still reaches the page as a single mesh.
    verts: list[Points] = []
    groups: list[dict[str, Any]] = []
    offset = 0
    for obj in objects.values():
        verts.append(obj.vertices)
        for group in obj.groups:
            mat = materials.get(group.material)
            groups.append(
                {
                    "material": group.material,
                    "color": list(mat.diffuse) if mat else [0.5, 0.5, 0.5],
                    "opacity": mat.opacity if mat else 1.0,
                    "indices": (group.faces + offset).ravel().tolist(),
                }
            )
        offset += len(obj.vertices)

    world = np.concatenate(verts, axis=0)
    # +90 degrees about Z: (x, y) -> (-y, x), taking the authored front (-Y) to +X.
    body = np.stack([-world[:, 1], world[:, 0], world[:, 2]], axis=1)
    if centre_across:
        body[:, 1] -= (body[:, 1].min() + body[:, 1].max()) / 2.0
    return {"vertices": body.ravel().tolist(), "groups": groups}
