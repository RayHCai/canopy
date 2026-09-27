"""Where the job is: the operator's envelope, and the house inferred inside it.

The swarm is never told where the lot is (ADR 0016). A real crew knows two
things before take-off: where they put the drones down, and how far they are
cleared to fly from there. :func:`survey_envelope` turns those into the box the
map covers and the geofence holds -- configuration, not anything read from
the scene.

Inside that box sit the house, its yard, and whatever of the neighbours and
the street the envelope happens to take in. The swarm has to decide for itself
which building it came for. The crew launches from in front of it, so it is
the mapped building nearest the pads. A building is recognised the same way
the site stage traces walls (:mod:`canopy.site.outline`): a plan column is wall
when it fills a band above head height, which bushes, meters, fences and
tree trunks do not, and a connected run of such columns that spans more than
clutter does is a building.

A tree crown can fill that band too: low-poly crowns hang well below 2.8 m,
and one in the front yard sits nearer the pads than the house. Two things
tell it from a wall, and the swarm can see both. A wall stands on the
ground, so the column under the band is solid, where under a crown the drones
see open air past a thin trunk; a column whose underside is observed mostly
FREE is not wall. And a wall is a *run*: the component has to span
``house_min_wall_m``, not merely hold that many cells' worth, which a round
crown of the same area does not. The survey then reaches
``map.survey_margin_m`` past the house's plan extent -- enough for the meter,
the service run and the wall-side space a battery needs -- and frontiers and
inspection targets beyond it are left alone.

The inference is redone as the map grows rather than latched. A house first
seen as one wall grows its box as its other walls are mapped, and the survey
region grows with it; the frontiers along the known walls' ends sit inside
the margin, which is what draws the swarm round to see the rest. The mapper
only ever widens the region (:class:`~canopy.mapping.Mapper`), so a later,
different pick can add to the survey but never pull it off a house it has
already taken in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from canopy.contracts import MapState, Occ, Points

if TYPE_CHECKING:
    from canopy.config import Config, MapCfg

__all__ = ["infer_survey_bounds", "survey_envelope"]

#: 8-connectivity in plan: a wall running diagonally across the grid is one wall.
_PLAN_8 = np.ones((3, 3), dtype=bool)

#: Bottom of the "underside" checked below the wall band, metres above the
#: grid floor: clear of the ground voxel and the grass on it.
_UNDERSIDE_FLOOR_M = 0.5

#: A column whose observed underside is more than this share FREE stands on
#: air -- a crown, an eave, a branch -- not on the ground.
_MAX_FREE_UNDERSIDE = 0.5


def survey_envelope(pads: Points, cfg: Config) -> npt.NDArray[np.float64]:
    """Box the operator's flight envelope around the launch pads, shape ``(2, 3)``.

    Parameters
    ----------
    pads
        Launch pad per drone, shape ``(N, 3)``. Where the crew set the drones
        down is the one position a real survey knows before take-off.
    cfg
        Supplies ``safety.envelope_x_m`` / ``envelope_y_m``, measured from the
        pads' centroid, and ``map.height_m`` for the top.

    Returns
    -------
    numpy.ndarray
        ``[[xmin, ymin, 0], [xmax, ymax, map.height_m]]`` in world metres.
    """
    centre = np.asarray(pads, dtype=np.float64).reshape(-1, 3)[:, :2].mean(axis=0)
    x_lo, x_hi = cfg.safety.envelope_x_m
    y_lo, y_hi = cfg.safety.envelope_y_m
    return np.array(
        [
            [centre[0] + x_lo, centre[1] + y_lo, 0.0],
            [centre[0] + x_hi, centre[1] + y_hi, cfg.map.height_m],
        ],
        dtype=np.float64,
    )


def infer_survey_bounds(
    state: MapState, launch_xy: npt.NDArray[np.float64], cfg: MapCfg
) -> npt.NDArray[np.float64] | None:
    """Find the house nearest the launch point and box the survey around it.

    Parameters
    ----------
    state
        The swarm's map; only ``occ`` and the grid geometry are read.
    launch_xy
        Plan position the swarm launched from, shape ``(2,)``.
    cfg
        Supplies the wall band, fill, least wall span and the survey margin.

    Returns
    -------
    numpy.ndarray or None
        ``[[xmin, ymin], [xmax, ymax]]``: the house's plan extent grown by
        ``cfg.survey_margin_m``. ``None`` while no building has been mapped.
    """
    z0 = float(state.origin[2])
    k0 = max(int(np.ceil((cfg.house_band_z_m[0] - z0) / state.voxel)), 0)
    k1 = min(int(np.floor((cfg.house_band_z_m[1] - z0) / state.voxel)), state.occ.shape[2])
    if k1 <= k0:
        return None
    fill = np.mean(state.occ[:, :, k0:k1] == Occ.OCC, axis=2)
    wall = fill >= cfg.house_min_fill
    u0 = max(int(np.ceil((_UNDERSIDE_FLOOR_M - z0) / state.voxel)), 0)
    if u0 < k0:
        # UNKNOWN underneath (hidden behind a bush, say) is not evidence of
        # air, so only an observed FREE underside disqualifies a column.
        free_under = np.mean(state.occ[:, :, u0:k0] == Occ.FREE, axis=2)
        wall &= free_under <= _MAX_FREE_UNDERSIDE
    labels, n = ndimage.label(wall, structure=_PLAN_8)
    if n == 0:
        return None
    # Plan span of each component: the longer side of its bounding box.
    spans = np.array(
        [
            max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            for sl in ndimage.find_objects(labels)
        ]
    )
    buildings = 1 + np.flatnonzero(spans * state.voxel >= cfg.house_min_wall_m)
    if buildings.size == 0:
        return None

    cells = np.argwhere(np.isin(labels, buildings))
    centres = state.origin[None, :2] + (cells + 0.5) * state.voxel
    nearest = int(np.argmin(np.linalg.norm(centres - np.asarray(launch_xy)[None, :], axis=1)))
    house = cells[labels[tuple(cells[nearest])] == labels[cells[:, 0], cells[:, 1]]]
    lo = state.origin[:2] + house.min(axis=0) * state.voxel - cfg.survey_margin_m
    hi = state.origin[:2] + (house.max(axis=0) + 1) * state.voxel + cfg.survey_margin_m
    return np.array([lo, hi], dtype=np.float64)
