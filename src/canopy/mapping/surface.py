"""Which mapped surfaces the swarm has photographed well, judged from its own scans.

Coverage counts a surface only when it was seen close and near square-on
(``map.coverage_max_range_m``, ``map.coverage_max_incidence_deg``). Judging
"square-on" needs the surface's normal, and a drone is never handed the mesh's.
What it does have is a range image: a spinning lidar reports its returns on
an azimuth x elevation grid (:attr:`~canopy.contracts.Observation.grid_shape`),
so each return's neighbours are known and the local surface is the plane
through them. The normal is the cross product of the central differences
across the grid, which is what real lidar pipelines do before any meshing.

A return on a depth edge has a neighbour on a farther surface, and the
difference across the edge runs nearly along the ray. The normal that comes
out is then nearly perpendicular to the ray -- a grazing incidence -- so an
edge fails the incidence test instead of counting falsely. That is the right
way round for a mask that decides where the swarm has *finished* looking: a
missed return costs a revisit, a false one leaves a surface unphotographed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import Observation, Points

if TYPE_CHECKING:
    from canopy.config import MapCfg

__all__ = ["SurfaceTracker", "estimate_normals"]

#: Rays a central difference spans along each grid axis.
_STENCIL = 3


def estimate_normals(obs: Observation) -> tuple[Points, npt.NDArray[np.bool_]]:
    """Estimate each return's surface normal from its range-image neighbours.

    Parameters
    ----------
    obs
        One sweep. Without a :attr:`~Observation.grid_shape` there are no
        neighbours to use, and no normal is valid.

    Returns
    -------
    tuple of numpy.ndarray
        ``(normals, valid)``: unit normals, shape ``(N, 3)``, with an
        arbitrary sign, and whether each one could be estimated -- the return
        and its four grid neighbours are all hits and not collinear. The top
        and bottom elevation rows have no neighbour on one side and are never
        valid; azimuth wraps around.
    """
    n = obs.dist.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    if obs.grid_shape is None:
        return normals, valid
    n_az, n_el = obs.grid_shape
    if n_az * n_el != n or min(n_az, n_el) < _STENCIL:
        return normals, valid

    hit = np.isfinite(obs.dist)
    # Misses become NaN points, so any difference across one is NaN and fails
    # the finite check below without a separate neighbour mask per direction.
    reach = np.where(hit, obs.dist, np.nan)
    pts = (obs.origin[None, :] + obs.dirs * reach[:, None]).reshape(n_az, n_el, 3)
    d_az = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d_el = np.full_like(pts, np.nan)
    d_el[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    cross = np.cross(d_az, d_el).reshape(n, 3)
    length = np.linalg.norm(cross, axis=1)
    ok = np.isfinite(length) & (length > 0.0)
    normals[ok] = cross[ok] / length[ok, None]
    valid[ok] = True
    return normals, valid


class SurfaceTracker:
    """Marks the map voxels a photo-quality return has landed in.

    Parameters
    ----------
    cfg
        Supplies the coverage range and incidence limits.
    origin, voxel, shape
        The occupancy grid this mask is aligned with.
    """

    def __init__(
        self,
        cfg: MapCfg,
        origin: npt.NDArray[np.float64],
        voxel: float,
        shape: tuple[int, int, int],
    ) -> None:
        self._max_range_m = cfg.coverage_max_range_m
        self._min_cos = float(np.cos(np.radians(cfg.coverage_max_incidence_deg)))
        self._origin = np.asarray(origin, dtype=np.float64)
        self._voxel = voxel
        self._shape = shape
        self._seen = np.zeros(shape, dtype=np.bool_)

    @property
    def seen(self) -> npt.NDArray[np.bool_]:
        """The mask itself, shape ``(X, Y, Z)``; updated in place, never replaced."""
        return self._seen

    def integrate(self, obs: Observation) -> None:
        """Mark the voxels of this sweep's close, square-on returns.

        Parameters
        ----------
        obs
            One sweep.
        """
        normals, valid = estimate_normals(obs)
        cos_incidence = np.abs(np.einsum("ij,ij->i", obs.dirs, normals))
        good = valid & (obs.dist < self._max_range_m) & (cos_incidence > self._min_cos)
        if not np.any(good):
            return
        pts = obs.origin[None, :] + obs.dirs[good] * obs.dist[good, None]
        idx = np.floor((pts - self._origin) / self._voxel).astype(np.intp)
        inside = np.all((idx >= 0) & (idx < np.asarray(self._shape)), axis=1)
        i, j, k = idx[inside].T
        self._seen[i, j, k] = True
