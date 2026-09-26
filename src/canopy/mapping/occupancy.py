"""Occupancy grid: the planner's view of the world.

Every scan is fused into one shared voxel grid spanning the surveyed lot
(``manifest.lot_bounds``) from the ground to ``cfg.height_m``. Anything a ray
hits or passes through outside that box is dropped: the swarm's job is the lot,
not the neighbours, so a grid that stopped at the fence is what keeps
:class:`SceneObject.background` hits from ever reaching the planner.

Carving free space from every ray would dominate the per-scan cost for no
planning benefit -- a corridor of FREE voxels a few rays apart looks the same
to the frontier extractor as one from every ray -- so only every
``carve_ray_stride``-th ray carves. Every hit, strided or not, still marks OCC:
misses are common and cheap to skip, but a stray unstrided ray hitting a wall
that the strided carve never touched must not be lost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import MapState, Occ, Scan, Vec3

if TYPE_CHECKING:
    from canopy.config import MapCfg

__all__ = ["integrate_occupancy", "mark_free_box", "new_map_state"]


def new_map_state(lot_bounds: npt.NDArray[np.float64], cfg: MapCfg, n_triangles: int) -> MapState:
    """Build an empty :class:`MapState` over the surveyed lot.

    Parameters
    ----------
    lot_bounds
        ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` of the surveyed lot; only
        ``x``/``y`` are used, ground truth ``zmin``/``zmax`` are ignored in
        favour of ``0`` to ``cfg.height_m`` (the swarm never maps below the
        ground plane).
    cfg
        Map configuration; ``voxel_m`` sets the grid resolution.
    n_triangles
        Total triangle count of the scene geometry, i.e. ``SceneGeometry.n_triangles``.

    Returns
    -------
    MapState
        All voxels UNKNOWN, no triangle seen.
    """
    bounds = np.asarray(lot_bounds, dtype=np.float64)
    origin = np.array([bounds[0, 0], bounds[0, 1], 0.0], dtype=np.float64)
    extent = np.array([bounds[1, 0] - bounds[0, 0], bounds[1, 1] - bounds[0, 1], cfg.height_m])
    shape = tuple(int(n) for n in np.round(extent / cfg.voxel_m))
    return MapState(
        occ=np.full(shape, Occ.UNKNOWN, dtype=np.uint8),
        origin=origin,
        voxel=cfg.voxel_m,
        tri_seen=np.zeros(n_triangles, dtype=np.bool_),
    )


def _world_to_index(points: npt.NDArray[np.float64], state: MapState) -> npt.NDArray[np.intp]:
    """Floor ``points`` to voxel indices; caller filters out-of-grid rows first."""
    return np.floor((points - state.origin) / state.voxel).astype(np.intp)


def _in_grid(idx: npt.NDArray[np.intp], shape: tuple[int, int, int]) -> npt.NDArray[np.bool_]:
    """Whether each row of ``idx`` addresses a real voxel."""
    lo = np.all(idx >= 0, axis=1)
    hi = np.all(idx < np.asarray(shape), axis=1)
    return np.asarray(lo & hi, dtype=np.bool_)


def integrate_occupancy(state: MapState, scan: Scan, cfg: MapCfg, max_range_m: float) -> bool:
    """Fuse one scan's occupancy into ``state.occ`` in place.

    Parameters
    ----------
    state
        Shared map, mutated in place.
    scan
        One 360-degree ray cast.
    cfg
        Map configuration; supplies the carve stride, step and stop-short.
    max_range_m
        Sensor's maximum range (``cfg.sensor.max_range_m``); misses carve out
        to this range rather than to infinity.

    Returns
    -------
    bool
        Whether any voxel changed value, so callers (frontier extraction) can
        skip work when a scan added nothing new.
    """
    shape = state.occ.shape
    # Tracked from the writes themselves: copying and comparing the whole grid
    # per scan cost as much as the hit marking. A voxel changes exactly when
    # the carve finds it UNKNOWN or a hit finds it anything but OCC.
    changed = False

    # -- carve free space along a stride of rays --------------------------
    #
    # Building one (rays, steps, 3) array of sample points and then flooring
    # and masking it, as a first cut of this function did, spends most of its
    # time moving that stacked array through memory. Keeping x/y/z as three
    # separate (rays, steps) arrays -- and float32, since a voxel is 25 cm and
    # single precision is accurate to well under a millimetre at lot scale --
    # is the difference between ~20 ms and ~5 ms for one 10,800-ray scan.
    # Going further, each axis is computed in place in one reused buffer from
    # a contiguous direction column, and the three indices are folded into
    # one flat index before masking. That halves it again: the time was in
    # temporaries and three separate masked gathers, not arithmetic.
    strided_dist = scan.dist[:: cfg.carve_ray_stride]
    strided_dirs = scan.dirs[:: cfg.carve_ray_stride].astype(np.float32)
    reach = (np.minimum(strided_dist, max_range_m) - cfg.carve_stop_short_m).astype(np.float32)
    n_steps = np.floor(np.clip(reach, 0.0, None) / cfg.carve_step_m).astype(np.int64)
    if n_steps.size and n_steps.max() > 0:
        max_steps = int(n_steps.max())
        # Step index 0 .. max_steps for every strided ray, then mask each ray
        # to its own reach: a ragged loop-free way to vectorize rays of
        # different length.
        offsets = np.arange(max_steps + 1, dtype=np.float32) * np.float32(cfg.carve_step_m)
        keep = offsets[None, :] <= reach[:, None]

        grid_origin = state.origin.astype(np.float32)
        ray_origin = scan.origin.astype(np.float32)
        voxel = np.float32(state.voxel)
        dir_cols = np.ascontiguousarray(strided_dirs.T)
        scaled = np.empty(keep.shape, dtype=np.float32)
        flat = np.zeros(keep.shape, dtype=np.intp)
        for a in range(3):
            # (origin + dir * offset - grid_origin) / voxel, same operations in
            # the same order as the out-of-place form, so the same voxels.
            np.multiply(dir_cols[a][:, None], offsets[None, :], out=scaled)
            scaled += ray_origin[a]
            scaled -= grid_origin[a]
            scaled /= voxel
            # floor(q) lies in [0, n) exactly when q does, and for q >= 0 the
            # cast's truncation is the floor; rows it gets wrong are masked.
            keep &= scaled >= 0
            keep &= scaled < shape[a]
            flat *= shape[a]
            flat += scaled.astype(np.intp)
        lin = flat[keep]
        # Never downgrade OCC to FREE: only touch voxels still UNKNOWN.
        free_mask = np.take(state.occ, lin) == Occ.UNKNOWN
        np.put(state.occ, lin[free_mask], Occ.FREE)
        changed = bool(np.any(free_mask))

    # -- mark every finite hit OCC, not just the strided rays --------------
    hit = np.isfinite(scan.dist)
    if np.any(hit):
        hit_pts = scan.origin[None, :] + scan.dirs[hit] * scan.dist[hit, None]
        idx = _world_to_index(hit_pts, state)
        keep_hit = _in_grid(idx, shape)
        hi, hj, hk = idx[keep_hit].T
        changed |= bool(np.any(state.occ[hi, hj, hk] != Occ.OCC))
        state.occ[hi, hj, hk] = Occ.OCC

    return changed


def mark_free_box(state: MapState, lo: Vec3, hi: Vec3) -> None:
    """Force UNKNOWN voxels inside a world box to FREE.

    This is the operator-surveyed launch column: before takeoff the mission
    seeds a small box around the pad as known-safe, because nothing has flown
    yet to observe it and the shield refuses to fly through UNKNOWN space.
    Already-OCC voxels are left alone -- a surveyed box is a promise about
    empty air, not licence to overwrite a mapped obstacle.

    Parameters
    ----------
    state
        Shared map, mutated in place.
    lo, hi
        Opposite corners of the box, in world metres.
    """
    lo_a, hi_a = np.minimum(lo, hi), np.maximum(lo, hi)
    lo_idx = np.floor((lo_a - state.origin) / state.voxel).astype(np.intp)
    hi_idx = np.ceil((hi_a - state.origin) / state.voxel).astype(np.intp)
    lo_idx = np.clip(lo_idx, 0, np.asarray(state.occ.shape))
    hi_idx = np.clip(hi_idx, 0, np.asarray(state.occ.shape))
    sl = tuple(slice(int(lo_idx[a]), int(hi_idx[a])) for a in range(3))
    region = state.occ[sl]
    region[region == Occ.UNKNOWN] = Occ.FREE
