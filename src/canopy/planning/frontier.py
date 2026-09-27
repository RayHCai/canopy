"""Frontier extraction: where the swarm should explore next.

A frontier is a cluster of FREE voxels touching the unknown, reachable by some
live drone and worth a viewpoint. Everything here is array ops over the whole
map grid -- the FRONTIER state runs once per replan (1 Hz), and the map can be
200x176x48 voxels, so a per-voxel Python loop would blow the tick budget.

Both kinds of target are kept inside the geofence and, once the mapper has
found the house, inside ``MapState.survey_bounds``: the envelope takes in the
neighbours and the street, and a frontier there is real but not the job.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.spatial import cKDTree

from canopy.contracts import MapState, Occ, Points, Vec3
from canopy.planning.pathing import PlanningGrid
from canopy.planning.safety import ClearanceMap

if TYPE_CHECKING:
    from canopy.config import MapCfg, PlannerCfg

__all__ = [
    "Frontier",
    "find_frontiers",
    "find_inspection_targets",
    "frontier_voxels",
    "inspected_fraction",
]


@dataclass(frozen=True, slots=True, eq=False)
class Frontier:
    """One cluster of unexplored-adjacent free space, with where to look at it from."""

    centroid: Vec3
    """Mean position of the cluster's voxels."""
    n_voxels: int
    """Cluster size before the gain weighting."""
    gain: float
    """``n_voxels``, doubled (``ground_band_gain``) if the centroid is low."""
    viewpoint: Vec3
    """Where a drone should stand to look at this cluster."""
    component: int
    """The :class:`~canopy.planning.pathing.PlanningGrid` label the viewpoint sits in."""


def frontier_voxels(
    state: MapState, clearance: ClearanceMap, planner: PlannerCfg
) -> npt.NDArray[np.bool_]:
    """FREE voxels with an UNKNOWN 6-neighbour, in scope and above ``frontier_min_z_m``.

    Parameters
    ----------
    state
        The shared occupancy grid.
    clearance
        Supplies the geofence; the map and clearance grids share indexing.
    planner
        Supplies ``frontier_min_z_m``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape like ``state.occ``.
    """
    free = state.occ == Occ.FREE
    unknown = state.occ == Occ.UNKNOWN
    touches_unknown = np.zeros_like(free)
    for _, _, shifted in _neighbours(unknown):
        touches_unknown |= shifted

    candidate: npt.NDArray[np.bool_] = free & touches_unknown
    if not np.any(candidate):
        return candidate

    above_min_z = _axis_centres(state)[2] >= planner.frontier_min_z_m
    result: npt.NDArray[np.bool_] = (
        candidate & _in_scope(state, clearance) & above_min_z[None, None, :]
    )
    return result


def _neighbours(
    mask: npt.NDArray[np.bool_],
) -> Iterator[tuple[int, int, npt.NDArray[np.bool_]]]:
    """Yield ``(axis, sign, shifted)`` for each of the six face neighbours.

    ``shifted[v]`` is ``mask`` at the neighbour of ``v`` one voxel along
    ``sign`` on ``axis``. A neighbour off the grid edge reads False, so the
    map's rim never counts as UNKNOWN (or FREE) space beyond it.
    """
    for axis in range(3):
        for sign in (1, -1):
            shifted = np.zeros_like(mask)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sign == 1:
                src[axis], dst[axis] = slice(1, None), slice(None, -1)
            else:
                src[axis], dst[axis] = slice(None, -1), slice(1, None)
            shifted[tuple(dst)] = mask[tuple(src)]
            yield axis, sign, shifted


def _axis_centres(state: MapState) -> list[npt.NDArray[np.float64]]:
    """World coordinate of every voxel centre along each axis: three 1-D arrays.

    Kept per axis rather than as a full ``(X, Y, Z, 3)`` grid: that grid is
    ~20 MB of float64 at lot scale, and building it twice per replan cost more
    than the frontier search it fed.
    """
    shape = state.occ.shape
    return [state.origin[a] + (np.arange(shape[a]) + 0.5) * state.voxel for a in range(3)]


def _in_scope(state: MapState, clearance: ClearanceMap | None) -> npt.NDArray[np.bool_]:
    """Whether each voxel centre is in the geofence and the survey region, shape like ``occ``.

    Both are boxes, so the test separates into one 1-D test per axis. No
    ``clearance`` skips the fence; no ``state.survey_bounds`` skips the region.
    """
    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    if clearance is not None:
        lo, hi = clearance.geofence[0].copy(), clearance.geofence[1].copy()
    if state.survey_bounds is not None:
        lo[:2] = np.maximum(lo[:2], state.survey_bounds[0])
        hi[:2] = np.minimum(hi[:2], state.survey_bounds[1])
    x, y, z = ((c >= lo[a]) & (c <= hi[a]) for a, c in enumerate(_axis_centres(state)))
    inside: npt.NDArray[np.bool_] = x[:, None, None] & y[None, :, None] & z[None, None, :]
    return inside


def _exposed(
    state: MapState,
) -> tuple[npt.NDArray[np.bool_], list[tuple[int, int, npt.NDArray[np.bool_]]]]:
    """Occupied voxels with a FREE face neighbour, and the six shifted FREE masks behind that."""
    faces = list(_neighbours(state.occ == Occ.FREE))
    exposed = np.zeros(state.occ.shape, dtype=np.bool_)
    for _, _, free_there in faces:
        exposed |= free_there
    exposed &= state.occ == Occ.OCC
    return exposed, faces


def inspected_fraction(state: MapState, map_cfg: MapCfg) -> float:
    """Estimate ground-band coverage from the swarm's own map, in ``[0, 1]``.

    Mapped surface -- occupied and exposed -- below ``map_cfg.ground_band_max_z_m``
    and inside the survey region, and the share of it ``surface_seen`` marks.
    It stands in for the simulator's triangle score, which the swarm cannot
    read, wherever a decision needs "how much is done".

    Parameters
    ----------
    state
        The shared map. ``0.0`` when it has no ``surface_seen`` or no surface.
    map_cfg
        Supplies ``ground_band_max_z_m``.
    """
    if state.surface_seen is None:
        return 0.0
    exposed, _ = _exposed(state)
    low = _axis_centres(state)[2] < map_cfg.ground_band_max_z_m
    surface = exposed & low[None, None, :] & _in_scope(state, None)
    total = int(np.count_nonzero(surface))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(surface & state.surface_seen)) / total


def _eligible_viewpoints(grid: PlanningGrid, components: Collection[int]) -> Points:
    """Planning-cell centres a live drone can reach."""
    eligible = grid.passable & np.isin(grid.labels, np.asarray(list(components), dtype=np.int32))
    return grid.centres(eligible)


def find_frontiers(
    state: MapState,
    clearance: ClearanceMap,
    grid: PlanningGrid,
    planner: PlannerCfg,
    map_cfg: MapCfg,
    *,
    components: Collection[int],
) -> list[Frontier]:
    """Cluster frontier voxels and pick a reachable viewpoint for each.

    Parameters
    ----------
    state
        The shared occupancy grid.
    clearance
        Distance-to-danger field, for the geofence and voxel geometry.
    grid
        The current coarse planning grid, labelled into components.
    planner
        Frontier and viewpoint tunables.
    map_cfg
        Supplies ``ground_band_max_z_m``.
    components
        Planning-grid labels that currently hold a live drone. A viewpoint
        outside all of them is unreachable this tick, so its cluster is
        dropped rather than assigned a doomed goal.

    Returns
    -------
    list[Frontier]
        Sorted by descending ``gain``.
    """
    voxels = frontier_voxels(state, clearance, planner)
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labels, n_labels = ndimage.label(voxels, structure=structure)
    if n_labels == 0:
        return []

    # Per-label sizes and index sums in one bincount pass each. ndimage.sum and
    # center_of_mass give the same numbers (the sums are small integers, exact
    # in float64) but walk the grid once per axis with a Python-level setup
    # per call, which made them most of the cost of this function.
    lab_ijk = np.nonzero(labels)
    lab = labels[lab_ijk]
    counts = np.bincount(lab, minlength=n_labels + 1)[1:]
    keep = counts >= planner.frontier_min_voxels
    if not np.any(keep):
        return []
    kept_counts = counts[keep]

    index_sums = np.stack(
        [np.bincount(lab, weights=lab_ijk[a], minlength=n_labels + 1)[1:] for a in range(3)],
        axis=1,
    )
    centroids_ijk = index_sums[keep] / kept_counts[:, None].astype(np.float64)
    centroids = state.origin + (centroids_ijk + 0.5) * state.voxel

    low = centroids[:, 2] < map_cfg.ground_band_max_z_m
    gains = np.where(low, kept_counts * planner.ground_band_gain, kept_counts.astype(np.float64))

    # One density field for the whole call: unknown-voxel fraction in a box of
    # side ~2*radius around each map voxel, a cheap separable stand-in for the
    # sphere the spec asks a viewpoint to be scored on.
    unknown = (state.occ == Occ.UNKNOWN).astype(np.float64)
    radius_vox = max(1, round(planner.viewpoint_gain_radius_m / state.voxel))
    density = ndimage.uniform_filter(unknown, size=2 * radius_vox + 1, mode="constant", cval=1.0)

    candidate_centres = _eligible_viewpoints(grid, components)
    if len(candidate_centres) == 0:
        return []
    candidate_idx = np.floor((candidate_centres - state.origin) / state.voxel).astype(np.intp)
    candidate_idx = np.clip(candidate_idx, 0, np.asarray(state.occ.shape) - 1)
    candidate_scores = density[candidate_idx[:, 0], candidate_idx[:, 1], candidate_idx[:, 2]]
    candidate_labels = grid.label_at(candidate_centres)

    lo, hi = planner.viewpoint_range_m
    frontiers = []
    for centroid, n_voxels, gain in zip(centroids, kept_counts, gains, strict=True):
        dist = np.linalg.norm(candidate_centres - centroid, axis=1)
        in_range = (dist >= lo) & (dist <= hi)
        if np.any(in_range):
            scores = np.where(in_range, candidate_scores, -np.inf)
            best_score = scores.max()
            tied = scores == best_score
            dists_tied = np.where(tied, dist, np.inf)
            chosen = int(np.argmin(dists_tied))
        else:
            near = dist <= 2.0 * hi
            if not np.any(near):
                continue  # unreachable this tick; drop the cluster
            dists_near = np.where(near, dist, np.inf)
            chosen = int(np.argmin(dists_near))
        viewpoint = candidate_centres[chosen]
        frontiers.append(
            Frontier(
                centroid=centroid,
                n_voxels=int(n_voxels),
                gain=float(gain),
                viewpoint=viewpoint,
                component=int(candidate_labels[chosen]),
            )
        )

    frontiers.sort(key=lambda f: f.gain, reverse=True)
    return frontiers


def find_inspection_targets(
    state: MapState,
    clearance: ClearanceMap,
    grid: PlanningGrid,
    planner: PlannerCfg,
    map_cfg: MapCfg,
    *,
    components: Collection[int],
) -> list[Frontier]:
    """Surfaces that are mapped but not yet seen well, with where to see them from.

    Voxel frontiers run dry long before the job is done. The sensor reaches
    12 m, so a few scans settle nearly every voxel of the lot, but coverage
    counts a surface only when it was seen from under 8 m and near square-on.
    What is left is occupied, exposed (it has a FREE face neighbour) and not
    ``surface_seen``. Exploration then turns into inspection, driven by the
    same assignment.

    Unseen surface voxels are bucketed into ``inspect_bucket_m`` cubes rather
    than connectivity-clustered. A wall or a lawn is one connected sheet, and
    a single cluster centroid in the middle of the yard would be a viewpoint
    that sees none of it. A bucket's outward normal is estimated from the
    directions to its voxels' FREE neighbours. The viewpoint is the reachable
    planning cell ``inspect_range_m`` out from the bucket, within
    ``inspect_max_angle_deg`` of that normal, preferring mid-range and
    square-on.

    Parameters
    ----------
    state
        The shared map. Nothing is returned when ``state.surface_seen`` is
        ``None``.
    clearance
        Supplies the geofence.
    grid
        The current planning grid, labelled into components.
    planner
        Inspection and gain tunables.
    map_cfg
        Supplies ``ground_band_max_z_m``.
    components
        Planning-grid labels that hold a live drone.

    Returns
    -------
    list[Frontier]
        One per bucket with a viewpoint, sorted by descending ``gain``.
        ``centroid`` is the surface and ``viewpoint`` is where to stand.
    """
    if state.surface_seen is None:
        return []
    exposed, faces = _exposed(state)
    target = exposed & ~state.surface_seen & _in_scope(state, clearance)
    if not np.any(target):
        return []

    # Centres and face normals only where there is a target: a few thousand
    # voxels, rather than full-grid (X, Y, Z, 3) arrays that are mostly zero.
    ijk = np.nonzero(target)
    axes = _axis_centres(state)
    pts = np.stack([axes[a][ijk[a]] for a in range(3)], axis=1)
    normals = np.zeros((len(pts), 3), dtype=np.float64)
    for axis, sign, free_there in faces:
        normals[:, axis] += sign * free_there[ijk]
    bucket_ijk = np.floor((pts - state.origin) / planner.inspect_bucket_m).astype(np.int64)
    _, bucket, counts = np.unique(bucket_ijk, axis=0, return_inverse=True, return_counts=True)
    bucket = bucket.reshape(-1)
    n_buckets = len(counts)
    centroid: npt.NDArray[np.float64] = np.stack(
        [np.bincount(bucket, pts[:, a], n_buckets) for a in range(3)], axis=1
    )
    centroid = centroid / counts[:, None]
    mean_normal: npt.NDArray[np.float64] = np.stack(
        [np.bincount(bucket, normals[:, a], n_buckets) for a in range(3)], axis=1
    )
    mean_normal = mean_normal / counts[:, None]

    candidates = _eligible_viewpoints(grid, components)
    if len(candidates) == 0:
        return []
    labels = grid.label_at(candidates)
    tree = cKDTree(candidates)
    lo, hi = planner.inspect_range_m
    mid = 0.5 * (lo + hi)
    cos_max = float(np.cos(np.radians(planner.inspect_max_angle_deg)))

    targets = []
    # One iteration per bucket (hundreds at most); the work inside is
    # vectorised over the few thousand candidate cells the tree returns.
    for b in np.flatnonzero(counts >= planner.inspect_min_voxels):
        near = np.asarray(tree.query_ball_point(centroid[b], hi), dtype=np.intp)
        if len(near) == 0:
            continue
        offset = candidates[near] - centroid[b]
        dist = np.linalg.norm(offset, axis=1)
        n_len = float(np.linalg.norm(mean_normal[b]))
        # A bucket whose faces point every which way (a bush) has no useful
        # normal, and then any direction is as good as another.
        facing = (
            (offset @ (mean_normal[b] / n_len)) / np.maximum(dist, 1e-9)
            if n_len > _MIN_NORMAL
            else np.ones_like(dist)
        )
        ok = (dist >= lo) & (facing >= cos_max)
        if not np.any(ok):
            continue
        score = np.where(ok, facing - np.abs(dist - mid) / (hi - lo), -np.inf)
        chosen = near[int(np.argmax(score))]
        n = int(counts[b])
        low = bool(centroid[b, 2] < map_cfg.ground_band_max_z_m)
        targets.append(
            Frontier(
                centroid=centroid[b],
                n_voxels=n,
                gain=float(n * planner.ground_band_gain) if low else float(n),
                viewpoint=candidates[chosen],
                component=int(labels[chosen]),
            )
        )
    targets.sort(key=lambda f: f.gain, reverse=True)
    return targets


#: Length of a bucket's mean face normal below which it has no dominant facing.
#: Unit normals that agree average to ~1; opposing ones cancel toward 0.
_MIN_NORMAL = 0.3
