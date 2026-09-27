"""Per-triangle coverage: the grayscale-to-colour reveal and the coverage score.

This is the simulator grading the swarm, not part of the swarm. It reads each
scan's ground-truth triangle ids, the mesh's normals and the manifest's
classes, so it is owned by :class:`~canopy.planning.MissionRun` beside the
mission and never by the :class:`~canopy.mapping.Mapper` the planner reads
(ADR 0016). The swarm's own notion of what it has photographed is
:mod:`canopy.mapping.surface`, judged from its scans alone.

The spec's coverage rule is "a photo-quality hit marks its triangle seen." That
is exactly right for a wall, which is a handful of large triangles, but wrong
for a bush: the asset library gives one bush ~9,000 tiny triangles, so a photo
that clearly reveals the whole shrub to a human still only tags the few dozen
triangles the rays happened to land on, leaving the mesh speckled grey forever
in the viewer and permanently short of ``coverage_ground_band``. This module
keeps the direct-hit rule but adds a second one: a photo-quality hit also marks
seen every triangle whose centroid falls in the same map voxel as the hit
point. Surfaces are effectively revealed at voxel granularity, which matches
what a human looking at the reveal actually judges "covered" to mean, without
loosening the rule for large, sparse walls where a voxel and a triangle are
already about the same size.

Precomputing which voxel owns each triangle's centroid, and inverting that into
a voxel -> triangle-ids lookup, means a newly seen voxel expands to its
triangles with an ``np.searchsorted`` slice instead of a Python scan over
every triangle in the scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import Cls, Points, Scan, SceneGeometry, SceneManifest

if TYPE_CHECKING:
    from canopy.config import MapCfg

__all__ = ["CoverageTracker"]

#: Classes counted toward the ground-band completion metric.
# PANEL (the breaker panel) is photographed alongside the meter, so it counts;
# CONDUIT is centimetres across and would add variance, not signal.
_GROUND_BAND_CLASSES = (Cls.WALL, Cls.DOOR, Cls.METER, Cls.PANEL, Cls.BUSH)


class CoverageTracker:
    """Fuses photo-quality hits into a per-triangle seen mask and area metrics.

    Everything that can be derived once from the (static) geometry -- centroid
    voxel indices, the voxel -> triangle lookup, non-background area totals --
    is precomputed in :meth:`__init__` so :meth:`integrate` is just array
    indexing plus a sort-free voxel expansion, safe to call once per scan.
    """

    def __init__(
        self,
        manifest: SceneManifest,
        geometry: SceneGeometry,
        cfg: MapCfg,
        origin: Points,
        voxel: float,
        shape: tuple[int, int, int],
    ) -> None:
        self._cfg = cfg
        self._origin = np.asarray(origin, dtype=np.float64)
        self._voxel = voxel
        self._shape = shape

        obj_background = np.array([o.background for o in manifest.objects], dtype=np.bool_)
        background = obj_background[geometry.tri_obj]
        # Revealable = on the surveyed lot. Counted = revealable and visible
        # from outside at all, so faces sealed inside the house stay out of the
        # metrics' denominators (SceneGeometry.tri_exterior). The two differ on
        # purpose: tri_exterior is a heuristic probed from one point per face,
        # and it calls a ground triangle straddling a wall, or a leaf deep in a
        # canopy, sealed. A photo-quality hit proves otherwise, so it reveals
        # the face even though the metrics still leave it out.
        self._revealable = ~background
        exterior = (
            np.ones(len(background), dtype=np.bool_)
            if geometry.tri_exterior is None
            else geometry.tri_exterior
        )
        self._counted = self._revealable & exterior

        # -- centroid -> flat voxel index, -1 for background / out-of-grid ---
        idx = np.floor((geometry.tri_centroid - self._origin) / voxel).astype(np.int64)
        in_grid = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
        flat = idx[:, 0] * shape[1] * shape[2] + idx[:, 1] * shape[2] + idx[:, 2]
        self._tri_voxel = np.where(in_grid & self._revealable, flat, -1)

        # -- voxel -> triangle ids, via a sort + searchsorted -----------------
        owned = np.flatnonzero(self._tri_voxel >= 0)
        order = np.argsort(self._tri_voxel[owned], kind="stable")
        self._voxel_sorted = self._tri_voxel[owned][order]
        self._tri_by_voxel = owned[order]

        self.tri_seen = np.zeros(geometry.n_triangles, dtype=np.bool_)
        self._seen_voxels = np.zeros(int(np.prod(shape)), dtype=np.bool_)
        # Batches of newly seen ids, merged on pop: a scan can reveal thousands
        # of triangles, and a per-id Python set update would loop over them.
        self._newly_seen: list[npt.NDArray[np.int64]] = []

        area = geometry.tri_area
        # Look up each object's class once, then broadcast onto every triangle
        # via its owning obj_id, instead of a per-triangle Python loop.
        cls_by_obj = np.array([o.cls for o in manifest.objects])
        cls_of_tri = cls_by_obj[geometry.tri_obj]
        non_ground = self._counted & (cls_of_tri != Cls.GROUND)
        self._total_area = float(area[non_ground].sum())

        band = (
            self._counted
            & np.isin(cls_of_tri, _GROUND_BAND_CLASSES)
            & (geometry.tri_centroid[:, 2] >= 0.0)
            & (geometry.tri_centroid[:, 2] <= cfg.ground_band_max_z_m)
        )
        self._band_mask = band
        self._band_area = float(area[band].sum())
        self._non_ground_mask = non_ground
        self._area = area

    def integrate(self, scan: Scan, geometry: SceneGeometry) -> None:
        """Mark triangles seen from one scan's photo-quality hits.

        Parameters
        ----------
        scan
            One 360-degree ray cast.
        geometry
            The static scene geometry ``scan.tri_ids`` indexes into.
        """
        hit = scan.tri_ids >= 0
        if not np.any(hit):
            return
        tri_ids = scan.tri_ids[hit]
        dist = scan.dist[hit]
        dirs = scan.dirs[hit]

        normals = geometry.tri_normal[tri_ids]
        cos_incidence = np.abs(np.einsum("ij,ij->i", dirs, normals))
        # A zero normal (degenerate triangle) can never clear the abs(cos) bar.
        incidence_deg = np.degrees(np.arccos(np.clip(cos_incidence, 0.0, 1.0)))
        quality = (dist < self._cfg.coverage_max_range_m) & (
            incidence_deg < self._cfg.coverage_max_incidence_deg
        )
        good_tri = tri_ids[quality]
        good_tri = good_tri[self._revealable[good_tri]]
        if good_tri.size == 0:
            return

        # Direct hits.
        newly_direct = good_tri[~self.tri_seen[good_tri]]
        self.tri_seen[good_tri] = True

        # Hit-point voxels expand to every triangle whose centroid lives there.
        hit_pts = scan.origin[None, :] + dirs[quality] * dist[quality, None]
        idx = np.floor((hit_pts - self._origin) / self._voxel).astype(np.int64)
        in_grid = np.all((idx >= 0) & (idx < np.asarray(self._shape)), axis=1)
        idx = idx[in_grid]
        flat = idx[:, 0] * self._shape[1] * self._shape[2] + idx[:, 1] * self._shape[2] + idx[:, 2]
        flat = np.unique(flat)
        flat = flat[~self._seen_voxels[flat]]
        self._seen_voxels[flat] = True

        expanded_tris: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        if flat.size:
            lo = np.searchsorted(self._voxel_sorted, flat, side="left")
            hi = np.searchsorted(self._voxel_sorted, flat, side="right")
            ranges = [self._tri_by_voxel[a:b] for a, b in zip(lo, hi, strict=True)]
            expanded_tris = np.concatenate(ranges).astype(np.int64) if ranges else expanded_tris
        newly_expanded = expanded_tris[~self.tri_seen[expanded_tris]]
        self.tri_seen[expanded_tris] = True

        self._newly_seen.append(np.asarray(newly_direct, dtype=np.int64))
        self._newly_seen.append(np.asarray(newly_expanded, dtype=np.int64))

    def pop_revealed(self) -> npt.NDArray[np.int64]:
        """Sorted unique triangle ids newly seen since the last call, then clear."""
        if not self._newly_seen:
            return np.zeros(0, dtype=np.int64)
        ids = np.unique(np.concatenate(self._newly_seen))
        self._newly_seen.clear()
        return ids

    @property
    def coverage_total(self) -> float:
        """Seen area over total area of non-background, non-ground triangles."""
        if self._total_area <= 0.0:
            return 0.0
        seen = self._area[self._non_ground_mask & self.tri_seen].sum()
        return float(seen / self._total_area)

    @property
    def coverage_ground_band(self) -> float:
        """Seen area over WALL/DOOR/METER/BUSH area in the ground band. The mission score."""
        if self._band_area <= 0.0:
            return 0.0
        seen = self._area[self._band_mask & self.tri_seen].sum()
        return float(seen / self._band_area)
