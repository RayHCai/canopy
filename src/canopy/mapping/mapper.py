"""The mapper: fuses every scan into the one shared :class:`MapState`.

Composes :mod:`canopy.mapping.occupancy` (planning),
:mod:`canopy.mapping.coverage` (the reveal and completion metric) and the
perception stage's :class:`~canopy.perception.ObjectDetector` (discovered
objects) behind one object, so the mission controller and the viewer each have
a single thing to poll. ``version`` replaces the spec's ``occ_changed`` flag
with a monotonic counter: a planner comparing "have I already reacted to this
version" survives being called more than once per scan, where a boolean flag
would not.

Coverage is ground truth and reads each scan's triangle ids. The detector is
handed only :meth:`Scan.observation`, the range and colour a real drone would
have, so nothing it discovers can come from the simulator's labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import MapState, Scan, SceneGeometry, SceneManifest, Vec3
from canopy.mapping.coverage import CoverageTracker
from canopy.mapping.occupancy import integrate_occupancy, mark_free_box, new_map_state
from canopy.perception import ObjectDetector

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["Mapper"]


class Mapper:
    """Owns the shared :class:`MapState` and updates it one scan at a time."""

    def __init__(self, manifest: SceneManifest, geometry: SceneGeometry, cfg: Config) -> None:
        """Build an empty map over ``manifest.lot_bounds`` and its coverage tracker.

        Parameters
        ----------
        manifest
            The scene's single source of truth (lot bounds, object classes).
        geometry
            Every scene triangle, for coverage accounting.
        cfg
            Full configuration; ``cfg.map``, ``cfg.sensor`` and
            ``cfg.perception`` are read.
        """
        self._map_cfg = cfg.map
        self._sensor_cfg = cfg.sensor
        self._state = new_map_state(manifest.lot_bounds, cfg.map, geometry.n_triangles)
        self._geometry = geometry
        self._coverage = CoverageTracker(
            manifest,
            geometry,
            cfg.map,
            self._state.origin,
            self._state.voxel,
            self._state.occ.shape,
        )
        # The coverage tracker owns the seen mask; MapState.tri_seen must be the
        # same array object so viewer and planner code reading either one agree.
        self._state.tri_seen = self._coverage.tri_seen
        # Likewise a live view of the tracker's photo-quality voxels, which the
        # planner turns into inspection targets.
        self._state.surface_seen = self._coverage.seen_voxels
        self._version = 0
        # The lot's extent is prior knowledge (the customer's parcel), not a
        # label, so the detector may use it to ignore the neighbours.
        self._detector = ObjectDetector(cfg.perception, manifest.lot_bounds)
        self._detect_period_s = 1.0 / cfg.perception.extract_hz
        self._next_detect_t = 0.0

    @property
    def state(self) -> MapState:
        """The shared map. Mutated in place by :meth:`integrate`."""
        return self._state

    @property
    def version(self) -> int:
        """Bumped every time ``state.occ`` changes.

        The planner compares this to the version it last replanned against,
        rather than consuming a one-shot "changed" flag, so multiple readers
        of the same map never race over who gets to see the change.
        """
        return self._version

    def integrate(self, scan: Scan) -> None:
        """Fuse one scan into occupancy, coverage and perception.

        Objects are re-extracted from the accumulated evidence at most
        ``perception.extract_hz`` times a second of scan time, on the first
        scan due; ``state.discovered`` is then replaced, never mutated.

        Parameters
        ----------
        scan
            One 360-degree ray cast from a live drone.
        """
        changed = integrate_occupancy(
            self._state, scan, self._map_cfg, self._sensor_cfg.max_range_m
        )
        if changed:
            self._version += 1
        self._coverage.integrate(scan, self._geometry)
        self._detector.integrate(scan.observation())
        if scan.t >= self._next_detect_t:
            self._state.discovered = self._detector.extract()
            self._next_detect_t = scan.t + self._detect_period_s

    def mark_free_box(self, lo: Vec3, hi: Vec3) -> None:
        """Seed a box of the map FREE ahead of any scan; see :func:`mark_free_box`."""
        before = self._state.occ.copy()
        mark_free_box(self._state, lo, hi)
        if np.any(self._state.occ != before):
            self._version += 1

    def pop_revealed(self) -> npt.NDArray[np.int64]:
        """Sorted unique global triangle ids newly seen since the last call."""
        return self._coverage.pop_revealed()

    @property
    def coverage_total(self) -> float:
        """Seen area over total area of non-background, non-ground triangles."""
        return self._coverage.coverage_total

    @property
    def coverage_ground_band(self) -> float:
        """Seen area over ground-band WALL/DOOR/METER/BUSH area. The mission metric."""
        return self._coverage.coverage_ground_band
