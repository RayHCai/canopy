"""The mapper: fuses every sweep into the one shared :class:`MapState`.

Composes :mod:`canopy.mapping.occupancy` (planning),
:mod:`canopy.mapping.surface` (what has been photographed well),
:mod:`canopy.mapping.property` (which building is the house) and the
perception stage's :class:`~canopy.perception.ObjectDetector` (discovered
objects) behind one object, so the mission controller and the viewer each have
a single thing to poll. ``version`` replaces the spec's ``occ_changed`` flag
with a monotonic counter: a planner comparing "have I already reacted to this
version" survives being called more than once per scan, where a boolean flag
would not.

The mapper is the drone side of the information boundary (ADR 0016). It takes
an :class:`~canopy.contracts.Observation` -- range, direction, colour and the
lidar's ray layout -- and the operator's envelope and launch point, and nothing
from the scene: no manifest, no geometry, no triangle or object ids. The
simulator's triangle coverage is scored beside the mission by
:class:`~canopy.mapping.coverage.CoverageTracker`, which this class never holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import DiscoveredObject, MapState, Observation, Vec3
from canopy.mapping.occupancy import integrate_occupancy, mark_free_box, new_map_state
from canopy.mapping.property import infer_survey_bounds
from canopy.mapping.surface import SurfaceTracker
from canopy.perception import ObjectDetector

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["Mapper"]


class Mapper:
    """Owns the shared :class:`MapState` and updates it one sweep at a time.

    Parameters
    ----------
    cfg
        Full configuration; ``cfg.map``, ``cfg.sensor`` and ``cfg.perception``
        are read.
    envelope
        The operator's flight envelope,
        :func:`~canopy.mapping.property.survey_envelope`, shape ``(2, 3)``.
        The map covers exactly this plan box.
    launch_xy
        Plan position the swarm launched from, shape ``(2,)``; the house is
        taken to be the mapped building nearest it.
    """

    def __init__(
        self, cfg: Config, envelope: npt.NDArray[np.float64], launch_xy: npt.NDArray[np.float64]
    ) -> None:
        self._map_cfg = cfg.map
        self._sensor_cfg = cfg.sensor
        self._launch_xy = np.asarray(launch_xy, dtype=np.float64)
        self._state = new_map_state(envelope, cfg.map)
        self._surface = SurfaceTracker(
            cfg.map, self._state.origin, self._state.voxel, self._state.occ.shape
        )
        # A live view, not a copy: the planner turns it into inspection targets.
        self._state.surface_seen = self._surface.seen
        self._version = 0
        # The envelope keeps the street and anything past it out of the
        # evidence; the survey region, once inferred, narrows what is reported.
        self._detector = ObjectDetector(cfg.perception, envelope)
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

    def integrate(self, obs: Observation) -> None:
        """Fuse one sweep into occupancy, the surface mask and perception.

        At most ``perception.extract_hz`` times a second of sweep time, on the
        first sweep due, the house is re-inferred (``state.survey_bounds``)
        and objects are re-extracted; ``state.discovered`` is then replaced,
        never mutated, and holds only objects inside the survey region.

        Parameters
        ----------
        obs
            One 360-degree sweep from a live drone.
        """
        changed = integrate_occupancy(self._state, obs, self._map_cfg, self._sensor_cfg.max_range_m)
        if changed:
            self._version += 1
        self._surface.integrate(obs)
        self._detector.integrate(obs)
        if obs.t >= self._next_detect_t:
            bounds = _widen(
                self._state.survey_bounds,
                infer_survey_bounds(self._state, self._launch_xy, self._map_cfg),
            )
            self._state.survey_bounds = bounds
            self._state.discovered = _within(self._detector.extract(), bounds)
            self._next_detect_t = obs.t + self._detect_period_s

    def mark_free_box(self, lo: Vec3, hi: Vec3) -> None:
        """Seed a box of the map FREE ahead of any scan; see :func:`mark_free_box`."""
        before = self._state.occ.copy()
        mark_free_box(self._state, lo, hi)
        if np.any(self._state.occ != before):
            self._version += 1


def _widen(
    old: npt.NDArray[np.float64] | None, new: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64] | None:
    """Return the plan box covering both ``old`` and ``new``; either may be ``None``.

    The survey region only grows. A house seen more completely widens it, and
    a different building picked later adds to it, but nothing the swarm has
    already taken in drops back out, which would end the survey early with
    the house half flown.
    """
    if old is None or new is None:
        return new if old is None else old
    return np.array([np.minimum(old[0], new[0]), np.maximum(old[1], new[1])], dtype=np.float64)


def _within(
    objects: dict[int, DiscoveredObject], bounds: npt.NDArray[np.float64] | None
) -> dict[int, DiscoveredObject]:
    """Keep the objects whose plan position lies inside ``bounds``; all of them if ``None``."""
    if bounds is None:
        return objects
    return {
        track: obj
        for track, obj in objects.items()
        if bool(np.all((obj.pos[:2] >= bounds[0]) & (obj.pos[:2] <= bounds[1])))
    }
