"""One mapping mission, end to end: world, mapper and controller in one loop.

:class:`MissionRun` is the loop every consumer drives -- the viewer one frame
at a time, tests and batch runs straight through with :meth:`MissionRun.run`.
Keeping the wiring here, rather than in the viewer, is what lets a headless test
exercise exactly what the window shows.

Per control tick::

    targets = controller.step(map)      # plan, follow, shield
    scans   = world.step(targets)       # move; scan on sensor ticks
    mapper.integrate(scan.observation()) for scan in scans   # the swarm
    coverage.integrate(scan) for scan in scans               # the score

This is also where the information boundary is drawn (ADR 0016). The world,
its sensor and the coverage score hold the scene; the mapper and controller
are handed only observations, the launch pads and the operator's envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from canopy.contracts import DroneState, Scan, SceneGeometry, SceneManifest
from canopy.errors import PlanningError
from canopy.mapping import CoverageTracker, Mapper, survey_envelope
from canopy.planning.mission import MissionController, Phase
from canopy.sim import SimWorld
from canopy.worldgen import launch_pads

if TYPE_CHECKING:
    from canopy.config import Config
    from canopy.sim import RaySensor

__all__ = ["MissionRun"]


class MissionRun:
    """A swarm of ``n_drones`` mapping one generated property.

    Parameters
    ----------
    manifest
        The property. Only its launch point reaches the swarm, as the pads
        the crew set down; the rest reaches the coverage score.
    geometry
        The manifest's triangles (:func:`canopy.sim.scene.load_geometry`).
    n_drones
        Swarm size, at least one.
    cfg
        Validated configuration.
    seed
        Seeds sensor noise, so a run is reproducible.
    sensor
        Optional prebuilt sensor for ``geometry``; see :class:`SimWorld`.
    """

    def __init__(
        self,
        manifest: SceneManifest,
        geometry: SceneGeometry,
        n_drones: int,
        cfg: Config,
        *,
        seed: int = 0,
        sensor: RaySensor | None = None,
    ) -> None:
        # Caught here, not downstream: an empty pad array reaches the mapper as
        # a NaN mean and fails there as a bare ValueError naming nothing.
        if n_drones < 1:
            msg = f"a mission needs at least one drone, got n_drones={n_drones}"
            raise PlanningError(msg)
        self._cfg = cfg
        self.pads = launch_pads(manifest.home, n_drones, cfg)
        self.world = SimWorld(geometry, self.pads, cfg, np.random.default_rng(seed), sensor=sensor)
        envelope = survey_envelope(self.pads, cfg)
        self.mapper = Mapper(cfg, envelope, self.pads[:, :2].mean(axis=0))
        self.controller = MissionController(cfg, envelope, self.pads)
        state = self.mapper.state
        self._geometry = geometry
        self._coverage = CoverageTracker(
            manifest, geometry, cfg.map, state.origin, state.voxel, state.occ.shape
        )
        for lo, hi in self.controller.launch_boxes():
            self.mapper.mark_free_box(lo, hi)
        for scan in self.world.scan_all():
            self._integrate(scan)

    @property
    def t(self) -> float:
        """Simulated seconds since launch."""
        return self.world.t

    @property
    def drones(self) -> list[DroneState]:
        """Every drone's latest state."""
        return self.world.drones

    @property
    def phase(self) -> Phase:
        """The controller's mission phase."""
        return self.controller.phase

    @property
    def done(self) -> bool:
        """Whether the mission has finished and every drone is home."""
        return self.controller.done

    @property
    def coverage_ground_band(self) -> float:
        """Ground-truth ground-band coverage, the mission score; never read by the swarm."""
        return self._coverage.coverage_ground_band

    @property
    def coverage_total(self) -> float:
        """Ground-truth coverage of every non-ground surface on the lot."""
        return self._coverage.coverage_total

    def pop_revealed(self) -> npt.NDArray[np.int64]:
        """Sorted unique triangle ids newly seen since the last call, for the reveal."""
        return self._coverage.pop_revealed()

    def tick(self) -> None:
        """Advance one control tick. A finished mission keeps its drones parked."""
        targets = self.controller.step(
            self.world.t,
            self.world.drones,
            self.mapper.state,
            map_version=self.mapper.version,
        )
        for scan in self.world.step(targets):
            self._integrate(scan)

    def _integrate(self, scan: Scan) -> None:
        """Hand the swarm what its sensor sensed, and score the scan against the scene."""
        self.mapper.integrate(scan.observation())
        self._coverage.integrate(scan, self._geometry)

    def run(self, max_t_s: float) -> None:
        """Tick until the mission is done or ``max_t_s`` of simulated time passes."""
        while not self.done and self.world.t < max_t_s:
            self.tick()
