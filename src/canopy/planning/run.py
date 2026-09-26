"""One mapping mission, end to end: world, mapper and controller in one loop.

:class:`MissionRun` is the loop every consumer drives -- the viewer one frame
at a time, tests and batch runs straight through with :meth:`MissionRun.run`.
Keeping the wiring here, rather than in the viewer, is what lets a headless test
exercise exactly what the window shows.

Per control tick::

    targets = controller.step(map)      # plan, follow, shield
    scans   = world.step(targets)       # move; scan on sensor ticks
    mapper.integrate(scan) for scan in scans
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from canopy.contracts import DroneState, SceneGeometry, SceneManifest
from canopy.mapping import Mapper
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
        The property. Only its lot bounds and launch point reach the planner;
        the geometry reaches the sensor and the coverage bookkeeping.
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
        self._cfg = cfg
        self.pads = launch_pads(manifest.home, n_drones, cfg)
        self.world = SimWorld(geometry, self.pads, cfg, np.random.default_rng(seed), sensor=sensor)
        self.mapper = Mapper(manifest, geometry, cfg)
        self.controller = MissionController(cfg, manifest.lot_bounds, self.pads)
        for lo, hi in self.controller.launch_boxes():
            self.mapper.mark_free_box(lo, hi)
        for scan in self.world.scan_all():
            self.mapper.integrate(scan)

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

    def tick(self) -> None:
        """Advance one control tick. A finished mission keeps its drones parked."""
        targets = self.controller.step(
            self.world.t,
            self.world.drones,
            self.mapper.state,
            map_version=self.mapper.version,
            coverage_ground_band=self.mapper.coverage_ground_band,
        )
        for scan in self.world.step(targets):
            self.mapper.integrate(scan)

    def run(self, max_t_s: float) -> None:
        """Tick until the mission is done or ``max_t_s`` of simulated time passes."""
        while not self.done and self.world.t < max_t_s:
            self.tick()
