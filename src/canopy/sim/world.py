"""The simulated world: time, drones, dynamics and sensing.

:class:`SimWorld` is deliberately ignorant of planning. It takes one target per
drone each control tick, moves the drones, and on sensor ticks returns one
:class:`~canopy.contracts.Scan` per live drone. Whoever drives it decides what
the targets are; the same world serves the viewer, tests and batch runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from canopy.contracts import DroneState, Points, Scan, SceneGeometry, Vec3
from canopy.errors import SimulationError
from canopy.sim.dynamics import KinematicDynamics
from canopy.sim.sensors import RaySensor

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["SimWorld"]


class SimWorld:
    """Drones flying through one generated scene.

    Parameters
    ----------
    geometry
        The scene's triangles, from :func:`canopy.sim.scene.load_geometry`.
    pads
        Start position per drone, shape ``(N, 3)``; drone ``i`` starts at
        ``pads[i]`` at rest.
    cfg
        Validated configuration. Reads ``sim`` and ``sensor``.
    rng
        Source of sensor noise and scan jitter. Pass a seeded generator for a
        reproducible run.
    sensor
        An already-built sensor for ``geometry``, to reuse its raycasting scene
        across runs on the same property (building one costs a noticeable
        fraction of a second). Built from ``geometry`` when omitted.
    """

    def __init__(
        self,
        geometry: SceneGeometry,
        pads: Points,
        cfg: Config,
        rng: np.random.Generator,
        *,
        sensor: RaySensor | None = None,
    ) -> None:
        self._cfg = cfg
        self._sensor = sensor if sensor is not None else RaySensor(geometry, cfg.sensor, rng)
        self._dynamics = KinematicDynamics(cfg.sim)
        self._states = [
            DroneState(
                drone_id=i, pos=np.asarray(p, dtype=np.float64).copy(), vel=np.zeros(3), yaw=0.0
            )
            for i, p in enumerate(np.asarray(pads, dtype=np.float64).reshape(-1, 3))
        ]
        self._dynamics.reset(self._states)
        # `.drones` hands these out before `step` has ever run; freeze them too,
        # matching the read-only snapshots `KinematicDynamics.step` returns.
        for state in self._states:
            state.pos.flags.writeable = False
            state.vel.flags.writeable = False
        self._tick = 0
        # Scans land on every Nth control tick; SimCfg.validate guarantees N is whole.
        self._sensor_every = cfg.sim.control_hz // cfg.sim.sensor_hz

    @property
    def t(self) -> float:
        """Simulated seconds since the start."""
        return self._tick * self._cfg.sim.dt

    @property
    def drones(self) -> list[DroneState]:
        """Every drone's latest state, in ``drone_id`` order.

        Each returned state is an independent snapshot, not the live simulator
        object, and its ``pos``/``vel`` arrays are marked read-only: writing
        through them (``state.pos[:] = ...``) raises ``ValueError`` rather than
        silently corrupting the next tick. Reassigning the attribute itself
        (``state.pos = ...``) is fine but has no effect on the simulation.
        """
        return list(self._states)

    @property
    def sensor(self) -> RaySensor:
        """The ray sensor, for reuse or for ground-truth checks in tests."""
        return self._sensor

    def scan_all(self) -> list[Scan]:
        """One scan per live drone from where it is now, without advancing time."""
        return [self._sensor.scan(s.drone_id, self.t, s.pos) for s in self._states if s.alive]

    def step(self, targets: Mapping[int, Vec3]) -> list[Scan]:
        """Advance one control tick.

        Parameters
        ----------
        targets
            Target position per ``drone_id``; a drone absent from it brakes.

        Returns
        -------
        list of Scan
            One per live drone on sensor ticks, empty otherwise.
        """
        self._states = self._dynamics.step(targets, self._cfg.sim.dt)
        self._tick += 1
        return self.scan_all() if self._tick % self._sensor_every == 0 else []

    def kill_drone(self, drone_id: int) -> None:
        """Mark a drone dead; the dynamics then land it where it is.

        Raises
        ------
        SimulationError
            If no drone has that id.
        """
        if not 0 <= drone_id < len(self._states):
            msg = f"no drone {drone_id}; the swarm has {len(self._states)}"
            raise SimulationError(msg)
        self._states[drone_id].alive = False
        self._dynamics.reset(self._states)
