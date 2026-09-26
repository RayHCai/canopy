"""Simulation: scene loading, the ray sensor, kinematic dynamics and the world clock.

:mod:`canopy.sim.scene` turns a :class:`~canopy.contracts.SceneManifest`'s OBJ
meshes into one :class:`~canopy.contracts.SceneGeometry` (:func:`load_geometry`).
:mod:`canopy.sim.sensors` casts an equirectangular grid of rays against that
geometry each sensor tick (:class:`RaySensor`). :mod:`canopy.sim.dynamics` moves
drones toward per-tick targets with a velocity-limited kinematic model, no
forces or attitude (:class:`KinematicDynamics`). :mod:`canopy.sim.world` composes
the three into the simulated world every consumer drives (:class:`SimWorld`):
one target per drone each control tick in, one :class:`~canopy.contracts.Scan`
per live drone out on sensor ticks.

Public API: :class:`KinematicDynamics`, :class:`RaySensor`, :class:`SimWorld`,
:func:`load_geometry`.
"""

from __future__ import annotations

from canopy.sim.dynamics import KinematicDynamics
from canopy.sim.scene import load_geometry
from canopy.sim.sensors import RaySensor
from canopy.sim.world import SimWorld

__all__ = ["KinematicDynamics", "RaySensor", "SimWorld", "load_geometry"]
