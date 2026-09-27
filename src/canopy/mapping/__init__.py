"""The shared world model: occupancy, inspection and discovered semantics.

:mod:`canopy.mapping.occupancy` holds the voxel occupancy grid the planner
reads, :mod:`canopy.mapping.surface` marks which mapped surfaces have been
photographed well, :mod:`canopy.mapping.property` sizes the map from the
operator's envelope and infers which building is the house, and
:mod:`canopy.mapping.mapper` composes them, plus the perception stage's object
detector, behind one object (:class:`Mapper`) so the mission controller and the
viewer each have a single thing to poll for the current
:class:`~canopy.contracts.MapState`. All of that is built from observations
alone (ADR 0016).

:mod:`canopy.mapping.coverage` is the exception: :class:`CoverageTracker`
scores the swarm against the simulator's triangles, for the reveal and the
report, and is kept out of the mapper on purpose.

Public API: :class:`CoverageTracker`, :class:`Mapper`, :func:`survey_envelope`.
"""

from __future__ import annotations

from canopy.mapping.coverage import CoverageTracker
from canopy.mapping.mapper import Mapper
from canopy.mapping.property import survey_envelope

__all__ = ["CoverageTracker", "Mapper", "survey_envelope"]
