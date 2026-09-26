"""The shared world model: occupancy, triangle coverage and discovered semantics.

:mod:`canopy.mapping.occupancy` holds the voxel occupancy grid the planner
reads, :mod:`canopy.mapping.coverage` tracks which triangles of the property
have been seen and how much of the ground band is covered, and
:mod:`canopy.mapping.mapper` composes both, plus the perception stage's object
detector, behind one object (:class:`Mapper`) so the mission controller and the
viewer each have a single thing to poll for the current
:class:`~canopy.contracts.MapState`.

Public API: :class:`Mapper`.
"""

from __future__ import annotations

from canopy.mapping.mapper import Mapper

__all__ = ["Mapper"]
