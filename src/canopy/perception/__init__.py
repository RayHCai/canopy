"""Perception: classify what the drones sense into objects, without ground truth.

The input is an :class:`~canopy.contracts.Observation` -- range and colour per
ray, and nothing the simulator knows about which object a ray hit. The output
is :class:`~canopy.contracts.DiscoveredObject` s, each with a class, a tight
upright box and a stable track id.

:mod:`canopy.perception.detector` holds the detector itself
(:class:`ObjectDetector`). It is built on :mod:`canopy.perception.colour`
(HSV colour rules), :mod:`canopy.perception.cells` (sparse evidence cells and
connected grouping) and :mod:`canopy.perception.boxes` (box fitting). Which
classes it looks for, and what tells each apart, is data: the
``perception.classes`` list in ``config/default.yaml``.

Public API: :class:`ObjectDetector`.
"""

from __future__ import annotations

from canopy.perception.detector import ObjectDetector

__all__ = ["ObjectDetector"]
