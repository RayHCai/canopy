"""Clearances between upright boxes, the one measurement siting rules keep needing.

Everything a battery must keep clear of -- a bush, a conduit riser, the meter
itself -- reaches the site stage as an :class:`~canopy.contracts.OrientedBox`:
upright, turned only about +Z. The battery is one too. So "how close is this
site to that object" is always the gap between two yaw-only boxes, which splits
exactly into a footprint gap in plan and a vertical gap in height.

The footprint gap between two rectangles is found without rasterising: if they
overlap (separating-axis test on their four edge directions) it is zero,
otherwise it is attained between a corner of one and an edge of the other, so
the least corner-to-rectangle distance, taken both ways, is exact. All of it is
broadcast over every candidate against every object at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from canopy.contracts import OrientedBox, Points

__all__ = ["Boxes", "box_gap", "point_gap"]

FloatArray = npt.NDArray[np.float64]

#: Corner signs of a rectangle in its own (along, across) frame, in winding order.
_CORNER_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _perp(u: FloatArray) -> FloatArray:
    """Rotate plan vectors a quarter turn anticlockwise."""
    return np.stack([-u[..., 1], u[..., 0]], axis=-1)


def _dot(a: FloatArray, b: FloatArray) -> FloatArray:
    out: FloatArray = np.sum(a * b, axis=-1)
    return out


@dataclass(frozen=True, slots=True, eq=False)
class Boxes:
    """A batch of upright boxes, as parallel arrays.

    Attributes
    ----------
    centre
        Footprint centres, shape ``(K, 2)``.
    axis
        Unit plan direction of each box's first extent, shape ``(K, 2)``.
    half
        Half extents along ``axis`` and across it, shape ``(K, 2)``.
    z
        Bottom and top heights, shape ``(K, 2)``.
    """

    centre: Points
    axis: Points
    half: Points
    z: Points

    def __len__(self) -> int:
        """Return the number of boxes."""
        return len(self.centre)

    @classmethod
    def from_oriented(cls, boxes: Sequence[OrientedBox]) -> Boxes:
        """Batch a sequence of :class:`~canopy.contracts.OrientedBox`."""
        if not boxes:
            empty = np.zeros((0, 2), dtype=np.float64)
            return cls(empty, empty, empty, empty)
        centre = np.array([b.center for b in boxes], dtype=np.float64)
        size = np.array([b.size for b in boxes], dtype=np.float64)
        yaw = np.array([b.yaw for b in boxes], dtype=np.float64)
        return cls(
            centre=centre[:, :2],
            axis=np.stack([np.cos(yaw), np.sin(yaw)], axis=1),
            half=size[:, :2] / 2.0,
            z=np.stack([centre[:, 2] - size[:, 2] / 2.0, centre[:, 2] + size[:, 2] / 2.0], axis=1),
        )

    def corners(self) -> FloatArray:
        """Footprint corners, shape ``(K, 4, 2)``, in winding order."""
        local = _CORNER_SIGNS[None] * self.half[:, None, :]
        return (
            self.centre[:, None, :]
            + local[..., :1] * self.axis[:, None, :]
            + local[..., 1:] * _perp(self.axis)[:, None, :]
        )


def _point_rect(
    p: FloatArray, centre: FloatArray, axis: FloatArray, half: FloatArray
) -> FloatArray:
    """Plan distance from points to rectangles, zero inside; all arguments broadcast."""
    q = p - centre
    along = np.abs(_dot(q, axis)) - half[..., 0]
    across = np.abs(_dot(q, _perp(axis))) - half[..., 1]
    return np.hypot(np.maximum(along, 0.0), np.maximum(across, 0.0))


def point_gap(points: Points, boxes: Boxes) -> FloatArray:
    """Plan distance from each point to each box's footprint.

    Parameters
    ----------
    points
        Plan points, shape ``(P, 2)``.
    boxes
        ``B`` boxes.

    Returns
    -------
    numpy.ndarray
        Shape ``(B, P)``; zero where a point is inside a footprint.
    """
    return _point_rect(
        points[None, :, :],
        boxes.centre[:, None, :],
        boxes.axis[:, None, :],
        boxes.half[:, None, :],
    )


def box_gap(a: Boxes, b: Boxes, *, plan: bool = False) -> FloatArray:
    """Shortest distance between every box of ``a`` and every box of ``b``.

    Parameters
    ----------
    a, b
        The boxes to compare.
    plan
        Ignore height and return the footprint gap alone. A ``clear_of`` rule
        wants this for "not under a window" -- the vertical gap between a
        wall-mounted window and a ground-standing battery is never the
        binding constraint, so including it would let the battery sit
        directly beneath one.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(a), len(b))``, in metres; zero where two boxes touch or
        overlap.
    """
    # Axes: 0 = a, 1 = b, 2 = corner.
    ac, au, ah = a.centre[:, None, None], a.axis[:, None, None], a.half[:, None, None]
    bc, bu, bh = b.centre[None, :, None], b.axis[None, :, None], b.half[None, :, None]
    a_in_b = _point_rect(a.corners()[:, None], bc, bu, bh).min(axis=-1)
    b_in_a = _point_rect(b.corners()[None, :], ac, au, ah).min(axis=-1)
    footprint = np.minimum(a_in_b, b_in_a)

    # Separating axes: the edge directions of both rectangles.
    delta = (b.centre[None, :] - a.centre[:, None])[:, :, None, :]
    axes = np.stack(
        np.broadcast_arrays(
            a.axis[:, None], _perp(a.axis)[:, None], b.axis[None, :], _perp(b.axis)[None, :]
        ),
        axis=2,
    )  # (A, B, 4, 2)
    ra = ah[..., 0] * np.abs(_dot(au, axes)) + ah[..., 1] * np.abs(_dot(_perp(au), axes))
    rb = bh[..., 0] * np.abs(_dot(bu, axes)) + bh[..., 1] * np.abs(_dot(_perp(bu), axes))
    separated = np.abs(_dot(delta, axes)) > ra + rb
    footprint = np.where(separated.any(axis=-1), footprint, 0.0)
    if plan:
        return footprint

    rise = np.maximum(b.z[None, :, 0] - a.z[:, None, 1], a.z[:, None, 0] - b.z[None, :, 1])
    return np.hypot(footprint, np.maximum(rise, 0.0))
