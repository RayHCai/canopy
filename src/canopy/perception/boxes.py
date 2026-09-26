"""Tight upright boxes around observed points.

A box is fitted in plan view as the minimum-area rectangle around the points
(the rectangle's orientation is one of the convex hull's edge directions, so
trying each edge is exact), then extruded over the points' height. That hugs
a meter flush with a wall or a conduit run along one, whatever the wall's
bearing, where an axis-aligned box around a wall at 30 degrees would be mostly
air.

One box is only tight around a compact shape. A conduit that rises up a wall
and then turns into the meter is an L, and one box around it is mostly air
beside the riser. :func:`split_parts` cuts such a shape into straight parts,
each boxed on its own, and leaves compact shapes whole: cutting a bush or a
meter in two never shrinks the total box volume much, so it is never worth it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, QhullError

from canopy.contracts import OrientedBox, Points

__all__ = ["fit_box", "pad_box", "split_parts"]

#: A box is symmetric under a half turn, so yaw is reported in [-pi/2, pi/2).
_HALF_TURN = np.pi
#: Fewest points that can span an area, below which a hull is not attempted.
_MIN_HULL_POINTS = 3
#: Fewest points that define a direction.
_MIN_DIRECTION_POINTS = 2


def fit_box(points: Points) -> OrientedBox:
    """Fit the tightest upright box around ``points``.

    Parameters
    ----------
    points
        Observed surface points, shape ``(N, 3)``, ``N >= 1``.

    Returns
    -------
    OrientedBox
        Its first ``size`` axis is the longer horizontal side, and ``yaw`` is
        that side's heading in ``[-pi/2, pi/2)``. Degenerate inputs (one point,
        a vertical line, a flat patch) give zero extents rather than failing.
    """
    pts = np.asarray(points, dtype=np.float64)
    xy = pts[:, :2]
    angle = _min_area_angle(xy)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u = xy[:, 0] * cos_a + xy[:, 1] * sin_a
    v = -xy[:, 0] * sin_a + xy[:, 1] * cos_a
    length, width = float(np.ptp(u)), float(np.ptp(v))
    mid_u, mid_v = (u.max() + u.min()) / 2.0, (v.max() + v.min()) / 2.0
    if width > length:
        length, width = width, length
        angle += np.pi / 2.0
    yaw = float(np.mod(angle + _HALF_TURN / 2.0, _HALF_TURN) - _HALF_TURN / 2.0)

    z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    center = np.array(
        [mid_u * cos_a - mid_v * sin_a, mid_u * sin_a + mid_v * cos_a, (z_lo + z_hi) / 2.0]
    )
    return OrientedBox(center=center, size=np.array([length, width, z_hi - z_lo]), yaw=yaw)


def pad_box(box: OrientedBox, margin_m: float, ground_snap_m: float) -> OrientedBox:
    """Grow ``box`` by ``margin_m`` on every side, and stand it on the ground if it is close.

    Observed points lie on an object's surface, so a box fitted to them cuts
    through the outermost surface; the margin moves the outline just outside
    it. Evidence is never collected right at ground level (the lawn would
    swamp it), so an object standing on the ground always looks a little
    short; a bottom within ``ground_snap_m`` of ``z = 0`` is extended to it.

    Parameters
    ----------
    box
        From :func:`fit_box`.
    margin_m
        Clearance added on each side, metres.
    ground_snap_m
        Largest gap between the padded bottom and the ground that is closed.

    Returns
    -------
    OrientedBox
        Same yaw; larger size, and a centre moved only by the ground snap.
    """
    size = np.asarray(box.size, dtype=np.float64) + 2.0 * margin_m
    center = np.asarray(box.center, dtype=np.float64).copy()
    bottom = center[2] - size[2] / 2.0
    top = center[2] + size[2] / 2.0
    if bottom <= ground_snap_m:
        bottom = 0.0
    center[2] = (bottom + top) / 2.0
    size[2] = top - bottom
    return OrientedBox(center=center, size=size, yaw=box.yaw)


def split_parts(
    points: Points, *, floor_m: float, min_saving: float, min_points: int, max_parts: int
) -> list[npt.NDArray[np.intp]]:
    """Cut a group of points into parts that box far more tightly than the whole.

    Greedy and recursive: the part whose best single cut saves the most box
    volume is cut, until no cut saves at least ``min_saving`` of its part's
    volume or there are ``max_parts`` parts. A cut is a plane square to the
    part's vertical or to one of its principal horizontal directions, so an L
    along a wall splits at its corner, and each side's extents are running
    minima and maxima over the points in cut order, which tries every cut
    position at once.

    Parameters
    ----------
    points
        The group's points, shape ``(N, 3)``.
    floor_m
        Added to every extent before volumes are compared, so a flat or
        one-cell-thin part still has a volume and a cut through nothing but
        air is not rewarded. The evidence cell size suits.
    min_saving
        Fraction of a part's box volume a cut must save, in ``(0, 1)``.
    min_points
        Fewest points either side of a cut may keep.
    max_parts
        Most parts to return.

    Returns
    -------
    list of numpy.ndarray
        Index arrays into ``points``, one per part; a single part holding
        every index when no cut is worth making.
    """
    parts = [np.arange(len(points))]
    cuts = [_best_cut(points, floor_m, min_points)]
    while len(parts) < max_parts:
        best = max(range(len(parts)), key=lambda i: cuts[i][0])
        if cuts[best][0] < min_saving:
            break
        part = parts.pop(best)
        _, side = cuts.pop(best)
        for piece in (part[side], part[~side]):
            parts.append(piece)
            cuts.append(_best_cut(points[piece], floor_m, min_points))
    return parts


def _best_cut(
    points: Points, floor_m: float, min_points: int
) -> tuple[float, npt.NDArray[np.bool_]]:
    """Return the best single cut of ``points``: its volume saving and one side's mask."""
    none = (0.0, np.zeros(len(points), dtype=np.bool_))
    if len(points) < 2 * min_points:
        return none
    # Only the cut planes' orientation is needed here, not the tightest box,
    # so the principal horizontal direction stands in for a hull fit.
    heading = _principal_angle(points[:, :2])
    cos_y, sin_y = np.cos(heading), np.sin(heading)
    # The points in that frame, where every cut is axis-aligned.
    local = np.column_stack(
        [
            points[:, 0] * cos_y + points[:, 1] * sin_y,
            -points[:, 0] * sin_y + points[:, 1] * cos_y,
            points[:, 2],
        ]
    )
    whole = float(np.prod(np.ptp(local, axis=0) + floor_m))
    best_total, best_side = np.inf, none[1]
    for axis in range(3):
        order = np.argsort(local[:, axis], kind="stable")
        ranked = local[order]
        head = np.prod(
            np.maximum.accumulate(ranked) - np.minimum.accumulate(ranked) + floor_m, axis=1
        )
        tail = np.prod(
            np.maximum.accumulate(ranked[::-1]) - np.minimum.accumulate(ranked[::-1]) + floor_m,
            axis=1,
        )[::-1]
        total = head[:-1] + tail[1:]  # cutting between ranks k and k + 1
        k = np.arange(len(points) - 1)
        allowed = (
            (ranked[1:, axis] > ranked[:-1, axis])
            & (k + 1 >= min_points)
            & (len(points) - k - 1 >= min_points)
        )
        if not np.any(allowed):
            continue
        cut = int(np.argmin(np.where(allowed, total, np.inf)))
        if total[cut] < best_total:
            best_total = float(total[cut])
            best_side = np.zeros(len(points), dtype=np.bool_)
            best_side[order[: cut + 1]] = True
    if not np.isfinite(best_total):
        return none
    return 1.0 - best_total / whole, best_side


def _min_area_angle(xy: Points) -> float:
    """Rotation (radians) whose axes bound ``xy`` in the least area.

    The minimum-area rectangle has a side along some convex-hull edge, so only
    the hull's edge directions need trying, all at once. Fewer than three
    distinct points, or points in a line, have no hull; their principal
    direction is the answer then.
    """
    unique = np.unique(np.round(xy, 6), axis=0)
    if len(unique) < _MIN_HULL_POINTS:
        return _principal_angle(unique)
    try:
        hull = unique[ConvexHull(unique).vertices]
    except QhullError:  # collinear: no area to minimise
        return _principal_angle(unique)
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2.0)
    cos_a, sin_a = np.cos(angles)[:, None], np.sin(angles)[:, None]
    u = hull[:, 0][None, :] * cos_a + hull[:, 1][None, :] * sin_a
    v = -hull[:, 0][None, :] * sin_a + hull[:, 1][None, :] * cos_a
    area = np.ptp(u, axis=1) * np.ptp(v, axis=1)
    return float(angles[int(np.argmin(area))])


def _principal_angle(xy: Points) -> float:
    """Heading of the direction ``xy`` spreads along most; 0 for a single point."""
    if len(xy) < _MIN_DIRECTION_POINTS:
        return 0.0
    centred = xy - xy.mean(axis=0)
    _, vecs = np.linalg.eigh(centred.T @ centred)
    major = vecs[:, -1]
    return float(np.arctan2(major[1], major[0]))
