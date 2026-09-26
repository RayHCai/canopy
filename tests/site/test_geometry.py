"""Clearances between upright boxes (:mod:`canopy.site.geometry`), pinned by hand-computed cases.

``box_gap`` and ``point_gap`` are the one measurement every placement rule
needs, so their exactness matters more than their internals: every case here
is a footprint and height whose gap was worked out on paper, not read back
from the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from canopy.contracts import OrientedBox, Vec3
from canopy.site import Boxes, box_gap, point_gap


def _box(
    center: tuple[float, float, float], size: tuple[float, float, float], yaw: float = 0.0
) -> OrientedBox:
    return OrientedBox(
        center=np.array(center, dtype=np.float64),
        size=np.array(size, dtype=np.float64),
        yaw=yaw,
    )


def _boxes(*specs: OrientedBox) -> Boxes:
    return Boxes.from_oriented(list(specs))


def test_axis_aligned_boxes_one_metre_apart_in_x() -> None:
    a = _boxes(_box((0.0, 0.0, 1.0), (2.0, 2.0, 2.0)))
    b = _boxes(_box((3.0, 0.0, 1.0), (2.0, 2.0, 2.0)))
    assert box_gap(a, b)[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_overlapping_boxes_have_zero_gap() -> None:
    a = _boxes(_box((0.0, 0.0, 1.0), (2.0, 2.0, 2.0)))
    b = _boxes(_box((1.0, 0.0, 1.0), (2.0, 2.0, 2.0)))
    assert box_gap(a, b)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_vertical_separation_only() -> None:
    # Same footprint (fully overlapping in plan), disjoint height bands 2 m apart.
    a = _boxes(_box((0.0, 0.0, 0.5), (2.0, 2.0, 1.0)))
    b = _boxes(_box((0.0, 0.0, 4.0), (2.0, 2.0, 2.0)))
    assert box_gap(a, b)[0, 0] == pytest.approx(2.0, abs=1e-9)


def test_diagonal_plan_and_height_gap_is_a_hypotenuse() -> None:
    # 1 m plan gap (as in the x-separation case) combined with a 2 m vertical gap.
    a = _boxes(_box((0.0, 0.0, 0.5), (2.0, 2.0, 1.0)))
    b = _boxes(_box((3.0, 0.0, 4.0), (2.0, 2.0, 2.0)))
    assert box_gap(a, b)[0, 0] == pytest.approx(math.hypot(1.0, 2.0), abs=1e-9)


def test_rotated_box_corner_to_face_gap() -> None:
    # B is a unit-half-extent square turned 45 degrees, so its nearest corner to
    # A sits sqrt(2) * half from its own centre, straight along -X from A's face.
    d = 2.0 + math.sqrt(2.0)
    a = _boxes(_box((0.0, 0.0, 1.0), (2.0, 2.0, 2.0)))
    b = _boxes(_box((d, 0.0, 1.0), (2.0, 2.0, 2.0), yaw=math.pi / 4.0))
    assert box_gap(a, b)[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_crossing_plus_shape_has_zero_gap_via_separating_axis() -> None:
    # A long thin horizontal bar and a long thin vertical bar overlapping in a
    # "plus": neither rectangle's corners land inside the other, so only the
    # separating-axis test (not a corner-in-rectangle check) can find the overlap.
    a = _boxes(_box((0.0, 0.0, 1.0), (6.0, 1.0, 2.0)))
    b = _boxes(_box((0.0, 0.0, 1.0), (1.0, 6.0, 2.0), yaw=math.pi / 2.0))
    assert box_gap(a, b)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_point_gap_inside_is_zero() -> None:
    boxes = _boxes(_box((0.0, 0.0, 1.0), (4.0, 2.0, 2.0)))
    points: Vec3 = np.array([[0.5, 0.5]])
    assert point_gap(points, boxes)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_point_gap_outside_is_exact() -> None:
    # Box half-extents (2, 1) centred at the origin; a point at (5, 3) is beyond
    # both extents, so the gap is the hypotenuse of the two overshoots.
    boxes = _boxes(_box((0.0, 0.0, 1.0), (4.0, 2.0, 2.0)))
    points: Vec3 = np.array([[5.0, 3.0]])
    expected = math.hypot(5.0 - 2.0, 3.0 - 1.0)
    assert point_gap(points, boxes)[0, 0] == pytest.approx(expected, abs=1e-9)
