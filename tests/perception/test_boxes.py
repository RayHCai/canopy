"""Tests for :mod:`canopy.perception.boxes`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from canopy.contracts import OrientedBox, Points
from canopy.perception.boxes import fit_box, pad_box, split_parts


def _rect_points(
    length: float, width: float, height: float, yaw: float, centre: tuple[float, float, float]
) -> Points:
    """Build a dense grid that exactly fills an upright ``length x width x height`` box, rotated."""
    u = np.linspace(-length / 2.0, length / 2.0, 5)
    v = np.linspace(-width / 2.0, width / 2.0, 3)
    z = np.linspace(centre[2] - height / 2.0, centre[2] + height / 2.0, 4)
    uu, vv, zz = np.meshgrid(u, v, z, indexing="ij")
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    xx = uu * cos_y - vv * sin_y + centre[0]
    yy = uu * sin_y + vv * cos_y + centre[1]
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _to_local_frame(points: Points, box: OrientedBox) -> npt.NDArray[np.float64]:
    """Project world points into ``box``'s own centred, unrotated frame."""
    cos_y, sin_y = np.cos(box.yaw), np.sin(box.yaw)
    rel = np.asarray(points, dtype=np.float64) - box.center
    u = rel[:, 0] * cos_y + rel[:, 1] * sin_y
    v = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
    return np.column_stack([u, v, rel[:, 2]])


def _cross_points() -> Points:
    """Two straight 30-point lines crossing at the origin, along X and Y."""
    arm = np.linspace(0.2, 3.0, 15)
    zeros = np.zeros(15)
    return np.vstack(
        [
            np.column_stack([arm, zeros, zeros]),
            np.column_stack([-arm, zeros, zeros]),
            np.column_stack([zeros, arm, zeros]),
            np.column_stack([zeros, -arm, zeros]),
        ]
    )


@pytest.mark.parametrize("true_yaw", [0.0, 0.4, -0.9, 2.0])
def test_fit_box_recovers_size_centre_and_yaw_of_a_rotated_rectangle(true_yaw: float) -> None:
    """A dense rotated rectangle's fitted box recovers its size, centre and canonical yaw."""
    length, width, height = 4.0, 1.5, 2.0
    centre = (3.0, -2.0, 1.0)
    box = fit_box(_rect_points(length, width, height, true_yaw, centre))

    assert box.size[0] >= box.size[1]
    assert box.size[0] == pytest.approx(length, abs=1e-4)
    assert box.size[1] == pytest.approx(width, abs=1e-4)
    assert box.size[2] == pytest.approx(height, abs=1e-4)
    assert np.allclose(box.center, centre, atol=1e-4)

    expected_yaw = ((true_yaw + np.pi / 2.0) % np.pi) - np.pi / 2.0
    assert box.yaw == pytest.approx(expected_yaw, abs=1e-4)
    assert -np.pi / 2.0 <= box.yaw < np.pi / 2.0


def test_corners_are_bottom_four_first_and_enclose_every_input_point() -> None:
    """corners() reports the bottom face first, and its extent bounds every fitted point."""
    points = _rect_points(4.0, 1.5, 2.0, 0.6, (1.0, 2.0, 0.5))
    box = fit_box(points)
    corners = box.corners()
    assert corners.shape == (8, 3)

    local_corners = _to_local_frame(corners, box)
    half = box.size / 2.0
    assert np.allclose(np.abs(local_corners), half, atol=1e-6)
    assert np.all(local_corners[:4, 2] < 0.0)  # bottom four first
    assert np.all(local_corners[4:, 2] > 0.0)

    local_points = _to_local_frame(points, box)
    assert np.all(np.abs(local_points) <= half + 1e-6)


@pytest.mark.parametrize(
    "points",
    [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]),
    ],
    ids=["single_point", "vertical_line", "collinear"],
)
def test_fit_box_on_degenerate_inputs_is_finite(points: Points) -> None:
    """A single point, a vertical run, or collinear points give a finite box and no exception."""
    box = fit_box(points)
    assert np.all(np.isfinite(box.center))
    assert np.all(np.isfinite(box.size))
    assert np.isfinite(box.yaw)
    assert np.all(box.size >= 0.0)


@pytest.mark.parametrize(
    ("center_z", "size_z", "yaw", "ground_snap_m", "expected_center_z", "expected_size_z"),
    [
        (5.0, 3.0, 0.3, 0.1, 5.0, 3.4),
        (0.25, 0.4, -0.6, 0.5, 0.325, 0.65),
    ],
    ids=["far_from_ground_unsnapped", "close_to_ground_snapped_to_zero"],
)
def test_pad_box_grows_sides_and_snaps_bottom_only_when_close(
    center_z: float,
    size_z: float,
    yaw: float,
    ground_snap_m: float,
    expected_center_z: float,
    expected_size_z: float,
) -> None:
    """pad_box grows every side by the margin, keeps yaw, and only zeroes a nearby bottom."""
    margin = 0.2
    box = OrientedBox(
        center=np.array([1.0, -2.0, center_z]), size=np.array([2.0, 1.0, size_z]), yaw=yaw
    )
    padded = pad_box(box, margin_m=margin, ground_snap_m=ground_snap_m)

    assert padded.yaw == pytest.approx(yaw)
    assert padded.center[0] == pytest.approx(1.0)
    assert padded.center[1] == pytest.approx(-2.0)
    assert padded.size[0] == pytest.approx(2.0 + 2.0 * margin)
    assert padded.size[1] == pytest.approx(1.0 + 2.0 * margin)
    assert padded.center[2] == pytest.approx(expected_center_z, abs=1e-9)
    assert padded.size[2] == pytest.approx(expected_size_z, abs=1e-9)


def test_split_parts_indices_partition_the_input() -> None:
    """Every index from 0..N-1 appears in exactly one returned part, whatever the shape."""
    rng = np.random.default_rng(11)
    points = rng.uniform(-1.0, 1.0, size=(50, 3))
    parts = split_parts(points, floor_m=0.05, min_saving=0.1, min_points=3, max_parts=5)
    all_idx = np.concatenate(parts)
    assert np.array_equal(np.sort(all_idx), np.arange(len(points)))


def test_l_shape_splits_into_two_parts_that_separate_the_legs() -> None:
    """A vertical riser plus a horizontal run at its top splits into its two legs."""
    riser = np.column_stack([np.zeros(26), np.zeros(26), np.linspace(0.0, 5.0, 26)])
    run = np.column_stack([np.linspace(0.0, 2.0, 11), np.zeros(11), np.full(11, 5.0)])
    points = np.vstack([riser, run])

    parts = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=3, max_parts=3)

    assert len(parts) == 2
    all_idx = np.concatenate(parts)
    assert np.array_equal(np.sort(all_idx), np.arange(len(points)))
    part_of = {int(idx): i for i, part in enumerate(parts) for idx in part}
    assert part_of[0] != part_of[len(points) - 1]  # riser's foot vs. the run's far end


@pytest.mark.parametrize("shape_name", ["compact_blob", "straight_pipe"])
def test_compact_and_straight_shapes_are_not_split(shape_name: str) -> None:
    """A round, compact object and a straight run both stay a single part."""
    if shape_name == "compact_blob":
        rng = np.random.default_rng(3)
        phi = rng.uniform(0.0, np.pi, 200)
        theta = rng.uniform(0.0, 2.0 * np.pi, 200)
        radius = 0.5
        points = np.column_stack(
            [
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi) + 1.0,
            ]
        )
    else:
        rng = np.random.default_rng(4)
        z = np.linspace(0.0, 5.0, 40)
        jitter = rng.uniform(-0.02, 0.02, size=(40, 2))
        points = np.column_stack([jitter[:, 0], jitter[:, 1], z])

    parts = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=3, max_parts=3)
    assert len(parts) == 1
    assert np.array_equal(np.sort(parts[0]), np.arange(len(points)))


def test_max_parts_caps_the_number_of_parts() -> None:
    """max_parts stops recursion even though a further cut still saves enough volume."""
    points = _cross_points()
    capped = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=3, max_parts=2)
    uncapped = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=3, max_parts=8)

    assert len(capped) == 2
    assert len(uncapped) == 3


def test_min_points_blocks_a_cut_that_would_make_a_too_small_part() -> None:
    """Raising min_points suppresses a cut that would otherwise leave a smaller part."""
    points = _cross_points()
    permissive = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=8, max_parts=8)
    strict = split_parts(points, floor_m=0.05, min_saving=0.4, min_points=16, max_parts=8)

    assert sorted(len(p) for p in permissive) == [15, 15, 30]
    assert sorted(len(p) for p in strict) == [30, 30]
    assert all(len(p) >= 16 for p in strict)
