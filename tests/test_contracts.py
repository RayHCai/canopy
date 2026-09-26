"""Contracts are an interface: changes here break every consumer."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.contracts import (
    CLASS_COLORS,
    Cls,
    DiscoveredObject,
    DroneState,
    Observation,
    Occ,
    OrientedBox,
    Scan,
)


def test_class_values_are_stable() -> None:
    """Manifest files store these values.

    They may therefore be appended to, but never reordered.
    """
    assert [(c.name, int(c)) for c in Cls] == [
        ("GROUND", 0),
        ("DRIVEWAY", 1),
        ("WALL", 2),
        ("ROOF", 3),
        ("DOOR", 4),
        ("WINDOW", 5),
        ("GARAGE_DOOR", 6),
        ("METER", 7),
        ("BUSH", 8),
        ("TREE", 9),
        ("FENCE", 10),
        ("AC_UNIT", 11),
        ("GAS_METER", 12),
        ("PANEL", 13),
        ("CONDUIT", 14),
        ("SHED", 15),
    ]


def test_occupancy_values_match_the_spec() -> None:
    assert (Occ.UNKNOWN, Occ.FREE, Occ.OCC) == (0, 1, 2)


def test_every_class_has_a_colour() -> None:
    """The reveal, the photo renderer and the viewer must agree on all of them."""
    assert set(CLASS_COLORS) == set(Cls)


def test_colours_are_valid_rgb() -> None:
    for cls, rgb in CLASS_COLORS.items():
        assert len(rgb) == 3, cls
        assert all(0 <= channel <= 255 for channel in rgb), cls


def test_array_dataclasses_have_no_generated_eq() -> None:
    """A generated __eq__ would raise on NumPy array comparison."""
    a = DroneState(drone_id=0, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0)
    b = DroneState(drone_id=0, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0)
    assert a != b  # identity comparison, and crucially it does not raise
    assert a == a  # noqa: PLR0124 -- identity comparison is the point


def test_slots_reject_typos() -> None:
    state = DroneState(drone_id=0, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0)
    with pytest.raises(AttributeError):
        state.postion = np.ones(3)  # type: ignore[attr-defined]


def test_scan_observation_drops_ids_and_shares_arrays() -> None:
    """observation() is what a real drone could have sensed: range and colour, no ids."""
    scan = Scan(
        drone_id=2,
        t=1.5,
        origin=np.array([1.0, 2.0, 3.0]),
        dirs=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        dist=np.array([2.0, np.inf]),
        obj_ids=np.array([5, -1], dtype=np.int32),
        tri_ids=np.array([9, -1], dtype=np.int32),
        rgb=np.array([[10, 20, 30], [184, 201, 218]], dtype=np.uint8),
    )

    obs = scan.observation()

    assert isinstance(obs, Observation)
    assert not hasattr(obs, "obj_ids")
    assert not hasattr(obs, "tri_ids")
    assert obs.drone_id == scan.drone_id
    assert obs.t == scan.t
    # Shared, not copied: perception gets exactly what the sim already built.
    assert obs.origin is scan.origin
    assert obs.dirs is scan.dirs
    assert obs.dist is scan.dist
    assert obs.rgb is scan.rgb


def test_oriented_box_corners_yaw_zero_matches_axis_aligned_box() -> None:
    """At yaw=0 the box's first size axis lies along +X, its second along +Y."""
    box = OrientedBox(center=np.array([1.0, 2.0, 3.0]), size=np.array([2.0, 4.0, 6.0]), yaw=0.0)

    corners = box.corners()

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
            [2.0, 0.0, 6.0],
            [0.0, 4.0, 6.0],
            [2.0, 4.0, 6.0],
        ]
    )
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners, expected, atol=1e-9)
    # Bottom four first.
    assert np.all(corners[:4, 2] < corners[4:, 2])


def test_oriented_box_corners_yaw_quarter_turn_swaps_length_and_width_axes() -> None:
    """At yaw=pi/2 the box's footprint is rotated a quarter turn about its centre."""
    box = OrientedBox(
        center=np.array([1.0, 2.0, 3.0]), size=np.array([2.0, 4.0, 6.0]), yaw=np.pi / 2.0
    )

    corners = box.corners()

    # The length axis (2 m, +X at yaw=0) now runs along +Y; the width axis
    # (4 m, +Y at yaw=0) now runs along -X. Height is unaffected by yaw.
    expected = np.array(
        [
            [3.0, 1.0, 0.0],
            [3.0, 3.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 3.0, 0.0],
            [3.0, 1.0, 6.0],
            [3.0, 3.0, 6.0],
            [-1.0, 1.0, 6.0],
            [-1.0, 3.0, 6.0],
        ]
    )
    np.testing.assert_allclose(corners, expected, atol=1e-9)


def test_oriented_box_is_frozen() -> None:
    box = OrientedBox(center=np.zeros(3), size=np.ones(3), yaw=0.0)
    with pytest.raises(AttributeError):
        box.yaw = 1.0  # type: ignore[misc]


def test_oriented_box_has_no_generated_eq() -> None:
    """A generated __eq__ would raise on NumPy array comparison, as for the other array types."""
    a = OrientedBox(center=np.zeros(3), size=np.ones(3), yaw=0.0)
    b = OrientedBox(center=np.zeros(3), size=np.ones(3), yaw=0.0)
    assert a != b  # identity comparison, and crucially it does not raise
    assert a == a  # noqa: PLR0124 -- identity comparison is the point


def test_observation_and_discovered_object_are_frozen() -> None:
    """What perception consumes and what it publishes cannot be edited in place.

    ``MapState.discovered`` is replaced, never mutated, and the viewer keeps
    references to published objects without copying them; freezing makes
    that a guarantee rather than a convention.
    """
    obs = Observation(
        drone_id=0, t=0.0, origin=np.zeros(3), dirs=np.zeros((0, 3)), dist=np.zeros(0)
    )
    found = DiscoveredObject(
        track_id=0,
        cls=Cls.METER,
        pos=np.zeros(3),
        box=OrientedBox(center=np.zeros(3), size=np.ones(3), yaw=0.0),
        n_hits=1,
        confidence=0.5,
        first_seen_t=0.0,
        first_seen_by=0,
    )
    with pytest.raises(AttributeError):
        obs.t = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        found.confidence = 1.0  # type: ignore[misc]
