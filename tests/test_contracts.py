"""Contracts are an interface: changes here break every consumer."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.contracts import CLASS_COLORS, Cls, DroneState, Occ


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
