"""Vector and angle helpers."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.mathutil import braking_speed, clamp_norm, norm, unit, wrap_to_pi


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (np.pi / 2, np.pi / 2),
        (-np.pi / 2, -np.pi / 2),
        (3 * np.pi, -np.pi),
        (2 * np.pi, 0.0),
        (-3 * np.pi, -np.pi),
    ],
)
def test_wrap_to_pi(angle: float, expected: float) -> None:
    assert wrap_to_pi(angle) == pytest.approx(expected)


def test_wrap_to_pi_stays_in_range() -> None:
    for angle in np.linspace(-20.0, 20.0, 401):
        assert -np.pi <= wrap_to_pi(float(angle)) < np.pi


def test_unit_normalises() -> None:
    assert norm(unit(np.array([3.0, 4.0, 0.0]))) == pytest.approx(1.0)


def test_unit_of_zero_is_zero_not_nan() -> None:
    """Division by a near-zero length would poison every downstream waypoint."""
    result = unit(np.zeros(3))
    assert np.all(np.isfinite(result))
    assert norm(result) == pytest.approx(0.0)


def test_clamp_norm_shortens_only_when_needed() -> None:
    long = np.array([10.0, 0.0, 0.0])
    assert norm(clamp_norm(long, 2.0)) == pytest.approx(2.0)
    short = np.array([0.5, 0.0, 0.0])
    assert clamp_norm(short, 2.0) == pytest.approx(short)


def test_clamp_norm_preserves_direction() -> None:
    v = np.array([1.0, 2.0, -2.0])
    clamped = clamp_norm(v, 1.0)
    assert unit(clamped) == pytest.approx(unit(v))


def test_braking_speed_is_capped_by_v_max() -> None:
    assert braking_speed(1000.0, v_max=3.0, a_max=4.0) == pytest.approx(3.0)


def test_braking_speed_goes_to_zero_at_the_target() -> None:
    assert braking_speed(0.0, v_max=3.0, a_max=4.0) == pytest.approx(0.0)


def test_braking_speed_allows_stopping_within_the_distance() -> None:
    """From speed v, stopping distance is v^2 / 2a; it must not exceed what is left."""
    a_max = 4.0
    for distance in (0.05, 0.2, 1.0, 5.0):
        v = braking_speed(distance, v_max=100.0, a_max=a_max)
        assert v**2 / (2 * a_max) == pytest.approx(distance)
