"""Tests for :mod:`canopy.perception.colour`."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.perception.colour import ColourRule, Hsv, rgb_to_hsv


def _hsv_row(hue: float, sat: float, val: float, chroma: float) -> Hsv:
    """Build a single-colour :class:`Hsv` for exact boundary checks."""
    return Hsv(
        hue=np.array([hue], dtype=np.float64),
        sat=np.array([sat], dtype=np.float64),
        val=np.array([val], dtype=np.float64),
        chroma=np.array([chroma], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("rgb", "hue", "sat", "val", "chroma"),
    [
        ((255.0, 0.0, 0.0), 0.0, 1.0, 1.0, 1.0),
        ((0.0, 255.0, 0.0), 120.0, 1.0, 1.0, 1.0),
        ((0.0, 0.0, 255.0), 240.0, 1.0, 1.0, 1.0),
        ((255.0, 255.0, 0.0), 60.0, 1.0, 1.0, 1.0),
        ((128.0, 128.0, 128.0), 0.0, 0.0, 128.0 / 255.0, 0.0),
        ((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0),
        ((255.0, 255.0, 255.0), 0.0, 0.0, 1.0, 0.0),
    ],
    ids=["red", "green", "blue", "yellow", "grey", "black", "white"],
)
def test_rgb_to_hsv_known_colours(
    rgb: tuple[float, float, float], hue: float, sat: float, val: float, chroma: float
) -> None:
    """Pure primaries, grey, black and white convert to their textbook HSV values."""
    result = rgb_to_hsv(np.array([rgb], dtype=np.float64))
    assert result.hue[0] == pytest.approx(hue, abs=1e-9)
    assert result.sat[0] == pytest.approx(sat, abs=1e-9)
    assert result.val[0] == pytest.approx(val, abs=1e-9)
    assert result.chroma[0] == pytest.approx(chroma, abs=1e-9)
    assert 0.0 <= result.hue[0] < 360.0


def test_hue_and_channels_stay_in_contract_ranges_for_random_colours() -> None:
    """Hue lands in [0, 360) and sat/val/chroma in [0, 1] for arbitrary colours."""
    rng = np.random.default_rng(12345)
    rgb = rng.uniform(0.0, 255.0, size=(200, 3))
    hsv = rgb_to_hsv(rgb)
    assert np.all(hsv.hue >= 0.0)
    assert np.all(hsv.hue < 360.0)
    assert np.all(hsv.sat >= 0.0)
    assert np.all(hsv.sat <= 1.0)
    assert np.all(hsv.val >= 0.0)
    assert np.all(hsv.val <= 1.0)
    assert np.all(hsv.chroma >= 0.0)
    assert np.all(hsv.chroma <= 1.0)


def test_fractional_averaged_colour_is_accepted() -> None:
    """A pooled colour with fractional channels (an average of hits) converts cleanly."""
    pooled = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]]).mean(axis=0, keepdims=True)
    hsv = rgb_to_hsv(pooled)
    assert hsv.hue[0] == pytest.approx(60.0)
    assert hsv.sat[0] == pytest.approx(1.0)
    assert hsv.val[0] == pytest.approx(127.5 / 255.0)
    assert hsv.chroma[0] == pytest.approx(127.5 / 255.0)


@pytest.mark.parametrize(
    ("hue", "sat", "val", "expected"),
    [
        (10.0, 0.5, 0.5, True),
        (20.0, 0.5, 0.5, True),
        (9.999, 0.5, 0.5, False),
        (20.001, 0.5, 0.5, False),
        (15.0, 0.2, 0.5, True),
        (15.0, 0.8, 0.5, True),
        (15.0, 0.1999, 0.5, False),
        (15.0, 0.8001, 0.5, False),
        (15.0, 0.5, 0.3, True),
        (15.0, 0.5, 0.9, True),
        (15.0, 0.5, 0.2999, False),
        (15.0, 0.5, 0.9001, False),
    ],
    ids=[
        "hue_lo",
        "hue_hi",
        "hue_below_lo",
        "hue_above_hi",
        "sat_lo",
        "sat_hi",
        "sat_below_lo",
        "sat_above_hi",
        "val_lo",
        "val_hi",
        "val_below_lo",
        "val_above_hi",
    ],
)
def test_exact_rule_bounds_are_inclusive(
    hue: float, sat: float, val: float, expected: bool
) -> None:
    """At tolerance 0, each dimension's bounds admit exactly and reject just outside."""
    rule = ColourRule(hue_deg=(10.0, 20.0), sat=(0.2, 0.8), val=(0.3, 0.9))
    admitted = rule.admits(_hsv_row(hue, sat, val, chroma=0.5), tolerance=0.0)
    assert bool(admitted[0]) == expected


def test_wrapping_hue_arc_admits_across_zero_and_rejects_far_side() -> None:
    """A hue arc written lo > hi wraps through 0 and still excludes the far side."""
    rule = ColourRule(hue_deg=(340.0, 20.0), sat=(0.0, 1.0), val=(0.0, 1.0))
    hsv = Hsv(
        hue=np.array([350.0, 10.0, 180.0]),
        sat=np.full(3, 0.5),
        val=np.full(3, 0.5),
        chroma=np.full(3, 0.5),
    )
    admitted = rule.admits(hsv, tolerance=0.0)
    assert np.array_equal(admitted, np.array([True, True, False]))


def test_full_span_hue_admits_every_hue() -> None:
    """A 0-360 degree arc admits any hue at all."""
    rule = ColourRule(hue_deg=(0.0, 360.0), sat=(0.0, 1.0), val=(0.0, 1.0))
    hues = np.array([0.0, 45.0, 180.0, 270.0, 359.999])
    hsv = Hsv(hue=hues, sat=np.full(5, 0.5), val=np.full(5, 0.5), chroma=np.full(5, 0.5))
    admitted = rule.admits(hsv, tolerance=0.0)
    assert np.all(admitted)


def test_pale_colour_hue_slack_grows_with_noise_and_needs_nonzero_tolerance() -> None:
    """A low-chroma colour just past the arc needs both noise and tolerance to be admitted."""
    rule = ColourRule(hue_deg=(100.0, 140.0), sat=(0.0, 1.0), val=(0.0, 1.0))
    hsv = Hsv(
        hue=np.full(2, 145.0),  # 5 degrees past the arc's upper bound
        sat=np.full(2, 0.5),
        val=np.full(2, 0.5),
        chroma=np.full(2, 0.1),  # pale: little chroma to pin down hue
    )

    exact = rule.admits(hsv, tolerance=0.0)
    assert not bool(exact[0])  # exact rule: zero slack, whatever the noise would be

    # hue_slack = 60 * sqrt(2) * (tolerance * noise) / chroma.
    per_row = rule.admits(hsv, noise=np.array([0.01, 0.0001]), tolerance=1.0)
    assert bool(per_row[0])  # slack ~8.5 deg covers the 5 deg overshoot
    assert not bool(per_row[1])  # slack ~0.08 deg does not


def test_zero_chroma_and_zero_value_admit_without_numpy_warnings() -> None:
    """Grey (chroma 0) and black (value 0) rows run clean under filterwarnings=error."""
    rule = ColourRule(hue_deg=(170.0, 190.0), sat=(0.4, 0.6), val=(0.4, 0.6))
    hsv = Hsv(
        hue=np.array([0.0, 0.0]),
        sat=np.array([0.0, 0.0]),
        val=np.array([0.0, 0.5]),
        chroma=np.array([0.0, 0.0]),
    )
    # A warning here (e.g. from dividing by zero chroma or value) would fail the
    # test outright under filterwarnings=error, so reaching the assertions below
    # already proves the guarded divides in ``admits`` never trip one.
    admitted = rule.admits(hsv, noise=0.05, tolerance=2.0)
    assert admitted.dtype == np.bool_
    assert admitted.shape == (2,)
    # Both fail on value alone: hue and sat are unconstrained for a colour with
    # no chroma or no value, but neither row's value bound is close enough.
    assert np.array_equal(admitted, np.array([False, False]))
