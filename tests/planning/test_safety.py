"""Tests for :class:`canopy.planning.safety.Shield` landing behaviour."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from canopy.config import Config
from canopy.contracts import DroneState, Vec3
from canopy.planning.safety import ClearanceMap, Shield


def _state(
    drone_id: int, pos: Sequence[float], vel: Sequence[float] = (0.0, 0.0, 0.0)
) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        pos=np.array(pos, dtype=np.float64),
        vel=np.array(vel, dtype=np.float64),
        yaw=0.0,
    )


def test_landing_flag_lets_a_drone_descend_onto_the_ground(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    """Straight down onto the ground is held ``"map"`` normally, approved when landing."""
    cm = clearance()
    shield = Shield(cfg.safety, cfg.sim)
    drone = _state(0, (0.0, 0.0, 0.3))
    target: Vec3 = np.array([0.0, 0.0, 0.0])

    grounded = shield.filter([drone], {0: target}, cm, landing=[])
    assert grounded.held.get(0) == "map"
    assert 0 not in grounded.targets

    landing = shield.filter([drone], {0: target}, cm, landing=[0])
    assert 0 not in landing.held
    np.testing.assert_array_equal(landing.targets[0], target)


def test_landing_drone_still_holds_for_a_nearby_drone(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    """The drone-to-drone check still applies to a landing drone."""
    cm = clearance()
    shield = Shield(cfg.safety, cfg.sim)
    landing_drone = _state(0, (0.0, 0.0, 0.3))
    # Well inside min_separation_m (1.5 m) of where the landing drone is headed.
    parked = _state(1, (0.0, 0.0, 0.05))
    target: Vec3 = np.array([0.0, 0.0, 0.0])

    verdict = shield.filter([landing_drone, parked], {0: target}, cm, landing=[0])
    assert verdict.held.get(0) == "drone 1"
    assert 0 not in verdict.targets


def test_landing_flag_has_no_effect_without_a_nearby_conflict(
    cfg: Config, clearance: Callable[..., ClearanceMap]
) -> None:
    """A lone drone not landing straight down is unaffected by the landing flag."""
    cm = clearance()
    shield = Shield(cfg.safety, cfg.sim)
    drone = _state(0, (0.0, 0.0, 3.0))
    target: Vec3 = np.array([1.0, 1.0, 3.0])

    normal = shield.filter([drone], {0: target}, cm, landing=[])
    landing = shield.filter([drone], {0: target}, cm, landing=[0])
    assert 0 in normal.targets
    assert 0 in landing.targets


def test_segment_clear_catches_a_corner_the_old_half_voxel_sampling_missed() -> None:
    """A segment that clips only a voxel's corner must still be refused.

    The old algorithm sampled ``start -> end`` at ``linspace(0, 1, n)`` with
    ``n = ceil(length / (0.5 * voxel)) + 1``. For the segment below that steps
    from voxel ``(0, 1, 1)`` straight to ``(1, 0, 1)``, skipping the unsafe
    corner voxel ``(1, 1, 1)`` entirely -- confirmed inline below by
    reproducing that exact formula. ``sample_segment``'s extra samples, one
    between each pair of consecutive voxel-boundary crossings, must catch it.
    """
    voxel = 1.0
    field = np.full((4, 4, 4), 10.0)
    field[1, 1, 1] = 0.0  # the one unsafe voxel, clipped only at its corner
    origin = np.zeros(3)
    geofence = np.array([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]])
    cm = ClearanceMap(field, origin, voxel, geofence)

    start = np.array([0.5, 1.6, 1.5])
    end = np.array([1.5, 0.5, 1.5])

    # Reproduce the old sampling exactly, to confirm it would have missed the
    # corner voxel and called this segment clear.
    length = float(np.linalg.norm(end - start))
    n_old = max(2, int(np.ceil(length / (0.5 * voxel))) + 1)
    t_old = np.linspace(0.0, 1.0, n_old)
    old_samples = start + t_old[:, None] * (end - start)
    old_idx = np.floor((old_samples - origin) / voxel).astype(np.intp)
    assert not np.any(np.all(old_idx == np.array([1, 1, 1]), axis=1)), (
        "test setup is wrong: the old sampling should have missed voxel (1, 1, 1)"
    )
    assert np.all(cm.clearance_at(old_samples) >= 1.0), "old samples must all read safe"

    assert cm.segment_clear(start, end, 1.0) is False


def test_segment_clear_prefixes_of_a_clear_segment_stay_clear() -> None:
    """Any prefix of a segment ``segment_clear`` accepts is itself accepted.

    Restricted to segments that start outside the recovery margin (the shield's
    normal case): recovery segments have their own "no worse than the best
    reached so far" rule, which is not expected to survive truncation -- a
    prefix may simply not have reached the gain yet.

    Also checks that :meth:`ClearanceMap.sample_segment` visits voxels in
    order, 26-connected step to step, since a skipped or teleporting sample
    would let a segment slip through an unsafe voxel undetected.
    """
    rng = np.random.default_rng(0)
    shape = (10, 10, 6)
    voxel = 0.3
    origin = np.zeros(3)
    # Biased toward higher clearance so a healthy fraction of random segments
    # are fully clear; a bounded search then collects a fixed target count,
    # rather than a fixed number of draws that can run short by chance.
    field = rng.uniform(0.3, 1.5, size=shape)
    geofence = np.array([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]])
    cm = ClearanceMap(field, origin, voxel, geofence)

    lo = origin
    hi = origin + np.array(shape) * voxel
    target, max_attempts = 60, 5000
    n_checked = 0
    attempts = 0
    while n_checked < target and attempts < max_attempts:
        attempts += 1
        a = rng.uniform(lo, hi)
        b = rng.uniform(lo, hi)
        required = float(rng.uniform(0.1, 0.8))
        if cm.clearance_at(a[None, :])[0] < required:
            continue  # recovery case: prefix invariant does not apply
        if not cm.segment_clear(a, b, required):
            continue
        n_checked += 1

        for s in (0.1, 0.4, 0.6, 0.9):
            prefix_end = a + s * (b - a)
            assert cm.segment_clear(a, prefix_end, required)

        samples = cm.sample_segment(a, b)
        idx = np.floor((samples - origin) / voxel).astype(np.intp)
        steps = np.diff(idx, axis=0)
        assert np.all(np.abs(steps) <= 1), "consecutive samples must be 26-connected"
        direction = b - a
        projection = (samples - a) @ direction
        assert np.all(np.diff(projection) >= -1e-9), "samples must be ordered along the segment"

    assert n_checked == target, (
        f"only found {n_checked} clear segments in {attempts} attempts; "
        "field is not producing enough clear segments to be meaningful"
    )
