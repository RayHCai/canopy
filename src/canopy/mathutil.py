"""Small vector and angle helpers shared across modules.

Deliberately tiny and dependency-free beyond NumPy. Anything that grows a
policy of its own belongs in the module that owns that policy, not here.
"""

from __future__ import annotations

import numpy as np

from canopy.contracts import Vec3

__all__ = ["braking_speed", "clamp_norm", "norm", "unit", "wrap_to_pi"]

#: Distances and speeds below this are treated as zero.
EPS = 1e-9


def norm(v: Vec3) -> float:
    """Euclidean length of ``v``."""
    return float(np.linalg.norm(v))


def unit(v: Vec3) -> Vec3:
    """Return ``v`` scaled to unit length, or a zero vector if ``v`` is ~zero."""
    length = norm(v)
    if length < EPS:
        return np.zeros_like(v)
    return np.asarray(v / length, dtype=np.float64)


def clamp_norm(v: Vec3, max_norm: float) -> Vec3:
    """Scale ``v`` down so its length is at most ``max_norm``."""
    length = norm(v)
    if length <= max_norm or length < EPS:
        return np.asarray(v, dtype=np.float64)
    return np.asarray(v * (max_norm / length), dtype=np.float64)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def braking_speed(distance: float, v_max: float, a_max: float) -> float:
    """Fastest speed from which ``a_max`` still stops the drone within ``distance``.

    Without this a waypoint follower that always commands ``v_max`` overshoots
    and orbits its target forever.
    """
    if distance <= EPS:
        return 0.0
    return float(min(v_max, np.sqrt(2.0 * a_max * distance)))
