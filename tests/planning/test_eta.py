"""The mapping ETA: projecting the frontier count's drain to zero.

Driven with synthetic frontier counts rather than a real mission, so each test
pins one behaviour of the estimator in isolation and runs instantly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from canopy.planning import MappingEta
from canopy.planning.mission import Phase

#: Control tick, matching the shipped ``sim.control_hz`` of 20.
_DT = 0.05


def _feed(
    eta: MappingEta, t0: float, t1: float, count: Callable[[float], float] | int, phase: Phase
) -> float:
    """Tick ``eta`` from ``t0`` to ``t1`` with ``count`` frontiers; return the last ``t``."""
    t = t0
    for step in np.arange(t0, t1, _DT):
        t = float(step)
        n = count if isinstance(count, int) else round(count(t))
        eta.update(t, n, phase)
    return t


def test_no_estimate_during_takeoff_or_while_the_count_is_rising() -> None:
    eta = MappingEta(timeout_s=300.0)
    _feed(eta, 0.0, 3.0, 0, Phase.TAKEOFF)
    assert eta.remaining_s is None

    _feed(eta, 3.0, 20.0, lambda t: 4.0 * t, Phase.EXPLORE)
    assert eta.remaining_s is None
    assert eta.done_at_s is None


def test_a_steady_drain_projects_to_when_the_count_reaches_zero() -> None:
    eta = MappingEta(timeout_s=300.0)
    _feed(eta, 0.0, 10.0, 60, Phase.EXPLORE)
    # 60 frontiers draining at 1 per second from t = 10 s: zero at t = 70 s.
    t = _feed(eta, 10.0, 40.0, lambda t: 60.0 - (t - 10.0), Phase.EXPLORE)
    remaining = eta.remaining_s
    assert remaining is not None
    # Smoothing lags the count by about its time constant, so allow a few seconds.
    assert remaining == pytest.approx(70.0 - t, abs=4.0)


def test_the_estimate_never_runs_past_the_mission_timeout() -> None:
    eta = MappingEta(timeout_s=45.0)
    _feed(eta, 0.0, 10.0, 200, Phase.EXPLORE)
    # A slow drain that would take minutes; the controller gives up at 45 s.
    t = _feed(eta, 10.0, 30.0, lambda t: 200.0 - 0.5 * (t - 10.0), Phase.EXPLORE)
    assert eta.remaining_s == pytest.approx(45.0 - t)


def test_leaving_exploration_marks_mapping_done_and_stays_done() -> None:
    eta = MappingEta(timeout_s=300.0)
    _feed(eta, 0.0, 10.0, 30, Phase.EXPLORE)
    eta.update(10.0, 0, Phase.RETURN)
    assert eta.done_at_s == 10.0
    assert eta.remaining_s == 0.0

    eta.update(25.0, 0, Phase.DONE)
    assert eta.done_at_s == 10.0
    assert eta.remaining_s == 0.0
