"""How much simulated time a mapping mission has left before exploration ends.

Exploration ends when no reachable frontier is left (or at ``sim.timeout_s``;
see :meth:`~canopy.planning.mission.MissionController._termination`), so the
estimate follows the frontier count rather than coverage. Traced over seeded
3-drone missions, coverage is a poor clock: ground-band coverage typically
saturates 20-50 s before the end and total coverage plateaus somewhere in
0.9-0.99, so "(1 - coverage) / rate" never converges. The frontier count, by
contrast, rises while the swarm opens the property up, peaks, then drains
roughly linearly to zero.

:class:`MappingEta` therefore smooths the count (the planner only refreshes it
at ``sim.replan_hz`` and it jitters by a few targets between replans), tracks
the smoothed peak, and projects the average drain rate since the count left
that peak forward to zero. On those traces this is within about 6 s over the last 30% of a
mission and about 2 s over the last 15%. Early on it is rough -- a count can
rise again after a local peak -- so no estimate is offered until the count has
fallen a meaningful amount for a meaningful time.
"""

from __future__ import annotations

import math

from canopy.planning.mission import Phase

__all__ = ["MappingEta"]

#: Time constant (s) of the exponential smoothing applied to the frontier count.
_SMOOTHING_S = 3.0

#: The drain is timed from the last moment the smoothed count was within this
#: many frontiers of its peak, so a plateau at the top -- common, while the
#: swarm works through the targets it has just opened up -- does not dilute
#: the rate as though it had been draining all along.
_PEAK_BAND = 1.0

#: The smoothed count must have fallen this many frontiers below its peak ...
_MIN_DRAIN = 5.0

#: ... over at least this many simulated seconds before a rate is trusted.
_MIN_SINCE_PEAK_S = 5.0


class MappingEta:
    """Online estimate of the simulated time until exploration ends.

    Feed it every control tick with :meth:`update`; read :attr:`remaining_s`.

    Parameters
    ----------
    timeout_s
        The mission's hard exploration timeout (``sim.timeout_s``). No estimate
        runs past it, because the controller will not.
    """

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s
        self._t = 0.0
        self._smoothed: float | None = None
        self._peak = -math.inf
        self._t_peak = 0.0
        self._remaining_s: float | None = None
        self._done_at_s: float | None = None

    @property
    def remaining_s(self) -> float | None:
        """Estimated simulated seconds left to explore.

        ``None`` while there is no trustworthy trend yet, and ``0.0`` once
        exploration has ended.
        """
        if self._done_at_s is not None:
            return 0.0
        return self._remaining_s

    @property
    def done_at_s(self) -> float | None:
        """Simulated time exploration ended, or ``None`` while it has not."""
        return self._done_at_s

    def update(self, t: float, n_frontiers: int, phase: Phase) -> None:
        """Fold in the mission's state after the tick ending at ``t``.

        Parameters
        ----------
        t
            Simulated seconds since launch.
        n_frontiers
            Reachable targets the controller currently holds.
        phase
            The controller's phase. Only ``EXPLORE`` informs the estimate;
            anything after it marks exploration as finished.
        """
        if phase is Phase.TAKEOFF or self._done_at_s is not None:
            self._t = t
            return
        if phase is not Phase.EXPLORE:
            self._done_at_s = t
            self._remaining_s = 0.0
            return
        dt = t - self._t
        self._t = t
        if self._smoothed is None:
            self._smoothed = float(n_frontiers)
        else:
            self._smoothed += -math.expm1(-dt / _SMOOTHING_S) * (n_frontiers - self._smoothed)
        self._peak = max(self._peak, self._smoothed)
        if self._smoothed >= self._peak - _PEAK_BAND:
            self._t_peak = t
        drained = self._peak - self._smoothed
        since = t - self._t_peak
        if drained < _MIN_DRAIN or since < _MIN_SINCE_PEAK_S:
            self._remaining_s = None
            return
        estimate = self._smoothed * since / drained
        self._remaining_s = min(estimate, max(self._timeout_s - t, 0.0))
