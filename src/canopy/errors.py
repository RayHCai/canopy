"""Exception hierarchy.

Every error Canopy raises on purpose derives from :class:`CanopyError`, so a
caller can distinguish "the simulation disagreed with us" from "NumPy threw".
"""

from __future__ import annotations

__all__ = [
    "CanopyError",
    "ConfigError",
    "DependencyMissingError",
    "SimulationError",
]


class CanopyError(Exception):
    """Base class for all deliberate Canopy failures."""


class ConfigError(CanopyError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DependencyMissingError(CanopyError):
    """An optional extra is required for the requested mode but not installed."""


class SimulationError(CanopyError):
    """The simulator was driven into an invalid state."""
