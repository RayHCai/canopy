"""Exception hierarchy.

Every error Canopy raises on purpose derives from :class:`CanopyError`, so a
caller can distinguish "the simulation disagreed with us" from "NumPy threw".
"""

from __future__ import annotations

__all__ = [
    "AssetError",
    "CanopyError",
    "ConfigError",
    "DependencyMissingError",
    "GeodataError",
    "PlanningError",
    "SimulationError",
    "SiteError",
    "SiteRejectedError",
    "WorldgenError",
]


class CanopyError(Exception):
    """Base class for all deliberate Canopy failures."""


class ConfigError(CanopyError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DependencyMissingError(CanopyError):
    """An optional extra is required for the requested mode but not installed."""


class SimulationError(CanopyError):
    """The simulator was driven into an invalid state."""


class PlanningError(CanopyError):
    """No safe path exists, or a goal lies in space the drone may not enter.

    Expected in normal operation -- a frontier behind unmapped space is
    unreachable until it is mapped -- so the mission controller catches it and
    blacklists the goal rather than aborting.
    """


class SiteError(CanopyError):
    """The finished map holds nothing to site a battery against.

    Raised when no meter was discovered, or when no house wall could be traced
    near the one that was. Expected in normal operation -- a mission that ran
    out of time may never have seen the meter -- so a caller reports it rather
    than aborting.
    """


class WorldgenError(CanopyError):
    """Procedural generation could not produce a valid property."""


class AssetError(WorldgenError):
    """The model library is missing, malformed, or asked for something it lacks.

    A subclass of :class:`WorldgenError` because every asset lookup happens
    during generation: a caller that only wants "the field could not be built"
    need not know the model database exists.
    """


class GeodataError(WorldgenError):
    """Real-world data for an address could not be fetched or made sense of.

    The network failed, a provider refused or returned something unreadable, or
    no building was mapped at the address. Expected in normal use -- public
    geodata services are best-effort -- so a caller reports it and lets the
    user try again rather than aborting.
    """


class SiteRejectedError(WorldgenError):
    """An address resolved, but it is not a property the generator should or can build.

    Raised for an address the residential check rejects, a house outside what
    the generator can represent (too big, too many storeys, too irregular a
    footprint), or a house with no wall that will take a meter. The message
    names the value that failed and the setting it failed against.
    """
