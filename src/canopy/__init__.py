"""Canopy: an autonomous drone swarm that surveys a house for battery placement.

The package is one monolith with a subpackage per module boundary. Data crosses
those boundaries only as the dataclasses in :mod:`canopy.contracts`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
