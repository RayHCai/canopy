"""Logging setup.

Canopy logs; it does not print. Entry points call :func:`configure` exactly
once, and every module uses ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure", "get_logger"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(verbosity: int = 0, *, stream: object = None) -> None:
    """Install a single stderr handler on the ``canopy`` logger.

    Parameters
    ----------
    verbosity
        ``0`` for INFO, ``1`` or more for DEBUG. ``-1`` or less for WARNING.
    stream
        Destination stream; defaults to ``sys.stderr``.
    """
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    elif verbosity < 0:
        level = logging.WARNING

    # An explicit env override wins, so a batch run can be quieted without
    # touching call sites.
    if (env := os.environ.get("CANOPY_LOG_LEVEL")) is not None:
        level = logging.getLevelNamesMapping().get(env.upper(), level)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger("canopy")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``."""
    return logging.getLogger(name)
