"""Shared bits for the CLI entry points.

Not a stage facade -- ``cli`` holds entry points and nothing imports it back
(see ``tests/test_architecture.py``), so this module is a plain internal
helper rather than a re-export surface with a public ``__all__`` contract.
Every ``canopy-*`` script wires up the same ``-v/-q`` verbosity pair and feeds
it to :func:`canopy.log.configure`; this is that wiring, written once.
"""

from __future__ import annotations

import argparse

from canopy import log

__all__ = ["add_verbosity_args", "configure_logging"]


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``-v/--verbose`` and ``-q/--quiet`` flags to ``parser``.

    Parameters
    ----------
    parser
        The parser (or subparser) to add the flags to.
    """
    parser.add_argument("-v", "--verbose", action="count", default=0, help="repeat for DEBUG")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")


def configure_logging(args: argparse.Namespace) -> None:
    """Configure logging from parsed ``-v``/``-q`` flags.

    Parameters
    ----------
    args
        Namespace returned by a parser built with :func:`add_verbosity_args`.
        ``--quiet`` wins over any number of ``--verbose`` repeats.
    """
    log.configure(-1 if args.quiet else args.verbose)
