"""``canopy-view``: watch the swarm in the Canopy desktop window.

Opens a native window with the swarm orbiting the launch area. Swarm size and
scene live in the window's settings panel (the gear, top right); the flags here
only choose where it starts.

Examples
--------
Default swarm (``viewer.drones`` in the config)::

    uv run canopy-view

Five drones, reproducible layout, web inspector enabled::

    canopy-view --drones 5 --seed 42 --debug
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from canopy import log
from canopy.config import load_config
from canopy.errors import CanopyError
from canopy.viz import ViewerSession, launch

__all__ = ["build_parser", "main"]

_log = log.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command line for ``canopy-view``."""
    parser = argparse.ArgumentParser(
        prog="canopy-view",
        description="Open the Canopy desktop viewer.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="config YAML (default: the shipped config/default.yaml)",
    )
    parser.add_argument(
        "--drones",
        type=int,
        default=None,
        metavar="N",
        help="swarm size at launch (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=None, help="scene seed (default: random)")
    parser.add_argument("--debug", action="store_true", help="enable the web inspector")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="repeat for DEBUG")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    log.configure(-1 if args.quiet else args.verbose)

    try:
        session = ViewerSession(load_config(args.config), drones=args.drones, seed=args.seed)
        _log.info("opening viewer: %d drone(s), seed=%d", session.n_drones, session.seed)
        launch(session, debug=args.debug)
    except CanopyError as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":  # Windows spawns rather than forks; always guard.
    sys.exit(main())
