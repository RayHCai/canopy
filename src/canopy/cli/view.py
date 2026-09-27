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

A property rebuilt from a saved address, fetched earlier with ``canopy-site``::

    canopy-view --site out/sites/3f9a2c1e0b7d
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from canopy import log
from canopy.cli._common import add_verbosity_args, configure_logging
from canopy.config import load_config
from canopy.errors import CanopyError
from canopy.viz import ViewerSession, launch
from canopy.worldgen import load_snapshot

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
    parser.add_argument(
        "--site",
        default=None,
        metavar="REF",
        help=(
            "load a saved site snapshot (a snapshot.json path, its directory, or a bare "
            "site id under out/sites) and start in address mode, built from it"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="enable the web inspector")
    add_verbosity_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    configure_logging(args)

    try:
        site = load_snapshot(args.site) if args.site is not None else None
        session = ViewerSession(
            load_config(args.config), drones=args.drones, seed=args.seed, site=site
        )
        _log.info(
            "opening viewer: %d drone(s), seed=%d%s",
            session.n_drones,
            session.seed,
            "" if site is None else f", site={site.site_id}",
        )
        launch(session, debug=args.debug)
    except CanopyError as exc:
        _log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":  # Windows spawns rather than forks; always guard.
    sys.exit(main())
