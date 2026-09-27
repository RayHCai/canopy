"""``canopy-site``: resolve a street address into a saved site snapshot.

Three subcommands, each a thin wrapper over :mod:`canopy.worldgen`'s
address-seeded API: ``suggest`` lists matches for partly typed text,
``fetch`` resolves one, fetches and freezes the real-world data around it and
saves the result, and ``show`` prints a snapshot already on disk. Building the
property itself is ``canopy-view --site`` or ``canopy.worldgen.generate_field``
(``site=``) -- this tool only gets the data onto disk.

Examples
--------
::

    canopy-site suggest "12 oak street"
    canopy-site fetch "12 oak street, springfield" --pick 1
    canopy-site show out/sites/3f9a2c1e0b7d
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from canopy import log
from canopy.cli._common import add_verbosity_args, configure_logging
from canopy.config import Config, load_config
from canopy.contracts import ResidentialDecision, ResolvedAddress, SiteSnapshot
from canopy.errors import CanopyError, GeodataError, SiteRejectedError, WorldgenError
from canopy.worldgen import fetch_site, load_snapshot, save_snapshot, suggest_addresses

__all__ = ["build_parser", "main"]

_log = log.get_logger(__name__)

#: Exit codes, named for the messages below rather than left as bare literals.
_OK = 0
_USAGE = 2
_NEEDS_CONFIRMATION = 3
_REJECTED = 4
_GEODATA_FAILED = 5


def build_parser() -> argparse.ArgumentParser:
    """Command line for ``canopy-site``."""
    parser = argparse.ArgumentParser(
        prog="canopy-site",
        description="Resolve a street address into a saved site snapshot.",
    )
    add_verbosity_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    suggest = sub.add_parser("suggest", help="list address matches for partly typed text")
    suggest.add_argument("text", metavar="TEXT", help="partial address to search for")
    suggest.add_argument("--config", type=Path, default=None, metavar="PATH", help="config YAML")

    fetch = sub.add_parser("fetch", help="fetch, check and save the site at an address")
    fetch.add_argument("text", metavar="TEXT", help="address to search for and fetch")
    fetch.add_argument(
        "--pick",
        type=int,
        default=1,
        metavar="N",
        help="1-based rank of the suggestion to fetch (default: 1, the best match)",
    )
    fetch.add_argument(
        "--yes",
        action="store_true",
        help="build the site even if the residential check could not confirm it is a home",
    )
    fetch.add_argument("--config", type=Path, default=None, metavar="PATH", help="config YAML")

    show = sub.add_parser("show", help="print a previously saved site snapshot")
    show.add_argument("ref", metavar="REF", help="snapshot.json path, its directory, or a site id")

    return parser


def _print_matches(matches: Sequence[ResolvedAddress]) -> None:
    """List suggestions, 1-based, in the order ``--pick`` indexes them."""
    for i, addr in enumerate(matches, start=1):
        print(f"{i}. {addr.label}  [{addr.provider}]  ({addr.lat_deg:.6f}, {addr.lon_deg:.6f})")


def _print_verdict(snapshot: SiteSnapshot) -> None:
    verdict = snapshot.verdict
    print(f"address:     {snapshot.address.label}")
    print(f"residential: {verdict.decision.value} (p={verdict.p_residential:.2f})")
    for reason in verdict.reasons:
        print(f"  - {reason}")


def _print_observed(snapshot: SiteSnapshot) -> None:
    """One line on what the fetch actually saw, the same facts the viewer's card shows."""
    house = snapshot.house
    bits = ["footprint"]
    if house.levels is not None:
        bits.append(f"{house.levels} storey" + ("" if house.levels == 1 else "s"))
    if house.roof_shape is not None:
        bits.append(f"{house.roof_shape} roof")
    n_trees = int(snapshot.trees.reshape(-1, 3).shape[0])
    if n_trees:
        bits.append(f"{n_trees} tree" + ("" if n_trees == 1 else "s"))
    if snapshot.neighbours:
        n = len(snapshot.neighbours)
        bits.append(f"{n} neighbour" + ("" if n == 1 else "s"))
    if snapshot.street is not None:
        bits.append(snapshot.street_name or "street")
    print(f"observed:    {', '.join(bits)}")


def _print_notes(snapshot: SiteSnapshot) -> None:
    if not snapshot.notes:
        return
    print("notes:")
    for note in snapshot.notes:
        print(f"  - {note}")


def _suggest_cmd(args: argparse.Namespace, cfg: Config) -> int:
    """``canopy-site suggest``: list matches, best first."""
    try:
        matches = suggest_addresses(args.text, cfg)
    except GeodataError as exc:
        _log.error("%s", exc)
        return _GEODATA_FAILED
    if not matches:
        print(f"no address suggestions matched {args.text!r}")
        return _OK
    _print_matches(matches)
    return _OK


def _resolve_pick(args: argparse.Namespace, cfg: Config) -> ResolvedAddress | int:
    """Suggest matches for ``args.text`` and return the ``--pick``-th, or an exit code."""
    try:
        matches = suggest_addresses(args.text, cfg)
    except GeodataError as exc:
        _log.error("%s", exc)
        return _GEODATA_FAILED
    if not matches:
        _log.error("no address suggestions matched %r", args.text)
        return _GEODATA_FAILED
    pick = int(args.pick)
    if not 1 <= pick <= len(matches):
        _log.error("--pick must be in [1, %d], got %d", len(matches), pick)
        return _USAGE
    return matches[pick - 1]


def _fetch_cmd(args: argparse.Namespace, cfg: Config) -> int:
    """``canopy-site fetch``: resolve ``--pick``, fetch it, and save the result."""
    picked = _resolve_pick(args, cfg)
    if isinstance(picked, int):
        return picked
    address = picked

    try:
        snapshot = fetch_site(address, cfg, query=args.text)
    except SiteRejectedError as exc:
        _log.error("%s", exc)
        return _REJECTED
    except GeodataError as exc:
        _log.error("%s", exc)
        return _GEODATA_FAILED

    if snapshot.verdict.decision is ResidentialDecision.ASK and not args.yes:
        _print_verdict(snapshot)
        _print_observed(snapshot)
        _log.error("could not confirm this is a home; re-run with --yes to build it anyway")
        return _NEEDS_CONFIRMATION

    path = save_snapshot(snapshot)
    print(f"site id:     {snapshot.site_id}")
    print(f"saved to:    {path}")
    _print_verdict(snapshot)
    _print_observed(snapshot)
    _print_notes(snapshot)
    return _OK


def _show_cmd(args: argparse.Namespace) -> int:
    """``canopy-site show``: print a snapshot already on disk."""
    try:
        snapshot = load_snapshot(args.ref)
    except WorldgenError as exc:
        _log.error("%s", exc)
        return _USAGE
    print(f"site id:     {snapshot.site_id}")
    _print_verdict(snapshot)
    _print_observed(snapshot)
    _print_notes(snapshot)
    return _OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    configure_logging(args)

    if args.command == "show":
        return _show_cmd(args)

    try:
        cfg = load_config(args.config)
    except CanopyError as exc:
        _log.error("%s", exc)
        return _USAGE

    if args.command == "suggest":
        return _suggest_cmd(args, cfg)
    return _fetch_cmd(args, cfg)


if __name__ == "__main__":  # Windows spawns rather than forks; always guard.
    sys.exit(main())
