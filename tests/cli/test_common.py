"""``canopy.cli._common``: the shared ``-v/-q`` flags and their wiring to ``log.configure``."""

from __future__ import annotations

import argparse
import logging

import pytest

from canopy import log
from canopy.cli._common import add_verbosity_args, configure_logging


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_verbosity_args(parser)
    return parser.parse_args(argv)


def test_defaults_are_not_verbose_or_quiet() -> None:
    args = _parse([])
    assert args.verbose == 0
    assert args.quiet is False


def test_verbose_counts_repeats() -> None:
    assert _parse(["-v"]).verbose == 1
    assert _parse(["-v", "-v"]).verbose == 2
    assert _parse(["--verbose", "--verbose", "--verbose"]).verbose == 3


def test_quiet_is_a_flag() -> None:
    assert _parse(["-q"]).quiet is True
    assert _parse(["--quiet"]).quiet is True


@pytest.mark.parametrize(
    ("argv", "expected_level"),
    [
        ([], logging.INFO),
        (["-v"], logging.DEBUG),
        (["-v", "-v"], logging.DEBUG),
        (["-q"], logging.WARNING),
    ],
)
def test_configure_logging_maps_flags_to_level(argv: list[str], expected_level: int) -> None:
    configure_logging(_parse(argv))
    assert logging.getLogger("canopy").level == expected_level


def test_quiet_wins_over_verbose() -> None:
    """``-q`` and ``-v`` together is a nonsensical combination, but ``-q`` wins."""
    configure_logging(_parse(["-v", "-q"]))
    assert logging.getLogger("canopy").level == logging.WARNING


def test_configure_logging_matches_log_configure_directly() -> None:
    """The helper is exactly ``log.configure(-1 if quiet else verbose)``, not a new policy."""
    configure_logging(_parse(["-v", "-v"]))
    by_helper = logging.getLogger("canopy").level
    log.configure(2)
    by_hand = logging.getLogger("canopy").level
    assert by_helper == by_hand
