"""The `canopy-site` entry point.

``suggest_addresses`` and ``fetch_site`` are monkeypatched everywhere: this
module is about the CLI's own exit codes and output, not about geodata, so no
test here ever touches the network. ``monkeypatch.chdir`` puts each fetch's
``out/sites`` under ``tmp_path``, exactly as a real run would relative to
wherever it was invoked.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import canopy.cli.site as site_cli
from canopy.cli.site import main
from canopy.config import Config
from canopy.contracts import ResidentialDecision, ResolvedAddress, SiteSnapshot
from canopy.errors import GeodataError, SiteRejectedError
from canopy.worldgen import load_snapshot, save_snapshot

_ADDRESS = ResolvedAddress(
    label="12 Oak Street, Springfield",
    provider="fixture",
    ref="fixture:way/1",
    lat_deg=40.1,
    lon_deg=-75.2,
)


def test_suggest_lists_matches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [_ADDRESS])  # noqa: ARG005

    assert main(["suggest", "12 Oak"]) == 0

    out = capsys.readouterr().out
    assert "12 Oak Street, Springfield" in out
    assert "fixture" in out


def test_suggest_with_no_matches_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [])  # noqa: ARG005

    assert main(["suggest", "nowhere"]) == 0
    assert "no address suggestions" in capsys.readouterr().out


def test_suggest_geodata_error_exits_five(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(text: str, cfg: Config) -> list[ResolvedAddress]:
        msg = f"geocoder unreachable for {text!r}"
        raise GeodataError(msg)

    monkeypatch.setattr(site_cli, "suggest_addresses", boom)

    assert main(["suggest", "12 Oak"]) == 5
    assert "geocoder unreachable" in capsys.readouterr().err


def test_fetch_success_prints_site_id_and_writes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    site_snapshot: SiteSnapshot,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [site_snapshot.address])  # noqa: ARG005
    monkeypatch.setattr(
        site_cli,
        "fetch_site",
        lambda address, cfg, *, query="": site_snapshot,  # noqa: ARG005
    )

    assert main(["fetch", "12 test street"]) == 0

    out = capsys.readouterr().out
    assert site_snapshot.site_id in out
    assert site_snapshot.address.label in out
    saved = load_snapshot(site_snapshot.site_id)
    assert saved.site_id == site_snapshot.site_id


def test_fetch_pick_out_of_range_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [_ADDRESS])  # noqa: ARG005

    assert main(["fetch", "12 Oak", "--pick", "2"]) == 2
    assert "--pick" in capsys.readouterr().err


def test_fetch_ask_without_yes_exits_three_and_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_site: Callable[..., SiteSnapshot],
) -> None:
    ask_site = make_site(decision=ResidentialDecision.ASK)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [ask_site.address])  # noqa: ARG005
    monkeypatch.setattr(site_cli, "fetch_site", lambda address, cfg, *, query="": ask_site)  # noqa: ARG005

    assert main(["fetch", "test"]) == 3

    result = capsys.readouterr()
    assert "confirm" in result.err.lower()
    assert "p=" in result.out
    assert not (tmp_path / "out" / "sites" / ask_site.site_id).exists()


def test_fetch_ask_with_yes_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_site: Callable[..., SiteSnapshot],
) -> None:
    ask_site = make_site(decision=ResidentialDecision.ASK)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [ask_site.address])  # noqa: ARG005
    monkeypatch.setattr(site_cli, "fetch_site", lambda address, cfg, *, query="": ask_site)  # noqa: ARG005

    assert main(["fetch", "test", "--yes"]) == 0

    assert capsys.readouterr().out
    assert (tmp_path / "out" / "sites" / ask_site.site_id / "snapshot.json").exists()


def test_fetch_rejected_exits_four(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [_ADDRESS])  # noqa: ARG005

    def boom(address: ResolvedAddress, cfg: Config, *, query: str = "") -> SiteSnapshot:
        msg = f"{address.label} is zoned commercial"
        raise SiteRejectedError(msg)

    monkeypatch.setattr(site_cli, "fetch_site", boom)

    assert main(["fetch", "test"]) == 4
    assert "zoned commercial" in capsys.readouterr().err


def test_fetch_geodata_error_exits_five(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site_cli, "suggest_addresses", lambda text, cfg: [_ADDRESS])  # noqa: ARG005

    def boom(address: ResolvedAddress, cfg: Config, *, query: str = "") -> SiteSnapshot:
        msg = "overpass timed out"
        raise GeodataError(msg)

    monkeypatch.setattr(site_cli, "fetch_site", boom)

    assert main(["fetch", "test"]) == 5
    assert "overpass timed out" in capsys.readouterr().err


def test_show_prints_a_saved_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    site_snapshot: SiteSnapshot,
) -> None:
    monkeypatch.chdir(tmp_path)
    save_snapshot(site_snapshot)

    assert main(["show", site_snapshot.site_id]) == 0

    out = capsys.readouterr().out
    assert site_snapshot.site_id in out
    assert site_snapshot.address.label in out


def test_show_missing_snapshot_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["show", "no-such-site"]) == 2
    assert capsys.readouterr().err
