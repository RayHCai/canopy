"""The `canopy-fly` entry point."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from canopy.cli.fly_single import build_parser, fly, main
from canopy.config import Config


def test_flies_the_whole_path(cfg: Config) -> None:
    summary = fly(cfg)
    assert summary.completed
    assert summary.waypoints_reached == summary.waypoints_total
    assert summary.waypoints_total == 1 + cfg.demo.laps * cfg.demo.orbit_waypoints
    assert summary.max_speed_ms <= cfg.sim.v_max + 1e-9
    # A polyline flown with rounded corners is a little longer than the polyline.
    assert summary.distance_flown_m == pytest.approx(summary.path_length_m, rel=0.15)
    assert summary.final_pos[2] == pytest.approx(cfg.demo.takeoff_altitude_m, abs=0.5)


def test_exit_code_is_zero_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet"]) == 0
    assert "flight summary" in capsys.readouterr().out


def test_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet", "--json", "--laps", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["dynamics"] == "kinematic"


def test_cli_overrides_reach_the_flight(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet", "--json", "--laps", "1", "--radius", "2", "--altitude", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # 1 lap of 24 waypoints plus the climb.
    assert payload["waypoints_total"] == 25
    assert payload["final_pos"][2] == pytest.approx(3.0, abs=0.5)


def test_bad_config_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("sim: {}\n", encoding="utf-8")
    assert main(["--quiet", "--config", str(bad)]) == 2


def test_gui_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--gui", "--no-gui"])


def test_gui_defaults_to_none_so_config_wins() -> None:
    assert build_parser().parse_args([]).gui is None


def test_summary_renders_every_field(cfg: Config) -> None:
    rendered = fly(cfg).render()
    for label in ("dynamics", "waypoints", "completed", "realtime factor", "battery left"):
        assert label in rendered


def test_realtime_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--realtime", "--no-realtime"])


def test_realtime_defaults_to_none_so_gui_decides() -> None:
    assert build_parser().parse_args([]).realtime is None


def test_realtime_throttles_to_wall_clock(cfg: Config) -> None:
    """A watchable flight must take roughly as long as it claims to."""
    short = dataclasses.replace(
        cfg, demo=dataclasses.replace(cfg.demo, laps=1, orbit_waypoints=4, orbit_radius_m=0.5)
    )
    summary = fly(short, realtime=True)
    assert summary.wall_time_s >= summary.sim_time_s * 0.5
