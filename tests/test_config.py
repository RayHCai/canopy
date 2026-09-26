"""The config loader must fail loudly, not silently."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from canopy.config import Config, load_config, load_rules
from canopy.errors import ConfigError


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_load_and_match_the_spec(cfg: Config) -> None:
    assert cfg.sim.control_hz == 20
    assert cfg.sim.sensor_hz == 5
    assert cfg.sim.dynamics == "kinematic"
    assert cfg.sim.v_max == pytest.approx(3.0)
    assert cfg.sensor.n_rays == 180 * 60 == 10_800
    assert cfg.map.voxel_m == pytest.approx(0.25)
    assert cfg.planner.orbit_altitudes == (2.0, 4.5, 8.0)
    assert cfg.safety.min_separation_m == pytest.approx(1.5)


def test_derived_values(cfg: Config) -> None:
    assert cfg.sim.dt == pytest.approx(0.05)
    # 0.25 m drone radius + 0.35 m margin
    assert cfg.safety.inflation_m == pytest.approx(0.60)
    # 60 Hz inner PID / 20 Hz control tick
    assert cfg.physics_steps_per_tick == 3


def test_config_is_frozen(cfg: Config) -> None:
    with pytest.raises(AttributeError):
        cfg.sim.v_max = 99.0  # type: ignore[misc]


def test_unknown_key_is_rejected(tmp_path: Path, cfg: Config) -> None:
    """A typo in the YAML must not be silently ignored."""
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["v_maxx"] = 4.0
    with pytest.raises(ConfigError, match=re.escape("unknown key(s) ['v_maxx']")):
        load_config(_write(tmp_path, raw))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    del raw["sim"]["v_max"]
    with pytest.raises(ConfigError, match=re.escape("missing key(s) ['v_max']")):
        load_config(_write(tmp_path, raw))


def test_wrong_scalar_type_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["control_hz"] = "twenty"
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(_write(tmp_path, raw))


def test_bool_is_not_accepted_as_a_number(tmp_path: Path) -> None:
    """Bool subclasses int; the loader must not let that through."""
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["v_max"] = True
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(_write(tmp_path, raw))


def test_unknown_dynamics_backend_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["dynamics"] = "warp-drive"
    with pytest.raises(ConfigError, match=re.escape("sim.dynamics must be one of")):
        load_config(_write(tmp_path, raw))


def test_sensor_rate_must_divide_control_rate(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["sensor_hz"] = 7
    with pytest.raises(ConfigError, match="integer multiple"):
        load_config(_write(tmp_path, raw))


def test_physics_rates_must_nest_exactly(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["physics"]["ctrl_freq"] = 50  # 240 % 50 != 0
    with pytest.raises(ConfigError, match=re.escape("physics.pyb_freq")):
        load_config(_write(tmp_path, raw))


def test_wrong_tuple_arity_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["demo"]["home"] = [0.0, 0.0]
    with pytest.raises(ConfigError, match=re.escape("expected 3 item(s)")):
        load_config(_write(tmp_path, raw))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_file_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path)


def test_rules_load(tmp_path: Path) -> None:
    rules = load_rules()
    assert rules["battery_w_m"] == pytest.approx(1.1)
    assert rules["weights"]["conduit_per_m"] == pytest.approx(1.0)
