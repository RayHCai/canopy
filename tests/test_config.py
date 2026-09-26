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
    assert cfg.sim.v_max == pytest.approx(3.0)
    assert cfg.sensor.n_rays == 180 * 60 == 10_800
    assert cfg.map.voxel_m == pytest.approx(0.25)
    assert cfg.planner.orbit_altitudes == (2.0, 4.5, 8.0)
    assert cfg.safety.min_separation_m == pytest.approx(1.5)


def test_derived_values(cfg: Config) -> None:
    assert cfg.sim.dt == pytest.approx(0.05)
    # 0.25 m drone radius + 0.35 m margin
    assert cfg.safety.inflation_m == pytest.approx(0.60)


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


def test_sensor_rate_must_divide_control_rate(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sim"]["sensor_hz"] = 7
    with pytest.raises(ConfigError, match="integer multiple"):
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
    assert rules["battery"]["width_m"] == pytest.approx(0.93)
    assert isinstance(rules["rules"], list)
    assert rules["rules"][0]["rule"] == "meter_distance"


def test_planning_grid_must_nest_in_the_map_grid(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["map"]["plan_voxel_m"] = 0.6
    with pytest.raises(ConfigError, match=re.escape("integer multiple of map.voxel_m")):
        load_config(_write(tmp_path, raw))


def test_viewer_swarm_must_fit_the_launch_area(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["worldgen"]["launch_area_pads"] = raw["viewer"]["max_drones"] - 1
    with pytest.raises(ConfigError, match=re.escape("at least viewer.max_drones")):
        load_config(_write(tmp_path, raw))


def test_plan_factor(cfg: Config) -> None:
    assert cfg.map.plan_factor == 2


def test_perception_class_entry_may_omit_optional_rules(tmp_path: Path) -> None:
    """A class entry states only the cls it needs; every other rule takes its default."""
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"].append({"cls": "GAS_METER"})

    cfg = load_config(_write(tmp_path, raw))

    entry = next(c for c in cfg.perception.classes if c.cls == "GAS_METER")
    assert entry.report is True
    assert entry.hue_deg == (0.0, 360.0)
    assert entry.sat == (0.0, 1.0)
    assert entry.min_hits == 10
    assert entry.near_cls is None
    assert entry.near_m is None


def test_perception_class_with_unknown_name_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"].append({"cls": "NOT_A_CLASS"})
    with pytest.raises(ConfigError, match="is not a class"):
        load_config(_write(tmp_path, raw))


def test_duplicate_perception_class_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"].append(dict(raw["perception"]["classes"][1]))  # a second METER
    with pytest.raises(ConfigError, match=re.escape("lists ['METER'] more than once")):
        load_config(_write(tmp_path, raw))


def test_perception_near_cls_must_name_a_listed_class(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"].append({"cls": "GAS_METER", "near_cls": "PANEL", "near_m": 0.5})
    with pytest.raises(ConfigError, match="is not one of the listed classes"):
        load_config(_write(tmp_path, raw))


def test_perception_near_cls_requires_near_m(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"].append({"cls": "GAS_METER", "near_cls": "WALL"})
    with pytest.raises(ConfigError, match="near_cls and near_m must be given together"):
        load_config(_write(tmp_path, raw))


def test_perception_class_sat_outside_unit_interval_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"]["classes"][1]["sat"] = [0.5, 1.5]  # METER
    with pytest.raises(ConfigError, match=re.escape("sat must satisfy 0 <= lo <= hi <= 1")):
        load_config(_write(tmp_path, raw))


def test_viewer_detection_color_with_unknown_class_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["viewer"]["detection_colors"].append({"cls": "NOT_A_CLASS", "rgb": [0, 0, 0]})
    with pytest.raises(ConfigError, match="is not a class"):
        load_config(_write(tmp_path, raw))


def test_viewer_detection_color_channel_above_255_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["viewer"]["detection_colors"][0]["rgb"] = [256, 0, 0]  # METER
    with pytest.raises(ConfigError, match=re.escape("channels must be in [0, 255]")):
        load_config(_write(tmp_path, raw))


def test_sensor_ambient_outside_unit_interval_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sensor"]["ambient"] = 1.5
    with pytest.raises(ConfigError, match=re.escape("sensor.ambient must be in [0, 1]")):
        load_config(_write(tmp_path, raw))


def test_sensor_zero_sun_dir_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sensor"]["sun_dir"] = [0.0, 0.0, 0.0]
    with pytest.raises(ConfigError, match="sun_dir must not be the zero vector"):
        load_config(_write(tmp_path, raw))


def test_default_perception_classes_and_viewer_detection_colors(cfg: Config) -> None:
    """The shipped config's WALL entry is context-only; METER, CONDUIT and BUSH are reported.

    Each reported class also has a viewer outline colour.
    """
    by_name = {c.cls: c for c in cfg.perception.classes}
    assert set(by_name) == {"WALL", "METER", "CONDUIT", "BUSH"}
    assert by_name["WALL"].report is False
    for name in ("METER", "CONDUIT", "BUSH"):
        assert by_name[name].report is True

    color_classes = {c.cls for c in cfg.viewer.detection_colors}
    assert color_classes == {"METER", "CONDUIT", "BUSH"}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("track_match_m", 0.0), ("box_margin_m", -0.01), ("ground_snap_m", -0.01)],
)
def test_perception_track_and_box_fields_reject_bad_values(
    tmp_path: Path, field: str, bad_value: float
) -> None:
    """A non-positive track_match_m, or a negative box_margin_m/ground_snap_m, is rejected."""
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["perception"][field] = bad_value
    expected = re.escape(
        "perception.track_match_m must be positive, and box_margin_m and ground_snap_m non-negative"
    )
    with pytest.raises(ConfigError, match=expected):
        load_config(_write(tmp_path, raw))


def test_duplicate_viewer_detection_color_class_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["viewer"]["detection_colors"].append(
        dict(raw["viewer"]["detection_colors"][0])
    )  # 2nd METER
    with pytest.raises(ConfigError, match=re.escape("lists ['METER'] more than once")):
        load_config(_write(tmp_path, raw))


def test_sensor_negative_rgb_noise_is_rejected(tmp_path: Path) -> None:
    """Colour noise is a standard deviation, so a negative one is a typo, not a setting."""
    raw = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    raw["sensor"]["rgb_noise"] = -1.0
    with pytest.raises(ConfigError, match="rgb_noise must be non-negative"):
        load_config(_write(tmp_path, raw))
