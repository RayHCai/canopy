"""Typed configuration.

``config/default.yaml`` is the single source of truth for every tunable. This
module turns it into frozen dataclasses so the rest of the codebase never
indexes a dict by string, and so a typo in the YAML fails loudly at load time
instead of silently at hour 19 of the hackathon.

Canopy is an application rather than a library, so the config directory lives at
the repository root (where it is easy to edit mid-run) and is located by walking
up from this module. Set ``CANOPY_CONFIG_DIR`` to override.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np
import yaml

from canopy.contracts import Vec3
from canopy.errors import ConfigError

__all__ = [
    "Config",
    "DemoCfg",
    "MapCfg",
    "PhysicsCfg",
    "PlannerCfg",
    "SafetyCfg",
    "SensorCfg",
    "SimCfg",
    "WorldgenCfg",
    "default_config_path",
    "default_rules_path",
    "load_config",
    "load_rules",
]

_T = TypeVar("_T")

_VALID_DYNAMICS = ("kinematic", "pybullet")

#: ``tuple[X, ...]`` has exactly two type arguments, the second being ``Ellipsis``.
_VARIADIC_TUPLE_ARGS = 2


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimCfg:
    """Timing and motion limits shared by every dynamics backend."""

    control_hz: int
    sensor_hz: int
    replan_hz: int
    timeout_s: float
    dynamics: str
    v_max: float
    a_max: float
    battery_drain_per_10s: float
    rth_battery: float
    yaw_rate_deg_s: float

    @property
    def dt(self) -> float:
        """Seconds per control tick."""
        return 1.0 / self.control_hz

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the section is internally inconsistent."""
        if self.dynamics not in _VALID_DYNAMICS:
            msg = f"sim.dynamics must be one of {_VALID_DYNAMICS}, got {self.dynamics!r}"
            raise ConfigError(msg)
        for name in ("control_hz", "sensor_hz", "replan_hz"):
            if getattr(self, name) <= 0:
                msg = f"sim.{name} must be positive"
                raise ConfigError(msg)
        if self.control_hz % self.sensor_hz:
            msg = (
                f"sim.control_hz ({self.control_hz}) must be an integer multiple of "
                f"sim.sensor_hz ({self.sensor_hz}) so scans land on control ticks"
            )
            raise ConfigError(msg)
        if self.v_max <= 0 or self.a_max <= 0:
            msg = "sim.v_max and sim.a_max must be positive"
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class PhysicsCfg:
    """gym-pybullet-drones backend settings. Read only in ``pybullet`` mode."""

    drone_model: str
    pyb_freq: int
    ctrl_freq: int
    gui: bool
    setpoint_speed_ms: float

    def validate(self, control_hz: int) -> None:
        """Raise :class:`ConfigError` unless the three rates nest exactly."""
        if self.pyb_freq % self.ctrl_freq:
            msg = (
                f"physics.pyb_freq ({self.pyb_freq}) must be an integer multiple of "
                f"physics.ctrl_freq ({self.ctrl_freq})"
            )
            raise ConfigError(msg)
        if self.ctrl_freq % control_hz:
            msg = (
                f"physics.ctrl_freq ({self.ctrl_freq}) must be an integer multiple of "
                f"sim.control_hz ({control_hz}) so one control tick is a whole number "
                f"of inner steps"
            )
            raise ConfigError(msg)
        if self.setpoint_speed_ms <= 0:
            msg = "physics.setpoint_speed_ms must be positive"
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class SensorCfg:
    """360-degree ray sensor geometry and noise."""

    az_rays: int
    el_rays: int
    el_min_deg: float
    el_max_deg: float
    max_range_m: float
    range_noise_m: float

    @property
    def n_rays(self) -> int:
        """Rays per scan."""
        return self.az_rays * self.el_rays


@dataclass(frozen=True, slots=True)
class WorldgenCfg:
    """Procedural property generation."""

    lot_m: tuple[float, float]
    bush_density_per_10m: float
    p_meter_occluded: float
    trees: tuple[int, int]
    max_edge_m: float


@dataclass(frozen=True, slots=True)
class MapCfg:
    """Occupancy grid, coverage and semantics thresholds."""

    voxel_m: float
    height_m: float
    plan_voxel_m: float
    min_hits: int
    coverage_max_range_m: float
    coverage_max_incidence_deg: float


@dataclass(frozen=True, slots=True)
class PlannerCfg:
    """Orbit, frontier and assignment tunables."""

    orbit_altitudes: tuple[float, ...]
    orbit_margin_m: float
    orbit_waypoints_per_ring: int
    frontier_min_voxels: int
    gain_lambda: float
    goal_conflict_radius_m: float
    goal_conflict_penalty: float
    hysteresis: float
    blacklist_s: float
    done_ground_coverage: float


@dataclass(frozen=True, slots=True)
class SafetyCfg:
    """Shield margins and geofence."""

    drone_radius_m: float
    margin_m: float
    min_separation_m: float
    ceiling_m: float
    geofence_inset_m: float
    stuck_window_s: float
    stuck_min_progress_m: float

    @property
    def inflation_m(self) -> float:
        """Total radius by which occupied space is grown before planning."""
        return self.drone_radius_m + self.margin_m


@dataclass(frozen=True, slots=True)
class DemoCfg:
    """Obstacle-free single-drone flight check (``canopy-fly``)."""

    home: tuple[float, float, float]
    takeoff_altitude_m: float
    orbit_radius_m: float
    orbit_waypoints: int
    laps: int
    waypoint_tolerance_m: float

    @property
    def home_xyz(self) -> Vec3:
        """``home`` as a float64 array."""
        return np.asarray(self.home, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Config:
    """The whole of ``config/default.yaml``, validated."""

    sim: SimCfg
    physics: PhysicsCfg
    sensor: SensorCfg
    worldgen: WorldgenCfg
    map: MapCfg
    planner: PlannerCfg
    safety: SafetyCfg
    demo: DemoCfg

    def validate(self) -> None:
        """Run every section's cross-field checks."""
        self.sim.validate()
        self.physics.validate(self.sim.control_hz)

    @property
    def physics_steps_per_tick(self) -> int:
        """Inner PID steps per control tick in ``pybullet`` mode."""
        return self.physics.ctrl_freq // self.sim.control_hz


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _config_dir() -> Path:
    """Locate the directory holding ``default.yaml``."""
    if (env := os.environ.get("CANOPY_CONFIG_DIR")) is not None:
        path = Path(env).expanduser()
        if not path.is_dir():
            msg = f"CANOPY_CONFIG_DIR={env!r} is not a directory"
            raise ConfigError(msg)
        return path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "default.yaml"
        if candidate.is_file():
            return candidate.parent

    msg = (
        "could not locate config/default.yaml by walking up from "
        f"{Path(__file__).resolve()}; set CANOPY_CONFIG_DIR or pass an explicit path"
    )
    raise ConfigError(msg)


def default_config_path() -> Path:
    """Path to the shipped ``default.yaml``."""
    return _config_dir() / "default.yaml"


def default_rules_path() -> Path:
    """Path to the shipped ``rules.yaml``."""
    return _config_dir() / "rules.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file that must hold a top-level mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read config {path}: {exc}"
        raise ConfigError(msg) from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        msg = f"{path} must contain a YAML mapping at the top level"
        raise ConfigError(msg)
    return loaded


def _coerce_union(annotation: Any, value: Any, where: str) -> Any:
    """Coerce an ``X | None`` annotation."""
    if value is None:
        return None
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) != 1:
        msg = f"{where}: unsupported union annotation {annotation!r}"
        raise ConfigError(msg)
    return _coerce(args[0], value, where)


def _coerce_tuple(annotation: Any, value: Any, where: str) -> tuple[Any, ...]:
    """Coerce a ``tuple[...]`` annotation from a YAML sequence."""
    if not isinstance(value, (list, tuple)):
        msg = f"{where}: expected a sequence, got {type(value).__name__}"
        raise ConfigError(msg)
    args = list(get_args(annotation))
    if len(args) == _VARIADIC_TUPLE_ARGS and args[1] is Ellipsis:  # tuple[X, ...]
        return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if len(args) != len(value):
        msg = f"{where}: expected {len(args)} item(s), got {len(value)}"
        raise ConfigError(msg)
    return tuple(
        _coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
    )


def _coerce_scalar(annotation: Any, value: Any, where: str) -> Any:
    """Coerce a bool, int, float or str, rejecting silent conversions."""
    # bool must be checked before int: bool is a subclass of int.
    if annotation is bool:
        if not isinstance(value, bool):
            msg = f"{where}: expected a boolean, got {value!r}"
            raise ConfigError(msg)
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{where}: expected an integer, got {value!r}"
            raise ConfigError(msg)
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"{where}: expected a number, got {value!r}"
            raise ConfigError(msg)
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            msg = f"{where}: expected a string, got {value!r}"
            raise ConfigError(msg)
        return value
    msg = f"{where}: unsupported annotation {annotation!r}"
    raise ConfigError(msg)


def _coerce(annotation: Any, value: Any, where: str) -> Any:
    """Convert one YAML node to its annotated type."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return _coerce_union(annotation, value, where)
    if origin is tuple:
        return _coerce_tuple(annotation, value, where)
    # `isinstance(annotation, type)` narrows to a dataclass *class*, not an
    # instance of one, which is what _build needs.
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            msg = f"{where}: expected a mapping, got {type(value).__name__}"
            raise ConfigError(msg)
        return _build(annotation, value, where)
    return _coerce_scalar(annotation, value, where)


def _build(cls: type[_T], data: dict[str, Any], where: str) -> _T:
    """Construct a config dataclass, rejecting unknown and missing keys."""
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]

    if unknown := sorted(set(data) - set(names)):
        msg = f"{where or 'config'}: unknown key(s) {unknown}; known keys are {names}"
        raise ConfigError(msg)
    if missing := sorted(set(names) - set(data)):
        msg = f"{where or 'config'}: missing key(s) {missing}"
        raise ConfigError(msg)

    kwargs = {n: _coerce(hints[n], data[n], f"{where}.{n}" if where else n) for n in names}
    return cls(**kwargs)


def load_config(path: Path | str | None = None) -> Config:
    """Load, coerce and validate the configuration.

    Parameters
    ----------
    path
        YAML file to read. Defaults to the shipped ``config/default.yaml``.

    Returns
    -------
    Config
        A validated, frozen configuration tree.

    Raises
    ------
    ConfigError
        If the file is unreadable, has unknown or missing keys, holds a value of
        the wrong type, or fails a cross-field check.
    """
    resolved = Path(path) if path is not None else default_config_path()
    cfg = _build(Config, _read_yaml(resolved), "")
    cfg.validate()
    return cfg


def load_rules(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``rules.yaml``.

    Placement rules stay a plain mapping on purpose: they stand in for the real
    SSR checklist and their shape is expected to change wholesale.
    """
    resolved = Path(path) if path is not None else default_rules_path()
    return _read_yaml(resolved)
