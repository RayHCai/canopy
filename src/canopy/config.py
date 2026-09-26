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
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np
import yaml

from canopy.contracts import Cls, Vec3
from canopy.errors import ConfigError

__all__ = [
    "Config",
    "DemoCfg",
    "DetectionColorCfg",
    "MapCfg",
    "PerceptionCfg",
    "PerceptionClassCfg",
    "PlannerCfg",
    "SafetyCfg",
    "SensorCfg",
    "SimCfg",
    "ViewerCfg",
    "WorldgenCfg",
    "build_section",
    "default_config_path",
    "default_rules_path",
    "load_config",
    "load_rules",
]

_T = TypeVar("_T")

#: ``tuple[X, ...]`` has exactly two type arguments, the second being ``Ellipsis``.
_VARIADIC_TUPLE_ARGS = 2

#: Largest value of an 8-bit colour channel.
_RGB_MAX = 255

#: Degrees in a full turn of hue.
_FULL_TURN_DEG = 360.0


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimCfg:
    """Timing and kinematic motion limits."""

    control_hz: int
    sensor_hz: int
    replan_hz: int
    timeout_s: float
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
class SensorCfg:
    """360-degree ray sensor geometry and noise."""

    az_rays: int
    el_rays: int
    el_min_deg: float
    el_max_deg: float
    max_range_m: float
    range_noise_m: float
    sun_dir: tuple[float, float, float]
    """Direction toward the sun, world frame. Normalised on use."""
    ambient: float
    """Brightness of a surface facing away from the sun, as a fraction of its colour."""
    rgb_noise: float
    """Standard deviation of Gaussian noise on each colour channel, in 8-bit levels."""
    sky_rgb: tuple[int, int, int]
    """Colour reported for a ray that hits nothing."""

    @property
    def n_rays(self) -> int:
        """Rays per scan."""
        return self.az_rays * self.el_rays

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the colour model is unusable."""
        if not np.any(np.asarray(self.sun_dir)):
            msg = "sensor.sun_dir must not be the zero vector"
            raise ConfigError(msg)
        if not 0.0 <= self.ambient <= 1.0:
            msg = f"sensor.ambient must be in [0, 1], got {self.ambient}"
            raise ConfigError(msg)
        if self.rgb_noise < 0.0:
            msg = f"sensor.rgb_noise must be non-negative, got {self.rgb_noise}"
            raise ConfigError(msg)
        _check_rgb(self.sky_rgb, "sensor.sky_rgb")


@dataclass(frozen=True, slots=True)
class WorldgenCfg:
    """Procedural property generation."""

    lot_m: tuple[float, float]
    bush_density_per_10m: float
    p_meter_occluded: float
    trees: tuple[int, int]
    max_edge_m: float
    launch_area_pads: int
    """Swarm size the launch area is kept clear for: nothing is placed within
    ``planner.launch_clear_radius_m`` of any of this many pads. A larger swarm's
    outer pads fall outside that guarantee."""


@dataclass(frozen=True, slots=True)
class PerceptionClassCfg:
    """One kind of object the detector looks for, described as colour and shape rules.

    Teaching the detector a new class means adding one of these to
    ``perception.classes`` in ``config/default.yaml``; no code changes. Every
    rule except ``cls`` is optional, so an entry states only what tells its
    class apart. Colour rules use hue, saturation and value because sunlight
    and shadow scale a surface's brightness but leave its hue and saturation
    alone.
    """

    cls: str
    """Name of a :class:`~canopy.contracts.Cls` member, e.g. ``METER``."""
    report: bool = True
    """Whether matches become discovered objects. ``False`` makes the class
    context only: evidence that other classes' ``near_cls`` rules measure to."""
    hue_deg: tuple[float, float] = (0.0, 360.0)
    """Hue range in degrees. ``lo > hi`` wraps through red, e.g. ``[340, 20]``."""
    sat: tuple[float, float] = (0.0, 1.0)
    """HSV saturation range."""
    val: tuple[float, float] = (0.0, 1.0)
    """HSV value (brightness) range."""
    z_m: tuple[float, float] = (0.0, 12.0)
    """Height band the class's evidence is collected from. An object cut off by
    the top of the band continues above it, so it is not reported."""
    cell_m: float = 0.05
    """Edge of the cubes evidence is pooled into. Small cells keep thin objects
    apart from their neighbours; large ones are cheaper for big objects."""
    link_cells: int = 1
    """Occupied cells up to this many apart (Chebyshev) join one object, which
    bridges the gaps between sparse samples of a thin object."""
    link_up_cells: int | None = None
    """The same, vertically; ``link_cells`` when omitted. A thin upright
    object, such as a conduit riser, is hit at a few sparse heights (a drone
    at one altitude keeps sampling the same rows), so its vertical gaps are
    wider than any horizontal gap to a neighbour it must stay apart from."""
    neighbour_colour: bool = False
    """Judge each cell's colour pooled with its 26 neighbours'. For pale classes:
    a low-chroma cell rarely holds enough hits to judge alone, and pooling also
    drops the cells between two nearly touching look-alikes (a conduit beside a
    grey breaker panel). Saturated classes are judged well cell by cell."""
    min_hits: int = 10
    """Ray hits an object needs before it is reported."""
    length_m: tuple[float, float] | None = None
    """Range of the object's longest extent, horizontal or vertical."""
    width_m: tuple[float, float] | None = None
    """Range of its middle extent."""
    thickness_m: tuple[float, float] | None = None
    """Range of its shortest extent."""
    height_m: tuple[float, float] | None = None
    """Range of its vertical extent."""
    bottom_m: tuple[float, float] | None = None
    """Range of the height of its lowest point."""
    min_elongation: float | None = None
    """Minimum ratio of its longest extent to its middle one."""
    near_cls: str | None = None
    """A class this one must be found within ``near_m`` of, e.g. a meter on a WALL."""
    near_m: float | None = None
    """Distance limit for ``near_cls``, between the closest observed points."""

    def validate(self, where: str, known: list[str]) -> None:
        """Raise :class:`ConfigError` unless this entry is usable.

        Parameters
        ----------
        where
            Where the entry sits in the YAML, for the error message.
        known
            Every class listed alongside this one, which ``near_cls`` must name.
        """
        _check_cls(self.cls, where)
        for name in ("sat", "val"):
            lo, hi = getattr(self, name)
            if not 0.0 <= lo <= hi <= 1.0:
                msg = f"{where}.{name} must satisfy 0 <= lo <= hi <= 1, got {[lo, hi]}"
                raise ConfigError(msg)
        if not all(0.0 <= h <= _FULL_TURN_DEG for h in self.hue_deg):
            msg = f"{where}.hue_deg must lie in [0, {_FULL_TURN_DEG:g}], got {list(self.hue_deg)}"
            raise ConfigError(msg)
        for name in ("z_m", "length_m", "width_m", "thickness_m", "height_m", "bottom_m"):
            band = getattr(self, name)
            if band is not None and band[0] > band[1]:
                msg = f"{where}.{name} must be [lo, hi] with lo <= hi, got {list(band)}"
                raise ConfigError(msg)
        if self.cell_m <= 0.0 or self.link_cells < 1 or self.min_hits < 1:
            msg = f"{where}: cell_m must be positive, link_cells and min_hits at least 1"
            raise ConfigError(msg)
        if self.link_up_cells is not None and self.link_up_cells < 1:
            msg = f"{where}.link_up_cells must be at least 1, got {self.link_up_cells}"
            raise ConfigError(msg)
        if (self.near_cls is None) != (self.near_m is None):
            msg = f"{where}: near_cls and near_m must be given together"
            raise ConfigError(msg)
        if self.near_cls is not None and self.near_cls not in known:
            msg = f"{where}.near_cls {self.near_cls!r} is not one of the listed classes {known}"
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class PerceptionCfg:
    """The colour-and-geometry object detector (:mod:`canopy.perception`)."""

    extract_hz: float
    """How often evidence is re-clustered into objects."""
    max_range_m: float
    """Hits farther than this are too sparse to help, so they are skipped."""
    colour_noise: float
    """Standard deviation of one hit's colour, per channel, in 8-bit levels: the
    detector's calibration of its sensor."""
    colour_tolerance: float
    """Standard errors of slack a hit or cell's colour gets before it is turned
    away. An object's colour, pooled over all its hits, gets none."""
    min_cell_hits: int
    """Hits a cell needs before it counts, which drops one-off stray samples."""
    box_margin_m: float
    """Clearance added around an object's observed points, so its box outlines
    the object rather than cutting through its outermost surface."""
    ground_snap_m: float
    """A box whose bottom is this close to the ground is extended down to it."""
    split_saving: float
    """An object is cut into parts, each boxed separately, when the parts' boxes
    are at least this fraction smaller in total volume than one box around the
    whole: an L-shaped conduit run is two straight pipes, while a bush or a
    meter never shrinks that much."""
    max_parts: int
    """Most parts one connected object may be cut into."""
    track_match_m: float
    """An object keeps its track id if it is found again within this distance."""
    confidence_hits: float
    """Hits at which confidence reaches ``1 - 1/e``."""
    classes: tuple[PerceptionClassCfg, ...]
    """Every class the detector knows, context classes included."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless every class entry is usable."""
        if self.extract_hz <= 0.0 or self.max_range_m <= 0.0 or self.confidence_hits <= 0.0:
            msg = "perception.extract_hz, max_range_m and confidence_hits must be positive"
            raise ConfigError(msg)
        if self.colour_noise < 0.0 or self.colour_tolerance < 0.0:
            msg = "perception.colour_noise and colour_tolerance must be non-negative"
            raise ConfigError(msg)
        if self.min_cell_hits < 1 or self.max_parts < 1:
            msg = "perception.min_cell_hits and max_parts must be at least 1"
            raise ConfigError(msg)
        if not 0.0 < self.split_saving < 1.0:
            msg = f"perception.split_saving must be in (0, 1), got {self.split_saving}"
            raise ConfigError(msg)
        # A non-positive match distance would re-issue every track id on every
        # extraction, and a negative margin can shrink a box past nothing.
        if self.track_match_m <= 0.0 or self.box_margin_m < 0.0 or self.ground_snap_m < 0.0:
            msg = (
                "perception.track_match_m must be positive, and box_margin_m and "
                "ground_snap_m non-negative"
            )
            raise ConfigError(msg)
        names = [c.cls for c in self.classes]
        if dupes := sorted({n for n in names if names.count(n) > 1}):
            msg = f"perception.classes lists {dupes} more than once"
            raise ConfigError(msg)
        for i, entry in enumerate(self.classes):
            entry.validate(f"perception.classes[{i}] ({entry.cls})", names)


@dataclass(frozen=True, slots=True)
class MapCfg:
    """Occupancy grid, coverage and semantics thresholds."""

    voxel_m: float
    height_m: float
    plan_voxel_m: float
    min_hits: int
    coverage_max_range_m: float
    coverage_max_incidence_deg: float
    ground_band_max_z_m: float
    carve_ray_stride: int
    carve_step_m: float
    carve_stop_short_m: float

    @property
    def plan_factor(self) -> int:
        """Map voxels per planning voxel along each axis."""
        return round(self.plan_voxel_m / self.voxel_m)

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless the planning grid nests in the map grid."""
        if self.voxel_m <= 0:
            msg = f"map.voxel_m must be positive, got {self.voxel_m}"
            raise ConfigError(msg)
        # The planner min-pools whole blocks of map voxels into one planning
        # voxel; a fractional ratio would straddle block edges.
        if self.plan_factor < 1 or not np.isclose(
            self.plan_factor * self.voxel_m, self.plan_voxel_m
        ):
            msg = (
                f"map.plan_voxel_m ({self.plan_voxel_m}) must be a positive integer "
                f"multiple of map.voxel_m ({self.voxel_m})"
            )
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class PlannerCfg:
    """Orbit, frontier and assignment tunables."""

    orbit_altitudes: tuple[float, ...]
    orbit_margin_m: float
    orbit_waypoints_per_ring: int
    frontier_min_voxels: int
    frontier_min_z_m: float
    ground_band_gain: float
    viewpoint_range_m: tuple[float, float]
    viewpoint_gain_radius_m: float
    inspect_bucket_m: float
    inspect_min_voxels: int
    inspect_range_m: tuple[float, float]
    inspect_max_angle_deg: float
    gain_lambda: float
    goal_conflict_radius_m: float
    goal_conflict_penalty: float
    hysteresis: float
    blacklist_s: float
    visited_radius_m: float
    done_ground_coverage: float | None
    takeoff_altitude_m: float
    launch_clear_radius_m: float
    waypoint_tolerance_m: float
    landed_altitude_m: float
    idle_return_s: float


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
    hold_replan_s: float

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
class DetectionColorCfg:
    """Outline colour the viewer draws one detected class in."""

    cls: str
    """Name of a :class:`~canopy.contracts.Cls` member."""
    rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ViewerCfg:
    """Desktop viewer (``canopy-view``) settings."""

    drones: int
    max_drones: int
    width_px: int
    height_px: int
    detection_colors: tuple[DetectionColorCfg, ...]
    """Box outline colour per detected class."""
    detection_default_rgb: tuple[int, int, int]
    """Outline colour for a detected class with no entry in ``detection_colors``."""
    site_rgb: tuple[int, int, int]
    """Outline colour of a suggested battery site whose verdict is PASS."""
    site_warning_rgb: tuple[int, int, int]
    """Outline colour of a suggested site whose verdict is MANUAL_REVIEW."""
    site_reject_rgb: tuple[int, int, int]
    """Outline colour of a suggested site whose verdict is REJECT."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless the default swarm fits the limit."""
        if self.max_drones < 1:
            msg = f"viewer.max_drones must be at least 1, got {self.max_drones}"
            raise ConfigError(msg)
        if not 1 <= self.drones <= self.max_drones:
            msg = f"viewer.drones must be in [1, {self.max_drones}], got {self.drones}"
            raise ConfigError(msg)
        names = [entry.cls for entry in self.detection_colors]
        if dupes := sorted({n for n in names if names.count(n) > 1}):
            msg = f"viewer.detection_colors lists {dupes} more than once"
            raise ConfigError(msg)
        for i, entry in enumerate(self.detection_colors):
            where = f"viewer.detection_colors[{i}]"
            _check_cls(entry.cls, where)
            _check_rgb(entry.rgb, f"{where}.rgb")
        _check_rgb(self.detection_default_rgb, "viewer.detection_default_rgb")
        _check_rgb(self.site_rgb, "viewer.site_rgb")
        _check_rgb(self.site_warning_rgb, "viewer.site_warning_rgb")
        _check_rgb(self.site_reject_rgb, "viewer.site_reject_rgb")


@dataclass(frozen=True, slots=True)
class Config:
    """The whole of ``config/default.yaml``, validated."""

    sim: SimCfg
    sensor: SensorCfg
    worldgen: WorldgenCfg
    perception: PerceptionCfg
    map: MapCfg
    planner: PlannerCfg
    safety: SafetyCfg
    demo: DemoCfg
    viewer: ViewerCfg

    def validate(self) -> None:
        """Run every section's cross-field checks."""
        self.sim.validate()
        self.sensor.validate()
        self.perception.validate()
        self.map.validate()
        self.viewer.validate()
        # The planner treats every pad's column as surveyed clear, so the
        # viewer must not offer a swarm wider than worldgen keeps clear.
        if self.worldgen.launch_area_pads < self.viewer.max_drones:
            msg = (
                f"worldgen.launch_area_pads ({self.worldgen.launch_area_pads}) must be at "
                f"least viewer.max_drones ({self.viewer.max_drones})"
            )
            raise ConfigError(msg)


def _check_cls(name: str, where: str) -> None:
    """Raise :class:`ConfigError` unless ``name`` names a :class:`Cls` member."""
    if name not in Cls.__members__:
        msg = f"{where}: {name!r} is not a class; expected one of {list(Cls.__members__)}"
        raise ConfigError(msg)


def _check_rgb(rgb: tuple[int, int, int], where: str) -> None:
    """Raise :class:`ConfigError` unless every channel is an 8-bit level."""
    if not all(0 <= c <= _RGB_MAX for c in rgb):
        msg = f"{where} channels must be in [0, {_RGB_MAX}], got {list(rgb)}"
        raise ConfigError(msg)


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
    """Construct a config dataclass, rejecting unknown keys and missing required ones.

    A field with a default may be left out, which is what lets a perception
    class entry state only the rules it needs. Fields without one, which is
    every field of every section, must be present.
    """
    hints = get_type_hints(cls)
    declared = fields(cls)  # type: ignore[arg-type]
    names = [f.name for f in declared]
    required = [f.name for f in declared if f.default is MISSING and f.default_factory is MISSING]

    if unknown := sorted(set(data) - set(names)):
        msg = f"{where or 'config'}: unknown key(s) {unknown}; known keys are {names}"
        raise ConfigError(msg)
    if missing := sorted(set(required) - set(data)):
        msg = f"{where or 'config'}: missing key(s) {missing}"
        raise ConfigError(msg)

    kwargs = {
        n: _coerce(hints[n], data[n], f"{where}.{n}" if where else n) for n in names if n in data
    }
    return cls(**kwargs)


def build_section(cls: type[_T], data: Any, where: str) -> _T:
    """Build any config dataclass from a YAML mapping, as strictly as ``load_config`` does.

    Unknown keys, missing required keys and wrongly typed values all raise.
    Public so a stage with its own YAML (the site stage's ``rules.yaml``) gets
    the same guarantees without a second, drifting copy of the coercion.

    Parameters
    ----------
    cls
        A dataclass whose fields are bools, ints, floats, strings, tuples of
        those, ``X | None``, or nested dataclasses of the same.
    data
        The parsed YAML node.
    where
        Location for error messages, e.g. ``"rules.yaml rules[2]"``.

    Raises
    ------
    ConfigError
        If ``data`` is not a mapping or does not fit ``cls``.
    """
    if not isinstance(data, dict):
        msg = f"{where}: expected a mapping, got {type(data).__name__}"
        raise ConfigError(msg)
    return _build(cls, data, where)


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

    Placement rules stay a plain mapping here on purpose: they stand in for the
    real SSR checklist and their shape is expected to change wholesale. The
    site stage gives them their typed meaning (``canopy.site.load_site_rules``).
    """
    resolved = Path(path) if path is not None else default_rules_path()
    return _read_yaml(resolved)
