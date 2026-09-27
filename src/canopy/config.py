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
    "InvariantsCfg",
    "MapCfg",
    "PerceptionCfg",
    "PerceptionClassCfg",
    "PlannerCfg",
    "ResidentialCfg",
    "SafetyCfg",
    "SensorCfg",
    "SimCfg",
    "SiteCfg",
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

#: Fewer waypoints than this cannot close an orbit ring (matches
#: ``canopy.planning.waypoints._MIN_ORBIT_WAYPOINTS``).
_MIN_DEMO_ORBIT_WAYPOINTS = 3

#: Degrees in a half turn; the widest an incidence angle can meaningfully be.
_HALF_TURN_DEG = 180.0


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
        if self.timeout_s <= 0:
            msg = f"sim.timeout_s must be positive, got {self.timeout_s}"
            raise ConfigError(msg)
        if self.battery_drain_per_10s < 0:
            msg = (
                f"sim.battery_drain_per_10s must be non-negative, got {self.battery_drain_per_10s}"
            )
            raise ConfigError(msg)
        if not 0.0 <= self.rth_battery <= 1.0:
            msg = f"sim.rth_battery must be in [0, 1], got {self.rth_battery}"
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
        if self.az_rays < 1 or self.el_rays < 1:
            msg = (
                f"sensor.az_rays and el_rays must be at least 1, got "
                f"{self.az_rays} and {self.el_rays}"
            )
            raise ConfigError(msg)
        if self.max_range_m <= 0.0:
            msg = f"sensor.max_range_m must be positive, got {self.max_range_m}"
            raise ConfigError(msg)
        if self.el_min_deg > self.el_max_deg:
            msg = (
                f"sensor.el_min_deg ({self.el_min_deg}) must not exceed el_max_deg "
                f"({self.el_max_deg})"
            )
            raise ConfigError(msg)
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
class InvariantsCfg:
    """Hard rules every generated property must meet, and how hard to try to meet them.

    Checked as each role is placed, not once at the end: bushes are placed to
    hide the meter and windows are fitted around it, so a meter moved after
    them would leave both wrong. A role that breaks a rule is redrawn with its
    corner clearance relaxed rung by rung before generation gives up (see
    ``docs/adr/0015-address-seeded-sites.md``).
    """

    meter_centre_height_m: tuple[float, float]
    """Allowed height of the meter's centre above grade (spec.md, ``test_worldgen``)."""
    meter_min_corner_clearance_m: float
    """Closest a meter may sit to either end of its wall, after every relaxation."""
    meter_working_space_m: float
    """Clear depth in front of the meter, inside the lot and outside the house,
    that anyone reading or servicing it needs. A hard rule."""
    corner_clearance_ladder_m: tuple[float, ...]
    """Corner clearances to retry a failed role with, in order."""
    repairs_per_rung: int
    """Redraws at each rung of the ladder before stepping down to the next."""
    meter_approach_m: float
    """Clear depth in front of the meter, inside the geofence, that the INSPECT
    close-up needs: its 1.5 m standoff plus the shield's 0.6 m inflation.

    Reported in :attr:`~canopy.contracts.SceneManifest.notes`, never repaired:
    real meters do sit in narrow side yards, and redrawing every such property
    would quietly bias the simulation toward easy houses."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the rules or the repair ladder are unusable."""
        lo, hi = self.meter_centre_height_m
        if not 0.0 < lo < hi:
            msg = f"worldgen.invariants.meter_centre_height_m must be 0 < lo < hi, got {[lo, hi]}"
            raise ConfigError(msg)
        floor = self.meter_min_corner_clearance_m
        if floor < 0.0:
            msg = f"worldgen.invariants.meter_min_corner_clearance_m must be >= 0, got {floor}"
            raise ConfigError(msg)
        if below := [v for v in self.corner_clearance_ladder_m if v < floor]:
            msg = (
                f"worldgen.invariants.corner_clearance_ladder_m has rungs {below} below "
                f"meter_min_corner_clearance_m ({floor}), which the check would then reject"
            )
            raise ConfigError(msg)
        if self.repairs_per_rung < 1:
            msg = f"worldgen.invariants.repairs_per_rung must be >= 1, got {self.repairs_per_rung}"
            raise ConfigError(msg)
        for name in ("meter_working_space_m", "meter_approach_m"):
            if getattr(self, name) < 0.0:
                msg = f"worldgen.invariants.{name} must be >= 0"
                raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class ResidentialCfg:
    """The residential check: how much each piece of evidence moves the odds.

    Each ``w_`` weight is the log-odds that piece of evidence adds when present;
    absent evidence adds nothing. The address text itself says almost nothing,
    so the evidence is about what stands at the address.
    """

    accept_p: float
    """At or above this probability the address is taken as a home."""
    reject_p: float
    """At or below this probability it is refused; in between, the user is asked."""
    house_area_m2: tuple[float, float]
    """Footprint area range that reads as one house rather than a shed or a block."""
    w_building_house: float
    """The building under the pin is mapped as a house type (house, detached, ...)."""
    w_building_residential: float
    """It is mapped as residential but not a house type (``residential``, ``apartments``)."""
    w_building_nonresidential: float
    """It is mapped as commercial, industrial, civic or religious."""
    w_poi_inside: float
    """A shop, office or amenity is mapped inside the footprint."""
    w_landuse_residential: float
    """The pin lies in an area mapped ``landuse=residential``."""
    w_landuse_nonresidential: float
    """The pin lies in an area mapped commercial, retail or industrial."""
    w_house_sized: float
    """The footprint area falls within :attr:`house_area_m2`."""
    w_low_rise: float
    """The building has at most three levels."""
    w_unit_suite: float
    """The address names a suite, which homes rarely have."""

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless the decision bands are ordered."""
        if not 0.0 < self.reject_p < self.accept_p < 1.0:
            msg = (
                "worldgen.site.residential must have 0 < reject_p < accept_p < 1, got "
                f"reject_p={self.reject_p}, accept_p={self.accept_p}"
            )
            raise ConfigError(msg)
        lo, hi = self.house_area_m2
        if not 0.0 < lo < hi:
            msg = f"worldgen.site.residential.house_area_m2 must be 0 < lo < hi, got {[lo, hi]}"
            raise ConfigError(msg)


#: Address search providers ``worldgen.site.geocoder`` may name.
_GEOCODERS = frozenset({"auto", "photon", "geoapify"})
#: Length of an ISO 3166-1 alpha-2 country code.
_COUNTRY_CODE_LEN = 2


@dataclass(frozen=True, slots=True)
class SiteCfg:
    """Address-seeded sites: where the data comes from and how a real site is rebuilt."""

    geocoder: str
    """``"auto"``, ``"photon"`` or ``"geoapify"``. ``"auto"`` picks Geoapify when
    ``geoapify_key_env`` is set in the environment, else Photon."""
    geocoder_url: str
    """Photon ``/api/`` endpoint for address search-as-you-type."""
    geoapify_url: str
    """Geoapify address-autocomplete endpoint."""
    geoapify_key_env: str
    """Name of the environment variable holding the Geoapify API key. The key
    itself never goes in config, which is checked in."""
    suggest_countries: tuple[str, ...]
    """ISO 3166-1 alpha-2 codes suggestions are limited to; empty for worldwide."""
    overpass_url: str
    """Overpass API ``interpreter`` endpoint for OpenStreetMap features."""
    user_agent: str
    """Sent with every request; both public services require an identifying one."""
    timeout_s: float
    """Socket timeout for one address-search request."""
    overpass_timeout_s: float
    """Socket timeout for one Overpass request. Much longer than ``timeout_s``:
    the public server queues a request until a slot frees up, and that wait
    (measured at up to 20 s under load) counts against it before the query
    even starts."""
    overpass_query_timeout_s: float
    """Run-time limit sent inside the Overpass query. Kept below
    ``overpass_timeout_s`` so the server abandons a query before this client
    does, and small because the public server schedules a query with a small
    limit sooner."""
    overpass_retries: int
    """Retries after a busy (429/502/503/504) or timed-out Overpass attempt."""
    overpass_backoff_s: float
    """Wait before the first retry; each later one waits twice as long."""
    suggest_limit: int
    """Address suggestions offered per keystroke."""
    aoi_radius_m: float
    """Half-width of the area fetched around the address."""
    snap_radius_m: float
    """How far from the pin to look for the building when none contains it."""
    max_neighbours: int
    """Nearest neighbouring buildings kept as background scenery."""
    party_wall_gap_m: float
    """A house wall with a neighbour this close outside it is shared, not exterior."""
    storey_m: float
    """Storey height, for counting storeys in a mapped total height."""
    roof_allowance_m: float
    """Height of the roof above the top storey, likewise."""
    street_centre_offset_m: float
    """Front lot line to street centreline: where ``frontage_strip`` lays the street."""
    default_street_width_m: float
    default_front_yard_m: float
    """Front yard when no street was mapped to measure one from."""
    min_front_yard_m: float
    """Smallest front yard; keeps the launch pads clear of the house."""
    default_side_yard_m: float
    """Side yard when no neighbour was mapped beside the house."""
    default_back_yard_m: float
    """Back yard when no neighbour was mapped behind the house."""
    min_lot_m: tuple[float, float]
    max_lot_m: tuple[float, float]
    """The mapper's voxel grid is sized from the lot, so this bounds its memory."""
    max_footprint_m2: float
    """Largest house footprint the generator will rebuild."""
    max_levels: int
    """Most storeys the house model library can represent."""
    raster_m: float
    """Cell size when fitting axis-aligned blocks to a mapped footprint."""
    max_masses: int
    """Most blocks a footprint is fitted with; the house rule composes at most three."""
    min_mass_m: float
    """Narrowest block worth keeping: anything thinner is a bay window, not a wing."""
    min_fit_iou: float
    """Refuse a footprint the blocks overlap less than this (intersection over union)."""
    warn_fit_iou: float
    """Note a fit below this in the snapshot, but still build it."""
    residential: ResidentialCfg

    def validate(self) -> None:
        """Raise :class:`ConfigError` if a limit or a threshold is out of order."""
        positive = (
            "timeout_s",
            "overpass_timeout_s",
            "overpass_query_timeout_s",
            "aoi_radius_m",
            "snap_radius_m",
            "storey_m",
            "default_street_width_m",
            "max_footprint_m2",
            "raster_m",
            "min_mass_m",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                msg = f"worldgen.site.{name} must be positive, got {getattr(self, name)}"
                raise ConfigError(msg)
        for name in ("suggest_limit", "max_levels", "max_masses"):
            if getattr(self, name) < 1:
                msg = f"worldgen.site.{name} must be at least 1, got {getattr(self, name)}"
                raise ConfigError(msg)
        if self.geocoder not in _GEOCODERS:
            msg = (
                f"worldgen.site.geocoder must be one of {sorted(_GEOCODERS)}, got {self.geocoder!r}"
            )
            raise ConfigError(msg)
        bad_codes = [
            c for c in self.suggest_countries if len(c) != _COUNTRY_CODE_LEN or not c.isalpha()
        ]
        if bad_codes:
            msg = f"worldgen.site.suggest_countries must be two-letter ISO codes, got {bad_codes}"
            raise ConfigError(msg)
        if self.overpass_query_timeout_s >= self.overpass_timeout_s:
            msg = (
                f"worldgen.site.overpass_query_timeout_s ({self.overpass_query_timeout_s}) must be "
                f"below overpass_timeout_s ({self.overpass_timeout_s}), or this client hangs up "
                "on answers in flight"
            )
            raise ConfigError(msg)
        if self.overpass_retries < 0 or self.max_neighbours < 0:
            msg = "worldgen.site.overpass_retries and max_neighbours must be >= 0"
            raise ConfigError(msg)
        if any(lo > hi for lo, hi in zip(self.min_lot_m, self.max_lot_m, strict=True)):
            msg = (
                f"worldgen.site.min_lot_m {list(self.min_lot_m)} must not exceed "
                f"max_lot_m {list(self.max_lot_m)}"
            )
            raise ConfigError(msg)
        if not 0.0 < self.min_fit_iou <= self.warn_fit_iou <= 1.0:
            msg = (
                "worldgen.site must have 0 < min_fit_iou <= warn_fit_iou <= 1, got "
                f"{self.min_fit_iou} and {self.warn_fit_iou}"
            )
            raise ConfigError(msg)
        self.residential.validate()


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
    invariants: InvariantsCfg
    site: SiteCfg

    def validate(self) -> None:
        """Run the nested sections' checks."""
        self.invariants.validate()
        self.site.validate()


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
    house_band_z_m: tuple[float, float]
    """Height band a mapped plan column must fill to count as building wall."""
    house_min_fill: float
    """Fraction of :attr:`house_band_z_m` that must be occupied, in ``(0, 1]``."""
    house_min_wall_m: float
    """Least run of wall, in metres, that counts as a building rather than clutter."""
    survey_margin_m: float
    """How far past the inferred house the survey reaches, in plan."""

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
        lo, hi = self.house_band_z_m
        if not 0.0 <= lo < hi <= self.height_m:
            msg = (
                f"map.house_band_z_m {list(self.house_band_z_m)} must be [low, high] "
                f"inside [0, map.height_m = {self.height_m}]"
            )
            raise ConfigError(msg)
        if not 0.0 < self.house_min_fill <= 1.0:
            msg = f"map.house_min_fill must be in (0, 1], got {self.house_min_fill}"
            raise ConfigError(msg)
        if self.survey_margin_m < 0.0:
            msg = f"map.survey_margin_m must not be negative, got {self.survey_margin_m}"
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

    def validate(self) -> None:
        """Raise :class:`ConfigError` if a range, radius or threshold is unusable."""
        if not self.orbit_altitudes or any(a <= 0.0 for a in self.orbit_altitudes):
            msg = (
                "planner.orbit_altitudes must be non-empty and positive, got "
                f"{list(self.orbit_altitudes)}"
            )
            raise ConfigError(msg)
        if self.orbit_waypoints_per_ring < 1:
            msg = (
                "planner.orbit_waypoints_per_ring must be at least 1, got "
                f"{self.orbit_waypoints_per_ring}"
            )
            raise ConfigError(msg)
        # A viewpoint or waypoint distance, a bucket edge and the swarm's own
        # takeoff climb must all be positive to mean anything.
        positive = ("inspect_bucket_m", "takeoff_altitude_m", "waypoint_tolerance_m")
        for name in positive:
            if getattr(self, name) <= 0.0:
                msg = f"planner.{name} must be positive, got {getattr(self, name)}"
                raise ConfigError(msg)
        # These are radii, gains and durations: zero legitimately disables the
        # behaviour they gate (no bonus, no conflict penalty, no cooldown), but
        # negative is never meaningful.
        non_negative = (
            "orbit_margin_m",
            "ground_band_gain",
            "viewpoint_gain_radius_m",
            "gain_lambda",
            "goal_conflict_radius_m",
            "goal_conflict_penalty",
            "hysteresis",
            "blacklist_s",
            "visited_radius_m",
            "launch_clear_radius_m",
            "landed_altitude_m",
            "idle_return_s",
        )
        for name in non_negative:
            if getattr(self, name) < 0.0:
                msg = f"planner.{name} must be non-negative, got {getattr(self, name)}"
                raise ConfigError(msg)
        if self.frontier_min_voxels < 0 or self.inspect_min_voxels < 0:
            msg = "planner.frontier_min_voxels and inspect_min_voxels must be non-negative"
            raise ConfigError(msg)
        for name in ("viewpoint_range_m", "inspect_range_m"):
            lo, hi = getattr(self, name)
            if not 0.0 < lo <= hi:
                msg = f"planner.{name} must be [lo, hi] with 0 < lo <= hi, got {[lo, hi]}"
                raise ConfigError(msg)
        if not 0.0 < self.inspect_max_angle_deg <= _HALF_TURN_DEG:
            msg = (
                f"planner.inspect_max_angle_deg must be in (0, {_HALF_TURN_DEG:g}], got "
                f"{self.inspect_max_angle_deg}"
            )
            raise ConfigError(msg)
        if self.done_ground_coverage is not None and not 0.0 <= self.done_ground_coverage <= 1.0:
            msg = (
                "planner.done_ground_coverage must be null or in [0, 1], got "
                f"{self.done_ground_coverage}"
            )
            raise ConfigError(msg)


@dataclass(frozen=True, slots=True)
class SafetyCfg:
    """Shield margins and geofence."""

    drone_radius_m: float
    margin_m: float
    min_separation_m: float
    ceiling_m: float
    geofence_inset_m: float
    envelope_x_m: tuple[float, float]
    """Operator flight envelope along world x, metres from the launch pads' centroid."""
    envelope_y_m: tuple[float, float]
    """Operator flight envelope along world y, metres from the launch pads' centroid."""
    stuck_window_s: float
    stuck_min_progress_m: float
    hold_replan_s: float

    @property
    def inflation_m(self) -> float:
        """Total radius by which occupied space is grown before planning."""
        return self.drone_radius_m + self.margin_m

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless the envelope contains the launch point."""
        for name, (lo, hi) in (("x", self.envelope_x_m), ("y", self.envelope_y_m)):
            if not lo < 0.0 < hi:
                msg = (
                    f"safety.envelope_{name}_m {[lo, hi]} must straddle 0: the envelope "
                    "is measured from the launch point and has to contain it"
                )
                raise ConfigError(msg)


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

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the demo flight path is unflyable.

        Mirrors the checks :func:`~canopy.planning.waypoints.demo_path` and
        :func:`~canopy.planning.waypoints.orbit_ring` would otherwise raise
        deep in ``canopy-fly``, so a bad value fails at load time instead.
        """
        if self.takeoff_altitude_m <= 0.0:
            msg = f"demo.takeoff_altitude_m must be positive, got {self.takeoff_altitude_m}"
            raise ConfigError(msg)
        if self.orbit_radius_m <= 0.0:
            msg = f"demo.orbit_radius_m must be positive, got {self.orbit_radius_m}"
            raise ConfigError(msg)
        if self.orbit_waypoints < _MIN_DEMO_ORBIT_WAYPOINTS:
            msg = (
                f"demo.orbit_waypoints must be at least {_MIN_DEMO_ORBIT_WAYPOINTS} to "
                f"close a loop, got {self.orbit_waypoints}"
            )
            raise ConfigError(msg)
        if self.laps < 1:
            msg = f"demo.laps must be at least 1, got {self.laps}"
            raise ConfigError(msg)
        if self.waypoint_tolerance_m <= 0.0:
            msg = f"demo.waypoint_tolerance_m must be positive, got {self.waypoint_tolerance_m}"
            raise ConfigError(msg)


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
    review_api_url: str
    """Base URL of the review API a finished run is uploaded to (ADR 0017).

    Read only by :meth:`~canopy.viz.ViewerSession.intake`: Python never
    calls it itself, the page does, once :meth:`~canopy.viz.ViewerSession.run_record`
    gives it something to upload.
    """
    scene_cache_max: int
    """Generated scene directories (``out/viewer/<seed>``) kept, most recently used first.

    Every re-roll and address build writes a fresh one; older ones past this
    count are deleted. The loaded scene's directory is always kept regardless.
    """

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
        if self.scene_cache_max < 1:
            msg = f"viewer.scene_cache_max must be at least 1, got {self.scene_cache_max}"
            raise ConfigError(msg)


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
        self.worldgen.validate()
        self.perception.validate()
        self.map.validate()
        self.planner.validate()
        self.safety.validate()
        self.demo.validate()
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
