"""The desktop viewer: a headless mapping mission and the window that renders it.

``canopy-view`` opens a native window (pywebview) whose page draws a real
mapping mission -- the same :class:`~canopy.planning.run.MissionRun` loop
``canopy-fly`` and the batch evaluator drive -- with three.js. The work is
split so that only one function knows a window exists:

:class:`ViewerSession`
    Pure Python. Owns the generated property, the mission run and simulated
    time. The page drives it, but so can a test, with no display attached.
:func:`launch`
    The single place pywebview is imported, lazily, so a machine without the
    ``viewer`` dependency group still imports :mod:`canopy` cleanly.

Why a web renderer in a desktop window, rather than the PyBullet GUI or Rerun,
is recorded in ``docs/adr/0005-desktop-viewer.md``.

Time is driven by the page. Each animation frame it reports how much wall-clock
time has passed; the session moves a playback clock on by that much and returns
poses interpolated between the two recorded control ticks either side of it.
Fixed ticks keep the motion identical to a headless run; interpolation is what
lets a 20 Hz simulation look smooth at the display's refresh rate. The settings
panel's time-speed slider scales that wall-clock time on this side, after it is
clamped, so a slow frame at 10x still cannot demand more than ten times the
catch-up a slow frame at 1x can.

Ticks are not equally expensive. Most take about a millisecond, but a replan
(clearance field, frontier search, assignment, paths) takes the better part of
a second, once per simulated second. Run inline in the page's frame call, that
froze the drones for the whole replan while the camera kept moving, then made
them jump. So :meth:`ViewerSession.start` moves the simulation onto its own
thread, which records a snapshot per tick into a buffer kept ``_LEAD_S`` of
wall time ahead of playback; a frame call only reads the buffer. When the
simulation cannot keep up -- a high time scale, a slow machine -- playback slows
down smoothly as the buffer drains rather than stalling when it runs dry.
Without :meth:`~ViewerSession.start` the session ticks inline, deterministically,
which is what the tests drive.

The property mesh reaches the page through :meth:`ViewerSession.world`, base64
encoded rather than sent as JSON number arrays: the pywebview bridge round-trips
through JSON, and a few hundred thousand triangles is a few megabytes as raw
binary but many times that as a JSON list of floats. Newly seen triangles are
reported the same way, as the ids the mapper has just revealed, so the page can
paint them in true colour without re-fetching the whole mesh.

The drones are drawn from the authored ``canopy_scout`` model in the asset
library, read here with the same OBJ reader worldgen uses and handed to the page
as plain arrays. The page ships only three.js core, not its loaders, and cannot
see ``assets/`` from the web directory pywebview serves -- and keeping the file
format on the Python side means one OBJ reader, not two.

Every frame also lists the objects perception has classified so far, each as
an upright box the page outlines in its class's colour
(``viewer.detection_colors``). They come from the mapper's
:attr:`~canopy.contracts.MapState.discovered`, which the detector builds from
range and colour alone, so the outlines show what the swarm worked out, not
what the scene manifest says.

Once mapping is complete -- the mission has left exploration -- the session
asks the site stage for the best battery sites on the finished map
(:func:`~canopy.site.suggest_sites`, rules from ``config/rules.yaml``) and
every later frame carries them. They are computed on the simulation thread at
the tick exploration ends and ride along in that tick's snapshot, so they
appear on screen when the end of mapping does, not seconds before it. The page
draws each site with the ``base_core_battery`` model, served like the drone.
"""

from __future__ import annotations

import base64
import functools
import math
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from canopy import mathutil
from canopy.contracts import (
    DiscoveredObject,
    DroneState,
    Points,
    SceneGeometry,
    SceneManifest,
    SiteCandidate,
    Vec3,
)
from canopy.errors import ConfigError, DependencyMissingError, SimulationError, SiteError
from canopy.log import get_logger
from canopy.planning import MissionRun
from canopy.sim import RaySensor, load_geometry
from canopy.site import SiteRules, load_site_rules, suggest_sites
from canopy.worldgen import default_assets_dir, generate_field, read_mtl, read_obj

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = [
    "BATTERY_MODEL",
    "DRONE_MODEL",
    "MAX_TIME_SCALE",
    "MIN_DRONES",
    "MIN_TIME_SCALE",
    "WEB_DIR",
    "ViewerSession",
    "battery_model",
    "drone_model",
    "launch",
]

_log = get_logger(__name__)

#: A swarm of zero is not a scene worth rendering.
MIN_DRONES = 1

#: Bounds of the time-speed slider: simulated seconds per wall-clock second.
#: The top end is what the simulation thread can sustain: a 3-drone mission
#: runs at roughly 2.5-3x real time, so a higher setting would only be
#: capped by the playback throttle and look like the slider does nothing.
MIN_TIME_SCALE = 0.5
MAX_TIME_SCALE = 2.0

#: The page, its script and its vendored three.js.
WEB_DIR = Path(__file__).resolve().parent / "web"

#: The drone model, relative to the asset root (the directory holding
#: ``models/`` and ``obj_export/``). Its MTL sits beside it with the same stem.
DRONE_MODEL = Path("obj_export/assets/canopy_scout.obj")

#: The battery model drawn at each suggested site, relative to the asset root.
BATTERY_MODEL = Path("obj_export/assets/base_core_battery.obj")

#: Mission phases during which the map is still being built. Site suggestions
#: wait until the mission has left them.
_MAPPING_PHASES = frozenset({"takeoff", "explore"})

#: Longest wall-clock gap one call may simulate. A window that was dragged,
#: minimised or paused in a debugger would otherwise ask for seconds of catch-up
#: at once, and the drones would visibly teleport.
_MAX_FRAME_S = 0.25

#: How far ahead of playback, in wall-clock seconds, the simulation thread
#: records. Longer than the slowest replan, so one never drains the buffer.
_LEAD_S = 2.0

#: Playback runs at full speed while at least this fraction of the lead is
#: buffered, and slows in proportion below it.
_FULL_SPEED_FRACTION = 0.5

#: How long the simulation thread sleeps when the buffer is full.
_IDLE_WAIT_S = 0.005

#: Snapshot times are sums of ``dt``; this absorbs their rounding.
_T_EPS = 1e-9

#: Seeds are drawn from this many bits, so they stay readable in the UI.
_SEED_BITS = 31

#: Where generated scenes live by default, one subdirectory per seed.
_DEFAULT_SCENE_DIR = Path("out") / "viewer"


def _b64(array: npt.NDArray[Any]) -> str:
    """Base64-encode an array's raw bytes, little-endian, for the JS bridge."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


@dataclass(frozen=True, slots=True, eq=False)
class _Snapshot:
    """Everything a frame shows about one control tick, captured when it ran.

    Playback trails the simulation, so a frame cannot read the live mission:
    by the time a tick is on screen the controller may be seconds further on.
    """

    t: float
    states: list[DroneState]
    phase: str
    ground_band: float
    total: float
    #: Per drone id: task kind and goal.
    tasks: dict[int, tuple[str, Vec3]]
    landed: frozenset[int]
    #: Target centroids, shape ``(k, 3)``.
    frontiers: Points
    #: Triangle ids first seen on this tick.
    revealed: npt.NDArray[np.int64]
    #: Objects classified so far. The mapper replaces, never mutates, the
    #: objects it publishes, so holding them here needs no copy.
    detections: tuple[DiscoveredObject, ...]
    #: Suggested battery sites as the page draws them; ``None`` while mapping.
    site: dict[str, Any] | None


def _snapshot(run: MissionRun, site: dict[str, Any] | None = None) -> _Snapshot:
    """Capture ``run`` as it stands after its latest tick."""
    frontiers = run.controller.frontiers
    return _Snapshot(
        t=run.t,
        states=run.drones,
        phase=str(run.phase),
        ground_band=run.mapper.coverage_ground_band,
        total=run.mapper.coverage_total,
        tasks={i: (task.kind, task.goal.copy()) for i, task in run.controller.tasks.items()},
        landed=run.controller.landed,
        frontiers=np.array([f.centroid for f in frontiers], dtype=np.float64).reshape(-1, 3),
        revealed=run.mapper.pop_revealed(),
        detections=tuple(run.mapper.state.discovered.values()),
        site=site,
    )


class ViewerSession:
    """A swarm mapping one generated property, advanced on demand.

    Generating the property and building its raycasting scene costs roughly a
    second, so both are cached per seed and only rebuilt when the seed changes;
    a drone-count change reuses them and only rebuilds the mission itself.

    Two locks, always taken in this order: ``_sim_lock`` guards the mission
    and is held for a whole tick; ``_lock`` guards the snapshot buffer and
    playback clock and is only ever held briefly. pywebview delivers page
    calls on worker threads, so a frame call must never wait on the first --
    that would bring back exactly the stall the simulation thread removes.

    Parameters
    ----------
    cfg
        Validated configuration. Reads ``sim``, ``safety`` and ``viewer``.
    drones
        Initial swarm size. Defaults to ``cfg.viewer.drones``.
    seed
        Scene seed. Defaults to a fresh random one.
    scene_dir
        Directory generated scenes are written under, one subdirectory per
        seed. Defaults to ``out/viewer``. Tests pass a ``tmp_path``.
    site_rules
        Battery placement rules. Defaults to the shipped ``config/rules.yaml``.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        drones: int | None = None,
        seed: int | None = None,
        scene_dir: Path | None = None,
        site_rules: SiteRules | None = None,
    ) -> None:
        self._cfg = cfg
        self._site_rules = load_site_rules() if site_rules is None else site_rules
        # Suggested sites for the current mission, once mapping is complete.
        self._site: dict[str, Any] | None = None
        self._sim_lock = threading.Lock()
        self._lock = threading.Lock()
        self._scene_dir = _DEFAULT_SCENE_DIR if scene_dir is None else Path(scene_dir)
        self._n = self._checked_count(cfg.viewer.drones if drones is None else drones)
        self._seed = _fresh_seed() if seed is None else seed
        # Bumped on every reset so the page can drop frames that were already in
        # flight when the mission changed underneath them.
        self._epoch = -1
        # Declared, not assigned: set for real by _load_scene()/_reset() below,
        # which is the only place either ever runs before a public method could
        # observe them.
        self._manifest: SceneManifest
        self._geometry: SceneGeometry
        self._sensor: RaySensor
        self._world_payload: dict[str, Any]
        self._run: MissionRun
        # The tick on screen (at or before the playback clock), the recorded
        # ticks after it, and revealed ids passed but not yet sent to the page.
        self._shown: _Snapshot
        self._buffer: deque[_Snapshot] = deque()
        self._unsent: list[npt.NDArray[np.int64]] = []
        self._play_t = 0.0
        # A playback setting, not part of the mission: it survives resets.
        self._time_scale = 1.0
        # Outline colour per detected class name, on the page's [0, 1] scale.
        self._box_colors = {
            entry.cls: [c / 255.0 for c in entry.rgb] for entry in cfg.viewer.detection_colors
        }
        self._box_default = [c / 255.0 for c in cfg.viewer.detection_default_rgb]
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._load_scene()
        self._reset()

    # -- introspection ------------------------------------------------------
    @property
    def config(self) -> Config:
        """The configuration the session was built from."""
        return self._cfg

    @property
    def n_drones(self) -> int:
        """Current swarm size."""
        return self._n

    @property
    def seed(self) -> int:
        """Seed of the current scene."""
        return self._seed

    @property
    def epoch(self) -> int:
        """Number of resets so far; increases whenever the mission is rebuilt."""
        return self._epoch

    @property
    def sim_time_s(self) -> float:
        """Simulated time of the tick on screen, seconds since the last reset."""
        return self._shown.t

    @property
    def time_scale(self) -> float:
        """Simulated seconds advanced per wall-clock second."""
        return self._time_scale

    @property
    def states(self) -> list[DroneState]:
        """Drone states at the tick on screen."""
        return list(self._shown.states)

    # -- simulation thread ------------------------------------------------------
    def start(self) -> None:
        """Run the simulation on its own thread, ahead of playback.

        Until this is called :meth:`advance` ticks inline, which is simple and
        deterministic but stalls a frame for as long as its slowest tick.
        Idempotent.
        """
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._produce_loop, name="canopy-sim", daemon=True)
        self._worker.start()

    def close(self) -> None:
        """Stop the simulation thread, if running, and wait for its tick to finish."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        self._stop.set()
        worker.join()

    # -- commands from the page ----------------------------------------------
    def set_drone_count(self, n: int) -> dict[str, Any]:
        """Rebuild the mission with ``n`` drones, all back on their pads.

        The property is not regenerated: only the mission (pads, controller,
        world/sensor wiring) is rebuilt, reusing the cached scene.

        Raises
        ------
        ConfigError
            If ``n`` is outside ``[MIN_DRONES, viewer.max_drones]``.
        """
        with self._sim_lock, self._lock:
            self._n = self._checked_count(n)
            self._reset()
            return self._scene()

    def restart(self) -> dict[str, Any]:
        """Rewind to simulated time zero: same property, same swarm, fresh mission.

        Everything the swarm learned (map, detections, sites) goes with the old
        mission; the property and the time scale stay.
        """
        with self._sim_lock, self._lock:
            self._reset()
            return self._scene()

    def new_scene(self, seed: int | None = None) -> dict[str, Any]:
        """Start over with a new seed (random unless given), regenerating the property."""
        with self._sim_lock, self._lock:
            self._seed = _fresh_seed() if seed is None else seed
            self._load_scene()
            self._reset()
            return self._scene()

    def set_time_scale(self, scale: float) -> dict[str, Any]:
        """Run simulated time at ``scale`` times wall-clock speed.

        Only the rate changes: the mission is not rebuilt and the control tick
        stays fixed, so a faster run takes the same steps as a 1x one, just
        more of them per frame. With the simulation thread running, a scale
        it cannot sustain plays back at whatever rate it does sustain.

        Raises
        ------
        ConfigError
            If ``scale`` is outside ``[MIN_TIME_SCALE, MAX_TIME_SCALE]`` or not
            finite.
        """
        with self._lock:
            if not MIN_TIME_SCALE <= scale <= MAX_TIME_SCALE:
                msg = f"time scale must be in [{MIN_TIME_SCALE}, {MAX_TIME_SCALE}], got {scale}"
                raise ConfigError(msg)
            self._time_scale = scale
            return self._scene()

    def scene(self) -> dict[str, Any]:
        """Describe the static layout the page builds once per reset.

        Returns
        -------
        dict
            ``epoch``, ``seed``, ``drones``, ``max_drones``, ``time_scale``,
            ``min_time_scale``, ``max_time_scale``, ``pads`` (one
            ``[x, y, z]`` per drone) and ``lot`` (``[[xmin, ymin, zmin],
            [xmax, ymax, zmax]]``). World frame, metres, Z-up.
        """
        with self._lock:
            return self._scene()

    def world(self) -> dict[str, Any]:
        """Describe the generated property's mesh, cached per seed.

        Returns
        -------
        dict
            ``seed`` and ``objects``: per object its ``id``, ``cls`` (class
            name), ``color`` (``[r, g, b]`` in ``[0, 1]``), ``background``,
            ``tri_offset`` (its first global triangle id) and base64
            ``positions`` (little-endian float32 ``xyz``, world Z-up) /
            ``indices`` (little-endian uint32) into them.
        """
        with self._lock:
            return self._world_payload

    def advance(self, wall_dt_s: float) -> dict[str, Any]:
        """Move playback on by ``wall_dt_s`` seconds and return the pose to draw.

        Parameters
        ----------
        wall_dt_s
            Wall-clock time since the previous call, in seconds. Clamped to
            ``_MAX_FRAME_S``, then multiplied by :attr:`time_scale`. Without
            the simulation thread, the ticks it needs are run here; with it,
            playback also slows as the recorded lead runs short, and never
            passes the newest recorded tick.

        Returns
        -------
        dict
            ``epoch``, ``t``, ``phase``, ``coverage`` (``ground_band``,
            ``total``), ``drones`` (per drone ``id``, interpolated ``pos``,
            ``yaw``, ``vel``, ``alive``, ``landed``, ``task``, ``goal``),
            ``frontiers`` (target centroids), ``revealed`` (base64 int32
            global triangle ids newly seen since the previous call) and
            ``detections`` (per classified object its track ``id``, ``cls``
            name, outline ``color`` ``[r, g, b]`` in ``[0, 1]``, box
            ``center``, ``size`` (length, width, height), ``yaw`` and
            ``confidence``) and ``site``: ``None`` while the swarm is still
            mapping, then ``sites`` (best first, per site its ``rank``,
            ``pos`` on the wall at ground level, ``yaw`` of the wall's
            outward normal, ``meter`` position, ``cost``, per-rule
            ``breakdown``, ``warnings`` (empty unless it breaks a rule) and
            outline ``color`` ``[r, g, b]`` in ``[0, 1]``), ``battery``
            (``[width, depth, height]``, or ``None``) and ``message`` (why
            there are no sites, else ``None``).

        Raises
        ------
        SimulationError
            If ``wall_dt_s`` is negative or not finite.
        """
        if not math.isfinite(wall_dt_s) or wall_dt_s < 0.0:
            msg = f"wall_dt_s must be a finite, non-negative number, got {wall_dt_s}"
            raise SimulationError(msg)
        step = min(wall_dt_s, _MAX_FRAME_S) * self._time_scale
        if self._worker is None:
            with self._lock:
                target = self._play_t + step
            # One tick past the target, so there is a later tick to interpolate
            # toward and the fraction of a tick carries to the next call.
            while self._latest_t() <= target + _T_EPS:
                self._produce_one()
        with self._lock:
            if self._worker is not None:
                lead = _LEAD_S * self._time_scale
                ahead = self._buffer[-1].t - self._play_t if self._buffer else 0.0
                step *= min(1.0, ahead / (_FULL_SPEED_FRACTION * lead))
            self._play_t += step
            return self._consume()

    # -- internals ----------------------------------------------------------
    def _checked_count(self, n: int) -> int:
        """Return ``n`` if it is a valid swarm size, else raise."""
        limit = self._cfg.viewer.max_drones
        if not MIN_DRONES <= n <= limit:
            msg = f"drone count must be in [{MIN_DRONES}, {limit}], got {n}"
            raise ConfigError(msg)
        return n

    def _load_scene(self) -> None:
        """Generate the property, load its geometry and build its sensor for ``self._seed``.

        Cached: worldgen and the Open3D BVH build cost roughly a second, and a
        drone-count change must not pay it again.
        """
        out_dir = self._scene_dir / str(self._seed)
        manifest = generate_field(self._seed, self._cfg, out_dir=out_dir)
        geometry = load_geometry(manifest)
        sensor = RaySensor(geometry, self._cfg.sensor, np.random.default_rng(self._seed))
        self._manifest = manifest
        self._geometry = geometry
        self._sensor = sensor
        self._world_payload = _world_payload(self._seed, manifest, geometry)

    def _reset(self) -> None:
        """Rebuild the mission and put every drone back on its pad."""
        self._run = MissionRun(
            self._manifest,
            self._geometry,
            self._n,
            self._cfg,
            seed=self._seed,
            sensor=self._sensor,
        )
        self._site = None
        self._shown = _snapshot(self._run)
        self._buffer.clear()
        # The launch scan's reveals go out with the first frame.
        self._unsent = [self._shown.revealed]
        self._play_t = 0.0
        self._epoch += 1
        _log.debug("viewer reset: %d drone(s), seed=%d, epoch=%d", self._n, self._seed, self._epoch)

    def _scene(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "seed": self._seed,
            "drones": self._n,
            "max_drones": self._cfg.viewer.max_drones,
            "time_scale": self._time_scale,
            "min_time_scale": MIN_TIME_SCALE,
            "max_time_scale": MAX_TIME_SCALE,
            "pads": self._run.pads.tolist(),
            "lot": self._manifest.lot_bounds.tolist(),
        }

    def _latest_t(self) -> float:
        """Time of the newest recorded tick."""
        with self._lock:
            return self._buffer[-1].t if self._buffer else self._shown.t

    def _produce_one(self) -> None:
        """Run one control tick and record it."""
        with self._sim_lock:
            self._run.tick()
            if self._site is None and self._run.phase not in _MAPPING_PHASES:
                self._site = self._suggest_sites()
            snap = _snapshot(self._run, self._site)
            # Taken while still holding the sim lock, so a reset cannot slip in
            # between the tick and the append and leave a stale snapshot behind.
            with self._lock:
                self._buffer.append(snap)

    def _produce_loop(self) -> None:
        """Keep the buffer ``_LEAD_S`` of wall time ahead of playback (the simulation thread)."""
        while not self._stop.is_set():
            with self._lock:
                newest = self._buffer[-1].t if self._buffer else self._shown.t
                behind = newest < self._play_t + _LEAD_S * self._time_scale
            if behind:
                self._produce_one()
            else:
                self._stop.wait(_IDLE_WAIT_S)

    def _consume(self) -> dict[str, Any]:
        """Advance the tick on screen to the playback clock and build the frame.

        Caller holds ``_lock``.
        """
        while self._buffer and self._buffer[0].t <= self._play_t + _T_EPS:
            self._shown = self._buffer.popleft()
            self._unsent.append(self._shown.revealed)
        if self._buffer:
            nxt = self._buffer[0]
            alpha = (self._play_t - self._shown.t) / (nxt.t - self._shown.t)
        else:
            # Out of recorded ticks: hold on the last one rather than run the
            # clock on into time nothing has simulated yet.
            nxt = self._shown
            alpha = 0.0
            self._play_t = self._shown.t
        # Each tick's ids are sorted; several ticks' worth must be merged to stay so.
        revealed = np.unique(np.concatenate([np.empty(0, np.int64), *self._unsent]))
        self._unsent = []
        return self._frame(self._shown, nxt, alpha, revealed)

    def _frame(
        self, prev: _Snapshot, cur: _Snapshot, alpha: float, revealed: npt.NDArray[np.int64]
    ) -> dict[str, Any]:
        prev_pos = np.stack([s.pos for s in prev.states])
        cur_pos = np.stack([s.pos for s in cur.states])
        pos = prev_pos + alpha * (cur_pos - prev_pos)
        prev_yaw = np.array([s.yaw for s in prev.states])
        cur_yaw = np.array([s.yaw for s in cur.states])
        # Interpolate the short way round, or a drone crossing +/-pi spins a lap.
        yaw_step = np.angle(np.exp(1j * (cur_yaw - prev_yaw)))
        yaw = prev_yaw + alpha * yaw_step

        drones = []
        for row, state in enumerate(prev.states):
            task = prev.tasks.get(state.drone_id)
            drones.append(
                {
                    "id": state.drone_id,
                    "pos": pos[row].tolist(),
                    "yaw": mathutil.wrap_to_pi(float(yaw[row])),
                    "vel": state.vel.tolist(),
                    "alive": state.alive,
                    "landed": state.drone_id in prev.landed,
                    "task": task[0] if task is not None else "idle",
                    "goal": task[1].tolist() if task is not None else None,
                }
            )

        return {
            "epoch": self._epoch,
            "t": prev.t,
            "phase": prev.phase,
            "coverage": {"ground_band": prev.ground_band, "total": prev.total},
            "drones": drones,
            "frontiers": prev.frontiers.tolist(),
            "revealed": _b64(revealed.astype(np.int32)),
            "detections": [self._detection(obj) for obj in prev.detections],
            "site": prev.site,
        }

    def _suggest_sites(self) -> dict[str, Any]:
        """Site the battery on the finished map, as the page draws the result.

        Caller holds ``_sim_lock``. A map with nothing to site against -- no
        meter, no wall beside it -- is an outcome to show, not an error to
        raise, so it comes back as an empty list with the reason.
        """
        try:
            sites = suggest_sites(self._run.mapper.state, self._site_rules)
        except SiteError as exc:
            _log.info("no battery site suggested: %s", exc)
            return {"sites": [], "message": str(exc), "battery": None}
        _log.info("suggested %d battery site(s) at t=%.1f s", len(sites), self._run.t)
        battery = self._site_rules.battery
        return {
            "sites": [self._site_entry(rank, site) for rank, site in enumerate(sites, start=1)],
            "message": None,
            "battery": [battery.width_m, battery.depth_m, battery.height_m],
        }

    def _site_entry(self, rank: int, site: SiteCandidate) -> dict[str, Any]:
        """One suggested site as the page draws it."""
        viewer = self._cfg.viewer
        rgb = viewer.site_warning_rgb if site.warnings else viewer.site_rgb
        return {
            "rank": rank,
            "pos": site.pos.tolist(),
            "yaw": math.atan2(float(site.wall_normal[1]), float(site.wall_normal[0])),
            "meter": site.conduit[0].tolist(),
            "cost": site.cost,
            # JSON has no infinity: a rule with nothing to measure against reports none.
            "breakdown": {k: v if math.isfinite(v) else None for k, v in site.breakdown.items()},
            "warnings": list(site.warnings),
            "color": [c / 255.0 for c in rgb],
        }

    def _detection(self, obj: DiscoveredObject) -> dict[str, Any]:
        """One classified object as the page draws it."""
        name = obj.cls.name
        return {
            "id": obj.track_id,
            "cls": name,
            "color": self._box_colors.get(name, self._box_default),
            "center": obj.box.center.tolist(),
            "size": obj.box.size.tolist(),
            "yaw": obj.box.yaw,
            "confidence": obj.confidence,
        }


def _world_payload(seed: int, manifest: SceneManifest, geometry: SceneGeometry) -> dict[str, Any]:
    """Build the base64 mesh payload :meth:`ViewerSession.world` returns.

    Built once per seed rather than per call: encoding a quarter-million
    triangles is a few milliseconds, but there is no reason to redo it every
    time the page asks.
    """
    objects = []
    for obj in manifest.objects:
        vertices = geometry.vertices[obj.obj_id]
        faces = geometry.faces[obj.obj_id]
        objects.append(
            {
                "id": obj.obj_id,
                "cls": obj.cls.name,
                "color": [c / 255.0 for c in obj.color],
                "background": obj.background,
                "tri_offset": int(geometry.obj_tri_offset[obj.obj_id]),
                "positions": _b64(vertices.astype(np.float32)),
                "indices": _b64(faces.astype(np.uint32).ravel()),
            }
        )
    return {"seed": seed, "objects": objects}


def _fresh_seed() -> int:
    return secrets.randbits(_SEED_BITS)


@functools.cache
def drone_model(path: Path | None = None) -> dict[str, Any]:
    """Load the drone mesh the page draws, in the body frame.

    The body frame is the one :class:`~canopy.contracts.DroneState` yaw is
    measured in: +X forward at yaw 0, +Z up, origin at the landing-gear contact
    point, metres. The authored scout faces its native +Z, which
    :func:`~canopy.worldgen.objio.read_obj` turns into world -Y, so it is turned
    a further +90 degrees about Z here to put the camera on +X.

    Cached: the file does not change under a running viewer, and the page asks
    once per load.

    Parameters
    ----------
    path
        OBJ to read. Defaults to :data:`DRONE_MODEL` under the asset root. The
        MTL is the file of the same stem beside it.

    Returns
    -------
    dict
        ``vertices``: flat ``[x, y, z, ...]`` in the body frame. ``groups``: per
        material its ``material`` name, ``color`` ``[r, g, b]`` (sRGB, from
        ``Kd``), ``opacity`` and flat triangle ``indices`` into ``vertices``.

    Raises
    ------
    AssetError
        If the OBJ or MTL is missing or malformed.
    """
    obj_path = default_assets_dir().parent / DRONE_MODEL if path is None else path
    return _front_on_x(obj_path)


@functools.cache
def battery_model(path: Path | None = None) -> dict[str, Any]:
    """Load the battery mesh the page draws at each suggested site, in the site frame.

    The site frame is the one a :class:`~canopy.contracts.SiteCandidate`
    implies: +X out of the wall along its normal, +Z up, origin on the wall at
    ground level. The authored battery faces its native +Z with its back on
    the wall, so the drone's +90 degree turn puts its front on +X too. It is
    then slid along the wall until its bounding box -- which takes in a
    side-mounted disconnect -- is centred on the site, because that box is
    what ``rules.yaml``'s ``battery`` describes and every clearance was
    measured from.

    Parameters
    ----------
    path
        OBJ to read. Defaults to :data:`BATTERY_MODEL` under the asset root.

    Returns
    -------
    dict
        As :func:`drone_model`.

    Raises
    ------
    AssetError
        If the OBJ or MTL is missing or malformed.
    """
    obj_path = default_assets_dir().parent / BATTERY_MODEL if path is None else path
    return _front_on_x(obj_path, centre_across=True)


def _front_on_x(obj_path: Path, *, centre_across: bool = False) -> dict[str, Any]:
    """Read an authored model facing native +Z and turn it to face +X.

    With ``centre_across``, also centre its bounding box on ``y = 0``.
    """
    objects = read_obj(obj_path)
    materials = read_mtl(obj_path.with_suffix(".mtl"))

    # Every group of every object shares one vertex array, so a multi-object
    # file still reaches the page as a single mesh.
    verts: list[Points] = []
    groups: list[dict[str, Any]] = []
    offset = 0
    for obj in objects.values():
        verts.append(obj.vertices)
        for group in obj.groups:
            mat = materials.get(group.material)
            groups.append(
                {
                    "material": group.material,
                    "color": list(mat.diffuse) if mat else [0.5, 0.5, 0.5],
                    "opacity": mat.opacity if mat else 1.0,
                    "indices": (group.faces + offset).ravel().tolist(),
                }
            )
        offset += len(obj.vertices)

    world = np.concatenate(verts, axis=0)
    # +90 degrees about Z: (x, y) -> (-y, x), taking the authored front (-Y) to +X.
    body = np.stack([-world[:, 1], world[:, 0], world[:, 2]], axis=1)
    if centre_across:
        body[:, 1] -= (body[:, 1].min() + body[:, 1].max()) / 2.0
    return {"vertices": body.ravel().tolist(), "groups": groups}


class _Bridge:
    """What the page may call, as ``window.pywebview.api.<name>``.

    pywebview exposes every public attribute of this object to JavaScript, so it
    is kept deliberately thin and the session itself is never handed over.
    """

    def __init__(self, session: ViewerSession) -> None:
        self._session = session

    def scene(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.scene`."""
        return self._session.scene()

    def world(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.world`."""
        return self._session.world()

    def frame(self, wall_dt_s: float) -> dict[str, Any]:
        """See :meth:`ViewerSession.advance`."""
        return self._session.advance(float(wall_dt_s))

    def set_drone_count(self, n: int) -> dict[str, Any]:
        """See :meth:`ViewerSession.set_drone_count`."""
        return self._session.set_drone_count(int(n))

    def set_time_scale(self, scale: float) -> dict[str, Any]:
        """See :meth:`ViewerSession.set_time_scale`."""
        return self._session.set_time_scale(float(scale))

    def restart(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.restart`."""
        return self._session.restart()

    def new_scene(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.new_scene`."""
        return self._session.new_scene()

    def drone_model(self) -> dict[str, Any]:
        """See :func:`drone_model`."""
        return drone_model()

    def battery_model(self) -> dict[str, Any]:
        """See :func:`battery_model`."""
        return battery_model()


def launch(session: ViewerSession, *, debug: bool = False) -> None:
    """Open the Canopy window on ``session`` and block until it is closed.

    Parameters
    ----------
    session
        The mission to render. Its simulation thread is started here and
        stopped when the window closes.
    debug
        Enable the web inspector (right-click, Inspect).

    Raises
    ------
    DependencyMissingError
        If the ``viewer`` dependency group (pywebview) is not installed.
    """
    try:
        import webview  # noqa: PLC0415 - optional dependency, see module docstring
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "canopy-view needs the `viewer` dependency group: run `uv sync --group viewer`"
        raise DependencyMissingError(msg) from exc

    viewer = session.config.viewer
    session.start()
    webview.create_window(
        "Canopy",
        # A plain path, not a file:// URI: pywebview serves local paths over its
        # built-in HTTP server, and the page's ES-module imports need one.
        url=str(WEB_DIR / "index.html"),
        js_api=_Bridge(session),
        width=viewer.width_px,
        height=viewer.height_px,
        min_size=(720, 480),
        maximized=True,
        background_color="#EEF1F4",
    )
    try:
        webview.start(debug=debug)
    finally:
        session.close()
