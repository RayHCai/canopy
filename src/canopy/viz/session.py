"""The viewer's headless half: a real mapping mission, advanced on demand.

``canopy-view`` draws a real mapping mission -- the same
:class:`~canopy.planning.MissionRun` loop ``canopy-fly`` and the batch
evaluator drive. :class:`ViewerSession` is pure Python: it owns the generated
property, the mission run and simulated time. The page drives it through
:mod:`canopy.viz.bridge`, but so can a test, with no display attached. The
address-build workflow it exposes lives in :mod:`canopy.viz.site_jobs`, and the
per-tick snapshots and run log in :mod:`canopy.viz.recording`.

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

Every frame also lists the objects perception has classified so far, each as
an upright box the page outlines in its class's colour
(``viewer.detection_colors``). They come from the mapper's
:attr:`~canopy.contracts.MapState.discovered`, which the detector builds from
range and colour alone, so the outlines show what the swarm worked out, not
what the scene manifest says.

Once mapping is complete -- the mission has left exploration -- the session
asks the site stage to judge the finished map against the checklist
(:func:`~canopy.site.assess_site`, rules from ``config/rules.yaml``) and every
later frame carries the result: an overall verdict, a short justification, and
the offered sites, each with its own verdict. They are computed on the
simulation thread at the tick exploration ends and ride along in that tick's
snapshot, so they appear on screen when the end of mapping does, not seconds
before it. The page draws each site with the ``base_core_battery`` model
(:mod:`canopy.viz.models`), served like the drone.

Each generated property is written under ``scene_dir/<seed>`` (``out/viewer``
by default). Every re-roll and address build writes a new one, so the
directory is trimmed to the ``viewer.scene_cache_max`` most recently used
seeds whenever a scene is built; the loaded scene's own directory is never
removed, and address snapshots (``out/sites``) live elsewhere and are never
touched.
"""

from __future__ import annotations

import base64
import functools
import math
import secrets
import shutil
import threading
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from canopy import mathutil
from canopy.contracts import (
    DiscoveredObject,
    DroneState,
    SceneGeometry,
    SceneManifest,
    SiteCandidate,
    SiteSnapshot,
    SiteVerdict,
)
from canopy.errors import ConfigError, SimulationError, SiteError, WorldgenError
from canopy.log import get_logger
from canopy.planning import MappingEta, MissionRun
from canopy.sim import RaySensor, load_geometry
from canopy.site import SiteRules, assess_site, load_site_rules
from canopy.viz.recording import T_EPS, RunRecorder, Snapshot, take_snapshot
from canopy.viz.site_jobs import JobGate, SiteJobs, location_doc
from canopy.worldgen import generate_field

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["MAX_TIME_SCALE", "MIN_DRONES", "MIN_TIME_SCALE", "ViewerSession"]

_log = get_logger(__name__)

#: A swarm of zero is not a scene worth rendering.
MIN_DRONES = 1

#: Bounds of the time-speed slider: simulated seconds per wall-clock second.
#: The top end is what the simulation thread can sustain: a 3-drone mission
#: runs at roughly 2.5-3x real time, so a higher setting would only be
#: capped by the playback throttle and look like the slider does nothing.
MIN_TIME_SCALE = 0.5
MAX_TIME_SCALE = 2.0

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

#: Seeds are drawn from this many bits, so they stay readable in the UI.
_SEED_BITS = 31

#: Where generated scenes live by default, one subdirectory per seed.
_DEFAULT_SCENE_DIR = Path("out") / "viewer"


def _b64(array: npt.NDArray[Any]) -> str:
    """Base64-encode an array's raw bytes, little-endian, for the JS bridge."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``, for ``ViewerSession.run_record``.

    pywebview's own bridge (every other payload in this module) round-trips
    NaN/Infinity as the non-standard JSON its embedded browser also accepts;
    ``run_record`` is the one payload that leaves Python for another process
    (ADR 0017), over a boundary that has no reason to share that leniency.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True, slots=True, eq=False)
class _Scene:
    """One generated property, ready to fly: everything a seed (and site) builds.

    Built whole, outside ``_lock``, then swapped in with a single assignment,
    so a build that fails -- an address no wall can take a meter on -- leaves
    the session exactly on the scene it was showing.
    """

    seed: int
    #: The pinned address, if any; ``None`` is random mode.
    site: SiteSnapshot | None
    manifest: SceneManifest
    geometry: SceneGeometry
    sensor: RaySensor
    #: What :meth:`ViewerSession.world` returns; see :func:`_world_payload`.
    world_payload: dict[str, Any]


def _evict_scenes(scene_dir: Path, keep: Collection[Path], limit: int) -> None:
    """Trim ``scene_dir`` to its ``limit`` most recently used seed directories.

    Only directories named like a seed (all digits) are candidates, so
    anything else a user put there survives; directories in ``keep`` are
    never removed and count toward ``limit``. Recency is the directory's
    mtime, which :meth:`ViewerSession._load_scene` refreshes on every build.
    A directory that cannot be removed -- a mesh still open elsewhere on
    Windows, say -- is logged and left for the next pass: this is a cache,
    and failing to trim it must not fail the scene that triggered the trim.
    For the same reason a directory that vanishes mid-scan (another viewer
    process trimming the same cache, a user deleting it) is simply skipped,
    and a scan that fails outright skips this pass.
    """
    try:
        seed_dirs = [p for p in scene_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    except OSError as exc:
        _log.warning("could not scan scene cache %s: %s", scene_dir, exc)
        return
    kept = [p for p in seed_dirs if p in keep]
    others: list[tuple[float, Path]] = []
    for p in seed_dirs:
        if p in keep:
            continue
        try:
            others.append((p.stat().st_mtime, p))
        except OSError:
            continue  # gone since the listing: nothing left to evict
    others.sort(key=lambda entry: entry[0], reverse=True)
    for _, stale in others[max(0, limit - len(kept)) :]:
        try:
            shutil.rmtree(stale)
        except OSError as exc:
            _log.warning("could not evict cached scene %s: %s", stale, exc)
        else:
            _log.debug("evicted cached scene %s", stale)


class ViewerSession:
    """A swarm mapping one generated property, advanced on demand.

    Generating the property and building its raycasting scene costs roughly a
    second, so both are cached per seed and only rebuilt when the seed changes;
    a drone-count change reuses them and only rebuilds the mission itself.

    Three locks. ``_sim_lock`` guards the mission and the loaded scene and is
    held for a whole tick, or a whole scene build; ``_lock`` guards the
    snapshot buffer, playback clock and the scene/mission the page reads, and
    is only ever held briefly; ``_job_lock``, owned by the
    :class:`~canopy.viz.site_jobs.SiteJobs` collaborator, guards a site
    build's bookkeeping the same way. Taken together it is always in this
    order -- ``_sim_lock``, then ``_lock``, then ``_job_lock`` -- and only a
    site install (:meth:`_install_site`) ever needs all three.

    pywebview delivers page calls on worker threads, so a frame must never
    wait on a lock held across something slow. That ruled out holding
    ``_sim_lock``/``_lock`` across a replan, which is why the simulation runs
    on its own thread (see the module docstring); it equally rules out holding
    ``_lock`` or ``_job_lock`` across a network fetch, a call to
    :func:`~canopy.worldgen.generate_field` or building a
    :class:`~canopy.planning.MissionRun`. So every rebuild takes the same
    shape: build the new scene and mission under ``_sim_lock`` alone -- the
    simulation thread pauses, playback keeps drawing from the buffer -- then
    take ``_lock`` just to swap them in. A site build's own worker threads
    only ever take ``_job_lock`` to check or record progress, never around the
    slow call itself.

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
    site
        Real-world data to build the property from, in place of a bare seed.
        ``None`` (the default) is random mode; :meth:`new_scene` then re-rolls
        everything. Given a snapshot, the property is rebuilt from it and
        :meth:`new_scene` re-rolls only what the snapshot could not see.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        drones: int | None = None,
        seed: int | None = None,
        scene_dir: Path | None = None,
        site_rules: SiteRules | None = None,
        site: SiteSnapshot | None = None,
    ) -> None:
        self._cfg = cfg
        self._site_rules = load_site_rules() if site_rules is None else site_rules
        # Suggested battery sites for the current mission, once mapping is
        # complete. Unrelated to the scene's `site` -- this is the finished
        # mission's output, not the property's own address data -- but the two
        # names collided before this feature existed, so this one keeps the
        # name the frame's "site" JSON key and Snapshot.site already commit to.
        self._battery_site: dict[str, Any] | None = None
        self._sim_lock = threading.Lock()
        self._lock = threading.Lock()
        self._site_jobs = SiteJobs(cfg, self._install_site)
        self._scene_dir = _DEFAULT_SCENE_DIR if scene_dir is None else Path(scene_dir)
        self._n = self._checked_count(cfg.viewer.drones if drones is None else drones)
        # Bumped on every reset so the page can drop frames that were already in
        # flight when the mission changed underneath them.
        self._epoch = -1
        # Declared, not assigned: set for real by _commit() below, which is the
        # only place any of them is ever assigned before a public method could
        # observe them.
        self._loaded: _Scene
        self._run: MissionRun
        self._eta: MappingEta
        # The whole simulated mission recorded so far (ADR 0017), replaced by
        # _commit() on every reset; see run_record().
        self._recorder: RunRecorder
        # The tick on screen (at or before the playback clock), the recorded
        # ticks after it, and revealed ids passed but not yet sent to the page.
        self._shown: Snapshot
        self._buffer: deque[Snapshot] = deque()
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
        scene = self._load_scene(_fresh_seed() if seed is None else seed, site, current=None)
        # No locks yet: nothing else can hold a reference to this session.
        self._commit(scene, self._n, self._new_run(scene, self._n))

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
        return self._loaded.seed

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
        with self._sim_lock:
            n = self._checked_count(n)
            run = self._new_run(self._loaded, n)
            with self._lock:
                self._commit(self._loaded, n, run)
                return self._scene()

    def restart(self) -> dict[str, Any]:
        """Rewind to simulated time zero: same property, same swarm, fresh mission.

        Everything the swarm learned (map, detections, sites) goes with the old
        mission; the property and the time scale stay.
        """
        with self._sim_lock:
            run = self._new_run(self._loaded, self._n)
            with self._lock:
                self._commit(self._loaded, self._n, run)
                return self._scene()

    def new_scene(self, seed: int | None = None) -> dict[str, Any]:
        """Start over with a new seed (random unless given), regenerating the property.

        In address mode this re-rolls only what the pinned snapshot left to
        the generator (the meter, openings, bushes, ...): :meth:`_load_scene`
        always rebuilds from whatever the loaded scene pins, so there is no
        special case here for either mode.
        """
        with self._sim_lock:
            new_seed = _fresh_seed() if seed is None else seed
            scene = self._load_scene(new_seed, self._loaded.site, current=self._loaded)
            run = self._new_run(scene, self._n)
            with self._lock:
                self._commit(scene, self._n, run)
                return self._scene()

    def use_random_location(self) -> dict[str, Any]:
        """Clear any pinned address and load a fresh random scene.

        A site build already in flight is superseded exactly as a newer one
        would be (see :meth:`start_site_build`): forgetting the live job means
        its eventual result, however it turns out, matches nothing by the time
        it arrives and is dropped.
        """
        self._site_jobs.supersede()
        with self._sim_lock:
            scene = self._load_scene(_fresh_seed(), None, current=self._loaded)
            run = self._new_run(scene, self._n)
            with self._lock:
                self._commit(scene, self._n, run)
                return self._scene()

    def suggest_addresses(self, text: str) -> dict[str, Any]:
        """Address suggestions for the combobox, run outside every session lock.

        pywebview runs each bridge call on its own worker thread, so a slow
        geocoder only delays this call; it never competes with a frame for
        ``_sim_lock``/``_lock``, which this never touches.

        Returns
        -------
        dict
            ``suggestions``: a list of ``{label, provider, ref, lat_deg,
            lon_deg, attribution}``, best first. ``error``: the geocoder's message, or
            ``None`` -- never raised to the page, which has no use for a
            Python traceback.
        """
        return self._site_jobs.suggest(text)

    def start_site_build(self, suggestion: dict[str, Any], query: str) -> dict[str, Any]:
        """Fetch ``suggestion`` and, once accepted, build it into the scene.

        Runs on a daemon worker thread, outside every session lock, exactly
        like :meth:`suggest_addresses`. Starting a build while one is already
        running supersedes it -- see :mod:`canopy.viz.site_jobs` on supersession.

        Parameters
        ----------
        suggestion
            One entry of :meth:`suggest_addresses`' ``suggestions``.
        query
            The text that was being searched when it was picked, passed on to
            :func:`~canopy.worldgen.fetch_site` for its own notes.

        Returns
        -------
        dict
            ``{"job_id": int}``. Poll progress with :meth:`site_job`.
        """
        return self._site_jobs.start(suggestion, query)

    def confirm_site_build(self, job_id: int) -> dict[str, Any]:
        """Move a ``needs_confirmation`` job on to ``building``, on a fresh worker thread.

        A no-op that just reports the current status, if ``job_id`` is not a
        job actually awaiting confirmation -- it was already resolved,
        cancelled, or superseded by a later one.

        Raises
        ------
        SimulationError
            If the job reached ``needs_confirmation`` with no fetched snapshot
            to build from, which the fetch step never should have allowed --
            this can only mean a bug in :mod:`canopy.viz.site_jobs`, not bad
            input.
        """
        return self._site_jobs.confirm(job_id)

    def cancel_site_build(self, job_id: int) -> dict[str, Any]:
        """Mark ``job_id`` cancelled, if it is still the live, unfinished job.

        Its worker thread, if one is running, is not interrupted -- a network
        fetch has no cancellation hook -- but the state flip is enough: every
        step it takes before touching the scene checks this first and drops
        its result if it finds ``cancelled``, the same check a superseding job
        would trip.
        """
        return self._site_jobs.cancel(job_id)

    def site_job(self, job_id: int) -> dict[str, Any]:
        """Report progress of a build started by :meth:`start_site_build`.

        Returns
        -------
        dict
            ``job_id``, ``state`` (``running``, ``needs_confirmation``,
            ``done``, ``failed`` or ``cancelled`` -- the last also standing in
            for "not the live job", e.g. one superseded by a later call),
            ``stage`` (``fetching`` or ``building``), ``error`` (the failure
            message, else ``None``), ``verdict`` (as
            :class:`~canopy.contracts.ResidentialVerdict`, once fetched, else
            ``None``) and ``location`` (as the ``location`` key of
            :meth:`scene`, built from the fetched snapshot alone since
            generation may not have run yet; ``None`` before the fetch
            completes).
        """
        return self._site_jobs.status(job_id)

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
            ``[x, y, z]`` per drone), ``lot`` (``[[xmin, ymin, zmin],
            [xmax, ymax, zmax]]``, world frame, metres, Z-up) and
            ``location``: ``mode`` (``"random"`` or ``"address"``), ``label``,
            ``site_id``, ``verdict`` (``p_residential``, ``decision``,
            ``reasons``, or ``None`` in random mode), ``observed`` and
            ``inferred`` (short strings on what the data showed versus what
            the generator had to invent), ``notes`` (the manifest's) and
            ``attribution`` (the snapshot's distinct source attributions).
        """
        with self._lock:
            return self._scene()

    def world(self) -> dict[str, Any]:
        """Describe the generated property's mesh, cached per seed.

        Returns
        -------
        dict
            ``seed`` and ``objects``: per object its ``id``, ``cls`` (class
            name), ``color`` (``[r, g, b]`` in ``[0, 1]``, the flat class
            colour, still what the reveal falls back to and what a photo
            shades), ``background``, ``tri_offset`` (its first global triangle
            id), base64 ``positions`` (little-endian float32 ``xyz``, world
            Z-up) / ``indices`` (little-endian uint32) into them, and base64
            ``colors`` (uint8 ``rgb`` per face, sRGB 0..255, in face order) --
            present only when the object carries authored material runs,
            i.e. :attr:`~canopy.contracts.SceneObject.materials` is non-empty.
        """
        with self._lock:
            return self._loaded.world_payload

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
            there are no sites, else ``None``), and ``mapping``:
            ``remaining_s`` (estimated simulated seconds of exploration left,
            :class:`~canopy.planning.MappingEta`; ``None`` until the frontier
            count has a trend to project, ``0.0`` once mapping has ended) and
            ``done_at_s`` (simulated time mapping ended, else ``None``). Wall
            time left is the page's to derive, from :attr:`time_scale`.

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
            while self._latest_t() <= target + T_EPS:
                self._produce_one()
        with self._lock:
            if self._worker is not None:
                lead = _LEAD_S * self._time_scale
                ahead = self._buffer[-1].t - self._play_t if self._buffer else 0.0
                step *= min(1.0, ahead / (_FULL_SPEED_FRACTION * lead))
            self._play_t += step
            return self._consume()

    def run_record(self) -> dict[str, Any]:
        """Describe the whole simulated mission so far, for the dashboard upload (ADR 0017).

        Unlike :meth:`scene` and :meth:`advance`, this reflects every tick the
        simulation thread has produced, not the tick :attr:`sim_time_s`
        reports: the module docstring's ``_LEAD_S`` buffer means playback is
        always a little behind, and the page uploads once it catches up to the
        tick that reached ``done``, so the record must already hold everything
        by then. Every float is finite -- see :func:`_json_safe` -- so the
        result round-trips through ``json.dumps(..., allow_nan=False)``.

        Returns
        -------
        dict
            ``seed``, ``drones``, ``sim_time_s`` (latest recorded tick),
            ``phase`` (the live mission's, ahead of whatever :meth:`advance`
            last returned), ``mapped_at_s`` (simulated time mapping ended,
            else ``None``), ``coverage`` (``ground_band``, ``total``), ``lot``
            and ``location`` (as :meth:`scene`), ``tracks`` (per drone ``id``,
            ``samples`` -- ``[t, x, y, z, yaw, battery]`` rows roughly
            half a second apart, plus the first tick -- current ``alive`` and
            ``battery``), ``events`` (as recorded by
            :class:`~canopy.viz.recording.RunRecorder`: ``t``, ``drone_id``,
            ``type`` (``capture``,
            ``low_battery``, ``failure`` or ``complete``), ``message``,
            ``status`` (``Flying``, ``Idle`` or ``Failed``), ``battery_pct``
            and ``task``), ``detections`` (as :meth:`advance`'s, for every
            object the mapper has discovered so far) and ``site`` (as
            :meth:`advance`'s ``site.sites`` entry shape, or ``None`` before
            mapping ends).
        """
        # All under _sim_lock alone, in one hold, so the tracks, the mission
        # and the scene they describe all come from the same reset: _commit
        # writes _loaded/_n/_run/_recorder only under _sim_lock (and _lock),
        # and _battery_site is written under _sim_lock alone (_produce_one).
        # _lock is not needed and not taken -- nothing here is playback state.
        with self._sim_lock:
            scene = self._loaded
            record = {
                "seed": scene.seed,
                "drones": self._n,
                "sim_time_s": self._run.t,
                "phase": str(self._run.phase),
                "mapped_at_s": self._eta.done_at_s,
                "coverage": {
                    "ground_band": self._run.coverage_ground_band,
                    "total": self._run.coverage_total,
                },
                "lot": scene.manifest.lot_bounds.tolist(),
                "location": location_doc(scene.site, scene.manifest.notes, scene.manifest),
                "tracks": self._recorder.tracks(),
                "events": self._recorder.events(),
                "detections": [
                    self._detection(obj) for obj in self._run.mapper.state.discovered.values()
                ],
                "site": self._battery_site,
            }
        # _json_safe is necessarily Any-typed (it recurses into whatever it is
        # given); the cast just tells mypy what every caller already knows,
        # that sanitising a dict[str, Any] still leaves a dict[str, Any].
        return cast("dict[str, Any]", _json_safe(record))

    def intake(self) -> dict[str, Any]:
        """Where the review API is, and today's swarm-size bounds (ADR 0017's intake step).

        Returns
        -------
        dict
            ``api_url`` (``viewer.review_api_url``), ``drones`` (current swarm
            size), ``max_drones`` and ``min_drones``.
        """
        return {
            "api_url": self._cfg.viewer.review_api_url,
            "drones": self._n,
            "max_drones": self._cfg.viewer.max_drones,
            "min_drones": MIN_DRONES,
        }

    # -- internals ----------------------------------------------------------
    def _checked_count(self, n: int) -> int:
        """Return ``n`` if it is a valid swarm size, else raise."""
        limit = self._cfg.viewer.max_drones
        if not MIN_DRONES <= n <= limit:
            msg = f"drone count must be in [{MIN_DRONES}, {limit}], got {n}"
            raise ConfigError(msg)
        return n

    def _load_scene(
        self, seed: int, site: SiteSnapshot | None, *, current: _Scene | None
    ) -> _Scene:
        """Generate the property, load its geometry and build its sensor for ``seed``.

        Slow -- worldgen and the Open3D BVH build cost roughly a second -- so
        the caller holds ``_sim_lock`` (serialising builds, and the cache
        trim below, against each other) but never ``_lock``. The result is
        built whole and assigned nothing, so the caller commits it or, if this
        raises, the session is untouched. Kept per seed: a drone-count change
        reuses the loaded scene and never comes here. Rebuilds from ``site``
        when one is pinned, so a re-roll in address mode still only touches
        what the snapshot left to the generator.

        ``current`` is the scene still loaded, if any, whose directory the
        cache trim must spare along with the new one.

        Raises
        ------
        WorldgenError
            If the property cannot be generated, e.g. an address with no wall
            that takes a meter.
        """
        out_dir = self._scene_dir / str(seed)
        manifest = generate_field(seed, self._cfg, out_dir=out_dir, site=site)
        geometry = load_geometry(manifest)
        sensor = RaySensor(geometry, self._cfg.sensor, np.random.default_rng(seed))
        scene = _Scene(
            seed=seed,
            site=site,
            manifest=manifest,
            geometry=geometry,
            sensor=sensor,
            world_payload=_world_payload(seed, manifest, geometry),
        )
        # generate_field rewrites the files in place, which leaves the
        # directory's own mtime alone; touch it so recency means "last built".
        out_dir.touch()
        keep = {out_dir}
        if current is not None:
            keep.add(self._scene_dir / str(current.seed))
        _evict_scenes(self._scene_dir, keep, self._cfg.viewer.scene_cache_max)
        return scene

    def _new_run(self, scene: _Scene, n: int) -> MissionRun:
        """Build a fresh mission of ``n`` drones over ``scene``. Caller holds ``_sim_lock`` only.

        Not free -- pads, the survey envelope and the launch scan -- which is
        why it is built before ``_lock`` is taken, not under it.
        """
        return MissionRun(
            scene.manifest, scene.geometry, n, self._cfg, seed=scene.seed, sensor=scene.sensor
        )

    def _install_site(self, snapshot: SiteSnapshot, gate: JobGate) -> bool:
        """Build ``snapshot`` into a fresh scene and swap it in (a site-job worker thread).

        The ``install`` callback :class:`~canopy.viz.site_jobs.SiteJobs` was
        given. Takes ``_sim_lock`` for the build, then ``_lock`` for the swap,
        and only inside that has ``gate`` take ``_job_lock`` -- the documented
        order, never reversed. The swap itself runs inside
        :meth:`~canopy.viz.site_jobs.JobGate.commit_if_live`, so the liveness
        check, the commit and the job turning ``done`` are one atomic step.
        Returns ``False`` without touching the session if the job was
        cancelled or superseded, before the build or during it.

        Raises
        ------
        WorldgenError
            If the snapshot cannot be built; nothing is committed.
        """
        with self._sim_lock:
            # A hint only: skips a second of generation for a job already dead.
            if not gate.is_live():
                return False
            scene = self._load_scene(_fresh_seed(), snapshot, current=self._loaded)
            run = self._new_run(scene, self._n)
            with self._lock:
                return gate.commit_if_live(functools.partial(self._commit, scene, self._n, run))

    def _commit(self, scene: _Scene, n: int, run: MissionRun) -> None:
        """Make ``scene`` and ``run`` current, every drone back on its pad.

        Caller holds ``_sim_lock`` and ``_lock`` (or is ``__init__``). Cheap
        by construction: everything slow was built before either was taken.
        """
        self._loaded = scene
        self._n = n
        self._run = run
        self._battery_site = None
        self._eta = MappingEta(self._cfg.sim.timeout_s)
        self._shown = take_snapshot(self._run, self._eta)
        self._buffer.clear()
        # The launch scan's reveals go out with the first frame.
        self._unsent = [self._shown.revealed]
        self._play_t = 0.0
        self._epoch += 1
        # The recorded mission (ADR 0017) starts over too, from the pre-tick
        # state; see RunRecorder for why that logs no fleet events.
        self._recorder = RunRecorder(run, self._cfg)
        self._recorder.record(self._shown)
        _log.debug("viewer reset: %d drone(s), seed=%d, epoch=%d", n, scene.seed, self._epoch)

    def _scene(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "seed": self._loaded.seed,
            "drones": self._n,
            "max_drones": self._cfg.viewer.max_drones,
            "time_scale": self._time_scale,
            "min_time_scale": MIN_TIME_SCALE,
            "max_time_scale": MAX_TIME_SCALE,
            "pads": self._run.pads.tolist(),
            "lot": self._loaded.manifest.lot_bounds.tolist(),
            "location": location_doc(
                self._loaded.site, self._loaded.manifest.notes, self._loaded.manifest
            ),
        }

    def _latest_t(self) -> float:
        """Time of the newest recorded tick."""
        with self._lock:
            return self._buffer[-1].t if self._buffer else self._shown.t

    def _produce_one(self) -> None:
        """Run one control tick, record it for the page's buffer, and log it for ADR 0017."""
        with self._sim_lock:
            self._run.tick()
            self._eta.update(self._run.t, len(self._run.controller.frontiers), self._run.phase)
            if self._battery_site is None and self._run.phase not in _MAPPING_PHASES:
                self._battery_site = self._suggest_sites()
            snap = take_snapshot(self._run, self._eta, self._battery_site)
            self._recorder.record(snap)
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
        while self._buffer and self._buffer[0].t <= self._play_t + T_EPS:
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
        self, prev: Snapshot, cur: Snapshot, alpha: float, revealed: npt.NDArray[np.int64]
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
            "mapping": {"remaining_s": prev.eta_s, "done_at_s": prev.mapped_at_s},
        }

    def _suggest_sites(self) -> dict[str, Any]:
        """Judge the finished map against the checklist, as the page draws the result.

        Caller holds ``_sim_lock``. A map with nothing to site against -- no
        meter, no wall beside it -- is an outcome to show, not an error to
        raise, so it comes back as an empty list with the reason.
        """
        try:
            assessment = assess_site(self._run.mapper.state, self._site_rules)
        except SiteError as exc:
            _log.info("no battery site suggested: %s", exc)
            return {"sites": [], "message": str(exc), "battery": None}
        _log.info(
            "battery siting: %s (%d site(s) offered) at t=%.1f s",
            assessment.verdict.value,
            len(assessment.sites),
            self._run.t,
        )
        battery = self._site_rules.battery
        return {
            "sites": [
                self._site_entry(rank, site) for rank, site in enumerate(assessment.sites, start=1)
            ],
            "message": None,
            "battery": [battery.width_m, battery.depth_m, battery.height_m],
            "verdict": assessment.verdict.value,
            "justification": assessment.justification,
        }

    def _site_entry(self, rank: int, site: SiteCandidate) -> dict[str, Any]:
        """One suggested site as the page draws it."""
        viewer = self._cfg.viewer
        rgb = {
            SiteVerdict.PASS: viewer.site_rgb,
            SiteVerdict.MANUAL_REVIEW: viewer.site_warning_rgb,
            SiteVerdict.REJECT: viewer.site_reject_rgb,
        }[site.verdict]
        return {
            "rank": rank,
            "pos": site.pos.tolist(),
            "yaw": math.atan2(float(site.wall_normal[1]), float(site.wall_normal[0])),
            "meter": site.conduit[0].tolist(),
            "cost": site.cost,
            # JSON has no infinity: a rule with nothing to measure against reports none.
            "breakdown": {k: v if math.isfinite(v) else None for k, v in site.breakdown.items()},
            "verdict": site.verdict.value,
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

    Raises
    ------
    WorldgenError
        If an object's :attr:`~canopy.contracts.SceneObject.materials` runs do
        not add up to its face count. :meth:`~canopy.worldgen.assets.AssetLibrary.build`
        guarantees they do, so this can only mean a hand-built or hand-edited
        manifest, and painting it anyway would silently misalign colours past
        wherever the counts first diverge.
    """
    objects = []
    for obj in manifest.objects:
        vertices = geometry.vertices[obj.obj_id]
        faces = geometry.faces[obj.obj_id]
        entry: dict[str, Any] = {
            "id": obj.obj_id,
            "cls": obj.cls.name,
            "color": [c / 255.0 for c in obj.color],
            "background": obj.background,
            "tri_offset": int(geometry.obj_tri_offset[obj.obj_id]),
            "positions": _b64(vertices.astype(np.float32)),
            "indices": _b64(faces.astype(np.uint32).ravel()),
        }
        if obj.materials:
            covered = sum(run.n_faces for run in obj.materials)
            if covered != len(faces):
                msg = (
                    f"object {obj.obj_id} ({obj.cls.name}): material runs cover {covered} "
                    f"faces, but the mesh has {len(faces)}"
                )
                raise WorldgenError(msg)
            entry["colors"] = _b64(
                np.repeat(
                    np.array([run.rgb for run in obj.materials], dtype=np.uint8),
                    [run.n_faces for run in obj.materials],
                    axis=0,
                )
            )
        objects.append(entry)
    return {"seed": seed, "objects": objects}


def _fresh_seed() -> int:
    return secrets.randbits(_SEED_BITS)
