"""Seeding the viewer's property from a real street address, as a background job.

A property can be seeded from a real street address rather than a bare seed
(see ``docs/adr/0015-address-seeded-sites.md``).
:meth:`~canopy.viz.ViewerSession.start_site_build` fetches the address and,
once its residential check clears, builds it, both on a daemon worker thread
outside every session lock -- the fetch alone can take several seconds, and
must freeze the animation no more than a replan may. Progress reaches the page
through :meth:`~canopy.viz.ViewerSession.site_job`, which it polls.

:class:`SiteJobs` is that workflow, owned by the session as a collaborator.
It holds the job bookkeeping and the session's third lock, ``_job_lock``,
which guards only that bookkeeping and is never held across a call to
``fetch_site``, ``save_snapshot`` or anything that builds a scene. The one
step that touches the session -- swapping the built property in -- goes
through the ``install`` callback the session hands over, which takes the
session's own locks first and only then asks this module, via a liveness
predicate that takes ``_job_lock``, whether the job is still wanted. That
keeps the session-wide order ``_sim_lock`` -> ``_lock`` -> ``_job_lock``
intact: nothing here ever calls into the session while holding
``_job_lock``.

Starting another build, or calling
:meth:`~canopy.viz.ViewerSession.use_random_location`, supersedes whatever job
was already in flight: the superseded one keeps running (a live network call
cannot be aborted from here) but its result, whenever it arrives, is checked
against the current job and simply dropped if it no longer matches.

:func:`location_doc` also lives here, because the job status and the
session's ``scene()`` report the same ``location`` payload.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from canopy.contracts import (
    Cls,
    Provenance,
    ResidentialDecision,
    ResidentialVerdict,
    ResolvedAddress,
    SceneManifest,
    SiteSnapshot,
)
from canopy.errors import GeodataError, SimulationError, SiteRejectedError, WorldgenError
from canopy.log import get_logger
from canopy.worldgen import fetch_site, save_snapshot
from canopy.worldgen import suggest_addresses as _suggest_addresses

if TYPE_CHECKING:
    from canopy.config import Config

__all__ = ["InstallSite", "JobGate", "SiteJobs", "location_doc"]

_log = get_logger(__name__)

#: The session's side of a build: ``install(snapshot, gate) -> installed``.
#: See :class:`SiteJobs` for the contract and :class:`JobGate` for ``gate``.
InstallSite = Callable[[SiteSnapshot, "JobGate"], bool]

#: :class:`_SiteJob` states past which nothing more will ever happen to it.
_TERMINAL_JOB_STATES = frozenset({"done", "failed", "cancelled"})

#: What the generator always has to invent for an address-built property,
#: because map data shows a footprint, not a meter, an opening or a shrub.
#: Shown while a site job has no manifest yet; once one exists, the list is
#: read off its objects' provenance instead (:func:`_inferred_summary`).
_INFERRED_SUMMARY: tuple[str, ...] = ("electric meter", "windows and doors", "bushes")

#: How the site card names each class the generator inferred, in the order
#: it lists them. Classes that share a name (a door and a garage door) are
#: listed once; classes missing here (the ground) are never listed.
_INFERRED_NAMES: dict[Cls, str] = {
    Cls.METER: "electric meter",
    Cls.PANEL: "panel and conduit",
    Cls.CONDUIT: "panel and conduit",
    Cls.WINDOW: "windows and doors",
    Cls.DOOR: "windows and doors",
    Cls.GARAGE_DOOR: "windows and doors",
    Cls.ROOF: "roof",
    Cls.TREE: "trees",
    Cls.BUSH: "bushes",
    Cls.AC_UNIT: "AC unit",
    Cls.GAS_METER: "gas meter",
    Cls.FENCE: "fence",
    Cls.SHED: "shed",
}


@dataclass(slots=True)
class _SiteJob:
    """Progress of one address fetch-and-build, polled by the page.

    Mutated in place by its worker thread(s) and read by
    :meth:`SiteJobs.status`, both under ``_job_lock``. Only the most recent
    job is tracked: starting a new one replaces :attr:`SiteJobs._job`
    wholesale, so a stale worker's checks against it (``self._job is job``)
    simply start failing -- see the module docstring on supersession.
    """

    job_id: int
    state: str = "running"
    """One of ``running``, ``needs_confirmation``, ``done``, ``failed``, ``cancelled``."""
    stage: str = "fetching"
    """Human-readable progress within ``state == "running"``: ``fetching`` or ``building``."""
    detail: str = ""
    """A line more about the stage, e.g. that the map server is busy and being retried."""
    error: str | None = None
    """The failure message, once ``state`` is ``failed``."""
    snapshot: SiteSnapshot | None = None
    """Set as soon as :func:`~canopy.worldgen.fetch_site` returns, whatever its verdict."""


class JobGate:
    """One job's liveness, as the session's ``install`` callback sees it.

    Both methods take ``_job_lock``, the innermost lock, so the session may
    call them while holding ``_sim_lock`` and ``_lock``.

    Parameters
    ----------
    job_lock
        :class:`SiteJobs`' ``_job_lock``.
    job
        The job being installed.
    live
        Whether ``job`` is still current and uncancelled; caller holds
        ``job_lock``.
    """

    def __init__(self, job_lock: threading.Lock, job: _SiteJob, live: Callable[[], bool]) -> None:
        self._job_lock = job_lock
        self._job = job
        self._live = live

    def is_live(self) -> bool:
        """Whether the job is still current and uncancelled -- a hint, not a promise.

        For skipping a build nobody wants any more; the answer can change as
        soon as it is returned, so it must never guard a commit.
        """
        with self._job_lock:
            return self._live()

    def commit_if_live(self, commit: Callable[[], None]) -> bool:
        """Run ``commit`` and mark the job ``done``, atomically, if it is still live.

        Check, commit and state change share one hold of ``_job_lock``, so a
        cancel can land wholly before (nothing installs) or wholly after
        (a no-op on a ``done`` job), never between -- the page never sees a
        job reported ``cancelled`` whose scene went live. ``commit`` must
        therefore be cheap and must not take ``_job_lock`` itself.
        """
        with self._job_lock:
            if not self._live():
                return False
            commit()
            self._job.state = "done"
            return True


class SiteJobs:
    """The address fetch-and-build workflow behind one viewer session.

    Parameters
    ----------
    cfg
        Validated configuration, passed on to the geocoder and the fetch.
    install
        Swaps a fetched snapshot into the session's scene. Called on a worker
        thread with the snapshot and a :class:`JobGate`; it must take its own
        locks, generate the property, and install only through
        :meth:`JobGate.commit_if_live` (which takes ``_job_lock``, the
        innermost lock), which also marks the job ``done``. It may call
        :meth:`JobGate.is_live` first to skip a build nobody wants. Returns
        whether it installed; raises :class:`~canopy.errors.WorldgenError`
        if the snapshot cannot be built, having installed nothing.
    """

    def __init__(self, cfg: Config, install: InstallSite) -> None:
        self._cfg = cfg
        self._install = install
        self._job_lock = threading.Lock()
        self._job: _SiteJob | None = None
        self._job_seq = 0

    def supersede(self) -> None:
        """Forget the live job, so its eventual result lands nowhere."""
        with self._job_lock:
            self._job = None

    def suggest(self, text: str) -> dict[str, Any]:
        """See :meth:`~canopy.viz.ViewerSession.suggest_addresses`."""
        try:
            matches = _suggest_addresses(text, self._cfg)
        except GeodataError as exc:
            return {"suggestions": [], "error": str(exc)}
        return {
            "suggestions": [
                {
                    "label": a.label,
                    "provider": a.provider,
                    "ref": a.ref,
                    "lat_deg": a.lat_deg,
                    "lon_deg": a.lon_deg,
                    "attribution": a.attribution,
                }
                for a in matches
            ],
            "error": None,
        }

    def start(self, suggestion: dict[str, Any], query: str) -> dict[str, Any]:
        """See :meth:`~canopy.viz.ViewerSession.start_site_build`."""
        address = ResolvedAddress(
            label=str(suggestion["label"]),
            provider=str(suggestion["provider"]),
            ref=str(suggestion["ref"]),
            lat_deg=float(suggestion["lat_deg"]),
            lon_deg=float(suggestion["lon_deg"]),
            attribution=str(suggestion.get("attribution") or ""),
        )
        with self._job_lock:
            self._job_seq += 1
            job = _SiteJob(job_id=self._job_seq)
            self._job = job
        threading.Thread(
            target=self._guarded,
            args=(job, self._run_site_job, job, address, query),
            daemon=True,
            name="canopy-site-fetch",
        ).start()
        return {"job_id": job.job_id}

    def confirm(self, job_id: int) -> dict[str, Any]:
        """See :meth:`~canopy.viz.ViewerSession.confirm_site_build`."""
        with self._job_lock:
            job = self._job
            if job is None or job.job_id != job_id or job.state != "needs_confirmation":
                return self._job_status(job, job_id)
            if job.snapshot is None:
                msg = f"site job {job_id} reached needs_confirmation with no snapshot"
                raise SimulationError(msg)
            snapshot = job.snapshot
            job.state, job.stage = "running", "building"
            status = self._job_status(job, job_id)
        threading.Thread(
            target=self._guarded,
            args=(job, self._finish_site_job, job, snapshot),
            daemon=True,
            name="canopy-site-build",
        ).start()
        return status

    def cancel(self, job_id: int) -> dict[str, Any]:
        """See :meth:`~canopy.viz.ViewerSession.cancel_site_build`."""
        with self._job_lock:
            job = self._job
            if job is not None and job.job_id == job_id and job.state not in _TERMINAL_JOB_STATES:
                job.state, job.stage = "cancelled", ""
            return self._job_status(job, job_id)

    def status(self, job_id: int) -> dict[str, Any]:
        """See :meth:`~canopy.viz.ViewerSession.site_job`."""
        with self._job_lock:
            return self._job_status(self._job, job_id)

    # -- worker threads -------------------------------------------------------
    def _run_site_job(self, job: _SiteJob, address: ResolvedAddress, query: str) -> None:
        """Stage ``fetching``, then hand off to confirmation or to building (a worker thread).

        Runs outside every session lock, so a slow geocoder or a slow Overpass
        query never delays a frame. Every step re-checks :attr:`_job` before it
        writes anything the page or the scene could observe, so a superseded
        or cancelled job's outcome, whenever it arrives, lands nowhere.
        """

        def on_retry(next_attempt: int, attempts: int, wait_s: float) -> None:
            with self._job_lock:
                if self._job is job:
                    job.detail = (
                        f"The map server is busy; trying again in {wait_s:.0f} s "
                        f"(attempt {next_attempt} of {attempts})."
                    )

        try:
            snapshot = fetch_site(address, self._cfg, query=query, on_retry=on_retry)
        except (GeodataError, SiteRejectedError) as exc:
            self._fail_job(job, str(exc))
            return
        with self._job_lock:
            if self._job is not job or job.state == "cancelled":
                return
            job.snapshot = snapshot
            if snapshot.verdict.decision is ResidentialDecision.ASK:
                job.state, job.stage = "needs_confirmation", ""
                return
        self._finish_site_job(job, snapshot)

    def _finish_site_job(self, job: _SiteJob, snapshot: SiteSnapshot) -> None:
        """Stage ``building``: save the snapshot, then swap it into the scene (a worker thread).

        Shared by the initial ACCEPT path and by :meth:`confirm`'s own worker
        thread, so a build started either way goes through exactly one place
        that touches the session. The ``install`` callback commits only
        through :meth:`JobGate.commit_if_live`, under the session's locks, so
        a job cancelled or superseded while this was saving or building never
        reaches the live scene, and one that does is ``done`` in the same step.
        """
        with self._job_lock:
            if self._job is not job or job.state == "cancelled":
                return
            job.stage, job.detail = "building", ""
        try:
            save_snapshot(snapshot)
        except WorldgenError as exc:
            self._fail_job(job, str(exc))
            return
        try:
            gate = JobGate(self._job_lock, job, functools.partial(self._live, job))
            # The gate marks the job done when it commits; nothing to do after.
            self._install(snapshot, gate)
        except WorldgenError as exc:
            # The address fetched but cannot be built -- no wall takes a
            # meter, say. install() commits nothing until generation succeeds,
            # so the session stays on the scene it is still showing, and a
            # later New seed does not retry the address that failed.
            self._fail_job(job, str(exc))

    def _live(self, job: _SiteJob) -> bool:
        """Whether ``job`` is still the current, uncancelled job. Caller holds ``_job_lock``."""
        return self._job is job and job.state != "cancelled"

    def _fail_job(self, job: _SiteJob, message: str) -> None:
        """Mark ``job`` failed with ``message``, unless it was superseded or cancelled."""
        with self._job_lock:
            if self._job is job and job.state != "cancelled":
                job.state, job.error = "failed", message

    def _guarded(self, job: _SiteJob, work: Callable[..., None], *args: Any) -> None:
        """Run one step of a site job on its worker thread, failing the job on any surprise.

        The expected failures -- no data, not a home, no wall for a meter --
        are handled where they happen. Anything else is a bug, but a worker
        thread that dies takes its job with it and the page would poll
        "running" forever, so it is logged with its traceback and reported as
        the job's failure instead.
        """
        try:
            work(*args)
        except Exception as exc:  # the thread boundary: report, never die silently
            _log.exception("site job %d failed unexpectedly", job.job_id)
            self._fail_job(job, f"internal error: {exc}")

    def _job_status(self, job: _SiteJob | None, job_id: int) -> dict[str, Any]:
        """Build the dict :meth:`status` reports. Caller holds ``_job_lock``.

        ``job_id`` not matching the live job reads the same as an explicit
        cancellation: whatever the page was polling for, there is nothing more
        to report for it.
        """
        if job is None or job.job_id != job_id:
            return {
                "job_id": job_id,
                "state": "cancelled",
                "stage": "",
                "error": None,
                "verdict": None,
                "location": None,
            }
        snapshot = job.snapshot
        return {
            "job_id": job.job_id,
            "state": job.state,
            "stage": job.stage,
            "detail": job.detail,
            "error": job.error,
            "verdict": None if snapshot is None else _verdict_payload(snapshot.verdict),
            "location": None if snapshot is None else location_doc(snapshot, snapshot.notes),
        }


def _verdict_payload(verdict: ResidentialVerdict) -> dict[str, Any]:
    """Format a :class:`~canopy.contracts.ResidentialVerdict` as the page reads it."""
    return {
        "p_residential": verdict.p_residential,
        "decision": verdict.decision.value,
        "reasons": list(verdict.reasons),
    }


def _observed_summary(site: SiteSnapshot) -> list[str]:
    """Short human strings for what the fetched data actually showed.

    Everything here is read straight off the snapshot: the footprint,
    storeys, roof, trees, neighbours and street were all mapped, not guessed,
    so none of this needs the generated manifest.
    """
    house = site.house
    observed = ["footprint"]
    if house.levels is not None:
        observed.append(f"{house.levels} storey" + ("" if house.levels == 1 else "s"))
    if house.roof_shape is not None:
        observed.append(f"{house.roof_shape} roof")
    n_trees = int(np.asarray(site.trees, dtype=np.float64).reshape(-1, 3).shape[0])
    if n_trees:
        observed.append(f"{n_trees} tree" + ("" if n_trees == 1 else "s"))
    if site.neighbours:
        n = len(site.neighbours)
        observed.append(f"{n} neighbour" + ("" if n == 1 else "s"))
    if site.street is not None:
        observed.append(site.street_name or "street")
    return observed


def _inferred_summary(manifest: SceneManifest) -> list[str]:
    """Name what the generator invented on the surveyed lot, from each object's provenance.

    Read off the built property rather than assumed, so a site with no mapped
    trees lists its trees, and one whose roof shape was mapped does not list
    the roof. Repaired objects count as inferred: a repair is a redraw.
    """
    invented = {
        obj.cls
        for obj in manifest.objects
        if not obj.background and obj.provenance is not Provenance.OBSERVED
    }
    return list(dict.fromkeys(name for cls, name in _INFERRED_NAMES.items() if cls in invented))


def _distinct_attributions(site: SiteSnapshot) -> list[str]:
    """Every source's attribution text, once each, in first-seen order.

    One that another already contains (the geocoder's credit repeats the map
    data's) is dropped rather than printed twice.
    """
    texts = list(dict.fromkeys(src.attribution for src in site.sources))
    return [t for t in texts if not any(t != other and t in other for other in texts)]


def location_doc(
    site: SiteSnapshot | None, notes: Sequence[str], manifest: SceneManifest | None = None
) -> dict[str, Any]:
    """Build the ``location`` payload :meth:`ViewerSession.scene` and :meth:`site_job` share.

    ``notes`` is taken separately from ``site`` because the two callers have a
    different set on hand: :meth:`ViewerSession.scene` reads the generated
    manifest's, once a scene exists; a job being polled before generation has
    run yet has only the snapshot's own.

    Parameters
    ----------
    site
        The pinned snapshot, or ``None`` in random mode.
    notes
        Findings for a reviewer to see, most specific to the caller.
    manifest
        The property built from ``site``, when there is one yet: the
        inferred list is then read off its provenance.
    """
    if site is None:
        return {
            "mode": "random",
            "label": None,
            "site_id": "",
            "verdict": None,
            "observed": [],
            "inferred": [],
            "notes": list(notes),
            "attribution": [],
        }
    return {
        "mode": "address",
        "label": site.address.label,
        "site_id": site.site_id,
        "verdict": _verdict_payload(site.verdict),
        "observed": _observed_summary(site),
        "inferred": (
            _inferred_summary(manifest) if manifest is not None else list(_INFERRED_SUMMARY)
        ),
        "notes": list(notes),
        "attribution": _distinct_attributions(site),
    }
