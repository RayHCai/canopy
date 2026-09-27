""":class:`~canopy.viz.site_jobs.SiteJobs`: a cancel never splits "installed" from "done".

Driven with a fake ``install`` callback instead of a real session, so each
test can land a cancel at an exact point in the install -- between the
liveness hint and the commit, or after the commit -- without sleeps or
timing. ``fetch_site`` and ``save_snapshot`` are monkeypatched: no network,
no disk.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import canopy.viz.site_jobs as site_jobs_module
from canopy.config import Config
from canopy.contracts import SiteSnapshot
from canopy.viz.site_jobs import JobGate, SiteJobs

_SUGGESTION = {
    "label": "1 Test Street, Testville",
    "provider": "fixture",
    "ref": "fixture:way/1",
    "lat_deg": 1.0,
    "lon_deg": 2.0,
}


def _run_job(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    site: SiteSnapshot,
    tmp_path: Path,
    install: Callable[[SiteJobs, int, JobGate], bool],
) -> tuple[SiteJobs, dict[str, Any]]:
    """Start one accepted job, let ``install`` run to completion, and return the final status."""
    monkeypatch.setattr(site_jobs_module, "fetch_site", lambda *_a, **_k: site)
    monkeypatch.setattr(site_jobs_module, "save_snapshot", lambda *_a, **_k: tmp_path)
    finished = threading.Event()
    job_id: list[int] = []
    started = threading.Event()

    def fake_install(snapshot: SiteSnapshot, gate: JobGate) -> bool:
        started.wait(5.0)
        try:
            return install(jobs, job_id[0], gate)
        finally:
            finished.set()

    jobs = SiteJobs(cfg, fake_install)
    job_id.append(jobs.start(_SUGGESTION, "")["job_id"])
    started.set()
    assert finished.wait(5.0), "install never ran"
    return jobs, jobs.status(job_id[0])


def test_a_cancel_between_the_liveness_hint_and_the_commit_installs_nothing(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    committed: list[bool] = []

    def install(jobs: SiteJobs, job_id: int, gate: JobGate) -> bool:
        assert gate.is_live()
        jobs.cancel(job_id)  # lands after the hint, before the commit
        return gate.commit_if_live(lambda: committed.append(True))

    _, status = _run_job(cfg, monkeypatch, site_snapshot, tmp_path, install)

    assert committed == [], "a cancelled job's scene must never go live"
    assert status["state"] == "cancelled"


def test_a_committed_job_is_done_and_a_later_cancel_does_not_undo_it(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    committed: list[bool] = []

    def install(jobs: SiteJobs, job_id: int, gate: JobGate) -> bool:
        installed = gate.commit_if_live(lambda: committed.append(True))
        jobs.cancel(job_id)  # lands after the commit: too late to matter
        return installed

    _, status = _run_job(cfg, monkeypatch, site_snapshot, tmp_path, install)

    assert committed == [True]
    assert status["state"] == "done"
