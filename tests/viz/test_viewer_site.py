"""Address-seeded sites in the desktop viewer: :class:`ViewerSession`'s job bookkeeping.

``suggest_addresses`` and ``fetch_site`` are monkeypatched throughout: they are
network calls, and these tests are about what :class:`ViewerSession` does with
their result -- job state, lock discipline, the ``location`` payload -- not
about geodata itself, which has its own tests in ``tests/worldgen``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import canopy.viz.site_jobs as site_jobs_module
from canopy.config import Config
from canopy.contracts import ResidentialDecision, ResolvedAddress, SiteBuilding, SiteSnapshot
from canopy.errors import GeodataError, SiteRejectedError
from canopy.viz.session import ViewerSession

#: A suggestion dict good enough for any test that does not care about its
#: content, because ``fetch_site`` is monkeypatched to ignore it too.
_ANY_SUGGESTION = {
    "label": "1 Test Street, Testville",
    "provider": "fixture",
    "ref": "fixture:way/1",
    "lat_deg": 1.0,
    "lon_deg": 2.0,
}


def _suggestion(address: ResolvedAddress) -> dict[str, Any]:
    """One entry of :meth:`ViewerSession.suggest_addresses`' ``suggestions``."""
    return {
        "label": address.label,
        "provider": address.provider,
        "ref": address.ref,
        "lat_deg": address.lat_deg,
        "lon_deg": address.lon_deg,
        "attribution": address.attribution,
    }


def _wait_for_state(
    session: ViewerSession, job_id: int, state: str, *, timeout: float = 5.0
) -> dict[str, Any]:
    """Poll :meth:`ViewerSession.site_job` until it reports ``state``, or fail the test."""
    deadline = time.monotonic() + timeout
    status = session.site_job(job_id)
    while status["state"] != state:
        if time.monotonic() > deadline:
            pytest.fail(f"job {job_id} never reached {state!r}; last status was {status}")
        time.sleep(0.01)
        status = session.site_job(job_id)
    return status


def test_suggest_addresses_returns_matches(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ViewerSession(cfg, drones=1, seed=101, scene_dir=tmp_path)
    address = ResolvedAddress(
        label="12 Oak Street, Springfield",
        provider="fixture",
        ref="fixture:way/1",
        lat_deg=40.1,
        lon_deg=-75.2,
    )
    monkeypatch.setattr(site_jobs_module, "_suggest_addresses", lambda text, cfg: [address])  # noqa: ARG005

    result = session.suggest_addresses("12 Oak")

    assert result["error"] is None
    assert result["suggestions"] == [_suggestion(address)]


def test_suggest_addresses_turns_geodata_error_into_a_string(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ViewerSession(cfg, drones=1, seed=102, scene_dir=tmp_path)

    def boom(text: str, cfg: Config) -> list[ResolvedAddress]:
        msg = f"geocoder unreachable for {text!r}"
        raise GeodataError(msg)

    monkeypatch.setattr(site_jobs_module, "_suggest_addresses", boom)

    result = session.suggest_addresses("12 Oak")

    assert result["suggestions"] == []
    assert result["error"] is not None
    assert "12 Oak" in result["error"]


@pytest.mark.slow
def test_build_job_reaches_done_and_scene_reports_address_mode(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site_snapshot: SiteSnapshot
) -> None:
    session = ViewerSession(cfg, drones=1, seed=103, scene_dir=tmp_path)
    monkeypatch.setattr(
        site_jobs_module,
        "fetch_site",
        lambda address, cfg, *, query="", on_retry=None: site_snapshot,  # noqa: ARG005
    )
    before_epoch = session.scene()["epoch"]

    job = session.start_site_build(_suggestion(site_snapshot.address), "12 test street")
    status = _wait_for_state(session, job["job_id"], "done")

    assert status["error"] is None
    assert status["location"] is not None
    assert status["location"]["site_id"] == site_snapshot.site_id

    scene = session.scene()
    assert scene["epoch"] > before_epoch
    assert scene["location"]["mode"] == "address"
    assert scene["location"]["label"] == site_snapshot.address.label
    assert scene["location"]["site_id"] == site_snapshot.site_id
    assert scene["location"]["verdict"]["decision"] == "accept"
    assert "footprint" in scene["location"]["observed"]
    assert scene["location"]["inferred"]
    assert scene["location"]["attribution"]
    assert session.world()["seed"] == scene["seed"]


@pytest.mark.slow
def test_ask_job_waits_in_needs_confirmation_until_confirmed(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_site: Callable[..., SiteSnapshot],
) -> None:
    ask_site = make_site(decision=ResidentialDecision.ASK)
    session = ViewerSession(cfg, drones=1, seed=104, scene_dir=tmp_path)
    monkeypatch.setattr(
        site_jobs_module,
        "fetch_site",
        lambda address, cfg, *, query="", on_retry=None: ask_site,  # noqa: ARG005
    )

    job = session.start_site_build(_suggestion(ask_site.address), "")
    status = _wait_for_state(session, job["job_id"], "needs_confirmation")
    assert status["verdict"] is not None
    assert status["verdict"]["decision"] == "ask"
    assert status["location"]["label"] == ask_site.address.label

    # It does not resolve on its own: still waiting a moment later.
    time.sleep(0.05)
    assert session.site_job(job["job_id"])["state"] == "needs_confirmation"
    assert session.scene()["location"]["mode"] == "random"

    session.confirm_site_build(job["job_id"])
    _wait_for_state(session, job["job_id"], "done")
    assert session.scene()["location"]["site_id"] == ask_site.site_id


def test_rejected_job_fails_with_the_message(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ViewerSession(cfg, drones=1, seed=105, scene_dir=tmp_path)

    def boom(
        address: ResolvedAddress, cfg: Config, *, query: str = "", on_retry: object = None
    ) -> SiteSnapshot:
        msg = f"{address.label} is zoned commercial"
        raise SiteRejectedError(msg)

    monkeypatch.setattr(site_jobs_module, "fetch_site", boom)

    job = session.start_site_build(_ANY_SUGGESTION, "")
    status = _wait_for_state(session, job["job_id"], "failed")

    assert status["error"] is not None
    assert "zoned commercial" in status["error"]
    assert session.scene()["location"]["mode"] == "random"


def test_an_address_that_fetches_but_cannot_be_built_fails_and_keeps_the_old_scene(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_site: Callable[..., SiteSnapshot],
) -> None:
    """A build the generator refuses fails the job; it must not wedge the session.

    The lot here is a plain box 0.3 m larger than the house all round, so no
    wall has the meter's working space and generation itself raises after
    the fetch succeeded. The session must stay
    on the scene it was showing -- still random mode, still its old seed -- so
    a later New seed does not retry the address that failed.
    """
    session = ViewerSession(cfg, drones=1, seed=109, scene_dir=tmp_path)
    box = SiteBuilding(
        footprint=np.array([[-6.0, -7.0], [6.0, -7.0], [6.0, 7.0], [-6.0, 7.0]]),
        levels=2,
        height_m=None,
        roof_shape="gabled",
        source="fixture:way/1",
    )
    unbuildable = make_site(house=box, lot_m=(12.6, 14.6))
    monkeypatch.setattr(site_jobs_module, "fetch_site", lambda *_a, **_k: unbuildable)
    monkeypatch.setattr(site_jobs_module, "save_snapshot", lambda *_a, **_k: tmp_path)
    before = session.scene()

    job = session.start_site_build(_ANY_SUGGESTION, "")
    status = _wait_for_state(session, job["job_id"], "failed", timeout=30.0)

    assert "meter_working_space" in (status["error"] or "")
    after = session.scene()
    assert after["location"]["mode"] == "random"
    assert after["seed"] == before["seed"]


@pytest.mark.slow
def test_a_newer_job_supersedes_an_older_one(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site_snapshot: SiteSnapshot,
    make_site: Callable[..., SiteSnapshot],
) -> None:
    older_site = make_site(decision=ResidentialDecision.ACCEPT, lot_m=(30.0, 40.0))
    gate = threading.Event()

    def slow_fetch(
        address: ResolvedAddress, cfg: Config, *, query: str = "", on_retry: object = None
    ) -> SiteSnapshot:
        gate.wait(5.0)
        return older_site

    session = ViewerSession(cfg, drones=1, seed=106, scene_dir=tmp_path)
    monkeypatch.setattr(site_jobs_module, "fetch_site", slow_fetch)
    old_job = session.start_site_build(_ANY_SUGGESTION, "old")

    monkeypatch.setattr(
        site_jobs_module,
        "fetch_site",
        lambda address, cfg, *, query="", on_retry=None: site_snapshot,  # noqa: ARG005
    )
    new_job = session.start_site_build(_suggestion(site_snapshot.address), "new")
    assert new_job["job_id"] != old_job["job_id"]
    _wait_for_state(session, new_job["job_id"], "done")
    assert session.scene()["location"]["site_id"] == site_snapshot.site_id

    # Let the superseded fetch finish and give its worker a moment to run; its
    # result must land nowhere.
    gate.set()
    time.sleep(0.3)
    assert session.scene()["location"]["site_id"] == site_snapshot.site_id
    assert session.site_job(old_job["job_id"])["state"] == "cancelled"


@pytest.mark.slow
def test_use_random_location_returns_to_random_mode(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site_snapshot: SiteSnapshot
) -> None:
    session = ViewerSession(cfg, drones=1, seed=107, scene_dir=tmp_path)
    monkeypatch.setattr(
        site_jobs_module,
        "fetch_site",
        lambda address, cfg, *, query="", on_retry=None: site_snapshot,  # noqa: ARG005
    )
    job = session.start_site_build(_suggestion(site_snapshot.address), "")
    _wait_for_state(session, job["job_id"], "done")
    assert session.scene()["location"]["mode"] == "address"

    scene = session.use_random_location()

    assert scene["location"]["mode"] == "random"
    assert scene["location"]["site_id"] == ""
    assert session.scene()["location"]["mode"] == "random"


def test_advance_is_not_blocked_while_a_fetch_is_in_flight(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame call must not wait on a site's fetch, however long the fetch takes.

    The fetch is parked for ``stuck_s``. Had ``advance()`` waited on it -- a
    session lock held across the network call -- it could not return before
    the fetch did, so returning well inside that window while the job still
    reports "running" proves the two are independent. No tighter wall-clock
    budget is asserted: how fast ``advance()`` itself is belongs to
    ``test_started_session_advances_fast_and_moves_time_forward`` in
    ``test_viewer.py``, and a load-sensitive budget here would only make this
    test flaky without making it stricter.
    """
    gate = threading.Event()
    stuck_s = 5.0

    def stuck_fetch(
        address: ResolvedAddress, cfg: Config, *, query: str = "", on_retry: object = None
    ) -> SiteSnapshot:
        gate.wait(stuck_s)
        msg = "never resolves within this test"
        raise GeodataError(msg)

    session = ViewerSession(cfg, drones=1, seed=108, scene_dir=tmp_path)
    monkeypatch.setattr(site_jobs_module, "fetch_site", stuck_fetch)
    job = session.start_site_build(_ANY_SUGGESTION, "")

    try:
        started = time.perf_counter()
        frame = session.advance(0.01)
        elapsed = time.perf_counter() - started
        assert elapsed < stuck_s / 2, f"advance() took {elapsed:.3f}s: it waited on the fetch"
        assert session.site_job(job["job_id"])["state"] == "running"
        assert frame["epoch"] == 0
    finally:
        gate.set()  # release the stuck worker thread before the test ends


def test_a_busy_map_server_is_reported_while_the_fetch_retries(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry's progress reaches the page as the job's ``detail``, while still running."""
    reported = threading.Event()
    gate = threading.Event()

    def retrying_fetch(
        address: ResolvedAddress,
        cfg: Config,
        *,
        query: str = "",
        on_retry: Callable[[int, int, float], None] | None = None,
    ) -> SiteSnapshot:
        assert on_retry is not None
        on_retry(2, 4, 8.0)
        reported.set()
        gate.wait(5.0)
        msg = "never resolves within this test"
        raise GeodataError(msg)

    session = ViewerSession(cfg, drones=1, seed=109, scene_dir=tmp_path)
    monkeypatch.setattr(site_jobs_module, "fetch_site", retrying_fetch)
    job = session.start_site_build(_ANY_SUGGESTION, "")
    try:
        assert reported.wait(5.0)
        status = session.site_job(job["job_id"])
        assert status["state"] == "running"
        assert "busy" in status["detail"]
        assert "attempt 2 of 4" in status["detail"]
    finally:
        gate.set()
