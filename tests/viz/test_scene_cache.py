"""The viewer's generated-scene cache stays bounded (``viewer.scene_cache_max``).

Every re-roll writes a fresh ``scene_dir/<seed>`` directory, so the session
trims the least recently used ones whenever it builds a scene. These tests
pre-fill a scene directory with stale fake seeds and check what survives a
build: the newest seeds up to the limit, always the loaded scene, and
anything that is not a seed directory at all.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import pytest

from canopy.config import Config
from canopy.viz.session import ViewerSession

#: Fake cached seeds, oldest first; none of them collides with a seed built below.
_FAKE_SEEDS = tuple(range(900, 910))


def _with_cache_max(cfg: Config, limit: int) -> Config:
    return dataclasses.replace(cfg, viewer=dataclasses.replace(cfg.viewer, scene_cache_max=limit))


def _prefill(scene_dir: Path) -> Path:
    """Write stale fake seed dirs with ascending mtimes, plus a non-seed dir; return the latter."""
    for age, seed in enumerate(_FAKE_SEEDS):
        seed_dir = scene_dir / str(seed)
        seed_dir.mkdir(parents=True)
        (seed_dir / "manifest.json").write_text("{}", encoding="utf-8")
        # Well in the past, so a freshly built scene is always the newest.
        stamp = 1_000_000_000 + age
        os.utime(seed_dir, (stamp, stamp))
    other = scene_dir / "notes"
    other.mkdir()
    os.utime(other, (0, 0))
    return other


def _seed_dirs(scene_dir: Path) -> set[str]:
    return {p.name for p in scene_dir.iterdir() if p.is_dir() and p.name.isdigit()}


def test_building_a_scene_trims_the_cache_to_the_newest_seeds(cfg: Config, tmp_path: Path) -> None:
    other = _prefill(tmp_path)

    ViewerSession(_with_cache_max(cfg, 3), drones=1, seed=1, scene_dir=tmp_path)

    # The loaded scene plus the two most recently used fakes.
    assert _seed_dirs(tmp_path) == {"1", str(_FAKE_SEEDS[-1]), str(_FAKE_SEEDS[-2])}
    assert other.is_dir(), "a directory that is not a seed must never be evicted"


def test_a_seed_dir_vanishing_mid_scan_does_not_fail_the_build(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process deleting a cached seed between listing and ``stat`` is skipped.

    The first ``stat`` of the doomed directory (the listing's ``is_dir``)
    sees it; the second (reading its mtime) deletes it for real first, as a
    second viewer trimming the same cache would, so ``stat`` raises
    ``FileNotFoundError`` exactly where the race would.
    """
    _prefill(tmp_path)
    doomed = tmp_path / str(_FAKE_SEEDS[0])
    real_stat = Path.stat
    calls = 0

    def racing_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        nonlocal calls
        if self == doomed:
            calls += 1
            if calls == 2:
                shutil.rmtree(doomed)
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", racing_stat)

    session = ViewerSession(_with_cache_max(cfg, 3), drones=1, seed=1, scene_dir=tmp_path)

    assert calls >= 2, "the race was never staged"
    assert session.seed == 1
    assert _seed_dirs(tmp_path) == {"1", str(_FAKE_SEEDS[-1]), str(_FAKE_SEEDS[-2])}


@pytest.mark.slow
def test_the_loaded_scene_survives_a_re_roll_even_at_a_limit_of_one(
    cfg: Config, tmp_path: Path
) -> None:
    """At ``scene_cache_max: 1`` the old scene is still loaded while the new one builds.

    Both are kept through the build (the old one is what the page is showing
    until the swap), and the old one only becomes evictable at the next build.
    """
    _prefill(tmp_path)
    session = ViewerSession(_with_cache_max(cfg, 1), drones=1, seed=1, scene_dir=tmp_path)
    assert _seed_dirs(tmp_path) == {"1"}

    session.new_scene(seed=2)
    assert session.seed == 2
    assert _seed_dirs(tmp_path) == {"1", "2"}

    session.new_scene(seed=3)
    assert _seed_dirs(tmp_path) == {"2", "3"}
