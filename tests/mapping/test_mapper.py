"""Tests for :mod:`canopy.mapping.mapper`.

The mapper takes only :class:`~canopy.contracts.Observation` -- no manifest, no
geometry (ADR 0016) -- so every scan here is a hand-built :class:`Scan` fed
through :meth:`~canopy.contracts.Scan.observation`, exactly as
:class:`~canopy.planning.MissionRun` hands the mapper a live sweep.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from canopy.config import Config
from canopy.contracts import Cls, Observation, Scan
from canopy.mapping.mapper import Mapper

#: Operator envelope covering the synthetic scenes below, shape ``(2, 3)``.
_ENVELOPE: npt.NDArray[np.float64] = np.array([[-15.0, -20.0, 0.0], [15.0, 20.0, 12.0]])
_LAUNCH_XY: npt.NDArray[np.float64] = np.array([0.0, 0.0])


def _scan(
    origin: npt.NDArray[np.float64],
    dirs: npt.NDArray[np.float64],
    dist: npt.NDArray[np.float64],
    *,
    rgb: npt.NDArray[np.uint8] | None = None,
    grid_shape: tuple[int, int] | None = None,
    t: float = 0.0,
) -> Scan:
    """Build a :class:`Scan` with placeholder ground-truth ids the mapper never sees."""
    n = dist.shape[0]
    return Scan(
        drone_id=0,
        t=t,
        origin=origin,
        dirs=dirs,
        dist=dist,
        obj_ids=np.full(n, -1, dtype=np.int32),
        tri_ids=np.full(n, -1, dtype=np.int32),
        rgb=rgb,
        grid_shape=grid_shape,
    )


def _single_ray_hit() -> Observation:
    """One ray from ``(0, 0, 1)`` straight at a point 2 m away."""
    origin = np.array([0.0, 0.0, 1.0])
    dirs = np.array([[1.0, 0.0, 0.0]])
    dist = np.array([2.0])
    return _scan(origin, dirs, dist).observation()


def test_version_bumps_only_on_change(cfg: Config) -> None:
    """Repeating the same scan (or nothing new) must not bump ``version``."""
    mapper = Mapper(cfg, _ENVELOPE, _LAUNCH_XY)
    assert mapper.version == 0

    obs = _single_ray_hit()
    mapper.integrate(obs)
    v1 = mapper.version
    assert v1 > 0

    mapper.integrate(obs)
    assert mapper.version == v1


def test_mark_free_box_bumps_version(cfg: Config) -> None:
    """Seeding the launch column is itself an occupancy change."""
    mapper = Mapper(cfg, _ENVELOPE, _LAUNCH_XY)

    lo = mapper.state.origin
    hi = lo + np.array([1.0, 1.0, 1.0])
    mapper.mark_free_box(lo, hi)
    assert mapper.version == 1
    mapper.mark_free_box(lo, hi)
    assert mapper.version == 1


def _wall_grid_observation(
    cfg: Config,
    *,
    origin: npt.NDArray[np.float64] | None = None,
    wall_y: float = 5.0,
    n_az: int = 32,
    n_el: int = 9,
) -> Observation:
    """Build a synthetic 360-degree sweep of an infinite flat wall at ``y = wall_y``.

    Built by ray/plane intersection, not a mesh: exactly the azimuth-major
    grid layout a real lidar reports (:attr:`~canopy.contracts.Observation.grid_shape`),
    so :class:`~canopy.mapping.surface.SurfaceTracker` can estimate a normal
    from each return's neighbours and mark it seen.
    """
    origin = np.array([0.0, 0.0, 1.5]) if origin is None else origin
    az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    el = np.deg2rad(np.linspace(-45.0, 45.0, n_el))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    dirs = np.stack(
        [
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    dy = dirs[:, 1]
    # Only rays pointed usefully at the wall (not near-glancing, which blows up
    # the intersection distance) count as forward; the rest miss, as they
    # would if the wall were finite.
    forward = dy > 0.2
    dist = np.where(forward, (wall_y - origin[1]) / np.where(forward, dy, 1.0), np.inf)
    dist = np.where(dist <= cfg.sensor.max_range_m, dist, np.inf)
    return _scan(origin, dirs, dist, grid_shape=(n_az, n_el)).observation()


def test_surface_seen_is_a_live_view_updated_by_integrate(cfg: Config) -> None:
    """``mapper.state.surface_seen`` shares memory with the tracker: no re-fetch needed."""
    mapper = Mapper(cfg, _ENVELOPE, _LAUNCH_XY)

    surface_seen = mapper.state.surface_seen
    assert surface_seen is not None
    assert surface_seen.shape == mapper.state.occ.shape
    assert not surface_seen.any()

    mapper.integrate(_wall_grid_observation(cfg))

    assert surface_seen.any()
    assert mapper.state.surface_seen is surface_seen


def _timed_integrate(mapper: Mapper, obs: Observation) -> float:
    """Milliseconds for one :meth:`Mapper.integrate` call."""
    start = time.perf_counter()
    mapper.integrate(obs)
    return (time.perf_counter() - start) * 1000.0


def test_integrate_is_fast(cfg: Config) -> None:
    """Performance target: one full-size scan integrates in well under 20 ms.

    Grid-shaped so the surface tracker's normal estimation runs too, not just
    occupancy carving: both are on every real sweep's critical path now.
    """
    mapper = Mapper(cfg, _ENVELOPE, _LAUNCH_XY)
    n_az, n_el = cfg.sensor.az_rays, cfg.sensor.el_rays
    n_rays = n_az * n_el
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(n_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dist = rng.uniform(0.5, cfg.sensor.max_range_m, size=n_rays)
    obs = _scan(np.zeros(3), dirs, dist, grid_shape=(n_az, n_el)).observation()

    # Best-of-N: a shared test box under load has enough scheduling noise to
    # spike any one run well past its true cost, so take the fastest of a
    # handful rather than a single sample.
    best_ms = min(_timed_integrate(mapper, obs) for _ in range(5))
    assert best_ms < 20.0, f"integrate took {best_ms:.2f} ms"


def _meter_on_wall_scan(t: float, *, coloured: bool) -> Scan:
    """One scan of a yellow meter-sized box on a beige wall at ``y = 5``, from 3 m away.

    The rays are aimed straight at sampled surface points, so the test says
    exactly what the detector sees; ``coloured=False`` is the same scan
    without its colour channel.
    """
    rng = np.random.default_rng(int(t * 10))
    wall = np.column_stack(
        [rng.uniform(-2.0, 2.0, 2000), np.full(2000, 5.0), rng.uniform(0.3, 3.0, 2000)]
    )
    lo, hi = np.array([-0.15, 4.67, 1.25]), np.array([0.15, 5.0, 1.96])
    meter = lo + rng.uniform(size=(2000, 3)) * (hi - lo)
    face = rng.integers(0, 3, 2000)
    meter[np.arange(2000), face] = np.where(rng.random(2000) < 0.5, lo[face], hi[face])
    points = np.concatenate([wall, meter])
    colour = np.concatenate(
        [np.tile([225, 205, 170], (2000, 1)), np.tile([230, 200, 30], (2000, 1))]
    )
    origin = np.array([0.0, 2.0, 1.5])
    ray = points - origin
    dist = np.linalg.norm(ray, axis=1)
    return _scan(
        origin,
        ray / dist[:, None],
        dist,
        rgb=colour.astype(np.uint8) if coloured else None,
        t=t,
    )


def test_discovered_comes_from_coloured_scans(cfg: Config) -> None:
    """The mapper publishes what its detector finds, replacing ``discovered``, never mutating it.

    Colourless scans leave nothing to classify. Objects are re-extracted on
    the first scan due each ``1 / perception.extract_hz`` seconds of scan time.
    """
    mapper = Mapper(cfg, _ENVELOPE, _LAUNCH_XY)

    mapper.integrate(_meter_on_wall_scan(0.0, coloured=False).observation())
    assert mapper.state.discovered == {}
    empty = mapper.state.discovered

    period = 1.0 / cfg.perception.extract_hz
    for step in range(1, 6):
        mapper.integrate(_meter_on_wall_scan(step * period / 5.0, coloured=True).observation())
    found = mapper.state.discovered
    assert found is not empty
    assert empty == {}
    assert [o.cls for o in found.values()] == [Cls.METER]
    (meter,) = found.values()
    np.testing.assert_allclose(meter.box.center, [0.0, 4.835, 1.605], atol=0.03)
