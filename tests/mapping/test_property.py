"""Tests for :mod:`canopy.mapping.property`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config, SafetyCfg
from canopy.contracts import MapState, Occ
from canopy.errors import ConfigError
from canopy.mapping.property import infer_survey_bounds, survey_envelope


def test_survey_envelope_is_offset_from_the_pads_centroid(cfg: Config) -> None:
    """The box is the pads' xy centroid plus the configured x/y offsets, 0 to ``map.height_m``."""
    pads = np.array([[10.0, -20.0, 0.0], [14.0, -20.0, 0.0]])
    centre = pads[:, :2].mean(axis=0)
    envelope = survey_envelope(pads, cfg)

    x_lo, x_hi = cfg.safety.envelope_x_m
    y_lo, y_hi = cfg.safety.envelope_y_m
    np.testing.assert_allclose(envelope[0], [centre[0] + x_lo, centre[1] + y_lo, 0.0])
    np.testing.assert_allclose(envelope[1], [centre[0] + x_hi, centre[1] + y_hi, cfg.map.height_m])


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda safety: replace(safety, envelope_x_m=(1.0, 10.0)),  # both positive
        lambda safety: replace(safety, envelope_y_m=(-10.0, -1.0)),  # both negative
    ],
    ids=["envelope_x_m does not contain 0", "envelope_y_m does not contain 0"],
)
def test_safety_cfg_validate_rejects_an_envelope_not_straddling_zero(
    cfg: Config, make_bad: Callable[[SafetyCfg], SafetyCfg]
) -> None:
    """The envelope is measured from the launch point, so it must contain it."""
    with pytest.raises(ConfigError):
        make_bad(cfg.safety).validate()


# ---------------------------------------------------------------------------
# infer_survey_bounds
# ---------------------------------------------------------------------------
def _blank_state(cfg: Config, nx: int, ny: int) -> MapState:
    """Build an all-FREE map, ``nx`` x ``ny`` voxels in plan, full ``map.height_m`` tall."""
    nz = round(cfg.map.height_m / cfg.map.voxel_m)
    occ = np.full((nx, ny, nz), Occ.FREE, dtype=np.uint8)
    return MapState(occ=occ, origin=np.zeros(3), voxel=cfg.map.voxel_m)


def _band_indices(cfg: Config, state: MapState) -> tuple[int, int]:
    """Return the occupancy grid's k-index range spanning ``map.house_band_z_m``."""
    v = state.voxel
    z0 = float(state.origin[2])
    k0 = int(np.ceil((cfg.map.house_band_z_m[0] - z0) / v))
    k1 = int(np.floor((cfg.map.house_band_z_m[1] - z0) / v))
    return k0, k1


def _mark_wall(state: MapState, cfg: Config, i_range: tuple[int, int], j: int) -> None:
    """Stand a wall on the ground: columns ``i_range`` at row ``j``, OCC up through the band."""
    _, k1 = _band_indices(cfg, state)
    state.occ[i_range[0] : i_range[1], j, 0:k1] = Occ.OCC


def test_infer_survey_bounds_is_none_on_an_empty_map(cfg: Config) -> None:
    """Nothing occupied anywhere: no wall, so no house."""
    state = _blank_state(cfg, 40, 40)
    assert infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map) is None


def test_infer_survey_bounds_ignores_a_wall_run_shorter_than_house_min_wall_m(
    cfg: Config,
) -> None:
    """A short run of wall (a fence post, a meter) is clutter, not a building."""
    state = _blank_state(cfg, 40, 40)
    n_short = int(cfg.map.house_min_wall_m / cfg.map.voxel_m) - 2
    _mark_wall(state, cfg, (10, 10 + n_short), j=20)
    assert infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map) is None


def test_infer_survey_bounds_ignores_occupancy_below_the_house_band(cfg: Config) -> None:
    """A long run occupied only below ``house_band_z_m`` (a hedge or a fence) is not a wall."""
    state = _blank_state(cfg, 40, 40)
    v = state.voxel
    n_long = int(cfg.map.house_min_wall_m / v) + 4
    k_bush_top = int(np.floor((cfg.map.house_band_z_m[0] - 0.3) / v))
    assert k_bush_top > 0
    state.occ[10 : 10 + n_long, 20, 0:k_bush_top] = Occ.OCC
    assert infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map) is None


def test_infer_survey_bounds_picks_the_building_nearest_launch_and_grows_it_by_the_margin(
    cfg: Config,
) -> None:
    """Two wall runs far apart: the nearer one is the house, boxed and grown by the margin."""
    state = _blank_state(cfg, 120, 40)
    v = state.voxel
    n_wall = int(cfg.map.house_min_wall_m / v) + 4
    j = 20
    _mark_wall(state, cfg, (10, 10 + n_wall), j=j)
    _mark_wall(state, cfg, (90, 90 + n_wall), j=j)

    launch_near_a = state.origin[:2] + np.array([12.0 * v, j * v])
    bounds = infer_survey_bounds(state, launch_near_a, cfg.map)
    assert bounds is not None

    lo_x = state.origin[0] + 10 * v - cfg.map.survey_margin_m
    hi_x = state.origin[0] + (10 + n_wall) * v + cfg.map.survey_margin_m
    lo_y = state.origin[1] + j * v - cfg.map.survey_margin_m
    hi_y = state.origin[1] + (j + 1) * v + cfg.map.survey_margin_m
    np.testing.assert_allclose(bounds, [[lo_x, lo_y], [hi_x, hi_y]])
    # The far building was not chosen: its wall lies well outside the returned box.
    assert bounds[1, 0] < state.origin[0] + 90 * v


def test_infer_survey_bounds_grows_as_the_house_is_mapped_further(cfg: Config) -> None:
    """A longer wall run gives a larger box: the region grows as the map does, not just latches."""
    state = _blank_state(cfg, 80, 40)
    v = state.voxel
    n_wall = int(cfg.map.house_min_wall_m / v) + 4
    j = 20
    launch = state.origin[:2] + np.array([10.0 * v, j * v])

    _mark_wall(state, cfg, (10, 10 + n_wall), j=j)
    small = infer_survey_bounds(state, launch, cfg.map)
    assert small is not None

    _mark_wall(state, cfg, (10, 10 + n_wall + 20), j=j)
    grown = infer_survey_bounds(state, launch, cfg.map)
    assert grown is not None
    assert grown[1, 0] > small[1, 0]
    np.testing.assert_allclose(grown[0], small[0])


def test_infer_survey_bounds_reads_only_occ_and_grid_geometry(
    cfg: Config, lot: npt.NDArray[np.float64]
) -> None:
    """A voxel size other than ``map.voxel_m`` still gives a self-consistent box.

    :func:`infer_survey_bounds` takes its geometry from ``state`` itself, not
    from ``cfg.map.voxel_m``, so a caller may pass any grid it likes.
    """
    voxel = 0.5
    nx, ny = 60, 40
    nz = round(cfg.map.height_m / voxel)
    occ = np.full((nx, ny, nz), Occ.FREE, dtype=np.uint8)
    state = MapState(occ=occ, origin=lot[0].copy(), voxel=voxel)
    n_wall = int(cfg.map.house_min_wall_m / voxel) + 4
    _mark_wall(state, cfg, (5, 5 + n_wall), j=10)

    launch = state.origin[:2] + np.array([7.0 * voxel, 10.0 * voxel])
    bounds = infer_survey_bounds(state, launch, cfg.map)
    assert bounds is not None
    lo_x = state.origin[0] + 5 * voxel - cfg.map.survey_margin_m
    hi_x = state.origin[0] + (5 + n_wall) * voxel + cfg.map.survey_margin_m
    np.testing.assert_allclose(bounds[:, 0], [lo_x, hi_x])


def test_infer_survey_bounds_ignores_a_tree_crown_standing_on_air(cfg: Config) -> None:
    """A wide crown fills the band, but the drones see open air under it: not a building.

    The crown is nearer the launch than the house, which is how a front-yard
    tree once took the survey off the house entirely.
    """
    state = _blank_state(cfg, 120, 60)
    v = state.voxel
    k0, k1 = _band_indices(cfg, state)
    n_crown = int(cfg.map.house_min_wall_m / v) + 4
    state.occ[10 : 10 + n_crown, 10 : 10 + n_crown, k0:k1] = Occ.OCC
    # The trunk: one solid column under the crown's middle.
    mid = 10 + n_crown // 2
    state.occ[mid, mid, 0:k0] = Occ.OCC
    n_wall = int(cfg.map.house_min_wall_m / v) + 4
    _mark_wall(state, cfg, (80, 80 + n_wall), j=40)

    bounds = infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map)
    assert bounds is not None
    lo_x = state.origin[0] + 80 * v - cfg.map.survey_margin_m
    np.testing.assert_allclose(bounds[0, 0], lo_x)


def test_infer_survey_bounds_keeps_a_wall_whose_underside_is_unobserved(cfg: Config) -> None:
    """UNKNOWN under the band (hidden behind a bush) is no evidence of air: still a wall."""
    state = _blank_state(cfg, 60, 40)
    v = state.voxel
    k0, k1 = _band_indices(cfg, state)
    n_wall = int(cfg.map.house_min_wall_m / v) + 4
    state.occ[10 : 10 + n_wall, 20, 0:k0] = Occ.UNKNOWN
    state.occ[10 : 10 + n_wall, 20, k0:k1] = Occ.OCC
    assert infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map) is not None


def test_infer_survey_bounds_needs_a_span_not_just_an_area(cfg: Config) -> None:
    """A compact block with as many cells as a wall run, but too short a span, is not a building."""
    state = _blank_state(cfg, 60, 60)
    v = state.voxel
    _, k1 = _band_indices(cfg, state)
    n_cells = int(cfg.map.house_min_wall_m / v)
    side = int(np.ceil(np.sqrt(n_cells))) + 1
    assert side * v < cfg.map.house_min_wall_m
    state.occ[20 : 20 + side, 20 : 20 + side, 0:k1] = Occ.OCC
    assert infer_survey_bounds(state, np.array([0.0, 0.0]), cfg.map) is None
