"""Tracing house walls from occupancy (:func:`canopy.site.trace_house`).

Every synthetic house here is built as a handful of axis-aligned wall boxes
via the ``make_map`` fixture, 3 m tall so they fill the default
``outline.band_z_m`` = [1.5, 2.8] m band the shipped rules.yaml uses.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from canopy.contracts import MapState, Vec3
from canopy.errors import SiteError
from canopy.site import HouseOutline, OutlineSpec, load_site_rules, trace_house

Box = tuple[tuple[float, float, float], tuple[float, float, float]]

_SPEC: OutlineSpec = load_site_rules().outline

#: A 10 x 8 m rectangular ring, centred at the origin, walls 0.25 m thick.
_RECT_WALLS: list[Box] = [
    ((-5.0, 3.75, 0.0), (5.0, 4.0, 3.0)),  # north
    ((-5.0, -4.0, 0.0), (5.0, -3.75, 3.0)),  # south
    ((4.75, -4.0, 0.0), (5.0, 4.0, 3.0)),  # east
    ((-5.0, -4.0, 0.0), (-4.75, 4.0, 3.0)),  # west
]

#: An L-shaped footprint (notch cut from the north-east corner), same thickness.
_L_WALLS: list[Box] = [
    ((-5.0, -4.0, 0.0), (5.0, -3.75, 3.0)),  # south
    ((4.75, -4.0, 0.0), (5.0, 1.0, 3.0)),  # east (lower)
    ((1.0, 0.75, 0.0), (5.0, 1.0, 3.0)),  # notch bottom
    ((1.0, 1.0, 0.0), (1.25, 4.0, 3.0)),  # notch left
    ((-5.0, 3.75, 0.0), (1.0, 4.0, 3.0)),  # north
    ((-5.0, -4.0, 0.0), (-4.75, 4.0, 3.0)),  # west
]

_METER_NEAR: Vec3 = np.array([5.15, 0.0, 1.5])


def _outline(
    make_map: Callable[..., MapState], walls: list[Box], near: Vec3 | None = None
) -> HouseOutline:
    return trace_house(make_map(boxes=walls), _METER_NEAR if near is None else near, _SPEC)


def test_rectangle_ring_traces_four_walls_anticlockwise_and_outward(
    make_map: Callable[..., MapState],
) -> None:
    outline = _outline(make_map, _RECT_WALLS)
    assert len(outline.walls) == 4

    x, y = outline.polygon[:, 0], outline.polygon[:, 1]
    signed_area = np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    assert signed_area > 0.0, "polygon must wind anticlockwise"

    centre = outline.polygon.mean(axis=0)
    lengths = sorted(w.length for w in outline.walls)
    assert lengths[0] == pytest.approx(8.0, abs=0.3)
    assert lengths[0] == pytest.approx(lengths[1], abs=0.3)
    assert lengths[2] == pytest.approx(10.0, abs=0.3)
    assert lengths[2] == pytest.approx(lengths[3], abs=0.3)

    for wall in outline.walls:
        mid = (wall.start + wall.end) / 2.0
        assert np.dot(mid - centre, wall.normal) > 0.0, "normal must point away from the house"
        # Every wall midpoint must sit near its true outer face (x = +-5 or y = +-4).
        gap = min(
            abs(abs(mid[0]) - 5.0) if abs(wall.normal[0]) > 0.5 else np.inf,
            abs(abs(mid[1]) - 4.0) if abs(wall.normal[1]) > 0.5 else np.inf,
        )
        assert gap < 0.2


def test_l_shaped_house_traces_six_walls(make_map: Callable[..., MapState]) -> None:
    outline = _outline(make_map, _L_WALLS)
    assert len(outline.walls) == 6


def test_doorway_gap_is_bridged(make_map: Callable[..., MapState]) -> None:
    # A 0.5 m hole in the south wall, well under 2 * outline.close_m = 1.5 m.
    gapped = [
        ((-5.0, -4.0, 0.0), (-0.25, -3.75, 3.0)),
        ((0.25, -4.0, 0.0), (5.0, -3.75, 3.0)),
        _RECT_WALLS[0],  # north
        *_RECT_WALLS[2:],  # east, west
    ]
    outline = _outline(make_map, gapped)
    assert len(outline.walls) == 4
    south = min(outline.walls, key=lambda w: w.normal[1])
    assert south.length == pytest.approx(10.0, abs=0.3)


def test_bushes_against_the_wall_do_not_change_the_outline(
    make_map: Callable[..., MapState],
) -> None:
    # 1.2 m bushes, well below outline.band_z_m[0] = 1.5 m, sitting just outside
    # the east wall.
    bushes: list[Box] = [
        ((5.0, -1.0, 0.0), (5.4, -0.5, 1.2)),
        ((5.0, 0.5, 0.0), (5.4, 1.0, 1.2)),
    ]
    baseline = _outline(make_map, _RECT_WALLS)
    with_bushes = _outline(make_map, [*_RECT_WALLS, *bushes])
    assert len(with_bushes.walls) == len(baseline.walls)
    assert np.allclose(with_bushes.polygon, baseline.polygon, atol=1e-9)


def test_a_second_building_far_away_is_ignored(make_map: Callable[..., MapState]) -> None:
    far_house: list[Box] = [
        ((x0 + 20.0, y0 - 10.0, z0), (x1 + 20.0, y1 - 10.0, z1))
        for (x0, y0, z0), (x1, y1, z1) in _RECT_WALLS
    ]
    outline = _outline(make_map, [*_RECT_WALLS, *far_house])
    assert len(outline.walls) == 4
    assert np.all(np.abs(outline.polygon[:, 0]) < 6.0)
    assert np.all(np.abs(outline.polygon[:, 1]) < 5.0)


def test_no_wall_near_the_given_point_raises_site_error(
    make_map: Callable[..., MapState],
) -> None:
    # The house centre is roughly 5 m from every wall, well beyond max_meter_gap_m.
    with pytest.raises(SiteError):
        _outline(make_map, _RECT_WALLS, near=np.array([0.0, 0.0, 1.5]))


def test_map_with_no_walls_at_all_raises_site_error(make_map: Callable[..., MapState]) -> None:
    with pytest.raises(SiteError):
        _outline(make_map, [], near=_METER_NEAR)
