"""Tests for canopy.worldgen.evidence.

Turning a mapped footprint into the blocks, storey count and roof a house rule
can build from, and the checks that refuse a footprint the model library
cannot represent.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Polygon

from canopy.config import Config
from canopy.contracts import SiteBuilding, SiteSnapshot
from canopy.errors import SiteRejectedError
from canopy.worldgen import placement as rules
from canopy.worldgen.evidence import (
    check_buildable,
    dominant_orientation,
    levels_for,
    regularize_footprint,
    roof_kind,
    site_evidence,
)

#: A plain 12 x 8 m rectangle, centred on the origin.
_RECT: npt.NDArray[np.float64] = np.array([[-6.0, -4.0], [6.0, -4.0], [6.0, 4.0], [-6.0, 4.0]])

#: The fixture site's own L-shaped footprint (see tests/conftest.py:make_site_snapshot):
#: a 12 x 8 m main block plus a 5 x 6 m wing on its back-left corner.
_L_SHAPE: npt.NDArray[np.float64] = np.array(
    [[-6.0, -7.0], [6.0, -7.0], [6.0, 1.0], [-1.0, 1.0], [-1.0, 7.0], [-6.0, 7.0]]
)

#: A 12 x 4 m top bar over a 4 x 3 m stem: unlike an L, both of the stem's
#: sides recede from the bar's footprint.
_T_SHAPE: npt.NDArray[np.float64] = np.array(
    [
        [-2.0, 0.0],
        [2.0, 0.0],
        [2.0, 3.0],
        [6.0, 3.0],
        [6.0, 7.0],
        [-6.0, 7.0],
        [-6.0, 3.0],
        [-2.0, 3.0],
    ]
)

#: The rectangle with a small bay -- 0.8 m deep, narrower than min_mass_m --
#: bumped out of its +x wall.
_BAY_RECTANGLE: npt.NDArray[np.float64] = np.array(
    [
        [-6.0, -4.0],
        [6.0, -4.0],
        [6.0, -1.0],
        [6.8, -1.0],
        [6.8, 1.0],
        [6.0, 1.0],
        [6.0, 4.0],
        [-6.0, 4.0],
    ]
)


def _rotate(points: npt.NDArray[np.float64], angle_rad: float) -> npt.NDArray[np.float64]:
    """Turn ``points`` anticlockwise by ``angle_rad`` about the origin."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rot.T


def _comb(
    base_h: float, tooth_w: float, tooth_h: float, n: int, gap: float
) -> npt.NDArray[np.float64]:
    """Build a base rectangle with ``n`` teeth narrower than any fittable block.

    Each tooth is a deliberate irregularity: too narrow to register as its own
    mass, so it can only ever be lost area, not a block the fit takes credit
    for.
    """
    total = n * tooth_w + (n + 1) * gap
    half = total / 2.0
    pts: list[tuple[float, float]] = [(-half, 0.0)]
    x = -half
    for _ in range(n):
        x += gap
        pts.append((x, 0.0))
        pts.append((x, tooth_h))
        pts.append((x + tooth_w, tooth_h))
        pts.append((x + tooth_w, 0.0))
        x += tooth_w
    pts.append((half, 0.0))
    pts.append((half, -base_h))
    pts.append((-half, -base_h))
    return np.array(pts, dtype=np.float64)


#: 9 x 3 m base plus four 1 x 8 m teeth: about 46% of the polygon's area falls
#: outside the one block that fits, so the plan reads as too irregular.
_COMB_SHAPE: npt.NDArray[np.float64] = _comb(base_h=3.0, tooth_w=1.0, tooth_h=8.0, n=4, gap=1.0)


def _building(
    x: float,
    y: float,
    *,
    levels: int | None = None,
    height_m: float | None = None,
    roof_shape: str | None = None,
) -> SiteBuilding:
    """Build a plain ``x`` by ``y`` rectangle, centred on the origin, for a check_buildable case."""
    return SiteBuilding(
        footprint=np.array([[-x / 2, -y / 2], [x / 2, -y / 2], [x / 2, y / 2], [-x / 2, y / 2]]),
        levels=levels,
        height_m=height_m,
        roof_shape=roof_shape,
        source="fixture:test",
    )


# ---------------------------------------------------------------------------
# dominant_orientation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle_deg", [0.0, 15.0, -20.0, 44.0, 46.0, 80.0, -80.0])
def test_dominant_orientation_recovers_a_rectangles_rotation_mod_90(angle_deg: float) -> None:
    rotated = _rotate(_RECT, math.radians(angle_deg))
    expected = math.radians(((angle_deg + 45.0) % 90.0) - 45.0)
    assert dominant_orientation(rotated) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("angle_deg", [0.0, 23.0, -37.0, 60.0])
def test_dominant_orientation_recovers_an_l_shapes_rotation_mod_90(angle_deg: float) -> None:
    rotated = _rotate(_L_SHAPE, math.radians(angle_deg))
    expected = math.radians(((angle_deg + 45.0) % 90.0) - 45.0)
    assert dominant_orientation(rotated) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# regularize_footprint
# ---------------------------------------------------------------------------
def test_regularize_footprint_fits_a_rectangle_with_one_exact_block(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    fit = regularize_footprint(
        _RECT,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    assert len(fit.blocks) == 1
    assert np.allclose(fit.blocks[0].size[:2], [12.0, 8.0], atol=1e-6)
    assert np.allclose(fit.blocks[0].centre, [0.0, 0.0], atol=1e-6)
    assert fit.iou == pytest.approx(1.0, abs=1e-6)


def test_regularize_footprint_fits_an_l_shape_with_two_blocks(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    fit = regularize_footprint(
        _L_SHAPE,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    assert len(fit.blocks) == 2
    # Largest first: the 12 x 8 m main block, then the 5 x 6 m wing.
    assert tuple(float(v) for v in fit.blocks[0].size[:2]) == pytest.approx((12.0, 8.0))
    assert tuple(float(v) for v in fit.blocks[1].size[:2]) == pytest.approx((5.0, 6.0))
    assert fit.iou == pytest.approx(1.0, abs=1e-6)


def test_regularize_footprint_fits_a_t_shape_with_two_blocks(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    fit = regularize_footprint(
        _T_SHAPE,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    assert len(fit.blocks) == 2
    assert tuple(float(v) for v in fit.blocks[0].size[:2]) == pytest.approx((12.0, 4.0))
    assert tuple(float(v) for v in fit.blocks[1].size[:2]) == pytest.approx((4.0, 3.0))
    assert fit.iou == pytest.approx(1.0, abs=1e-6)


def test_regularize_footprint_drops_a_bay_narrower_than_min_mass(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    fit = regularize_footprint(
        _BAY_RECTANGLE,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    assert len(fit.blocks) == 1
    assert np.allclose(fit.blocks[0].size[:2], [12.0, 8.0], atol=1e-6)
    assert fit.iou > 0.9


def test_regularize_footprint_after_undoing_a_rotation_matches_the_unrotated_fit(
    cfg: Config,
) -> None:
    site_cfg = cfg.worldgen.site
    rotated = _rotate(_L_SHAPE, math.radians(23.0))
    aligned = _rotate(rotated, -dominant_orientation(rotated))
    fit = regularize_footprint(
        aligned,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    assert len(fit.blocks) == 2
    assert fit.iou > 0.99


def test_regularize_footprint_blocks_union_into_one_connected_house_frame(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    fit = regularize_footprint(
        _L_SHAPE,
        raster_m=site_cfg.raster_m,
        max_masses=site_cfg.max_masses,
        min_mass_m=site_cfg.min_mass_m,
    )
    frame = rules.house_frame(fit.blocks)  # must not raise WorldgenError
    assert Polygon(frame.footprint).area == pytest.approx(126.0, abs=0.5)


# ---------------------------------------------------------------------------
# check_buildable
# ---------------------------------------------------------------------------
def test_check_buildable_rejects_an_oversized_footprint(cfg: Config) -> None:
    building = _building(25.0, 20.0, levels=1)  # 500 m^2, over max_footprint_m2 (400)
    with pytest.raises(SiteRejectedError, match="max_footprint_m2"):
        check_buildable(building, cfg.worldgen.site)


def test_check_buildable_rejects_too_many_levels(cfg: Config) -> None:
    building = _building(10.0, 8.0, levels=3)  # over max_levels (2)
    with pytest.raises(SiteRejectedError, match="max_levels"):
        check_buildable(building, cfg.worldgen.site)


def test_check_buildable_rejects_an_irregular_outline(cfg: Config) -> None:
    building = SiteBuilding(
        footprint=_COMB_SHAPE, levels=1, height_m=None, roof_shape=None, source="fixture:test"
    )
    with pytest.raises(SiteRejectedError, match="min_fit_iou"):
        check_buildable(building, cfg.worldgen.site)


# ---------------------------------------------------------------------------
# roof_kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("roof_shape", "expected_kind", "note_substring"),
    [
        ("gabled", "gable", None),
        ("flat", "flat", None),
        ("hipped", "gable", "hipped"),
        (None, None, None),
    ],
)
def test_roof_kind_maps_mapped_shapes_onto_buildable_roofs(
    roof_shape: str | None, expected_kind: str | None, note_substring: str | None
) -> None:
    kind, note = roof_kind(roof_shape)
    assert kind == expected_kind
    if note_substring is None:
        assert note is None
    else:
        assert note is not None
        assert note_substring in note


# ---------------------------------------------------------------------------
# levels_for
# ---------------------------------------------------------------------------
def test_levels_for_prefers_the_mapped_level_count(cfg: Config) -> None:
    building = _building(10.0, 8.0, levels=2, height_m=999.0)  # height must be ignored
    assert levels_for(building, cfg.worldgen.site) == 2


def test_levels_for_counts_storeys_from_a_mapped_height(cfg: Config) -> None:
    site_cfg = cfg.worldgen.site
    height = 2 * site_cfg.storey_m + site_cfg.roof_allowance_m  # exactly two storeys plus the roof
    building = _building(10.0, 8.0, height_m=height)
    assert levels_for(building, site_cfg) == 2


def test_levels_for_is_none_without_levels_or_height(cfg: Config) -> None:
    building = _building(10.0, 8.0)
    assert levels_for(building, cfg.worldgen.site) is None


# ---------------------------------------------------------------------------
# site_evidence and without_party_walls
# ---------------------------------------------------------------------------
def test_site_evidence_reads_the_default_site(cfg: Config, site_snapshot: SiteSnapshot) -> None:
    evidence = site_evidence(site_snapshot, cfg.worldgen.site)
    assert len(evidence.blocks) == 2
    assert evidence.levels == 2
    assert evidence.roof == "gable"
    assert evidence.fit_iou == pytest.approx(1.0, abs=1e-6)
    assert evidence.notes == ()


def test_without_party_walls_drops_only_the_wall_a_neighbour_covers(
    cfg: Config, make_site: object
) -> None:
    # A neighbour flush against the main block's +x wall (x=6, y in [-7, 1]),
    # covering it well past the half of it the party-wall test requires.
    neighbour = SiteBuilding(
        footprint=np.array([[6.0, -8.0], [14.0, -8.0], [14.0, 2.0], [6.0, 2.0]]),
        levels=1,
        height_m=None,
        roof_shape=None,
        source="fixture:neighbour",
    )
    snapshot = make_site(neighbours=(neighbour,))  # type: ignore[operator]
    evidence = site_evidence(snapshot, cfg.worldgen.site)
    frame = rules.house_frame(evidence.blocks)
    trimmed = evidence.without_party_walls(frame, cfg.worldgen.site.party_wall_gap_m)

    def is_shared_wall(wall: rules.WallSegment) -> bool:
        return (
            abs(float(wall.a[0]) - 6.0) < 1e-6
            and abs(float(wall.b[0]) - 6.0) < 1e-6
            and float(wall.normal[0]) > 0.0
        )

    assert sum(is_shared_wall(w) for w in frame.walls) == 1
    assert not any(is_shared_wall(w) for w in trimmed.walls)
    assert len(trimmed.walls) == len(frame.walls) - 1
    kept = {(tuple(w.a), tuple(w.b)) for w in trimmed.walls}
    expected = {(tuple(w.a), tuple(w.b)) for w in frame.walls if not is_shared_wall(w)}
    assert kept == expected
