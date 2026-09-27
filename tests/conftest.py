"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config, load_config
from canopy.contracts import (
    MapState,
    Occ,
    ResidentialDecision,
    ResidentialVerdict,
    ResolvedAddress,
    SiteBuilding,
    SiteSnapshot,
    SourceRecord,
)
from canopy.planning.safety import ClearanceMap, geofence_box
from canopy.worldgen.snapshot import finalize

#: ``(min_corner, max_corner)`` of an axis-aligned box, in metres.
Box = tuple[tuple[float, float, float], tuple[float, float, float]]

#: The spec's default 30 x 40 m lot, 12 m tall.
_LOT: npt.NDArray[np.float64] = np.array([[-15.0, -20.0, 0.0], [15.0, 20.0, 12.0]])


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Return the shipped default configuration, loaded once per session."""
    return load_config()


@pytest.fixture
def lot() -> npt.NDArray[np.float64]:
    """Lot bounds ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` used by :func:`make_map`."""
    return _LOT.copy()


@pytest.fixture
def make_map(cfg: Config) -> Callable[..., MapState]:
    """Build a fully observed :class:`MapState` over the :func:`lot`.

    Everything is FREE except the ground layer and the given ``boxes`` (OCC)
    and ``unknown`` boxes (UNKNOWN), so tests control exactly what the shield
    and planner see without needing a sensor or a mapper.
    """

    def build(boxes: Sequence[Box] = (), unknown: Sequence[Box] = ()) -> MapState:
        v = cfg.map.voxel_m
        shape = tuple(int(n) for n in np.round((_LOT[1] - _LOT[0]) / v))
        occ = np.full(shape, Occ.FREE, dtype=np.uint8)
        occ[:, :, 0] = Occ.OCC
        for value, group in ((Occ.OCC, boxes), (Occ.UNKNOWN, unknown)):
            for lo, hi in group:
                i0 = np.floor((np.asarray(lo) - _LOT[0]) / v).astype(int)
                i1 = np.ceil((np.asarray(hi) - _LOT[0]) / v).astype(int)
                occ[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]] = value
        return MapState(occ=occ, origin=_LOT[0].copy(), voxel=v)

    return build


@pytest.fixture
def clearance(
    cfg: Config, make_map: Callable[..., MapState], lot: npt.NDArray[np.float64]
) -> Callable[..., ClearanceMap]:
    """Build a :class:`ClearanceMap` over a :func:`make_map` world, fenced to the lot."""

    def build(boxes: Sequence[Box] = (), unknown: Sequence[Box] = ()) -> ClearanceMap:
        return ClearanceMap.from_map(make_map(boxes, unknown), geofence_box(lot, cfg.safety))

    return build


# ---------------------------------------------------------------------------
# Address-seeded sites
# ---------------------------------------------------------------------------
def _box(x0: float, y0: float, x1: float, y1: float) -> npt.NDArray[np.float64]:
    """Counter-clockwise rectangle outline, shape ``(4, 2)``."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _building(outline: npt.NDArray[np.float64], levels: int | None, source: str) -> SiteBuilding:
    return SiteBuilding(
        footprint=outline, levels=levels, height_m=None, roof_shape=None, source=source
    )


def make_site_snapshot(
    *,
    house: SiteBuilding | None = None,
    neighbours: Sequence[SiteBuilding] | None = None,
    trees: npt.NDArray[np.float64] | None = None,
    poles: npt.NDArray[np.float64] | None = None,
    lot_m: tuple[float, float] = (26.0, 36.0),
    decision: ResidentialDecision = ResidentialDecision.ACCEPT,
) -> SiteSnapshot:
    """Build a plausible address-built site, already in the world frame, with no network.

    The default is a two-storey L-shaped house on a 26 x 36 m lot: a 12 x 8 m
    main block with its front wall 11 m back from the front lot line, and a
    5 x 6 m wing behind its left end. Neighbours stand beside, behind and
    across the street; the street runs along ``-y`` 5.25 m beyond the front lot
    line, the way ``frontage_strip`` lays one out; one tree is mapped in the
    back yard and one utility pole just past the front right corner.
    """
    half_y = lot_m[1] / 2.0
    default_house = SiteBuilding(
        footprint=np.array(
            [[-6.0, -7.0], [6.0, -7.0], [6.0, 1.0], [-1.0, 1.0], [-1.0, 7.0], [-6.0, 7.0]]
        ),
        levels=2,
        height_m=None,
        roof_shape="gabled",
        source="fixture:way/1",
    )
    default_neighbours = (
        _building(_box(-27.0, -8.0, -17.0, 2.0), 1, "fixture:way/2"),
        _building(_box(17.0, -7.0, 28.0, 3.0), 2, "fixture:way/3"),
        _building(_box(-5.0, 22.0, 7.0, 31.0), None, "fixture:way/4"),
        _building(_box(-6.0, -42.0, 6.0, -33.0), 1, "fixture:way/5"),
    )
    centreline = -half_y - 5.25
    snapshot = SiteSnapshot(
        site_id="",
        address=ResolvedAddress(
            label="12 Test Street, Testville",
            provider="fixture",
            ref="fixture:way/1",
            lat_deg=40.0,
            lon_deg=-75.0,
        ),
        verdict=ResidentialVerdict(
            p_residential=0.95 if decision is ResidentialDecision.ACCEPT else 0.5,
            decision=decision,
            reasons=("the building is mapped as a house",),
        ),
        anchor_lat_deg=40.0,
        anchor_lon_deg=-75.0,
        origin_enu_m=np.zeros(2, dtype=np.float64),
        north_rad=float(np.pi / 2.0),
        lot_m=lot_m,
        house=house if house is not None else default_house,
        neighbours=tuple(neighbours) if neighbours is not None else default_neighbours,
        trees=trees if trees is not None else np.array([[8.0, 10.0, 2.5]], dtype=np.float64),
        poles=poles if poles is not None else np.array([[14.0, -19.0]], dtype=np.float64),
        street=np.array([[-75.0, centreline], [75.0, centreline]], dtype=np.float64),
        street_name="Test Street",
        street_width_m=7.5,
        aoi=np.array([[-75.0, -75.0], [75.0, 75.0]], dtype=np.float64),
        sources=(
            SourceRecord(
                provider="fixture",
                dataset="OpenStreetMap",
                licence="ODbL-1.0",
                attribution="(c) OpenStreetMap contributors",
                retrieved_at="2026-09-26T00:00:00Z",
            ),
        ),
    )
    return finalize(snapshot)


@pytest.fixture(scope="session")
def site_snapshot() -> SiteSnapshot:
    """Return the default :func:`make_site_snapshot` site, built once per session."""
    return make_site_snapshot()


@pytest.fixture(scope="session")
def make_site() -> Callable[..., SiteSnapshot]:
    """:func:`make_site_snapshot` itself, for tests that need a variant site."""
    return make_site_snapshot


# ---------------------------------------------------------------------------
# Network-marked tests: opt in, never on by default
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--run-network``, so a plain test run never talks to a live service."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run tests marked 'network' (talks to live public geodata services)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``network``-marked tests unless ``--run-network`` was given."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
