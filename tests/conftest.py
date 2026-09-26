"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config, load_config
from canopy.contracts import MapState, Occ
from canopy.planning.safety import ClearanceMap, geofence_box

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
        return MapState(
            occ=occ, origin=_LOT[0].copy(), voxel=v, tri_seen=np.zeros(0, dtype=np.bool_)
        )

    return build


@pytest.fixture
def clearance(
    cfg: Config, make_map: Callable[..., MapState], lot: npt.NDArray[np.float64]
) -> Callable[..., ClearanceMap]:
    """Build a :class:`ClearanceMap` over a :func:`make_map` world, fenced to the lot."""

    def build(boxes: Sequence[Box] = (), unknown: Sequence[Box] = ()) -> ClearanceMap:
        return ClearanceMap.from_map(make_map(boxes, unknown), geofence_box(lot, cfg.safety))

    return build
