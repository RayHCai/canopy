"""Vectorized WGS84 geodesy: geodetic degrees, ECEF metres, and the world frame.

Every coordinate a provider returns is latitude and longitude in degrees;
every placement rule wants metres in the world frame. This module is the
whole bridge, in the same two stages as
:class:`~canopy.contracts.SiteSnapshot` itself:

* :func:`enu_from_geodetic` / :func:`geodetic_from_enu` project to and from a
  local east-north frame about an anchor (the address pin), by way of ECEF.
  That is exact 3D geometry, not a flat-Earth approximation, so it stays
  accurate to millimetres however far a neighbour or a street sits from the
  pin -- a linear approximation from the anchor's local radii of curvature
  drifts quadratically with distance and would not.
* :func:`world_from_enu` / :func:`enu_from_world` then apply the snapshot's
  own rigid transform: shift to the lot centre and turn the street to face
  ``-y``.

NumPy only, and every function is vectorized over its leading array
dimension, so projecting a whole OSM way's vertices is one call, not a loop.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from canopy.contracts import Points2

__all__ = [
    "WGS84_A_M",
    "WGS84_F",
    "enu_from_geodetic",
    "enu_from_world",
    "geodetic_from_enu",
    "world_from_enu",
]

#: WGS84 semi-major axis, metres.
WGS84_A_M = 6378137.0
#: WGS84 flattening.
WGS84_F = 1.0 / 298.257223563

#: Eccentricity squared: ``e^2 = f (2 - f)``.
_E2 = WGS84_F * (2.0 - WGS84_F)

#: Newton steps for ECEF -> geodetic latitude. WGS84's eccentricity converges
#: to sub-millimetre well within this many.
_LAT_ITERS = 5


def _geodetic_to_ecef(
    lat_deg: npt.NDArray[np.float64], lon_deg: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Points on the ellipsoid (height 0) to ECEF metres, shape ``(..., 3)``.

    Broadcasting ``lat_deg`` and ``lon_deg`` together first, rather than
    leaving it to the arithmetic below, is what lets a scalar anchor and an
    array of vertices share one code path: every one of ``x``, ``y`` and
    ``z`` ends up the same shape, which :func:`numpy.stack` requires.
    """
    lat_deg, lon_deg = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=np.float64), np.asarray(lon_deg, dtype=np.float64)
    )
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A_M / np.sqrt(1.0 - _E2 * sin_lat**2)
    x = n * cos_lat * np.cos(lon)
    y = n * cos_lat * np.sin(lon)
    z = n * (1.0 - _E2) * sin_lat
    return np.stack([x, y, z], axis=-1)


def _ecef_to_geodetic(
    ecef: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ECEF metres to geodetic degrees, by Newton iteration on latitude.

    Assumes every point is on the ellipsoid (height 0), true of anything
    built from :func:`enu_from_geodetic`'s own output with the up component
    dropped, which is the only input this module ever hands it.
    """
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - _E2))
    for _ in range(_LAT_ITERS):
        sin_lat = np.sin(lat)
        n = WGS84_A_M / np.sqrt(1.0 - _E2 * sin_lat**2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon)


def _enu_basis(
    anchor_lat_deg: float, anchor_lon_deg: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """East, north and up unit vectors, in ECEF, at the anchor."""
    lat, lon = math.radians(anchor_lat_deg), math.radians(anchor_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


def _rotation(angle_rad: float) -> npt.NDArray[np.float64]:
    """2D rotation matrix, anticlockwise by ``angle_rad``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def enu_from_geodetic(
    lat_deg: npt.NDArray[np.float64] | float,
    lon_deg: npt.NDArray[np.float64] | float,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
) -> Points2:
    """Project geodetic coordinates to local east-north metres about an anchor.

    Goes by way of ECEF: the true 3D displacement to each point, resolved
    onto the anchor's local east and north axes. The up component (curvature
    away from the tangent plane) is computed and dropped -- everything this
    simulates stands at ground level, so it never mattered in the first
    place -- which is the "dropping up" this function's docstring promises.

    Parameters
    ----------
    lat_deg, lon_deg
        Geodetic coordinates, scalar or shape ``(N,)``.
    anchor_lat_deg, anchor_lon_deg
        Origin of the local frame, e.g. the address pin.

    Returns
    -------
    Points2
        East and north metres from the anchor, shape ``(N, 2)`` (``N=1`` for
        scalar input).
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
    ecef = _geodetic_to_ecef(lat, lon)
    anchor_ecef = _geodetic_to_ecef(
        np.asarray(anchor_lat_deg, dtype=np.float64), np.asarray(anchor_lon_deg, dtype=np.float64)
    )
    east_hat, north_hat, _up_hat = _enu_basis(anchor_lat_deg, anchor_lon_deg)
    d = ecef - anchor_ecef
    return np.stack([d @ east_hat, d @ north_hat], axis=-1)


def geodetic_from_enu(
    enu: Points2, anchor_lat_deg: float, anchor_lon_deg: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Invert :func:`enu_from_geodetic`: local east-north metres to geodetic degrees.

    Reconstructs the ECEF point on the anchor's tangent plane (up = 0) and
    solves for its geodetic coordinates by Newton iteration on latitude --
    exact ellipsoid geometry, not a linear approximation from the anchor's
    local radii of curvature, so this stays accurate to millimetres at any
    range rather than only close to the anchor.

    Parameters
    ----------
    enu
        East and north metres from the anchor, shape ``(N, 2)`` (or ``(2,)``
        for one point).
    anchor_lat_deg, anchor_lon_deg
        The local frame's origin, as given to :func:`enu_from_geodetic`.

    Returns
    -------
    tuple[NDArray, NDArray]
        Latitude and longitude in degrees, one axis shorter than ``enu``.
    """
    arr = np.asarray(enu, dtype=np.float64)
    east_hat, north_hat, _up_hat = _enu_basis(anchor_lat_deg, anchor_lon_deg)
    anchor_ecef = _geodetic_to_ecef(
        np.asarray(anchor_lat_deg, dtype=np.float64), np.asarray(anchor_lon_deg, dtype=np.float64)
    )
    ecef = anchor_ecef + arr[..., 0:1] * east_hat + arr[..., 1:2] * north_hat
    return _ecef_to_geodetic(ecef)


def world_from_enu(enu: Points2, origin_enu: npt.NDArray[np.float64], north_rad: float) -> Points2:
    """Turn local east-north metres into the :class:`~canopy.contracts.SiteSnapshot` world frame.

    ``world = R (enu - origin)``, with ``R`` turning by ``north_rad - pi / 2``
    -- see :attr:`~canopy.contracts.SiteSnapshot.origin_enu_m`.
    """
    rot = _rotation(north_rad - math.pi / 2.0)
    return (np.asarray(enu, dtype=np.float64) - np.asarray(origin_enu, dtype=np.float64)) @ rot.T


def enu_from_world(
    world: Points2, origin_enu: npt.NDArray[np.float64], north_rad: float
) -> Points2:
    """Invert :func:`world_from_enu`."""
    rot = _rotation(north_rad - math.pi / 2.0)
    return np.asarray(world, dtype=np.float64) @ rot + np.asarray(origin_enu, dtype=np.float64)
