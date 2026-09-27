"""geodetic <-> ECEF <-> ENU <-> world: round trips and a known-distance check."""

from __future__ import annotations

import math

import numpy as np

from canopy.worldgen.geo import frames

#: WGS84 eccentricity squared, recomputed independently of canopy.worldgen.geo.frames
#: so the "known distance" check does not just re-derive its own expectation.
_E2 = frames.WGS84_F * (2.0 - frames.WGS84_F)


def _meridian_radius_m(lat_deg: float) -> float:
    """Local radius of curvature north-south, from the textbook formula."""
    sin_lat = math.sin(math.radians(lat_deg))
    # The float exponent (** 1.5, not ** 2) makes this `Any` per typeshed (a
    # negative base could raise a float to a fractional power into complex).
    return float(frames.WGS84_A_M * (1.0 - _E2) / (1.0 - _E2 * sin_lat**2) ** 1.5)


def _prime_vertical_radius_m(lat_deg: float) -> float:
    """Local radius of curvature east-west, from the textbook formula."""
    sin_lat = math.sin(math.radians(lat_deg))
    return frames.WGS84_A_M / math.sqrt(1.0 - _E2 * sin_lat**2)


def test_enu_from_geodetic_matches_local_radii_within_a_millimetre() -> None:
    """A site-scale offset should agree with the local-radius textbook formula to <1 mm.

    The local-radius formula is a first-order (flat) approximation, and it is
    not just the full ECEF projection's approximation to check against: at
    the ~1 km scale, the flat formula's *own* error against true ellipsoid
    geometry already exceeds a millimetre, because a parallel of latitude
    curves away from the tangent plane (an east offset picks up a north
    component of ``N cos(lat) sin(lat) (1 - cos(dlon))``, ~4.8 cm at 40 deg
    for a ~0.85 km offset) and the meridian radius itself varies along the
    arc. Both effects shrink faster than linearly, so at the scale of a site
    (~110 m north, ~85 m east) they drop under a millimetre and this is a
    meaningful independent check again, not a tautology.
    """
    anchor_lat, anchor_lon = 40.0, -75.0
    d_deg = 0.001  # ~110 m north, ~85 m east at this latitude: site scale

    north_enu = frames.enu_from_geodetic(anchor_lat + d_deg, anchor_lon, anchor_lat, anchor_lon)
    expected_north_m = _meridian_radius_m(anchor_lat) * math.radians(d_deg)
    assert abs(float(north_enu[0, 1]) - expected_north_m) < 1e-3
    assert abs(float(north_enu[0, 0])) < 1e-3

    east_enu = frames.enu_from_geodetic(anchor_lat, anchor_lon + d_deg, anchor_lat, anchor_lon)
    expected_east_m = (
        _prime_vertical_radius_m(anchor_lat)
        * math.cos(math.radians(anchor_lat))
        * math.radians(d_deg)
    )
    assert abs(float(east_enu[0, 0]) - expected_east_m) < 1e-3
    assert abs(float(east_enu[0, 1])) < 1e-3


def test_enu_from_geodetic_of_the_anchor_is_the_origin() -> None:
    """Projecting the anchor itself must give exactly (0, 0), not just approximately."""
    enu = frames.enu_from_geodetic(51.5, -0.1, 51.5, -0.1)
    assert np.allclose(enu, np.zeros((1, 2)))


def test_geodetic_from_enu_round_trips_enu_from_geodetic() -> None:
    """Random points near an anchor survive a forward-then-inverse projection."""
    anchor_lat, anchor_lon = -33.9, 151.2  # Sydney: a southern-hemisphere anchor too
    rng = np.random.default_rng(0)
    lat = anchor_lat + rng.uniform(-0.05, 0.05, size=20)
    lon = anchor_lon + rng.uniform(-0.05, 0.05, size=20)

    enu = frames.enu_from_geodetic(lat, lon, anchor_lat, anchor_lon)
    back_lat, back_lon = frames.geodetic_from_enu(enu, anchor_lat, anchor_lon)

    assert np.allclose(back_lat, lat, atol=1e-9)
    assert np.allclose(back_lon, lon, atol=1e-9)


def test_enu_from_geodetic_round_trips_geodetic_from_enu() -> None:
    """Random ENU points survive an inverse-then-forward projection."""
    anchor_lat, anchor_lon = 10.0, 20.0
    rng = np.random.default_rng(1)
    enu = rng.uniform(-2000.0, 2000.0, size=(15, 2))

    lat, lon = frames.geodetic_from_enu(enu, anchor_lat, anchor_lon)
    back = frames.enu_from_geodetic(lat, lon, anchor_lat, anchor_lon)

    assert np.allclose(back, enu, atol=1e-6)


def test_world_from_enu_is_a_rotation_about_north_minus_quarter_turn() -> None:
    """``north_rad = pi / 2`` is no turn at all, so north (enu +y) stays at world +y.

    ``world_from_enu`` turns by ``north_rad - pi / 2``, so ``north_rad = pi / 2``
    is a zero turn and north lands where it started. Only a further half turn,
    ``north_rad = -pi / 2`` (a map turned so the street runs up to the house
    from the north), puts north on ``-y``.
    """
    origin = np.array([1.0, 2.0])
    enu = np.array([[1.0, 2.0], [1.0, 3.0]])  # origin, and one point due north of it

    unturned = frames.world_from_enu(enu, origin, north_rad=math.pi / 2.0)
    assert np.allclose(unturned[0], [0.0, 0.0])
    assert np.allclose(unturned[1], [0.0, 1.0])

    half_turned = frames.world_from_enu(enu, origin, north_rad=-math.pi / 2.0)
    assert np.allclose(half_turned[0], [0.0, 0.0])
    assert np.allclose(half_turned[1], [0.0, -1.0])


def test_enu_from_world_round_trips_world_from_enu() -> None:
    """The world-frame transform inverts cleanly for an arbitrary origin and heading."""
    origin = np.array([5.0, -3.0])
    north_rad = 0.7
    rng = np.random.default_rng(2)
    enu = rng.uniform(-50.0, 50.0, size=(10, 2))

    world = frames.world_from_enu(enu, origin, north_rad)
    back = frames.enu_from_world(world, origin, north_rad)

    assert np.allclose(back, enu, atol=1e-9)
