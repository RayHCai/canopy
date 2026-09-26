"""Tests for :class:`canopy.perception.ObjectDetector` on synthetic observations.

No simulator is involved: surfaces are sampled as coloured points and turned
into :class:`~canopy.contracts.Observation` s from a few drone positions, so
each test states exactly what the detector saw. There is no occlusion, which
the rules never rely on. The scene is a stretch of house wall (the plane
``y = 5``) carrying a meter and a conduit riser, a bush in front of it, and
look-alikes the detector must turn down.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from canopy.config import Config
from canopy.contracts import Cls, DiscoveredObject, Observation, Points
from canopy.perception import ObjectDetector

_LOT: npt.NDArray[np.float64] = np.array([[-15.0, -20.0, 0.0], [15.0, 20.0, 12.0]])

# The default palette (contracts.CLASS_COLORS) plus the look-alikes' colours.
_WALL = (225, 205, 170)
_METER = (230, 200, 30)
_CONDUIT = (170, 175, 185)
_BUSH = (40, 110, 40)
_PANEL = (190, 190, 195)
_STREET_LINE = (226, 184, 40)

_METER_LO = np.array([-0.15, 4.67, 1.25])
_METER_HI = np.array([0.15, 5.0, 1.96])
_CONDUIT_AXIS = np.array([0.5, 4.95])
_BUSH_CENTRE = np.array([-1.5, 4.0, 0.0])

#: Drone positions: around the house scene, and near the far look-alikes so
#: they are in range too.
_ORIGINS = np.array(
    [
        [0.0, 1.5, 1.5],
        [-2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [-1.0, 0.5, 3.0],
        [1.0, 3.0, 1.0],
        [-7.0, -8.0, 2.0],
        [-6.0, -4.0, 2.0],
    ]
)

Surface = tuple[Points, tuple[int, int, int]]


def _box(rng: np.random.Generator, lo: npt.ArrayLike, hi: npt.ArrayLike, n: int) -> Points:
    """``n`` points spread over the six faces of an axis-aligned box, by area."""
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    size = hi_a - lo_a
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]]).repeat(2)
    face = rng.choice(6, size=n, p=areas / areas.sum())
    pts = lo_a + rng.uniform(size=(n, 3)) * size
    axis = face // 2
    pts[np.arange(n), axis] = np.where(face % 2 == 0, lo_a[axis], hi_a[axis])
    return pts


def _dome(rng: np.random.Generator, centre: npt.ArrayLike, radii: npt.ArrayLike, n: int) -> Points:
    """``n`` points on the upper half of an ellipsoid standing on the ground: a bush."""
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts: Points = np.asarray(centre) + d * np.asarray(radii)
    return pts


def _pipe(
    rng: np.random.Generator, axis_xy: npt.ArrayLike, z: tuple[float, float], n: int
) -> Points:
    """``n`` points on a vertical pipe 6 cm across."""
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    axis = np.asarray(axis_xy, dtype=np.float64)
    return np.column_stack(
        [
            axis[0] + 0.03 * np.cos(theta),
            axis[1] + 0.03 * np.sin(theta),
            rng.uniform(z[0], z[1], n),
        ]
    )


def _house(rng: np.random.Generator) -> list[Surface]:
    """Return the wall with a meter and a conduit riser on it, and a bush 0.5 m in front."""
    wall = np.column_stack(
        [rng.uniform(-3.0, 3.0, 3000), np.full(3000, 5.0), rng.uniform(0.3, 3.0, 3000)]
    )
    return [
        (wall, _WALL),
        (_box(rng, _METER_LO, _METER_HI, 2500), _METER),
        (_pipe(rng, _CONDUIT_AXIS, (0.0, 1.5), 800), _CONDUIT),
        (_dome(rng, _BUSH_CENTRE, (0.5, 0.5, 0.6), 1500), _BUSH),
    ]


def _observe(detector: ObjectDetector, surfaces: list[Surface], seed: int, t0: float = 0.0) -> None:
    """Integrate one observation per drone position, with fresh shading and noise."""
    rng = np.random.default_rng(seed)
    pts = np.concatenate([p for p, _ in surfaces])
    base = np.concatenate(
        [np.tile(np.asarray(rgb, dtype=np.float64), (len(p), 1)) for p, rgb in surfaces]
    )
    for i, origin in enumerate(_ORIGINS):
        shade = rng.uniform(0.35, 1.0, (len(pts), 1))
        rgb = np.clip(np.rint(base * shade + rng.normal(0.0, 3.0, base.shape)), 0, 255)
        ray = pts - origin
        dist = np.linalg.norm(ray, axis=1)
        detector.integrate(
            Observation(
                drone_id=i % 3,
                t=t0 + 0.2 * i,
                origin=origin,
                dirs=ray / dist[:, None],
                dist=dist,
                rgb=rgb.astype(np.uint8),
            )
        )


def _of(objects: dict[int, DiscoveredObject], cls: Cls) -> list[DiscoveredObject]:
    return [o for o in objects.values() if o.cls is cls]


@pytest.fixture
def house_objects(cfg: Config) -> dict[int, DiscoveredObject]:
    """Return what the detector reports after seeing the house from every position."""
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, _house(np.random.default_rng(0)), seed=1)
    return detector.extract()


def test_meter_is_found_once_and_boxed_tightly(
    cfg: Config, house_objects: dict[int, DiscoveredObject]
) -> None:
    """The wall-mounted yellow box is one METER whose box hugs it, plus the outline margin."""
    (meter,) = _of(house_objects, Cls.METER)
    margin = 2.0 * cfg.perception.box_margin_m
    np.testing.assert_allclose(meter.box.center, (_METER_LO + _METER_HI) / 2.0, atol=0.03)
    true_size = np.sort(_METER_HI - _METER_LO)[::-1]
    np.testing.assert_allclose(np.sort(meter.box.size)[::-1], true_size + margin, atol=0.04)


def test_conduit_is_found_once_and_its_box_stands_on_the_ground(
    house_objects: dict[int, DiscoveredObject],
) -> None:
    """The riser is one thin CONDUIT whose box reaches from the ground to the pipe's top."""
    (conduit,) = _of(house_objects, Cls.CONDUIT)
    bottom = conduit.box.center[2] - conduit.box.size[2] / 2.0
    top = conduit.box.center[2] + conduit.box.size[2] / 2.0
    assert bottom == pytest.approx(0.0, abs=1e-9)
    assert top == pytest.approx(1.5, abs=0.06)
    np.testing.assert_allclose(conduit.box.center[:2], _CONDUIT_AXIS, atol=0.03)
    assert max(conduit.box.size[:2]) < 0.2


def test_bush_near_the_house_is_found_and_boxed(
    house_objects: dict[int, DiscoveredObject],
) -> None:
    """The dome in front of the wall is one BUSH, standing on the ground, about 1 m across."""
    (bush,) = _of(house_objects, Cls.BUSH)
    np.testing.assert_allclose(bush.box.center[:2], _BUSH_CENTRE[:2], atol=0.05)
    assert bush.box.center[2] - bush.box.size[2] / 2.0 == pytest.approx(0.0, abs=1e-9)
    assert bush.box.size[2] == pytest.approx(0.6, abs=0.1)
    np.testing.assert_allclose(bush.box.size[:2], 1.0, atol=0.12)


def test_context_class_is_never_reported(house_objects: dict[int, DiscoveredObject]) -> None:
    """WALL is ``report: false``: it anchors other rules but is never an object itself."""
    assert _of(house_objects, Cls.WALL) == []
    assert all(o.cls in (Cls.METER, Cls.CONDUIT, Cls.BUSH) for o in house_objects.values())


def test_look_alikes_are_turned_down(cfg: Config) -> None:
    """Right colour, wrong place or shape, or right shape, wrong colour: none are reported.

    A yellow street line lies flat on the ground; a yellow meter-sized box
    stands far from any wall; a thin neutral-grey strip on the wall has a
    conduit's shape but a breaker panel's colour; a bush stands far from the
    house; and a bush-green hedge rises above the bush band.
    """
    rng = np.random.default_rng(2)
    scene = [
        *_house(rng),
        (_box(rng, (2.0, 0.0, 0.0), (6.0, 0.12, 0.02), 1500), _STREET_LINE),
        (_box(rng, (-8.15, -5.15, 1.25), (-7.85, -4.85, 1.96), 2500), _METER),
        (_box(rng, (2.0, 4.9, 0.6), (2.05, 4.98, 1.6), 800), _PANEL),
        (_dome(rng, (-8.0, -12.0, 0.0), (0.5, 0.5, 0.6), 1500), _BUSH),
        (_box(rng, (-2.8, 4.3, 0.0), (-2.4, 4.7, 3.0), 2500), _BUSH),
    ]
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, scene, seed=3)
    objects = detector.extract()
    assert len(_of(objects, Cls.METER)) == 1
    assert len(_of(objects, Cls.CONDUIT)) == 1
    (bush,) = _of(objects, Cls.BUSH)
    np.testing.assert_allclose(bush.box.center[:2], _BUSH_CENTRE[:2], atol=0.05)


def test_track_ids_are_stable_and_new_objects_get_new_ones(cfg: Config) -> None:
    """Re-finding an object keeps its id; an object seen for the first time gets a fresh one."""
    rng = np.random.default_rng(4)
    house = _house(rng)
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, house, seed=5)
    first = detector.extract()
    ids = {o.cls: o.track_id for o in first.values()}

    second_bush = (_dome(rng, (2.0, 4.0, 0.0), (0.4, 0.4, 0.5), 1200), _BUSH)
    _observe(detector, [*house, second_bush], seed=6, t0=2.0)
    second = detector.extract()
    assert {o.cls: o.track_id for o in second.values() if o.track_id in first} == ids
    (new,) = [o for o in second.values() if o.track_id not in first]
    assert new.cls is Cls.BUSH
    assert new.track_id > max(first)


def test_confidence_follows_the_evidence(cfg: Config) -> None:
    """Confidence is ``1 - exp(-hits / confidence_hits)``: in [0, 1], never lowered by more hits."""
    house = _house(np.random.default_rng(7))
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, house, seed=8)
    before = detector.extract()
    _observe(detector, house, seed=9, t0=2.0)
    after = detector.extract()
    assert before.keys() == after.keys()
    for track_id, obj in after.items():
        expected = 1.0 - np.exp(-obj.n_hits / cfg.perception.confidence_hits)
        assert obj.confidence == pytest.approx(expected)
        assert obj.n_hits > before[track_id].n_hits
        assert before[track_id].confidence <= obj.confidence <= 1.0


def test_colourless_observations_change_nothing(cfg: Config) -> None:
    """Colourless observations add nothing, so extract returns the very same dict."""
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, _house(np.random.default_rng(10)), seed=11)
    objects = detector.extract()
    dirs = np.tile([0.0, 1.0, 0.0], (10, 1))
    detector.integrate(
        Observation(drone_id=0, t=5.0, origin=np.zeros(3), dirs=dirs, dist=np.full(10, 4.0))
    )
    assert detector.extract() is objects


def test_first_seen_records_the_earliest_evidence(cfg: Config) -> None:
    """An object's first sighting is the time and drone of its earliest evidence."""
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe(detector, _house(np.random.default_rng(12)), seed=13, t0=10.0)
    for obj in detector.extract().values():
        assert obj.first_seen_t == pytest.approx(10.0)
        assert obj.first_seen_by == 0


# ---------------------------------------------------------------------------
# DOOR, GARAGE_DOOR, WINDOW, GAS_METER, AC_UNIT and PANEL.
#
# A second wall (``y = 10``, clear of the first scene's evidence) so these
# tests do not depend on where the meter/conduit/bush fixture puts things.
# ---------------------------------------------------------------------------
_WALL_Y = 10.0
_DOOR_RGB = (110, 75, 45)
_OFFWHITE_RGB = (232, 230, 224)  # the procedural garage panel's display colour -- see below
_WINDOW_RGB = (120, 170, 210)
_GAS_METER_RGB = (200, 120, 30)
_SHED_RGB = (150, 125, 100)  # a look-alike hue to the door and the gas meter, at low saturation
_AC_UNIT_RGB = (180, 185, 190)

_ORIGINS_2 = np.array(
    [
        [0.0, 6.5, 1.5],
        [-3.0, 7.0, 2.0],
        [3.0, 7.0, 2.0],
        [-2.0, 5.5, 3.0],
        [2.0, 8.0, 1.0],
        [0.0, 7.0, 4.5],
    ]
)


def _wall2(rng: np.random.Generator, n: int = 4000, z_max: float = 5.5) -> Surface:
    """Return a second wall plane, tall enough for an upper-storey window."""
    return (
        np.column_stack(
            [rng.uniform(-6.0, 6.0, n), np.full(n, _WALL_Y), rng.uniform(0.3, z_max, n)]
        ),
        _WALL,
    )


def _observe2(detector: ObjectDetector, surfaces: list[Surface], seed: int) -> None:
    """Like :func:`_observe`, but from positions facing the second wall."""
    rng = np.random.default_rng(seed)
    pts = np.concatenate([p for p, _ in surfaces])
    base = np.concatenate(
        [np.tile(np.asarray(rgb, dtype=np.float64), (len(p), 1)) for p, rgb in surfaces]
    )
    for i, origin in enumerate(_ORIGINS_2):
        shade = rng.uniform(0.35, 1.0, (len(pts), 1))
        rgb = np.clip(np.rint(base * shade + rng.normal(0.0, 3.0, base.shape)), 0, 255)
        ray = pts - origin
        dist = np.linalg.norm(ray, axis=1)
        detector.integrate(
            Observation(
                drone_id=i % 3,
                t=0.2 * i,
                origin=origin,
                dirs=ray / dist[:, None],
                dist=dist,
                rgb=rgb.astype(np.uint8),
            )
        )


def test_door_and_garage_door_share_a_colour_and_are_told_apart_by_size(cfg: Config) -> None:
    """A 1.0 m door and a 2.8 m garage door, painted the same, land in the right class each.

    Authored and neighbour houses paint a garage door the same timber brown as
    the front door (``config/default.yaml``'s ``GARAGE_DOOR`` comment), so
    colour cannot separate them; only their span along the wall can.
    """
    rng = np.random.default_rng(20)
    door = (_box(rng, (-1.0, _WALL_Y - 0.04, 0.0), (0.0, _WALL_Y, 2.1), 2000), _DOOR_RGB)
    garage = (_box(rng, (2.0, _WALL_Y - 0.04, 0.0), (4.8, _WALL_Y, 2.2), 3000), _DOOR_RGB)
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe2(detector, [_wall2(rng), door, garage], seed=21)
    objects = detector.extract()
    (found_door,) = _of(objects, Cls.DOOR)
    (found_garage,) = _of(objects, Cls.GARAGE_DOOR)
    # The door's box is taller than it is wide; the garage door's is the other
    # way round, so the span is each box's *middle* extent, not its largest.
    door_span = float(np.sort(found_door.box.size)[1])
    garage_span = float(np.sort(found_garage.box.size)[-1])
    assert door_span < 1.5
    assert garage_span > 2.2


def test_an_offwhite_garage_sized_panel_is_not_a_garage_door(cfg: Config) -> None:
    """A garage-sized off-white panel is not a garage door, nor a door.

    Garage doors sense as DOOR's brown (the procedural panel's off-white is
    display-only, ADR 0014). Off-white sits between that brown and WALL's
    beige in hue and saturation, so a GARAGE_DOOR band widened to reach it
    would admit every wall; this pins the band away from it.
    """
    rng = np.random.default_rng(22)
    offwhite = (
        _box(rng, (2.0, _WALL_Y - 0.04, 0.0), (4.8, _WALL_Y, 2.2), 3000),
        _OFFWHITE_RGB,
    )
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe2(detector, [_wall2(rng), offwhite], seed=23)
    objects = detector.extract()
    assert _of(objects, Cls.GARAGE_DOOR) == []
    assert _of(objects, Cls.DOOR) == []


def test_window_ground_and_upper_floor_are_both_found(cfg: Config) -> None:
    """A sill-height window and one a storey above are both reported, not just the lower one.

    ``WINDOW`` has no ``bottom_m``: the siting rule is what cares about height,
    so perception must report every window regardless of storey.
    """
    rng = np.random.default_rng(24)
    ground = (_box(rng, (-1.0, _WALL_Y - 0.04, 0.9), (0.0, _WALL_Y, 2.1), 1500), _WINDOW_RGB)
    upper = (_box(rng, (-1.0, _WALL_Y - 0.04, 3.7), (0.0, _WALL_Y, 4.9), 1500), _WINDOW_RGB)
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe2(detector, [_wall2(rng), ground, upper], seed=25)
    windows = _of(detector.extract(), Cls.WINDOW)
    assert len(windows) == 2
    centres_z = sorted(float(w.box.center[2]) for w in windows)
    np.testing.assert_allclose(centres_z, [1.5, 4.3], atol=0.1)


def test_gas_meter_is_told_apart_from_meter_and_shed_beige(cfg: Config) -> None:
    """A second, orange meter is its own class; a shed-beige box its own size is turned down.

    The gas meter's hue (32 deg) sits between the door's and the shed's, so
    only its higher saturation (0.85 against the shed's 0.33) keeps the
    look-alike, sized and placed identically, out of every class.
    """
    rng = np.random.default_rng(26)
    meter = (_box(rng, (-4.15, _WALL_Y - 0.33, 1.25), (-3.85, _WALL_Y, 1.96), 2000), _METER)
    gas_meter = (
        _box(rng, (-0.23, _WALL_Y - 0.285, 0.0), (0.23, _WALL_Y, 0.8), 2000),
        _GAS_METER_RGB,
    )
    shed_lookalike = (
        _box(rng, (3.77, _WALL_Y - 0.285, 0.0), (4.23, _WALL_Y, 0.8), 2000),
        _SHED_RGB,
    )
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe2(detector, [_wall2(rng), meter, gas_meter, shed_lookalike], seed=27)
    objects = detector.extract()
    assert len(_of(objects, Cls.METER)) == 1
    assert len(_of(objects, Cls.GAS_METER)) == 1
    assert {o.cls for o in objects.values()} == {Cls.METER, Cls.GAS_METER}


def test_ac_unit_panel_and_conduit_are_told_apart_by_geometry(cfg: Config) -> None:
    """Three near-identical greys land in the right class by shape, not by colour alone.

    AC_UNIT is a chunky box standing on the ground, set back from the wall;
    PANEL is a thin wall box at chest height near a meter; CONDUIT is a thin,
    tall pipe. All three sit close enough on the wall that only their shape
    and ``near_cls`` rules -- not their nearly-identical saturation -- tell
    them apart.
    """
    rng = np.random.default_rng(28)
    meter = (_box(rng, (-3.15, _WALL_Y - 0.33, 1.25), (-2.85, _WALL_Y, 1.96), 1500), _METER)
    panel = (_box(rng, (-2.45, _WALL_Y - 0.175, 0.72), (-1.95, _WALL_Y, 1.881), 2000), _PANEL)
    ac_unit = (
        _box(rng, (1.0, _WALL_Y - 1.5, 0.0), (2.0, _WALL_Y - 0.5, 0.9), 2500),
        _AC_UNIT_RGB,
    )
    conduit = (_pipe(rng, (3.5, _WALL_Y - 0.03), (0.0, 1.5), 1000), _CONDUIT)
    detector = ObjectDetector(cfg.perception, _LOT)
    _observe2(detector, [_wall2(rng), meter, panel, ac_unit, conduit], seed=29)
    objects = detector.extract()
    (found_ac,) = _of(objects, Cls.AC_UNIT)
    (found_panel,) = _of(objects, Cls.PANEL)
    (found_conduit,) = _of(objects, Cls.CONDUIT)
    np.testing.assert_allclose(found_ac.box.center[:2], (1.5, _WALL_Y - 1.0), atol=0.1)
    np.testing.assert_allclose(found_panel.box.center[:2], (-2.2, _WALL_Y - 0.0875), atol=0.1)
    np.testing.assert_allclose(found_conduit.box.center[:2], (3.5, _WALL_Y - 0.03), atol=0.1)
