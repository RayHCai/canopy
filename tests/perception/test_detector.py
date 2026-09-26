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
