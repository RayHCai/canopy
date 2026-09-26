"""Shared data contracts.

Every piece of data that crosses a module boundary is defined here. Field names
are load-bearing: renaming one means updating every consumer, so treat this
module as an interface, not an implementation detail.

Units are metres, seconds and radians. The world frame is Z-up with the origin
at lot centre and the ground plane at ``z = 0``.

Notes
-----
The array-bearing dataclasses are declared ``eq=False``. A generated
``__eq__`` would compare NumPy arrays with ``==``, producing an elementwise
array that raises ``ValueError`` the moment anything truth-tests the result.
Compare fields explicitly instead (``np.allclose`` and friends).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "CLASS_COLORS",
    "Cls",
    "DiscoveredObject",
    "DroneState",
    "DroneTask",
    "MapState",
    "Observation",
    "Occ",
    "OrientedBox",
    "Photo",
    "Points",
    "Rgb",
    "Scan",
    "SceneGeometry",
    "SceneManifest",
    "SceneObject",
    "SiteAssessment",
    "SiteCandidate",
    "SiteResult",
    "SiteVerdict",
    "TaskKind",
    "Vec3",
]

#: A single 3-vector, shape ``(3,)``.
Vec3: TypeAlias = npt.NDArray[np.float64]
#: A stack of 3-vectors, shape ``(N, 3)``.
Points: TypeAlias = npt.NDArray[np.float64]
#: 8-bit RGB triple.
Rgb: TypeAlias = tuple[int, int, int]


class Cls(IntEnum):
    """Semantic class of a scene object.

    Values are stable and are written into the scene manifest, so they may be
    appended to but never reordered.
    """

    GROUND = 0
    DRIVEWAY = 1
    WALL = 2
    ROOF = 3
    DOOR = 4
    WINDOW = 5
    GARAGE_DOOR = 6
    METER = 7
    BUSH = 8
    TREE = 9
    FENCE = 10
    AC_UNIT = 11
    GAS_METER = 12
    PANEL = 13
    """Exterior load centre: the breaker panel the service conduit feeds."""
    CONDUIT = 14
    """Rigid conduit and fittings joining the meter to the panel or the wall."""
    SHED = 15
    """Detached outbuilding. A flight obstacle, and never a battery site."""


class Occ(IntEnum):
    """Occupancy state of a voxel in :attr:`MapState.occ`."""

    UNKNOWN = 0
    FREE = 1
    OCC = 2


class SiteVerdict(StrEnum):
    """The photo-review outcome for a battery site, or for the survey overall.

    Three-valued because a real reviewer never silently defaults to approval:
    a site (or a whole survey) is either clean, needs a human look, or is a
    non-starter. ``str`` valued so it serializes straight into the viewer's
    JSON payload without a translation table.
    """

    PASS = "pass"
    MANUAL_REVIEW = "manual_review"
    REJECT = "reject"


#: True colours for the grayscale-to-colour reveal and for photo shading.
#:
#: This is a contract, not a tunable: the mapper, the viewer and the photo
#: renderer must agree pixel for pixel, so it lives beside the class enum
#: rather than in ``config/default.yaml``.
CLASS_COLORS: dict[Cls, Rgb] = {
    Cls.GROUND: (96, 140, 70),
    Cls.DRIVEWAY: (150, 150, 150),
    Cls.WALL: (225, 205, 170),
    Cls.ROOF: (120, 60, 50),
    Cls.DOOR: (110, 75, 45),
    Cls.GARAGE_DOOR: (110, 75, 45),
    Cls.WINDOW: (120, 170, 210),
    Cls.METER: (230, 200, 30),
    Cls.GAS_METER: (200, 120, 30),
    Cls.AC_UNIT: (180, 185, 190),
    Cls.BUSH: (40, 110, 40),
    Cls.TREE: (70, 130, 50),
    Cls.FENCE: (160, 130, 100),
    Cls.PANEL: (190, 190, 195),
    Cls.CONDUIT: (170, 175, 185),
    Cls.SHED: (150, 125, 100),
}

#: Trunk colour for :attr:`Cls.TREE`; the crown uses ``CLASS_COLORS[Cls.TREE]``.
TREE_TRUNK_COLOR: Rgb = (100, 70, 40)

#: Valid values for :attr:`DroneTask.kind`.
TaskKind: TypeAlias = str


@dataclass(slots=True, eq=False)
class SceneObject:
    """One mesh in the generated property, with its pose already baked in."""

    obj_id: int
    """Unique, and equal to the Open3D geometry id, i.e. manifest order."""
    cls: Cls
    mesh_path: str
    """OBJ file in world coordinates."""
    color: Rgb
    """True RGB, used for the reveal and for photos."""
    wall_normal: Vec3 | None = None
    """Outward wall normal. Set for meter, door and window; ``None`` otherwise."""
    asset_id: str = ""
    """Model-library entry this mesh was built from, e.g. ``"bush_round"``.

    The pose is already baked into ``mesh_path``, so nothing downstream *needs*
    this. It is carried anyway because it is the only record of which model the
    generator chose: without it a surprising scene cannot be traced back to the
    library entry that produced it. Empty for objects built without a library.
    """
    background: bool = False
    """Scenery outside the surveyed lot: the neighbours' houses and lawns.

    Background is real geometry -- it is in the raycasting scene, so rays hit it
    and a drone flying the lot edge sees it -- but it is never the swarm's job.
    The mapper does not count it toward coverage and the viewer draws it in true
    colour from the start. Keeping it in the manifest rather than inventing it in
    the viewer is what keeps the manifest the single source of truth for sensing
    and visualisation.
    """


@dataclass(slots=True, eq=False)
class SceneGeometry:
    """Every scene triangle, loaded once from the manifest's OBJ files.

    Triangles are indexed two ways. Per object, ``faces[k]`` indexes into
    ``vertices[k]``, which is what a renderer wants. Globally, the triangles of
    all objects are concatenated in manifest order, which is what
    :attr:`Scan.tri_ids` and :attr:`MapState.tri_seen` use; object ``k`` owns
    global ids ``obj_tri_offset[k]`` up to ``obj_tri_offset[k + 1]``.

    The per-triangle arrays exist for coverage accounting (area-weighted, with a
    photo-quality incidence rule) and are ground truth. The planner never reads
    them.
    """

    vertices: list[Points]
    """Per object, world-space vertices, shape ``(V_k, 3)``."""
    faces: list[npt.NDArray[np.int32]]
    """Per object, vertex indices into ``vertices[k]``, shape ``(F_k, 3)``."""
    obj_tri_offset: npt.NDArray[np.int64]
    """First global triangle id of each object, shape ``(n_objects + 1,)``; the
    last entry is the total triangle count."""
    tri_obj: npt.NDArray[np.int32]
    """Owning ``obj_id`` of each global triangle, shape ``(T,)``."""
    tri_normal: Points
    """Unit face normal of each global triangle, shape ``(T, 3)``. Winding in
    authored art is not consistent, so treat the sign as arbitrary."""
    tri_area: npt.NDArray[np.float64]
    """Area of each global triangle in square metres, shape ``(T,)``."""
    tri_centroid: Points
    """Centroid of each global triangle, shape ``(T, 3)``."""
    tri_exterior: npt.NDArray[np.bool_] | None = None
    """Whether each triangle can be seen from outside at all, shape ``(T,)``.

    False for faces sealed inside other geometry, such as a house shell's
    inner wall faces and partitions (most of its WALL area). Coverage counts
    only exterior faces, since a metric that includes surfaces no drone can
    reach never completes. ``None`` means every face counts, which is what
    hand-built test geometry wants.
    """
    obj_color: npt.NDArray[np.uint8] | None = None
    """True RGB of each object, shape ``(n_objects, 3)``, copied from the manifest.

    The sensor shades this into the colour a drone sees. Like the rest of this
    type it is ground truth: the renderer reads it, perception never does.
    ``None`` for hand-built test geometry, which the sensor paints mid-grey.
    """

    @property
    def n_triangles(self) -> int:
        """Total triangle count across every object."""
        return int(self.obj_tri_offset[-1])


@dataclass(slots=True, eq=False)
class SceneManifest:
    """The single source of truth for sensing and visualisation."""

    seed: int
    lot_bounds: npt.NDArray[np.float64]
    """``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``, shape ``(2, 3)``."""
    footprint: list[tuple[float, float]]
    """House outline polygon, counter-clockwise."""
    objects: list[SceneObject]
    home: Vec3
    """Launch pad position."""
    gt_meter_id: int
    """Ground truth, for evaluation only. Never read by the planner."""


@dataclass(slots=True, eq=False)
class DroneState:
    """Pose and health of one drone at one instant."""

    drone_id: int
    pos: Vec3
    vel: Vec3
    yaw: float
    alive: bool = True
    battery: float = 1.0
    """Remaining charge in ``[0, 1]``; drains linearly while flying."""


@dataclass(frozen=True, slots=True, eq=False)
class Observation:
    """What a drone's ranger senses: range and colour, never identities.

    This is the information boundary between the simulator and the algorithm.
    The sim knows which object every ray hit, and :class:`Scan` carries those
    ids for the coverage metric and the reveal. Perception is handed only
    this, so a classifier cannot read the label it is meant to infer.
    """

    drone_id: int
    t: float
    origin: Vec3
    """Where the sensor was: the drone's own position, which it always knows."""
    dirs: Points
    """Unit ray directions, shape ``(N, 3)``."""
    dist: npt.NDArray[np.float64]
    """Hit distances, shape ``(N,)``; ``inf`` on miss."""
    rgb: npt.NDArray[np.uint8] | None = None
    """Shaded colour of each hit, shape ``(N, 3)``; the sky colour on a miss.

    ``None`` when the sensor has no colour channel (hand-built test scans),
    which leaves nothing to classify.
    """


@dataclass(slots=True, eq=False)
class Scan:
    """One 360-degree ray cast. Misses carry ``inf`` distance and id ``-1``."""

    drone_id: int
    t: float
    origin: Vec3
    dirs: Points
    """Unit ray directions, shape ``(N, 3)``."""
    dist: npt.NDArray[np.float64]
    """Hit distances, shape ``(N,)``; ``inf`` on miss."""
    obj_ids: npt.NDArray[np.int32]
    """Hit object ids, shape ``(N,)``; ``-1`` on miss. Ground truth."""
    tri_ids: npt.NDArray[np.int32]
    """Global triangle indices, shape ``(N,)``; ``-1`` on miss. Ground truth."""
    rgb: npt.NDArray[np.uint8] | None = None
    """Shaded colour of each hit, shape ``(N, 3)``; see :attr:`Observation.rgb`."""

    def observation(self) -> Observation:
        """Return what a real drone could have sensed: this scan without its ids."""
        return Observation(
            drone_id=self.drone_id,
            t=self.t,
            origin=self.origin,
            dirs=self.dirs,
            dist=self.dist,
            rgb=self.rgb,
        )


@dataclass(frozen=True, slots=True, eq=False)
class OrientedBox:
    """A box standing upright in the world frame, turned about +Z by ``yaw``.

    Everything on a property stands on the ground or hangs on a vertical wall,
    so a yaw-only box hugs it as closely as a fully rotated one would, while
    staying trivial to draw and to reason about.
    """

    center: Vec3
    size: Vec3
    """Full extents in metres: along the yaw heading, across it, and height."""
    yaw: float
    """Heading of the first ``size`` axis, radians anticlockwise from +X."""

    def corners(self) -> Points:
        """Return the eight corners, shape ``(8, 3)``, bottom four first."""
        half = np.asarray(self.size, dtype=np.float64) / 2.0
        signs = np.array(
            [[sx, sy, sz] for sz in (-1.0, 1.0) for sy in (-1.0, 1.0) for sx in (-1.0, 1.0)]
        )
        local = signs * half
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        corners: Points = local @ rot.T + np.asarray(self.center, dtype=np.float64)
        return corners


@dataclass(frozen=True, slots=True, eq=False)
class DiscoveredObject:
    """An object the swarm has seen often enough to commit to.

    Perception finds these from range and colour alone, so the key is a track
    id the detector allocated, never a scene object id.
    """

    track_id: int
    """Stable for as long as the detector keeps finding the object."""
    cls: Cls
    pos: Vec3
    """Mean observed surface point."""
    box: OrientedBox
    """Tight upright box around every observed point of the object."""
    n_hits: int
    """Ray hits that make up the object."""
    confidence: float
    """In ``[0, 1]``, rising with the evidence behind the object."""
    first_seen_t: float
    """When any part of the object was first sensed."""
    first_seen_by: int
    """The drone that sensed it then."""


@dataclass(slots=True, eq=False)
class MapState:
    """The shared world model: occupancy, coverage and semantics."""

    occ: npt.NDArray[np.uint8]
    """Voxel grid, shape ``(X, Y, Z)``, values from :class:`Occ`."""
    origin: Vec3
    """World position of the minimum corner of voxel ``(0, 0, 0)``.

    Voxel ``(i, j, k)`` spans ``origin + [i, j, k] * voxel`` to
    ``origin + [i + 1, j + 1, k + 1] * voxel``.
    """
    voxel: float
    tri_seen: npt.NDArray[np.bool_]
    """Per-triangle observed mask over the global triangle index, shape ``(T,)``."""
    discovered: dict[int, DiscoveredObject] = field(default_factory=dict)
    """Objects perception has committed to, by track id.

    Replaced wholesale, never mutated, each time perception re-extracts
    objects, so a reader holding the old dict keeps a consistent snapshot.
    """
    surface_seen: npt.NDArray[np.bool_] | None = None
    """Voxels a photo-quality ray has landed in, shape ``(X, Y, Z)``.

    ``occ`` says a surface is *there*; this says it has been seen well enough
    to count -- close and square-on, by the coverage rule. The 12 m sensor
    settles almost every voxel from a distance, so exploration that stops at
    ``occ`` leaves most walls known but unphotographed. The planner uses the
    difference, occupied but not seen, to choose inspection targets. It is
    built from scans alone, so reading it is not peeking at ground truth.
    ``None`` for maps without a mapper behind them, such as hand-built test
    grids.
    """


@dataclass(slots=True, eq=False)
class DroneTask:
    """What one drone should do until the next replan."""

    drone_id: int
    kind: TaskKind
    """One of ``takeoff``, ``orbit``, ``frontier``, ``inspect``, ``yield``, ``escape``,
    ``rth``, ``land``, ``hold``."""
    goal: Vec3
    look_at: Vec3 | None
    path: Points
    """Waypoints from the current position to the goal, shape ``(K, 3)``."""


@dataclass(slots=True, eq=False)
class SiteCandidate:
    """One scored battery location, with the work it implies."""

    pos: Vec3
    """Ground point at the wall."""
    wall_normal: Vec3
    cost: float
    breakdown: dict[str, float]
    """Per-term contributions: ``conduit_m``, ``bushes``, penalties."""
    conduit: Points
    """Polyline from meter to site, shape ``(K, 3)``."""
    bushes_to_remove: list[int]
    verdict: SiteVerdict
    """This site's own photo-review outcome, independent of the others offered."""
    warnings: tuple[str, ...] = ()
    """One sentence per rule this site fails or is marginal on.

    Empty for a site that meets every rule cleanly. Covers both a rule that
    fails outright and one that is merely marginal (a measurement or another
    photo could clear it) -- :attr:`verdict` is what tells the two apart, not
    the presence of a warning.
    """


@dataclass(slots=True, eq=False)
class Photo:
    """One captured SSR frame and the viewpoint it came from."""

    label: str
    path: str
    cam_pos: Vec3
    look_at: Vec3


@dataclass(slots=True, eq=False)
class SiteResult:
    """The mission's product: what to build, where, and the evidence for it."""

    meter: DiscoveredObject | None
    candidates: list[SiteCandidate]
    """Sorted best-first, at most three."""
    photos: list[Photo]
    coverage_ground_band: float
    mission_time_s: float


@dataclass(frozen=True, slots=True, eq=False)
class SiteAssessment:
    """The whole-property siting call: a verdict, the offered sites, and why.

    A reviewer opens the SSR wanting one answer first -- approve, look closer,
    or send someone out -- and the ranked sites second. Keeping the two
    together (rather than letting a caller infer the verdict from the sites'
    own) is what lets :attr:`verdict` weigh every candidate that was ever
    scored, not just the handful kept for display.
    """

    verdict: SiteVerdict
    """Overall call: PASS if any candidate passes, REJECT only if every one
    of them does, MANUAL_REVIEW otherwise."""
    sites: tuple[SiteCandidate, ...]
    """Offered sites, best first. Empty only when no wall could take a battery."""
    justification: str
    """One to three sentences a reviewer can act on without opening the map."""
    n_candidates: int
    """How many candidates were scored, including the ones not offered."""
