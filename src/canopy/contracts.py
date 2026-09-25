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
from enum import IntEnum
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
    "Occ",
    "Photo",
    "Points",
    "Rgb",
    "Scan",
    "SceneManifest",
    "SceneObject",
    "SiteCandidate",
    "SiteResult",
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


class Occ(IntEnum):
    """Occupancy state of a voxel in :attr:`MapState.occ`."""

    UNKNOWN = 0
    FREE = 1
    OCC = 2


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


@dataclass(slots=True, eq=False)
class SceneManifest:
    """The single source of truth for physics, sensing and visualisation."""

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
    """Hit object ids, shape ``(N,)``; ``-1`` on miss."""
    tri_ids: npt.NDArray[np.int32]
    """Global triangle indices, shape ``(N,)``; ``-1`` on miss."""


@dataclass(slots=True, eq=False)
class DiscoveredObject:
    """An object the swarm has seen often enough to commit to."""

    obj_id: int
    cls: Cls
    pos: Vec3
    """Running mean of observed hit points."""
    n_hits: int
    confidence: float
    """``1.0`` for the ground-truth detector."""
    first_seen_t: float
    first_seen_by: int


@dataclass(slots=True, eq=False)
class MapState:
    """The shared world model: occupancy, coverage and semantics."""

    occ: npt.NDArray[np.uint8]
    """Voxel grid, shape ``(X, Y, Z)``, values from :class:`Occ`."""
    origin: Vec3
    """World position of voxel ``(0, 0, 0)``."""
    voxel: float
    tri_seen: npt.NDArray[np.bool_]
    """Per-triangle observed mask over the global triangle index, shape ``(T,)``."""
    discovered: dict[int, DiscoveredObject] = field(default_factory=dict)


@dataclass(slots=True, eq=False)
class DroneTask:
    """What one drone should do until the next replan."""

    drone_id: int
    kind: TaskKind
    """One of ``orbit``, ``frontier``, ``inspect``, ``rth``, ``hold``."""
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
