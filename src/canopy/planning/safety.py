"""Safety shield: the last check between the planner and the motors.

Two pieces, split so the planner and the shield judge space identically:

* :class:`ClearanceMap` turns the shared occupancy grid into a metric
  distance-to-danger field. Only FREE voxels count as safe -- UNKNOWN is
  treated exactly like OCC, so a drone never enters space nobody has observed
  -- and the geofence is folded into the same field.
* :class:`Shield` vets every drone's commanded motion, every control tick,
  against that field and against every other drone. A motion that fails is
  replaced by a hold (brake to a stop), never by a new plan: re-routing is the
  planner's job, and a shield that improvised paths would be a second planner
  nobody tested.

Everything here reads the *mapped* world. The ground-truth scene is off limits:
a shield that peeked at it would pass every test and fail the first real
flight.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from canopy import mathutil
from canopy.contracts import DroneState, MapState, Occ, Points, Vec3

if TYPE_CHECKING:
    from canopy.config import SafetyCfg, SimCfg

__all__ = [
    "ClearanceMap",
    "Shield",
    "ShieldVerdict",
    "geofence_box",
    "segment_distances",
]

#: Metric 3-D field on the map grid, shape ``(X, Y, Z)``.
FloatGrid = npt.NDArray[np.float64]

#: Slack on "no worse than now" comparisons. Clearances and separations are
#: exact reads of the same field, so this only absorbs float round-off.
_TOLERANCE_M = 1e-9


def geofence_box(lot_bounds: npt.NDArray[np.float64], cfg: SafetyCfg) -> npt.NDArray[np.float64]:
    """Box the drone *centre* must stay inside, shape ``(2, 3)``.

    The lot shrunk horizontally by ``geofence_inset_m`` and capped at
    ``ceiling_m``. There is no floor: the ground is an obstacle in the map, and
    a floor here would put the launch pad at ``z = 0`` outside the fence.
    """
    bounds = np.asarray(lot_bounds, dtype=np.float64)
    inset = cfg.geofence_inset_m
    return np.array(
        [
            [bounds[0, 0] + inset, bounds[0, 1] + inset, -np.inf],
            [bounds[1, 0] - inset, bounds[1, 1] - inset, cfg.ceiling_m],
        ],
        dtype=np.float64,
    )


class ClearanceMap:
    """Distance from any point to the nearest space a drone may not occupy.

    Built once per map update (the replan rate), then queried many times per
    control tick, so all the work happens in the constructor.

    Parameters
    ----------
    clearance
        Metres from each voxel to the nearest non-FREE voxel, shape
        ``(X, Y, Z)``. Zero for voxels that are themselves unsafe.
    origin
        Minimum corner of voxel ``(0, 0, 0)``.
    voxel
        Edge length in metres.
    geofence
        ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` for the drone centre.
    """

    def __init__(
        self,
        clearance: FloatGrid,
        origin: Vec3,
        voxel: float,
        geofence: npt.NDArray[np.float64],
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel = float(voxel)
        self.geofence = np.asarray(geofence, dtype=np.float64)
        # Bake the fence into the grid so the planner, which only sees the grid,
        # respects it too; clearance_at() additionally checks exact points.
        self.field = np.where(self._fence_mask(clearance.shape), clearance, 0.0)

    @classmethod
    def from_map(cls, state: MapState, geofence: npt.NDArray[np.float64]) -> ClearanceMap:
        """Derive the clearance field from the shared occupancy grid.

        The Euclidean distance transform measures voxel centre to voxel centre;
        half a voxel is subtracted so the value approximates the distance to the
        near face of the offending voxel. The grid is padded with one layer of
        unsafe voxels first: without it the transform treats the array edge as
        open sky, and a drone at the rim of the map would look far from danger
        while flying into unmapped space.
        """
        free = np.pad(state.occ == Occ.FREE, 1, constant_values=False)
        centre_distance = np.asarray(
            ndimage.distance_transform_edt(free, sampling=state.voxel), dtype=np.float64
        )[1:-1, 1:-1, 1:-1]
        clearance = np.maximum(centre_distance - 0.5 * state.voxel, 0.0)
        return cls(clearance, state.origin, state.voxel, geofence)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape ``(X, Y, Z)``."""
        x, y, z = self.field.shape
        return int(x), int(y), int(z)

    def clearance_at(self, points: Points) -> npt.NDArray[np.float64]:
        """Clearance at each point, shape ``(N,)``.

        Nearest-voxel lookup. Points outside the grid or the geofence read
        zero: unmapped and forbidden space are equally unsafe.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        idx = np.floor((pts - self.origin) / self.voxel).astype(np.intp)
        in_grid = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        in_fence = np.all((pts >= self.geofence[0]) & (pts <= self.geofence[1]), axis=1)
        safe = in_grid & in_fence

        out = np.zeros(len(pts), dtype=np.float64)
        i, j, k = idx[safe].T
        out[safe] = self.field[i, j, k]
        return out

    def sample_segment(self, start: Vec3, end: Vec3) -> Points:
        """Points along ``start -> end``, in order, hitting every voxel it passes through.

        Half-voxel steps alone cannot skip a voxel the segment passes through
        the middle of, but can step over one whose corner it clips -- and which
        corners get stepped over depends on where the segment ends. The shield
        checks the first stopping distance of a leg the planner checked whole,
        so the two would disagree about the same line, and a leg the planner
        hands out could be refused on every tick forever. One extra sample
        midway between each pair of consecutive voxel-boundary crossings makes
        the visited voxels exact, so any prefix of a clear segment is clear.
        The half-voxel steps stay for callers that measure distances to the
        samples rather than looking voxels up.
        """
        a = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - a
        n = max(2, int(np.ceil(mathutil.norm(delta) / (0.5 * self.voxel))) + 1)
        crossings = [np.array([0.0, 1.0])]
        for axis in np.flatnonzero(np.abs(delta) > mathutil.EPS):
            lo, hi = sorted((a[axis], a[axis] + delta[axis]))
            first = np.ceil((lo - self.origin[axis]) / self.voxel)
            last = np.floor((hi - self.origin[axis]) / self.voxel)
            planes = self.origin[axis] + np.arange(first, last + 1) * self.voxel
            crossings.append((planes - a[axis]) / delta[axis])
        cross = np.unique(np.clip(np.concatenate(crossings), 0.0, 1.0))
        t = np.unique(np.concatenate([np.linspace(0.0, 1.0, n), 0.5 * (cross[:-1] + cross[1:])]))
        return np.asarray(a + t[:, None] * delta, dtype=np.float64)

    def segment_clear(
        self, start: Vec3, end: Vec3, required_m: float, *, require_gain: bool = True
    ) -> bool:
        """Whether flying ``start -> end`` keeps ``required_m`` clearance.

        A segment starting inside the margin (a drone on the launch pad reads
        zero) is judged by a recovery rule instead:

        * it may pass through no unsafe voxel except the one it starts in --
          it may leave unsafe space but never enter any; and
        * clearance may never fall below the best already reached along the
          segment, capped at ``required_m``.

        Both are needed because "no worse than the start" is vacuous from a
        start of zero: every point, the inside of the wall beside the pad
        included, satisfies it. Zero is reserved for unsafe voxels (a FREE
        voxel reads at least half a voxel), so the first rule is exact.

        Parameters
        ----------
        start, end
            Segment endpoints.
        required_m
            Clearance to keep once reached.
        require_gain
            In recovery, also demand that ``end`` be clearer than ``start``, so
            a grounded drone can climb out but not slide along the lawn. Off
            for a braking segment, which may be a single point.
        """
        samples = self.sample_segment(start, end)
        values = self.clearance_at(samples)
        if values[0] >= required_m:
            return bool(np.all(values >= required_m))
        cells = np.floor((samples - self.origin) / self.voxel)
        in_start_cell = np.all(cells == cells[0], axis=1)
        if bool(np.any((values <= 0.0) & ~in_start_cell)):
            return False
        floor = np.minimum(np.maximum.accumulate(values), required_m)
        if not bool(np.all(values >= floor - _TOLERANCE_M)):
            return False
        return not require_gain or bool(values[-1] > values[0])

    def _fence_mask(self, shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
        """Whether each voxel centre lies inside the geofence."""
        axes = [
            (self.origin[a] + (np.arange(shape[a]) + 0.5) * self.voxel >= self.geofence[0, a])
            & (self.origin[a] + (np.arange(shape[a]) + 0.5) * self.voxel <= self.geofence[1, a])
            for a in range(3)
        ]
        mask: npt.NDArray[np.bool_] = (
            axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
        )
        return mask


def segment_distances(p1: Points, q1: Points, p2: Points, q2: Points) -> npt.NDArray[np.float64]:
    """Minimum distance between segment pairs ``p1[i]-q1[i]`` and ``p2[i]-q2[i]``.

    Vectorised closest-point-between-segments (Ericson, *Real-Time Collision
    Detection*, section 5.1.9). Zero-length segments are points and handled.

    Parameters
    ----------
    p1, q1, p2, q2
        Segment endpoints, each shape ``(M, 3)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(M,)``.
    """
    eps = mathutil.EPS
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    f = np.einsum("ij,ij->i", d2, r)
    a_safe = np.where(a > eps, a, 1.0)
    e_safe = np.where(e > eps, e, 1.0)

    # General case, then clamp t into the segment and re-derive s from it.
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0, 1), 0)
    t = (b * s + f) / e_safe
    s = np.where(
        t < 0, np.clip(-c / a_safe, 0, 1), np.where(t > 1, np.clip((b - c) / a_safe, 0, 1), s)
    )
    t = np.clip(t, 0, 1)

    # Degenerate segments, most specific last so it wins.
    second_is_point = e <= eps
    s = np.where(second_is_point, np.clip(-c / a_safe, 0, 1), s)
    t = np.where(second_is_point, 0.0, t)
    first_is_point = a <= eps
    s = np.where(first_is_point, 0.0, s)
    t = np.where(first_is_point, np.clip(f / e_safe, 0, 1), t)
    t = np.where(first_is_point & second_is_point, 0.0, t)

    gap = (p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])
    return np.asarray(np.linalg.norm(gap, axis=1), dtype=np.float64)


#: A swept segment ``(start, end)`` a drone may occupy before it can stop.
_Segment = tuple[Vec3, Vec3]


@dataclass(frozen=True, slots=True, eq=False)
class ShieldVerdict:
    """What the shield let through on one tick."""

    targets: dict[int, Vec3]
    """Approved targets, ready for :meth:`KinematicDynamics.step`. A drone
    missing from here brakes to a stop."""
    held: dict[int, str] = field(default_factory=dict)
    """Drones the shield stopped, with the reason (``"map"`` or ``"drone N"``)."""


class Shield:
    """Per-tick vetting of commanded motion against the map and each other.

    Each drone owns two swept segments: the *brake* segment it would cover if
    told to stop now, and the *go* segment toward its target, one full stopping
    distance long. A drone may go only if both are clear. Since the go segment
    already contains a stop, a drone approved this tick can always be held
    next tick and still come to rest in space that was checked -- the
    invariant the whole shield rests on.

    Drones are decided in priority order, lowest ``drone_id`` first. Each checks
    its segments against the reservations of drones already decided and
    against the brake segments of drones not yet decided (the worst they can
    do is stop). A held drone reserves only its brake segment, so a drone
    further down the order still sees it.

    Two drones on a collision course therefore both end up holding, safely
    apart, until the mission controller re-plans one of them around the other.
    :meth:`held_for_s` tells it when; ``safety.hold_replan_s`` says how long to
    wait.

    Recovery. A drone already inside a margin -- on the launch pad, where the
    ground voxel reads zero clearance, or spawned 1 m from its neighbour on a
    line with 1.5 m separation -- could never move under a strict rule. It may
    instead make any move that does not bring it closer to the violated
    constraint; for the map that means climbing out without dipping back in
    (see :meth:`ClearanceMap.segment_clear`), so a grounded drone can take
    off but can neither slide along the lawn nor lift off through a wall.
    """

    def __init__(self, safety: SafetyCfg, sim: SimCfg) -> None:
        self._safety = safety
        self._sim = sim
        self._held_ticks: dict[int, int] = {}

    def held_for_s(self, drone_id: int) -> float:
        """Seconds ``drone_id`` has been held continuously; zero if moving."""
        return self._held_ticks.get(drone_id, 0) * self._sim.dt

    def filter(
        self,
        states: Sequence[DroneState],
        targets: Mapping[int, Vec3],
        clearance: ClearanceMap,
        *,
        landing: Collection[int] = (),
    ) -> ShieldVerdict:
        """Approve or hold each drone's target for this tick.

        Parameters
        ----------
        states
            Every drone, dead ones included: a drone falling out of the sky is
            still something to avoid.
        targets
            The follower's current waypoint per ``drone_id``. Drones without one
            are braking already and only reserve their brake segment.
        clearance
            The current :class:`ClearanceMap`.
        landing
            Drones descending onto their own launch pad. Touchdown is the one
            move that must end inside the ground margin, which the map check
            exists to forbid, so for these drones only the drone-to-drone check
            applies. It is safe because the mission hands out a landing only
            from a hover directly above a pad the operator surveyed clear before
            launch -- a vertical drop through space the drone has already flown,
            or, above the take-off hover, has checked clear on the map.

        Returns
        -------
        ShieldVerdict
            Approved targets and the held drones with reasons.
        """
        # The loop over drones is inherently sequential: each decision changes
        # what the next drone must avoid. The per-drone work is vectorised.
        live = sorted((s for s in states if s.alive), key=lambda s: s.drone_id)
        reserved: dict[int, list[_Segment]] = {
            s.drone_id: [(s.pos, np.array([s.pos[0], s.pos[1], 0.0]))]
            for s in states
            if not s.alive
        }
        # Until a drone is decided, assume the worst it will do is stop.
        for s in live:
            reserved[s.drone_id] = [self._brake_segment(s)]
        positions = {s.drone_id: s.pos for s in states}

        approved: dict[int, Vec3] = {}
        held: dict[int, str] = {}
        for s in live:
            target = targets.get(s.drone_id)
            if target is None:
                continue
            brake = reserved[s.drone_id][0]
            go = self._go_segment(s, np.asarray(target, dtype=np.float64))
            map_reason = None if s.drone_id in landing else self._map_conflict(go, brake, clearance)
            reason = map_reason or self._drone_conflict(s, [go, brake], reserved, positions)
            if reason is None:
                approved[s.drone_id] = target
                reserved[s.drone_id] = [go, brake]
            else:
                held[s.drone_id] = reason

        for s in live:
            if s.drone_id in held:
                self._held_ticks[s.drone_id] = self._held_ticks.get(s.drone_id, 0) + 1
            else:
                self._held_ticks.pop(s.drone_id, None)
        return ShieldVerdict(targets=approved, held=held)

    # -- swept volumes -----------------------------------------------------
    def _stopping_distance(self, speed: float) -> float:
        """Distance covered while braking from ``speed``, plus one reaction tick."""
        return speed * speed / (2.0 * self._sim.a_max) + speed * self._sim.dt

    def _brake_segment(self, state: DroneState) -> _Segment:
        """Where the drone ends up if held from this tick on."""
        speed = mathutil.norm(state.vel)
        return state.pos, state.pos + mathutil.unit(state.vel) * self._stopping_distance(speed)

    def _go_segment(self, state: DroneState, target: Vec3) -> _Segment:
        """Straight toward ``target``, one worst-case stopping distance long.

        Worst case, not current speed: the drone may be at ``v_max`` by the time
        it has to stop, and a short look-ahead at low speed would let it
        accelerate into a check it can no longer pass.
        """
        reach = self._stopping_distance(self._sim.v_max)
        offset = target - state.pos
        length = min(mathutil.norm(offset), reach)
        return state.pos, state.pos + mathutil.unit(offset) * length

    # -- checks --------------------------------------------------------------
    def _map_conflict(self, go: _Segment, brake: _Segment, clearance: ClearanceMap) -> str | None:
        """``"map"`` if either segment leaves known-free, fenced space."""
        required = self._safety.inflation_m
        ok = clearance.segment_clear(*go, required) and clearance.segment_clear(
            *brake, required, require_gain=False
        )
        return None if ok else "map"

    def _drone_conflict(
        self,
        state: DroneState,
        mine: list[_Segment],
        reserved: Mapping[int, list[_Segment]],
        positions: Mapping[int, Vec3],
    ) -> str | None:
        """``"drone N"`` for the first drone whose reservation comes too close."""
        others = [
            (other, seg)
            for other, segs in reserved.items()
            if other != state.drone_id
            for seg in segs
        ]
        if not others:
            return None
        pairs = [(m, seg) for m in mine for _, seg in others]
        ids = np.array([other for _ in mine for other, _ in others])
        gaps = segment_distances(
            np.array([m[0] for m, _ in pairs]),
            np.array([m[1] for m, _ in pairs]),
            np.array([o[0] for _, o in pairs]),
            np.array([o[1] for _, o in pairs]),
        )
        # Drones already closer than the minimum may keep, but not shrink, the gap.
        current = np.array([mathutil.norm(state.pos - positions[i]) for i in ids])
        required = np.minimum(self._safety.min_separation_m, current) - _TOLERANCE_M
        bad = gaps < required
        return f"drone {int(ids[np.argmax(bad)])}" if bool(bad.any()) else None
