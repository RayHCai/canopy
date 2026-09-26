"""Collision-free paths through the mapped world.

A 3-D least-cost search (:class:`skimage.graph.MCP_Geometric`) on a coarse
planning grid, followed by line-of-sight shortcutting on the fine map grid.

Coarse, because the search is the expensive part and runs once per drone per
replan; at 0.5 m the lot is ~115k cells instead of ~920k. Fine for the
shortcut checks, because those decide what the drone actually flies and must
agree with what the :class:`~canopy.planning.safety.Shield` will accept.

The coarse grid is made conservative rather than approximate: a planning cell
is passable only if every map voxel in it, *and one voxel beyond it*, has the
required clearance. The halo is what lets a raw diagonal step between two
passable cells be trusted without a line-of-sight check -- the step's
midpoint sits on a corner shared with cells that may be blocked, and its
nearest map voxel may belong to one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.graph import MCP_Geometric

from canopy import mathutil
from canopy.contracts import Points, Vec3
from canopy.errors import PlanningError
from canopy.planning.safety import ClearanceMap

__all__ = ["PlanningGrid", "plan_path", "planning_grid"]


def plan_path(
    start: Vec3,
    goal: Vec3,
    clearance: ClearanceMap,
    *,
    plan_factor: int,
    required_m: float,
    avoid: Points | None = None,
    avoid_radius_m: float = 0.0,
) -> Points:
    """Shortest safe path from ``start`` to ``goal``.

    Parameters
    ----------
    start
        Current drone position. Its cell is always passable, so a drone that
        has drifted into a margin can plan its way out. A drone on the ground
        cannot: every cell beside the ground is inside the margin, so TAKEOFF
        must climb straight up (the shield allows that) before planning.
    goal
        Destination; must itself have ``required_m`` clearance.
    clearance
        The current :class:`ClearanceMap`.
    plan_factor
        Map voxels per planning cell along each axis (``MapCfg.plan_factor``).
    required_m
        Clearance every waypoint and segment must keep (``SafetyCfg.inflation_m``).
    avoid
        Other drones to route around, shape ``(K, 3)``. The mission controller
        passes these when the shield has held a drone for too long: the other
        drones are then parked in its way, and the map alone cannot know that.
    avoid_radius_m
        Distance to keep from every ``avoid`` point (``min_separation_m``).

    Returns
    -------
    Points
        Waypoints from ``start`` to ``goal`` inclusive, shape ``(K, 3)``, ``K >= 2``.

    Raises
    ------
    PlanningError
        If ``plan_factor`` is not positive, either endpoint is off the map,
        the goal is unsafe, or no path exists.
    """
    if plan_factor < 1:
        msg = f"plan_factor must be a positive integer, got {plan_factor}"
        raise PlanningError(msg)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    avoid_pts = (
        np.zeros((0, 3), dtype=np.float64)
        if avoid is None
        else np.asarray(avoid, dtype=np.float64).reshape(-1, 3)
    )

    goal_clearance = float(clearance.clearance_at(goal)[0])
    if goal_clearance < required_m:
        msg = (
            f"goal {goal.round(2).tolist()} has {goal_clearance:.2f} m clearance, "
            f"needs {required_m}"
        )
        raise PlanningError(msg)
    if len(avoid_pts) and float(np.linalg.norm(avoid_pts - goal, axis=1).min()) < avoid_radius_m:
        msg = f"goal {goal.round(2).tolist()} is within {avoid_radius_m} m of a drone to avoid"
        raise PlanningError(msg)

    cell = clearance.voxel * plan_factor
    passable = _coarse_passable(clearance, plan_factor, required_m)
    if len(avoid_pts):
        passable &= ~_near_points(passable.shape, clearance.origin, cell, avoid_pts, avoid_radius_m)

    start_idx = _cell_index(start, clearance.origin, cell, passable.shape, "start")
    goal_idx = _cell_index(goal, clearance.origin, cell, passable.shape, "goal")
    passable[start_idx] = True
    if not passable[goal_idx]:
        msg = (
            f"goal {goal.round(2).tolist()} lies in a planning cell without "
            f"{required_m} m clearance"
        )
        raise PlanningError(msg)

    mcp = MCP_Geometric(np.where(passable, 1.0, np.inf))
    cumulative, _ = mcp.find_costs([start_idx], [goal_idx])
    if not np.isfinite(cumulative[goal_idx]):
        msg = f"no safe path from {start.round(2).tolist()} to {goal.round(2).tolist()}"
        raise PlanningError(msg)

    cells = np.asarray(mcp.traceback(goal_idx), dtype=np.float64)
    raw = clearance.origin + (cells + 0.5) * cell
    raw[0], raw[-1] = start, goal
    if len(raw) == 1:  # start and goal share a cell
        raw = np.vstack([start, goal])
    return _shortcut(raw, clearance, required_m, avoid_pts, avoid_radius_m)


@dataclass(frozen=True, slots=True, eq=False)
class PlanningGrid:
    """The coarse passability grid used by :func:`plan_path`, exposed for reuse.

    Frontier search needs the same notion of "reachable" the path planner
    uses -- a frontier viewpoint is only useful if a drone can actually get
    there -- so this is the same conservative min-pooled grid, labelled into
    connected components instead of searched point-to-point.
    """

    passable: npt.NDArray[np.bool_]
    """Coarse cell passability, shape ``(X', Y', Z')``."""
    labels: npt.NDArray[np.int32]
    """26-connected component id per cell; ``0`` marks an impassable cell."""
    origin: Vec3
    """World position of planning cell ``(0, 0, 0)``'s minimum corner."""
    cell: float
    """Planning cell edge length in metres (``clearance.voxel * plan_factor``)."""
    forced: npt.NDArray[np.bool_] | None = None
    """Cells passable only because a start sits in them, shape like ``passable``.

    A drone whose cell is forced is inside a margin at planning resolution,
    even if its exact position is clear. It shares a component with at most
    whatever happens to touch its cell, and the first leg of any path out
    of it is unchecked. The mission controller moves such a drone to a
    genuinely passable cell before planning anything else for it.
    """

    def label_at(self, points: Points) -> npt.NDArray[np.int32]:
        """Component label at each point, ``0`` if outside the grid or impassable.

        Parameters
        ----------
        points
            Query points, shape ``(N, 3)``.

        Returns
        -------
        numpy.ndarray
            Labels, shape ``(N,)``, ``int32``.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        idx = np.floor((pts - self.origin) / self.cell).astype(np.intp)
        in_grid = np.all((idx >= 0) & (idx < np.asarray(self.passable.shape)), axis=1)
        out = np.zeros(len(pts), dtype=np.int32)
        i, j, k = idx[in_grid].T
        out[in_grid] = self.labels[i, j, k]
        return out

    def centres(self, mask: npt.NDArray[np.bool_]) -> Points:
        """World-space centres of the cells where ``mask`` is true.

        Parameters
        ----------
        mask
            Boolean array shaped like :attr:`passable`.

        Returns
        -------
        Points
            Shape ``(K, 3)``, in the order ``np.argwhere`` visits ``mask``.
        """
        idx = np.argwhere(mask)
        result: Points = self.origin + (idx.astype(np.float64) + 0.5) * self.cell
        return result


def planning_grid(
    clearance: ClearanceMap,
    *,
    plan_factor: int,
    required_m: float,
    starts: Points,
) -> PlanningGrid:
    """Build the coarse passability grid, labelled into 26-connected components.

    Parameters
    ----------
    clearance
        The current :class:`ClearanceMap`.
    plan_factor
        Map voxels per planning cell along each axis.
    required_m
        Clearance every passable cell must keep.
    starts
        Drone positions, shape ``(N, 3)``. Each start's cell is forced
        passable, exactly as :func:`plan_path` does for its own start, so a
        drone that has drifted into a margin still gets a component. A start
        outside the map is silently ignored: some drones may be dead or
        off-grid, and that is not this function's problem to raise about.

    Returns
    -------
    PlanningGrid

    Raises
    ------
    PlanningError
        If ``plan_factor`` is not a positive integer.
    """
    if plan_factor < 1:
        msg = f"plan_factor must be a positive integer, got {plan_factor}"
        raise PlanningError(msg)
    cell = clearance.voxel * plan_factor
    passable = _coarse_passable(clearance, plan_factor, required_m)

    pts = np.atleast_2d(np.asarray(starts, dtype=np.float64)).reshape(-1, 3)
    idx = np.floor((pts - clearance.origin) / cell).astype(np.intp)
    in_grid = np.all((idx >= 0) & (idx < np.asarray(passable.shape)), axis=1)
    i, j, k = idx[in_grid].T
    forced = np.zeros_like(passable)
    forced[i, j, k] = ~passable[i, j, k]
    passable[i, j, k] = True

    # 26-connectivity: every voxel in a 3x3x3 neighbourhood is a neighbour.
    structure = ndimage.generate_binary_structure(3, 3)
    labels, _ = ndimage.label(passable, structure=structure)
    return PlanningGrid(
        passable=passable,
        labels=labels.astype(np.int32),
        origin=np.asarray(clearance.origin, dtype=np.float64),
        cell=cell,
        forced=forced,
    )


#: Pooled grids per clearance map. A replan builds one ClearanceMap and then
#: plans a path per drone on it, plus the component labelling, and the 3-D
#: minimum filter below is the bulk of each of those. The weak key lets a
#: stale map's grids go with it.
_POOLED: WeakKeyDictionary[ClearanceMap, dict[tuple[int, float], npt.NDArray[np.bool_]]] = (
    WeakKeyDictionary()
)


def _coarse_passable(
    clearance: ClearanceMap, factor: int, required_m: float
) -> npt.NDArray[np.bool_]:
    """Min-pool the clearance field, with a one-voxel halo, into planning cells.

    Cached per ``clearance`` (whose field is fixed at construction); the
    result is a fresh copy, because callers force start cells passable in it.
    """
    cached = _POOLED.setdefault(clearance, {})
    key = (factor, required_m)
    if key not in cached:
        cached[key] = _pool_passable(clearance, factor, required_m)
    return cached[key].copy()


def _pool_passable(
    clearance: ClearanceMap, factor: int, required_m: float
) -> npt.NDArray[np.bool_]:
    """Compute what :func:`_coarse_passable` caches."""
    # Outside the map is unsafe, hence cval=0 for both the halo and the padding.
    halo = ndimage.minimum_filter(clearance.field, size=3, mode="constant", cval=0.0)
    pad = [(0, (-n) % factor) for n in halo.shape]
    padded = np.pad(halo, pad, constant_values=0.0)
    x, y, z = (n // factor for n in padded.shape)
    pooled = padded.reshape(x, factor, y, factor, z, factor).min(axis=(1, 3, 5))
    passable: npt.NDArray[np.bool_] = pooled >= required_m
    return passable


def _near_points(
    shape: tuple[int, ...], origin: Vec3, cell: float, points: Points, radius: float
) -> npt.NDArray[np.bool_]:
    """Cells that come within ``radius`` of any point.

    Padded by the cell's half-diagonal: a raw step between two cell centres
    passes that far from either, and must not graze the radius mid-step.
    """
    axes = [origin[a] + (np.arange(shape[a]) + 0.5) * cell for a in range(3)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    reach = radius + 0.5 * np.sqrt(3.0) * cell
    near = np.zeros(shape, dtype=np.bool_)
    for p in points:  # K is the swarm size; each iteration is a full-grid vector op
        near |= np.sum((centres - p) ** 2, axis=-1) < reach * reach
    return near


def _cell_index(
    point: Vec3, origin: Vec3, cell: float, shape: tuple[int, ...], name: str
) -> tuple[int, int, int]:
    """Planning cell containing ``point``."""
    idx = np.floor((point - origin) / cell).astype(np.intp)
    if np.any(idx < 0) or np.any(idx >= np.asarray(shape)):
        msg = f"{name} {point.round(2).tolist()} is outside the map"
        raise PlanningError(msg)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _shortcut(
    path: Points,
    clearance: ClearanceMap,
    required_m: float,
    avoid: Points,
    avoid_radius_m: float,
) -> Points:
    """Greedy string-pulling: from each kept waypoint, jump as far ahead as is visible.

    Scans forward and stops at the first blocked candidate rather than testing
    every later waypoint. That can miss a longer shortcut past an obstruction,
    but costs one check per waypoint instead of one per pair, and the raw
    search path is already near-optimal apart from its grid staircase.
    """
    kept = [path[0]]
    i = 0
    last = len(path) - 1
    while i < last:
        j = i + 1  # raw steps are safe by construction of the coarse grid
        while j < last and _visible(
            path[i], path[j + 1], clearance, required_m, avoid, avoid_radius_m
        ):
            j += 1
        kept.append(path[j])
        i = j
    return np.asarray(kept, dtype=np.float64)


def _visible(
    a: Vec3,
    b: Vec3,
    clearance: ClearanceMap,
    required_m: float,
    avoid: Points,
    avoid_radius_m: float,
) -> bool:
    """Whether the straight segment ``a -> b`` keeps every required distance.

    Uses the shield's own map rule, recovery leniency included, so a shortcut
    the planner takes is one the shield passes.
    """
    if not clearance.segment_clear(a, b, required_m):
        return False
    if not len(avoid):
        return True
    samples = clearance.sample_segment(a, b)
    gaps = np.linalg.norm(samples[:, None, :] - avoid[None, :, :], axis=-1)
    floors = np.minimum(avoid_radius_m, np.linalg.norm(avoid - a, axis=1))
    return bool(np.all(gaps >= floors - mathutil.EPS))
