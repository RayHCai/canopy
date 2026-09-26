"""Drone-to-frontier assignment: a linear assignment problem, solved once per replan.

Cost balances three things: travel distance to the viewpoint, the frontier's
gain (bigger unexplored regions win ties), and a conflict penalty that keeps
two drones from converging on the same neighbourhood. Hysteresis then damps
the churn a fresh solve would otherwise cause every tick.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from canopy.contracts import Vec3
from canopy.planning.frontier import Frontier

if TYPE_CHECKING:
    from canopy.config import PlannerCfg

__all__ = ["assign_frontiers"]

#: Stand-in for "infeasible" in the cost matrix. Large enough that no real
#: cost (distance minus gain plus penalty) reaches it, finite so the Hungarian
#: solver's internals stay well-behaved.
_INFEASIBLE = 1e9


def assign_frontiers(
    positions: Mapping[int, Vec3],
    frontiers: Sequence[Frontier],
    current_goals: Mapping[int, Vec3],
    planner: PlannerCfg,
    *,
    components: Mapping[int, int],
) -> dict[int, int]:
    """Match live drones to frontiers, minimizing travel cost with anti-thrash rules.

    Parameters
    ----------
    positions
        Current position per ``drone_id``.
    frontiers
        Candidate frontiers this replan, as returned by
        :func:`~canopy.planning.frontier.find_frontiers`.
    current_goals
        Each drone's goal viewpoint from the previous assignment, if any.
        Drones absent here have no goal to hold onto.
    planner
        Supplies ``gain_lambda``, ``goal_conflict_radius_m``,
        ``goal_conflict_penalty`` and ``hysteresis``.
    components
        Each drone's current :class:`~canopy.planning.pathing.PlanningGrid`
        label. A drone missing here, or whose label does not match a
        frontier's ``component``, cannot reach that frontier -- the map
        between them is not connected yet -- so the pair is infeasible.

    Returns
    -------
    dict[int, int]
        ``drone_id`` -> index into ``frontiers``. Drones with no feasible
        frontier (more drones than frontiers, or every frontier infeasible
        for them) are absent; the mission controller holds them.
    """
    drone_ids = sorted(positions)
    if not drone_ids or not frontiers:
        return {}

    cost = _cost_matrix(drone_ids, positions, frontiers, current_goals, planner, components)
    assignment = _solve(drone_ids, cost)

    # Hysteresis: a drone whose current goal is still represented among this
    # tick's frontiers keeps it unless the fresh assignment is decisively
    # better. Without this, a frontier's gain flickering by a voxel between
    # replans would send drones back and forth instead of finishing an area.
    keepers: dict[int, int] = {}
    for d_pos, drone_id in enumerate(drone_ids):
        goal = current_goals.get(drone_id)
        if goal is None:
            continue
        f_star = _nearest_feasible(goal, frontiers, cost[d_pos], planner.goal_conflict_radius_m)
        if f_star is None:
            continue
        chosen = assignment.get(drone_id)
        if chosen == f_star:
            continue
        cost_keep = cost[d_pos, f_star]
        if cost_keep >= _INFEASIBLE:
            continue
        cost_new = cost[d_pos, chosen] if chosen is not None else _INFEASIBLE
        if cost_new >= cost_keep - planner.hysteresis * abs(cost_keep):
            keepers[drone_id] = f_star

    if not keepers:
        return assignment

    # Pass 2: fix the keepers' rows and frontiers, re-solve everyone else, merge.
    remaining_drones = [d for d in drone_ids if d not in keepers]
    used_frontiers = set(keepers.values())
    remaining_frontier_idx = [i for i in range(len(frontiers)) if i not in used_frontiers]
    result = dict(keepers)
    if remaining_drones and remaining_frontier_idx:
        sub_cost = cost[
            np.ix_([drone_ids.index(d) for d in remaining_drones], remaining_frontier_idx)
        ]
        sub_assignment = _solve(remaining_drones, sub_cost)
        for d, local_f in sub_assignment.items():
            result[d] = remaining_frontier_idx[local_f]
    return result


def _cost_matrix(
    drone_ids: list[int],
    positions: Mapping[int, Vec3],
    frontiers: Sequence[Frontier],
    current_goals: Mapping[int, Vec3],
    planner: PlannerCfg,
    components: Mapping[int, int],
) -> npt.NDArray[np.float64]:
    """Build the ``(D, F)`` cost matrix, infeasible pairs at ``_INFEASIBLE``."""
    pos = np.array([positions[d] for d in drone_ids], dtype=np.float64)
    viewpoints = np.array([f.viewpoint for f in frontiers], dtype=np.float64)
    gains = np.array([f.gain for f in frontiers], dtype=np.float64)
    dist = np.linalg.norm(pos[:, None, :] - viewpoints[None, :, :], axis=-1)
    cost = dist - planner.gain_lambda * gains[None, :]

    # Conflict penalty: a frontier near another drone's current goal is
    # de-prioritized for everyone else, so two drones don't head into the
    # same unexplored region while a dozen others sit unclaimed.
    if current_goals:
        goal_pts = np.array(list(current_goals.values()), dtype=np.float64)
        goal_owners = list(current_goals)
        near_goal = (
            np.linalg.norm(viewpoints[:, None, :] - goal_pts[None, :, :], axis=-1)
            <= planner.goal_conflict_radius_m
        )  # (F, G)
        for d_pos, drone_id in enumerate(drone_ids):
            other_mask = np.array([owner != drone_id for owner in goal_owners])
            conflicted = (
                np.any(near_goal[:, other_mask], axis=1)
                if other_mask.any()
                else np.zeros(len(frontiers), dtype=np.bool_)
            )
            cost[d_pos, conflicted] += planner.goal_conflict_penalty

    infeasible = np.ones((len(drone_ids), len(frontiers)), dtype=np.bool_)
    for d_pos, drone_id in enumerate(drone_ids):
        comp = components.get(drone_id)
        if comp is None:
            continue
        infeasible[d_pos] = np.array([f.component != comp for f in frontiers])
    return np.where(infeasible, _INFEASIBLE, cost)


def _solve(drone_ids: list[int], cost: npt.NDArray[np.float64]) -> dict[int, int]:
    """Hungarian solve, discarding any pairing that landed on the infeasible sentinel."""
    rows, cols = linear_sum_assignment(cost)
    result: dict[int, int] = {}
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] < _INFEASIBLE:
            result[drone_ids[r]] = int(c)
    return result


def _nearest_feasible(
    goal: Vec3,
    frontiers: Sequence[Frontier],
    drone_cost_row: npt.NDArray[np.float64],
    radius_m: float,
) -> int | None:
    """Index of the frontier whose viewpoint is nearest ``goal`` within ``radius_m``.

    ``None`` if no frontier is within range, or the nearest one is infeasible
    for this drone (its cost reads the infeasible sentinel).
    """
    viewpoints = np.array([f.viewpoint for f in frontiers], dtype=np.float64)
    dist = np.linalg.norm(viewpoints - goal, axis=1)
    in_range = dist <= radius_m
    if not np.any(in_range):
        return None
    masked = np.where(in_range, dist, np.inf)
    idx = int(np.argmin(masked))
    if drone_cost_row[idx] >= _INFEASIBLE:
        return None
    return idx
