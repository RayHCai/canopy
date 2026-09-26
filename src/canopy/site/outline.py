"""The house's walls, traced from the swarm's occupancy map.

A battery hangs on an exterior wall, so siting starts from where the walls
are. The site stage may not read the scene manifest's footprint (that is
ground truth), and perception does not report walls as objects -- ``WALL`` is
a context-only class, and a whole house would come back as a few coarse boxes
anyway. What the swarm *has* mapped is occupancy, and in occupancy a wall has a
signature nothing else on a property shares: it is solid all the way up a
storey.

So a plan column counts as wall when most of its voxels in a band above head
height (``outline.band_z_m``) are occupied. That band is chosen to miss
everything that crowds a wall at ground level -- bushes, the AC unit, the
meter, a fence -- while every house wall, at least one 3 m storey, fills it.
Tree crowns start above their trunks and trunks are too thin to connect, so
the wall columns split into one component per building, and the house is the
component nearest the meter.

That component is a one- or two-voxel ring with gaps where the map is thin.
A morphological closing (``outline.close_m``) bridges the gaps, filling the
ring's interior turns it into a footprint, and its boundary is simplified to a
polygon (``outline.simplify_m``) whose edges are the walls. The traced
boundary runs along the voxel faces outside the wall; each wall is moved in
half a voxel, the expected position of a surface somewhere inside its voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import measure

from canopy.contracts import MapState, Occ, Points, Vec3
from canopy.errors import SiteError

if TYPE_CHECKING:
    from canopy.site.rules import OutlineSpec

__all__ = ["HouseOutline", "Wall", "trace_house"]

#: 8-connectivity in plan: a wall running diagonally across the grid is one wall.
_PLAN_8 = np.ones((3, 3), dtype=bool)

#: Fewest corners a footprint polygon can have.
_MIN_CORNERS = 3


@dataclass(frozen=True, slots=True, eq=False)
class Wall:
    """One straight exterior wall, seen from above."""

    start: npt.NDArray[np.float64]
    """Plan point, shape ``(2,)``. Walls run anticlockwise around the house."""
    end: npt.NDArray[np.float64]
    """Plan point, shape ``(2,)``."""
    normal: npt.NDArray[np.float64]
    """Outward unit normal in plan, shape ``(2,)``."""

    @property
    def length(self) -> float:
        """Wall length in metres."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def tangent(self) -> npt.NDArray[np.float64]:
        """Unit direction from ``start`` to ``end``."""
        return (self.end - self.start) / self.length


@dataclass(frozen=True, slots=True, eq=False)
class HouseOutline:
    """The traced house: its footprint polygon and the walls along it.

    A harness is a cable fastened to the wall, so several placement rules
    (``harness_run``) need distance *along the perimeter*, not as the crow
    flies. That perimeter is parameterised by a single arc coordinate ``s``,
    zero at the start of ``walls[0]`` and increasing anticlockwise (the same
    direction the walls run in) up to :attr:`perimeter`, where it wraps.
    """

    polygon: Points
    """Footprint corners in plan, anticlockwise, shape ``(K, 2)``, not closed."""
    walls: tuple[Wall, ...]
    """One per polygon edge, in the same order."""

    @property
    def perimeter(self) -> float:
        """Total length of the traced walls, metres."""
        return float(sum(wall.length for wall in self.walls))

    def _wall_starts(self) -> npt.NDArray[np.float64]:
        """Arc coordinate of each wall's start, cumulative around the ring."""
        lengths = np.array([wall.length for wall in self.walls])
        return np.concatenate([[0.0], np.cumsum(lengths)[:-1]])

    def arc_of(
        self, points: Points
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project plan points onto the nearest wall.

        Looping over walls rather than points is deliberate: a house has a
        handful of walls and callers pass anywhere from one point (the meter)
        to every candidate site, so this stays vectorised over the arg that
        can be large.

        Parameters
        ----------
        points
            Plan points, shape ``(P, 2)``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(wall_index, s, perp_dist)``, each shape ``(P,)``: the nearest
            wall, that wall's arc coordinate at the projection, and the
            perpendicular distance from the point to the wall's line
            (unsigned, so a point on either side reads the same).
        """
        starts = self._wall_starts()
        n = len(points)
        best_wall = np.zeros(n, dtype=np.int64)
        best_s = np.zeros(n)
        best_d = np.full(n, np.inf)
        for i, wall in enumerate(self.walls):
            rel = points - wall.start
            along = np.clip(rel @ wall.tangent, 0.0, wall.length)
            perp = np.abs(rel @ wall.normal)
            closer = perp < best_d
            best_wall = np.where(closer, i, best_wall)
            best_s = np.where(closer, starts[i] + along, best_s)
            best_d = np.where(closer, perp, best_d)
        return best_wall, best_s, best_d

    def point_at(self, s: npt.NDArray[np.float64]) -> Points:
        """Return the plan point at each arc coordinate, wrapping at the perimeter."""
        wrapped = np.mod(s, self.perimeter)
        starts = self._wall_starts()
        idx = np.clip(np.searchsorted(starts, wrapped, side="right") - 1, 0, len(self.walls) - 1)
        wall_start = np.array([w.start for w in self.walls])[idx]
        tangent = np.array([w.tangent for w in self.walls])[idx]
        along = wrapped - starts[idx]
        result: Points = wall_start + along[:, None] * tangent
        return result

    def walk(self, s_from: float, s_to: float) -> Points:
        """Polyline from ``s_from`` to ``s_to``, walking increasing ``s`` (anticlockwise).

        Wraps at the perimeter, so this is always one specific direction
        round the ring; the other direction is ``walk(s_to, s_from)``
        reversed. Every corner crossed becomes a vertex, so the result
        follows the walls rather than cutting across the house.
        """
        perimeter = self.perimeter
        start = float(np.mod(s_from, perimeter))
        length = float(np.mod(s_to - s_from, perimeter))
        starts = self._wall_starts()
        ends = starts + np.array([w.length for w in self.walls])
        # Offset of each wall's far corner from the walk's start, in [0, perimeter).
        corner_offsets = np.mod(ends - start, perimeter)
        passed = np.sort(corner_offsets[corner_offsets < length])
        offsets = np.concatenate([[0.0], passed, [length]])
        return self.point_at(start + offsets)


def trace_house(state: MapState, near: Vec3, spec: OutlineSpec) -> HouseOutline:
    """Trace the walls of the building nearest ``near`` from the occupancy map.

    Parameters
    ----------
    state
        The swarm's map. Only :attr:`~canopy.contracts.MapState.occ` is read.
    near
        A point on or beside the house, such as the meter. World frame.
    spec
        Tracing parameters (``rules.yaml`` ``outline``).

    Returns
    -------
    HouseOutline
        At least three walls, anticlockwise.

    Raises
    ------
    SiteError
        If no wall was mapped within ``spec.max_meter_gap_m`` of ``near``, or
        the walls found there do not close into a footprint.
    """
    v = state.voxel
    origin = np.asarray(state.origin, dtype=np.float64)
    k0 = max(int(np.ceil((spec.band_z_m[0] - origin[2]) / v)), 0)
    k1 = min(int(np.floor((spec.band_z_m[1] - origin[2]) / v)), state.occ.shape[2])
    fill = (state.occ[:, :, k0:k1] == Occ.OCC).mean(axis=2)
    labels, _ = ndimage.label(fill >= spec.min_fill, structure=_PLAN_8)

    cells = np.argwhere(labels > 0)
    if not len(cells):
        msg = f"no wall was mapped between z={spec.band_z_m[0]} and z={spec.band_z_m[1]} m"
        raise SiteError(msg)
    gaps = np.linalg.norm(origin[:2] + (cells + 0.5) * v - np.asarray(near)[:2], axis=1)
    nearest = int(np.argmin(gaps))
    if gaps[nearest] > spec.max_meter_gap_m:
        msg = (
            f"the nearest mapped wall is {gaps[nearest]:.2f} m from {np.round(near, 2).tolist()}, "
            f"beyond outline.max_meter_gap_m={spec.max_meter_gap_m}"
        )
        raise SiteError(msg)
    house = labels == labels[tuple(cells[nearest])]

    # Pad so the closing cannot erode against the array edge.
    steps = max(int(np.ceil(spec.close_m / v)), 1)
    pad = steps + 1
    closed = ndimage.binary_closing(np.pad(house, pad), structure=_PLAN_8, iterations=steps)
    footprint = ndimage.binary_fill_holes(closed)
    contours = measure.find_contours(footprint.astype(np.float64), 0.5)
    ring = measure.approximate_polygon(max(contours, key=len), tolerance=spec.simplify_m / v)[:-1]
    if len(ring) < _MIN_CORNERS:
        msg = f"the walls near {np.round(near, 2).tolist()} do not close into a footprint"
        raise SiteError(msg)
    polygon: Points = origin[:2] + (ring - pad + 0.5) * v
    x, y = polygon[:, 0], polygon[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0.0:  # clockwise
        polygon = polygon[::-1].copy()

    walls = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        tangent = (end - start) / np.linalg.norm(end - start)
        normal = np.array([tangent[1], -tangent[0]])
        inset = normal * (v / 2.0)
        walls.append(Wall(start=start - inset, end=end - inset, normal=normal))
    return HouseOutline(polygon=polygon, walls=tuple(walls))
