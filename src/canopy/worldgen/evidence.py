"""What an address's map data pins down, in the form the placement rules consume.

A :class:`~canopy.contracts.SiteSnapshot` says where a real house stands, how
its footprint is shaped, who its neighbours are and where the street runs. The
generator does not get a second builder for that: every role in the model
library already exists, and an address only changes where a role's parameters
come from. :func:`site_evidence` does the translation, and the one piece of
real geometry it needs is fitting a mapped footprint -- any simple polygon --
with the one to three axis-aligned :class:`~canopy.worldgen.placement.Mass`
blocks the house rule composes (:func:`regularize_footprint`).

The same fit decides whether a house can be rebuilt at all
(:func:`check_buildable`), and the fetch step refuses an address on it before
anything is saved. That is why this module sits below the network code rather
than inside it: both the online fetch and the offline build read it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from canopy.errors import SiteRejectedError, WorldgenError
from canopy.worldgen.placement import HouseFrame, Mass, Obstacle

if TYPE_CHECKING:
    from canopy.config import SiteCfg
    from canopy.contracts import Points2, SiteBuilding, SiteSnapshot

__all__ = [
    "FootprintFit",
    "RoofKind",
    "SiteEvidence",
    "check_buildable",
    "dominant_orientation",
    "levels_for",
    "regularize_footprint",
    "roof_kind",
    "site_evidence",
]

#: The two roof forms the model library can build.
RoofKind = Literal["gable", "flat"]

#: Mapped roof shapes the library builds exactly. Everything else pitched is
#: built as a gable and noted, since a gable is the closest silhouette.
_EXACT_ROOFS: dict[str, RoofKind] = {"gabled": "gable", "flat": "flat"}

#: Below this, an edge vote is noise from a duplicated vertex.
_EPS_M = 1e-9

#: Fraction of a wall's outward samples that must land inside a neighbour for
#: the wall to count as shared. Half, so a neighbour that covers most of the
#: wall claims it and a corner that merely brushes it does not.
_PARTY_WALL_FRACTION = 0.5

#: Samples along each wall for the party-wall test, at the midpoints of equal
#: stretches so neither corner is sampled.
_PARTY_WALL_SAMPLES = 9


@dataclass(frozen=True, slots=True, eq=False)
class FootprintFit:
    """A mapped footprint approximated by axis-aligned blocks."""

    blocks: tuple[Mass, ...]
    """One to three blocks, largest first, each touching an earlier one along a
    wall. Heights are zero: the house rule sets them from the storey count."""
    iou: float
    """Intersection over union of the blocks' union and the mapped polygon."""


def _rotation(angle_rad: float) -> npt.NDArray[np.float64]:
    """2D rotation matrix, anticlockwise by ``angle_rad``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def dominant_orientation(polygon: Points2) -> float:
    """Return the direction a footprint's walls mostly run, modulo a quarter turn.

    Each edge votes for its own direction, weighted by its length, after the
    angle is multiplied by four: that folds walls at right angles to each other
    onto one vote, so an L-shaped plan's two wall directions agree instead of
    cancelling. The result is the circular mean of the votes, divided back.

    Parameters
    ----------
    polygon
        Outline, shape ``(N, 2)``, either winding, first point not repeated.

    Returns
    -------
    float
        Radians in ``(-pi/4, pi/4]``. Turning the footprint by the negative of
        this lines its walls up with the axes. ``0.0`` for a degenerate outline.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    c = float(np.sum(lengths * np.cos(4.0 * angles)))
    s = float(np.sum(lengths * np.sin(4.0 * angles)))
    if math.hypot(c, s) < _EPS_M:
        return 0.0
    return math.atan2(s, c) / 4.0


def _largest_rectangle(
    mask: npt.NDArray[np.bool_], min_rows: int, min_cols: int
) -> tuple[int, int, int, int] | None:
    """Largest all-true axis-aligned rectangle in ``mask`` at least the given size.

    The classic histogram sweep: row by row, each column's height is how many
    true cells stand above it, and a stack finds, for every bar, the widest
    span it is the shortest bar of. Any rectangle meeting the minimums lies
    inside the widest span of its own shortest bar, which has the same height
    and at least the same width -- so checking only those spans, filtered by
    the minimums, still finds the best rectangle that meets them.

    Returns
    -------
    tuple[int, int, int, int] | None
        ``(row0, row1, col0, col1)``, half-open, or ``None`` if nothing qualifies.
    """
    rows, cols = mask.shape
    heights = np.zeros(cols + 1, dtype=np.int64)  # trailing zero flushes the stack
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for r in range(rows):
        heights[:cols] = np.where(mask[r], heights[:cols] + 1, 0)
        stack: list[int] = []
        for c in range(cols + 1):
            h = int(heights[c])
            while stack and int(heights[stack[-1]]) >= h:
                top_h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                width = c - left
                area = top_h * width
                if top_h >= min_rows and width >= min_cols and area > best_area:
                    best_area = area
                    best = (r - top_h + 1, r + 1, left, c)
            stack.append(c)
    return best


def regularize_footprint(
    polygon: Points2, *, raster_m: float, max_masses: int, min_mass_m: float
) -> FootprintFit:
    """Fit a footprint with up to ``max_masses`` axis-aligned blocks.

    The polygon is rasterised at ``raster_m`` and blocks are taken greedily:
    the largest rectangle of cells inside the outline first, then the largest
    rectangle of what is left that shares at least ``min_mass_m`` of wall with
    the blocks so far, so the union stays one connected plan the way the house
    rule builds it. Blocks narrower than ``min_mass_m`` either way are bay
    windows and porches, not wings, and are left out.

    Parameters
    ----------
    polygon
        Outline, shape ``(N, 2)``, already turned so its walls run along the
        axes (see :func:`dominant_orientation`).
    raster_m
        Cell size. Block edges land on this grid.
    max_masses
        Most blocks to fit.
    min_mass_m
        Narrowest block, and the least wall a block must share.

    Returns
    -------
    FootprintFit
        The blocks, in the polygon's own frame, and how well they fit it.

    Raises
    ------
    WorldgenError
        If the outline is degenerate or no block of the minimum size fits it.
    """
    geom = Polygon(np.asarray(polygon, dtype=np.float64)).buffer(0.0)
    if geom.is_empty or geom.area <= 0.0:
        msg = f"footprint with {len(polygon)} vertices has no area to fit blocks to"
        raise WorldgenError(msg)

    minx, miny, maxx, maxy = geom.bounds
    n_cols = max(math.ceil((maxx - minx) / raster_m), 1)
    n_rows = max(math.ceil((maxy - miny) / raster_m), 1)
    xs = minx + (np.arange(n_cols) + 0.5) * raster_m
    ys = miny + (np.arange(n_rows) + 0.5) * raster_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(geom, grid_x, grid_y)

    min_cells = max(math.ceil(min_mass_m / raster_m - 1e-9), 1)
    covered = np.zeros_like(inside)
    ignored = np.zeros_like(inside)
    cells: list[tuple[int, int, int, int]] = []
    while len(cells) < max_masses:
        found = _largest_rectangle(inside & ~covered & ~ignored, min_cells, min_cells)
        if found is None:
            break
        r0, r1, c0, c1 = found
        if cells and _shared_cells(covered, found) < min_cells:
            # Out of reach of the plan so far (attached by a sliver, or not at
            # all); skip it rather than let the union fall apart.
            ignored[r0:r1, c0:c1] = True
            continue
        cells.append(found)
        covered[r0:r1, c0:c1] = True

    if not cells:
        msg = (
            f"no {min_mass_m} m block fits inside the footprint "
            f"({maxx - minx:.1f} x {maxy - miny:.1f} m, {geom.area:.0f} m^2)"
        )
        raise WorldgenError(msg)

    blocks = tuple(
        Mass(
            centre=np.array(
                [minx + (c0 + c1) * raster_m / 2.0, miny + (r0 + r1) * raster_m / 2.0],
                dtype=np.float64,
            ),
            size=np.array([(c1 - c0) * raster_m, (r1 - r0) * raster_m, 0.0], dtype=np.float64),
        )
        for r0, r1, c0, c1 in cells
    )
    union = unary_union([box(*b.bounds_xy) for b in blocks])
    iou = float(union.intersection(geom).area / union.union(geom).area)
    return FootprintFit(blocks=blocks, iou=iou)


def _shared_cells(covered: npt.NDArray[np.bool_], rect: tuple[int, int, int, int]) -> int:
    """Longest run of cells along which ``rect`` borders ``covered``, in cells.

    Counted on each of the rectangle's four sides separately, so two blocks
    that meet only at a corner share nothing.
    """
    r0, r1, c0, c1 = rect
    rows, cols = covered.shape
    runs = [
        int(covered[r0:r1, c0 - 1].sum()) if c0 > 0 else 0,
        int(covered[r0:r1, c1].sum()) if c1 < cols else 0,
        int(covered[r0 - 1, c0:c1].sum()) if r0 > 0 else 0,
        int(covered[r1, c0:c1].sum()) if r1 < rows else 0,
    ]
    return max(runs)


def levels_for(building: SiteBuilding, cfg: SiteCfg) -> int | None:
    """Storeys above ground: as mapped, else counted from the mapped height.

    A total height includes the roof, so :attr:`SiteCfg.roof_allowance_m` comes
    off before dividing by the storey height. ``None`` when the data gives
    neither.
    """
    if building.levels is not None:
        return max(int(building.levels), 1)
    if building.height_m is not None:
        walls = building.height_m - cfg.roof_allowance_m
        return max(round(walls / cfg.storey_m), 1)
    return None


def roof_kind(roof_shape: str | None) -> tuple[RoofKind | None, str | None]:
    """Map an OpenStreetMap roof shape onto a roof the library can build.

    Returns
    -------
    tuple[RoofKind | None, str | None]
        The roof to build (``None`` if unmapped, so the rule samples one), and
        a note to record when the mapped shape had to be approximated.
    """
    if roof_shape is None:
        return None, None
    shape = roof_shape.strip().lower()
    if exact := _EXACT_ROOFS.get(shape):
        return exact, None
    return "gable", f"roof mapped as {shape!r}, built as a gable"


def check_buildable(building: SiteBuilding, cfg: SiteCfg) -> FootprintFit:
    """Fit the house and refuse it if the generator cannot represent it.

    Parameters
    ----------
    building
        The house, with its footprint already turned to the world frame.
    cfg
        The ``worldgen.site`` settings holding the limits.

    Returns
    -------
    FootprintFit
        The fit, for the caller to build from or to note.

    Raises
    ------
    SiteRejectedError
        If the footprint is larger than ``max_footprint_m2``, the house has more
        storeys than ``max_levels``, or the blocks fit it worse than
        ``min_fit_iou``. The message names the value and the setting.
    """
    area = float(Polygon(building.footprint).buffer(0.0).area)
    if area > cfg.max_footprint_m2:
        msg = (
            f"the house footprint is {area:.0f} m^2, larger than "
            f"worldgen.site.max_footprint_m2 = {cfg.max_footprint_m2:.0f}, the largest single "
            "house the generator rebuilds (an apartment block, a commercial building or a "
            "large house with outbuildings attached all land here)"
        )
        raise SiteRejectedError(msg)
    levels = levels_for(building, cfg)
    if levels is not None and levels > cfg.max_levels:
        msg = (
            f"the house has {levels} storeys, more than worldgen.site.max_levels = "
            f"{cfg.max_levels} the model library can build"
        )
        raise SiteRejectedError(msg)
    try:
        fit = regularize_footprint(
            building.footprint,
            raster_m=cfg.raster_m,
            max_masses=cfg.max_masses,
            min_mass_m=cfg.min_mass_m,
        )
    except WorldgenError as exc:
        raise SiteRejectedError(str(exc)) from exc
    if fit.iou < cfg.min_fit_iou:
        msg = (
            f"{len(fit.blocks)} axis-aligned blocks cover the footprint with IoU "
            f"{fit.iou:.2f}, below worldgen.site.min_fit_iou = {cfg.min_fit_iou}; the plan "
            "is too irregular to rebuild faithfully"
        )
        raise SiteRejectedError(msg)
    return fit


@dataclass(frozen=True, slots=True, eq=False)
class SiteEvidence:
    """A snapshot in the form the placement rules read (``PlacementContext.site``)."""

    snapshot: SiteSnapshot
    blocks: tuple[Mass, ...]
    """The house footprint as blocks, in the world frame, zero height."""
    fit_iou: float
    levels: int | None
    """Storeys, as mapped or counted from a mapped height; ``None`` to sample."""
    roof: RoofKind | None
    """Roof to build; ``None`` to sample."""
    notes: tuple[str, ...]
    """Approximations made here, on top of the snapshot's own."""
    neighbours: shapely.Geometry
    """Union of the neighbours' footprints, for the party-wall test."""

    def lot_obstacles(
        self, lot_bounds: npt.NDArray[np.float64], trunk_radius_m: float
    ) -> tuple[Obstacle, ...]:
        """Discs the surveyed lot's rules must keep clear of from the very first role.

        Mapped trees are placed by the tree role, which runs after the bushes
        and the wall-mounted equipment, so without this a bush could be drawn
        onto a real trunk. And a neighbour's footprint can poke into an
        inferred lot at a corner; the lot's rules only know what they placed
        themselves, so each such intrusion is covered by one disc around its
        bounding box -- coarse, but on the safe side.
        """
        lot = box(*lot_bounds[0, :2], *lot_bounds[1, :2])
        trees = np.asarray(self.snapshot.trees, dtype=np.float64).reshape(-1, 3)
        found = [
            Obstacle(xy=xy.copy(), radius_m=trunk_radius_m)
            for xy in trees[:, :2]
            if lot.contains(Point(float(xy[0]), float(xy[1])))
        ]
        intrusion = self.neighbours.intersection(lot)
        parts = getattr(intrusion, "geoms", [intrusion])
        for part in parts:
            if part.is_empty or part.area <= 0.0:
                continue
            minx, miny, maxx, maxy = part.bounds
            found.append(
                Obstacle(
                    xy=np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0], dtype=np.float64),
                    radius_m=math.hypot(maxx - minx, maxy - miny) / 2.0,
                )
            )
        return tuple(found)

    def without_party_walls(self, frame: HouseFrame, gap_m: float) -> HouseFrame:
        """Drop the walls the house shares with a neighbour.

        A semi-detached house or a terrace has walls that are not exterior at
        all: nothing can be mounted on them, planted against them or glazed.
        A wall counts as shared when most points just outside it, ``gap_m``
        out, fall inside a mapped neighbour.
        """
        if self.neighbours.is_empty or not frame.walls:
            return frame
        t = (np.arange(_PARTY_WALL_SAMPLES) + 0.5) / _PARTY_WALL_SAMPLES
        kept = []
        for wall in frame.walls:
            probes = wall.a + np.outer(t, wall.b - wall.a) + wall.normal * gap_m
            hits = shapely.contains_xy(self.neighbours, probes[:, 0], probes[:, 1])
            if float(np.mean(hits)) < _PARTY_WALL_FRACTION:
                kept.append(wall)
        return dataclasses.replace(frame, walls=tuple(kept))


def site_evidence(snapshot: SiteSnapshot, cfg: SiteCfg) -> SiteEvidence:
    """Read a snapshot into :class:`SiteEvidence`.

    Raises
    ------
    SiteRejectedError
        If the house cannot be rebuilt (see :func:`check_buildable`). The fetch
        step already refused such a snapshot; this guards one edited by hand or
        checked against a stricter configuration.
    """
    fit = check_buildable(snapshot.house, cfg)
    roof, roof_note = roof_kind(snapshot.house.roof_shape)
    notes: list[str] = []
    if fit.iou < cfg.warn_fit_iou:
        notes.append(
            f"house footprint approximated by {len(fit.blocks)} block(s) at IoU {fit.iou:.2f}"
        )
    if roof_note is not None:
        notes.append(roof_note)
    outlines = [Polygon(b.footprint).buffer(0.0) for b in snapshot.neighbours]
    neighbours = (
        unary_union([p for p in outlines if not p.is_empty]) if outlines else MultiPolygon()
    )
    return SiteEvidence(
        snapshot=snapshot,
        blocks=fit.blocks,
        fit_iou=fit.iou,
        levels=levels_for(snapshot.house, cfg),
        roof=roof,
        notes=tuple(notes),
        neighbours=neighbours,
    )
