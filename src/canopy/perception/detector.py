"""The object detector: from sensed range and colour to classified, boxed objects.

It works in two stages, at two rates.

Every scan, :meth:`ObjectDetector.integrate` pools each coloured hit into the
evidence of every class whose colour rule and height band it passes. One hit
is one noisy sample, so it is turned away only when its colour is confidently
outside the rule (see :mod:`canopy.perception.colour`). Classes are
independent, so adding one never changes what another sees.

Every ``1 / extract_hz`` seconds, :meth:`ObjectDetector.extract` turns that
evidence into objects. A cell counts if its pooled colour is not confidently
outside the rule, a test that tightens as the cell's hits add up. Counted cells
are grouped into connected objects, and each object's colour, pooled over all
of its hits, must then pass the rule exactly. That last average is what tells
a blue-grey conduit from the neutral-grey breaker panel beside it, which no
single hit can. Each object is fitted with a tight upright box and reported
only if it passes every shape rule its class states: its extents, elongation,
height above the ground, and distance to another class (a meter must be on a
wall, a bush near the house). An object found again keeps its track id.

Colour narrows the candidates and geometry decides between them. The meter's
yellow is shared by the street's centre line, which lies flat on the ground;
the conduit's grey is shared by the panel and the AC unit, which are too thick
or stand off the wall. Nothing here sees what the simulator knows: the input is
an :class:`~canopy.contracts.Observation`, which carries no object or triangle
ids, and ``tests/perception/test_boundary.py`` fails if this package names them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from canopy.contracts import Cls, DiscoveredObject, Observation, OrientedBox, Points, Vec3
from canopy.perception.boxes import fit_box, pad_box, split_parts
from canopy.perception.cells import CellMap, label_cells
from canopy.perception.colour import ColourRule, rgb_to_hsv

if TYPE_CHECKING:
    from canopy.config import PerceptionCfg, PerceptionClassCfg

__all__ = ["ObjectDetector"]

IntArray = npt.NDArray[np.int64]

#: Largest value of an 8-bit colour channel.
_CHANNEL_MAX = 255.0
#: Fewest evidence cells either side of a cut that splits an object into parts.
_MIN_PART_CELLS = 3


@dataclass(frozen=True, slots=True, eq=False)
class _Candidate:
    """An object found by one extraction, before its track id is settled."""

    cls: Cls
    box: OrientedBox
    pos: Vec3
    n_hits: int
    first_seen_t: float
    first_seen_by: int


class _ClassModel:
    """One class's rules and the evidence pooled for it."""

    def __init__(self, spec: PerceptionClassCfg, lot_lo: Vec3, lot_hi: Vec3) -> None:
        self.spec = spec
        self.cls = Cls[spec.cls]
        self.colour = ColourRule(spec.hue_deg, spec.sat, spec.val)
        lo = np.array([lot_lo[0], lot_lo[1], spec.z_m[0]], dtype=np.float64)
        hi = np.array([lot_hi[0], lot_hi[1], spec.z_m[1]], dtype=np.float64)
        self.cells = CellMap(lo, hi, spec.cell_m)

    def counted(self, min_cell_hits: int, noise: float, tolerance: float) -> IntArray:
        """Rows of the cells that count: enough hits, and a colour not confidently wrong.

        A cell is judged on its own colour and, for a ``neighbour_colour``
        class, again pooled with its neighbours'
        (:meth:`CellMap.neighbourhood_sums`), which catches cells too sparse
        to judge alone. ``noise`` is one hit's per-channel standard deviation
        (``[0, 1]`` scale); a mean of ``n`` hits has ``1 / sqrt(n)`` of it.
        """
        rows = np.flatnonzero(self.cells.hits >= min_cell_hits)
        if rows.size == 0:
            return rows
        own = self.colour.admits(
            rgb_to_hsv(self.cells.mean_rgb(rows)),
            noise=noise / np.sqrt(self.cells.hits[rows]),
            tolerance=tolerance,
        )
        rows = rows[own]
        if rows.size == 0 or not self.spec.neighbour_colour:
            return rows
        rgb_sum, hits = self.cells.neighbourhood_sums(rows)
        local = self.colour.admits(
            rgb_to_hsv(rgb_sum / hits[:, None]), noise=noise / np.sqrt(hits), tolerance=tolerance
        )
        kept: IntArray = rows[local]
        return kept


class ObjectDetector:
    """Finds, classifies and boxes objects from what drones sense.

    Parameters
    ----------
    cfg
        The ``perception`` config section: shared tunables plus one entry per
        class, in :attr:`~canopy.config.PerceptionCfg.classes`.
    lot_bounds
        ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` of the surveyed lot. The
        parcel's extent is prior knowledge a real survey has too. Only ``x``
        and ``y`` are used, which keeps the neighbours' houses and the street
        out of the evidence.
    """

    def __init__(self, cfg: PerceptionCfg, lot_bounds: npt.NDArray[np.float64]) -> None:
        self._cfg = cfg
        bounds = np.asarray(lot_bounds, dtype=np.float64)
        self._models = tuple(_ClassModel(spec, bounds[0], bounds[1]) for spec in cfg.classes)
        self._by_name = {m.spec.cls: m for m in self._models}
        self._xy_lo = bounds[0, :2]
        self._xy_hi = bounds[1, :2]
        bands = [m.spec.z_m for m in self._models] or [(0.0, 0.0)]
        self._z_lo = min(lo for lo, _ in bands)
        self._z_hi = max(hi for _, hi in bands)
        # One hit's per-channel colour noise on the [0, 1] scale that HSV uses.
        self._noise = cfg.colour_noise / _CHANNEL_MAX
        self._objects: dict[int, DiscoveredObject] = {}
        self._next_track_id = 0
        # Set when evidence arrives, so an extraction with nothing new to say
        # returns the last result instead of redoing it.
        self._dirty = False

    @property
    def objects(self) -> dict[int, DiscoveredObject]:
        """Objects as of the last :meth:`extract`, by track id."""
        return self._objects

    def integrate(self, obs: Observation) -> None:
        """Pool one observation's coloured hits into the evidence of every class they match.

        Parameters
        ----------
        obs
            One ranger sweep. An observation without colour carries nothing
            to classify and is skipped.
        """
        if obs.rgb is None:
            return
        hit = np.isfinite(obs.dist) & (obs.dist <= self._cfg.max_range_m)
        points = obs.origin[None, :] + obs.dirs[hit] * obs.dist[hit, None]
        keep = (
            np.all((points[:, :2] >= self._xy_lo) & (points[:, :2] < self._xy_hi), axis=1)
            & (points[:, 2] >= self._z_lo)
            & (points[:, 2] < self._z_hi)
        )
        if not np.any(keep):
            return
        points = points[keep]
        rgb = obs.rgb[hit][keep]
        hsv = rgb_to_hsv(rgb)
        z = points[:, 2]
        for model in self._models:
            lo, hi = model.spec.z_m
            match = (
                (z >= lo)
                & (z < hi)
                & model.colour.admits(hsv, noise=self._noise, tolerance=self._cfg.colour_tolerance)
            )
            if np.any(match):
                model.cells.add(points[match], rgb[match], obs.t, obs.drone_id)
                self._dirty = True

    def extract(self) -> dict[int, DiscoveredObject]:
        """Group the pooled evidence into objects and keep those their class's rules accept.

        Returns
        -------
        dict
            Objects by track id. A new dict whenever anything changed; the one
            returned before is never mutated, so a reader may keep it as a
            snapshot.
        """
        if not self._dirty:
            return self._objects
        self._dirty = False
        counted = {
            m.spec.cls: m.counted(self._cfg.min_cell_hits, self._noise, self._cfg.colour_tolerance)
            for m in self._models
        }
        # One lookup per class that some rule measures distance to.
        context: dict[str, Any] = {}
        for name in {m.spec.near_cls for m in self._models if m.spec.near_cls is not None}:
            rows = counted[name]
            if rows.size:
                context[name] = cKDTree(self._by_name[name].cells.mean_pos(rows))
        candidates: list[_Candidate] = []
        for model in self._models:
            if model.spec.report:
                candidates.extend(self._find(model, counted[model.spec.cls], context))
        self._objects = self._track(candidates)
        return self._objects

    # -- internals ------------------------------------------------------------
    def _find(
        self, model: _ClassModel, rows: IntArray, context: dict[str, Any]
    ) -> list[_Candidate]:
        """Return the objects among ``model``'s counted cells that pass its rules.

        Cells are grouped by connectivity, and a group cut off by the top of
        its height band is dropped whole. A group shaped like an L is cut into
        straight parts (:func:`~canopy.perception.boxes.split_parts`), and each
        part is judged as an object of its own: its pooled colour first, since
        that is cheap and settles most junk, then its box's shape.
        """
        if rows.size == 0:
            return []
        spec = model.spec
        n_groups, labels = label_cells(
            model.cells.coords(rows), spec.link_cells, spec.link_up_cells
        )
        group_hits = np.bincount(labels, weights=model.cells.hits[rows], minlength=n_groups)
        found = []
        for group in np.flatnonzero(group_hits >= spec.min_hits):
            members = rows[labels == group]
            points = model.cells.mean_pos(members)
            # An object reaching the top of its band continues above it: only
            # the part inside the band was pooled, so it is not the object it
            # looks like (a tree's trunk is not a bush).
            if float(points[:, 2].max()) >= spec.z_m[1] - spec.cell_m:
                continue
            parts = split_parts(
                points,
                floor_m=spec.cell_m,
                min_saving=self._cfg.split_saving,
                min_points=_MIN_PART_CELLS,
                max_parts=self._cfg.max_parts,
            )
            for part in parts:
                cells = members[part]
                hits = model.cells.hits[cells]
                if hits.sum() < spec.min_hits or not _coloured(model, cells):
                    continue
                box = fit_box(points[part])
                if not _shaped(model, box, points[part], context):
                    continue
                first = int(np.argmin(model.cells.first_t[cells]))
                found.append(
                    _Candidate(
                        cls=model.cls,
                        box=pad_box(box, self._cfg.box_margin_m, self._cfg.ground_snap_m),
                        pos=np.average(points[part], axis=0, weights=hits),
                        n_hits=int(hits.sum()),
                        first_seen_t=float(model.cells.first_t[cells][first]),
                        first_seen_by=int(model.cells.first_by[cells][first]),
                    )
                )
        return found

    def _track(self, candidates: list[_Candidate]) -> dict[int, DiscoveredObject]:
        """Give each candidate the track id of the nearest same-class object from last time.

        Matching is greedy, closest pair first, within ``track_match_m`` of box
        centres. A box's centre drifts as more sides of its object are seen,
        by far less than the gap between two separate objects of one class.
        """
        previous = list(self._objects.values())
        pairs = sorted(
            (float(np.linalg.norm(cand.box.center - old.box.center)), i, j)
            for i, cand in enumerate(candidates)
            for j, old in enumerate(previous)
            if old.cls is cand.cls
        )
        track_ids = [-1] * len(candidates)
        taken: set[int] = set()
        for dist, i, j in pairs:
            if dist > self._cfg.track_match_m:
                break
            if track_ids[i] < 0 and j not in taken:
                track_ids[i] = previous[j].track_id
                taken.add(j)

        objects: dict[int, DiscoveredObject] = {}
        for matched, cand in zip(track_ids, candidates, strict=True):
            track_id = matched if matched >= 0 else self._new_track_id()
            objects[track_id] = DiscoveredObject(
                track_id=track_id,
                cls=cand.cls,
                pos=cand.pos,
                box=cand.box,
                n_hits=cand.n_hits,
                confidence=float(1.0 - np.exp(-cand.n_hits / self._cfg.confidence_hits)),
                first_seen_t=cand.first_seen_t,
                first_seen_by=cand.first_seen_by,
            )
        return objects

    def _new_track_id(self) -> int:
        """Allocate the next unused track id; ids are never reused."""
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id


def _coloured(model: _ClassModel, cells: IntArray) -> bool:
    """Whether an object's colour, pooled over all of its hits, passes its class's rule exactly.

    Hundreds of hits average away the noise that single hits and cells were
    allowed. This is judged per part, after an L-shaped group is split: a
    conduit and the strip of breaker panel it touches pool to a passing grey
    together, while the strip alone is plainly neutral.
    """
    pooled = model.cells.rgb_sum[cells].sum(axis=0) / model.cells.hits[cells].sum()
    return bool(model.colour.admits(rgb_to_hsv(pooled[None, :]))[0])


def _shaped(model: _ClassModel, box: OrientedBox, points: Points, context: dict[str, Any]) -> bool:
    """Whether one boxed object passes its class's shape rules and its ``near_cls`` rule."""
    spec = model.spec
    length, width, thickness = np.sort(box.size)[::-1]
    height = float(box.size[2])
    bottom = float(box.center[2]) - height / 2.0
    if not (
        _within(spec.length_m, length)
        and _within(spec.width_m, width)
        and _within(spec.thickness_m, thickness)
        and _within(spec.height_m, height)
        and _within(spec.bottom_m, bottom)
    ):
        return False
    if spec.min_elongation is not None and length < spec.min_elongation * width:
        return False
    if spec.near_cls is None or spec.near_m is None:
        return True
    tree = context.get(spec.near_cls)
    if tree is None:
        return False
    dist, _ = tree.query(points, k=1, distance_upper_bound=spec.near_m)
    return bool(np.any(np.isfinite(dist)))


def _within(band: tuple[float, float] | None, value: float) -> bool:
    """Whether ``value`` lies in the closed ``band``; an absent band admits anything."""
    return band is None or band[0] <= value <= band[1]
