"""Sparse evidence cells, and grouping them into connected objects.

A class's evidence is pooled into small cubes rather than kept as raw hits:
thousands of hits land on a bush every second, and pooling keeps the memory
bounded by the surface actually seen instead of by the flight time. Each
occupied cell keeps a hit count and running sums of position and colour, so
its mean position stays finer than the cell and its mean colour averages the
sensor noise away.

Cells live in parallel arrays sorted by a flat key. That keeps each update to
one sort of the incoming batch plus a binary search into the existing keys,
with no Python loop over hits or cells. A dense grid at a conduit's 4 cm
resolution would be tens of millions of cells for one lot; the sparse one holds
only the few thousand the evidence touched.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from canopy.contracts import Points, Vec3

__all__ = ["CellMap", "label_cells"]

IntArray = npt.NDArray[np.int64]


class CellMap:
    """Hit evidence pooled into cubes of edge ``cell_m`` over the box ``[lo, hi)``.

    Parameters
    ----------
    lo, hi
        Opposite corners of the region covered, world metres. Hits outside it
        are ignored.
    cell_m
        Cube edge in metres.
    """

    def __init__(self, lo: Vec3, hi: Vec3, cell_m: float) -> None:
        self.cell_m = float(cell_m)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.shape = np.maximum(
            np.ceil((np.asarray(hi, dtype=np.float64) - self.lo) / self.cell_m), 1
        ).astype(np.int64)
        self.keys: IntArray = np.zeros(0, dtype=np.int64)
        """Flat cell keys, ascending, shape ``(M,)``."""
        self.hits: IntArray = np.zeros(0, dtype=np.int64)
        """Hits pooled into each cell, shape ``(M,)``."""
        self.pos_sum: Points = np.zeros((0, 3), dtype=np.float64)
        """Sum of hit positions per cell, shape ``(M, 3)``."""
        self.rgb_sum: Points = np.zeros((0, 3), dtype=np.float64)
        """Sum of hit colours per cell (0-255 scale), shape ``(M, 3)``."""
        self.first_t: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        """Time of each cell's first hit, shape ``(M,)``."""
        self.first_by: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        """Drone that made each cell's first hit, shape ``(M,)``."""

    def __len__(self) -> int:
        """Return the number of occupied cells."""
        return int(self.keys.size)

    def add(self, points: Points, rgb: npt.NDArray[np.uint8], t: float, drone_id: int) -> None:
        """Pool one batch of hits, all sensed by one drone at one instant.

        Parameters
        ----------
        points
            Hit positions, shape ``(N, 3)``.
        rgb
            Their sensed colours, shape ``(N, 3)``.
        t, drone_id
            When and by whom, recorded for cells this batch creates.
        """
        idx = np.floor((points - self.lo) / self.cell_m).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=1)
        if not np.any(inside):
            return
        idx, points, colours = idx[inside], points[inside], rgb[inside].astype(np.float64)

        batch_keys, inverse = np.unique(self._encode(idx), return_inverse=True)
        counts = np.bincount(inverse, minlength=batch_keys.size).astype(np.int64)
        pos = np.stack([np.bincount(inverse, points[:, a], batch_keys.size) for a in range(3)], 1)
        col = np.stack([np.bincount(inverse, colours[:, a], batch_keys.size) for a in range(3)], 1)

        slot = np.searchsorted(self.keys, batch_keys)
        known = slot < self.keys.size
        known[known] = self.keys[slot[known]] == batch_keys[known]
        rows = slot[known]
        self.hits[rows] += counts[known]
        self.pos_sum[rows] += pos[known]
        self.rgb_sum[rows] += col[known]

        fresh = ~known
        if np.any(fresh):
            # np.insert places every new row before the existing row it sorts
            # ahead of, all positions taken against the old arrays, so the
            # keys stay sorted without a re-sort.
            at = slot[fresh]
            n_new = int(fresh.sum())
            self.keys = np.insert(self.keys, at, batch_keys[fresh])
            self.hits = np.insert(self.hits, at, counts[fresh])
            self.pos_sum = np.insert(self.pos_sum, at, pos[fresh], axis=0)
            self.rgb_sum = np.insert(self.rgb_sum, at, col[fresh], axis=0)
            self.first_t = np.insert(self.first_t, at, np.full(n_new, t))
            self.first_by = np.insert(self.first_by, at, np.full(n_new, drone_id, dtype=np.int32))

    def coords(self, rows: IntArray | None = None) -> IntArray:
        """Integer cell coordinates of the given rows (all rows when omitted), shape ``(M, 3)``."""
        keys = self.keys if rows is None else self.keys[rows]
        ny, nz = int(self.shape[1]), int(self.shape[2])
        return np.stack([keys // (ny * nz), (keys // nz) % ny, keys % nz], axis=1)

    def neighbourhood_sums(self, rows: IntArray) -> tuple[Points, IntArray]:
        """Colour sums and hit counts of each given cell plus its 26 neighbours.

        A surface's colour is locally constant, and one small cell rarely holds
        enough hits to judge a pale colour alone; judged with its neighbours it
        gets several times the evidence. Where two differently coloured objects
        nearly touch, the cells between them pool both colours and fail both
        classes' rules, which is what keeps the two apart.

        Parameters
        ----------
        rows
            Cells to pool around, ascending, shape ``(M,)``.

        Returns
        -------
        tuple
            Summed colour (0-255 scale), shape ``(M, 3)``, and summed hits,
            shape ``(M,)``, each over the cell and whichever neighbours exist.
        """
        rgb = np.zeros((rows.size, 3), dtype=np.float64)
        hits = np.zeros(rows.size, dtype=np.int64)
        for at, neighbour in self._neighbours(rows):
            rgb[at] += self.rgb_sum[neighbour]
            hits[at] += self.hits[neighbour]
        return rgb, hits

    def _neighbours(self, rows: IntArray) -> Iterator[tuple[IntArray, IntArray]]:
        """For each of the 27 offsets (itself included), which rows have a cell there, and its row.

        Yields
        ------
        tuple
            Positions within ``rows``, and the row of the occupied cell at
            that offset from each.
        """
        coords = self.coords(rows)
        keys = self.keys[rows]
        ny, nz = int(self.shape[1]), int(self.shape[2])
        # Per axis, which rows may step -1, 0 or +1 without leaving the grid;
        # an in-grid step is then a constant offset on the flat key.
        can_step = [
            {-1: coords[:, a] > 0, 0: None, 1: coords[:, a] < self.shape[a] - 1} for a in range(3)
        ]
        last = self.keys.size - 1
        for di, dj, dk in itertools.product((-1, 0, 1), repeat=3):
            # Rows ascend, so these queries do too, which keeps the binary
            # search cheap.
            query = keys + (di * ny + dj) * nz + dk
            slot = np.minimum(np.searchsorted(self.keys, query), last)
            found = self.keys[slot] == query
            for axis, step in enumerate((di, dj, dk)):
                limit = can_step[axis][step]
                if limit is not None:
                    found &= limit
            yield np.flatnonzero(found), slot[found]

    def mean_pos(self, rows: IntArray | None = None) -> Points:
        """Mean hit position of each given cell, shape ``(M, 3)``."""
        sl = slice(None) if rows is None else rows
        mean: Points = self.pos_sum[sl] / self.hits[sl, None]
        return mean

    def mean_rgb(self, rows: IntArray | None = None) -> Points:
        """Mean sensed colour of each given cell (0-255 scale), shape ``(M, 3)``."""
        sl = slice(None) if rows is None else rows
        mean: Points = self.rgb_sum[sl] / self.hits[sl, None]
        return mean

    def _encode(self, idx: IntArray) -> IntArray:
        ny, nz = int(self.shape[1]), int(self.shape[2])
        key: IntArray = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
        return key


def label_cells(
    coords: IntArray, link_cells: int, link_up_cells: int | None = None
) -> tuple[int, IntArray]:
    """Group cells into connected components.

    Two cells are connected when their horizontal coordinates differ by at
    most ``link_cells`` and their heights by at most ``link_up_cells``. At 1
    and 1 that is ordinary 26-connectivity; larger values bridge the gaps a
    sparsely sampled thin object leaves between its hits.

    Parameters
    ----------
    coords
        Integer cell coordinates, shape ``(M, 3)``, unique.
    link_cells
        Horizontal linking distance in cells, at least 1.
    link_up_cells
        Vertical linking distance in cells; ``link_cells`` when omitted.

    Returns
    -------
    tuple
        The number of components, and each cell's component label, shape ``(M,)``.
    """
    n = len(coords)
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64)
    reach = np.array(
        [link_cells, link_cells, link_cells if link_up_cells is None else link_up_cells],
        dtype=np.int64,
    )
    # Keys only need to be unique and order-preserving within this batch, so
    # the box spanned by the coordinates (plus the link margin) is the grid.
    base = coords.min(axis=0) - reach
    extent = coords.max(axis=0) - base + reach + 1
    local = coords - base

    def encode(c: IntArray) -> IntArray:
        key: IntArray = (c[:, 0] * extent[1] + c[:, 1]) * extent[2] + c[:, 2]
        return key

    keys = encode(local)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    spans = [range(-int(r), int(r) + 1) for r in reach]
    # Half of the neighbourhood suffices: the graph is undirected.
    offsets = [o for o in itertools.product(*spans) if o > (0, 0, 0)]
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    for off in offsets:
        # The link margin in `base`/`extent` keeps every neighbour inside the
        # grid, so no key wraps onto a different row of cells.
        nb_keys = encode(local + np.asarray(off, dtype=np.int64))
        slot = np.minimum(np.searchsorted(sorted_keys, nb_keys), n - 1)
        found = sorted_keys[slot] == nb_keys
        rows.append(np.flatnonzero(found))
        cols.append(order[slot[found]])
    src = np.concatenate(rows)
    dst = np.concatenate(cols)
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    return int(n_comp), np.asarray(labels, dtype=np.int64)
