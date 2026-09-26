"""Tests for :mod:`canopy.perception.cells`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from canopy.perception.cells import CellMap, label_cells


def test_add_pools_hits_counts_mean_position_and_mean_colour() -> None:
    """Two hits landing in one cell pool into one row; a third stays in its own cell."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([2.0, 2.0, 2.0]), cell_m=1.0)
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.3, 0.2, 0.4],
            [1.5, 1.5, 1.5],
        ]
    )
    rgb = np.array([[10, 20, 30], [30, 40, 50], [100, 110, 120]], dtype=np.uint8)

    cell_map.add(points, rgb, t=1.0, drone_id=0)

    assert len(cell_map) == 2
    assert cell_map.coords().tolist() == [[0, 0, 0], [1, 1, 1]]
    assert cell_map.hits.tolist() == [2, 1]
    np.testing.assert_allclose(
        cell_map.mean_pos(np.array([0], dtype=np.int64))[0], points[:2].mean(axis=0)
    )
    np.testing.assert_allclose(
        cell_map.mean_rgb(np.array([0], dtype=np.int64))[0],
        rgb[:2].astype(np.float64).mean(axis=0),
    )
    np.testing.assert_allclose(cell_map.mean_pos()[1], points[2])
    np.testing.assert_allclose(cell_map.mean_rgb()[1], rgb[2].astype(np.float64))


def test_add_with_every_point_outside_the_grid_leaves_the_map_empty() -> None:
    """add()'s early return for an all-outside batch changes nothing."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([1.0, 1.0, 1.0]), cell_m=1.0)

    cell_map.add(
        np.array([[-1.0, -1.0, -1.0], [5.0, 5.0, 5.0]]),
        np.zeros((2, 3), dtype=np.uint8),
        t=0.0,
        drone_id=0,
    )

    assert len(cell_map) == 0


def test_points_outside_lo_hi_are_ignored() -> None:
    """A mixed batch keeps only the hits inside [lo, hi); hi itself is exclusive."""
    cell_map = CellMap(lo=np.array([0.0, 0.0, 0.0]), hi=np.array([2.0, 2.0, 2.0]), cell_m=1.0)
    points = np.array(
        [
            [-0.1, 0.5, 0.5],  # below lo on x
            [0.5, 0.5, 2.0],  # sits at hi on z, which is exclusive
            [0.5, 0.5, 0.5],  # inside
            [5.0, 5.0, 5.0],  # far outside
        ]
    )

    cell_map.add(points, np.zeros((4, 3), dtype=np.uint8), t=0.0, drone_id=0)

    assert len(cell_map) == 1
    assert cell_map.hits.tolist() == [1]
    np.testing.assert_allclose(cell_map.mean_pos()[0], points[2])


def test_keys_stay_ascending_when_a_batch_mixes_new_and_existing_cells() -> None:
    """A new cell inserted between two known ones, and one appended past the last, stay sorted."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([4.0, 1.0, 1.0]), cell_m=1.0)
    cell_map.add(
        np.array([[0.5, 0.5, 0.5], [3.5, 0.5, 0.5]]),
        np.zeros((2, 3), dtype=np.uint8),
        t=1.0,
        drone_id=1,
    )
    assert cell_map.keys.tolist() == [0, 3]

    # Cell 3 and cell 0 already exist; cell 1 is new and sorts between them.
    cell_map.add(
        np.array([[3.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        np.zeros((3, 3), dtype=np.uint8),
        t=2.0,
        drone_id=2,
    )

    assert cell_map.keys.tolist() == [0, 1, 3]
    assert np.all(np.diff(cell_map.keys) > 0)
    assert cell_map.hits.tolist() == [2, 1, 2]


def test_first_t_and_first_by_set_on_creation_and_not_overwritten() -> None:
    """A second hit on an existing cell adds to its count but leaves first_t/first_by alone."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([1.0, 1.0, 1.0]), cell_m=1.0)
    point = np.array([[0.5, 0.5, 0.5]])
    rgb = np.array([[1, 2, 3]], dtype=np.uint8)

    cell_map.add(point, rgb, t=1.0, drone_id=7)
    cell_map.add(point, rgb, t=2.0, drone_id=9)

    assert cell_map.hits.tolist() == [2]
    assert cell_map.first_t.tolist() == [1.0]
    assert cell_map.first_by.tolist() == [7]


def test_coords_returns_integer_cell_coordinates() -> None:
    """coords() decodes each row's flat key back to its (i, j, k) grid index."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([3.0, 2.0, 4.0]), cell_m=1.0)
    cell_map.add(
        np.array([[2.5, 1.5, 3.5], [0.5, 0.5, 0.5]]),
        np.zeros((2, 3), dtype=np.uint8),
        t=0.0,
        drone_id=0,
    )

    assert cell_map.coords().tolist() == [[0, 0, 0], [2, 1, 3]]
    assert cell_map.coords(np.array([0], dtype=np.int64)).tolist() == [[0, 0, 0]]
    assert cell_map.coords(np.array([1], dtype=np.int64)).tolist() == [[2, 1, 3]]


def test_neighbourhood_sums_of_an_isolated_cell_is_its_own_sum() -> None:
    """With no populated neighbours, a cell's neighbourhood sum is exactly its own evidence."""
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([5.0, 5.0, 5.0]), cell_m=1.0)
    cell_map.add(
        np.array([[2.5, 2.5, 2.5]]), np.array([[10, 20, 30]], dtype=np.uint8), t=0.0, drone_id=0
    )

    rgb_sum, hits = cell_map.neighbourhood_sums(np.array([0], dtype=np.int64))

    np.testing.assert_allclose(rgb_sum[0], cell_map.rgb_sum[0])
    assert hits.tolist() == cell_map.hits.tolist() == [1]


def _brute_force_neighbourhood_sums(
    cell_map: CellMap, rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Recompute neighbourhood_sums by an explicit 3x3x3 search, to check the vectorised version."""
    coords = cell_map.coords()
    targets = cell_map.coords(rows)
    rgb = np.zeros((rows.size, 3), dtype=np.float64)
    hits = np.zeros(rows.size, dtype=np.int64)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i, cell in enumerate(targets):
        for offset in offsets:
            match = np.flatnonzero(np.all(coords == cell + np.array(offset), axis=1))
            if match.size:
                rgb[i] += cell_map.rgb_sum[match[0]]
                hits[i] += cell_map.hits[match[0]]
    return rgb, hits


def test_neighbourhood_sums_matches_brute_force_3x3x3_search() -> None:
    """On a small random map, the vectorised 26-neighbour sum matches an explicit search."""
    rng = np.random.default_rng(42)
    cell_map = CellMap(lo=np.zeros(3), hi=np.array([4.0, 4.0, 4.0]), cell_m=1.0)
    points = rng.uniform(0.0, 4.0, size=(200, 3))
    rgb = rng.integers(0, 256, size=(200, 3), dtype=np.uint8)
    cell_map.add(points, rgb, t=0.0, drone_id=0)
    rows = np.arange(len(cell_map), dtype=np.int64)

    got_rgb, got_hits = cell_map.neighbourhood_sums(rows)
    want_rgb, want_hits = _brute_force_neighbourhood_sums(cell_map, rows)

    np.testing.assert_allclose(got_rgb, want_rgb)
    assert got_hits.tolist() == want_hits.tolist()


@pytest.mark.parametrize(("link_cells", "expect_joined"), [(1, False), (2, False), (3, True)])
def test_label_cells_two_blobs_three_cells_apart(link_cells: int, expect_joined: bool) -> None:
    """Two single cells 3 apart split into components until the link distance reaches the gap."""
    coords = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.int64)

    n_comp, labels = label_cells(coords, link_cells=link_cells)

    assert n_comp == (1 if expect_joined else 2)
    assert (labels[0] == labels[1]) == expect_joined


def test_label_cells_diagonal_neighbours_join_at_link_1() -> None:
    """A cell and its space-diagonal neighbour join under ordinary 26-connectivity."""
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)

    n_comp, labels = label_cells(coords, link_cells=1)

    assert n_comp == 1
    assert labels[0] == labels[1]


@pytest.mark.parametrize(("link_up_cells", "expect_joined"), [(1, False), (3, True)])
def test_label_cells_vertical_linking_uses_link_up_cells(
    link_up_cells: int, expect_joined: bool
) -> None:
    """Two cells 3 apart vertically join only once link_up_cells reaches that gap."""
    coords = np.array([[0, 0, 0], [0, 0, 3]], dtype=np.int64)

    n_comp, labels = label_cells(coords, link_cells=1, link_up_cells=link_up_cells)

    assert n_comp == (1 if expect_joined else 2)
    assert (labels[0] == labels[1]) == expect_joined


def test_label_cells_horizontal_gap_not_bridged_by_link_up_cells() -> None:
    """link_up_cells widens only the vertical reach; a horizontal gap still needs link_cells."""
    coords = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.int64)

    n_comp, labels = label_cells(coords, link_cells=1, link_up_cells=4)

    assert n_comp == 2
    assert labels[0] != labels[1]


def test_label_cells_empty_input_returns_zero_components_and_empty_array() -> None:
    """No cells means no components, and a correctly typed empty label array."""
    n_comp, labels = label_cells(np.zeros((0, 3), dtype=np.int64), link_cells=1)

    assert n_comp == 0
    assert labels.shape == (0,)
    assert labels.dtype == np.int64
