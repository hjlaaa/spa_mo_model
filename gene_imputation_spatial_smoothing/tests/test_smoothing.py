#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_spatch_smoothed_knn_imputation.py"
SPEC = importlib.util.spec_from_file_location("smoothed_imputation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SpatialSmoothingTest(unittest.TestCase):
    def test_eight_neighbor_grid(self) -> None:
        coordinates = np.asarray([(x, y) for x in range(3) for y in range(3)])
        matrix, counts, offsets = MODULE.build_grid_spatial_smoothing_matrix(coordinates, 1.5)
        self.assertEqual(len(offsets), 8)
        self.assertEqual(counts[4], 8)
        self.assertEqual(counts[0], 3)
        np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).reshape(-1), 1)
        self.assertEqual(matrix[4, 4], 0)
        np.testing.assert_allclose(matrix[4].data, np.full(8, 1 / 8))

    def test_isolated_spot_keeps_itself(self) -> None:
        coordinates = np.asarray([[0, 0], [10, 10]])
        matrix, counts, _ = MODULE.build_grid_spatial_smoothing_matrix(coordinates, 1.5)
        np.testing.assert_array_equal(counts, [0, 0])
        np.testing.assert_allclose(matrix.toarray(), np.eye(2))

    def test_block_formula_matches_direct_loops(self) -> None:
        coordinates = np.asarray([(x, y) for x in range(3) for y in range(3)])
        spatial_mean, _, _ = MODULE.build_grid_spatial_smoothing_matrix(coordinates, 1.5)
        source = np.arange(9 * 4, dtype=np.float32).reshape(9, 4)
        alpha = 0.25
        smoothed = (1 - alpha) * source + alpha * spatial_mean.dot(source)
        neighbors = np.asarray([[0, 1, 3], [4, 5, 7]], dtype=np.int32)
        expected = np.vstack([smoothed[row].mean(axis=0) for row in neighbors])
        transfer = MODULE.base.make_uniform_transfer_matrix(neighbors, len(source))
        actual = MODULE.base.aggregate_block(transfer, sp.csr_matrix(smoothed))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
