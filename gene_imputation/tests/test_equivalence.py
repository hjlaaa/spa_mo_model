#!/usr/bin/env python3
"""Tests that blocked sparse transfer exactly matches COSIE's KNN mean."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_spatch_knn_imputation.py"
SPEC = importlib.util.spec_from_file_location("knn_imputation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EquivalenceTest(unittest.TestCase):
    def test_sparse_transfer_matches_cosie_loop(self) -> None:
        rng = np.random.default_rng(42)
        source = rng.poisson(0.3, size=(31, 17)).astype(np.float32)
        source[rng.random(source.shape) < 0.7] = 0
        neighbors = np.vstack([rng.choice(len(source), size=5, replace=False) for _ in range(13)]).astype(np.int32)
        expected = np.vstack([source[row].mean(axis=0) for row in neighbors])
        transfer = MODULE.make_uniform_transfer_matrix(neighbors, len(source))
        actual_blocks = []
        sparse_source = sp.csr_matrix(source)
        for start in range(0, source.shape[1], 4):
            actual_blocks.append(MODULE.aggregate_block(transfer, sparse_source[:, start : start + 4]))
        actual = np.hstack(actual_blocks)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_transfer_rows_sum_to_one(self) -> None:
        neighbors = np.asarray([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
        transfer = MODULE.make_uniform_transfer_matrix(neighbors, 6)
        np.testing.assert_allclose(np.asarray(transfer.sum(axis=1)).reshape(-1), np.ones(2))

    def test_invalid_neighbor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.make_uniform_transfer_matrix(np.asarray([[0, 7]], dtype=np.int32), 7)


if __name__ == "__main__":
    unittest.main()

