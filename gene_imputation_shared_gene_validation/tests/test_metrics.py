#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_shared_gene_validation.py"
SPEC = importlib.util.spec_from_file_location("shared_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MetricTest(unittest.TestCase):
    def test_pearson_columns(self) -> None:
        left = np.asarray([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=np.float32)
        right = np.asarray([[2, 1], [4, 2], [6, 3], [8, 4]], dtype=np.float32)
        np.testing.assert_allclose(MODULE.pearson_columns(left, right), [1, -1])

    def test_spearman_is_rank_based(self) -> None:
        left = np.asarray([[1], [2], [3], [4]], dtype=np.float32)
        right = np.asarray([[10], [20], [40], [80]], dtype=np.float32)
        np.testing.assert_allclose(MODULE.spearman_columns(left, right), [1])

    def test_constant_column_is_nan(self) -> None:
        left = np.ones((5, 1), dtype=np.float32)
        right = np.arange(5, dtype=np.float32)[:, None]
        self.assertTrue(np.isnan(MODULE.pearson_columns(left, right)[0]))


if __name__ == "__main__":
    unittest.main()
