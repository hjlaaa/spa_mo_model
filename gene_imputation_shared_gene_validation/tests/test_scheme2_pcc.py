from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "compute_scheme2_pcc.py"
SPEC = importlib.util.spec_from_file_location("scheme2_pcc", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Scheme2PccTest(unittest.TestCase):
    def test_normalize_total_log1p(self) -> None:
        matrix = np.asarray([[1, 3], [0, 0], [2, 2]], dtype=np.float32)
        totals = np.asarray([4, 0, 4], dtype=np.float64)
        observed = MODULE.normalize_total_log1p(matrix, totals, 100.0)
        expected = np.log1p(
            np.asarray([[25, 75], [0, 0], [50, 50]], dtype=np.float64)
        )
        np.testing.assert_allclose(observed, expected)

    def test_block_transform_uses_full_gene_row_total(self) -> None:
        full = np.asarray([[1, 3, 6], [2, 0, 8]], dtype=np.float32)
        totals = full.sum(axis=1)
        first = MODULE.normalize_total_log1p(full[:, :2], totals, 10.0)
        expected = np.log1p(np.asarray([[1, 3], [2, 0]], dtype=np.float64))
        np.testing.assert_allclose(first, expected)

    def test_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.normalize_total_log1p(
                np.asarray([[1, -1]], dtype=np.float32),
                np.asarray([0], dtype=np.float64),
                10000.0,
            )

    def test_blockwise_normalized_row_pcc_matches_full_matrix(self) -> None:
        pred = np.asarray(
            [[1, 3, 6, 2], [2, 0, 8, 5], [5, 2, 1, 4]], dtype=np.float32
        )
        true = np.asarray(
            [[2, 4, 3, 3], [1, 1, 7, 6], [4, 3, 2, 6]], dtype=np.float32
        )
        pred_total = pred.sum(axis=1)
        true_total = true.sum(axis=1)
        pred_full = MODULE.normalize_total_log1p(pred, pred_total, 10000.0)
        true_full = MODULE.normalize_total_log1p(true, true_total, 10000.0)
        direct = MODULE.common.pearson_values(pred_full, true_full, axis=1)

        moments = MODULE.common.empty_row_moments(len(pred))
        for start, end in ((0, 2), (2, 4)):
            pred_block = MODULE.normalize_total_log1p(
                pred[:, start:end], pred_total, 10000.0
            )
            true_block = MODULE.normalize_total_log1p(
                true[:, start:end], true_total, 10000.0
            )
            MODULE.common.update_true_moments(moments, true_block)
            MODULE.common.update_prediction_moments(
                moments, "unsmoothed", pred_block, true_block
            )
        streamed = MODULE.common.finalize_row_pcc(
            moments, "unsmoothed", pred.shape[1]
        )
        np.testing.assert_allclose(streamed, direct, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
