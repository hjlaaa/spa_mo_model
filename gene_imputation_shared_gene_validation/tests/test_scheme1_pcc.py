from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "compute_scheme1_pcc.py"
SPEC = importlib.util.spec_from_file_location("scheme1_pcc", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Scheme1PccTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pred = np.asarray(
            [[1, 3, 2, 8], [2, 5, 4, 1], [7, 1, 3, 6]], dtype=np.float32
        )
        self.true = np.asarray(
            [[2, 2, 4, 7], [1, 6, 3, 2], [5, 2, 4, 8]], dtype=np.float32
        )

    def test_vectorized_values_match_numpy(self) -> None:
        row_expected = np.asarray(
            [np.corrcoef(x, y)[0, 1] for x, y in zip(self.pred, self.true)]
        )
        col_expected = np.asarray(
            [
                np.corrcoef(self.pred[:, index], self.true[:, index])[0, 1]
                for index in range(self.pred.shape[1])
            ]
        )
        np.testing.assert_allclose(
            MODULE.pearson_values(self.pred, self.true, axis=1), row_expected
        )
        np.testing.assert_allclose(
            MODULE.pearson_values(self.pred, self.true, axis=0), col_expected
        )

    def test_streaming_row_moments_match_direct_row_pcc(self) -> None:
        moments = MODULE.empty_row_moments(len(self.pred))
        for start, end in ((0, 2), (2, 4)):
            pred_block = self.pred[:, start:end]
            true_block = self.true[:, start:end]
            MODULE.update_true_moments(moments, true_block)
            MODULE.update_prediction_moments(
                moments, "unsmoothed", pred_block, true_block
            )
        streamed = MODULE.finalize_row_pcc(
            moments, "unsmoothed", self.pred.shape[1]
        )
        direct = MODULE.pearson_values(self.pred, self.true, axis=1)
        np.testing.assert_allclose(streamed, direct, rtol=1e-14, atol=1e-14)

    def test_mean_excludes_constant_slice_nan(self) -> None:
        pred = np.vstack([self.pred, np.ones((1, self.pred.shape[1]))])
        true = np.vstack([self.true, np.ones((1, self.true.shape[1]))])
        values = MODULE.pearson_values(pred, true, axis=1)
        mean, valid = MODULE.mean_finite(values)
        self.assertEqual(valid, 3)
        self.assertAlmostEqual(mean, float(np.mean(values[:3])))


if __name__ == "__main__":
    unittest.main()
