from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "plot_shared_gene_validation.py"
SPEC = importlib.util.spec_from_file_location("pcc_plots", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PccPlotDataTest(unittest.TestCase):
    def test_finite_mean_excludes_nan(self) -> None:
        observed = MODULE.finite_mean(np.asarray([0.2, np.nan, 0.4]))
        self.assertAlmostEqual(observed, 0.3)

    def test_paired_valid_uses_common_finite_entries(self) -> None:
        left, right = MODULE.paired_valid(
            np.asarray([0.1, np.nan, 0.3, 0.4]),
            np.asarray([0.2, 0.2, np.nan, 0.5]),
        )
        np.testing.assert_allclose(left, [0.1, 0.4])
        np.testing.assert_allclose(right, [0.2, 0.5])

    def test_detection_strata_cover_positive_genes(self) -> None:
        frame = pd.DataFrame(
            {
                "target_nonzero_fraction": np.linspace(0.01, 0.8, 12),
                "unsmoothed_pcc": np.linspace(0.1, 0.4, 12),
                "smoothed_pcc": np.linspace(0.11, 0.41, 12),
            }
        )
        observed = MODULE.stratum_summary(frame)
        self.assertEqual(observed["genes"].sum(), 12)
        self.assertEqual(
            observed["stratum"].tolist(),
            ["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        )
        self.assertTrue(np.allclose(observed["mean_delta"], 0.01))


if __name__ == "__main__":
    unittest.main()
