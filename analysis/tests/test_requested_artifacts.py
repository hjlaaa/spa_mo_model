"""Regression checks for the two requested analysis output contracts."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.artifacts import requested_analysis_artifacts
from analysis.protocols import REQUESTED_PLOT_KS, SPECS


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_bytes(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"value\n")


class RequestedArtifactContractTest(unittest.TestCase):
    def test_hln_requested_protocol_and_generic_layout(self):
        self.assertEqual(SPECS["human_lymph_node"].joint_ks, (8, 10, 12))
        self.assertEqual(REQUESTED_PLOT_KS["human_lymph_node"], [8, 10, 12])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            base = output / "standardized_embedding"
            write_json(base / "config.json", {"dataset": "Human Lymph Node",
                "model_version": "test", "post_ot_graphsage_scale": 0.5})
            write_json(base / "analysis_source_manifest.json", {"complete": True})
            write_json(base / "analysis_completion.json", {"dataset": "Human Lymph Node",
                "analysis": str(base), "joint_k_values": [8, 10, 12],
                "independent_k_values": [8, 10, 12],
                "internal_metrics": str(base / "metrics/internal_metrics.csv"),
                "batch_metrics": str(base / "metrics/batch_metrics.csv"),
                "supervised_metrics": None})
            for name in ("internal", "batch"):
                write_bytes(base / f"metrics/{name}_metrics.csv")
            write_json(output / "figures/analysis_source_manifest.json", {"complete": True})
            write_bytes(output / "figures/spatial_section.png")
            requested_analysis_artifacts("human_lymph_node", output,
                model_version="test", post_ot_graphsage_scale=0.5,
                k_values=[8, 10, 12], require_plots=True)
            (base / "metrics/batch_metrics.csv").unlink()
            with self.assertRaisesRegex(ValueError, "Missing or empty"):
                requested_analysis_artifacts("human_lymph_node", output)

    def test_spatch_direct_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            write_json(base / "config.json", {"dataset": "SPATCH",
                "model_version": "test", "post_ot_graphsage_scale": 0.5})
            write_json(base / "analysis_source_manifest.json", {"complete": True})
            write_json(base / "analysis_completion_manifest.json", {"status": "complete",
                "dataset": "SPATCH", "analysis": str(base),
                "model_version": "test", "post_ot_graphsage_scale": 0.5,
                "k_values": [5, 8], "spatial_plot_count": 1,
                **{f"{name}_metrics": str(base / f"metrics/{name}_metrics.csv")
                   for name in ("internal", "batch", "supervised")}})
            for name in ("internal", "batch", "supervised"):
                write_bytes(base / f"metrics/{name}_metrics.csv")
            write_bytes(base / "clustering/spatial_section.png")
            requested_analysis_artifacts("spatch", base, model_version="test",
                post_ot_graphsage_scale=0.5, k_values=[5, 8], require_plots=True)
            with self.assertRaisesRegex(ValueError, "K values"):
                requested_analysis_artifacts("spatch", base, k_values=[5, 10])


if __name__ == "__main__":
    unittest.main()
