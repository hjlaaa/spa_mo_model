#!/usr/bin/env python3
"""Run the established result_v3 standardized analysis for one MISAR-seq run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_result_v3_standardized as base
from scripts import compare_misar_seq_kmeans_preprocessing as misar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    base.RESULT_ROOT = args.result_root.resolve()
    data = base._misar_data()
    misar.validate(data)
    misar.run_scheme(data, base.SCHEME)
    removed = base._prune(data.output_root, misar.METRIC_KS, misar.PLOT_KS)
    dataset_manifest = base._annotate(data.output_root, "MISAR-seq", removed)
    manifest = {
        "analysis": "standardized_embedding_only",
        "training_seed": 42,
        "model_variant": "dual_graphsage_with_pre_post_ot_independent_parameters",
        "graphsage_self_path_mode": "no_self_linear",
        "dynamic_candidate_source": dataset_manifest["dynamic_candidate_source"],
        "attention_context_gate_enabled": dataset_manifest[
            "attention_context_gate_enabled"
        ],
        "datasets": [dataset_manifest],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    output = base.RESULT_ROOT / "misar_seq_standardized_analysis_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("MISAR_RESULT_V5_STANDARDIZED_ANALYSIS: PASS", flush=True)


if __name__ == "__main__":
    main()
