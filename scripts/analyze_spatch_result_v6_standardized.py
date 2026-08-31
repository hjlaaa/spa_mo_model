#!/usr/bin/env python3
"""Run the result_v5-compatible standardized analysis for SPATCH result_v6."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_result_v3_large_standardized as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    base.RESULT_ROOT = args.result_root.resolve()
    dataset_manifest = base.analyze_spatch()
    run_summary = json.loads(
        (
            base.RESULT_ROOT
            / "spatch"
            / base.VARIANT
            / "run_summary.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "analysis": "standardized_embedding_only",
        "training_seed": 42,
        "model_variant": "dual_graphsage_with_pre_post_ot_independent_parameters",
        "architecture": run_summary.get("architecture"),
        "pre_post_graphsage_parameter_sharing": run_summary.get(
            "pre_post_graphsage_parameter_sharing"
        ),
        "ot_refresh_embedding_key": run_summary.get("ot_refresh_embedding_key"),
        "graphsage_self_path_mode": "no_self_linear",
        "dynamic_candidate_source": run_summary.get("dynamic_candidate_source", "final"),
        "attention_context_gate_enabled": bool(
            run_summary.get("attention_context_gate_enabled", False)
        ),
        "datasets": [dataset_manifest],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    output = base.RESULT_ROOT / "spatch_standardized_analysis_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("SPATCH_RESULT_V6_STANDARDIZED_ANALYSIS: PASS", flush=True)


if __name__ == "__main__":
    main()
