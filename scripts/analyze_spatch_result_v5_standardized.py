#!/usr/bin/env python3
"""Run the established result_v3 standardized analysis for one SPATCH run."""

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
        "model_variant": "v3_bidirectional_sparse_uot_fixed_lc0.1_with_delayed_ot_refresh",
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
    print("SPATCH_RESULT_V5_STANDARDIZED_ANALYSIS: PASS", flush=True)


if __name__ == "__main__":
    main()
