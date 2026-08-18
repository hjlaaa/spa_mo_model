#!/usr/bin/env python3
"""Analyze the MouseBrain result_v4 run with the established v3 protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_mousebrain_kmeans_preprocessing import (  # noqa: E402
    MAX_ITER,
    METRIC_KS,
    N_INIT,
    PLOT_KS,
    POINT_SIZE,
    SEED,
    SECTION_DIRS,
    SECTIONS,
    SPATIAL_NEIGHBOR_K,
    MethodData,
    aligned_metadata,
    run_scheme,
    validate,
    write_scheme_summary,
)


RESULT_ROOT = (
    PROJECT_ROOT
    / "result_v4"
    / "mousebrain"
    / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)
RUN_ROOT = RESULT_ROOT / "epochs_200"
DATA_DIR = PROJECT_ROOT / "data" / "dataset_MouseBrain"
SCHEME = "standardized_embedding"


def load_data() -> MethodData:
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in SECTIONS:
        embedding_path = (
            RUN_ROOT / "final_embeddings" / f"{section}_final_embedding.npy"
        )
        rna_path = DATA_DIR / SECTION_DIRS[section] / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(
                f"{section}: embedding rows {len(embedding)} != {len(names)}"
            )
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.append(embedding_path)

    section_array = np.asarray(sections)
    barcode_array = np.asarray(barcodes)
    coordinates, truth = aligned_metadata(DATA_DIR, section_array, barcode_array)
    return MethodData(
        name=f"spa_mo_model_{RESULT_ROOT.parents[1].name}",
        embedding=np.vstack(arrays),
        sections=section_array,
        barcodes=barcode_array,
        coords=coordinates,
        truth=truth,
        data_dir=DATA_DIR,
        source_paths=sources,
        output_root=RESULT_ROOT / "analysis",
        shared_sources={},
    )


def prune_unretained_joint_directories(data: MethodData) -> list[str]:
    cluster_root = data.output_root / SCHEME / "clustering"
    removed: list[str] = []
    for k in METRIC_KS:
        if k in PLOT_KS:
            continue
        target = cluster_root / f"joint_k{k}"
        if target.is_dir():
            if target.resolve().parent != cluster_root.resolve():
                raise ValueError(f"unsafe prune target: {target}")
            shutil.rmtree(target)
            removed.append(str(target))
    write_scheme_summary(data, SCHEME)
    return removed


def main() -> None:
    global RESULT_ROOT, RUN_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "result_v4",
        help="Top-level result root containing the MouseBrain run.",
    )
    args = parser.parse_args()
    RESULT_ROOT = (
        args.result_root.resolve()
        / "mousebrain"
        / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    )
    RUN_ROOT = RESULT_ROOT / "epochs_200"
    data = load_data()
    validate(data)
    run_scheme(data, SCHEME)
    removed = prune_unretained_joint_directories(data)
    run_summary = json.loads((RUN_ROOT / "run_summary.json").read_text())
    dynamic_source = run_summary.get("dynamic_candidate_source", "final")
    context_gate_enabled = bool(
        run_summary.get("attention_context_gate_enabled", False)
    )
    if dynamic_source == "final" and not context_gate_enabled:
        model_variant = "v3_bidirectional_sparse_uot_fixed_lc0.1"
    elif dynamic_source == "final" and context_gate_enabled:
        model_variant = "microenvironment_attention_gate_with_v3_dynamic_ot"
    elif dynamic_source == "fused" and not context_gate_enabled:
        model_variant = "fused_dynamic_ot_without_microenvironment_attention_gate"
    else:
        model_variant = "microenvironment_aware_bidirectional_sparse_uot_fixed_lc0.1"
    config_path = data.output_root / SCHEME / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "training_seed": 42,
            "model_variant": model_variant,
            "dynamic_candidate_source": dynamic_source,
            "dynamic_semantic_source": dynamic_source,
            "dynamic_context_source": f"{dynamic_source}_spatial_context",
            "attention_context_source": (
                "fused_spatial_context" if context_gate_enabled else None
            ),
            "attention_context_gate_enabled": context_gate_enabled,
        }
    )
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "MouseBrain",
        "training_run": str(RUN_ROOT),
        "analysis": str(data.output_root / SCHEME),
        "analysis_scope": "standardized_embedding_only",
        "training_seed": 42,
        "model_variant": model_variant,
        "dynamic_candidate_source": dynamic_source,
        "dynamic_semantic_source": dynamic_source,
        "dynamic_context_source": f"{dynamic_source}_spatial_context",
        "attention_context_source": (
            "fused_spatial_context" if context_gate_enabled else None
        ),
        "attention_context_gate_enabled": context_gate_enabled,
        "n_obs": int(len(data.embedding)),
        "kmeans": {
            "seed": SEED,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "metric_k": METRIC_KS,
            "retained_k": PLOT_KS,
        },
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_plot_point_size": POINT_SIZE,
        "removed_generated_intermediate_directories": removed,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (RESULT_ROOT / "standardized_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("MOUSEBRAIN_V4_STANDARDIZED_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
