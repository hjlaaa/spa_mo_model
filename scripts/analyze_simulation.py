#!/usr/bin/env python3
"""Analyze spa_mo_model Simulation embeddings with full common metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


ORIGINAL = Path(__file__).with_name("analyze_misar_seq_clustering.py")
RUN_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/simulation/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)
DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Simulation")
SECTIONS = [f"Simulation{i}" for i in range(1, 6)]


def _load_embeddings() -> dict[str, np.ndarray]:
    summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    result = {}
    for section in SECTIONS:
        path = Path(summary["saved_files"]["final_embeddings"][section])
        result[section] = np.load(path)
    return result


def _simulation_diagnostics(output_dir: Path) -> None:
    embeddings = _load_embeddings()
    x = StandardScaler().fit_transform(np.vstack([embeddings[s] for s in SECTIONS]))
    spfac, rna_nsfac, adt_nsfac = [], [], []
    for section in SECTIONS:
        rna = ad.read_h5ad(DATA_DIR / section / "adata_RNA.h5ad", backed="r")
        adt = ad.read_h5ad(DATA_DIR / section / "adata_ADT.h5ad", backed="r")
        try:
            spfac.append(np.asarray(rna.obsm["spfac"]))
            rna_nsfac.append(np.asarray(rna.obsm["nsfac"]))
            adt_nsfac.append(np.asarray(adt.obsm["nsfac"]))
        finally:
            rna.file.close()
            adt.file.close()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, truth in (
        ("spfac_recovery", np.vstack(spfac)),
        ("rna_nsfac_leakage", np.vstack(rna_nsfac)),
        ("adt_nsfac_leakage", np.vstack(adt_nsfac)),
    ):
        pred = cross_val_predict(LinearRegression(), x, truth, cv=cv, n_jobs=-1)
        rows.append(
            {
                "diagnostic": name,
                "cv_linear_r2_variance_weighted": float(
                    r2_score(truth, pred, multioutput="variance_weighted")
                ),
                "interpretation": "higher_is_better" if name == "spfac_recovery" else "lower_is_better",
            }
        )
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "simulation_factor_diagnostics.csv", index=False)

    retrieval = []
    for left, right in zip(SECTIONS[:-1], SECTIONS[1:]):
        a = embeddings[left] / np.maximum(np.linalg.norm(embeddings[left], axis=1, keepdims=True), 1e-12)
        b = embeddings[right] / np.maximum(np.linalg.norm(embeddings[right], axis=1, keepdims=True), 1e-12)
        similarity = a @ b.T
        ranks = 1 + np.sum(similarity > np.diag(similarity)[:, None], axis=1)
        retrieval.append(
            {
                "source": left,
                "target": right,
                "top1_accuracy": float(np.mean(ranks == 1)),
                "recall_at_5": float(np.mean(ranks <= 5)),
                "mean_foscttm": float(np.mean((ranks - 1) / (len(ranks) - 1))),
            }
        )
    pd.DataFrame(retrieval).to_csv(metrics_dir / "cross_section_spot_retrieval.csv", index=False)
    with (output_dir / "SUMMARY.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Simulation-specific diagnostics\n\n"
            "- Spatial-factor recovery and RNA/ADT nuisance-factor leakage: "
            "`metrics/simulation_factor_diagnostics.csv`.\n"
            "- Same-grid cross-section retrieval/FOSCTTM: "
            "`metrics/cross_section_spot_retrieval.csv`.\n"
            "- Ground-truth factors were used only during this analysis, never in training.\n"
        )


def main() -> None:
    source = (
        ORIGINAL.read_text(encoding="utf-8")
        .replace("MISAR-seq", "Simulation")
        .replace(
            '"kmeans_input": "raw_final_embedding"',
            '"kmeans_input": "standardized_final_embedding"',
        )
    )
    namespace = {"__file__": str(ORIGINAL), "__name__": "simulation_spa_mo_analysis"}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    namespace["DEFAULT_SECTION_ORDER"] = list(SECTIONS)
    namespace["DEFAULT_LABEL_KEYS"] = ["spatial_domain"]

    def standardized_fit_predict(embedding: np.ndarray, n_clusters: int, args) -> np.ndarray:
        standardized = StandardScaler().fit_transform(embedding)
        return (
            namespace["choose_kmeans"](args, n_clusters)
            .fit_predict(standardized)
            .astype(int)
        )

    namespace["fit_predict"] = standardized_fit_predict
    sys.argv[1:1] = [
        "--input_dir", str(RUN_DIR),
        "--output_dir", str(RUN_DIR / "clustering_analysis"),
        "--section_order", ",".join(SECTIONS),
        "--n_clusters", "5,8,10,12",
        "--label_keys", "spatial_domain",
        "--seed", "42",
        "--point_size", "24",
        "--kmeans_method", "kmeans",
        "--n_init", "20",
        "--metric_sample_size", "10000",
        "--batch_metrics_max_samples", "0",
        "--batch_metric_seed", "42",
    ]
    namespace["main"]()
    summary_path = RUN_DIR / "clustering_analysis" / "clustering_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["kmeans_input"] = "standardized_final_embedding"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _simulation_diagnostics(RUN_DIR / "clustering_analysis")


if __name__ == "__main__":
    main()
