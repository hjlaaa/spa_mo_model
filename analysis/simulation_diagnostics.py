"""Simulation factor recovery, nuisance leakage and same-grid retrieval.

Consumes prepared run embeddings and existing ground-truth factors. No training,
preprocessing or clustering is performed here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from analysis.clustering import fitted_space
from analysis.inputs import load_simulation_embeddings, load_simulation_factors


def simulation_diagnostics(
    run_dir: Path, data_dir: Path, output_dir: Path, section_order: list[str]
) -> None:
    embeddings = load_simulation_embeddings(run_dir, section_order)
    x = fitted_space(np.vstack([embeddings[s] for s in section_order]), "standardized_embedding")[0]
    spfac, rna_nsfac, adt_nsfac = load_simulation_factors(data_dir, section_order)
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
    for left, right in zip(section_order[:-1], section_order[1:]):
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
