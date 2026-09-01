#!/usr/bin/env python3
"""Analyze one full-spot SPATCH v7 run with only the requested metrics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_spatch_kmeans_preprocessing as v6  # noqa: E402
from scripts.batch_correction_metrics import compute_batch_correction_metrics  # noqa: E402


VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
SECTIONS = tuple(v6.SECTIONS)
SECTION_COUNTS = dict(v6.SECTION_COUNTS)
N_OBS = v6.N_OBS
TRUTH_LABELS = tuple(v6.LABELS)
K_VALUES = (5, 8, 10, 12, 14, 16, 20)
SUPERVISED_METRICS = ("ARI", "NMI", "Homogeneity", "Completeness")
INTERNAL_METRICS = ("ASW", "ASW_Scaled", "CH", "DBI")
BATCH_METRICS = ("bASW", "bLISI", "kBET", "PCR_score")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "spot_metadata.csv.gz"
    columns = ["block_id", "section", "original_rna_obs_name", "x", "y", *TRUTH_LABELS]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"block_id": str, "section": str, "original_rna_obs_name": str},
        low_memory=False,
    ).rename(columns={"block_id": "spot_id"})
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["section"] = frame["section"].astype(str)
    frame["x"] = pd.to_numeric(frame["x"], errors="raise").astype(np.float32)
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(np.float32)
    if len(frame) != N_OBS:
        raise ValueError(f"Expected {N_OBS} SPATCH rows, got {len(frame)}.")
    if frame["section"].value_counts().to_dict() != SECTION_COUNTS:
        raise ValueError("SPATCH section counts differ from v6.")
    if frame[["section", "spot_id"]].duplicated().any():
        raise ValueError("Duplicate SPATCH section/spot IDs.")
    if not np.isfinite(frame[["x", "y"]].to_numpy()).all():
        raise ValueError("Non-finite SPATCH spatial coordinates.")
    return frame


def load_embedding(run_dir: Path) -> np.ndarray:
    arrays = []
    for section in SECTIONS:
        path = run_dir / f"final_embeddings_{section}.npy"
        array = np.load(path, mmap_mode="r")
        if array.shape != (SECTION_COUNTS[section], 128):
            raise ValueError(f"Unexpected {section} embedding shape: {array.shape}.")
        arrays.append(np.asarray(array, dtype=np.float32))
    embedding = np.vstack(arrays)
    if not np.isfinite(embedding).all():
        raise ValueError("SPATCH embedding contains non-finite values.")
    return embedding


def fixed_sample_indices(n: int) -> np.ndarray:
    rng = np.random.RandomState(v6.SEED)
    return np.sort(rng.permutation(n)[: min(v6.ASW_SAMPLE_SIZE, n)])


def supervised_rows(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    mode: str,
    scope: str,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_name in TRUTH_LABELS:
        truth = metadata[label_name]
        valid = truth.notna().to_numpy()
        if int(valid.sum()) < 2:
            continue
        y_true = truth.loc[valid].astype(str).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_pred = labels[valid]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_name,
                "k": k,
                "n_obs": int(valid.sum()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "Homogeneity": float(homogeneity_score(y_true, y_pred)),
                "Completeness": float(completeness_score(y_true, y_pred)),
            }
        )
    return rows


def internal_row(
    space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
    mode: str,
    scope: str,
    k: int,
) -> dict[str, Any]:
    sampled_labels = labels[sample_indices]
    asw = float(silhouette_score(sample_distances, sampled_labels, metric="precomputed"))
    return {
        "mode": mode,
        "scope": scope,
        "k": k,
        "n_obs": len(labels),
        "ASW": asw,
        "ASW_Scaled": float((asw + 1.0) / 2.0),
        "CH": float(calinski_harabasz_score(space, labels)),
        "DBI": float(davies_bouldin_score(space, labels)),
    }


def persist_clustering(
    output: Path,
    local_metadata: pd.DataFrame,
    mode: str,
    scope: str,
    k: int,
    labels: np.ndarray,
    centers_path: Path,
) -> None:
    v6.retained_outputs(
        output,
        local_metadata,
        mode,
        scope,
        k,
        labels,
        centers_path,
    )


def analyze(result_root: Path, model_version: str, scale: float) -> dict[str, Any]:
    run_dir = result_root / "spatch" / VARIANT
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("post_ot_graphsage_scale") != scale:
        raise ValueError(
            f"Training scale differs: {summary.get('post_ot_graphsage_scale')} != {scale}."
        )
    output = run_dir / "analysis" / "standardized_embedding"
    completion = output / "analysis_completion_manifest.json"
    if completion.is_file():
        print(json.dumps(json.loads(completion.read_text()), indent=2), flush=True)
        print(f"SPATCH_{model_version}_REQUESTED_ANALYSIS_ALREADY_COMPLETE", flush=True)
        return json.loads(completion.read_text())
    metrics_dir = output / "metrics"
    work_cache = output / "_fit_cache"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    work_cache.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(run_dir)
    embedding = load_embedding(run_dir)
    internal: list[dict[str, Any]] = []
    supervised: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    scopes = [("joint", "combined", np.ones(N_OBS, dtype=bool))]
    scopes.extend(
        ("independent", section, metadata["section"].eq(section).to_numpy())
        for section in SECTIONS
    )

    for mode, scope, mask in scopes:
        print(f"[{model_version}] prepare {mode}:{scope}", flush=True)
        raw = embedding if mode == "joint" else embedding[mask]
        scaler = StandardScaler(copy=True)
        space = scaler.fit_transform(raw)
        scaler_arrays.update(
            {
                f"{scope}_mean": scaler.mean_,
                f"{scope}_scale": scaler.scale_,
                f"{scope}_var": scaler.var_,
            }
        )
        local_metadata = metadata if mode == "joint" else metadata.loc[mask].reset_index(drop=True)
        sample_indices = fixed_sample_indices(len(space))
        sample_distances = pairwise_distances(
            space[sample_indices], metric="euclidean", n_jobs=-1
        )
        np.fill_diagonal(sample_distances, 0.0)
        for k in K_VALUES:
            prefix = f"{mode}_{scope}_k{k}"
            label_path = work_cache / f"{prefix}.npy"
            info_path = work_cache / f"{prefix}.json"
            centers_path = work_cache / f"{prefix}_centers.npy"
            print(f"[{model_version}] {prefix}", flush=True)
            labels, _ = v6.fit_or_load(
                label_path,
                info_path,
                centers_path,
                space,
                k,
            )
            internal.append(
                internal_row(
                    space,
                    labels,
                    sample_indices,
                    sample_distances,
                    mode,
                    scope,
                    k,
                )
            )
            supervised.extend(supervised_rows(local_metadata, labels, mode, scope, k))
            persist_clustering(
                output,
                local_metadata,
                mode,
                scope,
                k,
                labels,
                centers_path,
            )
        del sample_distances, space
        gc.collect()

    internal_frame = pd.DataFrame(internal)
    supervised_frame = pd.DataFrame(supervised)
    internal_frame.to_csv(metrics_dir / "internal_metrics.csv", index=False)
    supervised_frame.to_csv(metrics_dir / "supervised_metrics.csv", index=False)
    batch_all = compute_batch_correction_metrics(
        embedding,
        metadata["section"].to_numpy(),
        dataset="SPATCH",
        method=f"spa_mo_model_result_{model_version}",
        batch_label_name="section",
        max_samples=100000,
        asw_sample_size=10000,
        lisi_neighbors=90,
        kbet_neighbors=50,
        seed=42,
        pcr_components=50,
    )
    batch_frame = pd.DataFrame(
        [
            {
                "dataset": "SPATCH",
                "n_obs": N_OBS,
                **{metric: batch_all[metric] for metric in BATCH_METRICS},
            }
        ]
    )
    batch_frame.to_csv(metrics_dir / "batch_metrics.csv", index=False)
    np.savez_compressed(output / "scaler_parameters.npz", **scaler_arrays)
    config = {
        "dataset": "SPATCH",
        "model_version": model_version,
        "post_ot_graphsage_scale": scale,
        "preprocessing": "standardized_embedding",
        "clustering": {
            "implementation": "sklearn.cluster.MiniBatchKMeans",
            "k_values": list(K_VALUES),
            "modes": ["joint", "independent"],
            "seed": v6.SEED,
            "n_init": v6.N_INIT,
            "max_iter": v6.MAX_ITER,
            "batch_size": v6.BATCH_SIZE,
        },
        "metrics": {
            "supervised": list(SUPERVISED_METRICS),
            "unsupervised": list(INTERNAL_METRICS),
            "batch": list(BATCH_METRICS),
        },
        "cluster_asw_sample_size": v6.ASW_SAMPLE_SIZE,
        "spatial_cluster_plots": True,
        "umap_computed": False,
        "umap_plots": False,
        "source_run_summary": str(summary_path),
        "source_run_summary_sha256": sha256_file(summary_path),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completion_payload = {
        "status": "complete",
        "dataset": "SPATCH",
        "model_version": model_version,
        "post_ot_graphsage_scale": scale,
        "analysis": str(output),
        "k_values": list(K_VALUES),
        "internal_metrics": str(metrics_dir / "internal_metrics.csv"),
        "supervised_metrics": str(metrics_dir / "supervised_metrics.csv"),
        "batch_metrics": str(metrics_dir / "batch_metrics.csv"),
        "spatial_plot_count": len(list((output / "clustering").glob("*/spatial_*.png"))),
        "umap_computed": False,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    completion.write_text(
        json.dumps(completion_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = result_root / "spatch_requested_metrics_manifest.json"
    manifest_path.write_text(
        json.dumps(completion_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion_payload, indent=2, ensure_ascii=False), flush=True)
    print(f"SPATCH_{model_version}_REQUESTED_ANALYSIS_PASS", flush=True)
    return completion_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--post-ot-graphsage-scale", type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("Scale must be finite and non-negative.")
    analyze(args.result_root.resolve(), args.model_version, args.post_ot_graphsage_scale)


if __name__ == "__main__":
    main()
