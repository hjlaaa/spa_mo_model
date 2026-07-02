#!/usr/bin/env python3
"""MISAR-seq final-embedding clustering and metric analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_correction_metrics import compute_batch_correction_metrics


DEFAULT_SECTION_ORDER = ["dataset4", "dataset3", "dataset2", "dataset1"]
DEFAULT_LABEL_KEYS = [
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MISAR-seq StageMultiModalModel embeddings.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--section_order", default=None)
    parser.add_argument("--n_clusters", default="8,10,12,14")
    parser.add_argument("--label_keys", default=",".join(DEFAULT_LABEL_KEYS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--point_size", type=float, default=8.0)
    parser.add_argument("--plot_max_points", type=int, default=0)
    parser.add_argument("--kmeans_method", choices=["minibatch", "kmeans"], default="kmeans")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--metric_sample_size", type=int, default=10000)
    parser.add_argument("--spatial_neighbor_k", type=int, default=6)
    parser.add_argument(
        "--batch_metrics_max_samples",
        "--batch_metric_max_samples",
        "--batch_metric_sample_size",
        type=int,
        default=0,
        help="Maximum spots used for batch metrics; <=0 uses all valid spots.",
    )
    parser.add_argument("--batch_asw_sample_size", type=int, default=10000)
    parser.add_argument("--batch_lisi_neighbors", "--batch_metric_neighbors", type=int, default=90)
    parser.add_argument("--kbet_neighbors", type=int, default=50)
    parser.add_argument("--batch_metric_seed", type=int, default=0)
    parser.add_argument("--kbet_alpha", type=float, default=0.05)
    parser.add_argument("--pcr_components", "--pcr_n_components", type=int, default=50)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Cluster list cannot be empty.")
    if any(value <= 1 for value in values):
        raise ValueError("Cluster counts must be greater than 1.")
    return values


def parse_str_list(text: str | None, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values or list(default)


def load_array(input_dir: Path, section: str, kind: str, summary: dict[str, Any]) -> tuple[np.ndarray, str]:
    saved = summary.get("saved_files", {})
    if kind == "embedding":
        from_summary = (saved.get("final_embeddings") or {}).get(section)
        candidates = [input_dir / f"final_embeddings_{section}.npy"]
    elif kind == "spatial":
        from_summary = (saved.get("spatial") or {}).get(section)
        candidates = [input_dir / f"spatial_{section}.npy"]
    elif kind == "spot_index":
        from_summary = (saved.get("selected_spot_indices") or {}).get(section)
        candidates = [input_dir / f"selected_spot_indices_{section}.npy"]
    else:
        raise ValueError(f"Unsupported array kind: {kind}")
    if from_summary:
        candidates.insert(0, Path(from_summary))
    for path in candidates:
        if path.exists():
            return np.load(path), str(path)
    raise FileNotFoundError(f"Could not find {kind} for {section}; tried {candidates}")


def load_obs_metadata(input_dir: Path, section: str, summary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    from_summary = ((summary.get("saved_files") or {}).get("obs_metadata") or {}).get(section)
    candidates = []
    if from_summary:
        candidates.append(Path(from_summary))
    candidates.append(input_dir / f"obs_metadata_{section}.csv")
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)
    raise FileNotFoundError(f"Could not find obs metadata for {section}; tried {candidates}")


def load_inputs(input_dir: Path, section_order_arg: str | None):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")
    summary = load_json(summary_path)
    section_order = parse_str_list(section_order_arg, summary.get("section_names") or DEFAULT_SECTION_ORDER)
    embeddings = {}
    spatial = {}
    spot_indices = {}
    obs_meta = {}
    source_paths = {"run_summary": str(summary_path)}
    for section in section_order:
        embeddings[section], source_paths[f"embedding_{section}"] = load_array(input_dir, section, "embedding", summary)
        spatial[section], source_paths[f"spatial_{section}"] = load_array(input_dir, section, "spatial", summary)
        spot_indices[section], source_paths[f"spot_index_{section}"] = load_array(input_dir, section, "spot_index", summary)
        obs_meta[section], source_paths[f"obs_metadata_{section}"] = load_obs_metadata(input_dir, section, summary)
        n = embeddings[section].shape[0]
        if spatial[section].shape[0] != n:
            raise ValueError(f"{section}: spatial rows {spatial[section].shape[0]} != embedding rows {n}.")
        if spot_indices[section].shape[0] != n:
            raise ValueError(f"{section}: spot index rows {spot_indices[section].shape[0]} != embedding rows {n}.")
        if obs_meta[section].shape[0] != n:
            raise ValueError(f"{section}: obs metadata rows {obs_meta[section].shape[0]} != embedding rows {n}.")
    return embeddings, spatial, spot_indices, obs_meta, summary, source_paths, section_order


def choose_kmeans(args, n_clusters: int):
    if args.kmeans_method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=int(args.n_init), max_iter=int(args.max_iter))
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        batch_size=int(args.batch_size),
        reassignment_ratio=0.01,
    )


def fit_predict(embedding: np.ndarray, n_clusters: int, args) -> np.ndarray:
    return choose_kmeans(args, n_clusters).fit_predict(embedding).astype(int)


def finite_label_mask(labels: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.Series(labels).astype("string")
    return arr.notna().to_numpy() & (arr.astype(str).to_numpy() != "nan")


def sampled_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def safe_embedding_cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict[str, float | int | None]:
    unique = np.unique(labels)
    metrics: dict[str, float | int | None] = {
        "n_clusters_observed": int(unique.shape[0]),
        "cluster_asw": None,
        "cluster_asw_scaled": None,
        "dbi": None,
        "calinski_harabasz": None,
    }
    if unique.shape[0] < 2 or unique.shape[0] >= embedding.shape[0]:
        return metrics
    idx = sampled_indices(embedding.shape[0], sample_size, seed)
    x = StandardScaler().fit_transform(embedding[idx])
    y = labels[idx]
    if np.unique(y).shape[0] < 2 or np.unique(y).shape[0] >= y.shape[0]:
        return metrics
    asw = float(silhouette_score(x, y))
    metrics["cluster_asw"] = asw
    metrics["cluster_asw_scaled"] = float((asw + 1.0) / 2.0)
    metrics["dbi"] = float(davies_bouldin_score(x, y))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, y))
    return metrics


def supervised_metrics(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    label_table: pd.DataFrame,
    label_keys: list[str],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for key in label_keys:
        if key not in label_table.columns:
            continue
        raw = label_table[key]
        mask = finite_label_mask(raw)
        if mask.sum() < 3:
            continue
        true = raw.astype(str).to_numpy()[mask]
        pred = cluster_labels[mask]
        row: dict[str, Any] = {
            "label_key": key,
            "n_labeled": int(mask.sum()),
            "n_true_labels": int(np.unique(true).shape[0]),
            "ari": float(adjusted_rand_score(true, pred)),
            "nmi": float(normalized_mutual_info_score(true, pred)),
            "label_asw": None,
            "label_asw_scaled": None,
        }
        if np.unique(true).shape[0] >= 2 and np.unique(true).shape[0] < true.shape[0]:
            idx_all = np.where(mask)[0]
            idx = sampled_indices(idx_all.shape[0], sample_size, seed)
            chosen = idx_all[idx]
            y = raw.astype(str).to_numpy()[chosen]
            if np.unique(y).shape[0] >= 2 and np.unique(y).shape[0] < y.shape[0]:
                x = StandardScaler().fit_transform(embedding[chosen])
                asw = float(silhouette_score(x, y))
                row["label_asw"] = asw
                row["label_asw_scaled"] = float((asw + 1.0) / 2.0)
        rows.append(row)
    return rows


def spatial_neighbor_agreement(coords: np.ndarray, labels: np.ndarray, k: int) -> float | None:
    if coords.shape[0] <= 1:
        return None
    k_eff = max(1, min(int(k), coords.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(coords[:, :2])
    idx = nn.kneighbors(coords[:, :2], return_distance=False)[:, 1:]
    return float(np.mean(labels[idx] == labels[:, None]))


def plot_spatial(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    point_size: float,
    dpi: int,
    max_points: int,
    seed: int,
) -> None:
    idx = sampled_indices(coords.shape[0], max_points, seed) if max_points and max_points > 0 else np.arange(coords.shape[0])
    labels_subset = labels[idx].astype(str)
    categories, codes = np.unique(labels_subset, return_inverse=True)
    plt.figure(figsize=(5.2, 4.8))
    scatter = plt.scatter(
        coords[idx, 0],
        coords[idx, 1],
        c=codes,
        s=point_size,
        cmap="tab20",
        linewidths=0,
        alpha=0.95,
    )
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    if categories.shape[0] <= 20:
        handles, _ = scatter.legend_elements(num=categories.shape[0])
        plt.legend(
            handles,
            categories.tolist(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=6,
            frameon=False,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def write_labels(path: Path, section: str, spot_indices: np.ndarray, obs_meta: pd.DataFrame, labels: np.ndarray) -> None:
    table = pd.DataFrame(
        {
            "section": section,
            "spot_index": spot_indices.astype(int),
            "obs_name": obs_meta["obs_name"].astype(str) if "obs_name" in obs_meta else np.arange(len(labels)).astype(str),
            "cluster": labels.astype(int),
        }
    )
    for key in DEFAULT_LABEL_KEYS + ["Sample", "stage"]:
        if key in obs_meta.columns:
            table[key] = obs_meta[key].astype(str).to_numpy()
    table.to_csv(path, index=False)


def save_batch_correction_metrics(output_dir: Path, embeddings: dict[str, np.ndarray], section_order: list[str], args):
    stacked = np.vstack([embeddings[section] for section in section_order])
    batch_labels = np.concatenate(
        [np.full(embeddings[section].shape[0], section, dtype=object) for section in section_order]
    )
    metrics = compute_batch_correction_metrics(
        stacked,
        batch_labels,
        dataset="MISAR-seq",
        method="spa_mo_model",
        batch_label_name="section",
        max_samples=int(args.batch_metrics_max_samples),
        asw_sample_size=int(args.batch_asw_sample_size),
        lisi_neighbors=int(args.batch_lisi_neighbors),
        kbet_neighbors=int(args.kbet_neighbors),
        seed=int(args.batch_metric_seed),
        kbet_alpha=float(args.kbet_alpha),
        pcr_components=int(args.pcr_components),
    )
    path = output_dir / "batch_correction_metrics.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return metrics, str(path)


def run_joint(
    embeddings,
    spatial,
    spot_indices,
    obs_meta,
    output_dir,
    n_clusters,
    label_keys,
    args,
):
    out_dir = output_dir / f"joint_k{n_clusters}"
    ensure_dir(out_dir)
    section_order = list(embeddings.keys())
    stacked = np.vstack([embeddings[section] for section in section_order])
    labels_all = fit_predict(stacked, n_clusters, args)
    offsets = np.cumsum([0] + [embeddings[section].shape[0] for section in section_order])
    label_table = pd.concat([obs_meta[section] for section in section_order], ignore_index=True)

    files = []
    metrics_rows = []
    cluster_metrics = safe_embedding_cluster_metrics(stacked, labels_all, args.metric_sample_size, args.seed)
    for row in supervised_metrics(stacked, labels_all, label_table, label_keys, args.metric_sample_size, args.seed):
        metrics_rows.append({"mode": "joint", "n_clusters": n_clusters, **cluster_metrics, **row})
    metrics_rows.append(
        {
            "mode": "joint",
            "n_clusters": n_clusters,
            **cluster_metrics,
            "label_key": "section",
            "n_labeled": int(labels_all.shape[0]),
            "n_true_labels": int(len(section_order)),
            "ari": float(adjusted_rand_score(label_table["section"].astype(str), labels_all)),
            "nmi": float(normalized_mutual_info_score(label_table["section"].astype(str), labels_all)),
            "label_asw": None,
            "label_asw_scaled": None,
        }
    )

    counts = []
    for idx, section in enumerate(section_order):
        start, end = offsets[idx], offsets[idx + 1]
        section_labels = labels_all[start:end]
        label_path = out_dir / f"joint_k{n_clusters}_{section}_labels.csv"
        write_labels(label_path, section, spot_indices[section], obs_meta[section], section_labels)
        files.append(str(label_path))
        png_path = out_dir / f"joint_k{n_clusters}_{section}_spatial.png"
        plot_spatial(
            spatial[section],
            section_labels,
            f"joint k={n_clusters} {section}",
            png_path,
            args.point_size,
            args.dpi,
            args.plot_max_points,
            args.seed,
        )
        files.append(str(png_path))
        counts.extend(
            {
                "section": section,
                "cluster": int(cluster),
                "count": int(count),
                "fraction": float(count / len(section_labels)),
            }
            for cluster, count in zip(*np.unique(section_labels, return_counts=True))
        )
        metrics_rows.append(
            {
                "mode": "joint_section_spatial",
                "n_clusters": n_clusters,
                "section": section,
                "label_key": "cluster",
                "spatial_neighbor_agreement": spatial_neighbor_agreement(
                    spatial[section],
                    section_labels,
                    args.spatial_neighbor_k,
                ),
            }
        )
    count_path = out_dir / f"joint_k{n_clusters}_cluster_counts.csv"
    pd.DataFrame(counts).to_csv(count_path, index=False)
    files.append(str(count_path))
    return {"mode": "joint", "n_clusters": n_clusters, "files": files, "metrics": metrics_rows}


def run_independent(
    embeddings,
    spatial,
    spot_indices,
    obs_meta,
    output_dir,
    n_clusters,
    label_keys,
    args,
):
    out_dir = output_dir / f"independent_k{n_clusters}"
    ensure_dir(out_dir)
    files = []
    metrics_rows = []
    counts = []
    for section, embedding in embeddings.items():
        labels = fit_predict(embedding, n_clusters, args)
        label_path = out_dir / f"independent_k{n_clusters}_{section}_labels.csv"
        write_labels(label_path, section, spot_indices[section], obs_meta[section], labels)
        files.append(str(label_path))
        png_path = out_dir / f"independent_k{n_clusters}_{section}_spatial.png"
        plot_spatial(
            spatial[section],
            labels,
            f"independent k={n_clusters} {section}",
            png_path,
            args.point_size,
            args.dpi,
            args.plot_max_points,
            args.seed,
        )
        files.append(str(png_path))
        cluster_metrics = safe_embedding_cluster_metrics(embedding, labels, args.metric_sample_size, args.seed)
        for row in supervised_metrics(embedding, labels, obs_meta[section], label_keys, args.metric_sample_size, args.seed):
            metrics_rows.append({"mode": "independent", "section": section, "n_clusters": n_clusters, **cluster_metrics, **row})
        metrics_rows.append(
            {
                "mode": "independent_section_spatial",
                "section": section,
                "n_clusters": n_clusters,
                "label_key": "cluster",
                "spatial_neighbor_agreement": spatial_neighbor_agreement(spatial[section], labels, args.spatial_neighbor_k),
            }
        )
        counts.extend(
            {
                "section": section,
                "cluster": int(cluster),
                "count": int(count),
                "fraction": float(count / len(labels)),
            }
            for cluster, count in zip(*np.unique(labels, return_counts=True))
        )
    count_path = out_dir / f"independent_k{n_clusters}_cluster_counts.csv"
    pd.DataFrame(counts).to_csv(count_path, index=False)
    files.append(str(count_path))
    return {"mode": "independent", "n_clusters": n_clusters, "files": files, "metrics": metrics_rows}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "clustering_analysis"
    ensure_dir(output_dir)
    n_clusters_list = parse_int_list(args.n_clusters)
    label_keys = parse_str_list(args.label_keys, DEFAULT_LABEL_KEYS)
    embeddings, spatial, spot_indices, obs_meta, run_summary, source_paths, section_order = load_inputs(
        input_dir,
        args.section_order,
    )
    batch_metrics, batch_metrics_path = save_batch_correction_metrics(output_dir, embeddings, section_order, args)

    results = []
    all_metrics = []
    for n_clusters in n_clusters_list:
        results.append(
            run_joint(embeddings, spatial, spot_indices, obs_meta, output_dir, n_clusters, label_keys, args)
        )
        results.append(
            run_independent(embeddings, spatial, spot_indices, obs_meta, output_dir, n_clusters, label_keys, args)
        )
    for result in results:
        all_metrics.extend(result["metrics"])
    metrics_path = output_dir / "clustering_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

    output_files = [str(metrics_path), batch_metrics_path]
    for result in results:
        output_files.extend(result["files"])
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "section_order": section_order,
        "seed": int(args.seed),
        "n_clusters": n_clusters_list,
        "label_keys": label_keys,
        "kmeans_method": args.kmeans_method,
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "metric_sample_size": int(args.metric_sample_size),
        "embedding_shapes": {section: list(embeddings[section].shape) for section in section_order},
        "spatial_shapes": {section: list(spatial[section].shape) for section in section_order},
        "source_paths": source_paths,
        "run_summary_mode": run_summary.get("mode"),
        "run_summary_epochs": run_summary.get("epochs"),
        "batch_correction_metrics_path": batch_metrics_path,
        "batch_correction_metrics": batch_metrics,
        "clustering_metrics_path": str(metrics_path),
        "results": results,
        "output_files": output_files,
    }
    summary_path = output_dir / "clustering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("MISAR_SEQ_CLUSTERING_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
