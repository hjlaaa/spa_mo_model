#!/usr/bin/env python3
"""CRC Stereo-CITE-seq final-embedding clustering analysis."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_correction_metrics import compute_batch_correction_metrics


SECTION_ORDER = ["CRC_003", "CRC_006"]


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster CRC Stereo-CITE-seq final embeddings.")
    parser.add_argument("--input_dir", required=True, help="Directory produced by run_crc_stereocite.py.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to input_dir/clustering_analysis.",
    )
    parser.add_argument("--n_clusters", default="5,8,10,15", help="Comma-separated KMeans cluster counts.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--point_size", type=float, default=0.6)
    parser.add_argument(
        "--kmeans_method",
        choices=["minibatch", "kmeans"],
        default="minibatch",
        help="Use minibatch for full-spot runs; use kmeans for exact smaller analyses.",
    )
    parser.add_argument("--batch_size", type=int, default=8192, help="MiniBatchKMeans batch size.")
    parser.add_argument("--n_init", type=int, default=10, help="Number of KMeans initializations.")
    parser.add_argument("--max_iter", type=int, default=300, help="Maximum KMeans iterations.")
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=0,
        help="Optional max points per section for PNG plotting only. 0 plots all points.",
    )
    parser.add_argument(
        "--batch_metrics_max_samples",
        "--batch_metric_max_samples",
        "--batch_metric_sample_size",
        type=int,
        default=50000,
        help="Maximum spots used for batch metrics; <=0 uses all valid spots.",
    )
    parser.add_argument("--batch_asw_sample_size", type=int, default=10000)
    parser.add_argument(
        "--batch_lisi_neighbors",
        "--batch_metric_neighbors",
        type=int,
        default=90,
    )
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


def parse_cluster_counts(text: str) -> list[int]:
    counts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not counts:
        raise ValueError("--n_clusters must contain at least one integer.")
    if any(count <= 1 for count in counts):
        raise ValueError("All cluster counts must be greater than 1.")
    return counts


def resolve_existing_path(input_dir: Path, candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{joined}")


def load_array(input_dir: Path, section: str, kind: str, summary: dict[str, Any] | None) -> tuple[np.ndarray, str]:
    saved_files = (summary or {}).get("saved_files", {})
    summary_path = None
    if kind == "embedding":
        summary_path = (saved_files.get("final_embeddings") or {}).get(section)
        candidates = [
            input_dir / f"final_embeddings_{section}.npy",
            input_dir / "final_embeddings" / f"{section}_final_embedding.npy",
        ]
    elif kind == "spatial":
        summary_path = (saved_files.get("spatial") or {}).get(section)
        candidates = [input_dir / f"spatial_{section}.npy"]
    elif kind == "spot_index":
        summary_path = (saved_files.get("selected_spot_indices") or {}).get(section)
        candidates = [input_dir / f"selected_spot_indices_{section}.npy"]
    else:
        raise ValueError(f"Unsupported array kind: {kind}")

    if summary_path:
        candidates.insert(0, Path(summary_path))
    path = resolve_existing_path(input_dir, candidates, f"{kind} for {section}")
    return np.load(path), str(path)


def load_inputs(input_dir: Path):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")
    summary = load_json(summary_path)
    loss_history_path = input_dir / "loss_history.json"
    loss_history = load_json(loss_history_path) if loss_history_path.exists() else None

    embeddings = {}
    spatial = {}
    spot_indices = {}
    source_paths = {"run_summary": str(summary_path)}
    if loss_history_path.exists():
        source_paths["loss_history"] = str(loss_history_path)

    for section in SECTION_ORDER:
        embeddings[section], source_paths[f"embedding_{section}"] = load_array(input_dir, section, "embedding", summary)
        spatial[section], source_paths[f"spatial_{section}"] = load_array(input_dir, section, "spatial", summary)
        spot_indices[section], source_paths[f"spot_index_{section}"] = load_array(input_dir, section, "spot_index", summary)
        if embeddings[section].ndim != 2:
            raise ValueError(f"{section} embedding must be 2D, got {embeddings[section].shape}.")
        if spatial[section].ndim != 2 or spatial[section].shape[1] < 2:
            raise ValueError(f"{section} spatial must be [N, >=2], got {spatial[section].shape}.")
        n_spots = embeddings[section].shape[0]
        if spatial[section].shape[0] != n_spots:
            raise ValueError(f"{section} embedding/spatial rows differ: {n_spots} vs {spatial[section].shape[0]}.")
        if spot_indices[section].shape[0] != n_spots:
            raise ValueError(
                f"{section} embedding/spot index rows differ: {n_spots} vs {spot_indices[section].shape[0]}."
            )

    return embeddings, spatial, spot_indices, summary, loss_history, source_paths


def save_label_csv(path: Path, labels: np.ndarray, coords: np.ndarray, spot_indices: np.ndarray) -> None:
    table = pd.DataFrame(
        {
            "row_index": np.arange(labels.shape[0], dtype=np.int64),
            "selected_spot_index": spot_indices.astype(np.int64, copy=False),
            "cluster": labels.astype(np.int64, copy=False),
            "x": coords[:, 0].astype(float, copy=False),
            "y": coords[:, 1].astype(float, copy=False),
        }
    )
    table.to_csv(path, index=False)


def save_count_csv(path: Path, labels_by_section: dict[str, np.ndarray], n_clusters: int, mode: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "section", "cluster", "count", "section_fraction"])
        for section, labels in labels_by_section.items():
            counts = np.bincount(labels.astype(int), minlength=n_clusters)
            total = int(labels.shape[0])
            for cluster, count in enumerate(counts):
                writer.writerow([mode, section, cluster, int(count), float(count / total)])


def _plot_indices(n_points: int, max_points: int, seed: int) -> np.ndarray | slice:
    if max_points <= 0 or max_points >= n_points:
        return slice(None)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def plot_spatial(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    n_clusters: int,
    point_size: float,
    dpi: int,
    plot_max_points: int,
    seed: int,
):
    plot_idx = _plot_indices(labels.shape[0], plot_max_points, seed)
    coords_plot = coords[plot_idx]
    labels_plot = labels[plot_idx]
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        c=labels_plot,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
        rasterized=True,
    )
    suffix = "" if isinstance(plot_idx, slice) else f" ({labels_plot.shape[0]:,} plotted)"
    ax.set_title(title + suffix)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fit_kmeans(embedding: np.ndarray, n_clusters: int, seed: int, args) -> np.ndarray:
    if args.kmeans_method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
            batch_size=int(args.batch_size),
            reassignment_ratio=0.01,
        )
    return model.fit_predict(embedding).astype(int)


def save_batch_correction_metrics(
    output_dir: Path,
    embeddings: dict[str, np.ndarray],
    args,
) -> tuple[dict[str, Any], str]:
    stacked = np.vstack([embeddings[section] for section in SECTION_ORDER])
    batch_labels = np.concatenate(
        [
            np.full(embeddings[section].shape[0], section, dtype=object)
            for section in SECTION_ORDER
        ]
    )
    metrics = compute_batch_correction_metrics(
        stacked,
        batch_labels,
        dataset="CRC Stereo-CITE-seq",
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


def run_joint_kmeans(
    embeddings: dict[str, np.ndarray],
    spatial: dict[str, np.ndarray],
    spot_indices: dict[str, np.ndarray],
    output_dir: Path,
    n_clusters: int,
    seed: int,
    point_size: float,
    dpi: int,
    args,
) -> dict[str, Any]:
    mode_dir = output_dir / f"joint_k{n_clusters}"
    ensure_dir(mode_dir)
    stacked = np.vstack([embeddings[section] for section in SECTION_ORDER])
    labels_all = fit_kmeans(stacked, n_clusters=n_clusters, seed=seed, args=args)

    labels_by_section = {}
    start = 0
    output_files = []
    for section in SECTION_ORDER:
        end = start + embeddings[section].shape[0]
        labels = labels_all[start:end]
        labels_by_section[section] = labels
        start = end
        label_path = mode_dir / f"joint_k{n_clusters}_{section}_labels.csv"
        png_path = mode_dir / f"joint_k{n_clusters}_{section}_spatial.png"
        save_label_csv(label_path, labels, spatial[section], spot_indices[section])
        plot_spatial(
            png_path,
            spatial[section],
            labels,
            f"CRC joint KMeans k={n_clusters} - {section}",
            n_clusters,
            point_size,
            dpi,
            int(args.plot_max_points),
            seed,
        )
        output_files.extend([str(label_path), str(png_path)])

    count_path = mode_dir / f"joint_k{n_clusters}_cluster_counts.csv"
    save_count_csv(count_path, labels_by_section, n_clusters, "joint")
    output_files.append(str(count_path))
    return {
        "mode": "joint",
        "n_clusters": n_clusters,
        "kmeans_method": args.kmeans_method,
        "output_dir": str(mode_dir),
        "files": output_files,
    }


def run_independent_kmeans(
    embeddings: dict[str, np.ndarray],
    spatial: dict[str, np.ndarray],
    spot_indices: dict[str, np.ndarray],
    output_dir: Path,
    n_clusters: int,
    seed: int,
    point_size: float,
    dpi: int,
    args,
) -> dict[str, Any]:
    mode_dir = output_dir / f"independent_k{n_clusters}"
    ensure_dir(mode_dir)
    labels_by_section = {}
    output_files = []
    for section in SECTION_ORDER:
        labels = fit_kmeans(embeddings[section], n_clusters=n_clusters, seed=seed, args=args)
        labels_by_section[section] = labels
        label_path = mode_dir / f"independent_k{n_clusters}_{section}_labels.csv"
        png_path = mode_dir / f"independent_k{n_clusters}_{section}_spatial.png"
        save_label_csv(label_path, labels, spatial[section], spot_indices[section])
        plot_spatial(
            png_path,
            spatial[section],
            labels,
            f"CRC independent KMeans k={n_clusters} - {section}",
            n_clusters,
            point_size,
            dpi,
            int(args.plot_max_points),
            seed,
        )
        output_files.extend([str(label_path), str(png_path)])

    count_path = mode_dir / f"independent_k{n_clusters}_cluster_counts.csv"
    save_count_csv(count_path, labels_by_section, n_clusters, "independent")
    output_files.append(str(count_path))
    return {
        "mode": "independent",
        "n_clusters": n_clusters,
        "kmeans_method": args.kmeans_method,
        "output_dir": str(mode_dir),
        "files": output_files,
    }


def plot_loss_history(output_dir: Path, loss_history: list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(loss_history, list) or not loss_history:
        return []
    loss_dir = output_dir / "training_curves"
    ensure_dir(loss_dir)
    epochs = np.array([record.get("epoch") for record in loss_history], dtype=float)
    files = []
    for key in ["total_loss", "crossview_loss", "reconstruction_loss", "weighted_crossview_loss"]:
        values = np.array([record.get(key, np.nan) for record in loss_history], dtype=float)
        if np.all(np.isnan(values)):
            continue
        path = loss_dir / f"{key}.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, values, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"CRC training {key}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    return files


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "clustering_analysis"
    ensure_dir(output_dir)

    n_clusters_list = parse_cluster_counts(args.n_clusters)
    embeddings, spatial, spot_indices, run_summary, loss_history, source_paths = load_inputs(input_dir)
    batch_metrics, batch_metrics_path = save_batch_correction_metrics(output_dir, embeddings, args)

    results = []
    training_curve_files = plot_loss_history(output_dir, loss_history)
    for n_clusters in n_clusters_list:
        total_n = sum(embeddings[section].shape[0] for section in SECTION_ORDER)
        if n_clusters > total_n:
            raise ValueError(f"n_clusters={n_clusters} exceeds total spot count {total_n}.")
        for section in SECTION_ORDER:
            if n_clusters > embeddings[section].shape[0]:
                raise ValueError(f"n_clusters={n_clusters} exceeds {section} spot count {embeddings[section].shape[0]}.")
        results.append(
            run_joint_kmeans(
                embeddings,
                spatial,
                spot_indices,
                output_dir,
                n_clusters,
                args.seed,
                args.point_size,
                args.dpi,
                args,
            )
        )
        results.append(
            run_independent_kmeans(
                embeddings,
                spatial,
                spot_indices,
                output_dir,
                n_clusters,
                args.seed,
                args.point_size,
                args.dpi,
                args,
            )
        )

    output_files = [file_path for result in results for file_path in result["files"]]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "n_clusters": n_clusters_list,
        "kmeans_method": args.kmeans_method,
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "plot_max_points": int(args.plot_max_points),
        "point_size": float(args.point_size),
        "dpi": int(args.dpi),
        "sections": SECTION_ORDER,
        "embedding_shapes": {section: list(embeddings[section].shape) for section in SECTION_ORDER},
        "spatial_shapes": {section: list(spatial[section].shape) for section in SECTION_ORDER},
        "selected_spot_index_shapes": {section: list(spot_indices[section].shape) for section in SECTION_ORDER},
        "source_paths": source_paths,
        "run_summary_mode": run_summary.get("mode"),
        "loss_history_epochs": len(loss_history) if isinstance(loss_history, list) else None,
        "batch_correction_metrics_path": batch_metrics_path,
        "batch_correction_metrics": batch_metrics,
        "training_curve_files": training_curve_files,
        "results": results,
        "output_files": output_files + training_curve_files + [batch_metrics_path],
    }
    summary_path = output_dir / "clustering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    output_files.append(str(summary_path))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("CRC_STEREOCITE_CLUSTERING: PASS")


if __name__ == "__main__":
    main()
