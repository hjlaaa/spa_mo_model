#!/usr/bin/env python3
"""MouseBrain final-embedding clustering analysis.

This script follows the COSIE downstream style of applying KMeans to final
embeddings and plotting spatial cluster maps, while adapting IO to this
project's saved MouseBrain V2 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_correction_metrics import compute_batch_correction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster MouseBrain final embeddings.")
    parser.add_argument("--embedding_dir", required=True, help="Directory containing s*_final_embedding.npy files.")
    parser.add_argument("--config", required=True, help="MouseBrain preprocessing/training config JSON.")
    parser.add_argument("--output_dir", required=True, help="Output directory for clustering results.")
    parser.add_argument("--cluster_list", default="5,6,8,10", help="Comma-separated KMeans cluster counts.")
    parser.add_argument("--random_state", type=int, default=0, help="KMeans random state.")
    parser.add_argument(
        "--metric_seed",
        type=int,
        default=0,
        help="Fixed subsampling seed for embedding metrics; independent of KMeans seed.",
    )
    parser.add_argument(
        "--label_keys",
        default="annotations,RegionLoupe",
        help="Comma-separated biological obs columns used for ARI/NMI.",
    )
    parser.add_argument("--spatial_key", default=None, help="Override spatial key; defaults to config preprocessing.spatial_key.")
    parser.add_argument("--dpi", type=int, default=250, help="Spatial plot DPI.")
    parser.add_argument("--point_size", type=float, default=8.0, help="Spatial scatter point size.")
    parser.add_argument("--no_invert_y", action="store_true", help="Do not invert y axis in spatial plots.")
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


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_cluster_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--cluster_list must contain at least one integer.")
    if any(value <= 1 for value in values):
        raise ValueError("All cluster counts must be greater than 1.")
    return values


def parse_label_keys(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--label_keys must contain at least one obs column.")
    if len(set(values)) != len(values):
        raise ValueError("--label_keys contains duplicate columns.")
    return values


def load_embeddings(embedding_dir: Path, section_order: list[str]) -> dict[str, np.ndarray]:
    embeddings = {}
    for section in section_order:
        path = embedding_dir / f"{section}_final_embedding.npy"
        if not path.exists():
            raise FileNotFoundError(f"200 epoch final embedding not found: {path}")
        emb = np.load(path)
        if emb.ndim != 2:
            raise ValueError(f"Embedding for {section} must be 2D, got shape {emb.shape}.")
        embeddings[section] = emb
    return embeddings


def load_spatial_from_config(
    config: dict[str, Any],
    section_order: list[str],
    spatial_key: str,
    label_keys: list[str],
):
    section_by_id = {section["section_id"]: section for section in config["sections"]}
    spatial = {}
    obs_names = {}
    biological_labels = {}
    source_paths = {}
    for section in section_order:
        if section not in section_by_id:
            raise KeyError(f"Section {section} is missing from config sections.")
        rna_path = Path(section_by_id[section]["rna_input"])
        if not rna_path.exists():
            raise FileNotFoundError(f"RNA h5ad for {section} not found: {rna_path}")
        adata = ad.read_h5ad(rna_path)
        if spatial_key not in adata.obsm:
            raise KeyError(f"{section} RNA h5ad does not contain obsm[{spatial_key!r}].")
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"Spatial coordinates for {section} must be [N, >=2], got {coords.shape}.")
        spatial[section] = coords[:, :2].copy()
        obs_names[section] = np.asarray(adata.obs_names.astype(str))
        missing_label_keys = [key for key in label_keys if key not in adata.obs]
        if missing_label_keys:
            raise KeyError(
                f"{section} RNA h5ad is missing biological label columns: "
                f"{missing_label_keys}."
            )
        biological_labels[section] = {
            key: adata.obs[key].astype(object).to_numpy(copy=True)
            for key in label_keys
        }
        source_paths[section] = str(rna_path)
    return spatial, obs_names, biological_labels, source_paths


def validate_alignment(embeddings, spatial, obs_names):
    info = {}
    for section, emb in embeddings.items():
        if section not in spatial:
            raise KeyError(f"Missing spatial coordinates for section {section}.")
        if emb.shape[0] != spatial[section].shape[0]:
            raise ValueError(
                f"Embedding/spatial row mismatch for {section}: "
                f"{emb.shape[0]} vs {spatial[section].shape[0]}."
            )
        if emb.shape[0] != len(obs_names[section]):
            raise ValueError(
                f"Embedding/obs_names row mismatch for {section}: "
                f"{emb.shape[0]} vs {len(obs_names[section])}."
            )
        info[section] = {
            "embedding_shape": list(emb.shape),
            "spatial_shape": list(spatial[section].shape),
            "obs_names_count": int(len(obs_names[section])),
        }
    return info


def save_label_csv(path: Path, labels, coords, obs_names):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spot_index", "obs_name", "cluster", "x", "y"])
        for idx, (label, coord, obs_name) in enumerate(zip(labels, coords, obs_names)):
            writer.writerow([idx, obs_name, int(label), float(coord[0]), float(coord[1])])


def plot_spatial(path: Path, coords, labels, title: str, n_clusters: int, point_size: float, invert_y: bool, dpi: int):
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_rows(path: Path, header: list[str], rows: list[list[Any]]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def compute_section_cluster_counts(labels_by_section: dict[str, np.ndarray], n_clusters: int):
    rows = []
    for section, labels in labels_by_section.items():
        total = len(labels)
        counts = np.bincount(labels.astype(int), minlength=n_clusters)
        for cluster, count in enumerate(counts):
            rows.append([section, cluster, int(count), float(count / total)])
    return rows


def compute_joint_composition(labels_by_section: dict[str, np.ndarray], n_clusters: int, section_order: list[str]):
    rows = []
    for cluster in range(n_clusters):
        counts = [int(np.sum(labels_by_section[section] == cluster)) for section in section_order]
        total = sum(counts)
        fractions = [float(count / total) if total else 0.0 for count in counts]
        rows.append([cluster, *counts, total, *fractions])
    return rows


def spatial_neighbor_agreement(coords, labels, k: int = 6):
    if len(labels) <= 1:
        return float("nan")
    effective_k = min(k + 1, len(labels))
    nn = NearestNeighbors(n_neighbors=effective_k)
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)
    neighbor_indices = indices[:, 1:]
    same = labels[neighbor_indices] == labels[:, None]
    return float(np.mean(same))


def compute_biological_label_metrics(
    labels_by_section: dict[str, np.ndarray],
    biological_labels: dict[str, dict[str, np.ndarray]],
    section_order: list[str],
    label_keys: list[str],
    mode: str,
    n_clusters: int,
) -> list[dict[str, Any]]:
    """Compute fixed-label ARI/NMI without selecting K from the scores."""

    rows: list[dict[str, Any]] = []
    for label_key in label_keys:
        section_rows: list[dict[str, Any]] = []
        pooled_true: list[np.ndarray] = []
        pooled_pred: list[np.ndarray] = []
        for section in section_order:
            true_values = np.asarray(biological_labels[section][label_key], dtype=object)
            predicted = np.asarray(labels_by_section[section], dtype=int)
            if true_values.shape[0] != predicted.shape[0]:
                raise ValueError(
                    f"{section} label/prediction row mismatch for {label_key}: "
                    f"{true_values.shape[0]} vs {predicted.shape[0]}."
                )
            valid = np.asarray(pd.notna(true_values), dtype=bool)
            true_valid = true_values[valid].astype(str)
            pred_valid = predicted[valid]
            if true_valid.size < 2 or np.unique(true_valid).size < 2:
                raise ValueError(
                    f"{section} label {label_key!r} needs at least two valid classes."
                )
            row = {
                "mode": mode,
                "n_clusters": int(n_clusters),
                "label_key": label_key,
                "scope": "section",
                "section": section,
                "n_spots": int(true_valid.size),
                "n_true_classes": int(np.unique(true_valid).size),
                "n_predicted_clusters": int(np.unique(pred_valid).size),
                "ari": float(adjusted_rand_score(true_valid, pred_valid)),
                "nmi": float(
                    normalized_mutual_info_score(
                        true_valid,
                        pred_valid,
                        average_method="arithmetic",
                    )
                ),
            }
            rows.append(row)
            section_rows.append(row)
            pooled_true.append(true_valid)
            pooled_pred.append(pred_valid)

        weights = np.asarray([row["n_spots"] for row in section_rows], dtype=float)
        for scope, weighted in (("section_macro", False), ("section_weighted", True)):
            aggregate = {
                "mode": mode,
                "n_clusters": int(n_clusters),
                "label_key": label_key,
                "scope": scope,
                "section": "__all_sections__",
                "n_spots": int(weights.sum()),
                "n_true_classes": int(
                    np.unique(np.concatenate(pooled_true)).size
                ),
                "n_predicted_clusters": int(n_clusters),
                "ari": float(
                    np.average(
                        [row["ari"] for row in section_rows],
                        weights=weights if weighted else None,
                    )
                ),
                "nmi": float(
                    np.average(
                        [row["nmi"] for row in section_rows],
                        weights=weights if weighted else None,
                    )
                ),
            }
            rows.append(aggregate)

        # Joint KMeans has one globally consistent cluster-ID space, so a
        # pooled score is meaningful. Independent section-wise KMeans does not.
        if mode == "joint":
            true_all = np.concatenate(pooled_true)
            pred_all = np.concatenate(pooled_pred)
            rows.append(
                {
                    "mode": mode,
                    "n_clusters": int(n_clusters),
                    "label_key": label_key,
                    "scope": "pooled",
                    "section": "__all_sections__",
                    "n_spots": int(true_all.size),
                    "n_true_classes": int(np.unique(true_all).size),
                    "n_predicted_clusters": int(np.unique(pred_all).size),
                    "ari": float(adjusted_rand_score(true_all, pred_all)),
                    "nmi": float(
                        normalized_mutual_info_score(
                            true_all,
                            pred_all,
                            average_method="arithmetic",
                        )
                    ),
                }
            )
    return rows


def save_batch_correction_metrics(
    output_dir: Path,
    embeddings: dict[str, np.ndarray],
    section_order: list[str],
    args,
) -> tuple[dict[str, Any], str]:
    stacked = np.vstack([embeddings[section] for section in section_order])
    batch_labels = np.concatenate(
        [
            np.full(embeddings[section].shape[0], section, dtype=object)
            for section in section_order
        ]
    )
    metrics = compute_batch_correction_metrics(
        stacked,
        batch_labels,
        dataset="MouseBrain",
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


def run_joint_clustering(
    embeddings,
    spatial,
    obs_names,
    section_order,
    n_clusters,
    random_state,
    output_dir,
    point_size,
    invert_y,
    dpi,
    biological_labels,
    label_keys,
):
    ensure_dir(output_dir)
    stacked = np.vstack([embeddings[section] for section in section_order])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    all_labels = kmeans.fit_predict(stacked).astype(int)

    labels_by_section = {}
    start = 0
    for section in section_order:
        end = start + embeddings[section].shape[0]
        labels = all_labels[start:end]
        labels_by_section[section] = labels
        start = end
        save_label_csv(output_dir / f"labels_{section}.csv", labels, spatial[section], obs_names[section])
        plot_spatial(
            output_dir / f"spatial_{section}.png",
            spatial[section],
            labels,
            title=f"MouseBrain joint KMeans k={n_clusters} - {section}",
            n_clusters=n_clusters,
            point_size=point_size,
            invert_y=invert_y,
            dpi=dpi,
        )

    composition_header = (
        ["cluster"]
        + [f"{section}_count" for section in section_order]
        + ["total_count"]
        + [f"{section}_fraction" for section in section_order]
    )
    composition_rows = compute_joint_composition(labels_by_section, n_clusters, section_order)
    write_rows(output_dir / "cluster_composition.csv", composition_header, composition_rows)
    write_rows(
        output_dir / "section_cluster_count.csv",
        ["section", "cluster", "count", "section_fraction"],
        compute_section_cluster_counts(labels_by_section, n_clusters),
    )
    continuity_rows = [
        [section, spatial_neighbor_agreement(spatial[section], labels_by_section[section])]
        for section in section_order
    ]
    write_rows(output_dir / "spatial_continuity.csv", ["section", "neighbor_same_cluster_fraction"], continuity_rows)
    biological_metric_rows = compute_biological_label_metrics(
        labels_by_section,
        biological_labels,
        section_order,
        label_keys,
        mode="joint",
        n_clusters=n_clusters,
    )
    biological_metrics_path = output_dir / "biological_label_metrics.csv"
    pd.DataFrame(biological_metric_rows).to_csv(biological_metrics_path, index=False)
    return {
        "output_dir": str(output_dir),
        "composition_path": str(output_dir / "cluster_composition.csv"),
        "section_cluster_count_path": str(output_dir / "section_cluster_count.csv"),
        "spatial_continuity_path": str(output_dir / "spatial_continuity.csv"),
        "max_section_fraction_by_cluster": [
            max(row[-len(section_order) :]) for row in composition_rows
        ],
        "mean_spatial_neighbor_agreement": float(np.nanmean([row[1] for row in continuity_rows])),
        "biological_label_metrics_path": str(biological_metrics_path),
        "biological_label_metrics": biological_metric_rows,
    }


def run_independent_clustering(
    embeddings,
    spatial,
    obs_names,
    section_order,
    n_clusters,
    random_state,
    output_dir,
    point_size,
    invert_y,
    dpi,
    biological_labels,
    label_keys,
):
    ensure_dir(output_dir)
    labels_by_section = {}
    for section in section_order:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(embeddings[section]).astype(int)
        labels_by_section[section] = labels
        save_label_csv(output_dir / f"labels_{section}.csv", labels, spatial[section], obs_names[section])
        plot_spatial(
            output_dir / f"spatial_{section}.png",
            spatial[section],
            labels,
            title=f"MouseBrain independent KMeans k={n_clusters} - {section}",
            n_clusters=n_clusters,
            point_size=point_size,
            invert_y=invert_y,
            dpi=dpi,
        )
    write_rows(
        output_dir / "section_cluster_count.csv",
        ["section", "cluster", "count", "section_fraction"],
        compute_section_cluster_counts(labels_by_section, n_clusters),
    )
    continuity_rows = [
        [section, spatial_neighbor_agreement(spatial[section], labels_by_section[section])]
        for section in section_order
    ]
    write_rows(output_dir / "spatial_continuity.csv", ["section", "neighbor_same_cluster_fraction"], continuity_rows)
    biological_metric_rows = compute_biological_label_metrics(
        labels_by_section,
        biological_labels,
        section_order,
        label_keys,
        mode="independent",
        n_clusters=n_clusters,
    )
    biological_metrics_path = output_dir / "biological_label_metrics.csv"
    pd.DataFrame(biological_metric_rows).to_csv(biological_metrics_path, index=False)
    return {
        "output_dir": str(output_dir),
        "section_cluster_count_path": str(output_dir / "section_cluster_count.csv"),
        "spatial_continuity_path": str(output_dir / "spatial_continuity.csv"),
        "mean_spatial_neighbor_agreement": float(np.nanmean([row[1] for row in continuity_rows])),
        "biological_label_metrics_path": str(biological_metrics_path),
        "biological_label_metrics": biological_metric_rows,
    }


def main():
    args = parse_args()
    config = load_json(args.config)
    section_order = list(config.get("section_order") or [section["section_id"] for section in config["sections"]])
    spatial_key = args.spatial_key or config.get("preprocessing", {}).get("spatial_key", "spatial")
    cluster_list = parse_cluster_list(args.cluster_list)
    label_keys = parse_label_keys(args.label_keys)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    embeddings = load_embeddings(embedding_dir, section_order)
    spatial, obs_names, biological_labels, spatial_source_paths = load_spatial_from_config(
        config,
        section_order,
        spatial_key,
        label_keys,
    )
    input_info = validate_alignment(embeddings, spatial, obs_names)
    batch_metrics, batch_metrics_path = save_batch_correction_metrics(
        output_dir,
        embeddings,
        section_order,
        args,
    )

    summary = {
        "embedding_dir": str(embedding_dir),
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "section_order": section_order,
        "cluster_list": cluster_list,
        "random_state": args.random_state,
        "metric_seed": args.metric_seed,
        "label_keys": label_keys,
        "spatial_key": spatial_key,
        "invert_y": not args.no_invert_y,
        "inputs": input_info,
        "spatial_source_paths": spatial_source_paths,
        "batch_correction_metrics_path": batch_metrics_path,
        "batch_correction_metrics": batch_metrics,
        "joint": {},
        "independent": {},
    }

    for n_clusters in cluster_list:
        print(f"Running joint KMeans k={n_clusters}")
        joint_dir = output_dir / f"joint_k{n_clusters}"
        summary["joint"][str(n_clusters)] = run_joint_clustering(
            embeddings,
            spatial,
            obs_names,
            section_order,
            n_clusters,
            args.random_state,
            joint_dir,
            args.point_size,
            invert_y=not args.no_invert_y,
            dpi=args.dpi,
            biological_labels=biological_labels,
            label_keys=label_keys,
        )

        print(f"Running independent KMeans k={n_clusters}")
        independent_dir = output_dir / f"independent_k{n_clusters}"
        summary["independent"][str(n_clusters)] = run_independent_clustering(
            embeddings,
            spatial,
            obs_names,
            section_order,
            n_clusters,
            args.random_state,
            independent_dir,
            args.point_size,
            invert_y=not args.no_invert_y,
            dpi=args.dpi,
            biological_labels=biological_labels,
            label_keys=label_keys,
        )

    biological_metric_rows = []
    for mode in ("joint", "independent"):
        for n_clusters in cluster_list:
            biological_metric_rows.extend(
                summary[mode][str(n_clusters)]["biological_label_metrics"]
            )
    biological_metrics_path = output_dir / "biological_label_metrics.csv"
    pd.DataFrame(biological_metric_rows).to_csv(
        biological_metrics_path,
        index=False,
    )
    summary["biological_label_metrics_path"] = str(biological_metrics_path)

    with open(output_dir / "clustering_analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    from complete_spa_mo_analysis import complete_analysis

    complete_analysis(
        dataset="MouseBrain",
        run_dir=embedding_dir.parent,
        analysis_dir=output_dir.parent,
        clustering_dir=output_dir,
        sections=section_order,
        seed=int(args.metric_seed),
        metric_sample_size=5000,
        spatial_neighbor_k=6,
        spatial_coords_override=spatial,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("MOUSEBRAIN_CLUSTERING_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
