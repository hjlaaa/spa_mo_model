#!/usr/bin/env python3
"""MISAR-style scalable analysis for full-spot HESTA RNA-only embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_correction_metrics import compute_batch_correction_metrics


DEFAULT_RUN = Path(
    "/home/hujinlan/spa_mo_model/result_v4C/human_embryo_rna_only/"
    "fullspot_200ep_bidirectional_sparse_uot_fused_poolfused_seed42"
)


def int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 2 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated integers >=2.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--joint-k", type=int_list, default=[5, 8, 10, 12, 14, 16])
    parser.add_argument("--independent-k", type=int_list, default=[5, 8, 10, 12, 14, 16])
    parser.add_argument("--plot-k", type=int_list, default=[5, 8, 10, 12, 14, 16])
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    # Keep this just above the scalable-neighbor threshold in
    # batch_correction_metrics.py so million-spot runs use HNSW rather than
    # constructing an exact 100k x 100k neighborhood search.
    parser.add_argument("--batch-metric-sample-size", type=int, default=100001)
    parser.add_argument("--batch-asw-sample-size", type=int, default=10000)
    parser.add_argument("--plot-sample-per-section", type=int, default=50000)
    parser.add_argument("--spatial-neighbor-k", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    if size <= 0 or size >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def load_inputs(run_dir: Path):
    summary = load_json(run_dir / "run_summary.json")
    if summary.get("status") != "PASS" or summary.get("mode") != "train":
        raise ValueError("Input run_summary does not describe a successful training run.")
    if summary.get("model_mode", {}).get("modalities") != ["RNA"]:
        raise ValueError("Expected an RNA-only model run.")
    manifest = summary["preprocess_manifest"]
    sections = list(summary["section_order"])
    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for section in sections:
        embedding_path = Path(summary["saved_files"]["embeddings"][section])
        annotation_path = Path(manifest["annotation_files"][section])
        embeddings[section] = np.load(embedding_path, mmap_mode="r")
        header = pd.read_csv(annotation_path, nrows=0).columns
        wanted = [
            name for name in
            ("section", "original_row", "obs_name", "cellid", "celltype", "stage", "x", "y")
            if name in header
        ]
        metadata[section] = pd.read_csv(annotation_path, usecols=wanted)
        if embeddings[section].ndim != 2 or embeddings[section].shape[1] != 128:
            raise ValueError(f"{section}: unexpected embedding shape {embeddings[section].shape}.")
        if len(embeddings[section]) != len(metadata[section]):
            raise ValueError(f"{section}: embedding/metadata row mismatch.")
    return summary, sections, embeddings, metadata


def fit_scaler_chunked(arrays: list[np.ndarray], chunk_size: int = 50000) -> StandardScaler:
    scaler = StandardScaler()
    for values in arrays:
        for start in range(0, len(values), chunk_size):
            scaler.partial_fit(np.asarray(values[start : start + chunk_size], dtype=np.float32))
    return scaler


def transform_to_memmap(
    path: Path,
    arrays: list[np.ndarray],
    scaler: StandardScaler,
    chunk_size: int = 50000,
) -> np.ndarray:
    n_rows = sum(len(values) for values in arrays)
    n_cols = int(arrays[0].shape[1])
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
    offset = 0
    for values in arrays:
        for start in range(0, len(values), chunk_size):
            chunk = np.asarray(values[start : start + chunk_size], dtype=np.float32)
            transformed = scaler.transform(chunk).astype(np.float32, copy=False)
            output[offset : offset + len(chunk)] = transformed
            offset += len(chunk)
    output.flush()
    del output
    return np.load(path, mmap_mode="r")


def prepare_spaces(
    output_dir: Path,
    sections: list[str],
    embeddings: dict[str, np.ndarray],
    reuse: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    joint_path = cache / "joint_standardized.npy"
    section_paths = {section: cache / f"independent_standardized_{section}.npy" for section in sections}
    scaler_path = cache / "scaler_parameters.npz"
    expected_joint_shape = (sum(len(embeddings[s]) for s in sections), 128)
    if reuse and joint_path.exists() and scaler_path.exists() and all(path.exists() for path in section_paths.values()):
        joint = np.load(joint_path, mmap_mode="r")
        independent = {section: np.load(path, mmap_mode="r") for section, path in section_paths.items()}
        if joint.shape != expected_joint_shape:
            raise ValueError("Cached joint standardized embedding has the wrong shape.")
        return joint, independent, {"reused": True, "scaler_file": str(scaler_path)}

    print("[analysis] fitting joint StandardScaler", flush=True)
    joint_scaler = fit_scaler_chunked([embeddings[s] for s in sections])
    joint = transform_to_memmap(joint_path, [embeddings[s] for s in sections], joint_scaler)
    scaler_values: dict[str, np.ndarray] = {
        "joint_mean": joint_scaler.mean_, "joint_scale": joint_scaler.scale_, "joint_var": joint_scaler.var_
    }
    independent = {}
    for section in sections:
        print(f"[analysis] standardizing independent scope {section}", flush=True)
        scaler = fit_scaler_chunked([embeddings[section]])
        independent[section] = transform_to_memmap(section_paths[section], [embeddings[section]], scaler)
        scaler_values[f"{section}_mean"] = scaler.mean_
        scaler_values[f"{section}_scale"] = scaler.scale_
        scaler_values[f"{section}_var"] = scaler.var_
    np.savez_compressed(scaler_path, **scaler_values)
    return joint, independent, {"reused": False, "scaler_file": str(scaler_path)}


def cluster_metrics(space: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    x = np.asarray(space[indices], dtype=np.float32)
    y = labels[indices]
    unique = np.unique(y)
    result = {
        "metric_n_obs": int(len(indices)),
        "n_clusters_observed": int(len(unique)),
        "cluster_asw": np.nan,
        "cluster_asw_scaled": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if 1 < len(unique) < len(y):
        asw = float(silhouette_score(x, y))
        result.update(
            cluster_asw=asw,
            cluster_asw_scaled=(asw + 1.0) / 2.0,
            calinski_harabasz=float(calinski_harabasz_score(x, y)),
            davies_bouldin=float(davies_bouldin_score(x, y)),
        )
    return result


def label_asw(space: np.ndarray, truth: np.ndarray, indices: np.ndarray) -> float:
    chosen_truth = truth[indices]
    valid = pd.notna(chosen_truth) & (chosen_truth.astype(str) != "nan")
    if valid.sum() < 3 or len(np.unique(chosen_truth[valid])) < 2:
        return float("nan")
    return float(silhouette_score(np.asarray(space[indices][valid]), chosen_truth[valid].astype(str)))


def external_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    valid = pd.notna(truth) & (truth.astype(str) != "nan")
    actual = truth[valid].astype(str)
    predicted = prediction[valid]
    if len(actual) < 2 or len(np.unique(actual)) < 2:
        return {
            "n_obs_labeled": int(len(actual)), "n_label_classes": int(len(np.unique(actual))),
            "ari": np.nan, "nmi": np.nan, "homogeneity": np.nan,
            "completeness": np.nan, "v_measure": np.nan,
        }
    return {
        "n_obs_labeled": int(len(actual)),
        "n_label_classes": int(len(np.unique(actual))),
        "ari": float(adjusted_rand_score(actual, predicted)),
        "nmi": float(normalized_mutual_info_score(actual, predicted)),
        "homogeneity": float(homogeneity_score(actual, predicted)),
        "completeness": float(completeness_score(actual, predicted)),
        "v_measure": float(v_measure_score(actual, predicted)),
    }


def fit_minibatch(space: np.ndarray, k: int, args: argparse.Namespace) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=int(k), random_state=args.seed, n_init=args.n_init,
        max_iter=args.max_iter, batch_size=args.batch_size,
        max_no_improvement=20, reassignment_ratio=0.01,
    )
    return model.fit_predict(space).astype(np.int32, copy=False)


def build_spatial_neighbors(
    output_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    k: int,
    reuse: bool,
) -> dict[str, np.ndarray]:
    cache = output_dir / "cache"
    result = {}
    for section in sections:
        path = cache / f"spatial_{k}nn_{section}.npy"
        if reuse and path.exists():
            result[section] = np.load(path, mmap_mode="r")
            continue
        coords = metadata[section][["x", "y"]].to_numpy(dtype=np.float64)
        k_eff = min(int(k), len(coords) - 1)
        print(f"[analysis] exact spatial {k_eff}-NN: {section} ({len(coords)} spots)", flush=True)
        indices = (
            NearestNeighbors(n_neighbors=k_eff + 1, algorithm="kd_tree", n_jobs=-1)
            .fit(coords)
            .kneighbors(coords, return_distance=False)[:, 1:]
        )
        np.save(path, indices.astype(np.int32))
        result[section] = np.load(path, mmap_mode="r")
    return result


def spatial_agreement(labels: np.ndarray, neighbors: np.ndarray) -> float:
    return float(np.mean(labels[np.asarray(neighbors)] == labels[:, None]))


def save_label_arrays(
    directory: Path,
    sections: list[str],
    labels: np.ndarray,
    offsets: dict[str, tuple[int, int]],
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    np.save(directory / "labels_all.npy", labels)
    files["all"] = str(directory / "labels_all.npy")
    for section in sections:
        start, end = offsets[section]
        path = directory / f"labels_{section}.npy"
        np.save(path, labels[start:end])
        files[section] = str(path)
    return files


def plot_spatial_panels(
    path: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray],
    title: str,
    maximum: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), squeeze=False)
    last = None
    for offset, section in enumerate(sections):
        axis = axes.ravel()[offset]
        frame = metadata[section]
        chosen = sample_indices(len(frame), maximum, seed + offset)
        last = axis.scatter(
            frame["x"].to_numpy()[chosen], frame["y"].to_numpy()[chosen],
            c=labels[section][chosen], s=1, cmap="tab20", linewidths=0, rasterized=True,
        )
        axis.set_title(section)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in axes.ravel()[len(sections):]:
        axis.axis("off")
    fig.suptitle(title)
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.55, label="cluster")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_ot(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted((run_dir / "ot_prior_topk").glob("*_metadata.json")):
        stem = metadata_path.name.removesuffix("_metadata.json")
        meta = load_json(metadata_path)
        arrays = {}
        for name in ("confidence", "row_mass", "raw_topk_mass", "topk_coverage", "tail_mass", "target_hit_count"):
            path = metadata_path.with_name(f"{stem}_{name}.npy")
            if path.exists():
                arrays[name] = np.load(path, mmap_mode="r")
        row: dict[str, Any] = {
            "pair": stem,
            "source_section": meta.get("source_section"),
            "target_section": meta.get("target_section"),
            "candidate_source": meta.get("candidate_source"),
            "context_source": meta.get("context_source"),
            "topology_context_weight": meta.get("topology_context_weight"),
            "semantic_cost_weight": meta.get("semantic_cost_weight"),
            "candidate_k": meta.get("candidate_k"),
            "attention_topk": meta.get("attention_topk"),
            "topology_topk_changed_fraction": meta.get("topology_topk_changed_fraction"),
        }
        for name, values in arrays.items():
            finite = np.asarray(values)[np.isfinite(values)]
            row[f"{name}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            row[f"{name}_median"] = float(np.median(finite)) if finite.size else np.nan
        if "target_hit_count" in arrays:
            hits = np.asarray(arrays["target_hit_count"])
            row["target_coverage_fraction"] = float(np.mean(hits > 0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics" / "ot_pair_metrics.csv", index=False)
    return frame


def training_diagnostics(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    history = load_json(run_dir / "loss_history.json")
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "metrics" / "training_loss_metrics.csv", index=False)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["epoch"], frame["total_loss"], label="total/reconstruction")
    axis.plot(frame["epoch"], frame["crossview_loss"], label="crossview")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / "training_loss_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "epochs": int(len(frame)),
        "first_total_loss": float(frame.iloc[0]["total_loss"]),
        "last_total_loss": float(frame.iloc[-1]["total_loss"]),
        "min_total_loss": float(frame["total_loss"].min()),
        "all_crossview_zero": bool((frame["crossview_loss"] == 0).all()),
    }


def make_summary(
    output_dir: Path,
    total_spots: int,
    sections: list[str],
    cluster_metrics_frame: pd.DataFrame,
    label_metrics_frame: pd.DataFrame,
    spatial_frame: pd.DataFrame,
    batch: dict[str, Any],
    ot_frame: pd.DataFrame,
    training: dict[str, Any],
) -> None:
    joint = cluster_metrics_frame[cluster_metrics_frame["mode"].eq("joint")]
    best_internal = joint.loc[joint["cluster_asw"].idxmax()]
    joint_celltype = label_metrics_frame[
        label_metrics_frame["mode"].eq("joint") & label_metrics_frame["label"].eq("celltype")
    ]
    best_celltype = joint_celltype.loc[joint_celltype["ari"].idxmax()]
    spatial_summary = spatial_frame.groupby(["mode", "k"])["neighbor_same_cluster_fraction"].mean()
    lines = [
        "# Human embryo RNA-only standardized-embedding analysis",
        "",
        f"- total analyzed spots: {total_spots}",
        f"- sections: {', '.join(sections)}",
        "- clustering: MiniBatchKMeans on standardized final embedding",
        f"- internal metrics: fixed {int(cluster_metrics_frame['metric_n_obs'].max()):,}-spot sample per scope",
        f"- spatial continuity: exact {int(spatial_frame['spatial_neighbor_k'].iloc[0])}-NN within each section",
        "",
        "## Main results",
        "",
        f"- best joint Cluster ASW: k={int(best_internal['k'])}, ASW={best_internal['cluster_asw']:.4f}, "
        f"CH={best_internal['calinski_harabasz']:.2f}, DBI={best_internal['davies_bouldin']:.4f}",
        f"- best joint celltype agreement: k={int(best_celltype['k'])}, "
        f"ARI={best_celltype['ari']:.4f}, NMI={best_celltype['nmi']:.4f}",
        f"- section diagnostics: bASW={batch.get('bASW', float('nan')):.4f}, "
        f"bLISI={batch.get('bLISI', float('nan')):.4f}, kBET={batch.get('kBET', float('nan')):.4f}, "
        f"PCR_score={batch.get('PCR_score', float('nan')):.4f}",
        f"- training loss: {training['first_total_loss']:.4f} -> {training['last_total_loss']:.4f}",
        f"- final OT pairs: {len(ot_frame)}; mean topology-changed fraction="
        f"{ot_frame['topology_topk_changed_fraction'].mean():.4f}",
        "",
        "## Mean spatial neighbor agreement",
        "",
        "| mode | k | agreement |",
        "|---|---:|---:|",
    ]
    for (mode, k), value in spatial_summary.items():
        lines.append(f"| {mode} | {int(k)} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Embryonic section/stage is biological time, not a nuisance batch. bASW/bLISI/kBET/PCR are reported only for comparability with MISAR-seq.",
            "- High spatial agreement can indicate coherent tissue domains, but can also reflect over-smoothing or large clusters.",
            "- Celltype annotations provide biological agreement metrics but were not used as a training target.",
            "- Internal metrics are sampled because the dataset contains over one million spots.",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "analysis" / "standardized_embedding"
    )
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "clustering").mkdir(parents=True, exist_ok=True)
    summary, sections, embeddings, metadata = load_inputs(run_dir)
    total_spots = sum(len(embeddings[section]) for section in sections)
    offsets = {}
    offset = 0
    for section in sections:
        offsets[section] = (offset, offset + len(embeddings[section]))
        offset += len(embeddings[section])
    joint_space, independent_spaces, cache_info = prepare_spaces(
        output_dir, sections, embeddings, args.reuse_cache
    )
    neighbors = build_spatial_neighbors(
        output_dir, sections, metadata, args.spatial_neighbor_k, args.reuse_cache
    )
    all_celltypes = np.concatenate([metadata[s]["celltype"].astype(str).to_numpy() for s in sections])
    all_sections = np.concatenate([np.repeat(s, len(embeddings[s])) for s in sections])
    joint_metric_idx = sample_indices(total_spots, args.metric_sample_size, args.seed)
    section_metric_idx = {
        section: sample_indices(len(embeddings[section]), args.metric_sample_size, args.seed + i + 1)
        for i, section in enumerate(sections)
    }
    joint_label_asw = label_asw(joint_space, all_celltypes, joint_metric_idx)
    section_asw = label_asw(joint_space, all_sections, joint_metric_idx)
    independent_label_asw = {
        section: label_asw(
            independent_spaces[section], metadata[section]["celltype"].astype(str).to_numpy(),
            section_metric_idx[section]
        )
        for section in sections
    }

    cluster_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    retained = set(args.plot_k)
    for k in args.joint_k:
        print(f"[analysis] joint MiniBatchKMeans k={k}", flush=True)
        labels_all = fit_minibatch(joint_space, k, args)
        label_dir = output_dir / "clustering" / f"joint_k{k}"
        files = save_label_arrays(label_dir, sections, labels_all, offsets)
        internal = cluster_metrics(joint_space, labels_all, joint_metric_idx)
        section_external = external_metrics(all_sections, labels_all)
        cluster_rows.append(
            {
                "mode": "joint", "scope": "combined", "k": k,
                "n_obs": total_spots, "embedding_dim": 128, **internal,
                "section_ari": section_external["ari"], "section_nmi": section_external["nmi"],
                "section_asw": section_asw, "metric_space": "standardized_embedding",
                "labels_path": files["all"],
            }
        )
        label_rows.append(
            {
                "mode": "joint", "scope": "combined", "label": "celltype", "k": k,
                **external_metrics(all_celltypes, labels_all),
                "label_asw": joint_label_asw,
                "label_asw_scaled": (joint_label_asw + 1.0) / 2.0,
            }
        )
        labels_by_section = {}
        for section in sections:
            start, end = offsets[section]
            values = labels_all[start:end]
            labels_by_section[section] = values
            spatial_rows.append(
                {
                    "mode": "joint", "k": k, "section": section,
                    "n_obs": len(values), "spatial_neighbor_k": args.spatial_neighbor_k,
                    "neighbor_same_cluster_fraction": spatial_agreement(values, neighbors[section]),
                    "labels_path": files[section],
                }
            )
        counts = []
        for section, values in labels_by_section.items():
            for cluster, count in zip(*np.unique(values, return_counts=True)):
                counts.append(
                    {"section": section, "cluster": int(cluster), "count": int(count), "fraction": count / len(values)}
                )
        pd.DataFrame(counts).to_csv(label_dir / "cluster_counts.csv", index=False)
        if k in retained:
            plot_spatial_panels(
                label_dir / "spatial_all_sections.png", sections, metadata, labels_by_section,
                f"joint standardized embedding, k={k}", args.plot_sample_per_section, args.seed
            )

    for k in args.independent_k:
        print(f"[analysis] independent MiniBatchKMeans k={k}", flush=True)
        label_dir = output_dir / "clustering" / f"independent_k{k}"
        label_dir.mkdir(parents=True, exist_ok=True)
        labels_by_section = {}
        counts = []
        for index, section in enumerate(sections):
            labels = fit_minibatch(independent_spaces[section], k, args)
            labels_by_section[section] = labels
            label_path = label_dir / f"labels_{section}.npy"
            np.save(label_path, labels)
            internal = cluster_metrics(independent_spaces[section], labels, section_metric_idx[section])
            cluster_rows.append(
                {
                    "mode": "independent", "scope": section, "k": k,
                    "n_obs": len(labels), "embedding_dim": 128, **internal,
                    "section_ari": np.nan, "section_nmi": np.nan, "section_asw": np.nan,
                    "metric_space": "standardized_embedding", "labels_path": str(label_path),
                }
            )
            truth = metadata[section]["celltype"].astype(str).to_numpy()
            label_rows.append(
                {
                    "mode": "independent", "scope": section, "label": "celltype", "k": k,
                    **external_metrics(truth, labels),
                    "label_asw": independent_label_asw[section],
                    "label_asw_scaled": (independent_label_asw[section] + 1.0) / 2.0,
                }
            )
            spatial_rows.append(
                {
                    "mode": "independent", "k": k, "section": section,
                    "n_obs": len(labels), "spatial_neighbor_k": args.spatial_neighbor_k,
                    "neighbor_same_cluster_fraction": spatial_agreement(labels, neighbors[section]),
                    "labels_path": str(label_path),
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                counts.append(
                    {"section": section, "cluster": int(cluster), "count": int(count), "fraction": count / len(labels)}
                )
        pd.DataFrame(counts).to_csv(label_dir / "cluster_counts.csv", index=False)
        if k in retained:
            plot_spatial_panels(
                label_dir / "spatial_all_sections.png", sections, metadata, labels_by_section,
                f"independent standardized embedding, k={k}", args.plot_sample_per_section, args.seed
            )

    cluster_frame = pd.DataFrame(cluster_rows)
    label_frame = pd.DataFrame(label_rows)
    spatial_frame = pd.DataFrame(spatial_rows)
    cluster_frame.to_csv(output_dir / "metrics" / "clustering_metrics.csv", index=False)
    label_frame.to_csv(output_dir / "metrics" / "clustering_metrics_by_label.csv", index=False)
    spatial_frame.to_csv(output_dir / "metrics" / "spatial_continuity.csv", index=False)
    spatial_frame.groupby(["mode", "k"], as_index=False)["neighbor_same_cluster_fraction"].mean().to_csv(
        output_dir / "metrics" / "spatial_continuity_summary.csv", index=False
    )
    best_rows = []
    for (mode, scope, label), frame in label_frame.groupby(["mode", "scope", "label"]):
        best_rows.append(frame.loc[frame["ari"].idxmax()].to_dict())
    pd.DataFrame(best_rows).to_csv(output_dir / "metrics" / "best_clustering_metrics_by_label.csv", index=False)

    batch_idx = sample_indices(total_spots, args.batch_metric_sample_size, args.seed + 100)
    print(f"[analysis] section-mixing diagnostics on {len(batch_idx)} spots", flush=True)
    batch_metrics = compute_batch_correction_metrics(
        np.asarray(joint_space[batch_idx]), all_sections[batch_idx],
        dataset="Human embryo HESTA", method="spa_mo_model_result_v4C_RNA_only",
        batch_label_name="developmental_section", max_samples=0,
        asw_sample_size=args.batch_asw_sample_size, lisi_neighbors=90,
        kbet_neighbors=50, seed=args.seed, pcr_components=50,
    )
    batch_metrics["n_obs_total_actual"] = total_spots
    batch_metrics["interpretation"] = "Developmental section is biological time, not a nuisance batch."
    pd.DataFrame([batch_metrics]).to_csv(output_dir / "metrics" / "section_mixing_diagnostics.csv", index=False)
    ot_frame = summarize_ot(run_dir, output_dir)
    training = training_diagnostics(run_dir, output_dir)
    config = {
        "dataset": "Human embryo HESTA RNA-only",
        "method": "spa_mo_model_result_v4C",
        "preprocessing": "standardized_embedding",
        "kmeans_type": "sklearn.cluster.MiniBatchKMeans",
        "joint_metric_k_values": args.joint_k,
        "independent_metric_k_values": args.independent_k,
        "plot_k_values": args.plot_k,
        "metric_sample_size": args.metric_sample_size,
        "batch_metric_sample_size": args.batch_metric_sample_size,
        "spatial_neighbor_k": args.spatial_neighbor_k,
        "spatial_graph_scope": "exact_knn_within_each_section",
        "n_obs": total_spots,
        "section_counts": {section: len(embeddings[section]) for section in sections},
        "embedding_dim": 128,
        "label_keys": ["celltype"],
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "cache": cache_info,
        "dynamic_candidate_source": "fused",
        "dynamic_context_source": "fused_spatial_context",
        "dynamic_ot_cost": "0.8 * semantic cosine cost + 0.2 * local-context cosine cost",
        "attention_context_gate_enabled": True,
        "interpretation": {
            "section_metrics": "diagnostic only because section encodes developmental time",
            "sampled_internal_metrics": "required for million-spot scalability",
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": str(run_dir),
    }
    write_json(output_dir / "config.json", config)
    make_summary(
        output_dir, total_spots, sections, cluster_frame, label_frame,
        spatial_frame, batch_metrics, ot_frame, training
    )
    manifest = {
        "status": "PASS", "output_dir": str(output_dir), "config": str(output_dir / "config.json"),
        "summary": str(output_dir / "SUMMARY.md"),
        "metrics": {
            "clustering": str(output_dir / "metrics" / "clustering_metrics.csv"),
            "labels": str(output_dir / "metrics" / "clustering_metrics_by_label.csv"),
            "spatial": str(output_dir / "metrics" / "spatial_continuity.csv"),
            "section_mixing": str(output_dir / "metrics" / "section_mixing_diagnostics.csv"),
            "ot_pairs": str(output_dir / "metrics" / "ot_pair_metrics.csv"),
            "training": str(output_dir / "metrics" / "training_loss_metrics.csv"),
        },
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print("HUMAN_EMBRYO_REFERENCE_ANALYSIS: PASS", flush=True)


if __name__ == "__main__":
    main()
