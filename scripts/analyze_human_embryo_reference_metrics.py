#!/usr/bin/env python3
"""MISAR-style scalable analysis for full-spot HESTA RNA-only embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
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
    parser.add_argument(
        "--reuse-flat-labels", action="store_true",
        help="Reuse existing flat MiniBatchKMeans label arrays while refreshing plots and metrics.",
    )
    parser.add_argument(
        "--skip-hierarchy", action="store_true",
        help="Skip the nested weighted-Ward hierarchy derived from the largest flat-k partition.",
    )
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
    k: int,
) -> None:
    if k < 2 or k > 40:
        raise ValueError(f"The discrete cluster plot requires 2 <= k <= 40; got {k}.")
    colors = list(plt.get_cmap("tab20").colors)
    if k > len(colors):
        colors.extend(list(plt.get_cmap("tab20b").colors[: k - len(colors)]))
    colors = colors[:k]
    cmap = ListedColormap(colors, name=f"tab20_family_{k}_discrete")
    boundaries = np.arange(-0.5, k + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), squeeze=False)
    for offset, section in enumerate(sections):
        axis = axes.ravel()[offset]
        frame = metadata[section]
        chosen = sample_indices(len(frame), maximum, seed + offset)
        values = np.asarray(labels[section][chosen], dtype=np.int32)
        if values.size and (values.min() < 0 or values.max() >= k):
            raise ValueError(f"{section}: cluster labels fall outside [0, {k - 1}].")
        axis.scatter(
            frame["x"].to_numpy()[chosen], frame["y"].to_numpy()[chosen],
            c=values, s=1, cmap=cmap, norm=norm, linewidths=0, rasterized=True,
        )
        axis.set_title(section)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in axes.ravel()[len(sections):]:
        axis.axis("off")
    fig.suptitle(title)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar, ax=axes.ravel().tolist(), shrink=0.55,
        ticks=np.arange(k), boundaries=boundaries, spacing="uniform", label="cluster",
    )
    colorbar.ax.set_yticklabels([str(value) for value in range(k)])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def categorical_colors(n_categories: int) -> list[tuple[float, float, float, float]]:
    colors: list[tuple[float, float, float, float]] = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(list(plt.get_cmap(name).colors))
    if n_categories > len(colors):
        colors.extend([plt.get_cmap("hsv")(value) for value in np.linspace(0, 1, n_categories - len(colors), endpoint=False)])
    return colors[:n_categories]


def plot_celltype_reference(
    output_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    maximum: int,
    seed: int,
) -> dict[str, Any]:
    reference_dir = output_dir / "celltype_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    categories = sorted(
        {
            value
            for section in sections
            for value in metadata[section]["celltype"].astype(str).tolist()
            if value != "nan"
        }
    )
    if not categories:
        raise ValueError("No valid celltype annotations were found.")
    colors = categorical_colors(len(categories))
    category_index = {name: index for index, name in enumerate(categories)}
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), squeeze=False)
    count_rows: list[dict[str, Any]] = []
    for offset, section in enumerate(sections):
        axis = axes.ravel()[offset]
        frame = metadata[section]
        truth = frame["celltype"].astype(str).to_numpy()
        chosen = sample_indices(len(frame), maximum, seed + offset)
        valid = truth[chosen] != "nan"
        chosen_valid = chosen[valid]
        point_colors = [colors[category_index[value]] for value in truth[chosen_valid]]
        axis.scatter(
            frame["x"].to_numpy()[chosen_valid], frame["y"].to_numpy()[chosen_valid],
            c=point_colors, s=1, linewidths=0, rasterized=True,
        )
        axis.set_title(section)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
        for celltype, count in frame["celltype"].astype(str).value_counts().items():
            if celltype != "nan":
                count_rows.append(
                    {
                        "section": section,
                        "celltype": celltype,
                        "count": int(count),
                        "fraction": float(count / len(frame)),
                        "color": to_hex(colors[category_index[celltype]]),
                    }
                )
    for axis in axes.ravel()[len(sections):]:
        axis.axis("off")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, color=colors[index], label=name)
        for index, name in enumerate(categories)
    ]
    fig.suptitle(f"celltype reference annotations ({len(categories)} classes)")
    fig.legend(
        handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5),
        ncol=2, fontsize=7, title="celltype", markerscale=1.2,
    )
    fig.subplots_adjust(right=0.72)
    figure_path = reference_dir / "spatial_all_sections_by_celltype.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(reference_dir / "celltype_counts_by_section.csv", index=False)
    color_map = pd.DataFrame(
        {"celltype": categories, "color": [to_hex(color) for color in colors]}
    )
    color_map.to_csv(reference_dir / "celltype_color_map.csv", index=False)
    return {
        "n_celltypes": len(categories),
        "figure": str(figure_path),
        "counts": str(reference_dir / "celltype_counts_by_section.csv"),
        "color_map": str(reference_dir / "celltype_color_map.csv"),
    }


def cluster_centers_from_labels(
    space: np.ndarray,
    labels: np.ndarray,
    n_leaves: int,
    chunk_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(n_leaves, dtype=np.int64)
    sums = np.zeros((n_leaves, int(space.shape[1])), dtype=np.float64)
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        chunk_labels = np.asarray(labels[start:end], dtype=np.int32)
        if chunk_labels.size and (chunk_labels.min() < 0 or chunk_labels.max() >= n_leaves):
            raise ValueError("Leaf labels fall outside the expected hierarchy range.")
        chunk = np.asarray(space[start:end], dtype=np.float64)
        for leaf in np.unique(chunk_labels):
            selected = chunk_labels == leaf
            counts[leaf] += int(selected.sum())
            sums[leaf] += chunk[selected].sum(axis=0)
    if np.any(counts == 0):
        raise ValueError(f"Empty leaf clusters cannot form a hierarchy: {np.flatnonzero(counts == 0).tolist()}")
    centers = sums / counts[:, None]
    return centers, counts


def build_weighted_ward_tree(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[int, list[list[int]]]]:
    """Agglomerate flat leaf clusters with a size-weighted Ward SSE cost."""

    n_leaves = int(len(counts))
    nodes: dict[int, dict[str, Any]] = {
        leaf: {
            "center": np.asarray(centers[leaf], dtype=np.float64),
            "count": int(counts[leaf]),
            "leaves": [leaf],
        }
        for leaf in range(n_leaves)
    }
    active = set(range(n_leaves))
    cuts: dict[int, list[list[int]]] = {
        n_leaves: [[leaf] for leaf in range(n_leaves)]
    }
    linkage_rows: list[list[float]] = []
    merge_rows: list[dict[str, Any]] = []
    previous_height = 0.0
    for step in range(n_leaves - 1):
        active_sorted = sorted(active)
        best: Optional[tuple[float, int, int]] = None
        for left_offset, left in enumerate(active_sorted[:-1]):
            left_node = nodes[left]
            for right in active_sorted[left_offset + 1 :]:
                right_node = nodes[right]
                squared_distance = float(np.sum((left_node["center"] - right_node["center"]) ** 2))
                ward_delta = (
                    left_node["count"] * right_node["count"]
                    / (left_node["count"] + right_node["count"])
                    * squared_distance
                )
                candidate = (ward_delta, left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("Weighted Ward hierarchy could not select a merge.")
        ward_delta, left, right = best
        new_node = n_leaves + step
        left_node = nodes[left]
        right_node = nodes[right]
        merged_count = int(left_node["count"] + right_node["count"])
        merged_center = (
            left_node["center"] * left_node["count"]
            + right_node["center"] * right_node["count"]
        ) / merged_count
        merged_leaves = sorted(left_node["leaves"] + right_node["leaves"])
        height = max(previous_height, math.sqrt(max(float(ward_delta), 0.0)))
        previous_height = height
        nodes[new_node] = {
            "center": merged_center,
            "count": merged_count,
            "leaves": merged_leaves,
        }
        active.remove(left)
        active.remove(right)
        active.add(new_node)
        linkage_rows.append([float(left), float(right), height, float(merged_count)])
        merge_rows.append(
            {
                "merge_step": step + 1,
                "clusters_after_merge": n_leaves - step - 1,
                "left_node": left,
                "right_node": right,
                "parent_node": new_node,
                "left_count": int(left_node["count"]),
                "right_count": int(right_node["count"]),
                "parent_count": merged_count,
                "ward_sse_increase": float(ward_delta),
                "dendrogram_height": height,
                "left_leaf_clusters": ",".join(map(str, left_node["leaves"])),
                "right_leaf_clusters": ",".join(map(str, right_node["leaves"])),
                "parent_leaf_clusters": ",".join(map(str, merged_leaves)),
            }
        )
        cuts[len(active)] = [nodes[node]["leaves"].copy() for node in sorted(active)]
    return np.asarray(linkage_rows, dtype=np.float64), pd.DataFrame(merge_rows), cuts


def labels_from_hierarchy_cut(
    leaf_labels: np.ndarray,
    groups: list[list[int]],
    n_leaves: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    ordered = sorted((sorted(group) for group in groups), key=lambda group: min(group))
    mapping = np.full(n_leaves, -1, dtype=np.int32)
    rows = []
    for cluster, leaves in enumerate(ordered):
        mapping[np.asarray(leaves, dtype=np.int32)] = cluster
        rows.append(
            {
                "hierarchical_cluster": cluster,
                "leaf_k": n_leaves,
                "leaf_clusters": ",".join(map(str, leaves)),
                "n_leaf_clusters": len(leaves),
            }
        )
    if np.any(mapping < 0):
        raise RuntimeError("Hierarchy cut does not cover every leaf cluster.")
    return mapping[np.asarray(leaf_labels, dtype=np.int32)], pd.DataFrame(rows)


def plot_hierarchy_tree(path: Path, linkage: np.ndarray, title: str, n_leaves: int) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage, labels=[f"leaf {leaf}" for leaf in range(n_leaves)],
        leaf_rotation=45, leaf_font_size=8, color_threshold=0, above_threshold_color="#333333",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(f"flat k={n_leaves} leaf cluster")
    axis.set_ylabel("sqrt(weighted Ward SSE increase)")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_nested_labels(
    labels_by_k: dict[int, np.ndarray],
    mode: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = []
    ordered = sorted(labels_by_k)
    for low_k, high_k in zip(ordered[:-1], ordered[1:]):
        low = labels_by_k[low_k]
        high = labels_by_k[high_k]
        violations = 0
        for high_cluster in np.unique(high):
            if len(np.unique(low[high == high_cluster])) != 1:
                violations += 1
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "lower_k": low_k,
                "higher_k": high_k,
                "higher_clusters_checked": int(len(np.unique(high))),
                "nesting_violations": violations,
                "is_strict_refinement": violations == 0,
            }
        )
    return rows


def run_hierarchical_analysis(
    output_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    joint_space: np.ndarray,
    independent_spaces: dict[str, np.ndarray],
    offsets: dict[str, tuple[int, int]],
    neighbors: dict[str, np.ndarray],
    all_celltypes: np.ndarray,
    joint_metric_idx: np.ndarray,
    section_metric_idx: dict[str, np.ndarray],
    joint_label_asw: float,
    independent_label_asw: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hierarchy_k = sorted(set(args.plot_k))
    leaf_k = max(hierarchy_k)
    if leaf_k not in args.joint_k or leaf_k not in args.independent_k:
        raise ValueError("The largest hierarchy k must exist in both flat joint and independent results.")
    hierarchy_dir = output_dir / "hierarchical_clustering"
    tree_dir = hierarchy_dir / "trees"
    metrics_dir = hierarchy_dir / "metrics"
    tree_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    print(f"[analysis] weighted-Ward joint hierarchy from flat k={leaf_k}", flush=True)
    joint_leaf_labels = np.load(
        output_dir / "clustering" / f"joint_k{leaf_k}" / "labels_all.npy", mmap_mode="r"
    )
    centers, counts = cluster_centers_from_labels(joint_space, joint_leaf_labels, leaf_k)
    linkage, merges, cuts = build_weighted_ward_tree(centers, counts)
    np.save(tree_dir / "joint_linkage.npy", linkage)
    merges.to_csv(tree_dir / "joint_tree_merges.csv", index=False)
    plot_hierarchy_tree(
        tree_dir / "joint_dendrogram.png", linkage,
        f"joint weighted-Ward hierarchy from flat k={leaf_k}", leaf_k,
    )
    joint_hierarchical: dict[int, np.ndarray] = {}
    for k in hierarchy_k:
        labels_all, mapping = labels_from_hierarchy_cut(joint_leaf_labels, cuts[k], leaf_k)
        joint_hierarchical[k] = labels_all
        label_dir = hierarchy_dir / f"joint_k{k}"
        files = save_label_arrays(label_dir, sections, labels_all, offsets)
        mapping.to_csv(label_dir / f"leaf_k{leaf_k}_to_cluster_k{k}.csv", index=False)
        internal = cluster_metrics(joint_space, labels_all, joint_metric_idx)
        cluster_rows.append(
            {
                "mode": "joint", "scope": "combined", "k": k,
                "n_obs": len(labels_all), "embedding_dim": 128, **internal,
                "metric_space": "standardized_embedding", "labels_path": files["all"],
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
        count_rows = []
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
            for cluster, count in zip(*np.unique(values, return_counts=True)):
                count_rows.append(
                    {"section": section, "cluster": int(cluster), "count": int(count), "fraction": count / len(values)}
                )
        pd.DataFrame(count_rows).to_csv(label_dir / "cluster_counts.csv", index=False)
        plot_spatial_panels(
            label_dir / "spatial_all_sections.png", sections, metadata, labels_by_section,
            f"nested joint weighted-Ward cut, k={k}", args.plot_sample_per_section, args.seed, k,
        )
    validation_rows.extend(validate_nested_labels(joint_hierarchical, "joint", "combined"))

    independent_hierarchical: dict[str, dict[int, np.ndarray]] = {}
    cut_mappings: dict[str, dict[int, pd.DataFrame]] = {}
    for section in sections:
        print(f"[analysis] weighted-Ward independent hierarchy: {section}", flush=True)
        leaf_labels = np.load(
            output_dir / "clustering" / f"independent_k{leaf_k}" / f"labels_{section}.npy",
            mmap_mode="r",
        )
        centers, counts = cluster_centers_from_labels(independent_spaces[section], leaf_labels, leaf_k)
        linkage, merges, cuts = build_weighted_ward_tree(centers, counts)
        np.save(tree_dir / f"independent_{section}_linkage.npy", linkage)
        merges.to_csv(tree_dir / f"independent_{section}_tree_merges.csv", index=False)
        plot_hierarchy_tree(
            tree_dir / f"independent_{section}_dendrogram.png", linkage,
            f"{section} weighted-Ward hierarchy from flat k={leaf_k}", leaf_k,
        )
        independent_hierarchical[section] = {}
        cut_mappings[section] = {}
        for k in hierarchy_k:
            labels, mapping = labels_from_hierarchy_cut(leaf_labels, cuts[k], leaf_k)
            independent_hierarchical[section][k] = labels
            cut_mappings[section][k] = mapping
        validation_rows.extend(
            validate_nested_labels(independent_hierarchical[section], "independent", section)
        )

    for k in hierarchy_k:
        label_dir = hierarchy_dir / f"independent_k{k}"
        label_dir.mkdir(parents=True, exist_ok=True)
        labels_by_section = {}
        count_rows = []
        for section in sections:
            labels = independent_hierarchical[section][k]
            labels_by_section[section] = labels
            label_path = label_dir / f"labels_{section}.npy"
            np.save(label_path, labels)
            mapping = cut_mappings[section][k].copy()
            mapping.insert(0, "section", section)
            mapping.to_csv(label_dir / f"leaf_k{leaf_k}_to_cluster_k{k}_{section}.csv", index=False)
            internal = cluster_metrics(independent_spaces[section], labels, section_metric_idx[section])
            cluster_rows.append(
                {
                    "mode": "independent", "scope": section, "k": k,
                    "n_obs": len(labels), "embedding_dim": 128, **internal,
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
                count_rows.append(
                    {"section": section, "cluster": int(cluster), "count": int(count), "fraction": count / len(labels)}
                )
        pd.DataFrame(count_rows).to_csv(label_dir / "cluster_counts.csv", index=False)
        plot_spatial_panels(
            label_dir / "spatial_all_sections.png", sections, metadata, labels_by_section,
            f"nested independent weighted-Ward cut, k={k}", args.plot_sample_per_section, args.seed, k,
        )

    cluster_frame = pd.DataFrame(cluster_rows)
    label_frame = pd.DataFrame(label_rows)
    spatial_frame = pd.DataFrame(spatial_rows)
    validation_frame = pd.DataFrame(validation_rows)
    if not bool(validation_frame["is_strict_refinement"].all()):
        raise RuntimeError("The generated hierarchy failed its nested-refinement validation.")
    cluster_frame.to_csv(metrics_dir / "clustering_metrics.csv", index=False)
    label_frame.to_csv(metrics_dir / "clustering_metrics_by_label.csv", index=False)
    spatial_frame.to_csv(metrics_dir / "spatial_continuity.csv", index=False)
    spatial_frame.groupby(["mode", "k"], as_index=False)["neighbor_same_cluster_fraction"].mean().to_csv(
        metrics_dir / "spatial_continuity_summary.csv", index=False
    )
    validation_frame.to_csv(metrics_dir / "hierarchy_validation.csv", index=False)
    hierarchy_config = {
        "method": "size-weighted Ward agglomeration of flat MiniBatchKMeans leaf centroids",
        "leaf_k": leaf_k,
        "cut_k_values": hierarchy_k,
        "nested_refinement_validated": True,
        "joint_dendrogram": str(tree_dir / "joint_dendrogram.png"),
        "tree_directory": str(tree_dir),
        "metrics_directory": str(metrics_dir),
    }
    write_json(hierarchy_dir / "hierarchy_config.json", hierarchy_config)
    return hierarchy_config


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
    celltype_reference: dict[str, Any],
    hierarchy: Optional[dict[str, Any]],
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
        f"- celltype reference plot: {celltype_reference['n_celltypes']} annotated classes",
        "- batch correction applied before training: no (Harmony=false)",
        "- balanced feature selection: section-balanced HVG/SVD fitting; this is not batch correction",
        f"- nested hierarchy: {'enabled and validated' if hierarchy is not None else 'skipped'}",
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
    print("[analysis] plotting celltype reference annotations", flush=True)
    celltype_reference = plot_celltype_reference(
        output_dir, sections, metadata, args.plot_sample_per_section, args.seed
    )
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
        label_dir = output_dir / "clustering" / f"joint_k{k}"
        existing_labels = label_dir / "labels_all.npy"
        if args.reuse_flat_labels and existing_labels.exists():
            print(f"[analysis] reusing joint labels k={k}", flush=True)
            # Copy before save_label_arrays rewrites the same labels_all.npy path.
            labels_all = np.asarray(np.load(existing_labels, mmap_mode="r"), dtype=np.int32).copy()
            if labels_all.shape != (total_spots,) or len(np.unique(labels_all)) != k:
                raise ValueError(f"Existing joint k={k} labels failed shape/cluster validation.")
        else:
            print(f"[analysis] joint MiniBatchKMeans k={k}", flush=True)
            labels_all = fit_minibatch(joint_space, k, args)
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
                f"joint standardized embedding, k={k}", args.plot_sample_per_section, args.seed, k
            )

    for k in args.independent_k:
        label_dir = output_dir / "clustering" / f"independent_k{k}"
        label_dir.mkdir(parents=True, exist_ok=True)
        labels_by_section = {}
        counts = []
        for index, section in enumerate(sections):
            label_path = label_dir / f"labels_{section}.npy"
            if args.reuse_flat_labels and label_path.exists():
                print(f"[analysis] reusing independent labels {section} k={k}", flush=True)
                labels = np.load(label_path, mmap_mode="r")
                if labels.shape != (len(embeddings[section]),) or len(np.unique(labels)) != k:
                    raise ValueError(f"Existing independent {section} k={k} labels failed validation.")
            else:
                print(f"[analysis] independent MiniBatchKMeans {section} k={k}", flush=True)
                labels = fit_minibatch(independent_spaces[section], k, args)
                np.save(label_path, labels)
            labels_by_section[section] = labels
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
                f"independent standardized embedding, k={k}", args.plot_sample_per_section, args.seed, k
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

    hierarchy_info = None
    if not args.skip_hierarchy:
        hierarchy_info = run_hierarchical_analysis(
            output_dir, sections, metadata, joint_space, independent_spaces, offsets,
            neighbors, all_celltypes, joint_metric_idx, section_metric_idx,
            joint_label_asw, independent_label_asw, args,
        )

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
        "flat_labels_reused": bool(args.reuse_flat_labels),
        "dynamic_candidate_source": "fused",
        "dynamic_context_source": "fused_spatial_context",
        "dynamic_ot_cost": "0.8 * semantic cosine cost + 0.2 * local-context cosine cost",
        "attention_context_gate_enabled": True,
        "batch_correction": {
            "applied": False,
            "harmony_used": False,
            "section_balanced_hvg_svd": True,
            "section_balanced_feature_selection_is_batch_correction": False,
            "note": "Developmental section is biological time; section diagnostics are evaluation-only.",
        },
        "cluster_plot_palette": {
            "type": "discrete",
            "base": "tab20",
            "number_of_colors_equals_k": True,
        },
        "celltype_reference": celltype_reference,
        "hierarchy": hierarchy_info,
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
        spatial_frame, batch_metrics, ot_frame, training, celltype_reference, hierarchy_info
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
            "celltype_reference": celltype_reference,
            "hierarchical_clustering": hierarchy_info,
        },
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print("HUMAN_EMBRYO_REFERENCE_ANALYSIS: PASS", flush=True)


if __name__ == "__main__":
    main()
