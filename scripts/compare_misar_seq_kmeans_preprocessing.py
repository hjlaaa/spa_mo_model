#!/usr/bin/env python3
"""Create independent raw-vs-standardized KMeans results for MISAR-seq.

Original training/analysis outputs are read-only.  Each method is aligned with
its own MISAR-seq data copy and written to a new comparison root.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from compare_mousebrain_kmeans_preprocessing import (
    external,
    f4,
    fitted_space,
    internal,
    md_table,
    safe_asw,
    scaler_payload,
    sha256,
    spatial_agreement,
    valid_truth,
)


SECTIONS = ["dataset1", "dataset2", "dataset3", "dataset4"]
SECTION_COUNTS = {
    "dataset1": 2129,
    "dataset2": 1949,
    "dataset3": 1777,
    "dataset4": 1263,
}
LABEL_KEYS = [
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
]
METRIC_KS = list(range(2, 17))
PLOT_KS = [5, 8, 10, 12, 14, 16]
PRUNED_CLUSTERING_KS = [
    k for k in METRIC_KS if k not in PLOT_KS
]
PLOT_POINT_SIZE = 18.0
PLOT_DPI = 220
SEED = 0
N_INIT = 20
MAX_ITER = 300
SPATIAL_NEIGHBOR_K = 6
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/"
    "misar_seq_preprocessing_comparison_report.md"
)
SCRIPT_PATH = Path(__file__).resolve()


@dataclass
class MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    truth: dict[str, np.ndarray]
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]


def clean_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
        else:
            text = str(value)
            result[i] = (
                text[2:-1]
                if text.startswith("b'") and text.endswith("'")
                else text
            )
    return result


def aligned_metadata(
    data_dir: Path, sections: np.ndarray, barcodes: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coords = np.empty((len(sections), 2), dtype=float)
    truth = {
        label: np.full(len(sections), np.nan, dtype=object)
        for label in LABEL_KEYS
    }
    for section in SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            positions = source.get_indexer(wanted)
            if (positions < 0).any():
                examples = wanted[positions < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: {int((positions < 0).sum())} barcodes "
                    f"failed alignment; examples={examples}"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[positions, :2]
            for label in LABEL_KEYS:
                if label not in rna.obs:
                    raise ValueError(f"{rna_path}: missing label {label}")
                truth[label][mask] = clean_truth(
                    rna.obs[label].to_numpy()[positions]
                )
        finally:
            rna.file.close()
    return coords, truth


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/misar_seq/"
        "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/MISAR-seq")
    arrays, sections, barcodes, sources = [], [], [], []
    for section in SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        metadata = pd.read_csv(metadata_path)
        embedding = np.load(embedding_path)
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            own_names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        saved_names = metadata["obs_name"].astype(str).to_numpy()
        if (
            len(embedding) != len(saved_names)
            or not np.array_equal(saved_names, own_names)
        ):
            raise ValueError(
                f"spa_mo_model {section}: embedding/metadata/data mismatch"
            )
        arrays.append(embedding)
        sections.extend([section] * len(saved_names))
        barcodes.extend(saved_names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections)
    barcode_array = np.asarray(barcodes)
    coords, truth = aligned_metadata(data_dir, section_array, barcode_array)
    old = run / "clustering_analysis_k8_10_12_14_16"
    return MethodData(
        "spa_mo_model",
        np.vstack(arrays),
        section_array,
        barcode_array,
        coords,
        truth,
        data_dir,
        sources,
        Path(
            "/home/hujinlan/spa_mo_model/results/"
            "misar_seq_preprocessing_comparison"
        ),
        {"batch_correction_metrics.csv": old / "batch_correction_metrics.csv"},
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000"
    )
    table_path = old / "tables" / "factors_with_metadata_and_coordinates.csv"
    table = pd.read_csv(table_path)
    factor_columns = [
        column for column in table if column.startswith("Factor")
    ]
    sections = table["section"].astype(str).to_numpy()
    barcodes = table["original_barcode"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/MISAR-seq")
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    return MethodData(
        "MOFA+",
        table[factor_columns].to_numpy(float),
        sections,
        barcodes,
        coords,
        truth,
        data_dir,
        [table_path],
        Path(
            "/home/hujinlan/mofa+/analysis/"
            "misar_seq_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "r2_total_by_view_group.csv": old
            / "tables"
            / "r2_total_by_view_group.csv",
        },
    )


def load_cosie() -> MethodData:
    run = Path("/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full")
    old = run / "analysis"
    table_path = old / "tables" / "embeddings_with_metadata.csv"
    table = pd.read_csv(table_path)
    embedding_columns = [
        column for column in table if column.startswith("COSIE")
    ]
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [
        run / "final_embeddings" / f"{section}_final_embedding.npy"
        for section in SECTIONS
    ]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    if (
        final_embedding.shape != table_embedding.shape
        or not np.allclose(final_embedding, table_embedding)
    ):
        raise ValueError("COSIE: analysis table is not the final embedding")
    sections = table["section"].astype(str).to_numpy()
    barcode_column = (
        "original_barcode"
        if "original_barcode" in table
        else "obs_name"
    )
    barcodes = table[barcode_column].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/cosie/data/MISAR-seq")
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    return MethodData(
        "COSIE",
        table_embedding,
        sections,
        barcodes,
        coords,
        truth,
        data_dir,
        [table_path, *final_paths],
        Path(
            "/home/hujinlan/cosie_runs/"
            "misar_seq_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv"
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/misar_seq_spamosaic_full/"
        "misar_seq_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs["section"].astype(str).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path(
        "/home/hujinlan/SpaMosaic-dev/demo/data/MISAR-seq"
    )
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    old = Path(
        "/home/hujinlan/SpaMosaic-dev/analysis/"
        "misar_seq_spamosaic_full"
    )
    return MethodData(
        "SpaMosaic",
        embedding,
        sections,
        barcodes,
        coords,
        truth,
        data_dir,
        [h5ad_path],
        Path(
            "/home/hujinlan/SpaMosaic-dev/analysis/"
            "misar_seq_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "modality_alignment.csv": old
            / "metrics"
            / "modality_alignment.csv",
        },
    )


def validate(
    data: MethodData,
    allow_existing_output: bool = False,
) -> None:
    n_obs = len(data.embedding)
    if (
        n_obs != 7118
        or len(data.sections) != n_obs
        or len(data.barcodes) != n_obs
    ):
        raise ValueError(f"{data.name}: expected/aligned 7118 rows, got {n_obs}")
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f"{data.name}: non-finite embedding or coordinates")
    counts = {
        key: int(value)
        for key, value in pd.Series(data.sections).value_counts().items()
    }
    if counts != SECTION_COUNTS:
        raise ValueError(f"{data.name}: unexpected section counts {counts}")
    if len(set(zip(data.sections, data.barcodes))) != n_obs:
        raise ValueError(f"{data.name}: duplicate section/barcode keys")
    for label in LABEL_KEYS:
        valid = valid_truth(data.truth[label])
        if valid.sum() != 7118 or len(np.unique(data.truth[label][valid])) < 2:
            raise ValueError(
                f"{data.name}: label {label} is incomplete or invalid"
            )
    if data.output_root.exists() and not allow_existing_output:
        raise FileExistsError(f"refusing to overwrite {data.output_root}")
    if allow_existing_output and not data.output_root.is_dir():
        raise FileNotFoundError(
            f"{data.name}: missing existing comparison root "
            f"{data.output_root}"
        )
    missing = [
        str(path)
        for path in data.source_paths + list(data.shared_sources.values())
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{data.name}: missing sources {missing}")


def save_labels(
    path: Path, data: MethodData, mask: np.ndarray, labels: np.ndarray
) -> None:
    payload: dict[str, Any] = {
        "section": data.sections[mask],
        "obs_name": data.barcodes[mask],
        "cluster": labels.astype(int),
        "x": data.coords[mask, 0],
        "y": data.coords[mask, 1],
    }
    for label in LABEL_KEYS:
        payload[label] = data.truth[label][mask]
    pd.DataFrame(payload).to_csv(path, index=False)


def plot_spatial(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    categories, codes = np.unique(labels, return_inverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=PLOT_POINT_SIZE,
        cmap="tab20",
        linewidths=0,
        alpha=0.95,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    handles, _ = scatter.legend_elements(num=len(categories))
    ax.legend(
        handles,
        [str(item) for item in categories],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def label_metrics(
    data: MethodData,
    space: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    mode: str,
    scope: str,
    k: int,
    asw_cache: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    rows = []
    for label_key in LABEL_KEYS:
        values = data.truth[label_key][mask]
        valid = valid_truth(values)
        cache_key = (scope, label_key)
        if cache_key not in asw_cache:
            asw_cache[cache_key] = safe_asw(space[valid], values[valid])
        label_asw = asw_cache[cache_key]
        truth = values[valid].astype(str)
        prediction = labels[valid]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_key,
                "n_obs_labeled": int(valid.sum()),
                "n_label_classes": len(np.unique(truth)),
                "k": k,
                **external(truth, prediction),
                "label_asw": label_asw,
                "label_asw_scaled": (label_asw + 1.0) / 2.0,
            }
        )
    return rows


def run_scheme(data: MethodData, scheme: str) -> None:
    output = data.output_root / scheme
    clustering_root = output / "clustering"
    metrics_root = output / "metrics"
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    label_asw_cache: dict[tuple[str, str], float] = {}

    joint_space, joint_scaler = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, "joint"))
    section_asw = safe_asw(joint_space, data.sections)
    all_mask = np.ones(len(data.embedding), dtype=bool)

    for k in METRIC_KS:
        labels_all = KMeans(
            k,
            random_state=SEED,
            n_init=N_INIT,
            max_iter=MAX_ITER,
        ).fit_predict(joint_space)
        k_directory = clustering_root / f"joint_k{k}"
        k_directory.mkdir()
        all_labels_path = k_directory / "labels_all.csv"
        save_labels(all_labels_path, data, all_mask, labels_all)
        internal_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": len(labels_all),
                "embedding_dim": data.embedding.shape[1],
                **internal(joint_space, labels_all),
                "section_ari": adjusted_rand_score(
                    data.sections, labels_all
                ),
                "section_nmi": normalized_mutual_info_score(
                    data.sections, labels_all
                ),
                "section_asw": section_asw,
                "metric_space": scheme,
                "labels_path": str(all_labels_path),
            }
        )
        external_rows.extend(
            label_metrics(
                data,
                joint_space,
                labels_all,
                all_mask,
                "joint",
                "combined",
                k,
                label_asw_cache,
            )
        )
        count_rows = []
        for section in SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            labels_path = k_directory / f"labels_{section}.csv"
            save_labels(labels_path, data, mask, labels)
            if k in PLOT_KS:
                plot_spatial(
                    k_directory / f"spatial_{section}.png",
                    data.coords[mask],
                    labels,
                    f"{data.name} {scheme} joint K={k} {section}",
                )
            spatial_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "n_obs": int(mask.sum()),
                    "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction": spatial_agreement(
                        data.coords[mask], labels
                    ),
                    "labels_path": str(labels_path),
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(
            k_directory / "cluster_counts.csv", index=False
        )

    for k in METRIC_KS:
        k_directory = clustering_root / f"independent_k{k}"
        k_directory.mkdir()
        count_rows = []
        for section in SECTIONS:
            mask = data.sections == section
            section_space, scaler = fitted_space(
                data.embedding[mask], scheme
            )
            if scaler is not None:
                scaler_arrays.update(scaler_payload(scaler, section))
            labels = KMeans(
                k,
                random_state=SEED,
                n_init=N_INIT,
                max_iter=MAX_ITER,
            ).fit_predict(section_space)
            labels_path = k_directory / f"labels_{section}.csv"
            save_labels(labels_path, data, mask, labels)
            if k in PLOT_KS:
                plot_spatial(
                    k_directory / f"spatial_{section}.png",
                    data.coords[mask],
                    labels,
                    f"{data.name} {scheme} independent K={k} {section}",
                )
            internal_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **internal(section_space, labels),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "metric_space": scheme,
                    "labels_path": str(labels_path),
                }
            )
            external_rows.extend(
                label_metrics(
                    data,
                    section_space,
                    labels,
                    mask,
                    "independent",
                    section,
                    k,
                    label_asw_cache,
                )
            )
            spatial_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "n_obs": int(mask.sum()),
                    "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction": spatial_agreement(
                        data.coords[mask], labels
                    ),
                    "labels_path": str(labels_path),
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(
            k_directory / "cluster_counts.csv", index=False
        )

    metrics = pd.DataFrame(internal_rows)
    external_frame = pd.DataFrame(external_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / "clustering_metrics.csv", index=False)
    external_frame.to_csv(
        metrics_root / "clustering_metrics_by_label.csv", index=False
    )
    joint_external = external_frame[external_frame["mode"] == "joint"]
    best = joint_external.loc[
        joint_external.groupby("label")["ari"].idxmax()
    ].sort_values("label")
    best.to_csv(
        metrics_root / "best_clustering_metrics_by_label.csv", index=False
    )
    spatial.to_csv(metrics_root / "spatial_continuity.csv", index=False)
    (
        spatial.groupby(["mode", "k"], as_index=False)[
            "neighbor_same_cluster_fraction"
        ]
        .mean()
        .rename(
            columns={
                "neighbor_same_cluster_fraction":
                "mean_spatial_neighbor_agreement"
            }
        )
        .to_csv(metrics_root / "spatial_continuity_summary.csv", index=False)
    )
    if scaler_arrays:
        np.savez_compressed(
            output / "scaler_parameters.npz", **scaler_arrays
        )

    config = {
        "dataset": "MISAR-seq",
        "method": data.name,
        "preprocessing": scheme,
        "kmeans_input": (
            "raw_final_embedding"
            if scheme == "raw_embedding"
            else "standardized_final_embedding"
        ),
        "metric_space": (
            "raw_final_embedding"
            if scheme == "raw_embedding"
            else "standardized_final_embedding"
        ),
        "standardization": (
            "none"
            if scheme == "raw_embedding"
            else "sklearn.preprocessing.StandardScaler"
        ),
        "joint_scaler_scope": (
            "none" if scheme == "raw_embedding" else "all_7118_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "kmeans_type": "sklearn.cluster.KMeans",
        "joint_metric_k_values": METRIC_KS,
        "independent_metric_k_values": METRIC_KS,
        "spatial_metric_k_values": METRIC_KS,
        "plot_k_values": PLOT_KS,
        "plot_point_size": PLOT_POINT_SIZE,
        "plot_dpi": PLOT_DPI,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "n_obs": len(data.embedding),
        "section_counts": SECTION_COUNTS,
        "embedding_dim": data.embedding.shape[1],
        "cluster_asw_sample_size": 0,
        "cluster_asw_rule": "full_n_obs_in_each_scope",
        "ch_dbi_sample_size": 0,
        "ch_dbi_rule": "full_n_obs_in_each_scope",
        "label_keys": LABEL_KEYS,
        "label_asw_sample_size": 0,
        "label_asw_rule": "full_nonmissing_labels_in_each_scope",
        "label_valid_counts": {
            label: int(valid_truth(data.truth[label]).sum())
            for label in LABEL_KEYS
        },
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "exact_knn_within_each_section",
        "spatial_graph_n_obs": SECTION_COUNTS,
        "joint_and_independent_labels_cover_all_spots": True,
        "external_internal_spatial_and_figures_reuse_saved_labels": True,
        "source_data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in data.source_paths
        ],
        "generation_script": str(SCRIPT_PATH),
        "generation_script_sha256": sha256(SCRIPT_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "SUMMARY.md").write_text(
        f"# {data.name}: {scheme}\n\n"
        f"- spots: 7118; dimensions: {data.embedding.shape[1]}\n"
        f"- exact KMeans: seed={SEED}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}\n"
        f"- joint/independent metrics and labels: K={METRIC_KS}\n"
        f"- spatial figures: K={PLOT_KS}; point size={PLOT_POINT_SIZE:g}\n"
        "- Cluster ASW, Label ASW, CH and DBI: full valid samples\n"
        f"- spatial continuity: exact {SPATIAL_NEIGHBOR_K}-NN per section\n"
        "- every metric and figure reuses the saved full label CSV files\n"
    )


def redraw_selected_spatial_figures(data: MethodData) -> int:
    """Redraw selected K values from saved labels without rerunning KMeans."""
    removed = 0
    for scheme in ["raw_embedding", "standardized_embedding"]:
        output = data.output_root / scheme
        clustering_root = output / "clustering"
        metrics_root = output / "metrics"

        for figure_path in clustering_root.glob(
            "*_k*/spatial_*.png"
        ):
            k = int(figure_path.parent.name.rsplit("_k", 1)[1])
            if k not in PLOT_KS:
                figure_path.unlink()
                removed += 1

        figure_rows: list[dict[str, Any]] = []
        for mode in ["joint", "independent"]:
            for k in PLOT_KS:
                directory = clustering_root / f"{mode}_k{k}"
                for section in SECTIONS:
                    labels_path = directory / f"labels_{section}.csv"
                    labels = pd.read_csv(
                        labels_path,
                        dtype={"section": str, "obs_name": str},
                    )
                    expected_mask = data.sections == section
                    if (
                        len(labels) != SECTION_COUNTS[section]
                        or not labels["section"].eq(section).all()
                        or not np.array_equal(
                            labels["obs_name"].to_numpy(),
                            data.barcodes[expected_mask],
                        )
                        or not np.allclose(
                            labels[["x", "y"]].to_numpy(float),
                            data.coords[expected_mask],
                        )
                    ):
                        raise ValueError(
                            f"{data.name} {scheme} {mode} K={k} "
                            f"{section}: labels/coordinates mismatch"
                        )
                    figure_path = directory / f"spatial_{section}.png"
                    print(
                        f"[{data.name}] redraw {scheme} {mode} K={k} "
                        f"{section}: n={len(labels)}, "
                        f"point_size={PLOT_POINT_SIZE:g}",
                        flush=True,
                    )
                    plot_spatial(
                        figure_path,
                        labels[["x", "y"]].to_numpy(float),
                        labels["cluster"].to_numpy(int),
                        f"{data.name} {scheme} {mode} K={k} {section}",
                    )
                    figure_rows.append(
                        {
                            "mode": mode,
                            "k": k,
                            "section": section,
                            "plot_n_obs": len(labels),
                            "point_size": PLOT_POINT_SIZE,
                            "dpi": PLOT_DPI,
                            "figure_path": str(figure_path),
                            "figure_sha256": sha256(figure_path),
                            "labels_path": str(labels_path),
                            "labels_sha256": sha256(labels_path),
                        }
                    )

        figures_path = metrics_root / "figure_manifest.csv"
        pd.DataFrame(figure_rows).to_csv(figures_path, index=False)

        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "plot_k_values": PLOT_KS,
                "plot_point_size": PLOT_POINT_SIZE,
                "plot_dpi": PLOT_DPI,
                "generation_script": str(SCRIPT_PATH),
                "generation_script_sha256": sha256(SCRIPT_PATH),
                "figures_redrawn_at": (
                    datetime.now().astimezone().isoformat()
                ),
            }
        )
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )

        summary_path = output / "SUMMARY.md"
        summary_lines = summary_path.read_text().splitlines()
        summary_lines = [
            (
                f"- spatial figures: K={PLOT_KS}; "
                f"point size={PLOT_POINT_SIZE:g}"
                if line.startswith("- spatial figures:")
                else line
            )
            for line in summary_lines
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n")
    return removed


def prune_unplotted_clustering_directories(data: MethodData) -> int:
    """Delete only non-retained joint/independent clustering directories."""
    deleted = 0
    for scheme in ["raw_embedding", "standardized_embedding"]:
        output = data.output_root / scheme
        clustering_root = (output / "clustering").resolve()
        targets: list[Path] = []
        for directory in clustering_root.iterdir():
            if not directory.is_dir() or "_k" not in directory.name:
                continue
            mode, k_text = directory.name.rsplit("_k", 1)
            if (
                mode not in {"joint", "independent"}
                or not k_text.isdigit()
                or int(k_text) in PLOT_KS
            ):
                continue
            if int(k_text) not in PRUNED_CLUSTERING_KS:
                raise ValueError(
                    f"unexpected clustering K target: {directory}"
                )
            if directory.resolve().parent != clustering_root:
                raise ValueError(
                    f"unsafe clustering target resolution: {directory}"
                )
            targets.append(directory)
        for directory in sorted(targets):
            print(f"[{data.name}] delete {directory}", flush=True)
            shutil.rmtree(directory)
            deleted += 1

        remaining = {
            int(directory.name.rsplit("_k", 1)[1])
            for directory in clustering_root.iterdir()
            if directory.is_dir()
            and directory.name.rsplit("_k", 1)[0]
            in {"joint", "independent"}
            and directory.name.rsplit("_k", 1)[1].isdigit()
        }
        if remaining != set(PLOT_KS):
            raise ValueError(
                f"{data.name} {scheme}: unexpected remaining K {remaining}"
            )

        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "retained_clustering_k_values": PLOT_KS,
                "deleted_clustering_k_values": PRUNED_CLUSTERING_KS,
                "metrics_k_values_without_saved_label_directories":
                PRUNED_CLUSTERING_KS,
                "external_internal_spatial_and_figures_reuse_saved_labels":
                False,
                "retained_k_metrics_and_figures_reuse_saved_labels": True,
                "clustering_directories_pruned_at": (
                    datetime.now().astimezone().isoformat()
                ),
            }
        )
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )

        summary_path = output / "SUMMARY.md"
        summary_lines = summary_path.read_text().splitlines()
        updated: list[str] = []
        for line in summary_lines:
            if line.startswith(
                "- joint/independent metrics and labels:"
            ):
                updated.append(
                    f"- joint/independent metrics: K={METRIC_KS}"
                )
                updated.append(
                    f"- retained clustering directories and full labels: "
                    f"K={PLOT_KS}"
                )
                updated.append(
                    f"- metrics for K={PRUNED_CLUSTERING_KS} remain in CSV, "
                    "but their historical labels_path targets were removed"
                )
            elif line == (
                "- every metric and figure reuses the saved full label CSV files"
            ):
                updated.append(
                    "- retained-K metrics and every figure reuse the saved "
                    "full label CSV files"
                )
            else:
                updated.append(line)
        summary_path.write_text("\n".join(updated) + "\n")
    return deleted


def copy_shared(data: MethodData) -> None:
    output = data.output_root / "shared_metrics"
    output.mkdir()
    rows = []
    for name, source in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        rows.append(
            {
                "artifact": name,
                "source": str(source),
                "source_sha256": sha256(source),
                "copied_sha256": sha256(target),
                "preprocessing_dependent": False,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "source_manifest.csv", index=False
    )


def compare_schemes(data: MethodData) -> None:
    output = data.output_root / "comparison_metrics"
    output.mkdir()
    stability_rows = []
    for mode in ["joint", "independent"]:
        scopes = ["combined"] if mode == "joint" else SECTIONS
        for k in METRIC_KS:
            for scope in scopes:
                relative = (
                    Path("clustering") / f"joint_k{k}" / "labels_all.csv"
                    if mode == "joint"
                    else Path("clustering")
                    / f"independent_k{k}"
                    / f"labels_{scope}.csv"
                )
                raw = pd.read_csv(
                    data.output_root / "raw_embedding" / relative
                )
                standardized = pd.read_csv(
                    data.output_root / "standardized_embedding" / relative
                )
                if not raw[["section", "obs_name"]].equals(
                    standardized[["section", "obs_name"]]
                ):
                    raise ValueError(
                        f"{data.name}: row mismatch {mode} K={k} {scope}"
                    )
                stability_rows.append(
                    {
                        "mode": mode,
                        "scope": scope,
                        "k": k,
                        "n_obs": len(raw),
                        "raw_vs_standardized_ari": adjusted_rand_score(
                            raw["cluster"], standardized["cluster"]
                        ),
                        "raw_vs_standardized_nmi":
                        normalized_mutual_info_score(
                            raw["cluster"], standardized["cluster"]
                        ),
                    }
                )
    pd.DataFrame(stability_rows).to_csv(
        output / "label_stability.csv", index=False
    )

    def delta_table(
        filename: str, keys: list[str], metrics: list[str]
    ) -> None:
        raw = pd.read_csv(
            data.output_root / "raw_embedding" / "metrics" / filename
        )
        standardized = pd.read_csv(
            data.output_root
            / "standardized_embedding"
            / "metrics"
            / filename
        )
        merged = raw.merge(
            standardized,
            on=keys,
            suffixes=("_raw", "_standardized"),
            validate="one_to_one",
        )
        for metric in metrics:
            merged[f"{metric}_delta_standardized_minus_raw"] = (
                merged[f"{metric}_standardized"]
                - merged[f"{metric}_raw"]
            )
        merged.to_csv(
            output / f"{Path(filename).stem}_deltas.csv", index=False
        )

    delta_table(
        "clustering_metrics.csv",
        ["mode", "scope", "k", "n_obs", "embedding_dim"],
        [
            "cluster_asw",
            "cluster_asw_scaled",
            "calinski_harabasz",
            "davies_bouldin",
            "section_ari",
            "section_nmi",
            "section_asw",
        ],
    )
    delta_table(
        "clustering_metrics_by_label.csv",
        [
            "mode",
            "scope",
            "label",
            "n_obs_labeled",
            "n_label_classes",
            "k",
        ],
        [
            "ari",
            "nmi",
            "homogeneity",
            "completeness",
            "v_measure",
            "label_asw",
            "label_asw_scaled",
        ],
    )


def write_manifest(data: MethodData) -> None:
    payload = {
        "dataset": "MISAR-seq",
        "method": data.name,
        "legacy_results_modified": False,
        "comparison_root": str(data.output_root),
        "branches": {
            "raw_embedding": str(data.output_root / "raw_embedding"),
            "standardized_embedding": str(
                data.output_root / "standardized_embedding"
            ),
        },
        "comparison_metrics": str(data.output_root / "comparison_metrics"),
        "shared_metrics": str(data.output_root / "shared_metrics"),
        "metric_k_values": METRIC_KS,
        "plot_k_values": PLOT_KS,
        "retained_clustering_k_values": PLOT_KS,
        "deleted_clustering_k_values": PRUNED_CLUSTERING_KS,
        "metrics_k_values_without_saved_label_directories":
        PRUNED_CLUSTERING_KS,
        "plot_point_size": PLOT_POINT_SIZE,
        "plot_dpi": PLOT_DPI,
        "source_data_dir": str(data.data_dir),
        "source_file_hashes": {
            str(path): sha256(path) for path in data.source_paths
        },
        "generation_script": str(SCRIPT_PATH),
        "generation_script_sha256": sha256(SCRIPT_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (
        data.output_root / "preprocessing_comparison_manifest.json"
    ).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def build_report(
    methods: list[MethodData],
    overwrite: bool = False,
) -> None:
    if REPORT_PATH.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    lines = [
        "# MISAR-seq：Raw 与 Standardized embedding + KMeans 对比报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        "- 本报告仅引用新增 comparison 结果；旧分析目录和原综合报告未覆盖。",
        "- Raw 与 Standardized 分别在自身 KMeans 输入空间计算距离型指标；跨方案变化需结合标签稳定性解释。",
        "",
        "## 1. 新结果目录",
        "",
        md_table(
            ["方法", "comparison root", "方法自有数据副本"],
            [[m.name, m.output_root, m.data_dir] for m in methods],
        ),
        "",
        "每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。",
        "",
        "## 2. 统一标准",
        "",
        md_table(
            ["项目", "统一值"],
            [
                ["spot/barcode", "各方法自有数据；section + original barcode 精确对齐"],
                ["spot 数", "7118（2129/1949/1777/1263）"],
                ["raw 分支", "原始最终 embedding 直接输入 KMeans"],
                ["standardized 分支", "StandardScaler；joint 全体拟合，independent 每section拟合"],
                ["KMeans", "sklearn exact KMeans"],
                ["joint/independent指标 K", "2–16"],
                [
                    "保留的聚类文件夹/labels/空间图 K",
                    "5、8、10、12、14、16",
                ],
                ["空间图点大小", "18"],
                ["随机参数", "seed=0，n_init=20，max_iter=300"],
                ["Cluster ASW/CH/DBI", "每个scope全量样本"],
                ["外部指标", "5类生物标签；全部7118个spot"],
                ["Label ASW", "每个scope全部有效标签spot"],
                ["空间连续性", "每section全点，exact 6-NN"],
                [
                    "labels复用",
                    "保留K的外部、内部、空间指标和图复用全量labels",
                ],
                ["batch指标", "KMeans分支无关；复制并记录SHA256"],
            ],
        ),
        "",
        "## 3. 数据、标签与 embedding 对齐",
        "",
        md_table(
            ["方法", "维度", "dataset1", "dataset2", "dataset3", "dataset4", "标签有效 n"],
            [
                [
                    m.name,
                    m.embedding.shape[1],
                    SECTION_COUNTS["dataset1"],
                    SECTION_COUNTS["dataset2"],
                    SECTION_COUNTS["dataset3"],
                    SECTION_COUNTS["dataset4"],
                    int(valid_truth(m.truth["Y"]).sum()),
                ]
                for m in methods
            ],
        ),
        "",
        "### 最终 embedding 来源",
        "",
    ]
    for method in methods:
        lines.append(f"- {method.name}")
        lines.extend(f"  - `{path}`" for path in method.source_paths)

    lines += [
        "",
        "## 4. 外部指标：Raw / Standardized 严格对齐对比",
        "",
        "同一方法、同一标签的Raw与Standardized行相邻；best K在各分支内按joint ARI独立选择。",
        "",
    ]
    best_rows = []
    for method in methods:
        best_by_scheme = {}
        internal_by_scheme = {}
        for scheme in ["raw_embedding", "standardized_embedding"]:
            best_by_scheme[scheme] = pd.read_csv(
                method.output_root
                / scheme
                / "metrics"
                / "best_clustering_metrics_by_label.csv"
            ).set_index("label")
            internal_by_scheme[scheme] = pd.read_csv(
                method.output_root
                / scheme
                / "metrics"
                / "clustering_metrics.csv"
            )
        for label in LABEL_KEYS:
            for scheme, scheme_label in [
                ("raw_embedding", "Raw"),
                ("standardized_embedding", "Standardized"),
            ]:
                row = best_by_scheme[scheme].loc[label]
                k = int(row["k"])
                internal_row = internal_by_scheme[scheme][
                    (internal_by_scheme[scheme]["mode"] == "joint")
                    & (internal_by_scheme[scheme]["scope"] == "combined")
                    & (internal_by_scheme[scheme]["k"] == k)
                ].iloc[0]
                best_rows.append(
                    [
                        method.name,
                        label,
                        scheme_label,
                        k,
                        int(row["n_obs_labeled"]),
                        int(row["n_label_classes"]),
                        f4(row["ari"]),
                        f4(row["nmi"]),
                        f4(row["homogeneity"]),
                        f4(row["completeness"]),
                        f4(row["v_measure"]),
                        f4(internal_row["cluster_asw"]),
                        f4(internal_row["cluster_asw_scaled"]),
                        f4(row["label_asw"]),
                        f4(row["label_asw_scaled"]),
                    ]
                )
    lines += [
        md_table(
            [
                "方法",
                "标签",
                "预处理",
                "best K",
                "有效 n",
                "标签类数",
                "ARI",
                "NMI",
                "Homogeneity",
                "Completeness",
                "V-measure",
                "Cluster ASW raw",
                "Cluster ASW scaled",
                "Label ASW raw",
                "Label ASW scaled",
            ],
            best_rows,
        ),
        "",
        "完整joint/independent K=2–16逐标签指标见各分支 `metrics/clustering_metrics_by_label.csv`。",
        "",
        "## 5. 内部、section诊断与空间指标：Raw / Standardized 严格对齐对比",
        "",
    ]
    metric_rows = []
    for method in methods:
        for k in PLOT_KS:
            for scheme, scheme_label in [
                ("raw_embedding", "Raw"),
                ("standardized_embedding", "Standardized"),
            ]:
                metrics = pd.read_csv(
                    method.output_root
                    / scheme
                    / "metrics"
                    / "clustering_metrics.csv"
                )
                spatial = pd.read_csv(
                    method.output_root
                    / scheme
                    / "metrics"
                    / "spatial_continuity_summary.csv"
                )
                joint = metrics[
                    metrics["mode"] == "joint"
                ].set_index("k")
                joint_spatial = spatial[
                    spatial["mode"] == "joint"
                ].set_index("k")
                row = joint.loc[k]
                metric_rows.append(
                    [
                        method.name,
                        k,
                        scheme_label,
                        int(row["n_obs"]),
                        f4(row["cluster_asw"]),
                        f4(row["cluster_asw_scaled"]),
                        int(row["n_obs"]),
                        f4(row["calinski_harabasz"]),
                        f4(row["davies_bouldin"]),
                        f4(row["section_ari"]),
                        f4(row["section_nmi"]),
                        int(row["n_obs"]),
                        f4(row["section_asw"]),
                        f4(
                            joint_spatial.loc[
                                k, "mean_spatial_neighbor_agreement"
                            ]
                        ),
                    ]
                )
    lines += [
        md_table(
            [
                "方法",
                "K",
                "预处理",
                "ASW n",
                "Cluster ASW raw",
                "Cluster ASW scaled",
                "CH/DBI n",
                "CH",
                "DBI",
                "section ARI",
                "section NMI",
                "section ASW n",
                "section ASW raw",
                "joint spatial",
            ],
            metric_rows,
        ),
        "",
        "完整joint/independent K=2–16内部与空间指标仍保存在各分支 `metrics/`；聚类文件夹、全量labels和空间图仅保留K=5/8/10/12/14/16，点大小统一为18。CSV中其他K的历史`labels_path`已随对应文件夹删除而失效。",
        "",
        "### 5.1 Joint全K最佳内部指标",
        "",
        "下表在现有K=2–16指标CSV中直接选择最优行；不重新聚类，也不恢复已删除的非绘图K文件夹。",
        "",
    ]
    best_internal_rows = []
    for method in methods:
        for scheme, scheme_label in [
            ("raw_embedding", "Raw"),
            ("standardized_embedding", "Standardized"),
        ]:
            metrics = pd.read_csv(
                method.output_root
                / scheme
                / "metrics"
                / "clustering_metrics.csv"
            )
            joint = metrics[metrics["mode"] == "joint"]
            best_asw = joint.loc[joint["cluster_asw"].idxmax()]
            best_ch = joint.loc[joint["calinski_harabasz"].idxmax()]
            best_dbi = joint.loc[joint["davies_bouldin"].idxmin()]
            max_section_ari = joint.loc[joint["section_ari"].idxmax()]
            max_section_nmi = joint.loc[joint["section_nmi"].idxmax()]
            best_internal_rows.append(
                [
                    method.name,
                    scheme_label,
                    int(best_asw["k"]),
                    f4(best_asw["cluster_asw"]),
                    f4(best_asw["cluster_asw_scaled"]),
                    int(best_ch["k"]),
                    f4(best_ch["calinski_harabasz"]),
                    int(best_dbi["k"]),
                    f4(best_dbi["davies_bouldin"]),
                    int(max_section_ari["k"]),
                    f4(max_section_ari["section_ari"]),
                    int(max_section_nmi["k"]),
                    f4(max_section_nmi["section_nmi"]),
                ]
            )
    lines += [
        md_table(
            [
                "方法",
                "预处理",
                "best ASW K",
                "ASW raw",
                "ASW scaled",
                "best CH K",
                "CH",
                "best DBI K",
                "DBI",
                "max section ARI K",
                "section ARI",
                "max section NMI K",
                "section NMI",
            ],
            best_internal_rows,
        ),
        "",
        "### 5.2 Independent逐section内部指标",
        "",
        "展示保留聚类结果的K=5/8/10/12/14/16；同一方法、同一K、同一section的Raw与Standardized行严格相邻。ASW、CH和DBI均使用该section全量spot。",
        "",
    ]
    independent_rows = []
    for method in methods:
        for k in PLOT_KS:
            for section in SECTIONS:
                for scheme, scheme_label in [
                    ("raw_embedding", "Raw"),
                    ("standardized_embedding", "Standardized"),
                ]:
                    metrics = pd.read_csv(
                        method.output_root
                        / scheme
                        / "metrics"
                        / "clustering_metrics.csv"
                    )
                    row = metrics[
                        (metrics["mode"] == "independent")
                        & (metrics["scope"] == section)
                        & (metrics["k"] == k)
                    ].iloc[0]
                    independent_rows.append(
                        [
                            method.name,
                            k,
                            section,
                            scheme_label,
                            int(row["n_obs"]),
                            int(row["n_obs"]),
                            f4(row["cluster_asw"]),
                            f4(row["cluster_asw_scaled"]),
                            int(row["n_obs"]),
                            f4(row["calinski_harabasz"]),
                            f4(row["davies_bouldin"]),
                        ]
                    )
    lines += [
        md_table(
            [
                "方法",
                "K",
                "section",
                "预处理",
                "scope n",
                "ASW n",
                "Cluster ASW raw",
                "Cluster ASW scaled",
                "CH/DBI n",
                "CH",
                "DBI",
            ],
            independent_rows,
        ),
        "",
        "### 5.3 Joint与Independent空间连续性",
        "",
        "展示保留聚类结果的K=5/8/10/12/14/16；每个值为四个section全量spot空间近邻同簇比例的算术平均，Raw与Standardized行严格相邻。",
        "",
    ]
    spatial_rows = []
    for method in methods:
        for k in PLOT_KS:
            for mode, mode_label in [
                ("joint", "Joint"),
                ("independent", "Independent"),
            ]:
                for scheme, scheme_label in [
                    ("raw_embedding", "Raw"),
                    ("standardized_embedding", "Standardized"),
                ]:
                    spatial = pd.read_csv(
                        method.output_root
                        / scheme
                        / "metrics"
                        / "spatial_continuity_summary.csv"
                    )
                    row = spatial[
                        (spatial["mode"] == mode)
                        & (spatial["k"] == k)
                    ].iloc[0]
                    spatial_rows.append(
                        [
                            method.name,
                            k,
                            mode_label,
                            scheme_label,
                            f4(row["mean_spatial_neighbor_agreement"]),
                        ]
                    )
    lines += [
        md_table(
            [
                "方法",
                "K",
                "模式",
                "预处理",
                "Spatial neighbor agreement",
            ],
            spatial_rows,
        ),
        "",
        "## 6. 两种预处理的标签稳定性",
        "",
    ]
    stability_rows = []
    for method in methods:
        stability = pd.read_csv(
            method.output_root / "comparison_metrics" / "label_stability.csv"
        )
        stability = stability[
            (stability["mode"] == "joint")
            & stability["k"].isin(PLOT_KS)
        ]
        for _, row in stability.iterrows():
            stability_rows.append(
                [
                    method.name,
                    int(row["k"]),
                    f4(row["raw_vs_standardized_ari"]),
                    f4(row["raw_vs_standardized_nmi"]),
                ]
            )
    lines += [
        md_table(
            ["方法", "K", "raw vs standardized ARI", "NMI"],
            stability_rows,
        ),
        "",
        "完整joint和independent K=2–16稳定性见各方法 `comparison_metrics/label_stability.csv`；内部和外部指标差值见同目录另外两个CSV。",
        "",
        "## 7. 共享指标及参数",
        "",
    ]
    batch_sampling_rows = []
    batch_parameter_rows = []
    batch_value_rows = []
    for method in methods:
        batch = pd.read_csv(
            method.output_root
            / "shared_metrics"
            / "batch_correction_metrics.csv"
        ).iloc[0]
        batch_sampling_rows.append(
            [
                method.name,
                str(batch.get("batch_key", "NA")),
                int(batch.get("n_obs_total", batch.get("n_obs", 7118))),
                int(batch.get("n_used", batch.get("n_obs", 7118))),
                int(batch.get("n_batches", 4)),
                str(batch.get("batch_categories", "NA")),
                str(batch.get("batch_counts", "NA")),
                int(batch.get("max_samples", 0)),
                int(batch.get("sample_size_requested", 0)),
                int(batch.get("seed", 0)),
                str(batch.get("embedding_scaled", "NA")),
                int(
                    batch.get(
                        "bASW_sample_size",
                        batch.get("n_used", batch.get("n_obs", 7118)),
                    )
                ),
            ]
        )
        batch_parameter_rows.append(
            [
                method.name,
                str(batch.get("knn_backend", "NA")),
                int(batch.get("bLISI_neighbors", batch.get("n_neighbors", 90))),
                int(batch.get("kBET_neighbors", 50)),
                f4(batch.get("kBET_alpha", 0.05)),
                int(batch.get("PCR_components", batch.get("pcr_n_components"))),
            ]
        )
        batch_value_rows.append(
            [
                method.name,
                f4(
                    batch.get(
                        "bASW_raw",
                        batch.get("basw_batch_silhouette_mean"),
                    )
                ),
                f4(batch.get("bASW", batch.get("basw_score_abs_mean"))),
                f4(batch.get("bLISI_raw", batch.get("blisi_mean"))),
                f4(batch.get("bLISI", batch.get("blisi_normalized"))),
                f4(
                    batch.get(
                        "kBET_rejection_rate",
                        batch.get("kbet_rejection_rate"),
                    )
                ),
                f4(batch.get("kBET", batch.get("kbet_acceptance_rate"))),
                f4(
                    batch.get(
                        "PCR_batch_R2",
                        batch.get("pcr_batch_r2"),
                    )
                ),
                f4(batch.get("PCR_score", batch.get("pcr_score"))),
            ]
        )
    lines += [
        "### 7.1 Batch样本范围",
        "",
        md_table(
            [
                "方法",
                "batch key",
                "n total",
                "n used",
                "n batches",
                "batch categories",
                "batch counts",
                "max samples",
                "requested n",
                "seed",
                "embedding scaled",
                "bASW n",
            ],
            batch_sampling_rows,
        ),
        "",
        "### 7.2 Batch近邻与PCR参数",
        "",
        md_table(
            [
                "方法",
                "kNN backend",
                "bLISI k",
                "kBET k",
                "alpha",
                "PCR PCs",
            ],
            batch_parameter_rows,
        ),
        "",
        "### 7.3 Batch完整指标值",
        "",
        md_table(
            [
                "方法",
                "bASW raw",
                "bASW score",
                "bLISI raw",
                "bLISI normalized",
                "kBET rejection",
                "kBET acceptance",
                "PCR batch R2",
                "PCR score",
            ],
            batch_value_rows,
        ),
        "",
        "batch指标不随本次KMeans输入分支变化，故只保存和展示一份；来源文件与复制文件SHA256见 `shared_metrics/source_manifest.csv`。",
        "",
        "### 7.4 MOFA+视图解释度R²",
        "",
    ]
    mofa = next(method for method in methods if method.name == "MOFA+")
    r2 = pd.read_csv(
        mofa.output_root
        / "shared_metrics"
        / "r2_total_by_view_group.csv"
    )
    lines += [
        md_table(
            ["View", "Group", "R2"],
            [
                [row["View"], row["Group"], f4(row["R2"])]
                for _, row in r2.iterrows()
            ],
        ),
        "",
        "该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。",
        "",
        "### 7.5 SpaMosaic模态对齐",
        "",
    ]
    spamosaic = next(
        method for method in methods if method.name == "SpaMosaic"
    )
    alignment = pd.read_csv(
        spamosaic.output_root
        / "shared_metrics"
        / "modality_alignment.csv"
    )
    lines += [
        md_table(
            [
                "group",
                "n obs",
                "RNA-ATAC cosine mean",
                "RNA-ATAC cosine median",
            ],
            [
                [
                    row["group"],
                    int(row["n_obs"]),
                    f4(row["rna_atac_embedding_cosine_mean"]),
                    f4(row["rna_atac_embedding_cosine_median"]),
                ]
                for _, row in alignment.iterrows()
            ],
        ),
        "",
        "该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。",
        "",
        "## 8. 结论与使用建议",
        "",
        "1. Raw与Standardized均保留joint/independent K=2–16指标CSV；聚类文件夹与全量labels仅保留K=5/8/10/12/14/16。",
        "2. 跨方法排名必须固定同一预处理分支；建议Standardized作为主分析，Raw作为敏感性分析。",
        "3. 同时报告raw-vs-standardized ARI/NMI，以判断预处理对聚类分配的影响。",
        "4. 生物标签外部指标与Label ASW使用全部7118个spot；section ARI/NMI仅是数据来源依赖诊断。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate only the report from existing comparison results.",
    )
    actions.add_argument(
        "--redraw-spatial-figures",
        action="store_true",
        help=(
            "Redraw only K=5/8/10/12/14/16 spatial figures with point "
            "size 18 from saved labels; do not rerun KMeans or metrics."
        ),
    )
    actions.add_argument(
        "--prune-unplotted-clustering",
        action="store_true",
        help=(
            "Delete clustering directories outside K=5/8/10/12/14/16; "
            "keep existing metric CSV files unchanged."
        ),
    )
    args = parser.parse_args()
    methods = [load_spa(), load_mofa(), load_cosie(), load_spamosaic()]
    if args.report_only:
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    if args.redraw_spatial_figures:
        removed = 0
        for method in methods:
            validate(method, allow_existing_output=True)
            removed += redraw_selected_spatial_figures(method)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(
            f"report: {REPORT_PATH}; removed_unselected_figures={removed}",
            flush=True,
        )
        return
    if args.prune_unplotted_clustering:
        deleted = 0
        for method in methods:
            validate(method, allow_existing_output=True)
            deleted += prune_unplotted_clustering_directories(method)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(
            f"report: {REPORT_PATH}; deleted_clustering_directories={deleted}",
            flush=True,
        )
        return
    for method in methods:
        validate(method)
    if REPORT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    for method in methods:
        print(f"[{method.name}] raw_embedding", flush=True)
        run_scheme(method, "raw_embedding")
        print(f"[{method.name}] standardized_embedding", flush=True)
        run_scheme(method, "standardized_embedding")
        copy_shared(method)
        compare_schemes(method)
        prune_unplotted_clustering_directories(method)
        write_manifest(method)
    build_report(methods)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
