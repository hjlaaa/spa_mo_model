#!/usr/bin/env python3
"""Create raw-vs-standardized KMeans results for Mouse_Thymus.

The four methods use only their own data and final embeddings.  KMeans, CH and
DBI use every spot.  Cluster ASW uses one persisted, section-stratified set of
10,000 section/barcode identities shared across methods, preprocessing schemes
and K values. Metric CSV files retain K=2..12, while clustering directories,
labels, and spatial figures are retained only for K=5/8/10/12.
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
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances

from compare_mouse_spleen_kmeans_preprocessing import (
    f4,
    md_table,
    metric_space,
    scaler_payload,
    sha256,
    spatial_agreement,
)


SECTIONS = [f"Mouse_Thymus{i}" for i in range(1, 5)]
SECTION_COUNTS = {
    "Mouse_Thymus1": 4697,
    "Mouse_Thymus2": 4253,
    "Mouse_Thymus3": 4646,
    "Mouse_Thymus4": 4228,
}
ASW_SECTION_COUNTS = {
    "Mouse_Thymus1": 2635,
    "Mouse_Thymus2": 2386,
    "Mouse_Thymus3": 2607,
    "Mouse_Thymus4": 2372,
}
N_OBS = sum(SECTION_COUNTS.values())
ASW_N_OBS = sum(ASW_SECTION_COUNTS.values())
METRIC_KS = list(range(2, 13))
PLOT_KS = [5, 8, 10, 12]
PRUNED_CLUSTERING_KS = [
    k for k in METRIC_KS if k not in PLOT_KS
]
SEED = 0
N_INIT = 20
MAX_ITER = 300
SPATIAL_NEIGHBOR_K = 6
POINT_SIZES = {
    "Mouse_Thymus1": 8.0,
    "Mouse_Thymus2": 5.0,
    "Mouse_Thymus3": 5.0,
    "Mouse_Thymus4": 5.0,
}
PLOT_DPI = 220
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/"
    "mouse_thymus_preprocessing_comparison_report.md"
)
SCRIPT_PATH = Path(__file__).resolve()


@dataclass
class MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]


def aligned_coords(
    data_dir: Path, sections: np.ndarray, barcodes: np.ndarray
) -> np.ndarray:
    coords = np.empty((len(sections), 2), dtype=float)
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
            if not {"x", "y"}.issubset(rna.obs.columns):
                raise ValueError(f"{rna_path}: missing canonical obs[x,y]")
            coords[mask] = rna.obs[["x", "y"]].to_numpy()[positions]
        finally:
            rna.file.close()
    return coords


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/mouse_thymus/"
        "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/Mouse_Thymus")
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        metadata = pd.read_csv(metadata_path)
        embedding = np.load(embedding_path)
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            own_names = rna.obs_names.astype(str).to_numpy()
            own_coords = rna.obs[["x", "y"]].to_numpy()
        finally:
            rna.file.close()
        saved_names = metadata["obs_name"].astype(str).to_numpy()
        saved_coords = metadata[["spatial_x", "spatial_y"]].to_numpy()
        if (
            len(embedding) != len(saved_names)
            or not np.array_equal(saved_names, own_names)
            or not np.allclose(saved_coords, own_coords)
        ):
            raise ValueError(
                f"spa_mo_model {section}: embedding/metadata/data mismatch"
            )
        arrays.append(embedding)
        sections.extend([section] * len(saved_names))
        barcodes.extend(saved_names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / "clustering_analysis"
    return MethodData(
        "spa_mo_model",
        np.vstack(arrays),
        section_array,
        barcode_array,
        aligned_coords(data_dir, section_array, barcode_array),
        data_dir,
        sources,
        Path(
            "/home/hujinlan/spa_mo_model/results/"
            "mouse_thymus_preprocessing_comparison"
        ),
        {"batch_correction_metrics.csv": old / "batch_correction_metrics.csv"},
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "mouse_thymus_mofa_hvg2000_k10_iter1000"
    )
    table_path = old / "tables" / "factors_with_metadata_and_coordinates.csv"
    table = pd.read_csv(table_path)
    factor_columns = [
        column for column in table.columns if column.startswith("Factor")
    ]
    sections = table["section"].astype(str).to_numpy()
    barcodes = table["original_barcode"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/Mouse_Thymus")
    return MethodData(
        "MOFA+",
        table[factor_columns].to_numpy(float),
        sections,
        barcodes,
        aligned_coords(data_dir, sections, barcodes),
        data_dir,
        [table_path],
        Path(
            "/home/hujinlan/mofa+/analysis/"
            "mouse_thymus_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "r2_total_by_view_group.csv": old
            / "tables"
            / "r2_total_by_view_group.csv",
            "input_audit.csv": old / "input_audit.csv",
        },
    )


def load_cosie() -> MethodData:
    run = Path(
        "/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full"
    )
    old = run / "analysis"
    table_path = old / "tables" / "embeddings_with_metadata.csv"
    table = pd.read_csv(table_path)
    embedding_columns = [
        column for column in table.columns if column.startswith("COSIE")
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
        "obs_name.1" if "obs_name.1" in table.columns else "obs_name"
    )
    barcodes = table[barcode_column].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/cosie/data/Mouse_Thymus")
    return MethodData(
        "COSIE",
        table_embedding,
        sections,
        barcodes,
        aligned_coords(data_dir, sections, barcodes),
        data_dir,
        [table_path, *final_paths],
        Path(
            "/home/hujinlan/cosie_runs/"
            "mouse_thymus_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv"
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/mouse_thymus_spamosaic/"
        "mouse_thymus_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs["section"].astype(str).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path(
        "/home/hujinlan/SpaMosaic-dev/demo/data/Mouse_Thymus"
    )
    old = Path(
        "/home/hujinlan/SpaMosaic-dev/analysis/mouse_thymus_spamosaic"
    )
    return MethodData(
        "SpaMosaic",
        embedding,
        sections,
        barcodes,
        aligned_coords(data_dir, sections, barcodes),
        data_dir,
        [h5ad_path],
        Path(
            "/home/hujinlan/SpaMosaic-dev/analysis/"
            "mouse_thymus_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "modality_alignment_cosine.csv": old
            / "metrics"
            / "modality_alignment_cosine.csv",
        },
    )


def validate(
    data: MethodData,
    allow_existing_output: bool = False,
) -> None:
    n_obs = len(data.embedding)
    if (
        n_obs != N_OBS
        or len(data.sections) != n_obs
        or len(data.barcodes) != n_obs
        or len(data.coords) != n_obs
    ):
        raise ValueError(
            f"{data.name}: expected/aligned {N_OBS} rows, got {n_obs}"
        )
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f"{data.name}: non-finite embedding or coordinates")
    counts = {
        str(key): int(value)
        for key, value in pd.Series(data.sections).value_counts().items()
    }
    if counts != SECTION_COUNTS:
        raise ValueError(f"{data.name}: unexpected section counts {counts}")
    if len(set(zip(data.sections, data.barcodes))) != n_obs:
        raise ValueError(f"{data.name}: duplicate section/barcode keys")
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


def build_asw_sample(methods: list[MethodData]) -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "section": methods[0].sections,
            "obs_name": methods[0].barcodes,
        }
    )
    section_rank = {section: rank for rank, section in enumerate(SECTIONS)}
    reference["section_rank"] = reference["section"].map(section_rank)
    reference = reference.sort_values(
        ["section_rank", "obs_name"], kind="mergesort"
    ).drop(columns="section_rank")
    reference_index = pd.MultiIndex.from_frame(reference)
    for method in methods[1:]:
        candidate = pd.MultiIndex.from_arrays(
            [method.sections, method.barcodes],
            names=["section", "obs_name"],
        )
        if len(candidate) != len(reference_index) or set(candidate) != set(
            reference_index
        ):
            raise ValueError(
                f"{method.name}: section/barcode universe differs from other methods"
            )

    rng = np.random.default_rng(SEED)
    rows = []
    sample_order = 0
    for section in SECTIONS:
        candidates = reference[reference["section"] == section].reset_index(
            drop=True
        )
        chosen = np.sort(
            rng.choice(
                len(candidates),
                size=ASW_SECTION_COUNTS[section],
                replace=False,
            )
        )
        selected = candidates.iloc[chosen]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "sample_order": sample_order,
                    "section": row["section"],
                    "obs_name": row["obs_name"],
                    "section_n_total": SECTION_COUNTS[section],
                    "section_n_sampled": ASW_SECTION_COUNTS[section],
                    "sampling_seed": SEED,
                }
            )
            sample_order += 1
    result = pd.DataFrame(rows)
    if len(result) != ASW_N_OBS or result[
        ["section", "obs_name"]
    ].duplicated().any():
        raise ValueError("invalid fixed ASW sample")
    return result


def sample_mask(data: MethodData, sample: pd.DataFrame) -> np.ndarray:
    selected = set(zip(sample["section"], sample["obs_name"]))
    mask = np.fromiter(
        (
            (section, barcode) in selected
            for section, barcode in zip(data.sections, data.barcodes)
        ),
        dtype=bool,
        count=len(data.embedding),
    )
    if int(mask.sum()) != ASW_N_OBS:
        raise ValueError(f"{data.name}: fixed ASW sample does not map to 10000 spots")
    for section, expected in ASW_SECTION_COUNTS.items():
        if int(np.sum(mask & (data.sections == section))) != expected:
            raise ValueError(f"{data.name}: ASW stratum mismatch for {section}")
    return mask


def safe_precomputed_asw(
    distances: np.ndarray, labels: np.ndarray
) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(distances, labels, metric="precomputed"))


def internal_metrics(
    full_space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
) -> dict[str, float | int]:
    sample_labels = labels[sample_indices]
    asw = safe_precomputed_asw(sample_distances, sample_labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "cluster_asw_n_obs": len(sample_indices),
        "calinski_harabasz": float(
            calinski_harabasz_score(full_space, labels)
        ),
        "davies_bouldin": float(davies_bouldin_score(full_space, labels)),
        "ch_dbi_n_obs": len(labels),
    }


def save_labels(
    path: Path, data: MethodData, mask: np.ndarray, labels: np.ndarray
) -> None:
    pd.DataFrame(
        {
            "section": data.sections[mask],
            "obs_name": data.barcodes[mask],
            "cluster": labels.astype(int),
            "x": data.coords[mask, 0],
            "y": data.coords[mask, 1],
        }
    ).to_csv(path, index=False)


def plot_spatial(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    section: str,
) -> None:
    categories, codes = np.unique(labels, return_inverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=POINT_SIZES[section],
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


def run_scheme(
    data: MethodData, scheme: str, asw_sample: pd.DataFrame
) -> None:
    output = data.output_root / scheme
    clustering_root = output / "clustering"
    metrics_root = output / "metrics"
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    fixed_mask = sample_mask(data, asw_sample)
    all_mask = np.ones(len(data.embedding), dtype=bool)

    joint_space, joint_scaler = metric_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, "joint"))
    joint_sample_indices = np.flatnonzero(fixed_mask)
    joint_distances = pairwise_distances(
        joint_space[joint_sample_indices],
        metric="euclidean",
        n_jobs=-1,
    )
    np.fill_diagonal(joint_distances, 0.0)
    section_asw = safe_precomputed_asw(
        joint_distances, data.sections[joint_sample_indices]
    )

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
                **internal_metrics(
                    joint_space,
                    labels_all,
                    joint_sample_indices,
                    joint_distances,
                ),
                "section_ari": adjusted_rand_score(
                    data.sections, labels_all
                ),
                "section_nmi": normalized_mutual_info_score(
                    data.sections, labels_all
                ),
                "section_asw": section_asw,
                "section_asw_n_obs": ASW_N_OBS,
                "metric_space": scheme,
                "labels_path": str(all_labels_path),
            }
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
                    section,
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
    del joint_distances

    independent_counts: dict[int, list[dict[str, Any]]] = {
        k: [] for k in METRIC_KS
    }
    for k in METRIC_KS:
        (clustering_root / f"independent_k{k}").mkdir()
    for section in SECTIONS:
        mask = data.sections == section
        section_space, scaler = metric_space(data.embedding[mask], scheme)
        if scaler is not None:
            scaler_arrays.update(scaler_payload(scaler, section))
        local_sample_mask = fixed_mask[mask]
        local_sample_indices = np.flatnonzero(local_sample_mask)
        local_distances = pairwise_distances(
            section_space[local_sample_indices],
            metric="euclidean",
            n_jobs=-1,
        )
        np.fill_diagonal(local_distances, 0.0)
        for k in METRIC_KS:
            labels = KMeans(
                k,
                random_state=SEED,
                n_init=N_INIT,
                max_iter=MAX_ITER,
            ).fit_predict(section_space)
            k_directory = clustering_root / f"independent_k{k}"
            labels_path = k_directory / f"labels_{section}.csv"
            save_labels(labels_path, data, mask, labels)
            if k in PLOT_KS:
                plot_spatial(
                    k_directory / f"spatial_{section}.png",
                    data.coords[mask],
                    labels,
                    f"{data.name} {scheme} independent K={k} {section}",
                    section,
                )
            internal_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **internal_metrics(
                        section_space,
                        labels,
                        local_sample_indices,
                        local_distances,
                    ),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "section_asw_n_obs": 0,
                    "metric_space": scheme,
                    "labels_path": str(labels_path),
                }
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
                independent_counts[k].append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(mask.sum()),
                    }
                )
        del local_distances
    for k, rows in independent_counts.items():
        pd.DataFrame(rows).to_csv(
            clustering_root / f"independent_k{k}" / "cluster_counts.csv",
            index=False,
        )

    metrics = pd.DataFrame(internal_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / "clustering_metrics.csv", index=False)
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

    sample_path = data.output_root / "shared_metrics" / "asw_sample.csv"
    config = {
        "dataset": "Mouse_Thymus",
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
            "none" if scheme == "raw_embedding" else "all_17824_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "kmeans_type": "sklearn.cluster.KMeans",
        "joint_metric_k_values": METRIC_KS,
        "independent_metric_k_values": METRIC_KS,
        "spatial_metric_k_values": METRIC_KS,
        "plot_k_values": PLOT_KS,
        "plot_point_size_by_section": POINT_SIZES,
        "plot_dpi": PLOT_DPI,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "n_obs": len(data.embedding),
        "section_counts": SECTION_COUNTS,
        "embedding_dim": data.embedding.shape[1],
        "cluster_asw_sample_size_joint": ASW_N_OBS,
        "cluster_asw_sample_size_independent": ASW_SECTION_COUNTS,
        "cluster_asw_rule":
        "fixed_section_stratified_section_barcode_sample_shared_everywhere",
        "cluster_asw_sampling_seed": SEED,
        "cluster_asw_sample_file": str(sample_path),
        "cluster_asw_sample_sha256": sha256(sample_path),
        "ch_dbi_sample_size": 0,
        "ch_dbi_rule": "full_n_obs_in_each_scope",
        "label_asw": "not_applicable_no_ground_truth_labels",
        "section_asw_sample_size": ASW_N_OBS,
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "exact_knn_within_each_section",
        "spatial_graph_n_obs": SECTION_COUNTS,
        "joint_and_independent_labels_cover_all_spots": True,
        "internal_spatial_and_figures_reuse_saved_labels": True,
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
        f"- spots: {N_OBS}; dimensions: {data.embedding.shape[1]}\n"
        f"- exact KMeans: seed={SEED}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}\n"
        f"- joint/independent metrics: K={METRIC_KS}\n"
        f"- retained clustering directories and full labels: K={PLOT_KS}\n"
        f"- spatial figures: K={PLOT_KS}; point sizes={POINT_SIZES}\n"
        f"- metrics for K={PRUNED_CLUSTERING_KS} remain in CSV, but their "
        "historical labels_path targets were removed\n"
        f"- Cluster ASW: fixed section-stratified {ASW_N_OBS}-spot sample; "
        f"independent quotas={ASW_SECTION_COUNTS}\n"
        "- CH and DBI: full samples in each scope\n"
        "- Label ASW: not applicable because no ground-truth label exists\n"
        f"- spatial continuity: exact {SPATIAL_NEIGHBOR_K}-NN per section\n"
        "- retained-K metrics and every figure reuse saved full label CSVs\n"
    )


def redraw_selected_spatial_figures(data: MethodData) -> None:
    """Redraw retained K figures from saved labels without rerunning KMeans."""
    for scheme in ["raw_embedding", "standardized_embedding"]:
        output = data.output_root / scheme
        clustering_root = output / "clustering"
        metrics_root = output / "metrics"
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
                        f"point_size={POINT_SIZES[section]:g}",
                        flush=True,
                    )
                    plot_spatial(
                        figure_path,
                        labels[["x", "y"]].to_numpy(float),
                        labels["cluster"].to_numpy(int),
                        f"{data.name} {scheme} {mode} K={k} {section}",
                        section,
                    )
                    figure_rows.append(
                        {
                            "mode": mode,
                            "k": k,
                            "section": section,
                            "plot_n_obs": len(labels),
                            "point_size": POINT_SIZES[section],
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
                "plot_point_size_by_section": POINT_SIZES,
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
                f"- spatial figures: K={PLOT_KS}; point sizes={POINT_SIZES}"
                if line.startswith("- spatial figures:")
                else line
            )
            for line in summary_lines
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n")


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
                "internal_spatial_and_figures_reuse_saved_labels": False,
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
                "- joint/independent metrics and full labels:"
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
                "- every internal/spatial metric and figure reuses saved label CSVs"
            ):
                updated.append(
                    "- retained-K metrics and every figure reuse saved full "
                    "label CSVs"
                )
            else:
                updated.append(line)
        summary_path.write_text("\n".join(updated) + "\n")
    return deleted


def prepare_shared(
    data: MethodData, asw_sample: pd.DataFrame
) -> None:
    output = data.output_root / "shared_metrics"
    output.mkdir(parents=True)
    asw_path = output / "asw_sample.csv"
    asw_sample.to_csv(asw_path, index=False)
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
    sample_manifest = {
        "sample_file": str(asw_path),
        "sha256": sha256(asw_path),
        "n_obs": ASW_N_OBS,
        "section_counts": ASW_SECTION_COUNTS,
        "seed": SEED,
        "identity_key": ["section", "obs_name"],
        "selection":
        "section-proportional largest-remainder quotas; sorted candidates; "
        "numpy default_rng(seed=0) without replacement",
        "shared_across_methods_preprocessing_k_and_modes": True,
    }
    (output / "asw_sample_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2, ensure_ascii=False) + "\n"
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
                key = ["section", "obs_name"]
                if not raw[key].equals(standardized[key]):
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

    raw_metrics = pd.read_csv(
        data.output_root
        / "raw_embedding"
        / "metrics"
        / "clustering_metrics.csv"
    )
    standardized_metrics = pd.read_csv(
        data.output_root
        / "standardized_embedding"
        / "metrics"
        / "clustering_metrics.csv"
    )
    keys = [
        "mode",
        "scope",
        "k",
        "n_obs",
        "embedding_dim",
        "cluster_asw_n_obs",
        "ch_dbi_n_obs",
    ]
    merged = raw_metrics.merge(
        standardized_metrics,
        on=keys,
        suffixes=("_raw", "_standardized"),
        validate="one_to_one",
    )
    for metric in [
        "cluster_asw",
        "cluster_asw_scaled",
        "calinski_harabasz",
        "davies_bouldin",
        "section_ari",
        "section_nmi",
        "section_asw",
    ]:
        merged[f"{metric}_delta_standardized_minus_raw"] = (
            merged[f"{metric}_standardized"]
            - merged[f"{metric}_raw"]
        )
    merged.to_csv(output / "metric_deltas.csv", index=False)


def write_manifest(data: MethodData) -> None:
    sample_path = data.output_root / "shared_metrics" / "asw_sample.csv"
    payload = {
        "dataset": "Mouse_Thymus",
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
        "plot_point_size_by_section": POINT_SIZES,
        "plot_dpi": PLOT_DPI,
        "asw_sample": {
            "path": str(sample_path),
            "sha256": sha256(sample_path),
            "n_obs": ASW_N_OBS,
            "section_counts": ASW_SECTION_COUNTS,
        },
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
    methods: list[MethodData], overwrite: bool = False
) -> None:
    if REPORT_PATH.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    sample_hashes = {
        sha256(m.output_root / "shared_metrics" / "asw_sample.csv")
        for m in methods
    }
    if len(sample_hashes) != 1:
        raise ValueError("ASW sample files do not have identical hashes")
    sample_hash = next(iter(sample_hashes))
    lines = [
        "# Mouse_Thymus：Raw 与 Standardized embedding + KMeans 对比报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        "- 本报告只引用新增 comparison 结果；旧分析目录和原综合报告未覆盖。",
        "- Raw与Standardized分别在自身KMeans输入空间计算距离型指标；跨方案变化需结合标签稳定性解释。",
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
                ["spot/barcode", "各方法自有数据；section + original barcode精确对齐"],
                ["spot数", "17824（4697/4253/4646/4228）"],
                ["embedding", "各方法当前最终embedding"],
                ["raw分支", "原始最终embedding直接输入KMeans"],
                ["standardized分支", "StandardScaler；joint全体拟合，independent每section拟合"],
                ["KMeans", "sklearn exact KMeans"],
                ["joint/independent指标K", "2–12"],
                ["保留的聚类文件夹/labels/空间图K", "5、8、10、12"],
                [
                    "空间图点大小",
                    "Mouse_Thymus1=8；Mouse_Thymus2/3/4=5",
                ],
                ["随机参数", "seed=0，n_init=20，max_iter=300"],
                ["Cluster ASW", "固定同一批section分层10000个spot"],
                ["ASW分层数", "2635/2386/2607/2372"],
                ["CH/DBI", "每个scope全量样本"],
                ["Label ASW", "无真实细胞/区域标签，不计算"],
                ["空间连续性", "每section全点，exact 6-NN"],
                [
                    "labels复用",
                    "保留K的内部、空间指标和图复用保存的全量labels",
                ],
                ["batch指标", "KMeans分支无关；复制原结果并记录SHA256"],
            ],
        ),
        "",
        "固定ASW名单保存在每个根目录 `shared_metrics/asw_sample.csv`；四份文件SHA256完全相同：",
        "",
        f"`{sample_hash}`",
        "",
        "## 3. 数据与 embedding 对齐",
        "",
        md_table(
            ["方法", "维度", "Thymus1", "Thymus2", "Thymus3", "Thymus4", "总spot"],
            [
                [
                    m.name,
                    m.embedding.shape[1],
                    *[
                        int((m.sections == section).sum())
                        for section in SECTIONS
                    ],
                    len(m.embedding),
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
        "## 4. Joint内部与空间指标：Raw / Standardized上下对齐",
        "",
        "同一方法、同一K的Raw与Standardized行相邻，所有列采用完全相同定义。",
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
                        int(row["cluster_asw_n_obs"]),
                        f4(row["cluster_asw"]),
                        f4(row["cluster_asw_scaled"]),
                        int(row["ch_dbi_n_obs"]),
                        f4(row["calinski_harabasz"]),
                        f4(row["davies_bouldin"]),
                        f4(row["section_ari"]),
                        f4(row["section_nmi"]),
                        int(row["section_asw_n_obs"]),
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
        "完整joint/independent K=2–12内部与空间指标仍保存在各分支 `metrics/`；聚类文件夹、全量labels和空间图仅保留K=5/8/10/12。Mouse_Thymus1点大小为8，Mouse_Thymus2/3/4点大小为5。CSV中其他K的历史`labels_path`已随对应文件夹删除而失效。",
        "",
        "### 4.1 Joint全K最佳内部指标",
        "",
        "下表在现有joint K=2–12指标CSV中直接选择最优行，不重新聚类，也不恢复已删除的非绘图K文件夹。",
        "",
    ]
    best_rows = []
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
            best_rows.append(
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
            best_rows,
        ),
        "",
        "### 4.2 Independent逐section内部指标",
        "",
        "展示保留聚类结果的K=5/8/10/12；同一方法、同一K、同一section的Raw与Standardized行严格相邻。ASW使用固定分层样本在对应section中的子集，CH/DBI使用该section全量spot。",
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
                            int(row["cluster_asw_n_obs"]),
                            f4(row["cluster_asw"]),
                            f4(row["cluster_asw_scaled"]),
                            int(row["ch_dbi_n_obs"]),
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
        "### 4.3 Joint与Independent空间连续性",
        "",
        "展示K=5/8/10/12；每个值为四个section全量spot空间近邻同簇比例的算术平均，Raw与Standardized行严格相邻。",
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
        "## 5. 两种预处理的标签稳定性",
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
        "完整joint和independent K=2–12稳定性见各方法 `comparison_metrics/label_stability.csv`；内部指标差值见 `metric_deltas.csv`。",
        "",
        "## 6. 共享指标及参数",
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
                int(batch.get("n_obs_total", batch.get("n_obs", N_OBS))),
                int(batch.get("n_used", batch.get("n_obs", N_OBS))),
                int(batch.get("n_batches", len(SECTIONS))),
                str(batch.get("batch_categories", "NA")),
                str(batch.get("batch_counts", "NA")),
                int(batch.get("max_samples", 0)),
                int(batch.get("sample_size_requested", 0)),
                int(batch.get("seed", 0)),
                str(batch.get("embedding_scaled", "NA")),
                int(
                    batch.get(
                        "bASW_sample_size",
                        batch.get("n_used", batch.get("n_obs", ASW_N_OBS)),
                    )
                ),
            ]
        )
        batch_parameter_rows.append(
            [
                method.name,
                str(batch.get("knn_backend", "NA")),
                int(batch.get("bLISI_neighbors", 90)),
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
        "### 6.1 Batch样本范围",
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
        "### 6.2 Batch近邻与PCR参数",
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
        "### 6.3 Batch完整指标值",
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
        "batch指标不随本次KMeans输入分支变化，故只保存一份；来源和复制文件SHA256见 `shared_metrics/source_manifest.csv`。PCR PCs按 `min(50, embedding维度)`。",
        "",
        "### 6.4 MOFA+视图解释度R²",
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
        "### 6.5 SpaMosaic模态对齐",
        "",
    ]
    spamosaic = next(
        method for method in methods if method.name == "SpaMosaic"
    )
    alignment = pd.read_csv(
        spamosaic.output_root
        / "shared_metrics"
        / "modality_alignment_cosine.csv"
    )
    lines += [
        md_table(
            [
                "pair",
                "group",
                "n spots",
                "mean cosine",
                "median cosine",
                "std cosine",
            ],
            [
                [
                    row["pair"],
                    row["group"],
                    int(row["n_spots"]),
                    f4(row["mean_cosine"]),
                    f4(row["median_cosine"]),
                    f4(row["std_cosine"]),
                ]
                for _, row in alignment.iterrows()
            ],
        ),
        "",
        "该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。",
        "",
        "## 7. 结论与使用建议",
        "",
        "1. Raw与Standardized均保留joint/independent K=2–12指标CSV；聚类文件夹与全量labels仅保留K=5/8/10/12。",
        "2. Cluster ASW的具体10000个section/barcode身份已持久化并在四种方法、两种预处理、所有K和joint/independent间固定复用；CH/DBI始终全量。",
        "3. 跨方法排名必须固定同一预处理分支；建议Standardized作为主分析，Raw作为敏感性分析。",
        "4. Mouse_Thymus没有统一真实区域标签；section ARI/NMI是切片来源依赖诊断，不是生物学聚类准确率。",
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
            "Redraw K=5/8/10/12 figures from saved labels with section "
            "point sizes 8/5/5/5; do not rerun KMeans or metrics."
        ),
    )
    actions.add_argument(
        "--prune-unplotted-clustering",
        action="store_true",
        help=(
            "Delete clustering directories outside K=5/8/10/12; keep "
            "existing metric CSV files unchanged."
        ),
    )
    args = parser.parse_args()
    methods = [load_spa(), load_mofa(), load_cosie(), load_spamosaic()]
    if args.report_only:
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    if args.redraw_spatial_figures:
        for method in methods:
            validate(method, allow_existing_output=True)
            redraw_selected_spatial_figures(method)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
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
    asw_sample = build_asw_sample(methods)
    for method in methods:
        prepare_shared(method, asw_sample)
        print(f"[{method.name}] raw_embedding", flush=True)
        run_scheme(method, "raw_embedding", asw_sample)
        print(f"[{method.name}] standardized_embedding", flush=True)
        run_scheme(method, "standardized_embedding", asw_sample)
        compare_schemes(method)
        prune_unplotted_clustering_directories(method)
        write_manifest(method)
    build_report(methods)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
