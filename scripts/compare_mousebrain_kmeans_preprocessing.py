#!/usr/bin/env python3
"""Create independent raw-vs-standardized KMeans results for MouseBrain.

Original training/analysis outputs are read-only.  Each method is aligned with
its own MouseBrain data copy and written to a new comparison root.  Metric
tables retain joint K=2-12, while clustering directories, saved labels, and
figures retain K=5/6/8/10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
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
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SECTIONS = ["s1", "s2", "s3"]
SECTION_DIRS = {
    "s1": "dataset_MouseBrain_SectionA",
    "s2": "dataset_MouseBrain_SectionB",
    "s3": "dataset_MouseBrain_SectionC",
}
SECTION_NAMES = {"s1": "SectionA", "s2": "SectionB", "s3": "SectionC"}
SECTION_COUNTS = {"s1": 2384, "s2": 2820, "s3": 2662}
BIOLOGICAL_LABEL_KEYS = [
    "RegionLoupe",
    "annotations",
    "celltype",
    "Y.l1",
    "Y",
]
GROUP_LABEL_KEY = "group"
LABEL_KEYS = [*BIOLOGICAL_LABEL_KEYS, GROUP_LABEL_KEY]
METRIC_KS = list(range(2, 13))
PLOT_KS = [5, 6, 8, 10]
PRUNED_CLUSTERING_KS = [k for k in METRIC_KS if k not in PLOT_KS]
SEED = 0
N_INIT = 20
MAX_ITER = 300
SPATIAL_NEIGHBOR_K = 6
POINT_SIZE = 10.0
DPI = 220
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/mousebrain_preprocessing_comparison_report.md"
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
            continue
        text = str(value)
        if text.startswith("b'") and text.endswith("'"):
            text = text[2:-1]
        result[i] = text
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
        rna_path = data_dir / SECTION_DIRS[section] / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            pos = source.get_indexer(wanted)
            if (pos < 0).any():
                examples = wanted[pos < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: {int((pos < 0).sum())} barcodes failed "
                    f"alignment; examples={examples}"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[pos, :2]
            for label in LABEL_KEYS:
                if label in rna.obs:
                    truth[label][mask] = clean_truth(
                        rna.obs[label].to_numpy()[pos]
                    )
        finally:
            rna.file.close()
    # MouseBrain source copies do not contain a literal `group` column.
    # The original comprehensive report defined group as the section/source
    # partition, which is exactly the aligned s1/s2/s3 vector used here.
    truth[GROUP_LABEL_KEY] = sections.astype(str).copy()
    return coords, truth


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/mousebrain_test/"
        "fullspot_warmup_schedule/epochs_200"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/dataset_MouseBrain")
    arrays, sections, barcodes, sources = [], [], [], []
    for section in SECTIONS:
        emb_path = run / "final_embeddings" / f"{section}_final_embedding.npy"
        rna_path = data_dir / SECTION_DIRS[section] / "adata_RNA.h5ad"
        embedding = np.load(emb_path)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(f"spa_mo_model {section}: embedding/data row mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.append(emb_path)
    section_arr = np.asarray(sections)
    barcode_arr = np.asarray(barcodes)
    coords, truth = aligned_metadata(data_dir, section_arr, barcode_arr)
    old = run / "analysis"
    return MethodData(
        "spa_mo_model",
        np.vstack(arrays),
        section_arr,
        barcode_arr,
        coords,
        truth,
        data_dir,
        sources,
        Path(
            "/home/hujinlan/spa_mo_model/results/"
            "mousebrain_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "clustering"
            / "batch_correction_metrics.csv"
        },
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "mousebrain_mofa_rna_meta_uni_hvg2000_k10_iter1000"
    )
    table_path = old / "tables" / "factors_with_metadata_and_coordinates.csv"
    table = pd.read_csv(table_path)
    factor_cols = [column for column in table if column.startswith("Factor")]
    section_map = {"SectionA": "s1", "SectionB": "s2", "SectionC": "s3"}
    sections = table["section"].astype(str).map(section_map).to_numpy()
    barcodes = table["original_barcode"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/dataset_MouseBrain")
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    return MethodData(
        "MOFA+",
        table[factor_cols].to_numpy(float),
        sections,
        barcodes,
        coords,
        truth,
        data_dir,
        [table_path],
        Path(
            "/home/hujinlan/mofa+/analysis/"
            "mousebrain_preprocessing_comparison"
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
    run = Path(
        "/home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full"
    )
    old = run / "analysis"
    table_path = old / "tables" / "embeddings_with_metadata.csv"
    table = pd.read_csv(table_path)
    emb_cols = [column for column in table if column.startswith("COSIE")]
    sections = table["section"].astype(str).to_numpy()
    barcodes = table["obs_name"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/cosie/data/dataset_MouseBrain")
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    final_paths = [
        run / "final_embeddings" / f"{section}_final_embedding.npy"
        for section in SECTIONS
    ]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    table_embedding = table[emb_cols].to_numpy(float)
    if (
        final_embedding.shape != table_embedding.shape
        or not np.allclose(final_embedding, table_embedding)
    ):
        raise ValueError("COSIE: analysis table is not the saved final embedding")
    sources = [table_path, *final_paths]
    return MethodData(
        "COSIE",
        table_embedding,
        sections,
        barcodes,
        coords,
        truth,
        data_dir,
        sources,
        Path(
            "/home/hujinlan/cosie_runs/"
            "mousebrain_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv"
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/mousebrain_spamosaic/"
        "mousebrain_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    section_map = {"SectionA": "s1", "SectionB": "s2", "SectionC": "s3"}
    sections = adata.obs["section"].astype(str).map(section_map).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path(
        "/home/hujinlan/SpaMosaic-dev/demo/data/dataset_MouseBrain"
    )
    coords, truth = aligned_metadata(data_dir, sections, barcodes)
    old = Path("/home/hujinlan/SpaMosaic-dev/analysis/mousebrain_spamosaic")
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
            "mousebrain_preprocessing_comparison"
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


def valid_truth(values: np.ndarray) -> np.ndarray:
    return pd.notna(values) & ~np.isin(values.astype(str), ["nan", "None", ""])


def validate(
    data: MethodData, allow_existing_output: bool = False
) -> None:
    n = len(data.embedding)
    if n != 7866 or len(data.sections) != n or len(data.barcodes) != n:
        raise ValueError(f"{data.name}: expected/aligned 7866 rows, got {n}")
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f"{data.name}: non-finite embedding or coordinates")
    counts = {
        key: int(value)
        for key, value in pd.Series(data.sections).value_counts().items()
    }
    if counts != SECTION_COUNTS:
        raise ValueError(f"{data.name}: unexpected section counts {counts}")
    if len(set(zip(data.sections, data.barcodes))) != n:
        raise ValueError(f"{data.name}: duplicate section/barcode keys")
    for label in LABEL_KEYS:
        valid = valid_truth(data.truth[label])
        if valid.sum() < 2 or len(np.unique(data.truth[label][valid])) < 2:
            raise ValueError(f"{data.name}: invalid ground truth {label}")
    if data.output_root.exists() and not allow_existing_output:
        raise FileExistsError(f"refusing to overwrite {data.output_root}")
    missing = [
        str(path)
        for path in data.source_paths + list(data.shared_sources.values())
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{data.name}: missing sources {missing}")


def fitted_space(raw: np.ndarray, scheme: str):
    if scheme == "raw_embedding":
        return raw, None
    scaler = StandardScaler()
    return scaler.fit_transform(raw), scaler


def safe_asw(space: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(space, labels))


def internal(space: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    asw = safe_asw(space, labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "calinski_harabasz": float(calinski_harabasz_score(space, labels)),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
    }


def external(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "homogeneity": float(homogeneity_score(truth, labels)),
        "completeness": float(completeness_score(truth, labels)),
        "v_measure": float(v_measure_score(truth, labels)),
    }


def spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=SPATIAL_NEIGHBOR_K + 1).fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def save_labels(
    path: Path, data: MethodData, mask: np.ndarray, labels: np.ndarray
) -> None:
    frame: dict[str, Any] = {
        "section": data.sections[mask],
        "obs_name": data.barcodes[mask],
        "cluster": labels.astype(int),
        "x": data.coords[mask, 0],
        "y": data.coords[mask, 1],
    }
    for label in LABEL_KEYS:
        frame[label] = data.truth[label][mask]
    pd.DataFrame(frame).to_csv(path, index=False)


def plot_spatial(
    path: Path, coords: np.ndarray, labels: np.ndarray, title: str
) -> None:
    categories, codes = np.unique(labels, return_inverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=POINT_SIZE,
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
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def scaler_payload(scaler: StandardScaler, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": scaler.mean_,
        f"{prefix}_scale": scaler.scale_,
        f"{prefix}_var": scaler.var_,
    }


def label_metrics(
    data: MethodData,
    space: np.ndarray,
    labels: np.ndarray,
    base_mask: np.ndarray,
    mode: str,
    scope: str,
    k: int,
    label_asw_cache: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    rows = []
    base_indices = np.flatnonzero(base_mask)
    for label_key in LABEL_KEYS:
        values = data.truth[label_key][base_mask]
        valid = valid_truth(values)
        if valid.sum() < 2 or len(np.unique(values[valid])) < 2:
            continue
        cache_key = (scope, label_key)
        if cache_key not in label_asw_cache:
            label_asw_cache[cache_key] = safe_asw(space[valid], values[valid])
        truth = values[valid].astype(str)
        predicted = labels[valid]
        raw_asw = label_asw_cache[cache_key]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_key,
                "n_obs_labeled": int(valid.sum()),
                "n_label_classes": len(np.unique(truth)),
                "k": k,
                **external(truth, predicted),
                "label_asw": raw_asw,
                "label_asw_scaled": (raw_asw + 1.0) / 2.0,
                "label_rows_in_full_order": ",".join(
                    map(str, base_indices[valid][:5])
                ),
            }
        )
    return rows


def run_scheme(data: MethodData, scheme: str) -> None:
    out = data.output_root / scheme
    cluster_root = out / "clustering"
    metrics_root = out / "metrics"
    cluster_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scalers: dict[str, np.ndarray] = {}
    label_asw_cache: dict[tuple[str, str], float] = {}

    joint_space, joint_scaler = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scalers.update(scaler_payload(joint_scaler, "joint"))
    section_asw = safe_asw(joint_space, data.sections)
    all_mask = np.ones(len(data.embedding), dtype=bool)

    for k in METRIC_KS:
        labels_all = KMeans(
            k, random_state=SEED, n_init=N_INIT, max_iter=MAX_ITER
        ).fit_predict(joint_space)
        kdir = cluster_root / f"joint_k{k}"
        kdir.mkdir()
        all_path = kdir / "labels_all.csv"
        save_labels(all_path, data, all_mask, labels_all)
        internal_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": len(labels_all),
                "embedding_dim": data.embedding.shape[1],
                **internal(joint_space, labels_all),
                "section_ari": adjusted_rand_score(data.sections, labels_all),
                "section_nmi": normalized_mutual_info_score(
                    data.sections, labels_all
                ),
                "section_asw": section_asw,
                "metric_space": scheme,
                "labels_path": str(all_path),
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
            label_path = kdir / f"labels_{section}.csv"
            save_labels(label_path, data, mask, labels)
            if k in PLOT_KS:
                plot_spatial(
                    kdir / f"spatial_{section}.png",
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
                    "labels_path": str(label_path),
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
        pd.DataFrame(count_rows).to_csv(kdir / "cluster_counts.csv", index=False)

    for k in PLOT_KS:
        kdir = cluster_root / f"independent_k{k}"
        kdir.mkdir()
        count_rows = []
        for section in SECTIONS:
            mask = data.sections == section
            space, scaler = fitted_space(data.embedding[mask], scheme)
            if scaler is not None:
                scalers.update(scaler_payload(scaler, section))
            labels = KMeans(
                k, random_state=SEED, n_init=N_INIT, max_iter=MAX_ITER
            ).fit_predict(space)
            label_path = kdir / f"labels_{section}.csv"
            save_labels(label_path, data, mask, labels)
            plot_spatial(
                kdir / f"spatial_{section}.png",
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
                    **internal(space, labels),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "metric_space": scheme,
                    "labels_path": str(label_path),
                }
            )
            external_rows.extend(
                label_metrics(
                    data,
                    space,
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
                    "labels_path": str(label_path),
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
        pd.DataFrame(count_rows).to_csv(kdir / "cluster_counts.csv", index=False)

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
    if scalers:
        np.savez_compressed(out / "scaler_parameters.npz", **scalers)

    config = {
        "dataset": "MouseBrain",
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
            "none" if scheme == "raw_embedding" else "all_7866_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "kmeans_type": "sklearn.cluster.KMeans",
        "metric_k_values": METRIC_KS,
        "plot_and_independent_k_values": PLOT_KS,
        "retained_clustering_k_values": PLOT_KS,
        "deleted_clustering_k_values": PRUNED_CLUSTERING_KS,
        "metrics_k_values_without_saved_label_directories":
        PRUNED_CLUSTERING_KS,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "n_obs": len(data.embedding),
        "section_counts": SECTION_COUNTS,
        "embedding_dim": data.embedding.shape[1],
        "asw_sample_size": 0,
        "asw_sample_rule": "full_valid_n_obs",
        "ch_dbi_sample_size": 0,
        "ch_dbi_sample_rule": "full_n_obs",
        "label_keys": LABEL_KEYS,
        "label_asw_rule": "full_nonmissing_labels_in_same_metric_space",
        "label_valid_counts": {
            label: int(valid_truth(data.truth[label]).sum())
            for label in LABEL_KEYS
        },
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "fit_separately_within_each_section",
        "spatial_graph_n_obs": SECTION_COUNTS,
        "plots_and_metrics_reuse_same_label_files": False,
        "retained_k_plots_and_metrics_reuse_same_label_files": True,
        "source_data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in data.source_paths
        ],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    write_scheme_summary(data, scheme)


def write_scheme_summary(data: MethodData, scheme: str) -> None:
    (data.output_root / scheme / "SUMMARY.md").write_text(
        f"# {data.name}: {scheme}\n\n"
        f"- spots: 7866; dimensions: {data.embedding.shape[1]}\n"
        f"- KMeans: sklearn exact, seed={SEED}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}\n"
        f"- joint metric K: {METRIC_KS}; independent metric K: {PLOT_KS}\n"
        f"- retained clustering directories/labels/plots K: {PLOT_KS}\n"
        f"- deleted joint clustering directory K: {PRUNED_CLUSTERING_KS}; "
        "metric CSV rows remain, so their historical labels_path values no "
        "longer resolve\n"
        "- external labels: five biological annotations plus group, where "
        "group is the aligned s1/s2/s3 section-source diagnostic\n"
        "- internal and label ASW: full valid samples in KMeans metric space\n"
        f"- spatial continuity: exact {SPATIAL_NEIGHBOR_K}-NN per section\n"
        "- for retained K, figures and metrics reuse the saved label CSV "
        "files\n"
    )


def prune_unselected_clustering_directories(data: MethodData) -> int:
    deleted = 0
    expected_remaining = {
        f"{mode}_k{k}" for mode in ("joint", "independent") for k in PLOT_KS
    }
    for scheme in ("raw_embedding", "standardized_embedding"):
        output = data.output_root / scheme
        cluster_root = output / "clustering"
        if not cluster_root.is_dir():
            raise FileNotFoundError(
                f"{data.name} {scheme}: missing {cluster_root}"
            )

        for directory in sorted(cluster_root.iterdir()):
            if not directory.is_dir():
                continue
            parts = directory.name.rsplit("_k", 1)
            if (
                len(parts) != 2
                or parts[0] not in {"joint", "independent"}
                or not parts[1].isdigit()
            ):
                continue
            k = int(parts[1])
            if k not in METRIC_KS:
                raise ValueError(
                    f"{data.name} {scheme}: unexpected clustering K={k}"
                )
            if k in PLOT_KS:
                continue
            if directory.resolve().parent != cluster_root.resolve():
                raise ValueError(f"unsafe deletion target {directory}")
            print(f"[{data.name}] delete {directory}", flush=True)
            shutil.rmtree(directory)
            deleted += 1

        remaining = {
            directory.name
            for directory in cluster_root.iterdir()
            if directory.is_dir()
            and directory.name.rsplit("_k", 1)[0]
            in {"joint", "independent"}
            and directory.name.rsplit("_k", 1)[1].isdigit()
        }
        if remaining != expected_remaining:
            raise ValueError(
                f"{data.name} {scheme}: unexpected remaining directories "
                f"{sorted(remaining)}"
            )

        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "retained_clustering_k_values": PLOT_KS,
                "deleted_clustering_k_values": PRUNED_CLUSTERING_KS,
                "metrics_k_values_without_saved_label_directories":
                PRUNED_CLUSTERING_KS,
                "plots_and_metrics_reuse_same_label_files": False,
                "retained_k_plots_and_metrics_reuse_same_label_files": True,
                "clustering_directories_pruned_at": (
                    datetime.now().astimezone().isoformat()
                ),
            }
        )
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        write_scheme_summary(data, scheme)
    return deleted


def labels_vector_sha256(labels: np.ndarray) -> str:
    values = np.ascontiguousarray(labels, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def add_group_diagnostics(data: MethodData) -> None:
    all_mask = np.ones(len(data.embedding), dtype=bool)
    group_values = data.truth[GROUP_LABEL_KEY].astype(str)
    if not np.array_equal(group_values, data.sections.astype(str)):
        raise ValueError(f"{data.name}: group is not identical to section")

    for scheme in ("raw_embedding", "standardized_embedding"):
        output = data.output_root / scheme
        metrics_root = output / "metrics"
        space, _ = fitted_space(data.embedding, scheme)
        group_asw = safe_asw(space, group_values)
        internal_frame = pd.read_csv(
            metrics_root / "clustering_metrics.csv"
        )
        joint_internal = (
            internal_frame[internal_frame["mode"] == "joint"]
            .set_index("k")
            .sort_index()
        )
        if set(joint_internal.index.astype(int)) != set(METRIC_KS):
            raise ValueError(
                f"{data.name} {scheme}: incomplete joint metric K values"
            )
        if not np.allclose(
            joint_internal["section_asw"].to_numpy(float),
            group_asw,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{data.name} {scheme}: group ASW does not match section ASW"
            )

        group_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        for k in METRIC_KS:
            labels_path = (
                output / "clustering" / f"joint_k{k}" / "labels_all.csv"
            )
            if labels_path.exists():
                saved = pd.read_csv(labels_path)
                expected_order = pd.DataFrame(
                    {
                        "section": data.sections.astype(str),
                        "obs_name": data.barcodes.astype(str),
                    }
                )
                if not saved[["section", "obs_name"]].astype(str).equals(
                    expected_order
                ):
                    raise ValueError(
                        f"{data.name} {scheme} K={k}: labels row mismatch"
                    )
                labels = saved["cluster"].to_numpy(int)
                labels_source = "retained_labels_csv"
                labels_file_sha256 = sha256(labels_path)
                labels_path_value = str(labels_path)
            else:
                labels = KMeans(
                    k,
                    random_state=SEED,
                    n_init=N_INIT,
                    max_iter=MAX_ITER,
                ).fit_predict(space)
                labels_source = (
                    "deterministic_kmeans_recomputed_for_group_diagnostic"
                )
                labels_file_sha256 = ""
                labels_path_value = ""

            group_metrics = external(group_values, labels)
            expected = joint_internal.loc[k]
            if not np.isclose(
                group_metrics["ari"],
                float(expected["section_ari"]),
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                group_metrics["nmi"],
                float(expected["section_nmi"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{data.name} {scheme} K={k}: recomputed group "
                    "ARI/NMI does not match existing section diagnostics"
                )

            common = {
                "mode": "joint",
                "scope": "combined",
                "label": GROUP_LABEL_KEY,
                "n_obs_labeled": len(group_values),
                "n_label_classes": len(np.unique(group_values)),
                "k": k,
                **group_metrics,
                "label_asw": group_asw,
                "label_asw_scaled": (group_asw + 1.0) / 2.0,
                "label_rows_in_full_order": "0,1,2,3,4",
            }
            group_rows.append(common)
            provenance_rows.append(
                {
                    **common,
                    "cluster_asw": float(expected["cluster_asw"]),
                    "cluster_asw_scaled": float(
                        expected["cluster_asw_scaled"]
                    ),
                    "group_definition": "aligned section vector: s1/s2/s3",
                    "labels_source": labels_source,
                    "labels_path": labels_path_value,
                    "labels_file_sha256": labels_file_sha256,
                    "labels_vector_sha256": labels_vector_sha256(labels),
                    "seed": SEED,
                    "n_init": N_INIT,
                    "max_iter": MAX_ITER,
                    "metric_space": scheme,
                }
            )

        external_path = metrics_root / "clustering_metrics_by_label.csv"
        external_frame = pd.read_csv(external_path)
        external_frame = external_frame[
            external_frame["label"] != GROUP_LABEL_KEY
        ]
        external_frame = pd.concat(
            [external_frame, pd.DataFrame(group_rows)],
            ignore_index=True,
        )
        external_frame.to_csv(external_path, index=False)
        joint_external = external_frame[external_frame["mode"] == "joint"]
        best = joint_external.loc[
            joint_external.groupby("label")["ari"].idxmax()
        ].sort_values("label")
        best.to_csv(
            metrics_root / "best_clustering_metrics_by_label.csv",
            index=False,
        )
        pd.DataFrame(provenance_rows).to_csv(
            metrics_root / "group_diagnostic_metrics.csv",
            index=False,
        )

        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "label_keys": LABEL_KEYS,
                "label_valid_counts": {
                    label: int(valid_truth(data.truth[label]).sum())
                    for label in LABEL_KEYS
                },
                "group_definition": "aligned section vector: s1/s2/s3",
                "group_diagnostic_k_values": METRIC_KS,
                "group_diagnostic_reuses_retained_labels_for_k":
                PLOT_KS,
                "group_diagnostic_recomputes_labels_without_saving_for_k":
                PRUNED_CLUSTERING_KS,
                "group_diagnostics_added_at": (
                    datetime.now().astimezone().isoformat()
                ),
            }
        )
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        write_scheme_summary(data, scheme)

    raw_external = pd.read_csv(
        data.output_root
        / "raw_embedding"
        / "metrics"
        / "clustering_metrics_by_label.csv"
    )
    standardized_external = pd.read_csv(
        data.output_root
        / "standardized_embedding"
        / "metrics"
        / "clustering_metrics_by_label.csv"
    )
    keys = [
        "mode",
        "scope",
        "label",
        "n_obs_labeled",
        "n_label_classes",
        "k",
    ]
    metrics = [
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "label_asw",
        "label_asw_scaled",
    ]
    merged = raw_external.merge(
        standardized_external,
        on=keys,
        suffixes=("_raw", "_standardized"),
        validate="one_to_one",
    )
    for metric in metrics:
        merged[f"{metric}_delta_standardized_minus_raw"] = (
            merged[f"{metric}_standardized"] - merged[f"{metric}_raw"]
        )
    comparison_root = data.output_root / "comparison_metrics"
    merged.to_csv(
        comparison_root / "clustering_metrics_by_label_deltas.csv",
        index=False,
    )

    raw_group = pd.read_csv(
        data.output_root
        / "raw_embedding"
        / "metrics"
        / "group_diagnostic_metrics.csv"
    )
    standardized_group = pd.read_csv(
        data.output_root
        / "standardized_embedding"
        / "metrics"
        / "group_diagnostic_metrics.csv"
    )
    group_keys = [
        "mode",
        "scope",
        "label",
        "n_obs_labeled",
        "n_label_classes",
        "k",
    ]
    group_metrics = [
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "label_asw",
        "label_asw_scaled",
        "cluster_asw",
        "cluster_asw_scaled",
    ]
    group_delta = raw_group.merge(
        standardized_group,
        on=group_keys,
        suffixes=("_raw", "_standardized"),
        validate="one_to_one",
    )
    for metric in group_metrics:
        group_delta[f"{metric}_delta_standardized_minus_raw"] = (
            group_delta[f"{metric}_standardized"]
            - group_delta[f"{metric}_raw"]
        )
    group_delta.to_csv(
        comparison_root / "group_diagnostic_metrics_deltas.csv",
        index=False,
    )

    best_rows = []
    for scheme, scheme_label in (
        ("raw_embedding", "Raw"),
        ("standardized_embedding", "Standardized"),
    ):
        frame = pd.read_csv(
            data.output_root
            / scheme
            / "metrics"
            / "group_diagnostic_metrics.csv"
        )
        row = frame.loc[frame["ari"].idxmax()].to_dict()
        row["preprocessing"] = scheme_label
        best_rows.append(row)
    pd.DataFrame(best_rows).to_csv(
        comparison_root / "group_best_diagnostic_comparison.csv",
        index=False,
    )


def copy_shared(data: MethodData) -> None:
    out = data.output_root / "shared_metrics"
    out.mkdir()
    rows = []
    for name, source in data.shared_sources.items():
        target = out / name
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
    pd.DataFrame(rows).to_csv(out / "source_manifest.csv", index=False)


def compare_schemes(data: MethodData) -> None:
    out = data.output_root / "comparison_metrics"
    out.mkdir()
    stability_rows = []
    for mode, ks in (("joint", METRIC_KS), ("independent", PLOT_KS)):
        scopes = ["combined"] if mode == "joint" else SECTIONS
        for k in ks:
            for scope in scopes:
                rel = (
                    Path("clustering") / f"joint_k{k}" / "labels_all.csv"
                    if mode == "joint"
                    else Path("clustering")
                    / f"independent_k{k}"
                    / f"labels_{scope}.csv"
                )
                raw = pd.read_csv(data.output_root / "raw_embedding" / rel)
                std = pd.read_csv(
                    data.output_root / "standardized_embedding" / rel
                )
                if not raw[["section", "obs_name"]].equals(
                    std[["section", "obs_name"]]
                ):
                    raise ValueError(f"{data.name}: row mismatch {mode} K={k}")
                stability_rows.append(
                    {
                        "mode": mode,
                        "scope": scope,
                        "k": k,
                        "n_obs": len(raw),
                        "raw_vs_standardized_ari": adjusted_rand_score(
                            raw["cluster"], std["cluster"]
                        ),
                        "raw_vs_standardized_nmi": normalized_mutual_info_score(
                            raw["cluster"], std["cluster"]
                        ),
                    }
                )
    pd.DataFrame(stability_rows).to_csv(
        out / "label_stability.csv", index=False
    )

    def delta_table(filename: str, keys: list[str], metrics: list[str]) -> None:
        raw = pd.read_csv(
            data.output_root / "raw_embedding" / "metrics" / filename
        )
        std = pd.read_csv(
            data.output_root / "standardized_embedding" / "metrics" / filename
        )
        merged = raw.merge(
            std,
            on=keys,
            suffixes=("_raw", "_standardized"),
            validate="one_to_one",
        )
        for metric in metrics:
            merged[f"{metric}_delta_standardized_minus_raw"] = (
                merged[f"{metric}_standardized"] - merged[f"{metric}_raw"]
            )
        merged.to_csv(out / f"{Path(filename).stem}_deltas.csv", index=False)

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
        "dataset": "MouseBrain",
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
        "joint_metric_k_values": METRIC_KS,
        "independent_metric_k_values": PLOT_KS,
        "retained_clustering_k_values": PLOT_KS,
        "deleted_clustering_k_values": PRUNED_CLUSTERING_KS,
        "metrics_k_values_without_saved_label_directories":
        PRUNED_CLUSTERING_KS,
        "external_label_keys": LABEL_KEYS,
        "group_definition": "aligned section vector: s1/s2/s3",
        "group_diagnostic_k_values": METRIC_KS,
        "group_diagnostic_outputs": {
            "raw": str(
                data.output_root
                / "raw_embedding"
                / "metrics"
                / "group_diagnostic_metrics.csv"
            ),
            "standardized": str(
                data.output_root
                / "standardized_embedding"
                / "metrics"
                / "group_diagnostic_metrics.csv"
            ),
            "deltas": str(
                data.output_root
                / "comparison_metrics"
                / "group_diagnostic_metrics_deltas.csv"
            ),
        },
        "source_data_dir": str(data.data_dir),
        "source_file_hashes": {
            str(path): sha256(path) for path in data.source_paths
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (data.output_root / "preprocessing_comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|")

    def display_width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            for char in value
        )

    def is_number(value: str) -> bool:
        if value in {"NA", "NaN", ""}:
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False

    header_values = [clean(value) for value in headers]
    row_values = [[clean(value) for value in row] for row in rows]
    if any(len(row) != len(header_values) for row in row_values):
        raise ValueError("Markdown table row width does not match its header.")
    numeric = [
        bool(row_values)
        and all(
            is_number(row[column])
            for row in row_values
            if row[column] not in {"NA", "NaN", ""}
        )
        for column in range(len(header_values))
    ]
    widths = [
        max(
            3,
            display_width(header_values[column]),
            max(
                (display_width(row[column]) for row in row_values),
                default=0,
            ),
        )
        for column in range(len(header_values))
    ]

    def padded(value: str, column: int) -> str:
        spaces = " " * (widths[column] - display_width(value))
        return spaces + value if numeric[column] else value + spaces

    lines = [
        "| "
        + " | ".join(
            padded(value, column)
            for column, value in enumerate(header_values)
        )
        + " |",
        "| "
        + " | ".join(
            ("-" * (widths[column] - 1) + ":")
            if numeric[column]
            else "-" * widths[column]
            for column in range(len(header_values))
        )
        + " |",
    ]
    lines += [
        "| "
        + " | ".join(
            padded(value, column) for column, value in enumerate(row)
        )
        + " |"
        for row in row_values
    ]
    return "\n".join(lines)


def f4(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def build_report(methods: list[MethodData], overwrite: bool = False) -> None:
    if REPORT_PATH.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    lines = [
        "# MouseBrain：Raw 与 Standardized embedding + KMeans 对比报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        "- 本报告仅引用本次新增 comparison 结果；旧分析目录和原综合报告未覆盖。",
        "- 两分支分别在自身 KMeans 输入空间计算距离型指标；跨方案变化应结合标签稳定性解释。",
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
                ["spot 数", "7866（s1=2384，s2=2820，s3=2662）"],
                ["raw 分支", "原始最终 embedding 直接输入 KMeans"],
                ["standardized 分支", "StandardScaler；joint 全体拟合，independent 每切片拟合"],
                ["KMeans", "sklearn exact KMeans"],
                ["joint 指标 K", "2–12"],
                ["independent 指标 K", "5、6、8、10"],
                ["保留的聚类文件夹/labels/图 K", "5、6、8、10"],
                ["随机参数", "seed=0，n_init=20，max_iter=300"],
                ["ASW/CH/DBI", "KMeans 同一输入空间，全量样本"],
                [
                    "外部指标",
                    "5类生物标签 + group(section来源)诊断；缺失标签逐标签剔除",
                ],
                ["Label ASW", "对应输入空间内全部非缺失标签 spot"],
                ["空间连续性", "每切片全点建图，exact 6-NN"],
                ["保留 K 的图与指标", "复用同一套已保存 label CSV"],
                ["batch 指标", "不依赖本次 KMeans 分支；复制并记录 SHA256"],
            ],
        ),
        "",
        "### 与旧 MouseBrain 结果的关系",
        "",
        "旧分析脚本/结果存在不同默认种子（MOFA+=1、COSIE=1、SpaMosaic=1234）以及 spa_mo_model 部分 ASW 使用 5,000 spot 抽样等口径差异。本次 raw 和 standardized 标签、外部指标与内部指标均按统一 seed=0 和全量样本重新生成，未复制旧聚类数值；报告中的聚类数值只来自本次新增目录。batch、MOFA+ R2 和 SpaMosaic 模态对齐属于本次 KMeans 预处理无关指标，才以带 SHA256 的方式复制到 `shared_metrics/`。",
        "",
        "## 3. 数据、标签与 embedding 对齐",
        "",
        md_table(
            [
                "方法",
                "维度",
                "s1",
                "s2",
                "s3",
                "RegionLoupe n",
                "celltype n",
                "group n",
            ],
            [
                [
                    m.name,
                    m.embedding.shape[1],
                    SECTION_COUNTS["s1"],
                    SECTION_COUNTS["s2"],
                    SECTION_COUNTS["s3"],
                    int(valid_truth(m.truth["RegionLoupe"]).sum()),
                    int(valid_truth(m.truth["celltype"]).sum()),
                    int(valid_truth(m.truth[GROUP_LABEL_KEY]).sum()),
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
        "## 4. 外部指标：Raw / Standardized 上下对齐对比",
        "",
        "同一方法、同一标签的 Raw 与 Standardized 行相邻，列定义完全一致；"
        "group定义为对齐后的section来源向量s1/s2/s3，属于切片依赖诊断。",
        "",
    ]
    best_rows = []
    for method in methods:
        best_by_scheme = {}
        for scheme in ["raw_embedding", "standardized_embedding"]:
            best_by_scheme[scheme] = pd.read_csv(
                method.output_root
                / scheme
                / "metrics"
                / "best_clustering_metrics_by_label.csv"
            ).set_index("label")
        for label in LABEL_KEYS:
            for scheme, scheme_label in [
                ("raw_embedding", "Raw"),
                ("standardized_embedding", "Standardized"),
            ]:
                row = best_by_scheme[scheme].loc[label]
                best_rows.append(
                    [
                        method.name,
                        label,
                        scheme_label,
                        int(row["k"]),
                        int(row["n_obs_labeled"]),
                        f4(row["ari"]),
                        f4(row["nmi"]),
                        f4(row["v_measure"]),
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
                "ARI",
                "NMI",
                "V-measure",
                "Label ASW scaled",
            ],
            best_rows,
        ),
        "",
        "best K 在每个预处理分支内按 joint ARI 独立选择；完整 K=2–12 "
        "外部指标见各分支 `metrics/clustering_metrics_by_label.csv`。",
        "",
        "### group完整诊断：按ARI选择best K",
        "",
        "下表补齐原综合报告中的group诊断。Raw与Standardized严格相邻；"
        "group反映section来源依赖，不应作为生物聚类准确率解释。",
        "",
    ]
    group_rows = []
    for method in methods:
        for scheme, scheme_label in [
            ("raw_embedding", "Raw"),
            ("standardized_embedding", "Standardized"),
        ]:
            group = pd.read_csv(
                method.output_root
                / scheme
                / "metrics"
                / "group_diagnostic_metrics.csv"
            )
            row = group.loc[group["ari"].idxmax()]
            group_rows.append(
                [
                    method.name,
                    scheme_label,
                    int(row["k"]),
                    int(row["n_obs_labeled"]),
                    f4(row["ari"]),
                    f4(row["nmi"]),
                    f4(row["homogeneity"]),
                    f4(row["completeness"]),
                    f4(row["v_measure"]),
                    f4(row["cluster_asw"]),
                    f4(row["cluster_asw_scaled"]),
                    f4(row["label_asw"]),
                    f4(row["label_asw_scaled"]),
                ]
            )
    lines += [
        md_table(
            [
                "方法",
                "预处理",
                "best K",
                "有效 n",
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
            group_rows,
        ),
        "",
        "group完整K=2–12结果、labels来源/哈希及KMeans参数见各分支 "
        "`metrics/group_diagnostic_metrics.csv`；逐K的Standardized-minus-Raw"
        "差值见 `comparison_metrics/group_diagnostic_metrics_deltas.csv`。",
        "",
        "## 5. 内部、切片诊断与空间指标：Raw / Standardized 上下对齐对比",
        "",
        "同一方法、同一 K 的 Raw 与 Standardized 行相邻。",
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
                joint = metrics[metrics["mode"] == "joint"].set_index("k")
                sj = spatial[spatial["mode"] == "joint"].set_index("k")
                row = joint.loc[k]
                metric_rows.append(
                    [
                        method.name,
                        k,
                        scheme_label,
                        f4(row["cluster_asw_scaled"]),
                        f4(row["calinski_harabasz"]),
                        f4(row["davies_bouldin"]),
                        f4(row["section_ari"]),
                        f4(row["section_nmi"]),
                        f4(sj.loc[k, "mean_spatial_neighbor_agreement"]),
                    ]
                )
    lines += [
        md_table(
            [
                "方法",
                "K",
                "预处理",
                "Cluster ASW scaled",
                "CH",
                "DBI",
                "section ARI",
                "section NMI",
                "joint spatial",
            ],
            metric_rows,
        ),
        "",
        "各分支 `metrics/` 保留完整 joint K=2–12 以及 independent "
        "K=5/6/8/10 指标；`clustering/` 中的文件夹、全量 labels 与空间图"
        "仅保留 K=5/6/8/10。其他 joint K 的指标 CSV 行仍保留，但其历史 "
        "`labels_path` 已随对应文件夹删除而失效。",
    ]

    lines += ["", "## 6. 两种预处理的标签稳定性", ""]
    stability_rows = []
    for method in methods:
        frame = pd.read_csv(
            method.output_root / "comparison_metrics" / "label_stability.csv"
        )
        frame = frame[
            (frame["mode"] == "joint") & frame["k"].isin(PLOT_KS)
        ]
        for _, row in frame.iterrows():
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
        "完整joint/independent标签稳定性、内部/外部指标差值及group专用"
        "逐K差值分别见各方法 `comparison_metrics/`。",
        "",
        "## 7. 共享指标",
        "",
    ]
    batch_rows = []
    for method in methods:
        batch = pd.read_csv(
            method.output_root
            / "shared_metrics"
            / "batch_correction_metrics.csv"
        ).iloc[0]
        batch_rows.append(
            [
                method.name,
                int(batch.get("n_used", batch.get("n_obs", 7866))),
                f4(batch.get("bASW", batch.get("basw_score_abs_mean"))),
                f4(batch.get("bLISI", batch.get("blisi_normalized"))),
                f4(batch.get("kBET", batch.get("kbet_acceptance_rate"))),
                f4(batch.get("PCR_score")),
            ]
        )
    lines += [
        md_table(
            ["方法", "n used", "bASW", "bLISI", "kBET", "PCR score"],
            batch_rows,
        ),
        "",
        "MOFA+ R2 与 SpaMosaic 模态对齐也保存在各自 `shared_metrics/`；复制来源及 SHA256 见 `source_manifest.csv`。",
        "",
        "## 8. 结论与使用建议",
        "",
        "1. raw 与 standardized 均作为完整、可复核的一等结果保存，不仅记录在报告中。",
        "2. 正式跨方法排名必须固定同一预处理分支；建议 standardized 作为主分析、raw 作为敏感性分析。",
        "3. 同时报告 raw-vs-standardized ARI/NMI，可区分尺度变化造成的指标变化与实际聚类重分配。",
        "4. 生物标签外部指标使用逐标签非缺失样本；group及section "
        "ARI/NMI仅是切片来源依赖诊断，不是生物准确率。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate only the report from existing comparison results.",
    )
    maintenance.add_argument(
        "--prune-unselected-clustering",
        action="store_true",
        help=(
            "Delete clustering directories outside K=5/6/8/10; retain "
            "existing metric CSV files."
        ),
    )
    maintenance.add_argument(
        "--add-group-diagnostics",
        action="store_true",
        help=(
            "Add full joint K=2-12 group/section external diagnostics to "
            "both preprocessing branches without restoring pruned "
            "clustering directories."
        ),
    )
    args = parser.parse_args()
    methods = [load_spa(), load_mofa(), load_cosie(), load_spamosaic()]
    if args.report_only:
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    if args.prune_unselected_clustering:
        deleted = 0
        for method in methods:
            validate(method, allow_existing_output=True)
            deleted += prune_unselected_clustering_directories(method)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(
            f"report: {REPORT_PATH}; "
            f"deleted_clustering_directories={deleted}",
            flush=True,
        )
        return
    if args.add_group_diagnostics:
        for method in methods:
            validate(method, allow_existing_output=True)
            print(f"[{method.name}] group diagnostics", flush=True)
            add_group_diagnostics(method)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
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
        add_group_diagnostics(method)
        prune_unselected_clustering_directories(method)
        write_manifest(method)
    build_report(methods)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
