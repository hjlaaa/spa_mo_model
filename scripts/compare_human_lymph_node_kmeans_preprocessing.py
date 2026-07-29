#!/usr/bin/env python3
"""Create independent raw-vs-standardized KMeans results for Human_Lymph_Node.

The original training and analysis outputs are read-only inputs.  Every method is
aligned against its own data copy and receives a new comparison root.  Metric
tables retain joint K=2-12, while clustering directories, saved labels, and
figures retain K=5/8/10/12.
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
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SECTIONS = ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]
METRIC_KS = list(range(2, 13))
PLOT_KS = [5, 8, 10, 12]
PRUNED_CLUSTERING_KS = [k for k in METRIC_KS if k not in PLOT_KS]
SEED = 0
N_INIT = 20
MAX_ITER = 300
SPATIAL_NEIGHBOR_K = 6
POINT_SIZE = 10.0
DPI = 220
METHOD_ORDER = ["spa_mo_model", "MOFA+", "COSIE", "SpaMosaic"]
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/"
    "human_lymph_node_preprocessing_comparison_report.md"
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            pos = source.get_indexer(wanted)
            if (pos < 0).any():
                raise ValueError(
                    f"{rna_path}: {int((pos < 0).sum())} barcodes failed alignment"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[pos, :2]
        finally:
            rna.file.close()
    return coords


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/human_lymph_node/"
        "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/Human_Lymph_Node")
    arrays, sections, barcodes, sources = [], [], [], []
    for section in SECTIONS:
        emb_path = run / f"final_embeddings_{section}.npy"
        idx_path = run / f"selected_spot_indices_{section}.npy"
        spatial_path = run / f"spatial_{section}.npy"
        emb = np.load(emb_path)
        idx = np.load(idx_path).astype(int)
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[idx]
            own_coords = np.asarray(rna.obsm["spatial"])[idx, :2]
        finally:
            rna.file.close()
        saved_coords = np.load(spatial_path)[:, :2]
        if emb.shape[0] != len(idx) or not np.allclose(saved_coords, own_coords):
            raise ValueError(f"spa_mo_model {section}: saved rows/spatial do not align")
        arrays.append(emb)
        sections.extend([section] * len(idx))
        barcodes.extend(names.tolist())
        sources.extend([emb_path, idx_path, spatial_path, rna_path])
    section_arr = np.asarray(sections, dtype=str)
    barcode_arr = np.asarray(barcodes, dtype=str)
    old = run / "clustering_analysis"
    return MethodData(
        "spa_mo_model",
        np.vstack(arrays),
        section_arr,
        barcode_arr,
        aligned_coords(data_dir, section_arr, barcode_arr),
        data_dir,
        sources,
        Path(
            "/home/hujinlan/spa_mo_model/results/"
            "human_lymph_node_preprocessing_comparison"
        ),
        {"batch_correction_metrics.csv": old / "batch_correction_metrics.csv"},
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "human_lymph_node_mofa_hvg2000_k10_iter1000"
    )
    table_path = old / "tables" / "factors_with_metadata_and_coordinates.csv"
    table = pd.read_csv(table_path)
    factor_cols = [column for column in table if column.startswith("Factor")]
    sections = table["section"].astype(str).to_numpy()
    barcodes = table["original_barcode"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/Human_Lymph_Node")
    shared = {
        "batch_correction_metrics.csv": old
        / "metrics"
        / "batch_correction_metrics.csv",
        "input_audit.csv": old / "input_audit.csv",
    }
    r2 = old / "tables" / "r2_total_by_view_group.csv"
    if r2.exists():
        shared["r2_total_by_view_group.csv"] = r2
    return MethodData(
        "MOFA+",
        table[factor_cols].to_numpy(float),
        sections,
        barcodes,
        aligned_coords(data_dir, sections, barcodes),
        data_dir,
        [table_path],
        Path(
            "/home/hujinlan/mofa+/analysis/"
            "human_lymph_node_preprocessing_comparison"
        ),
        shared,
    )


def load_cosie() -> MethodData:
    old = Path(
        "/home/hujinlan/cosie_runs/"
        "human_lymph_node_cosie_rna_adt_full/analysis"
    )
    table_path = old / "tables" / "embeddings_with_metadata.csv"
    table = pd.read_csv(table_path)
    emb_cols = [column for column in table if column.startswith("COSIE")]
    sections = table["section"].astype(str).to_numpy()
    barcode_col = "obs_name.1" if "obs_name.1" in table else "obs_name"
    barcodes = table[barcode_col].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/cosie/data/Human_Lymph_Node")
    return MethodData(
        "COSIE",
        table[emb_cols].to_numpy(float),
        sections,
        barcodes,
        aligned_coords(data_dir, sections, barcodes),
        data_dir,
        [table_path],
        Path(
            "/home/hujinlan/cosie_runs/"
            "human_lymph_node_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv"
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/human_lymph_node_spamosaic/"
        "human_lymph_node_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs["section"].astype(str).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path(
        "/home/hujinlan/SpaMosaic-dev/demo/data/Human_Lymph_Node"
    )
    old = Path(
        "/home/hujinlan/SpaMosaic-dev/analysis/"
        "human_lymph_node_spamosaic"
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
            "human_lymph_node_preprocessing_comparison"
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
    data: MethodData, allow_existing_output: bool = False
) -> None:
    n = data.embedding.shape[0]
    if n != 6843 or len(data.sections) != n or len(data.barcodes) != n:
        raise ValueError(f"{data.name}: expected/aligned 6843 rows, got {n}")
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f"{data.name}: non-finite embedding or coordinates")
    counts = dict(pd.Series(data.sections).value_counts())
    if counts != {"Human_Lymph_Node_A1": 3484, "Human_Lymph_Node_D1": 3359}:
        raise ValueError(f"{data.name}: unexpected section counts {counts}")
    if len(set(zip(data.sections, data.barcodes))) != n:
        raise ValueError(f"{data.name}: duplicate section/barcode keys")
    if data.output_root.exists() and not allow_existing_output:
        raise FileExistsError(f"refusing to overwrite {data.output_root}")
    missing = [str(path) for path in data.shared_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{data.name}: missing shared sources {missing}")


def metric_space(raw: np.ndarray, scheme: str):
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


def spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=SPATIAL_NEIGHBOR_K + 1).fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


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


def run_scheme(data: MethodData, scheme: str) -> None:
    out = data.output_root / scheme
    cluster_root = out / "clustering"
    metrics_root = out / "metrics"
    cluster_root.mkdir(parents=True)
    metrics_root.mkdir()
    metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scalers: dict[str, np.ndarray] = {}

    joint_space, joint_scaler = metric_space(data.embedding, scheme)
    if joint_scaler is not None:
        scalers.update(scaler_payload(joint_scaler, "joint"))
    section_asw = safe_asw(joint_space, data.sections)

    for k in METRIC_KS:
        labels_all = KMeans(
            k, random_state=SEED, n_init=N_INIT, max_iter=MAX_ITER
        ).fit_predict(joint_space)
        kdir = cluster_root / f"joint_k{k}"
        kdir.mkdir()
        all_mask = np.ones(len(labels_all), dtype=bool)
        save_labels(kdir / "labels_all.csv", data, all_mask, labels_all)
        metric_rows.append(
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
                "labels_path": str(kdir / "labels_all.csv"),
            }
        )
        count_rows = []
        for section in SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            save_labels(kdir / f"labels_{section}.csv", data, mask, labels)
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
                    "labels_path": str(kdir / f"labels_{section}.csv"),
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
            space, scaler = metric_space(data.embedding[mask], scheme)
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
            metric_rows.append(
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

    metrics = pd.DataFrame(metric_rows)
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
    if scalers:
        np.savez_compressed(out / "scaler_parameters.npz", **scalers)

    config = {
        "dataset": "Human_Lymph_Node",
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
            "none" if scheme == "raw_embedding" else "all_6843_spots"
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
        "section_counts": {
            key: int(value)
            for key, value in pd.Series(data.sections).value_counts().items()
        },
        "embedding_dim": data.embedding.shape[1],
        "asw_sample_size": 0,
        "asw_sample_rule": "full_n_obs",
        "ch_dbi_sample_size": 0,
        "ch_dbi_sample_rule": "full_n_obs",
        "label_asw": "not_applicable_no_ground_truth_labels",
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "fit_separately_within_each_section",
        "spatial_graph_n_obs": {
            key: int(value)
            for key, value in pd.Series(data.sections).value_counts().items()
        },
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
    summary = (
        f"# {data.name}: {scheme}\n\n"
        f"- spots: {len(data.embedding)}; dimensions: {data.embedding.shape[1]}\n"
        f"- KMeans: sklearn exact, seed={SEED}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}\n"
        f"- joint metric K: {METRIC_KS}; independent metric K: {PLOT_KS}\n"
        f"- retained clustering directories/labels/plots K: {PLOT_KS}\n"
        f"- deleted joint clustering directory K: {PRUNED_CLUSTERING_KS}; "
        "metric CSV rows remain, so their historical labels_path values no "
        "longer resolve\n"
        f"- metrics: full sample in the same space used by KMeans\n"
        f"- spatial continuity: exact {SPATIAL_NEIGHBOR_K}-NN within each section\n"
        "- for retained K, labels are shared by figures and metrics\n"
    )
    (data.output_root / scheme / "SUMMARY.md").write_text(summary)


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
    rows = []
    for mode, ks in (("joint", METRIC_KS), ("independent", PLOT_KS)):
        scopes = ["combined"] if mode == "joint" else SECTIONS
        for k in ks:
            for scope in scopes:
                if mode == "joint":
                    rel = Path("clustering") / f"joint_k{k}" / "labels_all.csv"
                else:
                    rel = (
                        Path("clustering")
                        / f"independent_k{k}"
                        / f"labels_{scope}.csv"
                    )
                raw = pd.read_csv(data.output_root / "raw_embedding" / rel)
                std = pd.read_csv(
                    data.output_root / "standardized_embedding" / rel
                )
                key = ["section", "obs_name"]
                if not raw[key].equals(std[key]):
                    raise ValueError(f"{data.name}: label row mismatch {mode} K={k}")
                rows.append(
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
    pd.DataFrame(rows).to_csv(out / "label_stability.csv", index=False)

    metric_files = {}
    for scheme in ("raw_embedding", "standardized_embedding"):
        frame = pd.read_csv(
            data.output_root / scheme / "metrics" / "clustering_metrics.csv"
        )
        metric_files[scheme] = frame
    keys = ["mode", "scope", "k", "n_obs", "embedding_dim"]
    merged = metric_files["raw_embedding"].merge(
        metric_files["standardized_embedding"],
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
            merged[f"{metric}_standardized"] - merged[f"{metric}_raw"]
        )
    merged.to_csv(out / "metric_deltas.csv", index=False)


def write_manifest(data: MethodData) -> None:
    payload = {
        "dataset": "Human_Lymph_Node",
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
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_report(methods: list[MethodData], overwrite: bool = False) -> None:
    if REPORT_PATH.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    lines = [
        "# Human_Lymph_Node：Raw 与 Standardized embedding + KMeans 对比报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        "- 本报告只引用本次新增的 comparison 结果；原分析目录未覆盖。",
        "- 两套方案分别在各自 KMeans 输入空间计算 ASW、CH、DBI；因此适合在同一预处理方案内跨方法比较，跨方案数值变化应结合标签稳定性解释。",
        "",
        "## 1. 新结果目录",
        "",
        md_table(
            ["方法", "comparison root", "方法自有数据副本"],
            [[m.name, m.output_root, m.data_dir] for m in methods],
        ),
        "",
        "每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和 `preprocessing_comparison_manifest.json`。",
        "",
        "## 2. 统一标准",
        "",
        md_table(
            ["项目", "统一值"],
            [
                ["spot / barcode", "各方法自有数据副本；section + original barcode 精确对齐"],
                ["spot 数", "6843（A1=3484，D1=3359）"],
                ["embedding", "各方法当前最终 embedding"],
                ["raw 分支", "原始最终 embedding 直接输入 KMeans"],
                ["standardized 分支", "StandardScaler；joint 全体拟合，independent 每切片拟合"],
                ["KMeans", "sklearn exact KMeans"],
                ["joint 指标 K", "2–12"],
                ["independent 指标 K", "5、8、10、12"],
                ["保留的聚类文件夹/labels/图 K", "5、8、10、12"],
                ["随机参数", "seed=0，n_init=20，max_iter=300"],
                ["ASW / CH / DBI", "KMeans 同一输入空间，全量样本"],
                ["Label ASW", "无真实细胞/区域标签，不计算"],
                ["空间连续性", "每切片全点建图，exact 6-NN"],
                ["保留 K 的图与指标", "复用同一套已保存 label CSV"],
                ["batch 指标", "预处理无关；复制原最终结果到 shared_metrics 并记录 SHA256"],
            ],
        ),
        "",
        "## 3. 数据与 embedding 对齐",
        "",
        md_table(
            ["方法", "维度", "A1", "D1"],
            [
                [
                    m.name,
                    m.embedding.shape[1],
                    int((m.sections == SECTIONS[0]).sum()),
                    int((m.sections == SECTIONS[1]).sum()),
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
        "同一方法、同一 K 的 Raw 与 Standardized 行相邻，所有列采用完全相同的定义。",
        "",
    ]
    rows = []
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
                rows.append(
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
                        f4(sj.loc[k, "mean_spatial_neighbor_agreement"]),
                    ]
                )
    lines += [
        md_table(
            [
                "方法",
                "K",
                "预处理",
                "ASW n",
                "ASW raw",
                "ASW scaled",
                "CH/DBI n",
                "CH",
                "DBI",
                "section ARI",
                "section NMI",
                "section ASW n",
                "section ASW raw",
                "joint spatial",
            ],
            rows,
        ),
        "",
        "各分支 `metrics/` 保留完整 joint K=2–12 以及 independent "
        "K=5/8/10/12 指标；`clustering/` 中的文件夹、全量 labels 与空间图"
        "仅保留 K=5/8/10/12。其他 joint K 的指标 CSV 行仍保留，但其历史 "
        "`labels_path` 已随对应文件夹删除而失效。",
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
        "同一方法、同一K、同一section的Raw与Standardized行严格相邻；ASW、CH和DBI均使用该section全量spot。",
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
                "ASW raw",
                "ASW scaled",
                "CH/DBI n",
                "CH",
                "DBI",
            ],
            independent_rows,
        ),
        "",
        "### 4.3 Joint与Independent空间连续性",
        "",
        "每个值为A1和D1两个section全量spot空间近邻同簇比例的算术平均；Raw与Standardized行严格相邻。",
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
    ]

    lines += ["", "## 5. 两种预处理的标签稳定性", ""]
    rows = []
    for method in methods:
        stability = pd.read_csv(
            method.output_root / "comparison_metrics" / "label_stability.csv"
        )
        stability = stability[
            (stability["mode"] == "joint")
            & stability["k"].isin(PLOT_KS)
        ]
        for _, row in stability.iterrows():
            rows.append(
                [
                    method.name,
                    int(row["k"]),
                    f4(row["raw_vs_standardized_ari"]),
                    f4(row["raw_vs_standardized_nmi"]),
                ]
            )
    lines += [
        md_table(["方法", "K", "raw vs standardized ARI", "NMI"], rows),
        "",
        "完整 joint K=2–12 及 independent K=5/8/10/12 稳定性见各方法 `comparison_metrics/label_stability.csv`；逐项指标差值见 `metric_deltas.csv`。",
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
                int(batch.get("n_obs_total", batch.get("n_obs", 6843))),
                int(batch.get("n_used", batch.get("n_obs", 6843))),
                int(batch.get("n_batches", 2)),
                str(batch.get("batch_categories", "NA")),
                str(batch.get("batch_counts", "NA")),
                int(batch.get("max_samples", 0)),
                int(batch.get("sample_size_requested", 0)),
                int(batch.get("seed", 0)),
                str(batch.get("embedding_scaled", "NA")),
                int(
                    batch.get(
                        "bASW_sample_size",
                        batch.get("n_used", batch.get("n_obs", 6843)),
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
        "这些 batch 指标不随本次 KMeans 输入预处理变化，故只保存一份；来源与复制文件的 SHA256 位于 `shared_metrics/source_manifest.csv`。",
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
        "1. 两套预处理结果均已作为一等结果保存在原始文件体系中，而非只写入报告；标签、指标、图、配置和 scaler 参数可独立复核。",
        "2. 进行方法主比较时，应固定选择同一分支；不要把 raw 分支某方法与 standardized 分支另一方法直接排名。",
        "3. 建议正文以 standardized 分支作为尺度可比的主结果，raw 分支作为敏感性分析；同时报告 raw-vs-standardized ARI/NMI，说明结论对预处理的稳健程度。",
        "4. Human_Lymph_Node 缺少统一真实区域标签，因此 section ARI/NMI 是批次/切片分离诊断，不是生物学聚类准确率。",
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
            "Delete clustering directories outside K=5/8/10/12; retain "
            "existing metric CSV files."
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
        prune_unselected_clustering_directories(method)
        write_manifest(method)
    build_report(methods)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
