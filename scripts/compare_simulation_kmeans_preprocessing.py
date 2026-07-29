#!/usr/bin/env python3
"""Create independent raw-vs-standardized KMeans results for Simulation.

This script is intentionally read-only with respect to the original training and
analysis directories.  It loads each method's saved final representation, aligns
metadata against that method's own Simulation data directory, and writes a new
comparison root for each method.
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


SECTIONS = [f"Simulation{i}" for i in range(1, 6)]
KS = [5, 8, 10, 12]
SEED = 42
N_INIT = 20
MAX_ITER = 300
SPATIAL_NEIGHBOR_K = 6
POINT_SIZE = 28.0
DPI = 180

REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/simulation_preprocessing_comparison_report.md"
)


@dataclass
class MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    truth: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]


METHOD_ORDER = ["spa_mo_model", "MOFA+", "COSIE", "SpaMosaic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Independent Markdown report path.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Regenerate only the report from existing comparison results; "
            "do not rerun KMeans, metrics, or plots."
        ),
    )
    parser.add_argument(
        "--refresh-spatial-only",
        action="store_true",
        help=(
            "Refresh spatial-continuity files/config/report in existing comparison "
            "roots without rerunning KMeans or touching labels/plots."
        ),
    )
    parser.add_argument(
        "--refresh-plots-only",
        action="store_true",
        help=(
            "Refresh spatial plots and point-size metadata in existing comparison "
            "roots without rerunning KMeans or touching labels/metrics."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_and_coords(
    data_dir: Path,
    sections: np.ndarray,
    barcodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.empty(len(sections), dtype=object)
    coords = np.empty((len(sections), 2), dtype=float)
    for section in SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source_names = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            pos = source_names.get_indexer(wanted)
            if (pos < 0).any():
                missing = wanted[pos < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: failed to align {int((pos < 0).sum())} barcodes; "
                    f"examples={missing}"
                )
            if "spatial_domain" in rna.obs:
                sec_truth = rna.obs["spatial_domain"].astype(str).to_numpy()[pos]
            elif "spfac" in rna.obsm:
                spfac = np.asarray(rna.obsm["spfac"])[pos]
                sec_truth = np.full(len(pos), "background", dtype=object)
                active = spfac.sum(axis=1) > 0
                sec_truth[active] = [
                    f"sp{i + 1}" for i in spfac[active].argmax(axis=1)
                ]
            else:
                raise ValueError(f"{rna_path}: no spatial_domain or spfac truth.")
            spatial = np.asarray(rna.obsm["spatial"])[pos, :2]
        finally:
            rna.file.close()
        truth[mask] = sec_truth
        coords[mask] = spatial
    return truth.astype(str), coords


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/simulation/"
        "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/Simulation")
    arrays = []
    sections = []
    barcodes = []
    sources: list[Path] = []
    for section in SECTIONS:
        emb_path = run / f"final_embeddings_{section}.npy"
        meta_path = run / f"obs_metadata_{section}.csv"
        meta = pd.read_csv(meta_path)
        arrays.append(np.load(emb_path))
        sections.extend([section] * len(meta))
        barcodes.extend(meta["obs_name"].astype(str).tolist())
        sources.extend([emb_path, meta_path])
    section_arr = np.asarray(sections, dtype=str)
    barcode_arr = np.asarray(barcodes, dtype=str)
    truth, coords = truth_and_coords(data_dir, section_arr, barcode_arr)
    old_analysis = run / "clustering_analysis"
    return MethodData(
        name="spa_mo_model",
        embedding=np.vstack(arrays),
        sections=section_arr,
        barcodes=barcode_arr,
        truth=truth,
        coords=coords,
        data_dir=data_dir,
        source_paths=sources,
        output_root=Path(
            "/home/hujinlan/spa_mo_model/results/"
            "simulation_preprocessing_comparison"
        ),
        shared_sources={
            "simulation_factor_diagnostics.csv": old_analysis
            / "metrics"
            / "simulation_factor_diagnostics.csv",
            "cross_section_spot_retrieval.csv": old_analysis
            / "metrics"
            / "cross_section_spot_retrieval.csv",
            "batch_correction_metrics.csv": old_analysis
            / "batch_correction_metrics.csv",
        },
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "simulation_mofa_hvg1000_k10_iter1000"
    )
    table_path = old / "tables" / "factors_with_metadata.csv"
    table = pd.read_csv(table_path)
    factor_cols = [col for col in table if col.startswith("Factor")]
    sections = table["group"].astype(str).to_numpy()
    barcodes = table["original_barcode"].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/Simulation")
    truth, coords = truth_and_coords(data_dir, sections, barcodes)
    return MethodData(
        name="MOFA+",
        embedding=table[factor_cols].to_numpy(dtype=float),
        sections=sections,
        barcodes=barcodes,
        truth=truth,
        coords=coords,
        data_dir=data_dir,
        source_paths=[table_path],
        output_root=Path(
            "/home/hujinlan/mofa+/analysis/simulation_preprocessing_comparison"
        ),
        shared_sources={
            "simulation_factor_diagnostics.csv": old
            / "metrics"
            / "simulation_factor_diagnostics.csv",
            "cross_section_spot_retrieval.csv": old
            / "metrics"
            / "cross_section_spot_retrieval.csv",
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "r2_total_by_view_group.csv": old
            / "tables"
            / "r2_total_by_view_group.csv",
        },
    )


def load_cosie() -> MethodData:
    old = Path("/home/hujinlan/cosie_runs/simulation_cosie_rna_adt_full/analysis")
    table_path = old / "tables" / "embeddings_with_metadata.csv"
    table = pd.read_csv(table_path, index_col=0)
    emb_cols = [col for col in table if col.startswith("COSIE")]
    barcode_col = "obs_name.1" if "obs_name.1" in table else "obs_name"
    sections = table["section"].astype(str).to_numpy()
    barcodes = table[barcode_col].astype(str).to_numpy()
    data_dir = Path("/home/hujinlan/cosie/data/Simulation")
    truth, coords = truth_and_coords(data_dir, sections, barcodes)
    return MethodData(
        name="COSIE",
        embedding=table[emb_cols].to_numpy(dtype=float),
        sections=sections,
        barcodes=barcodes,
        truth=truth,
        coords=coords,
        data_dir=data_dir,
        source_paths=[table_path],
        output_root=Path(
            "/home/hujinlan/cosie_runs/simulation_preprocessing_comparison"
        ),
        shared_sources={
            "simulation_factor_diagnostics.csv": old
            / "metrics"
            / "simulation_factor_diagnostics.csv",
            "cross_section_spot_retrieval.csv": old
            / "metrics"
            / "cross_section_spot_retrieval.csv",
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/simulation_spamosaic/"
        "simulation_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs["section"].astype(str).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path("/home/hujinlan/SpaMosaic-dev/demo/data/Simulation")
    truth, coords = truth_and_coords(data_dir, sections, barcodes)
    old = Path("/home/hujinlan/SpaMosaic-dev/analysis/simulation_spamosaic")
    return MethodData(
        name="SpaMosaic",
        embedding=embedding,
        sections=sections,
        barcodes=barcodes,
        truth=truth,
        coords=coords,
        data_dir=data_dir,
        source_paths=[h5ad_path],
        output_root=Path(
            "/home/hujinlan/SpaMosaic-dev/analysis/"
            "simulation_preprocessing_comparison"
        ),
        shared_sources={
            "simulation_factor_diagnostics.csv": old
            / "metrics"
            / "simulation_factor_diagnostics.csv",
            "cross_section_spot_retrieval.csv": old
            / "metrics"
            / "cross_section_spot_retrieval.csv",
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv",
            "modality_alignment_cosine.csv": old
            / "metrics"
            / "modality_alignment_cosine.csv",
        },
    )


def validate_method(data: MethodData) -> None:
    n = data.embedding.shape[0]
    if (
        len(data.sections) != n
        or len(data.barcodes) != n
        or len(data.truth) != n
        or len(data.coords) != n
    ):
        raise ValueError(f"{data.name}: row counts are inconsistent.")
    if n != 6480:
        raise ValueError(f"{data.name}: expected 6480 spots, observed {n}.")
    if not np.isfinite(data.embedding).all():
        raise ValueError(f"{data.name}: embedding contains non-finite values.")
    if set(np.unique(data.sections)) != set(SECTIONS):
        raise ValueError(f"{data.name}: section set is inconsistent.")
    if set(np.unique(data.truth)) != {"background", "sp1", "sp2", "sp3", "sp4"}:
        raise ValueError(f"{data.name}: spatial-domain truth is inconsistent.")
    if data.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing comparison root: {data.output_root}"
        )
    missing = [str(path) for path in data.shared_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{data.name}: missing shared sources: {missing}")


def fitted_space(
    raw: np.ndarray,
    scheme: str,
) -> tuple[np.ndarray, StandardScaler | None]:
    if scheme == "raw_embedding":
        return raw, None
    scaler = StandardScaler()
    return scaler.fit_transform(raw), scaler


def spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    neighbors = NearestNeighbors(n_neighbors=SPATIAL_NEIGHBOR_K + 1)
    neighbors.fit(coords[:, :2])
    idx = neighbors.kneighbors(coords[:, :2], return_distance=False)[:, 1:]
    return float(np.mean(labels[idx] == labels[:, None]))


def safe_silhouette(space: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(space, labels))


def external_metrics(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "homogeneity": float(homogeneity_score(truth, labels)),
        "completeness": float(completeness_score(truth, labels)),
        "v_measure": float(v_measure_score(truth, labels)),
    }


def internal_metrics(space: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    asw = safe_silhouette(space, labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": float((asw + 1.0) / 2.0),
        "calinski_harabasz": float(calinski_harabasz_score(space, labels)),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
    }


def save_labels(
    path: Path,
    data: MethodData,
    mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "section": data.sections[mask],
            "obs_name": data.barcodes[mask],
            "cluster": labels.astype(int),
            "spatial_domain": data.truth[mask],
            "x": data.coords[mask, 0],
            "y": data.coords[mask, 1],
        }
    ).to_csv(path, index=False)


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
    clustering_dir = out / "clustering"
    metrics_dir = out / "metrics"
    clustering_dir.mkdir(parents=True)
    metrics_dir.mkdir()

    metric_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}

    joint_space, joint_scaler = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, "joint"))
    label_asw = safe_silhouette(joint_space, data.truth)
    section_asw = safe_silhouette(joint_space, data.sections)

    for k in KS:
        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=N_INIT,
            max_iter=MAX_ITER,
        )
        labels_all = model.fit_predict(joint_space).astype(int)
        joint_dir = clustering_dir / f"joint_k{k}"
        joint_dir.mkdir()
        row = {
            "mode": "joint",
            "scope": "combined",
            "k": k,
            "n_obs": len(labels_all),
            "embedding_dim": data.embedding.shape[1],
            **external_metrics(data.truth, labels_all),
            **internal_metrics(joint_space, labels_all),
            "label_asw": label_asw,
            "label_asw_scaled": (label_asw + 1.0) / 2.0,
            "section_ari": adjusted_rand_score(data.sections, labels_all),
            "section_nmi": normalized_mutual_info_score(data.sections, labels_all),
            "section_asw": section_asw,
        }
        metric_rows.append(row)
        counts = []
        for section in SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            save_labels(
                joint_dir / f"labels_{section}.csv",
                data,
                mask,
                labels,
            )
            plot_spatial(
                joint_dir / f"spatial_{section}.png",
                data.coords[mask],
                labels,
                f"{data.name} {scheme} joint K={k} {section}",
            )
            agreement = spatial_agreement(data.coords[mask], labels)
            continuity_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "neighbor_same_cluster_fraction": agreement,
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                counts.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": float(count / len(labels)),
                    }
                )
        pd.DataFrame(counts).to_csv(
            joint_dir / "cluster_counts.csv",
            index=False,
        )

        independent_dir = clustering_dir / f"independent_k{k}"
        independent_dir.mkdir()
        for section in SECTIONS:
            mask = data.sections == section
            sec_raw = data.embedding[mask]
            sec_space, sec_scaler = fitted_space(sec_raw, scheme)
            if sec_scaler is not None:
                key = section.replace(" ", "_")
                if f"{key}_mean" not in scaler_arrays:
                    scaler_arrays.update(scaler_payload(sec_scaler, key))
            labels = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=N_INIT,
                max_iter=MAX_ITER,
            ).fit_predict(sec_space).astype(int)
            save_labels(
                independent_dir / f"labels_{section}.csv",
                data,
                mask,
                labels,
            )
            plot_spatial(
                independent_dir / f"spatial_{section}.png",
                data.coords[mask],
                labels,
                f"{data.name} {scheme} independent K={k} {section}",
            )
            agreement = spatial_agreement(data.coords[mask], labels)
            continuity_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "neighbor_same_cluster_fraction": agreement,
                }
            )
            sec_label_asw = safe_silhouette(sec_space, data.truth[mask])
            metric_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **external_metrics(data.truth[mask], labels),
                    **internal_metrics(sec_space, labels),
                    "label_asw": sec_label_asw,
                    "label_asw_scaled": (sec_label_asw + 1.0) / 2.0,
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    continuity = pd.DataFrame(continuity_rows)
    metrics.to_csv(metrics_dir / "clustering_metrics.csv", index=False)
    continuity.to_csv(metrics_dir / "spatial_continuity.csv", index=False)
    (
        continuity.groupby(["mode", "k"], as_index=False)[
            "neighbor_same_cluster_fraction"
        ]
        .mean()
        .rename(
            columns={
                "neighbor_same_cluster_fraction": "mean_spatial_neighbor_agreement"
            }
        )
        .to_csv(metrics_dir / "spatial_continuity_summary.csv", index=False)
    )
    if scaler_arrays:
        np.savez_compressed(out / "scaler_parameters.npz", **scaler_arrays)

    config = {
        "dataset": "Simulation",
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
        "joint_scaler_scope": (
            "none" if scheme == "raw_embedding" else "all_6480_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "k_values": KS,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "n_obs": len(data.embedding),
        "embedding_dim": data.embedding.shape[1],
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "point_size": POINT_SIZE,
        "all_spots_used": True,
        "data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in data.source_paths
        ],
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    joint = metrics[metrics["mode"] == "joint"]
    best = joint.loc[joint["ari"].idxmax()]
    summary = (
        f"# {data.name} Simulation {scheme} KMeans analysis\n\n"
        f"- Spots: {len(data.embedding):,}\n"
        f"- Embedding dimensions: {data.embedding.shape[1]}\n"
        f"- K: {KS}\n"
        f"- KMeans: seed={SEED}, n_init={N_INIT}, max_iter={MAX_ITER}\n"
        f"- Metric space: `{config['metric_space']}`\n"
        f"- Best joint ARI: {best['ari']:.6f} at K={int(best['k'])}\n"
        f"- Best-row NMI: {best['nmi']:.6f}\n"
        f"- Labels and spatial plots: `clustering/`\n"
        f"- Complete metrics: `metrics/`\n"
    )
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")


def copy_shared_metrics(data: MethodData) -> None:
    shared = data.output_root / "shared_metrics"
    shared.mkdir(parents=True)
    source_manifest = {}
    for filename, source in data.shared_sources.items():
        target = shared / filename
        shutil.copy2(source, target)
        source_manifest[filename] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "copied_sha256": sha256(target),
            "preprocessing_dependency": "independent_of_kmeans_input",
        }
    (shared / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def refresh_spatial_outputs(data: MethodData) -> None:
    """Refresh only derived spatial-continuity tables in an existing result root."""
    if not data.output_root.exists():
        raise FileNotFoundError(data.output_root)
    for scheme in ("raw_embedding", "standardized_embedding"):
        rows = []
        for mode in ("joint", "independent"):
            for k in KS:
                cluster_dir = (
                    data.output_root / scheme / "clustering" / f"{mode}_k{k}"
                )
                for section in SECTIONS:
                    mask = data.sections == section
                    labels_path = cluster_dir / f"labels_{section}.csv"
                    labels = pd.read_csv(labels_path)
                    expected = data.barcodes[mask].astype(str)
                    actual = labels["obs_name"].astype(str).to_numpy()
                    if not np.array_equal(expected, actual):
                        raise ValueError(
                            f"{labels_path}: labels are not aligned to method rows."
                        )
                    rows.append(
                        {
                            "mode": mode,
                            "k": k,
                            "section": section,
                            "neighbor_same_cluster_fraction": spatial_agreement(
                                data.coords[mask],
                                labels["cluster"].to_numpy(dtype=int),
                            ),
                        }
                    )
        continuity = pd.DataFrame(rows)
        metrics_dir = data.output_root / scheme / "metrics"
        continuity.to_csv(metrics_dir / "spatial_continuity.csv", index=False)
        (
            continuity.groupby(["mode", "k"], as_index=False)[
                "neighbor_same_cluster_fraction"
            ]
            .mean()
            .rename(
                columns={
                    "neighbor_same_cluster_fraction": (
                        "mean_spatial_neighbor_agreement"
                    )
                }
            )
            .to_csv(
                metrics_dir / "spatial_continuity_summary.csv",
                index=False,
            )
        )
        config_path = data.output_root / scheme / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["spatial_neighbor_k"] = SPATIAL_NEIGHBOR_K
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def refresh_spatial_plots(data: MethodData) -> None:
    """Refresh only spatial plots and point-size metadata in an existing root."""
    if not data.output_root.exists():
        raise FileNotFoundError(data.output_root)
    for scheme in ("raw_embedding", "standardized_embedding"):
        for mode in ("joint", "independent"):
            for k in KS:
                cluster_dir = (
                    data.output_root / scheme / "clustering" / f"{mode}_k{k}"
                )
                for section in SECTIONS:
                    mask = data.sections == section
                    labels_path = cluster_dir / f"labels_{section}.csv"
                    labels = pd.read_csv(labels_path)
                    expected = data.barcodes[mask].astype(str)
                    actual = labels["obs_name"].astype(str).to_numpy()
                    if not np.array_equal(expected, actual):
                        raise ValueError(
                            f"{labels_path}: labels are not aligned to method rows."
                        )
                    plot_spatial(
                        cluster_dir / f"spatial_{section}.png",
                        data.coords[mask],
                        labels["cluster"].to_numpy(dtype=int),
                        f"{data.name} {scheme} {mode} K={k} {section}",
                    )
        config_path = data.output_root / scheme / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["point_size"] = POINT_SIZE
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def write_root_manifest(data: MethodData) -> None:
    payload = {
        "dataset": "Simulation",
        "method": data.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "comparison_root": str(data.output_root),
        "legacy_results_modified": False,
        "variants": {
            "raw_embedding": str(data.output_root / "raw_embedding"),
            "standardized_embedding": str(
                data.output_root / "standardized_embedding"
            ),
        },
        "shared_metrics": str(data.output_root / "shared_metrics"),
        "source_data_dir": str(data.data_dir),
    }
    (data.output_root / "preprocessing_comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in value
    )


def pad(value: Any, width: int, right: bool = False) -> str:
    text = str(value)
    spaces = max(0, width - display_width(text))
    return (" " * spaces + text) if right else (text + " " * spaces)


def md_table(
    headers: list[str],
    rows: list[list[Any]],
    numeric_columns: set[int] | None = None,
) -> str:
    numeric_columns = numeric_columns or set()
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(
            3,
            display_width(headers[index]),
            max((display_width(row[index]) for row in text_rows), default=0),
        )
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            pad(header, widths[index], index in numeric_columns)
            for index, header in enumerate(headers)
        )
        + " |"
    ]
    lines.append(
        "| "
        + " | ".join(
            ("-" * (widths[index] - 1) + ":")
            if index in numeric_columns
            else ("-" * widths[index])
            for index in range(len(headers))
        )
        + " |"
    )
    lines.extend(
        "| "
        + " | ".join(
            pad(value, widths[index], index in numeric_columns)
            for index, value in enumerate(row)
        )
        + " |"
        for row in text_rows
    )
    return "\n".join(lines)


def load_joint(data: MethodData, scheme: str) -> pd.DataFrame:
    frame = pd.read_csv(
        data.output_root
        / scheme
        / "metrics"
        / "clustering_metrics.csv"
    )
    return frame[frame["mode"] == "joint"].copy()


def metric_matrix(
    methods: dict[str, MethodData],
    scheme: str,
    column: str,
) -> str:
    tables = {name: load_joint(methods[name], scheme) for name in METHOD_ORDER}
    rows = []
    for k in KS:
        row: list[Any] = [k]
        for name in METHOD_ORDER:
            value = tables[name].loc[tables[name]["k"] == k, column].iloc[0]
            row.append(f"{value:.4f}")
        rows.append(row)
    return md_table(
        ["K", *METHOD_ORDER],
        rows,
        numeric_columns=set(range(5)),
    )


def best_table(methods: dict[str, MethodData], scheme: str) -> str:
    rows = []
    for name in METHOD_ORDER:
        joint = load_joint(methods[name], scheme)
        best = joint.loc[joint["ari"].idxmax()]
        rows.append(
            [
                name,
                int(best["k"]),
                f"{best['ari']:.4f}",
                f"{best['nmi']:.4f}",
                f"{best['homogeneity']:.4f}",
                f"{best['completeness']:.4f}",
                f"{best['v_measure']:.4f}",
                f"{best['cluster_asw']:.4f}",
                f"{best['cluster_asw_scaled']:.4f}",
            ]
        )
    return md_table(
        [
            "方法",
            "best K",
            "ARI",
            "NMI",
            "Homogeneity",
            "Completeness",
            "V-measure",
            "ASW raw",
            "ASW scaled",
        ],
        rows,
        numeric_columns=set(range(1, 9)),
    )


def internal_table(
    methods: dict[str, MethodData],
    scheme: str,
    metric: str,
) -> str:
    return metric_matrix(methods, scheme, metric)


def spatial_table(methods: dict[str, MethodData], scheme: str, mode: str) -> str:
    rows = []
    for k in KS:
        row: list[Any] = [k]
        for name in METHOD_ORDER:
            frame = pd.read_csv(
                methods[name].output_root
                / scheme
                / "metrics"
                / "spatial_continuity_summary.csv"
            )
            value = frame.loc[
                (frame["mode"] == mode) & (frame["k"] == k),
                "mean_spatial_neighbor_agreement",
            ].iloc[0]
            row.append(f"{value:.4f}")
        rows.append(row)
    return md_table(
        ["K", *METHOD_ORDER],
        rows,
        numeric_columns=set(range(5)),
    )


def sensitivity_table(methods: dict[str, MethodData]) -> str:
    rows = []
    for name in METHOD_ORDER:
        raw = load_joint(methods[name], "raw_embedding")
        std = load_joint(methods[name], "standardized_embedding")
        raw_best = raw.loc[raw["ari"].idxmax()]
        std_best = std.loc[std["ari"].idxmax()]
        raw_k5 = raw.loc[raw["k"] == 5].iloc[0]
        std_k5 = std.loc[std["k"] == 5].iloc[0]
        rows.append(
            [
                name,
                f"{raw_best['ari']:.4f} (K={int(raw_best['k'])})",
                f"{std_best['ari']:.4f} (K={int(std_best['k'])})",
                f"{std_best['ari'] - raw_best['ari']:+.4f}",
                f"{raw_k5['ari']:.4f}",
                f"{std_k5['ari']:.4f}",
                f"{std_k5['ari'] - raw_k5['ari']:+.4f}",
            ]
        )
    return md_table(
        [
            "方法",
            "Raw best ARI",
            "Standardized best ARI",
            "Δ best ARI",
            "Raw K=5 ARI",
            "Standardized K=5 ARI",
            "Δ K=5 ARI",
        ],
        rows,
        numeric_columns=set(range(1, 7)),
    )


def section_diagnostic_table(methods: dict[str, MethodData]) -> str:
    rows = []
    for name in METHOD_ORDER:
        for k in KS:
            for scheme, scheme_label in [
                ("raw_embedding", "Raw"),
                ("standardized_embedding", "Standardized"),
            ]:
                joint = load_joint(methods[name], scheme).set_index("k")
                row = joint.loc[k]
                rows.append(
                    [
                        name,
                        k,
                        scheme_label,
                        int(row["n_obs"]),
                        f"{row['section_ari']:.6f}",
                        f"{row['section_nmi']:.6f}",
                        f"{row['section_asw']:.6f}",
                    ]
                )
    return md_table(
        [
            "方法",
            "K",
            "预处理",
            "n",
            "section ARI",
            "section NMI",
            "section ASW raw",
        ],
        rows,
        numeric_columns={1, 3, 4, 5, 6},
    )


def batch_values_table(methods: dict[str, MethodData]) -> str:
    rows = []
    for name in METHOD_ORDER:
        batch = pd.read_csv(
            methods[name].output_root
            / "shared_metrics"
            / "batch_correction_metrics.csv"
        ).iloc[0]
        rows.append(
            [
                name,
                str(batch.get("batch_key", "NA")),
                int(batch.get("n_used", batch.get("n_obs", 6480))),
                int(
                    batch.get(
                        "bASW_sample_size",
                        batch.get("n_used", batch.get("n_obs", 6480)),
                    )
                ),
                f"{float(batch.get('bASW_raw')):.6f}",
                f"{float(batch.get('bASW')):.6f}",
                f"{float(batch.get('bLISI_raw')):.6f}",
                f"{float(batch.get('bLISI')):.6f}",
                f"{float(batch.get('kBET_rejection_rate')):.6f}",
                f"{float(batch.get('kBET')):.6f}",
                f"{float(batch.get('PCR_batch_R2')):.6f}",
                f"{float(batch.get('PCR_score')):.6f}",
            ]
        )
    return md_table(
        [
            "方法",
            "batch key",
            "n used",
            "bASW n",
            "bASW raw",
            "bASW score",
            "bLISI raw",
            "bLISI normalized",
            "kBET rejection",
            "kBET acceptance",
            "PCR batch R2",
            "PCR score",
        ],
        rows,
        numeric_columns=set(range(2, 12)),
    )


def factor_diagnostic_table(methods: dict[str, MethodData]) -> str:
    frames = {
        name: pd.read_csv(
            methods[name].output_root
            / "shared_metrics"
            / "simulation_factor_diagnostics.csv"
        ).set_index("diagnostic")
        for name in METHOD_ORDER
    }
    diagnostics = [
        ("spfac_recovery", "spfac recovery", "越高越好"),
        ("rna_nsfac_leakage", "RNA nsfac leakage", "越低越好"),
        ("adt_nsfac_leakage", "ADT nsfac leakage", "越低越好"),
    ]
    rows = []
    for key, label, direction in diagnostics:
        rows.append(
            [
                label,
                *[
                    f"{frames[name].loc[key, 'cv_linear_r2_variance_weighted']:.4f}"
                    for name in METHOD_ORDER
                ],
                direction,
            ]
        )
    return md_table(
        ["诊断项", *METHOD_ORDER, "方向"],
        rows,
        numeric_columns={1, 2, 3, 4},
    )


def retrieval_table(methods: dict[str, MethodData]) -> str:
    rows = []
    for name in METHOD_ORDER:
        retrieval = pd.read_csv(
            methods[name].output_root
            / "shared_metrics"
            / "cross_section_spot_retrieval.csv"
        )
        rows.append(
            [
                name,
                int(len(retrieval)),
                f"{retrieval['top1_accuracy'].mean():.4f}",
                f"{retrieval['recall_at_5'].mean():.4f}",
                f"{retrieval['mean_foscttm'].mean():.4f}",
            ]
        )
    return md_table(
        [
            "方法",
            "相邻切片对数",
            "Top-1 accuracy",
            "Recall@5",
            "Mean FOSCTTM",
        ],
        rows,
        numeric_columns={1, 2, 3, 4},
    )


def mofa_r2_table(methods: dict[str, MethodData]) -> str:
    r2 = pd.read_csv(
        methods["MOFA+"].output_root
        / "shared_metrics"
        / "r2_total_by_view_group.csv"
    )
    return md_table(
        ["View", "Group", "R2"],
        [
            [row["View"], row["Group"], f"{row['R2']:.4f}"]
            for _, row in r2.iterrows()
        ],
        numeric_columns={2},
    )


def spamosaic_alignment_table(methods: dict[str, MethodData]) -> str:
    alignment = pd.read_csv(
        methods["SpaMosaic"].output_root
        / "shared_metrics"
        / "modality_alignment_cosine.csv"
    )
    return md_table(
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
                f"{row['mean_cosine']:.4f}",
                f"{row['median_cosine']:.4f}",
                f"{row['std_cosine']:.4f}",
            ]
            for _, row in alignment.iterrows()
        ],
        numeric_columns={2, 3, 4, 5},
    )


def ranking_text(methods: dict[str, MethodData], scheme: str, fixed_k: int) -> str:
    scores = []
    for name in METHOD_ORDER:
        joint = load_joint(methods[name], scheme)
        value = float(joint.loc[joint["k"] == fixed_k, "ari"].iloc[0])
        scores.append((name, value))
    scores.sort(key=lambda item: item[1], reverse=True)
    return " > ".join(f"{name} ({score:.4f})" for name, score in scores)


def write_report(methods: dict[str, MethodData], report_path: Path) -> None:
    result_rows = [
        [name, str(methods[name].output_root)] for name in METHOD_ORDER
    ]
    config_rows = [
        [
            "Raw embedding + KMeans",
            "原始最终 embedding",
            "原始最终 embedding",
            "不拟合",
        ],
        [
            "Standardized embedding + KMeans",
            "StandardScaler 后 embedding",
            "StandardScaler 后 embedding",
            "joint 全量拟合；independent 每个 section 分别拟合",
        ],
    ]
    parts = [
        "# Simulation：Raw 与 Standardized embedding + KMeans 对比报告",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}。",
        "",
        "本报告只比较同一批已训练最终表示在两种 KMeans 输入预处理下的结果。"
        "没有重新训练模型，也没有覆盖四种方法原有的 Simulation 训练或分析目录。",
        "",
        "## 结果目录",
        "",
        md_table(["方法", "独立对比目录"], result_rows),
        "",
        "每个目录均包含 `raw_embedding/`、`standardized_embedding/` 和 "
        "`shared_metrics/`。逐 spot labels、全量空间图、完整指标、配置及 "
        "Scaler 参数均保存在对应目录中；原始 embedding 不重复复制。",
        "",
        "## 统一评价口径",
        "",
        md_table(
            ["方案", "KMeans 输入", "距离类指标空间", "Scaler 范围"],
            config_rows,
        ),
        "",
        f"两套方案都使用全部 6,480 个 spot、K={KS}、seed={SEED}、"
        f"n_init={N_INIT}、max_iter={MAX_ITER}。空间连续性均使用 "
        f"{SPATIAL_NEIGHBOR_K} 近邻，空间图 spot size={POINT_SIZE:g}。"
        "ARI、NMI、Homogeneity、Completeness 与 V-measure 使用相同的 "
        "`spatial_domain` 真值。每种方法只读取自己数据目录中的真值和坐标。",
        "",
        "距离类指标遵循各自方案：Raw 方案的 ASW/CH/DBI/Label ASW 在 raw "
        "embedding 空间计算；Standardized 方案在 standardized embedding "
        "空间计算。因此同一方案内的四方法比较口径一致，而两方案之间的变化代表"
        "完整预处理方案敏感性。",
        "",
        "表格中的 `ASW raw` 沿用总报告命名，表示未做 `(ASW+1)/2` 映射的 "
        "silhouette 原值，并不表示该行一定使用 raw embedding；其实际距离空间"
        "由所在预处理方案决定。",
        "",
        "## Raw embedding + KMeans",
        "",
        "### 共同 K 的空间域 ARI",
        "",
        metric_matrix(methods, "raw_embedding", "ari"),
        "",
        "### 共同 K 的空间域 NMI",
        "",
        metric_matrix(methods, "raw_embedding", "nmi"),
        "",
        "### 各方法最佳空间域外部指标",
        "",
        best_table(methods, "raw_embedding"),
        "",
        "### 无监督内部指标",
        "",
        "ASW raw：",
        "",
        internal_table(methods, "raw_embedding", "cluster_asw"),
        "",
        "CH：",
        "",
        internal_table(methods, "raw_embedding", "calinski_harabasz"),
        "",
        "DBI：",
        "",
        internal_table(methods, "raw_embedding", "davies_bouldin"),
        "",
        "### 空间连续性",
        "",
        "Joint：",
        "",
        spatial_table(methods, "raw_embedding", "joint"),
        "",
        "Independent：",
        "",
        spatial_table(methods, "raw_embedding", "independent"),
        "",
        "## Standardized embedding + KMeans",
        "",
        "### 共同 K 的空间域 ARI",
        "",
        metric_matrix(methods, "standardized_embedding", "ari"),
        "",
        "### 共同 K 的空间域 NMI",
        "",
        metric_matrix(methods, "standardized_embedding", "nmi"),
        "",
        "### 各方法最佳空间域外部指标",
        "",
        best_table(methods, "standardized_embedding"),
        "",
        "### 无监督内部指标",
        "",
        "ASW raw：",
        "",
        internal_table(methods, "standardized_embedding", "cluster_asw"),
        "",
        "CH：",
        "",
        internal_table(
            methods,
            "standardized_embedding",
            "calinski_harabasz",
        ),
        "",
        "DBI：",
        "",
        internal_table(methods, "standardized_embedding", "davies_bouldin"),
        "",
        "### 空间连续性",
        "",
        "Joint：",
        "",
        spatial_table(methods, "standardized_embedding", "joint"),
        "",
        "Independent：",
        "",
        spatial_table(methods, "standardized_embedding", "independent"),
        "",
        "## 两种预处理方案的敏感性",
        "",
        sensitivity_table(methods),
        "",
        "固定 K=5 的 Raw ARI 排名："
        + ranking_text(methods, "raw_embedding", 5)
        + "。",
        "",
        "固定 K=5 的 Standardized ARI 排名："
        + ranking_text(methods, "standardized_embedding", 5)
        + "。",
        "",
        "Simulation 的 `spatial_domain` 真值为 5 类，因此固定 K=5 是主要排名；"
        "跨 K 取最佳值作为补充敏感性结果，不与固定 K=5 排名混为一谈。",
        "",
        "## Section依赖诊断：Raw / Standardized上下对齐",
        "",
        "同一方法、同一K的Raw与Standardized行相邻。section ARI/NMI衡量"
        "联合聚类与五张切片身份的一致性，section ASW使用相应预处理分支的"
        "KMeans输入空间；这些指标是切片依赖诊断，不是空间域聚类准确率。",
        "",
        section_diagnostic_table(methods),
        "",
        "## 不受 KMeans 输入预处理影响的共享指标",
        "",
        "模拟因子恢复、nuisance leakage、跨切片同网格检索和固定口径的 batch "
        "correction 指标不由 KMeans 输入决定，故每种方法只在 "
        "`shared_metrics/` 中保存一份，并附源文件 SHA-256 清单。MOFA+ 的 "
        "view/group R2 与 SpaMosaic 的模态对齐指标也按同样方式保存。",
        "",
        "### Batch数值表",
        "",
        "bASW score、bLISI normalized、kBET acceptance和PCR score越高越好；"
        "kBET rejection与PCR batch R2越低越好。",
        "",
        batch_values_table(methods),
        "",
        "### 模拟因子诊断表",
        "",
        "数值为cross-validation的variance-weighted linear R²。空间因子恢复"
        "越高越好，RNA/ADT nuisance因子泄漏越低越好。",
        "",
        factor_diagnostic_table(methods),
        "",
        "### 跨切片检索表",
        "",
        "下表汇总四对相邻切片的均值；Top-1与Recall@5越高越好，"
        "Mean FOSCTTM越低越好。",
        "",
        retrieval_table(methods),
        "",
        "### MOFA+视图解释度R²表",
        "",
        mofa_r2_table(methods),
        "",
        "该表由MOFA+模型输出决定，与本次KMeans输入预处理无关。",
        "",
        "### SpaMosaic模态对齐表",
        "",
        spamosaic_alignment_table(methods),
        "",
        "该表由SpaMosaic模型输出决定，与本次KMeans输入预处理无关。",
        "",
        "## 结论",
        "",
        "两套结果应作为平行评价口径同时保留。Raw 方案评价模型输出的原生欧氏"
        "几何；Standardized 方案评价各维等权后的几何。StandardScaler 对不同"
        "模型并非中性变换，因此报告同时给出两套完整结果和差值，不根据某一种"
        "方法的优势选择性汇报。",
        "",
    ]
    report_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if sum(
        [
            bool(args.report_only),
            bool(args.refresh_spatial_only),
            bool(args.refresh_plots_only),
        ]
    ) > 1:
        raise ValueError(
            "--report-only, --refresh-spatial-only and "
            "--refresh-plots-only are mutually exclusive."
        )
    methods = {
        data.name: data
        for data in (load_spa(), load_mofa(), load_cosie(), load_spamosaic())
    }
    if args.report_only:
        write_report(methods, args.report)
        print(
            json.dumps(
                {
                    "refreshed": "report_only",
                    "report": str(args.report),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.refresh_plots_only:
        for data in methods.values():
            refresh_spatial_plots(data)
        write_report(methods, args.report)
        print(
            json.dumps(
                {
                    "refreshed": "spatial_plots_only",
                    "point_size": POINT_SIZE,
                    "report": str(args.report),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.refresh_spatial_only:
        for data in methods.values():
            refresh_spatial_outputs(data)
        write_report(methods, args.report)
        print(
            json.dumps(
                {
                    "refreshed": "spatial_continuity_only",
                    "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                    "report": str(args.report),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report}")
    for data in methods.values():
        validate_method(data)
    for data in methods.values():
        run_scheme(data, "raw_embedding")
        run_scheme(data, "standardized_embedding")
        copy_shared_metrics(data)
        write_root_manifest(data)
    write_report(methods, args.report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "result_roots": {
                    name: str(data.output_root) for name, data in methods.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
