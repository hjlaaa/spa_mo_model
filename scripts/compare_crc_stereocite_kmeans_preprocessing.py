#!/usr/bin/env python3
"""Direct, traceable CRC Stereo-CITE-seq preprocessing comparison.

Each method uses its own data and final embedding.  Raw and StandardScaler
branches use MiniBatchKMeans and write full labels before metrics/figures are
derived directly from those labels.  No report-derived metric chain is used.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from compare_mouse_thymus_kmeans_preprocessing import (
    f4,
    internal_metrics,
    md_table,
    metric_space,
    safe_precomputed_asw,
    scaler_payload,
    sha256,
)


SECTIONS = ["CRC_003_bin20", "CRC_006_bin20"]
SHORT_NAMES = {
    "CRC_003_bin20": "CRC_003",
    "CRC_006_bin20": "CRC_006",
}
SECTION_COUNTS = {
    "CRC_003_bin20": 166279,
    "CRC_006_bin20": 446095,
}
ASW_SECTION_COUNTS = {
    "CRC_003_bin20": 2716,
    "CRC_006_bin20": 7284,
}
PLOT_SECTION_COUNTS = SECTION_COUNTS.copy()
N_OBS = sum(SECTION_COUNTS.values())
ASW_N_OBS = sum(ASW_SECTION_COUNTS.values())
KS = [5, 10, 15, 20, 25]
SEED = 0
N_INIT = 20
MAX_ITER = 300
BATCH_SIZE = 8192
SPATIAL_NEIGHBOR_K = 6
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/"
    "crc_stereocite_preprocessing_comparison_report.md"
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
            coords[mask] = np.asarray(rna.obsm["spatial"])[positions, :2]
        finally:
            rna.file.close()
    return coords


def load_spa() -> MethodData:
    run = Path(
        "/home/hujinlan/spa_mo_model/results/crc_stereocite/"
        "fullspot_200ep_bidirectional_ot_attention_all_checkpoint_"
        "detailmem_lc0.1_seed42"
    )
    data_dir = Path(
        "/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq"
    )
    arrays, sections, barcodes, sources = [], [], [], []
    for section in SECTIONS:
        short = SHORT_NAMES[section]
        embedding_path = run / f"final_embeddings_{short}.npy"
        indices_path = run / f"selected_spot_indices_{short}.npy"
        spatial_path = run / f"spatial_{short}.npy"
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            own_names = rna.obs_names.astype(str).to_numpy()[indices]
            own_coords = np.asarray(rna.obsm["spatial"])[indices, :2]
        finally:
            rna.file.close()
        if (
            len(embedding) != len(indices)
            or len(np.unique(indices)) != len(indices)
            or not np.allclose(saved_coords, own_coords)
        ):
            raise ValueError(
                f"spa_mo_model {section}: embedding/index/spatial mismatch"
            )
        arrays.append(embedding)
        sections.extend([section] * len(indices))
        barcodes.extend(own_names.tolist())
        sources.extend([embedding_path, indices_path, spatial_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / "clustering_analysis_k8_10_15"
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
            "crc_stereocite_preprocessing_comparison"
        ),
        {"batch_correction_metrics.csv": old / "batch_correction_metrics.csv"},
    )


def load_mofa() -> MethodData:
    old = Path(
        "/home/hujinlan/mofa+/analysis/"
        "crc_stereocite_mofa_hvg2000_k10_full_iter1000_cpu_float64"
    )
    table_path = old / "tables" / "factors_with_metadata_and_coordinates.csv"
    header = pd.read_csv(table_path, nrows=0)
    factor_columns = [
        column for column in header.columns if column.startswith("Factor")
    ]
    columns = [*factor_columns, "section", "original_barcode"]
    table = pd.read_csv(
        table_path,
        usecols=columns,
        dtype={"section": str, "original_barcode": str},
    )
    sections = table["section"].to_numpy()
    barcodes = table["original_barcode"].to_numpy()
    data_dir = Path("/home/hujinlan/mofa+/data/CRC_Stereo-CITE-seq")
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
            "crc_stereocite_preprocessing_comparison"
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
        "/home/hujinlan/cosie_runs/"
        "crc_cosie_rna_adt_metacell_6x6_sparse"
    )
    data_dir = Path("/home/hujinlan/cosie/data/CRC_Stereo-CITE-seq")
    arrays, sections, barcodes, sources = [], [], [], []
    for section in SECTIONS:
        short = SHORT_NAMES[section]
        embedding_path = (
            run / "final_embeddings" / f"{short}_final_embedding.npy"
        )
        names_path = run / f"obs_names_{short}.npy"
        indices_path = run / f"selected_spot_indices_{short}.npy"
        spatial_path = run / f"spatial_{short}.npy"
        embedding = np.load(embedding_path)
        names = np.load(names_path, allow_pickle=True).astype(str)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            own_names = rna.obs_names.astype(str).to_numpy()[indices]
            own_coords = np.asarray(rna.obsm["spatial"])[indices, :2]
        finally:
            rna.file.close()
        if (
            len(embedding) != len(names)
            or not np.array_equal(names, own_names)
            or not np.allclose(saved_coords, own_coords)
        ):
            raise ValueError(
                f"COSIE {section}: embedding/name/index/spatial mismatch"
            )
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.extend(
            [embedding_path, names_path, indices_path, spatial_path]
        )
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / "analysis"
    return MethodData(
        "COSIE",
        np.vstack(arrays),
        section_array,
        barcode_array,
        aligned_coords(data_dir, section_array, barcode_array),
        data_dir,
        sources,
        Path(
            "/home/hujinlan/cosie_runs/"
            "crc_stereocite_preprocessing_comparison"
        ),
        {
            "batch_correction_metrics.csv": old
            / "metrics"
            / "batch_correction_metrics.csv"
        },
    )


def load_spamosaic() -> MethodData:
    h5ad_path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/"
        "crc_stereocite_spamosaic_full_gpu_ce/"
        "crc_stereocite_spamosaic_embeddings.h5ad"
    )
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs["section"].astype(str).to_numpy()
    barcodes = adata.obs["original_barcode"].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm["merged_emb"]).copy()
    data_dir = Path(
        "/home/hujinlan/SpaMosaic-dev/demo/data/CRC_Stereo-CITE-seq"
    )
    old = Path(
        "/home/hujinlan/SpaMosaic-dev/analysis/"
        "crc_stereocite_spamosaic_full_gpu_ce"
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
            "crc_stereocite_preprocessing_comparison"
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
    if not (
        n_obs == N_OBS
        == len(data.sections)
        == len(data.barcodes)
        == len(data.coords)
    ):
        raise ValueError(f"{data.name}: expected {N_OBS}, got {n_obs}")
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


def sorted_order(data: MethodData) -> np.ndarray:
    rank = np.array(
        [{section: i for i, section in enumerate(SECTIONS)}[x]
         for x in data.sections],
        dtype=int,
    )
    return np.lexsort((data.barcodes, rank))


def validate_cross_method_alignment(methods: list[MethodData]) -> None:
    reference = methods[0]
    reference_order = sorted_order(reference)
    reference_sections = reference.sections[reference_order]
    reference_barcodes = reference.barcodes[reference_order]
    reference_coords = reference.coords[reference_order]
    for method in methods[1:]:
        order = sorted_order(method)
        if not np.array_equal(method.sections[order], reference_sections):
            raise ValueError(f"{method.name}: section universe mismatch")
        if not np.array_equal(method.barcodes[order], reference_barcodes):
            raise ValueError(f"{method.name}: barcode universe mismatch")
        if not np.allclose(method.coords[order], reference_coords):
            raise ValueError(f"{method.name}: spatial positions mismatch")


def barcode_hash(prefix: str, barcode: str) -> str:
    return hashlib.sha256(f"{prefix}{barcode}".encode("utf-8")).hexdigest()


def select_by_barcode_hash(
    reference: MethodData,
    counts: dict[str, int],
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for section in SECTIONS:
        barcodes = reference.barcodes[reference.sections == section]
        frame = pd.DataFrame({"section": section, "obs_name": barcodes})
        frame["barcode_sha256"] = [
            barcode_hash(prefix, value) for value in frame["obs_name"]
        ]
        frame = frame.sort_values(
            ["barcode_sha256", "obs_name"], kind="mergesort"
        ).head(counts[section]).reset_index(drop=True)
        frame["sample_order_within_section"] = np.arange(len(frame))
        frame["section_n_total"] = SECTION_COUNTS[section]
        frame["section_n_sampled"] = counts[section]
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if (
        len(result) != sum(counts.values())
        or result[["section", "obs_name"]].duplicated().any()
    ):
        raise ValueError("invalid deterministic barcode-hash sample")
    return result


def key_mask(data: MethodData, sample: pd.DataFrame) -> np.ndarray:
    selected = set(zip(sample["section"], sample["obs_name"]))
    return np.fromiter(
        (
            (section, barcode) in selected
            for section, barcode in zip(data.sections, data.barcodes)
        ),
        dtype=bool,
        count=len(data.embedding),
    )


def ordered_barcode_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_spatial_graph(coords: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=SPATIAL_NEIGHBOR_K + 1,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(coords)
    raw = model.kneighbors(coords, return_distance=False)
    row_ids = np.arange(len(coords))[:, None]
    without_self = raw[raw != row_ids].reshape(
        len(coords), SPATIAL_NEIGHBOR_K
    )
    return without_self.astype(np.int32)


def save_labels(
    path: Path, data: MethodData, mask: np.ndarray, labels: np.ndarray
) -> str:
    pd.DataFrame(
        {
            "section": data.sections[mask],
            "obs_name": data.barcodes[mask],
            "cluster": labels.astype(int),
            "x": data.coords[mask, 0],
            "y": data.coords[mask, 1],
        }
    ).to_csv(path, index=False)
    return sha256(path)


def fit_minibatch(space: np.ndarray, k: int) -> tuple[np.ndarray, MiniBatchKMeans]:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=MAX_ITER,
        batch_size=BATCH_SIZE,
        compute_labels=True,
        random_state=SEED,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=N_INIT,
        reassignment_ratio=0.01,
    )
    return model.fit_predict(space), model


def save_model(
    directory: Path, model: MiniBatchKMeans, labels_path: Path
) -> None:
    np.save(directory / "cluster_centers.npy", model.cluster_centers_)
    payload = {
        "class": "sklearn.cluster.MiniBatchKMeans",
        "parameters": {
            "n_clusters": int(model.n_clusters),
            "seed": SEED,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "batch_size": BATCH_SIZE,
            "init": "k-means++",
            "tol": 0.0,
            "max_no_improvement": 10,
            "init_size": None,
            "reassignment_ratio": 0.01,
        },
        "n_iter": int(model.n_iter_),
        "n_steps": int(model.n_steps_),
        "inertia": float(model.inertia_),
        "labels_path": str(labels_path),
        "labels_sha256": sha256(labels_path),
    }
    (directory / "minibatchkmeans_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def plot_spatial(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=0.45,
        cmap=plt.get_cmap("tab20", k),
        vmin=-0.5,
        vmax=k - 0.5,
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, ticks=np.arange(k))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def prepare_shared(
    data: MethodData,
    asw_sample: pd.DataFrame,
    plot_sample: pd.DataFrame,
) -> dict[str, np.ndarray]:
    output = data.output_root / "shared_metrics"
    output.mkdir(parents=True)
    asw_path = output / "asw_sample.csv"
    plot_path = output / "plot_sample.csv"
    asw_sample.to_csv(asw_path, index=False)
    plot_sample.to_csv(plot_path, index=False)

    copy_rows = []
    for name, source in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        copy_rows.append(
            {
                "artifact": name,
                "source": str(source),
                "source_sha256": sha256(source),
                "copied_sha256": sha256(target),
                "preprocessing_dependent": False,
            }
        )
    pd.DataFrame(copy_rows).to_csv(
        output / "source_manifest.csv", index=False
    )

    graphs: dict[str, np.ndarray] = {}
    graph_rows = []
    for section in SECTIONS:
        mask = data.sections == section
        graph = build_spatial_graph(data.coords[mask])
        graph_path = output / f"spatial_neighbors_{section}.npz"
        np.savez_compressed(graph_path, neighbor_indices=graph)
        graphs[section] = graph
        graph_rows.append(
            {
                "section": section,
                "n_obs": int(mask.sum()),
                "neighbor_k": SPATIAL_NEIGHBOR_K,
                "algorithm": "sklearn NearestNeighbors kd_tree exact",
                "barcode_order_sha256": ordered_barcode_sha256(
                    data.barcodes[mask]
                ),
                "graph_path": str(graph_path),
                "graph_sha256": sha256(graph_path),
            }
        )
    pd.DataFrame(graph_rows).to_csv(
        output / "spatial_graph_manifest.csv", index=False
    )
    sample_manifest = {
        "asw_sample": {
            "path": str(asw_path),
            "sha256": sha256(asw_path),
            "n_obs": ASW_N_OBS,
            "section_counts": ASW_SECTION_COUNTS,
            "selection": "lowest SHA256('asw_seed0|' + barcode) per section",
        },
        "plot_sample": {
            "path": str(plot_path),
            "sha256": sha256(plot_path),
            "n_obs": sum(PLOT_SECTION_COUNTS.values()),
            "section_counts": PLOT_SECTION_COUNTS,
            "selection": (
                "all spots in each section; rows stored in deterministic "
                "SHA256(barcode) order"
            ),
        },
        "identity_key": ["section", "obs_name"],
        "shared_across_all_methods_preprocessing_modes_and_k": True,
    }
    (output / "sampling_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return graphs


def spatial_agreement_from_graph(
    labels: np.ndarray, neighbors: np.ndarray
) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def run_scheme(
    data: MethodData,
    scheme: str,
    asw_sample: pd.DataFrame,
    plot_sample: pd.DataFrame,
    graphs: dict[str, np.ndarray],
) -> None:
    output = data.output_root / scheme
    clustering_root = output / "clustering"
    metrics_root = output / "metrics"
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    asw_mask = key_mask(data, asw_sample)
    plot_mask = key_mask(data, plot_sample)
    if int(asw_mask.sum()) != ASW_N_OBS:
        raise ValueError(f"{data.name}: ASW sample mapping failed")
    for section in SECTIONS:
        expected = ASW_SECTION_COUNTS[section]
        if int(np.sum(asw_mask & (data.sections == section))) != expected:
            raise ValueError(f"{data.name}: ASW count mismatch {section}")
        if (
            int(np.sum(plot_mask & (data.sections == section)))
            != PLOT_SECTION_COUNTS[section]
        ):
            raise ValueError(f"{data.name}: plot count mismatch {section}")

    metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    all_mask = np.ones(N_OBS, dtype=bool)
    asw_path = data.output_root / "shared_metrics" / "asw_sample.csv"
    plot_path = data.output_root / "shared_metrics" / "plot_sample.csv"
    asw_hash = sha256(asw_path)
    plot_hash = sha256(plot_path)

    joint_space, joint_scaler = metric_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, "joint"))
    joint_asw_indices = np.flatnonzero(asw_mask)
    joint_distances = pairwise_distances(
        joint_space[joint_asw_indices], metric="euclidean", n_jobs=-1
    )
    np.fill_diagonal(joint_distances, 0.0)
    section_asw = safe_precomputed_asw(
        joint_distances, data.sections[joint_asw_indices]
    )

    for k in KS:
        labels_all, model = fit_minibatch(joint_space, k)
        directory = clustering_root / f"joint_k{k}"
        directory.mkdir()
        labels_all_path = directory / "labels_all.csv"
        labels_all_sha = save_labels(
            labels_all_path, data, all_mask, labels_all
        )
        save_model(directory, model, labels_all_path)
        metric_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": N_OBS,
                "embedding_dim": data.embedding.shape[1],
                **internal_metrics(
                    joint_space,
                    labels_all,
                    joint_asw_indices,
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
                "labels_path": str(labels_all_path),
                "labels_sha256": labels_all_sha,
                "asw_sample_path": str(asw_path),
                "asw_sample_sha256": asw_hash,
            }
        )
        count_rows = []
        for section in SECTIONS:
            section_mask = data.sections == section
            section_labels = labels_all[section_mask]
            labels_path = directory / f"labels_{section}.csv"
            labels_sha = save_labels(
                labels_path, data, section_mask, section_labels
            )
            graph_path = (
                data.output_root
                / "shared_metrics"
                / f"spatial_neighbors_{section}.npz"
            )
            spatial_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "n_obs": int(section_mask.sum()),
                    "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction":
                    spatial_agreement_from_graph(
                        section_labels, graphs[section]
                    ),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "spatial_graph_path": str(graph_path),
                    "spatial_graph_sha256": sha256(graph_path),
                }
            )
            local_plot_mask = plot_mask[section_mask]
            figure_path = directory / f"spatial_{section}.png"
            plot_spatial(
                figure_path,
                data.coords[section_mask][local_plot_mask],
                section_labels[local_plot_mask],
                f"{data.name} {scheme} joint K={k} {section}",
                k,
            )
            figure_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "plot_n_obs": int(local_plot_mask.sum()),
                    "figure_path": str(figure_path),
                    "figure_sha256": sha256(figure_path),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "plot_sample_path": str(plot_path),
                    "plot_sample_sha256": plot_hash,
                }
            )
            for cluster, count in zip(
                *np.unique(section_labels, return_counts=True)
            ):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(section_mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(
            directory / "cluster_counts.csv", index=False
        )
    del joint_distances

    for section in SECTIONS:
        section_mask = data.sections == section
        section_space, scaler = metric_space(
            data.embedding[section_mask], scheme
        )
        if scaler is not None:
            scaler_arrays.update(scaler_payload(scaler, section))
        local_asw_mask = asw_mask[section_mask]
        local_plot_mask = plot_mask[section_mask]
        local_asw_indices = np.flatnonzero(local_asw_mask)
        local_distances = pairwise_distances(
            section_space[local_asw_indices],
            metric="euclidean",
            n_jobs=-1,
        )
        np.fill_diagonal(local_distances, 0.0)
        for k in KS:
            labels, model = fit_minibatch(section_space, k)
            directory = clustering_root / f"independent_k{k}"
            directory.mkdir(exist_ok=True)
            labels_path = directory / f"labels_{section}.csv"
            labels_sha = save_labels(
                labels_path, data, section_mask, labels
            )
            model_directory = directory / section
            model_directory.mkdir()
            save_model(model_directory, model, labels_path)
            metric_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(section_mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **internal_metrics(
                        section_space,
                        labels,
                        local_asw_indices,
                        local_distances,
                    ),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "section_asw_n_obs": 0,
                    "metric_space": scheme,
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "asw_sample_path": str(asw_path),
                    "asw_sample_sha256": asw_hash,
                }
            )
            graph_path = (
                data.output_root
                / "shared_metrics"
                / f"spatial_neighbors_{section}.npz"
            )
            spatial_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "n_obs": int(section_mask.sum()),
                    "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction":
                    spatial_agreement_from_graph(labels, graphs[section]),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "spatial_graph_path": str(graph_path),
                    "spatial_graph_sha256": sha256(graph_path),
                }
            )
            figure_path = directory / f"spatial_{section}.png"
            plot_spatial(
                figure_path,
                data.coords[section_mask][local_plot_mask],
                labels[local_plot_mask],
                f"{data.name} {scheme} independent K={k} {section}",
                k,
            )
            figure_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "plot_n_obs": int(local_plot_mask.sum()),
                    "figure_path": str(figure_path),
                    "figure_sha256": sha256(figure_path),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "plot_sample_path": str(plot_path),
                    "plot_sample_sha256": plot_hash,
                }
            )
        del local_distances

    for k in KS:
        rows = []
        for section in SECTIONS:
            labels = pd.read_csv(
                clustering_root
                / f"independent_k{k}"
                / f"labels_{section}.csv",
                usecols=["cluster"],
            )["cluster"].to_numpy()
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / SECTION_COUNTS[section],
                    }
                )
        pd.DataFrame(rows).to_csv(
            clustering_root / f"independent_k{k}" / "cluster_counts.csv",
            index=False,
        )

    metrics = pd.DataFrame(metric_rows)
    spatial = pd.DataFrame(spatial_rows)
    figures = pd.DataFrame(figure_rows)
    metrics_path = metrics_root / "clustering_metrics.csv"
    spatial_path = metrics_root / "spatial_continuity.csv"
    spatial_summary_path = metrics_root / "spatial_continuity_summary.csv"
    figures_path = metrics_root / "figure_manifest.csv"
    metrics.to_csv(metrics_path, index=False)
    spatial.to_csv(spatial_path, index=False)
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
        .to_csv(spatial_summary_path, index=False)
    )
    figures.to_csv(figures_path, index=False)
    if scaler_arrays:
        np.savez_compressed(
            output / "scaler_parameters.npz", **scaler_arrays
        )

    config = {
        "dataset": "CRC_Stereo-CITE-seq",
        "method": data.name,
        "preprocessing": scheme,
        "kmeans_input": (
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
            "none" if scheme == "raw_embedding" else "all_612374_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "clustering_class": "sklearn.cluster.MiniBatchKMeans",
        "k_values": KS,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "batch_size": BATCH_SIZE,
        "n_obs": N_OBS,
        "section_counts": SECTION_COUNTS,
        "embedding_dim": data.embedding.shape[1],
        "cluster_asw_sample_size_joint": ASW_N_OBS,
        "cluster_asw_sample_size_independent": ASW_SECTION_COUNTS,
        "cluster_asw_sample_file": str(asw_path),
        "cluster_asw_sample_sha256": asw_hash,
        "ch_dbi_rule": "full_n_obs_in_each_scope",
        "ch_dbi_sample_size": 0,
        "label_asw": "not_applicable_no_ground_truth_labels",
        "plot_sample_size_per_section": PLOT_SECTION_COUNTS,
        "plot_sample_file": str(plot_path),
        "plot_sample_sha256": plot_hash,
        "plot_selection": (
            "all spots in each section; rows stored in deterministic "
            "SHA256(barcode) order"
        ),
        "full_spot_spatial_plots": True,
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "exact_kd_tree_within_each_section",
        "spatial_graph_n_obs": SECTION_COUNTS,
        "full_labels_saved_for_all_methods_modes_and_k": True,
        "metrics_and_figures_derived_directly_from_saved_labels": True,
        "report_derived_metrics_chain": False,
        "source_data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in data.source_paths
        ],
        "generation_script": str(SCRIPT_PATH),
        "generation_script_sha256": sha256(SCRIPT_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    provenance_rows = []
    for role, path in [
        ("config", config_path),
        ("clustering_metrics", metrics_path),
        ("spatial_continuity", spatial_path),
        ("spatial_continuity_summary", spatial_summary_path),
        ("figure_manifest", figures_path),
        ("asw_sample", asw_path),
        ("plot_sample", plot_path),
    ]:
        provenance_rows.append(
            {"role": role, "path": str(path), "sha256": sha256(path)}
        )
    pd.DataFrame(provenance_rows).to_csv(
        metrics_root / "direct_provenance_manifest.csv", index=False
    )
    (output / "SUMMARY.md").write_text(
        f"# {data.name}: {scheme}\n\n"
        f"- spots: {N_OBS}; dimensions: {data.embedding.shape[1]}\n"
        f"- MiniBatchKMeans: seed={SEED}, n_init={N_INIT}, "
        f"max_iter={MAX_ITER}, batch_size={BATCH_SIZE}\n"
        f"- joint/independent full labels, metrics and figures: K={KS}\n"
        f"- Cluster ASW: fixed section-stratified {ASW_N_OBS}-spot sample\n"
        "- CH/DBI: full scope; Label ASW: not applicable\n"
        f"- figures: all spots (section counts: {PLOT_SECTION_COUNTS})\n"
        "- metrics and figures are generated directly from saved labels\n"
        "- no report-derived metric chain is used\n"
    )


def redraw_full_spot_figures(
    data: MethodData,
    plot_universe: pd.DataFrame,
) -> None:
    """Redraw existing clustering labels without refitting or recomputing metrics."""
    shared = data.output_root / "shared_metrics"
    plot_path = shared / "plot_sample.csv"
    plot_universe.to_csv(plot_path, index=False)
    plot_hash = sha256(plot_path)

    sampling_path = shared / "sampling_manifest.json"
    sampling = json.loads(sampling_path.read_text())
    sampling["plot_sample"] = {
        "path": str(plot_path),
        "sha256": plot_hash,
        "n_obs": N_OBS,
        "section_counts": PLOT_SECTION_COUNTS,
        "selection": (
            "all spots in each section; rows stored in deterministic "
            "SHA256(barcode) order"
        ),
    }
    sampling["shared_across_all_methods_preprocessing_modes_and_k"] = True
    sampling["figures_use_all_spots"] = True
    sampling["figures_redrawn_at"] = (
        datetime.now().astimezone().isoformat()
    )
    sampling_path.write_text(
        json.dumps(sampling, indent=2, ensure_ascii=False) + "\n"
    )

    plot_mask = key_mask(data, plot_universe)
    if int(plot_mask.sum()) != N_OBS or not bool(np.all(plot_mask)):
        raise ValueError(f"{data.name}: full plot universe mapping failed")

    for scheme in ["raw_embedding", "standardized_embedding"]:
        output = data.output_root / scheme
        clustering_root = output / "clustering"
        metrics_root = output / "metrics"
        figure_rows: list[dict[str, Any]] = []
        for mode in ["joint", "independent"]:
            for k in KS:
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
                        f"{section}: n={len(labels)}",
                        flush=True,
                    )
                    plot_spatial(
                        figure_path,
                        labels[["x", "y"]].to_numpy(float),
                        labels["cluster"].to_numpy(int),
                        f"{data.name} {scheme} {mode} K={k} {section}",
                        k,
                    )
                    figure_rows.append(
                        {
                            "mode": mode,
                            "k": k,
                            "section": section,
                            "plot_n_obs": len(labels),
                            "plot_scope": "all_spots_in_section",
                            "figure_path": str(figure_path),
                            "figure_sha256": sha256(figure_path),
                            "labels_path": str(labels_path),
                            "labels_sha256": sha256(labels_path),
                            "plot_sample_path": str(plot_path),
                            "plot_sample_sha256": plot_hash,
                        }
                    )

        figures_path = metrics_root / "figure_manifest.csv"
        pd.DataFrame(figure_rows).to_csv(figures_path, index=False)

        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "plot_sample_size_per_section": PLOT_SECTION_COUNTS,
                "plot_sample_file": str(plot_path),
                "plot_sample_sha256": plot_hash,
                "plot_selection": (
                    "all spots in each section; rows stored in deterministic "
                    "SHA256(barcode) order"
                ),
                "full_spot_spatial_plots": True,
                "plot_n_obs_total": N_OBS,
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
                f"- figures: all spots (section counts: "
                f"{PLOT_SECTION_COUNTS})"
                if line.startswith("- figures:")
                else line
            )
            for line in summary_lines
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n")

        provenance_path = metrics_root / "direct_provenance_manifest.csv"
        provenance = pd.read_csv(provenance_path)
        refreshed = {
            "config": config_path,
            "figure_manifest": figures_path,
            "plot_sample": plot_path,
        }
        for role, path in refreshed.items():
            mask = provenance["role"].eq(role)
            if not mask.any():
                provenance = pd.concat(
                    [
                        provenance,
                        pd.DataFrame(
                            [{"role": role, "path": str(path), "sha256": sha256(path)}]
                        ),
                    ],
                    ignore_index=True,
                )
            else:
                provenance.loc[mask, "path"] = str(path)
                provenance.loc[mask, "sha256"] = sha256(path)
        provenance.to_csv(provenance_path, index=False)


def compare_schemes(data: MethodData) -> None:
    output = data.output_root / "comparison_metrics"
    output.mkdir()
    stability_rows = []
    for mode in ["joint", "independent"]:
        scopes = ["combined"] if mode == "joint" else SECTIONS
        for k in KS:
            for scope in scopes:
                relative = (
                    Path("clustering") / f"joint_k{k}" / "labels_all.csv"
                    if mode == "joint"
                    else Path("clustering")
                    / f"independent_k{k}"
                    / f"labels_{scope}.csv"
                )
                raw = pd.read_csv(
                    data.output_root / "raw_embedding" / relative,
                    usecols=["section", "obs_name", "cluster"],
                    dtype={"section": str, "obs_name": str},
                )
                standardized = pd.read_csv(
                    data.output_root / "standardized_embedding" / relative,
                    usecols=["section", "obs_name", "cluster"],
                    dtype={"section": str, "obs_name": str},
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
                        "raw_labels_path": str(
                            data.output_root / "raw_embedding" / relative
                        ),
                        "raw_labels_sha256": sha256(
                            data.output_root / "raw_embedding" / relative
                        ),
                        "standardized_labels_path": str(
                            data.output_root
                            / "standardized_embedding"
                            / relative
                        ),
                        "standardized_labels_sha256": sha256(
                            data.output_root
                            / "standardized_embedding"
                            / relative
                        ),
                    }
                )
    stability_path = output / "label_stability.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)

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
    delta_path = output / "metric_deltas.csv"
    merged.to_csv(delta_path, index=False)
    pd.DataFrame(
        [
            {
                "role": "label_stability",
                "path": str(stability_path),
                "sha256": sha256(stability_path),
            },
            {
                "role": "metric_deltas",
                "path": str(delta_path),
                "sha256": sha256(delta_path),
            },
        ]
    ).to_csv(output / "direct_provenance_manifest.csv", index=False)


def write_manifest(data: MethodData) -> None:
    shared = data.output_root / "shared_metrics"
    asw_path = shared / "asw_sample.csv"
    plot_path = shared / "plot_sample.csv"
    payload = {
        "dataset": "CRC_Stereo-CITE-seq",
        "method": data.name,
        "legacy_results_modified": False,
        "report_derived_metrics_chain": False,
        "comparison_root": str(data.output_root),
        "branches": {
            "raw_embedding": str(data.output_root / "raw_embedding"),
            "standardized_embedding": str(
                data.output_root / "standardized_embedding"
            ),
        },
        "comparison_metrics": str(data.output_root / "comparison_metrics"),
        "shared_metrics": str(shared),
        "asw_sample_sha256": sha256(asw_path),
        "plot_sample_sha256": sha256(plot_path),
        "full_spot_spatial_plots": True,
        "plot_section_counts": PLOT_SECTION_COUNTS,
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
    asw_hashes = {
        sha256(m.output_root / "shared_metrics" / "asw_sample.csv")
        for m in methods
    }
    plot_hashes = {
        sha256(m.output_root / "shared_metrics" / "plot_sample.csv")
        for m in methods
    }
    if len(asw_hashes) != 1 or len(plot_hashes) != 1:
        raise ValueError("cross-method sample hashes differ")
    asw_hash = next(iter(asw_hashes))
    plot_hash = next(iter(plot_hashes))
    lines = [
        "# CRC_Stereo-CITE-seq：Raw 与 Standardized embedding + MiniBatchKMeans 对比报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        "- 本报告只读取统一脚本直接生成的labels、指标CSV和provenance manifest；不使用 `report_derived_metrics/` 或其他报告派生计算链。",
        "- 本次结果位于新增comparison目录，旧分析结果和原综合报告未覆盖。",
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
                ["spot数", "612374（CRC_003=166279，CRC_006=446095）"],
                ["raw分支", "原始最终embedding直接输入MiniBatchKMeans"],
                ["standardized分支", "StandardScaler；joint全体拟合，independent每section拟合"],
                ["聚类器", "sklearn.cluster.MiniBatchKMeans"],
                ["K", "5、10、15、20、25；joint与independent一致"],
                ["随机及迭代参数", "seed=0，n_init=20，max_iter=300，batch_size=8192"],
                ["Cluster ASW", "固定同一批section分层10000个spot"],
                ["ASW分层数", "CRC_003=2716，CRC_006=7284"],
                ["CH/DBI", "每个scope全量样本"],
                ["Label ASW", "无真实细胞/区域标签，不计算"],
                ["空间连续性", "每section全点，exact kd-tree 6-NN"],
                [
                    "空间图",
                    "每section使用全量spot（CRC_003=166279，CRC_006=446095）",
                ],
                ["labels", "全部方法、模式、分支和K均保存全量labels"],
                ["可追溯性", "指标与图直接引用labels路径及SHA256；无报告派生链"],
            ],
        ),
        "",
        "## 3. 固定ASW抽样与全量绘图位置一致性",
        "",
        md_table(
            ["用途", "CRC_003", "CRC_006", "总数", "跨方法SHA256"],
            [
                ["Cluster ASW", 2716, 7284, 10000, asw_hash],
                ["空间绘图", 166279, 446095, 612374, plot_hash],
            ],
        ),
        "",
        "ASW名单为每section按 `SHA256('asw_seed0|' + barcode)` 排序后截取；空间图使用两个section的全部spot。`plot_sample.csv`现作为全量绘图spot清单，按 `SHA256(barcode)` 确定性排序。四种方法的清单文件哈希完全相同，且全部612374个坐标已按barcode逐点核对一致。",
        "",
        "## 4. 数据与 embedding 对齐",
        "",
        md_table(
            ["方法", "维度", "CRC_003", "CRC_006", "总spot"],
            [
                [
                    m.name,
                    m.embedding.shape[1],
                    int((m.sections == SECTIONS[0]).sum()),
                    int((m.sections == SECTIONS[1]).sum()),
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
        "## 5. Joint内部与空间指标：Raw / Standardized上下对齐",
        "",
        "同一方法、同一K的Raw与Standardized行相邻；表中值直接读取各分支 `metrics/clustering_metrics.csv` 和 `spatial_continuity_summary.csv`。",
        "",
    ]
    metric_rows = []
    for method in methods:
        for k in KS:
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
        "joint和independent完整指标均覆盖K=5/10/15/20/25；各分支 `metrics/direct_provenance_manifest.csv` 记录指标文件、抽样文件和配置的SHA256。",
        "",
        "### 5.1 Joint全K最佳内部指标",
        "",
        "下表只对现有K=5/10/15/20/25直接选择最优CSV行，不增加K=8，也不生成新指标。",
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
        "### 5.2 Independent逐section内部指标",
        "",
        "同一方法、同一K、同一section的Raw与Standardized行严格相邻；ASW使用固定分层样本在对应section中的子集，CH/DBI使用该section全量spot。",
        "",
    ]
    independent_rows = []
    for method in methods:
        for k in KS:
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
                            SHORT_NAMES[section],
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
        "### 5.3 Joint与Independent空间连续性",
        "",
        "每个值均为两个section全量spot的空间近邻同簇比例算术平均；Raw与Standardized行严格相邻。",
        "",
    ]
    spatial_rows = []
    for method in methods:
        for k in KS:
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
        stability = stability[stability["mode"] == "joint"]
        for _, row in stability.iterrows():
            stability_rows.append(
                [
                    method.name,
                    int(row["k"]),
                    int(row["n_obs"]),
                    f4(row["raw_vs_standardized_ari"]),
                    f4(row["raw_vs_standardized_nmi"]),
                ]
            )
    lines += [
        md_table(
            ["方法", "K", "全量n", "raw vs standardized ARI", "NMI"],
            stability_rows,
        ),
        "",
        "完整joint/independent稳定性及两套labels的路径和SHA256见各方法 `comparison_metrics/label_stability.csv`。",
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
                int(batch.get("n_obs_total", batch.get("n_obs", N_OBS))),
                int(batch.get("n_used", batch.get("n_obs", N_OBS))),
                int(batch.get("n_batches", 2)),
                str(batch.get("batch_categories", "NA")),
                str(batch.get("batch_counts", "NA")),
                int(batch.get("max_samples", 0)),
                int(batch.get("sample_size_requested", 0)),
                int(batch.get("seed", 0)),
                str(batch.get("embedding_scaled", "NA")),
                int(batch.get("bASW_sample_size", ASW_N_OBS)),
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
        "batch指标不随本次KMeans输入分支变化，故只保存一份；原文件与复制文件SHA256见 `shared_metrics/source_manifest.csv`。",
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
        "## 8. 结论与使用建议",
        "",
        "1. Raw与Standardized均作为完整结果保存；跨方法排名必须固定同一预处理分支。",
        "2. Standardized可作为尺度可比的主分析，Raw作为敏感性分析；同时报告全量labels的raw-vs-standardized ARI/NMI。",
        "3. 所有内部、空间指标和图片均由统一脚本直接读取本次保存的labels生成，并记录labels SHA256；报告不再连接任何派生式计算链。",
        "4. CRC数据没有统一真实区域标签；section ARI/NMI是样本来源依赖诊断，不是生物学聚类准确率。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate only the report from existing direct results.",
    )
    actions.add_argument(
        "--redraw-full-spot-plots",
        action="store_true",
        help=(
            "Redraw all existing CRC comparison clustering figures with "
            "every spot; do not refit clustering or recompute metrics."
        ),
    )
    args = parser.parse_args()
    methods = [load_spa(), load_mofa(), load_cosie(), load_spamosaic()]
    if args.report_only:
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    if args.redraw_full_spot_plots:
        for method in methods:
            validate(method, allow_existing_output=True)
        validate_cross_method_alignment(methods)
        plot_universe = select_by_barcode_hash(
            methods[0], PLOT_SECTION_COUNTS, ""
        )
        for method in methods:
            redraw_full_spot_figures(method, plot_universe)
            write_manifest(method)
        build_report(methods, overwrite=True)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    for method in methods:
        validate(method)
    validate_cross_method_alignment(methods)
    if REPORT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite report {REPORT_PATH}")
    asw_sample = select_by_barcode_hash(
        methods[0], ASW_SECTION_COUNTS, "asw_seed0|"
    )
    plot_sample = select_by_barcode_hash(
        methods[0], PLOT_SECTION_COUNTS, ""
    )
    for method in methods:
        print(f"[{method.name}] prepare shared artifacts", flush=True)
        graphs = prepare_shared(method, asw_sample, plot_sample)
        print(f"[{method.name}] raw_embedding", flush=True)
        run_scheme(
            method,
            "raw_embedding",
            asw_sample,
            plot_sample,
            graphs,
        )
        print(f"[{method.name}] standardized_embedding", flush=True)
        run_scheme(
            method,
            "standardized_embedding",
            asw_sample,
            plot_sample,
            graphs,
        )
        compare_schemes(method)
        write_manifest(method)
    build_report(methods)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
