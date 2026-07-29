#!/usr/bin/env python3
"""Independent raw-vs-standardized MiniBatchKMeans comparison for spatch.

Only spa_mo_model and COSIE are included because they are the two methods with
successful full-spot spatch embeddings.  Every method reads its own run metadata
and its own final embeddings.  Original training/analysis directories are
strictly read-only.

The script is resumable while a scheme is running.  K=2..20 are fitted and
evaluated, while only K=5/8/10/12/16/20 receive individual result directories
and full-spot plots.  All K labels are retained in one compressed archive.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass
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
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SECTIONS = ["section1", "section2"]
SECTION_COUNTS = {"section1": 665399, "section2": 403563}
N_OBS = 1068962
LABELS = ["cell_type_common", "spatial_cluster", "codex_coarse_label"]
ALL_KS = list(range(2, 21))
RETAIN_KS = [5, 8, 10, 12, 16, 20]
SEED = 42
N_INIT = 20
MAX_ITER = 300
BATCH_SIZE = 4096
ASW_SAMPLE_SIZE = 10000
SPATIAL_NEIGHBOR_K = 6
POINT_SIZE = 0.2
DPI = 220

SCRIPT_PATH = Path(__file__).resolve()
REPORT_PATH = Path(
    "/home/hujinlan/spa_mo_model/spatch_preprocessing_comparison_report.md"
)


@dataclass
class MethodSpec:
    key: str
    name: str
    run_dir: Path
    data_dir: Path
    embedding_paths: dict[str, Path]
    metadata_path: Path
    id_column: str
    output_root: Path
    old_analysis: Path
    batch_metrics_source: Path


METHODS = {
    "spa": MethodSpec(
        key="spa",
        name="spa_mo_model",
        run_dir=Path(
            "/home/hujinlan/spa_mo_model/results/spatch/"
            "fullspot_200ep_gpu_seed42"
        ),
        data_dir=Path("/home/hujinlan/spa_mo_model/data/spatch"),
        embedding_paths={
            "section1": Path(
                "/home/hujinlan/spa_mo_model/results/spatch/"
                "fullspot_200ep_gpu_seed42/final_embeddings_section1.npy"
            ),
            "section2": Path(
                "/home/hujinlan/spa_mo_model/results/spatch/"
                "fullspot_200ep_gpu_seed42/final_embeddings_section2.npy"
            ),
        },
        metadata_path=Path(
            "/home/hujinlan/spa_mo_model/results/spatch/"
            "fullspot_200ep_gpu_seed42/spot_metadata.csv.gz"
        ),
        id_column="block_id",
        output_root=Path(
            "/home/hujinlan/spa_mo_model/results/"
            "spatch_preprocessing_comparison"
        ),
        old_analysis=Path(
            "/home/hujinlan/spa_mo_model/results/spatch/"
            "fullspot_200ep_gpu_seed42/clustering_analysis"
        ),
        batch_metrics_source=Path(
            "/home/hujinlan/spa_mo_model/results/spatch/"
            "fullspot_200ep_gpu_seed42/clustering_analysis/"
            "batch_correction_metrics.csv"
        ),
    ),
    "cosie": MethodSpec(
        key="cosie",
        name="COSIE",
        run_dir=Path(
            "/home/hujinlan/cosie_runs/"
            "spatch_cosie_rna_protein_he_metacell_6x6"
        ),
        data_dir=Path("/home/hujinlan/cosie/data/spatch"),
        embedding_paths={
            "section1": Path(
                "/home/hujinlan/cosie_runs/"
                "spatch_cosie_rna_protein_he_metacell_6x6/"
                "final_embeddings/section1_final_embedding.npy"
            ),
            "section2": Path(
                "/home/hujinlan/cosie_runs/"
                "spatch_cosie_rna_protein_he_metacell_6x6/"
                "final_embeddings/section2_final_embedding.npy"
            ),
        },
        metadata_path=Path(
            "/home/hujinlan/cosie_runs/"
            "spatch_cosie_rna_protein_he_metacell_6x6/"
            "spot_metadata.csv.gz"
        ),
        id_column="spot_id",
        output_root=Path(
            "/home/hujinlan/cosie_runs/spatch_preprocessing_comparison"
        ),
        old_analysis=Path(
            "/home/hujinlan/cosie_runs/"
            "spatch_cosie_rna_protein_he_metacell_6x6/analysis"
        ),
        batch_metrics_source=Path(
            "/home/hujinlan/cosie_runs/"
            "spatch_cosie_rna_protein_he_metacell_6x6/"
            "analysis/metrics/batch_correction_metrics.csv"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["raw_embedding", "standardized_embedding"],
        default=["raw_embedding", "standardized_embedding"],
    )
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(spec: MethodSpec) -> pd.DataFrame:
    usecols = [
        spec.id_column,
        "section",
        "original_rna_obs_name",
        "x",
        "y",
        *LABELS,
    ]
    frame = pd.read_csv(
        spec.metadata_path,
        usecols=usecols,
        dtype={
            spec.id_column: str,
            "section": str,
            "original_rna_obs_name": str,
        },
        low_memory=False,
    ).rename(columns={spec.id_column: "spot_id"})
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["section"] = frame["section"].astype(str)
    frame["x"] = pd.to_numeric(frame["x"], errors="raise").astype(np.float32)
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(np.float32)
    return frame


def load_embedding(spec: MethodSpec) -> np.ndarray:
    arrays = []
    for section in SECTIONS:
        array = np.load(spec.embedding_paths[section], mmap_mode="r")
        if array.shape[0] != SECTION_COUNTS[section]:
            raise ValueError(
                f"{spec.name} {section}: unexpected embedding shape {array.shape}"
            )
        arrays.append(np.asarray(array))
    return np.vstack(arrays)


def validate_metadata(spec: MethodSpec, metadata: pd.DataFrame) -> None:
    if len(metadata) != N_OBS:
        raise ValueError(f"{spec.name}: expected {N_OBS}, got {len(metadata)}")
    counts = metadata["section"].value_counts().to_dict()
    if counts != SECTION_COUNTS:
        raise ValueError(f"{spec.name}: section counts differ: {counts}")
    if metadata[["section", "spot_id"]].duplicated().any():
        raise ValueError(f"{spec.name}: duplicate section/spot_id")
    if not np.isfinite(metadata[["x", "y"]].to_numpy()).all():
        raise ValueError(f"{spec.name}: non-finite coordinates")
    for path in [
        *spec.embedding_paths.values(),
        spec.metadata_path,
        spec.batch_metrics_source,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)


def validate_cross_method() -> None:
    spa = load_metadata(METHODS["spa"])
    cosie = load_metadata(METHODS["cosie"])
    columns = ["section", "spot_id", "original_rna_obs_name", *LABELS]
    for column in columns:
        left = spa[column].astype("string").fillna("<NA>").to_numpy()
        right = cosie[column].astype("string").fillna("<NA>").to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"cross-method metadata mismatch: {column}")
    if not np.allclose(
        spa[["x", "y"]].to_numpy(), cosie[["x", "y"]].to_numpy()
    ):
        raise ValueError("cross-method coordinate mismatch")


def fixed_sample_indices(n: int) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return np.sort(rng.permutation(n)[: min(ASW_SAMPLE_SIZE, n)])


def sample_manifest(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    joint = fixed_sample_indices(len(metadata))
    for order, index in enumerate(joint):
        rows.append(
            {
                "scope": "combined",
                "sample_order": order,
                "row_index_within_scope": int(index),
                "section": metadata.iloc[index]["section"],
                "spot_id": metadata.iloc[index]["spot_id"],
            }
        )
    for section in SECTIONS:
        mask = metadata["section"].eq(section).to_numpy()
        local = metadata.loc[mask].reset_index(drop=True)
        selected = fixed_sample_indices(len(local))
        for order, index in enumerate(selected):
            rows.append(
                {
                    "scope": section,
                    "sample_order": order,
                    "row_index_within_scope": int(index),
                    "section": section,
                    "spot_id": local.iloc[index]["spot_id"],
                }
            )
    return pd.DataFrame(rows)


def build_spatial_graph(coords: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=SPATIAL_NEIGHBOR_K + 1,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(coords)
    raw = model.kneighbors(coords, return_distance=False)
    row = np.arange(len(coords))[:, None]
    result = raw[raw != row].reshape(len(coords), SPATIAL_NEIGHBOR_K)
    return result.astype(np.int32)


def prepare_shared(
    spec: MethodSpec,
    metadata: pd.DataFrame,
    reference_root: Path | None,
) -> dict[str, np.ndarray]:
    shared = spec.output_root / "shared_metrics"
    shared.mkdir(parents=True, exist_ok=True)
    sample_path = shared / "asw_sample_identities.csv"
    if reference_root is None:
        sample_manifest(metadata).to_csv(sample_path, index=False)
    else:
        shutil.copy2(
            reference_root / "shared_metrics" / "asw_sample_identities.csv",
            sample_path,
        )
    shutil.copy2(
        spec.batch_metrics_source,
        shared / "batch_correction_metrics.csv",
    )
    copy_rows = [
        {
            "artifact": "batch_correction_metrics.csv",
            "source": str(spec.batch_metrics_source),
            "source_sha256": sha256(spec.batch_metrics_source),
            "copied_sha256": sha256(shared / "batch_correction_metrics.csv"),
            "kmeans_preprocessing_dependent": False,
        }
    ]
    pd.DataFrame(copy_rows).to_csv(shared / "source_manifest.csv", index=False)

    graphs = {}
    graph_rows = []
    for section in SECTIONS:
        path = shared / f"spatial_neighbors_{section}.npz"
        if reference_root is None:
            mask = metadata["section"].eq(section).to_numpy()
            graph = build_spatial_graph(
                metadata.loc[mask, ["x", "y"]].to_numpy()
            )
            np.savez_compressed(path, neighbor_indices=graph)
        else:
            shutil.copy2(
                reference_root
                / "shared_metrics"
                / f"spatial_neighbors_{section}.npz",
                path,
            )
            graph = np.load(path)["neighbor_indices"]
        graphs[section] = graph
        graph_rows.append(
            {
                "section": section,
                "n_obs": SECTION_COUNTS[section],
                "neighbor_k": SPATIAL_NEIGHBOR_K,
                "algorithm": "sklearn NearestNeighbors kd_tree exact",
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(graph_rows).to_csv(
        shared / "spatial_graph_manifest.csv", index=False
    )
    return graphs


def metric_space(
    raw: np.ndarray, scheme: str
) -> tuple[np.ndarray, StandardScaler | None]:
    if scheme == "raw_embedding":
        return raw, None
    scaler = StandardScaler(copy=True)
    return scaler.fit_transform(raw), scaler


def scaler_payload(
    scaler: StandardScaler, prefix: str
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": scaler.mean_,
        f"{prefix}_scale": scaler.scale_,
        f"{prefix}_var": scaler.var_,
    }


def fit_model(space: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
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
    labels = model.fit_predict(space).astype(np.int16)
    info = {
        "k": k,
        "n_iter": int(model.n_iter_),
        "n_steps": int(model.n_steps_),
        "inertia": float(model.inertia_),
    }
    return labels, info, model.cluster_centers_


def fit_or_load(
    cache_path: Path,
    info_path: Path,
    centers_path: Path,
    space: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cache_path.exists() and info_path.exists():
        return np.load(cache_path), json.loads(info_path.read_text())
    labels, info, centers = fit_model(space, k)
    np.save(cache_path, labels)
    np.save(centers_path, centers)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return labels, info


def external_score(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "homogeneity": float(homogeneity_score(truth, labels)),
        "completeness": float(completeness_score(truth, labels)),
        "v_measure": float(v_measure_score(truth, labels)),
    }


def internal_score(
    space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
) -> dict[str, float | int]:
    sampled_labels = labels[sample_indices]
    asw = float(
        silhouette_score(
            sample_distances,
            sampled_labels,
            metric="precomputed",
        )
    )
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "cluster_asw_n_obs": len(sample_indices),
        "calinski_harabasz": float(
            calinski_harabasz_score(space, labels)
        ),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
        "ch_dbi_n_obs": len(labels),
    }


def label_asw(
    space: np.ndarray,
    truth: pd.Series,
    metadata: pd.DataFrame,
    scope: str,
    label: str,
) -> tuple[float, pd.DataFrame]:
    valid = truth.notna().to_numpy()
    valid_indices = np.flatnonzero(valid)
    values = truth.loc[valid].astype(str).to_numpy()
    if len(np.unique(values)) < 2:
        return float("nan"), pd.DataFrame()
    selected_local = fixed_sample_indices(len(valid_indices))
    selected = valid_indices[selected_local]
    asw = float(silhouette_score(space[selected], values[selected_local]))
    identities = metadata.iloc[selected][["section", "spot_id"]].copy()
    identities.insert(0, "label", label)
    identities.insert(0, "scope", scope)
    identities.insert(2, "sample_order", np.arange(len(identities)))
    return asw, identities


def spatial_agreement(labels: np.ndarray, graph: np.ndarray) -> float:
    return float(np.mean(labels[graph] == labels[:, None]))


def save_section_labels(
    path: Path,
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "spot_id": metadata["spot_id"].astype(str),
            "cluster": labels.astype(int),
        }
    ).to_csv(path, index=False)


def plot_spatial(
    path: Path,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        metadata["x"],
        metadata["y"],
        c=labels,
        s=POINT_SIZE,
        cmap=plt.get_cmap("tab20", min(k, 20)),
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
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def retained_outputs(
    scheme_root: Path,
    metadata: pd.DataFrame,
    mode: str,
    scope: str,
    k: int,
    labels: np.ndarray,
    centers_path: Path,
) -> None:
    directory = scheme_root / "clustering" / f"{mode}_k{k}"
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "joint":
        for section in SECTIONS:
            mask = metadata["section"].eq(section).to_numpy()
            section_meta = metadata.loc[mask].reset_index(drop=True)
            path = directory / f"labels_{section}.csv"
            save_section_labels(path, section_meta, labels[mask])
            plot_spatial(
                directory / f"spatial_{section}.png",
                section_meta,
                labels[mask],
                f"{scheme_root.parent.name} {scheme_root.name} joint K={k} {section}",
                k,
            )
    else:
        section_meta = metadata.reset_index(drop=True)
        path = directory / f"labels_{scope}.csv"
        save_section_labels(path, section_meta, labels)
        plot_spatial(
            directory / f"spatial_{scope}.png",
            section_meta,
            labels,
            f"{scheme_root.parent.name} {scheme_root.name} independent K={k} {scope}",
            k,
        )
    target_centers = directory / f"cluster_centers_{scope}.npy"
    if not target_centers.exists():
        shutil.copy2(centers_path, target_centers)


def archive_cache(cache_root: Path, output_path: Path) -> None:
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for path in sorted(cache_root.glob("*")):
            archive.write(path, arcname=path.name)
    shutil.rmtree(cache_root)


def run_scheme(
    spec: MethodSpec,
    metadata: pd.DataFrame,
    embedding: np.ndarray,
    graphs: dict[str, np.ndarray],
    scheme: str,
) -> None:
    output = spec.output_root / scheme
    completion = output / "analysis_completion_manifest.json"
    if completion.exists():
        print(f"[{spec.name}] {scheme}: already complete", flush=True)
        return
    metrics_dir = output / "metrics"
    clustering_dir = output / "clustering"
    cache_root = output / "_label_cache"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(exist_ok=True)
    cache_root.mkdir(exist_ok=True)

    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    label_sample_rows: list[pd.DataFrame] = []
    scaler_arrays: dict[str, np.ndarray] = {}

    scopes = [("joint", "combined", np.ones(N_OBS, dtype=bool))]
    scopes.extend(
        (
            "independent",
            section,
            metadata["section"].eq(section).to_numpy(),
        )
        for section in SECTIONS
    )

    for mode, scope, mask in scopes:
        print(f"[{spec.name}] {scheme} {mode}:{scope} prepare metric space", flush=True)
        raw = embedding if mode == "joint" else embedding[mask]
        space, scaler = metric_space(raw, scheme)
        if scaler is not None:
            scaler_arrays.update(scaler_payload(scaler, scope))
        local_meta = metadata if mode == "joint" else metadata.loc[mask].reset_index(drop=True)
        sample_indices = fixed_sample_indices(len(space))
        sample_distances = pairwise_distances(
            space[sample_indices], metric="euclidean", n_jobs=-1
        )
        np.fill_diagonal(sample_distances, 0.0)

        label_asw_values = {}
        for label in LABELS:
            value, identities = label_asw(
                space,
                local_meta[label],
                local_meta,
                scope,
                label,
            )
            label_asw_values[label] = value
            if not identities.empty:
                label_sample_rows.append(identities)

        for k in ALL_KS:
            prefix = f"{mode}_{scope}_k{k}"
            cache_path = cache_root / f"{prefix}.npy"
            info_path = cache_root / f"{prefix}.json"
            centers_path = cache_root / f"{prefix}_centers.npy"
            print(f"[{spec.name}] {scheme} {prefix}", flush=True)
            labels, fit_info = fit_or_load(
                cache_path,
                info_path,
                centers_path,
                space,
                k,
            )
            fit_rows.append(
                {
                    "mode": mode,
                    "scope": scope,
                    **fit_info,
                    "labels_cache": str(cache_path),
                    "centers_cache": str(centers_path),
                }
            )
            row = {
                "mode": mode,
                "scope": scope,
                "k": k,
                "n_obs": len(labels),
                "embedding_dim": embedding.shape[1],
                **internal_score(
                    space,
                    labels,
                    sample_indices,
                    sample_distances,
                ),
                "metric_space": scheme,
            }
            if mode == "joint":
                row["section_ari"] = adjusted_rand_score(
                    metadata["section"], labels
                )
                row["section_nmi"] = normalized_mutual_info_score(
                    metadata["section"], labels
                )
            else:
                row["section_ari"] = np.nan
                row["section_nmi"] = np.nan
            internal_rows.append(row)

            for label in LABELS:
                valid = local_meta[label].notna().to_numpy()
                if valid.sum() < 2:
                    continue
                truth = local_meta.loc[valid, label].astype(str).to_numpy()
                if len(np.unique(truth)) < 2:
                    continue
                external_rows.append(
                    {
                        "mode": mode,
                        "scope": scope,
                        "k": k,
                        "label": label,
                        "n_labeled": int(valid.sum()),
                        "n_label_classes": len(np.unique(truth)),
                        **external_score(truth, labels[valid]),
                        "label_asw": label_asw_values[label],
                        "label_asw_scaled": (
                            label_asw_values[label] + 1.0
                        )
                        / 2.0,
                        "label_asw_n_obs": min(
                            ASW_SAMPLE_SIZE, int(valid.sum())
                        ),
                        "metric_space": scheme,
                    }
                )

            if mode == "joint":
                offset = 0
                for section in SECTIONS:
                    count = SECTION_COUNTS[section]
                    section_labels = labels[offset : offset + count]
                    spatial_rows.append(
                        {
                            "mode": mode,
                            "scope": scope,
                            "k": k,
                            "section": section,
                            "n_obs": count,
                            "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                            "neighbor_same_cluster_fraction": spatial_agreement(
                                section_labels, graphs[section]
                            ),
                        }
                    )
                    offset += count
            else:
                spatial_rows.append(
                    {
                        "mode": mode,
                        "scope": scope,
                        "k": k,
                        "section": scope,
                        "n_obs": len(labels),
                        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
                        "neighbor_same_cluster_fraction": spatial_agreement(
                            labels, graphs[scope]
                        ),
                    }
                )

            if k in RETAIN_KS:
                retained_outputs(
                    output,
                    local_meta,
                    mode,
                    scope,
                    k,
                    labels,
                    centers_path,
                )
        del sample_distances
        if scaler is not None:
            del space
        gc.collect()

    pd.DataFrame(internal_rows).to_csv(
        metrics_dir / "internal_clustering_metrics.csv", index=False
    )
    pd.DataFrame(external_rows).to_csv(
        metrics_dir / "external_clustering_metrics.csv", index=False
    )
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(metrics_dir / "spatial_continuity.csv", index=False)
    (
        spatial.groupby(["mode", "k"], as_index=False)[
            "neighbor_same_cluster_fraction"
        ]
        .mean()
        .to_csv(metrics_dir / "spatial_continuity_summary.csv", index=False)
    )
    pd.DataFrame(fit_rows).to_csv(
        metrics_dir / "all_k_fit_manifest.csv", index=False
    )
    if label_sample_rows:
        pd.concat(label_sample_rows, ignore_index=True).drop_duplicates().to_csv(
            metrics_dir / "label_asw_sample_identities.csv", index=False
        )
    if scaler_arrays:
        np.savez_compressed(output / "scaler_parameters.npz", **scaler_arrays)

    archive_path = output / "all_k_labels_and_fit_cache.zip"
    archive_cache(cache_root, archive_path)
    config = {
        "dataset": "spatch",
        "method": spec.name,
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
        "clustering_class": "sklearn.cluster.MiniBatchKMeans",
        "all_metric_k": ALL_KS,
        "retained_result_directory_k": RETAIN_KS,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "batch_size": BATCH_SIZE,
        "n_obs": N_OBS,
        "section_counts": SECTION_COUNTS,
        "embedding_dim": embedding.shape[1],
        "cluster_asw_sample_size": ASW_SAMPLE_SIZE,
        "cluster_asw_sampling": (
            "sklearn-compatible RandomState(42) permutation per scope; "
            "identities persisted in shared_metrics/asw_sample_identities.csv"
        ),
        "label_asw_sample_size": ASW_SAMPLE_SIZE,
        "label_asw_sampling": (
            "sklearn-compatible RandomState(42) permutation over valid labels"
        ),
        "ch_dbi_rule": "full_n_obs_in_each_scope",
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "full_spot_spatial_plots": True,
        "point_size": POINT_SIZE,
        "source_data_dir": str(spec.data_dir),
        "source_metadata": {
            "path": str(spec.metadata_path),
            "sha256": sha256(spec.metadata_path),
        },
        "source_embeddings": [
            {"path": str(path), "sha256": sha256(path)}
            for path in spec.embedding_paths.values()
        ],
        "generation_script": str(SCRIPT_PATH),
        "generation_script_sha256": sha256(SCRIPT_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "method": spec.name,
                "scheme": scheme,
                "completed_at": datetime.now().astimezone().isoformat(),
                "all_k_labels_archive": str(archive_path),
                "all_k_labels_archive_sha256": sha256(archive_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def compare_schemes(spec: MethodSpec) -> None:
    output = spec.output_root / "comparison_metrics"
    output.mkdir(exist_ok=True)
    rows = []
    for mode, scopes in (
        ("joint", ["combined"]),
        ("independent", SECTIONS),
    ):
        for scope in scopes:
            for k in ALL_KS:
                member = f"{mode}_{scope}_k{k}.npy"
                values = {}
                for scheme in ("raw_embedding", "standardized_embedding"):
                    archive_path = (
                        spec.output_root
                        / scheme
                        / "all_k_labels_and_fit_cache.zip"
                    )
                    with zipfile.ZipFile(archive_path) as archive:
                        with archive.open(member) as handle:
                            values[scheme] = np.load(handle)
                rows.append(
                    {
                        "mode": mode,
                        "scope": scope,
                        "k": k,
                        "n_obs": len(values["raw_embedding"]),
                        "raw_vs_standardized_ari": adjusted_rand_score(
                            values["raw_embedding"],
                            values["standardized_embedding"],
                        ),
                        "raw_vs_standardized_nmi": normalized_mutual_info_score(
                            values["raw_embedding"],
                            values["standardized_embedding"],
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(
        output / "raw_vs_standardized_label_stability.csv", index=False
    )


def write_root_manifest(spec: MethodSpec) -> None:
    payload = {
        "dataset": "spatch",
        "method": spec.name,
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "legacy_results_modified": False,
        "variants": {
            "raw_embedding": str(spec.output_root / "raw_embedding"),
            "standardized_embedding": str(
                spec.output_root / "standardized_embedding"
            ),
        },
        "shared_metrics": str(spec.output_root / "shared_metrics"),
        "comparison_metrics": str(spec.output_root / "comparison_metrics"),
        "source_data_dir": str(spec.data_dir),
    }
    (spec.output_root / "preprocessing_comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def f4(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    values = [[str(value) for value in row] for row in rows]

    def width(text: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            for char in text
        )

    def padded(text: str, target: int) -> str:
        return text + " " * max(0, target - width(text))

    widths = [
        max(width(headers[i]), max((width(row[i]) for row in values), default=0))
        for i in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(padded(headers[i], widths[i]) for i in range(len(headers)))
        + " |",
        "| "
        + " | ".join("-" * widths[i] for i in range(len(headers)))
        + " |",
    ]
    for row in values:
        lines.append(
            "| "
            + " | ".join(padded(row[i], widths[i]) for i in range(len(headers)))
            + " |"
        )
    return "\n".join(lines)


def read_internal(spec: MethodSpec, scheme: str) -> pd.DataFrame:
    return pd.read_csv(
        spec.output_root
        / scheme
        / "metrics"
        / "internal_clustering_metrics.csv"
    )


def read_external(spec: MethodSpec, scheme: str) -> pd.DataFrame:
    return pd.read_csv(
        spec.output_root
        / scheme
        / "metrics"
        / "external_clustering_metrics.csv"
    )


def metric_matrix(
    specs: list[MethodSpec], scheme: str, metric: str
) -> str:
    rows = []
    for k in RETAIN_KS:
        row = [k]
        for spec in specs:
            frame = read_internal(spec, scheme)
            value = frame.loc[
                (frame["mode"] == "joint") & (frame["k"] == k), metric
            ].iloc[0]
            row.append(f4(value))
        rows.append(row)
    return md_table(["K", *[spec.name for spec in specs]], rows)


def best_external_table(
    specs: list[MethodSpec], scheme: str
) -> str:
    rows = []
    for label in LABELS:
        for spec in specs:
            frame = read_external(spec, scheme)
            internal = read_internal(spec, scheme)
            subset = frame[
                (frame["mode"] == "joint") & (frame["label"] == label)
            ]
            best = subset.loc[subset["ari"].idxmax()]
            best_k = int(best["k"])
            internal_at_best = internal[
                (internal["mode"] == "joint")
                & (internal["scope"] == "combined")
                & (internal["k"] == best_k)
            ].iloc[0]
            rows.append(
                [
                    label,
                    spec.name,
                    best_k,
                    f4(best["ari"]),
                    f4(best["nmi"]),
                    f4(best["homogeneity"]),
                    f4(best["completeness"]),
                    f4(best["v_measure"]),
                    f4(internal_at_best["cluster_asw"]),
                    f4(internal_at_best["cluster_asw_scaled"]),
                    int(internal_at_best["cluster_asw_n_obs"]),
                    f4(best["label_asw"]),
                    f4(best["label_asw_scaled"]),
                    int(best["label_asw_n_obs"]),
                    int(best["n_labeled"]),
                ]
            )
    return md_table(
        [
            "标签",
            "方法",
            "best K",
            "ARI",
            "NMI",
            "Homogeneity",
            "Completeness",
            "V-measure",
            "Cluster ASW raw",
            "Cluster ASW scaled",
            "Cluster ASW n",
            "Label ASW raw",
            "Label ASW scaled",
            "Label ASW n",
            "n labeled",
        ],
        rows,
    )


def spatial_table(
    specs: list[MethodSpec], scheme: str, mode: str
) -> str:
    rows = []
    for k in RETAIN_KS:
        row = [k]
        for spec in specs:
            frame = pd.read_csv(
                spec.output_root
                / scheme
                / "metrics"
                / "spatial_continuity_summary.csv"
            )
            value = frame.loc[
                (frame["mode"] == mode) & (frame["k"] == k),
                "neighbor_same_cluster_fraction",
            ].iloc[0]
            row.append(f4(value))
        rows.append(row)
    return md_table(["K", *[spec.name for spec in specs]], rows)


def stability_table(specs: list[MethodSpec]) -> str:
    rows = []
    for k in RETAIN_KS:
        row = [k]
        for spec in specs:
            frame = pd.read_csv(
                spec.output_root
                / "comparison_metrics"
                / "raw_vs_standardized_label_stability.csv"
            )
            value = frame.loc[
                (frame["mode"] == "joint") & (frame["k"] == k),
                "raw_vs_standardized_ari",
            ].iloc[0]
            row.append(f4(value))
        rows.append(row)
    return md_table(["K", *[spec.name for spec in specs]], rows)


def best_internal_table(specs: list[MethodSpec], scheme: str) -> str:
    rows = []
    for spec in specs:
        frame = read_internal(spec, scheme)
        joint = frame[frame["mode"] == "joint"]
        best_asw = joint.loc[joint["cluster_asw"].idxmax()]
        best_ch = joint.loc[joint["calinski_harabasz"].idxmax()]
        best_dbi = joint.loc[joint["davies_bouldin"].idxmin()]
        rows.append(
            [
                spec.name,
                f"{f4(best_asw['cluster_asw'])} (K={int(best_asw['k'])})",
                f"{f4(best_asw['cluster_asw_scaled'])} (K={int(best_asw['k'])})",
                f"{f4(best_ch['calinski_harabasz'])} (K={int(best_ch['k'])})",
                f"{f4(best_dbi['davies_bouldin'])} (K={int(best_dbi['k'])})",
            ]
        )
    return md_table(
        [
            "方法",
            "best ASW raw",
            "best ASW scaled",
            "best CH",
            "best DBI",
        ],
        rows,
    )


def batch_table(specs: list[MethodSpec]) -> str:
    sampling_rows = []
    parameter_rows = []
    value_rows = []

    def value(row: pd.Series, key: str, fallback: Any = "NA") -> Any:
        if key not in row.index or pd.isna(row[key]):
            return fallback
        return row[key]

    for spec in specs:
        frame = pd.read_csv(
            spec.output_root
            / "shared_metrics"
            / "batch_correction_metrics.csv"
        )
        row = frame.iloc[0]
        sampling_rows.append(
            [
                spec.name,
                value(row, "dataset"),
                value(row, "batch_key"),
                value(row, "batch_label"),
                int(value(row, "n_obs", N_OBS)),
                int(value(row, "n_obs_total", value(row, "n_obs", N_OBS))),
                int(value(row, "n_valid_total", value(row, "n_obs", N_OBS))),
                int(value(row, "n_used")),
                int(value(row, "n_batches")),
                value(row, "batch_categories"),
                value(row, "batch_counts"),
                int(value(row, "max_samples", 0)),
                value(row, "sample_size_requested"),
                value(row, "seed"),
                value(row, "embedding_scaled"),
                int(value(row, "bASW_sample_size")),
            ]
        )
        parameter_rows.append(
            [
                spec.name,
                value(row, "knn_backend"),
                int(value(row, "bLISI_neighbors")),
                int(value(row, "kBET_neighbors")),
                f4(value(row, "kBET_alpha")),
                int(value(row, "PCR_components")),
            ]
        )
        value_rows.append(
            [
                spec.name,
                f4(value(row, "bASW_raw")),
                f4(value(row, "bASW")),
                f4(value(row, "bLISI_raw")),
                f4(value(row, "bLISI")),
                f4(value(row, "kBET_rejection_rate")),
                f4(value(row, "kBET")),
                f4(value(row, "PCR_batch_R2")),
                f4(value(row, "PCR_score")),
            ]
        )
    return "\n\n".join(
        [
            "### Batch样本范围",
            md_table(
                [
                    "方法",
                    "dataset",
                    "batch key",
                    "batch label",
                    "n obs",
                    "n total",
                    "n valid",
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
                sampling_rows,
            ),
            "### Batch近邻与PCR参数",
            md_table(
                [
                    "方法",
                    "kNN backend",
                    "bLISI k",
                    "kBET k",
                    "alpha",
                    "PCR PCs",
                ],
                parameter_rows,
            ),
            "### Batch完整指标值",
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
                value_rows,
            ),
        ]
    )


def build_report(specs: list[MethodSpec]) -> None:
    for spec in specs:
        if not (
            spec.output_root
            / "preprocessing_comparison_manifest.json"
        ).exists():
            raise FileNotFoundError(f"incomplete result: {spec.output_root}")
    result_rows = [[spec.name, str(spec.output_root)] for spec in specs]
    lines = [
        "# spatch：Raw 与 Standardized embedding + MiniBatchKMeans 对比报告",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}。",
        "",
        "本报告只比较成功生成完整spatch最终embedding的spa_mo_model和COSIE。"
        "没有重新训练模型，也没有覆盖原训练或分析目录。",
        "",
        "## 结果目录",
        "",
        md_table(["方法", "独立结果目录"], result_rows),
        "",
        "## 统一评价口径",
        "",
        md_table(
            ["项目", "统一设置"],
            [
                ["spot", "全量1,068,962；section1=665,399，section2=403,563"],
                ["Raw分支", "原始最终embedding输入MiniBatchKMeans，距离指标在raw空间"],
                [
                    "Standardized分支",
                    "StandardScaler后输入MiniBatchKMeans，距离指标在standardized空间",
                ],
                [
                    "聚类参数",
                    "K=2–20，seed42，n_init20，max_iter300，batch_size4096",
                ],
                ["结果目录K", "仅5/8/10/12/16/20；其他K集中保存在单一labels归档"],
                ["Cluster/Label ASW", "固定相同身份的10,000 spot，random_state=42"],
                ["CH/DBI", "joint全量1,068,962；independent各section全量"],
                ["空间连续性", "每section精确6近邻；由对应分支保存labels直接计算"],
                ["空间图", "K=5/8/10/12/16/20，两个section均使用全量spot"],
            ],
        ),
        "",
        "每种方法仍只读取自己运行目录中的最终embedding和由各自`data/spatch`"
        "生成的spot metadata；两份metadata的spot_id、顺序、标签和坐标已逐点核对一致。"
        "两边Protein预处理均剔除DAPI。COSIE仍是用户指定的metacell+6×6方案；"
        "spa_mo_model仍是无空间块聚合的full-spot方案。",
        "",
        "表格中的`ASW raw`表示未做`(ASW+1)/2`映射的silhouette原值，"
        "不等同于Raw embedding分支。",
    ]
    for scheme, title in (
        ("raw_embedding", "Raw embedding + MiniBatchKMeans"),
        (
            "standardized_embedding",
            "Standardized embedding + MiniBatchKMeans",
        ),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "### 共同保留K的无监督内部指标",
                "",
                "ASW raw：",
                "",
                metric_matrix(specs, scheme, "cluster_asw"),
                "",
                "ASW scaled：",
                "",
                metric_matrix(specs, scheme, "cluster_asw_scaled"),
                "",
                "CH：",
                "",
                metric_matrix(specs, scheme, "calinski_harabasz"),
                "",
                "DBI：",
                "",
                metric_matrix(specs, scheme, "davies_bouldin"),
                "",
                "section ARI diagnostic：",
                "",
                metric_matrix(specs, scheme, "section_ari"),
                "",
                "section NMI diagnostic：",
                "",
                metric_matrix(specs, scheme, "section_nmi"),
                "",
                "### K=2–20范围内最佳内部指标",
                "",
                best_internal_table(specs, scheme),
                "",
                "### K=2–20按ARI选择的最佳外部指标",
                "",
                best_external_table(specs, scheme),
                "",
                "### 空间连续性",
                "",
                "Joint：",
                "",
                spatial_table(specs, scheme, "joint"),
                "",
                "Independent：",
                "",
                spatial_table(specs, scheme, "independent"),
            ]
        )
    lines.extend(
        [
            "",
            "## Raw与Standardized聚类标签稳定性",
            "",
            "下表为同一方法两种预处理joint labels之间的ARI；越接近1表示"
            "预处理对聚类划分影响越小。",
            "",
            stability_table(specs),
            "",
            "完整K=2–20、joint/independent稳定性保存在各方法"
            "`comparison_metrics/raw_vs_standardized_label_stability.csv`。",
            "",
            "## Batch Correction Metrics",
            "",
            "Batch指标不依赖本次KMeans输入分支，因此每种方法只在"
            "`shared_metrics/`保存一份，并记录源文件与复制文件SHA-256。"
            "Batch计算继续沿用统一的100,000样本及bASW 10,000口径。",
            "",
            batch_table(specs),
            "",
            "`NA`表示对应方法的当前源CSV没有记录该字段，报告不做推断补值；"
            "同义的旧版兼容列不重复展示。",
            "",
            "## 仍然存在的方法固有差异",
            "",
            md_table(
                ["差异", "影响大小", "说明"],
                [
                    [
                        "embedding维度128 vs 384",
                        "中到大",
                        "影响欧氏距离、CH/DBI、ASW和聚类几何；属于模型输出差异",
                    ],
                    [
                        "full-spot vs COSIE metacell+6×6",
                        "大",
                        "可能显著影响局部平滑与空间连续性；属于用户指定训练方案",
                    ],
                    [
                        "模型目标和训练轮数200 vs 600",
                        "大",
                        "影响最终表示，但属于待比较的方法本身",
                    ],
                ],
            ),
            "",
            "## 结论与解释原则",
            "",
            "Raw分支评价模型输出的原生欧氏几何；Standardized分支评价各维等权后的"
            "欧氏几何。两套结果平行保存，不根据某个方法在哪一分支更有优势而选择性"
            "汇报。spatch三个标签的覆盖范围不同，不能跨标签直接比较绝对分数；"
            "空间连续性高也不能单独等价为生物学聚类更准确。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = [METHODS[key] for key in ["spa", "cosie"]]
    if args.report_only:
        build_report(specs)
        print(f"report: {REPORT_PATH}", flush=True)
        return
    validate_cross_method()
    reference_root: Path | None = None
    for key in args.methods:
        spec = METHODS[key]
        metadata = load_metadata(spec)
        validate_metadata(spec, metadata)
        spec.output_root.mkdir(parents=True, exist_ok=True)
        graphs = prepare_shared(spec, metadata, reference_root)
        if reference_root is None:
            reference_root = spec.output_root
        print(f"[{spec.name}] load embedding", flush=True)
        embedding = load_embedding(spec)
        if len(embedding) != N_OBS or not np.isfinite(embedding).all():
            raise ValueError(f"{spec.name}: invalid embedding")
        for scheme in args.schemes:
            run_scheme(spec, metadata, embedding, graphs, scheme)
        if all(
            (
                spec.output_root
                / scheme
                / "analysis_completion_manifest.json"
            ).exists()
            for scheme in ("raw_embedding", "standardized_embedding")
        ):
            compare_schemes(spec)
            write_root_manifest(spec)
        del embedding, metadata, graphs
        gc.collect()
    if all(
        (
            spec.output_root / "preprocessing_comparison_manifest.json"
        ).exists()
        for spec in specs
    ):
        build_report(specs)
        print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
