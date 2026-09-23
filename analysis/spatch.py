#!/usr/bin/env python3
"""Analyze one full-spot SPATCH v7 run with only the requested metrics."""

from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
from sklearn.preprocessing import StandardScaler


import shutil
from analysis.plotting import plot_spatch_spatial as plot_spatial
from analysis.clustering import (spatch_fit_model, minibatch_parameters, fitted_space)
from analysis.protocols import SPATCH, SPATCH_REQUESTED, joint_per_section_protocol
from analysis.metrics import (spatch_supervised_rows, spatch_requested_internal_row as internal_row,
    compute_joint_per_section_supervised_metrics)
from analysis.sampling import permutation_sample_indices
from analysis.batch_metrics import compute_batch_correction_metrics
from analysis.inputs import load_requested_spatch_metadata, load_requested_spatch_embedding
from data_io.saved_assignments import read_assignment_array
from analysis.cache import (array_identity, begin_analysis, cache_hit, check_output_path, file_identity,
    finish_analysis, frame_identity, implementation_identity, run_identity, save_cache)

SECTIONS = ["section1", "section2"]
SECTION_COUNTS = {"section1": 665399, "section2": 403563}
N_OBS = sum(SECTION_COUNTS.values())
SEED = SPATCH['SEED']
N_INIT = SPATCH['N_INIT']
MAX_ITER = SPATCH['MAX_ITER']
BATCH_SIZE = SPATCH['BATCH_SIZE']
ASW_SAMPLE_SIZE = SPATCH['ASW_SAMPLE_SIZE']
TRUTH_LABELS = SPATCH_REQUESTED['TRUTH_LABELS']
K_VALUES = SPATCH_REQUESTED['K_VALUES']
from analysis.protocols import REQUESTED_SUPERVISED_METRICS
SUPERVISED_METRICS = tuple(REQUESTED_SUPERVISED_METRICS)
from analysis.protocols import REQUESTED_INTERNAL_METRICS
INTERNAL_METRICS = tuple(REQUESTED_INTERNAL_METRICS)
from analysis.protocols import REQUESTED_BATCH_METRICS
BATCH_METRICS = tuple(REQUESTED_BATCH_METRICS)


def fit_model(space: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    return spatch_fit_model(space, k, seed=SEED, n_init=N_INIT, max_iter=MAX_ITER, batch_size=BATCH_SIZE)


def fit_parameters(k: int = 8) -> dict:
    return minibatch_parameters(k, seed=SEED, n_init=N_INIT, max_iter=MAX_ITER, batch_size=BATCH_SIZE)


def fit_or_load(
    cache_path: Path,
    info_path: Path,
    centers_path: Path,
    space: np.ndarray,
    k: int,
    *, source_identity: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    identity = {"source": source_identity, "clustering": fit_parameters(k),
                "implementation": implementation_identity("spatch-fit-b9-v1")}
    manifest = info_path.with_suffix(".source.json")
    files = [cache_path, info_path, centers_path]
    if cache_hit(manifest, identity, files):
        return read_assignment_array(cache_path), json.loads(info_path.read_text())
    labels, info, centers = fit_model(space, k)
    np.save(cache_path, labels)
    np.save(centers_path, centers)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    save_cache(manifest, identity, files)
    return labels, info


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(run_dir: Path) -> pd.DataFrame:
    return load_requested_spatch_metadata(run_dir, truth_labels=TRUTH_LABELS,
                                         n_obs=N_OBS, section_counts=SECTION_COUNTS)


def load_embedding(run_dir: Path) -> np.ndarray:
    return load_requested_spatch_embedding(run_dir, section_order=SECTIONS,
                                          section_counts=SECTION_COUNTS)


def fixed_sample_indices(n: int) -> np.ndarray:
    return permutation_sample_indices(n, ASW_SAMPLE_SIZE, SEED)


def supervised_rows(metadata, labels, mode, scope, k):
    return spatch_supervised_rows(metadata, labels, mode, scope, k, truth_labels=TRUTH_LABELS)


def persist_clustering(
    output: Path,
    local_metadata: pd.DataFrame,
    mode: str,
    scope: str,
    k: int,
    labels: np.ndarray,
    centers_path: Path,
) -> None:
    retained_outputs(
        output,
        local_metadata,
        mode,
        scope,
        k,
        labels,
        centers_path,
    )


def analyze(run_dir: Path, model_version: str, scale: float, *, output_dir: Path,
            joint_per_section: bool = False, k_values=None,
            modes=("joint", "independent")) -> dict[str, Any]:
    k_values = K_VALUES if k_values is None else tuple(k_values)
    output = check_output_path(output_dir, [run_dir])
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("post_ot_graphsage_scale") != scale:
        raise ValueError(
            f"Training scale differs: {summary.get('post_ot_graphsage_scale')} != {scale}."
        )
    metadata = load_metadata(run_dir)
    embedding = load_embedding(run_dir)
    source = {
        "dataset": "SPATCH", "model_version": model_version,
        "section_order": list(SECTIONS),
        "embeddings": [file_identity(run_dir / f"final_embeddings_{s}.npy") for s in SECTIONS],
        "metadata_file": file_identity(run_dir / "spot_metadata.csv.gz"),
        "aligned_metadata": frame_identity(metadata), "run": run_identity(summary),
        "protocol": {"k_values": list(k_values), "clustering": fit_parameters(),
                     "scaler": StandardScaler(copy=True).get_params(),
                     "scaler_scope": {"joint": "all_spots", "independent": "per_section"},
                     "truth_labels": list(TRUTH_LABELS), "truth_filter": "notna, at least two classes",
                     "internal_metrics": list(INTERNAL_METRICS), "supervised_metrics": list(SUPERVISED_METRICS),
                     "asw_sample_size": ASW_SAMPLE_SIZE, "sample_seed": SEED,
                     "sample": "sorted RandomState permutation prefix, euclidean distances",
                     "batch": {"space": "raw", "metrics": list(BATCH_METRICS), "max_samples": 100000,
                               "asw_sample_size": 10000, "lisi_neighbors": 90, "kbet_neighbors": 50,
                               "seed": 42, "pcr_components": 50,
                               "kbet_alpha": compute_batch_correction_metrics.__kwdefaults__["kbet_alpha"]}},
        "implementation": implementation_identity("spatch-requested-b9-v1"),
    }
    if modes != ("joint", "independent"):
        source["protocol"]["modes"] = list(modes)
    if joint_per_section:
        source["protocol"]["joint_per_section"] = joint_per_section_protocol("SPATCH", k_values, SECTIONS)
    completion = output / "analysis_completion_manifest.json"
    if begin_analysis(output, source, input_dirs=[run_dir]):
        record = json.loads(completion.read_text())
        record["analysis"] = str(output)
        if record.get("joint_per_section", {}).get("metrics") is not None:
            record["joint_per_section"]["metrics"] = str(output / "metrics/joint_per_section_supervised_metrics.csv")
        for name in ("internal_metrics", "supervised_metrics", "batch_metrics"):
            record[name] = str(output / "metrics" / f"{name}.csv")
        print(json.dumps(record, indent=2), flush=True)
        print(f"SPATCH_{model_version}_REQUESTED_ANALYSIS_ALREADY_COMPLETE", flush=True)
        return record
    metrics_dir = output / "metrics"
    work_cache = output / "_fit_cache"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    work_cache.mkdir(parents=True, exist_ok=True)

    internal: list[dict[str, Any]] = []
    supervised: list[dict[str, Any]] = []
    per_section_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    scopes = [("joint", "combined", np.ones(N_OBS, dtype=bool))]
    scopes.extend(
        ("independent", section, metadata["section"].eq(section).to_numpy())
        for section in SECTIONS
    )

    for mode, scope, mask in scopes:
        if mode not in modes:
            continue
        print(f"[{model_version}] prepare {mode}:{scope}", flush=True)
        raw = embedding if mode == "joint" else embedding[mask]
        space, scaler = fitted_space(raw, "standardized_embedding")
        scaler_arrays.update(
            {
                f"{scope}_mean": scaler.mean_,
                f"{scope}_scale": scaler.scale_,
                f"{scope}_var": scaler.var_,
            }
        )
        local_metadata = metadata if mode == "joint" else metadata.loc[mask].reset_index(drop=True)
        sample_indices = fixed_sample_indices(len(space))
        sample_distances = pairwise_distances(
            space[sample_indices], metric="euclidean", n_jobs=-1
        )
        np.fill_diagonal(sample_distances, 0.0)
        fit_source = {"source": source, "mode": mode, "scope": scope, "space": array_identity(space)}
        for k in k_values:
            prefix = f"{mode}_{scope}_k{k}"
            label_path = work_cache / f"{prefix}.npy"
            info_path = work_cache / f"{prefix}.json"
            centers_path = work_cache / f"{prefix}_centers.npy"
            print(f"[{model_version}] {prefix}", flush=True)
            labels, _ = fit_or_load(
                label_path,
                info_path,
                centers_path,
                space,
                k,
                source_identity=fit_source,
            )
            internal.append(
                internal_row(
                    space,
                    labels,
                    sample_indices,
                    sample_distances,
                    mode,
                    scope,
                    k,
                )
            )
            supervised.extend(supervised_rows(local_metadata, labels, mode, scope, k))
            if joint_per_section and mode == "joint":
                per_section_rows.extend(compute_joint_per_section_supervised_metrics(
                    "SPATCH", labels, metadata["section"].to_numpy(),
                    {"cell_type_common": metadata["cell_type_common"].to_numpy()},
                    k=k, section_order=SECTIONS,
                    joint_label_source=str(label_path),
                    truth_source=str(run_dir / "spot_metadata.csv.gz"),
                )["rows"])
            persist_clustering(
                output,
                local_metadata,
                mode,
                scope,
                k,
                labels,
                centers_path,
            )
        del sample_distances, space
        gc.collect()

    if joint_per_section and per_section_rows:
        pd.DataFrame(per_section_rows).to_csv(metrics_dir / "joint_per_section_supervised_metrics.csv", index=False)
    internal_frame = pd.DataFrame(internal)
    supervised_frame = pd.DataFrame(supervised)
    internal_frame.to_csv(metrics_dir / "internal_metrics.csv", index=False)
    supervised_frame.to_csv(metrics_dir / "supervised_metrics.csv", index=False)
    batch_all = compute_batch_correction_metrics(
        embedding,
        metadata["section"].to_numpy(),
        dataset="SPATCH",
        method=f"spa_mo_model_result_{model_version}",
        batch_label_name="section",
        max_samples=100000,
        asw_sample_size=10000,
        lisi_neighbors=90,
        kbet_neighbors=50,
        seed=42,
        pcr_components=50,
    )
    batch_frame = pd.DataFrame(
        [
            {
                "dataset": "SPATCH",
                "n_obs": N_OBS,
                **{metric: batch_all[metric] for metric in BATCH_METRICS},
            }
        ]
    )
    batch_frame.to_csv(metrics_dir / "batch_metrics.csv", index=False)
    np.savez_compressed(output / "scaler_parameters.npz", **scaler_arrays)
    config = {
        "dataset": "SPATCH",
        "model_version": model_version,
        "post_ot_graphsage_scale": scale,
        "preprocessing": "standardized_embedding",
        "clustering": {
            "implementation": "sklearn.cluster.MiniBatchKMeans",
            "k_values": list(k_values),
            "modes": ["joint", "independent"],
            "seed": SEED,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "batch_size": BATCH_SIZE,
        },
        "metrics": {
            "supervised": list(SUPERVISED_METRICS),
            "unsupervised": list(INTERNAL_METRICS),
            "batch": list(BATCH_METRICS),
        },
        "cluster_asw_sample_size": ASW_SAMPLE_SIZE,
        "spatial_cluster_plots": True,
        "umap_computed": False,
        "umap_plots": False,
        "source_run_summary": str(summary_path),
        "source_run_summary_sha256": sha256_file(summary_path),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completion_payload = {
        "status": "complete",
        "dataset": "SPATCH",
        "model_version": model_version,
        "post_ot_graphsage_scale": scale,
        "analysis": str(output),
        "k_values": list(k_values),
        "internal_metrics": str(metrics_dir / "internal_metrics.csv"),
        "supervised_metrics": str(metrics_dir / "supervised_metrics.csv"),
        "batch_metrics": str(metrics_dir / "batch_metrics.csv"),
        "spatial_plot_count": len(list((output / "clustering").glob("*/spatial_*.png"))),
        "umap_computed": False,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    if joint_per_section:
        completion_payload["joint_per_section"] = {
            "status": source["protocol"]["joint_per_section"]["status"],
            "metrics": str(metrics_dir / "joint_per_section_supervised_metrics.csv") if per_section_rows else None,
            "aggregation": "none",
        }
    completion.write_text(
        json.dumps(completion_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = output / "spatch_requested_metrics_manifest.json"
    manifest_path.write_text(
        json.dumps(completion_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    finish_analysis(output, source)
    print(json.dumps(completion_payload, indent=2, ensure_ascii=False), flush=True)
    print(f"SPATCH_{model_version}_REQUESTED_ANALYSIS_PASS", flush=True)
    return completion_payload


import zipfile
from sklearn.neighbors import NearestNeighbors
from analysis.cache import digest_file as sha256, summary_identity
from analysis.clustering import fitted_space as metric_space, scaler_payload
from analysis.metrics import spatch_label_asw, external_metrics as external_score, spatch_internal_metrics as internal_score
LABELS = SPATCH['LABELS']
ALL_KS = SPATCH['ALL_KS']
RETAIN_KS = SPATCH['RETAIN_KS']
SPATIAL_NEIGHBOR_K = 6
POINT_SIZE = 0.2
SCRIPT_PATH = Path(__file__).resolve()


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
    spec: Any,
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


def source_identity(spec: Any, metadata: pd.DataFrame) -> dict:
    return {"dataset": "SPATCH", "method": spec.name,
            "section_order": list(SECTIONS),
            "embeddings": [file_identity(spec.embedding_paths[s]) for s in SECTIONS],
            "metadata_file": file_identity(spec.metadata_path),
            "aligned_metadata": frame_identity(metadata), "run": summary_identity(spec.run_dir),
            "historical_batch_metrics": {"source": file_identity(spec.batch_metrics_source),
                "provenance": "historical/unverified; copied values, not recomputed or certified against this embedding"},
            "protocol": {"k_values": ALL_KS, "retained_k": RETAIN_KS, "clustering": fit_parameters(),
                "scaler": StandardScaler(copy=True).get_params(),
                "scaler_scope": {"joint": "all_spots", "independent": "per_section"},
                "labels": LABELS, "label_filter": "notna; at least two classes",
                "asw_sample_size": ASW_SAMPLE_SIZE, "sample_seed": SEED,
                "sampling": "sorted RandomState permutation per scope/valid truth subset",
                "internal": ["ASW", "scaled ASW", "CH full", "DBI full"],
                "external": ["ARI", "NMI", "homogeneity", "completeness", "v_measure", "label ASW"],
                "spatial_neighbor_k": SPATIAL_NEIGHBOR_K, "neighbor_algorithm": "exact kd_tree"},
            "implementation": implementation_identity("spatch-comparison-b9-v1")}


def label_asw(space, truth, metadata, scope, label):
    return spatch_label_asw(space, truth, metadata, scope, label, sample_size=ASW_SAMPLE_SIZE, seed=SEED)


def spatial_agreement(labels: np.ndarray, graph: np.ndarray) -> float:
    return float(np.mean(labels[graph] == labels[:, None]))


def archive_cache(cache_root: Path, output_path: Path) -> None:
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for path in sorted(cache_root.glob("*")):
            archive.write(path, arcname=path.name)
    shutil.rmtree(cache_root)


def run_scheme(
    spec: Any,
    metadata: pd.DataFrame,
    embedding: np.ndarray,
    graphs: dict[str, np.ndarray],
    scheme: str,
    *, source: dict, all_ks=None, retain_ks=None, modes=("joint", "independent"),
) -> None:
    all_ks = ALL_KS if all_ks is None else list(all_ks)
    retain_ks = RETAIN_KS if retain_ks is None else list(retain_ks)
    output = spec.output_root / scheme
    completion = output / "analysis_completion_manifest.json"
    identity = {"source": source, "scheme": scheme,
                "graphs": {s: array_identity(graphs[s]) for s in SECTIONS}}
    if modes != ("joint", "independent") or all_ks != ALL_KS or retain_ks != RETAIN_KS:
        identity["selection"] = {"modes": list(modes), "all_ks": all_ks, "retain_ks": retain_ks}
    if begin_analysis(output, identity, input_dirs=[spec.run_dir]):
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
        if mode not in modes:
            continue
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
        fit_source = {"source": identity, "mode": mode, "scope": scope, "space": array_identity(space)}

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

        for k in all_ks:
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
                source_identity=fit_source,
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

            if k in retain_ks:
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
        "all_metric_k": all_ks,
        "retained_result_directory_k": retain_ks,
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
            "sha256": source["metadata_file"]["sha256"],
        },
        "source_embeddings": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in source["embeddings"]
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
    finish_analysis(output, identity)
