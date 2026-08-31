#!/usr/bin/env python3
"""Evaluate two KNN imputation variants using all shared genes as pseudo-missing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


BASE_DIR = Path(__file__).resolve().parents[1] / "gene_imputation"
sys.path.insert(0, str(BASE_DIR))
import run_spatch_knn_imputation as base  # noqa: E402


FORMAT_NAME = "spa_mo_model.shared_gene_pseudomissing_validation"
FLOAT_METRICS = (
    "target_mean",
    "target_nonzero_fraction",
    "source_mean",
    "source_nonzero_fraction",
    "baseline_prediction_mean",
    "smoothed_prediction_mean",
    "baseline_pearson_raw_all",
    "smoothed_pearson_raw_all",
    "baseline_pearson_log1p_all",
    "smoothed_pearson_log1p_all",
    "baseline_spearman_sample",
    "smoothed_spearman_sample",
    "baseline_detection_average_precision_sample",
    "smoothed_detection_average_precision_sample",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("audit", "predict", "summarize", "all"), default="all"
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Only configuration schema_version=1 is supported.")
    run_dir = Path(config["run_dir"]).resolve()
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return config, run_dir, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_h5ad_names(path: Path, axis: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        frame = handle[axis]
        key = frame.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        return base.decode_strings(frame[str(key)])


def audit(config: dict[str, Any], run_dir: Path, config_hash: str) -> dict[str, Any]:
    required = {
        key: Path(config[key]).resolve()
        for key in (
            "target_rna_h5ad",
            "source_rna_h5ad",
            "baseline_neighbor_manifest",
            "smoothed_neighbor_manifest",
            "spatial_graph_manifest",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation inputs: {missing}")
    baseline_manifest = json.loads(
        required["baseline_neighbor_manifest"].read_text(encoding="utf-8")
    )
    smoothed_manifest = json.loads(
        required["smoothed_neighbor_manifest"].read_text(encoding="utf-8")
    )
    if not baseline_manifest.get("complete") or not smoothed_manifest.get("complete"):
        raise ValueError("Both KNN manifests must be complete.")
    baseline_neighbors = np.load(baseline_manifest["neighbors_path"], mmap_mode="r")
    smoothed_neighbors = np.load(smoothed_manifest["neighbors_path"], mmap_mode="r")
    if not np.array_equal(baseline_neighbors, smoothed_neighbors):
        raise ValueError("Baseline and smoothed KNN tables differ; paired validation is confounded.")
    if baseline_neighbors.shape[1] != int(config["knn_k"]):
        raise ValueError("KNN K differs from validation configuration.")

    target_names = read_h5ad_names(required["target_rna_h5ad"], "var")
    source_names = read_h5ad_names(required["source_rna_h5ad"], "var")
    target_lookup = {gene: index for index, gene in enumerate(target_names)}
    source_indices = np.asarray(
        [index for index, gene in enumerate(source_names) if gene in target_lookup], dtype=np.int32
    )
    shared_genes = source_names[source_indices]
    target_indices = np.asarray([target_lookup[gene] for gene in shared_genes], dtype=np.int32)
    if len(shared_genes) != int(config["expected_shared_genes"]):
        raise ValueError(
            f"Expected {config['expected_shared_genes']} shared genes, found {len(shared_genes)}."
        )
    if len(set(shared_genes.tolist())) != len(shared_genes):
        raise ValueError("Shared gene names are not unique.")
    missing_markers = [gene for gene in config["marker_genes"] if gene not in set(shared_genes)]
    if missing_markers:
        raise ValueError(f"Configured marker genes are not shared: {missing_markers}")

    with h5py.File(required["target_rna_h5ad"], "r") as target_handle, h5py.File(
        required["source_rna_h5ad"], "r"
    ) as source_handle:
        target_shape = base.h5ad_shape(target_handle)
        source_shape = base.h5ad_shape(source_handle)
        target_spatial = np.asarray(target_handle["obsm"]["spatial"][:])
    if target_shape[0] != baseline_neighbors.shape[0]:
        raise ValueError("Target RNA rows do not match KNN queries.")
    if source_shape[0] != int(baseline_manifest["source_count"]):
        raise ValueError("Source RNA rows do not match KNN reference rows.")

    sample_size = min(int(config["sample_target_spots"]), target_shape[0])
    rng = np.random.default_rng(int(config["random_seed"]))
    sample_indices = np.sort(rng.choice(target_shape[0], size=sample_size, replace=False)).astype(
        np.int32
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "shared_source_var_indices.npy", source_indices)
    np.save(run_dir / "shared_target_var_indices.npy", target_indices)
    np.save(run_dir / "sample_target_spot_indices.npy", sample_indices)
    with (run_dir / "shared_genes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shared_gene_index", "gene", "source_var_index", "target_var_index"])
        for index, (gene, source_index, target_index) in enumerate(
            zip(shared_genes, source_indices, target_indices)
        ):
            writer.writerow([index, gene, int(source_index), int(target_index)])

    graph_manifest = json.loads(
        required["spatial_graph_manifest"].read_text(encoding="utf-8")
    )
    graph_path = Path(graph_manifest["path"])
    graph = sp.load_npz(graph_path).tocsr()
    if graph.shape != (source_shape[0], source_shape[0]):
        raise ValueError("Source spatial smoothing graph shape is incompatible.")
    row_sums = np.asarray(graph.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-6):
        raise ValueError("Source spatial graph is not row normalized.")
    if abs(float(graph_manifest["alpha"]) - float(config["smoothing_alpha"])) > 1e-12:
        raise ValueError("Smoothing alpha differs from the graph run.")

    audit_payload = {
        "format": FORMAT_NAME,
        "config_hash": config_hash,
        "run_name": config["run_name"],
        "target_shape": list(target_shape),
        "source_shape": list(source_shape),
        "shared_gene_count": int(len(shared_genes)),
        "sample_target_spots": int(sample_size),
        "target_spots": int(target_shape[0]),
        "source_spots": int(source_shape[0]),
        "knn_shape": list(baseline_neighbors.shape),
        "knn_tables_elementwise_identical": True,
        "spatial_graph_shape": list(graph.shape),
        "spatial_graph_nnz": int(graph.nnz),
        "smoothing_alpha": float(config["smoothing_alpha"]),
        "marker_genes": list(config["marker_genes"]),
        "target_spatial_shape": list(target_spatial.shape),
        "inputs": {
            key: base.file_record(path, full_sha256=False) for key, path in required.items()
        },
    }
    base.atomic_write_json(run_dir / "input_audit.json", audit_payload)
    base.atomic_write_json(run_dir / "effective_config.json", config)
    base.LOGGER.info(
        "Audit passed: shared genes=%d target spots=%d sample=%d KNN exact=True",
        len(shared_genes),
        target_shape[0],
        sample_size,
    )
    return audit_payload


def load_audit(run_dir: Path, config_hash: str) -> dict[str, Any]:
    path = run_dir / "input_audit.json"
    if not path.is_file():
        raise FileNotFoundError("Run the audit stage first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config_hash:
        raise ValueError("Existing audit belongs to another configuration.")
    return payload


def pearson_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Pearson inputs must be equal two-dimensional arrays.")
    count = left.shape[0]
    sum_left = np.sum(left, axis=0, dtype=np.float64)
    sum_right = np.sum(right, axis=0, dtype=np.float64)
    sum_left2 = np.einsum("ij,ij->j", left, left, dtype=np.float64)
    sum_right2 = np.einsum("ij,ij->j", right, right, dtype=np.float64)
    sum_cross = np.einsum("ij,ij->j", left, right, dtype=np.float64)
    covariance = sum_cross - sum_left * sum_right / count
    variance_left = sum_left2 - np.square(sum_left) / count
    variance_right = sum_right2 - np.square(sum_right) / count
    denominator = np.sqrt(np.maximum(variance_left * variance_right, 0))
    return np.divide(
        covariance,
        denominator,
        out=np.full(left.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def spearman_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return pearson_columns(rankdata(left, axis=0), rankdata(right, axis=0))


def detection_average_precision_columns(
    truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    result = np.full(truth.shape[1], np.nan, dtype=np.float64)
    for column in range(truth.shape[1]):
        detected = truth[:, column] > 0
        positives = int(np.count_nonzero(detected))
        if positives == 0 or positives == len(detected) or np.ptp(prediction[:, column]) == 0:
            continue
        result[column] = average_precision_score(detected, prediction[:, column])
    return result


def initialize_cache(
    path: Path,
    config: dict[str, Any],
    audit_payload: dict[str, Any],
    genes: np.ndarray,
    spatial: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    block_size = int(config["gene_block_size"])
    block_count = math.ceil(len(genes) / block_size)
    marker_genes = list(config["marker_genes"])
    strings = h5py.string_dtype(encoding="utf-8")
    compression = config.get("compression")
    compression_kwargs: dict[str, Any] = {}
    if compression:
        compression_kwargs = {"compression": compression, "shuffle": True}
        if compression == "gzip":
            compression_kwargs["compression_opts"] = int(config["compression_level"])
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "format": FORMAT_NAME,
                "config_hash": audit_payload["config_hash"],
                "complete": False,
                "primary_metric": "spearman_sample",
                "sample_target_spots": len(sample_indices),
                "target_spots": spatial.shape[0],
                "shared_genes": len(genes),
                "created_at_unix": time.time(),
            }
        )
        handle.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=strings)
        handle.create_dataset("sample_target_spot_indices", data=sample_indices)
        handle.create_dataset("spatial", data=spatial)
        metrics = handle.create_group("metrics")
        for metric in FLOAT_METRICS:
            metrics.create_dataset(metric, shape=(len(genes),), dtype=np.float64, fillvalue=np.nan)
        marker_group = handle.create_group("marker_predictions")
        marker_group.create_dataset(
            "genes", data=np.asarray(marker_genes, dtype=object), dtype=strings
        )
        chunk = (min(4096, spatial.shape[0]), 1)
        for name in ("observed", "baseline", "smoothed"):
            marker_group.create_dataset(
                name,
                shape=(spatial.shape[0], len(marker_genes)),
                dtype=np.float32,
                chunks=chunk,
                fillvalue=np.nan,
                **compression_kwargs,
            )
        handle.create_dataset(
            "completed_gene_blocks", shape=(block_count,), dtype=np.bool_, fillvalue=False
        )
        handle.flush()


def predict_and_score(
    config: dict[str, Any], run_dir: Path, audit_payload: dict[str, Any]
) -> dict[str, Any]:
    baseline_neighbor_manifest = json.loads(
        Path(config["baseline_neighbor_manifest"]).read_text(encoding="utf-8")
    )
    neighbors = np.load(baseline_neighbor_manifest["neighbors_path"], mmap_mode="r")
    transfer = base.make_uniform_transfer_matrix(neighbors, int(audit_payload["source_spots"]))
    graph_manifest = json.loads(
        Path(config["spatial_graph_manifest"]).read_text(encoding="utf-8")
    )
    spatial_mean = sp.load_npz(graph_manifest["path"]).tocsr().astype(np.float32)
    alpha = np.float32(config["smoothing_alpha"])
    source_indices = np.load(run_dir / "shared_source_var_indices.npy")
    target_indices = np.load(run_dir / "shared_target_var_indices.npy")
    sample_indices = np.load(run_dir / "sample_target_spot_indices.npy")
    table = pd.read_csv(run_dir / "shared_genes.csv")
    genes = table["gene"].astype(str).to_numpy()
    marker_lookup = {gene: index for index, gene in enumerate(config["marker_genes"])}

    base.LOGGER.info("Loading and indexing source shared-gene expression.")
    source_full = base.load_h5ad_csr(Path(config["source_rna_h5ad"]))
    source_shared = source_full[:, source_indices].tocsc().astype(np.float32, copy=False)
    del source_full
    source_shared.sort_indices()
    base.LOGGER.info("Loading and indexing target pseudo-ground-truth expression.")
    target_full = base.load_h5ad_csr(Path(config["target_rna_h5ad"]))
    target_shared = target_full[:, target_indices].tocsc().astype(np.float32, copy=False)
    del target_full
    target_shared.sort_indices()
    base.LOGGER.info(
        "Matrices ready: source=%s nnz=%d target=%s nnz=%d %s",
        source_shared.shape,
        source_shared.nnz,
        target_shared.shape,
        target_shared.nnz,
        base.memory_gib(),
    )

    with h5py.File(config["target_rna_h5ad"], "r") as target_handle:
        target_spatial = np.asarray(target_handle["obsm"]["spatial"][:])
    cache_path = run_dir / "shared_gene_validation_cache.h5"
    if not cache_path.exists():
        initialize_cache(
            cache_path, config, audit_payload, genes, target_spatial, sample_indices
        )
    block_size = int(config["gene_block_size"])
    started = time.time()
    with h5py.File(cache_path, "r+") as cache:
        if cache.attrs.get("config_hash") != audit_payload["config_hash"]:
            raise ValueError("Existing validation cache belongs to another configuration.")
        completed = cache["completed_gene_blocks"]
        for block_index in range(len(completed)):
            if bool(completed[block_index]):
                continue
            start = block_index * block_size
            end = min(len(genes), start + block_size)
            block_started = time.time()
            source_block = source_shared[:, start:end].tocsr()
            baseline_prediction = base.aggregate_block(transfer, source_block)
            smoothed_source = (
                source_block.multiply(np.float32(1.0) - alpha)
                + (spatial_mean @ source_block).multiply(alpha)
            ).tocsr()
            smoothed_source.sum_duplicates()
            smoothed_source.eliminate_zeros()
            smoothed_prediction = base.aggregate_block(transfer, smoothed_source)
            observed = np.asarray(target_shared[:, start:end].toarray(), dtype=np.float32)

            metric_values: dict[str, np.ndarray] = {
                "target_mean": observed.mean(axis=0, dtype=np.float64),
                "target_nonzero_fraction": np.count_nonzero(observed, axis=0)
                / float(observed.shape[0]),
                "source_mean": np.asarray(source_block.mean(axis=0)).reshape(-1),
                "source_nonzero_fraction": np.asarray(source_block.getnnz(axis=0)).reshape(-1)
                / float(source_block.shape[0]),
                "baseline_prediction_mean": baseline_prediction.mean(axis=0, dtype=np.float64),
                "smoothed_prediction_mean": smoothed_prediction.mean(axis=0, dtype=np.float64),
                "baseline_pearson_raw_all": pearson_columns(observed, baseline_prediction),
                "smoothed_pearson_raw_all": pearson_columns(observed, smoothed_prediction),
            }
            observed_log = np.log1p(observed)
            baseline_log = np.log1p(baseline_prediction)
            metric_values["baseline_pearson_log1p_all"] = pearson_columns(
                observed_log, baseline_log
            )
            del baseline_log
            smoothed_log = np.log1p(smoothed_prediction)
            metric_values["smoothed_pearson_log1p_all"] = pearson_columns(
                observed_log, smoothed_log
            )
            del observed_log, smoothed_log

            observed_sample = observed[sample_indices]
            baseline_sample = baseline_prediction[sample_indices]
            smoothed_sample = smoothed_prediction[sample_indices]
            metric_values["baseline_spearman_sample"] = spearman_columns(
                observed_sample, baseline_sample
            )
            metric_values["smoothed_spearman_sample"] = spearman_columns(
                observed_sample, smoothed_sample
            )
            metric_values[
                "baseline_detection_average_precision_sample"
            ] = detection_average_precision_columns(observed_sample, baseline_sample)
            metric_values[
                "smoothed_detection_average_precision_sample"
            ] = detection_average_precision_columns(observed_sample, smoothed_sample)

            for name, values in metric_values.items():
                cache["metrics"][name][start:end] = values
            for local_index, gene in enumerate(genes[start:end]):
                if gene not in marker_lookup:
                    continue
                marker_index = marker_lookup[gene]
                cache["marker_predictions"]["observed"][:, marker_index] = observed[:, local_index]
                cache["marker_predictions"]["baseline"][:, marker_index] = baseline_prediction[
                    :, local_index
                ]
                cache["marker_predictions"]["smoothed"][:, marker_index] = smoothed_prediction[
                    :, local_index
                ]
            completed[block_index] = True
            cache.flush()
            del (
                source_block,
                baseline_prediction,
                smoothed_source,
                smoothed_prediction,
                observed,
                observed_sample,
                baseline_sample,
                smoothed_sample,
            )
            base.LOGGER.info(
                "Validation blocks: %d/%d genes=%d:%d block=%.1fs elapsed=%.1fs %s",
                int(np.count_nonzero(completed[:])),
                len(completed),
                start,
                end,
                time.time() - block_started,
                time.time() - started,
                base.memory_gib(),
            )
        cache.attrs["complete"] = bool(np.all(completed[:]))
        cache.attrs["completed_at_unix"] = time.time()
        cache.flush()
        complete = bool(cache.attrs["complete"])
    manifest = {
        "format": FORMAT_NAME,
        "config_hash": audit_payload["config_hash"],
        "cache_path": str(cache_path),
        "shared_genes": int(len(genes)),
        "target_spots": int(audit_payload["target_spots"]),
        "sample_target_spots": int(len(sample_indices)),
        "gene_block_size": block_size,
        "completed_blocks": int(math.ceil(len(genes) / block_size)),
        "complete": complete,
        "elapsed_seconds_this_process": time.time() - started,
        "memory": base.memory_gib(),
    }
    base.atomic_write_json(run_dir / "prediction_manifest.json", manifest)
    return manifest


def paired_summary(
    table: pd.DataFrame,
    baseline_column: str,
    smoothed_column: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    valid = np.isfinite(table[baseline_column]) & np.isfinite(table[smoothed_column])
    baseline_values = table.loc[valid, baseline_column].to_numpy()
    smoothed_values = table.loc[valid, smoothed_column].to_numpy()
    difference = smoothed_values - baseline_values
    if not len(difference):
        return {"valid_genes": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sample = rng.integers(0, len(difference), size=len(difference))
        boot[index] = np.median(difference[sample])
    return {
        "valid_genes": int(len(difference)),
        "baseline_median": float(np.median(baseline_values)),
        "smoothed_median": float(np.median(smoothed_values)),
        "median_paired_delta": float(np.median(difference)),
        "mean_paired_delta": float(np.mean(difference)),
        "fraction_genes_improved": float(np.mean(difference > 0)),
        "fraction_genes_tied": float(np.mean(difference == 0)),
        "median_delta_bootstrap_95ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "delta_quantiles": {
            str(prob): float(value)
            for prob, value in zip(
                (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1),
                np.quantile(difference, (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)),
            )
        },
    }


def summarize(
    config: dict[str, Any], run_dir: Path, audit_payload: dict[str, Any]
) -> dict[str, Any]:
    prediction_manifest = json.loads(
        (run_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    if not prediction_manifest.get("complete"):
        raise ValueError("Cannot summarize an incomplete validation cache.")
    cache_path = Path(prediction_manifest["cache_path"])
    with h5py.File(cache_path, "r") as cache:
        genes = cache["genes"].asstr()[:]
        payload: dict[str, Any] = {"gene": genes}
        for metric in FLOAT_METRICS:
            payload[metric] = cache["metrics"][metric][:]
    table = pd.DataFrame(payload)
    metric_pairs = {
        "spearman_sample": (
            "baseline_spearman_sample",
            "smoothed_spearman_sample",
        ),
        "pearson_log1p_all": (
            "baseline_pearson_log1p_all",
            "smoothed_pearson_log1p_all",
        ),
        "pearson_raw_all": (
            "baseline_pearson_raw_all",
            "smoothed_pearson_raw_all",
        ),
        "detection_average_precision_sample": (
            "baseline_detection_average_precision_sample",
            "smoothed_detection_average_precision_sample",
        ),
    }
    results: dict[str, Any] = {}
    for offset, (name, (baseline_column, smoothed_column)) in enumerate(metric_pairs.items()):
        table[f"delta_{name}"] = table[smoothed_column] - table[baseline_column]
        results[name] = paired_summary(
            table,
            baseline_column,
            smoothed_column,
            int(config["bootstrap_replicates"]),
            int(config["random_seed"]) + offset,
        )
    metrics_path = run_dir / "shared_gene_accuracy_metrics.csv.gz"
    table.to_csv(metrics_path, index=False, compression="gzip")

    valid_detection = table["target_nonzero_fraction"] > 0
    strata_table = table.loc[valid_detection].copy()
    strata_table["target_detection_stratum"] = pd.qcut(
        strata_table["target_nonzero_fraction"],
        q=4,
        labels=("Q1 lowest", "Q2", "Q3", "Q4 highest"),
        duplicates="drop",
    )
    strata_rows: list[dict[str, Any]] = []
    for stratum, frame in strata_table.groupby("target_detection_stratum", observed=True):
        for metric_name, (baseline_column, smoothed_column) in metric_pairs.items():
            valid = np.isfinite(frame[baseline_column]) & np.isfinite(frame[smoothed_column])
            selected = frame.loc[valid]
            delta = selected[smoothed_column] - selected[baseline_column]
            strata_rows.append(
                {
                    "target_detection_stratum": str(stratum),
                    "metric": metric_name,
                    "genes": int(len(selected)),
                    "target_nonzero_fraction_min": float(frame["target_nonzero_fraction"].min()),
                    "target_nonzero_fraction_max": float(frame["target_nonzero_fraction"].max()),
                    "baseline_median": float(selected[baseline_column].median()),
                    "smoothed_median": float(selected[smoothed_column].median()),
                    "median_delta": float(delta.median()),
                    "fraction_improved": float(np.mean(delta > 0)),
                }
            )
    strata = pd.DataFrame(strata_rows)
    strata_path = run_dir / "accuracy_by_target_detection_stratum.csv"
    strata.to_csv(strata_path, index=False)

    primary = results["spearman_sample"]
    summary = {
        "format": FORMAT_NAME,
        "config_hash": audit_payload["config_hash"],
        "status": "complete",
        "validation_design": "post-hoc shared-gene pseudo-missing prediction",
        "shared_genes": int(len(table)),
        "target_spots_full_metrics": int(audit_payload["target_spots"]),
        "target_spots_rank_metrics": int(audit_payload["sample_target_spots"]),
        "primary_metric": "spearman_sample",
        "metric_results": results,
        "primary_conclusion": {
            "baseline_median": primary.get("baseline_median"),
            "smoothed_median": primary.get("smoothed_median"),
            "median_delta": primary.get("median_paired_delta"),
            "fraction_genes_improved": primary.get("fraction_genes_improved"),
        },
        "paths": {
            "metrics": str(metrics_path),
            "detection_strata": str(strata_path),
            "cache": str(cache_path),
        },
        "limitations": [
            "Shared genes were available during embedding training, so this post-hoc validation can be optimistic.",
            "Section1 Xenium and section2 Visium HD have different count scales; rank correlation is the primary endpoint.",
            "A strict held-out-gene test requires retraining embeddings after removing evaluation genes from section1 inputs.",
        ],
    }
    base.atomic_write_json(run_dir / "validation_summary.json", summary)
    elapsed_seconds = float(prediction_manifest.get("elapsed_seconds_this_process", 0.0))
    peak_rss_gib = float(prediction_manifest.get("memory", {}).get("peak_rss_gib", 0.0))
    metric_labels = {
        "spearman_sample": "Spearman, fixed 20k spots (primary)",
        "pearson_log1p_all": "log1p Pearson, all spots",
        "pearson_raw_all": "Raw Pearson, all spots",
        "detection_average_precision_sample": (
            "Detection average precision, fixed 20k spots"
        ),
    }
    metric_rows = []
    for metric_name in metric_pairs:
        result = results[metric_name]
        metric_rows.append(
            f"| {metric_labels[metric_name]} | {result['baseline_median']:.5f} | "
            f"{result['smoothed_median']:.5f} | "
            f"{result['median_paired_delta']:+.5f} | "
            f"{result['fraction_genes_improved']:.2%} |"
        )
    primary_strata = strata.loc[strata["metric"] == "spearman_sample"]
    stratum_rows = []
    for row in primary_strata.itertuples(index=False):
        stratum_rows.append(
            f"| {row.target_detection_stratum} | {row.baseline_median:.5f} | "
            f"{row.smoothed_median:.5f} | {row.median_delta:+.5f} | "
            f"{row.fraction_improved:.2%} |"
        )
    report = f"""# Shared-gene pseudo-missing validation

Status: **complete**

- Shared genes: {len(table):,}
- Full-section Pearson spots: {audit_payload['target_spots']:,}
- Fixed-sample Spearman/AP spots: {audit_payload['sample_target_spots']:,}
- Primary metric: per-gene Spearman on the fixed sample
- Validation prediction runtime: {elapsed_seconds / 60:.1f} min; peak RSS: {peak_rss_gib:.2f} GiB

| Metric | Unsmoothed median | Smoothed median | Median paired delta | Genes improved |
|---|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

The primary median-delta bootstrap 95% CI is [{primary['median_delta_bootstrap_95ci'][0]:.5f}, {primary['median_delta_bootstrap_95ci'][1]:.5f}].

## Detection-rate strata

| section1 detection stratum | Unsmoothed median Spearman | Smoothed median Spearman | Median paired delta | Genes improved |
|---|---:|---:|---:|---:|
{chr(10).join(stratum_rows)}

Q1 genes are extremely sparse in section1 and their per-gene correlations are
not reliable evidence for choosing a method. The smoothing advantage is much
clearer and more consistent in Q2-Q4.

## Result files

- Per-gene metrics: `{metrics_path}`
- Detection strata: `{strata_path}`
- Machine-readable summary: `{run_dir / 'validation_summary.json'}`
- Formal figures and guide: `{run_dir / 'figures'}`

## Interpretation boundary

This is a post-hoc pseudo-missing diagnostic. The shared genes were present
when the embeddings were trained, so the scores may be optimistic. Because
Xenium and Visium HD have different count scales, Spearman is the primary
comparison. A strict leakage-free benchmark requires model retraining with
held-out target genes.
"""
    (run_dir / "SUMMARY.md").write_text(report, encoding="utf-8")
    base.LOGGER.info("Summary written: %s", run_dir / "SUMMARY.md")
    return summary


def main() -> None:
    args = parse_args()
    config, run_dir, config_hash = load_config(args.config.resolve())
    base.setup_logging(run_dir)
    base.LOGGER.info(
        "Run=%s stage=%s config_hash=%s", config["run_name"], args.stage, config_hash
    )
    if args.stage in ("audit", "all"):
        audit_payload = audit(config, run_dir, config_hash)
    else:
        audit_payload = load_audit(run_dir, config_hash)
    if args.stage in ("predict", "all"):
        predict_and_score(config, run_dir, audit_payload)
    if args.stage in ("summarize", "all"):
        summarize(config, run_dir, audit_payload)
    base.LOGGER.info("Requested stage completed successfully. %s", base.memory_gib())


if __name__ == "__main__":
    main()
