#!/usr/bin/env python3
"""Memory-bounded spatially smoothed reference KNN gene imputation for spatch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


BASE_DIR = Path(__file__).resolve().parents[1] / "gene_imputation"
sys.path.insert(0, str(BASE_DIR))
import run_spatch_knn_imputation as base  # noqa: E402


FORMAT_NAME = "spa_mo_model.spatial_smoothed_knn_gene_imputation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("audit", "neighbors", "spatial_graph", "impute", "validate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--limit-target-spots", type=int, default=None)
    parser.add_argument("--limit-genes", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def build_grid_spatial_smoothing_matrix(
    coordinates: np.ndarray, radius_grid_units: float
) -> tuple[sp.csr_matrix, np.ndarray, list[tuple[int, int]]]:
    """Build a row-normalized neighbor mean on a unique integer grid."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Spatial coordinates must have shape (n_spots, 2).")
    rounded = np.rint(coordinates).astype(np.int64)
    if not np.array_equal(coordinates, rounded):
        raise ValueError("The spatch grid smoother requires exact integer coordinates.")
    if not base.unique_rows(rounded):
        raise ValueError("Spatial coordinates must be unique.")
    if radius_grid_units <= 0:
        raise ValueError("radius_grid_units must be positive.")

    maximum_offset = int(math.floor(radius_grid_units))
    offsets = [
        (dx, dy)
        for dx in range(-maximum_offset, maximum_offset + 1)
        for dy in range(-maximum_offset, maximum_offset + 1)
        if (dx != 0 or dy != 0)
        and math.hypot(dx, dy) <= radius_grid_units + 1e-12
    ]
    if not offsets:
        raise ValueError("The radius does not include any non-self grid position.")

    minima = rounded.min(axis=0)
    shifted = rounded - minima
    shape = tuple((shifted.max(axis=0) + 1).tolist())
    lookup = np.full(shape, -1, dtype=np.int32)
    lookup[shifted[:, 0], shifted[:, 1]] = np.arange(len(shifted), dtype=np.int32)

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for dx, dy in offsets:
        candidate = shifted + np.asarray([dx, dy], dtype=np.int64)
        in_bounds = (
            (candidate[:, 0] >= 0)
            & (candidate[:, 0] < shape[0])
            & (candidate[:, 1] >= 0)
            & (candidate[:, 1] < shape[1])
        )
        rows = np.flatnonzero(in_bounds)
        columns = lookup[candidate[in_bounds, 0], candidate[in_bounds, 1]]
        present = columns >= 0
        row_parts.append(rows[present].astype(np.int32, copy=False))
        column_parts.append(columns[present].astype(np.int32, copy=False))

    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    neighbor_counts = np.bincount(rows, minlength=len(rounded)).astype(np.int32)
    isolated = np.flatnonzero(neighbor_counts == 0).astype(np.int32)
    if len(isolated):
        rows = np.concatenate((rows, isolated))
        columns = np.concatenate((columns, isolated))
    normalizers = np.where(neighbor_counts > 0, neighbor_counts, 1)
    weights = (1.0 / normalizers[rows]).astype(np.float32)
    matrix = sp.csr_matrix(
        (weights, (rows, columns)), shape=(len(rounded), len(rounded)), dtype=np.float32
    )
    matrix.sort_indices()
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("Spatial smoothing rows do not sum to one.")
    return matrix, neighbor_counts, offsets


def build_or_load_spatial_graph(
    config: dict[str, Any], run_dir: Path, audit: dict[str, Any]
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    graph_path = run_dir / "source_spatial_neighbor_mean.npz"
    manifest_path = run_dir / "source_spatial_neighbor_mean_manifest.json"
    if graph_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != audit["config_hash"]:
            raise ValueError("Existing spatial graph belongs to a different configuration.")
        graph = sp.load_npz(graph_path).tocsr()
        expected_shape = (int(audit["source_rna_shape"][0]),) * 2
        if graph.shape != expected_shape:
            raise ValueError("Existing spatial graph has an incompatible shape.")
        base.LOGGER.info("Reused source spatial graph: %s", graph_path)
        return graph, manifest

    settings = config["smoothing"]
    if settings["method"] != "spatial_residual_neighbor_mean":
        raise ValueError("Only spatial_residual_neighbor_mean is supported.")
    if int(settings.get("iterations", 1)) != 1:
        raise ValueError("This workflow deliberately supports exactly one smoothing iteration.")
    if not settings.get("exclude_self_from_neighbor_mean", True):
        raise ValueError("This version requires self-excluded neighbor means.")
    if settings.get("isolated_spot_policy") != "self":
        raise ValueError("This version requires isolated_spot_policy='self'.")
    alpha = float(settings["alpha"])
    if not 0 <= alpha <= 1:
        raise ValueError("Smoothing alpha must be between zero and one.")

    with h5py.File(config["source_rna_h5ad"], "r") as handle:
        coordinates = np.asarray(handle["obsm"][settings["coordinate_key"]][:])
    started = time.time()
    graph, counts, offsets = build_grid_spatial_smoothing_matrix(
        coordinates, float(settings["radius_grid_units"])
    )
    sp.save_npz(graph_path, graph, compressed=True)
    active_counts = counts[counts > 0]
    manifest = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "path": str(graph_path),
        "shape": list(graph.shape),
        "nnz": int(graph.nnz),
        "coordinate_key": settings["coordinate_key"],
        "coordinate_grid_step_um": float(settings["coordinate_grid_step_um"]),
        "radius_grid_units": float(settings["radius_grid_units"]),
        "radius_um_approx": float(
            settings["coordinate_grid_step_um"] * settings["radius_grid_units"]
        ),
        "offsets": [list(value) for value in offsets],
        "isolated_spots": int(np.count_nonzero(counts == 0)),
        "neighbor_count_min_nonisolated": int(active_counts.min()),
        "neighbor_count_median_nonisolated": float(np.median(active_counts)),
        "neighbor_count_max": int(counts.max()),
        "alpha": alpha,
        "complete": True,
    }
    base.atomic_write_json(manifest_path, manifest)
    base.LOGGER.info(
        "Built source spatial graph in %.2fs: shape=%s nnz=%d isolated=%d %s",
        time.time() - started,
        graph.shape,
        graph.nnz,
        manifest["isolated_spots"],
        base.memory_gib(),
    )
    return graph, manifest


def initialize_smoothed_output(
    path: Path,
    config: dict[str, Any],
    audit: dict[str, Any],
    target_genes: np.ndarray,
    target_spatial: np.ndarray,
    source_nnz: np.ndarray,
    source_total: np.ndarray,
) -> None:
    base.initialize_output(
        path, config, audit, target_genes, target_spatial, source_nnz, source_total
    )
    with h5py.File(path, "r+") as handle:
        handle.attrs["format"] = FORMAT_NAME
        handle.attrs["algorithm"] = (
            "one-step spatial residual smoothing of source expression, then uniform embedding-KNN mean"
        )
        handle.attrs["smoothing_alpha"] = float(config["smoothing"]["alpha"])
        handle.attrs["smoothing_radius_grid_units"] = float(
            config["smoothing"]["radius_grid_units"]
        )
        stats = handle["gene_stats"]
        stats.create_dataset(
            "smoothed_source_nnz", shape=(len(target_genes),), dtype=np.int64, fillvalue=-1
        )
        stats.create_dataset(
            "smoothed_source_total", shape=(len(target_genes),), dtype=np.float64, fillvalue=np.nan
        )
        handle.flush()


def impute_smoothed(
    config: dict[str, Any], run_dir: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    neighbor_manifest_path = run_dir / "neighbor_manifest.json"
    if not neighbor_manifest_path.is_file():
        raise FileNotFoundError("Run the neighbors stage first.")
    neighbor_manifest = json.loads(neighbor_manifest_path.read_text(encoding="utf-8"))
    if neighbor_manifest.get("config_hash") != audit["config_hash"] or not neighbor_manifest.get(
        "complete"
    ):
        raise ValueError("Neighbor cache is incomplete or incompatible.")
    neighbors = np.load(neighbor_manifest["neighbors_path"], mmap_mode="r")
    source_count = int(audit["source_embedding_shape"][0])
    transfer = base.make_uniform_transfer_matrix(neighbors, source_count)
    spatial_mean, graph_manifest = build_or_load_spatial_graph(config, run_dir, audit)
    base.LOGGER.info(
        "Transfer W=%s nnz=%d; spatial S=%s nnz=%d %s",
        transfer.shape,
        transfer.nnz,
        spatial_mean.shape,
        spatial_mean.nnz,
        base.memory_gib(),
    )

    source_indices = np.load(run_dir / "target_source_var_indices.npy")
    with (run_dir / "target_genes.csv").open("r", encoding="utf-8", newline="") as handle:
        target_genes = np.asarray([row["gene"] for row in csv.DictReader(handle)], dtype=object)
    full_source = base.load_h5ad_csr(Path(config["source_rna_h5ad"]))
    source_target = full_source[:, source_indices].tocsr().astype(np.float32, copy=False)
    del full_source
    source_target.sort_indices()
    source_nnz = np.asarray(source_target.getnnz(axis=0)).reshape(-1).astype(np.int64)
    source_total = np.asarray(source_target.sum(axis=0)).reshape(-1).astype(np.float64)
    with h5py.File(config["target_rna_h5ad"], "r") as handle:
        target_spatial = np.asarray(handle["obsm"]["spatial"][:])

    output_path = run_dir / (
        f"imputed_{config['target_section']}_{config['source_section']}_only_genes_spatial_smoothed.h5"
    )
    if not output_path.exists():
        initialize_smoothed_output(
            output_path,
            config,
            audit,
            target_genes,
            target_spatial,
            source_nnz,
            source_total,
        )

    settings = config["imputation"]
    alpha = float(config["smoothing"]["alpha"])
    block_size = int(settings["gene_block_size"])
    started = time.time()
    with h5py.File(output_path, "r+") as handle:
        if handle.attrs.get("config_hash") != audit["config_hash"]:
            raise ValueError("Existing output belongs to a different configuration.")
        if handle["X"].shape != (neighbors.shape[0], len(target_genes)):
            raise ValueError("Existing output has an incompatible shape.")
        completed = handle["completed_gene_blocks"]
        for block_index in range(len(completed)):
            if bool(completed[block_index]):
                continue
            start = block_index * block_size
            end = min(len(target_genes), start + block_size)
            block_started = time.time()
            raw_block = source_target[:, start:end]
            neighbor_block = spatial_mean @ raw_block
            smoothed_block = (
                raw_block.multiply(np.float32(1.0 - alpha))
                + neighbor_block.multiply(np.float32(alpha))
            ).tocsr()
            smoothed_block.sum_duplicates()
            smoothed_block.eliminate_zeros()
            prediction = base.aggregate_block(transfer, smoothed_block)
            handle["X"][:, start:end] = prediction.astype(
                settings["output_dtype"], copy=False
            )
            stats = handle["gene_stats"]
            stats["smoothed_source_nnz"][start:end] = np.asarray(
                smoothed_block.getnnz(axis=0)
            ).reshape(-1)
            stats["smoothed_source_total"][start:end] = np.asarray(
                smoothed_block.sum(axis=0)
            ).reshape(-1)
            stats["predicted_mean"][start:end] = prediction.mean(axis=0, dtype=np.float64)
            stats["predicted_nonzero_fraction"][start:end] = np.count_nonzero(
                prediction, axis=0
            ) / float(prediction.shape[0])
            stats["predicted_max"][start:end] = prediction.max(axis=0)
            completed[block_index] = True
            handle.flush()
            del raw_block, neighbor_block, smoothed_block, prediction
            base.LOGGER.info(
                "Smoothed gene blocks: %d/%d genes=%d:%d block=%.1fs elapsed=%.1fs file=%.2fGiB %s",
                int(np.count_nonzero(completed[:])),
                len(completed),
                start,
                end,
                time.time() - block_started,
                time.time() - started,
                output_path.stat().st_size / 1024**3,
                base.memory_gib(),
            )
        handle.attrs["complete"] = bool(np.all(completed[:]))
        handle.attrs["completed_at_unix"] = time.time()
        handle.flush()
        complete = bool(handle.attrs["complete"])

    manifest = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "output_path": str(output_path),
        "shape": [int(neighbors.shape[0]), int(len(target_genes))],
        "dtype": settings["output_dtype"],
        "gene_block_size": block_size,
        "compression": settings.get("compression"),
        "aggregation": "uniform_mean_after_spatial_residual_smoothing",
        "smoothing": graph_manifest,
        "complete": complete,
        "file_size_bytes": output_path.stat().st_size,
        "uncompressed_X_bytes": int(
            neighbors.shape[0]
            * len(target_genes)
            * np.dtype(settings["output_dtype"]).itemsize
        ),
    }
    base.atomic_write_json(run_dir / "imputation_manifest.json", manifest)
    return manifest


def validate_smoothed(
    config: dict[str, Any], run_dir: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    neighbor_manifest = json.loads((run_dir / "neighbor_manifest.json").read_text())
    imputation_manifest = json.loads((run_dir / "imputation_manifest.json").read_text())
    if not neighbor_manifest.get("complete") or not imputation_manifest.get("complete"):
        raise ValueError("Validation requires complete neighbors and imputation.")
    neighbors = np.load(neighbor_manifest["neighbors_path"], mmap_mode="r")
    spatial_mean, graph_manifest = build_or_load_spatial_graph(config, run_dir, audit)
    source_indices = np.load(run_dir / "target_source_var_indices.npy")
    check_spots = min(100, len(neighbors))
    check_genes = min(25, len(source_indices))
    full_source = base.load_h5ad_csr(Path(config["source_rna_h5ad"]))
    raw = full_source[:, source_indices[:check_genes]].tocsr()
    del full_source
    alpha = float(config["smoothing"]["alpha"])
    smoothed = (
        raw.multiply(np.float32(1.0 - alpha))
        + (spatial_mean @ raw).multiply(np.float32(alpha))
    ).tocsr()
    neighbor_check = np.asarray(neighbors[:check_spots])
    expected = (
        smoothed[neighbor_check.reshape(-1)]
        .toarray()
        .reshape(check_spots, neighbor_check.shape[1], check_genes)
        .mean(axis=1)
        .astype(np.float32)
    )
    output_path = Path(imputation_manifest["output_path"])
    with h5py.File(output_path, "r") as handle:
        observed = handle["X"][:check_spots, :check_genes]
        completed = handle["completed_gene_blocks"][:]
        output_shape = list(handle["X"].shape)
        output_dtype = str(handle["X"].dtype)
        output_complete = bool(handle.attrs.get("complete", False))
        sampled = handle["X"][: min(2048, len(neighbors)), : min(64, len(source_indices))]
    row_sums = np.asarray(spatial_mean.sum(axis=1)).reshape(-1)
    difference = np.abs(observed - expected)
    checks = {
        "config_hash_match": imputation_manifest.get("config_hash") == audit["config_hash"],
        "hdf5_complete": output_complete,
        "all_gene_blocks_complete": bool(np.all(completed)),
        "shape_match": output_shape
        == [int(audit["effective_target_spots"]), int(audit["effective_imputation_gene_count"])],
        "dtype_float32": output_dtype == "float32",
        "sampled_output_finite": bool(np.isfinite(sampled).all()),
        "spatial_graph_complete": bool(graph_manifest.get("complete")),
        "spatial_graph_nonnegative": bool(spatial_mean.data.min() >= 0),
        "spatial_graph_rows_sum_to_one": bool(
            np.allclose(row_sums, 1.0, rtol=0, atol=1e-6)
        ),
        "direct_smoothed_cosie_loop_allclose": bool(
            np.allclose(observed, expected, rtol=1e-6, atol=1e-7)
        ),
    }
    validation = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "direct_loop_check": {
            "spots": check_spots,
            "genes": check_genes,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        },
    }
    base.atomic_write_json(run_dir / "validation.json", validation)
    if validation["status"] != "pass":
        raise RuntimeError(
            f"Validation failed: {[name for name, passed in checks.items() if not passed]}"
        )
    base.LOGGER.info(
        "Validation passed; direct smoothed loop max abs error=%.3g",
        validation["direct_loop_check"]["max_absolute_error"],
    )
    return validation


def compare_to_baseline(
    config: dict[str, Any], run_dir: Path, smoothed_path: Path
) -> dict[str, Any]:
    baseline_path = Path(config["baseline_output"])
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline output not found: {baseline_path}")
    sample_spots = int(config.get("comparison", {}).get("sample_target_spots", 20_000))
    with h5py.File(smoothed_path, "r") as smooth, h5py.File(baseline_path, "r") as baseline:
        if smooth["X"].shape != baseline["X"].shape:
            raise ValueError("Smoothed and baseline output shapes differ.")
        smooth_genes = smooth["var_names"].asstr()[:]
        baseline_genes = baseline["var_names"].asstr()[:]
        if not np.array_equal(smooth_genes, baseline_genes):
            raise ValueError("Smoothed and baseline gene orders differ.")
        sample_spots = min(sample_spots, smooth["X"].shape[0])
        gene_count = smooth["X"].shape[1]
        smooth_full_means = smooth["gene_stats"]["predicted_mean"][:]
        baseline_full_means = baseline["gene_stats"]["predicted_mean"][:]
        smooth_nonzero_fractions = smooth["gene_stats"]["predicted_nonzero_fraction"][:]
        baseline_nonzero_fractions = baseline["gene_stats"]["predicted_nonzero_fraction"][:]
        block_size = int(config["imputation"]["gene_block_size"])
        rows: list[pd.DataFrame] = []
        total_abs = total_squared = total_a = total_b = 0.0
        total_aa = total_bb = total_ab = 0.0
        total_exact = total_values = 0
        for start in range(0, gene_count, block_size):
            end = min(gene_count, start + block_size)
            a = np.asarray(smooth["X"][:sample_spots, start:end], dtype=np.float64)
            b = np.asarray(baseline["X"][:sample_spots, start:end], dtype=np.float64)
            difference = a - b
            sum_a = a.sum(axis=0)
            sum_b = b.sum(axis=0)
            sum_aa = np.square(a).sum(axis=0)
            sum_bb = np.square(b).sum(axis=0)
            sum_ab = (a * b).sum(axis=0)
            covariance = sum_ab - sum_a * sum_b / sample_spots
            variance_a = sum_aa - np.square(sum_a) / sample_spots
            variance_b = sum_bb - np.square(sum_b) / sample_spots
            denominator = np.sqrt(np.maximum(variance_a * variance_b, 0))
            correlation = np.divide(
                covariance,
                denominator,
                out=np.full_like(covariance, np.nan),
                where=denominator > 0,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "target_gene_index": np.arange(start, end, dtype=np.int32),
                        "gene": smooth_genes[start:end],
                        "baseline_full_mean": baseline_full_means[start:end],
                        "smoothed_full_mean": smooth_full_means[start:end],
                        "sample_mean_absolute_difference": np.mean(np.abs(difference), axis=0),
                        "sample_rmse": np.sqrt(np.mean(np.square(difference), axis=0)),
                        "sample_pearson": correlation,
                    }
                )
            )
            total_abs += float(np.abs(difference).sum())
            total_squared += float(np.square(difference).sum())
            total_a += float(sum_a.sum())
            total_b += float(sum_b.sum())
            total_aa += float(sum_aa.sum())
            total_bb += float(sum_bb.sum())
            total_ab += float(sum_ab.sum())
            total_exact += int(np.count_nonzero(a == b))
            total_values += int(a.size)
    table = pd.concat(rows, ignore_index=True)
    table_path = run_dir / "comparison_to_unsmoothed_gene_metrics.csv.gz"
    table.to_csv(table_path, index=False, compression="gzip")
    covariance = total_ab - total_a * total_b / total_values
    variance_a = total_aa - total_a**2 / total_values
    variance_b = total_bb - total_b**2 / total_values
    baseline_neighbor_manifest_path = baseline_path.parent / "neighbor_manifest.json"
    current_neighbor_manifest_path = run_dir / "neighbor_manifest.json"
    neighbor_tables_exact = None
    if baseline_neighbor_manifest_path.is_file() and current_neighbor_manifest_path.is_file():
        baseline_neighbor_manifest = json.loads(
            baseline_neighbor_manifest_path.read_text(encoding="utf-8")
        )
        current_neighbor_manifest = json.loads(
            current_neighbor_manifest_path.read_text(encoding="utf-8")
        )
        baseline_neighbors = np.load(
            baseline_neighbor_manifest["neighbors_path"], mmap_mode="r"
        )
        current_neighbors = np.load(
            current_neighbor_manifest["neighbors_path"], mmap_mode="r"
        )
        neighbor_tables_exact = bool(np.array_equal(baseline_neighbors, current_neighbors))
    comparison = {
        "baseline_path": str(baseline_path),
        "neighbor_tables_elementwise_identical": neighbor_tables_exact,
        "sample_strategy": "first_target_rows_in_preserved_raw_order",
        "sample_target_spots": sample_spots,
        "genes": int(len(table)),
        "sample_values": total_values,
        "sample_global_mae": total_abs / total_values,
        "sample_global_rmse": math.sqrt(total_squared / total_values),
        "sample_global_pearson": covariance / math.sqrt(variance_a * variance_b),
        "sample_exact_fraction": total_exact / total_values,
        "per_gene_sample_pearson_quantiles": {
            str(prob): float(value)
            for prob, value in zip(
                (0, 0.25, 0.5, 0.75, 0.99, 1),
                table["sample_pearson"].dropna().quantile(
                    (0, 0.25, 0.5, 0.75, 0.99, 1)
                ),
            )
        },
        "per_gene_sample_mae_quantiles": {
            str(prob): float(value)
            for prob, value in zip(
                (0, 0.25, 0.5, 0.75, 0.99, 1),
                table["sample_mean_absolute_difference"].quantile(
                    (0, 0.25, 0.5, 0.75, 0.99, 1)
                ),
            )
        },
        "full_output_distribution": {
            "baseline_grand_mean": float(np.mean(baseline_full_means)),
            "smoothed_grand_mean": float(np.mean(smooth_full_means)),
            "baseline_zero_genes": int(np.count_nonzero(baseline_nonzero_fractions == 0)),
            "smoothed_zero_genes": int(np.count_nonzero(smooth_nonzero_fractions == 0)),
            "baseline_median_gene_nonzero_fraction": float(
                np.median(baseline_nonzero_fractions)
            ),
            "smoothed_median_gene_nonzero_fraction": float(
                np.median(smooth_nonzero_fractions)
            ),
        },
        "per_gene_table": str(table_path),
    }
    base.atomic_write_json(run_dir / "comparison_to_unsmoothed.json", comparison)
    return comparison


def full_imputation_performance_from_log(run_dir: Path) -> dict[str, Any] | None:
    """Recover measured full-run timing and peak RSS from block progress logs."""
    log_path = run_dir / "pipeline.log"
    if not log_path.is_file():
        return None
    pattern = re.compile(
        r"Smoothed gene blocks: (\d+)/(\d+).*elapsed=([0-9.]+)s.*"
        r"'peak_rss_gib': ([0-9.]+)"
    )
    records = [
        (int(done), int(total), float(elapsed), float(peak))
        for done, total, elapsed, peak in pattern.findall(
            log_path.read_text(encoding="utf-8")
        )
    ]
    complete = [record for record in records if record[0] == record[1]]
    if not complete:
        return None
    last = complete[-1]
    return {
        "source": str(log_path),
        "completed_gene_blocks": last[1],
        "elapsed_seconds": last[2],
        "peak_rss_gib": max(record[3] for record in records),
    }


def summarize_smoothed(
    config: dict[str, Any], run_dir: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    summary = base.summarize(config, run_dir, audit)
    output_path = Path(summary["output"]["output_path"])
    comparison = compare_to_baseline(config, run_dir, output_path)
    graph_manifest = json.loads(
        (run_dir / "source_spatial_neighbor_mean_manifest.json").read_text(encoding="utf-8")
    )
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    summary["format"] = FORMAT_NAME
    summary["method"] = "one-step spatial residual reference smoothing, then COSIE-compatible KNN mean"
    summary["smoothing"] = config["smoothing"]
    summary["spatial_graph"] = graph_manifest
    summary["comparison_to_unsmoothed"] = comparison
    summary["validation"] = validation
    full_performance = full_imputation_performance_from_log(run_dir)
    summary["full_imputation_performance"] = full_performance
    base.atomic_write_json(run_dir / "run_summary.json", summary)
    performance_text = (
        f"{full_performance['elapsed_seconds']:.1f} seconds; peak RSS "
        f"{full_performance['peak_rss_gib']:.2f} GiB"
        if full_performance
        else "not available"
    )
    text = f"""# {config['run_name']}\n\nStatus: **complete**\n\n- Method: section2 spatial residual smoothing, then COSIE-compatible Annoy KNN mean\n- Smoothing: `(1-alpha) X + alpha S X`, alpha={config['smoothing']['alpha']}\n- Local graph: radius {config['smoothing']['radius_grid_units']} grid units (~{graph_manifest['radius_um_approx']:.2f} um), {len(graph_manifest['offsets'])} allowed offsets\n- Isolated reference spots kept unchanged: {graph_manifest['isolated_spots']:,}\n- Transfer: `{config['source_section']}` -> `{config['target_section']}`\n- Shapes: {summary['target_spots']:,} target spots x {summary['target_genes']:,} source-only genes\n- K / metric: {summary['k']} / `{summary['metric']}`\n- Direct smoothed-loop validation: {validation['status']}\n- KNN table elementwise identical to unsmoothed control: {comparison['neighbor_tables_elementwise_identical']}\n- Compared with unsmoothed baseline on first {comparison['sample_target_spots']:,} preserved-order target rows\n- Baseline comparison global Pearson: {comparison['sample_global_pearson']:.6g}\n- Baseline comparison global MAE: {comparison['sample_global_mae']:.6g}\n- Result: `{output_path}`\n- Per-gene baseline comparison: `{comparison['per_gene_table']}`\n- Compressed file size: {output_path.stat().st_size / 1024**3:.2f} GiB\n- Measured full imputation: {performance_text}\n\nThis is an alpha={config['smoothing']['alpha']} comparison run.  Keep the original unsmoothed result as alpha=0.\n"""
    (run_dir / "SUMMARY.md").write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config, run_dir, config_hash = base.load_config(args.config.resolve(), args.run_dir)
    if (args.limit_target_spots is not None or args.limit_genes is not None) and args.run_dir is None:
        raise ValueError("Limited tests require a separate explicit --run-dir.")
    base.setup_logging(run_dir)
    base.LOGGER.info(
        "Run=%s stage=%s config_hash=%s", config["run_name"], args.stage, config_hash
    )
    if args.stage in ("audit", "all"):
        audit = base.audit_inputs(
            config,
            run_dir,
            config_hash,
            args.limit_target_spots,
            args.limit_genes,
        )
    else:
        audit = base.load_or_require_audit(run_dir, config_hash)
    if args.stage in ("neighbors", "all"):
        base.compute_neighbors(config, run_dir, audit)
    if args.stage in ("spatial_graph", "all"):
        build_or_load_spatial_graph(config, run_dir, audit)
    if args.stage in ("impute", "all"):
        impute_smoothed(config, run_dir, audit)
    if args.stage in ("validate", "all"):
        validate_smoothed(config, run_dir, audit)
    if args.stage in ("summarize", "all"):
        summarize_smoothed(config, run_dir, audit)
    base.LOGGER.info("Requested stage completed successfully. %s", base.memory_gib())


if __name__ == "__main__":
    main()
