#!/usr/bin/env python3
"""Compute scheme-1 raw-scale PCC for both shared-gene imputation variants.

The metric definitions are implemented locally.  For a spots-by-genes matrix,
``pcc_cell`` is the mean of finite row-wise Pearson correlations and
``pcc_gene`` is the mean of finite column-wise Pearson correlations.

The full 665,399 by 4,828 prediction matrices are never materialized.  Gene
PCC is evaluated blockwise, while algebraically equivalent row moments are
accumulated to recover exact per-cell PCC after the last gene block.
"""

from __future__ import annotations

import argparse
import hashlib
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "gene_imputation"
sys.path.insert(0, str(BASE_DIR))
import run_spatch_knn_imputation as base  # noqa: E402


FORMAT_NAME = "spa_mo_model.scheme1_raw_pcc"
FORMAT_VERSION = 1
CHECKPOINT_NAME = "scheme1_pcc_checkpoint.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-every-blocks",
        type=int,
        default=8,
        help="Persist row moments after this many completed gene blocks.",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config, Path(config["run_dir"]).resolve(), config_hash


def pearson_values(matrix1: np.ndarray, matrix2: np.ndarray, axis: int) -> np.ndarray:
    """Vectorized Pearson r along rows (axis=1) or columns (axis=0)."""
    a = np.asarray(matrix1, dtype=np.float64)
    b = np.asarray(matrix2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"PCC inputs must be equal 2-D arrays: {a.shape} vs {b.shape}")
    am = a.mean(axis=axis, keepdims=True)
    bm = b.mean(axis=axis, keepdims=True)
    ac = a - am
    bc = b - bm
    numerator = (ac * bc).sum(axis=axis)
    denominator = np.sqrt((ac**2).sum(axis=axis) * (bc**2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def mean_finite(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    return (
        float(np.mean(values[valid])) if valid.any() else float("nan"),
        int(np.count_nonzero(valid)),
    )


def empty_row_moments(n_spots: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(n_spots, dtype=np.float64)
        for name in (
            "true_sum",
            "true_sum2",
            "unsmoothed_sum",
            "unsmoothed_sum2",
            "unsmoothed_cross",
            "smoothed_sum",
            "smoothed_sum2",
            "smoothed_cross",
        )
    }


def update_true_moments(moments: dict[str, np.ndarray], true: np.ndarray) -> None:
    moments["true_sum"] += np.sum(true, axis=1, dtype=np.float64)
    moments["true_sum2"] += np.einsum("ij,ij->i", true, true, dtype=np.float64)


def update_prediction_moments(
    moments: dict[str, np.ndarray], name: str, pred: np.ndarray, true: np.ndarray
) -> None:
    moments[f"{name}_sum"] += np.sum(pred, axis=1, dtype=np.float64)
    moments[f"{name}_sum2"] += np.einsum("ij,ij->i", pred, pred, dtype=np.float64)
    moments[f"{name}_cross"] += np.einsum("ij,ij->i", pred, true, dtype=np.float64)


def finalize_row_pcc(
    moments: dict[str, np.ndarray], name: str, n_genes: int
) -> np.ndarray:
    true_sum = moments["true_sum"]
    pred_sum = moments[f"{name}_sum"]
    numerator = moments[f"{name}_cross"] - pred_sum * true_sum / n_genes
    true_variance = moments["true_sum2"] - np.square(true_sum) / n_genes
    pred_variance = moments[f"{name}_sum2"] - np.square(pred_sum) / n_genes
    denominator = np.sqrt(np.maximum(true_variance * pred_variance, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def initialize_checkpoint(
    path: Path,
    config_hash: str,
    n_spots: int,
    n_genes: int,
    block_size: int,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "config_hash": config_hash,
                "n_spots": n_spots,
                "n_genes": n_genes,
                "block_size": block_size,
                "completed_gene_end": 0,
                "complete": False,
                "elapsed_seconds": 0.0,
            }
        )
        row_group = handle.create_group("row_moments")
        for name in empty_row_moments(0):
            row_group.create_dataset(name, shape=(n_spots,), dtype=np.float64, fillvalue=0.0)
        gene_group = handle.create_group("gene_pcc")
        for name in ("unsmoothed", "smoothed"):
            gene_group.create_dataset(name, shape=(n_genes,), dtype=np.float64, fillvalue=np.nan)


def load_checkpoint(
    path: Path, config_hash: str, n_spots: int, n_genes: int, block_size: int
) -> tuple[int, float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not path.exists():
        initialize_checkpoint(path, config_hash, n_spots, n_genes, block_size)
    with h5py.File(path, "r") as handle:
        expected = (n_spots, n_genes, block_size)
        observed = (
            int(handle.attrs["n_spots"]),
            int(handle.attrs["n_genes"]),
            int(handle.attrs["block_size"]),
        )
        if handle.attrs.get("config_hash") != config_hash or observed != expected:
            raise ValueError("Existing PCC checkpoint is incompatible with this run.")
        completed_end = int(handle.attrs["completed_gene_end"])
        elapsed_seconds = float(handle.attrs.get("elapsed_seconds", 0.0))
        moments = {name: handle["row_moments"][name][:] for name in handle["row_moments"]}
        gene_pcc = {name: handle["gene_pcc"][name][:] for name in handle["gene_pcc"]}
    return completed_end, elapsed_seconds, moments, gene_pcc


def save_checkpoint(
    path: Path,
    completed_end: int,
    moments: dict[str, np.ndarray],
    gene_pcc: dict[str, np.ndarray],
    complete: bool,
    elapsed_seconds: float,
) -> None:
    with h5py.File(path, "r+") as handle:
        for name, values in moments.items():
            handle["row_moments"][name][:] = values
        for name, values in gene_pcc.items():
            handle["gene_pcc"][name][:] = values
        handle.attrs["completed_gene_end"] = completed_end
        handle.attrs["complete"] = complete
        handle.attrs["elapsed_seconds"] = elapsed_seconds
        handle.flush()


def load_shared_context(
    config: dict[str, Any], run_dir: Path, config_hash: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix, Path, Path]:
    audit = json.loads((run_dir / "input_audit.json").read_text(encoding="utf-8"))
    if audit.get("config_hash") != config_hash:
        raise ValueError("Shared-gene audit and configuration hashes differ.")
    genes = pd.read_csv(run_dir / "shared_genes.csv")["gene"].astype(str).to_numpy()
    source_indices = np.load(run_dir / "shared_source_var_indices.npy")
    target_indices = np.load(run_dir / "shared_target_var_indices.npy")
    if not (len(genes) == len(source_indices) == len(target_indices)):
        raise ValueError("Shared-gene metadata arrays differ in length.")

    baseline_manifest_path = Path(config["baseline_neighbor_manifest"]).resolve()
    smoothed_manifest_path = Path(config["smoothed_neighbor_manifest"]).resolve()
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    smoothed_manifest = json.loads(smoothed_manifest_path.read_text(encoding="utf-8"))
    baseline_neighbors = np.load(baseline_manifest["neighbors_path"], mmap_mode="r")
    smoothed_neighbors = np.load(smoothed_manifest["neighbors_path"], mmap_mode="r")
    if not np.array_equal(baseline_neighbors, smoothed_neighbors):
        raise ValueError("The two variants do not use the same KNN table.")
    transfer = base.make_uniform_transfer_matrix(
        baseline_neighbors, int(baseline_manifest["source_count"])
    )
    spatial_manifest = json.loads(
        Path(config["spatial_graph_manifest"]).read_text(encoding="utf-8")
    )
    spatial_mean = sp.load_npz(spatial_manifest["path"]).tocsr().astype(np.float32)
    return (
        genes,
        source_indices,
        target_indices,
        baseline_neighbors,
        spatial_mean,
        baseline_manifest_path.parent,
        smoothed_manifest_path.parent,
    )


def variant_payload(
    summary: dict[str, Any], variant: str, shared_summary_path: Path
) -> dict[str, Any]:
    result = summary["variants"][variant]
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "language": "zh-CN",
        "status": "complete",
        "variant": variant,
        "evaluation_scope": "使用section1实测值作为伪真值的4,828个共享基因",
        "production_source_only_genes_are_not_directly_scored": True,
        "expression_version": "scheme1_raw_scale",
        "preprocessing": {
            "prediction": "source原始count经过embedding-KNN邻居均值聚合",
            "truth": "section1原始count",
            "normalize_total": False,
            "log1p": False,
            "scale": False,
        },
        "aggregation_standard": "对有限的逐行或逐列Pearson r取算术平均值",
        "pcc_cell": result["pcc_cell"],
        "pcc_gene": result["pcc_gene"],
        "valid_cells": result["valid_cells"],
        "valid_genes": result["valid_genes"],
        "shared_summary": str(shared_summary_path),
    }


def logged_scheme1_peak_rss_gib(run_dir: Path) -> float:
    """Recover the peak recorded during the full calculation on cached reruns."""
    log_path = run_dir / "pipeline.log"
    if not log_path.is_file():
        return 0.0
    pattern = re.compile(r"'peak_rss_gib': ([0-9.eE+-]+)")
    peaks: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "Scheme1 PCC block" not in line:
            continue
        match = pattern.search(line)
        if match:
            peaks.append(float(match.group(1)))
    return max(peaks, default=0.0)


def write_reports(
    run_dir: Path,
    baseline_run_dir: Path,
    smoothed_run_dir: Path,
    genes: np.ndarray,
    cell_pcc: dict[str, np.ndarray],
    gene_pcc: dict[str, np.ndarray],
    config_hash: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    per_gene_path = run_dir / "scheme1_raw_pcc_per_gene.csv.gz"
    pd.DataFrame(
        {
            "gene": genes,
            "unsmoothed_pcc": gene_pcc["unsmoothed"],
            "smoothed_pcc": gene_pcc["smoothed"],
        }
    ).to_csv(per_gene_path, index=False, compression="gzip")
    per_cell_path = run_dir / "scheme1_raw_pcc_per_cell.npz"
    np.savez_compressed(
        per_cell_path,
        unsmoothed_pcc=cell_pcc["unsmoothed"],
        smoothed_pcc=cell_pcc["smoothed"],
    )

    variants: dict[str, Any] = {}
    for name in ("unsmoothed", "smoothed"):
        pcc_cell, valid_cells = mean_finite(cell_pcc[name])
        pcc_gene_value, valid_genes = mean_finite(gene_pcc[name])
        variants[name] = {
            "pcc_cell": pcc_cell,
            "pcc_gene": pcc_gene_value,
            "valid_cells": valid_cells,
            "valid_genes": valid_genes,
        }
    memory = base.memory_gib()
    memory["peak_rss_gib"] = max(
        memory["peak_rss_gib"], logged_scheme1_peak_rss_gib(run_dir)
    )
    summary = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "language": "zh-CN",
        "status": "complete",
        "config_hash": config_hash,
        "expression_version": "scheme1_raw_scale",
        "matrix_orientation": "spots_by_genes",
        "spots": int(len(cell_pcc["unsmoothed"])),
        "shared_genes": int(len(genes)),
        "preprocessing": {
            "prediction": "source原始count经过embedding-KNN邻居均值聚合",
            "truth": "section1原始count",
            "normalize_total": False,
            "log1p": False,
            "scale": False,
        },
        "metric_definition": {
            "pcc_cell": "逐行计算Pearson r，再对有限结果取算术平均值",
            "pcc_gene": "逐列计算Pearson r，再对有限结果取算术平均值",
        },
        "variants": variants,
        "delta_smoothed_minus_unsmoothed": {
            "pcc_cell": variants["smoothed"]["pcc_cell"]
            - variants["unsmoothed"]["pcc_cell"],
            "pcc_gene": variants["smoothed"]["pcc_gene"]
            - variants["unsmoothed"]["pcc_gene"],
        },
        "paths": {
            "per_cell": str(per_cell_path),
            "per_gene": str(per_gene_path),
        },
        "elapsed_seconds": elapsed_seconds,
        "memory": memory,
        "interpretation_boundary": (
            "这些分数只评价4,828个共享基因。13,257个source-only正式预测基因"
            "在section1中没有实测真值。"
        ),
    }
    summary_path = run_dir / "scheme1_raw_pcc_summary.json"
    base.atomic_write_json(summary_path, summary)

    for path, variant in (
        (baseline_run_dir / "scheme1_shared4828_pcc.json", "unsmoothed"),
        (smoothed_run_dir / "scheme1_shared4828_pcc.json", "smoothed"),
    ):
        base.atomic_write_json(path, variant_payload(summary, variant, summary_path))
        value = summary["variants"][variant]
        variant_label = "未平滑" if variant == "unsmoothed" else "平滑"
        text = f"""# 方案1共享基因PCC

状态：**已完成**

- 方案：{variant_label}（`{variant}`）
- 表达尺度：raw source-count KNN预测值与section1 raw真值
- 共享基因数：{len(genes):,}
- 目标空间点数：{len(cell_pcc[variant]):,}
- `pcc_cell`：{value['pcc_cell']:.8f}（{value['valid_cells']:,}个有限结果空间点）
- `pcc_gene`：{value['pcc_gene']:.8f}（{value['valid_genes']:,}个有限结果基因）
- 汇总方式：对有限的逐空间点或逐基因Pearson相关系数取算术平均值
- 机器可读结果：`{path}`
- 两版共享基因对照结果：`{summary_path}`

这些数值评价的是作为伪缺失目标的4,828个共享基因。由于section1没有
13,257个source-only正式预测基因的实测真值，因此这些数值不是这13,257个
基因的直接准确性结果。
"""
        (path.parent / "SCHEME1_PCC.md").write_text(text, encoding="utf-8")

    report = f"""# 方案1 raw-scale PCC

状态：**已完成**

| 方案 | pcc_cell | 有效空间点数 | pcc_gene | 有效基因数 |
|---|---:|---:|---:|---:|
| 未平滑 | {variants['unsmoothed']['pcc_cell']:.8f} | {variants['unsmoothed']['valid_cells']:,} | {variants['unsmoothed']['pcc_gene']:.8f} | {variants['unsmoothed']['valid_genes']:,} |
| 平滑 | {variants['smoothed']['pcc_cell']:.8f} | {variants['smoothed']['valid_cells']:,} | {variants['smoothed']['pcc_gene']:.8f} | {variants['smoothed']['valid_genes']:,} |

- 平滑减去未平滑的`pcc_cell`：{summary['delta_smoothed_minus_unsmoothed']['pcc_cell']:+.8f}
- 平滑减去未平滑的`pcc_gene`：{summary['delta_smoothed_minus_unsmoothed']['pcc_gene']:+.8f}
- 输入版本：raw预测值和raw真值；不使用`normalize_total`、`log1p`或`scale`
- 汇总方式：对有限的逐行或逐列Pearson相关系数取算术平均值
- 计算时间：{elapsed_seconds / 60:.2f}分钟
- 逐空间点结果：`{per_cell_path}`
- 逐基因结果：`{per_gene_path}`
- 机器可读汇总：`{summary_path}`

## 解释边界

这里评价的是作为伪缺失目标的4,828个共享基因。对于section1中真正未测量的
13,257个基因，本结果不能提供基于实测真值的准确性。
"""
    (run_dir / "SCHEME1_PCC.md").write_text(report, encoding="utf-8")
    return summary


def run(config_path: Path, checkpoint_every_blocks: int) -> dict[str, Any]:
    if checkpoint_every_blocks <= 0:
        raise ValueError("checkpoint_every_blocks must be positive.")
    config, run_dir, config_hash = load_config(config_path.resolve())
    base.setup_logging(run_dir)
    (
        genes,
        source_indices,
        target_indices,
        neighbors,
        spatial_mean,
        baseline_run_dir,
        smoothed_run_dir,
    ) = load_shared_context(config, run_dir, config_hash)
    n_spots = int(neighbors.shape[0])
    n_genes = len(genes)
    block_size = int(config["gene_block_size"])
    checkpoint_path = run_dir / CHECKPOINT_NAME
    completed_end, prior_elapsed_seconds, moments, gene_pcc = load_checkpoint(
        checkpoint_path, config_hash, n_spots, n_genes, block_size
    )
    if completed_end == n_genes:
        base.LOGGER.info("PCC checkpoint is already complete; regenerating reports.")
        cell_pcc = {
            name: finalize_row_pcc(moments, name, n_genes)
            for name in ("unsmoothed", "smoothed")
        }
        return write_reports(
            run_dir,
            baseline_run_dir,
            smoothed_run_dir,
            genes,
            cell_pcc,
            gene_pcc,
            config_hash,
            prior_elapsed_seconds,
        )

    transfer = base.make_uniform_transfer_matrix(neighbors, spatial_mean.shape[0])
    alpha = np.float32(config["smoothing_alpha"])
    base.LOGGER.info("Loading source and truth raw-count matrices for scheme-1 PCC.")
    source_full = base.load_h5ad_csr(Path(config["source_rna_h5ad"]))
    source_shared = source_full[:, source_indices].tocsc().astype(np.float32, copy=False)
    del source_full
    target_full = base.load_h5ad_csr(Path(config["target_rna_h5ad"]))
    target_shared = target_full[:, target_indices].tocsc().astype(np.float32, copy=False)
    del target_full
    source_shared.sort_indices()
    target_shared.sort_indices()

    started = time.time()
    start_block = completed_end // block_size
    total_blocks = math.ceil(n_genes / block_size)
    for block_index in range(start_block, total_blocks):
        start = block_index * block_size
        end = min(n_genes, start + block_size)
        block_started = time.time()
        source_block = source_shared[:, start:end].tocsr()
        true = np.asarray(target_shared[:, start:end].toarray(), dtype=np.float32)
        unsmoothed = base.aggregate_block(transfer, source_block)
        smoothed_source = (
            source_block.multiply(np.float32(1.0) - alpha)
            + (spatial_mean @ source_block).multiply(alpha)
        ).tocsr()
        smoothed_source.sum_duplicates()
        smoothed_source.eliminate_zeros()
        smoothed = base.aggregate_block(transfer, smoothed_source)

        gene_pcc["unsmoothed"][start:end] = pearson_values(
            unsmoothed, true, axis=0
        )
        gene_pcc["smoothed"][start:end] = pearson_values(smoothed, true, axis=0)
        update_true_moments(moments, true)
        update_prediction_moments(moments, "unsmoothed", unsmoothed, true)
        update_prediction_moments(moments, "smoothed", smoothed, true)
        del source_block, true, unsmoothed, smoothed_source, smoothed

        should_checkpoint = (
            (block_index + 1) % checkpoint_every_blocks == 0
            or end == n_genes
        )
        if should_checkpoint:
            save_checkpoint(
                checkpoint_path,
                end,
                moments,
                gene_pcc,
                complete=end == n_genes,
                elapsed_seconds=prior_elapsed_seconds + time.time() - started,
            )
        base.LOGGER.info(
            "Scheme1 PCC block %d/%d genes=%d:%d block=%.1fs elapsed=%.1fs %s",
            block_index + 1,
            total_blocks,
            start,
            end,
            time.time() - block_started,
            time.time() - started,
            base.memory_gib(),
        )

    cell_pcc = {
        name: finalize_row_pcc(moments, name, n_genes)
        for name in ("unsmoothed", "smoothed")
    }
    summary = write_reports(
        run_dir,
        baseline_run_dir,
        smoothed_run_dir,
        genes,
        cell_pcc,
        gene_pcc,
        config_hash,
        prior_elapsed_seconds + time.time() - started,
    )
    base.LOGGER.info("Scheme-1 PCC complete: %s", json.dumps(summary["variants"]))
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args.config, args.checkpoint_every_blocks)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
