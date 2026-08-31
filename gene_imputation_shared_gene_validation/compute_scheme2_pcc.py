#!/usr/bin/env python3
"""计算方案2 normalize_total + log1p尺度的pcc_cell和pcc_gene。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import compute_scheme1_pcc as common


FORMAT_NAME = "spa_mo_model.scheme2_normlog_pcc"
FORMAT_VERSION = 1
CHECKPOINT_NAME = "scheme2_pcc_checkpoint.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_scheme2_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("仅支持schema_version=1。")
    target_sum = float(config["normalize_total_target_sum"])
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("normalize_total_target_sum必须是有限正数。")
    base_path = Path(config["base_validation_config"]).resolve()
    base_config, run_dir, base_hash = common.load_config(base_path)
    canonical = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    scheme2_hash = hashlib.sha256(
        f"{canonical}|base_config_hash={base_hash}".encode("utf-8")
    ).hexdigest()
    return config, base_config, run_dir, base_hash, scheme2_hash


def normalize_total_log1p(
    matrix: np.ndarray, row_totals: np.ndarray, target_sum: float
) -> np.ndarray:
    """在给定完整行总量的条件下，对一个基因块执行normalize_total和log1p。"""
    values = np.asarray(matrix, dtype=np.float64)
    totals = np.asarray(row_totals, dtype=np.float64)
    if values.ndim != 2 or totals.shape != (values.shape[0],):
        raise ValueError("表达矩阵与行总量形状不匹配。")
    if np.any(values < 0) or np.any(totals < 0):
        raise ValueError("normalize_total输入不能包含负数。")
    positive = totals > 0
    if np.any(positive):
        values[positive] *= (target_sum / totals[positive])[:, np.newaxis]
    if np.any(~positive):
        values[~positive] = 0.0
    np.log1p(values, out=values)
    return values


def initialize_checkpoint(
    path: Path,
    scheme2_hash: str,
    base_hash: str,
    n_spots: int,
    n_genes: int,
    block_size: int,
    target_sum: float,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "scheme2_config_hash": scheme2_hash,
                "base_config_hash": base_hash,
                "n_spots": n_spots,
                "n_genes": n_genes,
                "block_size": block_size,
                "normalize_total_target_sum": target_sum,
                "completed_gene_end": 0,
                "complete": False,
                "elapsed_seconds": 0.0,
            }
        )
        row_group = handle.create_group("row_moments")
        for name in common.empty_row_moments(0):
            row_group.create_dataset(
                name, shape=(n_spots,), dtype=np.float64, fillvalue=0.0
            )
        gene_group = handle.create_group("gene_pcc")
        for name in ("unsmoothed", "smoothed"):
            gene_group.create_dataset(
                name, shape=(n_genes,), dtype=np.float64, fillvalue=np.nan
            )


def load_checkpoint(
    path: Path,
    scheme2_hash: str,
    base_hash: str,
    n_spots: int,
    n_genes: int,
    block_size: int,
    target_sum: float,
) -> tuple[int, float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not path.exists():
        initialize_checkpoint(
            path,
            scheme2_hash,
            base_hash,
            n_spots,
            n_genes,
            block_size,
            target_sum,
        )
    with h5py.File(path, "r") as handle:
        expected_shape = (n_spots, n_genes, block_size)
        observed_shape = (
            int(handle.attrs["n_spots"]),
            int(handle.attrs["n_genes"]),
            int(handle.attrs["block_size"]),
        )
        compatible = (
            handle.attrs.get("scheme2_config_hash") == scheme2_hash
            and handle.attrs.get("base_config_hash") == base_hash
            and observed_shape == expected_shape
            and float(handle.attrs["normalize_total_target_sum"]) == target_sum
        )
        if not compatible:
            raise ValueError("现有方案2 PCC断点与当前配置不兼容。")
        completed_end = int(handle.attrs["completed_gene_end"])
        elapsed_seconds = float(handle.attrs.get("elapsed_seconds", 0.0))
        moments = {
            name: handle["row_moments"][name][:]
            for name in handle["row_moments"]
        }
        gene_pcc = {
            name: handle["gene_pcc"][name][:] for name in handle["gene_pcc"]
        }
    return completed_end, elapsed_seconds, moments, gene_pcc


def save_checkpoint(
    path: Path,
    completed_end: int,
    elapsed_seconds: float,
    moments: dict[str, np.ndarray],
    gene_pcc: dict[str, np.ndarray],
    complete: bool,
) -> None:
    with h5py.File(path, "r+") as handle:
        for name, values in moments.items():
            handle["row_moments"][name][:] = values
        for name, values in gene_pcc.items():
            handle["gene_pcc"][name][:] = values
        handle.attrs["completed_gene_end"] = completed_end
        handle.attrs["elapsed_seconds"] = elapsed_seconds
        handle.attrs["complete"] = complete
        handle.flush()


def load_scheme1_row_totals(
    run_dir: Path,
    base_hash: str,
    n_spots: int,
    n_genes: int,
) -> dict[str, np.ndarray]:
    path = run_dir / common.CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            "缺少方案1断点；方案2需要复用其中完整4,828基因的raw行总量。"
        )
    with h5py.File(path, "r") as handle:
        compatible = (
            bool(handle.attrs.get("complete", False))
            and handle.attrs.get("config_hash") == base_hash
            and int(handle.attrs["n_spots"]) == n_spots
            and int(handle.attrs["n_genes"]) == n_genes
            and int(handle.attrs["completed_gene_end"]) == n_genes
        )
        if not compatible:
            raise ValueError("方案1断点不完整或与方案2输入不兼容。")
        group = handle["row_moments"]
        totals = {
            "true": group["true_sum"][:],
            "unsmoothed": group["unsmoothed_sum"][:],
            "smoothed": group["smoothed_sum"][:],
        }
    for name, values in totals.items():
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} raw行总量包含无效值。")
    return totals


def logged_scheme2_peak_rss_gib(run_dir: Path) -> float:
    log_path = run_dir / "pipeline.log"
    if not log_path.is_file():
        return 0.0
    pattern = re.compile(r"'peak_rss_gib': ([0-9.eE+-]+)")
    peaks: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "Scheme2 PCC block" not in line:
            continue
        match = pattern.search(line)
        if match:
            peaks.append(float(match.group(1)))
    return max(peaks, default=0.0)


def variant_payload(
    summary: dict[str, Any], variant: str, summary_path: Path
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
        "expression_version": summary["expression_version"],
        "preprocessing": summary["preprocessing"],
        "aggregation_standard": "对有限的逐行或逐列Pearson r取算术平均值",
        "pcc_cell": result["pcc_cell"],
        "pcc_gene": result["pcc_gene"],
        "valid_cells": result["valid_cells"],
        "valid_genes": result["valid_genes"],
        "shared_summary": str(summary_path),
    }


def write_reports(
    run_dir: Path,
    baseline_run_dir: Path,
    smoothed_run_dir: Path,
    genes: np.ndarray,
    cell_pcc: dict[str, np.ndarray],
    gene_pcc: dict[str, np.ndarray],
    scheme2_config: dict[str, Any],
    scheme2_hash: str,
    base_hash: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    prefix = str(scheme2_config["output_prefix"])
    per_gene_path = run_dir / f"{prefix}_per_gene.csv.gz"
    pd.DataFrame(
        {
            "gene": genes,
            "unsmoothed_pcc": gene_pcc["unsmoothed"],
            "smoothed_pcc": gene_pcc["smoothed"],
        }
    ).to_csv(per_gene_path, index=False, compression="gzip")
    per_cell_path = run_dir / f"{prefix}_per_cell.npz"
    np.savez_compressed(
        per_cell_path,
        unsmoothed_pcc=cell_pcc["unsmoothed"],
        smoothed_pcc=cell_pcc["smoothed"],
    )

    variants: dict[str, Any] = {}
    for name in ("unsmoothed", "smoothed"):
        pcc_cell, valid_cells = common.mean_finite(cell_pcc[name])
        pcc_gene_value, valid_genes = common.mean_finite(gene_pcc[name])
        variants[name] = {
            "pcc_cell": pcc_cell,
            "pcc_gene": pcc_gene_value,
            "valid_cells": valid_cells,
            "valid_genes": valid_genes,
        }
    memory = common.base.memory_gib()
    memory["peak_rss_gib"] = max(
        memory["peak_rss_gib"], logged_scheme2_peak_rss_gib(run_dir)
    )
    target_sum = float(scheme2_config["normalize_total_target_sum"])
    summary = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "language": "zh-CN",
        "status": "complete",
        "scheme2_config_hash": scheme2_hash,
        "base_config_hash": base_hash,
        "expression_version": scheme2_config["expression_version"],
        "matrix_orientation": "spots_by_genes",
        "spots": int(len(cell_pcc["unsmoothed"])),
        "shared_genes": int(len(genes)),
        "preprocessing": {
            "gene_universe": "两个切片中顺序一致的4,828个共享基因",
            "normalize_total": True,
            "normalize_total_target_sum": target_sum,
            "normalize_total_scope": "每个矩阵分别按空间点执行",
            "log1p": True,
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
    summary_path = run_dir / f"{prefix}_summary.json"
    common.base.atomic_write_json(summary_path, summary)

    for path, variant in (
        (baseline_run_dir / "scheme2_shared4828_pcc.json", "unsmoothed"),
        (smoothed_run_dir / "scheme2_shared4828_pcc.json", "smoothed"),
    ):
        common.base.atomic_write_json(
            path, variant_payload(summary, variant, summary_path)
        )
        value = variants[variant]
        variant_label = "未平滑" if variant == "unsmoothed" else "平滑"
        text = f"""# 方案2共享基因PCC

状态：**已完成**

- 方案：{variant_label}（`{variant}`）
- 表达预处理：在相同4,828个共享基因范围内，对pred和true分别执行
  `normalize_total(target_sum={target_sum:g}) → log1p`，不使用`scale`
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
        (path.parent / "SCHEME2_PCC.md").write_text(text, encoding="utf-8")

    report = f"""# 方案2 normalize_total + log1p PCC

状态：**已完成**

| 方案 | pcc_cell | 有效空间点数 | pcc_gene | 有效基因数 |
|---|---:|---:|---:|---:|
| 未平滑 | {variants['unsmoothed']['pcc_cell']:.8f} | {variants['unsmoothed']['valid_cells']:,} | {variants['unsmoothed']['pcc_gene']:.8f} | {variants['unsmoothed']['valid_genes']:,} |
| 平滑 | {variants['smoothed']['pcc_cell']:.8f} | {variants['smoothed']['valid_cells']:,} | {variants['smoothed']['pcc_gene']:.8f} | {variants['smoothed']['valid_genes']:,} |

- 平滑减去未平滑的`pcc_cell`：{summary['delta_smoothed_minus_unsmoothed']['pcc_cell']:+.8f}
- 平滑减去未平滑的`pcc_gene`：{summary['delta_smoothed_minus_unsmoothed']['pcc_gene']:+.8f}
- 预处理：在相同4,828个共享基因范围内，对pred和true分别执行
  `normalize_total(target_sum={target_sum:g}) → log1p`，不使用`scale`
- 汇总方式：对有限的逐行或逐列Pearson相关系数取算术平均值
- 计算时间：{elapsed_seconds / 60:.2f}分钟
- 逐空间点结果：`{per_cell_path}`
- 逐基因结果：`{per_gene_path}`
- 机器可读汇总：`{summary_path}`

## 解释边界

这里评价的是作为伪缺失目标的4,828个共享基因。对于section1中真正未测量的
13,257个基因，本结果不能提供基于实测真值的准确性。
"""
    (run_dir / "SCHEME2_PCC.md").write_text(report, encoding="utf-8")
    return summary


def run(config_path: Path) -> dict[str, Any]:
    (
        scheme2_config,
        base_config,
        run_dir,
        base_hash,
        scheme2_hash,
    ) = load_scheme2_config(config_path.resolve())
    common.base.setup_logging(run_dir)
    (
        genes,
        source_indices,
        target_indices,
        neighbors,
        spatial_mean,
        baseline_run_dir,
        smoothed_run_dir,
    ) = common.load_shared_context(base_config, run_dir, base_hash)
    n_spots = int(neighbors.shape[0])
    n_genes = len(genes)
    block_size = int(scheme2_config["gene_block_size"])
    checkpoint_every = int(scheme2_config["checkpoint_every_blocks"])
    target_sum = float(scheme2_config["normalize_total_target_sum"])
    if block_size <= 0 or checkpoint_every <= 0:
        raise ValueError("gene_block_size和checkpoint_every_blocks必须为正整数。")

    raw_totals = load_scheme1_row_totals(
        run_dir, base_hash, n_spots, n_genes
    )
    checkpoint_path = run_dir / CHECKPOINT_NAME
    completed_end, prior_elapsed, moments, gene_pcc = load_checkpoint(
        checkpoint_path,
        scheme2_hash,
        base_hash,
        n_spots,
        n_genes,
        block_size,
        target_sum,
    )
    if completed_end == n_genes:
        common.base.LOGGER.info("方案2 PCC断点已完成，直接重新生成报告。")
        cell_pcc = {
            name: common.finalize_row_pcc(moments, name, n_genes)
            for name in ("unsmoothed", "smoothed")
        }
        return write_reports(
            run_dir,
            baseline_run_dir,
            smoothed_run_dir,
            genes,
            cell_pcc,
            gene_pcc,
            scheme2_config,
            scheme2_hash,
            base_hash,
            prior_elapsed,
        )

    transfer = common.base.make_uniform_transfer_matrix(
        neighbors, spatial_mean.shape[0]
    )
    alpha = np.float32(base_config["smoothing_alpha"])
    common.base.LOGGER.info("加载source与truth raw-count矩阵，开始方案2 PCC计算。")
    source_full = common.base.load_h5ad_csr(Path(base_config["source_rna_h5ad"]))
    source_shared = (
        source_full[:, source_indices].tocsc().astype(np.float32, copy=False)
    )
    del source_full
    target_full = common.base.load_h5ad_csr(Path(base_config["target_rna_h5ad"]))
    target_shared = (
        target_full[:, target_indices].tocsc().astype(np.float32, copy=False)
    )
    del target_full
    source_shared.sort_indices()
    target_shared.sort_indices()

    started = time.time()
    total_blocks = math.ceil(n_genes / block_size)
    for block_index in range(completed_end // block_size, total_blocks):
        start = block_index * block_size
        end = min(n_genes, start + block_size)
        block_started = time.time()
        source_block = source_shared[:, start:end].tocsr()
        true_raw = np.asarray(
            target_shared[:, start:end].toarray(), dtype=np.float32
        )
        unsmoothed_raw = common.base.aggregate_block(transfer, source_block)
        smoothed_source = (
            source_block.multiply(np.float32(1.0) - alpha)
            + (spatial_mean @ source_block).multiply(alpha)
        ).tocsr()
        smoothed_source.sum_duplicates()
        smoothed_source.eliminate_zeros()
        smoothed_raw = common.base.aggregate_block(transfer, smoothed_source)

        true = normalize_total_log1p(true_raw, raw_totals["true"], target_sum)
        unsmoothed = normalize_total_log1p(
            unsmoothed_raw, raw_totals["unsmoothed"], target_sum
        )
        smoothed = normalize_total_log1p(
            smoothed_raw, raw_totals["smoothed"], target_sum
        )
        del true_raw, unsmoothed_raw, smoothed_raw

        gene_pcc["unsmoothed"][start:end] = common.pearson_values(
            unsmoothed, true, axis=0
        )
        gene_pcc["smoothed"][start:end] = common.pearson_values(
            smoothed, true, axis=0
        )
        common.update_true_moments(moments, true)
        common.update_prediction_moments(
            moments, "unsmoothed", unsmoothed, true
        )
        common.update_prediction_moments(moments, "smoothed", smoothed, true)
        del source_block, smoothed_source, true, unsmoothed, smoothed

        elapsed = prior_elapsed + time.time() - started
        if (block_index + 1) % checkpoint_every == 0 or end == n_genes:
            save_checkpoint(
                checkpoint_path,
                end,
                elapsed,
                moments,
                gene_pcc,
                complete=end == n_genes,
            )
        common.base.LOGGER.info(
            "Scheme2 PCC block %d/%d genes=%d:%d block=%.1fs elapsed=%.1fs %s",
            block_index + 1,
            total_blocks,
            start,
            end,
            time.time() - block_started,
            elapsed,
            common.base.memory_gib(),
        )

    cell_pcc = {
        name: common.finalize_row_pcc(moments, name, n_genes)
        for name in ("unsmoothed", "smoothed")
    }
    summary = write_reports(
        run_dir,
        baseline_run_dir,
        smoothed_run_dir,
        genes,
        cell_pcc,
        gene_pcc,
        scheme2_config,
        scheme2_hash,
        base_hash,
        prior_elapsed + time.time() - started,
    )
    common.base.LOGGER.info(
        "方案2 PCC计算完成：%s", json.dumps(summary["variants"], ensure_ascii=False)
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
