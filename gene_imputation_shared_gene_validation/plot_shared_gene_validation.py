#!/usr/bin/env python3
"""为共享基因伪缺失验证生成以方案1/方案2 PCC为核心的正式图件。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_PLOT_PATH = Path(__file__).resolve().parents[1] / "gene_imputation" / "plot_imputation_results.py"
SPEC = importlib.util.spec_from_file_location("baseline_plotting", BASE_PLOT_PATH)
base_plot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(base_plot)


SCHEMES = {
    "scheme1": {
        "label": "方案1：原表达尺度",
        "plot_label": "Scheme 1: raw expression scale",
        "cell_file": "scheme1_raw_pcc_per_cell.npz",
        "gene_file": "scheme1_raw_pcc_per_gene.csv.gz",
        "summary_file": "scheme1_raw_pcc_summary.json",
    },
    "scheme2": {
        "label": "方案2：normalize_total → log1p",
        "plot_label": "Scheme 2: normalize_total → log1p",
        "cell_file": "scheme2_normlog_pcc_per_cell.npz",
        "gene_file": "scheme2_normlog_pcc_per_gene.csv.gz",
        "summary_file": "scheme2_normlog_pcc_summary.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    return float(np.mean(values[valid])) if valid.any() else float("nan")


def paired_valid(unsmoothed: np.ndarray, smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(unsmoothed, dtype=np.float64)
    right = np.asarray(smoothed, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"配对PCC形状不一致：{left.shape} vs {right.shape}")
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def load_pcc_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    old_metrics = pd.read_csv(run_dir / "shared_gene_accuracy_metrics.csv.gz")
    detection = old_metrics[["gene", "target_nonzero_fraction"]].copy()
    if detection["gene"].duplicated().any():
        raise ValueError("共享基因检测率表存在重复gene。")

    results: dict[str, dict[str, Any]] = {}
    reference_genes: np.ndarray | None = None
    for scheme, settings in SCHEMES.items():
        cell_path = run_dir / settings["cell_file"]
        gene_path = run_dir / settings["gene_file"]
        summary_path = run_dir / settings["summary_file"]
        with np.load(cell_path) as payload:
            cells = {
                "unsmoothed": np.asarray(payload["unsmoothed_pcc"], dtype=np.float64),
                "smoothed": np.asarray(payload["smoothed_pcc"], dtype=np.float64),
            }
        genes = pd.read_csv(gene_path)
        required = {"gene", "unsmoothed_pcc", "smoothed_pcc"}
        if not required.issubset(genes.columns):
            raise ValueError(f"{gene_path}缺少字段：{sorted(required - set(genes.columns))}")
        genes = genes.merge(detection, on="gene", how="left", validate="one_to_one")
        if genes["target_nonzero_fraction"].isna().any():
            raise ValueError(f"{scheme}逐gene PCC缺少section1检测率。")
        current_genes = genes["gene"].astype(str).to_numpy()
        if reference_genes is None:
            reference_genes = current_genes
        elif not np.array_equal(reference_genes, current_genes):
            raise ValueError("方案1和方案2的逐gene PCC顺序不一致。")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for variant in ("unsmoothed", "smoothed"):
            cell_mean = finite_mean(cells[variant])
            gene_mean = finite_mean(genes[f"{variant}_pcc"].to_numpy())
            expected = summary["variants"][variant]
            if not np.isclose(cell_mean, float(expected["pcc_cell"]), atol=1e-12):
                raise ValueError(f"{scheme} {variant} pcc_cell与汇总文件不一致。")
            if not np.isclose(gene_mean, float(expected["pcc_gene"]), atol=1e-12):
                raise ValueError(f"{scheme} {variant} pcc_gene与汇总文件不一致。")

        results[scheme] = {
            "label": settings["label"],
            "plot_label": settings["plot_label"],
            "cells": cells,
            "genes": genes,
            "summary": summary,
            "paths": {"cell": str(cell_path), "gene": str(gene_path), "summary": str(summary_path)},
        }
    return results


def draw_pcc_distribution(
    axis: plt.Axes,
    unsmoothed: np.ndarray,
    smoothed: np.ndarray,
    xlabel: str,
    title: str,
) -> None:
    baseline, smooth = paired_valid(unsmoothed, smoothed)
    combined = np.concatenate((baseline, smooth))
    lower, upper = np.quantile(combined, (0.005, 0.995))
    if upper <= lower:
        upper = lower + 1.0
    bins = np.linspace(float(lower), float(upper), 70)
    axis.hist(baseline, bins=bins, density=True, alpha=0.55, color="#31688e", label="Unsmoothed")
    axis.hist(smooth, bins=bins, density=True, alpha=0.55, color="#f8961e", label="Smoothed")
    baseline_mean = float(np.mean(baseline))
    smooth_mean = float(np.mean(smooth))
    improved = float(np.mean(smooth > baseline))
    axis.axvline(baseline_mean, color="#31688e", linestyle="--", linewidth=1.5)
    axis.axvline(smooth_mean, color="#f8961e", linestyle="--", linewidth=1.5)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(f"{title}\nMean {baseline_mean:.4f} → {smooth_mean:.4f}; improved {improved:.1%}")
    axis.legend(frameon=False)


def plot_pcc_distributions(
    results: dict[str, dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for row, scheme in enumerate(("scheme1", "scheme2")):
        payload = results[scheme]
        draw_pcc_distribution(
            axes[row, 0], payload["cells"]["unsmoothed"], payload["cells"]["smoothed"],
            "Per-cell Pearson r", f"{payload['plot_label']}: pcc_cell",
        )
        draw_pcc_distribution(
            axes[row, 1], payload["genes"]["unsmoothed_pcc"].to_numpy(),
            payload["genes"]["smoothed_pcc"].to_numpy(), "Per-gene Pearson r",
            f"{payload['plot_label']}: pcc_gene",
        )
    figure.suptitle("Shared-gene pseudo-missing validation: PCC distributions (4,828 genes)", fontweight="bold")
    figure.text(
        0.5, 0.008,
        "Histograms show the 0.5th–99.5th percentile range; dashed lines and reported values are arithmetic means over finite Pearson r values.",
        ha="center", color="#444444",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    return base_plot.save_figure(
        figure, output_dir, "01_pcc_metric_distributions", config["plot_formats"], int(config["plot_dpi"])
    )


def paired_limits(left: np.ndarray, right: np.ndarray, robust: bool) -> tuple[float, float]:
    values = np.concatenate((left, right))
    if robust:
        lower, upper = np.quantile(values, (0.001, 0.999))
    else:
        lower, upper = float(np.min(values)), float(np.max(values))
    padding = max(float(upper - lower) * 0.035, 1e-3)
    return float(lower - padding), float(upper + padding)


def draw_paired_cell_pcc(
    axis: plt.Axes, unsmoothed: np.ndarray, smoothed: np.ndarray, title: str
) -> None:
    baseline, smooth = paired_valid(unsmoothed, smoothed)
    lower, upper = paired_limits(baseline, smooth, robust=True)
    artist = axis.hexbin(
        baseline, smooth, gridsize=75, mincnt=1, bins="log",
        extent=(lower, upper, lower, upper), cmap="magma",
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Unsmoothed per-cell Pearson r")
    axis.set_ylabel("Smoothed per-cell Pearson r")
    axis.set_title(
        f"{title}: pcc_cell\nMean Δ={np.mean(smooth - baseline):+.4f}; improved={np.mean(smooth > baseline):.1%}"
    )
    plt.colorbar(artist, ax=axis, label="log10(cells per hexbin)")


def draw_paired_gene_pcc(axis: plt.Axes, genes: pd.DataFrame, title: str) -> None:
    baseline = genes["unsmoothed_pcc"].to_numpy(dtype=np.float64)
    smooth = genes["smoothed_pcc"].to_numpy(dtype=np.float64)
    detection = genes["target_nonzero_fraction"].to_numpy(dtype=np.float64)
    valid = np.isfinite(baseline) & np.isfinite(smooth)
    baseline, smooth, detection = baseline[valid], smooth[valid], detection[valid]
    lower, upper = paired_limits(baseline, smooth, robust=False)
    artist = axis.scatter(
        baseline, smooth, c=np.log10(detection + 1e-6), s=7, alpha=0.55,
        cmap="viridis", linewidths=0, rasterized=True,
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Unsmoothed per-gene Pearson r")
    axis.set_ylabel("Smoothed per-gene Pearson r")
    axis.set_title(
        f"{title}: pcc_gene\nMean Δ={np.mean(smooth - baseline):+.4f}; improved={np.mean(smooth > baseline):.1%}"
    )
    plt.colorbar(artist, ax=axis, label="log10(section1 observed detection fraction + 1e-6)")


def plot_paired_pcc(
    results: dict[str, dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row, scheme in enumerate(("scheme1", "scheme2")):
        payload = results[scheme]
        draw_paired_cell_pcc(
            axes[row, 0], payload["cells"]["unsmoothed"], payload["cells"]["smoothed"], payload["plot_label"]
        )
        draw_paired_gene_pcc(axes[row, 1], payload["genes"], payload["plot_label"])
    figure.suptitle("Paired PCC comparison: smoothed versus unsmoothed", fontweight="bold")
    figure.text(
        0.5, 0.008, "Points above the diagonal favor smoothing; per-cell panels show the 0.1st–99.9th percentile range.",
        ha="center", color="#444444",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    return base_plot.save_figure(
        figure, output_dir, "02_paired_pcc", config["plot_formats"], int(config["plot_dpi"])
    )


def assign_detection_strata(genes: pd.DataFrame) -> pd.Series:
    positive = genes["target_nonzero_fraction"] > 0
    strata = pd.Series(pd.NA, index=genes.index, dtype="object")
    strata.loc[positive] = pd.qcut(
        genes.loc[positive, "target_nonzero_fraction"], q=4,
        labels=("Q1 lowest", "Q2", "Q3", "Q4 highest"), duplicates="drop",
    ).astype("object")
    return strata


def stratum_summary(genes: pd.DataFrame) -> pd.DataFrame:
    frame = genes.copy()
    frame["检测率分层"] = assign_detection_strata(frame)
    rows: list[dict[str, Any]] = []
    for stratum in ("Q1 lowest", "Q2", "Q3", "Q4 highest"):
        selected = frame.loc[frame["检测率分层"] == stratum]
        valid = np.isfinite(selected["unsmoothed_pcc"]) & np.isfinite(selected["smoothed_pcc"])
        selected = selected.loc[valid]
        delta = selected["smoothed_pcc"] - selected["unsmoothed_pcc"]
        rows.append(
            {
                "stratum": stratum,
                "genes": len(selected),
                "unsmoothed_mean": selected["unsmoothed_pcc"].mean(),
                "smoothed_mean": selected["smoothed_pcc"].mean(),
                "mean_delta": delta.mean(),
                "fraction_improved": np.mean(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def draw_stratum_bars(axis: plt.Axes, summary: pd.DataFrame, title: str) -> None:
    x = np.arange(len(summary))
    width = 0.36
    axis.bar(x - width / 2, summary["unsmoothed_mean"], width, color="#31688e", label="Unsmoothed")
    axis.bar(x + width / 2, summary["smoothed_mean"], width, color="#f8961e", label="Smoothed")
    axis.set_xticks(x, summary["stratum"])
    axis.set_ylabel("Mean per-gene PCC within stratum")
    axis.set_title(title)
    axis.legend(frameon=False)


def draw_delta_hexbin(axis: plt.Axes, genes: pd.DataFrame, title: str) -> None:
    detection = genes["target_nonzero_fraction"].to_numpy(dtype=np.float64)
    delta = genes["smoothed_pcc"].to_numpy(dtype=np.float64) - genes["unsmoothed_pcc"].to_numpy(dtype=np.float64)
    valid = np.isfinite(delta) & (detection > 0)
    artist = axis.hexbin(
        np.log10(detection[valid]), delta[valid], gridsize=60, mincnt=1, bins="log", cmap="magma"
    )
    axis.axhline(0, linestyle="--", color="#777777", linewidth=1)
    axis.set_xlabel("log10(section1 observed detection fraction)")
    axis.set_ylabel("Per-gene PCC Δ (smoothed − unsmoothed)")
    axis.set_title(title)
    plt.colorbar(artist, ax=axis, label="log10(genes per hexbin)")


def plot_pcc_detection_strata(
    results: dict[str, dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> tuple[list[str], pd.DataFrame]:
    summaries = {scheme: stratum_summary(results[scheme]["genes"]) for scheme in ("scheme1", "scheme2")}
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    draw_stratum_bars(axes[0, 0], summaries["scheme1"], "Scheme 1 by observed detection stratum")
    draw_stratum_bars(axes[0, 1], summaries["scheme2"], "Scheme 2 by observed detection stratum")

    x = np.arange(4)
    axes[0, 2].plot(
        x, summaries["scheme1"]["fraction_improved"], marker="o", linewidth=2,
        color="#31688e", label="Scheme 1",
    )
    axes[0, 2].plot(
        x, summaries["scheme2"]["fraction_improved"], marker="o", linewidth=2,
        color="#f8961e", label="Scheme 2",
    )
    axes[0, 2].axhline(0.5, linestyle="--", color="#777777", linewidth=1)
    axes[0, 2].set_xticks(x, summaries["scheme1"]["stratum"])
    axes[0, 2].set_ylim(0, 1.02)
    axes[0, 2].set_ylabel("Fraction of genes improved by smoothing")
    axes[0, 2].set_title("Consistency of improvement across strata")
    axes[0, 2].legend(frameon=False)

    draw_delta_hexbin(axes[1, 0], results["scheme1"]["genes"], "Scheme 1: PCC gain versus detection")
    draw_delta_hexbin(axes[1, 1], results["scheme2"]["genes"], "Scheme 2: PCC gain versus detection")

    scheme1_genes = results["scheme1"]["genes"].copy()
    scheme2_genes = results["scheme2"]["genes"].copy()
    rank_table = scheme1_genes[["gene"]].copy()
    rank_table["方案1Δ"] = scheme1_genes["smoothed_pcc"] - scheme1_genes["unsmoothed_pcc"]
    rank_table["方案2Δ"] = scheme2_genes["smoothed_pcc"] - scheme2_genes["unsmoothed_pcc"]
    rank_table["两方案平均Δ"] = rank_table[["方案1Δ", "方案2Δ"]].mean(axis=1)
    ranked = rank_table.dropna().sort_values("两方案平均Δ")
    selected = pd.concat((ranked.head(8), ranked.tail(8))).drop_duplicates("gene")
    y = np.arange(len(selected))
    height = 0.38
    axes[1, 2].barh(y - height / 2, selected["方案1Δ"], height, color="#31688e", label="Scheme 1")
    axes[1, 2].barh(y + height / 2, selected["方案2Δ"], height, color="#f8961e", label="Scheme 2")
    axes[1, 2].axvline(0, color="#777777", linewidth=1)
    axes[1, 2].set_yticks(y, selected["gene"])
    axes[1, 2].set_xlabel("Per-gene PCC Δ")
    axes[1, 2].set_title("Largest gains/losses ranked by mean Δ across schemes")
    axes[1, 2].legend(frameon=False)

    figure.suptitle("PCC stratified by section1 observed gene detection rate", fontweight="bold")
    figure.text(
        0.5, 0.008, "Q1 genes are extremely sparse and their PCC estimates are unstable; method comparisons should emphasize Q2–Q4.",
        ha="center", color="#444444",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    paths = base_plot.save_figure(
        figure, output_dir, "03_pcc_by_detection_stratum", config["plot_formats"], int(config["plot_dpi"])
    )

    output_rows = []
    for scheme in ("scheme1", "scheme2"):
        frame = summaries[scheme].copy()
        frame.insert(0, "方案", results[scheme]["label"])
        output_rows.append(frame)
    output_table = pd.concat(output_rows, ignore_index=True).rename(
        columns={
            "stratum": "检测率分层",
            "genes": "有效gene数",
            "unsmoothed_mean": "未平滑PCC均值",
            "smoothed_mean": "平滑PCC均值",
            "mean_delta": "PCC平均差值_平滑减未平滑",
            "fraction_improved": "平滑改善gene比例",
        }
    )
    output_table["检测率分层"] = output_table["检测率分层"].replace(
        {"Q1 lowest": "Q1 最低", "Q4 highest": "Q4 最高"}
    )
    return paths, output_table


def relative_log_intensity(values: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    transformed = np.log1p(np.asarray(values, dtype=np.float32))
    cap = float(np.quantile(transformed, quantile))
    if cap <= 0:
        return np.zeros_like(transformed)
    return np.clip(transformed / cap, 0, 1)


def marker_lookup(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    lookups: dict[str, dict[str, float]] = {}
    for scheme in ("scheme1", "scheme2"):
        table = results[scheme]["genes"].set_index("gene")
        for gene, row in table.iterrows():
            entry = lookups.setdefault(str(gene), {})
            entry[f"{scheme}_unsmoothed"] = float(row["unsmoothed_pcc"])
            entry[f"{scheme}_smoothed"] = float(row["smoothed_pcc"])
            entry["detection"] = float(row["target_nonzero_fraction"])
    return lookups


def plot_marker_spatial_validation(
    cache_path: Path,
    results: dict[str, dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    with h5py.File(cache_path, "r") as cache:
        spatial = cache["spatial"][:]
        genes = cache["marker_predictions"]["genes"].asstr()[:]
        observed = cache["marker_predictions"]["observed"][:]
        baseline = cache["marker_predictions"]["baseline"][:]
        smoothed = cache["marker_predictions"]["smoothed"][:]
    if not (np.isfinite(observed).all() and np.isfinite(baseline).all() and np.isfinite(smoothed).all()):
        raise ValueError("Marker预测缓存包含未完成值。")
    scores = marker_lookup(results)
    missing = [gene for gene in genes if gene not in scores]
    if missing:
        raise ValueError(f"Marker缺少逐gene PCC：{missing}")

    figure, axes = plt.subplots(len(genes), 3, figsize=(13.8, 3.7 * len(genes)), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    panels = ((observed, "Observed section1"), (baseline, "Unsmoothed prediction"), (smoothed, "Smoothed prediction"))
    for row, gene in enumerate(genes):
        gene_scores = scores[gene]
        subtitles = (
            f"Detection fraction={gene_scores['detection']:.3f}",
            "Scheme 1 r={:.3f}; Scheme 2 r={:.3f}".format(
                gene_scores["scheme1_unsmoothed"], gene_scores["scheme2_unsmoothed"]
            ),
            "Scheme 1 r={:.3f}; Scheme 2 r={:.3f}".format(
                gene_scores["scheme1_smoothed"], gene_scores["scheme2_smoothed"]
            ),
        )
        for column, ((values, label), subtitle) in enumerate(zip(panels, subtitles)):
            relative = relative_log_intensity(values[:, row])
            image = base_plot.expression_image(spatial, relative, transform="none")
            artist = axes[row, column].imshow(
                image, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest"
            )
            axes[row, column].set_title(f"{gene} — {label}\n{subtitle}", fontweight="bold")
            axes[row, column].set_axis_off()
            figure.colorbar(artist, ax=axes[row, column], fraction=0.035, pad=0.02)
    figure.suptitle("Shared-marker pseudo-missing spatial validation in section1", fontweight="bold")
    figure.text(
        0.5, 0.004,
        "Each panel is scaled to its own 99.5th percentile of log1p expression; compare spatial localization, not absolute color intensity. Title r values are per-gene PCCs.",
        ha="center", color="#444444",
    )
    figure.tight_layout(rect=(0, 0.012, 1, 0.975))
    return base_plot.save_figure(
        figure, output_dir, "04_shared_marker_spatial_validation", config["plot_formats"], int(config["plot_dpi"])
    )


def write_figure_guide(path: Path, strata_path: Path) -> None:
    path.write_text(
        "# 4,828个共享基因伪缺失验证：PCC正式图件\n\n"
        "## 01_pcc_metric_distributions\n\n"
        "四个面板分别展示方案1、方案2的逐cell和逐gene Pearson r分布。"
        "未平滑为蓝色，平滑为橙色；虚线和标题均使用有限相关系数的算术平均值，"
        "与`pcc_cell`、`pcc_gene`正式定义一致。直方图只显示0.5%–99.5%分位范围，"
        "但标题均值使用全部有限值。\n\n"
        "## 02_paired_pcc\n\n"
        "同一个cell或gene的未平滑PCC位于横轴，平滑PCC位于纵轴。"
        "对角线上方表示平滑后PCC提高。逐cell数据量较大，因此使用hexbin；"
        "逐gene点颜色表示section1真实检测率。\n\n"
        "## 03_pcc_by_detection_stratum\n\n"
        "按section1真实非零率将gene分为Q1–Q4，比较每层逐gene PCC的算术平均值、"
        "改善比例以及PCC变化与检测率的关系。Q1极稀疏，相关性不稳定，应重点参考Q2–Q4。"
        f"分层数值保存在`{strata_path}`。\n\n"
        "## 04_shared_marker_spatial_validation\n\n"
        "每行依次展示section1实测、未平滑预测和平滑预测。标题补充了该gene在方案1、"
        "方案2下的逐gene PCC及section1检测率。每个面板独立按自身log1p表达99.5%分位缩放，"
        "只能比较空间定位，不能比较绝对颜色数值。\n\n"
        "## 评价边界\n\n"
        "这是训练后的共享基因伪缺失诊断；共享基因参与过embedding训练，因此可能乐观。"
        "13,257个section2-only基因没有section1真值，不能由这些图直接得到真实PCC。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config, config_hash = load_config(args.config.resolve())
    base_plot.setup_style()
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
            "axes.unicode_minus": False,
        }
    )
    run_dir = Path(config["run_dir"])
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_pcc_results(run_dir)

    validation_summary = json.loads((run_dir / "validation_summary.json").read_text(encoding="utf-8"))
    cache_path = Path(validation_summary["paths"]["cache"])
    outputs: dict[str, list[str]] = {}
    outputs["pcc_distributions"] = plot_pcc_distributions(results, output_dir, config)
    outputs["paired_pcc"] = plot_paired_pcc(results, output_dir, config)
    outputs["pcc_detection_strata"], strata_table = plot_pcc_detection_strata(results, output_dir, config)
    strata_path = output_dir / "pcc_by_target_detection_stratum.csv"
    strata_table.to_csv(strata_path, index=False)
    outputs["marker_spatial_validation"] = plot_marker_spatial_validation(
        cache_path, results, output_dir, config
    )

    guide_path = output_dir / "FIGURE_GUIDE.md"
    write_figure_guide(guide_path, strata_path)
    manifest = {
        "format": "spa_mo_model.shared_gene_validation_pcc_figures",
        "format_version": 2,
        "language": "zh-CN",
        "plot_config_hash": config_hash,
        "config": config,
        "pcc_inputs": {scheme: results[scheme]["paths"] for scheme in ("scheme1", "scheme2")},
        "marker_cache": str(cache_path),
        "detection_strata_table": str(strata_path),
        "figure_guide": str(guide_path),
        "outputs": outputs,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
