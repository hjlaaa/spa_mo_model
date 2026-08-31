#!/usr/bin/env python3
"""Create formal figures for spatially smoothed imputation and its baseline contrast."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Only plot schema_version=1 is supported.")
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def output_h5(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "imputation_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete imputation result: {run_dir}")
    return Path(manifest["output_path"])


def plot_three_way_spatial_contrast(
    baseline_h5: Path,
    smoothed_h5: Path,
    genes: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    with h5py.File(baseline_h5, "r") as baseline, h5py.File(smoothed_h5, "r") as smoothed:
        coordinates = np.asarray(baseline["obsm"]["spatial"][:])
        if not np.array_equal(coordinates, smoothed["obsm"]["spatial"][:]):
            raise ValueError("Baseline and smoothed section1 spatial rows differ.")
        baseline_values = base_plot.read_gene_columns(baseline, genes)
        smoothed_values = base_plot.read_gene_columns(smoothed, genes)

    figure, axes = plt.subplots(len(genes), 3, figsize=(13.2, 3.4 * len(genes)), squeeze=False)
    expression_cmap = plt.get_cmap("viridis").copy()
    expression_cmap.set_bad("white")
    difference_cmap = plt.get_cmap("coolwarm").copy()
    difference_cmap.set_bad("white")
    quantile = float(config["color_cap_quantile"])
    for row, gene in enumerate(genes):
        baseline_image = base_plot.expression_image(
            coordinates, baseline_values[:, row], config["expression_transform"]
        )
        smoothed_image = base_plot.expression_image(
            coordinates, smoothed_values[:, row], config["expression_transform"]
        )
        combined = np.concatenate(
            (baseline_image[np.isfinite(baseline_image)], smoothed_image[np.isfinite(smoothed_image)])
        )
        cap = base_plot.robust_cap(combined, quantile)
        difference = smoothed_image - baseline_image
        difference_cap = base_plot.robust_cap(np.abs(difference), quantile)
        for column, (image, title) in enumerate(
            ((baseline_image, "Unsmoothed"), (smoothed_image, "Smoothed"))
        ):
            artist = axes[row, column].imshow(
                image,
                cmap=expression_cmap,
                vmin=0,
                vmax=cap,
                origin="upper",
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{gene} — {title}", fontweight="bold")
            axes[row, column].set_axis_off()
            figure.colorbar(artist, ax=axes[row, column], fraction=0.035, pad=0.02)
        artist = axes[row, 2].imshow(
            difference,
            cmap=difference_cmap,
            vmin=-difference_cap,
            vmax=difference_cap,
            origin="upper",
            interpolation="nearest",
        )
        axes[row, 2].set_title(f"{gene} — smoothed minus unsmoothed", fontweight="bold")
        axes[row, 2].set_axis_off()
        figure.colorbar(artist, ax=axes[row, 2], fraction=0.035, pad=0.02)
    figure.suptitle(
        "Effect of section2 reference smoothing on section1 imputation",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.004,
        "Expression panels share a per-gene log1p scale; differences are log1p(smoothed) − log1p(unsmoothed).",
        ha="center",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.012, 1, 0.975))
    return base_plot.save_figure(
        figure,
        output_dir,
        "02_baseline_smoothed_spatial_contrast",
        config["formats"],
        int(config["dpi"]),
    )


def plot_smoothing_comparison_qc(
    run_dir: Path,
    baseline_h5: Path,
    smoothed_h5: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    table = pd.read_csv(run_dir / "comparison_to_unsmoothed_gene_metrics.csv.gz")
    with h5py.File(baseline_h5, "r") as baseline, h5py.File(smoothed_h5, "r") as smoothed:
        baseline_nonzero = baseline["gene_stats"]["predicted_nonzero_fraction"][:]
        smoothed_nonzero = smoothed["gene_stats"]["predicted_nonzero_fraction"][:]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    correlation = table["sample_pearson"].dropna().to_numpy()
    axes[0, 0].hist(correlation, bins=np.linspace(0.9, 1, 61), color="#31688e", alpha=0.9)
    axes[0, 0].axvline(np.median(correlation), color="#d62728", linestyle="--", label=f"median={np.median(correlation):.4f}")
    axes[0, 0].set_xlabel("Per-gene Pearson correlation")
    axes[0, 0].set_ylabel("Genes")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Spatial-pattern preservation")

    mae = table["sample_mean_absolute_difference"].to_numpy()
    axes[0, 1].hist(np.log10(mae + 1e-8), bins=70, color="#35b779", alpha=0.9)
    axes[0, 1].set_xlabel("log10(per-gene MAE + 1e-8)")
    axes[0, 1].set_ylabel("Genes")
    axes[0, 1].set_title("Magnitude of smoothing effect")

    baseline_mean = table["baseline_full_mean"].to_numpy()
    smoothed_mean = table["smoothed_full_mean"].to_numpy()
    mean_hex = axes[0, 2].hexbin(
        np.log10(baseline_mean + 1e-7),
        np.log10(smoothed_mean + 1e-7),
        gridsize=60,
        mincnt=1,
        bins="log",
        cmap="magma",
    )
    limits = [
        min(np.log10(baseline_mean + 1e-7).min(), np.log10(smoothed_mean + 1e-7).min()),
        max(np.log10(baseline_mean + 1e-7).max(), np.log10(smoothed_mean + 1e-7).max()),
    ]
    axes[0, 2].plot(limits, limits, "--", color="white", linewidth=1)
    axes[0, 2].set_xlabel("log10 baseline full mean + 1e-7")
    axes[0, 2].set_ylabel("log10 smoothed full mean + 1e-7")
    axes[0, 2].set_title("Mean-expression preservation")
    figure.colorbar(mean_hex, ax=axes[0, 2], label="log10 genes per hexbin")

    coverage_hex = axes[1, 0].hexbin(
        baseline_nonzero,
        smoothed_nonzero,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axes[1, 0].plot([0, 1], [0, 1], "--", color="white", linewidth=1)
    axes[1, 0].set_xlabel("Baseline nonzero fraction")
    axes[1, 0].set_ylabel("Smoothed nonzero fraction")
    axes[1, 0].set_title("Change in spatial coverage")
    figure.colorbar(coverage_hex, ax=axes[1, 0], label="log10 genes per hexbin")

    delta_nonzero = smoothed_nonzero - baseline_nonzero
    axes[1, 1].hist(delta_nonzero, bins=70, color="#f8961e", alpha=0.9)
    axes[1, 1].axvline(np.median(delta_nonzero), color="#d62728", linestyle="--", label=f"median={np.median(delta_nonzero):.3f}")
    axes[1, 1].set_xlabel("Smoothed − baseline nonzero fraction")
    axes[1, 1].set_ylabel("Genes")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title("Nonzero-coverage expansion")

    top = table.nlargest(20, "sample_mean_absolute_difference").sort_values(
        "sample_mean_absolute_difference"
    )
    axes[1, 2].barh(
        top["gene"], top["sample_mean_absolute_difference"], color="#440154"
    )
    axes[1, 2].set_xlabel("Sample mean absolute difference")
    axes[1, 2].set_title("Genes most affected by smoothing")

    comparison = json.loads(
        (run_dir / "comparison_to_unsmoothed.json").read_text(encoding="utf-8")
    )
    figure.suptitle(
        "Smoothed versus unsmoothed imputation\n"
        f"global Pearson={comparison['sample_global_pearson']:.5f}, "
        f"MAE={comparison['sample_global_mae']:.5g}, "
        f"KNN tables identical={comparison['neighbor_tables_elementwise_identical']}",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return base_plot.save_figure(
        figure,
        output_dir,
        "03_smoothing_comparison_qc",
        config["formats"],
        int(config["dpi"]),
    )


def main() -> None:
    args = parse_args()
    config, config_hash = load_config(args.config.resolve())
    base_plot.setup_style()
    run_dir = Path(config["run_dir"])
    baseline_run_dir = Path(config["baseline_run_dir"])
    output_dir = Path(config["output_dir"])
    smoothed_h5 = output_h5(run_dir)
    baseline_h5 = output_h5(baseline_run_dir)
    genes = list(config["marker_genes"])
    outputs: dict[str, list[str]] = {}
    outputs["smoothed_marker_maps"] = base_plot.plot_marker_maps(
        smoothed_h5,
        genes,
        output_dir,
        config,
        stem="01_smoothed_spatial_marker_maps",
        title="Spatially smoothed reference KNN-imputed expression in section1",
    )
    outputs["three_way_spatial_contrast"] = plot_three_way_spatial_contrast(
        baseline_h5, smoothed_h5, genes, output_dir, config
    )
    outputs["smoothing_comparison_qc"] = plot_smoothing_comparison_qc(
        run_dir, baseline_h5, smoothed_h5, output_dir, config
    )
    figure_manifest = {
        "format": "spa_mo_model.smoothed_imputation_formal_figures",
        "plot_config_hash": config_hash,
        "plot_config": config,
        "smoothed_h5": str(smoothed_h5),
        "baseline_h5": str(baseline_h5),
        "outputs": outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    guide_path = output_dir / "FIGURE_GUIDE.md"
    guide_path.write_text(
        "# 参考数据局部平滑插补正式图件\n\n"
        "## 01_smoothed_spatial_marker_maps\n\n"
        "平滑参考表达后得到的六个代表性基因 section1 空间图。用于查看平滑版的组织结构和 marker 区域。\n\n"
        "## 02_baseline_smoothed_spatial_contrast\n\n"
        "每行一个基因，三列依次为未平滑、平滑以及 `log1p(平滑)-log1p(未平滑)`。前两列严格共用同一个基因色阶；差异图红色表示平滑后增加，蓝色表示减少。这是判断局部去噪和边界扩散的核心图。\n\n"
        "## 03_smoothing_comparison_qc\n\n"
        "展示所有基因的空间相关性、MAE、平均表达保持、非零覆盖变化及受平滑影响最大的基因。当前 KNN 表与未平滑对照逐元素一致，因此图中差异只来自参考数据平滑。\n\n"
        "PNG 适合快速查看，PDF 适合汇报与排版。具体配置和输入路径记录在 `figure_manifest.json`。\n",
        encoding="utf-8",
    )
    figure_manifest["figure_guide"] = str(guide_path)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
