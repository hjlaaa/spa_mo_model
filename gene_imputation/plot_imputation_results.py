#!/usr/bin/env python3
"""Create reproducible formal result figures for an unsmoothed imputation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_gene_columns(handle: h5py.File, genes: list[str]) -> np.ndarray:
    available = handle["var_names"].asstr()[:]
    lookup = {gene: index for index, gene in enumerate(available)}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise KeyError(f"Requested genes absent from imputation output: {missing}")
    indices = np.asarray([lookup[gene] for gene in genes], dtype=np.int64)
    order = np.argsort(indices)
    sorted_values = np.asarray(handle["X"][:, indices[order]], dtype=np.float32)
    return sorted_values[:, np.argsort(order)]


def expression_image(
    coordinates: np.ndarray, values: np.ndarray, transform: str = "log1p"
) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    if coordinates.dtype.kind not in "iu" or not np.array_equal(coordinates, np.rint(coordinates)):
        raise ValueError("Exact image rendering requires integer section1 coordinates.")
    minima = coordinates.min(axis=0)
    shifted = coordinates - minima
    width, height = (shifted.max(axis=0) + 1).tolist()
    image = np.full((height, width), np.nan, dtype=np.float32)
    if transform == "log1p":
        rendered = np.log1p(values).astype(np.float32, copy=False)
    elif transform == "none":
        rendered = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported expression transform: {transform}")
    image[shifted[:, 1], shifted[:, 0]] = rendered
    return image


def robust_cap(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return 1.0
    cap = float(np.quantile(finite, quantile))
    return cap if cap > 0 else 1.0


def save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi if extension.lower() == "png" else None)
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_marker_maps(
    h5_path: Path,
    genes: list[str],
    output_dir: Path,
    config: dict[str, Any],
    stem: str = "01_spatial_marker_maps",
    title: str = "Unsmoothed KNN-imputed expression in section1",
) -> list[str]:
    with h5py.File(h5_path, "r") as handle:
        coordinates = np.asarray(handle["obsm"]["spatial"][:])
        values = read_gene_columns(handle, genes)
    columns = 3
    rows = math.ceil(len(genes) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.2, 4.0 * rows), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    for index, gene in enumerate(genes):
        axis = axes.flat[index]
        image = expression_image(coordinates, values[:, index], config["expression_transform"])
        cap = robust_cap(image, float(config["color_cap_quantile"]))
        artist = axis.imshow(image, cmap=cmap, vmin=0, vmax=cap, origin="upper", interpolation="nearest")
        axis.set_title(gene, fontweight="bold")
        axis.set_axis_off()
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.035, pad=0.02)
        colorbar.set_label("log1p predicted count")
    for index in range(len(genes), rows * columns):
        axes.flat[index].set_axis_off()
    figure.suptitle(title, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        f"Color capped independently at q={config['color_cap_quantile']}; white denotes no section1 spot.",
        ha="center",
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    return save_figure(figure, output_dir, stem, config["formats"], int(config["dpi"]))


def plot_gene_qc(
    run_dir: Path, output_dir: Path, config: dict[str, Any]
) -> list[str]:
    table = pd.read_csv(run_dir / "gene_qc.csv.gz")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    mean = table["predicted_mean"].to_numpy()
    nonzero = table["predicted_nonzero_fraction"].to_numpy()
    source_nnz = table["source_nnz"].to_numpy()

    axes[0, 0].hist(np.log10(mean + 1e-7), bins=70, color="#31688e", alpha=0.9)
    axes[0, 0].set_xlabel("log10(predicted mean + 1e-7)")
    axes[0, 0].set_ylabel("Genes")
    axes[0, 0].set_title("Predicted mean distribution")

    axes[0, 1].hist(nonzero, bins=np.linspace(0, 1, 51), color="#35b779", alpha=0.9)
    axes[0, 1].axvline(np.median(nonzero), color="#d62728", linestyle="--", label=f"median={np.median(nonzero):.3f}")
    axes[0, 1].set_xlabel("Fraction of section1 spots predicted nonzero")
    axes[0, 1].set_ylabel("Genes")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title("Spatial nonzero coverage")

    artist = axes[1, 0].hexbin(
        np.log10(source_nnz + 1),
        np.log10(mean + 1e-7),
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="magma",
    )
    axes[1, 0].set_xlabel("log10(source nonzero spots + 1)")
    axes[1, 0].set_ylabel("log10(predicted mean + 1e-7)")
    axes[1, 0].set_title("Source support versus prediction")
    figure.colorbar(artist, ax=axes[1, 0], label="log10 gene count per hexbin")

    top = table.nlargest(20, "predicted_mean").sort_values("predicted_mean")
    axes[1, 1].barh(top["gene"], np.log1p(top["predicted_mean"]), color="#440154")
    axes[1, 1].set_xlabel("log1p predicted mean")
    axes[1, 1].set_title("Top predicted-mean genes")
    figure.suptitle("Gene-level quality control: unsmoothed imputation", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(figure, output_dir, "02_gene_qc_overview", config["formats"], int(config["dpi"]))


def plot_knn_qc(
    run_dir: Path, output_dir: Path, config: dict[str, Any]
) -> list[str]:
    manifest = json.loads((run_dir / "neighbor_manifest.json").read_text(encoding="utf-8"))
    distances = np.load(manifest["distances_path"], mmap_mode="r")
    sample_size = min(int(config["knn_sample_spots"]), len(distances))
    rng = np.random.default_rng(int(config["random_seed"]))
    rows = np.sort(rng.choice(len(distances), sample_size, replace=False))
    sample = np.asarray(distances[rows])
    reuse_path = run_dir / "source_neighbor_reuse.npy"
    if reuse_path.is_file():
        reuse = np.load(reuse_path)
    else:
        neighbors = np.load(manifest["neighbors_path"], mmap_mode="r")
        reuse = np.bincount(neighbors.reshape(-1), minlength=int(manifest["source_count"]))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(sample[:, 0], bins=70, density=True, alpha=0.75, color="#31688e", label="top-1")
    axes[0, 0].hist(sample.reshape(-1), bins=70, density=True, alpha=0.55, color="#f8961e", label="all K")
    axes[0, 0].set_xlabel("Annoy Euclidean distance")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Neighbor-distance distributions")

    ranks = np.arange(1, sample.shape[1] + 1)
    median = np.median(sample, axis=0)
    lower, upper = np.quantile(sample, (0.1, 0.9), axis=0)
    axes[0, 1].plot(ranks, median, color="#31688e", linewidth=2, label="median")
    axes[0, 1].fill_between(ranks, lower, upper, color="#31688e", alpha=0.2, label="10–90%")
    axes[0, 1].set_xlabel("Neighbor rank")
    axes[0, 1].set_ylabel("Distance")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title("Distance by KNN rank")

    used = reuse[reuse > 0]
    axes[1, 0].hist(np.log10(used), bins=70, color="#35b779", alpha=0.9)
    axes[1, 0].axvline(np.log10(np.median(used)), color="#d62728", linestyle="--", label=f"median reuse={np.median(used):.0f}")
    axes[1, 0].set_xlabel("log10 reuse count among used section2 spots")
    axes[1, 0].set_ylabel("Source spots")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title("Source-neighbor reuse")

    sorted_reuse = np.sort(reuse)[::-1]
    cumulative = np.cumsum(sorted_reuse, dtype=np.float64) / sorted_reuse.sum()
    source_fraction = np.arange(1, len(sorted_reuse) + 1) / len(sorted_reuse)
    axes[1, 1].plot(source_fraction, cumulative, color="#440154", linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    axes[1, 1].set_xlabel("Fraction of section2 source spots (highest reuse first)")
    axes[1, 1].set_ylabel("Cumulative fraction of neighbor assignments")
    axes[1, 1].set_title(f"Coverage: {np.count_nonzero(reuse):,}/{len(reuse):,} source spots used")
    figure.suptitle("Embedding-KNN quality control", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(figure, output_dir, "03_knn_qc_overview", config["formats"], int(config["dpi"]))


def main() -> None:
    args = parse_args()
    config, config_hash = load_config(args.config.resolve())
    setup_style()
    run_dir = Path(config["run_dir"])
    output_dir = Path(config["output_dir"])
    manifest = json.loads((run_dir / "imputation_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Cannot plot an incomplete imputation result.")
    h5_path = Path(manifest["output_path"])
    outputs: dict[str, list[str]] = {}
    outputs["spatial_marker_maps"] = plot_marker_maps(
        h5_path, list(config["marker_genes"]), output_dir, config
    )
    outputs["gene_qc"] = plot_gene_qc(run_dir, output_dir, config)
    outputs["knn_qc"] = plot_knn_qc(run_dir, output_dir, config)
    figure_manifest = {
        "format": "spa_mo_model.imputation_formal_figures",
        "plot_config_hash": config_hash,
        "plot_config": config,
        "imputation_manifest": str(run_dir / "imputation_manifest.json"),
        "input_h5": str(h5_path),
        "outputs": outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    guide_path = output_dir / "FIGURE_GUIDE.md"
    guide_path.write_text(
        "# 未平滑基因插补正式图件\n\n"
        "## 01_spatial_marker_maps\n\n"
        "section1 全体 spot 的六个代表性插补基因空间图。表达值使用 `log1p`，每个基因独立截断至 99.5% 分位数；白色为没有 spot 的组织空白。该图用于检查组织区域特异性和异常散点。\n\n"
        "## 02_gene_qc_overview\n\n"
        "展示逐基因预测均值、空间非零覆盖、section2 来源支持度以及平均表达最高的基因。用于识别低支持、极稀疏或被少数高表达基因主导的结果。\n\n"
        "## 03_knn_qc_overview\n\n"
        "展示 KNN 距离、距离随邻居名次的变化、section2 参考 spot 被复用的次数以及邻居分配集中度。它评价 embedding 邻居传递本身，不代表缺失基因的真实预测准确率。\n\n"
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
