#!/usr/bin/env python3
"""Add v3-style per-section plots to human-embryo analysis results.

This is intentionally a plotting-only completion pass: it reuses the existing
cell-type annotations and saved clustering labels without refitting a model or
changing any metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_RUN_DIRS = [
    Path("/home/hujinlan/spa_mo_model/result_v5/human_embryo_harmony"),
    Path("/home/hujinlan/spa_mo_model/result_v6/human_embryo_harmony"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=DEFAULT_RUN_DIRS,
        help="Run directories whose analysis outputs should be completed.",
    )
    parser.add_argument("--maximum-per-section", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate per-section PNGs that already exist.",
    )
    return parser.parse_args()


def sample_indices(n_obs: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= n_obs:
        return np.arange(n_obs, dtype=np.int64)
    return np.sort(
        np.random.default_rng(seed).choice(n_obs, size=maximum, replace=False)
    )


def palette(count: int) -> list[Any]:
    colors: list[Any] = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(plt.get_cmap(name).colors)
    if count > len(colors):
        colors.extend(
            plt.get_cmap("hsv")(
                np.linspace(0, 1, count - len(colors), endpoint=False)
            )
        )
    return colors[:count]


def scatter_axis(
    axis: Any,
    frame: pd.DataFrame,
    labels: np.ndarray,
    color_lookup: dict[Any, Any],
    maximum: int,
    seed: int,
) -> None:
    chosen = sample_indices(len(frame), maximum, seed)
    values = labels[chosen]
    axis.scatter(
        frame["x"].to_numpy()[chosen],
        frame["y"].to_numpy()[chosen],
        c=[color_lookup[value] for value in values],
        s=1,
        linewidths=0,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.axis("off")


def add_legend(
    figure: Any,
    values: list[Any],
    color_lookup: dict[Any, Any],
    title: str,
    ncol: int = 1,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            color=color_lookup[value],
            label=str(value),
        )
        for value in values
    ]
    figure.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        title=title,
        fontsize=7,
        ncol=ncol,
    )


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=220, bbox_inches="tight")
    temporary.replace(path)
    plt.close(figure)


def load_metadata(
    run_summary: dict[str, Any], sections: list[str]
) -> dict[str, pd.DataFrame]:
    annotation_files = run_summary["preprocess_manifest"]["annotation_files"]
    metadata: dict[str, pd.DataFrame] = {}
    for section in sections:
        path = Path(annotation_files[section])
        frame = pd.read_csv(path, usecols=["celltype", "x", "y"])
        expected = int(run_summary["final_embedding_shapes"][section][0])
        if len(frame) != expected:
            raise ValueError(
                f"{section}: annotation rows={len(frame):,}, expected={expected:,}."
            )
        metadata[section] = frame
    return metadata


def load_celltype_colors(
    analysis_dir: Path, metadata: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    color_map_path = analysis_dir / "celltype_reference" / "celltype_color_map.csv"
    color_map = pd.read_csv(color_map_path)
    required = {"celltype", "color"}
    if not required.issubset(color_map.columns):
        raise ValueError(f"{color_map_path} lacks columns {sorted(required)}.")
    lookup = dict(
        zip(color_map["celltype"].astype(str), color_map["color"].astype(str))
    )
    observed = {
        value
        for frame in metadata.values()
        for value in frame["celltype"].astype(str)
        if value != "nan"
    }
    missing = sorted(observed.difference(lookup))
    if missing:
        raise ValueError(f"{color_map_path} lacks cell types: {missing}")
    return lookup


def plot_celltypes(
    analysis_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    maximum: int,
    seed: int,
    overwrite: bool,
) -> int:
    output_dir = analysis_dir / "celltype_reference" / "per_section"
    color_lookup = load_celltype_colors(analysis_dir, metadata)
    written = 0
    for index, section in enumerate(sections):
        output_path = output_dir / f"{section}.png"
        if output_path.exists() and not overwrite:
            continue
        raw = metadata[section]
        labels = raw["celltype"].astype(str).to_numpy()
        valid = labels != "nan"
        frame = raw.loc[valid].reset_index(drop=True)
        labels = labels[valid]
        actual = sorted(np.unique(labels).tolist())
        figure, axis = plt.subplots(figsize=(10, 7))
        scatter_axis(
            axis, frame, labels, color_lookup, maximum, seed + index
        )
        axis.set_title(f"celltype reference: {section}")
        add_legend(figure, actual, color_lookup, "celltype", ncol=2)
        figure.subplots_adjust(right=0.7)
        save_figure(figure, output_path)
        written += 1
    return written


def cluster_directories(clustering_dir: Path) -> list[tuple[str, int, Path]]:
    result: list[tuple[str, int, Path]] = []
    for mode in ("joint", "independent"):
        for directory in clustering_dir.glob(f"{mode}_k*"):
            try:
                k = int(directory.name.rsplit("_k", 1)[1])
            except ValueError:
                continue
            result.append((mode, k, directory))
    return sorted(result, key=lambda value: (value[0], value[1]))


def plot_clusters(
    analysis_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    maximum: int,
    seed: int,
    overwrite: bool,
) -> int:
    directories = cluster_directories(analysis_dir / "clustering")
    if not directories:
        raise FileNotFoundError(f"No clustering directories under {analysis_dir}.")
    max_k = max(k for _, k, _ in directories)
    color_lookup = {value: color for value, color in enumerate(palette(max_k))}
    written = 0
    for mode, k, directory in directories:
        output_dir = directory / "per_section"
        for index, section in enumerate(sections):
            output_path = output_dir / f"{section}.png"
            if output_path.exists() and not overwrite:
                continue
            label_path = directory / f"labels_{section}.npy"
            labels = np.load(label_path, mmap_mode="r")
            if labels.shape != (len(metadata[section]),):
                raise ValueError(
                    f"{label_path}: shape={labels.shape}, expected "
                    f"({len(metadata[section])},)."
                )
            actual = sorted(int(value) for value in np.unique(labels))
            if actual and (actual[0] < 0 or actual[-1] >= max_k):
                raise ValueError(f"{label_path}: labels outside [0, {max_k - 1}].")
            figure, axis = plt.subplots(figsize=(8.5, 7))
            scatter_axis(
                axis,
                metadata[section],
                labels,
                color_lookup,
                maximum,
                seed + index,
            )
            title = f"{mode} standardized embedding, k={k}"
            axis.set_title(f"{title}\n{section}")
            add_legend(
                figure,
                actual,
                color_lookup,
                "cluster",
                ncol=2 if len(actual) > 18 else 1,
            )
            figure.subplots_adjust(right=0.78)
            save_figure(figure, output_path)
            written += 1
    return written


def complete_run(
    run_dir: Path, maximum: int, seed: int, overwrite: bool
) -> tuple[int, int]:
    run_dir = run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    run_summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    sections = list(run_summary["section_order"])
    metadata = load_metadata(run_summary, sections)
    celltype_count = plot_celltypes(
        analysis_dir, sections, metadata, maximum, seed, overwrite
    )
    cluster_count = plot_clusters(
        analysis_dir, sections, metadata, maximum, seed, overwrite
    )
    return celltype_count, cluster_count


def main() -> None:
    args = parse_args()
    if args.maximum_per_section <= 0:
        raise ValueError("--maximum-per-section must be positive.")
    for run_dir in args.run_dirs:
        print(f"[per-section] completing {run_dir}", flush=True)
        celltype_count, cluster_count = complete_run(
            run_dir,
            args.maximum_per_section,
            args.seed,
            args.overwrite,
        )
        print(
            f"[per-section] wrote {celltype_count} celltype and "
            f"{cluster_count} clustering PNGs",
            flush=True,
        )


if __name__ == "__main__":
    main()
