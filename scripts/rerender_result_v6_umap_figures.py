#!/usr/bin/env python3
"""Re-render result_v6 UMAP figures from saved coordinates without refitting UMAP."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from generate_result_v6_spamosaic_umap import (
    has_meaningful_biological_labels,
    human_celltype_colors,
    plot_individual,
    plot_panel_c,
    plot_panel_e,
    sha256,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result_v6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key from umap_config.json, or 'all' (default).",
    )
    return parser.parse_args()


def color_map(analysis_dir: Path, primary_label: str) -> dict[str, str] | None:
    if primary_label != "celltype":
        return None
    path = analysis_dir / "celltype_reference/celltype_color_map.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["celltype"].astype(str), table["color"].astype(str)))


def rerender(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    table = pd.read_csv(config["coordinates_table"], low_memory=False)
    output_dir = Path(config["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    modalities = list(config["input_modalities"])
    input_coordinates = {
        modality: table[
            [f"input_{modality.lower()}_UMAP1", f"input_{modality.lower()}_UMAP2"]
        ].to_numpy(dtype=np.float32)
        for modality in modalities
    }
    integrated = table[
        ["integrated_UMAP1", "integrated_UMAP2"]
    ].to_numpy(dtype=np.float32)
    joint_labels = {
        int(column.removeprefix("joint_k")): table[column].to_numpy(dtype=np.int32)
        for column in table.columns
        if column.startswith("joint_k")
    }
    bundle = SimpleNamespace(
        name=config["dataset"],
        display_name=config["display_name"],
        analysis_dir=Path(config["analysis_dir"]),
        metadata=table,
        primary_label=config["primary_biological_label"],
        representative_k=int(config["representative_joint_k"]),
        joint_labels=joint_labels,
    )
    point_size = 2.0 if len(table) >= 50_000 else 8.0
    seed = int(config["umap"]["random_state"])
    custom_biology = color_map(bundle.analysis_dir, bundle.primary_label)
    include_biology = has_meaningful_biological_labels(bundle)

    plot_panel_c(bundle, input_coordinates, figures_dir, point_size, seed)
    plot_panel_e(
        bundle, integrated, figures_dir, point_size, seed, custom_biology
    )

    section_values = table["section_label"].to_numpy()
    biology_values = table[bundle.primary_label].to_numpy()
    for offset, (modality, coordinates) in enumerate(input_coordinates.items()):
        plot_individual(
            figures_dir / f"input_{modality.lower()}_by_section.png",
            coordinates,
            section_values,
            f"{bundle.display_name} input {modality} UMAP by section",
            "section",
            point_size,
            seed + offset,
        )
        biology_path = (
            figures_dir / f"input_{modality.lower()}_by_{bundle.primary_label}.png"
        )
        if include_biology:
            plot_individual(
                biology_path,
                coordinates,
                biology_values,
                f"{bundle.display_name} input {modality} UMAP by {bundle.primary_label}",
                "biology",
                point_size,
                seed + offset,
                custom_biology,
            )
        else:
            biology_path.unlink(missing_ok=True)

    plot_individual(
        figures_dir / "integrated_by_section.png",
        integrated,
        section_values,
        f"{bundle.display_name} integrated embedding UMAP by section",
        "section",
        point_size,
        seed,
    )
    integrated_biology_path = (
        figures_dir / f"integrated_by_{bundle.primary_label}.png"
    )
    if include_biology:
        plot_individual(
            integrated_biology_path,
            integrated,
            biology_values,
            f"{bundle.display_name} integrated embedding UMAP by {bundle.primary_label}",
            "biology",
            point_size,
            seed,
            custom_biology,
        )
    else:
        integrated_biology_path.unlink(missing_ok=True)
    cluster_dir = figures_dir / "integrated_joint_clusters"
    for k, labels in sorted(joint_labels.items()):
        plot_individual(
            cluster_dir / f"integrated_by_joint_k{k}.png",
            integrated,
            labels.astype(str),
            f"{bundle.display_name} integrated embedding UMAP by joint k={k}",
            "cluster",
            point_size,
            seed,
        )

    config["figure_layout"] = {
        "title": "centered; panel-letter prefixes omitted",
        "multi_panel_legend": "below each subplot",
        "single_panel_legend": (
            "right for at most 10 short labels; below for many or long labels"
        ),
        "coordinates_refit": False,
    }
    config["figure_render"] = {
        "script": Path(__file__).resolve(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "rendered_at": datetime.now().astimezone().isoformat(),
    }
    write_json(config_path, config)
    print(f"[rerender-umap] completed {config['dataset']}: {figures_dir}", flush=True)


def main() -> None:
    args = parse_args()
    configs = sorted(RESULT_ROOT.rglob("analysis/umap/umap_config.json"))
    selected = []
    for path in configs:
        config = json.loads(path.read_text())
        if args.dataset == "all" or config["dataset"] == args.dataset:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No result_v6 UMAP config for dataset={args.dataset!r}.")
    for path in selected:
        rerender(path)


if __name__ == "__main__":
    main()
