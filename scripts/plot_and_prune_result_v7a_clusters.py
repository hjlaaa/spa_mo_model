#!/usr/bin/env python3
"""Add v6-style spatial cluster plots and retain only selected result_v7A K values."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import analyze_result_v7a_requested_metrics as analysis  # noqa: E402
from scripts import compare_misar_seq_kmeans_preprocessing as misar  # noqa: E402
from scripts import compare_mouse_spleen_kmeans_preprocessing as spleen  # noqa: E402
from scripts import compare_mouse_thymus_kmeans_preprocessing as thymus  # noqa: E402
from scripts import compare_simulation_kmeans_preprocessing as simulation  # noqa: E402
from scripts import compare_mousebrain_kmeans_preprocessing as mousebrain  # noqa: E402


RETAINED_K = {
    "mousebrain": (9, 10, 11),
    "misar_seq": (12, 14, 15),
    "mouse_spleen": (3, 5),
    "mouse_thymus": (5, 8, 10),
    "simulation": (5,),
}

PLOT_MODULES = {
    "mousebrain": mousebrain,
    "misar_seq": misar,
    "mouse_spleen": spleen,
    "mouse_thymus": thymus,
    "simulation": simulation,
}


def parse_k(directory: Path) -> tuple[str, int]:
    for mode in ("joint", "independent"):
        prefix = f"{mode}_k"
        if directory.name.startswith(prefix):
            return mode, int(directory.name.removeprefix(prefix))
    raise ValueError(f"Unexpected clustering directory: {directory}")


def plot_dataset(result_root: Path, key: str) -> dict[str, Any]:
    data = analysis.load_data(result_root, key)
    module = PLOT_MODULES[key]
    retained = RETAINED_K[key]
    output = Path(data.output_root) / "standardized_embedding"
    cluster_root = output / "clustering"
    if not cluster_root.is_dir():
        raise FileNotFoundError(cluster_root)

    sections = np.asarray(data.sections).astype(str)
    barcodes = np.asarray(data.barcodes).astype(str)
    coords = np.asarray(data.coords)
    generated: list[str] = []
    for mode in ("joint", "independent"):
        for k in retained:
            directory = cluster_root / f"{mode}_k{k}"
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for section in np.unique(sections):
                mask = sections == section
                label_path = directory / f"labels_{section}.csv"
                frame = pd.read_csv(label_path)
                if len(frame) != int(mask.sum()):
                    raise ValueError(
                        f"{label_path}: {len(frame)} rows != expected {int(mask.sum())}."
                    )
                if not np.array_equal(
                    frame["obs_name"].astype(str).to_numpy(), barcodes[mask]
                ):
                    raise ValueError(f"{label_path}: barcode order does not match embedding.")
                labels = frame["cluster"].to_numpy(dtype=int)
                plot_path = directory / f"spatial_{section}.png"
                title = (
                    f"{data.name} standardized_embedding {mode} "
                    f"K={k} {section}"
                )
                if key == "mouse_thymus":
                    module.plot_spatial(
                        plot_path,
                        coords[mask],
                        labels,
                        title,
                        section,
                    )
                else:
                    module.plot_spatial(plot_path, coords[mask], labels, title)
                if not plot_path.is_file() or plot_path.stat().st_size == 0:
                    raise RuntimeError(f"Plot was not created correctly: {plot_path}")
                generated.append(str(plot_path))

    return {
        "key": key,
        "dataset": analysis.SPECS[key].display_name,
        "analysis": str(output),
        "cluster_root": str(cluster_root),
        "retained_k": list(retained),
        "generated_plots": generated,
    }


def prune_dataset(record: dict[str, Any]) -> dict[str, Any]:
    key = record["key"]
    retained = set(RETAINED_K[key])
    output = Path(record["analysis"])
    cluster_root = Path(record["cluster_root"])
    removed: list[str] = []
    for directory in sorted(cluster_root.iterdir()):
        if not directory.is_dir():
            continue
        mode, k = parse_k(directory)
        if k in retained:
            continue
        if directory.resolve().parent != cluster_root.resolve():
            raise ValueError(f"Unsafe prune target: {directory}")
        shutil.rmtree(directory)
        removed.append(str(directory))

    remaining = sorted(
        directory.name for directory in cluster_root.iterdir() if directory.is_dir()
    )
    expected = sorted(
        f"{mode}_k{k}"
        for mode in ("joint", "independent")
        for k in retained
    )
    if remaining != expected:
        raise ValueError(f"Unexpected remaining directories for {key}: {remaining}")

    metrics_dir = output / "metrics"
    filtered_metrics: dict[str, dict[str, Any]] = {}
    for filename in ("internal_metrics.csv", "supervised_metrics.csv"):
        path = metrics_dir / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        before = len(frame)
        frame = frame[frame["k"].isin(retained)].copy()
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        filtered_metrics[filename] = {
            "rows_before": before,
            "rows_after": len(frame),
            "retained_k": sorted(frame["k"].unique().astype(int).tolist()),
        }

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["kmeans"]["joint_k_values"] = sorted(retained)
    config["kmeans"]["independent_k_values"] = sorted(retained)
    config["retained_clustering_k_values"] = sorted(retained)
    config["plot_k_values"] = sorted(retained)
    config["cluster_plot_style_source"] = "result_v6 dataset-specific plotting functions"
    config["cluster_plot_count"] = len(record["generated_plots"])
    config["removed_clustering_directories"] = removed
    config["plots_and_pruning_updated_at"] = datetime.now().astimezone().isoformat()
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        **record,
        "removed_directories": removed,
        "remaining_directories": remaining,
        "filtered_metrics": filtered_metrics,
    }


def update_analysis_manifest(result_root: Path, records: list[dict[str, Any]]) -> None:
    path = result_root / "requested_metrics_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_name = {record["dataset"]: record for record in records}
    for dataset in manifest["datasets"]:
        record = by_name[dataset["dataset"]]
        dataset["joint_k_values"] = record["retained_k"]
        dataset["independent_k_values"] = record["retained_k"]
        dataset["cluster_plot_count"] = len(record["generated_plots"])
    manifest["clustering_retention_policy"] = {
        record["dataset"]: record["retained_k"] for record in records
    }
    manifest["cluster_plot_style_source"] = (
        "result_v6 dataset-specific plotting functions"
    )
    manifest["plots_and_pruning_updated_at"] = (
        datetime.now().astimezone().isoformat()
    )
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "result_v7A",
    )
    args = parser.parse_args()
    result_root = args.result_root.resolve()

    # Complete and verify every requested plot before deleting any directory.
    plotted = [plot_dataset(result_root, key) for key in RETAINED_K]
    records = [prune_dataset(record) for record in plotted]
    update_analysis_manifest(result_root, records)
    manifest = {
        "operation": "add_v6_style_cluster_plots_and_prune_unretained_k",
        "result_root": str(result_root),
        "datasets": records,
        "total_generated_plots": sum(
            len(record["generated_plots"]) for record in records
        ),
        "total_removed_directories": sum(
            len(record["removed_directories"]) for record in records
        ),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    output = result_root / "clustering_plot_prune_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
