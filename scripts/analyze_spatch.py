#!/usr/bin/env python3
"""Analyze full-resolution spatch spa_mo_model embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_misar_seq_clustering as base
from complete_spa_mo_analysis import complete_analysis


SECTIONS = ["section1", "section2"]
LABELS = ["cell_type_common", "spatial_cluster", "codex_coarse_label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze spa_mo_model spatch embeddings.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "spatch" / "fullspot_200ep_gpu_seed42",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", default=",".join(map(str, range(2, 21))))
    parser.add_argument("--metric_sample_size", type=int, default=10000)
    parser.add_argument(
        "--plot_max_points",
        type=int,
        default=0,
        help="Maximum points per spatial plot; <=0 draws every spot.",
    )
    parser.add_argument("--point_size", type=float, default=0.2)
    parser.add_argument("--spatial_neighbor_k", type=int, default=6)
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Redraw retained spatial cluster plots from saved full-spot labels.",
    )
    args = parser.parse_args()
    args.section_order = ",".join(SECTIONS)
    args.label_keys = ",".join(LABELS)
    args.dpi = 220
    args.kmeans_method = "minibatch"
    args.batch_size = 4096
    args.n_init = 20
    args.max_iter = 300
    args.batch_metrics_max_samples = 100000
    args.batch_asw_sample_size = 10000
    args.batch_lisi_neighbors = 90
    args.kbet_neighbors = 50
    args.batch_metric_seed = args.seed
    args.kbet_alpha = 0.05
    args.pcr_components = 50
    args.unified_full_metric_space = True
    args.retain_k_directories = [5, 8, 10, 12, 16, 20]
    return args


def load_inputs(input_dir: Path):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(input_dir / "spot_metadata.csv.gz", low_memory=False)
    embeddings = {}
    spatial = {}
    spot_indices = {}
    obs_meta = {}
    sources = {"metadata": str(input_dir / "spot_metadata.csv.gz")}
    for section in SECTIONS:
        candidates = [
            input_dir / f"final_embeddings_{section}.npy",
            input_dir / "final_embeddings" / f"{section}_final_embedding.npy",
        ]
        emb_path = next((path for path in candidates if path.is_file()), None)
        if emb_path is None:
            raise FileNotFoundError(
                f"Cannot locate {section} embedding; tried: {candidates}"
            )
        emb = np.load(emb_path, mmap_mode="r")
        meta = metadata.loc[metadata["section"].astype(str).eq(section)].reset_index(drop=True)
        if len(meta) != emb.shape[0]:
            raise ValueError(f"{section}: embedding/metadata row mismatch.")
        embeddings[section] = np.asarray(emb)
        spatial[section] = meta[["x", "y"]].to_numpy(dtype=np.float32)
        spot_indices[section] = np.arange(len(meta), dtype=np.int64)
        obs_meta[section] = meta
        sources[f"embedding_{section}"] = str(emb_path)
    return embeddings, spatial, spot_indices, obs_meta, summary, sources, SECTIONS


def install_spatial_neighbor_cache() -> None:
    """Avoid rebuilding the same million-spot spatial index for every k."""
    neighbor_cache: dict[tuple[int, int], np.ndarray] = {}

    def cached_agreement(
        coords: np.ndarray, labels: np.ndarray, k: int
    ) -> float | None:
        if coords.shape[0] <= 1:
            return None
        k_eff = max(1, min(int(k), coords.shape[0] - 1))
        key = (id(coords), k_eff)
        if key not in neighbor_cache:
            neighbor_cache[key] = (
                NearestNeighbors(n_neighbors=k_eff + 1)
                .fit(coords[:, :2])
                .kneighbors(coords[:, :2], return_distance=False)[:, 1:]
            )
        indices = neighbor_cache[key]
        return float(np.mean(labels[indices] == labels[:, None]))

    base.spatial_neighbor_agreement = cached_agreement


def redraw_retained_spatial_plots(
    output_dir: Path,
    spatial: dict[str, np.ndarray],
    sections: list[str],
    args: argparse.Namespace,
) -> int:
    """Replace retained spatial plots without rerunning clustering or metrics."""
    written = 0
    for mode in ("joint", "independent"):
        for k in args.retain_k_directories:
            mode_dir = output_dir / f"{mode}_k{k}"
            if not mode_dir.is_dir():
                raise FileNotFoundError(mode_dir)
            for section in sections:
                label_path = mode_dir / f"{mode}_k{k}_{section}_labels.csv"
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
                labels = pd.read_csv(label_path, usecols=["cluster"])["cluster"].to_numpy()
                if len(labels) != len(spatial[section]):
                    raise ValueError(
                        f"{label_path}: labels={len(labels)} but spatial={len(spatial[section])}"
                    )
                png_path = mode_dir / f"{mode}_k{k}_{section}_spatial.png"
                base.plot_spatial(
                    spatial[section],
                    labels,
                    f"{mode} k={k} {section}",
                    png_path,
                    args.point_size,
                    args.dpi,
                    args.plot_max_points,
                    args.seed,
                )
                written += 1

    summary_path = output_dir / "clustering_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["plot_max_points"] = int(args.plot_max_points)
        summary["spatial_plot_point_policy"] = "all_spots"
        summary["spatial_plot_spot_counts"] = {
            section: int(len(spatial[section])) for section in sections
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return written


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "clustering_analysis"
    base.ensure_dir(output_dir)
    embeddings, spatial, spot_indices, obs_meta, run_summary, sources, sections = load_inputs(input_dir)
    if args.plots_only:
        written = redraw_retained_spatial_plots(
            output_dir, spatial, sections, args
        )
        print(f"SPATCH_SPA_MO_MODEL_FULL_SPATIAL_PLOTS: PASS ({written} files)")
        return
    stacked_for_metrics = np.vstack([embeddings[section] for section in sections]).astype(
        np.float32, copy=False
    )
    args._joint_metric_embedding = StandardScaler().fit_transform(stacked_for_metrics)
    del stacked_for_metrics
    args._section_metric_embeddings = {
        section: StandardScaler().fit_transform(
            np.asarray(embeddings[section], dtype=np.float32)
        )
        for section in sections
    }
    for section in sections:
        spatial_path = input_dir / f"spatial_{section}.npy"
        if not spatial_path.is_file():
            np.save(spatial_path, spatial[section])
        sources[f"spatial_{section}"] = str(spatial_path)
    install_spatial_neighbor_cache()
    batch_metrics, batch_path = base.save_batch_correction_metrics(
        output_dir, embeddings, sections, args
    )
    batch_metrics["dataset"] = "spatch"
    pd.DataFrame([batch_metrics]).to_csv(batch_path, index=False)
    results = []
    metrics = []
    for k in base.parse_int_list(args.n_clusters):
        results.append(
            base.run_joint(
                embeddings, spatial, spot_indices, obs_meta, output_dir, k, LABELS, args
            )
        )
        results.append(
            base.run_independent(
                embeddings, spatial, spot_indices, obs_meta, output_dir, k, LABELS, args
            )
        )
    for result in results:
        metrics.extend(result["metrics"])
    metric_path = output_dir / "clustering_metrics.csv"
    pd.DataFrame(metrics).to_csv(metric_path, index=False)
    summary = {
        "dataset": "spatch",
        "method": "spa_mo_model",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sections": sections,
        "section_order": sections,
        "seed": args.seed,
        "labels": LABELS,
        "label_keys": LABELS,
        "cluster_scan": list(range(2, 21)),
        "n_clusters": base.parse_int_list(args.n_clusters),
        "spatial_point_size": args.point_size,
        "point_size": args.point_size,
        "dpi": args.dpi,
        "plot_max_points": args.plot_max_points,
        "kmeans_method": args.kmeans_method,
        "batch_size": args.batch_size,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "metric_sample_size": args.metric_sample_size,
        "evaluation_protocol": {
            "kmeans_input": "raw_final_embedding",
            "metric_standardization": "StandardScaler_fit_on_full_embedding",
            "silhouette_sampling": "sklearn_sample_size_10000_random_state_42",
            "calinski_harabasz_n": int(sum(len(embeddings[s]) for s in sections)),
            "davies_bouldin_n": int(sum(len(embeddings[s]) for s in sections)),
            "batch_metrics_n": args.batch_metrics_max_samples,
            "batch_asw_n": args.batch_asw_sample_size,
        },
        "spatial_neighbor_k": args.spatial_neighbor_k,
        "batch_metrics_max_samples": args.batch_metrics_max_samples,
        "batch_asw_sample_size": args.batch_asw_sample_size,
        "batch_lisi_neighbors": args.batch_lisi_neighbors,
        "kbet_neighbors": args.kbet_neighbors,
        "kbet_alpha": args.kbet_alpha,
        "pcr_components": args.pcr_components,
        "batch_metric_seed": args.batch_metric_seed,
        "embedding_shapes": {key: list(value.shape) for key, value in embeddings.items()},
        "spatial_shapes": {key: list(value.shape) for key, value in spatial.items()},
        "batch_correction_metrics_path": batch_path,
        "batch_correction_metrics": batch_metrics,
        "source_paths": sources,
        "results": results,
    }
    (output_dir / "clustering_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    complete_analysis(
        dataset="spatch",
        run_dir=input_dir,
        analysis_dir=output_dir,
        clustering_dir=output_dir,
        sections=sections,
        seed=args.seed,
        metric_sample_size=args.metric_sample_size,
        spatial_neighbor_k=args.spatial_neighbor_k,
        primary_metrics_path=metric_path,
    )
    manifest_path = output_dir / "analysis_completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_parameters"] = str(input_dir / "run_summary.json")
    manifest["analysis_parameters"] = str(output_dir / "clustering_summary.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("SPATCH_SPA_MO_MODEL_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
