#!/usr/bin/env python3
"""Complete common spa_mo_model metrics and write a compact analysis summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, default=None)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-sample-size", type=int, default=10000)
    parser.add_argument("--spatial-neighbor-k", type=int, default=6)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _embedding_path(run_dir: Path, section: str, summary: dict[str, Any]) -> Path:
    saved = summary.get("saved_files", {}).get("final_embeddings", {})
    candidates = [
        Path(saved[section]) if section in saved else None,
        run_dir / f"final_embeddings_{section}.npy",
        run_dir / "final_embeddings" / f"{section}_final_embedding.npy",
        run_dir / f"{section}_embedding.npy",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate embedding for {section} under {run_dir}")


def _spatial_path(run_dir: Path, section: str, summary: dict[str, Any]) -> Path:
    saved = summary.get("saved_files", {}).get("spatial", {})
    candidates = [
        Path(saved[section]) if section in saved else None,
        run_dir / f"spatial_{section}.npy",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate spatial coordinates for {section} under {run_dir}")


def _find_label_file(mode_dir: Path, section: str) -> Path:
    files = sorted(mode_dir.glob("*labels*.csv"))
    exact = [path for path in files if section in path.stem]
    if len(exact) == 1:
        return exact[0]
    suffix = [path for path in files if path.stem.endswith(f"_{section}")]
    if len(suffix) == 1:
        return suffix[0]
    raise FileNotFoundError(
        f"Expected one label CSV for section={section} in {mode_dir}; found {exact or files}"
    )


def _sample_indices(n_obs: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n_obs:
        return np.arange(n_obs, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=sample_size, replace=False))


def _cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict[str, float | int]:
    idx = _sample_indices(len(labels), sample_size, seed)
    x = StandardScaler().fit_transform(np.asarray(embedding[idx], dtype=np.float64))
    y = np.asarray(labels[idx])
    unique = np.unique(y)
    result: dict[str, float | int] = {
        "n_obs": int(len(labels)),
        "metric_sample_size": int(len(idx)),
        "n_clusters_observed": int(len(unique)),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if 1 < len(unique) < len(y):
        result["silhouette"] = float(silhouette_score(x, y))
        result["calinski_harabasz"] = float(calinski_harabasz_score(x, y))
        result["davies_bouldin"] = float(davies_bouldin_score(x, y))
    return result


def _load_mode_labels(
    mode_dir: Path,
    sections: list[str],
    embeddings: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    tables = {}
    for section in sections:
        table = pd.read_csv(_find_label_file(mode_dir, section))
        if "cluster" not in table.columns:
            raise KeyError(f"{mode_dir}: labels for {section} lack a cluster column")
        if len(table) != len(embeddings[section]):
            raise ValueError(
                f"{mode_dir}: {section} labels={len(table)} but embedding={len(embeddings[section])}"
            )
        tables[section] = table
    return tables


def _spatial_continuity(
    tables: dict[str, pd.DataFrame],
    n_neighbors: int,
    neighbor_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for section, table in tables.items():
        if not {"x", "y"}.issubset(table.columns):
            rows.append({"section": section, "neighbor_same_cluster_fraction": np.nan})
            continue
        if section not in neighbor_cache:
            coords = table[["x", "y"]].to_numpy(dtype=np.float64)
            k = min(max(1, n_neighbors), len(coords) - 1)
            neighbor_cache[section] = (
                NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
                .fit(coords)
                .kneighbors(coords, return_distance=False)[:, 1:]
            )
        labels = table["cluster"].to_numpy()
        indices = neighbor_cache[section]
        rows.append(
            {
                "section": section,
                "neighbor_same_cluster_fraction": float(
                    np.mean(labels[indices] == labels[:, None])
                ),
            }
        )
    return pd.DataFrame(rows)


def _read_batch_metrics(analysis_dir: Path, clustering_dir: Path) -> pd.DataFrame:
    candidates = [
        analysis_dir / "batch_correction_metrics.csv",
        clustering_dir / "batch_correction_metrics.csv",
        analysis_dir / "metrics" / "batch_correction_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _write_summary(
    dataset: str,
    run_dir: Path,
    analysis_dir: Path,
    sections: list[str],
    embeddings: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    spatial: pd.DataFrame,
    batch: pd.DataFrame,
) -> Path:
    lines = [
        f"# {dataset} spa_mo_model Analysis Summary",
        "",
        f"Run directory: `{run_dir}`",
        f"Analysis directory: `{analysis_dir}`",
        f"Total spots: {sum(len(x) for x in embeddings.values())}",
        f"Sections: {', '.join(sections)}",
        "Embedding shapes: "
        + ", ".join(f"{section}={list(embeddings[section].shape)}" for section in sections),
        "",
        "## Batch correction metrics",
        "",
        "bASW, bLISI, kBET and PCR_score are higher-is-better; "
        "PCR_batch_R2 is lower-is-better.",
        "",
    ]
    if batch.empty:
        lines.append("Batch correction metrics were not found.")
    else:
        row = batch.iloc[0]
        lines.extend(
            [
                "| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |",
                "|---:|---:|---:|---:|---:|",
                f"| {_fmt(row.get('bASW'))} | {_fmt(row.get('bLISI'))} | "
                f"{_fmt(row.get('kBET'))} | {_fmt(row.get('PCR_score'))} | "
                f"{_fmt(row.get('PCR_batch_R2'), 6)} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Joint clustering metrics",
            "",
            "Silhouette and Calinski-Harabasz are higher-is-better; "
            "Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: "
            "values near zero indicate that clusters are not simply section labels.",
            "",
            "| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    joint = metrics[(metrics["mode"] == "joint") & (metrics["scope"] == "combined")]
    for _, row in joint.sort_values("k").iterrows():
        lines.append(
            f"| {int(row['k'])} | {_fmt(row['silhouette'])} | "
            f"{_fmt(row['calinski_harabasz'])} | {_fmt(row['davies_bouldin'])} | "
            f"{_fmt(row['section_ARI_diagnostic'])} | "
            f"{_fmt(row['section_NMI_diagnostic'])} | "
            f"{int(row['metric_sample_size'])} |"
        )

    lines.extend(
        [
            "",
            "## Spatial continuity",
            "",
            "Spatial continuity is the fraction of nearest spatial neighbors assigned "
            "to the same cluster; higher is smoother but may also reflect over-smoothing.",
            "",
            "| Mode | k | Mean neighbor agreement |",
            "|---|---:|---:|",
        ]
    )
    for (mode, k), frame in spatial.groupby(["mode", "k"], sort=True):
        lines.append(
            f"| {mode} | {int(k)} | "
            f"{_fmt(frame['neighbor_same_cluster_fraction'].mean())} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Section diagnostics measure batch dependence, not biological accuracy. "
            "Biological ARI/NMI should only be interpreted when reliable cell-type or "
            "domain annotations are present.",
            "",
            "## Output files",
            "",
            f"- Common metrics: `{analysis_dir / 'metrics' / 'common_clustering_metrics.csv'}`",
            f"- Section diagnostics: `{analysis_dir / 'metrics' / 'section_diagnostic_metrics.csv'}`",
            f"- Spatial continuity summary: `{analysis_dir / 'metrics' / 'spatial_continuity_summary.csv'}`",
        ]
    )
    path = analysis_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def complete_analysis(
    dataset: str,
    run_dir: Path,
    analysis_dir: Path,
    clustering_dir: Path,
    sections: list[str],
    seed: int = 42,
    metric_sample_size: int = 10000,
    spatial_neighbor_k: int = 6,
) -> dict[str, str]:
    run_summary = _load_json(run_dir / "run_summary.json")
    embeddings = {
        section: np.load(_embedding_path(run_dir, section, run_summary))
        for section in sections
    }
    spatial_coords = {
        section: np.load(_spatial_path(run_dir, section, run_summary))
        for section in sections
    }
    for section, values in embeddings.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{section}: embedding must be a finite 2D matrix")

    mode_dirs = []
    for mode in ("joint", "independent"):
        for path in clustering_dir.glob(f"{mode}_k*"):
            try:
                k = int(path.name.rsplit("_k", 1)[1])
            except ValueError:
                continue
            mode_dirs.append((mode, k, path))
    if not mode_dirs:
        raise FileNotFoundError(f"No joint_k*/independent_k* directories in {clustering_dir}")

    metric_rows: list[dict[str, Any]] = []
    spatial_frames = []
    diagnostic_rows = []
    neighbor_cache: dict[str, np.ndarray] = {}
    for mode, k, mode_dir in sorted(mode_dirs, key=lambda item: (item[1], item[0])):
        tables = _load_mode_labels(mode_dir, sections, embeddings)
        for section, table in tables.items():
            if not {"x", "y"}.issubset(table.columns):
                coords = spatial_coords[section]
                if len(coords) != len(table):
                    raise ValueError(
                        f"{section}: spatial rows={len(coords)} but labels={len(table)}"
                    )
                table["x"] = coords[:, 0]
                table["y"] = coords[:, 1]
        continuity_path = mode_dir / "spatial_continuity.csv"
        if continuity_path.exists():
            continuity = pd.read_csv(continuity_path)
            if "sample" in continuity.columns and "section" not in continuity.columns:
                continuity = continuity.rename(columns={"sample": "section"})
            value_col = "neighbor_same_cluster_fraction"
            if value_col not in continuity.columns or continuity[value_col].isna().all():
                continuity = _spatial_continuity(
                    tables, spatial_neighbor_k, neighbor_cache
                )
                continuity.to_csv(continuity_path, index=False)
        else:
            continuity = _spatial_continuity(
                tables, spatial_neighbor_k, neighbor_cache
            )
            continuity.to_csv(continuity_path, index=False)
        continuity["mode"] = mode
        continuity["k"] = k
        spatial_frames.append(continuity)

        if mode == "joint":
            x = np.vstack([embeddings[section] for section in sections])
            labels = np.concatenate(
                [tables[section]["cluster"].to_numpy() for section in sections]
            )
            section_labels = np.concatenate(
                [np.full(len(tables[section]), section, dtype=object) for section in sections]
            )
            row = {
                "mode": mode,
                "k": k,
                "scope": "combined",
                **_cluster_metrics(x, labels, metric_sample_size, seed),
                "section_ARI_diagnostic": float(
                    adjusted_rand_score(section_labels, labels)
                ),
                "section_NMI_diagnostic": float(
                    normalized_mutual_info_score(section_labels, labels)
                ),
            }
            metric_rows.append(row)
            diagnostic_rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "section_ARI_diagnostic": row["section_ARI_diagnostic"],
                    "section_NMI_diagnostic": row["section_NMI_diagnostic"],
                }
            )
        else:
            for offset, section in enumerate(sections):
                labels = tables[section]["cluster"].to_numpy()
                metric_rows.append(
                    {
                        "mode": mode,
                        "k": k,
                        "scope": section,
                        **_cluster_metrics(
                            embeddings[section],
                            labels,
                            metric_sample_size,
                            seed + offset,
                        ),
                        "section_ARI_diagnostic": np.nan,
                        "section_NMI_diagnostic": np.nan,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    spatial = pd.concat(spatial_frames, ignore_index=True)
    metrics_dir = analysis_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "common_clustering_metrics.csv"
    diagnostic_path = metrics_dir / "section_diagnostic_metrics.csv"
    spatial_path = metrics_dir / "spatial_continuity_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    spatial.to_csv(spatial_path, index=False)
    batch = _read_batch_metrics(analysis_dir, clustering_dir)
    summary_path = _write_summary(
        dataset,
        run_dir,
        analysis_dir,
        sections,
        embeddings,
        metrics,
        spatial,
        batch,
    )
    manifest = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "clustering_dir": str(clustering_dir),
        "sections": sections,
        "metric_sample_size": metric_sample_size,
        "spatial_neighbor_k": spatial_neighbor_k,
        "metrics": str(metrics_path),
        "section_diagnostics": str(diagnostic_path),
        "spatial_continuity": str(spatial_path),
        "summary": str(summary_path),
    }
    manifest_path = analysis_dir / "analysis_completion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in manifest.items() if isinstance(value, str)}


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    clustering_dir = args.clustering_dir or analysis_dir
    result = complete_analysis(
        dataset=args.dataset,
        run_dir=args.run_dir,
        analysis_dir=analysis_dir,
        clustering_dir=clustering_dir,
        sections=args.sections,
        seed=args.seed,
        metric_sample_size=args.metric_sample_size,
        spatial_neighbor_k=args.spatial_neighbor_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
