#!/usr/bin/env python3
"""Aggregate and rank Phase-1 contrastive-loss experiments.

The script discovers completed ``evaluation/cluster_seed_*`` directories below
``result_v2_Contrastive_loss``.  It keeps model seed and clustering seed separate,
uses only predeclared biological-label endpoints for the primary comparison, and
does not select a best K.

Primary biological endpoints
----------------------------
* MouseBrain ``RegionLoupe`` at K=9 and ``annotations`` at K=11.
  Independent clustering uses the section-size-weighted score; joint clustering
  uses the pooled score.
* MISAR-seq ``Y`` at K=14 using joint clustering.

The overall ranking is exploratory.  First, methods are ranked within every
matched endpoint.  Endpoint ranks are averaged within metric family and dataset,
then metric families are averaged within dataset, and finally datasets receive
equal weight.  This prevents a dataset with more sections or K values from
dominating only because it emits more CSV rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMMON_METRICS = {
    "silhouette": "higher",
    "calinski_harabasz": "higher",
    "davies_bouldin": "lower",
}
BATCH_METRICS = {
    "bASW": "higher",
    "bLISI": "higher",
    "kBET": "higher",
    "PCR_score": "higher",
}
BIO_METRICS = {"ari": "higher", "nmi": "higher"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("result_v2_Contrastive_loss"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-root>/_comparison",
    )
    parser.add_argument(
        "--experiment-regex",
        default=r"^(L0_|L1_|L2_|L3_|L4_)",
        help="Only aggregate experiment directory names matching this regex.",
    )
    parser.add_argument(
        "--expected-cluster-seeds",
        default="0,1,2,3,4",
        help="Required clustering seeds for a run to enter summaries/rankings.",
    )
    parser.add_argument(
        "--model-seeds",
        default=None,
        help=(
            "Optional comma-separated training/model seeds to include, for example "
            "42. By default all discovered model seeds are included."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["source_file"] = str(path)
    return frame


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float(values.mean())
    return float(np.average(values[valid], weights=weights[valid]))


def _mean_std_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: Iterable[str],
    seed_column: str = "cluster_seed",
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, part in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_cluster_seeds"] = int(part[seed_column].nunique())
        for column in value_columns:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_metadata(eval_dir: Path, results_root: Path) -> dict:
    relative = eval_dir.relative_to(results_root)
    parts = relative.parts
    evaluation_index = parts.index("evaluation")
    prefix = parts[:evaluation_index]
    seed_part = next(part for part in prefix if part.startswith("seed_"))
    cluster_match = re.fullmatch(r"cluster_seed_(\d+)", parts[evaluation_index + 1])
    if cluster_match is None:
        raise ValueError(f"Cannot parse clustering seed from {eval_dir}")
    return {
        "dataset": prefix[0],
        "experiment": prefix[1],
        "model_seed": int(seed_part.removeprefix("seed_")),
        "cluster_seed": int(cluster_match.group(1)),
        "run_dir": str(results_root.joinpath(*prefix)),
        "evaluation_dir": str(eval_dir),
    }


def discover_evaluations(results_root: Path, experiment_regex: str) -> pd.DataFrame:
    pattern = re.compile(experiment_regex)
    rows: list[dict] = []
    for common_file in sorted(
        results_root.glob("**/evaluation/cluster_seed_*/metrics/common_clustering_metrics.csv")
    ):
        eval_dir = common_file.parent.parent
        metadata = _run_metadata(eval_dir, results_root)
        if pattern.search(metadata["experiment"]):
            rows.append(metadata)
    frame = pd.DataFrame(rows).drop_duplicates()
    if frame.empty:
        raise RuntimeError(f"No completed evaluations found below {results_root}")
    return frame.sort_values(
        ["dataset", "experiment", "model_seed", "cluster_seed"]
    ).reset_index(drop=True)


def annotate_complete_runs(
    inventory: pd.DataFrame, expected_cluster_seeds: set[int]
) -> pd.DataFrame:
    inventory = inventory.copy()
    group_columns = ["dataset", "experiment", "model_seed", "run_dir"]
    records: list[dict] = []
    for keys, part in inventory.groupby(group_columns, dropna=False, sort=True):
        observed = set(int(seed) for seed in part["cluster_seed"])
        record = dict(zip(group_columns, keys))
        record["observed_cluster_seeds"] = ",".join(map(str, sorted(observed)))
        record["n_observed_cluster_seeds"] = len(observed)
        record["expected_cluster_seeds"] = ",".join(
            map(str, sorted(expected_cluster_seeds))
        )
        record["complete_seed_set"] = observed == expected_cluster_seeds
        records.append(record)
    status = pd.DataFrame(records)
    return inventory.merge(status, on=group_columns, how="left", validate="many_to_one")


def aggregate_common(evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    per_seed_rows: list[dict] = []
    for metadata in evaluations.to_dict("records"):
        path = Path(metadata["evaluation_dir"]) / "metrics/common_clustering_metrics.csv"
        frame = _read_csv(path)
        for key, value in metadata.items():
            frame[key] = value
        raw_parts.append(frame)
        group_columns = ["mode", "k"]
        for keys, part in frame.groupby(group_columns, dropna=False, sort=True):
            row = {key: metadata[key] for key in metadata}
            row.update(dict(zip(group_columns, keys)))
            row["n_scopes"] = int(len(part))
            weights = part.get("n_obs", pd.Series(np.ones(len(part)), index=part.index))
            for metric in COMMON_METRICS:
                row[metric] = _weighted_mean(part[metric], weights)
            per_seed_rows.append(row)
    raw = pd.concat(raw_parts, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows)
    summary = _mean_std_summary(
        per_seed,
        ["dataset", "experiment", "model_seed", "mode", "k"],
        COMMON_METRICS,
    )
    return raw, summary


def aggregate_spatial(evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    per_seed_rows: list[dict] = []
    for metadata in evaluations.to_dict("records"):
        path = Path(metadata["evaluation_dir"]) / "metrics/spatial_continuity_summary.csv"
        frame = _read_csv(path)
        for key, value in metadata.items():
            frame[key] = value
        raw_parts.append(frame)
        for keys, part in frame.groupby(["mode", "k"], dropna=False, sort=True):
            row = {key: metadata[key] for key in metadata}
            row.update(dict(zip(["mode", "k"], keys)))
            row["n_sections"] = int(len(part))
            row["spatial_continuity"] = float(
                pd.to_numeric(
                    part["neighbor_same_cluster_fraction"], errors="coerce"
                ).mean()
            )
            per_seed_rows.append(row)
    raw = pd.concat(raw_parts, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows)
    summary = _mean_std_summary(
        per_seed,
        ["dataset", "experiment", "model_seed", "mode", "k"],
        ["spatial_continuity"],
    )
    return raw, summary


def _find_batch_file(eval_dir: Path) -> Path | None:
    candidates = [
        eval_dir / "batch_correction_metrics.csv",
        eval_dir / "clustering/batch_correction_metrics.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def aggregate_batch(evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    for metadata in evaluations.to_dict("records"):
        path = _find_batch_file(Path(metadata["evaluation_dir"]))
        if path is None:
            continue
        frame = _read_csv(path)
        for key, value in metadata.items():
            frame[key] = value
        raw_parts.append(frame)
    if not raw_parts:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(raw_parts, ignore_index=True)
    summary = _mean_std_summary(
        raw,
        ["dataset", "experiment", "model_seed"],
        BATCH_METRICS,
    )
    return raw, summary


def aggregate_mousebrain_biology(
    evaluations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    subset = evaluations[evaluations["dataset"] == "mousebrain"]
    for metadata in subset.to_dict("records"):
        path = Path(metadata["evaluation_dir"]) / "clustering/biological_label_metrics.csv"
        if not path.exists():
            continue
        frame = _read_csv(path)
        for key, value in metadata.items():
            frame[key] = value
        raw_parts.append(frame)
    if not raw_parts:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(raw_parts, ignore_index=True)
    expected_k = {"RegionLoupe": 9, "annotations": 11}
    selected = raw[
        raw["label_key"].isin(expected_k)
        & raw.apply(lambda row: row["n_clusters"] == expected_k[row["label_key"]], axis=1)
        & (
            ((raw["mode"] == "independent") & (raw["scope"] == "section_weighted"))
            | ((raw["mode"] == "joint") & (raw["scope"] == "pooled"))
        )
    ].copy()
    summary = _mean_std_summary(
        selected,
        [
            "dataset",
            "experiment",
            "model_seed",
            "mode",
            "n_clusters",
            "label_key",
            "scope",
        ],
        BIO_METRICS,
    )
    return raw, summary


def aggregate_misar_biology(
    evaluations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    per_seed_rows: list[dict] = []
    subset = evaluations[evaluations["dataset"] == "misar_seq"]
    for metadata in subset.to_dict("records"):
        path = Path(metadata["evaluation_dir"]) / "clustering_metrics.csv"
        if not path.exists():
            continue
        frame = _read_csv(path)
        for key, value in metadata.items():
            frame[key] = value
        biological = frame[
            frame["label_key"].notna()
            & frame["label_key"].ne("cluster")
            & frame["mode"].isin(["joint", "independent"])
        ].copy()
        raw_parts.append(biological)
        for keys, part in biological.groupby(
            ["mode", "n_clusters", "label_key"], dropna=False, sort=True
        ):
            row = {key: metadata[key] for key in metadata}
            row.update(dict(zip(["mode", "n_clusters", "label_key"], keys)))
            row["n_sections"] = int(part["section"].nunique(dropna=True))
            weights = part.get(
                "n_labeled", pd.Series(np.ones(len(part)), index=part.index)
            )
            for metric in BIO_METRICS:
                row[metric] = _weighted_mean(part[metric], weights)
            per_seed_rows.append(row)
    if not raw_parts:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(raw_parts, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows)
    # Primary MISAR endpoint is predeclared as joint Y at its 14-class K.
    selected = per_seed[
        (per_seed["mode"] == "joint")
        & (per_seed["label_key"] == "Y")
        & (per_seed["n_clusters"] == 14)
    ].copy()
    summary = _mean_std_summary(
        selected,
        ["dataset", "experiment", "model_seed", "mode", "n_clusters", "label_key"],
        BIO_METRICS,
    )
    return raw, summary


def aggregate_training(evaluations: pd.DataFrame) -> pd.DataFrame:
    run_rows = evaluations[
        ["dataset", "experiment", "model_seed", "run_dir"]
    ].drop_duplicates()
    rows: list[dict] = []
    for metadata in run_rows.to_dict("records"):
        run_dir = Path(metadata["run_dir"])
        path = run_dir / "loss_history.json"
        if not path.exists():
            continue
        history = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        row = dict(metadata)
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            preprocessing = run_summary.get("preprocessing") or {}
            resolved_config = (
                run_summary.get("resolved_model_config")
                or preprocessing.get("resolved_model_config")
                or {}
            )
            row["device"] = (
                run_summary.get("device")
                or preprocessing.get("device")
                or (resolved_config.get("training") or {}).get("device")
            )
        row["n_epochs"] = int(len(history))
        numeric = history.select_dtypes(include=[np.number])
        row["all_finite"] = bool(np.isfinite(numeric.to_numpy()).all())
        for metric in [
            "total_loss",
            "reconstruction_loss",
            "crossview_loss",
            "weighted_crossview_loss",
            "crossmodal_top1_accuracy",
            "projection_std_min",
            "projection_std_median",
            "contrastive_infonce_loss",
            "contrastive_variance_loss",
            "contrastive_covariance_loss",
            "contrastive_batch_size",
        ]:
            if metric not in history:
                continue
            values = pd.to_numeric(history[metric], errors="coerce")
            row[f"{metric}_first"] = float(values.iloc[0])
            row[f"{metric}_final"] = float(values.iloc[-1])
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_abs_max"] = float(values.abs().max())
            row[f"{metric}_last20_mean"] = float(values.tail(20).mean())
            row[f"{metric}_max_abs_step"] = float(values.diff().abs().max())
            row[f"{metric}_negative_epochs"] = int((values < 0).sum())
        batch_size = row.get("contrastive_batch_size_final")
        top1 = row.get("crossmodal_top1_accuracy_last20_mean")
        if batch_size is not None and batch_size > 0:
            row["retrieval_random_chance"] = 1.0 / batch_size
            if top1 is not None:
                row["retrieval_top1_lift_over_chance"] = top1 * batch_size
        rows.append(row)
    return pd.DataFrame(rows)


def _section_embedding_paths(run_dir: Path) -> dict[str, Path]:
    nested = run_dir / "final_embeddings"
    if nested.is_dir():
        return {
            path.name.removesuffix("_final_embedding.npy"): path
            for path in sorted(nested.glob("*_final_embedding.npy"))
        }
    return {
        path.name.removeprefix("final_embeddings_").removesuffix(".npy"): path
        for path in sorted(run_dir.glob("final_embeddings_*.npy"))
    }


def aggregate_embedding_similarity(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Compare final coordinates for runs whose shared initialization is matched."""
    runs = evaluations[
        ["dataset", "experiment", "model_seed", "run_dir"]
    ].drop_duplicates()
    rows: list[dict] = []
    for (dataset, model_seed), part in runs.groupby(
        ["dataset", "model_seed"], sort=True
    ):
        records = part.sort_values("experiment").to_dict("records")
        for left, right in itertools.combinations(records, 2):
            left_paths = _section_embedding_paths(Path(left["run_dir"]))
            right_paths = _section_embedding_paths(Path(right["run_dir"]))
            sections = sorted(set(left_paths) & set(right_paths))
            if not sections:
                continue
            dot = left_norm2 = right_norm2 = diff_norm2 = absolute_sum = 0.0
            n_values = 0
            all_finite = True
            for section in sections:
                left_array = np.load(left_paths[section], mmap_mode="r")
                right_array = np.load(right_paths[section], mmap_mode="r")
                if left_array.shape != right_array.shape:
                    all_finite = False
                    continue
                left64 = np.asarray(left_array, dtype=np.float64)
                right64 = np.asarray(right_array, dtype=np.float64)
                all_finite = all_finite and bool(
                    np.isfinite(left64).all() and np.isfinite(right64).all()
                )
                difference = left64 - right64
                dot += float(np.sum(left64 * right64))
                left_norm2 += float(np.sum(left64 * left64))
                right_norm2 += float(np.sum(right64 * right64))
                diff_norm2 += float(np.sum(difference * difference))
                absolute_sum += float(np.sum(np.abs(difference)))
                n_values += int(left64.size)
            denominator = math.sqrt(left_norm2 * right_norm2)
            symmetric_scale = 0.5 * (
                math.sqrt(left_norm2) + math.sqrt(right_norm2)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model_seed": model_seed,
                    "experiment_a": left["experiment"],
                    "experiment_b": right["experiment"],
                    "n_sections": len(sections),
                    "n_values": n_values,
                    "all_finite_and_shape_matched": all_finite,
                    "direct_flat_cosine": dot / denominator if denominator else np.nan,
                    "symmetric_relative_l2": (
                        math.sqrt(diff_norm2) / symmetric_scale
                        if symmetric_scale
                        else np.nan
                    ),
                    "mean_absolute_difference": (
                        absolute_sum / n_values if n_values else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_rows(
    common: pd.DataFrame,
    spatial: pd.DataFrame,
    batch: pd.DataFrame,
    mousebrain_bio: pd.DataFrame,
    misar_bio: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add_rows(
        frame: pd.DataFrame,
        family: str,
        metric_directions: dict[str, str],
        identity_columns: list[str],
    ) -> None:
        if frame.empty:
            return
        for record in frame.to_dict("records"):
            identity = "|".join(str(record[column]) for column in identity_columns)
            for metric, direction in metric_directions.items():
                value = record.get(f"{metric}_mean")
                if value is None or pd.isna(value):
                    continue
                rows.append(
                    {
                        "dataset": record["dataset"],
                        "experiment": record["experiment"],
                        "model_seed": record["model_seed"],
                        "family": family,
                        "endpoint": f"{family}|{identity}|{metric}",
                        "metric": metric,
                        "direction": direction,
                        "value": float(value),
                    }
                )

    add_rows(common, "internal", COMMON_METRICS, ["mode", "k"])
    add_rows(spatial, "spatial", {"spatial_continuity": "higher"}, ["mode", "k"])
    add_rows(batch, "batch", BATCH_METRICS, [])
    add_rows(
        mousebrain_bio,
        "biology",
        BIO_METRICS,
        ["mode", "n_clusters", "label_key", "scope"],
    )
    add_rows(
        misar_bio,
        "biology",
        BIO_METRICS,
        ["mode", "n_clusters", "label_key"],
    )
    return pd.DataFrame(rows)


def rank_endpoints(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked_parts: list[pd.DataFrame] = []
    group_columns = ["dataset", "model_seed", "family", "endpoint"]
    for _, part in endpoints.groupby(group_columns, dropna=False, sort=True):
        part = part.copy()
        ascending = part["direction"].iloc[0] == "lower"
        part["endpoint_rank"] = part["value"].rank(
            method="average", ascending=ascending
        )
        part["n_methods_endpoint"] = int(part["experiment"].nunique())
        ranked_parts.append(part)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    family = (
        ranked.groupby(
            ["dataset", "experiment", "model_seed", "family"], as_index=False
        )
        .agg(
            family_mean_rank=("endpoint_rank", "mean"),
            n_endpoints=("endpoint", "nunique"),
        )
        .sort_values(["dataset", "family", "family_mean_rank", "experiment"])
    )
    dataset = (
        family.groupby(["dataset", "experiment", "model_seed"], as_index=False)
        .agg(
            dataset_mean_rank=("family_mean_rank", "mean"),
            n_families=("family", "nunique"),
            dataset_n_endpoints=("n_endpoints", "sum"),
        )
        .sort_values(["dataset", "dataset_mean_rank", "experiment"])
    )
    dataset["dataset_rank"] = dataset.groupby(
        ["dataset", "model_seed"]
    )["dataset_mean_rank"].rank(method="average", ascending=True)
    overall = (
        dataset.groupby(["experiment", "model_seed"], as_index=False)
        .agg(
            overall_mean_rank=("dataset_mean_rank", "mean"),
            mean_dataset_rank=("dataset_rank", "mean"),
            n_datasets=("dataset", "nunique"),
            overall_n_endpoints=("dataset_n_endpoints", "sum"),
        )
        .sort_values(["overall_mean_rank", "mean_dataset_rank", "experiment"])
    )
    required_dataset_count = int(endpoints["dataset"].nunique())
    overall["eligible_for_overall"] = overall["n_datasets"].eq(
        required_dataset_count
    )
    overall["overall_rank"] = np.nan
    eligible = overall["eligible_for_overall"]
    overall.loc[eligible, "overall_rank"] = (
        overall.loc[eligible]
        .groupby("model_seed")["overall_mean_rank"]
        .rank(method="average", ascending=True)
    )
    return ranked, family, dataset.merge(overall, on=["experiment", "model_seed"])


def paired_strict_deltas(
    endpoints: pd.DataFrame,
    baseline: str = "L0_legacy_strict_init_matched",
    candidate: str = "L3_vicreg_strict_init_matched",
) -> pd.DataFrame:
    subset = endpoints[endpoints["experiment"].isin([baseline, candidate])].copy()
    index = ["dataset", "model_seed", "family", "endpoint", "metric", "direction"]
    wide = subset.pivot_table(index=index, columns="experiment", values="value").reset_index()
    if baseline not in wide or candidate not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[baseline, candidate]).copy()
    wide["candidate_minus_baseline"] = wide[candidate] - wide[baseline]
    wide["oriented_improvement"] = np.where(
        wide["direction"].eq("higher"),
        wide["candidate_minus_baseline"],
        -wide["candidate_minus_baseline"],
    )
    tolerance = 1e-12
    wide["outcome"] = np.select(
        [
            wide["oriented_improvement"] > tolerance,
            wide["oriented_improvement"] < -tolerance,
        ],
        ["L3_win", "L3_loss"],
        default="tie",
    )
    return wide.sort_values(["dataset", "family", "endpoint"])


def summarize_strict_metrics(
    strict_deltas: pd.DataFrame,
    baseline: str = "L0_legacy_strict_init_matched",
    candidate: str = "L3_vicreg_strict_init_matched",
) -> pd.DataFrame:
    """Average matched strict endpoints within each dataset/family/metric."""
    if strict_deltas.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    groups = ["dataset", "family", "metric", "direction"]
    for keys, part in strict_deltas.groupby(groups, dropna=False, sort=True):
        counts = part["outcome"].value_counts()
        row = dict(zip(groups, keys))
        row["n_endpoints"] = int(len(part))
        row["L0_mean"] = float(part[baseline].mean())
        row["L3_mean"] = float(part[candidate].mean())
        row["L3_minus_L0_mean"] = float(part["candidate_minus_baseline"].mean())
        row["oriented_improvement_mean"] = float(part["oriented_improvement"].mean())
        row["L3_win"] = int(counts.get("L3_win", 0))
        row["L3_loss"] = int(counts.get("L3_loss", 0))
        row["tie"] = int(counts.get("tie", 0))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_strict_model_seed_stability(
    strict_deltas: pd.DataFrame,
    baseline: str = "L0_legacy_strict_init_matched",
    candidate: str = "L3_vicreg_strict_init_matched",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize endpoint deltas first within, then across model seeds."""
    if strict_deltas.empty:
        return pd.DataFrame(), pd.DataFrame()
    per_seed_rows: list[dict] = []
    groups = ["dataset", "model_seed", "family", "metric", "direction"]
    for keys, part in strict_deltas.groupby(groups, dropna=False, sort=True):
        counts = part["outcome"].value_counts()
        row = dict(zip(groups, keys))
        row["n_endpoints"] = int(len(part))
        row["L0_endpoint_mean"] = float(part[baseline].mean())
        row["L3_endpoint_mean"] = float(part[candidate].mean())
        row["L3_minus_L0_endpoint_mean"] = float(
            part["candidate_minus_baseline"].mean()
        )
        row["oriented_improvement_endpoint_mean"] = float(
            part["oriented_improvement"].mean()
        )
        row["L3_win"] = int(counts.get("L3_win", 0))
        row["L3_loss"] = int(counts.get("L3_loss", 0))
        row["tie"] = int(counts.get("tie", 0))
        per_seed_rows.append(row)
    per_seed = pd.DataFrame(per_seed_rows)

    across_rows: list[dict] = []
    across_groups = ["dataset", "family", "metric", "direction"]
    for keys, part in per_seed.groupby(across_groups, dropna=False, sort=True):
        improvements = part["oriented_improvement_endpoint_mean"]
        row = dict(zip(across_groups, keys))
        row["n_model_seeds"] = int(part["model_seed"].nunique())
        row["model_seeds"] = ",".join(map(str, sorted(part["model_seed"].unique())))
        row["L0_mean_across_model_seeds"] = float(part["L0_endpoint_mean"].mean())
        row["L3_mean_across_model_seeds"] = float(part["L3_endpoint_mean"].mean())
        row["L3_minus_L0_mean_across_model_seeds"] = float(
            part["L3_minus_L0_endpoint_mean"].mean()
        )
        row["oriented_improvement_mean_across_model_seeds"] = float(
            improvements.mean()
        )
        row["oriented_improvement_std_across_model_seeds"] = (
            float(improvements.std(ddof=1)) if len(improvements) > 1 else 0.0
        )
        row["model_seeds_improved"] = int((improvements > 1e-12).sum())
        row["model_seeds_degraded"] = int((improvements < -1e-12).sum())
        row["model_seeds_tied"] = int((improvements.abs() <= 1e-12).sum())
        across_rows.append(row)
    return per_seed, pd.DataFrame(across_rows)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_无可用数据_"
    shown = frame[columns].copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: _fmt(value))
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, rule, *body])


def write_report(
    output_dir: Path,
    inventory: pd.DataFrame,
    evaluations: pd.DataFrame,
    common: pd.DataFrame,
    spatial: pd.DataFrame,
    batch: pd.DataFrame,
    mousebrain_bio: pd.DataFrame,
    misar_bio: pd.DataFrame,
    training: pd.DataFrame,
    embedding_similarity: pd.DataFrame,
    family_ranks: pd.DataFrame,
    dataset_and_overall: pd.DataFrame,
    strict_deltas: pd.DataFrame,
    strict_metric_summary: pd.DataFrame,
    strict_model_seed_summary: pd.DataFrame,
) -> None:
    overall_columns = [
        "experiment",
        "model_seed",
        "overall_rank",
        "overall_mean_rank",
        "mean_dataset_rank",
        "n_datasets",
        "eligible_for_overall",
        "overall_n_endpoints",
    ]
    overall = dataset_and_overall[overall_columns].drop_duplicates()
    overall = overall.sort_values(
        ["eligible_for_overall", "overall_rank", "experiment"],
        ascending=[False, True, True],
    )
    dataset_table = dataset_and_overall[
        [
            "dataset",
            "experiment",
            "dataset_rank",
            "dataset_mean_rank",
            "n_families",
            "dataset_n_endpoints",
        ]
    ].sort_values(["dataset", "dataset_rank", "experiment"])

    strict_counts = pd.DataFrame()
    if not strict_deltas.empty:
        strict_counts = (
            strict_deltas.groupby(["dataset", "family", "outcome"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for column in ["L3_win", "L3_loss", "tie"]:
            if column not in strict_counts:
                strict_counts[column] = 0

    mb_view = mousebrain_bio[
        [
            "experiment",
            "model_seed",
            "mode",
            "label_key",
            "n_clusters",
            "ari_mean",
            "ari_std",
            "nmi_mean",
            "nmi_std",
        ]
    ].sort_values(["label_key", "mode", "experiment"])
    misar_view = misar_bio[
        [
            "experiment",
            "model_seed",
            "mode",
            "label_key",
            "n_clusters",
            "ari_mean",
            "ari_std",
            "nmi_mean",
            "nmi_std",
        ]
    ].sort_values(["experiment"])

    training_columns = [
        "dataset",
        "experiment",
        "model_seed",
        "device",
        "n_epochs",
        "all_finite",
        "crossview_loss_min",
        "crossview_loss_max",
        "crossview_loss_abs_max",
        "crossview_loss_max_abs_step",
        "reconstruction_loss_final",
        "crossmodal_top1_accuracy_last20_mean",
        "retrieval_random_chance",
        "retrieval_top1_lift_over_chance",
        "projection_std_min_last20_mean",
        "projection_std_median_last20_mean",
    ]
    training_view = training[
        [column for column in training_columns if column in training]
    ].sort_values(["dataset", "experiment"])
    mixed_device_datasets: list[str] = []
    if "device" in training:
        mixed_device_datasets = sorted(
            dataset
            for dataset, part in training.groupby("dataset", sort=True)
            if part["device"].dropna().nunique() > 1
        )
    device_warning = (
        "- 硬件可比性警告：以下数据集在所汇总方法间混用了不同 device："
        + ", ".join(mixed_device_datasets)
        + "。这些数据集仍可用于开发期参考，但不属于硬件完全一致的严格比较。"
        if mixed_device_datasets
        else "- 所有纳入比较的方法在各数据集内使用相同 device。"
    )
    l4_similarity = embedding_similarity[
        embedding_similarity["experiment_a"].eq("L4_no_crossview_matched")
        | embedding_similarity["experiment_b"].eq("L4_no_crossview_matched")
    ].copy()
    if not l4_similarity.empty:
        l4_similarity = l4_similarity[
            [
                "dataset",
                "experiment_a",
                "experiment_b",
                "direct_flat_cosine",
                "symmetric_relative_l2",
                "mean_absolute_difference",
            ]
        ].sort_values(["dataset", "experiment_a", "experiment_b"])

    lines = [
        "# 跨视图损失实验自动汇总",
        "",
        "> 本报告由 `scripts/summarize_contrastive_experiments.py` 从结果 CSV 自动生成。",
        "> 排名是开发期探索性汇总，不替代多模型训练 seed 的配对统计。",
        "",
        "## 覆盖情况",
        "",
        f"- 文件系统中发现 {len(inventory)} 个模型运行 × 聚类 seed 评估目录。",
        f"- 其中 {len(evaluations)} 个属于聚类 seed 集合完整的运行并进入汇总；未完成的运行会自动排除，避免网络中断或在途任务污染排名。",
        f"- 数据集：{', '.join(sorted(evaluations['dataset'].unique()))}。",
        f"- 方法：{', '.join(sorted(evaluations['experiment'].unique()))}。",
        device_warning,
        "- 生物标签主指标固定为 MouseBrain RegionLoupe K=9、annotations K=11，以及 MISAR Y K=14；没有按结果挑选 best K。",
        "",
        "## 探索性整体排名",
        "",
        _markdown_table(overall, overall_columns),
        "",
        "只有覆盖当前全部数据集的方法才获得 `overall_rank`；排名始终在同一 model seed 内进行，尚未完成全部数据集的方法保留分数据集结果但整体名次为 NA。流程是先在同数据集、同模型 seed、同 mode/K/指标内排名，再依次对指标端点、指标族和数据集等权平均。内部聚类、空间、batch、生物标签四类指标不会因 CSV 行数不同而直接获得额外权重。",
        "",
        "## 各数据集排名",
        "",
        _markdown_table(dataset_table, list(dataset_table.columns)),
        "",
        "## 严格同初始化 L3 相对 L0 的端点胜负",
        "",
        _markdown_table(
            strict_counts,
            [column for column in ["dataset", "family", "L3_win", "L3_loss", "tie"] if column in strict_counts],
        ),
        "",
        "这里的胜负按每个预声明端点和每个可用模型 seed 判断：Silhouette/CH/空间/batch/ARI/NMI 越高越好，DBI 越低越好。单个端点胜负不代表统计显著性。",
        "",
        "## 严格 L0/L3 各指标均值与差值",
        "",
        _markdown_table(
            strict_metric_summary,
            [
                column
                for column in [
                    "dataset",
                    "family",
                    "metric",
                    "n_endpoints",
                    "L0_mean",
                    "L3_mean",
                    "L3_minus_L0_mean",
                    "oriented_improvement_mean",
                ]
                if column in strict_metric_summary
            ],
        ),
        "",
        "`oriented_improvement_mean` 已统一方向：正值表示 L3 改善，负值表示 L3 退化；DBI 的原始差值方向与该列相反。均值只在同一数据集、同一指标的预声明 mode/K 端点间计算。",
        "",
        "## 严格 L0/L3 跨模型 seed 稳定性",
        "",
        _markdown_table(
            strict_model_seed_summary,
            [
                column
                for column in [
                    "dataset",
                    "family",
                    "metric",
                    "n_model_seeds",
                    "model_seeds",
                    "oriented_improvement_mean_across_model_seeds",
                    "oriented_improvement_std_across_model_seeds",
                    "model_seeds_improved",
                    "model_seeds_degraded",
                    "model_seeds_tied",
                ]
                if column in strict_model_seed_summary
            ],
        ),
        "",
        "该表先在每个模型 seed 内对预声明 mode/K 端点取均值，再跨模型 seed 汇总；因此 clustering seed 波动与模型训练 seed 波动没有混在一起。",
        "",
        "## MouseBrain 固定 K 标签指标",
        "",
        _markdown_table(mb_view, list(mb_view.columns)),
        "",
        "## MISAR 固定 K 标签指标",
        "",
        _markdown_table(misar_view, list(misar_view.columns)),
        "",
        "## 训练数值稳定性",
        "",
        _markdown_table(training_view, list(training_view.columns)),
        "",
        "## 与 L4（无跨视图梯度）的最终 embedding 直接相似度",
        "",
        _markdown_table(l4_similarity, list(l4_similarity.columns)),
        "",
        "这些坐标级相似度只适用于本轮同模型 seed、共享层同初始化的配对实验。cosine 越接近 1、relative L2 越接近 0，说明加入的跨视图目标相对 L4 对最终表示改动越小；它不是旋转不变的表示相似性指标。",
        "",
        "## 机器可读产物",
        "",
        "- `evaluation_inventory.csv`：评估目录清单。",
        "- `common_clustering_summary.csv`：内部聚类指标，按聚类 seed 汇总。",
        "- `spatial_summary.csv`：空间连续性，先按 section 宏平均再跨聚类 seed 汇总。",
        "- `batch_summary.csv`：batch correction 指标。",
        "- `mousebrain_primary_biology_summary.csv`、`misar_primary_biology_summary.csv`：固定 K 主标签指标。",
        "- `endpoint_values_and_ranks.csv`、`family_ranks.csv`、`dataset_and_overall_ranks.csv`：排名中间量。",
        "- `strict_L3_minus_L0_endpoints.csv`：严格同初始化的配对端点差值。",
        "- `strict_L3_minus_L0_metric_summary.csv`：按数据集和指标汇总后的严格差值。",
        "- `strict_L3_minus_L0_per_model_seed.csv`、`strict_L3_minus_L0_model_seed_summary.csv`：严格配对的模型 seed 稳定性。",
        "- `training_stability.csv`：loss 范围、突变幅度与有限性审计。",
        "- `embedding_pairwise_similarity.csv`：同初始化运行的最终坐标差异。",
        "",
        "## 解释边界",
        "",
        "- 当前每个方法只有一个模型训练 seed=42；五个 clustering seeds 只能衡量聚类初始化波动，不能替代模型训练 seed。",
        "- Human Lymph Node、Mouse Spleen、Mouse Thymus 当前没有进入主排名的生物真值标签，内部几何、空间与 batch 指标之间可能存在权衡。",
        "- Batch mixing 过强也可能代表过校正，因此 batch 分数必须与有标签数据集的生物保持度联合解释。",
        "- 多模型 seed 完成前，整体排名只用于选择下一轮候选，不作为最终论文结论。",
    ]
    (output_dir / "contrastive_experiment_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_cluster_seeds = {
        int(value.strip())
        for value in args.expected_cluster_seeds.split(",")
        if value.strip()
    }
    if not expected_cluster_seeds:
        raise ValueError("--expected-cluster-seeds cannot be empty")
    inventory = discover_evaluations(results_root, args.experiment_regex)
    if args.model_seeds is not None:
        model_seeds = {
            int(value.strip())
            for value in args.model_seeds.split(",")
            if value.strip()
        }
        if not model_seeds:
            raise ValueError("--model-seeds cannot be empty when provided")
        inventory = inventory[inventory["model_seed"].isin(model_seeds)].copy()
        if inventory.empty:
            raise RuntimeError(
                f"No evaluation matched requested model seeds {sorted(model_seeds)}"
            )
    inventory = annotate_complete_runs(inventory, expected_cluster_seeds)
    evaluations = inventory[inventory["complete_seed_set"]].copy()
    if evaluations.empty:
        raise RuntimeError("No run has the complete required clustering-seed set")
    common_raw, common_summary = aggregate_common(evaluations)
    spatial_raw, spatial_summary = aggregate_spatial(evaluations)
    batch_raw, batch_summary = aggregate_batch(evaluations)
    mousebrain_raw, mousebrain_summary = aggregate_mousebrain_biology(evaluations)
    misar_raw, misar_summary = aggregate_misar_biology(evaluations)
    training = aggregate_training(evaluations)
    embedding_similarity = aggregate_embedding_similarity(evaluations)

    endpoints = _endpoint_rows(
        common_summary,
        spatial_summary,
        batch_summary,
        mousebrain_summary,
        misar_summary,
    )
    endpoint_ranks, family_ranks, dataset_and_overall = rank_endpoints(endpoints)
    strict_deltas = paired_strict_deltas(endpoints)
    strict_metric_summary = summarize_strict_metrics(strict_deltas)
    strict_per_model_seed, strict_model_seed_summary = (
        summarize_strict_model_seed_stability(strict_deltas)
    )
    dataset_metric_means = (
        endpoints.groupby(
            ["dataset", "experiment", "model_seed", "family", "metric", "direction"],
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"value": "endpoint_mean"})
    )

    tables = {
        "evaluation_inventory.csv": inventory,
        "common_clustering_raw.csv": common_raw,
        "common_clustering_summary.csv": common_summary,
        "spatial_raw.csv": spatial_raw,
        "spatial_summary.csv": spatial_summary,
        "batch_raw.csv": batch_raw,
        "batch_summary.csv": batch_summary,
        "mousebrain_biology_raw.csv": mousebrain_raw,
        "mousebrain_primary_biology_summary.csv": mousebrain_summary,
        "misar_biology_raw.csv": misar_raw,
        "misar_primary_biology_summary.csv": misar_summary,
        "training_stability.csv": training,
        "embedding_pairwise_similarity.csv": embedding_similarity,
        "endpoint_values_and_ranks.csv": endpoint_ranks,
        "family_ranks.csv": family_ranks,
        "dataset_and_overall_ranks.csv": dataset_and_overall,
        "strict_L3_minus_L0_endpoints.csv": strict_deltas,
        "strict_L3_minus_L0_metric_summary.csv": strict_metric_summary,
        "strict_L3_minus_L0_per_model_seed.csv": strict_per_model_seed,
        "strict_L3_minus_L0_model_seed_summary.csv": strict_model_seed_summary,
        "dataset_family_metric_means.csv": dataset_metric_means,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    write_report(
        output_dir,
        inventory,
        evaluations,
        common_summary,
        spatial_summary,
        batch_summary,
        mousebrain_summary,
        misar_summary,
        training,
        embedding_similarity,
        family_ranks,
        dataset_and_overall,
        strict_deltas,
        strict_metric_summary,
        strict_model_seed_summary,
    )
    print(f"Aggregated {len(evaluations)} evaluations into {output_dir}")


if __name__ == "__main__":
    main()
