"""Read and verify persisted joint assignments from existing analysis results.

These functions never cluster or write into an input run. Supervised evaluation
and filtering use the P6 metric authority; pooled values only check alignment.
"""
from __future__ import annotations

from data_io import saved_assignments as sa
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from analysis.metrics import (
    joint_per_section_valid_truth as valid_truth,
    joint_per_section_metric_row as make_metric_row,
)
from analysis.protocols import EVALUATION_MODE, MISSING_STRINGS

RESULT_COLUMNS = [
    "dataset",
    "method",
    "preprocessing",
    "evaluation_mode",
    "section",
    "label",
    "k",
    "n_obs_section",
    "n_obs_labeled",
    "n_truth_classes",
    "n_joint_clusters_observed",
    "ari",
    "nmi",
    "joint_label_source",
    "truth_source",
]


def joint_directories(analysis: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in (analysis / "clustering").glob("joint_k*"):
        match = re.fullmatch(r"joint_k(\d+)", path.name)
        if match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def section_label_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("labels_*.csv")
        if path.name != "labels_all.csv"
    )


def validate_existing_pooled_metric(
    *,
    dataset: str,
    analysis: Path,
    label: str,
    k: int,
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
) -> None:
    """Alignment-only check; pooled values are never written to output."""
    metrics_path = analysis / "metrics/supervised_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    expected = metrics[
        metrics["mode"].eq("joint")
        & metrics["scope"].eq("combined")
        & metrics["label"].eq(label)
        & metrics["k"].eq(k)
    ]
    if len(expected) != 1:
        raise ValueError(
            f"{dataset}/k={k}/{label}: expected one existing pooled row, got {len(expected)}"
        )
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    expected_row = expected.iloc[0]
    if int(expected_row["n_obs"]) != len(y_true):
        raise ValueError(
            f"{dataset}/k={k}/{label}: pooled n_obs mismatch "
            f"{len(y_true)} != {expected_row['n_obs']}"
        )
    if not (
        np.isclose(ari, expected_row["ARI"], rtol=0.0, atol=1e-12)
        and np.isclose(nmi, expected_row["NMI"], rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            f"{dataset}/k={k}/{label}: pooled alignment check failed; "
            f"ARI {ari} vs {expected_row['ARI']}, NMI {nmi} vs {expected_row['NMI']}"
        )


def validate_joint_concatenation(
    analysis: Path, directory: Path, predictions: list[np.ndarray], k: int
) -> None:
    concatenated = np.concatenate(predictions)
    labels_all_path = directory / "labels_all.csv"
    cache_path = analysis / f"_fit_cache/joint_combined_k{k}.npy"
    if labels_all_path.is_file():
        reference = sa.read_assignment_table(labels_all_path, usecols=["cluster"])["cluster"].to_numpy()
    elif cache_path.is_file():
        reference = np.asarray(sa.read_assignment_array(cache_path))
    else:
        raise FileNotFoundError(
            f"No combined joint-label reference for {directory}"
        )
    if not np.array_equal(concatenated, reference):
        raise ValueError(
            f"{directory}: concatenated per-section labels differ from saved joint labels"
        )


def calculate_inline_dataset(
    dataset: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    analysis: Path = config["analysis"]
    labels: list[str] = config["labels"]
    rows: list[dict[str, Any]] = []
    for k, directory in joint_directories(analysis):
        print(f"[{dataset}] k={k}", flush=True)
        prediction_parts: list[np.ndarray] = []
        pooled: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            label: ([], []) for label in labels
        }
        for label_path in section_label_files(directory):
            needed = ["section", "cluster", *labels]
            frame = sa.read_assignment_table(label_path, usecols=needed, low_memory=False)
            section_values = frame["section"].astype(str).unique()
            if len(section_values) != 1:
                raise ValueError(f"{label_path}: expected one section")
            section = str(section_values[0])
            prediction = frame["cluster"].to_numpy()
            prediction_parts.append(prediction)
            for label in labels:
                row, y_true, y_pred = make_metric_row(
                    dataset=dataset,
                    section=section,
                    label=label,
                    k=k,
                    truth=frame[label],
                    prediction=prediction,
                    joint_label_source=label_path,
                    truth_source=label_path,
                )
                rows.append(row)
                pooled[label][0].append(y_true)
                pooled[label][1].append(y_pred)
        validate_joint_concatenation(analysis, directory, prediction_parts, k)
        for label, (truths, predictions) in pooled.items():
            validate_existing_pooled_metric(
                dataset=dataset,
                analysis=analysis,
                label=label,
                k=k,
                truths=truths,
                predictions=predictions,
            )
    return rows


def load_spatch_metadata(analysis: Path) -> tuple[Path, dict[str, pd.DataFrame]]:
    run_dir = analysis.parents[1]
    truth_path = run_dir / "spot_metadata.csv.gz"
    metadata = pd.read_csv(
        truth_path,
        usecols=["block_id", "section", "cell_type_common"],
        dtype={"block_id": str, "section": str},
        low_memory=False,
    ).rename(columns={"block_id": "spot_id"})
    if metadata[["section", "spot_id"]].duplicated().any():
        raise ValueError(f"Duplicate SPATCH section/spot_id rows in {truth_path}")
    sections = {
        str(section): frame.reset_index(drop=True)
        for section, frame in metadata.groupby("section", sort=False)
    }
    return truth_path, sections


def calculate_spatch(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = "SPATCH"
    analysis: Path = config["analysis"]
    truth_path, metadata_by_section = load_spatch_metadata(analysis)
    rows: list[dict[str, Any]] = []
    for k, directory in joint_directories(analysis):
        print(f"[{dataset}] k={k}", flush=True)
        prediction_parts: list[np.ndarray] = []
        pooled_truths: list[np.ndarray] = []
        pooled_predictions: list[np.ndarray] = []
        for label_path in section_label_files(directory):
            section = label_path.stem.removeprefix("labels_")
            if section not in metadata_by_section:
                raise ValueError(f"{label_path}: section absent from SPATCH metadata")
            frame = sa.read_assignment_table(
                label_path,
                usecols=["spot_id", "cluster"],
                dtype={"spot_id": str},
            )
            metadata = metadata_by_section[section]
            if not np.array_equal(
                frame["spot_id"].astype(str).to_numpy(),
                metadata["spot_id"].astype(str).to_numpy(),
            ):
                raise ValueError(f"{label_path}: spot_id order differs from metadata")
            prediction = frame["cluster"].to_numpy()
            prediction_parts.append(prediction)
            row, y_true, y_pred = make_metric_row(
                dataset=dataset,
                section=section,
                label="cell_type_common",
                k=k,
                truth=metadata["cell_type_common"],
                prediction=prediction,
                joint_label_source=label_path,
                truth_source=truth_path,
            )
            rows.append(row)
            pooled_truths.append(y_true)
            pooled_predictions.append(y_pred)
        validate_joint_concatenation(analysis, directory, prediction_parts, k)
        validate_existing_pooled_metric(
            dataset=dataset,
            analysis=analysis,
            label="cell_type_common",
            k=k,
            truths=pooled_truths,
            predictions=pooled_predictions,
        )
    return rows
