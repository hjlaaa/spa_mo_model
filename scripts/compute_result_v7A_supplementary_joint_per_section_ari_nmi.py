#!/usr/bin/env python3
"""Compute result_v7A per-section ARI/NMI from persisted joint labels.

No clustering is run. Each saved joint partition is restricted to one section
and evaluated there. The script writes a separate supplementary directory and
refuses to overwrite it if it already exists.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT = Path("/home/hujinlan/spa_mo_model")
RESULT_ROOT = PROJECT / "result_v7A"
OUTPUT = RESULT_ROOT / "supplementary_joint_per_section_ari_nmi"
EVALUATION_MODE = "joint_labels_evaluated_per_section_without_reclustering"

DATASETS: dict[str, dict[str, Any]] = {
    "MISAR-seq": {
        "key": "misar_seq",
        "analysis": RESULT_ROOT
        / "misar_seq/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding",
        "labels": [
            "Y",
            "Combined_Clusters_annotation",
            "Combined_Clusters",
            "RNA_Clusters",
            "ATAC_Clusters",
        ],
        "truth_mode": "inline",
    },
    "Mouse Spleen": {
        "key": "mouse_spleen",
        "analysis": RESULT_ROOT
        / "mouse_spleen/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding",
        "labels": [],
        "truth_mode": "none",
    },
    "Mouse Thymus": {
        "key": "mouse_thymus",
        "analysis": RESULT_ROOT
        / "mouse_thymus/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding",
        "labels": [],
        "truth_mode": "none",
    },
    "MouseBrain": {
        "key": "mousebrain",
        "analysis": RESULT_ROOT
        / "mousebrain/v3_bidirectional_sparse_fixed_lc0.1/analysis/standardized_embedding",
        "labels": ["RegionLoupe", "annotations"],
        "truth_mode": "inline",
    },
    "Simulation": {
        "key": "simulation",
        "analysis": RESULT_ROOT
        / "simulation/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding",
        "labels": ["spatial_domain"],
        "truth_mode": "inline",
    },
    "SPATCH": {
        "key": "spatch",
        "analysis": RESULT_ROOT
        / "spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding",
        "labels": ["cell_type_common"],
        "truth_mode": "spatch_metadata",
    },
}

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
MISSING_STRINGS = {"", "nan", "na", "none", "null", "<na>"}


def valid_truth(values: pd.Series) -> np.ndarray:
    strings = values.astype("string")
    return (
        strings.notna()
        & ~strings.str.strip().str.lower().isin(MISSING_STRINGS)
    ).to_numpy()


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


def make_metric_row(
    *,
    dataset: str,
    section: str,
    label: str,
    k: int,
    truth: pd.Series,
    prediction: np.ndarray,
    joint_label_source: Path,
    truth_source: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction)
    if len(truth) != len(prediction):
        raise ValueError(
            f"{dataset}/{section}/k={k}/{label}: truth={len(truth)}, labels={len(prediction)}"
        )
    valid = valid_truth(truth)
    y_true = truth.iloc[np.flatnonzero(valid)].astype(str).to_numpy()
    y_pred = prediction[valid]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        raise ValueError(f"{dataset}/{section}/k={k}/{label}: insufficient truth")
    row = {
        "dataset": dataset,
        "method": "Spa-MO",
        "preprocessing": "standardized_embedding",
        "evaluation_mode": EVALUATION_MODE,
        "section": section,
        "label": label,
        "k": int(k),
        "n_obs_section": int(len(truth)),
        "n_obs_labeled": int(len(y_true)),
        "n_truth_classes": int(len(np.unique(y_true))),
        "n_joint_clusters_observed": int(len(np.unique(y_pred))),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "joint_label_source": str(joint_label_source),
        "truth_source": str(truth_source),
    }
    return row, y_true, y_pred


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
        reference = pd.read_csv(labels_all_path, usecols=["cluster"])["cluster"].to_numpy()
    elif cache_path.is_file():
        reference = np.asarray(np.load(cache_path))
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
            frame = pd.read_csv(label_path, usecols=needed, low_memory=False)
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
            frame = pd.read_csv(
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


def calculate_all() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, config in DATASETS.items():
        analysis: Path = config["analysis"]
        if not analysis.is_dir():
            print(f"[{dataset}] missing result; skipped", flush=True)
            continue
        if config["truth_mode"] == "none":
            print(f"[{dataset}] no biological ground truth; skipped", flush=True)
            continue
        if config["truth_mode"] == "inline":
            rows.extend(calculate_inline_dataset(dataset, config))
        elif config["truth_mode"] == "spatch_metadata":
            rows.extend(calculate_spatch(config))
        else:
            raise ValueError(f"Unknown truth mode for {dataset}")
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        ["dataset", "label", "k", "section"], kind="stable"
    )
    if frame.duplicated(["dataset", "label", "k", "section"]).any():
        raise ValueError("Duplicate dataset/label/k/section rows")
    if not np.isfinite(frame[["ari", "nmi"]].to_numpy()).all():
        raise ValueError("Non-finite ARI/NMI values")

    OUTPUT.mkdir(parents=True)
    frame.to_csv(OUTPUT / "joint_per_section_ari_nmi.csv", index=False)

    status_rows: list[dict[str, Any]] = []
    for dataset, config in DATASETS.items():
        analysis: Path = config["analysis"]
        subset = frame[frame["dataset"].eq(dataset)]
        exists = analysis.is_dir()
        if not exists:
            status = "skipped_missing_result"
            note = "No selected result directory was present; skipped as requested."
        elif config["truth_mode"] == "none":
            status = "skipped_no_biological_ground_truth"
            note = "Saved result contains no biological ground truth; skipped as requested."
        elif dataset == "MouseBrain":
            status = "computed_requested_labels_only"
            note = "Only RegionLoupe and annotations were reported."
        elif dataset == "SPATCH":
            status = "computed_requested_label_only"
            note = "Only cell_type_common was reported."
        else:
            status = "computed"
            note = "Persisted joint labels were evaluated separately within each section."
        k_values = [k for k, _ in joint_directories(analysis)] if exists else []
        status_row = {
            "dataset": dataset,
            "status": status,
            "analysis_root": str(analysis),
            "available_joint_k_values": ";".join(map(str, k_values)),
            "evaluated_k_values": ";".join(
                map(str, sorted(subset["k"].astype(int).unique()))
            ),
            "reported_sections": ";".join(
                sorted(subset["section"].astype(str).unique())
            ),
            "reported_labels": ";".join(
                sorted(subset["label"].astype(str).unique())
            ),
            "n_result_rows": int(len(subset)),
            "note": note,
        }
        status_rows.append(status_row)

        dataset_dir = OUTPUT / config["key"]
        dataset_dir.mkdir()
        if not subset.empty:
            subset.to_csv(dataset_dir / "joint_per_section_ari_nmi.csv", index=False)
        (dataset_dir / "STATUS.json").write_text(
            json.dumps(status_row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    pd.DataFrame(status_rows).to_csv(OUTPUT / "dataset_status.csv", index=False)

    created_at = datetime.now().astimezone().isoformat()
    provenance = {
        "analysis": "supplementary_joint_per_section_ari_nmi",
        "created_at": created_at,
        "method": "Spa-MO",
        "result_version": "result_v7A",
        "definition": (
            "For each persisted joint clustering, restrict its labels and biological "
            "ground truth to one section, then compute sklearn ARI and NMI."
        ),
        "evaluation_mode": EVALUATION_MODE,
        "result_root": str(RESULT_ROOT),
        "n_metric_rows": int(len(frame)),
        "reclustering_performed": False,
        "pooled_metric_reported": False,
        "macro_or_weighted_aggregation_reported": False,
        "independent_clustering_used": False,
        "existing_result_files_modified": False,
        "alignment_validation": (
            "Per-section joint labels were concatenated and matched against labels_all.csv "
            "or the persisted joint fit cache. Existing combined supervised ARI/NMI were "
            "also reproduced within absolute tolerance 1e-12; pooled values were not reported."
        ),
        "reporting_label_standard": {
            "MouseBrain": ["RegionLoupe", "annotations"],
            "SPATCH": ["cell_type_common"],
            "missing_dataset_result": "skip",
            "missing_biological_ground_truth": "skip",
        },
        "dataset_analysis_roots": {
            dataset: str(config["analysis"])
            for dataset, config in DATASETS.items()
        },
    }
    (OUTPUT / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    computed = frame.groupby("dataset").size().to_dict()
    readme = f"""# Spa-MO result_v7A joint 聚类逐 section ARI/NMI 补算结果

生成时间：`{created_at}`

## 计算口径

本目录是独立补算结果，不覆盖或修改任何 `result_v7A` 原始结果。

对每套已经保存的 `joint_k*` 聚类标签：

1. 不重新运行聚类；
2. 使用同一次 joint 聚类产生的标签，按 section 切分；
3. 在每个 section 内以非缺失生物学真值分别计算 ARI 和 NMI；
4. 每行表示一个 `dataset × label × k × section`；
5. 不报告 pooled 指标，不报告 macro/weighted 平均，不使用 independent 聚类标签。

## 数据集范围和特殊规则

- MISAR-seq：计算保存结果中的 5 个真值字段，{computed.get('MISAR-seq', 0)} 行。
- MouseBrain：只计算 `RegionLoupe` 和 `annotations`，{computed.get('MouseBrain', 0)} 行。
- Simulation：计算 `spatial_domain`，{computed.get('Simulation', 0)} 行。
- SPATCH：只计算 `cell_type_common`，{computed.get('SPATCH', 0)} 行。
- Mouse Spleen、Mouse Thymus：保存结果没有生物学真值，已跳过。
- 总结果行数：{len(frame)}。

## 文件

- `joint_per_section_ari_nmi.csv`：全部可计算的逐 section ARI/NMI。
- `dataset_status.csv`：各已有数据集的计算或跳过状态。
- 各数据集子目录：对应结果及 `STATUS.json`。
- `provenance_manifest.json`：输入来源、计算定义、对齐校验和非覆盖声明。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    rows = calculate_all()
    write_outputs(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
