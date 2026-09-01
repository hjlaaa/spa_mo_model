#!/usr/bin/env python3
"""Run only the requested clustering, internal, supervised, and batch metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mousebrain_v4_standardized as mousebrain_loader  # noqa: E402
from scripts import analyze_result_v3_standardized as small_loader  # noqa: E402
from scripts import compare_misar_seq_kmeans_preprocessing as misar  # noqa: E402
from scripts import compare_mouse_spleen_kmeans_preprocessing as spleen  # noqa: E402
from scripts import compare_mouse_thymus_kmeans_preprocessing as thymus  # noqa: E402
from scripts import compare_simulation_kmeans_preprocessing as simulation  # noqa: E402
from scripts.batch_correction_metrics import compute_batch_correction_metrics  # noqa: E402


MOUSE_RUN = "v3_bidirectional_sparse_fixed_lc0.1"
OTHER_RUN = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
INTERNAL_METRICS = ["ASW", "ASW_Scaled", "CH", "DBI"]
SUPERVISED_METRICS = ["ARI", "NMI", "Homogeneity", "Completeness"]
BATCH_METRICS = ["bASW", "bLISI", "kBET", "PCR_score"]


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    display_name: str
    joint_ks: tuple[int, ...]
    independent_ks: tuple[int, ...]
    seed: int
    n_init: int = 20
    max_iter: int = 300
    batch_max_samples: int = 0
    batch_asw_sample_size: int = 10000
    batch_seed: int = 0


SPECS = {
    "mousebrain": AnalysisSpec(
        "mousebrain", "MouseBrain", tuple(range(2, 13)),
        (5, 6, 8, 9, 10, 11, 12), 0, batch_max_samples=50000,
    ),
    "misar_seq": AnalysisSpec(
        "misar_seq", "MISAR-seq", tuple(range(2, 17)), tuple(range(2, 17)), 0,
    ),
    "mouse_spleen": AnalysisSpec(
        "mouse_spleen", "Mouse Spleen", tuple(range(2, 13)), tuple(range(2, 13)),
        0, batch_max_samples=50000,
    ),
    "simulation": AnalysisSpec(
        "simulation", "Simulation", (5, 8, 10, 12), (5, 8, 10, 12),
        42, batch_seed=42,
    ),
    "mouse_thymus": AnalysisSpec(
        "mouse_thymus", "Mouse Thymus", tuple(range(2, 13)), tuple(range(2, 13)),
        0,
    ),
}


def validate_existing_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("supervised_metrics") != SUPERVISED_METRICS:
        raise ValueError("Existing manifest has an unexpected supervised metric set.")
    if manifest.get("unsupervised_metrics") != INTERNAL_METRICS:
        raise ValueError("Existing manifest has an unexpected unsupervised metric set.")
    if manifest.get("batch_metrics") != BATCH_METRICS:
        raise ValueError("Existing manifest has an unexpected batch metric set.")
    expected_datasets = {spec.display_name for spec in SPECS.values()}
    records = {record["dataset"]: record for record in manifest.get("datasets", [])}
    if set(records) != expected_datasets:
        raise ValueError(
            f"Existing manifest datasets differ: {set(records)} != {expected_datasets}"
        )
    internal_columns = ["mode", "scope", "k", "n_obs", *INTERNAL_METRICS]
    supervised_columns = [
        "mode", "scope", "label", "k", "n_obs", *SUPERVISED_METRICS
    ]
    batch_columns = ["dataset", "n_obs", *BATCH_METRICS]
    for record in records.values():
        internal = pd.read_csv(record["internal_metrics"])
        batch = pd.read_csv(record["batch_metrics"])
        if list(internal.columns) != internal_columns:
            raise ValueError(f"Unexpected internal metric columns for {record['dataset']}.")
        if list(batch.columns) != batch_columns:
            raise ValueError(f"Unexpected batch metric columns for {record['dataset']}.")
        if not np.isfinite(internal.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError(f"Non-finite internal metric for {record['dataset']}.")
        if not np.isfinite(batch.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError(f"Non-finite batch metric for {record['dataset']}.")
        supervised_path = record.get("supervised_metrics")
        if record.get("supervised_skipped"):
            if supervised_path is not None:
                raise ValueError(f"Skipped supervised output exists for {record['dataset']}.")
        else:
            supervised = pd.read_csv(supervised_path)
            if list(supervised.columns) != supervised_columns:
                raise ValueError(
                    f"Unexpected supervised metric columns for {record['dataset']}."
                )
            if not np.isfinite(
                supervised.select_dtypes(include=[np.number]).to_numpy()
            ).all():
                raise ValueError(f"Non-finite supervised metric for {record['dataset']}.")
    return manifest


def load_data(result_root: Path, key: str):
    if key == "mousebrain":
        mousebrain_loader.RESULT_ROOT = result_root / "mousebrain" / MOUSE_RUN
        mousebrain_loader.RUN_ROOT = mousebrain_loader.RESULT_ROOT / "epochs_200"
        return mousebrain_loader.load_data()

    small_loader.RESULT_ROOT = result_root
    if key == "misar_seq":
        return small_loader._misar_data()
    if key == "mouse_spleen":
        return small_loader._paired_rna_adt_data(spleen, key)
    if key == "simulation":
        return small_loader._simulation_data()
    if key == "mouse_thymus":
        return small_loader._thymus_data()
    raise KeyError(key)


def valid_truth(values: np.ndarray) -> np.ndarray:
    text = np.asarray(values).astype(str)
    return ~np.isin(text, ["", "nan", "NaN", "None", "NA", "N/A", "unknown"])


def truth_arrays(data: Any, key: str) -> dict[str, np.ndarray]:
    if key in {"mousebrain", "misar_seq"}:
        return {name: np.asarray(values) for name, values in data.truth.items()}
    if key == "simulation":
        return {"spatial_domain": np.asarray(data.truth)}
    return {}


def save_labels(
    path: Path,
    data: Any,
    mask: np.ndarray,
    labels: np.ndarray,
    truth: dict[str, np.ndarray],
) -> None:
    payload: dict[str, Any] = {
        "section": np.asarray(data.sections)[mask],
        "obs_name": np.asarray(data.barcodes)[mask],
        "cluster": labels.astype(int),
    }
    coords = getattr(data, "coords", None)
    if coords is not None:
        payload["x"] = np.asarray(coords)[mask, 0]
        payload["y"] = np.asarray(coords)[mask, 1]
    for label_name, values in truth.items():
        payload[label_name] = values[mask]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload).to_csv(path, index=False)


def internal_metrics(
    space: np.ndarray,
    labels: np.ndarray,
    asw_mask: np.ndarray | None,
) -> dict[str, float]:
    metric_space = space if asw_mask is None else space[asw_mask]
    metric_labels = labels if asw_mask is None else labels[asw_mask]
    unique = np.unique(metric_labels)
    if len(unique) < 2 or len(unique) >= len(metric_labels):
        asw = float("nan")
    else:
        asw = float(silhouette_score(metric_space, metric_labels))
    return {
        "ASW": asw,
        "ASW_Scaled": float((asw + 1.0) / 2.0),
        "CH": float(calinski_harabasz_score(space, labels)),
        "DBI": float(davies_bouldin_score(space, labels)),
    }


def supervised_rows(
    truth: dict[str, np.ndarray],
    base_mask: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    scope: str,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_name, all_values in truth.items():
        values = all_values[base_mask]
        valid = valid_truth(values)
        if int(valid.sum()) < 2 or len(np.unique(values[valid])) < 2:
            continue
        y_true = values[valid].astype(str)
        y_pred = labels[valid]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_name,
                "k": k,
                "n_obs": int(valid.sum()),
                "ARI": float(adjusted_rand_score(y_true, y_pred)),
                "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
                "Homogeneity": float(homogeneity_score(y_true, y_pred)),
                "Completeness": float(completeness_score(y_true, y_pred)),
            }
        )
    return rows


def fixed_asw_mask(data: Any, key: str) -> np.ndarray | None:
    if key != "mouse_thymus":
        return None
    sample_path = (
        ROOT
        / "results/mouse_thymus_preprocessing_comparison/shared_metrics/asw_sample.csv"
    )
    sample = pd.read_csv(sample_path)
    return thymus.sample_mask(data, sample)


def analyze_dataset(
    result_root: Path,
    spec: AnalysisSpec,
    model_version: str,
    post_ot_graphsage_scale: float,
) -> dict[str, Any]:
    data = load_data(result_root, spec.key)
    raw = np.asarray(data.embedding, dtype=np.float64)
    sections = np.asarray(data.sections).astype(str)
    truth = truth_arrays(data, spec.key)
    output = Path(data.output_root) / "standardized_embedding"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}")
    clustering_root = output / "clustering"
    metrics_root = output / "metrics"
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()

    scaler = StandardScaler()
    joint_space = scaler.fit_transform(raw)
    scaler_arrays: dict[str, np.ndarray] = {
        "joint_mean": scaler.mean_,
        "joint_scale": scaler.scale_,
        "joint_var": scaler.var_,
    }
    all_mask = np.ones(len(raw), dtype=bool)
    persisted_asw_mask = fixed_asw_mask(data, spec.key)
    internal_rows: list[dict[str, Any]] = []
    supervised: list[dict[str, Any]] = []

    for k in spec.joint_ks:
        labels = KMeans(
            n_clusters=k,
            random_state=spec.seed,
            n_init=spec.n_init,
            max_iter=spec.max_iter,
        ).fit_predict(joint_space)
        directory = clustering_root / f"joint_k{k}"
        save_labels(directory / "labels_all.csv", data, all_mask, labels, truth)
        for section in np.unique(sections):
            mask = sections == section
            save_labels(directory / f"labels_{section}.csv", data, mask, labels[mask], truth)
        internal_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": len(labels),
                **internal_metrics(joint_space, labels, persisted_asw_mask),
            }
        )
        supervised.extend(
            supervised_rows(
                truth, all_mask, labels, mode="joint", scope="combined", k=k
            )
        )

    for section in np.unique(sections):
        mask = sections == section
        section_scaler = StandardScaler()
        section_space = section_scaler.fit_transform(raw[mask])
        scaler_arrays.update(
            {
                f"{section}_mean": section_scaler.mean_,
                f"{section}_scale": section_scaler.scale_,
                f"{section}_var": section_scaler.var_,
            }
        )
        local_asw_mask = None if persisted_asw_mask is None else persisted_asw_mask[mask]
        for k in spec.independent_ks:
            labels = KMeans(
                n_clusters=k,
                random_state=spec.seed,
                n_init=spec.n_init,
                max_iter=spec.max_iter,
            ).fit_predict(section_space)
            directory = clustering_root / f"independent_k{k}"
            save_labels(directory / f"labels_{section}.csv", data, mask, labels, truth)
            internal_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": len(labels),
                    **internal_metrics(section_space, labels, local_asw_mask),
                }
            )
            supervised.extend(
                supervised_rows(
                    truth,
                    mask,
                    labels,
                    mode="independent",
                    scope=section,
                    k=k,
                )
            )

    pd.DataFrame(internal_rows).to_csv(
        metrics_root / "internal_metrics.csv", index=False
    )
    supervised_path: str | None = None
    if supervised:
        path = metrics_root / "supervised_metrics.csv"
        pd.DataFrame(supervised).to_csv(path, index=False)
        supervised_path = str(path)

    batch_all = compute_batch_correction_metrics(
        raw,
        sections,
        dataset=spec.display_name,
        method=f"spa_mo_model_result_{model_version}",
        batch_label_name="section",
        max_samples=spec.batch_max_samples,
        asw_sample_size=spec.batch_asw_sample_size,
        lisi_neighbors=90,
        kbet_neighbors=50,
        seed=spec.batch_seed,
        pcr_components=50,
    )
    batch_row = {
        "dataset": spec.display_name,
        "n_obs": int(len(raw)),
        **{metric: batch_all[metric] for metric in BATCH_METRICS},
    }
    pd.DataFrame([batch_row]).to_csv(metrics_root / "batch_metrics.csv", index=False)
    np.savez_compressed(output / "scaler_parameters.npz", **scaler_arrays)

    config = {
        "dataset": spec.display_name,
        "model_version": model_version,
        "post_ot_graphsage_scale": post_ot_graphsage_scale,
        "preprocessing": "standardized_embedding",
        "standardization": "sklearn.preprocessing.StandardScaler",
        "joint_scaler_scope": "all_spots",
        "independent_scaler_scope": "fit_per_section",
        "kmeans": {
            "implementation": "sklearn.cluster.KMeans",
            "joint_k_values": list(spec.joint_ks),
            "independent_k_values": list(spec.independent_ks),
            "seed": spec.seed,
            "n_init": spec.n_init,
            "max_iter": spec.max_iter,
        },
        "metrics": {
            "supervised": SUPERVISED_METRICS if supervised else [],
            "supervised_skipped": not bool(supervised),
            "unsupervised": INTERNAL_METRICS,
            "batch": BATCH_METRICS,
        },
        "batch_metric_parameters": {
            "batch_key": "section",
            "max_samples": spec.batch_max_samples,
            "asw_sample_size": spec.batch_asw_sample_size,
            "lisi_neighbors": 90,
            "kbet_neighbors": 50,
            "pcr_components": 50,
            "seed": spec.batch_seed,
        },
        "n_obs": int(len(raw)),
        "embedding_dim": int(raw.shape[1]),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset": spec.display_name,
        "analysis": str(output),
        "internal_metrics": str(metrics_root / "internal_metrics.csv"),
        "supervised_metrics": supervised_path,
        "batch_metrics": str(metrics_root / "batch_metrics.csv"),
        "supervised_skipped": not bool(supervised),
        "joint_k_values": list(spec.joint_ks),
        "independent_k_values": list(spec.independent_ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-version", default="v7A")
    parser.add_argument("--post-ot-graphsage-scale", type=float, default=0.5)
    args = parser.parse_args()
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post-ot-graphsage-scale must be finite and non-negative.")
    result_root = args.result_root.resolve()
    manifest_path = result_root / "requested_metrics_manifest.json"
    if manifest_path.is_file():
        manifest = validate_existing_manifest(manifest_path)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        print(
            f"RESULT_{args.model_version.upper()}_EXISTING_ANALYSIS_VALIDATION: PASS",
            flush=True,
        )
        return
    results = [
        analyze_dataset(
            result_root,
            spec,
            args.model_version,
            args.post_ot_graphsage_scale,
        )
        for spec in SPECS.values()
    ]
    manifest = {
        "analysis": f"{args.model_version}_requested_metrics_only",
        "model_version": args.model_version,
        "post_ot_graphsage_scale": args.post_ot_graphsage_scale,
        "result_root": str(result_root),
        "supervised_metrics": SUPERVISED_METRICS,
        "unsupervised_metrics": INTERNAL_METRICS,
        "batch_metrics": BATCH_METRICS,
        "datasets": results,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
