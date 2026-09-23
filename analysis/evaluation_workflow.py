"""Reader-free evaluation orchestration for explicitly selected protocols.

Callers own input loading, truth selection and specialized comparison operations.
No dataset dispatcher; cache completion occurs only after successful work.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from analysis.batch_metrics import compute_batch_correction_metrics
from analysis.clustering import kmeans_labels, fitted_space
from analysis.metrics import (compute_joint_per_section_supervised_metrics,
    requested_internal_metrics as internal_metrics, requested_supervised_rows as supervised_rows)
from analysis.protocols import AnalysisSpec, joint_per_section_protocol
from analysis.cache import (array_identity, begin_analysis, check_output_path, data_identity,
    finish_analysis, implementation_identity, file_identity)

from analysis.protocols import REQUESTED_INTERNAL_METRICS as INTERNAL_METRICS
from analysis.protocols import REQUESTED_SUPERVISED_METRICS as SUPERVISED_METRICS
from analysis.protocols import REQUESTED_BATCH_METRICS as BATCH_METRICS

@contextmanager
def evaluation_session(output, source, *, input_dirs):
    """One B9 begin/success boundary; exceptions leave the original failure footprint.

    A cache hit is handled by the caller's existing result layout. This does not
    catch failures, retry, choose protocols or run any reader/scientific policy.
    """
    reused = begin_analysis(output, source, input_dirs=input_dirs)
    yield reused
    if not reused:
        finish_analysis(output, source)


def append_scope_metrics(internal_rows, supervised, space, labels, truth,
                         base_mask, asw_mask, *, mode, scope, k):
    """Append the existing requested rows in internal-then-supervised order."""
    internal_rows.append({"mode": mode, "scope": scope, "k": k,
                          "n_obs": len(labels), **internal_metrics(space, labels, asw_mask)})
    supervised.extend(supervised_rows(truth, base_mask, labels, mode=mode, scope=scope, k=k))


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

def evaluate_standardized_scopes(data, spec, model_version, post_ot_graphsage_scale, *,
                                 output, raw, sections, truth, input_dir,
                                 persisted_asw_mask=None, joint_per_section=False):
    """Evaluate prepared arrays with the existing requested scope/output protocol.

    raw, sections and truth are the caller's established dtype and truth projection.
    Independent scopes remain section-first; batch metrics still consume raw.
    """
    source = {"dataset": spec.key, "data": data_identity(data, truth),
              "model_version": model_version, "post_ot_graphsage_scale": post_ot_graphsage_scale,
              "protocol": {"spec": asdict(spec),
                  "clustering": KMeans(random_state=spec.seed, n_init=spec.n_init, max_iter=spec.max_iter).get_params(),
                  "scaler": StandardScaler().get_params(), "input_dtype": "float64",
                  "scaler_scope": {"joint": "all_spots", "independent": "per_section"},
                  "independent_section_order": list(np.unique(sections)),
                  "internal": INTERNAL_METRICS, "supervised": SUPERVISED_METRICS,
                  "truth_filter": ["", "nan", "NaN", "None", "NA", "N/A", "unknown"],
                  "asw_mask": None if persisted_asw_mask is None else array_identity(persisted_asw_mask),
                  "asw_scope": "full scope except persisted Thymus sample mask",
                  "batch": {"metrics": BATCH_METRICS, "space": "raw", "lisi_neighbors": 90,
                            "kbet_neighbors": 50, "pcr_components": 50,
                            "kbet_alpha": compute_batch_correction_metrics.__kwdefaults__["kbet_alpha"]}},
              "implementation": implementation_identity("generic-requested-b9-v1")}
    section_order = list(dict.fromkeys(sections))
    if joint_per_section:
        source["protocol"]["joint_per_section"] = joint_per_section_protocol(spec.display_name, spec.joint_ks, section_order)
    completion = output / "analysis_completion.json"
    with evaluation_session(output, source, input_dirs=[input_dir]) as reused:
        if reused:
            record = json.loads(completion.read_text(encoding="utf-8"))
            record["analysis"] = str(output)
            if record.get("joint_per_section", {}).get("metrics") is not None:
                record["joint_per_section"]["metrics"] = str(output / "metrics/joint_per_section_supervised_metrics.csv")
            for name in ("internal_metrics", "supervised_metrics", "batch_metrics"):
                if record[name] is not None:
                    record[name] = str(output / "metrics" / f"{name}.csv")
            return record
        clustering_root = output / "clustering"
        metrics_root = output / "metrics"
        clustering_root.mkdir(parents=True, exist_ok=True)
        metrics_root.mkdir(exist_ok=True)
    
        joint_space, scaler = fitted_space(raw, "standardized_embedding")
        scaler_arrays: dict[str, np.ndarray] = {
            "joint_mean": scaler.mean_,
            "joint_scale": scaler.scale_,
            "joint_var": scaler.var_,
        }
        all_mask = np.ones(len(raw), dtype=bool)
        internal_rows: list[dict[str, Any]] = []
        supervised: list[dict[str, Any]] = []
        per_section_rows: list[dict[str, Any]] = []
    
        for k in spec.joint_ks:
            labels = kmeans_labels(joint_space, n_clusters=k, random_state=spec.seed, n_init=spec.n_init, max_iter=spec.max_iter)
            if joint_per_section:
                per_section_rows.extend(compute_joint_per_section_supervised_metrics(
                    spec.display_name, labels, sections, truth, k=k, section_order=section_order,
                    joint_label_source=str(clustering_root / f"joint_k{k}/labels_all.csv"),
                    truth_source=str(output / "analysis_source_manifest.json") + "#identity.data.truth",
                )["rows"])
            directory = clustering_root / f"joint_k{k}"
            save_labels(directory / "labels_all.csv", data, all_mask, labels, truth)
            for section in np.unique(sections):
                mask = sections == section
                save_labels(directory / f"labels_{section}.csv", data, mask, labels[mask], truth)
            append_scope_metrics(internal_rows, supervised, joint_space, labels, truth,
                                 all_mask, persisted_asw_mask, mode="joint", scope="combined", k=k)

        for section in np.unique(sections):
            mask = sections == section
            section_space, section_scaler = fitted_space(raw[mask], "standardized_embedding")
            scaler_arrays.update(
                {
                    f"{section}_mean": section_scaler.mean_,
                    f"{section}_scale": section_scaler.scale_,
                    f"{section}_var": section_scaler.var_,
                }
            )
            local_asw_mask = None if persisted_asw_mask is None else persisted_asw_mask[mask]
            for k in spec.independent_ks:
                labels = kmeans_labels(section_space, n_clusters=k, random_state=spec.seed, n_init=spec.n_init, max_iter=spec.max_iter)
                directory = clustering_root / f"independent_k{k}"
                save_labels(directory / f"labels_{section}.csv", data, mask, labels, truth)
                append_scope_metrics(internal_rows, supervised, section_space, labels, truth,
                                     mask, local_asw_mask, mode="independent", scope=section, k=k)

        if joint_per_section and per_section_rows:
            pd.DataFrame(per_section_rows).to_csv(metrics_root / "joint_per_section_supervised_metrics.csv", index=False)
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
        record = {
            "dataset": spec.display_name,
            "analysis": str(output),
            "internal_metrics": str(metrics_root / "internal_metrics.csv"),
            "supervised_metrics": supervised_path,
            "batch_metrics": str(metrics_root / "batch_metrics.csv"),
            "supervised_skipped": not bool(supervised),
            "joint_k_values": list(spec.joint_ks),
            "independent_k_values": list(spec.independent_ks),
        }
        if joint_per_section:
            record["joint_per_section"] = {
                "status": source["protocol"]["joint_per_section"]["status"],
                "metrics": str(metrics_root / "joint_per_section_supervised_metrics.csv") if per_section_rows else None,
                "aggregation": "none",
            }
        completion.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record


def evaluate_saved_joint(data, *, dataset, display_name, truth, sections, protocol,
                         loaded, assignments_dir, output_dir, run_dir, protocol_name):
    """Evaluate already aligned joint labels; never load or recluster assignments."""
    source = {"data": data_identity(data, truth), "protocol_name": protocol_name, "protocol": protocol,
              "assignments": {str(k): [file_identity(p) for p in files] for k, (_, files) in loaded.items()},
              "implementation": implementation_identity("joint-per-section-export-v1")}
    with evaluation_session(output_dir, source, input_dirs=[run_dir, *([assignments_dir] if loaded else [])]) as reused:
        if reused:
            return json.loads((output_dir / "summary.json").read_text())
        rows = []
        for k, (labels, files) in loaded.items():
            rows.extend(compute_joint_per_section_supervised_metrics(display_name, labels,
                data.sections, truth, k=k, section_order=sections,
                joint_label_source=";".join(str(p) for p in files),
                truth_source=str(output_dir / "analysis_source_manifest.json") + "#identity.data.truth")["rows"])
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "joint_per_section_supervised_metrics.csv", index=False)
        record = {"dataset": dataset, "protocol": protocol_name, "scope": "joint_per_section",
                  "status": protocol["status"], "rows": len(rows), "aggregation": "none"}
        (output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
