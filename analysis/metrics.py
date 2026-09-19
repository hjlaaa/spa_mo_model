"""Numerical metrics; protocol differences and output fields remain explicit."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import StandardScaler
from analysis.sampling import sample_indices, permutation_sample_indices
from analysis.protocols import EVALUATION_MODE, MISSING_STRINGS
from analysis.protocols import joint_per_section_protocol


def comparison_valid_truth(values: np.ndarray) -> np.ndarray:
    return pd.notna(values) & ~np.isin(values.astype(str), ["nan", "None", ""])


def safe_asw(space: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(space, labels))


def internal_metrics(space: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    asw = safe_asw(space, labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "calinski_harabasz": float(calinski_harabasz_score(space, labels)),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
    }


def external_metrics(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "homogeneity": float(homogeneity_score(truth, labels)),
        "completeness": float(completeness_score(truth, labels)),
        "v_measure": float(v_measure_score(truth, labels)),
    }


def simulation_internal_metrics(space: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    asw = safe_asw(space, labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": float((asw + 1.0) / 2.0),
        "calinski_harabasz": float(calinski_harabasz_score(space, labels)),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
    }


def comparison_label_rows(
    data: Any,
    space: np.ndarray,
    labels: np.ndarray,
    base_mask: np.ndarray,
    mode: str,
    scope: str,
    k: int,
    label_asw_cache: dict[tuple[str, str], float],
    *, label_keys,
) -> list[dict[str, Any]]:
    rows = []
    base_indices = np.flatnonzero(base_mask)
    for label_key in label_keys:
        values = data.truth[label_key][base_mask]
        valid = comparison_valid_truth(values)
        if valid.sum() < 2 or len(np.unique(values[valid])) < 2:
            continue
        cache_key = (scope, label_key)
        if cache_key not in label_asw_cache:
            label_asw_cache[cache_key] = safe_asw(space[valid], values[valid])
        truth = values[valid].astype(str)
        predicted = labels[valid]
        raw_asw = label_asw_cache[cache_key]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_key,
                "n_obs_labeled": int(valid.sum()),
                "n_label_classes": len(np.unique(truth)),
                "k": k,
                **external_metrics(truth, predicted),
                "label_asw": raw_asw,
                "label_asw_scaled": (raw_asw + 1.0) / 2.0,
                "label_rows_in_full_order": ",".join(
                    map(str, base_indices[valid][:5])
                ),
            }
        )
    return rows


def misar_label_rows(
    data: Any,
    space: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    mode: str,
    scope: str,
    k: int,
    asw_cache: dict[tuple[str, str], float],
    *, label_keys,
) -> list[dict[str, Any]]:
    rows = []
    for label_key in label_keys:
        values = data.truth[label_key][mask]
        valid = comparison_valid_truth(values)
        cache_key = (scope, label_key)
        if cache_key not in asw_cache:
            asw_cache[cache_key] = safe_asw(space[valid], values[valid])
        label_asw = asw_cache[cache_key]
        truth = values[valid].astype(str)
        prediction = labels[valid]
        rows.append(
            {
                "mode": mode,
                "scope": scope,
                "label": label_key,
                "n_obs_labeled": int(valid.sum()),
                "n_label_classes": len(np.unique(truth)),
                "k": k,
                **external_metrics(truth, prediction),
                "label_asw": label_asw,
                "label_asw_scaled": (label_asw + 1.0) / 2.0,
            }
        )
    return rows


def requested_valid_truth(values: np.ndarray) -> np.ndarray:
    text = np.asarray(values).astype(str)
    return ~np.isin(text, ["", "nan", "NaN", "None", "NA", "N/A", "unknown"])


def requested_internal_metrics(
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


def requested_supervised_rows(
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
        valid = requested_valid_truth(values)
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


def safe_precomputed_asw(
    distances: np.ndarray, labels: np.ndarray
) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    return float(silhouette_score(distances, labels, metric="precomputed"))


def sampled_internal_metrics(
    full_space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
) -> dict[str, float | int]:
    sample_labels = labels[sample_indices]
    asw = safe_precomputed_asw(sample_distances, sample_labels)
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "cluster_asw_n_obs": len(sample_indices),
        "calinski_harabasz": float(
            calinski_harabasz_score(full_space, labels)
        ),
        "davies_bouldin": float(davies_bouldin_score(full_space, labels)),
        "ch_dbi_n_obs": len(labels),
    }


def spatch_internal_metrics(
    space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
) -> dict[str, float | int]:
    sampled_labels = labels[sample_indices]
    asw = float(
        silhouette_score(
            sample_distances,
            sampled_labels,
            metric="precomputed",
        )
    )
    return {
        "cluster_asw": asw,
        "cluster_asw_scaled": (asw + 1.0) / 2.0,
        "cluster_asw_n_obs": len(sample_indices),
        "calinski_harabasz": float(
            calinski_harabasz_score(space, labels)
        ),
        "davies_bouldin": float(davies_bouldin_score(space, labels)),
        "ch_dbi_n_obs": len(labels),
    }


def spatch_label_asw(
    space: np.ndarray,
    truth: pd.Series,
    metadata: pd.DataFrame,
    scope: str,
    label: str,
    *, sample_size: int, seed: int,
) -> tuple[float, pd.DataFrame]:
    valid = truth.notna().to_numpy()
    valid_indices = np.flatnonzero(valid)
    values = truth.loc[valid].astype(str).to_numpy()
    if len(np.unique(values)) < 2:
        return float("nan"), pd.DataFrame()
    selected_local = permutation_sample_indices(len(valid_indices), sample_size, seed)
    selected = valid_indices[selected_local]
    asw = float(silhouette_score(space[selected], values[selected_local]))
    identities = metadata.iloc[selected][["section", "spot_id"]].copy()
    identities.insert(0, "label", label)
    identities.insert(0, "scope", scope)
    identities.insert(2, "sample_order", np.arange(len(identities)))
    return asw, identities


def spatch_supervised_rows(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    mode: str,
    scope: str,
    k: int,
    *, truth_labels,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_name in truth_labels:
        truth = metadata[label_name]
        valid = truth.notna().to_numpy()
        if int(valid.sum()) < 2:
            continue
        y_true = truth.loc[valid].astype(str).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
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


def spatch_requested_internal_row(
    space: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_distances: np.ndarray,
    mode: str,
    scope: str,
    k: int,
) -> dict[str, Any]:
    sampled_labels = labels[sample_indices]
    asw = float(silhouette_score(sample_distances, sampled_labels, metric="precomputed"))
    return {
        "mode": mode,
        "scope": scope,
        "k": k,
        "n_obs": len(labels),
        "ASW": asw,
        "ASW_Scaled": float((asw + 1.0) / 2.0),
        "CH": float(calinski_harabasz_score(space, labels)),
        "DBI": float(davies_bouldin_score(space, labels)),
    }


def embryo_cluster_metrics(space: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    x = np.asarray(space[indices], dtype=np.float32)
    y = labels[indices]
    unique = np.unique(y)
    result = {
        "metric_n_obs": int(len(indices)),
        "n_clusters_observed": int(len(unique)),
        "cluster_asw": np.nan,
        "cluster_asw_scaled": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if 1 < len(unique) < len(y):
        asw = float(silhouette_score(x, y))
        result.update(
            cluster_asw=asw,
            cluster_asw_scaled=(asw + 1.0) / 2.0,
            calinski_harabasz=float(calinski_harabasz_score(x, y)),
            davies_bouldin=float(davies_bouldin_score(x, y)),
        )
    return result


def embryo_label_asw(space: np.ndarray, truth: np.ndarray, indices: np.ndarray) -> float:
    chosen_truth = truth[indices]
    valid = pd.notna(chosen_truth) & (chosen_truth.astype(str) != "nan")
    if valid.sum() < 3 or len(np.unique(chosen_truth[valid])) < 2:
        return float("nan")
    return float(silhouette_score(np.asarray(space[indices][valid]), chosen_truth[valid].astype(str)))


def embryo_external_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    valid = pd.notna(truth) & (truth.astype(str) != "nan")
    actual = truth[valid].astype(str)
    predicted = prediction[valid]
    if len(actual) < 2 or len(np.unique(actual)) < 2:
        return {
            "n_obs_labeled": int(len(actual)), "n_label_classes": int(len(np.unique(actual))),
            "ari": np.nan, "nmi": np.nan, "homogeneity": np.nan,
            "completeness": np.nan, "v_measure": np.nan,
        }
    return {
        "n_obs_labeled": int(len(actual)),
        "n_label_classes": int(len(np.unique(actual))),
        "ari": float(adjusted_rand_score(actual, predicted)),
        "nmi": float(normalized_mutual_info_score(actual, predicted)),
        "homogeneity": float(homogeneity_score(actual, predicted)),
        "completeness": float(completeness_score(actual, predicted)),
        "v_measure": float(v_measure_score(actual, predicted)),
    }


def finite_label_mask(labels: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.Series(labels).astype("string")
    return arr.notna().to_numpy() & (arr.astype(str).to_numpy() != "nan")


def legacy_safe_embedding_cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
    *,
    metric_embedding: np.ndarray | None = None,
    full_ch_dbi: bool = False,
    sklearn_silhouette_sampling: bool = False,
) -> dict[str, float | int | None]:
    unique = np.unique(labels)
    metrics: dict[str, float | int | None] = {
        "n_clusters_observed": int(unique.shape[0]),
        "cluster_asw": None,
        "cluster_asw_scaled": None,
        "dbi": None,
        "calinski_harabasz": None,
    }
    if unique.shape[0] < 2 or unique.shape[0] >= embedding.shape[0]:
        return metrics
    if metric_embedding is None:
        idx = sample_indices(embedding.shape[0], sample_size, seed)
        x = StandardScaler().fit_transform(embedding[idx])
        y = labels[idx]
    else:
        x = metric_embedding
        y = labels
    if np.unique(y).shape[0] < 2 or np.unique(y).shape[0] >= y.shape[0]:
        return metrics
    if sklearn_silhouette_sampling:
        actual_sample_size = min(int(sample_size), len(y)) if sample_size > 0 else None
        asw = float(
            silhouette_score(
                x,
                y,
                sample_size=actual_sample_size,
                random_state=seed,
            )
        )
    else:
        asw = float(silhouette_score(x, y))
    metrics["cluster_asw"] = asw
    metrics["cluster_asw_scaled"] = float((asw + 1.0) / 2.0)
    if full_ch_dbi or metric_embedding is None:
        metrics["dbi"] = float(davies_bouldin_score(x, y))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, y))
    metrics["metric_n_obs"] = int(len(y))
    metrics["silhouette_sample_size"] = int(min(sample_size, len(y)))
    return metrics


def legacy_supervised_metrics(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    label_table: pd.DataFrame,
    label_keys: list[str],
    sample_size: int,
    seed: int,
    *,
    metric_embedding: np.ndarray | None = None,
    sklearn_silhouette_sampling: bool = False,
    label_asw_cache: dict[tuple[str, str], float] | None = None,
    cache_scope: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for key in label_keys:
        if key not in label_table.columns:
            continue
        raw = label_table[key]
        mask = finite_label_mask(raw)
        if mask.sum() < 3:
            continue
        true = raw.astype(str).to_numpy()[mask]
        pred = cluster_labels[mask]
        row: dict[str, Any] = {
            "label_key": key,
            "n_labeled": int(mask.sum()),
            "n_true_labels": int(np.unique(true).shape[0]),
            "ari": float(adjusted_rand_score(true, pred)),
            "nmi": float(normalized_mutual_info_score(true, pred)),
            "homogeneity": float(homogeneity_score(true, pred)),
            "completeness": float(completeness_score(true, pred)),
            "v_measure": float(v_measure_score(true, pred)),
            "label_asw": None,
            "label_asw_scaled": None,
        }
        if np.unique(true).shape[0] >= 2 and np.unique(true).shape[0] < true.shape[0]:
            cache_key = (cache_scope, key)
            if label_asw_cache is not None and cache_key in label_asw_cache:
                asw = label_asw_cache[cache_key]
            else:
                if metric_embedding is not None and sklearn_silhouette_sampling:
                    x = metric_embedding[mask]
                    asw = float(
                        silhouette_score(
                            x,
                            true,
                            sample_size=min(int(sample_size), len(true)),
                            random_state=seed,
                        )
                    )
                else:
                    idx_all = np.where(mask)[0]
                    idx = sample_indices(idx_all.shape[0], sample_size, seed)
                    chosen = idx_all[idx]
                    y = raw.astype(str).to_numpy()[chosen]
                    if np.unique(y).shape[0] < 2 or np.unique(y).shape[0] >= y.shape[0]:
                        rows.append(row)
                        continue
                    x = StandardScaler().fit_transform(embedding[chosen])
                    asw = float(silhouette_score(x, y))
                if label_asw_cache is not None:
                    label_asw_cache[cache_key] = asw
            row["label_asw"] = asw
            row["label_asw_scaled"] = float((asw + 1.0) / 2.0)
        rows.append(row)
    return rows


def joint_per_section_valid_truth(values: pd.Series) -> np.ndarray:
    strings = values.astype("string")
    return (
        strings.notna()
        & ~strings.str.strip().str.lower().isin(MISSING_STRINGS)
    ).to_numpy()


def joint_per_section_metric_row(
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
    valid = joint_per_section_valid_truth(truth)
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


def compute_joint_per_section_supervised_metrics(
    dataset: str,
    joint_labels: np.ndarray,
    sections: np.ndarray,
    truth: dict[str, np.ndarray],
    *,
    k: int,
    section_order,
    joint_label_source: str = "in_memory_joint_assignments",
    truth_source: str = "aligned_loader_truth",
) -> dict:
    """Evaluate joint assignments without fitting a scaler or clustering.

    Each mask selects prediction and truth in the same original row order.
    Per-section rows are authoritative; the reference defines no aggregation.
    Callers bind these inputs and this protocol in their B9 source manifest.
    """
    protocol = joint_per_section_protocol(dataset, [k], section_order)
    if protocol["status"] != "applicable":
        return {"protocol": protocol, "rows": []}
    predicted = np.asarray(joint_labels)
    sections = np.asarray(sections)
    if predicted.ndim != 1 or sections.shape != predicted.shape:
        raise ValueError("Joint assignments and sections must be aligned 1D arrays")
    if len(set(section_order)) != len(section_order) or set(sections) != set(section_order):
        raise ValueError("section_order must cover each supplied section exactly once")
    for label in protocol["truth_labels"]:
        if np.asarray(truth[label]).shape != predicted.shape:
            raise ValueError(f"{dataset}/{label}: truth and joint labels are not aligned")
    rows = []
    for section in section_order:
        mask = sections == section
        for label in protocol["truth_labels"]:
            row, _, _ = joint_per_section_metric_row(
                dataset=dataset, section=section, label=label, k=k,
                truth=pd.Series(np.asarray(truth[label])[mask]), prediction=predicted[mask],
                joint_label_source=joint_label_source, truth_source=truth_source,
            )
            rows.append({"scope": "joint_per_section", **row})
    return {"protocol": protocol, "rows": rows}
