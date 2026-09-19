"""Batch-correction diagnostics for saved embeddings.

This module intentionally mirrors the metric definitions used by
``/home/hujinlan/mofa+/scripts/batch_correction_metrics.py`` so spa_mo_model
and the baseline reports use the same BASW / BLISI / kBET / PCR logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_batch_correction_metrics(
    embedding: np.ndarray,
    batch_labels: Sequence[Any] | np.ndarray,
    *,
    dataset: str,
    method: str,
    batch_label_name: str = "section",
    max_samples: int = 50000,
    asw_sample_size: int = 10000,
    lisi_neighbors: int = 90,
    kbet_neighbors: int = 50,
    seed: int = 0,
    kbet_alpha: float = 0.05,
    pcr_components: int = 50,
) -> dict[str, Any]:
    """Compute BASW, BLISI, kBET, and PCR batch diagnostics.

    Higher is better for ``bASW``, ``bLISI``, ``kBET`` and ``PCR_score``.
    Lower is better for ``kBET_rejection_rate`` and ``PCR_batch_R2``.
    ``max_samples <= 0`` uses all valid observations.
    """

    x = np.asarray(embedding, dtype=float)
    labels = np.asarray(batch_labels).astype(str)
    if x.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {x.shape}.")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"embedding rows and labels differ: {x.shape[0]} vs {labels.shape[0]}.")

    valid = np.isfinite(x).all(axis=1)
    x = x[valid]
    labels = labels[valid]
    n_valid_total = int(labels.shape[0])

    if n_valid_total < 3 or len(np.unique(labels)) < 2:
        return {
            "dataset": dataset,
            "method": method,
            "batch_key": batch_label_name,
            "batch_label": batch_label_name,
            "n_obs": int(embedding.shape[0]),
            "n_obs_total": int(embedding.shape[0]),
            "n_valid_total": n_valid_total,
            "n_used": n_valid_total,
            "n_batches": int(len(np.unique(labels))),
            "batch_categories": ",".join(map(str, np.unique(labels))),
            "bASW_raw": float("nan"),
            "bASW": float("nan"),
            "bLISI_raw": float("nan"),
            "bLISI": float("nan"),
            "kBET_rejection_rate": float("nan"),
            "kBET": float("nan"),
            "PCR_batch_R2": float("nan"),
            "PCR_score": float("nan"),
            "note": "Need at least 2 batches and 3 observations.",
        }

    rng = np.random.default_rng(seed)
    if max_samples and max_samples > 0 and n_valid_total > max_samples:
        idx = np.sort(rng.choice(n_valid_total, size=int(max_samples), replace=False))
        x = x[idx]
        labels = labels[idx]

    x = StandardScaler().fit_transform(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = int(x.shape[0])
    categories, codes = np.unique(labels, return_inverse=True)
    n_batches = int(len(categories))
    batch_counts = dict(zip(categories.tolist(), np.bincount(codes, minlength=n_batches).astype(int).tolist()))

    b_asw_raw = np.nan
    b_asw = np.nan
    b_asw_sample_n = n
    if n_batches >= 2 and n > n_batches:
        sil_size = min(int(asw_sample_size), n) if asw_sample_size and asw_sample_size > 0 else None
        b_asw_sample_n = int(sil_size) if sil_size is not None else n
        b_asw_raw = float(silhouette_score(x, labels, sample_size=sil_size, random_state=seed))
        b_asw = float(1.0 - abs(b_asw_raw))

    lisi_k = max(1, min(int(lisi_neighbors), n - 1))
    kbet_k = max(1, min(int(kbet_neighbors), n - 1))
    neighbor_idx, knn_backend = _knn_indices(x, max(lisi_k, kbet_k), seed)

    lisi_idx = neighbor_idx[:, :lisi_k]
    neigh_codes = codes[lisi_idx]
    counts = np.stack([(neigh_codes == i).sum(axis=1) for i in range(n_batches)], axis=1).astype(float)
    probs = counts / float(lisi_k)
    lisi = 1.0 / np.sum(probs * probs, axis=1)
    b_lisi_raw = float(np.mean(lisi))
    b_lisi = float((b_lisi_raw - 1.0) / (n_batches - 1.0)) if n_batches > 1 else np.nan
    b_lisi = float(np.clip(b_lisi, 0.0, 1.0)) if np.isfinite(b_lisi) else np.nan

    kbet_idx = neighbor_idx[:, :kbet_k]
    kbet_codes = codes[kbet_idx]
    obs = np.stack([(kbet_codes == i).sum(axis=1) for i in range(n_batches)], axis=1).astype(float)
    global_probs = np.bincount(codes, minlength=n_batches).astype(float) / float(n)
    expected = kbet_k * global_probs
    safe = expected > 0
    chi_stat = ((obs[:, safe] - expected[safe]) ** 2 / expected[safe]).sum(axis=1)
    pvals = chi2.sf(chi_stat, df=int(safe.sum() - 1))
    rejection_rate = float(np.mean(pvals < kbet_alpha))
    kbet = float(1.0 - rejection_rate)

    pcr_batch_r2 = _weighted_pcr_batch_r2(x, codes, n_batches, pcr_components, seed)
    pcr_score = float(1.0 - pcr_batch_r2) if np.isfinite(pcr_batch_r2) else np.nan

    metrics: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "batch_key": batch_label_name,
        "batch_label": batch_label_name,
        "n_obs": int(embedding.shape[0]),
        "n_obs_total": int(embedding.shape[0]),
        "n_valid_total": n_valid_total,
        "n_used": n,
        "n_batches": n_batches,
        "batch_categories": ",".join(map(str, categories)),
        "batch_counts": _json_safe_dict(batch_counts),
        "max_samples": int(max_samples),
        "sample_size_requested": int(max_samples),
        "seed": int(seed),
        "knn_backend": knn_backend,
        "embedding_scaled": True,
        "bASW_raw": b_asw_raw,
        "bASW": b_asw,
        "bASW_sample_size": b_asw_sample_n,
        "bLISI_raw": b_lisi_raw,
        "bLISI": b_lisi,
        "bLISI_neighbors": int(lisi_k),
        "kBET_rejection_rate": rejection_rate,
        "kBET": kbet,
        "kBET_neighbors": int(kbet_k),
        "kBET_alpha": float(kbet_alpha),
        "PCR_batch_R2": pcr_batch_r2,
        "PCR_score": pcr_score,
        "PCR_components": int(min(pcr_components, x.shape[1], max(1, n - 1))),
        "note": "Higher is better for bASW, bLISI, kBET and PCR_score; lower is better for PCR_batch_R2.",
    }
    metrics.update(
        {
            "basw_batch_silhouette_mean": b_asw_raw,
            "basw_score_abs_mean": b_asw,
            "blisi_mean": b_lisi_raw,
            "blisi_normalized": b_lisi,
            "n_neighbors": int(lisi_k),
            "kbet_rejection_rate": rejection_rate,
            "kbet_acceptance_rate": kbet,
            "pcr_batch_r2": pcr_batch_r2,
            "pcr_score": pcr_score,
            "pcr_n_components": metrics["PCR_components"],
        }
    )
    return metrics


def _weighted_pcr_batch_r2(
    x: np.ndarray,
    batch_codes: np.ndarray,
    n_batches: int,
    pcr_components: int,
    seed: int,
) -> float:
    if n_batches < 2 or x.shape[0] < 3:
        return np.nan
    n_components = int(min(pcr_components, x.shape[1], max(1, x.shape[0] - 1)))
    if n_components < 1:
        return np.nan
    svd_solver = "randomized" if n_components < min(x.shape) else "auto"
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=seed)
    pcs = pca.fit_transform(x)
    design = pd.get_dummies(batch_codes, drop_first=True).to_numpy(dtype=float)
    if design.shape[1] == 0:
        return np.nan
    reg = LinearRegression()
    r2_values = []
    for i in range(pcs.shape[1]):
        y = pcs[:, i]
        reg.fit(design, y)
        pred = reg.predict(design)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
        r2_values.append(r2)
    weights = pca.explained_variance_ratio_
    return float(np.average(np.asarray(r2_values), weights=weights))


def _knn_indices(x: np.ndarray, n_neighbors: int, seed: int) -> tuple[np.ndarray, str]:
    n_neighbors = max(1, min(int(n_neighbors), x.shape[0] - 1))
    if x.shape[0] > 100000:
        try:
            import hnswlib

            x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))
            index = hnswlib.Index(space="l2", dim=x32.shape[1])
            index.init_index(
                max_elements=x32.shape[0],
                ef_construction=100,
                M=16,
                random_seed=int(seed),
            )
            index.add_items(x32, np.arange(x32.shape[0]), num_threads=-1)
            index.set_ef(max(100, n_neighbors + 1))
            labels, _ = index.knn_query(x32, k=n_neighbors + 1, num_threads=-1)
            labels = labels.astype(np.int64, copy=False)
            self_mask = labels[:, 0] == np.arange(x32.shape[0])
            if np.all(self_mask):
                return labels[:, 1:], "hnswlib_l2_approx"
            cleaned = np.empty((x32.shape[0], n_neighbors), dtype=np.int64)
            for i, row in enumerate(labels):
                keep = row[row != i][:n_neighbors]
                if keep.shape[0] < n_neighbors:
                    keep = np.pad(keep, (0, n_neighbors - keep.shape[0]), mode="edge")
                cleaned[i] = keep
            return cleaned, "hnswlib_l2_approx"
        except Exception:
            pass

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:], "sklearn_exact"


def _json_safe_dict(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            safe[str(key)] = value.item()
        else:
            safe[str(key)] = value
    return safe
