"""Standardization and clustering with explicit, caller-owned protocol values."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def fitted_space(raw: np.ndarray, scheme: str):
    if scheme == "raw_embedding":
        return raw, None
    scaler = StandardScaler()
    return scaler.fit_transform(raw), scaler


def scaler_payload(scaler: StandardScaler, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": scaler.mean_,
        f"{prefix}_scale": scaler.scale_,
        f"{prefix}_var": scaler.var_,
    }


def fit_scaler_chunked(arrays: list[np.ndarray], chunk_size: int = 50000) -> StandardScaler:
    scaler = StandardScaler()
    for values in arrays:
        for start in range(0, len(values), chunk_size):
            scaler.partial_fit(np.asarray(values[start : start + chunk_size], dtype=np.float32))
    return scaler


def transform_to_memmap(
    path: Path,
    arrays: list[np.ndarray],
    scaler: StandardScaler,
    chunk_size: int = 50000,
) -> np.ndarray:
    n_rows = sum(len(values) for values in arrays)
    n_cols = int(arrays[0].shape[1])
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
    offset = 0
    for values in arrays:
        for start in range(0, len(values), chunk_size):
            chunk = np.asarray(values[start : start + chunk_size], dtype=np.float32)
            transformed = scaler.transform(chunk).astype(np.float32, copy=False)
            output[offset : offset + len(chunk)] = transformed
            offset += len(chunk)
    output.flush()
    del output
    return np.load(path, mmap_mode="r")


def embryo_minibatch_labels(space: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int, batch_size: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=int(k), random_state=seed, n_init=n_init,
        max_iter=max_iter, batch_size=batch_size,
        max_no_improvement=20, reassignment_ratio=0.01,
    )
    return model.fit_predict(space).astype(np.int32, copy=False)


def fit_minibatch_model(space: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int, batch_size: int) -> tuple[np.ndarray, MiniBatchKMeans]:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=max_iter,
        batch_size=batch_size,
        compute_labels=True,
        random_state=seed,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=0.01,
    )
    return model.fit_predict(space), model


def spatch_fit_model(space: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int, batch_size: int) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    raw_labels, model = fit_minibatch_model(
        space, k, seed=seed, n_init=n_init, max_iter=max_iter, batch_size=batch_size,
    )
    labels = raw_labels.astype(np.int16)
    info = {
        "k": k,
        "n_iter": int(model.n_iter_),
        "n_steps": int(model.n_steps_),
        "inertia": float(model.inertia_),
    }
    return labels, info, model.cluster_centers_


def legacy_kmeans(n_clusters: int, *, method: str, seed: int, n_init: int, max_iter: int, batch_size: int):
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=int(n_init), max_iter=int(max_iter))
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=int(n_init),
        max_iter=int(max_iter),
        batch_size=int(batch_size),
        reassignment_ratio=0.01,
    )


def kmeans_labels(space: np.ndarray, **parameters) -> np.ndarray:
    """Fit exactly once with the explicit parameters of the caller's protocol."""
    return KMeans(**parameters).fit_predict(space)


def minibatch_parameters(k: int, *, seed: int, n_init: int, max_iter: int, batch_size: int) -> dict:
    """Full estimator parameters used by the existing B9 fit-cache identity."""
    return MiniBatchKMeans(n_clusters=k, init="k-means++", max_iter=max_iter,
                          batch_size=batch_size, compute_labels=True, random_state=seed,
                          tol=0.0, max_no_improvement=10, init_size=None,
                          n_init=n_init, reassignment_ratio=0.01).get_params()


def stack_selected(
    arrays: dict[str, np.ndarray],
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> np.ndarray:
    return np.vstack(
        [np.asarray(arrays[section][selections[section]], dtype=np.float32) for section in sections]
    )


def scale_full_then_select(
    arrays: dict[str, np.ndarray],
    sections: list[str],
    selections: dict[str, np.ndarray],
    chunk_size: int = 50_000,
) -> np.ndarray:
    scaler = StandardScaler()
    for section in sections:
        values = arrays[section]
        for start in range(0, len(values), chunk_size):
            scaler.partial_fit(
                np.asarray(values[start : start + chunk_size], dtype=np.float32)
            )
    selected = stack_selected(arrays, sections, selections)
    return scaler.transform(selected).astype(np.float32, copy=False)


def apply_saved_scaler(
    values: np.ndarray, scaler_path: Path, prefix: str
) -> np.ndarray:
    saved = np.load(scaler_path)
    mean = np.asarray(saved[f"{prefix}_mean"], dtype=np.float32)
    scale = np.asarray(saved[f"{prefix}_scale"], dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32, copy=False)
