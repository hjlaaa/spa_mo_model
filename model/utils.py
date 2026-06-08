"""Thin adapters and validators for COSIE-style preprocessing.

These helpers intentionally avoid implementing normalization, log transforms,
scaling, PCA, CLR, or other preprocessing algorithms. The actual data
preprocessing remains in ``model.data_preprocessing`` and follows COSIE.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def is_missing_input(x: Any) -> bool:
    """Return True for the missing-input sentinels accepted by wrappers."""

    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "na"}:
        return True
    return False


def ensure_dir(path: str | os.PathLike | None) -> None:
    """Create a directory or a file's parent directory if needed."""

    if is_missing_input(path):
        return
    path_obj = Path(path)
    target_dir = path_obj if path_obj.suffix == "" else path_obj.parent
    if str(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)


def load_h5ad_if_needed(x: Any):
    """Load an h5ad path if needed; return AnnData-like objects unchanged."""

    if is_missing_input(x):
        return None
    if isinstance(x, (str, os.PathLike)):
        import scanpy as sc

        return sc.read_h5ad(str(x))
    return x


def get_embedding_shape(x: Any):
    """Best-effort shape extraction for summaries."""

    if is_missing_input(x):
        return None
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if hasattr(x, "X") and hasattr(x.X, "shape"):
        return tuple(x.X.shape)
    return None


def standard_modality_result(
    modality: str,
    present: bool,
    shape=None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build a small, JSON-friendly modality summary."""

    return {
        "modality": modality,
        "present": bool(present),
        "shape": list(shape) if shape is not None else None,
        "message": message,
    }


def check_spatial_key(adata: Any, spatial_key: str = "spatial") -> bool:
    """Raise a clear error if an AnnData object lacks the requested spatial key."""

    if adata is None:
        return False
    if not hasattr(adata, "obsm"):
        raise TypeError("Expected an AnnData-like object with .obsm.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData object is missing obsm['{spatial_key}'].")
    return True


def check_obs_names_consistency(adatas: Iterable[Any]) -> bool:
    """Warn when non-missing AnnData objects do not share obs_names order."""

    present = [adata for adata in adatas if adata is not None]
    if len(present) <= 1:
        return True
    reference = list(present[0].obs_names)
    for idx, adata in enumerate(present[1:], start=1):
        if list(adata.obs_names) != reference:
            warnings.warn(
                "AnnData obs_names differ across modalities. COSIE-style "
                "preprocessing does not reorder by obs_names; it relies on "
                "input row order and spatial consistency. Mismatch at "
                f"non-missing modality index {idx}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
    return True


def check_spatial_consistency(
    adatas: Iterable[Any],
    spatial_key: str = "spatial",
    raise_on_mismatch: bool = True,
) -> bool:
    """Check COSIE-style same-section spatial consistency."""

    spatial_arrays = []
    for adata in adatas:
        if adata is None:
            continue
        check_spatial_key(adata, spatial_key=spatial_key)
        spatial_arrays.append(np.asarray(adata.obsm[spatial_key]))

    if len(spatial_arrays) <= 1:
        return True

    reference = spatial_arrays[0]
    for idx, spatial in enumerate(spatial_arrays[1:], start=1):
        if not np.array_equal(reference, spatial):
            message = (
                "Inconsistent spatial coordinates across modalities in the "
                f"same section for obsm['{spatial_key}']; mismatch at "
                f"non-missing modality index {idx}."
            )
            if raise_on_mismatch:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return False
    return True


def summarize_adata(adata: Any, spatial_key: str = "spatial") -> dict[str, Any] | None:
    """Return a compact AnnData summary for CLI output."""

    if adata is None:
        return None
    obsm_shapes = {}
    if hasattr(adata, "obsm"):
        for key in adata.obsm.keys():
            value = adata.obsm[key]
            obsm_shapes[key] = list(value.shape) if hasattr(value, "shape") else None
    return {
        "shape": list(adata.shape) if hasattr(adata, "shape") else None,
        "n_obs": int(adata.n_obs) if hasattr(adata, "n_obs") else None,
        "n_vars": int(adata.n_vars) if hasattr(adata, "n_vars") else None,
        "has_spatial": bool(hasattr(adata, "obsm") and spatial_key in adata.obsm),
        "obsm": obsm_shapes,
    }


def compute_knn_graph(spatial_coords: Any, k: int = 5, device=None):
    """Construct a spatial KNN graph without changing spot order.

    Adapted from /home/hujinlan/cosie/COSIE/utils.py::compute_knn_graph.
    This project-stage helper is intentionally limited to spatial coordinates;
    feature graph construction remains disabled in the first model version.

    Parameters
    ----------
    spatial_coords
        Spatial coordinate array or tensor with shape ``[n_spots, coord_dim]``.
    k
        Number of spatial neighbors. The output includes self edges, matching
        COSIE's ``n_neighbors + 1`` behavior.
    device
        Optional target device for the returned ``edge_index``.

    Returns
    -------
    torch.LongTensor
        PyTorch Geometric-style ``edge_index`` with shape ``[2, n_edges]``.
    """

    import torch
    from sklearn.neighbors import NearestNeighbors

    if k <= 0:
        raise ValueError("k must be positive for spatial KNN graph construction.")

    if isinstance(spatial_coords, torch.Tensor):
        coords = spatial_coords.detach().cpu().numpy()
    else:
        coords = np.asarray(spatial_coords)

    if coords.ndim != 2:
        raise ValueError(
            "spatial_coords must be a 2D array with shape [n_spots, coord_dim]."
        )

    n_spots = coords.shape[0]
    if n_spots <= k:
        raise ValueError(
            f"Need more spots than k for KNN graph construction; got n_spots={n_spots}, k={k}."
        )

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    source = indices[:, 0].repeat(k + 1)
    target = indices[:, 0:].flatten()
    edge_index = np.vstack((source, target))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def compute_spatial_knn_graph_with_weights(
    spatial_coords: Any,
    k: int = 5,
    include_self_loop: bool = True,
    undirected: bool = True,
    delta: float = 1e-8,
    device=None,
):
    """Construct a spatial KNN graph with source-normalized distance weights.

    Self-loops are assigned raw weight 1.0 and participate in the same
    source-row normalization as all neighbor edges. When ``undirected=True``,
    reverse edges are added and duplicate edges are coalesced by taking the
    larger raw weight before the final source-row normalization. Spot order is
    never changed.
    """

    import torch
    from sklearn.neighbors import NearestNeighbors

    if k <= 0:
        raise ValueError("k must be positive for spatial KNN graph construction.")

    if isinstance(spatial_coords, torch.Tensor):
        coords = spatial_coords.detach().cpu().numpy()
    else:
        coords = np.asarray(spatial_coords)

    if coords.ndim != 2:
        raise ValueError(
            "spatial_coords must be a 2D array with shape [n_spots, coord_dim]."
        )

    n_spots = coords.shape[0]
    if n_spots <= k:
        raise ValueError(
            f"Need more spots than k for KNN graph construction; got n_spots={n_spots}, k={k}."
        )

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    sigma = np.maximum(distances[:, -1], delta)

    edge_weight_map: dict[tuple[int, int], float] = {}
    for src in range(n_spots):
        for dist, tgt in zip(distances[src], indices[src]):
            if src == int(tgt):
                if not include_self_loop:
                    continue
                raw_weight = 1.0
            else:
                raw_weight = float(np.exp(-(dist**2) / ((sigma[src] ** 2) + delta)))

            key = (src, int(tgt))
            edge_weight_map[key] = max(edge_weight_map.get(key, 0.0), raw_weight)
            if undirected:
                reverse_key = (int(tgt), src)
                edge_weight_map[reverse_key] = max(
                    edge_weight_map.get(reverse_key, 0.0),
                    raw_weight,
                )

    if include_self_loop:
        for src in range(n_spots):
            key = (src, src)
            edge_weight_map[key] = max(edge_weight_map.get(key, 0.0), 1.0)

    row_sum = np.zeros(n_spots, dtype=np.float64)
    for (src, _tgt), weight in edge_weight_map.items():
        row_sum[src] += weight

    edges = sorted(edge_weight_map.keys())
    source = np.array([src for src, _tgt in edges], dtype=np.int64)
    target = np.array([tgt for _src, tgt in edges], dtype=np.int64)
    weights = np.array(
        [
            edge_weight_map[(src, tgt)] / (row_sum[src] + delta)
            for src, tgt in edges
        ],
        dtype=np.float32,
    )

    edge_index = torch.tensor(np.vstack((source, target)), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    return edge_index, edge_weight


def l2_normalize(x, dim: int = -1, eps: float = 1e-8):
    """L2-normalize a tensor along ``dim`` with an epsilon guard."""

    import torch

    tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=True)
    return tensor / norm.clamp_min(eps)


def cosine_cost_matrix(
    x,
    y,
    eps: float = 1e-8,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
):
    """Compute clipped cosine distance cost for UOT matching."""

    x_norm = l2_normalize(x, dim=-1, eps=eps)
    y_norm = l2_normalize(y, dim=-1, eps=eps)
    cost = 1.0 - x_norm @ y_norm.t()
    return cost.clamp(clip_min, clip_max)
