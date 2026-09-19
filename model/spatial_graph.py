"""Weighted spatial KNN graph used by Stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor_utils import tensor_to_numpy


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
        coords = tensor_to_numpy(spatial_coords)
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

