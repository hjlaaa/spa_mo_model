"""Section-local feature neighbours for the pre-OT graph only.

The v15A/C rule appends uniform feature edges to normalized spatial edges,
then normalizes their combined source rows. Duplicate spatial/feature edges
deliberately remain separate contributions. Forward never builds a graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_feature_knn_graph(fused_embedding, k: int):
    """Detached FP32 cosine KNN, exact FAISS FlatIP, excluding self edges."""
    from .faiss_candidate_search import build_faiss_candidates

    if k < 1 or fused_embedding.ndim != 2 or fused_embedding.shape[0] <= k:
        raise ValueError("Feature KNN requires a 2D embedding with n_spots > k >= 1.")
    z = F.normalize(fused_embedding.detach().float(), p=2, dim=1)
    result = build_faiss_candidates(
        z, z, candidate_k=k + 1, backend="faiss_flat",
        faiss_device="gpu" if z.is_cuda else "cpu", query_batch_size=2048,
    )
    candidates = torch.as_tensor(result["candidate_idx"], dtype=torch.long, device="cpu")
    source = torch.arange(len(candidates)).unsqueeze(1)
    nonself = candidates != source
    keep = nonself & (nonself.cumsum(dim=1) <= k)
    if not torch.all(keep.sum(dim=1) == k) or (candidates[keep] < 0).any():
        raise ValueError("FAISS did not return enough valid nonself feature neighbours.")
    target = candidates[keep].reshape(len(candidates), k)
    return torch.stack((source.expand(-1, k).reshape(-1), target.reshape(-1))).to(z.device)


class FeatureGraphCache:
    """Run-local non-parametric state, absent from weights-only state_dict.

    Like the OT prior, it belongs to one fixed spot ordering. Independent
    experiments construct fresh models. It starts empty until explicit refresh.
    """

    def __init__(self, *, enabled: bool, k_spatial: int):
        if type(enabled) is not bool:
            raise ValueError("feature_graph.enabled must be a boolean.")
        if enabled and k_spatial < 1:
            raise ValueError("Feature graph requires positive spatial neighbour count.")
        self.enabled = enabled
        self.k_spatial = k_spatial
        self.k_feature = (k_spatial + 1) // 2
        self.edges = {}
        self.n_spots = {}
        self.generation = 0

    def refresh(self, fused_embeddings):
        if not self.enabled:
            raise RuntimeError("Cannot refresh a disabled feature graph.")
        if not fused_embeddings:
            raise ValueError("Feature graph refresh requires fused embeddings.")
        # Commit only after every section has been built successfully.
        edges = {s: compute_feature_knn_graph(z, self.k_feature)
                 for s, z in fused_embeddings.items()}
        self.edges = edges
        self.n_spots = {s: len(z) for s, z in fused_embeddings.items()}
        self.generation += 1
        return {str(s): {"k_spatial": self.k_spatial, "k_feature": self.k_feature,
                         "feature_edge_count": int(e.shape[1])}
                for s, e in edges.items()}

    def combine(self, section, fused, spatial_edge_index, spatial_edge_weight):
        if not self.enabled or not self.edges:
            return spatial_edge_index, spatial_edge_weight
        if section not in self.edges or self.n_spots[section] != len(fused):
            raise ValueError("Feature graph section/spot count changed; use a fresh model.")
        feature_edges = self.edges[section].to(fused.device)
        feature_weights = spatial_edge_weight.new_full(
            (feature_edges.shape[1],), 1.0 / self.k_spatial,
        )
        edges = torch.cat((spatial_edge_index, feature_edges), dim=1)
        weights = torch.cat((spatial_edge_weight, feature_weights))
        row_sum = weights.new_zeros(len(fused))
        row_sum.index_add_(0, edges[0], weights)
        return edges, weights / row_sum[edges[0]]
