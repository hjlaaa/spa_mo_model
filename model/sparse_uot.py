"""Candidate-restricted sparse UOT priors."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .faiss_candidate_search import build_faiss_candidates


def _as_float_tensor(x: Any, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _as_numpy_float32(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def _l2_normalize_tensor(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def _resolve_section_order(mapping: Mapping[str, Any], section_order: Sequence[str] | None):
    if section_order is None:
        return sorted(mapping.keys())
    missing = [section for section in section_order if section not in mapping]
    if missing:
        raise KeyError(f"section_order contains sections missing from input: {missing}")
    return list(section_order)


def _valid_modalities(
    feature_dict: Mapping[str, Mapping[str, Any]],
    source_section: str,
    target_section: str,
    modalities: Iterable[str],
) -> list[str]:
    used = []
    for modality in modalities:
        source = feature_dict[source_section].get(modality)
        target = feature_dict[target_section].get(modality)
        if source is not None and target is not None:
            used.append(modality)
    if not used:
        raise ValueError(f"No shared modalities for sparse UOT {source_section}->{target_section}.")
    return used


def _union_candidate_indices(candidate_arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    n_source = candidate_arrays[0].shape[0]
    rows: list[np.ndarray] = []
    max_len = 0
    for row_idx in range(n_source):
        union = np.unique(np.concatenate([arr[row_idx] for arr in candidate_arrays]).astype(np.int64))
        union = union[union >= 0]
        if union.size == 0:
            raise ValueError(f"Source row {row_idx} has no candidate targets.")
        rows.append(union)
        max_len = max(max_len, int(union.size))

    candidate_idx = torch.full((n_source, max_len), -1, dtype=torch.long)
    candidate_mask = torch.zeros((n_source, max_len), dtype=torch.bool)
    for row_idx, values in enumerate(rows):
        length = int(values.size)
        candidate_idx[row_idx, :length] = torch.as_tensor(values, dtype=torch.long)
        candidate_mask[row_idx, :length] = True
    return candidate_idx, candidate_mask


def _compute_candidate_cosine_cost(
    source_embeddings,
    target_embeddings,
    candidate_idx: torch.Tensor,
    candidate_mask: torch.Tensor,
    stabilizer: float,
    device,
    chunk_size: int = 2048,
) -> torch.Tensor:
    source = _l2_normalize_tensor(_as_float_tensor(source_embeddings, device=device), stabilizer)
    target = _l2_normalize_tensor(_as_float_tensor(target_embeddings, device=device), stabilizer)
    idx = candidate_idx.to(device=device)
    mask = candidate_mask.to(device=device)
    n_source, k = idx.shape
    costs = torch.empty((n_source, k), dtype=torch.float32, device=device)
    safe_idx = idx.clamp_min(0)
    for start in range(0, n_source, chunk_size):
        end = min(start + chunk_size, n_source)
        gathered = target[safe_idx[start:end].reshape(-1)].reshape(end - start, k, -1)
        sim = (source[start:end, None, :] * gathered).sum(dim=-1)
        cost = (1.0 - sim).clamp(0.0, 2.0)
        cost = torch.where(mask[start:end], cost, torch.full_like(cost, 2.0))
        costs[start:end] = cost
    return costs


def _truncate_candidates_by_cost(
    candidate_idx: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_cost: torch.Tensor,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_source, current_k = candidate_idx.shape
    effective_k = min(int(candidate_k), int(current_k))
    if effective_k == current_k:
        return candidate_idx, candidate_mask, candidate_cost
    score = torch.where(
        candidate_mask.to(candidate_cost.device),
        -candidate_cost,
        torch.full_like(candidate_cost, -float("inf")),
    )
    _, positions = torch.topk(score, k=effective_k, dim=1)
    idx = candidate_idx.to(candidate_cost.device).gather(1, positions).cpu()
    mask = candidate_mask.to(candidate_cost.device).gather(1, positions).cpu()
    cost = candidate_cost.gather(1, positions)
    return idx, mask, cost


@torch.no_grad()
def sparse_unbalanced_sinkhorn_topk(
    candidate_idx: torch.Tensor | np.ndarray,
    candidate_cost: torch.Tensor | np.ndarray,
    n_source: int,
    n_target: int,
    candidate_mask: torch.Tensor | np.ndarray | None = None,
    epsilon: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 100,
    stabilizer: float = 1e-8,
    attention_topk: int = 10,
    device=None,
) -> dict[str, torch.Tensor]:
    """Run sparse unbalanced Sinkhorn on candidate edges and return row top-k."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if attention_topk <= 0:
        raise ValueError("attention_topk must be positive.")

    device = torch.device(device or "cpu")
    idx = torch.as_tensor(candidate_idx, dtype=torch.long, device=device)
    cost = torch.as_tensor(candidate_cost, dtype=torch.float32, device=device)
    if idx.ndim != 2 or cost.shape != idx.shape:
        raise ValueError(f"candidate_idx and candidate_cost must both be [N,K], got {idx.shape} and {cost.shape}.")
    if candidate_mask is None:
        mask = idx >= 0
    else:
        mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=device)
    if mask.shape != idx.shape:
        raise ValueError(f"candidate_mask shape {mask.shape} does not match candidate_idx {idx.shape}.")

    n_source = int(n_source)
    n_target = int(n_target)
    safe_idx = idx.clamp_min(0)
    row_ids = torch.arange(n_source, device=device).unsqueeze(1).expand_as(idx)
    edge_src = row_ids[mask]
    edge_tgt = safe_idx[mask]
    edge_cost = cost[mask]
    if edge_src.numel() == 0:
        raise ValueError("No candidate edges available for sparse UOT.")

    kernel_edge = torch.exp(-edge_cost / float(epsilon)).clamp_min(float(stabilizer))
    a = torch.full((n_source,), 1.0 / max(n_source, 1), dtype=torch.float32, device=device)
    b = torch.full((n_target,), 1.0 / max(n_target, 1), dtype=torch.float32, device=device)
    rho_a = float(tau_a) / (float(tau_a) + float(epsilon))
    rho_b = float(tau_b) / (float(tau_b) + float(epsilon))
    u = torch.ones(n_source, dtype=torch.float32, device=device)
    v = torch.ones(n_target, dtype=torch.float32, device=device)

    for _ in range(int(max_iter)):
        kv = torch.zeros(n_source, dtype=torch.float32, device=device)
        kv.scatter_add_(0, edge_src, kernel_edge * v[edge_tgt])
        u = (a / (kv + float(stabilizer))).pow(rho_a)

        ktu = torch.zeros(n_target, dtype=torch.float32, device=device)
        ktu.scatter_add_(0, edge_tgt, kernel_edge * u[edge_src])
        v = (b / (ktu + float(stabilizer))).pow(rho_b)

    p_edge = u[edge_src] * kernel_edge * v[edge_tgt]
    p_edge = torch.nan_to_num(p_edge, nan=0.0, posinf=0.0, neginf=0.0)
    p_edge = p_edge / (p_edge.sum() + float(stabilizer))

    p_matrix = torch.zeros_like(cost)
    p_matrix[mask] = p_edge
    effective_topk = min(int(attention_topk), int(idx.shape[1]))
    topk_raw, topk_pos = torch.topk(p_matrix, k=effective_topk, dim=1)
    topk_idx = safe_idx.gather(1, topk_pos)
    raw_topk_mass = topk_raw.sum(dim=1)
    topk_weight = topk_raw / (raw_topk_mass[:, None] + float(stabilizer))
    row_mass = torch.zeros(n_source, dtype=torch.float32, device=device)
    row_mass.scatter_add_(0, edge_src, p_edge)
    topk_coverage = torch.clamp(raw_topk_mass / (row_mass + float(stabilizer)), min=0.0, max=1.0)
    tail_mass = torch.clamp(row_mass - raw_topk_mass, min=0.0)

    target_hit_count = torch.zeros(n_target, dtype=torch.long, device=device)
    target_hit_count.scatter_add_(0, edge_tgt, torch.ones_like(edge_tgt, dtype=torch.long))

    return {
        "topk_idx": topk_idx.cpu(),
        "topk_weight": topk_weight.cpu(),
        "confidence": topk_coverage.cpu(),
        "row_mass": row_mass.cpu(),
        "raw_topk_mass": raw_topk_mass.cpu(),
        "topk_coverage": topk_coverage.cpu(),
        "tail_mass": tail_mass.cpu(),
        "target_hit_count": target_hit_count.cpu(),
    }


def _candidate_qc(candidate_idx: torch.Tensor, candidate_mask: torch.Tensor, n_target: int) -> dict[str, Any]:
    valid = candidate_idx[candidate_mask]
    unique = torch.unique(valid)
    hit_count = torch.zeros(n_target, dtype=torch.long)
    if valid.numel() > 0:
        hit_count.scatter_add_(0, valid.long(), torch.ones_like(valid, dtype=torch.long))
    return {
        "candidate_target_unique_count": int(unique.numel()),
        "candidate_target_unique_coverage": float(unique.numel() / max(n_target, 1)),
        "candidate_edge_count": int(valid.numel()),
        "candidate_target_hit_count_min": int(hit_count.min().item()) if hit_count.numel() else 0,
        "candidate_target_hit_count_max": int(hit_count.max().item()) if hit_count.numel() else 0,
        "candidate_target_hit_count_mean": float(hit_count.float().mean().item()) if hit_count.numel() else 0.0,
    }


@torch.no_grad()
def compute_initial_candidate_sparse_uot_prior(
    feature_dict: Mapping[str, Mapping[str, Any]],
    section_order: Sequence[str] | None,
    modalities: Iterable[str],
    initial_modality_candidate_k: int = 100,
    candidate_k: int = 200,
    attention_topk: int = 10,
    candidate_backend: str = "faiss_ivf",
    faiss_nlist: int = 4096,
    faiss_nprobe: int = 64,
    faiss_device: str = "auto",
    faiss_train_sample_size: int = 100000,
    faiss_query_batch_size: int | None = 8192,
    seed: int = 42,
    epsilon: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 100,
    stabilizer: float = 1e-8,
    device=None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build initial sparse UOT priors from single-modality candidate unions."""

    resolved_sections = _resolve_section_order(feature_dict, section_order)
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    device = torch.device(device or "cpu")
    for source_section, target_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        start_time = time.time()
        modalities_used = _valid_modalities(feature_dict, source_section, target_section, modalities)
        candidate_arrays = []
        search_metadata = {}
        for modality in modalities_used:
            result = build_faiss_candidates(
                feature_dict[source_section][modality],
                feature_dict[target_section][modality],
                candidate_k=initial_modality_candidate_k,
                backend=candidate_backend,
                nlist=faiss_nlist,
                nprobe=faiss_nprobe,
                faiss_device=faiss_device,
                train_sample_size=faiss_train_sample_size,
                query_batch_size=faiss_query_batch_size,
                seed=seed,
            )
            candidate_arrays.append(result["candidate_idx"])
            search_metadata[modality] = result["metadata"]

        candidate_idx, candidate_mask = _union_candidate_indices(candidate_arrays)
        costs = []
        for modality in modalities_used:
            costs.append(
                _compute_candidate_cosine_cost(
                    feature_dict[source_section][modality],
                    feature_dict[target_section][modality],
                    candidate_idx,
                    candidate_mask,
                    stabilizer=stabilizer,
                    device=device,
                )
            )
        candidate_cost = torch.stack(costs, dim=0).mean(dim=0)
        candidate_idx, candidate_mask, candidate_cost = _truncate_candidates_by_cost(
            candidate_idx,
            candidate_mask,
            candidate_cost,
            candidate_k=candidate_k,
        )
        sparse_start = time.time()
        sparse = sparse_unbalanced_sinkhorn_topk(
            candidate_idx=candidate_idx,
            candidate_mask=candidate_mask,
            candidate_cost=candidate_cost,
            n_source=candidate_idx.shape[0],
            n_target=int(_as_numpy_float32(feature_dict[target_section][modalities_used[0]]).shape[0]),
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            attention_topk=attention_topk,
            device=device,
        )
        sparse_time = time.time() - sparse_start
        n_target = int(_as_numpy_float32(feature_dict[target_section][modalities_used[0]]).shape[0])
        metadata = {
            "ot_prior_mode": "candidate_sparse",
            "candidate_source": "initial_modalities",
            "modalities_used": modalities_used,
            "modality_order": modalities_used,
            "cost_definition": "mean_m(1 - cosine(z_source_m, z_target_m)) over candidate union",
            "candidate_backend": candidate_backend,
            "initial_modality_candidate_k": int(initial_modality_candidate_k),
            "candidate_k": int(candidate_idx.shape[1]),
            "attention_topk": int(sparse["topk_idx"].shape[1]),
            "faiss_nlist": int(faiss_nlist),
            "faiss_nprobe": int(faiss_nprobe),
            "faiss_device": faiss_device,
            "faiss_train_sample_size": int(faiss_train_sample_size),
            "faiss_query_batch_size": (
                int(faiss_query_batch_size) if faiss_query_batch_size is not None else None
            ),
            "uot_epsilon": float(epsilon),
            "uot_tau_a": float(tau_a),
            "uot_tau_b": float(tau_b),
            "uot_max_iter": int(max_iter),
            "uot_stabilizer": float(stabilizer),
            "candidate_search_metadata": search_metadata,
            "candidate_search_time_sec": float(
                sum(item["candidate_search_time_sec"] for item in search_metadata.values())
            ),
            "sparse_uot_time_sec": float(sparse_time),
            "total_prior_time_sec": float(time.time() - start_time),
            **_candidate_qc(candidate_idx, candidate_mask, n_target),
        }
        priors[(source_section, target_section)] = {
            "P_dense": None,
            **sparse,
            "modalities_used": modalities_used,
            "metadata": metadata,
        }
    return priors


@torch.no_grad()
def update_candidate_sparse_uot_prior_from_embeddings(
    embedding_dict: Mapping[str, Any],
    section_order: Sequence[str] | None,
    candidate_k: int = 200,
    attention_topk: int = 10,
    candidate_backend: str = "faiss_ivf",
    faiss_nlist: int = 4096,
    faiss_nprobe: int = 64,
    faiss_device: str = "auto",
    faiss_train_sample_size: int = 100000,
    faiss_query_batch_size: int | None = 8192,
    seed: int = 42,
    epsilon: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 100,
    stabilizer: float = 1e-8,
    device=None,
    candidate_source: str = "fused",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build dynamic sparse UOT priors from current fused/final embeddings."""

    resolved_sections = _resolve_section_order(embedding_dict, section_order)
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    device = torch.device(device or "cpu")
    for source_section, target_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        start_time = time.time()
        result = build_faiss_candidates(
            embedding_dict[source_section],
            embedding_dict[target_section],
            candidate_k=candidate_k,
            backend=candidate_backend,
            nlist=faiss_nlist,
            nprobe=faiss_nprobe,
            faiss_device=faiss_device,
            train_sample_size=faiss_train_sample_size,
            query_batch_size=faiss_query_batch_size,
            seed=seed,
        )
        candidate_idx = torch.as_tensor(result["candidate_idx"], dtype=torch.long)
        candidate_mask = candidate_idx >= 0
        candidate_cost = _compute_candidate_cosine_cost(
            embedding_dict[source_section],
            embedding_dict[target_section],
            candidate_idx,
            candidate_mask,
            stabilizer=stabilizer,
            device=device,
        )
        sparse_start = time.time()
        n_target = int(_as_numpy_float32(embedding_dict[target_section]).shape[0])
        sparse = sparse_unbalanced_sinkhorn_topk(
            candidate_idx=candidate_idx,
            candidate_mask=candidate_mask,
            candidate_cost=candidate_cost,
            n_source=candidate_idx.shape[0],
            n_target=n_target,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            attention_topk=attention_topk,
            device=device,
        )
        sparse_time = time.time() - sparse_start
        modality_label = f"{candidate_source}_embedding"
        metadata = {
            "ot_prior_mode": "candidate_sparse",
            "candidate_source": candidate_source,
            "modalities_used": [modality_label],
            "modality_order": [modality_label],
            "cost_definition": f"1 - cosine({candidate_source}_source, {candidate_source}_target)",
            "candidate_backend": candidate_backend,
            "candidate_k": int(candidate_idx.shape[1]),
            "attention_topk": int(sparse["topk_idx"].shape[1]),
            "faiss_nlist": int(faiss_nlist),
            "faiss_nprobe": int(faiss_nprobe),
            "faiss_device": faiss_device,
            "faiss_train_sample_size": int(faiss_train_sample_size),
            "faiss_query_batch_size": (
                int(faiss_query_batch_size) if faiss_query_batch_size is not None else None
            ),
            "uot_epsilon": float(epsilon),
            "uot_tau_a": float(tau_a),
            "uot_tau_b": float(tau_b),
            "uot_max_iter": int(max_iter),
            "uot_stabilizer": float(stabilizer),
            "candidate_search_metadata": result["metadata"],
            "candidate_search_time_sec": float(result["metadata"]["candidate_search_time_sec"]),
            "sparse_uot_time_sec": float(sparse_time),
            "total_prior_time_sec": float(time.time() - start_time),
            **_candidate_qc(candidate_idx, candidate_mask, n_target),
        }
        priors[(source_section, target_section)] = {
            "P_dense": None,
            **sparse,
            "modalities_used": [modality_label],
            "metadata": metadata,
        }
    return priors
