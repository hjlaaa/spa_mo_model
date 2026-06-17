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


def _candidate_edges_from_search(
    candidate_idx: np.ndarray,
    reverse_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(candidate_idx, dtype=torch.long)
    mask = idx >= 0
    query_ids = torch.arange(idx.shape[0], dtype=torch.long).unsqueeze(1).expand_as(idx)
    if reverse_query:
        edge_src = idx[mask]
        edge_tgt = query_ids[mask]
    else:
        edge_src = query_ids[mask]
        edge_tgt = idx[mask]
    return edge_src.long(), edge_tgt.long()


def _deduplicate_edges(
    edge_src: torch.Tensor,
    edge_tgt: torch.Tensor,
    n_target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_src.numel() == 0:
        raise ValueError("No candidate edges available after bidirectional candidate search.")
    keys = edge_src.long() * int(n_target) + edge_tgt.long()
    unique_keys = torch.unique(keys)
    return (unique_keys // int(n_target)).long(), (unique_keys % int(n_target)).long()


def _compute_edge_cosine_cost(
    source_embeddings,
    target_embeddings,
    edge_src: torch.Tensor,
    edge_tgt: torch.Tensor,
    stabilizer: float,
    device,
    chunk_size: int = 200000,
) -> torch.Tensor:
    source = _l2_normalize_tensor(_as_float_tensor(source_embeddings, device=device), stabilizer)
    target = _l2_normalize_tensor(_as_float_tensor(target_embeddings, device=device), stabilizer)
    edge_src_device = edge_src.to(device=device)
    edge_tgt_device = edge_tgt.to(device=device)
    costs = torch.empty(edge_src_device.shape[0], dtype=torch.float32, device=device)
    for start in range(0, int(edge_src_device.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(edge_src_device.shape[0]))
        src = edge_src_device[start:end]
        tgt = edge_tgt_device[start:end]
        sim = (source[src] * target[tgt]).sum(dim=-1)
        costs[start:end] = (1.0 - sim).clamp(0.0, 2.0)
    return costs


def _prune_edges_bidirectional(
    edge_src: torch.Tensor,
    edge_tgt: torch.Tensor,
    edge_cost: torch.Tensor,
    n_source: int,
    n_target: int,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_src = edge_src.cpu().long()
    edge_tgt = edge_tgt.cpu().long()
    edge_cost_cpu = edge_cost.detach().cpu().float()
    keep = torch.zeros(edge_src.shape[0], dtype=torch.bool)
    candidate_k = int(candidate_k)

    for group_ids, n_groups in ((edge_src, int(n_source)), (edge_tgt, int(n_target))):
        order = torch.argsort(group_ids)
        sorted_groups = group_ids[order]
        counts = torch.bincount(sorted_groups, minlength=n_groups)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        for group in range(n_groups):
            start = int(offsets[group])
            end = int(offsets[group + 1])
            if end <= start:
                continue
            group_order = order[start:end]
            k = min(candidate_k, int(group_order.numel()))
            _, local_pos = torch.topk(-edge_cost_cpu[group_order], k=k, dim=0)
            keep[group_order[local_pos]] = True

    if not bool(keep.any()):
        raise ValueError("Bidirectional pruning removed all candidate edges.")
    kept = keep.nonzero(as_tuple=False).flatten()
    return edge_src[kept], edge_tgt[kept], edge_cost_cpu[kept]


def _edges_to_sparse_topk(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mass: torch.Tensor,
    n_source: int,
    n_target: int,
    attention_topk: int,
    stabilizer: float,
    device,
) -> dict[str, torch.Tensor]:
    device = torch.device(device or "cpu")
    source_ids = source_ids.to(device=device, dtype=torch.long)
    target_ids = target_ids.to(device=device, dtype=torch.long)
    mass = mass.to(device=device, dtype=torch.float32)
    n_source = int(n_source)
    n_target = int(n_target)
    attention_topk = int(min(attention_topk, max(n_target, 1)))

    topk_idx = torch.zeros((n_source, attention_topk), dtype=torch.long, device=device)
    topk_raw = torch.zeros((n_source, attention_topk), dtype=torch.float32, device=device)
    row_mass = torch.zeros(n_source, dtype=torch.float32, device=device)
    row_mass.scatter_add_(0, source_ids, mass)

    if source_ids.numel() > 0:
        order = torch.argsort(source_ids)
        sorted_source = source_ids[order]
        counts = torch.bincount(sorted_source, minlength=n_source)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        for row in range(n_source):
            start = int(offsets[row].item())
            end = int(offsets[row + 1].item())
            if end <= start:
                continue
            group_order = order[start:end]
            k = min(attention_topk, int(group_order.numel()))
            values, local_pos = torch.topk(mass[group_order], k=k, dim=0)
            selected = group_order[local_pos]
            topk_idx[row, :k] = target_ids[selected]
            topk_raw[row, :k] = values
            if k < attention_topk:
                topk_idx[row, k:] = topk_idx[row, 0]

    raw_topk_mass = topk_raw.sum(dim=1)
    topk_weight = topk_raw / (raw_topk_mass[:, None] + float(stabilizer))
    topk_coverage = torch.clamp(raw_topk_mass / (row_mass + float(stabilizer)), min=0.0, max=1.0)
    tail_mass = torch.clamp(row_mass - raw_topk_mass, min=0.0)
    target_hit_count = torch.zeros(n_target, dtype=torch.long, device=device)
    if target_ids.numel() > 0:
        target_hit_count.scatter_add_(0, target_ids, torch.ones_like(target_ids, dtype=torch.long))
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


@torch.no_grad()
def sparse_unbalanced_sinkhorn_bidirectional_topk(
    edge_src: torch.Tensor | np.ndarray,
    edge_tgt: torch.Tensor | np.ndarray,
    edge_cost: torch.Tensor | np.ndarray,
    n_source: int,
    n_target: int,
    epsilon: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 100,
    stabilizer: float = 1e-8,
    attention_topk: int = 10,
    device=None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run one sparse UOT on A-B edges and derive both directional priors."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    device = torch.device(device or "cpu")
    edge_src = torch.as_tensor(edge_src, dtype=torch.long, device=device)
    edge_tgt = torch.as_tensor(edge_tgt, dtype=torch.long, device=device)
    edge_cost = torch.as_tensor(edge_cost, dtype=torch.float32, device=device)
    if not (edge_src.shape == edge_tgt.shape == edge_cost.shape):
        raise ValueError("edge_src, edge_tgt, and edge_cost must have identical 1D shape.")
    if edge_src.numel() == 0:
        raise ValueError("No candidate edges available for sparse UOT.")

    n_source = int(n_source)
    n_target = int(n_target)
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

    left_to_right = _edges_to_sparse_topk(
        source_ids=edge_src,
        target_ids=edge_tgt,
        mass=p_edge,
        n_source=n_source,
        n_target=n_target,
        attention_topk=attention_topk,
        stabilizer=stabilizer,
        device=device,
    )
    right_to_left = _edges_to_sparse_topk(
        source_ids=edge_tgt,
        target_ids=edge_src,
        mass=p_edge,
        n_source=n_target,
        n_target=n_source,
        attention_topk=attention_topk,
        stabilizer=stabilizer,
        device=device,
    )
    return {
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "edge_count": int(edge_src.numel()),
    }


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


def _bidirectional_support_qc(edge_src: torch.Tensor, edge_tgt: torch.Tensor, n_source: int, n_target: int) -> dict[str, Any]:
    src_count = torch.bincount(edge_src.cpu().long(), minlength=int(n_source))
    tgt_count = torch.bincount(edge_tgt.cpu().long(), minlength=int(n_target))
    return {
        "bidirectional_support_edge_count": int(edge_src.numel()),
        "source_covered": int((src_count > 0).sum().item()),
        "target_covered": int((tgt_count > 0).sum().item()),
        "source_coverage": float((src_count > 0).float().mean().item()) if src_count.numel() else 0.0,
        "target_coverage": float((tgt_count > 0).float().mean().item()) if tgt_count.numel() else 0.0,
        "source_degree_min": int(src_count.min().item()) if src_count.numel() else 0,
        "source_degree_max": int(src_count.max().item()) if src_count.numel() else 0,
        "target_degree_min": int(tgt_count.min().item()) if tgt_count.numel() else 0,
        "target_degree_max": int(tgt_count.max().item()) if tgt_count.numel() else 0,
    }


def _make_direction_prior(
    sparse_direction: Mapping[str, torch.Tensor],
    modalities_used: list[str],
    source_section: str,
    target_section: str,
    left_section: str,
    right_section: str,
    metadata: Mapping[str, Any],
    directional_extraction: str,
) -> dict[str, Any]:
    direction = f"{source_section}->{target_section}"
    return {
        "P_dense": None,
        **sparse_direction,
        "modalities_used": modalities_used,
        "direction": direction,
        "pair": (left_section, right_section),
        "metadata": {
            **dict(metadata),
            "direction": direction,
            "source_section": source_section,
            "target_section": target_section,
            "direction_meaning": "source receives information from target",
            "directional_extraction": directional_extraction,
            "pair": [left_section, right_section],
        },
    }


@torch.no_grad()
def compute_initial_bidirectional_candidate_sparse_uot_prior(
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
    """Build bidirectional sparse UOT priors from one shared A-B coupling."""

    resolved_sections = _resolve_section_order(feature_dict, section_order)
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    device = torch.device(device or "cpu")
    for left_section, right_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        start_time = time.time()
        modalities_used = _valid_modalities(feature_dict, left_section, right_section, modalities)
        edge_src_list: list[torch.Tensor] = []
        edge_tgt_list: list[torch.Tensor] = []
        search_metadata: dict[str, Any] = {}
        for modality in modalities_used:
            forward = build_faiss_candidates(
                feature_dict[left_section][modality],
                feature_dict[right_section][modality],
                candidate_k=initial_modality_candidate_k,
                backend=candidate_backend,
                nlist=faiss_nlist,
                nprobe=faiss_nprobe,
                faiss_device=faiss_device,
                train_sample_size=faiss_train_sample_size,
                query_batch_size=faiss_query_batch_size,
                seed=seed,
            )
            reverse = build_faiss_candidates(
                feature_dict[right_section][modality],
                feature_dict[left_section][modality],
                candidate_k=initial_modality_candidate_k,
                backend=candidate_backend,
                nlist=faiss_nlist,
                nprobe=faiss_nprobe,
                faiss_device=faiss_device,
                train_sample_size=faiss_train_sample_size,
                query_batch_size=faiss_query_batch_size,
                seed=seed,
            )
            f_src, f_tgt = _candidate_edges_from_search(forward["candidate_idx"], reverse_query=False)
            r_src, r_tgt = _candidate_edges_from_search(reverse["candidate_idx"], reverse_query=True)
            edge_src_list.extend([f_src, r_src])
            edge_tgt_list.extend([f_tgt, r_tgt])
            search_metadata[f"{modality}:left_query_right"] = forward["metadata"]
            search_metadata[f"{modality}:right_query_left"] = reverse["metadata"]

        n_left = int(_as_numpy_float32(feature_dict[left_section][modalities_used[0]]).shape[0])
        n_right = int(_as_numpy_float32(feature_dict[right_section][modalities_used[0]]).shape[0])
        edge_src, edge_tgt = _deduplicate_edges(
            torch.cat(edge_src_list),
            torch.cat(edge_tgt_list),
            n_target=n_right,
        )
        costs = []
        for modality in modalities_used:
            costs.append(
                _compute_edge_cosine_cost(
                    feature_dict[left_section][modality],
                    feature_dict[right_section][modality],
                    edge_src,
                    edge_tgt,
                    stabilizer=stabilizer,
                    device=device,
                )
            )
        edge_cost = torch.stack(costs, dim=0).mean(dim=0)
        edge_src, edge_tgt, edge_cost = _prune_edges_bidirectional(
            edge_src=edge_src,
            edge_tgt=edge_tgt,
            edge_cost=edge_cost,
            n_source=n_left,
            n_target=n_right,
            candidate_k=candidate_k,
        )
        sparse_start = time.time()
        sparse = sparse_unbalanced_sinkhorn_bidirectional_topk(
            edge_src=edge_src,
            edge_tgt=edge_tgt,
            edge_cost=edge_cost,
            n_source=n_left,
            n_target=n_right,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            attention_topk=attention_topk,
            device=device,
        )
        sparse_time = time.time() - sparse_start
        metadata = {
            "ot_prior_mode": "candidate_sparse",
            "bidirectional_ot_attention": True,
            "candidate_source": "initial_modalities",
            "modalities_used": modalities_used,
            "modality_order": modalities_used,
            "cost_definition": "mean_m(1 - cosine(z_left_m, z_right_m)) over bidirectional candidate union",
            "candidate_backend": candidate_backend,
            "initial_modality_candidate_k": int(initial_modality_candidate_k),
            "candidate_k": int(candidate_k),
            "attention_topk": int(attention_topk),
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
            **_bidirectional_support_qc(edge_src, edge_tgt, n_left, n_right),
        }
        priors[(left_section, right_section)] = _make_direction_prior(
            sparse["left_to_right"],
            modalities_used,
            source_section=left_section,
            target_section=right_section,
            left_section=left_section,
            right_section=right_section,
            metadata=metadata,
            directional_extraction="row-wise top-k over sparse coupling rows",
        )
        priors[(right_section, left_section)] = _make_direction_prior(
            sparse["right_to_left"],
            modalities_used,
            source_section=right_section,
            target_section=left_section,
            left_section=left_section,
            right_section=right_section,
            metadata=metadata,
            directional_extraction="column-wise top-k over sparse coupling columns",
        )
    return priors


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


@torch.no_grad()
def update_bidirectional_candidate_sparse_uot_prior_from_embeddings(
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
    candidate_source: str = "final",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build dynamic bidirectional sparse UOT priors from current embeddings."""

    resolved_sections = _resolve_section_order(embedding_dict, section_order)
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    device = torch.device(device or "cpu")
    modality_label = f"{candidate_source}_embedding"
    for left_section, right_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        start_time = time.time()
        forward = build_faiss_candidates(
            embedding_dict[left_section],
            embedding_dict[right_section],
            candidate_k=candidate_k,
            backend=candidate_backend,
            nlist=faiss_nlist,
            nprobe=faiss_nprobe,
            faiss_device=faiss_device,
            train_sample_size=faiss_train_sample_size,
            query_batch_size=faiss_query_batch_size,
            seed=seed,
        )
        reverse = build_faiss_candidates(
            embedding_dict[right_section],
            embedding_dict[left_section],
            candidate_k=candidate_k,
            backend=candidate_backend,
            nlist=faiss_nlist,
            nprobe=faiss_nprobe,
            faiss_device=faiss_device,
            train_sample_size=faiss_train_sample_size,
            query_batch_size=faiss_query_batch_size,
            seed=seed,
        )
        f_src, f_tgt = _candidate_edges_from_search(forward["candidate_idx"], reverse_query=False)
        r_src, r_tgt = _candidate_edges_from_search(reverse["candidate_idx"], reverse_query=True)
        n_left = int(_as_numpy_float32(embedding_dict[left_section]).shape[0])
        n_right = int(_as_numpy_float32(embedding_dict[right_section]).shape[0])
        edge_src, edge_tgt = _deduplicate_edges(
            torch.cat([f_src, r_src]),
            torch.cat([f_tgt, r_tgt]),
            n_target=n_right,
        )
        edge_cost = _compute_edge_cosine_cost(
            embedding_dict[left_section],
            embedding_dict[right_section],
            edge_src,
            edge_tgt,
            stabilizer=stabilizer,
            device=device,
        )
        edge_src, edge_tgt, edge_cost = _prune_edges_bidirectional(
            edge_src=edge_src,
            edge_tgt=edge_tgt,
            edge_cost=edge_cost,
            n_source=n_left,
            n_target=n_right,
            candidate_k=candidate_k,
        )
        sparse_start = time.time()
        sparse = sparse_unbalanced_sinkhorn_bidirectional_topk(
            edge_src=edge_src,
            edge_tgt=edge_tgt,
            edge_cost=edge_cost,
            n_source=n_left,
            n_target=n_right,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            attention_topk=attention_topk,
            device=device,
        )
        sparse_time = time.time() - sparse_start
        metadata = {
            "ot_prior_mode": "candidate_sparse",
            "bidirectional_ot_attention": True,
            "candidate_source": candidate_source,
            "modalities_used": [modality_label],
            "modality_order": [modality_label],
            "cost_definition": f"1 - cosine({candidate_source}_left, {candidate_source}_right)",
            "candidate_backend": candidate_backend,
            "candidate_k": int(candidate_k),
            "attention_topk": int(attention_topk),
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
            "candidate_search_metadata": {
                f"{candidate_source}:left_query_right": forward["metadata"],
                f"{candidate_source}:right_query_left": reverse["metadata"],
            },
            "candidate_search_time_sec": float(
                forward["metadata"]["candidate_search_time_sec"]
                + reverse["metadata"]["candidate_search_time_sec"]
            ),
            "sparse_uot_time_sec": float(sparse_time),
            "total_prior_time_sec": float(time.time() - start_time),
            **_bidirectional_support_qc(edge_src, edge_tgt, n_left, n_right),
        }
        priors[(left_section, right_section)] = _make_direction_prior(
            sparse["left_to_right"],
            [modality_label],
            source_section=left_section,
            target_section=right_section,
            left_section=left_section,
            right_section=right_section,
            metadata=metadata,
            directional_extraction="row-wise top-k over sparse coupling rows",
        )
        priors[(right_section, left_section)] = _make_direction_prior(
            sparse["right_to_left"],
            [modality_label],
            source_section=right_section,
            target_section=left_section,
            left_section=left_section,
            right_section=right_section,
            metadata=metadata,
            directional_extraction="column-wise top-k over sparse coupling columns",
        )
    return priors
