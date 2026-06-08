"""Cross-stage UOT matching utilities for model stage V2.

The functions in this file follow the COSIE naming style for linkage logic,
but they are project-specific UOT priors. They do not implement COSIE triplet
linkage and they do not participate in backpropagation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .utils import cosine_cost_matrix, l2_normalize


def _as_float_tensor(x: Any, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _resolve_section_order(mapping: Mapping[str, Any], section_order: Sequence[str] | None):
    if section_order is None:
        return sorted(mapping.keys())
    missing = [section for section in section_order if section not in mapping]
    if missing:
        raise KeyError(f"section_order contains sections missing from input: {missing}")
    return list(section_order)


@torch.no_grad()
def unbalanced_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    epsilon: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    check_every: int = 10,
    delta: float = 1e-8,
) -> torch.Tensor:
    """Compute an unbalanced entropic OT coupling with Sinkhorn updates."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive for unbalanced Sinkhorn.")

    cost = _as_float_tensor(cost)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D, got shape {tuple(cost.shape)}.")

    n_source, n_target = cost.shape
    device = cost.device
    dtype = cost.dtype
    if a is None:
        a = torch.full((n_source,), 1.0 / n_source, device=device, dtype=dtype)
    else:
        a = _as_float_tensor(a, device=device)
    if b is None:
        b = torch.full((n_target,), 1.0 / n_target, device=device, dtype=dtype)
    else:
        b = _as_float_tensor(b, device=device)

    kernel = torch.exp(-cost / epsilon).clamp_min(delta)
    rho_a = tau_a / (tau_a + epsilon)
    rho_b = tau_b / (tau_b + epsilon)
    u = torch.ones(n_source, device=device, dtype=dtype)
    v = torch.ones(n_target, device=device, dtype=dtype)

    check_every = max(int(check_every), 1)
    for iteration in range(int(max_iter)):
        prev_u = u
        prev_v = v
        u = (a / (kernel @ v + delta)).pow(rho_a)
        v = (b / (kernel.t() @ u + delta)).pow(rho_b)
        if (iteration + 1) % check_every == 0:
            diff = max(
                torch.max(torch.abs(u - prev_u)).item(),
                torch.max(torch.abs(v - prev_v)).item(),
            )
            if diff < tol:
                break

    coupling = u[:, None] * kernel * v[None, :]
    coupling = torch.nan_to_num(coupling, nan=0.0, posinf=0.0, neginf=0.0)
    return coupling


def normalize_coupling_total_mass(P: torch.Tensor, delta: float = 1e-8) -> torch.Tensor:
    """Normalize a coupling matrix by its total mass."""

    return P / (P.sum() + delta)


def sparsify_coupling_topk(
    P: torch.Tensor,
    topk: int = 10,
    delta: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Keep top-k targets per source row and compute row confidence."""

    if P.ndim != 2:
        raise ValueError(f"P must be 2D, got shape {tuple(P.shape)}.")
    n_source, n_target = P.shape
    effective_topk = min(int(topk), n_target)
    values, indices = torch.topk(P, k=effective_topk, dim=1)
    row_topk_mass = values.sum(dim=1, keepdim=True)
    topk_weight = values / (row_topk_mass + delta)
    row_mass = P.sum(dim=1)
    uniform_source_mass = 1.0 / max(n_source, 1)
    confidence = torch.clamp(row_mass / (uniform_source_mass + delta), max=1.0)
    return {
        "topk_idx": indices,
        "topk_weight": topk_weight,
        "confidence": confidence,
        "row_mass": row_mass,
    }


def _compute_global_zscore_stats(
    feature_dict: Mapping[str, Mapping[str, Any]],
    section_order: Sequence[str],
    modality: str,
    delta: float,
):
    tensors = []
    input_dim = None
    for section in section_order:
        features = feature_dict[section].get(modality)
        if features is None:
            continue
        tensor = _as_float_tensor(features)
        if input_dim is None:
            input_dim = tensor.shape[1]
        elif tensor.shape[1] != input_dim:
            raise ValueError(
                f"Modality {modality} has inconsistent feature dimensions across sections."
            )
        tensors.append(tensor)
    if not tensors:
        return None
    stacked = torch.cat(tensors, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, unbiased=False, keepdim=True).clamp_min(delta)
    return mean, std


def _zscore_and_normalize(features: Any, mean: torch.Tensor, std: torch.Tensor, delta: float):
    tensor = _as_float_tensor(features, device=mean.device)
    return l2_normalize((tensor - mean) / std, dim=-1, eps=delta)


@torch.no_grad()
def compute_initial_multimodal_uot_prior(
    feature_dict: Mapping[str, Mapping[str, Any]],
    section_order: Sequence[str] | None,
    modalities: Iterable[str],
    epsilon_init: float = 0.08,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    topk: int = 10,
    delta: float = 1e-8,
    check_every: int = 10,
    clip_cost_min: float = 0.0,
    clip_cost_max: float = 2.0,
    keep_dense: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute adjacent-section multimodal UOT priors from preprocessed features."""

    resolved_sections = _resolve_section_order(feature_dict, section_order)
    modality_list = list(modalities)
    stats = {
        modality: _compute_global_zscore_stats(
            feature_dict,
            resolved_sections,
            modality,
            delta=delta,
        )
        for modality in modality_list
    }

    priors: dict[tuple[str, str], dict[str, Any]] = {}
    for source_section, target_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        couplings = []
        modalities_used = []
        for modality in modality_list:
            if modality not in feature_dict[source_section] or modality not in feature_dict[target_section]:
                continue
            if stats[modality] is None:
                continue
            mean, std = stats[modality]
            source = _zscore_and_normalize(
                feature_dict[source_section][modality],
                mean,
                std,
                delta=delta,
            )
            target = _zscore_and_normalize(
                feature_dict[target_section][modality],
                mean,
                std,
                delta=delta,
            )
            cost = cosine_cost_matrix(
                source,
                target,
                eps=delta,
                clip_min=clip_cost_min,
                clip_max=clip_cost_max,
            )
            P = unbalanced_sinkhorn(
                cost,
                epsilon=epsilon_init,
                tau_a=tau_a,
                tau_b=tau_b,
                max_iter=max_iter,
                tol=tol,
                check_every=check_every,
                delta=delta,
            )
            couplings.append(normalize_coupling_total_mass(P, delta=delta))
            modalities_used.append(modality)

        if not couplings:
            raise ValueError(
                f"No shared modalities available to compute UOT prior for "
                f"{source_section}->{target_section}."
            )

        P_mean = normalize_coupling_total_mass(torch.stack(couplings, dim=0).mean(dim=0), delta=delta)
        sparse = sparsify_coupling_topk(P_mean, topk=topk, delta=delta)
        priors[(source_section, target_section)] = {
            "P_dense": P_mean if keep_dense else None,
            "topk_idx": sparse["topk_idx"],
            "topk_weight": sparse["topk_weight"],
            "confidence": sparse["confidence"],
            "row_mass": sparse["row_mass"],
            "modalities_used": modalities_used,
        }

    return priors


@torch.no_grad()
def update_uot_prior_from_embeddings(
    final_embedding_dict: Mapping[str, torch.Tensor],
    section_order: Sequence[str] | None,
    epsilon_update: float = 0.05,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    topk: int = 10,
    delta: float = 1e-8,
    check_every: int = 10,
    clip_cost_min: float = 0.0,
    clip_cost_max: float = 2.0,
    keep_dense: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Update adjacent-section UOT priors from detached final embeddings."""

    resolved_sections = _resolve_section_order(final_embedding_dict, section_order)
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    for source_section, target_section in zip(resolved_sections[:-1], resolved_sections[1:]):
        source = l2_normalize(
            _as_float_tensor(final_embedding_dict[source_section]),
            dim=-1,
            eps=delta,
        )
        target = l2_normalize(
            _as_float_tensor(final_embedding_dict[target_section]),
            dim=-1,
            eps=delta,
        )
        cost = cosine_cost_matrix(
            source,
            target,
            eps=delta,
            clip_min=clip_cost_min,
            clip_max=clip_cost_max,
        )
        P = unbalanced_sinkhorn(
            cost,
            epsilon=epsilon_update,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            tol=tol,
            check_every=check_every,
            delta=delta,
        )
        P = normalize_coupling_total_mass(P, delta=delta)
        sparse = sparsify_coupling_topk(P, topk=topk, delta=delta)
        priors[(source_section, target_section)] = {
            "P_dense": P if keep_dense else None,
            "topk_idx": sparse["topk_idx"],
            "topk_weight": sparse["topk_weight"],
            "confidence": sparse["confidence"],
            "row_mass": sparse["row_mass"],
            "modalities_used": ["final_embedding"],
        }
    return priors
