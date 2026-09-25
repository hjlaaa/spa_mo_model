"""COSIE-style cross-view losses used by the first model stage."""

from __future__ import annotations

import itertools
import sys
from typing import Mapping

import torch


class _GaussianSpatialLaplacian(torch.autograd.Function):
    """Chunked Gaussian graph loss with O(ND) backward storage."""

    @staticmethod
    def forward(ctx, embedding, edge_index, edge_weight, edge_batch_size):
        source, target = edge_index
        weights = edge_weight.float()
        row_weight = torch.zeros(embedding.shape[0], device=embedding.device, dtype=torch.float32)
        for start in range(0, source.numel(), edge_batch_size):
            end = min(start + edge_batch_size, source.numel())
            src, dst = source[start:end], target[start:end]
            nonself = src != dst
            row_weight.index_add_(0, src[nonself], weights[start:end][nonself])
        if bool((row_weight <= 0).any()):
            raise ValueError("Every spot needs at least one non-self spatial edge.")
        loss = torch.zeros((), device=embedding.device, dtype=torch.float32)
        for start in range(0, source.numel(), edge_batch_size):
            end = min(start + edge_batch_size, source.numel())
            src, dst = source[start:end], target[start:end]
            nonself = src != dst
            src, dst = src[nonself], dst[nonself]
            if src.numel() == 0:
                continue
            strength = weights[start:end][nonself] / row_weight[src]
            difference = embedding[src].float() - embedding[dst].float()
            loss += (strength * difference.square().mean(dim=1)).sum()
        ctx.save_for_backward(embedding, edge_index, edge_weight, row_weight)
        ctx.edge_batch_size = edge_batch_size
        return loss / embedding.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        embedding, edge_index, edge_weight, row_weight = ctx.saved_tensors
        source, target = edge_index
        weights = edge_weight.float()
        gradient = torch.zeros(embedding.shape, device=embedding.device, dtype=torch.float32)
        factor = 2.0 / (embedding.shape[0] * embedding.shape[1])
        for start in range(0, source.numel(), ctx.edge_batch_size):
            end = min(start + ctx.edge_batch_size, source.numel())
            src, dst = source[start:end], target[start:end]
            nonself = src != dst
            src, dst = src[nonself], dst[nonself]
            if src.numel() == 0:
                continue
            strength = weights[start:end][nonself] / row_weight[src]
            difference = embedding[src].float() - embedding[dst].float()
            contribution = factor * strength.unsqueeze(1) * difference
            gradient.index_add_(0, src, contribution)
            gradient.index_add_(0, dst, -contribution)
        return (gradient * grad_output.float()).to(embedding.dtype), None, None, None


def gaussian_spatial_laplacian_loss(
    embedding: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    edge_batch_size: int = 200000,
) -> torch.Tensor:
    """Gaussian-weighted non-self smoothness, averaged per spot and dimension.

    Reuses the spatial GraphSAGE graph and renormalizes after removing its
    self-loops. Forward/backward chunk edges for full-spot CRC/SPATCH runs.
    """
    if embedding.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected embedding [N, D] and edge_index [2, E].")
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
        raise ValueError("edge_weight must have one value per edge.")
    if edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be positive.")
    return _GaussianSpatialLaplacian.apply(embedding, edge_index, edge_weight, edge_batch_size)


# Adapted from /home/hujinlan/cosie/COSIE/loss.py::compute_joint
def compute_joint(view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
    """Compute COSIE's dimension-level joint-dependency matrix.

    This follows COSIE's original implementation. It does not build an
    ``[N, N]`` sample similarity matrix and does not define spot-level positive
    or negative pairs.
    """

    bn, k = view1.size()
    assert view2.size(0) == bn and view2.size(1) == k

    p_i_j = torch.matmul(view1.transpose(0, 1), view2)
    p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j


# Adapted from /home/hujinlan/cosie/COSIE/loss.py::crossview_contrastive_Loss
def crossview_contrastive_Loss(
    view1: torch.Tensor,
    view2: torch.Tensor,
    gamma: float = 9.0,
    EPS: float = sys.float_info.epsilon,
) -> torch.Tensor:
    """Compute COSIE's cross-view contrastive loss.

    This is COSIE's dimension-level cross-view objective, not InfoNCE or
    CLIP-style sample-level contrastive learning. It does not use temperature,
    same-spot positives, different-spot negatives, or an ``[N, N]`` similarity
    matrix.
    """

    view1 = view1.float()
    view2 = view2.float()
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    eps_i_j = torch.tensor([EPS], device=p_i_j.device)
    eps_j = torch.tensor([EPS], device=p_j.device)
    eps_i = torch.tensor([EPS], device=p_i.device)
    p_i_j = torch.where(p_i_j < EPS, eps_i_j, p_i_j)
    p_j = torch.where(p_j < EPS, eps_j, p_j)
    p_i = torch.where(p_i < EPS, eps_i, p_i)

    loss = -p_i_j * (
        torch.log(p_i_j)
        - (gamma + 1) * torch.log(p_j)
        - (gamma + 1) * torch.log(p_i)
    )

    loss = loss.sum()

    return loss


def compute_pairwise_cosie_crossview_loss(
    latent_dict_for_one_section: Mapping[str, torch.Tensor],
    gamma: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum COSIE cross-view losses over all observed modality pairs.

    Parameters
    ----------
    latent_dict_for_one_section
        Mapping from modality name to latent tensor with shape ``[N, 128]``.
    gamma
        COSIE cross-view entropy regularization coefficient.

    Returns
    -------
    total_loss, loss_detail_dict
        ``total_loss`` is a scalar tensor. ``loss_detail_dict`` maps names such
        as ``"HE__RNA"`` to the corresponding scalar tensor.
    """

    present = [
        (modality, latent)
        for modality, latent in latent_dict_for_one_section.items()
        if latent is not None
    ]
    if not present:
        return torch.tensor(0.0), {}

    first_latent = present[0][1]
    total_loss = torch.zeros((), device=first_latent.device, dtype=first_latent.dtype)
    loss_detail_dict: dict[str, torch.Tensor] = {}

    for (mod1, latent1), (mod2, latent2) in itertools.combinations(present, 2):
        if latent1.shape != latent2.shape:
            raise ValueError(
                "COSIE cross-view loss requires paired latent tensors with the "
                f"same shape; got {mod1}={tuple(latent1.shape)} and "
                f"{mod2}={tuple(latent2.shape)}."
            )
        pair_loss = crossview_contrastive_Loss(latent1, latent2, gamma=gamma)
        loss_detail_dict[f"{mod1}__{mod2}"] = pair_loss
        total_loss = total_loss + pair_loss

    return total_loss, loss_detail_dict
