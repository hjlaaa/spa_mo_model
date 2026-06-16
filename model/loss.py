"""COSIE-style cross-view losses used by the first model stage."""

from __future__ import annotations

import itertools
import sys
from typing import Mapping

import torch
import torch.nn.functional as F


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


def compute_reconstruction_loss(
    recon_dict_for_one_section: Mapping[str, torch.Tensor],
    target_feature_dict_for_one_section: Mapping[str, torch.Tensor],
    lambda_by_modality: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MSE reconstruction loss for preprocessed modality embeddings.

    The reconstruction target is the preprocessed feature tensor consumed by
    the model, such as ``feature_dict[section]["RNA"]``. It is not the raw HE
    image, raw RNA count matrix, or raw protein/metabolite matrix.
    """

    if not recon_dict_for_one_section:
        return torch.tensor(0.0), {}

    first_recon = next(iter(recon_dict_for_one_section.values()))
    total_loss = torch.zeros((), device=first_recon.device, dtype=first_recon.dtype)
    detail: dict[str, torch.Tensor] = {}
    weights = lambda_by_modality or {}

    for modality, recon in recon_dict_for_one_section.items():
        if modality not in target_feature_dict_for_one_section:
            raise KeyError(f"Missing reconstruction target for modality {modality}.")
        target = target_feature_dict_for_one_section[modality]
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=recon.dtype, device=recon.device)
        else:
            target = target.to(device=recon.device, dtype=recon.dtype)
        if recon.shape != target.shape:
            raise ValueError(
                f"Reconstruction shape mismatch for {modality}: "
                f"recon={tuple(recon.shape)}, target={tuple(target.shape)}."
            )
        loss = F.mse_loss(recon, target)
        detail[modality] = loss
        total_loss = total_loss + float(weights.get(modality, 1.0)) * loss

    return total_loss, detail
