"""Cross-view losses used by the first multimodal model stage."""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

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


def compute_corrected_cosie_joint(
    view1: torch.Tensor,
    view2: torch.Tensor,
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a valid non-negative COSIE dimension-level joint distribution.

    Unlike :func:`compute_joint`, this corrected variant first converts every
    spot's latent dimensions into a probability distribution.  The resulting
    dimension-by-dimension co-occurrence matrix is clamped and *then*
    renormalized, so its marginals are well-defined even when encoder latents
    contain negative values.  The operation remains dimension-level and never
    constructs an ``[N, N]`` spot similarity matrix.
    """

    if not isinstance(view1, torch.Tensor) or not isinstance(view2, torch.Tensor):
        raise TypeError("Corrected COSIE views must be torch.Tensor instances.")
    if view1.ndim != 2 or view2.ndim != 2:
        raise ValueError(
            "Corrected COSIE views must have shape [spots, dimensions]; "
            f"got {tuple(view1.shape)} and {tuple(view2.shape)}."
        )
    if view1.shape != view2.shape:
        raise ValueError(
            "Corrected COSIE requires paired views with identical shapes; "
            f"got {tuple(view1.shape)} and {tuple(view2.shape)}."
        )
    if view1.shape[0] < 1 or view1.shape[1] < 1:
        raise ValueError("Corrected COSIE requires at least one spot and dimension.")
    if view1.device != view2.device:
        raise ValueError("Corrected COSIE requires both views on the same device.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps}.")

    # Keep probability construction and logarithms in fp32 under AMP.  Softmax
    # provides the required non-negativity without changing the encoder/fusion
    # architecture or adding a projection head.
    with torch.autocast(device_type=view1.device.type, enabled=False):
        probability1 = torch.softmax(view1.float() / float(temperature), dim=1)
        probability2 = torch.softmax(view2.float() / float(temperature), dim=1)
        joint = torch.matmul(probability1.transpose(0, 1), probability2)
        joint = 0.5 * (joint + joint.transpose(0, 1))
        joint = joint.clamp_min(float(eps))
        joint = joint / joint.sum().clamp_min(float(eps))
    return joint


def corrected_cosie_dimension_loss(
    view1: torch.Tensor,
    view2: torch.Tensor,
    gamma: float = 5.0,
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the numerically corrected COSIE dimension-level objective."""

    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and non-negative, got {gamma}.")
    joint = compute_corrected_cosie_joint(
        view1=view1,
        view2=view2,
        temperature=temperature,
        eps=eps,
    )
    marginal_i = joint.sum(dim=1, keepdim=True).clamp_min(float(eps))
    marginal_j = joint.sum(dim=0, keepdim=True).clamp_min(float(eps))
    loss = -joint * (
        torch.log(joint.clamp_min(float(eps)))
        - (float(gamma) + 1.0) * torch.log(marginal_i)
        - (float(gamma) + 1.0) * torch.log(marginal_j)
    )
    diagnostics = {
        "corrected_joint_min": joint.detach().min(),
        "corrected_joint_max": joint.detach().max(),
        "corrected_joint_sum": joint.detach().sum(),
    }
    return loss.sum(), diagnostics


def compute_pairwise_corrected_cosie_crossview_loss(
    latent_dict_for_one_section: Mapping[str, torch.Tensor],
    gamma: float = 5.0,
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean corrected COSIE loss over all observed modality pairs."""

    present = [
        (modality, latent)
        for modality, latent in latent_dict_for_one_section.items()
        if latent is not None
    ]
    if len(present) < 2:
        raise ValueError("Corrected COSIE requires at least two observed modalities.")

    pair_losses: list[torch.Tensor] = []
    joint_mins: list[torch.Tensor] = []
    joint_maxes: list[torch.Tensor] = []
    joint_sums: list[torch.Tensor] = []
    detail: dict[str, torch.Tensor] = {}
    for (modality1, latent1), (modality2, latent2) in itertools.combinations(
        present, 2
    ):
        pair_loss, pair_diagnostics = corrected_cosie_dimension_loss(
            latent1,
            latent2,
            gamma=gamma,
            temperature=temperature,
            eps=eps,
        )
        pair_name = f"{modality1}__{modality2}"
        pair_losses.append(pair_loss)
        detail[f"{pair_name}/corrected_cosie"] = pair_loss
        for diagnostic_name, destination in (
            ("corrected_joint_min", joint_mins),
            ("corrected_joint_max", joint_maxes),
            ("corrected_joint_sum", joint_sums),
        ):
            value = pair_diagnostics[diagnostic_name]
            destination.append(value)
            detail[f"{pair_name}/{diagnostic_name}"] = value

    total_loss = torch.stack(pair_losses).mean()
    detail.update(
        {
            "corrected_cosie": total_loss,
            "corrected_joint_min": torch.stack(joint_mins).min(),
            "corrected_joint_max": torch.stack(joint_maxes).max(),
            "corrected_joint_sum": torch.stack(joint_sums).mean(),
        }
    )
    return total_loss, detail


def _validate_projected_view(view: torch.Tensor, name: str) -> None:
    if not isinstance(view, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(view).__name__}.")
    if view.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, dim], got {tuple(view.shape)}.")
    if view.shape[0] < 2:
        raise ValueError(
            f"{name} has batch size {view.shape[0]}; contrastive learning requires "
            "at least two paired spots so that each spot has a negative."
        )
    if view.shape[1] < 1:
        raise ValueError(f"{name} must have at least one projection dimension.")


def _symmetric_infonce_with_diagnostics(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float,
    normalize_for_infonce: bool,
    normalization_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_projected_view(view_a, "view_a")
    _validate_projected_view(view_b, "view_b")
    if view_a.shape != view_b.shape:
        raise ValueError(
            "Symmetric InfoNCE requires paired views with identical shapes; "
            f"got {tuple(view_a.shape)} and {tuple(view_b.shape)}."
        )
    if view_a.device != view_b.device:
        raise ValueError(
            "Symmetric InfoNCE requires both views on the same device; "
            f"got {view_a.device} and {view_b.device}."
        )
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if not math.isfinite(normalization_eps) or normalization_eps <= 0:
        raise ValueError(
            f"normalization_eps must be positive and finite, got {normalization_eps}."
        )

    # Keep the numerically sensitive similarity matrix and cross entropy in
    # fp32 even when the projection heads run under bf16/fp16 autocast.
    with torch.autocast(device_type=view_a.device.type, enabled=False):
        projected_a = view_a.float()
        projected_b = view_b.float()
        if normalize_for_infonce:
            projected_a = F.normalize(
                projected_a,
                p=2,
                dim=1,
                eps=float(normalization_eps),
            )
            projected_b = F.normalize(
                projected_b,
                p=2,
                dim=1,
                eps=float(normalization_eps),
            )

        logits = torch.matmul(projected_a, projected_b.transpose(0, 1)) / float(
            temperature
        )
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_a_to_b = F.cross_entropy(logits, targets)
        loss_b_to_a = F.cross_entropy(logits.transpose(0, 1), targets)
        loss = 0.5 * (loss_a_to_b + loss_b_to_a)

        with torch.no_grad():
            top1_a_to_b = (logits.argmax(dim=1) == targets).float().mean()
            top1_b_to_a = (logits.argmax(dim=0) == targets).float().mean()

    return loss, top1_a_to_b, top1_b_to_a


def symmetric_infonce_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float = 0.2,
    normalize_for_infonce: bool = True,
    normalization_eps: float = 1e-4,
    negative_mode: str = "in_batch",
) -> torch.Tensor:
    """Compute same-spot, in-batch symmetric InfoNCE for two projected views.

    Row ``i`` in both tensors must describe the same spot. Other rows are used
    as negatives. Sampling is deliberately handled by the stage model so that
    every modality pair in a section shares exactly the same spot indices.
    """

    if negative_mode != "in_batch":
        raise NotImplementedError(
            "Only negative_mode='in_batch' is implemented in this phase; "
            f"got {negative_mode!r}. Queue-based negatives are intentionally deferred."
        )
    loss, _, _ = _symmetric_infonce_with_diagnostics(
        view_a=view_a,
        view_b=view_b,
        temperature=float(temperature),
        normalize_for_infonce=bool(normalize_for_infonce),
        normalization_eps=float(normalization_eps),
    )
    return loss


def vicreg_regularization(
    views: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return VICReg variance and covariance penalties averaged over views.

    The inputs must be the raw, non-L2-normalized projection-head outputs.
    Variance and covariance are computed once per modality, rather than once
    per modality pair, so adding a third modality does not duplicate either
    regularizer.
    """

    if not math.isfinite(variance_target) or variance_target <= 0:
        raise ValueError(
            f"variance_target must be positive and finite, got {variance_target}."
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps}.")

    if isinstance(views, Mapping):
        named_views = list(views.items())
    else:
        named_views = [(f"view_{index}", view) for index, view in enumerate(views)]
    if not named_views:
        raise ValueError("VICReg requires at least one projected view.")

    variance_losses: list[torch.Tensor] = []
    covariance_losses: list[torch.Tensor] = []
    expected_batch_size: int | None = None
    expected_device: torch.device | None = None

    for name, view in named_views:
        _validate_projected_view(view, str(name))
        if expected_batch_size is None:
            expected_batch_size = int(view.shape[0])
            expected_device = view.device
        elif int(view.shape[0]) != expected_batch_size:
            raise ValueError(
                "All VICReg views must have the same paired batch size; "
                f"expected {expected_batch_size}, got {view.shape[0]} for {name}."
            )
        if view.device != expected_device:
            raise ValueError("All VICReg views must be on the same device.")

        with torch.autocast(device_type=view.device.type, enabled=False):
            projected = view.float()
            centered = projected - projected.mean(dim=0, keepdim=True)
            variance = centered.square().sum(dim=0) / float(projected.shape[0] - 1)
            std = torch.sqrt(variance + float(eps))
            variance_losses.append(F.relu(float(variance_target) - std).mean())

            covariance = torch.matmul(centered.transpose(0, 1), centered) / float(
                projected.shape[0] - 1
            )
            off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
            covariance_losses.append(
                off_diagonal.square().sum() / float(projected.shape[1])
            )

    return torch.stack(variance_losses).mean(), torch.stack(covariance_losses).mean()


def compute_pairwise_crossview_loss(
    projected_latents: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute pair-mean symmetric InfoNCE plus modality-mean VICReg.

    ``projected_latents`` must already contain the same sampled spot rows for
    every modality. The returned details contain scalar tensors only; neither
    the projections nor the quadratic similarity matrices are retained there.
    """

    method = str(config.get("method", "symmetric_infonce_vicreg"))
    supported_methods = {"symmetric_infonce", "symmetric_infonce_vicreg"}
    if method not in supported_methods:
        raise ValueError(
            f"compute_pairwise_crossview_loss does not support method {method!r}; "
            f"expected one of {sorted(supported_methods)}."
        )
    if str(config.get("negative_mode", "in_batch")) != "in_batch":
        raise NotImplementedError(
            "Only negative_mode='in_batch' is implemented in this phase."
        )
    if str(config.get("pair_reduction", "mean")) != "mean":
        raise ValueError("Only pair_reduction='mean' is supported.")
    if bool(config.get("detach_target", False)):
        raise NotImplementedError("detach_target=True is not part of symmetric InfoNCE.")

    present = [
        (modality, projection)
        for modality, projection in projected_latents.items()
        if projection is not None
    ]
    if len(present) < 2:
        raise ValueError(
            "Cross-view InfoNCE requires at least two observed modality projections."
        )

    temperature = float(config.get("temperature", 0.2))
    normalization_eps = float(config.get("infonce_normalization_eps", 1e-4))
    normalize_for_infonce = bool(
        config.get(
            "normalize_for_infonce",
            config.get("normalize_projection", True),
        )
    )
    pair_losses: list[torch.Tensor] = []
    directional_top1: list[torch.Tensor] = []
    detail: dict[str, torch.Tensor] = {}

    for (modality_a, view_a), (modality_b, view_b) in itertools.combinations(present, 2):
        pair_loss, top1_a_to_b, top1_b_to_a = _symmetric_infonce_with_diagnostics(
            view_a=view_a,
            view_b=view_b,
            temperature=temperature,
            normalize_for_infonce=normalize_for_infonce,
            normalization_eps=normalization_eps,
        )
        pair_losses.append(pair_loss)
        directional_top1.extend([top1_a_to_b, top1_b_to_a])
        detail[f"{modality_a}__{modality_b}/infonce"] = pair_loss
        detail[f"{modality_a}_to_{modality_b}/top1"] = top1_a_to_b.detach()
        detail[f"{modality_b}_to_{modality_a}/top1"] = top1_b_to_a.detach()

    infonce_loss = torch.stack(pair_losses).mean()
    zero = infonce_loss.new_zeros(())
    if method == "symmetric_infonce_vicreg":
        variance_loss, covariance_loss = vicreg_regularization(
            dict(present),
            variance_target=float(config.get("variance_target", 1.0)),
            eps=float(config.get("vicreg_eps", 1e-4)),
        )
    else:
        variance_loss, covariance_loss = zero, zero

    lambda_var = float(config.get("lambda_var", 1.0))
    lambda_cov = float(config.get("lambda_cov", 0.04))
    if (
        not math.isfinite(lambda_var)
        or not math.isfinite(lambda_cov)
        or lambda_var < 0
        or lambda_cov < 0
    ):
        raise ValueError("lambda_var and lambda_cov must be finite and non-negative.")
    total_loss = infonce_loss + lambda_var * variance_loss + lambda_cov * covariance_loss

    with torch.no_grad():
        projection_stds = torch.cat(
            [
                projection.float().std(dim=0, correction=1)
                for _, projection in present
            ]
        )
        batch_size = int(present[0][1].shape[0])
        detail.update(
            {
                "infonce": infonce_loss,
                "variance": variance_loss,
                "covariance": covariance_loss,
                "crossview_total": total_loss,
                "crossmodal_top1_accuracy": torch.stack(directional_top1).mean().detach(),
                "contrastive_batch_size": torch.tensor(
                    batch_size, device=infonce_loss.device, dtype=torch.long
                ),
                "contrastive_num_negatives": torch.tensor(
                    batch_size - 1, device=infonce_loss.device, dtype=torch.long
                ),
                "projection_std_min": projection_stds.min().detach(),
                "projection_std_median": projection_stds.median().detach(),
            }
        )

    return total_loss, detail


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
