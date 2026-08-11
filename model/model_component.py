"""Model components for the first spa_mo_model multimodal stage."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _build_activation(name: str) -> nn.Module:
    if name == "GELU":
        return nn.GELU()
    if name == "ReLU":
        return nn.ReLU()
    if name == "SiLU":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_norm(name: str | None, dim: int) -> nn.Module | None:
    if name is None:
        return None
    if name == "LayerNorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported normalization: {name}")


class ModalityMLPEncoder(nn.Module):
    """Modality-specific MLP encoder used in this project's first model stage.

    This is intentionally not COSIE's GraphAutoencoder. It maps each
    preprocessed modality feature matrix to a 128-dimensional normalized latent
    representation so COSIE's cross-view loss can be applied downstream.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 128,
        activation: str = "GELU",
        dropout: float = 0.1,
        norm: str | None = "LayerNorm",
        l2_normalize_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.l2_normalize_output = bool(l2_normalize_output)

        dims = [self.input_dim] + [int(dim) for dim in hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            norm_layer = _build_norm(norm, out_dim)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)
        if self.l2_normalize_output:
            z = F.normalize(z, p=2, dim=1)
        return z


class FusionMLP(nn.Module):
    """Fuse observed modality latents into one 128-dimensional spot embedding.

    This fusion module is a project-specific addition after COSIE-style
    cross-view alignment. COSIE's original final embedding keeps the
    concatenated ``128 * num_modalities`` representation instead of projecting
    it back to 128 dimensions.
    """

    def __init__(
        self,
        modality_order: Sequence[str],
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 128,
        activation: str = "GELU",
        dropout: float = 0.1,
        norm: str | None = "LayerNorm",
        residual: str | None = "mean_residual",
    ):
        super().__init__()
        if not (2 <= len(modality_order) <= 3):
            raise ValueError("FusionMLP expects two or three observed modalities.")
        self.modality_order = list(modality_order)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.residual = residual

        input_dim = self.latent_dim * len(self.modality_order)
        dims = [input_dim] + [int(dim) for dim in hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            norm_layer = _build_norm(norm, out_dim)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.network = nn.Sequential(*layers)
        self.output_norm = _build_norm(norm, self.output_dim)

    def forward(self, latent_dict_for_one_section: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [
            modality
            for modality in self.modality_order
            if modality not in latent_dict_for_one_section
        ]
        if missing:
            raise KeyError(f"FusionMLP missing required modalities: {missing}")

        latents = [latent_dict_for_one_section[modality] for modality in self.modality_order]
        n_spots = latents[0].shape[0]
        for modality, latent in zip(self.modality_order, latents):
            if latent.shape != (n_spots, self.latent_dim):
                raise ValueError(
                    f"Expected {modality} latent shape {(n_spots, self.latent_dim)}, "
                    f"got {tuple(latent.shape)}."
                )

        z_concat = torch.cat(latents, dim=-1)
        h_mlp = self.network(z_concat)

        if self.residual == "mean_residual":
            if self.output_dim != self.latent_dim:
                raise ValueError("mean_residual requires output_dim == latent_dim.")
            h_mean = torch.stack(latents, dim=0).mean(dim=0)
            h_fused = h_mlp + h_mean
        elif self.residual in {False, None, "none"}:
            h_fused = h_mlp
        else:
            raise ValueError(f"Unsupported fusion residual mode: {self.residual}")

        if self.output_norm is not None:
            h_fused = self.output_norm(h_fused)
        return h_fused


class WeightedResidualGraphSAGE(nn.Module):
    """One-layer weighted residual GraphSAGE over spatial neighbors.

    Each forward pass starts from the current fused embedding ``Z``. The module
    does not cache or reuse a previous epoch's graph embedding.
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        self_path_mode: str = "legacy",
        dropout: float = 0.1,
        activation: str = "GELU",
        norm: str | None = "LayerNorm",
        residual: bool = True,
        edge_batch_size: int | None = 200000,
    ):
        super().__init__()
        if input_dim != output_dim and residual:
            raise ValueError("Residual GraphSAGE requires input_dim == output_dim.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual = bool(residual)
        self.self_path_mode = str(self_path_mode)
        supported_self_path_modes = {"legacy", "no_adj_self", "no_self_linear"}
        if self.self_path_mode not in supported_self_path_modes:
            raise ValueError(
                f"Unsupported GraphSAGE self_path_mode {self.self_path_mode!r}; "
                f"expected one of {sorted(supported_self_path_modes)}."
            )
        self.edge_batch_size = None if edge_batch_size is None else int(edge_batch_size)
        # Keep self_linear in every mode so checkpoints and state_dict keys are
        # identical across the G0 ablations. G0-b bypasses it in forward().
        self.self_linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.neigh_linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = _build_norm(norm, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_batch_size: int | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}.")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have shape [E].")

        source = edge_index[0].to(x.device)
        target = edge_index[1].to(x.device)
        weight = edge_weight.to(device=x.device, dtype=x.dtype)
        neigh = torch.zeros_like(x)
        batch_size = self.edge_batch_size if edge_batch_size is None else edge_batch_size
        if batch_size is None or int(batch_size) <= 0:
            neigh.index_add_(0, source, x[target] * weight.unsqueeze(-1))
        else:
            batch_size = int(batch_size)
            num_edges = int(source.shape[0])
            for start in range(0, num_edges, batch_size):
                end = min(start + batch_size, num_edges)
                source_b = source[start:end]
                target_b = target[start:end]
                weight_b = weight[start:end]
                msg_b = x[target_b] * weight_b.unsqueeze(-1)
                neigh.index_add_(0, source_b, msg_b)

        if self.self_path_mode == "no_self_linear":
            self_message = torch.zeros(
                (x.shape[0], self.output_dim),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            self_message = self.self_linear(x)
        neighbor_message = self.neigh_linear(neigh)
        out = self.activation(self_message + neighbor_message + self.bias)
        out = self.dropout(out)
        if self.residual:
            out = x + out
        if self.norm is not None:
            out = self.norm(out)
        if not return_diagnostics:
            return out

        eps = torch.finfo(x.dtype).eps
        row_weight_sum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        row_weight_sum.index_add_(0, source, weight)
        self_loop_mask = source == target
        diagnostics: dict[str, object] = {
            "self_path_mode": self.self_path_mode,
            "edge_count": int(source.numel()),
            "self_loop_count": int(self_loop_mask.sum().detach().cpu().item()),
            "row_weight_sum_min": float(row_weight_sum.min().detach().cpu().item()),
            "row_weight_sum_max": float(row_weight_sum.max().detach().cpu().item()),
            "self_message_mean_l2": float(
                torch.linalg.vector_norm(self_message.float(), dim=1).mean().detach().cpu().item()
            ),
            "neighbor_message_mean_l2": float(
                torch.linalg.vector_norm(neighbor_message.float(), dim=1).mean().detach().cpu().item()
            ),
            "residual_input_mean_l2": float(
                torch.linalg.vector_norm(x.float(), dim=1).mean().detach().cpu().item()
            ),
            "self_to_neighbor_norm_ratio": float(
                (
                    torch.linalg.vector_norm(self_message.float(), dim=1).mean()
                    / torch.linalg.vector_norm(neighbor_message.float(), dim=1)
                    .mean()
                    .clamp_min(eps)
                )
                .detach()
                .cpu()
                .item()
            ),
        }
        return out, diagnostics


def spatial_pool_self_excluded(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    edge_batch_size: int | None = 200000,
    eps: float = 1e-8,
    l2_normalize: bool = True,
) -> torch.Tensor:
    """Return a detached, self-excluded weighted spatial-neighbor mean.

    The aggregation deliberately uses two edge-batched passes: the first
    recomputes per-source weight sums after self-loop removal and the second
    accumulates weighted neighbor embeddings.  It therefore never materializes
    a full ``[E, D]`` message tensor.  When ``l2_normalize`` is enabled, node
    chunks are normalized in place so a second full context matrix is not kept
    alive.
    """

    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError(f"z must be a non-empty 2D tensor, got {tuple(z.shape)}.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight must have shape [E].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    embedding = z.detach()
    source = edge_index[0].to(device=embedding.device)
    target = edge_index[1].to(device=embedding.device)
    num_edges = int(source.shape[0])
    batch_size = int(edge_batch_size or 200000)
    if batch_size <= 0:
        batch_size = 200000

    # FP32 row sums avoid BF16 underflow while adding only O(N) storage.
    row_sum = torch.zeros(
        embedding.shape[0],
        device=embedding.device,
        dtype=torch.float32,
    )
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        source_b = source[start:end]
        target_b = target[start:end]
        keep_b = source_b != target_b
        source_b = source_b[keep_b]
        weight_b = edge_weight[start:end].to(
            device=embedding.device,
            dtype=torch.float32,
        )[keep_b]
        row_sum.index_add_(0, source_b, weight_b)

    context = torch.zeros_like(embedding)
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        source_b = source[start:end]
        target_b = target[start:end]
        keep_b = source_b != target_b
        source_b = source_b[keep_b]
        target_b = target_b[keep_b]
        weight_b = edge_weight[start:end].to(
            device=embedding.device,
            dtype=torch.float32,
        )[keep_b]
        normalized_weight_b = weight_b / row_sum[source_b].clamp_min(float(eps))
        message_b = embedding[target_b] * normalized_weight_b.to(
            dtype=embedding.dtype
        ).unsqueeze(-1)
        context.index_add_(0, source_b, message_b)

    if l2_normalize:
        num_nodes = int(context.shape[0])
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            context_b = context[start:end]
            norm_b = torch.linalg.vector_norm(
                context_b.float(),
                dim=1,
                keepdim=True,
            ).clamp_min(float(eps))
            context[start:end] = (context_b.float() / norm_b).to(context.dtype)
    return context.detach()


class OTGuidedAttention(nn.Module):
    """Single-direction top-k OT-guided cross-stage attention."""

    def __init__(
        self,
        dim: int = 128,
        d_attn: int = 128,
        beta: float = 0.2,
        beta_warmup: bool = False,
        beta_schedule: Sequence[Sequence[float]] | None = None,
        dropout: float = 0.1,
        gate: str = "scalar",
        use_confidence: bool = True,
        residual: bool = True,
        norm: str | None = "LayerNorm",
        delta: float = 1e-8,
        context_gate_enabled: bool = True,
        context_consistency_backprop_to_alpha: bool = False,
    ):
        super().__init__()
        if d_attn != dim:
            raise ValueError(
                "OTGuidedAttention currently requires d_attn == dim because the scalar gate "
                "uses [source_h, message, source_h - message, source_h * message, r_ctx]. "
                "Please keep d_attn equal to dim unless the gate design is changed."
            )
        if gate != "scalar":
            raise ValueError("Only scalar gate is supported in OTGuidedAttention.")
        self.dim = int(dim)
        self.d_attn = int(d_attn)
        self.beta = float(beta)
        self.beta_warmup = bool(beta_warmup)
        self.beta_schedule = list(beta_schedule or [])
        self.use_confidence = bool(use_confidence)
        self.residual = bool(residual)
        self.delta = float(delta)
        self.context_gate_enabled = bool(context_gate_enabled)
        self.context_consistency_backprop_to_alpha = bool(
            context_consistency_backprop_to_alpha
        )
        self.W_Q = nn.Linear(self.dim, self.d_attn)
        self.W_K = nn.Linear(self.dim, self.d_attn)
        self.W_V = nn.Linear(self.dim, self.d_attn)
        self.W_O = nn.Linear(self.d_attn, self.dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * self.dim + (1 if self.context_gate_enabled else 0), self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, 1),
            nn.Sigmoid(),
        )
        self.norm = _build_norm(norm, self.dim)

    def _current_beta(self, epoch: int | None) -> float:
        if not self.beta_warmup or epoch is None:
            return self.beta
        for start, end, value in self.beta_schedule:
            if epoch >= int(start) and (int(end) < 0 or epoch <= int(end)):
                return float(value)
        return self.beta

    def forward(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        confidence: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        epoch: int | None = None,
        source_chunk_size: int | None = None,
        checkpoint_attention: bool = False,
    ) -> torch.Tensor:
        update = self.compute_update_only(
            source_h=source_h,
            target_h=target_h,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            confidence=confidence,
            source_context=source_context,
            target_context=target_context,
            epoch=epoch,
            source_chunk_size=source_chunk_size,
            checkpoint_attention=checkpoint_attention,
        )
        return self.apply_update(source_h, update)

    def compute_update_only(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        confidence: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        epoch: int | None = None,
        source_chunk_size: int | None = None,
        checkpoint_attention: bool = False,
    ) -> torch.Tensor:
        """Return only the OT-guided directional update before residual add.

        This is used by synchronous bidirectional attention, where multiple
        directional updates must be computed from the same base embeddings and
        averaged before one residual update is applied.
        """

        topk_idx = topk_idx.to(source_h.device)
        topk_weight = topk_weight.to(device=source_h.device, dtype=source_h.dtype)
        confidence = confidence.to(device=source_h.device, dtype=source_h.dtype)
        target_h = target_h.to(source_h.device)
        if self.context_gate_enabled:
            source_context = source_context.detach().to(device=source_h.device)
            target_context = target_context.detach().to(device=source_h.device)
            if source_context.shape != source_h.shape:
                raise ValueError(
                    "source_context must match source_h shape; "
                    f"got {tuple(source_context.shape)} vs {tuple(source_h.shape)}."
                )
            if target_context.shape != target_h.shape:
                raise ValueError(
                    "target_context must match target_h shape; "
                    f"got {tuple(target_context.shape)} vs {tuple(target_h.shape)}."
                )
        else:
            # Keep tensor placeholders in the checkpoint signature while the
            # v3-compatible 512-dimensional gate ignores spatial context.
            source_context = source_context.to(device=source_h.device)
            target_context = target_context.to(device=source_h.device)

        use_checkpoint = bool(
            checkpoint_attention
            and self.training
            and torch.is_grad_enabled()
        )

        if source_chunk_size is not None and int(source_chunk_size) > 0:
            chunk_size = int(source_chunk_size)
            chunks: list[torch.Tensor] = []
            for start in range(0, int(source_h.shape[0]), chunk_size):
                end = min(start + chunk_size, int(source_h.shape[0]))
                chunks.append(
                    self._compute_update_chunk_maybe_checkpointed(
                        source_h=source_h[start:end],
                        target_h=target_h,
                        topk_idx=topk_idx[start:end],
                        topk_weight=topk_weight[start:end],
                        confidence=confidence[start:end],
                        source_context=source_context[start:end],
                        target_context=target_context,
                        epoch=epoch,
                        use_checkpoint=use_checkpoint,
                    )
                )
            return torch.cat(chunks, dim=0)

        return self._compute_update_chunk_maybe_checkpointed(
            source_h=source_h,
            target_h=target_h,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            confidence=confidence,
            source_context=source_context,
            target_context=target_context,
            epoch=epoch,
            use_checkpoint=use_checkpoint,
        )

    def _compute_update_chunk_maybe_checkpointed(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        confidence: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        epoch: int | None = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        if not use_checkpoint:
            return self._compute_update_chunk(
                source_h=source_h,
                target_h=target_h,
                topk_idx=topk_idx,
                topk_weight=topk_weight,
                confidence=confidence,
                source_context=source_context,
                target_context=target_context,
                epoch=epoch,
            )

        def chunk_fn(
            source_h_chunk: torch.Tensor,
            target_h_full: torch.Tensor,
            topk_weight_chunk: torch.Tensor,
            confidence_chunk: torch.Tensor,
            source_context_chunk: torch.Tensor,
            target_context_full: torch.Tensor,
        ) -> torch.Tensor:
            return self._compute_update_chunk(
                source_h=source_h_chunk,
                target_h=target_h_full,
                topk_idx=topk_idx,
                topk_weight=topk_weight_chunk,
                confidence=confidence_chunk,
                source_context=source_context_chunk,
                target_context=target_context_full,
                epoch=epoch,
            )

        return torch_checkpoint(
            chunk_fn,
            source_h,
            target_h,
            topk_weight,
            confidence,
            source_context,
            target_context,
            use_reentrant=False,
            preserve_rng_state=True,
        )

    def _compute_update_chunk(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        confidence: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        q = self.W_Q(source_h)
        candidate_h = target_h[topk_idx]
        k = self.W_K(candidate_h)
        v = self.W_V(candidate_h)

        beta = self._current_beta(epoch)
        scores = (q.unsqueeze(1) * k).sum(dim=-1) / math.sqrt(self.d_attn)
        log_prior = torch.log(topk_weight.float().clamp_min(self.delta))
        scores = scores.float() + beta * log_prior
        alpha = torch.softmax(scores, dim=1).to(v.dtype)
        message = (alpha.unsqueeze(-1) * v).sum(dim=1)
        message_bar = self.W_O(message)

        gate_features = [
            source_h,
            message_bar,
            source_h - message_bar,
            source_h * message_bar,
        ]
        if self.context_gate_enabled:
            r_ctx = self.compute_context_reliability(
                alpha=alpha,
                source_context=source_context,
                target_context=target_context,
                topk_idx=topk_idx,
                topk_weight=topk_weight,
            ).to(dtype=source_h.dtype)
            gate_features.append(r_ctx)
        gate_input = torch.cat(gate_features, dim=-1)
        gate = self.gate_mlp(gate_input)
        if self.use_confidence:
            update_scale = confidence.unsqueeze(-1) * gate
        else:
            update_scale = gate
        return update_scale * message_bar

    def compute_context_reliability(
        self,
        *,
        alpha: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted top-k context reliability ``[N, 1]``.

        Section contexts are already detached and L2-normalized by the spatial
        pooling branch.  Dot products are accumulated in FP32.  Zero-weight
        padded candidates receive neutral reliability (0.5), preventing their
        repeated index from injecting an arbitrary context value without
        changing the existing attention-score behavior.
        """

        candidate_context = target_context[topk_idx]
        # FP32 batched matrix-vector multiplication preserves FP32 dot-product
        # accumulation without materializing another [chunk, K, D] product.
        cosine = torch.bmm(
            candidate_context.float(),
            source_context.float().unsqueeze(-1),
        ).squeeze(-1)
        cosine = cosine.clamp(min=-1.0, max=1.0)
        pairwise_reliability = 0.5 * (1.0 + cosine)
        valid_candidate = topk_weight.float() > 0.0
        pairwise_reliability = torch.where(
            valid_candidate,
            pairwise_reliability,
            torch.full_like(pairwise_reliability, 0.5),
        )
        reliability = (
            alpha.float() * pairwise_reliability
        ).sum(dim=1, keepdim=True)
        if not self.context_consistency_backprop_to_alpha:
            reliability = reliability.detach()
        return reliability

    def apply_update(self, source_h: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Apply the residual/dropout/norm step to a precomputed update."""

        update = self.dropout(update)
        if self.residual:
            h_tilde = source_h + update
        else:
            h_tilde = update
        if self.norm is not None:
            h_tilde = self.norm(h_tilde)
        return h_tilde


class ModalityDecoder(nn.Module):
    """Decode final 128-d embeddings back to preprocessed modality features."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 50,
        activation: str = "GELU",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
