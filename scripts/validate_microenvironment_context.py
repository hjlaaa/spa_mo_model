#!/usr/bin/env python3
"""Validate fused microenvironment OT refresh and context-aware attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.model_component import spatial_pool_self_excluded
from model.stage_model import StageMultiModalModel


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(seed: int) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sizes = {"A": 19, "B": 27, "C": 23}
    feature_dict = {
        section: {
            "RNA": torch.randn(n, 12, generator=generator),
            "Protein": torch.randn(n, 12, generator=generator),
        }
        for section, n in sizes.items()
    }
    spatial_dict = {
        section: torch.rand(n, 2, generator=generator)
        for section, n in sizes.items()
    }
    return feature_dict, spatial_dict


def make_config(device: str, backprop_to_alpha: bool = False) -> dict[str, Any]:
    config = get_default_model_config()
    config["training"]["device"] = device
    config["graph"]["knn_neighbors_spatial"] = 4
    config["encoder"]["dropout"] = 0.0
    config["fusion"]["dropout"] = 0.0
    config["graphsage"]["dropout"] = 0.0
    config["graphsage"]["edge_batch_size"] = 17
    config["ot_attention"]["dropout"] = 0.0
    config["ot_attention"]["context_consistency_backprop_to_alpha"] = bool(
        backprop_to_alpha
    )
    config["decoder"]["dropout"] = 0.0
    config["uot"]["enabled"] = True
    config["uot"]["dynamic_refresh_source"] = "fused"
    config["uot"]["topology_aware_refresh_enabled"] = True
    config["uot"]["topology_context_weight"] = 0.2
    return config


def tensor_distribution(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten().cpu()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]),
    )
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.detach().float().flatten().cpu()
    y = y.detach().float().flatten().cpu()
    if x.numel() != y.numel() or x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-12:
        return None
    return float((x * y).sum() / denominator)


def validate_pooling(device: torch.device) -> dict[str, Any]:
    z = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 0.0]],
        device=device,
    )
    # Node 3 has only a self-loop and therefore exercises the empty-context path.
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2, 3], [0, 1, 2, 0, 1, 0, 2, 3]],
        dtype=torch.long,
        device=device,
    )
    edge_weight = torch.tensor(
        [0.2, 0.3, 0.5, 0.7, 0.3, 0.6, 0.4, 1.0],
        device=device,
    )
    context = spatial_pool_self_excluded(
        z.requires_grad_(True),
        edge_index,
        edge_weight,
        edge_batch_size=2,
        eps=1e-8,
        l2_normalize=True,
    )
    expected_raw = torch.stack(
        [0.375 * z[1] + 0.625 * z[2], z[0], z[0], torch.zeros_like(z[0])]
    ).detach()
    expected = F.normalize(expected_raw, p=2, dim=1, eps=1e-8)
    max_abs_diff = float((context - expected).abs().max().cpu())
    if max_abs_diff > 1e-6:
        raise AssertionError(f"Batched context does not match dense reference: {max_abs_diff}")
    if context.requires_grad:
        raise AssertionError("Spatial context unexpectedly requires gradients.")
    if not torch.isfinite(context).all():
        raise AssertionError("Spatial context contains non-finite values.")
    return {
        "shape": list(context.shape),
        "requires_grad": bool(context.requires_grad),
        "dense_reference_max_abs_diff": max_abs_diff,
        "isolated_node_context": context[3].detach().cpu().tolist(),
    }


def validate_reliability_gradient(model: StageMultiModalModel, device: torch.device) -> dict[str, Any]:
    attention = model.ot_attention
    n_source, n_target, k, dim = 7, 11, 4, model.latent_dim
    generator = torch.Generator(device="cpu").manual_seed(99)
    source_context = F.normalize(
        torch.randn(n_source, dim, generator=generator).to(device), dim=1
    ).detach()
    target_context = F.normalize(
        torch.randn(n_target, dim, generator=generator).to(device), dim=1
    ).detach()
    topk_idx = torch.randint(0, n_target, (n_source, k), generator=generator).to(device)
    topk_weight = torch.rand(n_source, k, generator=generator).to(device)
    topk_weight[0, -1] = 0.0

    logits_default = torch.randn(
        n_source, k, generator=generator
    ).to(device).requires_grad_(True)
    alpha_default = torch.softmax(logits_default, dim=1)
    attention.context_consistency_backprop_to_alpha = False
    r_default = attention.compute_context_reliability(
        alpha=alpha_default,
        source_context=source_context,
        target_context=target_context,
        topk_idx=topk_idx,
        topk_weight=topk_weight,
    )
    if r_default.requires_grad:
        raise AssertionError("Default r_ctx unexpectedly requires gradients.")

    logits_enabled = logits_default.detach().clone().requires_grad_(True)
    alpha_enabled = torch.softmax(logits_enabled, dim=1)
    attention.context_consistency_backprop_to_alpha = True
    r_enabled = attention.compute_context_reliability(
        alpha=alpha_enabled,
        source_context=source_context,
        target_context=target_context,
        topk_idx=topk_idx,
        topk_weight=topk_weight,
    )
    if not r_enabled.requires_grad:
        raise AssertionError("Enabled r_ctx did not recover its alpha gradient path.")
    r_enabled.sum().backward()
    alpha_path_grad_norm = float(torch.linalg.vector_norm(logits_enabled.grad).detach().cpu())
    if alpha_path_grad_norm <= 0.0:
        raise AssertionError("Enabled r_ctx produced no gradient to attention logits.")
    attention.context_consistency_backprop_to_alpha = False
    return {
        "default_r_requires_grad": bool(r_default.requires_grad),
        "enabled_r_requires_grad": bool(r_enabled.requires_grad),
        "enabled_logits_grad_norm": alpha_path_grad_norm,
        "padded_candidate_finite": bool(torch.isfinite(r_default).all()),
    }


def initialize_prior(
    model: StageMultiModalModel,
    feature_dict: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    model.initialize_candidate_sparse_ot_prior(
        feature_dict,
        section_order=["A", "B", "C"],
        initial_modality_candidate_k=6,
        candidate_k=8,
        attention_topk=4,
        candidate_backend="blockwise",
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=15,
        bidirectional=True,
    )
    expected = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}
    if set(model.ot_prior or {}) != expected:
        raise AssertionError(f"Unexpected initial bidirectional prior keys: {model.ot_prior}")
    metadata = (model.ot_prior or {})[("A", "B")]["metadata"]
    modalities = list((model.ot_prior or {})[("A", "B")]["modalities_used"])
    if modalities == ["fused_embedding"]:
        raise AssertionError("Initial OT was unexpectedly replaced by fused bootstrap.")
    return {
        "prior_keys": [list(key) for key in sorted(expected)],
        "modalities_used": modalities,
        "cost_definition": metadata.get("cost_definition"),
    }


def validate_model_flow(
    model: StageMultiModalModel,
    feature_dict: dict[str, dict[str, torch.Tensor]],
    spatial_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_inputs: list[torch.Tensor] = []
    gate_outputs: list[torch.Tensor] = []

    def gate_pre_hook(_module, args):
        gate_inputs.append(args[0].detach().float().cpu())

    def gate_hook(_module, _args, output):
        gate_outputs.append(output.detach().float().cpu())

    pre_handle = model.ot_attention.gate_mlp.register_forward_pre_hook(gate_pre_hook)
    out_handle = model.ot_attention.gate_mlp.register_forward_hook(gate_hook)
    model.eval()
    with torch.no_grad():
        outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_dict,
            section_order=["A", "B", "C"],
            epoch=1,
            decoder_chunk_size=11,
            ot_attention_source_chunk_size=7,
            bidirectional_ot_attention=True,
            checkpoint_ot_attention=False,
        )
    pre_handle.remove()
    out_handle.remove()

    for section, n_spots in {"A": 19, "B": 27, "C": 23}.items():
        context = outputs["context_embeddings"][section]
        if tuple(context.shape) != (n_spots, 128):
            raise AssertionError(f"Unexpected context shape for {section}: {context.shape}")
        if context.requires_grad:
            raise AssertionError(f"Context for {section} unexpectedly requires gradients.")
        if not torch.isfinite(context).all():
            raise AssertionError(f"Context for {section} contains non-finite values.")

    if not gate_inputs or any(tensor.shape[1] != 513 for tensor in gate_inputs):
        raise AssertionError("Gate did not receive [chunk, 513] inputs.")
    r_ctx = torch.cat([value[:, -1] for value in gate_inputs])
    gate = torch.cat(gate_outputs).flatten()

    # Reconstruct the middle-section synchronous mean update exactly.
    h_b = outputs["graphsage_embeddings"]["B"]
    c_b = outputs["context_embeddings"]["B"]
    directional_updates = []
    directional_shapes: dict[str, list[int]] = {}
    for target in ["A", "C"]:
        prior = (model.ot_prior or {})[("B", target)]
        update = model.ot_attention.compute_update_only(
            source_h=h_b,
            target_h=outputs["graphsage_embeddings"][target],
            topk_idx=prior["topk_idx"],
            topk_weight=prior["topk_weight"],
            confidence=prior["confidence"],
            source_context=c_b,
            target_context=outputs["context_embeddings"][target],
            epoch=1,
            source_chunk_size=7,
            checkpoint_attention=False,
        )
        directional_updates.append(update)
        directional_shapes[f"B_from_{target}"] = list(update.shape)
    manual_b = model.ot_attention.apply_update(
        h_b,
        (directional_updates[0] + directional_updates[1]) / 2.0,
    )
    averaging_diff = float(
        (manual_b - outputs["final_embeddings"]["B"]).abs().max().detach().cpu()
    )
    if averaging_diff > 1e-6:
        raise AssertionError(f"Three-section mean update mismatch: {averaging_diff}")

    embeddings, contexts, refresh_diagnostics = model.prepare_ot_prior_refresh(outputs)
    for section in ["A", "B", "C"]:
        if not torch.allclose(embeddings[section], outputs["fused_embeddings"][section]):
            raise AssertionError("Dynamic OT semantic snapshot is not fused z.")
        if contexts is None or not torch.allclose(
            contexts[section], outputs["context_embeddings"][section]
        ):
            raise AssertionError("Dynamic OT context snapshot is not Pool(fused z).")

    model.update_candidate_sparse_ot_prior(
        embeddings,
        section_order=["A", "B", "C"],
        candidate_k=8,
        attention_topk=4,
        candidate_backend="blockwise",
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=15,
        candidate_source="fused",
        bidirectional=True,
        context_embedding_dict=contexts,
        topology_context_weight=0.2,
    )
    dynamic_metadata = (model.ot_prior or {})[("A", "B")]["metadata"]
    expected_metadata = {
        "retrieval_source": "fused",
        "semantic_source": "fused",
        "context_source": "fused_spatial_context",
    }
    for key, expected in expected_metadata.items():
        if dynamic_metadata.get(key) != expected:
            raise AssertionError(
                f"Dynamic OT metadata {key}={dynamic_metadata.get(key)!r}, expected {expected!r}."
            )

    first_layer = model.ot_attention.gate_mlp[0]
    summary = {
        "context_shapes": {
            section: list(value.shape)
            for section, value in outputs["context_embeddings"].items()
        },
        "context_requires_grad": {
            section: bool(value.requires_grad)
            for section, value in outputs["context_embeddings"].items()
        },
        "directional_update_shapes": directional_shapes,
        "middle_section_mean_update_max_abs_diff": averaging_diff,
        "gate_input_shapes": sorted({tuple(value.shape) for value in gate_inputs}),
        "r_ctx_distribution": tensor_distribution(r_ctx),
        "gate_distribution": tensor_distribution(gate),
        "r_ctx_gate_correlation": correlation(r_ctx, gate),
        "gate_context_column_weight_norm": float(
            torch.linalg.vector_norm(first_layer.weight[:, -1].detach().float()).cpu()
        ),
        "dynamic_refresh_diagnostics": refresh_diagnostics,
        "dynamic_ot_metadata": {
            key: dynamic_metadata.get(key)
            for key in [
                "retrieval_source",
                "semantic_source",
                "context_source",
                "cost_definition",
                "topology_context_weight",
                "semantic_cost_weight",
            ]
        },
    }
    return summary, outputs


def benchmark_components(
    model: StageMultiModalModel,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(321)
    n_source, n_target, dim, k = 2048, 2304, model.latent_dim, 10
    source_h = torch.randn(n_source, dim, generator=generator).to(device).requires_grad_(True)
    target_h = torch.randn(n_target, dim, generator=generator).to(device).requires_grad_(True)
    source_context = F.normalize(
        torch.randn(n_source, dim, generator=generator).to(device), dim=1
    ).detach()
    target_context = F.normalize(
        torch.randn(n_target, dim, generator=generator).to(device), dim=1
    ).detach()
    topk_idx = torch.randint(0, n_target, (n_source, k), generator=generator).to(device)
    topk_weight = torch.rand(n_source, k, generator=generator).to(device)
    topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True)
    confidence = torch.rand(n_source, generator=generator).to(device)

    edge_source = torch.arange(n_source, device=device).repeat_interleave(k + 1)
    offsets = torch.arange(k + 1, device=device).repeat(n_source)
    edge_target = (edge_source + offsets) % n_source
    edge_index = torch.stack([edge_source, edge_target])
    edge_weight = torch.ones(edge_index.shape[1], device=device)
    z = torch.randn(n_source, dim, generator=generator).to(device)

    synchronize(device)
    start = time.perf_counter()
    pooled = spatial_pool_self_excluded(
        z,
        edge_index,
        edge_weight,
        edge_batch_size=4096,
        eps=1e-8,
        l2_normalize=True,
    )
    synchronize(device)
    spatial_pool_sec = time.perf_counter() - start

    logits = torch.randn(n_source, k, generator=generator).to(device).requires_grad_(True)
    alpha = torch.softmax(logits, dim=1)
    synchronize(device)
    start = time.perf_counter()
    reliability = model.ot_attention.compute_context_reliability(
        alpha=alpha,
        source_context=source_context,
        target_context=target_context,
        topk_idx=topk_idx,
        topk_weight=topk_weight,
    )
    synchronize(device)
    reliability_sec = time.perf_counter() - start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    update = model.ot_attention.compute_update_only(
        source_h=source_h,
        target_h=target_h,
        topk_idx=topk_idx,
        topk_weight=topk_weight,
        confidence=confidence,
        source_context=source_context,
        target_context=target_context,
        source_chunk_size=512,
        checkpoint_attention=True,
    )
    synchronize(device)
    forward_sec = time.perf_counter() - start
    forward_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    start = time.perf_counter()
    update.float().square().mean().backward()
    synchronize(device)
    backward_sec = time.perf_counter() - start
    context_column_grad = model.ot_attention.gate_mlp[0].weight.grad[:, -1]
    context_column_grad_norm = float(
        torch.linalg.vector_norm(context_column_grad.detach().float()).cpu()
    )
    if context_column_grad_norm <= 0.0:
        raise AssertionError("The gate context-input column received no gradient.")
    backward_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "device": str(device),
        "spatial_pool_seconds": float(spatial_pool_sec),
        "context_reliability_seconds": float(reliability_sec),
        "context_aware_attention_forward_seconds": float(forward_sec),
        "context_aware_attention_backward_seconds": float(backward_sec),
        "forward_peak_allocated_bytes": forward_peak,
        "backward_peak_allocated_bytes": backward_peak,
        "gate_context_column_grad_norm": context_column_grad_norm,
        "pool_shape": list(pooled.shape),
        "reliability_shape": list(reliability.shape),
        "max_edge_message_shape": [4096, dim],
        "max_candidate_context_shape": [512, k, dim],
    }


def validate_bfloat16(model: StageMultiModalModel, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return {"supported": False, "reason": "CUDA device does not support BF16."}
    dim, n_source, n_target, k = model.latent_dim, 13, 17, 4
    source_h = torch.randn(n_source, dim, device=device, dtype=torch.bfloat16).requires_grad_(True)
    target_h = torch.randn(n_target, dim, device=device, dtype=torch.bfloat16).requires_grad_(True)
    source_context = F.normalize(source_h.detach().float(), dim=1).to(torch.bfloat16)
    target_context = F.normalize(target_h.detach().float(), dim=1).to(torch.bfloat16)
    topk_idx = torch.randint(0, n_target, (n_source, k), device=device)
    topk_weight = torch.rand(n_source, k, device=device, dtype=torch.float32)
    topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True)
    confidence = torch.rand(n_source, device=device, dtype=torch.float32)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    model.ot_attention.zero_grad(set_to_none=True)
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        update = model.ot_attention.compute_update_only(
            source_h=source_h,
            target_h=target_h,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            confidence=confidence,
            source_context=source_context,
            target_context=target_context,
            source_chunk_size=5,
            checkpoint_attention=True,
        )
        loss = update.float().square().mean()
    loss.backward()
    if update.dtype != torch.bfloat16:
        raise AssertionError(f"BF16 attention returned unexpected dtype {update.dtype}.")
    if not torch.isfinite(update).all() or not torch.isfinite(loss):
        raise AssertionError("BF16 attention produced non-finite values.")
    return {
        "supported": True,
        "output_dtype": str(update.dtype),
        "output_shape": list(update.shape),
        "loss": float(loss.detach().cpu()),
        "source_grad_finite": bool(torch.isfinite(source_h.grad).all()),
        "target_grad_finite": bool(torch.isfinite(target_h.grad).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    feature_dict, spatial_dict = make_inputs(int(args.seed))
    model = StageMultiModalModel(
        config=make_config(args.device),
        feature_dict=feature_dict,
    )
    model.to(device)
    initial_ot = initialize_prior(model, feature_dict)
    pooling = validate_pooling(device)
    gradient_path = validate_reliability_gradient(model, device)
    flow, _ = validate_model_flow(model, feature_dict, spatial_dict, device)
    benchmark = benchmark_components(model, device)
    bfloat16 = validate_bfloat16(model, device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": args.device,
                "initial_ot": initial_ot,
                "pooling": pooling,
                "gradient_path": gradient_path,
                "model_flow": flow,
                "benchmark": benchmark,
                "bfloat16": bfloat16,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
