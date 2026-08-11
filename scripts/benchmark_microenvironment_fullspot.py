#!/usr/bin/env python3
"""Synthetic CRC full-spot memory/runtime trace for microenvironment components."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_component import OTGuidedAttention, spatial_pool_self_excluded


def mib(value: int) -> float:
    return float(value / 1024**2)


def memory(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    return {
        "allocated_mib": mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": mib(torch.cuda.memory_reserved(device)),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }


def ring_graph(n_nodes: int, neighbors: int, device: torch.device):
    # Include offset zero as the graph self-loop; SpatialPool removes it.
    source = torch.arange(n_nodes, device=device).repeat_interleave(neighbors + 1)
    offsets = torch.arange(neighbors + 1, device=device).repeat(n_nodes)
    target = (source + offsets) % n_nodes
    edge_index = torch.stack([source, target])
    edge_weight = torch.full(
        (edge_index.shape[1],),
        1.0 / float(neighbors + 1),
        device=device,
        dtype=torch.float32,
    )
    return edge_index, edge_weight


def build_context(
    n_nodes: int,
    dim: int,
    neighbors: int,
    edge_batch_size: int,
    device: torch.device,
):
    z = torch.randn(n_nodes, dim, device=device, dtype=torch.bfloat16)
    edge_index, edge_weight = ring_graph(n_nodes, neighbors, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    context = spatial_pool_self_excluded(
        z,
        edge_index,
        edge_weight,
        edge_batch_size=edge_batch_size,
        eps=1e-8,
        l2_normalize=True,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    stats = {
        "n_nodes": n_nodes,
        "edge_count": int(edge_index.shape[1]),
        "seconds": float(elapsed),
        **memory(device),
    }
    del z, edge_index, edge_weight
    torch.cuda.empty_cache()
    return context, stats


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device("cuda")
    torch.manual_seed(42)
    dim = 128
    neighbors = 10
    edge_batch_size = 100000
    source_chunk_size = 50000
    attention_topk = 10
    n_small = 166279
    n_large = 446095

    attention = OTGuidedAttention(
        dim=dim,
        d_attn=dim,
        beta=0.2,
        dropout=0.0,
        context_consistency_backprop_to_alpha=False,
    ).to(device).train()

    context_small, pool_small = build_context(
        n_small, dim, neighbors, edge_batch_size, device
    )
    context_large, pool_large = build_context(
        n_large, dim, neighbors, edge_batch_size, device
    )
    if context_small.requires_grad or context_large.requires_grad:
        raise AssertionError("Full-spot context unexpectedly requires gradients.")

    # Exercise the larger-source direction because it has the largest number of
    # source chunks and output/update storage.
    source_h = torch.randn(
        n_large, dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    target_h = torch.randn(
        n_small, dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    topk_idx = torch.randint(
        0, n_small, (n_large, attention_topk), device=device
    )
    topk_weight = torch.full(
        (n_large, attention_topk),
        1.0 / float(attention_topk),
        device=device,
        dtype=torch.bfloat16,
    )
    confidence = torch.rand(n_large, device=device, dtype=torch.bfloat16)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    reliability_chunks = []
    for chunk_start in range(0, n_large, source_chunk_size):
        chunk_end = min(chunk_start + source_chunk_size, n_large)
        alpha_chunk = topk_weight[chunk_start:chunk_end]
        reliability_chunks.append(
            attention.compute_context_reliability(
                alpha=alpha_chunk,
                source_context=context_large[chunk_start:chunk_end],
                target_context=context_small,
                topk_idx=topk_idx[chunk_start:chunk_end],
                topk_weight=topk_weight[chunk_start:chunk_end],
            )
        )
    reliability = torch.cat(reliability_chunks)
    torch.cuda.synchronize(device)
    reliability_seconds = time.perf_counter() - start
    reliability_memory = memory(device)
    if not torch.isfinite(reliability).all():
        raise AssertionError("Full-spot context reliability contains non-finite values.")
    del reliability, reliability_chunks

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        update = attention.compute_update_only(
            source_h=source_h,
            target_h=target_h,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            confidence=confidence,
            source_context=context_large,
            target_context=context_small,
            source_chunk_size=source_chunk_size,
            checkpoint_attention=True,
        )
        loss = update.float().square().mean()
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    forward_memory = memory(device)

    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - start
    backward_memory = memory(device)
    gate_context_grad_norm = float(
        torch.linalg.vector_norm(
            attention.gate_mlp[0].weight.grad[:, -1].detach().float()
        ).cpu()
    )
    if gate_context_grad_norm <= 0.0:
        raise AssertionError("Gate context column received no full-spot gradient.")

    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(device),
                "dtype": "bfloat16",
                "shape": {
                    "small_section_spots": n_small,
                    "large_section_spots": n_large,
                    "dim": dim,
                    "spatial_neighbors": neighbors,
                    "attention_topk": attention_topk,
                    "edge_batch_size": edge_batch_size,
                    "attention_source_chunk_size": source_chunk_size,
                },
                "spatial_pool_small": pool_small,
                "spatial_pool_large": pool_large,
                "large_source_attention": {
                    "direction_shape": [n_large, n_small],
                    "update_shape": list(update.shape),
                    "forward_seconds": float(forward_seconds),
                    "backward_seconds": float(backward_seconds),
                    "context_reliability_only_seconds": float(reliability_seconds),
                    "context_reliability_only_memory": reliability_memory,
                    "forward_memory": forward_memory,
                    "backward_memory": backward_memory,
                    "gate_context_column_grad_norm": gate_context_grad_norm,
                    "max_candidate_context_shape": [
                        source_chunk_size,
                        attention_topk,
                        dim,
                    ],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
