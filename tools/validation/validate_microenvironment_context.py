#!/usr/bin/env python3
"""Small validation of self-excluded pooling and OT-stage topology refresh."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F


from model.configure import get_default_model_config
from model.model_component import spatial_pool_self_excluded
from model.stage_model import StageMultiModalModel


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


def make_config(device: str) -> dict[str, Any]:
    config = get_default_model_config()
    config["training"]["device"] = device
    config["graph"]["knn_neighbors_spatial"] = 4
    config["encoder"]["dropout"] = 0.0
    config["fusion"]["dropout"] = 0.0
    config["graphsage"]["dropout"] = 0.0
    config["graphsage"]["edge_batch_size"] = 17
    config["ot_attention"]["dropout"] = 0.0
    config["decoder"]["dropout"] = 0.0
    config["uot"]["enabled"] = True
    config["uot"]["topology_aware_refresh_enabled"] = True
    config["uot"]["topology_context_weight"] = 0.2
    return config


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
    )
    expected = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}
    if set(model.ot_prior or {}) != expected:
        raise AssertionError(f"Unexpected initial bidirectional prior keys: {model.ot_prior}")
    metadata = (model.ot_prior or {})[("A", "B")]["metadata"]
    modalities = list((model.ot_prior or {})[("A", "B")]["modalities_used"])
    if modalities != ["RNA", "Protein"]:
        raise AssertionError("Initial OT did not use both input modalities.")
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
    model.eval()
    with torch.no_grad():
        outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_dict,
            section_order=["A", "B", "C"],
            epoch=1,
            decoder_chunk_size=11,
            ot_attention_source_chunk_size=7,
            checkpoint_ot_attention=False,
        )
    embeddings, contexts, diagnostics = model.prepare_ot_prior_refresh(outputs)
    if contexts is None:
        raise AssertionError("Topology refresh did not produce OT spatial contexts.")
    for section, semantic in embeddings.items():
        if not torch.equal(semantic, outputs["ot_embeddings"][section]):
            raise AssertionError("Dynamic OT semantic snapshot is not the OT stage.")
        context = contexts[section]
        if context.shape != semantic.shape:
            raise AssertionError("Dynamic OT context shape does not match the OT stage.")
        if semantic.requires_grad or context.requires_grad:
            raise AssertionError("Dynamic OT semantic/context snapshots must be detached.")
        if not torch.isfinite(context).all():
            raise AssertionError("Dynamic OT spatial context contains non-finite values.")

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
        context_embedding_dict=contexts,
        topology_context_weight=0.2,
    )
    expected_keys = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}
    if set(model.ot_prior or {}) != expected_keys:
        raise AssertionError("Topology refresh lost a bidirectional adjacent prior.")
    expected_metadata = {
        "retrieval_source": "ot",
        "semantic_source": "ot",
        "context_source": "ot_spatial_context",
        "topology_context_weight": 0.2,
        "semantic_cost_weight": 0.8,
    }
    for (source, target), prior in model.ot_prior.items():
        for key, expected in expected_metadata.items():
            if prior["metadata"].get(key) != expected:
                raise AssertionError(f"Dynamic OT metadata {key} does not match {expected!r}.")
        indices = prior["topk_idx"]
        if indices.shape[0] != embeddings[source].shape[0]:
            raise AssertionError("Direction prior rows do not match its source section.")
        if int(indices.min()) < 0 or int(indices.max()) >= embeddings[target].shape[0]:
            raise AssertionError("Direction prior contains an invalid target-local index.")
    return {
        "prior_keys": [list(key) for key in sorted(expected_keys)],
        "context_shapes": {section: list(value.shape) for section, value in contexts.items()},
        "context_requires_grad": {section: value.requires_grad for section, value in contexts.items()},
        "dynamic_refresh_diagnostics": diagnostics,
        "dynamic_ot_metadata": expected_metadata,
    }, outputs


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
    model = StageMultiModalModel(config=make_config(args.device), feature_dict=feature_dict)
    model.to(device)
    initial_ot = initialize_prior(model, feature_dict)
    pooling = validate_pooling(device)
    flow, _ = validate_model_flow(model, feature_dict, spatial_dict, device)
    print(json.dumps({
        "status": "PASS", "device": args.device,
        "initial_ot": initial_ot, "pooling": pooling, "model_flow": flow,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
