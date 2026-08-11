#!/usr/bin/env python3
"""Small deterministic validation for experiment C."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.model_component import OTGuidedAttention
from model.stage_model import StageMultiModalModel


def validate_attention() -> dict:
    torch.manual_seed(42)
    module = OTGuidedAttention(
        dim=128,
        d_attn=128,
        dropout=0.0,
        context_gate_enabled=False,
    )
    module.train()
    if module.gate_mlp[0].in_features != 512:
        raise AssertionError("Experiment C gate must have 512 input features.")
    n_source, n_target, k = 9, 13, 4
    source_base = torch.randn(n_source, 128)
    target_base = torch.randn(n_target, 128)
    topk_idx = torch.randint(0, n_target, (n_source, k))
    topk_weight = torch.rand(n_source, k).clamp_min(1e-3)
    topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True)
    confidence = torch.rand(n_source)

    def run(checkpoint: bool):
        module.zero_grad(set_to_none=True)
        source = source_base.clone().requires_grad_(True)
        target = target_base.clone().requires_grad_(True)
        update = module.compute_update_only(
            source_h=source,
            target_h=target,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            confidence=confidence,
            source_context=source.new_empty(0),
            target_context=target.new_empty(0),
            source_chunk_size=3,
            checkpoint_attention=checkpoint,
        )
        loss = update.float().square().mean()
        loss.backward()
        return (
            update.detach(),
            loss.detach(),
            source.grad.detach(),
            target.grad.detach(),
            module.W_Q.weight.grad.detach().clone(),
        )

    plain = run(False)
    checkpointed = run(True)
    diffs = {
        "output": float((plain[0] - checkpointed[0]).abs().max()),
        "loss": float((plain[1] - checkpointed[1]).abs()),
        "source_gradient": float((plain[2] - checkpointed[2]).abs().max()),
        "target_gradient": float((plain[3] - checkpointed[3]).abs().max()),
        "q_weight_gradient": float((plain[4] - checkpointed[4]).abs().max()),
    }
    if max(diffs.values()) > 1e-6:
        raise AssertionError(f"Checkpoint mismatch: {diffs}")
    return {"gate_input_dim": 512, "checkpoint_max_abs_differences": diffs}


def validate_model_flow() -> dict:
    torch.manual_seed(7)
    sizes = {"A": 11, "B": 14, "C": 12}
    features = {
        section: {
            "RNA": torch.randn(n, 10),
            "Protein": torch.randn(n, 8),
        }
        for section, n in sizes.items()
    }
    spatial = {section: torch.rand(n, 2) for section, n in sizes.items()}
    config = get_default_model_config()
    config["training"]["device"] = "cpu"
    config["graph"]["knn_neighbors_spatial"] = 3
    config["graphsage"]["edge_batch_size"] = 7
    config["ot_attention"]["dropout"] = 0.0
    config["ot_attention"]["context_gate_enabled"] = False
    config["uot"]["dynamic_refresh_source"] = "fused"
    model = StageMultiModalModel(config=config, feature_dict=features)
    model.initialize_candidate_sparse_ot_prior(
        features,
        section_order=["A", "B", "C"],
        initial_modality_candidate_k=5,
        candidate_k=6,
        attention_topk=3,
        candidate_backend="blockwise",
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=10,
        bidirectional=True,
    )
    model.train()
    training_output = model(
        feature_dict=features,
        spatial_loc_dict=spatial,
        section_order=["A", "B", "C"],
        training_loss_only=True,
        return_full_outputs=False,
        bidirectional_ot_attention=True,
        ot_attention_source_chunk_size=4,
        checkpoint_ot_attention=True,
    )
    if training_output.get("context_embeddings"):
        raise AssertionError("Context pooling ran in the experiment C training hot path.")
    training_output["losses"]["total_loss"].backward()

    model.eval()
    with torch.no_grad():
        snapshot = model(
            feature_dict=features,
            spatial_loc_dict=spatial,
            section_order=["A", "B", "C"],
            bidirectional_ot_attention=True,
            ot_attention_source_chunk_size=4,
            checkpoint_ot_attention=True,
        )
    embeddings, contexts, diagnostics = model.prepare_ot_prior_refresh(snapshot)
    if contexts is None:
        raise AssertionError("Full eval snapshot did not produce fused spatial contexts.")
    for section, n in sizes.items():
        if tuple(contexts[section].shape) != (n, 128):
            raise AssertionError(f"Bad context shape for {section}.")
        if contexts[section].requires_grad:
            raise AssertionError(f"Context unexpectedly requires grad for {section}.")
        if not torch.allclose(embeddings[section], snapshot["fused_embeddings"][section]):
            raise AssertionError("Dynamic semantic source is not fused.")
    model.update_candidate_sparse_ot_prior(
        embeddings,
        section_order=["A", "B", "C"],
        candidate_k=6,
        attention_topk=3,
        candidate_backend="blockwise",
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=10,
        candidate_source="fused",
        bidirectional=True,
        context_embedding_dict=contexts,
        topology_context_weight=0.2,
    )
    metadata = model.ot_prior[("A", "B")]["metadata"]
    expected = {
        "retrieval_source": "fused",
        "semantic_source": "fused",
        "context_source": "fused_spatial_context",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"Unexpected dynamic OT metadata: {metadata}")
    return {
        "bidirectional_prior_keys": [list(key) for key in sorted(model.ot_prior)],
        "context_shapes": {key: list(value.shape) for key, value in contexts.items()},
        "context_requires_grad": {key: value.requires_grad for key, value in contexts.items()},
        "dynamic_ot_metadata": expected,
        "refresh_diagnostics": diagnostics,
    }


if __name__ == "__main__":
    print(json.dumps({"status": "PASS", "attention": validate_attention(), "flow": validate_model_flow()}, indent=2))
