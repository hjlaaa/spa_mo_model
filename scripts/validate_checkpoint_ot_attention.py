#!/usr/bin/env python3
"""Validate OT-guided attention activation checkpointing on synthetic data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.stage_model import StageMultiModalModel


def clone_prior(prior: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    cloned: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in prior.items():
        cloned[key] = {}
        for item_key, item_value in value.items():
            if isinstance(item_value, torch.Tensor):
                cloned[key][item_key] = item_value.detach().clone()
            else:
                cloned[key][item_key] = copy.deepcopy(item_value)
    return cloned


def make_config(device: str) -> dict[str, Any]:
    config = get_default_model_config()
    config["training"]["device"] = device
    config["graph"]["knn_neighbors_spatial"] = 4
    config["encoder"]["dropout"] = 0.0
    config["fusion"]["dropout"] = 0.0
    config["graphsage"]["dropout"] = 0.0
    config["graphsage"]["edge_batch_size"] = 64
    config["ot_attention"]["dropout"] = 0.0
    config["decoder"]["dropout"] = 0.0
    config["uot"]["enabled"] = True
    config["ot_attention"]["enabled"] = True
    return config


def make_synthetic_inputs(seed: int, n_left: int = 37, n_right: int = 53, feature_dim: int = 16):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature_dict = {
        "CRC_003": {
            "RNA": torch.randn(n_left, feature_dim, generator=generator),
            "Protein": torch.randn(n_left, feature_dim, generator=generator),
        },
        "CRC_006": {
            "RNA": torch.randn(n_right, feature_dim, generator=generator),
            "Protein": torch.randn(n_right, feature_dim, generator=generator),
        },
    }
    spatial_loc_dict = {
        "CRC_003": torch.rand(n_left, 2, generator=generator),
        "CRC_006": torch.rand(n_right, 2, generator=generator),
    }
    return feature_dict, spatial_loc_dict


def max_grad_diff(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    max_diff = 0.0
    for (_, param_a), (_, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if param_a.grad is None and param_b.grad is None:
            continue
        if param_a.grad is None or param_b.grad is None:
            return float("inf")
        max_diff = max(max_diff, float((param_a.grad - param_b.grad).abs().max().item()))
    return max_diff


def compare_priors(
    prior_a: dict[tuple[str, str], dict[str, Any]],
    prior_b: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if set(prior_a.keys()) != set(prior_b.keys()):
        raise AssertionError(f"Prior keys differ: {set(prior_a.keys())} vs {set(prior_b.keys())}")

    stats: dict[str, Any] = {}
    for key in sorted(prior_a.keys()):
        a = prior_a[key]
        b = prior_b[key]
        direction = f"{key[0]}_to_{key[1]}"
        if not torch.equal(a["topk_idx"], b["topk_idx"]):
            raise AssertionError(f"topk_idx changed for {direction}.")
        if not torch.allclose(a["topk_weight"], b["topk_weight"], atol=0.0, rtol=0.0):
            raise AssertionError(f"topk_weight changed for {direction}.")
        row_sum = a["topk_weight"].sum(dim=1)
        stats[direction] = {
            "topk_shape": list(a["topk_idx"].shape),
            "row_sum_min": float(row_sum.min().item()),
            "row_sum_max": float(row_sum.max().item()),
            "row_sum_max_abs_diff_from_1": float((row_sum - 1.0).abs().max().item()),
        }
    return stats


def run_case(bidirectional: bool, device: str, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    feature_dict, spatial_loc_dict = make_synthetic_inputs(seed)
    section_order = ["CRC_003", "CRC_006"]
    config = make_config(device)

    base_model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    base_model.initialize_candidate_sparse_ot_prior(
        feature_dict,
        section_order=section_order,
        initial_modality_candidate_k=8,
        candidate_k=12,
        attention_topk=5,
        candidate_backend="blockwise",
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=20,
        bidirectional=bidirectional,
    )
    state_dict = copy.deepcopy(base_model.state_dict())
    prior = clone_prior(base_model.ot_prior or {})

    plain_model = StageMultiModalModel(config=copy.deepcopy(config), feature_dict=feature_dict)
    checkpoint_model = StageMultiModalModel(config=copy.deepcopy(config), feature_dict=feature_dict)
    plain_model.load_state_dict(state_dict)
    checkpoint_model.load_state_dict(state_dict)
    plain_model.ot_prior = clone_prior(prior)
    checkpoint_model.ot_prior = clone_prior(prior)
    plain_model.train()
    checkpoint_model.train()

    common_kwargs = {
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "section_order": section_order,
        "epoch": 1,
        "decoder_chunk_size": 17,
        "ot_attention_source_chunk_size": 13,
        "bidirectional_ot_attention": bidirectional,
    }
    plain_outputs = plain_model(**common_kwargs, checkpoint_ot_attention=False)
    checkpoint_outputs = checkpoint_model(**common_kwargs, checkpoint_ot_attention=True)

    plain_loss = plain_outputs["losses"]["total_loss"].float()
    checkpoint_loss = checkpoint_outputs["losses"]["total_loss"].float()
    if not bool(torch.isfinite(plain_loss).item()):
        raise AssertionError("Plain total_loss is not finite.")
    if not bool(torch.isfinite(checkpoint_loss).item()):
        raise AssertionError("Checkpoint total_loss is not finite.")

    final_embedding_max_abs_diff: dict[str, float] = {}
    final_embedding_shapes: dict[str, list[int]] = {}
    for section in section_order:
        plain_embedding = plain_outputs["final_embeddings"][section]
        checkpoint_embedding = checkpoint_outputs["final_embeddings"][section]
        if plain_embedding.shape != checkpoint_embedding.shape:
            raise AssertionError(f"{section} final embedding shapes differ.")
        final_embedding_shapes[section] = list(plain_embedding.shape)
        final_embedding_max_abs_diff[section] = float(
            (plain_embedding - checkpoint_embedding).abs().max().item()
        )

    plain_loss.backward()
    checkpoint_loss.backward()
    grad_max_abs_diff = max_grad_diff(plain_model, checkpoint_model)
    prior_stats = compare_priors(plain_model.ot_prior or {}, checkpoint_model.ot_prior or {})

    loss_abs_diff = float((plain_loss.detach() - checkpoint_loss.detach()).abs().item())
    if loss_abs_diff > 1e-5:
        raise AssertionError(f"Loss mismatch with dropout=0: {loss_abs_diff}")
    if max(final_embedding_max_abs_diff.values()) > 1e-5:
        raise AssertionError(f"Final embedding mismatch: {final_embedding_max_abs_diff}")
    if grad_max_abs_diff > 1e-5:
        raise AssertionError(f"Gradient mismatch: {grad_max_abs_diff}")

    return {
        "bidirectional_ot_attention": bool(bidirectional),
        "plain_loss": float(plain_loss.detach().cpu().item()),
        "checkpoint_loss": float(checkpoint_loss.detach().cpu().item()),
        "loss_abs_diff": loss_abs_diff,
        "final_embedding_shapes": final_embedding_shapes,
        "final_embedding_max_abs_diff": final_embedding_max_abs_diff,
        "grad_max_abs_diff": grad_max_abs_diff,
        "prior_keys": [list(key) for key in sorted(prior.keys())],
        "prior_stats": prior_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")

    summary = {
        "status": "PASS",
        "device": args.device,
        "seed": int(args.seed),
        "dropout": 0.0,
        "cases": {
            "single_direction": run_case(False, args.device, int(args.seed)),
            "bidirectional": run_case(True, args.device, int(args.seed)),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
