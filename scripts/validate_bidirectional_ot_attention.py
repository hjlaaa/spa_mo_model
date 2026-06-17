#!/usr/bin/env python3
"""Lightweight correctness checks for bidirectional OT-guided attention."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.stage_model import StageMultiModalModel


def _assert_topk_prior(prior, n_source: int, n_target: int, attention_topk: int) -> dict[str, float]:
    topk_idx = prior["topk_idx"]
    topk_weight = prior["topk_weight"]
    confidence = prior["confidence"]
    row_mass = prior["row_mass"]
    raw_topk_mass = prior["raw_topk_mass"]

    assert tuple(topk_idx.shape) == (n_source, attention_topk)
    assert tuple(topk_weight.shape) == (n_source, attention_topk)
    assert int(topk_idx.min().item()) >= 0
    assert int(topk_idx.max().item()) < n_target
    assert torch.isfinite(topk_weight).all()
    assert torch.isfinite(confidence).all()
    assert torch.isfinite(row_mass).all()
    assert torch.isfinite(raw_topk_mass).all()
    assert (topk_weight >= 0).all()
    assert tuple(confidence.shape) == (n_source,)
    assert tuple(row_mass.shape) == (n_source,)
    assert tuple(raw_topk_mass.shape) == (n_source,)

    row_sum = topk_weight.sum(dim=1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4)
    return {
        "idx_min": int(topk_idx.min().item()),
        "idx_max": int(topk_idx.max().item()),
        "row_sum_min": float(row_sum.min().item()),
        "row_sum_max": float(row_sum.max().item()),
        "confidence_min": float(confidence.min().item()),
        "confidence_max": float(confidence.max().item()),
        "row_mass_min": float(row_mass.min().item()),
        "row_mass_max": float(row_mass.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_backend", choices=["blockwise", "faiss_ivf", "faiss_flat"], default="blockwise")
    parser.add_argument("--faiss_nlist", type=int, default=16)
    parser.add_argument("--faiss_nprobe", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(7)
    n_crc003 = 37
    n_crc006 = 53
    feature_dim = 16
    attention_topk = 5

    feature_dict = {
        "CRC_003": {
            "RNA": torch.randn(n_crc003, feature_dim),
            "Protein": torch.randn(n_crc003, feature_dim),
        },
        "CRC_006": {
            "RNA": torch.randn(n_crc006, feature_dim),
            "Protein": torch.randn(n_crc006, feature_dim),
        },
    }
    spatial_loc_dict = {
        "CRC_003": torch.rand(n_crc003, 2),
        "CRC_006": torch.rand(n_crc006, 2),
    }
    section_order = ["CRC_003", "CRC_006"]

    config = get_default_model_config()
    config["training"]["device"] = "cpu"
    config["graph"]["k_neighbors"] = 4
    config["graphsage"]["edge_batch_size"] = 64
    config["uot"]["enabled"] = True
    config["ot_attention"]["enabled"] = True

    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    model.initialize_candidate_sparse_ot_prior(
        feature_dict,
        section_order=section_order,
        initial_modality_candidate_k=8,
        candidate_k=12,
        attention_topk=attention_topk,
        candidate_backend=args.candidate_backend,
        faiss_nlist=int(args.faiss_nlist),
        faiss_nprobe=int(args.faiss_nprobe),
        faiss_device="cpu",
        epsilon=0.05,
        tau_a=1.0,
        tau_b=1.0,
        max_iter=20,
        bidirectional=True,
    )

    expected_keys = {("CRC_003", "CRC_006"), ("CRC_006", "CRC_003")}
    actual_keys = set(model.ot_prior.keys())
    assert expected_keys.issubset(actual_keys), actual_keys

    forward_stats = _assert_topk_prior(
        model.ot_prior[("CRC_003", "CRC_006")],
        n_source=n_crc003,
        n_target=n_crc006,
        attention_topk=attention_topk,
    )
    reverse_stats = _assert_topk_prior(
        model.ot_prior[("CRC_006", "CRC_003")],
        n_source=n_crc006,
        n_target=n_crc003,
        attention_topk=attention_topk,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            section_order=section_order,
            epoch=0,
            decoder_chunk_size=16,
            ot_attention_source_chunk_size=16,
            bidirectional_ot_attention=True,
        )

    total_loss = outputs["losses"]["total_loss"]
    assert torch.isfinite(total_loss).item()
    assert tuple(outputs["final_embeddings"]["CRC_003"].shape) == (n_crc003, 128)
    assert tuple(outputs["final_embeddings"]["CRC_006"].shape) == (n_crc006, 128)

    max_diff_003 = (
        outputs["final_embeddings"]["CRC_003"] - outputs["graphsage_embeddings"]["CRC_003"]
    ).abs().max().item()
    max_diff_006 = (
        outputs["final_embeddings"]["CRC_006"] - outputs["graphsage_embeddings"]["CRC_006"]
    ).abs().max().item()
    assert max_diff_003 > 1e-8
    assert max_diff_006 > 1e-8

    summary = {
        "status": "PASS",
        "candidate_backend": args.candidate_backend,
        "prior_keys": [list(key) for key in sorted(actual_keys)],
        "CRC_003_to_CRC_006": forward_stats,
        "CRC_006_to_CRC_003": reverse_stats,
        "total_loss": float(total_loss.item()),
        "final_embedding_shapes": {
            section: list(tensor.shape)
            for section, tensor in outputs["final_embeddings"].items()
        },
        "final_vs_graphsage_max_abs_diff": {
            "CRC_003": float(max_diff_003),
            "CRC_006": float(max_diff_006),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
