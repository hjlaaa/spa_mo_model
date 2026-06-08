#!/usr/bin/env python3
"""Smoke test entry for the V2 stage multimodal model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.stage_model import StageMultiModalModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run StageMultiModalModel checks.")
    parser.add_argument("--smoke_test", action="store_true", help="Run synthetic V2 smoke tests.")
    return parser.parse_args()


def make_smoke_config():
    config = get_default_model_config()
    config["training"]["device"] = "cpu"
    config["uot"]["max_iter"] = 100
    config["uot"]["check_every"] = 10
    config["uot"]["tol"] = 1e-5
    return config


def assert_finite_loss(outputs, case_name):
    for key, value in outputs["losses"].items():
        if not torch.isfinite(value).item():
            raise AssertionError(f"{case_name}: {key} is not finite: {value}")


def check_two_stage_case(case_name, feature_dict, spatial_loc_dict, third_modality):
    config = make_smoke_config()
    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    outputs = model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=None,
    )

    for section, n_spots in {"s1": 80, "s2": 90}.items():
        for modality in ["HE", "RNA", third_modality]:
            latent_shape = tuple(outputs["latent_dict"][section][modality].shape)
            if latent_shape != (n_spots, 128):
                raise AssertionError(f"{case_name}: {section} {modality} latent {latent_shape}")
            recon_shape = tuple(outputs["reconstructions"][section][modality].shape)
            target_shape = tuple(feature_dict[section][modality].shape)
            if recon_shape != target_shape:
                raise AssertionError(
                    f"{case_name}: {section} {modality} recon {recon_shape}, expected {target_shape}"
                )

        for key in ["fused_embeddings", "graphsage_embeddings", "final_embeddings"]:
            shape = tuple(outputs[key][section].shape)
            if shape != (n_spots, 128):
                raise AssertionError(f"{case_name}: {section} {key} shape {shape}")

        graph = outputs["spatial_graph_dict"][section]
        if graph["edge_index"].shape[0] != 2:
            raise AssertionError(f"{case_name}: {section} edge_index shape {graph['edge_index'].shape}")
        if graph["edge_weight"].ndim != 1:
            raise AssertionError(f"{case_name}: {section} edge_weight shape {graph['edge_weight'].shape}")

    prior = outputs["ot_prior"][("s1", "s2")]
    if tuple(prior["topk_idx"].shape) != (80, 10):
        raise AssertionError(f"{case_name}: topk_idx shape {tuple(prior['topk_idx'].shape)}")
    if tuple(prior["topk_weight"].shape) != (80, 10):
        raise AssertionError(f"{case_name}: topk_weight shape {tuple(prior['topk_weight'].shape)}")
    if tuple(prior["confidence"].shape) != (80,):
        raise AssertionError(f"{case_name}: confidence shape {tuple(prior['confidence'].shape)}")

    assert_finite_loss(outputs, case_name)
    outputs["losses"]["total_loss"].backward()

    print(f"{case_name}: PASS")
    print(f"  s1 latent/fused/graphsage/final: (80, 128)")
    print(f"  s2 latent/fused/graphsage/final: (90, 128)")
    print(f"  ot_prior keys: {list(outputs['ot_prior'].keys())}")
    print(f"  topk_idx: {tuple(prior['topk_idx'].shape)}")
    print(f"  topk_weight: {tuple(prior['topk_weight'].shape)}")
    print(f"  confidence: {tuple(prior['confidence'].shape)}")
    print(f"  total_loss: {float(outputs['losses']['total_loss'].detach().cpu())}")


def check_three_stage_case():
    torch.manual_seed(83)
    rng = np.random.default_rng(83)
    feature_dict = {
        "s1": {
            "HE": torch.randn(50, 50),
            "RNA": torch.randn(50, 50),
            "Protein": torch.randn(50, 20),
        },
        "s2": {
            "HE": torch.randn(55, 50),
            "RNA": torch.randn(55, 50),
            "Protein": torch.randn(55, 20),
        },
        "s3": {
            "HE": torch.randn(60, 50),
            "RNA": torch.randn(60, 50),
            "Protein": torch.randn(60, 20),
        },
    }
    spatial_loc_dict = {
        "s1": rng.random((50, 2)),
        "s2": rng.random((55, 2)),
        "s3": rng.random((60, 2)),
    }
    config = make_smoke_config()
    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    outputs = model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        section_order=["s1", "s2", "s3"],
        epoch=1,
    )
    expected_keys = {("s1", "s2"), ("s2", "s3")}
    if set(outputs["ot_prior"].keys()) != expected_keys:
        raise AssertionError(f"three_stage: OT keys {outputs['ot_prior'].keys()}")
    if not torch.allclose(outputs["final_embeddings"]["s3"], outputs["graphsage_embeddings"]["s3"]):
        raise AssertionError("three_stage: last section should not receive OT attention")
    for section, n_spots in {"s1": 50, "s2": 55, "s3": 60}.items():
        if tuple(outputs["final_embeddings"][section].shape) != (n_spots, 128):
            raise AssertionError(f"three_stage: {section} final shape mismatch")
    assert_finite_loss(outputs, "three_stage")
    outputs["losses"]["total_loss"].backward()
    print("three_stage_forward_attention: PASS")
    print(f"  ot_prior keys: {list(outputs['ot_prior'].keys())}")
    print("  s3 final equals graphsage output: True")


def run_smoke_test() -> None:
    torch.manual_seed(8)
    rng = np.random.default_rng(8)

    protein_feature_dict = {
        "s1": {
            "HE": torch.randn(80, 50),
            "RNA": torch.randn(80, 50),
            "Protein": torch.randn(80, 20),
        },
        "s2": {
            "HE": torch.randn(90, 50),
            "RNA": torch.randn(90, 50),
            "Protein": torch.randn(90, 20),
        },
    }
    protein_spatial = {
        "s1": rng.random((80, 2)),
        "s2": rng.random((90, 2)),
    }
    check_two_stage_case(
        "HE_RNA_Protein_two_stage",
        protein_feature_dict,
        protein_spatial,
        third_modality="Protein",
    )

    metabolite_feature_dict = {
        "s1": {
            "HE": torch.randn(80, 50),
            "RNA": torch.randn(80, 50),
            "Metabolite": torch.randn(80, 50),
        },
        "s2": {
            "HE": torch.randn(90, 50),
            "RNA": torch.randn(90, 50),
            "Metabolite": torch.randn(90, 50),
        },
    }
    metabolite_spatial = {
        "s1": rng.random((80, 2)),
        "s2": rng.random((90, 2)),
    }
    check_two_stage_case(
        "HE_RNA_Metabolite_two_stage",
        metabolite_feature_dict,
        metabolite_spatial,
        third_modality="Metabolite",
    )

    check_three_stage_case()


def main() -> None:
    args = parse_args()
    if not args.smoke_test:
        raise SystemExit("Pass --smoke_test to run the synthetic model check.")
    run_smoke_test()


if __name__ == "__main__":
    main()
