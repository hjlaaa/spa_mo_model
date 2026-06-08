#!/usr/bin/env python3
"""Temporary verification script for Model Stage V2.

This file is intentionally placed under tmp_verification/ and can be deleted
after verification. It does not participate in normal project execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.linkage_construction import (
    sparsify_coupling_topk,
    unbalanced_sinkhorn,
)
from model.stage_model import StageMultiModalModel, should_update_ot
from model.utils import compute_spatial_knn_graph_with_weights, cosine_cost_matrix


def make_test_config():
    config = get_default_model_config()
    config["training"]["device"] = "cpu"
    config["uot"]["max_iter"] = 80
    config["uot"]["check_every"] = 10
    config["uot"]["tol"] = 1e-5
    return config


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_shape(tensor, shape, message):
    assert_true(tuple(tensor.shape) == tuple(shape), f"{message}: got {tuple(tensor.shape)}, expected {shape}")


def assert_finite(tensor, message):
    assert_true(torch.isfinite(tensor).all().item(), message)


def check_default_config():
    cfg = get_default_model_config()
    checks = {
        "graphsage.enabled": cfg["graphsage"]["enabled"] is True,
        "graphsage.input_dim": cfg["graphsage"]["input_dim"] == 128,
        "graphsage.output_dim": cfg["graphsage"]["output_dim"] == 128,
        "graphsage.num_layers": cfg["graphsage"]["num_layers"] == 1,
        "graphsage.dropout": cfg["graphsage"]["dropout"] == 0.1,
        "graphsage.activation": cfg["graphsage"]["activation"] == "GELU",
        "graphsage.norm": cfg["graphsage"]["norm"] == "LayerNorm",
        "graphsage.residual": cfg["graphsage"]["residual"] is True,
        "graphsage.use_distance_weight": cfg["graphsage"]["use_distance_weight"] is True,
        "uot.enabled": cfg["uot"]["enabled"] is True,
        "uot.epsilon_init": cfg["uot"]["epsilon_init"] == 0.08,
        "uot.epsilon_update": cfg["uot"]["epsilon_update"] == 0.05,
        "uot.tau_a": cfg["uot"]["tau_a"] == 1.0,
        "uot.tau_b": cfg["uot"]["tau_b"] == 1.0,
        "uot.max_iter": cfg["uot"]["max_iter"] == 1000,
        "uot.tol": cfg["uot"]["tol"] == 1e-6,
        "uot.update_interval": cfg["uot"]["update_interval"] == 20,
        "uot.topk": cfg["uot"]["topk"] == 10,
        "uot.use_momentum": cfg["uot"]["use_momentum"] is False,
        "uot.normalize_total_mass": cfg["uot"]["normalize_total_mass"] is True,
        "ot_attention.enabled": cfg["ot_attention"]["enabled"] is True,
        "ot_attention.direction": cfg["ot_attention"]["direction"] == "forward",
        "ot_attention.d_attn": cfg["ot_attention"]["d_attn"] == 128,
        "ot_attention.beta": cfg["ot_attention"]["beta"] == 0.2,
        "ot_attention.gate": cfg["ot_attention"]["gate"] == "scalar",
        "ot_attention.use_confidence": cfg["ot_attention"]["use_confidence"] is True,
        "ot_attention.residual": cfg["ot_attention"]["residual"] is True,
        "ot_attention.norm": cfg["ot_attention"]["norm"] == "LayerNorm",
        "decoder.enabled": cfg["decoder"]["enabled"] is True,
        "decoder.hidden_dim": cfg["decoder"]["hidden_dim"] == 128,
        "decoder.activation": cfg["decoder"]["activation"] == "GELU",
        "decoder.dropout": cfg["decoder"]["dropout"] == 0.1,
        "reconstruction.enabled": cfg["reconstruction"]["enabled"] is True,
        "reconstruction.loss": cfg["reconstruction"]["loss"] == "mse",
        "loss.lambda_contrast": "lambda_contrast" in cfg["loss"],
        "loss.lambda_reconstruction": "lambda_reconstruction" in cfg["loss"],
        "loss.use_ot_loss": cfg["loss"]["use_ot_loss"] is False,
        "loss.use_spatial_smooth_loss": cfg["loss"]["use_spatial_smooth_loss"] is False,
        "loss.use_gate_regularization": cfg["loss"]["use_gate_regularization"] is False,
    }
    failures = [name for name, ok in checks.items() if not ok]
    assert_true(not failures, f"default config check failures: {failures}")
    return checks


def check_graph_and_uot_primitives():
    rng = np.random.default_rng(101)
    coords = rng.random((16, 2))
    edge_index, edge_weight = compute_spatial_knn_graph_with_weights(coords, k=5)
    assert_shape(edge_index, (2, edge_index.shape[1]), "edge_index shape")
    assert_shape(edge_weight, (edge_index.shape[1],), "edge_weight shape")
    assert_true(edge_index.shape[1] > 16, "edge graph should contain neighbors beyond self-loops")
    for source in range(coords.shape[0]):
        mask = edge_index[0] == source
        row_sum = edge_weight[mask].sum()
        assert_true(torch.allclose(row_sum, torch.tensor(1.0), atol=1e-5), f"edge weights not row-normalized for source {source}")

    x = torch.randn(11, 8)
    y = torch.randn(13, 8)
    cost = cosine_cost_matrix(x, y)
    assert_shape(cost, (11, 13), "cosine cost shape")
    assert_true((cost >= 0).all().item() and (cost <= 2).all().item(), "cosine cost not clipped to [0,2]")
    P = unbalanced_sinkhorn(cost, epsilon=0.05, max_iter=50)
    assert_shape(P, (11, 13), "sinkhorn coupling shape")
    assert_finite(P, "sinkhorn coupling has non-finite values")
    sparse = sparsify_coupling_topk(P / (P.sum() + 1e-8), topk=10)
    assert_shape(sparse["topk_idx"], (11, 10), "topk_idx shape")
    assert_shape(sparse["topk_weight"], (11, 10), "topk_weight shape")
    assert_shape(sparse["confidence"], (11,), "confidence shape")
    assert_true(torch.allclose(sparse["topk_weight"].sum(dim=1), torch.ones(11), atol=1e-5), "topk weights are not row-normalized")
    return {
        "edge_index_shape": list(edge_index.shape),
        "edge_weight_shape": list(edge_weight.shape),
        "uot_shape": list(P.shape),
        "topk_shape": list(sparse["topk_idx"].shape),
    }


def run_two_stage_case(case_name, third_modality, n1, n2, d3):
    torch.manual_seed(11 + n1)
    rng = np.random.default_rng(11 + n2)
    feature_dict = {
        "s1": {
            "HE": torch.randn(n1, 50),
            "RNA": torch.randn(n1, 50),
            third_modality: torch.randn(n1, d3),
        },
        "s2": {
            "HE": torch.randn(n2, 50),
            "RNA": torch.randn(n2, 50),
            third_modality: torch.randn(n2, d3),
        },
    }
    spatial_loc_dict = {
        "s1": rng.random((n1, 2)),
        "s2": rng.random((n2, 2)),
    }
    model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
    outputs = model(feature_dict, spatial_loc_dict, section_order=["s1", "s2"], epoch=1)
    for section, n in {"s1": n1, "s2": n2}.items():
        for modality in ["HE", "RNA", third_modality]:
            assert_shape(outputs["latent_dict"][section][modality], (n, 128), f"{case_name} {section} {modality} latent")
            target_dim = feature_dict[section][modality].shape[1]
            assert_shape(outputs["reconstructions"][section][modality], (n, target_dim), f"{case_name} {section} {modality} reconstruction")
        assert_shape(outputs["fused_embeddings"][section], (n, 128), f"{case_name} {section} fused")
        assert_shape(outputs["graphsage_embeddings"][section], (n, 128), f"{case_name} {section} graphsage")
        assert_shape(outputs["final_embeddings"][section], (n, 128), f"{case_name} {section} final")

    prior = outputs["ot_prior"][("s1", "s2")]
    assert_shape(prior["topk_idx"], (n1, 10), f"{case_name} topk_idx")
    assert_shape(prior["topk_weight"], (n1, 10), f"{case_name} topk_weight")
    assert_shape(prior["confidence"], (n1,), f"{case_name} confidence")
    for key in ["total_loss", "crossview_loss", "reconstruction_loss"]:
        assert_finite(outputs["losses"][key], f"{case_name} {key} non-finite")
    outputs["losses"]["total_loss"].backward()
    assert_true(any(p.grad is not None for p in model.parameters()), f"{case_name} backward produced no gradients")
    return {
        "latent_s1": [n1, 128],
        "fused_s1": list(outputs["fused_embeddings"]["s1"].shape),
        "graphsage_s1": list(outputs["graphsage_embeddings"]["s1"].shape),
        "final_s1": list(outputs["final_embeddings"]["s1"].shape),
        "topk_idx": list(prior["topk_idx"].shape),
        "topk_weight": list(prior["topk_weight"].shape),
        "confidence": list(prior["confidence"].shape),
        "losses": {
            key: float(value.detach().cpu())
            for key, value in outputs["losses"].items()
        },
    }


def run_three_stage_case():
    torch.manual_seed(333)
    rng = np.random.default_rng(333)
    feature_dict = {
        "s1": {"HE": torch.randn(45, 50), "RNA": torch.randn(45, 50), "Protein": torch.randn(45, 20)},
        "s2": {"HE": torch.randn(48, 50), "RNA": torch.randn(48, 50), "Protein": torch.randn(48, 20)},
        "s3": {"HE": torch.randn(52, 50), "RNA": torch.randn(52, 50), "Protein": torch.randn(52, 20)},
    }
    spatial_loc_dict = {
        "s1": rng.random((45, 2)),
        "s2": rng.random((48, 2)),
        "s3": rng.random((52, 2)),
    }
    model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
    outputs = model(feature_dict, spatial_loc_dict, section_order=["s1", "s2", "s3"], epoch=1)
    expected = {("s1", "s2"), ("s2", "s3")}
    assert_true(set(outputs["ot_prior"].keys()) == expected, f"three-stage OT keys mismatch: {outputs['ot_prior'].keys()}")
    assert_true(torch.allclose(outputs["final_embeddings"]["s3"], outputs["graphsage_embeddings"]["s3"], atol=1e-6), "last stage final embedding should equal GraphSAGE embedding")
    outputs["losses"]["total_loss"].backward()
    return {
        "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
        "s3_final_equals_graphsage": bool(torch.allclose(outputs["final_embeddings"]["s3"], outputs["graphsage_embeddings"]["s3"], atol=1e-6)),
    }


def run_ot_update_case():
    torch.manual_seed(44)
    rng = np.random.default_rng(44)
    feature_dict = {
        "s1": {"HE": torch.randn(30, 50), "RNA": torch.randn(30, 50), "Protein": torch.randn(30, 20)},
        "s2": {"HE": torch.randn(35, 50), "RNA": torch.randn(35, 50), "Protein": torch.randn(35, 20)},
    }
    spatial_loc_dict = {"s1": rng.random((30, 2)), "s2": rng.random((35, 2))}
    model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
    model.initialize_ot_prior(feature_dict, section_order=["s1", "s2"])
    old_topk = model.ot_prior[("s1", "s2")]["topk_idx"].clone()
    outputs = model(feature_dict, spatial_loc_dict, section_order=["s1", "s2"], epoch=1)
    with torch.no_grad():
        updated = model.update_ot_prior(outputs["final_embeddings"], section_order=["s1", "s2"])
    new_topk = model.ot_prior[("s1", "s2")]["topk_idx"]
    assert_shape(new_topk, (30, 10), "updated topk_idx")
    assert_true(model.ot_prior is updated, "model.ot_prior was not replaced by update result")
    assert_true(not updated[("s1", "s2")]["topk_weight"].requires_grad, "updated topk_weight should not require grad")
    assert_true(should_update_ot(20, 20) and not should_update_ot(19, 20), "should_update_ot interval logic failed")
    return {
        "old_topk_shape": list(old_topk.shape),
        "new_topk_shape": list(new_topk.shape),
        "topk_changed": bool(not torch.equal(old_topk, new_topk)),
        "requires_grad": bool(updated[("s1", "s2")]["topk_weight"].requires_grad),
    }


def expect_error(name, func, expected_text):
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - verification should capture exact message.
        message = str(exc)
        assert_true(expected_text in message, f"{name}: expected {expected_text!r} in {message!r}")
        return {"raised": type(exc).__name__, "message": message}
    raise AssertionError(f"{name}: expected an error but none was raised")


def run_error_tests():
    rng = np.random.default_rng(5)

    def missing_third():
        feature_dict = {"s1": {"HE": torch.randn(20, 50), "RNA": torch.randn(20, 50)}}
        StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)

    def protein_and_metabolite():
        feature_dict = {
            "s1": {
                "HE": torch.randn(20, 50),
                "RNA": torch.randn(20, 50),
                "Protein": torch.randn(20, 20),
                "Metabolite": torch.randn(20, 50),
            }
        }
        StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)

    def missing_spatial():
        feature_dict = {"s1": {"HE": torch.randn(20, 50), "RNA": torch.randn(20, 50), "Protein": torch.randn(20, 20)}}
        model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
        model(feature_dict, {})

    def spot_mismatch():
        feature_dict = {"s1": {"HE": torch.randn(20, 50), "RNA": torch.randn(19, 50), "Protein": torch.randn(20, 20)}}
        model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
        model(feature_dict, {"s1": rng.random((20, 2))})

    def bad_section_order():
        feature_dict = {"s1": {"HE": torch.randn(20, 50), "RNA": torch.randn(20, 50), "Protein": torch.randn(20, 20)}}
        model = StageMultiModalModel(config=make_test_config(), feature_dict=feature_dict)
        model(feature_dict, {"s1": rng.random((20, 2))}, section_order=["s2"])

    return {
        "missing_third_modality": expect_error("missing_third_modality", missing_third, "supported sets"),
        "protein_and_metabolite": expect_error("protein_and_metabolite", protein_and_metabolite, "supported sets"),
        "missing_spatial": expect_error("missing_spatial", missing_spatial, "Missing spatial coordinates"),
        "spot_mismatch": expect_error("spot_mismatch", spot_mismatch, "same spot count"),
        "bad_section_order": expect_error("bad_section_order", bad_section_order, "section_order contains sections"),
    }


def main():
    results = {
        "config": check_default_config(),
        "graph_and_uot_primitives": check_graph_and_uot_primitives(),
        "two_stage_HE_RNA_Protein": run_two_stage_case("HE_RNA_Protein", "Protein", 80, 90, 20),
        "two_stage_HE_RNA_Metabolite": run_two_stage_case("HE_RNA_Metabolite", "Metabolite", 70, 85, 50),
        "three_stage_attention": run_three_stage_case(),
        "ot_update": run_ot_update_case(),
        "error_tests": run_error_tests(),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("MODEL_STAGE_V2_VERIFICATION: PASS_NUMERICAL_TESTS")


if __name__ == "__main__":
    main()
