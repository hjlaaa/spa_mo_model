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


def fusion_input_dim(model, modality_order):
    combo_key = "__".join(modality_order)
    module = model.fusion_modules[combo_key]
    for layer in module.network:
        if isinstance(layer, torch.nn.Linear):
            return layer.in_features
    raise AssertionError(f"No Linear layer found in fusion module {combo_key}.")


def check_two_section_case(case_name, feature_dict, spatial_loc_dict, modality_order):
    config = make_smoke_config()
    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    expected_fusion_input_dim = 128 * len(modality_order)
    actual_fusion_input_dim = fusion_input_dim(model, modality_order)
    if actual_fusion_input_dim != expected_fusion_input_dim:
        raise AssertionError(
            f"{case_name}: fusion input dim {actual_fusion_input_dim}, "
            f"expected {expected_fusion_input_dim}"
        )

    outputs = model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=None,
    )

    expected_spots = {
        section: int(next(value for value in modalities.values() if value is not None).shape[0])
        for section, modalities in feature_dict.items()
    }
    for section, n_spots in expected_spots.items():
        observed_recon_modalities = set(outputs["reconstructions"][section].keys())
        if observed_recon_modalities != set(modality_order):
            raise AssertionError(
                f"{case_name}: {section} recon modalities {sorted(observed_recon_modalities)}, "
                f"expected {sorted(modality_order)}"
            )
        for modality in modality_order:
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

    for unexpected in set(["HE", "RNA", "Protein", "Metabolite"]) - set(modality_order):
        if unexpected in outputs["reconstructions"]["s1"]:
            raise AssertionError(f"{case_name}: unexpected {unexpected} reconstruction")

    prior = outputs["ot_prior"][("s1", "s2")]
    source_n = expected_spots["s1"]
    if tuple(prior["topk_idx"].shape) != (source_n, 10):
        raise AssertionError(f"{case_name}: topk_idx shape {tuple(prior['topk_idx'].shape)}")
    if tuple(prior["topk_weight"].shape) != (source_n, 10):
        raise AssertionError(f"{case_name}: topk_weight shape {tuple(prior['topk_weight'].shape)}")
    if tuple(prior["confidence"].shape) != (source_n,):
        raise AssertionError(f"{case_name}: confidence shape {tuple(prior['confidence'].shape)}")
    if list(prior["modalities_used"]) != list(modality_order):
        raise AssertionError(
            f"{case_name}: OT modalities_used {prior['modalities_used']}, expected {list(modality_order)}"
        )

    assert_finite_loss(outputs, case_name)
    outputs["losses"]["total_loss"].backward()

    print(f"{case_name}: PASS")
    print(f"  modality_order: {list(modality_order)}")
    print(f"  fusion_input_dim: {actual_fusion_input_dim}")
    for section, n_spots in expected_spots.items():
        print(f"  {section} latent/fused/graphsage/final: ({n_spots}, 128)")
    print(f"  ot_prior keys: {list(outputs['ot_prior'].keys())}")
    print(f"  topk_idx: {tuple(prior['topk_idx'].shape)}")
    print(f"  topk_weight: {tuple(prior['topk_weight'].shape)}")
    print(f"  confidence: {tuple(prior['confidence'].shape)}")
    print(f"  total_loss: {float(outputs['losses']['total_loss'].detach().cpu())}")


def expect_value_error(case_name, feature_dict, spatial_loc_dict, expected_text):
    try:
        model = StageMultiModalModel(config=make_smoke_config(), feature_dict=feature_dict)
        model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            processed_data_dict=None,
        )
    except ValueError as exc:
        message = str(exc)
        if expected_text not in message:
            raise AssertionError(
                f"{case_name}: expected error containing {expected_text!r}, got {message!r}"
            ) from exc
        print(f"{case_name}: PASS")
        print(f"  expected ValueError: {message}")
        return
    raise AssertionError(f"{case_name}: expected ValueError but forward succeeded.")


def run_smoke_test() -> None:
    torch.manual_seed(8)
    rng = np.random.default_rng(8)

    rna_protein_feature_dict = {
        "s1": {
            "RNA": torch.randn(100, 50),
            "Protein": torch.randn(100, 50),
        },
        "s2": {
            "RNA": torch.randn(120, 50),
            "Protein": torch.randn(120, 50),
        },
    }
    rna_protein_spatial = {
        "s1": rng.random((100, 2)),
        "s2": rng.random((120, 2)),
    }
    check_two_section_case(
        "RNA_Protein_two_modality",
        rna_protein_feature_dict,
        rna_protein_spatial,
        modality_order=("RNA", "Protein"),
    )

    he_rna_feature_dict = {
        "s1": {
            "HE": torch.randn(80, 50),
            "RNA": torch.randn(80, 50),
        },
        "s2": {
            "HE": torch.randn(90, 50),
            "RNA": torch.randn(90, 50),
        },
    }
    he_rna_spatial = {
        "s1": rng.random((80, 2)),
        "s2": rng.random((90, 2)),
    }
    check_two_section_case(
        "HE_RNA_two_modality",
        he_rna_feature_dict,
        he_rna_spatial,
        modality_order=("HE", "RNA"),
    )

    he_protein_feature_dict = {
        "s1": {
            "HE": torch.randn(80, 50),
            "Protein": torch.randn(80, 50),
        },
        "s2": {
            "HE": torch.randn(90, 50),
            "Protein": torch.randn(90, 50),
        },
    }
    he_protein_spatial = {
        "s1": rng.random((80, 2)),
        "s2": rng.random((90, 2)),
    }
    check_two_section_case(
        "HE_Protein_two_modality",
        he_protein_feature_dict,
        he_protein_spatial,
        modality_order=("HE", "Protein"),
    )

    mousebrain_like_feature_dict = {
        "s1": {
            "HE": torch.randn(100, 50),
            "RNA": torch.randn(100, 50),
            "Metabolite": torch.randn(100, 50),
        },
        "s2": {
            "HE": torch.randn(120, 50),
            "RNA": torch.randn(120, 50),
            "Metabolite": torch.randn(120, 50),
        },
    }
    mousebrain_like_spatial = {
        "s1": rng.random((100, 2)),
        "s2": rng.random((120, 2)),
    }
    check_two_section_case(
        "HE_RNA_Metabolite_three_modality",
        mousebrain_like_feature_dict,
        mousebrain_like_spatial,
        modality_order=("HE", "RNA", "Metabolite"),
    )

    none_he_feature_dict = {
        "s1": {
            "HE": None,
            "RNA": torch.randn(100, 50),
            "Protein": torch.randn(100, 50),
        },
        "s2": {
            "HE": None,
            "RNA": torch.randn(120, 50),
            "Protein": torch.randn(120, 50),
        },
    }
    none_he_spatial = {
        "s1": rng.random((100, 2)),
        "s2": rng.random((120, 2)),
    }
    check_two_section_case(
        "RNA_Protein_with_HE_None",
        none_he_feature_dict,
        none_he_spatial,
        modality_order=("RNA", "Protein"),
    )

    expect_value_error(
        "invalid_single_modality",
        {
            "s1": {"RNA": torch.randn(40, 50)},
            "s2": {"RNA": torch.randn(45, 50)},
        },
        {
            "s1": rng.random((40, 2)),
            "s2": rng.random((45, 2)),
        },
        "at least two",
    )
    expect_value_error(
        "invalid_four_modality",
        {
            "s1": {
                "HE": torch.randn(40, 50),
                "RNA": torch.randn(40, 50),
                "Protein": torch.randn(40, 50),
                "Metabolite": torch.randn(40, 50),
            },
            "s2": {
                "HE": torch.randn(45, 50),
                "RNA": torch.randn(45, 50),
                "Protein": torch.randn(45, 50),
                "Metabolite": torch.randn(45, 50),
            },
        },
        {
            "s1": rng.random((40, 2)),
            "s2": rng.random((45, 2)),
        },
        "supported two- or three-modality set",
    )
    expect_value_error(
        "invalid_RNA_Protein_Metabolite",
        {
            "s1": {
                "RNA": torch.randn(40, 50),
                "Protein": torch.randn(40, 50),
                "Metabolite": torch.randn(40, 50),
            },
            "s2": {
                "RNA": torch.randn(45, 50),
                "Protein": torch.randn(45, 50),
                "Metabolite": torch.randn(45, 50),
            },
        },
        {
            "s1": rng.random((40, 2)),
            "s2": rng.random((45, 2)),
        },
        "supported two- or three-modality set",
    )
    expect_value_error(
        "invalid_inconsistent_section_modalities",
        {
            "s1": {
                "RNA": torch.randn(40, 50),
                "Protein": torch.randn(40, 50),
            },
            "s2": {
                "HE": torch.randn(45, 50),
                "RNA": torch.randn(45, 50),
                "Protein": torch.randn(45, 50),
            },
        },
        {
            "s1": rng.random((40, 2)),
            "s2": rng.random((45, 2)),
        },
        "same observed modality set",
    )
    expect_value_error(
        "invalid_spot_count_mismatch",
        {
            "s1": {
                "RNA": torch.randn(100, 50),
                "Protein": torch.randn(99, 50),
            },
            "s2": {
                "RNA": torch.randn(120, 50),
                "Protein": torch.randn(120, 50),
            },
        },
        {
            "s1": rng.random((100, 2)),
            "s2": rng.random((120, 2)),
        },
        "same spot count",
    )
    expect_value_error(
        "invalid_feature_dim_mismatch",
        {
            "s1": {
                "RNA": torch.randn(100, 50),
                "Protein": torch.randn(100, 50),
            },
            "s2": {
                "RNA": torch.randn(120, 60),
                "Protein": torch.randn(120, 50),
            },
        },
        {
            "s1": rng.random((100, 2)),
            "s2": rng.random((120, 2)),
        },
        "inconsistent input dimensions",
    )


def main() -> None:
    args = parse_args()
    run_smoke_test()


if __name__ == "__main__":
    main()
