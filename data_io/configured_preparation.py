"""Explicit configured input preparation; no dataset guessing or training tasks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from model.configure import reject_unsupported_preprocess_config
from .datasets import build_mousebrain_section, build_cosie_data_dict, restore_section_keys, preprocess_multisection_cosie_style
from .preprocessing import load_cosie_style_data
from .paired_preparation import PreparedDataset


@dataclass
class MouseBrainInput:
    """Existing UNI in RNA obsm plus RNA/Metabolite; original nested schema."""
    config: Mapping[str, Any]
    max_spots: int | None = None


def subset_section_result(section_result: dict[str, Any], max_spots: int | None):
    if max_spots is None:
        return section_result
    if max_spots <= 0:
        raise ValueError("--max_spots_per_section must be positive.")

    subset_modalities = {}
    for modality, adata in section_result["modalities"].items():
        if adata is None:
            subset_modalities[modality] = None
            continue
        if adata.n_obs < max_spots:
            raise ValueError(
                f"{section_result['section_id']} {modality} has only {adata.n_obs} spots, "
                f"cannot take first {max_spots}."
            )
        subset_modalities[modality] = adata[:max_spots].copy()

    section_result = dict(section_result)
    section_result["modalities"] = subset_modalities
    section_result["messages"] = list(section_result.get("messages", [])) + [
        f"{section_result['section_id']}: subset to first {max_spots} spots for all modalities"
    ]
    return section_result


def build_mousebrain_sections(config: Mapping[str, Any], max_spots: int | None):
    misplaced = {
        "n_comps", "hvg_num", "hvg_num_by_modality", "target_sum", "use_harmony",
        "spatial_key", "uni_feature_key", "rna_gene_id_key",
    }.intersection(config)
    if misplaced:
        raise ValueError(
            f"MouseBrain expects these fields inside preprocessing: {sorted(misplaced)}"
        )
    preprocessing = config.get("preprocessing", {})
    spatial_key = preprocessing.get("spatial_key", "spatial")
    uni_feature_key = preprocessing.get("uni_feature_key", "uni_feature")
    rna_gene_id_key = preprocessing.get("rna_gene_id_key", "gene_ids")

    section_results = []
    messages = []
    for section in config["sections"]:
        result = build_mousebrain_section(
            section_id=section["section_id"],
            rna_path=section["rna_input"],
            metabolite_path=section["metabolite_input"],
            spatial_key=spatial_key,
            uni_feature_key=uni_feature_key,
            rna_gene_id_key=rna_gene_id_key,
        )
        result = subset_section_result(result, max_spots=max_spots)
        section_results.append(result)
        messages.extend(result.get("messages", []))
    return section_results, messages


def preprocess_mousebrain(config: Mapping[str, Any], max_spots: int | None):
    reject_unsupported_preprocess_config(config)
    preprocessing = config.get("preprocessing", {})
    if config.get("metacell", False) or preprocessing.get("metacell", False):
        raise ValueError("metacell=true is no longer supported; use full-spot inputs.")
    section_results, messages = build_mousebrain_sections(config, max_spots=max_spots)
    data_dict = build_cosie_data_dict(section_results)
    feature_dict, spatial_loc_dict, processed_data_dict = load_cosie_style_data(
        data_dict,
        n_comps=preprocessing.get("n_comps", 50),
        hvg_num=preprocessing.get("hvg_num", 3000),
        hvg_num_by_modality=preprocessing.get("hvg_num_by_modality"),
        target_sum=preprocessing.get("target_sum"),
        use_harmony=preprocessing.get("use_harmony", True),
    )
    section_ids = [result["section_id"] for result in section_results]
    feature_dict = restore_section_keys(feature_dict, section_ids)
    spatial_loc_dict = restore_section_keys(spatial_loc_dict, section_ids)
    return {
        "section_results": section_results,
        "messages": messages,
        "data_dict": data_dict,
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "processed_data_dict": processed_data_dict,
    }


def _prepared_anndata_result(result, section_order, provenance):
    """Borrow already loaded identities; do not add an alignment acceptance rule."""
    sections = result["section_results"]
    input_sections = [item["section_id"] for item in sections]
    identity = {
        item["section_id"]: {
            "evidence": "existing per-modality AnnData identity; original adapter checks apply",
            "modalities": {
                modality: {"obs_names": obj.obs_names, "input_feature_names": obj.var_names,
                           "obs_metadata": obj.obs, "feature_metadata": obj.var}
                for modality, obj in item["modalities"].items() if obj is not None
            },
            "coordinates": result["spatial_loc_dict"][item["section_id"]],
            "derived_feature_names": None,
        }
        for item in sections
    }
    return PreparedDataset(
        section_order=section_order,
        feature_dict=result["feature_dict"], spatial_loc_dict=result["spatial_loc_dict"],
        processed_data_dict=result["processed_data_dict"], identity=identity,
        truth={"status": "not_selected", "labels": None,
               "evidence": "No new truth reading; existing obs remain available by reference."},
        provenance=provenance,
        preparation_audit={
            "input_section_order": input_sections,
            "section_key_map": {f"s{i + 1}": section for i, section in enumerate(input_sections)},
            "messages": result["messages"],
            "identity_policy": "No new reordering, joining or stricter obs validation.",
        },
        compatibility_context={"legacy_result": result},
    )


@dataclass
class BundleInput:
    """Already model-ready values; only the existing feature float32 boundary."""
    path: Any
    section_order: list[str] | None = None


def to_float_tensors(feature_dict):
    converted = {}
    for section, modalities in feature_dict.items():
        converted[section] = {}
        for modality, value in modalities.items():
            if isinstance(value, torch.Tensor):
                converted[section][modality] = value.float()
            else:
                converted[section][modality] = torch.as_tensor(value, dtype=torch.float32)
    return converted


def _prepared_model_ready(inputs, provenance):
    return PreparedDataset(
        section_order=inputs["section_order"], feature_dict=inputs["feature_dict"],
        spatial_loc_dict=inputs["spatial_loc_dict"],
        processed_data_dict=inputs["processed_data_dict"],
        identity={section: {"evidence": "row-order evidence only", "obs_names": None,
                            "input_feature_names": None}
                  for section in inputs["feature_dict"]},
        truth={"status": "not_provided", "labels": None,
               "evidence": "No additional identity/truth metadata loaded."},
        provenance=provenance,
        preparation_audit={"messages": inputs["messages"],
                           "section_order_policy": "Preserve None/empty/duplicates; existing consumer resolves or rejects."},
        compatibility_context={"legacy_inputs": inputs},
    )


@dataclass
class RawConfigInput:
    """Explicit generic flat config; existing per-modality/HE input fields only."""
    config: Mapping[str, Any]
    section_order: list[str] | None = None


@dataclass
class SyntheticInput:
    """Explicit smoke data; preserve the original builder's fixed seed and sizes."""
    section_order: list[str] | None = None


def build_synthetic_training_inputs():
    torch.manual_seed(88)
    rng = np.random.default_rng(88)
    feature_dict = {
        "s1": {
            "HE": torch.randn(36, 50),
            "RNA": torch.randn(36, 50),
            "Protein": torch.randn(36, 20),
        },
        "s2": {
            "HE": torch.randn(42, 50),
            "RNA": torch.randn(42, 50),
            "Protein": torch.randn(42, 20),
        },
    }
    spatial_loc_dict = {
        "s1": rng.random((36, 2)),
        "s2": rng.random((42, 2)),
    }
    return {
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "processed_data_dict": None,
        "section_order": ["s1", "s2"],
        "messages": ["Built synthetic smoke-test inputs."],
    }


def prepare_dataset(spec) -> PreparedDataset:
    """Dispatch only on explicit input spec types, never filenames or dataset IDs."""
    if isinstance(spec, MouseBrainInput):
        result = preprocess_mousebrain(spec.config, spec.max_spots)
        # Preserve the runner's existing fallback, separately from raw section order.
        section_order = spec.config.get("section_order") or sorted(result["feature_dict"].keys())
        return _prepared_anndata_result(result, section_order, {
            "input_kind": "mousebrain_existing_uni", "config": spec.config,
            "HE_source": "RNA.obsm configured uni_feature_key; no image inference",
            "rna_identity": "configured RNA.var gene_ids", "dtype_policy": "existing COSIE boundary",
        })
    if isinstance(spec, BundleInput):
        bundle = torch.load(spec.path, map_location="cpu")
        if "feature_dict" not in bundle or "spatial_loc_dict" not in bundle:
            raise KeyError("--input_bundle must contain feature_dict and spatial_loc_dict.")
        feature_dict = to_float_tensors(bundle["feature_dict"])
        spatial_loc_dict = bundle["spatial_loc_dict"]
        section_order = spec.section_order or bundle.get("section_order")
        inputs = {
            "feature_dict": feature_dict,
            "spatial_loc_dict": spatial_loc_dict,
            "processed_data_dict": bundle.get("processed_data_dict"),
            "section_order": section_order,
            "messages": ["Loaded preprocessed tensors from input_bundle."],
        }
        return _prepared_model_ready(inputs, {
            "input_kind": "model_ready_bundle", "path": str(spec.path),
            "dtype_policy": "feature: Tensor.float()/as_tensor(float32); spatial/processed unchanged",
            "preprocessing": "not run", "barcode_evidence": "not supplied by this loader",
        })
    if isinstance(spec, RawConfigInput):
        cfg = spec.config
        misplaced = {"n_comps", "hvg_num", "target_sum", "use_harmony"}.intersection(
            cfg.get("preprocessing", {})
        )
        if misplaced:
            raise ValueError(
                f"Generic preprocessing expects these fields at the top level: {sorted(misplaced)}"
            )
        sections = cfg["sections"]
        result = preprocess_multisection_cosie_style(
            sections=sections,
            n_comps=cfg.get("n_comps", 50),
            hvg_num=cfg.get("hvg_num", 3000),
            target_sum=cfg.get("target_sum"),
            use_harmony=cfg.get("use_harmony", True),
            config=cfg,
        )
        section_order = spec.section_order or result.get("section_ids")
        prepared = _prepared_anndata_result(result, section_order, {
            "input_kind": "generic_raw_config", "config": cfg,
            "dtype_policy": "existing COSIE boundary", "schema": "existing generic flat config",
            "HE_inputs": [
                {key: section.get(key) for key in (
                    "he_reference_adata_input", "he_feature_input", "he_input", "he_mask_input"
                )} for section in sections
            ],
            "HE_policy": "Explicit input fields; original reference/feature/image precedence; see adapter messages.",
        })
        prepared.compatibility_context["legacy_inputs"] = {
            "feature_dict": result["feature_dict"],
            "spatial_loc_dict": result["spatial_loc_dict"],
            "processed_data_dict": result["processed_data_dict"],
            "section_order": section_order,
            "messages": result.get("messages", []),
        }
        return prepared
    if isinstance(spec, SyntheticInput):
        inputs = build_synthetic_training_inputs()
        prepared = _prepared_model_ready(inputs, {
            "input_kind": "synthetic", "builder_seed": 88,
            "dtype_policy": "original torch.randn and default_rng.random outputs",
        })
        prepared.section_order = spec.section_order or inputs["section_order"]
        for identity in prepared.identity.values():
            identity["evidence"] = "synthetic generated row order; no biological identity"
        return prepared
    raise TypeError(f"Unsupported preparation input spec: {type(spec).__name__}")
