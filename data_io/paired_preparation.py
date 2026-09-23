"""Explicit paired preparation, with caller-owned resources and three narrow hooks.

CRC, HLN and Spleen share this preparation algorithm and object ownership.
Input identity and filtering remain explicit strategies supplied by each caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

from .paired import validate_rna_adt_alignment, select_obs_indices, subset_to_memory, rename_section_keys
from .preprocessing import load_cosie_style_data

SUFFIX_RE = re.compile(r".+-[0-9]+$")


@dataclass
class PairSpec:
    """Explicit ordered input paths and existing reader/identity strategies."""

    samples: Mapping[str, str]
    read_pair: Callable
    prepare_rna: Callable
    filter_shared_genes: Callable | None = None


@dataclass
class PreparedDataset:
    """Logical result only: all payloads remain borrowed, without coercion/copy.

    Identity describes input features separately from derived PCA/Harmony axes.
    Truth status records loading evidence, not absence in the original dataset.
    compatibility_context is transitional and is not a universal adapter contract.
    """

    section_order: list[str]
    feature_dict: dict[str, Any]
    spatial_loc_dict: dict[str, Any]
    processed_data_dict: Any
    identity: dict[str, Any]
    truth: dict[str, Any]
    provenance: dict[str, Any]
    preparation_audit: dict[str, Any]
    compatibility_context: dict[str, Any]


def prepare_paired_dataset(
    spec: PairSpec, *, data_dir: Path, preprocessing: Mapping[str, Any],
    max_shared_genes: int | None, max_spots_per_section: int | None,
    spot_sampling: str, rng: np.random.Generator, resources: Mapping[str, dict],
    metadata_ready: Callable, spots_selected: Callable, observe: Callable,
) -> PreparedDataset:
    """Prepare two sections without resolving defaults, reseeding, saving or closing.

    resources["RNA"] and resources["Protein"] are the caller's original backed
    registries. Successful reader returns are registered immediately; the caller
    retains its existing finally cleanup, including when any hook raises.
    metadata_ready writes existing CSV/txt metadata; spots_selected writes optional
    indices and returns their legacy paths; observe records stages/reset_peak only.
    """
    samples = spec.samples
    first_section, second_section = samples
    section_key_map = {"s1": first_section, "s2": second_section}
    read_pair = spec.read_pair
    prepare_rna = spec.prepare_rna
    filter_shared_genes = spec.filter_shared_genes
    backed_rna = resources["RNA"]
    backed_adt = resources["Protein"]
    rna_info, duplicate_tables, alignment_summary = {}, {}, {}

    for section, sample_dir in samples.items():
        rna, adt = read_pair(data_dir, sample_dir)
        backed_rna[section] = rna
        backed_adt[section] = adt
        rna_info[section], duplicate_tables[section] = prepare_rna(rna)
        alignment_summary[section] = validate_rna_adt_alignment(section, rna, adt)
    observe("read_raw_h5ad_make_unique_validate_alignment")

    rna003 = backed_rna[first_section]
    rna006 = backed_rna[second_section]
    shared_set_006 = set(rna006.var_names.astype(str))
    shared_genes_all = [gene for gene in rna003.var_names.astype(str) if gene in shared_set_006]
    if filter_shared_genes is not None:
        shared_genes_all = filter_shared_genes(rna003, rna006, shared_genes_all)
    if not shared_genes_all:
        raise ValueError("No shared RNA genes found after var_names_make_unique().")
    selected_shared_genes = (
        shared_genes_all[: max_shared_genes]
        if max_shared_genes is not None
        else shared_genes_all
    )

    rna003_var = rna003.var.loc[selected_shared_genes]
    shared_suffix_genes_selected = [
        gene
        for gene in selected_shared_genes
        if bool(SUFFIX_RE.match(gene))
        and str(rna003_var.loc[gene, "gene_symbol_original"]) != gene
    ]
    rna003_var_all = rna003.var.loc[shared_genes_all]
    shared_suffix_genes_all = [
        gene
        for gene in shared_genes_all
        if bool(SUFFIX_RE.match(gene))
        and str(rna003_var_all.loc[gene, "gene_symbol_original"]) != gene
    ]
    observe(
        "shared_gene_alignment",
        extra={
            "shared_gene_count_total": int(len(shared_genes_all)),
            "shared_gene_count_used": int(len(selected_shared_genes)),
        },
    )

    adt_markers_003 = list(map(str, backed_adt[first_section].var_names))
    adt_markers_006 = list(map(str, backed_adt[second_section].var_names))
    adt_marker_set_same = set(adt_markers_003) == set(adt_markers_006)
    adt_marker_order_same = adt_markers_003 == adt_markers_006
    if not backed_adt[first_section].var_names.is_unique or not backed_adt[second_section].var_names.is_unique:
        raise ValueError("ADT marker var_names must be unique.")
    if not adt_marker_set_same:
        raise ValueError(f"{first_section} and {second_section} ADT marker sets differ.")
    if not adt_marker_order_same:
        raise ValueError(f"{first_section} and {second_section} ADT marker order differs.")

    metadata_ready(duplicate_tables, selected_shared_genes, adt_markers_003)
    observe("saved_lightweight_metadata")

    obs_indices = {
        section: select_obs_indices(
            backed_rna[section].n_obs,
            max_spots_per_section,
            spot_sampling,
            rng,
        )
        for section in samples
    }
    selected_obs_names_preview = {
        section: list(backed_rna[section].obs_names[indices][:10])
        if not isinstance(indices, slice)
        else list(backed_rna[section].obs_names[:10])
        for section, indices in obs_indices.items()
    }
    saved_selected_spot_indices = spots_selected(obs_indices)
    observe(
        "selected_spots",
        extra={
            "max_spots_per_section": int(max_spots_per_section)
            if max_spots_per_section is not None
            else None,
            "spot_sampling": spot_sampling,
        },
    )

    rna003_mem = subset_to_memory(backed_rna[first_section], obs_indices[first_section], selected_shared_genes)
    rna006_mem = subset_to_memory(backed_rna[second_section], obs_indices[second_section], selected_shared_genes)
    adt003_mem = subset_to_memory(backed_adt[first_section], obs_indices[first_section], None)
    adt006_mem = subset_to_memory(backed_adt[second_section], obs_indices[second_section], None)
    observe(
        "loaded_selected_anndata_to_memory",
        extra={
            f"rna_{first_section}_shape": list(rna003_mem.shape),
            f"rna_{second_section}_shape": list(rna006_mem.shape),
            f"adt_{first_section}_shape": list(adt003_mem.shape),
            f"adt_{second_section}_shape": list(adt006_mem.shape),
        },
    )

    if list(rna003_mem.var_names) != list(rna006_mem.var_names):
        raise ValueError("Subset RNA shared gene order is not identical after alignment.")
    if not rna003_mem.var_names.is_unique or not rna006_mem.var_names.is_unique:
        raise ValueError("Subset RNA var_names are not unique.")

    data_dict = {
        "RNA": [rna003_mem, rna006_mem],
        "Protein": [adt003_mem, adt006_mem],
        "HE": [None, None],
        "Metabolite": [None, None],
    }
    observe("constructed_data_dict")

    observe(reset_peak=True)
    observe("cosie_preprocessing_start")
    feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
        data_dict,
        **preprocessing,
    )
    preprocessing_generated_keys = list(feature_dict_raw.keys())
    feature_dict = rename_section_keys(feature_dict_raw, section_key_map)
    spatial_loc_dict = rename_section_keys(spatial_loc_dict_raw, section_key_map)
    section_order = [first_section, second_section]
    observe(
        "cosie_preprocessing_end",
        extra={
            "feature_dict_shapes": {section: {modality: list(tensor.shape) for modality, tensor in modalities.items()} for section, modalities in feature_dict.items()},
            "spatial_loc_dict_shapes": {section: list(np.asarray(spatial).shape) for section, spatial in spatial_loc_dict.items()},
        },
    )

    preparation_audit = {
        "rna_info": rna_info,
        "duplicate_tables": duplicate_tables,
        "alignment_summary": alignment_summary,
        "shared_genes_all": shared_genes_all,
        "selected_shared_genes": selected_shared_genes,
        "shared_suffix_genes_selected": shared_suffix_genes_selected,
        "shared_suffix_genes_all": shared_suffix_genes_all,
        "adt_markers_003": adt_markers_003,
        "adt_marker_set_same": adt_marker_set_same,
        "adt_marker_order_same": adt_marker_order_same,
        "obs_indices": obs_indices,
        "selected_obs_names_preview": selected_obs_names_preview,
        "preprocessing_generated_keys": preprocessing_generated_keys,
        "section_key_map": section_key_map,
    }
    identity = {
        section: {
            "obs_names": data_dict["RNA"][i].obs_names,
            "source_rows": obs_indices[section],
            "source_n_obs": backed_rna[section].n_obs,
            "alignment_evidence": alignment_summary[section],
            "input_feature_ids": {"RNA": selected_shared_genes, "Protein": adt_markers_003},
            "protein_display_names": data_dict["Protein"][i].var.get("adt_marker"),
            "derived_feature_axes": {
                "kind": "harmony" if preprocessing["use_harmony"] else "pca",
                "names": None,
            },
        }
        for i, section in enumerate(section_order)
    }
    return PreparedDataset(
        section_order=section_order,
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=processed_data_dict,
        identity=identity,
        truth={"status": "not_loaded", "labels": None, "applicability": "not_assessed"},
        provenance={
            "input_kind": "raw_paired_h5ad",
            "data_dir": str(data_dir),
            "samples": samples,
            "reader": f"{read_pair.__module__}.{read_pair.__name__}",
            "rna_identity": f"{prepare_rna.__module__}.{prepare_rna.__name__}",
            "shared_gene_filter": None if filter_shared_genes is None else f"{filter_shared_genes.__module__}.{filter_shared_genes.__name__}",
            "source_content_hash": None,
            "evidence_limit": "Existing reader metadata only; no additional truth or source-content hashing.",
        },
        preparation_audit=preparation_audit,
        compatibility_context={
            "data_dict": data_dict,
            "saved_selected_spot_indices": saved_selected_spot_indices,
        },
    )
