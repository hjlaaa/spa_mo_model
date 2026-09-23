"""Explicit multi-section RNA/secondary-modality preparation.

Keep MISAR identity/intersection semantics and adapter-owned input algorithms.
Resources and all output writes remain with the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np

from .paired import select_obs_indices, subset_to_memory
from .misar import common_var_names, make_unique_rna_var_names, validate_rna_atac_alignment, rename_section_keys
from .paired_preparation import PreparedDataset
from .preprocessing import load_cosie_style_data


@dataclass
class MultisectionSpec:
    """Ordered sections and explicit reader/modality/truth strategies, no guessing."""
    section_order: list[str]
    section_info: Mapping[str, Mapping[str, str]]
    read_pair: Callable
    secondary_modality: str
    secondary_name: str
    truth_provider: Callable | None = None


def prepare_multisection_dataset(
    spec: MultisectionSpec, *, data_dir: Path, preprocessing: Mapping[str, Any],
    max_shared_genes: int | None, max_shared_peaks: int | None,
    max_spots_per_section: int | None, spot_sampling: str,
    rng: np.random.Generator, resources: Mapping[str, dict],
    metadata_ready: Callable, spots_selected: Callable,
    obs_materialized: Callable, observe: Callable,
) -> PreparedDataset:
    """Prepare using the original order/copy points; never save, reseed or close.

    The four narrow hooks preserve existing metadata/indices/obs writes and
    memory observation at their original stages. Adapter audits precede this
    call in the owning runner and are not duplicated or deferred here.
    """
    section_order = spec.section_order
    secondary_modality = spec.secondary_modality
    secondary_name = spec.secondary_name
    backed_rna = resources["RNA"]
    backed_atac = resources[secondary_modality]
    rna_info, duplicate_tables, alignment_summary = {}, {}, {}

    for section in section_order:
        rna, atac = spec.read_pair(data_dir, section)
        backed_rna[section] = rna
        backed_atac[section] = atac
        rna_info[section], duplicate_tables[section] = make_unique_rna_var_names(section, rna)
        alignment_summary[section] = validate_rna_atac_alignment(section, rna, atac, secondary_modality)
    observe("read_raw_h5ad_make_unique_validate_alignment")

    shared_genes = common_var_names(backed_rna, section_order, max_shared_genes, "RNA genes")
    shared_peaks = common_var_names(backed_atac, section_order, max_shared_peaks, f"{secondary_modality} peaks")
    metadata_ready(duplicate_tables, shared_genes, shared_peaks)
    observe(
        "shared_feature_alignment",
        extra={
            "shared_gene_count_used": int(len(shared_genes)),
            "shared_peak_count_used": int(len(shared_peaks)),
        },
    )

    obs_indices = {
        section: select_obs_indices(
            backed_rna[section].n_obs,
            max_spots_per_section,
            spot_sampling,
            rng,
        )
        for section in section_order
    }
    saved_selected_spot_indices = spots_selected(obs_indices)
    observe("selected_spots")

    rna_mem = {
        section: subset_to_memory(backed_rna[section], obs_indices[section], shared_genes)
        for section in section_order
    }
    atac_mem = {
        section: subset_to_memory(backed_atac[section], obs_indices[section], shared_peaks)
        for section in section_order
    }
    observe(
        "loaded_selected_anndata_to_memory",
        extra={
            f"rna_{section}_shape": list(rna_mem[section].shape)
            for section in section_order
        }
        | {
            f"{secondary_name}_{section}_shape": list(atac_mem[section].shape)
            for section in section_order
        },
    )

    obs_metadata_paths = obs_materialized(rna_mem, obs_indices)

    data_dict = {
        "RNA": [rna_mem[section] for section in section_order],
        secondary_modality: [atac_mem[section] for section in section_order],
    }
    observe(reset_peak=True)
    observe("cosie_preprocessing_start")
    feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
        data_dict,
        **preprocessing,
    )
    preprocessing_generated_keys = list(feature_dict_raw.keys())
    feature_dict = rename_section_keys(feature_dict_raw, section_order)
    spatial_loc_dict = rename_section_keys(spatial_loc_dict_raw, section_order)
    observe(
        "cosie_preprocessing_end",
        extra={
            "feature_dict_shapes": {s: {m: list(x.shape) for m, x in mods.items()} for s, mods in feature_dict.items()},
            "spatial_loc_dict_shapes": {s: list(np.asarray(x).shape) for s, x in spatial_loc_dict.items()},
        },
    )

    preparation_audit = {
        "rna_info": rna_info,
        "duplicate_tables": duplicate_tables,
        "alignment_summary": alignment_summary,
        "shared_genes": shared_genes,
        "shared_peaks": shared_peaks,
        "obs_indices": obs_indices,
        "preprocessing_generated_keys": preprocessing_generated_keys,
        "section_key_map": {f"s{i + 1}": s for i, s in enumerate(section_order)},
    }
    identity = {
        s: {
            "obs_names": rna_mem[s].obs_names,
            "source_rows": obs_indices[s],
            "source_n_obs": backed_rna[s].n_obs,
            "obs_metadata": rna_mem[s].obs,
            "coordinates": spatial_loc_dict[s],
            "input_feature_names": {"RNA": shared_genes, secondary_modality: shared_peaks},
            "rna_original_symbols": rna_mem[s].var["gene_symbol_original"],
            "secondary_metadata": atac_mem[s].var,
            "derived_feature_axes": {"kind": "harmony" if preprocessing["use_harmony"] else "pca", "names": None},
        }
        for s in section_order
    }
    truth = (
        spec.truth_provider(rna_mem, atac_mem)
        if spec.truth_provider is not None
        else {"status": "not_selected", "labels": None, "applicability": "not_assessed",
              "evidence": "Existing obs retained; no biological target selected or fabricated."}
    )
    reader = getattr(spec.read_pair, "func", spec.read_pair)
    return PreparedDataset(
        section_order=section_order, feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict, processed_data_dict=processed_data_dict,
        identity=identity, truth=truth,
        provenance={
            "input_kind": "explicit_multisection_pair", "data_dir": str(data_dir),
            "section_info": spec.section_info,
            "reader": f"{reader.__module__}.{reader.__name__}",
            "rna_identity": "data_io.misar.make_unique_rna_var_names",
            "common_features": "data_io.misar.common_var_names",
            "secondary_modality": secondary_modality,
            "truth_provider": None if spec.truth_provider is None else f"{spec.truth_provider.__module__}.{spec.truth_provider.__name__}",
            "source_content_hash": None,
        },
        preparation_audit=preparation_audit,
        compatibility_context={
            "data_dict": data_dict, "rna_mem": rna_mem, "atac_mem": atac_mem,
            "saved_selected_spot_indices": saved_selected_spot_indices,
            "obs_metadata_paths": obs_metadata_paths,
        },
    )
