#!/usr/bin/env python3
"""CRC Stereo-CITE-seq RNA+Protein pipeline.

This script reads raw CRC h5ad files, applies RNA ``var_names_make_unique()``
in memory, aligns shared RNA genes, reuses COSIE-style preprocessing, and runs
a StageMultiModalModel dry run or small-scale training. It never writes
processed h5ad files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paired_entry import run_legacy_paired_entry
from scripts.paired_entry import run_paired_entry
from training.paired_task import (
    summarize_data_dict, summarize_feature_dict, summarize_spatial_loc_dict,
)
from training import artifacts as _artifacts
from training.paired_task import summarize_outputs as _legacy_summarize_outputs
from scripts.paired_cli import (
    SAMPLES, get_dataset_defaults, parse_args, build_model_config, resolve_run_config,
)
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from training.fit import (
    json_safe,
    bytes_to_gib,
    release_python_and_cuda_cache,
    amp_enabled,
    autocast_context,
    make_grad_scaler,
    CudaMemoryMonitor,
    make_forward_memory_recorder,
    run_one_forward,
    sparse_prior_kwargs,
    initialize_model_ot_prior,
    update_model_ot_prior,
    train_small_crc_model,
)
from data_io.paired import (
    prepare_rna_var_names_make_unique, read_backed_pair, subset_to_memory,
    spatial_range as _paired_spatial_range,
    validate_rna_adt_alignment as _paired_validate_rna_adt_alignment,
    select_obs_indices as _paired_select_obs_indices,
    rename_section_keys as _paired_rename_section_keys,
)
from data_io.preprocessing import load_cosie_style_data
from data_io.paired_preparation import PairSpec, prepare_paired_dataset
from model.stage_model import StageMultiModalModel
from model.tensor_utils import tensor_to_numpy


AUTO_SECTION_KEY_MAP = {
    "s1": "CRC_003",
    "s2": "CRC_006",
}
SUFFIX_RE = re.compile(r".+-[0-9]+$")


def ensure_dir(path: Path) -> None:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.ensure_dir(path)


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    """Compatibility delegate; the mechanical implementation lives in data_io.paired."""
    return _paired_spatial_range(spatial)

def validate_rna_adt_alignment(section: str, rna: ad.AnnData, adt: ad.AnnData) -> dict[str, Any]:
    """Compatibility delegate; the mechanical implementation lives in data_io.paired."""
    return _paired_validate_rna_adt_alignment(section, rna, adt)

def select_obs_indices(
    n_obs: int,
    max_spots: int | None,
    sampling: str,
    rng: np.random.Generator,
) -> np.ndarray | slice:
    """Compatibility delegate; the mechanical implementation lives in data_io.paired."""
    return _paired_select_obs_indices(n_obs, max_spots, sampling, rng)


def rename_section_keys(mapping: Mapping[str, Any], section_key_map=None) -> dict[str, Any]:
    """Compatibility delegate; the mechanical implementation lives in data_io.paired."""
    if section_key_map is None:
        section_key_map = AUTO_SECTION_KEY_MAP
    return _paired_rename_section_keys(mapping, section_key_map)

def save_list(path: Path, values: list[str]) -> None:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_list(path, values)


def indexer_to_numpy(indexer: np.ndarray | slice, n_obs: int) -> np.ndarray:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.indexer_to_numpy(indexer, n_obs)


def save_selected_spot_indices(
    output_dir: Path,
    obs_indices: Mapping[str, np.ndarray | slice],
    backed_rna: Mapping[str, ad.AnnData],
) -> dict[str, str]:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_selected_spot_indices(output_dir, obs_indices, backed_rna)


def save_spatial_arrays(output_dir: Path, spatial_loc_dict: Mapping[str, Any]) -> dict[str, str]:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_spatial_arrays(output_dir, spatial_loc_dict)


def save_final_embeddings(
    output_dir: Path,
    final_embeddings: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_final_embeddings(output_dir, final_embeddings)


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
    save_candidate_qc: bool = False,
) -> dict[str, dict[str, str]]:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_ot_prior_topk(output_dir, ot_prior, final_embeddings, run_mode, save_candidate_qc=save_candidate_qc)


def summarize_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy summary projection delegated to its training owner."""
    return _legacy_summarize_outputs(outputs)


def run_crc_pipeline(
    args, *, samples=None, read_pair=None, prepare_rna=None,
    filter_shared_genes=None, status_prefix="CRC_STEREOCITE", run_config=None,
    prepare_dataset=None, pair_spec=None,
) -> dict[str, Any]:
    """Compatibility forwarding API; entry assembly lives in scripts.paired_entry."""
    return run_legacy_paired_entry(
        args, samples=samples, read_pair=read_pair, prepare_rna=prepare_rna,
        filter_shared_genes=filter_shared_genes, status_prefix=status_prefix,
        run_config=run_config, prepare_dataset=prepare_dataset, pair_spec=pair_spec,
    )


def main() -> None:
    args = parse_args()
    run_crc_pipeline(args, run_config=resolve_run_config(args))


if __name__ == "__main__":
    main()
