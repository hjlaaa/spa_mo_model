#!/usr/bin/env python3
"""MISAR-seq RNA+ATAC training pipeline for StageMultiModalModel."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.multisection_entry import run_legacy_multisection_entry
from scripts.multisection_entry import (
    run_multisection_entry, release_cache, save_list, indexer_to_numpy,
    save_selected_spot_indices, save_obs_metadata, validate_args,
)
from training.multisection_task import (
    summarize_data_dict, summarize_feature_dict, summarize_spatial_loc_dict,
)
from training import artifacts as _artifacts
from scripts.multisection_cli import (
    DEFAULT_SECTION_ORDER, get_dataset_defaults, parse_args, parse_section_order, make_model_config, resolve_run_config,
)
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from data_io.misar import common_var_names, make_unique_rna_var_names, read_backed_pair
from data_io.preprocessing import load_cosie_style_data
from data_io.paired import select_obs_indices, subset_to_memory, spatial_range as _data_spatial_range
from data_io.misar import (
    LABEL_COLUMNS, validate_rna_atac_alignment as _data_validate_alignment,
    rename_section_keys as _data_rename_section_keys,
)
from data_io.misar_preparation import MultisectionSpec, prepare_multisection_dataset
from model.stage_model import StageMultiModalModel
from training.fit import (
    CudaMemoryMonitor,
    amp_enabled,
    initialize_model_ot_prior,
    json_safe,
    run_one_forward,
    train_small_crc_model,
)
from training.artifacts import (
    ensure_dir,
    save_final_embeddings,
    save_ot_prior_topk,
    save_spatial_arrays,
)


from data_io.misar import SECTION_INFO


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    """Compatibility delegate to the data-layer authority."""
    return _data_spatial_range(spatial)

def validate_rna_atac_alignment(
    section: str, rna: ad.AnnData, atac: ad.AnnData, secondary_modality: str = "ATAC"
) -> dict[str, Any]:
    """Compatibility delegate to the data-layer authority."""
    return _data_validate_alignment(section, rna, atac, secondary_modality)


def rename_section_keys(mapping: Mapping[str, Any], section_order: list[str]) -> dict[str, Any]:
    """Compatibility delegate to the data-layer authority."""
    return _data_rename_section_keys(mapping, section_order)


def run_misar_pipeline(
    args,
    *,
    read_pair: Callable[[Path, str], tuple[ad.AnnData, ad.AnnData]] | None = None,
    section_info: Mapping[str, Mapping[str, str]] | None = None,
    dataset_name: str = "MISAR-seq",
    secondary_modality: str = "ATAC",
    secondary_name: str = "atac",
    run_config=None,
    prepare_dataset=None, input_spec=None,
) -> dict[str, Any]:
    """Compatibility forwarding API; entry assembly lives in scripts.multisection_entry."""
    return run_legacy_multisection_entry(
        args, read_pair=read_pair, section_info=section_info, dataset_name=dataset_name,
        secondary_modality=secondary_modality, secondary_name=secondary_name,
        run_config=run_config, prepare_dataset=prepare_dataset, input_spec=input_spec,
    )


def main() -> None:
    args = parse_args()
    run_config = resolve_run_config(args)
    run_misar_pipeline(
        args, run_config=run_config, prepare_dataset=prepare_multisection_dataset,
        input_spec=MultisectionSpec(
            section_order=run_config["section_order"], section_info=SECTION_INFO,
            read_pair=partial(read_backed_pair, section_info=SECTION_INFO),
            secondary_modality="ATAC", secondary_name="atac",
        ),
    )


if __name__ == "__main__":
    main()
