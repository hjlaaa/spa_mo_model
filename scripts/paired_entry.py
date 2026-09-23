"""Shared CLI-side preparation/resource shell for paired training entries.

Configuration is resolved by each caller before entering this shell. Raw input
preparation and its hooks stay outside training.paired_task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

from data_io.paired_preparation import prepare_paired_dataset
from training.artifacts import ensure_dir, save_list, save_selected_spot_indices
from training.fit import CudaMemoryMonitor
from training.paired_task import run_paired_training_task


def run_paired_entry(
    args, *, run_config, pair_spec, status_prefix="CRC_STEREOCITE",
    prepare_dataset=prepare_paired_dataset,
) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    if args.max_spots_per_section is not None and args.max_spots_per_section <= 0:
        raise ValueError("--max_spots_per_section must be a positive integer when provided.")
    if args.max_shared_genes is not None and args.max_shared_genes <= 0:
        raise ValueError("--max_shared_genes must be positive when provided.")
    if args.train and args.epochs <= 0:
        raise ValueError("--epochs must be positive when --train is set.")
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")
    for name in [
        "initial_modality_candidate_k",
        "candidate_k",
        "attention_topk",
        "faiss_nlist",
        "faiss_nprobe",
        "faiss_train_sample_size",
        "uot_max_iter",
        "graphsage_edge_batch_size",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    for name in ["decoder_chunk_size", "ot_attention_source_chunk_size"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.amp_dtype != "none" and args.device != "cuda":
        raise ValueError("--amp_dtype can only be enabled with --device cuda.")
    if args.log_cuda_memory_detail:
        args.log_cuda_memory = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    memory_monitor = CudaMemoryMonitor(
        enabled=bool(args.log_cuda_memory),
        output_dir=output_dir,
        requested_device=args.device,
    )
    memory_monitor.record("script_start")

    backed_rna: dict[str, ad.AnnData] = {}
    backed_adt: dict[str, ad.AnnData] = {}
    rna_info: dict[str, dict[str, Any]] = {}
    duplicate_tables: dict[str, pd.DataFrame] = {}
    alignment_summary: dict[str, dict[str, Any]] = {}

    try:
        def metadata_ready(duplicate_tables, selected_shared_genes, adt_markers):
            for section, table in duplicate_tables.items():
                table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
            save_list(output_dir / "shared_gene_symbols_make_unique.txt", selected_shared_genes)
            save_list(output_dir / "adt_marker_list.txt", adt_markers)

        def spots_selected(obs_indices):
            if args.save_outputs:
                return save_selected_spot_indices(output_dir, obs_indices, backed_rna)
            return {}

        def observe(stage=None, *, extra=None, reset_peak=False):
            if reset_peak:
                memory_monitor.reset_peak()
            if stage is not None:
                if extra is None:
                    memory_monitor.record(stage)
                else:
                    memory_monitor.record(stage, extra=extra)

        prepared = prepare_dataset(
            pair_spec, data_dir=data_dir, preprocessing=run_config["preprocessing"],
            max_shared_genes=args.max_shared_genes,
            max_spots_per_section=args.max_spots_per_section,
            spot_sampling=args.spot_sampling, rng=rng,
            resources={"RNA": backed_rna, "Protein": backed_adt},
            metadata_ready=metadata_ready, spots_selected=spots_selected, observe=observe,
        )
        return run_paired_training_task(
            run_config, prepared, memory_monitor=memory_monitor,
            protein_var_names={section: obj.var_names for section, obj in backed_adt.items()},
            status_prefix=status_prefix,
        )
    finally:
        for adata_obj in list(backed_rna.values()) + list(backed_adt.values()):
            if getattr(adata_obj, "isbacked", False):
                adata_obj.file.close()


from scripts.paired_cli import SAMPLES, resolve_run_config
from data_io.paired_preparation import PairSpec
from data_io.paired import read_backed_pair, prepare_rna_var_names_make_unique


def run_legacy_paired_entry(
    args, *, samples=None, read_pair=None, prepare_rna=None,
    filter_shared_genes=None, status_prefix="CRC_STEREOCITE", run_config=None,
    prepare_dataset=None, pair_spec=None,
) -> dict[str, Any]:
    """Legacy entry signature; task implementation lives in training.paired_task."""
    samples = SAMPLES if samples is None else samples
    first_section, second_section = samples
    if run_config is None:
        run_config = resolve_run_config(args, samples=samples)
    read_pair = read_backed_pair if read_pair is None else read_pair
    prepare_rna = prepare_rna_var_names_make_unique if prepare_rna is None else prepare_rna
    if prepare_dataset is None:
        prepare_dataset = prepare_paired_dataset
        if pair_spec is None:
            pair_spec = PairSpec(
                samples=samples, read_pair=read_pair, prepare_rna=prepare_rna,
                filter_shared_genes=filter_shared_genes,
            )
    return run_paired_entry(
        args, run_config=run_config, pair_spec=pair_spec,
        status_prefix=status_prefix, prepare_dataset=prepare_dataset,
    )
