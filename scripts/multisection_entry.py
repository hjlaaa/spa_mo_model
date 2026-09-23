"""Shared CLI-side MISAR-family preparation and resource ownership shell."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from functools import partial
import gc
import anndata as ad
import numpy as np
import pandas as pd
import torch
from data_io.misar import SECTION_INFO, read_backed_pair
from data_io.misar_preparation import MultisectionSpec, prepare_multisection_dataset
from training import artifacts as _artifacts
from training.artifacts import ensure_dir
from training.fit import CudaMemoryMonitor
from training.multisection_task import run_multisection_training_task

def release_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_list(path: Path, values: list[str]) -> None:
    """Compatibility delegate to the public artifact writer."""
    return _artifacts.save_list(path, map(str, values))

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

def save_obs_metadata(
    output_dir: Path,
    section_order: list[str],
    rna_mem: Mapping[str, ad.AnnData],
    obs_indices: Mapping[str, np.ndarray | slice],
    backed_rna: Mapping[str, ad.AnnData],
    section_info=SECTION_INFO,
) -> dict[str, str]:
    paths = {}
    for section in section_order:
        obs = rna_mem[section].obs.copy()
        obs.insert(0, "spot_index", indexer_to_numpy(obs_indices[section], backed_rna[section].n_obs))
        obs.insert(1, "obs_name", rna_mem[section].obs_names.astype(str).to_numpy())
        spatial = np.asarray(rna_mem[section].obsm["spatial"])
        obs["spatial_x"] = spatial[:, 0]
        obs["spatial_y"] = spatial[:, 1]
        obs["section"] = section
        obs["stage"] = section_info[section]["stage"]
        path = output_dir / f"obs_metadata_{section}.csv"
        obs.to_csv(path, index=False)
        paths[section] = str(path)
    return paths

def validate_args(args, secondary_name: str = "atac") -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    if args.max_spots_per_section is not None and args.max_spots_per_section <= 0:
        raise ValueError("--max_spots_per_section must be positive when provided.")
    if args.max_shared_genes is not None and args.max_shared_genes <= 0:
        raise ValueError("--max_shared_genes must be positive when provided.")
    if args.max_shared_peaks is not None and args.max_shared_peaks <= 0:
        raise ValueError("--max_shared_peaks must be positive when provided.")
    if args.train and args.epochs <= 0:
        raise ValueError("--epochs must be positive when --train is set.")
    if args.amp_dtype != "none" and args.device != "cuda":
        raise ValueError("--amp_dtype can only be enabled with --device cuda.")
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
        "hvg_num",
        f"hvg_num_{secondary_name}",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    for name in ["decoder_chunk_size", "ot_attention_source_chunk_size"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.log_cuda_memory_detail:
        args.log_cuda_memory = True


def run_multisection_entry(
    args, *, run_config, section_info, dataset_name,
    secondary_modality, secondary_name, read_pair=None,
    prepare_dataset=None, input_spec=None, validated=False,
):
    if not validated:
        validate_args(args, secondary_name)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    section_order = run_config["section_order"]
    memory_monitor = CudaMemoryMonitor(
        enabled=bool(args.log_cuda_memory),
        output_dir=output_dir,
        requested_device=args.device,
    )
    memory_monitor.record("script_start")

    backed_rna: dict[str, ad.AnnData] = {}
    backed_atac: dict[str, ad.AnnData] = {}
    rna_info: dict[str, dict[str, Any]] = {}
    duplicate_tables: dict[str, pd.DataFrame] = {}
    alignment_summary: dict[str, dict[str, Any]] = {}

    # Keep task objects alive through the original finally/empty_cache boundary.
    execution_state = {}
    try:
        if prepare_dataset is None:
            prepare_dataset = prepare_multisection_dataset
        if input_spec is None:
            input_spec = MultisectionSpec(
                section_order=section_order, section_info=section_info,
                read_pair=read_pair if read_pair is not None else partial(read_backed_pair, section_info=section_info),
                secondary_modality=secondary_modality, secondary_name=secondary_name,
            )

        def metadata_ready(duplicate_tables, shared_genes, shared_peaks):
            for section, table in duplicate_tables.items():
                table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
            save_list(output_dir / "shared_gene_symbols_make_unique.txt", shared_genes)
            save_list(output_dir / f"shared_{secondary_name}_peaks.txt", shared_peaks)

        def spots_selected(obs_indices):
            if args.save_outputs:
                return save_selected_spot_indices(output_dir, obs_indices, backed_rna)
            return {}

        def obs_materialized(rna_mem, obs_indices):
            if args.save_outputs:
                return save_obs_metadata(output_dir, section_order, rna_mem, obs_indices, backed_rna, section_info)
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
            input_spec, data_dir=data_dir, preprocessing=run_config["preprocessing"],
            max_shared_genes=args.max_shared_genes, max_shared_peaks=args.max_shared_peaks,
            max_spots_per_section=args.max_spots_per_section, spot_sampling=args.spot_sampling,
            rng=rng, resources={"RNA": backed_rna, secondary_modality: backed_atac},
            metadata_ready=metadata_ready, spots_selected=spots_selected,
            obs_materialized=obs_materialized, observe=observe,
        )
        return run_multisection_training_task(
            run_config, prepared, memory_monitor=memory_monitor,
            section_info=section_info, dataset_name=dataset_name,
            secondary_modality=secondary_modality, secondary_name=secondary_name,
            execution_state=execution_state,
        )
    finally:
        for adata_obj in list(backed_rna.values()) + list(backed_atac.values()):
            if getattr(adata_obj, "isbacked", False):
                adata_obj.file.close()
        release_cache()


from scripts.multisection_cli import resolve_run_config
from typing import Callable


def run_legacy_multisection_entry(
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
    validate_args(args, secondary_name)
    if section_info is None:
        section_info = SECTION_INFO
    if run_config is None:
        run_config = resolve_run_config(
            args, section_info=section_info, dataset_name=dataset_name,
            secondary_modality=secondary_modality, secondary_name=secondary_name,
        )
    return run_multisection_entry(
        args, run_config=run_config, section_info=section_info,
        dataset_name=dataset_name, secondary_modality=secondary_modality,
        secondary_name=secondary_name, read_pair=read_pair,
        prepare_dataset=prepare_dataset, input_spec=input_spec, validated=True,
    )
