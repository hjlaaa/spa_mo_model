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

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from data_io.misar import common_var_names, make_unique_rna_var_names, read_backed_pair
from data_io.preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel
from training.fit import (
    CudaMemoryMonitor,
    amp_enabled,
    initialize_model_ot_prior,
    json_safe,
    run_one_forward,
    train_small_crc_model,
)
from run_crc_stereocite import (
    ensure_dir,
    save_final_embeddings,
    save_ot_prior_topk,
    save_spatial_arrays,
    select_obs_indices,
    subset_to_memory,
)


from data_io.misar import SECTION_INFO


DEFAULT_SECTION_ORDER = ["dataset4", "dataset3", "dataset2", "dataset1"]
LABEL_COLUMNS = [
    "Sample",
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
]


def get_dataset_defaults(*, dataset_name="MISAR-seq", secondary_name="atac"):
    """Current entry defaults; suite arguments remain explicit overrides."""
    return {
        'data_dir': f"/home/hujinlan/spa_mo_model/data/{dataset_name}",
        'section_order': ",".join(DEFAULT_SECTION_ORDER),
        'max_spots_per_section': None,
        'spot_sampling': "first",
        'max_shared_genes': None,
        'max_shared_peaks': None,
        'train': False,
        'epochs': 0,
        'lambda_contrast': None,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'update_interval': 20,
        'log_every': 1,
        'device': "cpu",
        'candidate_backend': "faiss_ivf",
        'initial_modality_candidate_k': 100,
        'candidate_k': 200,
        'attention_topk': 10,
        'faiss_nlist': 256,
        'faiss_nprobe': 32,
        'faiss_device': "auto",
        'faiss_train_sample_size': 20000,
        'faiss_query_batch_size': 2048,
        'uot_epsilon': 0.05,
        'uot_tau_a': 1.0,
        'uot_tau_b': 1.0,
        'uot_stabilizer': 1e-8,
        'uot_max_iter': 100,
        'spatial_knn_k': 5,
        'graphsage_edge_batch_size': 50000,
        'post_ot_graphsage_scale': 1.0,
        'graphsage_dropout': 0.1,
        'encoder_dropout': 0.1,
        'fusion_dropout': 0.1,
        'ot_attention_dropout': 0.1,
        'decoder_dropout': 0.1,
        'training_loss_only': False,
        'decoder_chunk_size': 50000,
        'ot_attention_source_chunk_size': 50000,
        'checkpoint_ot_attention': False,
        'checkpoint_encoder_fusion': False,
        'checkpoint_decoder_chunks': False,
        'checkpoint_graph_encoder': False,
        'amp_dtype': "none",
        'cache_spatial_graphs': False,
        'save_candidate_qc': False,
        'save_outputs': False,
        'save_embeddings': False,
        'save_ot_prior_topk': False,
        'log_cuda_memory': False,
        'log_cuda_memory_detail': False,
        'output_dir': "/home/hujinlan/spa_mo_model/results/misar_seq/dry_run",
        'seed': 0,
        'n_comps': 50,
        'hvg_num': 3000,
        f'hvg_num_{secondary_name}': 3000,
        'no_harmony': False,
    }


def parse_args(
    argv: list[str] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
    dataset_name: str = "MISAR-seq",
    secondary_modality: str = "ATAC",
    secondary_name: str = "atac",
):
    parser = argparse.ArgumentParser(description=f"Run {dataset_name} RNA+{secondary_modality} StageMultiModalModel pipeline.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--section_order",
        default=None,
        help="Comma-separated section order used for adjacent OT links.",
    )
    parser.add_argument("--max_spots_per_section", type=int, default=None)
    parser.add_argument("--spot_sampling", choices=["first", "random"], default=None)
    parser.add_argument("--max_shared_genes", type=int, default=None)
    parser.add_argument("--max_shared_peaks", type=int, default=None)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default=None)
    parser.add_argument("--initial_modality_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--attention_topk", type=int, default=None)
    parser.add_argument("--faiss_nlist", type=int, default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_device", choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--faiss_train_sample_size", type=int, default=None)
    parser.add_argument("--faiss_query_batch_size", type=int, default=None)
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=None)
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument("--graphsage_dropout", type=float, default=None)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=None)
    parser.add_argument("--ot_attention_dropout", type=float, default=None)
    parser.add_argument("--decoder_dropout", type=float, default=None)
    parser.add_argument("--training_loss_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decoder_chunk_size", type=int, default=None)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=None)
    parser.add_argument("--checkpoint_ot_attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_encoder_fusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_decoder_chunks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_graph_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--cache_spatial_graphs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_candidate_qc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_ot_prior_topk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory_detail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None, help="HVG count for RNA.")
    parser.add_argument(f"--hvg_num_{secondary_name}", type=int, default=None, help=f"Highly variable peak count for {secondary_modality}.")
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, default=None)
    return parse_dataset_args(parser, argv, {**get_dataset_defaults(dataset_name=dataset_name, secondary_name=secondary_name), **(defaults or {})})


def parse_section_order(text: str, section_info=SECTION_INFO) -> list[str]:
    order = [item.strip() for item in text.split(",") if item.strip()]
    if not order:
        raise ValueError("--section_order must contain at least one section.")
    unknown = [section for section in order if section not in section_info]
    if unknown:
        raise ValueError(f"Unknown MISAR sections in --section_order: {unknown}")
    if len(set(order)) != len(order):
        raise ValueError("--section_order contains duplicated sections.")
    return order


def release_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(spatial)
    return {
        "x": [float(arr[:, 0].min()), float(arr[:, 0].max())],
        "y": [float(arr[:, 1].min()), float(arr[:, 1].max())],
    }


def validate_rna_atac_alignment(
    section: str, rna: ad.AnnData, atac: ad.AnnData, secondary_modality: str = "ATAC"
) -> dict[str, Any]:
    if list(rna.obs_names) != list(atac.obs_names):
        raise ValueError(f"{section}: RNA and {secondary_modality} obs_names are not identical.")
    if "spatial" not in rna.obsm:
        raise KeyError(f"{section}: RNA is missing obsm['spatial'].")
    if "spatial" not in atac.obsm:
        raise KeyError(f"{section}: {secondary_modality} is missing obsm['spatial'].")
    rna_spatial = np.asarray(rna.obsm["spatial"])
    atac_spatial = np.asarray(atac.obsm["spatial"])
    if rna_spatial.shape != atac_spatial.shape:
        raise ValueError(f"{section}: RNA and {secondary_modality} spatial shapes differ.")
    spatial_match = bool(np.allclose(rna_spatial, atac_spatial))
    if not spatial_match:
        raise ValueError(f"{section}: RNA and {secondary_modality} spatial coordinates differ.")
    label_summary = {}
    for column in LABEL_COLUMNS:
        if column == "ATAC_Clusters":
            column = f"{secondary_modality}_Clusters"
        if column in rna.obs:
            label_summary[column] = {
                "n_unique": int(rna.obs[column].astype(str).nunique(dropna=False)),
                "top_counts": {
                    str(key): int(value)
                    for key, value in rna.obs[column].astype(str).value_counts(dropna=False).head(20).items()
                },
            }
    return {
        "spot_count": int(rna.n_obs),
        "obs_names_match": True,
        "spatial_shape": list(rna_spatial.shape),
        "spatial_match": spatial_match,
        "spatial_range": spatial_range(rna_spatial),
        "labels": label_summary,
    }


def save_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def indexer_to_numpy(indexer: np.ndarray | slice, n_obs: int) -> np.ndarray:
    if isinstance(indexer, slice):
        return np.arange(n_obs, dtype=np.int64)
    return np.asarray(indexer, dtype=np.int64)


def save_selected_spot_indices(
    output_dir: Path,
    obs_indices: Mapping[str, np.ndarray | slice],
    backed_rna: Mapping[str, ad.AnnData],
) -> dict[str, str]:
    paths = {}
    for section, indexer in obs_indices.items():
        path = output_dir / f"selected_spot_indices_{section}.npy"
        np.save(path, indexer_to_numpy(indexer, backed_rna[section].n_obs))
        paths[section] = str(path)
    return paths


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


def summarize_data_dict(data_dict: Mapping[str, list[ad.AnnData | None]]) -> dict[str, Any]:
    return {
        modality: [
            None
            if adata_obj is None
            else {
                "shape": list(adata_obj.shape),
                "var_names_is_unique": bool(adata_obj.var_names.is_unique),
                "obsm_keys": list(adata_obj.obsm.keys()),
            }
            for adata_obj in sections
        ]
        for modality, sections in data_dict.items()
    }


def summarize_feature_dict(feature_dict: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        section: {modality: list(tensor.shape) for modality, tensor in modalities.items()}
        for section, modalities in feature_dict.items()
    }


def summarize_spatial_loc_dict(spatial_loc_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {section: list(np.asarray(spatial).shape) for section, spatial in spatial_loc_dict.items()}


def rename_section_keys(mapping: Mapping[str, Any], section_order: list[str]) -> dict[str, Any]:
    key_map = {f"s{idx + 1}": section for idx, section in enumerate(section_order)}
    return {key_map.get(section, section): value for section, value in mapping.items()}


def make_model_config(args, secondary_modality: str = "ATAC") -> dict[str, Any]:
    config = resolve_model_config(
        input_config={
            "training": {
                "device": args.device, "epochs": int(args.epochs),
                "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            },
            "uot": {
                "max_iter": int(args.uot_max_iter), "topk": int(args.attention_topk),
                "epsilon_update": float(args.uot_epsilon),
                "tau_a": float(args.uot_tau_a), "tau_b": float(args.uot_tau_b),
                "update_interval": int(args.update_interval),
            },
            "graph": {"knn_neighbors_spatial": int(args.spatial_knn_k)},
            "graphsage": {
                "edge_batch_size": int(args.graphsage_edge_batch_size),
                "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
            },
        },
        explicit_overrides={"loss": {"lambda_contrast": (
            float(args.lambda_contrast) if args.lambda_contrast is not None else None
        )}},
    )
    config["model"]["modalities_supported"] = ["RNA", secondary_modality]
    config["model"]["valid_modality_sets"] = [["RNA", secondary_modality]]
    config["encoder"]["dropout"] = float(args.encoder_dropout)
    config["fusion"]["dropout"] = float(args.fusion_dropout)
    config["graphsage"]["dropout"] = float(args.graphsage_dropout)
    config["ot_attention"]["dropout"] = float(args.ot_attention_dropout)
    config["decoder"]["dropout"] = float(args.decoder_dropout)
    config["reconstruction"]["lambda_by_modality"][secondary_modality] = 1.0
    return config


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


def resolve_run_config(
    args, *, section_info=None, dataset_name="MISAR-seq",
    secondary_modality="ATAC", secondary_name="atac",
):
    section_info = SECTION_INFO if section_info is None else section_info
    return describe_run_config(
        args, make_model_config(args, secondary_modality), dataset=dataset_name,
        section_order=parse_section_order(args.section_order, section_info),
        modalities=["RNA", secondary_modality],
        preprocessing={
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            "hvg_num_by_modality": {
                "RNA": int(args.hvg_num),
                secondary_modality: int(getattr(args, f"hvg_num_{secondary_name}")),
            },
            "target_sum": None,
            "use_harmony": not args.no_harmony,
        },
    )


def run_misar_pipeline(
    args,
    *,
    read_pair: Callable[[Path, str], tuple[ad.AnnData, ad.AnnData]] | None = None,
    section_info: Mapping[str, Mapping[str, str]] | None = None,
    dataset_name: str = "MISAR-seq",
    secondary_modality: str = "ATAC",
    secondary_name: str = "atac",
    run_config=None,
) -> dict[str, Any]:
    validate_args(args, secondary_name)
    if section_info is None:
        section_info = SECTION_INFO
    if run_config is None:
        run_config = resolve_run_config(
            args, section_info=section_info, dataset_name=dataset_name,
            secondary_modality=secondary_modality, secondary_name=secondary_name,
        )
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

    try:
        for section in section_order:
            if read_pair is None:
                rna, atac = read_backed_pair(data_dir, section, section_info)
            else:
                rna, atac = read_pair(data_dir, section)
            backed_rna[section] = rna
            backed_atac[section] = atac
            rna_info[section], duplicate_tables[section] = make_unique_rna_var_names(section, rna)
            alignment_summary[section] = validate_rna_atac_alignment(section, rna, atac, secondary_modality)
        memory_monitor.record("read_raw_h5ad_make_unique_validate_alignment")

        shared_genes = common_var_names(backed_rna, section_order, args.max_shared_genes, "RNA genes")
        shared_peaks = common_var_names(backed_atac, section_order, args.max_shared_peaks, f"{secondary_modality} peaks")
        for section, table in duplicate_tables.items():
            table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
        save_list(output_dir / "shared_gene_symbols_make_unique.txt", shared_genes)
        save_list(output_dir / f"shared_{secondary_name}_peaks.txt", shared_peaks)
        memory_monitor.record(
            "shared_feature_alignment",
            extra={
                "shared_gene_count_used": int(len(shared_genes)),
                "shared_peak_count_used": int(len(shared_peaks)),
            },
        )

        obs_indices = {
            section: select_obs_indices(
                backed_rna[section].n_obs,
                args.max_spots_per_section,
                args.spot_sampling,
                rng,
            )
            for section in section_order
        }
        saved_selected_spot_indices = {}
        if args.save_outputs:
            saved_selected_spot_indices = save_selected_spot_indices(output_dir, obs_indices, backed_rna)
        memory_monitor.record("selected_spots")

        rna_mem = {
            section: subset_to_memory(backed_rna[section], obs_indices[section], shared_genes)
            for section in section_order
        }
        atac_mem = {
            section: subset_to_memory(backed_atac[section], obs_indices[section], shared_peaks)
            for section in section_order
        }
        memory_monitor.record(
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

        obs_metadata_paths = {}
        if args.save_outputs:
            obs_metadata_paths = save_obs_metadata(output_dir, section_order, rna_mem, obs_indices, backed_rna, section_info)

        data_dict = {
            "RNA": [rna_mem[section] for section in section_order],
            secondary_modality: [atac_mem[section] for section in section_order],
        }
        memory_monitor.reset_peak()
        memory_monitor.record("cosie_preprocessing_start")
        feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
            data_dict,
            **run_config["preprocessing"],
        )
        preprocessing_generated_keys = list(feature_dict_raw.keys())
        feature_dict = rename_section_keys(feature_dict_raw, section_order)
        spatial_loc_dict = rename_section_keys(spatial_loc_dict_raw, section_order)
        memory_monitor.record(
            "cosie_preprocessing_end",
            extra={
                "feature_dict_shapes": summarize_feature_dict(feature_dict),
                "spatial_loc_dict_shapes": summarize_spatial_loc_dict(spatial_loc_dict),
            },
        )

        model_config = run_config["model_config"]
        memory_monitor.reset_peak()
        memory_monitor.record("model_init_start")
        model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
        memory_monitor.record("model_init_end")
        resolved_modality_order = list(model._resolve_modality_order(feature_dict[section_order[0]]))
        if resolved_modality_order != ["RNA", secondary_modality]:
            raise ValueError(f"Expected {['RNA', secondary_modality]}, got {resolved_modality_order}.")

        memory_monitor.reset_peak()
        memory_monitor.record("initial_ot_prior_start")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        memory_monitor.record("initial_ot_prior_end")
        first_pair = (section_order[0], section_order[1]) if len(section_order) > 1 else None
        if first_pair is None:
            raise ValueError("MISAR run requires at least two sections.")
        initial_prior = model.ot_prior[first_pair]
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        expected_keys = {
            key
            for left, right in zip(section_order[:-1], section_order[1:])
            for key in [(left, right), (right, left)]
        }
        actual_keys = set((model.ot_prior or {}).keys())
        if not expected_keys.issubset(actual_keys):
            raise ValueError(f"Bidirectional OT prior is missing keys: {expected_keys - actual_keys}")

        history: list[dict[str, float]] | None = None
        ot_updates: list[int] = []
        if args.train:
            history, outputs, ot_updates = train_small_crc_model(
                model,
                feature_dict,
                spatial_loc_dict,
                processed_data_dict,
                section_order,
                args,
                memory_monitor=memory_monitor,
            )
        else:
            model.eval()
            with torch.no_grad():
                memory_monitor.reset_peak()
                memory_monitor.record("dry_run_forward_start", epoch=0)
                outputs = run_one_forward(
                    model,
                    feature_dict,
                    spatial_loc_dict,
                    processed_data_dict,
                    section_order,
                    epoch=0,
                    decoder_chunk_size=int(args.decoder_chunk_size),
                    ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                    cache_spatial_graphs=bool(args.cache_spatial_graphs),
                    checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                    checkpoint_encoder_fusion=bool(args.checkpoint_encoder_fusion),
                    checkpoint_decoder_chunks=bool(args.checkpoint_decoder_chunks),
                    checkpoint_graph_encoder=bool(args.checkpoint_graph_encoder),
                )
                memory_monitor.record("dry_run_forward_end", epoch=0)

        reconstruction_keys = {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        }
        expected_reconstruction_keys = {section: [secondary_modality, "RNA"] for section in section_order}
        if reconstruction_keys != expected_reconstruction_keys:
            raise ValueError(f"Unexpected reconstruction keys: {reconstruction_keys}")
        total_loss_finite = bool(torch.isfinite(outputs["losses"]["total_loss"]).item())
        if not total_loss_finite:
            raise ValueError("total_loss is not finite.")

        embedding_paths = {}
        spatial_paths = {}
        ot_prior_topk_files = {}
        if args.save_embeddings:
            memory_monitor.reset_peak()
            memory_monitor.record("save_embeddings_start")
            embedding_paths = save_final_embeddings(output_dir, outputs["final_embeddings"])
            memory_monitor.record("save_embeddings_end")
        if args.save_outputs:
            memory_monitor.reset_peak()
            memory_monitor.record("save_spatial_arrays_start")
            spatial_paths = save_spatial_arrays(output_dir, spatial_loc_dict)
            memory_monitor.record("save_spatial_arrays_end")
        if args.save_ot_prior_topk:
            memory_monitor.reset_peak()
            memory_monitor.record("save_ot_prior_topk_start")
            ot_prior_topk_files = save_ot_prior_topk(
                output_dir / "ot_prior_topk",
                outputs.get("ot_prior"),
                outputs["final_embeddings"],
                "training_final_eval" if args.train else "dry_run",
                save_candidate_qc=bool(args.save_candidate_qc),
            )
            memory_monitor.record("save_ot_prior_topk_end")

        summary = {
            "resolved_config": run_config,
            "mode": "train" if args.train else "dry_run",
            "dataset": dataset_name,
            "input_data_path": str(data_dir),
            "output_dir": str(output_dir),
            "section_names": section_order,
            "section_info": {section: section_info[section] for section in section_order},
            "preprocessing_generated_keys": preprocessing_generated_keys,
            "seed": int(args.seed),
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            f"hvg_num_{secondary_name}": int(getattr(args, f"hvg_num_{secondary_name}")),
            "use_harmony": not args.no_harmony,
            "spot_sampling": args.spot_sampling,
            "max_spots_per_section": int(args.max_spots_per_section)
            if args.max_spots_per_section is not None
            else None,
            "max_shared_genes": int(args.max_shared_genes) if args.max_shared_genes is not None else None,
            "max_shared_peaks": int(args.max_shared_peaks) if args.max_shared_peaks is not None else None,
            "shared_gene_count_used": int(len(shared_genes)),
            "shared_peak_count_used": int(len(shared_peaks)),
            "train": bool(args.train),
            "epochs": int(args.epochs) if args.train else 0,
            "lambda_contrast": float(model_config["loss"]["lambda_contrast"]),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "update_interval": int(args.update_interval),
            "ot_updates": ot_updates,
            "uot_max_iter": int(args.uot_max_iter),
            "ot_prior_mode": "candidate_sparse",
            "bidirectional_ot_attention": True,
            "candidate_backend": args.candidate_backend,
            "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
            "candidate_k": int(args.candidate_k),
            "attention_topk": int(args.attention_topk),
            "faiss_nlist": int(args.faiss_nlist),
            "faiss_nprobe": int(args.faiss_nprobe),
            "faiss_device": args.faiss_device,
            "faiss_train_sample_size": int(args.faiss_train_sample_size),
            "faiss_query_batch_size": int(args.faiss_query_batch_size),
            "dynamic_candidate_source": "ot",
            "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
            "pre_post_graphsage_parameter_sharing": False,
            "ot_refresh_embedding_key": "ot_embeddings",
            "uot_epsilon": float(args.uot_epsilon),
            "uot_tau_a": float(args.uot_tau_a),
            "uot_tau_b": float(args.uot_tau_b),
            "uot_stabilizer": float(args.uot_stabilizer),
            "spatial_knn_k": int(args.spatial_knn_k),
            "graphsage_edge_batch_size": int(args.graphsage_edge_batch_size),
            "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
            "training_loss_only": bool(args.training_loss_only),
            "decoder_chunk_size": int(args.decoder_chunk_size),
            "ot_attention_source_chunk_size": int(args.ot_attention_source_chunk_size),
            "checkpoint_ot_attention": bool(args.checkpoint_ot_attention),
            "checkpoint_encoder_fusion": bool(args.checkpoint_encoder_fusion),
            "checkpoint_decoder_chunks": bool(args.checkpoint_decoder_chunks),
            "checkpoint_graph_encoder": bool(args.checkpoint_graph_encoder),
            "amp_dtype": args.amp_dtype,
            "amp_enabled": bool(amp_enabled(args)),
            "cache_spatial_graphs": bool(args.cache_spatial_graphs),
            "log_cuda_memory": bool(args.log_cuda_memory),
            "log_cuda_memory_detail": bool(args.log_cuda_memory_detail),
            "cuda_memory_trace_path": str(memory_monitor.output_path) if args.log_cuda_memory else None,
            "save_candidate_qc": bool(args.save_candidate_qc),
            "ot_prior_metadata": {
                f"{source}_to_{target}": prior.get("metadata", {})
                for (source, target), prior in (outputs.get("ot_prior") or {}).items()
            },
            "rna_make_unique": rna_info,
            "alignment": alignment_summary,
            "data_dict": summarize_data_dict(data_dict),
            "feature_dict_shapes": summarize_feature_dict(feature_dict),
            "spatial_loc_dict_shapes": summarize_spatial_loc_dict(spatial_loc_dict),
            "processed_data_dict_generated": processed_data_dict is not None,
            "resolved_modality_order": resolved_modality_order,
            "final_embedding_shapes": {
                section: list(tensor.shape)
                for section, tensor in outputs["final_embeddings"].items()
            },
            "reconstruction_keys": reconstruction_keys,
            "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
            "initial_ot_prior_modalities_used": {
                f"{first_pair[0]}_to_{first_pair[1]}": initial_ot_modalities_used,
            },
            "losses": {
                key: float(value.detach().cpu())
                for key, value in outputs["losses"].items()
            },
            "total_loss_finite": total_loss_finite,
            "training_history": history,
            "saved_embeddings": bool(embedding_paths),
            "saved_ot_prior_topk": bool(ot_prior_topk_files),
            "saved_files": {
                "run_summary": str(output_dir / "run_summary.json"),
                "loss_history": str(output_dir / "loss_history.json") if history is not None else None,
                "shared_gene_symbols_make_unique": str(output_dir / "shared_gene_symbols_make_unique.txt"),
                f"shared_{secondary_name}_peaks": str(output_dir / f"shared_{secondary_name}_peaks.txt"),
                "duplicate_gene_summary": {
                    section: str(output_dir / f"duplicate_gene_summary_{section}.csv")
                    for section in section_order
                },
                "selected_spot_indices": saved_selected_spot_indices,
                "obs_metadata": obs_metadata_paths,
                "spatial": spatial_paths,
                "final_embeddings": embedding_paths,
                "ot_prior_topk": ot_prior_topk_files,
            },
            "confirmations": {
                "modalities_used": ["RNA", secondary_modality],
                f"{secondary_name}_used_as_true_{secondary_modality}": True,
                f"{secondary_name}_used_as_Protein": False,
                f"{secondary_name}_used_as_Metabolite": False,
                "fabricated_HE": False,
                "overwrote_original_h5ad": False,
                "saved_processed_h5ad": False,
                "count_aggregation_performed": False,
            },
        }
        memory_monitor.record("write_run_summary_start")
        with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
        if history is not None:
            with open(output_dir / "loss_history.json", "w", encoding="utf-8") as handle:
                json.dump(json_safe(history), handle, indent=2, ensure_ascii=False)
        memory_monitor.record("write_run_summary_end")

        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
        print("MISAR_SEQ_TRAIN: PASS" if args.train else "MISAR_SEQ_DRY_RUN: PASS")
        return summary
    finally:
        for adata_obj in list(backed_rna.values()) + list(backed_atac.values()):
            if getattr(adata_obj, "isbacked", False):
                adata_obj.file.close()
        release_cache()


def main() -> None:
    args = parse_args()
    run_misar_pipeline(args, run_config=resolve_run_config(args))


if __name__ == "__main__":
    main()
