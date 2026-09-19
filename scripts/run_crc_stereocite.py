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

from training.config import (
    resolve_model_config, parse_dataset_args, describe_run_config,
    add_feature_graph_argument, feature_graph_overrides,
)
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
)
from data_io.preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel
from model.tensor_utils import tensor_to_numpy


SAMPLES = {
    "CRC_003": "CRC_003_bin20",
    "CRC_006": "CRC_006_bin20",
}
AUTO_SECTION_KEY_MAP = {
    "s1": "CRC_003",
    "s2": "CRC_006",
}
SUFFIX_RE = re.compile(r".+-[0-9]+$")


def get_dataset_defaults():
    """Current entry defaults; suite arguments remain explicit overrides."""
    return {
        'data_dir': "/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq",
        'max_spots_per_section': None,
        'max_shared_genes': 3000,
        'spot_sampling': "first",
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
        'faiss_nlist': 4096,
        'faiss_nprobe': 64,
        'faiss_device': "auto",
        'faiss_train_sample_size': 100000,
        'faiss_query_batch_size': 8192,
        'uot_epsilon': 0.05,
        'uot_tau_a': 1.0,
        'uot_tau_b': 1.0,
        'uot_stabilizer': 1e-8,
        'spatial_knn_k': 5,
        'graphsage_edge_batch_size': 200000,
        'post_ot_graphsage_scale': 1.0,
        'training_loss_only': False,
        'decoder_chunk_size': 0,
        'ot_attention_source_chunk_size': 0,
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
        'output_dir': "/home/hujinlan/spa_mo_model/results/crc_stereocite/dry_run_make_unique_pipeline",
        'seed': 0,
        'n_comps': 50,
        'hvg_num': 3000,
        'uot_max_iter': 100,
        'no_harmony': False,
    }


def parse_args(
    argv=None, *, defaults=None, dataset_name="CRC Stereo-CITE-seq",
    sample_dirs=("CRC_003_bin20", "CRC_006_bin20"),
):
    parser = argparse.ArgumentParser(description=f"Run {dataset_name} RNA+Protein pipeline.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help=f"Directory containing {sample_dirs[0]} and {sample_dirs[1]}.",
    )
    parser.add_argument(
        "--max_spots_per_section",
        type=int,
        default=None,
        help=(
            "Optional per-section spot subset size. Omit this argument to use all spots; "
            "set a positive integer to run a controlled subset."
        ),
    )
    parser.add_argument("--max_shared_genes", type=int, default=None)
    parser.add_argument(
        "--spot_sampling",
        choices=["first", "random"],
        default=None,
        help="How to choose the subset spots within each section.",
    )
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, help="Run a small training loop after preprocessing.", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs when --train is set.")
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
    parser.add_argument(
        "--faiss_query_batch_size",
        type=int,
        default=None,
        help=(
            "Number of source embeddings per FAISS index.search call. "
            "Use a smaller value such as 4096/2048/1024 to reduce FAISS GPU temporary memory."
        ),
    )
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument(
        "--spatial_knn_k",
        type=int,
        default=None,
        help=(
            "Number of non-self spatial neighbors for the weighted KNN graph. "
            "The graph also includes self-loops."
        ),
    )
    parser.add_argument(
        "--graphsage_edge_batch_size",
        type=int,
        default=None,
        help=(
            "Number of spatial graph edges processed per GraphSAGE message-passing chunk. "
            "Lower this if GraphSAGE OOMs on full-spot runs."
        ),
    )
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument(
        "--training_loss_only",
        action=argparse.BooleanOptionalAction,
        help="During train epochs, return only loss tensors/scalars instead of full graph-bearing outputs.",
        default=None,
    )
    parser.add_argument(
        "--decoder_chunk_size",
        type=int,
        default=None,
        help="If positive, compute decoder reconstruction loss in spot chunks.",
    )
    parser.add_argument(
        "--ot_attention_source_chunk_size",
        type=int,
        default=None,
        help="If positive, compute OT-guided attention in source-spot chunks.",
    )
    parser.add_argument(
        "--checkpoint_ot_attention",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint each OT-guided attention source chunk during training. "
            "This trades extra backward recomputation time for lower activation memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_encoder_fusion",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint modality-specific encoder and FusionMLP forwards during training. "
            "This trades extra backward recomputation time for lower activation memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_decoder_chunks",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint each chunked decoder reconstruction loss during training. "
            "Requires --decoder_chunk_size > 0 and is most useful with --training_loss_only."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_graph_encoder",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint the graph encoder forward during training. "
            "The current graph encoder implementation is WeightedResidualGraphSAGE."
        ),
        default=None,
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["none", "bf16", "fp16"],
        default=None,
        help="Optional CUDA autocast dtype for training forward. Default keeps full float32 behavior.",
    )
    parser.add_argument(
        "--cache_spatial_graphs",
        action=argparse.BooleanOptionalAction,
        help="Cache CPU spatial KNN graphs and move them to the target device in each forward.",
        default=None,
    )
    parser.add_argument("--save_candidate_qc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction, help="Save lightweight run outputs.", default=None)
    parser.add_argument("--save_embeddings", action=argparse.BooleanOptionalAction, help="Save final embeddings when available.", default=None)
    parser.add_argument(
        "--save_ot_prior_topk",
        action=argparse.BooleanOptionalAction,
        help="Save sparse top-k OT prior when available. Dense P is never saved.",
        default=None,
    )
    parser.add_argument(
        "--log_cuda_memory",
        action=argparse.BooleanOptionalAction,
        help="Write per-stage CUDA memory statistics to cuda_memory_trace.jsonl in output_dir.",
        default=None,
    )
    parser.add_argument(
        "--log_cuda_memory_detail",
        action=argparse.BooleanOptionalAction,
        help=(
            "Add optional in-forward CUDA memory detail events to cuda_memory_trace.jsonl. "
            "This is diagnostic-only and implies --log_cuda_memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, help="Disable Harmony during preprocessing.", default=None)
    add_feature_graph_argument(parser)
    return parse_dataset_args(parser, argv, {**get_dataset_defaults(), **(defaults or {})})


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(spatial)
    return {
        "x": [float(arr[:, 0].min()), float(arr[:, 0].max())],
        "y": [float(arr[:, 1].min()), float(arr[:, 1].max())],
    }


def validate_rna_adt_alignment(section: str, rna: ad.AnnData, adt: ad.AnnData) -> dict[str, Any]:
    obs_match = list(rna.obs_names) == list(adt.obs_names)
    if not obs_match:
        raise ValueError(f"{section}: RNA and ADT obs_names are not identical.")
    if "spatial" not in rna.obsm:
        raise KeyError(f"{section}: RNA is missing obsm['spatial'].")
    if "spatial" not in adt.obsm:
        raise KeyError(f"{section}: ADT is missing obsm['spatial'].")
    rna_spatial = np.asarray(rna.obsm["spatial"])
    adt_spatial = np.asarray(adt.obsm["spatial"])
    spatial_shape_match = rna_spatial.shape == adt_spatial.shape
    if not spatial_shape_match:
        raise ValueError(f"{section}: RNA and ADT spatial shapes differ.")
    spatial_match = bool(np.allclose(rna_spatial, adt_spatial))
    if not spatial_match:
        raise ValueError(f"{section}: RNA and ADT spatial coordinates differ.")
    return {
        "spot_count": int(rna.n_obs),
        "obs_names_match": obs_match,
        "spatial_key_present": True,
        "spatial_shape": list(rna_spatial.shape),
        "spatial_match": spatial_match,
        "spatial_range": spatial_range(rna_spatial),
    }


def select_obs_indices(
    n_obs: int,
    max_spots: int | None,
    sampling: str,
    rng: np.random.Generator,
) -> np.ndarray | slice:
    if max_spots is None or max_spots <= 0 or max_spots >= n_obs:
        return slice(None)
    if sampling == "first":
        return np.arange(max_spots)
    if sampling == "random":
        # Sort sampled indices so backed slicing keeps original spot order.
        return np.sort(rng.choice(n_obs, size=max_spots, replace=False))
    raise ValueError(f"Unsupported spot sampling mode: {sampling}")


def summarize_data_dict(data_dict: Mapping[str, list[ad.AnnData | None]]) -> dict[str, Any]:
    summary = {}
    for modality, sections in data_dict.items():
        summary[modality] = [
            None
            if adata_obj is None
            else {
                "shape": list(adata_obj.shape),
                "var_names_is_unique": bool(adata_obj.var_names.is_unique),
                "obsm_keys": list(adata_obj.obsm.keys()),
            }
            for adata_obj in sections
        ]
    return summary


def summarize_feature_dict(feature_dict: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        section: {modality: list(tensor.shape) for modality, tensor in modalities.items()}
        for section, modalities in feature_dict.items()
    }


def summarize_spatial_loc_dict(spatial_loc_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {section: list(np.asarray(spatial).shape) for section, spatial in spatial_loc_dict.items()}


def rename_section_keys(mapping: Mapping[str, Any], section_key_map=None) -> dict[str, Any]:
    if section_key_map is None:
        section_key_map = AUTO_SECTION_KEY_MAP
    return {
        section_key_map.get(section, section): value
        for section, value in mapping.items()
    }


def save_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


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


def save_spatial_arrays(output_dir: Path, spatial_loc_dict: Mapping[str, Any]) -> dict[str, str]:
    paths = {}
    for section, spatial in spatial_loc_dict.items():
        path = output_dir / f"spatial_{section}.npy"
        np.save(path, np.asarray(spatial))
        paths[section] = str(path)
    return paths


def save_final_embeddings(
    output_dir: Path,
    final_embeddings: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    paths = {}
    for section, tensor in final_embeddings.items():
        path = output_dir / f"final_embeddings_{section}.npy"
        np.save(path, tensor_to_numpy(tensor))
        paths[section] = str(path)
    return paths


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
    save_candidate_qc: bool = False,
) -> dict[str, dict[str, str]]:
    ensure_dir(output_dir)
    files: dict[str, dict[str, str]] = {}
    if not ot_prior:
        return files

    for (source_section, target_section), prior in ot_prior.items():
        pair_key = f"{source_section}_to_{target_section}"
        required = ["topk_idx", "topk_weight", "confidence", "row_mass"]
        if any(name not in prior for name in required):
            continue

        arrays = {
            "topk_idx": prior["topk_idx"].detach().cpu().numpy(),
            "topk_weight": prior["topk_weight"].detach().cpu().numpy(),
            "confidence": prior["confidence"].detach().cpu().numpy(),
            "row_mass": prior["row_mass"].detach().cpu().numpy(),
        }
        if save_candidate_qc:
            for optional_name in ["raw_topk_mass", "topk_coverage", "tail_mass", "target_hit_count"]:
                if optional_name in prior:
                    arrays[optional_name] = prior[optional_name].detach().cpu().numpy()
        paths = {
            name: output_dir / f"{pair_key}_{name}.npy"
            for name in arrays.keys()
        }
        paths.update({
            "metadata": output_dir / f"{pair_key}_metadata.json",
        })
        for name, array in arrays.items():
            np.save(paths[name], array)

        metadata = {
            "source_section": source_section,
            "target_section": target_section,
            "direction_meaning": "source receives information from target",
            "topk": int(arrays["topk_idx"].shape[1]) if arrays["topk_idx"].ndim == 2 else None,
            "n_source": int(final_embeddings[source_section].shape[0]),
            "n_target": int(final_embeddings[target_section].shape[0]),
            "modalities_used": list(prior.get("modalities_used", [])),
            "run_mode": run_mode,
            "note": "Saved sparse top-k UOT prior from model.ot_prior after final evaluation. X_to_Y means source X receives information from target Y. Dense P was not saved.",
        }
        metadata.update(prior.get("metadata", {}))
        with open(paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(json_safe(metadata), handle, indent=2, ensure_ascii=False)
        files[pair_key] = {name: str(path) for name, path in paths.items()}
    return files


def summarize_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_embedding_shapes": {
            section: list(tensor.shape)
            for section, tensor in outputs["final_embeddings"].items()
        },
        "reconstruction_keys": {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        },
        "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
        "losses": {
            key: float(value.detach().cpu())
            for key, value in outputs["losses"].items()
        },
        "total_loss_finite": bool(torch.isfinite(outputs["losses"]["total_loss"]).item()),
    }


def build_model_config(args):
    model_config = resolve_model_config(
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
        explicit_overrides={**feature_graph_overrides(args), "loss": {"lambda_contrast": (
            float(args.lambda_contrast) if args.lambda_contrast is not None else None
        )}},
    )
    return model_config


def resolve_run_config(args, *, samples=None, dataset_name="CRC Stereo-CITE-seq"):
    samples = SAMPLES if samples is None else samples
    return describe_run_config(
        args, build_model_config(args), dataset=dataset_name,
        section_order=list(samples), modalities=["RNA", "Protein"],
        preprocessing={
            "n_comps": args.n_comps,
            "hvg_num": args.hvg_num,
            "hvg_num_by_modality": {"RNA": args.hvg_num, "Protein": None},
            "target_sum": None,
            "use_harmony": not args.no_harmony,
        },
    )


def run_crc_pipeline(
    args, *, samples=None, read_pair=None, prepare_rna=None,
    filter_shared_genes=None, status_prefix="CRC_STEREOCITE", run_config=None,
) -> dict[str, Any]:
    samples = SAMPLES if samples is None else samples
    first_section, second_section = samples
    if run_config is None:
        run_config = resolve_run_config(args, samples=samples)
    section_key_map = {"s1": first_section, "s2": second_section}
    read_pair = read_backed_pair if read_pair is None else read_pair
    prepare_rna = prepare_rna_var_names_make_unique if prepare_rna is None else prepare_rna
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
        for section, sample_dir in samples.items():
            rna, adt = read_pair(data_dir, sample_dir)
            backed_rna[section] = rna
            backed_adt[section] = adt
            rna_info[section], duplicate_tables[section] = prepare_rna(rna)
            alignment_summary[section] = validate_rna_adt_alignment(section, rna, adt)
        memory_monitor.record("read_raw_h5ad_make_unique_validate_alignment")

        rna003 = backed_rna[first_section]
        rna006 = backed_rna[second_section]
        shared_set_006 = set(rna006.var_names.astype(str))
        shared_genes_all = [gene for gene in rna003.var_names.astype(str) if gene in shared_set_006]
        if filter_shared_genes is not None:
            shared_genes_all = filter_shared_genes(rna003, rna006, shared_genes_all)
        if not shared_genes_all:
            raise ValueError("No shared RNA genes found after var_names_make_unique().")
        selected_shared_genes = (
            shared_genes_all[: args.max_shared_genes]
            if args.max_shared_genes is not None
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
        memory_monitor.record(
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

        for section, table in duplicate_tables.items():
            table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
        save_list(output_dir / "shared_gene_symbols_make_unique.txt", selected_shared_genes)
        save_list(output_dir / "adt_marker_list.txt", adt_markers_003)
        memory_monitor.record("saved_lightweight_metadata")

        obs_indices = {
            section: select_obs_indices(
                backed_rna[section].n_obs,
                args.max_spots_per_section,
                args.spot_sampling,
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
        saved_selected_spot_indices = {}
        if args.save_outputs:
            saved_selected_spot_indices = save_selected_spot_indices(output_dir, obs_indices, backed_rna)
        memory_monitor.record(
            "selected_spots",
            extra={
                "max_spots_per_section": int(args.max_spots_per_section)
                if args.max_spots_per_section is not None
                else None,
                "spot_sampling": args.spot_sampling,
            },
        )

        rna003_mem = subset_to_memory(backed_rna[first_section], obs_indices[first_section], selected_shared_genes)
        rna006_mem = subset_to_memory(backed_rna[second_section], obs_indices[second_section], selected_shared_genes)
        adt003_mem = subset_to_memory(backed_adt[first_section], obs_indices[first_section], None)
        adt006_mem = subset_to_memory(backed_adt[second_section], obs_indices[second_section], None)
        memory_monitor.record(
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
        memory_monitor.record("constructed_data_dict")

        memory_monitor.reset_peak()
        memory_monitor.record("cosie_preprocessing_start")
        feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
            data_dict,
            **run_config["preprocessing"],
        )
        preprocessing_generated_keys = list(feature_dict_raw.keys())
        feature_dict = rename_section_keys(feature_dict_raw, section_key_map)
        spatial_loc_dict = rename_section_keys(spatial_loc_dict_raw, section_key_map)
        section_order = [first_section, second_section]
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
        resolved_modality_order = list(model._resolve_modality_order(feature_dict[first_section]))
        if resolved_modality_order != ["RNA", "Protein"]:
            raise ValueError(f"Expected ['RNA', 'Protein'], got {resolved_modality_order}.")

        memory_monitor.reset_peak()
        memory_monitor.record("initial_ot_prior_start")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        memory_monitor.record("initial_ot_prior_end")
        initial_prior = model.ot_prior[(first_section, second_section)]
        expected_keys = {(first_section, second_section), (second_section, first_section)}
        actual_keys = set((model.ot_prior or {}).keys())
        if not expected_keys.issubset(actual_keys):
            raise ValueError(f"Bidirectional OT prior is missing direction keys: {expected_keys - actual_keys}")
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        if initial_prior.get("metadata", {}).get("ot_prior_mode") != "candidate_sparse":
            raise ValueError("Expected candidate_sparse initial OT prior metadata.")
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
                    memory_recorder=make_forward_memory_recorder(
                        memory_monitor,
                        bool(args.log_cuda_memory_detail),
                        0,
                        "dry_run_forward",
                    ),
                )
                memory_monitor.record("dry_run_forward_end", epoch=0)

        prior = outputs["ot_prior"][(first_section, second_section)]
        reconstruction_keys = {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        }
        ot_modalities_used = list(prior.get("modalities_used", []))
        total_loss_finite = bool(torch.isfinite(outputs["losses"]["total_loss"]).item())
        if reconstruction_keys != {first_section: ["Protein", "RNA"], second_section: ["Protein", "RNA"]}:
            raise ValueError(f"Unexpected reconstruction keys: {reconstruction_keys}")
        # Initial UOT remains multimodal RNA/Protein. Dynamic refresh preserves
        # the original detached final-embedding source by default.
        accepted_ot_modalities = [["RNA", "Protein"]]
        if args.train and ot_updates:
            accepted_ot_modalities.append(["final_embedding"])
            accepted_ot_modalities.append(["ot_embedding"])
        accepted_ot_modalities.append(["RNA", "Protein"])
        accepted_ot_modalities.append(["ot_embedding"])
        if ot_modalities_used not in accepted_ot_modalities:
            raise ValueError(f"Unexpected OT modalities_used: {ot_modalities_used}")
        reverse_prior = outputs["ot_prior"].get((second_section, first_section))
        if reverse_prior is None:
            raise ValueError(f"Bidirectional OT prior is missing {second_section}<-{first_section} direction.")
        if int(prior["topk_idx"].max().item()) >= int(feature_dict[second_section]["RNA"].shape[0]):
            raise ValueError(f"{first_section}<-{second_section} topk_idx contains out-of-range target indices.")
        if int(reverse_prior["topk_idx"].max().item()) >= int(feature_dict[first_section]["RNA"].shape[0]):
            raise ValueError(f"{second_section}<-{first_section} topk_idx contains out-of-range target indices.")
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

        sample_summaries = {}
        for section in section_order:
            sample_summaries[section] = {
                **alignment_summary[section],
                "shared_rna_gene_count_total": int(len(shared_genes_all)),
                "shared_rna_gene_count_used": int(len(selected_shared_genes)),
                "adt_marker_count": int(len(adt_markers_003)),
                "subset_spot_count": int(feature_dict[section]["RNA"].shape[0]),
            }

        summary = {
            "resolved_config": run_config,
            "mode": "train" if args.train else "dry_run",
            "input_data_path": str(data_dir),
            "output_dir": str(output_dir),
            "section_names": section_order,
            "preprocessing_generated_keys": preprocessing_generated_keys,
            "section_key_mapping": section_key_map,
            "seed": int(args.seed),
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            "use_harmony": not args.no_harmony,
            "spot_sampling": args.spot_sampling,
            "selected_obs_names_preview": selected_obs_names_preview,
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
            "faiss_query_batch_size": (
                int(args.faiss_query_batch_size)
                if args.faiss_query_batch_size is not None
                else None
            ),
            "dynamic_candidate_source": "ot",
            "architecture": (
                "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder"
            ),
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
            "cuda_memory_event_count_before_summary": int(memory_monitor.event_count),
            "save_candidate_qc": bool(args.save_candidate_qc),
            "ot_prior_metadata": {
                f"{source}_to_{target}": prior.get("metadata", {})
                for (source, target), prior in (outputs.get("ot_prior") or {}).items()
            },
            "rna_make_unique": rna_info,
            "shared_gene_count_total": int(len(shared_genes_all)),
            "shared_gene_count_used": int(len(selected_shared_genes)),
            "shared_gene_artificial_suffix_count_total": int(len(shared_suffix_genes_all)),
            "shared_gene_artificial_suffix_count_used": int(len(shared_suffix_genes_selected)),
            "shared_gene_artificial_suffix_examples": shared_suffix_genes_all[:20],
            "max_spots_per_section": (
                int(args.max_spots_per_section)
                if args.max_spots_per_section is not None
                else None
            ),
            "max_shared_genes": int(args.max_shared_genes) if args.max_shared_genes is not None else None,
            "protein_marker_count": int(len(adt_markers_003)),
            "adt_var_names_unique": {
                first_section: bool(backed_adt[first_section].var_names.is_unique),
                second_section: bool(backed_adt[second_section].var_names.is_unique),
            },
            "adt_marker_set_same": bool(adt_marker_set_same),
            "adt_marker_order_same": bool(adt_marker_order_same),
            "alignment": sample_summaries,
            "data_dict": summarize_data_dict(data_dict),
            "feature_dict_shapes": summarize_feature_dict(feature_dict),
            "spatial_loc_dict_shapes": summarize_spatial_loc_dict(spatial_loc_dict),
            "processed_data_dict_generated": processed_data_dict is not None,
            "continued_into_preprocessing": True,
            "feature_dict_generated": True,
            "model_initialized": True,
            "initial_ot_prior_initialized": True,
            "epoch_1_completed": bool(history and len(history) >= 1),
            "epoch_2_completed": bool(history and len(history) >= 2),
            "resolved_modality_order": resolved_modality_order,
            "final_embedding_shapes": {
                section: list(tensor.shape)
                for section, tensor in outputs["final_embeddings"].items()
            },
            "reconstruction_keys": reconstruction_keys,
            "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
            "initial_ot_prior_modalities_used": {
                f"{first_section}_to_{second_section}": initial_ot_modalities_used,
            },
            "ot_prior_modalities_used": {
                f"{first_section}_to_{second_section}": ot_modalities_used,
            },
            "losses": {
                key: float(value.detach().cpu())
                for key, value in outputs["losses"].items()
            },
            "total_loss_finite": total_loss_finite,
            "training_history": history,
            "saved_embeddings": bool(embedding_paths),
            "saved_ot_prior_topk": bool(ot_prior_topk_files),
            "ot_prior_topk_saved": bool(ot_prior_topk_files),
            "ot_prior_topk_reason": None if ot_prior_topk_files else (
                "not requested" if not args.save_ot_prior_topk else "no sparse OT prior available"
            ),
            "saved_files": {
                "run_summary": str(output_dir / "run_summary.json"),
                "loss_history": str(output_dir / "loss_history.json") if history is not None else None,
                "shared_gene_symbols_make_unique": str(output_dir / "shared_gene_symbols_make_unique.txt"),
                "adt_marker_list": str(output_dir / "adt_marker_list.txt"),
                f"duplicate_gene_summary_{first_section}": str(output_dir / f"duplicate_gene_summary_{first_section}.csv"),
                f"duplicate_gene_summary_{second_section}": str(output_dir / f"duplicate_gene_summary_{second_section}.csv"),
                "selected_spot_indices": saved_selected_spot_indices,
                "spatial": spatial_paths,
                "final_embeddings": embedding_paths,
                "ot_prior_topk": ot_prior_topk_files,
            },
            "confirmations": {
                "saved_processed_h5ad": False,
                "overwrote_original_h5ad": False,
                "created_processed_data_directory": False,
                "fabricated_HE": False,
                "fabricated_Metabolite": False,
                "adt_used_as_Protein": True,
                "adt_used_as_Metabolite": False,
                "count_aggregation_performed": False,
                "full_crc_uot_constructed": False,
            },
            "notes": [
                "Suffixes such as MATR3-1 and ABCF2-1 are engineering names generated by AnnData var_names_make_unique(); they are not biological gene IDs.",
                f"selected shared genes follow {first_section} make_unique order.",
                "max_shared_genes limits the dry-run gene list for speed and should be revisited for formal experiments.",
            ],
        }
        memory_monitor.record("write_run_summary_start")
        with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
        if history is not None:
            memory_monitor.record("write_loss_history_start")
            with open(output_dir / "loss_history.json", "w", encoding="utf-8") as handle:
                json.dump(json_safe(history), handle, indent=2, ensure_ascii=False)
            memory_monitor.record("write_loss_history_end")
        memory_monitor.record("write_run_summary_end")

        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
        print(f"{status_prefix}_TRAIN: PASS" if args.train else f"{status_prefix}_DRY_RUN: PASS")
        return summary
    finally:
        for adata_obj in list(backed_rna.values()) + list(backed_adt.values()):
            if getattr(adata_obj, "isbacked", False):
                adata_obj.file.close()


def main() -> None:
    args = parse_args()
    run_crc_pipeline(args, run_config=resolve_run_config(args))


if __name__ == "__main__":
    main()
