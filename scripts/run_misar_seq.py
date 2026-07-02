#!/usr/bin/env python3
"""MISAR-seq RNA+ATAC training pipeline for StageMultiModalModel."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.configure import get_default_model_config
from model.data_preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel
from run_crc_stereocite import (
    CudaMemoryMonitor,
    amp_enabled,
    ensure_dir,
    initialize_model_ot_prior,
    json_safe,
    run_one_forward,
    save_final_embeddings,
    save_ot_prior_topk,
    save_spatial_arrays,
    select_obs_indices,
    subset_to_memory,
    train_small_crc_model,
)


SECTION_INFO = {
    "dataset1": {"dir": "dataset1", "sample": "E18_5-S1", "stage": "E18.5"},
    "dataset2": {"dir": "dataset2", "sample": "E15_5-S1", "stage": "E15.5"},
    "dataset3": {"dir": "dataset3", "sample": "E13_5-S1", "stage": "E13.5"},
    "dataset4": {"dir": "dataset4", "sample": "E11_0-S1", "stage": "E11.0"},
}
DEFAULT_SECTION_ORDER = ["dataset4", "dataset3", "dataset2", "dataset1"]
LABEL_COLUMNS = [
    "Sample",
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MISAR-seq RNA+ATAC StageMultiModalModel pipeline.")
    parser.add_argument("--data_dir", default="/home/hujinlan/spa_mo_model/data/MISAR-seq")
    parser.add_argument(
        "--section_order",
        default=",".join(DEFAULT_SECTION_ORDER),
        help="Comma-separated section order used for adjacent OT links.",
    )
    parser.add_argument("--max_spots_per_section", type=int, default=None)
    parser.add_argument("--spot_sampling", choices=["first", "random"], default="first")
    parser.add_argument("--max_shared_genes", type=int, default=None)
    parser.add_argument("--max_shared_peaks", type=int, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ot_prior_mode", choices=["dense", "candidate_sparse"], default="candidate_sparse")
    parser.add_argument("--bidirectional_ot_attention", action="store_true")
    parser.add_argument("--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default="faiss_ivf")
    parser.add_argument("--initial_modality_candidate_k", type=int, default=100)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--faiss_nlist", type=int, default=256)
    parser.add_argument("--faiss_nprobe", type=int, default=32)
    parser.add_argument("--faiss_device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--faiss_train_sample_size", type=int, default=20000)
    parser.add_argument("--faiss_query_batch_size", type=int, default=2048)
    parser.add_argument("--dynamic_candidate_source", choices=["fused", "final"], default="final")
    parser.add_argument("--uot_epsilon", type=float, default=0.05)
    parser.add_argument("--uot_tau_a", type=float, default=1.0)
    parser.add_argument("--uot_tau_b", type=float, default=1.0)
    parser.add_argument("--uot_stabilizer", type=float, default=1e-8)
    parser.add_argument("--uot_max_iter", type=int, default=100)
    parser.add_argument("--spatial_knn_k", type=int, default=5)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=50000)
    parser.add_argument("--graphsage_dropout", type=float, default=0.1)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--ot_attention_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--training_loss_only", action="store_true")
    parser.add_argument("--decoder_chunk_size", type=int, default=50000)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=50000)
    parser.add_argument("--checkpoint_ot_attention", action="store_true")
    parser.add_argument("--checkpoint_encoder_fusion", action="store_true")
    parser.add_argument("--checkpoint_decoder_chunks", action="store_true")
    parser.add_argument("--checkpoint_graph_encoder", action="store_true")
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--cache_spatial_graphs", action="store_true")
    parser.add_argument("--save_candidate_qc", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--save_ot_prior_topk", action="store_true")
    parser.add_argument("--log_cuda_memory", action="store_true")
    parser.add_argument("--log_cuda_memory_detail", action="store_true")
    parser.add_argument("--output_dir", default="/home/hujinlan/spa_mo_model/results/misar_seq/dry_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_comps", type=int, default=50)
    parser.add_argument("--hvg_num", type=int, default=3000, help="HVG count for RNA.")
    parser.add_argument("--hvg_num_atac", type=int, default=3000, help="Highly variable peak count for ATAC.")
    parser.add_argument("--no_harmony", action="store_true")
    return parser.parse_args()


def parse_section_order(text: str) -> list[str]:
    order = [item.strip() for item in text.split(",") if item.strip()]
    if not order:
        raise ValueError("--section_order must contain at least one section.")
    unknown = [section for section in order if section not in SECTION_INFO]
    if unknown:
        raise ValueError(f"Unknown MISAR sections in --section_order: {unknown}")
    if len(set(order)) != len(order):
        raise ValueError("--section_order contains duplicated sections.")
    return order


def release_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_backed_pair(data_dir: Path, section: str) -> tuple[ad.AnnData, ad.AnnData]:
    section_dir = data_dir / SECTION_INFO[section]["dir"]
    rna_path = section_dir / "adata_RNA.h5ad"
    atac_path = section_dir / "adata_ATAC.h5ad"
    if not rna_path.exists():
        raise FileNotFoundError(rna_path)
    if not atac_path.exists():
        raise FileNotFoundError(atac_path)
    return ad.read_h5ad(rna_path, backed="r"), ad.read_h5ad(atac_path, backed="r")


def make_unique_rna_var_names(section: str, rna: ad.AnnData) -> tuple[dict[str, Any], pd.DataFrame]:
    before_unique = bool(rna.var_names.is_unique)
    original = pd.Index(rna.var_names.astype(str))
    counts = pd.Series(original).value_counts()
    duplicated = original.duplicated(keep=False)
    rna.var["gene_symbol_original"] = original.to_numpy()
    rna.var["was_duplicate_gene_symbol"] = duplicated
    rna.var["gene_symbol_original_count"] = [int(counts[symbol]) for symbol in original]
    rna.var_names = pd.Index(original)
    rna.var_names_make_unique()
    if not rna.var_names.is_unique:
        raise ValueError(f"{section}: RNA var_names are still not unique after var_names_make_unique().")
    rna.var["gene_symbol_make_unique"] = rna.var_names.astype(str)
    table = pd.DataFrame(
        {
            "gene_symbol_original": original.to_numpy(),
            "gene_symbol_make_unique": rna.var_names.astype(str).to_numpy(),
            "was_duplicate_gene_symbol": duplicated,
            "gene_symbol_original_count": [int(counts[symbol]) for symbol in original],
        }
    )
    duplicate_summary = (
        table[table["was_duplicate_gene_symbol"]]
        .groupby("gene_symbol_original", sort=True)
        .agg(
            count=("gene_symbol_make_unique", "size"),
            make_unique_names=("gene_symbol_make_unique", lambda values: ";".join(values)),
        )
        .reset_index()
    )
    return (
        {
            "shape": list(rna.shape),
            "var_names_unique_before": before_unique,
            "var_names_unique_after": bool(rna.var_names.is_unique),
            "duplicate_gene_symbol_groups": int(duplicate_summary.shape[0]),
            "duplicate_extra_columns": int((duplicate_summary["count"] - 1).sum())
            if not duplicate_summary.empty
            else 0,
        },
        duplicate_summary,
    )


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(spatial)
    return {
        "x": [float(arr[:, 0].min()), float(arr[:, 0].max())],
        "y": [float(arr[:, 1].min()), float(arr[:, 1].max())],
    }


def validate_rna_atac_alignment(section: str, rna: ad.AnnData, atac: ad.AnnData) -> dict[str, Any]:
    if list(rna.obs_names) != list(atac.obs_names):
        raise ValueError(f"{section}: RNA and ATAC obs_names are not identical.")
    if "spatial" not in rna.obsm:
        raise KeyError(f"{section}: RNA is missing obsm['spatial'].")
    if "spatial" not in atac.obsm:
        raise KeyError(f"{section}: ATAC is missing obsm['spatial'].")
    rna_spatial = np.asarray(rna.obsm["spatial"])
    atac_spatial = np.asarray(atac.obsm["spatial"])
    if rna_spatial.shape != atac_spatial.shape:
        raise ValueError(f"{section}: RNA and ATAC spatial shapes differ.")
    spatial_match = bool(np.allclose(rna_spatial, atac_spatial))
    if not spatial_match:
        raise ValueError(f"{section}: RNA and ATAC spatial coordinates differ.")
    label_summary = {}
    for column in LABEL_COLUMNS:
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


def common_var_names(
    adatas: Mapping[str, ad.AnnData],
    section_order: list[str],
    max_features: int | None,
    label: str,
) -> list[str]:
    common = pd.Index(adatas[section_order[0]].var_names.astype(str))
    for section in section_order[1:]:
        common = common.intersection(pd.Index(adatas[section].var_names.astype(str)))
    if common.empty:
        raise ValueError(f"No shared {label} features found across MISAR sections.")
    selected = list(common[:max_features]) if max_features is not None else list(common)
    if not selected:
        raise ValueError(f"No {label} features selected.")
    return selected


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
        obs["stage"] = SECTION_INFO[section]["stage"]
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


def make_model_config(args) -> dict[str, Any]:
    config = get_default_model_config()
    config["model"]["modalities_supported"] = ["RNA", "ATAC"]
    config["model"]["valid_modality_sets"] = [["RNA", "ATAC"]]
    config["training"]["device"] = args.device
    config["training"]["epochs"] = int(args.epochs)
    config["training"]["lr"] = float(args.lr)
    config["training"]["weight_decay"] = float(args.weight_decay)
    config["encoder"]["dropout"] = float(args.encoder_dropout)
    config["fusion"]["dropout"] = float(args.fusion_dropout)
    config["graphsage"]["dropout"] = float(args.graphsage_dropout)
    config["ot_attention"]["dropout"] = float(args.ot_attention_dropout)
    config["decoder"]["dropout"] = float(args.decoder_dropout)
    if args.lambda_contrast is not None:
        config["loss"]["lambda_contrast"] = float(args.lambda_contrast)
    config["uot"]["max_iter"] = int(args.uot_max_iter)
    config["uot"]["topk"] = int(args.attention_topk)
    config["uot"]["epsilon_update"] = float(args.uot_epsilon)
    config["uot"]["tau_a"] = float(args.uot_tau_a)
    config["uot"]["tau_b"] = float(args.uot_tau_b)
    config["uot"]["check_every"] = 10
    config["uot"]["tol"] = 1e-5
    config["uot"]["update_interval"] = int(args.update_interval)
    config["graph"]["knn_neighbors_spatial"] = int(args.spatial_knn_k)
    config["graphsage"]["edge_batch_size"] = int(args.graphsage_edge_batch_size)
    config["reconstruction"]["lambda_by_modality"]["ATAC"] = 1.0
    return config


def validate_args(args) -> None:
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
        "hvg_num_atac",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    for name in ["decoder_chunk_size", "ot_attention_source_chunk_size"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.log_cuda_memory_detail:
        args.log_cuda_memory = True


def run_misar_pipeline(args) -> dict[str, Any]:
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    section_order = parse_section_order(args.section_order)
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
            rna, atac = read_backed_pair(data_dir, section)
            backed_rna[section] = rna
            backed_atac[section] = atac
            rna_info[section], duplicate_tables[section] = make_unique_rna_var_names(section, rna)
            alignment_summary[section] = validate_rna_atac_alignment(section, rna, atac)
        memory_monitor.record("read_raw_h5ad_make_unique_validate_alignment")

        shared_genes = common_var_names(backed_rna, section_order, args.max_shared_genes, "RNA genes")
        shared_peaks = common_var_names(backed_atac, section_order, args.max_shared_peaks, "ATAC peaks")
        for section, table in duplicate_tables.items():
            table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
        save_list(output_dir / "shared_gene_symbols_make_unique.txt", shared_genes)
        save_list(output_dir / "shared_atac_peaks.txt", shared_peaks)
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
                f"atac_{section}_shape": list(atac_mem[section].shape)
                for section in section_order
            },
        )

        obs_metadata_paths = {}
        if args.save_outputs:
            obs_metadata_paths = save_obs_metadata(output_dir, section_order, rna_mem, obs_indices, backed_rna)

        data_dict = {
            "RNA": [rna_mem[section] for section in section_order],
            "ATAC": [atac_mem[section] for section in section_order],
        }
        memory_monitor.reset_peak()
        memory_monitor.record("cosie_preprocessing_start")
        feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
            data_dict,
            n_comps=int(args.n_comps),
            hvg_num=int(args.hvg_num),
            hvg_num_by_modality={"RNA": int(args.hvg_num), "ATAC": int(args.hvg_num_atac)},
            target_sum=None,
            use_harmony=not args.no_harmony,
            metacell=False,
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

        model_config = make_model_config(args)
        memory_monitor.reset_peak()
        memory_monitor.record("model_init_start")
        model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
        memory_monitor.record("model_init_end")
        resolved_modality_order = list(model._resolve_modality_order(feature_dict[section_order[0]]))
        if resolved_modality_order != ["RNA", "ATAC"]:
            raise ValueError(f"Expected ['RNA', 'ATAC'], got {resolved_modality_order}.")

        memory_monitor.reset_peak()
        memory_monitor.record("initial_ot_prior_start")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        memory_monitor.record("initial_ot_prior_end")
        first_pair = (section_order[0], section_order[1]) if len(section_order) > 1 else None
        if first_pair is None:
            raise ValueError("MISAR run requires at least two sections.")
        initial_prior = model.ot_prior[first_pair]
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        if args.ot_prior_mode == "dense" and initial_ot_modalities_used != ["RNA", "ATAC"]:
            raise ValueError(f"Unexpected initial OT modalities_used: {initial_ot_modalities_used}")
        if args.bidirectional_ot_attention and args.ot_prior_mode == "candidate_sparse":
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
                    bidirectional_ot_attention=bool(args.bidirectional_ot_attention),
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
        expected_reconstruction_keys = {section: ["ATAC", "RNA"] for section in section_order}
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
            "mode": "train" if args.train else "dry_run",
            "dataset": "MISAR-seq",
            "input_data_path": str(data_dir),
            "output_dir": str(output_dir),
            "section_names": section_order,
            "section_info": {section: SECTION_INFO[section] for section in section_order},
            "preprocessing_generated_keys": preprocessing_generated_keys,
            "seed": int(args.seed),
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            "hvg_num_atac": int(args.hvg_num_atac),
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
            "ot_prior_mode": args.ot_prior_mode,
            "bidirectional_ot_attention": bool(args.bidirectional_ot_attention),
            "candidate_backend": args.candidate_backend,
            "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
            "candidate_k": int(args.candidate_k),
            "attention_topk": int(args.attention_topk),
            "faiss_nlist": int(args.faiss_nlist),
            "faiss_nprobe": int(args.faiss_nprobe),
            "faiss_device": args.faiss_device,
            "faiss_train_sample_size": int(args.faiss_train_sample_size),
            "faiss_query_batch_size": int(args.faiss_query_batch_size),
            "dynamic_candidate_source": args.dynamic_candidate_source,
            "uot_epsilon": float(args.uot_epsilon),
            "uot_tau_a": float(args.uot_tau_a),
            "uot_tau_b": float(args.uot_tau_b),
            "uot_stabilizer": float(args.uot_stabilizer),
            "spatial_knn_k": int(args.spatial_knn_k),
            "graphsage_edge_batch_size": int(args.graphsage_edge_batch_size),
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
                "shared_atac_peaks": str(output_dir / "shared_atac_peaks.txt"),
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
                "modalities_used": ["RNA", "ATAC"],
                "atac_used_as_true_ATAC": True,
                "atac_used_as_Protein": False,
                "atac_used_as_Metabolite": False,
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
    run_misar_pipeline(args)


if __name__ == "__main__":
    main()
