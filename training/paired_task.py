"""Paired RNA/Protein training task over resolved config and PreparedDataset.

No argv parsing, config resolving, seeding, input reading or resource ownership.
The caller supplies the existing monitor and already loaded marker-name evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from argparse import Namespace
from typing import Any, Mapping

import anndata as ad
import numpy as np
import torch

from data_io.paired_preparation import PreparedDataset
from model.stage_model import StageMultiModalModel
from .task_runtime import ExecutionPolicy, execute_prepared_task
from . import artifacts as _artifacts
from .artifacts import save_final_embeddings, save_spatial_arrays, save_ot_prior_topk
from .fit import (
    json_safe, amp_enabled, initialize_model_ot_prior, train_small_crc_model,
    run_one_forward, make_forward_memory_recorder,
)

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


def run_paired_training_task(
    run_config, prepared: PreparedDataset, *, memory_monitor,
    protein_var_names, status_prefix="CRC_STEREOCITE",
) -> dict[str, Any]:
    """Run the existing paired task; status_prefix is display text only.

    protein_var_names holds references to already loaded input Index objects,
    solely for the legacy summary. It does not open or read raw data.
    """
    # Keep the original Namespace type and shared resolved option storage.
    args = Namespace()
    args.__dict__ = run_config["runner"]
    first_section, second_section = run_config["section_order"]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    section_order = prepared.section_order
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    processed_data_dict = prepared.processed_data_dict
    data_dict = prepared.compatibility_context["data_dict"]
    saved_selected_spot_indices = prepared.compatibility_context["saved_selected_spot_indices"]
    rna_info = prepared.preparation_audit["rna_info"]
    duplicate_tables = prepared.preparation_audit["duplicate_tables"]
    alignment_summary = prepared.preparation_audit["alignment_summary"]
    shared_genes_all = prepared.preparation_audit["shared_genes_all"]
    selected_shared_genes = prepared.preparation_audit["selected_shared_genes"]
    shared_suffix_genes_selected = prepared.preparation_audit["shared_suffix_genes_selected"]
    shared_suffix_genes_all = prepared.preparation_audit["shared_suffix_genes_all"]
    adt_markers_003 = prepared.preparation_audit["adt_markers_003"]
    adt_marker_set_same = prepared.preparation_audit["adt_marker_set_same"]
    adt_marker_order_same = prepared.preparation_audit["adt_marker_order_same"]
    obs_indices = prepared.preparation_audit["obs_indices"]
    selected_obs_names_preview = prepared.preparation_audit["selected_obs_names_preview"]
    preprocessing_generated_keys = prepared.preparation_audit["preprocessing_generated_keys"]
    section_key_map = prepared.preparation_audit["section_key_map"]

    model_config = run_config["model_config"]
    result = execute_prepared_task(
        run_config, prepared, memory_monitor=memory_monitor,
        policy=ExecutionPolicy("Protein", "paired", False, True, True),
    )
    outputs = result["outputs"]
    history = result["history"]
    ot_updates = result["ot_updates"]
    resolved_modality_order = result["resolved_modality_order"]
    initial_ot_modalities_used = result["initial_ot_modalities_used"]
    reconstruction_keys = result["reconstruction_keys"]
    total_loss_finite = result["total_loss_finite"]
    ot_modalities_used = result["ot_modalities_used"]
    embedding_paths = result["embedding_paths"]
    spatial_paths = result["spatial_paths"]
    ot_prior_topk_files = result["ot_prior_topk_files"]

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
            first_section: bool(protein_var_names[first_section].is_unique),
            second_section: bool(protein_var_names[second_section].is_unique),
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
    _artifacts.save_json(output_dir / "run_summary.json", summary, transform=json_safe)
    if history is not None:
        memory_monitor.record("write_loss_history_start")
        _artifacts.save_json(output_dir / "loss_history.json", history, transform=json_safe)
        memory_monitor.record("write_loss_history_end")
    memory_monitor.record("write_run_summary_end")

    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
    print(f"{status_prefix}_TRAIN: PASS" if args.train else f"{status_prefix}_DRY_RUN: PASS")
    return summary


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

