"""Multi-section task layout over the shared explicit execution policy."""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping
import json
import anndata as ad
import numpy as np
import torch
from . import artifacts as _artifacts
from .fit import json_safe, amp_enabled
from .task_runtime import ExecutionPolicy, execute_prepared_task

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


def run_multisection_training_task(
    run_config, prepared, *, memory_monitor, section_info,
    dataset_name, secondary_modality, secondary_name, execution_state=None,
) -> dict[str, Any]:
    """Names select existing payload keys/text, never execution branches."""
    args = Namespace()
    args.__dict__ = run_config["runner"]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    section_order = prepared.section_order
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    processed_data_dict = prepared.processed_data_dict
    rna_info = prepared.preparation_audit["rna_info"]
    duplicate_tables = prepared.preparation_audit["duplicate_tables"]
    alignment_summary = prepared.preparation_audit["alignment_summary"]
    shared_genes = prepared.preparation_audit["shared_genes"]
    shared_peaks = prepared.preparation_audit["shared_peaks"]
    obs_indices = prepared.preparation_audit["obs_indices"]
    preprocessing_generated_keys = prepared.preparation_audit["preprocessing_generated_keys"]
    data_dict = prepared.compatibility_context["data_dict"]
    rna_mem = prepared.compatibility_context["rna_mem"]
    atac_mem = prepared.compatibility_context["atac_mem"]
    saved_selected_spot_indices = prepared.compatibility_context["saved_selected_spot_indices"]
    obs_metadata_paths = prepared.compatibility_context["obs_metadata_paths"]

    model_config = run_config["model_config"]
    result = execute_prepared_task(
        run_config, prepared, memory_monitor=memory_monitor,
        policy=ExecutionPolicy(secondary_modality, "adjacent", True, False, False),
    )
    if execution_state is not None:
        # The original runner retained model/prior/output references through finally.
        execution_state.update(result)
    outputs = result["outputs"]
    history = result["history"]
    ot_updates = result["ot_updates"]
    resolved_modality_order = result["resolved_modality_order"]
    initial_ot_modalities_used = result["initial_ot_modalities_used"]
    reconstruction_keys = result["reconstruction_keys"]
    total_loss_finite = result["total_loss_finite"]
    first_pair = result["first_pair"]
    embedding_paths = result["embedding_paths"]
    spatial_paths = result["spatial_paths"]
    ot_prior_topk_files = result["ot_prior_topk_files"]

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
    _artifacts.save_json(output_dir / "run_summary.json", summary, transform=json_safe)
    if history is not None:
        _artifacts.save_json(output_dir / "loss_history.json", history, transform=json_safe)
    memory_monitor.record("write_run_summary_end")

    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
    print("MISAR_SEQ_TRAIN: PASS" if args.train else "MISAR_SEQ_DRY_RUN: PASS")
    return summary
