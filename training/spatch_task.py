"""SPATCH model-ready task; GPU admission and raw/reuse preparation belong to caller."""
from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
import os
from . import artifacts as _artifacts
from .artifacts import save_final_embeddings
from .fit import CudaMemoryMonitor, initialize_model_ot_prior, json_safe, train_small_crc_model
from .task_runtime import initialize_prepared_task


@dataclass(frozen=True)
class ModalityOrderPolicy:
    section: str
    expected_modalities: tuple[str, ...]


def _json_pathlike_default(value):
    """Serialize summary paths without accepting other unsupported objects."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_spatch_training_task(run_config, prepared, *, policy: ModalityOrderPolicy, gpu, execution_state):
    """Borrow resolved inputs; caller keeps execution references through its finally."""
    args = Namespace()
    args.__dict__ = run_config["runner"]
    feature_dict = prepared.feature_dict
    spatial_dict = prepared.spatial_loc_dict
    audits = prepared.preparation_audit
    cache_info = prepared.compatibility_context["cache_info"]
    processed = prepared.processed_data_dict
    section_order = prepared.section_order
    config = run_config["model_config"]
    model, _, _ = initialize_prepared_task(config, prepared, args, defer_prior=True)
    execution_state["model"] = model
    if list(model._resolve_modality_order(feature_dict[policy.section])) != list(policy.expected_modalities):
        raise ValueError("Unexpected modality order.")

    monitor = CudaMemoryMonitor(True, args.output_dir, "cuda")
    execution_state["monitor"] = monitor
    initialize_model_ot_prior(model, feature_dict, section_order, args)
    history, outputs, ot_updates = train_small_crc_model(
        model, feature_dict, spatial_dict, processed, section_order, args, monitor
    )
    execution_state.update(history=history, outputs=outputs, ot_updates=ot_updates)
    embedding_paths = save_final_embeddings(args.output_dir, outputs["final_embeddings"])
    summary = {
        "resolved_config": run_config,
        "dataset": "spatch",
        "method": "spa_mo_model",
        "mode": "train",
        "input_data_path": str(args.preprocessed_cache_dir),
        "input_data_kind": "model_ready_preprocessed_cache",
        "output_dir": str(args.output_dir),
        "section_names": section_order,
        "full_spot": True,
        "spatial_block_aggregation": False,
        "gpu": gpu,
        "train": True,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_comps": args.n_comps,
        "hvg_num": args.hvg_num,
        "hvg_num_by_modality": {
            "RNA": args.hvg_num,
            "Protein": None,
            "HE": None,
        },
        "use_harmony": not args.no_harmony,
        "spot_sampling": "all",
        "max_spots_per_section": None,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_contrast": args.lambda_contrast,
        "update_interval": args.update_interval,
        "uot_max_iter": args.uot_max_iter,
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "candidate_backend": args.candidate_backend,
        "initial_modality_candidate_k": args.initial_modality_candidate_k,
        "candidate_k": args.candidate_k,
        "attention_topk": args.attention_topk,
        "faiss_nlist": args.faiss_nlist,
        "faiss_nprobe": args.faiss_nprobe,
        "faiss_device": args.faiss_device,
        "faiss_train_sample_size": args.faiss_train_sample_size,
        "faiss_query_batch_size": args.faiss_query_batch_size,
        "dynamic_candidate_source": "ot",
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "uot_epsilon": args.uot_epsilon,
        "uot_tau_a": args.uot_tau_a,
        "uot_tau_b": args.uot_tau_b,
        "uot_stabilizer": args.uot_stabilizer,
        "spatial_knn_k": args.spatial_knn_k,
        "graphsage_edge_batch_size": args.graphsage_edge_batch_size,
        "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        "training_loss_only": args.training_loss_only,
        "decoder_chunk_size": args.decoder_chunk_size,
        "ot_attention_source_chunk_size": args.ot_attention_source_chunk_size,
        "checkpoint_ot_attention": args.checkpoint_ot_attention,
        "checkpoint_encoder_fusion": args.checkpoint_encoder_fusion,
        "checkpoint_decoder_chunks": args.checkpoint_decoder_chunks,
        "checkpoint_graph_encoder": args.checkpoint_graph_encoder,
        "amp_dtype": args.amp_dtype,
        "amp_enabled": args.amp_dtype != "none",
        "cache_spatial_graphs": args.cache_spatial_graphs,
        "log_cuda_memory": args.log_cuda_memory,
        "log_cuda_memory_detail": args.log_cuda_memory_detail,
        "cuda_memory_trace_path": str(args.output_dir / "cuda_memory_trace.jsonl"),
        "save_candidate_qc": args.save_candidate_qc,
        "modalities": ["HE", "RNA", "Protein"],
        "dapi_removed_both_sections": True,
        "memory_optimization": {
            "sequential_modality_preprocessing": True,
            "delayed_he_loading": True,
            "lightweight_anndata": True,
            "float32_inputs": True,
            "retain_processed_anndata": False,
            "model_logic_changed": True,
        },
        "preprocessing_cache": cache_info,
        "alignment": audits,
        "feature_shapes": {
            section: {mod: list(value.shape) for mod, value in mods.items()}
            for section, mods in feature_dict.items()
        },
        "embedding_paths": embedding_paths,
        "training_history": history,
        "ot_updates": ot_updates,
    }
    execution_state["summary"] = summary
    _artifacts.save_json(args.output_dir / "loss_history.json", history,
                         transform=json_safe, buffered=True, ensure_ascii=True)
    _artifacts.save_json(args.output_dir / "run_summary.json", summary,
                         transform=json_safe, buffered=True, default=_json_pathlike_default)
    print("SPATCH_SPA_MO_MODEL_TRAIN: PASS")
