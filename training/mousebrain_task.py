"""MouseBrain task layout with its unconditional train-mode epoch0 preflight."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from data_io.common import ensure_dir
from data_io.datasets import summarize_data_dict, summarize_feature_dict, summarize_spatial_loc_dict
from . import artifacts as _artifacts
from .fit import train_small_crc_model
from .task_runtime import initialize_prepared_task, run_epoch0_preflight, Epoch0Policy

def json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def collect_alignment_summary(section_results):
    summary = {}
    for result in section_results:
        section_id = result["section_id"]
        modalities = result["modalities"]
        rna = modalities["RNA"]
        meta = modalities["Metabolite"]
        he = modalities["HE"]
        summary[section_id] = {
            "rna_spots": int(rna.n_obs),
            "metabolite_spots": int(meta.n_obs),
            "he_spots": int(he.n_obs),
            "obs_names_match": list(rna.obs_names) == list(meta.obs_names) == list(he.obs_names),
            "spatial_match": bool(
                np.array_equal(rna.obsm["spatial"], meta.obsm["spatial"])
                and np.array_equal(rna.obsm["spatial"], he.obsm["spatial"])
            ),
            "uni_feature_shape": list(he.X.shape),
            "metabolite_feature_shape": list(meta.X.shape),
            "rna_feature_shape": list(rna.X.shape),
        }
    return summary


def summarize_outputs(outputs):
    return {
        "final_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["final_embeddings"].items()
        },
        "fused_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["fused_embeddings"].items()
        },
        "graphsage_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["graphsage_embeddings"].items()
        },
        "reconstructions": {
            section: {
                modality: list(tensor.shape)
                for modality, tensor in modalities.items()
            }
            for section, modalities in outputs["reconstructions"].items()
        },
        "ot_prior_keys": [list(key) for key in (outputs["ot_prior"] or {}).keys()],
        "losses": {
            key: float(value.detach().cpu())
            for key, value in outputs["losses"].items()
        },
        "messages": outputs.get("messages", []),
    }


def save_embeddings(output_dir: Path, final_embeddings: Mapping[str, torch.Tensor]):
    """Preserve the MouseBrain layout while delegating tensor/array writing."""
    embedding_dir = output_dir / "final_embeddings"
    ensure_dir(embedding_dir)
    return _artifacts.save_final_embeddings(
        embedding_dir, final_embeddings, filename_template="{section}_final_embedding.npy",
    )


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
):
    """Delegate writing with MouseBrain's existing metadata and error policies."""
    return _artifacts.save_ot_prior_topk(
        output_dir, ot_prior, final_embeddings, run_mode,
        skip_incomplete=False, direction_meaning=None,
        note="Saved sparse top-k UOT prior from model.ot_prior after final evaluation.",
        metadata_transform=json_safe, prepare_directory=ensure_dir,
    )


def save_run_artifacts(
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: list[dict[str, float]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
):
    ensure_dir(output_dir)
    config_copy = output_dir / "mousebrain_config_used.json"
    shutil.copyfile(config_path, config_copy)
    embedding_paths = save_embeddings(output_dir, final_embeddings)
    full_summary = dict(summary)
    full_summary["config_copy"] = str(config_copy)
    full_summary["final_embedding_paths"] = embedding_paths
    _artifacts.save_json(output_dir / "run_summary.json", full_summary, transform=json_safe)
    if history is not None:
        _artifacts.save_json(output_dir / "loss_history.json", history)


def run_mousebrain_training_task(run_config, prepared, *, lambda_schedule):
    """Consume resolved object references; preserve the separate summary snapshot."""
    args = run_config["runner"]
    model_config = run_config["model_config"]
    resolved_config = run_config["summary"]
    config = resolved_config["input_config"]
    config_path = Path(args.config)
    output_dir = Path(resolved_config["output_dir"])
    max_spots = args.max_spots_per_section
    epochs = int(model_config["training"]["epochs"])
    prep = prepared.compatibility_context["legacy_result"]
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    processed_data_dict = prepared.processed_data_dict
    section_order = prepared.section_order

    model, initial_prior, _ = initialize_prepared_task(model_config, prepared, args)
    dry_outputs = run_epoch0_preflight(
        model, prepared, policy=Epoch0Policy(eval_mode=False),
    )

    first_parameter = next(model.parameters())
    resolved_config["execution"]["device"] = str(first_parameter.device)
    resolved_config["execution"]["parameter_dtype"] = str(first_parameter.dtype)

    preprocessing_summary = {
        "dataset_name": config.get("dataset_name", "MouseBrain"),
        "section_order": section_order,
        "max_spots_per_section": max_spots,
        "use_harmony": config.get("preprocessing", {}).get("use_harmony", True),
        "hvg_num_by_modality": config.get("preprocessing", {}).get("hvg_num_by_modality"),
        "lambda_contrast": model_config["loss"]["lambda_contrast"],
        "lambda_contrast_schedule": args.lambda_contrast_schedule,
        "device": model_config["training"]["device"],
        "seed": int(args.seed),
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "candidate_backend": str(args.candidate_backend),
        "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "dynamic_candidate_source": "ot",
        "uot_epsilon": float(args.uot_epsilon),
        "uot_tau_a": float(args.uot_tau_a),
        "uot_tau_b": float(args.uot_tau_b),
        "uot_max_iter": int(args.uot_max_iter),
        "update_interval": int(args.update_interval),
        "spatial_knn_k": int(args.spatial_knn_k),
        "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "topology_aware_refresh_enabled": bool(
            model_config["uot"].get("topology_aware_refresh_enabled", False)
        ),
        "topology_context_weight": float(
            model_config["uot"].get("topology_context_weight", 0.0)
        ),
        "initial_ot_prior_metadata": {
            f"{left}_to_{right}": json_safe(prior.get("metadata", {}))
            for (left, right), prior in initial_prior.items()
        },
        "data_dict": summarize_data_dict(prep["data_dict"]),
        "feature_dict": summarize_feature_dict(feature_dict),
        "spatial_loc_dict": summarize_spatial_loc_dict(spatial_loc_dict),
        "processed_data_dict_generated": processed_data_dict is not None,
        "alignment": collect_alignment_summary(prep["section_results"]),
        "adapter_messages": prep["messages"],
    }

    if args.dry_run:
        ot_prior_topk_files = {}
        ot_prior_topk_dir = None
        if args.save_ot_prior_topk:
            ot_prior_topk_dir = Path(args.ot_prior_output_dir) if args.ot_prior_output_dir else output_dir / "ot_prior_topk"
            ot_prior_topk_files = save_ot_prior_topk(
                output_dir=ot_prior_topk_dir,
                ot_prior=dry_outputs["ot_prior"],
                final_embeddings=dry_outputs["final_embeddings"],
                run_mode="dry_run",
            )
        summary = {
            "mode": "dry_run",
            "resolved_config": resolved_config,
            "preprocessing": preprocessing_summary,
            "forward": summarize_outputs(dry_outputs),
            "loss_finite": bool(torch.isfinite(dry_outputs["losses"]["total_loss"]).item()),
            "saved_ot_prior_topk": bool(args.save_ot_prior_topk),
        }
        if args.save_ot_prior_topk:
            summary["ot_prior_topk_dir"] = str(ot_prior_topk_dir)
            summary["ot_prior_topk_files"] = ot_prior_topk_files
        save_run_artifacts(
            output_dir=output_dir,
            config_path=config_path,
            config=config,
            summary=summary,
            history=None,
            final_embeddings=dry_outputs["final_embeddings"],
        )
        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
        print("MOUSEBRAIN_DRY_RUN: PASS")
        return summary

    execution = resolved_config["execution"]
    fit_args = argparse.Namespace(
        **vars(args),
        lr=float(model_config["training"]["lr"]),
        weight_decay=float(model_config["training"]["weight_decay"]),
        amp_dtype=execution["amp_dtype"],
        training_loss_only=execution["training_loss_only"],
        decoder_chunk_size=execution["decoder_chunk_size"] or 0,
        ot_attention_source_chunk_size=execution["ot_attention_source_chunk_size"] or 0,
        cache_spatial_graphs=execution["cache_spatial_graphs"],
        checkpoint_ot_attention=execution["checkpoint_ot_attention"],
        checkpoint_encoder_fusion=execution["checkpoint_encoder_fusion"],
        checkpoint_decoder_chunks=execution["checkpoint_decoder_chunks"],
        checkpoint_graph_encoder=execution["checkpoint_graph_encoder"],
        log_every=1,
        log_cuda_memory_detail=False,
    )
    history, final_outputs, ot_updates = train_small_crc_model(
        model, feature_dict, spatial_loc_dict, processed_data_dict, section_order,
        fit_args,
        lambda_contrast_schedule=lambda_schedule,
        clear_step_state=False,
        record_elapsed_time=False,
        allow_empty_epochs=True,
    )

    ot_prior_topk_files = {}
    ot_prior_topk_dir = None
    if args.save_ot_prior_topk:
        ot_prior_topk_dir = Path(args.ot_prior_output_dir) if args.ot_prior_output_dir else output_dir / "ot_prior_topk"
        ot_prior_topk_files = save_ot_prior_topk(
            output_dir=ot_prior_topk_dir,
            ot_prior=final_outputs["ot_prior"],
            final_embeddings=final_outputs["final_embeddings"],
            run_mode="training_final_eval",
        )

    summary = {
        "mode": "train",
        "resolved_config": resolved_config,
        "epochs": epochs,
        "seed": int(args.seed),
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "candidate_backend": str(args.candidate_backend),
        "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "dynamic_candidate_source": "ot",
        "uot_epsilon": float(args.uot_epsilon),
        "uot_tau_a": float(args.uot_tau_a),
        "uot_tau_b": float(args.uot_tau_b),
        "uot_max_iter": int(args.uot_max_iter),
        "update_interval": int(args.update_interval),
        "spatial_knn_k": int(args.spatial_knn_k),
        "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "topology_aware_refresh_enabled": bool(
            model_config["uot"].get("topology_aware_refresh_enabled", False)
        ),
        "topology_context_weight": float(
            model_config["uot"].get("topology_context_weight", 0.0)
        ),
        "ot_updates": ot_updates,
        "preprocessing": preprocessing_summary,
        "forward": summarize_outputs(final_outputs),
        "loss_finite": bool(torch.isfinite(final_outputs["losses"]["total_loss"]).item()),
        "saved_ot_prior_topk": bool(args.save_ot_prior_topk),
    }
    if args.save_ot_prior_topk:
        summary["ot_prior_topk_dir"] = str(ot_prior_topk_dir)
        summary["ot_prior_topk_files"] = ot_prior_topk_files
    save_run_artifacts(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        summary=summary,
        history=history,
        final_embeddings=final_outputs["final_embeddings"],
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
    print("MOUSEBRAIN_TRAINING: PASS")
    return summary

