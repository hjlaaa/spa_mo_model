"""Generic yield-driven training task; input resolution remains with the caller.

The fit iterator owns all epochs, optimizer steps, refreshes and final eval.
This task preserves the separate periodic and final artifact policies.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from model.stage_model import StageMultiModalModel
from data_io.common import ensure_dir
from . import artifacts as _artifacts
from .fit import iter_fit_model
from .task_runtime import initialize_prepared_task


@dataclass(frozen=True)
class YieldExecutionPolicy:
    clear_step_state: bool
    record_elapsed_time: bool
    allow_empty_epochs: bool
    record_loss_weights: bool


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



def summarize_outputs(outputs):
    return {
        "fused_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["fused_embeddings"].items()
        },
        "graphsage_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["graphsage_embeddings"].items()
        },
        "final_embeddings": {
            section: list(tensor.shape)
            for section, tensor in outputs["final_embeddings"].items()
        },
        "reconstructions": {
            section: {
                modality: list(tensor.shape)
                for modality, tensor in modalities.items()
            }
            for section, modalities in outputs["reconstructions"].items()
        },
        "ot_prior_keys": [list(key) for key in (outputs["ot_prior"] or {}).keys()],
    }



def save_training_artifacts(
    output_dir: Path,
    model: StageMultiModalModel,
    config: dict[str, Any],
    history: list[dict[str, float]],
    outputs,
    section_order,
    save_embeddings: bool,
):
    ensure_dir(output_dir)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "section_order": section_order,
        "input_dims": model.input_dims,
    }
    _artifacts.save_weights(output_dir / "stage_model_v2_last.pt", checkpoint)

    _artifacts.save_json(output_dir / "training_history.json", history)

    # Keep summary projection after file-open, including its failure footprint.
    _artifacts.save_json(
        output_dir / "training_summary.json", outputs,
        transform=lambda current: {
            "section_order": section_order,
            "output_shapes": summarize_outputs(current),
            "config": json_safe(config),
        },
    )

    if save_embeddings:
        embedding_dir = output_dir / "final_embeddings"
        ensure_dir(embedding_dir)
        _artifacts.save_final_embeddings(
            embedding_dir, outputs["final_embeddings"],
            filename_template="{section}_final_embedding.npy",
        )



def run_generic_training_task(config, prepared, *, args, section_order, policy: YieldExecutionPolicy):
    """Consume resolved config and borrowed prepared arrays; no input I/O or resolve."""
    inputs = prepared.compatibility_context["legacy_inputs"]
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    model, _, _ = initialize_prepared_task(config, prepared, args, defer_prior=True)
    uot_config = model.config["uot"]
    # Preserve configured OT scalars; retrieval settings are the existing Stage defaults.
    sparse_kwargs = {
        "candidate_k": 200,
        "attention_topk": int(uot_config["topk"]),
        "candidate_backend": getattr(args, "candidate_backend", "faiss_ivf"),
        "faiss_nlist": 4096,
        "faiss_nprobe": 64,
        "faiss_device": "auto",
        "faiss_train_sample_size": 100000,
        "faiss_query_batch_size": 8192,
        "seed": 42,
        "tau_a": float(uot_config["tau_a"]),
        "tau_b": float(uot_config["tau_b"]),
        "max_iter": int(uot_config["max_iter"]),
        "stabilizer": float(model.config["ot_attention"]["delta"]),
    }
    if uot_config["enabled"]:
        model.initialize_candidate_sparse_ot_prior(
            feature_dict,
            section_order=section_order,
            initial_modality_candidate_k=100,
            epsilon=float(uot_config["epsilon_init"]),
            **sparse_kwargs,
        )
    else:
        model.ot_prior = {}

    fit_args = argparse.Namespace(
        epochs=int(config["training"]["epochs"]),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        device=config["training"]["device"],
        amp_dtype="none",
        update_interval=int(uot_config["update_interval"]),
        **{key: value for key, value in sparse_kwargs.items()
           if key not in {"tau_a", "tau_b", "max_iter", "stabilizer"}},
        uot_epsilon=float(uot_config["epsilon_update"]),
        uot_tau_a=sparse_kwargs["tau_a"],
        uot_tau_b=sparse_kwargs["tau_b"],
        uot_max_iter=sparse_kwargs["max_iter"],
        uot_stabilizer=sparse_kwargs["stabilizer"],
        training_loss_only=False,
        decoder_chunk_size=0,
        ot_attention_source_chunk_size=0,
        cache_spatial_graphs=False,
        checkpoint_ot_attention=False,
        checkpoint_encoder_fusion=False,
        checkpoint_decoder_chunks=False,
        checkpoint_graph_encoder=False,
        log_every=args.log_every,
        log_cuda_memory_detail=False,
    )
    for is_final, history, outputs, _ in iter_fit_model(
        model, feature_dict, spatial_loc_dict, None, section_order, fit_args,
        clear_step_state=policy.clear_step_state,
        record_elapsed_time=policy.record_elapsed_time,
        allow_empty_epochs=policy.allow_empty_epochs,
        record_loss_weights=policy.record_loss_weights,
        refresh_ot=bool(uot_config["enabled"]),
        yield_every=args.save_every if args.output_dir else 0,
    ):
        if is_final:
            final_outputs = outputs
        else:
            save_training_artifacts(
                output_dir=Path(args.output_dir),
                model=model,
                config=config,
                history=history,
                outputs=outputs,
                section_order=section_order,
                save_embeddings=False,
            )

    if args.output_dir:
        save_training_artifacts(
            output_dir=Path(args.output_dir),
            model=model,
            config=config,
            history=history,
            outputs=final_outputs,
            section_order=section_order,
            save_embeddings=args.save_embeddings or args.smoke_test,
        )

    print("Training complete.")
    print(json.dumps(summarize_outputs(final_outputs), indent=2, ensure_ascii=False))
    return model, final_outputs, history

