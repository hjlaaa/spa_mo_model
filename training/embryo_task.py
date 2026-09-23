"""Human Embryo task after the caller's audit and HESTA preparation exits."""
from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import torch
from . import artifacts as _artifacts
from .artifacts import save_final_embeddings, save_ot_prior_topk
from .fit import CudaMemoryMonitor, initialize_model_ot_prior, json_safe, run_one_forward, train_small_crc_model
from .task_runtime import initialize_prepared_task


@dataclass(frozen=True)
class SingleModalityPolicy:
    initial_uot: bool
    dry_eval: bool


def write_json(path: Path, value: Any) -> None:
    _artifacts.save_json(path, value, transform=json_safe, buffered=True)


def run_embryo_training_task(run_config, prepared, *, policy: SingleModalityPolicy, requested_action, data_dir):
    """Borrow resolved config/arrays; no audit, preparation, or CLI resolution."""
    args = Namespace()
    args.__dict__ = run_config["runner"]
    output_dir = Path(run_config["output_dir"])
    preprocess_manifest = prepared.compatibility_context["preprocess_manifest"]
    external_source_summary = prepared.compatibility_context["external_source_summary"]
    feature_dict = prepared.feature_dict
    spatial_dict = prepared.spatial_loc_dict
    section_order = prepared.section_order
    model_config = run_config["model_config"]
    model, _, _ = initialize_prepared_task(model_config, prepared, args, defer_prior=True)
    resolved = list(model._resolve_modality_order(feature_dict[section_order[0]]))
    if resolved != ["RNA"]:
        raise AssertionError(f"Expected the RNA-only branch, got {resolved}.")
    memory_monitor = CudaMemoryMonitor(
        enabled=bool(args.log_cuda_memory), output_dir=output_dir, requested_device=args.device
    )
    start = time.time()
    if policy.initial_uot:
        initialize_model_ot_prior(model, feature_dict, section_order, args)
    else:
        model.ot_prior = {}
    history = None
    ot_updates: list[int] = []
    if args.train:
        history, outputs, ot_updates = train_small_crc_model(
            model, feature_dict, spatial_dict, None, section_order, args, memory_monitor
        )
        _artifacts.save_weights(
            payload={"model_state_dict": model.state_dict(), "model_config": model_config, "section_order": section_order},
            path=output_dir / "model_checkpoint.pt",
        )
        write_json(output_dir / "loss_history.json", history)
    else:
        if policy.dry_eval:
            model.eval()
        with torch.no_grad():
            outputs = run_one_forward(
                model,
                feature_dict,
                spatial_dict,
                None,
                section_order,
                epoch=0,
                decoder_chunk_size=args.decoder_chunk_size,
                ot_attention_source_chunk_size=args.ot_attention_source_chunk_size,
                cache_spatial_graphs=args.cache_spatial_graphs,
            )
    if outputs["mode"] != {
        "single_modality": True,
        "modalities": ["RNA"],
        "contrastive_skipped": True,
        "fusion": "identity",
    }:
        raise AssertionError(f"Unexpected model mode metadata: {outputs['mode']}")
    crossview = float(outputs["losses"]["crossview_loss"].detach().cpu())
    if crossview != 0.0:
        raise AssertionError(f"RNA-only cross-view loss must be zero, got {crossview}.")
    embedding_paths = save_final_embeddings(output_dir, outputs["final_embeddings"])
    ot_paths = save_ot_prior_topk(
        output_dir / "ot_prior_topk",
        outputs.get("ot_prior"),
        outputs["final_embeddings"],
        requested_action,
        save_candidate_qc=args.save_candidate_qc,
    )
    summary = {
        "resolved_config": run_config,
        "status": "PASS",
        "mode": requested_action,
        "input_data_path": str(data_dir),
        "output_dir": str(output_dir),
        "model_mode": outputs["mode"],
        "modalities": ["RNA"],
        "single_modality_switch": model_config["model"]["single_modality_mode"],
        "section_order": section_order,
        "feature_shapes": {
            section: list(feature_dict[section]["RNA"].shape) for section in section_order
        },
        "final_embedding_shapes": {
            section: list(outputs["final_embeddings"][section].shape) for section in section_order
        },
        "losses": {name: float(value.detach().cpu()) for name, value in outputs["losses"].items()},
        "crossview_loss_skipped": crossview == 0.0,
        "fusion_mode": "identity",
        "reconstruction_modalities": {
            section: sorted(outputs["reconstructions"][section]) for section in section_order
        },
        "harmony_used": bool(preprocess_manifest.get("harmony_used")),
        "full_spot": args.max_spots_per_section is None,
        "per_spot_output": True,
        "preprocessing": {
            "harmony_used": bool(preprocess_manifest.get("harmony_used")),
            "harmony_batch_key": preprocess_manifest.get("harmony_batch_key"),
            "harmony_implementation": preprocess_manifest.get("harmony_implementation"),
            "external_preprocessed_run": str(Path(args.preprocessed_run_dir).resolve())
            if args.preprocessed_run_dir is not None else None,
            "max_spots_per_section": args.max_spots_per_section,
        },
        "rna_source": "layers/counts",
        "ot_prior_mode": "disabled" if args.disable_uot else "candidate_sparse",
        "bidirectional_ot_attention": not args.disable_uot,
        "dynamic_candidate_source": "ot",
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "ot_updates": ot_updates,
        "elapsed_time_sec": float(time.time() - start),
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3)
        if args.device == "cuda" else None,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3)
        if args.device == "cuda" else None,
        "preprocess_manifest": preprocess_manifest,
        "external_preprocessing_summary": str(
            Path(args.preprocessed_run_dir).resolve() / "run_summary.json"
        ) if external_source_summary is not None else None,
        "saved_files": {
            "embeddings": embedding_paths,
            "final_embeddings": embedding_paths,
            "ot_prior_topk": ot_paths,
            "checkpoint": str(output_dir / "model_checkpoint.pt") if args.train else None,
            "loss_history": str(output_dir / "loss_history.json") if history is not None else None,
        },
        "confirmations": {
            "original_h5ad_modified": False,
            "fabricated_missing_modality": False,
            "contrastive_loss_computed": False,
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary

