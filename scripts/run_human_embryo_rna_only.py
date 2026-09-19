#!/usr/bin/env python3
"""Train/evaluate SpaMO on the seven HESTA human-embryo RNA sections.

The default action is a read-only audit.  Use ``--preprocess_only`` to build a
memory-bounded feature cache, ``--dry_run`` for one model forward, or ``--train``
for optimization.  Original h5ad files are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from model.stage_model import StageMultiModalModel
from data_io.hesta import (
    audit_hesta_files,
    load_preprocessed_manifest,
    preprocess_hesta_rna,
    resolve_input_paths,
)
from training.fit import (
    CudaMemoryMonitor,
    initialize_model_ot_prior,
    json_safe,
    run_one_forward,
    train_small_crc_model,
)
from scripts.run_crc_stereocite import (
    save_final_embeddings,
    save_ot_prior_topk,
)


def get_dataset_defaults():
    """Current entry defaults; suite arguments remain explicit overrides."""
    return {
        'data_dir': "/home/hujinlan/human_embryo",
        'output_dir': "/home/hujinlan/spa_mo_model/results/human_embryo_rna_only",
        'preprocess_only': False,
        'dry_run': False,
        'train': False,
        'single_modality': "RNA",
        'reuse_preprocessed': False,
        'preprocessed_run_dir': None,
        'require_harmony': False,
        'hvg_num': 3000,
        'hvg_sample_per_section': 5000,
        'svd_fit_per_section': 20000,
        'n_comps': 50,
        'target_sum': 10000.0,
        'min_counts': 10.0,
        'min_genes': 5,
        'max_pct_mt': 30.0,
        'max_spots_per_section': None,
        'source_chunk_rows': 4096,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'device': "cuda",
        'seed': 42,
        'log_every': 1,
        'update_interval': 20,
        'uot_max_iter': 100,
        'uot_epsilon': 0.05,
        'uot_tau_a': 1.0,
        'uot_tau_b': 1.0,
        'uot_stabilizer': 1e-8,
        'disable_uot': False,
        'candidate_backend': "faiss_ivf",
        'initial_modality_candidate_k': 100,
        'candidate_k': 200,
        'attention_topk': 10,
        'faiss_nlist': 4096,
        'faiss_nprobe': 64,
        'faiss_device': "auto",
        'faiss_train_sample_size': 100000,
        'faiss_query_batch_size': 2048,
        'spatial_knn_k': 5,
        'graphsage_edge_batch_size': 200000,
        'decoder_chunk_size': 50000,
        'ot_attention_source_chunk_size': 50000,
        'training_loss_only': True,
        'checkpoint_ot_attention': True,
        'checkpoint_encoder_fusion': True,
        'checkpoint_decoder_chunks': True,
        'checkpoint_graph_encoder': True,
        'cache_spatial_graphs': True,
        'amp_dtype': "bf16",
        'save_candidate_qc': False,
        'log_cuda_memory': False,
        'log_cuda_memory_detail': False,
        'lambda_contrast': 0.0,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HESTA human-embryo dataset through SpaMO's explicit RNA-only mode."
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preprocess_only", action=argparse.BooleanOptionalAction, default=None)
    action.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=None)
    action.add_argument("--train", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--single_modality",
        choices=["RNA", "Protein", "HE", "Metabolite"],
        default=None,
        help="Explicit model branch. This HESTA adapter supplies RNA and therefore requires RNA.",
    )
    parser.add_argument("--reuse_preprocessed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--preprocessed_run_dir",
        default=None,
        help="Read a preprocess_manifest from another successful run instead of output_dir/preprocessed.",
    )
    parser.add_argument(
        "--require_harmony",
        action=argparse.BooleanOptionalAction,
        help="Reject preprocessing caches that do not explicitly record Harmony correction.",
        default=None,
    )
    parser.add_argument("--hvg_num", type=int, default=None)
    parser.add_argument("--hvg_sample_per_section", type=int, default=None)
    parser.add_argument("--svd_fit_per_section", type=int, default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--target_sum", type=float, default=None)
    parser.add_argument("--min_counts", type=float, default=None)
    parser.add_argument("--min_genes", type=int, default=None)
    parser.add_argument("--max_pct_mt", type=float, default=None)
    parser.add_argument("--max_spots_per_section", type=int, default=None)
    parser.add_argument("--source_chunk_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--disable_uot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default=None)
    parser.add_argument("--initial_modality_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--attention_topk", type=int, default=None)
    parser.add_argument("--faiss_nlist", type=int, default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_device", choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--faiss_train_sample_size", type=int, default=None)
    parser.add_argument("--faiss_query_batch_size", type=int, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    # Optional model override; absence preserves the existing parsed/default schema.
    parser.add_argument("--post_ot_graphsage_scale", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=None)
    parser.add_argument("--decoder_chunk_size", type=int, default=None)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=None)
    parser.add_argument("--training_loss_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_ot_attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_encoder_fusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_decoder_chunks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_graph_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache_spatial_graphs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--save_candidate_qc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory_detail", action=argparse.BooleanOptionalAction, default=None)
    return parse_dataset_args(parser, argv, get_dataset_defaults())


def validate_args(args: argparse.Namespace) -> None:
    scale = getattr(args, "post_ot_graphsage_scale", None)
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")
    if args.single_modality != "RNA":
        raise ValueError(
            "These HESTA files contain RNA only. Use --single_modality RNA; Protein-only and "
            "HE-only are supported by the model switch but require their corresponding features."
        )
    if args.train and args.epochs <= 0:
        raise ValueError("--epochs must be positive with --train.")
    if args.device == "cuda" and (args.train or args.dry_run) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if args.device == "cuda" and (args.train or args.dry_run):
        probe = torch.ones((64, 64), device="cuda")
        probe = probe @ probe
        torch.cuda.synchronize()
        del probe
    if args.amp_dtype != "none" and args.device != "cuda" and (args.train or args.dry_run):
        raise ValueError("AMP requires --device cuda.")
    positive = (
        "hvg_num", "hvg_sample_per_section", "svd_fit_per_section", "n_comps",
        "source_chunk_rows", "spatial_knn_k", "graphsage_edge_batch_size",
        "initial_modality_candidate_k", "candidate_k", "attention_topk", "uot_max_iter",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.log_cuda_memory_detail:
        args.log_cuda_memory = True


def load_external_preprocessed_run(
    run_dir: Path, maximum_per_section: int | None
) -> tuple[dict[str, Any], dict, dict, dict[str, Any]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"External preprocessing summary is missing: {summary_path}")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "PASS":
        raise ValueError("External preprocessing run is not successful.")
    manifest = source_summary["preprocess_manifest"]
    feature_dict = {}
    spatial_dict = {}
    for section in manifest["section_order"]:
        features = np.load(manifest["feature_files"][section], mmap_mode="c")
        spatial = np.load(manifest["spatial_files"][section], mmap_mode="r")
        if len(features) != len(spatial):
            raise ValueError(f"{section}: feature/spatial counts differ in external cache.")
        if maximum_per_section is not None:
            if maximum_per_section <= 0 or maximum_per_section > len(features):
                raise ValueError(
                    f"Invalid --max_spots_per_section={maximum_per_section} for {section}."
                )
            features = np.array(features[:maximum_per_section], dtype=np.float32, copy=True)
            spatial = np.array(spatial[:maximum_per_section], copy=True)
        feature_dict[section] = {"RNA": features}
        spatial_dict[section] = spatial
    return manifest, feature_dict, spatial_dict, source_summary


def build_model_config(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_model_config()
    config["model"]["single_modality_mode"] = {
        "enabled": True,
        "modality": args.single_modality,
    }
    config["training"].update(
        device=args.device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    config["loss"]["lambda_contrast"] = 0.0
    config["uot"]["enabled"] = not bool(args.disable_uot)
    config["ot_attention"]["enabled"] = not bool(args.disable_uot)
    config["uot"].update(
        max_iter=int(args.uot_max_iter),
        topk=int(args.attention_topk),
        epsilon_update=float(args.uot_epsilon),
        tau_a=float(args.uot_tau_a),
        tau_b=float(args.uot_tau_b),
        update_interval=int(args.update_interval),
    )
    config["graph"]["knn_neighbors_spatial"] = int(args.spatial_knn_k)
    config["graphsage"]["edge_batch_size"] = int(args.graphsage_edge_batch_size)
    scale = getattr(args, "post_ot_graphsage_scale", None)
    if scale is not None:
        config["graphsage"]["post_ot_graphsage_scale"] = float(scale)
    return config


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_run_config(args):
    return describe_run_config(
        args, build_model_config(args), dataset="Human Embryo",
        section_order=None, modalities=["RNA"],
        preprocessing={
            "hvg_num": args.hvg_num,
            "hvg_sample_per_section": args.hvg_sample_per_section,
            "svd_fit_per_section": args.svd_fit_per_section,
            "n_comps": args.n_comps,
            "target_sum": args.target_sum,
            "min_counts": args.min_counts,
            "min_genes": args.min_genes,
            "max_pct_mt": args.max_pct_mt,
            "max_spots_per_section": args.max_spots_per_section,
            "source_chunk_rows": args.source_chunk_rows,
            "seed": args.seed,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    run_config = resolve_run_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_input_paths(data_dir)
    audit = audit_hesta_files(paths)
    write_json(output_dir / "input_audit.json", audit)
    requested_action = (
        "train" if args.train else "dry_run" if args.dry_run else "preprocess_only" if args.preprocess_only else "audit_only"
    )
    if requested_action == "audit_only":
        summary = {
            "status": "PASS",
            "mode": "audit_only",
            "input_audit": audit,
            "next_step": "Use --preprocess_only, --dry_run, or --train explicitly.",
        }
        write_json(output_dir / "run_summary.json", summary)
        return summary

    cache_dir = output_dir / "preprocessed"
    external_source_summary = None
    if args.preprocessed_run_dir is not None:
        if not args.reuse_preprocessed:
            raise ValueError("--preprocessed_run_dir requires --reuse_preprocessed.")
        preprocess_manifest, feature_dict, spatial_dict, external_source_summary = (
            load_external_preprocessed_run(
                Path(args.preprocessed_run_dir).resolve(), args.max_spots_per_section
            )
        )
    elif not args.reuse_preprocessed:
        preprocess_manifest = preprocess_hesta_rna(
            paths,
            cache_dir,
            **run_config["preprocessing"],
        )
    else:
        preprocess_manifest, _, _ = load_preprocessed_manifest(cache_dir)
    if args.require_harmony and not preprocess_manifest.get("harmony_used"):
        raise ValueError("Mandatory Harmony correction is missing from preprocessing metadata.")
    if requested_action == "preprocess_only":
        summary = {
            "status": "PASS",
            "mode": requested_action,
            "single_modality": args.single_modality,
            "preprocess_manifest": preprocess_manifest,
        }
        write_json(output_dir / "run_summary.json", summary)
        return summary

    if args.preprocessed_run_dir is None:
        preprocess_manifest, feature_dict, spatial_dict = load_preprocessed_manifest(cache_dir)
    section_order = list(preprocess_manifest["section_order"])
    run_config["section_order"] = section_order
    model_config = run_config["model_config"]
    model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
    resolved = list(model._resolve_modality_order(feature_dict[section_order[0]]))
    if resolved != ["RNA"]:
        raise AssertionError(f"Expected the RNA-only branch, got {resolved}.")
    memory_monitor = CudaMemoryMonitor(
        enabled=bool(args.log_cuda_memory), output_dir=output_dir, requested_device=args.device
    )
    start = time.time()
    if not args.disable_uot:
        initialize_model_ot_prior(model, feature_dict, section_order, args)
    else:
        model.ot_prior = {}
    history = None
    ot_updates: list[int] = []
    if args.train:
        history, outputs, ot_updates = train_small_crc_model(
            model, feature_dict, spatial_dict, None, section_order, args, memory_monitor
        )
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model_config, "section_order": section_order},
            output_dir / "model_checkpoint.pt",
        )
        write_json(output_dir / "loss_history.json", history)
    else:
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


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
        print("HUMAN_EMBRYO_RNA_ONLY: PASS")
    except Exception as exc:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "device_requested": args.device,
            "cpu_fallback_used": False,
        }
        write_json(output_dir / "failure.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
