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

from training.entry_defaults import embryo_defaults as _entry_defaults
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from model.stage_model import StageMultiModalModel
from data_io.hesta import (
    audit_hesta_files,
    resolve_input_paths,
)
from data_io.embryo_preparation import (
    ManifestInput, ExternalRunInput, RawHestaInput, prepare_source, prepare_dataset,
    load_external_preprocessed_run as _load_external_preprocessed_run,
)
from training.fit import (
    CudaMemoryMonitor,
    initialize_model_ot_prior,
    json_safe,
    run_one_forward,
    train_small_crc_model,
)
from training import artifacts as _artifacts
from training.embryo_task import run_embryo_training_task, SingleModalityPolicy, write_json
from training.artifacts import (
    save_final_embeddings,
    save_ot_prior_topk,
)


def get_dataset_defaults():
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults()


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
    """Compatibility signature; authoritative loader lives in data_io."""
    return _load_external_preprocessed_run(run_dir, maximum_per_section)


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
        preparation_source = prepare_source(ExternalRunInput(
            Path(args.preprocessed_run_dir).resolve(), args.max_spots_per_section
        ))
        preprocess_manifest = preparation_source.manifest
        external_source_summary = preparation_source.external_source_summary
    elif not args.reuse_preprocessed:
        preparation_source = prepare_source(RawHestaInput(
            paths, cache_dir, run_config["preprocessing"]
        ))
        preprocess_manifest = preparation_source.manifest
    else:
        preparation_source = prepare_source(ManifestInput(cache_dir))
        preprocess_manifest = preparation_source.manifest
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

    prepared = prepare_dataset(preparation_source)
    preprocess_manifest = prepared.compatibility_context["preprocess_manifest"]
    feature_dict = prepared.feature_dict
    spatial_dict = prepared.spatial_loc_dict
    section_order = prepared.section_order
    run_config["section_order"] = section_order
    return run_embryo_training_task(
        run_config, prepared,
        policy=SingleModalityPolicy(initial_uot=not args.disable_uot, dry_eval=True),
        requested_action=requested_action, data_dir=data_dir,
    )


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
