#!/usr/bin/env python3
"""Train spa_mo_model on full-resolution spatch HE+RNA+Protein data."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_io.spatch_preparation import validate_source_stats as _legacy_validate_source_stats
from training.entry_defaults import spatch_defaults as _entry_defaults
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from data_io.spatch_preparation import (
    SECTIONS, FILES, CACHE_SCHEMA_VERSION, CACHE_METADATA_FILES,
    SpatchCacheInput, SpatchRawInput, prepare_dataset,
    canonical_ids as _cache_canonical_ids,
    sha256_file as _cache_sha256_file,
    cache_parameters as _cache_parameters,
    validate_cache_input_identity as _cache_validate_input_identity,
    load_preprocessed_cache as _cache_load_preprocessed_cache,
)
from model.stage_model import StageMultiModalModel
from training.fit import (
    CudaMemoryMonitor,
    initialize_model_ot_prior,
    json_safe,
    train_small_crc_model,
)
from training.spatch_task import run_spatch_training_task, ModalityOrderPolicy, _json_pathlike_default
from training import artifacts as _artifacts
from training.artifacts import (
    save_final_embeddings,
)


from training.entry_defaults import SPATCH_TRAINING_DEFAULTS as TRAINING_DEFAULTS


def get_dataset_defaults():
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spa_mo_model on full-resolution spatch.")
    parser.add_argument("--interaction_neighbor_weight", type=float, default=None,
                        help="Target spatial-neighbor fraction in attention values (default: 0; experiment: 0.25).")
    parser.add_argument("--input_mode", choices=["reuse", "raw"], default=None,
                        help="reuse validates an existing cache; raw prepares six aligned H5AD inputs.")
    parser.add_argument("--output_cache_dir", type=Path, default=None,
                        help="raw only: new cache directory; defaults to output_dir/preparation_cache. Never overwritten.")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Raw mode: six aligned H5AD root. Reuse: optional recorded source selection; raw files are not read.",
    )
    parser.add_argument(
        "--input_identity_manifest", type=Path, default=None,
        help="Optional trusted source_files JSON for an explicitly requested raw input; raw files are not read.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lambda_spatial_gaussian", type=float, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--candidate_backend", default=None)
    parser.add_argument("--initial_modality_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--attention_topk", type=int, default=None)
    parser.add_argument("--faiss_nlist", type=int, default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_device", default=None)
    parser.add_argument("--faiss_train_sample_size", type=int, default=None)
    parser.add_argument("--faiss_query_batch_size", type=int, default=None)
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=None)
    parser.add_argument("--decoder_chunk_size", type=int, default=None)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=None)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "none"], default=None)
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument(
        "--preprocessed_cache_dir",
        type=Path,
        default=None,
        help="Shared model-ready PCA/Harmony feature cache directory.",
    )
    parser.add_argument(
        "--build_preprocessed_cache",
        action=argparse.BooleanOptionalAction,
        help="Unsupported legacy switch; use explicit --input_mode raw to prepare new inputs.",
        default=None,
    )
    parser.add_argument("--spatial_enhancement", action=argparse.BooleanOptionalAction,
                        default=False, help="Raw mode: add spatial KNN sums to PCA/Harmony features.")
    parser.add_argument("--spatial_enhancement_k", type=int, default=10)
    parser.add_argument("--spatial_enhancement_weight", type=float, default=0.2)
    parser.add_argument("--spatial_enhancement_include_self",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parse_dataset_args(parser, argv, get_dataset_defaults())


def require_gpu(device: str) -> dict[str, object]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("spa_mo_model spatch is GPU-only and CUDA is unavailable.")
    probe = torch.ones(1, device="cuda")
    del probe
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def canonical_ids(section: str, coords: np.ndarray) -> pd.Index:
    """Compatibility delegate to the authoritative SPATCH cache adapter."""
    return _cache_canonical_ids(section, coords)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compatibility delegate to the authoritative SPATCH cache adapter."""
    return _cache_sha256_file(path, chunk_size)


def cache_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Compatibility delegate to the authoritative SPATCH cache adapter."""
    return _cache_parameters(args)


def validate_source_stats(
    manifest: dict[str, object],
    paths: dict[str, dict[str, Path]],
) -> None:
    """Legacy API only; not used by the strict raw/reuse cache path."""
    return _legacy_validate_source_stats(manifest, paths)


def validate_cache_input_identity(
    manifest: dict[str, object],
    data_dir: Path | None,
    identity_manifest: Path | None,
) -> dict[str, object]:
    """Compatibility delegate to the authoritative SPATCH cache adapter."""
    return _cache_validate_input_identity(manifest, data_dir, identity_manifest)


def load_preprocessed_cache(
    cache_dir: Path,
    data_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, object],
]:
    """Compatibility delegate to the authoritative SPATCH cache adapter."""
    return _cache_load_preprocessed_cache(cache_dir, data_dir, output_dir, args)


def process_memory_gib() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            values[key] = float(value.strip().split()[0]) / 1024**2
    return {
        "rss_gib": values.get("VmRSS", float("nan")),
        "peak_rss_gib": values.get("VmHWM", float("nan")),
    }


def record_memory(output_dir: Path, stage: str) -> None:
    payload = {"time": time.time(), "stage": stage, **process_memory_gib()}
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "cuda_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            }
        )
    with open(output_dir / "memory_timeline.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(f"SPATCH_MEMORY {json.dumps(payload, sort_keys=True)}", flush=True)


def build_model_config(args):
    config = resolve_model_config(input_config={
        "training": {
            "device": "cuda", "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "loss": {"lambda_contrast": args.lambda_contrast,
                 "lambda_spatial_gaussian": args.lambda_spatial_gaussian or 0.0},
        "uot": {
            "max_iter": args.uot_max_iter, "topk": args.attention_topk,
            "update_interval": args.update_interval,
            "epsilon_update": args.uot_epsilon,
            "tau_a": args.uot_tau_a, "tau_b": args.uot_tau_b,
        },
        "graph": {"knn_neighbors_spatial": args.spatial_knn_k},
        "graphsage": {
            "edge_batch_size": args.graphsage_edge_batch_size,
            "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        },
    }, explicit_overrides={
        "ot_attention": {"interaction_neighbor_weight": getattr(args, "interaction_neighbor_weight", None)},
    })
    return config


def resolve_run_config(args):
    # These are fixed SPATCH execution settings, not cache identity fields.
    for name, value in TRAINING_DEFAULTS.items():
        setattr(args, name, value)
    return describe_run_config(
        args, build_model_config(args), dataset="spatch",
        section_order=list(SECTIONS), modalities=["HE", "RNA", "Protein"],
        preprocessing=cache_parameters(args),
    )




def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")
    if args.build_preprocessed_cache:
        raise ValueError("SPATCH training only loads an existing cache; --build_preprocessed_cache is disabled.")
    # Keep existing reuse resolved-config/summary fields unchanged. New input
    # selection stays local; raw source details travel in cache_info/provenance.
    input_mode, output_cache_dir = args.input_mode, args.output_cache_dir
    del args.input_mode, args.output_cache_dir
    if input_mode == "reuse":
        if output_cache_dir is not None:
            raise ValueError("--output_cache_dir is only valid with --input_mode raw.")
        if args.preprocessed_cache_dir is None:
            raise ValueError("SPATCH training requires an existing --preprocessed_cache_dir; preprocessing is disabled.")
        args.preprocessed_cache_dir = args.preprocessed_cache_dir.resolve()
        if not (args.preprocessed_cache_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing SPATCH preprocessing cache manifest: {args.preprocessed_cache_dir / 'manifest.json'}")
        if args.output_dir == args.preprocessed_cache_dir or args.preprocessed_cache_dir in args.output_dir.parents:
            raise ValueError("SPATCH cache is read-only; output_dir must be outside the cache.")
    else:
        if args.data_dir is None:
            raise ValueError("--input_mode raw requires --data_dir with six aligned H5AD inputs.")
        if args.preprocessed_cache_dir is not None or args.input_identity_manifest is not None:
            raise ValueError("raw mode does not accept --preprocessed_cache_dir or --input_identity_manifest.")
        from data_io.spatch_raw import validate_raw_destination
        args.preprocessed_cache_dir = validate_raw_destination(
            output_cache_dir if output_cache_dir is not None else args.output_dir / "preparation_cache",
            args.output_dir,
        )
    run_config = resolve_run_config(args)
    if (args.output_dir / "run_summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"Existing result: {args.output_dir}; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale in ("run_failure.json", "memory_timeline.jsonl"):
            stale_path = args.output_dir / stale
            if stale_path.exists():
                stale_path.unlink()
    try:
        gpu = require_gpu(args.device)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if input_mode == "reuse":
            record_memory(args.output_dir, "preprocessed_cache_load_start")
            prepared = prepare_dataset(SpatchCacheInput(
                args.preprocessed_cache_dir, args.data_dir, args.output_dir, args
            ))
        else:
            record_memory(args.output_dir, "raw_preparation_start")
            prepared = prepare_dataset(SpatchRawInput(
                args.data_dir, args.output_dir, args.preprocessed_cache_dir, args, record_memory
            ))
        feature_dict = prepared.feature_dict
        spatial_dict = prepared.spatial_loc_dict
        audits = prepared.preparation_audit
        cache_info = prepared.compatibility_context["cache_info"]
        processed = prepared.processed_data_dict
        record_memory(args.output_dir, "preprocessed_cache_load_end" if input_mode == "reuse" else "raw_preparation_end")
        section_order = prepared.section_order
        execution_state = {}
        run_spatch_training_task(
            run_config, prepared,
            policy=ModalityOrderPolicy(section="section1", expected_modalities=("HE", "RNA", "Protein")),
            gpu=gpu, execution_state=execution_state,
        )
    except Exception as exc:
        failure = {"dataset": "spatch", "method": "spa_mo_model", "status": "failed", "error": repr(exc)}
        _artifacts.save_json(args.output_dir / "run_failure.json", failure, buffered=True)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
