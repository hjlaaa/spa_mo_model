#!/usr/bin/env python3
"""Run MouseBrain HE+RNA+Metabolite data through the stage model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import (
    get_default_model_config,
    reject_unsupported_preprocess_config,
)
from data_io.preprocessing import load_cosie_style_data
from data_io.datasets import (
    build_cosie_data_dict,
    build_mousebrain_section,
    restore_section_keys,
    summarize_data_dict,
    summarize_feature_dict,
    summarize_spatial_loc_dict,
)
from data_io.configured_preparation import (
    MouseBrainInput, prepare_dataset,
    subset_section_result as _subset_section_result,
    build_mousebrain_sections as _build_mousebrain_sections,
    preprocess_mousebrain as _preprocess_mousebrain,
)
from model.stage_model import StageMultiModalModel
from model.tensor_utils import tensor_to_numpy
from data_io.common import ensure_dir
from training import artifacts as _artifacts
from training.mousebrain_task import (
    run_mousebrain_training_task, json_safe, collect_alignment_summary,
    summarize_outputs, save_embeddings, save_ot_prior_topk, save_run_artifacts,
)
from training.fit import (
    initialize_model_ot_prior, lambda_for_epoch, sparse_prior_kwargs,
    train_small_crc_model, update_model_ot_prior,
)
from training.entry_defaults import mousebrain_defaults as _entry_defaults
from training.config import (
    load_json, resolve_model_config, resolve_option_values, config_source_layers, parse_dataset_args,
)


from training.entry_defaults import MOUSEBRAIN_MODEL_DEFAULTS as MODEL_DEFAULTS
from training.entry_defaults import MOUSEBRAIN_UOT_HELPER_DEFAULTS as UOT_HELPER_DEFAULTS

from training.entry_defaults import MOUSEBRAIN_TRAINING_DEFAULTS as TRAINING_DEFAULTS


def get_dataset_defaults():
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run MouseBrain training.")
    parser.add_argument("--config", required=True, help="MouseBrain preprocessing/training JSON.", default=None)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, help="Run preprocessing + one forward pass only.", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max_spots_per_section", type=int, default=None, help="Use the first N spots per section.")
    parser.add_argument("--lambda_contrast", type=float, default=None, help="Override loss.lambda_contrast.")
    parser.add_argument(
        "--lambda_contrast_schedule",
        default=None,
        help='Epoch schedule like "1-5:1e-4,6-10:3e-4,11-15:1e-3".',
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Override training device.")
    parser.add_argument("--output_dir", default=None, help="Override config training.output_dir.")
    parser.add_argument("--seed", type=int, default=None, help="Training and OT candidate-search seed.")
    parser.add_argument(
        "--candidate_backend",
        choices=["faiss_ivf", "faiss_flat", "blockwise"],
        default=None,
    )
    parser.add_argument("--initial_modality_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--attention_topk", type=int, default=None)
    parser.add_argument("--faiss_nlist", type=int, default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_device", choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--faiss_train_sample_size", type=int, default=None)
    parser.add_argument("--faiss_query_batch_size", type=int, default=None)
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument(
        "--save_ot_prior_topk",
        action=argparse.BooleanOptionalAction,
        help="Save sparse top-k OT prior tensors for downstream matching QC.",
        default=None,
    )
    parser.add_argument(
        "--ot_prior_output_dir",
        default=None,
        help="Optional directory for sparse top-k OT prior files; defaults to output_dir/ot_prior_topk.",
    )
    return parse_dataset_args(parser, argv, get_dataset_defaults())


def parse_lambda_contrast_schedule(schedule_text: str | None):
    if not schedule_text:
        return None
    schedule = []
    for chunk in schedule_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk or "-" not in chunk.split(":", 1)[0]:
            raise ValueError(
                "Invalid --lambda_contrast_schedule entry. Expected format like "
                "'1-5:1e-4,6-10:3e-4'."
            )
        epoch_range, value_text = chunk.split(":", 1)
        start_text, end_text = epoch_range.split("-", 1)
        start_epoch = int(start_text)
        end_epoch = int(end_text)
        if start_epoch <= 0 or end_epoch < start_epoch:
            raise ValueError(f"Invalid epoch range in schedule entry: {chunk}")
        schedule.append((start_epoch, end_epoch, float(value_text)))
    if not schedule:
        raise ValueError("--lambda_contrast_schedule was provided but no valid entries were parsed.")
    return schedule




def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_section_result(section_result: dict[str, Any], max_spots: int | None):
    """Compatibility delegate to the data_io preparation implementation."""
    return _subset_section_result(section_result, max_spots)




def build_mousebrain_sections(config: Mapping[str, Any], max_spots: int | None):
    """Compatibility delegate to the data_io preparation implementation."""
    return _build_mousebrain_sections(config, max_spots)


def preprocess_mousebrain(config: Mapping[str, Any], max_spots: int | None):
    """Compatibility delegate to the data_io preparation implementation."""
    return _preprocess_mousebrain(config, max_spots)


def build_model_config(
    config: Mapping[str, Any],
    epochs: int | None,
    lambda_contrast: float | None = None,
    device: str | None = None,
    update_interval: int | None = None,
    attention_topk: int | None = None,
    uot_epsilon: float | None = None,
    uot_tau_a: float | None = None,
    uot_tau_b: float | None = None,
    uot_max_iter: int | None = None,
    spatial_knn_k: int | None = None,
    post_ot_graphsage_scale: float | None = None,
):
    # Preserve MouseBrain entry defaults, then merge model and input layers.
    model_config = get_default_model_config()
    for section, values in MODEL_DEFAULTS.items():
        model_config[section].update(values)

    # None means the user did not provide this CLI override; False is explicit.
    overrides = {
        "training": {"epochs": epochs, "device": device},
        "loss": {"lambda_contrast": lambda_contrast},
        "uot": {
            "update_interval": update_interval,
            "topk": attention_topk,
            "epsilon_update": uot_epsilon,
            "tau_a": uot_tau_a,
            "tau_b": uot_tau_b,
            "max_iter": uot_max_iter,
        },
        "graph": {"knn_neighbors_spatial": spatial_knn_k},
        "graphsage": {"post_ot_graphsage_scale": post_ot_graphsage_scale},
    }
    model_config = resolve_model_config(
        base=model_config,
        model_config=config.get("model", {}),
        input_config={"training": config.get("training", {})},
        explicit_overrides=overrides,
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return model_config


def resolve_run_config(config: Mapping[str, Any], args):
    """Resolve the MouseBrain model and helper arguments once, before any I/O work."""
    model_config = build_model_config(
        config,
        epochs=args.epochs,
        lambda_contrast=args.lambda_contrast,
        device=args.device,
        update_interval=args.update_interval,
        attention_topk=args.attention_topk,
        uot_epsilon=args.uot_epsilon,
        uot_tau_a=args.uot_tau_a,
        uot_tau_b=args.uot_tau_b,
        uot_max_iter=args.uot_max_iter,
        spatial_knn_k=args.spatial_knn_k,
        post_ot_graphsage_scale=args.post_ot_graphsage_scale,
    )
    resolved = argparse.Namespace(**vars(args))
    training = model_config["training"]
    uot = model_config["uot"]
    # These helper inputs previously lived only in argparse. Their JSON keys
    # now live beside the other sparse UOT settings, with the old entry defaults.
    uot_defaults = UOT_HELPER_DEFAULTS
    uot_cli = {
        key: getattr(args, "uot_stabilizer" if key == "stabilizer" else key)
        for key in uot_defaults
    }
    for key, value in resolve_option_values(uot, uot_cli, uot_defaults).items():
        setattr(resolved, "uot_stabilizer" if key == "stabilizer" else key, value)
    training_defaults = TRAINING_DEFAULTS
    training_cli = {key: getattr(args, key) for key in training_defaults}
    for key, value in resolve_option_values(training, training_cli, training_defaults).items():
        setattr(resolved, key, value)
    resolved.epochs = int(training["epochs"])
    resolved.device = training["device"]
    resolved.lambda_contrast = float(model_config["loss"]["lambda_contrast"])
    resolved.attention_topk = int(uot["topk"])
    resolved.uot_epsilon = float(uot["epsilon_update"])
    resolved.uot_tau_a = float(uot["tau_a"])
    resolved.uot_tau_b = float(uot["tau_b"])
    resolved.uot_max_iter = int(uot["max_iter"])
    resolved.update_interval = int(uot["update_interval"])
    resolved.spatial_knn_k = int(model_config["graph"]["knn_neighbors_spatial"])
    resolved.post_ot_graphsage_scale = float(model_config["graphsage"]["post_ot_graphsage_scale"])
    return model_config, resolved










def run_mousebrain(args):
    if args.lambda_contrast is not None and args.lambda_contrast_schedule is not None:
        raise ValueError("Use either --lambda_contrast or --lambda_contrast_schedule, not both.")
    config_path = Path(args.config)
    config = load_json(config_path)
    source_layers = config_source_layers(
        default=["model.configure.get_default_model_config", "MouseBrain runner defaults"],
        model_config={"path": str(config_path), "section": "model"} if config.get("model") else None,
        dataset_input={"path": str(config_path)},
        explicit_cli={
            name: value for name, value in vars(args).items()
            if name != "config" and value is not None
            and (name not in {"dry_run", "save_ot_prior_topk"} or value)
        },
    )
    model_config, args = resolve_run_config(config, args)
    if args.update_interval <= 0:
        raise ValueError("--update_interval must be positive.")
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")

    seed_everything(int(args.seed))
    max_spots = args.max_spots_per_section
    output_dir = Path(args.output_dir)
    if max_spots is not None:
        output_dir = output_dir / f"maxspots_{max_spots}"
    if args.dry_run:
        if output_dir.name != "dry_run":
            output_dir = output_dir / "dry_run"
    else:
        epochs_for_dir = int(args.epochs)
        epoch_dir_name = f"epochs_{epochs_for_dir}"
        if output_dir.name != epoch_dir_name:
            output_dir = output_dir / epoch_dir_name

    prepared = prepare_dataset(MouseBrainInput(config, max_spots))
    prep = prepared.compatibility_context["legacy_result"]
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    processed_data_dict = prepared.processed_data_dict
    section_order = prepared.section_order
    lambda_schedule = parse_lambda_contrast_schedule(args.lambda_contrast_schedule)
    epochs = int(model_config["training"]["epochs"])

    resolved_config = {
        "dataset": config.get("dataset_name", "MouseBrain"),
        "section_order": section_order,
        "modalities": config.get("modalities"),
        "input_config": config,
        "source_layers": source_layers,
        "model_config": deepcopy(model_config),
        "runner": vars(args).copy(),
        "output_dir": str(output_dir),
        "refresh_source": "ot",
        "execution": {
            "precision": "fp32",
            "amp_dtype": "none",
            "autocast_enabled": False,
            "grad_scaler_enabled": False,
            "training_loss_only": False,
            "return_full_outputs": True,
            "decoder_chunk_size": None,
            "ot_attention_source_chunk_size": None,
            "cache_spatial_graphs": False,
            "checkpoint_ot_attention": False,
            "checkpoint_encoder_fusion": False,
            "checkpoint_decoder_chunks": False,
            "checkpoint_graph_encoder": False,
        },
    }

    return run_mousebrain_training_task(
        {"summary": resolved_config, "model_config": model_config, "runner": args},
        prepared, lambda_schedule=lambda_schedule,
    )


def main():
    args = parse_args()
    run_mousebrain(args)


if __name__ == "__main__":
    main()
