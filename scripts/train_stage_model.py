#!/usr/bin/env python3
"""Train the V2 StageMultiModalModel from COSIE-style preprocessing outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.entry_defaults import GENERIC_CLI_DEFAULTS, GENERIC_SMOKE_DEFAULTS
from training.config import load_json, resolve_model_config
from data_io.datasets import preprocess_multisection_cosie_style
from data_io.configured_preparation import (
    BundleInput, RawConfigInput, SyntheticInput, prepare_dataset,
    to_float_tensors as _to_float_tensors,
    build_synthetic_training_inputs as _build_synthetic_training_inputs,
)
from training.generic_task import (
    YieldExecutionPolicy, run_generic_training_task,
    # Existing layout/helper imports remain compatible without duplicate bodies.
    json_safe, summarize_outputs, save_training_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train StageMultiModalModel V2 on preprocessed multimodal features.",
    )
    parser.add_argument(
        "--input_bundle",
        default=None,
        help=(
            "Torch .pt bundle containing feature_dict and spatial_loc_dict from preprocessing. "
            "Expected keys: feature_dict, spatial_loc_dict, optional section_order."
        ),
    )
    parser.add_argument(
        "--preprocess_config",
        default=None,
        help="Optional preprocessing JSON; used only when --input_bundle is not provided.",
    )
    parser.add_argument("--model_config", default=None, help="Optional JSON overrides for model config.")
    parser.add_argument("--output_dir", default=None, help="Directory for checkpoints and embeddings.")
    parser.add_argument("--section_order", nargs="*", default=None, help="Explicit stage/section order.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N.")
    parser.add_argument("--log_every", type=int, default=GENERIC_CLI_DEFAULTS["log_every"])
    parser.add_argument("--save_every", type=int, default=GENERIC_CLI_DEFAULTS["save_every"], help="Save checkpoint every N epochs; 0 disables.")
    parser.add_argument("--ot_update_interval", type=int, default=None)
    parser.add_argument(
        "--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default=GENERIC_CLI_DEFAULTS["candidate_backend"],
    )
    parser.add_argument("--save_embeddings", action="store_true", help="Save final embeddings as .npy.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a tiny synthetic training smoke test.")
    return parser.parse_args()


def to_float_tensors(feature_dict):
    """Compatibility delegate preserving the original feature-only float32 rule."""
    return _to_float_tensors(feature_dict)


def prepare_training_inputs(args):
    if args.smoke_test:
        return prepare_dataset(SyntheticInput(args.section_order))

    if args.input_bundle:
        return prepare_dataset(BundleInput(args.input_bundle, args.section_order))

    if args.preprocess_config:
        cfg = load_json(args.preprocess_config)
        return prepare_dataset(RawConfigInput(cfg, args.section_order))

    raise ValueError("Provide --input_bundle, --preprocess_config, or --smoke_test.")


def load_preprocessed_inputs(args):
    """Legacy dictionary view of the authoritative PreparedDataset result."""
    return prepare_training_inputs(args).compatibility_context["legacy_inputs"]


def build_synthetic_training_inputs():
    """Compatibility delegate; the original synthetic algorithm lives in data_io."""
    return _build_synthetic_training_inputs()


def build_model_config(args):
    supplied = None
    if args.model_config:
        supplied = load_json(args.model_config)
        # These JSON fields are only consumed by the MouseBrain runner. This
        # entry uses its existing CLI backend and fixed retrieval settings.
        unused = {
            "uot": {
                "candidate_backend", "initial_modality_candidate_k", "candidate_k",
                "faiss_nlist", "faiss_nprobe", "faiss_device",
                "faiss_train_sample_size", "faiss_query_batch_size", "stabilizer",
            },
            "training": {"seed", "max_spots_per_section", "output_dir"},
        }
        fields = [f"{section}.{key}" for section, keys in unused.items()
                  for key in sorted(keys.intersection(supplied.get(section, {})))]
        if fields:
            raise ValueError(f"Unsupported JSON configuration fields for generic trainer: {fields}")
    config = resolve_model_config(
        model_config=supplied,
        explicit_overrides={
            "training": {
                "epochs": args.epochs, "lr": args.lr,
                "weight_decay": args.weight_decay, "device": args.device,
            },
            "uot": {"update_interval": args.ot_update_interval},
        },
    )
    if args.smoke_test:
        config["training"]["epochs"] = args.epochs or GENERIC_SMOKE_DEFAULTS["epochs"]
        config["training"]["device"] = args.device or GENERIC_SMOKE_DEFAULTS["device"]
        config["uot"]["max_iter"] = GENERIC_SMOKE_DEFAULTS["uot_max_iter"]
    return config


def train_stage_model(args):
    config = build_model_config(args)
    prepared = prepare_training_inputs(args)
    inputs = prepared.compatibility_context["legacy_inputs"]
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    section_order = args.section_order or prepared.section_order

    return run_generic_training_task(
        config, prepared, args=args, section_order=section_order,
        policy=YieldExecutionPolicy(
            clear_step_state=False, record_elapsed_time=False,
            allow_empty_epochs=True, record_loss_weights=False,
        ),
    )


def main():
    args = parse_args()
    if args.smoke_test and args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "tmp_verification" / "stage_training_smoke")
    train_stage_model(args)


if __name__ == "__main__":
    main()
