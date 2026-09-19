#!/usr/bin/env python3
"""Train the V2 StageMultiModalModel from COSIE-style preprocessing outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import load_json, resolve_model_config
from data_io.datasets import preprocess_multisection_cosie_style
from model.stage_model import StageMultiModalModel
from model.tensor_utils import tensor_to_numpy
from training.fit import iter_fit_model
from data_io.common import ensure_dir


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
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs; 0 disables.")
    parser.add_argument("--ot_update_interval", type=int, default=None)
    parser.add_argument(
        "--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default="faiss_ivf",
    )
    parser.add_argument("--save_embeddings", action="store_true", help="Save final embeddings as .npy.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a tiny synthetic training smoke test.")
    return parser.parse_args()


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


def to_float_tensors(feature_dict):
    converted = {}
    for section, modalities in feature_dict.items():
        converted[section] = {}
        for modality, value in modalities.items():
            if isinstance(value, torch.Tensor):
                converted[section][modality] = value.float()
            else:
                converted[section][modality] = torch.as_tensor(value, dtype=torch.float32)
    return converted


def load_preprocessed_inputs(args):
    if args.smoke_test:
        return build_synthetic_training_inputs()

    if args.input_bundle:
        bundle = torch.load(args.input_bundle, map_location="cpu")
        if "feature_dict" not in bundle or "spatial_loc_dict" not in bundle:
            raise KeyError("--input_bundle must contain feature_dict and spatial_loc_dict.")
        feature_dict = to_float_tensors(bundle["feature_dict"])
        spatial_loc_dict = bundle["spatial_loc_dict"]
        section_order = args.section_order or bundle.get("section_order")
        return {
            "feature_dict": feature_dict,
            "spatial_loc_dict": spatial_loc_dict,
            "processed_data_dict": bundle.get("processed_data_dict"),
            "section_order": section_order,
            "messages": ["Loaded preprocessed tensors from input_bundle."],
        }

    if args.preprocess_config:
        cfg = load_json(args.preprocess_config)
        misplaced = {"n_comps", "hvg_num", "target_sum", "use_harmony"}.intersection(
            cfg.get("preprocessing", {})
        )
        if misplaced:
            raise ValueError(
                f"Generic preprocessing expects these fields at the top level: {sorted(misplaced)}"
            )
        sections = cfg["sections"]
        result = preprocess_multisection_cosie_style(
            sections=sections,
            n_comps=cfg.get("n_comps", 50),
            hvg_num=cfg.get("hvg_num", 3000),
            target_sum=cfg.get("target_sum"),
            use_harmony=cfg.get("use_harmony", True),
            config=cfg,
        )
        section_order = args.section_order or result.get("section_ids")
        return {
            "feature_dict": result["feature_dict"],
            "spatial_loc_dict": result["spatial_loc_dict"],
            "processed_data_dict": result["processed_data_dict"],
            "section_order": section_order,
            "messages": result.get("messages", []),
        }

    raise ValueError("Provide --input_bundle, --preprocess_config, or --smoke_test.")


def build_synthetic_training_inputs():
    torch.manual_seed(88)
    rng = np.random.default_rng(88)
    feature_dict = {
        "s1": {
            "HE": torch.randn(36, 50),
            "RNA": torch.randn(36, 50),
            "Protein": torch.randn(36, 20),
        },
        "s2": {
            "HE": torch.randn(42, 50),
            "RNA": torch.randn(42, 50),
            "Protein": torch.randn(42, 20),
        },
    }
    spatial_loc_dict = {
        "s1": rng.random((36, 2)),
        "s2": rng.random((42, 2)),
    }
    return {
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "processed_data_dict": None,
        "section_order": ["s1", "s2"],
        "messages": ["Built synthetic smoke-test inputs."],
    }


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
        config["training"]["epochs"] = args.epochs or 3
        config["training"]["device"] = args.device or "cpu"
        config["uot"]["max_iter"] = 50
    return config


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
    torch.save(checkpoint, output_dir / "stage_model_v2_last.pt")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "section_order": section_order,
                "output_shapes": summarize_outputs(outputs),
                "config": json_safe(config),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if save_embeddings:
        embedding_dir = output_dir / "final_embeddings"
        ensure_dir(embedding_dir)
        for section, embedding in outputs["final_embeddings"].items():
            np.save(embedding_dir / f"{section}_final_embedding.npy", tensor_to_numpy(embedding))


def train_stage_model(args):
    config = build_model_config(args)
    inputs = load_preprocessed_inputs(args)
    feature_dict = inputs["feature_dict"]
    spatial_loc_dict = inputs["spatial_loc_dict"]
    section_order = args.section_order or inputs.get("section_order")

    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
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
        clear_step_state=False,
        record_elapsed_time=False,
        allow_empty_epochs=True,
        record_loss_weights=False,
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


def main():
    args = parse_args()
    if args.smoke_test and args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "tmp_verification" / "stage_training_smoke")
    train_stage_model(args)


if __name__ == "__main__":
    main()
