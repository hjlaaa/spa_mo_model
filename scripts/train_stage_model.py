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

from model.configure import get_default_model_config
from model.multimodal_preprocessing import preprocess_multisection_cosie_style
from model.stage_model import StageMultiModalModel, should_update_ot
from model.utils import ensure_dir


def recursive_update(base: dict[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    if updates is None:
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


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
    parser.add_argument("--save_embeddings", action="store_true", help="Save final embeddings as .npy.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a tiny synthetic training smoke test.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
        sections = cfg["sections"]
        result = preprocess_multisection_cosie_style(
            sections=sections,
            n_comps=cfg.get("n_comps", 50),
            hvg_num=cfg.get("hvg_num", 3000),
            target_sum=cfg.get("target_sum"),
            use_harmony=cfg.get("use_harmony", True),
            metacell=cfg.get("metacell", False),
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
    config = get_default_model_config()
    if args.model_config:
        recursive_update(config, load_json(args.model_config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.ot_update_interval is not None:
        config["uot"]["update_interval"] = args.ot_update_interval
    if args.smoke_test:
        config["training"]["epochs"] = args.epochs or 3
        config["training"]["device"] = args.device or "cpu"
        config["uot"]["max_iter"] = 50
        config["uot"]["tol"] = 1e-5
    return config


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


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
    inputs = load_preprocessed_inputs(args)
    feature_dict = inputs["feature_dict"]
    spatial_loc_dict = inputs["spatial_loc_dict"]
    section_order = args.section_order or inputs.get("section_order")
    config = build_model_config(args)

    model = StageMultiModalModel(config=config, feature_dict=feature_dict)
    model.initialize_ot_prior(feature_dict, section_order=section_order)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    epochs = int(config["training"]["epochs"])
    update_interval = int(config["uot"]["update_interval"])
    history: list[dict[str, float]] = []
    last_outputs = None

    for epoch in range(1, epochs + 1):
        model.train()
        outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            section_order=section_order,
            epoch=epoch,
        )
        loss = outputs["losses"]["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "total_loss": float(outputs["losses"]["total_loss"].detach().cpu()),
            "crossview_loss": float(outputs["losses"]["crossview_loss"].detach().cpu()),
            "reconstruction_loss": float(outputs["losses"]["reconstruction_loss"].detach().cpu()),
        }
        history.append(record)
        last_outputs = outputs

        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == epochs):
            print(
                f"epoch={epoch} "
                f"total={record['total_loss']:.6f} "
                f"crossview={record['crossview_loss']:.6f} "
                f"reconstruction={record['reconstruction_loss']:.6f}"
            )

        if should_update_ot(epoch, update_interval):
            model.eval()
            with torch.no_grad():
                eval_outputs = model(
                    feature_dict=feature_dict,
                    spatial_loc_dict=spatial_loc_dict,
                    section_order=section_order,
                    epoch=epoch,
                )
                model.update_ot_prior(eval_outputs["final_embeddings"], section_order=section_order)
            print(f"Updated OT prior at epoch {epoch}.")

        if args.output_dir and args.save_every > 0 and epoch % args.save_every == 0:
            output_dir = Path(args.output_dir)
            save_training_artifacts(
                output_dir=output_dir,
                model=model,
                config=config,
                history=history,
                outputs=last_outputs,
                section_order=section_order,
                save_embeddings=False,
            )

    model.eval()
    with torch.no_grad():
        final_outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            section_order=section_order,
            epoch=epochs,
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
