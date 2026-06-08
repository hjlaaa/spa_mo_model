#!/usr/bin/env python3
"""Run MouseBrain HE+RNA+Metabolite data through the V2 stage model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.data_preprocessing import load_cosie_style_data
from model.multimodal_preprocessing import (
    build_cosie_data_dict,
    build_mousebrain_section,
    summarize_data_dict,
    summarize_feature_dict,
    summarize_spatial_loc_dict,
)
from model.stage_model import StageMultiModalModel, should_update_ot
from model.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run MouseBrain real-data V2 test.")
    parser.add_argument("--config", required=True, help="MouseBrain preprocessing/training JSON.")
    parser.add_argument("--dry_run", action="store_true", help="Run preprocessing + one forward pass only.")
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
    parser.add_argument(
        "--save_ot_prior_topk",
        action="store_true",
        help="Save sparse top-k OT prior tensors for downstream matching QC.",
    )
    parser.add_argument(
        "--ot_prior_output_dir",
        default=None,
        help="Optional directory for sparse top-k OT prior files; defaults to output_dir/ot_prior_topk.",
    )
    return parser.parse_args()


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def lambda_for_epoch(epoch: int, schedule, default_lambda: float) -> float:
    if schedule is None:
        return float(default_lambda)
    for start_epoch, end_epoch, value in schedule:
        if start_epoch <= epoch <= end_epoch:
            return float(value)
    return float(default_lambda)


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


def subset_section_result(section_result: dict[str, Any], max_spots: int | None):
    if max_spots is None:
        return section_result
    if max_spots <= 0:
        raise ValueError("--max_spots_per_section must be positive.")

    subset_modalities = {}
    for modality, adata in section_result["modalities"].items():
        if adata is None:
            subset_modalities[modality] = None
            continue
        if adata.n_obs < max_spots:
            raise ValueError(
                f"{section_result['section_id']} {modality} has only {adata.n_obs} spots, "
                f"cannot take first {max_spots}."
            )
        subset_modalities[modality] = adata[:max_spots].copy()

    section_result = dict(section_result)
    section_result["modalities"] = subset_modalities
    section_result["messages"] = list(section_result.get("messages", [])) + [
        f"{section_result['section_id']}: subset to first {max_spots} spots for all modalities"
    ]
    return section_result


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


def build_mousebrain_sections(config: Mapping[str, Any], max_spots: int | None):
    preprocessing = config.get("preprocessing", {})
    spatial_key = preprocessing.get("spatial_key", "spatial")
    uni_feature_key = preprocessing.get("uni_feature_key", "uni_feature")
    rna_gene_id_key = preprocessing.get("rna_gene_id_key", "gene_ids")

    section_results = []
    messages = []
    for section in config["sections"]:
        result = build_mousebrain_section(
            section_id=section["section_id"],
            rna_path=section["rna_input"],
            metabolite_path=section["metabolite_input"],
            spatial_key=spatial_key,
            uni_feature_key=uni_feature_key,
            rna_gene_id_key=rna_gene_id_key,
        )
        result = subset_section_result(result, max_spots=max_spots)
        section_results.append(result)
        messages.extend(result.get("messages", []))
    return section_results, messages


def preprocess_mousebrain(config: Mapping[str, Any], max_spots: int | None):
    preprocessing = config.get("preprocessing", {})
    section_results, messages = build_mousebrain_sections(config, max_spots=max_spots)
    data_dict = build_cosie_data_dict(section_results)
    feature_dict, spatial_loc_dict, processed_data_dict = load_cosie_style_data(
        data_dict,
        n_comps=preprocessing.get("n_comps", 50),
        hvg_num=preprocessing.get("hvg_num", 3000),
        hvg_num_by_modality=preprocessing.get("hvg_num_by_modality"),
        target_sum=preprocessing.get("target_sum"),
        use_harmony=preprocessing.get("use_harmony", True),
        metacell=preprocessing.get("metacell", False),
    )
    return {
        "section_results": section_results,
        "messages": messages,
        "data_dict": data_dict,
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "processed_data_dict": processed_data_dict,
    }


def build_model_config(
    config: Mapping[str, Any],
    epochs: int | None,
    lambda_contrast: float | None = None,
    device: str | None = None,
):
    model_config = get_default_model_config()
    training = config.get("training", {})
    model_config["training"]["epochs"] = int(epochs if epochs is not None else training.get("epochs", 5))
    model_config["training"]["lr"] = float(training.get("lr", model_config["training"]["lr"]))
    model_config["training"]["weight_decay"] = float(training.get("weight_decay", model_config["training"]["weight_decay"]))
    if "device" in training:
        model_config["training"]["device"] = training["device"]
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
        model_config["training"]["device"] = device
    if "model" in config:
        recursive_update(model_config, config["model"])
    if lambda_contrast is not None:
        model_config["loss"]["lambda_contrast"] = float(lambda_contrast)
    return model_config


def recursive_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


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
    embedding_dir = output_dir / "final_embeddings"
    ensure_dir(embedding_dir)
    paths = {}
    for section, tensor in final_embeddings.items():
        path = embedding_dir / f"{section}_final_embedding.npy"
        np.save(path, tensor.detach().cpu().numpy())
        paths[section] = str(path)
    return paths


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
):
    """Save sparse top-k UOT priors without saving dense coupling matrices."""

    ensure_dir(output_dir)
    files: dict[str, dict[str, str]] = {}
    if not ot_prior:
        return files

    note = "Saved sparse top-k UOT prior from model.ot_prior after final evaluation."
    for (source_section, target_section), prior in ot_prior.items():
        pair_key = f"{source_section}_to_{target_section}"
        topk_idx = prior["topk_idx"].detach().cpu().numpy()
        topk_weight = prior["topk_weight"].detach().cpu().numpy()
        confidence = prior["confidence"].detach().cpu().numpy()
        row_mass = prior["row_mass"].detach().cpu().numpy()

        paths = {
            "topk_idx": output_dir / f"{pair_key}_topk_idx.npy",
            "topk_weight": output_dir / f"{pair_key}_topk_weight.npy",
            "confidence": output_dir / f"{pair_key}_confidence.npy",
            "row_mass": output_dir / f"{pair_key}_row_mass.npy",
            "metadata": output_dir / f"{pair_key}_metadata.json",
        }
        np.save(paths["topk_idx"], topk_idx)
        np.save(paths["topk_weight"], topk_weight)
        np.save(paths["confidence"], confidence)
        np.save(paths["row_mass"], row_mass)

        metadata = {
            "source_section": source_section,
            "target_section": target_section,
            "topk": int(topk_idx.shape[1]) if topk_idx.ndim == 2 else None,
            "n_source": int(final_embeddings[source_section].shape[0]),
            "n_target": int(final_embeddings[target_section].shape[0]),
            "modalities_used": list(prior.get("modalities_used", [])),
            "has_dense_P": prior.get("P_dense") is not None,
            "run_mode": run_mode,
            "note": note,
        }
        with open(paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(json_safe(metadata), handle, indent=2, ensure_ascii=False)

        files[pair_key] = {name: str(path) for name, path in paths.items()}

    return files


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
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(full_summary), handle, indent=2, ensure_ascii=False)
    if history is not None:
        with open(output_dir / "loss_history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)


def run_mousebrain(args):
    if args.lambda_contrast is not None and args.lambda_contrast_schedule is not None:
        raise ValueError("Use either --lambda_contrast or --lambda_contrast_schedule, not both.")

    config_path = Path(args.config)
    config = load_json(config_path)
    training = config.get("training", {})
    max_spots = args.max_spots_per_section
    if max_spots is None:
        max_spots = training.get("max_spots_per_section")

    output_dir = Path(args.output_dir or training.get("output_dir", PROJECT_ROOT / "results" / "mousebrain_test"))
    if max_spots is not None:
        output_dir = output_dir / f"maxspots_{max_spots}"
    if args.dry_run:
        if output_dir.name != "dry_run":
            output_dir = output_dir / "dry_run"
    else:
        epochs_for_dir = int(args.epochs if args.epochs is not None else training.get("epochs", 5))
        epoch_dir_name = f"epochs_{epochs_for_dir}"
        if output_dir.name != epoch_dir_name:
            output_dir = output_dir / epoch_dir_name

    prep = preprocess_mousebrain(config, max_spots=max_spots)
    feature_dict = prep["feature_dict"]
    spatial_loc_dict = prep["spatial_loc_dict"]
    processed_data_dict = prep["processed_data_dict"]
    section_order = config.get("section_order") or sorted(feature_dict.keys())
    model_config = build_model_config(
        config,
        epochs=args.epochs,
        lambda_contrast=args.lambda_contrast,
        device=args.device,
    )
    lambda_schedule = parse_lambda_contrast_schedule(args.lambda_contrast_schedule)
    epochs = int(model_config["training"]["epochs"])

    model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
    model.initialize_ot_prior(feature_dict, section_order=section_order)

    with torch.no_grad():
        dry_outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            processed_data_dict=processed_data_dict,
            section_order=section_order,
            epoch=0,
        )

    preprocessing_summary = {
        "dataset_name": config.get("dataset_name", "MouseBrain"),
        "section_order": section_order,
        "max_spots_per_section": max_spots,
        "use_harmony": config.get("preprocessing", {}).get("use_harmony", True),
        "hvg_num_by_modality": config.get("preprocessing", {}).get("hvg_num_by_modality"),
        "lambda_contrast": model_config["loss"]["lambda_contrast"],
        "lambda_contrast_schedule": args.lambda_contrast_schedule,
        "device": model_config["training"]["device"],
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["training"]["lr"]),
        weight_decay=float(model_config["training"]["weight_decay"]),
    )
    update_interval = int(model_config["uot"]["update_interval"])
    history = []
    ot_updates = []

    for epoch in range(1, epochs + 1):
        model.train()
        current_lambda_contrast = lambda_for_epoch(
            epoch,
            lambda_schedule,
            model_config["loss"]["lambda_contrast"],
        )
        model.config["loss"]["lambda_contrast"] = current_lambda_contrast
        outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            processed_data_dict=processed_data_dict,
            section_order=section_order,
            epoch=epoch,
        )
        loss = outputs["losses"]["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "lambda_contrast": current_lambda_contrast,
            "total_loss": float(outputs["losses"]["total_loss"].detach().cpu()),
            "crossview_loss": float(outputs["losses"]["crossview_loss"].detach().cpu()),
            "reconstruction_loss": float(outputs["losses"]["reconstruction_loss"].detach().cpu()),
        }
        record["weighted_crossview_loss"] = record["lambda_contrast"] * record["crossview_loss"]
        history.append(record)
        print(
            f"epoch={epoch} total={record['total_loss']:.6f} "
            f"lambda_contrast={record['lambda_contrast']:.6g} "
            f"weighted_crossview={record['weighted_crossview_loss']:.6f} "
            f"crossview={record['crossview_loss']:.6f} "
            f"reconstruction={record['reconstruction_loss']:.6f}"
        )

        if should_update_ot(epoch, update_interval):
            model.eval()
            with torch.no_grad():
                eval_outputs = model(
                    feature_dict=feature_dict,
                    spatial_loc_dict=spatial_loc_dict,
                    processed_data_dict=processed_data_dict,
                    section_order=section_order,
                    epoch=epoch,
                )
                model.update_ot_prior(eval_outputs["final_embeddings"], section_order=section_order)
            ot_updates.append(epoch)
            print(f"Updated OT prior at epoch {epoch}.")

    model.eval()
    model.config["loss"]["lambda_contrast"] = lambda_for_epoch(
        epochs,
        lambda_schedule,
        model_config["loss"]["lambda_contrast"],
    )
    with torch.no_grad():
        final_outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            processed_data_dict=processed_data_dict,
            section_order=section_order,
            epoch=epochs,
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
        "epochs": epochs,
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


def main():
    args = parse_args()
    run_mousebrain(args)


if __name__ == "__main__":
    main()
