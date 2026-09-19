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
from model.stage_model import StageMultiModalModel
from model.tensor_utils import tensor_to_numpy
from data_io.common import ensure_dir
from training.fit import (
    initialize_model_ot_prior, lambda_for_epoch, sparse_prior_kwargs,
    train_small_crc_model, update_model_ot_prior,
)
from training.config import (
    load_json, resolve_model_config, resolve_option_values, config_source_layers, parse_dataset_args,
)


MODEL_DEFAULTS = {"training": {"epochs": 5}, "uot": {"max_iter": 100}}
UOT_HELPER_DEFAULTS = {
    "candidate_backend": "faiss_ivf",
    "initial_modality_candidate_k": 100,
    "candidate_k": 200,
    "faiss_nlist": 256,
    "faiss_nprobe": 32,
    "faiss_device": "auto",
    "faiss_train_sample_size": 10000,
    "faiss_query_batch_size": 2048,
    "stabilizer": 1e-8,
}

TRAINING_DEFAULTS = {
    "seed": 42,
    "max_spots_per_section": None,
    "output_dir": str(PROJECT_ROOT / "results" / "mousebrain_test"),
}


def get_dataset_defaults():
    """Current entry defaults; suite arguments remain explicit overrides."""
    return {
        'config': None,
        'dry_run': False,
        'epochs': None,
        'max_spots_per_section': None,
        'lambda_contrast': None,
        'lambda_contrast_schedule': None,
        'device': None,
        'output_dir': None,
        'seed': None,
        'candidate_backend': None,
        'initial_modality_candidate_k': None,
        'candidate_k': None,
        'attention_topk': None,
        'faiss_nlist': None,
        'faiss_nprobe': None,
        'faiss_device': None,
        'faiss_train_sample_size': None,
        'faiss_query_batch_size': None,
        'uot_epsilon': None,
        'uot_tau_a': None,
        'uot_tau_b': None,
        'uot_max_iter': None,
        'uot_stabilizer': None,
        'update_interval': None,
        'spatial_knn_k': None,
        'post_ot_graphsage_scale': None,
        'save_ot_prior_topk': False,
        'ot_prior_output_dir': None,
    }


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    misplaced = {
        "n_comps", "hvg_num", "hvg_num_by_modality", "target_sum", "use_harmony",
        "spatial_key", "uni_feature_key", "rna_gene_id_key",
    }.intersection(config)
    if misplaced:
        raise ValueError(
            f"MouseBrain expects these fields inside preprocessing: {sorted(misplaced)}"
        )
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
    reject_unsupported_preprocess_config(config)
    preprocessing = config.get("preprocessing", {})
    if config.get("metacell", False) or preprocessing.get("metacell", False):
        raise ValueError("metacell=true is no longer supported; use full-spot inputs.")
    section_results, messages = build_mousebrain_sections(config, max_spots=max_spots)
    data_dict = build_cosie_data_dict(section_results)
    feature_dict, spatial_loc_dict, processed_data_dict = load_cosie_style_data(
        data_dict,
        n_comps=preprocessing.get("n_comps", 50),
        hvg_num=preprocessing.get("hvg_num", 3000),
        hvg_num_by_modality=preprocessing.get("hvg_num_by_modality"),
        target_sum=preprocessing.get("target_sum"),
        use_harmony=preprocessing.get("use_harmony", True),
    )
    section_ids = [result["section_id"] for result in section_results]
    feature_dict = restore_section_keys(feature_dict, section_ids)
    spatial_loc_dict = restore_section_keys(spatial_loc_dict, section_ids)
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
        np.save(path, tensor_to_numpy(tensor))
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
            "run_mode": run_mode,
            "note": note,
        }
        metadata.update(prior.get("metadata", {}))
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

    prep = preprocess_mousebrain(config, max_spots=max_spots)
    feature_dict = prep["feature_dict"]
    spatial_loc_dict = prep["spatial_loc_dict"]
    processed_data_dict = prep["processed_data_dict"]
    section_order = config.get("section_order") or sorted(feature_dict.keys())
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

    model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
    initial_prior = initialize_model_ot_prior(
        model,
        feature_dict,
        section_order,
        args,
    )

    with torch.no_grad():
        dry_outputs = model(
            feature_dict=feature_dict,
            spatial_loc_dict=spatial_loc_dict,
            processed_data_dict=processed_data_dict,
            section_order=section_order,
            epoch=0,
        )

    first_parameter = next(model.parameters())
    resolved_config["execution"]["device"] = str(first_parameter.device)
    resolved_config["execution"]["parameter_dtype"] = str(first_parameter.dtype)

    preprocessing_summary = {
        "dataset_name": config.get("dataset_name", "MouseBrain"),
        "section_order": section_order,
        "max_spots_per_section": max_spots,
        "use_harmony": config.get("preprocessing", {}).get("use_harmony", True),
        "hvg_num_by_modality": config.get("preprocessing", {}).get("hvg_num_by_modality"),
        "lambda_contrast": model_config["loss"]["lambda_contrast"],
        "lambda_contrast_schedule": args.lambda_contrast_schedule,
        "device": model_config["training"]["device"],
        "seed": int(args.seed),
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "candidate_backend": str(args.candidate_backend),
        "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "dynamic_candidate_source": "ot",
        "uot_epsilon": float(args.uot_epsilon),
        "uot_tau_a": float(args.uot_tau_a),
        "uot_tau_b": float(args.uot_tau_b),
        "uot_max_iter": int(args.uot_max_iter),
        "update_interval": int(args.update_interval),
        "spatial_knn_k": int(args.spatial_knn_k),
        "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "topology_aware_refresh_enabled": bool(
            model_config["uot"].get("topology_aware_refresh_enabled", False)
        ),
        "topology_context_weight": float(
            model_config["uot"].get("topology_context_weight", 0.0)
        ),
        "initial_ot_prior_metadata": {
            f"{left}_to_{right}": json_safe(prior.get("metadata", {}))
            for (left, right), prior in initial_prior.items()
        },
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
            "resolved_config": resolved_config,
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

    execution = resolved_config["execution"]
    fit_args = argparse.Namespace(
        **vars(args),
        lr=float(model_config["training"]["lr"]),
        weight_decay=float(model_config["training"]["weight_decay"]),
        amp_dtype=execution["amp_dtype"],
        training_loss_only=execution["training_loss_only"],
        decoder_chunk_size=execution["decoder_chunk_size"] or 0,
        ot_attention_source_chunk_size=execution["ot_attention_source_chunk_size"] or 0,
        cache_spatial_graphs=execution["cache_spatial_graphs"],
        checkpoint_ot_attention=execution["checkpoint_ot_attention"],
        checkpoint_encoder_fusion=execution["checkpoint_encoder_fusion"],
        checkpoint_decoder_chunks=execution["checkpoint_decoder_chunks"],
        checkpoint_graph_encoder=execution["checkpoint_graph_encoder"],
        log_every=1,
        log_cuda_memory_detail=False,
    )
    history, final_outputs, ot_updates = train_small_crc_model(
        model, feature_dict, spatial_loc_dict, processed_data_dict, section_order,
        fit_args,
        lambda_contrast_schedule=lambda_schedule,
        clear_step_state=False,
        record_elapsed_time=False,
        allow_empty_epochs=True,
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
        "resolved_config": resolved_config,
        "epochs": epochs,
        "seed": int(args.seed),
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "candidate_backend": str(args.candidate_backend),
        "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "dynamic_candidate_source": "ot",
        "uot_epsilon": float(args.uot_epsilon),
        "uot_tau_a": float(args.uot_tau_a),
        "uot_tau_b": float(args.uot_tau_b),
        "uot_max_iter": int(args.uot_max_iter),
        "update_interval": int(args.update_interval),
        "spatial_knn_k": int(args.spatial_knn_k),
        "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "topology_aware_refresh_enabled": bool(
            model_config["uot"].get("topology_aware_refresh_enabled", False)
        ),
        "topology_context_weight": float(
            model_config["uot"].get("topology_context_weight", 0.0)
        ),
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
