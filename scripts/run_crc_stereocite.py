#!/usr/bin/env python3
"""CRC Stereo-CITE-seq RNA+Protein pipeline.

This script reads raw CRC h5ad files, applies RNA ``var_names_make_unique()``
in memory, aligns shared RNA genes, reuses COSIE-style preprocessing, and runs
a StageMultiModalModel dry run or small-scale training. It never writes
processed h5ad files.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.data_preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel, should_update_ot


SAMPLES = {
    "CRC_003": "CRC_003_bin20",
    "CRC_006": "CRC_006_bin20",
}
AUTO_SECTION_KEY_MAP = {
    "s1": "CRC_003",
    "s2": "CRC_006",
}
SUFFIX_RE = re.compile(r".+-[0-9]+$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CRC Stereo-CITE-seq RNA+Protein pipeline.")
    parser.add_argument(
        "--data_dir",
        default="/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq",
        help="Directory containing CRC_003_bin20 and CRC_006_bin20.",
    )
    parser.add_argument(
        "--max_spots_per_section",
        type=int,
        default=None,
        help=(
            "Optional per-section spot subset size. Omit this argument to use all spots; "
            "set a positive integer to run a controlled subset."
        ),
    )
    parser.add_argument("--max_shared_genes", type=int, default=3000)
    parser.add_argument(
        "--spot_sampling",
        choices=["first", "random"],
        default="first",
        help="How to choose the subset spots within each section.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=True,
        help="Run one forward pass only. This is the default unless --train is set.",
    )
    parser.add_argument("--train", action="store_true", help="Run a small training loop after preprocessing.")
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs when --train is set.")
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ot_prior_mode", choices=["dense", "candidate_sparse"], default="dense")
    parser.add_argument(
        "--bidirectional_ot_attention",
        action="store_true",
        help=(
            "Use synchronous bidirectional adjacent-section OT-guided attention. "
            "Only changes candidate_sparse OT prior construction when explicitly enabled."
        ),
    )
    parser.add_argument("--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default="faiss_ivf")
    parser.add_argument("--initial_modality_candidate_k", type=int, default=100)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--faiss_nlist", type=int, default=4096)
    parser.add_argument("--faiss_nprobe", type=int, default=64)
    parser.add_argument("--faiss_device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--faiss_train_sample_size", type=int, default=100000)
    parser.add_argument(
        "--faiss_query_batch_size",
        type=int,
        default=8192,
        help=(
            "Number of source embeddings per FAISS index.search call. "
            "Use a smaller value such as 4096/2048/1024 to reduce FAISS GPU temporary memory."
        ),
    )
    parser.add_argument("--dynamic_candidate_source", choices=["fused", "final"], default="final")
    parser.add_argument("--uot_epsilon", type=float, default=0.05)
    parser.add_argument("--uot_tau_a", type=float, default=1.0)
    parser.add_argument("--uot_tau_b", type=float, default=1.0)
    parser.add_argument("--uot_stabilizer", type=float, default=1e-8)
    parser.add_argument(
        "--graphsage_edge_batch_size",
        type=int,
        default=200000,
        help=(
            "Number of spatial graph edges processed per GraphSAGE message-passing chunk. "
            "Lower this if GraphSAGE OOMs on full-spot runs."
        ),
    )
    parser.add_argument(
        "--training_loss_only",
        action="store_true",
        help="During train epochs, return only loss tensors/scalars instead of full graph-bearing outputs.",
    )
    parser.add_argument(
        "--decoder_chunk_size",
        type=int,
        default=0,
        help="If positive, compute decoder reconstruction loss in spot chunks.",
    )
    parser.add_argument(
        "--ot_attention_source_chunk_size",
        type=int,
        default=0,
        help="If positive, compute OT-guided attention in source-spot chunks.",
    )
    parser.add_argument(
        "--checkpoint_ot_attention",
        action="store_true",
        help=(
            "Activation-checkpoint each OT-guided attention source chunk during training. "
            "This trades extra backward recomputation time for lower activation memory."
        ),
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="Optional CUDA autocast dtype for training forward. Default keeps full float32 behavior.",
    )
    parser.add_argument(
        "--cache_spatial_graphs",
        action="store_true",
        help="Cache CPU spatial KNN graphs and move them to the target device in each forward.",
    )
    parser.add_argument("--save_candidate_qc", action="store_true")
    parser.add_argument("--save_outputs", action="store_true", help="Save lightweight run outputs.")
    parser.add_argument("--save_embeddings", action="store_true", help="Save final embeddings when available.")
    parser.add_argument(
        "--save_ot_prior_topk",
        action="store_true",
        help="Save sparse top-k OT prior when available. Dense P is never saved.",
    )
    parser.add_argument(
        "--log_cuda_memory",
        action="store_true",
        help="Write per-stage CUDA memory statistics to cuda_memory_trace.jsonl in output_dir.",
    )
    parser.add_argument(
        "--log_cuda_memory_detail",
        action="store_true",
        help=(
            "Add optional in-forward CUDA memory detail events to cuda_memory_trace.jsonl. "
            "This is diagnostic-only and implies --log_cuda_memory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/home/hujinlan/spa_mo_model/results/crc_stereocite/dry_run_make_unique_pipeline",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_comps", type=int, default=50)
    parser.add_argument("--hvg_num", type=int, default=3000)
    parser.add_argument("--uot_max_iter", type=int, default=100)
    parser.add_argument("--no_harmony", action="store_true", help="Disable Harmony during preprocessing.")
    return parser.parse_args()


def json_safe(value: Any):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def release_python_and_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_enabled(args) -> bool:
    return bool(args.device == "cuda" and args.amp_dtype != "none")


def autocast_context(args):
    if not amp_enabled(args):
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(args):
    enabled = bool(args.device == "cuda" and args.amp_dtype == "fp16")
    return torch.cuda.amp.GradScaler(enabled=enabled)


class CudaMemoryMonitor:
    """Append lightweight CUDA memory events to a JSONL file."""

    def __init__(self, enabled: bool, output_dir: Path, requested_device: str):
        self.enabled = bool(enabled)
        self.requested_device = requested_device
        self.output_path = output_dir / "cuda_memory_trace.jsonl"
        self.event_count = 0
        if self.enabled:
            self.output_path.write_text("", encoding="utf-8")

    def reset_peak(self) -> None:
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record(
        self,
        stage: str,
        epoch: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        event: dict[str, Any] = {
            "event_index": int(self.event_count),
            "time_sec": float(time.time()),
            "stage": str(stage),
            "epoch": int(epoch) if epoch is not None else None,
            "requested_device": self.requested_device,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            device_index = int(torch.cuda.current_device())
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            event.update(
                {
                    "device_index": device_index,
                    "device_name": torch.cuda.get_device_name(device_index),
                    "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(device_index)),
                    "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(device_index)),
                    "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(device_index)),
                    "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(device_index)),
                    "free_gib": bytes_to_gib(free_bytes),
                    "total_gib": bytes_to_gib(total_bytes),
                }
            )
        if extra:
            event.update(json_safe(dict(extra)))

        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(event), ensure_ascii=False) + "\n")
        self.event_count += 1
        return event


def make_forward_memory_recorder(
    memory_monitor: CudaMemoryMonitor | None,
    enabled: bool,
    epoch: int,
    phase: str,
):
    if memory_monitor is None or not enabled or not memory_monitor.enabled:
        return None

    def record_forward_detail(stage: str, extra: Mapping[str, Any] | None = None) -> None:
        memory_monitor.record(f"{phase}_detail_{stage}", epoch=epoch, extra=extra)

    return record_forward_detail


def spatial_range(spatial: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(spatial)
    return {
        "x": [float(arr[:, 0].min()), float(arr[:, 0].max())],
        "y": [float(arr[:, 1].min()), float(arr[:, 1].max())],
    }


def prepare_rna_var_names_make_unique(rna: ad.AnnData) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply requested in-memory RNA var-name metadata and make_unique logic."""

    original_symbols = pd.Index(rna.var_names.astype(str))
    counts = pd.Series(original_symbols).value_counts()
    was_duplicate = original_symbols.duplicated(keep=False)

    rna.var["gene_symbol_original"] = original_symbols.to_numpy()
    rna.var["was_duplicate_gene_symbol"] = was_duplicate
    rna.var["gene_symbol_original_count"] = [int(counts[symbol]) for symbol in original_symbols]

    # The CRC h5ad index can be categorical. Convert to a plain string Index so
    # AnnData can append engineering suffixes such as "-1" in memory.
    rna.var_names = pd.Index(original_symbols)
    rna.var_names_make_unique()
    if not rna.var_names.is_unique:
        raise ValueError("RNA var_names are still not unique after var_names_make_unique().")
    rna.var["gene_symbol_make_unique"] = rna.var_names.astype(str)

    var_table = pd.DataFrame(
        {
            "gene_symbol_original": original_symbols.to_numpy(),
            "gene_symbol_make_unique": rna.var_names.astype(str).to_numpy(),
            "was_duplicate_gene_symbol": was_duplicate,
            "gene_symbol_original_count": [int(counts[symbol]) for symbol in original_symbols],
        }
    )
    duplicate_summary = (
        var_table[var_table["was_duplicate_gene_symbol"]]
        .groupby("gene_symbol_original", sort=True)
        .agg(
            count=("gene_symbol_make_unique", "size"),
            make_unique_names=("gene_symbol_make_unique", lambda values: ";".join(values)),
        )
        .reset_index()
    )

    info = {
        "original_shape": list(rna.shape),
        "var_names_unique_before": False,
        "make_unique_gene_count": int(rna.n_vars),
        "make_unique_var_names_is_unique": bool(rna.var_names.is_unique),
        "duplicate_gene_symbol_groups": int(duplicate_summary.shape[0]),
        "duplicate_extra_columns": int((duplicate_summary["count"] - 1).sum()) if not duplicate_summary.empty else 0,
        "artificial_suffix_gene_count": int((var_table["gene_symbol_original"] != var_table["gene_symbol_make_unique"]).sum()),
        "artificial_suffix_examples": var_table.loc[
            var_table["gene_symbol_original"] != var_table["gene_symbol_make_unique"],
            "gene_symbol_make_unique",
        ].head(20).tolist(),
    }
    return info, duplicate_summary


def read_backed_pair(data_dir: Path, sample_dir: str) -> tuple[ad.AnnData, ad.AnnData]:
    sample_path = data_dir / sample_dir
    rna_path = sample_path / "adata_RNA.h5ad"
    adt_path = sample_path / "adata_ADT.h5ad"
    if not rna_path.exists():
        raise FileNotFoundError(rna_path)
    if not adt_path.exists():
        raise FileNotFoundError(adt_path)
    return ad.read_h5ad(rna_path, backed="r"), ad.read_h5ad(adt_path, backed="r")


def validate_rna_adt_alignment(section: str, rna: ad.AnnData, adt: ad.AnnData) -> dict[str, Any]:
    obs_match = list(rna.obs_names) == list(adt.obs_names)
    if not obs_match:
        raise ValueError(f"{section}: RNA and ADT obs_names are not identical.")
    if "spatial" not in rna.obsm:
        raise KeyError(f"{section}: RNA is missing obsm['spatial'].")
    if "spatial" not in adt.obsm:
        raise KeyError(f"{section}: ADT is missing obsm['spatial'].")
    rna_spatial = np.asarray(rna.obsm["spatial"])
    adt_spatial = np.asarray(adt.obsm["spatial"])
    spatial_shape_match = rna_spatial.shape == adt_spatial.shape
    if not spatial_shape_match:
        raise ValueError(f"{section}: RNA and ADT spatial shapes differ.")
    spatial_match = bool(np.allclose(rna_spatial, adt_spatial))
    if not spatial_match:
        raise ValueError(f"{section}: RNA and ADT spatial coordinates differ.")
    return {
        "spot_count": int(rna.n_obs),
        "obs_names_match": obs_match,
        "spatial_key_present": True,
        "spatial_shape": list(rna_spatial.shape),
        "spatial_match": spatial_match,
        "spatial_range": spatial_range(rna_spatial),
    }


def select_obs_indices(
    n_obs: int,
    max_spots: int | None,
    sampling: str,
    rng: np.random.Generator,
) -> np.ndarray | slice:
    if max_spots is None or max_spots <= 0 or max_spots >= n_obs:
        return slice(None)
    if sampling == "first":
        return np.arange(max_spots)
    if sampling == "random":
        # Sort sampled indices so backed slicing keeps original spot order.
        return np.sort(rng.choice(n_obs, size=max_spots, replace=False))
    raise ValueError(f"Unsupported spot sampling mode: {sampling}")


def subset_to_memory(
    backed: ad.AnnData,
    obs_indices: np.ndarray | slice,
    var_names: list[str] | None = None,
) -> ad.AnnData:
    view = backed[obs_indices, :] if var_names is None else backed[obs_indices, var_names]
    subset = view.to_memory()
    if "spatial" in subset.obsm:
        subset.obsm["spatial"] = np.asarray(subset.obsm["spatial"]).copy()
    return subset


def summarize_data_dict(data_dict: Mapping[str, list[ad.AnnData | None]]) -> dict[str, Any]:
    summary = {}
    for modality, sections in data_dict.items():
        summary[modality] = [
            None
            if adata_obj is None
            else {
                "shape": list(adata_obj.shape),
                "var_names_is_unique": bool(adata_obj.var_names.is_unique),
                "obsm_keys": list(adata_obj.obsm.keys()),
            }
            for adata_obj in sections
        ]
    return summary


def summarize_feature_dict(feature_dict: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        section: {modality: list(tensor.shape) for modality, tensor in modalities.items()}
        for section, modalities in feature_dict.items()
    }


def summarize_spatial_loc_dict(spatial_loc_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {section: list(np.asarray(spatial).shape) for section, spatial in spatial_loc_dict.items()}


def rename_section_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        AUTO_SECTION_KEY_MAP.get(section, section): value
        for section, value in mapping.items()
    }


def save_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def indexer_to_numpy(indexer: np.ndarray | slice, n_obs: int) -> np.ndarray:
    if isinstance(indexer, slice):
        return np.arange(n_obs, dtype=np.int64)
    return np.asarray(indexer, dtype=np.int64)


def save_selected_spot_indices(
    output_dir: Path,
    obs_indices: Mapping[str, np.ndarray | slice],
    backed_rna: Mapping[str, ad.AnnData],
) -> dict[str, str]:
    paths = {}
    for section, indexer in obs_indices.items():
        path = output_dir / f"selected_spot_indices_{section}.npy"
        np.save(path, indexer_to_numpy(indexer, backed_rna[section].n_obs))
        paths[section] = str(path)
    return paths


def save_spatial_arrays(output_dir: Path, spatial_loc_dict: Mapping[str, Any]) -> dict[str, str]:
    paths = {}
    for section, spatial in spatial_loc_dict.items():
        path = output_dir / f"spatial_{section}.npy"
        np.save(path, np.asarray(spatial))
        paths[section] = str(path)
    return paths


def save_final_embeddings(
    output_dir: Path,
    final_embeddings: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    paths = {}
    for section, tensor in final_embeddings.items():
        path = output_dir / f"final_embeddings_{section}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        paths[section] = str(path)
    return paths


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
    save_candidate_qc: bool = False,
) -> dict[str, dict[str, str]]:
    ensure_dir(output_dir)
    files: dict[str, dict[str, str]] = {}
    if not ot_prior:
        return files

    for (source_section, target_section), prior in ot_prior.items():
        pair_key = f"{source_section}_to_{target_section}"
        required = ["topk_idx", "topk_weight", "confidence", "row_mass"]
        if any(name not in prior for name in required):
            continue

        arrays = {
            "topk_idx": prior["topk_idx"].detach().cpu().numpy(),
            "topk_weight": prior["topk_weight"].detach().cpu().numpy(),
            "confidence": prior["confidence"].detach().cpu().numpy(),
            "row_mass": prior["row_mass"].detach().cpu().numpy(),
        }
        if save_candidate_qc:
            for optional_name in ["raw_topk_mass", "topk_coverage", "tail_mass", "target_hit_count"]:
                if optional_name in prior:
                    arrays[optional_name] = prior[optional_name].detach().cpu().numpy()
        paths = {
            name: output_dir / f"{pair_key}_{name}.npy"
            for name in arrays.keys()
        }
        paths.update({
            "metadata": output_dir / f"{pair_key}_metadata.json",
        })
        for name, array in arrays.items():
            np.save(paths[name], array)

        metadata = {
            "source_section": source_section,
            "target_section": target_section,
            "direction_meaning": "source receives information from target",
            "topk": int(arrays["topk_idx"].shape[1]) if arrays["topk_idx"].ndim == 2 else None,
            "n_source": int(final_embeddings[source_section].shape[0]),
            "n_target": int(final_embeddings[target_section].shape[0]),
            "modalities_used": list(prior.get("modalities_used", [])),
            "has_dense_P": prior.get("P_dense") is not None,
            "run_mode": run_mode,
            "note": "Saved sparse top-k UOT prior from model.ot_prior after final evaluation. X_to_Y means source X receives information from target Y. Dense P was not saved.",
        }
        metadata.update(prior.get("metadata", {}))
        with open(paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(json_safe(metadata), handle, indent=2, ensure_ascii=False)
        files[pair_key] = {name: str(path) for name, path in paths.items()}
    return files


def summarize_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_embedding_shapes": {
            section: list(tensor.shape)
            for section, tensor in outputs["final_embeddings"].items()
        },
        "reconstruction_keys": {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        },
        "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
        "losses": {
            key: float(value.detach().cpu())
            for key, value in outputs["losses"].items()
        },
        "total_loss_finite": bool(torch.isfinite(outputs["losses"]["total_loss"]).item()),
    }


def run_one_forward(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    spatial_loc_dict: Mapping[str, Any],
    processed_data_dict: Any,
    section_order: list[str],
    epoch: int,
    training_loss_only: bool = False,
    decoder_chunk_size: int = 0,
    ot_attention_source_chunk_size: int = 0,
    cache_spatial_graphs: bool = False,
    bidirectional_ot_attention: bool = False,
    checkpoint_ot_attention: bool = False,
    memory_recorder=None,
) -> dict[str, Any]:
    return model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=processed_data_dict,
        section_order=section_order,
        epoch=epoch,
        training_loss_only=training_loss_only,
        decoder_chunk_size=decoder_chunk_size,
        ot_attention_source_chunk_size=ot_attention_source_chunk_size,
        cache_spatial_graphs=cache_spatial_graphs,
        bidirectional_ot_attention=bidirectional_ot_attention,
        checkpoint_ot_attention=checkpoint_ot_attention,
        memory_recorder=memory_recorder,
    )


def sparse_prior_kwargs(args) -> dict[str, Any]:
    return {
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "candidate_backend": args.candidate_backend,
        "faiss_nlist": int(args.faiss_nlist),
        "faiss_nprobe": int(args.faiss_nprobe),
        "faiss_device": args.faiss_device,
        "faiss_train_sample_size": int(args.faiss_train_sample_size),
        "faiss_query_batch_size": (
            int(args.faiss_query_batch_size)
            if args.faiss_query_batch_size is not None
            else None
        ),
        "seed": int(args.seed),
        "epsilon": float(args.uot_epsilon),
        "tau_a": float(args.uot_tau_a),
        "tau_b": float(args.uot_tau_b),
        "max_iter": int(args.uot_max_iter),
        "stabilizer": float(args.uot_stabilizer),
    }


def initialize_model_ot_prior(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    section_order: list[str],
    args,
):
    if args.ot_prior_mode == "dense":
        return model.initialize_ot_prior(feature_dict, section_order=section_order)
    if args.ot_prior_mode == "candidate_sparse":
        kwargs = sparse_prior_kwargs(args)
        return model.initialize_candidate_sparse_ot_prior(
            feature_dict,
            section_order=section_order,
            initial_modality_candidate_k=int(args.initial_modality_candidate_k),
            bidirectional=bool(args.bidirectional_ot_attention),
            **kwargs,
        )
    raise ValueError(f"Unsupported ot_prior_mode: {args.ot_prior_mode}")


def update_model_ot_prior(
    model: StageMultiModalModel,
    eval_outputs: Mapping[str, Any],
    section_order: list[str],
    args,
):
    if args.ot_prior_mode == "dense":
        return model.update_ot_prior(eval_outputs["final_embeddings"], section_order=section_order)
    if args.ot_prior_mode == "candidate_sparse":
        embeddings = (
            eval_outputs["fused_embeddings"]
            if args.dynamic_candidate_source == "fused"
            else eval_outputs["final_embeddings"]
        )
        return model.update_candidate_sparse_ot_prior(
            embeddings,
            section_order=section_order,
            candidate_source=args.dynamic_candidate_source,
            bidirectional=bool(args.bidirectional_ot_attention),
            **sparse_prior_kwargs(args),
        )
    raise ValueError(f"Unsupported ot_prior_mode: {args.ot_prior_mode}")


def train_small_crc_model(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    spatial_loc_dict: Mapping[str, Any],
    processed_data_dict: Any,
    section_order: list[str],
    args,
    memory_monitor: CudaMemoryMonitor | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any], list[int]]:
    epochs = int(args.epochs)
    if epochs <= 0:
        raise ValueError("--epochs must be positive when --train is set.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(args)
    history: list[dict[str, float]] = []
    ot_updates: list[int] = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        if memory_monitor is not None:
            memory_monitor.record("epoch_start", epoch=epoch)
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_forward_start", epoch=epoch)
        with autocast_context(args):
            outputs = run_one_forward(
                model,
                feature_dict,
                spatial_loc_dict,
                processed_data_dict,
                section_order,
                epoch=epoch,
                training_loss_only=bool(args.training_loss_only),
                decoder_chunk_size=int(args.decoder_chunk_size),
                ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                cache_spatial_graphs=bool(args.cache_spatial_graphs),
                bidirectional_ot_attention=bool(args.bidirectional_ot_attention),
                checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                memory_recorder=make_forward_memory_recorder(
                    memory_monitor,
                    bool(args.log_cuda_memory_detail),
                    epoch,
                    "train_forward",
                ),
            )
        forward_memory = (
            memory_monitor.record("epoch_forward_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )
        loss = outputs["losses"]["total_loss"].float()
        optimizer.zero_grad(set_to_none=True)
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_backward_start", epoch=epoch)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_memory = (
            memory_monitor.record("epoch_backward_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_optimizer_step_start", epoch=epoch)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_memory = (
            memory_monitor.record("epoch_optimizer_step_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )

        total_loss_value = float(loss.detach().cpu().item())
        crossview_loss_value = float(outputs["losses"]["crossview_loss"].detach().cpu().item())
        reconstruction_loss_value = float(outputs["losses"]["reconstruction_loss"].detach().cpu().item())
        record = {
            "epoch": int(epoch),
            "lambda_contrast": float(model.config["loss"]["lambda_contrast"]),
            "total_loss": total_loss_value,
            "crossview_loss": crossview_loss_value,
            "reconstruction_loss": reconstruction_loss_value,
            "elapsed_time_sec": float(time.time() - start_time),
        }
        record["weighted_crossview_loss"] = record["lambda_contrast"] * record["crossview_loss"]
        if memory_monitor is not None:
            stage_events = [forward_memory, backward_memory, optimizer_memory]
            record.update(
                {
                    "cuda_allocated_gib": optimizer_memory.get("allocated_gib"),
                    "cuda_reserved_gib": optimizer_memory.get("reserved_gib"),
                    "cuda_epoch_max_allocated_gib": max(
                        (
                            event.get("max_allocated_gib", 0.0) or 0.0
                            for event in stage_events
                        ),
                        default=0.0,
                    ),
                    "cuda_epoch_max_reserved_gib": max(
                        (
                            event.get("max_reserved_gib", 0.0) or 0.0
                            for event in stage_events
                        ),
                        default=0.0,
                    ),
                    "cuda_forward_max_allocated_gib": forward_memory.get("max_allocated_gib"),
                    "cuda_backward_max_allocated_gib": backward_memory.get("max_allocated_gib"),
                    "cuda_optimizer_max_allocated_gib": optimizer_memory.get("max_allocated_gib"),
                }
            )
        history.append(record)

        if args.log_every > 0 and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == epochs):
            print(
                f"epoch={epoch} total={record['total_loss']:.6f} "
                f"lambda_contrast={record['lambda_contrast']:.6g} "
                f"weighted_crossview={record['weighted_crossview_loss']:.6f} "
                f"crossview={record['crossview_loss']:.6f} "
                f"reconstruction={record['reconstruction_loss']:.6f}"
            )

        if memory_monitor is not None:
            memory_monitor.record("epoch_cleanup_start", epoch=epoch)
        del outputs
        del loss
        release_python_and_cuda_cache()
        if memory_monitor is not None:
            memory_monitor.record("epoch_cleanup_end", epoch=epoch)

        if should_update_ot(epoch, int(args.update_interval)):
            model.eval()
            with torch.no_grad():
                if memory_monitor is not None:
                    memory_monitor.reset_peak()
                    memory_monitor.record("ot_update_forward_start", epoch=epoch)
                eval_outputs = run_one_forward(
                    model,
                    feature_dict,
                    spatial_loc_dict,
                    processed_data_dict,
                    section_order,
                    epoch=epoch,
                    decoder_chunk_size=int(args.decoder_chunk_size),
                    ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                    cache_spatial_graphs=bool(args.cache_spatial_graphs),
                    bidirectional_ot_attention=bool(args.bidirectional_ot_attention),
                    checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                    memory_recorder=make_forward_memory_recorder(
                        memory_monitor,
                        bool(args.log_cuda_memory_detail),
                        epoch,
                        "ot_update_forward",
                    ),
                )
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_forward_end", epoch=epoch)
                    memory_monitor.reset_peak()
                    memory_monitor.record("ot_update_prior_start", epoch=epoch)
                update_model_ot_prior(model, eval_outputs, section_order, args)
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_prior_end", epoch=epoch)
                del eval_outputs
                release_python_and_cuda_cache()
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_cleanup_end", epoch=epoch)
            ot_updates.append(epoch)
            print(f"Updated OT prior at epoch {epoch}.")

    model.eval()
    with torch.no_grad():
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("final_eval_start", epoch=epochs)
        final_outputs = run_one_forward(
            model,
            feature_dict,
            spatial_loc_dict,
            processed_data_dict,
            section_order,
            epoch=epochs,
            decoder_chunk_size=int(args.decoder_chunk_size),
            ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
            cache_spatial_graphs=bool(args.cache_spatial_graphs),
            bidirectional_ot_attention=bool(args.bidirectional_ot_attention),
            checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
            memory_recorder=make_forward_memory_recorder(
                memory_monitor,
                bool(args.log_cuda_memory_detail),
                epochs,
                "final_eval_forward",
            ),
        )
        if memory_monitor is not None:
            memory_monitor.record("final_eval_end", epoch=epochs)
    return history, final_outputs, ot_updates


def run_crc_pipeline(args) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    if args.max_spots_per_section is not None and args.max_spots_per_section <= 0:
        raise ValueError("--max_spots_per_section must be a positive integer when provided.")
    if args.max_shared_genes is not None and args.max_shared_genes <= 0:
        raise ValueError("--max_shared_genes must be positive when provided.")
    if args.train and args.epochs <= 0:
        raise ValueError("--epochs must be positive when --train is set.")
    for name in [
        "initial_modality_candidate_k",
        "candidate_k",
        "attention_topk",
        "faiss_nlist",
        "faiss_nprobe",
        "faiss_train_sample_size",
        "uot_max_iter",
        "graphsage_edge_batch_size",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    for name in ["decoder_chunk_size", "ot_attention_source_chunk_size"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.amp_dtype != "none" and args.device != "cuda":
        raise ValueError("--amp_dtype can only be enabled with --device cuda.")
    if args.log_cuda_memory_detail:
        args.log_cuda_memory = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    memory_monitor = CudaMemoryMonitor(
        enabled=bool(args.log_cuda_memory),
        output_dir=output_dir,
        requested_device=args.device,
    )
    memory_monitor.record("script_start")

    backed_rna: dict[str, ad.AnnData] = {}
    backed_adt: dict[str, ad.AnnData] = {}
    rna_info: dict[str, dict[str, Any]] = {}
    duplicate_tables: dict[str, pd.DataFrame] = {}
    alignment_summary: dict[str, dict[str, Any]] = {}

    try:
        for section, sample_dir in SAMPLES.items():
            rna, adt = read_backed_pair(data_dir, sample_dir)
            backed_rna[section] = rna
            backed_adt[section] = adt
            rna_info[section], duplicate_tables[section] = prepare_rna_var_names_make_unique(rna)
            alignment_summary[section] = validate_rna_adt_alignment(section, rna, adt)
        memory_monitor.record("read_raw_h5ad_make_unique_validate_alignment")

        rna003 = backed_rna["CRC_003"]
        rna006 = backed_rna["CRC_006"]
        shared_set_006 = set(rna006.var_names.astype(str))
        shared_genes_all = [gene for gene in rna003.var_names.astype(str) if gene in shared_set_006]
        if not shared_genes_all:
            raise ValueError("No shared RNA genes found after var_names_make_unique().")
        selected_shared_genes = (
            shared_genes_all[: args.max_shared_genes]
            if args.max_shared_genes is not None
            else shared_genes_all
        )

        rna003_var = rna003.var.loc[selected_shared_genes]
        shared_suffix_genes_selected = [
            gene
            for gene in selected_shared_genes
            if bool(SUFFIX_RE.match(gene))
            and str(rna003_var.loc[gene, "gene_symbol_original"]) != gene
        ]
        rna003_var_all = rna003.var.loc[shared_genes_all]
        shared_suffix_genes_all = [
            gene
            for gene in shared_genes_all
            if bool(SUFFIX_RE.match(gene))
            and str(rna003_var_all.loc[gene, "gene_symbol_original"]) != gene
        ]
        memory_monitor.record(
            "shared_gene_alignment",
            extra={
                "shared_gene_count_total": int(len(shared_genes_all)),
                "shared_gene_count_used": int(len(selected_shared_genes)),
            },
        )

        adt_markers_003 = list(map(str, backed_adt["CRC_003"].var_names))
        adt_markers_006 = list(map(str, backed_adt["CRC_006"].var_names))
        adt_marker_set_same = set(adt_markers_003) == set(adt_markers_006)
        adt_marker_order_same = adt_markers_003 == adt_markers_006
        if not backed_adt["CRC_003"].var_names.is_unique or not backed_adt["CRC_006"].var_names.is_unique:
            raise ValueError("ADT marker var_names must be unique.")
        if not adt_marker_set_same:
            raise ValueError("CRC_003 and CRC_006 ADT marker sets differ.")
        if not adt_marker_order_same:
            raise ValueError("CRC_003 and CRC_006 ADT marker order differs.")

        for section, table in duplicate_tables.items():
            table.to_csv(output_dir / f"duplicate_gene_summary_{section}.csv", index=False)
        save_list(output_dir / "shared_gene_symbols_make_unique.txt", selected_shared_genes)
        save_list(output_dir / "adt_marker_list.txt", adt_markers_003)
        memory_monitor.record("saved_lightweight_metadata")

        obs_indices = {
            section: select_obs_indices(
                backed_rna[section].n_obs,
                args.max_spots_per_section,
                args.spot_sampling,
                rng,
            )
            for section in SAMPLES
        }
        selected_obs_names_preview = {
            section: list(backed_rna[section].obs_names[indices][:10])
            if not isinstance(indices, slice)
            else list(backed_rna[section].obs_names[:10])
            for section, indices in obs_indices.items()
        }
        saved_selected_spot_indices = {}
        if args.save_outputs:
            saved_selected_spot_indices = save_selected_spot_indices(output_dir, obs_indices, backed_rna)
        memory_monitor.record(
            "selected_spots",
            extra={
                "max_spots_per_section": int(args.max_spots_per_section)
                if args.max_spots_per_section is not None
                else None,
                "spot_sampling": args.spot_sampling,
            },
        )

        rna003_mem = subset_to_memory(backed_rna["CRC_003"], obs_indices["CRC_003"], selected_shared_genes)
        rna006_mem = subset_to_memory(backed_rna["CRC_006"], obs_indices["CRC_006"], selected_shared_genes)
        adt003_mem = subset_to_memory(backed_adt["CRC_003"], obs_indices["CRC_003"], None)
        adt006_mem = subset_to_memory(backed_adt["CRC_006"], obs_indices["CRC_006"], None)
        memory_monitor.record(
            "loaded_selected_anndata_to_memory",
            extra={
                "rna_CRC_003_shape": list(rna003_mem.shape),
                "rna_CRC_006_shape": list(rna006_mem.shape),
                "adt_CRC_003_shape": list(adt003_mem.shape),
                "adt_CRC_006_shape": list(adt006_mem.shape),
            },
        )

        if list(rna003_mem.var_names) != list(rna006_mem.var_names):
            raise ValueError("Subset RNA shared gene order is not identical after alignment.")
        if not rna003_mem.var_names.is_unique or not rna006_mem.var_names.is_unique:
            raise ValueError("Subset RNA var_names are not unique.")

        data_dict = {
            "RNA": [rna003_mem, rna006_mem],
            "Protein": [adt003_mem, adt006_mem],
            "HE": [None, None],
            "Metabolite": [None, None],
        }
        memory_monitor.record("constructed_data_dict")

        memory_monitor.reset_peak()
        memory_monitor.record("cosie_preprocessing_start")
        feature_dict_raw, spatial_loc_dict_raw, processed_data_dict = load_cosie_style_data(
            data_dict,
            n_comps=args.n_comps,
            hvg_num=args.hvg_num,
            hvg_num_by_modality={"RNA": args.hvg_num, "Protein": None},
            target_sum=None,
            use_harmony=not args.no_harmony,
            metacell=False,
        )
        preprocessing_generated_keys = list(feature_dict_raw.keys())
        feature_dict = rename_section_keys(feature_dict_raw)
        spatial_loc_dict = rename_section_keys(spatial_loc_dict_raw)
        section_order = ["CRC_003", "CRC_006"]
        memory_monitor.record(
            "cosie_preprocessing_end",
            extra={
                "feature_dict_shapes": summarize_feature_dict(feature_dict),
                "spatial_loc_dict_shapes": summarize_spatial_loc_dict(spatial_loc_dict),
            },
        )

        model_config = get_default_model_config()
        model_config["training"]["device"] = args.device
        model_config["training"]["epochs"] = int(args.epochs)
        model_config["training"]["lr"] = float(args.lr)
        model_config["training"]["weight_decay"] = float(args.weight_decay)
        if args.lambda_contrast is not None:
            model_config["loss"]["lambda_contrast"] = float(args.lambda_contrast)
        model_config["uot"]["max_iter"] = int(args.uot_max_iter)
        model_config["uot"]["topk"] = int(args.attention_topk)
        model_config["uot"]["epsilon_update"] = float(args.uot_epsilon)
        model_config["uot"]["tau_a"] = float(args.uot_tau_a)
        model_config["uot"]["tau_b"] = float(args.uot_tau_b)
        model_config["uot"]["check_every"] = 10
        model_config["uot"]["tol"] = 1e-5
        model_config["uot"]["update_interval"] = int(args.update_interval)
        model_config["graphsage"]["edge_batch_size"] = int(args.graphsage_edge_batch_size)
        memory_monitor.reset_peak()
        memory_monitor.record("model_init_start")
        model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
        memory_monitor.record("model_init_end")
        resolved_modality_order = list(model._resolve_modality_order(feature_dict["CRC_003"]))
        if resolved_modality_order != ["RNA", "Protein"]:
            raise ValueError(f"Expected ['RNA', 'Protein'], got {resolved_modality_order}.")

        memory_monitor.reset_peak()
        memory_monitor.record("initial_ot_prior_start")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        memory_monitor.record("initial_ot_prior_end")
        initial_prior = model.ot_prior[("CRC_003", "CRC_006")]
        if args.bidirectional_ot_attention and args.ot_prior_mode == "candidate_sparse":
            expected_keys = {("CRC_003", "CRC_006"), ("CRC_006", "CRC_003")}
            actual_keys = set((model.ot_prior or {}).keys())
            if not expected_keys.issubset(actual_keys):
                raise ValueError(f"Bidirectional OT prior is missing direction keys: {expected_keys - actual_keys}")
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        if args.ot_prior_mode == "dense" and initial_ot_modalities_used != ["RNA", "Protein"]:
            raise ValueError(f"Unexpected initial OT modalities_used: {initial_ot_modalities_used}")
        if args.ot_prior_mode == "candidate_sparse" and initial_prior.get("metadata", {}).get("ot_prior_mode") != "candidate_sparse":
            raise ValueError("Expected candidate_sparse initial OT prior metadata.")
        history: list[dict[str, float]] | None = None
        ot_updates: list[int] = []
        if args.train:
            history, outputs, ot_updates = train_small_crc_model(
                model,
                feature_dict,
                spatial_loc_dict,
                processed_data_dict,
                section_order,
                args,
                memory_monitor=memory_monitor,
            )
        else:
            with torch.no_grad():
                memory_monitor.reset_peak()
                memory_monitor.record("dry_run_forward_start", epoch=0)
                outputs = run_one_forward(
                    model,
                    feature_dict,
                    spatial_loc_dict,
                    processed_data_dict,
                    section_order,
                    epoch=0,
                    decoder_chunk_size=int(args.decoder_chunk_size),
                    ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                    cache_spatial_graphs=bool(args.cache_spatial_graphs),
                    bidirectional_ot_attention=bool(args.bidirectional_ot_attention),
                    checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                    memory_recorder=make_forward_memory_recorder(
                        memory_monitor,
                        bool(args.log_cuda_memory_detail),
                        0,
                        "dry_run_forward",
                    ),
                )
                memory_monitor.record("dry_run_forward_end", epoch=0)

        prior = outputs["ot_prior"][("CRC_003", "CRC_006")]
        reconstruction_keys = {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        }
        ot_modalities_used = list(prior.get("modalities_used", []))
        total_loss_finite = bool(torch.isfinite(outputs["losses"]["total_loss"]).item())
        if reconstruction_keys != {"CRC_003": ["Protein", "RNA"], "CRC_006": ["Protein", "RNA"]}:
            raise ValueError(f"Unexpected reconstruction keys: {reconstruction_keys}")
        # Initial UOT is multimodal RNA/Protein. After a training-time dynamic
        # update, the prior is recomputed from final embeddings by design.
        accepted_ot_modalities = [["RNA", "Protein"]]
        if args.train and ot_updates:
            accepted_ot_modalities.append(["final_embedding"])
            accepted_ot_modalities.append([f"{args.dynamic_candidate_source}_embedding"])
        if args.ot_prior_mode == "candidate_sparse":
            accepted_ot_modalities.append(["RNA", "Protein"])
            accepted_ot_modalities.append([f"{args.dynamic_candidate_source}_embedding"])
        if ot_modalities_used not in accepted_ot_modalities:
            raise ValueError(f"Unexpected OT modalities_used: {ot_modalities_used}")
        if args.bidirectional_ot_attention and args.ot_prior_mode == "candidate_sparse":
            reverse_prior = outputs["ot_prior"].get(("CRC_006", "CRC_003"))
            if reverse_prior is None:
                raise ValueError("Bidirectional OT prior is missing CRC_006<-CRC_003 direction.")
            if int(prior["topk_idx"].max().item()) >= int(feature_dict["CRC_006"]["RNA"].shape[0]):
                raise ValueError("CRC_003<-CRC_006 topk_idx contains out-of-range target indices.")
            if int(reverse_prior["topk_idx"].max().item()) >= int(feature_dict["CRC_003"]["RNA"].shape[0]):
                raise ValueError("CRC_006<-CRC_003 topk_idx contains out-of-range target indices.")
        if not total_loss_finite:
            raise ValueError("total_loss is not finite.")

        embedding_paths = {}
        spatial_paths = {}
        ot_prior_topk_files = {}
        if args.save_embeddings:
            memory_monitor.reset_peak()
            memory_monitor.record("save_embeddings_start")
            embedding_paths = save_final_embeddings(output_dir, outputs["final_embeddings"])
            memory_monitor.record("save_embeddings_end")
        if args.save_outputs:
            memory_monitor.reset_peak()
            memory_monitor.record("save_spatial_arrays_start")
            spatial_paths = save_spatial_arrays(output_dir, spatial_loc_dict)
            memory_monitor.record("save_spatial_arrays_end")
        if args.save_ot_prior_topk:
            memory_monitor.reset_peak()
            memory_monitor.record("save_ot_prior_topk_start")
            ot_prior_topk_files = save_ot_prior_topk(
                output_dir / "ot_prior_topk",
                outputs.get("ot_prior"),
                outputs["final_embeddings"],
                "training_final_eval" if args.train else "dry_run",
                save_candidate_qc=bool(args.save_candidate_qc),
            )
            memory_monitor.record("save_ot_prior_topk_end")

        sample_summaries = {}
        for section in section_order:
            sample_summaries[section] = {
                **alignment_summary[section],
                "shared_rna_gene_count_total": int(len(shared_genes_all)),
                "shared_rna_gene_count_used": int(len(selected_shared_genes)),
                "adt_marker_count": int(len(adt_markers_003)),
                "subset_spot_count": int(feature_dict[section]["RNA"].shape[0]),
            }

        summary = {
            "mode": "train" if args.train else "dry_run",
            "input_data_path": str(data_dir),
            "output_dir": str(output_dir),
            "section_names": section_order,
            "preprocessing_generated_keys": preprocessing_generated_keys,
            "section_key_mapping": AUTO_SECTION_KEY_MAP,
            "seed": int(args.seed),
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            "use_harmony": not args.no_harmony,
            "spot_sampling": args.spot_sampling,
            "selected_obs_names_preview": selected_obs_names_preview,
            "train": bool(args.train),
            "epochs": int(args.epochs) if args.train else 0,
            "lambda_contrast": float(model_config["loss"]["lambda_contrast"]),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "update_interval": int(args.update_interval),
            "ot_updates": ot_updates,
            "uot_max_iter": int(args.uot_max_iter),
            "ot_prior_mode": args.ot_prior_mode,
            "bidirectional_ot_attention": bool(args.bidirectional_ot_attention),
            "candidate_backend": args.candidate_backend,
            "initial_modality_candidate_k": int(args.initial_modality_candidate_k),
            "candidate_k": int(args.candidate_k),
            "attention_topk": int(args.attention_topk),
            "faiss_nlist": int(args.faiss_nlist),
            "faiss_nprobe": int(args.faiss_nprobe),
            "faiss_device": args.faiss_device,
            "faiss_train_sample_size": int(args.faiss_train_sample_size),
            "faiss_query_batch_size": (
                int(args.faiss_query_batch_size)
                if args.faiss_query_batch_size is not None
                else None
            ),
            "dynamic_candidate_source": args.dynamic_candidate_source,
            "uot_epsilon": float(args.uot_epsilon),
            "uot_tau_a": float(args.uot_tau_a),
            "uot_tau_b": float(args.uot_tau_b),
            "uot_stabilizer": float(args.uot_stabilizer),
            "graphsage_edge_batch_size": int(args.graphsage_edge_batch_size),
            "training_loss_only": bool(args.training_loss_only),
            "decoder_chunk_size": int(args.decoder_chunk_size),
            "ot_attention_source_chunk_size": int(args.ot_attention_source_chunk_size),
            "checkpoint_ot_attention": bool(args.checkpoint_ot_attention),
            "amp_dtype": args.amp_dtype,
            "amp_enabled": bool(amp_enabled(args)),
            "cache_spatial_graphs": bool(args.cache_spatial_graphs),
            "log_cuda_memory": bool(args.log_cuda_memory),
            "log_cuda_memory_detail": bool(args.log_cuda_memory_detail),
            "cuda_memory_trace_path": str(memory_monitor.output_path) if args.log_cuda_memory else None,
            "cuda_memory_event_count_before_summary": int(memory_monitor.event_count),
            "save_candidate_qc": bool(args.save_candidate_qc),
            "ot_prior_metadata": {
                f"{source}_to_{target}": prior.get("metadata", {})
                for (source, target), prior in (outputs.get("ot_prior") or {}).items()
            },
            "rna_make_unique": rna_info,
            "shared_gene_count_total": int(len(shared_genes_all)),
            "shared_gene_count_used": int(len(selected_shared_genes)),
            "shared_gene_artificial_suffix_count_total": int(len(shared_suffix_genes_all)),
            "shared_gene_artificial_suffix_count_used": int(len(shared_suffix_genes_selected)),
            "shared_gene_artificial_suffix_examples": shared_suffix_genes_all[:20],
            "max_spots_per_section": (
                int(args.max_spots_per_section)
                if args.max_spots_per_section is not None
                else None
            ),
            "max_shared_genes": int(args.max_shared_genes) if args.max_shared_genes is not None else None,
            "protein_marker_count": int(len(adt_markers_003)),
            "adt_var_names_unique": {
                "CRC_003": bool(backed_adt["CRC_003"].var_names.is_unique),
                "CRC_006": bool(backed_adt["CRC_006"].var_names.is_unique),
            },
            "adt_marker_set_same": bool(adt_marker_set_same),
            "adt_marker_order_same": bool(adt_marker_order_same),
            "alignment": sample_summaries,
            "data_dict": summarize_data_dict(data_dict),
            "feature_dict_shapes": summarize_feature_dict(feature_dict),
            "spatial_loc_dict_shapes": summarize_spatial_loc_dict(spatial_loc_dict),
            "processed_data_dict_generated": processed_data_dict is not None,
            "continued_into_preprocessing": True,
            "feature_dict_generated": True,
            "model_initialized": True,
            "initial_ot_prior_initialized": True,
            "epoch_1_completed": bool(history and len(history) >= 1),
            "epoch_2_completed": bool(history and len(history) >= 2),
            "resolved_modality_order": resolved_modality_order,
            "final_embedding_shapes": {
                section: list(tensor.shape)
                for section, tensor in outputs["final_embeddings"].items()
            },
            "reconstruction_keys": reconstruction_keys,
            "ot_prior_keys": [list(key) for key in outputs["ot_prior"].keys()],
            "initial_ot_prior_modalities_used": {
                "CRC_003_to_CRC_006": initial_ot_modalities_used,
            },
            "ot_prior_modalities_used": {
                "CRC_003_to_CRC_006": ot_modalities_used,
            },
            "losses": {
                key: float(value.detach().cpu())
                for key, value in outputs["losses"].items()
            },
            "total_loss_finite": total_loss_finite,
            "training_history": history,
            "saved_embeddings": bool(embedding_paths),
            "saved_ot_prior_topk": bool(ot_prior_topk_files),
            "ot_prior_topk_saved": bool(ot_prior_topk_files),
            "ot_prior_topk_reason": None if ot_prior_topk_files else (
                "not requested" if not args.save_ot_prior_topk else "no sparse OT prior available"
            ),
            "saved_files": {
                "run_summary": str(output_dir / "run_summary.json"),
                "loss_history": str(output_dir / "loss_history.json") if history is not None else None,
                "shared_gene_symbols_make_unique": str(output_dir / "shared_gene_symbols_make_unique.txt"),
                "adt_marker_list": str(output_dir / "adt_marker_list.txt"),
                "duplicate_gene_summary_CRC_003": str(output_dir / "duplicate_gene_summary_CRC_003.csv"),
                "duplicate_gene_summary_CRC_006": str(output_dir / "duplicate_gene_summary_CRC_006.csv"),
                "selected_spot_indices": saved_selected_spot_indices,
                "spatial": spatial_paths,
                "final_embeddings": embedding_paths,
                "ot_prior_topk": ot_prior_topk_files,
            },
            "confirmations": {
                "saved_processed_h5ad": False,
                "overwrote_original_h5ad": False,
                "created_processed_data_directory": False,
                "fabricated_HE": False,
                "fabricated_Metabolite": False,
                "adt_used_as_Protein": True,
                "adt_used_as_Metabolite": False,
                "count_aggregation_performed": False,
                "full_crc_uot_constructed": False,
            },
            "notes": [
                "Suffixes such as MATR3-1 and ABCF2-1 are engineering names generated by AnnData var_names_make_unique(); they are not biological gene IDs.",
                "selected shared genes follow CRC_003 make_unique order.",
                "max_shared_genes limits the dry-run gene list for speed and should be revisited for formal experiments.",
            ],
        }
        memory_monitor.record("write_run_summary_start")
        with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
        if history is not None:
            memory_monitor.record("write_loss_history_start")
            with open(output_dir / "loss_history.json", "w", encoding="utf-8") as handle:
                json.dump(json_safe(history), handle, indent=2, ensure_ascii=False)
            memory_monitor.record("write_loss_history_end")
        memory_monitor.record("write_run_summary_end")

        print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
        print("CRC_STEREOCITE_TRAIN: PASS" if args.train else "CRC_STEREOCITE_DRY_RUN: PASS")
        return summary
    finally:
        for adata_obj in list(backed_rna.values()) + list(backed_adt.values()):
            if getattr(adata_obj, "isbacked", False):
                adata_obj.file.close()


def main() -> None:
    args = parse_args()
    run_crc_pipeline(args)


if __name__ == "__main__":
    main()
