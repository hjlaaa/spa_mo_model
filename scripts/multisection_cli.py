"""Shared CLI parsing and config projection; not an executable entrypoint.

Defaults remain in training.entry_defaults. No raw data, Stage, fit or export.
"""
from __future__ import annotations
import argparse
from typing import Any, Mapping
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from training.entry_defaults import misar_defaults as _entry_defaults
from data_io.misar import SECTION_INFO


DEFAULT_SECTION_ORDER = ["dataset4", "dataset3", "dataset2", "dataset1"]

def get_dataset_defaults(*, dataset_name="MISAR-seq", secondary_name="atac"):
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults(dataset_name=dataset_name, secondary_name=secondary_name, section_order=DEFAULT_SECTION_ORDER)

def parse_args(
    argv: list[str] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
    dataset_name: str = "MISAR-seq",
    secondary_modality: str = "ATAC",
    secondary_name: str = "atac",
):
    parser = argparse.ArgumentParser(description=f"Run {dataset_name} RNA+{secondary_modality} StageMultiModalModel pipeline.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--section_order",
        default=None,
        help="Comma-separated section order used for adjacent OT links.",
    )
    parser.add_argument("--max_spots_per_section", type=int, default=None)
    parser.add_argument("--spot_sampling", choices=["first", "random"], default=None)
    parser.add_argument("--max_shared_genes", type=int, default=None)
    parser.add_argument("--max_shared_peaks", type=int, default=None)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lambda_spatial_gaussian", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--candidate_backend", choices=["faiss_ivf", "faiss_flat", "blockwise"], default=None)
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
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=None)
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument("--graphsage_dropout", type=float, default=None)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=None)
    parser.add_argument("--ot_attention_dropout", type=float, default=None)
    parser.add_argument("--decoder_dropout", type=float, default=None)
    parser.add_argument("--training_loss_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decoder_chunk_size", type=int, default=None)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=None)
    parser.add_argument("--checkpoint_ot_attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_encoder_fusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_decoder_chunks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint_graph_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--cache_spatial_graphs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_candidate_qc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_ot_prior_topk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_cuda_memory_detail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None, help="HVG count for RNA.")
    parser.add_argument(f"--hvg_num_{secondary_name}", type=int, default=None, help=f"Highly variable peak count for {secondary_modality}.")
    parser.add_argument("--spatial_enhancement", action=argparse.BooleanOptionalAction,
                        default=False, help="Add spatial KNN neighbor feature sums after PCA/Harmony.")
    parser.add_argument("--spatial_enhancement_k", type=int, default=10,
                        help="Number of non-self spatial neighbors for preprocessing.")
    parser.add_argument("--spatial_enhancement_weight", type=float, default=0.2)
    parser.add_argument("--spatial_enhancement_include_self",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, default=None)
    return parse_dataset_args(parser, argv, {**get_dataset_defaults(dataset_name=dataset_name, secondary_name=secondary_name), **(defaults or {})})

def parse_section_order(text: str, section_info=SECTION_INFO) -> list[str]:
    order = [item.strip() for item in text.split(",") if item.strip()]
    if not order:
        raise ValueError("--section_order must contain at least one section.")
    unknown = [section for section in order if section not in section_info]
    if unknown:
        raise ValueError(f"Unknown MISAR sections in --section_order: {unknown}")
    if len(set(order)) != len(order):
        raise ValueError("--section_order contains duplicated sections.")
    return order

def make_model_config(args, secondary_modality: str = "ATAC") -> dict[str, Any]:
    config = resolve_model_config(
        input_config={
            "training": {
                "device": args.device, "epochs": int(args.epochs),
                "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            },
            "uot": {
                "max_iter": int(args.uot_max_iter), "topk": int(args.attention_topk),
                "epsilon_update": float(args.uot_epsilon),
                "tau_a": float(args.uot_tau_a), "tau_b": float(args.uot_tau_b),
                "update_interval": int(args.update_interval),
            },
            "graph": {"knn_neighbors_spatial": int(args.spatial_knn_k)},
            "graphsage": {
                "edge_batch_size": int(args.graphsage_edge_batch_size),
                "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
            },
        },
        explicit_overrides={"loss": {
            "lambda_contrast": float(args.lambda_contrast) if args.lambda_contrast is not None else None,
            "lambda_spatial_gaussian": args.lambda_spatial_gaussian,
        }},
    )
    config["model"]["modalities_supported"] = ["RNA", secondary_modality]
    config["model"]["valid_modality_sets"] = [["RNA", secondary_modality]]
    config["encoder"]["dropout"] = float(args.encoder_dropout)
    config["fusion"]["dropout"] = float(args.fusion_dropout)
    config["graphsage"]["dropout"] = float(args.graphsage_dropout)
    config["ot_attention"]["dropout"] = float(args.ot_attention_dropout)
    config["decoder"]["dropout"] = float(args.decoder_dropout)
    config["reconstruction"]["lambda_by_modality"][secondary_modality] = 1.0
    return config

def resolve_run_config(
    args, *, section_info=None, dataset_name="MISAR-seq",
    secondary_modality="ATAC", secondary_name="atac",
):
    section_info = SECTION_INFO if section_info is None else section_info
    return describe_run_config(
        args, make_model_config(args, secondary_modality), dataset=dataset_name,
        section_order=parse_section_order(args.section_order, section_info),
        modalities=["RNA", secondary_modality],
        preprocessing={
            "n_comps": int(args.n_comps),
            "hvg_num": int(args.hvg_num),
            "hvg_num_by_modality": {
                "RNA": int(args.hvg_num),
                secondary_modality: int(getattr(args, f"hvg_num_{secondary_name}")),
            },
            "target_sum": None,
            "use_harmony": not args.no_harmony,
            "spatial_enhancement": {
                "enabled": args.spatial_enhancement,
                "k": args.spatial_enhancement_k,
                "weight": args.spatial_enhancement_weight,
                "include_self": args.spatial_enhancement_include_self,
            },
        },
    )
