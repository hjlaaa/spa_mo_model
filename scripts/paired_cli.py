"""Shared CLI parsing and config projection; not an executable entrypoint.

Defaults remain in training.entry_defaults. No raw data, Stage, fit or export.
"""
from __future__ import annotations
import argparse
from typing import Any, Mapping
from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from training.entry_defaults import crc_defaults as _entry_defaults


SAMPLES = {
    "CRC_003": "CRC_003_bin20",
    "CRC_006": "CRC_006_bin20",
}

def get_dataset_defaults():
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults()

def parse_args(
    argv=None, *, defaults=None, dataset_name="CRC Stereo-CITE-seq",
    sample_dirs=("CRC_003_bin20", "CRC_006_bin20"),
):
    parser = argparse.ArgumentParser(description=f"Run {dataset_name} RNA+Protein pipeline.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help=f"Directory containing {sample_dirs[0]} and {sample_dirs[1]}.",
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
    parser.add_argument("--max_shared_genes", type=int, default=None)
    parser.add_argument(
        "--spot_sampling",
        choices=["first", "random"],
        default=None,
        help="How to choose the subset spots within each section.",
    )
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, help="Run a small training loop after preprocessing.", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs when --train is set.")
    parser.add_argument("--lambda_contrast", type=float, default=None)
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
    parser.add_argument(
        "--faiss_query_batch_size",
        type=int,
        default=None,
        help=(
            "Number of source embeddings per FAISS index.search call. "
            "Use a smaller value such as 4096/2048/1024 to reduce FAISS GPU temporary memory."
        ),
    )
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument(
        "--spatial_knn_k",
        type=int,
        default=None,
        help=(
            "Number of non-self spatial neighbors for the weighted KNN graph. "
            "The graph also includes self-loops."
        ),
    )
    parser.add_argument(
        "--graphsage_edge_batch_size",
        type=int,
        default=None,
        help=(
            "Number of spatial graph edges processed per GraphSAGE message-passing chunk. "
            "Lower this if GraphSAGE OOMs on full-spot runs."
        ),
    )
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument(
        "--training_loss_only",
        action=argparse.BooleanOptionalAction,
        help="During train epochs, return only loss tensors/scalars instead of full graph-bearing outputs.",
        default=None,
    )
    parser.add_argument(
        "--decoder_chunk_size",
        type=int,
        default=None,
        help="If positive, compute decoder reconstruction loss in spot chunks.",
    )
    parser.add_argument(
        "--ot_attention_source_chunk_size",
        type=int,
        default=None,
        help="If positive, compute OT-guided attention in source-spot chunks.",
    )
    parser.add_argument(
        "--checkpoint_ot_attention",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint each OT-guided attention source chunk during training. "
            "This trades extra backward recomputation time for lower activation memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_encoder_fusion",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint modality-specific encoder and FusionMLP forwards during training. "
            "This trades extra backward recomputation time for lower activation memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_decoder_chunks",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint each chunked decoder reconstruction loss during training. "
            "Requires --decoder_chunk_size > 0 and is most useful with --training_loss_only."
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint_graph_encoder",
        action=argparse.BooleanOptionalAction,
        help=(
            "Activation-checkpoint the graph encoder forward during training. "
            "The current graph encoder implementation is WeightedResidualGraphSAGE."
        ),
        default=None,
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["none", "bf16", "fp16"],
        default=None,
        help="Optional CUDA autocast dtype for training forward. Default keeps full float32 behavior.",
    )
    parser.add_argument(
        "--cache_spatial_graphs",
        action=argparse.BooleanOptionalAction,
        help="Cache CPU spatial KNN graphs and move them to the target device in each forward.",
        default=None,
    )
    parser.add_argument("--save_candidate_qc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction, help="Save lightweight run outputs.", default=None)
    parser.add_argument("--save_embeddings", action=argparse.BooleanOptionalAction, help="Save final embeddings when available.", default=None)
    parser.add_argument(
        "--save_ot_prior_topk",
        action=argparse.BooleanOptionalAction,
        help="Save sparse top-k OT prior when available. Dense P is never saved.",
        default=None,
    )
    parser.add_argument(
        "--log_cuda_memory",
        action=argparse.BooleanOptionalAction,
        help="Write per-stage CUDA memory statistics to cuda_memory_trace.jsonl in output_dir.",
        default=None,
    )
    parser.add_argument(
        "--log_cuda_memory_detail",
        action=argparse.BooleanOptionalAction,
        help=(
            "Add optional in-forward CUDA memory detail events to cuda_memory_trace.jsonl. "
            "This is diagnostic-only and implies --log_cuda_memory."
        ),
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, help="Disable Harmony during preprocessing.", default=None)
    return parse_dataset_args(parser, argv, {**get_dataset_defaults(), **(defaults or {})})

def build_model_config(args):
    model_config = resolve_model_config(
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
        explicit_overrides={"loss": {"lambda_contrast": (
            float(args.lambda_contrast) if args.lambda_contrast is not None else None
        )}},
    )
    return model_config

def resolve_run_config(args, *, samples=None, dataset_name="CRC Stereo-CITE-seq"):
    samples = SAMPLES if samples is None else samples
    return describe_run_config(
        args, build_model_config(args), dataset=dataset_name,
        section_order=list(samples), modalities=["RNA", "Protein"],
        preprocessing={
            "n_comps": args.n_comps,
            "hvg_num": args.hvg_num,
            "hvg_num_by_modality": {"RNA": args.hvg_num, "Protein": None},
            "target_sum": None,
            "use_harmony": not args.no_harmony,
        },
    )
