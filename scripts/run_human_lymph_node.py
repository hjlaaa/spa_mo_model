#!/usr/bin/env python3
"""Train spa_mo_model on paired Human Lymph Node RNA + ADT sections."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_crc_stereocite as crc
from data_io.paired import (
    read_lymph_pair, prepare_rna_gene_ids, filter_globally_nonzero_genes,
)


DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Human_Lymph_Node")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/human_lymph_node/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)

SAMPLES = {
    "Human_Lymph_Node_A1": "Human_Lymph_Node_A1",
    "Human_Lymph_Node_D1": "Human_Lymph_Node_D1",
}


def get_dataset_defaults(*, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Dataset-specific values over the shared parser defaults."""
    return {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'train': True,
        'epochs': 200,
        'seed': 42,
        'device': 'cuda',
        'max_shared_genes': 20000,
        'hvg_num': 3000,
        'lambda_contrast': 0.1,
        'candidate_backend': 'faiss_ivf',
        'faiss_nlist': 256,
        'faiss_nprobe': 32,
        'faiss_train_sample_size': 10000,
        'faiss_query_batch_size': 2048,
        'initial_modality_candidate_k': 100,
        'candidate_k': 200,
        'attention_topk': 10,
        'spatial_knn_k': 10,
        'graphsage_edge_batch_size': 100000,
        'training_loss_only': True,
        'decoder_chunk_size': 2048,
        'ot_attention_source_chunk_size': 1024,
        'checkpoint_ot_attention': True,
        'checkpoint_encoder_fusion': True,
        'checkpoint_decoder_chunks': True,
        'checkpoint_graph_encoder': True,
        'amp_dtype': 'bf16',
        'cache_spatial_graphs': True,
        'save_candidate_qc': True,
        'save_outputs': True,
        'save_embeddings': True,
        'save_ot_prior_topk': True,
        'log_cuda_memory': True,
    }


def parse_args(
    argv=None, *, data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
    dataset_name="Human Lymph Node", samples=None,
):
    # Preserve the existing wrapper defaults; explicit CLI values still win.
    defaults = get_dataset_defaults(data_dir=data_dir, output_dir=output_dir)
    return crc.parse_args(
        argv, defaults=defaults, dataset_name=dataset_name,
        sample_dirs=tuple((SAMPLES if samples is None else samples).values()),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    run_config = crc.resolve_run_config(args, samples=SAMPLES, dataset_name="Human Lymph Node")
    crc.run_crc_pipeline(
        args, samples=SAMPLES, read_pair=read_lymph_pair,
        prepare_rna=prepare_rna_gene_ids,
        filter_shared_genes=filter_globally_nonzero_genes,
        status_prefix="HUMAN_LYMPH_NODE",
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
