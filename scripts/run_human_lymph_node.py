#!/usr/bin/env python3
"""Train spa_mo_model on paired Human Lymph Node RNA + ADT sections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ORIGINAL = Path(__file__).with_name("run_crc_stereocite.py")
DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Human_Lymph_Node")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/human_lymph_node/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)


def _load_adapted_pipeline() -> dict:
    source = ORIGINAL.read_text(encoding="utf-8")
    replacements = [
        ("CRC_003_bin20", "Human_Lymph_Node_A1"),
        ("CRC_006_bin20", "Human_Lymph_Node_D1"),
        ("CRC_003", "Human_Lymph_Node_A1"),
        ("CRC_006", "Human_Lymph_Node_D1"),
        ("CRC Stereo-CITE-seq", "Human Lymph Node"),
        ("CRC_STEREOCITE", "HUMAN_LYMPH_NODE"),
    ]
    for old, new in replacements:
        source = source.replace(old, new)

    shared_assignment = (
        "shared_genes_all = [gene for gene in rna003.var_names.astype(str) "
        "if gene in shared_set_006]"
    )
    if shared_assignment not in source:
        raise RuntimeError("Could not locate shared-gene assignment in the base pipeline.")
    source = source.replace(
        shared_assignment,
        shared_assignment
        + "\n        shared_genes_all = filter_globally_nonzero_genes("
        "rna003, rna006, shared_genes_all)",
        1,
    )

    namespace = {
        "__file__": str(ORIGINAL),
        "__name__": "human_lymph_node_spa_mo_pipeline",
    }
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    return namespace


def _set_gene_ids(adata, modality: str) -> tuple[pd.Index, pd.Index]:
    if "gene_ids" not in adata.var:
        raise KeyError(f"{modality} is missing required var['gene_ids'].")
    original = pd.Index(adata.var_names.astype(str))
    feature_ids = pd.Index(adata.var["gene_ids"].astype(str))
    if feature_ids.hasnans or not feature_ids.is_unique:
        raise ValueError(f"{modality} var['gene_ids'] must be complete and unique.")
    adata.var[f"{modality.lower()}_name_original"] = original.to_numpy()
    adata.var_names = feature_ids
    return original, feature_ids


def _prepare_rna_gene_ids(rna):
    symbols, feature_ids = _set_gene_ids(rna, "RNA")
    counts = pd.Series(symbols).value_counts()
    duplicated = symbols.duplicated(keep=False)
    rna.var["gene_symbol_original"] = symbols.to_numpy()
    rna.var["was_duplicate_gene_symbol"] = duplicated
    rna.var["gene_symbol_original_count"] = [int(counts[s]) for s in symbols]
    rna.var["gene_symbol_make_unique"] = feature_ids.to_numpy()

    table = pd.DataFrame(
        {
            "gene_symbol_original": symbols.to_numpy(),
            "gene_symbol_make_unique": feature_ids.to_numpy(),
            "was_duplicate_gene_symbol": duplicated,
            "gene_symbol_original_count": [int(counts[s]) for s in symbols],
        }
    )
    duplicate_summary = (
        table[table["was_duplicate_gene_symbol"]]
        .groupby("gene_symbol_original", sort=True)
        .agg(
            count=("gene_symbol_make_unique", "size"),
            make_unique_names=("gene_symbol_make_unique", lambda x: ";".join(x)),
        )
        .reset_index()
    )
    info = {
        "original_shape": list(rna.shape),
        "var_names_unique_before": bool(symbols.is_unique),
        "make_unique_gene_count": int(rna.n_vars),
        "make_unique_var_names_is_unique": True,
        "duplicate_gene_symbol_groups": int(len(duplicate_summary)),
        "duplicate_extra_columns": int(duplicated.sum() - len(duplicate_summary)),
        "artificial_suffix_gene_count": 0,
        "artificial_suffix_examples": [],
        "feature_id_source": "var['gene_ids']",
    }
    return info, duplicate_summary


def _to_memory_matrix(adata, feature_ids: list[str]):
    matrix = adata[:, feature_ids].X
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return matrix


def _filter_globally_nonzero_genes(rna_a, rna_d, feature_ids: list[str]) -> list[str]:
    total = np.zeros(len(feature_ids), dtype=np.float64)
    for adata in (rna_a, rna_d):
        matrix = _to_memory_matrix(adata, feature_ids)
        total += np.asarray(matrix.sum(axis=0)).ravel()
    kept = [feature for feature, value in zip(feature_ids, total) if value > 0]
    print(f"Removed {len(feature_ids) - len(kept)} globally all-zero RNA genes.")
    return kept


def _inject_defaults() -> None:
    defaults = [
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(OUTPUT_DIR),
        "--train",
        "--epochs", "200",
        "--seed", "42",
        "--device", "cuda",
        "--max_shared_genes", "20000",
        "--hvg_num", "3000",
        "--lambda_contrast", "0.1",
        "--ot_prior_mode", "candidate_sparse",
        "--bidirectional_ot_attention",
        "--candidate_backend", "faiss_ivf",
        "--faiss_nlist", "256",
        "--faiss_nprobe", "32",
        "--faiss_train_sample_size", "10000",
        "--faiss_query_batch_size", "2048",
        "--initial_modality_candidate_k", "100",
        "--candidate_k", "200",
        "--attention_topk", "10",
        "--spatial_knn_k", "10",
        "--graphsage_edge_batch_size", "100000",
        "--training_loss_only",
        "--decoder_chunk_size", "2048",
        "--ot_attention_source_chunk_size", "1024",
        "--checkpoint_ot_attention",
        "--checkpoint_encoder_fusion",
        "--checkpoint_decoder_chunks",
        "--checkpoint_graph_encoder",
        "--amp_dtype", "bf16",
        "--cache_spatial_graphs",
        "--save_candidate_qc",
        "--save_outputs",
        "--save_embeddings",
        "--save_ot_prior_topk",
        "--log_cuda_memory",
    ]
    sys.argv[1:1] = defaults


def main() -> None:
    namespace = _load_adapted_pipeline()
    original_read_pair = namespace["read_backed_pair"]

    def read_pair_with_feature_ids(data_dir, sample_dir):
        rna, adt = original_read_pair(data_dir, sample_dir)
        _set_gene_ids(adt, "ADT")
        adt.var["adt_marker"] = adt.var["adt_name_original"].astype(str).to_numpy()
        return rna, adt

    namespace["read_backed_pair"] = read_pair_with_feature_ids
    namespace["prepare_rna_var_names_make_unique"] = _prepare_rna_gene_ids
    namespace["filter_globally_nonzero_genes"] = _filter_globally_nonzero_genes
    _inject_defaults()
    args = namespace["parse_args"]()
    namespace["run_crc_pipeline"](args)


if __name__ == "__main__":
    main()
