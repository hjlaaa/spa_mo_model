"""Paired RNA/ADT reading and feature identity adaptations.

Keep input feature and spot order; preprocessing and training stay with callers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


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


def set_gene_ids(adata, modality: str) -> tuple[pd.Index, pd.Index]:
    if "gene_ids" not in adata.var:
        raise KeyError(f"{modality} is missing required var['gene_ids'].")
    original = pd.Index(adata.var_names.astype(str))
    feature_ids = pd.Index(adata.var["gene_ids"].astype(str))
    if feature_ids.hasnans or not feature_ids.is_unique:
        raise ValueError(f"{modality} var['gene_ids'] must be complete and unique.")
    adata.var[f"{modality.lower()}_name_original"] = original.to_numpy()
    adata.var_names = feature_ids
    return original, feature_ids


def prepare_rna_gene_ids(rna):
    symbols, feature_ids = set_gene_ids(rna, "RNA")
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


def filter_globally_nonzero_genes(rna_a, rna_d, feature_ids: list[str]) -> list[str]:
    total = np.zeros(len(feature_ids), dtype=np.float64)
    for adata in (rna_a, rna_d):
        matrix = _to_memory_matrix(adata, feature_ids)
        total += np.asarray(matrix.sum(axis=0)).ravel()
    kept = [feature for feature, value in zip(feature_ids, total) if value > 0]
    print(f"Removed {len(feature_ids) - len(kept)} globally all-zero RNA genes.")
    return kept


def prepare_spleen_adt(adt) -> None:
    markers = pd.Index(adt.var_names.astype(str))
    if markers.hasnans or not markers.is_unique:
        raise ValueError("Mouse Spleen ADT marker names must be complete and unique.")
    adt.var["adt_name_original"] = markers.to_numpy()
    adt.var["adt_marker"] = markers.to_numpy()


def read_lymph_pair(data_dir: Path, sample_dir: str):
    rna, adt = read_backed_pair(data_dir, sample_dir)
    set_gene_ids(adt, "ADT")
    adt.var["adt_marker"] = adt.var["adt_name_original"].astype(str).to_numpy()
    return rna, adt


def read_spleen_pair(data_dir: Path, sample_dir: str):
    rna, adt = read_backed_pair(data_dir, sample_dir)
    prepare_spleen_adt(adt)
    return rna, adt
