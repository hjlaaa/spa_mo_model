"""Pure paired-data helpers shared by MISAR runners and analysis readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import pandas as pd
import numpy as np
from .paired import spatial_range


SECTION_INFO = {
    "dataset1": {"dir": "dataset1", "sample": "E18_5-S1", "stage": "E18.5"},
    "dataset2": {"dir": "dataset2", "sample": "E15_5-S1", "stage": "E15.5"},
    "dataset3": {"dir": "dataset3", "sample": "E13_5-S1", "stage": "E13.5"},
    "dataset4": {"dir": "dataset4", "sample": "E11_0-S1", "stage": "E11.0"},
}

def read_backed_pair(
    data_dir: Path, section: str, section_info: Mapping[str, Mapping[str, str]]
) -> tuple[ad.AnnData, ad.AnnData]:
    section_dir = data_dir / section_info[section]["dir"]
    rna_path = section_dir / "adata_RNA.h5ad"
    atac_path = section_dir / "adata_ATAC.h5ad"
    if not rna_path.exists():
        raise FileNotFoundError(rna_path)
    if not atac_path.exists():
        raise FileNotFoundError(atac_path)
    return ad.read_h5ad(rna_path, backed="r"), ad.read_h5ad(atac_path, backed="r")


def make_unique_rna_var_names(section: str, rna: ad.AnnData) -> tuple[dict[str, Any], pd.DataFrame]:
    before_unique = bool(rna.var_names.is_unique)
    original = pd.Index(rna.var_names.astype(str))
    counts = pd.Series(original).value_counts()
    duplicated = original.duplicated(keep=False)
    rna.var["gene_symbol_original"] = original.to_numpy()
    rna.var["was_duplicate_gene_symbol"] = duplicated
    rna.var["gene_symbol_original_count"] = [int(counts[symbol]) for symbol in original]
    rna.var_names = pd.Index(original)
    rna.var_names_make_unique()
    if not rna.var_names.is_unique:
        raise ValueError(f"{section}: RNA var_names are still not unique after var_names_make_unique().")
    rna.var["gene_symbol_make_unique"] = rna.var_names.astype(str)
    table = pd.DataFrame(
        {
            "gene_symbol_original": original.to_numpy(),
            "gene_symbol_make_unique": rna.var_names.astype(str).to_numpy(),
            "was_duplicate_gene_symbol": duplicated,
            "gene_symbol_original_count": [int(counts[symbol]) for symbol in original],
        }
    )
    duplicate_summary = (
        table[table["was_duplicate_gene_symbol"]]
        .groupby("gene_symbol_original", sort=True)
        .agg(
            count=("gene_symbol_make_unique", "size"),
            make_unique_names=("gene_symbol_make_unique", lambda values: ";".join(values)),
        )
        .reset_index()
    )
    return (
        {
            "shape": list(rna.shape),
            "var_names_unique_before": before_unique,
            "var_names_unique_after": bool(rna.var_names.is_unique),
            "duplicate_gene_symbol_groups": int(duplicate_summary.shape[0]),
            "duplicate_extra_columns": int((duplicate_summary["count"] - 1).sum())
            if not duplicate_summary.empty
            else 0,
        },
        duplicate_summary,
    )


def common_var_names(
    adatas: Mapping[str, ad.AnnData],
    section_order: list[str],
    max_features: int | None,
    label: str,
) -> list[str]:
    common = pd.Index(adatas[section_order[0]].var_names.astype(str))
    for section in section_order[1:]:
        common = common.intersection(pd.Index(adatas[section].var_names.astype(str)))
    if common.empty:
        raise ValueError(f"No shared {label} features found across MISAR sections.")
    selected = list(common[:max_features]) if max_features is not None else list(common)
    if not selected:
        raise ValueError(f"No {label} features selected.")
    return selected


LABEL_COLUMNS = [
    "Sample",
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
]

def validate_rna_atac_alignment(
    section: str, rna: ad.AnnData, atac: ad.AnnData, secondary_modality: str = "ATAC"
) -> dict[str, Any]:
    if list(rna.obs_names) != list(atac.obs_names):
        raise ValueError(f"{section}: RNA and {secondary_modality} obs_names are not identical.")
    if "spatial" not in rna.obsm:
        raise KeyError(f"{section}: RNA is missing obsm['spatial'].")
    if "spatial" not in atac.obsm:
        raise KeyError(f"{section}: {secondary_modality} is missing obsm['spatial'].")
    rna_spatial = np.asarray(rna.obsm["spatial"])
    atac_spatial = np.asarray(atac.obsm["spatial"])
    if rna_spatial.shape != atac_spatial.shape:
        raise ValueError(f"{section}: RNA and {secondary_modality} spatial shapes differ.")
    spatial_match = bool(np.allclose(rna_spatial, atac_spatial))
    if not spatial_match:
        raise ValueError(f"{section}: RNA and {secondary_modality} spatial coordinates differ.")
    label_summary = {}
    for column in LABEL_COLUMNS:
        if column == "ATAC_Clusters":
            column = f"{secondary_modality}_Clusters"
        if column in rna.obs:
            label_summary[column] = {
                "n_unique": int(rna.obs[column].astype(str).nunique(dropna=False)),
                "top_counts": {
                    str(key): int(value)
                    for key, value in rna.obs[column].astype(str).value_counts(dropna=False).head(20).items()
                },
            }
    return {
        "spot_count": int(rna.n_obs),
        "obs_names_match": True,
        "spatial_shape": list(rna_spatial.shape),
        "spatial_match": spatial_match,
        "spatial_range": spatial_range(rna_spatial),
        "labels": label_summary,
    }



def rename_section_keys(mapping: Mapping[str, Any], section_order: list[str]) -> dict[str, Any]:
    key_map = {f"s{idx + 1}": section for idx, section in enumerate(section_order)}
    return {key_map.get(section, section): value for section, value in mapping.items()}
