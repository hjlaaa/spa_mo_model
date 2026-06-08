"""COSIE-style tabular modality preprocessing.

This module migrates the preprocessing functions used by COSIE without making
the new project depend on ``/home/hujinlan/cosie`` at runtime.
"""

from __future__ import annotations

import numpy as np
import scipy
import scanpy as sc
import anndata as ad
import torch


def canonicalize_modality(modality: str) -> str:
    """Map user-facing aliases to COSIE internal modality names."""

    aliases = {
        "ADT": "Protein",
        "adt": "Protein",
        "protein": "Protein",
        "Proteomics": "Protein",
        "proteomics": "Protein",
        "metabolite": "Metabolite",
        "Metabolomics": "Metabolite",
        "metabolomics": "Metabolite",
        "H&E": "HE",
        "h&e": "HE",
        "he": "HE",
        "rna": "RNA",
    }
    return aliases.get(modality, modality)


def _resolve_hvg_num_for_modality(hvg_num, hvg_num_by_modality, modality):
    """Return a modality-specific HVG count while preserving global defaults."""

    if hvg_num_by_modality is None:
        return hvg_num
    canonical = canonicalize_modality(modality)
    if canonical in hvg_num_by_modality:
        return hvg_num_by_modality[canonical]
    if modality in hvg_num_by_modality:
        return hvg_num_by_modality[modality]
    return hvg_num


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::preprocess_adata
def preprocess_adata(adata_raw, modality, hvg_num=3000, n_comps=50, target_sum=None):
    """
    Preprocess an AnnData object using COSIE's modality-specific rules.

    COSIE behavior retained:
    - HE: PCA directly on existing image embeddings.
    - RNA/RNA_panel2/Metabolite and other non-Protein omics: optional HVG,
      normalize_total, log1p, scale, PCA.
    - Protein: CLR per cell, scale, PCA with COSIE's protein component rule.
    """

    modality = canonicalize_modality(modality)
    adata_obj = adata_raw.copy()
    adata_obj.var_names_make_unique()

    if modality == "HE":
        sc.tl.pca(adata_obj, n_comps=n_comps)
    else:
        if hvg_num and len(adata_obj.var_names) > hvg_num:
            if modality in {
                "RNA",
                "RNA_panel2",
                "H3K27me3",
                "H3K27ac",
                "ATAC",
                "H3K4me3",
                "Metabolite",
            }:
                use_batch = "batch" in adata_obj.obs
                if modality in ["RNA", "RNA_panel2"]:
                    sc.pp.highly_variable_genes(
                        adata_obj,
                        n_top_genes=hvg_num,
                        flavor="seurat_v3",
                        batch_key="batch" if use_batch else None,
                    )
                else:
                    sc.pp.highly_variable_genes(
                        adata_obj,
                        n_top_genes=hvg_num,
                        batch_key="batch" if use_batch else None,
                    )
                adata_obj = adata_obj[:, adata_obj.var["highly_variable"]]

        if modality == "Protein":
            adata_obj = clr_normalize_each_cell(adata_obj)
            sc.pp.scale(adata_obj)
            n_proteins = adata_obj.shape[1]

            if n_proteins >= n_comps:
                sc.tl.pca(adata_obj, n_comps=n_comps)
            elif n_proteins >= 20:
                sc.tl.pca(adata_obj, n_comps=20)
            else:
                sc.tl.pca(adata_obj, n_comps=15)
        else:
            if target_sum:
                sc.pp.normalize_total(adata_obj, target_sum=target_sum)
            else:
                sc.pp.normalize_total(adata_obj)
            sc.pp.log1p(adata_obj)
            sc.pp.scale(adata_obj)
            sc.tl.pca(adata_obj, n_comps=n_comps)

    return adata_obj


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::load_data
def load_data(
    data_dict,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
    use_harmony=True,
    metacell=False,
    hvg_num_by_modality=None,
):
    """
    Process COSIE-style ``data_dict`` into model-ready feature tensors.

    Input format:
    ``{modality: [AnnData_or_None_for_s1, AnnData_or_None_for_s2, ...]}``
    with spatial coordinates stored in ``.obsm['spatial']``.
    """

    data_dict = {
        canonicalize_modality(modality): sections
        for modality, sections in data_dict.items()
    }

    if metacell:
        print("Combine adjacent 4 cells into metacell to save memory and speed up computation")
        data_dict = construct_metacell_data_dict(data_dict)

    feature_dict = {}
    spatial_loc_dict = {}
    num_sections = max(len(sections) for sections in data_dict.values())

    shared_modalities = {
        modality: [adata_obj for adata_obj in sections if adata_obj is not None]
        for modality, sections in data_dict.items()
        if sum(x is not None for x in sections) > 1
    }

    shared_modality_sections = {
        modality: [idx for idx, adata_obj in enumerate(data_dict[modality]) if adata_obj is not None]
        for modality in shared_modalities
    }

    for modality, adata_list in shared_modalities.items():
        print(f"-------- Processing shared modality {modality} across sections --------")

        if modality == "HE":
            adata_sub_list = []
            for i, adata_obj in enumerate(adata_list):
                adata_sub = adata_obj.copy()
                adata_sub.obs_names = (
                    adata_sub.obs_names + f"_{shared_modality_sections[modality][i]}"
                )
                adata_sub_list.append(adata_sub)
        else:
            common_var_names = adata_list[0].var_names
            for adata_obj in adata_list[1:]:
                common_var_names = common_var_names.intersection(adata_obj.var_names)

            adata_sub_list = []
            for i, adata_obj in enumerate(adata_list):
                adata_sub = adata_obj[:, common_var_names].copy()
                adata_sub.obs_names = (
                    adata_sub.obs_names + f"_{shared_modality_sections[modality][i]}"
                )
                adata_sub_list.append(adata_sub)

        adata_combined = ad.concat(adata_sub_list)
        adata_combined.obs["batch"] = [
            f"batch_{shared_modality_sections[modality][i]}"
            for i, adata_obj in enumerate(adata_list)
            for _ in range(adata_obj.shape[0])
        ]

        adata_combined = preprocess_adata(
            adata_combined,
            modality,
            hvg_num=_resolve_hvg_num_for_modality(
                hvg_num,
                hvg_num_by_modality,
                modality,
            ),
            n_comps=n_comps,
        )
        if use_harmony:
            print(f"Running Harmony for {modality}")
            sc.external.pp.harmony_integrate(adata_combined, key="batch")
            pca_data_combined = adata_combined.obsm["X_pca_harmony"]
        else:
            pca_data_combined = adata_combined.obsm["X_pca"]

        split_indices = np.cumsum([adata_obj.shape[0] for adata_obj in adata_list])[:-1]
        combined_data_splits = np.split(pca_data_combined, split_indices)

        print(shared_modality_sections)
        for i, section in enumerate(shared_modality_sections[modality]):
            print(i, section)
            key_name = f"{modality}_harmony" if use_harmony else f"{modality}_pca"
            data_dict[modality][section].obsm[key_name] = combined_data_splits[i]
            if section not in feature_dict:
                feature_dict[section] = {}

            print(feature_dict.keys())
            shared_data = combined_data_splits[i].copy()
            feature_dict[section][modality] = torch.from_numpy(shared_data).float()
            del shared_data
    print(feature_dict.keys())

    for modality, sections in data_dict.items():
        if modality in shared_modalities:
            continue

        for section, adata_obj in enumerate(sections):
            if adata_obj is not None:
                print(f"-------- Processing unique modality {modality} for section {section + 1} --------")
                if section not in feature_dict:
                    feature_dict[section] = {}
                adata_processed = preprocess_adata(
                    adata_obj,
                    modality,
                    hvg_num=_resolve_hvg_num_for_modality(
                        hvg_num,
                        hvg_num_by_modality,
                        modality,
                    ),
                    n_comps=n_comps,
                    target_sum=target_sum,
                )
                pca_data = adata_processed.obsm["X_pca"].copy()
                data_dict[modality][section].obsm[f"{modality}_pca"] = pca_data
                feature_dict[section][modality] = torch.from_numpy(pca_data).float()
                del pca_data
    print(feature_dict.keys())
    feature_dict = {f"s{int(k) + 1}": v for k, v in feature_dict.items()}
    print(feature_dict.keys())

    for section_idx in range(num_sections):
        print(f"Extracting spatial location for section {section_idx + 1}")
        spatial_list = []
        for modality, sections in data_dict.items():
            if (
                section_idx < len(sections)
                and sections[section_idx] is not None
                and "spatial" in sections[section_idx].obsm
            ):
                spatial_list.append(sections[section_idx].obsm["spatial"])

        if len(spatial_list) == 1:
            spatial_loc_dict[f"s{section_idx + 1}"] = spatial_list[0]
        elif len(spatial_list) > 1:
            if all(np.array_equal(spatial_list[0], spatial) for spatial in spatial_list[1:]):
                spatial_loc_dict[f"s{section_idx + 1}"] = spatial_list[0]
            else:
                raise ValueError(
                    f"Section {section_idx + 1} contains inconsistent spatial information "
                    "across different modalities!"
                )

    return feature_dict, spatial_loc_dict, data_dict


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::clr_normalize_each_cell
def clr_normalize_each_cell(adata_obj, inplace=True):
    """Normalize each cell's protein counts using COSIE's CLR implementation."""

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata_obj = adata_obj.copy()

    adata_obj.X = np.apply_along_axis(
        seurat_clr,
        1,
        (adata_obj.X.A if scipy.sparse.issparse(adata_obj.X) else np.array(adata_obj.X)),
    )
    return adata_obj


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::metacell_construction_optimized
def metacell_construction_optimized(adata_obj):
    """Aggregate every 2x2 spatially adjacent grid of cells into one metacell."""

    import scipy.sparse as sp

    spatial = adata_obj.obsm["spatial"]
    expr = adata_obj.X
    if sp.issparse(expr):
        expr = expr.toarray()

    y = spatial[:, 0]
    x = spatial[:, 1]
    y0, x0 = y.min(), x.min()
    uniq_y = np.unique(y)
    uniq_x = np.unique(x)
    dy = np.diff(uniq_y)
    dx = np.diff(uniq_x)
    step_y = np.min(dy[dy > 0])
    step_x = np.min(dx[dx > 0])

    grid_y = np.round((y - y0) / step_y).astype(int)
    grid_x = np.round((x - x0) / step_x).astype(int)

    block_y = grid_y // 2
    block_x = grid_x // 2

    from collections import defaultdict

    blocks = defaultdict(list)
    for idx, (by, bx) in enumerate(zip(block_y, block_x)):
        blocks[(by, bx)].append(idx)

    meta_expr = []
    meta_coords = []
    meta_to_original = []
    for (by, bx), indices in blocks.items():
        meta_to_original.append(indices)
        meta_expr.append(expr[indices].mean(axis=0))
        meta_coords.append(spatial[indices].mean(axis=0))

    meta_X = np.vstack(meta_expr)
    adata_meta = sc.AnnData(X=meta_X)
    adata_meta.var_names = adata_obj.var_names.copy()
    adata_meta.obsm["spatial"] = np.vstack(meta_coords)
    adata_meta.uns["meta_to_original"] = meta_to_original
    adata_meta.uns["original_cell_num"] = adata_obj.n_obs

    return adata_meta


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::construct_metacell_data_dict
def construct_metacell_data_dict(data_dict):
    """Apply COSIE metacell construction to every non-missing AnnData object."""

    metacell_dict = {}

    for modality, adata_list in data_dict.items():
        new_list = []
        for adata_obj in adata_list:
            if adata_obj is None:
                new_list.append(None)
            else:
                new_adata = metacell_construction_optimized(adata_obj)
                new_list.append(new_adata)
        metacell_dict[modality] = new_list

    return metacell_dict


# Adapted from /home/hujinlan/cosie/COSIE/data_preprocessing.py::reconstruct_metacell_to_original
def reconstruct_metacell_to_original(adata_metacell, metacell_embedding):
    """Expand metacell-level embeddings back to original cells."""

    meta_to_original = adata_metacell.uns["meta_to_original"]
    original_cell_num = adata_metacell.uns["original_cell_num"]

    original_embedding = np.zeros((original_cell_num, metacell_embedding.shape[1]))

    for meta_idx, original_indices in enumerate(meta_to_original):
        original_embedding[original_indices] = metacell_embedding[meta_idx]

    return original_embedding


def preprocess_rna_adata(
    adata_obj,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
):
    """Wrapper for COSIE RNA preprocessing."""

    return preprocess_adata(
        adata_obj,
        modality="RNA",
        hvg_num=hvg_num,
        n_comps=n_comps,
        target_sum=target_sum,
    )


def preprocess_protein_adata(
    adata_obj,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
):
    """Wrapper for COSIE Protein/ADT preprocessing."""

    return preprocess_adata(
        adata_obj,
        modality="Protein",
        hvg_num=hvg_num,
        n_comps=n_comps,
        target_sum=target_sum,
    )


def preprocess_metabolite_adata(
    adata_obj,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
):
    """Wrapper for COSIE's generic Metabolite branch."""

    return preprocess_adata(
        adata_obj,
        modality="Metabolite",
        hvg_num=hvg_num,
        n_comps=n_comps,
        target_sum=target_sum,
    )


def load_cosie_style_data(
    data_dict,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
    use_harmony=True,
    metacell=False,
    hvg_num_by_modality=None,
):
    """Alias for migrated COSIE ``load_data`` with canonical modality names."""

    return load_data(
        data_dict,
        n_comps=n_comps,
        hvg_num=hvg_num,
        target_sum=target_sum,
        use_harmony=use_harmony,
        metacell=metacell,
        hvg_num_by_modality=hvg_num_by_modality,
    )
