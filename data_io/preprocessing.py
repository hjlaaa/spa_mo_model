"""COSIE-style tabular modality preprocessing.

This module migrates the preprocessing functions used by COSIE without making
the new project depend on ``/home/hujinlan/cosie`` at runtime.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from numbers import Real

import numpy as np
import scipy
import scanpy as sc
import anndata as ad
import torch
from sklearn.neighbors import NearestNeighbors


def spatial_enhance_features(feature_dict, spatial_loc_dict, *, k=10, weight=0.2,
                             include_self=False):
    """Replace each modality with its spatial KNN enhanced features per section.

    k counts non-self neighbors. include_self adds a self edge to that sum;
    the original feature is always retained by the residual addition.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("spatial_enhancement.k must be a positive integer")
    if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(weight):
        raise ValueError("spatial_enhancement.weight must be finite")
    if not isinstance(include_self, bool):
        raise ValueError("spatial_enhancement.include_self must be boolean")

    for section, modalities in feature_dict.items():
        if section not in spatial_loc_dict:
            raise ValueError(f"Missing spatial coordinates for section {section}")
        coords = np.asarray(spatial_loc_dict[section], dtype=np.float32)
        if coords.ndim != 2 or 0 in coords.shape or not np.isfinite(coords).all():
            raise ValueError(f"Invalid spatial coordinates for section {section}")
        n_cells = coords.shape[0]
        n_neighbors = min(int(k), n_cells - 1)
        if n_neighbors:
            # X=None excludes the fitted point by row index, including for
            # spots with identical coordinates.
            indices = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(
                X=None, return_distance=False
            )
            rows = np.repeat(np.arange(n_cells), n_neighbors)
            cols = indices.reshape(-1)
            data = np.ones(cols.size, dtype=np.float32)
            graph = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        else:
            graph = scipy.sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
        if include_self:
            graph = graph + scipy.sparse.eye(n_cells, format="csr", dtype=np.float32)

        for modality, feature in modalities.items():
            if feature.ndim != 2 or feature.shape[0] != n_cells:
                raise ValueError(
                    f"{section}/{modality}: feature rows do not match spatial coordinates"
                )
            values = feature.detach().cpu().numpy()
            output = values + weight * (graph @ values)
            modalities[modality] = torch.from_numpy(
                np.ascontiguousarray(output, dtype=np.float32)
            )
    return feature_dict


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
def preprocess_adata(
    adata_raw,
    modality,
    hvg_num=3000,
    n_comps=50,
    target_sum=None,
    copy_input=True,
):
    """
    Preprocess an AnnData object using COSIE's modality-specific rules.

    COSIE behavior retained:
    - HE: PCA directly on existing image embeddings.
    - RNA/RNA_panel2/Metabolite and other non-Protein omics: optional HVG,
      normalize_total, log1p, scale, PCA.
    - Protein: CLR per cell, scale, PCA with COSIE's protein component rule.
    """

    modality = canonicalize_modality(modality)
    adata_obj = adata_raw.copy() if copy_input else adata_raw
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
    hvg_num_by_modality=None,
    memory_efficient=False,
    retain_processed=True,
    spatial_enhancement=None,
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

        if memory_efficient:
            common_var_names = adata_list[0].var_names
            for adata_obj in adata_list[1:]:
                common_var_names = common_var_names.intersection(adata_obj.var_names)
            concat_sources = [
                adata_obj
                if adata_obj.n_vars == len(common_var_names)
                and adata_obj.var_names.equals(common_var_names)
                else adata_obj[:, common_var_names]
                for adata_obj in adata_list
            ]
            section_keys = [
                str(section) for section in shared_modality_sections[modality]
            ]
            adata_combined = ad.concat(
                concat_sources,
                keys=section_keys,
                index_unique="_",
            )
        elif modality == "HE":
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

        if not memory_efficient:
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
            target_sum=target_sum,
            copy_input=not memory_efficient,
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
            if retain_processed:
                data_dict[modality][section].obsm[key_name] = combined_data_splits[i]
            if section not in feature_dict:
                feature_dict[section] = {}

            print(feature_dict.keys())
            shared_data = np.ascontiguousarray(
                combined_data_splits[i], dtype=np.float32
            )
            feature_dict[section][modality] = torch.from_numpy(shared_data)
            del shared_data
        if memory_efficient:
            del combined_data_splits, pca_data_combined, adata_combined
            gc.collect()
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
                if retain_processed:
                    data_dict[modality][section].obsm[f"{modality}_pca"] = pca_data
                feature_dict[section][modality] = torch.from_numpy(
                    np.ascontiguousarray(pca_data, dtype=np.float32)
                )
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

    if spatial_enhancement is not None:
        if not isinstance(spatial_enhancement, Mapping):
            raise ValueError("spatial_enhancement must be a mapping")
        unknown = set(spatial_enhancement) - {"enabled", "k", "weight", "include_self"}
        if unknown:
            raise ValueError(f"Unknown spatial_enhancement options: {sorted(unknown)}")
        enabled = spatial_enhancement.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("spatial_enhancement.enabled must be boolean")
        if enabled:
            feature_dict = spatial_enhance_features(
                feature_dict,
                spatial_loc_dict,
                k=spatial_enhancement.get("k", 10),
                weight=spatial_enhancement.get("weight", 0.2),
                include_self=spatial_enhancement.get("include_self", False),
            )

    return feature_dict, spatial_loc_dict, data_dict if retain_processed else None


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


def load_cosie_style_data(
    data_dict,
    n_comps=50,
    hvg_num=3000,
    target_sum=None,
    use_harmony=True,
    hvg_num_by_modality=None,
    memory_efficient=False,
    retain_processed=True,
    spatial_enhancement=None,
):
    """Alias for migrated COSIE ``load_data`` with canonical modality names."""

    return load_data(
        data_dict,
        n_comps=n_comps,
        hvg_num=hvg_num,
        target_sum=target_sum,
        use_harmony=use_harmony,
        hvg_num_by_modality=hvg_num_by_modality,
        memory_efficient=memory_efficient,
        retain_processed=retain_processed,
        spatial_enhancement=spatial_enhancement,
    )
