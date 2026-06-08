"""Unified COSIE-style multimodal preprocessing entry points."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from .configure import get_default_preprocess_config
from .data_preprocessing import load_cosie_style_data
from .image_preprocessing import (
    build_he_adata_from_feature_file,
    build_he_adata_from_image_and_mask,
    build_he_adata_from_uni_feature,
)
from .utils import (
    check_obs_names_consistency,
    check_spatial_consistency,
    get_embedding_shape,
    is_missing_input,
    load_h5ad_if_needed,
)

COSIE_DATA_MODALITIES = ("HE", "RNA", "Protein", "Metabolite")


def _config_value(config: dict[str, Any] | None, key: str, default=None):
    if not config:
        return default
    if key in config:
        return config[key]
    preprocessing = config.get("preprocessing", {})
    if key in preprocessing:
        return preprocessing[key]
    he_image = config.get("he_image", {})
    if key in he_image:
        return he_image[key]
    paths = config.get("paths", {})
    if key in paths:
        return paths[key]
    return default


def _ensure_cosie_spatial_key(adata, spatial_key="spatial"):
    """Adapter: copy configured spatial key to COSIE's required 'spatial' key."""

    if adata is None:
        return None
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData object is missing obsm['{spatial_key}'].")
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm[spatial_key]).copy()
    elif spatial_key != "spatial" and not np.array_equal(adata.obsm["spatial"], adata.obsm[spatial_key]):
        raise ValueError(
            "AnnData contains both obsm['spatial'] and the configured "
            f"obsm['{spatial_key}'], but they differ."
        )
    return adata


def _first_spatial_from_modalities(modalities):
    for adata in modalities.values():
        if adata is not None and "spatial" in adata.obsm:
            return np.asarray(adata.obsm["spatial"]).copy()
    return None


def build_mousebrain_section(
    section_id: str,
    rna_path: str,
    metabolite_path: str,
    spatial_key: str = "spatial",
    uni_feature_key: str = "uni_feature",
    rna_gene_id_key: str = "gene_ids",
):
    """Build a MouseBrain HE+RNA+Metabolite section without changing spot order."""

    import scanpy as sc

    rna = sc.read_h5ad(rna_path)
    meta = sc.read_h5ad(metabolite_path)

    original_obs_names = rna.obs_names.copy()
    original_n_obs = rna.n_obs

    if rna_gene_id_key not in rna.var:
        raise KeyError(
            f"{section_id}: RNA var['{rna_gene_id_key}'] is required for unique gene IDs."
        )
    if rna.var[rna_gene_id_key].isna().any():
        raise ValueError(f"{section_id}: RNA var['{rna_gene_id_key}'] contains missing values.")
    gene_ids = rna.var[rna_gene_id_key].astype(str)
    if not gene_ids.is_unique:
        duplicated = gene_ids[gene_ids.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"{section_id}: RNA var['{rna_gene_id_key}'] must be unique; "
            f"examples of duplicated IDs: {duplicated}"
        )
    rna.var["gene_symbol"] = rna.var_names.astype(str)
    rna.var_names = gene_ids
    rna.var_names_make_unique()

    if list(rna.obs_names) != list(original_obs_names) or rna.n_obs != original_n_obs:
        raise RuntimeError(f"{section_id}: RNA gene ID adaptation changed spot order/count.")

    if spatial_key not in rna.obsm:
        raise KeyError(f"{section_id}: RNA is missing obsm['{spatial_key}'].")
    if spatial_key not in meta.obsm:
        raise KeyError(f"{section_id}: Metabolite is missing obsm['{spatial_key}'].")
    if uni_feature_key not in rna.obsm:
        raise KeyError(f"{section_id}: RNA is missing obsm['{uni_feature_key}'] for HE features.")

    he = ad.AnnData(X=np.asarray(rna.obsm[uni_feature_key]).copy())
    he.obs_names = rna.obs_names.copy()
    he.obsm["spatial"] = np.asarray(rna.obsm[spatial_key]).copy()
    if spatial_key != "spatial":
        he.obsm[spatial_key] = he.obsm["spatial"].copy()

    if rna.n_obs != meta.n_obs or rna.n_obs != he.n_obs:
        raise ValueError(
            f"{section_id}: spot count mismatch: "
            f"RNA={rna.n_obs}, Metabolite={meta.n_obs}, HE={he.n_obs}."
        )
    if list(rna.obs_names) != list(meta.obs_names):
        raise ValueError(f"{section_id}: RNA and Metabolite obs_names are not identical.")
    if not np.array_equal(np.asarray(rna.obsm[spatial_key]), np.asarray(meta.obsm[spatial_key])):
        raise ValueError(f"{section_id}: RNA and Metabolite spatial coordinates differ.")
    if not np.array_equal(np.asarray(rna.obsm[spatial_key]), np.asarray(he.obsm["spatial"])):
        raise ValueError(f"{section_id}: RNA and HE spatial coordinates differ.")

    return {
        "section_id": section_id,
        "modalities": {
            "HE": he,
            "RNA": _ensure_cosie_spatial_key(rna, spatial_key=spatial_key),
            "Protein": None,
            "Metabolite": _ensure_cosie_spatial_key(meta, spatial_key=spatial_key),
        },
        "modality_present": {
            "HE": True,
            "RNA": True,
            "Protein": False,
            "Metabolite": True,
        },
        "messages": [
            f"{section_id}: HE built from obsm['{uni_feature_key}'] with shape {he.shape}",
            f"{section_id}: RNA var_names set from var['{rna_gene_id_key}']",
            f"{section_id}: Metabolite loaded with shape {meta.shape}",
        ],
    }


def build_section_modalities(
    section_id,
    he_input=None,
    he_mask_input=None,
    he_feature_input=None,
    he_reference_adata_input=None,
    rna_input=None,
    protein_input=None,
    metabolite_input=None,
    spatial_key="spatial",
    uni_feature_key="UNI_feature",
    config=None,
):
    """
    Build one section's COSIE-style modality dictionary.

    ADT/protein inputs are always stored internally under COSIE's ``Protein``
    modality name.
    """

    cfg = get_default_preprocess_config()
    if config:
        cfg.update({k: v for k, v in config.items() if k not in {"paths", "preprocessing", "he_image"}})
        for section in ("paths", "preprocessing", "he_image"):
            if section in config:
                cfg[section].update(config[section])

    dry_run = bool(_config_value(cfg, "dry_run", False))
    uni_dir = _config_value(cfg, "uni_dir", _config_value(cfg, "uni_dir", None))
    if uni_dir is None:
        uni_dir = cfg["paths"]["uni_dir"]

    messages = []
    modalities = {modality: None for modality in COSIE_DATA_MODALITIES}

    rna_adata = load_h5ad_if_needed(rna_input)
    protein_adata = load_h5ad_if_needed(protein_input)
    metabolite_adata = load_h5ad_if_needed(metabolite_input)

    if rna_adata is not None:
        modalities["RNA"] = _ensure_cosie_spatial_key(rna_adata, spatial_key=spatial_key)
        messages.append(f"{section_id}: loaded RNA with shape {modalities['RNA'].shape}")
    else:
        messages.append(f"{section_id}: RNA missing")

    if protein_adata is not None:
        modalities["Protein"] = _ensure_cosie_spatial_key(protein_adata, spatial_key=spatial_key)
        messages.append(f"{section_id}: loaded Protein/ADT with shape {modalities['Protein'].shape}")
    else:
        messages.append(f"{section_id}: Protein/ADT missing")

    if metabolite_adata is not None:
        modalities["Metabolite"] = _ensure_cosie_spatial_key(
            metabolite_adata,
            spatial_key=spatial_key,
        )
        messages.append(f"{section_id}: loaded Metabolite with shape {modalities['Metabolite'].shape}")
    else:
        messages.append(f"{section_id}: Metabolite missing")

    he_adata = None
    if not is_missing_input(he_reference_adata_input):
        reference_adata = load_h5ad_if_needed(he_reference_adata_input)
        if uni_feature_key in reference_adata.obsm:
            he_adata = build_he_adata_from_uni_feature(
                reference_adata,
                uni_feature_key=uni_feature_key,
                spatial_key=spatial_key,
            )
            messages.append(
                f"{section_id}: built HE from reference obsm['{uni_feature_key}'] "
                f"with shape {he_adata.shape}"
            )
        elif is_missing_input(he_feature_input):
            messages.append(
                f"{section_id}: HE reference was provided but obsm['{uni_feature_key}'] "
                "was not found"
            )

    if he_adata is None and not is_missing_input(he_feature_input):
        spatial = None
        if not is_missing_input(he_reference_adata_input):
            reference_adata = load_h5ad_if_needed(he_reference_adata_input)
            if spatial_key in reference_adata.obsm:
                spatial = reference_adata.obsm[spatial_key]
        if spatial is None:
            spatial = _first_spatial_from_modalities(modalities)
        he_adata = build_he_adata_from_feature_file(
            he_feature_input,
            spatial=spatial,
            spatial_key=spatial_key,
        )
        messages.append(f"{section_id}: built HE from feature file with shape {he_adata.shape}")

    if he_adata is None and not is_missing_input(he_input):
        if is_missing_input(he_mask_input):
            raise ValueError("he_mask_input is required when he_input is a raw HE image.")
        if dry_run:
            messages.append(
                f"{section_id}: HE image+mask provided; skipped UNI extraction in dry_run"
            )
        else:
            he_adata = build_he_adata_from_image_and_mask(
                he_input,
                he_mask_input,
                uni_dir=uni_dir,
                device=_config_value(cfg, "device", None),
                batch_size=_config_value(cfg, "batch_size", 128),
                num_workers=_config_value(cfg, "num_workers", 4),
                output_cache_path=_config_value(cfg, "output_cache_path", None),
                spatial_key=spatial_key,
            )
            messages.append(f"{section_id}: built HE from image+mask with shape {he_adata.shape}")

    if he_adata is not None:
        modalities["HE"] = _ensure_cosie_spatial_key(he_adata, spatial_key="spatial")
    else:
        messages.append(f"{section_id}: HE missing")

    present_adatas = [adata for adata in modalities.values() if adata is not None]
    if len(present_adatas) == 0:
        raise ValueError(f"Section {section_id} has no non-missing modalities.")

    check_obs_names_consistency(present_adatas)
    check_spatial_consistency(present_adatas, spatial_key="spatial", raise_on_mismatch=True)

    modality_present = {
        modality: modalities[modality] is not None
        for modality in COSIE_DATA_MODALITIES
    }

    return {
        "section_id": section_id,
        "modalities": modalities,
        "modality_present": modality_present,
        "messages": messages,
    }


def build_cosie_data_dict(section_results):
    """Build COSIE ``data_dict`` while preserving section order."""

    data_dict = {modality: [] for modality in COSIE_DATA_MODALITIES}
    for section_result in section_results:
        modalities = section_result["modalities"]
        for modality in COSIE_DATA_MODALITIES:
            data_dict[modality].append(modalities.get(modality))
    return data_dict


def preprocess_multisection_cosie_style(
    sections,
    n_comps=50,
    hvg_num=3000,
    hvg_num_by_modality=None,
    target_sum=None,
    use_harmony=True,
    metacell=False,
    config=None,
    dry_run=False,
):
    """
    Build COSIE-style data structures and optionally run COSIE preprocessing.

    Returns ``data_dict``, ``feature_dict``, ``spatial_loc_dict`` and
    ``processed_data_dict``. With ``dry_run=True``, only input loading and
    data_dict construction are performed.
    """

    cfg = get_default_preprocess_config()
    if config:
        for key, value in config.items():
            if key in {"paths", "preprocessing", "he_image"} and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    cfg["dry_run"] = dry_run

    spatial_key = _config_value(cfg, "spatial_key", "spatial")
    uni_feature_key = _config_value(cfg, "uni_feature_key", "UNI_feature")
    if hvg_num_by_modality is None:
        hvg_num_by_modality = _config_value(cfg, "hvg_num_by_modality", None)

    section_results = []
    messages = []
    section_ids = []
    for idx, section in enumerate(sections):
        section_id = section.get("section_id") or section.get("stage_id") or f"s{idx + 1}"
        section_ids.append(section_id)
        section_result = build_section_modalities(
            section_id=section_id,
            he_input=section.get("he_input"),
            he_mask_input=section.get("he_mask_input"),
            he_feature_input=section.get("he_feature_input"),
            he_reference_adata_input=section.get("he_reference_adata_input"),
            rna_input=section.get("rna_input"),
            protein_input=section.get("protein_input"),
            metabolite_input=section.get("metabolite_input"),
            spatial_key=section.get("spatial_key", spatial_key),
            uni_feature_key=section.get("uni_feature_key", uni_feature_key),
            config=cfg,
        )
        section_results.append(section_result)
        messages.extend(section_result["messages"])

    data_dict = build_cosie_data_dict(section_results)

    modality_present = {
        section_result["section_id"]: section_result["modality_present"]
        for section_result in section_results
    }

    if dry_run:
        feature_dict = {}
        spatial_loc_dict = {}
        processed_data_dict = None
        messages.append("dry_run=True: skipped load_cosie_style_data()")
    else:
        feature_dict, spatial_loc_dict, processed_data_dict = load_cosie_style_data(
            data_dict,
            n_comps=n_comps,
            hvg_num=hvg_num,
            hvg_num_by_modality=hvg_num_by_modality,
            target_sum=target_sum,
            use_harmony=use_harmony,
            metacell=metacell,
        )

    return {
        "data_dict": data_dict,
        "feature_dict": feature_dict,
        "spatial_loc_dict": spatial_loc_dict,
        "processed_data_dict": processed_data_dict,
        "modality_present": modality_present,
        "section_ids": section_ids,
        "section_results": section_results,
        "messages": messages,
    }


def summarize_data_dict(data_dict, spatial_key="spatial"):
    """Return JSON-friendly shape summaries for a COSIE data_dict."""

    summary = {}
    for modality, adata_list in data_dict.items():
        summary[modality] = []
        for adata in adata_list:
            if adata is None:
                summary[modality].append(None)
            else:
                summary[modality].append(
                    {
                        "shape": list(adata.shape),
                        "spatial_shape": (
                            list(adata.obsm[spatial_key].shape)
                            if spatial_key in adata.obsm
                            else None
                        ),
                        "obsm_keys": list(adata.obsm.keys()),
                    }
                )
    return summary


def summarize_feature_dict(feature_dict):
    """Return JSON-friendly shape summaries for COSIE feature_dict."""

    summary = {}
    for section, modalities in feature_dict.items():
        summary[section] = {}
        for modality, tensor in modalities.items():
            summary[section][modality] = list(get_embedding_shape(tensor))
    return summary


def summarize_spatial_loc_dict(spatial_loc_dict):
    """Return JSON-friendly shape summaries for spatial_loc_dict."""

    return {
        section: list(spatial.shape) if hasattr(spatial, "shape") else None
        for section, spatial in spatial_loc_dict.items()
    }
