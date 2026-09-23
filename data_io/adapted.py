"""Existing Thymus and Simulation pair readers shared by training and analysis.

Only file reading and unchanged AnnData adaptation live here. No training,
preprocessing orchestration, default configuration, or audit output writing.
"""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

THYMUS_SECTIONS = [f"Mouse_Thymus{i}" for i in range(1, 5)]
ADT_ORDER = [
    "CD5",
    "CD68",
    "CD4",
    "CD29",
    "CD8a",
    "CD3",
    "CD44",
    "CD90_2",
    "CD11c",
    "CD31",
    "F4_80",
    "CD45R_B220",
    "Rat_IgG2a",
    "CD11b",
    "CD19",
    "Mouse_IgG2a",
    "CD169",
]

ADT_RAW = {
    "Mouse_Thymus1": {
        "CD5": "Mouse-CD5",
        "CD68": "Mouse-CD68",
        "CD4": "Mouse-CD4",
        "CD29": "Mouse-CD29",
        "CD8a": "Mouse-CD8a",
        "CD3": "Mouse-CD3",
        "CD44": "Mouse-CD44",
        "CD90_2": "Mouse-CD90-2",
        "CD11c": "Mouse-CD11c",
        "CD31": "Mouse-CD31-Pecam",
        "F4_80": "Mouse-F480",
        "CD45R_B220": "Mouse-CD45R-B220",
        "Rat_IgG2a": "Rat-IgG2a",
        "CD11b": "Ms-Hu-CD11b",
        "CD19": "Mouse-CD19",
        "Mouse_IgG2a": "Mouse-IgG2a",
        "CD169": "Mouse-CD169",
    },
}
ADT_RAW_OTHER = {
    "CD5": "mouse_CD5",
    "CD68": "mouse_CD68",
    "CD4": "mouse_CD4",
    "CD29": "mouse_rat_CD29",
    "CD8a": "mouse_CD8a",
    "CD3": "mouse_CD3",
    "CD44": "mouse_human_CD44",
    "CD90_2": "mouse_CD90_2",
    "CD11c": "mouse_CD11c",
    "CD31": "mouse_CD31",
    "F4_80": "mouse_F4_80",
    "CD45R_B220": "mouse_human_CD45R_B220",
    "Rat_IgG2a": "Rat_IgG2a",
    "CD11b": "mouse_human_CD11b",
    "CD19": "mouse_CD19",
    "Mouse_IgG2a": "Mouse_IgG2a",
    "CD169": "mouse_CD169_Siglec-1",
}
for _section in THYMUS_SECTIONS[1:]:
    ADT_RAW[_section] = ADT_RAW_OTHER


def canonical_thymus_coords(rna: ad.AnnData) -> np.ndarray:
    if not {"x", "y"}.issubset(rna.obs.columns):
        raise KeyError("Mouse Thymus RNA requires obs['x'] and obs['y'].")
    coords = rna.obs.loc[:, ["x", "y"]].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or np.unique(coords, axis=0).shape[0] != rna.n_obs:
        raise ValueError("Mouse Thymus RNA coordinates must be finite and unique.")
    return coords


def adapt_thymus_pair(section: str, rna: ad.AnnData, adt: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    if list(rna.obs_names.astype(str)) != list(adt.obs_names.astype(str)):
        raise ValueError(f"{section}: RNA and ADT spot order differs.")
    if not rna.var_names.is_unique:
        raise ValueError(f"{section}: RNA var_names must be unique.")
    coords = canonical_thymus_coords(rna)
    rna.obsm["spatial"] = coords.copy()
    adt.obsm["spatial"] = coords.copy()
    for obj in (rna, adt):
        obj.obs["x"] = coords[:, 0]
        obj.obs["y"] = coords[:, 1]

    raw_names = pd.Index(adt.var_names.astype(str))
    raw_by_marker = ADT_RAW[section]
    missing = [raw_by_marker[name] for name in ADT_ORDER if raw_by_marker[name] not in raw_names]
    if missing:
        raise KeyError(f"{section}: missing mapped ADT markers: {missing}")
    positions = [int(raw_names.get_loc(raw_by_marker[name])) for name in ADT_ORDER]
    adt = adt[:, positions].copy()
    adt.var["adt_name_original"] = [raw_by_marker[name] for name in ADT_ORDER]
    adt.var["adt_marker"] = ADT_ORDER
    adt.var_names = pd.Index(ADT_ORDER, name="adt_marker")
    return rna, adt


def read_thymus_pair(data_dir: Path, section: str) -> tuple[ad.AnnData, ad.AnnData]:
    folder = data_dir / section
    rna = ad.read_h5ad(folder / "adata_RNA.h5ad")
    adt = ad.read_h5ad(folder / "adata_ADT.h5ad")
    return adapt_thymus_pair(section, rna, adt)


def simulation_spatial_domain(spfac: np.ndarray) -> np.ndarray:
    active = np.asarray(spfac).sum(axis=1) > 0
    labels = np.full(len(spfac), "background", dtype=object)
    labels[active] = np.asarray(
        [f"sp{i + 1}" for i in np.asarray(spfac)[active].argmax(axis=1)]
    )
    return labels


def adapt_simulation_pair(
    section: str, rna: ad.AnnData, adt: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    if list(rna.obs_names.astype(str)) != list(adt.obs_names.astype(str)):
        raise ValueError(f"{section}: RNA and ADT spot order differs.")
    if rna.shape != (1296, 1000) or adt.shape != (1296, 100):
        raise ValueError(f"{section}: unexpected RNA/ADT shapes {rna.shape}/{adt.shape}.")
    for name, obj in (("RNA", rna), ("ADT", adt)):
        values = np.asarray(obj.X)
        if not np.isfinite(values).all() or values.min() < 0:
            raise ValueError(f"{section} {name}: X must be finite and nonnegative.")
        if not np.allclose(values, np.rint(values)):
            raise ValueError(f"{section} {name}: X must contain integer-valued observations.")
    coords = np.asarray(rna.obsm["spatial"], dtype=np.float32)
    if coords.shape != (rna.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError(f"{section}: invalid RNA obsm['spatial'].")
    if not np.array_equal(coords, np.asarray(adt.obsm["spatial"])):
        raise ValueError(f"{section}: RNA and ADT spatial coordinates differ.")
    spfac = np.asarray(rna.obsm["spfac"], dtype=np.float32)
    if spfac.shape != (rna.n_obs, 4):
        raise ValueError(f"{section}: expected four spatial factors.")
    rna_ns = np.asarray(rna.obsm["nsfac"], dtype=np.float32)
    adt_ns = np.asarray(adt.obsm["nsfac"], dtype=np.float32)
    for obj in (rna, adt):
        obj.obsm["spatial"] = coords.copy()
        obj.obs["x"] = coords[:, 0]
        obj.obs["y"] = coords[:, 1]
        obj.obs["spatial_domain"] = simulation_spatial_domain(spfac)
        obj.obs["original_barcode"] = obj.obs_names.astype(str)
        for i in range(4):
            obj.obs[f"spfac_{i + 1}"] = spfac[:, i]
    for i in range(3):
        rna.obs[f"rna_nsfac_{i + 1}"] = rna_ns[:, i]
        rna.obs[f"adt_nsfac_{i + 1}"] = adt_ns[:, i]
        adt.obs[f"rna_nsfac_{i + 1}"] = rna_ns[:, i]
        adt.obs[f"adt_nsfac_{i + 1}"] = adt_ns[:, i]
    return rna, adt


def read_simulation_pair(data_dir: Path, section: str) -> tuple[ad.AnnData, ad.AnnData]:
    folder = data_dir / section
    rna = ad.read_h5ad(folder / "adata_RNA.h5ad")
    adt = ad.read_h5ad(folder / "adata_ADT.h5ad")
    return adapt_simulation_pair(section, rna, adt)


def simulation_prepared_truth(rna_mem, protein_mem) -> dict:
    """Expose already adapted, selected scientific metadata without copying it.

    This explicit strategy is used only by the Simulation caller. It neither
    reads labels nor changes their scope; factor arrays retain their owner.
    """
    return {
        "status": "provided",
        "labels": {section: rna.obs["spatial_domain"] for section, rna in rna_mem.items()},
        "factors": {
            section: {
                "spfac": rna.obsm["spfac"],
                "rna_nsfac": rna.obsm["nsfac"],
                "protein_nsfac": protein_mem[section].obsm["nsfac"],
            }
            for section, rna in rna_mem.items()
        },
        "source": "adapt_simulation_pair: existing spatial_domain/spfac/nsfac",
        "scope": "selected spots in section_order; metadata only",
    }
