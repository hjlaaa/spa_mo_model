"""Mechanical selected-row access; legacy visualization policy exports remain lazy."""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
from data_io.paired import subset_to_memory

def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def selected_original_indices(run_dir: Path, saved_section: str, selection: np.ndarray) -> np.ndarray:
    trained = np.load(run_dir / f'selected_spot_indices_{saved_section}.npy', mmap_mode='r')
    return np.asarray(trained[selection], dtype=np.int64)


def subset_obs_vars_memory_safe(value: ad.AnnData, obs_indices: np.ndarray, var_names: list[str] | None, chunk_size: int=5000) -> ad.AnnData:
    """Subset backed matrices without using two h5py fancy index vectors."""
    obs_indices = np.asarray(obs_indices, dtype=np.int64)
    if not getattr(value, 'isbacked', False):
        view = value[obs_indices, :] if var_names is None else value[obs_indices, var_names]
        return view.copy()
    if var_names is None:
        return subset_to_memory(value, obs_indices, None)
    chunks = []
    for start in range(0, len(obs_indices), chunk_size):
        rows = obs_indices[start:start + chunk_size]
        chunk = value[rows, :].to_memory()
        chunks.append(chunk[:, var_names].copy())
    if not chunks:
        raise ValueError('Cannot preprocess an empty observation selection.')
    if len(chunks) == 1:
        return chunks[0]
    return ad.concat(chunks, axis=0, join='inner', merge='same', index_unique=None)


# Compatibility only: formal analysis caller imports the workflow at its owner.
_LEGACY_VISUALIZATION = ('preprocess_paired_rna_protein_inputs', 'preprocess_adapted_rna_protein_inputs', 'preprocess_misar_inputs', 'preprocess_mousebrain_inputs', 'load_spatch_modality_subset')

def __getattr__(name):
    if name in _LEGACY_VISUALIZATION:
        from analysis import visualization_preparation
        return getattr(visualization_preparation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
