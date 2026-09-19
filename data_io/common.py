"""Thin adapters and validators for COSIE-style preprocessing.

These helpers intentionally avoid implementing normalization, log transforms,
scaling, PCA, CLR, or other preprocessing algorithms. The actual data
preprocessing remains in ``data_io.preprocessing`` and follows COSIE.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def is_missing_input(x: Any) -> bool:
    """Return True for the missing-input sentinels accepted by wrappers."""

    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "na"}:
        return True
    return False


def ensure_dir(path: str | os.PathLike | None) -> None:
    """Create a directory or a file's parent directory if needed."""

    if is_missing_input(path):
        return
    path_obj = Path(path)
    target_dir = path_obj if path_obj.suffix == "" else path_obj.parent
    if str(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)


def load_h5ad_if_needed(x: Any):
    """Load an h5ad path if needed; return AnnData-like objects unchanged."""

    if is_missing_input(x):
        return None
    if isinstance(x, (str, os.PathLike)):
        import scanpy as sc

        return sc.read_h5ad(str(x))
    return x


def get_embedding_shape(x: Any):
    """Best-effort shape extraction for summaries."""

    if is_missing_input(x):
        return None
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if hasattr(x, "X") and hasattr(x.X, "shape"):
        return tuple(x.X.shape)
    return None


def check_spatial_key(adata: Any, spatial_key: str = "spatial") -> bool:
    """Raise a clear error if an AnnData object lacks the requested spatial key."""

    if adata is None:
        return False
    if not hasattr(adata, "obsm"):
        raise TypeError("Expected an AnnData-like object with .obsm.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData object is missing obsm['{spatial_key}'].")
    return True


def check_obs_names_consistency(adatas: Iterable[Any]) -> bool:
    """Warn when non-missing AnnData objects do not share obs_names order."""

    present = [adata for adata in adatas if adata is not None]
    if len(present) <= 1:
        return True
    reference = list(present[0].obs_names)
    for idx, adata in enumerate(present[1:], start=1):
        if list(adata.obs_names) != reference:
            warnings.warn(
                "AnnData obs_names differ across modalities. COSIE-style "
                "preprocessing does not reorder by obs_names; it relies on "
                "input row order and spatial consistency. Mismatch at "
                f"non-missing modality index {idx}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
    return True


def check_spatial_consistency(
    adatas: Iterable[Any],
    spatial_key: str = "spatial",
    raise_on_mismatch: bool = True,
) -> bool:
    """Check COSIE-style same-section spatial consistency."""

    spatial_arrays = []
    for adata in adatas:
        if adata is None:
            continue
        check_spatial_key(adata, spatial_key=spatial_key)
        spatial_arrays.append(np.asarray(adata.obsm[spatial_key]))

    if len(spatial_arrays) <= 1:
        return True

    reference = spatial_arrays[0]
    for idx, spatial in enumerate(spatial_arrays[1:], start=1):
        if not np.array_equal(reference, spatial):
            message = (
                "Inconsistent spatial coordinates across modalities in the "
                f"same section for obsm['{spatial_key}']; mismatch at "
                f"non-missing modality index {idx}."
            )
            if raise_on_mismatch:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return False
    return True
