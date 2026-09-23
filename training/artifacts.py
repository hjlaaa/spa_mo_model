"""Artifact mechanics, with caller-selected existing layouts and error behavior.

Callers choose payloads, output directories, export flags and invocation time.
The existing tensor/JSON conversion authorities are reused without new policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from model.tensor_utils import tensor_to_numpy
from .fit import json_safe


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def indexer_to_numpy(indexer: np.ndarray | slice, n_obs: int) -> np.ndarray:
    if isinstance(indexer, slice):
        return np.arange(n_obs, dtype=np.int64)
    return np.asarray(indexer, dtype=np.int64)


def save_selected_spot_indices(
    output_dir: Path,
    obs_indices: Mapping[str, np.ndarray | slice],
    backed_rna: Mapping[str, Any],
) -> dict[str, str]:
    paths = {}
    for section, indexer in obs_indices.items():
        path = output_dir / f"selected_spot_indices_{section}.npy"
        np.save(path, indexer_to_numpy(indexer, backed_rna[section].n_obs))
        paths[section] = str(path)
    return paths


def save_spatial_arrays(output_dir: Path, spatial_loc_dict: Mapping[str, Any]) -> dict[str, str]:
    paths = {}
    for section, spatial in spatial_loc_dict.items():
        path = output_dir / f"spatial_{section}.npy"
        np.save(path, np.asarray(spatial))
        paths[section] = str(path)
    return paths


def save_final_embeddings(
    output_dir: Path,
    final_embeddings: Mapping[str, torch.Tensor],
    *,
    filename_template: str = "final_embeddings_{section}.npy",
) -> dict[str, str]:
    paths = {}
    for section, tensor in final_embeddings.items():
        path = output_dir / filename_template.format(section=section)
        np.save(path, tensor_to_numpy(tensor))
        paths[section] = str(path)
    return paths


def save_json(
    path: Path,
    payload: Any,
    *,
    transform: Callable[[Any], Any] | None = None,
    buffered: bool = False,
    ensure_ascii: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Preserve the caller's projection and serialize-before-write policy.

    Streaming callers open first. Buffered callers serialize before opening or
    truncating the target. Optional defaults are explicit caller policies.
    """
    options = {"indent": 2}
    if not ensure_ascii:
        options["ensure_ascii"] = False
    if default is not None:
        options["default"] = default
    if buffered:
        write_text = path.write_text
        text = json.dumps(transform(payload) if transform is not None else payload, **options)
        write_text(text, encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(transform(payload) if transform is not None else payload, handle, **options)


def save_weights(path: Path, payload: Any) -> None:
    """Write the caller's existing torch payload without adding resume state."""
    torch.save(payload, path)


def save_ot_prior_topk(
    output_dir: Path,
    ot_prior: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    final_embeddings: Mapping[str, torch.Tensor],
    run_mode: str,
    save_candidate_qc: bool = False,
    *,
    skip_incomplete: bool = True,
    direction_meaning: str | None = "source receives information from target",
    note: str = "Saved sparse top-k UOT prior from model.ot_prior after final evaluation. X_to_Y means source X receives information from target Y. Dense P was not saved.",
    metadata_transform: Callable[[Any], Any] = json_safe,
    prepare_directory: Callable[[Path], None] = ensure_dir,
) -> dict[str, dict[str, str]]:
    """Write sparse priors; defaults preserve CRC, policies preserve other callers.

    Directory preparation and JSON projection remain explicit because existing
    callers differ in their accepted paths and metadata representations.
    """
    prepare_directory(output_dir)
    files: dict[str, dict[str, str]] = {}
    if not ot_prior:
        return files

    for (source_section, target_section), prior in ot_prior.items():
        pair_key = f"{source_section}_to_{target_section}"
        required = ["topk_idx", "topk_weight", "confidence", "row_mass"]
        if skip_incomplete and any(name not in prior for name in required):
            continue

        arrays = {
            "topk_idx": prior["topk_idx"].detach().cpu().numpy(),
            "topk_weight": prior["topk_weight"].detach().cpu().numpy(),
            "confidence": prior["confidence"].detach().cpu().numpy(),
            "row_mass": prior["row_mass"].detach().cpu().numpy(),
        }
        if save_candidate_qc:
            for optional_name in ["raw_topk_mass", "topk_coverage", "tail_mass", "target_hit_count"]:
                if optional_name in prior:
                    arrays[optional_name] = prior[optional_name].detach().cpu().numpy()
        paths = {
            name: output_dir / f"{pair_key}_{name}.npy"
            for name in arrays.keys()
        }
        paths.update({
            "metadata": output_dir / f"{pair_key}_metadata.json",
        })
        for name, array in arrays.items():
            np.save(paths[name], array)

        metadata = {
            "source_section": source_section,
            "target_section": target_section,
        }
        if direction_meaning is not None:
            metadata["direction_meaning"] = direction_meaning
        metadata.update({
            "topk": int(arrays["topk_idx"].shape[1]) if arrays["topk_idx"].ndim == 2 else None,
            "n_source": int(final_embeddings[source_section].shape[0]),
            "n_target": int(final_embeddings[target_section].shape[0]),
            "modalities_used": list(prior.get("modalities_used", [])),
            "run_mode": run_mode,
            "note": note,
        })
        metadata.update(prior.get("metadata", {}))
        save_json(paths["metadata"], metadata, transform=metadata_transform)
        files[pair_key] = {name: str(path) for name, path in paths.items()}
    return files
