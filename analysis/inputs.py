"""Existing saved-array/metadata readers for general and large analysis inputs.

No preprocessing cache, model, training entry, metric or plotting imports.
Historical saved_files lookup precedence and insufficient-ID fallbacks are
retained as they were; these readers do not infer missing historical evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _crc_resolve_existing_path(input_dir: Path, candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{joined}")


def _crc_load_array(input_dir: Path, section: str, kind: str, summary: dict[str, Any] | None) -> tuple[np.ndarray, str]:
    saved_files = (summary or {}).get("saved_files", {})
    summary_path = None
    if kind == "embedding":
        summary_path = (saved_files.get("final_embeddings") or {}).get(section)
        candidates = [
            input_dir / f"final_embeddings_{section}.npy",
            input_dir / "final_embeddings" / f"{section}_final_embedding.npy",
        ]
    elif kind == "spatial":
        summary_path = (saved_files.get("spatial") or {}).get(section)
        candidates = [input_dir / f"spatial_{section}.npy"]
    elif kind == "spot_index":
        summary_path = (saved_files.get("selected_spot_indices") or {}).get(section)
        candidates = [input_dir / f"selected_spot_indices_{section}.npy"]
    else:
        raise ValueError(f"Unsupported array kind: {kind}")

    if summary_path:
        candidates.insert(0, Path(summary_path))
    path = _crc_resolve_existing_path(input_dir, candidates, f"{kind} for {section}")
    return np.load(path), str(path)


def load_crc_inputs(input_dir: Path, section_order):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")
    summary = load_json(summary_path)
    loss_history_path = input_dir / "loss_history.json"
    loss_history = load_json(loss_history_path) if loss_history_path.exists() else None

    embeddings = {}
    spatial = {}
    spot_indices = {}
    obs_names = {}
    source_paths = {"run_summary": str(summary_path)}
    if loss_history_path.exists():
        source_paths["loss_history"] = str(loss_history_path)

    for section in section_order:
        embeddings[section], source_paths[f"embedding_{section}"] = _crc_load_array(input_dir, section, "embedding", summary)
        spatial[section], source_paths[f"spatial_{section}"] = _crc_load_array(input_dir, section, "spatial", summary)
        spot_indices[section], source_paths[f"spot_index_{section}"] = _crc_load_array(input_dir, section, "spot_index", summary)
        if embeddings[section].ndim != 2:
            raise ValueError(f"{section} embedding must be 2D, got {embeddings[section].shape}.")
        if spatial[section].ndim != 2 or spatial[section].shape[1] < 2:
            raise ValueError(f"{section} spatial must be [N, >=2], got {spatial[section].shape}.")
        n_spots = embeddings[section].shape[0]
        if spatial[section].shape[0] != n_spots:
            raise ValueError(f"{section} embedding/spatial rows differ: {n_spots} vs {spatial[section].shape[0]}.")
        if spot_indices[section].shape[0] != n_spots:
            raise ValueError(
                f"{section} embedding/spot index rows differ: {n_spots} vs {spot_indices[section].shape[0]}."
            )
        data_root = Path(summary.get("input_data_path", ""))
        rna_path = data_root / section / "adata_RNA.h5ad"
        if rna_path.exists():
            import anndata as ad

            rna = ad.read_h5ad(rna_path, backed="r")
            try:
                all_obs_names = np.asarray(rna.obs_names.astype(str))
                obs_names[section] = all_obs_names[
                    spot_indices[section].astype(np.int64, copy=False)
                ]
            finally:
                rna.file.close()
            source_paths[f"obs_names_{section}"] = str(rna_path)
        else:
            obs_names[section] = np.asarray(
                [str(value) for value in spot_indices[section]],
                dtype=object,
            )

    return (
        embeddings,
        spatial,
        spot_indices,
        obs_names,
        summary,
        loss_history,
        source_paths,
    )


def _misar_parse_str_list(text: str | None, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values or list(default)


def _misar_load_array(input_dir: Path, section: str, kind: str, summary: dict[str, Any]) -> tuple[np.ndarray, str]:
    saved = summary.get("saved_files", {})
    if kind == "embedding":
        from_summary = (saved.get("final_embeddings") or {}).get(section)
        candidates = [input_dir / f"final_embeddings_{section}.npy"]
    elif kind == "spatial":
        from_summary = (saved.get("spatial") or {}).get(section)
        candidates = [input_dir / f"spatial_{section}.npy"]
    elif kind == "spot_index":
        from_summary = (saved.get("selected_spot_indices") or {}).get(section)
        candidates = [input_dir / f"selected_spot_indices_{section}.npy"]
    else:
        raise ValueError(f"Unsupported array kind: {kind}")
    if from_summary:
        candidates.insert(0, Path(from_summary))
    for path in candidates:
        if path.exists():
            return np.load(path), str(path)
    raise FileNotFoundError(f"Could not find {kind} for {section}; tried {candidates}")


def _misar_load_obs_metadata(input_dir: Path, section: str, summary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    from_summary = ((summary.get("saved_files") or {}).get("obs_metadata") or {}).get(section)
    candidates = []
    if from_summary:
        candidates.append(Path(from_summary))
    candidates.append(input_dir / f"obs_metadata_{section}.csv")
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)
    raise FileNotFoundError(f"Could not find obs metadata for {section}; tried {candidates}")


def load_misar_inputs(input_dir: Path, section_order_arg: str | None, default_section_order):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")
    summary = load_json(summary_path)
    section_order = _misar_parse_str_list(section_order_arg, summary.get("section_names") or default_section_order)
    embeddings = {}
    spatial = {}
    spot_indices = {}
    obs_meta = {}
    source_paths = {"run_summary": str(summary_path)}
    for section in section_order:
        embeddings[section], source_paths[f"embedding_{section}"] = _misar_load_array(input_dir, section, "embedding", summary)
        spatial[section], source_paths[f"spatial_{section}"] = _misar_load_array(input_dir, section, "spatial", summary)
        spot_indices[section], source_paths[f"spot_index_{section}"] = _misar_load_array(input_dir, section, "spot_index", summary)
        obs_meta[section], source_paths[f"obs_metadata_{section}"] = _misar_load_obs_metadata(input_dir, section, summary)
        n = embeddings[section].shape[0]
        if spatial[section].shape[0] != n:
            raise ValueError(f"{section}: spatial rows {spatial[section].shape[0]} != embedding rows {n}.")
        if spot_indices[section].shape[0] != n:
            raise ValueError(f"{section}: spot index rows {spot_indices[section].shape[0]} != embedding rows {n}.")
        if obs_meta[section].shape[0] != n:
            raise ValueError(f"{section}: obs metadata rows {obs_meta[section].shape[0]} != embedding rows {n}.")
    return embeddings, spatial, spot_indices, obs_meta, summary, source_paths, section_order


def load_spatch_inputs(input_dir: Path, section_order):
    from data_io.large_results import load_spatch_inputs as read
    return read(input_dir, section_order)


def load_spatch_metadata(metadata_path: Path, id_column: str, labels) -> pd.DataFrame:
    from data_io.large_results import load_spatch_metadata as read
    return read(metadata_path, id_column, labels)


def load_spatch_embedding(embedding_paths, *, section_order, section_counts, name: str) -> np.ndarray:
    from data_io.large_results import load_spatch_embedding as read
    return read(embedding_paths, section_order=section_order, section_counts=section_counts, name=name)


def load_requested_spatch_metadata(run_dir: Path, *, truth_labels, n_obs, section_counts) -> pd.DataFrame:
    frame = load_spatch_metadata(run_dir / "spot_metadata.csv.gz", "block_id", truth_labels)
    if len(frame) != n_obs:
        raise ValueError(f"Expected {n_obs} SPATCH rows, got {len(frame)}.")
    if frame["section"].value_counts().to_dict() != section_counts:
        raise ValueError("SPATCH section counts differ from v6.")
    if frame[["section", "spot_id"]].duplicated().any():
        raise ValueError("Duplicate SPATCH section/spot IDs.")
    if not np.isfinite(frame[["x", "y"]].to_numpy()).all():
        raise ValueError("Non-finite SPATCH spatial coordinates.")
    return frame


def load_requested_spatch_embedding(run_dir: Path, *, section_order, section_counts) -> np.ndarray:
    arrays = []
    for section in section_order:
        path = run_dir / f"final_embeddings_{section}.npy"
        array = np.load(path, mmap_mode="r")
        if array.shape != (section_counts[section], 128):
            raise ValueError(f"Unexpected {section} embedding shape: {array.shape}.")
        arrays.append(np.asarray(array, dtype=np.float32))
    embedding = np.vstack(arrays)
    if not np.isfinite(embedding).all():
        raise ValueError("SPATCH embedding contains non-finite values.")
    from data_io.large_results import borrowed_input
    return borrowed_input(embedding, evidence={"kind": "section_file_row_order", "section_order": section_order},
        provenance={"run_dir": run_dir, "truth_status": "not_loaded"}).embedding


def load_human_embryo(run_dir: Path):
    from data_io.large_results import load_human_embryo as read
    return read(run_dir)


def load_human_embryo_plot_metadata(run_summary: dict[str, Any], sections: list[str]) -> dict[str, pd.DataFrame]:
    from data_io.large_results import load_human_embryo_plot_metadata as read
    return read(run_summary, sections)


def load_simulation_embeddings(run_dir: Path, section_order) -> dict[str, np.ndarray]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    result = {}
    for section in section_order:
        path = Path(summary["saved_files"]["final_embeddings"][section])
        result[section] = np.load(path)
    return result


def load_simulation_factors(data_dir: Path, section_order):
    import anndata as ad

    spfac, rna_nsfac, adt_nsfac = [], [], []
    for section in section_order:
        rna = ad.read_h5ad(data_dir / section / "adata_RNA.h5ad", backed="r")
        adt = ad.read_h5ad(data_dir / section / "adata_ADT.h5ad", backed="r")
        try:
            spfac.append(np.asarray(rna.obsm["spfac"]))
            rna_nsfac.append(np.asarray(rna.obsm["nsfac"]))
            adt_nsfac.append(np.asarray(adt.obsm["nsfac"]))
        finally:
            rna.file.close()
            adt.file.close()
    return spfac, rna_nsfac, adt_nsfac
