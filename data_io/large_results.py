"""Explicit large-result formats. No cohort, sampling or scientific policy."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from data_io.analysis_results import AlignedAnalysisInput


def borrowed_input(embedding, *, sections=None, spot_ids=None, coords=None, truth=None,
                   evidence, provenance):
    """Borrow only supplied arrays/metadata; None means not loaded, never invented IDs."""
    return AlignedAnalysisInput(embedding, sections, spot_ids, coords, truth, evidence, provenance)

def load_spatch_inputs(input_dir: Path, section_order):
    summary_path = input_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(input_dir / "spot_metadata.csv.gz", low_memory=False)
    embeddings = {}
    spatial = {}
    spot_indices = {}
    obs_meta = {}
    sources = {"metadata": str(input_dir / "spot_metadata.csv.gz")}
    for section in section_order:
        candidates = [
            input_dir / f"final_embeddings_{section}.npy",
            input_dir / "final_embeddings" / f"{section}_final_embedding.npy",
        ]
        emb_path = next((path for path in candidates if path.is_file()), None)
        if emb_path is None:
            raise FileNotFoundError(
                f"Cannot locate {section} embedding; tried: {candidates}"
            )
        emb = np.load(emb_path, mmap_mode="r")
        meta = metadata.loc[metadata["section"].astype(str).eq(section)].reset_index(drop=True)
        if len(meta) != emb.shape[0]:
            raise ValueError(f"{section}: embedding/metadata row mismatch.")
        embeddings[section] = np.asarray(emb)
        spatial[section] = meta[["x", "y"]].to_numpy(dtype=np.float32)
        spot_indices[section] = np.arange(len(meta), dtype=np.int64)
        obs_meta[section] = meta
        contract = borrowed_input(embeddings[section], sections=meta["section"].to_numpy(),
            spot_ids=meta["block_id"].to_numpy() if "block_id" in meta else None,
            coords=spatial[section], evidence={"kind": "metadata_block_id_and_row_order" if "block_id" in meta else "row_order_only"},
            provenance={"embedding": emb_path, "metadata": meta, "truth_status": "available_in_metadata"})
        embeddings[section] = contract.embedding
        sources[f"embedding_{section}"] = str(emb_path)
    return embeddings, spatial, spot_indices, obs_meta, summary, sources, section_order


def load_spatch_metadata(metadata_path: Path, id_column: str, labels) -> pd.DataFrame:
    usecols = [
        id_column,
        "section",
        "original_rna_obs_name",
        "x",
        "y",
        *labels,
    ]
    frame = pd.read_csv(
        metadata_path,
        usecols=usecols,
        dtype={
            id_column: str,
            "section": str,
            "original_rna_obs_name": str,
        },
        low_memory=False,
    ).rename(columns={id_column: "spot_id"})
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["section"] = frame["section"].astype(str)
    frame["x"] = pd.to_numeric(frame["x"], errors="raise").astype(np.float32)
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(np.float32)
    return frame


def load_spatch_embedding(embedding_paths, *, section_order, section_counts, name: str) -> np.ndarray:
    arrays = []
    for section in section_order:
        array = np.load(embedding_paths[section], mmap_mode="r")
        if array.shape[0] != section_counts[section]:
            raise ValueError(
                f"{name} {section}: unexpected embedding shape {array.shape}"
            )
        arrays.append(np.asarray(array))
    embedding = np.vstack(arrays)
    return borrowed_input(embedding, evidence={"kind": "section_file_row_order", "section_order": section_order},
                          provenance={"source_paths": embedding_paths, "truth_status": "not_loaded"}).embedding



def load_human_embryo(run_dir: Path):
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("mode") != "train":
        raise ValueError("Input run_summary does not describe a successful training run.")
    if summary.get("model_mode", {}).get("modalities") != ["RNA"]:
        raise ValueError("Expected an RNA-only model run.")
    manifest = summary["preprocess_manifest"]
    sections = list(summary["section_order"])
    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for section in sections:
        embedding_path = Path(summary["saved_files"]["embeddings"][section])
        annotation_path = Path(manifest["annotation_files"][section])
        embeddings[section] = np.load(embedding_path, mmap_mode="r")
        header = pd.read_csv(annotation_path, nrows=0).columns
        wanted = [
            name for name in
            ("section", "original_row", "obs_name", "cellid", "celltype", "stage", "x", "y")
            if name in header
        ]
        metadata[section] = pd.read_csv(annotation_path, usecols=wanted)
        if embeddings[section].ndim != 2 or embeddings[section].shape[1] != 128:
            raise ValueError(f"{section}: unexpected embedding shape {embeddings[section].shape}.")
        if len(embeddings[section]) != len(metadata[section]):
            raise ValueError(f"{section}: embedding/metadata row mismatch.")
    for section in sections:
        frame = metadata[section]
        id_column = next((key for key in ("obs_name", "cellid", "original_row") if key in frame), None)
        contract = borrowed_input(embeddings[section],
            sections=frame["section"].to_numpy() if "section" in frame else None,
            spot_ids=frame[id_column].to_numpy() if id_column else None,
            evidence={"kind": "annotation_column" if id_column else "row_order_only",
                      "id_column": id_column, "section_from_layout": section,
                      "source_row": frame["original_row"].to_numpy() if "original_row" in frame else None},
            provenance={"embedding": summary["saved_files"]["embeddings"][section],
                        "annotation": manifest["annotation_files"][section],
                        "metadata": frame, "truth_status": "available_in_metadata"})
        embeddings[section] = contract.embedding
    return summary, sections, embeddings, metadata


def load_human_embryo_plot_metadata(
    run_summary: dict[str, Any], sections: list[str]
) -> dict[str, pd.DataFrame]:
    annotation_files = run_summary["preprocess_manifest"]["annotation_files"]
    metadata: dict[str, pd.DataFrame] = {}
    for section in sections:
        path = Path(annotation_files[section])
        frame = pd.read_csv(path, usecols=["celltype", "x", "y"])
        expected = int(run_summary["final_embedding_shapes"][section][0])
        if len(frame) != expected:
            raise ValueError(
                f"{section}: annotation rows={len(frame):,}, expected={expected:,}."
            )
        metadata[section] = frame
    return metadata



def read_section_arrays(paths, *, mmap_mode="r"):
    """Explicit ordered path mapping; no selection, casting or materialization."""
    return {section: np.load(path, mmap_mode=mmap_mode) for section, path in paths.items()}


def read_selected_obs(path, selection):
    frame = pd.read_csv(path).iloc[selection].reset_index(drop=True)
    frame['spot_id'] = frame['obs_name'].astype(str)
    return frame.drop(columns=['section'], errors='ignore')
