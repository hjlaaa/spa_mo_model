#!/usr/bin/env python3
"""Generate SpaMosaic-style input (panel c) and integrated (panel e) UMAPs.

The script is deliberately post-hoc: it reads existing result_v6 embeddings,
saved cluster labels, and source modalities without retraining spa_mo_model.
Large datasets use one deterministic, section-proportional visualization
sample shared by every before/after panel.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.data_preprocessing import load_cosie_style_data  # noqa: E402
from model.multimodal_preprocessing import (  # noqa: E402
    build_cosie_data_dict,
    build_mousebrain_section,
)
from scripts.run_crc_stereocite import (  # noqa: E402
    prepare_rna_var_names_make_unique,
    read_backed_pair as read_crc_backed_pair,
    subset_to_memory,
)
from scripts.run_misar_seq import (  # noqa: E402
    common_var_names,
    make_unique_rna_var_names,
    read_backed_pair,
)
from scripts import run_mouse_thymus, run_simulation  # noqa: E402


RESULT_ROOT = PROJECT_ROOT / "result_v6"
DATASETS = (
    "human_embryo",
    "misar_seq",
    "mousebrain",
    "spatch",
    "crc_stereocite",
    "human_lymph_node",
    "mouse_spleen",
    "mouse_thymus",
    "simulation",
)
SECTION_COLORS = list(plt.get_cmap("tab10").colors)
MISSING_COLOR = "#d3d3d3"


@dataclass
class DatasetBundle:
    name: str
    display_name: str
    run_dir: Path
    analysis_dir: Path
    sections: list[str]
    section_counts: dict[str, int]
    selections: dict[str, np.ndarray]
    metadata: pd.DataFrame
    input_features: dict[str, np.ndarray]
    integrated_features: np.ndarray
    primary_label: str
    joint_labels: dict[int, np.ndarray]
    representative_k: int
    input_provenance: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite-umap",
        action="store_true",
        help="Recompute cached UMAP coordinate arrays.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical_palette(count: int) -> list[Any]:
    colors: list[Any] = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(plt.get_cmap(name).colors)
    if count > len(colors):
        colors.extend(
            plt.get_cmap("hsv")(
                np.linspace(0, 1, count - len(colors), endpoint=False)
            )
        )
    return colors[:count]


def natural_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def clean_labels(values: Any) -> np.ndarray:
    series = pd.Series(values, dtype=object)
    result = series.astype(str).to_numpy(dtype=object)
    missing = series.isna().to_numpy() | np.isin(
        np.char.lower(result.astype(str)), ["nan", "none", "na", ""]
    )
    result[missing] = "NA"
    return result


def proportional_selections(
    sections: list[str],
    counts: dict[str, int],
    maximum: int,
    seed: int,
) -> dict[str, np.ndarray]:
    total = sum(counts.values())
    if maximum <= 0 or total <= maximum:
        return {
            section: np.arange(counts[section], dtype=np.int64)
            for section in sections
        }
    exact = np.array([maximum * counts[s] / total for s in sections])
    quotas = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - quotas))[: maximum - int(quotas.sum())]:
        quotas[index] += 1
    rng = np.random.default_rng(seed)
    return {
        section: np.sort(
            rng.choice(counts[section], size=int(quota), replace=False)
        ).astype(np.int64)
        for section, quota in zip(sections, quotas)
    }


def stack_selected(
    arrays: dict[str, np.ndarray],
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> np.ndarray:
    return np.vstack(
        [np.asarray(arrays[section][selections[section]], dtype=np.float32) for section in sections]
    )


def scale_full_then_select(
    arrays: dict[str, np.ndarray],
    sections: list[str],
    selections: dict[str, np.ndarray],
    chunk_size: int = 50_000,
) -> np.ndarray:
    scaler = StandardScaler()
    for section in sections:
        values = arrays[section]
        for start in range(0, len(values), chunk_size):
            scaler.partial_fit(
                np.asarray(values[start : start + chunk_size], dtype=np.float32)
            )
    selected = stack_selected(arrays, sections, selections)
    return scaler.transform(selected).astype(np.float32, copy=False)


def apply_saved_scaler(
    values: np.ndarray, scaler_path: Path, prefix: str
) -> np.ndarray:
    saved = np.load(scaler_path)
    mean = np.asarray(saved[f"{prefix}_mean"], dtype=np.float32)
    scale = np.asarray(saved[f"{prefix}_scale"], dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32, copy=False)


def validate_features(name: str, values: np.ndarray, n_obs: int) -> None:
    if values.ndim != 2 or len(values) != n_obs:
        raise ValueError(f"{name}: invalid shape {values.shape}; expected {n_obs} rows.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite values detected.")


def section_manifest(
    sections: list[str],
    selections: dict[str, np.ndarray],
    aliases: dict[str, str],
) -> pd.DataFrame:
    frames = []
    offset = 0
    for section in sections:
        index = selections[section]
        frames.append(
            pd.DataFrame(
                {
                    "sample_index": np.arange(offset, offset + len(index)),
                    "section": section,
                    "section_label": aliases.get(section, section),
                    "section_row_index": index,
                }
            )
        )
        offset += len(index)
    return pd.concat(frames, ignore_index=True)


def load_npy_joint_labels(
    clustering_dir: Path,
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    result = {}
    for directory in sorted(clustering_dir.glob("joint_k*")):
        try:
            k = int(directory.name.split("joint_k", 1)[1])
        except ValueError:
            continue
        chunks = []
        for section in sections:
            labels = np.load(directory / f"labels_{section}.npy", mmap_mode="r")
            chunks.append(np.asarray(labels[selections[section]], dtype=np.int32))
        result[k] = np.concatenate(chunks)
    return result


def load_csv_joint_labels(
    clustering_dir: Path,
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    result = {}
    for directory in sorted(clustering_dir.glob("joint_k*")):
        try:
            k = int(directory.name.split("joint_k", 1)[1])
        except ValueError:
            continue
        chunks = []
        for section in sections:
            table = pd.read_csv(directory / f"labels_{section}.csv")
            if "cluster" not in table:
                raise KeyError(f"{directory}: missing cluster column for {section}.")
            chunks.append(
                table["cluster"].to_numpy(dtype=np.int32)[selections[section]]
            )
        result[k] = np.concatenate(chunks)
    return result


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_original_indices(
    run_dir: Path,
    saved_section: str,
    selection: np.ndarray,
) -> np.ndarray:
    trained = np.load(
        run_dir / f"selected_spot_indices_{saved_section}.npy",
        mmap_mode="r",
    )
    return np.asarray(trained[selection], dtype=np.int64)


def subset_obs_vars_memory_safe(
    value: ad.AnnData,
    obs_indices: np.ndarray,
    var_names: list[str] | None,
    chunk_size: int = 5_000,
) -> ad.AnnData:
    """Subset backed matrices without using two h5py fancy index vectors."""
    obs_indices = np.asarray(obs_indices, dtype=np.int64)
    if not getattr(value, "isbacked", False):
        view = value[obs_indices, :] if var_names is None else value[obs_indices, var_names]
        return view.copy()
    if var_names is None:
        return subset_to_memory(value, obs_indices, None)
    chunks = []
    for start in range(0, len(obs_indices), chunk_size):
        rows = obs_indices[start : start + chunk_size]
        # Load observations with a single fancy index, then select variables
        # from the in-memory chunk. h5py rejects two simultaneous vectors.
        chunk = value[rows, :].to_memory()
        chunks.append(chunk[:, var_names].copy())
    if not chunks:
        raise ValueError("Cannot preprocess an empty observation selection.")
    if len(chunks) == 1:
        return chunks[0]
    return ad.concat(chunks, axis=0, join="inner", merge="same", index_unique=None)


def preprocess_paired_rna_protein_inputs(
    dataset: str,
    run_dir: Path,
    sections: list[str],
    saved_names: dict[str, str],
    source_dirs: dict[str, str],
    selections: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = PROJECT_ROOT / "data" / {
        "crc_stereocite": "CRC_Stereo-CITE-seq",
        "human_lymph_node": "Human_Lymph_Node",
        "mouse_spleen": "Mouse_Spleen",
    }[dataset]
    shared_genes = read_lines(run_dir / "shared_gene_symbols_make_unique.txt")
    marker_path = run_dir / "adt_marker_list.txt"
    markers = read_lines(marker_path) if marker_path.is_file() else None
    rna_subsets: list[ad.AnnData] = []
    protein_subsets: list[ad.AnnData] = []
    spot_ids: dict[str, np.ndarray] = {}
    backed: list[ad.AnnData] = []
    try:
        for section in sections:
            saved = saved_names[section]
            source_dir = source_dirs[section]
            if dataset == "crc_stereocite":
                rna, protein = read_crc_backed_pair(data_dir, source_dir)
                prepare_rna_var_names_make_unique(rna)
            else:
                folder = data_dir / source_dir
                rna = ad.read_h5ad(folder / "adata_RNA.h5ad", backed="r")
                protein = ad.read_h5ad(folder / "adata_ADT.h5ad", backed="r")
                if "gene_ids" not in rna.var:
                    raise KeyError(f"{dataset}/{section}: RNA var['gene_ids'] is required.")
                rna.var_names = pd.Index(rna.var["gene_ids"].astype(str))
                if dataset == "human_lymph_node":
                    if "gene_ids" not in protein.var:
                        raise KeyError(
                            f"{dataset}/{section}: ADT var['gene_ids'] is required."
                        )
                    protein.var_names = pd.Index(protein.var["gene_ids"].astype(str))
            backed.extend([rna, protein])
            original = selected_original_indices(run_dir, saved, selections[section])
            spot_ids[section] = rna.obs_names.astype(str).to_numpy()[original]
            rna_subsets.append(
                subset_obs_vars_memory_safe(rna, original, shared_genes)
            )
            protein_subsets.append(
                subset_obs_vars_memory_safe(protein, original, markers)
            )
    finally:
        for value in backed:
            value.file.close()

    feature_dict, _, _ = load_cosie_style_data(
        {"RNA": rna_subsets, "Protein": protein_subsets},
        n_comps=50,
        hvg_num=3000,
        hvg_num_by_modality={"RNA": 3000, "Protein": None},
        target_sum=None,
        use_harmony=True,
        metacell=False,
        memory_efficient=dataset == "crc_stereocite",
        retain_processed=False,
    )
    features: dict[str, np.ndarray] = {}
    for modality in ("RNA", "Protein"):
        arrays = {
            section: feature_dict[f"s{i + 1}"][modality].detach().cpu().numpy()
            for i, section in enumerate(sections)
        }
        local = {
            section: np.arange(len(arrays[section]), dtype=np.int64)
            for section in sections
        }
        features[modality] = scale_full_then_select(arrays, sections, local)
    return features, spot_ids


def load_paired_rna_protein(
    dataset: str,
    display_name: str,
    analysis_sections: list[str],
    saved_names: dict[str, str],
    source_dirs: dict[str, str],
    maximum: int,
    seed: int,
) -> DatasetBundle:
    run_dir = RESULT_ROOT / dataset / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    counts = {
        section: int(
            np.load(
                run_dir / f"final_embeddings_{saved_names[section]}.npy",
                mmap_mode="r",
            ).shape[0]
        )
        for section in analysis_sections
    }
    selections = proportional_selections(analysis_sections, counts, maximum, seed)
    metadata = section_manifest(analysis_sections, selections, {})
    input_features, spot_ids = preprocess_paired_rna_protein_inputs(
        dataset,
        run_dir,
        analysis_sections,
        saved_names,
        source_dirs,
        selections,
    )
    metadata["spot_id"] = np.concatenate(
        [spot_ids[section] for section in analysis_sections]
    )
    primary_label = "reference_annotation"
    metadata[primary_label] = "not_available"
    if dataset == "human_lymph_node":
        annotation_path = (
            PROJECT_ROOT
            / "data/Human_Lymph_Node/Human_Lymph_Node_A1/annotation.csv"
        )
        annotation = pd.read_csv(annotation_path).set_index("Barcode")["manual-anno"]
        mask = metadata["section"].eq("Human_Lymph_Node_A1")
        metadata.loc[mask, primary_label] = (
            metadata.loc[mask, "spot_id"].map(annotation).fillna("NA")
        )
    final_arrays = {
        section: np.load(
            run_dir / f"final_embeddings_{saved_names[section]}.npy",
            mmap_mode="r",
        )
        for section in analysis_sections
    }
    final_selected = stack_selected(final_arrays, analysis_sections, selections)
    analysis_dir = run_dir / "analysis"
    integrated = apply_saved_scaler(
        final_selected,
        analysis_dir / "standardized_embedding/scaler_parameters.npz",
        "joint",
    )
    joint = load_csv_joint_labels(
        analysis_dir / "standardized_embedding/clustering",
        analysis_sections,
        selections,
    )
    scope = (
        "deterministic_visualization_sample"
        if sum(counts.values()) > sum(len(v) for v in selections.values())
        else "full_dataset"
    )
    return DatasetBundle(
        dataset,
        display_name,
        run_dir,
        analysis_dir,
        analysis_sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        primary_label,
        joint,
        5,
        {
            "RNA": f"training-matched Harmony RNA features; preprocessing scope={scope}",
            "Protein": f"training-matched Harmony protein features; preprocessing scope={scope}",
            "preprocessing_scope": scope,
        },
    )


def load_crc_stereocite(maximum: int, seed: int) -> DatasetBundle:
    sections = ["CRC_003_bin20", "CRC_006_bin20"]
    return load_paired_rna_protein(
        "crc_stereocite",
        "CRC Stereo-CITE-seq",
        sections,
        {"CRC_003_bin20": "CRC_003", "CRC_006_bin20": "CRC_006"},
        {section: section for section in sections},
        maximum,
        seed,
    )


def load_human_lymph_node(maximum: int, seed: int) -> DatasetBundle:
    sections = ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]
    return load_paired_rna_protein(
        "human_lymph_node",
        "Human Lymph Node",
        sections,
        {section: section for section in sections},
        {section: section for section in sections},
        maximum,
        seed,
    )


def load_mouse_spleen(maximum: int, seed: int) -> DatasetBundle:
    sections = ["Mouse_Spleen1", "Mouse_Spleen2"]
    return load_paired_rna_protein(
        "mouse_spleen",
        "Mouse Spleen",
        sections,
        {section: section for section in sections},
        {section: section for section in sections},
        maximum,
        seed,
    )


def preprocess_adapted_rna_protein_inputs(
    adapter: Any,
    data_dir: Path,
    run_dir: Path,
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    namespace = adapter._load_pipeline()
    shared_genes = read_lines(run_dir / "shared_gene_symbols_make_unique.txt")
    shared_proteins = read_lines(run_dir / "shared_adt_peaks.txt")
    rna_values: list[ad.AnnData] = []
    protein_values: list[ad.AnnData] = []
    for section in sections:
        rna, protein = namespace["read_backed_pair"](data_dir, section)
        namespace["make_unique_rna_var_names"](section, rna)
        original = selected_original_indices(run_dir, section, selections[section])
        rna_values.append(
            subset_obs_vars_memory_safe(rna, original, shared_genes)
        )
        protein_values.append(
            subset_obs_vars_memory_safe(protein, original, shared_proteins)
        )
        if getattr(rna, "isbacked", False):
            rna.file.close()
        if getattr(protein, "isbacked", False):
            protein.file.close()
    hvg_rna = 1000 if adapter is run_simulation else 3000
    hvg_protein = 100 if adapter is run_simulation else len(shared_proteins)
    feature_dict, _, _ = load_cosie_style_data(
        {"RNA": rna_values, "Protein": protein_values},
        n_comps=50,
        hvg_num=hvg_rna,
        hvg_num_by_modality={"RNA": hvg_rna, "Protein": hvg_protein},
        target_sum=None,
        use_harmony=True,
        metacell=False,
        retain_processed=False,
    )
    result: dict[str, np.ndarray] = {}
    for modality in ("RNA", "Protein"):
        arrays = {
            section: feature_dict[f"s{i + 1}"][modality].detach().cpu().numpy()
            for i, section in enumerate(sections)
        }
        local = {
            section: np.arange(len(arrays[section]), dtype=np.int64)
            for section in sections
        }
        result[modality] = scale_full_then_select(arrays, sections, local)
    return result


def load_adapted_rna_protein(
    dataset: str,
    display_name: str,
    adapter: Any,
    data_name: str,
    maximum: int,
    seed: int,
) -> DatasetBundle:
    run_dir = RESULT_ROOT / dataset / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    sections = list(summary["section_names"])
    counts = {section: int(summary["final_embedding_shapes"][section][0]) for section in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    metadata = section_manifest(sections, selections, {})
    chunks = []
    for section in sections:
        frame = pd.read_csv(run_dir / f"obs_metadata_{section}.csv").iloc[
            selections[section]
        ].reset_index(drop=True)
        frame["spot_id"] = frame["obs_name"].astype(str)
        frame = frame.drop(columns=["section"], errors="ignore")
        chunks.append(frame)
    metadata = pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)
    if dataset == "simulation":
        primary_label = "spatial_domain"
        metadata[primary_label] = clean_labels(metadata[primary_label])
    else:
        primary_label = "reference_annotation"
        metadata[primary_label] = "not_available"
    input_features = preprocess_adapted_rna_protein_inputs(
        adapter,
        PROJECT_ROOT / "data" / data_name,
        run_dir,
        sections,
        selections,
    )
    final_arrays = {
        section: np.load(run_dir / f"final_embeddings_{section}.npy", mmap_mode="r")
        for section in sections
    }
    final_selected = stack_selected(final_arrays, sections, selections)
    analysis_dir = run_dir / "analysis"
    integrated = apply_saved_scaler(
        final_selected,
        analysis_dir / "standardized_embedding/scaler_parameters.npz",
        "joint",
    )
    joint = load_csv_joint_labels(
        analysis_dir / "standardized_embedding/clustering", sections, selections
    )
    return DatasetBundle(
        dataset,
        display_name,
        run_dir,
        analysis_dir,
        sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        primary_label,
        joint,
        5,
        {
            "RNA": "reconstructed full-run Harmony RNA model-input features",
            "Protein": "reconstructed full-run Harmony protein model-input features",
            "preprocessing_scope": "full_dataset",
        },
    )


def load_mouse_thymus(maximum: int, seed: int) -> DatasetBundle:
    return load_adapted_rna_protein(
        "mouse_thymus",
        "Mouse Thymus",
        run_mouse_thymus,
        "Mouse_Thymus",
        maximum,
        seed,
    )


def load_simulation(maximum: int, seed: int) -> DatasetBundle:
    return load_adapted_rna_protein(
        "simulation",
        "Simulation",
        run_simulation,
        "Simulation",
        maximum,
        seed,
    )


def load_human(maximum: int, seed: int) -> DatasetBundle:
    run_dir = RESULT_ROOT / "human_embryo_harmony"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    sections = list(summary["section_order"])
    counts = {s: int(summary["final_embedding_shapes"][s][0]) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    metadata = section_manifest(sections, selections, {})
    meta_chunks = []
    for section in sections:
        path = Path(summary["preprocess_manifest"]["annotation_files"][section])
        header = pd.read_csv(path, nrows=0).columns
        wanted = [c for c in ("obs_name", "cellid", "celltype", "stage") if c in header]
        frame = pd.read_csv(path, usecols=wanted).iloc[selections[section]].reset_index(drop=True)
        id_col = "obs_name" if "obs_name" in frame else "cellid" if "cellid" in frame else None
        frame["spot_id"] = (
            frame[id_col].astype(str)
            if id_col is not None
            else [f"{section}:{i}" for i in selections[section]]
        )
        meta_chunks.append(frame)
    labels = pd.concat(meta_chunks, ignore_index=True)
    metadata = pd.concat([metadata, labels], axis=1)
    metadata["celltype"] = clean_labels(metadata["celltype"])

    input_arrays = {
        section: np.load(summary["preprocess_manifest"]["feature_files"][section], mmap_mode="r")
        for section in sections
    }
    input_features = {
        "RNA": scale_full_then_select(input_arrays, sections, selections)
    }
    final_arrays = {
        section: np.load(summary["saved_files"]["final_embeddings"][section], mmap_mode="r")
        for section in sections
    }
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(
        final_selected, run_dir / "analysis/cache/scaler_parameters.npz", "joint"
    )
    joint = load_npy_joint_labels(run_dir / "analysis/clustering", sections, selections)
    return DatasetBundle(
        "human_embryo",
        "Human embryo",
        run_dir,
        run_dir / "analysis",
        sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        "celltype",
        joint,
        10,
        {
            "RNA": "saved full-run Harmony RNA features; scaler fitted on all spots",
            "preprocessing_scope": "full_dataset",
        },
    )


def preprocess_misar_inputs(
    run_dir: Path, sections: list[str]
) -> dict[str, np.ndarray]:
    data_dir = PROJECT_ROOT / "data/MISAR-seq"
    backed_rna: dict[str, ad.AnnData] = {}
    backed_atac: dict[str, ad.AnnData] = {}
    try:
        for section in sections:
            backed_rna[section], backed_atac[section] = read_backed_pair(data_dir, section)
            make_unique_rna_var_names(section, backed_rna[section])
        shared_genes = common_var_names(backed_rna, sections, None, "RNA genes")
        shared_peaks = common_var_names(backed_atac, sections, None, "ATAC peaks")
        selected = {
            section: np.load(run_dir / f"selected_spot_indices_{section}.npy")
            for section in sections
        }
        rna = [
            subset_to_memory(backed_rna[s], selected[s], shared_genes) for s in sections
        ]
        atac = [
            subset_to_memory(backed_atac[s], selected[s], shared_peaks) for s in sections
        ]
    finally:
        for value in [*backed_rna.values(), *backed_atac.values()]:
            value.file.close()
    feature_dict, _, _ = load_cosie_style_data(
        {"RNA": rna, "ATAC": atac},
        n_comps=50,
        hvg_num=3000,
        hvg_num_by_modality={"RNA": 3000, "ATAC": 3000},
        target_sum=None,
        use_harmony=True,
        metacell=False,
        retain_processed=False,
    )
    result = {}
    for modality in ("RNA", "ATAC"):
        arrays = {
            section: feature_dict[f"s{i + 1}"][modality].detach().cpu().numpy()
            for i, section in enumerate(sections)
        }
        selections = {
            section: np.arange(len(arrays[section]), dtype=np.int64)
            for section in sections
        }
        result[modality] = scale_full_then_select(arrays, sections, selections)
    return result


def load_misar(maximum: int, seed: int) -> DatasetBundle:
    run_dir = RESULT_ROOT / "misar_seq/bidirectional_sparse_uot_fixed_lc0.1_seed42"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    sections = list(summary["section_names"])
    counts = {s: int(summary["final_embedding_shapes"][s][0]) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {s: str(summary["section_info"][s]["stage"]) for s in sections}
    metadata = section_manifest(sections, selections, aliases)
    chunks = []
    for section in sections:
        frame = pd.read_csv(run_dir / f"obs_metadata_{section}.csv").iloc[
            selections[section]
        ].reset_index(drop=True)
        frame["spot_id"] = frame["obs_name"].astype(str)
        frame = frame.drop(columns=["section"], errors="ignore")
        chunks.append(frame)
    metadata = pd.concat(
        [metadata, pd.concat(chunks, ignore_index=True)], axis=1
    )
    metadata["Combined_Clusters_annotation"] = clean_labels(
        metadata["Combined_Clusters_annotation"]
    )
    input_full = preprocess_misar_inputs(run_dir, sections)
    input_features = {}
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    for modality, values in input_full.items():
        chunks = [
            values[offsets[i] + selections[s]] for i, s in enumerate(sections)
        ]
        input_features[modality] = np.vstack(chunks).astype(np.float32)
    final_arrays = {
        s: np.load(run_dir / f"final_embeddings_{s}.npy", mmap_mode="r")
        for s in sections
    }
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(
        final_selected,
        run_dir / "analysis/standardized_embedding/scaler_parameters.npz",
        "joint",
    )
    joint = load_csv_joint_labels(
        run_dir / "analysis/standardized_embedding/clustering", sections, selections
    )
    return DatasetBundle(
        "misar_seq",
        "MISAR-seq",
        run_dir,
        run_dir / "analysis",
        sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        "Combined_Clusters_annotation",
        joint,
        5,
        {
            "RNA": "reconstructed full-run Harmony RNA model-input features",
            "ATAC": "reconstructed full-run Harmony ATAC model-input features",
            "preprocessing_scope": "full_dataset",
        },
    )


def preprocess_mousebrain_inputs(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    section_results = [
        build_mousebrain_section(
            section_id=section["section_id"],
            rna_path=section["rna_input"],
            metabolite_path=section["metabolite_input"],
            spatial_key=config["preprocessing"].get("spatial_key", "spatial"),
            uni_feature_key=config["preprocessing"].get("uni_feature_key", "uni_feature"),
            rna_gene_id_key=config["preprocessing"].get("rna_gene_id_key", "gene_ids"),
        )
        for section in config["sections"]
    ]
    data_dict = build_cosie_data_dict(section_results)
    feature_dict, _, _ = load_cosie_style_data(
        data_dict,
        n_comps=50,
        hvg_num=3000,
        hvg_num_by_modality={"RNA": 3000, "Metabolite": None},
        target_sum=None,
        use_harmony=True,
        metacell=False,
        retain_processed=False,
    )
    sections = [section["section_id"] for section in config["sections"]]
    result = {}
    for modality in ("RNA", "Metabolite", "HE"):
        arrays = {
            section: feature_dict[f"s{i + 1}"][modality].detach().cpu().numpy()
            for i, section in enumerate(sections)
        }
        selections = {
            section: np.arange(len(arrays[section]), dtype=np.int64)
            for section in sections
        }
        result[modality] = scale_full_then_select(arrays, sections, selections)
    return result, section_results


def load_mousebrain(maximum: int, seed: int) -> DatasetBundle:
    run_dir = RESULT_ROOT / "mousebrain/v3_bidirectional_sparse_fixed_lc0.1/epochs_200"
    analysis_dir = run_dir.parent / "analysis"
    config = json.loads((run_dir / "mousebrain_config_used.json").read_text())
    sections = list(config["section_order"])
    counts = {
        section: int(np.load(run_dir / f"final_embeddings/{section}_final_embedding.npy", mmap_mode="r").shape[0])
        for section in sections
    }
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {"s1": "SectionA", "s2": "SectionB", "s3": "SectionC"}
    metadata = section_manifest(sections, selections, aliases)
    input_full, section_results = preprocess_mousebrain_inputs(config)
    chunks = []
    for index, section in enumerate(sections):
        rna = section_results[index]["modalities"]["RNA"]
        frame = rna.obs.copy().iloc[selections[section]].reset_index(drop=True)
        frame["spot_id"] = rna.obs_names.astype(str).to_numpy()[selections[section]]
        chunks.append(frame)
    metadata = pd.concat(
        [metadata, pd.concat(chunks, ignore_index=True)], axis=1
    )
    metadata["RegionLoupe"] = clean_labels(metadata["RegionLoupe"])
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    input_features = {
        modality: np.vstack(
            [values[offsets[i] + selections[s]] for i, s in enumerate(sections)]
        ).astype(np.float32)
        for modality, values in input_full.items()
    }
    final_arrays = {
        s: np.load(run_dir / f"final_embeddings/{s}_final_embedding.npy", mmap_mode="r")
        for s in sections
    }
    final_all = np.vstack([np.asarray(final_arrays[s], dtype=np.float32) for s in sections])
    scaler = StandardScaler().fit(final_all)
    integrated_all = scaler.transform(final_all).astype(np.float32)
    integrated = np.vstack(
        [integrated_all[offsets[i] + selections[s]] for i, s in enumerate(sections)]
    )
    joint = {}
    for k in (5, 6, 8, 10):
        labels = KMeans(
            n_clusters=k, random_state=0, n_init=20, max_iter=300
        ).fit_predict(integrated_all).astype(np.int32)
        joint[k] = np.concatenate(
            [labels[offsets[i] + selections[s]] for i, s in enumerate(sections)]
        )
    return DatasetBundle(
        "mousebrain",
        "MouseBrain",
        run_dir,
        analysis_dir,
        sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        "RegionLoupe",
        joint,
        10,
        {
            "RNA": "reconstructed full-run Harmony RNA model-input features",
            "Metabolite": "reconstructed full-run Harmony metabolite model-input features",
            "HE": "reconstructed full-run Harmony HE model-input features",
            "preprocessing_scope": "full_dataset",
        },
    )


def load_spatch_modality_subset(
    modality: str,
    sections: list[str],
    selections: dict[str, np.ndarray],
) -> np.ndarray:
    filenames = {
        "section1": {
            "RNA": "adata_xenium_bin_filter.h5ad",
            "Protein": "adata_codex_bin_filter.h5ad",
            "HE": "adata_he.h5ad",
        },
        "section2": {
            "RNA": "adata_hd_filter.h5ad",
            "Protein": "adata_codex_filter.h5ad",
            "HE": "adata_he.h5ad",
        },
    }
    backed = [
        ad.read_h5ad(
            PROJECT_ROOT / "data/spatch" / section / filenames[section][modality],
            backed="r",
        )
        for section in sections
    ]
    try:
        var_names: list[str] | None
        if modality == "RNA":
            common = pd.Index(backed[0].var_names.astype(str))
            for value in backed[1:]:
                common = common.intersection(pd.Index(value.var_names.astype(str)))
            var_names = list(common)
        elif modality == "Protein":
            marker_file = RESULT_ROOT / "spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42/protein_markers_after_dapi_removal.csv"
            var_names = pd.read_csv(marker_file)["protein_marker"].astype(str).tolist()
        else:
            var_names = None
        sampled = []
        for value, section in zip(backed, sections):
            # h5py permits only one fancy index at a time for backed arrays.
            # Load the selected observations first, then select named variables
            # from the in-memory AnnData object.
            subset = subset_to_memory(value, selections[section], None)
            if var_names is not None:
                missing = sorted(set(var_names).difference(subset.var_names.astype(str)))
                if missing:
                    raise KeyError(
                        f"{modality} {section}: {len(missing)} requested variables "
                        f"are absent; examples={missing[:5]}"
                    )
                subset = subset[:, var_names].copy()
            sampled.append(subset)
    finally:
        for value in backed:
            value.file.close()
    hvg_by_modality = {"RNA": 3000, "Protein": None, "HE": None}
    feature_dict, _, _ = load_cosie_style_data(
        {modality: sampled},
        n_comps=50,
        hvg_num=3000,
        hvg_num_by_modality={modality: hvg_by_modality[modality]},
        target_sum=None,
        use_harmony=True,
        metacell=False,
        memory_efficient=True,
        retain_processed=False,
    )
    arrays = {
        section: feature_dict[f"s{i + 1}"][modality].detach().cpu().numpy()
        for i, section in enumerate(sections)
    }
    local = {
        section: np.arange(len(arrays[section]), dtype=np.int64)
        for section in sections
    }
    result = scale_full_then_select(arrays, sections, local)
    del feature_dict, sampled, arrays
    gc.collect()
    return result


def load_spatch(maximum: int, seed: int) -> DatasetBundle:
    run_dir = RESULT_ROOT / "spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    sections = list(summary["section_names"])
    counts = {s: int(summary["alignment"][s]["n_spots"]) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {"section1": "Section 1", "section2": "Section 2"}
    metadata = section_manifest(sections, selections, aliases)
    full_meta = pd.read_csv(
        run_dir / "spot_metadata.csv.gz",
        usecols=[
            "section", "original_rna_obs_name", "x", "y", "cell_type_common",
            "codex_coarse_label", "spatial_cluster", "high_quality",
        ],
        low_memory=False,
    )
    chunks = []
    for section in sections:
        frame = full_meta[full_meta["section"].eq(section)].reset_index(drop=True)
        frame = frame.iloc[selections[section]].reset_index(drop=True)
        frame["spot_id"] = [
            f"{section}:{int(x)}:{int(y)}" for x, y in zip(frame["x"], frame["y"])
        ]
        frame = frame.drop(columns=["section"], errors="ignore")
        chunks.append(frame)
    metadata = pd.concat(
        [metadata, pd.concat(chunks, ignore_index=True)], axis=1
    )
    metadata["cell_type_common"] = clean_labels(metadata["cell_type_common"])
    input_features = {}
    feature_cache_dir = run_dir / "analysis/umap/tables"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    expected_n = sum(len(selections[section]) for section in sections)
    for modality in ("RNA", "Protein", "HE"):
        cache_path = feature_cache_dir / f"preprocessed_input_{modality.lower()}_features.npy"
        if cache_path.exists():
            cached = np.load(cache_path, mmap_mode="r")
            if cached.ndim == 2 and cached.shape[0] == expected_n and cached.shape[1] > 1:
                print(f"[umap] reusing {cache_path}", flush=True)
                input_features[modality] = cached
                continue
        print(f"[umap] SPATCH sampled input preprocessing: {modality}", flush=True)
        features = load_spatch_modality_subset(modality, sections, selections)
        np.save(cache_path, np.asarray(features, dtype=np.float32))
        input_features[modality] = np.load(cache_path, mmap_mode="r")
    final_arrays = {
        s: np.load(run_dir / f"final_embeddings_{s}.npy", mmap_mode="r")
        for s in sections
    }
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(
        final_selected,
        run_dir / "analysis/standardized_embedding/scaler_parameters.npz",
        "combined",
    )
    joint = load_csv_joint_labels(
        run_dir / "analysis/standardized_embedding/clustering", sections, selections
    )
    return DatasetBundle(
        "spatch",
        "SPATCH",
        run_dir,
        run_dir / "analysis",
        sections,
        counts,
        selections,
        metadata,
        input_features,
        integrated,
        "cell_type_common",
        joint,
        5,
        {
            "RNA": "training-matched Harmony preprocessing fitted on the shared visualization sample",
            "Protein": "training-matched Harmony preprocessing fitted on the shared visualization sample",
            "HE": "training-matched Harmony preprocessing fitted on the shared visualization sample",
            "preprocessing_scope": "deterministic_visualization_sample",
        },
    )


LOADERS = {
    "human_embryo": load_human,
    "misar_seq": load_misar,
    "mousebrain": load_mousebrain,
    "spatch": load_spatch,
    "crc_stereocite": load_crc_stereocite,
    "human_lymph_node": load_human_lymph_node,
    "mouse_spleen": load_mouse_spleen,
    "mouse_thymus": load_mouse_thymus,
    "simulation": load_simulation,
}


def fit_umap(
    values: np.ndarray,
    cache_path: Path,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    overwrite: bool,
) -> np.ndarray:
    if cache_path.exists() and not overwrite:
        coordinates = np.load(cache_path)
        if coordinates.shape != (len(values), 2):
            raise ValueError(f"Stale UMAP cache shape at {cache_path}: {coordinates.shape}.")
        print(f"[umap] reusing {cache_path}", flush=True)
        return coordinates
    print(f"[umap] fitting {cache_path.stem}: n={len(values):,}, d={values.shape[1]}", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        init="spectral",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    coordinates = reducer.fit_transform(np.asarray(values, dtype=np.float32))
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid UMAP result for {cache_path}.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, coordinates)
    return coordinates


def color_lookup(
    values: np.ndarray,
    kind: str,
    custom: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    cleaned = clean_labels(values)
    categories = sorted(
        [value for value in np.unique(cleaned).tolist() if value != "NA"],
        key=natural_key,
    )
    if kind == "section":
        colors = SECTION_COLORS[: len(categories)]
    else:
        colors = categorical_palette(len(categories))
    lookup = dict(zip(categories, colors))
    if custom:
        lookup.update({str(key): value for key, value in custom.items() if str(key) in categories})
    if "NA" in cleaned:
        categories.append("NA")
        lookup["NA"] = MISSING_COLOR
    return categories, lookup


def scatter_panel(
    axis: Any,
    coordinates: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    lookup: dict[str, Any],
    title: str,
    point_size: float,
    seed: int,
    legend_below: bool = True,
    legend_ncol: int | None = None,
) -> None:
    cleaned = clean_labels(labels)
    order = np.random.default_rng(seed).permutation(len(coordinates))
    axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[lookup[value] for value in cleaned[order]],
        s=point_size,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    axis.set_title(title, fontsize=10)
    axis.set_aspect("equal", adjustable="datalim")
    axis.axis("off")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=4,
            color=lookup[value], label=str(value),
        )
        for value in categories
    ]
    ncol = (
        legend_ncol
        if legend_ncol is not None
        else max(1, min(6, math.ceil(len(categories) / 5)))
    )
    if legend_below:
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            frameon=False,
            fontsize=6,
            ncol=ncol,
            handletextpad=0.25,
            columnspacing=0.6,
        )
    else:
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=False,
            fontsize=7,
            ncol=ncol,
        )


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_individual(
    path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    kind: str,
    point_size: float,
    seed: int,
    custom: dict[str, Any] | None = None,
) -> None:
    categories, lookup = color_lookup(labels, kind, custom)
    maximum_label_length = max((len(str(value)) for value in categories), default=0)
    legend_on_right = len(categories) <= 10 and maximum_label_length <= 24
    if legend_on_right:
        figure, axis = plt.subplots(figsize=(8.2, 5.8))
        scatter_panel(
            axis,
            coordinates,
            labels,
            categories,
            lookup,
            "",
            point_size,
            seed,
            False,
            1,
        )
        figure.subplots_adjust(left=0.05, right=0.75, bottom=0.06, top=0.88)
    else:
        if maximum_label_length > 24:
            legend_ncol = min(4, max(1, len(categories)))
            figure_width = 10.5
        elif maximum_label_length > 14:
            legend_ncol = min(5, max(1, len(categories)))
            figure_width = 9.2
        else:
            legend_ncol = min(8, max(1, len(categories)))
            figure_width = 8.2
        legend_rows = max(1, math.ceil(len(categories) / legend_ncol))
        figure_height = 5.8 + min(2.2, 0.25 * legend_rows)
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        scatter_panel(
            axis,
            coordinates,
            labels,
            categories,
            lookup,
            "",
            point_size,
            seed,
            True,
            legend_ncol,
        )
        figure.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=min(0.43, 0.13 + 0.045 * legend_rows),
            top=0.88,
        )
    figure.suptitle(title, x=0.5, ha="center", fontsize=12)
    save_figure(figure, path)


def has_meaningful_biological_labels(bundle: DatasetBundle) -> bool:
    """Return true only when at least one biological annotation is available."""
    values = pd.Series(bundle.metadata[bundle.primary_label], dtype="object")
    normalized = values.astype(str).str.strip().str.lower()
    missing = normalized.isin(
        {"", "na", "n/a", "nan", "none", "null", "not_available", "not available"}
    )
    return bool((values.notna() & ~missing).any())


def plot_panel_c(
    bundle: DatasetBundle,
    input_coordinates: dict[str, np.ndarray],
    figures_dir: Path,
    point_size: float,
    seed: int,
) -> None:
    modalities = list(input_coordinates)
    section_values = bundle.metadata["section_label"].to_numpy()
    section_categories, section_lookup = color_lookup(section_values, "section")
    include_biology = has_meaningful_biological_labels(bundle)
    biological_values = bundle.metadata[bundle.primary_label].to_numpy()
    biological_categories, biological_lookup = (
        color_lookup(biological_values, "biology")
        if include_biology else ([], {})
    )
    many_biology_labels = include_biology and len(biological_categories) > 30
    columns_per_modality = 2 if include_biology else 1
    figure, axes = plt.subplots(
        1,
        columns_per_modality * len(modalities),
        figsize=(
            max(7.2, 7.2 * len(modalities)),
            7.7 if many_biology_labels else 6.7,
        ),
        squeeze=False,
    )
    for index, modality in enumerate(modalities):
        left = axes[0, columns_per_modality * index]
        scatter_panel(
            left, input_coordinates[modality], section_values,
            section_categories, section_lookup, f"{modality}\nSection label",
            point_size, seed + index,
        )
        if include_biology:
            right = axes[0, columns_per_modality * index + 1]
            scatter_panel(
                right, input_coordinates[modality], biological_values,
                biological_categories, biological_lookup,
                f"{modality}\n{bundle.primary_label} label",
                point_size, seed + index,
                legend_ncol=4 if many_biology_labels else None,
            )
    figure.suptitle(
        f"{bundle.display_name}: input-modality UMAP",
        x=0.5, ha="center", fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(
        bottom=0.39 if many_biology_labels else 0.27,
        wspace=0.08,
        top=0.88,
    )
    save_figure(figure, figures_dir / "panel_c_input_modality_umap.png")


def plot_panel_e(
    bundle: DatasetBundle,
    integrated_coordinates: np.ndarray,
    figures_dir: Path,
    point_size: float,
    seed: int,
    custom_biology: dict[str, Any] | None,
) -> None:
    k = bundle.representative_k
    if k not in bundle.joint_labels:
        raise KeyError(f"{bundle.name}: representative k={k} labels are unavailable.")
    section_values = bundle.metadata["section_label"].to_numpy()
    cluster_values = bundle.joint_labels[k].astype(str)
    biological_values = bundle.metadata[bundle.primary_label].to_numpy()
    sets = [
        (section_values, "section", "Section label", None),
        (cluster_values, "cluster", f"Joint cluster (k={k})", None),
    ]
    if has_meaningful_biological_labels(bundle):
        sets.append(
            (biological_values, "biology", f"{bundle.primary_label} label", custom_biology)
        )
    figure, axes = plt.subplots(
        1, len(sets), figsize=(6 * len(sets), 6.7), squeeze=False
    )
    for index, (labels, kind, title, custom) in enumerate(sets):
        categories, lookup = color_lookup(labels, kind, custom)
        scatter_panel(
            axes[0, index], integrated_coordinates, labels, categories, lookup,
            title, point_size, seed + index,
        )
    figure.suptitle(
        f"{bundle.display_name}: integrated embedding UMAP",
        x=0.5, ha="center", fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(bottom=0.27, wspace=0.08, top=0.88)
    save_figure(figure, figures_dir / "panel_e_integrated_embedding_umap.png")


def human_celltype_colors(bundle: DatasetBundle) -> dict[str, Any] | None:
    if bundle.name != "human_embryo":
        return None
    path = bundle.run_dir / "analysis/celltype_reference/celltype_color_map.csv"
    table = pd.read_csv(path)
    return dict(zip(table["celltype"].astype(str), table["color"].astype(str)))


def generate_dataset(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    output_dir = bundle.analysis_dir / "umap"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    n_obs = len(bundle.metadata)
    point_size = 2.0 if n_obs >= 50_000 else 8.0
    input_coordinates = {}
    for modality, values in bundle.input_features.items():
        validate_features(f"{bundle.name} input {modality}", values, n_obs)
        input_coordinates[modality] = fit_umap(
            values,
            tables_dir / f"input_{modality.lower()}_umap.npy",
            args.n_neighbors,
            args.min_dist,
            args.seed,
            args.overwrite_umap,
        )
    validate_features(f"{bundle.name} integrated", bundle.integrated_features, n_obs)
    integrated_coordinates = fit_umap(
        bundle.integrated_features,
        tables_dir / "integrated_embedding_umap.npy",
        args.n_neighbors,
        args.min_dist,
        args.seed,
        args.overwrite_umap,
    )
    custom_biology = human_celltype_colors(bundle)
    include_biology = has_meaningful_biological_labels(bundle)
    plot_panel_c(bundle, input_coordinates, figures_dir, point_size, args.seed)
    plot_panel_e(
        bundle, integrated_coordinates, figures_dir, point_size, args.seed,
        custom_biology,
    )

    section_values = bundle.metadata["section_label"].to_numpy()
    biology_values = bundle.metadata[bundle.primary_label].to_numpy()
    for offset, (modality, coordinates) in enumerate(input_coordinates.items()):
        plot_individual(
            figures_dir / f"input_{modality.lower()}_by_section.png",
            coordinates, section_values,
            f"{bundle.display_name} input {modality} UMAP by section",
            "section", point_size, args.seed + offset,
        )
        biological_path = (
            figures_dir / f"input_{modality.lower()}_by_{bundle.primary_label}.png"
        )
        if include_biology:
            plot_individual(
                biological_path, coordinates, biology_values,
                f"{bundle.display_name} input {modality} UMAP by {bundle.primary_label}",
                "biology", point_size, args.seed + offset, custom_biology,
            )
        else:
            biological_path.unlink(missing_ok=True)
    plot_individual(
        figures_dir / "integrated_by_section.png",
        integrated_coordinates, section_values,
        f"{bundle.display_name} integrated embedding UMAP by section",
        "section", point_size, args.seed,
    )
    integrated_biological_path = (
        figures_dir / f"integrated_by_{bundle.primary_label}.png"
    )
    if include_biology:
        plot_individual(
            integrated_biological_path, integrated_coordinates, biology_values,
            f"{bundle.display_name} integrated embedding UMAP by {bundle.primary_label}",
            "biology", point_size, args.seed, custom_biology,
        )
    else:
        integrated_biological_path.unlink(missing_ok=True)
    cluster_dir = figures_dir / "integrated_joint_clusters"
    for k, labels in sorted(bundle.joint_labels.items()):
        plot_individual(
            cluster_dir / f"integrated_by_joint_k{k}.png",
            integrated_coordinates, labels.astype(str),
            f"{bundle.display_name} integrated embedding UMAP by joint k={k}",
            "cluster", point_size, args.seed,
        )

    coordinates_table = bundle.metadata.copy()
    for modality, coordinates in input_coordinates.items():
        key = modality.lower()
        coordinates_table[f"input_{key}_UMAP1"] = coordinates[:, 0]
        coordinates_table[f"input_{key}_UMAP2"] = coordinates[:, 1]
    coordinates_table["integrated_UMAP1"] = integrated_coordinates[:, 0]
    coordinates_table["integrated_UMAP2"] = integrated_coordinates[:, 1]
    for k, labels in sorted(bundle.joint_labels.items()):
        coordinates_table[f"joint_k{k}"] = labels
    coordinate_path = tables_dir / "umap_coordinates_and_labels.csv.gz"
    coordinates_table.to_csv(coordinate_path, index=False, compression="gzip")
    sample_path = tables_dir / "sample_indices_by_section.npz"
    np.savez_compressed(sample_path, **bundle.selections)

    config = {
        "status": "PASS",
        "dataset": bundle.name,
        "display_name": bundle.display_name,
        "run_dir": bundle.run_dir,
        "analysis_dir": bundle.analysis_dir,
        "output_dir": output_dir,
        "panel_mapping": {
            "c": (
                "input-modality UMAPs colored by section and primary biological label"
                if include_biology else
                "input-modality UMAPs colored by section; biological annotation omitted because unavailable"
            ),
            "e": (
                "integrated final-embedding UMAP colored by section, representative joint cluster, and primary biological label"
                if include_biology else
                "integrated final-embedding UMAP colored by section and representative joint cluster; biological annotation omitted because unavailable"
            ),
        },
        "n_obs_total": sum(bundle.section_counts.values()),
        "n_obs_umap": n_obs,
        "sampled": n_obs < sum(bundle.section_counts.values()),
        "sampling": "section-proportional uniform without replacement",
        "section_counts_total": bundle.section_counts,
        "section_counts_umap": {s: len(bundle.selections[s]) for s in bundle.sections},
        "section_order": bundle.sections,
        "primary_biological_label": bundle.primary_label,
        "primary_biological_label_available": include_biology,
        "input_modalities": list(bundle.input_features),
        "input_provenance": bundle.input_provenance,
        "integrated_input": "joint StandardScaler-transformed saved final embedding",
        "umap": {
            "implementation": "umap-learn",
            "version": umap.__version__,
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "euclidean",
            "init": "spectral",
            "random_state": args.seed,
            "transform_seed": args.seed,
            "n_jobs": 1,
            "low_memory": True,
        },
        "joint_cluster_k_values": sorted(bundle.joint_labels),
        "representative_joint_k": bundle.representative_k,
        "representative_k_rule": "existing best/nearest retained biological-label result; MouseBrain uses the established retained k=10",
        "coordinates_table": coordinate_path,
        "sample_indices": sample_path,
        "figures": {
            "panel_c": figures_dir / "panel_c_input_modality_umap.png",
            "panel_e": figures_dir / "panel_e_integrated_embedding_umap.png",
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "script": Path(__file__).resolve(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output_dir / "umap_config.json", config)
    readme = [
        f"# {bundle.display_name} SpaMosaic-style UMAP", "",
        "- Panel c: model-input modality representations before spa_mo_model integration.",
        "- Panel e: saved final embedding after spa_mo_model integration.",
        f"- UMAP spots: {n_obs:,} of {sum(bundle.section_counts.values()):,}.",
        (
            f"- Primary biological label: `{bundle.primary_label}`."
            if include_biology else
            "- Primary biological annotation is unavailable; annotation-colored UMAPs are intentionally omitted."
        ),
        f"- Representative joint clustering: k={bundle.representative_k}.",
        "- UMAP is a qualitative visualization; integration claims should also use the saved quantitative metrics.",
        "",
        "All before/after panels use the exact same spot identities. Large datasets use a deterministic section-proportional sample.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[umap] completed {bundle.name}: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in selected:
        print(f"[umap] loading {name}", flush=True)
        bundle = LOADERS[name](args.max_samples, args.seed)
        generate_dataset(bundle, args)
        del bundle
        gc.collect()


if __name__ == "__main__":
    main()
