"""Read existing analysis inputs without training, sorting, scaling or clustering.

These are the established dataset readers with explicit run/data/section inputs.
Returned dictionaries retain the fields used by the existing MethodData records.
No biological truth is synthesized for datasets whose readers had none.
"""
from __future__ import annotations

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from data_io.multisection_results import contract_payload, aligned_obs_metadata, read_full_rna_result, read_misar_result, read_simulation_result
from data_io.multisection_results import load_section_metadata as _load_section_metadata
from data_io.saved_assignments import SavedAssignmentLayout, load_saved_assignments
from data_io.analysis_results import PairedResultLayout, read_paired_result, aligned_coords as _aligned_coords


def clean_mousebrain_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
            continue
        text = str(value)
        if text.startswith("b'") and text.endswith("'"):
            text = text[2:-1]
        result[i] = text
    return result


def aligned_mousebrain_metadata(data_dir, sections, barcodes, *, section_order, label_keys, section_dirs, group_label, snapshots=None):
    coords, truth = aligned_obs_metadata(data_dir, sections, barcodes,
        section_order=section_order, section_dirs=section_dirs, label_keys=label_keys,
        clean_values=clean_mousebrain_truth, required_labels=False, snapshots=snapshots)
    truth[group_label] = sections.astype(str).copy()
    return coords, truth


def clean_misar_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
        else:
            text = str(value)
            result[i] = (
                text[2:-1]
                if text.startswith("b'") and text.endswith("'")
                else text
            )
    return result


def aligned_misar_metadata(data_dir, sections, barcodes, *, section_order, label_keys, snapshots=None):
    return aligned_obs_metadata(data_dir, sections, barcodes, section_order=section_order,
        section_dirs={s: s for s in section_order}, label_keys=label_keys,
        clean_values=clean_misar_truth, required_labels=True, snapshots=snapshots)


def aligned_crc_coords(data_dir, sections, barcodes, *, section_order):
    return _aligned_coords(data_dir, sections, barcodes, section_order=section_order)


def load_mousebrain(run_dir, data_dir, *, section_order, section_dirs, label_keys, group_label, name, output_root):
    embedding, sections, barcodes, coords, truth, sources = read_full_rna_result(
        run_dir, data_dir, section_order=section_order, section_dirs=section_dirs,
        label_keys=label_keys, clean_values=clean_mousebrain_truth, required_labels=False,
        error_template='{section}: embedding rows {rows} != {names}')
    truth[group_label] = sections.astype(str).copy()
    return contract_payload(name=name, embedding=embedding, sections=sections,
        barcodes=barcodes, coords=coords, truth=truth, data_dir=data_dir,
        source_paths=sources, output_root=output_root, shared_sources={})


def load_paired_rna_adt(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        section_order, {s: s for s in section_order}, validate_source_coords=False,
        unique_indices=False, source_paths_include_rna=True, saved_coords_output=True,
        error_template='{run_parent}/{section}: saved row counts do not align'))
    return result.legacy(name=name, data_dir=data_dir, output_root=output_root, shared_sources={})


def load_misar(run, data_dir, *, section_order, label_keys, name, output_root):
    return read_misar_result(run, data_dir, section_order=section_order, label_keys=label_keys,
        clean_values=clean_misar_truth, name=name, output_root=output_root)


def load_section_metadata(run, *, section_order):
    return _load_section_metadata(run, section_order=section_order)


def load_thymus(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    embedding, sections, barcodes, coords, sources = load_section_metadata(run, section_order=section_order)
    return contract_payload(name=name, embedding=embedding, sections=sections, barcodes=barcodes,
                coords=coords, data_dir=data_dir, source_paths=sources,
                output_root=output_root, shared_sources={})


def load_simulation(run, data_dir, *, section_order, name, output_root):
    return read_simulation_result(run, data_dir, section_order=section_order, name=name, output_root=output_root)


def load_crc(run: Path, data_dir: Path, *, section_order, short_names, output_root: Path):
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        section_order, short_names, spatial_before_rna=True,
        error_template='CRC {section}: embedding/index/spatial mismatch'))
    return result.legacy(name='spa_mo_model', data_dir=data_dir, output_root=output_root, shared_sources={})


def load_joint_assignments(analysis_dir: Path, k: int, section_values, barcodes):
    """Compatibility projection of the explicit legacy joint-export layout."""
    directory = Path(analysis_dir) / "clustering" / f"joint_k{k}"
    result = load_saved_assignments(SavedAssignmentLayout(
        directory / "labels_all.csv", directory / "labels_all.npy",
        Path(analysis_dir) / "all_k_labels_and_fit_cache.zip",
        f"joint_combined_k{k}.npy", directory), section_values, barcodes)
    return result.labels, result.source_paths
