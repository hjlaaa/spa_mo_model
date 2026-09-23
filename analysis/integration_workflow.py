"""Shared input assembly for existing integration visualizations.

Callers supply paths, section order, selected rows and saved-scaler layout.
Scientific modality preparation, cohort selection and output policy stay with
their existing callers. No UMAP, clustering, metric or training execution here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from data_io.large_results import read_section_arrays
from data_io.saved_assignments import read_assignment_table, read_assignment_array
from analysis.clustering import stack_selected, apply_saved_scaler

def section_manifest(sections: list[str], selections: dict[str, np.ndarray], aliases: dict[str, str]) -> pd.DataFrame:
    frames = []
    offset = 0
    for section in sections:
        index = selections[section]
        frames.append(pd.DataFrame({'sample_index': np.arange(offset, offset + len(index)), 'section': section, 'section_label': aliases.get(section, section), 'section_row_index': index}))
        offset += len(index)
    return pd.concat(frames, ignore_index=True)

def load_npy_joint_labels(clustering_dir: Path, sections: list[str], selections: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    result = {}
    for directory in sorted(clustering_dir.glob('joint_k*')):
        try:
            k = int(directory.name.split('joint_k', 1)[1])
        except ValueError:
            continue
        chunks = []
        for section in sections:
            labels = read_assignment_array(directory / f'labels_{section}.npy', mmap_mode='r')
            chunks.append(np.asarray(labels[selections[section]], dtype=np.int32))
        result[k] = np.concatenate(chunks)
    return result

def load_csv_joint_labels(clustering_dir: Path, sections: list[str], selections: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    result = {}
    for directory in sorted(clustering_dir.glob('joint_k*')):
        try:
            k = int(directory.name.split('joint_k', 1)[1])
        except ValueError:
            continue
        chunks = []
        for section in sections:
            table = read_assignment_table(directory / f'labels_{section}.csv')
            if 'cluster' not in table:
                raise KeyError(f'{directory}: missing cluster column for {section}.')
            chunks.append(table['cluster'].to_numpy(dtype=np.int32)[selections[section]])
        result[k] = np.concatenate(chunks)
    return result


def append_selected_metadata(metadata, chunks):
    """Original row-wise frames then column-wise manifest join; no re-alignment."""
    return pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)


def select_modality_rows(input_full, sections, selections, offsets):
    """Select from already prepared full-run matrices, in supplied modality order."""
    result = {}
    for modality, values in input_full.items():
        chunks = [values[offsets[i] + selections[s]] for i, s in enumerate(sections)]
        result[modality] = np.vstack(chunks).astype(np.float32)
    return result


def load_saved_integration(embedding_paths, sections, selections, *, scaler_path,
                           scaler_prefix, clustering_dir, label_format):
    """Read once, select once, apply the existing saved scaler, then read labels.

    CSV/NPY is an explicit artifact layout, not a dataset or method decision.
    These legacy labels carry row-order evidence; no barcode is synthesized.
    """
    final_arrays = read_section_arrays(embedding_paths)
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(final_selected, scaler_path, scaler_prefix)
    if label_format == 'csv':
        joint = load_csv_joint_labels(clustering_dir, sections, selections)
    elif label_format == 'npy':
        joint = load_npy_joint_labels(clustering_dir, sections, selections)
    else:
        raise ValueError(f'Unsupported saved assignment layout: {label_format}')
    return integrated, joint
