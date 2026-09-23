"""Existing joint-label formats and identity checks; no clustering or metric policy."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SavedAssignmentLayout:
    all_csv: Path
    all_npy: Path
    archive: Path
    archive_member: str
    section_csv_dir: Path


@dataclass
class SavedAssignments:
    labels: np.ndarray
    source_paths: list
    identity_evidence: list


def load_saved_assignments(layout: SavedAssignmentLayout, section_values, barcodes):
    """CSV→NPY→archive→per-section CSV precedence; absent IDs remain unverified.

    `verified_columns=[]` denotes row-order evidence only. It does not make the
    requested barcode vector into evidence about the file. Per-section CSVs
    historically do not validate their optional section column; preserve that.
    """
    section_values = np.asarray(section_values).astype(str)
    barcodes = np.asarray(barcodes).astype(str)
    directory = layout.section_csv_dir
    evidence = []
    path = layout.all_csv
    if path.is_file():
        frame = read_assignment_table(path, dtype={"section": str, "obs_name": str, "spot_id": str})
        if "section" in frame and not np.array_equal(frame["section"].to_numpy(), section_values):
            raise ValueError(f"Joint label section order differs: {path}")
        for key in ("obs_name", "spot_id"):
            if key in frame and not np.array_equal(frame[key].to_numpy(), barcodes):
                raise ValueError(f"Joint label spot order differs: {path}")
        labels = frame["cluster"].to_numpy()
        sources = [path]
        evidence.append({'source': path, 'verified_columns': [k for k in ('section','obs_name','spot_id') if k in frame]})
    elif layout.all_npy.is_file():
        path = layout.all_npy
        labels = read_assignment_array(path)
        sources = [path]
        evidence.append({'source': path, 'verified_columns': []})
    elif layout.archive.is_file():
        import zipfile
        path = layout.archive
        with zipfile.ZipFile(path) as archive:
            with archive.open(layout.archive_member) as handle:
                labels = read_assignment_array(handle)
        sources = [path]
        evidence.append({'source': path, 'verified_columns': [], 'member': layout.archive_member})
    else:
        labels = np.empty(len(section_values), dtype=np.int64)
        sources = []
        for section in dict.fromkeys(section_values):
            mask = section_values == section
            path = directory / f"labels_{section}.csv"
            frame = read_assignment_table(path, dtype={"section": str, "obs_name": str, "spot_id": str})
            for key in ("obs_name", "spot_id"):
                if key in frame and not np.array_equal(frame[key].to_numpy(), barcodes[mask]):
                    raise ValueError(f"Joint label spot order differs: {path}")
            if len(frame) != int(mask.sum()):
                raise ValueError(f"Joint label row count differs: {path}")
            labels[mask] = frame["cluster"].to_numpy()
            sources.append(path)
            evidence.append({'source': path, 'section_from_layout': section, 'verified_columns': [k for k in ('obs_name','spot_id') if k in frame]})
    if labels.shape != (len(section_values),):
        raise ValueError("Joint assignment shape differs from loaded spots")
    return SavedAssignments(labels, sources, evidence)



def read_assignment_table(path, **read_options):
    """Read the explicitly chosen CSV layout; caller owns identity/column policy."""
    return pd.read_csv(path, **read_options)


def read_assignment_array(path, **load_options):
    """Read an explicit NPY/file-like archive member; preserve mmap/dtype options."""
    return np.load(path, **load_options)


def selected_csv_assignments(table, expected_ids, local, *, missing_context):
    id_column = 'spot_id' if 'spot_id' in table else 'obs_name'
    ids = table[id_column].astype(str).to_numpy()
    if local.max(initial=-1) < len(table) and np.array_equal(ids[local], expected_ids):
        return table['cluster'].to_numpy(dtype=np.int32)[local]
    positions = pd.Index(ids).get_indexer(expected_ids)
    if np.any(positions < 0):
        raise KeyError(f'{missing_context}: {np.sum(positions < 0)} spots missing.')
    return table['cluster'].to_numpy(dtype=np.int32)[positions]
