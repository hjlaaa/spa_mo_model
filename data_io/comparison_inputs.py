"""Preserved cross-method inputs/identity readers with explicit source paths."""
from __future__ import annotations
from .paired_comparison_inputs import (
    crc_stereocite_MethodData,
    crc_stereocite_aligned_coords,
    crc_stereocite_load_spa,
    crc_stereocite_load_mofa,
    crc_stereocite_load_cosie,
    crc_stereocite_load_spamosaic,
    human_lymph_node_MethodData,
    human_lymph_node_aligned_coords,
    human_lymph_node_load_spa,
    human_lymph_node_load_mofa,
    human_lymph_node_load_cosie,
    human_lymph_node_load_spamosaic,
    mouse_spleen_MethodData,
    mouse_spleen_aligned_coords,
    mouse_spleen_load_spa,
    mouse_spleen_load_mofa,
    mouse_spleen_load_cosie,
    mouse_spleen_load_spamosaic,
    CRC_STEREOCITE_SECTIONS,
    CRC_STEREOCITE_SHORT_NAMES,
    HUMAN_LYMPH_NODE_SECTIONS,
    MOUSE_SPLEEN_SECTIONS
)
from dataclasses import dataclass
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd


MOUSE_SPLEEN_SECTION_COUNTS = {'Mouse_Spleen1': 2568, 'Mouse_Spleen2': 2768}

MOUSE_SPLEEN_N_OBS = sum(MOUSE_SPLEEN_SECTION_COUNTS.values())

def mouse_spleen_validate(data: mouse_spleen_MethodData, allow_existing_output: bool=False) -> None:
    n_obs = len(data.embedding)
    if n_obs != MOUSE_SPLEEN_N_OBS or len(data.sections) != n_obs or len(data.barcodes) != n_obs or (len(data.coords) != n_obs):
        raise ValueError(f'{data.name}: expected/aligned {MOUSE_SPLEEN_N_OBS} rows, got {n_obs}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = {str(key): int(value) for (key, value) in pd.Series(data.sections).value_counts().items()}
    if counts != MOUSE_SPLEEN_SECTION_COUNTS:
        raise ValueError(f'{data.name}: unexpected section counts {counts}')
    if len(set(zip(data.sections, data.barcodes))) != n_obs:
        raise ValueError(f'{data.name}: duplicate section/barcode keys')
    if data.output_root.exists() and (not allow_existing_output):
        raise FileExistsError(f'refusing to overwrite {data.output_root}')
    if allow_existing_output and (not data.output_root.is_dir()):
        raise FileNotFoundError(f'{data.name}: missing existing comparison root {data.output_root}')
    missing = [str(path) for path in data.source_paths + list(data.shared_sources.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing sources {missing}')


CRC_STEREOCITE_SECTION_COUNTS = {'CRC_003_bin20': 166279, 'CRC_006_bin20': 446095}

CRC_STEREOCITE_N_OBS = sum(CRC_STEREOCITE_SECTION_COUNTS.values())

def crc_stereocite_validate(data: crc_stereocite_MethodData, allow_existing_output: bool=False) -> None:
    n_obs = len(data.embedding)
    if not n_obs == CRC_STEREOCITE_N_OBS == len(data.sections) == len(data.barcodes) == len(data.coords):
        raise ValueError(f'{data.name}: expected {CRC_STEREOCITE_N_OBS}, got {n_obs}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = {str(key): int(value) for (key, value) in pd.Series(data.sections).value_counts().items()}
    if counts != CRC_STEREOCITE_SECTION_COUNTS:
        raise ValueError(f'{data.name}: unexpected section counts {counts}')
    if len(set(zip(data.sections, data.barcodes))) != n_obs:
        raise ValueError(f'{data.name}: duplicate section/barcode keys')
    if data.output_root.exists() and (not allow_existing_output):
        raise FileExistsError(f'refusing to overwrite {data.output_root}')
    if allow_existing_output and (not data.output_root.is_dir()):
        raise FileNotFoundError(f'{data.name}: missing existing comparison root {data.output_root}')
    missing = [str(path) for path in data.source_paths + list(data.shared_sources.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing sources {missing}')

def crc_stereocite_sorted_order(data: crc_stereocite_MethodData) -> np.ndarray:
    rank = np.array([{section: i for (i, section) in enumerate(CRC_STEREOCITE_SECTIONS)}[x] for x in data.sections], dtype=int)
    return np.lexsort((data.barcodes, rank))

def crc_stereocite_validate_cross_method_alignment(methods: list[crc_stereocite_MethodData]) -> None:
    reference = methods[0]
    reference_order = crc_stereocite_sorted_order(reference)
    reference_sections = reference.sections[reference_order]
    reference_barcodes = reference.barcodes[reference_order]
    reference_coords = reference.coords[reference_order]
    for method in methods[1:]:
        order = crc_stereocite_sorted_order(method)
        if not np.array_equal(method.sections[order], reference_sections):
            raise ValueError(f'{method.name}: section universe mismatch')
        if not np.array_equal(method.barcodes[order], reference_barcodes):
            raise ValueError(f'{method.name}: barcode universe mismatch')
        if not np.allclose(method.coords[order], reference_coords):
            raise ValueError(f'{method.name}: spatial positions mismatch')


def human_lymph_node_validate(data: human_lymph_node_MethodData, allow_existing_output: bool=False) -> None:
    n = data.embedding.shape[0]
    if n != 6843 or len(data.sections) != n or len(data.barcodes) != n:
        raise ValueError(f'{data.name}: expected/aligned 6843 rows, got {n}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = dict(pd.Series(data.sections).value_counts())
    if counts != {'Human_Lymph_Node_A1': 3484, 'Human_Lymph_Node_D1': 3359}:
        raise ValueError(f'{data.name}: unexpected section counts {counts}')
    if len(set(zip(data.sections, data.barcodes))) != n:
        raise ValueError(f'{data.name}: duplicate section/barcode keys')
    if data.output_root.exists() and (not allow_existing_output):
        raise FileExistsError(f'refusing to overwrite {data.output_root}')
    missing = [str(path) for path in data.shared_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing shared sources {missing}')



_LEGACY_ANALYSIS_EXPORTS = ('mousebrain_MethodData', 'MOUSEBRAIN_SECTIONS', 'MOUSEBRAIN_GROUP_LABEL_KEY', 'MOUSEBRAIN_LABEL_KEYS', 'MOUSEBRAIN_SECTION_DIRS', 'mousebrain_aligned_metadata', 'mousebrain_load_spa', 'mousebrain_load_mofa', 'mousebrain_load_cosie', 'mousebrain_load_spamosaic', 'MOUSEBRAIN_SECTION_COUNTS', 'mousebrain_validate', 'simulation_MethodData', 'SIMULATION_SECTIONS', 'simulation_truth_and_coords', 'simulation_load_spa', 'simulation_load_mofa', 'simulation_load_cosie', 'simulation_load_spamosaic', 'simulation_validate_method', 'mouse_thymus_MethodData', 'MOUSE_THYMUS_SECTIONS', 'mouse_thymus_aligned_coords', 'mouse_thymus_load_spa', 'mouse_thymus_load_mofa', 'mouse_thymus_load_cosie', 'mouse_thymus_load_spamosaic', 'MOUSE_THYMUS_SECTION_COUNTS', 'MOUSE_THYMUS_N_OBS', 'mouse_thymus_validate', 'misar_seq_MethodData', 'MISAR_SEQ_SECTIONS', 'MISAR_SEQ_LABEL_KEYS', 'misar_seq_aligned_metadata', 'misar_seq_load_spa', 'misar_seq_load_mofa', 'misar_seq_load_cosie', 'misar_seq_load_spamosaic', 'MISAR_SEQ_SECTION_COUNTS', 'misar_seq_validate')

def __getattr__(name):
    if name in ('spatch_MethodSpec', 'SPATCH_LABELS', 'SPATCH_SECTIONS', 'SPATCH_SECTION_COUNTS', 'SPATCH_N_OBS', 'spatch_load_metadata', 'spatch_load_embedding', 'spatch_validate_metadata'):
        from analysis import spatch_readers
        return getattr(spatch_readers, name)
    if name in _LEGACY_ANALYSIS_EXPORTS:
        from analysis import comparison_readers
        return getattr(comparison_readers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
