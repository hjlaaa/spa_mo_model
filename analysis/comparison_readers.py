"""Analysis-owned format adapters and unchanged cohort/truth policy."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.protocols import MOUSEBRAIN, MISAR
from analysis.metrics import comparison_valid_truth
from analysis.loaders import aligned_mousebrain_metadata, aligned_misar_metadata
from data_io import external_result_formats as formats
from data_io.reference_results import stack_embedding_files, validate_embedding_table
from data_io.multisection_results import aligned_obs_xy, open_aligned_rna_metadata

@dataclass
class mousebrain_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    truth: dict[str, np.ndarray]
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

MOUSEBRAIN_SECTIONS = ['s1', 's2', 's3']

MOUSEBRAIN_GROUP_LABEL_KEY = 'group'

MOUSEBRAIN_LABEL_KEYS = MOUSEBRAIN['LABEL_KEYS']

MOUSEBRAIN_SECTION_DIRS = {'s1': 'dataset_MouseBrain_SectionA', 's2': 'dataset_MouseBrain_SectionB', 's3': 'dataset_MouseBrain_SectionC'}

def mousebrain_aligned_metadata(data_dir, sections, barcodes, *, snapshots=None):
    return aligned_mousebrain_metadata(data_dir, sections, barcodes, section_order=MOUSEBRAIN_SECTIONS,
        label_keys=MOUSEBRAIN_LABEL_KEYS, section_dirs=MOUSEBRAIN_SECTION_DIRS,
        group_label=MOUSEBRAIN_GROUP_LABEL_KEY, snapshots=snapshots)

MOUSEBRAIN_SECTION_COUNTS = {'s1': 2384, 's2': 2820, 's3': 2662}

def mousebrain_validate(data: mousebrain_MethodData, allow_existing_output: bool=False) -> None:
    n = len(data.embedding)
    if n != 7866 or len(data.sections) != n or len(data.barcodes) != n:
        raise ValueError(f'{data.name}: expected/aligned 7866 rows, got {n}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = {key: int(value) for (key, value) in pd.Series(data.sections).value_counts().items()}
    if counts != MOUSEBRAIN_SECTION_COUNTS:
        raise ValueError(f'{data.name}: unexpected section counts {counts}')
    if len(set(zip(data.sections, data.barcodes))) != n:
        raise ValueError(f'{data.name}: duplicate section/barcode keys')
    for label in MOUSEBRAIN_LABEL_KEYS:
        valid = comparison_valid_truth(data.truth[label])
        if valid.sum() < 2 or len(np.unique(data.truth[label][valid])) < 2:
            raise ValueError(f'{data.name}: invalid ground truth {label}')
    if data.output_root.exists() and (not allow_existing_output):
        raise FileExistsError(f'refusing to overwrite {data.output_root}')
    missing = [str(path) for path in data.source_paths + list(data.shared_sources.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing sources {missing}')

@dataclass
class simulation_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    truth: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

SIMULATION_SECTIONS = [f'Simulation{i}' for i in range(1, 6)]

def _simulation_truth(rna, pos, rna_path):
    if 'spatial_domain' in rna.obs:
        sec_truth = rna.obs['spatial_domain'].astype(str).to_numpy()[pos]
    elif 'spfac' in rna.obsm:
        spfac = np.asarray(rna.obsm['spfac'])[pos]
        sec_truth = np.full(len(pos), 'background', dtype=object)
        active = spfac.sum(axis=1) > 0
        sec_truth[active] = [f'sp{i + 1}' for i in spfac[active].argmax(axis=1)]
    else:
        raise ValueError(f'{rna_path}: no spatial_domain or spfac truth.')
    return sec_truth


def simulation_truth_and_coords(data_dir, sections, barcodes):
    truth = np.empty(len(sections), dtype=object)
    coords = np.empty((len(sections), 2), dtype=float)
    for section in SIMULATION_SECTIONS:
        mask = sections == section
        path = data_dir / section / 'adata_RNA.h5ad'
        with open_aligned_rna_metadata(path, barcodes, mask) as (rna, positions):
            section_truth = _simulation_truth(rna, positions, path)
            spatial = np.asarray(rna.obsm['spatial'])[positions, :2]
        truth[mask] = section_truth
        coords[mask] = spatial
    return truth.astype(str), coords

def simulation_validate_method(data: simulation_MethodData, allow_existing_output: bool=False) -> None:
    n = data.embedding.shape[0]
    if len(data.sections) != n or len(data.barcodes) != n or len(data.truth) != n or (len(data.coords) != n):
        raise ValueError(f'{data.name}: row counts are inconsistent.')
    if n != 6480:
        raise ValueError(f'{data.name}: expected 6480 spots, observed {n}.')
    if not np.isfinite(data.embedding).all():
        raise ValueError(f'{data.name}: embedding contains non-finite values.')
    if set(np.unique(data.sections)) != set(SIMULATION_SECTIONS):
        raise ValueError(f'{data.name}: section set is inconsistent.')
    if set(np.unique(data.truth)) != {'background', 'sp1', 'sp2', 'sp3', 'sp4'}:
        raise ValueError(f'{data.name}: spatial-domain truth is inconsistent.')
    if data.output_root.exists() and not allow_existing_output:
        raise FileExistsError(f'Refusing to overwrite existing comparison root: {data.output_root}')
    missing = [str(path) for path in data.shared_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing shared sources: {missing}')

@dataclass
class mouse_thymus_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

MOUSE_THYMUS_SECTIONS = [f'Mouse_Thymus{i}' for i in range(1, 5)]

def mouse_thymus_aligned_coords(data_dir, sections, barcodes, *, snapshots=None):
    return aligned_obs_xy(data_dir, sections, barcodes, section_order=MOUSE_THYMUS_SECTIONS, snapshots=snapshots)

MOUSE_THYMUS_SECTION_COUNTS = {'Mouse_Thymus1': 4697, 'Mouse_Thymus2': 4253, 'Mouse_Thymus3': 4646, 'Mouse_Thymus4': 4228}

MOUSE_THYMUS_N_OBS = sum(MOUSE_THYMUS_SECTION_COUNTS.values())

def mouse_thymus_validate(data: mouse_thymus_MethodData, allow_existing_output: bool=False) -> None:
    n_obs = len(data.embedding)
    if n_obs != MOUSE_THYMUS_N_OBS or len(data.sections) != n_obs or len(data.barcodes) != n_obs or (len(data.coords) != n_obs):
        raise ValueError(f'{data.name}: expected/aligned {MOUSE_THYMUS_N_OBS} rows, got {n_obs}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = {str(key): int(value) for (key, value) in pd.Series(data.sections).value_counts().items()}
    if counts != MOUSE_THYMUS_SECTION_COUNTS:
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

@dataclass
class misar_seq_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    truth: dict[str, np.ndarray]
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

MISAR_SEQ_SECTIONS = ['dataset1', 'dataset2', 'dataset3', 'dataset4']

MISAR_SEQ_LABEL_KEYS = MISAR['LABEL_KEYS']

def misar_seq_aligned_metadata(data_dir, sections, barcodes, *, snapshots=None):
    return aligned_misar_metadata(data_dir, sections, barcodes, section_order=MISAR_SEQ_SECTIONS,
        label_keys=MISAR_SEQ_LABEL_KEYS, snapshots=snapshots)

MISAR_SEQ_SECTION_COUNTS = {'dataset1': 2129, 'dataset2': 1949, 'dataset3': 1777, 'dataset4': 1263}

def misar_seq_validate(data: misar_seq_MethodData, allow_existing_output: bool=False) -> None:
    n_obs = len(data.embedding)
    if n_obs != 7118 or len(data.sections) != n_obs or len(data.barcodes) != n_obs:
        raise ValueError(f'{data.name}: expected/aligned 7118 rows, got {n_obs}')
    if not np.isfinite(data.embedding).all() or not np.isfinite(data.coords).all():
        raise ValueError(f'{data.name}: non-finite embedding or coordinates')
    counts = {key: int(value) for (key, value) in pd.Series(data.sections).value_counts().items()}
    if counts != MISAR_SEQ_SECTION_COUNTS:
        raise ValueError(f'{data.name}: unexpected section counts {counts}')
    if len(set(zip(data.sections, data.barcodes))) != n_obs:
        raise ValueError(f'{data.name}: duplicate section/barcode keys')
    for label in MISAR_SEQ_LABEL_KEYS:
        valid = comparison_valid_truth(data.truth[label])
        if valid.sum() != 7118 or len(np.unique(data.truth[label][valid])) < 2:
            raise ValueError(f'{data.name}: label {label} is incomplete or invalid')
    if data.output_root.exists() and (not allow_existing_output):
        raise FileExistsError(f'refusing to overwrite {data.output_root}')
    if allow_existing_output and (not data.output_root.is_dir()):
        raise FileNotFoundError(f'{data.name}: missing existing comparison root {data.output_root}')
    missing = [str(path) for path in data.source_paths + list(data.shared_sources.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f'{data.name}: missing sources {missing}')

def mousebrain_load_spa(paths, output_root) -> mousebrain_MethodData:
    run, data_dir, arrays, section_arr, barcode_arr, sources, snapshots = formats.mousebrain_load_spa(paths, output_root, section_order=MOUSEBRAIN_SECTIONS, section_dirs=MOUSEBRAIN_SECTION_DIRS, label_keys=MOUSEBRAIN_LABEL_KEYS)
    coords, truth = mousebrain_aligned_metadata(data_dir, section_arr, barcode_arr, snapshots=snapshots)
    old = run / 'analysis'
    return mousebrain_MethodData(**formats._mousebrain_payload('spa_mo_model', np.vstack(arrays), section_arr, barcode_arr, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'clustering' / 'batch_correction_metrics.csv'}))

def mousebrain_load_mofa(paths, output_root) -> mousebrain_MethodData:
    old, table_path, table, factor_cols, sections, barcodes, data_dir = formats.mousebrain_load_mofa(paths, output_root, section_order=MOUSEBRAIN_SECTIONS, section_dirs=MOUSEBRAIN_SECTION_DIRS, label_keys=MOUSEBRAIN_LABEL_KEYS)
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    return mousebrain_MethodData(**formats._mousebrain_payload('MOFA+', table[factor_cols].to_numpy(float), sections, barcodes, coords, truth, data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'}))

def mousebrain_load_cosie(paths, output_root) -> mousebrain_MethodData:
    run, old, table_path, table, emb_cols, sections, barcodes, data_dir = formats.mousebrain_load_cosie(paths, output_root, section_order=MOUSEBRAIN_SECTIONS, section_dirs=MOUSEBRAIN_SECTION_DIRS, label_keys=MOUSEBRAIN_LABEL_KEYS)
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MOUSEBRAIN_SECTIONS]
    final_embedding = stack_embedding_files(final_paths)
    table_embedding = table[emb_cols].to_numpy(float)
    validate_embedding_table(final_embedding, table_embedding, error='COSIE: analysis table is not the saved final embedding')
    sources = [table_path, *final_paths]
    return mousebrain_MethodData(**formats._mousebrain_payload('COSIE', table_embedding, sections, barcodes, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'}))

def mousebrain_load_spamosaic(paths, output_root) -> mousebrain_MethodData:
    h5ad_path, adata, sections, barcodes, embedding, data_dir = formats.mousebrain_load_spamosaic(paths, output_root, section_order=MOUSEBRAIN_SECTIONS, section_dirs=MOUSEBRAIN_SECTION_DIRS, label_keys=MOUSEBRAIN_LABEL_KEYS)
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return mousebrain_MethodData(**formats._mousebrain_payload('SpaMosaic', embedding, sections, barcodes, coords, truth, data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'}))

def simulation_load_spa(paths, output_root) -> simulation_MethodData:
    run, data_dir, arrays, section_arr, barcode_arr, sources = formats.simulation_load_spa(paths, output_root, section_order=SIMULATION_SECTIONS)
    (truth, coords) = simulation_truth_and_coords(data_dir, section_arr, barcode_arr)
    old_analysis = run / 'clustering_analysis'
    return simulation_MethodData(**formats._simulation_payload(name='spa_mo_model', embedding=np.vstack(arrays), sections=section_arr, barcodes=barcode_arr, truth=truth, coords=coords, data_dir=data_dir, source_paths=sources, output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old_analysis / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old_analysis / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old_analysis / 'batch_correction_metrics.csv'}))

def simulation_load_mofa(paths, output_root) -> simulation_MethodData:
    old, table_path, table, factor_cols, sections, barcodes, data_dir = formats.simulation_load_mofa(paths, output_root, section_order=SIMULATION_SECTIONS)
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    return simulation_MethodData(**formats._simulation_payload(name='MOFA+', embedding=table[factor_cols].to_numpy(dtype=float), sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[table_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'}))

def simulation_load_cosie(paths, output_root) -> simulation_MethodData:
    old, table_path, table, emb_cols, sections, barcodes, data_dir = formats.simulation_load_cosie(paths, output_root, section_order=SIMULATION_SECTIONS)
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    return simulation_MethodData(**formats._simulation_payload(name='COSIE', embedding=table[emb_cols].to_numpy(dtype=float), sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[table_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'}))

def simulation_load_spamosaic(paths, output_root) -> simulation_MethodData:
    h5ad_path, adata, sections, barcodes, embedding, data_dir = formats.simulation_load_spamosaic(paths, output_root, section_order=SIMULATION_SECTIONS)
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return simulation_MethodData(**formats._simulation_payload(name='SpaMosaic', embedding=embedding, sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[h5ad_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'}))

def mouse_thymus_load_spa(paths, output_root) -> mouse_thymus_MethodData:
    return mouse_thymus_MethodData(**formats.mouse_thymus_load_spa(paths, output_root, align=mouse_thymus_aligned_coords, section_order=MOUSE_THYMUS_SECTIONS))

def mouse_thymus_load_mofa(paths, output_root) -> mouse_thymus_MethodData:
    return mouse_thymus_MethodData(**formats.mouse_thymus_load_mofa(paths, output_root, align=mouse_thymus_aligned_coords, section_order=MOUSE_THYMUS_SECTIONS))

def mouse_thymus_load_cosie(paths, output_root) -> mouse_thymus_MethodData:
    return mouse_thymus_MethodData(**formats.mouse_thymus_load_cosie(paths, output_root, align=mouse_thymus_aligned_coords, section_order=MOUSE_THYMUS_SECTIONS))

def mouse_thymus_load_spamosaic(paths, output_root) -> mouse_thymus_MethodData:
    return mouse_thymus_MethodData(**formats.mouse_thymus_load_spamosaic(paths, output_root, align=mouse_thymus_aligned_coords, section_order=MOUSE_THYMUS_SECTIONS))

def misar_seq_load_spa(paths, output_root) -> misar_seq_MethodData:
    return misar_seq_MethodData(**formats.misar_seq_load_spa(paths, output_root, align=misar_seq_aligned_metadata, section_order=MISAR_SEQ_SECTIONS, label_keys=MISAR_SEQ_LABEL_KEYS))

def misar_seq_load_mofa(paths, output_root) -> misar_seq_MethodData:
    return misar_seq_MethodData(**formats.misar_seq_load_mofa(paths, output_root, align=misar_seq_aligned_metadata, section_order=MISAR_SEQ_SECTIONS, label_keys=MISAR_SEQ_LABEL_KEYS))

def misar_seq_load_cosie(paths, output_root) -> misar_seq_MethodData:
    return misar_seq_MethodData(**formats.misar_seq_load_cosie(paths, output_root, align=misar_seq_aligned_metadata, section_order=MISAR_SEQ_SECTIONS, label_keys=MISAR_SEQ_LABEL_KEYS))

def misar_seq_load_spamosaic(paths, output_root) -> misar_seq_MethodData:
    return misar_seq_MethodData(**formats.misar_seq_load_spamosaic(paths, output_root, align=misar_seq_aligned_metadata, section_order=MISAR_SEQ_SECTIONS, label_keys=MISAR_SEQ_LABEL_KEYS))

mousebrain_MethodData.__module__ = 'data_io.comparison_inputs'

simulation_MethodData.__module__ = 'data_io.comparison_inputs'

mouse_thymus_MethodData.__module__ = 'data_io.comparison_inputs'

misar_seq_MethodData.__module__ = 'data_io.comparison_inputs'
