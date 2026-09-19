"""Preserved cross-method inputs/identity readers with explicit source paths."""
from __future__ import annotations
from analysis.inputs import load_spatch_embedding
from analysis.inputs import load_spatch_metadata
from analysis.loaders import aligned_crc_coords
from analysis.loaders import aligned_misar_metadata
from analysis.loaders import aligned_mousebrain_metadata
from analysis.metrics import comparison_valid_truth
from analysis.protocols import MISAR
from analysis.protocols import MOUSEBRAIN
from analysis.protocols import SPATCH
from dataclasses import dataclass
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

@dataclass
class mouse_spleen_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

MOUSE_SPLEEN_SECTIONS = ['Mouse_Spleen1', 'Mouse_Spleen2']

def mouse_spleen_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray) -> np.ndarray:
    """Align spatial coordinates by the method-local section/barcode keys."""
    coords = np.empty((len(sections), 2), dtype=float)
    for section in MOUSE_SPLEEN_SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            positions = source.get_indexer(wanted)
            if (positions < 0).any():
                examples = wanted[positions < 0][:5].tolist()
                raise ValueError(f'{rna_path}: {int((positions < 0).sum())} barcodes failed alignment; examples={examples}')
            coords[mask] = np.asarray(rna.obsm['spatial'])[positions, :2]
        finally:
            rna.file.close()
    return coords

def mouse_spleen_load_spa(paths, output_root) -> mouse_spleen_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in MOUSE_SPLEEN_SECTIONS:
        embedding_path = run / f'final_embeddings_{section}.npy'
        indices_path = run / f'selected_spot_indices_{section}.npy'
        spatial_path = run / f'spatial_{section}.npy'
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()[indices]
            own_coords = np.asarray(rna.obsm['spatial'])[indices, :2]
        finally:
            rna.file.close()
        saved_coords = np.load(spatial_path)[:, :2]
        if len(embedding) != len(indices) or len(np.unique(indices)) != len(indices) or (not np.allclose(saved_coords, own_coords)):
            raise ValueError(f'spa_mo_model {section}: embedding/index/spatial mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(indices))
        barcodes.extend(own_names.tolist())
        sources.extend([embedding_path, indices_path, spatial_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / 'clustering_analysis'
    return mouse_spleen_MethodData('spa_mo_model', np.vstack(arrays), section_array, barcode_array, mouse_spleen_aligned_coords(data_dir, section_array, barcode_array), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def mouse_spleen_load_mofa(paths, output_root) -> mouse_spleen_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = pd.read_csv(table_path)
    factor_columns = [column for column in table.columns if column.startswith('Factor')]
    sections = table['section'].astype(str).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    return mouse_spleen_MethodData('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv', 'input_audit.csv': old / 'input_audit.csv'})

def mouse_spleen_load_cosie(paths, output_root) -> mouse_spleen_MethodData:
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path)
    embedding_columns = [column for column in table.columns if column.startswith('COSIE')]
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MOUSE_SPLEEN_SECTIONS]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    if final_embedding.shape != table_embedding.shape or not np.allclose(final_embedding, table_embedding):
        raise ValueError('COSIE: analysis table is not the saved final embedding')
    sections = table['section'].astype(str).to_numpy()
    barcode_column = 'obs_name.1' if 'obs_name.1' in table.columns else 'obs_name'
    barcodes = table[barcode_column].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    return mouse_spleen_MethodData('COSIE', table_embedding, sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def mouse_spleen_load_spamosaic(paths, output_root) -> mouse_spleen_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return mouse_spleen_MethodData('SpaMosaic', embedding, sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

def mousebrain_aligned_metadata(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray):
    return aligned_mousebrain_metadata(data_dir, sections, barcodes, section_order=MOUSEBRAIN_SECTIONS, label_keys=MOUSEBRAIN_LABEL_KEYS, section_dirs=MOUSEBRAIN_SECTION_DIRS, group_label=MOUSEBRAIN_GROUP_LABEL_KEY)

def mousebrain_load_spa(paths, output_root) -> mousebrain_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    (arrays, sections, barcodes, sources) = ([], [], [], [])
    for section in MOUSEBRAIN_SECTIONS:
        emb_path = run / 'final_embeddings' / f'{section}_final_embedding.npy'
        rna_path = data_dir / MOUSEBRAIN_SECTION_DIRS[section] / 'adata_RNA.h5ad'
        embedding = np.load(emb_path)
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(f'spa_mo_model {section}: embedding/data row mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.append(emb_path)
    section_arr = np.asarray(sections)
    barcode_arr = np.asarray(barcodes)
    (coords, truth) = mousebrain_aligned_metadata(data_dir, section_arr, barcode_arr)
    old = run / 'analysis'
    return mousebrain_MethodData('spa_mo_model', np.vstack(arrays), section_arr, barcode_arr, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'clustering' / 'batch_correction_metrics.csv'})

def mousebrain_load_mofa(paths, output_root) -> mousebrain_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = pd.read_csv(table_path)
    factor_cols = [column for column in table if column.startswith('Factor')]
    section_map = {'SectionA': 's1', 'SectionB': 's2', 'SectionC': 's3'}
    sections = table['section'].astype(str).map(section_map).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    return mousebrain_MethodData('MOFA+', table[factor_cols].to_numpy(float), sections, barcodes, coords, truth, data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})

def mousebrain_load_cosie(paths, output_root) -> mousebrain_MethodData:
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path)
    emb_cols = [column for column in table if column.startswith('COSIE')]
    sections = table['section'].astype(str).to_numpy()
    barcodes = table['obs_name'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MOUSEBRAIN_SECTIONS]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    table_embedding = table[emb_cols].to_numpy(float)
    if final_embedding.shape != table_embedding.shape or not np.allclose(final_embedding, table_embedding):
        raise ValueError('COSIE: analysis table is not the saved final embedding')
    sources = [table_path, *final_paths]
    return mousebrain_MethodData('COSIE', table_embedding, sections, barcodes, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def mousebrain_load_spamosaic(paths, output_root) -> mousebrain_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    section_map = {'SectionA': 's1', 'SectionB': 's2', 'SectionC': 's3'}
    sections = adata.obs['section'].astype(str).map(section_map).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = mousebrain_aligned_metadata(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return mousebrain_MethodData('SpaMosaic', embedding, sections, barcodes, coords, truth, data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

def simulation_truth_and_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.empty(len(sections), dtype=object)
    coords = np.empty((len(sections), 2), dtype=float)
    for section in SIMULATION_SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            source_names = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            pos = source_names.get_indexer(wanted)
            if (pos < 0).any():
                missing = wanted[pos < 0][:5].tolist()
                raise ValueError(f'{rna_path}: failed to align {int((pos < 0).sum())} barcodes; examples={missing}')
            if 'spatial_domain' in rna.obs:
                sec_truth = rna.obs['spatial_domain'].astype(str).to_numpy()[pos]
            elif 'spfac' in rna.obsm:
                spfac = np.asarray(rna.obsm['spfac'])[pos]
                sec_truth = np.full(len(pos), 'background', dtype=object)
                active = spfac.sum(axis=1) > 0
                sec_truth[active] = [f'sp{i + 1}' for i in spfac[active].argmax(axis=1)]
            else:
                raise ValueError(f'{rna_path}: no spatial_domain or spfac truth.')
            spatial = np.asarray(rna.obsm['spatial'])[pos, :2]
        finally:
            rna.file.close()
        truth[mask] = sec_truth
        coords[mask] = spatial
    return (truth.astype(str), coords)

def simulation_load_spa(paths, output_root) -> simulation_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays = []
    sections = []
    barcodes = []
    sources: list[Path] = []
    for section in SIMULATION_SECTIONS:
        emb_path = run / f'final_embeddings_{section}.npy'
        meta_path = run / f'obs_metadata_{section}.csv'
        meta = pd.read_csv(meta_path)
        arrays.append(np.load(emb_path))
        sections.extend([section] * len(meta))
        barcodes.extend(meta['obs_name'].astype(str).tolist())
        sources.extend([emb_path, meta_path])
    section_arr = np.asarray(sections, dtype=str)
    barcode_arr = np.asarray(barcodes, dtype=str)
    (truth, coords) = simulation_truth_and_coords(data_dir, section_arr, barcode_arr)
    old_analysis = run / 'clustering_analysis'
    return simulation_MethodData(name='spa_mo_model', embedding=np.vstack(arrays), sections=section_arr, barcodes=barcode_arr, truth=truth, coords=coords, data_dir=data_dir, source_paths=sources, output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old_analysis / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old_analysis / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old_analysis / 'batch_correction_metrics.csv'})

def simulation_load_mofa(paths, output_root) -> simulation_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata.csv'
    table = pd.read_csv(table_path)
    factor_cols = [col for col in table if col.startswith('Factor')]
    sections = table['group'].astype(str).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    return simulation_MethodData(name='MOFA+', embedding=table[factor_cols].to_numpy(dtype=float), sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[table_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})

def simulation_load_cosie(paths, output_root) -> simulation_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path, index_col=0)
    emb_cols = [col for col in table if col.startswith('COSIE')]
    barcode_col = 'obs_name.1' if 'obs_name.1' in table else 'obs_name'
    sections = table['section'].astype(str).to_numpy()
    barcodes = table[barcode_col].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    return simulation_MethodData(name='COSIE', embedding=table[emb_cols].to_numpy(dtype=float), sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[table_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def simulation_load_spamosaic(paths, output_root) -> simulation_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    (truth, coords) = simulation_truth_and_coords(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return simulation_MethodData(name='SpaMosaic', embedding=embedding, sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=[h5ad_path], output_root=output_root, shared_sources={'simulation_factor_diagnostics.csv': old / 'metrics' / 'simulation_factor_diagnostics.csv', 'cross_section_spot_retrieval.csv': old / 'metrics' / 'cross_section_spot_retrieval.csv', 'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

def mouse_thymus_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray) -> np.ndarray:
    coords = np.empty((len(sections), 2), dtype=float)
    for section in MOUSE_THYMUS_SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            positions = source.get_indexer(wanted)
            if (positions < 0).any():
                examples = wanted[positions < 0][:5].tolist()
                raise ValueError(f'{rna_path}: {int((positions < 0).sum())} barcodes failed alignment; examples={examples}')
            if not {'x', 'y'}.issubset(rna.obs.columns):
                raise ValueError(f'{rna_path}: missing canonical obs[x,y]')
            coords[mask] = rna.obs[['x', 'y']].to_numpy()[positions]
        finally:
            rna.file.close()
    return coords

def mouse_thymus_load_spa(paths, output_root) -> mouse_thymus_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in MOUSE_THYMUS_SECTIONS:
        embedding_path = run / f'final_embeddings_{section}.npy'
        metadata_path = run / f'obs_metadata_{section}.csv'
        metadata = pd.read_csv(metadata_path)
        embedding = np.load(embedding_path)
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()
            own_coords = rna.obs[['x', 'y']].to_numpy()
        finally:
            rna.file.close()
        saved_names = metadata['obs_name'].astype(str).to_numpy()
        saved_coords = metadata[['spatial_x', 'spatial_y']].to_numpy()
        if len(embedding) != len(saved_names) or not np.array_equal(saved_names, own_names) or (not np.allclose(saved_coords, own_coords)):
            raise ValueError(f'spa_mo_model {section}: embedding/metadata/data mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(saved_names))
        barcodes.extend(saved_names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / 'clustering_analysis'
    return mouse_thymus_MethodData('spa_mo_model', np.vstack(arrays), section_array, barcode_array, mouse_thymus_aligned_coords(data_dir, section_array, barcode_array), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def mouse_thymus_load_mofa(paths, output_root) -> mouse_thymus_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = pd.read_csv(table_path)
    factor_columns = [column for column in table.columns if column.startswith('Factor')]
    sections = table['section'].astype(str).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    return mouse_thymus_MethodData('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, mouse_thymus_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv', 'input_audit.csv': old / 'input_audit.csv'})

def mouse_thymus_load_cosie(paths, output_root) -> mouse_thymus_MethodData:
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path)
    embedding_columns = [column for column in table.columns if column.startswith('COSIE')]
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MOUSE_THYMUS_SECTIONS]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    if final_embedding.shape != table_embedding.shape or not np.allclose(final_embedding, table_embedding):
        raise ValueError('COSIE: analysis table is not the final embedding')
    sections = table['section'].astype(str).to_numpy()
    barcode_column = 'obs_name.1' if 'obs_name.1' in table.columns else 'obs_name'
    barcodes = table[barcode_column].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    return mouse_thymus_MethodData('COSIE', table_embedding, sections, barcodes, mouse_thymus_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def mouse_thymus_load_spamosaic(paths, output_root) -> mouse_thymus_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return mouse_thymus_MethodData('SpaMosaic', embedding, sections, barcodes, mouse_thymus_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

def misar_seq_aligned_metadata(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray):
    return aligned_misar_metadata(data_dir, sections, barcodes, section_order=MISAR_SEQ_SECTIONS, label_keys=MISAR_SEQ_LABEL_KEYS)

def misar_seq_load_spa(paths, output_root) -> misar_seq_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    (arrays, sections, barcodes, sources) = ([], [], [], [])
    for section in MISAR_SEQ_SECTIONS:
        embedding_path = run / f'final_embeddings_{section}.npy'
        metadata_path = run / f'obs_metadata_{section}.csv'
        metadata = pd.read_csv(metadata_path)
        embedding = np.load(embedding_path)
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        saved_names = metadata['obs_name'].astype(str).to_numpy()
        if len(embedding) != len(saved_names) or not np.array_equal(saved_names, own_names):
            raise ValueError(f'spa_mo_model {section}: embedding/metadata/data mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(saved_names))
        barcodes.extend(saved_names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections)
    barcode_array = np.asarray(barcodes)
    (coords, truth) = misar_seq_aligned_metadata(data_dir, section_array, barcode_array)
    old = run / 'clustering_analysis_k8_10_12_14_16'
    return misar_seq_MethodData('spa_mo_model', np.vstack(arrays), section_array, barcode_array, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def misar_seq_load_mofa(paths, output_root) -> misar_seq_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = pd.read_csv(table_path)
    factor_columns = [column for column in table if column.startswith('Factor')]
    sections = table['section'].astype(str).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = misar_seq_aligned_metadata(data_dir, sections, barcodes)
    return misar_seq_MethodData('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, coords, truth, data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})

def misar_seq_load_cosie(paths, output_root) -> misar_seq_MethodData:
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path)
    embedding_columns = [column for column in table if column.startswith('COSIE')]
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MISAR_SEQ_SECTIONS]
    final_embedding = np.vstack([np.load(path) for path in final_paths])
    if final_embedding.shape != table_embedding.shape or not np.allclose(final_embedding, table_embedding):
        raise ValueError('COSIE: analysis table is not the final embedding')
    sections = table['section'].astype(str).to_numpy()
    barcode_column = 'original_barcode' if 'original_barcode' in table else 'obs_name'
    barcodes = table[barcode_column].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = misar_seq_aligned_metadata(data_dir, sections, barcodes)
    return misar_seq_MethodData('COSIE', table_embedding, sections, barcodes, coords, truth, data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def misar_seq_load_spamosaic(paths, output_root) -> misar_seq_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    (coords, truth) = misar_seq_aligned_metadata(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return misar_seq_MethodData('SpaMosaic', embedding, sections, barcodes, coords, truth, data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment.csv': old / 'metrics' / 'modality_alignment.csv'})

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










@dataclass
class crc_stereocite_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

CRC_STEREOCITE_SECTIONS = ['CRC_003_bin20', 'CRC_006_bin20']

def crc_stereocite_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray):
    return aligned_crc_coords(data_dir, sections, barcodes, section_order=CRC_STEREOCITE_SECTIONS)

CRC_STEREOCITE_SHORT_NAMES = {'CRC_003_bin20': 'CRC_003', 'CRC_006_bin20': 'CRC_006'}

def crc_stereocite_load_spa(paths, output_root) -> crc_stereocite_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    (arrays, sections, barcodes, sources) = ([], [], [], [])
    for section in CRC_STEREOCITE_SECTIONS:
        short = CRC_STEREOCITE_SHORT_NAMES[section]
        embedding_path = run / f'final_embeddings_{short}.npy'
        indices_path = run / f'selected_spot_indices_{short}.npy'
        spatial_path = run / f'spatial_{short}.npy'
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()[indices]
            own_coords = np.asarray(rna.obsm['spatial'])[indices, :2]
        finally:
            rna.file.close()
        if len(embedding) != len(indices) or len(np.unique(indices)) != len(indices) or (not np.allclose(saved_coords, own_coords)):
            raise ValueError(f'spa_mo_model {section}: embedding/index/spatial mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(indices))
        barcodes.extend(own_names.tolist())
        sources.extend([embedding_path, indices_path, spatial_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / 'clustering_analysis_k8_10_15'
    return crc_stereocite_MethodData('spa_mo_model', np.vstack(arrays), section_array, barcode_array, crc_stereocite_aligned_coords(data_dir, section_array, barcode_array), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def crc_stereocite_load_mofa(paths, output_root) -> crc_stereocite_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    header = pd.read_csv(table_path, nrows=0)
    factor_columns = [column for column in header.columns if column.startswith('Factor')]
    columns = [*factor_columns, 'section', 'original_barcode']
    table = pd.read_csv(table_path, usecols=columns, dtype={'section': str, 'original_barcode': str})
    sections = table['section'].to_numpy()
    barcodes = table['original_barcode'].to_numpy()
    data_dir = Path(paths['data_dir'])
    return crc_stereocite_MethodData('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, crc_stereocite_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})

def crc_stereocite_load_cosie(paths, output_root) -> crc_stereocite_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    (arrays, sections, barcodes, sources) = ([], [], [], [])
    for section in CRC_STEREOCITE_SECTIONS:
        short = CRC_STEREOCITE_SHORT_NAMES[section]
        embedding_path = run / 'final_embeddings' / f'{short}_final_embedding.npy'
        names_path = run / f'obs_names_{short}.npy'
        indices_path = run / f'selected_spot_indices_{short}.npy'
        spatial_path = run / f'spatial_{short}.npy'
        embedding = np.load(embedding_path)
        names = np.load(names_path, allow_pickle=True).astype(str)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()[indices]
            own_coords = np.asarray(rna.obsm['spatial'])[indices, :2]
        finally:
            rna.file.close()
        if len(embedding) != len(names) or not np.array_equal(names, own_names) or (not np.allclose(saved_coords, own_coords)):
            raise ValueError(f'COSIE {section}: embedding/name/index/spatial mismatch')
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, names_path, indices_path, spatial_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    old = run / 'analysis'
    return crc_stereocite_MethodData('COSIE', np.vstack(arrays), section_array, barcode_array, crc_stereocite_aligned_coords(data_dir, section_array, barcode_array), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def crc_stereocite_load_spamosaic(paths, output_root) -> crc_stereocite_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return crc_stereocite_MethodData('SpaMosaic', embedding, sections, barcodes, crc_stereocite_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

@dataclass
class human_lymph_node_MethodData:
    name: str
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    coords: np.ndarray
    data_dir: Path
    source_paths: list[Path]
    output_root: Path
    shared_sources: dict[str, Path]

HUMAN_LYMPH_NODE_SECTIONS = ['Human_Lymph_Node_A1', 'Human_Lymph_Node_D1']

def human_lymph_node_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray) -> np.ndarray:
    coords = np.empty((len(sections), 2), dtype=float)
    for section in HUMAN_LYMPH_NODE_SECTIONS:
        mask = sections == section
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            pos = source.get_indexer(wanted)
            if (pos < 0).any():
                raise ValueError(f'{rna_path}: {int((pos < 0).sum())} barcodes failed alignment')
            coords[mask] = np.asarray(rna.obsm['spatial'])[pos, :2]
        finally:
            rna.file.close()
    return coords

def human_lymph_node_load_spa(paths, output_root) -> human_lymph_node_MethodData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    (arrays, sections, barcodes, sources) = ([], [], [], [])
    for section in HUMAN_LYMPH_NODE_SECTIONS:
        emb_path = run / f'final_embeddings_{section}.npy'
        idx_path = run / f'selected_spot_indices_{section}.npy'
        spatial_path = run / f'spatial_{section}.npy'
        emb = np.load(emb_path)
        idx = np.load(idx_path).astype(int)
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            names = rna.obs_names.astype(str).to_numpy()[idx]
            own_coords = np.asarray(rna.obsm['spatial'])[idx, :2]
        finally:
            rna.file.close()
        saved_coords = np.load(spatial_path)[:, :2]
        if emb.shape[0] != len(idx) or not np.allclose(saved_coords, own_coords):
            raise ValueError(f'spa_mo_model {section}: saved rows/spatial do not align')
        arrays.append(emb)
        sections.extend([section] * len(idx))
        barcodes.extend(names.tolist())
        sources.extend([emb_path, idx_path, spatial_path, rna_path])
    section_arr = np.asarray(sections, dtype=str)
    barcode_arr = np.asarray(barcodes, dtype=str)
    old = run / 'clustering_analysis'
    return human_lymph_node_MethodData('spa_mo_model', np.vstack(arrays), section_arr, barcode_arr, human_lymph_node_aligned_coords(data_dir, section_arr, barcode_arr), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def human_lymph_node_load_mofa(paths, output_root) -> human_lymph_node_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = pd.read_csv(table_path)
    factor_cols = [column for column in table if column.startswith('Factor')]
    sections = table['section'].astype(str).to_numpy()
    barcodes = table['original_barcode'].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    shared = {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'input_audit.csv': old / 'input_audit.csv'}
    r2 = old / 'tables' / 'r2_total_by_view_group.csv'
    if r2.exists():
        shared['r2_total_by_view_group.csv'] = r2
    return human_lymph_node_MethodData('MOFA+', table[factor_cols].to_numpy(float), sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, shared)

def human_lymph_node_load_cosie(paths, output_root) -> human_lymph_node_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = pd.read_csv(table_path)
    emb_cols = [column for column in table if column.startswith('COSIE')]
    sections = table['section'].astype(str).to_numpy()
    barcode_col = 'obs_name.1' if 'obs_name.1' in table else 'obs_name'
    barcodes = table[barcode_col].astype(str).to_numpy()
    data_dir = Path(paths['data_dir'])
    return human_lymph_node_MethodData('COSIE', table[emb_cols].to_numpy(float), sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def human_lymph_node_load_spamosaic(paths, output_root) -> human_lymph_node_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections = adata.obs['section'].astype(str).to_numpy()
    barcodes = adata.obs['original_barcode'].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm['merged_emb']).copy()
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return human_lymph_node_MethodData('SpaMosaic', embedding, sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

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

@dataclass
class spatch_MethodSpec:
    key: str
    name: str
    run_dir: Path
    data_dir: Path
    embedding_paths: dict[str, Path]
    metadata_path: Path
    id_column: str
    output_root: Path
    old_analysis: Path
    batch_metrics_source: Path

SPATCH_LABELS = SPATCH['LABELS']

def spatch_load_metadata(spec: spatch_MethodSpec) -> pd.DataFrame:
    return load_spatch_metadata(spec.metadata_path, spec.id_column, SPATCH_LABELS)

SPATCH_SECTIONS = ['section1', 'section2']

SPATCH_SECTION_COUNTS = {'section1': 665399, 'section2': 403563}

def spatch_load_embedding(spec: spatch_MethodSpec) -> np.ndarray:
    return load_spatch_embedding(spec.embedding_paths, section_order=SPATCH_SECTIONS, section_counts=SPATCH_SECTION_COUNTS, name=spec.name)

SPATCH_N_OBS = 1068962

def spatch_validate_metadata(spec: spatch_MethodSpec, metadata: pd.DataFrame) -> None:
    if len(metadata) != SPATCH_N_OBS:
        raise ValueError(f'{spec.name}: expected {SPATCH_N_OBS}, got {len(metadata)}')
    counts = metadata['section'].value_counts().to_dict()
    if counts != SPATCH_SECTION_COUNTS:
        raise ValueError(f'{spec.name}: section counts differ: {counts}')
    if metadata[['section', 'spot_id']].duplicated().any():
        raise ValueError(f'{spec.name}: duplicate section/spot_id')
    if not np.isfinite(metadata[['x', 'y']].to_numpy()).all():
        raise ValueError(f'{spec.name}: non-finite coordinates')
    for path in [*spec.embedding_paths.values(), spec.metadata_path, spec.batch_metrics_source]:
        if not path.exists():
            raise FileNotFoundError(path)
