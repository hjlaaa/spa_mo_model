"""Cross-method UMAP aligned to an explicitly supplied reference cohort."""
from __future__ import annotations

import argparse

import hashlib

import json

import math

import re


from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import Any

import anndata as ad

import matplotlib

matplotlib.use('Agg')



import numpy as np

import pandas as pd

import umap

from data_io.large_results import borrowed_input
from data_io.reference_results import normalize_sections, read_reference_metadata
from data_io.saved_assignments import read_assignment_table, read_assignment_array, selected_csv_assignments
from analysis.plotting import plot_comparison_panel_e

from analysis.cache import check_output_path, begin_analysis, finish_analysis, array_identity, frame_identity, implementation_identity

from analysis.umap import fit_umap as shared_fit_umap

from analysis.plotting import plot_individual
from analysis.umap_workflow import run_method_projection

METHODS = ('cosie', 'mofa', 'spamosaic', 'harmony')

DATASETS = ('crc_stereocite', 'human_embryo', 'human_lymph_node', 'misar_seq', 'mouse_spleen', 'mouse_thymus', 'mousebrain', 'simulation', 'spatch')

METHOD_NAMES = {'cosie': 'COSIE', 'mofa': 'MOFA+', 'spamosaic': 'SpaMosaic', 'harmony': 'Harmony'}

METHOD_DATASETS = {'cosie': set(DATASETS), 'mofa': set(DATASETS), 'spamosaic': set(DATASETS), 'harmony': {'human_embryo'}}

MOUSEBRAIN_SECTION_ALIASES = {'SectionA': 's1', 'SectionB': 's2', 'SectionC': 's3', 's1': 's1', 's2': 's2', 's3': 's3'}


@dataclass
class ReferenceData:
    dataset: str
    config: dict[str, Any]
    metadata: pd.DataFrame
    sections: list[str]
    primary_label: str
    representative_k: int
    reference_dir: Path

@dataclass
class MethodData:
    method: str
    dataset: str
    analysis_root: Path
    output_dir: Path
    source_config: dict[str, Any]
    embedding: np.ndarray
    joint_labels: dict[int, np.ndarray]
    provenance: dict[str, Any]

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def normalize_section(dataset: str, values: pd.Series | np.ndarray) -> np.ndarray:
    return normalize_sections(values, MOUSEBRAIN_SECTION_ALIASES if dataset == 'mousebrain' else None)


def canonical_keys(sections: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    from data_io.reference_results import canonical_keys as read
    return read(sections, identifiers)

def load_reference(dataset: str, reference_dir: Path) -> ReferenceData:
    config = json.loads((reference_dir / 'umap_config.json').read_text())
    table_path = Path(config['coordinates_table'])
    metadata = read_reference_metadata(table_path, context=dataset,
        section_aliases=MOUSEBRAIN_SECTION_ALIASES if dataset == 'mousebrain' else None,
        drop_prefixes=('input_', 'integrated_UMAP', 'joint_k'))
    return ReferenceData(dataset=dataset, config=config, metadata=metadata, sections=list(config['section_order']), primary_label=config['primary_biological_label'], representative_k=int(config['representative_joint_k']), reference_dir=reference_dir)

def source_id_column(columns: list[str]) -> str:
    from data_io.reference_results import source_id_column as read
    return read(columns)

def feature_columns(columns: list[str], method: str) -> list[str]:
    from data_io.reference_results import feature_columns as read
    return read(columns, r'^COSIE\d+$' if method == 'cosie' else r'^Factor\d+$')

def selected_positions_from_keys(dataset: str, source_sections: np.ndarray, source_ids: np.ndarray, reference: ReferenceData) -> np.ndarray:
    from data_io.reference_results import selected_positions_from_keys as read
    return read(dataset, source_sections, source_ids, reference, section_aliases=MOUSEBRAIN_SECTION_ALIASES if dataset == 'mousebrain' else None)

def load_generic_selected_raw(method: str, dataset: str, source_path: Path, reference: ReferenceData) -> tuple[np.ndarray, dict[str, Any]]:
    from data_io.reference_results import read_selected_result
    return read_selected_result(source_path, reference, file_format={'.csv': 'csv', '.h5ad': 'h5ad'}.get(source_path.suffix),
        feature_pattern=r'^COSIE\d+$' if method == 'cosie' else r'^Factor\d+$', context=dataset,
        section_aliases=MOUSEBRAIN_SECTION_ALIASES if dataset == 'mousebrain' else None)

def select_by_section_rows(arrays: dict[str, np.ndarray], reference: ReferenceData) -> np.ndarray:
    from data_io.reference_results import select_by_section_rows as read
    return read(arrays, reference)

def validate_section_ids(ids_by_section: dict[str, np.ndarray], reference: ReferenceData) -> None:
    from data_io.reference_results import validate_section_ids as validate
    return validate(ids_by_section, reference)

def load_cosie_large_selected_raw(dataset: str, config: dict[str, Any], reference: ReferenceData) -> tuple[np.ndarray, dict[str, Any]]:
    if dataset == 'crc_stereocite':
        source_files = [Path(item['path']) for item in config['source_files']]
        arrays = {}
        ids = {}
        for section in reference.sections:
            embedding_path = next((path for path in source_files if path.suffix == '.npy' and section.split('_bin20')[0] in path.name and ('final_embedding' in path.name)))
            arrays[section] = np.load(embedding_path, mmap_mode='r')
            obs_path = embedding_path.parents[1] / f"obs_names_{section.split('_bin20')[0]}.npy"
            ids[section] = np.load(obs_path, allow_pickle=True)
        validate_section_ids(ids, reference)
        return (select_by_section_rows(arrays, reference), {'source_embedding': [path for path in source_files if 'final_embedding' in path.name], 'source_type': 'section-specific COSIE final embedding NPY', 'alignment': 'validated section-local row and obs_name'})
    if dataset == 'spatch':
        arrays = {Path(item['path']).stem.split('_final_embedding')[0]: np.load(item['path'], mmap_mode='r') for item in config['source_embeddings']}
        metadata_path = Path(config['source_metadata']['path'])
        source_metadata = pd.read_csv(metadata_path, usecols=['section', 'spot_id'], low_memory=False)
        ids = {section: source_metadata.loc[source_metadata['section'].eq(section), 'spot_id'].astype(str).to_numpy() for section in reference.sections}
        validate_section_ids(ids, reference)
        return (select_by_section_rows(arrays, reference), {'source_embedding': [item['path'] for item in config['source_embeddings']], 'source_metadata': metadata_path, 'source_type': 'section-specific COSIE final embedding NPY', 'alignment': 'validated section-local row and spot_id'})
    raise ValueError(dataset)

def apply_saved_standardization(values: np.ndarray, scaler_path: Path) -> tuple[np.ndarray, str]:
    scaler = np.load(scaler_path)
    prefix = 'combined' if 'combined_mean' in scaler.files else 'joint'
    mean = np.asarray(scaler[f'{prefix}_mean'], dtype=np.float32)
    scale = np.asarray(scaler[f'{prefix}_scale'], dtype=np.float32)
    safe_scale = np.where(scale == 0, 1.0, scale)
    if values.shape[1] != len(mean):
        raise ValueError(f'{scaler_path}: embedding dimension {values.shape[1]} != scaler {len(mean)}.')
    standardized = ((values - mean) / safe_scale).astype(np.float32)
    return (standardized, prefix)

def find_label_file(cluster_dir: Path, section: str) -> Path:
    direct = cluster_dir / f'labels_{section}.csv'
    if direct.is_file():
        return direct
    matches = []
    for path in cluster_dir.glob('labels_*.csv'):
        stem_section = path.stem.split('labels_', 1)[1]
        if MOUSEBRAIN_SECTION_ALIASES.get(stem_section, stem_section) == section:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f'{cluster_dir}: cannot uniquely resolve label file for {section}: {matches}')
    return matches[0]

def load_csv_joint_labels(clustering_dir: Path, reference: ReferenceData) -> dict[int, np.ndarray]:
    labels_by_k: dict[int, np.ndarray] = {}
    for cluster_dir in sorted(clustering_dir.glob('joint_k*')):
        try:
            k = int(cluster_dir.name.split('joint_k', 1)[1])
        except ValueError:
            continue
        result = np.empty(len(reference.metadata), dtype=np.int32)
        for section in reference.sections:
            mask = reference.metadata['section'].eq(section).to_numpy()
            expected_ids = reference.metadata.loc[mask, 'spot_id'].astype(str).to_numpy()
            local = reference.metadata.loc[mask, 'section_row_index'].to_numpy(dtype=np.int64)
            table = read_assignment_table(find_label_file(cluster_dir, section), low_memory=False)
            selected = selected_csv_assignments(table, expected_ids, local, missing_context=f'{cluster_dir}/{section}')
            result[mask] = selected
        labels_by_k[k] = result
    return labels_by_k

def load_human_data(method: str, analysis_root: Path, reference: ReferenceData) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any], dict[str, Any]]:
    run_root = analysis_root.parent
    summary = json.loads((run_root / 'run_summary.json').read_text())
    full = np.load(analysis_root / 'cache/joint_standardized.npy', mmap_mode='r')
    counts = {}
    for section in reference.sections:
        path = analysis_root / f'clustering/joint_k{reference.representative_k}/labels_{section}.npy'
        counts[section] = len(read_assignment_array(path, mmap_mode='r'))
    offsets = dict(zip(reference.sections, np.cumsum([0, *[counts[s] for s in reference.sections[:-1]]])))
    selected = np.empty((len(reference.metadata), full.shape[1]), dtype=np.float32)
    for section in reference.sections:
        mask = reference.metadata['section'].eq(section).to_numpy()
        local = reference.metadata.loc[mask, 'section_row_index'].to_numpy(dtype=np.int64)
        selected[mask] = np.asarray(full[offsets[section] + local], dtype=np.float32)
    labels_by_k = {}
    for cluster_dir in sorted((analysis_root / 'clustering').glob('joint_k*')):
        try:
            k = int(cluster_dir.name.split('joint_k', 1)[1])
        except ValueError:
            continue
        result = np.empty(len(reference.metadata), dtype=np.int32)
        complete = True
        for section in reference.sections:
            path = cluster_dir / f'labels_{section}.npy'
            if not path.is_file():
                complete = False
                break
            mask = reference.metadata['section'].eq(section).to_numpy()
            local = reference.metadata.loc[mask, 'section_row_index'].to_numpy(dtype=np.int64)
            result[mask] = np.asarray(read_assignment_array(path, mmap_mode='r')[local], dtype=np.int32)
        if complete:
            labels_by_k[k] = result
    provenance = {'source_embedding': analysis_root / 'cache/joint_standardized.npy', 'source_type': 'saved human embryo joint standardized embedding', 'alignment': 'shared preprocessed feature-row order + result_v6 section_row_index'}
    contract = borrowed_input(selected, sections=reference.metadata["section"].to_numpy(),
        evidence={"kind": "shared_preprocessed_row_order", "selected_rows": reference.metadata["section_row_index"].to_numpy()}, provenance=provenance)
    return (contract.embedding, labels_by_k, provenance, summary)

def load_method_data(method: str, dataset: str, reference: ReferenceData, analysis_root, output_dir) -> MethodData:
    if not analysis_root.is_dir():
        raise FileNotFoundError(f'Missing analysis directory: {analysis_root}')
    if dataset == 'human_embryo':
        (embedding, joint_labels, provenance, source_config) = load_human_data(method, analysis_root, reference)
    else:
        config_path = analysis_root / 'config.json'
        source_config = json.loads(config_path.read_text())
        if method == 'cosie' and dataset in {'crc_stereocite', 'spatch'}:
            (raw, provenance) = load_cosie_large_selected_raw(dataset, source_config, reference)
        else:
            source_path = Path(source_config['source_files'][0]['path'])
            (raw, provenance) = load_generic_selected_raw(method, dataset, source_path, reference)
        (embedding, scaler_prefix) = apply_saved_standardization(raw, analysis_root / 'scaler_parameters.npz')
        provenance['scaler'] = analysis_root / 'scaler_parameters.npz'
        provenance['scaler_prefix'] = scaler_prefix
        joint_labels = load_csv_joint_labels(analysis_root / 'clustering', reference)
    if embedding.shape[0] != len(reference.metadata):
        raise ValueError(f'{method}/{dataset}: embedding row mismatch {embedding.shape}.')
    if not np.isfinite(embedding).all():
        raise ValueError(f'{method}/{dataset}: non-finite standardized embedding values.')
    if reference.representative_k not in joint_labels:
        raise KeyError(f'{method}/{dataset}: representative k={reference.representative_k} unavailable; available={sorted(joint_labels)}')
    return MethodData(method=method, dataset=dataset, analysis_root=analysis_root, output_dir=output_dir, source_config=source_config, embedding=embedding, joint_labels=joint_labels, provenance=provenance)

def fit_umap(values: np.ndarray, cache_path: Path, n_neighbors: int, min_dist: float, seed: int, overwrite: bool, *, source_identity: dict | None=None) -> np.ndarray:
    return shared_fit_umap(values, cache_path, n_neighbors, min_dist, seed, overwrite, source_identity=source_identity or {}, force_float32=False)

def custom_biology_colors(data: MethodData, reference: ReferenceData) -> dict[str, str] | None:
    if reference.dataset != 'human_embryo':
        return None
    path = data.analysis_root / 'celltype_reference/celltype_color_map.csv'
    table = pd.read_csv(path)
    return dict(zip(table['celltype'].astype(str), table['color'].astype(str)))

from analysis.plotting import meaningful_biology_mask

def plot_panel_e(data: MethodData, reference: ReferenceData, coordinates: np.ndarray, figures_dir: Path, seed: int) -> None:
    return plot_comparison_panel_e(data, reference, coordinates, figures_dir, seed, method_name=METHOD_NAMES[data.method], custom_biology=custom_biology_colors(data, reference) if meaningful_biology_mask(reference).any() else None)

def generate_outputs(data: MethodData, reference: ReferenceData, args: argparse.Namespace) -> dict[str, Any]:
    data.output_dir = check_output_path(data.output_dir, [data.analysis_root, reference.reference_dir])
    source = {'embedding': array_identity(data.embedding), 'metadata': frame_identity(reference.metadata), 'method': data.method, 'dataset': reference.dataset, 'labels': {str(k): array_identity(v) for (k, v) in data.joint_labels.items()}, 'primary_label': reference.primary_label, 'representative_k': reference.representative_k, 'protocol': {'n_neighbors': args.n_neighbors, 'min_dist': args.min_dist, 'seed': args.seed}, 'colors': custom_biology_colors(data, reference) if meaningful_biology_mask(reference).any() else None, 'implementation': implementation_identity('comparison-umap-panels-v1')}
    if begin_analysis(data.output_dir, source, input_dirs=[data.analysis_root, reference.reference_dir]):
        cached = json.loads((data.output_dir / 'umap_config.json').read_text())
        cached['output_dir'] = str(data.output_dir)
        return cached
    table_path, biological_path, biology_mask, include_biology = run_method_projection(
        data, reference, args, source=source, method_name=METHOD_NAMES[data.method],
        plot_panel=plot_panel_e, custom_biology_colors=custom_biology_colors)
    figures_dir = data.output_dir / 'figures'
    config = {'status': 'PASS', 'method': METHOD_NAMES[data.method], 'method_key': data.method, 'dataset': reference.dataset, 'display_name': reference.config['display_name'], 'analysis_root': data.analysis_root, 'output_dir': data.output_dir, 'embedding_source': data.provenance, 'embedding_dim': int(data.embedding.shape[1]), 'standardized_embedding_only': True, 'n_obs_total': int(reference.config['n_obs_total']), 'n_obs_umap': len(reference.metadata), 'sampled': bool(reference.config['sampled']), 'sampling_reference': reference.reference_dir / 'tables/sample_indices_by_section.npz', 'alignment_reference': Path(reference.config['coordinates_table']), 'alignment_key': 'section + spot_id (or shared human embryo preprocessed row order)', 'section_order': reference.sections, 'section_counts_umap': reference.config['section_counts_umap'], 'primary_biological_label': reference.primary_label, 'biological_annotation': {'included_in_figures': include_biology, 'n_annotated': int(biology_mask.sum()), 'n_unavailable': int((~biology_mask).sum()), 'omission_rule': 'omit biological-label figures when no meaningful annotation is available'}, 'joint_cluster_k_values': sorted(data.joint_labels), 'representative_joint_k': reference.representative_k, 'panel_mapping': {'e': 'method standardized-embedding UMAP colored by section, method-specific representative joint cluster, and reference biological label' if include_biology else 'method standardized-embedding UMAP colored by section and method-specific representative joint cluster; biological-label panel omitted because annotation is unavailable'}, 'umap': {'implementation': 'umap-learn', 'version': umap.__version__, 'n_components': 2, 'n_neighbors': args.n_neighbors, 'min_dist': args.min_dist, 'metric': 'euclidean', 'init': 'spectral', 'random_state': args.seed, 'transform_seed': args.seed, 'n_jobs': 1, 'low_memory': True}, 'coordinates_table': table_path, 'figures': {'panel_e': figures_dir / 'panel_e_integrated_embedding_umap.png', 'by_section': figures_dir / 'integrated_by_section.png', 'by_biology': biological_path if include_biology else None}, 'source_analysis_config': data.source_config, 'created_at': datetime.now().astimezone().isoformat(), 'script': Path(__file__).resolve(), 'script_sha256': sha256(Path(__file__).resolve())}
    write_json(data.output_dir / 'umap_config.json', config)
    readme = [f"# {METHOD_NAMES[data.method]} {reference.config['display_name']} UMAP", '', "This directory contains a SpaMosaic-paper-style panel e generated from the method's standardized embedding.", '', f'- Visualization spots and biological labels are aligned to `{reference.reference_dir}`.', "- Joint-cluster colors use this comparison method's own standardized-embedding labels.", f'- Representative joint clustering: k={reference.representative_k}.', f'- Biological-label figures include {int(biology_mask.sum()):,} annotated spots.' if include_biology else '- Biological-label figures are omitted because no meaningful primary annotation is available.', f'- UMAP: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric=euclidean, seed={args.seed}.', '- Raw/input-modality panel c is not duplicated because it is method-independent and already exists in the reference workflow.', '']
    (data.output_dir / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    finish_analysis(data.output_dir, source)
    return config


def run_method_umaps(entries, output_root, args):
    """Explicit method/dataset/input list; independent output and existing status semantics."""
    output_root=check_output_path(output_root)
    keys=[(item['method'],item['dataset']) for item in entries]
    if not keys or len(keys)!=len(set(keys)):raise ValueError('Expected distinct nonempty method/dataset entries')
    statuses=[]
    for item in entries:
        method,dataset=item['method'],item['dataset']
        if method not in METHODS or dataset not in METHOD_DATASETS[method]:raise ValueError(f'Unsupported method/dataset: {method}/{dataset}')
        reference_dir=Path(item['reference_dir']);analysis_dir=Path(item['analysis_dir'])
        check_output_path(output_root,[reference_dir,analysis_dir])
        reference=load_reference(dataset,reference_dir)
        try:
            data=load_method_data(method,dataset,reference,analysis_dir,output_root/method/dataset)
            config=generate_outputs(data,reference,args)
            statuses.append({'method':METHOD_NAMES[method],'dataset':dataset,'status':'PASS','output_dir':config['output_dir']})
        except FileNotFoundError as error:
            statuses.append({'method':METHOD_NAMES[method],'dataset':dataset,'status':'UNAVAILABLE','reason':str(error)})
    manifest={'status':'PASS' if any(item['status']=='PASS' for item in statuses) else 'UNAVAILABLE',
        'results':statuses,'created_at':datetime.now().astimezone().isoformat(),
        'script':Path(__file__).resolve(),'script_sha256':sha256(Path(__file__).resolve())}
    output_root.mkdir(parents=True,exist_ok=True)
    write_json(output_root/'comparison_method_umap_manifest.json',manifest)
    return json_safe(manifest)
