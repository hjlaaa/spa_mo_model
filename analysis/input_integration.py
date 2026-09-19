"""Explicit run/data/analysis inputs for SpaMosaic-style panels; no SpaMosaic model execution."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from analysis.protocols import SPATCH_UMAP_INPUT_FILES
from analysis.cache import check_output_path, begin_analysis, finish_analysis, file_identity, frame_identity, array_identity, implementation_identity
from analysis.clustering import kmeans_labels, fitted_space
from data_io.adapted import read_thymus_pair, read_simulation_pair
from analysis.plotting import clean_labels
from analysis.sampling import proportional_selections
from analysis.clustering import stack_selected
from analysis.clustering import scale_full_then_select
from analysis.clustering import apply_saved_scaler

from data_io.visualization_inputs import load_spatch_modality_subset, preprocess_adapted_rna_protein_inputs, preprocess_misar_inputs, preprocess_mousebrain_inputs, preprocess_paired_rna_protein_inputs

DATASETS = ('human_embryo', 'misar_seq', 'mousebrain', 'spatch', 'crc_stereocite', 'human_lymph_node', 'mouse_spleen', 'mouse_thymus', 'simulation')

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
            labels = np.load(directory / f'labels_{section}.npy', mmap_mode='r')
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
            table = pd.read_csv(directory / f'labels_{section}.csv')
            if 'cluster' not in table:
                raise KeyError(f'{directory}: missing cluster column for {section}.')
            chunks.append(table['cluster'].to_numpy(dtype=np.int32)[selections[section]])
        result[k] = np.concatenate(chunks)
    return result

def load_paired_rna_protein(dataset: str, display_name: str, analysis_sections: list[str], saved_names: dict[str, str], source_dirs: dict[str, str], maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    counts = {section: int(np.load(run_dir / f'final_embeddings_{saved_names[section]}.npy', mmap_mode='r').shape[0]) for section in analysis_sections}
    selections = proportional_selections(analysis_sections, counts, maximum, seed)
    metadata = section_manifest(analysis_sections, selections, {})
    (input_features, spot_ids) = preprocess_paired_rna_protein_inputs(dataset, run_dir, analysis_sections, saved_names, source_dirs, selections, data_dir=data_dir)
    metadata['spot_id'] = np.concatenate([spot_ids[section] for section in analysis_sections])
    primary_label = 'reference_annotation'
    metadata[primary_label] = 'not_available'
    if dataset == 'human_lymph_node':
        annotation_path = data_dir / 'Human_Lymph_Node_A1/annotation.csv'
        annotation = pd.read_csv(annotation_path).set_index('Barcode')['manual-anno']
        mask = metadata['section'].eq('Human_Lymph_Node_A1')
        metadata.loc[mask, primary_label] = metadata.loc[mask, 'spot_id'].map(annotation).fillna('NA')
    final_arrays = {section: np.load(run_dir / f'final_embeddings_{saved_names[section]}.npy', mmap_mode='r') for section in analysis_sections}
    final_selected = stack_selected(final_arrays, analysis_sections, selections)
    integrated = apply_saved_scaler(final_selected, analysis_dir / 'standardized_embedding/scaler_parameters.npz', 'joint')
    joint = load_csv_joint_labels(analysis_dir / 'standardized_embedding/clustering', analysis_sections, selections)
    scope = 'deterministic_visualization_sample' if sum(counts.values()) > sum((len(v) for v in selections.values())) else 'full_dataset'
    return DatasetBundle(dataset, display_name, run_dir, analysis_dir, analysis_sections, counts, selections, metadata, input_features, integrated, primary_label, joint, 5, {'RNA': f'training-matched Harmony RNA features; preprocessing scope={scope}', 'Protein': f'training-matched Harmony protein features; preprocessing scope={scope}', 'preprocessing_scope': scope})

def load_crc_stereocite(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    sections = ['CRC_003_bin20', 'CRC_006_bin20']
    return load_paired_rna_protein('crc_stereocite', 'CRC Stereo-CITE-seq', sections, {'CRC_003_bin20': 'CRC_003', 'CRC_006_bin20': 'CRC_006'}, {section: section for section in sections}, maximum, seed, run_dir=run_dir, data_dir=data_dir, analysis_dir=analysis_dir)

def load_human_lymph_node(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    sections = ['Human_Lymph_Node_A1', 'Human_Lymph_Node_D1']
    return load_paired_rna_protein('human_lymph_node', 'Human Lymph Node', sections, {section: section for section in sections}, {section: section for section in sections}, maximum, seed, run_dir=run_dir, data_dir=data_dir, analysis_dir=analysis_dir)

def load_mouse_spleen(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    sections = ['Mouse_Spleen1', 'Mouse_Spleen2']
    return load_paired_rna_protein('mouse_spleen', 'Mouse Spleen', sections, {section: section for section in sections}, {section: section for section in sections}, maximum, seed, run_dir=run_dir, data_dir=data_dir, analysis_dir=analysis_dir)

def load_adapted_rna_protein(dataset: str, display_name: str, read_pair, data_name: str, maximum: int, seed: int, *, hvg_rna: int, hvg_protein: int | None, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    summary = json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8'))
    sections = list(summary['section_names'])
    counts = {section: int(summary['final_embedding_shapes'][section][0]) for section in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    metadata = section_manifest(sections, selections, {})
    chunks = []
    for section in sections:
        frame = pd.read_csv(run_dir / f'obs_metadata_{section}.csv').iloc[selections[section]].reset_index(drop=True)
        frame['spot_id'] = frame['obs_name'].astype(str)
        frame = frame.drop(columns=['section'], errors='ignore')
        chunks.append(frame)
    metadata = pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)
    if dataset == 'simulation':
        primary_label = 'spatial_domain'
        metadata[primary_label] = clean_labels(metadata[primary_label])
    else:
        primary_label = 'reference_annotation'
        metadata[primary_label] = 'not_available'
    input_features = preprocess_adapted_rna_protein_inputs(read_pair, data_dir, run_dir, sections, selections, hvg_rna=hvg_rna, hvg_protein=hvg_protein)
    final_arrays = {section: np.load(run_dir / f'final_embeddings_{section}.npy', mmap_mode='r') for section in sections}
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(final_selected, analysis_dir / 'standardized_embedding/scaler_parameters.npz', 'joint')
    joint = load_csv_joint_labels(analysis_dir / 'standardized_embedding/clustering', sections, selections)
    return DatasetBundle(dataset, display_name, run_dir, analysis_dir, sections, counts, selections, metadata, input_features, integrated, primary_label, joint, 5, {'RNA': 'reconstructed full-run Harmony RNA model-input features', 'Protein': 'reconstructed full-run Harmony protein model-input features', 'preprocessing_scope': 'full_dataset'})

def load_mouse_thymus(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    return load_adapted_rna_protein('mouse_thymus', 'Mouse Thymus', read_thymus_pair, 'Mouse_Thymus', maximum, seed, hvg_rna=3000, hvg_protein=None, run_dir=run_dir, data_dir=data_dir, analysis_dir=analysis_dir)

def load_simulation(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    return load_adapted_rna_protein('simulation', 'Simulation', read_simulation_pair, 'Simulation', maximum, seed, hvg_rna=1000, hvg_protein=100, run_dir=run_dir, data_dir=data_dir, analysis_dir=analysis_dir)

def load_human(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    summary = json.loads((run_dir / 'run_summary.json').read_text())
    sections = list(summary['section_order'])
    counts = {s: int(summary['final_embedding_shapes'][s][0]) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    metadata = section_manifest(sections, selections, {})
    meta_chunks = []
    for section in sections:
        path = Path(summary['preprocess_manifest']['annotation_files'][section])
        header = pd.read_csv(path, nrows=0).columns
        wanted = [c for c in ('obs_name', 'cellid', 'celltype', 'stage') if c in header]
        frame = pd.read_csv(path, usecols=wanted).iloc[selections[section]].reset_index(drop=True)
        id_col = 'obs_name' if 'obs_name' in frame else 'cellid' if 'cellid' in frame else None
        frame['spot_id'] = frame[id_col].astype(str) if id_col is not None else [f'{section}:{i}' for i in selections[section]]
        meta_chunks.append(frame)
    labels = pd.concat(meta_chunks, ignore_index=True)
    metadata = pd.concat([metadata, labels], axis=1)
    metadata['celltype'] = clean_labels(metadata['celltype'])
    input_arrays = {section: np.load(summary['preprocess_manifest']['feature_files'][section], mmap_mode='r') for section in sections}
    input_features = {'RNA': scale_full_then_select(input_arrays, sections, selections)}
    final_arrays = {section: np.load(summary['saved_files']['final_embeddings'][section], mmap_mode='r') for section in sections}
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(final_selected, analysis_dir / 'cache/scaler_parameters.npz', 'joint')
    joint = load_npy_joint_labels(analysis_dir / 'clustering', sections, selections)
    return DatasetBundle('human_embryo', 'Human embryo', run_dir, analysis_dir, sections, counts, selections, metadata, input_features, integrated, 'celltype', joint, 10, {'RNA': 'saved full-run Harmony RNA features; scaler fitted on all spots', 'preprocessing_scope': 'full_dataset'})

def load_misar(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    summary = json.loads((run_dir / 'run_summary.json').read_text())
    sections = list(summary['section_names'])
    counts = {s: int(summary['final_embedding_shapes'][s][0]) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {s: str(summary['section_info'][s]['stage']) for s in sections}
    metadata = section_manifest(sections, selections, aliases)
    chunks = []
    for section in sections:
        frame = pd.read_csv(run_dir / f'obs_metadata_{section}.csv').iloc[selections[section]].reset_index(drop=True)
        frame['spot_id'] = frame['obs_name'].astype(str)
        frame = frame.drop(columns=['section'], errors='ignore')
        chunks.append(frame)
    metadata = pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)
    metadata['Combined_Clusters_annotation'] = clean_labels(metadata['Combined_Clusters_annotation'])
    input_full = preprocess_misar_inputs(run_dir, sections, data_dir=data_dir)
    input_features = {}
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    for (modality, values) in input_full.items():
        chunks = [values[offsets[i] + selections[s]] for (i, s) in enumerate(sections)]
        input_features[modality] = np.vstack(chunks).astype(np.float32)
    final_arrays = {s: np.load(run_dir / f'final_embeddings_{s}.npy', mmap_mode='r') for s in sections}
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(final_selected, analysis_dir / 'standardized_embedding/scaler_parameters.npz', 'joint')
    joint = load_csv_joint_labels(analysis_dir / 'standardized_embedding/clustering', sections, selections)
    return DatasetBundle('misar_seq', 'MISAR-seq', run_dir, analysis_dir, sections, counts, selections, metadata, input_features, integrated, 'Combined_Clusters_annotation', joint, 5, {'RNA': 'reconstructed full-run Harmony RNA model-input features', 'ATAC': 'reconstructed full-run Harmony ATAC model-input features', 'preprocessing_scope': 'full_dataset'})

def load_mousebrain(maximum: int, seed: int, *, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    config = json.loads((run_dir / 'mousebrain_config_used.json').read_text())
    sections = list(config['section_order'])
    counts = {section: int(np.load(run_dir / f'final_embeddings/{section}_final_embedding.npy', mmap_mode='r').shape[0]) for section in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {'s1': 'SectionA', 's2': 'SectionB', 's3': 'SectionC'}
    metadata = section_manifest(sections, selections, aliases)
    (input_full, section_results) = preprocess_mousebrain_inputs(config)
    chunks = []
    for (index, section) in enumerate(sections):
        rna = section_results[index]['modalities']['RNA']
        frame = rna.obs.copy().iloc[selections[section]].reset_index(drop=True)
        frame['spot_id'] = rna.obs_names.astype(str).to_numpy()[selections[section]]
        chunks.append(frame)
    metadata = pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)
    metadata['RegionLoupe'] = clean_labels(metadata['RegionLoupe'])
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    input_features = {modality: np.vstack([values[offsets[i] + selections[s]] for (i, s) in enumerate(sections)]).astype(np.float32) for (modality, values) in input_full.items()}
    final_arrays = {s: np.load(run_dir / f'final_embeddings/{s}_final_embedding.npy', mmap_mode='r') for s in sections}
    final_all = np.vstack([np.asarray(final_arrays[s], dtype=np.float32) for s in sections])
    (integrated_all, _) = fitted_space(final_all, 'standardized_embedding')
    integrated_all = integrated_all.astype(np.float32)
    integrated = np.vstack([integrated_all[offsets[i] + selections[s]] for (i, s) in enumerate(sections)])
    joint = {}
    from analysis.protocols import MOUSEBRAIN, MOUSEBRAIN_UMAP_KS
    for k in MOUSEBRAIN_UMAP_KS:
        labels = kmeans_labels(integrated_all, n_clusters=k, random_state=MOUSEBRAIN['SEED'], n_init=MOUSEBRAIN['N_INIT'], max_iter=MOUSEBRAIN['MAX_ITER']).astype(np.int32)
        joint[k] = np.concatenate([labels[offsets[i] + selections[s]] for (i, s) in enumerate(sections)])
    return DatasetBundle('mousebrain', 'MouseBrain', run_dir, analysis_dir, sections, counts, selections, metadata, input_features, integrated, 'RegionLoupe', joint, 10, {'RNA': 'reconstructed full-run Harmony RNA model-input features', 'Metabolite': 'reconstructed full-run Harmony metabolite model-input features', 'HE': 'reconstructed full-run Harmony HE model-input features', 'preprocessing_scope': 'full_dataset'})

def load_spatch(maximum: int, seed: int, *, feature_cache_dir: Path, run_dir: Path, data_dir: Path, analysis_dir: Path) -> DatasetBundle:
    summary = json.loads((run_dir / 'run_summary.json').read_text())
    sections = list(summary['section_names'])
    counts = {s: int(summary['alignment'][s]['n_spots']) for s in sections}
    selections = proportional_selections(sections, counts, maximum, seed)
    aliases = {'section1': 'Section 1', 'section2': 'Section 2'}
    metadata = section_manifest(sections, selections, aliases)
    full_meta = pd.read_csv(run_dir / 'spot_metadata.csv.gz', usecols=['section', 'original_rna_obs_name', 'x', 'y', 'cell_type_common', 'codex_coarse_label', 'spatial_cluster', 'high_quality'], low_memory=False)
    chunks = []
    for section in sections:
        frame = full_meta[full_meta['section'].eq(section)].reset_index(drop=True)
        frame = frame.iloc[selections[section]].reset_index(drop=True)
        frame['spot_id'] = [f'{section}:{int(x)}:{int(y)}' for (x, y) in zip(frame['x'], frame['y'])]
        frame = frame.drop(columns=['section'], errors='ignore')
        chunks.append(frame)
    metadata = pd.concat([metadata, pd.concat(chunks, ignore_index=True)], axis=1)
    metadata['cell_type_common'] = clean_labels(metadata['cell_type_common'])
    input_features = {}
    feature_cache_dir = check_output_path(feature_cache_dir, [run_dir])
    input_source = {'metadata': frame_identity(metadata), 'sections': sections, 'selection': {s: array_identity(selections[s]) for s in sections}, 'raw_inputs': [file_identity(data_dir / section / filename) for section in sections for filename in SPATCH_UMAP_INPUT_FILES[section].values()], 'markers': file_identity(run_dir / 'protein_markers_after_dapi_removal.csv'), 'implementation': implementation_identity('spatch-d8c-input-modalities-v1')}
    input_cache_hit = begin_analysis(feature_cache_dir, input_source, input_dirs=[run_dir])
    expected_n = sum((len(selections[section]) for section in sections))
    for modality in ('RNA', 'Protein', 'HE'):
        cache_path = feature_cache_dir / f'preprocessed_input_{modality.lower()}_features.npy'
        if input_cache_hit:
            cached = np.load(cache_path, mmap_mode='r')
            if cached.ndim == 2 and cached.shape[0] == expected_n and (cached.shape[1] > 1):
                print(f'[umap] reusing {cache_path}', flush=True)
                input_features[modality] = cached
                continue
            raise ValueError(f'Invalid saved input feature shape: {cache_path}; use a new output directory')
        print(f'[umap] SPATCH sampled input preprocessing: {modality}', flush=True)
        features = load_spatch_modality_subset(modality, sections, selections, data_dir=data_dir, run_dir=run_dir)
        np.save(cache_path, np.asarray(features, dtype=np.float32))
        input_features[modality] = np.load(cache_path, mmap_mode='r')
    finish_analysis(feature_cache_dir, input_source)
    final_arrays = {s: np.load(run_dir / f'final_embeddings_{s}.npy', mmap_mode='r') for s in sections}
    final_selected = stack_selected(final_arrays, sections, selections)
    integrated = apply_saved_scaler(final_selected, analysis_dir / 'standardized_embedding/scaler_parameters.npz', 'combined')
    joint = load_csv_joint_labels(analysis_dir / 'standardized_embedding/clustering', sections, selections)
    return DatasetBundle('spatch', 'SPATCH', run_dir, analysis_dir, sections, counts, selections, metadata, input_features, integrated, 'cell_type_common', joint, 5, {'RNA': 'training-matched Harmony preprocessing fitted on the shared visualization sample', 'Protein': 'training-matched Harmony preprocessing fitted on the shared visualization sample', 'HE': 'training-matched Harmony preprocessing fitted on the shared visualization sample', 'preprocessing_scope': 'deterministic_visualization_sample'})


def load_input_integration(dataset, run_dir, data_dir, analysis_dir, *, maximum, seed, feature_cache_dir):
    """Select the existing dataset-specific visualization loader explicitly."""
    paths=dict(run_dir=Path(run_dir),data_dir=Path(data_dir),analysis_dir=Path(analysis_dir))
    if dataset=='mousebrain': return load_mousebrain(maximum,seed,**paths)
    if dataset=='human_embryo': return load_human(maximum,seed,**paths)
    if dataset=='misar_seq': return load_misar(maximum,seed,**paths)
    if dataset=='spatch': return load_spatch(maximum,seed,feature_cache_dir=feature_cache_dir,**paths)
    if dataset=='crc_stereocite': return load_crc_stereocite(maximum,seed,**paths)
    if dataset=='human_lymph_node': return load_human_lymph_node(maximum,seed,**paths)
    if dataset=='mouse_spleen': return load_mouse_spleen(maximum,seed,**paths)
    if dataset=='mouse_thymus': return load_mouse_thymus(maximum,seed,**paths)
    if dataset=='simulation': return load_simulation(maximum,seed,**paths)
    raise ValueError(f'Unsupported dataset: {dataset}')


def run_input_integration(args):
    from analysis.umap import generate_dataset
    if args.max_samples<=0: raise ValueError('--max-samples must be positive')
    output=check_output_path(args.output_dir,[args.run_dir,args.data_dir,args.analysis_dir])
    bundle=load_input_integration(args.dataset,args.run_dir,args.data_dir,args.analysis_dir,
        maximum=args.max_samples,seed=args.seed,feature_cache_dir=output/'input_modalities')
    generate_dataset(bundle,args,output_dir=output/'umap')
    return {'status':'PASS','dataset':args.dataset,'output_dir':str(output/'umap')}
