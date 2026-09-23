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

from analysis.visualization_preparation import load_spatch_modality_subset, preprocess_adapted_rna_protein_inputs, preprocess_misar_inputs, preprocess_mousebrain_inputs, preprocess_paired_rna_protein_inputs

from data_io.large_results import read_section_arrays, read_selected_obs, borrowed_input
from analysis.integration_workflow import (section_manifest, load_npy_joint_labels,
    load_csv_joint_labels, append_selected_metadata, select_modality_rows, load_saved_integration)

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

    def __post_init__(self):
        contract = borrowed_input(self.integrated_features,
            sections=self.metadata['section'].to_numpy() if 'section' in self.metadata else None,
            spot_ids=self.metadata['spot_id'].to_numpy() if 'spot_id' in self.metadata else None,
            evidence={'kind': 'workflow_selected_rows', 'selection': self.selections,
                      'identity_source': 'existing workflow metadata; row tokens are not new barcodes'},
            provenance={'input': self.input_provenance, 'metadata': self.metadata})
        self.integrated_features = contract.embedding


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
    integrated, joint = load_saved_integration(
        {section: run_dir / f'final_embeddings_{saved_names[section]}.npy' for section in analysis_sections},
        analysis_sections, selections, scaler_path=analysis_dir / 'standardized_embedding/scaler_parameters.npz',
        scaler_prefix='joint', clustering_dir=analysis_dir / 'standardized_embedding/clustering', label_format='csv')
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
        frame = read_selected_obs(run_dir / f'obs_metadata_{section}.csv', selections[section])
        chunks.append(frame)
    metadata = append_selected_metadata(metadata, chunks)
    if dataset == 'simulation':
        primary_label = 'spatial_domain'
        metadata[primary_label] = clean_labels(metadata[primary_label])
    else:
        primary_label = 'reference_annotation'
        metadata[primary_label] = 'not_available'
    input_features = preprocess_adapted_rna_protein_inputs(read_pair, data_dir, run_dir, sections, selections, hvg_rna=hvg_rna, hvg_protein=hvg_protein)
    integrated, joint = load_saved_integration(
        {section: run_dir / f'final_embeddings_{section}.npy' for section in sections}, sections, selections,
        scaler_path=analysis_dir / 'standardized_embedding/scaler_parameters.npz', scaler_prefix='joint',
        clustering_dir=analysis_dir / 'standardized_embedding/clustering', label_format='csv')
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
    metadata = append_selected_metadata(metadata, meta_chunks)
    metadata['celltype'] = clean_labels(metadata['celltype'])
    input_arrays = read_section_arrays({section: summary['preprocess_manifest']['feature_files'][section] for section in sections})
    input_features = {'RNA': scale_full_then_select(input_arrays, sections, selections)}
    integrated, joint = load_saved_integration(
        {section: summary['saved_files']['final_embeddings'][section] for section in sections}, sections, selections,
        scaler_path=analysis_dir / 'cache/scaler_parameters.npz', scaler_prefix='joint',
        clustering_dir=analysis_dir / 'clustering', label_format='npy')
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
        frame = read_selected_obs(run_dir / f'obs_metadata_{section}.csv', selections[section])
        chunks.append(frame)
    metadata = append_selected_metadata(metadata, chunks)
    metadata['Combined_Clusters_annotation'] = clean_labels(metadata['Combined_Clusters_annotation'])
    input_full = preprocess_misar_inputs(run_dir, sections, data_dir=data_dir)
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    input_features = select_modality_rows(input_full, sections, selections, offsets)
    integrated, joint = load_saved_integration(
        {s: run_dir / f'final_embeddings_{s}.npy' for s in sections}, sections, selections,
        scaler_path=analysis_dir / 'standardized_embedding/scaler_parameters.npz', scaler_prefix='joint',
        clustering_dir=analysis_dir / 'standardized_embedding/clustering', label_format='csv')
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
    metadata = append_selected_metadata(metadata, chunks)
    metadata['RegionLoupe'] = clean_labels(metadata['RegionLoupe'])
    offsets = np.cumsum([0, *[counts[s] for s in sections]])
    input_features = select_modality_rows(input_full, sections, selections, offsets)
    final_arrays = read_section_arrays({s: run_dir / f'final_embeddings/{s}_final_embedding.npy' for s in sections})
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
    metadata = append_selected_metadata(metadata, chunks)
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
    integrated, joint = load_saved_integration(
        {s: run_dir / f'final_embeddings_{s}.npy' for s in sections}, sections, selections,
        scaler_path=analysis_dir / 'standardized_embedding/scaler_parameters.npz', scaler_prefix='combined',
        clustering_dir=analysis_dir / 'standardized_embedding/clustering', label_format='csv')
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
