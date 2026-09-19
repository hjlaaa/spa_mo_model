"""Existing modality preparation for input/integration visualization (not training)."""
from __future__ import annotations
import gc
from pathlib import Path
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd
from analysis.protocols import SPATCH_UMAP_INPUT_FILES
from data_io.preprocessing import load_cosie_style_data
from data_io.datasets import build_cosie_data_dict, build_mousebrain_section
from data_io.paired import prepare_rna_var_names_make_unique, read_backed_pair as read_crc_backed_pair, subset_to_memory
from data_io.misar import common_var_names, make_unique_rna_var_names, read_backed_pair
from data_io.misar import SECTION_INFO as MISAR_SECTION_INFO
from analysis.clustering import scale_full_then_select

def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def selected_original_indices(run_dir: Path, saved_section: str, selection: np.ndarray) -> np.ndarray:
    trained = np.load(run_dir / f'selected_spot_indices_{saved_section}.npy', mmap_mode='r')
    return np.asarray(trained[selection], dtype=np.int64)

def subset_obs_vars_memory_safe(value: ad.AnnData, obs_indices: np.ndarray, var_names: list[str] | None, chunk_size: int=5000) -> ad.AnnData:
    """Subset backed matrices without using two h5py fancy index vectors."""
    obs_indices = np.asarray(obs_indices, dtype=np.int64)
    if not getattr(value, 'isbacked', False):
        view = value[obs_indices, :] if var_names is None else value[obs_indices, var_names]
        return view.copy()
    if var_names is None:
        return subset_to_memory(value, obs_indices, None)
    chunks = []
    for start in range(0, len(obs_indices), chunk_size):
        rows = obs_indices[start:start + chunk_size]
        chunk = value[rows, :].to_memory()
        chunks.append(chunk[:, var_names].copy())
    if not chunks:
        raise ValueError('Cannot preprocess an empty observation selection.')
    if len(chunks) == 1:
        return chunks[0]
    return ad.concat(chunks, axis=0, join='inner', merge='same', index_unique=None)

def preprocess_paired_rna_protein_inputs(dataset: str, run_dir: Path, sections: list[str], saved_names: dict[str, str], source_dirs: dict[str, str], selections: dict[str, np.ndarray], *, data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared_genes = read_lines(run_dir / 'shared_gene_symbols_make_unique.txt')
    marker_path = run_dir / 'adt_marker_list.txt'
    markers = read_lines(marker_path) if marker_path.is_file() else None
    rna_subsets: list[ad.AnnData] = []
    protein_subsets: list[ad.AnnData] = []
    spot_ids: dict[str, np.ndarray] = {}
    backed: list[ad.AnnData] = []
    try:
        for section in sections:
            saved = saved_names[section]
            source_dir = source_dirs[section]
            if dataset == 'crc_stereocite':
                (rna, protein) = read_crc_backed_pair(data_dir, source_dir)
                prepare_rna_var_names_make_unique(rna)
            else:
                folder = data_dir / source_dir
                rna = ad.read_h5ad(folder / 'adata_RNA.h5ad', backed='r')
                protein = ad.read_h5ad(folder / 'adata_ADT.h5ad', backed='r')
                if 'gene_ids' not in rna.var:
                    raise KeyError(f"{dataset}/{section}: RNA var['gene_ids'] is required.")
                rna.var_names = pd.Index(rna.var['gene_ids'].astype(str))
                if dataset == 'human_lymph_node':
                    if 'gene_ids' not in protein.var:
                        raise KeyError(f"{dataset}/{section}: ADT var['gene_ids'] is required.")
                    protein.var_names = pd.Index(protein.var['gene_ids'].astype(str))
            backed.extend([rna, protein])
            original = selected_original_indices(run_dir, saved, selections[section])
            spot_ids[section] = rna.obs_names.astype(str).to_numpy()[original]
            rna_subsets.append(subset_obs_vars_memory_safe(rna, original, shared_genes))
            protein_subsets.append(subset_obs_vars_memory_safe(protein, original, markers))
    finally:
        for value in backed:
            value.file.close()
    (feature_dict, _, _) = load_cosie_style_data({'RNA': rna_subsets, 'Protein': protein_subsets}, n_comps=50, hvg_num=3000, hvg_num_by_modality={'RNA': 3000, 'Protein': None}, target_sum=None, use_harmony=True, memory_efficient=dataset == 'crc_stereocite', retain_processed=False)
    features: dict[str, np.ndarray] = {}
    for modality in ('RNA', 'Protein'):
        arrays = {section: feature_dict[f's{i + 1}'][modality].detach().cpu().numpy() for (i, section) in enumerate(sections)}
        local = {section: np.arange(len(arrays[section]), dtype=np.int64) for section in sections}
        features[modality] = scale_full_then_select(arrays, sections, local)
    return (features, spot_ids)

def preprocess_adapted_rna_protein_inputs(read_pair, data_dir: Path, run_dir: Path, sections: list[str], selections: dict[str, np.ndarray], *, hvg_rna: int, hvg_protein: int | None) -> dict[str, np.ndarray]:
    shared_genes = read_lines(run_dir / 'shared_gene_symbols_make_unique.txt')
    shared_proteins = read_lines(run_dir / 'shared_adt_peaks.txt')
    rna_values: list[ad.AnnData] = []
    protein_values: list[ad.AnnData] = []
    for section in sections:
        (rna, protein) = read_pair(data_dir, section)
        make_unique_rna_var_names(section, rna)
        original = selected_original_indices(run_dir, section, selections[section])
        rna_values.append(subset_obs_vars_memory_safe(rna, original, shared_genes))
        protein_values.append(subset_obs_vars_memory_safe(protein, original, shared_proteins))
        if getattr(rna, 'isbacked', False):
            rna.file.close()
        if getattr(protein, 'isbacked', False):
            protein.file.close()
    hvg_protein = len(shared_proteins) if hvg_protein is None else hvg_protein
    (feature_dict, _, _) = load_cosie_style_data({'RNA': rna_values, 'Protein': protein_values}, n_comps=50, hvg_num=hvg_rna, hvg_num_by_modality={'RNA': hvg_rna, 'Protein': hvg_protein}, target_sum=None, use_harmony=True, retain_processed=False)
    result: dict[str, np.ndarray] = {}
    for modality in ('RNA', 'Protein'):
        arrays = {section: feature_dict[f's{i + 1}'][modality].detach().cpu().numpy() for (i, section) in enumerate(sections)}
        local = {section: np.arange(len(arrays[section]), dtype=np.int64) for section in sections}
        result[modality] = scale_full_then_select(arrays, sections, local)
    return result

def preprocess_misar_inputs(run_dir: Path, sections: list[str], *, data_dir: Path) -> dict[str, np.ndarray]:
    backed_rna: dict[str, ad.AnnData] = {}
    backed_atac: dict[str, ad.AnnData] = {}
    try:
        for section in sections:
            (backed_rna[section], backed_atac[section]) = read_backed_pair(data_dir, section, MISAR_SECTION_INFO)
            make_unique_rna_var_names(section, backed_rna[section])
        shared_genes = common_var_names(backed_rna, sections, None, 'RNA genes')
        shared_peaks = common_var_names(backed_atac, sections, None, 'ATAC peaks')
        selected = {section: np.load(run_dir / f'selected_spot_indices_{section}.npy') for section in sections}
        rna = [subset_to_memory(backed_rna[s], selected[s], shared_genes) for s in sections]
        atac = [subset_to_memory(backed_atac[s], selected[s], shared_peaks) for s in sections]
    finally:
        for value in [*backed_rna.values(), *backed_atac.values()]:
            value.file.close()
    (feature_dict, _, _) = load_cosie_style_data({'RNA': rna, 'ATAC': atac}, n_comps=50, hvg_num=3000, hvg_num_by_modality={'RNA': 3000, 'ATAC': 3000}, target_sum=None, use_harmony=True, retain_processed=False)
    result = {}
    for modality in ('RNA', 'ATAC'):
        arrays = {section: feature_dict[f's{i + 1}'][modality].detach().cpu().numpy() for (i, section) in enumerate(sections)}
        selections = {section: np.arange(len(arrays[section]), dtype=np.int64) for section in sections}
        result[modality] = scale_full_then_select(arrays, sections, selections)
    return result

def preprocess_mousebrain_inputs(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    section_results = [build_mousebrain_section(section_id=section['section_id'], rna_path=section['rna_input'], metabolite_path=section['metabolite_input'], spatial_key=config['preprocessing'].get('spatial_key', 'spatial'), uni_feature_key=config['preprocessing'].get('uni_feature_key', 'uni_feature'), rna_gene_id_key=config['preprocessing'].get('rna_gene_id_key', 'gene_ids')) for section in config['sections']]
    data_dict = build_cosie_data_dict(section_results)
    (feature_dict, _, _) = load_cosie_style_data(data_dict, n_comps=50, hvg_num=3000, hvg_num_by_modality={'RNA': 3000, 'Metabolite': None}, target_sum=None, use_harmony=True, retain_processed=False)
    sections = [section['section_id'] for section in config['sections']]
    result = {}
    for modality in ('RNA', 'Metabolite', 'HE'):
        arrays = {section: feature_dict[f's{i + 1}'][modality].detach().cpu().numpy() for (i, section) in enumerate(sections)}
        selections = {section: np.arange(len(arrays[section]), dtype=np.int64) for section in sections}
        result[modality] = scale_full_then_select(arrays, sections, selections)
    return (result, section_results)

def load_spatch_modality_subset(modality: str, sections: list[str], selections: dict[str, np.ndarray], *, data_dir: Path, run_dir: Path) -> np.ndarray:
    filenames = SPATCH_UMAP_INPUT_FILES
    backed = [ad.read_h5ad(data_dir / section / filenames[section][modality], backed='r') for section in sections]
    try:
        var_names: list[str] | None
        if modality == 'RNA':
            common = pd.Index(backed[0].var_names.astype(str))
            for value in backed[1:]:
                common = common.intersection(pd.Index(value.var_names.astype(str)))
            var_names = list(common)
        elif modality == 'Protein':
            marker_file = run_dir / 'protein_markers_after_dapi_removal.csv'
            var_names = pd.read_csv(marker_file)['protein_marker'].astype(str).tolist()
        else:
            var_names = None
        sampled = []
        for (value, section) in zip(backed, sections):
            subset = subset_to_memory(value, selections[section], None)
            if var_names is not None:
                missing = sorted(set(var_names).difference(subset.var_names.astype(str)))
                if missing:
                    raise KeyError(f'{modality} {section}: {len(missing)} requested variables are absent; examples={missing[:5]}')
                subset = subset[:, var_names].copy()
            sampled.append(subset)
    finally:
        for value in backed:
            value.file.close()
    hvg_by_modality = {'RNA': 3000, 'Protein': None, 'HE': None}
    (feature_dict, _, _) = load_cosie_style_data({modality: sampled}, n_comps=50, hvg_num=3000, hvg_num_by_modality={modality: hvg_by_modality[modality]}, target_sum=None, use_harmony=True, memory_efficient=True, retain_processed=False)
    arrays = {section: feature_dict[f's{i + 1}'][modality].detach().cpu().numpy() for (i, section) in enumerate(sections)}
    local = {section: np.arange(len(arrays[section]), dtype=np.int64) for section in sections}
    result = scale_full_then_select(arrays, sections, local)
    del feature_dict, sampled, arrays
    gc.collect()
    return result
