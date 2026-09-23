"""Explicit external result file layouts; no analysis/protocol imports.

MouseBrain and Simulation readers return borrowed format fields before analysis truth derivation.
Other adapters retain explicit mechanical identity alignment arguments. These readers
never infer a format from method names, dimensions or filenames.
"""
from data_io.reference_results import (read_result_table, prefix_columns, merged_embedding,
    table_identity, stack_embedding_files, validate_embedding_table)
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from .multisection_results import contract_payload, read_saved_raw_identity, read_full_rna_identity, iter_saved_section_files

def _mousebrain_payload(name, embedding, sections, barcodes, coords, truth, data_dir, source_paths, output_root, shared_sources):
    return contract_payload(name=name, embedding=embedding, sections=sections, barcodes=barcodes, coords=coords, truth=truth, data_dir=data_dir, source_paths=source_paths, output_root=output_root, shared_sources=shared_sources)

def _simulation_payload(name, embedding, sections, barcodes, truth, coords, data_dir, source_paths, output_root, shared_sources):
    return contract_payload(name=name, embedding=embedding, sections=sections, barcodes=barcodes, truth=truth, coords=coords, data_dir=data_dir, source_paths=source_paths, output_root=output_root, shared_sources=shared_sources)

def _mouse_thymus_payload(name, embedding, sections, barcodes, coords, data_dir, source_paths, output_root, shared_sources):
    return contract_payload(name=name, embedding=embedding, sections=sections, barcodes=barcodes, coords=coords, data_dir=data_dir, source_paths=source_paths, output_root=output_root, shared_sources=shared_sources)

def _misar_seq_payload(name, embedding, sections, barcodes, coords, truth, data_dir, source_paths, output_root, shared_sources):
    return contract_payload(name=name, embedding=embedding, sections=sections, barcodes=barcodes, coords=coords, truth=truth, data_dir=data_dir, source_paths=source_paths, output_root=output_root, shared_sources=shared_sources)

def mousebrain_load_spa(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays, section_arr, barcode_arr, sources, snapshots = read_full_rna_identity(
        run, data_dir, section_order=section_order, section_dirs=section_dirs,
        label_keys=label_keys, error_template='spa_mo_model {section}: embedding/data row mismatch')
    return run, data_dir, arrays, section_arr, barcode_arr, sources, snapshots

def mousebrain_load_mofa(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = read_result_table(table_path)
    factor_cols = prefix_columns(table, 'Factor')
    section_map = {'SectionA': 's1', 'SectionB': 's2', 'SectionC': 's3'}
    sections, barcodes = table_identity(table, section_column='section', spot_column='original_barcode', section_aliases=section_map)
    data_dir = Path(paths['data_dir'])
    return old, table_path, table, factor_cols, sections, barcodes, data_dir

def mousebrain_load_cosie(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path)
    emb_cols = prefix_columns(table, 'COSIE')
    sections, barcodes = table_identity(table, section_column='section', spot_column='obs_name')
    data_dir = Path(paths['data_dir'])
    return run, old, table_path, table, emb_cols, sections, barcodes, data_dir

def mousebrain_load_spamosaic(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    section_map = {'SectionA': 's1', 'SectionB': 's2', 'SectionC': 's3'}
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode', section_aliases=section_map)
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    return h5ad_path, adata, sections, barcodes, embedding, data_dir

def simulation_load_spa(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays = []
    sections = []
    barcodes = []
    sources: list[Path] = []
    for section, embedding, meta, emb_path, meta_path in iter_saved_section_files(run, section_order, metadata_first=True):
        arrays.append(embedding)
        sections.extend([section] * len(meta))
        barcodes.extend(meta['obs_name'].astype(str).tolist())
        sources.extend([emb_path, meta_path])
    section_arr = np.asarray(sections, dtype=str)
    barcode_arr = np.asarray(barcodes, dtype=str)
    return run, data_dir, arrays, section_arr, barcode_arr, sources

def simulation_load_mofa(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata.csv'
    table = read_result_table(table_path)
    factor_cols = prefix_columns(table, 'Factor')
    sections, barcodes = table_identity(table, section_column='group', spot_column='original_barcode')
    data_dir = Path(paths['data_dir'])
    return old, table_path, table, factor_cols, sections, barcodes, data_dir

def simulation_load_cosie(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path, index_col=0)
    emb_cols = prefix_columns(table, 'COSIE')
    barcode_col = 'obs_name.1' if 'obs_name.1' in table else 'obs_name'
    sections, barcodes = table_identity(table, section_column='section', spot_column=barcode_col)
    data_dir = Path(paths['data_dir'])
    return old, table_path, table, emb_cols, sections, barcodes, data_dir

def simulation_load_spamosaic(paths, output_root, *, section_order, section_dirs=None, label_keys=()):
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    return h5ad_path, adata, sections, barcodes, embedding, data_dir

def mouse_thymus_load_spa(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays, section_array, barcode_array, sources, snapshots = read_saved_raw_identity(
        run, data_dir, section_order=section_order, check_xy=True)
    old = run / 'clustering_analysis'
    return _mouse_thymus_payload('spa_mo_model', np.vstack(arrays), section_array, barcode_array, align(data_dir, section_array, barcode_array, snapshots=snapshots), data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def mouse_thymus_load_mofa(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = read_result_table(table_path)
    factor_columns = prefix_columns(table.columns, 'Factor')
    sections, barcodes = table_identity(table, section_column='section', spot_column='original_barcode')
    data_dir = Path(paths['data_dir'])
    return _mouse_thymus_payload('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, align(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv', 'input_audit.csv': old / 'input_audit.csv'})

def mouse_thymus_load_cosie(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path)
    embedding_columns = prefix_columns(table.columns, 'COSIE')
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in section_order]
    final_embedding = stack_embedding_files(final_paths)
    validate_embedding_table(final_embedding, table_embedding, error='COSIE: analysis table is not the final embedding')
    barcode_column = 'obs_name.1' if 'obs_name.1' in table.columns else 'obs_name'
    sections, barcodes = table_identity(table, section_column='section', spot_column=barcode_column)
    data_dir = Path(paths['data_dir'])
    return _mouse_thymus_payload('COSIE', table_embedding, sections, barcodes, align(data_dir, sections, barcodes), data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def mouse_thymus_load_spamosaic(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return _mouse_thymus_payload('SpaMosaic', embedding, sections, barcodes, align(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})

def misar_seq_load_spa(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    arrays, section_array, barcode_array, sources, snapshots = read_saved_raw_identity(
        run, data_dir, section_order=section_order, label_keys=label_keys)
    coords, truth = align(data_dir, section_array, barcode_array, snapshots=snapshots)
    old = run / 'clustering_analysis_k8_10_12_14_16'
    return _misar_seq_payload('spa_mo_model', np.vstack(arrays), section_array, barcode_array, coords, truth, data_dir, sources, output_root, {'batch_correction_metrics.csv': old / 'batch_correction_metrics.csv'})

def misar_seq_load_mofa(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = read_result_table(table_path)
    factor_columns = prefix_columns(table, 'Factor')
    sections, barcodes = table_identity(table, section_column='section', spot_column='original_barcode')
    data_dir = Path(paths['data_dir'])
    (coords, truth) = align(data_dir, sections, barcodes)
    return _misar_seq_payload('MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, coords, truth, data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})

def misar_seq_load_cosie(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path)
    embedding_columns = prefix_columns(table, 'COSIE')
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in section_order]
    final_embedding = stack_embedding_files(final_paths)
    validate_embedding_table(final_embedding, table_embedding, error='COSIE: analysis table is not the final embedding')
    barcode_column = 'original_barcode' if 'original_barcode' in table else 'obs_name'
    sections, barcodes = table_identity(table, section_column='section', spot_column=barcode_column)
    data_dir = Path(paths['data_dir'])
    (coords, truth) = align(data_dir, sections, barcodes)
    return _misar_seq_payload('COSIE', table_embedding, sections, barcodes, coords, truth, data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})

def misar_seq_load_spamosaic(paths, output_root, *, align, section_order, section_dirs=None, label_keys=()):
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    (coords, truth) = align(data_dir, sections, barcodes)
    old = Path(paths['analysis_dir'])
    return _misar_seq_payload('SpaMosaic', embedding, sections, barcodes, coords, truth, data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment.csv': old / 'metrics' / 'modality_alignment.csv'})

