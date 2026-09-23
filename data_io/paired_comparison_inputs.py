"""Explicit CRC/HLN/Spleen comparison layouts; no clustering/truth policy imports."""
from __future__ import annotations
from dataclasses import dataclass
from data_io.reference_results import (read_result_table, prefix_columns, merged_embedding,
    table_identity, stack_embedding_files, validate_embedding_table)
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from .analysis_results import PairedResultLayout, read_paired_result, aligned_coords, aligned_result


def _method_input(factory, name, embedding, sections, barcodes, coords, data_dir, source_paths, output_root, shared_sources):
    result = aligned_result(embedding, sections, barcodes, coords, source_paths,
                            evidence={'kind': 'explicit_section_barcode_lookup'})
    return factory(**result.legacy(name=name, data_dir=data_dir, output_root=output_root,
                                   shared_sources=shared_sources))


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


def mouse_spleen_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray):
    return aligned_coords(data_dir, sections, barcodes, section_order=MOUSE_SPLEEN_SECTIONS, examples=True)


def mouse_spleen_load_spa(paths, output_root) -> mouse_spleen_MethodData:
    run, data_dir = Path(paths['run_dir']), Path(paths['data_dir'])
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        MOUSE_SPLEEN_SECTIONS, {s: s for s in MOUSE_SPLEEN_SECTIONS}, spatial_before_rna=False, error_template='spa_mo_model {section}: embedding/index/spatial mismatch'))
    return mouse_spleen_MethodData(**result.legacy(name='spa_mo_model', data_dir=data_dir,
        output_root=output_root, shared_sources={'batch_correction_metrics.csv': run / 'clustering_analysis' / 'batch_correction_metrics.csv'}))


def mouse_spleen_load_mofa(paths, output_root) -> mouse_spleen_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = read_result_table(table_path)
    factor_columns = prefix_columns(table.columns, 'Factor')
    sections, barcodes = table_identity(table, section_column='section', spot_column='original_barcode')
    data_dir = Path(paths['data_dir'])
    return _method_input(mouse_spleen_MethodData, 'MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv', 'input_audit.csv': old / 'input_audit.csv'})


def mouse_spleen_load_cosie(paths, output_root) -> mouse_spleen_MethodData:
    run = Path(paths['run_dir'])
    old = run / 'analysis'
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path)
    embedding_columns = prefix_columns(table.columns, 'COSIE')
    table_embedding = table[embedding_columns].to_numpy(float)
    final_paths = [run / 'final_embeddings' / f'{section}_final_embedding.npy' for section in MOUSE_SPLEEN_SECTIONS]
    final_embedding = stack_embedding_files(final_paths)
    validate_embedding_table(final_embedding, table_embedding, error='COSIE: analysis table is not the saved final embedding')
    barcode_column = 'obs_name.1' if 'obs_name.1' in table.columns else 'obs_name'
    sections, barcodes = table_identity(table, section_column='section', spot_column=barcode_column)
    data_dir = Path(paths['data_dir'])
    return _method_input(mouse_spleen_MethodData, 'COSIE', table_embedding, sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path, *final_paths], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})


def mouse_spleen_load_spamosaic(paths, output_root) -> mouse_spleen_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return _method_input(mouse_spleen_MethodData, 'SpaMosaic', embedding, sections, barcodes, mouse_spleen_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})


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
    return aligned_coords(data_dir, sections, barcodes, section_order=CRC_STEREOCITE_SECTIONS, examples=True)


CRC_STEREOCITE_SHORT_NAMES = {'CRC_003_bin20': 'CRC_003', 'CRC_006_bin20': 'CRC_006'}


def crc_stereocite_load_spa(paths, output_root) -> crc_stereocite_MethodData:
    run, data_dir = Path(paths['run_dir']), Path(paths['data_dir'])
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        CRC_STEREOCITE_SECTIONS, CRC_STEREOCITE_SHORT_NAMES, spatial_before_rna=True, error_template='spa_mo_model {section}: embedding/index/spatial mismatch'))
    return crc_stereocite_MethodData(**result.legacy(name='spa_mo_model', data_dir=data_dir,
        output_root=output_root, shared_sources={'batch_correction_metrics.csv': run / 'clustering_analysis_k8_10_15' / 'batch_correction_metrics.csv'}))


def crc_stereocite_load_mofa(paths, output_root) -> crc_stereocite_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    header = read_result_table(table_path, nrows=0)
    factor_columns = prefix_columns(header.columns, 'Factor')
    columns = [*factor_columns, 'section', 'original_barcode']
    table = read_result_table(table_path, usecols=columns, dtype={'section': str, 'original_barcode': str})
    sections = table['section'].to_numpy()
    barcodes = table['original_barcode'].to_numpy()
    data_dir = Path(paths['data_dir'])
    return _method_input(crc_stereocite_MethodData, 'MOFA+', table[factor_columns].to_numpy(float), sections, barcodes, crc_stereocite_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'r2_total_by_view_group.csv': old / 'tables' / 'r2_total_by_view_group.csv'})


def crc_stereocite_load_cosie(paths, output_root) -> crc_stereocite_MethodData:
    run, data_dir = Path(paths['run_dir']), Path(paths['data_dir'])
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        CRC_STEREOCITE_SECTIONS, CRC_STEREOCITE_SHORT_NAMES,
        embedding_template='final_embeddings/{section}_final_embedding.npy',
        names_template='obs_names_{section}.npy', spatial_before_rna=True,
        error_template='COSIE {section}: embedding/name/index/spatial mismatch'))
    return crc_stereocite_MethodData(**result.legacy(name='COSIE', data_dir=data_dir,
        output_root=output_root, shared_sources={'batch_correction_metrics.csv': run / 'analysis/metrics/batch_correction_metrics.csv'}))


def crc_stereocite_load_spamosaic(paths, output_root) -> crc_stereocite_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return _method_input(crc_stereocite_MethodData, 'SpaMosaic', embedding, sections, barcodes, crc_stereocite_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})


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


def human_lymph_node_aligned_coords(data_dir: Path, sections: np.ndarray, barcodes: np.ndarray):
    return aligned_coords(data_dir, sections, barcodes, section_order=HUMAN_LYMPH_NODE_SECTIONS, examples=False)


def human_lymph_node_load_spa(paths, output_root) -> human_lymph_node_MethodData:
    run, data_dir = Path(paths['run_dir']), Path(paths['data_dir'])
    result = read_paired_result(run, data_dir, layout=PairedResultLayout(
        HUMAN_LYMPH_NODE_SECTIONS, {s: s for s in HUMAN_LYMPH_NODE_SECTIONS}, spatial_before_rna=False, unique_indices=False, source_paths_include_rna=True, shape_row_count=True, alignment_examples=False, error_template='spa_mo_model {section}: saved rows/spatial do not align'))
    return human_lymph_node_MethodData(**result.legacy(name='spa_mo_model', data_dir=data_dir,
        output_root=output_root, shared_sources={'batch_correction_metrics.csv': run / 'clustering_analysis' / 'batch_correction_metrics.csv'}))


def human_lymph_node_load_mofa(paths, output_root) -> human_lymph_node_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'factors_with_metadata_and_coordinates.csv'
    table = read_result_table(table_path)
    factor_cols = prefix_columns(table, 'Factor')
    sections, barcodes = table_identity(table, section_column='section', spot_column='original_barcode')
    data_dir = Path(paths['data_dir'])
    shared = {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'input_audit.csv': old / 'input_audit.csv'}
    r2 = old / 'tables' / 'r2_total_by_view_group.csv'
    if r2.exists():
        shared['r2_total_by_view_group.csv'] = r2
    return _method_input(human_lymph_node_MethodData, 'MOFA+', table[factor_cols].to_numpy(float), sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, shared)


def human_lymph_node_load_cosie(paths, output_root) -> human_lymph_node_MethodData:
    old = Path(paths['analysis_dir'])
    table_path = old / 'tables' / 'embeddings_with_metadata.csv'
    table = read_result_table(table_path)
    emb_cols = prefix_columns(table, 'COSIE')
    barcode_col = 'obs_name.1' if 'obs_name.1' in table else 'obs_name'
    sections, barcodes = table_identity(table, section_column='section', spot_column=barcode_col)
    data_dir = Path(paths['data_dir'])
    return _method_input(human_lymph_node_MethodData, 'COSIE', table[emb_cols].to_numpy(float), sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [table_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv'})


def human_lymph_node_load_spamosaic(paths, output_root) -> human_lymph_node_MethodData:
    h5ad_path = Path(paths['embedding_file'])
    adata = ad.read_h5ad(h5ad_path)
    sections, barcodes = table_identity(adata.obs, section_column='section', spot_column='original_barcode')
    embedding = merged_embedding(adata, copy=True)
    data_dir = Path(paths['data_dir'])
    old = Path(paths['analysis_dir'])
    return _method_input(human_lymph_node_MethodData, 'SpaMosaic', embedding, sections, barcodes, human_lymph_node_aligned_coords(data_dir, sections, barcodes), data_dir, [h5ad_path], output_root, {'batch_correction_metrics.csv': old / 'metrics' / 'batch_correction_metrics.csv', 'modality_alignment_cosine.csv': old / 'metrics' / 'modality_alignment_cosine.csv'})


crc_stereocite_MethodData.__module__ = 'data_io.comparison_inputs'

human_lymph_node_MethodData.__module__ = 'data_io.comparison_inputs'

mouse_spleen_MethodData.__module__ = 'data_io.comparison_inputs'
