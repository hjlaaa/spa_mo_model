"""Explicit result tables and reference-row alignment; no scientific transforms."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import anndata as ad
from data_io.large_results import borrowed_input


def normalize_sections(values, aliases):
    cleaned = pd.Series(values, dtype='string').astype(str)
    if aliases is not None:
        cleaned = cleaned.map(aliases).fillna(cleaned)
    return cleaned.to_numpy(dtype=str)

def canonical_keys(sections: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(sections.astype(str), '\x1f'), identifiers.astype(str))


def source_id_column(columns: list[str]) -> str:
    for name in ('original_barcode', 'obs_name.1', 'obs_name', 'spot_id'):
        if name in columns:
            return name
    raise KeyError(f'No spot identifier column found among {columns}.')


def feature_columns(columns: list[str], pattern: str) -> list[str]:
    result = [column for column in columns if re.match(pattern, column)]
    return sorted(result, key=lambda value: int(re.search('\\d+', value).group()))


def selected_positions_from_keys(context: str, source_sections: np.ndarray, source_ids: np.ndarray, reference, *, section_aliases=None) -> np.ndarray:
    source_keys = canonical_keys(normalize_sections(source_sections, section_aliases), source_ids)
    if len(np.unique(source_keys)) != len(source_keys):
        raise ValueError(f'{context}: source section/spot identifiers are not unique.')
    reference_keys = canonical_keys(reference.metadata['section'].to_numpy(dtype=str), reference.metadata['spot_id'].to_numpy(dtype=str))
    positions = pd.Index(source_keys).get_indexer(reference_keys)
    if np.any(positions < 0):
        examples = reference_keys[positions < 0][:5].tolist()
        raise KeyError(f'{context}: {np.sum(positions < 0)} reference spots absent: {examples}')
    return positions.astype(np.int64)


def read_selected_result(source_path: Path, reference, *, file_format, feature_pattern, context, section_aliases=None) -> tuple[np.ndarray, dict[str, Any]]:
    if file_format == 'csv':
        columns = read_result_table(source_path, nrows=0).columns.tolist()
        features = feature_columns(columns, feature_pattern)
        id_column = source_id_column(columns)
        selected_columns = list(dict.fromkeys(['section', id_column, *features]))
        table = read_result_table(source_path, usecols=selected_columns, low_memory=False)
        positions = selected_positions_from_keys(context, table['section'].astype(str).to_numpy(), table[id_column].astype(str).to_numpy(), reference, section_aliases=section_aliases)
        values = table[features].to_numpy(dtype=np.float32)[positions]
        provenance = {'source_embedding': source_path, 'source_type': 'CSV feature columns', 'feature_columns': features, 'alignment': 'exact section + spot/barcode key'}
        return (borrowed_input(values, evidence={"kind": "explicit_section_spot_lookup"}, provenance=provenance).embedding, provenance)
    if file_format == 'h5ad':
        value = ad.read_h5ad(source_path, backed='r')
        try:
            if 'merged_emb' not in value.obsm:
                raise KeyError(f"{source_path}: obsm['merged_emb'] is required.")
            id_column = source_id_column(value.obs.columns.tolist())
            positions = selected_positions_from_keys(context, value.obs['section'].astype(str).to_numpy(), value.obs[id_column].astype(str).to_numpy(), reference, section_aliases=section_aliases)
            merged = merged_embedding(value, dtype=np.float32)
            values = merged[positions]
        finally:
            value.file.close()
        provenance = {'source_embedding': source_path, 'source_type': "AnnData obsm['merged_emb']", 'alignment': 'exact section + spot/barcode key'}
        return (borrowed_input(values, evidence={"kind": "explicit_section_spot_lookup"}, provenance=provenance).embedding, provenance)
    raise ValueError(f'Unsupported source embedding: {source_path}')


def select_by_section_rows(arrays: dict[str, np.ndarray], reference) -> np.ndarray:
    result = np.empty((len(reference.metadata), next(iter(arrays.values())).shape[1]), dtype=np.float32)
    for section in reference.sections:
        mask = reference.metadata['section'].eq(section).to_numpy()
        local = reference.metadata.loc[mask, 'section_row_index'].to_numpy(dtype=np.int64)
        if local.max(initial=-1) >= len(arrays[section]):
            raise IndexError(f'{reference.dataset}/{section}: reference row is out of range.')
        result[mask] = np.asarray(arrays[section][local], dtype=np.float32)
    return result


def validate_section_ids(ids_by_section: dict[str, np.ndarray], reference) -> None:
    for section in reference.sections:
        mask = reference.metadata['section'].eq(section).to_numpy()
        local = reference.metadata.loc[mask, 'section_row_index'].to_numpy(dtype=np.int64)
        observed = np.asarray(ids_by_section[section]).astype(str)[local]
        expected = reference.metadata.loc[mask, 'spot_id'].astype(str).to_numpy()
        if not np.array_equal(observed, expected):
            mismatch = int(np.flatnonzero(observed != expected)[0])
            raise ValueError(f'{reference.dataset}/{section}: source/reference spot mismatch at sample position {mismatch}: {observed[mismatch]} != {expected[mismatch]}')



def read_result_table(path, **read_options):
    """Explicit CSV options, no method or scientific policy."""
    return pd.read_csv(path, **read_options)


def prefix_columns(columns, prefix):
    """Legacy prefix layouts use file order, unlike numbered UMAP columns."""
    return [column for column in columns if column.startswith(prefix)]


def merged_embedding(value, *, dtype=None, copy=False):
    array = np.asarray(value.obsm['merged_emb'], dtype=dtype)
    return array.copy() if copy else array


def table_identity(table, *, section_column, spot_column, section_aliases=None):
    """Parse explicit format columns in file order, without choosing a cohort.

    Comparison aliases intentionally retain Series.map's missing-value behavior;
    reference aliases use normalize_sections and its established fallback instead.
    """
    sections = table[section_column].astype(str)
    if section_aliases is not None:
        sections = sections.map(section_aliases)
    return sections.to_numpy(), table[spot_column].astype(str).to_numpy()


def stack_embedding_files(paths):
    """Existing eager NPY stack in the caller's order; no cast or extra copy."""
    return np.vstack([np.load(path) for path in paths])


def validate_embedding_table(final_embedding, table_embedding, *, error):
    """Keep the original shape short circuit and np.allclose defaults."""
    if final_embedding.shape != table_embedding.shape or not np.allclose(final_embedding, table_embedding):
        raise ValueError(error)


def read_reference_metadata(path, *, context, section_aliases, drop_prefixes):
    """Read an existing reference cohort; no sampling, sorting or feature transform.

    The caller supplies fields to omit and aliases. Row selection already exists
    in this file; section_row_index and all biological columns are left intact.
    """
    metadata = read_result_table(path, low_memory=False)
    drop_columns = [column for column in metadata if any(column.startswith(prefix) for prefix in drop_prefixes)]
    metadata = metadata.drop(columns=drop_columns)
    metadata['section'] = normalize_sections(metadata['section'], section_aliases)
    metadata['spot_id'] = metadata['spot_id'].astype(str)
    if metadata.duplicated(['section', 'spot_id']).any():
        raise ValueError(f'{context}: duplicate reference section/spot_id keys.')
    return metadata
