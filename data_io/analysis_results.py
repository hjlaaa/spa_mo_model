"""Mechanical paired-result reading and identity alignment; no analysis policy imports."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class AlignedAnalysisInput:
    """Borrowed arrays in their existing row order; truth is never inferred."""
    embedding: np.ndarray
    sections: np.ndarray
    spot_ids: np.ndarray
    coords: np.ndarray
    truth: Any
    identity_evidence: dict
    provenance: dict

    def legacy(self, *, name, data_dir, output_root, shared_sources):
        # Keep the old absent-truth attribute/key (not an invented empty truth array).
        return dict(name=name, embedding=self.embedding, sections=self.sections,
                    barcodes=self.spot_ids, coords=self.coords, data_dir=data_dir,
                    source_paths=self.provenance['source_paths'], output_root=output_root,
                    shared_sources=shared_sources)


def aligned_result(embedding, sections, spot_ids, coords, source_paths, *, evidence):
    return AlignedAnalysisInput(embedding, sections, spot_ids, coords, None,
                                evidence, {'source_paths': source_paths, 'truth_status': 'not_loaded'})


@dataclass(frozen=True)
class PairedResultLayout:
    section_order: Any
    saved_names: Any
    embedding_template: str = 'final_embeddings_{section}.npy'
    names_template: str | None = None
    spatial_before_rna: bool = False
    validate_source_coords: bool = True
    unique_indices: bool = True
    source_paths_include_rna: bool = False
    saved_coords_output: bool = False
    shape_row_count: bool = False
    error_template: str = 'spa_mo_model {section}: embedding/index/spatial mismatch'
    alignment_examples: bool = True


def _align_positions(source, wanted, rna_path, *, examples):
    positions = pd.Index(source).get_indexer(pd.Index(wanted.astype(str)))
    if (positions < 0).any():
        message = f'{rna_path}: {int((positions < 0).sum())} barcodes failed alignment'
        if examples:
            message += f'; examples={pd.Index(wanted)[positions < 0][:5].tolist()}'
        raise ValueError(message)
    return positions


def aligned_coords(data_dir, sections, barcodes, *, section_order, examples=True):
    """External-method coordinate lookup, preserving each adapter's error text."""
    coords = np.empty((len(sections), 2), dtype=float)
    for section in section_order:
        mask = sections == section
        path = data_dir / section / 'adata_RNA.h5ad'
        rna = ad.read_h5ad(path, backed='r')
        try:
            positions = _align_positions(rna.obs_names.astype(str), barcodes[mask], path, examples=examples)
            coords[mask] = np.asarray(rna.obsm['spatial'])[positions, :2]
        finally:
            rna.file.close()
    return coords


def read_paired_result(run, data_dir, *, layout: PairedResultLayout):
    """One RNA open per section; keep legacy local checks and deferred ID errors.

    The former second pass did get_indexer after all local file/shape checks.
    Save its small result/error while the first handle is open, and surface the
    error at that same logical boundary. Do not retain full raw matrices/AnnData.
    """
    arrays, section_values, barcode_values, sources, coordinates = [], [], [], [], []
    aligned = []
    for section in layout.section_order:
        short = layout.saved_names[section]
        embedding_path = run / layout.embedding_template.format(section=short)
        index_path = run / f'selected_spot_indices_{short}.npy'
        spatial_path = run / f'spatial_{short}.npy'
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        embedding = np.load(embedding_path)
        names_path = None
        names = None
        if layout.names_template is not None:
            names_path = run / layout.names_template.format(section=short)
            names = np.load(names_path, allow_pickle=True).astype(str)
        indices = np.load(index_path).astype(int)
        if layout.spatial_before_rna:
            saved_coords = np.load(spatial_path)[:, :2]
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            source_names = rna.obs_names.astype(str).to_numpy()
            own_names = source_names[indices]
            if layout.validate_source_coords:
                own_coords = np.asarray(rna.obsm['spatial'])[indices, :2]
            # Source names were already obtained by integer selection. We still
            # execute the original pandas uniqueness/alignment check (including
            # duplicates outside the selected subset), without reopening RNA.
            if not layout.saved_coords_output:
                try:
                    positions = _align_positions(source_names, own_names if names is None else names,
                                                 rna_path, examples=layout.alignment_examples)
                    # The later name equality check guarantees these are the same rows.
                    aligned.append((None, own_coords))
                except Exception as exc:
                    aligned.append(((type(exc), exc.args), None))
        finally:
            rna.file.close()
        if not layout.spatial_before_rna:
            saved_coords = np.load(spatial_path)[:, :2]
        if names is not None:
            invalid = len(embedding) != len(names) or not np.array_equal(names, own_names) or not np.allclose(saved_coords, own_coords)
        elif layout.saved_coords_output:
            invalid = len(embedding) != len(indices) or len(saved_coords) != len(indices)
        else:
            row_count = embedding.shape[0] if layout.shape_row_count else len(embedding)
            invalid = row_count != len(indices) or (layout.unique_indices and len(np.unique(indices)) != len(indices)) or not np.allclose(saved_coords, own_coords)
        if invalid:
            raise ValueError(layout.error_template.format(section=section, run_parent=run.parent.name))
        if names is None:
            names = own_names
        arrays.append(embedding)
        section_values.extend([section] * len(names))
        barcode_values.extend(names.tolist())
        if layout.saved_coords_output:
            coordinates.append(saved_coords)
        sources.append(embedding_path)
        if names_path is not None:
            sources.append(names_path)
        sources.extend([index_path, spatial_path])
        if layout.source_paths_include_rna:
            sources.append(rna_path)
    sections = np.asarray(section_values, dtype=str)
    barcodes = np.asarray(barcode_values, dtype=str)
    embedding = np.vstack(arrays)
    if layout.saved_coords_output:
        coords = np.vstack(coordinates)
    else:
        coords = np.empty((len(sections), 2), dtype=float)
        for section, (error, values) in zip(layout.section_order, aligned):
            if error is not None:
                error_type, args = error
                raise error_type(*args)
            coords[sections == section] = values
    return aligned_result(embedding, sections, barcodes, coords, sources, evidence={
        'kind': 'selected_raw_obs_names', 'section_order': list(layout.section_order),
        'barcode_lookup_checked': not layout.saved_coords_output,
        'source_coordinates_checked': layout.validate_source_coords,
        'coordinates_source': 'saved_array' if layout.saved_coords_output else 'raw_spatial',
    })
