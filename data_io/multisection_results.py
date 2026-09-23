"""Result/obs identity mechanics. Scientific label policy is supplied by analysis."""
from contextlib import contextmanager
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from .analysis_results import AlignedAnalysisInput, _align_positions


def contract_payload(**payload):
    """Borrow the existing arrays and preserve legacy field presence and order."""
    result = AlignedAnalysisInput(
        payload['embedding'], payload['sections'], payload['barcodes'], payload['coords'],
        payload.get('truth'), {'kind': 'caller_supplied_section_spot_ids',
         'embedding_identity': 'legacy row order; no new embedded barcode evidence',
         'alignment': 'existing adapter checks only'},
        {'source_paths': payload['source_paths'],
         'truth_status': 'loaded' if 'truth' in payload else 'not_loaded'})
    fields = dict(embedding=result.embedding, sections=result.sections,
                  barcodes=result.spot_ids, coords=result.coords, truth=result.truth,
                  source_paths=result.provenance['source_paths'])
    return {key: fields[key] if key in fields else value for key, value in payload.items()}


def capture_obs(rna, label_keys):
    """Retain only names, spatial and requested obs columns, not AnnData or X.

    These are the same to_numpy/asarray boundaries as the former second open.
    Missing spatial/labels are checked later, at the old alignment boundary.
    """
    return {'names': rna.obs_names.astype(str),
            'spatial': np.asarray(rna.obsm['spatial']) if 'spatial' in rna.obsm else None,
            'obs': {key: rna.obs[key].to_numpy() for key in label_keys if key in rna.obs}}


def aligned_obs_metadata(data_dir, sections, barcodes, *, section_order, section_dirs,
                         label_keys, clean_values, required_labels, snapshots=None):
    coords = np.empty((len(sections), 2), dtype=float)
    truth = {label: np.full(len(sections), np.nan, dtype=object) for label in label_keys}
    for section in section_order:
        mask = sections == section
        path = data_dir / section_dirs[section] / 'adata_RNA.h5ad'
        def consume(source):
            positions = _align_positions(source['names'], barcodes[mask], path, examples=True)
            if source['spatial'] is None:
                raise KeyError('spatial')
            coords[mask] = source['spatial'][positions, :2]
            for label in label_keys:
                if label not in source['obs']:
                    if required_labels:
                        raise ValueError(f'{path}: missing label {label}')
                else:
                    truth[label][mask] = clean_values(source['obs'][label][positions])
        if snapshots is not None:
            consume(snapshots[section])
        else:
            rna = ad.read_h5ad(path, backed='r')
            try:
                consume(capture_obs(rna, label_keys))
            finally:
                rna.file.close()
    return coords, truth


def read_full_rna_identity(run, data_dir, *, section_order, section_dirs, label_keys, error_template):
    arrays, sections, barcodes, sources, snapshots = [], [], [], [], {}
    for section in section_order:
        path = run / 'final_embeddings' / f'{section}_final_embedding.npy'
        embedding = np.load(path)
        rna = ad.read_h5ad(data_dir / section_dirs[section] / 'adata_RNA.h5ad', backed='r')
        try:
            names = rna.obs_names.astype(str).to_numpy()
            snapshots[section] = capture_obs(rna, label_keys)
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(error_template.format(section=section, rows=len(embedding), names=len(names)))
        arrays.append(embedding); sections.extend([section] * len(names)); barcodes.extend(names.tolist()); sources.append(path)
    section_values = np.asarray(sections)
    spot_ids = np.asarray(barcodes)
    return arrays, section_values, spot_ids, sources, snapshots


def read_full_rna_result(run, data_dir, *, section_order, section_dirs, label_keys,
                         clean_values, required_labels, error_template):
    arrays, section_values, spot_ids, sources, snapshots = read_full_rna_identity(
        run, data_dir, section_order=section_order, section_dirs=section_dirs,
        label_keys=label_keys, error_template=error_template)
    coords, truth = aligned_obs_metadata(data_dir, section_values, spot_ids,
        section_order=section_order, section_dirs=section_dirs, label_keys=label_keys,
        clean_values=clean_values, required_labels=required_labels, snapshots=snapshots)
    return np.vstack(arrays), section_values, spot_ids, coords, truth, sources


def read_misar_result(run: Path, data_dir: Path, *, section_order, label_keys, clean_values, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section, embedding, metadata, embedding_path, metadata_path in iter_saved_section_files(run, section_order):
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"MISAR-seq/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    coords, truth = aligned_obs_metadata(data_dir, section_array, barcode_array,
        section_order=section_order, section_dirs={s: s for s in section_order},
        label_keys=label_keys, clean_values=clean_values, required_labels=True)
    return contract_payload(name=name, embedding=np.vstack(arrays), sections=section_array,
                barcodes=barcode_array, coords=coords, truth=truth, data_dir=data_dir,
                source_paths=sources, output_root=output_root, shared_sources={})


def load_section_metadata(run: Path, *, section_order):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section, embedding, metadata, embedding_path, metadata_path in iter_saved_section_files(run, section_order):
        names = metadata["obs_name"].astype(str).to_numpy()
        spatial = metadata[["spatial_x", "spatial_y"]].to_numpy(float)
        if len(embedding) != len(names) or len(spatial) != len(names):
            raise ValueError(f"{run.parent.name}/{section}: saved row counts do not align")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        coords.append(spatial)
        sources.extend([embedding_path, metadata_path])
    return (
        np.vstack(arrays),
        np.asarray(sections, dtype=str),
        np.asarray(barcodes, dtype=str),
        np.vstack(coords),
        sources,
    )


def read_simulation_result(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    truth: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section, embedding, metadata, embedding_path, metadata_path in iter_saved_section_files(run, section_order):
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"Simulation/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        truth.extend(metadata["spatial_domain"].astype(str).tolist())
        coords.append(metadata[["spatial_x", "spatial_y"]].to_numpy(float))
        sources.extend([embedding_path, metadata_path])
    return contract_payload(name=name, embedding=np.vstack(arrays), sections=np.asarray(sections, dtype=str),
                barcodes=np.asarray(barcodes, dtype=str), truth=np.asarray(truth, dtype=str),
                coords=np.vstack(coords), data_dir=data_dir, source_paths=sources,
                output_root=output_root, shared_sources={})


def read_saved_raw_identity(run, data_dir, *, section_order, label_keys=(), check_xy=False):
    """Existing MISAR/Thymus comparison first pass; reuse captured obs later."""
    arrays, sections, barcodes, sources, snapshots = [], [], [], [], {}
    for section, embedding, metadata, embedding_path, metadata_path in iter_saved_section_files(run, section_order, metadata_first=True):
        rna = ad.read_h5ad(data_dir / section / 'adata_RNA.h5ad', backed='r')
        try:
            own_names = rna.obs_names.astype(str).to_numpy()
            if check_xy:
                own_coords = rna.obs[['x', 'y']].to_numpy()
                snapshots[section] = {'names': rna.obs_names.astype(str), 'xy': own_coords}
            else:
                snapshots[section] = capture_obs(rna, label_keys)
        finally:
            rna.file.close()
        saved_names = metadata['obs_name'].astype(str).to_numpy()
        if check_xy:
            saved_coords = metadata[['spatial_x', 'spatial_y']].to_numpy()
        if (len(embedding) != len(saved_names) or not np.array_equal(saved_names, own_names)
                or (check_xy and not np.allclose(saved_coords, own_coords))):
            raise ValueError(f'spa_mo_model {section}: embedding/metadata/data mismatch')
        arrays.append(embedding); sections.extend([section] * len(saved_names)); barcodes.extend(saved_names.tolist())
        sources.extend([embedding_path, metadata_path])
    return arrays, np.asarray(sections, dtype=str), np.asarray(barcodes, dtype=str), sources, snapshots


def aligned_obs_xy(data_dir, sections, barcodes, *, section_order, snapshots=None):
    coords = np.empty((len(sections), 2), dtype=float)
    for section in section_order:
        mask = sections == section
        path = data_dir / section / 'adata_RNA.h5ad'
        if snapshots is not None:
            source = snapshots[section]
            positions = _align_positions(source['names'], barcodes[mask], path, examples=True)
            coords[mask] = source['xy'][positions]
        else:
            rna = ad.read_h5ad(path, backed='r')
            try:
                positions = _align_positions(rna.obs_names.astype(str), barcodes[mask], path, examples=True)
                if not {'x', 'y'}.issubset(rna.obs.columns):
                    raise ValueError(f'{path}: missing canonical obs[x,y]')
                coords[mask] = rna.obs[['x', 'y']].to_numpy()[positions]
            finally:
                rna.file.close()
    return coords


@contextmanager
def open_aligned_rna_metadata(path, spot_ids, selection):
    """Borrow backed obs/obsm and row positions; caller owns interpretation.

    No expression matrix, factor transform, truth selection or coords copy is
    performed here. The borrowed AnnData is valid only within the with block.
    """
    rna = ad.read_h5ad(path, backed='r')
    try:
        source = pd.Index(rna.obs_names.astype(str))
        wanted = pd.Index(spot_ids[selection].astype(str))
        positions = source.get_indexer(wanted)
        if (positions < 0).any():
            missing = wanted[positions < 0][:5].tolist()
            raise ValueError(f'{path}: failed to align {int((positions < 0).sum())} barcodes; examples={missing}')
        yield rna, positions
    finally:
        rna.file.close()


def iter_saved_section_files(run, section_order, *, metadata_first=False):
    """One explicit saved NPY/CSV layout; preserve each caller's read priority."""
    for section in section_order:
        embedding_path = run / f'final_embeddings_{section}.npy'
        metadata_path = run / f'obs_metadata_{section}.csv'
        if metadata_first:
            metadata = pd.read_csv(metadata_path)
            embedding = np.load(embedding_path)
        else:
            embedding = np.load(embedding_path)
            metadata = pd.read_csv(metadata_path)
        yield section, embedding, metadata, embedding_path, metadata_path
