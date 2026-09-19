"""Read existing analysis inputs without training, sorting, scaling or clustering.

These are the established dataset readers with explicit run/data/section inputs.
Returned dictionaries retain the fields used by the existing MethodData records.
No biological truth is synthesized for datasets whose readers had none.
"""
from __future__ import annotations

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd


def clean_mousebrain_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
            continue
        text = str(value)
        if text.startswith("b'") and text.endswith("'"):
            text = text[2:-1]
        result[i] = text
    return result


def aligned_mousebrain_metadata(
    data_dir: Path, sections: np.ndarray, barcodes: np.ndarray,
    *, section_order, label_keys, section_dirs, group_label
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coords = np.empty((len(sections), 2), dtype=float)
    truth = {
        label: np.full(len(sections), np.nan, dtype=object)
        for label in label_keys
    }
    for section in section_order:
        mask = sections == section
        rna_path = data_dir / section_dirs[section] / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            pos = source.get_indexer(wanted)
            if (pos < 0).any():
                examples = wanted[pos < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: {int((pos < 0).sum())} barcodes failed "
                    f"alignment; examples={examples}"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[pos, :2]
            for label in label_keys:
                if label in rna.obs:
                    truth[label][mask] = clean_mousebrain_truth(
                        rna.obs[label].to_numpy()[pos]
                    )
        finally:
            rna.file.close()
    # MouseBrain source copies do not contain a literal `group` column.
    # The original comprehensive report defined group as the section/source
    # partition, which is exactly the aligned s1/s2/s3 vector used here.
    truth[group_label] = sections.astype(str).copy()
    return coords, truth


def clean_misar_truth(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if pd.isna(value):
            result[i] = np.nan
        else:
            text = str(value)
            result[i] = (
                text[2:-1]
                if text.startswith("b'") and text.endswith("'")
                else text
            )
    return result


def aligned_misar_metadata(
    data_dir: Path, sections: np.ndarray, barcodes: np.ndarray,
    *, section_order, label_keys
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coords = np.empty((len(sections), 2), dtype=float)
    truth = {
        label: np.full(len(sections), np.nan, dtype=object)
        for label in label_keys
    }
    for section in section_order:
        mask = sections == section
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            positions = source.get_indexer(wanted)
            if (positions < 0).any():
                examples = wanted[positions < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: {int((positions < 0).sum())} barcodes "
                    f"failed alignment; examples={examples}"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[positions, :2]
            for label in label_keys:
                if label not in rna.obs:
                    raise ValueError(f"{rna_path}: missing label {label}")
                truth[label][mask] = clean_misar_truth(
                    rna.obs[label].to_numpy()[positions]
                )
        finally:
            rna.file.close()
    return coords, truth


def aligned_crc_coords(
    data_dir: Path, sections: np.ndarray, barcodes: np.ndarray, *, section_order
) -> np.ndarray:
    coords = np.empty((len(sections), 2), dtype=float)
    for section in section_order:
        mask = sections == section
        rna_path = data_dir / section / "adata_RNA.h5ad"
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            source = pd.Index(rna.obs_names.astype(str))
            wanted = pd.Index(barcodes[mask].astype(str))
            positions = source.get_indexer(wanted)
            if (positions < 0).any():
                examples = wanted[positions < 0][:5].tolist()
                raise ValueError(
                    f"{rna_path}: {int((positions < 0).sum())} barcodes "
                    f"failed alignment; examples={examples}"
                )
            coords[mask] = np.asarray(rna.obsm["spatial"])[positions, :2]
        finally:
            rna.file.close()
    return coords


def load_mousebrain(run_dir: Path, data_dir: Path, *, section_order, section_dirs,
                    label_keys, group_label, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in section_order:
        embedding_path = (
            run_dir / "final_embeddings" / f"{section}_final_embedding.npy"
        )
        rna_path = data_dir / section_dirs[section] / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(
                f"{section}: embedding rows {len(embedding)} != {len(names)}"
            )
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.append(embedding_path)

    section_array = np.asarray(sections)
    barcode_array = np.asarray(barcodes)
    coordinates, truth = aligned_mousebrain_metadata(data_dir, section_array, barcode_array,
        section_order=section_order, section_dirs=section_dirs, label_keys=label_keys, group_label=group_label)
    return dict(
        name=name,
        embedding=np.vstack(arrays),
        sections=section_array,
        barcodes=barcode_array,
        coords=coordinates,
        truth=truth,
        data_dir=data_dir,
        source_paths=sources,
        output_root=output_root,
        shared_sources={},
    )


def load_paired_rna_adt(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in section_order:
        embedding_path = run / f"final_embeddings_{section}.npy"
        index_path = run / f"selected_spot_indices_{section}.npy"
        spatial_path = run / f"spatial_{section}.npy"
        rna_path = data_dir / section / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        indices = np.load(index_path).astype(int)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
        finally:
            rna.file.close()
        spatial = np.load(spatial_path)[:, :2]
        if len(embedding) != len(indices) or len(spatial) != len(indices):
            raise ValueError(f"{run.parent.name}/{section}: saved row counts do not align")
        arrays.append(embedding)
        sections.extend([section] * len(indices))
        barcodes.extend(names.tolist())
        coords.append(spatial)
        sources.extend([embedding_path, index_path, spatial_path, rna_path])
    return dict(name=name, embedding=np.vstack(arrays), sections=np.asarray(sections, dtype=str),
                barcodes=np.asarray(barcodes, dtype=str), coords=np.vstack(coords),
                data_dir=data_dir, source_paths=sources, output_root=output_root, shared_sources={})


def load_misar(run: Path, data_dir: Path, *, section_order, label_keys, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in section_order:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"MISAR-seq/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    coords, truth = aligned_misar_metadata(data_dir, section_array, barcode_array,
        section_order=section_order, label_keys=label_keys)
    return dict(name=name, embedding=np.vstack(arrays), sections=section_array,
                barcodes=barcode_array, coords=coords, truth=truth, data_dir=data_dir,
                source_paths=sources, output_root=output_root, shared_sources={})


def load_section_metadata(run: Path, *, section_order):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in section_order:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
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


def load_thymus(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    embedding, sections, barcodes, coords, sources = load_section_metadata(run, section_order=section_order)
    return dict(name=name, embedding=embedding, sections=sections, barcodes=barcodes,
                coords=coords, data_dir=data_dir, source_paths=sources,
                output_root=output_root, shared_sources={})


def load_simulation(run: Path, data_dir: Path, *, section_order, name: str, output_root: Path):
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    truth: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in section_order:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"Simulation/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        truth.extend(metadata["spatial_domain"].astype(str).tolist())
        coords.append(metadata[["spatial_x", "spatial_y"]].to_numpy(float))
        sources.extend([embedding_path, metadata_path])
    return dict(name=name, embedding=np.vstack(arrays), sections=np.asarray(sections, dtype=str),
                barcodes=np.asarray(barcodes, dtype=str), truth=np.asarray(truth, dtype=str),
                coords=np.vstack(coords), data_dir=data_dir, source_paths=sources,
                output_root=output_root, shared_sources={})


def load_crc(run: Path, data_dir: Path, *, section_order, short_names, output_root: Path):
    arrays: list[np.ndarray] = []
    section_values: list[str] = []
    barcode_values: list[str] = []
    source_paths: list[Path] = []
    for section in section_order:
        short = short_names[section]
        embedding_path = run / f"final_embeddings_{short}.npy"
        indices_path = run / f"selected_spot_indices_{short}.npy"
        spatial_path = run / f"spatial_{short}.npy"
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna = ad.read_h5ad(data_dir / section / "adata_RNA.h5ad", backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
            source_coords = np.asarray(rna.obsm["spatial"])[indices, :2]
        finally:
            rna.file.close()
        if (
            len(embedding) != len(indices)
            or len(np.unique(indices)) != len(indices)
            or not np.allclose(saved_coords, source_coords)
        ):
            raise ValueError(f"CRC {section}: embedding/index/spatial mismatch")
        arrays.append(embedding)
        section_values.extend([section] * len(indices))
        barcode_values.extend(names.tolist())
        source_paths.extend([embedding_path, indices_path, spatial_path])

    sections = np.asarray(section_values, dtype=str)
    barcodes = np.asarray(barcode_values, dtype=str)
    return dict(
        name="spa_mo_model",
        embedding=np.vstack(arrays),
        sections=sections,
        barcodes=barcodes,
        coords=aligned_crc_coords(data_dir, sections, barcodes, section_order=section_order),
        data_dir=data_dir,
        source_paths=source_paths,
        output_root=output_root,
        shared_sources={},
    )


def load_joint_assignments(analysis_dir: Path, k: int, section_values, barcodes):
    """Read saved joint labels in loader order; never recluster or reorder spots."""
    section_values = np.asarray(section_values).astype(str)
    barcodes = np.asarray(barcodes).astype(str)
    directory = Path(analysis_dir) / "clustering" / f"joint_k{k}"
    path = directory / "labels_all.csv"
    if path.is_file():
        frame = pd.read_csv(path, dtype={"section": str, "obs_name": str, "spot_id": str})
        if "section" in frame and not np.array_equal(frame["section"].to_numpy(), section_values):
            raise ValueError(f"Joint label section order differs: {path}")
        for key in ("obs_name", "spot_id"):
            if key in frame and not np.array_equal(frame[key].to_numpy(), barcodes):
                raise ValueError(f"Joint label spot order differs: {path}")
        labels = frame["cluster"].to_numpy()
        sources = [path]
    elif (directory / "labels_all.npy").is_file():
        path = directory / "labels_all.npy"
        labels = np.load(path)
        sources = [path]
    elif (Path(analysis_dir) / "all_k_labels_and_fit_cache.zip").is_file():
        import zipfile
        path = Path(analysis_dir) / "all_k_labels_and_fit_cache.zip"
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"joint_combined_k{k}.npy") as handle:
                labels = np.load(handle)
        sources = [path]
    else:
        labels = np.empty(len(section_values), dtype=np.int64)
        sources = []
        for section in dict.fromkeys(section_values):
            mask = section_values == section
            path = directory / f"labels_{section}.csv"
            frame = pd.read_csv(path, dtype={"section": str, "obs_name": str, "spot_id": str})
            for key in ("obs_name", "spot_id"):
                if key in frame and not np.array_equal(frame[key].to_numpy(), barcodes[mask]):
                    raise ValueError(f"Joint label spot order differs: {path}")
            if len(frame) != int(mask.sum()):
                raise ValueError(f"Joint label row count differs: {path}")
            labels[mask] = frame["cluster"].to_numpy()
            sources.append(path)
    if labels.shape != (len(section_values),):
        raise ValueError("Joint assignment shape differs from loaded spots")
    return labels, sources
