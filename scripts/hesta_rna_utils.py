#!/usr/bin/env python3
"""Memory-bounded HESTA RNA preprocessing helpers.

The HESTA h5ad files store raw counts as CSR under ``layers/counts``.  AnnData
backed mode currently materializes that layer, so this module reads the CSR
arrays with h5py and transforms bounded row blocks instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


DEFAULT_HESTA_FILES = [
    "CS12-13_E2S1_HESTA.h5ad",
    "CS14-15_E1S1_HESTA.h5ad",
    "CS17_E1S1_HESTA.h5ad",
    "CS18_E1S1_HESTA.h5ad",
    "CS19_E1S1_HESTA.h5ad",
    "CS20_E1S1_HESTA.h5ad",
    "CS23_E1S1_HESTA.h5ad",
]


def _decode(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_encoded_array(node: h5py.Dataset | h5py.Group, rows: np.ndarray | None = None) -> np.ndarray:
    """Read an AnnData array/string/categorical element without loading X."""

    if isinstance(node, h5py.Group):
        if node.attrs.get("encoding-type") != "categorical":
            raise TypeError(f"Unsupported encoded group: {node.name}")
        categories = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:] if rows is None else node["codes"][rows], dtype=np.int64)
        result = np.full(codes.shape, None, dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    values = node[:] if rows is None else node[rows]
    if values.dtype.kind in {"O", "S", "U"}:
        return _decode(values)
    return np.asarray(values)


def section_name(path: Path) -> str:
    suffix = "_HESTA"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def resolve_input_paths(data_dir: Path) -> list[Path]:
    paths = [data_dir / name for name in DEFAULT_HESTA_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing requested HESTA files: " + ", ".join(missing))
    return paths


def audit_hesta_files(paths: list[Path]) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            x_shape = tuple(int(value) for value in handle["X"].attrs["shape"])
            counts_shape = tuple(int(value) for value in handle["layers/counts"].attrs["shape"])
            if x_shape != counts_shape:
                raise ValueError(f"{path.name}: X and layers/counts shapes differ.")
            if handle["layers/counts"].attrs.get("encoding-type") != "csr_matrix":
                raise TypeError(f"{path.name}: layers/counts is not CSR.")
            if "spatial" not in handle["obsm"]:
                raise KeyError(f"{path.name}: missing obsm/spatial.")
            if handle["obsm/spatial"].shape != (x_shape[0], 2):
                raise ValueError(f"{path.name}: unexpected spatial shape.")
            obs_columns = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["obs"].attrs.get("column-order", [])
            ]
            sections[section_name(path)] = {
                "path": str(path),
                "file_size_bytes": int(path.stat().st_size),
                "shape": list(x_shape),
                "counts_layer": True,
                "spatial_shape": list(handle["obsm/spatial"].shape),
                "obs_columns": obs_columns,
                "celltype_present": "celltype" in handle["obs"],
                "stage_present": "stage" in handle["obs"],
            }
    return {
        "sections": sections,
        "section_order": list(sections),
        "total_spots": int(sum(item["shape"][0] for item in sections.values())),
        "modalities": ["RNA"],
        "rna_source": "layers/counts",
    }


def _read_csr_source_block(group: h5py.Group, start: int, end: int) -> sp.csr_matrix:
    shape = tuple(int(value) for value in group.attrs["shape"])
    indptr = np.asarray(group["indptr"][start : end + 1], dtype=np.int64)
    data_start, data_end = int(indptr[0]), int(indptr[-1])
    data = np.asarray(group["data"][data_start:data_end])
    indices = np.asarray(group["indices"][data_start:data_end], dtype=np.int32)
    indptr -= data_start
    return sp.csr_matrix((data, indices, indptr), shape=(end - start, shape[1]))


def read_csr_rows_columns(
    group: h5py.Group,
    rows: np.ndarray,
    columns: np.ndarray,
    source_chunk_rows: int,
) -> sp.csr_matrix:
    """Read sorted arbitrary rows while bounding each source CSR read."""

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if rows.size == 0:
        return sp.csr_matrix((0, columns.size), dtype=np.float32)
    if np.any(rows[1:] < rows[:-1]):
        raise ValueError("rows must be sorted")
    chunks: list[sp.csr_matrix] = []
    buckets = rows // int(source_chunk_rows)
    for bucket in np.unique(buckets):
        mask = buckets == bucket
        chosen = rows[mask]
        source_start = int(bucket) * int(source_chunk_rows)
        source_end = min(source_start + int(source_chunk_rows), int(group.attrs["shape"][0]))
        block = _read_csr_source_block(group, source_start, source_end)
        block = block[chosen - source_start, :][:, columns]
        chunks.append(block.astype(np.float32, copy=False))
    return sp.vstack(chunks, format="csr")


def normalize_log1p(counts: sp.csr_matrix, total_counts: np.ndarray, target_sum: float) -> sp.csr_matrix:
    counts = counts.astype(np.float32, copy=True)
    factors = np.divide(
        float(target_sum),
        np.asarray(total_counts, dtype=np.float32),
        out=np.zeros(len(total_counts), dtype=np.float32),
        where=np.asarray(total_counts) > 0,
    )
    counts.data *= np.repeat(factors, np.diff(counts.indptr))
    np.log1p(counts.data, out=counts.data)
    return counts


def stratified_block_sample(
    sorted_rows: np.ndarray,
    size: int,
    rng: np.random.Generator,
    max_blocks: int = 16,
) -> np.ndarray:
    """Sample across a section using a few contiguous blocks for CSR I/O locality."""

    sorted_rows = np.asarray(sorted_rows, dtype=np.int64)
    size = min(int(size), sorted_rows.size)
    if size >= sorted_rows.size:
        return sorted_rows.copy()
    n_blocks = min(int(max_blocks), max(1, size // 100))
    block_size = int(np.ceil(size / n_blocks))
    max_start = max(0, sorted_rows.size - block_size)
    anchors = np.linspace(0, max_start, n_blocks, dtype=np.int64)
    stride = max(1, sorted_rows.size // n_blocks)
    starts = []
    for anchor in anchors:
        jitter = int(rng.integers(-max(1, stride // 4), max(2, stride // 4 + 1)))
        starts.append(min(max(int(anchor) + jitter, 0), max_start))
    positions = np.unique(
        np.concatenate(
            [np.arange(start, min(start + block_size, sorted_rows.size)) for start in starts]
        )
    )
    if positions.size < size:
        unused = np.setdiff1d(np.arange(sorted_rows.size), positions, assume_unique=True)
        positions = np.concatenate(
            [positions, rng.choice(unused, size=size - positions.size, replace=False)]
        )
    return np.sort(sorted_rows[np.sort(positions[:size])])


def selected_rows_from_qc(
    handle: h5py.File,
    min_counts: float,
    min_genes: int,
    max_pct_mt: float,
    max_spots: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_counts = np.asarray(handle["obs/total_counts"][:], dtype=np.float64)
    n_genes = np.asarray(handle["obs/n_genes_by_counts"][:], dtype=np.int64)
    pct_mt = np.asarray(handle["obs/pct_counts_mt"][:], dtype=np.float64)
    keep = (
        np.isfinite(total_counts)
        & np.isfinite(pct_mt)
        & (total_counts >= float(min_counts))
        & (n_genes >= int(min_genes))
        & (pct_mt <= float(max_pct_mt))
    )
    eligible = np.flatnonzero(keep)
    if max_spots is not None and max_spots > 0 and eligible.size > max_spots:
        selected = stratified_block_sample(eligible, int(max_spots), rng)
    else:
        selected = eligible
    return selected.astype(np.int64), {
        "original_spots": int(total_counts.size),
        "qc_pass_spots": int(eligible.size),
        "selected_spots": int(selected.size),
        "filtered_spots": int(total_counts.size - eligible.size),
        "total_counts_selected_median": float(np.median(total_counts[selected])) if selected.size else None,
        "n_genes_selected_median": float(np.median(n_genes[selected])) if selected.size else None,
    }


def _common_gene_index(paths: list[Path]) -> tuple[list[str], dict[str, np.ndarray]]:
    genes_by_section: dict[str, np.ndarray] = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            genes = _decode(handle["var/_index"][:])
        if len(set(genes.tolist())) != len(genes):
            raise ValueError(f"{path.name}: var_names are not unique.")
        genes_by_section[section_name(path)] = genes
    common = set(genes_by_section[section_name(paths[0])].tolist())
    for genes in genes_by_section.values():
        common.intersection_update(genes.tolist())
    first_genes = genes_by_section[section_name(paths[0])]
    ordered = [gene for gene in first_genes if gene in common]
    indices = {}
    for section, genes in genes_by_section.items():
        lookup = {gene: index for index, gene in enumerate(genes)}
        indices[section] = np.asarray([lookup[gene] for gene in ordered], dtype=np.int64)
    return ordered, indices


def _write_annotations(handle: h5py.File, rows: np.ndarray, section: str, path: Path) -> None:
    columns: dict[str, Any] = {
        "section": np.repeat(section, rows.size),
        "original_row": rows,
        "obs_name": read_encoded_array(handle["obs/_index"], rows),
    }
    for name in (
        "cellid", "celltype", "stage", "total_counts", "n_genes_by_counts", "pct_counts_mt", "x", "y"
    ):
        if name in handle["obs"]:
            columns[name] = read_encoded_array(handle[f"obs/{name}"], rows)
    pd.DataFrame(columns).to_csv(path, index=False, compression="gzip")


def preprocess_hesta_rna(
    paths: list[Path],
    cache_dir: Path,
    *,
    hvg_num: int,
    hvg_sample_per_section: int,
    svd_fit_per_section: int,
    n_comps: int,
    target_sum: float,
    min_counts: float,
    min_genes: int,
    max_pct_mt: float,
    max_spots_per_section: int | None,
    source_chunk_rows: int,
    seed: int,
) -> dict[str, Any]:
    """Build cached 50-D RNA features without materializing full count matrices."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    section_order = [section_name(path) for path in paths]
    selected: dict[str, np.ndarray] = {}
    qc: dict[str, Any] = {}
    for path in paths:
        section = section_name(path)
        with h5py.File(path, "r") as handle:
            selected[section], qc[section] = selected_rows_from_qc(
                handle, min_counts, min_genes, max_pct_mt, max_spots_per_section, rng
            )
        if selected[section].size < max(n_comps + 1, 10):
            raise ValueError(f"{section}: too few spots remain after QC.")

    common_genes, common_indices = _common_gene_index(paths)
    sample_counts: list[sp.csr_matrix] = []
    sample_batches: list[str] = []
    sample_rows_by_section: dict[str, np.ndarray] = {}
    for path in paths:
        section = section_name(path)
        rows = selected[section]
        sample_size = min(int(hvg_sample_per_section), rows.size)
        sample_rows = stratified_block_sample(rows, sample_size, rng)
        sample_rows_by_section[section] = sample_rows
        with h5py.File(path, "r") as handle:
            matrix = read_csr_rows_columns(
                handle["layers/counts"], sample_rows, common_indices[section], source_chunk_rows
            )
        sample_counts.append(matrix)
        sample_batches.extend([section] * sample_size)
    hvg_adata = ad.AnnData(
        X=sp.vstack(sample_counts, format="csr"),
        obs=pd.DataFrame({"batch": pd.Categorical(sample_batches)}),
        var=pd.DataFrame(index=pd.Index(common_genes)),
    )
    requested_hvg_count = min(int(hvg_num), hvg_adata.n_vars)
    hvg_method = "seurat_v3_batch_balanced"
    try:
        sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=requested_hvg_count,
            flavor="seurat_v3",
            layer=None,
            batch_key="batch",
        )
    except ValueError:
        # Very small smoke-test subsets can make LOESS singular.  Use a
        # deterministic count-dispersion ranking only in that degenerate case.
        means = np.asarray(hvg_adata.X.mean(axis=0)).ravel()
        second = np.asarray(hvg_adata.X.power(2).mean(axis=0)).ravel()
        dispersion = np.maximum(second - means**2, 0.0) / np.maximum(means, 1e-8)
        selected_columns = np.argsort(dispersion, kind="stable")[-requested_hvg_count:]
        hvg_adata.var["highly_variable"] = False
        hvg_adata.var.iloc[selected_columns, hvg_adata.var.columns.get_loc("highly_variable")] = True
        hvg_method = "count_dispersion_fallback_after_singular_seurat_v3"
    hvg_genes = hvg_adata.var_names[hvg_adata.var["highly_variable"]].astype(str).tolist()
    if len(hvg_genes) <= n_comps:
        raise ValueError("The selected HVG count must exceed n_comps.")
    del hvg_adata, sample_counts

    hvg_columns: dict[str, np.ndarray] = {}
    for path in paths:
        section = section_name(path)
        with h5py.File(path, "r") as handle:
            genes = _decode(handle["var/_index"][:])
        lookup = {gene: index for index, gene in enumerate(genes)}
        hvg_columns[section] = np.asarray([lookup[gene] for gene in hvg_genes], dtype=np.int64)

    fit_matrices: list[sp.csr_matrix] = []
    for path in paths:
        section = section_name(path)
        rows = selected[section]
        fit_size = min(int(svd_fit_per_section), rows.size)
        fit_rows = stratified_block_sample(rows, fit_size, rng)
        with h5py.File(path, "r") as handle:
            matrix = read_csr_rows_columns(
                handle["layers/counts"], fit_rows, hvg_columns[section], source_chunk_rows
            )
            totals = np.asarray(handle["obs/total_counts"][fit_rows], dtype=np.float32)
        fit_matrices.append(normalize_log1p(matrix, totals, target_sum))
    fit_matrix = sp.vstack(fit_matrices, format="csr")
    svd = TruncatedSVD(n_components=int(n_comps), n_iter=7, random_state=int(seed))
    fit_scores = svd.fit_transform(fit_matrix).astype(np.float32)
    score_mean = fit_scores.mean(axis=0, dtype=np.float64).astype(np.float32)
    score_std = fit_scores.std(axis=0, dtype=np.float64).astype(np.float32)
    score_std[score_std < 1e-6] = 1.0
    del fit_matrix, fit_matrices, fit_scores

    feature_files: dict[str, str] = {}
    spatial_files: dict[str, str] = {}
    index_files: dict[str, str] = {}
    annotation_files: dict[str, str] = {}
    for path in paths:
        section = section_name(path)
        rows = selected[section]
        feature_path = cache_dir / f"rna_features_{section}.npy"
        spatial_path = cache_dir / f"spatial_{section}.npy"
        index_path = cache_dir / f"selected_rows_{section}.npy"
        annotation_path = cache_dir / f"annotations_{section}.csv.gz"
        output = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(rows.size, int(n_comps))
        )
        write_offset = 0
        with h5py.File(path, "r") as handle:
            buckets = rows // int(source_chunk_rows)
            for bucket in np.unique(buckets):
                batch_rows = rows[buckets == bucket]
                matrix = read_csr_rows_columns(
                    handle["layers/counts"], batch_rows, hvg_columns[section], source_chunk_rows
                )
                totals = np.asarray(handle["obs/total_counts"][batch_rows], dtype=np.float32)
                scores = svd.transform(normalize_log1p(matrix, totals, target_sum)).astype(np.float32)
                scores = (scores - score_mean) / score_std
                output[write_offset : write_offset + batch_rows.size] = scores
                write_offset += batch_rows.size
            np.save(spatial_path, np.asarray(handle["obsm/spatial"][rows], dtype=np.float32))
            _write_annotations(handle, rows, section, annotation_path)
        output.flush()
        del output
        np.save(index_path, rows)
        feature_files[section] = str(feature_path)
        spatial_files[section] = str(spatial_path)
        index_files[section] = str(index_path)
        annotation_files[section] = str(annotation_path)

    (cache_dir / "hvg_genes.txt").write_text("\n".join(hvg_genes) + "\n", encoding="utf-8")
    np.savez_compressed(
        cache_dir / "svd_model.npz",
        components=svd.components_.astype(np.float32),
        explained_variance=svd.explained_variance_.astype(np.float32),
        explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
        score_mean=score_mean,
        score_std=score_std,
    )
    manifest = {
        "version": 1,
        "section_order": section_order,
        "feature_files": feature_files,
        "spatial_files": spatial_files,
        "selected_index_files": index_files,
        "annotation_files": annotation_files,
        "hvg_genes_file": str(cache_dir / "hvg_genes.txt"),
        "svd_model_file": str(cache_dir / "svd_model.npz"),
        "feature_method": "balanced-HVG raw counts -> library-size normalize -> log1p -> balanced-fit TruncatedSVD -> component z-score",
        "rna_source": "layers/counts",
        "harmony_used": False,
        "common_gene_count": len(common_genes),
        "hvg_count": len(hvg_genes),
        "hvg_method": hvg_method,
        "n_comps": int(n_comps),
        "target_sum": float(target_sum),
        "qc_thresholds": {
            "min_counts": float(min_counts),
            "min_genes": int(min_genes),
            "max_pct_mt": float(max_pct_mt),
        },
        "max_spots_per_section": max_spots_per_section,
        "hvg_sample_per_section": int(hvg_sample_per_section),
        "svd_fit_per_section": int(svd_fit_per_section),
        "seed": int(seed),
        "qc": qc,
    }
    (cache_dir / "preprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_preprocessed_manifest(cache_dir: Path) -> tuple[dict[str, Any], dict, dict]:
    manifest_path = cache_dir / "preprocess_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Preprocessing cache is missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dict = {
        # Copy-on-write mmap is writable from PyTorch's perspective without
        # permitting model code to modify the on-disk cache.
        section: {"RNA": np.load(manifest["feature_files"][section], mmap_mode="c")}
        for section in manifest["section_order"]
    }
    spatial_dict = {
        section: np.load(manifest["spatial_files"][section], mmap_mode="r")
        for section in manifest["section_order"]
    }
    return manifest, feature_dict, spatial_dict
