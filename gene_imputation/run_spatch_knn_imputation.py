#!/usr/bin/env python3
"""Memory-bounded, resumable COSIE-style KNN gene imputation for spatch."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from annoy import AnnoyIndex


LOGGER = logging.getLogger("gene_imputation")
FORMAT_NAME = "spa_mo_model.knn_gene_imputation"
FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("audit", "neighbors", "impute", "validate", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--limit-target-spots",
        type=int,
        default=None,
        help="Testing only. Use a separate --run-dir when setting this option.",
    )
    parser.add_argument(
        "--limit-genes",
        type=int,
        default=None,
        help="Testing only. Use a separate --run-dir when setting this option.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, full_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if full_sha256:
        result["sha256"] = file_sha256(path)
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def memory_gib() -> dict[str, float]:
    current = float("nan")
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                current = float(line.split()[1]) / 1024**2
                break
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    return {"rss_gib": current, "peak_rss_gib": float(peak)}


def decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[:]
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values],
        dtype=object,
    )


def h5ad_shape(handle: h5py.File) -> tuple[int, int]:
    x = handle["X"]
    if isinstance(x, h5py.Group):
        return tuple(int(v) for v in x.attrs["shape"])
    return tuple(int(v) for v in x.shape)


def h5ad_var_names(handle: h5py.File) -> np.ndarray:
    frame = handle["var"]
    index_key = frame.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    return decode_strings(frame[str(index_key)])


def load_h5ad_csr(path: Path) -> sp.csr_matrix:
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if not isinstance(x, h5py.Group) or x.attrs.get("encoding-type") != "csr_matrix":
            raise TypeError(f"Expected CSR-backed X in {path}.")
        shape = tuple(int(v) for v in x.attrs["shape"])
        data = np.asarray(x["data"][:], dtype=np.float32)
        indices = np.asarray(x["indices"][:], dtype=np.int32)
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def unique_rows(values: np.ndarray) -> bool:
    structured = np.ascontiguousarray(values).view(
        np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
    )
    return np.unique(structured).size == len(values)


def load_config(path: Path, run_dir_override: Path | None) -> tuple[dict[str, Any], Path, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Only configuration schema_version=1 is supported.")
    run_dir = (run_dir_override or Path(config["run_dir"])).resolve()
    effective = json.loads(json.dumps(config))
    effective["run_dir"] = str(run_dir)
    config_hash = sha256_text(canonical_json(effective))
    return effective, run_dir, config_hash


def audit_inputs(
    config: dict[str, Any],
    run_dir: Path,
    config_hash: str,
    limit_target_spots: int | None,
    limit_genes: int | None,
) -> dict[str, Any]:
    required = {
        key: Path(config[key]).resolve()
        for key in (
            "model_run_summary",
            "spot_metadata",
            "target_embedding",
            "source_embedding",
            "target_rna_h5ad",
            "source_rna_h5ad",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    target_embedding = np.load(required["target_embedding"], mmap_mode="r")
    source_embedding = np.load(required["source_embedding"], mmap_mode="r")
    if target_embedding.ndim != 2 or source_embedding.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional arrays.")
    if target_embedding.shape[1] != source_embedding.shape[1]:
        raise ValueError("Target and source embedding dimensions differ.")
    if target_embedding.dtype != np.float32 or source_embedding.dtype != np.float32:
        raise TypeError("This workflow expects float32 model embeddings.")

    with h5py.File(required["target_rna_h5ad"], "r") as target_handle, h5py.File(
        required["source_rna_h5ad"], "r"
    ) as source_handle:
        target_shape = h5ad_shape(target_handle)
        source_shape = h5ad_shape(source_handle)
        target_genes_all = h5ad_var_names(target_handle)
        source_genes_all = h5ad_var_names(source_handle)
        target_spatial = np.asarray(target_handle["obsm"]["spatial"][:])
        source_spatial = np.asarray(source_handle["obsm"]["spatial"][:])

    if target_shape[0] != target_embedding.shape[0]:
        raise ValueError(f"Target RNA/embedding row mismatch: {target_shape[0]} vs {target_embedding.shape[0]}")
    if source_shape[0] != source_embedding.shape[0]:
        raise ValueError(f"Source RNA/embedding row mismatch: {source_shape[0]} vs {source_embedding.shape[0]}")
    if len(set(target_genes_all.tolist())) != len(target_genes_all):
        raise ValueError("Target RNA var_names are not unique.")
    if len(set(source_genes_all.tolist())) != len(source_genes_all):
        raise ValueError("Source RNA var_names are not unique.")
    if not unique_rows(target_spatial) or not unique_rows(source_spatial):
        raise ValueError("Spatial coordinates must be unique within each section.")

    target_gene_set = set(target_genes_all.tolist())
    source_only_mask = np.asarray([gene not in target_gene_set for gene in source_genes_all], dtype=bool)
    source_only_indices = np.flatnonzero(source_only_mask).astype(np.int32)
    source_only_genes = source_genes_all[source_only_indices]
    if config["target_gene_rule"] != "source_only":
        raise ValueError("Only target_gene_rule='source_only' is currently supported.")
    if limit_genes is not None:
        if limit_genes <= 0:
            raise ValueError("--limit-genes must be positive.")
        source_only_indices = source_only_indices[:limit_genes]
        source_only_genes = source_only_genes[:limit_genes]

    n_target = target_embedding.shape[0]
    if limit_target_spots is not None:
        if limit_target_spots <= 0:
            raise ValueError("--limit-target-spots must be positive.")
        n_target = min(n_target, limit_target_spots)

    summary = json.loads(required["model_run_summary"].read_text(encoding="utf-8"))
    alignment = summary.get("alignment", {})
    for section, expected in (
        (config["target_section"], target_shape[0]),
        (config["source_section"], source_shape[0]),
    ):
        if section in alignment and int(alignment[section].get("n_spots", -1)) != expected:
            raise ValueError(f"run_summary alignment mismatch for {section}.")

    # Verify the metadata ordering against the raw spatial rows without loading
    # the full CSV into a large pandas object.
    expected_spatial = {
        config["target_section"]: target_spatial,
        config["source_section"]: source_spatial,
    }
    offsets = {section: 0 for section in expected_spatial}
    for chunk in pd.read_csv(
        required["spot_metadata"], usecols=["section", "x", "y"], chunksize=100_000
    ):
        for section, coordinates in expected_spatial.items():
            selected = chunk.loc[chunk["section"] == section, ["x", "y"]].to_numpy()
            if not len(selected):
                continue
            start = offsets[section]
            end = start + len(selected)
            if end > len(coordinates) or not np.array_equal(selected, coordinates[start:end]):
                raise ValueError(f"spot_metadata order does not match raw coordinates for {section}.")
            offsets[section] = end
    if any(offsets[section] != len(values) for section, values in expected_spatial.items()):
        raise ValueError(f"spot_metadata section counts are incomplete: {offsets}")

    full_hash = bool(config.get("fingerprints", {}).get("full_sha256", True))
    records = {key: file_record(path, full_hash) for key, path in required.items()}
    audit = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at_unix": time.time(),
        "config_hash": config_hash,
        "run_name": config["run_name"],
        "target_section": config["target_section"],
        "source_section": config["source_section"],
        "target_embedding_shape": list(target_embedding.shape),
        "source_embedding_shape": list(source_embedding.shape),
        "effective_target_spots": int(n_target),
        "embedding_dim": int(target_embedding.shape[1]),
        "target_rna_shape": list(target_shape),
        "source_rna_shape": list(source_shape),
        "target_gene_count": int(len(target_genes_all)),
        "source_gene_count": int(len(source_genes_all)),
        "shared_gene_count": int(source_shape[1] - int(source_only_mask.sum())),
        "source_only_gene_count_full": int(source_only_mask.sum()),
        "effective_imputation_gene_count": int(len(source_only_genes)),
        "target_spatial_unique": True,
        "source_spatial_unique": True,
        "metadata_spatial_order_match": True,
        "limited_run": bool(limit_target_spots is not None or limit_genes is not None),
        "inputs": records,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
    }
    atomic_write_json(run_dir / "input_audit.json", audit)
    atomic_write_json(run_dir / "effective_config.json", config)
    np.save(run_dir / "target_source_var_indices.npy", source_only_indices)
    with (run_dir / "target_genes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_gene_index", "gene", "source_var_index"])
        for index, (gene, source_index) in enumerate(zip(source_only_genes, source_only_indices)):
            writer.writerow([index, gene, int(source_index)])
    LOGGER.info("Input audit passed: target spots=%d, source spots=%d, genes=%d", n_target, source_shape[0], len(source_only_genes))
    return audit


def load_or_require_audit(run_dir: Path, config_hash: str) -> dict[str, Any]:
    path = run_dir / "input_audit.json"
    if not path.is_file():
        raise FileNotFoundError("input_audit.json is missing; run --stage audit first.")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("config_hash") != config_hash:
        raise ValueError("Existing audit was produced by a different effective configuration.")
    return audit


def annoy_metadata(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    knn = config["knn"]
    return {
        "backend": "annoy",
        "metric": knn["metric"],
        "dimension": int(audit["embedding_dim"]),
        "source_count": int(audit["source_embedding_shape"][0]),
        "source_embedding_sha256": audit["inputs"]["source_embedding"].get("sha256"),
        "n_trees": int(knn["n_trees"]),
        "seed": int(knn["seed"]),
    }


def build_or_load_annoy(config: dict[str, Any], run_dir: Path, audit: dict[str, Any]) -> AnnoyIndex:
    knn = config["knn"]
    if knn.get("backend") != "annoy":
        raise ValueError("Only the COSIE-compatible Annoy backend is supported.")
    expected = annoy_metadata(config, audit)
    index_path = run_dir / "source_embedding.ann"
    metadata_path = run_dir / "source_embedding.ann.json"
    index = AnnoyIndex(expected["dimension"], expected["metric"])
    if index_path.is_file() and metadata_path.is_file():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("Existing Annoy index metadata is incompatible with this run.")
        if not index.load(str(index_path)):
            raise RuntimeError("Annoy failed to load the saved source index.")
        LOGGER.info("Reused Annoy index: %s", index_path)
        return index

    source = np.load(config["source_embedding"], mmap_mode="r")
    index.set_seed(int(knn["seed"]))
    started = time.time()
    for row in range(source.shape[0]):
        index.add_item(row, np.asarray(source[row], dtype=np.float32))
        if row and row % 100_000 == 0:
            LOGGER.info("Annoy index add progress: %d/%d", row, source.shape[0])
    try:
        index.build(int(knn["n_trees"]), n_jobs=-1)
    except TypeError:
        index.build(int(knn["n_trees"]))
    if not index.save(str(index_path)):
        raise RuntimeError("Annoy failed to save the source index.")
    atomic_write_json(metadata_path, expected)
    LOGGER.info("Built Annoy index in %.1f seconds (%s)", time.time() - started, memory_gib())
    return index


def compute_neighbors(config: dict[str, Any], run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    knn = config["knn"]
    k = int(knn["k"])
    source_count = int(audit["source_embedding_shape"][0])
    target_count = int(audit["effective_target_spots"])
    if not 1 <= k <= source_count:
        raise ValueError(f"Invalid K={k} for {source_count} source spots.")
    query_chunk_size = int(knn["query_chunk_size"])
    n_chunks = math.ceil(target_count / query_chunk_size)
    neighbors_path = run_dir / f"neighbors_{config['target_section']}_to_{config['source_section']}_k{k}.npy"
    distances_path = run_dir / f"neighbor_distances_{config['target_section']}_to_{config['source_section']}_k{k}.npy"
    completion_path = run_dir / "neighbor_completed_chunks.npy"

    if neighbors_path.exists():
        neighbors = np.lib.format.open_memmap(neighbors_path, mode="r+")
        distances = np.lib.format.open_memmap(distances_path, mode="r+")
        completed = np.lib.format.open_memmap(completion_path, mode="r+")
        if neighbors.shape != (target_count, k) or distances.shape != (target_count, k) or completed.shape != (n_chunks,):
            raise ValueError("Existing neighbor-cache shapes are incompatible.")
    else:
        if distances_path.exists() or completion_path.exists():
            raise ValueError("Partial neighbor-cache files are inconsistent.")
        neighbors = np.lib.format.open_memmap(neighbors_path, mode="w+", dtype=np.int32, shape=(target_count, k))
        distances = np.lib.format.open_memmap(distances_path, mode="w+", dtype=np.float32, shape=(target_count, k))
        completed = np.lib.format.open_memmap(completion_path, mode="w+", dtype=np.bool_, shape=(n_chunks,))
        completed[:] = False
        completed.flush()

    if bool(np.all(completed)):
        LOGGER.info("Neighbor cache is already complete.")
    else:
        index = build_or_load_annoy(config, run_dir, audit)
        target = np.load(config["target_embedding"], mmap_mode="r")
        workers = max(1, int(knn.get("query_workers", 1)))
        search_k = int(knn.get("search_k", -1))
        started = time.time()

        def query(row: int) -> tuple[list[int], list[float]]:
            return index.get_nns_by_vector(
                np.asarray(target[row], dtype=np.float32),
                k,
                search_k=search_k,
                include_distances=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_index in range(n_chunks):
                if bool(completed[chunk_index]):
                    continue
                start = chunk_index * query_chunk_size
                end = min(target_count, start + query_chunk_size)
                results = list(executor.map(query, range(start, end), chunksize=32))
                neighbors[start:end] = np.asarray([value[0] for value in results], dtype=np.int32)
                distances[start:end] = np.asarray([value[1] for value in results], dtype=np.float32)
                if np.any(neighbors[start:end] < 0) or np.any(neighbors[start:end] >= source_count):
                    raise RuntimeError(f"Annoy returned an invalid neighbor in chunk {chunk_index}.")
                neighbors.flush()
                distances.flush()
                completed[chunk_index] = True
                completed.flush()
                elapsed = time.time() - started
                done = int(completed.sum())
                LOGGER.info(
                    "Neighbor chunks: %d/%d; rows=%d/%d; elapsed=%.1fs; %s",
                    done,
                    n_chunks,
                    end,
                    target_count,
                    elapsed,
                    memory_gib(),
                )

    metadata = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "algorithm": "COSIE-compatible Annoy KNN",
        "target_section": config["target_section"],
        "source_section": config["source_section"],
        "target_count": target_count,
        "source_count": source_count,
        "embedding_dim": int(audit["embedding_dim"]),
        "k": k,
        "metric": knn["metric"],
        "n_trees": int(knn["n_trees"]),
        "search_k": int(knn.get("search_k", -1)),
        "seed": int(knn["seed"]),
        "neighbors_path": str(neighbors_path),
        "distances_path": str(distances_path),
        "complete": bool(np.all(completed)),
    }
    atomic_write_json(run_dir / "neighbor_manifest.json", metadata)
    return metadata


def make_uniform_transfer_matrix(neighbors: np.ndarray, source_count: int) -> sp.csr_matrix:
    if neighbors.ndim != 2:
        raise ValueError("neighbors must be a two-dimensional integer matrix.")
    target_count, k = neighbors.shape
    if k <= 0:
        raise ValueError("neighbors must contain at least one column.")
    indices = np.asarray(neighbors, dtype=np.int32).reshape(-1)
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= source_count):
        raise ValueError("Neighbor index is outside the source range.")
    indptr = np.arange(0, target_count * k + 1, k, dtype=np.int64)
    weights = np.full(target_count * k, 1.0 / float(k), dtype=np.float32)
    return sp.csr_matrix((weights, indices, indptr), shape=(target_count, source_count))


def aggregate_block(transfer: sp.csr_matrix, source_block: sp.spmatrix) -> np.ndarray:
    result = transfer @ source_block
    if sp.issparse(result):
        result = result.toarray()
    return np.asarray(result, dtype=np.float32)


def hdf5_compression_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    compression = settings.get("compression")
    if compression in (None, "none", "None"):
        return {}
    result: dict[str, Any] = {
        "compression": compression,
        "shuffle": bool(settings.get("shuffle", True)),
    }
    if compression == "gzip":
        result["compression_opts"] = int(settings.get("compression_level", 1))
    return result


def initialize_output(
    path: Path,
    config: dict[str, Any],
    audit: dict[str, Any],
    target_genes: np.ndarray,
    target_spatial: np.ndarray,
    source_nnz: np.ndarray,
    source_total: np.ndarray,
) -> None:
    settings = config["imputation"]
    target_count = int(audit["effective_target_spots"])
    gene_count = len(target_genes)
    gene_block_size = int(settings["gene_block_size"])
    n_blocks = math.ceil(gene_count / gene_block_size)
    spot_chunk = min(target_count, int(settings["hdf5_spot_chunk"]))
    gene_chunk = min(gene_count, gene_block_size)
    strings = h5py.string_dtype(encoding="utf-8")
    compression = hdf5_compression_kwargs(settings)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "config_hash": audit["config_hash"],
                "run_name": config["run_name"],
                "algorithm": "uniform mean over cross-section embedding KNN",
                "target_section": config["target_section"],
                "source_section": config["source_section"],
                "k": int(config["knn"]["k"]),
                "metric": config["knn"]["metric"],
                "expression_scale": "source RNA X (raw integer counts in this dataset)",
                "complete": False,
                "created_at_unix": time.time(),
            }
        )
        handle.create_dataset(
            "X",
            shape=(target_count, gene_count),
            dtype=np.dtype(settings["output_dtype"]),
            chunks=(spot_chunk, gene_chunk),
            fillvalue=0.0,
            **compression,
        )
        handle.create_dataset("var_names", data=np.asarray(target_genes, dtype=object), dtype=strings)
        obs_names = np.asarray(
            [f"{config['target_section']}:{int(x)}:{int(y)}" for x, y in target_spatial[:target_count]],
            dtype=object,
        )
        handle.create_dataset("obs_names", data=obs_names, dtype=strings)
        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=target_spatial[:target_count])
        stats = handle.create_group("gene_stats")
        stats.create_dataset("source_nnz", data=np.asarray(source_nnz, dtype=np.int64))
        stats.create_dataset("source_total", data=np.asarray(source_total, dtype=np.float64))
        stats.create_dataset("predicted_mean", shape=(gene_count,), dtype=np.float64, fillvalue=np.nan)
        stats.create_dataset("predicted_nonzero_fraction", shape=(gene_count,), dtype=np.float64, fillvalue=np.nan)
        stats.create_dataset("predicted_max", shape=(gene_count,), dtype=np.float32, fillvalue=np.nan)
        handle.create_dataset("completed_gene_blocks", shape=(n_blocks,), dtype=np.bool_, fillvalue=False)
        handle.flush()


def impute(config: dict[str, Any], run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    neighbor_manifest_path = run_dir / "neighbor_manifest.json"
    if not neighbor_manifest_path.is_file():
        raise FileNotFoundError("neighbor_manifest.json is missing; run --stage neighbors first.")
    neighbor_manifest = json.loads(neighbor_manifest_path.read_text(encoding="utf-8"))
    if neighbor_manifest.get("config_hash") != audit["config_hash"] or not neighbor_manifest.get("complete"):
        raise ValueError("Neighbor cache is incomplete or belongs to a different configuration.")

    neighbors = np.load(neighbor_manifest["neighbors_path"], mmap_mode="r")
    source_count = int(audit["source_embedding_shape"][0])
    LOGGER.info("Building sparse uniform transfer matrix W from %s", neighbors.shape)
    transfer = make_uniform_transfer_matrix(neighbors, source_count)
    LOGGER.info("Transfer matrix: shape=%s nnz=%d %s", transfer.shape, transfer.nnz, memory_gib())

    source_indices = np.load(run_dir / "target_source_var_indices.npy")
    with (run_dir / "target_genes.csv").open("r", encoding="utf-8", newline="") as handle:
        target_genes = np.asarray([row["gene"] for row in csv.DictReader(handle)], dtype=object)
    if len(target_genes) != len(source_indices):
        raise ValueError("target_genes.csv and target_source_var_indices.npy differ in length.")

    LOGGER.info("Loading source CSR matrix once; this remains sparse in memory.")
    full_source = load_h5ad_csr(Path(config["source_rna_h5ad"]))
    source_target = full_source[:, source_indices].tocsr().astype(np.float32, copy=False)
    del full_source
    source_target.sort_indices()
    source_nnz = np.asarray(source_target.getnnz(axis=0)).reshape(-1).astype(np.int64)
    source_total = np.asarray(source_target.sum(axis=0)).reshape(-1).astype(np.float64)
    LOGGER.info("Source target matrix: shape=%s nnz=%d %s", source_target.shape, source_target.nnz, memory_gib())

    with h5py.File(config["target_rna_h5ad"], "r") as target_handle:
        target_spatial = np.asarray(target_handle["obsm"]["spatial"][:])

    output_path = run_dir / f"imputed_{config['target_section']}_{config['source_section']}_only_genes.h5"
    if not output_path.exists():
        initialize_output(
            output_path,
            config,
            audit,
            target_genes,
            target_spatial,
            source_nnz,
            source_total,
        )

    settings = config["imputation"]
    gene_block_size = int(settings["gene_block_size"])
    started = time.time()
    with h5py.File(output_path, "r+") as handle:
        if handle.attrs.get("config_hash") != audit["config_hash"]:
            raise ValueError("Existing output HDF5 belongs to another configuration.")
        if handle["X"].shape != (neighbors.shape[0], len(target_genes)):
            raise ValueError("Existing output HDF5 has an incompatible X shape.")
        completed = handle["completed_gene_blocks"]
        for block_index in range(len(completed)):
            if bool(completed[block_index]):
                continue
            start = block_index * gene_block_size
            end = min(len(target_genes), start + gene_block_size)
            block_started = time.time()
            prediction = aggregate_block(transfer, source_target[:, start:end])
            handle["X"][:, start:end] = prediction.astype(settings["output_dtype"], copy=False)
            handle["gene_stats"]["predicted_mean"][start:end] = prediction.mean(axis=0, dtype=np.float64)
            handle["gene_stats"]["predicted_nonzero_fraction"][start:end] = np.count_nonzero(
                prediction, axis=0
            ) / float(prediction.shape[0])
            handle["gene_stats"]["predicted_max"][start:end] = prediction.max(axis=0)
            completed[block_index] = True
            handle.flush()
            del prediction
            LOGGER.info(
                "Gene blocks: %d/%d; genes=%d:%d; block=%.1fs; elapsed=%.1fs; file=%.2fGiB; %s",
                int(np.count_nonzero(completed[:])),
                len(completed),
                start,
                end,
                time.time() - block_started,
                time.time() - started,
                output_path.stat().st_size / 1024**3,
                memory_gib(),
            )
        handle.attrs["complete"] = bool(np.all(completed[:]))
        handle.attrs["completed_at_unix"] = time.time()
        handle.flush()
        complete = bool(handle.attrs["complete"])

    manifest = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "output_path": str(output_path),
        "shape": [int(neighbors.shape[0]), int(len(target_genes))],
        "dtype": settings["output_dtype"],
        "gene_block_size": gene_block_size,
        "compression": settings.get("compression"),
        "compression_level": settings.get("compression_level"),
        "aggregation": "uniform_mean",
        "complete": complete,
        "file_size_bytes": output_path.stat().st_size,
        "uncompressed_X_bytes": int(neighbors.shape[0] * len(target_genes) * np.dtype(settings["output_dtype"]).itemsize),
    }
    atomic_write_json(run_dir / "imputation_manifest.json", manifest)
    return manifest


def sampled_quantiles(values: np.ndarray, seed: int, sample_size: int = 2_000_000) -> dict[str, float]:
    flat = values.reshape(-1)
    if len(flat) > sample_size:
        rng = np.random.default_rng(seed)
        selected = np.asarray(flat[rng.choice(len(flat), size=sample_size, replace=False)])
    else:
        selected = np.asarray(flat)
    probs = (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
    quantiles = np.quantile(selected, probs)
    return {str(prob): float(value) for prob, value in zip(probs, quantiles)}


def validate_result(config: dict[str, Any], run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    neighbor_manifest = json.loads((run_dir / "neighbor_manifest.json").read_text(encoding="utf-8"))
    imputation_manifest = json.loads((run_dir / "imputation_manifest.json").read_text(encoding="utf-8"))
    if not neighbor_manifest.get("complete") or not imputation_manifest.get("complete"):
        raise ValueError("Validation requires complete neighbor and imputation stages.")

    neighbors = np.load(neighbor_manifest["neighbors_path"], mmap_mode="r")
    source_indices = np.load(run_dir / "target_source_var_indices.npy")
    check_spots = min(100, neighbors.shape[0])
    check_genes = min(25, len(source_indices))
    source_full = load_h5ad_csr(Path(config["source_rna_h5ad"]))
    source_check = source_full[:, source_indices[:check_genes]].tocsr()
    del source_full
    neighbor_check = np.asarray(neighbors[:check_spots])
    expected = (
        source_check[neighbor_check.reshape(-1)]
        .toarray()
        .reshape(check_spots, neighbor_check.shape[1], check_genes)
        .mean(axis=1)
        .astype(np.float32)
    )

    output_path = Path(imputation_manifest["output_path"])
    with h5py.File(output_path, "r") as handle:
        observed = handle["X"][:check_spots, :check_genes]
        completed = handle["completed_gene_blocks"][:]
        source_nnz = handle["gene_stats"]["source_nnz"][:]
        predicted_nonzero = handle["gene_stats"]["predicted_nonzero_fraction"][:]
        target_spatial = handle["obsm"]["spatial"][:]
        target_genes = handle["var_names"].asstr()[:]
        attrs_complete = bool(handle.attrs.get("complete", False))
        sampled_output = handle["X"][
            : min(2048, handle["X"].shape[0]), : min(64, handle["X"].shape[1])
        ]
        output_shape = list(handle["X"].shape)
        output_dtype = str(handle["X"].dtype)
    with h5py.File(config["target_rna_h5ad"], "r") as target_handle:
        raw_target_spatial = np.asarray(target_handle["obsm"]["spatial"][: len(target_spatial)])
    genes_csv = pd.read_csv(run_dir / "target_genes.csv")
    difference = np.abs(observed - expected)
    validations = {
        "config_hash_match": imputation_manifest.get("config_hash") == audit["config_hash"],
        "hdf5_complete_attribute": attrs_complete,
        "all_gene_blocks_complete": bool(np.all(completed)),
        "output_shape_match": output_shape
        == [int(audit["effective_target_spots"]), int(audit["effective_imputation_gene_count"])],
        "output_dtype_float32": output_dtype == "float32",
        "spatial_rows_match_raw_target": bool(np.array_equal(target_spatial, raw_target_spatial)),
        "gene_order_match_csv": bool(
            np.array_equal(target_genes, genes_csv["gene"].astype(str).to_numpy())
        ),
        "sampled_output_finite": bool(np.isfinite(sampled_output).all()),
        "source_zero_genes_predict_zero": bool(np.all(predicted_nonzero[source_nnz == 0] == 0)),
        "neighbors_in_source_range": bool(
            neighbors.min() >= 0
            and neighbors.max() < int(audit["source_embedding_shape"][0])
        ),
        "sampled_neighbor_rows_unique": bool(
            all(
                len(np.unique(row)) == neighbors.shape[1]
                for row in neighbors[: min(10_000, len(neighbors))]
            )
        ),
        "direct_cosie_loop_allclose": bool(
            np.allclose(observed, expected, rtol=1e-6, atol=1e-7)
        ),
    }
    k = int(config["knn"]["k"])
    validation = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "status": "pass" if all(validations.values()) else "fail",
        "checks": validations,
        "direct_cosie_loop_check": {
            "spots": check_spots,
            "genes": check_genes,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "prediction_times_k_integer_max_error": float(
                np.max(np.abs(observed * k - np.rint(observed * k)))
            ),
        },
    }
    atomic_write_json(run_dir / "validation.json", validation)
    if validation["status"] != "pass":
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Result validation failed: {failed}")
    LOGGER.info(
        "Validation passed; direct COSIE-loop max abs error=%.3g",
        validation["direct_cosie_loop_check"]["max_absolute_error"],
    )
    return validation


def summarize(config: dict[str, Any], run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    neighbor_manifest = json.loads((run_dir / "neighbor_manifest.json").read_text(encoding="utf-8"))
    imputation_manifest = json.loads((run_dir / "imputation_manifest.json").read_text(encoding="utf-8"))
    neighbors = np.load(neighbor_manifest["neighbors_path"], mmap_mode="r")
    distances = np.load(neighbor_manifest["distances_path"], mmap_mode="r")
    reuse = np.bincount(neighbors.reshape(-1), minlength=int(neighbor_manifest["source_count"]))
    np.save(run_dir / "source_neighbor_reuse.npy", reuse.astype(np.int64, copy=False))
    output_path = Path(imputation_manifest["output_path"])
    with h5py.File(output_path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("Cannot summarize an incomplete imputation HDF5.")
        source_nnz = handle["gene_stats"]["source_nnz"][:]
        source_total = handle["gene_stats"]["source_total"][:]
        predicted_mean = handle["gene_stats"]["predicted_mean"][:]
        predicted_nonzero = handle["gene_stats"]["predicted_nonzero_fraction"][:]
        predicted_max = handle["gene_stats"]["predicted_max"][:]
        target_genes = handle["var_names"].asstr()[:]

    gene_qc_path = run_dir / "gene_qc.csv.gz"
    pd.DataFrame(
        {
            "target_gene_index": np.arange(len(target_genes), dtype=np.int32),
            "gene": target_genes,
            "source_nnz": source_nnz,
            "source_total": source_total,
            "predicted_mean": predicted_mean,
            "predicted_nonzero_fraction": predicted_nonzero,
            "predicted_max": predicted_max,
        }
    ).to_csv(gene_qc_path, index=False, compression="gzip")
    validation_path = run_dir / "validation.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else None
    )

    summary = {
        "format": FORMAT_NAME,
        "config_hash": audit["config_hash"],
        "run_name": config["run_name"],
        "status": "complete",
        "method": "COSIE-compatible Annoy KNN with uniform source-expression mean",
        "target_section": config["target_section"],
        "source_section": config["source_section"],
        "embedding_version": str(Path(config["target_embedding"]).parts[-4]),
        "target_spots": int(neighbors.shape[0]),
        "source_spots": int(neighbor_manifest["source_count"]),
        "target_genes": int(len(source_nnz)),
        "k": int(neighbor_manifest["k"]),
        "metric": neighbor_manifest["metric"],
        "distance_quantiles_sampled": sampled_quantiles(distances, int(config["knn"]["seed"])),
        "top1_distance_quantiles": sampled_quantiles(distances[:, 0], int(config["knn"]["seed"])),
        "source_neighbor_reuse": {
            "unused_source_spots": int(np.count_nonzero(reuse == 0)),
            "used_source_spots": int(np.count_nonzero(reuse > 0)),
            "max_reuse": int(reuse.max()),
            "median_reuse_among_used": float(np.median(reuse[reuse > 0])) if np.any(reuse > 0) else 0.0,
        },
        "gene_qc": {
            "zero_in_source": int(np.count_nonzero(source_nnz == 0)),
            "zero_after_imputation": int(np.count_nonzero(predicted_nonzero == 0)),
            "predicted_mean_quantiles": {
                str(prob): float(value)
                for prob, value in zip((0, 0.25, 0.5, 0.75, 1), np.quantile(predicted_mean, (0, 0.25, 0.5, 0.75, 1)))
            },
            "predicted_nonzero_fraction_quantiles": {
                str(prob): float(value)
                for prob, value in zip((0, 0.25, 0.5, 0.75, 1), np.quantile(predicted_nonzero, (0, 0.25, 0.5, 0.75, 1)))
            },
            "global_predicted_max": float(np.max(predicted_max)),
        },
        "validation": validation,
        "gene_qc_table": str(gene_qc_path),
        "source_neighbor_reuse_path": str(run_dir / "source_neighbor_reuse.npy"),
        "output": imputation_manifest,
        "memory_at_summary": memory_gib(),
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    distance_median = summary["distance_quantiles_sampled"]["0.5"]
    validation_status = validation["status"] if validation else "not run"
    text = f"""# {config['run_name']}\n\nStatus: **complete**\n\n- Method: COSIE-compatible Annoy KNN, uniform mean of source raw RNA counts\n- Transfer: `{config['source_section']}` → `{config['target_section']}`\n- Embedding: `{config['target_embedding']}`\n- Shapes: {neighbors.shape[0]:,} target spots × {len(source_nnz):,} source-only genes\n- K / metric: {neighbor_manifest['k']} / `{neighbor_manifest['metric']}`\n- Sampled median neighbor distance: {distance_median:.6g}\n- Source spots used by at least one neighbor: {summary['source_neighbor_reuse']['used_source_spots']:,} / {neighbor_manifest['source_count']:,}\n- Genes with zero source counts: {summary['gene_qc']['zero_in_source']:,}\n- Direct COSIE-loop validation: {validation_status}\n- Result: `{output_path}`\n- Per-gene QC: `{gene_qc_path}`\n- Compressed file size: {output_path.stat().st_size / 1024**3:.2f} GiB\n- Raw float32 X size: {imputation_manifest['uncompressed_X_bytes'] / 1024**3:.2f} GiB\n\nThe HDF5 matrix is chunked. Read selected genes or spot blocks; do not load `/X` in full.\n"""
    (run_dir / "SUMMARY.md").write_text(text, encoding="utf-8")
    LOGGER.info("Summary written: %s", run_dir / "SUMMARY.md")
    return summary


def main() -> None:
    args = parse_args()
    config, run_dir, config_hash = load_config(args.config.resolve(), args.run_dir)
    if (args.limit_target_spots is not None or args.limit_genes is not None) and args.run_dir is None:
        raise ValueError("Limited tests require an explicit separate --run-dir.")
    setup_logging(run_dir)
    LOGGER.info("Run=%s stage=%s config_hash=%s", config["run_name"], args.stage, config_hash)

    if args.stage in ("audit", "all"):
        audit = audit_inputs(
            config,
            run_dir,
            config_hash,
            args.limit_target_spots,
            args.limit_genes,
        )
    else:
        audit = load_or_require_audit(run_dir, config_hash)

    if args.stage in ("neighbors", "all"):
        compute_neighbors(config, run_dir, audit)
    if args.stage in ("impute", "all"):
        impute(config, run_dir, audit)
    if args.stage in ("validate", "all"):
        validate_result(config, run_dir, audit)
    if args.stage in ("summarize", "all"):
        summarize(config, run_dir, audit)
    LOGGER.info("Requested stage completed successfully. %s", memory_gib())


if __name__ == "__main__":
    main()
