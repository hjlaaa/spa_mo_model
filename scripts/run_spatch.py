#!/usr/bin/env python3
"""Train spa_mo_model on full-resolution spatch HE+RNA+Protein data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import resolve_model_config, parse_dataset_args, describe_run_config
from data_io.preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel
from training.fit import (
    CudaMemoryMonitor,
    initialize_model_ot_prior,
    json_safe,
    train_small_crc_model,
)
from scripts.run_crc_stereocite import (
    save_final_embeddings,
)


SECTIONS = ("section1", "section2")
FILES = {
    "section1": ("adata_xenium_bin_filter.h5ad", "adata_codex_bin_filter.h5ad", "adata_he.h5ad"),
    "section2": ("adata_hd_filter.h5ad", "adata_codex_filter.h5ad", "adata_he.h5ad"),
}
CACHE_SCHEMA_VERSION = 1
CACHE_METADATA_FILES = (
    "spot_metadata.csv.gz",
    "shared_rna_genes.csv",
    "protein_markers_after_dapi_removal.csv",
)


TRAINING_DEFAULTS = {
    "training_loss_only": True,
    "checkpoint_ot_attention": True,
    "checkpoint_encoder_fusion": True,
    "checkpoint_decoder_chunks": True,
    "checkpoint_graph_encoder": True,
    "cache_spatial_graphs": True,
    "log_every": 1,
    "log_cuda_memory": True,
    "log_cuda_memory_detail": True,
    "save_candidate_qc": False,
}


def get_dataset_defaults():
    """Current entry defaults; suite arguments remain explicit overrides."""
    return {
        'data_dir': None,
        'input_identity_manifest': None,
        'output_dir': PROJECT_ROOT / "results" / "spatch" / "fullspot_200ep_gpu_seed42",
        'epochs': 200,
        'seed': 42,
        'device': "cuda",
        'n_comps': 50,
        'hvg_num': 3000,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'lambda_contrast': 0.1,
        'update_interval': 20,
        'uot_max_iter': 100,
        'candidate_backend': "faiss_ivf",
        'initial_modality_candidate_k': 100,
        'candidate_k': 200,
        'attention_topk': 10,
        'faiss_nlist': 4096,
        'faiss_nprobe': 64,
        'faiss_device': "gpu",
        'faiss_train_sample_size': 100000,
        'faiss_query_batch_size': 2048,
        'uot_epsilon': 0.05,
        'uot_tau_a': 1.0,
        'uot_tau_b': 1.0,
        'uot_stabilizer': 1e-8,
        'spatial_knn_k': 10,
        'graphsage_edge_batch_size': 100000,
        'decoder_chunk_size': 8192,
        'ot_attention_source_chunk_size': 2048,
        'amp_dtype': "bf16",
        'post_ot_graphsage_scale': 1.0,
        'preprocessed_cache_dir': None,
        'build_preprocessed_cache': False,
        'no_harmony': False,
        'overwrite': False,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spa_mo_model on full-resolution spatch.")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Optional raw source selection; default consumes the selected cache's recorded input.",
    )
    parser.add_argument(
        "--input_identity_manifest", type=Path, default=None,
        help="Optional trusted source_files JSON for an explicitly requested raw input; raw files are not read.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_comps", type=int, default=None)
    parser.add_argument("--hvg_num", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--uot_max_iter", type=int, default=None)
    parser.add_argument("--candidate_backend", default=None)
    parser.add_argument("--initial_modality_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--attention_topk", type=int, default=None)
    parser.add_argument("--faiss_nlist", type=int, default=None)
    parser.add_argument("--faiss_nprobe", type=int, default=None)
    parser.add_argument("--faiss_device", default=None)
    parser.add_argument("--faiss_train_sample_size", type=int, default=None)
    parser.add_argument("--faiss_query_batch_size", type=int, default=None)
    parser.add_argument("--uot_epsilon", type=float, default=None)
    parser.add_argument("--uot_tau_a", type=float, default=None)
    parser.add_argument("--uot_tau_b", type=float, default=None)
    parser.add_argument("--uot_stabilizer", type=float, default=None)
    parser.add_argument("--spatial_knn_k", type=int, default=None)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=None)
    parser.add_argument("--decoder_chunk_size", type=int, default=None)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=None)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "none"], default=None)
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=None,
        help="Fixed scale on the post-OT GraphSAGE residual branch.",
    )
    parser.add_argument(
        "--preprocessed_cache_dir",
        type=Path,
        default=None,
        help="Shared model-ready PCA/Harmony feature cache directory.",
    )
    parser.add_argument(
        "--build_preprocessed_cache",
        action=argparse.BooleanOptionalAction,
        help="Unsupported by this cache-only training entry; requesting build stops before preprocessing.",
        default=None,
    )
    parser.add_argument("--no_harmony", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parse_dataset_args(parser, argv, get_dataset_defaults())


def require_gpu(device: str) -> dict[str, object]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("spa_mo_model spatch is GPU-only and CUDA is unavailable.")
    probe = torch.ones(1, device="cuda")
    del probe
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def canonical_ids(section: str, coords: np.ndarray) -> pd.Index:
    values = pd.Index([f"{section}:{int(x)}:{int(y)}" for x, y in coords])
    if not values.is_unique:
        raise ValueError(f"{section}: canonical coordinate IDs are not unique.")
    return values


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def expected_input_paths(data_dir: Path) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for section in SECTIONS:
        section_dir = data_dir / section
        paths[section] = {
            key: section_dir / name
            for key, name in zip(("RNA", "Protein", "HE"), FILES[section])
        }
        for modality, path in paths[section].items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {section} {modality}: {path}")
    return paths


def cache_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Historical schema-1 build fields; retain_processed is provenance at load time."""
    return {
        "sections": list(SECTIONS),
        "modalities": ["RNA", "Protein", "HE"],
        "n_comps": int(args.n_comps),
        "hvg_num": int(args.hvg_num),
        "hvg_num_by_modality": {"RNA": int(args.hvg_num), "Protein": None, "HE": None},
        "target_sum": None,
        "use_harmony": not bool(args.no_harmony),
        "metacell": False,
        "memory_efficient": True,
        "retain_processed": False,
        "dapi_removed": True,
    }


def source_file_records(paths: dict[str, dict[str, Path]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for section in SECTIONS:
        for modality in ("RNA", "Protein", "HE"):
            path = paths[section][modality].resolve()
            stat = path.stat()
            records.append(
                {
                    "section": section,
                    "modality": modality,
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": sha256_file(path),
                }
            )
    return records


def validate_source_stats(
    manifest: dict[str, object],
    paths: dict[str, dict[str, Path]],
) -> None:
    records = {
        (record["section"], record["modality"]): record
        for record in manifest["source_files"]
    }
    for section in SECTIONS:
        for modality in ("RNA", "Protein", "HE"):
            path = paths[section][modality].resolve()
            record = records.get((section, modality))
            if record is None:
                raise ValueError(f"Cache lacks source record for {section}/{modality}.")
            stat = path.stat()
            actual = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
            expected = (
                record["path"],
                int(record["size_bytes"]),
                int(record["mtime_ns"]),
            )
            if actual != expected:
                raise ValueError(
                    f"SPATCH input changed for {section}/{modality}: {actual} != {expected}."
                )


def save_preprocessed_cache(
    cache_dir: Path,
    feature_dict: dict[str, dict[str, torch.Tensor]],
    spatial_dict: dict[str, np.ndarray],
    paths: dict[str, dict[str, Path]],
    audits: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    cache_dir = cache_dir.resolve()
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite preprocessing cache: {cache_dir}")
    temporary = cache_dir.with_name(f".{cache_dir.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    arrays: list[dict[str, object]] = []
    for section in SECTIONS:
        for modality in ("RNA", "Protein", "HE"):
            value = feature_dict[section][modality]
            array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
            relative = f"{section}_{modality}.npy"
            path = temporary / relative
            np.save(path, array, allow_pickle=False)
            arrays.append(
                {
                    "kind": "feature",
                    "section": section,
                    "modality": modality,
                    "relative_path": relative,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        coords = np.ascontiguousarray(spatial_dict[section], dtype=np.float32)
        relative = f"{section}_spatial.npy"
        path = temporary / relative
        np.save(path, coords, allow_pickle=False)
        arrays.append(
            {
                "kind": "spatial",
                "section": section,
                "modality": None,
                "relative_path": relative,
                "shape": list(coords.shape),
                "dtype": str(coords.dtype),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    artifacts: list[dict[str, object]] = []
    for name in CACHE_METADATA_FILES:
        source = args.output_dir / name
        target = temporary / name
        shutil.copy2(source, target)
        artifacts.append(
            {
                "relative_path": name,
                "size_bytes": int(target.stat().st_size),
                "sha256": sha256_file(target),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": "spatch",
        "cache_boundary": "model_ready_feature_dict_after_PCA_Harmony",
        "parameters": cache_parameters(args),
        "arrays": arrays,
        "metadata_artifacts": artifacts,
        "source_files": source_file_records(paths),
        "alignment": audits,
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "anndata": package_version("anndata"),
            "scanpy": package_version("scanpy"),
            "harmonypy": package_version("harmonypy"),
            "scikit-learn": package_version("scikit-learn"),
        },
        "preprocessing_sources": {
            "run_spatch.py": sha256_file(Path(__file__).resolve()),
            "data_io/preprocessing.py": sha256_file(
                PROJECT_ROOT / "data_io/preprocessing.py"
            ),
        },
        "created_at": time.time(),
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(cache_dir)
    manifest["manifest_sha256"] = sha256_file(cache_dir / "manifest.json")
    return manifest


def validate_cache_input_identity(
    manifest: dict[str, object],
    data_dir: Path | None,
    identity_manifest: Path | None,
) -> dict[str, object]:
    """Compare recorded input identities without opening or re-hashing raw data."""
    expected_keys = {(section, modality) for section in SECTIONS for modality in ("RNA", "Protein", "HE")}

    def index_sources(records, label):
        if not isinstance(records, list):
            raise ValueError(f"{label}: missing source_files records.")
        indexed = {}
        for record in records:
            key = (record.get("section"), record.get("modality"))
            if key not in expected_keys or key in indexed:
                raise ValueError(f"{label}: unexpected or duplicate source identity {key}.")
            digest = record.get("sha256")
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or not isinstance(record.get("path"), str)
                or not Path(record["path"]).is_absolute()
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] < 0
            ):
                raise ValueError(f"{label}: incomplete source identity for {key}.")
            indexed[key] = record
        if set(indexed) != expected_keys:
            raise ValueError(f"{label}: missing section/modality source identities.")
        return indexed

    cached = index_sources(manifest.get("source_files"), "Cache")
    requested_dir = Path(data_dir).resolve() if data_dir is not None else None
    evidence = "selected_cache_recorded_input"
    if identity_manifest is not None:
        identity_manifest = Path(identity_manifest).resolve()
        identity = json.loads(identity_manifest.read_text(encoding="utf-8"))
        if identity.get("dataset", "spatch") != "spatch":
            raise ValueError("Requested input identity dataset conflicts with SPATCH cache.")
        requested = index_sources(identity.get("source_files"), "Requested input")
        for key, record in requested.items():
            if (record["sha256"], record["size_bytes"]) != (cached[key]["sha256"], cached[key]["size_bytes"]):
                raise ValueError(f"Requested input identity conflicts with cache for {key}.")
        evidence = "caller_supplied_source_records_match_cache"
    else:
        requested = cached
    if requested_dir is not None:
        for section in SECTIONS:
            for modality, filename in zip(("RNA", "Protein", "HE"), FILES[section]):
                requested_path = (requested_dir / section / filename).resolve()
                recorded_path = Path(requested[(section, modality)]["path"]).resolve()
                if requested_path != recorded_path:
                    raise ValueError(
                        f"Cannot establish requested input identity for {section}/{modality}: "
                        "--data_dir differs from the supplied source records; provide a matching "
                        "--input_identity_manifest. Raw files are not inspected or rebuilt."
                    )
        if identity_manifest is None:
            evidence = "selected_cache_recorded_source_location"
    return {
        "requested_data_dir": str(requested_dir) if requested_dir is not None else None,
        "input_identity_manifest": str(identity_manifest) if identity_manifest is not None else None,
        "evidence": evidence,
        "current_raw_files_verified": False,
        "historical_source_files": manifest["source_files"],
    }


def load_preprocessed_cache(
    cache_dir: Path,
    data_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, object],
]:
    cache_dir = cache_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == cache_dir or cache_dir in output_dir.parents:
        raise ValueError("SPATCH cache is read-only; output_dir must be outside the cache.")
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SPATCH preprocessing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported SPATCH preprocessing cache schema.")
    if manifest.get("dataset") != "spatch" or manifest.get("cache_boundary") != "model_ready_feature_dict_after_PCA_Harmony":
        raise ValueError("SPATCH cache dataset or model-ready boundary differs.")
    parameters = manifest.get("parameters", {})
    expected_parameters = cache_parameters(args)
    if not isinstance(parameters, dict) or set(parameters) - set(expected_parameters):
        raise ValueError("SPATCH cache has unsupported preprocessing parameter fields.")
    for key, expected in expected_parameters.items():
        if key != "retain_processed" and (key not in parameters or parameters[key] != expected):
            raise ValueError(f"SPATCH cache parameter {key} differs: {parameters.get(key)!r} != {expected!r}.")
    if args.n_comps <= 0:
        raise ValueError("SPATCH cache n_comps must be positive.")
    input_identity = validate_cache_input_identity(
        manifest, data_dir, getattr(args, "input_identity_manifest", None)
    )
    # These are schema-1 records, not a request to run the current preprocessor.
    # Whole source hashes and raw path/mtime are retained below as provenance.
    expected_layout = [
        (section, kind, modality)
        for section in SECTIONS
        for kind, modality in (("feature", "RNA"), ("feature", "Protein"), ("feature", "HE"), ("spatial", None))
    ]
    arrays = manifest.get("arrays", [])
    if [(r.get("section"), r.get("kind"), r.get("modality")) for r in arrays] != expected_layout:
        raise ValueError("SPATCH cache array section/modality identity or order differs.")
    artifacts = manifest.get("metadata_artifacts", [])
    if [r.get("relative_path") for r in artifacts] != list(CACHE_METADATA_FILES):
        raise ValueError("SPATCH cache metadata artifact identity or order differs.")
    relative_paths = [r.get("relative_path") for r in arrays + artifacts]
    if any(not isinstance(p, str) for p in relative_paths) or len(set(relative_paths)) != len(relative_paths):
        raise ValueError("SPATCH cache has invalid or duplicate artifact paths.")

    def checked_file(record):
        relative = Path(record["relative_path"])
        path = (cache_dir / relative).resolve()
        if relative.is_absolute() or cache_dir not in path.parents:
            raise ValueError(f"Cached artifact must remain inside the cache: {relative}")
        if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Invalid cached artifact file or size: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Cached artifact checksum mismatch: {path}")
        return path

    metadata_paths = {record["relative_path"]: checked_file(record) for record in artifacts}
    genes = pd.read_csv(metadata_paths["shared_rna_genes.csv"])["gene"]
    markers = pd.read_csv(metadata_paths["protein_markers_after_dapi_removal.csv"])["protein_marker"]
    if genes.isna().any() or not genes.is_unique or len(genes) < args.n_comps:
        raise ValueError("SPATCH cached shared RNA gene identity/count differs.")
    if markers.isna().any() or not markers.is_unique or len(markers) != 16 or markers.str.upper().eq("DAPI").any():
        raise ValueError("SPATCH cached Protein marker identity or DAPI removal differs.")
    # Existing SPATCH inputs have 16 retained markers; interpret the original PCA dimension contract.
    expected_dims = {"RNA": args.n_comps, "Protein": args.n_comps if args.n_comps <= 16 else 15, "HE": args.n_comps}
    alignment = manifest.get("alignment", {})
    if set(alignment) != set(SECTIONS):
        raise ValueError("SPATCH cache alignment section identity differs.")
    for section in SECTIONS:
        audit = alignment[section]
        n_spots = audit.get("n_spots")
        if not isinstance(n_spots, int) or n_spots <= 0:
            raise ValueError(f"{section}: invalid cached alignment spot count.")
        if audit.get("rna_shape") != [n_spots, len(genes)] or audit.get("protein_shape_after_dapi_removal") != [n_spots, len(markers)]:
            raise ValueError(f"{section}: cached alignment feature identity/count differs.")
        he_shape = audit.get("he_shape", [])
        if len(he_shape) != 2 or he_shape[0] != n_spots or he_shape[1] < args.n_comps:
            raise ValueError(f"{section}: cached HE alignment shape differs.")
        if any(audit.get(key) is not True for key in ("spatial_alignment", "canonical_ids_unique", "dapi_removed")):
            raise ValueError(f"{section}: cached alignment/DAPI identity is not established.")
    feature_dict: dict[str, dict[str, torch.Tensor]] = {}
    spatial_dict: dict[str, np.ndarray] = {}
    for record in arrays:
        path = checked_file(record)
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Cached array metadata mismatch: {path}")
        section = record["section"]
        width = expected_dims[record["modality"]] if record["kind"] == "feature" else 2
        if array.dtype != np.float32 or list(array.shape) != [alignment[section]["n_spots"], width]:
            raise ValueError(f"{section}/{record['modality']}: cached spot count, feature dimensions or dtype differ.")
        if not np.isfinite(array).all():
            raise ValueError(f"Cached array contains non-finite values: {path}")
        if record["kind"] == "feature":
            feature_dict.setdefault(section, {})[record["modality"]] = torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32)
            )
        else:
            spatial_dict[section] = np.ascontiguousarray(array, dtype=np.float32)
    metadata = pd.read_csv(
        metadata_paths["spot_metadata.csv.gz"], usecols=["block_id", "section", "x", "y"],
        dtype={"block_id": str, "section": str},
    )
    if len(metadata) != sum(alignment[s]["n_spots"] for s in SECTIONS):
        raise ValueError("SPATCH cached metadata spot count differs.")
    offset = 0
    for section in SECTIONS:
        count = alignment[section]["n_spots"]
        rows = metadata.iloc[offset:offset + count]
        coords = spatial_dict[section]
        if not rows["section"].eq(section).all() or not np.array_equal(rows[["x", "y"]].to_numpy(), coords):
            raise ValueError(f"{section}: cached metadata section/spot order or spatial coordinates differ.")
        if not np.array_equal(rows["block_id"].to_numpy(), canonical_ids(section, coords).to_numpy()):
            raise ValueError(f"{section}: cached metadata canonical spot identity/order differs.")
        offset += count
    # Complete all checks before copying metadata; never write through an output alias into the cache.
    for name in CACHE_METADATA_FILES:
        destination = (output_dir / name).resolve()
        if output_dir not in destination.parents or destination == metadata_paths[name]:
            raise ValueError(f"Metadata output escapes output_dir or aliases the cache: {destination}")
    for name in CACHE_METADATA_FILES:
        shutil.copy2(metadata_paths[name], output_dir / name)
    cache_info = {
        "mode": "loaded",
        "path": str(cache_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_identity": input_identity,
        "historical_preprocessing_sources": manifest.get("preprocessing_sources", {}),
        "historical_preprocessing_parameters": parameters,
        "feature_row_identity_limit": (
            "Schema 1 has no barcode binding inside each feature array. Checksums and metadata/spatial "
            "order checks do not independently establish every feature row's biological identity."
        ),
    }
    return feature_dict, spatial_dict, alignment, cache_info


def process_memory_gib() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            values[key] = float(value.strip().split()[0]) / 1024**2
    return {
        "rss_gib": values.get("VmRSS", float("nan")),
        "peak_rss_gib": values.get("VmHWM", float("nan")),
    }


def record_memory(output_dir: Path, stage: str) -> None:
    payload = {"time": time.time(), "stage": stage, **process_memory_gib()}
    if torch.cuda.is_available():
        payload.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "cuda_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            }
        )
    with open(output_dir / "memory_timeline.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(f"SPATCH_MEMORY {json.dumps(payload, sort_keys=True)}", flush=True)


def inspect_full_inputs(data_dir: Path, output_dir: Path):
    paths = expected_input_paths(data_dir)
    backed = {}
    for section in SECTIONS:
        for key, path in paths[section].items():
            backed[(section, key)] = ad.read_h5ad(path, backed="r")

    try:
        rna_names = [pd.Index(backed[(section, "RNA")].var_names.astype(str)) for section in SECTIONS]
        if any(not names.is_unique for names in rna_names):
            raise ValueError("RNA var_names must be unique.")
        common_set = set(rna_names[1])
        common_genes = [gene for gene in rna_names[0] if gene in common_set]
        if len(common_genes) != 4828:
            raise ValueError(f"Expected 4828 common genes, found {len(common_genes)}.")

        audits = {}
        metadata_tables = []
        retained_protein_markers = None
        for section in SECTIONS:
            raw_rna = backed[(section, "RNA")]
            raw_protein = backed[(section, "Protein")]
            raw_he = backed[(section, "HE")]
            if not (raw_rna.n_obs == raw_protein.n_obs == raw_he.n_obs):
                raise ValueError(f"{section}: modality row counts differ.")
            coords = np.asarray(raw_rna.obsm["spatial"])
            if not np.array_equal(coords, np.asarray(raw_protein.obsm["spatial"])):
                raise ValueError(f"{section}: RNA/Protein coordinates differ.")
            if not np.array_equal(coords, np.asarray(raw_he.obsm["spatial"])):
                raise ValueError(f"{section}: RNA/HE coordinates differ.")
            ids = canonical_ids(section, coords)

            protein_names = pd.Index(raw_protein.var_names.astype(str))
            keep_protein = protein_names.str.upper() != "DAPI"
            if int((~keep_protein).sum()) != 1 or int(keep_protein.sum()) != 16:
                raise ValueError(f"{section}: DAPI exclusion failed.")

            protein_markers = protein_names[keep_protein].astype(str).tolist()
            if retained_protein_markers is None:
                retained_protein_markers = protein_markers
            elif retained_protein_markers != protein_markers:
                raise ValueError("Protein marker order differs across sections after DAPI removal.")
            if "DAPI" in {name.upper() for name in protein_markers}:
                raise RuntimeError(f"{section}: DAPI remains in Protein.")

            meta = raw_rna.obs.copy()
            meta.insert(0, "block_id", ids.astype(str))
            meta.insert(1, "section", section)
            meta["original_rna_obs_name"] = raw_rna.obs_names.astype(str)
            meta["x"] = coords[:, 0]
            meta["y"] = coords[:, 1]
            if section == "section1":
                meta["cell_type_common"] = meta["annotation_transferred"].astype("string").mask(
                    meta["annotation_transferred"].astype("string").eq("Unknown")
                )
                if "annotation_transferred" in raw_protein.obs:
                    meta["codex_coarse_label"] = raw_protein.obs[
                        "annotation_transferred"
                    ].astype("string").to_numpy()
            else:
                meta["cell_type_common"] = meta["annotation"].astype("string")
            metadata_tables.append(meta.reset_index(drop=True))
            audits[section] = {
                "n_spots": int(raw_rna.n_obs),
                "rna_shape": [int(raw_rna.n_obs), len(common_genes)],
                "protein_shape_before": list(raw_protein.shape),
                "protein_shape_after_dapi_removal": [
                    int(raw_protein.n_obs),
                    len(protein_markers),
                ],
                "he_shape": list(raw_he.shape),
                "spatial_alignment": True,
                "canonical_ids_unique": True,
                "dapi_removed": True,
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(metadata_tables, ignore_index=True).to_csv(
            output_dir / "spot_metadata.csv.gz", index=False, compression="gzip"
        )
        pd.Series(common_genes, name="gene").to_csv(output_dir / "shared_rna_genes.csv", index=False)
        pd.Series(
            retained_protein_markers, name="protein_marker"
        ).to_csv(output_dir / "protein_markers_after_dapi_removal.csv", index=False)
        return paths, common_genes, retained_protein_markers, audits
    finally:
        for obj in backed.values():
            obj.file.close()


def load_lightweight_modality(
    path: Path,
    section: str,
    modality: str,
    feature_names: list[str] | None,
) -> ad.AnnData:
    raw = ad.read_h5ad(path, backed="r")
    try:
        selected = raw if feature_names is None else raw[:, feature_names]
        matrix = selected.X
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        elif sp.issparse(matrix):
            matrix = matrix.copy()
        else:
            matrix = np.asarray(matrix)
        if sp.issparse(matrix):
            matrix = matrix.astype(np.float32, copy=False)
        else:
            matrix = np.asarray(matrix, dtype=np.float32, order="C")
        coords = np.asarray(raw.obsm["spatial"], dtype=np.float32)
        ids = canonical_ids(section, coords)
        var_names = pd.Index(selected.var_names.astype(str))
        result = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=ids.copy()),
            var=pd.DataFrame(index=var_names.copy()),
        )
        result.obsm["spatial"] = coords
        if modality == "Protein" and "DAPI" in {
            name.upper() for name in result.var_names.astype(str)
        }:
            raise RuntimeError(f"{section}: DAPI remains in lightweight Protein input.")
        return result
    finally:
        raw.file.close()


def preprocess_modalities_sequentially(
    paths: dict[str, dict[str, Path]],
    common_genes: list[str],
    protein_markers: list[str],
    args: argparse.Namespace,
):
    feature_dict: dict[str, dict[str, torch.Tensor]] = {}
    spatial_dict: dict[str, np.ndarray] = {}
    selectors = {
        "RNA": common_genes,
        "Protein": protein_markers,
        "HE": None,
    }
    hvg_counts = {"RNA": args.hvg_num, "Protein": None, "HE": None}
    for modality in ("RNA", "Protein", "HE"):
        record_memory(args.output_dir, f"{modality}_load_start")
        modality_inputs = [
            load_lightweight_modality(
                paths[section][modality],
                section,
                modality,
                selectors[modality],
            )
            for section in SECTIONS
        ]
        record_memory(args.output_dir, f"{modality}_loaded")
        feature_raw, spatial_raw, _ = load_cosie_style_data(
            {modality: modality_inputs},
            n_comps=args.n_comps,
            hvg_num=args.hvg_num,
            hvg_num_by_modality={modality: hvg_counts[modality]},
            target_sum=None,
            use_harmony=not args.no_harmony,
            memory_efficient=True,
            retain_processed=False,
        )
        modality_features = rename_sections(feature_raw)
        modality_spatial = rename_sections(spatial_raw)
        for section in SECTIONS:
            feature_dict.setdefault(section, {})[modality] = modality_features[section][modality]
            if section not in spatial_dict:
                spatial_dict[section] = np.asarray(
                    modality_spatial[section], dtype=np.float32
                )
            elif not np.array_equal(spatial_dict[section], modality_spatial[section]):
                raise ValueError(f"{section}: sequential modality spatial coordinates differ.")
        del modality_inputs, feature_raw, spatial_raw, modality_features, modality_spatial
        gc.collect()
        record_memory(args.output_dir, f"{modality}_processed_and_released")
    return feature_dict, spatial_dict, None


def rename_sections(mapping):
    return {("section1" if key == "s1" else "section2" if key == "s2" else key): value for key, value in mapping.items()}


def build_model_config(args):
    config = resolve_model_config(input_config={
        "training": {
            "device": "cuda", "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "loss": {"lambda_contrast": args.lambda_contrast},
        "uot": {
            "max_iter": args.uot_max_iter, "topk": args.attention_topk,
            "update_interval": args.update_interval,
            "epsilon_update": args.uot_epsilon,
            "tau_a": args.uot_tau_a, "tau_b": args.uot_tau_b,
        },
        "graph": {"knn_neighbors_spatial": args.spatial_knn_k},
        "graphsage": {
            "edge_batch_size": args.graphsage_edge_batch_size,
            "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
        },
    })
    return config


def resolve_run_config(args):
    # These are fixed SPATCH execution settings, not cache identity fields.
    for name, value in TRAINING_DEFAULTS.items():
        setattr(args, name, value)
    return describe_run_config(
        args, build_model_config(args), dataset="spatch",
        section_order=list(SECTIONS), modalities=["HE", "RNA", "Protein"],
        preprocessing=cache_parameters(args),
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")
    if args.build_preprocessed_cache:
        raise ValueError("SPATCH training only loads an existing cache; --build_preprocessed_cache is disabled.")
    if args.preprocessed_cache_dir is None:
        raise ValueError("SPATCH training requires an existing --preprocessed_cache_dir; preprocessing is disabled.")
    args.preprocessed_cache_dir = args.preprocessed_cache_dir.resolve()
    if not (args.preprocessed_cache_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing SPATCH preprocessing cache manifest: {args.preprocessed_cache_dir / 'manifest.json'}")
    if args.output_dir == args.preprocessed_cache_dir or args.preprocessed_cache_dir in args.output_dir.parents:
        raise ValueError("SPATCH cache is read-only; output_dir must be outside the cache.")
    run_config = resolve_run_config(args)
    if (args.output_dir / "run_summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"Existing result: {args.output_dir}; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale in ("run_failure.json", "memory_timeline.jsonl"):
            stale_path = args.output_dir / stale
            if stale_path.exists():
                stale_path.unlink()
    try:
        gpu = require_gpu(args.device)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        record_memory(args.output_dir, "preprocessed_cache_load_start")
        feature_dict, spatial_dict, audits, cache_info = load_preprocessed_cache(
            args.preprocessed_cache_dir,
            args.data_dir,
            args.output_dir,
            args,
        )
        processed = None
        record_memory(args.output_dir, "preprocessed_cache_load_end")
        section_order = run_config["section_order"]
        config = run_config["model_config"]
        model = StageMultiModalModel(config=config, feature_dict=feature_dict)
        if list(model._resolve_modality_order(feature_dict["section1"])) != ["HE", "RNA", "Protein"]:
            raise ValueError("Unexpected modality order.")

        monitor = CudaMemoryMonitor(True, args.output_dir, "cuda")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        history, outputs, ot_updates = train_small_crc_model(
            model, feature_dict, spatial_dict, processed, section_order, args, monitor
        )
        embedding_paths = save_final_embeddings(args.output_dir, outputs["final_embeddings"])
        summary = {
            "resolved_config": run_config,
            "dataset": "spatch",
            "method": "spa_mo_model",
            "mode": "train",
            "input_data_path": str(args.preprocessed_cache_dir),
            "input_data_kind": "model_ready_preprocessed_cache",
            "output_dir": str(args.output_dir),
            "section_names": section_order,
            "full_spot": True,
            "spatial_block_aggregation": False,
            "gpu": gpu,
            "train": True,
            "epochs": args.epochs,
            "seed": args.seed,
            "n_comps": args.n_comps,
            "hvg_num": args.hvg_num,
            "hvg_num_by_modality": {
                "RNA": args.hvg_num,
                "Protein": None,
                "HE": None,
            },
            "use_harmony": not args.no_harmony,
            "spot_sampling": "all",
            "max_spots_per_section": None,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_contrast": args.lambda_contrast,
            "update_interval": args.update_interval,
            "uot_max_iter": args.uot_max_iter,
            "ot_prior_mode": "candidate_sparse",
            "bidirectional_ot_attention": True,
            "candidate_backend": args.candidate_backend,
            "initial_modality_candidate_k": args.initial_modality_candidate_k,
            "candidate_k": args.candidate_k,
            "attention_topk": args.attention_topk,
            "faiss_nlist": args.faiss_nlist,
            "faiss_nprobe": args.faiss_nprobe,
            "faiss_device": args.faiss_device,
            "faiss_train_sample_size": args.faiss_train_sample_size,
            "faiss_query_batch_size": args.faiss_query_batch_size,
            "dynamic_candidate_source": "ot",
            "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
            "pre_post_graphsage_parameter_sharing": False,
            "ot_refresh_embedding_key": "ot_embeddings",
            "uot_epsilon": args.uot_epsilon,
            "uot_tau_a": args.uot_tau_a,
            "uot_tau_b": args.uot_tau_b,
            "uot_stabilizer": args.uot_stabilizer,
            "spatial_knn_k": args.spatial_knn_k,
            "graphsage_edge_batch_size": args.graphsage_edge_batch_size,
            "post_ot_graphsage_scale": float(args.post_ot_graphsage_scale),
            "training_loss_only": args.training_loss_only,
            "decoder_chunk_size": args.decoder_chunk_size,
            "ot_attention_source_chunk_size": args.ot_attention_source_chunk_size,
            "checkpoint_ot_attention": args.checkpoint_ot_attention,
            "checkpoint_encoder_fusion": args.checkpoint_encoder_fusion,
            "checkpoint_decoder_chunks": args.checkpoint_decoder_chunks,
            "checkpoint_graph_encoder": args.checkpoint_graph_encoder,
            "amp_dtype": args.amp_dtype,
            "amp_enabled": args.amp_dtype != "none",
            "cache_spatial_graphs": args.cache_spatial_graphs,
            "log_cuda_memory": args.log_cuda_memory,
            "log_cuda_memory_detail": args.log_cuda_memory_detail,
            "cuda_memory_trace_path": str(args.output_dir / "cuda_memory_trace.jsonl"),
            "save_candidate_qc": args.save_candidate_qc,
            "modalities": ["HE", "RNA", "Protein"],
            "dapi_removed_both_sections": True,
            "memory_optimization": {
                "sequential_modality_preprocessing": True,
                "delayed_he_loading": True,
                "lightweight_anndata": True,
                "float32_inputs": True,
                "retain_processed_anndata": False,
                "model_logic_changed": True,
            },
            "preprocessing_cache": cache_info,
            "alignment": audits,
            "feature_shapes": {
                section: {mod: list(value.shape) for mod, value in mods.items()}
                for section, mods in feature_dict.items()
            },
            "embedding_paths": embedding_paths,
            "training_history": history,
            "ot_updates": ot_updates,
        }
        (args.output_dir / "loss_history.json").write_text(
            json.dumps(json_safe(history), indent=2), encoding="utf-8"
        )
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("SPATCH_SPA_MO_MODEL_TRAIN: PASS")
    except Exception as exc:
        failure = {"dataset": "spatch", "method": "spa_mo_model", "status": "failed", "error": repr(exc)}
        (args.output_dir / "run_failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
