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

from model.configure import get_default_model_config
from model.data_preprocessing import load_cosie_style_data
from model.stage_model import StageMultiModalModel
from scripts.run_crc_stereocite import (
    CudaMemoryMonitor,
    initialize_model_ot_prior,
    json_safe,
    save_final_embeddings,
    train_small_crc_model,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spa_mo_model on full-resolution spatch.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "spatch")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "spatch" / "fullspot_200ep_gpu_seed42",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_comps", type=int, default=50)
    parser.add_argument("--hvg_num", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_contrast", type=float, default=0.1)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--uot_max_iter", type=int, default=100)
    parser.add_argument("--candidate_backend", default="faiss_ivf")
    parser.add_argument("--initial_modality_candidate_k", type=int, default=100)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--faiss_nlist", type=int, default=4096)
    parser.add_argument("--faiss_nprobe", type=int, default=64)
    parser.add_argument("--faiss_device", default="gpu")
    parser.add_argument("--faiss_train_sample_size", type=int, default=100000)
    parser.add_argument("--faiss_query_batch_size", type=int, default=2048)
    parser.add_argument(
        "--dynamic_candidate_source",
        choices=["fused", "ot", "final"],
        default="ot",
    )
    parser.add_argument(
        "--disable_context_attention_gate",
        action="store_true",
        default=True,
        help="Use the v3-compatible 512D attention gate without local-context reliability.",
    )
    parser.add_argument(
        "--enable_context_attention_gate",
        action="store_false",
        dest="disable_context_attention_gate",
        help="Explicitly enable the experimental 513D microenvironment-aware gate.",
    )
    parser.add_argument("--uot_epsilon", type=float, default=0.05)
    parser.add_argument("--uot_tau_a", type=float, default=1.0)
    parser.add_argument("--uot_tau_b", type=float, default=1.0)
    parser.add_argument("--uot_stabilizer", type=float, default=1e-8)
    parser.add_argument("--spatial_knn_k", type=int, default=10)
    parser.add_argument("--graphsage_edge_batch_size", type=int, default=100000)
    parser.add_argument("--decoder_chunk_size", type=int, default=8192)
    parser.add_argument("--ot_attention_source_chunk_size", type=int, default=2048)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument(
        "--post_ot_graphsage_scale",
        type=float,
        default=1.0,
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
        action="store_true",
        help="Build the shared cache atomically after preprocessing; refuses to overwrite.",
    )
    parser.add_argument("--no_harmony", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
            "model/data_preprocessing.py": sha256_file(
                PROJECT_ROOT / "model/data_preprocessing.py"
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


def load_preprocessed_cache(
    cache_dir: Path,
    data_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, object],
]:
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SPATCH preprocessing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported SPATCH preprocessing cache schema.")
    if manifest.get("parameters") != cache_parameters(args):
        raise ValueError(
            f"SPATCH cache parameters differ: {manifest.get('parameters')} != {cache_parameters(args)}"
        )
    expected_sources = manifest.get("preprocessing_sources", {})
    actual_sources = {
        "run_spatch.py": sha256_file(Path(__file__).resolve()),
        "model/data_preprocessing.py": sha256_file(
            PROJECT_ROOT / "model/data_preprocessing.py"
        ),
    }
    if expected_sources != actual_sources:
        raise ValueError(
            f"SPATCH preprocessing implementation changed: {actual_sources} != {expected_sources}."
        )
    paths = expected_input_paths(data_dir.resolve())
    validate_source_stats(manifest, paths)
    feature_dict: dict[str, dict[str, torch.Tensor]] = {}
    spatial_dict: dict[str, np.ndarray] = {}
    for record in manifest["arrays"]:
        path = cache_dir / record["relative_path"]
        if not path.is_file() or int(path.stat().st_size) != int(record["size_bytes"]):
            raise ValueError(f"Invalid cached array file: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Cached array checksum mismatch: {path}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Cached array metadata mismatch: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Cached array contains non-finite values: {path}")
        section = record["section"]
        if record["kind"] == "feature":
            feature_dict.setdefault(section, {})[record["modality"]] = torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32)
            )
        else:
            spatial_dict[section] = np.ascontiguousarray(array, dtype=np.float32)
    for record in manifest["metadata_artifacts"]:
        source = cache_dir / record["relative_path"]
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"Cached metadata checksum mismatch: {source}")
        shutil.copy2(source, output_dir / record["relative_path"])
    expected_modalities = {"RNA", "Protein", "HE"}
    for section in SECTIONS:
        if set(feature_dict.get(section, {})) != expected_modalities:
            raise ValueError(f"Cache modalities differ for {section}: {feature_dict.get(section, {})}")
        if section not in spatial_dict:
            raise ValueError(f"Cache lacks spatial coordinates for {section}.")
    cache_info = {
        "mode": "loaded",
        "path": str(cache_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    return feature_dict, spatial_dict, manifest["alignment"], cache_info


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
            metacell=False,
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


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if not np.isfinite(args.post_ot_graphsage_scale) or args.post_ot_graphsage_scale < 0:
        raise ValueError("--post_ot_graphsage_scale must be finite and non-negative.")
    if args.build_preprocessed_cache and args.preprocessed_cache_dir is None:
        raise ValueError("--build_preprocessed_cache requires --preprocessed_cache_dir.")
    if args.preprocessed_cache_dir is not None:
        args.preprocessed_cache_dir = args.preprocessed_cache_dir.resolve()
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
        if args.preprocessed_cache_dir is not None and not args.build_preprocessed_cache:
            record_memory(args.output_dir, "preprocessed_cache_load_start")
            feature_dict, spatial_dict, audits, cache_info = load_preprocessed_cache(
                args.preprocessed_cache_dir,
                args.data_dir.resolve(),
                args.output_dir,
                args,
            )
            processed = None
            record_memory(args.output_dir, "preprocessed_cache_load_end")
        else:
            record_memory(args.output_dir, "input_inspection_start")
            paths, common_genes, protein_markers, audits = inspect_full_inputs(
                args.data_dir.resolve(), args.output_dir
            )
            record_memory(args.output_dir, "input_inspection_end")
            feature_dict, spatial_dict, processed = preprocess_modalities_sequentially(
                paths,
                common_genes,
                protein_markers,
                args,
            )
            cache_info = {"mode": "disabled", "path": None}
            if args.build_preprocessed_cache:
                record_memory(args.output_dir, "preprocessed_cache_save_start")
                cache_manifest = save_preprocessed_cache(
                    args.preprocessed_cache_dir,
                    feature_dict,
                    spatial_dict,
                    paths,
                    audits,
                    args,
                )
                cache_info = {
                    "mode": "built",
                    "path": str(args.preprocessed_cache_dir),
                    "manifest": str(args.preprocessed_cache_dir / "manifest.json"),
                    "manifest_sha256": cache_manifest["manifest_sha256"],
                }
                record_memory(args.output_dir, "preprocessed_cache_save_end")
        section_order = list(SECTIONS)
        config = get_default_model_config()
        config["training"].update(
            {"device": "cuda", "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay}
        )
        config["loss"]["lambda_contrast"] = args.lambda_contrast
        config["ot_attention"]["context_gate_enabled"] = not bool(
            args.disable_context_attention_gate
        )
        config["uot"].update(
            {
                "max_iter": args.uot_max_iter,
                "topk": args.attention_topk,
                "update_interval": args.update_interval,
                "epsilon_update": args.uot_epsilon,
                "tau_a": args.uot_tau_a,
                "tau_b": args.uot_tau_b,
            }
        )
        config["graph"]["knn_neighbors_spatial"] = args.spatial_knn_k
        config["graphsage"]["edge_batch_size"] = args.graphsage_edge_batch_size
        config["graphsage"]["post_ot_graphsage_scale"] = float(
            args.post_ot_graphsage_scale
        )
        model = StageMultiModalModel(config=config, feature_dict=feature_dict)
        if list(model._resolve_modality_order(feature_dict["section1"])) != ["HE", "RNA", "Protein"]:
            raise ValueError("Unexpected modality order.")

        args.ot_prior_mode = "candidate_sparse"
        args.bidirectional_ot_attention = True
        args.training_loss_only = True
        args.checkpoint_ot_attention = True
        args.checkpoint_encoder_fusion = True
        args.checkpoint_decoder_chunks = True
        args.checkpoint_graph_encoder = True
        args.cache_spatial_graphs = True
        args.log_every = 1
        args.log_cuda_memory = True
        args.log_cuda_memory_detail = True
        args.save_candidate_qc = False
        monitor = CudaMemoryMonitor(True, args.output_dir, "cuda")
        initialize_model_ot_prior(model, feature_dict, section_order, args)
        history, outputs, ot_updates = train_small_crc_model(
            model, feature_dict, spatial_dict, processed, section_order, args, monitor
        )
        embedding_paths = save_final_embeddings(args.output_dir, outputs["final_embeddings"])
        summary = {
            "dataset": "spatch",
            "method": "spa_mo_model",
            "mode": "train",
            "input_data_path": str(args.data_dir.resolve()),
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
            "ot_prior_mode": args.ot_prior_mode,
            "bidirectional_ot_attention": args.bidirectional_ot_attention,
            "candidate_backend": args.candidate_backend,
            "initial_modality_candidate_k": args.initial_modality_candidate_k,
            "candidate_k": args.candidate_k,
            "attention_topk": args.attention_topk,
            "faiss_nlist": args.faiss_nlist,
            "faiss_nprobe": args.faiss_nprobe,
            "faiss_device": args.faiss_device,
            "faiss_train_sample_size": args.faiss_train_sample_size,
            "faiss_query_batch_size": args.faiss_query_batch_size,
            "dynamic_candidate_source": args.dynamic_candidate_source,
            "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
            "pre_post_graphsage_parameter_sharing": False,
            "ot_refresh_embedding_key": "ot_embeddings",
            "attention_context_gate_enabled": not args.disable_context_attention_gate,
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
