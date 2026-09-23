"""Explicit SPATCH raw/reuse preparation with one strict schema-1 exit.

Reuse never prepares raw inputs or falls back; raw is selected by its own spec.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from .paired_preparation import PreparedDataset


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


def cache_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Historical schema-1 build fields; retain_processed is provenance at load time."""
    params = {
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
    if getattr(args, "spatial_enhancement", False):
        params["spatial_enhancement"] = {
            "enabled": True,
            "k": getattr(args, "spatial_enhancement_k", 10),
            "weight": getattr(args, "spatial_enhancement_weight", 0.2),
            "include_self": getattr(args, "spatial_enhancement_include_self", False),
        }
    return params


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
        if key == "spatial_enhancement" and key not in parameters:
            # An unenhanced schema-1 cache can be enhanced in memory after
            # strict validation; the stored arrays remain unchanged.
            continue
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


@dataclass
class SpatchCacheInput:
    """Explicit cache selection plus the existing loader options/output location."""

    cache_dir: Path
    data_dir: Path | None
    output_dir: Path
    args: argparse.Namespace


@dataclass
class SpatchRawInput:
    """Six aligned H5ADs, including existing HE features, and a new cache target."""

    data_dir: Path
    output_dir: Path
    cache_dir: Path
    args: argparse.Namespace
    observe: Callable[[Path, str], None] | None = None


def prepare_dataset(spec: SpatchCacheInput | SpatchRawInput) -> PreparedDataset:
    """Borrow validated model-ready payloads; retain schema-1 identity limits.

    Metadata copies remain part of the original loader, after all its checks.
    Metadata identity is represented by file/range references, without retaining
    the loader's full DataFrame or reading annotation columns it did not load.
    """
    if isinstance(spec, SpatchRawInput):
        from .spatch_raw import prepare_raw_dataset
        return prepare_raw_dataset(spec)
    if not isinstance(spec, SpatchCacheInput):
        raise TypeError(f"Unsupported SPATCH input spec: {type(spec).__name__}")
    features, spatial, alignment, cache_info = load_preprocessed_cache(
        spec.cache_dir, spec.data_dir, spec.output_dir, spec.args
    )
    if getattr(spec.args, "spatial_enhancement", False):
        settings = cache_parameters(spec.args)["spatial_enhancement"]
        if "spatial_enhancement" not in cache_info["historical_preprocessing_parameters"]:
            from .preprocessing import spatial_enhance_features
            spatial_enhance_features(
                features, spatial, k=settings["k"], weight=settings["weight"],
                include_self=settings["include_self"],
            )
            cache_info["runtime_spatial_enhancement"] = settings
    metadata_path = str(Path(cache_info["path"]) / "spot_metadata.csv.gz")
    metadata_output_path = str(spec.output_dir.resolve() / "spot_metadata.csv.gz")
    identity = {}
    truth = {}
    offset = 0
    for section in SECTIONS:
        count = alignment[section]["n_spots"]
        rows = slice(offset, offset + count)
        identity[section] = {
            "section": section,
            "row_count": count,
            "metadata_path": metadata_path,
            "metadata_output_path": metadata_output_path,
            "metadata_rows": rows,
            "verified_columns": ("block_id", "section", "x", "y"),
            "block_id_rule": "section:int(x):int(y)",
            "evidence": "row-order + external metadata identity",
            "feature_row_identity_limit": cache_info["feature_row_identity_limit"],
            "input_gene_file": str(Path(cache_info["path"]) / "shared_rna_genes.csv"),
            "input_marker_file": str(Path(cache_info["path"]) / "protein_markers_after_dapi_removal.csv"),
            "derived_feature_names": None,
        }
        truth[section] = {
            "status": "not_loaded",
            "metadata_path": metadata_path,
            "metadata_rows": rows,
            "note": "Only block_id/section/x/y are read; annotation availability is not inferred.",
        }
        offset += count
    return PreparedDataset(
        section_order=list(SECTIONS),
        feature_dict=features,
        spatial_loc_dict=spatial,
        processed_data_dict=None,
        identity=identity,
        truth=truth,
        provenance={
            "input_kind": "spatch_model_ready_cache",
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_info": cache_info,
        },
        preparation_audit=alignment,
        compatibility_context={
            "feature_dict": features,
            "spatial_dict": spatial,
            "alignment": alignment,
            "cache_info": cache_info,
        },
    )


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

