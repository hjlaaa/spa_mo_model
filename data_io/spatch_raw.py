"""SPATCH six aligned H5AD inputs to a newly published schema-1 cache.

The scientific preparation functions are mechanically migrated from run_spatch.
HE is an existing feature matrix; this adapter does not run image inference.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .preprocessing import load_cosie_style_data, spatial_enhance_features
from .spatch_preparation import (
    SECTIONS, FILES, CACHE_SCHEMA_VERSION, CACHE_METADATA_FILES,
    canonical_ids, sha256_file, cache_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _publish_new_cache(temporary: Path, destination: Path) -> None:
    """Linux atomic no-replace directory publication, including empty targets.

    Never fall back to replace/rename: those can overwrite an existing empty
    directory after the initial existence check. Leave staging data on failure.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def validate_raw_destination(cache_dir: Path, output_dir: Path) -> Path:
    """Reject existing caches and output aliases before any raw preparation."""
    if os.path.lexists(cache_dir):
        raise FileExistsError(f"Refusing to overwrite preprocessing cache: {cache_dir}")
    cache_dir = cache_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == cache_dir or cache_dir in output_dir.parents:
        raise ValueError("SPATCH cache is read-only; output_dir must be outside the cache.")
    return cache_dir


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



def save_preprocessed_cache(
    cache_dir: Path,
    feature_dict: dict[str, dict[str, torch.Tensor]],
    spatial_dict: dict[str, np.ndarray],
    paths: dict[str, dict[str, Path]],
    audits: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    if os.path.lexists(cache_dir):
        raise FileExistsError(f"Refusing to overwrite preprocessing cache: {cache_dir}")
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
            "data_io/spatch_raw.py": sha256_file(Path(__file__).resolve()),
            "data_io/spatch_preparation.py": sha256_file(
                PROJECT_ROOT / "data_io/spatch_preparation.py"
            ),
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
    _publish_new_cache(temporary, cache_dir)
    manifest["manifest_sha256"] = sha256_file(cache_dir / "manifest.json")
    return manifest



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
    *, observe=None,
):
    observe = observe or (lambda output_dir, stage: None)
    feature_dict: dict[str, dict[str, torch.Tensor]] = {}
    spatial_dict: dict[str, np.ndarray] = {}
    selectors = {
        "RNA": common_genes,
        "Protein": protein_markers,
        "HE": None,
    }
    hvg_counts = {"RNA": args.hvg_num, "Protein": None, "HE": None}
    for modality in ("RNA", "Protein", "HE"):
        observe(args.output_dir, f"{modality}_load_start")
        modality_inputs = [
            load_lightweight_modality(
                paths[section][modality],
                section,
                modality,
                selectors[modality],
            )
            for section in SECTIONS
        ]
        observe(args.output_dir, f"{modality}_loaded")
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
        observe(args.output_dir, f"{modality}_processed_and_released")
    if getattr(args, "spatial_enhancement", False):
        feature_dict = spatial_enhance_features(
            feature_dict, spatial_dict,
            k=args.spatial_enhancement_k,
            weight=args.spatial_enhancement_weight,
            include_self=args.spatial_enhancement_include_self,
        )
        observe(args.output_dir, "spatial_enhancement_complete")
    return feature_dict, spatial_dict, None



def rename_sections(mapping):
    return {("section1" if key == "s1" else "section2" if key == "s2" else key): value for key, value in mapping.items()}



def prepare_raw_dataset(spec):
    """Prepare six aligned H5AD inputs, publish once, then use strict reuse.

    The caller owns seeding, run output/failure handling and memory observation.
    This path never selects an existing cache or runs UNI image inference.
    """
    from .spatch_preparation import SpatchCacheInput, prepare_dataset

    cache_dir = validate_raw_destination(spec.cache_dir, spec.output_dir)
    observe = spec.observe or (lambda output_dir, stage: None)
    observe(spec.output_dir, "input_inspection_start")
    paths, common_genes, protein_markers, audits = inspect_full_inputs(
        spec.data_dir.resolve(), spec.output_dir
    )
    observe(spec.output_dir, "input_inspection_end")
    features, spatial, processed = preprocess_modalities_sequentially(
        paths, common_genes, protein_markers, spec.args, observe=observe
    )
    manifest = save_preprocessed_cache(
        cache_dir, features, spatial, paths, audits, spec.args
    )
    # No preparation payload is held across the strict cache reload.
    del features, spatial, processed
    gc.collect()
    prepared = prepare_dataset(SpatchCacheInput(
        cache_dir, spec.data_dir, spec.output_dir, spec.args
    ))
    origin = {
        "input_kind": "spatch_aligned_h5ad",
        "data_dir": str(spec.data_dir.resolve()),
        "seed": spec.args.seed,
        "source_files": manifest["source_files"],
        "generated_manifest": str(cache_dir / "manifest.json"),
        "image_inference": False,
    }
    prepared.provenance["input_kind"] = "spatch_raw_h5ad"
    prepared.provenance["raw_preparation"] = origin
    prepared.compatibility_context["cache_info"]["raw_preparation"] = origin
    for evidence in prepared.truth.values():
        evidence["note"] += " Raw inspection copied existing annotations into metadata; reload does not read them."
    return prepared
