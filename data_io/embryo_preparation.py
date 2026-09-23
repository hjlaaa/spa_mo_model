"""HESTА source preparation and model-ready handoff, without new metadata reads.

The source phase remains separate so preprocess-only need not reload raw outputs.
HESTА algorithms and on-disk schemas remain authoritative in :mod:`hesta`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hesta import load_preprocessed_manifest, preprocess_hesta_rna
from .paired_preparation import PreparedDataset


@dataclass
class ManifestInput:
    cache_dir: Path


@dataclass
class ExternalRunInput:
    run_dir: Path
    maximum_per_section: int | None = None


@dataclass
class RawHestaInput:
    paths: list[Path]
    cache_dir: Path
    preprocessing: dict[str, Any]


@dataclass
class HestaSource:
    """Transitional source state; no model arrays are fabricated for early exits."""

    spec: Any
    manifest: dict[str, Any]
    feature_dict: dict | None = None
    spatial_dict: dict | None = None
    external_source_summary: dict | None = None


def prepare_source(spec: ManifestInput | ExternalRunInput | RawHestaInput) -> HestaSource:
    """Preserve the original source-phase reads, before require_harmony/early exit."""
    if isinstance(spec, ManifestInput):
        manifest, _, spatial = load_preprocessed_manifest(spec.cache_dir)
        # The old runner's repeated '_' assignment retained this spatial dict
        # through its early exit or model run. Keep that mmap lifetime as well.
        return HestaSource(spec, manifest, spatial_dict=spatial)
    if isinstance(spec, ExternalRunInput):
        manifest, features, spatial, summary = load_external_preprocessed_run(
            spec.run_dir, spec.maximum_per_section
        )
        return HestaSource(spec, manifest, features, spatial, summary)
    if isinstance(spec, RawHestaInput):
        manifest = preprocess_hesta_rna(spec.paths, spec.cache_dir, **spec.preprocessing)
        return HestaSource(spec, manifest)
    raise TypeError(f"Unsupported HESTA input specification: {type(spec).__name__}")


def prepare_dataset(source: HestaSource) -> PreparedDataset:
    """Load the original model inputs and borrow arrays; do not read annotations."""
    spec = source.spec
    if isinstance(spec, ExternalRunInput):
        manifest = source.manifest
        features, spatial = source.feature_dict, source.spatial_dict
        selection = slice(None, spec.maximum_per_section)
        provenance = {
            "input_kind": "external_preprocessed_run",
            "source_summary_path": str(spec.run_dir / "run_summary.json"),
            "preprocess_manifest": manifest,
        }
    else:
        manifest, features, spatial = load_preprocessed_manifest(spec.cache_dir)
        selection = slice(None)
        provenance = {
            "input_kind": "raw_hesta" if isinstance(spec, RawHestaInput) else "existing_hesta_manifest",
            "manifest_path": str(spec.cache_dir / "preprocess_manifest.json"),
            "preprocess_manifest": manifest,
        }
        if isinstance(spec, RawHestaInput):
            provenance["raw_input_paths"] = spec.paths
    section_order = list(manifest["section_order"])
    identity = {}
    truth = {}
    annotations = manifest.get("annotation_files")
    source_indices = manifest.get("selected_index_files")
    for section in section_order:
        # These optional fields were not validated/read by the old loader.
        annotation = annotations.get(section) if isinstance(annotations, dict) else None
        source_rows = source_indices.get(section) if isinstance(source_indices, dict) else None
        identity[section] = {
            "section": section,
            "row_count": len(features[section]["RNA"]),
            "row_selection": selection,
            "source_row_file": source_rows,
            "annotation_file": annotation,
            "metadata_status": "reference_only" if annotation or source_rows else "not_provided",
            "evidence": "manifest references, contents not loaded" if annotation or source_rows else "row-order evidence only",
            "derived_feature_names": None,
        }
        truth[section] = {
            "status": "not_loaded" if annotation else "not_provided",
            "annotation_file": annotation,
            "row_selection": selection,
            "note": "Annotation columns/values are not inspected; section is not biological stage or celltype.",
        }
    return PreparedDataset(
        section_order=section_order,
        feature_dict=features,
        spatial_loc_dict=spatial,
        processed_data_dict=None,
        identity=identity,
        truth=truth,
        provenance=provenance,
        preparation_audit={
            "harmony_used": manifest.get("harmony_used"),
            "qc": manifest.get("qc"),
            "feature_method": manifest.get("feature_method"),
            "selected_index_files": manifest.get("selected_index_files"),
            "annotation_files": manifest.get("annotation_files"),
        },
        compatibility_context={
            "preprocess_manifest": manifest,
            "external_source_summary": source.external_source_summary,
            "feature_dict": features,
            "spatial_dict": spatial,
        },
    )


def load_external_preprocessed_run(
    run_dir: Path, maximum_per_section: int | None
) -> tuple[dict[str, Any], dict, dict, dict[str, Any]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"External preprocessing summary is missing: {summary_path}")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "PASS":
        raise ValueError("External preprocessing run is not successful.")
    manifest = source_summary["preprocess_manifest"]
    feature_dict = {}
    spatial_dict = {}
    for section in manifest["section_order"]:
        features = np.load(manifest["feature_files"][section], mmap_mode="c")
        spatial = np.load(manifest["spatial_files"][section], mmap_mode="r")
        if len(features) != len(spatial):
            raise ValueError(f"{section}: feature/spatial counts differ in external cache.")
        if maximum_per_section is not None:
            if maximum_per_section <= 0 or maximum_per_section > len(features):
                raise ValueError(
                    f"Invalid --max_spots_per_section={maximum_per_section} for {section}."
                )
            features = np.array(features[:maximum_per_section], dtype=np.float32, copy=True)
            spatial = np.array(spatial[:maximum_per_section], copy=True)
        feature_dict[section] = {"RNA": features}
        spatial_dict[section] = spatial
    return manifest, feature_dict, spatial_dict, source_summary
