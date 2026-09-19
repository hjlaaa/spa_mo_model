"""Content identities and explicit ownership for new analysis outputs.

This does not migrate historical caches. A mismatch asks the caller to use a
new output directory; existing files are never removed or silently adopted.
Callers compute source identities once and pass them to individual fit caches.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

MANIFEST = "analysis_source_manifest.json"
SCHEMA = "analysis-source-v1"
_WRITABLE_RESULT_ROOT = ContextVar("writable_result_root", default=None)


@contextmanager
def writable_result_root(root=None):
    """Explicitly authorize one new experiment's result root for this call.

    The default protection and all input/output overlap checks remain active.
    No experiment/version name is encoded in analysis internals.
    """
    root = None if root is None else Path(root).resolve()
    if root is not None and not root.name.startswith("result_"):
        raise ValueError("The writable result root must name a result_* directory.")
    token = _WRITABLE_RESULT_ROOT.set(root)
    try:
        yield
    finally:
        _WRITABLE_RESULT_ROOT.reset(token)


class CacheMismatch(ValueError):
    """Existing output cannot be reused under the requested source contract."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    identity = {"path": str(path), "sha256": digest_file(path)}
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        identity.update(shape=list(array.shape), dtype=str(array.dtype))
    return identity


def array_identity(values) -> dict:
    array = np.asarray(values)
    if array.dtype.kind in "OU":
        # JSON preserves string lengths/order and distinguishes null from text.
        payload = pd.Series(array.reshape(-1)).to_json(orient="values", force_ascii=False).encode()
    else:
        payload = np.ascontiguousarray(array).tobytes()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(payload).hexdigest()}


def frame_identity(frame: pd.DataFrame) -> dict:
    return {"columns": list(frame.columns), "index": array_identity(frame.index.to_numpy()),
            "values": {str(column): array_identity(frame[column].to_numpy()) for column in frame.columns}}


def run_identity(summary: dict) -> dict:
    """Select training/source meaning, excluding logs, timings and output paths."""
    keys = ("dataset", "dataset_name", "section_order", "sections", "modalities",
            "model_version", "model_variant", "model_mode", "architecture", "seed",
            "epochs", "post_ot_graphsage_scale", "dynamic_candidate_source",
            "attention_context_gate_enabled", "harmony_used", "preprocessing",
            "model_config", "refresh_source", "execution")
    result = {key: summary[key] for key in keys if key in summary}
    resolved = summary.get("resolved_config")
    if isinstance(resolved, dict):
        result["resolved"] = {key: resolved[key] for key in keys if key in resolved}
        # P3b runners also expose their effective scalar CLI/runtime values.
        for name in ("runner", "training", "runtime", "data", "preprocess"):
            if isinstance(resolved.get(name), dict):
                result["resolved"][name] = {
                    key: value for key, value in resolved[name].items()
                    if key not in {"output_dir", "output_root", "log_file", "log_path", "overwrite", "verbose"}
                }
    result["resolved_available"] = isinstance(resolved, dict)
    return _without_run_logging(result)


def _without_run_logging(value):
    if isinstance(value, dict):
        return {key: _without_run_logging(item) for key, item in value.items()
                if key not in {"output_dir", "output_root", "verbose", "logs", "created_at", "updated_at"}
                and not key.startswith("log_")}
    if isinstance(value, list):
        return [_without_run_logging(item) for item in value]
    return value


def summary_identity(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_summary.json"
    if not path.is_file():
        return {"resolved_available": False, "limitation": "run summary unavailable"}
    return run_identity(json.loads(path.read_text(encoding="utf-8")))


def implementation_identity(version: str) -> dict:
    # Bump the entry's version when its numerical analysis implementation changes.
    # Training, plotting-only edits and unrelated repository files are not keys.
    return {"version": version, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__}


def data_identity(data, truth: dict) -> dict:
    """Bind the actual aligned values returned by existing dataset loaders."""
    paths = list(dict.fromkeys(Path(p).resolve() for p in data.source_paths))
    files = [file_identity(p) for p in paths if p.suffix != ".h5ad"]
    embedding_paths = [p for p in paths if "embedding" in p.name and p.suffix == ".npy"]
    run_dir = embedding_paths[0].parent if embedding_paths else Path(data.data_dir)
    if run_dir.name == "final_embeddings":
        run_dir = run_dir.parent
    return {
        "data_dir": str(Path(data.data_dir).resolve()),
        "files_in_loader_order": files,
        "embedding": array_identity(data.embedding),
        "section_order": list(dict.fromkeys(np.asarray(data.sections).astype(str))),
        "sections": array_identity(data.sections), "barcodes": array_identity(data.barcodes),
        "coords": array_identity(data.coords) if getattr(data, "coords", None) is not None else None,
        "truth": {name: array_identity(values) for name, values in truth.items()},
        "run": summary_identity(run_dir),
        "annotation_evidence": "actual aligned loader values; no invented historical barcode evidence",
        "raw_metadata_sources": [str(p) for p in paths if p.suffix == ".h5ad"],
    }


def check_output_path(output: Path, input_dirs=()) -> Path:
    output = Path(output).resolve()
    forbidden = {"results", "preprocessed_cache", "gene_imputation",
                 "gene_imputation_spatial_smoothing", "gene_imputation_shared_gene_validation"}
    allowed = _WRITABLE_RESULT_ROOT.get()
    result_roots = [p for p in (output, *output.parents) if p.name.startswith("result_")]
    if any(part in forbidden for part in output.parts) or any(p != allowed for p in result_roots):
        raise CacheMismatch(f"Historical/protected output is read-only: {output}. Choose a new independent output directory.")
    for directory in input_dirs:
        source = Path(directory).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise CacheMismatch(f"Analysis input and output must be separate: {source} / {output}")
    return output


def _json_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_manifest(path: Path, identity: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("identity") != json.loads(json.dumps(identity)):
        raise CacheMismatch(f"Analysis cache source/protocol mismatch: {path}. Use a new independent output directory.")
    return payload


def _check_artifacts(root: Path, payload: dict) -> None:
    for name, digest in payload.get("artifacts", {}).items():
        path = root / name
        if not path.is_file() or digest_file(path) != digest:
            raise CacheMismatch(f"Analysis cache artifact missing/changed: {path}. Use a new independent output directory.")


def begin_analysis(output: Path, identity: dict, *, input_dirs=()) -> bool:
    """Claim an empty new output, or validate an already owned output."""
    output = check_output_path(output, input_dirs)
    path = output / MANIFEST
    if path.is_file():
        payload = _read_manifest(path, identity)
        if payload.get("complete"):
            _check_artifacts(output, payload)
            return True
        return False
    if output.exists() and any(output.iterdir()):
        raise CacheMismatch(f"Provenance unavailable (historical/unverified): {output}. Use a new independent output directory; existing files are unchanged.")
    output.mkdir(parents=True, exist_ok=True)
    _json_write(path, {"schema": SCHEMA, "identity": identity, "complete": False})
    return False


def finish_analysis(output: Path, identity: dict) -> None:
    output = Path(output)
    _read_manifest(output / MANIFEST, identity)
    artifacts = {str(p.relative_to(output)): digest_file(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p != output / MANIFEST}
    _json_write(output / MANIFEST, {"schema": SCHEMA, "identity": identity,
                                  "complete": True, "artifacts": artifacts})


def cache_hit(manifest: Path, identity: dict, files: list[Path]) -> bool:
    """Fit-cache read boundary; caller supplies a once-computed source identity."""
    if not manifest.is_file():
        if any(Path(p).exists() for p in files):
            raise CacheMismatch(f"Cache provenance unavailable: {manifest}. Use a new independent output directory.")
        return False
    payload = _read_manifest(manifest, identity)
    if set(payload.get("artifacts", {})) != {p.name for p in files}:
        raise CacheMismatch(f"Cache artifact set mismatch: {manifest}")
    _check_artifacts(manifest.parent, payload)
    return True


def save_cache(manifest: Path, identity: dict, files: list[Path]) -> None:
    _json_write(manifest, {"schema": SCHEMA, "identity": identity,
                          "artifacts": {p.name: digest_file(p) for p in files}})
