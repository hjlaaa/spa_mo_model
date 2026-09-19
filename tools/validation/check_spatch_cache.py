#!/usr/bin/env python3
"""Small read-only schema=1 cache checks with explicit reference and fresh output directories."""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

spatch = importlib.import_module("scripts.run_spatch")
from scripts import run_experiments as batch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


_guard = None


def audit(event, args):
    if _guard is None:
        return
    cache, events, raw_root, raw_reads = _guard
    write = False
    paths = []
    if event == "open":
        path, mode, flags = args
        write = bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        paths = [path]
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).absolute()
            if resolved == raw_root or raw_root in resolved.parents or resolved.suffix == ".h5ad":
                raw_reads.append(str(resolved))
                raise AssertionError(f"Raw data opened during cache consumption: {resolved}")
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.chmod", "os.utime", "os.truncate", "os.link", "os.symlink"}:
        write = True
        paths = list(args[:2]) if event in {"os.rename", "os.link", "os.symlink"} else list(args[:1])
    if write:
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).absolute()
                if resolved == cache or cache in resolved.parents:
                    events.append({"event": event, "path": str(resolved)})
                    raise AssertionError(f"Forbidden cache write: {event}: {resolved}")


sys.addaudithook(audit)


def fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    cache, raw, output = (root / name for name in ("cache", "raw", "output"))
    cache.mkdir()
    output.mkdir()
    args = SimpleNamespace(n_comps=50, hvg_num=3000, no_harmony=False, data_dir=raw, output_dir=output)
    params = spatch.cache_parameters(args)
    records, source_files, metadata, alignment = [], [], [], {}
    values = {}
    for si, (section, n) in enumerate(zip(spatch.SECTIONS, (3, 5))):
        coords = np.array([[si * 100 + i * 3, i * 7] for i in range(n)], dtype=np.float32)
        values[section] = {}
        for mi, (modality, dims) in enumerate((("RNA", 50), ("Protein", 15), ("HE", 50))):
            value = (np.arange(n * dims, dtype=np.float32).reshape(n, dims) + 1 + 1000 * si + 100 * mi) / 19
            values[section][modality] = value
            path = cache / f"{section}_{modality}.npy"
            np.save(path, value, allow_pickle=False)
            records.append({"kind": "feature", "section": section, "modality": modality, "relative_path": path.name, "shape": list(value.shape), "dtype": str(value.dtype), "size_bytes": path.stat().st_size, "sha256": sha(path)})
            source = raw / section / spatch.FILES[section][mi]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"synthetic-{section}-{modality}".encode())
            source_files.append({"section": section, "modality": modality, "path": str(source), "size_bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns, "sha256": sha(source)})
        path = cache / f"{section}_spatial.npy"
        np.save(path, coords, allow_pickle=False)
        values[section]["spatial"] = coords
        records.append({"kind": "spatial", "section": section, "modality": None, "relative_path": path.name, "shape": list(coords.shape), "dtype": str(coords.dtype), "size_bytes": path.stat().st_size, "sha256": sha(path)})
        metadata.extend({"block_id": f"{section}:{int(x)}:{int(y)}", "section": section, "original_rna_obs_name": f"spot_{i}", "x": x, "y": y, "cell_type_common": "synthetic"} for i, (x, y) in enumerate(coords))
        alignment[section] = {"n_spots": n, "rna_shape": [n, 4828], "protein_shape_before": [n, 17], "protein_shape_after_dapi_removal": [n, 16], "he_shape": [n, 2048], "spatial_alignment": True, "canonical_ids_unique": True, "dapi_removed": True}
    pd.DataFrame(metadata).to_csv(cache / "spot_metadata.csv.gz", index=False, compression="gzip")
    pd.Series([f"gene_{i}" for i in range(4828)], name="gene").to_csv(cache / "shared_rna_genes.csv", index=False)
    pd.Series([f"marker_{i}" for i in range(16)], name="protein_marker").to_csv(cache / "protein_markers_after_dapi_removal.csv", index=False)
    artifacts = [{"relative_path": name, "size_bytes": (cache / name).stat().st_size, "sha256": sha(cache / name)} for name in spatch.CACHE_METADATA_FILES]
    manifest = {"schema_version": 1, "dataset": "spatch", "cache_boundary": "model_ready_feature_dict_after_PCA_Harmony", "parameters": params, "arrays": records, "metadata_artifacts": artifacts, "source_files": source_files, "alignment": alignment, "software_versions": {"python": "synthetic fixture"}, "preprocessing_sources": {"run_spatch.py": sha(REPO / "scripts/run_spatch.py"), "model/data_preprocessing.py": sha(REPO / "data_io/preprocessing.py")}, "created_at": 0}
    write_json(cache / "manifest.json", manifest)
    return SimpleNamespace(root=root, cache=cache, raw=raw, output=output, args=args, manifest=manifest, values=values)


def persist(f):
    write_json(f.cache / "manifest.json", f.manifest)


def rewrite_metadata(f, transform):
    path = f.cache / "spot_metadata.csv.gz"
    frame = transform(pd.read_csv(path))
    frame.to_csv(path, index=False, compression="gzip")
    record = next(r for r in f.manifest["metadata_artifacts"] if r["relative_path"] == path.name)
    record.update(size_bytes=path.stat().st_size, sha256=sha(path))
    persist(f)


def rewrite_artifact(f, name, transform):
    path = f.cache / name
    transform(pd.read_csv(path)).to_csv(path, index=False)
    record = next(r for r in f.manifest["metadata_artifacts"] if r["relative_path"] == name)
    record.update(size_bytes=path.stat().st_size, sha256=sha(path))
    persist(f)


@contextlib.contextmanager
def guarded(f):
    global _guard
    events, calls, raw_reads = [], {}, []
    original = snapshot(f.cache)
    names = ("inspect_full_inputs", "preprocess_modalities_sequentially", "load_cosie_style_data", "save_preprocessed_cache")
    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            raise AssertionError(f"Forbidden preprocessing/build call: {name}")
        return fail
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(patch.object(spatch, name, forbidden(name)))
        _guard = (f.cache.absolute(), events, (f.root / "raw").absolute(), raw_reads)
        try:
            yield calls, events
        finally:
            _guard = None
            assert snapshot(f.cache) == original, "Cache content changed during call"
            assert not events, f"Cache write events: {events}"
            assert not calls, f"Preprocess/build events: {calls}"
            assert not raw_reads, f"Raw content reads: {raw_reads}"


def load_case(f):
    result, error = None, None
    with guarded(f) as (calls, events):
        try:
            result = spatch.load_preprocessed_cache(f.cache, f.raw, f.output, f.args)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    if result is not None:
        features, spatial, alignment, cache_info = result
        assert list(features) == list(spatch.SECTIONS)
        for section in spatch.SECTIONS:
            assert list(features[section]) == ["RNA", "Protein", "HE"]
            for modality in ("RNA", "Protein", "HE"):
                assert features[section][modality].dtype == torch.float32
                assert np.array_equal(features[section][modality].numpy(), f.values[section][modality])
            assert np.array_equal(spatial[section], f.values[section]["spatial"])
        assert cache_info["mode"] == "loaded"
        if "input_identity" in cache_info:
            info = cache_info["input_identity"]
            assert info["current_raw_files_verified"] is False
            assert info["historical_source_files"] == f.manifest["source_files"]
            assert info["requested_data_dir"] == (str(f.raw.resolve()) if f.raw is not None else None)
            assert info["input_identity_manifest"] == (str(f.args.input_identity_manifest.resolve()) if getattr(f.args, "input_identity_manifest", None) is not None else None)
            assert cache_info["historical_preprocessing_sources"] == f.manifest["preprocessing_sources"]
            assert cache_info["historical_preprocessing_parameters"] == f.manifest["parameters"]
            assert "barcode" in cache_info["feature_row_identity_limit"].lower()
            assert "do not independently" in cache_info["feature_row_identity_limit"].lower()
            assert cache_info["path"] == str(f.cache.resolve())
            assert cache_info["manifest_sha256"] == sha(f.cache / "manifest.json")
    return {"accepted": result is not None, "error": error, "preprocess_build_calls": calls, "cache_write_events": events, "cache_unchanged": True, "metadata_output_files": sorted(p.name for p in f.output.iterdir())}




def array_change(f, section, modality, transform):
    record = next(r for r in f.manifest["arrays"] if r["section"] == section and r["modality"] == modality)
    path = f.cache / record["relative_path"]
    value = transform(np.load(path, allow_pickle=False))
    np.save(path, value, allow_pickle=False)
    record.update(shape=list(value.shape), dtype=str(value.dtype), size_bytes=path.stat().st_size, sha256=sha(path))
    f.values[section][modality if modality is not None else "spatial"] = value
    persist(f)


def identity(f, *, wrong_digest=False, wrong_size=False, missing=False):
    relocated = f.root / "relocated_raw"
    records = copy.deepcopy(f.manifest["source_files"])
    for record in records:
        record["path"] = str(relocated / record["section"] / Path(record["path"]).name)
    if wrong_digest:
        records[0]["sha256"] = "f" * 64
    if wrong_size:
        records[0]["size_bytes"] += 1
    if missing:
        records.pop()
    path = f.root / "input_identity.json"
    write_json(path, {"source_files": records})
    f.args.input_identity_manifest = path
    f.raw = relocated


class ReachedModel(BaseException):
    pass


def main_case(f, extra, expected_model):
    called = {"model": 0, "require_gpu": 0, "training": 0, "save_embeddings": 0, "prior": 0}
    def model(*args, **kwargs):
        called["model"] += 1
        features = kwargs["feature_dict"]
        for section in spatch.SECTIONS:
            for modality in ("RNA", "Protein", "HE"):
                assert np.array_equal(features[section][modality].numpy(), f.values[section][modality])
        raise ReachedModel()
    def gpu(*args, **kwargs):
        called["require_gpu"] += 1
        return {"name": "synthetic-gpu-spy"}
    def forbidden(name):
        def fail(*args, **kwargs):
            called[name] += 1
            raise AssertionError(f"Training stage executed: {name}")
        return fail
    error = None
    reached = False
    argv = ["run_spatch.py", "--output_dir", str(f.output)] + extra
    with guarded(f) as (calls, events), contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(spatch, "StageMultiModalModel", model))
        stack.enter_context(patch.object(spatch, "require_gpu", gpu))
        stack.enter_context(patch.object(spatch, "record_memory", lambda *a, **k: None))
        stack.enter_context(patch.object(torch.cuda, "is_available", lambda: False))
        for name, count_name in (("train_small_crc_model", "training"), ("save_final_embeddings", "save_embeddings"), ("initialize_model_ot_prior", "prior")):
            stack.enter_context(patch.object(spatch, name, forbidden(count_name)))
        try:
            spatch.main()
        except ReachedModel:
            reached = True
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    assert reached is expected_model, {"reached_model": reached, "expected": expected_model, "error": error}
    assert called["training"] == called["save_embeddings"] == called["prior"] == 0
    if not expected_model:
        assert error and error["type"] in {"ValueError", "FileNotFoundError"}, error
        assert called["model"] == 0, called
    return {"reached_model": reached, "error": error, "calls": called, "preprocess_build_calls": calls, "cache_write_events": events, "cache_unchanged": True}


def check_summary(path, cache_dir, reference):
    version = reference["label"]
    scale = 0.5
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mode": "train",
        "epochs": 200,
        "seed": 42,
        "architecture": reference["architecture"],
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "dynamic_candidate_source": "ot",
        "ot_updates": reference["updates"],
        "post_ot_graphsage_scale": scale,
        "use_harmony": True,
        "spatial_knn_k": 10,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{version} SPATCH training settings differ: {mismatches}")
    cache = summary.get("preprocessing_cache", {})
    valid_modes = {"loaded"}
    if cache.get("mode") not in valid_modes or Path(cache.get("path", "")) != cache_dir:
        raise ValueError(f"{version} cache use differs: {cache}")
    if not (cache_dir / "manifest.json").is_file():
        raise FileNotFoundError(cache_dir / "manifest.json")
    return {
        "summary": str(path),
        "validated": expected,
        "cache": cache,
    }


def check_command(f, output_root, reference):
    # Frozen command is test input, not a second source of dataset defaults.
    tokens = list(reference["command"][2:])
    tokens[tokens.index("--data_dir") + 1] = str(f.raw)
    tokens[tokens.index("--preprocessed_cache_dir") + 1] = str(f.cache)
    i = tokens.index("--output_dir")
    del tokens[i:i + 2]
    args = batch.parse_args(["--datasets", "spatch", "--output-root", str(output_root),
                            "--runner-args", "spatch", json.dumps(tokens)])
    command = batch.build_tasks(args)[0]["command"]
    resolved = spatch.parse_args(command[2:])
    # Exercise the authority loader, guarded against raw I/O, build and cache writes.
    spatch.load_preprocessed_cache(f.cache, f.raw, f.output, resolved)
    return command


def suite_summary(f, mode, reference):
    return {"mode": "train", "epochs": 200, "seed": 42, "architecture": reference["architecture"], "pre_post_graphsage_parameter_sharing": False, "ot_refresh_embedding_key": "ot_embeddings", "dynamic_candidate_source": "ot", "attention_context_gate_enabled": False, "ot_updates": reference["updates"], "post_ot_graphsage_scale": 0.5, "use_harmony": True, "spatial_knn_k": 10, "preprocessing_cache": {"mode": mode, "path": str(f.cache)}}


def check_new(output_dir, reference_dir, report_name):
    HERE = Path(output_dir).resolve()
    reference_dir = Path(reference_dir).resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    reference = json.loads((reference_dir / "spatch_command.json").read_text())
    target = HERE / report_name
    if target.parent != HERE or target.suffix != ".json":
        raise ValueError("Report must be a new JSON filename within this verification directory")
    if target.exists():
        raise FileExistsError(f"Report already exists; choose a new name: {target}")
    old_path = reference_dir / "old_behavior.json"
    old_sha = sha(old_path)
    old = json.loads(old_path.read_text())
    production_names = ("scripts/run_spatch.py", "data_io/preprocessing.py")
    production_before = {name: sha(REPO / name) for name in production_names}
    cases = {}
    case_root = HERE / target.stem

    def fresh(name):
        f = fixture(case_root / name)
        # An unchanged historical source fingerprint must remain consumable.
        f.manifest["preprocessing_sources"] = {"run_spatch.py": old["production_sources"]["scripts/run_spatch.py"], "model/data_preprocessing.py": old["production_sources"]["model/data_preprocessing.py"]}
        f.args.input_identity_manifest = None
        persist(f)
        return f

    def check(name, accepted, mutate=None):
        try:
            f = fresh(name)
            if mutate:
                mutate(f)
            observed = load_case(f)
            assert observed["accepted"] is accepted, observed
            if accepted:
                assert observed["metadata_output_files"] == sorted(spatch.CACHE_METADATA_FILES), observed
            else:
                assert observed["error"] and observed["error"]["type"] in {"ValueError", "FileNotFoundError"}, observed
                assert not observed["metadata_output_files"], observed
            cases[name] = {"status": "PASS", "expected_accepted": accepted, **observed}
        except Exception as exc:
            cases[name] = {"status": "FAIL", "expected_accepted": accepted, "test_error": {"type": type(exc).__name__, "message": str(exc)}}

    def change_manifest(f, fn):
        fn(f.manifest)
        persist(f)

    check("historical_schema_normal", True)
    def frozen_old_cache(f):
        f.cache = reference_dir / "old_cases" / "normal" / "cache"
        f.raw = reference_dir / "old_cases" / "normal" / "raw"
        f.manifest = json.loads((f.cache / "manifest.json").read_text())
    check("frozen_old_cache_direct_replay", True, frozen_old_cache)
    check("default_cache_source_no_data_dir", True, lambda f: setattr(f, "raw", None))
    check("unrelated_training_source_hash", True, lambda f: change_manifest(f, lambda m: m["preprocessing_sources"].update({"run_spatch.py": "0" * 64})))
    check("preprocessing_source_file_reorganization", True, lambda f: change_manifest(f, lambda m: m["preprocessing_sources"].update({"model/data_preprocessing.py": "1" * 64})))
    check("unrelated_training_arguments", True, lambda f: vars(f.args).update(epochs=3, lr=0.125, amp_dtype="none", decoder_chunk_size=7, ot_attention_source_chunk_size=9, checkpoint_decoder_chunks=False))
    check("retain_processed_provenance_only", True, lambda f: change_manifest(f, lambda m: m["parameters"].update(retain_processed=True)))
    def touch_raw(f):
        path = Path(f.manifest["source_files"][0]["path"])
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    check("raw_mtime_only", True, touch_raw)
    def missing_raw(f):
        for record in f.manifest["source_files"]:
            Path(record["path"]).unlink()
    check("historical_raw_files_unavailable", True, missing_raw)
    check("relocated_raw_same_identity", True, lambda f: identity(f))
    check("relocated_raw_conflicting_digest", False, lambda f: identity(f, wrong_digest=True))
    check("relocated_raw_conflicting_size", False, lambda f: identity(f, wrong_size=True))
    check("relocated_raw_incomplete_identity", False, lambda f: identity(f, missing=True))
    check("explicit_other_raw_without_evidence", False, lambda f: setattr(f, "raw", f.root / "other_raw"))
    check("cache_source_record_missing", False, lambda f: change_manifest(f, lambda m: m["source_files"].pop()))
    check("cache_source_record_duplicate", False, lambda f: change_manifest(f, lambda m: m["source_files"].append(copy.deepcopy(m["source_files"][0]))))
    def wrong_identity_dataset(f):
        identity(f)
        path = f.args.input_identity_manifest
        value = json.loads(path.read_text())
        value["dataset"] = "other_dataset"
        write_json(path, value)
    check("identity_manifest_wrong_dataset", False, wrong_identity_dataset)
    check("unknown_preprocessing_parameter", False, lambda f: change_manifest(f, lambda m: m["parameters"].update(unknown_preprocessing_algorithm=True)))
    for key, value in (("schema_version", 2), ("dataset", "other"), ("cache_boundary", "raw_counts")):
        check(f"wrong_{key}", False, lambda f, key=key, value=value: change_manifest(f, lambda m: m.update({key: value})))
    for key, value in (("n_comps", 40), ("hvg_num", 1000), ("use_harmony", False), ("metacell", True), ("dapi_removed", False), ("memory_efficient", False), ("target_sum", 10000), ("hvg_num_by_modality", {"RNA": 3000, "Protein": 100, "HE": None})):
        check(f"incompatible_parameter_{key}", False, lambda f, key=key, value=value: change_manifest(f, lambda m: m["parameters"].update({key: value})))
    check("section_order_changed", False, lambda f: change_manifest(f, lambda m: m["parameters"].update(sections=["section2", "section1"])))
    check("modality_order_changed", False, lambda f: change_manifest(f, lambda m: m["parameters"].update(modalities=["HE", "RNA", "Protein"])))
    check("missing_modality_record", False, lambda f: change_manifest(f, lambda m: m["arrays"].pop(1)))
    check("duplicate_array_record", False, lambda f: change_manifest(f, lambda m: m["arrays"].append(copy.deepcopy(m["arrays"][0]))))
    check("array_section_order_changed", False, lambda f: change_manifest(f, lambda m: m.update(arrays=m["arrays"][4:] + m["arrays"][:4])))
    check("unsupported_array_kind", False, lambda f: change_manifest(f, lambda m: m["arrays"][3].update(kind="other")))
    check("extra_array_section", False, lambda f: change_manifest(f, lambda m: m["arrays"][0].update(section="section3")))
    check("protein_wrong_dimension_50", False, lambda f: array_change(f, "section1", "Protein", lambda a: np.zeros((len(a), 50), dtype=np.float32)))
    check("feature_spot_count_changed", False, lambda f: array_change(f, "section1", "RNA", lambda a: a[:-1].copy()))
    check("spatial_spot_count_changed", False, lambda f: array_change(f, "section2", None, lambda a: a[:-1].copy()))
    check("spatial_wrong_width", False, lambda f: array_change(f, "section2", None, lambda a: a[:, :1].copy()))
    check("array_nan", False, lambda f: array_change(f, "section1", "RNA", lambda a: np.full_like(a, np.nan)))
    check("array_inf", False, lambda f: array_change(f, "section1", "RNA", lambda a: np.full_like(a, np.inf)))
    check("array_dtype_float64", False, lambda f: array_change(f, "section1", "RNA", lambda a: a.astype(np.float64)))
    check("array_checksum_wrong", False, lambda f: change_manifest(f, lambda m: m["arrays"][0].update(sha256="0" * 64)))
    check("array_file_size_wrong", False, lambda f: change_manifest(f, lambda m: m["arrays"][0].update(size_bytes=1)))
    check("array_shape_record_wrong", False, lambda f: change_manifest(f, lambda m: m["arrays"][0].update(shape=[3, 49])))
    check("array_file_missing", False, lambda f: (f.cache / f.manifest["arrays"][0]["relative_path"]).unlink())
    check("metadata_row_order_changed", False, lambda f: rewrite_metadata(f, lambda frame: frame.iloc[[1, 0, *range(2, len(frame))]]))
    check("metadata_row_count_changed", False, lambda f: rewrite_metadata(f, lambda frame: frame.iloc[:-1]))
    check("metadata_section_changed", False, lambda f: rewrite_metadata(f, lambda frame: frame.assign(section="section3")))
    check("metadata_coordinates_changed", False, lambda f: rewrite_metadata(f, lambda frame: frame.assign(x=frame.x + 1)))
    check("metadata_id_duplicate", False, lambda f: rewrite_metadata(f, lambda frame: frame.assign(block_id="duplicate")))
    check("metadata_required_column_missing", False, lambda f: rewrite_metadata(f, lambda frame: frame.drop(columns=["block_id"])))
    check("metadata_checksum_wrong", False, lambda f: change_manifest(f, lambda m: m["metadata_artifacts"][0].update(sha256="0" * 64)))
    check("shared_gene_duplicate", False, lambda f: rewrite_artifact(f, "shared_rna_genes.csv", lambda frame: frame.assign(gene="duplicate")))
    check("protein_marker_dapi", False, lambda f: rewrite_artifact(f, "protein_markers_after_dapi_removal.csv", lambda frame: pd.DataFrame({"protein_marker": ["DAPI", *frame.protein_marker.tolist()[1:]]})))
    check("protein_marker_wrong_count", False, lambda f: rewrite_artifact(f, "protein_markers_after_dapi_removal.csv", lambda frame: frame.iloc[:-1]))
    check("alignment_n_conflict", False, lambda f: change_manifest(f, lambda m: m["alignment"]["section1"].update(n_spots=4)))
    check("alignment_false", False, lambda f: change_manifest(f, lambda m: m["alignment"]["section1"].update(spatial_alignment=False)))
    check("missing_manifest", False, lambda f: (f.cache / "manifest.json").unlink())

    for name, child in (("loader_output_equals_cache", False), ("loader_output_inside_cache", True)):
        try:
            f = fresh(name)
            output = f.cache / "new_output" if child else f.cache
            error = None
            with guarded(f) as (calls, events):
                try:
                    spatch.load_preprocessed_cache(f.cache, f.raw, output, f.args)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
            assert error and error["type"] == "ValueError", error
            cases[name] = {"status": "PASS", "error": error, "preprocess_build_calls": calls, "cache_write_events": events, "cache_unchanged": True}
        except Exception as exc:
            cases[name] = {"status": "FAIL", "test_error": {"type": type(exc).__name__, "message": str(exc)}}

    for name, make_extra, expected_model in (
        ("main_normal_stops_before_training", lambda f: ["--preprocessed_cache_dir", str(f.cache)], True),
        ("main_explicit_identity_stops_before_training", lambda f: ["--preprocessed_cache_dir", str(f.cache), "--data_dir", str(f.raw), "--input_identity_manifest", str(f.args.input_identity_manifest)], True),
        ("main_no_cache", lambda f: [], False),
        ("main_missing_cache", lambda f: ["--preprocessed_cache_dir", str(f.root / "absent_cache")], False),
        ("main_build_rejected", lambda f: ["--preprocessed_cache_dir", str(f.cache), "--build_preprocessed_cache"], False),
        ("main_other_raw_without_identity", lambda f: ["--preprocessed_cache_dir", str(f.cache), "--data_dir", str(f.root / "other_raw")], False),
    ):
        try:
            f = fresh(name)
            if name == "main_explicit_identity_stops_before_training":
                identity(f)
            result = main_case(f, make_extra(f), expected_model)
            cases[name] = {"status": "PASS", **result}
        except Exception as exc:
            cases[name] = {"status": "FAIL", "test_error": {"type": type(exc).__name__, "message": str(exc)}}

    for name, missing, mode, accepted in (("suite_command_existing_cache", False, None, True), ("suite_command_missing_cache", True, None, False), ("suite_validate_loaded", False, "loaded", True), ("suite_validate_built", False, "built", False), ("suite_validate_missing_manifest", True, "loaded", False)):
        try:
            f = fresh(name)
            test_root = f.root / "suite_root"
            if mode:
                summary_path = test_root / "spatch" / "run_summary.json"
                summary_path.parent.mkdir(parents=True)
                write_json(summary_path, suite_summary(f, mode, reference))
            if missing:
                (f.cache / "manifest.json").unlink()
            result, error = None, None
            with guarded(f) as (calls, events):
                try:
                    result = check_summary(summary_path, f.cache, reference) if mode else check_command(f, test_root, reference)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
            assert (error is None) is accepted, {"result": result, "error": error}
            if accepted and mode is None:
                assert "--build_preprocessed_cache" not in result
                for key, value in (("--epochs", "200"), ("--n_comps", "50"), ("--hvg_num", "3000"), ("--amp_dtype", "bf16"), ("--decoder_chunk_size", "8192"), ("--ot_attention_source_chunk_size", "2048"), ("--post_ot_graphsage_scale", "0.5")):
                    assert result[result.index(key) + 1] == value
            if not accepted:
                assert error["type"] in {"ValueError", "FileNotFoundError"}
            cases[name] = {"status": "PASS", "accepted": accepted, "error": error, "result": result, "preprocess_build_calls": calls, "cache_write_events": events, "cache_unchanged": True}
        except Exception as exc:
            cases[name] = {"status": "FAIL", "test_error": {"type": type(exc).__name__, "message": str(exc)}}

    assert sha(old_path) == old_sha, "Frozen old report changed"
    assert {name: sha(REPO / name) for name in production_names} == production_before, "Production source changed during checks"
    failed = [name for name, case in cases.items() if case["status"] != "PASS"]
    report = {"phase": "NEW_CONTRACT", "status": "FAIL" if failed else "PASS", "old_report_sha256": old_sha, "production_sources": {name: sha(REPO / name) for name in ("scripts/run_spatch.py", "data_io/preprocessing.py")}, "cases": cases, "failed": failed, "limitations": ["Synthetic schema=1 only; no real cache arrays, raw h5ad, metadata, GPU, training or preprocessing executed.", "Same stored bytes/metadata/spatial alignment do not independently prove feature-row biological identity; arrays contain no ordered barcode binding.", "Explicit input identity JSON is caller-supplied provenance; no current raw bytes are verified.", "Main succeeds up to a model-construction sentinel; model, training, OT and production exports are not executed."]}
    write_json(target, report)
    print(json.dumps({"report": str(target), "status": report["status"], "cases": len(cases), "failed": failed}))
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Read-only C1 old_behavior.json/old_cases and frozen spatch_command.json")
    parser.add_argument("--report", default="cache_contract.json")
    args = parser.parse_args()
    if not check_new(args.output_dir, args.reference_dir, args.report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
