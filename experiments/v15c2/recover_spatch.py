"""Recover completed SPATCH exports, then run only the three missing analyses.

Keep original training source identity, nonzero exit codes and failure files.
No training entry point or optimizer is invoked by this recovery workflow.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .plan import ROOT, output_root, render_job, source_identity
from .supervisor import artifacts, atomic_json, completed, now


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(state, control):
    """The only changed original source may be the reviewed serializer fix."""
    for name, old_hash in state["source_identity"]["files"].items():
        archived = control / "source" / name
        if digest(archived) != old_hash:
            raise ValueError(f"Training source snapshot changed: {name}")
        if name == "training/fit.py":
            expected = archived.read_text().replace(
                "def json_safe(value: Any):\n",
                "def json_safe(value: Any):\n    if isinstance(value, Path):\n        return str(value)\n",
                1,
            )
            if (ROOT / name).read_text() != expected:
                raise ValueError("Unexpected training runtime change beyond Path serialization.")
        elif digest(ROOT / name) != old_hash:
            raise ValueError(f"Unexpected original source/config change: {name}")


def recover_summary(record, suite, ds, state):
    import numpy as np
    from scripts.run_spatch import parse_args, resolve_run_config, SECTIONS, CACHE_METADATA_FILES
    from training.fit import json_safe

    run = Path(record["output_dir"])
    if record["status"] != "failed" or record["returncode"] != 1:
        raise ValueError(f"Not the known failed export: {record['id']}")
    if (run / "run_summary.json").exists():
        raise FileExistsError(run / "run_summary.json")
    failure = json.loads((run / "run_failure.json").read_text())
    if failure["error"] != "TypeError('Object of type PosixPath is not JSON serializable')":
        raise ValueError("Unexpected training failure; refusing metadata-only recovery.")
    args = parse_args(record["command"][2:])
    resolved = resolve_run_config(args)
    if args.seed != record["seed"] or args.epochs != 200 or args.device != "cuda":
        raise ValueError("Unexpected recorded training configuration.")
    cache = args.preprocessed_cache_dir
    manifest_path = cache / "manifest.json"
    if digest(manifest_path) != ds["cache_manifest_sha256"]:
        raise ValueError("Preprocessing manifest changed.")
    manifest = json.loads(manifest_path.read_text())
    history = json.loads((run / "loss_history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, 201)):
        raise ValueError("Training history is incomplete.")
    if not all(np.isfinite(row[k]) for row in history for k in
               ("total_loss", "crossview_loss", "reconstruction_loss")):
        raise ValueError("Training contains nonfinite losses.")
    refreshes = json.loads((run / "feature_graph_refresh.json").read_text())
    if [row["epoch"] for row in refreshes] != [100, 120, 140, 160, 180, 200]:
        raise ValueError("Incomplete graph/UOT refreshes.")
    with (run / "cuda_memory_trace.jsonl").open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    if last is None or last["stage"] != "final_eval_end" or last["epoch"] != 200 or not last["cuda_available"]:
        raise ValueError("Final CUDA evaluation completion not established.")
    evidence = {}
    for name in CACHE_METADATA_FILES:
        if digest(run / name) != digest(cache / name):
            raise ValueError(f"Copied preprocessing metadata differs: {name}")
        evidence[name] = {"sha256": digest(run / name), "matches_cache": True}
    embeddings = {}
    for section in SECTIONS:
        path = run / f"final_embeddings_{section}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (manifest["alignment"][section]["n_spots"], 128)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"Embedding schema differs: {path}")
        for start in range(0, len(array), 65536):
            if not np.isfinite(array[start:start + 65536]).all():
                raise ValueError(f"Nonfinite embedding: {path}")
        evidence[path.name] = {"sha256": digest(path), "shape": list(array.shape),
                               "dtype": str(array.dtype), "all_values_finite": True}
        embeddings[section] = str(path)
    for name in ("loss_history.json", "feature_graph_refresh.json", "cuda_memory_trace.jsonl", "run_failure.json"):
        evidence[name] = {"sha256": digest(run / name)}
    recovery = {"recovered_at": now(), "method": "metadata_only_from_saved_evidence",
                "retrained": False, "original_exit_code": record["returncode"],
                "original_training_source_sha256": state["source_identity"]["sha256"],
                "original_command": record["command"], "evidence": evidence,
                "note": "Reconstructed summary; unavailable transient runtime fields are omitted. Original failure retained."}
    summary = {
        **json_safe(vars(args)), "resolved_config": json_safe(resolved),
        "dataset": "spatch", "method": "spa_mo_model", "model_version": "v15C-2",
        "mode": "train", "train": True, "full_spot": True,
        "input_data_path": str(cache), "input_data_kind": "model_ready_preprocessed_cache",
        "section_names": list(SECTIONS), "modalities": ["HE", "RNA", "Protein"],
        "use_harmony": not args.no_harmony,
        "architecture": "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder",
        "pre_post_graphsage_parameter_sharing": False, "ot_refresh_embedding_key": "ot_embeddings",
        "dynamic_candidate_source": "ot", "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True, "post_ot_graphsage_scale": args.post_ot_graphsage_scale,
        "alignment": manifest["alignment"],
        "preprocessing_cache": {"mode": "loaded", "path": str(cache),
                                "manifest_sha256": digest(manifest_path), "provenance": "recovered from frozen command and verified cache metadata"},
        "gpu": {"device_name": last["device_name"], "cuda_available": True,
                "evidence_source": str(run / "cuda_memory_trace.jsonl")},
        "embedding_paths": embeddings, "training_history": history,
        "ot_updates": [row["epoch"] for row in refreshes], "total_loss_finite": True,
        "summary_recovery": recovery,
    }
    return summary


def prepare(control, state, suite):
    recovery_dir = control / "spatch_recovery"
    receipt = recovery_dir / "preparation.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["source_identity"] != source_identity(suite):
            raise ValueError("Recovery source changed since preparation.")
        for seed in suite["seeds"]:
            if not completed(state["tasks"][f"spatch/seed_{seed}/train"]):
                raise ValueError("Recovered training outputs failed validation.")
        return
    verify_sources(state, control)
    ds = next(d for d in suite["datasets"] if d["id"] == "spatch")
    prepared = []
    for seed in suite["seeds"]:
        record = state["tasks"][f"spatch/seed_{seed}/train"]
        summary = recover_summary(record, suite, ds, state)
        prepared.append((record, summary))
        print(f"Validated SPATCH seed={seed}: 200 epochs, complete finite embeddings, CUDA final eval.", flush=True)
    recovery_dir.mkdir(exist_ok=True)
    backup = recovery_dir / "status_before_recovery.json"
    if backup.exists():
        raise FileExistsError("An interrupted recovery exists; inspect it before proceeding.")
    atomic_json(backup, state)
    for record, summary in prepared:
        state["attempt_history"].append(copy.deepcopy(record))
        atomic_json(Path(record["output_dir"]) / "run_summary.json", summary)
        record.update(status="completed", recovery=summary["summary_recovery"], recovered_at=now())
        # returncode remains 1: the original subprocess failed at export.
        record["artifacts"] = artifacts(record)
    identity = source_identity(suite)
    state["spatch_recovery"] = {"prepared_at": now(), "retrained": False,
                               "source_identity": identity, "backup": str(backup)}
    state.update(status="analysis_pending", updated_at=now())
    atomic_json(control / "status.json", state)
    atomic_json(receipt, {"prepared_at": now(), "source_identity": identity,
                          "seeds": suite["seeds"], "retrained": False})


def analyze(control, state, suite):
    ds = next(d for d in suite["datasets"] if d["id"] == "spatch")
    state.update(status="running", supervisor_pid=os.getpid(), updated_at=now())
    child = None

    def stop(signum, frame):
        raise KeyboardInterrupt("Recovery supervisor interrupted")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    atomic_json(control / "status.json", state)
    try:
        for seed in suite["seeds"]:
            key = f"spatch/seed_{seed}/analysis"
            previous = state["tasks"].get(key)
            if completed(previous):
                continue
            train = state["tasks"][f"spatch/seed_{seed}/train"]
            if not completed(train):
                raise ValueError("Recovered training artifacts changed.")
            attempt = 1 if previous is None else previous["attempt"] + 1
            while True:
                job = render_job(suite, ds, seed, "analysis", attempt, train["output_dir"])
                log = control / "logs" / f"spatch_seed{seed}_analysis_{attempt:02d}.log"
                if not Path(job["output_dir"]).exists() and not log.exists():
                    break
                attempt += 1
            if previous:
                state["attempt_history"].append(copy.deepcopy(previous))
            job.update(status="running", started_at=now(), log=str(log), recovery_analysis=True)
            state["tasks"][key] = job
            atomic_json(control / "status.json", state)
            print(f"{now()} START {key}", flush=True)
            with log.open("x") as handle:
                child = subprocess.Popen(job["command"], cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
                job["pid"] = child.pid
                atomic_json(control / "status.json", state)
                code = child.wait()
                child = None
            job.update(returncode=code, finished_at=now())
            try:
                if code:
                    raise RuntimeError(f"Analysis exited with code {code}")
                job["artifacts"] = artifacts(job)
                job["status"] = "completed"
            except Exception as exc:
                job.update(status="failed", error=str(exc))
            state["updated_at"] = now()
            atomic_json(control / "status.json", state)
            print(f"{now()} {job['status'].upper()} {key}", flush=True)
        expected = 2 * len(suite["datasets"]) * len(suite["seeds"])
        state["status"] = "completed" if sum(t["status"] == "completed" for t in state["tasks"].values()) == expected else "completed_with_failures"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        for job in state["tasks"].values():
            if job["status"] == "running":
                job.update(status="interrupted", finished_at=now())
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["updated_at"] = now()
        atomic_json(control / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="After validation/recovery, run only missing SPATCH analyses.")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach and not args.execute:
        parser.error("--detach requires --execute")
    # Always use the suite frozen at the time of training.
    control = ROOT / "result_v15C-2/_control"
    suite = json.loads((control / "suite_config.json").read_text())
    if output_root(suite) != control.parent or suite["seeds"] != [42, 43, 44]:
        raise ValueError("Unexpected recovery suite.")
    if args.detach:
        with (control / "spatch_recovery_supervisor.log").open("a") as log:
            process = subprocess.Popen([suite["python"], "-m", "experiments.v15c2.recover_spatch", "--execute"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(json.dumps({"supervisor_pid": process.pid, "log": str(log.name)}))
        return
    with (control / "suite.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((control / "status.json").read_text())
        if state["status"] == "running":
            raise RuntimeError("Queue already running; inspect it before recovery.")
        prepare(control, state, suite)
        if args.execute:
            analyze(control, state, suite)


if __name__ == "__main__":
    main()
