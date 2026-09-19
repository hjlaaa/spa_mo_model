"""Sequential subprocess queue with a locked ledger and fresh retry outputs.

Nothing in model/, training/, data_io/ or analysis/ imports this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .plan import ROOT, check_inputs, output_root, render_job, source_identity
from .gpu import probe_gpu


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def artifacts(job):
    """Require completed outputs, not just a zero subprocess return code."""
    out = Path(job["output_dir"])
    if job["stage"] == "train":
        summary_path = out / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("epochs") != 200 or summary.get("seed") != job["seed"]:
            raise ValueError("Training epoch/seed mismatch.")
        if summary.get("ot_updates") != [100, 120, 140, 160, 180, 200]:
            raise ValueError("Unexpected UOT refresh schedule.")
        resolved = summary["resolved_config"]
        config = resolved["model_config"]
        if config["training"]["device"] != "cuda":
            raise ValueError("Training did not use the required CUDA configuration.")
        if config["feature_graph"]["enabled"] is not True:
            raise ValueError("Feature graph was not enabled.")
        finite = summary.get("total_loss_finite", summary.get("loss_finite"))
        if job["dataset"] == "spatch":
            history = summary.get("training_history", [])
            finite = len(history) == 200 and all(math.isfinite(r["total_loss"]) for r in history)
        if finite is not True:
            raise ValueError("Training did not report finite loss.")
        trace = out / "feature_graph_refresh.json"
        events = json.loads(trace.read_text())
        if [e["epoch"] for e in events] != summary["ot_updates"] or any(
            e["refresh_passes_before_uot"] != 2 or
            not e["uot_refresh_embeddings_from_updated_feature_graph"] for e in events
        ):
            raise ValueError("Feature graph/UOT lifecycle validation failed.")
        embeddings = list(out.glob("final_embeddings_*.npy"))
        if job["dataset"] == "mousebrain":
            embeddings = list((out / "final_embeddings").glob("*_final_embedding.npy"))
        expected = 3 if job["dataset"] == "mousebrain" else (4 if job["dataset"] in ("misar_seq", "mouse_thymus") else 2)
        if len(embeddings) != expected:
            raise ValueError("Missing final embeddings.")
        paths = [summary_path, trace, *embeddings]
    else:
        paths = [out / "standardized_embedding" / "metrics" / name for name in
                 ("internal_metrics.csv", "batch_metrics.csv")]
        if job["dataset"] in ("mousebrain", "misar_seq", "spatch"):
            paths.append(out / "standardized_embedding/metrics/supervised_metrics.csv")
        paths.append(out / "standardized_embedding/config.json")
    records = {}
    for path in paths:
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError(f"Empty output: {path}")
        records[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return records


def completed(record):
    if not record or record.get("status") != "completed":
        return False
    if artifacts(record) != record["artifacts"]:
        raise ValueError(f"Completed artifacts were modified: {record['id']}")
    return True


def run(suite, *, resume=False, stop_on_error=False):
    if Path(sys.executable).resolve() != Path(suite["python"]).resolve():
        raise RuntimeError(f"Use the configured interpreter: {suite['python']}")
    check_inputs(suite)
    # Runtime probe is performed only on explicit execution, before creating
    # experiment outputs or launching any training/analysis process.
    probe = probe_gpu(suite["python"])
    root = output_root(suite)
    root.mkdir(parents=True, exist_ok=True)
    control = root / "_control"
    control.mkdir(exist_ok=True)
    with (control / "suite.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another v15C-2 supervisor owns the queue.") from None
        return _run_locked(suite, control, resume, stop_on_error, probe)


def _run_locked(suite, control, resume, stop_on_error, gpu):
    status_path = control / "status.json"
    identity = source_identity(suite)
    if status_path.exists():
        if not resume:
            raise RuntimeError("Existing queue; use --resume to skip completed stages.")
        state = json.loads(status_path.read_text())
        if state["source_identity"] != identity:
            raise RuntimeError("Source/config changed; refusing to mix implementations in one suite.")
        for record in state["tasks"].values():
            if record.get("status") == "running" and record.get("pid"):
                proc = Path(f"/proc/{record['pid']}/cmdline")
                if proc.exists() and str(ROOT).encode() in proc.read_bytes():
                    raise RuntimeError(f"Previous child still running (PID {record['pid']}); stop it before resuming.")
    else:
        state = {"experiment": suite["experiment"], "created_at": now(),
                 "source_identity": identity, "tasks": {}, "attempt_history": [],
                 "python": sys.version, "gpu": gpu,
                 "environment": {k: os.environ.get(k) for k in (
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
        packages = {}
        for name in ("torch", "numpy", "scikit-learn", "scanpy", "anndata", "harmonypy", "faiss-gpu"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "unavailable"
        state["packages"] = packages
        atomic_json(control / "suite_config.json", suite)
        snapshot = control / "source"
        for relative in identity["files"]:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
    state.update(status="running", supervisor_pid=os.getpid(), updated_at=now())
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    atomic_json(status_path, state)
    try:
        for seed in suite["seeds"]:
            for ds in suite["datasets"]:
                trained_dir = None
                for stage in ("train", "analysis"):
                    if stopping:
                        break
                    template = render_job(suite, ds, seed, stage)
                    previous = state["tasks"].get(template["id"])
                    if completed(previous):
                        if stage == "train":
                            trained_dir = previous["output_dir"]
                        continue
                    if stage == "analysis" and trained_dir is None:
                        break
                    if previous:
                        state["attempt_history"].append(previous)
                    attempt = 1 if not previous else previous["attempt"] + 1
                    job = render_job(suite, ds, seed, stage, attempt, trained_dir)
                    # Orphan directories from an interrupted supervisor are
                    # preserved; a new attempt always uses an unused path.
                    log = control / "logs" / f"{ds['id']}_seed{seed}_{stage}_{attempt:02d}.log"
                    while Path(job["output_dir"]).exists() or log.exists():
                        attempt += 1
                        job = render_job(suite, ds, seed, stage, attempt, trained_dir)
                        log = control / "logs" / f"{ds['id']}_seed{seed}_{stage}_{attempt:02d}.log"
                    log.parent.mkdir(exist_ok=True)
                    job.update(status="running", started_at=now(), log=str(log))
                    state["tasks"][job["id"]] = job
                    atomic_json(status_path, state)
                    print(f"{now()} START {job['id']} attempt={attempt}", flush=True)
                    try:
                        with log.open("x") as handle:
                            child = subprocess.Popen(job["command"], cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
                            job["pid"] = child.pid
                            atomic_json(status_path, state)
                            stop_deadline = None
                            while True:
                                try:
                                    code = child.wait(timeout=1)
                                    break
                                except subprocess.TimeoutExpired:
                                    if stopping:
                                        if stop_deadline is None:
                                            stop_deadline = time.monotonic() + 30
                                        if time.monotonic() >= stop_deadline:
                                            os.killpg(child.pid, signal.SIGKILL)
                        child = None
                        job["returncode"] = code
                        if stopping:
                            job["status"] = "interrupted"
                        elif code:
                            job["status"] = "failed"
                        else:
                            job["artifacts"] = artifacts(job)
                            job["status"] = "completed"
                    except Exception as exc:
                        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    job["finished_at"] = now()
                    state["updated_at"] = now()
                    atomic_json(status_path, state)
                    print(f"{now()} {job['status'].upper()} {job['id']}", flush=True)
                    if job["status"] != "completed":
                        if stop_on_error:
                            stopping = True
                        break
                    if stage == "train":
                        trained_dir = job["output_dir"]
                if stopping:
                    break
            if stopping:
                break
        count = sum(r["status"] == "completed" for r in state["tasks"].values())
        expected_count = 2 * len(suite["datasets"]) * len(suite["seeds"])
        state["status"] = "completed" if count == expected_count else ("interrupted" if stopping else "completed_with_failures")
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state["updated_at"] = now()
        atomic_json(status_path, state)
    return 0 if state["status"] == "completed" else 1
