#!/usr/bin/env python3
"""Detached serial SPATCH v7A/v7B/v7C training with one shared Harmony cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "spatch_v7ABC_suite"
LOG_ROOT = SUITE_ROOT / "logs"
STATUS_PATH = SUITE_ROOT / "suite_status.json"
PID_PATH = SUITE_ROOT / "suite.pid"
EXIT_PATH = SUITE_ROOT / "suite.exit.json"
SUPERVISOR_LOG = LOG_ROOT / "suite_supervisor.log"
CACHE_DIR = ROOT / "preprocessed_cache/spatch/v6_harmony_n50_hvg3000"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
VERSIONS = (("v7A", 0.5), ("v7B", 0.25), ("v7C", 0.75))
K_VALUES = [5, 8, 10, 12, 14, 16, 20]
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]
EXPECTED_ARCHITECTURE = (
    "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder"
)
SUPERVISED_METRICS = ["ARI", "NMI", "Homogeneity", "Completeness"]
INTERNAL_METRICS = ["ASW", "ASW_Scaled", "CH", "DBI"]
BATCH_METRICS = ["bASW", "bLISI", "kBET", "PCR_score"]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gpu_check(python: str, attempts: int = 5) -> dict[str, Any]:
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        probe = subprocess.run(
            [
                python,
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "x=torch.ones((512,512),device='cuda'); y=x@x; "
                    "torch.cuda.synchronize(); "
                    "print(torch.cuda.get_device_name(0),float(y[0,0]))"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and probe.returncode == 0:
            fields = [item.strip() for item in smi.stdout.strip().split(",")]
            free_mib = int(fields[4]) if len(fields) >= 5 else 0
            if free_mib >= 16000:
                return {
                    "checked_at": now(),
                    "attempt": attempt,
                    "nvidia_smi": smi.stdout.strip(),
                    "torch_probe": probe.stdout.strip(),
                    "free_mib": free_mib,
                }
            failures.append(f"attempt={attempt}: only {free_mib} MiB free")
        else:
            failures.append(
                f"attempt={attempt}: smi={smi.stderr.strip()} torch={probe.stderr.strip()}"
            )
        if attempt < attempts:
            time.sleep(10)
    raise RuntimeError(f"GPU unavailable after {attempts} attempts: {failures}")


def training_command(python: str, version: str, scale: float) -> list[str]:
    output = ROOT / f"result_{version}" / "spatch" / VARIANT
    command = [
        python,
        str(ROOT / "scripts/run_spatch.py"),
        "--data_dir",
        str(ROOT / "data/spatch"),
        "--output_dir",
        str(output),
        "--epochs",
        "200",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--n_comps",
        "50",
        "--hvg_num",
        "3000",
        "--lr",
        "0.001",
        "--weight_decay",
        "0",
        "--lambda_contrast",
        "0.1",
        "--update_interval",
        "20",
        "--uot_max_iter",
        "100",
        "--candidate_backend",
        "faiss_ivf",
        "--initial_modality_candidate_k",
        "100",
        "--candidate_k",
        "200",
        "--attention_topk",
        "10",
        "--faiss_nlist",
        "4096",
        "--faiss_nprobe",
        "64",
        "--faiss_device",
        "gpu",
        "--faiss_train_sample_size",
        "100000",
        "--faiss_query_batch_size",
        "2048",
        "--dynamic_candidate_source",
        "ot",
        "--uot_epsilon",
        "0.05",
        "--uot_tau_a",
        "1.0",
        "--uot_tau_b",
        "1.0",
        "--uot_stabilizer",
        "1e-8",
        "--spatial_knn_k",
        "10",
        "--graphsage_edge_batch_size",
        "100000",
        "--decoder_chunk_size",
        "8192",
        "--ot_attention_source_chunk_size",
        "2048",
        "--amp_dtype",
        "bf16",
        "--post_ot_graphsage_scale",
        str(scale),
        "--preprocessed_cache_dir",
        str(CACHE_DIR),
    ]
    if version == "v7A" and not (CACHE_DIR / "manifest.json").is_file():
        command.append("--build_preprocessed_cache")
    return command


def analysis_command(python: str, version: str, scale: float) -> list[str]:
    return [
        python,
        str(ROOT / "scripts/analyze_spatch_v7_requested_metrics.py"),
        "--result-root",
        str(ROOT / f"result_{version}"),
        "--model-version",
        version,
        "--post-ot-graphsage-scale",
        str(scale),
    ]


def tasks(python: str) -> list[tuple[str, list[str], bool]]:
    result: list[tuple[str, list[str], bool]] = []
    for version, scale in VERSIONS:
        result.append((f"train_{version}", training_command(python, version, scale), True))
        result.append((f"analyze_{version}", analysis_command(python, version, scale), False))
    return result


def validate_training(version: str, scale: float) -> dict[str, Any]:
    path = ROOT / f"result_{version}" / "spatch" / VARIANT / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mode": "train",
        "epochs": 200,
        "seed": 42,
        "architecture": EXPECTED_ARCHITECTURE,
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "dynamic_candidate_source": "ot",
        "attention_context_gate_enabled": False,
        "ot_updates": EXPECTED_OT_UPDATES,
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
    valid_modes = {"built", "loaded"} if version == "v7A" else {"loaded"}
    if cache.get("mode") not in valid_modes or Path(cache.get("path", "")) != CACHE_DIR:
        raise ValueError(f"{version} cache use differs: {cache}")
    if not (CACHE_DIR / "manifest.json").is_file():
        raise FileNotFoundError(CACHE_DIR / "manifest.json")
    return {
        "summary": str(path),
        "validated": expected,
        "cache": cache,
    }


def validate_analysis(version: str, scale: float) -> dict[str, Any]:
    output = ROOT / f"result_{version}" / "spatch" / VARIANT / "analysis/standardized_embedding"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config.get("model_version") != version or config.get("post_ot_graphsage_scale") != scale:
        raise ValueError(f"{version} analysis version/scale differs.")
    if config.get("clustering", {}).get("k_values") != K_VALUES:
        raise ValueError(f"{version} K values differ.")
    if config.get("metrics") != {
        "supervised": SUPERVISED_METRICS,
        "unsupervised": INTERNAL_METRICS,
        "batch": BATCH_METRICS,
    }:
        raise ValueError(f"{version} metric set differs: {config.get('metrics')}")
    if config.get("umap_computed") is not False or config.get("umap_plots") is not False:
        raise ValueError(f"{version} unexpectedly computed UMAP.")
    internal = pd.read_csv(output / "metrics/internal_metrics.csv")
    supervised = pd.read_csv(output / "metrics/supervised_metrics.csv")
    batch = pd.read_csv(output / "metrics/batch_metrics.csv")
    if list(internal.columns) != ["mode", "scope", "k", "n_obs", *INTERNAL_METRICS]:
        raise ValueError(f"{version} internal metric columns differ.")
    if list(supervised.columns) != [
        "mode", "scope", "label", "k", "n_obs", *SUPERVISED_METRICS
    ]:
        raise ValueError(f"{version} supervised metric columns differ.")
    if list(batch.columns) != ["dataset", "n_obs", *BATCH_METRICS]:
        raise ValueError(f"{version} batch metric columns differ.")
    for frame in (internal, supervised, batch):
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError(f"{version} contains non-finite requested metrics.")
    if sorted(internal["k"].unique().astype(int).tolist()) != K_VALUES:
        raise ValueError(f"{version} internal K values differ.")
    if sorted(supervised["k"].unique().astype(int).tolist()) != K_VALUES:
        raise ValueError(f"{version} supervised K values differ.")
    directories = sorted(
        path.name for path in (output / "clustering").iterdir() if path.is_dir()
    )
    expected_directories = sorted(
        f"{mode}_k{k}" for mode in ("joint", "independent") for k in K_VALUES
    )
    if directories != expected_directories:
        raise ValueError(f"{version} clustering directories differ.")
    plots = sorted((output / "clustering").glob("*/spatial_*.png"))
    if len(plots) != 28:
        raise ValueError(f"{version} expected 28 spatial plots, got {len(plots)}.")
    for path in plots:
        if path.stat().st_size <= 1000:
            raise ValueError(f"Invalid plot: {path}")
        with Image.open(path) as image:
            image.verify()
    return {
        "analysis": str(output),
        "k_values": K_VALUES,
        "spatial_plot_count": len(plots),
        "umap_computed": False,
    }


def launch_daemon(python: str, from_task: str) -> None:
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if PID_PATH.is_file():
        try:
            existing_pid = int(PID_PATH.read_text().strip())
            cmdline = Path(f"/proc/{existing_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (ValueError, OSError):
            pass
        else:
            if "run_spatch_v7abc_suite.py" in cmdline:
                raise RuntimeError(f"SPATCH v7ABC supervisor is active: PID {existing_pid}")
    command = [
        python,
        str(Path(__file__).resolve()),
        "--python",
        python,
        "--from-task",
        from_task,
    ]
    with SUPERVISOR_LOG.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pid": process.pid,
                "pid_file": str(PID_PATH),
                "status_file": str(STATUS_PATH),
                "exit_file": str(EXIT_PATH),
                "supervisor_log": str(SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_suite(python: str, from_task: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    all_tasks = tasks(python)
    names = [name for name, _, _ in all_tasks]
    if from_task not in names:
        raise ValueError(f"Unknown task {from_task}; expected one of {names}.")
    selected = all_tasks[names.index(from_task):]
    status = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if from_task != names[0] and STATUS_PATH.is_file()
        else {
            "suite": "SPATCH_v7A_v7B_v7C_shared_Harmony_cache",
            "supervisor_pid": os.getpid(),
            "started_at": now(),
            "dataset": "SPATCH",
            "versions": {version: scale for version, scale in VERSIONS},
            "cache_policy": "v7A_builds_once; v7B_and_v7C_require_and_reuse",
            "preprocessed_cache_dir": str(CACHE_DIR),
            "k_values": K_VALUES,
            "metrics": {
                "supervised": SUPERVISED_METRICS,
                "unsupervised": INTERNAL_METRICS,
                "batch": BATCH_METRICS,
            },
            "umap_computed": False,
            "status": "running",
            "tasks": {},
        }
    )
    status.update({"supervisor_pid": os.getpid(), "status": "running"})
    status.pop("failed_task", None)
    status.pop("error", None)
    status.pop("traceback", None)
    status.pop("finished_at", None)
    write_json(STATUS_PATH, status)
    EXIT_PATH.unlink(missing_ok=True)
    current = "initialization"
    try:
        for current, command, needs_gpu in selected:
            version = current.rsplit("_", 1)[1]
            scale = dict(VERSIONS)[version]
            task = {
                "status": "checking_gpu" if needs_gpu else "running",
                "started_at": now(),
                "command": command,
                "log": str(LOG_ROOT / f"{current}.log"),
            }
            status["tasks"][current] = task
            write_json(STATUS_PATH, status)
            if needs_gpu:
                task["gpu_check"] = gpu_check(python)
                task["status"] = "running"
                write_json(STATUS_PATH, status)
            started = time.perf_counter()
            with Path(task["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=environment,
                )
            task.update(
                {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "finished_at": now(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if result.returncode == 0:
                task["validation"] = (
                    validate_training(version, scale)
                    if needs_gpu
                    else validate_analysis(version, scale)
                )
            write_json(STATUS_PATH, status)
            if result.returncode != 0:
                raise RuntimeError(f"{current} failed with return code {result.returncode}")
        cache_hashes = {
            version: status["tasks"][f"train_{version}"]["validation"]["cache"][
                "manifest_sha256"
            ]
            for version, _ in VERSIONS
        }
        if len(set(cache_hashes.values())) != 1:
            raise ValueError(f"Versions did not use the identical cache: {cache_hashes}")
        status["shared_cache_manifest_sha256"] = next(iter(cache_hashes.values()))
        status.update({"status": "completed", "finished_at": now()})
        write_json(STATUS_PATH, status)
        write_json(EXIT_PATH, {"exit_code": 0, "finished_at": now()})
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_task": current,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at": now(),
            }
        )
        write_json(STATUS_PATH, status)
        write_json(EXIT_PATH, {"exit_code": 1, "error": repr(error), "finished_at": now()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument(
        "--from-task",
        choices=[name for name, _, _ in tasks(sys.executable)],
        default="train_v7A",
    )
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python, args.from_task)
    else:
        run_suite(args.python, args.from_task)


if __name__ == "__main__":
    main()
