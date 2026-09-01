#!/usr/bin/env python3
"""Run the five requested result_v7A datasets with result_v6 settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v7A"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"
PID_PATH = RESULT_ROOT / "suite.pid"
EXIT_PATH = RESULT_ROOT / "suite.exit.json"
SUPERVISOR_LOG = LOG_ROOT / "suite_supervisor.log"
MOUSE_RUN = "v3_bidirectional_sparse_fixed_lc0.1"
OTHER_RUN = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
POST_OT_SCALE = 0.5
MODEL_VERSION = "v7A"
SUITE_NAME = "result_v7A_post_ot_graphsage_scale_0.5"
BASELINE = "result_v6"
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]
EXPECTED_ARCHITECTURE = (
    "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder"
)


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


def gpu_check(python: str, label: str, attempts: int = 5) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
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
                    "print(torch.cuda.get_device_name(0), float(y[0,0]))"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and probe.returncode == 0:
            return {
                "checked_at": now(),
                "attempt": attempt,
                "nvidia_smi": smi.stdout.strip(),
                "torch_probe": probe.stdout.strip(),
            }
        errors.append(
            {
                "attempt": attempt,
                "nvidia_smi": smi.stderr.strip(),
                "torch": probe.stderr.strip(),
            }
        )
        if attempt < attempts:
            time.sleep(5)
    raise RuntimeError(f"GPU unavailable for {label}: {errors}")


def output(dataset: str, mousebrain: bool = False) -> str:
    run_name = MOUSE_RUN if mousebrain else OTHER_RUN
    return str(RESULT_ROOT / dataset / run_name)


def training_commands(python: str) -> list[tuple[str, list[str]]]:
    scale = str(POST_OT_SCALE)
    return [
        (
            "train_mousebrain",
            [
                python, str(ROOT / "scripts/run_mousebrain_v2.py"),
                "--config", str(ROOT / "data/configs/mousebrain_preprocess_train.json"),
                "--epochs", "200", "--lambda_contrast", "0.1",
                "--device", "cuda", "--output_dir", output("mousebrain", True),
                "--seed", "42", "--ot_prior_mode", "candidate_sparse",
                "--bidirectional_ot_attention", "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--faiss_nlist", "256",
                "--faiss_nprobe", "32", "--faiss_device", "auto",
                "--faiss_train_sample_size", "10000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "ot",
                "--uot_epsilon", "0.05", "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0", "--uot_max_iter", "100",
                "--update_interval", "20", "--spatial_knn_k", "5",
                "--post_ot_graphsage_scale", scale, "--save_ot_prior_topk",
            ],
        ),
        (
            "train_misar_seq",
            [
                python, str(ROOT / "scripts/run_misar_seq.py"),
                "--output_dir", output("misar_seq"), "--train", "--epochs", "200",
                "--seed", "42", "--device", "cuda", "--hvg_num", "3000",
                "--hvg_num_atac", "3000", "--lambda_contrast", "0.1",
                "--ot_prior_mode", "candidate_sparse", "--bidirectional_ot_attention",
                "--candidate_backend", "faiss_ivf", "--faiss_nlist", "128",
                "--faiss_nprobe", "32", "--faiss_device", "cpu",
                "--faiss_train_sample_size", "20000",
                "--faiss_query_batch_size", "2048",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--dynamic_candidate_source", "ot",
                "--spatial_knn_k", "5", "--graphsage_edge_batch_size", "50000",
                "--post_ot_graphsage_scale", scale,
                "--training_loss_only", "--decoder_chunk_size", "50000",
                "--ot_attention_source_chunk_size", "50000",
                "--checkpoint_ot_attention", "--checkpoint_encoder_fusion",
                "--checkpoint_decoder_chunks", "--checkpoint_graph_encoder",
                "--amp_dtype", "none", "--cache_spatial_graphs",
                "--save_candidate_qc", "--save_outputs", "--save_embeddings",
                "--save_ot_prior_topk", "--update_interval", "20",
            ],
        ),
        *[
            (
                f"train_{dataset}",
                [
                    python, str(ROOT / f"scripts/run_{dataset}.py"),
                    "--output_dir", output(dataset),
                    "--dynamic_candidate_source", "ot",
                    "--update_interval", "20",
                    "--post_ot_graphsage_scale", scale,
                ],
            )
            for dataset in ("mouse_spleen", "simulation", "mouse_thymus")
        ],
    ]


def summary_path(task_name: str) -> Path:
    dataset = task_name.removeprefix("train_")
    base = RESULT_ROOT / dataset
    if dataset == "mousebrain":
        return base / MOUSE_RUN / "epochs_200" / "run_summary.json"
    return base / OTHER_RUN / "run_summary.json"


def validate_training(task_name: str) -> dict[str, Any]:
    path = summary_path(task_name)
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
        "post_ot_graphsage_scale": POST_OT_SCALE,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{task_name}: invalid {MODEL_VERSION} settings: {mismatches}"
        )
    return {"summary": str(path), "validated": expected}


def analysis_command(python: str) -> tuple[str, list[str]]:
    return (
        "analyze_requested_metrics",
        [
            python,
            str(ROOT / "scripts/analyze_result_v7a_requested_metrics.py"),
            "--result-root",
            str(RESULT_ROOT),
        ],
    )


def launch_daemon(python: str, from_task: str | None) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [python, str(Path(__file__).resolve()), "--python", python]
    if from_task:
        command.extend(["--from-task", from_task])
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
                "supervisor_log": str(SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_suite(python: str, from_task: str | None) -> None:
    tasks = [
        *((name, command, True) for name, command in training_commands(python)),
        (*analysis_command(python), False),
    ]
    names = [name for name, _, _ in tasks]
    if from_task:
        if from_task not in names:
            raise ValueError(f"Unknown --from-task {from_task!r}; expected one of {names}")
        tasks = tasks[names.index(from_task):]

    status = {
        "suite": SUITE_NAME,
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "python": python,
        "datasets": ["MouseBrain", "MISAR-seq", "Mouse_Spleen", "Simulation", "Mouse_Thymus"],
        "baseline": BASELINE,
        "post_ot_graphsage_scale": POST_OT_SCALE,
        "expected_ot_updates": EXPECTED_OT_UPDATES,
        "status": "running",
        "tasks": {},
    }
    if from_task and STATUS_PATH.is_file():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        status.update({"supervisor_pid": os.getpid(), "status": "running"})
        status.pop("failed_task", None)
        status.pop("finished_at", None)
    write_json(STATUS_PATH, status)
    EXIT_PATH.unlink(missing_ok=True)

    current = "initialization"
    try:
        for current, command, needs_gpu in tasks:
            log_path = LOG_ROOT / f"{current}.log"
            task = {
                "status": "checking_gpu" if needs_gpu else "running",
                "started_at": now(),
                "command": command,
                "log": str(log_path),
            }
            status["tasks"][current] = task
            write_json(STATUS_PATH, status)
            if needs_gpu:
                task["gpu_check"] = gpu_check(python, current)
                task["status"] = "running"
                write_json(STATUS_PATH, status)
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            task.update(
                {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "finished_at": now(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if result.returncode == 0 and needs_gpu:
                task["validation"] = validate_training(current)
            write_json(STATUS_PATH, status)
            if result.returncode != 0:
                raise RuntimeError(f"{current} failed with return code {result.returncode}")
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_task": current,
                "error": repr(error),
                "finished_at": now(),
            }
        )
        write_json(STATUS_PATH, status)
        write_json(EXIT_PATH, {"exit_code": 1, "error": repr(error), "finished_at": now()})
        raise

    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    write_json(EXIT_PATH, {"exit_code": 0, "finished_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--from-task", default=None)
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        launch_daemon(args.python, args.from_task)
    else:
        run_suite(args.python, args.from_task)


if __name__ == "__main__":
    main()
