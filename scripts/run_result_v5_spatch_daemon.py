#!/usr/bin/env python3
"""Detached SPATCH v5 training followed by the established v3 analysis."""

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


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v5"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "spatch_suite_status.json"
PID_PATH = RESULT_ROOT / "spatch_suite.pid"
EXIT_PATH = RESULT_ROOT / "spatch_suite.exit"
SUPERVISOR_LOG = LOG_ROOT / "spatch_suite_supervisor.log"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def gpu_check(python: str, attempts: int = 5) -> dict:
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        print(f"[{now()}] GPU check attempt {attempt}/{attempts}", flush=True)
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
                    "print(torch.cuda.get_device_name(0), float(y[0,0]))"
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
    raise RuntimeError(f"GPU check failed after {attempts} attempts: {failures}")


def task_commands(python: str) -> list[tuple[str, list[str]]]:
    output = RESULT_ROOT / "spatch" / VARIANT
    return [
        (
            "train_spatch",
            [
                python, str(ROOT / "scripts/run_spatch.py"),
                "--data_dir", str(ROOT / "data/spatch"),
                "--output_dir", str(output),
                "--epochs", "200", "--seed", "42", "--device", "cuda",
                "--n_comps", "50", "--hvg_num", "3000",
                "--lr", "0.001", "--weight_decay", "0",
                "--lambda_contrast", "0.1", "--update_interval", "20",
                "--uot_max_iter", "100", "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--faiss_nlist", "4096",
                "--faiss_nprobe", "64", "--faiss_device", "gpu",
                "--faiss_train_sample_size", "100000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "final",
                "--uot_epsilon", "0.05", "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0", "--uot_stabilizer", "1e-8",
                "--spatial_knn_k", "10", "--graphsage_edge_batch_size", "100000",
                "--decoder_chunk_size", "8192",
                "--ot_attention_source_chunk_size", "2048",
                "--amp_dtype", "bf16",
            ],
        ),
        (
            "analyze_spatch",
            [
                python, str(ROOT / "scripts/analyze_spatch_result_v5_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
    ]


def validate_training() -> None:
    summary_path = RESULT_ROOT / "spatch" / VARIANT / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "train" or summary.get("epochs") != 200:
        raise ValueError("SPATCH summary does not confirm a 200-epoch training run.")
    if summary.get("ot_updates") != EXPECTED_OT_UPDATES:
        raise ValueError(f"Unexpected SPATCH OT schedule: {summary.get('ot_updates')}")
    if summary.get("gpu", {}).get("name") is None:
        raise ValueError("SPATCH summary does not confirm GPU execution.")


def launch_daemon(python: str, from_task: str) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if PID_PATH.is_file():
        try:
            existing_pid = int(PID_PATH.read_text().strip())
            cmdline_path = Path(f"/proc/{existing_pid}/cmdline")
            cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (ValueError, OSError):
            pass
        else:
            if "run_result_v5_spatch_daemon.py" in cmdline:
                raise RuntimeError(
                    f"SPATCH supervisor is already active: PID {existing_pid}"
                )
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
    new_status = {
        "suite": "result_v5_spatch_delayed_ot_refresh",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "dataset": "spatch",
        "baseline": str(RESULT_ROOT.parent / "result_v3/spatch" / VARIANT),
        "epochs": 200,
        "seed": 42,
        "device": "cuda",
        "expected_ot_updates": EXPECTED_OT_UPDATES,
        "status": "checking_gpu",
        "tasks": {},
    }
    status = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if from_task == "analyze_spatch" and STATUS_PATH.is_file()
        else new_status
    )
    status["status"] = "running"
    status.pop("finished_at", None)
    status.pop("traceback", None)
    write_json(STATUS_PATH, status)
    if from_task == "train_spatch":
        status["status"] = "checking_gpu"
        write_json(STATUS_PATH, status)
        status["gpu_check"] = gpu_check(python)
        status["status"] = "running"
        write_json(STATUS_PATH, status)
    tasks = task_commands(python)
    task_names = [name for name, _ in tasks]
    tasks = tasks[task_names.index(from_task):]
    for name, command in tasks:
        log_path = LOG_ROOT / f"{name}.log"
        task = {
            "status": "running",
            "started_at": now(),
            "command": command,
            "log": str(log_path),
        }
        status["tasks"][name] = task
        write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}", flush=True)
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
        if result.returncode == 0 and name == "train_spatch":
            validate_training()
            task["ot_schedule_validated"] = True
        write_json(STATUS_PATH, status)
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed with return code {result.returncode}")
        print(f"[{now()}] COMPLETED {name}", flush=True)
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    EXIT_PATH.write_text("0\n", encoding="utf-8")
    print(f"[{now()}] RESULT_V5_SPATCH_SUITE: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument(
        "--from-task",
        choices=["train_spatch", "analyze_spatch"],
        default="train_spatch",
    )
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python, args.from_task)
        return
    try:
        run_suite(args.python, args.from_task)
    except Exception:
        payload = (
            json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if STATUS_PATH.is_file()
            else {}
        )
        payload.update(
            {
                "status": "failed",
                "finished_at": now(),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(STATUS_PATH, payload)
        EXIT_PATH.write_text("1\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
