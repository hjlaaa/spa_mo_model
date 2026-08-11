#!/usr/bin/env python3
"""Detached serial runner for experiment-C CRC Stereo-CITE-seq and spatch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import run_result_v4_large_suite as base


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v4C"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "large_suite_status.json"
PID_PATH = RESULT_ROOT / "large_suite.pid"
SUPERVISOR_LOG = LOG_ROOT / "large_suite_supervisor.log"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_commands(python: str) -> list[tuple[str, list[str]]]:
    base.RESULT_ROOT = RESULT_ROOT
    tasks = base.task_commands(python)
    adjusted: list[tuple[str, list[str]]] = []
    for name, original in tasks:
        command = list(original)
        if name.startswith("train_"):
            command.append("--disable_context_attention_gate")
        adjusted.append((name, command))
    return adjusted


def launch_daemon(python: str) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [python, str(Path(__file__).resolve()), "--python", python]
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


def run_suite(python: str) -> None:
    status = {
        "suite": "result_v4C_large_fused_ot_without_context_attention_gate",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "dataset_order": ["CRC_Stereo-CITE-seq", "spatch"],
        "epochs": 200,
        "seed": 42,
        "lambda_contrast": 0.1,
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "dynamic_candidate_source": "fused",
        "dynamic_semantic_source": "fused",
        "dynamic_context_source": "fused_spatial_context",
        "attention_context_source": None,
        "attention_context_gate_enabled": False,
        "tasks": {},
        "status": "running",
    }
    write_json(STATUS_PATH, status)
    for name, command in task_commands(python):
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
            )
        task.update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "finished_at": now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(STATUS_PATH, status)
        print(f"[{now()}] {task['status'].upper()} {name}", flush=True)
        if result.returncode != 0:
            status.update(
                {"status": "failed", "failed_task": name, "finished_at": now()}
            )
            write_json(STATUS_PATH, status)
            raise SystemExit(result.returncode)
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    print(f"[{now()}] RESULT_V4C_LARGE_SUITE: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python)
    else:
        run_suite(args.python)


if __name__ == "__main__":
    main()
