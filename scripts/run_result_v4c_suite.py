#!/usr/bin/env python3
"""Run experiment C: fused/context dynamic OT without the context attention gate."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import run_result_v4b_suite as base


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v4C"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def commands(python: str) -> list[tuple[str, list[str]]]:
    base.RESULT_ROOT = RESULT_ROOT
    tasks = base.commands(python)
    adjusted: list[tuple[str, list[str]]] = []
    for name, original in tasks:
        command = list(original)
        if name.startswith("train_"):
            if "--dynamic_candidate_source" in command:
                source_index = command.index("--dynamic_candidate_source") + 1
                command[source_index] = "fused"
            command.append("--disable_context_attention_gate")
        adjusted.append((name, command))
    return adjusted


def main() -> None:
    python = sys.executable
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "suite": "result_v4C_fused_dynamic_ot_without_context_attention_gate",
        "started_at": now(),
        "dataset_order": [
            "dataset_MouseBrain", "Human_Lymph_Node", "MISAR-seq",
            "Mouse_Spleen", "Mouse_Thymus", "Simulation",
        ],
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
        "status": "running",
        "tasks": {},
    }
    write_json(STATUS_PATH, status)
    for name, command in commands(python):
        log_path = LOG_ROOT / f"{name}.log"
        task = {
            "status": "running", "started_at": now(),
            "command": command, "log": str(log_path),
        }
        status["tasks"][name] = task
        write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
            )
        task.update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "finished_at": now(),
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
    print(f"[{now()}] RESULT_V4C_SUITE: PASS", flush=True)


if __name__ == "__main__":
    main()
