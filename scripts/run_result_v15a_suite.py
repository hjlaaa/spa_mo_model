#!/usr/bin/env python3
"""Run the six v15A datasets and their v7A-format analyses serially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import run_result_v7a_suite as v7a
import run_spatch_v7abc_suite as spatch_v7


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v15A"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"
PID_PATH = RESULT_ROOT / "suite.pid"
EXIT_PATH = RESULT_ROOT / "suite.exit.json"
SUPERVISOR_LOG = LOG_ROOT / "suite_supervisor.log"
PYTHON = "/home/hujinlan/miniconda3/envs/cosie/bin/python"
MODEL_VERSION = "v15A"
POST_OT_SCALE = 0.5

# Reuse the exact v7A five-dataset commands and validators with a new result root.
v7a.RESULT_ROOT = RESULT_ROOT
v7a.LOG_ROOT = LOG_ROOT
v7a.MODEL_VERSION = MODEL_VERSION


def tasks(python: str) -> list[tuple[str, list[str], bool]]:
    training = [
        (name, command, True) for name, command in v7a.training_commands(python)
    ]
    training.append(
        (
            "train_spatch",
            spatch_v7.training_command(python, MODEL_VERSION, POST_OT_SCALE),
            True,
        )
    )
    analyses = [
        (
            "analyze_requested_metrics",
            [
                python,
                str(ROOT / "scripts/analyze_result_v7a_requested_metrics.py"),
                "--result-root",
                str(RESULT_ROOT),
                "--model-version",
                MODEL_VERSION,
            ],
            False,
        ),
        (
            "plot_and_prune_clusters",
            [
                python,
                str(ROOT / "scripts/plot_and_prune_result_v7a_clusters.py"),
                "--result-root",
                str(RESULT_ROOT),
            ],
            False,
        ),
        (
            "analyze_spatch",
            spatch_v7.analysis_command(python, MODEL_VERSION, POST_OT_SCALE),
            False,
        ),
    ]
    return training + analyses


def validate_task(name: str, needs_gpu: bool):
    if name == "train_spatch":
        return spatch_v7.validate_training(MODEL_VERSION, POST_OT_SCALE)
    if name == "analyze_spatch":
        return spatch_v7.validate_analysis(MODEL_VERSION, POST_OT_SCALE)
    if name == "plot_and_prune_clusters":
        return {
            "manifest": str(RESULT_ROOT / "clustering_plot_prune_manifest.json")
        }
    if needs_gpu:
        return v7a.validate_training(name)
    return {"manifest": str(RESULT_ROOT / "requested_metrics_manifest.json")}


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
                "exit_file": str(EXIT_PATH),
                "supervisor_log": str(SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_suite(python: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    status = {
        "suite": "result_v15A_pre_ot_spatial_feature_combined_graph",
        "supervisor_pid": os.getpid(),
        "started_at": v7a.now(),
        "python": python,
        "datasets": [
            "MouseBrain",
            "MISAR-seq",
            "Mouse_Spleen",
            "Simulation",
            "Mouse_Thymus",
            "SPATCH",
        ],
        "baseline": "result_v7A",
        "post_ot_graphsage_scale": POST_OT_SCALE,
        "feature_graph": {
            "input": "fused_z_detached",
            "knn": "section_wise_faiss_inner_product",
            "k_feature": "round_half_up(0.5 * k_spatial)",
            "pre_ot_only": True,
        },
        "expected_ot_updates": v7a.EXPECTED_OT_UPDATES,
        "status": "running",
        "tasks": {},
    }
    v7a.write_json(STATUS_PATH, status)
    EXIT_PATH.unlink(missing_ok=True)

    current = "initialization"
    try:
        for current, command, needs_gpu in tasks(python):
            log_path = LOG_ROOT / f"{current}.log"
            task = {
                "status": "checking_gpu" if needs_gpu else "running",
                "started_at": v7a.now(),
                "command": command,
                "log": str(log_path),
            }
            status["tasks"][current] = task
            v7a.write_json(STATUS_PATH, status)
            if needs_gpu:
                task["gpu_check"] = spatch_v7.gpu_check(python)
                task["status"] = "running"
                v7a.write_json(STATUS_PATH, status)

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
                    "finished_at": v7a.now(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if result.returncode == 0:
                task["validation"] = validate_task(current, needs_gpu)
            v7a.write_json(STATUS_PATH, status)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{current} failed with return code {result.returncode}"
                )
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_task": current,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at": v7a.now(),
            }
        )
        v7a.write_json(STATUS_PATH, status)
        v7a.write_json(
            EXIT_PATH,
            {"exit_code": 1, "error": repr(error), "finished_at": v7a.now()},
        )
        raise

    status.update({"status": "completed", "finished_at": v7a.now()})
    v7a.write_json(STATUS_PATH, status)
    v7a.write_json(EXIT_PATH, {"exit_code": 0, "finished_at": v7a.now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        launch_daemon(args.python)
    else:
        run_suite(args.python)


if __name__ == "__main__":
    main()
