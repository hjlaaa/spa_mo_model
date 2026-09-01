#!/usr/bin/env python3
"""Run v7B like the persisted v7A suite, changing only post-OT scale to 0.25."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_result_v7a_suite as suite  # noqa: E402


RESULT_ROOT = ROOT / "result_v7B"
POST_OT_SCALE = 0.25
MODEL_VERSION = "v7B"


def configure_suite() -> None:
    suite.RESULT_ROOT = RESULT_ROOT
    suite.LOG_ROOT = RESULT_ROOT / "logs"
    suite.STATUS_PATH = RESULT_ROOT / "suite_status.json"
    suite.PID_PATH = RESULT_ROOT / "suite.pid"
    suite.EXIT_PATH = RESULT_ROOT / "suite.exit.json"
    suite.SUPERVISOR_LOG = suite.LOG_ROOT / "suite_supervisor.log"
    suite.POST_OT_SCALE = POST_OT_SCALE
    suite.MODEL_VERSION = MODEL_VERSION
    suite.SUITE_NAME = "result_v7B_post_ot_graphsage_scale_0.25"
    suite.BASELINE = "result_v7A"

    def postprocess_command(python: str) -> tuple[str, list[str]]:
        return (
            "analyze_plot_prune_requested_metrics",
            [
                python,
                str(ROOT / "scripts/postprocess_result_variant.py"),
                "--result-root",
                str(RESULT_ROOT),
                "--model-version",
                MODEL_VERSION,
                "--post-ot-graphsage-scale",
                str(POST_OT_SCALE),
                "--python",
                python,
            ],
        )

    suite.analysis_command = postprocess_command


def launch_daemon(python: str, from_task: str | None) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    suite.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [python, str(Path(__file__).resolve()), "--python", python]
    if from_task:
        command.extend(["--from-task", from_task])
    with suite.SUPERVISOR_LOG.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    suite.PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pid": process.pid,
                "pid_file": str(suite.PID_PATH),
                "status_file": str(suite.STATUS_PATH),
                "supervisor_log": str(suite.SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--from-task", default=None)
    args = parser.parse_args()
    configure_suite()
    if args.daemon:
        launch_daemon(args.python, args.from_task)
    else:
        suite.run_suite(args.python, args.from_task)


if __name__ == "__main__":
    main()
