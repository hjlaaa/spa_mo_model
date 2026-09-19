"""Plan by default; only --execute can launch the v15C-2 queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .plan import ROOT, check_inputs, jobs, load_suite, output_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--check", action="store_true", help="Read-only input/provenance check; no data loading or GPU probe.")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--check-gpu", action="store_true", help="Check CUDA/FAISS GPU with retries; no training or analysis.")
    parser.add_argument("--detach", action="store_true", help="Launch supervisor in a detached session; requires --execute.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if (args.detach or args.resume or args.stop_on_error) and not args.execute:
        parser.error("--detach/--resume/--stop-on-error require --execute")
    suite = load_suite()
    if args.check_gpu:
        from .gpu import probe_gpu
        print(json.dumps(probe_gpu(suite["python"]), indent=2))
        return 0
    if args.status:
        path = output_root(suite) / "_control/status.json"
        print(path.read_text() if path.exists() else json.dumps({"status": "not_started"}))
        return 0
    if args.check:
        print(json.dumps(check_inputs(suite), indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"status": "planned_only", "jobs": jobs(suite)}, indent=2))
        return 0
    if args.detach:
        check_inputs(suite)
        root = output_root(suite)
        root.mkdir(parents=True, exist_ok=True)
        command = [suite["python"], "-m", "experiments.v15c2.run", "--execute"]
        if args.resume:
            command.append("--resume")
        if args.stop_on_error:
            command.append("--stop-on-error")
        with (root / "supervisor.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        print(json.dumps({"supervisor_pid": process.pid, "status": "launch_requested",
                          "log": str(root / "supervisor.log")}, indent=2))
        return 0
    from .supervisor import run
    return run(suite, resume=args.resume, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
