#!/usr/bin/env python3
"""Run explicit dataset experiments, each in its canonical CLI subprocess.

Dataset defaults live in the runners. --runner-args passes a JSON token list
unchanged; shared overrides follow it and take precedence. Child relative
paths are relative to the repository, as in the existing suites.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# CLI routing only: no dataset hyperparameters or input defaults here.
RUNNERS = {
    "mousebrain": "run_mousebrain.py",
    "human_embryo": "run_human_embryo_rna_only.py",
    "misar": "run_misar_seq.py",
    "spatch": "run_spatch.py",
    "crc": "run_crc_stereocite.py",
    "human_lymph_node": "run_human_lymph_node.py",
    "mouse_spleen": "run_mouse_spleen.py",
    "mouse_thymus": "run_mouse_thymus.py",
    "simulation": "run_simulation.py",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=RUNNERS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner-args", nargs=2, action="append", default=[],
                        metavar=("DATASET", "JSON_ARGV"),
                        help='Repeatable, e.g. mousebrain \'["--config", "data/configs/mousebrain_preprocess_train.json"]\'.')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interaction-neighbor-weight", type=float, default=None)
    parser.add_argument("--post-ot-graphsage-scale", type=float, default=None)
    parser.add_argument("--keep-going", action="store_true", help="Run remaining datasets after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or starting children.")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    scale = args.post_ot_graphsage_scale
    if scale is not None and (not math.isfinite(scale) or scale < 0):
        parser.error("--post-ot-graphsage-scale must be finite and non-negative")
    rho = args.interaction_neighbor_weight
    if rho is not None and not 0.0 <= rho <= 1.0:
        parser.error("--interaction-neighbor-weight must be finite and between 0 and 1")
    extras = {}
    for dataset, encoded in args.runner_args:
        if dataset not in args.datasets:
            parser.error(f"--runner-args dataset is not selected: {dataset}")
        try:
            tokens = json.loads(encoded)
        except json.JSONDecodeError as error:
            parser.error(f"--runner-args {dataset}: {error}")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            parser.error("--runner-args must be a JSON list of strings")
        if any(token.split("=", 1)[0] in {"--output_dir", "--help", "-h", "--dry_run", "--preprocess_only", "--no-train"} for token in tokens):
            parser.error("Output root and training mode are controlled by the batch entry")
        extras.setdefault(dataset, []).extend(tokens)
    args.runner_args = extras
    return args


def build_tasks(args):
    output_root = args.output_root.expanduser().resolve()
    tasks = []
    for dataset in args.datasets:
        command = [args.python, str(ROOT / "scripts" / RUNNERS[dataset])]
        if dataset not in {"mousebrain", "spatch"}:
            command.append("--train")
        command.extend(args.runner_args.get(dataset, []))
        for name in ("epochs", "seed", "device", "post_ot_graphsage_scale", "interaction_neighbor_weight"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name, str(value)])
        output = output_root / dataset
        command.extend(["--output_dir", str(output)])
        tasks.append({"dataset": dataset, "command": command, "output_dir": str(output)})
    return tasks


def now():
    return datetime.now().astimezone().isoformat()


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_tasks(tasks, output_root, *, keep_going=False):
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "batch_status.json"
    status = {"status": "running", "started_at": now(), "tasks": [dict(task, status="pending") for task in tasks]}
    # A distinct batch record avoids overwriting a previous experiment's status.
    with status_path.open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    exit_code = 0
    for index, task in enumerate(status["tasks"]):
        task.update(status="running", started_at=now())
        write_status(status_path, status)
        try:
            completed = subprocess.run(task["command"], cwd=ROOT)
            task["returncode"] = completed.returncode
            task["status"] = "succeeded" if completed.returncode == 0 else "failed"
        except OSError as error:
            task.update(status="failed", returncode=None, error=str(error))
        except KeyboardInterrupt:
            task.update(status="interrupted", returncode=None)
            exit_code = 130
        task["finished_at"] = now()
        if task["status"] != "succeeded":
            exit_code = exit_code or 1
        if exit_code == 130 or (task["status"] == "failed" and not keep_going):
            for remaining in status["tasks"][index + 1:]:
                remaining["status"] = "skipped"
            break
        write_status(status_path, status)
    status.update(status="succeeded" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed", finished_at=now())
    write_status(status_path, status)
    return exit_code


def main(argv=None):
    args = parse_args(argv)
    tasks = build_tasks(args)
    if args.dry_run:
        print(json.dumps(tasks, indent=2))
        return 0
    return run_tasks(tasks, args.output_root, keep_going=args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
