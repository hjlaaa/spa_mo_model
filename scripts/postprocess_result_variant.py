#!/usr/bin/env python3
"""Run the requested metrics, v6-style plots, and K pruning for one variant."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--post-ot-graphsage-scale", type=float, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    commands = [
        [
            args.python,
            str(ROOT / "scripts/analyze_result_v7a_requested_metrics.py"),
            "--result-root",
            str(args.result_root.resolve()),
            "--model-version",
            args.model_version,
            "--post-ot-graphsage-scale",
            str(args.post_ot_graphsage_scale),
        ],
        [
            args.python,
            str(ROOT / "scripts/plot_and_prune_result_v7a_clusters.py"),
            "--result-root",
            str(args.result_root.resolve()),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
