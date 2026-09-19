#!/usr/bin/env python3
"""Simulation-only factor and retrieval diagnostics; use evaluate for clustering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from analysis.cache import check_output_path
from analysis.simulation_diagnostics import simulation_diagnostics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = check_output_path(args.output_dir, (args.run_dir, args.data_dir))
    # Diagnostics have no cache: an existing output is never silently reused.
    output.mkdir(parents=True, exist_ok=False)
    simulation_diagnostics(args.run_dir, args.data_dir, output,
                           [f"Simulation{i}" for i in range(1, 6)])


if __name__ == "__main__":
    main()
