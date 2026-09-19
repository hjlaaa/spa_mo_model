#!/usr/bin/env python3
"""Evaluate an explicit existing run into a separate new analysis directory."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    from analysis.evaluation import DATASETS
    from analysis.protocols import SCOPES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--run-dir', '--input-dir', dest='run_dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--protocol', required=True, choices=['requested','comparison','embryo-reference','embryo-per-section','umap-rerender'])
    parser.add_argument('--scope', nargs='+', choices=tuple(SCOPES), default=['joint','independent'])
    parser.add_argument('--joint-per-section', action='store_true', help='Evaluate existing joint assignments per section; never recluster or average sections.')
    parser.add_argument('--k', type=int, nargs='+')
    parser.add_argument('--plot-k', type=int, nargs='+')
    parser.add_argument('--data-dir', type=Path, help='Existing raw metadata/truth source, required by the P6a loader for this dataset.')
    parser.add_argument('--assignments-dir', type=Path, help='Existing analysis labels for joint_per_section alone or Embryo per-section rendering.')
    parser.add_argument('--asw-sample', type=Path, help='Existing Mouse Thymus requested ASW sample, preserving its spot identities.')
    parser.add_argument('--batch-metrics-source', type=Path, help='SPATCH comparison historical batch metric CSV; reported as unverified historical provenance.')
    parser.add_argument('--preprocessing', choices=['raw_embedding','standardized_embedding'], default='standardized_embedding')
    parser.add_argument('--model-version', default='v7A', help='Explicit source label, never inferred from directory names.')
    parser.add_argument('--method-name', help='Explicit method display name; defaults to the retained workflow label.')
    parser.add_argument('--post-ot-graphsage-scale', type=float, default=.5)
    parser.add_argument('--no-plots', action='store_true', help='Skip rendering for the five generic requested workflows.')
    parser.add_argument('--skip-hierarchy', action='store_true')
    parser.add_argument('--plot-seed', type=int, help='Explicit Embryo per-section rendering seed; default is its existing seed 42.')
    parser.add_argument('--maximum-per-section', type=int, default=50000)
    parser.add_argument('--umap-config', type=Path, help='Saved UMAP config for umap-rerender; defaults to run-dir/umap_config.json.')
    args = parser.parse_args(argv)
    if args.k is not None and any(k < 2 for k in args.k):
        parser.error('--k values must be >=2')
    if args.maximum_per_section <= 0:
        parser.error('sample limits must be positive')
    if len(args.scope) != len(set(args.scope)):
        parser.error('--scope entries must be distinct')
    return args


def main(argv=None):
    args = parse_args(argv)
    from analysis.evaluation import evaluate
    result = evaluate(args)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
