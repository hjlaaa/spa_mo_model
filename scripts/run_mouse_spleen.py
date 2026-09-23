#!/usr/bin/env python3
"""Train spa_mo_model on paired Mouse Spleen RNA + ADT sections."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import paired_cli as crc
from scripts.paired_entry import run_paired_entry
from training.entry_defaults import paired_rna_protein_defaults as _entry_defaults
from data_io.paired_preparation import PairSpec, prepare_paired_dataset
from data_io.paired import (
    read_spleen_pair, prepare_rna_gene_ids, filter_globally_nonzero_genes,
)


DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Mouse_Spleen")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/mouse_spleen/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)

SAMPLES = {"Mouse_Spleen1": "Mouse_Spleen1", "Mouse_Spleen2": "Mouse_Spleen2"}


def get_dataset_defaults():
    """Spleen uses the paired RNA/ADT settings with its own paths and identity."""
    return _entry_defaults(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)


def parse_args(argv=None):
    return crc.parse_args(
        argv, defaults=get_dataset_defaults(), dataset_name="Mouse Spleen",
        sample_dirs=tuple(SAMPLES.values()),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    run_config = crc.resolve_run_config(args, samples=SAMPLES, dataset_name="Mouse Spleen")
    run_paired_entry(
        args,
        status_prefix="MOUSE_SPLEEN",
        run_config=run_config,
        prepare_dataset=prepare_paired_dataset,
        pair_spec=PairSpec(SAMPLES, read_spleen_pair, prepare_rna_gene_ids, filter_globally_nonzero_genes),
    )


if __name__ == "__main__":
    main()
