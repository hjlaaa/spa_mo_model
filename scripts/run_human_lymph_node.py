#!/usr/bin/env python3
"""Train spa_mo_model on paired Human Lymph Node RNA + ADT sections."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.entry_defaults import paired_rna_protein_defaults as _entry_defaults
from scripts import paired_cli as crc
from scripts.paired_entry import run_paired_entry
from data_io.paired_preparation import PairSpec, prepare_paired_dataset
from data_io.paired import (
    read_lymph_pair, prepare_rna_gene_ids, filter_globally_nonzero_genes,
)


DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Human_Lymph_Node")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/human_lymph_node/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)

SAMPLES = {
    "Human_Lymph_Node_A1": "Human_Lymph_Node_A1",
    "Human_Lymph_Node_D1": "Human_Lymph_Node_D1",
}


def get_dataset_defaults(*, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults(data_dir=data_dir, output_dir=output_dir)


def parse_args(
    argv=None, *, data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
    dataset_name="Human Lymph Node", samples=None,
):
    # Preserve the existing wrapper defaults; explicit CLI values still win.
    defaults = get_dataset_defaults(data_dir=data_dir, output_dir=output_dir)
    return crc.parse_args(
        argv, defaults=defaults, dataset_name=dataset_name,
        sample_dirs=tuple((SAMPLES if samples is None else samples).values()),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    run_config = crc.resolve_run_config(args, samples=SAMPLES, dataset_name="Human Lymph Node")
    run_paired_entry(
        args,
        status_prefix="HUMAN_LYMPH_NODE",
        run_config=run_config,
        prepare_dataset=prepare_paired_dataset,
        pair_spec=PairSpec(SAMPLES, read_lymph_pair, prepare_rna_gene_ids, filter_globally_nonzero_genes),
    )


if __name__ == "__main__":
    main()
