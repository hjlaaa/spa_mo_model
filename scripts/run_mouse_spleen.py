#!/usr/bin/env python3
"""Train spa_mo_model on paired Mouse Spleen RNA + ADT sections."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_crc_stereocite as crc
from scripts import run_human_lymph_node as shared
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
    return shared.get_dataset_defaults(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)


def parse_args(argv=None):
    return crc.parse_args(
        argv, defaults=get_dataset_defaults(), dataset_name="Mouse Spleen",
        sample_dirs=tuple(SAMPLES.values()),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    run_config = crc.resolve_run_config(args, samples=SAMPLES, dataset_name="Mouse Spleen")
    crc.run_crc_pipeline(
        args, samples=SAMPLES, read_pair=read_spleen_pair,
        prepare_rna=prepare_rna_gene_ids,
        filter_shared_genes=filter_globally_nonzero_genes,
        status_prefix="MOUSE_SPLEEN",
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
