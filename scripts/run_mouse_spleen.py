#!/usr/bin/env python3
"""Train spa_mo_model on paired Mouse Spleen RNA + ADT sections."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import run_human_lymph_node as shared


ORIGINAL = Path(__file__).with_name("run_crc_stereocite.py")
DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Mouse_Spleen")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/mouse_spleen/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)


def _load_pipeline() -> dict:
    source = ORIGINAL.read_text(encoding="utf-8")
    for old, new in [
        ("CRC_003_bin20", "Mouse_Spleen1"),
        ("CRC_006_bin20", "Mouse_Spleen2"),
        ("CRC_003", "Mouse_Spleen1"),
        ("CRC_006", "Mouse_Spleen2"),
        ("CRC Stereo-CITE-seq", "Mouse Spleen"),
        ("CRC_STEREOCITE", "MOUSE_SPLEEN"),
    ]:
        source = source.replace(old, new)
    shared_assignment = (
        "shared_genes_all = [gene for gene in rna003.var_names.astype(str) "
        "if gene in shared_set_006]"
    )
    if shared_assignment not in source:
        raise RuntimeError("Could not locate shared-gene assignment in the base pipeline.")
    source = source.replace(
        shared_assignment,
        shared_assignment
        + "\n        shared_genes_all = filter_globally_nonzero_genes("
        "rna003, rna006, shared_genes_all)",
        1,
    )
    namespace = {"__file__": str(ORIGINAL), "__name__": "mouse_spleen_spa_mo_pipeline"}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    return namespace


def _setup_adt_names(adt) -> None:
    markers = pd.Index(adt.var_names.astype(str))
    if markers.hasnans or not markers.is_unique:
        raise ValueError("Mouse Spleen ADT marker names must be complete and unique.")
    adt.var["adt_name_original"] = markers.to_numpy()
    adt.var["adt_marker"] = markers.to_numpy()


def main() -> None:
    namespace = _load_pipeline()
    original_read_pair = namespace["read_backed_pair"]

    def read_pair(data_dir, sample_dir):
        rna, adt = original_read_pair(data_dir, sample_dir)
        _setup_adt_names(adt)
        return rna, adt

    namespace["read_backed_pair"] = read_pair
    namespace["prepare_rna_var_names_make_unique"] = shared._prepare_rna_gene_ids
    namespace["filter_globally_nonzero_genes"] = shared._filter_globally_nonzero_genes
    shared.DATA_DIR = DATA_DIR
    shared.OUTPUT_DIR = OUTPUT_DIR
    shared._inject_defaults()
    args = namespace["parse_args"]()
    namespace["run_crc_pipeline"](args)


if __name__ == "__main__":
    main()
