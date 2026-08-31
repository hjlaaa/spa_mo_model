#!/usr/bin/env python3
"""Use the common safe selected-gene exporter on a smoothed imputation HDF5."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "gene_imputation" / "extract_imputed_genes.py"),
    run_name="__main__",
)
