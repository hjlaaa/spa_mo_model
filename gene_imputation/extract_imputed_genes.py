#!/usr/bin/env python3
"""Export selected genes from a full imputation HDF5 without loading all genes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full imputation HDF5.")
    parser.add_argument("--genes", nargs="*", default=[], help="Gene names to export.")
    parser.add_argument(
        "--gene-file",
        type=Path,
        default=None,
        help="Optional text file containing one gene name per line.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output compressed NPZ.")
    return parser.parse_args()


def requested_genes(args: argparse.Namespace) -> list[str]:
    genes = [str(gene).strip() for gene in args.genes if str(gene).strip()]
    if args.gene_file is not None:
        genes.extend(
            line.strip()
            for line in args.gene_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    # Preserve request order while removing duplicate names.
    return list(dict.fromkeys(genes))


def main() -> None:
    args = parse_args()
    genes = requested_genes(args)
    if not genes:
        raise ValueError("Specify at least one gene with --genes or --gene-file.")

    with h5py.File(args.input, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("The input imputation file is not marked complete.")
        available = handle["var_names"].asstr()[:]
        lookup = {gene: index for index, gene in enumerate(available)}
        missing = [gene for gene in genes if gene not in lookup]
        if missing:
            preview = ", ".join(missing[:10])
            raise KeyError(f"Genes absent from imputed targets ({len(missing)}): {preview}")

        requested_indices = np.asarray([lookup[gene] for gene in genes], dtype=np.int64)
        # h5py requires increasing fancy indices. Read in sorted file order and
        # restore the caller's requested order afterwards.
        sorted_order = np.argsort(requested_indices)
        sorted_indices = requested_indices[sorted_order]
        sorted_values = np.asarray(handle["X"][:, sorted_indices], dtype=np.float32)
        values = sorted_values[:, np.argsort(sorted_order)]
        obs_names = handle["obs_names"].asstr()[:]
        spatial = np.asarray(handle["obsm"]["spatial"][:])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=values,
        genes=np.asarray(genes, dtype=str),
        obs_names=np.asarray(obs_names, dtype=str),
        spatial=spatial,
        source_h5=np.asarray(str(args.input.resolve())),
    )
    print(f"Saved {values.shape[0]:,} spots x {values.shape[1]:,} genes to {args.output}")


if __name__ == "__main__":
    main()
