#!/usr/bin/env python3
"""Train spa_mo_model jointly on five paired Simulation RNA + ADT sections."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.entry_defaults import simulation_defaults as _entry_defaults
from scripts import multisection_cli as misar
from scripts.multisection_entry import run_multisection_entry
from data_io.misar_preparation import MultisectionSpec, prepare_multisection_dataset
from data_io.adapted import (
    simulation_spatial_domain as _spatial_domain,
    adapt_simulation_pair as _adapt_pair, read_simulation_pair as read_pair,
    simulation_prepared_truth,
)


DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Simulation")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/simulation/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)
SECTIONS = [f"Simulation{i}" for i in range(1, 6)]


def _write_adapter_audit(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    common_rna: set[str] | None = None
    common_adt: set[str] | None = None
    for section in SECTIONS:
        rna = ad.read_h5ad(data_dir / section / "adata_RNA.h5ad")
        adt = ad.read_h5ad(data_dir / section / "adata_ADT.h5ad")
        _adapt_pair(section, rna, adt)
        genes, proteins = set(rna.var_names.astype(str)), set(adt.var_names.astype(str))
        common_rna = genes if common_rna is None else common_rna & genes
        common_adt = proteins if common_adt is None else common_adt & proteins
        counts = pd.Series(rna.obs["spatial_domain"]).value_counts()
        rows.append(
            {
                "section": section,
                "rna_shape": f"{rna.n_obs}x{rna.n_vars}",
                "adt_shape": f"{adt.n_obs}x{adt.n_vars}",
                "paired_spot_order": True,
                "paired_spatial": True,
                "matrix_source": "X",
                "background": int(counts.get("background", 0)),
                "sp1": int(counts.get("sp1", 0)),
                "sp2": int(counts.get("sp2", 0)),
                "sp3": int(counts.get("sp3", 0)),
                "sp4": int(counts.get("sp4", 0)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "input_adaptation_audit.csv", index=False)
    (output_dir / "input_adaptation.json").write_text(
        json.dumps(
            {
                "dataset_path": str(data_dir),
                "sections": SECTIONS,
                "total_spots": 1296 * len(SECTIONS),
                "shared_rna_features": len(common_rna or set()),
                "shared_adt_features": len(common_adt or set()),
                "matrix_source": "X (layers['counts'] intentionally not used)",
                "truth_policy": (
                    "spatial_domain=background for zero spfac rows; otherwise sp1..sp4; "
                    "truth is retained only as output metadata for analysis"
                ),
                "graph_policy": "spatial graph built independently inside each section",
                "barcode_policy": "section-prefixed IDs in combined outputs",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def get_dataset_defaults():
    """Compatibility entry: defaults are owned by training.entry_defaults."""
    return _entry_defaults(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, sections=SECTIONS)


def parse_args(argv: list[str] | None = None):
    defaults = get_dataset_defaults()
    return misar.parse_args(
        argv,
        defaults=defaults,
        dataset_name='Simulation',
        secondary_modality="Protein",
        secondary_name="adt",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    section_info = {
        section: {"dir": section, "sample": section, "stage": section}
        for section in SECTIONS
    }
    run_config = misar.resolve_run_config(
        args, section_info=section_info, dataset_name="Simulation",
        secondary_modality="Protein", secondary_name="adt",
    )
    _write_adapter_audit(Path(args.data_dir), Path(args.output_dir))
    run_multisection_entry(
        args,
        read_pair=read_pair,
        section_info=section_info,
        dataset_name='Simulation',
        secondary_modality="Protein",
        secondary_name="adt",
        run_config=run_config,
        prepare_dataset=prepare_multisection_dataset,
        input_spec=MultisectionSpec(
            section_order=run_config["section_order"], section_info=section_info,
            read_pair=read_pair, secondary_modality="Protein", secondary_name="adt",
            truth_provider=simulation_prepared_truth,
        ),
    )

if __name__ == "__main__":
    main()
