#!/usr/bin/env python3
"""Train spa_mo_model on four paired Mouse Thymus RNA + ADT sections."""

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

from scripts import run_misar_seq as misar
from data_io.adapted import (
    THYMUS_SECTIONS as SECTIONS, ADT_ORDER, ADT_RAW, ADT_RAW_OTHER,
    canonical_thymus_coords as _canonical_coords,
    adapt_thymus_pair as _adapt_pair, read_thymus_pair as read_pair,
)


DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Mouse_Thymus")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/mouse_thymus/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)

def _write_adapter_audit(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    common_rna: set[str] | None = None
    for section in SECTIONS:
        rna = ad.read_h5ad(data_dir / section / "adata_RNA.h5ad", backed="r")
        adt = ad.read_h5ad(data_dir / section / "adata_ADT.h5ad", backed="r")
        try:
            genes = set(rna.var_names.astype(str))
            common_rna = genes if common_rna is None else common_rna & genes
            obs_coords = rna.obs.loc[:, ["x", "y"]].to_numpy()
            rows.append(
                {
                    "section": section,
                    "rna_shape": f"{rna.n_obs}x{rna.n_vars}",
                    "adt_shape_raw": f"{adt.n_obs}x{adt.n_vars}",
                    "paired_spot_order": list(rna.obs_names) == list(adt.obs_names),
                    "rna_var_unique": bool(rna.var_names.is_unique),
                    "adt_shared_after_mapping": len(ADT_ORDER),
                    "rna_obs_vs_obsm_spatial_equal": bool(
                        np.array_equal(obs_coords, np.asarray(rna.obsm["spatial"]))
                    ),
                    "canonical_coordinate_source": "RNA obs[x,y]",
                }
            )
        finally:
            rna.file.close()
            adt.file.close()
    pd.DataFrame(rows).to_csv(output_dir / "input_adaptation_audit.csv", index=False)
    (output_dir / "input_adaptation.json").write_text(
        json.dumps(
            {
                "sections": SECTIONS,
                "total_spots": 17824,
                "shared_rna_features": len(common_rna or set()),
                "shared_adt_features": ADT_ORDER,
                "coordinate_policy": "RNA obs[x,y] copied to RNA/ADT obsm['spatial']",
                "barcode_policy": "section-prefixed IDs in combined outputs",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def get_dataset_defaults():
    """Dataset-specific values over the shared parser defaults."""
    return {
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "section_order": ",".join(SECTIONS),
        "train": True,
        "epochs": 200,
        "seed": 42,
        "device": "cuda",
        "hvg_num": 3000,
        "hvg_num_adt": len(ADT_ORDER),
        "lambda_contrast": 0.1,
        "candidate_backend": "faiss_ivf",
        "faiss_nlist": 256,
        "faiss_nprobe": 32,
        "faiss_train_sample_size": 10000,
        "faiss_query_batch_size": 2048,
        "initial_modality_candidate_k": 100,
        "candidate_k": 200,
        "attention_topk": 10,
        "spatial_knn_k": 10,
        "graphsage_edge_batch_size": 100000,
        "training_loss_only": True,
        "decoder_chunk_size": 2048,
        "ot_attention_source_chunk_size": 1024,
        "checkpoint_ot_attention": True,
        "checkpoint_encoder_fusion": True,
        "checkpoint_decoder_chunks": True,
        "checkpoint_graph_encoder": True,
        "amp_dtype": "bf16",
        "cache_spatial_graphs": True,
        "save_candidate_qc": True,
        "save_outputs": True,
        "save_embeddings": True,
        "save_ot_prior_topk": True,
        "log_cuda_memory": True,
    }


def parse_args(argv: list[str] | None = None):
    defaults = get_dataset_defaults()
    return misar.parse_args(
        argv,
        defaults=defaults,
        dataset_name='Mouse Thymus',
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
        args, section_info=section_info, dataset_name="Mouse Thymus",
        secondary_modality="Protein", secondary_name="adt",
    )
    _write_adapter_audit(Path(args.data_dir), Path(args.output_dir))
    misar.run_misar_pipeline(
        args,
        read_pair=read_pair,
        section_info=section_info,
        dataset_name='Mouse Thymus',
        secondary_modality="Protein",
        secondary_name="adt",
        run_config=run_config,
    )

if __name__ == "__main__":
    main()
