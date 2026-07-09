#!/usr/bin/env python3
"""Train spa_mo_model on four paired Mouse Thymus RNA + ADT sections."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ORIGINAL = Path(__file__).with_name("run_misar_seq.py")
DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Mouse_Thymus")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/mouse_thymus/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)
SECTIONS = [f"Mouse_Thymus{i}" for i in range(1, 5)]
ADT_ORDER = [
    "CD5",
    "CD68",
    "CD4",
    "CD29",
    "CD8a",
    "CD3",
    "CD44",
    "CD90_2",
    "CD11c",
    "CD31",
    "F4_80",
    "CD45R_B220",
    "Rat_IgG2a",
    "CD11b",
    "CD19",
    "Mouse_IgG2a",
    "CD169",
]

ADT_RAW = {
    "Mouse_Thymus1": {
        "CD5": "Mouse-CD5",
        "CD68": "Mouse-CD68",
        "CD4": "Mouse-CD4",
        "CD29": "Mouse-CD29",
        "CD8a": "Mouse-CD8a",
        "CD3": "Mouse-CD3",
        "CD44": "Mouse-CD44",
        "CD90_2": "Mouse-CD90-2",
        "CD11c": "Mouse-CD11c",
        "CD31": "Mouse-CD31-Pecam",
        "F4_80": "Mouse-F480",
        "CD45R_B220": "Mouse-CD45R-B220",
        "Rat_IgG2a": "Rat-IgG2a",
        "CD11b": "Ms-Hu-CD11b",
        "CD19": "Mouse-CD19",
        "Mouse_IgG2a": "Mouse-IgG2a",
        "CD169": "Mouse-CD169",
    },
}
ADT_RAW_OTHER = {
    "CD5": "mouse_CD5",
    "CD68": "mouse_CD68",
    "CD4": "mouse_CD4",
    "CD29": "mouse_rat_CD29",
    "CD8a": "mouse_CD8a",
    "CD3": "mouse_CD3",
    "CD44": "mouse_human_CD44",
    "CD90_2": "mouse_CD90_2",
    "CD11c": "mouse_CD11c",
    "CD31": "mouse_CD31",
    "F4_80": "mouse_F4_80",
    "CD45R_B220": "mouse_human_CD45R_B220",
    "Rat_IgG2a": "Rat_IgG2a",
    "CD11b": "mouse_human_CD11b",
    "CD19": "mouse_CD19",
    "Mouse_IgG2a": "Mouse_IgG2a",
    "CD169": "mouse_CD169_Siglec-1",
}
for _section in SECTIONS[1:]:
    ADT_RAW[_section] = ADT_RAW_OTHER


def _canonical_coords(rna: ad.AnnData) -> np.ndarray:
    if not {"x", "y"}.issubset(rna.obs.columns):
        raise KeyError("Mouse Thymus RNA requires obs['x'] and obs['y'].")
    coords = rna.obs.loc[:, ["x", "y"]].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or np.unique(coords, axis=0).shape[0] != rna.n_obs:
        raise ValueError("Mouse Thymus RNA coordinates must be finite and unique.")
    return coords


def _adapt_pair(section: str, rna: ad.AnnData, adt: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    if list(rna.obs_names.astype(str)) != list(adt.obs_names.astype(str)):
        raise ValueError(f"{section}: RNA and ADT spot order differs.")
    if not rna.var_names.is_unique:
        raise ValueError(f"{section}: RNA var_names must be unique.")
    coords = _canonical_coords(rna)
    rna.obsm["spatial"] = coords.copy()
    adt.obsm["spatial"] = coords.copy()
    for obj in (rna, adt):
        obj.obs["x"] = coords[:, 0]
        obj.obs["y"] = coords[:, 1]

    raw_names = pd.Index(adt.var_names.astype(str))
    raw_by_marker = ADT_RAW[section]
    missing = [raw_by_marker[name] for name in ADT_ORDER if raw_by_marker[name] not in raw_names]
    if missing:
        raise KeyError(f"{section}: missing mapped ADT markers: {missing}")
    positions = [int(raw_names.get_loc(raw_by_marker[name])) for name in ADT_ORDER]
    adt = adt[:, positions].copy()
    adt.var["adt_name_original"] = [raw_by_marker[name] for name in ADT_ORDER]
    adt.var["adt_marker"] = ADT_ORDER
    adt.var_names = pd.Index(ADT_ORDER, name="adt_marker")
    return rna, adt


def _load_pipeline() -> dict:
    source = ORIGINAL.read_text(encoding="utf-8")
    source = source.replace("MISAR-seq", "Mouse Thymus")
    source = source.replace("ATAC", "Protein").replace("atac", "adt")
    namespace = {"__file__": str(ORIGINAL), "__name__": "mouse_thymus_spa_mo_pipeline"}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    namespace["SECTION_INFO"] = {
        section: {"dir": section, "sample": section, "stage": section}
        for section in SECTIONS
    }
    namespace["DEFAULT_SECTION_ORDER"] = list(SECTIONS)

    def read_pair(data_dir: Path, section: str):
        folder = data_dir / section
        rna = ad.read_h5ad(folder / "adata_RNA.h5ad")
        adt = ad.read_h5ad(folder / "adata_ADT.h5ad")
        return _adapt_pair(section, rna, adt)

    namespace["read_backed_pair"] = read_pair
    return namespace


def _write_adapter_audit() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    common_rna: set[str] | None = None
    for section in SECTIONS:
        rna = ad.read_h5ad(DATA_DIR / section / "adata_RNA.h5ad", backed="r")
        adt = ad.read_h5ad(DATA_DIR / section / "adata_ADT.h5ad", backed="r")
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
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "input_adaptation_audit.csv", index=False)
    (OUTPUT_DIR / "input_adaptation.json").write_text(
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


def main() -> None:
    _write_adapter_audit()
    namespace = _load_pipeline()
    sys.argv[1:1] = [
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(OUTPUT_DIR),
        "--section_order", ",".join(SECTIONS),
        "--train",
        "--epochs", "200",
        "--seed", "42",
        "--device", "cuda",
        "--hvg_num", "3000",
        "--hvg_num_adt", str(len(ADT_ORDER)),
        "--lambda_contrast", "0.1",
        "--ot_prior_mode", "candidate_sparse",
        "--bidirectional_ot_attention",
        "--candidate_backend", "faiss_ivf",
        "--faiss_nlist", "256",
        "--faiss_nprobe", "32",
        "--faiss_train_sample_size", "10000",
        "--faiss_query_batch_size", "2048",
        "--initial_modality_candidate_k", "100",
        "--candidate_k", "200",
        "--attention_topk", "10",
        "--spatial_knn_k", "10",
        "--graphsage_edge_batch_size", "100000",
        "--training_loss_only",
        "--decoder_chunk_size", "2048",
        "--ot_attention_source_chunk_size", "1024",
        "--checkpoint_ot_attention",
        "--checkpoint_encoder_fusion",
        "--checkpoint_decoder_chunks",
        "--checkpoint_graph_encoder",
        "--amp_dtype", "bf16",
        "--cache_spatial_graphs",
        "--save_candidate_qc",
        "--save_outputs",
        "--save_embeddings",
        "--save_ot_prior_topk",
        "--log_cuda_memory",
    ]
    args = namespace["parse_args"]()
    namespace["run_misar_pipeline"](args)


if __name__ == "__main__":
    main()
