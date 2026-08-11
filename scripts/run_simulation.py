#!/usr/bin/env python3
"""Train spa_mo_model jointly on five paired Simulation RNA + ADT sections."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ORIGINAL = Path(__file__).with_name("run_misar_seq.py")
DATA_DIR = Path("/home/hujinlan/spa_mo_model/data/Simulation")
OUTPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/result_v4/simulation/"
    "bidirectional_sparse_uot_fixed_lc0.1_seed42"
)
SECTIONS = [f"Simulation{i}" for i in range(1, 6)]


def _spatial_domain(spfac: np.ndarray) -> np.ndarray:
    active = np.asarray(spfac).sum(axis=1) > 0
    labels = np.full(len(spfac), "background", dtype=object)
    labels[active] = np.asarray(
        [f"sp{i + 1}" for i in np.asarray(spfac)[active].argmax(axis=1)]
    )
    return labels


def _adapt_pair(
    section: str, rna: ad.AnnData, adt: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    if list(rna.obs_names.astype(str)) != list(adt.obs_names.astype(str)):
        raise ValueError(f"{section}: RNA and ADT spot order differs.")
    if rna.shape != (1296, 1000) or adt.shape != (1296, 100):
        raise ValueError(f"{section}: unexpected RNA/ADT shapes {rna.shape}/{adt.shape}.")
    for name, obj in (("RNA", rna), ("ADT", adt)):
        values = np.asarray(obj.X)
        if not np.isfinite(values).all() or values.min() < 0:
            raise ValueError(f"{section} {name}: X must be finite and nonnegative.")
        if not np.allclose(values, np.rint(values)):
            raise ValueError(f"{section} {name}: X must contain integer-valued observations.")
    coords = np.asarray(rna.obsm["spatial"], dtype=np.float32)
    if coords.shape != (rna.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError(f"{section}: invalid RNA obsm['spatial'].")
    if not np.array_equal(coords, np.asarray(adt.obsm["spatial"])):
        raise ValueError(f"{section}: RNA and ADT spatial coordinates differ.")
    spfac = np.asarray(rna.obsm["spfac"], dtype=np.float32)
    if spfac.shape != (rna.n_obs, 4):
        raise ValueError(f"{section}: expected four spatial factors.")
    rna_ns = np.asarray(rna.obsm["nsfac"], dtype=np.float32)
    adt_ns = np.asarray(adt.obsm["nsfac"], dtype=np.float32)
    for obj in (rna, adt):
        obj.obsm["spatial"] = coords.copy()
        obj.obs["x"] = coords[:, 0]
        obj.obs["y"] = coords[:, 1]
        obj.obs["spatial_domain"] = _spatial_domain(spfac)
        obj.obs["original_barcode"] = obj.obs_names.astype(str)
        for i in range(4):
            obj.obs[f"spfac_{i + 1}"] = spfac[:, i]
    for i in range(3):
        rna.obs[f"rna_nsfac_{i + 1}"] = rna_ns[:, i]
        rna.obs[f"adt_nsfac_{i + 1}"] = adt_ns[:, i]
        adt.obs[f"rna_nsfac_{i + 1}"] = rna_ns[:, i]
        adt.obs[f"adt_nsfac_{i + 1}"] = adt_ns[:, i]
    return rna, adt


def _load_pipeline() -> dict:
    source = ORIGINAL.read_text(encoding="utf-8")
    source = source.replace("MISAR-seq", "Simulation")
    source = source.replace("ATAC", "Protein").replace("atac", "adt")
    namespace = {"__file__": str(ORIGINAL), "__name__": "simulation_spa_mo_pipeline"}
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
    rows: list[dict] = []
    common_rna: set[str] | None = None
    common_adt: set[str] | None = None
    for section in SECTIONS:
        rna = ad.read_h5ad(DATA_DIR / section / "adata_RNA.h5ad")
        adt = ad.read_h5ad(DATA_DIR / section / "adata_ADT.h5ad")
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
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "input_adaptation_audit.csv", index=False)
    (OUTPUT_DIR / "input_adaptation.json").write_text(
        json.dumps(
            {
                "dataset_path": str(DATA_DIR),
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
        "--hvg_num", "1000",
        "--hvg_num_adt", "100",
        "--lambda_contrast", "0.1",
        "--ot_prior_mode", "candidate_sparse",
        "--bidirectional_ot_attention",
        "--candidate_backend", "faiss_flat",
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
    namespace["run_misar_pipeline"](namespace["parse_args"]())


if __name__ == "__main__":
    main()
