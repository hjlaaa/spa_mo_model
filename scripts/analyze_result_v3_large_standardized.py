#!/usr/bin/env python3
"""Run the historical standardized-embedding analyses for result_v3 large datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import compare_crc_stereocite_kmeans_preprocessing as crc
import compare_spatch_kmeans_preprocessing as spatch


VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
RESULT_ROOT = PROJECT_ROOT / "result_v3"


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def annotate_config(config_path: Path, training_run: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = training_run / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dynamic_source = summary.get("dynamic_candidate_source", "final")
    context_gate_enabled = bool(
        summary.get("attention_context_gate_enabled", False)
    )
    if dynamic_source == "final" and not context_gate_enabled:
        model_variant = "v3_bidirectional_sparse_uot_fixed_lc0.1"
    elif dynamic_source == "fused" and not context_gate_enabled:
        model_variant = "fused_dynamic_ot_without_microenvironment_attention_gate"
    else:
        model_variant = "microenvironment_aware_bidirectional_sparse_uot_fixed_lc0.1"
    payload.update(
        {
            "training_run": str(training_run),
            "training_seed": 42,
            "model_variant": model_variant,
            "graphsage_self_path_mode": "no_self_linear",
            "dynamic_candidate_source": dynamic_source,
            "dynamic_semantic_source": dynamic_source,
            "dynamic_context_source": f"{dynamic_source}_spatial_context",
            "attention_context_source": (
                "fused_spatial_context" if context_gate_enabled else None
            ),
            "attention_context_gate_enabled": context_gate_enabled,
            "dynamic_ot_cost": (
                "0.8 * semantic cosine cost + "
                "0.2 * local-context cosine cost"
            ),
            "analysis_generation_script": str(Path(__file__).resolve()),
        }
    )
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def analyze_crc() -> dict[str, object]:
    run = RESULT_ROOT / "crc_stereocite" / VARIANT
    summary_path = run / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    output_root = run / "analysis"
    completion = output_root / "standardized_embedding" / "config.json"
    if completion.exists():
        print("CRC standardized analysis already complete", flush=True)
        config = json.loads(completion.read_text(encoding="utf-8"))
        return {
            "dataset": "CRC_Stereo-CITE-seq",
            "training_run": str(run),
            "analysis": str(completion.parent),
            "n_obs": int(config["n_obs"]),
            "retained_k": list(config["k_values"]),
        }

    data_dir = PROJECT_ROOT / "data" / "CRC_Stereo-CITE-seq"
    arrays: list[np.ndarray] = []
    section_values: list[str] = []
    barcode_values: list[str] = []
    source_paths: list[Path] = []
    for section in crc.SECTIONS:
        short = crc.SHORT_NAMES[section]
        embedding_path = run / f"final_embeddings_{short}.npy"
        indices_path = run / f"selected_spot_indices_{short}.npy"
        spatial_path = run / f"spatial_{short}.npy"
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        saved_coords = np.load(spatial_path)[:, :2]
        rna = ad.read_h5ad(data_dir / section / "adata_RNA.h5ad", backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
            source_coords = np.asarray(rna.obsm["spatial"])[indices, :2]
        finally:
            rna.file.close()
        if (
            len(embedding) != len(indices)
            or len(np.unique(indices)) != len(indices)
            or not np.allclose(saved_coords, source_coords)
        ):
            raise ValueError(f"CRC {section}: embedding/index/spatial mismatch")
        arrays.append(embedding)
        section_values.extend([section] * len(indices))
        barcode_values.extend(names.tolist())
        source_paths.extend([embedding_path, indices_path, spatial_path])

    sections = np.asarray(section_values, dtype=str)
    barcodes = np.asarray(barcode_values, dtype=str)
    data = crc.MethodData(
        name="spa_mo_model",
        embedding=np.vstack(arrays),
        sections=sections,
        barcodes=barcodes,
        coords=crc.aligned_coords(data_dir, sections, barcodes),
        data_dir=data_dir,
        source_paths=source_paths,
        output_root=output_root,
        shared_sources={},
    )
    crc.validate(data)
    shared_old = (
        PROJECT_ROOT
        / "results"
        / "crc_stereocite_preprocessing_comparison"
        / "shared_metrics"
    )
    shared_new = output_root / "shared_metrics"
    shared_new.mkdir(parents=True)
    for name in (
        "asw_sample.csv",
        "plot_sample.csv",
        "sampling_manifest.json",
        "spatial_graph_manifest.csv",
        "spatial_neighbors_CRC_003_bin20.npz",
        "spatial_neighbors_CRC_006_bin20.npz",
    ):
        copy_file(shared_old / name, shared_new / name)
    asw_sample = pd.read_csv(shared_new / "asw_sample.csv", dtype=str)
    plot_sample = pd.read_csv(shared_new / "plot_sample.csv", dtype=str)
    graphs = {
        section: np.load(shared_new / f"spatial_neighbors_{section}.npz")[
            "neighbor_indices"
        ]
        for section in crc.SECTIONS
    }
    crc.run_scheme(
        data,
        "standardized_embedding",
        asw_sample,
        plot_sample,
        graphs,
    )
    config_path = output_root / "standardized_embedding" / "config.json"
    annotate_config(config_path, run)
    print("CRC standardized analysis: PASS", flush=True)
    return {
        "dataset": "CRC_Stereo-CITE-seq",
        "training_run": str(run),
        "analysis": str(config_path.parent),
        "n_obs": crc.N_OBS,
        "retained_k": crc.KS,
    }


def analyze_spatch() -> dict[str, object]:
    run = RESULT_ROOT / "spatch" / VARIANT
    summary_path = run / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    output_root = run / "analysis"
    completion = (
        output_root
        / "standardized_embedding"
        / "analysis_completion_manifest.json"
    )
    if completion.exists():
        print("spatch standardized analysis already complete", flush=True)
        return {
            "dataset": "spatch",
            "training_run": str(run),
            "analysis": str(completion.parent),
            "n_obs": spatch.N_OBS,
            "retained_k": spatch.RETAIN_KS,
        }

    spec = spatch.MethodSpec(
        key=f"spa_{RESULT_ROOT.name}",
        name="spa_mo_model",
        run_dir=run,
        data_dir=PROJECT_ROOT / "data" / "spatch",
        embedding_paths={
            section: run / f"final_embeddings_{section}.npy"
            for section in spatch.SECTIONS
        },
        metadata_path=run / "spot_metadata.csv.gz",
        id_column="block_id",
        output_root=output_root,
        old_analysis=run / "analysis",
        batch_metrics_source=(
            PROJECT_ROOT
            / "results"
            / "spatch_preprocessing_comparison"
            / "shared_metrics"
            / "batch_correction_metrics.csv"
        ),
    )
    metadata = spatch.load_metadata(spec)
    spatch.validate_metadata(spec, metadata)
    embedding = spatch.load_embedding(spec)
    if len(embedding) != spatch.N_OBS or not np.isfinite(embedding).all():
        raise ValueError("spatch: invalid result_v3 embedding")

    shared_old = (
        PROJECT_ROOT
        / "results"
        / "spatch_preprocessing_comparison"
        / "shared_metrics"
    )
    shared_new = output_root / "shared_metrics"
    shared_new.mkdir(parents=True)
    for name in (
        "asw_sample_identities.csv",
        "spatial_graph_manifest.csv",
        "spatial_neighbors_section1.npz",
        "spatial_neighbors_section2.npz",
    ):
        copy_file(shared_old / name, shared_new / name)
    graphs = {
        section: np.load(shared_new / f"spatial_neighbors_{section}.npz")[
            "neighbor_indices"
        ]
        for section in spatch.SECTIONS
    }
    spatch.run_scheme(
        spec,
        metadata,
        embedding,
        graphs,
        "standardized_embedding",
    )
    config_path = output_root / "standardized_embedding" / "config.json"
    annotate_config(config_path, run)
    print("spatch standardized analysis: PASS", flush=True)
    return {
        "dataset": "spatch",
        "training_run": str(run),
        "analysis": str(config_path.parent),
        "n_obs": spatch.N_OBS,
        "retained_k": spatch.RETAIN_KS,
    }


def main() -> None:
    global RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Result root containing crc_stereocite and spatch runs.",
    )
    args = parser.parse_args()
    RESULT_ROOT = args.result_root.resolve()
    records = [analyze_crc(), analyze_spatch()]
    crc_summary = json.loads(
        (RESULT_ROOT / "crc_stereocite" / VARIANT / "run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    dynamic_source = crc_summary.get("dynamic_candidate_source", "final")
    context_gate_enabled = bool(
        crc_summary.get("attention_context_gate_enabled", False)
    )
    if dynamic_source == "final" and not context_gate_enabled:
        model_variant = "v3_bidirectional_sparse_uot_fixed_lc0.1"
    elif dynamic_source == "fused" and not context_gate_enabled:
        model_variant = "fused_dynamic_ot_without_microenvironment_attention_gate"
    else:
        model_variant = "microenvironment_aware_bidirectional_sparse_uot_fixed_lc0.1"
    manifest = {
        "analysis": "standardized_embedding_only",
        "training_seed": 42,
        "model_variant": model_variant,
        "graphsage_self_path_mode": "no_self_linear",
        "dynamic_candidate_source": dynamic_source,
        "dynamic_semantic_source": dynamic_source,
        "dynamic_context_source": f"{dynamic_source}_spatial_context",
        "attention_context_source": (
            "fused_spatial_context" if context_gate_enabled else None
        ),
        "attention_context_gate_enabled": context_gate_enabled,
        "dynamic_ot_cost": (
            "0.8 * semantic cosine cost + 0.2 * local-context cosine cost"
        ),
        "datasets": records,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    path = RESULT_ROOT / "large_dataset_standardized_analysis_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{RESULT_ROOT.name.upper()}_LARGE_STANDARDIZED_ANALYSIS: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
