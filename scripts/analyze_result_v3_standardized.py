#!/usr/bin/env python3
"""Run the historical standardized-embedding analyses for result_v3 datasets."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import compare_human_lymph_node_kmeans_preprocessing as hln
import compare_misar_seq_kmeans_preprocessing as misar
import compare_mouse_spleen_kmeans_preprocessing as spleen
import compare_mouse_thymus_kmeans_preprocessing as thymus
import compare_simulation_kmeans_preprocessing as simulation


ROOT = Path("/home/hujinlan/spa_mo_model")
RESULT_ROOT = ROOT / "result_v3"
RUN_NAME = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
SCHEME = "standardized_embedding"
SCRIPT_PATH = Path(__file__).resolve()


def _run(dataset: str) -> Path:
    return RESULT_ROOT / dataset / RUN_NAME


def _paired_rna_adt_data(module, dataset: str):
    run = _run(dataset)
    data_dir = ROOT / "data" / {
        "human_lymph_node": "Human_Lymph_Node",
        "mouse_spleen": "Mouse_Spleen",
    }[dataset]
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in module.SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        index_path = run / f"selected_spot_indices_{section}.npy"
        spatial_path = run / f"spatial_{section}.npy"
        rna_path = data_dir / section / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        indices = np.load(index_path).astype(int)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
        finally:
            rna.file.close()
        spatial = np.load(spatial_path)[:, :2]
        if len(embedding) != len(indices) or len(spatial) != len(indices):
            raise ValueError(f"{dataset}/{section}: saved row counts do not align")
        arrays.append(embedding)
        sections.extend([section] * len(indices))
        barcodes.extend(names.tolist())
        coords.append(spatial)
        sources.extend([embedding_path, index_path, spatial_path, rna_path])
    return module.MethodData(
        "spa_mo_model_v3",
        np.vstack(arrays),
        np.asarray(sections, dtype=str),
        np.asarray(barcodes, dtype=str),
        np.vstack(coords),
        data_dir,
        sources,
        run / "analysis",
        {},
    )


def _misar_data():
    run = _run("misar_seq")
    data_dir = ROOT / "data" / "MISAR-seq"
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in misar.SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"MISAR-seq/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, metadata_path])
    section_array = np.asarray(sections, dtype=str)
    barcode_array = np.asarray(barcodes, dtype=str)
    coords, truth = misar.aligned_metadata(data_dir, section_array, barcode_array)
    return misar.MethodData(
        "spa_mo_model_v3",
        np.vstack(arrays),
        section_array,
        barcode_array,
        coords,
        truth,
        data_dir,
        sources,
        run / "analysis",
        {},
    )


def _metadata_data(module, dataset: str, data_name: str):
    run = _run(dataset)
    data_dir = ROOT / "data" / data_name
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in module.SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
        names = metadata["obs_name"].astype(str).to_numpy()
        spatial = metadata[["spatial_x", "spatial_y"]].to_numpy(float)
        if len(embedding) != len(names) or len(spatial) != len(names):
            raise ValueError(f"{dataset}/{section}: saved row counts do not align")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        coords.append(spatial)
        sources.extend([embedding_path, metadata_path])
    return (
        run,
        data_dir,
        np.vstack(arrays),
        np.asarray(sections, dtype=str),
        np.asarray(barcodes, dtype=str),
        np.vstack(coords),
        sources,
    )


def _thymus_data():
    run, data_dir, embedding, sections, barcodes, coords, sources = _metadata_data(
        thymus, "mouse_thymus", "Mouse_Thymus"
    )
    return thymus.MethodData(
        "spa_mo_model_v3",
        embedding,
        sections,
        barcodes,
        coords,
        data_dir,
        sources,
        run / "analysis",
        {},
    )


def _simulation_data():
    run = _run("simulation")
    data_dir = ROOT / "data" / "Simulation"
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    truth: list[str] = []
    coords: list[np.ndarray] = []
    sources: list[Path] = []
    for section in simulation.SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        metadata_path = run / f"obs_metadata_{section}.csv"
        embedding = np.load(embedding_path)
        metadata = pd.read_csv(metadata_path)
        names = metadata["obs_name"].astype(str).to_numpy()
        if len(embedding) != len(names):
            raise ValueError(f"Simulation/{section}: embedding/metadata mismatch")
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        truth.extend(metadata["spatial_domain"].astype(str).tolist())
        coords.append(metadata[["spatial_x", "spatial_y"]].to_numpy(float))
        sources.extend([embedding_path, metadata_path])
    return simulation.MethodData(
        "spa_mo_model_v3",
        np.vstack(arrays),
        np.asarray(sections, dtype=str),
        np.asarray(barcodes, dtype=str),
        np.asarray(truth, dtype=str),
        np.vstack(coords),
        data_dir,
        sources,
        run / "analysis",
        {},
    )


def _prune(output: Path, metric_ks: list[int], retained: list[int]) -> list[str]:
    cluster_root = output / SCHEME / "clustering"
    removed: list[str] = []
    for mode in ("joint", "independent"):
        for k in metric_ks:
            if k in retained:
                continue
            target = cluster_root / f"{mode}_k{k}"
            if target.is_dir():
                if target.resolve().parent != cluster_root.resolve():
                    raise ValueError(f"unsafe prune target: {target}")
                shutil.rmtree(target)
                removed.append(str(target))
    return removed


def _annotate(output: Path, dataset: str, removed: list[str]) -> dict:
    config_path = output / SCHEME / "config.json"
    config = json.loads(config_path.read_text())
    retained = (
        config.get("retained_clustering_k_values")
        or config.get("plot_k_values")
        or config.get("plot_and_independent_k_values")
        or config.get("k_values")
    )
    removed_k = sorted(
        {
            int(Path(path).name.rsplit("_k", 1)[1])
            for path in removed
            if "_k" in Path(path).name
        }
    )
    config.update(
        {
            "training_seed": 42,
            "model_variant": "bidirectional_sparse_uot_fixed_lc0.1",
            "graphsage_self_path_mode": "no_self_linear",
            "dynamic_candidate_source": "final",
            "dynamic_ot_cost": (
                "0.8 * semantic cosine cost + 0.2 * local-context cosine cost"
            ),
            "analysis_entrypoint": str(SCRIPT_PATH),
            "retained_clustering_k_values": retained,
            "deleted_clustering_k_values": removed_k,
            "metrics_k_values_without_saved_label_directories": removed_k,
            "removed_generated_intermediate_directories": removed,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return {
        "dataset": dataset,
        "training_run": str(output.parent),
        "analysis": str(output / SCHEME),
        "n_obs": int(config["n_obs"]),
        "retained_k": retained,
        "removed_intermediate_directory_count": len(removed),
    }


def main() -> None:
    manifests: list[dict] = []

    hln_data = _paired_rna_adt_data(hln, "human_lymph_node")
    hln.validate(hln_data)
    hln.run_scheme(hln_data, SCHEME)
    removed = _prune(hln_data.output_root, hln.METRIC_KS, hln.PLOT_KS)
    manifests.append(_annotate(hln_data.output_root, "Human_Lymph_Node", removed))
    print("Human_Lymph_Node standardized analysis: PASS", flush=True)

    misar_data = _misar_data()
    misar.validate(misar_data)
    misar.run_scheme(misar_data, SCHEME)
    removed = _prune(misar_data.output_root, misar.METRIC_KS, misar.PLOT_KS)
    manifests.append(_annotate(misar_data.output_root, "MISAR-seq", removed))
    print("MISAR-seq standardized analysis: PASS", flush=True)

    spleen_data = _paired_rna_adt_data(spleen, "mouse_spleen")
    spleen.validate(spleen_data)
    spleen.run_scheme(spleen_data, SCHEME)
    removed = _prune(spleen_data.output_root, spleen.METRIC_KS, spleen.PLOT_KS)
    manifests.append(_annotate(spleen_data.output_root, "Mouse_Spleen", removed))
    print("Mouse_Spleen standardized analysis: PASS", flush=True)

    thymus_data = _thymus_data()
    thymus.validate(thymus_data)
    sample = pd.read_csv(
        ROOT
        / "results/mouse_thymus_preprocessing_comparison/shared_metrics/asw_sample.csv"
    )
    sample_source = (
        ROOT
        / "results/mouse_thymus_preprocessing_comparison/shared_metrics/asw_sample.csv"
    )
    sample_target = thymus_data.output_root / "shared_metrics" / "asw_sample.csv"
    sample_target.parent.mkdir(parents=True)
    shutil.copy2(sample_source, sample_target)
    thymus.run_scheme(thymus_data, SCHEME, sample)
    removed = _prune(thymus_data.output_root, thymus.METRIC_KS, thymus.PLOT_KS)
    manifests.append(_annotate(thymus_data.output_root, "Mouse_Thymus", removed))
    print("Mouse_Thymus standardized analysis: PASS", flush=True)

    simulation_data = _simulation_data()
    simulation.validate_method(simulation_data)
    simulation.run_scheme(simulation_data, SCHEME)
    manifests.append(_annotate(simulation_data.output_root, "Simulation", []))
    print("Simulation standardized analysis: PASS", flush=True)

    manifest = {
        "analysis": "standardized_embedding_only",
        "training_seed": 42,
        "model_variant": "bidirectional_sparse_uot_fixed_lc0.1",
        "graphsage_self_path_mode": "no_self_linear",
        "dynamic_candidate_source": "final",
        "dynamic_ot_cost": (
            "0.8 * semantic cosine cost + 0.2 * local-context cosine cost"
        ),
        "datasets": manifests,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    path = RESULT_ROOT / "five_dataset_standardized_analysis_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("RESULT_V3_FIVE_DATASET_STANDARDIZED_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
