#!/usr/bin/env python3
"""Analyze the three MouseBrain v3 runs with the established standardized protocol."""

from __future__ import annotations

import hashlib
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

from scripts.compare_mousebrain_kmeans_preprocessing import (  # noqa: E402
    METRIC_KS,
    N_INIT,
    MAX_ITER,
    PLOT_KS,
    POINT_SIZE,
    SEED,
    SECTION_DIRS,
    SECTIONS,
    SPATIAL_NEIGHBOR_K,
    MethodData,
    aligned_metadata,
    run_scheme,
    validate,
    write_scheme_summary,
)


RESULT_ROOT = PROJECT_ROOT / "result_v3" / "mousebrain"
DATA_DIR = PROJECT_ROOT / "data" / "dataset_MouseBrain"
VERSIONS = {
    "v1_dense_unidirectional_warmup": "V1 dense unidirectional warmup",
    "v2_bidirectional_sparse_warmup": "V2 bidirectional sparse warmup",
    "v3_bidirectional_sparse_fixed_lc0.1": "V3 bidirectional sparse fixed lc=0.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_version(version: str, display_name: str) -> MethodData:
    version_root = RESULT_ROOT / version
    run_root = version_root / "epochs_200"
    arrays: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    source_paths: list[Path] = []

    for section in SECTIONS:
        embedding_path = (
            run_root / "final_embeddings" / f"{section}_final_embedding.npy"
        )
        rna_path = DATA_DIR / SECTION_DIRS[section] / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()
        finally:
            rna.file.close()
        if len(embedding) != len(names):
            raise ValueError(
                f"{version} {section}: embedding rows {len(embedding)} != {len(names)}"
            )
        arrays.append(embedding)
        sections.extend([section] * len(names))
        barcodes.extend(names.tolist())
        source_paths.append(embedding_path)

    section_array = np.asarray(sections)
    barcode_array = np.asarray(barcodes)
    coordinates, truth = aligned_metadata(
        DATA_DIR,
        section_array,
        barcode_array,
    )
    return MethodData(
        name=display_name,
        embedding=np.vstack(arrays),
        sections=section_array,
        barcodes=barcode_array,
        coords=coordinates,
        truth=truth,
        data_dir=DATA_DIR,
        source_paths=source_paths,
        output_root=version_root / "analysis",
        shared_sources={},
    )


def prune_unretained_joint_directories(data: MethodData) -> list[str]:
    cluster_root = data.output_root / "standardized_embedding" / "clustering"
    deleted: list[str] = []
    for k in METRIC_KS:
        if k in PLOT_KS:
            continue
        target = cluster_root / f"joint_k{k}"
        if target.is_dir():
            if target.resolve().parent != cluster_root.resolve():
                raise ValueError(f"Unsafe pruning target: {target}")
            shutil.rmtree(target)
            deleted.append(str(target))

    expected = {
        *(f"joint_k{k}" for k in PLOT_KS),
        *(f"independent_k{k}" for k in PLOT_KS),
    }
    remaining = {path.name for path in cluster_root.iterdir() if path.is_dir()}
    if remaining != expected:
        raise ValueError(
            f"{data.name}: unexpected retained clustering directories {sorted(remaining)}"
        )

    config_path = data.output_root / "standardized_embedding" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_summary = data.output_root.parent / "epochs_200" / "run_summary.json"
    config.update(
        {
            "analysis_scope": "standardized_embedding_only",
            "training_run_summary": str(run_summary),
            "training_run_summary_sha256": sha256(run_summary),
            "clustering_directories_pruned_at": datetime.now().astimezone().isoformat(),
            "point_size": POINT_SIZE,
        }
    )
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_scheme_summary(data, "standardized_embedding")
    return deleted


def build_comparison(version_data: list[tuple[str, MethodData]]) -> None:
    best_frames: list[pd.DataFrame] = []
    internal_frames: list[pd.DataFrame] = []
    for version, data in version_data:
        metrics_root = data.output_root / "standardized_embedding" / "metrics"
        best = pd.read_csv(metrics_root / "best_clustering_metrics_by_label.csv")
        best.insert(0, "version", version)
        best_frames.append(best)
        internal = pd.read_csv(metrics_root / "clustering_metrics.csv")
        internal.insert(0, "version", version)
        internal_frames.append(internal)

    pd.concat(best_frames, ignore_index=True).to_csv(
        RESULT_ROOT / "standardized_best_clustering_metrics_by_label.csv",
        index=False,
    )
    pd.concat(internal_frames, ignore_index=True).to_csv(
        RESULT_ROOT / "standardized_internal_clustering_metrics.csv",
        index=False,
    )


def main() -> None:
    version_data: list[tuple[str, MethodData]] = []
    deleted: dict[str, list[str]] = {}
    for version, display_name in VERSIONS.items():
        data = load_version(version, display_name)
        validate(data)
        run_scheme(data, "standardized_embedding")
        deleted[version] = prune_unretained_joint_directories(data)
        version_data.append((version, data))

    build_comparison(version_data)
    manifest = {
        "dataset": "MouseBrain",
        "result_root": str(RESULT_ROOT),
        "versions": list(VERSIONS),
        "analysis": "standardized_embedding_only",
        "standardization": "sklearn.preprocessing.StandardScaler",
        "joint_scaler_scope": "all_7866_spots",
        "independent_scaler_scope": "fit_per_section",
        "kmeans": {
            "class": "sklearn.cluster.KMeans",
            "seed": SEED,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
            "metric_k": METRIC_KS,
            "retained_k": PLOT_KS,
        },
        "spatial_plot_point_size": POINT_SIZE,
        "spatial_neighbor_k": SPATIAL_NEIGHBOR_K,
        "deleted_generated_intermediate_directories": deleted,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (RESULT_ROOT / "standardized_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("MOUSEBRAIN_V3_STANDARDIZED_ANALYSIS: PASS")


if __name__ == "__main__":
    main()
