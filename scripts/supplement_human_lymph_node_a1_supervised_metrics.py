#!/usr/bin/env python3
"""Supplement Human_Lymph_Node A1 supervised clustering metrics.

This script leaves the original unsupervised analyses unchanged.  It recreates
the exact standardized KMeans spaces for five methods, aligns the A1 cluster
assignments to ``annotation.csv`` by barcode, and writes supplementary external
metrics plus auditable per-spot labels into each method's existing
``standardized_embedding/metrics`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


A1 = "Human_Lymph_Node_A1"
D1 = "Human_Lymph_Node_D1"
SECTIONS = [A1, D1]
K_VALUES = list(range(2, 13))
RETAINED_K_VALUES = [5, 8, 10, 12]
SEED = 0
N_INIT = 20
MAX_ITER = 300
LABEL_KEY = "manual-anno"


@dataclass(frozen=True)
class LoadedData:
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    output_root: Path
    annotation_path: Path
    loader: Callable[[], LoadedData]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spa_mo_model() -> LoadedData:
    run = Path(
        "/home/hujinlan/spa_mo_model/result_v6/human_lymph_node/"
        "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    )
    data_dir = Path("/home/hujinlan/spa_mo_model/data/Human_Lymph_Node")
    embeddings: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in SECTIONS:
        embedding_path = run / f"final_embeddings_{section}.npy"
        indices_path = run / f"selected_spot_indices_{section}.npy"
        rna_path = data_dir / section / "adata_RNA.h5ad"
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        rna = ad.read_h5ad(rna_path, backed="r")
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
        finally:
            rna.file.close()
        if embedding.shape[0] != len(indices):
            raise ValueError(f"{section}: embedding/index row mismatch")
        embeddings.append(np.asarray(embedding))
        sections.extend([section] * len(indices))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, indices_path, rna_path])
    return LoadedData(
        np.vstack(embeddings),
        np.asarray(sections, dtype=str),
        np.asarray(barcodes, dtype=str),
        tuple(sources),
    )


def load_cosie() -> LoadedData:
    path = Path(
        "/home/hujinlan/cosie_runs/human_lymph_node_cosie_rna_adt_full/"
        "analysis/tables/embeddings_with_metadata.csv"
    )
    table = pd.read_csv(path)
    embedding_columns = [column for column in table if column.startswith("COSIE")]
    barcode_column = "obs_name.1" if "obs_name.1" in table else "obs_name"
    return LoadedData(
        table[embedding_columns].to_numpy(float),
        table["section"].astype(str).to_numpy(),
        table[barcode_column].astype(str).to_numpy(),
        (path,),
    )


def load_present() -> LoadedData:
    run = Path(
        "/home/hujinlan/PRESENT/result/human_lymph_node/source_default_rna_adt"
    )
    embedding_path = run / "training/embeddings.npy"
    adata_path = run / "training/adata_output.h5ad"
    embedding = np.load(embedding_path)
    output = ad.read_h5ad(adata_path, backed="r")
    try:
        sections = output.obs["section"].astype(str).to_numpy()
        barcodes = output.obs["original_barcode"].astype(str).to_numpy()
    finally:
        output.file.close()
    return LoadedData(
        np.asarray(embedding), sections, barcodes, (embedding_path, adata_path)
    )


def load_spamosaic() -> LoadedData:
    path = Path(
        "/home/hujinlan/SpaMosaic-dev/runs/human_lymph_node_spamosaic/"
        "human_lymph_node_spamosaic_embeddings.h5ad"
    )
    output = ad.read_h5ad(path)
    return LoadedData(
        np.asarray(output.obsm["merged_emb"]).copy(),
        output.obs["section"].astype(str).to_numpy(),
        output.obs["original_barcode"].astype(str).to_numpy(),
        (path,),
    )


def load_mofa() -> LoadedData:
    path = Path(
        "/home/hujinlan/mofa+/analysis/"
        "human_lymph_node_mofa_hvg2000_k10_iter1000/"
        "tables/factors_with_metadata_and_coordinates.csv"
    )
    table = pd.read_csv(path)
    factor_columns = [column for column in table if column.startswith("Factor")]
    return LoadedData(
        table[factor_columns].to_numpy(float),
        table["section"].astype(str).to_numpy(),
        table["original_barcode"].astype(str).to_numpy(),
        (path,),
    )


METHODS = {
    "spa_mo_model_v6": MethodSpec(
        "spa_mo_model_v6",
        "spa_mo_model result_v6",
        Path(
            "/home/hujinlan/spa_mo_model/result_v6/human_lymph_node/"
            "bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/"
            "standardized_embedding"
        ),
        Path(
            "/home/hujinlan/spa_mo_model/data/Human_Lymph_Node/"
            "Human_Lymph_Node_A1/annotation.csv"
        ),
        load_spa_mo_model,
    ),
    "cosie": MethodSpec(
        "cosie",
        "COSIE",
        Path(
            "/home/hujinlan/cosie_runs/"
            "human_lymph_node_preprocessing_comparison/standardized_embedding"
        ),
        Path(
            "/home/hujinlan/cosie/data/Human_Lymph_Node/"
            "Human_Lymph_Node_A1/annotation.csv"
        ),
        load_cosie,
    ),
    "present": MethodSpec(
        "present",
        "PRESENT",
        Path(
            "/home/hujinlan/PRESENT/result/human_lymph_node/"
            "source_default_rna_adt/analysis/standardized_embedding"
        ),
        Path(
            "/home/hujinlan/PRESENT/data/Human_Lymph_Node/"
            "Human_Lymph_Node_A1/annotation.csv"
        ),
        load_present,
    ),
    "spamosaic": MethodSpec(
        "spamosaic",
        "SpaMosaic",
        Path(
            "/home/hujinlan/SpaMosaic-dev/analysis/"
            "human_lymph_node_preprocessing_comparison/standardized_embedding"
        ),
        Path(
            "/home/hujinlan/SpaMosaic-dev/demo/data/Human_Lymph_Node/"
            "Human_Lymph_Node_A1/annotation.csv"
        ),
        load_spamosaic,
    ),
    "mofa": MethodSpec(
        "mofa",
        "MOFA+",
        Path(
            "/home/hujinlan/mofa+/analysis/"
            "human_lymph_node_preprocessing_comparison/standardized_embedding"
        ),
        Path(
            "/home/hujinlan/mofa+/data/Human_Lymph_Node/"
            "Human_Lymph_Node_A1/annotation.csv"
        ),
        load_mofa,
    ),
}


def standardize(
    embedding: np.ndarray, scaler: np.lib.npyio.NpzFile, prefix: str
) -> np.ndarray:
    mean = scaler[f"{prefix}_mean"]
    scale = scaler[f"{prefix}_scale"]
    if embedding.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError(f"{prefix}: embedding/scaler dimension mismatch")
    # StandardScaler.transform preserves a floating input array's dtype and
    # applies subtraction/division in two in-place steps.  Keeping that exact
    # behavior matters for deterministic KMeans reproduction on PRESENT's
    # float32 embedding near a few assignment boundaries.
    transformed = np.array(embedding, copy=True)
    if not np.issubdtype(transformed.dtype, np.floating):
        transformed = transformed.astype(np.float64)
    transformed -= mean
    transformed /= scale
    return transformed


def external_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(truth, prediction)),
        "nmi": float(normalized_mutual_info_score(truth, prediction)),
        "homogeneity": float(homogeneity_score(truth, prediction)),
        "completeness": float(completeness_score(truth, prediction)),
        "v_measure": float(v_measure_score(truth, prediction)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [
        "mode",
        "scope",
        "k",
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "label_asw",
    ]
    selected = frame[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate_inputs(spec: MethodSpec, data: LoadedData) -> dict:
    config_path = spec.output_root / "config.json"
    scaler_path = spec.output_root / "scaler_parameters.npz"
    if not config_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(f"{spec.display_name}: missing standardized result files")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    n_obs, embedding_dim = data.embedding.shape
    if n_obs != len(data.sections) or n_obs != len(data.barcodes):
        raise ValueError(f"{spec.display_name}: inconsistent input row counts")
    if n_obs != int(config["n_obs"]) or embedding_dim != int(config["embedding_dim"]):
        raise ValueError(
            f"{spec.display_name}: loaded shape {data.embedding.shape} does not match config"
        )
    if not np.isfinite(data.embedding).all():
        raise ValueError(f"{spec.display_name}: non-finite embedding values")
    counts = pd.Series(data.sections).value_counts().to_dict()
    expected = {key: int(value) for key, value in config["section_counts"].items()}
    if counts != expected:
        raise ValueError(f"{spec.display_name}: section counts {counts} != {expected}")
    identities = pd.MultiIndex.from_arrays([data.sections, data.barcodes])
    if identities.has_duplicates:
        raise ValueError(f"{spec.display_name}: duplicate section/barcode identities")
    return config


def load_truth(spec: MethodSpec, a1_barcodes: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    annotation = pd.read_csv(spec.annotation_path)
    required = {"Barcode", LABEL_KEY}
    if not required.issubset(annotation.columns):
        raise ValueError(f"{spec.annotation_path}: required columns are {sorted(required)}")
    if annotation["Barcode"].astype(str).duplicated().any():
        raise ValueError(f"{spec.annotation_path}: duplicate annotation barcodes")
    annotation["Barcode"] = annotation["Barcode"].astype(str)
    annotation[LABEL_KEY] = annotation[LABEL_KEY].astype("string")
    lookup = annotation.set_index("Barcode")[LABEL_KEY]
    truth = pd.Series(a1_barcodes).map(lookup)
    if truth.isna().any() or truth.astype(str).str.strip().eq("").any():
        raise ValueError(
            f"{spec.display_name}: {int(truth.isna().sum())} A1 spots lack annotation"
        )
    if truth.nunique() < 2:
        raise ValueError(f"{spec.display_name}: fewer than two annotation classes")
    return truth.astype(str).to_numpy(), annotation


def validate_retained_labels(
    spec: MethodSpec,
    a1_barcodes: np.ndarray,
    predictions: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for mode in ("joint", "independent"):
        for k in RETAINED_K_VALUES:
            path = (
                spec.output_root
                / "clustering"
                / f"{mode}_k{k}"
                / f"labels_{A1}.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"{spec.display_name}: missing retained labels {path}")
            saved = pd.read_csv(path)
            if not {"obs_name", "cluster"}.issubset(saved.columns):
                raise ValueError(f"{path}: missing obs_name/cluster")
            if saved["obs_name"].astype(str).duplicated().any():
                raise ValueError(f"{path}: duplicate barcodes")
            predicted = pd.Series(predictions[mode][k], index=a1_barcodes)
            aligned = predicted.reindex(saved["obs_name"].astype(str))
            if aligned.isna().any() or len(saved) != len(a1_barcodes):
                raise ValueError(f"{path}: saved/predicted A1 identities do not match")
            reproduction_ari = float(
                adjusted_rand_score(saved["cluster"].to_numpy(), aligned.to_numpy())
            )
            if not np.isclose(reproduction_ari, 1.0, atol=1e-12):
                raise ValueError(
                    f"{spec.display_name} {mode} k={k}: retained labels were not reproduced "
                    f"(ARI={reproduction_ari})"
                )
            result[f"{mode}_k{k}"] = {
                "reproduction_ari": reproduction_ari,
                "n_obs": int(len(saved)),
                "saved_labels_path": str(path),
                "saved_labels_sha256": sha256(path),
            }
    return result


def evaluate(spec: MethodSpec, dry_run: bool = False) -> dict:
    data = spec.loader()
    original_config = validate_inputs(spec, data)
    a1_mask = data.sections == A1
    a1_barcodes = data.barcodes[a1_mask]
    truth, annotation = load_truth(spec, a1_barcodes)
    scaler_path = spec.output_root / "scaler_parameters.npz"
    scaler = np.load(scaler_path)
    try:
        joint_space = standardize(data.embedding, scaler, "joint")
        independent_space = standardize(
            data.embedding[a1_mask], scaler, A1
        )
    finally:
        scaler.close()

    spaces = {
        "joint": joint_space,
        "independent": independent_space,
    }
    predictions: dict[str, dict[int, np.ndarray]] = {
        "joint": {},
        "independent": {},
    }
    rows: list[dict] = []
    for mode, space in spaces.items():
        evaluation_space = joint_space[a1_mask] if mode == "joint" else space
        label_asw = float(silhouette_score(evaluation_space, truth))
        for k in K_VALUES:
            labels = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=N_INIT,
                max_iter=MAX_ITER,
            ).fit_predict(space)
            a1_labels = labels[a1_mask] if mode == "joint" else labels
            predictions[mode][k] = np.asarray(a1_labels, dtype=np.int16)
            rows.append(
                {
                    "mode": mode,
                    "scope": "combined_A1_subset" if mode == "joint" else A1,
                    "label": LABEL_KEY,
                    "n_obs_labeled": int(len(truth)),
                    "n_label_classes": int(np.unique(truth).size),
                    "k": k,
                    **external_metrics(truth, a1_labels),
                    "label_asw": label_asw,
                    "label_asw_scaled": (label_asw + 1.0) / 2.0,
                }
            )
    validation = validate_retained_labels(spec, a1_barcodes, predictions)
    metrics = pd.DataFrame(rows)
    best = (
        metrics.loc[metrics.groupby(["mode", "label"])["ari"].idxmax()]
        .sort_values(["mode", "label"])
        .reset_index(drop=True)
    )

    annotation_barcodes = set(annotation["Barcode"])
    evaluated_barcodes = set(a1_barcodes)
    labels_table = pd.DataFrame(
        {
            "section": A1,
            "obs_name": a1_barcodes,
            LABEL_KEY: truth,
        }
    )
    for mode in ("joint", "independent"):
        for k in K_VALUES:
            labels_table[f"{mode}_k{k}"] = predictions[mode][k]

    metrics_root = spec.output_root / "metrics"
    metrics_path = metrics_root / "clustering_metrics_by_label.csv"
    best_path = metrics_root / "best_clustering_metrics_by_label.csv"
    labels_path = metrics_root / "a1_supervised_cluster_labels.csv.gz"
    config_path = metrics_root / "a1_supervised_metrics_config.json"
    readme_path = spec.output_root / "A1_SUPERVISED_METRICS.md"
    outputs = [metrics_path, best_path, labels_path, config_path, readme_path]

    manifest = {
        "status": "DRY_RUN" if dry_run else "PASS",
        "dataset": "Human_Lymph_Node",
        "method": spec.display_name,
        "evaluation_section": A1,
        "unannotated_section_excluded": D1,
        "label": LABEL_KEY,
        "annotation_path": str(spec.annotation_path),
        "annotation_sha256": sha256(spec.annotation_path),
        "annotation_rows_total": int(len(annotation)),
        "annotation_rows_evaluated": int(len(truth)),
        "annotation_rows_not_in_method_output": int(
            len(annotation_barcodes - evaluated_barcodes)
        ),
        "n_label_classes": int(np.unique(truth).size),
        "label_class_counts": {
            str(key): int(value)
            for key, value in pd.Series(truth).value_counts().sort_index().items()
        },
        "modes": {
            "joint": "KMeans fit on A1+D1; supervised metrics evaluated on A1 only",
            "independent": "KMeans fit and supervised metrics evaluated on A1 only",
        },
        "k_values": K_VALUES,
        "standardization": "reuse saved StandardScaler mean/scale parameters",
        "kmeans": {
            "implementation": "sklearn.cluster.KMeans",
            "random_state": SEED,
            "n_init": N_INIT,
            "max_iter": MAX_ITER,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "anndata": ad.__version__,
        },
        "metrics": [
            "ari",
            "nmi",
            "homogeneity",
            "completeness",
            "v_measure",
            "label_asw",
            "label_asw_scaled",
        ],
        "label_asw_rule": "full A1 labeled spots in the same standardized metric space",
        "retained_label_reproduction_validation": validation,
        "original_analysis_config": str(spec.output_root / "config.json"),
        "original_analysis_config_sha256": sha256(spec.output_root / "config.json"),
        "scaler_parameters": str(scaler_path),
        "scaler_parameters_sha256": sha256(scaler_path),
        "embedding_sources": [
            {"path": str(path), "sha256": sha256(path)} for path in data.source_paths
        ],
        "original_config_label_asw": original_config.get("label_asw"),
        "outputs": [str(path) for path in outputs],
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }

    if dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    metrics_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    best.to_csv(best_path, index=False)
    labels_table.to_csv(labels_path, index=False, compression="gzip")
    manifest["outputs"] = [
        {"path": str(path), "sha256": sha256(path)}
        for path in (metrics_path, best_path, labels_path)
    ] + [{"path": str(readme_path)}]
    config_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {spec.display_name}: Human Lymph Node A1 supervised metrics",
        "",
        f"- A1 labeled spots evaluated: {len(truth):,}",
        f"- manual annotation classes: {np.unique(truth).size}",
        "- D1 is excluded from supervised scoring because it has no annotation.",
        "- joint: KMeans is fit on A1+D1 and scored on the A1 subset.",
        "- independent: KMeans is fit and scored on A1 only.",
        f"- K values: {K_VALUES}",
        "- retained K=5/8/10/12 labels were reproduced with ARI=1 before scoring.",
        "",
        "## Best ARI by mode",
        "",
        markdown_table(best),
        "",
        "## Outputs",
        "",
        f"- `{metrics_path}`",
        f"- `{best_path}`",
        f"- `{labels_path}`",
        f"- `{config_path}`",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    # Refresh the config after the README exists so all non-config outputs have hashes.
    manifest["outputs"] = [
        {"path": str(path), "sha256": sha256(path)}
        for path in (metrics_path, best_path, labels_path, readme_path)
    ]
    config_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{spec.display_name}: PASS; n={len(truth)}, classes={np.unique(truth).size}, "
        f"joint best k={int(best.loc[best['mode'].eq('joint'), 'k'].iloc[0])}, "
        f"independent best k={int(best.loc[best['mode'].eq('independent'), 'k'].iloc[0])}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[*METHODS, "all"],
        default=["all"],
        help="Methods to evaluate (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and calculate without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(METHODS) if "all" in args.methods else args.methods
    for key in selected:
        print(f"[{METHODS[key].display_name}] evaluating", flush=True)
        evaluate(METHODS[key], dry_run=args.dry_run)
    print("HUMAN_LYMPH_NODE_A1_SUPERVISED_METRICS: PASS", flush=True)


if __name__ == "__main__":
    main()
