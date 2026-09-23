"""Single-method CRC and lymph-node comparison workflows, with original artifacts.

These retain different sampling and spatial diagnostic protocols. They receive
P6a data and use P6b standardization/clustering/metrics; no method data is loaded here.
"""
from __future__ import annotations
from analysis.exact_kmeans_workflow import execute_exact_kmeans_scopes
from data_io import saved_assignments as sa
from pathlib import Path
from datetime import datetime
from typing import Any
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from analysis.cache import digest_file as sha256
from analysis.clustering import fitted_space as metric_space, scaler_payload, kmeans_labels, fit_minibatch_model
from analysis.metrics import (sampled_internal_metrics as internal_metrics, safe_precomputed_asw,
    safe_asw, internal_metrics as internal)
from analysis.sampling import select_by_barcode_hash as barcode_sample, key_mask
from analysis.protocols import CRC, HLN
from analysis.plotting import plot_crc_spatial, plot_spatial_categories

CRC_SECTIONS = ["CRC_003_bin20", "CRC_006_bin20"]


CRC_SHORT_NAMES = {
    "CRC_003_bin20": "CRC_003",
    "CRC_006_bin20": "CRC_006",
}


CRC_SECTION_COUNTS = {
    "CRC_003_bin20": 166279,
    "CRC_006_bin20": 446095,
}


CRC_ASW_SECTION_COUNTS = CRC['ASW_SECTION_COUNTS']


CRC_PLOT_SECTION_COUNTS = CRC_SECTION_COUNTS.copy()


CRC_N_OBS = sum(CRC_SECTION_COUNTS.values())


CRC_ASW_N_OBS = sum(CRC_ASW_SECTION_COUNTS.values())


CRC_KS = CRC['KS']


CRC_SEED = CRC['SEED']


CRC_N_INIT = CRC['N_INIT']


CRC_MAX_ITER = CRC['MAX_ITER']


CRC_BATCH_SIZE = CRC['BATCH_SIZE']


CRC_SPATIAL_NEIGHBOR_K = 6


CRC_SCRIPT_PATH = Path(__file__).resolve()


def crc_select_by_barcode_hash(reference: Any, counts: dict[str, int], prefix: str) -> pd.DataFrame:
    return barcode_sample(reference, counts, prefix, section_order=CRC_SECTIONS, section_counts=CRC_SECTION_COUNTS)


def crc_ordered_barcode_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def crc_build_spatial_graph(coords: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=CRC_SPATIAL_NEIGHBOR_K + 1,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(coords)
    raw = model.kneighbors(coords, return_distance=False)
    row_ids = np.arange(len(coords))[:, None]
    without_self = raw[raw != row_ids].reshape(
        len(coords), CRC_SPATIAL_NEIGHBOR_K
    )
    return without_self.astype(np.int32)


def crc_save_labels(
    path: Path, data: Any, mask: np.ndarray, labels: np.ndarray
) -> str:
    pd.DataFrame(
        {
            "section": data.sections[mask],
            "obs_name": data.barcodes[mask],
            "cluster": labels.astype(int),
            "x": data.coords[mask, 0],
            "y": data.coords[mask, 1],
        }
    ).to_csv(path, index=False)
    return sha256(path)


def crc_fit_minibatch(space: np.ndarray, k: int) -> tuple[np.ndarray, MiniBatchKMeans]:
    return fit_minibatch_model(space, k, seed=CRC_SEED, n_init=CRC_N_INIT, max_iter=CRC_MAX_ITER, batch_size=CRC_BATCH_SIZE)


def crc_save_model(
    directory: Path, model: MiniBatchKMeans, labels_path: Path
) -> None:
    np.save(directory / "cluster_centers.npy", model.cluster_centers_)
    payload = {
        "class": "sklearn.cluster.MiniBatchKMeans",
        "parameters": {
            "n_clusters": int(model.n_clusters),
            "seed": CRC_SEED,
            "n_init": CRC_N_INIT,
            "max_iter": CRC_MAX_ITER,
            "batch_size": CRC_BATCH_SIZE,
            "init": "k-means++",
            "tol": 0.0,
            "max_no_improvement": 10,
            "init_size": None,
            "reassignment_ratio": 0.01,
        },
        "n_iter": int(model.n_iter_),
        "n_steps": int(model.n_steps_),
        "inertia": float(model.inertia_),
        "labels_path": str(labels_path),
        "labels_sha256": sha256(labels_path),
    }
    (directory / "minibatchkmeans_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def crc_prepare_shared(
    data: Any,
    asw_sample: pd.DataFrame,
    plot_sample: pd.DataFrame,
) -> dict[str, np.ndarray]:
    output = data.output_root / "shared_metrics"
    output.mkdir(parents=True)
    asw_path = output / "asw_sample.csv"
    plot_path = output / "plot_sample.csv"
    asw_sample.to_csv(asw_path, index=False)
    plot_sample.to_csv(plot_path, index=False)

    copy_rows = []
    for name, source in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        copy_rows.append(
            {
                "artifact": name,
                "source": str(source),
                "source_sha256": sha256(source),
                "copied_sha256": sha256(target),
                "preprocessing_dependent": False,
            }
        )
    pd.DataFrame(copy_rows).to_csv(
        output / "source_manifest.csv", index=False
    )

    graphs: dict[str, np.ndarray] = {}
    graph_rows = []
    for section in CRC_SECTIONS:
        mask = data.sections == section
        graph = crc_build_spatial_graph(data.coords[mask])
        graph_path = output / f"spatial_neighbors_{section}.npz"
        np.savez_compressed(graph_path, neighbor_indices=graph)
        graphs[section] = graph
        graph_rows.append(
            {
                "section": section,
                "n_obs": int(mask.sum()),
                "neighbor_k": CRC_SPATIAL_NEIGHBOR_K,
                "algorithm": "sklearn NearestNeighbors kd_tree exact",
                "barcode_order_sha256": crc_ordered_barcode_sha256(
                    data.barcodes[mask]
                ),
                "graph_path": str(graph_path),
                "graph_sha256": sha256(graph_path),
            }
        )
    pd.DataFrame(graph_rows).to_csv(
        output / "spatial_graph_manifest.csv", index=False
    )
    sample_manifest = {
        "asw_sample": {
            "path": str(asw_path),
            "sha256": sha256(asw_path),
            "n_obs": CRC_ASW_N_OBS,
            "section_counts": CRC_ASW_SECTION_COUNTS,
            "selection": "lowest SHA256('asw_seed0|' + barcode) per section",
        },
        "plot_sample": {
            "path": str(plot_path),
            "sha256": sha256(plot_path),
            "n_obs": sum(CRC_PLOT_SECTION_COUNTS.values()),
            "section_counts": CRC_PLOT_SECTION_COUNTS,
            "selection": (
                "all spots in each section; rows stored in deterministic "
                "SHA256(barcode) order"
            ),
        },
        "identity_key": ["section", "obs_name"],
        "shared_across_all_methods_preprocessing_modes_and_k": True,
    }
    (output / "sampling_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return graphs


def crc_spatial_agreement_from_graph(
    labels: np.ndarray, neighbors: np.ndarray
) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def crc_run_scheme(
    data: Any,
    scheme: str,
    asw_sample: pd.DataFrame,
    plot_sample: pd.DataFrame,
    graphs: dict[str, np.ndarray],
    *, joint_ks=None, independent_ks=None,
) -> None:
    joint_ks = CRC_KS if joint_ks is None else list(joint_ks)
    independent_ks = CRC_KS if independent_ks is None else list(independent_ks)
    output = data.output_root / scheme
    clustering_root = output / "clustering"
    metrics_root = output / "metrics"
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    asw_mask = key_mask(data, asw_sample)
    plot_mask = key_mask(data, plot_sample)
    if int(asw_mask.sum()) != CRC_ASW_N_OBS:
        raise ValueError(f"{data.name}: ASW sample mapping failed")
    for section in CRC_SECTIONS:
        expected = CRC_ASW_SECTION_COUNTS[section]
        if int(np.sum(asw_mask & (data.sections == section))) != expected:
            raise ValueError(f"{data.name}: ASW count mismatch {section}")
        if (
            int(np.sum(plot_mask & (data.sections == section)))
            != CRC_PLOT_SECTION_COUNTS[section]
        ):
            raise ValueError(f"{data.name}: plot count mismatch {section}")

    metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    all_mask = np.ones(CRC_N_OBS, dtype=bool)
    asw_path = data.output_root / "shared_metrics" / "asw_sample.csv"
    plot_path = data.output_root / "shared_metrics" / "plot_sample.csv"
    asw_hash = sha256(asw_path)
    plot_hash = sha256(plot_path)

    joint_space, joint_scaler = metric_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, "joint"))
    joint_asw_indices = np.flatnonzero(asw_mask)
    joint_distances = pairwise_distances(
        joint_space[joint_asw_indices], metric="euclidean", n_jobs=-1
    )
    np.fill_diagonal(joint_distances, 0.0)
    section_asw = safe_precomputed_asw(
        joint_distances, data.sections[joint_asw_indices]
    )

    for k in joint_ks:
        labels_all, model = crc_fit_minibatch(joint_space, k)
        directory = clustering_root / f"joint_k{k}"
        directory.mkdir()
        labels_all_path = directory / "labels_all.csv"
        labels_all_sha = crc_save_labels(
            labels_all_path, data, all_mask, labels_all
        )
        crc_save_model(directory, model, labels_all_path)
        metric_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": CRC_N_OBS,
                "embedding_dim": data.embedding.shape[1],
                **internal_metrics(
                    joint_space,
                    labels_all,
                    joint_asw_indices,
                    joint_distances,
                ),
                "section_ari": adjusted_rand_score(
                    data.sections, labels_all
                ),
                "section_nmi": normalized_mutual_info_score(
                    data.sections, labels_all
                ),
                "section_asw": section_asw,
                "section_asw_n_obs": CRC_ASW_N_OBS,
                "metric_space": scheme,
                "labels_path": str(labels_all_path),
                "labels_sha256": labels_all_sha,
                "asw_sample_path": str(asw_path),
                "asw_sample_sha256": asw_hash,
            }
        )
        count_rows = []
        for section in CRC_SECTIONS:
            section_mask = data.sections == section
            section_labels = labels_all[section_mask]
            labels_path = directory / f"labels_{section}.csv"
            labels_sha = crc_save_labels(
                labels_path, data, section_mask, section_labels
            )
            graph_path = (
                data.output_root
                / "shared_metrics"
                / f"spatial_neighbors_{section}.npz"
            )
            spatial_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "n_obs": int(section_mask.sum()),
                    "spatial_neighbor_k": CRC_SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction":
                    crc_spatial_agreement_from_graph(
                        section_labels, graphs[section]
                    ),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "spatial_graph_path": str(graph_path),
                    "spatial_graph_sha256": sha256(graph_path),
                }
            )
            local_plot_mask = plot_mask[section_mask]
            figure_path = directory / f"spatial_{section}.png"
            plot_crc_spatial(
                figure_path,
                data.coords[section_mask][local_plot_mask],
                section_labels[local_plot_mask],
                f"{data.name} {scheme} joint K={k} {section}",
                k,
            )
            figure_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "plot_n_obs": int(local_plot_mask.sum()),
                    "figure_path": str(figure_path),
                    "figure_sha256": sha256(figure_path),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "plot_sample_path": str(plot_path),
                    "plot_sample_sha256": plot_hash,
                }
            )
            for cluster, count in zip(
                *np.unique(section_labels, return_counts=True)
            ):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(section_mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(
            directory / "cluster_counts.csv", index=False
        )
    del joint_distances

    for section in CRC_SECTIONS:
        section_mask = data.sections == section
        section_space, scaler = metric_space(
            data.embedding[section_mask], scheme
        )
        if scaler is not None:
            scaler_arrays.update(scaler_payload(scaler, section))
        local_asw_mask = asw_mask[section_mask]
        local_plot_mask = plot_mask[section_mask]
        local_asw_indices = np.flatnonzero(local_asw_mask)
        local_distances = pairwise_distances(
            section_space[local_asw_indices],
            metric="euclidean",
            n_jobs=-1,
        )
        np.fill_diagonal(local_distances, 0.0)
        for k in independent_ks:
            labels, model = crc_fit_minibatch(section_space, k)
            directory = clustering_root / f"independent_k{k}"
            directory.mkdir(exist_ok=True)
            labels_path = directory / f"labels_{section}.csv"
            labels_sha = crc_save_labels(
                labels_path, data, section_mask, labels
            )
            model_directory = directory / section
            model_directory.mkdir()
            crc_save_model(model_directory, model, labels_path)
            metric_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(section_mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **internal_metrics(
                        section_space,
                        labels,
                        local_asw_indices,
                        local_distances,
                    ),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "section_asw_n_obs": 0,
                    "metric_space": scheme,
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "asw_sample_path": str(asw_path),
                    "asw_sample_sha256": asw_hash,
                }
            )
            graph_path = (
                data.output_root
                / "shared_metrics"
                / f"spatial_neighbors_{section}.npz"
            )
            spatial_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "n_obs": int(section_mask.sum()),
                    "spatial_neighbor_k": CRC_SPATIAL_NEIGHBOR_K,
                    "neighbor_same_cluster_fraction":
                    crc_spatial_agreement_from_graph(labels, graphs[section]),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "spatial_graph_path": str(graph_path),
                    "spatial_graph_sha256": sha256(graph_path),
                }
            )
            figure_path = directory / f"spatial_{section}.png"
            plot_crc_spatial(
                figure_path,
                data.coords[section_mask][local_plot_mask],
                labels[local_plot_mask],
                f"{data.name} {scheme} independent K={k} {section}",
                k,
            )
            figure_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "plot_n_obs": int(local_plot_mask.sum()),
                    "figure_path": str(figure_path),
                    "figure_sha256": sha256(figure_path),
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha,
                    "plot_sample_path": str(plot_path),
                    "plot_sample_sha256": plot_hash,
                }
            )
        del local_distances

    for k in independent_ks:
        rows = []
        for section in CRC_SECTIONS:
            labels = sa.read_assignment_table(
                clustering_root
                / f"independent_k{k}"
                / f"labels_{section}.csv",
                usecols=["cluster"],
            )["cluster"].to_numpy()
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / CRC_SECTION_COUNTS[section],
                    }
                )
        pd.DataFrame(rows).to_csv(
            clustering_root / f"independent_k{k}" / "cluster_counts.csv",
            index=False,
        )

    metrics = pd.DataFrame(metric_rows)
    spatial = pd.DataFrame(spatial_rows)
    figures = pd.DataFrame(figure_rows)
    metrics_path = metrics_root / "clustering_metrics.csv"
    spatial_path = metrics_root / "spatial_continuity.csv"
    spatial_summary_path = metrics_root / "spatial_continuity_summary.csv"
    figures_path = metrics_root / "figure_manifest.csv"
    metrics.to_csv(metrics_path, index=False)
    spatial.to_csv(spatial_path, index=False)
    (
        spatial.groupby(["mode", "k"], as_index=False)[
            "neighbor_same_cluster_fraction"
        ]
        .mean()
        .rename(
            columns={
                "neighbor_same_cluster_fraction":
                "mean_spatial_neighbor_agreement"
            }
        )
        .to_csv(spatial_summary_path, index=False)
    )
    figures.to_csv(figures_path, index=False)
    if scaler_arrays:
        np.savez_compressed(
            output / "scaler_parameters.npz", **scaler_arrays
        )

    config = {
        "dataset": "CRC_Stereo-CITE-seq",
        "method": data.name,
        "preprocessing": scheme,
        "kmeans_input": (
            "raw_final_embedding"
            if scheme == "raw_embedding"
            else "standardized_final_embedding"
        ),
        "standardization": (
            "none"
            if scheme == "raw_embedding"
            else "sklearn.preprocessing.StandardScaler"
        ),
        "joint_scaler_scope": (
            "none" if scheme == "raw_embedding" else "all_612374_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "clustering_class": "sklearn.cluster.MiniBatchKMeans",
        "k_values": CRC_KS,
        "seed": CRC_SEED,
        "n_init": CRC_N_INIT,
        "max_iter": CRC_MAX_ITER,
        "batch_size": CRC_BATCH_SIZE,
        "n_obs": CRC_N_OBS,
        "section_counts": CRC_SECTION_COUNTS,
        "embedding_dim": data.embedding.shape[1],
        "cluster_asw_sample_size_joint": CRC_ASW_N_OBS,
        "cluster_asw_sample_size_independent": CRC_ASW_SECTION_COUNTS,
        "cluster_asw_sample_file": str(asw_path),
        "cluster_asw_sample_sha256": asw_hash,
        "ch_dbi_rule": "full_n_obs_in_each_scope",
        "ch_dbi_sample_size": 0,
        "label_asw": "not_applicable_no_ground_truth_labels",
        "plot_sample_size_per_section": CRC_PLOT_SECTION_COUNTS,
        "plot_sample_file": str(plot_path),
        "plot_sample_sha256": plot_hash,
        "plot_selection": (
            "all spots in each section; rows stored in deterministic "
            "SHA256(barcode) order"
        ),
        "full_spot_spatial_plots": True,
        "spatial_neighbor_k": CRC_SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "exact_kd_tree_within_each_section",
        "spatial_graph_n_obs": CRC_SECTION_COUNTS,
        "full_labels_saved_for_all_methods_modes_and_k": True,
        "metrics_and_figures_derived_directly_from_saved_labels": True,
        "report_derived_metrics_chain": False,
        "source_data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in data.source_paths
        ],
        "generation_script": str(CRC_SCRIPT_PATH),
        "generation_script_sha256": sha256(CRC_SCRIPT_PATH),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    provenance_rows = []
    for role, path in [
        ("config", config_path),
        ("clustering_metrics", metrics_path),
        ("spatial_continuity", spatial_path),
        ("spatial_continuity_summary", spatial_summary_path),
        ("figure_manifest", figures_path),
        ("asw_sample", asw_path),
        ("plot_sample", plot_path),
    ]:
        provenance_rows.append(
            {"role": role, "path": str(path), "sha256": sha256(path)}
        )
    pd.DataFrame(provenance_rows).to_csv(
        metrics_root / "direct_provenance_manifest.csv", index=False
    )
    (output / "SUMMARY.md").write_text(
        f"# {data.name}: {scheme}\n\n"
        f"- spots: {CRC_N_OBS}; dimensions: {data.embedding.shape[1]}\n"
        f"- MiniBatchKMeans: seed={CRC_SEED}, n_init={CRC_N_INIT}, "
        f"max_iter={CRC_MAX_ITER}, batch_size={CRC_BATCH_SIZE}\n"
        f"- joint/independent full labels, metrics and figures: K={CRC_KS}\n"
        f"- Cluster ASW: fixed section-stratified {CRC_ASW_N_OBS}-spot sample\n"
        "- CH/DBI: full scope; Label ASW: not applicable\n"
        f"- figures: all spots (section counts: {CRC_PLOT_SECTION_COUNTS})\n"
        "- metrics and figures are generated directly from saved labels\n"
        "- no report-derived metric chain is used\n"
    )


HLN_SECTIONS = ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]


HLN_METRIC_KS = HLN['METRIC_KS']


HLN_PLOT_KS = [5, 8, 10, 12]


HLN_PRUNED_CLUSTERING_KS = [k for k in HLN_METRIC_KS if k not in HLN_PLOT_KS]


HLN_SEED = HLN['SEED']


HLN_N_INIT = HLN['N_INIT']


HLN_MAX_ITER = HLN['MAX_ITER']


HLN_SPATIAL_NEIGHBOR_K = 6


HLN_POINT_SIZE = 10.0


HLN_DPI = 220


def hln_spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=HLN_SPATIAL_NEIGHBOR_K + 1).fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def hln_save_labels(
    path: Path, data: Any, mask: np.ndarray, labels: np.ndarray
) -> None:
    pd.DataFrame(
        {
            "section": data.sections[mask],
            "obs_name": data.barcodes[mask],
            "cluster": labels.astype(int),
            "x": data.coords[mask, 0],
            "y": data.coords[mask, 1],
        }
    ).to_csv(path, index=False)


def hln_run_scheme(data: Any, scheme: str, *, joint_ks=None, independent_ks=None) -> None:
    joint_ks = HLN_METRIC_KS if joint_ks is None else list(joint_ks)
    independent_ks = HLN_PLOT_KS if independent_ks is None else list(independent_ks)
    out = data.output_root / scheme
    execute_exact_kmeans_scopes(
        data, scheme, section_order=HLN_SECTIONS,
        joint_ks=joint_ks, independent_ks=independent_ks,
        seed=HLN_SEED, n_init=HLN_N_INIT, max_iter=HLN_MAX_ITER,
        spatial_neighbor_k=HLN_SPATIAL_NEIGHBOR_K,
        joint_plot_ks=HLN_PLOT_KS, independent_plot_ks=None,
        joint_mask_per_fit=True, save_labels=hln_save_labels,
        plot_spatial=plot_hln_spatial, spatial_agreement=hln_spatial_agreement,
    )

    config = {
        "dataset": "Human_Lymph_Node",
        "method": data.name,
        "preprocessing": scheme,
        "kmeans_input": (
            "raw_final_embedding"
            if scheme == "raw_embedding"
            else "standardized_final_embedding"
        ),
        "metric_space": (
            "raw_final_embedding"
            if scheme == "raw_embedding"
            else "standardized_final_embedding"
        ),
        "standardization": (
            "none"
            if scheme == "raw_embedding"
            else "sklearn.preprocessing.StandardScaler"
        ),
        "joint_scaler_scope": (
            "none" if scheme == "raw_embedding" else "all_6843_spots"
        ),
        "independent_scaler_scope": (
            "none" if scheme == "raw_embedding" else "fit_per_section"
        ),
        "kmeans_type": "sklearn.cluster.KMeans",
        "metric_k_values": joint_ks,
        "plot_and_independent_k_values": independent_ks,
        "retained_clustering_k_values": HLN_PLOT_KS,
        "deleted_clustering_k_values": HLN_PRUNED_CLUSTERING_KS,
        "metrics_k_values_without_saved_label_directories":
        HLN_PRUNED_CLUSTERING_KS,
        "seed": HLN_SEED,
        "n_init": HLN_N_INIT,
        "max_iter": HLN_MAX_ITER,
        "n_obs": len(data.embedding),
        "section_counts": {
            key: int(value)
            for key, value in pd.Series(data.sections).value_counts().items()
        },
        "embedding_dim": data.embedding.shape[1],
        "asw_sample_size": 0,
        "asw_sample_rule": "full_n_obs",
        "ch_dbi_sample_size": 0,
        "ch_dbi_sample_rule": "full_n_obs",
        "label_asw": "not_applicable_no_ground_truth_labels",
        "spatial_neighbor_k": HLN_SPATIAL_NEIGHBOR_K,
        "spatial_graph_scope": "fit_separately_within_each_section",
        "spatial_graph_n_obs": {
            key: int(value)
            for key, value in pd.Series(data.sections).value_counts().items()
        },
        "plots_and_metrics_reuse_same_label_files": False,
        "retained_k_plots_and_metrics_reuse_same_label_files": True,
        "source_data_dir": str(data.data_dir),
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in data.source_paths
        ],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    hln_write_scheme_summary(data, scheme)


def hln_write_scheme_summary(data: Any, scheme: str) -> None:
    summary = (
        f"# {data.name}: {scheme}\n\n"
        f"- spots: {len(data.embedding)}; dimensions: {data.embedding.shape[1]}\n"
        f"- KMeans: sklearn exact, seed={HLN_SEED}, n_init={HLN_N_INIT}, "
        f"max_iter={HLN_MAX_ITER}\n"
        f"- joint metric K: {HLN_METRIC_KS}; independent metric K: {HLN_PLOT_KS}\n"
        f"- retained clustering directories/labels/plots K: {HLN_PLOT_KS}\n"
        f"- deleted joint clustering directory K: {HLN_PRUNED_CLUSTERING_KS}; "
        "metric CSV rows remain, so their historical labels_path values no "
        "longer resolve\n"
        f"- metrics: full sample in the same space used by KMeans\n"
        f"- spatial continuity: exact {HLN_SPATIAL_NEIGHBOR_K}-NN within each section\n"
        "- for retained K, labels are shared by figures and metrics\n"
    )
    (data.output_root / scheme / "SUMMARY.md").write_text(summary)


def plot_hln_spatial(path, coords, labels, title):
    return plot_spatial_categories(path, coords, labels, title, point_size=HLN_POINT_SIZE, dpi=HLN_DPI)
