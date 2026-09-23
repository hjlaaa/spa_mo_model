"""Ordinary exact KMeans scope workflow; callers own protocol and output policy.

No dataset dispatch, truth derivation, sampling, batch metrics or retention.
Joint section outputs slice a single fit. Independent scopes preserve K-first
iteration and reuse one lazily fitted scaler per section.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from analysis.clustering import fitted_space, kmeans_labels, scaler_payload
from analysis.metrics import internal_metrics, safe_asw


def execute_exact_kmeans_scopes(
    data: Any, scheme: str, *, section_order, joint_ks, independent_ks,
    seed, n_init, max_iter, spatial_neighbor_k, joint_plot_ks,
    independent_plot_ks, joint_mask_per_fit, save_labels, plot_spatial,
    spatial_agreement,
) -> None:
    """Use the aligned legacy input projection and explicit caller policies.

    The three functions are existing label/plot/spatial operations, not lifecycle
    hooks. None for independent_plot_ks means plot every requested independent K.
    joint_mask_per_fit preserves the callers' original mask allocation points.
    Config, summaries, provenance and downstream retention stay with callers.
    """
    out = data.output_root / scheme
    cluster_root = out / "clustering"
    metrics_root = out / "metrics"
    cluster_root.mkdir(parents=True)
    metrics_root.mkdir()
    metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scalers: dict[str, np.ndarray] = {}

    joint_space, joint_scaler = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scalers.update(scaler_payload(joint_scaler, "joint"))
    section_asw = safe_asw(joint_space, data.sections)

    all_mask = None if joint_mask_per_fit else np.ones(len(data.embedding), dtype=bool)
    for k in joint_ks:
        labels_all = kmeans_labels(joint_space, n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        kdir = cluster_root / f"joint_k{k}"
        kdir.mkdir()
        if joint_mask_per_fit:
            all_mask = np.ones(len(labels_all), dtype=bool)
        save_labels(kdir / "labels_all.csv", data, all_mask, labels_all)
        metric_rows.append(
            {
                "mode": "joint",
                "scope": "combined",
                "k": k,
                "n_obs": len(labels_all),
                "embedding_dim": data.embedding.shape[1],
                **internal_metrics(joint_space, labels_all),
                "section_ari": adjusted_rand_score(data.sections, labels_all),
                "section_nmi": normalized_mutual_info_score(
                    data.sections, labels_all
                ),
                "section_asw": section_asw,
                "metric_space": scheme,
                "labels_path": str(kdir / "labels_all.csv"),
            }
        )
        count_rows = []
        for section in section_order:
            mask = data.sections == section
            labels = labels_all[mask]
            save_labels(kdir / f"labels_{section}.csv", data, mask, labels)
            if k in joint_plot_ks:
                plot_spatial(
                    kdir / f"spatial_{section}.png",
                    data.coords[mask],
                    labels,
                    f"{data.name} {scheme} joint K={k} {section}",
                )
            spatial_rows.append(
                {
                    "mode": "joint",
                    "k": k,
                    "section": section,
                    "n_obs": int(mask.sum()),
                    "spatial_neighbor_k": spatial_neighbor_k,
                    "neighbor_same_cluster_fraction": spatial_agreement(
                        data.coords[mask], labels
                    ),
                    "labels_path": str(kdir / f"labels_{section}.csv"),
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(kdir / "cluster_counts.csv", index=False)

    # Fit on first use to preserve K-first writes and failure timing. Keep only
    # scalers, not all section embeddings; each later K transforms its own slice.
    independent_scalers = {}
    for k in independent_ks:
        kdir = cluster_root / f"independent_k{k}"
        kdir.mkdir()
        count_rows = []
        for section in section_order:
            mask = data.sections == section
            if section not in independent_scalers:
                space, scaler = fitted_space(data.embedding[mask], scheme)
                independent_scalers[section] = scaler
            else:
                scaler = independent_scalers[section]
                space = data.embedding[mask] if scaler is None else scaler.transform(data.embedding[mask])
            if scaler is not None:
                scalers.update(scaler_payload(scaler, section))
            labels = kmeans_labels(space, n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
            label_path = kdir / f"labels_{section}.csv"
            save_labels(label_path, data, mask, labels)
            if independent_plot_ks is None or k in independent_plot_ks:
                plot_spatial(
                    kdir / f"spatial_{section}.png",
                    data.coords[mask],
                    labels,
                    f"{data.name} {scheme} independent K={k} {section}",
                )
            metric_rows.append(
                {
                    "mode": "independent",
                    "scope": section,
                    "k": k,
                    "n_obs": int(mask.sum()),
                    "embedding_dim": data.embedding.shape[1],
                    **internal_metrics(space, labels),
                    "section_ari": float("nan"),
                    "section_nmi": float("nan"),
                    "section_asw": float("nan"),
                    "metric_space": scheme,
                    "labels_path": str(label_path),
                }
            )
            spatial_rows.append(
                {
                    "mode": "independent",
                    "k": k,
                    "section": section,
                    "n_obs": int(mask.sum()),
                    "spatial_neighbor_k": spatial_neighbor_k,
                    "neighbor_same_cluster_fraction": spatial_agreement(
                        data.coords[mask], labels
                    ),
                    "labels_path": str(label_path),
                }
            )
            for cluster, count in zip(*np.unique(labels, return_counts=True)):
                count_rows.append(
                    {
                        "section": section,
                        "cluster": int(cluster),
                        "count": int(count),
                        "fraction": count / int(mask.sum()),
                    }
                )
        pd.DataFrame(count_rows).to_csv(kdir / "cluster_counts.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / "clustering_metrics.csv", index=False)
    spatial.to_csv(metrics_root / "spatial_continuity.csv", index=False)
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
        .to_csv(metrics_root / "spatial_continuity_summary.csv", index=False)
    )
    if scalers:
        np.savez_compressed(out / "scaler_parameters.npz", **scalers)
