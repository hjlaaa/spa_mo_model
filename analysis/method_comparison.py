"""Standardized cross-method workflows; dataset protocols remain distinct.

Historical raw-vs-standardized and output-maintenance CLIs are retired.
"""
from __future__ import annotations
from analysis.exact_kmeans_workflow import execute_exact_kmeans_scopes
import matplotlib
matplotlib.use("Agg")
from analysis.clustering import fitted_space
from analysis.clustering import kmeans_labels
from analysis.clustering import scaler_payload
from analysis.comparisons import hln_spatial_agreement
from analysis.metrics import comparison_label_rows
from analysis.metrics import comparison_valid_truth
from analysis.metrics import external_metrics
from analysis.metrics import internal_metrics
from analysis.metrics import misar_label_rows
from analysis.metrics import safe_asw
from analysis.metrics import safe_precomputed_asw
from analysis.metrics import sampled_internal_metrics
from analysis.metrics import simulation_internal_metrics
from analysis.plotting import plot_spatial_categories
from analysis.protocols import MISAR
from analysis.protocols import MOUSEBRAIN
from analysis.protocols import SIMULATION
from analysis.protocols import SPLEEN
from analysis.protocols import THYMUS
from analysis.sampling import section_barcode_sample
from analysis.sampling import section_sample_mask
from datetime import datetime
from data_io import saved_assignments as sa
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import Any
import hashlib
import json
import numpy as np
import pandas as pd
import shutil

from data_io.comparison_inputs import mouse_spleen_MethodData, MOUSE_SPLEEN_SECTIONS, MOUSE_SPLEEN_SECTION_COUNTS, MOUSE_SPLEEN_N_OBS, human_lymph_node_MethodData
from analysis.comparison_readers import mousebrain_MethodData, MOUSEBRAIN_SECTIONS, MOUSEBRAIN_GROUP_LABEL_KEY, MOUSEBRAIN_LABEL_KEYS, MOUSEBRAIN_SECTION_COUNTS, simulation_MethodData, SIMULATION_SECTIONS, mouse_thymus_MethodData, MOUSE_THYMUS_SECTIONS, MOUSE_THYMUS_SECTION_COUNTS, MOUSE_THYMUS_N_OBS, misar_seq_MethodData, MISAR_SEQ_SECTIONS, MISAR_SEQ_LABEL_KEYS, MISAR_SEQ_SECTION_COUNTS

MOUSE_SPLEEN_SEED = SPLEEN['SEED']

MOUSE_SPLEEN_SCRIPT_PATH = Path(__file__).resolve()

MOUSE_SPLEEN_METRIC_KS = SPLEEN['METRIC_KS']

MOUSE_SPLEEN_PLOT_KS = [5, 8, 10, 12]

MOUSE_SPLEEN_PRUNED_CLUSTERING_KS = [k for k in MOUSE_SPLEEN_METRIC_KS if k not in MOUSE_SPLEEN_PLOT_KS]

MOUSE_SPLEEN_MAX_ITER = SPLEEN['MAX_ITER']

MOUSE_SPLEEN_N_INIT = SPLEEN['N_INIT']

MOUSE_SPLEEN_SPATIAL_NEIGHBOR_K = 6

MOUSE_SPLEEN_PLOT_DPI = 220

MOUSE_SPLEEN_PLOT_POINT_SIZE = 10.0

MOUSEBRAIN_SEED = MOUSEBRAIN['SEED']

MOUSEBRAIN_METRIC_KS = MOUSEBRAIN['METRIC_KS']

MOUSEBRAIN_PLOT_KS = [5, 6, 8, 10]

MOUSEBRAIN_PRUNED_CLUSTERING_KS = [k for k in MOUSEBRAIN_METRIC_KS if k not in MOUSEBRAIN_PLOT_KS]

MOUSEBRAIN_MAX_ITER = MOUSEBRAIN['MAX_ITER']

MOUSEBRAIN_SPATIAL_NEIGHBOR_K = 6

MOUSEBRAIN_N_INIT = MOUSEBRAIN['N_INIT']

MOUSEBRAIN_POINT_SIZE = 10.0

MOUSEBRAIN_DPI = 220

SIMULATION_SEED = SIMULATION['SEED']

SIMULATION_POINT_SIZE = 28.0

SIMULATION_MAX_ITER = SIMULATION['MAX_ITER']

SIMULATION_SPATIAL_NEIGHBOR_K = 6

SIMULATION_N_INIT = SIMULATION['N_INIT']

SIMULATION_KS = SIMULATION['KS']

SIMULATION_DPI = 180

MOUSE_THYMUS_SCRIPT_PATH = Path(__file__).resolve()

MOUSE_THYMUS_SPATIAL_NEIGHBOR_K = 6

MOUSE_THYMUS_PLOT_DPI = 220

MOUSE_THYMUS_SEED = THYMUS['SEED']

MOUSE_THYMUS_MAX_ITER = THYMUS['MAX_ITER']

MOUSE_THYMUS_N_INIT = THYMUS['N_INIT']

MOUSE_THYMUS_POINT_SIZES = {'Mouse_Thymus1': 8.0, 'Mouse_Thymus2': 5.0, 'Mouse_Thymus3': 5.0, 'Mouse_Thymus4': 5.0}

MOUSE_THYMUS_METRIC_KS = THYMUS['METRIC_KS']

MOUSE_THYMUS_ASW_SECTION_COUNTS = THYMUS['ASW_SECTION_COUNTS']

MOUSE_THYMUS_ASW_N_OBS = sum(MOUSE_THYMUS_ASW_SECTION_COUNTS.values())

MOUSE_THYMUS_PLOT_KS = [5, 8, 10, 12]

MOUSE_THYMUS_PRUNED_CLUSTERING_KS = [k for k in MOUSE_THYMUS_METRIC_KS if k not in MOUSE_THYMUS_PLOT_KS]

MISAR_SEQ_SEED = MISAR['SEED']

MISAR_SEQ_SCRIPT_PATH = Path(__file__).resolve()

MISAR_SEQ_METRIC_KS = MISAR['METRIC_KS']

MISAR_SEQ_PLOT_KS = [5, 8, 10, 12, 14, 16]

MISAR_SEQ_MAX_ITER = MISAR['MAX_ITER']

MISAR_SEQ_N_INIT = MISAR['N_INIT']

MISAR_SEQ_SPATIAL_NEIGHBOR_K = 6

MISAR_SEQ_PLOT_DPI = 220

MISAR_SEQ_PLOT_POINT_SIZE = 18.0

def mouse_spleen_save_labels(path: Path, data: mouse_spleen_MethodData, mask: np.ndarray, labels: np.ndarray) -> None:
    pd.DataFrame({'section': data.sections[mask], 'obs_name': data.barcodes[mask], 'cluster': labels.astype(int), 'x': data.coords[mask, 0], 'y': data.coords[mask, 1]}).to_csv(path, index=False)

def human_lymph_node_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def mouse_spleen_plot_spatial(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    return plot_spatial_categories(path, coords, labels, title, point_size=MOUSE_SPLEEN_PLOT_POINT_SIZE, dpi=MOUSE_SPLEEN_PLOT_DPI)

def mouse_spleen_run_scheme(data: mouse_spleen_MethodData, scheme: str) -> None:
    output = data.output_root / scheme
    execute_exact_kmeans_scopes(
        data, scheme, section_order=MOUSE_SPLEEN_SECTIONS,
        joint_ks=MOUSE_SPLEEN_METRIC_KS, independent_ks=MOUSE_SPLEEN_METRIC_KS,
        seed=MOUSE_SPLEEN_SEED, n_init=MOUSE_SPLEEN_N_INIT, max_iter=MOUSE_SPLEEN_MAX_ITER,
        spatial_neighbor_k=MOUSE_SPLEEN_SPATIAL_NEIGHBOR_K,
        joint_plot_ks=MOUSE_SPLEEN_PLOT_KS, independent_plot_ks=MOUSE_SPLEEN_PLOT_KS,
        joint_mask_per_fit=False, save_labels=mouse_spleen_save_labels,
        plot_spatial=mouse_spleen_plot_spatial, spatial_agreement=hln_spatial_agreement,
    )
    config = {'dataset': 'Mouse_Spleen', 'method': data.name, 'preprocessing': scheme, 'kmeans_input': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'metric_space': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'standardization': 'none' if scheme == 'raw_embedding' else 'sklearn.preprocessing.StandardScaler', 'joint_scaler_scope': 'none' if scheme == 'raw_embedding' else 'all_5336_spots', 'independent_scaler_scope': 'none' if scheme == 'raw_embedding' else 'fit_per_section', 'kmeans_type': 'sklearn.cluster.KMeans', 'joint_metric_k_values': MOUSE_SPLEEN_METRIC_KS, 'independent_metric_k_values': MOUSE_SPLEEN_METRIC_KS, 'spatial_metric_k_values': MOUSE_SPLEEN_METRIC_KS, 'plot_k_values': MOUSE_SPLEEN_PLOT_KS, 'plot_point_size': MOUSE_SPLEEN_PLOT_POINT_SIZE, 'plot_dpi': MOUSE_SPLEEN_PLOT_DPI, 'seed': MOUSE_SPLEEN_SEED, 'n_init': MOUSE_SPLEEN_N_INIT, 'max_iter': MOUSE_SPLEEN_MAX_ITER, 'n_obs': len(data.embedding), 'section_counts': MOUSE_SPLEEN_SECTION_COUNTS, 'embedding_dim': data.embedding.shape[1], 'cluster_asw_sample_size': 0, 'cluster_asw_rule': 'full_n_obs_in_each_scope', 'ch_dbi_sample_size': 0, 'ch_dbi_rule': 'full_n_obs_in_each_scope', 'label_asw': 'not_applicable_no_ground_truth_labels', 'spatial_neighbor_k': MOUSE_SPLEEN_SPATIAL_NEIGHBOR_K, 'spatial_graph_scope': 'exact_knn_within_each_section', 'spatial_graph_n_obs': MOUSE_SPLEEN_SECTION_COUNTS, 'joint_and_independent_labels_cover_all_spots': True, 'internal_spatial_and_figures_reuse_saved_labels': True, 'source_data_dir': str(data.data_dir), 'source_files': [{'path': str(path), 'sha256': human_lymph_node_sha256(path)} for path in data.source_paths], 'generation_script': str(MOUSE_SPLEEN_SCRIPT_PATH), 'generation_script_sha256': human_lymph_node_sha256(MOUSE_SPLEEN_SCRIPT_PATH), 'created_at': datetime.now().astimezone().isoformat()}
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    (output / 'SUMMARY.md').write_text(f'# {data.name}: {scheme}\n\n- spots: {MOUSE_SPLEEN_N_OBS}; dimensions: {data.embedding.shape[1]}\n- exact KMeans: seed={MOUSE_SPLEEN_SEED}, n_init={MOUSE_SPLEEN_N_INIT}, max_iter={MOUSE_SPLEEN_MAX_ITER}\n- joint/independent metrics: K={MOUSE_SPLEEN_METRIC_KS}\n- retained clustering directories and full labels: K={MOUSE_SPLEEN_PLOT_KS}\n- spatial figures: K={MOUSE_SPLEEN_PLOT_KS}; point size={MOUSE_SPLEEN_PLOT_POINT_SIZE:g}\n- metrics for K={MOUSE_SPLEEN_PRUNED_CLUSTERING_KS} remain in CSV, but their historical labels_path targets were removed\n- Cluster ASW, CH and DBI: full samples in each scope\n- Label ASW: not applicable because no ground-truth label exists\n- spatial continuity: exact {MOUSE_SPLEEN_SPATIAL_NEIGHBOR_K}-NN per section\n- retained-K metrics and every figure reuse saved full label CSVs\n')

def mouse_spleen_copy_shared(data: mouse_spleen_MethodData) -> None:
    output = data.output_root / 'shared_metrics'
    output.mkdir()
    rows = []
    for (name, source) in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        rows.append({'artifact': name, 'source': str(source), 'source_sha256': human_lymph_node_sha256(source), 'copied_sha256': human_lymph_node_sha256(target), 'preprocessing_dependent': False})
    pd.DataFrame(rows).to_csv(output / 'source_manifest.csv', index=False)

def mousebrain_save_labels(path: Path, data: mousebrain_MethodData, mask: np.ndarray, labels: np.ndarray) -> None:
    frame: dict[str, Any] = {'section': data.sections[mask], 'obs_name': data.barcodes[mask], 'cluster': labels.astype(int), 'x': data.coords[mask, 0], 'y': data.coords[mask, 1]}
    for label in MOUSEBRAIN_LABEL_KEYS:
        frame[label] = data.truth[label][mask]
    pd.DataFrame(frame).to_csv(path, index=False)

def mousebrain_label_metrics(data, space, labels, base_mask, mode, scope, k, label_asw_cache):
    return comparison_label_rows(data, space, labels, base_mask, mode, scope, k, label_asw_cache, label_keys=MOUSEBRAIN_LABEL_KEYS)

def mousebrain_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def mousebrain_spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=MOUSEBRAIN_SPATIAL_NEIGHBOR_K + 1).fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))

def mousebrain_write_scheme_summary(data: mousebrain_MethodData, scheme: str) -> None:
    (data.output_root / scheme / 'SUMMARY.md').write_text(f'# {data.name}: {scheme}\n\n- spots: 7866; dimensions: {data.embedding.shape[1]}\n- KMeans: sklearn exact, seed={MOUSEBRAIN_SEED}, n_init={MOUSEBRAIN_N_INIT}, max_iter={MOUSEBRAIN_MAX_ITER}\n- joint metric K: {MOUSEBRAIN_METRIC_KS}; independent metric K: {MOUSEBRAIN_PLOT_KS}\n- retained clustering directories/labels/plots K: {MOUSEBRAIN_PLOT_KS}\n- deleted joint clustering directory K: {MOUSEBRAIN_PRUNED_CLUSTERING_KS}; metric CSV rows remain, so their historical labels_path values no longer resolve\n- external labels: five biological annotations plus group, where group is the aligned s1/s2/s3 section-source diagnostic\n- internal and label ASW: full valid samples in KMeans metric space\n- spatial continuity: exact {MOUSEBRAIN_SPATIAL_NEIGHBOR_K}-NN per section\n- for retained K, figures and metrics reuse the saved label CSV files\n')

def mousebrain_plot_spatial(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    return plot_spatial_categories(path, coords, labels, title, point_size=MOUSEBRAIN_POINT_SIZE, dpi=MOUSEBRAIN_DPI)

def mousebrain_run_scheme(data: mousebrain_MethodData, scheme: str) -> None:
    out = data.output_root / scheme
    cluster_root = out / 'clustering'
    metrics_root = out / 'metrics'
    cluster_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scalers: dict[str, np.ndarray] = {}
    label_asw_cache: dict[tuple[str, str], float] = {}
    (joint_space, joint_scaler) = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scalers.update(scaler_payload(joint_scaler, 'joint'))
    section_asw = safe_asw(joint_space, data.sections)
    all_mask = np.ones(len(data.embedding), dtype=bool)
    for k in MOUSEBRAIN_METRIC_KS:
        labels_all = kmeans_labels(joint_space, n_clusters=k, random_state=MOUSEBRAIN_SEED, n_init=MOUSEBRAIN_N_INIT, max_iter=MOUSEBRAIN_MAX_ITER)
        kdir = cluster_root / f'joint_k{k}'
        kdir.mkdir()
        all_path = kdir / 'labels_all.csv'
        mousebrain_save_labels(all_path, data, all_mask, labels_all)
        internal_rows.append({'mode': 'joint', 'scope': 'combined', 'k': k, 'n_obs': len(labels_all), 'embedding_dim': data.embedding.shape[1], **internal_metrics(joint_space, labels_all), 'section_ari': adjusted_rand_score(data.sections, labels_all), 'section_nmi': normalized_mutual_info_score(data.sections, labels_all), 'section_asw': section_asw, 'metric_space': scheme, 'labels_path': str(all_path)})
        external_rows.extend(mousebrain_label_metrics(data, joint_space, labels_all, all_mask, 'joint', 'combined', k, label_asw_cache))
        count_rows = []
        for section in MOUSEBRAIN_SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            label_path = kdir / f'labels_{section}.csv'
            mousebrain_save_labels(label_path, data, mask, labels)
            if k in MOUSEBRAIN_PLOT_KS:
                mousebrain_plot_spatial(kdir / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} joint K={k} {section}')
            spatial_rows.append({'mode': 'joint', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MOUSEBRAIN_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': mousebrain_spatial_agreement(data.coords[mask], labels), 'labels_path': str(label_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                count_rows.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        pd.DataFrame(count_rows).to_csv(kdir / 'cluster_counts.csv', index=False)
    for k in MOUSEBRAIN_PLOT_KS:
        kdir = cluster_root / f'independent_k{k}'
        kdir.mkdir()
        count_rows = []
        for section in MOUSEBRAIN_SECTIONS:
            mask = data.sections == section
            (space, scaler) = fitted_space(data.embedding[mask], scheme)
            if scaler is not None:
                scalers.update(scaler_payload(scaler, section))
            labels = kmeans_labels(space, n_clusters=k, random_state=MOUSEBRAIN_SEED, n_init=MOUSEBRAIN_N_INIT, max_iter=MOUSEBRAIN_MAX_ITER)
            label_path = kdir / f'labels_{section}.csv'
            mousebrain_save_labels(label_path, data, mask, labels)
            mousebrain_plot_spatial(kdir / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} independent K={k} {section}')
            internal_rows.append({'mode': 'independent', 'scope': section, 'k': k, 'n_obs': int(mask.sum()), 'embedding_dim': data.embedding.shape[1], **internal_metrics(space, labels), 'section_ari': float('nan'), 'section_nmi': float('nan'), 'section_asw': float('nan'), 'metric_space': scheme, 'labels_path': str(label_path)})
            external_rows.extend(mousebrain_label_metrics(data, space, labels, mask, 'independent', section, k, label_asw_cache))
            spatial_rows.append({'mode': 'independent', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MOUSEBRAIN_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': mousebrain_spatial_agreement(data.coords[mask], labels), 'labels_path': str(label_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                count_rows.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        pd.DataFrame(count_rows).to_csv(kdir / 'cluster_counts.csv', index=False)
    metrics = pd.DataFrame(internal_rows)
    external_frame = pd.DataFrame(external_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / 'clustering_metrics.csv', index=False)
    external_frame.to_csv(metrics_root / 'clustering_metrics_by_label.csv', index=False)
    joint_external = external_frame[external_frame['mode'] == 'joint']
    best = joint_external.loc[joint_external.groupby('label')['ari'].idxmax()].sort_values('label')
    best.to_csv(metrics_root / 'best_clustering_metrics_by_label.csv', index=False)
    spatial.to_csv(metrics_root / 'spatial_continuity.csv', index=False)
    spatial.groupby(['mode', 'k'], as_index=False)['neighbor_same_cluster_fraction'].mean().rename(columns={'neighbor_same_cluster_fraction': 'mean_spatial_neighbor_agreement'}).to_csv(metrics_root / 'spatial_continuity_summary.csv', index=False)
    if scalers:
        np.savez_compressed(out / 'scaler_parameters.npz', **scalers)
    config = {'dataset': 'MouseBrain', 'method': data.name, 'preprocessing': scheme, 'kmeans_input': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'metric_space': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'standardization': 'none' if scheme == 'raw_embedding' else 'sklearn.preprocessing.StandardScaler', 'joint_scaler_scope': 'none' if scheme == 'raw_embedding' else 'all_7866_spots', 'independent_scaler_scope': 'none' if scheme == 'raw_embedding' else 'fit_per_section', 'kmeans_type': 'sklearn.cluster.KMeans', 'metric_k_values': MOUSEBRAIN_METRIC_KS, 'plot_and_independent_k_values': MOUSEBRAIN_PLOT_KS, 'retained_clustering_k_values': MOUSEBRAIN_PLOT_KS, 'deleted_clustering_k_values': MOUSEBRAIN_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': MOUSEBRAIN_PRUNED_CLUSTERING_KS, 'seed': MOUSEBRAIN_SEED, 'n_init': MOUSEBRAIN_N_INIT, 'max_iter': MOUSEBRAIN_MAX_ITER, 'n_obs': len(data.embedding), 'section_counts': MOUSEBRAIN_SECTION_COUNTS, 'embedding_dim': data.embedding.shape[1], 'asw_sample_size': 0, 'asw_sample_rule': 'full_valid_n_obs', 'ch_dbi_sample_size': 0, 'ch_dbi_sample_rule': 'full_n_obs', 'label_keys': MOUSEBRAIN_LABEL_KEYS, 'label_asw_rule': 'full_nonmissing_labels_in_same_metric_space', 'label_valid_counts': {label: int(comparison_valid_truth(data.truth[label]).sum()) for label in MOUSEBRAIN_LABEL_KEYS}, 'spatial_neighbor_k': MOUSEBRAIN_SPATIAL_NEIGHBOR_K, 'spatial_graph_scope': 'fit_separately_within_each_section', 'spatial_graph_n_obs': MOUSEBRAIN_SECTION_COUNTS, 'plots_and_metrics_reuse_same_label_files': False, 'retained_k_plots_and_metrics_reuse_same_label_files': True, 'source_data_dir': str(data.data_dir), 'source_files': [{'path': str(path), 'sha256': mousebrain_sha256(path)} for path in data.source_paths], 'created_at': datetime.now().astimezone().isoformat()}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    mousebrain_write_scheme_summary(data, scheme)

def mousebrain_copy_shared(data: mousebrain_MethodData) -> None:
    out = data.output_root / 'shared_metrics'
    out.mkdir()
    rows = []
    for (name, source) in data.shared_sources.items():
        target = out / name
        shutil.copy2(source, target)
        rows.append({'artifact': name, 'source': str(source), 'source_sha256': mousebrain_sha256(source), 'copied_sha256': mousebrain_sha256(target), 'preprocessing_dependent': False})
    pd.DataFrame(rows).to_csv(out / 'source_manifest.csv', index=False)

def mousebrain_labels_vector_sha256(labels: np.ndarray) -> str:
    values = np.ascontiguousarray(labels, dtype='<i8')
    return hashlib.sha256(values.tobytes()).hexdigest()

def mousebrain_add_group_diagnostics(data: mousebrain_MethodData) -> None:
    all_mask = np.ones(len(data.embedding), dtype=bool)
    group_values = data.truth[MOUSEBRAIN_GROUP_LABEL_KEY].astype(str)
    if not np.array_equal(group_values, data.sections.astype(str)):
        raise ValueError(f'{data.name}: group is not identical to section')
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        metrics_root = output / 'metrics'
        (space, _) = fitted_space(data.embedding, scheme)
        group_asw = safe_asw(space, group_values)
        internal_frame = pd.read_csv(metrics_root / 'clustering_metrics.csv')
        joint_internal = internal_frame[internal_frame['mode'] == 'joint'].set_index('k').sort_index()
        if set(joint_internal.index.astype(int)) != set(MOUSEBRAIN_METRIC_KS):
            raise ValueError(f'{data.name} {scheme}: incomplete joint metric K values')
        if not np.allclose(joint_internal['section_asw'].to_numpy(float), group_asw, rtol=0.0, atol=1e-12):
            raise ValueError(f'{data.name} {scheme}: group ASW does not match section ASW')
        group_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        for k in MOUSEBRAIN_METRIC_KS:
            labels_path = output / 'clustering' / f'joint_k{k}' / 'labels_all.csv'
            if labels_path.exists():
                saved = sa.read_assignment_table(labels_path)
                expected_order = pd.DataFrame({'section': data.sections.astype(str), 'obs_name': data.barcodes.astype(str)})
                if not saved[['section', 'obs_name']].astype(str).equals(expected_order):
                    raise ValueError(f'{data.name} {scheme} K={k}: labels row mismatch')
                labels = saved['cluster'].to_numpy(int)
                labels_source = 'retained_labels_csv'
                labels_file_sha256 = mousebrain_sha256(labels_path)
                labels_path_value = str(labels_path)
            else:
                labels = kmeans_labels(space, n_clusters=k, random_state=MOUSEBRAIN_SEED, n_init=MOUSEBRAIN_N_INIT, max_iter=MOUSEBRAIN_MAX_ITER)
                labels_source = 'deterministic_kmeans_recomputed_for_group_diagnostic'
                labels_file_sha256 = ''
                labels_path_value = ''
            group_metrics = external_metrics(group_values, labels)
            expected = joint_internal.loc[k]
            if not np.isclose(group_metrics['ari'], float(expected['section_ari']), rtol=0.0, atol=1e-12) or not np.isclose(group_metrics['nmi'], float(expected['section_nmi']), rtol=0.0, atol=1e-12):
                raise ValueError(f'{data.name} {scheme} K={k}: recomputed group ARI/NMI does not match existing section diagnostics')
            common = {'mode': 'joint', 'scope': 'combined', 'label': MOUSEBRAIN_GROUP_LABEL_KEY, 'n_obs_labeled': len(group_values), 'n_label_classes': len(np.unique(group_values)), 'k': k, **group_metrics, 'label_asw': group_asw, 'label_asw_scaled': (group_asw + 1.0) / 2.0, 'label_rows_in_full_order': '0,1,2,3,4'}
            group_rows.append(common)
            provenance_rows.append({**common, 'cluster_asw': float(expected['cluster_asw']), 'cluster_asw_scaled': float(expected['cluster_asw_scaled']), 'group_definition': 'aligned section vector: s1/s2/s3', 'labels_source': labels_source, 'labels_path': labels_path_value, 'labels_file_sha256': labels_file_sha256, 'labels_vector_sha256': mousebrain_labels_vector_sha256(labels), 'seed': MOUSEBRAIN_SEED, 'n_init': MOUSEBRAIN_N_INIT, 'max_iter': MOUSEBRAIN_MAX_ITER, 'metric_space': scheme})
        external_path = metrics_root / 'clustering_metrics_by_label.csv'
        external_frame = pd.read_csv(external_path)
        external_frame = external_frame[external_frame['label'] != MOUSEBRAIN_GROUP_LABEL_KEY]
        external_frame = pd.concat([external_frame, pd.DataFrame(group_rows)], ignore_index=True)
        external_frame.to_csv(external_path, index=False)
        joint_external = external_frame[external_frame['mode'] == 'joint']
        best = joint_external.loc[joint_external.groupby('label')['ari'].idxmax()].sort_values('label')
        best.to_csv(metrics_root / 'best_clustering_metrics_by_label.csv', index=False)
        pd.DataFrame(provenance_rows).to_csv(metrics_root / 'group_diagnostic_metrics.csv', index=False)
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'label_keys': MOUSEBRAIN_LABEL_KEYS, 'label_valid_counts': {label: int(comparison_valid_truth(data.truth[label]).sum()) for label in MOUSEBRAIN_LABEL_KEYS}, 'group_definition': 'aligned section vector: s1/s2/s3', 'group_diagnostic_k_values': MOUSEBRAIN_METRIC_KS, 'group_diagnostic_reuses_retained_labels_for_k': MOUSEBRAIN_PLOT_KS, 'group_diagnostic_recomputes_labels_without_saving_for_k': MOUSEBRAIN_PRUNED_CLUSTERING_KS, 'group_diagnostics_added_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        mousebrain_write_scheme_summary(data, scheme)
    raw_external = pd.read_csv(data.output_root / 'raw_embedding' / 'metrics' / 'clustering_metrics_by_label.csv')
    standardized_external = pd.read_csv(data.output_root / 'standardized_embedding' / 'metrics' / 'clustering_metrics_by_label.csv')
    keys = ['mode', 'scope', 'label', 'n_obs_labeled', 'n_label_classes', 'k']
    metrics = ['ari', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'label_asw', 'label_asw_scaled']
    merged = raw_external.merge(standardized_external, on=keys, suffixes=('_raw', '_standardized'), validate='one_to_one')
    for metric in metrics:
        merged[f'{metric}_delta_standardized_minus_raw'] = merged[f'{metric}_standardized'] - merged[f'{metric}_raw']
    comparison_root = data.output_root / 'comparison_metrics'
    merged.to_csv(comparison_root / 'clustering_metrics_by_label_deltas.csv', index=False)
    raw_group = pd.read_csv(data.output_root / 'raw_embedding' / 'metrics' / 'group_diagnostic_metrics.csv')
    standardized_group = pd.read_csv(data.output_root / 'standardized_embedding' / 'metrics' / 'group_diagnostic_metrics.csv')
    group_keys = ['mode', 'scope', 'label', 'n_obs_labeled', 'n_label_classes', 'k']
    group_metrics = ['ari', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'label_asw', 'label_asw_scaled', 'cluster_asw', 'cluster_asw_scaled']
    group_delta = raw_group.merge(standardized_group, on=group_keys, suffixes=('_raw', '_standardized'), validate='one_to_one')
    for metric in group_metrics:
        group_delta[f'{metric}_delta_standardized_minus_raw'] = group_delta[f'{metric}_standardized'] - group_delta[f'{metric}_raw']
    group_delta.to_csv(comparison_root / 'group_diagnostic_metrics_deltas.csv', index=False)
    best_rows = []
    for (scheme, scheme_label) in (('raw_embedding', 'Raw'), ('standardized_embedding', 'Standardized')):
        frame = pd.read_csv(data.output_root / scheme / 'metrics' / 'group_diagnostic_metrics.csv')
        row = frame.loc[frame['ari'].idxmax()].to_dict()
        row['preprocessing'] = scheme_label
        best_rows.append(row)
    pd.DataFrame(best_rows).to_csv(comparison_root / 'group_best_diagnostic_comparison.csv', index=False)

def simulation_save_labels(path: Path, data: simulation_MethodData, mask: np.ndarray, labels: np.ndarray) -> None:
    pd.DataFrame({'section': data.sections[mask], 'obs_name': data.barcodes[mask], 'cluster': labels.astype(int), 'spatial_domain': data.truth[mask], 'x': data.coords[mask, 0], 'y': data.coords[mask, 1]}).to_csv(path, index=False)

def simulation_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def simulation_spatial_agreement(coords: np.ndarray, labels: np.ndarray) -> float:
    neighbors = NearestNeighbors(n_neighbors=SIMULATION_SPATIAL_NEIGHBOR_K + 1)
    neighbors.fit(coords[:, :2])
    idx = neighbors.kneighbors(coords[:, :2], return_distance=False)[:, 1:]
    return float(np.mean(labels[idx] == labels[:, None]))

def simulation_plot_spatial(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    return plot_spatial_categories(path, coords, labels, title, point_size=SIMULATION_POINT_SIZE, dpi=SIMULATION_DPI)

def simulation_run_scheme(data: simulation_MethodData, scheme: str) -> None:
    out = data.output_root / scheme
    clustering_dir = out / 'clustering'
    metrics_dir = out / 'metrics'
    clustering_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    metric_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    (joint_space, joint_scaler) = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, 'joint'))
    label_asw = safe_asw(joint_space, data.truth)
    section_asw = safe_asw(joint_space, data.sections)
    for k in SIMULATION_KS:
        labels_all = kmeans_labels(joint_space, n_clusters=k, random_state=SIMULATION_SEED, n_init=SIMULATION_N_INIT, max_iter=SIMULATION_MAX_ITER).astype(int)
        joint_dir = clustering_dir / f'joint_k{k}'
        joint_dir.mkdir()
        row = {'mode': 'joint', 'scope': 'combined', 'k': k, 'n_obs': len(labels_all), 'embedding_dim': data.embedding.shape[1], **external_metrics(data.truth, labels_all), **simulation_internal_metrics(joint_space, labels_all), 'label_asw': label_asw, 'label_asw_scaled': (label_asw + 1.0) / 2.0, 'section_ari': adjusted_rand_score(data.sections, labels_all), 'section_nmi': normalized_mutual_info_score(data.sections, labels_all), 'section_asw': section_asw}
        metric_rows.append(row)
        counts = []
        for section in SIMULATION_SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            simulation_save_labels(joint_dir / f'labels_{section}.csv', data, mask, labels)
            simulation_plot_spatial(joint_dir / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} joint K={k} {section}')
            agreement = simulation_spatial_agreement(data.coords[mask], labels)
            continuity_rows.append({'mode': 'joint', 'k': k, 'section': section, 'neighbor_same_cluster_fraction': agreement})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                counts.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': float(count / len(labels))})
        pd.DataFrame(counts).to_csv(joint_dir / 'cluster_counts.csv', index=False)
        independent_dir = clustering_dir / f'independent_k{k}'
        independent_dir.mkdir()
        for section in SIMULATION_SECTIONS:
            mask = data.sections == section
            sec_raw = data.embedding[mask]
            (sec_space, sec_scaler) = fitted_space(sec_raw, scheme)
            if sec_scaler is not None:
                key = section.replace(' ', '_')
                if f'{key}_mean' not in scaler_arrays:
                    scaler_arrays.update(scaler_payload(sec_scaler, key))
            labels = kmeans_labels(sec_space, n_clusters=k, random_state=SIMULATION_SEED, n_init=SIMULATION_N_INIT, max_iter=SIMULATION_MAX_ITER).astype(int)
            simulation_save_labels(independent_dir / f'labels_{section}.csv', data, mask, labels)
            simulation_plot_spatial(independent_dir / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} independent K={k} {section}')
            agreement = simulation_spatial_agreement(data.coords[mask], labels)
            continuity_rows.append({'mode': 'independent', 'k': k, 'section': section, 'neighbor_same_cluster_fraction': agreement})
            sec_label_asw = safe_asw(sec_space, data.truth[mask])
            metric_rows.append({'mode': 'independent', 'scope': section, 'k': k, 'n_obs': int(mask.sum()), 'embedding_dim': data.embedding.shape[1], **external_metrics(data.truth[mask], labels), **simulation_internal_metrics(sec_space, labels), 'label_asw': sec_label_asw, 'label_asw_scaled': (sec_label_asw + 1.0) / 2.0, 'section_ari': float('nan'), 'section_nmi': float('nan'), 'section_asw': float('nan')})
    metrics = pd.DataFrame(metric_rows)
    continuity = pd.DataFrame(continuity_rows)
    metrics.to_csv(metrics_dir / 'clustering_metrics.csv', index=False)
    continuity.to_csv(metrics_dir / 'spatial_continuity.csv', index=False)
    continuity.groupby(['mode', 'k'], as_index=False)['neighbor_same_cluster_fraction'].mean().rename(columns={'neighbor_same_cluster_fraction': 'mean_spatial_neighbor_agreement'}).to_csv(metrics_dir / 'spatial_continuity_summary.csv', index=False)
    if scaler_arrays:
        np.savez_compressed(out / 'scaler_parameters.npz', **scaler_arrays)
    config = {'dataset': 'Simulation', 'method': data.name, 'preprocessing': scheme, 'kmeans_input': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'metric_space': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'joint_scaler_scope': 'none' if scheme == 'raw_embedding' else 'all_6480_spots', 'independent_scaler_scope': 'none' if scheme == 'raw_embedding' else 'fit_per_section', 'k_values': SIMULATION_KS, 'seed': SIMULATION_SEED, 'n_init': SIMULATION_N_INIT, 'max_iter': SIMULATION_MAX_ITER, 'n_obs': len(data.embedding), 'embedding_dim': data.embedding.shape[1], 'spatial_neighbor_k': SIMULATION_SPATIAL_NEIGHBOR_K, 'point_size': SIMULATION_POINT_SIZE, 'all_spots_used': True, 'data_dir': str(data.data_dir), 'source_files': [{'path': str(path), 'sha256': simulation_sha256(path)} for path in data.source_paths]}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    joint = metrics[metrics['mode'] == 'joint']
    best = joint.loc[joint['ari'].idxmax()]
    summary = f"# {data.name} Simulation {scheme} KMeans analysis\n\n- Spots: {len(data.embedding):,}\n- Embedding dimensions: {data.embedding.shape[1]}\n- K: {SIMULATION_KS}\n- KMeans: seed={SIMULATION_SEED}, n_init={SIMULATION_N_INIT}, max_iter={SIMULATION_MAX_ITER}\n- Metric space: `{config['metric_space']}`\n- Best joint ARI: {best['ari']:.6f} at K={int(best['k'])}\n- Best-row NMI: {best['nmi']:.6f}\n- Labels and spatial plots: `clustering/`\n- Complete metrics: `metrics/`\n"
    (out / 'SUMMARY.md').write_text(summary, encoding='utf-8')

def simulation_copy_shared_metrics(data: simulation_MethodData) -> None:
    shared = data.output_root / 'shared_metrics'
    shared.mkdir(parents=True)
    source_manifest = {}
    for (filename, source) in data.shared_sources.items():
        target = shared / filename
        shutil.copy2(source, target)
        source_manifest[filename] = {'source': str(source), 'source_sha256': simulation_sha256(source), 'copied_sha256': simulation_sha256(target), 'preprocessing_dependency': 'independent_of_kmeans_input'}
    (shared / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding='utf-8')

def mouse_thymus_save_labels(path: Path, data: mouse_thymus_MethodData, mask: np.ndarray, labels: np.ndarray) -> None:
    pd.DataFrame({'section': data.sections[mask], 'obs_name': data.barcodes[mask], 'cluster': labels.astype(int), 'x': data.coords[mask, 0], 'y': data.coords[mask, 1]}).to_csv(path, index=False)

def mouse_thymus_plot_spatial(path: Path, coords: np.ndarray, labels: np.ndarray, title: str, section: str) -> None:
    return plot_spatial_categories(path, coords, labels, title, point_size=MOUSE_THYMUS_POINT_SIZES[section], dpi=MOUSE_THYMUS_PLOT_DPI)

def mouse_thymus_sample_mask(data: mouse_thymus_MethodData, sample: pd.DataFrame) -> np.ndarray:
    return section_sample_mask(data, sample_counts=MOUSE_THYMUS_ASW_SECTION_COUNTS, sample_count=MOUSE_THYMUS_ASW_N_OBS, sample=sample)

def mouse_thymus_run_scheme(data: mouse_thymus_MethodData, scheme: str, asw_sample: pd.DataFrame) -> None:
    output = data.output_root / scheme
    clustering_root = output / 'clustering'
    metrics_root = output / 'metrics'
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    fixed_mask = mouse_thymus_sample_mask(data, asw_sample)
    all_mask = np.ones(len(data.embedding), dtype=bool)
    (joint_space, joint_scaler) = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, 'joint'))
    joint_sample_indices = np.flatnonzero(fixed_mask)
    joint_distances = pairwise_distances(joint_space[joint_sample_indices], metric='euclidean', n_jobs=-1)
    np.fill_diagonal(joint_distances, 0.0)
    section_asw = safe_precomputed_asw(joint_distances, data.sections[joint_sample_indices])
    for k in MOUSE_THYMUS_METRIC_KS:
        labels_all = kmeans_labels(joint_space, n_clusters=k, random_state=MOUSE_THYMUS_SEED, n_init=MOUSE_THYMUS_N_INIT, max_iter=MOUSE_THYMUS_MAX_ITER)
        k_directory = clustering_root / f'joint_k{k}'
        k_directory.mkdir()
        all_labels_path = k_directory / 'labels_all.csv'
        mouse_thymus_save_labels(all_labels_path, data, all_mask, labels_all)
        internal_rows.append({'mode': 'joint', 'scope': 'combined', 'k': k, 'n_obs': len(labels_all), 'embedding_dim': data.embedding.shape[1], **sampled_internal_metrics(joint_space, labels_all, joint_sample_indices, joint_distances), 'section_ari': adjusted_rand_score(data.sections, labels_all), 'section_nmi': normalized_mutual_info_score(data.sections, labels_all), 'section_asw': section_asw, 'section_asw_n_obs': MOUSE_THYMUS_ASW_N_OBS, 'metric_space': scheme, 'labels_path': str(all_labels_path)})
        count_rows = []
        for section in MOUSE_THYMUS_SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            labels_path = k_directory / f'labels_{section}.csv'
            mouse_thymus_save_labels(labels_path, data, mask, labels)
            if k in MOUSE_THYMUS_PLOT_KS:
                mouse_thymus_plot_spatial(k_directory / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} joint K={k} {section}', section)
            spatial_rows.append({'mode': 'joint', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MOUSE_THYMUS_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': hln_spatial_agreement(data.coords[mask], labels), 'labels_path': str(labels_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                count_rows.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        pd.DataFrame(count_rows).to_csv(k_directory / 'cluster_counts.csv', index=False)
    del joint_distances
    independent_counts: dict[int, list[dict[str, Any]]] = {k: [] for k in MOUSE_THYMUS_METRIC_KS}
    for k in MOUSE_THYMUS_METRIC_KS:
        (clustering_root / f'independent_k{k}').mkdir()
    for section in MOUSE_THYMUS_SECTIONS:
        mask = data.sections == section
        (section_space, scaler) = fitted_space(data.embedding[mask], scheme)
        if scaler is not None:
            scaler_arrays.update(scaler_payload(scaler, section))
        local_sample_mask = fixed_mask[mask]
        local_sample_indices = np.flatnonzero(local_sample_mask)
        local_distances = pairwise_distances(section_space[local_sample_indices], metric='euclidean', n_jobs=-1)
        np.fill_diagonal(local_distances, 0.0)
        for k in MOUSE_THYMUS_METRIC_KS:
            labels = kmeans_labels(section_space, n_clusters=k, random_state=MOUSE_THYMUS_SEED, n_init=MOUSE_THYMUS_N_INIT, max_iter=MOUSE_THYMUS_MAX_ITER)
            k_directory = clustering_root / f'independent_k{k}'
            labels_path = k_directory / f'labels_{section}.csv'
            mouse_thymus_save_labels(labels_path, data, mask, labels)
            if k in MOUSE_THYMUS_PLOT_KS:
                mouse_thymus_plot_spatial(k_directory / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} independent K={k} {section}', section)
            internal_rows.append({'mode': 'independent', 'scope': section, 'k': k, 'n_obs': int(mask.sum()), 'embedding_dim': data.embedding.shape[1], **sampled_internal_metrics(section_space, labels, local_sample_indices, local_distances), 'section_ari': float('nan'), 'section_nmi': float('nan'), 'section_asw': float('nan'), 'section_asw_n_obs': 0, 'metric_space': scheme, 'labels_path': str(labels_path)})
            spatial_rows.append({'mode': 'independent', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MOUSE_THYMUS_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': hln_spatial_agreement(data.coords[mask], labels), 'labels_path': str(labels_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                independent_counts[k].append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        del local_distances
    for (k, rows) in independent_counts.items():
        pd.DataFrame(rows).to_csv(clustering_root / f'independent_k{k}' / 'cluster_counts.csv', index=False)
    metrics = pd.DataFrame(internal_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / 'clustering_metrics.csv', index=False)
    spatial.to_csv(metrics_root / 'spatial_continuity.csv', index=False)
    spatial.groupby(['mode', 'k'], as_index=False)['neighbor_same_cluster_fraction'].mean().rename(columns={'neighbor_same_cluster_fraction': 'mean_spatial_neighbor_agreement'}).to_csv(metrics_root / 'spatial_continuity_summary.csv', index=False)
    if scaler_arrays:
        np.savez_compressed(output / 'scaler_parameters.npz', **scaler_arrays)
    sample_path = data.output_root / 'shared_metrics' / 'asw_sample.csv'
    config = {'dataset': 'Mouse_Thymus', 'method': data.name, 'preprocessing': scheme, 'kmeans_input': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'metric_space': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'standardization': 'none' if scheme == 'raw_embedding' else 'sklearn.preprocessing.StandardScaler', 'joint_scaler_scope': 'none' if scheme == 'raw_embedding' else 'all_17824_spots', 'independent_scaler_scope': 'none' if scheme == 'raw_embedding' else 'fit_per_section', 'kmeans_type': 'sklearn.cluster.KMeans', 'joint_metric_k_values': MOUSE_THYMUS_METRIC_KS, 'independent_metric_k_values': MOUSE_THYMUS_METRIC_KS, 'spatial_metric_k_values': MOUSE_THYMUS_METRIC_KS, 'plot_k_values': MOUSE_THYMUS_PLOT_KS, 'plot_point_size_by_section': MOUSE_THYMUS_POINT_SIZES, 'plot_dpi': MOUSE_THYMUS_PLOT_DPI, 'seed': MOUSE_THYMUS_SEED, 'n_init': MOUSE_THYMUS_N_INIT, 'max_iter': MOUSE_THYMUS_MAX_ITER, 'n_obs': len(data.embedding), 'section_counts': MOUSE_THYMUS_SECTION_COUNTS, 'embedding_dim': data.embedding.shape[1], 'cluster_asw_sample_size_joint': MOUSE_THYMUS_ASW_N_OBS, 'cluster_asw_sample_size_independent': MOUSE_THYMUS_ASW_SECTION_COUNTS, 'cluster_asw_rule': 'fixed_section_stratified_section_barcode_sample_shared_everywhere', 'cluster_asw_sampling_seed': MOUSE_THYMUS_SEED, 'cluster_asw_sample_file': str(sample_path), 'cluster_asw_sample_sha256': human_lymph_node_sha256(sample_path), 'ch_dbi_sample_size': 0, 'ch_dbi_rule': 'full_n_obs_in_each_scope', 'label_asw': 'not_applicable_no_ground_truth_labels', 'section_asw_sample_size': MOUSE_THYMUS_ASW_N_OBS, 'spatial_neighbor_k': MOUSE_THYMUS_SPATIAL_NEIGHBOR_K, 'spatial_graph_scope': 'exact_knn_within_each_section', 'spatial_graph_n_obs': MOUSE_THYMUS_SECTION_COUNTS, 'joint_and_independent_labels_cover_all_spots': True, 'internal_spatial_and_figures_reuse_saved_labels': True, 'source_data_dir': str(data.data_dir), 'source_files': [{'path': str(path), 'sha256': human_lymph_node_sha256(path)} for path in data.source_paths], 'generation_script': str(MOUSE_THYMUS_SCRIPT_PATH), 'generation_script_sha256': human_lymph_node_sha256(MOUSE_THYMUS_SCRIPT_PATH), 'created_at': datetime.now().astimezone().isoformat()}
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    (output / 'SUMMARY.md').write_text(f'# {data.name}: {scheme}\n\n- spots: {MOUSE_THYMUS_N_OBS}; dimensions: {data.embedding.shape[1]}\n- exact KMeans: seed={MOUSE_THYMUS_SEED}, n_init={MOUSE_THYMUS_N_INIT}, max_iter={MOUSE_THYMUS_MAX_ITER}\n- joint/independent metrics: K={MOUSE_THYMUS_METRIC_KS}\n- retained clustering directories and full labels: K={MOUSE_THYMUS_PLOT_KS}\n- spatial figures: K={MOUSE_THYMUS_PLOT_KS}; point sizes={MOUSE_THYMUS_POINT_SIZES}\n- metrics for K={MOUSE_THYMUS_PRUNED_CLUSTERING_KS} remain in CSV, but their historical labels_path targets were removed\n- Cluster ASW: fixed section-stratified {MOUSE_THYMUS_ASW_N_OBS}-spot sample; independent quotas={MOUSE_THYMUS_ASW_SECTION_COUNTS}\n- CH and DBI: full samples in each scope\n- Label ASW: not applicable because no ground-truth label exists\n- spatial continuity: exact {MOUSE_THYMUS_SPATIAL_NEIGHBOR_K}-NN per section\n- retained-K metrics and every figure reuse saved full label CSVs\n')

def mouse_thymus_prepare_shared(data: mouse_thymus_MethodData, asw_sample: pd.DataFrame) -> None:
    output = data.output_root / 'shared_metrics'
    output.mkdir(parents=True)
    asw_path = output / 'asw_sample.csv'
    asw_sample.to_csv(asw_path, index=False)
    rows = []
    for (name, source) in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        rows.append({'artifact': name, 'source': str(source), 'source_sha256': human_lymph_node_sha256(source), 'copied_sha256': human_lymph_node_sha256(target), 'preprocessing_dependent': False})
    pd.DataFrame(rows).to_csv(output / 'source_manifest.csv', index=False)
    sample_manifest = {'sample_file': str(asw_path), 'sha256': human_lymph_node_sha256(asw_path), 'n_obs': MOUSE_THYMUS_ASW_N_OBS, 'section_counts': MOUSE_THYMUS_ASW_SECTION_COUNTS, 'seed': MOUSE_THYMUS_SEED, 'identity_key': ['section', 'obs_name'], 'selection': 'section-proportional largest-remainder quotas; sorted candidates; numpy default_rng(seed=0) without replacement', 'shared_across_methods_preprocessing_k_and_modes': True}
    (output / 'asw_sample_manifest.json').write_text(json.dumps(sample_manifest, indent=2, ensure_ascii=False) + '\n')

def mouse_thymus_build_asw_sample(methods: list[mouse_thymus_MethodData]) -> pd.DataFrame:
    return section_barcode_sample(methods, section_order=MOUSE_THYMUS_SECTIONS, section_counts=MOUSE_THYMUS_SECTION_COUNTS, sample_counts=MOUSE_THYMUS_ASW_SECTION_COUNTS, sample_count=MOUSE_THYMUS_ASW_N_OBS, seed=MOUSE_THYMUS_SEED)

def misar_seq_save_labels(path: Path, data: misar_seq_MethodData, mask: np.ndarray, labels: np.ndarray) -> None:
    payload: dict[str, Any] = {'section': data.sections[mask], 'obs_name': data.barcodes[mask], 'cluster': labels.astype(int), 'x': data.coords[mask, 0], 'y': data.coords[mask, 1]}
    for label in MISAR_SEQ_LABEL_KEYS:
        payload[label] = data.truth[label][mask]
    pd.DataFrame(payload).to_csv(path, index=False)

def misar_seq_label_metrics(data, space, labels, mask, mode, scope, k, asw_cache):
    return misar_label_rows(data, space, labels, mask, mode, scope, k, asw_cache, label_keys=MISAR_SEQ_LABEL_KEYS)

def misar_seq_plot_spatial(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    return plot_spatial_categories(path, coords, labels, title, point_size=MISAR_SEQ_PLOT_POINT_SIZE, dpi=MISAR_SEQ_PLOT_DPI)

def misar_seq_run_scheme(data: misar_seq_MethodData, scheme: str) -> None:
    output = data.output_root / scheme
    clustering_root = output / 'clustering'
    metrics_root = output / 'metrics'
    clustering_root.mkdir(parents=True)
    metrics_root.mkdir()
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    scaler_arrays: dict[str, np.ndarray] = {}
    label_asw_cache: dict[tuple[str, str], float] = {}
    (joint_space, joint_scaler) = fitted_space(data.embedding, scheme)
    if joint_scaler is not None:
        scaler_arrays.update(scaler_payload(joint_scaler, 'joint'))
    section_asw = safe_asw(joint_space, data.sections)
    all_mask = np.ones(len(data.embedding), dtype=bool)
    for k in MISAR_SEQ_METRIC_KS:
        labels_all = kmeans_labels(joint_space, n_clusters=k, random_state=MISAR_SEQ_SEED, n_init=MISAR_SEQ_N_INIT, max_iter=MISAR_SEQ_MAX_ITER)
        k_directory = clustering_root / f'joint_k{k}'
        k_directory.mkdir()
        all_labels_path = k_directory / 'labels_all.csv'
        misar_seq_save_labels(all_labels_path, data, all_mask, labels_all)
        internal_rows.append({'mode': 'joint', 'scope': 'combined', 'k': k, 'n_obs': len(labels_all), 'embedding_dim': data.embedding.shape[1], **internal_metrics(joint_space, labels_all), 'section_ari': adjusted_rand_score(data.sections, labels_all), 'section_nmi': normalized_mutual_info_score(data.sections, labels_all), 'section_asw': section_asw, 'metric_space': scheme, 'labels_path': str(all_labels_path)})
        external_rows.extend(misar_seq_label_metrics(data, joint_space, labels_all, all_mask, 'joint', 'combined', k, label_asw_cache))
        count_rows = []
        for section in MISAR_SEQ_SECTIONS:
            mask = data.sections == section
            labels = labels_all[mask]
            labels_path = k_directory / f'labels_{section}.csv'
            misar_seq_save_labels(labels_path, data, mask, labels)
            if k in MISAR_SEQ_PLOT_KS:
                misar_seq_plot_spatial(k_directory / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} joint K={k} {section}')
            spatial_rows.append({'mode': 'joint', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MISAR_SEQ_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': mousebrain_spatial_agreement(data.coords[mask], labels), 'labels_path': str(labels_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                count_rows.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        pd.DataFrame(count_rows).to_csv(k_directory / 'cluster_counts.csv', index=False)
    for k in MISAR_SEQ_METRIC_KS:
        k_directory = clustering_root / f'independent_k{k}'
        k_directory.mkdir()
        count_rows = []
        for section in MISAR_SEQ_SECTIONS:
            mask = data.sections == section
            (section_space, scaler) = fitted_space(data.embedding[mask], scheme)
            if scaler is not None:
                scaler_arrays.update(scaler_payload(scaler, section))
            labels = kmeans_labels(section_space, n_clusters=k, random_state=MISAR_SEQ_SEED, n_init=MISAR_SEQ_N_INIT, max_iter=MISAR_SEQ_MAX_ITER)
            labels_path = k_directory / f'labels_{section}.csv'
            misar_seq_save_labels(labels_path, data, mask, labels)
            if k in MISAR_SEQ_PLOT_KS:
                misar_seq_plot_spatial(k_directory / f'spatial_{section}.png', data.coords[mask], labels, f'{data.name} {scheme} independent K={k} {section}')
            internal_rows.append({'mode': 'independent', 'scope': section, 'k': k, 'n_obs': int(mask.sum()), 'embedding_dim': data.embedding.shape[1], **internal_metrics(section_space, labels), 'section_ari': float('nan'), 'section_nmi': float('nan'), 'section_asw': float('nan'), 'metric_space': scheme, 'labels_path': str(labels_path)})
            external_rows.extend(misar_seq_label_metrics(data, section_space, labels, mask, 'independent', section, k, label_asw_cache))
            spatial_rows.append({'mode': 'independent', 'k': k, 'section': section, 'n_obs': int(mask.sum()), 'spatial_neighbor_k': MISAR_SEQ_SPATIAL_NEIGHBOR_K, 'neighbor_same_cluster_fraction': mousebrain_spatial_agreement(data.coords[mask], labels), 'labels_path': str(labels_path)})
            for (cluster, count) in zip(*np.unique(labels, return_counts=True)):
                count_rows.append({'section': section, 'cluster': int(cluster), 'count': int(count), 'fraction': count / int(mask.sum())})
        pd.DataFrame(count_rows).to_csv(k_directory / 'cluster_counts.csv', index=False)
    metrics = pd.DataFrame(internal_rows)
    external_frame = pd.DataFrame(external_rows)
    spatial = pd.DataFrame(spatial_rows)
    metrics.to_csv(metrics_root / 'clustering_metrics.csv', index=False)
    external_frame.to_csv(metrics_root / 'clustering_metrics_by_label.csv', index=False)
    joint_external = external_frame[external_frame['mode'] == 'joint']
    best = joint_external.loc[joint_external.groupby('label')['ari'].idxmax()].sort_values('label')
    best.to_csv(metrics_root / 'best_clustering_metrics_by_label.csv', index=False)
    spatial.to_csv(metrics_root / 'spatial_continuity.csv', index=False)
    spatial.groupby(['mode', 'k'], as_index=False)['neighbor_same_cluster_fraction'].mean().rename(columns={'neighbor_same_cluster_fraction': 'mean_spatial_neighbor_agreement'}).to_csv(metrics_root / 'spatial_continuity_summary.csv', index=False)
    if scaler_arrays:
        np.savez_compressed(output / 'scaler_parameters.npz', **scaler_arrays)
    config = {'dataset': 'MISAR-seq', 'method': data.name, 'preprocessing': scheme, 'kmeans_input': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'metric_space': 'raw_final_embedding' if scheme == 'raw_embedding' else 'standardized_final_embedding', 'standardization': 'none' if scheme == 'raw_embedding' else 'sklearn.preprocessing.StandardScaler', 'joint_scaler_scope': 'none' if scheme == 'raw_embedding' else 'all_7118_spots', 'independent_scaler_scope': 'none' if scheme == 'raw_embedding' else 'fit_per_section', 'kmeans_type': 'sklearn.cluster.KMeans', 'joint_metric_k_values': MISAR_SEQ_METRIC_KS, 'independent_metric_k_values': MISAR_SEQ_METRIC_KS, 'spatial_metric_k_values': MISAR_SEQ_METRIC_KS, 'plot_k_values': MISAR_SEQ_PLOT_KS, 'plot_point_size': MISAR_SEQ_PLOT_POINT_SIZE, 'plot_dpi': MISAR_SEQ_PLOT_DPI, 'seed': MISAR_SEQ_SEED, 'n_init': MISAR_SEQ_N_INIT, 'max_iter': MISAR_SEQ_MAX_ITER, 'n_obs': len(data.embedding), 'section_counts': MISAR_SEQ_SECTION_COUNTS, 'embedding_dim': data.embedding.shape[1], 'cluster_asw_sample_size': 0, 'cluster_asw_rule': 'full_n_obs_in_each_scope', 'ch_dbi_sample_size': 0, 'ch_dbi_rule': 'full_n_obs_in_each_scope', 'label_keys': MISAR_SEQ_LABEL_KEYS, 'label_asw_sample_size': 0, 'label_asw_rule': 'full_nonmissing_labels_in_each_scope', 'label_valid_counts': {label: int(comparison_valid_truth(data.truth[label]).sum()) for label in MISAR_SEQ_LABEL_KEYS}, 'spatial_neighbor_k': MISAR_SEQ_SPATIAL_NEIGHBOR_K, 'spatial_graph_scope': 'exact_knn_within_each_section', 'spatial_graph_n_obs': MISAR_SEQ_SECTION_COUNTS, 'joint_and_independent_labels_cover_all_spots': True, 'external_internal_spatial_and_figures_reuse_saved_labels': True, 'source_data_dir': str(data.data_dir), 'source_files': [{'path': str(path), 'sha256': mousebrain_sha256(path)} for path in data.source_paths], 'generation_script': str(MISAR_SEQ_SCRIPT_PATH), 'generation_script_sha256': mousebrain_sha256(MISAR_SEQ_SCRIPT_PATH), 'created_at': datetime.now().astimezone().isoformat()}
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    (output / 'SUMMARY.md').write_text(f'# {data.name}: {scheme}\n\n- spots: 7118; dimensions: {data.embedding.shape[1]}\n- exact KMeans: seed={MISAR_SEQ_SEED}, n_init={MISAR_SEQ_N_INIT}, max_iter={MISAR_SEQ_MAX_ITER}\n- joint/independent metrics and labels: K={MISAR_SEQ_METRIC_KS}\n- spatial figures: K={MISAR_SEQ_PLOT_KS}; point size={MISAR_SEQ_PLOT_POINT_SIZE:g}\n- Cluster ASW, Label ASW, CH and DBI: full valid samples\n- spatial continuity: exact {MISAR_SEQ_SPATIAL_NEIGHBOR_K}-NN per section\n- every metric and figure reuses the saved full label CSV files\n')

def misar_seq_copy_shared(data: misar_seq_MethodData) -> None:
    output = data.output_root / 'shared_metrics'
    output.mkdir()
    rows = []
    for (name, source) in data.shared_sources.items():
        target = output / name
        shutil.copy2(source, target)
        rows.append({'artifact': name, 'source': str(source), 'source_sha256': mousebrain_sha256(source), 'copied_sha256': mousebrain_sha256(target), 'preprocessing_dependent': False})
    pd.DataFrame(rows).to_csv(output / 'source_manifest.csv', index=False)

def human_lymph_node_copy_shared(data: human_lymph_node_MethodData) -> None:
    out = data.output_root / 'shared_metrics'
    out.mkdir()
    rows = []
    for (name, source) in data.shared_sources.items():
        target = out / name
        shutil.copy2(source, target)
        rows.append({'artifact': name, 'source': str(source), 'source_sha256': human_lymph_node_sha256(source), 'copied_sha256': human_lymph_node_sha256(target), 'preprocessing_dependent': False})
    pd.DataFrame(rows).to_csv(out / 'source_manifest.csv', index=False)


def comparison_protocol_identity(dataset):
    """Describe the retained protocol; file relocation is not a protocol change."""
    from analysis.protocols import HLN, CRC
    if dataset == 'mousebrain':
        parameters = dict(MOUSEBRAIN, independent_ks=MOUSEBRAIN_PLOT_KS,
                          plot_ks=MOUSEBRAIN_PLOT_KS, spatial_k=MOUSEBRAIN_SPATIAL_NEIGHBOR_K)
    elif dataset == 'misar_seq':
        parameters = dict(MISAR, plot_ks=MISAR_SEQ_PLOT_KS, spatial_k=MISAR_SEQ_SPATIAL_NEIGHBOR_K)
    elif dataset == 'human_lymph_node':
        parameters = dict(HLN)
    elif dataset == 'mouse_spleen':
        parameters = dict(SPLEEN, plot_ks=MOUSE_SPLEEN_PLOT_KS, spatial_k=MOUSE_SPLEEN_SPATIAL_NEIGHBOR_K)
    elif dataset == 'mouse_thymus':
        parameters = dict(THYMUS, plot_ks=MOUSE_THYMUS_PLOT_KS, spatial_k=MOUSE_THYMUS_SPATIAL_NEIGHBOR_K)
    elif dataset == 'simulation':
        parameters = dict(SIMULATION, spatial_k=SIMULATION_SPATIAL_NEIGHBOR_K)
    elif dataset == 'crc_stereocite':
        parameters = dict(CRC)
    else:
        raise ValueError(f'Unsupported comparison protocol: {dataset}')
    return {'dataset': dataset, 'parameters': parameters,
            'preprocessing': 'standardized_embedding',
            'scaler_scopes': {'joint': 'all spots', 'independent': 'per section'},
            'truth_filter': 'comparison_valid_truth',
            'metrics': 'retained dataset-specific internal/external/spatial tables',
            'batch_metrics': 'copy existing source with digest; no recomputation',
            'aggregation': 'none; per-method rows retained'}


def compare_embeddings(dataset, method_inputs, output_root):
    """Run only the retained standardized branch for explicitly supplied methods.

    Input dictionaries select an existing reader and its source paths. No model,
    preprocessing cache, baseline training, or raw-vs-standardized experiment is run.
    """
    from data_io import comparison_inputs as inputs
    from data_io import paired_comparison_inputs as paired_inputs
    from analysis import comparison_readers as family_inputs
    from analysis import comparisons, spatch
    from analysis.cache import (check_output_path, data_identity, implementation_identity, file_identity)
    from analysis.evaluation_workflow import evaluation_session

    output_root = check_output_path(output_root)
    keys = [item['method'] for item in method_inputs]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Methods must be a nonempty distinct list')
    if dataset == 'spatch':
        return compare_spatch_inputs(method_inputs, output_root)
    # Explicit dataset branches; each retains its own reader and scientific runtime.
    if dataset == 'mousebrain':
        readers = (family_inputs.mousebrain_load_spa, family_inputs.mousebrain_load_cosie, family_inputs.mousebrain_load_mofa, family_inputs.mousebrain_load_spamosaic)
        validate, run, shared = family_inputs.mousebrain_validate, mousebrain_run_scheme, mousebrain_copy_shared
    elif dataset == 'misar_seq':
        readers = (family_inputs.misar_seq_load_spa, family_inputs.misar_seq_load_cosie, family_inputs.misar_seq_load_mofa, family_inputs.misar_seq_load_spamosaic)
        validate, run, shared = family_inputs.misar_seq_validate, misar_seq_run_scheme, misar_seq_copy_shared
    elif dataset == 'human_lymph_node':
        readers = (paired_inputs.human_lymph_node_load_spa, paired_inputs.human_lymph_node_load_cosie, paired_inputs.human_lymph_node_load_mofa, paired_inputs.human_lymph_node_load_spamosaic)
        validate, run, shared = inputs.human_lymph_node_validate, comparisons.hln_run_scheme, human_lymph_node_copy_shared
    elif dataset == 'mouse_spleen':
        readers = (paired_inputs.mouse_spleen_load_spa, paired_inputs.mouse_spleen_load_cosie, paired_inputs.mouse_spleen_load_mofa, paired_inputs.mouse_spleen_load_spamosaic)
        validate, run, shared = inputs.mouse_spleen_validate, mouse_spleen_run_scheme, mouse_spleen_copy_shared
    elif dataset == 'mouse_thymus':
        readers = (family_inputs.mouse_thymus_load_spa, family_inputs.mouse_thymus_load_cosie, family_inputs.mouse_thymus_load_mofa, family_inputs.mouse_thymus_load_spamosaic)
        validate, run, shared = family_inputs.mouse_thymus_validate, mouse_thymus_run_scheme, mouse_thymus_prepare_shared
    elif dataset == 'simulation':
        readers = (family_inputs.simulation_load_spa, family_inputs.simulation_load_cosie, family_inputs.simulation_load_mofa, family_inputs.simulation_load_spamosaic)
        validate, run, shared = family_inputs.simulation_validate_method, simulation_run_scheme, simulation_copy_shared_metrics
    elif dataset == 'crc_stereocite':
        readers = (paired_inputs.crc_stereocite_load_spa, paired_inputs.crc_stereocite_load_cosie, paired_inputs.crc_stereocite_load_mofa, paired_inputs.crc_stereocite_load_spamosaic)
        validate, run, shared = inputs.crc_stereocite_validate, comparisons.crc_run_scheme, comparisons.crc_prepare_shared
    else:
        raise ValueError(f'Unsupported comparison dataset: {dataset}')
    methods = []
    for item in method_inputs:
        if item['method'] not in ('spa', 'cosie', 'mofa', 'spamosaic'):
            raise ValueError(f"Unsupported method: {item['method']}")
        paths = {key: Path(value) for key, value in item['paths'].items()}
        target = check_output_path(output_root / item['method'], paths.values())
        data = readers[('spa', 'cosie', 'mofa', 'spamosaic').index(item['method'])](paths, target)
        # Strict source/count/truth checks from the original method-specific reader.
        validate(data, allow_existing_output=data.output_root.exists())
        methods.append(data)
    identities = []
    for data in methods:
        truth = data.truth if hasattr(data, 'truth') else {}
        if dataset == 'simulation': truth = {'spatial_domain': truth}
        identities.append({'data': data_identity(data, truth),
            'shared_inputs': {name: file_identity(path) for name, path in data.shared_sources.items()}})
    identity = {'dataset': dataset, 'methods': keys, 'sources': identities,
        'protocol': comparison_protocol_identity(dataset),
        'implementation': implementation_identity('standardized-method-comparison-v1')}
    source_dirs = [data.data_dir for data in methods] + [path for data in methods for path in data.source_paths]
    with evaluation_session(output_root, identity, input_dirs=source_dirs) as reused:
        if reused:
            return json.loads((output_root / 'comparison_summary.json').read_text())
        if dataset == 'mouse_thymus': sample = mouse_thymus_build_asw_sample(methods)
        if dataset == 'crc_stereocite':
            inputs.crc_stereocite_validate_cross_method_alignment(methods)
            sample = comparisons.crc_select_by_barcode_hash(methods[0], comparisons.CRC_ASW_SECTION_COUNTS, 'asw_seed0|')
            plot_sample = comparisons.crc_select_by_barcode_hash(methods[0], comparisons.CRC_PLOT_SECTION_COUNTS, '')
        for data in methods:
            if dataset == 'crc_stereocite':
                graphs = shared(data, sample, plot_sample)
                run(data, 'standardized_embedding', sample, plot_sample, graphs)
            elif dataset == 'mouse_thymus':
                shared(data, sample)
                run(data, 'standardized_embedding', sample)
            else:
                run(data, 'standardized_embedding')
                shared(data)
                if dataset == 'mousebrain': mousebrain_add_group_diagnostics(data)
            retain_comparison_artifacts(data, dataset)
        result = collect_method_tables(dataset, keys, [data.output_root for data in methods], output_root)
        return result


def compare_spatch_inputs(method_inputs, output_root):
    from analysis.spatch_readers import spatch_MethodSpec, spatch_load_metadata, spatch_load_embedding, spatch_validate_metadata
    from analysis import spatch
    from analysis.cache import begin_analysis, finish_analysis, check_output_path
    specs, metadata, sources = [], [], []
    for item in method_inputs:
        if item['method'] not in ('spa', 'cosie'):
            raise ValueError('SPATCH comparison retains the spa and COSIE full-spot methods')
        paths = item['paths']
        spec = spatch_MethodSpec(key=item['method'], name='spa_mo_model' if item['method']=='spa' else 'COSIE',
            run_dir=Path(paths['run_dir']), data_dir=Path(paths['data_dir']),
            embedding_paths={s: Path(p) for s,p in paths['embedding_paths'].items()},
            metadata_path=Path(paths['metadata_path']), id_column=paths['id_column'],
            output_root=output_root/item['method'], old_analysis=Path(paths['analysis_dir']),
            batch_metrics_source=Path(paths['batch_metrics_source']))
        check_output_path(output_root, [spec.run_dir, spec.data_dir, spec.old_analysis])
        meta = spatch_load_metadata(spec); spatch_validate_metadata(spec, meta)
        specs.append(spec); metadata.append(meta); sources.append(spatch.source_identity(spec, meta))
    for meta in metadata[1:]:
        for column in ['section', 'spot_id', 'original_rna_obs_name', *spatch.LABELS]:
            left = metadata[0][column].astype('string').fillna('<NA>').to_numpy()
            right = meta[column].astype('string').fillna('<NA>').to_numpy()
            if not np.array_equal(left,right): raise ValueError(f'cross-method metadata mismatch: {column}')
        if not np.allclose(metadata[0][['x','y']].to_numpy(),meta[['x','y']].to_numpy()):
            raise ValueError('cross-method coordinate mismatch')
    identity={'dataset':'spatch','sources':sources,'preprocessing':'standardized_embedding','implementation':'standardized-spatch-comparison-v1'}
    if begin_analysis(output_root,identity,input_dirs=[s.run_dir for s in specs]):
        return json.loads((output_root/'comparison_summary.json').read_text())
    reference_root=None
    for spec,meta,source in zip(specs,metadata,sources):
        begin_analysis(spec.output_root,source,input_dirs=[spec.run_dir])
        graphs=spatch.prepare_shared(spec,meta,reference_root)
        if reference_root is None: reference_root=spec.output_root
        embedding=spatch_load_embedding(spec)
        if len(embedding)!=spatch.N_OBS or not np.isfinite(embedding).all():raise ValueError(f'{spec.name}: invalid embedding')
        spatch.run_scheme(spec,meta,embedding,graphs,'standardized_embedding',source=source)
        finish_analysis(spec.output_root,source)
    result=collect_method_tables('spatch',[s.key for s in specs],[s.output_root for s in specs],output_root)
    finish_analysis(output_root,identity)
    return result


def collect_method_tables(dataset, keys, method_roots, output_root):
    """Combine retained per-method tables without recomputing or averaging metrics."""
    tables={}
    for key,root in zip(keys,method_roots):
        for path in (root/'standardized_embedding/metrics').glob('*.csv'):
            frame=pd.read_csv(path);frame.insert(0,'comparison_method',key)
            tables.setdefault(path.name,[]).append(frame)
    destination=output_root/'tables';destination.mkdir()
    for name,frames in tables.items():pd.concat(frames,ignore_index=True).to_csv(destination/name,index=False)
    result={'status':'PASS','dataset':dataset,'methods':keys,'preprocessing':'standardized_embedding',
        'method_outputs':{key:str(root/'standardized_embedding') for key,root in zip(keys,method_roots)},
        'metric_tables':{name:str(destination/name) for name in tables},
        'aggregation':'none; per-method rows retained','raw_comparison':False}
    (output_root/'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


# Preserve the former default retained-K output layout in newly owned outputs.
from analysis.comparisons import HLN_METRIC_KS, HLN_PLOT_KS, HLN_PRUNED_CLUSTERING_KS, hln_write_scheme_summary
MISAR_SEQ_PRUNED_CLUSTERING_KS = [k for k in MISAR_SEQ_METRIC_KS if k not in MISAR_SEQ_PLOT_KS]


def _mousebrain_retain_clustering(data: mousebrain_MethodData) -> int:
    deleted = 0
    expected_remaining = {f'{mode}_k{k}' for mode in ('joint', 'independent') for k in MOUSEBRAIN_PLOT_KS}
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        cluster_root = output / 'clustering'
        if not cluster_root.is_dir():
            raise FileNotFoundError(f'{data.name} {scheme}: missing {cluster_root}')
        for directory in sorted(cluster_root.iterdir()):
            if not directory.is_dir():
                continue
            parts = directory.name.rsplit('_k', 1)
            if len(parts) != 2 or parts[0] not in {'joint', 'independent'} or (not parts[1].isdigit()):
                continue
            k = int(parts[1])
            if k not in MOUSEBRAIN_METRIC_KS:
                raise ValueError(f'{data.name} {scheme}: unexpected clustering K={k}')
            if k in MOUSEBRAIN_PLOT_KS:
                continue
            if directory.resolve().parent != cluster_root.resolve():
                raise ValueError(f'unsafe deletion target {directory}')
            print(f'[{data.name}] delete {directory}', flush=True)
            shutil.rmtree(directory)
            deleted += 1
        remaining = {directory.name for directory in cluster_root.iterdir() if directory.is_dir() and directory.name.rsplit('_k', 1)[0] in {'joint', 'independent'} and directory.name.rsplit('_k', 1)[1].isdigit()}
        if remaining != expected_remaining:
            raise ValueError(f'{data.name} {scheme}: unexpected remaining directories {sorted(remaining)}')
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'retained_clustering_k_values': MOUSEBRAIN_PLOT_KS, 'deleted_clustering_k_values': MOUSEBRAIN_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': MOUSEBRAIN_PRUNED_CLUSTERING_KS, 'plots_and_metrics_reuse_same_label_files': False, 'retained_k_plots_and_metrics_reuse_same_label_files': True, 'clustering_directories_pruned_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        mousebrain_write_scheme_summary(data, scheme)
    return deleted

def _human_lymph_node_retain_clustering(data: human_lymph_node_MethodData) -> int:
    deleted = 0
    expected_remaining = {f'{mode}_k{k}' for mode in ('joint', 'independent') for k in HLN_PLOT_KS}
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        cluster_root = output / 'clustering'
        if not cluster_root.is_dir():
            raise FileNotFoundError(f'{data.name} {scheme}: missing {cluster_root}')
        for directory in sorted(cluster_root.iterdir()):
            if not directory.is_dir():
                continue
            parts = directory.name.rsplit('_k', 1)
            if len(parts) != 2 or parts[0] not in {'joint', 'independent'} or (not parts[1].isdigit()):
                continue
            k = int(parts[1])
            if k not in HLN_METRIC_KS:
                raise ValueError(f'{data.name} {scheme}: unexpected clustering K={k}')
            if k in HLN_PLOT_KS:
                continue
            if directory.resolve().parent != cluster_root.resolve():
                raise ValueError(f'unsafe deletion target {directory}')
            print(f'[{data.name}] delete {directory}', flush=True)
            shutil.rmtree(directory)
            deleted += 1
        remaining = {directory.name for directory in cluster_root.iterdir() if directory.is_dir() and directory.name.rsplit('_k', 1)[0] in {'joint', 'independent'} and directory.name.rsplit('_k', 1)[1].isdigit()}
        if remaining != expected_remaining:
            raise ValueError(f'{data.name} {scheme}: unexpected remaining directories {sorted(remaining)}')
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'retained_clustering_k_values': HLN_PLOT_KS, 'deleted_clustering_k_values': HLN_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': HLN_PRUNED_CLUSTERING_KS, 'plots_and_metrics_reuse_same_label_files': False, 'retained_k_plots_and_metrics_reuse_same_label_files': True, 'clustering_directories_pruned_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        hln_write_scheme_summary(data, scheme)
    return deleted

def _mouse_spleen_retain_clustering(data: mouse_spleen_MethodData) -> int:
    """Delete only non-retained joint/independent clustering directories."""
    deleted = 0
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        clustering_root = (output / 'clustering').resolve()
        targets: list[Path] = []
        for directory in clustering_root.iterdir():
            if not directory.is_dir() or '_k' not in directory.name:
                continue
            (mode, k_text) = directory.name.rsplit('_k', 1)
            if mode not in {'joint', 'independent'} or not k_text.isdigit() or int(k_text) in MOUSE_SPLEEN_PLOT_KS:
                continue
            if int(k_text) not in MOUSE_SPLEEN_PRUNED_CLUSTERING_KS:
                raise ValueError(f'unexpected clustering K target: {directory}')
            if directory.resolve().parent != clustering_root:
                raise ValueError(f'unsafe clustering target resolution: {directory}')
            targets.append(directory)
        for directory in sorted(targets):
            print(f'[{data.name}] delete {directory}', flush=True)
            shutil.rmtree(directory)
            deleted += 1
        remaining = {int(directory.name.rsplit('_k', 1)[1]) for directory in clustering_root.iterdir() if directory.is_dir() and directory.name.rsplit('_k', 1)[0] in {'joint', 'independent'} and directory.name.rsplit('_k', 1)[1].isdigit()}
        if remaining != set(MOUSE_SPLEEN_PLOT_KS):
            raise ValueError(f'{data.name} {scheme}: unexpected remaining K {remaining}')
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'retained_clustering_k_values': MOUSE_SPLEEN_PLOT_KS, 'deleted_clustering_k_values': MOUSE_SPLEEN_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': MOUSE_SPLEEN_PRUNED_CLUSTERING_KS, 'internal_spatial_and_figures_reuse_saved_labels': False, 'retained_k_metrics_and_figures_reuse_saved_labels': True, 'clustering_directories_pruned_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        summary_path = output / 'SUMMARY.md'
        summary_lines = summary_path.read_text().splitlines()
        updated: list[str] = []
        for line in summary_lines:
            if line.startswith('- joint/independent metrics and full labels:'):
                updated.append(f'- joint/independent metrics: K={MOUSE_SPLEEN_METRIC_KS}')
                updated.append(f'- retained clustering directories and full labels: K={MOUSE_SPLEEN_PLOT_KS}')
                updated.append(f'- metrics for K={MOUSE_SPLEEN_PRUNED_CLUSTERING_KS} remain in CSV, but their historical labels_path targets were removed')
            elif line == '- every internal/spatial metric and figure reuses saved label CSVs':
                updated.append('- retained-K metrics and every figure reuse saved full label CSVs')
            else:
                updated.append(line)
        summary_path.write_text('\n'.join(updated) + '\n')
    return deleted

def _mouse_thymus_retain_clustering(data: mouse_thymus_MethodData) -> int:
    """Delete only non-retained joint/independent clustering directories."""
    deleted = 0
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        clustering_root = (output / 'clustering').resolve()
        targets: list[Path] = []
        for directory in clustering_root.iterdir():
            if not directory.is_dir() or '_k' not in directory.name:
                continue
            (mode, k_text) = directory.name.rsplit('_k', 1)
            if mode not in {'joint', 'independent'} or not k_text.isdigit() or int(k_text) in MOUSE_THYMUS_PLOT_KS:
                continue
            if int(k_text) not in MOUSE_THYMUS_PRUNED_CLUSTERING_KS:
                raise ValueError(f'unexpected clustering K target: {directory}')
            if directory.resolve().parent != clustering_root:
                raise ValueError(f'unsafe clustering target resolution: {directory}')
            targets.append(directory)
        for directory in sorted(targets):
            print(f'[{data.name}] delete {directory}', flush=True)
            shutil.rmtree(directory)
            deleted += 1
        remaining = {int(directory.name.rsplit('_k', 1)[1]) for directory in clustering_root.iterdir() if directory.is_dir() and directory.name.rsplit('_k', 1)[0] in {'joint', 'independent'} and directory.name.rsplit('_k', 1)[1].isdigit()}
        if remaining != set(MOUSE_THYMUS_PLOT_KS):
            raise ValueError(f'{data.name} {scheme}: unexpected remaining K {remaining}')
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'retained_clustering_k_values': MOUSE_THYMUS_PLOT_KS, 'deleted_clustering_k_values': MOUSE_THYMUS_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': MOUSE_THYMUS_PRUNED_CLUSTERING_KS, 'internal_spatial_and_figures_reuse_saved_labels': False, 'retained_k_metrics_and_figures_reuse_saved_labels': True, 'clustering_directories_pruned_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        summary_path = output / 'SUMMARY.md'
        summary_lines = summary_path.read_text().splitlines()
        updated: list[str] = []
        for line in summary_lines:
            if line.startswith('- joint/independent metrics and full labels:'):
                updated.append(f'- joint/independent metrics: K={MOUSE_THYMUS_METRIC_KS}')
                updated.append(f'- retained clustering directories and full labels: K={MOUSE_THYMUS_PLOT_KS}')
                updated.append(f'- metrics for K={MOUSE_THYMUS_PRUNED_CLUSTERING_KS} remain in CSV, but their historical labels_path targets were removed')
            elif line == '- every internal/spatial metric and figure reuses saved label CSVs':
                updated.append('- retained-K metrics and every figure reuse saved full label CSVs')
            else:
                updated.append(line)
        summary_path.write_text('\n'.join(updated) + '\n')
    return deleted

def _misar_seq_retain_clustering(data: misar_seq_MethodData) -> int:
    """Delete only non-retained joint/independent clustering directories."""
    deleted = 0
    for scheme in ('standardized_embedding',):
        output = data.output_root / scheme
        clustering_root = (output / 'clustering').resolve()
        targets: list[Path] = []
        for directory in clustering_root.iterdir():
            if not directory.is_dir() or '_k' not in directory.name:
                continue
            (mode, k_text) = directory.name.rsplit('_k', 1)
            if mode not in {'joint', 'independent'} or not k_text.isdigit() or int(k_text) in MISAR_SEQ_PLOT_KS:
                continue
            if int(k_text) not in MISAR_SEQ_PRUNED_CLUSTERING_KS:
                raise ValueError(f'unexpected clustering K target: {directory}')
            if directory.resolve().parent != clustering_root:
                raise ValueError(f'unsafe clustering target resolution: {directory}')
            targets.append(directory)
        for directory in sorted(targets):
            print(f'[{data.name}] delete {directory}', flush=True)
            shutil.rmtree(directory)
            deleted += 1
        remaining = {int(directory.name.rsplit('_k', 1)[1]) for directory in clustering_root.iterdir() if directory.is_dir() and directory.name.rsplit('_k', 1)[0] in {'joint', 'independent'} and directory.name.rsplit('_k', 1)[1].isdigit()}
        if remaining != set(MISAR_SEQ_PLOT_KS):
            raise ValueError(f'{data.name} {scheme}: unexpected remaining K {remaining}')
        config_path = output / 'config.json'
        config = json.loads(config_path.read_text())
        config.update({'retained_clustering_k_values': MISAR_SEQ_PLOT_KS, 'deleted_clustering_k_values': MISAR_SEQ_PRUNED_CLUSTERING_KS, 'metrics_k_values_without_saved_label_directories': MISAR_SEQ_PRUNED_CLUSTERING_KS, 'external_internal_spatial_and_figures_reuse_saved_labels': False, 'retained_k_metrics_and_figures_reuse_saved_labels': True, 'clustering_directories_pruned_at': datetime.now().astimezone().isoformat()})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
        summary_path = output / 'SUMMARY.md'
        summary_lines = summary_path.read_text().splitlines()
        updated: list[str] = []
        for line in summary_lines:
            if line.startswith('- joint/independent metrics and labels:'):
                updated.append(f'- joint/independent metrics: K={MISAR_SEQ_METRIC_KS}')
                updated.append(f'- retained clustering directories and full labels: K={MISAR_SEQ_PLOT_KS}')
                updated.append(f'- metrics for K={MISAR_SEQ_PRUNED_CLUSTERING_KS} remain in CSV, but their historical labels_path targets were removed')
            elif line == '- every metric and figure reuses the saved full label CSV files':
                updated.append('- retained-K metrics and every figure reuse the saved full label CSV files')
            else:
                updated.append(line)
        summary_path.write_text('\n'.join(updated) + '\n')
    return deleted

def retain_comparison_artifacts(data, dataset):
    """Apply the original output retention only during a new comparison run."""
    if dataset == 'mousebrain': return _mousebrain_retain_clustering(data)
    if dataset == 'human_lymph_node': return _human_lymph_node_retain_clustering(data)
    if dataset == 'mouse_spleen': return _mouse_spleen_retain_clustering(data)
    if dataset == 'mouse_thymus': return _mouse_thymus_retain_clustering(data)
    if dataset == 'misar_seq': return _misar_seq_retain_clustering(data)
    return 0
