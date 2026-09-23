"""A1 external manual annotation: preserved joint/independent reproduction and scores.

This optional workflow does not supply truth to canonical HLN evaluation.
"""
from __future__ import annotations
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score
from analysis.metrics import external_metrics
from analysis.clustering import kmeans_labels
from analysis.cache import check_output_path
from data_io.hln_annotations import LoadedData
from data_io.saved_assignments import read_assignment_table


A1 = 'Human_Lymph_Node_A1'

D1 = 'Human_Lymph_Node_D1'

SECTIONS = [A1, D1]

K_VALUES = list(range(2, 13))

RETAINED_K_VALUES = [5, 8, 10, 12]

SEED = 0

N_INIT = 20

MAX_ITER = 300

LABEL_KEY = 'manual-anno'

@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    analysis_root: Path
    annotation_path: Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def standardize(embedding: np.ndarray, scaler: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    mean = scaler[f'{prefix}_mean']
    scale = scaler[f'{prefix}_scale']
    if embedding.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError(f'{prefix}: embedding/scaler dimension mismatch')
    transformed = np.array(embedding, copy=True)
    if not np.issubdtype(transformed.dtype, np.floating):
        transformed = transformed.astype(np.float64)
    transformed -= mean
    transformed /= scale
    return transformed

def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = ['mode', 'scope', 'k', 'ari', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'label_asw']
    selected = frame[columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f'{float(value):.6f}')
            else:
                values.append(str(value).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)

def validate_inputs(spec: MethodSpec, data: LoadedData) -> dict:
    config_path = spec.analysis_root / 'config.json'
    scaler_path = spec.analysis_root / 'scaler_parameters.npz'
    if not config_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(f'{spec.display_name}: missing standardized result files')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    (n_obs, embedding_dim) = data.embedding.shape
    if n_obs != len(data.sections) or n_obs != len(data.barcodes):
        raise ValueError(f'{spec.display_name}: inconsistent input row counts')
    if n_obs != int(config['n_obs']) or embedding_dim != int(config['embedding_dim']):
        raise ValueError(f'{spec.display_name}: loaded shape {data.embedding.shape} does not match config')
    if not np.isfinite(data.embedding).all():
        raise ValueError(f'{spec.display_name}: non-finite embedding values')
    counts = pd.Series(data.sections).value_counts().to_dict()
    expected = {key: int(value) for (key, value) in config['section_counts'].items()}
    if counts != expected:
        raise ValueError(f'{spec.display_name}: section counts {counts} != {expected}')
    identities = pd.MultiIndex.from_arrays([data.sections, data.barcodes])
    if identities.has_duplicates:
        raise ValueError(f'{spec.display_name}: duplicate section/barcode identities')
    return config

def load_truth(spec: MethodSpec, a1_barcodes: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    annotation = pd.read_csv(spec.annotation_path)
    required = {'Barcode', LABEL_KEY}
    if not required.issubset(annotation.columns):
        raise ValueError(f'{spec.annotation_path}: required columns are {sorted(required)}')
    if annotation['Barcode'].astype(str).duplicated().any():
        raise ValueError(f'{spec.annotation_path}: duplicate annotation barcodes')
    annotation['Barcode'] = annotation['Barcode'].astype(str)
    annotation[LABEL_KEY] = annotation[LABEL_KEY].astype('string')
    lookup = annotation.set_index('Barcode')[LABEL_KEY]
    truth = pd.Series(a1_barcodes).map(lookup)
    if truth.isna().any() or truth.astype(str).str.strip().eq('').any():
        raise ValueError(f'{spec.display_name}: {int(truth.isna().sum())} A1 spots lack annotation')
    if truth.nunique() < 2:
        raise ValueError(f'{spec.display_name}: fewer than two annotation classes')
    return (truth.astype(str).to_numpy(), annotation)

def validate_retained_labels(spec: MethodSpec, a1_barcodes: np.ndarray, predictions: dict[str, dict[int, np.ndarray]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for mode in ('joint', 'independent'):
        for k in RETAINED_K_VALUES:
            path = spec.analysis_root / 'clustering' / f'{mode}_k{k}' / f'labels_{A1}.csv'
            if not path.is_file():
                raise FileNotFoundError(f'{spec.display_name}: missing retained labels {path}')
            saved = read_assignment_table(path)
            if not {'obs_name', 'cluster'}.issubset(saved.columns):
                raise ValueError(f'{path}: missing obs_name/cluster')
            if saved['obs_name'].astype(str).duplicated().any():
                raise ValueError(f'{path}: duplicate barcodes')
            predicted = pd.Series(predictions[mode][k], index=a1_barcodes)
            aligned = predicted.reindex(saved['obs_name'].astype(str))
            if aligned.isna().any() or len(saved) != len(a1_barcodes):
                raise ValueError(f'{path}: saved/predicted A1 identities do not match')
            reproduction_ari = float(adjusted_rand_score(saved['cluster'].to_numpy(), aligned.to_numpy()))
            if not np.isclose(reproduction_ari, 1.0, atol=1e-12):
                raise ValueError(f'{spec.display_name} {mode} k={k}: retained labels were not reproduced (ARI={reproduction_ari})')
            result[f'{mode}_k{k}'] = {'reproduction_ari': reproduction_ari, 'n_obs': int(len(saved)), 'saved_labels_path': str(path), 'saved_labels_sha256': sha256(path)}
    return result

def evaluate(spec: MethodSpec, dry_run: bool=False, *, data, output_dir) -> dict:
    output_dir = check_output_path(output_dir, [spec.analysis_root, spec.annotation_path, *data.source_paths])
    if not dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing annotation output: {output_dir}')
    original_config = validate_inputs(spec, data)
    a1_mask = data.sections == A1
    a1_barcodes = data.barcodes[a1_mask]
    (truth, annotation) = load_truth(spec, a1_barcodes)
    scaler_path = spec.analysis_root / 'scaler_parameters.npz'
    scaler = np.load(scaler_path)
    try:
        joint_space = standardize(data.embedding, scaler, 'joint')
        independent_space = standardize(data.embedding[a1_mask], scaler, A1)
    finally:
        scaler.close()
    spaces = {'joint': joint_space, 'independent': independent_space}
    predictions: dict[str, dict[int, np.ndarray]] = {'joint': {}, 'independent': {}}
    rows: list[dict] = []
    for (mode, space) in spaces.items():
        evaluation_space = joint_space[a1_mask] if mode == 'joint' else space
        label_asw = float(silhouette_score(evaluation_space, truth))
        for k in K_VALUES:
            labels = kmeans_labels(space, n_clusters=k, random_state=SEED, n_init=N_INIT, max_iter=MAX_ITER)
            a1_labels = labels[a1_mask] if mode == 'joint' else labels
            predictions[mode][k] = np.asarray(a1_labels, dtype=np.int16)
            rows.append({'mode': mode, 'scope': 'combined_A1_subset' if mode == 'joint' else A1, 'label': LABEL_KEY, 'n_obs_labeled': int(len(truth)), 'n_label_classes': int(np.unique(truth).size), 'k': k, **external_metrics(truth, a1_labels), 'label_asw': label_asw, 'label_asw_scaled': (label_asw + 1.0) / 2.0})
    validation = validate_retained_labels(spec, a1_barcodes, predictions)
    metrics = pd.DataFrame(rows)
    best = metrics.loc[metrics.groupby(['mode', 'label'])['ari'].idxmax()].sort_values(['mode', 'label']).reset_index(drop=True)
    annotation_barcodes = set(annotation['Barcode'])
    evaluated_barcodes = set(a1_barcodes)
    labels_table = pd.DataFrame({'section': A1, 'obs_name': a1_barcodes, LABEL_KEY: truth})
    for mode in ('joint', 'independent'):
        for k in K_VALUES:
            labels_table[f'{mode}_k{k}'] = predictions[mode][k]
    metrics_root = output_dir / 'metrics'
    metrics_path = metrics_root / 'clustering_metrics_by_label.csv'
    best_path = metrics_root / 'best_clustering_metrics_by_label.csv'
    labels_path = metrics_root / 'a1_supervised_cluster_labels.csv.gz'
    config_path = metrics_root / 'a1_supervised_metrics_config.json'
    readme_path = output_dir / 'A1_SUPERVISED_METRICS.md'
    outputs = [metrics_path, best_path, labels_path, config_path, readme_path]
    manifest = {'status': 'DRY_RUN' if dry_run else 'PASS', 'dataset': 'Human_Lymph_Node', 'method': spec.display_name, 'evaluation_section': A1, 'unannotated_section_excluded': D1, 'label': LABEL_KEY, 'annotation_path': str(spec.annotation_path), 'annotation_sha256': sha256(spec.annotation_path), 'annotation_rows_total': int(len(annotation)), 'annotation_rows_evaluated': int(len(truth)), 'annotation_rows_not_in_method_output': int(len(annotation_barcodes - evaluated_barcodes)), 'n_label_classes': int(np.unique(truth).size), 'label_class_counts': {str(key): int(value) for (key, value) in pd.Series(truth).value_counts().sort_index().items()}, 'modes': {'joint': 'KMeans fit on A1+D1; supervised metrics evaluated on A1 only', 'independent': 'KMeans fit and supervised metrics evaluated on A1 only'}, 'k_values': K_VALUES, 'standardization': 'reuse saved StandardScaler mean/scale parameters', 'kmeans': {'implementation': 'sklearn.cluster.KMeans', 'random_state': SEED, 'n_init': N_INIT, 'max_iter': MAX_ITER}, 'runtime': {'python_executable': sys.executable, 'python_version': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'anndata': ad.__version__}, 'metrics': ['ari', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'label_asw', 'label_asw_scaled'], 'label_asw_rule': 'full A1 labeled spots in the same standardized metric space', 'retained_label_reproduction_validation': validation, 'original_analysis_config': str(spec.analysis_root / 'config.json'), 'original_analysis_config_sha256': sha256(spec.analysis_root / 'config.json'), 'scaler_parameters': str(scaler_path), 'scaler_parameters_sha256': sha256(scaler_path), 'embedding_sources': [{'path': str(path), 'sha256': sha256(path)} for path in data.source_paths], 'original_config_label_asw': original_config.get('label_asw'), 'outputs': [str(path) for path in outputs], 'created_at': datetime.now().astimezone().isoformat(), 'script': str(Path(__file__).resolve()), 'script_sha256': sha256(Path(__file__).resolve())}
    if dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest
    metrics_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    best.to_csv(best_path, index=False)
    labels_table.to_csv(labels_path, index=False, compression='gzip')
    manifest['outputs'] = [{'path': str(path), 'sha256': sha256(path)} for path in (metrics_path, best_path, labels_path)] + [{'path': str(readme_path)}]
    config_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [f'# {spec.display_name}: Human Lymph Node A1 supervised metrics', '', f'- A1 labeled spots evaluated: {len(truth):,}', f'- manual annotation classes: {np.unique(truth).size}', '- D1 is excluded from supervised scoring because it has no annotation.', '- joint: KMeans is fit on A1+D1 and scored on the A1 subset.', '- independent: KMeans is fit and scored on A1 only.', f'- K values: {K_VALUES}', '- retained K=5/8/10/12 labels were reproduced with ARI=1 before scoring.', '', '## Best ARI by mode', '', markdown_table(best), '', '## Outputs', '', f'- `{metrics_path}`', f'- `{best_path}`', f'- `{labels_path}`', f'- `{config_path}`', '']
    readme_path.write_text('\n'.join(lines), encoding='utf-8')
    manifest['outputs'] = [{'path': str(path), 'sha256': sha256(path)} for path in (metrics_path, best_path, labels_path, readme_path)]
    config_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"{spec.display_name}: PASS; n={len(truth)}, classes={np.unique(truth).size}, joint best k={int(best.loc[best['mode'].eq('joint'), 'k'].iloc[0])}, independent best k={int(best.loc[best['mode'].eq('independent'), 'k'].iloc[0])}", flush=True)
    return manifest


def run_annotations(entries, output_root, *, dry_run=False):
    from data_io import hln_annotations as readers
    output_root=check_output_path(output_root)
    keys=[item['method'] for item in entries]
    if not keys or len(keys)!=len(set(keys)):raise ValueError('Expected distinct nonempty methods')
    results=[]
    for item in entries:
        method=item['method'];paths={key:Path(value) for key,value in item['paths'].items()}
        check_output_path(output_root,[Path(item['analysis_dir']),Path(item['annotation_path']),*paths.values()])
        if method=='spa': data=readers.load_spa_mo_model(paths);name='spa_mo_model'
        elif method=='cosie': data=readers.load_cosie(paths);name='COSIE'
        elif method=='present': data=readers.load_present(paths);name='PRESENT'
        elif method=='mofa': data=readers.load_mofa(paths);name='MOFA+'
        elif method=='spamosaic': data=readers.load_spamosaic(paths);name='SpaMosaic'
        else: raise ValueError(f'Unsupported method: {method}')
        spec=MethodSpec(method,item.get('display_name',name),Path(item['analysis_dir']),Path(item['annotation_path']))
        results.append(evaluate(spec,dry_run=dry_run,data=data,output_dir=output_root/method))
    return results
