"""SPATCH comparison cohort policy over neutral result readers."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.protocols import SPATCH
from data_io.large_results import load_spatch_embedding, load_spatch_metadata

@dataclass
class spatch_MethodSpec:
    key: str
    name: str
    run_dir: Path
    data_dir: Path
    embedding_paths: dict[str, Path]
    metadata_path: Path
    id_column: str
    output_root: Path
    old_analysis: Path
    batch_metrics_source: Path

SPATCH_LABELS = SPATCH['LABELS']

def spatch_load_metadata(spec: spatch_MethodSpec) -> pd.DataFrame:
    return load_spatch_metadata(spec.metadata_path, spec.id_column, SPATCH_LABELS)

SPATCH_SECTIONS = ['section1', 'section2']

SPATCH_SECTION_COUNTS = {'section1': 665399, 'section2': 403563}

def spatch_load_embedding(spec: spatch_MethodSpec) -> np.ndarray:
    return load_spatch_embedding(spec.embedding_paths, section_order=SPATCH_SECTIONS, section_counts=SPATCH_SECTION_COUNTS, name=spec.name)

SPATCH_N_OBS = 1068962

def spatch_validate_metadata(spec: spatch_MethodSpec, metadata: pd.DataFrame) -> None:
    if len(metadata) != SPATCH_N_OBS:
        raise ValueError(f'{spec.name}: expected {SPATCH_N_OBS}, got {len(metadata)}')
    counts = metadata['section'].value_counts().to_dict()
    if counts != SPATCH_SECTION_COUNTS:
        raise ValueError(f'{spec.name}: section counts differ: {counts}')
    if metadata[['section', 'spot_id']].duplicated().any():
        raise ValueError(f'{spec.name}: duplicate section/spot_id')
    if not np.isfinite(metadata[['x', 'y']].to_numpy()).all():
        raise ValueError(f'{spec.name}: non-finite coordinates')
    for path in [*spec.embedding_paths.values(), spec.metadata_path, spec.batch_metrics_source]:
        if not path.exists():
            raise FileNotFoundError(path)


spatch_MethodSpec.__module__ = 'data_io.comparison_inputs'
