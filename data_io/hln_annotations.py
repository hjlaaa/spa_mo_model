"""Read external HLN method embeddings for the manual A1 annotation workflow."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import anndata as ad
import numpy as np
import pandas as pd
SECTIONS = ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]
@dataclass(frozen=True)
class LoadedData:
    embedding: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    source_paths: tuple[Path, ...]

def load_spa_mo_model(paths) -> LoadedData:
    run = Path(paths['run_dir'])
    data_dir = Path(paths['data_dir'])
    embeddings: list[np.ndarray] = []
    sections: list[str] = []
    barcodes: list[str] = []
    sources: list[Path] = []
    for section in SECTIONS:
        embedding_path = run / f'final_embeddings_{section}.npy'
        indices_path = run / f'selected_spot_indices_{section}.npy'
        rna_path = data_dir / section / 'adata_RNA.h5ad'
        embedding = np.load(embedding_path)
        indices = np.load(indices_path).astype(int)
        rna = ad.read_h5ad(rna_path, backed='r')
        try:
            names = rna.obs_names.astype(str).to_numpy()[indices]
        finally:
            rna.file.close()
        if embedding.shape[0] != len(indices):
            raise ValueError(f'{section}: embedding/index row mismatch')
        embeddings.append(np.asarray(embedding))
        sections.extend([section] * len(indices))
        barcodes.extend(names.tolist())
        sources.extend([embedding_path, indices_path, rna_path])
    return LoadedData(np.vstack(embeddings), np.asarray(sections, dtype=str), np.asarray(barcodes, dtype=str), tuple(sources))

def load_cosie(paths) -> LoadedData:
    path = Path(paths['embedding_file'])
    table = pd.read_csv(path)
    embedding_columns = [column for column in table if column.startswith('COSIE')]
    barcode_column = 'obs_name.1' if 'obs_name.1' in table else 'obs_name'
    return LoadedData(table[embedding_columns].to_numpy(float), table['section'].astype(str).to_numpy(), table[barcode_column].astype(str).to_numpy(), (path,))

def load_present(paths) -> LoadedData:
    run = Path(paths['run_dir'])
    embedding_path = run / 'training/embeddings.npy'
    adata_path = run / 'training/adata_output.h5ad'
    embedding = np.load(embedding_path)
    output = ad.read_h5ad(adata_path, backed='r')
    try:
        sections = output.obs['section'].astype(str).to_numpy()
        barcodes = output.obs['original_barcode'].astype(str).to_numpy()
    finally:
        output.file.close()
    return LoadedData(np.asarray(embedding), sections, barcodes, (embedding_path, adata_path))

def load_spamosaic(paths) -> LoadedData:
    path = Path(paths['embedding_file'])
    output = ad.read_h5ad(path)
    return LoadedData(np.asarray(output.obsm['merged_emb']).copy(), output.obs['section'].astype(str).to_numpy(), output.obs['original_barcode'].astype(str).to_numpy(), (path,))

def load_mofa(paths) -> LoadedData:
    path = Path(paths['embedding_file'])
    table = pd.read_csv(path)
    factor_columns = [column for column in table if column.startswith('Factor')]
    return LoadedData(table[factor_columns].to_numpy(float), table['section'].astype(str).to_numpy(), table['original_barcode'].astype(str).to_numpy(), (path,))
