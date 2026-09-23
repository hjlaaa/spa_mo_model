"""Dataset protocols: preserve requested, comparison and historical CLI differences."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    display_name: str
    joint_ks: tuple[int, ...]
    independent_ks: tuple[int, ...]
    seed: int
    n_init: int = 20
    max_iter: int = 300
    batch_max_samples: int = 0
    batch_asw_sample_size: int = 10000
    batch_seed: int = 0


SPECS = {
    "mousebrain": AnalysisSpec(
        "mousebrain", "MouseBrain", tuple(range(2, 13)),
        (5, 6, 8, 9, 10, 11, 12), 0, batch_max_samples=50000,
    ),
    "misar_seq": AnalysisSpec(
        "misar_seq", "MISAR-seq", tuple(range(2, 17)), tuple(range(2, 17)), 0,
    ),
    "mouse_spleen": AnalysisSpec(
        "mouse_spleen", "Mouse Spleen", tuple(range(2, 13)), tuple(range(2, 13)),
        0, batch_max_samples=50000,
    ),
    "human_lymph_node": AnalysisSpec(
        "human_lymph_node", "Human Lymph Node", (8, 10, 12), (8, 10, 12),
        0, batch_max_samples=50000,
    ),
    "simulation": AnalysisSpec(
        "simulation", "Simulation", (5, 8, 10, 12), (5, 8, 10, 12),
        42, batch_seed=42,
    ),
    "mouse_thymus": AnalysisSpec(
        "mouse_thymus", "Mouse Thymus", tuple(range(2, 13)), tuple(range(2, 13)),
        0,
    ),
}

MOUSEBRAIN = {
    'LABEL_KEYS': ["RegionLoupe", "annotations", "celltype", "Y.l1", "Y", "group"],
    'METRIC_KS': list(range(2, 13)),
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

MISAR = {
    'LABEL_KEYS': [
    "Y",
    "Combined_Clusters_annotation",
    "Combined_Clusters",
    "RNA_Clusters",
    "ATAC_Clusters",
],
    'METRIC_KS': list(range(2, 17)),
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

HLN = {
    'METRIC_KS': list(range(2, 13)),
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

SPLEEN = {
    'METRIC_KS': list(range(2, 13)),
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

THYMUS = {
    'ASW_SECTION_COUNTS': {
    "Mouse_Thymus1": 2635,
    "Mouse_Thymus2": 2386,
    "Mouse_Thymus3": 2607,
    "Mouse_Thymus4": 2372,
},
    'METRIC_KS': list(range(2, 13)),
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

SIMULATION = {
    'KS': [5, 8, 10, 12],
    'SEED': 42,
    'N_INIT': 20,
    'MAX_ITER': 300,
}

CRC = {
    'ASW_SECTION_COUNTS': {
    "CRC_003_bin20": 2716,
    "CRC_006_bin20": 7284,
},
    'KS': [5, 10, 15, 20, 25],
    'SEED': 0,
    'N_INIT': 20,
    'MAX_ITER': 300,
    'BATCH_SIZE': 8192,
}

SPATCH = {
    'LABELS': ["cell_type_common", "spatial_cluster", "codex_coarse_label"],
    'ALL_KS': list(range(2, 21)),
    'RETAIN_KS': [5, 8, 10, 12, 16, 20],
    'SEED': 42,
    'N_INIT': 20,
    'MAX_ITER': 300,
    'BATCH_SIZE': 4096,
    'ASW_SAMPLE_SIZE': 10000,
}

SPATCH_REQUESTED = {
    'TRUTH_LABELS': tuple(SPATCH["LABELS"]),
    'K_VALUES': (5, 8, 10, 12, 14, 16, 20),
}

EVALUATION_MODE = "joint_labels_evaluated_per_section_without_reclustering"
MISSING_STRINGS = {"", "nan", "na", "none", "null", "<na>"}

# The historical suite supplies EXPERIMENT_K explicitly. The standalone CLI
# defaults are a different, still supported command; neither overwrites the other.
EMBRYO = {
    "CLI_K": [5, 8, 10, 12, 14, 16],
    "EXPERIMENT_K": [2, 3, 4, 10, 15, 20, 25],
    "seed": 42, "n_init": 3, "max_iter": 100, "batch_size": 8192,
    "metric_sample_size": 5000, "batch_metric_sample_size": 100001,
    "batch_asw_sample_size": 10000,
}

SCOPES = {
    "joint": "fit one joint scaler and clustering; evaluate all spots",
    "joint_per_section": "slice existing joint assignments; no scaler or clustering fit",
    "independent": "fit a scaler and clustering separately per section",
}

# This is the supplementary reference's actual reporting subset. Other valid
# labels continue to be evaluated by their existing joint/independent protocols.
JOINT_PER_SECTION_LABELS = {
    "MouseBrain": ("RegionLoupe", "annotations"),
    "MISAR-seq": tuple(MISAR["LABEL_KEYS"]),
    "SPATCH": ("cell_type_common",),
    "Simulation": ("spatial_domain",),
    "CRC": (), "Human Lymph Node": (), "Mouse Spleen": (), "Mouse Thymus": (),
    "Human Embryo": (),
}


def joint_per_section_protocol(dataset: str, k_values, section_order) -> dict:
    """An explicit cache protocol, independent of source file locations."""
    labels = JOINT_PER_SECTION_LABELS[dataset]
    status = "applicable" if labels else "not_applicable"
    if dataset == "Human Embryo":
        status = "not_in_reference_protocol"
    return {
        "dataset": dataset, "scope": "joint_per_section", "status": status,
        "assignment_scope": "joint", "standardization_scope": "joint",
        "reclustering": False, "k_values": list(k_values),
        "section_order": list(section_order), "truth_labels": list(labels),
        "metrics": ["ARI", "NMI"],
        "truth_filter": {
            "conversion": "pandas nullable string; original strings retained for scoring",
            "invalid": sorted(MISSING_STRINGS), "invalid_matching": "strip then lower",
            "minimum_valid_spots": 2, "minimum_truth_classes": 2,
            "insufficient_truth": "raise ValueError",
        },
        "aggregation": "none", "implementation": "joint-per-section-reference-v1",
    }


UMAP_DEFAULTS = {"n_neighbors": 30, "min_dist": 0.3, "seed": 42, "max_samples": 100000}


def umap_parameters(n_neighbors: int, min_dist: float, seed: int) -> dict:
    return {"n_components": 2, "n_neighbors": n_neighbors, "min_dist": min_dist,
            "metric": "euclidean", "init": "spectral", "random_state": seed,
            "transform_seed": seed, "n_jobs": 1, "low_memory": True, "verbose": True}

# Established post-requested spatial retention; separate from the comparison K grids.
REQUESTED_PLOT_KS = {
    "human_lymph_node": [8, 10, 12],
    "mousebrain": [9, 10, 11], "misar_seq": [12, 14, 15],
    "mouse_spleen": [3, 5], "mouse_thymus": [5, 8, 10], "simulation": [5],
}

SPATCH_UMAP_INPUT_FILES = {
    "section1": {
        "RNA": "adata_xenium_bin_filter.h5ad",
        "Protein": "adata_codex_bin_filter.h5ad",
        "HE": "adata_he.h5ad",
    },
    "section2": {
        "RNA": "adata_hd_filter.h5ad",
        "Protein": "adata_codex_filter.h5ad",
        "HE": "adata_he.h5ad",
    },
}

# Existing requested metric fields; list/tuple conversions at consumers preserve their APIs.
REQUESTED_INTERNAL_METRICS = ["ASW", "ASW_Scaled", "CH", "DBI"]
REQUESTED_SUPERVISED_METRICS = ["ARI", "NMI", "Homogeneity", "Completeness"]
REQUESTED_BATCH_METRICS = ["bASW", "bLISI", "kBET", "PCR_score"]
MOUSEBRAIN_UMAP_KS = (5, 6, 8, 10)
