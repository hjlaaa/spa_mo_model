"""Configuration helpers for COSIE-style preprocessing.

The paths below point at this project. ``REFERENCE_COSIE_ROOT`` is kept only
for provenance/debugging and is not used as a runtime dependency.
"""

PROJECT_ROOT = "/home/hujinlan/spa_mo_model"
MODEL_DIR = "/home/hujinlan/spa_mo_model/model"
DATA_DIR = "/home/hujinlan/spa_mo_model/data"
UNI_DIR = "/home/hujinlan/spa_mo_model/UNI"
UNI_CHECKPOINT = "/home/hujinlan/spa_mo_model/UNI/pytorch_model.bin"

# only for provenance/debug, not required at runtime
REFERENCE_COSIE_ROOT = "/home/hujinlan/cosie"

DEFAULT_N_COMPS = 50
DEFAULT_HVG_NUM = 3000
DEFAULT_TARGET_SUM = None
DEFAULT_USE_HARMONY = True
DEFAULT_SPATIAL_KEY = "spatial"
DEFAULT_UNI_FEATURE_KEY = "UNI_feature"

COSIE_MODALITIES = ("HE", "RNA", "Protein", "Metabolite")
MODALITY_ALIASES = {
    "HE": "HE",
    "H&E": "HE",
    "H_and_E": "HE",
    "RNA": "RNA",
    "RNA_panel2": "RNA_panel2",
    "Protein": "Protein",
    "protein": "Protein",
    "ADT": "Protein",
    "adt": "Protein",
    "Metabolite": "Metabolite",
    "metabolite": "Metabolite",
    "Metabolomics": "Metabolite",
    "metabolomics": "Metabolite",
}


def get_default_preprocess_config():
    """Return the default COSIE-style preprocessing configuration."""

    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "model_dir": MODEL_DIR,
            "data_dir": DATA_DIR,
            "uni_dir": UNI_DIR,
            "uni_checkpoint": UNI_CHECKPOINT,
            "reference_cosie_root": REFERENCE_COSIE_ROOT,
        },
        "modalities": {
            "canonical": list(COSIE_MODALITIES),
            "aliases": dict(MODALITY_ALIASES),
            "internal_protein_name": "Protein",
        },
        "preprocessing": {
            "n_comps": DEFAULT_N_COMPS,
            "hvg_num": DEFAULT_HVG_NUM,
            "hvg_num_by_modality": None,
            "target_sum": DEFAULT_TARGET_SUM,
            "use_harmony": DEFAULT_USE_HARMONY,
            "metacell": False,
            "spatial_key": DEFAULT_SPATIAL_KEY,
            "uni_feature_key": DEFAULT_UNI_FEATURE_KEY,
            "rna_var_names_source": None,
        },
        "he_image": {
            "superpixel_size": 16,
            "patch_size": 224,
            "batch_size": 128,
            "num_workers": 4,
            "device": None,
            "output_cache_path": None,
        },
    }


def get_default_model_config():
    """Return the first-stage multimodal model configuration.

    This stage intentionally reuses only COSIE's within-section cross-view
    contrastive loss. The encoder and fusion module are project-specific MLPs,
    not COSIE's GraphAutoencoder or Prediction_mlp.
    """

    return {
        "model": {
            "latent_dim": 128,
            "modalities_supported": ["HE", "RNA", "Protein", "Metabolite"],
            "valid_modality_sets": [
                ["HE", "RNA"],
                ["HE", "Protein"],
                ["HE", "Metabolite"],
                ["RNA", "Protein"],
                ["RNA", "Metabolite"],
                ["Protein", "Metabolite"],
                ["HE", "RNA", "Protein"],
                ["HE", "RNA", "Metabolite"],
            ],
        },
        "graph": {
            "use_spatial_graph": True,
            "knn_neighbors_spatial": 5,
            "use_feature_graph": False,
        },
        "encoder": {
            "type": "mlp",
            "hidden_dims": [256, 128],
            "output_dim": 128,
            "activation": "GELU",
            "dropout": 0.1,
            "norm": "LayerNorm",
            "residual": False,
            "l2_normalize_output": True,
        },
        "contrastive": {
            "method": "cosie_crossview",
            "gamma": 5.0,
            "loss_weight": 1.0,
            "pairwise_all_observed_modalities": True,
            "use_infonce": False,
            "use_temperature": False,
            "use_spot_positive_negative_pairs": False,
        },
        "fusion": {
            "mode": "concat_mlp_projection",
            "input_dim": 384,
            "hidden_dims": [256, 128],
            "output_dim": 128,
            "activation": "GELU",
            "dropout": 0.1,
            "norm": "LayerNorm",
            "residual": "mean_residual",
        },
        "training": {
            "epochs": 300,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "device": "cuda",
        },
        "graphsage": {
            "enabled": True,
            "input_dim": 128,
            "output_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "activation": "GELU",
            "norm": "LayerNorm",
            "residual": True,
            "use_distance_weight": True,
            "delta": 1e-8,
            "edge_batch_size": 200000,
        },
        "uot": {
            "enabled": True,
            "initial_from_modalities": True,
            "update_from_final_embedding": True,
            "epsilon_init": 0.08,
            "epsilon_update": 0.05,
            "tau_a": 1.0,
            "tau_b": 1.0,
            "max_iter": 1000,
            "tol": 1e-6,
            "check_every": 10,
            "update_interval": 20,
            "topk": 10,
            "use_momentum": False,
            "momentum": 0.0,
            "normalize_total_mass": True,
            "cost": "cosine",
            "clip_cost_min": 0.0,
            "clip_cost_max": 2.0,
            "keep_dense": False,
        },
        "ot_attention": {
            "enabled": True,
            "direction": "forward",
            "d_attn": 128,
            "beta": 0.2,
            "beta_warmup": False,
            "beta_schedule": [
                [1, 20, 0.1],
                [21, 40, 0.3],
                [41, -1, 0.5],
            ],
            "dropout": 0.1,
            "gate": "scalar",
            "use_confidence": True,
            "residual": True,
            "norm": "LayerNorm",
            "delta": 1e-8,
        },
        "decoder": {
            "enabled": True,
            "hidden_dim": 128,
            "activation": "GELU",
            "dropout": 0.1,
        },
        "reconstruction": {
            "enabled": True,
            "loss": "mse",
            "lambda_by_modality": {
                "HE": 1.0,
                "RNA": 1.0,
                "Protein": 1.0,
                "Metabolite": 1.0,
            },
        },
        "loss": {
            "lambda_contrast": 0.1,
            "lambda_reconstruction": 1.0,
            "use_ot_loss": False,
            "use_spatial_smooth_loss": False,
            "use_gate_regularization": False,
        },
    }
