"""Configuration helpers for COSIE-style preprocessing.

The paths below point at this project. ``REFERENCE_COSIE_ROOT`` is kept only
for provenance/debugging and is not used as a runtime dependency.
"""

PROJECT_ROOT = "/home/hujinlan/spa_mo_model"
MODEL_DIR = "/home/hujinlan/spa_mo_model/model"
DATA_DIR = "/home/hujinlan/spa_mo_model/data"
UNI_DIR = "/home/hujinlan/spa_mo_model/UNI"

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


def reject_unsupported_model_config(config):
    """Reject known retired or ineffective model fields, not dataset metadata."""
    unsupported = {
        "graph": {"use_spatial_graph", "use_feature_graph"},
        "encoder": {"type", "residual"},
        "contrastive": {
            "method", "loss_weight", "pairwise_all_observed_modalities",
            "use_infonce", "use_temperature", "use_spot_positive_negative_pairs",
        },
        "fusion": {"mode", "input_dim"},
        "graphsage": {"num_layers", "use_distance_weight"},
        "uot": {
            "initial_from_modalities", "use_momentum", "momentum",
            "normalize_total_mass", "cost", "tol", "check_every",
            "clip_cost_min", "clip_cost_max", "keep_dense",
        },
        "ot_attention": {"direction"},
        "reconstruction": {"loss"},
        "loss": {"use_ot_loss", "use_spatial_smooth_loss", "use_gate_regularization"},
    }
    fields = [
        f"{section}.{key}"
        for section, keys in unsupported.items()
        for key in sorted(keys.intersection(config.get(section, {})))
    ]
    fields.extend(sorted({
        "ot_prior_mode", "bidirectional_ot_attention", "dynamic_candidate_source", "metacell",
    }.intersection(config)))
    if fields:
        raise ValueError(f"Unsupported model configuration fields (retired or without a consumer): {fields}")


def reject_unsupported_preprocess_config(config):
    """Reject ineffective preprocessing knobs while retaining input identity."""
    unsupported = {"rna_var_names_source", "superpixel_size", "patch_size", "uni_checkpoint"}
    fields = sorted(unsupported.intersection(config))
    for section in ("preprocessing", "he_image", "paths"):
        fields.extend(f"{section}.{key}" for key in sorted(unsupported.intersection(config.get(section, {}))))
    if fields:
        raise ValueError(f"Unsupported preprocessing configuration fields (without a consumer): {fields}")


def get_default_preprocess_config():
    """Return the default COSIE-style preprocessing configuration."""

    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "model_dir": MODEL_DIR,
            "data_dir": DATA_DIR,
            "uni_dir": UNI_DIR,
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
            "spatial_key": DEFAULT_SPATIAL_KEY,
            "uni_feature_key": DEFAULT_UNI_FEATURE_KEY,
        },
        "he_image": {
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
            # Opt-in compatibility mode for genuinely single-modality data.
            # ``modality`` may be left as None to accept any one supported
            # modality, or set to HE/RNA/Protein/Metabolite to fail fast when
            # a dataset was wired to the wrong input branch.  The default is
            # deliberately disabled so existing multimodal runs are unchanged.
            "single_modality_mode": {
                "enabled": False,
                "modality": None,
            },
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
            "knn_neighbors_spatial": 5,
        },
        "feature_graph": {
            "enabled": False,
        },
        "encoder": {
            "hidden_dims": [256, 128],
            "output_dim": 128,
            "activation": "GELU",
            "dropout": 0.1,
            "norm": "LayerNorm",
            "l2_normalize_output": True,
        },
        "contrastive": {
            "gamma": 5.0,
        },
        "fusion": {
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
            "dropout": 0.1,
            "activation": "GELU",
            "norm": "LayerNorm",
            "residual": True,
            # V6-compatible default. Versioned experiment configs override
            # only this post-OT GraphSAGE residual/neighbor branch scale.
            "post_ot_graphsage_scale": 1.0,
            "delta": 1e-8,
            "edge_batch_size": 200000,
        },
        "uot": {
            "enabled": True,
            # Dynamic sparse OT refreshes from the OT-attention output before
            # the decoder-side GraphSAGE. This prevents post-OT spatial
            # smoothing from feeding back into subsequent OT matching.
            "topology_aware_refresh_enabled": True,
            "topology_context_weight": 0.2,
            "epsilon_init": 0.08,
            "epsilon_update": 0.05,
            "tau_a": 1.0,
            "tau_b": 1.0,
            "max_iter": 1000,
            "update_interval": 20,
            "topk": 10,
        },
        "ot_attention": {
            "enabled": True,
            # The scalar gate uses source/message features (4 * 128).
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
        },
    }
