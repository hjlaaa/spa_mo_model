# Model Validation Report

Generated on 2026-06-08 from direct inspection of `/home/hujinlan/spa_mo_model`.

## 1. Validation scope

This validation was performed from scratch by directly reading the current source code, configuration files, run scripts, existing docs, and selected existing result summaries. No model code, preprocessing code, training config, or result file was modified.

Required files read:

- `model/configure.py`
- `model/data_preprocessing.py`
- `model/multimodal_preprocessing.py`
- `model/image_preprocessing.py`
- `model/utils.py`
- `model/loss.py`
- `model/model_component.py`
- `model/linkage_construction.py`
- `model/stage_model.py`
- `model/__init__.py`
- `scripts/run_mousebrain_v2.py`
- `scripts/run_stage_model.py`

Additional files read because they affect reproducibility or downstream usage:

- `scripts/train_stage_model.py`
- `scripts/run_preprocessing.py`
- `scripts/analyze_mousebrain_clustering.py`
- `data/configs/mousebrain_preprocess_train.json`
- `results/mousebrain_test/fullspot_warmup_schedule/epochs_200/run_summary.json`
- `results/mousebrain_test/fullspot_warmup_schedule/epochs_200/loss_history.json`
- `docs/mousebrain_clustering_analysis_report.md`

Commands run:

```bash
find /home/hujinlan/spa_mo_model/model -maxdepth 1 -type f | sort
find /home/hujinlan/spa_mo_model/scripts -maxdepth 1 -type f | sort
find /home/hujinlan/spa_mo_model/docs -maxdepth 1 -type f | sort
grep -R "crossview_contrastive\|compute_joint\|vicreg\|InfoNCE\|temperature\|positive\|negative\|Prediction_mlp\|GraphAutoencoder\|GraphSAGE\|OTGuidedAttention\|unbalanced_sinkhorn\|uot\|sinkhorn\|topk\|confidence\|row_mass\|decoder\|reconstruction\|lambda_contrast\|lambda_reconstruction\|lambda_contrast_schedule\|weighted_crossview" model scripts -n
python -m py_compile model/configure.py model/data_preprocessing.py model/multimodal_preprocessing.py model/image_preprocessing.py model/utils.py model/loss.py model/model_component.py model/linkage_construction.py model/stage_model.py scripts/run_mousebrain_v2.py scripts/run_stage_model.py
python -m py_compile model/configure.py model/data_preprocessing.py model/multimodal_preprocessing.py model/image_preprocessing.py model/utils.py model/loss.py model/model_component.py model/linkage_construction.py model/stage_model.py scripts/run_mousebrain_v2.py
```

The first compile command failed because `scripts/run_stage_model.py` has a `from __future__ import annotations` statement after a second module-level docstring. The second compile command, excluding only `scripts/run_stage_model.py`, passed.

## 2. Project structure

The current `model/` directory contains exactly these top-level Python source files:

```text
model/__init__.py
model/configure.py
model/data_preprocessing.py
model/image_preprocessing.py
model/linkage_construction.py
model/loss.py
model/model_component.py
model/multimodal_preprocessing.py
model/stage_model.py
model/utils.py
```

This matches the requested COSIE-like structure. There are no extra top-level model files such as:

```text
graph_construction.py
ot_attention.py
uot.py
decoder.py
trainer.py
model_config.py
fusion.py
modality_encoder.py
cosie_alignment.py
```

The current `scripts/` directory contains:

```text
scripts/analyze_mousebrain_clustering.py
scripts/run_mousebrain_v2.py
scripts/run_preprocessing.py
scripts/run_stage_model.py
scripts/train_stage_model.py
```

The current `docs/` directory already contained multiple prior reports before this report was created, including model stage verification, MouseBrain run reports, loss history checks, and clustering analysis reports.

## 3. Overall pipeline summary

The actual implemented pipeline is:

```text
COSIE-style preprocessing:
    input AnnData/modalities per section
    -> data_dict: {modality: [AnnData_or_None_per_section]}
    -> feature_dict: {section: {modality: Tensor[N_s, D_m]}}
    -> spatial_loc_dict: {section: array_or_tensor[N_s, coord_dim]}
    -> processed_data_dict: data_dict with processed obsm entries

StageMultiModalModel:
    feature_dict + spatial_loc_dict
    -> validate section order and exact three-modality set
    -> per-modality MLP encoders
    -> latent_dict[section][modality] = Z_{s,m}, shape [N_s, 128]
    -> COSIE dimension-level cross-view loss within each section
    -> FusionMLP concat in fixed modality order, shape [N_s, 384] -> [N_s, 128]
    -> spatial KNN graph per section, K=5, weighted, undirected, self-looped
    -> one-layer WeightedResidualGraphSAGE, [N_s, 128] -> [N_s, 128]
    -> adjacent-section UOT prior, cached outside backprop
    -> top-k OT-guided one-way attention from each section to next section
    -> final_embeddings[section] = H_tilde_s, shape [N_s, 128]
    -> per-modality decoder reconstructs preprocessed feature_dict targets
    -> total_loss = lambda_reconstruction * reconstruction_loss
                  + lambda_contrast * crossview_loss
```

More explicit pseudocode:

```text
for section s in section_order:
    modalities_s = exactly one of:
        ["HE", "RNA", "Protein"]
        ["HE", "RNA", "Metabolite"]

    for modality m in modalities_s:
        X_{s,m} = feature_dict[s][m]              # [N_s, D_m]
        Z_{s,m} = Encoder_m(X_{s,m})              # [N_s, 128]

    L_cross_s = sum over modality pairs (m1, m2):
        crossview_contrastive_Loss(Z_{s,m1}, Z_{s,m2})

    Z_concat_s = concat([Z_{s,m} in configured order], dim=-1)  # [N_s, 384]
    Z_mlp_s = FusionMLP_body(Z_concat_s)                        # [N_s, 128]
    Z_mean_s = mean_m Z_{s,m}                                   # [N_s, 128]
    Z_s = LayerNorm(Z_mlp_s + Z_mean_s)                         # [N_s, 128]

    G_s = spatial KNN graph from spatial_loc_dict[s]
    H_s = WeightedResidualGraphSAGE(Z_s, G_s)                   # [N_s, 128]

if ot_prior is None:
    initialize OT prior from preprocessed modality features

for adjacent pair (s_t, s_{t+1}):
    H_tilde_{s_t} = OTGuidedAttention(
        source_h=H_{s_t},
        target_h=H_{s_{t+1}},
        prior=top-k UOT prior for s_t -> s_{t+1}
    )

last section:
    H_tilde_last = H_last

for section s, modality m:
    X_hat_{s,m} = Decoder_m(H_tilde_s)          # [N_s, D_m]

L_rec = sum_s sum_m lambda_m * MSE(X_hat_{s,m}, X_{s,m})
L_total = lambda_reconstruction * L_rec + lambda_contrast * sum_s L_cross_s
```

Important implementation notes:

- `processed_data_dict` is accepted by `StageMultiModalModel.forward()` for compatibility but is not used.
- UOT is cached in `self.ot_prior`; it is not a loss term and does not backpropagate.
- Dynamic UOT update uses `final_embeddings` from an evaluation forward pass after optimizer step.
- The last section has no target section and therefore does not receive OT attention.

## 4. Configuration summary

Default preprocessing configuration from `get_default_preprocess_config()`:

```text
paths.project_root = /home/hujinlan/spa_mo_model
paths.model_dir = /home/hujinlan/spa_mo_model/model
paths.data_dir = /home/hujinlan/spa_mo_model/data
paths.uni_dir = /home/hujinlan/spa_mo_model/UNI
paths.uni_checkpoint = /home/hujinlan/spa_mo_model/UNI/pytorch_model.bin
paths.reference_cosie_root = /home/hujinlan/cosie

modalities.canonical = ["HE", "RNA", "Protein", "Metabolite"]
modalities.internal_protein_name = "Protein"
aliases include H&E -> HE, ADT/adt/protein -> Protein, metabolomics variants -> Metabolite

preprocessing.n_comps = 50
preprocessing.hvg_num = 3000
preprocessing.hvg_num_by_modality = None
preprocessing.target_sum = None
preprocessing.use_harmony = True
preprocessing.metacell = False
preprocessing.spatial_key = "spatial"
preprocessing.uni_feature_key = "UNI_feature"
preprocessing.rna_var_names_source = None

he_image.superpixel_size = 16
he_image.patch_size = 224
he_image.batch_size = 128
he_image.num_workers = 4
he_image.device = None
he_image.output_cache_path = None
```

Default model configuration from `get_default_model_config()`:

```text
model.latent_dim = 128
model.modalities_supported = ["HE", "RNA", "Protein", "Metabolite"]
model.valid_modality_sets = [
    ["HE", "RNA", "Protein"],
    ["HE", "RNA", "Metabolite"],
]

graph.use_spatial_graph = True
graph.knn_neighbors_spatial = 5
graph.use_feature_graph = False

encoder.type = "mlp"
encoder.hidden_dims = [256, 128]
encoder.output_dim = 128
encoder.activation = "GELU"
encoder.dropout = 0.1
encoder.norm = "LayerNorm"
encoder.residual = False
encoder.l2_normalize_output = True

contrastive.method = "cosie_crossview"
contrastive.gamma = 5.0
contrastive.loss_weight = 1.0
contrastive.pairwise_all_observed_modalities = True
contrastive.use_infonce = False
contrastive.use_temperature = False
contrastive.use_spot_positive_negative_pairs = False

fusion.mode = "concat_mlp_projection"
fusion.input_dim = 384
fusion.hidden_dims = [256, 128]
fusion.output_dim = 128
fusion.activation = "GELU"
fusion.dropout = 0.1
fusion.norm = "LayerNorm"
fusion.residual = "mean_residual"

training.epochs = 300
training.lr = 0.001
training.weight_decay = 0.0
training.device = "cuda"

graphsage.enabled = True
graphsage.input_dim = 128
graphsage.output_dim = 128
graphsage.num_layers = 1
graphsage.dropout = 0.1
graphsage.activation = "GELU"
graphsage.norm = "LayerNorm"
graphsage.residual = True
graphsage.use_distance_weight = True
graphsage.delta = 1e-8

uot.enabled = True
uot.initial_from_modalities = True
uot.update_from_final_embedding = True
uot.epsilon_init = 0.08
uot.epsilon_update = 0.05
uot.tau_a = 1.0
uot.tau_b = 1.0
uot.max_iter = 1000
uot.tol = 1e-6
uot.check_every = 10
uot.update_interval = 20
uot.topk = 10
uot.use_momentum = False
uot.momentum = 0.0
uot.normalize_total_mass = True
uot.cost = "cosine"
uot.clip_cost_min = 0.0
uot.clip_cost_max = 2.0
uot.keep_dense = False

ot_attention.enabled = True
ot_attention.direction = "forward"
ot_attention.d_attn = 128
ot_attention.beta = 0.2
ot_attention.beta_warmup = False
ot_attention.beta_schedule = [[1, 20, 0.1], [21, 40, 0.3], [41, -1, 0.5]]
ot_attention.dropout = 0.1
ot_attention.gate = "scalar"
ot_attention.use_confidence = True
ot_attention.residual = True
ot_attention.norm = "LayerNorm"
ot_attention.delta = 1e-8

decoder.enabled = True
decoder.hidden_dim = 128
decoder.activation = "GELU"
decoder.dropout = 0.1

reconstruction.enabled = True
reconstruction.loss = "mse"
reconstruction.lambda_by_modality = {
    "HE": 1.0,
    "RNA": 1.0,
    "Protein": 1.0,
    "Metabolite": 1.0,
}

loss.lambda_contrast = 0.1
loss.lambda_reconstruction = 1.0
loss.use_ot_loss = False
loss.use_spatial_smooth_loss = False
loss.use_gate_regularization = False
```

Config fields whose behavior is fixed in code rather than fully switchable:

- `contrastive.loss_weight` is present but not used directly. The actual total-loss weight is `loss.lambda_contrast`.
- `contrastive.pairwise_all_observed_modalities` is present but not used as a conditional switch. The code always computes all pairwise combinations among the three section modalities.
- `graph.use_spatial_graph` is present but forward always builds a spatial graph. Whether GraphSAGE consumes it is controlled by `graphsage.enabled`.
- `graphsage.use_distance_weight` is present but the current forward path always passes computed distance weights to GraphSAGE.
- `uot.normalize_total_mass` is present but current UOT functions always normalize total mass in both initial and dynamic priors.
- `uot.use_momentum` and `uot.momentum` are present but no momentum update is implemented.
- `ot_attention.beta_schedule` is only used if `ot_attention.beta_warmup=True`; the default is `False`, so the default beta is constant `0.2`.

## 5. Input data and preprocessing interface

### 5.1 `data_dict`

The preprocessing entry uses a COSIE-style `data_dict`:

```text
data_dict = {
    "HE": [AnnData_or_None_for_section_1, AnnData_or_None_for_section_2, ...],
    "RNA": [AnnData_or_None_for_section_1, AnnData_or_None_for_section_2, ...],
    "Protein": [AnnData_or_None_for_section_1, AnnData_or_None_for_section_2, ...],
    "Metabolite": [AnnData_or_None_for_section_1, AnnData_or_None_for_section_2, ...],
}
```

The code canonicalizes modality aliases such as `ADT` to `Protein`, `protein` to `Protein`, lowercase `metabolite` to `Metabolite`, and lowercase `rna` to `RNA`.

### 5.2 `feature_dict`

`load_cosie_style_data()` returns:

```text
feature_dict = {
    "s1": {
        "HE": Tensor[N_1, D_HE],
        "RNA": Tensor[N_1, D_RNA],
        "Protein": Tensor[N_1, D_Protein]      # for HE+RNA+Protein
        # or
        "Metabolite": Tensor[N_1, D_Metabolite] # for HE+RNA+Metabolite
    },
    "s2": {...},
}
```

The section keys are generated from zero-based data_dict positions as `"s{index + 1}"`. For shared modalities across more than one section, features are derived from `X_pca_harmony` when `use_harmony=True`, otherwise from `X_pca`. For unique modalities present in only one section, features are derived from `X_pca`.

For the current MouseBrain 200 epoch run, existing `run_summary.json` records:

```text
s1: HE [2384, 50], RNA [2384, 50], Metabolite [2384, 50]
s2: HE [2820, 50], RNA [2820, 50], Metabolite [2820, 50]
s3: HE [2662, 50], RNA [2662, 50], Metabolite [2662, 50]
```

### 5.3 `spatial_loc_dict`

`spatial_loc_dict` is:

```text
spatial_loc_dict = {
    "s1": array_or_tensor[N_1, coord_dim],
    "s2": array_or_tensor[N_2, coord_dim],
}
```

It is built by scanning non-missing modalities for each section and collecting `.obsm["spatial"]`. If multiple modalities provide spatial coordinates, they must be exactly equal. If they differ, preprocessing raises a `ValueError`. The model later also checks that `spatial_loc_dict[section]` has the same row count as the feature matrices.

### 5.4 `processed_data_dict`

`processed_data_dict` is the processed `data_dict` returned by `load_cosie_style_data()`. It contains AnnData objects with added processed embeddings in `.obsm`, such as `RNA_harmony`, `HE_harmony`, `Metabolite_harmony`, or `*_pca`.

The model does not use `processed_data_dict`. `StageMultiModalModel.forward()` emits the message:

```text
processed_data_dict is accepted for pipeline compatibility but is not used by Model Stage V2.
```

### 5.5 Supported modality combinations

The model stage supports exactly these section-level modality sets:

```text
["HE", "RNA", "Protein"]
["HE", "RNA", "Metabolite"]
```

Thus:

- `HE + RNA + Protein` is supported.
- `HE + RNA + Metabolite` is supported.
- A section missing one of the required modalities is not supported by the model stage.
- A section with both `Protein` and `Metabolite` is not supported by the model stage because the present set has four modalities and does not equal either valid three-modality set.
- Sections with two modalities only are not supported by the model stage.

### 5.6 Missing modality behavior

Preprocessing can carry `None` entries in `data_dict`. However, the model stage is stricter:

- `_resolve_modality_order()` requires the keys of `feature_dict[section]` to exactly match one valid three-modality set.
- `initialize_from_feature_dict()` raises if any modality value is `None`.
- Forward requires all modalities in that section to have the same number of spots.

### 5.7 Spot count and row-order handling

Within a section:

- All modality feature tensors must have the same row count.
- Spatial coordinates must have the same row count as the feature tensors.
- Preprocessing checks spatial consistency but does not reorder by `obs_names`.
- `check_obs_names_consistency()` warns if non-missing AnnData objects do not share `obs_names` order.
- MouseBrain-specific preprocessing enforces identical `obs_names` for RNA and Metabolite and copies RNA `obs_names` to HE.

Across sections:

- Different sections may have different spot counts.
- UOT supports rectangular cost/coupling matrices.
- GraphSAGE is constructed independently per section.

### 5.8 Section order

For the model:

- If `section_order` is provided, the model verifies every feature_dict section is listed and preserves that order.
- If `section_order` is not provided, the model uses `sorted(feature_dict.keys())`.

For MouseBrain:

- `data/configs/mousebrain_preprocess_train.json` defines `section_order = ["s1", "s2", "s3"]`.
- `scripts/run_mousebrain_v2.py` uses `config.get("section_order") or sorted(feature_dict.keys())`.

### 5.9 MouseBrain-specific adapters

MouseBrain is handled by `build_mousebrain_section()` and `scripts/run_mousebrain_v2.py`.

Confirmed MouseBrain-specific behavior:

- The current route is `HE + RNA + Metabolite`.
- HE is built from RNA AnnData `obsm["uni_feature"]`, not from the raw image file.
- RNA `var_names` are set from `var["gene_ids"]`.
- Original RNA gene symbols are stored in `var["gene_symbol"]`.
- RNA row order and spot count are verified unchanged after gene ID adaptation.
- RNA and Metabolite must have identical `obs_names`.
- RNA, Metabolite, and HE spatial coordinates must match exactly.
- Config sets `hvg_num_by_modality = {"RNA": 3000, "Metabolite": null}`.
- Therefore RNA uses HVG selection when appropriate, while Metabolite skips HVG.
- Metabolite still follows the non-Protein branch after HVG skipping: normalize_total, log1p, scale, PCA.
- `use_harmony` is `true` in the current MouseBrain config and existing 200 epoch run.

## 6. Modality encoder validation

The encoder class is `ModalityMLPEncoder` in `model/model_component.py`.

### 6.1 Parameter sharing and modality independence

The model stores encoders in:

```python
self.encoders = nn.ModuleDict()
```

Each modality gets one independent encoder:

```text
Encoder_HE
Encoder_RNA
Encoder_Protein
Encoder_Metabolite
```

Only modalities present in the supplied `feature_dict` are instantiated. Parameters are shared across sections for the same modality. They are not section-specific.

### 6.2 Input dimension

Input dimensions are inferred from `feature_dict[section][modality].shape[1]` during `initialize_from_feature_dict()`.

Rules:

- Feature tensors must be 2D.
- If the same modality appears in multiple sections, its feature dimension must be identical across sections.
- If the input dimension differs across sections for the same modality, initialization raises `ValueError`.

For current MouseBrain preprocessing, all three modalities are 50-dimensional PCA/Harmony features.

### 6.3 Architecture

Default architecture for each modality:

```text
Linear(input_dim, 256)
LayerNorm(256)
GELU
Dropout(0.1)
Linear(256, 128)
LayerNorm(128)
GELU
Dropout(0.1)
Linear(128, 128)
L2 normalize over dim=1
```

There is no residual connection in the encoder. The config field `encoder.residual=False` matches this.

Formula:

```text
X_{s,m} in R^{N_s x D_m}
Z_{s,m} = normalize_l2(MLP_m(X_{s,m})) in R^{N_s x 128}
```

Output structure:

```text
latent_dict = {
    section: {
        modality: Tensor[N_section, 128]
    }
}
```

For `HE + RNA + Protein`, the section latent dict contains `HE`, `RNA`, and `Protein`.

For `HE + RNA + Metabolite`, the section latent dict contains `HE`, `RNA`, and `Metabolite`.

## 7. Alignment / contrastive loss validation

The current alignment loss is COSIE-style `crossview_contrastive_Loss` from `model/loss.py`.

### 7.1 What loss is used

Implemented:

- `compute_joint()`
- `crossview_contrastive_Loss()`
- `compute_pairwise_cosie_crossview_loss()`

Not implemented or not used:

- VICReg: not present.
- InfoNCE: not present.
- Temperature-scaled contrastive learning: not present.
- Spot-level positive/negative pair mining: not present.
- `[N, N]` spot similarity matrix: not present.

The keyword search found no active implementation of VICReg, InfoNCE, `Prediction_mlp`, or `GraphAutoencoder`. Those names only appear in explanatory comments/config text where the code says this project is not using them.

### 7.2 `compute_joint()`

Inputs:

```text
view1: Tensor[N, K]
view2: Tensor[N, K]
```

The code requires same shape and computes a dimension-level joint dependency matrix:

```text
p_ij_raw[a,b] = sum_n view1[n,a] * view2[n,b]
p_ij_sym = (p_ij_raw + p_ij_raw^T) / 2
p_ij = p_ij_sym / sum_{a,b} p_ij_sym[a,b]
```

Output:

```text
p_ij: Tensor[K, K]
```

With default latent dimension, `K=128`, so the joint matrix is `[128, 128]`, not `[N, N]`.

### 7.3 `crossview_contrastive_Loss()`

Inputs:

```text
view1: Tensor[N, 128]
view2: Tensor[N, 128]
gamma: default from config is 5.0
EPS: sys.float_info.epsilon
```

Steps:

```text
p_ij = compute_joint(view1, view2)
p_i = row_marginal(p_ij), expanded to [K, K]
p_j = col_marginal(p_ij), expanded to [K, K]
p_ij, p_i, p_j are clamped from below by EPS via torch.where

loss = sum_{i,j} -p_ij * (
    log(p_ij)
    - (gamma + 1) * log(p_j)
    - (gamma + 1) * log(p_i)
)
```

The loss can be negative. Existing MouseBrain 200 epoch `run_summary.json` records:

```text
crossview_loss = -789.4046630859375
reconstruction_loss = 25.804363250732422
total_loss = 17.910316467285156
```

This is consistent with:

```text
total_loss = reconstruction_loss + lambda_contrast * crossview_loss
```

with final scheduled `lambda_contrast = 0.01`.

### 7.4 Pair combinations

`compute_pairwise_cosie_crossview_loss()` uses `itertools.combinations()` over all present latent tensors in the section's insertion order.

For `HE + RNA + Protein`, the pairs are:

```text
HE__RNA
HE__Protein
RNA__Protein
```

For `HE + RNA + Metabolite`, the pairs are:

```text
HE__RNA
HE__Metabolite
RNA__Metabolite
```

Pair losses are summed within each section. Section losses are summed across sections. There is no averaging over pairs or sections.

### 7.5 Entry into total loss

Forward accumulates:

```text
crossview_loss = sum_s sum_pairs L_cross(s, pair)
reconstruction_loss = sum_s sum_m lambda_m * MSE(decoder_m(H_tilde_s), X_{s,m})
total_loss = lambda_reconstruction * reconstruction_loss
           + lambda_contrast * crossview_loss
```

Default:

```text
lambda_reconstruction = 1.0
lambda_contrast = 0.1
```

In MouseBrain warmup training, `scripts/run_mousebrain_v2.py` overrides `model.config["loss"]["lambda_contrast"]` every epoch if `--lambda_contrast_schedule` is provided.

### 7.6 Lambda schedule support

`scripts/run_mousebrain_v2.py` supports `--lambda_contrast_schedule`.

Schedule format:

```text
"1-5:1e-4,6-10:3e-4,11-15:1e-3"
```

Parsing rules:

- Split on commas.
- Each chunk must contain `range:value`.
- Range must be `start-end`.
- `start_epoch > 0`.
- `end_epoch >= start_epoch`.
- First matching range supplies the lambda for the epoch.
- If no schedule entry matches an epoch, the default `model_config["loss"]["lambda_contrast"]` is used.
- `--lambda_contrast` and `--lambda_contrast_schedule` are mutually exclusive.

The existing MouseBrain 200 epoch run used:

```text
1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2
```

`loss_history.json` records:

```text
epoch
lambda_contrast
total_loss
crossview_loss
reconstruction_loss
weighted_crossview_loss
```

The generic `scripts/train_stage_model.py` does not implement `lambda_contrast_schedule` and its `training_history.json` does not record `lambda_contrast` or `weighted_crossview_loss`.

## 8. FusionMLP validation

`FusionMLP` is implemented in `model/model_component.py` and instantiated once per valid modality set in `StageMultiModalModel.__init__()`.

### 8.1 Input and ordering

Fusion expects exactly three modality latents. The order is fixed by `model.valid_modality_sets`.

For `HE + RNA + Protein`:

```text
modality_order = ["HE", "RNA", "Protein"]
Z_concat = concat([Z_HE, Z_RNA, Z_Protein], dim=-1)
```

For `HE + RNA + Metabolite`:

```text
modality_order = ["HE", "RNA", "Metabolite"]
Z_concat = concat([Z_HE, Z_RNA, Z_Metabolite], dim=-1)
```

Each latent must have shape `[N, 128]`.

### 8.2 Architecture

Default architecture:

```text
Input: [N, 384]
Linear(384, 256)
LayerNorm(256)
GELU
Dropout(0.1)
Linear(256, 128)
LayerNorm(128)
GELU
Dropout(0.1)
Linear(128, 128)
mean residual
LayerNorm(128)
Output: [N, 128]
```

Formula:

```text
Z_concat = concat([Z_HE, Z_RNA, Z_omics]) in R^{N x 384}
Z_mlp = FusionMLP_body(Z_concat) in R^{N x 128}
Z_mean = mean(Z_HE, Z_RNA, Z_omics) in R^{N x 128}
Z = LayerNorm(Z_mlp + Z_mean) in R^{N x 128}
```

There is no L2 normalization after fusion.

## 9. Spatial graph validation

Spatial graph construction is implemented by `compute_spatial_knn_graph_with_weights()` in `model/utils.py` and called once per section in `StageMultiModalModel.forward()`.

### 9.1 Graph construction

Defaults:

```text
k = graph.knn_neighbors_spatial = 5
include_self_loop = True
undirected = True
delta = graphsage.delta = 1e-8
```

The implementation uses `sklearn.neighbors.NearestNeighbors` with its default Euclidean distance. It fits KNN independently for each section.

For each source spot `i`, the KNN query uses `n_neighbors=k+1` to include the self neighbor.

### 9.2 Raw edge weight formula

Let:

```text
d_ij = Euclidean distance from source i to neighbor j
sigma_i = max(distance from i to its kth non-self neighborhood boundary, delta)
```

The code sets `sigma_i = distances[i, -1]` from the `k+1` query result, clamped by `delta`.

Raw weights:

```text
raw_w_ii = 1.0 for self-loops
raw_w_ij = exp(-d_ij^2 / (sigma_i^2 + delta)) for non-self KNN edges
```

When `undirected=True`, the reverse edge `(j, i)` is added with the same raw weight. If duplicate edges occur, they are coalesced by taking the larger raw weight.

Self loops are added for every source after the KNN pass if missing.

### 9.3 Source-row normalization

For each source row:

```text
row_sum_i = sum_j raw_w_ij
edge_weight_ij = raw_w_ij / (row_sum_i + delta)
```

Thus edge weights are normalized by source. Each source row sums approximately to 1.

Output:

```text
edge_index: LongTensor[2, E]
edge_weight: FloatTensor[E]
```

Edges are sorted by `(source, target)`. Spot order is not changed.

### 9.4 Feature graph

`graph.use_feature_graph=False` by default. No feature graph is constructed in the stage model.

## 10. WeightedResidualGraphSAGE validation

`WeightedResidualGraphSAGE` is implemented in `model/model_component.py`.

Defaults:

```text
enabled = True
input_dim = 128
output_dim = 128
num_layers = 1
activation = GELU
dropout = 0.1
norm = LayerNorm
residual = True
use_distance_weight = True
```

The class is one-layer. There is no loop over `num_layers`; the config value is descriptive for the current implementation.

### 10.1 Aggregation

Inputs:

```text
x: Tensor[N, 128]
edge_index: Tensor[2, E]
edge_weight: Tensor[E]
```

The code interprets:

```text
source = edge_index[0]
target = edge_index[1]
```

Aggregation:

```text
n_i = sum over edges (i -> j) edge_weight_ij * x_j
```

Because the spatial graph includes self-loops, `x_i` also contributes to `n_i` through the self-loop edge. Separately, the layer also applies a self linear transform to `x_i`.

### 10.2 Layer formula

With `W_self`, `W_neigh`, and bias `b`:

```text
n_i = sum_j alpha_ij x_j
o_i = GELU(W_self x_i + W_neigh n_i + b)
o_i = Dropout(o_i)
h_i = LayerNorm(x_i + o_i)
```

If `residual=False`, the residual addition is skipped:

```text
h_i = LayerNorm(o_i)
```

### 10.3 Epoch state

Each forward pass starts from the current fused embedding `Z`. The GraphSAGE module does not cache graph embeddings and does not store `H` from one epoch as the next epoch input.

## 11. UOT / OT prior validation

This is the project-specific cross-stage matching component. It lives in `model/linkage_construction.py` and is called by `StageMultiModalModel.initialize_ot_prior()` and `StageMultiModalModel.update_ot_prior()`.

### 11.1 Position in the model

Actual position:

```text
Initial prior:
    preprocessed feature_dict modality embeddings
    -> global z-score per modality
    -> L2 normalize
    -> cosine cost
    -> unbalanced Sinkhorn
    -> dense coupling P_m
    -> total-mass normalization per modality
    -> mean over modalities
    -> total-mass normalization
    -> top-k sparse prior
    -> cached self.ot_prior

Dynamic update:
    final_embeddings H_tilde from eval forward
    -> detach / no_grad
    -> L2 normalize
    -> cosine cost
    -> unbalanced Sinkhorn
    -> total-mass normalization
    -> top-k sparse prior
    -> overwrite self.ot_prior

Forward attention:
    GraphSAGE embeddings H
    + cached top-k OT prior
    -> OT-guided attention
    -> final_embeddings H_tilde
```

UOT is not a loss. It does not participate in backpropagation. All UOT functions are decorated with `@torch.no_grad()`, and tensor conversion detaches tensors.

### 11.2 `unbalanced_sinkhorn()`

Function:

```text
unbalanced_sinkhorn(cost, a=None, b=None, epsilon=0.05,
                    tau_a=1.0, tau_b=1.0,
                    max_iter=1000, tol=1e-6,
                    check_every=10, delta=1e-8)
```

Input:

```text
cost: Tensor[n_source, n_target]
```

The function converts `cost` to a detached float32 tensor. It requires `cost.ndim == 2`.

Default masses:

```text
a_i = 1 / n_source for i=1..n_source
b_j = 1 / n_target for j=1..n_target
```

If `a` or `b` are provided, they are also converted to detached float32 tensors.

Kernel and exponents:

```text
K = exp(-C / epsilon).clamp_min(delta)
rho_a = tau_a / (tau_a + epsilon)
rho_b = tau_b / (tau_b + epsilon)
```

Initialization:

```text
u = ones(n_source)
v = ones(n_target)
```

Update loop:

```text
for iteration in range(max_iter):
    prev_u = u
    prev_v = v
    u = (a / (K v + delta)) ^ rho_a
    v = (b / (K^T u + delta)) ^ rho_b

    every check_every iterations:
        diff = max(max_abs(u - prev_u), max_abs(v - prev_v))
        if diff < tol:
            break
```

Output coupling:

```text
P = diag(u) K diag(v)
P = nan_to_num(P, nan=0, posinf=0, neginf=0)
```

Output shape:

```text
Tensor[n_source, n_target]
```

The function is no-grad and returns a detached coupling in practice.

### 11.3 Cost matrix

Cost construction is implemented in `cosine_cost_matrix()` in `model/utils.py`.

Formula:

```text
X_norm = X / max(||X||_2, eps)
Y_norm = Y / max(||Y||_2, eps)
C = 1 - X_norm Y_norm^T
C = clamp(C, clip_min, clip_max)
```

Defaults:

```text
clip_min = 0.0
clip_max = 2.0
eps/delta = 1e-8
```

Initial UOT computes a separate cost for each shared modality in adjacent sections. Dynamic UOT computes a cost from final embeddings only.

### 11.4 Initial multimodal UOT prior

Functions:

```text
StageMultiModalModel.initialize_ot_prior()
compute_initial_multimodal_uot_prior()
```

When initialized:

- `scripts/run_mousebrain_v2.py` explicitly calls `model.initialize_ot_prior(feature_dict, section_order)` before training.
- `scripts/train_stage_model.py` also explicitly calls it before training.
- If not explicitly called, `StageMultiModalModel.forward()` auto-initializes when UOT and OT attention are enabled and `self.ot_prior is None`.

Inputs:

```text
feature_dict
section_order
modalities = ["HE", "RNA", "Protein", "Metabolite"]
```

Only adjacent section pairs are used:

```text
(s1, s2), (s2, s3), ...
```

Per modality:

1. Skip the modality if absent in source or target section.
2. Compute global z-score stats across all sections in `section_order` that have this modality.
3. Require the same feature dimension across sections for that modality.
4. For source and target features:

```text
X_z = (X - global_mean_m) / global_std_m
X_norm = L2_normalize(X_z)
```

5. Compute cosine cost.
6. Run UOT with `epsilon_init`.
7. Normalize total mass:

```text
P_m_norm = P_m / (sum(P_m) + delta)
```

Multimodal averaging:

```text
P_init = normalize_total_mass(mean_m P_m_norm)
```

Then:

```text
sparse_prior = sparsify_coupling_topk(P_init, topk=10)
```

Returned structure for each adjacent pair:

```text
ot_prior[(source_section, target_section)] = {
    "P_dense": P_init if keep_dense else None,
    "topk_idx": Tensor[n_source, effective_topk],
    "topk_weight": Tensor[n_source, effective_topk],
    "confidence": Tensor[n_source],
    "row_mass": Tensor[n_source],
    "modalities_used": list[str],
}
```

Default `keep_dense=False`, so dense P is not retained. This is the memory-saving strategy.

For current MouseBrain, `modalities_used` should be `["HE", "RNA", "Metabolite"]` for adjacent pairs because `Protein` is absent.

### 11.5 Dynamic OT prior update

Functions:

```text
should_update_ot(epoch, update_interval=20)
StageMultiModalModel.update_ot_prior()
update_uot_prior_from_embeddings()
```

Update condition:

```text
epoch > 0 and epoch % update_interval == 0
```

Default update interval:

```text
20 epochs
```

Training scripts update after optimizer step:

```text
outputs = model(...)
loss.backward()
optimizer.step()

if should_update_ot(epoch, update_interval):
    model.eval()
    with torch.no_grad():
        eval_outputs = model(...)
        model.update_ot_prior(eval_outputs["final_embeddings"], section_order)
```

Thus dynamic UOT uses `final_embeddings`, i.e. `H_tilde`, from an eval forward pass with the current cached prior. It does not use raw modality features, fused embeddings, or pure GraphSAGE embeddings during update.

Dynamic update details:

```text
source = L2_normalize(detach(final_embeddings[source_section]))
target = L2_normalize(detach(final_embeddings[target_section]))
C = clipped cosine cost
P = unbalanced_sinkhorn(C, epsilon=epsilon_update)
P = normalize_total_mass(P)
sparse = top-k(P)
self.ot_prior = new priors
```

Only adjacent section pairs are updated.

No momentum is implemented. `self.ot_prior` is overwritten.

No OT loss is computed, and no gradient flows through UOT.

The existing 200 epoch MouseBrain run recorded OT updates at:

```text
20, 40, 60, 80, 100, 120, 140, 160, 180, 200
```

Because the script performs a final forward after the epoch loop, the final saved embeddings after epoch 200 use the OT prior refreshed at epoch 200.

### 11.6 Top-k sparsification

Function:

```text
sparsify_coupling_topk(P, topk=10, delta=1e-8)
```

Input:

```text
P: Tensor[n_source, n_target]
```

Effective top-k:

```text
effective_topk = min(topk, n_target)
```

Top-k extraction:

```text
values, indices = torch.topk(P, k=effective_topk, dim=1)
```

Shapes:

```text
topk_idx: Tensor[n_source, effective_topk]
topk_weight: Tensor[n_source, effective_topk]
row_mass: Tensor[n_source]
confidence: Tensor[n_source]
```

Formula:

```text
row_topk_mass_i = sum_{j in topk_i} P_ij
topk_weight_ij = P_ij / (row_topk_mass_i + delta)

row_mass_i = sum_j P_ij
uniform_source_mass = 1 / n_source
confidence_i = min(1, row_mass_i / (uniform_source_mass + delta))
```

If a row's top-k mass is extremely small, `topk_weight` becomes near zero because the denominator is only protected by `delta`; there is no fallback to uniform top-k weights.

### 11.7 UOT parameters

Defaults:

```text
epsilon_init = 0.08
epsilon_update = 0.05
tau_a = 1.0
tau_b = 1.0
max_iter = 1000
tol = 1e-6
check_every = 10
topk = 10
cost = cosine
clip_cost_min = 0.0
clip_cost_max = 2.0
keep_dense = False
```

`normalize_total_mass=True` is reflected by actual behavior but not used as a conditional flag.

## 12. OT-guided attention validation

`OTGuidedAttention` is implemented in `model/model_component.py` and instantiated once in `StageMultiModalModel`.

### 12.1 Direction and section handling

The attention is single-direction and adjacent only.

For each adjacent pair:

```text
source_section = section_order[t]
target_section = section_order[t+1]
source_h = graphsage_embeddings[source_section]
target_h = graphsage_embeddings[target_section]
```

The source section attends to top-k target spots selected by UOT.

The last section has no next section:

```text
final_embeddings[last_section] = graphsage_embeddings[last_section]
```

There is no bidirectional attention and no temporal Transformer.

### 12.2 Parameters

Defaults:

```text
dim = 128
d_attn = 128
beta = 0.2
beta_warmup = False
dropout = 0.1
gate = "scalar"
use_confidence = True
residual = True
norm = LayerNorm
delta = 1e-8
```

Projection layers:

```text
W_Q: Linear(128, 128)
W_K: Linear(128, 128)
W_V: Linear(128, 128)
W_O: Linear(128, 128)
```

Gate MLP:

```text
Linear(4 * 128, 128)
GELU
Dropout(0.1)
Linear(128, 1)
Sigmoid
```

Note: although `d_attn` is configurable, the gate input computes `source_h - message` and `source_h * message`. This assumes `d_attn == dim`. The default satisfies this, but non-128 `d_attn` would be shape-incompatible unless the code is changed.

### 12.3 Attention formula

Inputs:

```text
source_h: Tensor[N_source, 128]
target_h: Tensor[N_target, 128]
topk_idx: Tensor[N_source, topk]
topk_weight: Tensor[N_source, topk]
confidence: Tensor[N_source]
```

Candidate target embeddings:

```text
candidate_h_i = target_h[topk_idx_i]   # [topk, 128]
```

Projections:

```text
q_i = W_Q h_i
k_ij = W_K candidate_h_ij
v_ij = W_V candidate_h_ij
```

Score:

```text
score_ij = q_i^T k_ij / sqrt(d_attn)
         + beta * log(max(topk_weight_ij, delta))
```

Softmax:

```text
alpha_ij = softmax_j(score_ij)
```

Message:

```text
m_i = sum_j alpha_ij v_ij
m_bar_i = W_O m_i
```

Gate:

```text
gate_i = sigmoid(MLP([h_i, m_i, h_i - m_i, h_i * m_i]))
```

The gate uses the raw attention message `m_i`, not the projected `m_bar_i`. The residual update uses the projected `m_bar_i`.

Update:

```text
scale_i = confidence_i * gate_i       # if use_confidence=True
update_i = Dropout(scale_i * m_bar_i)
h_tilde_i = LayerNorm(h_i + update_i) # if residual=True
```

If `use_confidence=False`, `scale_i = gate_i`.

If `residual=False`, `h_tilde_i = LayerNorm(update_i)`.

Output shape:

```text
Tensor[N_source, 128]
```

### 12.4 Beta schedule

`OTGuidedAttention._current_beta()` supports schedule lookup only when:

```text
beta_warmup = True
epoch is not None
```

Default `beta_warmup=False`, so the default beta is constant:

```text
beta = 0.2
```

The existing MouseBrain 200 epoch run used a lambda contrast schedule, not an OT attention beta warmup schedule.

## 13. Decoder and reconstruction validation

### 13.1 ModalityDecoder

`ModalityDecoder` is implemented in `model/model_component.py`.

There is one decoder per modality in `self.decoders = nn.ModuleDict()`. Decoders are shared across sections for the same modality and are not section-specific.

Output dimension is inferred from the original model input dimension:

```text
output_dim_m = feature_dict[first_section_with_m][m].shape[1]
```

The same modality must have the same feature dimension across sections.

Default decoder structure:

```text
Input: H_tilde_s [N_s, 128]
Linear(128, 128)
GELU
Dropout(0.1)
Linear(128, D_m)
Output: X_hat_{s,m} [N_s, D_m]
```

Supported modalities are whichever of `HE`, `RNA`, `Protein`, and `Metabolite` are present in a valid modality set.

Output structure:

```text
reconstructions = {
    section: {
        modality: Tensor[N_section, original_preprocessed_feature_dim]
    }
}
```

### 13.2 Reconstruction target

The reconstruction target is the preprocessed feature tensor consumed by the model:

```text
target = feature_dict[section][modality]
```

It is not:

- Raw HE image pixels.
- Raw RNA count matrix.
- Raw protein count matrix.
- Raw metabolite intensity matrix.

For current MouseBrain, reconstruction targets are 50-dimensional processed features for HE, RNA, and Metabolite.

### 13.3 Reconstruction loss

Function:

```text
compute_reconstruction_loss(recon_dict_for_one_section,
                            target_feature_dict_for_one_section,
                            lambda_by_modality)
```

For each modality:

```text
L_rec_{s,m} = MSE(X_hat_{s,m}, X_{s,m})
```

`torch.nn.functional.mse_loss()` uses mean reduction by default.

Section reconstruction loss:

```text
L_rec_s = sum_m lambda_m * L_rec_{s,m}
```

All sections are summed:

```text
L_rec = sum_s L_rec_s
```

There is no averaging across sections or modalities beyond the internal MSE mean over tensor elements.

Default modality weights:

```text
HE = 1.0
RNA = 1.0
Protein = 1.0
Metabolite = 1.0
```

## 14. Total loss and training loop validation

### 14.1 Total loss

The forward method returns:

```text
losses = {
    "total_loss": total_loss,
    "crossview_loss": crossview_loss,
    "reconstruction_loss": reconstruction_loss,
}
```

Formula:

```text
total_loss = lambda_reconstruction * reconstruction_loss
           + lambda_contrast * crossview_loss
```

Default:

```text
lambda_reconstruction = 1.0
lambda_contrast = 0.1
```

Because `crossview_loss` can be negative, it can reduce the total loss.

### 14.2 Losses implemented and not implemented

Implemented and used:

- Reconstruction loss.
- COSIE-style alignment/cross-view contrastive loss.

Not implemented or not used:

- OT loss: config says `use_ot_loss=False`; no OT loss code is present.
- Spatial smooth loss: config says `use_spatial_smooth_loss=False`; no smooth loss code is present.
- Gate regularization: config says `use_gate_regularization=False`; no gate regularization code is present.
- Triplet loss: not present.
- Prediction loss: not present.
- Missing modality loss: not present.
- InfoNCE: not present.
- VICReg: not present.

### 14.3 `scripts/run_mousebrain_v2.py` training loop

MouseBrain training flow:

```text
load JSON config
build MouseBrain HE+RNA+Metabolite sections
load_cosie_style_data(...)
section_order = config["section_order"] or sorted(feature_dict.keys())
model_config = get_default_model_config() with training/config overrides
parse optional lambda_contrast_schedule
model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
model.initialize_ot_prior(feature_dict, section_order)
dry forward under no_grad for summary

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=model_config["training"]["lr"],
    weight_decay=model_config["training"]["weight_decay"],
)

for epoch in 1..epochs:
    model.train()
    current_lambda_contrast = lambda_for_epoch(...)
    model.config["loss"]["lambda_contrast"] = current_lambda_contrast
    outputs = model(..., epoch=epoch)
    loss = outputs["losses"]["total_loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    append history record

    if should_update_ot(epoch, update_interval):
        model.eval()
        with torch.no_grad():
            eval_outputs = model(..., epoch=epoch)
            model.update_ot_prior(eval_outputs["final_embeddings"], section_order)

final eval forward under no_grad
save final embeddings, run_summary.json, loss_history.json, config copy
```

Optimizer:

```text
Adam
lr = training.lr
weight_decay = training.weight_decay
```

Default model config has `device="cuda"` but `_select_device()` falls back to CPU if CUDA is unavailable. MouseBrain JSON default has `device="cpu"`, while the existing 200 epoch run summary records `device="cuda"`, so that run used a CLI or config override to CUDA.

### 14.4 `scripts/train_stage_model.py` training loop

Generic training supports:

- `--input_bundle`
- `--preprocess_config`
- `--model_config`
- `--output_dir`
- `--section_order`
- `--epochs`
- `--lr`
- `--weight_decay`
- `--device`
- `--ot_update_interval`
- `--save_embeddings`
- `--smoke_test`

It does not support `lambda_contrast_schedule`.

It saves:

```text
stage_model_v2_last.pt
training_history.json
training_summary.json
final_embeddings/*.npy if requested
```

The checkpoint includes model state dict, config, history, section_order, and input_dims. It does not save `self.ot_prior` because `ot_prior` is not a PyTorch parameter or buffer.

### 14.5 Existing 200 epoch MouseBrain schedule

Existing `run_summary.json` for:

```text
results/mousebrain_test/fullspot_warmup_schedule/epochs_200
```

records:

```text
epochs = 200
lambda_contrast_schedule = "1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2"
ot_updates = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
use_harmony = true
hvg_num_by_modality = {"RNA": 3000, "Metabolite": null}
```

`loss_history.json` records `lambda_contrast` and `weighted_crossview_loss` for every epoch.

## 15. Forward output and saved artifacts

### 15.1 `StageMultiModalModel.forward()` return fields

Actual return dict:

```text
{
    "fused_embeddings": dict[section -> Tensor[N_s, 128]],
    "graphsage_embeddings": dict[section -> Tensor[N_s, 128]],
    "final_embeddings": dict[section -> Tensor[N_s, 128]],
    "latent_dict": dict[section -> dict[modality -> Tensor[N_s, 128]]],
    "reconstructions": dict[section -> dict[modality -> Tensor[N_s, D_m]]],
    "spatial_graph_dict": dict[section -> {
        "edge_index": Tensor[2, E_s],
        "edge_weight": Tensor[E_s],
    }],
    "ot_prior": self.ot_prior,
    "losses": {
        "total_loss": Tensor scalar,
        "crossview_loss": Tensor scalar,
        "reconstruction_loss": Tensor scalar,
    },
    "loss_details": {
        "crossview": dict[section -> dict[pair_name -> Tensor scalar]],
        "reconstruction": dict[section -> dict[modality -> Tensor scalar]],
    },
    "messages": list[str],
}
```

This matches the requested fields. No extra `target_feature_dict` is returned.

### 15.2 MouseBrain saved artifacts

`scripts/run_mousebrain_v2.py` saves:

```text
output_dir/mousebrain_config_used.json
output_dir/run_summary.json
output_dir/loss_history.json           # training mode only
output_dir/final_embeddings/{section}_final_embedding.npy
```

It saves final embeddings only. It does not save:

- `fused_embeddings`
- `graphsage_embeddings`
- `latent_dict`
- decoder reconstructions
- dense OT prior
- top-k OT matching tensors

`run_summary.json` stores `ot_prior_keys` but not top-k indices or weights.

### 15.3 Generic training saved artifacts

`scripts/train_stage_model.py` saves, if `--output_dir` is provided:

```text
stage_model_v2_last.pt
training_history.json
training_summary.json
final_embeddings/{section}_final_embedding.npy  # only if --save_embeddings or --smoke_test
```

### 15.4 Downstream clustering embedding

`scripts/analyze_mousebrain_clustering.py` reads:

```text
{embedding_dir}/{section}_final_embedding.npy
```

Therefore downstream clustering uses `final_embeddings`, not `fused_embeddings`, not `graphsage_embeddings`, and not modality latents.

The existing clustering report confirms the 200 epoch full-spot final embeddings:

```text
s1_final_embedding.npy: (2384, 128)
s2_final_embedding.npy: (2820, 128)
s3_final_embedding.npy: (2662, 128)
```

## 16. MouseBrain-specific run configuration

Current MouseBrain config path:

```text
data/configs/mousebrain_preprocess_train.json
```

Config contents:

```text
dataset_name = "MouseBrain"
modalities = ["HE", "RNA", "Metabolite"]
section_order = ["s1", "s2", "s3"]

s1 RNA = data/dataset_MouseBrain/dataset_MouseBrain_SectionA/adata_RNA.h5ad
s1 Metabolite = data/dataset_MouseBrain/dataset_MouseBrain_SectionA/adata_meta.h5ad
s2 RNA = data/dataset_MouseBrain/dataset_MouseBrain_SectionB/adata_RNA.h5ad
s2 Metabolite = data/dataset_MouseBrain/dataset_MouseBrain_SectionB/adata_meta.h5ad
s3 RNA = data/dataset_MouseBrain/dataset_MouseBrain_SectionC/adata_RNA.h5ad
s3 Metabolite = data/dataset_MouseBrain/dataset_MouseBrain_SectionC/adata_meta.h5ad

preprocessing.n_comps = 50
preprocessing.use_harmony = true
preprocessing.spatial_key = "spatial"
preprocessing.uni_feature_key = "uni_feature"
preprocessing.rna_gene_id_key = "gene_ids"
preprocessing.hvg_num_by_modality = {"RNA": 3000, "Metabolite": null}

training.epochs = 5 in the JSON
training.lr = 0.001
training.weight_decay = 0.0
training.device = "cpu" in the JSON
training.max_spots_per_section = null
training.output_dir = /home/hujinlan/spa_mo_model/results/mousebrain_test
```

The existing 200 epoch run was not simply the JSON default. Existing output records show:

```text
result directory = /home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200
epochs = 200
device = cuda
lambda_contrast_schedule = 1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2
max_spots_per_section = null
```

MouseBrain actual data summary from existing 200 epoch run:

```text
s1:
    raw RNA: [2384, 32285]
    raw Metabolite: [2384, 1538]
    HE from uni_feature: [2384, 2048]
    processed feature_dict: HE/RNA/Metabolite each [2384, 50]

s2:
    raw RNA: [2820, 32285]
    raw Metabolite: [2820, 1538]
    HE from uni_feature: [2820, 2048]
    processed feature_dict: HE/RNA/Metabolite each [2820, 50]

s3:
    raw RNA: [2662, 32285]
    raw Metabolite: [2662, 1538]
    HE from uni_feature: [2662, 2048]
    processed feature_dict: HE/RNA/Metabolite each [2662, 50]
```

Existing run output:

```text
final_embeddings:
    s1 [2384, 128]
    s2 [2820, 128]
    s3 [2662, 128]

fused_embeddings:
    s1 [2384, 128]
    s2 [2820, 128]
    s3 [2662, 128]

graphsage_embeddings:
    s1 [2384, 128]
    s2 [2820, 128]
    s3 [2662, 128]

reconstructions:
    all sections: HE/RNA/Metabolite [N_s, 50]

ot_prior_keys:
    ("s1", "s2")
    ("s2", "s3")
```

## 17. Deviations from intended design

Overall, the core model pipeline substantially matches the intended design: modality-specific encoders, COSIE-style within-section alignment, 384-to-128 fusion, spatial KNN, weighted residual GraphSAGE, adjacent UOT prior, top-k OT-guided one-way attention, final embeddings, decoder reconstruction, and total reconstruction plus alignment loss are implemented.

However, the following deviations or implementation caveats were found.

### 17.1 `scripts/run_stage_model.py` fails py_compile

- File: `scripts/run_stage_model.py`
- Actual implementation: line 2 has one docstring and line 3 has a second string literal before `from __future__ import annotations` at line 4.
- Expected: future imports must occur after the module docstring and before any other statement.
- Severity: high for the requested compile validation, low for core model training because this is a smoke-test script.
- Suggested handling: move the second string into the first docstring or below the future import.

### 17.2 `processed_data_dict` is not used by the model

- File: `model/stage_model.py`
- Actual implementation: `processed_data_dict` is accepted for compatibility and ignored.
- Expected design text mentions preprocessing obtains `feature_dict / spatial_loc_dict / processed_data_dict`.
- Severity: low if `feature_dict` is intended as the model input; important for documentation.
- Suggested handling: keep as documented compatibility argument or remove from model-facing API if not needed.

### 17.3 Missing modality and four-modality cases are rejected

- File: `model/stage_model.py`
- Actual implementation: each section must exactly match `HE+RNA+Protein` or `HE+RNA+Metabolite`.
- Expected design: input is either `HE+RNA+Protein` or `HE+RNA+Metabolite`, so this is consistent for complete three-modality runs.
- Severity: low for current design, but relevant if future experiments require missing modality or four-modality training.
- Suggested handling: no change needed for the current stated design.

### 17.4 Several config switches are declarative but not operational

- Files: `model/configure.py`, `model/stage_model.py`, `model/linkage_construction.py`
- Actual implementation:
  - `contrastive.loss_weight` is not used; `loss.lambda_contrast` is used.
  - `contrastive.pairwise_all_observed_modalities` is not a switch; all pairs are always used.
  - `graph.use_spatial_graph` is not a switch in forward; the graph is always built.
  - `graphsage.use_distance_weight` is not a switch; weighted edge weights are always passed.
  - `uot.normalize_total_mass` is not a switch; normalization always occurs.
  - `uot.use_momentum` and `uot.momentum` are not implemented.
- Expected: config fields often imply controllable behavior.
- Severity: medium for reproducibility if users try non-default configs; low for current defaults.
- Suggested handling: document as fixed behavior or implement the switches.

### 17.5 `ot_attention.d_attn` is only safe at default value

- File: `model/model_component.py`
- Actual implementation: gate input uses `source_h - message` and `source_h * message`, requiring `message` to have the same dimension as `source_h`.
- Expected: `d_attn` appears configurable.
- Severity: low for default `d_attn=128`; medium if users change `d_attn`.
- Suggested handling: either enforce `d_attn == dim` in `__init__()` or build gate input from `message_bar`.

### 17.6 OT prior and top-k matches are not saved as artifacts

- Files: `scripts/run_mousebrain_v2.py`, `scripts/train_stage_model.py`
- Actual implementation: final embeddings and summaries are saved; `ot_prior` keys are summarized but dense or sparse prior tensors are not saved.
- Expected design did not explicitly require saving OT prior, but validation requested checking whether it is saved.
- Severity: low for training, medium for later OT matching QC.
- Suggested handling: add optional artifact saving for top-k `topk_idx`, `topk_weight`, `confidence`, and `row_mass` if downstream QC needs it.

### 17.7 Generic training script lacks contrast schedule

- File: `scripts/train_stage_model.py`
- Actual implementation: no `--lambda_contrast_schedule`; history lacks `lambda_contrast` and `weighted_crossview_loss`.
- Expected: MouseBrain script supports the schedule; user asked whether training scripts support it.
- Severity: low for MouseBrain because `run_mousebrain_v2.py` supports it; medium for generic training reproducibility.
- Suggested handling: port the schedule parser to `train_stage_model.py` if generic training should match MouseBrain.

## 18. Reproducibility recipe

### 18.1 Environment

The code requires Python packages used in the source:

```text
torch
numpy
scipy
scanpy
anndata
sklearn
PIL
skimage
timm and torchvision if raw HE UNI extraction is used
matplotlib if downstream clustering plots are used
```

For MouseBrain raw image extraction is not used in the current route because HE is built from `obsm["uni_feature"]`.

### 18.2 Generic input structure

Prepare per-section modality inputs as AnnData objects or h5ad paths. Each present AnnData must have spatial coordinates in the configured spatial key, copied or available as `.obsm["spatial"]`.

Build sections:

```text
sections = [
    {
        "section_id": "s1",
        "he_reference_adata_input": path_or_adata_with_UNI_feature,
        "rna_input": path_or_adata,
        "protein_input": path_or_adata,      # for HE+RNA+Protein
        # or
        "metabolite_input": path_or_adata,   # for HE+RNA+Metabolite
        "spatial_key": "spatial",
        "uni_feature_key": "UNI_feature",
    },
    ...
]
```

Run preprocessing:

```python
from model.multimodal_preprocessing import preprocess_multisection_cosie_style

prep = preprocess_multisection_cosie_style(
    sections=sections,
    n_comps=50,
    hvg_num=3000,
    hvg_num_by_modality=None,
    target_sum=None,
    use_harmony=True,
    metacell=False,
)

feature_dict = prep["feature_dict"]
spatial_loc_dict = prep["spatial_loc_dict"]
processed_data_dict = prep["processed_data_dict"]
section_order = prep["section_ids"]
```

Train model:

```python
from model.stage_model import StageMultiModalModel, should_update_ot
from model.configure import get_default_model_config
import torch

config = get_default_model_config()
model = StageMultiModalModel(config=config, feature_dict=feature_dict)
model.initialize_ot_prior(feature_dict, section_order=section_order)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["lr"],
    weight_decay=config["training"]["weight_decay"],
)

for epoch in range(1, config["training"]["epochs"] + 1):
    model.train()
    outputs = model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=processed_data_dict,
        section_order=section_order,
        epoch=epoch,
    )
    loss = outputs["losses"]["total_loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if should_update_ot(epoch, config["uot"]["update_interval"]):
        model.eval()
        with torch.no_grad():
            eval_outputs = model(
                feature_dict=feature_dict,
                spatial_loc_dict=spatial_loc_dict,
                processed_data_dict=processed_data_dict,
                section_order=section_order,
                epoch=epoch,
            )
            model.update_ot_prior(eval_outputs["final_embeddings"], section_order)
```

### 18.3 MouseBrain reproduction

Use config:

```text
/home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json
```

Default short run from JSON:

```bash
cd /home/hujinlan/spa_mo_model
python scripts/run_mousebrain_v2.py \
    --config data/configs/mousebrain_preprocess_train.json
```

Reproduce the existing 200 epoch full-spot warmup style:

```bash
cd /home/hujinlan/spa_mo_model
python scripts/run_mousebrain_v2.py \
    --config data/configs/mousebrain_preprocess_train.json \
    --epochs 200 \
    --device cuda \
    --lambda_contrast_schedule "1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2" \
    --output_dir results/mousebrain_test/fullspot_warmup_schedule
```

Expected output directory from that command shape:

```text
results/mousebrain_test/fullspot_warmup_schedule/epochs_200
```

Expected artifacts:

```text
mousebrain_config_used.json
run_summary.json
loss_history.json
final_embeddings/s1_final_embedding.npy
final_embeddings/s2_final_embedding.npy
final_embeddings/s3_final_embedding.npy
```

Downstream clustering on final embeddings:

```bash
cd /home/hujinlan/spa_mo_model
python scripts/analyze_mousebrain_clustering.py \
    --embedding_dir results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings \
    --config data/configs/mousebrain_preprocess_train.json \
    --output_dir results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering \
    --cluster_list 5,6,8,10
```

### 18.4 Key parameters to reproduce current model

```text
latent_dim = 128
encoder hidden_dims = [256, 128]
encoder output_dim = 128
encoder activation = GELU
encoder dropout = 0.1
encoder norm = LayerNorm
encoder L2 output = True

contrastive method = COSIE crossview
contrastive gamma = 5.0
lambda_contrast default = 0.1
MouseBrain 200 epoch schedule = 1e-4 for epochs 1-5,
                                3e-4 for epochs 6-10,
                                1e-3 for epochs 11-15,
                                1e-2 for epochs 16-200

fusion concat order = HE,RNA,Protein or HE,RNA,Metabolite
fusion input_dim = 384
fusion hidden_dims = [256, 128]
fusion output_dim = 128
fusion residual = mean_residual

spatial KNN K = 5
spatial graph = undirected with self-loops
edge weight = exp(-d^2/(sigma_i^2+delta)), source-normalized

GraphSAGE = one weighted residual layer, 128 -> 128

UOT epsilon_init = 0.08
UOT epsilon_update = 0.05
UOT tau_a = tau_b = 1.0
UOT max_iter = 1000
UOT tol = 1e-6
UOT check_every = 10
UOT update_interval = 20
UOT topk = 10
UOT cost = clipped cosine distance

OT attention direction = forward to next section
OT attention d_attn = 128
OT attention beta = 0.2
OT attention dropout = 0.1
OT attention gate = scalar
OT attention confidence = enabled

decoder hidden_dim = 128
decoder activation = GELU
decoder dropout = 0.1
reconstruction loss = MSE
lambda_reconstruction = 1.0
```

## 19. Compilation / lightweight validation results

Requested compile command:

```bash
python -m py_compile \
    model/configure.py \
    model/data_preprocessing.py \
    model/multimodal_preprocessing.py \
    model/image_preprocessing.py \
    model/utils.py \
    model/loss.py \
    model/model_component.py \
    model/linkage_construction.py \
    model/stage_model.py \
    scripts/run_mousebrain_v2.py \
    scripts/run_stage_model.py
```

Result:

```text
FAILED
```

Failure:

```text
File "scripts/run_stage_model.py", line 4
    from __future__ import annotations
SyntaxError: from __future__ imports must occur at the beginning of the file
```

Cause:

```text
line 2: """Smoke test entry for the V2 stage multimodal model."""
line 3: """... second string literal ..."""
line 4: from __future__ import annotations
```

The second string literal is a statement before the future import.

Follow-up compile command excluding only `scripts/run_stage_model.py`:

```bash
python -m py_compile \
    model/configure.py \
    model/data_preprocessing.py \
    model/multimodal_preprocessing.py \
    model/image_preprocessing.py \
    model/utils.py \
    model/loss.py \
    model/model_component.py \
    model/linkage_construction.py \
    model/stage_model.py \
    scripts/run_mousebrain_v2.py
```

Result:

```text
PASSED
```

No long training was started. No temporary validation scripts were created under `tmp_validation/`.

## 20. Final conclusion

Result:

```text
PARTIAL PASS
```

Core model architecture conclusion:

The implemented `StageMultiModalModel` largely conforms to the intended model design:

- It supports the two expected complete three-modality inputs: `HE+RNA+Protein` and `HE+RNA+Metabolite`.
- It uses independent modality-specific MLP encoders.
- It produces 128-dimensional modality latents.
- It applies COSIE-style within-section cross-view alignment.
- It concatenates three latents to 384 dimensions and projects to a 128-dimensional fused embedding.
- It builds section-local spatial KNN graphs with `K=5`.
- It applies one-layer weighted residual GraphSAGE.
- It computes adjacent-section UOT priors outside backprop.
- It sparsifies UOT priors by top-k.
- It applies one-way OT-guided cross-stage attention.
- It decodes final embeddings back to preprocessed modality features.
- Its total loss consists of reconstruction loss plus weighted alignment/crossview loss.

Main reasons for `PARTIAL PASS` rather than full `PASS`:

- The required compile check fails on `scripts/run_stage_model.py`.
- Several config fields are present but not operational switches.
- Generic training lacks the MouseBrain lambda contrast schedule support.
- OT prior/top-k matches are not saved as artifacts, only used in memory and summarized by keys.

Whether the report is enough to reproduce the current model:

```text
Yes. The architecture, data structures, formulas, default parameters, MouseBrain-specific settings, training loop, update timing, and saved artifacts are documented above.
```

Whether downstream analysis is reasonable to continue:

```text
Yes, with caution. Downstream clustering using saved final_embeddings is consistent with the implemented model. For OT matching QC or reproducibility of exact top-k cross-stage links, add explicit OT prior artifact saving first.
```

Remaining risks:

- `scripts/run_stage_model.py` cannot run until the future-import syntax issue is fixed.
- Non-default config settings such as `d_attn != 128`, `use_momentum=True`, or disabling distance weights may not behave as users expect.
- The crossview loss can be negative, so schedule and lambda values materially affect total loss interpretation.
- UOT updates at epoch 20 multiples depend on final embeddings produced using the previously cached prior.
