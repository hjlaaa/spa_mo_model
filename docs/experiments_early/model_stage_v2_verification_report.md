# Model Stage V2 Verification Report

## 1. Verification scope

Checked core implementation files:

- `/home/hujinlan/spa_mo_model/model/configure.py`
- `/home/hujinlan/spa_mo_model/model/utils.py`
- `/home/hujinlan/spa_mo_model/model/loss.py`
- `/home/hujinlan/spa_mo_model/model/model_component.py`
- `/home/hujinlan/spa_mo_model/model/linkage_construction.py`
- `/home/hujinlan/spa_mo_model/model/stage_model.py`
- `/home/hujinlan/spa_mo_model/model/__init__.py`
- `/home/hujinlan/spa_mo_model/scripts/run_stage_model.py`

Checked design/reference documents:

- `/home/hujinlan/spa_mo_model/docs/cosie_contrastive_learning_report.md`
- `/home/hujinlan/spa_mo_model/docs/COSIE 中切片内跨模态对比学习方法总结.pdf`
- `/home/hujinlan/spa_mo_model/docs/model_stage_v1_verification_report.md`

No additional UOT / GraphSAGE / OT-guided attention design document was found under `docs/`.

Model directory check found only COSIE-like files:

```text
__init__.py
configure.py
data_preprocessing.py
image_preprocessing.py
linkage_construction.py
loss.py
model_component.py
multimodal_preprocessing.py
stage_model.py
utils.py
```

Forbidden split files were not found: `graph_construction.py`, `ot_attention.py`, `uot.py`, `decoder.py`, `trainer.py`, `model_config.py`, `fusion.py`, `modality_encoder.py`, `cosie_alignment.py`.

## 2. Design boundary check

V1 preservation:

- Preserved `feature_dict` and `spatial_loc_dict` model inputs.
- `processed_data_dict` is not consumed by `StageMultiModalModel.forward`; it is carried by the training script bundle path but not part of the core model forward signature.
- `ModalityMLPEncoder` is present and used.
- Each modality encoder outputs 128-d latent tensors.
- COSIE-style `crossview_contrastive_Loss` is preserved.
- No InfoNCE implementation was found.
- No CLIP-style contrastive loss was found.
- No `[N, N]` spot-level similarity matrix was found.
- No explicit same-spot positive pair construction was found.
- No explicit different-spot negative pair construction was found.
- Three modality latents are concatenated to 384-d inside `FusionMLP`.
- `FusionMLP` outputs 128-d fused embeddings.

V2 implementation:

- FusionMLP output `Z [N,128]` enters `WeightedResidualGraphSAGE`.
- GraphSAGE is one weighted residual layer.
- GraphSAGE input and output dimensions are 128.
- Spatial KNN graph uses `K=5` from `graph.knn_neighbors_spatial`.
- Spatial graph is constructed independently for each section/stage.
- Spatial graph is undirected and includes self-loops.
- Spatial distance weights are computed and source-row normalized.
- GraphSAGE uses GELU, Dropout 0.1, LayerNorm, residual.
- GraphSAGE does not store previous epoch `H`; each forward starts from current `Z`.
- UOT / unbalanced Sinkhorn is implemented.
- UOT uses uniform mass by default.
- Initial UOT uses `epsilon_init=0.08`.
- Dynamic UOT uses `epsilon_update=0.05`.
- `tau_a=tau_b=1.0`.
- Default `max_iter=1000`, `tol=1e-6`.
- UOT functions are decorated with `@torch.no_grad()` and detach inputs.
- UOT is not included as a loss.
- Initial OT uses preprocessed single-modality embeddings.
- Initial OT computes per-modality couplings, total-mass normalizes each, then averages.
- Initial OT uses global z-score per modality across sections.
- Dynamic OT uses final embeddings via detached tensors.
- `should_update_ot(epoch, 20)` supports 20-epoch update cadence.
- Momentum update is not used.
- Coupling matrices are total-mass normalized.
- Couplings are top-k sparsified.
- Top-k default is 10.
- `topk_idx`, `topk_weight`, and `confidence` are stored.
- Confidence uses dense row mass divided by uniform source mass, clamped to 1.
- OT-guided attention is single direction from `S_s` to `S_{s+1}`.
- Last section does not receive cross-section attention and keeps GraphSAGE output.
- Attention score is `QK/sqrt(d) + beta * log(topk_weight + delta)`.
- Beta default is 0.2.
- Attention softmax is limited to top-k candidates.
- Scalar gate is implemented.
- Gate uses confidence, residual, Dropout, and LayerNorm.
- `h_tilde = m` is not used.
- `h_tilde = H + m` without gate/confidence is not used.
- Per-modality decoders are implemented.
- Decoder reconstructs preprocessed modality embeddings from `H_tilde [N,128]`.
- Decoder structure is `Linear(128,128) -> GELU -> Dropout(0.1) -> Linear(128, original_dim)`.
- Reconstruction target is `feature_dict[section][modality]`, not raw HE images, raw counts, or raw expression.
- Total loss is `lambda_reconstruction * reconstruction_loss + lambda_contrast * crossview_loss`.
- No OT loss was found.
- No spatial smooth loss was found.
- No gate regularization was found.
- No triplet linkage loss was found.
- No `Prediction_mlp` was implemented.
- No `GraphAutoencoder` was implemented.
- No missing modality prediction was implemented.
- No temporal Transformer was implemented.
- No downstream task was implemented.

Strict deviations are listed in Section 8.

## 3. Configuration check

Actual default config values in `model/configure.py`:

- `graphsage.enabled = True`
- `graphsage.input_dim = 128`
- `graphsage.output_dim = 128`
- `graphsage.num_layers = 1`
- `graphsage.dropout = 0.1`
- `graphsage.activation = "GELU"`
- `graphsage.norm = "LayerNorm"`
- `graphsage.residual = True`
- `graphsage.use_distance_weight = True`
- `uot.enabled = True`
- `uot.epsilon_init = 0.08`
- `uot.epsilon_update = 0.05`
- `uot.tau_a = 1.0`
- `uot.tau_b = 1.0`
- `uot.max_iter = 1000`
- `uot.tol = 1e-6`
- `uot.update_interval = 20`
- `uot.topk = 10`
- `uot.use_momentum = False`
- `uot.normalize_total_mass = True`
- `ot_attention.enabled = True`
- `ot_attention.direction = "forward"`
- `ot_attention.d_attn = 128`
- `ot_attention.beta = 0.2`
- `ot_attention.gate = "scalar"`
- `ot_attention.use_confidence = True`
- `ot_attention.residual = True`
- `ot_attention.norm = "LayerNorm"`
- `decoder.enabled = True`
- `decoder.hidden_dim = 128`
- `decoder.activation = "GELU"`
- `decoder.dropout = 0.1`
- `reconstruction.enabled = True`
- `reconstruction.loss = "mse"`
- `loss.lambda_contrast = 0.1`
- `loss.lambda_reconstruction = 1.0`
- `loss.use_ot_loss = False`
- `loss.use_spatial_smooth_loss = False`
- `loss.use_gate_regularization = False`

## 4. Function/class existence check

`utils.py`:

- `compute_spatial_knn_graph_with_weights(...)`: present, returns `[2,E] edge_index` and `[E] edge_weight`.
- `l2_normalize(...)`: present.
- `cosine_cost_matrix(...)`: present, computes clipped `1 - cosine` cost.

`linkage_construction.py`:

- `unbalanced_sinkhorn(...)`: present.
- `normalize_coupling_total_mass(...)`: present.
- `sparsify_coupling_topk(...)`: present.
- `compute_initial_multimodal_uot_prior(...)`: present.
- `update_uot_prior_from_embeddings(...)`: present.

`model_component.py`:

- `ModalityMLPEncoder`: present.
- `FusionMLP`: present.
- `WeightedResidualGraphSAGE`: present.
- `OTGuidedAttention`: present.
- `ModalityDecoder`: present.

`loss.py`:

- `compute_joint(...)`: present, COSIE-style dimension-level joint matrix.
- `crossview_contrastive_Loss(...)`: present, COSIE-style cross-view loss.
- `compute_pairwise_cosie_crossview_loss(...)`: present.
- `compute_reconstruction_loss(...)`: present, MSE against preprocessed feature tensors.

`stage_model.py`:

- `StageMultiModalModel.initialize_ot_prior(...)`: present.
- `StageMultiModalModel.update_ot_prior(...)`: present.
- `StageMultiModalModel.forward(...)`: present.
- `should_update_ot(...)`: present.

`StageMultiModalModel.forward(...)` returns:

- `fused_embeddings`
- `graphsage_embeddings`
- `final_embeddings`
- `latent_dict`
- `reconstructions`
- `spatial_graph_dict`
- `ot_prior`
- `losses`
- `loss_details`
- `messages`

## 5. Shape and numerical tests

Temporary verification script:

```text
/home/hujinlan/spa_mo_model/tmp_verification/verify_model_stage_v2.py
```

HE + RNA + Protein two-stage test:

- `latent_dict["s1"][mod]`: `[80,128]`
- `fused_embeddings["s1"]`: `[80,128]`
- `graphsage_embeddings["s1"]`: `[80,128]`
- `final_embeddings["s1"]`: `[80,128]`
- `reconstructions["s1"]["HE"]`: `[80,50]`
- `reconstructions["s1"]["RNA"]`: `[80,50]`
- `reconstructions["s1"]["Protein"]`: `[80,20]`
- `topk_idx`: `[80,10]`
- `topk_weight`: `[80,10]`
- `confidence`: `[80]`
- `total_loss`, `crossview_loss`, and `reconstruction_loss` were finite.
- `loss.backward()` ran successfully.

HE + RNA + Metabolite two-stage test:

- `latent_dict["s1"][mod]`: `[70,128]`
- `fused_embeddings["s1"]`: `[70,128]`
- `graphsage_embeddings["s1"]`: `[70,128]`
- `final_embeddings["s1"]`: `[70,128]`
- `topk_idx`: `[70,10]`
- `topk_weight`: `[70,10]`
- `confidence`: `[70]`
- `total_loss`, `crossview_loss`, and `reconstruction_loss` were finite.
- `loss.backward()` ran successfully.

Three-stage attention test:

- `ot_prior` contains `("s1", "s2")`.
- `ot_prior` contains `("s2", "s3")`.
- `ot_prior` does not contain any `("s3", ...)` key.
- `final_embeddings["s3"]` equals `graphsage_embeddings["s3"]` within tolerance.

OT update test:

- `initialize_ot_prior(...)` ran successfully.
- `update_ot_prior(...)` ran under `torch.no_grad()`.
- Updated `topk_idx` shape: `[30,10]`.
- Updated `topk_weight.requires_grad = False`.
- `should_update_ot(20, 20) = True`.
- `should_update_ot(19, 20) = False`.

Primitive graph/UOT checks:

- `compute_spatial_knn_graph_with_weights` returned `edge_index [2,120]`, `edge_weight [120]` for a 16-node test graph.
- Edge weights were source-row normalized to approximately 1.
- `cosine_cost_matrix` returned clipped `[0,2]` costs.
- `unbalanced_sinkhorn` returned finite coupling.
- `sparsify_coupling_topk` returned normalized top-k weights.

## 6. Error handling tests

All expected error cases raised clear errors:

- Missing third modality: `ValueError`, supported sets listed.
- Simultaneous Protein and Metabolite: `ValueError`, supported sets listed; four-modality input is not silently accepted.
- Missing spatial coordinates: `KeyError`, missing section named.
- Spot count mismatch across modalities: `ValueError`, mismatched modality and counts named.
- Bad `section_order`: `KeyError`, missing section named.

## 7. Commands executed

Core compile:

```bash
cd /home/hujinlan/spa_mo_model
python -m py_compile \
    model/configure.py \
    model/utils.py \
    model/loss.py \
    model/model_component.py \
    model/linkage_construction.py \
    model/stage_model.py \
    scripts/run_stage_model.py
```

Result: passed with no output.

Smoke test:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_stage_model.py --smoke_test
```

Key output:

```text
HE_RNA_Protein_two_stage: PASS
HE_RNA_Metabolite_two_stage: PASS
three_stage_forward_attention: PASS
```

Temporary verification script compile:

```bash
cd /home/hujinlan/spa_mo_model
python -m py_compile tmp_verification/verify_model_stage_v2.py
```

Result: passed with no output.

Strict verification script:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python tmp_verification/verify_model_stage_v2.py
```

Key output:

```text
MODEL_STAGE_V2_VERIFICATION: PASS_NUMERICAL_TESTS
```

## 8. Issues found

### Issue 1: OT gate input uses projected message instead of raw message

- Location: `/home/hujinlan/spa_mo_model/model/model_component.py`, `OTGuidedAttention.forward`, lines 285-289.
- Current code computes `message = sum_j alpha_ij v_j`, then `message_bar = W_O(message)`, and uses `gate_input = concat(source_h, message_bar, source_h - message_bar, source_h * message_bar)`.
- Design requested `gate_input = concat(source_h, m, source_h - m, source_h * m)`, followed by `m_bar = W_O(m)` for the residual update.
- Impact: no runtime failure; all shape and gradient tests pass. However, under strict reading, the gate is conditioned on projected message `m_bar` instead of raw attention message `m`, so this is not 100% identical to the written design.
- Suggested fix: change the gate input to use `message` and keep `message_bar` only for the residual update. Because `d_attn=128` equals `dim=128`, this is a small local change.
- Fixed this round: no. Core implementation was not modified during verification.

### Issue 2: `processed_data_dict` is not a direct `StageMultiModalModel.forward` input

- Location: `/home/hujinlan/spa_mo_model/model/stage_model.py`, `StageMultiModalModel.forward`, lines 297-303.
- Current signature accepts `feature_dict`, `spatial_loc_dict`, `section_order`, and `epoch`.
- The broader pipeline/training script can load or carry `processed_data_dict`, but the core model forward does not accept or use it.
- Impact: no runtime failure for the V2 model because V2 computations only require `feature_dict` and `spatial_loc_dict`. However, the verification checklist explicitly said to confirm receiving `feature_dict / spatial_loc_dict / processed_data_dict`, so this is a strict interface mismatch.
- Suggested fix: either document `processed_data_dict` as unused by V2 model stage, or add an optional `processed_data_dict=None` argument to `forward` for interface compatibility without changing behavior.
- Fixed this round: no. Core implementation was not modified during verification.

## 9. Temporary verification files

- `/home/hujinlan/spa_mo_model/tmp_verification/verify_model_stage_v2.py`
  - Purpose: strict temporary V2 verification script.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/__pycache__/verify_model_stage_v2.cpython-313.pyc`
  - Purpose: Python bytecode cache from compiling/running the temporary verification script.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/stage_training_smoke/stage_model_v2_last.pt`
  - Purpose: previous training smoke-test checkpoint.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/stage_training_smoke/training_history.json`
  - Purpose: previous training smoke-test log.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/stage_training_smoke/training_summary.json`
  - Purpose: previous training smoke-test summary.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/stage_training_smoke/final_embeddings/s1_final_embedding.npy`
  - Purpose: previous training smoke-test embedding output.
  - Can delete: yes.
  - Deleting affects project runtime: no.

- `/home/hujinlan/spa_mo_model/tmp_verification/stage_training_smoke/final_embeddings/s2_final_embedding.npy`
  - Purpose: previous training smoke-test embedding output.
  - Can delete: yes.
  - Deleting affects project runtime: no.

This report itself is documentation only:

- `/home/hujinlan/spa_mo_model/docs/model_stage_v2_verification_report.md`
  - Purpose: verification record.
  - Can delete: yes, if you want a cleaner project.
  - Deleting affects project runtime: no.

## 10. Cleanup command

To remove temporary verification artifacts:

```bash
rm -rf /home/hujinlan/spa_mo_model/tmp_verification
```

Optional, if you also want to remove this verification record:

```bash
rm -f /home/hujinlan/spa_mo_model/docs/model_stage_v2_verification_report.md
```

## 11. Final conclusion

PARTIAL PASS.

The current Model Stage V2 implementation is runnable and passes all numerical, shape, gradient, OT update, smoke, and error-handling tests. It preserves the V1 MLP encoder + COSIE-style cross-view loss + FusionMLP path and adds weighted residual GraphSAGE, UOT top-k prior, single-direction OT-guided attention, decoders, and reconstruction loss without adding forbidden losses or modules.

However, it is not 100% identical to the written design because:

1. `OTGuidedAttention` gates on `message_bar = W_O(m)` instead of raw `m`.
2. `StageMultiModalModel.forward` does not directly accept `processed_data_dict`.

Recommendation: address these two small interface/logic deviations before declaring a full PASS and entering the next modeling stage.
