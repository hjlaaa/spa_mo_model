# Variable Two-Modality Support Report

## 1. Scope

This targeted change generalized the existing `StageMultiModalModel` from fixed
three-modality inputs to legal 2-3 observed modality inputs. The work was kept
minimal and focused on:

- `model/configure.py`
- `model/model_component.py`
- `model/stage_model.py`
- `scripts/run_stage_model.py`

No separate CRC model was added. No core loss formula, UOT formula,
OT-guided attention formula, GraphSAGE formula, decoder structure, or
MouseBrain preprocessing path was changed.

## 2. Files modified

- `model/configure.py`
  - Expanded `model.valid_modality_sets`.
- `model/model_component.py`
  - Relaxed `FusionMLP` from exactly three modalities to two or three observed
    modalities.
- `model/stage_model.py`
  - Added canonical observed-modality resolution from non-`None` feature entries.
  - Enforced one consistent modality set across all sections in one model run.
  - Limited initial UOT prior construction to the run's real observed modalities.
- `scripts/run_stage_model.py`
  - Replaced the old fixed-three-modality smoke coverage with two synthetic
    checks: `RNA+Protein` and `HE+RNA+Metabolite`.
  - Made the script run smoke tests by default while keeping `--smoke_test`
    compatible.

## 3. Supported modality sets

The default config now allows these two-modality sets:

```python
["HE", "RNA"]
["HE", "Protein"]
["HE", "Metabolite"]
["RNA", "Protein"]
["RNA", "Metabolite"]
["Protein", "Metabolite"]
```

The original MouseBrain-compatible three-modality sets remain:

```python
["HE", "RNA", "Protein"]
["HE", "RNA", "Metabolite"]
```

Four-modality input is not supported. Additional three-modality sets such as
`["HE", "Protein", "Metabolite"]` and `["RNA", "Protein", "Metabolite"]`
were not added.

## 4. Modality parsing behavior

`StageMultiModalModel` now treats the real observed modalities for each section
as:

```python
present_modalities = {
    modality for modality, value in feature_dict[section].items()
    if value is not None
}
```

The resolved order is canonical:

```python
["HE", "RNA", "Protein", "Metabolite"]
```

Rules now enforced:

- fewer than two real modalities raises `ValueError`;
- unsupported modality names raise `ValueError`;
- unsupported combinations raise `ValueError`;
- all sections in one initialization / forward / initial OT prior run must use
  the same resolved modality set;
- no missing modality prediction is performed;
- no `None` modality is auto-filled;
- no fake HE modality is created.

For CRC, the intended resolved order is:

```python
["RNA", "Protein"]
```

## 5. FusionMLP behavior

`FusionMLP` still uses the original fusion formula:

```text
Z_concat = concat([Z_m for m in modality_order])
Z_mlp = MLP(Z_concat)
Z_mean = mean([Z_m for m in modality_order])
Z = LayerNorm(Z_mlp + Z_mean)
```

Only the concat input width is now determined by the number of observed
modalities:

```python
fusion_input_dim = latent_dim * len(modality_order)
```

Therefore:

- `HE+RNA+Metabolite`: `384 -> 128`
- `HE+RNA+Protein`: `384 -> 128`
- `RNA+Protein`: `256 -> 128`
- `HE+RNA`: `256 -> 128`

The existing hidden dims, activation, dropout, LayerNorm, and mean residual
logic were kept unchanged. The legacy config field `fusion.input_dim` is not
read by the current model code; each `FusionMLP` module computes its own input
dimension from `modality_order`.

## 6. Crossview loss

`model/loss.py` was not changed.

The existing COSIE-style pairwise crossview loss continues to run over all
observed modality pairs in one section:

- `RNA+Protein`: one pair, `RNA__Protein`;
- `HE+RNA+Metabolite`: three pairs, `HE__RNA`, `HE__Metabolite`,
  `RNA__Metabolite`.

No InfoNCE, VICReg, Barlow Twins, temperature scaling, positive-pair mining, or
loss reweighting was introduced.

## 7. Decoder and reconstruction

The decoder structure was not changed.

Because `target_feature_dict[section]` is now populated only from the resolved
real modality order, reconstruction only uses real observed modalities:

- CRC-style `RNA+Protein`: only `RNA` and `Protein` decoders/targets are used.
- MouseBrain-style `HE+RNA+Metabolite`: `HE`, `RNA`, and `Metabolite` are used.

No fake HE decoder target is created for CRC.

## 8. UOT compatibility

`unbalanced_sinkhorn()`, top-k sparsification, dynamic OT update, and
`OTGuidedAttention` were not changed.

Initial UOT prior construction now receives the resolved run modality order
instead of the full supported-modality list. For CRC-style input this means:

```text
P_init = normalize_mass(mean(P_RNA, P_Protein))
```

For MouseBrain-style input this remains:

```text
P_init = normalize_mass(mean(P_HE, P_RNA, P_Metabolite))
```

The existing `d_attn == dim` guard in `OTGuidedAttention` was preserved.
The existing `--save_ot_prior_topk` behavior in `scripts/run_mousebrain_v2.py`
was not changed.

## 9. CRC data path check

Checked:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq
```

Detected files:

```text
CRC_003_bin20/adata_RNA.h5ad
CRC_003_bin20/adata_ADT.h5ad
CRC_006_bin20/adata_RNA.h5ad
CRC_006_bin20/adata_ADT.h5ad
```

Metadata-only backed AnnData inspection:

| Sample | RNA shape | ADT/Protein shape | obs_names aligned | common spatial keys |
|---|---:|---:|---|---|
| `CRC_003_bin20` | `(166279, 28592)` | `(166279, 163)` | True | `spatial`, `spatial_original` |
| `CRC_006_bin20` | `(446095, 31809)` | `(446095, 163)` | True | `original_spatial`, `spatial` |

ADT is treated as `Protein`, not `Metabolite`. No HE modality was found or
fabricated. No CRC full training was run. No dense full cross-sample UOT cost
matrix was built.

Note: AnnData emitted warnings that RNA `var_names` are not unique. This report
does not modify CRC preprocessing.

## 10. Validation commands

System Python compile check:

```bash
cd /home/hujinlan/spa_mo_model
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

Result: PASS.

Cosie environment compile check:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python -m py_compile \
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

Result: PASS.

Synthetic smoke test:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_stage_model.py
```

Result: PASS.

## 11. Smoke test details

### RNA + Protein

Synthetic input:

```text
s1 RNA      [100, 50]
s1 Protein  [100, 50]
s2 RNA      [120, 50]
s2 Protein  [120, 50]
```

Result:

```text
RNA_Protein_two_modality: PASS
modality_order: ['RNA', 'Protein']
s1 final_embeddings shape: (100, 128)
s2 final_embeddings shape: (120, 128)
topk_idx shape: (100, 10)
topk_weight shape: (100, 10)
confidence shape: (100,)
```

The reconstruction output contained only `RNA` and `Protein`; it did not contain
`HE` or `Metabolite`.

### HE + RNA + Metabolite

Synthetic input:

```text
s1 HE          [100, 50]
s1 RNA         [100, 50]
s1 Metabolite  [100, 50]
s2 HE          [120, 50]
s2 RNA         [120, 50]
s2 Metabolite  [120, 50]
```

Result:

```text
HE_RNA_Metabolite_three_modality: PASS
modality_order: ['HE', 'RNA', 'Metabolite']
s1 final_embeddings shape: (100, 128)
s2 final_embeddings shape: (120, 128)
topk_idx shape: (100, 10)
topk_weight shape: (100, 10)
confidence shape: (100,)
```

This confirms the MouseBrain-like three-modality path still forwards.

## 12. Formula preservation checklist

- UOT formula: unchanged.
- Top-k sparsification formula: unchanged.
- OT-guided attention formula: unchanged.
- `OTGuidedAttention` `d_attn == dim` guard: unchanged.
- COSIE-style `crossview_contrastive_Loss()`: unchanged.
- `compute_joint()`: unchanged.
- GraphSAGE formula: unchanged.
- Decoder structure: unchanged.
- Reconstruction loss formula: unchanged.
- MouseBrain `HE+RNA+Metabolite` valid set: preserved.
- `--save_ot_prior_topk`: unchanged.

## 13. Not done

- No CRC full training was run.
- No CRC full dense UOT matrix was built.
- No CRC-specific independent model or large data pipeline was added.
- No missing-modality model was implemented.
- No four-modality support was added.
- No `HE+Protein+Metabolite` or `RNA+Protein+Metabolite` support was added.
- No preprocessing changes were made for duplicate RNA `var_names`.

## 14. Final conclusion

PASS.

The model now supports legal 2-3 observed modality sets, including CRC-style
`RNA+Protein`, while preserving the existing MouseBrain `HE+RNA+Metabolite`
path. The synthetic `RNA+Protein` and `HE+RNA+Metabolite` forwards both passed,
both compile checks passed, ADT is treated as `Protein`, and no fake HE modality
was introduced.
