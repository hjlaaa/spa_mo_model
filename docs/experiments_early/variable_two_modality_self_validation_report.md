# Variable Two-Modality Self-Validation Report

## 1. Scope

This report is a follow-up self-validation for the variable two-modality support
added to `StageMultiModalModel`.

Reviewed first:

- `docs/variable_two_modality_support_report.md`
- `model/configure.py`
- `model/model_component.py`
- `model/stage_model.py`
- `model/linkage_construction.py`
- `model/loss.py`
- `scripts/run_stage_model.py`
- `scripts/run_mousebrain_v2.py`

Goal: verify that legal 2-3 observed modality sets work, illegal modality sets
fail clearly, CRC `RNA+Protein` can run on a small real subset, and MouseBrain
`HE+RNA+Metabolite` remains intact.

## 2. Further code changes in this validation round

One validation-script change was made:

- `scripts/run_stage_model.py`
  - Added positive synthetic checks for `HE+RNA` and `HE+Protein`.
  - Added a `HE=None` check to verify `None` is ignored rather than filled.
  - Added negative checks for single-modality, four-modality, unsupported
    three-modality, inconsistent section modalities, within-section spot count
    mismatch, and cross-section feature dimension mismatch.
  - Added explicit fusion input dim checks.

No model code was changed in this round. No UOT, OT attention, crossview loss,
GraphSAGE, decoder, preprocessing, or training logic was changed.

## 3. Supported modality sets

Confirmed default `model.valid_modality_sets`:

```python
["HE", "RNA"]
["HE", "Protein"]
["HE", "Metabolite"]
["RNA", "Protein"]
["RNA", "Metabolite"]
["Protein", "Metabolite"]
["HE", "RNA", "Protein"]
["HE", "RNA", "Metabolite"]
```

Confirmed not supported:

- four-modality `HE+RNA+Protein+Metabolite`;
- `HE+Protein+Metabolite`;
- `RNA+Protein+Metabolite`.

## 4. Code audit results

Confirmed:

- observed modalities are parsed from non-`None` `feature_dict[section]`
  entries;
- no modality is auto-filled;
- all sections in one initialization / forward / initial OT prior run must use
  the same modality order;
- canonical ordering is `HE`, `RNA`, `Protein`, `Metabolite`;
- `FusionMLP` computes `input_dim = latent_dim * len(modality_order)`;
- `RNA+Protein` uses fusion input dim `256`;
- `HE+RNA` uses fusion input dim `256`;
- `HE+Protein` uses fusion input dim `256`;
- `HE+RNA+Metabolite` uses fusion input dim `384`;
- `crossview_contrastive_Loss()` and `compute_joint()` were not changed;
- `unbalanced_sinkhorn()`, `sparsify_coupling_topk()`, dynamic OT update, and
  `OTGuidedAttention` formulas were not changed;
- the `OTGuidedAttention` `d_attn == dim` guard is still present;
- `scripts/run_mousebrain_v2.py` still contains `--save_ot_prior_topk` and the
  sparse top-k saving function;
- no fake HE is created for CRC-style `RNA+Protein`;
- CRC ADT is treated as `Protein`, not `Metabolite`.

## 5. Positive synthetic tests

Command:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_stage_model.py
```

Result: PASS.

### RNA + Protein

Input:

```text
s1 RNA      [100, 50]
s1 Protein  [100, 50]
s2 RNA      [120, 50]
s2 Protein  [120, 50]
```

Observed:

```text
modality_order: ['RNA', 'Protein']
fusion_input_dim: 256
s1 final_embeddings: (100, 128)
s2 final_embeddings: (120, 128)
topk_idx: (100, 10)
topk_weight: (100, 10)
confidence: (100,)
reconstruction keys: RNA, Protein only
ot_prior modalities_used: RNA, Protein only
```

### HE + RNA + Metabolite

Input:

```text
s1 HE          [100, 50]
s1 RNA         [100, 50]
s1 Metabolite  [100, 50]
s2 HE          [120, 50]
s2 RNA         [120, 50]
s2 Metabolite  [120, 50]
```

Observed:

```text
modality_order: ['HE', 'RNA', 'Metabolite']
fusion_input_dim: 384
s1 final_embeddings: (100, 128)
s2 final_embeddings: (120, 128)
topk_idx: (100, 10)
topk_weight: (100, 10)
confidence: (100,)
reconstruction keys: HE, RNA, Metabolite
ot_prior modalities_used: HE, RNA, Metabolite only
```

### HE + RNA

Input:

```text
s1 HE   [80, 50]
s1 RNA  [80, 50]
s2 HE   [90, 50]
s2 RNA  [90, 50]
```

Observed:

```text
modality_order: ['HE', 'RNA']
fusion_input_dim: 256
s1 final_embeddings: (80, 128)
s2 final_embeddings: (90, 128)
forward: PASS
```

### HE + Protein

Input:

```text
s1 HE       [80, 50]
s1 Protein  [80, 50]
s2 HE       [90, 50]
s2 Protein  [90, 50]
```

Observed:

```text
modality_order: ['HE', 'Protein']
fusion_input_dim: 256
s1 final_embeddings: (80, 128)
s2 final_embeddings: (90, 128)
forward: PASS
```

## 6. Negative / invalid input tests

All required invalid cases produced `ValueError` with clear messages.

| Case | Expected | Result |
|---|---|---|
| `RNA` only | at least two modalities required | PASS |
| `HE+RNA+Protein+Metabolite` | unsupported modality set | PASS |
| `RNA+Protein+Metabolite` | unsupported modality set | PASS |
| `s1 RNA+Protein`, `s2 HE+RNA+Protein` | sections must use same modality set | PASS |
| `s1 RNA [100,50]`, `s1 Protein [99,50]` | spot count mismatch | PASS |
| `s1 RNA [100,50]`, `s2 RNA [120,60]` | feature dimension mismatch | PASS |

Representative messages:

```text
Each section must contain at least two real observed modalities; got ['RNA'].
Each section must contain exactly one supported two- or three-modality set.
All sections in one model run must use the same observed modality set.
All modalities in s1 must have the same spot count.
Modality RNA has inconsistent input dimensions: 50 vs 60.
```

## 7. None modality behavior

Tested:

```text
s1: HE=None, RNA [100,50], Protein [100,50]
s2: HE=None, RNA [120,50], Protein [120,50]
```

Observed:

```text
modality_order: ['RNA', 'Protein']
fusion_input_dim: 256
s1 final_embeddings: (100, 128)
s2 final_embeddings: (120, 128)
reconstruction keys: RNA, Protein only
ot_prior modalities_used: RNA, Protein only
```

Conclusion: `None` is ignored and not filled. No fake HE is created.

## 8. Real CRC subset validation

Path checked:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq
```

Files read:

```text
CRC_003_bin20/adata_RNA.h5ad
CRC_003_bin20/adata_ADT.h5ad
CRC_006_bin20/adata_RNA.h5ad
CRC_006_bin20/adata_ADT.h5ad
```

Subset strategy:

- first 500 spots from each CRC sample;
- RNA temporary smoke features: first 50 variables only;
- Protein temporary smoke features: all 163 ADT markers;
- spatial coordinates: first 500 rows from `obsm["spatial"]`;
- ADT mapped to `Protein`;
- no HE field was created;
- no full CRC training;
- no full `166279 x 446095` dense UOT matrix.

Result: PASS.

Observed:

```text
CRC_003 RNA full shape: (166279, 28592)
CRC_003 ADT full shape: (166279, 163)
CRC_003 subset RNA: (500, 50)
CRC_003 subset Protein: (500, 163)
CRC_003 subset spatial: (500, 2)
CRC_003 first 500 obs_names aligned: True

CRC_006 RNA full shape: (446095, 31809)
CRC_006 ADT full shape: (446095, 163)
CRC_006 subset RNA: (500, 50)
CRC_006 subset Protein: (500, 163)
CRC_006 subset spatial: (500, 2)
CRC_006 first 500 obs_names aligned: True

resolved_modality_order: ['RNA', 'Protein']
final_embeddings['CRC_003']: (500, 128)
final_embeddings['CRC_006']: (500, 128)
reconstruction keys: Protein, RNA
ot_prior modalities_used: ['RNA', 'Protein']
topk_idx: (500, 10)
total_loss_finite: True
```

AnnData warned that RNA `var_names` are not unique. This was only observed and
not modified.

## 9. MouseBrain regression dry run

Command:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --dry_run \
    --device cpu \
    --output_dir /home/hujinlan/spa_mo_model/results/mousebrain_test/variable_modality_regression_dry_run
```

Result: PASS.

Observed:

```text
MOUSEBRAIN_DRY_RUN: PASS
s1 final_embeddings: (2384, 128)
s2 final_embeddings: (2820, 128)
s3 final_embeddings: (2662, 128)
s1 reconstructions: HE, RNA, Metabolite
s2 reconstructions: HE, RNA, Metabolite
s3 reconstructions: HE, RNA, Metabolite
ot_prior_keys: s1->s2, s2->s3
loss_finite: true
saved_ot_prior_topk: false
```

New dry-run artifacts were written only under:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/variable_modality_regression_dry_run/dry_run
```

Existing 200 epoch result directories were not modified.

## 10. Compile validation

System Python:

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

Result: PASS.

Cosie environment:

```bash
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

## 11. Preservation checklist

Confirmed:

- UOT formula unchanged.
- Top-k sparsification unchanged.
- Dynamic OT update unchanged.
- OT-guided attention formula unchanged.
- `OTGuidedAttention` `d_attn == dim` guard preserved.
- COSIE-style `compute_joint()` unchanged.
- COSIE-style `crossview_contrastive_Loss()` unchanged.
- GraphSAGE formula unchanged.
- Decoder structure unchanged.
- `--save_ot_prior_topk` preserved.
- No fake HE created.
- CRC ADT used as `Protein`, not `Metabolite`.
- Four-modality input not supported.
- MouseBrain `HE+RNA+Metabolite` path still works.

## 12. Final conclusion

PASS.

The variable two-modality support is functioning as intended. Positive legal
2-3 modality inputs pass, illegal modality inputs fail clearly, `None` modality
entries are ignored rather than filled, a real CRC `RNA+Protein` 500-spot subset
forwards successfully, and the original MouseBrain `HE+RNA+Metabolite` dry run
still passes.
