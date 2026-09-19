# Model Validation Targeted Fix Report

Generated on 2026-06-08 for `/home/hujinlan/spa_mo_model`.

## 1. Scope

This targeted fix only handled the three requested items:

- A1: Re-check and validate `scripts/run_stage_model.py` py_compile status.
- B1: Add a defensive `d_attn == dim` guard in `OTGuidedAttention`.
- B2: Add optional saving of sparse OT prior / top-k matches for MouseBrain OT matching QC.

No model architecture, loss formula, UOT computation logic, preprocessing logic, contrastive loss, or long training workflow was changed.

## 2. A1 run_stage_model.py compile check

Checked the file header with:

```bash
sed -n '1,40p' /home/hujinlan/spa_mo_model/scripts/run_stage_model.py
```

The beginning of the file is now:

```text
#!/usr/bin/env python3
"""Smoke test entry for the V2 stage multimodal model."""

from __future__ import annotations
```

There is only the allowed shebang, module docstring, and blank line before `from __future__ import annotations`. The previous second string literal is gone.

Full compile check passed:

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

Result:

```text
PASS
```

The future import issue is resolved.

## 3. B1 OTGuidedAttention d_attn guard

Modified file:

```text
model/model_component.py
```

Change made in `OTGuidedAttention.__init__()`:

```python
if d_attn != dim:
    raise ValueError(
        "OTGuidedAttention currently requires d_attn == dim because the scalar gate "
        "uses [source_h, message, source_h - message, source_h * message]. "
        "Please keep d_attn equal to dim unless the gate design is changed."
    )
```

Reason:

The scalar gate still uses:

```python
[source_h, message, source_h - message, source_h * message]
```

so `message.shape[-1]` must match `source_h.shape[-1]`. This requires `d_attn == dim`.

Validation:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -c \
"from model.configure import get_default_model_config; from model.stage_model import StageMultiModalModel; StageMultiModalModel(config=get_default_model_config()); print('default_model_init: PASS')"
```

Result:

```text
default_model_init: PASS
```

Default config remains valid because:

```text
dim = 128
d_attn = 128
```

Attention formula was not changed. Gate input was not changed. Residual update was not changed. Default `d_attn=128` was not changed.

## 4. B2 OT prior top-k saving

Modified file:

```text
scripts/run_mousebrain_v2.py
```

New CLI parameters:

```text
--save_ot_prior_topk
--ot_prior_output_dir
```

Default behavior:

- `--save_ot_prior_topk` defaults to `False`.
- Existing runs without this flag do not save OT prior top-k files.
- Training loss, final embeddings, UOT computation, OT attention, and preprocessing are unchanged.

When enabled, the script saves sparse top-k prior files after the dry-run forward or after training final eval.

Default output directory:

```text
{output_dir}/ot_prior_topk
```

Optional override:

```text
--ot_prior_output_dir /custom/path
```

Files saved per adjacent pair, for example `s1_to_s2`:

```text
s1_to_s2_topk_idx.npy
s1_to_s2_topk_weight.npy
s1_to_s2_confidence.npy
s1_to_s2_row_mass.npy
s1_to_s2_metadata.json
```

Metadata contains:

```json
{
  "source_section": "s1",
  "target_section": "s2",
  "topk": 10,
  "n_source": 2384,
  "n_target": 2820,
  "modalities_used": ["HE", "RNA", "Metabolite"],
  "has_dense_P": false,
  "run_mode": "dry_run",
  "note": "Saved sparse top-k UOT prior from model.ot_prior after final evaluation."
}
```

Dense P is not saved. If `P_dense` exists in memory in a future config, this targeted change still only records `has_dense_P` in metadata and does not write dense P to disk.

`run_summary.json` now records:

```json
{
  "saved_ot_prior_topk": true,
  "ot_prior_topk_dir": ".../ot_prior_topk",
  "ot_prior_topk_files": {
    "s1_to_s2": {
      "topk_idx": "...",
      "topk_weight": "...",
      "confidence": "...",
      "row_mass": "...",
      "metadata": "..."
    }
  }
}
```

If the flag is not enabled, `run_summary.json` records:

```json
{
  "saved_ot_prior_topk": false
}
```

Small path handling note:

The script now avoids appending a duplicate mode suffix when an explicit `--output_dir` already ends with `dry_run` or the expected `epochs_N` name. This was needed so the requested dry-run test path resolves to:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run
```

rather than `dry_run/dry_run`. Default output behavior for ordinary runs is unchanged.

## 5. Validation commands

### 5.1 Full py_compile

Command:

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

Result:

```text
PASS
```

### 5.2 Default model init

The system `python` environment lacked `scanpy`, so default initialization was validated with the project `cosie` conda environment:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -c \
"from model.configure import get_default_model_config; from model.stage_model import StageMultiModalModel; StageMultiModalModel(config=get_default_model_config()); print('default_model_init: PASS')"
```

Result:

```text
default_model_init: PASS
```

### 5.3 Dry-run + save_ot_prior_topk

CUDA was not visible in the current `cosie` environment:

```text
cuda_available False
```

Therefore the dry-run validation used CPU, as permitted by the task:

```bash
cd /home/hujinlan/spa_mo_model

/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --dry_run \
    --device cpu \
    --save_ot_prior_topk \
    --output_dir /home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run
```

Result:

```text
MOUSEBRAIN_DRY_RUN: PASS
```

Saved OT prior directory:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run/ot_prior_topk
```

Files present:

```text
s1_to_s2_topk_idx.npy
s1_to_s2_topk_weight.npy
s1_to_s2_confidence.npy
s1_to_s2_row_mass.npy
s1_to_s2_metadata.json
s2_to_s3_topk_idx.npy
s2_to_s3_topk_weight.npy
s2_to_s3_confidence.npy
s2_to_s3_row_mass.npy
s2_to_s3_metadata.json
```

### 5.4 Shape check

Command:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python - <<'PY'
import numpy as np, json, pathlib
root = pathlib.Path("/home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run/ot_prior_topk")
for pair in ["s1_to_s2", "s2_to_s3"]:
    idx = np.load(root / f"{pair}_topk_idx.npy")
    w = np.load(root / f"{pair}_topk_weight.npy")
    c = np.load(root / f"{pair}_confidence.npy")
    r = np.load(root / f"{pair}_row_mass.npy")
    meta = json.load(open(root / f"{pair}_metadata.json"))
    print(pair, "idx", idx.shape, "weight", w.shape, "confidence", c.shape, "row_mass", r.shape, "meta", meta)
PY
```

Result:

```text
s1_to_s2 idx (2384, 10) weight (2384, 10) confidence (2384,) row_mass (2384,)
s2_to_s3 idx (2820, 10) weight (2820, 10) confidence (2820,) row_mass (2820,)
```

Metadata:

```text
s1_to_s2:
    n_source = 2384
    n_target = 2820
    modalities_used = ["HE", "RNA", "Metabolite"]
    has_dense_P = False
    run_mode = dry_run

s2_to_s3:
    n_source = 2820
    n_target = 2662
    modalities_used = ["HE", "RNA", "Metabolite"]
    has_dense_P = False
    run_mode = dry_run
```

`run_summary.json` check:

```text
saved_ot_prior_topk = True
ot_prior_topk_dir = /home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run/ot_prior_topk
pairs = ["s1_to_s2", "s2_to_s3"]
```

## 6. Files modified

Source files modified:

```text
model/model_component.py
scripts/run_mousebrain_v2.py
```

Report created:

```text
docs/model_validation_targeted_fix_report.md
```

No other model, loss, preprocessing, UOT computation, or training logic files were modified.

## 7. Temporary / test outputs

Test output directory created:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test/dry_run
```

This is a validation output only. It can be deleted after review. Deleting it will not affect the core project model code or existing 200 epoch MouseBrain results.

Cleanup command:

```bash
rm -rf /home/hujinlan/spa_mo_model/results/mousebrain_test/ot_prior_save_test
```

## 8. Final conclusion

```text
PASS
```

A1 is confirmed fixed and full py_compile passes.

B1 is fixed by adding a defensive `d_attn == dim` guard while preserving the current OT-guided attention formula.

B2 is fixed by adding optional sparse OT prior / top-k match saving behind `--save_ot_prior_topk`; the dry-run save test passed and produced the expected files and shapes.
