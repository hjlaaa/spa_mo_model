# MouseBrain Full-spot Warm-up Feasibility Report

## 1. GPU availability

GPU is available on the host, but it is not visible inside the default Codex sandbox.

Default sandbox check:

```text
torch 2.4.0
torch.version.cuda 12.1
cuda_available False
device_count 0
Can't initialize NVML
```

Host / elevated check:

```text
NVIDIA-SMI 535.183.01
Driver Version: 535.183.01
CUDA Version: 12.2
GPU 0: NVIDIA GeForce RTX 4090
Memory: 24564 MiB
```

PyTorch CUDA check outside the restricted sandbox:

```text
torch 2.4.0
torch.version.cuda 12.1
cuda_available True
device_count 1
device_name NVIDIA GeForce RTX 4090
allocated 0
reserved 0
```

Interpretation:

```text
The workstation has a usable RTX 4090 GPU. The earlier CUDA failure was caused by the default execution sandbox not exposing GPU device nodes, not by missing hardware.
```

## 2. Script changes

Modified file:

```text
/home/hujinlan/spa_mo_model/scripts/run_mousebrain_v2.py
```

The change was limited to training-script compatibility:

- Added `--device` to override `config["training"]["device"]`.
- Added `--lambda_contrast_schedule`.
- Added per-epoch schedule resolution in the training loop.
- Added `lambda_contrast` and `weighted_crossview_loss` to `loss_history.json`.

No V2 model core files were modified. The following model components were not changed:

```text
StageMultiModalModel
ModalityMLPEncoder
FusionMLP
WeightedResidualGraphSAGE
OTGuidedAttention
ModalityDecoder
COSIE-style crossview loss
UOT implementation
```

Script compile check passed:

```bash
cd /home/hujinlan/spa_mo_model
python -m py_compile scripts/run_mousebrain_v2.py
```

## 3. Full-spot GPU dry run

Command run:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --dry_run \
    --device cuda \
    --output_dir /home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_dry_run
```

Actual output directory:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_dry_run/dry_run
```

Result:

```text
MOUSEBRAIN_DRY_RUN: PASS
loss_finite: true
device: cuda
OOM: no
```

Final embedding shapes:

| Section | Shape |
|---|---:|
| s1 | 2384 x 128 |
| s2 | 2820 x 128 |
| s3 | 2662 x 128 |

Dry-run losses:

| Loss | Value |
|---|---:|
| total_loss | -124140.671875 |
| reconstruction_loss | 61.178108 |
| crossview_loss | -1242018.500000 |

## 4. Full-spot GPU 2 epoch feasibility training

Command run:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --epochs 2 \
    --device cuda \
    --lambda_contrast_schedule "1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2" \
    --output_dir /home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility
```

Actual output directory:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility/epochs_2
```

Result:

```text
MOUSEBRAIN_TRAINING: PASS
loss_finite: true
device: cuda
OOM: no
```

Loss history:

| Epoch | lambda_contrast | total_loss | reconstruction_loss | crossview_loss | weighted_crossview_loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 1e-4 | -229.774353 | 61.190304 | -2909646.750000 | -290.964675 |
| 2 | 1e-4 | 55.458954 | 60.379692 | -49207.367188 | -4.920737 |

All recorded loss values are finite.

`loss_history.json` contains:

```text
epoch
lambda_contrast
total_loss
crossview_loss
reconstruction_loss
weighted_crossview_loss
```

Final embedding files:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility/epochs_2/final_embeddings/s1_final_embedding.npy
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility/epochs_2/final_embeddings/s2_final_embedding.npy
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility/epochs_2/final_embeddings/s3_final_embedding.npy
```

Saved embedding shapes:

| Section | Shape |
|---|---:|
| s1 | 2384 x 128 |
| s2 | 2820 x 128 |
| s3 | 2662 x 128 |

## 5. CPU fallback checks

Before confirming GPU access outside the sandbox, CPU full-spot dry run and CPU 2 epoch feasibility were also run successfully. These are kept only as fallback feasibility records.

CPU result artifacts:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/dry_run/dry_run
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_2_feasibility/epochs_2
```

They are not required for the final GPU training command.

## 6. Feasibility interpretation

The full-spot MouseBrain preprocessing, forward pass, UOT prior construction, OT-guided attention, decoder reconstruction, and 2 epoch training loop all completed on GPU.

This confirms:

- GPU is usable when the command is run outside the restricted sandbox.
- Full-spot data shapes are compatible with the V2 model.
- No OOM occurred during full-spot dry run or 2 epoch training.
- Loss values remained finite.
- Per-epoch `lambda_contrast` and `weighted_crossview_loss` are correctly written to `loss_history.json`.

The 2 epoch run did not reach epoch 20, so dynamic OT prior update was not exercised in this full-spot GPU preflight. The requested preflight was intended to detect immediate full-spot runtime/OOM issues before launching the long 200 epoch run.

## 7. Final 200 epoch training command

Recommended command:

```bash
cd /home/hujinlan/spa_mo_model

/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --epochs 200 \
    --device cuda \
    --lambda_contrast_schedule "1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2" \
    --output_dir /home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200
```

Important note:

```text
Run this from a normal terminal/session where nvidia-smi works. The default Codex sandbox may not expose GPU devices.
```

## 8. Temporary files

No temporary verification files were created.

The dry run and 2 epoch feasibility outputs are result artifacts, not temporary scratch files:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_dry_run/dry_run
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/gpu_epochs_2_feasibility/epochs_2
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/dry_run/dry_run
/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_2_feasibility/epochs_2
```

They can be kept for auditability. Deleting them would not affect model code.

## 9. Final conclusion

PASS.

The host has an NVIDIA GeForce RTX 4090, the `cosie` PyTorch environment can use CUDA outside the restricted sandbox, and full-spot MouseBrain GPU dry run plus 2 epoch GPU feasibility training completed successfully with finite losses and correct final embedding shapes.

It is reasonable to start the 200 epoch full-spot GPU training manually using the command above. I did not start the 200 epoch run.
