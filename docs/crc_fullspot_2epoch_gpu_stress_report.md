# CRC Full-Spot 2-Epoch GPU Stress Report

## 1. Scope

This report records a controlled full-spot CRC Stereo-CITE-seq stress test for the existing RNA+Protein `StageMultiModalModel` path.

Requested stress setting:

- full spots for both CRC sections;
- `max_shared_genes=10000`;
- `epochs=2`;
- `lambda_contrast=0.1`;
- `device=cuda`;
- save outputs, final embeddings, and OT top-k if the run reaches those stages;
- default safety behavior: `--preflight_memory_check` enabled and `--force_full_uot` disabled.

The key goal was to determine whether the current dense-UOT implementation can handle the real CRC spot scale. The run must not fake success if dense UOT is too large.

## 2. Files Changed

Modified:

- `scripts/run_crc_stereocite.py`

Added capabilities:

- `--full_spots`;
- `--save_outputs`;
- `--save_embeddings`;
- `--save_ot_prior_topk`;
- `--preflight_memory_check` / `--no-preflight_memory_check`;
- `--force_full_uot`;
- `--log_memory`;
- dense UOT memory preflight;
- graceful full-spot preflight failure summary;
- final embedding saving when available;
- sparse OT top-k saving when available;
- selected spot index saving;
- spatial array saving when the run reaches model outputs.

Not modified:

- `model/stage_model.py`;
- `model/model_component.py`;
- `model/linkage_construction.py`;
- `model/loss.py`;
- `model/data_preprocessing.py`;
- `model/multimodal_preprocessing.py`;
- `scripts/run_mousebrain_v2.py`.

No UOT, OT attention, GraphSAGE, crossview loss, decoder, or preprocessing formula was changed.

## 3. Compile Check

Command:

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
    scripts/run_crc_stereocite.py
```

Result: PASS.

## 4. Stress Test Command

Command:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_crc_stereocite.py \
    --data_dir /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq \
    --full_spots \
    --max_shared_genes 10000 \
    --train \
    --epochs 2 \
    --lambda_contrast 0.1 \
    --device cuda \
    --save_outputs \
    --save_embeddings \
    --save_ot_prior_topk \
    --preflight_memory_check \
    --output_dir /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_2ep_lc0.1_gpu_stress
```

Result:

```text
FULLSPOT_STRESS_TEST: PRECHECK_FAIL_DENSE_UOT_TOO_LARGE
```

## 5. Full Spot Counts

CRC full spot counts from the raw h5ad files:

- `CRC_003`: 166,279 spots;
- `CRC_006`: 446,095 spots.

The run used full spot indices:

- `selected_spot_indices_CRC_003.npy`: shape `[166279]`, from 0 to 166278;
- `selected_spot_indices_CRC_006.npy`: shape `[446095]`, from 0 to 446094.

## 6. RNA / Protein Data Handling

RNA `var_names_make_unique()` was executed in memory:

- `CRC_003` RNA original shape: `[166279, 28592]`;
- `CRC_006` RNA original shape: `[446095, 31809]`;
- `CRC_003` make-unique gene count: 28,592;
- `CRC_006` make-unique gene count: 31,809;
- both RNA `var_names` were unique after `var_names_make_unique()`.

Shared genes:

- total shared genes after make-unique: 28,430;
- selected shared genes for this stress command: 10,000;
- total artificial suffix shared genes: 19;
- artificial suffix shared genes among selected first 10,000: 0.

ADT / Protein:

- ADT was mapped to `Protein`;
- ADT was not mapped to `Metabolite`;
- ADT marker count: 163;
- ADT `var_names` unique: true for both sections;
- ADT marker sets matched;
- ADT marker order matched.

No HE was fabricated. No Metabolite was fabricated. No duplicate count aggregation was performed.

## 7. RNA/ADT Alignment

`CRC_003`:

- RNA/ADT `obs_names` matched;
- `obsm["spatial"]` existed;
- spatial shape: `[166279, 2]`;
- RNA/ADT spatial coordinates matched;
- spatial x range: `[0.0, 515.0]`;
- spatial y range: `[0.0, 572.0]`.

`CRC_006`:

- RNA/ADT `obs_names` matched;
- `obsm["spatial"]` existed;
- spatial shape: `[446095, 2]`;
- RNA/ADT spatial coordinates matched;
- spatial x range: `[0.0, 798.0]`;
- spatial y range: `[0.0, 884.0]`.

## 8. Dense UOT Preflight

The current UOT implementation builds dense cost/kernel/coupling tensors. For adjacent full CRC sections:

- `n_source`: 166,279;
- `n_target`: 446,095;
- dense entries: 74,176,230,505;
- one dense float32 matrix: 276.328 GiB;
- estimated dense UOT core memory for cost + kernel + coupling: 828.984 GiB.

Available memory recorded by the stress command:

- available CPU memory: 55.756 GiB;
- available CUDA memory: null;
- total CUDA memory: null;
- `torch.cuda.is_available()` during verification: false.

The command requested `--device cuda`, but CUDA was not visible to this cosie environment at runtime. Independently, the dense UOT core estimate is far beyond the available CPU memory and would also exceed common GPU memory sizes. Because `--force_full_uot` was not set, the script correctly stopped before preprocessing and before initial OT prior construction.

## 9. Stage Progress

Completed:

- `read_raw_h5ad`;
- in-memory RNA `var_names_make_unique`;
- shared gene alignment;
- ADT marker validation;
- RNA/ADT `obs_names` and spatial validation;
- dense UOT preflight;
- lightweight metadata saving.

Not reached:

- COSIE preprocessing;
- `feature_dict` generation;
- `StageMultiModalModel` initialization;
- initial OT prior;
- epoch 1 forward;
- epoch 1 backward;
- epoch 1 optimizer step;
- epoch 2 forward/backward;
- dynamic OT update;
- final embedding saving;
- OT top-k saving.

This is expected and correct for the default non-forced full-spot stress test because dense UOT is too large.

## 10. Saved Outputs

Output directory:

```text
/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_2ep_lc0.1_gpu_stress
```

Saved files:

- `run_summary.json`;
- `shared_gene_symbols_make_unique.txt`;
- `adt_marker_list.txt`;
- `duplicate_gene_summary_CRC_003.csv`;
- `duplicate_gene_summary_CRC_006.csv`;
- `selected_spot_indices_CRC_003.npy`;
- `selected_spot_indices_CRC_006.npy`.

Not saved:

- processed h5ad files;
- full feature matrices;
- final embeddings;
- dense OT coupling;
- fake OT top-k.

OT top-k status:

- `ot_prior_topk_saved`: false;
- reason: dense UOT preflight failed before OT prior initialization.

## 11. Loss History

No training epoch was run, so no `loss_history.json` was generated.

Reason: the full dense UOT preflight failed before COSIE preprocessing and model initialization.

## 12. Interpretation

This stress test shows that the current dense-UOT model path cannot be honestly run on full CRC_003 x CRC_006 spot scale as-is.

The blocker is not the RNA+Protein modality support itself. The blocker is the dense adjacent-section UOT prior:

```text
166279 x 446095 = 74176230505 entries
```

One float32 dense matrix would require about 276.3 GiB, and the minimal dense UOT core estimate for cost + kernel + coupling is about 829.0 GiB. This excludes extra memory for gradients, temporary tensors, preprocessing, model activations, attention, sparse top-k outputs, and Python/AnnData overhead.

## 13. Final Conclusion

```text
FULLSPOT_STRESS_TEST: PRECHECK_FAIL_DENSE_UOT_TOO_LARGE
```

The script now supports full-spot stress-test preflight and controlled output saving. The requested full-spot 2-epoch GPU training did not proceed because the current dense UOT implementation is not feasible at CRC full spot scale without a different OT strategy, such as blockwise/streaming OT, candidate-restricted OT, approximate nearest-neighbor OT, or an explicitly downsampled/metacell stress path.
