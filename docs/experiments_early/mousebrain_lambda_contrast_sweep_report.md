# MouseBrain Lambda Contrast Sweep Report

## 1. Experiment status

Base config:

```text
/home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json
```

Both MouseBrain small-sample experiments completed successfully with first 500 spots per section and 25 epochs.

| Experiment | lambda_contrast | Output directory | Status |
|---|---:|---|---|
| A | 1e-3 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/lambda_sweep/lambda_1e-3_epochs_25_maxspots_500/maxspots_500/epochs_25` | PASS |
| B | 1e-4 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/lambda_sweep/lambda_1e-4_epochs_25_maxspots_500/maxspots_500/epochs_25` | PASS |

Note: `scripts/run_mousebrain_v2.py` was given a minimal `--lambda_contrast` override so these runs could change only `loss.lambda_contrast`. No V2 model core files were changed for this sweep.

## 2. Loss history files

Checked files:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test/lambda_sweep/lambda_1e-3_epochs_25_maxspots_500/maxspots_500/epochs_25/loss_history.json
/home/hujinlan/spa_mo_model/results/mousebrain_test/lambda_sweep/lambda_1e-4_epochs_25_maxspots_500/maxspots_500/epochs_25/loss_history.json
```

Available fields in both files:

```text
epoch
total_loss
crossview_loss
reconstruction_loss
```

`weighted_crossview_loss` below was computed as:

```text
lambda_contrast * crossview_loss
```

## 3. Lambda 1e-3 loss statistics

| Loss | First | Last | Min | Max |
|---|---:|---:|---:|---:|
| total_loss | -554.636353 | 52.599300 | -554.636353 | 63.720085 |
| reconstruction_loss | 69.073517 | 53.291161 | 53.291161 | 69.073517 |
| crossview_loss | -623709.812500 | -691.859924 | -623709.812500 | -691.859924 |
| weighted_crossview_loss | -623.709813 | -0.691860 | -623.709813 | -0.691860 |

All values are finite. Reconstruction loss decreased from `69.073517` to `53.291161`, a relative change of about `-22.85%`.

The first epoch is dominated by weighted crossview loss: `-623.709813` versus reconstruction `69.073517`. After the first few epochs, the weighted crossview term quickly shrinks in magnitude and becomes much smaller than reconstruction loss.

## 4. Lambda 1e-4 loss statistics

| Loss | First | Last | Min | Max |
|---|---:|---:|---:|---:|
| total_loss | 58.956703 | 53.153912 | 53.153912 | 67.812408 |
| reconstruction_loss | 69.004440 | 53.199284 | 53.199284 | 69.004440 |
| crossview_loss | -100477.359375 | -453.723145 | -100477.359375 | -453.723145 |
| weighted_crossview_loss | -10.047736 | -0.045372 | -10.047736 | -0.045372 |

All values are finite. Reconstruction loss decreased from `69.004440` to `53.199284`, a relative change of about `-22.90%`.

The weighted crossview term starts at `-10.047736`, which is much less dominant than the `1e-3` run. After early training, it becomes tiny relative to reconstruction loss.

## 5. Epoch-20 OT update check

For `lambda_contrast = 1e-3`:

| Epoch | total_loss | reconstruction_loss | crossview_loss | weighted_crossview_loss |
|---:|---:|---:|---:|---:|
| 18 | 56.807438 | 57.556694 | -749.255981 | -0.749256 |
| 19 | 56.194309 | 56.931244 | -736.934265 | -0.736934 |
| 20 | 55.527210 | 56.251007 | -723.797363 | -0.723797 |
| 21 | 54.964672 | 55.677814 | -713.141724 | -0.713142 |
| 22 | 54.362473 | 55.067699 | -705.226746 | -0.705227 |

Epoch 20 to 21 relative change:

```text
total_loss: -1.01%
reconstruction_loss: -1.02%
crossview_loss magnitude decreases slightly
```

For `lambda_contrast = 1e-4`:

| Epoch | total_loss | reconstruction_loss | crossview_loss | weighted_crossview_loss |
|---:|---:|---:|---:|---:|
| 18 | 57.367100 | 57.413296 | -461.970886 | -0.046197 |
| 19 | 56.743355 | 56.789421 | -460.679199 | -0.046068 |
| 20 | 56.156631 | 56.202515 | -458.819275 | -0.045882 |
| 21 | 55.431171 | 55.477024 | -458.520691 | -0.045852 |
| 22 | 54.872646 | 54.918297 | -456.514984 | -0.045651 |

Epoch 20 to 21 relative change:

```text
total_loss: -1.29%
reconstruction_loss: -1.29%
crossview_loss nearly unchanged
```

In both experiments, the epoch 20 OT prior update did not cause loss explosion. Loss remains finite and continues decreasing after epoch 20.

## 6. Loss balance comparison

`lambda_contrast = 1e-3`:

- First epoch weighted crossview loss is much larger in magnitude than reconstruction loss.
- Later epochs are stable, but the first update is strongly driven by the crossview objective.
- This setting is usable for short tests, but it is still aggressive at initialization.

`lambda_contrast = 1e-4`:

- First epoch weighted crossview loss is about `-10.05` versus reconstruction around `69.00`.
- After early training, weighted crossview is around `-0.05`, so total loss mostly tracks reconstruction.
- This setting is more conservative and better balanced for current MouseBrain small-scale testing.

Neither setting shows NaN, Inf, or instability. The main difference is loss balance: `1e-4` avoids the very large initial contrastive contribution seen at `1e-3`.

## 7. Recommendation

For the next MouseBrain small-sample runs, `lambda_contrast = 1e-4` looks healthier because it keeps the COSIE-style crossview term present without dominating the first epoch.

Recommended next step:

```text
Continue with lambda_contrast = 1e-4 for additional small-scale stability checks before considering larger spot counts.
```

If cross-modal alignment appears too weak in later embedding-level diagnostics, a middle value such as `3e-4` can be tested, but `1e-3` should be treated as a more aggressive setting.

## 8. Temporary files

No temporary verification files were created.

No existing results were deleted.

## 9. Final conclusion

PASS for the requested MouseBrain lambda contrast small-sample sweep.

Both `1e-3` and `1e-4` completed 25 epochs with finite losses and stable epoch 20 OT updates. `1e-4` is the healthier setting for the current training scale because its weighted crossview loss is less dominant and total loss more cleanly follows reconstruction loss.
