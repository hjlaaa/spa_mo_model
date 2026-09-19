# MouseBrain Loss History Check Report

## 1. Checked files

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_5/loss_history.json`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_25/loss_history.json`

Reference report read:

- `/home/hujinlan/spa_mo_model/docs/mousebrain_v2_run_report.md`

## 2. Available loss fields

Both JSON files contain:

- `epoch`
- `total_loss`
- `crossview_loss`
- `reconstruction_loss`

## 3. Five-epoch training loss check

All loss values are finite. No NaN or Inf was found.

`total_loss`:

- first value: `-96055.90625`
- last value: `-158.94406127929688`
- min: `-96055.90625`
- max: `-158.94406127929688`
- relative change: `0.9983452963228974`

`reconstruction_loss`:

- first value: `69.08438110351562`
- last value: `66.4221420288086`
- min: `66.4221420288086`
- max: `69.08438110351562`
- relative change: `-0.038536048701340295`

`crossview_loss`:

- first value: `-961249.875`
- last value: `-2253.662109375`
- min: `-961249.875`
- max: `-2253.662109375`
- relative change: `0.9976554877477878`

Interpretation:

- `reconstruction_loss` decreases smoothly by about 3.85%, which is a normal sign.
- `crossview_loss` remains finite but changes from a very large negative value toward a much smaller negative value.
- Because `total_loss = reconstruction_loss + 0.1 * crossview_loss`, `total_loss` is dominated by the large negative crossview term at epoch 1.
- Numerically, `total_loss` increases from very negative toward less negative, so it does not show a standard decreasing loss curve.
- There is no NaN, Inf, or abrupt instability in the 5-epoch run.

## 4. Twenty-five-epoch training loss check

All loss values are finite. No NaN or Inf was found.

`total_loss`:

- first value: `-52844.09375`
- last value: `-18.978973388671875`
- min: `-52844.09375`
- max: `-18.978973388671875`
- relative change: `0.9996408496760592`

`reconstruction_loss`:

- first value: `69.01834106445312`
- last value: `53.76976013183594`
- min: `53.76976013183594`
- max: `69.01834106445312`
- relative change: `-0.220935199215774`

`crossview_loss`:

- first value: `-529131.125`
- last value: `-727.4873046875`
- min: `-529131.125`
- max: `-727.4873046875`
- relative change: `0.9986251284978038`

Interpretation:

- `reconstruction_loss` decreases steadily by about 22.09%, which is a healthy optimization signal.
- `crossview_loss` remains finite throughout but rapidly moves from a very large negative magnitude toward a smaller negative magnitude.
- `total_loss` is therefore not monotonic in the usual decreasing direction; it moves from very negative toward less negative because the crossview term dominates the signed total objective.
- After the first few epochs, the losses become much smoother and remain stable.

## 5. Epoch-20 OT update check

Epoch 18-22 from the 25-epoch run:

| epoch | total_loss | reconstruction_loss | crossview_loss |
| ---: | ---: | ---: | ---: |
| 18 | `-22.456741333007812` | `57.786170959472656` | `-802.4291381835938` |
| 19 | `-22.30255889892578` | `57.22605895996094` | `-795.2861328125` |
| 20 | `-21.884967803955078` | `56.53763961791992` | `-784.22607421875` |
| 21 | `-20.585067749023438` | `56.04962921142578` | `-766.346923828125` |
| 22 | `-20.27165985107422` | `55.375450134277344` | `-756.4710693359375` |

Epoch 20 to 21 relative changes:

- `total_loss`: `0.05939693704720558`
- `reconstruction_loss`: `-0.008631602058241268`
- `crossview_loss`: `0.022798464598918496`

Interpretation:

- Epoch 20 before OT update is stable.
- Epoch 21 after OT update does not show loss explosion.
- `reconstruction_loss` continues decreasing after OT update.
- `crossview_loss` changes modestly after OT update.
- The OT update is stable in this small-scale test.

## 6. Loss balance check

The loss components are finite, but their scales are not balanced early in training.

At epoch 1 of the 5-epoch run:

- `reconstruction_loss = 69.08438110351562`
- `0.1 * crossview_loss = -96124.9875`
- `total_loss = -96055.90625`

At epoch 1 of the 25-epoch run:

- `reconstruction_loss = 69.01834106445312`
- `0.1 * crossview_loss = -52913.1125`
- `total_loss = -52844.09375`

So `total_loss` is initially dominated by the signed crossview term. This does not cause numerical instability, but it makes `total_loss` hard to interpret as a conventional monotonically decreasing training curve.

By epoch 25:

- `reconstruction_loss = 53.76976013183594`
- `0.1 * crossview_loss = -72.74873046875`
- `total_loss = -18.978973388671875`

The balance becomes less extreme later, but the initial scale mismatch is real. Current `lambda_contrast = 0.1` may still be too large for this MouseBrain small-sample setup if the goal is an interpretable total objective. A smaller `lambda_contrast` should be considered in a later tuning step, but no code or training change was made in this check.

## 7. Figures

No figures were generated because the user explicitly requested text-only loss inspection and no plotting.

## 8. Conclusion

PARTIAL PASS.

Reasons:

- All `total_loss`, `reconstruction_loss`, and `crossview_loss` values are finite.
- No NaN or Inf was found.
- `reconstruction_loss` decreases in both 5-epoch and 25-epoch runs.
- Epoch 20 OT update is stable; epoch 21 does not show loss explosion.
- Training remains numerically stable after OT update.

Why not full PASS:

- `total_loss` is strongly dominated by the signed crossview term early in training.
- `total_loss` does not behave like a conventional decreasing loss curve; it moves from very negative toward less negative.
- The component scale suggests `lambda_contrast` may need tuning before treating the total objective as well-balanced.
