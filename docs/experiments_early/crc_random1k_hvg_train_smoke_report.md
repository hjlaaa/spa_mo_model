# CRC Random 1k HVG Training Smoke Report

## 1. Scope

This check extended `scripts/run_crc_stereocite.py` to support small-scale CRC RNA+Protein training, then ran:

- a more realistic dry-run with 1000 random spots per section, 10000 shared genes, `hvg_num=3000`;
- a 25 epoch small training smoke run with `lambda_contrast=1e-2`.

No formal full CRC training was run. No original h5ad file was overwritten. No processed h5ad file or dense UOT matrix was saved.

## 2. Code Change

Modified file:

- `scripts/run_crc_stereocite.py`

Minimal additions:

- added `--spot_sampling {first,random}`;
- added `--train`;
- added `--epochs`;
- added `--lambda_contrast`;
- added `--lr`;
- added `--weight_decay`;
- added `--update_interval`;
- added `--log_every`;
- added a small Adam training loop for subset smoke tests;
- added `loss_history.json` saving in train mode;
- recorded both initial UOT prior modalities and final UOT prior modalities.

Core model code was not changed in this round. UOT, OT attention, GraphSAGE, decoder, and crossview loss formulas were not changed.

## 3. Compilation

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
    scripts/run_mousebrain_v2.py \
    scripts/run_stage_model.py \
    scripts/run_crc_stereocite.py
```

Result: PASS.

## 4. Realistic Dry-Run

Command:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_crc_stereocite.py \
    --data_dir /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq \
    --max_spots_per_section 1000 \
    --max_shared_genes 10000 \
    --spot_sampling random \
    --dry_run \
    --device cpu \
    --output_dir /home/hujinlan/spa_mo_model/results/crc_stereocite/dry_run_random1k_genes10k_hvg3k
```

Result: PASS.

Observed preprocessing:

- RNA HVG/PCA/Harmony path was triggered with `hvg_num=3000`.
- Protein PCA/Harmony path was also triggered.
- Harmony for RNA ran 10 iterations and stopped before convergence, without failing.
- Harmony for Protein converged after 2 iterations.

Key dry-run outputs:

- `resolved_modality_order`: `["RNA", "Protein"]`
- `feature_dict_shapes`:
  - `CRC_003/RNA`: `[1000, 50]`
  - `CRC_003/Protein`: `[1000, 50]`
  - `CRC_006/RNA`: `[1000, 50]`
  - `CRC_006/Protein`: `[1000, 50]`
- `final_embedding_shapes`:
  - `CRC_003`: `[1000, 128]`
  - `CRC_006`: `[1000, 128]`
- `reconstruction_keys`: only `RNA` and `Protein`
- `ot_prior_modalities_used`: `["RNA", "Protein"]`
- `total_loss_finite`: true

Output directory:

```text
/home/hujinlan/spa_mo_model/results/crc_stereocite/dry_run_random1k_genes10k_hvg3k
```

Saved files are only lightweight summaries/lists/CSVs. No `.h5ad` file was saved.

## 5. Small Training Smoke Run

Command:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_crc_stereocite.py \
    --data_dir /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq \
    --max_spots_per_section 1000 \
    --max_shared_genes 10000 \
    --spot_sampling random \
    --train \
    --epochs 25 \
    --lambda_contrast 1e-2 \
    --device cpu \
    --output_dir /home/hujinlan/spa_mo_model/results/crc_stereocite/train_random1k_genes10k_hvg3k_25ep_lc1e-2
```

Result: PASS.

Training settings:

- spots per section: 1000 random spots;
- shared genes used before preprocessing: 10000;
- `hvg_num`: 3000;
- `lambda_contrast`: `0.01`;
- epochs: 25;
- optimizer: Adam;
- learning rate: `0.001`;
- weight decay: `0.0`;
- OT update interval: 20.

Key training outputs:

- `resolved_modality_order`: `["RNA", "Protein"]`
- `feature_dict_shapes`: all section/modality feature matrices are `[1000, 50]`
- `final_embedding_shapes`:
  - `CRC_003`: `[1000, 128]`
  - `CRC_006`: `[1000, 128]`
- `reconstruction_keys`: only `RNA` and `Protein`
- initial UOT prior modalities: `["RNA", "Protein"]`
- final UOT prior modalities after dynamic update: `["final_embedding"]`
- OT prior was dynamically updated at epoch 20.
- `total_loss_finite`: true
- `loss_history.json` contains 25 epoch records.

Loss checkpoints:

| epoch | total_loss | crossview_loss | reconstruction_loss | weighted_crossview_loss |
|---:|---:|---:|---:|---:|
| 1 | -193.253159 | -20542.027344 | 12.167116 | -205.420273 |
| 25 | 9.258380 | -101.494484 | 10.273325 | -1.014945 |

Final eval losses:

- `total_loss`: `8.780326843261719`
- `crossview_loss`: `-101.56360626220703`
- `reconstruction_loss`: `9.795963287353516`

Output directory:

```text
/home/hujinlan/spa_mo_model/results/crc_stereocite/train_random1k_genes10k_hvg3k_25ep_lc1e-2
```

Saved files:

- `run_summary.json`
- `loss_history.json`
- `shared_gene_symbols_make_unique.txt`
- `adt_marker_list.txt`
- `duplicate_gene_summary_CRC_003.csv`
- `duplicate_gene_summary_CRC_006.csv`

No `.h5ad` file was saved.

## 6. Data Handling Confirmations

- CRC RNA `var_names_make_unique()` is still performed in memory.
- No duplicate-gene count aggregation was performed.
- ADT is used as `Protein`.
- ADT is not used as `Metabolite`.
- HE was not fabricated.
- Metabolite was not fabricated.
- Full CRC dense UOT was not constructed.
- Original CRC h5ad files were not overwritten.

## 7. Conclusion

PASS.

The CRC integrated pipeline now supports a realistic random-subset dry-run and a small-scale train mode. With 1000 random spots per section, 10000 shared genes, `hvg_num=3000`, and 25 epochs at `lambda_contrast=1e-2`, the RNA+Protein model path runs successfully and produces finite losses, valid `[1000, 128]` final embeddings, RNA/Protein-only reconstructions, and expected OT prior behavior.
