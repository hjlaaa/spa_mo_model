# MouseBrain V2 Run Report

## 1. Dataset paths

SectionA / `s1`:

- RNA: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionA/adata_RNA.h5ad`
- Metabolite: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionA/adata_meta.h5ad`

SectionB / `s2`:

- RNA: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionB/adata_RNA.h5ad`
- Metabolite: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionB/adata_meta.h5ad`

SectionC / `s3`:

- RNA: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionC/adata_RNA.h5ad`
- Metabolite: `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionC/adata_meta.h5ad`

Config:

- `/home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json`

## 2. Data adapter decisions

- HE uses precomputed `obsm["uni_feature"]` from the RNA AnnData.
- Raw HE JPG images are not used.
- Raw HE JPG + mask / white superpixel / patch extraction is not used.
- RNA `var_names` are set from `var["gene_ids"]`.
- RNA `hvg_num = 3000`.
- Metabolite uses `adata_meta.h5ad`, a 1538-feature `mz-*` matrix.
- Metabolite `hvg_num = None`; HVG is skipped for Metabolite.
- Metabolite still follows the COSIE generic non-Protein branch after HVG skipping: normalize/log1p/scale/PCA.
- `use_harmony = True`.
- Protein is not used for MouseBrain.

## 3. Alignment checks

Full dataset dry run:

| section | RNA spots | Metabolite spots | HE spots | obs_names match | spatial match | uni_feature shape | metabolite feature shape |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| s1 | 2384 | 2384 | 2384 | True | True | 2384 x 2048 | 2384 x 1538 |
| s2 | 2820 | 2820 | 2820 | True | True | 2820 x 2048 | 2820 x 1538 |
| s3 | 2662 | 2662 | 2662 | True | True | 2662 x 2048 | 2662 x 1538 |

Small-scale training tests used synchronized first-500-spot subsets for HE, RNA, Metabolite, spatial coordinates, and obs_names.

## 4. Preprocessing outputs

Full dry run with `use_harmony=True`:

| section | HE | RNA | Metabolite | spatial |
| --- | --- | --- | --- | --- |
| s1 | 2384 x 50 | 2384 x 50 | 2384 x 50 | 2384 x 2 |
| s2 | 2820 x 50 | 2820 x 50 | 2820 x 50 | 2820 x 2 |
| s3 | 2662 x 50 | 2662 x 50 | 2662 x 50 | 2662 x 2 |

`processed_data_dict` was generated.

500-spot training tests:

| section | HE | RNA | Metabolite | spatial |
| --- | --- | --- | --- | --- |
| s1 | 500 x 50 | 500 x 50 | 500 x 50 | 500 x 2 |
| s2 | 500 x 50 | 500 x 50 | 500 x 50 | 500 x 2 |
| s3 | 500 x 50 | 500 x 50 | 500 x 50 | 500 x 2 |

## 5. Dry run results

Command:

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --dry_run
```

Result: `MOUSEBRAIN_DRY_RUN: PASS`

Final embeddings:

- `s1`: `2384 x 128`
- `s2`: `2820 x 128`
- `s3`: `2662 x 128`

OT prior keys:

- `("s1", "s2")`
- `("s2", "s3")`

Loss was finite.

Saved outputs:

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/dry_run/run_summary.json`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/dry_run/final_embeddings/s1_final_embedding.npy`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/dry_run/final_embeddings/s2_final_embedding.npy`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/dry_run/final_embeddings/s3_final_embedding.npy`

## 6. Training test results

5 epoch, first 500 spots per section:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --epochs 5 \
    --max_spots_per_section 500
```

Result: `MOUSEBRAIN_TRAINING: PASS`

- Final embeddings: `s1/s2/s3 = 500 x 128`.
- OT updates: none, as expected because `epochs < 20`.
- Loss was finite.
- Loss history: `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_5/loss_history.json`
- Final embeddings directory: `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_5/final_embeddings`

25 epoch, first 500 spots per section:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
    --config /home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json \
    --epochs 25 \
    --max_spots_per_section 500
```

Result: `MOUSEBRAIN_TRAINING: PASS`

- Final embeddings: `s1/s2/s3 = 500 x 128`.
- OT updates: `[20]`.
- Runtime log printed `Updated OT prior at epoch 20.`
- Loss was finite.
- Loss history: `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_25/loss_history.json`
- Final embeddings directory: `/home/hujinlan/spa_mo_model/results/mousebrain_test/maxspots_500/epochs_25/final_embeddings`

## 7. Issues found

Resolved during integration:

- RNA `var_names` are duplicated gene symbols. The MouseBrain adapter now requires `var["gene_ids"]`, verifies uniqueness, stores original symbols in `var["gene_symbol"]`, and sets RNA `var_names` to gene IDs.
- Metabolite HVG can overflow on the large MALDI/MSI intensity scale if treated like RNA. The new per-modality HVG config keeps RNA HVG at 3000 and sets Metabolite HVG to `None`.
- HE feature key is lowercase `uni_feature`, not default `UNI_feature`. The MouseBrain config uses `uni_feature_key = "uni_feature"`.

Warnings observed:

- AnnData warns about duplicated original RNA `var_names` when reading files. This is expected before the adapter replaces them with unique `gene_ids`.
- Harmony for the small 500-spot HE run printed `Stopped before convergence` after 10 iterations. This was not a fatal error; Harmony returned embeddings and the run completed.

No unresolved blocking issues remain for the small-scale V2 test.

## 8. Temporary files

No temporary verification files were created for this MouseBrain run.

Run outputs are formal result artifacts under:

```text
/home/hujinlan/spa_mo_model/results/mousebrain_test
```

They can be deleted if you want to rerun from scratch:

```bash
rm -rf /home/hujinlan/spa_mo_model/results/mousebrain_test
```

## 9. Final conclusion

PASS for MouseBrain small-scale V2 test.

The full real-data dry run passed, the 5 epoch first-500-spot training test passed, and the 25 epoch first-500-spot training test passed with an OT prior update at epoch 20. MouseBrain is now connected through the intended `HE + RNA + Metabolite` route with Harmony enabled, RNA `gene_ids` used as feature IDs, and Metabolite HVG skipped.
