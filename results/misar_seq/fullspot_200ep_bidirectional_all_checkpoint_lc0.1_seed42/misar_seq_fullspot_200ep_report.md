# MISAR-seq spa_mo_model Full-Spot 200 Epoch Report

## Run Summary

- Dataset: `/home/hujinlan/spa_mo_model/data/MISAR-seq`
- Sections: `dataset4 -> dataset3 -> dataset2 -> dataset1`
- Biological order: `E11.0 -> E13.5 -> E15.5 -> E18.5`
- Modalities: `RNA + ATAC`
- ATAC handling: true `ATAC` modality, not Protein, not Metabolite
- HE: not fabricated
- Spots: `dataset4=1263`, `dataset3=1777`, `dataset2=1949`, `dataset1=2129`, total `7118`
- Shared features before HVG/PCA: `32285` RNA genes, `191034` ATAC peaks
- Preprocessing: Harmony enabled, `hvg_num=3000`, `hvg_num_atac=3000`, `n_comps=50`
- Training: 200 epochs, `lambda_contrast=0.1`, candidate sparse OT, bidirectional OT attention, all checkpoint switches enabled
- Runtime note: this run used CPU because the active `cosie` PyTorch reports `torch.cuda.is_available() == False`, even though `nvidia-smi` sees the 4090.

## Training Result

Final embedding shapes:

| section | final embedding |
| --- | --- |
| dataset4 | `[1263, 128]` |
| dataset3 | `[1777, 128]` |
| dataset2 | `[1949, 128]` |
| dataset1 | `[2129, 128]` |

Loss checkpoints:

| epoch | total_loss | crossview_loss | reconstruction_loss |
| ---: | ---: | ---: | ---: |
| 1 | -15113.0449 | -151438.6875 | 30.8241 |
| 20 | -1.7491 | -257.0664 | 23.9576 |
| 40 | -2.4762 | -239.1783 | 21.4417 |
| 100 | -6.3329 | -244.8983 | 18.1569 |
| 200 | -13.1763 | -292.5070 | 16.0744 |

Final eval losses:

| metric | value |
| --- | ---: |
| total_loss | -17.0210 |
| crossview_loss | -311.4626 |
| reconstruction_loss | 14.1252 |

Dynamic OT updates ran at epochs `20,40,60,80,100,120,140,160,180,200`.

## Batch Correction Metrics

Batch key is section. Higher is better for `bASW`, `bLISI`, `kBET`, and `PCR_score`; lower is better for `kBET_rejection_rate` and `PCR_batch_R2`.

| bASW | bLISI | kBET | kBET rejection | PCR_score | PCR_batch_R2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9567 | 0.2899 | 0.0334 | 0.9666 | 0.8625 | 0.1375 |

Interpretation: section mixing is still weak locally. bASW is high, but bLISI and especially kBET are low, and PCR_batch_R2 is not negligible. So the embedding preserves meaningful section/developmental-stage structure; this may be biological signal, batch effect, or both, and should not be treated as fully batch-corrected without additional checks.

## Joint Clustering

Best joint ARI by label:

| label | best k | ARI | NMI | cluster ASW scaled | DBI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Y | 14 | 0.2869 | 0.4090 | 0.5498 | 2.5550 |
| Combined_Clusters_annotation | 12 | 0.2705 | 0.4527 | 0.5510 | 2.5291 |
| RNA_Clusters | 12 | 0.3054 | 0.4614 | 0.5510 | 2.5291 |
| ATAC_Clusters | 12 | 0.2594 | 0.4396 | 0.5510 | 2.5291 |
| section | 8 | 0.2751 | 0.3693 | 0.5478 | 2.7900 |

Joint `k=12` is the most balanced setting for cross-section comparison: it has the best ASW/DBI among joint runs and the best or near-best ARI for most biological labels.

Mean spatial neighbor agreement for joint clustering:

| k | agreement |
| ---: | ---: |
| 8 | 0.7183 |
| 10 | 0.6885 |
| 12 | 0.6714 |
| 14 | 0.6431 |
| 16 | 0.6189 |

## Independent Clustering

Best independent ARI by section and label:

| section | label | best k | ARI | NMI |
| --- | --- | ---: | ---: | ---: |
| dataset1 | Combined_Clusters_annotation / Y | 8 | 0.4952 | 0.5681 |
| dataset1 | RNA_Clusters | 8 | 0.4249 | 0.4675 |
| dataset1 | ATAC_Clusters | 8 | 0.3773 | 0.5471 |
| dataset2 | Combined_Clusters_annotation / Y | 10 | 0.5707 | 0.6020 |
| dataset2 | RNA_Clusters | 8 | 0.5287 | 0.5265 |
| dataset2 | ATAC_Clusters | 8 | 0.6237 | 0.6241 |
| dataset3 | Combined_Clusters_annotation / Y | 8 | 0.3124 | 0.4018 |
| dataset3 | RNA_Clusters | 10 | 0.2194 | 0.3590 |
| dataset3 | ATAC_Clusters | 8 | 0.2512 | 0.3723 |
| dataset4 | Combined_Clusters_annotation / Y | 8 | 0.4492 | 0.5240 |
| dataset4 | RNA_Clusters | 8 | 0.4875 | 0.5187 |
| dataset4 | ATAC_Clusters | 8 | 0.3799 | 0.4212 |

Independent clustering is stronger for within-section label recovery, especially `dataset2` and `dataset1`. This is expected because independent KMeans does not force one shared cluster label space across developmental stages.

Mean spatial neighbor agreement for independent clustering:

| k | agreement |
| ---: | ---: |
| 8 | 0.6321 |
| 10 | 0.5568 |
| 12 | 0.5181 |
| 14 | 0.4884 |
| 16 | 0.4766 |

## Visual Read

The spatial plots show clear anatomical/region-like domains rather than random label noise. Joint `k=12` gives comparable labels across stages, while independent `k=8` gives cleaner within-section domains.

Quick view:

`/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/quick_view_joint_k12_independent_k8.png`

## Key Output Files

- Run summary: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/run_summary.json`
- Loss history: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/loss_history.json`
- Final embeddings: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_dataset*.npy`
- OT top-k priors: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/ot_prior_topk/`
- Clustering metrics: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/clustering_metrics.csv`
- Batch metrics: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/batch_correction_metrics.csv`
- Clustering summary: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/clustering_summary.json`

## Current Takeaway

This first MISAR-seq run is successful technically and biologically non-random. The best practical view is:

- Use `joint k=12` for cross-stage comparison.
- Use `independent k=8` for within-stage anatomy/region inspection.
- The model captures spatially coherent domains and has moderate agreement with provided labels.
- Batch/stage mixing is not strong; the embedding still carries developmental-stage structure. That may be desirable for developmental data, but if the goal is stronger stage correction, this needs a separate tuning pass.
