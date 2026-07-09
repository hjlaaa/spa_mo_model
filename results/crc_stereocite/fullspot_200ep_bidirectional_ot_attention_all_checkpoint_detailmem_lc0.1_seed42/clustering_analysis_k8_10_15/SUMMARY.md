# CRC Stereo-CITE-seq spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15`
Total spots: 612374
Sections: CRC_003, CRC_006
Embedding shapes: CRC_003=[166279, 128], CRC_006=[446095, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9962 | 0.4636 | 0.3104 | 0.9901 | 0.009913 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0402 | 323.1474 | 4.1382 | 0.0305 | 0.0614 | 10000 |
| 6 | 0.0367 | 288.5415 | 4.2247 | 0.0156 | 0.0445 | 10000 |
| 8 | 0.0378 | 247.0358 | 4.0845 | 0.0283 | 0.0561 | 10000 |
| 10 | 0.0346 | 211.6093 | 4.1502 | 0.0118 | 0.0555 | 10000 |
| 15 | 0.0327 | 163.7464 | 3.9235 | 0.0082 | 0.0443 | 10000 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | 0.5944 |
| independent | 6 | 0.5280 |
| independent | 8 | 0.4789 |
| independent | 10 | 0.3940 |
| independent | 15 | 0.3252 |
| joint | 5 | 0.6415 |
| joint | 6 | 0.5554 |
| joint | 8 | 0.4923 |
| joint | 10 | 0.4608 |
| joint | 15 | 0.3527 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/metrics/spatial_continuity_summary.csv`
