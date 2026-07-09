# CRC Stereo-CITE-seq spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25`
Total spots: 612374
Sections: CRC_003, CRC_006
Embedding shapes: CRC_003=[166279, 128], CRC_006=[446095, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9962 | 0.4636 | 0.3105 | 0.9901 | 0.009913 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.0308 | 135.6694 | 4.0676 | 0.0073 | 0.0430 | 10000 |
| 25 | 0.0295 | 117.5630 | 3.9466 | 0.0100 | 0.0490 | 10000 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 20 | 0.2545 |
| independent | 25 | 0.2202 |
| joint | 20 | 0.2850 |
| joint | 25 | 0.2464 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/metrics/spatial_continuity_summary.csv`
