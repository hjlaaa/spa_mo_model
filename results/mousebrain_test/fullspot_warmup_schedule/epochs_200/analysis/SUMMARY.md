# MouseBrain spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200`
Analysis directory: `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis`
Total spots: 7866
Sections: s1, s2, s3
Embedding shapes: s1=[2384, 128], s2=[2820, 128], s3=[2662, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9984 | 0.7302 | 0.2400 | 0.9866 | 0.013361 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.1632 | 625.7303 | 2.0955 | 0.0036 | 0.0035 | 5000 |
| 6 | 0.1436 | 580.2725 | 2.1753 | 0.0026 | 0.0034 | 5000 |
| 8 | 0.1472 | 498.6803 | 2.1953 | 0.0020 | 0.0037 | 5000 |
| 10 | 0.1368 | 437.8316 | 2.2336 | 0.0022 | 0.0048 | 5000 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | 0.8608 |
| independent | 6 | 0.8326 |
| independent | 8 | 0.7911 |
| independent | 10 | 0.7357 |
| joint | 5 | 0.8654 |
| joint | 6 | 0.8217 |
| joint | 8 | 0.7737 |
| joint | 10 | 0.7159 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics/spatial_continuity_summary.csv`
