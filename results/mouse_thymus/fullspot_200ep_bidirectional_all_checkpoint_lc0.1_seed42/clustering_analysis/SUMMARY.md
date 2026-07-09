# Mouse Thymus spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis`
Total spots: 17824
Sections: Mouse_Thymus1, Mouse_Thymus2, Mouse_Thymus3, Mouse_Thymus4
Embedding shapes: Mouse_Thymus1=[4697, 128], Mouse_Thymus2=[4253, 128], Mouse_Thymus3=[4646, 128], Mouse_Thymus4=[4228, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9075 | 0.4549 | 0.0025 | 0.7399 | 0.260052 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.2773 | 2157.7739 | 2.2222 | 0.3070 | 0.3826 | 10000 |
| 8 | 0.2473 | 1520.4362 | 2.0023 | 0.2434 | 0.3507 | 10000 |
| 10 | 0.1951 | 1297.3773 | 2.2446 | 0.2370 | 0.3239 | 10000 |
| 12 | 0.1846 | 1139.8644 | 2.1625 | 0.2351 | 0.3132 | 10000 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | NA |
| independent | 8 | NA |
| independent | 10 | NA |
| independent | 12 | NA |
| joint | 5 | NA |
| joint | 8 | NA |
| joint | 10 | NA |
| joint | 12 | NA |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/spatial_continuity_summary.csv`

## Mouse Thymus plot adaptation

- Clustering spot sizes: Mouse_Thymus1 = 8; Mouse_Thymus2/3/4 = 5.
