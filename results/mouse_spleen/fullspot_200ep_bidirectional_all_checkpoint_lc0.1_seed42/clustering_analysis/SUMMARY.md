# Mouse Spleen spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis`
Total spots: 5336
Sections: Mouse_Spleen1, Mouse_Spleen2
Embedding shapes: Mouse_Spleen1=[2568, 128], Mouse_Spleen2=[2768, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9999 | 0.9818 | 0.9695 | 0.9998 | 0.000226 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0606 | 281.1706 | 3.9512 | 0.0001 | 0.0004 | 5336 |
| 8 | 0.0498 | 208.3498 | 3.8118 | 0.0001 | 0.0007 | 5336 |
| 10 | 0.0510 | 172.1621 | 3.8608 | 0.0001 | 0.0007 | 5336 |
| 12 | 0.0448 | 148.3511 | 4.0155 | 0.0000 | 0.0006 | 5336 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | 0.5078 |
| independent | 8 | 0.3690 |
| independent | 10 | 0.3089 |
| independent | 12 | 0.2648 |
| joint | 5 | 0.4965 |
| joint | 8 | 0.3651 |
| joint | 10 | 0.3046 |
| joint | 12 | 0.2601 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/spatial_continuity_summary.csv`
