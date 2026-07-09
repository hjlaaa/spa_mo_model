# Human Lymph Node spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis`
Total spots: 6843
Sections: Human_Lymph_Node_A1, Human_Lymph_Node_D1
Embedding shapes: Human_Lymph_Node_A1=[3484, 128], Human_Lymph_Node_D1=[3359, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9989 | 0.9331 | 0.8241 | 0.9989 | 0.001090 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0691 | 509.1603 | 2.8074 | 0.0001 | 0.0005 | 6843 |
| 8 | 0.0761 | 384.3135 | 3.1453 | 0.0002 | 0.0007 | 6843 |
| 10 | 0.0727 | 325.5783 | 3.0325 | 0.0000 | 0.0005 | 6843 |
| 12 | 0.0683 | 293.5894 | 3.1610 | 0.0004 | 0.0011 | 6843 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | 0.6180 |
| independent | 8 | 0.5346 |
| independent | 10 | 0.4643 |
| independent | 12 | 0.4183 |
| joint | 5 | 0.6160 |
| joint | 8 | 0.5138 |
| joint | 10 | 0.4563 |
| joint | 12 | 0.4184 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/spatial_continuity_summary.csv`
