# MISAR-seq spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16`
Total spots: 7118
Sections: dataset4, dataset3, dataset2, dataset1
Embedding shapes: dataset4=[1263, 128], dataset3=[1777, 128], dataset2=[1949, 128], dataset1=[2129, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9567 | 0.2899 | 0.0334 | 0.8625 | 0.137547 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | Metric spots |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0955 | 401.0718 | 2.8016 | 0.2754 | 0.3700 | 7118 |
| 10 | 0.0940 | 356.8732 | 2.6958 | 0.2284 | 0.3535 | 7118 |
| 12 | 0.1021 | 329.7466 | 2.5281 | 0.1990 | 0.3374 | 7118 |
| 14 | 0.0990 | 299.1609 | 2.5533 | 0.1883 | 0.3280 | 7118 |
| 16 | 0.1006 | 275.2533 | 2.6417 | 0.1834 | 0.3342 | 7118 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 8 | 0.6256 |
| independent | 10 | 0.5555 |
| independent | 12 | 0.5225 |
| independent | 14 | 0.5032 |
| independent | 16 | 0.4675 |
| joint | 8 | 0.7174 |
| joint | 10 | 0.6784 |
| joint | 12 | 0.6713 |
| joint | 14 | 0.6439 |
| joint | 16 | 0.6253 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/metrics/spatial_continuity_summary.csv`

## Completed external metrics and plot adaptation

- `clustering_metrics.csv` includes ARI, NMI, Homogeneity, Completeness, V-measure, Label ASW and scaled Label ASW.
- Spatial clustering spot size: 15.
