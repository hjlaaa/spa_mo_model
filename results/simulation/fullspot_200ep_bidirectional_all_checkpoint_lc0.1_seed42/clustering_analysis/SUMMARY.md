# Simulation spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis`
Total spots: 6480
Sections: Simulation1, Simulation2, Simulation3, Simulation4, Simulation5
Embedding shapes: Simulation1=[1296, 128], Simulation2=[1296, 128], Simulation3=[1296, 128], Simulation4=[1296, 128], Simulation5=[1296, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9988 | 0.9767 | 0.9867 | 1.0000 | 0.000012 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | ASW spots | CH/DB spots |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.3430 | 1547.1149 | 1.3957 | -0.0006 | 0.0000 | 6480 | 6480 |
| 8 | 0.4371 | 2210.6932 | 0.9394 | -0.0008 | 0.0000 | 6480 | 6480 |
| 10 | 0.4715 | 2191.4295 | 1.0468 | -0.0008 | 0.0000 | 6480 | 6480 |
| 12 | 0.4900 | 2185.1643 | 1.0255 | -0.0009 | 0.0000 | 6480 | 6480 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 5 | 0.6130 |
| independent | 8 | 0.4041 |
| independent | 10 | 0.3715 |
| independent | 12 | 0.3314 |
| joint | 5 | 0.6588 |
| joint | 8 | 0.4058 |
| joint | 10 | 0.3703 |
| joint | 12 | 0.3300 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/spatial_continuity_summary.csv`

## Simulation-specific diagnostics

- Spatial-factor recovery and RNA/ADT nuisance-factor leakage: `metrics/simulation_factor_diagnostics.csv`.
- Same-grid cross-section retrieval/FOSCTTM: `metrics/cross_section_spot_retrieval.csv`.
- Ground-truth factors were used only during this analysis, never in training.
