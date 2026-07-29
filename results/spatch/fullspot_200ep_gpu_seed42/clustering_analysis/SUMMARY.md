# spatch spa_mo_model Analysis Summary

Run directory: `/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42`
Analysis directory: `/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis`
Total spots: 1068962
Sections: section1, section2
Embedding shapes: section1=[665399, 128], section2=[403563, 128]

## Batch correction metrics

bASW, bLISI, kBET and PCR_score are higher-is-better; PCR_batch_R2 is lower-is-better.

| bASW | bLISI | kBET | PCR_score | PCR_batch_R2 |
|---:|---:|---:|---:|---:|
| 0.9912 | 0.3595 | 0.1595 | 0.9930 | 0.007010 |

## Joint clustering metrics

Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin is lower-is-better. Section ARI/NMI are diagnostics: values near zero indicate that clusters are not simply section labels.

| k | Silhouette | CH | DB | Section ARI | Section NMI | ASW spots | CH/DB spots |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.1140 | 135263.3034 | 2.7918 | 0.0022 | 0.0007 | 10000 | 1068962 |
| 3 | 0.0968 | 100447.5277 | 2.9316 | -0.0015 | 0.0005 | 10000 | 1068962 |
| 4 | 0.0814 | 82042.6779 | 3.2750 | 0.0089 | 0.0082 | 10000 | 1068962 |
| 5 | 0.0945 | 81370.5237 | 2.7910 | 0.0009 | 0.0014 | 10000 | 1068962 |
| 6 | 0.0967 | 75744.5788 | 2.7253 | 0.0030 | 0.0058 | 10000 | 1068962 |
| 7 | 0.1086 | 73846.0497 | 2.4510 | 0.0013 | 0.0021 | 10000 | 1068962 |
| 8 | 0.1076 | 68146.3196 | 2.5013 | 0.0033 | 0.0065 | 10000 | 1068962 |
| 9 | 0.1108 | 67102.4456 | 2.4033 | 0.0035 | 0.0051 | 10000 | 1068962 |
| 10 | 0.1255 | 67823.9711 | 2.1518 | 0.0147 | 0.0360 | 10000 | 1068962 |
| 11 | 0.1282 | 66046.4871 | 2.1735 | 0.0029 | 0.0054 | 10000 | 1068962 |
| 12 | 0.1205 | 61453.0599 | 2.1882 | 0.0084 | 0.0154 | 10000 | 1068962 |
| 13 | 0.1286 | 61840.1332 | 2.0902 | 0.0123 | 0.0357 | 10000 | 1068962 |
| 14 | 0.1205 | 57246.7698 | 2.2035 | 0.0156 | 0.0398 | 10000 | 1068962 |
| 15 | 0.1303 | 56724.5700 | 2.1136 | 0.0113 | 0.0354 | 10000 | 1068962 |
| 16 | 0.1283 | 55131.7233 | 2.1541 | 0.0157 | 0.0438 | 10000 | 1068962 |
| 17 | 0.1187 | 51273.7962 | 2.2976 | 0.0025 | 0.0065 | 10000 | 1068962 |
| 18 | 0.1258 | 52146.1858 | 2.1672 | 0.0102 | 0.0336 | 10000 | 1068962 |
| 19 | 0.1307 | 51278.9938 | 2.0864 | 0.0127 | 0.0405 | 10000 | 1068962 |
| 20 | 0.1247 | 49334.7239 | 2.1283 | 0.0125 | 0.0443 | 10000 | 1068962 |

## Spatial continuity

Spatial continuity is the fraction of nearest spatial neighbors assigned to the same cluster; higher is smoother but may also reflect over-smoothing.

| Mode | k | Mean neighbor agreement |
|---|---:|---:|
| independent | 2 | 0.9590 |
| independent | 3 | 0.9146 |
| independent | 4 | 0.8862 |
| independent | 5 | 0.8626 |
| independent | 6 | 0.8524 |
| independent | 7 | 0.8097 |
| independent | 8 | 0.8192 |
| independent | 9 | 0.7819 |
| independent | 10 | 0.7997 |
| independent | 11 | 0.7629 |
| independent | 12 | 0.7530 |
| independent | 13 | 0.7314 |
| independent | 14 | 0.7307 |
| independent | 15 | 0.7217 |
| independent | 16 | 0.7072 |
| independent | 17 | 0.7046 |
| independent | 18 | 0.6738 |
| independent | 19 | 0.6751 |
| independent | 20 | 0.6540 |
| joint | 2 | 0.9511 |
| joint | 3 | 0.9156 |
| joint | 4 | 0.8787 |
| joint | 5 | 0.8477 |
| joint | 6 | 0.7934 |
| joint | 7 | 0.7715 |
| joint | 8 | 0.7257 |
| joint | 9 | 0.7490 |
| joint | 10 | 0.7515 |
| joint | 11 | 0.7263 |
| joint | 12 | 0.7325 |
| joint | 13 | 0.7038 |
| joint | 14 | 0.7133 |
| joint | 15 | 0.7003 |
| joint | 16 | 0.6983 |
| joint | 17 | 0.6662 |
| joint | 18 | 0.6416 |
| joint | 19 | 0.6840 |
| joint | 20 | 0.6562 |

## Interpretation limits

Section diagnostics measure batch dependence, not biological accuracy. Biological ARI/NMI should only be interpreted when reliable cell-type or domain annotations are present.

## Output files

- Common metrics: `/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/metrics/common_clustering_metrics.csv`
- Section diagnostics: `/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/metrics/section_diagnostic_metrics.csv`
- Spatial continuity summary: `/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/metrics/spatial_continuity_summary.csv`
