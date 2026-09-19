# MISAR-seq：Raw 与 Standardized embedding + KMeans 对比报告

- 生成时间：2026-07-27T14:39:10.272432+08:00
- 本报告仅引用新增 comparison 结果；旧分析目录和原综合报告未覆盖。
- Raw 与 Standardized 分别在自身 KMeans 输入空间计算距离型指标；跨方案变化需结合标签稳定性解释。

## 1. 新结果目录

| 方法         | comparison root                                                          | 方法自有数据副本                                 |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------ |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/misar_seq_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/MISAR-seq       |
| MOFA+        | /home/hujinlan/mofa+/analysis/misar_seq_preprocessing_comparison         | /home/hujinlan/mofa+/data/MISAR-seq              |
| COSIE        | /home/hujinlan/cosie_runs/misar_seq_preprocessing_comparison             | /home/hujinlan/cosie/data/MISAR-seq              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/misar_seq_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/MISAR-seq |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。

## 2. 统一标准

| 项目                             | 统一值                                                    |
| -------------------------------- | --------------------------------------------------------- |
| spot/barcode                     | 各方法自有数据；section + original barcode 精确对齐       |
| spot 数                          | 7118（2129/1949/1777/1263）                               |
| raw 分支                         | 原始最终 embedding 直接输入 KMeans                        |
| standardized 分支                | StandardScaler；joint 全体拟合，independent 每section拟合 |
| KMeans                           | sklearn exact KMeans                                      |
| joint/independent指标 K          | 2–16                                                      |
| 保留的聚类文件夹/labels/空间图 K | 5、8、10、12、14、16                                      |
| 空间图点大小                     | 18                                                        |
| 随机参数                         | seed=0，n_init=20，max_iter=300                           |
| Cluster ASW/CH/DBI               | 每个scope全量样本                                         |
| 外部指标                         | 5类生物标签；全部7118个spot                               |
| Label ASW                        | 每个scope全部有效标签spot                                 |
| 空间连续性                       | 每section全点，exact 6-NN                                 |
| labels复用                       | 保留K的外部、内部、空间指标和图复用全量labels             |
| batch指标                        | KMeans分支无关；复制并记录SHA256                          |

## 3. 数据、标签与 embedding 对齐

| 方法         | 维度 | dataset1 | dataset2 | dataset3 | dataset4 | 标签有效 n |
| ------------ | ---: | -------: | -------: | -------: | -------: | ---------: |
| spa_mo_model |  128 |     2129 |     1949 |     1777 |     1263 |       7118 |
| MOFA+        |   10 |     2129 |     1949 |     1777 |     1263 |       7118 |
| COSIE        |  256 |     2129 |     1949 |     1777 |     1263 |       7118 |
| SpaMosaic    |   32 |     2129 |     1949 |     1777 |     1263 |       7118 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_dataset1.npy`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_dataset1.csv`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_dataset2.npy`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_dataset2.csv`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_dataset3.npy`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_dataset3.csv`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_dataset4.npy`
  - `/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_dataset4.csv`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/analysis/tables/embeddings_with_metadata.csv`
  - `/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/final_embeddings/dataset1_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/final_embeddings/dataset2_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/final_embeddings/dataset3_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/final_embeddings/dataset4_final_embedding.npy`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/misar_seq_spamosaic_full/misar_seq_spamosaic_embeddings.h5ad`

## 4. 外部指标：Raw / Standardized 严格对齐对比

同一方法、同一标签的Raw与Standardized行相邻；best K在各分支内按joint ARI独立选择。

| 方法         | 标签                         | 预处理       | best K | 有效 n | 标签类数 |    ARI |    NMI | Homogeneity | Completeness | V-measure | Cluster ASW raw | Cluster ASW scaled | Label ASW raw | Label ASW scaled |
| ------------ | ---------------------------- | ------------ | -----: | -----: | -------: | -----: | -----: | ----------: | -----------: | --------: | --------------: | -----------------: | ------------: | ---------------: |
| spa_mo_model | Y                            | Raw          |     12 |   7118 |       14 | 0.2840 | 0.4041 |      0.4126 |       0.3959 |    0.4041 |          0.1076 |             0.5538 |       -0.0092 |           0.4954 |
| spa_mo_model | Y                            | Standardized |     14 |   7118 |       14 | 0.2839 | 0.4093 |      0.4302 |       0.3903 |    0.4093 |          0.0994 |             0.5497 |       -0.0090 |           0.4955 |
| spa_mo_model | Combined_Clusters_annotation | Raw          |      5 |   7118 |       16 | 0.2910 | 0.3878 |      0.3280 |       0.4743 |    0.3878 |          0.1122 |             0.5561 |        0.0061 |           0.5030 |
| spa_mo_model | Combined_Clusters_annotation | Standardized |      5 |   7118 |       16 | 0.2971 | 0.3953 |      0.3338 |       0.4847 |    0.3953 |          0.1073 |             0.5536 |        0.0076 |           0.5038 |
| spa_mo_model | Combined_Clusters            | Raw          |      5 |   7118 |       16 | 0.2910 | 0.3878 |      0.3280 |       0.4743 |    0.3878 |          0.1122 |             0.5561 |        0.0061 |           0.5030 |
| spa_mo_model | Combined_Clusters            | Standardized |      5 |   7118 |       16 | 0.2971 | 0.3953 |      0.3338 |       0.4847 |    0.3953 |          0.1073 |             0.5536 |        0.0076 |           0.5038 |
| spa_mo_model | RNA_Clusters                 | Raw          |     11 |   7118 |       13 | 0.3109 | 0.4629 |      0.4706 |       0.4555 |    0.4629 |          0.1022 |             0.5511 |        0.0288 |           0.5144 |
| spa_mo_model | RNA_Clusters                 | Standardized |     12 |   7118 |       13 | 0.3058 | 0.4622 |      0.4756 |       0.4495 |    0.4622 |          0.1020 |             0.5510 |        0.0280 |           0.5140 |
| spa_mo_model | ATAC_Clusters                | Raw          |      5 |   7118 |       13 | 0.2884 | 0.3741 |      0.3248 |       0.4408 |    0.3741 |          0.1122 |             0.5561 |        0.0257 |           0.5129 |
| spa_mo_model | ATAC_Clusters                | Standardized |      5 |   7118 |       13 | 0.2939 | 0.3814 |      0.3307 |       0.4506 |    0.3814 |          0.1073 |             0.5536 |        0.0271 |           0.5135 |
| MOFA+        | Y                            | Raw          |     10 |   7118 |       14 | 0.1498 | 0.2514 |      0.2443 |       0.2589 |    0.2514 |          0.1744 |             0.5872 |       -0.0906 |           0.4547 |
| MOFA+        | Y                            | Standardized |      7 |   7118 |       14 | 0.1952 | 0.2688 |      0.2333 |       0.3170 |    0.2688 |          0.1604 |             0.5802 |       -0.0738 |           0.4631 |
| MOFA+        | Combined_Clusters_annotation | Raw          |      5 |   7118 |       16 | 0.1981 | 0.2851 |      0.2315 |       0.3710 |    0.2851 |          0.2029 |             0.6015 |       -0.0344 |           0.4828 |
| MOFA+        | Combined_Clusters_annotation | Standardized |      7 |   7118 |       16 | 0.3214 | 0.3718 |      0.3281 |       0.4289 |    0.3718 |          0.1604 |             0.5802 |       -0.0084 |           0.4958 |
| MOFA+        | Combined_Clusters            | Raw          |      5 |   7118 |       16 | 0.1981 | 0.2851 |      0.2315 |       0.3710 |    0.2851 |          0.2029 |             0.6015 |       -0.0344 |           0.4828 |
| MOFA+        | Combined_Clusters            | Standardized |      7 |   7118 |       16 | 0.3214 | 0.3718 |      0.3281 |       0.4289 |    0.3718 |          0.1604 |             0.5802 |       -0.0084 |           0.4958 |
| MOFA+        | RNA_Clusters                 | Raw          |     13 |   7118 |       13 | 0.1457 | 0.2826 |      0.2912 |       0.2744 |    0.2826 |          0.1701 |             0.5851 |       -0.0627 |           0.4687 |
| MOFA+        | RNA_Clusters                 | Standardized |      7 |   7118 |       13 | 0.2044 | 0.3180 |      0.2778 |       0.3718 |    0.3180 |          0.1604 |             0.5802 |       -0.0404 |           0.4798 |
| MOFA+        | ATAC_Clusters                | Raw          |     10 |   7118 |       13 | 0.1979 | 0.3068 |      0.3138 |       0.3002 |    0.3068 |          0.1744 |             0.5872 |       -0.0416 |           0.4792 |
| MOFA+        | ATAC_Clusters                | Standardized |      7 |   7118 |       13 | 0.2833 | 0.3626 |      0.3291 |       0.4037 |    0.3626 |          0.1604 |             0.5802 |       -0.0242 |           0.4879 |
| COSIE        | Y                            | Raw          |     14 |   7118 |       14 | 0.3112 | 0.4530 |      0.4835 |       0.4262 |    0.4530 |          0.1660 |             0.5830 |        0.0057 |           0.5029 |
| COSIE        | Y                            | Standardized |     13 |   7118 |       14 | 0.2985 | 0.4453 |      0.4656 |       0.4267 |    0.4453 |          0.1597 |             0.5799 |        0.0107 |           0.5053 |
| COSIE        | Combined_Clusters_annotation | Raw          |      6 |   7118 |       16 | 0.3121 | 0.4183 |      0.3691 |       0.4827 |    0.4183 |          0.1755 |             0.5878 |        0.0310 |           0.5155 |
| COSIE        | Combined_Clusters_annotation | Standardized |      6 |   7118 |       16 | 0.3278 | 0.4345 |      0.3854 |       0.4980 |    0.4345 |          0.1574 |             0.5787 |        0.0409 |           0.5204 |
| COSIE        | Combined_Clusters            | Raw          |      6 |   7118 |       16 | 0.3121 | 0.4183 |      0.3691 |       0.4827 |    0.4183 |          0.1755 |             0.5878 |        0.0310 |           0.5155 |
| COSIE        | Combined_Clusters            | Standardized |      6 |   7118 |       16 | 0.3278 | 0.4345 |      0.3854 |       0.4980 |    0.4345 |          0.1574 |             0.5787 |        0.0409 |           0.5204 |
| COSIE        | RNA_Clusters                 | Raw          |     15 |   7118 |       13 | 0.2857 | 0.4832 |      0.5262 |       0.4468 |    0.4832 |          0.1704 |             0.5852 |        0.0234 |           0.5117 |
| COSIE        | RNA_Clusters                 | Standardized |     13 |   7118 |       13 | 0.2747 | 0.4692 |      0.4945 |       0.4464 |    0.4692 |          0.1597 |             0.5799 |        0.0298 |           0.5149 |
| COSIE        | ATAC_Clusters                | Raw          |      6 |   7118 |       13 | 0.3105 | 0.4082 |      0.3705 |       0.4546 |    0.4082 |          0.1755 |             0.5878 |        0.0598 |           0.5299 |
| COSIE        | ATAC_Clusters                | Standardized |      6 |   7118 |       13 | 0.3249 | 0.4202 |      0.3833 |       0.4648 |    0.4202 |          0.1574 |             0.5787 |        0.0704 |           0.5352 |
| SpaMosaic    | Y                            | Raw          |      6 |   7118 |       14 | 0.2329 | 0.3123 |      0.2738 |       0.3634 |    0.3123 |          0.2018 |             0.6009 |       -0.1187 |           0.4407 |
| SpaMosaic    | Y                            | Standardized |      6 |   7118 |       14 | 0.2309 | 0.3202 |      0.2822 |       0.3700 |    0.3202 |          0.2033 |             0.6017 |       -0.1091 |           0.4455 |
| SpaMosaic    | Combined_Clusters_annotation | Raw          |      6 |   7118 |       16 | 0.3451 | 0.4375 |      0.3901 |       0.4981 |    0.4375 |          0.2018 |             0.6009 |       -0.0742 |           0.4629 |
| SpaMosaic    | Combined_Clusters_annotation | Standardized |      6 |   7118 |       16 | 0.3440 | 0.4475 |      0.4012 |       0.5060 |    0.4475 |          0.2033 |             0.6017 |       -0.0718 |           0.4641 |
| SpaMosaic    | Combined_Clusters            | Raw          |      6 |   7118 |       16 | 0.3451 | 0.4375 |      0.3901 |       0.4981 |    0.4375 |          0.2018 |             0.6009 |       -0.0742 |           0.4629 |
| SpaMosaic    | Combined_Clusters            | Standardized |      6 |   7118 |       16 | 0.3440 | 0.4475 |      0.4012 |       0.5060 |    0.4475 |          0.2033 |             0.6017 |       -0.0718 |           0.4641 |
| SpaMosaic    | RNA_Clusters                 | Raw          |      6 |   7118 |       13 | 0.2359 | 0.3565 |      0.3146 |       0.4113 |    0.3565 |          0.2018 |             0.6009 |       -0.0741 |           0.4629 |
| SpaMosaic    | RNA_Clusters                 | Standardized |      6 |   7118 |       13 | 0.2345 | 0.3601 |      0.3194 |       0.4126 |    0.3601 |          0.2033 |             0.6017 |       -0.0706 |           0.4647 |
| SpaMosaic    | ATAC_Clusters                | Raw          |      4 |   7118 |       13 | 0.3346 | 0.4396 |      0.3650 |       0.5527 |    0.4396 |          0.2223 |             0.6112 |       -0.0566 |           0.4717 |
| SpaMosaic    | ATAC_Clusters                | Standardized |      6 |   7118 |       13 | 0.3131 | 0.4453 |      0.4108 |       0.4862 |    0.4453 |          0.2033 |             0.6017 |       -0.0521 |           0.4740 |

完整joint/independent K=2–16逐标签指标见各分支 `metrics/clustering_metrics_by_label.csv`。

## 5. 内部、section诊断与空间指标：Raw / Standardized 严格对齐对比

| 方法         |   K | 预处理       | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |        CH |    DBI | section ARI | section NMI | section ASW n | section ASW raw | joint spatial |
| ------------ | --: | ------------ | ----: | --------------: | -----------------: | -------: | --------: | -----: | ----------: | ----------: | ------------: | --------------: | ------------: |
| spa_mo_model |   5 | Raw          |  7118 |          0.1122 |             0.5561 |     7118 |  567.7348 | 2.4955 |      0.2040 |      0.2901 |          7118 |          0.0460 |        0.7751 |
| spa_mo_model |   5 | Standardized |  7118 |          0.1073 |             0.5536 |     7118 |  525.0399 | 2.5442 |      0.1999 |      0.2857 |          7118 |          0.0433 |        0.7804 |
| spa_mo_model |   8 | Raw          |  7118 |          0.0964 |             0.5482 |     7118 |  430.6178 | 2.7870 |      0.2683 |      0.3718 |          7118 |          0.0460 |        0.7034 |
| spa_mo_model |   8 | Standardized |  7118 |          0.0948 |             0.5474 |     7118 |  402.4261 | 2.7728 |      0.2714 |      0.3645 |          7118 |          0.0433 |        0.7221 |
| spa_mo_model |  10 | Raw          |  7118 |          0.0995 |             0.5497 |     7118 |  382.8218 | 2.6564 |      0.2289 |      0.3545 |          7118 |          0.0460 |        0.6788 |
| spa_mo_model |  10 | Standardized |  7118 |          0.0963 |             0.5481 |     7118 |  358.0109 | 2.6179 |      0.2189 |      0.3299 |          7118 |          0.0433 |        0.6784 |
| spa_mo_model |  12 | Raw          |  7118 |          0.1076 |             0.5538 |     7118 |  352.4335 | 2.4861 |      0.2011 |      0.3389 |          7118 |          0.0460 |        0.6726 |
| spa_mo_model |  12 | Standardized |  7118 |          0.1020 |             0.5510 |     7118 |  330.3665 | 2.5197 |      0.2014 |      0.3373 |          7118 |          0.0433 |        0.6724 |
| spa_mo_model |  14 | Raw          |  7118 |          0.1030 |             0.5515 |     7118 |  317.9058 | 2.5872 |      0.1922 |      0.3353 |          7118 |          0.0460 |        0.6420 |
| spa_mo_model |  14 | Standardized |  7118 |          0.0994 |             0.5497 |     7118 |  299.8675 | 2.5505 |      0.1873 |      0.3258 |          7118 |          0.0433 |        0.6453 |
| spa_mo_model |  16 | Raw          |  7118 |          0.0994 |             0.5497 |     7118 |  291.5228 | 2.5568 |      0.1765 |      0.3352 |          7118 |          0.0460 |        0.6280 |
| spa_mo_model |  16 | Standardized |  7118 |          0.1008 |             0.5504 |     7118 |  276.1262 | 2.6400 |      0.1812 |      0.3261 |          7118 |          0.0433 |        0.6142 |
| MOFA+        |   5 | Raw          |  7118 |          0.2029 |             0.6015 |     7118 | 1361.8980 | 1.4257 |      0.0253 |      0.0510 |          7118 |         -0.0265 |        0.6415 |
| MOFA+        |   5 | Standardized |  7118 |          0.1355 |             0.5677 |     7118 |  843.1104 | 1.8713 |      0.0400 |      0.0457 |          7118 |         -0.0356 |        0.6198 |
| MOFA+        |   8 | Raw          |  7118 |          0.1902 |             0.5951 |     7118 | 1203.9851 | 1.4379 |      0.0551 |      0.0970 |          7118 |         -0.0265 |        0.5001 |
| MOFA+        |   8 | Standardized |  7118 |          0.1668 |             0.5834 |     7118 |  845.5833 | 1.5376 |      0.0429 |      0.0792 |          7118 |         -0.0356 |        0.5304 |
| MOFA+        |  10 | Raw          |  7118 |          0.1744 |             0.5872 |     7118 | 1104.0012 | 1.4654 |      0.0452 |      0.0990 |          7118 |         -0.0265 |        0.4326 |
| MOFA+        |  10 | Standardized |  7118 |          0.1788 |             0.5894 |     7118 |  827.4954 | 1.4391 |      0.0530 |      0.1205 |          7118 |         -0.0356 |        0.5186 |
| MOFA+        |  12 | Raw          |  7118 |          0.1674 |             0.5837 |     7118 | 1022.7533 | 1.5043 |      0.0596 |      0.1335 |          7118 |         -0.0265 |        0.4257 |
| MOFA+        |  12 | Standardized |  7118 |          0.1645 |             0.5823 |     7118 |  787.0387 | 1.4803 |      0.0587 |      0.1267 |          7118 |         -0.0356 |        0.4676 |
| MOFA+        |  14 | Raw          |  7118 |          0.1635 |             0.5818 |     7118 |  962.7711 | 1.5367 |      0.0500 |      0.1237 |          7118 |         -0.0265 |        0.3858 |
| MOFA+        |  14 | Standardized |  7118 |          0.1673 |             0.5836 |     7118 |  745.3898 | 1.4795 |      0.0754 |      0.1591 |          7118 |         -0.0356 |        0.4620 |
| MOFA+        |  16 | Raw          |  7118 |          0.1610 |             0.5805 |     7118 |  899.5156 | 1.5435 |      0.0713 |      0.1716 |          7118 |         -0.0265 |        0.3917 |
| MOFA+        |  16 | Standardized |  7118 |          0.1666 |             0.5833 |     7118 |  712.5312 | 1.4677 |      0.0864 |      0.1842 |          7118 |         -0.0356 |        0.4427 |
| COSIE        |   5 | Raw          |  7118 |          0.1779 |             0.5890 |     7118 | 1272.9532 | 1.8035 |      0.2199 |      0.3051 |          7118 |          0.0674 |        0.8599 |
| COSIE        |   5 | Standardized |  7118 |          0.1572 |             0.5786 |     7118 |  977.1635 | 1.9545 |      0.2103 |      0.2976 |          7118 |          0.0642 |        0.8554 |
| COSIE        |   8 | Raw          |  7118 |          0.1667 |             0.5834 |     7118 |  990.5167 | 1.8564 |      0.1874 |      0.2845 |          7118 |          0.0674 |        0.7942 |
| COSIE        |   8 | Standardized |  7118 |          0.1480 |             0.5740 |     7118 |  780.0136 | 1.9413 |      0.2549 |      0.3400 |          7118 |          0.0642 |        0.8152 |
| COSIE        |  10 | Raw          |  7118 |          0.1629 |             0.5814 |     7118 |  894.8588 | 1.8490 |      0.2256 |      0.3344 |          7118 |          0.0674 |        0.7840 |
| COSIE        |  10 | Standardized |  7118 |          0.1553 |             0.5777 |     7118 |  717.5882 | 1.9372 |      0.2450 |      0.3460 |          7118 |          0.0642 |        0.7870 |
| COSIE        |  12 | Raw          |  7118 |          0.1786 |             0.5893 |     7118 |  839.3155 | 1.7708 |      0.2345 |      0.3769 |          7118 |          0.0674 |        0.7848 |
| COSIE        |  12 | Standardized |  7118 |          0.1672 |             0.5836 |     7118 |  671.9597 | 1.8624 |      0.2429 |      0.3925 |          7118 |          0.0642 |        0.7854 |
| COSIE        |  14 | Raw          |  7118 |          0.1660 |             0.5830 |     7118 |  772.1106 | 1.8465 |      0.2232 |      0.4039 |          7118 |          0.0674 |        0.7656 |
| COSIE        |  14 | Standardized |  7118 |          0.1670 |             0.5835 |     7118 |  618.5530 | 1.8925 |      0.2138 |      0.3911 |          7118 |          0.0642 |        0.7708 |
| COSIE        |  16 | Raw          |  7118 |          0.1706 |             0.5853 |     7118 |  717.6192 | 1.7899 |      0.2004 |      0.3863 |          7118 |          0.0674 |        0.7532 |
| COSIE        |  16 | Standardized |  7118 |          0.1603 |             0.5801 |     7118 |  580.9784 | 1.9016 |      0.2097 |      0.3984 |          7118 |          0.0642 |        0.7504 |
| SpaMosaic    |   5 | Raw          |  7118 |          0.2006 |             0.6003 |     7118 | 2863.2863 | 1.8505 |      0.0558 |      0.0671 |          7118 |         -0.0492 |        0.7397 |
| SpaMosaic    |   5 | Standardized |  7118 |          0.1957 |             0.5979 |     7118 | 1921.1075 | 1.8691 |      0.0541 |      0.0657 |          7118 |         -0.0348 |        0.7390 |
| SpaMosaic    |   8 | Raw          |  7118 |          0.2017 |             0.6008 |     7118 | 2103.3206 | 1.6868 |      0.0542 |      0.0801 |          7118 |         -0.0492 |        0.6689 |
| SpaMosaic    |   8 | Standardized |  7118 |          0.1972 |             0.5986 |     7118 | 1459.8757 | 1.6856 |      0.0525 |      0.0800 |          7118 |         -0.0348 |        0.6568 |
| SpaMosaic    |  10 | Raw          |  7118 |          0.1626 |             0.5813 |     7118 | 1800.6231 | 1.7465 |      0.0505 |      0.0794 |          7118 |         -0.0492 |        0.6213 |
| SpaMosaic    |  10 | Standardized |  7118 |          0.1598 |             0.5799 |     7118 | 1257.7328 | 1.7751 |      0.0486 |      0.0786 |          7118 |         -0.0348 |        0.6139 |
| SpaMosaic    |  12 | Raw          |  7118 |          0.1552 |             0.5776 |     7118 | 1584.0284 | 1.7880 |      0.0450 |      0.0794 |          7118 |         -0.0492 |        0.6011 |
| SpaMosaic    |  12 | Standardized |  7118 |          0.1551 |             0.5776 |     7118 | 1116.9682 | 1.7822 |      0.0419 |      0.0771 |          7118 |         -0.0348 |        0.6066 |
| SpaMosaic    |  14 | Raw          |  7118 |          0.1483 |             0.5742 |     7118 | 1433.8712 | 1.8411 |      0.0406 |      0.0787 |          7118 |         -0.0492 |        0.5671 |
| SpaMosaic    |  14 | Standardized |  7118 |          0.1447 |             0.5723 |     7118 | 1012.7825 | 1.8722 |      0.0388 |      0.0765 |          7118 |         -0.0348 |        0.5602 |
| SpaMosaic    |  16 | Raw          |  7118 |          0.1406 |             0.5703 |     7118 | 1297.5472 | 1.8925 |      0.0376 |      0.0781 |          7118 |         -0.0492 |        0.5380 |
| SpaMosaic    |  16 | Standardized |  7118 |          0.1303 |             0.5652 |     7118 |  922.5580 | 1.9062 |      0.0351 |      0.0765 |          7118 |         -0.0348 |        0.5307 |

完整joint/independent K=2–16内部与空间指标仍保存在各分支 `metrics/`；聚类文件夹、全量labels和空间图仅保留K=5/8/10/12/14/16，点大小统一为18。CSV中其他K的历史`labels_path`已随对应文件夹删除而失效。

### 5.1 Joint全K最佳内部指标

下表在现有K=2–16指标CSV中直接选择最优行；不重新聚类，也不恢复已删除的非绘图K文件夹。

| 方法         | 预处理       | best ASW K | ASW raw | ASW scaled | best CH K |        CH | best DBI K |    DBI | max section ARI K | section ARI | max section NMI K | section NMI |
| ------------ | ------------ | ---------: | ------: | ---------: | --------: | --------: | ---------: | -----: | ----------------: | ----------: | ----------------: | ----------: |
| spa_mo_model | Raw          |          5 |  0.1122 |     0.5561 |         2 |  898.6829 |         12 | 2.4861 |                 6 |      0.2894 |                 6 |      0.3737 |
| spa_mo_model | Standardized |          5 |  0.1073 |     0.5536 |         2 |  823.1615 |         13 | 2.4551 |                 7 |      0.2937 |                 7 |      0.3820 |
| MOFA+        | Raw          |          7 |  0.2107 |     0.6053 |         2 | 1499.5813 |          7 | 1.3198 |                16 |      0.0713 |                16 |      0.1716 |
| MOFA+        | Standardized |         11 |  0.1809 |     0.5904 |         2 |  892.4977 |         11 | 1.4165 |                16 |      0.0864 |                16 |      0.1842 |
| COSIE        | Raw          |          2 |  0.2229 |     0.6115 |         2 | 2259.7829 |          2 | 1.7323 |                11 |      0.2622 |                15 |      0.4190 |
| COSIE        | Standardized |          2 |  0.1733 |     0.5866 |         2 | 1541.9403 |         15 | 1.8008 |                13 |      0.2570 |                13 |      0.4237 |
| SpaMosaic    | Raw          |          2 |  0.4198 |     0.7099 |         2 | 6584.2731 |          2 | 0.9994 |                 3 |      0.0598 |                11 |      0.0837 |
| SpaMosaic    | Standardized |          2 |  0.3350 |     0.6675 |         2 | 3978.9140 |          2 | 1.2854 |                 3 |      0.0619 |                 8 |      0.0800 |

### 5.2 Independent逐section内部指标

展示保留聚类结果的K=5/8/10/12/14/16；同一方法、同一K、同一section的Raw与Standardized行严格相邻。ASW、CH和DBI均使用该section全量spot。

| 方法         |   K | section  | 预处理       | scope n | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |       CH |    DBI |
| ------------ | --: | -------- | ------------ | ------: | ----: | --------------: | -----------------: | -------: | -------: | -----: |
| spa_mo_model |   5 | dataset1 | Raw          |    2129 |  2129 |          0.0987 |             0.5494 |     2129 | 151.1517 | 2.7013 |
| spa_mo_model |   5 | dataset1 | Standardized |    2129 |  2129 |          0.0926 |             0.5463 |     2129 | 139.6224 | 2.7490 |
| spa_mo_model |   5 | dataset2 | Raw          |    1949 |  1949 |          0.1622 |             0.5811 |     1949 | 189.3379 | 2.2670 |
| spa_mo_model |   5 | dataset2 | Standardized |    1949 |  1949 |          0.1481 |             0.5741 |     1949 | 170.9517 | 2.2848 |
| spa_mo_model |   5 | dataset3 | Raw          |    1777 |  1777 |          0.0906 |             0.5453 |     1777 | 128.3537 | 2.9394 |
| spa_mo_model |   5 | dataset3 | Standardized |    1777 |  1777 |          0.0822 |             0.5411 |     1777 | 119.8304 | 3.0564 |
| spa_mo_model |   5 | dataset4 | Raw          |    1263 |  1263 |          0.1282 |             0.5641 |     1263 | 116.2715 | 2.6905 |
| spa_mo_model |   5 | dataset4 | Standardized |    1263 |  1263 |          0.1205 |             0.5603 |     1263 | 106.4057 | 2.8306 |
| spa_mo_model |   8 | dataset1 | Raw          |    2129 |  2129 |          0.0913 |             0.5457 |     2129 | 114.0103 | 2.9050 |
| spa_mo_model |   8 | dataset1 | Standardized |    2129 |  2129 |          0.0874 |             0.5437 |     2129 | 105.9379 | 2.9762 |
| spa_mo_model |   8 | dataset2 | Raw          |    1949 |  1949 |          0.1679 |             0.5840 |     1949 | 157.5140 | 1.8853 |
| spa_mo_model |   8 | dataset2 | Standardized |    1949 |  1949 |          0.1561 |             0.5780 |     1949 | 141.3345 | 2.2176 |
| spa_mo_model |   8 | dataset3 | Raw          |    1777 |  1777 |          0.0999 |             0.5500 |     1777 | 101.6316 | 2.5645 |
| spa_mo_model |   8 | dataset3 | Standardized |    1777 |  1777 |          0.0880 |             0.5440 |     1777 |  95.1655 | 2.6573 |
| spa_mo_model |   8 | dataset4 | Raw          |    1263 |  1263 |          0.1201 |             0.5600 |     1263 |  82.1238 | 2.8441 |
| spa_mo_model |   8 | dataset4 | Standardized |    1263 |  1263 |          0.1114 |             0.5557 |     1263 |  76.0808 | 2.8971 |
| spa_mo_model |  10 | dataset1 | Raw          |    2129 |  2129 |          0.0850 |             0.5425 |     2129 |  99.0981 | 2.9931 |
| spa_mo_model |  10 | dataset1 | Standardized |    2129 |  2129 |          0.0929 |             0.5464 |     2129 |  92.3697 | 2.8884 |
| spa_mo_model |  10 | dataset2 | Raw          |    1949 |  1949 |          0.1708 |             0.5854 |     1949 | 141.4027 | 2.1520 |
| spa_mo_model |  10 | dataset2 | Standardized |    1949 |  1949 |          0.1634 |             0.5817 |     1949 | 129.6445 | 2.1055 |
| spa_mo_model |  10 | dataset3 | Raw          |    1777 |  1777 |          0.0935 |             0.5467 |     1777 |  90.4157 | 2.7226 |
| spa_mo_model |  10 | dataset3 | Standardized |    1777 |  1777 |          0.0918 |             0.5459 |     1777 |  85.4175 | 2.7009 |
| spa_mo_model |  10 | dataset4 | Raw          |    1263 |  1263 |          0.0894 |             0.5447 |     1263 |  69.4641 | 2.9718 |
| spa_mo_model |  10 | dataset4 | Standardized |    1263 |  1263 |          0.0904 |             0.5452 |     1263 |  65.3619 | 2.8651 |
| spa_mo_model |  12 | dataset1 | Raw          |    2129 |  2129 |          0.0926 |             0.5463 |     2129 |  88.4752 | 2.9336 |
| spa_mo_model |  12 | dataset1 | Standardized |    2129 |  2129 |          0.0871 |             0.5436 |     2129 |  82.8311 | 2.9965 |
| spa_mo_model |  12 | dataset2 | Raw          |    1949 |  1949 |          0.1661 |             0.5830 |     1949 | 128.7637 | 2.1683 |
| spa_mo_model |  12 | dataset2 | Standardized |    1949 |  1949 |          0.1581 |             0.5790 |     1949 | 120.0849 | 2.1778 |
| spa_mo_model |  12 | dataset3 | Raw          |    1777 |  1777 |          0.0838 |             0.5419 |     1777 |  81.4449 | 2.7071 |
| spa_mo_model |  12 | dataset3 | Standardized |    1777 |  1777 |          0.0840 |             0.5420 |     1777 |  77.1715 | 2.6652 |
| spa_mo_model |  12 | dataset4 | Raw          |    1263 |  1263 |          0.0770 |             0.5385 |     1263 |  61.1628 | 2.9540 |
| spa_mo_model |  12 | dataset4 | Standardized |    1263 |  1263 |          0.0795 |             0.5397 |     1263 |  57.1867 | 2.8868 |
| spa_mo_model |  14 | dataset1 | Raw          |    2129 |  2129 |          0.0943 |             0.5472 |     2129 |  80.4927 | 2.9253 |
| spa_mo_model |  14 | dataset1 | Standardized |    2129 |  2129 |          0.0874 |             0.5437 |     2129 |  75.1242 | 2.9036 |
| spa_mo_model |  14 | dataset2 | Raw          |    1949 |  1949 |          0.1729 |             0.5865 |     1949 | 119.8726 | 2.0659 |
| spa_mo_model |  14 | dataset2 | Standardized |    1949 |  1949 |          0.1624 |             0.5812 |     1949 | 110.7728 | 2.1264 |
| spa_mo_model |  14 | dataset3 | Raw          |    1777 |  1777 |          0.0851 |             0.5426 |     1777 |  74.4611 | 2.6351 |
| spa_mo_model |  14 | dataset3 | Standardized |    1777 |  1777 |          0.0825 |             0.5413 |     1777 |  70.3035 | 2.6723 |
| spa_mo_model |  14 | dataset4 | Raw          |    1263 |  1263 |          0.0716 |             0.5358 |     1263 |  54.8663 | 2.8680 |
| spa_mo_model |  14 | dataset4 | Standardized |    1263 |  1263 |          0.0614 |             0.5307 |     1263 |  50.7655 | 3.0739 |
| spa_mo_model |  16 | dataset1 | Raw          |    2129 |  2129 |          0.0868 |             0.5434 |     2129 |  74.2671 | 2.8432 |
| spa_mo_model |  16 | dataset1 | Standardized |    2129 |  2129 |          0.0856 |             0.5428 |     2129 |  69.5687 | 2.8892 |
| spa_mo_model |  16 | dataset2 | Raw          |    1949 |  1949 |          0.1199 |             0.5599 |     1949 | 111.4195 | 2.2770 |
| spa_mo_model |  16 | dataset2 | Standardized |    1949 |  1949 |          0.1132 |             0.5566 |     1949 | 102.8277 | 2.3273 |
| spa_mo_model |  16 | dataset3 | Raw          |    1777 |  1777 |          0.0828 |             0.5414 |     1777 |  68.4963 | 2.6308 |
| spa_mo_model |  16 | dataset3 | Standardized |    1777 |  1777 |          0.0808 |             0.5404 |     1777 |  65.2362 | 2.6676 |
| spa_mo_model |  16 | dataset4 | Raw          |    1263 |  1263 |          0.0652 |             0.5326 |     1263 |  49.7191 | 2.8951 |
| spa_mo_model |  16 | dataset4 | Standardized |    1263 |  1263 |          0.0585 |             0.5293 |     1263 |  46.3961 | 2.9145 |
| MOFA+        |   5 | dataset1 | Raw          |    2129 |  2129 |          0.2685 |             0.6342 |     2129 | 630.3278 | 1.1979 |
| MOFA+        |   5 | dataset1 | Standardized |    2129 |  2129 |          0.2084 |             0.6042 |     2129 | 372.2465 | 1.6169 |
| MOFA+        |   5 | dataset2 | Raw          |    1949 |  1949 |          0.3020 |             0.6510 |     1949 | 737.4581 | 1.1247 |
| MOFA+        |   5 | dataset2 | Standardized |    1949 |  1949 |          0.1916 |             0.5958 |     1949 | 301.1109 | 1.6149 |
| MOFA+        |   5 | dataset3 | Raw          |    1777 |  1777 |          0.2256 |             0.6128 |     1777 | 568.0168 | 1.3362 |
| MOFA+        |   5 | dataset3 | Standardized |    1777 |  1777 |          0.1921 |             0.5960 |     1777 | 310.3169 | 1.5514 |
| MOFA+        |   5 | dataset4 | Raw          |    1263 |  1263 |          0.2811 |             0.6406 |     1263 | 443.7573 | 1.2657 |
| MOFA+        |   5 | dataset4 | Standardized |    1263 |  1263 |          0.2559 |             0.6279 |     1263 | 329.2775 | 1.3184 |
| MOFA+        |   8 | dataset1 | Raw          |    2129 |  2129 |          0.2311 |             0.6155 |     2129 | 540.3172 | 1.2616 |
| MOFA+        |   8 | dataset1 | Standardized |    2129 |  2129 |          0.2025 |             0.6012 |     2129 | 337.3696 | 1.4115 |
| MOFA+        |   8 | dataset2 | Raw          |    1949 |  1949 |          0.2556 |             0.6278 |     1949 | 637.9386 | 1.2459 |
| MOFA+        |   8 | dataset2 | Standardized |    1949 |  1949 |          0.1975 |             0.5987 |     1949 | 275.9082 | 1.5101 |
| MOFA+        |   8 | dataset3 | Raw          |    1777 |  1777 |          0.2274 |             0.6137 |     1777 | 490.8738 | 1.2959 |
| MOFA+        |   8 | dataset3 | Standardized |    1777 |  1777 |          0.1950 |             0.5975 |     1777 | 286.0727 | 1.4498 |
| MOFA+        |   8 | dataset4 | Raw          |    1263 |  1263 |          0.2518 |             0.6259 |     1263 | 395.7339 | 1.2160 |
| MOFA+        |   8 | dataset4 | Standardized |    1263 |  1263 |          0.2476 |             0.6238 |     1263 | 309.8517 | 1.3398 |
| MOFA+        |  10 | dataset1 | Raw          |    2129 |  2129 |          0.2091 |             0.6045 |     2129 | 488.7123 | 1.3813 |
| MOFA+        |  10 | dataset1 | Standardized |    2129 |  2129 |          0.1634 |             0.5817 |     2129 | 302.9533 | 1.5653 |
| MOFA+        |  10 | dataset2 | Raw          |    1949 |  1949 |          0.2269 |             0.6135 |     1949 | 583.6036 | 1.3037 |
| MOFA+        |  10 | dataset2 | Standardized |    1949 |  1949 |          0.1994 |             0.5997 |     1949 | 269.9460 | 1.3422 |
| MOFA+        |  10 | dataset3 | Raw          |    1777 |  1777 |          0.2193 |             0.6096 |     1777 | 452.1258 | 1.3226 |
| MOFA+        |  10 | dataset3 | Standardized |    1777 |  1777 |          0.1867 |             0.5934 |     1777 | 271.7243 | 1.4513 |
| MOFA+        |  10 | dataset4 | Raw          |    1263 |  1263 |          0.2515 |             0.6258 |     1263 | 363.8687 | 1.2042 |
| MOFA+        |  10 | dataset4 | Standardized |    1263 |  1263 |          0.2161 |             0.6080 |     1263 | 281.4530 | 1.3820 |
| MOFA+        |  12 | dataset1 | Raw          |    2129 |  2129 |          0.2066 |             0.6033 |     2129 | 453.7906 | 1.3561 |
| MOFA+        |  12 | dataset1 | Standardized |    2129 |  2129 |          0.1533 |             0.5766 |     2129 | 279.0838 | 1.5837 |
| MOFA+        |  12 | dataset2 | Raw          |    1949 |  1949 |          0.2199 |             0.6100 |     1949 | 541.8799 | 1.2854 |
| MOFA+        |  12 | dataset2 | Standardized |    1949 |  1949 |          0.1778 |             0.5889 |     1949 | 256.5952 | 1.3644 |
| MOFA+        |  12 | dataset3 | Raw          |    1777 |  1777 |          0.2089 |             0.6044 |     1777 | 407.4006 | 1.4139 |
| MOFA+        |  12 | dataset3 | Standardized |    1777 |  1777 |          0.1739 |             0.5870 |     1777 | 249.9192 | 1.5090 |
| MOFA+        |  12 | dataset4 | Raw          |    1263 |  1263 |          0.2488 |             0.6244 |     1263 | 348.4247 | 1.2287 |
| MOFA+        |  12 | dataset4 | Standardized |    1263 |  1263 |          0.1975 |             0.5988 |     1263 | 255.2151 | 1.4327 |
| MOFA+        |  14 | dataset1 | Raw          |    2129 |  2129 |          0.2077 |             0.6039 |     2129 | 428.4801 | 1.3680 |
| MOFA+        |  14 | dataset1 | Standardized |    2129 |  2129 |          0.1551 |             0.5775 |     2129 | 258.4446 | 1.6277 |
| MOFA+        |  14 | dataset2 | Raw          |    1949 |  1949 |          0.2309 |             0.6155 |     1949 | 512.8116 | 1.2168 |
| MOFA+        |  14 | dataset2 | Standardized |    1949 |  1949 |          0.1746 |             0.5873 |     1949 | 241.5637 | 1.4623 |
| MOFA+        |  14 | dataset3 | Raw          |    1777 |  1777 |          0.2012 |             0.6006 |     1777 | 380.4290 | 1.4401 |
| MOFA+        |  14 | dataset3 | Standardized |    1777 |  1777 |          0.1652 |             0.5826 |     1777 | 231.8126 | 1.5118 |
| MOFA+        |  14 | dataset4 | Raw          |    1263 |  1263 |          0.2382 |             0.6191 |     1263 | 336.1881 | 1.2496 |
| MOFA+        |  14 | dataset4 | Standardized |    1263 |  1263 |          0.1875 |             0.5937 |     1263 | 233.0693 | 1.4788 |
| MOFA+        |  16 | dataset1 | Raw          |    2129 |  2129 |          0.2074 |             0.6037 |     2129 | 402.1857 | 1.4338 |
| MOFA+        |  16 | dataset1 | Standardized |    2129 |  2129 |          0.1587 |             0.5793 |     2129 | 243.2661 | 1.5809 |
| MOFA+        |  16 | dataset2 | Raw          |    1949 |  1949 |          0.2193 |             0.6097 |     1949 | 482.1303 | 1.2660 |
| MOFA+        |  16 | dataset2 | Standardized |    1949 |  1949 |          0.1636 |             0.5818 |     1949 | 229.5970 | 1.5175 |
| MOFA+        |  16 | dataset3 | Raw          |    1777 |  1777 |          0.1801 |             0.5901 |     1777 | 356.8595 | 1.4425 |
| MOFA+        |  16 | dataset3 | Standardized |    1777 |  1777 |          0.1644 |             0.5822 |     1777 | 215.9596 | 1.5317 |
| MOFA+        |  16 | dataset4 | Raw          |    1263 |  1263 |          0.2346 |             0.6173 |     1263 | 319.5549 | 1.2487 |
| MOFA+        |  16 | dataset4 | Standardized |    1263 |  1263 |          0.1771 |             0.5886 |     1263 | 217.1339 | 1.5568 |
| COSIE        |   5 | dataset1 | Raw          |    2129 |  2129 |          0.2157 |             0.6079 |     2129 | 432.0182 | 1.6955 |
| COSIE        |   5 | dataset1 | Standardized |    2129 |  2129 |          0.1891 |             0.5945 |     2129 | 317.5323 | 1.9220 |
| COSIE        |   5 | dataset2 | Raw          |    1949 |  1949 |          0.2456 |             0.6228 |     1949 | 475.1894 | 1.6957 |
| COSIE        |   5 | dataset2 | Standardized |    1949 |  1949 |          0.2441 |             0.6220 |     1949 | 386.7542 | 1.7034 |
| COSIE        |   5 | dataset3 | Raw          |    1777 |  1777 |          0.1753 |             0.5876 |     1777 | 346.3909 | 1.8218 |
| COSIE        |   5 | dataset3 | Standardized |    1777 |  1777 |          0.1580 |             0.5790 |     1777 | 279.6866 | 1.9618 |
| COSIE        |   5 | dataset4 | Raw          |    1263 |  1263 |          0.2358 |             0.6179 |     1263 | 269.0094 | 1.3954 |
| COSIE        |   5 | dataset4 | Standardized |    1263 |  1263 |          0.2102 |             0.6051 |     1263 | 212.3954 | 1.5337 |
| COSIE        |   8 | dataset1 | Raw          |    2129 |  2129 |          0.2052 |             0.6026 |     2129 | 342.8179 | 1.6937 |
| COSIE        |   8 | dataset1 | Standardized |    2129 |  2129 |          0.1861 |             0.5930 |     2129 | 253.1140 | 1.8205 |
| COSIE        |   8 | dataset2 | Raw          |    1949 |  1949 |          0.2470 |             0.6235 |     1949 | 423.5070 | 1.4954 |
| COSIE        |   8 | dataset2 | Standardized |    1949 |  1949 |          0.2579 |             0.6290 |     1949 | 345.6845 | 1.5294 |
| COSIE        |   8 | dataset3 | Raw          |    1777 |  1777 |          0.1774 |             0.5887 |     1777 | 282.7480 | 1.6891 |
| COSIE        |   8 | dataset3 | Standardized |    1777 |  1777 |          0.1634 |             0.5817 |     1777 | 234.3078 | 1.8033 |
| COSIE        |   8 | dataset4 | Raw          |    1263 |  1263 |          0.1919 |             0.5960 |     1263 | 213.9964 | 1.6410 |
| COSIE        |   8 | dataset4 | Standardized |    1263 |  1263 |          0.1705 |             0.5853 |     1263 | 168.1931 | 1.7724 |
| COSIE        |  10 | dataset1 | Raw          |    2129 |  2129 |          0.1975 |             0.5988 |     2129 | 303.6814 | 1.6547 |
| COSIE        |  10 | dataset1 | Standardized |    2129 |  2129 |          0.1708 |             0.5854 |     2129 | 222.6061 | 1.8232 |
| COSIE        |  10 | dataset2 | Raw          |    1949 |  1949 |          0.2643 |             0.6321 |     1949 | 409.0577 | 1.3389 |
| COSIE        |  10 | dataset2 | Standardized |    1949 |  1949 |          0.2490 |             0.6245 |     1949 | 333.6832 | 1.4390 |
| COSIE        |  10 | dataset3 | Raw          |    1777 |  1777 |          0.1773 |             0.5887 |     1777 | 261.1807 | 1.6776 |
| COSIE        |  10 | dataset3 | Standardized |    1777 |  1777 |          0.1723 |             0.5862 |     1777 | 217.3906 | 1.7503 |
| COSIE        |  10 | dataset4 | Raw          |    1263 |  1263 |          0.1549 |             0.5774 |     1263 | 187.7147 | 1.8813 |
| COSIE        |  10 | dataset4 | Standardized |    1263 |  1263 |          0.1412 |             0.5706 |     1263 | 146.3182 | 2.0544 |
| COSIE        |  12 | dataset1 | Raw          |    2129 |  2129 |          0.1853 |             0.5926 |     2129 | 274.4479 | 1.7395 |
| COSIE        |  12 | dataset1 | Standardized |    2129 |  2129 |          0.1614 |             0.5807 |     2129 | 199.5937 | 1.9907 |
| COSIE        |  12 | dataset2 | Raw          |    1949 |  1949 |          0.2605 |             0.6303 |     1949 | 380.0009 | 1.4166 |
| COSIE        |  12 | dataset2 | Standardized |    1949 |  1949 |          0.2512 |             0.6256 |     1949 | 313.8609 | 1.4400 |
| COSIE        |  12 | dataset3 | Raw          |    1777 |  1777 |          0.1877 |             0.5939 |     1777 | 245.6448 | 1.5776 |
| COSIE        |  12 | dataset3 | Standardized |    1777 |  1777 |          0.1825 |             0.5913 |     1777 | 204.7306 | 1.6835 |
| COSIE        |  12 | dataset4 | Raw          |    1263 |  1263 |          0.1391 |             0.5695 |     1263 | 167.3728 | 1.9584 |
| COSIE        |  12 | dataset4 | Standardized |    1263 |  1263 |          0.1412 |             0.5706 |     1263 | 130.1056 | 1.9874 |
| COSIE        |  14 | dataset1 | Raw          |    2129 |  2129 |          0.1927 |             0.5964 |     2129 | 250.5909 | 1.6943 |
| COSIE        |  14 | dataset1 | Standardized |    2129 |  2129 |          0.1513 |             0.5757 |     2129 | 183.9614 | 2.1308 |
| COSIE        |  14 | dataset2 | Raw          |    1949 |  1949 |          0.2438 |             0.6219 |     1949 | 359.1520 | 1.4481 |
| COSIE        |  14 | dataset2 | Standardized |    1949 |  1949 |          0.2331 |             0.6165 |     1949 | 297.6095 | 1.5195 |
| COSIE        |  14 | dataset3 | Raw          |    1777 |  1777 |          0.2007 |             0.6004 |     1777 | 234.6514 | 1.5417 |
| COSIE        |  14 | dataset3 | Standardized |    1777 |  1777 |          0.1933 |             0.5967 |     1777 | 196.9505 | 1.6293 |
| COSIE        |  14 | dataset4 | Raw          |    1263 |  1263 |          0.1381 |             0.5690 |     1263 | 151.3804 | 1.9070 |
| COSIE        |  14 | dataset4 | Standardized |    1263 |  1263 |          0.1213 |             0.5607 |     1263 | 117.3839 | 2.1858 |
| COSIE        |  16 | dataset1 | Raw          |    2129 |  2129 |          0.1668 |             0.5834 |     2129 | 231.2926 | 1.9223 |
| COSIE        |  16 | dataset1 | Standardized |    2129 |  2129 |          0.1604 |             0.5802 |     2129 | 172.1891 | 2.0358 |
| COSIE        |  16 | dataset2 | Raw          |    1949 |  1949 |          0.2311 |             0.6155 |     1949 | 342.7446 | 1.4698 |
| COSIE        |  16 | dataset2 | Standardized |    1949 |  1949 |          0.2183 |             0.6091 |     1949 | 281.1973 | 1.5502 |
| COSIE        |  16 | dataset3 | Raw          |    1777 |  1777 |          0.2027 |             0.6014 |     1777 | 225.4633 | 1.5010 |
| COSIE        |  16 | dataset3 | Standardized |    1777 |  1777 |          0.2000 |             0.6000 |     1777 | 189.9896 | 1.5355 |
| COSIE        |  16 | dataset4 | Raw          |    1263 |  1263 |          0.1312 |             0.5656 |     1263 | 137.1440 | 1.9672 |
| COSIE        |  16 | dataset4 | Standardized |    1263 |  1263 |          0.1179 |             0.5590 |     1263 | 106.7098 | 2.0490 |
| SpaMosaic    |   5 | dataset1 | Raw          |    2129 |  2129 |          0.1897 |             0.5949 |     2129 | 839.2531 | 1.9033 |
| SpaMosaic    |   5 | dataset1 | Standardized |    2129 |  2129 |          0.1863 |             0.5931 |     2129 | 589.7049 | 1.9500 |
| SpaMosaic    |   5 | dataset2 | Raw          |    1949 |  1949 |          0.2536 |             0.6268 |     1949 | 985.2830 | 1.6296 |
| SpaMosaic    |   5 | dataset2 | Standardized |    1949 |  1949 |          0.2243 |             0.6121 |     1949 | 586.7422 | 1.7321 |
| SpaMosaic    |   5 | dataset3 | Raw          |    1777 |  1777 |          0.2324 |             0.6162 |     1777 | 601.5212 | 1.8164 |
| SpaMosaic    |   5 | dataset3 | Standardized |    1777 |  1777 |          0.2193 |             0.6097 |     1777 | 440.1142 | 1.8967 |
| SpaMosaic    |   5 | dataset4 | Raw          |    1263 |  1263 |          0.2951 |             0.6476 |     1263 | 540.4360 | 1.5427 |
| SpaMosaic    |   5 | dataset4 | Standardized |    1263 |  1263 |          0.2710 |             0.6355 |     1263 | 392.3616 | 1.6549 |
| SpaMosaic    |   8 | dataset1 | Raw          |    2129 |  2129 |          0.1784 |             0.5892 |     2129 | 635.2064 | 1.6885 |
| SpaMosaic    |   8 | dataset1 | Standardized |    2129 |  2129 |          0.1819 |             0.5910 |     2129 | 458.7861 | 1.6830 |
| SpaMosaic    |   8 | dataset2 | Raw          |    1949 |  1949 |          0.2201 |             0.6101 |     1949 | 717.1406 | 1.6101 |
| SpaMosaic    |   8 | dataset2 | Standardized |    1949 |  1949 |          0.2085 |             0.6042 |     1949 | 450.1292 | 1.6434 |
| SpaMosaic    |   8 | dataset3 | Raw          |    1777 |  1777 |          0.1506 |             0.5753 |     1777 | 442.9427 | 1.7668 |
| SpaMosaic    |   8 | dataset3 | Standardized |    1777 |  1777 |          0.1466 |             0.5733 |     1777 | 334.9071 | 1.8187 |
| SpaMosaic    |   8 | dataset4 | Raw          |    1263 |  1263 |          0.2281 |             0.6141 |     1263 | 394.8624 | 1.5817 |
| SpaMosaic    |   8 | dataset4 | Standardized |    1263 |  1263 |          0.1685 |             0.5842 |     1263 | 289.1881 | 1.8610 |
| SpaMosaic    |  10 | dataset1 | Raw          |    2129 |  2129 |          0.1698 |             0.5849 |     2129 | 542.8630 | 1.7994 |
| SpaMosaic    |  10 | dataset1 | Standardized |    2129 |  2129 |          0.1826 |             0.5913 |     2129 | 394.3496 | 1.6889 |
| SpaMosaic    |  10 | dataset2 | Raw          |    1949 |  1949 |          0.2103 |             0.6052 |     1949 | 620.9418 | 1.5615 |
| SpaMosaic    |  10 | dataset2 | Standardized |    1949 |  1949 |          0.1997 |             0.5998 |     1949 | 396.4007 | 1.6116 |
| SpaMosaic    |  10 | dataset3 | Raw          |    1777 |  1777 |          0.1537 |             0.5769 |     1777 | 384.8913 | 1.7633 |
| SpaMosaic    |  10 | dataset3 | Standardized |    1777 |  1777 |          0.1485 |             0.5742 |     1777 | 294.3362 | 1.8306 |
| SpaMosaic    |  10 | dataset4 | Raw          |    1263 |  1263 |          0.1676 |             0.5838 |     1263 | 338.9295 | 1.7776 |
| SpaMosaic    |  10 | dataset4 | Standardized |    1263 |  1263 |          0.1548 |             0.5774 |     1263 | 249.5509 | 1.7333 |
| SpaMosaic    |  12 | dataset1 | Raw          |    2129 |  2129 |          0.1650 |             0.5825 |     2129 | 474.2892 | 1.8246 |
| SpaMosaic    |  12 | dataset1 | Standardized |    2129 |  2129 |          0.1604 |             0.5802 |     2129 | 350.6369 | 1.8358 |
| SpaMosaic    |  12 | dataset2 | Raw          |    1949 |  1949 |          0.2003 |             0.6001 |     1949 | 553.8661 | 1.6213 |
| SpaMosaic    |  12 | dataset2 | Standardized |    1949 |  1949 |          0.1752 |             0.5876 |     1949 | 359.9228 | 1.6232 |
| SpaMosaic    |  12 | dataset3 | Raw          |    1777 |  1777 |          0.1428 |             0.5714 |     1777 | 338.4357 | 1.8612 |
| SpaMosaic    |  12 | dataset3 | Standardized |    1777 |  1777 |          0.1393 |             0.5697 |     1777 | 259.0912 | 1.9128 |
| SpaMosaic    |  12 | dataset4 | Raw          |    1263 |  1263 |          0.1596 |             0.5798 |     1263 | 298.5586 | 1.7622 |
| SpaMosaic    |  12 | dataset4 | Standardized |    1263 |  1263 |          0.1482 |             0.5741 |     1263 | 222.7034 | 1.8628 |
| SpaMosaic    |  14 | dataset1 | Raw          |    2129 |  2129 |          0.1579 |             0.5789 |     2129 | 424.5766 | 1.8482 |
| SpaMosaic    |  14 | dataset1 | Standardized |    2129 |  2129 |          0.1511 |             0.5756 |     2129 | 311.9377 | 1.9296 |
| SpaMosaic    |  14 | dataset2 | Raw          |    1949 |  1949 |          0.1753 |             0.5877 |     1949 | 501.4497 | 1.6745 |
| SpaMosaic    |  14 | dataset2 | Standardized |    1949 |  1949 |          0.1731 |             0.5866 |     1949 | 328.7079 | 1.6948 |
| SpaMosaic    |  14 | dataset3 | Raw          |    1777 |  1777 |          0.1404 |             0.5702 |     1777 | 305.3818 | 1.8924 |
| SpaMosaic    |  14 | dataset3 | Standardized |    1777 |  1777 |          0.1389 |             0.5694 |     1777 | 235.2796 | 1.9554 |
| SpaMosaic    |  14 | dataset4 | Raw          |    1263 |  1263 |          0.1564 |             0.5782 |     1263 | 268.4987 | 1.8334 |
| SpaMosaic    |  14 | dataset4 | Standardized |    1263 |  1263 |          0.1516 |             0.5758 |     1263 | 200.9549 | 1.8503 |
| SpaMosaic    |  16 | dataset1 | Raw          |    2129 |  2129 |          0.1488 |             0.5744 |     2129 | 383.9571 | 1.9092 |
| SpaMosaic    |  16 | dataset1 | Standardized |    2129 |  2129 |          0.1497 |             0.5749 |     2129 | 285.0596 | 1.9505 |
| SpaMosaic    |  16 | dataset2 | Raw          |    1949 |  1949 |          0.1720 |             0.5860 |     1949 | 455.5085 | 1.6905 |
| SpaMosaic    |  16 | dataset2 | Standardized |    1949 |  1949 |          0.1593 |             0.5796 |     1949 | 301.4185 | 1.6783 |
| SpaMosaic    |  16 | dataset3 | Raw          |    1777 |  1777 |          0.1295 |             0.5648 |     1777 | 280.5975 | 1.8926 |
| SpaMosaic    |  16 | dataset3 | Standardized |    1777 |  1777 |          0.1239 |             0.5619 |     1777 | 216.6694 | 1.8982 |
| SpaMosaic    |  16 | dataset4 | Raw          |    1263 |  1263 |          0.1488 |             0.5744 |     1263 | 244.9585 | 1.8726 |
| SpaMosaic    |  16 | dataset4 | Standardized |    1263 |  1263 |          0.1281 |             0.5640 |     1263 | 182.8881 | 1.9674 |

### 5.3 Joint与Independent空间连续性

展示保留聚类结果的K=5/8/10/12/14/16；每个值为四个section全量spot空间近邻同簇比例的算术平均，Raw与Standardized行严格相邻。

| 方法         |   K | 模式        | 预处理       | Spatial neighbor agreement |
| ------------ | --: | ----------- | ------------ | -------------------------: |
| spa_mo_model |   5 | Joint       | Raw          |                     0.7751 |
| spa_mo_model |   5 | Joint       | Standardized |                     0.7804 |
| spa_mo_model |   5 | Independent | Raw          |                     0.7019 |
| spa_mo_model |   5 | Independent | Standardized |                     0.7020 |
| spa_mo_model |   8 | Joint       | Raw          |                     0.7034 |
| spa_mo_model |   8 | Joint       | Standardized |                     0.7221 |
| spa_mo_model |   8 | Independent | Raw          |                     0.6256 |
| spa_mo_model |   8 | Independent | Standardized |                     0.6189 |
| spa_mo_model |  10 | Joint       | Raw          |                     0.6788 |
| spa_mo_model |  10 | Joint       | Standardized |                     0.6784 |
| spa_mo_model |  10 | Independent | Raw          |                     0.5555 |
| spa_mo_model |  10 | Independent | Standardized |                     0.5721 |
| spa_mo_model |  12 | Joint       | Raw          |                     0.6726 |
| spa_mo_model |  12 | Joint       | Standardized |                     0.6724 |
| spa_mo_model |  12 | Independent | Raw          |                     0.5225 |
| spa_mo_model |  12 | Independent | Standardized |                     0.5173 |
| spa_mo_model |  14 | Joint       | Raw          |                     0.6420 |
| spa_mo_model |  14 | Joint       | Standardized |                     0.6453 |
| spa_mo_model |  14 | Independent | Raw          |                     0.5032 |
| spa_mo_model |  14 | Independent | Standardized |                     0.4916 |
| spa_mo_model |  16 | Joint       | Raw          |                     0.6280 |
| spa_mo_model |  16 | Joint       | Standardized |                     0.6142 |
| spa_mo_model |  16 | Independent | Raw          |                     0.4675 |
| spa_mo_model |  16 | Independent | Standardized |                     0.4618 |
| MOFA+        |   5 | Joint       | Raw          |                     0.6415 |
| MOFA+        |   5 | Joint       | Standardized |                     0.6198 |
| MOFA+        |   5 | Independent | Raw          |                     0.5650 |
| MOFA+        |   5 | Independent | Standardized |                     0.6475 |
| MOFA+        |   8 | Joint       | Raw          |                     0.5001 |
| MOFA+        |   8 | Joint       | Standardized |                     0.5304 |
| MOFA+        |   8 | Independent | Raw          |                     0.4537 |
| MOFA+        |   8 | Independent | Standardized |                     0.5266 |
| MOFA+        |  10 | Joint       | Raw          |                     0.4326 |
| MOFA+        |  10 | Joint       | Standardized |                     0.5186 |
| MOFA+        |  10 | Independent | Raw          |                     0.3957 |
| MOFA+        |  10 | Independent | Standardized |                     0.4587 |
| MOFA+        |  12 | Joint       | Raw          |                     0.4257 |
| MOFA+        |  12 | Joint       | Standardized |                     0.4676 |
| MOFA+        |  12 | Independent | Raw          |                     0.3665 |
| MOFA+        |  12 | Independent | Standardized |                     0.4130 |
| MOFA+        |  14 | Joint       | Raw          |                     0.3858 |
| MOFA+        |  14 | Joint       | Standardized |                     0.4620 |
| MOFA+        |  14 | Independent | Raw          |                     0.3446 |
| MOFA+        |  14 | Independent | Standardized |                     0.3804 |
| MOFA+        |  16 | Joint       | Raw          |                     0.3917 |
| MOFA+        |  16 | Joint       | Standardized |                     0.4427 |
| MOFA+        |  16 | Independent | Raw          |                     0.3239 |
| MOFA+        |  16 | Independent | Standardized |                     0.3474 |
| COSIE        |   5 | Joint       | Raw          |                     0.8599 |
| COSIE        |   5 | Joint       | Standardized |                     0.8554 |
| COSIE        |   5 | Independent | Raw          |                     0.8038 |
| COSIE        |   5 | Independent | Standardized |                     0.8089 |
| COSIE        |   8 | Joint       | Raw          |                     0.7942 |
| COSIE        |   8 | Joint       | Standardized |                     0.8152 |
| COSIE        |   8 | Independent | Raw          |                     0.7488 |
| COSIE        |   8 | Independent | Standardized |                     0.7482 |
| COSIE        |  10 | Joint       | Raw          |                     0.7840 |
| COSIE        |  10 | Joint       | Standardized |                     0.7870 |
| COSIE        |  10 | Independent | Raw          |                     0.7076 |
| COSIE        |  10 | Independent | Standardized |                     0.7170 |
| COSIE        |  12 | Joint       | Raw          |                     0.7848 |
| COSIE        |  12 | Joint       | Standardized |                     0.7854 |
| COSIE        |  12 | Independent | Raw          |                     0.6930 |
| COSIE        |  12 | Independent | Standardized |                     0.6971 |
| COSIE        |  14 | Joint       | Raw          |                     0.7656 |
| COSIE        |  14 | Joint       | Standardized |                     0.7708 |
| COSIE        |  14 | Independent | Raw          |                     0.6763 |
| COSIE        |  14 | Independent | Standardized |                     0.6823 |
| COSIE        |  16 | Joint       | Raw          |                     0.7532 |
| COSIE        |  16 | Joint       | Standardized |                     0.7504 |
| COSIE        |  16 | Independent | Raw          |                     0.6559 |
| COSIE        |  16 | Independent | Standardized |                     0.6607 |
| SpaMosaic    |   5 | Joint       | Raw          |                     0.7397 |
| SpaMosaic    |   5 | Joint       | Standardized |                     0.7390 |
| SpaMosaic    |   5 | Independent | Raw          |                     0.7351 |
| SpaMosaic    |   5 | Independent | Standardized |                     0.7253 |
| SpaMosaic    |   8 | Joint       | Raw          |                     0.6689 |
| SpaMosaic    |   8 | Joint       | Standardized |                     0.6568 |
| SpaMosaic    |   8 | Independent | Raw          |                     0.6606 |
| SpaMosaic    |   8 | Independent | Standardized |                     0.6466 |
| SpaMosaic    |  10 | Joint       | Raw          |                     0.6213 |
| SpaMosaic    |  10 | Joint       | Standardized |                     0.6139 |
| SpaMosaic    |  10 | Independent | Raw          |                     0.6104 |
| SpaMosaic    |  10 | Independent | Standardized |                     0.6169 |
| SpaMosaic    |  12 | Joint       | Raw          |                     0.6011 |
| SpaMosaic    |  12 | Joint       | Standardized |                     0.6066 |
| SpaMosaic    |  12 | Independent | Raw          |                     0.5909 |
| SpaMosaic    |  12 | Independent | Standardized |                     0.5808 |
| SpaMosaic    |  14 | Joint       | Raw          |                     0.5671 |
| SpaMosaic    |  14 | Joint       | Standardized |                     0.5602 |
| SpaMosaic    |  14 | Independent | Raw          |                     0.5579 |
| SpaMosaic    |  14 | Independent | Standardized |                     0.5566 |
| SpaMosaic    |  16 | Joint       | Raw          |                     0.5380 |
| SpaMosaic    |  16 | Joint       | Standardized |                     0.5307 |
| SpaMosaic    |  16 | Independent | Raw          |                     0.5415 |
| SpaMosaic    |  16 | Independent | Standardized |                     0.5254 |

## 6. 两种预处理的标签稳定性

| 方法         |   K | raw vs standardized ARI |    NMI |
| ------------ | --: | ----------------------: | -----: |
| spa_mo_model |   5 |                  0.9369 | 0.9109 |
| spa_mo_model |   8 |                  0.7411 | 0.7904 |
| spa_mo_model |  10 |                  0.8102 | 0.8395 |
| spa_mo_model |  12 |                  0.9414 | 0.9372 |
| spa_mo_model |  14 |                  0.7556 | 0.8278 |
| spa_mo_model |  16 |                  0.7847 | 0.8389 |
| MOFA+        |   5 |                  0.5245 | 0.5864 |
| MOFA+        |   8 |                  0.5378 | 0.6205 |
| MOFA+        |  10 |                  0.4882 | 0.6158 |
| MOFA+        |  12 |                  0.5061 | 0.6330 |
| MOFA+        |  14 |                  0.4093 | 0.5894 |
| MOFA+        |  16 |                  0.4272 | 0.6145 |
| COSIE        |   5 |                  0.9112 | 0.8902 |
| COSIE        |   8 |                  0.5680 | 0.7060 |
| COSIE        |  10 |                  0.8584 | 0.8751 |
| COSIE        |  12 |                  0.8844 | 0.8955 |
| COSIE        |  14 |                  0.7807 | 0.8542 |
| COSIE        |  16 |                  0.8677 | 0.8998 |
| SpaMosaic    |   5 |                  0.8961 | 0.8509 |
| SpaMosaic    |   8 |                  0.8986 | 0.8680 |
| SpaMosaic    |  10 |                  0.8640 | 0.8481 |
| SpaMosaic    |  12 |                  0.7958 | 0.8069 |
| SpaMosaic    |  14 |                  0.8602 | 0.8590 |
| SpaMosaic    |  16 |                  0.5981 | 0.7532 |

完整joint和independent K=2–16稳定性见各方法 `comparison_metrics/label_stability.csv`；内部和外部指标差值见同目录另外两个CSV。

## 7. 共享指标及参数

### 7.1 Batch样本范围

| 方法         | batch key | n total | n used | n batches | batch categories                    | batch counts                                                             | max samples | requested n | seed | embedding scaled | bASW n |
| ------------ | --------- | ------: | -----: | --------: | ----------------------------------- | ------------------------------------------------------------------------ | ----------: | ----------: | ---: | ---------------- | -----: |
| spa_mo_model | section   |    7118 |   7118 |         4 | dataset1,dataset2,dataset3,dataset4 | {'dataset1': 2129, 'dataset2': 1949, 'dataset3': 1777, 'dataset4': 1263} |           0 |           0 |    0 | True             |   7118 |
| MOFA+        | group     |    7118 |   7118 |         4 | dataset1,dataset2,dataset3,dataset4 | NA                                                                       |       50000 |           0 |    0 | NA               |   7118 |
| COSIE        | group     |    7118 |   7118 |         4 | dataset1,dataset2,dataset3,dataset4 | NA                                                                       |       50000 |           0 |    0 | NA               |   7118 |
| SpaMosaic    | group     |    7118 |   7118 |         4 | dataset1,dataset2,dataset3,dataset4 | NA                                                                       |       50000 |           0 |    0 | NA               |   7118 |

### 7.2 Batch近邻与PCR参数

| 方法         | kNN backend   | bLISI k | kBET k |  alpha | PCR PCs |
| ------------ | ------------- | ------: | -----: | -----: | ------: |
| spa_mo_model | sklearn_exact |      90 |     50 | 0.0500 |      50 |
| MOFA+        | sklearn_exact |      90 |     50 | 0.0500 |      10 |
| COSIE        | sklearn_exact |      90 |     50 | 0.0500 |      50 |
| SpaMosaic    | sklearn_exact |      90 |     50 | 0.0500 |      32 |

### 7.3 Batch完整指标值

| 方法         | bASW raw | bASW score | bLISI raw | bLISI normalized | kBET rejection | kBET acceptance | PCR batch R2 | PCR score |
| ------------ | -------: | ---------: | --------: | ---------------: | -------------: | --------------: | -----------: | --------: |
| spa_mo_model |   0.0433 |     0.9567 |    1.8697 |           0.2899 |         0.9666 |          0.0334 |       0.1375 |    0.8625 |
| MOFA+        |  -0.0356 |     0.9644 |    1.8663 |           0.2888 |         0.9900 |          0.0100 |       0.0000 |    1.0000 |
| COSIE        |   0.0642 |     0.9358 |    1.3028 |           0.1009 |         0.9999 |          0.0001 |       0.1815 |    0.8185 |
| SpaMosaic    |  -0.0348 |     0.9652 |    2.6201 |           0.5400 |         0.8969 |          0.1031 |       0.0564 |    0.9436 |

batch指标不随本次KMeans输入分支变化，故只保存和展示一份；来源文件与复制文件SHA256见 `shared_metrics/source_manifest.csv`。

### 7.4 MOFA+视图解释度R²

| View | Group    |      R2 |
| ---- | -------- | ------: |
| ATAC | dataset1 | 17.2195 |
| ATAC | dataset2 | 21.9969 |
| ATAC | dataset3 | 18.6074 |
| ATAC | dataset4 | 16.2670 |
| RNA  | dataset1 |  5.0213 |
| RNA  | dataset2 |  6.6174 |
| RNA  | dataset3 |  5.1054 |
| RNA  | dataset4 |  6.6154 |

该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

### 7.5 SpaMosaic模态对齐

| group    | n obs | RNA-ATAC cosine mean | RNA-ATAC cosine median |
| -------- | ----: | -------------------: | ---------------------: |
| ALL      |  7118 |               0.0393 |                 0.0327 |
| dataset1 |  2129 |               0.0435 |                 0.0406 |
| dataset2 |  1949 |               0.0390 |                 0.0258 |
| dataset3 |  1777 |               0.0426 |                 0.0356 |
| dataset4 |  1263 |               0.0278 |                 0.0244 |

该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

## 8. 结论与使用建议

1. Raw与Standardized均保留joint/independent K=2–16指标CSV；聚类文件夹与全量labels仅保留K=5/8/10/12/14/16。
2. 跨方法排名必须固定同一预处理分支；建议Standardized作为主分析，Raw作为敏感性分析。
3. 同时报告raw-vs-standardized ARI/NMI，以判断预处理对聚类分配的影响。
4. 生物标签外部指标与Label ASW使用全部7118个spot；section ARI/NMI仅是数据来源依赖诊断。
