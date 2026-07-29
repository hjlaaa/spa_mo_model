# Mouse_Thymus：Raw 与 Standardized embedding + KMeans 对比报告

- 生成时间：2026-07-27T14:56:23.509674+08:00
- 本报告只引用新增 comparison 结果；旧分析目录和原综合报告未覆盖。
- Raw与Standardized分别在自身KMeans输入空间计算距离型指标；跨方案变化需结合标签稳定性解释。

## 1. 新结果目录

| 方法         | comparison root                                                             | 方法自有数据副本                                    |
| ------------ | --------------------------------------------------------------------------- | --------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_thymus_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/Mouse_Thymus       |
| MOFA+        | /home/hujinlan/mofa+/analysis/mouse_thymus_preprocessing_comparison         | /home/hujinlan/mofa+/data/Mouse_Thymus              |
| COSIE        | /home/hujinlan/cosie_runs/mouse_thymus_preprocessing_comparison             | /home/hujinlan/cosie/data/Mouse_Thymus              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mouse_thymus_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/Mouse_Thymus |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。

## 2. 统一标准

| 项目                            | 统一值                                                  |
| ------------------------------- | ------------------------------------------------------- |
| spot/barcode                    | 各方法自有数据；section + original barcode精确对齐      |
| spot数                          | 17824（4697/4253/4646/4228）                            |
| embedding                       | 各方法当前最终embedding                                 |
| raw分支                         | 原始最终embedding直接输入KMeans                         |
| standardized分支                | StandardScaler；joint全体拟合，independent每section拟合 |
| KMeans                          | sklearn exact KMeans                                    |
| joint/independent指标K          | 2–12                                                    |
| 保留的聚类文件夹/labels/空间图K | 5、8、10、12                                            |
| 空间图点大小                    | Mouse_Thymus1=8；Mouse_Thymus2/3/4=5                    |
| 随机参数                        | seed=0，n_init=20，max_iter=300                         |
| Cluster ASW                     | 固定同一批section分层10000个spot                        |
| ASW分层数                       | 2635/2386/2607/2372                                     |
| CH/DBI                          | 每个scope全量样本                                       |
| Label ASW                       | 无真实细胞/区域标签，不计算                             |
| 空间连续性                      | 每section全点，exact 6-NN                               |
| labels复用                      | 保留K的内部、空间指标和图复用保存的全量labels           |
| batch指标                       | KMeans分支无关；复制原结果并记录SHA256                  |

固定ASW名单保存在每个根目录 `shared_metrics/asw_sample.csv`；四份文件SHA256完全相同：

`00a6ede1cd4ca0d5a5d8af355561899d0cb1696745742e5ba887324e12db0c8e`

## 3. 数据与 embedding 对齐

| 方法         | 维度 | Thymus1 | Thymus2 | Thymus3 | Thymus4 | 总spot |
| ------------ | ---: | ------: | ------: | ------: | ------: | -----: |
| spa_mo_model |  128 |    4697 |    4253 |    4646 |    4228 |  17824 |
| MOFA+        |   10 |    4697 |    4253 |    4646 |    4228 |  17824 |
| COSIE        |  256 |    4697 |    4253 |    4646 |    4228 |  17824 |
| SpaMosaic    |   32 |    4697 |    4253 |    4646 |    4228 |  17824 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Thymus1.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_Mouse_Thymus1.csv`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Thymus2.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_Mouse_Thymus2.csv`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Thymus3.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_Mouse_Thymus3.csv`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Thymus4.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/obs_metadata_Mouse_Thymus4.csv`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/mouse_thymus_mofa_hvg2000_k10_iter1000/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/analysis/tables/embeddings_with_metadata.csv`
  - `/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/final_embeddings/Mouse_Thymus1_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/final_embeddings/Mouse_Thymus2_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/final_embeddings/Mouse_Thymus3_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/final_embeddings/Mouse_Thymus4_final_embedding.npy`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/mouse_thymus_spamosaic/mouse_thymus_spamosaic_embeddings.h5ad`

## 4. Joint内部与空间指标：Raw / Standardized上下对齐

同一方法、同一K的Raw与Standardized行相邻，所有列采用完全相同定义。

| 方法         |   K | 预处理       | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |        CH |    DBI | section ARI | section NMI | section ASW n | section ASW raw | joint spatial |
| ------------ | --: | ------------ | ----: | --------------: | -----------------: | -------: | --------: | -----: | ----------: | ----------: | ------------: | --------------: | ------------: |
| spa_mo_model |   5 | Raw          | 10000 |          0.3050 |             0.6525 |    17824 | 6098.4811 | 1.9051 |      0.3070 |      0.3826 |         10000 |          0.1170 |        0.6868 |
| spa_mo_model |   5 | Standardized | 10000 |          0.2690 |             0.6345 |    17824 | 3833.3540 | 2.1201 |      0.3047 |      0.3779 |         10000 |          0.0953 |        0.6781 |
| spa_mo_model |   8 | Raw          | 10000 |          0.2687 |             0.6344 |    17824 | 4215.1715 | 1.8422 |      0.2429 |      0.3508 |         10000 |          0.1170 |        0.5820 |
| spa_mo_model |   8 | Standardized | 10000 |          0.2366 |             0.6183 |    17824 | 2667.6360 | 1.9568 |      0.2454 |      0.3488 |         10000 |          0.0953 |        0.5701 |
| spa_mo_model |  10 | Raw          | 10000 |          0.2142 |             0.6071 |    17824 | 3564.2424 | 2.0765 |      0.2373 |      0.3241 |         10000 |          0.1170 |        0.4891 |
| spa_mo_model |  10 | Standardized | 10000 |          0.2020 |             0.6010 |    17824 | 2284.2243 | 2.0785 |      0.2385 |      0.3253 |         10000 |          0.0953 |        0.5094 |
| spa_mo_model |  12 | Raw          | 10000 |          0.2009 |             0.6004 |    17824 | 3098.1022 | 2.0299 |      0.2356 |      0.3129 |         10000 |          0.1170 |        0.4683 |
| spa_mo_model |  12 | Standardized | 10000 |          0.1687 |             0.5844 |    17824 | 2009.5916 | 2.1161 |      0.2374 |      0.3097 |         10000 |          0.0953 |        0.4481 |
| MOFA+        |   5 | Raw          | 10000 |          0.3201 |             0.6601 |    17824 | 6472.8227 | 0.9957 |      0.0298 |      0.1025 |         10000 |         -0.0137 |        0.7841 |
| MOFA+        |   5 | Standardized | 10000 |          0.1178 |             0.5589 |    17824 | 2224.5687 | 1.8277 |      0.0057 |      0.0159 |         10000 |         -0.0268 |        0.3832 |
| MOFA+        |   8 | Raw          | 10000 |          0.2295 |             0.6148 |    17824 | 5930.0885 | 1.1766 |      0.0283 |      0.0775 |         10000 |         -0.0137 |        0.4993 |
| MOFA+        |   8 | Standardized | 10000 |          0.1097 |             0.5548 |    17824 | 1895.3246 | 1.7297 |      0.0132 |      0.0397 |         10000 |         -0.0268 |        0.2704 |
| MOFA+        |  10 | Raw          | 10000 |          0.2284 |             0.6142 |    17824 | 5379.3663 | 1.1559 |      0.0462 |      0.1027 |         10000 |         -0.0137 |        0.4553 |
| MOFA+        |  10 | Standardized | 10000 |          0.1102 |             0.5551 |    17824 | 1721.4515 | 1.5828 |      0.0130 |      0.0407 |         10000 |         -0.0268 |        0.2469 |
| MOFA+        |  12 | Raw          | 10000 |          0.2150 |             0.6075 |    17824 | 4935.4811 | 1.2220 |      0.0525 |      0.1069 |         10000 |         -0.0137 |        0.4130 |
| MOFA+        |  12 | Standardized | 10000 |          0.1032 |             0.5516 |    17824 | 1585.5022 | 1.6141 |      0.0149 |      0.0435 |         10000 |         -0.0268 |        0.1996 |
| COSIE        |   5 | Raw          | 10000 |          0.2605 |             0.6302 |    17824 | 8172.7910 | 2.0520 |      0.3591 |      0.3914 |         10000 |          0.1440 |        0.8021 |
| COSIE        |   5 | Standardized | 10000 |          0.1979 |             0.5989 |    17824 | 3668.4683 | 2.3284 |      0.3591 |      0.3913 |         10000 |          0.0942 |        0.8068 |
| COSIE        |   8 | Raw          | 10000 |          0.1093 |             0.5547 |    17824 | 5244.5223 | 2.7510 |      0.2173 |      0.3374 |         10000 |          0.1440 |        0.6210 |
| COSIE        |   8 | Standardized | 10000 |          0.0946 |             0.5473 |    17824 | 2401.7649 | 3.0267 |      0.2164 |      0.3368 |         10000 |          0.0942 |        0.6165 |
| COSIE        |  10 | Raw          | 10000 |          0.1099 |             0.5550 |    17824 | 4284.0918 | 2.6195 |      0.1746 |      0.3226 |         10000 |          0.1440 |        0.5763 |
| COSIE        |  10 | Standardized | 10000 |          0.0954 |             0.5477 |    17824 | 1995.6007 | 2.8620 |      0.1752 |      0.3225 |         10000 |          0.0942 |        0.5655 |
| COSIE        |  12 | Raw          | 10000 |          0.1011 |             0.5505 |    17824 | 3657.6388 | 2.7375 |      0.1703 |      0.3080 |         10000 |          0.1440 |        0.5344 |
| COSIE        |  12 | Standardized | 10000 |          0.0884 |             0.5442 |    17824 | 1723.1975 | 2.9140 |      0.1681 |      0.3034 |         10000 |          0.0942 |        0.5320 |
| SpaMosaic    |   5 | Raw          | 10000 |          0.2196 |             0.6098 |    17824 | 6605.3450 | 1.8651 |      0.2450 |      0.3904 |         10000 |          0.1103 |        0.6962 |
| SpaMosaic    |   5 | Standardized | 10000 |          0.1938 |             0.5969 |    17824 | 5030.7365 | 2.0666 |      0.2459 |      0.3897 |         10000 |          0.0971 |        0.6876 |
| SpaMosaic    |   8 | Raw          | 10000 |          0.1633 |             0.5816 |    17824 | 4486.3985 | 2.0739 |      0.2310 |      0.3345 |         10000 |          0.1103 |        0.5194 |
| SpaMosaic    |   8 | Standardized | 10000 |          0.1481 |             0.5741 |    17824 | 3482.8113 | 2.1133 |      0.2323 |      0.3346 |         10000 |          0.0971 |        0.5103 |
| SpaMosaic    |  10 | Raw          | 10000 |          0.1154 |             0.5577 |    17824 | 3720.9752 | 2.3010 |      0.1595 |      0.3119 |         10000 |          0.1103 |        0.4393 |
| SpaMosaic    |  10 | Standardized | 10000 |          0.1194 |             0.5597 |    17824 | 2921.3068 | 2.2854 |      0.1614 |      0.3129 |         10000 |          0.0971 |        0.4266 |
| SpaMosaic    |  12 | Raw          | 10000 |          0.1095 |             0.5548 |    17824 | 3217.6369 | 2.3245 |      0.1388 |      0.3010 |         10000 |          0.1103 |        0.3976 |
| SpaMosaic    |  12 | Standardized | 10000 |          0.1096 |             0.5548 |    17824 | 2532.7399 | 2.2143 |      0.1298 |      0.2990 |         10000 |          0.0971 |        0.3887 |

完整joint/independent K=2–12内部与空间指标仍保存在各分支 `metrics/`；聚类文件夹、全量labels和空间图仅保留K=5/8/10/12。Mouse_Thymus1点大小为8，Mouse_Thymus2/3/4点大小为5。CSV中其他K的历史`labels_path`已随对应文件夹删除而失效。

### 4.1 Joint全K最佳内部指标

下表在现有joint K=2–12指标CSV中直接选择最优行，不重新聚类，也不恢复已删除的非绘图K文件夹。

| 方法         | 预处理       | best ASW K | ASW raw | ASW scaled | best CH K |         CH | best DBI K |    DBI | max section ARI K | section ARI | max section NMI K | section NMI |
| ------------ | ------------ | ---------: | ------: | ---------: | --------: | ---------: | ---------: | -----: | ----------------: | ----------: | ----------------: | ----------: |
| spa_mo_model | Raw          |          2 |  0.3718 |     0.6859 |         3 |  9551.7021 |          2 | 1.0914 |                 3 |      0.3348 |                 2 |      0.5567 |
| spa_mo_model | Standardized |          2 |  0.2959 |     0.6479 |         2 |  6172.0358 |          2 | 1.3582 |                 3 |      0.3352 |                 2 |      0.5578 |
| MOFA+        | Raw          |          2 |  0.7026 |     0.8513 |         2 |  7907.1544 |          2 | 0.7720 |                12 |      0.0525 |                12 |      0.1069 |
| MOFA+        | Standardized |          4 |  0.1677 |     0.5838 |         2 |  3264.2015 |         10 | 1.5828 |                12 |      0.0149 |                12 |      0.0435 |
| COSIE        | Raw          |          2 |  0.4560 |     0.7280 |         2 | 15900.9502 |          2 | 0.8624 |                 7 |      0.3609 |                 2 |      0.5879 |
| COSIE        | Standardized |          2 |  0.2675 |     0.6338 |         2 |  6018.8319 |          2 | 1.4309 |                 6 |      0.3598 |                 2 |      0.5879 |
| SpaMosaic    | Raw          |          2 |  0.4023 |     0.7011 |         2 | 11677.3550 |          2 | 1.0656 |                 3 |      0.3508 |                 2 |      0.5778 |
| SpaMosaic    | Standardized |          2 |  0.3506 |     0.6753 |         2 |  8992.1247 |          2 | 1.2155 |                 4 |      0.3535 |                 2 |      0.5786 |

### 4.2 Independent逐section内部指标

展示保留聚类结果的K=5/8/10/12；同一方法、同一K、同一section的Raw与Standardized行严格相邻。ASW使用固定分层样本在对应section中的子集，CH/DBI使用该section全量spot。

| 方法         |   K | section       | 预处理       | scope n | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |        CH |    DBI |
| ------------ | --: | ------------- | ------------ | ------: | ----: | --------------: | -----------------: | -------: | --------: | -----: |
| spa_mo_model |   5 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.2609 |             0.6304 |     4697 |  911.4110 | 2.0271 |
| spa_mo_model |   5 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.2433 |             0.6216 |     4697 |  706.2356 | 2.2160 |
| spa_mo_model |   5 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1668 |             0.5834 |     4253 |  840.0179 | 2.5198 |
| spa_mo_model |   5 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.2066 |             0.6033 |     4253 |  539.7652 | 2.6976 |
| spa_mo_model |   5 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.2208 |             0.6104 |     4646 |  940.2655 | 2.1887 |
| spa_mo_model |   5 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1880 |             0.5940 |     4646 |  621.9209 | 2.3670 |
| spa_mo_model |   5 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.2329 |             0.6164 |     4228 |  845.4683 | 2.1220 |
| spa_mo_model |   5 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1869 |             0.5935 |     4228 |  556.2305 | 2.3595 |
| spa_mo_model |   8 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0707 |             0.5353 |     4697 |  636.0333 | 2.3815 |
| spa_mo_model |   8 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.1940 |             0.5970 |     4697 |  502.8967 | 2.5431 |
| spa_mo_model |   8 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1636 |             0.5818 |     4253 |  584.4502 | 2.3254 |
| spa_mo_model |   8 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.0874 |             0.5437 |     4253 |  393.9093 | 2.5731 |
| spa_mo_model |   8 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1871 |             0.5936 |     4646 |  669.0564 | 2.0672 |
| spa_mo_model |   8 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1653 |             0.5826 |     4646 |  462.8488 | 2.1664 |
| spa_mo_model |   8 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1836 |             0.5918 |     4228 |  592.2232 | 2.1719 |
| spa_mo_model |   8 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1584 |             0.5792 |     4228 |  408.1776 | 2.2802 |
| spa_mo_model |  10 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0733 |             0.5367 |     4697 |  538.3845 | 2.3654 |
| spa_mo_model |  10 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0741 |             0.5370 |     4697 |  429.6961 | 2.3262 |
| spa_mo_model |  10 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1025 |             0.5513 |     4253 |  499.0872 | 2.3211 |
| spa_mo_model |  10 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1042 |             0.5521 |     4253 |  345.6104 | 2.2900 |
| spa_mo_model |  10 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1631 |             0.5816 |     4646 |  577.8736 | 2.1434 |
| spa_mo_model |  10 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1344 |             0.5672 |     4646 |  406.3448 | 2.2174 |
| spa_mo_model |  10 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1481 |             0.5741 |     4228 |  507.1007 | 2.1687 |
| spa_mo_model |  10 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1044 |             0.5522 |     4228 |  355.6077 | 2.3099 |
| spa_mo_model |  12 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0741 |             0.5371 |     4697 |  470.4343 | 2.4167 |
| spa_mo_model |  12 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0725 |             0.5362 |     4697 |  379.1031 | 2.2709 |
| spa_mo_model |  12 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.0975 |             0.5487 |     4253 |  444.4568 | 2.1885 |
| spa_mo_model |  12 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1011 |             0.5505 |     4253 |  314.8956 | 2.2022 |
| spa_mo_model |  12 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1359 |             0.5680 |     4646 |  514.9499 | 2.1208 |
| spa_mo_model |  12 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1459 |             0.5730 |     4646 |  368.9765 | 2.1273 |
| spa_mo_model |  12 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.0976 |             0.5488 |     4228 |  448.6819 | 2.2556 |
| spa_mo_model |  12 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1053 |             0.5527 |     4228 |  315.8699 | 2.2450 |
| MOFA+        |   5 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.2676 |             0.6338 |     4697 | 2330.7302 | 1.1747 |
| MOFA+        |   5 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.2079 |             0.6040 |     4697 | 1028.7902 | 1.2447 |
| MOFA+        |   5 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.2354 |             0.6177 |     4253 | 2331.0872 | 1.2708 |
| MOFA+        |   5 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1296 |             0.5648 |     4253 |  578.2524 | 1.8015 |
| MOFA+        |   5 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.2700 |             0.6350 |     4646 | 2904.4736 | 1.1703 |
| MOFA+        |   5 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1125 |             0.5562 |     4646 |  695.7684 | 1.8197 |
| MOFA+        |   5 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.2728 |             0.6364 |     4228 | 2778.1669 | 1.1826 |
| MOFA+        |   5 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1353 |             0.5676 |     4228 |  636.8800 | 1.7277 |
| MOFA+        |   8 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.2318 |             0.6159 |     4697 | 1950.2183 | 1.2244 |
| MOFA+        |   8 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.1668 |             0.5834 |     4697 |  863.5083 | 1.3772 |
| MOFA+        |   8 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.2188 |             0.6094 |     4253 | 1953.5748 | 1.2496 |
| MOFA+        |   8 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1184 |             0.5592 |     4253 |  455.3727 | 1.7942 |
| MOFA+        |   8 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.2639 |             0.6319 |     4646 | 2361.5348 | 1.1266 |
| MOFA+        |   8 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1153 |             0.5577 |     4646 |  565.9307 | 1.6785 |
| MOFA+        |   8 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.2683 |             0.6342 |     4228 | 2266.7141 | 1.1304 |
| MOFA+        |   8 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1245 |             0.5622 |     4228 |  519.0037 | 1.6864 |
| MOFA+        |  10 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.2190 |             0.6095 |     4697 | 1803.1909 | 1.2423 |
| MOFA+        |  10 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.1643 |             0.5822 |     4697 |  778.2085 | 1.2425 |
| MOFA+        |  10 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.2013 |             0.6007 |     4253 | 1743.6437 | 1.2735 |
| MOFA+        |  10 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1163 |             0.5581 |     4253 |  408.1658 | 1.6981 |
| MOFA+        |  10 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1951 |             0.5976 |     4646 | 2101.6249 | 1.3008 |
| MOFA+        |  10 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1130 |             0.5565 |     4646 |  504.7894 | 1.8184 |
| MOFA+        |  10 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1893 |             0.5946 |     4228 | 2028.0621 | 1.3056 |
| MOFA+        |  10 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1152 |             0.5576 |     4228 |  464.4257 | 1.6317 |
| MOFA+        |  12 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.2165 |             0.6082 |     4697 | 1682.8956 | 1.2199 |
| MOFA+        |  12 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.1650 |             0.5825 |     4697 |  737.7780 | 1.2359 |
| MOFA+        |  12 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1757 |             0.5878 |     4253 | 1576.5303 | 1.3774 |
| MOFA+        |  12 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1109 |             0.5555 |     4253 |  373.9943 | 1.7884 |
| MOFA+        |  12 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1871 |             0.5935 |     4646 | 1933.3735 | 1.2720 |
| MOFA+        |  12 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1131 |             0.5565 |     4646 |  462.4501 | 1.7412 |
| MOFA+        |  12 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1919 |             0.5959 |     4228 | 1881.4366 | 1.3141 |
| MOFA+        |  12 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1156 |             0.5578 |     4228 |  427.1902 | 1.7135 |
| COSIE        |   5 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0756 |             0.5378 |     4697 |  289.3224 | 2.8948 |
| COSIE        |   5 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0886 |             0.5443 |     4697 |  501.5056 | 2.4663 |
| COSIE        |   5 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1447 |             0.5724 |     4253 |  813.9263 | 2.4975 |
| COSIE        |   5 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1416 |             0.5708 |     4253 |  609.8747 | 2.4495 |
| COSIE        |   5 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1476 |             0.5738 |     4646 |  846.9676 | 2.3320 |
| COSIE        |   5 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1429 |             0.5714 |     4646 |  673.4270 | 2.3277 |
| COSIE        |   5 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1537 |             0.5768 |     4228 |  852.2463 | 2.3332 |
| COSIE        |   5 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1464 |             0.5732 |     4228 |  657.5344 | 2.3231 |
| COSIE        |   8 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0789 |             0.5394 |     4697 |  219.5561 | 2.9771 |
| COSIE        |   8 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0676 |             0.5338 |     4697 |  361.0463 | 2.6427 |
| COSIE        |   8 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1259 |             0.5630 |     4253 |  558.2300 | 2.4831 |
| COSIE        |   8 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1124 |             0.5562 |     4253 |  438.5160 | 2.4770 |
| COSIE        |   8 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1033 |             0.5517 |     4646 |  575.7688 | 2.6016 |
| COSIE        |   8 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1028 |             0.5514 |     4646 |  476.9591 | 2.5260 |
| COSIE        |   8 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1236 |             0.5618 |     4228 |  576.1793 | 2.4183 |
| COSIE        |   8 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1203 |             0.5602 |     4228 |  459.0250 | 2.4841 |
| COSIE        |  10 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0812 |             0.5406 |     4697 |  193.7247 | 2.9047 |
| COSIE        |  10 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0632 |             0.5316 |     4697 |  308.4530 | 2.7441 |
| COSIE        |  10 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1190 |             0.5595 |     4253 |  466.7428 | 2.5346 |
| COSIE        |  10 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.0924 |             0.5462 |     4253 |  375.6261 | 2.5108 |
| COSIE        |  10 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.0966 |             0.5483 |     4646 |  483.5816 | 2.6194 |
| COSIE        |  10 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.0930 |             0.5465 |     4646 |  402.8950 | 2.6110 |
| COSIE        |  10 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.0861 |             0.5430 |     4228 |  480.1949 | 2.6972 |
| COSIE        |  10 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1095 |             0.5547 |     4228 |  389.0946 | 2.4275 |
| COSIE        |  12 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0898 |             0.5449 |     4697 |  173.7285 | 2.7329 |
| COSIE        |  12 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0605 |             0.5302 |     4697 |  269.1189 | 2.7945 |
| COSIE        |  12 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.0894 |             0.5447 |     4253 |  403.7341 | 2.4949 |
| COSIE        |  12 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.0950 |             0.5475 |     4253 |  332.2078 | 2.4288 |
| COSIE        |  12 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1005 |             0.5502 |     4646 |  417.9343 | 2.5458 |
| COSIE        |  12 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.0870 |             0.5435 |     4646 |  352.7038 | 2.6288 |
| COSIE        |  12 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.0881 |             0.5440 |     4228 |  415.7946 | 2.6058 |
| COSIE        |  12 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.0918 |             0.5459 |     4228 |  340.6055 | 2.5373 |
| SpaMosaic    |   5 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.1066 |             0.5533 |     4697 |  686.5060 | 2.3290 |
| SpaMosaic    |   5 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.1009 |             0.5504 |     4697 |  532.3109 | 2.3718 |
| SpaMosaic    |   5 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1628 |             0.5814 |     4253 |  841.6971 | 2.2112 |
| SpaMosaic    |   5 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1605 |             0.5803 |     4253 |  610.4193 | 2.1612 |
| SpaMosaic    |   5 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1438 |             0.5719 |     4646 |  865.9623 | 2.1987 |
| SpaMosaic    |   5 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1538 |             0.5769 |     4646 |  651.9371 | 2.2016 |
| SpaMosaic    |   5 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1286 |             0.5643 |     4228 |  785.5434 | 2.2861 |
| SpaMosaic    |   5 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.1426 |             0.5713 |     4228 |  589.8404 | 2.2087 |
| SpaMosaic    |   8 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0979 |             0.5490 |     4697 |  502.8721 | 2.2116 |
| SpaMosaic    |   8 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0959 |             0.5479 |     4697 |  406.4082 | 2.2144 |
| SpaMosaic    |   8 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1113 |             0.5557 |     4253 |  589.2793 | 2.2594 |
| SpaMosaic    |   8 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.1024 |             0.5512 |     4253 |  443.5353 | 2.3436 |
| SpaMosaic    |   8 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1160 |             0.5580 |     4646 |  615.9876 | 2.1804 |
| SpaMosaic    |   8 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1156 |             0.5578 |     4646 |  480.1045 | 2.2166 |
| SpaMosaic    |   8 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1094 |             0.5547 |     4228 |  554.2129 | 2.1976 |
| SpaMosaic    |   8 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.0944 |             0.5472 |     4228 |  428.9462 | 2.3038 |
| SpaMosaic    |  10 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0904 |             0.5452 |     4697 |  435.9341 | 2.2332 |
| SpaMosaic    |  10 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0872 |             0.5436 |     4697 |  353.8758 | 2.2406 |
| SpaMosaic    |  10 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.1068 |             0.5534 |     4253 |  495.9383 | 2.2404 |
| SpaMosaic    |  10 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.0926 |             0.5463 |     4253 |  378.9067 | 2.3402 |
| SpaMosaic    |  10 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.1000 |             0.5500 |     4646 |  522.7211 | 2.2160 |
| SpaMosaic    |  10 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.1025 |             0.5512 |     4646 |  410.8661 | 2.2445 |
| SpaMosaic    |  10 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.1014 |             0.5507 |     4228 |  472.9463 | 2.1945 |
| SpaMosaic    |  10 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.0902 |             0.5451 |     4228 |  368.8479 | 2.2640 |
| SpaMosaic    |  12 | Mouse_Thymus1 | Raw          |    4697 |  2635 |          0.0871 |             0.5435 |     4697 |  383.8084 | 2.2112 |
| SpaMosaic    |  12 | Mouse_Thymus1 | Standardized |    4697 |  2635 |          0.0831 |             0.5415 |     4697 |  315.1777 | 2.2197 |
| SpaMosaic    |  12 | Mouse_Thymus2 | Raw          |    4253 |  2386 |          0.0845 |             0.5422 |     4253 |  435.6276 | 2.2530 |
| SpaMosaic    |  12 | Mouse_Thymus2 | Standardized |    4253 |  2386 |          0.0828 |             0.5414 |     4253 |  332.2508 | 2.2979 |
| SpaMosaic    |  12 | Mouse_Thymus3 | Raw          |    4646 |  2607 |          0.0944 |             0.5472 |     4646 |  457.4066 | 2.2462 |
| SpaMosaic    |  12 | Mouse_Thymus3 | Standardized |    4646 |  2607 |          0.0833 |             0.5417 |     4646 |  361.2112 | 2.3579 |
| SpaMosaic    |  12 | Mouse_Thymus4 | Raw          |    4228 |  2372 |          0.0934 |             0.5467 |     4228 |  415.2838 | 2.2134 |
| SpaMosaic    |  12 | Mouse_Thymus4 | Standardized |    4228 |  2372 |          0.0749 |             0.5375 |     4228 |  324.9780 | 2.3630 |

### 4.3 Joint与Independent空间连续性

展示K=5/8/10/12；每个值为四个section全量spot空间近邻同簇比例的算术平均，Raw与Standardized行严格相邻。

| 方法         |   K | 模式        | 预处理       | Spatial neighbor agreement |
| ------------ | --: | ----------- | ------------ | -------------------------: |
| spa_mo_model |   5 | Joint       | Raw          |                     0.6868 |
| spa_mo_model |   5 | Joint       | Standardized |                     0.6781 |
| spa_mo_model |   5 | Independent | Raw          |                     0.5034 |
| spa_mo_model |   5 | Independent | Standardized |                     0.4983 |
| spa_mo_model |   8 | Joint       | Raw          |                     0.5820 |
| spa_mo_model |   8 | Joint       | Standardized |                     0.5701 |
| spa_mo_model |   8 | Independent | Raw          |                     0.3924 |
| spa_mo_model |   8 | Independent | Standardized |                     0.3841 |
| spa_mo_model |  10 | Joint       | Raw          |                     0.4891 |
| spa_mo_model |  10 | Joint       | Standardized |                     0.5094 |
| spa_mo_model |  10 | Independent | Raw          |                     0.3383 |
| spa_mo_model |  10 | Independent | Standardized |                     0.3246 |
| spa_mo_model |  12 | Joint       | Raw          |                     0.4683 |
| spa_mo_model |  12 | Joint       | Standardized |                     0.4481 |
| spa_mo_model |  12 | Independent | Raw          |                     0.2862 |
| spa_mo_model |  12 | Independent | Standardized |                     0.2977 |
| MOFA+        |   5 | Joint       | Raw          |                     0.7841 |
| MOFA+        |   5 | Joint       | Standardized |                     0.3832 |
| MOFA+        |   5 | Independent | Raw          |                     0.5821 |
| MOFA+        |   5 | Independent | Standardized |                     0.3803 |
| MOFA+        |   8 | Joint       | Raw          |                     0.4993 |
| MOFA+        |   8 | Joint       | Standardized |                     0.2704 |
| MOFA+        |   8 | Independent | Raw          |                     0.4516 |
| MOFA+        |   8 | Independent | Standardized |                     0.2895 |
| MOFA+        |  10 | Joint       | Raw          |                     0.4553 |
| MOFA+        |  10 | Joint       | Standardized |                     0.2469 |
| MOFA+        |  10 | Independent | Raw          |                     0.3619 |
| MOFA+        |  10 | Independent | Standardized |                     0.2563 |
| MOFA+        |  12 | Joint       | Raw          |                     0.4130 |
| MOFA+        |  12 | Joint       | Standardized |                     0.1996 |
| MOFA+        |  12 | Independent | Raw          |                     0.3183 |
| MOFA+        |  12 | Independent | Standardized |                     0.2375 |
| COSIE        |   5 | Joint       | Raw          |                     0.8021 |
| COSIE        |   5 | Joint       | Standardized |                     0.8068 |
| COSIE        |   5 | Independent | Raw          |                     0.6192 |
| COSIE        |   5 | Independent | Standardized |                     0.6212 |
| COSIE        |   8 | Joint       | Raw          |                     0.6210 |
| COSIE        |   8 | Joint       | Standardized |                     0.6165 |
| COSIE        |   8 | Independent | Raw          |                     0.5072 |
| COSIE        |   8 | Independent | Standardized |                     0.5157 |
| COSIE        |  10 | Joint       | Raw          |                     0.5763 |
| COSIE        |  10 | Joint       | Standardized |                     0.5655 |
| COSIE        |  10 | Independent | Raw          |                     0.4427 |
| COSIE        |  10 | Independent | Standardized |                     0.4749 |
| COSIE        |  12 | Joint       | Raw          |                     0.5344 |
| COSIE        |  12 | Joint       | Standardized |                     0.5320 |
| COSIE        |  12 | Independent | Raw          |                     0.4245 |
| COSIE        |  12 | Independent | Standardized |                     0.4423 |
| SpaMosaic    |   5 | Joint       | Raw          |                     0.6962 |
| SpaMosaic    |   5 | Joint       | Standardized |                     0.6876 |
| SpaMosaic    |   5 | Independent | Raw          |                     0.4744 |
| SpaMosaic    |   5 | Independent | Standardized |                     0.4727 |
| SpaMosaic    |   8 | Joint       | Raw          |                     0.5194 |
| SpaMosaic    |   8 | Joint       | Standardized |                     0.5103 |
| SpaMosaic    |   8 | Independent | Raw          |                     0.3681 |
| SpaMosaic    |   8 | Independent | Standardized |                     0.3527 |
| SpaMosaic    |  10 | Joint       | Raw          |                     0.4393 |
| SpaMosaic    |  10 | Joint       | Standardized |                     0.4266 |
| SpaMosaic    |  10 | Independent | Raw          |                     0.3173 |
| SpaMosaic    |  10 | Independent | Standardized |                     0.3095 |
| SpaMosaic    |  12 | Joint       | Raw          |                     0.3976 |
| SpaMosaic    |  12 | Joint       | Standardized |                     0.3887 |
| SpaMosaic    |  12 | Independent | Raw          |                     0.2812 |
| SpaMosaic    |  12 | Independent | Standardized |                     0.2644 |

## 5. 两种预处理的标签稳定性

| 方法         |   K | raw vs standardized ARI |    NMI |
| ------------ | --: | ----------------------: | -----: |
| spa_mo_model |   5 |                  0.8412 | 0.8406 |
| spa_mo_model |   8 |                  0.9142 | 0.8960 |
| spa_mo_model |  10 |                  0.8201 | 0.8014 |
| spa_mo_model |  12 |                  0.8641 | 0.8577 |
| MOFA+        |   5 |                  0.1068 | 0.2512 |
| MOFA+        |   8 |                  0.1176 | 0.2448 |
| MOFA+        |  10 |                  0.1535 | 0.2757 |
| MOFA+        |  12 |                  0.1380 | 0.2635 |
| COSIE        |   5 |                  0.9325 | 0.9090 |
| COSIE        |   8 |                  0.8754 | 0.8837 |
| COSIE        |  10 |                  0.8068 | 0.8276 |
| COSIE        |  12 |                  0.8354 | 0.8604 |
| SpaMosaic    |   5 |                  0.8993 | 0.8902 |
| SpaMosaic    |   8 |                  0.8795 | 0.8687 |
| SpaMosaic    |  10 |                  0.6616 | 0.7571 |
| SpaMosaic    |  12 |                  0.7632 | 0.7969 |

完整joint和independent K=2–12稳定性见各方法 `comparison_metrics/label_stability.csv`；内部指标差值见 `metric_deltas.csv`。

## 6. 共享指标及参数

### 6.1 Batch样本范围

| 方法         | batch key | n total | n used | n batches | batch categories                                        | batch counts                                                                                 | max samples | requested n | seed | embedding scaled | bASW n |
| ------------ | --------- | ------: | -----: | --------: | ------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------: | ----------: | ---: | ---------------- | -----: |
| spa_mo_model | section   |   17824 |  17824 |         4 | Mouse_Thymus1,Mouse_Thymus2,Mouse_Thymus3,Mouse_Thymus4 | {'Mouse_Thymus1': 4697, 'Mouse_Thymus2': 4253, 'Mouse_Thymus3': 4646, 'Mouse_Thymus4': 4228} |           0 |           0 |    0 | True             |  10000 |
| MOFA+        | group     |   17824 |  17824 |         4 | Mouse_Thymus1,Mouse_Thymus2,Mouse_Thymus3,Mouse_Thymus4 | NA                                                                                           |       50000 |           0 |    0 | NA               |  10000 |
| COSIE        | group     |   17824 |  17824 |         4 | Mouse_Thymus1,Mouse_Thymus2,Mouse_Thymus3,Mouse_Thymus4 | NA                                                                                           |           0 |           0 |    0 | NA               |  10000 |
| SpaMosaic    | group     |   17824 |  17824 |         4 | Mouse_Thymus1,Mouse_Thymus2,Mouse_Thymus3,Mouse_Thymus4 | NA                                                                                           |           0 |           0 |    0 | NA               |  10000 |

### 6.2 Batch近邻与PCR参数

| 方法         | kNN backend   | bLISI k | kBET k |  alpha | PCR PCs |
| ------------ | ------------- | ------: | -----: | -----: | ------: |
| spa_mo_model | sklearn_exact |      90 |     50 | 0.0500 |      50 |
| MOFA+        | sklearn_exact |      90 |     50 | 0.0500 |      10 |
| COSIE        | sklearn_exact |      90 |     50 | 0.0500 |      50 |
| SpaMosaic    | sklearn_exact |      90 |     50 | 0.0500 |      32 |

### 6.3 Batch完整指标值

| 方法         | bASW raw | bASW score | bLISI raw | bLISI normalized | kBET rejection | kBET acceptance | PCR batch R2 | PCR score |
| ------------ | -------: | ---------: | --------: | ---------------: | -------------: | --------------: | -----------: | --------: |
| spa_mo_model |   0.0925 |     0.9075 |    2.3646 |           0.4549 |         0.9975 |          0.0025 |       0.2601 |    0.7399 |
| MOFA+        |  -0.0303 |     0.9697 |    2.7941 |           0.5980 |         0.8576 |          0.1424 |       0.0000 |    1.0000 |
| COSIE        |   0.0963 |     0.9037 |    2.1929 |           0.3976 |         1.0000 |          0.0000 |       0.2695 |    0.7305 |
| SpaMosaic    |   0.0943 |     0.9057 |    2.4142 |           0.4714 |         0.9971 |          0.0029 |       0.3348 |    0.6652 |

batch指标不随本次KMeans输入分支变化，故只保存一份；来源和复制文件SHA256见 `shared_metrics/source_manifest.csv`。PCR PCs按 `min(50, embedding维度)`。

### 6.4 MOFA+视图解释度R²

| View | Group         |      R2 |
| ---- | ------------- | ------: |
| ADT  | Mouse_Thymus1 | 94.2309 |
| ADT  | Mouse_Thymus2 | 83.0657 |
| ADT  | Mouse_Thymus3 | 88.7821 |
| ADT  | Mouse_Thymus4 | 89.3420 |
| RNA  | Mouse_Thymus1 |  1.9342 |
| RNA  | Mouse_Thymus2 |  1.3736 |
| RNA  | Mouse_Thymus3 |  1.1192 |
| RNA  | Mouse_Thymus4 |  1.3721 |

该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

### 6.5 SpaMosaic模态对齐

| pair       | group         | n spots | mean cosine | median cosine | std cosine |
| ---------- | ------------- | ------: | ----------: | ------------: | ---------: |
| adt_vs_rna | ALL           |   17824 |      0.0217 |        0.0210 |     0.0124 |
| adt_vs_rna | Mouse_Thymus1 |    4697 |      0.0113 |        0.0096 |     0.0098 |
| adt_vs_rna | Mouse_Thymus2 |    4253 |      0.0251 |        0.0242 |     0.0110 |
| adt_vs_rna | Mouse_Thymus3 |    4646 |      0.0254 |        0.0248 |     0.0114 |
| adt_vs_rna | Mouse_Thymus4 |    4228 |      0.0255 |        0.0251 |     0.0109 |

该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

## 7. 结论与使用建议

1. Raw与Standardized均保留joint/independent K=2–12指标CSV；聚类文件夹与全量labels仅保留K=5/8/10/12。
2. Cluster ASW的具体10000个section/barcode身份已持久化并在四种方法、两种预处理、所有K和joint/independent间固定复用；CH/DBI始终全量。
3. 跨方法排名必须固定同一预处理分支；建议Standardized作为主分析，Raw作为敏感性分析。
4. Mouse_Thymus没有统一真实区域标签；section ARI/NMI是切片来源依赖诊断，不是生物学聚类准确率。
