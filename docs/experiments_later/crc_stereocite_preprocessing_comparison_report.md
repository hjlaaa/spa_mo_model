# CRC_Stereo-CITE-seq：Raw 与 Standardized embedding + MiniBatchKMeans 对比报告

- 生成时间：2026-07-27T14:32:08.108804+08:00
- 本报告只读取统一脚本直接生成的labels、指标CSV和provenance manifest；不使用 `report_derived_metrics/` 或其他报告派生计算链。
- 本次结果位于新增comparison目录，旧分析结果和原综合报告未覆盖。

## 1. 新结果目录

| 方法         | comparison root                                                               | 方法自有数据副本                                           |
| ------------ | ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/crc_stereocite_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq       |
| MOFA+        | /home/hujinlan/mofa+/analysis/crc_stereocite_preprocessing_comparison         | /home/hujinlan/mofa+/data/CRC_Stereo-CITE-seq              |
| COSIE        | /home/hujinlan/cosie_runs/crc_stereocite_preprocessing_comparison             | /home/hujinlan/cosie/data/CRC_Stereo-CITE-seq              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/crc_stereocite_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/CRC_Stereo-CITE-seq |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。

## 2. 统一标准

| 项目             | 统一值                                                  |
| ---------------- | ------------------------------------------------------- |
| spot/barcode     | 各方法自有数据；section + original barcode精确对齐      |
| spot数           | 612374（CRC_003=166279，CRC_006=446095）                |
| raw分支          | 原始最终embedding直接输入MiniBatchKMeans                |
| standardized分支 | StandardScaler；joint全体拟合，independent每section拟合 |
| 聚类器           | sklearn.cluster.MiniBatchKMeans                         |
| K                | 5、10、15、20、25；joint与independent一致               |
| 随机及迭代参数   | seed=0，n_init=20，max_iter=300，batch_size=8192        |
| Cluster ASW      | 固定同一批section分层10000个spot                        |
| ASW分层数        | CRC_003=2716，CRC_006=7284                              |
| CH/DBI           | 每个scope全量样本                                       |
| Label ASW        | 无真实细胞/区域标签，不计算                             |
| 空间连续性       | 每section全点，exact kd-tree 6-NN                       |
| 空间图           | 每section使用全量spot（CRC_003=166279，CRC_006=446095） |
| labels           | 全部方法、模式、分支和K均保存全量labels                 |
| 可追溯性         | 指标与图直接引用labels路径及SHA256；无报告派生链        |

## 3. 固定ASW抽样与全量绘图位置一致性

| 用途        | CRC_003 | CRC_006 |   总数 | 跨方法SHA256                                                     |
| ----------- | ------: | ------: | -----: | ---------------------------------------------------------------- |
| Cluster ASW |    2716 |    7284 |  10000 | 158d54ef95d70179ecddb951a6d97a8a59581c990b69ad1fab3034a3b4cef2e0 |
| 空间绘图    |  166279 |  446095 | 612374 | 8d8f1f6f71c8488320f869df16cf9ca695b7d24aa864a221ce4a3ae7559b56c5 |

ASW名单为每section按 `SHA256('asw_seed0|' + barcode)` 排序后截取；空间图使用两个section的全部spot。`plot_sample.csv`现作为全量绘图spot清单，按 `SHA256(barcode)` 确定性排序。四种方法的清单文件哈希完全相同，且全部612374个坐标已按barcode逐点核对一致。

## 4. 数据与 embedding 对齐

| 方法         | 维度 | CRC_003 | CRC_006 | 总spot |
| ------------ | ---: | ------: | ------: | -----: |
| spa_mo_model |  128 |  166279 |  446095 | 612374 |
| MOFA+        |   10 |  166279 |  446095 | 612374 |
| COSIE        |  256 |  166279 |  446095 | 612374 |
| SpaMosaic    |   32 |  166279 |  446095 | 612374 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/final_embeddings_CRC_003.npy`
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/selected_spot_indices_CRC_003.npy`
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/spatial_CRC_003.npy`
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/final_embeddings_CRC_006.npy`
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/selected_spot_indices_CRC_006.npy`
  - `/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/spatial_CRC_006.npy`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/crc_stereocite_mofa_hvg2000_k10_full_iter1000_cpu_float64/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/final_embeddings/CRC_003_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/obs_names_CRC_003.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/selected_spot_indices_CRC_003.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/spatial_CRC_003.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/final_embeddings/CRC_006_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/obs_names_CRC_006.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/selected_spot_indices_CRC_006.npy`
  - `/home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/spatial_CRC_006.npy`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/crc_stereocite_spamosaic_full_gpu_ce/crc_stereocite_spamosaic_embeddings.h5ad`

## 5. Joint内部与空间指标：Raw / Standardized上下对齐

同一方法、同一K的Raw与Standardized行相邻；表中值直接读取各分支 `metrics/clustering_metrics.csv` 和 `spatial_continuity_summary.csv`。

| 方法         |   K | 预处理       | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |          CH |    DBI | section ARI | section NMI | section ASW n | section ASW raw | joint spatial |
| ------------ | --: | ------------ | ----: | --------------: | -----------------: | -------: | ----------: | -----: | ----------: | ----------: | ------------: | --------------: | ------------: |
| spa_mo_model |   5 | Raw          | 10000 |          0.0426 |             0.5213 |   612374 |  20832.1248 | 4.0013 |      0.0295 |      0.0605 |         10000 |          0.0036 |        0.6428 |
| spa_mo_model |   5 | Standardized | 10000 |          0.0368 |             0.5184 |   612374 |  20094.2397 | 4.4145 |      0.0207 |      0.0447 |         10000 |          0.0045 |        0.5655 |
| spa_mo_model |  10 | Raw          | 10000 |          0.0339 |             0.5169 |   612374 |  13462.3735 | 4.1742 |      0.0227 |      0.0647 |         10000 |          0.0036 |        0.4575 |
| spa_mo_model |  10 | Standardized | 10000 |          0.0333 |             0.5166 |   612374 |  13251.4519 | 4.1956 |      0.0156 |      0.0487 |         10000 |          0.0045 |        0.4292 |
| spa_mo_model |  15 | Raw          | 10000 |          0.0339 |             0.5169 |   612374 |  10335.0773 | 4.0613 |      0.0143 |      0.0484 |         10000 |          0.0036 |        0.3605 |
| spa_mo_model |  15 | Standardized | 10000 |          0.0352 |             0.5176 |   612374 |  10130.3237 | 3.9970 |      0.0138 |      0.0417 |         10000 |          0.0045 |        0.3348 |
| spa_mo_model |  20 | Raw          | 10000 |          0.0263 |             0.5132 |   612374 |   8303.5401 | 4.1453 |      0.0114 |      0.0489 |         10000 |          0.0036 |        0.2770 |
| spa_mo_model |  20 | Standardized | 10000 |          0.0286 |             0.5143 |   612374 |   8189.8049 | 4.0597 |      0.0090 |      0.0445 |         10000 |          0.0045 |        0.2704 |
| spa_mo_model |  25 | Raw          | 10000 |          0.0261 |             0.5130 |   612374 |   7125.5586 | 4.0097 |      0.0093 |      0.0507 |         10000 |          0.0036 |        0.2488 |
| spa_mo_model |  25 | Standardized | 10000 |          0.0283 |             0.5142 |   612374 |   6956.5594 | 3.9911 |      0.0052 |      0.0390 |         10000 |          0.0045 |        0.2312 |
| MOFA+        |   5 | Raw          | 10000 |          0.1288 |             0.5644 |   612374 | 102804.8618 | 2.0119 |      0.0150 |      0.0471 |         10000 |          0.0282 |        0.6592 |
| MOFA+        |   5 | Standardized | 10000 |          0.0678 |             0.5339 |   612374 |  43241.9411 | 2.4501 |      0.0054 |      0.0049 |         10000 |          0.0272 |        0.3937 |
| MOFA+        |  10 | Raw          | 10000 |          0.0954 |             0.5477 |   612374 |  67141.5869 | 2.0329 |      0.0258 |      0.0672 |         10000 |          0.0282 |        0.5038 |
| MOFA+        |  10 | Standardized | 10000 |          0.0643 |             0.5322 |   612374 |  34067.0081 | 2.1889 |      0.0054 |      0.0122 |         10000 |          0.0272 |        0.3384 |
| MOFA+        |  15 | Raw          | 10000 |          0.0859 |             0.5430 |   612374 |  51959.1839 | 2.0244 |      0.0218 |      0.0731 |         10000 |          0.0282 |        0.3982 |
| MOFA+        |  15 | Standardized | 10000 |          0.0715 |             0.5358 |   612374 |  29230.6679 | 2.0333 |      0.0124 |      0.0466 |         10000 |          0.0272 |        0.3072 |
| MOFA+        |  20 | Raw          | 10000 |          0.0815 |             0.5408 |   612374 |  43266.5435 | 1.9959 |      0.0180 |      0.0685 |         10000 |          0.0282 |        0.3543 |
| MOFA+        |  20 | Standardized | 10000 |          0.0730 |             0.5365 |   612374 |  25331.0499 | 1.9978 |      0.0127 |      0.0509 |         10000 |          0.0272 |        0.2780 |
| MOFA+        |  25 | Raw          | 10000 |          0.0805 |             0.5403 |   612374 |  38045.8816 | 1.9575 |      0.0147 |      0.0743 |         10000 |          0.0282 |        0.3229 |
| MOFA+        |  25 | Standardized | 10000 |          0.0678 |             0.5339 |   612374 |  22233.6375 | 2.0045 |      0.0119 |      0.0580 |         10000 |          0.0272 |        0.2521 |
| COSIE        |   5 | Raw          | 10000 |          0.1439 |             0.5720 |   612374 |  97182.1056 | 2.2633 |      0.1528 |      0.2078 |         10000 |          0.0758 |        0.9101 |
| COSIE        |   5 | Standardized | 10000 |          0.1177 |             0.5589 |   612374 |  63073.8641 | 2.4873 |      0.0787 |      0.1393 |         10000 |          0.0633 |        0.9181 |
| COSIE        |  10 | Raw          | 10000 |          0.1272 |             0.5636 |   612374 |  65338.9046 | 2.2879 |      0.0841 |      0.2027 |         10000 |          0.0758 |        0.8769 |
| COSIE        |  10 | Standardized | 10000 |          0.1191 |             0.5596 |   612374 |  45515.7077 | 2.6272 |      0.1015 |      0.2147 |         10000 |          0.0633 |        0.8843 |
| COSIE        |  15 | Raw          | 10000 |          0.1135 |             0.5568 |   612374 |  48568.4023 | 2.3432 |      0.0862 |      0.2364 |         10000 |          0.0758 |        0.8571 |
| COSIE        |  15 | Standardized | 10000 |          0.0944 |             0.5472 |   612374 |  35232.9293 | 2.6322 |      0.0752 |      0.2253 |         10000 |          0.0633 |        0.8499 |
| COSIE        |  20 | Raw          | 10000 |          0.1118 |             0.5559 |   612374 |  40083.9306 | 2.4863 |      0.0597 |      0.2036 |         10000 |          0.0758 |        0.8350 |
| COSIE        |  20 | Standardized | 10000 |          0.0855 |             0.5428 |   612374 |  28771.8241 | 2.5423 |      0.0630 |      0.2253 |         10000 |          0.0633 |        0.8401 |
| COSIE        |  25 | Raw          | 10000 |          0.1006 |             0.5503 |   612374 |  34870.8437 | 2.4484 |      0.0496 |      0.2130 |         10000 |          0.0758 |        0.8251 |
| COSIE        |  25 | Standardized | 10000 |          0.0869 |             0.5435 |   612374 |  24961.4990 | 2.6550 |      0.0476 |      0.2196 |         10000 |          0.0633 |        0.8225 |
| SpaMosaic    |   5 | Raw          | 10000 |          0.1288 |             0.5644 |   612374 |  85673.7861 | 2.2496 |      0.0381 |      0.0755 |         10000 |          0.0126 |        0.7755 |
| SpaMosaic    |   5 | Standardized | 10000 |          0.1293 |             0.5647 |   612374 |  77080.6943 | 2.3039 |      0.0571 |      0.0794 |         10000 |          0.0134 |        0.6490 |
| SpaMosaic    |  10 | Raw          | 10000 |          0.0908 |             0.5454 |   612374 |  53284.8352 | 2.4532 |      0.0226 |      0.0571 |         10000 |          0.0126 |        0.4678 |
| SpaMosaic    |  10 | Standardized | 10000 |          0.0942 |             0.5471 |   612374 |  48304.4203 | 2.3760 |      0.0201 |      0.0631 |         10000 |          0.0134 |        0.4416 |
| SpaMosaic    |  15 | Raw          | 10000 |          0.0917 |             0.5459 |   612374 |  41464.0730 | 2.2549 |      0.0152 |      0.0620 |         10000 |          0.0126 |        0.3558 |
| SpaMosaic    |  15 | Standardized | 10000 |          0.0849 |             0.5425 |   612374 |  37389.1857 | 2.2630 |      0.0148 |      0.0607 |         10000 |          0.0134 |        0.3417 |
| SpaMosaic    |  20 | Raw          | 10000 |          0.0847 |             0.5424 |   612374 |  35078.9333 | 2.1070 |      0.0143 |      0.0603 |         10000 |          0.0126 |        0.3070 |
| SpaMosaic    |  20 | Standardized | 10000 |          0.0846 |             0.5423 |   612374 |  31752.7059 | 2.1991 |      0.0112 |      0.0583 |         10000 |          0.0134 |        0.2910 |
| SpaMosaic    |  25 | Raw          | 10000 |          0.0827 |             0.5414 |   612374 |  29790.5846 | 2.1851 |      0.0121 |      0.0600 |         10000 |          0.0126 |        0.2708 |
| SpaMosaic    |  25 | Standardized | 10000 |          0.0801 |             0.5400 |   612374 |  27264.1000 | 2.1822 |      0.0104 |      0.0586 |         10000 |          0.0134 |        0.2586 |

joint和independent完整指标均覆盖K=5/10/15/20/25；各分支 `metrics/direct_provenance_manifest.csv` 记录指标文件、抽样文件和配置的SHA256。

### 5.1 Joint全K最佳内部指标

下表只对现有K=5/10/15/20/25直接选择最优CSV行，不增加K=8，也不生成新指标。

| 方法         | 预处理       | best ASW K | ASW raw | ASW scaled | best CH K |          CH | best DBI K |    DBI | max section ARI K | section ARI | max section NMI K | section NMI |
| ------------ | ------------ | ---------: | ------: | ---------: | --------: | ----------: | ---------: | -----: | ----------------: | ----------: | ----------------: | ----------: |
| spa_mo_model | Raw          |          5 |  0.0426 |     0.5213 |         5 |  20832.1248 |          5 | 4.0013 |                 5 |      0.0295 |                10 |      0.0647 |
| spa_mo_model | Standardized |          5 |  0.0368 |     0.5184 |         5 |  20094.2397 |         25 | 3.9911 |                 5 |      0.0207 |                10 |      0.0487 |
| MOFA+        | Raw          |          5 |  0.1288 |     0.5644 |         5 | 102804.8618 |         25 | 1.9575 |                10 |      0.0258 |                25 |      0.0743 |
| MOFA+        | Standardized |         20 |  0.0730 |     0.5365 |         5 |  43241.9411 |         20 | 1.9978 |                20 |      0.0127 |                25 |      0.0580 |
| COSIE        | Raw          |          5 |  0.1439 |     0.5720 |         5 |  97182.1056 |          5 | 2.2633 |                 5 |      0.1528 |                15 |      0.2364 |
| COSIE        | Standardized |         10 |  0.1191 |     0.5596 |         5 |  63073.8641 |          5 | 2.4873 |                10 |      0.1015 |                15 |      0.2253 |
| SpaMosaic    | Raw          |          5 |  0.1288 |     0.5644 |         5 |  85673.7861 |         20 | 2.1070 |                 5 |      0.0381 |                 5 |      0.0755 |
| SpaMosaic    | Standardized |          5 |  0.1293 |     0.5647 |         5 |  77080.6943 |         25 | 2.1822 |                 5 |      0.0571 |                 5 |      0.0794 |

### 5.2 Independent逐section内部指标

同一方法、同一K、同一section的Raw与Standardized行严格相邻；ASW使用固定分层样本在对应section中的子集，CH/DBI使用该section全量spot。

| 方法         |   K | section | 预处理       | scope n | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |         CH |    DBI |
| ------------ | --: | ------- | ------------ | ------: | ----: | --------------: | -----------------: | -------: | ---------: | -----: |
| spa_mo_model |   5 | CRC_003 | Raw          |  166279 |  2716 |          0.0327 |             0.5163 |   166279 |  5617.2997 | 4.6059 |
| spa_mo_model |   5 | CRC_003 | Standardized |  166279 |  2716 |          0.0307 |             0.5153 |   166279 |  5227.7020 | 4.8139 |
| spa_mo_model |   5 | CRC_006 | Raw          |  446095 |  7284 |          0.0468 |             0.5234 |   446095 | 16818.7257 | 3.7224 |
| spa_mo_model |   5 | CRC_006 | Standardized |  446095 |  7284 |          0.0433 |             0.5217 |   446095 | 15940.9139 | 3.8343 |
| spa_mo_model |  10 | CRC_003 | Raw          |  166279 |  2716 |          0.0252 |             0.5126 |   166279 |  3408.2813 | 4.3468 |
| spa_mo_model |  10 | CRC_003 | Standardized |  166279 |  2716 |          0.0195 |             0.5097 |   166279 |  3143.7092 | 4.5950 |
| spa_mo_model |  10 | CRC_006 | Raw          |  446095 |  7284 |          0.0371 |             0.5186 |   446095 | 10744.9457 | 3.9509 |
| spa_mo_model |  10 | CRC_006 | Standardized |  446095 |  7284 |          0.0407 |             0.5204 |   446095 | 10431.5693 | 3.8180 |
| spa_mo_model |  15 | CRC_003 | Raw          |  166279 |  2716 |          0.0174 |             0.5087 |   166279 |  2529.6117 | 4.2999 |
| spa_mo_model |  15 | CRC_003 | Standardized |  166279 |  2716 |          0.0206 |             0.5103 |   166279 |  2488.8232 | 4.2566 |
| spa_mo_model |  15 | CRC_006 | Raw          |  446095 |  7284 |          0.0340 |             0.5170 |   446095 |  8063.4020 | 4.1809 |
| spa_mo_model |  15 | CRC_006 | Standardized |  446095 |  7284 |          0.0371 |             0.5185 |   446095 |  7916.0414 | 3.9509 |
| spa_mo_model |  20 | CRC_003 | Raw          |  166279 |  2716 |          0.0229 |             0.5114 |   166279 |  2117.4189 | 4.0961 |
| spa_mo_model |  20 | CRC_003 | Standardized |  166279 |  2716 |          0.0199 |             0.5100 |   166279 |  2050.0128 | 4.1487 |
| spa_mo_model |  20 | CRC_006 | Raw          |  446095 |  7284 |          0.0374 |             0.5187 |   446095 |  6652.7423 | 3.8119 |
| spa_mo_model |  20 | CRC_006 | Standardized |  446095 |  7284 |          0.0350 |             0.5175 |   446095 |  6443.2082 | 3.8643 |
| spa_mo_model |  25 | CRC_003 | Raw          |  166279 |  2716 |          0.0214 |             0.5107 |   166279 |  1822.5601 | 4.0122 |
| spa_mo_model |  25 | CRC_003 | Standardized |  166279 |  2716 |          0.0208 |             0.5104 |   166279 |  1762.7300 | 4.0075 |
| spa_mo_model |  25 | CRC_006 | Raw          |  446095 |  7284 |          0.0339 |             0.5169 |   446095 |  5583.0795 | 3.9318 |
| spa_mo_model |  25 | CRC_006 | Standardized |  446095 |  7284 |          0.0308 |             0.5154 |   446095 |  5430.3043 | 3.9288 |
| MOFA+        |   5 | CRC_003 | Raw          |  166279 |  2716 |          0.1355 |             0.5677 |   166279 | 35098.7933 | 2.0509 |
| MOFA+        |   5 | CRC_003 | Standardized |  166279 |  2716 |          0.1081 |             0.5540 |   166279 | 17602.8034 | 2.1293 |
| MOFA+        |   5 | CRC_006 | Raw          |  446095 |  7284 |          0.1221 |             0.5610 |   446095 | 81074.2414 | 2.1362 |
| MOFA+        |   5 | CRC_006 | Standardized |  446095 |  7284 |          0.0875 |             0.5438 |   446095 | 35208.5115 | 2.2737 |
| MOFA+        |  10 | CRC_003 | Raw          |  166279 |  2716 |          0.1063 |             0.5532 |   166279 | 24067.1930 | 1.9799 |
| MOFA+        |  10 | CRC_003 | Standardized |  166279 |  2716 |          0.0874 |             0.5437 |   166279 | 12527.4527 | 2.0163 |
| MOFA+        |  10 | CRC_006 | Raw          |  446095 |  7284 |          0.0960 |             0.5480 |   446095 | 52377.1646 | 2.0382 |
| MOFA+        |  10 | CRC_006 | Standardized |  446095 |  7284 |          0.0818 |             0.5409 |   446095 | 27597.9784 | 2.1120 |
| MOFA+        |  15 | CRC_003 | Raw          |  166279 |  2716 |          0.0948 |             0.5474 |   166279 | 18349.7389 | 1.9510 |
| MOFA+        |  15 | CRC_003 | Standardized |  166279 |  2716 |          0.0770 |             0.5385 |   166279 | 10013.8632 | 2.2083 |
| MOFA+        |  15 | CRC_006 | Raw          |  446095 |  7284 |          0.0980 |             0.5490 |   446095 | 41861.4596 | 1.9046 |
| MOFA+        |  15 | CRC_006 | Standardized |  446095 |  7284 |          0.0748 |             0.5374 |   446095 | 22634.8971 | 2.0655 |
| MOFA+        |  20 | CRC_003 | Raw          |  166279 |  2716 |          0.0857 |             0.5428 |   166279 | 15260.6681 | 1.9008 |
| MOFA+        |  20 | CRC_003 | Standardized |  166279 |  2716 |          0.0685 |             0.5342 |   166279 |  8517.6712 | 2.0560 |
| MOFA+        |  20 | CRC_006 | Raw          |  446095 |  7284 |          0.0870 |             0.5435 |   446095 | 34395.0401 | 1.9950 |
| MOFA+        |  20 | CRC_006 | Standardized |  446095 |  7284 |          0.0731 |             0.5366 |   446095 | 19362.0778 | 2.0062 |
| MOFA+        |  25 | CRC_003 | Raw          |  166279 |  2716 |          0.0816 |             0.5408 |   166279 | 13120.0853 | 1.9402 |
| MOFA+        |  25 | CRC_003 | Standardized |  166279 |  2716 |          0.0698 |             0.5349 |   166279 |  7545.1363 | 2.0942 |
| MOFA+        |  25 | CRC_006 | Raw          |  446095 |  7284 |          0.0831 |             0.5415 |   446095 | 29359.5085 | 2.0400 |
| MOFA+        |  25 | CRC_006 | Standardized |  446095 |  7284 |          0.0668 |             0.5334 |   446095 | 16953.8803 | 2.1105 |
| COSIE        |   5 | CRC_003 | Raw          |  166279 |  2716 |          0.1600 |             0.5800 |   166279 | 35203.0676 | 1.9989 |
| COSIE        |   5 | CRC_003 | Standardized |  166279 |  2716 |          0.2138 |             0.6069 |   166279 | 24985.7658 | 1.9600 |
| COSIE        |   5 | CRC_006 | Raw          |  446095 |  7284 |          0.1516 |             0.5758 |   446095 | 67378.9233 | 1.9364 |
| COSIE        |   5 | CRC_006 | Standardized |  446095 |  7284 |          0.1361 |             0.5680 |   446095 | 44342.8744 | 2.2069 |
| COSIE        |  10 | CRC_003 | Raw          |  166279 |  2716 |          0.1154 |             0.5577 |   166279 | 21199.7563 | 2.4566 |
| COSIE        |  10 | CRC_003 | Standardized |  166279 |  2716 |          0.0648 |             0.5324 |   166279 | 14687.0714 | 2.7985 |
| COSIE        |  10 | CRC_006 | Raw          |  446095 |  7284 |          0.1199 |             0.5600 |   446095 | 45746.4103 | 2.0251 |
| COSIE        |  10 | CRC_006 | Standardized |  446095 |  7284 |          0.1060 |             0.5530 |   446095 | 31960.0308 | 2.3811 |
| COSIE        |  15 | CRC_003 | Raw          |  166279 |  2716 |          0.0998 |             0.5499 |   166279 | 15099.4262 | 2.7369 |
| COSIE        |  15 | CRC_003 | Standardized |  166279 |  2716 |          0.0677 |             0.5338 |   166279 | 10457.8782 | 3.0920 |
| COSIE        |  15 | CRC_006 | Raw          |  446095 |  7284 |          0.1078 |             0.5539 |   446095 | 35217.1264 | 2.2387 |
| COSIE        |  15 | CRC_006 | Standardized |  446095 |  7284 |          0.0920 |             0.5460 |   446095 | 24720.2350 | 2.4394 |
| COSIE        |  20 | CRC_003 | Raw          |  166279 |  2716 |          0.0820 |             0.5410 |   166279 | 12114.9612 | 2.6494 |
| COSIE        |  20 | CRC_003 | Standardized |  166279 |  2716 |          0.0557 |             0.5279 |   166279 |  8398.6794 | 3.0116 |
| COSIE        |  20 | CRC_006 | Raw          |  446095 |  7284 |          0.0931 |             0.5465 |   446095 | 28244.0823 | 2.5160 |
| COSIE        |  20 | CRC_006 | Standardized |  446095 |  7284 |          0.0876 |             0.5438 |   446095 | 20508.3660 | 2.4500 |
| COSIE        |  25 | CRC_003 | Raw          |  166279 |  2716 |          0.0585 |             0.5293 |   166279 | 10241.9852 | 2.6858 |
| COSIE        |  25 | CRC_003 | Standardized |  166279 |  2716 |          0.0451 |             0.5225 |   166279 |  7071.8256 | 3.1443 |
| COSIE        |  25 | CRC_006 | Raw          |  446095 |  7284 |          0.0935 |             0.5467 |   446095 | 24645.4589 | 2.5004 |
| COSIE        |  25 | CRC_006 | Standardized |  446095 |  7284 |          0.0814 |             0.5407 |   446095 | 17651.2496 | 2.6785 |
| SpaMosaic    |   5 | CRC_003 | Raw          |  166279 |  2716 |          0.1469 |             0.5734 |   166279 | 25100.3040 | 2.2391 |
| SpaMosaic    |   5 | CRC_003 | Standardized |  166279 |  2716 |          0.1656 |             0.5828 |   166279 | 25805.9144 | 2.0275 |
| SpaMosaic    |   5 | CRC_006 | Raw          |  446095 |  7284 |          0.1015 |             0.5508 |   446095 | 60285.8153 | 2.5071 |
| SpaMosaic    |   5 | CRC_006 | Standardized |  446095 |  7284 |          0.1283 |             0.5641 |   446095 | 51809.9986 | 2.4183 |
| SpaMosaic    |  10 | CRC_003 | Raw          |  166279 |  2716 |          0.1212 |             0.5606 |   166279 | 18204.7791 | 2.1522 |
| SpaMosaic    |  10 | CRC_003 | Standardized |  166279 |  2716 |          0.1104 |             0.5552 |   166279 | 15839.8340 | 2.2690 |
| SpaMosaic    |  10 | CRC_006 | Raw          |  446095 |  7284 |          0.0924 |             0.5462 |   446095 | 37760.1642 | 2.3266 |
| SpaMosaic    |  10 | CRC_006 | Standardized |  446095 |  7284 |          0.0891 |             0.5445 |   446095 | 34290.8345 | 2.2396 |
| SpaMosaic    |  15 | CRC_003 | Raw          |  166279 |  2716 |          0.0981 |             0.5490 |   166279 | 14008.0494 | 2.1069 |
| SpaMosaic    |  15 | CRC_003 | Standardized |  166279 |  2716 |          0.0848 |             0.5424 |   166279 | 12084.4861 | 2.2847 |
| SpaMosaic    |  15 | CRC_006 | Raw          |  446095 |  7284 |          0.0893 |             0.5446 |   446095 | 29781.3521 | 2.2402 |
| SpaMosaic    |  15 | CRC_006 | Standardized |  446095 |  7284 |          0.0810 |             0.5405 |   446095 | 26674.8805 | 2.2159 |
| SpaMosaic    |  20 | CRC_003 | Raw          |  166279 |  2716 |          0.0849 |             0.5425 |   166279 | 11331.8066 | 2.1863 |
| SpaMosaic    |  20 | CRC_003 | Standardized |  166279 |  2716 |          0.1086 |             0.5543 |   166279 | 10281.4074 | 2.1631 |
| SpaMosaic    |  20 | CRC_006 | Raw          |  446095 |  7284 |          0.0847 |             0.5424 |   446095 | 24773.7003 | 2.1947 |
| SpaMosaic    |  20 | CRC_006 | Standardized |  446095 |  7284 |          0.0787 |             0.5393 |   446095 | 22410.3022 | 2.2355 |
| SpaMosaic    |  25 | CRC_003 | Raw          |  166279 |  2716 |          0.0778 |             0.5389 |   166279 |  9921.0794 | 2.2194 |
| SpaMosaic    |  25 | CRC_003 | Standardized |  166279 |  2716 |          0.0794 |             0.5397 |   166279 |  8764.0852 | 2.2368 |
| SpaMosaic    |  25 | CRC_006 | Raw          |  446095 |  7284 |          0.0807 |             0.5403 |   446095 | 21420.7959 | 2.1813 |
| SpaMosaic    |  25 | CRC_006 | Standardized |  446095 |  7284 |          0.0736 |             0.5368 |   446095 | 19173.0163 | 2.2934 |

### 5.3 Joint与Independent空间连续性

每个值均为两个section全量spot的空间近邻同簇比例算术平均；Raw与Standardized行严格相邻。

| 方法         |   K | 模式        | 预处理       | Spatial neighbor agreement |
| ------------ | --: | ----------- | ------------ | -------------------------: |
| spa_mo_model |   5 | Joint       | Raw          |                     0.6428 |
| spa_mo_model |   5 | Joint       | Standardized |                     0.5655 |
| spa_mo_model |   5 | Independent | Raw          |                     0.6209 |
| spa_mo_model |   5 | Independent | Standardized |                     0.5821 |
| spa_mo_model |  10 | Joint       | Raw          |                     0.4575 |
| spa_mo_model |  10 | Joint       | Standardized |                     0.4292 |
| spa_mo_model |  10 | Independent | Raw          |                     0.4181 |
| spa_mo_model |  10 | Independent | Standardized |                     0.4143 |
| spa_mo_model |  15 | Joint       | Raw          |                     0.3605 |
| spa_mo_model |  15 | Joint       | Standardized |                     0.3348 |
| spa_mo_model |  15 | Independent | Raw          |                     0.3058 |
| spa_mo_model |  15 | Independent | Standardized |                     0.3085 |
| spa_mo_model |  20 | Joint       | Raw          |                     0.2770 |
| spa_mo_model |  20 | Joint       | Standardized |                     0.2704 |
| spa_mo_model |  20 | Independent | Raw          |                     0.2579 |
| spa_mo_model |  20 | Independent | Standardized |                     0.2503 |
| spa_mo_model |  25 | Joint       | Raw          |                     0.2488 |
| spa_mo_model |  25 | Joint       | Standardized |                     0.2312 |
| spa_mo_model |  25 | Independent | Raw          |                     0.2226 |
| spa_mo_model |  25 | Independent | Standardized |                     0.2206 |
| MOFA+        |   5 | Joint       | Raw          |                     0.6592 |
| MOFA+        |   5 | Joint       | Standardized |                     0.3937 |
| MOFA+        |   5 | Independent | Raw          |                     0.5963 |
| MOFA+        |   5 | Independent | Standardized |                     0.4745 |
| MOFA+        |  10 | Joint       | Raw          |                     0.5038 |
| MOFA+        |  10 | Joint       | Standardized |                     0.3384 |
| MOFA+        |  10 | Independent | Raw          |                     0.4657 |
| MOFA+        |  10 | Independent | Standardized |                     0.3490 |
| MOFA+        |  15 | Joint       | Raw          |                     0.3982 |
| MOFA+        |  15 | Joint       | Standardized |                     0.3072 |
| MOFA+        |  15 | Independent | Raw          |                     0.3938 |
| MOFA+        |  15 | Independent | Standardized |                     0.2900 |
| MOFA+        |  20 | Joint       | Raw          |                     0.3543 |
| MOFA+        |  20 | Joint       | Standardized |                     0.2780 |
| MOFA+        |  20 | Independent | Raw          |                     0.3291 |
| MOFA+        |  20 | Independent | Standardized |                     0.2418 |
| MOFA+        |  25 | Joint       | Raw          |                     0.3229 |
| MOFA+        |  25 | Joint       | Standardized |                     0.2521 |
| MOFA+        |  25 | Independent | Raw          |                     0.2913 |
| MOFA+        |  25 | Independent | Standardized |                     0.2168 |
| COSIE        |   5 | Joint       | Raw          |                     0.9101 |
| COSIE        |   5 | Joint       | Standardized |                     0.9181 |
| COSIE        |   5 | Independent | Raw          |                     0.9148 |
| COSIE        |   5 | Independent | Standardized |                     0.9198 |
| COSIE        |  10 | Joint       | Raw          |                     0.8769 |
| COSIE        |  10 | Joint       | Standardized |                     0.8843 |
| COSIE        |  10 | Independent | Raw          |                     0.8476 |
| COSIE        |  10 | Independent | Standardized |                     0.8448 |
| COSIE        |  15 | Joint       | Raw          |                     0.8571 |
| COSIE        |  15 | Joint       | Standardized |                     0.8499 |
| COSIE        |  15 | Independent | Raw          |                     0.8097 |
| COSIE        |  15 | Independent | Standardized |                     0.8062 |
| COSIE        |  20 | Joint       | Raw          |                     0.8350 |
| COSIE        |  20 | Joint       | Standardized |                     0.8401 |
| COSIE        |  20 | Independent | Raw          |                     0.7892 |
| COSIE        |  20 | Independent | Standardized |                     0.7949 |
| COSIE        |  25 | Joint       | Raw          |                     0.8251 |
| COSIE        |  25 | Joint       | Standardized |                     0.8225 |
| COSIE        |  25 | Independent | Raw          |                     0.7652 |
| COSIE        |  25 | Independent | Standardized |                     0.7695 |
| SpaMosaic    |   5 | Joint       | Raw          |                     0.7755 |
| SpaMosaic    |   5 | Joint       | Standardized |                     0.6490 |
| SpaMosaic    |   5 | Independent | Raw          |                     0.6252 |
| SpaMosaic    |   5 | Independent | Standardized |                     0.6305 |
| SpaMosaic    |  10 | Joint       | Raw          |                     0.4678 |
| SpaMosaic    |  10 | Joint       | Standardized |                     0.4416 |
| SpaMosaic    |  10 | Independent | Raw          |                     0.4157 |
| SpaMosaic    |  10 | Independent | Standardized |                     0.4086 |
| SpaMosaic    |  15 | Joint       | Raw          |                     0.3558 |
| SpaMosaic    |  15 | Joint       | Standardized |                     0.3417 |
| SpaMosaic    |  15 | Independent | Raw          |                     0.3218 |
| SpaMosaic    |  15 | Independent | Standardized |                     0.3090 |
| SpaMosaic    |  20 | Joint       | Raw          |                     0.3070 |
| SpaMosaic    |  20 | Joint       | Standardized |                     0.2910 |
| SpaMosaic    |  20 | Independent | Raw          |                     0.2789 |
| SpaMosaic    |  20 | Independent | Standardized |                     0.2672 |
| SpaMosaic    |  25 | Joint       | Raw          |                     0.2708 |
| SpaMosaic    |  25 | Joint       | Standardized |                     0.2586 |
| SpaMosaic    |  25 | Independent | Raw          |                     0.2443 |
| SpaMosaic    |  25 | Independent | Standardized |                     0.2369 |

## 6. 两种预处理的标签稳定性

| 方法         |   K |  全量n | raw vs standardized ARI |    NMI |
| ------------ | --: | -----: | ----------------------: | -----: |
| spa_mo_model |   5 | 612374 |                  0.4314 | 0.5034 |
| spa_mo_model |  10 | 612374 |                  0.4727 | 0.5874 |
| spa_mo_model |  15 | 612374 |                  0.4607 | 0.5673 |
| spa_mo_model |  20 | 612374 |                  0.2532 | 0.4396 |
| spa_mo_model |  25 | 612374 |                  0.2740 | 0.4685 |
| MOFA+        |   5 | 612374 |                  0.0782 | 0.0958 |
| MOFA+        |  10 | 612374 |                  0.1426 | 0.2404 |
| MOFA+        |  15 | 612374 |                  0.1650 | 0.3073 |
| MOFA+        |  20 | 612374 |                  0.1675 | 0.3381 |
| MOFA+        |  25 | 612374 |                  0.1842 | 0.3816 |
| COSIE        |   5 | 612374 |                  0.4540 | 0.5497 |
| COSIE        |  10 | 612374 |                  0.5744 | 0.6912 |
| COSIE        |  15 | 612374 |                  0.5099 | 0.6788 |
| COSIE        |  20 | 612374 |                  0.4760 | 0.6784 |
| COSIE        |  25 | 612374 |                  0.5645 | 0.7451 |
| SpaMosaic    |   5 | 612374 |                  0.6185 | 0.6331 |
| SpaMosaic    |  10 | 612374 |                  0.3661 | 0.5133 |
| SpaMosaic    |  15 | 612374 |                  0.2974 | 0.5092 |
| SpaMosaic    |  20 | 612374 |                  0.3771 | 0.5754 |
| SpaMosaic    |  25 | 612374 |                  0.3305 | 0.5594 |

完整joint/independent稳定性及两套labels的路径和SHA256见各方法 `comparison_metrics/label_stability.csv`。

## 7. 共享指标及参数

### 7.1 Batch样本范围

| 方法         | batch key | n total | n used | n batches | batch categories            | batch counts                           | max samples | requested n | seed | embedding scaled | bASW n |
| ------------ | --------- | ------: | -----: | --------: | --------------------------- | -------------------------------------- | ----------: | ----------: | ---: | ---------------- | -----: |
| spa_mo_model | section   |  612374 | 612374 |         2 | CRC_003,CRC_006             | {'CRC_003': 166279, 'CRC_006': 446095} |           0 |           0 |    0 | True             |  10000 |
| MOFA+        | group     |  612374 | 612374 |         2 | CRC_003_bin20,CRC_006_bin20 | NA                                     |           0 |           0 |    0 | NA               |  10000 |
| COSIE        | group     |  612374 | 612374 |         2 | CRC_003,CRC_006             | NA                                     |           0 |           0 |    0 | NA               |  10000 |
| SpaMosaic    | group     |  612374 | 612374 |         2 | CRC_003_bin20,CRC_006_bin20 | NA                                     |           0 |           0 |    0 | NA               |  10000 |

### 7.2 Batch近邻与PCR参数

| 方法         | kNN backend       | bLISI k | kBET k |  alpha | PCR PCs |
| ------------ | ----------------- | ------: | -----: | -----: | ------: |
| spa_mo_model | hnswlib_l2_approx |      90 |     50 | 0.0500 |      50 |
| MOFA+        | hnswlib_l2_approx |      90 |     50 | 0.0500 |      10 |
| COSIE        | hnswlib_l2_approx |      90 |     50 | 0.0500 |      50 |
| SpaMosaic    | hnswlib_l2_approx |      90 |     50 | 0.0500 |      32 |

### 7.3 Batch完整指标值

| 方法         | bASW raw | bASW score | bLISI raw | bLISI normalized | kBET rejection | kBET acceptance | PCR batch R2 | PCR score |
| ------------ | -------: | ---------: | --------: | ---------------: | -------------: | --------------: | -----------: | --------: |
| spa_mo_model |   0.0038 |     0.9962 |    1.4636 |           0.4636 |         0.6896 |          0.3104 |       0.0099 |    0.9901 |
| MOFA+        |   0.0309 |     0.9691 |    1.2925 |           0.2925 |         0.8253 |          0.1747 |       0.0000 |    1.0000 |
| COSIE        |   0.0622 |     0.9378 |    1.0306 |           0.0306 |         0.9889 |          0.0111 |       0.0670 |    0.9330 |
| SpaMosaic    |   0.0134 |     0.9866 |    1.3492 |           0.3492 |         0.8201 |          0.1799 |       0.0236 |    0.9764 |

batch指标不随本次KMeans输入分支变化，故只保存一份；原文件与复制文件SHA256见 `shared_metrics/source_manifest.csv`。

### 7.4 MOFA+视图解释度R²

| View | Group         |      R2 |
| ---- | ------------- | ------: |
| ADT  | CRC_003_bin20 | 14.7640 |
| ADT  | CRC_006_bin20 | 27.6624 |
| RNA  | CRC_003_bin20 |  0.4358 |
| RNA  | CRC_006_bin20 |  0.4497 |

该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

### 7.5 SpaMosaic模态对齐

| pair       | group         | n spots | mean cosine | median cosine | std cosine |
| ---------- | ------------- | ------: | ----------: | ------------: | ---------: |
| adt_vs_rna | ALL           |  612374 |      0.0115 |        0.0097 |     0.0113 |
| adt_vs_rna | CRC_003_bin20 |  166279 |      0.0096 |        0.0095 |     0.0082 |
| adt_vs_rna | CRC_006_bin20 |  446095 |      0.0122 |        0.0097 |     0.0121 |

该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

## 8. 结论与使用建议

1. Raw与Standardized均作为完整结果保存；跨方法排名必须固定同一预处理分支。
2. Standardized可作为尺度可比的主分析，Raw作为敏感性分析；同时报告全量labels的raw-vs-standardized ARI/NMI。
3. 所有内部、空间指标和图片均由统一脚本直接读取本次保存的labels生成，并记录labels SHA256；报告不再连接任何派生式计算链。
4. CRC数据没有统一真实区域标签；section ARI/NMI是样本来源依赖诊断，不是生物学聚类准确率。
