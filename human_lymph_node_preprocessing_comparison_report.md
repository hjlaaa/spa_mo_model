# Human_Lymph_Node：Raw 与 Standardized embedding + KMeans 对比报告

- 生成时间：2026-07-27T14:47:54.962241+08:00
- 本报告只引用本次新增的 comparison 结果；原分析目录未覆盖。
- 两套方案分别在各自 KMeans 输入空间计算 ASW、CH、DBI；因此适合在同一预处理方案内跨方法比较，跨方案数值变化应结合标签稳定性解释。

## 1. 新结果目录

| 方法         | comparison root                                                                 | 方法自有数据副本                                        |
| ------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/human_lymph_node_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/Human_Lymph_Node       |
| MOFA+        | /home/hujinlan/mofa+/analysis/human_lymph_node_preprocessing_comparison         | /home/hujinlan/mofa+/data/Human_Lymph_Node              |
| COSIE        | /home/hujinlan/cosie_runs/human_lymph_node_preprocessing_comparison             | /home/hujinlan/cosie/data/Human_Lymph_Node              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/human_lymph_node_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/Human_Lymph_Node |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和 `preprocessing_comparison_manifest.json`。

## 2. 统一标准

| 项目                         | 统一值                                                    |
| ---------------------------- | --------------------------------------------------------- |
| spot / barcode               | 各方法自有数据副本；section + original barcode 精确对齐   |
| spot 数                      | 6843（A1=3484，D1=3359）                                  |
| embedding                    | 各方法当前最终 embedding                                  |
| raw 分支                     | 原始最终 embedding 直接输入 KMeans                        |
| standardized 分支            | StandardScaler；joint 全体拟合，independent 每切片拟合    |
| KMeans                       | sklearn exact KMeans                                      |
| joint 指标 K                 | 2–12                                                      |
| independent 指标 K           | 5、8、10、12                                              |
| 保留的聚类文件夹/labels/图 K | 5、8、10、12                                              |
| 随机参数                     | seed=0，n_init=20，max_iter=300                           |
| ASW / CH / DBI               | KMeans 同一输入空间，全量样本                             |
| Label ASW                    | 无真实细胞/区域标签，不计算                               |
| 空间连续性                   | 每切片全点建图，exact 6-NN                                |
| 保留 K 的图与指标            | 复用同一套已保存 label CSV                                |
| batch 指标                   | 预处理无关；复制原最终结果到 shared_metrics 并记录 SHA256 |

## 3. 数据与 embedding 对齐

| 方法         | 维度 |   A1 |   D1 |
| ------------ | ---: | ---: | ---: |
| spa_mo_model |  128 | 3484 | 3359 |
| MOFA+        |   10 | 3484 | 3359 |
| COSIE        |  256 | 3484 | 3359 |
| SpaMosaic    |   32 | 3484 | 3359 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Human_Lymph_Node_A1.npy`
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/selected_spot_indices_Human_Lymph_Node_A1.npy`
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/spatial_Human_Lymph_Node_A1.npy`
  - `/home/hujinlan/spa_mo_model/data/Human_Lymph_Node/Human_Lymph_Node_A1/adata_RNA.h5ad`
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Human_Lymph_Node_D1.npy`
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/selected_spot_indices_Human_Lymph_Node_D1.npy`
  - `/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/spatial_Human_Lymph_Node_D1.npy`
  - `/home/hujinlan/spa_mo_model/data/Human_Lymph_Node/Human_Lymph_Node_D1/adata_RNA.h5ad`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/human_lymph_node_mofa_hvg2000_k10_iter1000/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/human_lymph_node_cosie_rna_adt_full/analysis/tables/embeddings_with_metadata.csv`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/human_lymph_node_spamosaic/human_lymph_node_spamosaic_embeddings.h5ad`

## 4. Joint内部与空间指标：Raw / Standardized上下对齐

同一方法、同一 K 的 Raw 与 Standardized 行相邻，所有列采用完全相同的定义。

| 方法         |   K | 预处理       | ASW n | ASW raw | ASW scaled | CH/DBI n |        CH |    DBI | section ARI | section NMI | section ASW n | section ASW raw | joint spatial |
| ------------ | --: | ------------ | ----: | ------: | ---------: | -------: | --------: | -----: | ----------: | ----------: | ------------: | --------------: | ------------: |
| spa_mo_model |   5 | Raw          |  6843 |  0.1015 |     0.5508 |     6843 |  564.4859 | 2.4803 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.7053 |
| spa_mo_model |   5 | Standardized |  6843 |  0.0978 |     0.5489 |     6843 |  527.4998 | 2.5429 |     -0.0000 |      0.0007 |          6843 |          0.0011 |        0.7093 |
| spa_mo_model |   8 | Raw          |  6843 |  0.0860 |     0.5430 |     6843 |  430.2131 | 2.6815 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.5749 |
| spa_mo_model |   8 | Standardized |  6843 |  0.0820 |     0.5410 |     6843 |  401.9315 | 2.7482 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.5598 |
| spa_mo_model |  10 | Raw          |  6843 |  0.0849 |     0.5424 |     6843 |  371.8724 | 2.8937 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.4790 |
| spa_mo_model |  10 | Standardized |  6843 |  0.0810 |     0.5405 |     6843 |  349.2462 | 2.9238 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.4685 |
| spa_mo_model |  12 | Raw          |  6843 |  0.0851 |     0.5425 |     6843 |  326.4650 | 2.7661 |     -0.0001 |      0.0006 |          6843 |          0.0011 |        0.4634 |
| spa_mo_model |  12 | Standardized |  6843 |  0.0801 |     0.5400 |     6843 |  308.1397 | 2.8134 |     -0.0000 |      0.0005 |          6843 |          0.0011 |        0.4447 |
| MOFA+        |   5 | Raw          |  6843 |  0.2745 |     0.6373 |     6843 |  889.0700 | 1.4516 |      0.0003 |      0.0011 |          6843 |          0.0010 |        0.5148 |
| MOFA+        |   5 | Standardized |  6843 |  0.2718 |     0.6359 |     6843 |  846.8471 | 1.4251 |      0.0002 |      0.0011 |          6843 |          0.0011 |        0.5180 |
| MOFA+        |   8 | Raw          |  6843 |  0.1819 |     0.5910 |     6843 |  891.3348 | 1.4260 |      0.0005 |      0.0017 |          6843 |          0.0010 |        0.4474 |
| MOFA+        |   8 | Standardized |  6843 |  0.1699 |     0.5850 |     6843 |  838.1938 | 1.4357 |      0.0002 |      0.0017 |          6843 |          0.0011 |        0.4383 |
| MOFA+        |  10 | Raw          |  6843 |  0.1992 |     0.5996 |     6843 |  859.3267 | 1.3502 |      0.0004 |      0.0012 |          6843 |          0.0010 |        0.4293 |
| MOFA+        |  10 | Standardized |  6843 |  0.1540 |     0.5770 |     6843 |  793.0058 | 1.3814 |      0.0019 |      0.0029 |          6843 |          0.0011 |        0.3824 |
| MOFA+        |  12 | Raw          |  6843 |  0.1692 |     0.5846 |     6843 |  812.3756 | 1.4108 |      0.0016 |      0.0025 |          6843 |          0.0010 |        0.3392 |
| MOFA+        |  12 | Standardized |  6843 |  0.1626 |     0.5813 |     6843 |  756.8574 | 1.4460 |      0.0014 |      0.0026 |          6843 |          0.0011 |        0.3458 |
| COSIE        |   5 | Raw          |  6843 |  0.2006 |     0.6003 |     6843 | 1601.8468 | 1.6211 |      0.0028 |      0.0028 |          6843 |          0.0073 |        0.7550 |
| COSIE        |   5 | Standardized |  6843 |  0.1847 |     0.5924 |     6843 | 1263.8205 | 1.7425 |      0.0026 |      0.0025 |          6843 |          0.0085 |        0.7647 |
| COSIE        |   8 | Raw          |  6843 |  0.1771 |     0.5886 |     6843 | 1246.3026 | 1.7803 |      0.0041 |      0.0049 |          6843 |          0.0073 |        0.6243 |
| COSIE        |   8 | Standardized |  6843 |  0.1601 |     0.5800 |     6843 |  972.4262 | 1.9355 |      0.0026 |      0.0033 |          6843 |          0.0085 |        0.6580 |
| COSIE        |  10 | Raw          |  6843 |  0.1675 |     0.5837 |     6843 | 1087.3361 | 1.8606 |      0.0038 |      0.0056 |          6843 |          0.0073 |        0.5810 |
| COSIE        |  10 | Standardized |  6843 |  0.1616 |     0.5808 |     6843 |  851.7784 | 1.9147 |      0.0046 |      0.0055 |          6843 |          0.0085 |        0.6130 |
| COSIE        |  12 | Raw          |  6843 |  0.1603 |     0.5801 |     6843 |  968.2372 | 1.8842 |      0.0028 |      0.0052 |          6843 |          0.0073 |        0.5574 |
| COSIE        |  12 | Standardized |  6843 |  0.1505 |     0.5753 |     6843 |  760.3959 | 2.0167 |      0.0036 |      0.0055 |          6843 |          0.0085 |        0.5760 |
| SpaMosaic    |   5 | Raw          |  6843 |  0.2099 |     0.6049 |     6843 | 2285.1139 | 1.5694 |      0.0001 |      0.0006 |          6843 |          0.0004 |        0.7022 |
| SpaMosaic    |   5 | Standardized |  6843 |  0.1789 |     0.5894 |     6843 | 1754.4991 | 1.7911 |      0.0001 |      0.0006 |          6843 |          0.0005 |        0.6987 |
| SpaMosaic    |   8 | Raw          |  6843 |  0.1891 |     0.5945 |     6843 | 1686.9200 | 1.6806 |     -0.0000 |      0.0005 |          6843 |          0.0004 |        0.6297 |
| SpaMosaic    |   8 | Standardized |  6843 |  0.1658 |     0.5829 |     6843 | 1332.2148 | 1.7360 |     -0.0000 |      0.0005 |          6843 |          0.0005 |        0.6263 |
| SpaMosaic    |  10 | Raw          |  6843 |  0.1594 |     0.5797 |     6843 | 1466.0439 | 1.7456 |      0.0000 |      0.0006 |          6843 |          0.0004 |        0.5753 |
| SpaMosaic    |  10 | Standardized |  6843 |  0.1481 |     0.5741 |     6843 | 1155.8133 | 1.8195 |     -0.0001 |      0.0005 |          6843 |          0.0005 |        0.5815 |
| SpaMosaic    |  12 | Raw          |  6843 |  0.1519 |     0.5759 |     6843 | 1306.9435 | 1.7080 |      0.0000 |      0.0006 |          6843 |          0.0004 |        0.5431 |
| SpaMosaic    |  12 | Standardized |  6843 |  0.1508 |     0.5754 |     6843 | 1028.0960 | 1.8047 |      0.0001 |      0.0008 |          6843 |          0.0005 |        0.5607 |

各分支 `metrics/` 保留完整 joint K=2–12 以及 independent K=5/8/10/12 指标；`clustering/` 中的文件夹、全量 labels 与空间图仅保留 K=5/8/10/12。其他 joint K 的指标 CSV 行仍保留，但其历史 `labels_path` 已随对应文件夹删除而失效。

### 4.1 Joint全K最佳内部指标

下表在现有joint K=2–12指标CSV中直接选择最优行，不重新聚类，也不恢复已删除的非绘图K文件夹。

| 方法         | 预处理       | best ASW K | ASW raw | ASW scaled | best CH K |        CH | best DBI K |    DBI | max section ARI K | section ARI | max section NMI K | section NMI |
| ------------ | ------------ | ---------: | ------: | ---------: | --------: | --------: | ---------: | -----: | ----------------: | ----------: | ----------------: | ----------: |
| spa_mo_model | Raw          |          3 |  0.1212 |     0.5606 |         2 |  851.0605 |          3 | 2.4353 |                 3 |      0.0002 |                 6 |      0.0006 |
| spa_mo_model | Standardized |          3 |  0.1132 |     0.5566 |         2 |  772.3735 |          3 | 2.5407 |                 3 |      0.0002 |                 6 |      0.0007 |
| MOFA+        | Raw          |          2 |  0.2990 |     0.6495 |         2 | 1008.9966 |         10 | 1.3502 |                11 |      0.0033 |                11 |      0.0035 |
| MOFA+        | Standardized |          2 |  0.2873 |     0.6436 |         2 |  920.1798 |         10 | 1.3814 |                11 |      0.0022 |                11 |      0.0029 |
| COSIE        | Raw          |          2 |  0.2437 |     0.6218 |         2 | 2612.3732 |          3 | 1.5106 |                 9 |      0.0045 |                 9 |      0.0056 |
| COSIE        | Standardized |          3 |  0.2050 |     0.6025 |         2 | 1882.5076 |          3 | 1.6905 |                 9 |      0.0047 |                 9 |      0.0057 |
| SpaMosaic    | Raw          |          2 |  0.3003 |     0.6501 |         2 | 3345.8710 |          3 | 1.3410 |                 5 |      0.0001 |                 7 |      0.0007 |
| SpaMosaic    | Standardized |          2 |  0.2654 |     0.6327 |         2 | 2687.4647 |          3 | 1.4773 |                 5 |      0.0001 |                12 |      0.0008 |

### 4.2 Independent逐section内部指标

同一方法、同一K、同一section的Raw与Standardized行严格相邻；ASW、CH和DBI均使用该section全量spot。

| 方法         |   K | section             | 预处理       | scope n | ASW n | ASW raw | ASW scaled | CH/DBI n |        CH |    DBI |
| ------------ | --: | ------------------- | ------------ | ------: | ----: | ------: | ---------: | -------: | --------: | -----: |
| spa_mo_model |   5 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1059 |     0.5529 |     3484 |  308.7096 | 2.4066 |
| spa_mo_model |   5 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1006 |     0.5503 |     3484 |  286.6152 | 2.4833 |
| spa_mo_model |   5 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.0956 |     0.5478 |     3359 |  258.5599 | 2.5533 |
| spa_mo_model |   5 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.0938 |     0.5469 |     3359 |  242.8902 | 2.6250 |
| spa_mo_model |   8 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.0886 |     0.5443 |     3484 |  232.6373 | 2.6874 |
| spa_mo_model |   8 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.0883 |     0.5441 |     3484 |  217.1653 | 2.7259 |
| spa_mo_model |   8 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.0815 |     0.5407 |     3359 |  199.7617 | 2.7202 |
| spa_mo_model |   8 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.0767 |     0.5384 |     3359 |  187.4309 | 2.7489 |
| spa_mo_model |  10 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.0936 |     0.5468 |     3484 |  201.9466 | 2.6507 |
| spa_mo_model |  10 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.0854 |     0.5427 |     3484 |  188.6109 | 2.8594 |
| spa_mo_model |  10 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.0795 |     0.5397 |     3359 |  173.7708 | 2.8837 |
| spa_mo_model |  10 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.0763 |     0.5381 |     3359 |  163.7465 | 2.9291 |
| spa_mo_model |  12 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.0843 |     0.5421 |     3484 |  178.9476 | 2.7253 |
| spa_mo_model |  12 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.0817 |     0.5408 |     3484 |  167.5783 | 2.7819 |
| spa_mo_model |  12 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.0766 |     0.5383 |     3359 |  152.9233 | 2.8273 |
| spa_mo_model |  12 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.0742 |     0.5371 |     3359 |  144.2829 | 2.8292 |
| MOFA+        |   5 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.2671 |     0.6335 |     3484 |  510.7545 | 1.5096 |
| MOFA+        |   5 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.2771 |     0.6385 |     3484 |  484.1026 | 1.3772 |
| MOFA+        |   5 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.2713 |     0.6357 |     3359 |  422.9141 | 1.4369 |
| MOFA+        |   5 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.2501 |     0.6251 |     3359 |  383.7473 | 1.5993 |
| MOFA+        |   8 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.2019 |     0.6009 |     3484 |  517.4325 | 1.4402 |
| MOFA+        |   8 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1861 |     0.5930 |     3484 |  477.3980 | 1.5285 |
| MOFA+        |   8 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1666 |     0.5833 |     3359 |  383.3025 | 1.5084 |
| MOFA+        |   8 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1583 |     0.5792 |     3359 |  375.4204 | 1.3584 |
| MOFA+        |  10 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1946 |     0.5973 |     3484 |  486.9065 | 1.4265 |
| MOFA+        |  10 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1825 |     0.5912 |     3484 |  452.4744 | 1.4723 |
| MOFA+        |  10 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1579 |     0.5789 |     3359 |  392.7085 | 1.4083 |
| MOFA+        |  10 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1479 |     0.5739 |     3359 |  380.2416 | 1.3097 |
| MOFA+        |  12 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1954 |     0.5977 |     3484 |  464.7091 | 1.2796 |
| MOFA+        |  12 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1763 |     0.5881 |     3484 |  429.3345 | 1.3760 |
| MOFA+        |  12 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1616 |     0.5808 |     3359 |  387.5269 | 1.3903 |
| MOFA+        |  12 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1526 |     0.5763 |     3359 |  365.5526 | 1.3978 |
| COSIE        |   5 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.2181 |     0.6090 |     3484 |  940.5499 | 1.5499 |
| COSIE        |   5 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.2020 |     0.6010 |     3484 |  732.2013 | 1.6508 |
| COSIE        |   5 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1968 |     0.5984 |     3359 |  718.8218 | 1.6430 |
| COSIE        |   5 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1765 |     0.5883 |     3359 |  568.0345 | 1.7923 |
| COSIE        |   8 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1893 |     0.5947 |     3484 |  725.5303 | 1.7317 |
| COSIE        |   8 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1738 |     0.5869 |     3484 |  564.8228 | 1.8384 |
| COSIE        |   8 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1764 |     0.5882 |     3359 |  572.2335 | 1.7501 |
| COSIE        |   8 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1599 |     0.5800 |     3359 |  445.6849 | 1.9559 |
| COSIE        |  10 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1751 |     0.5876 |     3484 |  637.5116 | 1.7769 |
| COSIE        |  10 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1696 |     0.5848 |     3484 |  494.2146 | 1.8456 |
| COSIE        |  10 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1669 |     0.5835 |     3359 |  498.6666 | 1.8537 |
| COSIE        |  10 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1585 |     0.5792 |     3359 |  391.8497 | 1.9578 |
| COSIE        |  12 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1677 |     0.5839 |     3484 |  567.6916 | 1.7896 |
| COSIE        |  12 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1569 |     0.5784 |     3484 |  442.9631 | 1.8878 |
| COSIE        |  12 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1649 |     0.5824 |     3359 |  448.1168 | 1.8370 |
| COSIE        |  12 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1525 |     0.5762 |     3359 |  351.4023 | 1.9834 |
| SpaMosaic    |   5 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.2134 |     0.6067 |     3484 | 1195.3624 | 1.5548 |
| SpaMosaic    |   5 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1823 |     0.5912 |     3484 |  918.6556 | 1.7738 |
| SpaMosaic    |   5 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.2069 |     0.6035 |     3359 | 1093.3643 | 1.5735 |
| SpaMosaic    |   5 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1757 |     0.5878 |     3359 |  838.8203 | 1.8073 |
| SpaMosaic    |   8 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1944 |     0.5972 |     3484 |  889.7074 | 1.6345 |
| SpaMosaic    |   8 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1693 |     0.5847 |     3484 |  706.3148 | 1.7012 |
| SpaMosaic    |   8 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1858 |     0.5929 |     3359 |  804.6988 | 1.6971 |
| SpaMosaic    |   8 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1628 |     0.5814 |     3359 |  632.2963 | 1.7658 |
| SpaMosaic    |  10 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1719 |     0.5860 |     3484 |  780.8666 | 1.6888 |
| SpaMosaic    |  10 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1544 |     0.5772 |     3484 |  616.1213 | 1.7541 |
| SpaMosaic    |  10 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1680 |     0.5840 |     3359 |  700.6493 | 1.6586 |
| SpaMosaic    |  10 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1467 |     0.5734 |     3359 |  548.0249 | 1.7638 |
| SpaMosaic    |  12 | Human_Lymph_Node_A1 | Raw          |    3484 |  3484 |  0.1576 |     0.5788 |     3484 |  694.9704 | 1.7083 |
| SpaMosaic    |  12 | Human_Lymph_Node_A1 | Standardized |    3484 |  3484 |  0.1569 |     0.5785 |     3484 |  549.9978 | 1.7482 |
| SpaMosaic    |  12 | Human_Lymph_Node_D1 | Raw          |    3359 |  3359 |  0.1479 |     0.5739 |     3359 |  622.7217 | 1.7325 |
| SpaMosaic    |  12 | Human_Lymph_Node_D1 | Standardized |    3359 |  3359 |  0.1337 |     0.5669 |     3359 |  488.6055 | 1.8639 |

### 4.3 Joint与Independent空间连续性

每个值为A1和D1两个section全量spot空间近邻同簇比例的算术平均；Raw与Standardized行严格相邻。

| 方法         |   K | 模式        | 预处理       | Spatial neighbor agreement |
| ------------ | --: | ----------- | ------------ | -------------------------: |
| spa_mo_model |   5 | Joint       | Raw          |                     0.7053 |
| spa_mo_model |   5 | Joint       | Standardized |                     0.7093 |
| spa_mo_model |   5 | Independent | Raw          |                     0.7050 |
| spa_mo_model |   5 | Independent | Standardized |                     0.7085 |
| spa_mo_model |   8 | Joint       | Raw          |                     0.5749 |
| spa_mo_model |   8 | Joint       | Standardized |                     0.5598 |
| spa_mo_model |   8 | Independent | Raw          |                     0.5577 |
| spa_mo_model |   8 | Independent | Standardized |                     0.5633 |
| spa_mo_model |  10 | Joint       | Raw          |                     0.4790 |
| spa_mo_model |  10 | Joint       | Standardized |                     0.4685 |
| spa_mo_model |  10 | Independent | Raw          |                     0.5028 |
| spa_mo_model |  10 | Independent | Standardized |                     0.4728 |
| spa_mo_model |  12 | Joint       | Raw          |                     0.4634 |
| spa_mo_model |  12 | Joint       | Standardized |                     0.4447 |
| spa_mo_model |  12 | Independent | Raw          |                     0.4611 |
| spa_mo_model |  12 | Independent | Standardized |                     0.4443 |
| MOFA+        |   5 | Joint       | Raw          |                     0.5148 |
| MOFA+        |   5 | Joint       | Standardized |                     0.5180 |
| MOFA+        |   5 | Independent | Raw          |                     0.5316 |
| MOFA+        |   5 | Independent | Standardized |                     0.5131 |
| MOFA+        |   8 | Joint       | Raw          |                     0.4474 |
| MOFA+        |   8 | Joint       | Standardized |                     0.4383 |
| MOFA+        |   8 | Independent | Raw          |                     0.4269 |
| MOFA+        |   8 | Independent | Standardized |                     0.4263 |
| MOFA+        |  10 | Joint       | Raw          |                     0.4293 |
| MOFA+        |  10 | Joint       | Standardized |                     0.3824 |
| MOFA+        |  10 | Independent | Raw          |                     0.3738 |
| MOFA+        |  10 | Independent | Standardized |                     0.3735 |
| MOFA+        |  12 | Joint       | Raw          |                     0.3392 |
| MOFA+        |  12 | Joint       | Standardized |                     0.3458 |
| MOFA+        |  12 | Independent | Raw          |                     0.3510 |
| MOFA+        |  12 | Independent | Standardized |                     0.3440 |
| COSIE        |   5 | Joint       | Raw          |                     0.7550 |
| COSIE        |   5 | Joint       | Standardized |                     0.7647 |
| COSIE        |   5 | Independent | Raw          |                     0.7596 |
| COSIE        |   5 | Independent | Standardized |                     0.7681 |
| COSIE        |   8 | Joint       | Raw          |                     0.6243 |
| COSIE        |   8 | Joint       | Standardized |                     0.6580 |
| COSIE        |   8 | Independent | Raw          |                     0.6460 |
| COSIE        |   8 | Independent | Standardized |                     0.6577 |
| COSIE        |  10 | Joint       | Raw          |                     0.5810 |
| COSIE        |  10 | Joint       | Standardized |                     0.6130 |
| COSIE        |  10 | Independent | Raw          |                     0.5908 |
| COSIE        |  10 | Independent | Standardized |                     0.6263 |
| COSIE        |  12 | Joint       | Raw          |                     0.5574 |
| COSIE        |  12 | Joint       | Standardized |                     0.5760 |
| COSIE        |  12 | Independent | Raw          |                     0.5637 |
| COSIE        |  12 | Independent | Standardized |                     0.6001 |
| SpaMosaic    |   5 | Joint       | Raw          |                     0.7022 |
| SpaMosaic    |   5 | Joint       | Standardized |                     0.6987 |
| SpaMosaic    |   5 | Independent | Raw          |                     0.7016 |
| SpaMosaic    |   5 | Independent | Standardized |                     0.6991 |
| SpaMosaic    |   8 | Joint       | Raw          |                     0.6297 |
| SpaMosaic    |   8 | Joint       | Standardized |                     0.6263 |
| SpaMosaic    |   8 | Independent | Raw          |                     0.6321 |
| SpaMosaic    |   8 | Independent | Standardized |                     0.6262 |
| SpaMosaic    |  10 | Joint       | Raw          |                     0.5753 |
| SpaMosaic    |  10 | Joint       | Standardized |                     0.5815 |
| SpaMosaic    |  10 | Independent | Raw          |                     0.5801 |
| SpaMosaic    |  10 | Independent | Standardized |                     0.5825 |
| SpaMosaic    |  12 | Joint       | Raw          |                     0.5431 |
| SpaMosaic    |  12 | Joint       | Standardized |                     0.5607 |
| SpaMosaic    |  12 | Independent | Raw          |                     0.5448 |
| SpaMosaic    |  12 | Independent | Standardized |                     0.5554 |


## 5. 两种预处理的标签稳定性

| 方法         |   K | raw vs standardized ARI |    NMI |
| ------------ | --: | ----------------------: | -----: |
| spa_mo_model |   5 |                  0.9250 | 0.9053 |
| spa_mo_model |   8 |                  0.8939 | 0.8745 |
| spa_mo_model |  10 |                  0.8574 | 0.8608 |
| spa_mo_model |  12 |                  0.7152 | 0.7672 |
| MOFA+        |   5 |                  0.9454 | 0.8780 |
| MOFA+        |   8 |                  0.8804 | 0.8925 |
| MOFA+        |  10 |                  0.5108 | 0.7197 |
| MOFA+        |  12 |                  0.8520 | 0.8759 |
| COSIE        |   5 |                  0.8909 | 0.8792 |
| COSIE        |   8 |                  0.5454 | 0.6980 |
| COSIE        |  10 |                  0.7310 | 0.7916 |
| COSIE        |  12 |                  0.6411 | 0.7367 |
| SpaMosaic    |   5 |                  0.9080 | 0.8872 |
| SpaMosaic    |   8 |                  0.8680 | 0.8617 |
| SpaMosaic    |  10 |                  0.8091 | 0.8291 |
| SpaMosaic    |  12 |                  0.5935 | 0.7297 |

完整 joint K=2–12 及 independent K=5/8/10/12 稳定性见各方法 `comparison_metrics/label_stability.csv`；逐项指标差值见 `metric_deltas.csv`。

## 6. 共享指标及参数

### 6.1 Batch样本范围

| 方法         | batch key | n total | n used | n batches | batch categories                        | batch counts                                               | max samples | requested n | seed | embedding scaled | bASW n |
| ------------ | --------- | ------: | -----: | --------: | --------------------------------------- | ---------------------------------------------------------- | ----------: | ----------: | ---: | ---------------- | -----: |
| spa_mo_model | section   |    6843 |   6843 |         2 | Human_Lymph_Node_A1,Human_Lymph_Node_D1 | {'Human_Lymph_Node_A1': 3484, 'Human_Lymph_Node_D1': 3359} |           0 |           0 |    0 | True             |   6843 |
| MOFA+        | group     |    6843 |   6843 |         2 | Human_Lymph_Node_A1,Human_Lymph_Node_D1 | NA                                                         |           0 |           0 |    0 | NA               |   6843 |
| COSIE        | group     |    6843 |   6843 |         2 | Human_Lymph_Node_A1,Human_Lymph_Node_D1 | NA                                                         |           0 |           0 |    0 | NA               |   6843 |
| SpaMosaic    | group     |    6843 |   6843 |         2 | Human_Lymph_Node_A1,Human_Lymph_Node_D1 | NA                                                         |           0 |           0 |    0 | NA               |   6843 |

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
| spa_mo_model |   0.0011 |     0.9989 |    1.9331 |           0.9331 |         0.1759 |          0.8241 |       0.0011 |    0.9989 |
| MOFA+        |   0.0011 |     0.9989 |    1.9204 |           0.9204 |         0.2442 |          0.7558 |       0.0000 |    1.0000 |
| COSIE        |   0.0085 |     0.9915 |    1.7600 |           0.7600 |         0.6608 |          0.3392 |       0.0067 |    0.9933 |
| SpaMosaic    |   0.0005 |     0.9995 |    1.9493 |           0.9493 |         0.1697 |          0.8303 |       0.0006 |    0.9994 |

这些 batch 指标不随本次 KMeans 输入预处理变化，故只保存一份；来源与复制文件的 SHA256 位于 `shared_metrics/source_manifest.csv`。

### 6.4 MOFA+视图解释度R²

| View | Group               |      R2 |
| ---- | ------------------- | ------: |
| ADT  | Human_Lymph_Node_A1 | 75.8021 |
| ADT  | Human_Lymph_Node_D1 | 76.3448 |
| RNA  | Human_Lymph_Node_A1 |  2.6175 |
| RNA  | Human_Lymph_Node_D1 |  1.9539 |

该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

### 6.5 SpaMosaic模态对齐

| pair       | group               | n spots | mean cosine | median cosine | std cosine |
| ---------- | ------------------- | ------: | ----------: | ------------: | ---------: |
| adt_vs_rna | ALL                 |    6843 |      0.0903 |        0.0904 |     0.0160 |
| adt_vs_rna | Human_Lymph_Node_A1 |    3484 |      0.0923 |        0.0922 |     0.0155 |
| adt_vs_rna | Human_Lymph_Node_D1 |    3359 |      0.0883 |        0.0886 |     0.0163 |

该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

## 7. 结论与使用建议

1. 两套预处理结果均已作为一等结果保存在原始文件体系中，而非只写入报告；标签、指标、图、配置和 scaler 参数可独立复核。
2. 进行方法主比较时，应固定选择同一分支；不要把 raw 分支某方法与 standardized 分支另一方法直接排名。
3. 建议正文以 standardized 分支作为尺度可比的主结果，raw 分支作为敏感性分析；同时报告 raw-vs-standardized ARI/NMI，说明结论对预处理的稳健程度。
4. Human_Lymph_Node 缺少统一真实区域标签，因此 section ARI/NMI 是批次/切片分离诊断，不是生物学聚类准确率。
