# Mouse_Spleen：Raw 与 Standardized embedding + KMeans 对比报告

- 生成时间：2026-07-27T14:56:23.067031+08:00
- 本报告只引用新增 comparison 结果；旧分析目录和原综合报告未覆盖。
- 两套方案分别在自身KMeans输入空间计算距离型指标；跨方案变化应结合标签稳定性解释。

## 1. 新结果目录

| 方法         | comparison root                                                             | 方法自有数据副本                                    |
| ------------ | --------------------------------------------------------------------------- | --------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_spleen_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/Mouse_Spleen       |
| MOFA+        | /home/hujinlan/mofa+/analysis/mouse_spleen_preprocessing_comparison         | /home/hujinlan/mofa+/data/Mouse_Spleen              |
| COSIE        | /home/hujinlan/cosie_runs/mouse_spleen_preprocessing_comparison             | /home/hujinlan/cosie/data/Mouse_Spleen              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mouse_spleen_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/Mouse_Spleen |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。

## 2. 统一标准

| 项目                            | 统一值                                                  |
| ------------------------------- | ------------------------------------------------------- |
| spot/barcode                    | 各方法自有数据；section + original barcode 精确对齐     |
| spot数                          | 5336（Mouse_Spleen1=2568，Mouse_Spleen2=2768）          |
| embedding                       | 各方法当前最终embedding                                 |
| raw分支                         | 原始最终embedding直接输入KMeans                         |
| standardized分支                | StandardScaler；joint全体拟合，independent每section拟合 |
| KMeans                          | sklearn exact KMeans                                    |
| joint/independent指标K          | 2–12                                                    |
| 保留的聚类文件夹/labels/空间图K | 5、8、10、12                                            |
| 空间图点大小                    | 10                                                      |
| 随机参数                        | seed=0，n_init=20，max_iter=300                         |
| Cluster ASW/CH/DBI              | 每个scope全量样本                                       |
| Label ASW                       | 无真实细胞/区域标签，不计算                             |
| 空间连续性                      | 每section全点，exact 6-NN                               |
| labels复用                      | 保留K的内部、空间指标和图复用保存的全量labels           |
| batch指标                       | KMeans分支无关；复制原结果并记录SHA256                  |

## 3. 数据与 embedding 对齐

| 方法         | 维度 | Mouse_Spleen1 | Mouse_Spleen2 | 总spot |
| ------------ | ---: | ------------: | ------------: | -----: |
| spa_mo_model |  128 |          2568 |          2768 |   5336 |
| MOFA+        |   10 |          2568 |          2768 |   5336 |
| COSIE        |  256 |          2568 |          2768 |   5336 |
| SpaMosaic    |   32 |          2568 |          2768 |   5336 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Spleen1.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/selected_spot_indices_Mouse_Spleen1.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/spatial_Mouse_Spleen1.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/final_embeddings_Mouse_Spleen2.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/selected_spot_indices_Mouse_Spleen2.npy`
  - `/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/spatial_Mouse_Spleen2.npy`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/mouse_spleen_mofa_hvg2000_k10_iter1000/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/analysis/tables/embeddings_with_metadata.csv`
  - `/home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/final_embeddings/Mouse_Spleen1_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/final_embeddings/Mouse_Spleen2_final_embedding.npy`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/mouse_spleen_spamosaic/mouse_spleen_spamosaic_embeddings.h5ad`

## 4. Joint内部与空间指标：Raw / Standardized上下对齐

同一方法、同一K的Raw与Standardized行相邻，所有列采用完全相同定义。

| 方法         |   K | 预处理       | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |        CH |    DBI | section ARI | section NMI | section ASW n | section ASW raw | joint spatial |
| ------------ | --: | ------------ | ----: | --------------: | -----------------: | -------: | --------: | -----: | ----------: | ----------: | ------------: | --------------: | ------------: |
| spa_mo_model |   5 | Raw          |  5336 |          0.0560 |             0.5280 |     5336 |  314.6511 | 3.4470 |     -0.0002 |      0.0003 |          5336 |          0.0001 |        0.5245 |
| spa_mo_model |   5 | Standardized |  5336 |          0.0549 |             0.5274 |     5336 |  300.0178 | 3.4821 |     -0.0002 |      0.0003 |          5336 |          0.0001 |        0.5208 |
| spa_mo_model |   8 | Raw          |  5336 |          0.0542 |             0.5271 |     5336 |  222.4236 | 3.6720 |     -0.0002 |      0.0003 |          5336 |          0.0001 |        0.3736 |
| spa_mo_model |   8 | Standardized |  5336 |          0.0529 |             0.5265 |     5336 |  213.3212 | 3.6439 |     -0.0002 |      0.0003 |          5336 |          0.0001 |        0.3688 |
| spa_mo_model |  10 | Raw          |  5336 |          0.0547 |             0.5274 |     5336 |  186.8534 | 3.6446 |     -0.0001 |      0.0004 |          5336 |          0.0001 |        0.3196 |
| spa_mo_model |  10 | Standardized |  5336 |          0.0506 |             0.5253 |     5336 |  180.0497 | 3.6596 |     -0.0001 |      0.0003 |          5336 |          0.0001 |        0.3199 |
| spa_mo_model |  12 | Raw          |  5336 |          0.0525 |             0.5262 |     5336 |  162.1944 | 3.6362 |     -0.0001 |      0.0005 |          5336 |          0.0001 |        0.2791 |
| spa_mo_model |  12 | Standardized |  5336 |          0.0465 |             0.5232 |     5336 |  156.2063 | 3.6348 |     -0.0001 |      0.0004 |          5336 |          0.0001 |        0.2844 |
| MOFA+        |   5 | Raw          |  5336 |          0.3040 |             0.6520 |     5336 | 1703.1988 | 1.0100 |      0.0020 |      0.0048 |          5336 |          0.0021 |        0.6326 |
| MOFA+        |   5 | Standardized |  5336 |          0.1190 |             0.5595 |     5336 |  587.0984 | 2.0589 |      0.0002 |      0.0014 |          5336 |          0.0022 |        0.4121 |
| MOFA+        |   8 | Raw          |  5336 |          0.2230 |             0.6115 |     5336 | 1517.6499 | 1.1947 |      0.0049 |      0.0085 |          5336 |          0.0021 |        0.4412 |
| MOFA+        |   8 | Standardized |  5336 |          0.1068 |             0.5534 |     5336 |  485.4208 | 1.8664 |      0.0016 |      0.0041 |          5336 |          0.0022 |        0.3156 |
| MOFA+        |  10 | Raw          |  5336 |          0.2139 |             0.6069 |     5336 | 1430.7264 | 1.1653 |      0.0031 |      0.0064 |          5336 |          0.0021 |        0.3796 |
| MOFA+        |  10 | Standardized |  5336 |          0.1016 |             0.5508 |     5336 |  434.6779 | 1.7909 |      0.0033 |      0.0063 |          5336 |          0.0022 |        0.2877 |
| MOFA+        |  12 | Raw          |  5336 |          0.2036 |             0.6018 |     5336 | 1327.3178 | 1.2427 |      0.0032 |      0.0063 |          5336 |          0.0021 |        0.3431 |
| MOFA+        |  12 | Standardized |  5336 |          0.0979 |             0.5490 |     5336 |  396.3998 | 1.8110 |      0.0024 |      0.0051 |          5336 |          0.0022 |        0.2463 |
| COSIE        |   5 | Raw          |  5336 |          0.1971 |             0.5986 |     5336 | 1598.7065 | 1.7243 |      0.0016 |      0.0021 |          5336 |          0.0086 |        0.6191 |
| COSIE        |   5 | Standardized |  5336 |          0.1548 |             0.5774 |     5336 |  975.1406 | 2.0580 |      0.0034 |      0.0038 |          5336 |          0.0105 |        0.6311 |
| COSIE        |   8 | Raw          |  5336 |          0.1417 |             0.5708 |     5336 | 1150.5246 | 2.0276 |      0.0027 |      0.0039 |          5336 |          0.0086 |        0.5106 |
| COSIE        |   8 | Standardized |  5336 |          0.1310 |             0.5655 |     5336 |  699.0797 | 2.1311 |      0.0024 |      0.0036 |          5336 |          0.0105 |        0.5577 |
| COSIE        |  10 | Raw          |  5336 |          0.1277 |             0.5638 |     5336 |  976.2318 | 2.0833 |      0.0022 |      0.0034 |          5336 |          0.0086 |        0.4779 |
| COSIE        |  10 | Standardized |  5336 |          0.1170 |             0.5585 |     5336 |  593.5245 | 2.2079 |      0.0059 |      0.0165 |          5336 |          0.0105 |        0.5122 |
| COSIE        |  12 | Raw          |  5336 |          0.1258 |             0.5629 |     5336 |  845.8275 | 2.1020 |      0.0040 |      0.0100 |          5336 |          0.0086 |        0.4525 |
| COSIE        |  12 | Standardized |  5336 |          0.0976 |             0.5488 |     5336 |  513.9031 | 2.4399 |      0.0054 |      0.0152 |          5336 |          0.0105 |        0.4638 |
| SpaMosaic    |   5 | Raw          |  5336 |          0.2134 |             0.6067 |     5336 | 2063.4699 | 1.7357 |      0.0004 |      0.0010 |          5336 |          0.0011 |        0.5508 |
| SpaMosaic    |   5 | Standardized |  5336 |          0.1843 |             0.5922 |     5336 | 1366.6379 | 1.9805 |      0.0004 |      0.0010 |          5336 |          0.0008 |        0.5518 |
| SpaMosaic    |   8 | Raw          |  5336 |          0.1516 |             0.5758 |     5336 | 1412.4116 | 2.0910 |      0.0006 |      0.0016 |          5336 |          0.0011 |        0.4408 |
| SpaMosaic    |   8 | Standardized |  5336 |          0.1506 |             0.5753 |     5336 |  943.8552 | 2.1736 |      0.0008 |      0.0022 |          5336 |          0.0008 |        0.4415 |
| SpaMosaic    |  10 | Raw          |  5336 |          0.1435 |             0.5717 |     5336 | 1167.1821 | 2.2139 |      0.0005 |      0.0018 |          5336 |          0.0011 |        0.3954 |
| SpaMosaic    |  10 | Standardized |  5336 |          0.1320 |             0.5660 |     5336 |  787.6468 | 2.3046 |      0.0005 |      0.0016 |          5336 |          0.0008 |        0.3870 |
| SpaMosaic    |  12 | Raw          |  5336 |          0.1429 |             0.5714 |     5336 |  997.3757 | 2.3097 |      0.0006 |      0.0025 |          5336 |          0.0011 |        0.3820 |
| SpaMosaic    |  12 | Standardized |  5336 |          0.1263 |             0.5632 |     5336 |  679.2975 | 2.2983 |      0.0004 |      0.0016 |          5336 |          0.0008 |        0.3570 |

完整joint/independent K=2–12内部与空间指标仍保存在各分支 `metrics/`；聚类文件夹、全量labels和空间图仅保留K=5/8/10/12，点大小统一为10。CSV中其他K的历史`labels_path`已随对应文件夹删除而失效。

### 4.1 Joint全K最佳内部指标

下表在现有joint K=2–12指标CSV中直接选择最优行，不重新聚类，也不恢复已删除的非绘图K文件夹。

| 方法         | 预处理       | best ASW K | ASW raw | ASW scaled | best CH K |        CH | best DBI K |    DBI | max section ARI K | section ARI | max section NMI K | section NMI |
| ------------ | ------------ | ---------: | ------: | ---------: | --------: | --------: | ---------: | -----: | ----------------: | ----------: | ----------------: | ----------: |
| spa_mo_model | Raw          |          2 |  0.1016 |     0.5508 |         2 |  619.8434 |          3 | 2.7904 |                 2 |      0.0002 |                12 |      0.0005 |
| spa_mo_model | Standardized |          2 |  0.0945 |     0.5472 |         2 |  571.9147 |          3 | 2.8641 |                 2 |      0.0000 |                 6 |      0.0005 |
| MOFA+        | Raw          |          4 |  0.3050 |     0.6525 |         4 | 1858.1768 |          5 | 1.0100 |                 8 |      0.0049 |                 8 |      0.0085 |
| MOFA+        | Standardized |          2 |  0.1348 |     0.5674 |         2 |  888.5984 |         11 | 1.7823 |                 9 |      0.0037 |                 9 |      0.0068 |
| COSIE        | Raw          |          2 |  0.2962 |     0.6481 |         2 | 2783.9117 |          2 | 1.3354 |                 2 |      0.0043 |                11 |      0.0107 |
| COSIE        | Standardized |          2 |  0.2084 |     0.6042 |         2 | 1603.8081 |          2 | 1.7891 |                10 |      0.0059 |                10 |      0.0165 |
| SpaMosaic    | Raw          |          2 |  0.3696 |     0.6848 |         2 | 3922.9683 |          2 | 1.1331 |                 3 |      0.0015 |                12 |      0.0025 |
| SpaMosaic    | Standardized |          2 |  0.2949 |     0.6474 |         2 | 2478.0284 |          2 | 1.4218 |                 3 |      0.0016 |                 8 |      0.0022 |

### 4.2 Independent逐section内部指标

展示保留聚类结果的K=5/8/10/12；同一方法、同一K、同一section的Raw与Standardized行严格相邻。ASW、CH和DBI均使用该section全量spot。

| 方法         |   K | section       | 预处理       | scope n | ASW n | Cluster ASW raw | Cluster ASW scaled | CH/DBI n |        CH |    DBI |
| ------------ | --: | ------------- | ------------ | ------: | ----: | --------------: | -----------------: | -------: | --------: | -----: |
| spa_mo_model |   5 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.0559 |             0.5280 |     2568 |  153.0593 | 3.4612 |
| spa_mo_model |   5 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.0544 |             0.5272 |     2568 |  145.8830 | 3.4780 |
| spa_mo_model |   5 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.0568 |             0.5284 |     2768 |  162.1975 | 3.4515 |
| spa_mo_model |   5 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.0548 |             0.5274 |     2768 |  154.6339 | 3.5188 |
| spa_mo_model |   8 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.0536 |             0.5268 |     2568 |  108.4835 | 3.6186 |
| spa_mo_model |   8 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.0534 |             0.5267 |     2568 |  104.0332 | 3.6181 |
| spa_mo_model |   8 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.0553 |             0.5277 |     2768 |  114.6843 | 3.6440 |
| spa_mo_model |   8 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.0533 |             0.5266 |     2768 |  110.0002 | 3.6510 |
| spa_mo_model |  10 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.0525 |             0.5263 |     2568 |   90.9722 | 3.5874 |
| spa_mo_model |  10 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.0541 |             0.5271 |     2568 |   87.7414 | 3.6042 |
| spa_mo_model |  10 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.0549 |             0.5274 |     2768 |   96.6306 | 3.6186 |
| spa_mo_model |  10 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.0516 |             0.5258 |     2768 |   93.1430 | 3.5868 |
| spa_mo_model |  12 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.0508 |             0.5254 |     2568 |   79.1354 | 3.5426 |
| spa_mo_model |  12 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.0521 |             0.5261 |     2568 |   76.3396 | 3.6197 |
| spa_mo_model |  12 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.0502 |             0.5251 |     2768 |   83.7278 | 3.5688 |
| spa_mo_model |  12 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.0494 |             0.5247 |     2768 |   80.9101 | 3.6554 |
| MOFA+        |   5 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2415 |             0.6207 |     2568 |  843.6779 | 1.2228 |
| MOFA+        |   5 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1220 |             0.5610 |     2568 |  294.9554 | 1.9971 |
| MOFA+        |   5 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.3040 |             0.6520 |     2768 |  975.9084 | 0.9936 |
| MOFA+        |   5 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1170 |             0.5585 |     2768 |  304.7173 | 2.0723 |
| MOFA+        |   8 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2310 |             0.6155 |     2568 |  757.1953 | 1.2070 |
| MOFA+        |   8 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1145 |             0.5572 |     2568 |  245.7628 | 1.8427 |
| MOFA+        |   8 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.2221 |             0.6111 |     2768 |  827.0812 | 1.2348 |
| MOFA+        |   8 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1123 |             0.5561 |     2768 |  254.4298 | 1.8427 |
| MOFA+        |  10 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2231 |             0.6115 |     2568 |  706.1948 | 1.1585 |
| MOFA+        |  10 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1062 |             0.5531 |     2568 |  221.4470 | 1.8137 |
| MOFA+        |  10 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.2157 |             0.6078 |     2768 |  774.0069 | 1.2588 |
| MOFA+        |  10 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1065 |             0.5532 |     2768 |  234.5255 | 1.7672 |
| MOFA+        |  12 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2148 |             0.6074 |     2568 |  662.0021 | 1.2322 |
| MOFA+        |  12 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1057 |             0.5528 |     2568 |  202.1870 | 1.7808 |
| MOFA+        |  12 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.2169 |             0.6085 |     2768 |  734.0291 | 1.2029 |
| MOFA+        |  12 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1020 |             0.5510 |     2768 |  215.5088 | 1.7593 |
| COSIE        |   5 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2033 |             0.6017 |     2568 |  801.2832 | 1.6527 |
| COSIE        |   5 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1611 |             0.5805 |     2568 |  477.9200 | 1.9853 |
| COSIE        |   5 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1989 |             0.5994 |     2768 |  835.3065 | 1.7180 |
| COSIE        |   5 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1606 |             0.5803 |     2768 |  518.2417 | 2.0548 |
| COSIE        |   8 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1760 |             0.5880 |     2568 |  590.0781 | 1.7353 |
| COSIE        |   8 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1439 |             0.5720 |     2568 |  350.6243 | 2.0344 |
| COSIE        |   8 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1441 |             0.5721 |     2768 |  598.1258 | 2.0350 |
| COSIE        |   8 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1159 |             0.5580 |     2768 |  371.8924 | 2.2628 |
| COSIE        |  10 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1539 |             0.5769 |     2568 |  504.8568 | 1.8875 |
| COSIE        |  10 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1245 |             0.5623 |     2568 |  297.8539 | 2.2371 |
| COSIE        |  10 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1252 |             0.5626 |     2768 |  506.0964 | 2.1520 |
| COSIE        |  10 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1048 |             0.5524 |     2768 |  315.2708 | 2.4012 |
| COSIE        |  12 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1317 |             0.5658 |     2568 |  439.3531 | 2.1158 |
| COSIE        |  12 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1044 |             0.5522 |     2568 |  258.5410 | 2.4039 |
| COSIE        |  12 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1154 |             0.5577 |     2768 |  438.0729 | 2.1559 |
| COSIE        |  12 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1013 |             0.5507 |     2768 |  272.6219 | 2.3824 |
| SpaMosaic    |   5 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.2185 |             0.6092 |     2568 | 1029.7506 | 1.6618 |
| SpaMosaic    |   5 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1864 |             0.5932 |     2568 |  671.8165 | 1.9349 |
| SpaMosaic    |   5 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.2124 |             0.6062 |     2768 | 1038.8921 | 1.7788 |
| SpaMosaic    |   5 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1840 |             0.5920 |     2768 |  698.1149 | 2.0107 |
| SpaMosaic    |   8 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1683 |             0.5842 |     2568 |  708.3319 | 2.0396 |
| SpaMosaic    |   8 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1520 |             0.5760 |     2568 |  466.3042 | 2.1661 |
| SpaMosaic    |   8 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1570 |             0.5785 |     2768 |  715.2345 | 2.0658 |
| SpaMosaic    |   8 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1514 |             0.5757 |     2768 |  484.2849 | 2.1870 |
| SpaMosaic    |  10 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1438 |             0.5719 |     2568 |  586.7485 | 2.2217 |
| SpaMosaic    |  10 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1380 |             0.5690 |     2568 |  388.8655 | 2.3235 |
| SpaMosaic    |  10 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1476 |             0.5738 |     2768 |  593.2988 | 2.2239 |
| SpaMosaic    |  10 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1387 |             0.5693 |     2768 |  403.9530 | 2.2774 |
| SpaMosaic    |  12 | Mouse_Spleen1 | Raw          |    2568 |  2568 |          0.1395 |             0.5697 |     2568 |  500.7396 | 2.2858 |
| SpaMosaic    |  12 | Mouse_Spleen1 | Standardized |    2568 |  2568 |          0.1260 |             0.5630 |     2568 |  336.8852 | 2.3197 |
| SpaMosaic    |  12 | Mouse_Spleen2 | Raw          |    2768 |  2768 |          0.1277 |             0.5639 |     2768 |  507.9310 | 2.2846 |
| SpaMosaic    |  12 | Mouse_Spleen2 | Standardized |    2768 |  2768 |          0.1299 |             0.5650 |     2768 |  349.7639 | 2.2983 |

### 4.3 Joint与Independent空间连续性

展示K=5/8/10/12；每个值为两个section全量spot空间近邻同簇比例的算术平均，Raw与Standardized行严格相邻。

| 方法         |   K | 模式        | 预处理       | Spatial neighbor agreement |
| ------------ | --: | ----------- | ------------ | -------------------------: |
| spa_mo_model |   5 | Joint       | Raw          |                     0.5245 |
| spa_mo_model |   5 | Joint       | Standardized |                     0.5208 |
| spa_mo_model |   5 | Independent | Raw          |                     0.5205 |
| spa_mo_model |   5 | Independent | Standardized |                     0.5194 |
| spa_mo_model |   8 | Joint       | Raw          |                     0.3736 |
| spa_mo_model |   8 | Joint       | Standardized |                     0.3688 |
| spa_mo_model |   8 | Independent | Raw          |                     0.3742 |
| spa_mo_model |   8 | Independent | Standardized |                     0.3704 |
| spa_mo_model |  10 | Joint       | Raw          |                     0.3196 |
| spa_mo_model |  10 | Joint       | Standardized |                     0.3199 |
| spa_mo_model |  10 | Independent | Raw          |                     0.3143 |
| spa_mo_model |  10 | Independent | Standardized |                     0.3238 |
| spa_mo_model |  12 | Joint       | Raw          |                     0.2791 |
| spa_mo_model |  12 | Joint       | Standardized |                     0.2844 |
| spa_mo_model |  12 | Independent | Raw          |                     0.2876 |
| spa_mo_model |  12 | Independent | Standardized |                     0.2826 |
| MOFA+        |   5 | Joint       | Raw          |                     0.6326 |
| MOFA+        |   5 | Joint       | Standardized |                     0.4121 |
| MOFA+        |   5 | Independent | Raw          |                     0.5969 |
| MOFA+        |   5 | Independent | Standardized |                     0.4129 |
| MOFA+        |   8 | Joint       | Raw          |                     0.4412 |
| MOFA+        |   8 | Joint       | Standardized |                     0.3156 |
| MOFA+        |   8 | Independent | Raw          |                     0.4343 |
| MOFA+        |   8 | Independent | Standardized |                     0.3396 |
| MOFA+        |  10 | Joint       | Raw          |                     0.3796 |
| MOFA+        |  10 | Joint       | Standardized |                     0.2877 |
| MOFA+        |  10 | Independent | Raw          |                     0.3905 |
| MOFA+        |  10 | Independent | Standardized |                     0.2796 |
| MOFA+        |  12 | Joint       | Raw          |                     0.3431 |
| MOFA+        |  12 | Joint       | Standardized |                     0.2463 |
| MOFA+        |  12 | Independent | Raw          |                     0.3538 |
| MOFA+        |  12 | Independent | Standardized |                     0.2562 |
| COSIE        |   5 | Joint       | Raw          |                     0.6191 |
| COSIE        |   5 | Joint       | Standardized |                     0.6311 |
| COSIE        |   5 | Independent | Raw          |                     0.6227 |
| COSIE        |   5 | Independent | Standardized |                     0.6367 |
| COSIE        |   8 | Joint       | Raw          |                     0.5106 |
| COSIE        |   8 | Joint       | Standardized |                     0.5577 |
| COSIE        |   8 | Independent | Raw          |                     0.5343 |
| COSIE        |   8 | Independent | Standardized |                     0.5529 |
| COSIE        |  10 | Joint       | Raw          |                     0.4779 |
| COSIE        |  10 | Joint       | Standardized |                     0.5122 |
| COSIE        |  10 | Independent | Raw          |                     0.4894 |
| COSIE        |  10 | Independent | Standardized |                     0.5006 |
| COSIE        |  12 | Joint       | Raw          |                     0.4525 |
| COSIE        |  12 | Joint       | Standardized |                     0.4638 |
| COSIE        |  12 | Independent | Raw          |                     0.4461 |
| COSIE        |  12 | Independent | Standardized |                     0.4655 |
| SpaMosaic    |   5 | Joint       | Raw          |                     0.5508 |
| SpaMosaic    |   5 | Joint       | Standardized |                     0.5518 |
| SpaMosaic    |   5 | Independent | Raw          |                     0.5558 |
| SpaMosaic    |   5 | Independent | Standardized |                     0.5524 |
| SpaMosaic    |   8 | Joint       | Raw          |                     0.4408 |
| SpaMosaic    |   8 | Joint       | Standardized |                     0.4415 |
| SpaMosaic    |   8 | Independent | Raw          |                     0.4494 |
| SpaMosaic    |   8 | Independent | Standardized |                     0.4464 |
| SpaMosaic    |  10 | Joint       | Raw          |                     0.3954 |
| SpaMosaic    |  10 | Joint       | Standardized |                     0.3870 |
| SpaMosaic    |  10 | Independent | Raw          |                     0.4013 |
| SpaMosaic    |  10 | Independent | Standardized |                     0.4028 |
| SpaMosaic    |  12 | Joint       | Raw          |                     0.3820 |
| SpaMosaic    |  12 | Joint       | Standardized |                     0.3570 |
| SpaMosaic    |  12 | Independent | Raw          |                     0.3655 |
| SpaMosaic    |  12 | Independent | Standardized |                     0.3622 |

## 5. 两种预处理的标签稳定性

| 方法         |   K | raw vs standardized ARI |    NMI |
| ------------ | --: | ----------------------: | -----: |
| spa_mo_model |   5 |                  0.8893 | 0.8776 |
| spa_mo_model |   8 |                  0.8046 | 0.8013 |
| spa_mo_model |  10 |                  0.6258 | 0.6721 |
| spa_mo_model |  12 |                  0.5922 | 0.6434 |
| MOFA+        |   5 |                  0.2156 | 0.2753 |
| MOFA+        |   8 |                  0.2128 | 0.3397 |
| MOFA+        |  10 |                  0.1810 | 0.3218 |
| MOFA+        |  12 |                  0.1784 | 0.3248 |
| COSIE        |   5 |                  0.8441 | 0.8308 |
| COSIE        |   8 |                  0.6941 | 0.7521 |
| COSIE        |  10 |                  0.6445 | 0.7459 |
| COSIE        |  12 |                  0.6706 | 0.7553 |
| SpaMosaic    |   5 |                  0.8703 | 0.8583 |
| SpaMosaic    |   8 |                  0.6865 | 0.7348 |
| SpaMosaic    |  10 |                  0.7506 | 0.7529 |
| SpaMosaic    |  12 |                  0.7138 | 0.7029 |

完整joint和independent K=2–12稳定性见各方法 `comparison_metrics/label_stability.csv`；内部指标差值见 `metric_deltas.csv`。

## 6. 共享指标及参数

### 6.1 Batch样本范围

| 方法         | batch key | n total | n used | n batches | batch categories            | batch counts                                   | max samples | requested n | seed | embedding scaled | bASW n |
| ------------ | --------- | ------: | -----: | --------: | --------------------------- | ---------------------------------------------- | ----------: | ----------: | ---: | ---------------- | -----: |
| spa_mo_model | section   |    5336 |   5336 |         2 | Mouse_Spleen1,Mouse_Spleen2 | {'Mouse_Spleen1': 2568, 'Mouse_Spleen2': 2768} |       50000 |       50000 |    0 | True             |   5336 |
| MOFA+        | group     |    5336 |   5336 |         2 | Mouse_Spleen1,Mouse_Spleen2 | NA                                             |       50000 |           0 |    0 | NA               |   5336 |
| COSIE        | group     |    5336 |   5336 |         2 | Mouse_Spleen1,Mouse_Spleen2 | NA                                             |       50000 |           0 |    0 | NA               |   5336 |
| SpaMosaic    | group     |    5336 |   5336 |         2 | Mouse_Spleen1,Mouse_Spleen2 | NA                                             |       50000 |           0 |    0 | NA               |   5336 |

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
| spa_mo_model |   0.0001 |     0.9999 |    1.9818 |           0.9818 |         0.0305 |          0.9695 |       0.0002 |    0.9998 |
| MOFA+        |   0.0022 |     0.9978 |    1.9227 |           0.9227 |         0.2644 |          0.7356 |       0.0000 |    1.0000 |
| COSIE        |   0.0105 |     0.9895 |    1.8115 |           0.8115 |         0.5870 |          0.4130 |       0.0098 |    0.9902 |
| SpaMosaic    |   0.0008 |     0.9992 |    1.9638 |           0.9638 |         0.1379 |          0.8621 |       0.0010 |    0.9990 |

batch指标不随本次KMeans输入分支变化，故只保存一份；来源和复制文件SHA256见 `shared_metrics/source_manifest.csv`。PCR PCs按 `min(50, embedding维度)`，因此MOFA+=10、SpaMosaic=32，其余为50。

### 6.4 MOFA+视图解释度R²

| View | Group         |      R2 |
| ---- | ------------- | ------: |
| ADT  | Mouse_Spleen1 | 79.0515 |
| ADT  | Mouse_Spleen2 | 80.2855 |
| RNA  | Mouse_Spleen1 |  1.4051 |
| RNA  | Mouse_Spleen2 |  1.4625 |

该指标由MOFA+模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

### 6.5 SpaMosaic模态对齐

| pair       | group         | n spots | mean cosine | median cosine | std cosine |
| ---------- | ------------- | ------: | ----------: | ------------: | ---------: |
| adt_vs_rna | ALL           |    5336 |      0.0412 |        0.0414 |     0.0115 |
| adt_vs_rna | Mouse_Spleen1 |    2568 |      0.0407 |        0.0411 |     0.0113 |
| adt_vs_rna | Mouse_Spleen2 |    2768 |      0.0418 |        0.0418 |     0.0117 |

该指标由SpaMosaic模型输出决定，与本次KMeans输入预处理无关，因此只保存并展示一份。

## 7. 结论与使用建议

1. Raw与Standardized均保留joint/independent K=2–12指标CSV；聚类文件夹与全量labels仅保留K=5/8/10/12。
2. 跨方法排名必须固定同一预处理分支；建议Standardized作为主分析，Raw作为敏感性分析。
3. 同时报告raw-vs-standardized ARI/NMI，以说明聚类分配对预处理的稳健程度。
4. Mouse_Spleen没有统一真实区域标签；section ARI/NMI是切片来源依赖诊断，不是生物学聚类准确率。
