# MouseBrain：Raw 与 Standardized embedding + KMeans 对比报告

- 生成时间：2026-07-27T14:17:37.498175+08:00
- 本报告仅引用本次新增 comparison 结果；旧分析目录和原综合报告未覆盖。
- 两分支分别在自身 KMeans 输入空间计算距离型指标；跨方案变化应结合标签稳定性解释。

## 1. 新结果目录

| 方法         | comparison root                                                           | 方法自有数据副本                                          |
| ------------ | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/mousebrain_preprocessing_comparison   | /home/hujinlan/spa_mo_model/data/dataset_MouseBrain       |
| MOFA+        | /home/hujinlan/mofa+/analysis/mousebrain_preprocessing_comparison         | /home/hujinlan/mofa+/data/dataset_MouseBrain              |
| COSIE        | /home/hujinlan/cosie_runs/mousebrain_preprocessing_comparison             | /home/hujinlan/cosie/data/dataset_MouseBrain              |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mousebrain_preprocessing_comparison | /home/hujinlan/SpaMosaic-dev/demo/data/dataset_MouseBrain |

每个根目录均含 `raw_embedding/`、`standardized_embedding/`、`comparison_metrics/`、`shared_metrics/` 和根级 manifest。

## 2. 统一标准

| 项目                         | 统一值                                                   |
| ---------------------------- | -------------------------------------------------------- |
| spot/barcode                 | 各方法自有数据；section + original barcode 精确对齐      |
| spot 数                      | 7866（s1=2384，s2=2820，s3=2662）                        |
| raw 分支                     | 原始最终 embedding 直接输入 KMeans                       |
| standardized 分支            | StandardScaler；joint 全体拟合，independent 每切片拟合   |
| KMeans                       | sklearn exact KMeans                                     |
| joint 指标 K                 | 2–12                                                     |
| independent 指标 K           | 5、6、8、10                                              |
| 保留的聚类文件夹/labels/图 K | 5、6、8、10                                              |
| 随机参数                     | seed=0，n_init=20，max_iter=300                          |
| ASW/CH/DBI                   | KMeans 同一输入空间，全量样本                            |
| 外部指标                     | 5类生物标签 + group(section来源)诊断；缺失标签逐标签剔除 |
| Label ASW                    | 对应输入空间内全部非缺失标签 spot                        |
| 空间连续性                   | 每切片全点建图，exact 6-NN                               |
| 保留 K 的图与指标            | 复用同一套已保存 label CSV                               |
| batch 指标                   | 不依赖本次 KMeans 分支；复制并记录 SHA256                |

### 与旧 MouseBrain 结果的关系

旧分析脚本/结果存在不同默认种子（MOFA+=1、COSIE=1、SpaMosaic=1234）以及 spa_mo_model 部分 ASW 使用 5,000 spot 抽样等口径差异。本次 raw 和 standardized 标签、外部指标与内部指标均按统一 seed=0 和全量样本重新生成，未复制旧聚类数值；报告中的聚类数值只来自本次新增目录。batch、MOFA+ R2 和 SpaMosaic 模态对齐属于本次 KMeans 预处理无关指标，才以带 SHA256 的方式复制到 `shared_metrics/`。

## 3. 数据、标签与 embedding 对齐

| 方法         | 维度 |   s1 |   s2 |   s3 | RegionLoupe n | celltype n | group n |
| ------------ | ---: | ---: | ---: | ---: | ------------: | ---------: | ------: |
| spa_mo_model |  128 | 2384 | 2820 | 2662 |          7866 |       5046 |    7866 |
| MOFA+        |   10 | 2384 | 2820 | 2662 |          7866 |       5046 |    7866 |
| COSIE        |  384 | 2384 | 2820 | 2662 |          7866 |       5046 |    7866 |
| SpaMosaic    |   32 | 2384 | 2820 | 2662 |          7866 |       5046 |    7866 |

### 最终 embedding 来源

- spa_mo_model
  - `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s1_final_embedding.npy`
  - `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s2_final_embedding.npy`
  - `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s3_final_embedding.npy`
- MOFA+
  - `/home/hujinlan/mofa+/analysis/mousebrain_mofa_rna_meta_uni_hvg2000_k10_iter1000/tables/factors_with_metadata_and_coordinates.csv`
- COSIE
  - `/home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/analysis/tables/embeddings_with_metadata.csv`
  - `/home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/final_embeddings/s1_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/final_embeddings/s2_final_embedding.npy`
  - `/home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/final_embeddings/s3_final_embedding.npy`
- SpaMosaic
  - `/home/hujinlan/SpaMosaic-dev/runs/mousebrain_spamosaic/mousebrain_spamosaic_embeddings.h5ad`

## 4. 外部指标：Raw / Standardized 上下对齐对比

同一方法、同一标签的 Raw 与 Standardized 行相邻，列定义完全一致；group定义为对齐后的section来源向量s1/s2/s3，属于切片依赖诊断。

| 方法         | 标签        | 预处理       | best K | 有效 n |    ARI |    NMI | V-measure | Label ASW scaled |
| ------------ | ----------- | ------------ | -----: | -----: | -----: | -----: | --------: | ---------------: |
| spa_mo_model | RegionLoupe | Raw          |      5 |   7866 | 0.7633 | 0.6768 |    0.6768 |           0.5569 |
| spa_mo_model | RegionLoupe | Standardized |      5 |   7866 | 0.7654 | 0.6782 |    0.6782 |           0.5539 |
| spa_mo_model | annotations | Raw          |      5 |   7866 | 0.6950 | 0.6300 |    0.6300 |           0.5500 |
| spa_mo_model | annotations | Standardized |      5 |   7866 | 0.6975 | 0.6312 |    0.6312 |           0.5465 |
| spa_mo_model | celltype    | Raw          |      5 |   5046 | 0.6737 | 0.6228 |    0.6228 |           0.5488 |
| spa_mo_model | celltype    | Standardized |      5 |   5046 | 0.6743 | 0.6223 |    0.6223 |           0.5453 |
| spa_mo_model | Y.l1        | Raw          |      5 |   7866 | 0.7633 | 0.6768 |    0.6768 |           0.5569 |
| spa_mo_model | Y.l1        | Standardized |      5 |   7866 | 0.7654 | 0.6782 |    0.6782 |           0.5539 |
| spa_mo_model | Y           | Raw          |      5 |   7866 | 0.4194 | 0.4903 |    0.4903 |           0.5012 |
| spa_mo_model | Y           | Standardized |      5 |   7866 | 0.4216 | 0.4918 |    0.4918 |           0.5014 |
| spa_mo_model | group       | Raw          |      5 |   7866 | 0.0036 | 0.0035 |    0.0035 |           0.5008 |
| spa_mo_model | group       | Standardized |      5 |   7866 | 0.0036 | 0.0036 |    0.0036 |           0.5008 |
| MOFA+        | RegionLoupe | Raw          |      5 |   7866 | 0.3027 | 0.3862 |    0.3862 |           0.4969 |
| MOFA+        | RegionLoupe | Standardized |      3 |   7866 | 0.3406 | 0.3455 |    0.3455 |           0.4980 |
| MOFA+        | annotations | Raw          |      5 |   7866 | 0.2440 | 0.3593 |    0.3593 |           0.4904 |
| MOFA+        | annotations | Standardized |      3 |   7866 | 0.2715 | 0.3166 |    0.3166 |           0.4915 |
| MOFA+        | celltype    | Raw          |      5 |   5046 | 0.2411 | 0.3542 |    0.3542 |           0.4930 |
| MOFA+        | celltype    | Standardized |      3 |   5046 | 0.2709 | 0.3116 |    0.3116 |           0.4942 |
| MOFA+        | Y.l1        | Raw          |      5 |   7866 | 0.3027 | 0.3862 |    0.3862 |           0.4969 |
| MOFA+        | Y.l1        | Standardized |      3 |   7866 | 0.3406 | 0.3455 |    0.3455 |           0.4980 |
| MOFA+        | Y           | Raw          |      5 |   7866 | 0.1563 | 0.2777 |    0.2777 |           0.4760 |
| MOFA+        | Y           | Standardized |      3 |   7866 | 0.1773 | 0.2532 |    0.2532 |           0.4769 |
| MOFA+        | group       | Raw          |     12 |   7866 | 0.0043 | 0.0100 |    0.0100 |           0.4951 |
| MOFA+        | group       | Standardized |     12 |   7866 | 0.0034 | 0.0079 |    0.0079 |           0.4956 |
| COSIE        | RegionLoupe | Raw          |      4 |   7866 | 0.6628 | 0.6079 |    0.6079 |           0.5513 |
| COSIE        | RegionLoupe | Standardized |      7 |   7866 | 0.5568 | 0.6279 |    0.6279 |           0.5430 |
| COSIE        | annotations | Raw          |      4 |   7866 | 0.5881 | 0.5629 |    0.5629 |           0.5436 |
| COSIE        | annotations | Standardized |      7 |   7866 | 0.4988 | 0.5879 |    0.5879 |           0.5384 |
| COSIE        | celltype    | Raw          |      4 |   5046 | 0.5687 | 0.5554 |    0.5554 |           0.5473 |
| COSIE        | celltype    | Standardized |      7 |   5046 | 0.4934 | 0.5839 |    0.5839 |           0.5423 |
| COSIE        | Y.l1        | Raw          |      4 |   7866 | 0.6628 | 0.6079 |    0.6079 |           0.5513 |
| COSIE        | Y.l1        | Standardized |      7 |   7866 | 0.5568 | 0.6279 |    0.6279 |           0.5430 |
| COSIE        | Y           | Raw          |      4 |   7866 | 0.3467 | 0.4181 |    0.4181 |           0.5017 |
| COSIE        | Y           | Standardized |      7 |   7866 | 0.3300 | 0.4605 |    0.4605 |           0.5026 |
| COSIE        | group       | Raw          |      9 |   7866 | 0.0134 | 0.0201 |    0.0201 |           0.5026 |
| COSIE        | group       | Standardized |     10 |   7866 | 0.0338 | 0.0587 |    0.0587 |           0.5031 |
| SpaMosaic    | RegionLoupe | Raw          |      3 |   7866 | 0.4822 | 0.4765 |    0.4765 |           0.5068 |
| SpaMosaic    | RegionLoupe | Standardized |      3 |   7866 | 0.4696 | 0.4707 |    0.4707 |           0.5245 |
| SpaMosaic    | annotations | Raw          |      3 |   7866 | 0.4196 | 0.4390 |    0.4390 |           0.4907 |
| SpaMosaic    | annotations | Standardized |     11 |   7866 | 0.4783 | 0.5656 |    0.5656 |           0.5216 |
| SpaMosaic    | celltype    | Raw          |      3 |   5046 | 0.3914 | 0.4317 |    0.4317 |           0.4959 |
| SpaMosaic    | celltype    | Standardized |     11 |   5046 | 0.4664 | 0.5598 |    0.5598 |           0.5254 |
| SpaMosaic    | Y.l1        | Raw          |      3 |   7866 | 0.4822 | 0.4765 |    0.4765 |           0.5068 |
| SpaMosaic    | Y.l1        | Standardized |      3 |   7866 | 0.4696 | 0.4707 |    0.4707 |           0.5245 |
| SpaMosaic    | Y           | Raw          |      3 |   7866 | 0.2454 | 0.3144 |    0.3144 |           0.4344 |
| SpaMosaic    | Y           | Standardized |     11 |   7866 | 0.2815 | 0.4086 |    0.4086 |           0.4648 |
| SpaMosaic    | group       | Raw          |      4 |   7866 | 0.0021 | 0.0015 |    0.0015 |           0.4962 |
| SpaMosaic    | group       | Standardized |     11 |   7866 | 0.0030 | 0.0035 |    0.0035 |           0.4966 |

best K 在每个预处理分支内按 joint ARI 独立选择；完整 K=2–12 外部指标见各分支 `metrics/clustering_metrics_by_label.csv`。

### group完整诊断：按ARI选择best K

下表补齐原综合报告中的group诊断。Raw与Standardized严格相邻；group反映section来源依赖，不应作为生物聚类准确率解释。

| 方法         | 预处理       | best K | 有效 n |    ARI |    NMI | Homogeneity | Completeness | V-measure | Cluster ASW raw | Cluster ASW scaled | Label ASW raw | Label ASW scaled |
| ------------ | ------------ | -----: | -----: | -----: | -----: | ----------: | -----------: | --------: | --------------: | -----------------: | ------------: | ---------------: |
| spa_mo_model | Raw          |      5 |   7866 | 0.0036 | 0.0035 |      0.0041 |       0.0030 |    0.0035 |          0.1704 |             0.5852 |        0.0016 |           0.5008 |
| spa_mo_model | Standardized |      5 |   7866 | 0.0036 | 0.0036 |      0.0042 |       0.0031 |    0.0036 |          0.1635 |             0.5817 |        0.0016 |           0.5008 |
| MOFA+        | Raw          |     12 |   7866 | 0.0043 | 0.0100 |      0.0159 |       0.0073 |    0.0100 |          0.1282 |             0.5641 |       -0.0097 |           0.4951 |
| MOFA+        | Standardized |     12 |   7866 | 0.0034 | 0.0079 |      0.0126 |       0.0058 |    0.0079 |          0.1258 |             0.5629 |       -0.0088 |           0.4956 |
| COSIE        | Raw          |      9 |   7866 | 0.0134 | 0.0201 |      0.0299 |       0.0152 |    0.0201 |          0.1389 |             0.5695 |        0.0053 |           0.5026 |
| COSIE        | Standardized |     10 |   7866 | 0.0338 | 0.0587 |      0.0900 |       0.0435 |    0.0587 |          0.1219 |             0.5610 |        0.0062 |           0.5031 |
| SpaMosaic    | Raw          |      4 |   7866 | 0.0021 | 0.0015 |      0.0016 |       0.0014 |    0.0015 |          0.4278 |             0.7139 |       -0.0076 |           0.4962 |
| SpaMosaic    | Standardized |     11 |   7866 | 0.0030 | 0.0035 |      0.0050 |       0.0027 |    0.0035 |          0.2737 |             0.6369 |       -0.0068 |           0.4966 |

group完整K=2–12结果、labels来源/哈希及KMeans参数见各分支 `metrics/group_diagnostic_metrics.csv`；逐K的Standardized-minus-Raw差值见 `comparison_metrics/group_diagnostic_metrics_deltas.csv`。

## 5. 内部、切片诊断与空间指标：Raw / Standardized 上下对齐对比

同一方法、同一 K 的 Raw 与 Standardized 行相邻。

| 方法         |   K | 预处理       | Cluster ASW scaled |         CH |    DBI | section ARI | section NMI | joint spatial |
| ------------ | --: | ------------ | -----------------: | ---------: | -----: | ----------: | ----------: | ------------: |
| spa_mo_model |   5 | Raw          |             0.5852 |  1032.6746 | 2.0604 |      0.0036 |      0.0035 |        0.8654 |
| spa_mo_model |   5 | Standardized |             0.5817 |   981.8246 | 2.1025 |      0.0036 |      0.0036 |        0.8644 |
| spa_mo_model |   6 | Raw          |             0.5732 |   953.3293 | 2.1618 |      0.0026 |      0.0034 |        0.8217 |
| spa_mo_model |   6 | Standardized |             0.5721 |   910.7665 | 2.1834 |      0.0027 |      0.0035 |        0.8233 |
| spa_mo_model |   8 | Raw          |             0.5754 |   819.2566 | 2.1601 |      0.0020 |      0.0037 |        0.7737 |
| spa_mo_model |   8 | Standardized |             0.5736 |   783.3636 | 2.1878 |      0.0020 |      0.0037 |        0.7769 |
| spa_mo_model |  10 | Raw          |             0.5698 |   715.7262 | 2.2082 |      0.0022 |      0.0048 |        0.7159 |
| spa_mo_model |  10 | Standardized |             0.5660 |   681.8217 | 2.2325 |      0.0024 |      0.0041 |        0.7032 |
| MOFA+        |   5 | Raw          |             0.5638 |   925.9960 | 1.9095 |      0.0023 |      0.0036 |        0.7399 |
| MOFA+        |   5 | Standardized |             0.5599 |   773.7902 | 2.0404 |      0.0022 |      0.0039 |        0.7356 |
| MOFA+        |   6 | Raw          |             0.5646 |   884.5595 | 1.9279 |      0.0023 |      0.0037 |        0.6959 |
| MOFA+        |   6 | Standardized |             0.5634 |   766.3907 | 1.9172 |      0.0033 |      0.0052 |        0.7085 |
| MOFA+        |   8 | Raw          |             0.5656 |   799.5314 | 1.7865 |      0.0038 |      0.0065 |        0.5201 |
| MOFA+        |   8 | Standardized |             0.5636 |   708.5626 | 1.7769 |      0.0022 |      0.0042 |        0.4557 |
| MOFA+        |  10 | Raw          |             0.5659 |   740.7207 | 1.6887 |      0.0029 |      0.0064 |        0.4192 |
| MOFA+        |  10 | Standardized |             0.5658 |   674.6059 | 1.6712 |      0.0026 |      0.0059 |        0.4213 |
| COSIE        |   5 | Raw          |             0.5867 |  1242.0699 | 2.0123 |      0.0058 |      0.0065 |        0.8519 |
| COSIE        |   5 | Standardized |             0.5734 |   958.1248 | 2.2186 |      0.0067 |      0.0077 |        0.8490 |
| COSIE        |   6 | Raw          |             0.5857 |  1134.2532 | 1.9352 |      0.0054 |      0.0066 |        0.8472 |
| COSIE        |   6 | Standardized |             0.5720 |   865.9596 | 2.1737 |      0.0047 |      0.0062 |        0.8469 |
| COSIE        |   8 | Raw          |             0.5803 |   950.3276 | 1.9981 |      0.0048 |      0.0062 |        0.8178 |
| COSIE        |   8 | Standardized |             0.5666 |   732.6402 | 2.2671 |      0.0161 |      0.0250 |        0.7775 |
| COSIE        |  10 | Raw          |             0.5697 |   825.4865 | 2.1314 |      0.0116 |      0.0188 |        0.7378 |
| COSIE        |  10 | Standardized |             0.5610 |   637.2423 | 2.3962 |      0.0338 |      0.0587 |        0.7366 |
| SpaMosaic    |   5 | Raw          |             0.6979 | 10236.8949 | 0.8255 |      0.0011 |      0.0012 |        0.8426 |
| SpaMosaic    |   5 | Standardized |             0.6775 |  5859.4249 | 1.1079 |      0.0018 |      0.0015 |        0.8532 |
| SpaMosaic    |   6 | Raw          |             0.7020 |  9823.0081 | 0.8418 |      0.0013 |      0.0013 |        0.8277 |
| SpaMosaic    |   6 | Standardized |             0.6584 |  5511.5143 | 1.1072 |      0.0014 |      0.0015 |        0.8297 |
| SpaMosaic    |   8 | Raw          |             0.6636 |  8883.1331 | 0.9422 |      0.0007 |      0.0009 |        0.7720 |
| SpaMosaic    |   8 | Standardized |             0.6355 |  4829.5398 | 1.1798 |      0.0016 |      0.0022 |        0.7910 |
| SpaMosaic    |  10 | Raw          |             0.6404 |  7985.0545 | 1.1034 |      0.0009 |      0.0014 |        0.7266 |
| SpaMosaic    |  10 | Standardized |             0.6312 |  4433.9250 | 1.1619 |      0.0025 |      0.0030 |        0.7889 |

各分支 `metrics/` 保留完整 joint K=2–12 以及 independent K=5/6/8/10 指标；`clustering/` 中的文件夹、全量 labels 与空间图仅保留 K=5/6/8/10。其他 joint K 的指标 CSV 行仍保留，但其历史 `labels_path` 已随对应文件夹删除而失效。

## 6. 两种预处理的标签稳定性

| 方法         |   K | raw vs standardized ARI |    NMI |
| ------------ | --: | ----------------------: | -----: |
| spa_mo_model |   5 |                  0.9827 | 0.9650 |
| spa_mo_model |   6 |                  0.9653 | 0.9507 |
| spa_mo_model |   8 |                  0.9570 | 0.9451 |
| spa_mo_model |  10 |                  0.8364 | 0.8431 |
| MOFA+        |   5 |                  0.8434 | 0.8130 |
| MOFA+        |   6 |                  0.8365 | 0.8151 |
| MOFA+        |   8 |                  0.5450 | 0.6204 |
| MOFA+        |  10 |                  0.8118 | 0.8232 |
| COSIE        |   5 |                  0.9110 | 0.8897 |
| COSIE        |   6 |                  0.9285 | 0.9144 |
| COSIE        |   8 |                  0.6519 | 0.7820 |
| COSIE        |  10 |                  0.6764 | 0.7643 |
| SpaMosaic    |   5 |                  0.6742 | 0.7079 |
| SpaMosaic    |   6 |                  0.8941 | 0.8527 |
| SpaMosaic    |   8 |                  0.5931 | 0.7039 |
| SpaMosaic    |  10 |                  0.4236 | 0.6159 |

完整joint/independent标签稳定性、内部/外部指标差值及group专用逐K差值分别见各方法 `comparison_metrics/`。

## 7. 共享指标

| 方法         | n used |   bASW |  bLISI |   kBET | PCR score |
| ------------ | -----: | -----: | -----: | -----: | --------: |
| spa_mo_model |   7866 | 0.9984 | 0.7302 | 0.2400 |    0.9866 |
| MOFA+        |   7866 | 0.9912 | 0.8188 | 0.4327 |    1.0000 |
| COSIE        |   7866 | 0.9938 | 0.4925 | 0.0679 |    0.9684 |
| SpaMosaic    |   7866 | 0.9932 | 0.8025 | 0.4237 |    0.9994 |

MOFA+ R2 与 SpaMosaic 模态对齐也保存在各自 `shared_metrics/`；复制来源及 SHA256 见 `source_manifest.csv`。

## 8. 结论与使用建议

1. raw 与 standardized 均作为完整、可复核的一等结果保存，不仅记录在报告中。
2. 正式跨方法排名必须固定同一预处理分支；建议 standardized 作为主分析、raw 作为敏感性分析。
3. 同时报告 raw-vs-standardized ARI/NMI，可区分尺度变化造成的指标变化与实际聚类重分配。
4. 生物标签外部指标使用逐标签非缺失样本；group及section ARI/NMI仅是切片来源依赖诊断，不是生物准确率。
