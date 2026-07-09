# spa_mo_model、MOFA+、COSIE、SpaMosaic 在 MouseBrain、CRC、MISAR-seq、Human_Lymph_Node、Mouse_Spleen 与 Mouse_Thymus 上的结果对比报告

生成时间：2026-07-01；MISAR-seq 结果更新于 2026-07-02；Human_Lymph_Node、Mouse_Spleen、Mouse_Thymus 及 spa_mo_model 公共指标更新于 2026-07-09。本文按照用户指定目录读取已有实验结果，并补算/汇总了 `spa_mo_model` 与 baseline 的聚类、空间连续性、section diagnostic 和 batch correction 指标；未修改原始数据集。

## 结果目录
| 数据集                 | 方法           | 目录                                                                                                                                 |
| ------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| MouseBrain          | spa_mo_model | /home/hujinlan/spa_mo_model/results/mousebrain_test                                                                                |
| MouseBrain          | MOFA+        | /home/hujinlan/mofa+/analysis/mousebrain_mofa_rna_meta_uni_hvg2000_k10_iter1000                                                    |
| MouseBrain          | COSIE        | /home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full                                                                        |
| MouseBrain          | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mousebrain_spamosaic                                                                         |
| CRC Stereo-CITE-seq | spa_mo_model | /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42 |
| CRC Stereo-CITE-seq | MOFA+        | /home/hujinlan/mofa+/analysis/crc_stereocite_mofa_hvg2000_k10_full_iter1000_cpu_float64                                            |
| CRC Stereo-CITE-seq | COSIE        | /home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse                                                                    |
| CRC Stereo-CITE-seq | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/crc_stereocite_spamosaic_full_gpu_ce                                                         |
| MISAR-seq           | spa_mo_model | /home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42                         |
| MISAR-seq           | MOFA+        | /home/hujinlan/mofa+/analysis/misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000                                                |
| MISAR-seq           | COSIE        | /home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/analysis                                                                      |
| MISAR-seq           | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/misar_seq_spamosaic_full                                                                    |
| Human_Lymph_Node    | spa_mo_model | /home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis |
| Human_Lymph_Node    | MOFA+        | /home/hujinlan/mofa+/analysis/human_lymph_node_mofa_hvg2000_k10_iter1000                                                         |
| Human_Lymph_Node    | COSIE        | /home/hujinlan/cosie_runs/human_lymph_node_cosie_rna_adt_full/analysis                                                           |
| Human_Lymph_Node    | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/human_lymph_node_spamosaic                                                                 |
| Mouse_Spleen        | spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis     |
| Mouse_Spleen        | MOFA+        | /home/hujinlan/mofa+/analysis/mouse_spleen_mofa_hvg2000_k10_iter1000                                                             |
| Mouse_Spleen        | COSIE        | /home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/analysis                                                               |
| Mouse_Spleen        | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mouse_spleen_spamosaic                                                                     |
| Mouse_Thymus        | spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis     |
| Mouse_Thymus        | MOFA+        | /home/hujinlan/mofa+/analysis/mouse_thymus_mofa_hvg2000_k10_iter1000                                                             |
| Mouse_Thymus        | COSIE        | /home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/analysis                                                               |
| Mouse_Thymus        | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mouse_thymus_spamosaic                                                                     |

## 指标解释与可比性说明

- ARI：Adjusted Rand Index，衡量聚类与已知标签的一致性，并对随机一致性做校正；越高越好。MouseBrain 有组织/细胞类型标签，因此可以做外部聚类评价。
- NMI：Normalized Mutual Information，衡量聚类与标签共享的信息量；越高越好。NMI 对标签粒度和簇数的敏感性与 ARI 不完全相同。
- Homogeneity / Completeness / V-measure：分别评价簇纯度、同一标签是否被集中到少数簇中，以及二者的调和平均；越高越好。
- ASW：Average Silhouette Width，原始 silhouette 范围为 [-1, 1]。本报告所有 ASW 对比均使用 `(原始值 + 1) / 2` 得到的 `ASW scaled`，范围为 [0, 1]；越高表示簇内更紧、簇间更分离。
- Cluster ASW 与 Label ASW：Cluster ASW 用预测聚类标签计算，反映聚类结构本身；Label ASW 用已知标签计算，反映真实标签在 embedding 空间中的分离程度。
- CH：Calinski-Harabasz 指数，越高通常表示簇间分离相对簇内离散更强，但受样本量、维度和 k 影响较大。
- DBI：Davies-Bouldin Index，越低越好，表示簇内离散与簇间距离的相对比值。
- group/section ARI/NMI：聚类与切片或样本来源的一致性诊断，不是单纯越高越好。若目标是跨样本混合，过高可能提示 batch/section effect；若样本之间确有生物差异，也可能包含真实结构。
- Spatial neighbor agreement：每个点的空间近邻中有多少比例属于同一聚类；越高表示空间连续性更强，但过高也可能意味着过度平滑。
- BASW / BLISI / kBET / PCR：batch correction 诊断指标，本文对 embedding 按 section/group 作为 batch 计算。BASW score、BLISI normalized、kBET acceptance rate、PCR score 越高通常表示 batch mixing 越好；kBET rejection rate 和 PCR batch R2 越低越好。spa_mo_model 的该批指标由本项目分析脚本基于标准化 embedding 计算，并已对齐 `/home/hujinlan/mofa+/scripts/batch_correction_metrics.py` 的计算口径：MouseBrain 使用全量 7,866 spots；CRC 的 bLISI/kBET/PCR 使用全量 612,374 spots 和 HNSW L2 approximate 近邻，CRC bASW 使用 10,000 spot silhouette sample。

四个方法的训练目标和配置不同：潜变量维度、HVG 数量、训练轮数、Harmony/metacell/OT/CE loss 等设置均不完全一致。因此这些表格适合做 baseline 级别横向参考，不应解释为严格受控的消融实验。

Human_Lymph_Node、Mouse_Spleen 与 Mouse_Thymus 的原始 `obs` 不包含可靠细胞类型或组织区域真值，因此只报告无监督内部指标、空间连续性、section diagnostic 和 batch correction；不把 section ARI/NMI 误写成生物学聚类准确率。

## MouseBrain

### 配置摘要
| 方法           | 主要设置                                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | HE/RNA/Metabolite 三模态；final embedding 128 维；结果为 fullspot_warmup_schedule/epochs_200；run_summary 记录训练 200 epoch、CUDA、Harmony=True。 |
| MOFA+        | RNA/meta/UNI 三视图；10 factors；iter1000；目录名显示 RNA HVG2000。                                                                           |
| COSIE        | HE/RNA/Metabolite 三模态；embedding 384 维；600 epoch；RNA HVG3000；Harmony=True。                                                         |
| SpaMosaic    | RNA/meta/uni_feature 三模态；merged_emb 32 维；本次 SpaMosaic 运行 100 epoch；RNA HVG2000。                                                   |


### MouseBrain 外部聚类指标：按 ARI 选择 best k
ASW 列均为 `(原始 silhouette + 1) / 2`。`spa_mo_model` 的 ARI/NMI/ASW 是从最终 embedding 补算得到。
| 方法           | 标签          | best_k | ARI    | NMI    | V-measure | Cluster ASW scaled | Label ASW scaled |
| ------------ | ----------- | ------ | ------ | ------ | --------- | ------------------ | ---------------- |
| spa_mo_model | RegionLoupe | 5      | 0.7654 | 0.6782 | 0.6782    | 0.5825             | 0.5541           |
| MOFA+        | RegionLoupe | 4      | 0.3684 | 0.3933 | 0.3933    | 0.5608             | 0.4966           |
| COSIE        | RegionLoupe | 7      | 0.5568 | 0.6279 | 0.6279    | 0.5747             | 0.5446           |
| SpaMosaic    | RegionLoupe | 3      | 0.4695 | 0.4705 | 0.4705    | 0.7053             | 0.5211           |
| spa_mo_model | annotations | 5      | 0.6975 | 0.6312 | 0.6312    | 0.5825             | 0.5460           |
| MOFA+        | annotations | 4      | 0.2952 | 0.3619 | 0.3619    | 0.5608             | 0.4902           |
| COSIE        | annotations | 7      | 0.4988 | 0.5879 | 0.5879    | 0.5747             | 0.5404           |
| SpaMosaic    | annotations | 11     | 0.4691 | 0.5662 | 0.5662    | 0.6347             | 0.5172           |
| spa_mo_model | celltype    | 5      | 0.6743 | 0.6223 | 0.6223    | 0.5825             | 0.5454           |
| MOFA+        | celltype    | 4      | 0.2969 | 0.3607 | 0.3607    | 0.5608             | 0.4930           |
| COSIE        | celltype    | 7      | 0.4934 | 0.5839 | 0.5839    | 0.5747             | 0.5436           |
| SpaMosaic    | celltype    | 11     | 0.4580 | 0.5592 | 0.5592    | 0.6347             | 0.5190           |
| spa_mo_model | Y.l1        | 5      | 0.7654 | 0.6782 | 0.6782    | 0.5825             | 0.5541           |
| MOFA+        | Y.l1        | 4      | 0.3684 | 0.3933 | 0.3933    | 0.5608             | 0.4966           |
| COSIE        | Y.l1        | 7      | 0.5568 | 0.6279 | 0.6279    | 0.5747             | 0.5446           |
| SpaMosaic    | Y.l1        | 3      | 0.4695 | 0.4705 | 0.4705    | 0.7053             | 0.5211           |
| spa_mo_model | Y           | 5      | 0.4216 | 0.4918 | 0.4918    | 0.5825             | 0.5021           |
| MOFA+        | Y           | 4      | 0.1929 | 0.2916 | 0.2916    | 0.5608             | 0.4749           |
| COSIE        | Y           | 7      | 0.3300 | 0.4605 | 0.4605    | 0.5747             | 0.5028           |
| SpaMosaic    | Y           | 11     | 0.2769 | 0.4085 | 0.4085    | 0.6347             | 0.4661           |
| spa_mo_model | group       | 5      | 0.0036 | 0.0036 | 0.0036    | 0.5825             | 0.5008           |
| MOFA+        | group       | 6      | 0.0033 | 0.0052 | 0.0052    | 0.5642             | 0.4955           |
| COSIE        | group       | 11     | 0.0298 | 0.0518 | 0.0518    | 0.5637             | 0.5039           |
| SpaMosaic    | group       | 11     | 0.0028 | 0.0035 | 0.0035    | 0.6347             | 0.4946           |


### MouseBrain ARI 对比
| 标签          | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----------- | ------------ | ------ | ------ | --------- |
| RegionLoupe | 0.7654       | 0.3684 | 0.5568 | 0.4695    |
| annotations | 0.6975       | 0.2952 | 0.4988 | 0.4691    |
| celltype    | 0.6743       | 0.2969 | 0.4934 | 0.4580    |
| Y.l1        | 0.7654       | 0.3684 | 0.5568 | 0.4695    |
| Y           | 0.4216       | 0.1929 | 0.3300 | 0.2769    |
| group       | 0.0036       | 0.0033 | 0.0298 | 0.0028    |

spa_mo_model 在本次 MouseBrain 补算中 ARI 最高：RegionLoupe/Y.l1 为 0.7654，annotations 为 0.6975，celltype 为 0.6743。COSIE 次之，在 RegionLoupe/Y.l1 为 0.5568、annotations 为 0.4988、celltype 为 0.4934；SpaMosaic 再次，MOFA+ 相对较低。需要注意，spa_mo_model 的这些指标是本报告从 final embedding 重新计算得到，celltype 等含缺失值标签会跳过缺失样本，因此与把缺失值当作一个类别的粗糙计算不可直接混用。

### MouseBrain NMI 对比
| 标签          | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----------- | ------------ | ------ | ------ | --------- |
| RegionLoupe | 0.6782       | 0.3933 | 0.6279 | 0.4705    |
| annotations | 0.6312       | 0.3619 | 0.5879 | 0.5662    |
| celltype    | 0.6223       | 0.3607 | 0.5839 | 0.5592    |
| Y.l1        | 0.6782       | 0.3933 | 0.6279 | 0.4705    |
| Y           | 0.4918       | 0.2916 | 0.4605 | 0.4085    |
| group       | 0.0036       | 0.0052 | 0.0518 | 0.0035    |

NMI 趋势与 ARI 基本一致：spa_mo_model 在主要 MouseBrain 标签上最高，COSIE 次之，SpaMosaic 在 annotations/celltype 上接近 COSIE，MOFA+ 相对较低。NMI 对标签层级更宽容，因此它和 ARI 一起说明 spa_mo_model 当前结果不仅具有空间连续性，也较好保留了 MouseBrain 的主要人工标签结构。

### MouseBrain Homogeneity / Completeness 对比

Homogeneity：

| 标签          | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| ------------- | -----------: | -----: | -----: | --------: |
| RegionLoupe   |       0.6416 | 0.3165 | 0.6579 |    0.3640 |
| annotations   |       0.5611 | 0.2770 | 0.5746 |    0.5874 |
| celltype      |       0.5491 | 0.2752 | 0.5648 |    0.5751 |
| Y.l1          |       0.6416 | 0.3165 | 0.6579 |    0.3640 |
| Y             |       0.4221 | 0.2172 | 0.4328 |    0.4066 |
| group         |       0.0042 | 0.0065 | 0.0812 |    0.0050 |

Completeness：

| 标签          | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| ------------- | -----------: | -----: | -----: | --------: |
| RegionLoupe   |       0.7191 | 0.5194 | 0.6005 |    0.6651 |
| annotations   |       0.7215 | 0.5218 | 0.6018 |    0.5465 |
| celltype      |       0.7181 | 0.5230 | 0.6043 |    0.5442 |
| Y.l1          |       0.7191 | 0.5194 | 0.6005 |    0.6651 |
| Y             |       0.5891 | 0.4439 | 0.4920 |    0.4105 |
| group         |       0.0031 | 0.0043 | 0.0380 |    0.0026 |

### MouseBrain ASW 对比
Cluster ASW scaled：
| 标签          | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----------- | ------------ | ------ | ------ | --------- |
| RegionLoupe | 0.5825       | 0.5608 | 0.5747 | 0.7053    |
| annotations | 0.5825       | 0.5608 | 0.5747 | 0.6347    |
| celltype    | 0.5825       | 0.5608 | 0.5747 | 0.6347    |
| Y.l1        | 0.5825       | 0.5608 | 0.5747 | 0.7053    |
| Y           | 0.5825       | 0.5608 | 0.5747 | 0.6347    |
| group       | 0.5825       | 0.5642 | 0.5637 | 0.6347    |

Label ASW scaled：
| 标签          | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----------- | ------------ | ------ | ------ | --------- |
| RegionLoupe | 0.5541       | 0.4966 | 0.5446 | 0.5211    |
| annotations | 0.5460       | 0.4902 | 0.5404 | 0.5172    |
| celltype    | 0.5454       | 0.4930 | 0.5436 | 0.5190    |
| Y.l1        | 0.5541       | 0.4966 | 0.5446 | 0.5211    |
| Y           | 0.5021       | 0.4749 | 0.5028 | 0.4661    |
| group       | 0.5008       | 0.4955 | 0.5039 | 0.4946    |

SpaMosaic 的 Cluster ASW 最高，说明其预测簇几何上最紧致；spa_mo_model 的 Cluster ASW 处在第二梯队，并且在 RegionLoupe、annotations、celltype、Y.l1 的 Label ASW 上略高于 COSIE，说明真实标签在其 embedding 中也有较好的分离。Y 标签粒度更细时，各方法 Label ASW 都接近 0.50，提示细粒度标签仍较难被清晰分开。

### MouseBrain 空间连续性
| 模式    | k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----- | -- | ------------ | ------ | ------ | --------- |
| 分样本聚类 | 5  | 0.8608       | 0.7314 | 0.8581 | 0.8495    |
| 分样本聚类 | 6  | 0.8326       | 0.6540 | 0.8523 | 0.8313    |
| 分样本聚类 | 8  | 0.7911       | 0.5095 | 0.8256 | 0.8040    |
| 分样本聚类 | 10 | 0.7357       | 0.4079 | 0.8017 | 0.7978    |
| 联合聚类  | 5  | 0.8654       | 0.7351 | 0.8493 | 0.8535    |
| 联合聚类  | 6  | 0.8217       | 0.7075 | 0.8462 | 0.8295    |
| 联合聚类  | 8  | 0.7737       | 0.4569 | 0.7761 | 0.7929    |
| 联合聚类  | 10 | 0.7159       | 0.4209 | 0.7397 | 0.7843    |

空间连续性方面，spa_mo_model 在联合聚类 k=5 最高，为 0.8654；COSIE 和 SpaMosaic 在较大 k 上更稳定。结合外部标签指标看，spa_mo_model 在 MouseBrain 上同时表现出较强的局部空间一致性和较高的主要标签一致性，而 COSIE/SpaMosaic 的优势更多体现在较大 k 的空间平滑稳定性或更紧致的簇几何。

### MouseBrain 无监督内部指标
| 方法           | reported k                | best ASW scaled | best ASW k | best DBI | best DBI k | best CH   | best CH k | max group ARI | max group ARI k |
| -------------- | ------------------------- | --------------: | ---------: | --------: | ---------: | --------: | --------: | ------------: | --------------: |
| spa_mo_model   | 2,3,4,5,6,7,8,9,10,11,12 |          0.5971 |          4 |    1.9934 |          4 | 1339.6170 |         2 |        0.0036 |               5 |
| MOFA+          | 2,3,4,5,6,7,8,9,10,11,12 |          0.6391 |          2 |    1.6710 |         10 |  774.3745 |         4 |        0.0033 |               6 |
| COSIE          | 2,3,4,5,6,7,8,9,10,11,12 |          0.5768 |          3 |    2.0871 |          7 | 1211.1279 |         3 |        0.0298 |              11 |
| SpaMosaic      | 2,3,4,5,6,7,8,9,10,11,12 |          0.8135 |          2 |    0.6676 |          2 | 7695.7079 |         3 |        0.0028 |              11 |

四种方法现在均已按 k=2-12 和同一指标口径补齐。SpaMosaic 的 ASW/DBI/CH 最强；MOFA+ 的最佳 ASW 和 DBI 优于 spa_mo_model/COSIE，但 CH 较低。group ARI 整体接近零，只有 COSIE 相对更高。

### MOFA+ MouseBrain 视图解释度 R2
| View | Group    | R2      |
| ---- | -------- | ------- |
| RNA  | SectionA | 5.8704  |
| RNA  | SectionB | 7.3094  |
| RNA  | SectionC | 6.5466  |
| UNI  | SectionA | 27.9376 |
| UNI  | SectionB | 35.0069 |
| UNI  | SectionC | 31.5513 |
| meta | SectionA | 4.9706  |
| meta | SectionB | 4.9299  |
| meta | SectionC | 3.9848  |

MOFA+ 的 R2 是方法特有指标，不能直接与 ARI/ASW 比较。该结果中 UNI/HE 特征的解释度明显高于 RNA 和 meta，说明 MOFA+ 因子主要解释图像/UNI 视图变化。

### SpaMosaic MouseBrain 模态对齐
| pair        | section  | n_spots | mean_cosine | median_cosine | std_cosine |
| ----------- | -------- | ------- | ----------- | ------------- | ---------- |
| meta_vs_rna | ALL      | 7866    | 0.9659      | 0.9810        | 0.0623     |
| meta_vs_rna | SectionA | 2384    | 0.9633      | 0.9798        | 0.0652     |
| meta_vs_rna | SectionB | 2820    | 0.9698      | 0.9823        | 0.0526     |
| meta_vs_rna | SectionC | 2662    | 0.9643      | 0.9806        | 0.0687     |
| meta_vs_uni | ALL      | 7866    | 0.9674      | 0.9810        | 0.0575     |
| meta_vs_uni | SectionA | 2384    | 0.9682      | 0.9807        | 0.0508     |
| meta_vs_uni | SectionB | 2820    | 0.9710      | 0.9822        | 0.0475     |
| meta_vs_uni | SectionC | 2662    | 0.9629      | 0.9798        | 0.0708     |
| rna_vs_uni  | ALL      | 7866    | 0.9779      | 0.9861        | 0.0315     |
| rna_vs_uni  | SectionA | 2384    | 0.9767      | 0.9845        | 0.0278     |
| rna_vs_uni  | SectionB | 2820    | 0.9793      | 0.9868        | 0.0286     |
| rna_vs_uni  | SectionC | 2662    | 0.9776      | 0.9867        | 0.0371     |

SpaMosaic 的模态间 cosine 相似度很高，说明三模态 latent 对齐紧密；但高对齐不必然等价于更高 ARI，需要结合标签和空间指标一起判断。

## CRC Stereo-CITE-seq

### 配置摘要
| 方法           | 主要设置                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| spa_mo_model | RNA/ADT 两模态；fullspot；200 epoch；bidirectional OT + attention；lambda_crossview=0.1；seed42；报告 k=8/10/15/20/25。 |
| MOFA+        | RNA/ADT 两视图；10 factors；iter1000；CPU float64；RNA HVG2000。                                                    |
| COSIE        | RNA/ADT；metacell 6x6 sparse；embedding 256 维；600 epoch；Harmony=True。                                         |
| SpaMosaic    | RNA/ADT；merged_emb 32 维；100 epoch；loss_type=ce；RNA HVG2000；ADT CLR；Harmony GPU=True。                        |


### CRC 共同 k=8/10 的无监督指标
四个方法都可比较 k=8 和 k=10。ASW scaled 均为 `(原始 silhouette + 1) / 2`；DBI 越低越好，其余通常越高越好。
| 方法           | k  | ASW scaled | ASW raw | CH         | DBI    | group ARI | group NMI |
| ------------ | -- | ---------- | ------- | ---------- | ------ | --------- | --------- |
| spa_mo_model | 8  | 0.5204     | 0.0408  | 15851.7216 | 3.9999 | 0.0283    | 0.0561    |
| MOFA+        | 8  | 0.5439     | 0.0878  | 40585.6299 | 2.0228 | 0.0007    | 0.0166    |
| COSIE        | 8  | 0.5599     | 0.1199  | 52809.6960 | 2.4652 | 0.1317    | 0.2232    |
| SpaMosaic    | 8  | 0.5512     | 0.1023  | 56606.7603 | 2.3601 | 0.0275    | 0.0652    |
| spa_mo_model | 10 | 0.5184     | 0.0367  | 13485.8551 | 4.1352 | 0.0118    | 0.0555    |
| MOFA+        | 10 | 0.5403     | 0.0805  | 37023.8431 | 2.0193 | 0.0128    | 0.0341    |
| COSIE        | 10 | 0.5627     | 0.1255  | 45958.7476 | 2.2984 | 0.0835    | 0.2051    |
| SpaMosaic    | 10 | 0.5499     | 0.0999  | 49612.5344 | 2.2502 | 0.0183    | 0.0644    |

在共同 k=8/10 上，COSIE 的 ASW scaled 最高，k=8 为 0.5599、k=10 为 0.5627；SpaMosaic 次之，MOFA+ 再次。spa_mo_model 的 ASW scaled 约 0.52，低于另外三种方法，DBI 也更高，说明其 CRC fullspot embedding 的簇间 silhouette 分离和 DBI 内部结构并不占优。COSIE 的 group ARI/NMI 最高，提示其聚类更强地携带 CRC_003/CRC_006 样本来源信息。

### CRC 各方法报告 k 范围内最佳内部指标
| 方法           | reported k                     | best ASW scaled | best ASW k | best DBI | best DBI k | best CH     | best CH k | max group ARI | max group ARI k |
| ------------ | ------------------------------ | --------------- | ---------- | -------- | ---------- | ----------- | --------- | ------------- | --------------- |
| spa_mo_model | 8,10,15,20,25                  | 0.5204          | 8          | 3.8727   | 15         | 15851.7216  | 8         | 0.0283        | 8               |
| MOFA+        | 2,3,4,5,6,7,8,9,10,11,12,20,25 | 0.5453          | 2          | 1.9295   | 25         | 62758.1445  | 2         | 0.0140        | 20.0000         |
| COSIE        | 2,3,4,5,6,7,8,9,10,11,12       | 0.5738          | 6          | 2.0833   | 6          | 97026.2326  | 2         | 0.2008        | 4.0000          |
| SpaMosaic    | 2,3,4,5,6,7,8,9,10,11,12       | 0.5797          | 2          | 2.0583   | 3          | 121861.9027 | 2         | 0.0546        | 5.0000          |

这个表按每个方法自己报告的 k 范围选最优值，因此不是严格同 k 对比。SpaMosaic 的最佳 ASW 和 CH 较高；COSIE 的 ASW 也较强但 group ARI 偏高；spa_mo_model 的最佳 ASW、DBI 和 CH 在当前 CRC 指标中均不占优，说明这次 CRC fullspot 运行在无监督内部聚类质量上弱于 COSIE/SpaMosaic/MOFA+。

### CRC 空间连续性
| 模式    | k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----- | -- | ------------ | ------ | ------ | --------- |
| 分样本聚类 | 5  | 0.5944       | 0.4664 | 0.9108 | 0.6390    |
| 分样本聚类 | 6  | 0.5280       | 0.4447 | 0.9091 | 0.5578    |
| 分样本聚类 | 8  | 0.4789       | 0.4395 | 0.8621 | 0.4805    |
| 分样本聚类 | 10 | 0.3940       | 0.3732 | 0.8662 | 0.4106    |
| 分样本聚类 | 15 | 0.3252       | 0.3078 | 0.8218 | 0.3169    |
| 分样本聚类 | 20 | 0.2545       | 0.2617 | 0.7857 | 0.2676    |
| 分样本聚类 | 25 | 0.2202       | 0.2323 | 0.7706 | 0.2335    |
| 联合聚类  | 5  | 0.6415       | 0.3958 | 0.9215 | 0.6599    |
| 联合聚类  | 6  | 0.5554       | 0.3880 | 0.9202 | 0.5931    |
| 联合聚类  | 8  | 0.4923       | 0.4186 | 0.8973 | 0.5222    |
| 联合聚类  | 10 | 0.4608       | 0.3797 | 0.8958 | 0.4507    |
| 联合聚类  | 15 | 0.3527       | 0.3179 | 0.8545 | 0.3507    |
| 联合聚类  | 20 | 0.2850       | 0.2949 | 0.8441 | 0.2954    |
| 联合聚类  | 25 | 0.2464       | 0.2630 | 0.8239 | 0.2624    |

CRC 空间连续性现已补齐表中全部 method-k 组合。COSIE 明显最高，联合聚类 k=5 为 0.9215，k=20 仍为 0.8441。spa_mo_model 在 k=5/6 分别为 0.6415/0.5554，在 k=8/10 约 0.46-0.49；MOFA+ 与 SpaMosaic 整体接近，明显低于 COSIE。

### MOFA+ CRC 视图解释度 R2
| View | Group         | R2      |
| ---- | ------------- | ------- |
| ADT  | CRC_003_bin20 | 14.7640 |
| ADT  | CRC_006_bin20 | 27.6624 |
| RNA  | CRC_003_bin20 | 0.4358  |
| RNA  | CRC_006_bin20 | 0.4497  |

MOFA+ 在 CRC 上对 ADT 的解释度明显高于 RNA：CRC_003 ADT 为 14.7640，CRC_006 ADT 为 27.6624，而 RNA 两个 section 约为 0.44，说明其因子主要解释蛋白/ADT 变化。

### SpaMosaic CRC 模态对齐
| pair       | group         | n_spots | mean_cosine | median_cosine | std_cosine |
| ---------- | ------------- | ------- | ----------- | ------------- | ---------- |
| adt_vs_rna | ALL           | 612374  | 0.0115      | 0.0097        | 0.0113     |
| adt_vs_rna | CRC_003_bin20 | 166279  | 0.0096      | 0.0095        | 0.0082     |
| adt_vs_rna | CRC_006_bin20 | 446095  | 0.0122      | 0.0097        | 0.0121     |

SpaMosaic CRC 的 adt_vs_rna mean cosine 约 0.0115，远低于 MouseBrain 三模态对齐结果，提示该运行中 RNA 与 ADT latent 对齐较弱，可能与 CE loss、ADT CLR、数据稀疏性或训练收敛有关。

## MISAR-seq

### 结果目录
| 方法         | 目录                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42       |
| MOFA+        | /home/hujinlan/mofa+/analysis/misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000                            |
| COSIE        | /home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/analysis                                                |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/misar_seq_spamosaic_full                                              |

### 配置摘要
| 方法         | 主要设置                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | RNA/ATAC 两模态；fullspot；200 epoch；bidirectional；lambda_crossview=0.1；seed42；报告 k=8/10/12/14/16。                       |
| MOFA+        | RNA/ATAC 两视图；10 factors；iter1000；RNA HVG2000；ATAC peak10000；Gaussian likelihood；scale_views=True。                     |
| COSIE        | RNA/ATAC；embedding 256 维；默认 600 epoch；HVG/variable feature 3000；Harmony=True；全图训练。                                  |
| SpaMosaic    | RNA/ATAC；merged_emb 32 维；100 epoch；RNA HVG2000；ATAC 严格使用原 SpaMosaic Epigenome_preprocess，n_peak=30000，TF-IDF/LSI。 |

后续 MISAR 表格固定使用同一列顺序：`spa_mo_model`、`MOFA+`、`COSIE`、`SpaMosaic`。ASW 均报告 `(原始 silhouette + 1) / 2` 后的 scaled 值，除非表名明确写为 raw。

### MISAR 外部聚类指标
以下表格均为按 ARI 选择的 best k。`spa_mo_model` 使用 joint clustering 行作为全量跨 dataset 对比；其 Homogeneity、Completeness、V-measure 与 section Label ASW 已在原分析脚本中补算。

Best k：
| 标签                         | spa_mo_model | MOFA+ | COSIE | SpaMosaic |
| ---------------------------- | ------------ | ----- | ----- | --------- |
| Y                            | 14           | 7     | 14    | 6         |
| Combined_Clusters_annotation | 12           | 7     | 6     | 6         |
| Combined_Clusters            | 12           | 7     | 6     | 6         |
| RNA_Clusters                 | 10           | 7     | 11    | 6         |
| ATAC_Clusters                | 12           | 7     | 6     | 6         |
| section                      | 8            | 14    | 10    | 3         |

ARI：
| 标签                         | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ---------------------------- | ------------ | ------ | ------ | --------- |
| Y                            | 0.2848       | 0.1953 | 0.2979 | 0.2309    |
| Combined_Clusters_annotation | 0.2701       | 0.3215 | 0.3278 | 0.3439    |
| Combined_Clusters            | 0.2701       | 0.3215 | 0.3278 | 0.3439    |
| RNA_Clusters                 | 0.3066       | 0.2048 | 0.2633 | 0.2344    |
| ATAC_Clusters                | 0.2597       | 0.2835 | 0.3249 | 0.3132    |
| section                      | 0.2754       | 0.0768 | 0.2597 | 0.0618    |

ARI 上，SpaMosaic 在 Combined_Clusters/Combined_Clusters_annotation 最高，为 0.3439；COSIE 在 Y 和 ATAC_Clusters 上最高，分别为 0.2979 和 0.3249；spa_mo_model 在 RNA_Clusters 上最高，为 0.3066。MOFA+ 在 Combined_Clusters 上接近 COSIE/SpaMosaic，但在 Y 和 RNA_Clusters 上较低。section 的 ARI 是 batch/section 诊断，不应简单解释为越高越好。

NMI：
| 标签                         | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ---------------------------- | ------------ | ------ | ------ | --------- |
| Y                            | 0.4079       | 0.2690 | 0.4500 | 0.3204    |
| Combined_Clusters_annotation | 0.4534       | 0.3720 | 0.4345 | 0.4477    |
| Combined_Clusters            | 0.4534       | 0.3720 | 0.4345 | 0.4477    |
| RNA_Clusters                 | 0.4460       | 0.3180 | 0.4448 | 0.3601    |
| ATAC_Clusters                | 0.4406       | 0.3625 | 0.4202 | 0.4454    |
| section                      | 0.3700       | 0.1609 | 0.3613 | 0.0552    |

NMI 上，spa_mo_model 在 Combined_Clusters、Combined_Clusters_annotation 和 RNA_Clusters 上最高；COSIE 在 Y 和 section 上最高；SpaMosaic 在 ATAC_Clusters 上最高。COSIE 与 spa_mo_model 较高的 section NMI 说明聚类中保留了更多 dataset 来源结构。

Homogeneity：
| 标签                         | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| ---------------------------- | -----------: | -----: | -----: | --------: |
| Y                            |       0.4288 | 0.2335 | 0.4754 |    0.2824 |
| Combined_Clusters_annotation |       0.4724 | 0.3283 | 0.3854 |    0.4013 |
| Combined_Clusters            |       0.4724 | 0.3283 | 0.3854 |    0.4013 |
| RNA_Clusters                 |       0.4448 | 0.2778 | 0.4485 |    0.3194 |
| ATAC_Clusters                |       0.4748 | 0.3291 | 0.3833 |    0.4109 |
| section                      |       0.4567 | 0.2239 | 0.4718 |    0.0488 |

Completeness：
| 标签                         | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| ---------------------------- | -----------: | -----: | -----: | --------: |
| Y                            |       0.3888 | 0.3173 | 0.4272 |    0.3702 |
| Combined_Clusters_annotation |       0.4359 | 0.4291 | 0.4980 |    0.5062 |
| Combined_Clusters            |       0.4359 | 0.4291 | 0.4980 |    0.5062 |
| RNA_Clusters                 |       0.4473 | 0.3718 | 0.4412 |    0.4125 |
| ATAC_Clusters                |       0.4110 | 0.4036 | 0.4648 |    0.4863 |
| section                      |       0.3109 | 0.1256 | 0.2928 |    0.0636 |

V-measure：
| 标签                         | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ---------------------------- | ------------ | ------ | ------ | --------- |
| Y                            | 0.4079       | 0.2690 | 0.4500 | 0.3204    |
| Combined_Clusters_annotation | 0.4534       | 0.3720 | 0.4345 | 0.4477    |
| Combined_Clusters            | 0.4534       | 0.3720 | 0.4345 | 0.4477    |
| RNA_Clusters                 | 0.4460       | 0.3180 | 0.4448 | 0.3601    |
| ATAC_Clusters                | 0.4406       | 0.3625 | 0.4202 | 0.4454    |
| section                      | 0.3700       | 0.1609 | 0.3613 | 0.0552    |

Cluster ASW scaled：
| 标签                         | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ---------------------------- | ------------ | ------ | ------ | --------- |
| Y                            | 0.5498       | 0.5803 | 0.5832 | 0.6014    |
| Combined_Clusters_annotation | 0.5510       | 0.5803 | 0.5785 | 0.6014    |
| Combined_Clusters            | 0.5510       | 0.5803 | 0.5785 | 0.6014    |
| RNA_Clusters                 | 0.5470       | 0.5803 | 0.5831 | 0.6014    |
| ATAC_Clusters                | 0.5510       | 0.5803 | 0.5785 | 0.6014    |
| section                      | 0.5478       | 0.5839 | 0.5793 | 0.6394    |

Label ASW scaled：
| 标签                         | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ---------------------------- | ------------ | ------ | ------ | --------- |
| Y                            | 0.4955       | 0.4635 | 0.5058 | 0.4456    |
| Combined_Clusters_annotation | 0.5038       | 0.4961 | 0.5212 | 0.4620    |
| Combined_Clusters            | 0.5038       | 0.4961 | 0.5212 | 0.4620    |
| RNA_Clusters                 | 0.5140       | 0.4799 | 0.5156 | 0.4639    |
| ATAC_Clusters                | 0.5135       | 0.4880 | 0.5371 | 0.4736    |
| section                      | 0.5216       | 0.4823 | 0.5319 | 0.4824    |

Cluster ASW 方面，SpaMosaic 在主要标签对应 best-k 行上最高，说明其 embedding 形成的簇几何分离更强；MOFA+ 与 COSIE 居中，spa_mo_model 的 Cluster ASW 相对偏低但较稳定。Label ASW 方面，COSIE 在多数标签上最高；spa_mo_model 对 RNA_Clusters 和 ATAC_Clusters 的标签空间分离也较好；SpaMosaic 虽然聚类更紧致，但主要生物标签 Label ASW 偏低，提示簇紧致和真值标签分离并不完全等价。

### MISAR 共同 k=8/10/12/14 的无监督指标
四个方法都报告 k=8、10、12、14。ASW scaled 均为 `(原始 silhouette + 1) / 2`；DBI 越低越好，其余通常越高越好。

ASW scaled：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.5478       | 0.5842 | 0.5742 | 0.5987    |
| 10 | 0.5458       | 0.5905 | 0.5793 | 0.5803    |
| 12 | 0.5510       | 0.5835 | 0.5841 | 0.5772    |
| 14 | 0.5498       | 0.5839 | 0.5832 | 0.5722    |

ASW raw：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.0956       | 0.1685 | 0.1485 | 0.1974    |
| 10 | 0.0916       | 0.1811 | 0.1586 | 0.1605    |
| 12 | 0.1020       | 0.1670 | 0.1683 | 0.1544    |
| 14 | 0.0996       | 0.1679 | 0.1665 | 0.1444    |

CH：
| k  | spa_mo_model | MOFA+    | COSIE     | SpaMosaic  |
| -- | ------------ | -------- | --------- | ---------- |
| 8  | 401.1949     | 845.5714 | 780.0098  | 1459.8714  |
| 10 | 354.5161     | 827.4764 | 718.2784  | 1257.8960  |
| 12 | 329.7294     | 787.0401 | 671.8928  | 1116.8803  |
| 14 | 299.1823     | 745.3429 | 623.6516  | 1012.8015  |

DBI：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 2.7900       | 1.5366 | 1.9367 | 1.6840    |
| 10 | 2.6928       | 1.4380 | 1.8755 | 1.7662    |
| 12 | 2.5291       | 1.4802 | 1.8614 | 1.7871    |
| 14 | 2.5550       | 1.4779 | 1.8198 | 1.8644    |

group ARI：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.2751       | 0.0428 | 0.2503 | 0.0523    |
| 10 | 0.2066       | 0.0525 | 0.2597 | 0.0487    |
| 12 | 0.1997       | 0.0587 | 0.2466 | 0.0421    |
| 14 | 0.1863       | 0.0768 | 0.2428 | 0.0382    |

group NMI：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.3693       | 0.0796 | 0.3335 | 0.0795    |
| 10 | 0.3440       | 0.1193 | 0.3613 | 0.0787    |
| 12 | 0.3375       | 0.1265 | 0.3979 | 0.0774    |
| 14 | 0.3249       | 0.1609 | 0.4134 | 0.0758    |

共同 k 上，SpaMosaic 的 ASW scaled 在 k=8 最高，为 0.5987；MOFA+ 在 k=10 达到 0.5905，并且 DBI 最低；COSIE 的 ASW 约 0.58 左右但 group ARI/NMI 明显更高；spa_mo_model 的 ASW 约 0.55，DBI 较高，说明内部簇几何分离不如 SpaMosaic/MOFA+。

### MISAR 各方法报告 k 范围内最佳内部指标
各方法 reported k 范围：
| 方法         | reported k                            |
| ------------ | ------------------------------------- |
| spa_mo_model | 8/10/12/14/16                         |
| MOFA+        | 2/3/4/5/6/7/8/9/10/11/12/13/14        |
| COSIE        | 2/3/4/5/6/7/8/9/10/11/12/13/14        |
| SpaMosaic    | 2/3/4/5/6/7/8/9/10/11/12/13/14        |

最佳内部指标：
| 指标          | spa_mo_model    | MOFA+            | COSIE            | SpaMosaic        |
| ------------- | --------------- | ---------------- | ---------------- | ---------------- |
| best ASW      | 0.5510 (k=12)   | 0.5905 (k=10)   | 0.5872 (k=13)   | 0.6670 (k=2)    |
| best DBI      | 2.5291 (k=12)   | 1.4380 (k=10)   | 1.7936 (k=13)   | 1.2854 (k=2)    |
| best CH       | 401.1949 (k=8)  | 892.5195 (k=2)  | 1541.9400 (k=2) | 3978.9139 (k=2) |
| max group ARI | 0.2751 (k=8)    | 0.0768 (k=14)   | 0.2597 (k=10)   | 0.0618 (k=3)    |

按各自报告 k 范围看，SpaMosaic 的最佳 ASW scaled 最高，为 0.6670；MOFA+ 次之，为 0.5905；COSIE 为 0.5872；spa_mo_model 为 0.5510。COSIE 与 spa_mo_model 的 max group ARI 较高，提示其部分聚类结构与 dataset 来源更相关。

### MISAR 空间连续性
分样本聚类：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.6256       | 0.5343 | 0.7474 | 0.6525    |
| 10 | 0.5555       | 0.4590 | 0.7170 | 0.6144    |
| 12 | 0.5225       | 0.4147 | 0.6980 | 0.5786    |
| 14 | 0.5032       | 0.3778 | 0.6791 | 0.5568    |
| 16 | 0.4675       | 0.3563 | 0.6724 | 0.5243    |

联合聚类：
| k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| -- | ------------ | ------ | ------ | --------- |
| 8  | 0.7174       | 0.5301 | 0.8091 | 0.6565    |
| 10 | 0.6784       | 0.5178 | 0.7883 | 0.6155    |
| 12 | 0.6713       | 0.4676 | 0.7848 | 0.6051    |
| 14 | 0.6439       | 0.4619 | 0.7719 | 0.5587    |
| 16 | 0.6253       | 0.4402 | 0.7609 | 0.5301    |

空间连续性方面，COSIE 在 joint 和 independent 模式下整体最高，joint k=8 为 0.8091，k=16 仍为 0.7609；spa_mo_model 次之，joint k=8/10 分别为 0.7174/0.6784；SpaMosaic 居中，MOFA+ 较低。结合 batch 指标看，COSIE 的空间连续性强，但也保留了较明显的 dataset/section 结构。

### MISAR Batch Correction Metrics
| 方法         | batch_key | n_used | kNN backend   | bASW   | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | ------ | ------------- | ------ | ------ | ------ | -------------- | --------- | ------------ |
| spa_mo_model | section   | 7118   | sklearn_exact | 0.9567 | 0.2899 | 0.0334 | 0.9666         | 0.8625    | 0.1375       |
| MOFA+        | group     | 7118   | sklearn_exact | 0.9644 | 0.2888 | 0.0100 | 0.9900         | 1.0000    | 0.0000       |
| COSIE        | group     | 7118   | sklearn_exact | 0.9358 | 0.1009 | 0.0001 | 0.9999         | 0.8185    | 0.1815       |
| SpaMosaic    | group     | 7118   | sklearn_exact | 0.9652 | 0.5400 | 0.1031 | 0.8969         | 0.9436    | 0.0564       |

MISAR 的 batch correction 诊断显示：SpaMosaic 的 bLISI 和 kBET 最高，说明局部 batch mixing 最好；MOFA+ 的 PCR_score 几乎为 1，表示线性 PCA 主成分中 batch 可解释方差几乎为 0，但其 kBET 很低；spa_mo_model 的 bASW 接近 0.9567，但 bLISI/kBET/PCR_score 均低于 SpaMosaic；COSIE 的 bASW 较低、kBET 接近 0，且 PCR_batch_R2 最高，说明 dataset 来源结构保留较多。

### MOFA+ MISAR 视图解释度 R2
| Group    | RNA R2 | ATAC R2 |
| -------- | ------ | ------- |
| dataset1 | 5.0213 | 17.2195 |
| dataset2 | 6.6174 | 21.9969 |
| dataset3 | 5.1054 | 18.6074 |
| dataset4 | 6.6154 | 16.2670 |

MOFA+ 在 MISAR 上对 ATAC 的解释度明显高于 RNA：ATAC 各 dataset R2 约 16.27-22.00，而 RNA 约 5.02-6.62，说明其因子主要解释 ATAC peak view 的变化。

### SpaMosaic MISAR 模态对齐
| group    | n_obs | RNA-ATAC cosine mean | RNA-ATAC cosine median |
| -------- | ----- | -------------------- | ---------------------- |
| ALL      | 7118  | 0.0393               | 0.0327                 |
| dataset1 | 2129  | 0.0435               | 0.0406                 |
| dataset2 | 1949  | 0.0390               | 0.0258                 |
| dataset3 | 1777  | 0.0426               | 0.0356                 |
| dataset4 | 1263  | 0.0278               | 0.0244                 |

SpaMosaic 的 RNA/ATAC embedding cosine 在 ALL 上约为 0.0393，整体模态 cosine 对齐较弱；dataset4 最低，dataset1 最高。这个指标是 SpaMosaic 特有的模态对齐诊断，不能直接与 MOFA+/COSIE 的因子或 embedding 指标等价比较。

### MISAR 指标来源补充
| 方法         | 来源                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16 |
| MOFA+        | /home/hujinlan/mofa+/analysis/misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000/metrics 与 tables                            |
| COSIE        | /home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/analysis/metrics 与 clustering                                            |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/misar_seq_spamosaic_full/metrics 与 clustering                                          |

### MISAR 小结
总体上，MISAR-seq 的结论比 MouseBrain/CRC 更分散：SpaMosaic 在共同 k 的 ASW、bLISI/kBET 和 batch mixing 上更强；COSIE 在空间连续性和部分标签 NMI 上更高，但 batch mixing 指标较弱且 section/group 相关性较强；spa_mo_model 在 RNA_Clusters ARI 上最高，空间连续性也处于第二梯队，但无监督内部簇几何和 batch mixing 不占优；MOFA+ 作为因子模型整体稳定，但主要标签 ARI/NMI 不突出，且 R2 显示解释度集中在 ATAC view。由于 SpaMosaic 这里严格使用原方法 ATAC/LSI 预处理，COSIE 使用其通用 omics PCA/Harmony 流程，MOFA+ 使用 peak 矩阵作为 view 输入，预处理差异仍然是不能严格横向归因的重要因素。

## Human_Lymph_Node

### 结果目录

| 方法 | 目录 |
| --- | --- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis |
| MOFA+ | /home/hujinlan/mofa+/analysis/human_lymph_node_mofa_hvg2000_k10_iter1000 |
| COSIE | /home/hujinlan/cosie_runs/human_lymph_node_cosie_rna_adt_full/analysis |
| SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/human_lymph_node_spamosaic |

### 配置摘要

| 方法 | 主要设置 |
| --- | --- |
| spa_mo_model | RNA/ADT；6,843 spots；embedding 128 维；200 epoch；RNA HVG3000；bidirectional OT/attention；seed42。 |
| MOFA+ | RNA/ADT；10 factors；RNA HVG2000；最多 iter1000，第 81 次迭代收敛；CPU float64。 |
| COSIE | RNA/ADT（内部记为 Protein）；embedding 256 维；600 epoch；RNA HVG3000；Harmony=True。 |
| SpaMosaic | RNA/ADT；merged_emb 32 维；100 epoch；RNA HVG2000；ADT CLR；Harmony GPU=True。 |

两张切片共 6,843 个 spot。原始 RNA 使用唯一 `gene_ids` 对齐，删除两张切片均为零的 28 个基因后保留 18,057 个共同 RNA 特征；31 个 ADT 全部保留。该数据集没有可靠细胞类型真值，因此以下 ARI/NMI 仅指 section/group diagnostic，不作为生物学准确率。

### Human_Lymph_Node 共同 k=5/8/10/12 的无监督指标

ASW scaled 为 `(ASW raw + 1) / 2`；ASW 与 CH 越高越好，DBI 越低越好。section ARI/NMI 越接近零，说明聚类越不容易直接退化为切片标签。

ASW scaled：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5345 | 0.6296 | 0.5835 |    0.5894 |
|  8 |       0.5380 | 0.5710 | 0.5751 |    0.5826 |
| 10 |       0.5363 | 0.5853 | 0.5682 |    0.5742 |
| 12 |       0.5341 | 0.5810 | 0.5744 |    0.5728 |

ASW raw：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0691 | 0.2591 | 0.1670 |    0.1789 |
|  8 |       0.0761 | 0.1419 | 0.1503 |    0.1651 |
| 10 |       0.0727 | 0.1706 | 0.1365 |    0.1484 |
| 12 |       0.0683 | 0.1620 | 0.1488 |    0.1457 |

CH：

|  k | spa_mo_model |    MOFA+ |     COSIE | SpaMosaic |
| --: | -----------: | -------: | --------: | --------: |
|  5 |     509.1603 | 856.0877 | 1135.2142 | 1754.4991 |
|  8 |     384.3135 | 805.5118 |  950.3695 | 1332.1561 |
| 10 |     325.5783 | 784.1474 |  821.6669 | 1155.7694 |
| 12 |     293.5894 | 756.9036 |  738.3298 | 1028.0646 |

DBI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       2.8074 | 1.4859 | 1.8580 |    1.7911 |
|  8 |       3.1453 | 1.4851 | 2.0363 |    1.7389 |
| 10 |       3.0325 | 1.4958 | 2.1245 |    1.8286 |
| 12 |       3.1610 | 1.4467 | 2.0332 |    1.8489 |

section ARI：

|  k | spa_mo_model |   MOFA+ |   COSIE | SpaMosaic |
| --: | -----------: | ------: | ------: | --------: |
|  5 |       0.0001 |  0.0002 |  0.0027 |    0.0001 |
|  8 |       0.0002 |  0.0033 |  0.0037 |    0.0000 |
| 10 |       0.0000 |  0.0026 |  0.0032 |   -0.0001 |
| 12 |       0.0004 |  0.0012 |  0.0030 |    0.0001 |

section NMI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0005 | 0.0010 | 0.0024 |    0.0006 |
|  8 |       0.0007 | 0.0034 | 0.0049 |    0.0005 |
| 10 |       0.0005 | 0.0034 | 0.0063 |    0.0004 |
| 12 |       0.0011 | 0.0024 | 0.0054 |    0.0008 |

共同 k 上，MOFA+ 的 ASW/DBI 整体较强，SpaMosaic 的 CH 最高且 section diagnostic 最接近零；COSIE 居中。spa_mo_model 的 section ARI/NMI 很低，说明切片混合充分，但 ASW 较低、DBI 较高，内部簇几何分离较弱。

### Human_Lymph_Node 各方法报告 k 范围内最佳内部指标

各方法 reported k 范围：

| 方法         | reported k                |
| ------------ | ------------------------- |
| spa_mo_model | 5/8/10/12                 |
| MOFA+        | 2/3/4/5/6/7/8/9/10/11/12 |
| COSIE        | 2/3/4/5/6/7/8/9/10/11/12 |
| SpaMosaic    | 2/3/4/5/6/7/8/9/10/11/12 |

最佳内部指标：

| 指标            |  spa_mo_model |          MOFA+ |           COSIE |       SpaMosaic |
| --------------- | ------------: | -------------: | --------------: | --------------: |
| best ASW scaled |  0.5380 (k=8) |   0.6436 (k=2) |    0.6024 (k=3) |    0.6327 (k=2) |
| best DBI        |  2.8074 (k=5) |   1.3157 (k=7) |    1.6967 (k=3) |    1.4778 (k=3) |
| best CH         | 509.1603 (k=5) | 920.1798 (k=2) | 1882.1533 (k=2) | 2687.4647 (k=2) |
| max section ARI | 0.0004 (k=12) |   0.0033 (k=8) |    0.0039 (k=7) |    0.0001 (k=5) |

该表按各方法自己报告的 k 范围选值，不能替代同 k 对比。section ARI 的“最大值”仅用于检查最强 section 依赖，不代表越高越好。

### Human_Lymph_Node 空间连续性

分样本聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6180 | 0.5162 | 0.7392 |    0.6994 |
|  8 |       0.5346 | 0.4301 | 0.6643 |    0.6256 |
| 10 |       0.4643 | 0.4191 | 0.6159 |    0.5830 |
| 12 |       0.4183 | 0.3448 | 0.5934 |    0.5568 |

联合聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6160 | 0.5137 | 0.7498 |    0.6987 |
|  8 |       0.5138 | 0.3915 | 0.6609 |    0.6253 |
| 10 |       0.4563 | 0.3680 | 0.5959 |    0.5818 |
| 12 |       0.4184 | 0.3412 | 0.5845 |    0.5534 |

COSIE 的空间连续性最高，SpaMosaic 次之，spa_mo_model 居中，MOFA+ 最低。结合内部指标看，COSIE/SpaMosaic 更平滑，但高连续性也可能包含过度平滑，不能单独解释为生物学准确率。

### Human_Lymph_Node Batch Correction Metrics

| 指标           | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| -------------- | -----------: | -----: | -----: | --------: |
| batch key      |      section |  group |  group |     group |
| n used         |         6843 |   6843 |   6843 |      6843 |
| bASW           |       0.9989 | 0.9989 | 0.9915 |    0.9995 |
| bLISI          |       0.9331 | 0.9204 | 0.7600 |    0.9493 |
| kBET           |       0.8241 | 0.7558 | 0.3392 |    0.8303 |
| kBET rejection |       0.1759 | 0.2442 | 0.6608 |    0.1697 |
| PCR score      |       0.9989 | 1.0000 | 0.9933 |    0.9994 |
| PCR batch R2   |       0.0011 | 0.0000 | 0.0067 |    0.0006 |

SpaMosaic 的 bASW、bLISI 和 kBET 整体最高；spa_mo_model 非常接近，MOFA+ 的 PCR_score 最好但局部混合略弱；COSIE 的 bLISI/kBET 最低，说明其空间平滑较强但 section mixing 较弱。

### MOFA+ Human_Lymph_Node 视图解释度 R2

| View | Group | R2 |
| --- | --- | ---: |
| ADT | Human_Lymph_Node_A1 | 75.8021 |
| ADT | Human_Lymph_Node_D1 | 76.3448 |
| RNA | Human_Lymph_Node_A1 | 2.6175 |
| RNA | Human_Lymph_Node_D1 | 1.9539 |

MOFA+ 的解释度主要集中于 ADT，RNA 的总解释度较低；该 R2 是方法特有指标，不能直接与聚类或 batch 指标等价比较。

### SpaMosaic Human_Lymph_Node 模态对齐

| group | n_spots | ADT-RNA cosine mean | median | std |
| --- | ---: | ---: | ---: | ---: |
| ALL | 6843 | 0.0903 | 0.0904 | 0.0160 |
| Human_Lymph_Node_A1 | 3484 | 0.0923 | 0.0922 | 0.0155 |
| Human_Lymph_Node_D1 | 3359 | 0.0883 | 0.0886 | 0.0163 |

### Human_Lymph_Node 小结

综合看，SpaMosaic 的 batch mixing 和 CH 最强且 section diagnostic 最低；MOFA+ 的簇几何指标较好；COSIE 的空间连续性最好但 batch mixing 较弱；spa_mo_model 的 batch correction 接近 SpaMosaic，但内部簇几何分离和空间连续性不占优。由于没有细胞类型真值，不能据此判断哪种方法具有最高生物学注释准确率。

## Mouse_Spleen

### 结果目录

| 方法 | 目录 |
| --- | --- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis |
| MOFA+ | /home/hujinlan/mofa+/analysis/mouse_spleen_mofa_hvg2000_k10_iter1000 |
| COSIE | /home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/analysis |
| SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/mouse_spleen_spamosaic |

### 配置摘要

| 方法 | 主要设置 |
| --- | --- |
| spa_mo_model | RNA/ADT；5,336 spots；embedding 128 维；200 epoch；RNA HVG3000；bidirectional OT/attention；seed42。 |
| MOFA+ | RNA/ADT；10 factors；RNA HVG2000；最多 iter1000，第 71 次迭代收敛；CPU float64。 |
| COSIE | RNA/ADT（内部记为 Protein）；embedding 256 维；600 epoch；RNA HVG3000；Harmony=True。 |
| SpaMosaic | RNA/ADT；merged_emb 32 维；100 epoch；RNA HVG2000；ADT CLR；Harmony GPU=True。 |

两张切片共 5,336 个 spot。RNA 使用唯一 `gene_ids` 对齐，统一删除 13,423 个全局零基因后保留 18,862 个特征；21 个 ADT marker 全部保留。两张切片存在大量同名 barcode，合并时均使用 section 前缀。该数据集没有细胞类型真值。

### Mouse_Spleen 共同 k=5/8/10/12 的无监督指标

ASW scaled：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5303 | 0.5595 | 0.5772 |    0.5920 |
|  8 |       0.5249 | 0.5536 | 0.5613 |    0.5756 |
| 10 |       0.5255 | 0.5483 | 0.5518 |    0.5646 |
| 12 |       0.5224 | 0.5499 | 0.5470 |    0.5607 |

ASW raw：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0606 | 0.1190 | 0.1544 |    0.1840 |
|  8 |       0.0498 | 0.1071 | 0.1226 |    0.1512 |
| 10 |       0.0510 | 0.0966 | 0.1036 |    0.1293 |
| 12 |       0.0448 | 0.0998 | 0.0941 |    0.1213 |

CH：

|  k | spa_mo_model |    MOFA+ |    COSIE | SpaMosaic |
| --: | -----------: | -------: | -------: | --------: |
|  5 |     281.1706 | 587.1120 | 971.2868 | 1366.6504 |
|  8 |     208.3498 | 485.3264 | 680.7594 |  943.8698 |
| 10 |     172.1621 | 434.2884 | 588.9241 |  787.6923 |
| 12 |     148.3511 | 401.9924 | 494.7700 |  678.6931 |

DBI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       3.9512 | 2.0578 | 2.0914 |    1.9788 |
|  8 |       3.8118 | 1.8618 | 2.3997 |    2.1789 |
| 10 |       3.8608 | 1.8343 | 2.3928 |    2.2956 |
| 12 |       4.0155 | 1.7268 | 2.5576 |    2.3456 |

section ARI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0001 | 0.0002 | 0.0032 |    0.0004 |
|  8 |       0.0001 | 0.0017 | 0.0071 |    0.0008 |
| 10 |       0.0001 | 0.0033 | 0.0023 |    0.0005 |
| 12 |       0.0000 | 0.0023 | 0.0027 |    0.0007 |

section NMI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0004 | 0.0015 | 0.0037 |    0.0011 |
|  8 |       0.0007 | 0.0041 | 0.0183 |    0.0020 |
| 10 |       0.0007 | 0.0061 | 0.0036 |    0.0016 |
| 12 |       0.0006 | 0.0051 | 0.0045 |    0.0022 |

SpaMosaic 在共同 k 的 ASW 和 CH 上整体最高；MOFA+ 在 k=8/10/12 的 DBI 最低；COSIE 的 ASW 通常位于第二梯队，但 k=8 的 section NMI 相对更高。spa_mo_model 的 section diagnostic 很低，但内部簇分离最弱。

### Mouse_Spleen 各方法报告 k 范围内最佳内部指标

各方法 reported k 范围：

| 方法         | reported k                |
| ------------ | ------------------------- |
| spa_mo_model | 5/8/10/12                 |
| MOFA+        | 2/3/4/5/6/7/8/9/10/11/12 |
| COSIE        | 2/3/4/5/6/7/8/9/10/11/12 |
| SpaMosaic    | 2/3/4/5/6/7/8/9/10/11/12 |

最佳内部指标：

| 指标            |  spa_mo_model |           MOFA+ |           COSIE |       SpaMosaic |
| --------------- | ------------: | --------------: | --------------: | --------------: |
| best ASW scaled |  0.5303 (k=5) |    0.5674 (k=2) |    0.6041 (k=2) |    0.6474 (k=2) |
| best DBI        |  3.8118 (k=8) |   1.7268 (k=12) |    1.7826 (k=2) |    1.4218 (k=2) |
| best CH         | 281.1706 (k=5) |  888.5976 (k=2) | 1597.7886 (k=2) | 2478.0284 (k=2) |
| max section ARI | 0.0001 (k=10) |   0.0033 (k=10) |    0.0071 (k=8) |    0.0016 (k=3) |

### Mouse_Spleen 空间连续性

分样本聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5078 | 0.4137 | 0.6227 |    0.5515 |
|  8 |       0.3690 | 0.3314 | 0.5319 |    0.4456 |
| 10 |       0.3089 | 0.2816 | 0.4979 |    0.3923 |
| 12 |       0.2648 | 0.2497 | 0.4543 |    0.3614 |

联合聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.4965 | 0.4122 | 0.6304 |    0.5507 |
|  8 |       0.3651 | 0.3165 | 0.5424 |    0.4422 |
| 10 |       0.3046 | 0.2776 | 0.4809 |    0.3875 |
| 12 |       0.2601 | 0.2586 | 0.4468 |    0.3596 |

COSIE 的空间连续性最高，SpaMosaic 第二，spa_mo_model 第三，MOFA+ 较低。Mouse_Spleen 的连续性随 k 增大快速下降，说明更细的簇划分会明显切碎空间区域。

### Mouse_Spleen Batch Correction Metrics

| 指标           | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| -------------- | -----------: | -----: | -----: | --------: |
| batch key      |      section |  group |  group |     group |
| n used         |         5336 |   5336 |   5336 |      5336 |
| bASW           |       0.9999 | 0.9978 | 0.9895 |    0.9992 |
| bLISI          |       0.9818 | 0.9227 | 0.8115 |    0.9638 |
| kBET           |       0.9695 | 0.7356 | 0.4130 |    0.8621 |
| kBET rejection |       0.0305 | 0.2644 | 0.5870 |    0.1379 |
| PCR score      |       0.9998 | 1.0000 | 0.9902 |    0.9990 |
| PCR batch R2   |       0.0002 | 0.0000 | 0.0098 |    0.0010 |

spa_mo_model 的 bASW、bLISI 和 kBET 最高，SpaMosaic 次之；MOFA+ 的 PCR_score 为 1，但局部混合弱于前两者；COSIE 的 batch mixing 最弱。

### MOFA+ Mouse_Spleen 视图解释度 R2

| View | Group | R2 |
| --- | --- | ---: |
| ADT | Mouse_Spleen1 | 79.0515 |
| ADT | Mouse_Spleen2 | 80.2855 |
| RNA | Mouse_Spleen1 | 1.4051 |
| RNA | Mouse_Spleen2 | 1.4625 |

MOFA+ 对 ADT 的解释度约为 79%–80%，明显高于 RNA 的约 1.4%。

### SpaMosaic Mouse_Spleen 模态对齐

| group | n_spots | ADT-RNA cosine mean | median | std |
| --- | ---: | ---: | ---: | ---: |
| ALL | 5336 | 0.0412 | 0.0414 | 0.0115 |
| Mouse_Spleen1 | 2568 | 0.0407 | 0.0411 | 0.0113 |
| Mouse_Spleen2 | 2768 | 0.0418 | 0.0418 | 0.0117 |

### Mouse_Spleen 小结

Mouse_Spleen 上，spa_mo_model 的 batch correction 最强，但内部簇几何最弱；SpaMosaic 的 ASW/CH 最强且 batch mixing 也较好；COSIE 的空间连续性最高但 batch effect 最明显；MOFA+ 的 DBI 较好且 PCR_score 最佳。由于没有生物学真值标签，这些结果应理解为不同目标之间的权衡。

## Mouse_Thymus

### 结果目录

| 方法         | 目录                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis |
| MOFA+        | /home/hujinlan/mofa+/analysis/mouse_thymus_mofa_hvg2000_k10_iter1000                                                         |
| COSIE        | /home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/analysis                                                           |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mouse_thymus_spamosaic                                                                 |

### 配置摘要

| 方法         | 主要设置                                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | RNA/ADT；17,824 spots；embedding 128 维；200 epoch；RNA HVG3000；17 ADT；bidirectional OT/attention；Harmony=True；seed42。 |
| MOFA+        | RNA/ADT；10 factors；RNA HVG2000；17 ADT；最多 iter1000，记录 261 次迭代；CPU float64。                                      |
| COSIE        | RNA/ADT（内部记为 Protein）；embedding 256 维；600 epoch；RNA HVG3000；Harmony=True；seed8。                                |
| SpaMosaic    | RNA/ADT；merged_emb 32 维；100 epoch；RNA HVG2000；17 ADT CLR；radius cutoff=150；Harmony GPU=True；seed1234。              |

四张切片分别包含 4,697、4,253、4,646 和 4,228 个 spot，共 17,824 个。RNA 按 20,293 个共同唯一 `var_names` 对齐；ADT 别名显式映射到 17 个共同 marker，并保留 Rat/Mouse IgG2a 控制项的差异。合并时使用 section 前缀解决跨切片 barcode 重名，空间坐标统一取 RNA `obs[x,y]`。聚类图使用 Mouse_Thymus1 spot size 8、Mouse_Thymus2/3/4 spot size 5。该数据集没有可靠细胞类型或组织区域真值。

### Mouse_Thymus 共同 k=5/8/10/12 的无监督指标

ASW scaled 为 `(ASW raw + 1) / 2`；ASW 与 CH 越高越好，DBI 越低越好。section ARI/NMI 是切片依赖诊断：越接近零，说明聚类越不容易直接退化为切片标签。

ASW scaled：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6386 | 0.5579 | 0.5989 |    0.5971 |
|  8 |       0.6237 | 0.5544 | 0.5460 |    0.5737 |
| 10 |       0.5976 | 0.5546 | 0.5467 |    0.5582 |
| 12 |       0.5923 | 0.5512 | 0.5429 |    0.5536 |

ASW raw：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.2773 | 0.1157 | 0.1978 |    0.1942 |
|  8 |       0.2473 | 0.1087 | 0.0921 |    0.1474 |
| 10 |       0.1951 | 0.1092 | 0.0933 |    0.1164 |
| 12 |       0.1846 | 0.1023 | 0.0858 |    0.1072 |

CH：

|  k | spa_mo_model |     MOFA+ |     COSIE | SpaMosaic |
| --: | -----------: | --------: | --------: | --------: |
|  5 |    2157.7739 | 2224.4750 | 3668.4686 | 5030.7368 |
|  8 |    1520.4362 | 1895.3361 | 2401.7643 | 3482.8020 |
| 10 |    1297.3773 | 1721.9197 | 1995.5971 | 2921.2978 |
| 12 |    1139.8644 | 1585.3288 | 1723.2028 | 2532.6325 |

DBI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       2.2222 | 1.8279 | 2.3281 |    2.0677 |
|  8 |       2.0023 | 1.7325 | 3.0265 |    2.1134 |
| 10 |       2.2446 | 1.5893 | 2.8594 |    2.2852 |
| 12 |       2.1625 | 1.6144 | 2.9187 |    2.2138 |

section ARI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.3070 | 0.0058 | 0.3591 |    0.2458 |
|  8 |       0.2434 | 0.0133 | 0.2164 |    0.2323 |
| 10 |       0.2370 | 0.0140 | 0.1749 |    0.1613 |
| 12 |       0.2351 | 0.0145 | 0.1682 |    0.1297 |

section NMI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.3826 | 0.0160 | 0.3913 |    0.3897 |
|  8 |       0.3507 | 0.0399 | 0.3368 |    0.3345 |
| 10 |       0.3239 | 0.0416 | 0.3225 |    0.3129 |
| 12 |       0.3132 | 0.0430 | 0.3034 |    0.2990 |

共同 k 上，spa_mo_model 的 ASW 最高，SpaMosaic 的 CH 最高，MOFA+ 的 DBI 最低。需要结合 section diagnostic 解读：MOFA+ 的 section ARI/NMI 远低于其他三种方法，说明其跨切片混合更充分；spa_mo_model、COSIE 和 SpaMosaic 的较高 ASW/CH 中包含明显的切片分离贡献，不能直接解释为更好的生物学聚类。

### Mouse_Thymus 各方法报告 k 范围内最佳内部指标

各方法 reported k 范围：

| 方法         | reported k                |
| ------------ | ------------------------- |
| spa_mo_model | 5/8/10/12                 |
| MOFA+        | 2/3/4/5/6/7/8/9/10/11/12 |
| COSIE        | 2/3/4/5/6/7/8/9/10/11/12 |
| SpaMosaic    | 2/3/4/5/6/7/8/9/10/11/12 |

最佳内部指标：

| 指标            |    spa_mo_model |           MOFA+ |           COSIE |       SpaMosaic |
| --------------- | --------------: | --------------: | --------------: | --------------: |
| best ASW scaled |    0.6386 (k=5) |    0.5815 (k=3) |    0.6339 (k=2) |    0.6753 (k=2) |
| best DBI        |    2.0023 (k=8) |   1.5893 (k=10) |    1.4309 (k=2) |    1.2155 (k=2) |
| best CH         | 2157.7739 (k=5) | 3264.1866 (k=2) | 6018.8316 (k=2) | 8992.1240 (k=2) |
| max section ARI |    0.3070 (k=5) |  0.0167 (k=11) |    0.3598 (k=6) |    0.3535 (k=4) |

该表按各方法自己报告的 k 范围选值，不能替代同 k 对比。SpaMosaic/COSIE 的最佳值集中在 k=2，同时 k=2 的 section ARI 分别为 0.3473/0.3500，因此这些较强的内部几何指标明显受切片结构影响。

### Mouse_Thymus 空间连续性

分样本聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5029 | 0.3807 | 0.6207 |    0.4733 |
|  8 |       0.3819 | 0.2901 | 0.5157 |    0.3481 |
| 10 |       0.3563 | 0.2564 | 0.4755 |    0.3131 |
| 12 |       0.3106 | 0.2359 | 0.4496 |    0.2663 |

联合聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6867 | 0.3829 | 0.8069 |    0.6873 |
|  8 |       0.5826 | 0.2707 | 0.6166 |    0.5104 |
| 10 |       0.4886 | 0.2479 | 0.5659 |    0.4269 |
| 12 |       0.4675 | 0.2008 | 0.5321 |    0.3884 |

COSIE 在所有共同 k 和两种聚类模式下的空间连续性最高；spa_mo_model 通常第二，SpaMosaic 接近，MOFA+ 最低。COSIE 的高连续性同时伴随较强 section effect，因此它表示更平滑的空间区域，但不等同于更准确的生物学区域。

### Mouse_Thymus Batch Correction Metrics

| 指标           | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| -------------- | -----------: | -----: | -----: | --------: |
| batch key      |      section |  group |  group |     group |
| n used         |        17824 |  17824 |  17824 |     17824 |
| bASW           |       0.9075 | 0.9697 | 0.9037 |    0.9057 |
| bLISI          |       0.4549 | 0.5980 | 0.3976 |    0.4714 |
| kBET           |       0.0025 | 0.1424 | 0.0000 |    0.0029 |
| kBET rejection |       0.9975 | 0.8576 | 1.0000 |    0.9971 |
| PCR score      |       0.7399 | 1.0000 | 0.7305 |    0.6652 |
| PCR batch R2   |       0.2601 | 0.0000 | 0.2695 |    0.3348 |

MOFA+ 的 bASW、bLISI、kBET 和 PCR score 均最高，且 PCR batch R2 接近零，是四种方法中跨切片混合最充分的。其余三种方法的 kBET 接近零，说明局部邻域仍明显受 section 构成影响；这与共同 k 表中的 section ARI/NMI 结论一致。

### MOFA+ Mouse_Thymus 视图解释度 R2

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

MOFA+ 的解释度主要集中在 ADT（约 83.1%–94.2%），RNA 解释度仅约 1.1%–1.9%。这是方法特有指标，不能直接等同于聚类质量。

### SpaMosaic Mouse_Thymus 模态对齐

| group         | n_spots | ADT-RNA cosine mean | median |    std |
| ------------- | ------: | ------------------: | -----: | -----: |
| ALL           |   17824 |              0.0217 | 0.0210 | 0.0124 |
| Mouse_Thymus1 |    4697 |              0.0113 | 0.0096 | 0.0098 |
| Mouse_Thymus2 |    4253 |              0.0251 | 0.0242 | 0.0110 |
| Mouse_Thymus3 |    4646 |              0.0254 | 0.0248 | 0.0114 |
| Mouse_Thymus4 |    4228 |              0.0255 | 0.0251 | 0.0109 |

### Mouse_Thymus 小结

Mouse_Thymus 上，spa_mo_model 在共同 k 的 ASW 最高且空间连续性较强，但 section 依赖明显；SpaMosaic 的 CH 最高，并在各自 k 范围内取得最佳 ASW/DBI/CH，但这些最佳值集中在 section 依赖较强的低 k；COSIE 的空间连续性最高，但 batch mixing 最弱之一；MOFA+ 的 DBI 与所有 batch correction 指标最好，跨切片整合最充分，不过空间连续性和 ASW 较低。由于没有生物学真值标签，不能仅凭这些内部指标确定生物学聚类的最终优胜者。

## 综合结论

1. MouseBrain：spa_mo_model 在本次补算的 ARI/NMI/Label ASW 上整体最好，并且保持较高空间连续性；COSIE 次之，SpaMosaic 的 Cluster ASW 很高，MOFA+ 相对较低。
2. CRC：COSIE 在共同 k=8/10 的 ASW scaled 和空间连续性上最好，但 section/group 相关性也更强；SpaMosaic 的最佳 ASW/CH 表现较好；spa_mo_model 在当前 fullspot 运行中无监督内部指标不占优，空间连续性处于中等；MOFA+ 稳定但整体不突出。
3. MISAR-seq：SpaMosaic 在共同 k 的 ASW、bLISI/kBET 和 batch mixing 上最强；COSIE 的空间连续性最高但 batch effect 诊断较弱；spa_mo_model 在 RNA_Clusters ARI 和空间连续性上有亮点，但内部簇几何和 batch mixing 不占优；MOFA+ 解释度主要集中在 ATAC view。
4. Human_Lymph_Node：SpaMosaic 的 batch mixing 与 CH 最强，MOFA+ 的簇几何指标较好，COSIE 的空间连续性最高，spa_mo_model 的 batch correction 很强但内部簇分离较弱。
5. Mouse_Spleen：spa_mo_model 的 batch correction 最强；SpaMosaic 的 ASW/CH 最强；COSIE 的空间连续性最高但 batch mixing 最弱；MOFA+ 的 DBI/PCR 指标较好。
6. Mouse_Thymus：spa_mo_model 的共同 k ASW 较高，SpaMosaic 的 CH 较高，COSIE 的空间连续性最高；但三者都保留明显 section 结构。MOFA+ 的 DBI 和 batch correction 最好，跨切片整合最充分。
7. 由于配置差异很大，尤其 COSIE 使用 Harmony/不同图训练设置，spa_mo_model 使用 fullspot OT/attention，SpaMosaic 使用原方法 ATAC/LSI 与 CE loss，MOFA+ 是因子模型，这些结果更适合作为 baseline 观察，不适合直接作为最终胜负判断。
8. 如果要进一步做严格比较，建议统一 k 列表、KMeans random seed/n_init、embedding 标准化、ASW sample size，并在每个数据集上明确 batch key 与主要生物标签的优先级。

## 派生文件

- 本报告：`/home/hujinlan/spa_mo_model/spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md`
- spa_mo_model MouseBrain 补算指标目录：`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics`
- spa_mo_model MouseBrain batch correction 指标：`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/batch_correction_metrics.csv`
- spa_mo_model CRC 空间连续性派生表：`/home/hujinlan/spa_mo_model/report_derived_metrics/spa_mo_model_crc_spatial_continuity_for_report.csv`
- spa_mo_model CRC batch correction 指标：`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/batch_correction_metrics.csv`
- spa_mo_model MISAR batch correction 指标：`/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/batch_correction_metrics.csv`
- spa_mo_model 公共指标补全脚本：`/home/hujinlan/spa_mo_model/scripts/complete_spa_mo_analysis.py`
- spa_mo_model MouseBrain 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics/common_clustering_metrics.csv`；`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/SUMMARY.md`
- spa_mo_model CRC 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/metrics/common_clustering_metrics.csv`；`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/SUMMARY.md`；`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/metrics/common_clustering_metrics.csv`；`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/SUMMARY.md`
- spa_mo_model MISAR-seq 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/metrics/common_clustering_metrics.csv`；`/home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/SUMMARY.md`
- spa_mo_model Human_Lymph_Node 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics`；`/home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/SUMMARY.md`
- spa_mo_model Mouse_Spleen 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics`；`/home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/SUMMARY.md`
- spa_mo_model Mouse_Thymus 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics`；`/home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/SUMMARY.md`

## 指标来源补充

| 数据集        | 方法           | 来源                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MouseBrain | spa_mo_model | /home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics/common_clustering_metrics.csv                                                                                                                                                                                                                                             |
| MouseBrain | MOFA+        | /home/hujinlan/mofa+/analysis/mousebrain_mofa_rna_meta_uni_hvg2000_k10_iter1000/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                 |
| MouseBrain | COSIE        | /home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/analysis/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                            |
| MouseBrain | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mousebrain_spamosaic/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                                      |
| CRC        | spa_mo_model | /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/metrics/common_clustering_metrics.csv; /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/metrics/common_clustering_metrics.csv |
| CRC        | MOFA+        | /home/hujinlan/mofa+/analysis/crc_stereocite_mofa_hvg2000_k10_full_iter1000_cpu_float64/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                          |
| CRC        | COSIE        | /home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/analysis/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                                         |
| CRC        | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/crc_stereocite_spamosaic_full_gpu_ce/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                                       |
| MISAR-seq | spa_mo_model | /home/hujinlan/spa_mo_model/results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis_k8_10_12_14_16/metrics/common_clustering_metrics.csv |
| MISAR-seq | MOFA+ | /home/hujinlan/mofa+/analysis/misar_mofa_rna_atac_hvg2000_peak10000_k10_iter1000/metrics |
| MISAR-seq | COSIE | /home/hujinlan/cosie_runs/misar_cosie_rna_atac_full/analysis/metrics |
| MISAR-seq | SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/misar_seq_spamosaic_full/metrics |
| Human_Lymph_Node | spa_mo_model | /home/hujinlan/spa_mo_model/results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv |
| Human_Lymph_Node | MOFA+ | /home/hujinlan/mofa+/analysis/human_lymph_node_mofa_hvg2000_k10_iter1000/metrics |
| Human_Lymph_Node | COSIE | /home/hujinlan/cosie_runs/human_lymph_node_cosie_rna_adt_full/analysis/metrics |
| Human_Lymph_Node | SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/human_lymph_node_spamosaic/metrics |
| Mouse_Spleen | spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv |
| Mouse_Spleen | MOFA+ | /home/hujinlan/mofa+/analysis/mouse_spleen_mofa_hvg2000_k10_iter1000/metrics |
| Mouse_Spleen | COSIE | /home/hujinlan/cosie_runs/mouse_spleen_cosie_rna_adt_full/analysis/metrics |
| Mouse_Spleen | SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/mouse_spleen_spamosaic/metrics |
| Mouse_Thymus | spa_mo_model | /home/hujinlan/spa_mo_model/results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics/common_clustering_metrics.csv |
| Mouse_Thymus | MOFA+ | /home/hujinlan/mofa+/analysis/mouse_thymus_mofa_hvg2000_k10_iter1000/metrics |
| Mouse_Thymus | COSIE | /home/hujinlan/cosie_runs/mouse_thymus_cosie_rna_adt_full/analysis/metrics |
| Mouse_Thymus | SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/mouse_thymus_spamosaic/metrics |

## Batch Correction Metrics

本节汇总 spa_mo_model 与本轮补算的 MOFA+、COSIE、SpaMosaic baseline。`bASW = 1 - abs(batch silhouette)`，`bLISI` 为归一化 batch LISI，`kBET = 1 - rejection_rate`，`PCR_score = 1 - PCR_batch_R2`；除 `PCR_batch_R2` 和 `kBET rejection` 越低越好外，其余越高越好。CRC 中 bLISI/kBET/PCR 均使用全量 612,374 个 spot 和 HNSW L2 approximate backend；由于标准 silhouette 全量精确计算需要全量两两距离，CRC bASW 均列出 `bASW n`。

### MouseBrain Batch Metrics
| 方法        | batch_key | n_used | kNN backend | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| --------- | --------- | ------ | ----------- | ------ | ------ | ------ | ------ | -------------- | --------- | ------------ |
| spa_mo_model | section   | 7866   | sklearn_exact | 0.9984 | 7866   | 0.7302 | 0.2400 | 0.7600         | 0.9866    | 0.0134       |
| MOFA+     | section   | 7866   | NA          | 0.9912 | 7866   | 0.8188 | 0.4327 | 0.5673         | 1.0000    | 0.0000       |
| COSIE     | section   | 7866   | NA          | 0.9938 | 7866   | 0.4925 | 0.0679 | 0.9321         | 0.9684    | 0.0316       |
| SpaMosaic | section   | 7866   | NA          | 0.9932 | 7866   | 0.8025 | 0.4237 | 0.5763         | 0.9994    | 0.0006       |

MouseBrain 中 MOFA+ 与 SpaMosaic 的 kBET 较高；spa_mo_model 的 bLISI 居中偏高，但 kBET 低于 MOFA+/SpaMosaic；COSIE 的 bLISI 和 kBET 明显较低，提示其局部邻域中的 section 混合较弱。四者 bASW 都接近 1，说明全局 batch silhouette 并不强。spa_mo_model 的 PCR_batch_R2 为 0.0134，低于 COSIE 但高于 MOFA+/SpaMosaic。

### CRC Batch Metrics
| 方法        | batch_key | n_used | kNN backend       | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| --------- | --------- | ------ | ----------------- | ------ | ------ | ------ | ------ | -------------- | --------- | ------------ |
| spa_mo_model | section   | 612374 | hnswlib_l2_approx | 0.9962 | 10000  | 0.4636 | 0.3104 | 0.6896         | 0.9901    | 0.0099       |
| MOFA+     | group     | 612374 | hnswlib_l2_approx | 0.9691 | 10000  | 0.2925 | 0.1747 | 0.8253         | 1.0000    | 0.0000       |
| COSIE     | group     | 612374 | hnswlib_l2_approx | 0.9378 | 10000  | 0.0306 | 0.0111 | 0.9889         | 0.9330    | 0.0670       |
| SpaMosaic | group     | 612374 | hnswlib_l2_approx | 0.9866 | 10000  | 0.3492 | 0.1799 | 0.8201         | 0.9764    | 0.0236       |

CRC 中 spa_mo_model 现在按全量 612,374 spot 和 HNSW approximate 近邻重算；其 bLISI/kBET 高于三个 baseline，PCR_batch_R2 也较低，说明当前 embedding 的 section 混合诊断较好。COSIE 的 bLISI/kBET 最低且 PCR_batch_R2 最高，说明其 embedding 中保留的 group/batch 结构最明显。CRC 的 bASW 仍和 baseline 一样只在 10,000 个 spot 上计算 silhouette sample，因为全量精确 silhouette 需要全量两两距离。

### MISAR-seq Batch Metrics

| 方法         | batch_key | n_used | kNN backend   | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   |   7118 | sklearn_exact | 0.9567 |   7118 | 0.2899 | 0.0334 |          0.9666 |    0.8625 |       0.1375 |
| MOFA+        | group     |   7118 | sklearn_exact | 0.9644 |   7118 | 0.2888 | 0.0100 |          0.9900 |    1.0000 |       0.0000 |
| COSIE        | group     |   7118 | sklearn_exact | 0.9358 |   7118 | 0.1009 | 0.0001 |          0.9999 |    0.8185 |       0.1815 |
| SpaMosaic    | group     |   7118 | sklearn_exact | 0.9652 |   7118 | 0.5400 | 0.1031 |          0.8969 |    0.9436 |       0.0564 |

MISAR-seq 中 SpaMosaic 的 bLISI/kBET 最好，MOFA+ 的 PCR_score 最好；COSIE 的局部 batch mixing 最弱。

### Human_Lymph_Node Batch Metrics

| 方法         | batch_key | n_used | kNN backend   | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   |   6843 | sklearn_exact | 0.9989 |   6843 | 0.9331 | 0.8241 |          0.1759 |    0.9989 |       0.0011 |
| MOFA+        | group     |   6843 | sklearn_exact | 0.9989 |   6843 | 0.9204 | 0.7558 |          0.2442 |    1.0000 |       0.0000 |
| COSIE        | group     |   6843 | sklearn_exact | 0.9915 |   6843 | 0.7600 | 0.3392 |          0.6608 |    0.9933 |       0.0067 |
| SpaMosaic    | group     |   6843 | sklearn_exact | 0.9995 |   6843 | 0.9493 | 0.8303 |          0.1697 |    0.9994 |       0.0006 |

Human_Lymph_Node 中 SpaMosaic 的 bASW、bLISI、kBET 最好，spa_mo_model 非常接近；MOFA+ 的 PCR_score 最好。

### Mouse_Spleen Batch Metrics

| 方法         | batch_key | n_used | kNN backend   | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   |   5336 | sklearn_exact | 0.9999 |   5336 | 0.9818 | 0.9695 |          0.0305 |    0.9998 |       0.0002 |
| MOFA+        | group     |   5336 | sklearn_exact | 0.9978 |   5336 | 0.9227 | 0.7356 |          0.2644 |    1.0000 |       0.0000 |
| COSIE        | group     |   5336 | sklearn_exact | 0.9895 |   5336 | 0.8115 | 0.4130 |          0.5870 |    0.9902 |       0.0098 |
| SpaMosaic    | group     |   5336 | sklearn_exact | 0.9992 |   5336 | 0.9638 | 0.8621 |          0.1379 |    0.9990 |       0.0010 |

Mouse_Spleen 中 spa_mo_model 的 bASW、bLISI、kBET 最好，SpaMosaic 第二；MOFA+ 的 PCR_score 最好。

### Mouse_Thymus Batch Metrics

| 方法         | batch_key | n_used | kNN backend   | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   |  17824 | sklearn_exact | 0.9075 |  10000 | 0.4549 | 0.0025 |          0.9975 |    0.7399 |       0.2601 |
| MOFA+        | group     |  17824 | sklearn_exact | 0.9697 |  10000 | 0.5980 | 0.1424 |          0.8576 |    1.0000 |       0.0000 |
| COSIE        | group     |  17824 | sklearn_exact | 0.9037 |  10000 | 0.3976 | 0.0000 |          1.0000 |    0.7305 |       0.2695 |
| SpaMosaic    | group     |  17824 | sklearn_exact | 0.9057 |  10000 | 0.4714 | 0.0029 |          0.9971 |    0.6652 |       0.3348 |

Mouse_Thymus 中 MOFA+ 的 bASW、bLISI、kBET 和 PCR_score 均最好，跨切片混合最充分。

MouseBrain/CRC 原汇总 CSV：`/home/hujinlan/spa_mo_model/report_derived_metrics/batch_correction_metrics_mofa_cosie_spamosaic_mousebrain_crc.csv`
