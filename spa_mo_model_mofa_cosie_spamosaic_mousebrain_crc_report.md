# spa_mo_model、MOFA+、COSIE、SpaMosaic 在 MouseBrain、CRC、MISAR-seq、Human_Lymph_Node、Mouse_Spleen、Mouse_Thymus、Simulation 与 spatch 上的结果对比报告

生成时间：2026-07-01；MISAR-seq 结果更新于 2026-07-02；Mouse_Spleen、Mouse_Thymus 及 spa_mo_model 公共指标更新于 2026-07-09；Simulation 结果更新于 2026-07-23；spatch 结果更新于 2026-07-25；Human_Lymph_Node 四方法标准对齐结果更新于 2026-07-25。本文按照用户指定目录读取已有实验结果，并补算/汇总了 `spa_mo_model` 与 baseline 的聚类、空间连续性、section diagnostic 和 batch correction 指标；未修改原始数据集。

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
| Simulation          | spa_mo_model | /home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis       |
| Simulation          | MOFA+        | /home/hujinlan/mofa+/analysis/simulation_mofa_hvg1000_k10_iter1000                                                               |
| Simulation          | COSIE        | /home/hujinlan/cosie_runs/simulation_cosie_rna_adt_full/analysis                                                                 |
| Simulation          | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/simulation_spamosaic                                                                       |
| spatch              | spa_mo_model | /home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis                                          |
| spatch              | MOFA+        | 未生成分析结果；失败记录：/home/hujinlan/mofa+/runs/spatch/run_failure.json                                                     |
| spatch              | COSIE        | /home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis                                                     |
| spatch              | SpaMosaic    | 未生成分析结果；失败记录：/home/hujinlan/SpaMosaic-dev/runs/spatch_spamosaic_full/run_failure.json                              |

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

Human_Lymph_Node、Mouse_Spleen 与 Mouse_Thymus 的原始 `obs` 不包含可靠细胞类型或组织区域真值，因此只报告无监督内部指标、空间连续性、section diagnostic 和 batch correction；不把 section ARI/NMI 误写成生物学聚类准确率。Simulation 则有五类 `spatial_domain` 真值（background、sp1–sp4），因此额外报告外部聚类指标、模拟因子恢复和同网格跨切片检索。spatch 只对成功完成训练与分析的 spa_mo_model 和 COSIE 做数值对比；MOFA+ 与 SpaMosaic 仅记录资源不足导致的失败，不把缺失值纳入排名。

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

本轮已将四种方法的下游评估统一为：各自最终 embedding 经 `StandardScaler` 后输入 `sklearn.cluster.KMeans`；`random_state=0`、`n_init=20`、`max_iter=300`；joint 内部指标统一计算 `k=2–12`；空间图与 independent clustering 统一使用 `k=5/8/10/12`；ASW、CH、DBI 和 batch 指标均使用全量 6,843 个 spot；空间近邻数为 6。每个 joint k 只生成一份 labels，内部指标和空间图直接复用；independent labels 同时用于空间连续性与空间图。四种方法仍分别读取自己目录内的数据、embedding 和结果。

### Human_Lymph_Node 共同 k=5/8/10/12 的无监督指标

ASW scaled 为 `(ASW raw + 1) / 2`；ASW 与 CH 越高越好，DBI 越低越好。section ARI/NMI 越接近零，说明聚类越不容易直接退化为切片标签。

ASW scaled：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5489 | 0.6359 | 0.5924 |    0.5894 |
|  8 |       0.5410 | 0.5850 | 0.5800 |    0.5830 |
| 10 |       0.5405 | 0.5770 | 0.5808 |    0.5741 |
| 12 |       0.5400 | 0.5813 | 0.5753 |    0.5754 |

ASW raw：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0978 | 0.2718 | 0.1847 |    0.1789 |
|  8 |       0.0820 | 0.1699 | 0.1601 |    0.1659 |
| 10 |       0.0810 | 0.1540 | 0.1616 |    0.1481 |
| 12 |       0.0801 | 0.1626 | 0.1505 |    0.1508 |

CH：

|  k | spa_mo_model |    MOFA+ |     COSIE | SpaMosaic |
| --: | -----------: | -------: | --------: | --------: |
|  5 |     527.4998 | 846.8471 | 1263.8206 | 1754.4991 |
|  8 |     401.9315 | 838.1938 |  972.4262 | 1332.2144 |
| 10 |     349.2461 | 793.0058 |  851.7784 | 1155.8133 |
| 12 |     308.1397 | 756.8574 |  760.3958 | 1028.0960 |

DBI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       2.5429 | 1.4251 | 1.7425 |    1.7911 |
|  8 |       2.7482 | 1.4357 | 1.9355 |    1.7371 |
| 10 |       2.9238 | 1.3814 | 1.9147 |    1.8195 |
| 12 |       2.8134 | 1.4460 | 2.0167 |    1.8047 |

section ARI：

|  k | spa_mo_model |   MOFA+ |   COSIE | SpaMosaic |
| --: | -----------: | ------: | ------: | --------: |
|  5 |      -0.0000 |  0.0002 |  0.0026 |    0.0001 |
|  8 |      -0.0000 |  0.0002 |  0.0026 |   -0.0001 |
| 10 |      -0.0000 |  0.0019 |  0.0046 |   -0.0001 |
| 12 |      -0.0000 |  0.0014 |  0.0036 |    0.0001 |

section NMI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.0007 | 0.0011 | 0.0025 |    0.0006 |
|  8 |       0.0005 | 0.0017 | 0.0033 |    0.0005 |
| 10 |       0.0005 | 0.0029 | 0.0055 |    0.0005 |
| 12 |       0.0005 | 0.0026 | 0.0055 |    0.0008 |

共同 k 上，MOFA+ 的 ASW/DBI 整体较强，SpaMosaic 的 CH 最高且 section diagnostic 接近零；COSIE 的 CH 和空间连续性较高，但 section diagnostic 也略高。spa_mo_model 的 section ARI/NMI 最接近零，但内部簇几何分离仍弱于三个 baseline。

### Human_Lymph_Node 各方法报告 k 范围内最佳内部指标

各方法 reported k 范围：

| 方法         | reported k                |
| ------------ | ------------------------- |
| spa_mo_model | 2/3/4/5/6/7/8/9/10/11/12 |
| MOFA+        | 2/3/4/5/6/7/8/9/10/11/12 |
| COSIE        | 2/3/4/5/6/7/8/9/10/11/12 |
| SpaMosaic    | 2/3/4/5/6/7/8/9/10/11/12 |

最佳内部指标：

| 指标            |  spa_mo_model |          MOFA+ |           COSIE |       SpaMosaic |
| --------------- | ------------: | -------------: | --------------: | --------------: |
| best ASW scaled |  0.5566 (k=3) |   0.6436 (k=2) |    0.6025 (k=3) |    0.6327 (k=2) |
| best DBI        |  2.5407 (k=3) |  1.3814 (k=10) |    1.6905 (k=3) |    1.4773 (k=3) |
| best CH         | 772.3736 (k=2) | 920.1798 (k=2) | 1882.5075 (k=2) | 2687.4647 (k=2) |
| max section ARI |  0.0002 (k=3) |  0.0022 (k=11) |    0.0047 (k=9) |    0.0001 (k=5) |

该表现在对四种方法使用完全相同的 `k=2–12` 搜索范围。section ARI 的“最大值”仅用于检查最强 section 依赖，不代表越高越好。

### Human_Lymph_Node 空间连续性

分样本聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6180 | 0.5131 | 0.7681 |    0.6991 |
|  8 |       0.5346 | 0.4263 | 0.6577 |    0.6262 |
| 10 |       0.4643 | 0.3735 | 0.6263 |    0.5825 |
| 12 |       0.4183 | 0.3440 | 0.6001 |    0.5554 |

联合聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6160 | 0.5180 | 0.7647 |    0.6987 |
|  8 |       0.5138 | 0.4383 | 0.6580 |    0.6262 |
| 10 |       0.4563 | 0.3824 | 0.6130 |    0.5815 |
| 12 |       0.4184 | 0.3458 | 0.5760 |    0.5607 |

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

统一标准后，SpaMosaic 的 batch mixing 和 CH 最强；MOFA+ 的 ASW/DBI 较好；COSIE 的空间连续性最高但 batch mixing 较弱；spa_mo_model 的 section mixing 很强，但内部簇几何分离和空间连续性不占优。由于没有细胞类型真值，不能据此判断哪种方法具有最高生物学注释准确率。

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

## Simulation

### 结果目录

| 方法         | 目录                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | /home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis |
| MOFA+        | /home/hujinlan/mofa+/analysis/simulation_mofa_hvg1000_k10_iter1000                                                         |
| COSIE        | /home/hujinlan/cosie_runs/simulation_cosie_rna_adt_full/analysis                                                           |
| SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/simulation_spamosaic                                                                 |

### 配置与数据适配摘要

| 方法         | 主要设置                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| spa_mo_model | RNA/ADT；embedding 128 维；200 epoch；RNA HVG1000；100 ADT；bidirectional OT/attention；5 切片联合训练；seed42。                           |
| MOFA+        | RNA/ADT 两个 view；10 factors；RNA HVG1000；100 ADT；最多 iter1000、实际约 71 次迭代收敛；5 个 group；CPU float64；seed1。               |
| COSIE        | RNA/ADT（内部记为 Protein）；embedding 256 维；600 epoch；RNA HVG1000；Harmony；完整空间图与相邻切片同模态 linkage；seed8。              |
| SpaMosaic    | RNA/ADT；`merged_emb` 32 维；100 epoch；RNA HVG1000；ADT CLR；radius cutoff=0.35；CE loss；Harmony GPU；seed1234。                        |

Simulation1–Simulation5 每张切片各 1,296 个 spot，共 6,480 个；每张切片包含 1,000 个 RNA feature 和 100 个 ADT feature。主评价真值为 `spatial_domain`，包含 background、sp1、sp2、sp3、sp4 五类。四种方法均从各自 `data` 目录读取对应副本，表达矩阵使用 `.X`，没有错误替换为 `layers['counts']`；真值只在训练结束后的分析阶段载入。SpaMosaic 明确使用单数目录 `/home/hujinlan/SpaMosaic-dev/demo/data/Simulation`，没有使用另一个复数目录 `Simulations`。本数据集的聚类空间图单独使用 spot size 24，不影响其他数据集。

本节已按统一口径重算：四种方法都先在各自完整最终 embedding/factor 上按维拟合 StandardScaler，再将 standardized embedding 输入 KMeans；统一使用 k=5/8/10/12、seed42、n_init20、max_iter300。ASW、CH、DBI 和 Label ASW 也在同一 standardized embedding 空间中计算。空间图、空间连续性和外部指标全部复用同一套 KMeans labels，不再二次聚类。每个 section 的 1,296 个 spot 及联合 6,480 个 spot 均为全量计算。

### Simulation 共同 k 的空间域 ARI

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.4437 | 0.3643 | 0.9936 |    0.5590 |
|  8 |       0.3800 | 0.5479 | 0.7358 |    0.6788 |
| 10 |       0.3868 | 0.3738 | 0.6413 |    0.5817 |
| 12 |       0.3799 | 0.3667 | 0.5920 |    0.5758 |

### Simulation 共同 k 的空间域 NMI

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.5559 | 0.5055 | 0.9897 |    0.6908 |
|  8 |       0.5505 | 0.7009 | 0.8680 |    0.7952 |
| 10 |       0.5207 | 0.5746 | 0.8214 |    0.7634 |
| 12 |       0.5225 | 0.5759 | 0.7918 |    0.7674 |

### Simulation 共同 k 中最佳空间域外部指标

下表按每种方法在共同 k=5/8/10/12 中 ARI 最高的行汇总；Homogeneity、Completeness、V-measure 和 Label ASW 均取同一行。

| 方法         | best k |    ARI |    NMI | Homogeneity | Completeness | V-measure | Label ASW raw | Label ASW scaled |
| ------------ | -----: | -----: | -----: | ----------: | -----------: | --------: | ------------: | ---------------: |
| spa_mo_model |      5 | 0.4437 | 0.5559 |      0.5590 |       0.5529 |    0.5559 |        0.3466 |           0.6733 |
| MOFA+        |      8 | 0.5479 | 0.7009 |      0.8050 |       0.6207 |    0.7009 |        0.1456 |           0.5728 |
| COSIE        |      5 | 0.9936 | 0.9897 |      0.9902 |       0.9893 |    0.9897 |        0.4364 |           0.7182 |
| SpaMosaic    |      8 | 0.6788 | 0.7952 |      0.9102 |       0.7060 |    0.7952 |        0.3458 |           0.6729 |

在共同 k 中，COSIE 的最佳 ARI/NMI 显著最高，SpaMosaic 第二，MOFA+ 第三，spa_mo_model 第四。SpaMosaic 与 MOFA+ 的最佳 ARI 位于 k=8，spa_mo_model 与 COSIE 的最佳 ARI 位于与真值类别数一致的 k=5。

### COSIE 高 ARI/NMI 复核

COSIE 的 k=5 结果经过独立复核后可重复，并未发现样本顺序错位、指标实现错误或训练阶段直接使用 `spatial_domain` 标签：

| 复核项                                            |    ARI |    NMI | 结论                                                  |
| ------------------------------------------------- | -----: | -----: | ----------------------------------------------------- |
| standardized 最终 256 维 embedding，seed42/n_init20/k=5 | 0.9936 | 0.9897 | 与正式分析 CSV 及空间图 labels 完全一致          |
| RNA 侧前 128 维分别标准化                         | 0.9961 | 0.9937 | RNA 侧已几乎完全恢复空间域                            |
| ADT 侧后 128 维分别标准化                         | 0.9393 | 0.9250 | ADT 侧也包含很强的空间域结构                          |
| 保存 labels 与统一参数重新 KMeans                 | 1.0000 | 1.0000 | labels 逐点完全一致                                   |
| 20 个随机种子的完整 256 维标准化结果              | 0.9936 | 0.9897 | ARI/NMI 的最小值、最大值和均值相同                    |
| barcode 与 `spatial_domain` 真值对齐              |      — |      — | 5 个 section、6,480 个 spot 全部逐点通过              |

Hungarian 匹配后只有 15/6,480 个 spot 与真值不一致。需要强调，这五张 Simulation 切片共享完全相同的空间网格、`spfac` 和空间域布局，而 COSIE 显式使用空间图并对最终表示做邻域聚合；再加上已知类别数 k=5，组合起来对 COSIE 非常有利。因此高分是真实可复现的本数据集结果，但不能外推成其在一般真实数据上的同等性能。

### Simulation 共同 k 的无监督内部指标

ASW scaled 为 `(ASW raw + 1) / 2`；ASW 与 CH 越高越好，DBI 越低越好。

ASW scaled：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6715 | 0.5862 | 0.7184 |    0.6753 |
|  8 |       0.7185 | 0.6024 | 0.7123 |    0.7044 |
| 10 |       0.7357 | 0.6095 | 0.6745 |    0.6730 |
| 12 |       0.7450 | 0.6284 | 0.6676 |    0.6817 |

ASW raw：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.3430 | 0.1723 | 0.4367 |    0.3506 |
|  8 |       0.4371 | 0.2048 | 0.4246 |    0.4088 |
| 10 |       0.4715 | 0.2190 | 0.3490 |    0.3459 |
| 12 |       0.4900 | 0.2567 | 0.3351 |    0.3634 |

CH：

|  k | spa_mo_model |    MOFA+ |     COSIE | SpaMosaic |
| --: | -----------: | -------: | --------: | --------: |
|  5 |    1547.1149 | 728.5088 | 3239.6548 | 2823.3047 |
|  8 |    2210.6932 | 720.8682 | 2858.1787 | 2657.2270 |
| 10 |    2191.4295 | 729.8282 | 2498.3677 | 2409.4406 |
| 12 |    2185.1643 | 751.1968 | 2321.5536 | 2267.6447 |

DBI：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       1.3957 | 1.9645 | 1.0999 |    1.3410 |
|  8 |       0.9394 | 1.5937 | 1.1168 |    0.9993 |
| 10 |       1.0468 | 1.4278 | 1.2734 |    1.1613 |
| 12 |       1.0255 | 1.3628 | 1.2327 |    1.1229 |

共同 k 中，spa_mo_model 在 k=8/10/12 的 ASW 与 DBI 上最好；COSIE 在 k=5 的 ASW、CH 和 DBI 最好，并在全部共同 k 的 CH 上最高；SpaMosaic 多数 k 的 ASW/DBI 位于中间，MOFA+ 的内部簇分离整体最弱。

### Simulation section 依赖诊断

表中每格为 `section ARI / section NMI`。所有值都接近零，说明四种 embedding 的联合聚类都没有退化成五张切片的身份标签。

|  k | spa_mo_model       | MOFA+              | COSIE              | SpaMosaic          |
| --: | -----------------: | -----------------: | -----------------: | -----------------: |
|  5 | -0.0006 / 0.000001 | -0.0006 / 0.000002 | -0.0006 / 0.000003 | -0.0006 / 0.000008 |
|  8 | -0.0007 / 0.000005 | -0.0007 / 0.000014 | -0.0007 / 0.000138 | -0.0007 / 0.000033 |
| 10 | -0.0008 / 0.000011 | -0.0008 / 0.000015 | -0.0007 / 0.000149 | -0.0008 / 0.000047 |
| 12 | -0.0009 / 0.000010 | -0.0009 / 0.000007 | -0.0007 / 0.000149 | -0.0008 / 0.000061 |

### Simulation 空间连续性

分切片独立聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6130 | 0.5062 | 0.9148 |    0.6735 |
|  8 |       0.4041 | 0.4229 | 0.7342 |    0.6277 |
| 10 |       0.3715 | 0.3447 | 0.6457 |    0.5171 |
| 12 |       0.3314 | 0.3043 | 0.5706 |    0.4620 |

联合聚类：

|  k | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| --: | -----------: | -----: | -----: | --------: |
|  5 |       0.6588 | 0.4914 | 0.9150 |    0.6575 |
|  8 |       0.4058 | 0.4918 | 0.7349 |    0.6313 |
| 10 |       0.3703 | 0.3463 | 0.6291 |    0.5242 |
| 12 |       0.3300 | 0.3195 | 0.5776 |    0.4893 |

COSIE 在所有共同 k 与两种聚类模式下的空间连续性均最高，SpaMosaic 通常第二。COSIE 的优势与其空间图和邻域聚合机制一致，但也需结合前述共享空间模板的模拟设计解读。

### Simulation Batch Correction Metrics

| 指标           | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic |
| -------------- | -----------: | -----: | -----: | --------: |
| batch key      |      section |  group |  group |     group |
| n used         |         6480 |   6480 |   6480 |      6480 |
| bASW           |       0.9988 | 0.9659 | 0.9979 |    0.9981 |
| bLISI          |       0.9767 | 0.9567 | 0.9798 |    0.9549 |
| kBET           |       0.9867 | 0.9381 | 0.9776 |    0.9369 |
| kBET rejection |       0.0133 | 0.0619 | 0.0224 |    0.0631 |
| PCR score      |       1.0000 | 1.0000 | 0.9996 |    1.0000 |
| PCR batch R2   |       0.0000 | 0.0000 | 0.0004 |    0.0000 |

spa_mo_model 的 bASW 与 kBET 最好，COSIE 的 bLISI 最好；四种方法的 PCR score 都接近 1。结合 section ARI/NMI 可见，五张切片在四种 embedding 中均已充分混合。

### Simulation 模拟因子恢复与 nuisance 泄漏

数值是 cross-validation 的 variance-weighted linear R2。`spfac recovery` 越高越好；RNA/ADT `nsfac leakage` 越低越好。

| 诊断项             | spa_mo_model |  MOFA+ |  COSIE | SpaMosaic | 方向       |
| ------------------ | -----------: | -----: | -----: | --------: | ---------- |
| spfac recovery     |       0.9833 | 0.8347 | 0.9784 |    0.9390 | 越高越好   |
| RNA nsfac leakage  |       0.9784 | 0.7279 | 0.8361 |    0.8354 | 越低越好   |
| ADT nsfac leakage  |       0.9592 | 0.9048 | 0.7707 |    0.8030 | 越低越好   |

spa_mo_model 与 COSIE 的空间因子恢复最高；但 spa_mo_model 同时保留了最多 RNA/ADT nuisance 信息。MOFA+ 的 RNA nuisance 泄漏最低，COSIE 的 ADT nuisance 泄漏最低。该表揭示了“恢复空间信号”和“去除模态特异噪声”之间的权衡，不能只看第一行。

### Simulation 相邻切片同网格 spot 检索

四对相邻切片取均值；Top-1 与 Recall@5 越高越好，FOSCTTM 越低越好。

| 方法         | Top-1 accuracy | Recall@5 | Mean FOSCTTM |
| ------------ | -------------: | -------: | ------------: |
| spa_mo_model |         0.1582 |   0.3895 |        0.0190 |
| MOFA+        |         0.0395 |   0.1597 |        0.0267 |
| COSIE        |         0.7610 |   0.9184 |        0.0014 |
| SpaMosaic    |         0.0897 |   0.2772 |        0.0222 |

COSIE 的同位置跨切片检索显著最好，这与其接近完美的空间域聚类及五张切片共享网格/空间模板一致；spa_mo_model 第二，SpaMosaic 第三，MOFA+ 第四。

### MOFA+ Simulation 视图解释度 R2

| View | Group       |      R2 |
| ---- | ----------- | ------: |
| ADT  | Simulation1 | 77.2002 |
| ADT  | Simulation2 | 73.9427 |
| ADT  | Simulation3 | 71.1125 |
| ADT  | Simulation4 | 68.5937 |
| ADT  | Simulation5 | 66.4350 |
| RNA  | Simulation1 | 19.8822 |
| RNA  | Simulation2 | 16.7121 |
| RNA  | Simulation3 | 14.4279 |
| RNA  | Simulation4 | 12.4533 |
| RNA  | Simulation5 | 10.8658 |

MOFA+ 的总解释度在 ADT 上约为 66.4%–77.2%，在 RNA 上约为 10.9%–19.9%，且两种 view 都从 Simulation1 到 Simulation5 递减。

### SpaMosaic Simulation 模态对齐

| group       | n_spots | ADT-RNA cosine mean | median |    std |
| ----------- | ------: | ------------------: | -----: | -----: |
| ALL         |    6480 |              0.1245 | 0.1265 | 0.0144 |
| Simulation1 |    1296 |              0.1251 | 0.1273 | 0.0140 |
| Simulation2 |    1296 |              0.1248 | 0.1265 | 0.0140 |
| Simulation3 |    1296 |              0.1246 | 0.1266 | 0.0143 |
| Simulation4 |    1296 |              0.1240 | 0.1259 | 0.0146 |
| Simulation5 |    1296 |              0.1240 | 0.1261 | 0.0148 |

### Simulation 小结

Simulation 上，COSIE 在 k=5 的空间域 ARI/NMI、空间连续性、跨切片同位置检索和 ADT nuisance 抑制方面最好；其高分已通过多项独立检查，但受到五张切片共享完全相同空间模板、显式空间图和已知 k=5 的共同促进，不能直接外推到真实数据。SpaMosaic 的最佳空间域 ARI/NMI 位居第二，并在 k=8 达峰；spa_mo_model 的空间因子恢复、batch mixing 和较高 k 的内部 ASW 很强，但 nuisance 泄漏最明显；MOFA+ 的聚类与检索较弱，不过 RNA nuisance 泄漏最低，且因子模型提供了独立的 view/group 解释度。

## spatch

### 运行状态与结果目录

| 方法         | 状态 | 结果或失败记录                                                                                                            |
| ------------ | ---- | ------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | 成功 | /home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis                                  |
| COSIE        | 成功 | /home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis                                             |
| MOFA+        | 失败 | /home/hujinlan/mofa+/runs/spatch/run_failure.json                                                                        |
| SpaMosaic    | 失败 | /home/hujinlan/SpaMosaic-dev/runs/spatch_spamosaic_full/run_failure.json                                                 |

本节的数值表格只比较成功完成训练和分析的 spa_mo_model 与 COSIE。MOFA+ 和 SpaMosaic 没有最终 embedding 或分析结果，因此不以空值参加对比，也不纳入优劣排序。

### 配置与数据适配摘要

| 方法         | 主要设置                                                                                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| spa_mo_model | RNA/Protein/HE 三模态；1,068,962 个 full spot；section1=665,399、section2=403,563；无空间块聚合；128 维 embedding；200 epoch；RNA HVG3000；Harmony；bidirectional OT/attention；BF16；seed42；RTX 4090。 |
| COSIE        | RNA/Protein/HE 三模态；完整 1,068,962 spot 输入与输出；仅本方法使用 metacell+6×6 子图方案，内部训练节点为 section1=189,649、section2=134,518；384 维 embedding；600 epoch；Harmony；seed42；RTX 4090。 |

两个成功方法都从各自项目的 `data/spatch` 读取数据，没有跨方法引用数据目录。两个 section 的 Protein 均从 17 个原始通道中剔除 DAPI，之后保留 16 个通道；预处理后 Protein 表征为 15 维。两者均未使用 4×4 空间块聚合。spa_mo_model 仅优化了延迟加载 HE、顺序预处理和避免重复 AnnData 等内存路径，`run_summary.json` 明确记录模型逻辑未改变。

本次统一重算后，两个方法都直接将最终模型输出的原始 embedding 输入 MiniBatchKMeans；KMeans 参数均为 k=2–20、seed42、n_init20、batch size4096、max_iter300。距离类评价统一在各自完整 embedding 上拟合 StandardScaler：Cluster ASW 与 Label ASW 均由 sklearn 使用 random_state=42 抽取相同位置的 10,000 个 spot；CH/DBI 均使用完整 1,068,962 个 spot；batch 指标均使用相同位置的 100,000 个 spot，其中 bASW 再抽取相同位置的 10,000 个。COSIE 也已补齐 k=20 的 joint/independent 空间结果；两个方法保留 k=5/8/10/12/16/20 的空间图均使用对应 section 的全部 spot 绘制。

### 资源不足方法记录

| 方法      | 失败阶段                         | 资源错误摘要                                                                                                            | 后续处理                                  |
| --------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| MOFA+     | 初始 ELBO 后第一次 W 更新        | 24 GB RTX 4090 上 CuPy 已分配约 15.97 GiB，继续申请约 7.96 GiB 时 OOM                                                   | 未切换 CPU；未生成模型；未开始分析        |
| SpaMosaic | GPU 训练 epoch 0/100             | full-spot 三模态训练中本进程已占约 22.54 GiB，另有约 1.01 GiB 外部占用，仅余 4.19 MiB；再申请 36 MiB 时 CUDA OOM       | 未切换 CPU、采样或空间聚合；未生成 embedding |

SpaMosaic 的 22.54 GiB 只是 OOM 时点的已用显存，不是可完成训练的最终峰值；该次运行尚未完成第一个 epoch 的反向传播。因此这里把两个方法记为“计算资源不足、无可比较结果”，而不是记为数值表现较差。

### spatch 外部聚类指标：按 ARI 选择 best k

两个方法都对联合 embedding 扫描 k=2–20，并在每个标签内按 ARI 选择 best k。`cell_type_common` 覆盖两个 section 的 956,139 个有效标签（16 类）；`spatial_cluster` 仅在 section2 有 344,594 个有效标签（8 类）；`codex_coarse_label` 仅在 section1 有 665,399 个有效标签（10 类）。Cluster/Label ASW scaled 均为 `(raw + 1) / 2`。

| 标签               | 方法         | best k |    ARI |    NMI | Homogeneity | Completeness | V-measure | Cluster ASW raw | Cluster ASW scaled | Label ASW raw | Label ASW scaled |
| ------------------ | ------------ | -----: | -----: | -----: | ----------: | -----------: | --------: | --------------: | -----------------: | ------------: | ---------------: |
| cell_type_common   | spa_mo_model |      3 | 0.5786 | 0.4020 |      0.3337 |       0.5055 |    0.4020 |          0.0968 |             0.5484 |        0.0247 |           0.5123 |
| cell_type_common   | COSIE        |      4 | 0.5433 | 0.3839 |      0.3406 |       0.4398 |    0.3839 |          0.1609 |             0.5804 |        0.0660 |           0.5330 |
| spatial_cluster    | spa_mo_model |      5 | 0.3301 | 0.4439 |      0.3928 |       0.5103 |    0.4439 |          0.0945 |             0.5472 |        0.0608 |           0.5304 |
| spatial_cluster    | COSIE        |      9 | 0.4422 | 0.5588 |      0.5493 |       0.5686 |    0.5588 |          0.1188 |             0.5594 |        0.0749 |           0.5374 |
| codex_coarse_label | spa_mo_model |      3 | 0.2555 | 0.1539 |      0.1277 |       0.1938 |    0.1539 |          0.0968 |             0.5484 |       -0.0066 |           0.4967 |
| codex_coarse_label | COSIE        |      4 | 0.2782 | 0.1840 |      0.1620 |       0.2130 |    0.1840 |          0.1609 |             0.5804 |        0.0070 |           0.5035 |

spa_mo_model 在 `cell_type_common` 的 best ARI/NMI 更高；COSIE 在 `spatial_cluster` 和 `codex_coarse_label` 的 ARI/NMI 更高，并且三个标签对应行的 Label ASW 都更高。三个标签的覆盖范围不同，不能把不同标签之间的绝对分数直接视为同一任务。

### spatch 共同 k=5/8/10/12/16/20 的无监督内部指标

两者都保留 k=2–20 的联合聚类数值，因此共同表使用 k=5/8/10/12/16/20。ASW 使用 sklearn 从相同 spot 顺序按 random_state=42 抽取的 10,000 spot；CH/DBI 均使用完整 1,068,962 spot。三个指标都在各自完整 embedding 拟合的 StandardScaler 空间中评价。

ASW scaled：

|  k | spa_mo_model |  COSIE |
| --: | -----------: | -----: |
|  5 |       0.5472 | 0.5553 |
|  8 |       0.5538 | 0.5552 |
| 10 |       0.5627 | 0.5506 |
| 12 |       0.5603 | 0.5492 |
| 16 |       0.5642 | 0.5426 |
| 20 |       0.5624 | 0.5410 |

ASW raw：

|  k | spa_mo_model |  COSIE |
| --: | -----------: | -----: |
|  5 |       0.0945 | 0.1106 |
|  8 |       0.1076 | 0.1103 |
| 10 |       0.1255 | 0.1012 |
| 12 |       0.1205 | 0.0985 |
| 16 |       0.1283 | 0.0852 |
| 20 |       0.1247 | 0.0821 |

CH：

|  k | spa_mo_model（n=1,068,962） | COSIE（n=1,068,962） |
| --: | --------------------------: | -------------------: |
|  5 |                  81370.5237 |          125180.5170 |
|  8 |                  68146.3196 |           89829.0336 |
| 10 |                  67823.9711 |           80650.9575 |
| 12 |                  61453.0599 |           69450.4001 |
| 16 |                  55131.7233 |           56475.5695 |
| 20 |                  49334.7239 |           48113.8362 |

DBI：

|  k | spa_mo_model（n=1,068,962） | COSIE（n=1,068,962） |
| --: | --------------------------: | -------------------: |
|  5 |                     2.7910 |               2.6081 |
|  8 |                     2.5013 |               2.6404 |
| 10 |                     2.1518 |               2.7315 |
| 12 |                     2.1882 |               2.7907 |
| 16 |                     2.1541 |               2.8175 |
| 20 |                     2.1283 |               2.7695 |

section ARI diagnostic：

|  k | spa_mo_model |   COSIE |
| --: | -----------: | ------: |
|  5 |       0.0009 | 0.0024 |
|  8 |       0.0033 | 0.0238 |
| 10 |       0.0147 | 0.0050 |
| 12 |       0.0084 | 0.0161 |
| 16 |       0.0157 | 0.0188 |
| 20 |       0.0125 | 0.0115 |

section NMI diagnostic：

|  k | spa_mo_model |  COSIE |
| --: | -----------: | -----: |
|  5 |       0.0014 | 0.0026 |
|  8 |       0.0065 | 0.0591 |
| 10 |       0.0360 | 0.0116 |
| 12 |       0.0154 | 0.0388 |
| 16 |       0.0438 | 0.0619 |
| 20 |       0.0443 | 0.0343 |

共同 k 上，COSIE 在 k=5/8 的 ASW 略高，spa_mo_model 在 k=10/12/16/20 更高。两者的 section ARI/NMI 整体都很低，未见聚类被 section 标签直接主导；较高 k 下仍有少量 section 关联结构。

### spatch 各方法报告 k 范围内最佳内部指标

两种方法都扫描 k=2–20。ASW scaled 仍按 `(raw + 1) / 2` 计算；CH/DBI 均为全量同口径。

| 指标            | spa_mo_model       | COSIE                |
| --------------- | ------------------ | -------------------- |
| best ASW scaled | 0.5654（k=19）     | 0.5864（k=2）        |
| best ASW raw    | 0.1307（k=19）     | 0.1727（k=2）        |
| best DBI        | 2.0864（k=19）     | 2.0585（k=2）        |
| best CH         | 135263.3034（k=2） | 238969.5396（k=2）   |
| max section ARI | 0.0157（k=16）     | 0.0238（k=8）        |
| max section NMI | 0.0443（k=20）     | 0.0724（k=19）       |

COSIE 的全范围 best ASW 出现在 k=2，而 spa_mo_model 出现在 k=19；低 k 往往更容易取得较高 silhouette，因此该“best”表必须与共同 k 表一起解释。CH/DBI 现在可以按相同样本口径比较，但仍会受到最终 embedding 维度和几何结构差异影响。

### spatch 空间连续性

空间连续性为两个 section 的空间近邻同簇比例算术平均。两个方法现在都生成并保留 k=5/8/10/12/16/20，因此六个 k 均可对比。

分 section 独立聚类：

|  k | spa_mo_model |  COSIE |
| --: | -----------: | -----: |
|  5 |       0.8626 | 0.9422 |
|  8 |       0.8192 | 0.9173 |
| 10 |       0.7997 | 0.9087 |
| 12 |       0.7530 | 0.8891 |
| 16 |       0.7072 | 0.8718 |
| 20 |       0.6540 | 0.8647 |

联合聚类：

|  k | spa_mo_model |  COSIE |
| --: | -----------: | -----: |
|  5 |       0.8477 | 0.9327 |
|  8 |       0.7257 | 0.9126 |
| 10 |       0.7515 | 0.8878 |
| 12 |       0.7325 | 0.8878 |
| 16 |       0.6983 | 0.8674 |
| 20 |       0.6562 | 0.8530 |

COSIE 在全部共同 k、joint 和 independent 两种模式下的空间连续性都更高。该结果与 COSIE 显式空间图和 metacell+6×6 训练设置有关；空间连续性高表示区域更平滑，但不能单独等价为生物学聚类更准确。

### spatch Batch Correction Metrics

| 方法         | batch_key | n_used | kNN backend   |  bASW | bASW n |  bLISI |   kBET | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   | 100000 | sklearn_exact | 0.9912 |  10000 | 0.3595 | 0.1595 |          0.8405 |    0.9930 |       0.0070 |
| COSIE        | group     | 100000 | sklearn_exact | 0.9933 |  10000 | 0.2380 | 0.0588 |          0.9412 |    0.9944 |       0.0056 |

两者现在使用相同位置的 100,000 个 spot，bASW 内部也使用相同位置的 10,000 个 spot。两者 bASW 和 PCR_score 都接近 1；spa_mo_model 的 bLISI 与 kBET 更高，说明局部邻域 section mixing 更好；COSIE 的 bASW/PCR_score 略高。

### spatch 统一后仍存在的差异

| 差异 | 影响大小 | 影响说明 |
| ---- | -------- | -------- |
| spa_mo_model 为128维 embedding，COSIE为384维 | 中到大 | 即使逐维标准化，维度仍会影响欧氏距离集中、ASW、CH、DBI、kNN batch指标和KMeans几何；这是最终模型输出差异，不是评价脚本口径错误。 |
| spa_mo_model按原方法使用full-spot训练；COSIE按用户指定使用metacell+6×6 | 大 | 会直接改变表示的局部平滑程度，尤其可能提高COSIE的空间连续性；属于实验配置差异，不能由统一分析消除。 |
| 两种模型的目标函数、网络结构和训练轮数不同（200 vs 600） | 大 | 这是要比较的方法本身，影响所有下游指标，但不属于分析不公平。 |
| 原始embedding直接用于KMeans，而不同模型输出轴的方差尺度不同 | 中到大 | 当前两边规则已一致，但KMeans不具尺度不变性；这是用户指定“使用原始embedding”的必然影响。距离评价仍统一使用全量StandardScaler空间。 |
| batch列名分别为`section`和`group` | 无 | 两列的内容、顺序和类别完全相同，都是section1/section2；只是字段名不同。 |

除以上方法固有差异外，本轮用于数值表格的spot、标签、顺序、KMeans参数、抽样位置、标准化范围、CH/DBI样本量、空间近邻参数及batch抽样均已统一，保留k的空间图也都改为全量spot。当前仍只使用单一seed42，未评价随机种子方差；这对两种方法的影响相同，但会限制结论的统计稳健性。

### spatch 小结

spatch 上只对成功的 spa_mo_model 和 COSIE 下结论。spa_mo_model 在 `cell_type_common` 的 best ARI/NMI、更高 k 的共同 ASW，以及 bLISI/kBET 上占优；COSIE 在 `spatial_cluster`、`codex_coarse_label` 的外部聚类指标和所有共同 k 的空间连续性上占优。两者的 section diagnostic 均较低，bASW/PCR 也都显示全局 section 可分性不强。MOFA+ 在第一次 GPU W 更新时因 CuPy OOM 失败，SpaMosaic 在 epoch 0 因 CUDA OOM 失败，二者没有结果可供公平对比。

## 综合结论

1. MouseBrain：spa_mo_model 在本次补算的 ARI/NMI/Label ASW 上整体最好，并且保持较高空间连续性；COSIE 次之，SpaMosaic 的 Cluster ASW 很高，MOFA+ 相对较低。
2. CRC：COSIE 在共同 k=8/10 的 ASW scaled 和空间连续性上最好，但 section/group 相关性也更强；SpaMosaic 的最佳 ASW/CH 表现较好；spa_mo_model 在当前 fullspot 运行中无监督内部指标不占优，空间连续性处于中等；MOFA+ 稳定但整体不突出。
3. MISAR-seq：SpaMosaic 在共同 k 的 ASW、bLISI/kBET 和 batch mixing 上最强；COSIE 的空间连续性最高但 batch effect 诊断较弱；spa_mo_model 在 RNA_Clusters ARI 和空间连续性上有亮点，但内部簇几何和 batch mixing 不占优；MOFA+ 解释度主要集中在 ATAC view。
4. Human_Lymph_Node（已统一下游标准）：SpaMosaic 的 batch mixing 与 CH 最强，MOFA+ 的 ASW/DBI 较好，COSIE 的空间连续性最高，spa_mo_model 的 section mixing 很强但内部簇分离较弱。
5. Mouse_Spleen：spa_mo_model 的 batch correction 最强；SpaMosaic 的 ASW/CH 最强；COSIE 的空间连续性最高但 batch mixing 最弱；MOFA+ 的 DBI/PCR 指标较好。
6. Mouse_Thymus：spa_mo_model 的共同 k ASW 较高，SpaMosaic 的 CH 较高，COSIE 的空间连续性最高；但三者都保留明显 section 结构。MOFA+ 的 DBI 和 batch correction 最好，跨切片整合最充分。
7. Simulation：COSIE 的空间域 ARI/NMI、空间连续性和跨切片同位置检索显著最好，SpaMosaic 的外部聚类指标第二；spa_mo_model 的空间因子恢复和 batch mixing 很强但 nuisance 泄漏明显；MOFA+ 的 RNA nuisance 泄漏最低。COSIE 的高分已复核无计算或标签泄漏错误，但高度受共享空间模板、空间图和 k=5 设置影响。
8. spatch：只比较成功运行的 spa_mo_model 与 COSIE。spa_mo_model 在 `cell_type_common`、较高 k 的 ASW 和局部 batch mixing 上更好；COSIE 在 `spatial_cluster`、`codex_coarse_label` 和空间连续性上更好。MOFA+ 与 SpaMosaic 均因 24 GB RTX 4090 显存不足而没有可比较结果。
9. 由于配置差异很大，尤其 COSIE 使用 Harmony/不同图训练设置，spa_mo_model 使用 fullspot OT/attention，SpaMosaic 使用原方法模态预处理与 CE loss，MOFA+ 是因子模型，这些结果更适合作为 baseline 观察，不适合直接作为最终胜负判断。
10. 如果要进一步做严格比较，建议统一 k 列表、KMeans random seed/n_init、embedding 标准化、ASW sample size，并在每个数据集上明确 batch key 与主要生物标签的优先级。

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
- spa_mo_model Simulation 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics`；`/home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/SUMMARY.md`
- MOFA+ Simulation 分析摘要：`/home/hujinlan/mofa+/analysis/simulation_mofa_hvg1000_k10_iter1000/SUMMARY.md`
- COSIE Simulation 分析摘要：`/home/hujinlan/cosie_runs/simulation_cosie_rna_adt_full/analysis/SUMMARY.md`
- SpaMosaic Simulation 分析摘要：`/home/hujinlan/SpaMosaic-dev/analysis/simulation_spamosaic/SUMMARY.md`
- spa_mo_model spatch 公共指标与摘要：`/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/metrics`；`/home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/SUMMARY.md`
- COSIE spatch 指标与摘要：`/home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis/metrics`；`/home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis/SUMMARY.md`
- MOFA+ spatch GPU OOM 记录：`/home/hujinlan/mofa+/runs/spatch/run_failure.json`
- SpaMosaic spatch GPU OOM 记录：`/home/hujinlan/SpaMosaic-dev/runs/spatch_spamosaic_full/run_failure.json`

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
| Simulation | spa_mo_model | /home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/metrics; /home/hujinlan/spa_mo_model/results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/clustering_analysis/clustering_metrics.csv |
| Simulation | MOFA+ | /home/hujinlan/mofa+/analysis/simulation_mofa_hvg1000_k10_iter1000/metrics |
| Simulation | COSIE | /home/hujinlan/cosie_runs/simulation_cosie_rna_adt_full/analysis/metrics |
| Simulation | SpaMosaic | /home/hujinlan/SpaMosaic-dev/analysis/simulation_spamosaic/metrics |
| spatch | spa_mo_model | /home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/clustering_metrics.csv; /home/hujinlan/spa_mo_model/results/spatch/fullspot_200ep_gpu_seed42/clustering_analysis/metrics |
| spatch | MOFA+ | 无指标；失败记录：/home/hujinlan/mofa+/runs/spatch/run_failure.json |
| spatch | COSIE | /home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis/metrics; /home/hujinlan/cosie_runs/spatch_cosie_rna_protein_he_metacell_6x6/analysis/clustering |
| spatch | SpaMosaic | 无指标；失败记录：/home/hujinlan/SpaMosaic-dev/runs/spatch_spamosaic_full/run_failure.json |

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

### Simulation Batch Metrics

| 方法         | batch_key | n_used | kNN backend   | bASW   | bASW n | bLISI  | kBET   | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   |   6480 | sklearn_exact | 0.9988 |   6480 | 0.9767 | 0.9867 |          0.0133 |    1.0000 |       0.0000 |
| MOFA+        | group     |   6480 | sklearn_exact | 0.9659 |   6480 | 0.9567 | 0.9381 |          0.0619 |    1.0000 |       0.0000 |
| COSIE        | group     |   6480 | sklearn_exact | 0.9979 |   6480 | 0.9798 | 0.9776 |          0.0224 |    0.9996 |       0.0004 |
| SpaMosaic    | group     |   6480 | sklearn_exact | 0.9981 |   6480 | 0.9549 | 0.9369 |          0.0631 |    1.0000 |       0.0000 |

Simulation 中 spa_mo_model 的 bASW/kBET 最好，COSIE 的 bLISI 最好；四种方法的 PCR score 均接近 1。

### spatch Batch Metrics

本表只列出成功完成分析的两个方法；MOFA+ 与 SpaMosaic 的 GPU OOM 原因已在 spatch 主章节记录，不以缺失值参加 batch correction 对比。

| 方法         | batch_key | n_used | kNN backend   |  bASW | bASW n |  bLISI |   kBET | kBET rejection | PCR_score | PCR_batch_R2 |
| ------------ | --------- | -----: | ------------- | -----: | -----: | -----: | -----: | --------------: | --------: | -----------: |
| spa_mo_model | section   | 100000 | sklearn_exact | 0.9912 |  10000 | 0.3595 | 0.1595 |          0.8405 |    0.9930 |       0.0070 |
| COSIE        | group     | 100000 | sklearn_exact | 0.9933 |  10000 | 0.2380 | 0.0588 |          0.9412 |    0.9944 |       0.0056 |

spatch 中 spa_mo_model 的 bLISI/kBET 更高，COSIE 的 bASW/PCR_score 略高；两者使用相同位置的100,000个spot，bASW也使用相同位置的10,000个spot。

MouseBrain/CRC 原汇总 CSV：`/home/hujinlan/spa_mo_model/report_derived_metrics/batch_correction_metrics_mofa_cosie_spamosaic_mousebrain_crc.csv`
