# spa_mo_model、MOFA+、COSIE、SpaMosaic 在 MouseBrain 与 CRC 上的结果对比报告

生成时间：2026-07-01。本文按照用户指定目录读取已有实验结果，并补算了 `spa_mo_model` MouseBrain 的外部聚类指标以及 `spa_mo_model` CRC 的空间连续性派生指标；未修改原始数据集。

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
| 方法           | reported k               | best ASW scaled | best ASW k | best DBI | best DBI k | best CH   | best CH k | max group ARI | max group ARI k |
| ------------ | ------------------------ | --------------- | ---------- | -------- | ---------- | --------- | --------- | ------------- | --------------- |
| COSIE        | 2,3,4,5,6,7,8,9,10,11,12 | 0.5768          | 3          | 2.0871   | 7          | 1211.1279 | 3         | 0.0298        | 11.0000         |
| spa_mo_model | 2,3,4,5,6,7,8,9,10,11,12 | 0.5971          | 4          | 1.9934   | 4          | 1339.6170 | 2         | NA            | NA              |

该表只列出有无监督内部指标 CSV 的方法。spa_mo_model 的这部分为本报告按 k=2-12 补算，COSIE 为已有分析输出；MOFA+ 和 SpaMosaic 的 MouseBrain 目录主要提供 best-by-label 聚类表，因此不强行补齐。

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
| spa_mo_model | 8  | 0.5204     | 0.0408  | 15851.7216 | 3.9999 | NA        | NA        |
| MOFA+        | 8  | 0.5439     | 0.0878  | 40585.6299 | 2.0228 | 0.0007    | 0.0166    |
| COSIE        | 8  | 0.5599     | 0.1199  | 52809.6960 | 2.4652 | 0.1317    | 0.2232    |
| SpaMosaic    | 8  | 0.5512     | 0.1023  | 56606.7603 | 2.3601 | 0.0275    | 0.0652    |
| spa_mo_model | 10 | 0.5184     | 0.0367  | 13485.8551 | 4.1352 | NA        | NA        |
| MOFA+        | 10 | 0.5403     | 0.0805  | 37023.8431 | 2.0193 | 0.0128    | 0.0341    |
| COSIE        | 10 | 0.5627     | 0.1255  | 45958.7476 | 2.2984 | 0.0835    | 0.2051    |
| SpaMosaic    | 10 | 0.5499     | 0.0999  | 49612.5344 | 2.2502 | 0.0183    | 0.0644    |

在共同 k=8/10 上，COSIE 的 ASW scaled 最高，k=8 为 0.5599、k=10 为 0.5627；SpaMosaic 次之，MOFA+ 再次。spa_mo_model 的 ASW scaled 约 0.52，低于另外三种方法，DBI 也更高，说明其 CRC fullspot embedding 的簇间 silhouette 分离和 DBI 内部结构并不占优。COSIE 的 group ARI/NMI 最高，提示其聚类更强地携带 CRC_003/CRC_006 样本来源信息。

### CRC 各方法报告 k 范围内最佳内部指标
| 方法           | reported k                     | best ASW scaled | best ASW k | best DBI | best DBI k | best CH     | best CH k | max group ARI | max group ARI k |
| ------------ | ------------------------------ | --------------- | ---------- | -------- | ---------- | ----------- | --------- | ------------- | --------------- |
| spa_mo_model | 8,10,15,20,25                  | 0.5204          | 8          | 3.8727   | 15         | 15851.7216  | 8         | NA            | NA              |
| MOFA+        | 2,3,4,5,6,7,8,9,10,11,12,20,25 | 0.5453          | 2          | 1.9295   | 25         | 62758.1445  | 2         | 0.0140        | 20.0000         |
| COSIE        | 2,3,4,5,6,7,8,9,10,11,12       | 0.5738          | 6          | 2.0833   | 6          | 97026.2326  | 2         | 0.2008        | 4.0000          |
| SpaMosaic    | 2,3,4,5,6,7,8,9,10,11,12       | 0.5797          | 2          | 2.0583   | 3          | 121861.9027 | 2         | 0.0546        | 5.0000          |

这个表按每个方法自己报告的 k 范围选最优值，因此不是严格同 k 对比。SpaMosaic 的最佳 ASW 和 CH 较高；COSIE 的 ASW 也较强但 group ARI 偏高；spa_mo_model 的最佳 ASW、DBI 和 CH 在当前 CRC 指标中均不占优，说明这次 CRC fullspot 运行在无监督内部聚类质量上弱于 COSIE/SpaMosaic/MOFA+。

### CRC 空间连续性
| 模式    | k  | spa_mo_model | MOFA+  | COSIE  | SpaMosaic |
| ----- | -- | ------------ | ------ | ------ | --------- |
| 分样本聚类 | 5  | NA           | 0.4664 | 0.9108 | 0.6390    |
| 分样本聚类 | 6  | NA           | 0.4447 | NA     | 0.5578    |
| 分样本聚类 | 8  | 0.4789       | 0.4395 | 0.8621 | 0.4805    |
| 分样本聚类 | 10 | 0.3940       | 0.3732 | 0.8662 | 0.4106    |
| 分样本聚类 | 15 | 0.3252       | NA     | 0.8218 | NA        |
| 分样本聚类 | 20 | 0.2545       | 0.2617 | 0.7857 | 0.2676    |
| 分样本聚类 | 25 | 0.2202       | 0.2323 | 0.7706 | 0.2335    |
| 联合聚类  | 5  | NA           | 0.3958 | 0.9215 | 0.6599    |
| 联合聚类  | 6  | NA           | 0.3880 | NA     | 0.5931    |
| 联合聚类  | 8  | 0.4923       | 0.4186 | 0.8973 | 0.5222    |
| 联合聚类  | 10 | 0.4608       | 0.3797 | 0.8958 | 0.4507    |
| 联合聚类  | 15 | 0.3527       | NA     | 0.8545 | NA        |
| 联合聚类  | 20 | 0.2850       | 0.2949 | 0.8441 | 0.2954    |
| 联合聚类  | 25 | 0.2464       | 0.2630 | 0.8239 | 0.2624    |

CRC 空间连续性上，COSIE 明显最高，联合聚类 k=5 为 0.9215，k=20 仍为 0.8441。spa_mo_model 的空间连续性是本报告从 label CSV 和坐标派生计算得到；它在 k=8/10 约 0.46-0.49，通常高于 MOFA+，与 SpaMosaic 接近但不稳定，明显低于 COSIE。MOFA+ 与 SpaMosaic 在 k=20/25 附近接近，约 0.26-0.30。

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

## 综合结论

1. MouseBrain：spa_mo_model 在本次补算的 ARI/NMI/Label ASW 上整体最好，并且保持较高空间连续性；COSIE 次之，SpaMosaic 的 Cluster ASW 很高，MOFA+ 相对较低。
2. CRC：COSIE 在共同 k=8/10 的 ASW scaled 和空间连续性上最好，但 section/group 相关性也更强；SpaMosaic 的最佳 ASW/CH 表现较好；spa_mo_model 在当前 fullspot 运行中无监督内部指标不占优，空间连续性处于中等；MOFA+ 稳定但整体不突出。
3. 由于配置差异很大，尤其 COSIE 使用 metacell、spa_mo_model 使用 fullspot OT/attention、SpaMosaic 使用 CE loss、MOFA+ 是因子模型，这些结果更适合作为 baseline 观察，不适合直接作为最终胜负判断。
4. 如果要进一步做严格比较，建议统一 k 列表、KMeans random seed/n_init、embedding 标准化、ASW sample size，并为 CRC 补充统一的细胞类型或区域真值标签。

## 派生文件

- 本报告：`/home/hujinlan/spa_mo_model/spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md`
- spa_mo_model MouseBrain 补算指标目录：`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/metrics`
- spa_mo_model MouseBrain batch correction 指标：`/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/batch_correction_metrics.csv`
- spa_mo_model CRC 空间连续性派生表：`/home/hujinlan/spa_mo_model/report_derived_metrics/spa_mo_model_crc_spatial_continuity_for_report.csv`
- spa_mo_model CRC batch correction 指标：`/home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/batch_correction_metrics.csv`

## 指标来源补充

| 数据集        | 方法           | 来源                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MouseBrain | spa_mo_model | final_embeddings + spa_mo_model/data/dataset_MouseBrain；补算 CSV 写入 analysis/metrics                                                                                                                                                                                                                                                                                           |
| MouseBrain | MOFA+        | /home/hujinlan/mofa+/analysis/mousebrain_mofa_rna_meta_uni_hvg2000_k10_iter1000/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                 |
| MouseBrain | COSIE        | /home/hujinlan/cosie_runs/mousebrain_cosie_rna_meta_he_full/analysis/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                            |
| MouseBrain | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/mousebrain_spamosaic/metrics/best_clustering_metrics_by_label.csv                                                                                                                                                                                                                                                                      |
| CRC        | spa_mo_model | /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k8_10_15/clustering_metrics.csv; /home/hujinlan/spa_mo_model/results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/clustering_analysis_k20_25/clustering_metrics.csv |
| CRC        | MOFA+        | /home/hujinlan/mofa+/analysis/crc_stereocite_mofa_hvg2000_k10_full_iter1000_cpu_float64/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                          |
| CRC        | COSIE        | /home/hujinlan/cosie_runs/crc_cosie_rna_adt_metacell_6x6_sparse/analysis/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                                         |
| CRC        | SpaMosaic    | /home/hujinlan/SpaMosaic-dev/analysis/crc_stereocite_spamosaic_full_gpu_ce/metrics/unsupervised_clustering_metrics.csv                                                                                                                                                                                                                                                       |

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

汇总 CSV：`/home/hujinlan/spa_mo_model/report_derived_metrics/batch_correction_metrics_mofa_cosie_spamosaic_mousebrain_crc.csv`
