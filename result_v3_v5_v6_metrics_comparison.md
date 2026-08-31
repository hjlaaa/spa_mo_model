# result_v3 / result_v5 / result_v6 分数据集指标对比

- Compared methods: `result_v3`, `result_v5`, `result_v6`
- Clustering mode: joint only
- Metric space: `standardized_embedding`
- 监督与无监督指标均在每个数据集指定的固定 K 下比较，以控制聚类数变量。
- 固定 K：mousebrain=10，MISAR-seq=14，mouse spleen=5，mouse thymus=8，human lymph node=10，simulation=5，SPATCH=12，CRC=20，human embryo=20。
- 表格格式参照 `/home/hujinlan/report_results/per_dataset`：不同指标类别分开，宽表使用“续表”，数值显示四位小数。
- 除 DBI 越低越好外，其余指标均越高越好。各续表中的版本行顺序均为 result_v3、result_v5、result_v6。

## 1. CRC Stereo-CITE-seq

- Annotation: `TODO / 缺失`
- Sections/samples: 2
- Metric K: 20

### Supervised clustering

TODO / 缺失（三个版本没有共同可比的监督指标）

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| crc_stereocite | result_v3 | 20 | standardized_embedding | 612374 | 128.0000 |
| crc_stereocite | result_v5 | 20 | standardized_embedding | 612374 | 128.0000 |
| crc_stereocite | result_v6 | 20 | standardized_embedding | 612374 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| crc_stereocite | 0.0277 | 0.5138 | 8101.7660 | 4.1871 | OK |
| crc_stereocite | 0.0268 | 0.5134 | 8148.9245 | 4.0950 | OK |
| crc_stereocite | 0.0394 | 0.5197 | 12579.2476 | 3.7229 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：v6 四项无监督指标全部最优。

## 2. Human embryo HESTA

- Annotation: `celltype`
- Sections/samples: 7
- Metric K: 20
- Batch interpretation: developmental section is biological time, not a nuisance batch.

### Supervised clustering

*续表 1/2*

| dataset | method | annotation | K | metric_space | n labeled |
|---|---|---|---:|---|---:|
| human_embryo | result_v3 | celltype | 20 | standardized_embedding | 1029902 |
| human_embryo | result_v5 | celltype | 20 | standardized_embedding | 1029902 |
| human_embryo | result_v6 | celltype | 20 | standardized_embedding | 1029902 |

*续表 2/2*

| dataset | n classes | ARI ↑ | NMI ↑ | Homogeneity ↑ | Completeness ↑ | status |
|---|---:|---:|---:|---:|---:|---|
| human_embryo | 42 | 0.1451 | 0.2702 | 0.2747 | 0.2660 | OK |
| human_embryo | 42 | 0.1606 | 0.2876 | 0.2949 | 0.2807 | OK |
| human_embryo | 42 | 0.1572 | 0.2887 | 0.2948 | 0.2828 | OK |

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| human_embryo | result_v3 | 20 | standardized_embedding | 1029902 | 128.0000 |
| human_embryo | result_v5 | 20 | standardized_embedding | 1029902 | 128.0000 |
| human_embryo | result_v6 | 20 | standardized_embedding | 1029902 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| human_embryo | 0.0116 | 0.5058 | 58.2152 | 3.9184 | OK |
| human_embryo | 0.0168 | 0.5084 | 59.6945 | 3.8769 | OK |
| human_embryo | 0.0150 | 0.5075 | 58.1955 | 3.8965 | OK |

### Batch integration

*续表 1/2*

| method | n | n used | bASW ↑ | bLISI ↑ |
|---|---:|---:|---:|---:|
| result_v3 | 100001 | 100001 | 0.9486 | 0.2868 |
| result_v5 | 100001 | 100001 | 0.9526 | 0.2834 |
| result_v6 | 100001 | 100001 | 0.9433 | 0.2857 |

*续表 2/2*

| method | kBET ↑ | PCR_score ↑ | status |
|---|---:|---:|---|
| result_v3 | 0.3134 | 0.9905 | OK |
| result_v5 | 0.3179 | 0.9956 | OK |
| result_v6 | 0.3078 | 0.9958 | OK |

结论：固定 K=20 后，v5 的 ARI、Homogeneity 及四项无监督指标最好；v6 的 NMI、Completeness 和 PCR_score 最好；v5 的 bASW、kBET 最好，v3 的 bLISI 最好。PCR_score 的 v3 使用 10 个 components，v5/v6 使用 50 个，微小差异需谨慎解释。

## 3. Human Lymph Node

- Annotation: `manual-anno`，仅 v6 具有对应监督结果
- Sections/samples: 2
- Metric K: 10

### Supervised clustering

TODO / 缺失（v3/v5 没有与 v6 A1 `manual-anno` 对应的监督指标，暂不比较）

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| human_lymph_node | result_v3 | 10 | standardized_embedding | 6843 | 128.0000 |
| human_lymph_node | result_v5 | 10 | standardized_embedding | 6843 | 128.0000 |
| human_lymph_node | result_v6 | 10 | standardized_embedding | 6843 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| human_lymph_node | 0.0859 | 0.5430 | 357.3155 | 2.5785 | OK |
| human_lymph_node | 0.0848 | 0.5424 | 362.5925 | 2.8065 | OK |
| human_lymph_node | 0.1232 | 0.5616 | 559.9553 | 2.1044 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=10 后，v6 四项无监督指标全部最优。

## 4. MISAR-seq

- Annotation: `Combined_Clusters_annotation`
- Sections/samples: 4
- Metric K: 14

### Supervised clustering

*续表 1/2*

| dataset | method | annotation | K | metric_space | n labeled |
|---|---|---|---:|---|---:|
| misar_seq | result_v3 | Combined_Clusters_annotation | 14 | standardized_embedding | 7118 |
| misar_seq | result_v5 | Combined_Clusters_annotation | 14 | standardized_embedding | 7118 |
| misar_seq | result_v6 | Combined_Clusters_annotation | 14 | standardized_embedding | 7118 |

*续表 2/2*

| dataset | n classes | ARI ↑ | NMI ↑ | Homogeneity ↑ | Completeness ↑ | status |
|---|---:|---:|---:|---:|---:|---|
| misar_seq | 16 | 0.2648 | 0.4654 | 0.4997 | 0.4355 | OK |
| misar_seq | 16 | 0.2581 | 0.4553 | 0.4877 | 0.4268 | OK |
| misar_seq | 16 | 0.2531 | 0.4631 | 0.4948 | 0.4352 | OK |

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| misar_seq | result_v3 | 14 | standardized_embedding | 7118 | 128.0000 |
| misar_seq | result_v5 | 14 | standardized_embedding | 7118 | 128.0000 |
| misar_seq | result_v6 | 14 | standardized_embedding | 7118 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| misar_seq | 0.1201 | 0.5600 | 369.1751 | 2.3332 | OK |
| misar_seq | 0.1393 | 0.5697 | 486.9321 | 2.0728 | OK |
| misar_seq | 0.1026 | 0.5513 | 311.4711 | 2.4688 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=14 后，v3 四项监督指标全部最好，v5 四项无监督指标全部最好。

## 5. Mouse Spleen

- Annotation: `TODO / 缺失`
- Sections/samples: 2
- Metric K: 5

### Supervised clustering

TODO / 缺失（三个版本没有共同可比的监督指标）

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| mouse_spleen | result_v3 | 5 | standardized_embedding | 5336 | 128.0000 |
| mouse_spleen | result_v5 | 5 | standardized_embedding | 5336 | 128.0000 |
| mouse_spleen | result_v6 | 5 | standardized_embedding | 5336 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| mouse_spleen | 0.0586 | 0.5293 | 308.5379 | 3.5073 | OK |
| mouse_spleen | 0.0588 | 0.5294 | 303.5036 | 3.5104 | OK |
| mouse_spleen | 0.0925 | 0.5462 | 581.0387 | 2.5793 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=5 后，v6 四项无监督指标全部最优。

## 6. Mouse Thymus

- Annotation: `TODO / 缺失`
- Sections/samples: 4
- Metric K: 8

### Supervised clustering

TODO / 缺失（三个版本没有共同可比的监督指标）

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| mouse_thymus | result_v3 | 8 | standardized_embedding | 17824 | 128.0000 |
| mouse_thymus | result_v5 | 8 | standardized_embedding | 17824 | 128.0000 |
| mouse_thymus | result_v6 | 8 | standardized_embedding | 17824 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| mouse_thymus | 0.2061 | 0.6030 | 2413.8545 | 2.3529 | OK |
| mouse_thymus | 0.2038 | 0.6019 | 2366.4818 | 2.2707 | OK |
| mouse_thymus | 0.2224 | 0.6112 | 2909.2521 | 1.9100 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=8 后，v6 四项无监督指标全部最好。

## 7. Mousebrain

- Annotation: `RegionLoupe`
- Sections/samples: 3
- result_v3 run: `v3_bidirectional_sparse_fixed_lc0.1`
- Metric K: 10

### Supervised clustering

*续表 1/2*

| dataset | method | annotation | K | metric_space | n labeled |
|---|---|---|---:|---|---:|
| mousebrain | result_v3 | RegionLoupe | 10 | standardized_embedding | 7866 |
| mousebrain | result_v5 | RegionLoupe | 10 | standardized_embedding | 7866 |
| mousebrain | result_v6 | RegionLoupe | 10 | standardized_embedding | 7866 |

*续表 2/2*

| dataset | n classes | ARI ↑ | NMI ↑ | Homogeneity ↑ | Completeness ↑ | status |
|---|---:|---:|---:|---:|---:|---|
| mousebrain | 9 | 0.4246 | 0.6041 | 0.6997 | 0.5314 | OK |
| mousebrain | 9 | 0.4143 | 0.6074 | 0.7065 | 0.5326 | OK |
| mousebrain | 9 | 0.3886 | 0.5891 | 0.6830 | 0.5179 | OK |

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| mousebrain | result_v3 | 10 | standardized_embedding | 7866 | 128.0000 |
| mousebrain | result_v5 | 10 | standardized_embedding | 7866 | 128.0000 |
| mousebrain | result_v6 | 10 | standardized_embedding | 7866 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| mousebrain | 0.1461 | 0.5730 | 688.5608 | 2.0998 | OK |
| mousebrain | 0.1540 | 0.5770 | 783.8678 | 2.0409 | OK |
| mousebrain | 0.1723 | 0.5861 | 911.3360 | 1.9300 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=10 后，v3 的 ARI 最高，v5 的 NMI、Homogeneity、Completeness 最高，v6 四项无监督指标全部最好。

## 8. Simulation

- Annotation: `spatial_domain`
- Sections/samples: 5
- Metric K: 5

### Supervised clustering

*续表 1/2*

| dataset | method | annotation | K | metric_space | n labeled |
|---|---|---|---:|---|---:|
| simulation | result_v3 | spatial_domain | 5 | standardized_embedding | 6480 |
| simulation | result_v5 | spatial_domain | 5 | standardized_embedding | 6480 |
| simulation | result_v6 | spatial_domain | 5 | standardized_embedding | 6480 |

*续表 2/2*

| dataset | n classes | ARI ↑ | NMI ↑ | Homogeneity ↑ | Completeness ↑ | status |
|---|---:|---:|---:|---:|---:|---|
| simulation | 5 | 0.7097 | 0.8044 | 0.8191 | 0.7902 | OK |
| simulation | 5 | 0.6124 | 0.7517 | 0.7611 | 0.7426 | OK |
| simulation | 5 | 0.6176 | 0.7464 | 0.7575 | 0.7355 | OK |

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| simulation | result_v3 | 5 | standardized_embedding | 6480 | 128.0000 |
| simulation | result_v5 | 5 | standardized_embedding | 6480 | 128.0000 |
| simulation | result_v6 | 5 | standardized_embedding | 6480 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| simulation | 0.3482 | 0.6741 | 1538.1198 | 1.4418 | OK |
| simulation | 0.3871 | 0.6936 | 1788.8928 | 1.4105 | OK |
| simulation | 0.3858 | 0.6929 | 1894.3280 | 1.3396 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=5 后，v3 四项监督指标最好；v5 的 ASW 和 ASW scaled 最好；v6 的 CH 和 DBI 最好。

## 9. SPATCH

- Annotation: `cell_type_common`
- Sections/samples: 2
- Metric K: 12

### Supervised clustering

*续表 1/2*

| dataset | method | annotation | K | metric_space | n labeled |
|---|---|---|---:|---|---:|
| spatch | result_v3 | cell_type_common | 12 | standardized_embedding | 956139 |
| spatch | result_v5 | cell_type_common | 12 | standardized_embedding | 956139 |
| spatch | result_v6 | cell_type_common | 12 | standardized_embedding | 956139 |

*续表 2/2*

| dataset | n classes | ARI ↑ | NMI ↑ | Homogeneity ↑ | Completeness ↑ | status |
|---|---:|---:|---:|---:|---:|---|
| spatch | 16 | 0.1494 | 0.3134 | 0.3982 | 0.2584 | OK |
| spatch | 16 | 0.1478 | 0.3254 | 0.4127 | 0.2686 | OK |
| spatch | 16 | 0.1564 | 0.3301 | 0.4191 | 0.2723 | OK |

### Unsupervised clustering

*续表 1/2*

| dataset | method | K | metric_space | n | embedding_dim |
|---|---|---:|---|---:|---:|
| spatch | result_v3 | 12 | standardized_embedding | 1068962 | 128.0000 |
| spatch | result_v5 | 12 | standardized_embedding | 1068962 | 128.0000 |
| spatch | result_v6 | 12 | standardized_embedding | 1068962 | 128.0000 |

*续表 2/2*

| dataset | ASW ↑ | ASW scaled ↑ | CH ↑ | DBI ↓ | status |
|---|---:|---:|---:|---:|---|
| spatch | 0.1232 | 0.5616 | 65309.9692 | 2.1743 | OK |
| spatch | 0.1324 | 0.5662 | 73538.8823 | 2.1006 | OK |
| spatch | 0.1527 | 0.5763 | 87377.7925 | 1.9736 | OK |

### Batch integration

TODO / 缺失（三个结果目录中没有指定四项批次指标的共同结果）

结论：固定 K=12 后，v6 四项监督指标和四项无监督指标全部最好。

## 总体结论

1. 固定 K 后，v6 的无监督表现仍最稳定：CRC Stereo-CITE、human lymph node、mouse spleen、mouse thymus、mousebrain、SPATCH 的四项无监督指标均为三版最优。
2. human embryo 在 K=20 时由 v5 获得最佳 ARI、Homogeneity 以及全部无监督指标，v6 获得最佳 NMI 和 Completeness。
3. MISAR-seq 在 K=14 时由 v3 获得全部监督指标最优、v5 获得全部无监督指标最优；simulation 在 K=5 时由 v3 获得全部监督指标最优。
4. SPATCH 在 K=12 时由 v6 同时获得全部监督和无监督指标最优。
5. 完整批次指标仅 human embryo 在三个结果目录中均存在；developmental section 包含生物学时间差异，不能只按混合程度判断模型优劣。

## 解释注意事项

- 每个数据集的三个版本以及监督、无监督两类指标均使用同一个指定 K，已经控制聚类数变量。
- 不同数据集使用不同 K，因此只能在同一数据集内部比较版本，不能直接横向比较不同数据集的 CH、DBI 等绝对值。
- ASW scaled=`(ASW+1)/2`，因此与 ASW 的版本排序完全一致。
- 未使用 `section_ari`、`section_nmi` 或 `section_asw` 替代用户指定的监督指标或批次指标。
