# MouseBrain Clustering Analysis Report

## 1. COSIE downstream reference

本轮读取并参考了以下 COSIE 下游分析文件：

- `/home/hujinlan/cosie/COSIE/downstream_analysis.py`
- `/home/hujinlan/cosie/Tutorials/1_Tutotial_of_SPOTS.ipynb`
- `/home/hujinlan/cosie/Tutorials/2_Tutorial_of_10x_Tonsil_integration.ipynb`

COSIE 的下游聚类分析核心逻辑主要在 `.py` 文件中实现，例如 `cluster_and_visualize()` 和 `cluster_and_visualize_superpixel()`，tutorial notebook 主要负责调用这些函数并展示结果。因此本项目采用 `.py` 脚本作为主要入口，未额外生成 notebook。

本轮新增脚本：

- `/home/hujinlan/spa_mo_model/scripts/analyze_mousebrain_clustering.py`

该脚本仿照 COSIE 的 KMeans + spatial visualization 风格，但适配了本项目保存的 `final_embeddings/*.npy` 和 MouseBrain config 中的 RNA h5ad spatial 坐标。

## 2. Input embeddings

本轮严格使用 200 epoch 全量 spot 训练输出，没有使用 dry-run 或 2 epoch embedding 替代。

| Section | Embedding path | Shape |
| --- | --- | --- |
| s1 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s1_final_embedding.npy` | `(2384, 128)` |
| s2 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s2_final_embedding.npy` | `(2820, 128)` |
| s3 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/final_embeddings/s3_final_embedding.npy` | `(2662, 128)` |

## 3. Spatial coordinates

空间坐标来自 MouseBrain 配置文件：

- `/home/hujinlan/spa_mo_model/data/configs/mousebrain_preprocess_train.json`

具体读取每个 section 的 RNA h5ad 中的 `.obsm["spatial"]`：

| Section | Spatial source | Spatial shape | Matches embedding rows |
| --- | --- | --- | --- |
| s1 | `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionA/adata_RNA.h5ad` | `(2384, 2)` | Yes |
| s2 | `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionB/adata_RNA.h5ad` | `(2820, 2)` | Yes |
| s3 | `/home/hujinlan/spa_mo_model/data/dataset_MouseBrain/dataset_MouseBrain_SectionC/adata_RNA.h5ad` | `(2662, 2)` | Yes |

分析过程中没有自动重排 spot 顺序。embedding 行数与 spatial 行数完全一致。

## 4. Joint clustering outputs

Joint clustering 将 s1/s2/s3 的 final embeddings 拼接后统一进行 KMeans，因此 joint label 可以跨 section 比较。

输出根目录：

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering`

| k | Output directory | Cluster composition summary | Mean spatial neighbor agreement | Section dominance |
| --- | --- | --- | --- | --- |
| 5 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k5` | `joint_k5/cluster_composition.csv` | `0.8654` | No strongly section-dominated cluster |
| 6 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k6` | `joint_k6/cluster_composition.csv` | `0.8217` | No strongly section-dominated cluster |
| 8 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k8` | `joint_k8/cluster_composition.csv` | `0.7737` | No strongly section-dominated cluster |
| 10 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k10` | `joint_k10/cluster_composition.csv` | `0.7159` | No strongly section-dominated cluster |

所有 joint clustering 结果均保存：

- `labels_s1.csv`
- `labels_s2.csv`
- `labels_s3.csv`
- `spatial_s1.png`
- `spatial_s2.png`
- `spatial_s3.png`
- `cluster_composition.csv`
- `section_cluster_count.csv`
- `spatial_continuity.csv`

从 cluster composition 看，k=5/6/8/10 中没有 cluster 被单一 section 强烈主导；各 cluster 的最大 section fraction 均低于约 `0.45`，未见明显单切片独占型 cluster。

## 5. Independent clustering outputs

Independent clustering 对每个 section 单独进行 KMeans。注意：independent clustering 中不同 section 的 label 编号不是同一个聚类概念，例如 s1 的 cluster 0 不能直接解释为与 s2 的 cluster 0 相同。

| k | Output directory | Mean spatial neighbor agreement |
| --- | --- | --- |
| 5 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k5` | `0.8608` |
| 6 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k6` | `0.8326` |
| 8 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k8` | `0.7911` |
| 10 | `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k10` | `0.7357` |

所有 independent clustering 结果均保存：

- `labels_s1.csv`
- `labels_s2.csv`
- `labels_s3.csv`
- `spatial_s1.png`
- `spatial_s2.png`
- `spatial_s3.png`
- `section_cluster_count.csv`
- `spatial_continuity.csv`

## 6. Generated figures

Joint clustering spatial plots:

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k5/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k5/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k5/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k6/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k6/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k6/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k8/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k8/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k8/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k10/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k10/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/joint_k10/spatial_s3.png`

Independent clustering spatial plots:

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k5/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k5/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k5/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k6/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k6/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k6/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k8/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k8/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k8/spatial_s3.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k10/spatial_s1.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k10/spatial_s2.png`
- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/independent_k10/spatial_s3.png`

Top-level summary:

- `/home/hujinlan/spa_mo_model/results/mousebrain_test/fullspot_warmup_schedule/epochs_200/analysis/clustering/clustering_analysis_summary.json`

## 7. Initial observations

Joint clustering 结果具有较好的空间连续性。随着 k 从 5 增加到 10，空间邻域一致性逐步下降，说明更大的 k 带来了更细粒度但也更碎片化的划分。

- joint k=5 最连续，但分辨率偏粗；
- joint k=6 在空间连续性和细节之间比较平衡；
- joint k=8 仍可作为后续备选；
- joint k=10 更细，但空间图上更容易出现碎片化。

Independent clustering 的空间连续性与 joint clustering 相近，k=6 下略高于 joint k=6，但 independent label 不能跨 section 直接比较，因此不适合作为跨阶段 cluster identity 的主结果。

从 cluster composition 看，joint clustering 没有明显 section effect：没有 cluster 几乎只来自某一个 section。对于后续 marker 或 metabolite enrichment，初步建议优先从 joint k=6 开始，同时保留 joint k=8 作为更细粒度备选。

## 8. Next-step recommendation

本轮未执行 marker analysis、metabolite enrichment、KNN modality prediction、OT matching QC、UMAP 或其他下游分析。

建议下一步可以按优先级继续：

1. 基于 joint k=6 做 RNA marker 和 metabolite enrichment。
2. 将 joint k=8 作为更细粒度对照。
3. 后续再做 KNN modality prediction，评估 fused embedding 是否保留单模态可预测信息。
4. 单独做 OT prior QC，检查跨 stage matching 是否与空间/组织结构相符。

## 9. Temporary files

本轮没有创建临时验证文件。

