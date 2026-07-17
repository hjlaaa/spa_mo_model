# spa_mo_model 性能提升详细实施方案

> 文档状态：设计方案，尚未实施  
> 编写日期：2026-07-10  
> 项目目录：`/home/hujinlan/spa_mo_model`  
> 对比报告：`spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md`

## 0. 文档目的与边界

本文档只制定后续修改方案，不包含任何模型代码、配置、训练脚本或结果文件的实际修改。

目标是让实施者可以按照本文档依次完成：

1. 修复当前训练目标中的数值与定义问题；
2. 修复 MISAR-seq ATAC 和 CRC RNA 特征选择问题；
3. 建立可定位性能下降阶段的统一诊断体系；
4. 在稳定基线之上改进融合、空间图和跨切片 OT；
5. 建立可重复、可公平比较的训练与评估流程；
6. 在不牺牲 MouseBrain 生物标签优势的前提下，提升内部簇几何、空间结构与跨切片整合效果。

本文档不授权直接覆盖既有结果，不要求一次性实现所有结构，也不建议在 P0 验收前开始大规模超参数搜索。

---

## 1. 当前问题摘要与修改优先级

### 1.1 当前模型的主要优势

- MouseBrain 的 ARI、NMI、Completeness 和主要 Label ASW 总体第一；
- MISAR-seq 的 RNA_Clusters ARI、Combined/RNA NMI 有明显竞争力；
- CRC、Human Lymph Node、Mouse Spleen 的 batch correction 较强；
- fullspot、candidate sparse UOT、双向 OT attention 和大数据内存优化链路已经能够运行。

这些优势必须作为后续修改的回归保护目标，尤其不能为了提升 ASW 或空间连续性而显著降低 MouseBrain 的生物标签指标。

### 1.2 当前主要短板

1. 跨视图损失数值定义不稳定，且不同 section 数、模态数下损失尺度不可比；
2. CRC、MISAR、Human Lymph Node、Mouse Spleen 的内部簇几何普遍较弱；
3. 空间连续性普遍落后于 COSIE；
4. MISAR 的 ATAC 使用普通表达矩阵预处理，不符合 ATAC 数据特性；
5. CRC 在 HVG 前按原始变量顺序截断共享基因；
6. Harmony 与 OT 可能在部分数据集上产生双重过校正；
7. UOT 只是不可微 prior，没有明确的匹配一致性训练目标；
8. 当前只保存 final embedding，无法定位性能是在 fusion、GraphSAGE 还是 OT attention 后下降；
9. fullspot 每个 epoch 只有一次 optimizer step，且缺少 scheduler、early stopping 和多 seed 统计；
10. 多个配置字段存在但当前实现没有消费，容易造成“配置已改、模型未变”的假实验。

### 1.3 总体执行顺序

| 阶段 | 优先级 | 内容 | 是否允许结构扩展 |
| --- | --- | --- | --- |
| Phase 0 | 准备 | 冻结基线、统一产物和回归指标 | 否 |
| Phase 1 | P0 | 修复跨视图损失和损失归一化 | 仅损失层 |
| Phase 2 | P0 | 修复 MISAR ATAC 与 CRC 基因选择 | 仅预处理 |
| Phase 3 | P0 | 建立 stage-level 诊断和统一评估几何 | 仅诊断/评估 |
| Phase 4 | P1 | Harmony × OT 消融与分阶段训练 | 小范围训练流程 |
| Phase 5 | P1 | 质量感知 shared/private 融合 | 是 |
| Phase 6 | P1 | 多尺度空间图和模态内图编码 | 是 |
| Phase 7 | P1 | 可拒配、可学习的跨切片 OT | 是 |
| Phase 8 | P1/P2 | 模态化重构目标 | 是 |
| Phase 9 | P2 | 优化器、scheduler、多 seed、公平评估 | 训练与评估 |

严格要求：Phase 1–3 未通过验收前，不进入 Phase 5–8 的结构叠加。

---

## 2. Phase 0：冻结现有基线与回归保护

### 2.1 目的

在开始任何修改前，明确“修改前”的模型、配置、数据选择和指标，防止后续无法判断提升来自哪里。

### 2.2 需要记录的基线

至少冻结以下六个现有运行：

- MouseBrain：`results/mousebrain_test/fullspot_warmup_schedule/epochs_200`
- CRC：`results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42`
- MISAR：`results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
- Human Lymph Node：`results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
- Mouse Spleen：`results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`
- Mouse Thymus：`results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42`

### 2.3 新实验目录规则

禁止覆盖以上目录。新目录统一使用：

```text
results_v2/<dataset>/<experiment_id>/seed_<seed>/
```

`experiment_id` 建议格式：

```text
p1_loss-<loss_name>_dim-<dim>_harmony-<0|1>_ot-<0|1>_graph-<graph_name>
```

例如：

```text
results_v2/misar/p1_loss-infonce_vicreg_dim-64_harmony-1_ot-0_graph-spatial1/seed_42
```

### 2.4 每个运行必须保存的元数据

在 `run_summary.json` 中补齐并固定以下字段：

```json
{
  "code_version": "git commit or manual version tag",
  "experiment_id": "...",
  "dataset": "...",
  "model_seed": 42,
  "clustering_seeds": [0, 1, 2, 3, 4],
  "preprocessing": {},
  "model_config": {},
  "loss_config": {},
  "optimizer_config": {},
  "training_steps": 0,
  "epochs": 0,
  "selected_checkpoint": "...",
  "selection_metric": "...",
  "input_feature_hashes": {},
  "environment": {}
}
```

如果运行环境没有 Git，可使用手动 `code_version`，格式为 `p0_baseline_YYYYMMDD`，但不得留空。

### 2.5 回归保护指标

#### MouseBrain 必须保护

- RegionLoupe ARI：不得比 0.7654 下降超过 0.01；
- annotations ARI：不得比 0.6975 下降超过 0.01；
- celltype ARI：不得比 0.6743 下降超过 0.01；
- RegionLoupe NMI：不得比 0.6782 下降超过 0.01；
- batch PCR R² 不得显著高于 0.0134；
- 联合 k=5 空间连续性不得低于 0.84。

这些阈值用于 P0/P1 开发阶段的回归保护，不代表最终论文显著性阈值。

#### 其他数据集改进目标

- CRC：共同 k=8/10 的 ASW 上升、DBI 下降，同时保持当前 batch 优势；
- MISAR：ATAC ARI、共同 k ASW/DBI、bLISI/kBET 至少有一项明确改善，且 RNA ARI 不下降超过 0.01；
- Human Lymph Node / Spleen：内部几何改善时，bLISI/kBET 不出现大幅退化；
- Thymus：优先降低 section effect，并消除 crossview loss 突刺。

### 2.6 Phase 0 验收

- [ ] 六个基线目录只读保留；
- [ ] 新实验目录规范确定；
- [ ] 所有后续配置可完整写入 `run_summary.json`；
- [ ] MouseBrain 回归阈值写入自动比较脚本；
- [ ] 基线指标汇总为机器可读 CSV，而不只存在于 Markdown 报告。

---

## 3. Phase 1 / P0：修复跨视图损失

### 3.1 问题定义

当前 `model/loss.py::compute_joint()` 对可正可负的 L2-normalized latent 计算：

```python
p_i_j = view1.T @ view2
p_i_j = (p_i_j + p_i_j.T) / 2
p_i_j = p_i_j / p_i_j.sum()
```

存在以下问题：

1. `p_i_j` 可能含负值，不是合法概率；
2. 有符号元素可能相互抵消，使分母接近零；
3. clamp 为 EPS 后没有重新归一化；
4. section 和模态 pair 直接求和，使 loss 随 section 数和模态数改变；
5. 当前 loss 不使用 same-spot 正样本和不同 spot 负样本；
6. 初始 crossview loss 量级远大于 reconstruction loss，导致固定 lambda 不可解释。

### 3.2 推荐实现选择

#### 主方案：Symmetric InfoNCE + VICReg variance/covariance

这是推荐作为正式 v2 默认方案的实现。

每个模态 encoder 输出用于聚类的 `z_modality`，再经过独立 projection head 得到 `p_modality`。对任意同一 section 内模态对 A/B：

- 正样本：同一 spot 的 `(A_i, B_i)`；
- 负样本：该 batch 内其他 spot；
- 损失：A→B 和 B→A 对称平均；
- 表示稳定项：对每个模态 projection 增加 variance 和 covariance 约束；
- 聚类 embedding 使用 encoder/fusion 输出，不直接使用 projection head 输出。

推荐公式：

```text
L_pair = 0.5 * [CE(sim(P_A, P_B)/T, diagonal) + CE(sim(P_B, P_A)/T, diagonal)]
L_vicreg = lambda_var * variance_loss + lambda_cov * covariance_loss
L_cross = mean_over_sections_and_pairs(L_pair) + L_vicreg
```

初始推荐超参数：

| 参数 | 初始值 | 搜索范围 |
| --- | ---: | --- |
| projection_dim | 64 | 32, 64, 128 |
| temperature | 0.2 | 0.07, 0.1, 0.2, 0.5 |
| lambda_var | 1.0 | 0.1, 1.0, 5.0 |
| lambda_cov | 0.04 | 0.01, 0.04, 0.1 |
| negative_mode | in_batch | in_batch, queue |
| max_contrastive_batch | 4096 | 1024, 2048, 4096 |

#### 备选方案 A：只使用 Symmetric InfoNCE

用于最小消融，验证几何改善是否主要来自 same-spot 对比目标。

如果只用 InfoNCE 出现维度坍塌、有效秩降低或对大 batch 过敏，再启用 VICReg 项。

#### 备选方案 B：修正后的 COSIE dimension loss

仅用于兼容性和机制对照，不建议直接作为最终默认方案。

修正要求：

1. 对 view 使用 `softmax(view / temperature, dim=1)` 或其他非负化；
2. 构造 joint 后 clamp；
3. clamp 后重新除以 joint 总和；
4. marginals 必须从重新归一化后的 joint 计算；
5. denominator 使用显式 eps；
6. 对 section 和 pair 取均值；
7. 输出 joint 的 min/max/sum 诊断。

不得只通过把 `lambda_contrast` 调小来掩盖概率定义问题。

### 3.3 配置接口设计

在 `get_default_model_config()` 的 `contrastive` 部分规划以下结构：

```python
"contrastive": {
    "method": "symmetric_infonce_vicreg",
    "projection_dim": 64,
    "projection_hidden_dim": 128,
    "temperature": 0.2,
    "normalize_projection": True,
    "negative_mode": "in_batch",
    "max_batch_size": 4096,
    "queue_size": 16384,
    "lambda_var": 1.0,
    "lambda_cov": 0.04,
    "pair_reduction": "mean",
    "section_reduction": "mean",
    "detach_target": False,
}
```

保留旧方法时显式命名：

```text
legacy_cosie_dimension
corrected_cosie_dimension
symmetric_infonce
symmetric_infonce_vicreg
```

禁止让 `cosie_crossview` 同时表示旧实现和修正实现。

### 3.4 预期修改文件

| 文件 | 预期修改 |
| --- | --- |
| `model/loss.py` | 新增稳定 crossview loss、VICReg 项、分块或批量计算接口；旧函数保留为 legacy |
| `model/model_component.py` | 新增 modality projection head，或独立 `ProjectionHead` 类 |
| `model/configure.py` | 增加 loss 方法与超参数配置 |
| `model/stage_model.py` | 按 section/pair 取均值；返回细分 loss；选择 loss 方法 |
| `scripts/run_*.py` | CLI/config 透传，不在脚本内复制 loss 逻辑 |
| `tests/test_crossview_loss.py` | 新增概率、数值、梯度、对称性和批量测试 |
| `scripts/validate_crossview_loss.py` | GPU/大 batch smoke 验证，可选 |

### 3.5 建议函数接口

```python
def symmetric_infonce_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float,
    max_batch_size: int | None = None,
    negative_mode: str = "in_batch",
) -> torch.Tensor:
    ...


def vicreg_regularization(
    views: Mapping[str, torch.Tensor],
    variance_target: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...


def compute_pairwise_crossview_loss(
    projected_latents: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
```

### 3.6 CRC 大数据实现要求

CRC 有 612,374 spots，不能构造全量 N×N similarity matrix。

推荐顺序：

1. 每次训练从每个 section 分层采样相同数量 spots；
2. 对每个采样 batch 计算 same-spot 跨模态 InfoNCE；
3. batch size 首先使用 2048；
4. 如果负样本不足，再使用 FIFO queue；
5. queue 内 tensor 必须 `detach()`，不得反向传播到历史 batch；
6. 保留 fullgraph reconstruction/GraphSAGE 路径时，可将 contrastive 子采样与全图 forward 解耦；
7. 记录每 epoch 实际 contrastive spot 数和 optimizer step 数。

不推荐对 CRC 直接分块计算全量 N×N logits，因为即使分块，计算量也过大。

### 3.7 Loss 归一化与权重策略

总损失规划为：

```text
L_total = w_rec * L_rec_mean
        + w_cross * L_cross_mean
        + w_var * L_var
        + w_cov * L_cov
        + 后续阶段的其他损失
```

所有分量必须先变成与 section 数、spot 数、模态对数无关的均值，再乘权重。

训练日志新增：

- raw loss；
- weighted loss；
- 每个 loss 对 encoder 参数的 gradient norm；
- projection effective rank；
- 表示每维标准差最小值/中位数；
- InfoNCE top-1 cross-modal retrieval accuracy；
- temperature；
- 实际 batch 和负样本数。

初期不要直接引入自动 GradNorm。先完成固定权重、可解释尺度；如仍存在明显梯度不平衡，再将 GradNorm 作为独立消融。

### 3.8 单元测试

`tests/test_crossview_loss.py` 至少包含：

1. 相同输入比随机打乱输入的 loss 更低；
2. A→B 与 B→A 对称平均结果不依赖调用顺序；
3. batch size 1 给出明确错误或受控行为；
4. 所有输出和梯度有限；
5. fp32、bf16 autocast 下不出现 NaN/Inf；
6. 输入包含负值时稳定；
7. legacy corrected joint 满足 `min >= 0`、`sum≈1`；
8. section/pair 数增加但复制相同数据时，mean reduction 不改变损失尺度；
9. projection variance 低于阈值时 variance loss 增大；
10. covariance 非对角元素增大时 covariance loss 增大。

### 3.9 Phase 1 实验矩阵

先在 MouseBrain、MISAR 和 Spleen 小规模/全量可行配置上运行：

| ID | crossview | VICReg | lambda schedule | 目的 |
| --- | --- | --- | --- | --- |
| L0 | legacy | 无 | 现有 | 基线 |
| L1 | corrected COSIE | 无 | 1e-4→1e-2 | 机制兼容 |
| L2 | symmetric InfoNCE | 无 | 1e-3→1e-1 | same-spot 效果 |
| L3 | symmetric InfoNCE | 有 | 1e-3→1e-1 | 推荐候选 |
| L4 | 无 crossview | 无 | 0 | 判断 crossview 净贡献 |

每个 ID 至少 seed 42、123、2026；最终候选再扩展到 5 seeds。

### 3.10 Phase 1 验收标准

- [ ] crossview loss 不再出现无解释的大负值；
- [ ] 不出现 loss 突刺、NaN、Inf；
- [ ] 同 spot retrieval 明显高于随机打乱；
- [ ] gradient norm 不再被单一分量长期支配；
- [ ] MouseBrain 回归指标通过；
- [ ] 至少在 CRC/MISAR/Spleen 中一个数据集的 ASW 或 DBI 改善；
- [ ] legacy、corrected 和新 loss 能通过配置明确切换；
- [ ] 不覆盖旧 checkpoint。

---

## 4. Phase 2 / P0：修复数据集特异预处理

### 4.1 MISAR-seq ATAC 预处理

### 4.1.1 推荐流程

```text
输入 peak count matrix
→ 删除全局极低覆盖 peaks
→ 选择 variable peaks
→ TF-IDF
→ log1p 或保持标准 TF-IDF 口径（二者做消融）
→ TruncatedSVD/LSI
→ 删除第一深度相关分量
→ 可选 Harmony
→ 输入 modality encoder
```

### 4.1.2 实现选择

#### 推荐实现：稀疏矩阵原生 TF-IDF + TruncatedSVD

- 不将 peak count 转为 dense；
- 使用 CSR/CSC 稀疏矩阵；
- `TruncatedSVD(n_components=n_lsi + 1)`；
- 删除第一分量后保留 `n_lsi`；
- 输出 float32；
- 保存 depth 与各 LSI 分量相关性。

#### 备选实现：调用 SpaMosaic 的现有预处理

不推荐作为最终项目依赖，因为会增加跨项目运行依赖；可用作数值对照，确认本地实现与参考实现一致。

### 4.1.3 配置接口

```python
"preprocessing": {
    "by_modality": {
        "ATAC": {
            "method": "tfidf_lsi",
            "n_variable_peaks": 30000,
            "n_lsi": 50,
            "drop_first_lsi": True,
            "min_cells_per_peak": 10,
            "tfidf_log1p": True,
            "use_harmony": True,
        }
    }
}
```

### 4.1.4 预期修改文件

| 文件 | 预期修改 |
| --- | --- |
| `model/data_preprocessing.py` | 新增 `preprocess_atac_adata()`；ATAC 不再走通用非 Protein 分支 |
| `model/configure.py` | 增加 modality-specific preprocessing 配置 |
| `scripts/run_misar_seq.py` | 透传 ATAC 参数、保存选中 peaks/LSI 诊断 |
| `tests/test_atac_preprocessing.py` | 稀疏性、形状、确定性、深度分量测试 |

### 4.1.5 ATAC 测试

- 输入 sparse matrix 后全过程保持稀疏，SVD 输出除外；
- 同 seed 结果一致；
- 第一 LSI 与 library size 相关性显著高于删除后的主要分量，或至少记录该诊断；
- 输出维度与配置一致；
- 空 peak、全零 spot、重复 peak 名给出明确处理；
- 4 个 MISAR section 的 peak 顺序严格一致；
- 删除第一分量的开关可独立消融。

### 4.1.6 MISAR 实验矩阵

| ID | Peaks | 表示 | Harmony |
| --- | ---: | --- | --- |
| A0 | 3k | 当前 normalize/log/PCA | 开 |
| A1 | 10k | TF-IDF/LSI | 开 |
| A2 | 30k | TF-IDF/LSI | 开 |
| A3 | 50k | TF-IDF/LSI | 开 |
| A4 | 30k | TF-IDF/LSI | 关 |

先固定模型与 loss，只比较预处理，不得同时修改图结构和 OT。

### 4.1.7 验收

- ATAC_Clusters ARI 不低于当前 0.2597；目标至少达到 0.28；
- RNA_Clusters ARI 不下降超过 0.01；
- 共同 k ASW 上升或 DBI 下降；
- bLISI/kBET 不恶化；
- 输出 peak 列表和 LSI 诊断可复现。

### 4.2 CRC 共享基因与 HVG 选择

### 4.2.1 当前问题

当前流程先取共享基因列表前 N 个，再在该子集上进行 HVG，可能遗漏后部真正高变基因。

### 4.2.2 推荐流程

```text
两 section 共享唯一基因集合
→ 在完整共享集合上分块计算统计量
→ 按 batch-aware HVG 或 deviance 排名选 3000
→ 只将选中的 3000 genes 加载到内存
→ normalize/log/scale/PCA/Harmony
```

### 4.2.3 实现选择

#### 推荐实现 A：分块均值/方差 + batch-aware 合并排名

适合超大 spot 数，工程风险较低：

1. 每个 section 分块计算每个基因的 `sum`、`sum_sq`、nonzero count；
2. 得到均值、方差、检测率；
3. 过滤检测率过低或均值过低基因；
4. 每个 section 内计算标准化 dispersion/deviance 近似分数；
5. 合并两个 section 的 rank；
6. 选 top 3000；
7. 保存完整排名表。

#### 备选实现 B：Scanpy seurat_v3 on downsampled spots

实现简单，但需要验证下采样对 HVG 排名稳定性；仅作为对照。

#### 备选实现 C：deviance-based feature selection

生物学上更合理，但实现和计算成本更高，可在 A 稳定后加入。

### 4.2.4 配置接口

```python
"crc_feature_selection": {
    "method": "streaming_batch_hvg",
    "n_top_genes": 3000,
    "min_detection_rate": 0.001,
    "chunk_genes": 512,
    "chunk_spots": 50000,
    "rank_merge": "mean_rank",
    "preserve_forced_markers": [],
}
```

### 4.2.5 预期修改文件

| 文件 | 预期修改 |
| --- | --- |
| `scripts/run_crc_stereocite.py` | 将“前 N 个共享基因”改为完整共享集合上的 streaming selection |
| `model/data_preprocessing.py` | 可复用通用 HVG 接口，但不处理 backed 读取编排 |
| `tests/test_crc_feature_selection.py` | 顺序不变性、分块等价性、重复基因测试 |

### 4.2.6 必须保存的产物

- `shared_gene_universe.txt`
- `selected_hvg.txt`
- `hvg_statistics_by_section.csv`
- `hvg_merged_ranking.csv`
- 强制 marker 的加入原因；
- 输入基因顺序 hash；
- 选中特征顺序 hash。

### 4.2.7 单元测试

1. 随机打乱 `var_names` 顺序不改变最终 HVG 集合；
2. streaming 与小矩阵一次性计算结果一致；
3. 两 section 基因顺序不同但集合相同时，结果正确对齐；
4. 重复基因、人工 `-1` 后缀和缺失基因处理明确；
5. 选中数量准确；
6. 全零和极低检测率基因不会入选；
7. 同 seed/同输入完全确定。

### 4.2.8 CRC 验收

- 不再存在按原始顺序截断；
- HVG 选择对输入列顺序不敏感；
- RNA PCA/embedding 有效秩和方差分布被记录；
- 共同 k ASW/DBI 至少一项改善；
- batch correction 不显著退化；
- 运行内存仍在可接受范围。

---

## 5. Phase 3 / P0：建立 stage-level 诊断与统一评估

### 5.1 目的

当前只评价 final embedding，无法判断以下哪一步造成几何下降或 batch 过校正：

```text
preprocessed modality features
→ modality latent
→ fused embedding
→ GraphSAGE embedding
→ OT-attention final embedding
```

### 5.2 每一级必须保存的表示

默认保存轻量格式：

```text
embeddings/
  modality/<section>/<modality>.npy
  fused/<section>.npy
  graphsage/<section>.npy
  final/<section>.npy
```

CRC 可配置：

- 全量保存 fused/graphsage/final；
- modality latent 可只保存分层采样和摘要统计；
- 如果磁盘不足，使用 float16 存档但评估时转 float32；
- 必须记录 dtype 和采样索引。

### 5.3 每一级统一计算的指标

#### 生物标签指标

- ARI、NMI、Homogeneity、Completeness、Label ASW；
- 只在有可靠标签的数据集计算；
- best-k 和 fixed-k 分开报告。

#### 内部几何

- ASW；
- DBI；
- CH；
- effective rank；
- participation ratio；
- 每维方差分位数；
- 最近邻距离分布。

#### Batch

- bASW；
- bLISI；
- kBET；
- PCR；
- section classifier cross-validation accuracy，作为补充诊断。

#### 空间

- 当前 spatial neighbor agreement；
- Moran's I；
- Geary's C；
- 邻域保持率；
- 聚类边界长度/碎片数；
- 若有真值，边界 F1 或边界邻域一致性。

#### 跨模态

- same-spot retrieval top-1/top-5/top-10；
- matched-vs-shuffled cosine；
- cross-modal kNN overlap；
- modality classifier accuracy：过高表示模态未对齐，过低同时需排除表示坍塌。

#### OT

- top-k coverage；
- normalized entropy；
- top1-top2 margin；
- reciprocal match rate；
- cycle consistency；
- target usage zero rate；
- target hit Gini/hubness；
- 若有标签，match-label agreement。

### 5.4 统一聚类与评估几何

当前部分脚本在原始 embedding 上聚类、在 StandardScaler 后的 embedding 上计算内部指标。后续统一为：

```text
fit_transform = StandardScaler(optional) → PCA(optional) → L2(optional)
cluster_input = fit_transform(embedding)
metric_input = 同一个 fit_transform(embedding)
```

配置必须明确：

```python
"evaluation_geometry": {
    "standardize": True,
    "pca_dim": None,
    "l2_normalize": False,
    "cluster_algorithm": "kmeans",
    "n_init": 20,
    "max_iter": 500,
    "seeds": [0, 1, 2, 3, 4],
}
```

所有 baseline 也必须用同一评估几何重新计算，不能只重算 spa_mo_model。

### 5.5 固定 k 与 best k

每个数据集同时输出：

1. 固定共同 k；
2. 若有真值，预先指定真实类别数；
3. best-k 作为补充；
4. best-k 不进入主要统计检验；
5. 每个 k 汇报 clustering seed 均值和标准差。

### 5.6 预期修改文件

| 文件 | 预期修改 |
| --- | --- |
| `model/stage_model.py` | 可选返回/保存各 stage embedding |
| `scripts/run_*.py` | 统一 artifact writer |
| `scripts/complete_spa_mo_analysis.py` | 统一几何、扩展 stage-level 指标 |
| `scripts/analyze_*_clustering.py` | 共用同一 clustering/evaluation helper |
| `scripts/evaluate_embedding_stages.py` | 建议新增统一入口 |
| `model/evaluation.py` | 建议新增可复用指标函数，避免脚本复制 |
| `tests/test_evaluation_geometry.py` | 聚类与指标输入一致性测试 |

### 5.7 报告表结构

生成：

```text
metrics/stage_metrics_long.csv
```

字段至少包括：

```text
dataset, experiment_id, seed, stage, section_mode, k,
metric_name, metric_value, metric_direction, n_used,
embedding_transform, clustering_algorithm, clustering_seed
```

### 5.8 Phase 3 验收

- [ ] fused、GraphSAGE、final 可以单独评价；
- [ ] 聚类和内部指标使用同一几何；
- [ ] baseline 与 spa_mo_model 使用同一评估脚本；
- [ ] 每项指标记录 n_used、seed、transform；
- [ ] 可以明确指出每个数据集在哪个 stage 指标下降；
- [ ] MouseBrain stage-level 回归报告可以自动生成。

---

## 6. Phase 4 / P1：Harmony × OT 与分阶段训练

### 6.1 目的

判断 CRC、Human Lymph Node、Spleen 上“batch mixing 强但内部几何弱”是否来自 Harmony 与 OT 双重整合。

### 6.2 四格消融

| ID | Harmony | OT attention | 解释 |
| --- | --- | --- | --- |
| H0O0 | 关 | 关 | 最小基线 |
| H1O0 | 开 | 关 | Harmony 净贡献 |
| H0O1 | 关 | 开 | OT 净贡献 |
| H1O1 | 开 | 开 | 当前组合 |

固定以下内容不变：

- 相同 HVG/LSI；
- 相同 stable crossview loss；
- 相同 latent dim；
- 相同图结构；
- 相同 optimizer steps；
- 相同 seeds。

### 6.3 分阶段训练流程

推荐：

```text
Stage A：encoder + decoder 预训练
Stage B：启用 stable crossview
Stage C：启用空间 GraphSAGE
Stage D：启用 OT attention
Stage E：后续可启用 OT consistency loss
```

建议初始 step 比例：

| 阶段 | 占总 steps | 启用模块 |
| --- | ---: | --- |
| A | 20% | reconstruction |
| B | 30% | + crossview |
| C | 20% | + spatial graph |
| D | 30% | + OT attention |

必须根据 optimizer steps 切换，而不是使用固定 epoch，因为数据集的每 epoch step 数可能不同。

### 6.4 选择规则

不能仅按 batch mixing 选择。无真值数据集使用代理复合分数：

```text
proxy_score = mean_rank(ASW, inverse_DBI, CH, spatial,
                        bASW, bLISI, kBET, PCR,
                        retrieval, effective_rank_penalty)
```

同时绘制 Pareto front：

- x：batch mixing；
- y：内部几何或生物保持；
- 颜色：空间连续性。

### 6.5 验收

- 明确每个数据集应使用 Harmony only、OT only、两者都用或都不用；
- CRC/HLN/Spleen 不再只靠 batch 指标选择模型；
- MouseBrain 当前优势得到保护；
- 将数据集特异选择写入配置，而不是散落在运行脚本中。

---

## 7. Phase 5 / P1：质量感知 shared/private 融合

### 7.1 当前限制

当前 FusionMLP：

```text
concat modality latents → MLP → 加所有 modality latent 的等权平均残差
```

不同模态噪声、信息量和可靠性不同，等权平均可能导致：

- ADT/HE 强模态支配；
- 弱模态噪声污染；
- shared embedding 被迫承载模态私有信号；
- 为了重构所有模态而损害聚类几何。

### 7.2 推荐架构

每个模态拆分：

```text
z_m = encoder_m(x_m)
s_m = shared_projector_m(z_m)
p_m = private_projector_m(z_m)
q_m = quality_gate_m(z_m, reconstruction_error, qc_features)
shared_fused = weighted_fusion({s_m}, weights={q_m})
decoder_m([shared_fused, p_m]) → reconstruct modality m
```

### 7.3 Gate 实现选择

#### 推荐：spot-level softmax gate

```python
gate_logits_m = gate_m(z_m)
weights = softmax(stack(gate_logits_m), dim=modality)
fused = sum_m weights_m * value_m
```

优点：可解释、每个 spot 可自适应。必须保存每个 modality 的 gate 分布。

#### 备选：全局可学习 modality weights

适合作为简单对照，但无法适应局部质量差异。

#### 暂不推荐：完整 Transformer 跨模态融合

参数量和解释复杂度较高，应在 gate 方案无法提升后再考虑。

### 7.4 防止 gate 坍塌

加入以下约束之一：

- gate entropy lower bound；
- modality dropout；
- 每个 batch 至少一定比例 spot 使用每个模态；
- gate prior 与 modality QC/uncertainty 一致；
- 限制单一模态长期平均权重不超过阈值，仅作为开发诊断，不作为硬规则。

### 7.5 Shared/private loss

```text
L = L_reconstruction([shared, private])
  + L_crossview(shared_m)
  + L_orthogonality(shared_m, private_m)
  + L_private_variance
  + L_gate_regularization
```

shared/private 正交项只需防止明显冗余，不应权重过大。

### 7.6 配置接口

```python
"fusion": {
    "mode": "quality_gated_shared_private",
    "shared_dim": 64,
    "private_dim": 32,
    "gate_mode": "spot_softmax",
    "gate_temperature": 1.0,
    "modality_dropout": 0.1,
    "gate_entropy_weight": 0.01,
    "orthogonality_weight": 0.01,
}
```

### 7.7 预期修改文件

- `model/model_component.py`：新增 shared/private projector、gate、fusion；
- `model/stage_model.py`：组织输出和 loss；
- `model/configure.py`：新配置；
- `model/loss.py`：orthogonality/gate regularization；
- `tests/test_gated_fusion.py`：权重归一化、缺失模态、dropout、梯度测试。

### 7.8 必须做的消融

| ID | Fusion |
| --- | --- |
| F0 | 当前 concat MLP + mean residual |
| F1 | concat MLP，无 mean residual |
| F2 | 全局 modality gate |
| F3 | spot-level gate |
| F4 | shared/private + spot-level gate |

先在 MouseBrain、CRC、Spleen 上比较：

- MouseBrain 检查生物标签是否保持；
- CRC 检查 RNA/ADT 是否均衡；
- Spleen 检查 batch 强但几何弱是否改善。

### 7.9 验收

- gate 权重可解释且不坍塌；
- 不同模态的 private latent 保留可重构信息；
- shared latent 的 modality classifier accuracy 降低，但 effective rank 不坍塌；
- CRC/HLN/Spleen 内部几何改善；
- MouseBrain 标签指标通过回归保护。

---

## 8. Phase 6 / P1：多尺度空间图与模态内图编码

### 8.1 当前限制

- 只使用空间 KNN 图；
- feature graph 配置存在但未实际使用；
- 只有一个 WeightedResidualGraphSAGE；
- `num_layers` 配置当前不控制真实层数；
- 空间信息在模态融合之后才进入；
- 没有边界保持或图重构目标。

### 8.2 推荐渐进实现

#### Step G1：修正配置与多层 residual GraphSAGE

先让 `num_layers` 真正生效：

```text
fused → GraphSAGE layer 1 → residual/norm
      → GraphSAGE layer 2 → residual/norm
```

搜索 1、2 层；暂不建议超过 3 层。

#### Step G2：多尺度空间图

同时构建：

- local KNN：k=5；
- medium KNN：k=10/15；
- radius graph：按平台物理尺度选择；
- self-loop。

每种尺度独立聚合后用 gate 融合：

```text
h = W_self x + sum_s alpha_s * Aggregate_s(x)
```

#### Step G3：模态内 feature graph

每个模态分别构建 feature kNN：

- RNA/ADT/ATAC/Metabolite 使用预处理后的 feature；
- HE 使用 UNI/PCA feature；
- feature graph 必须在 section 内构建；
- 排除 self 后取 top-k；
- 可选 mutual kNN，降低 hubness。

#### Step G4：模态内空间+特征双图编码

```text
h_m = Encoder_m(x_m, spatial_graph, feature_graph_m)
```

空间和 feature 消息分别投影，再 gate 融合；不要简单合并边后失去边类型。

### 8.3 边权选择

比较：

1. 当前距离核权重；
2. 可学习 scalar edge gate；
3. GAT-style attention；
4. 距离先验 + learned attention log-bias。

推荐先实现 2，再考虑 3/4，避免一步引入过多复杂性。

### 8.4 Boundary-aware 约束

只提高 neighbor agreement 会奖励过度平滑。建议加入：

- 局部一致性：相近且多模态相似的 spot 为正；
- hard boundary negative：空间相近但多模态差异大的 spot 不应被强制拉近；
- graph reconstruction：预测边是否同时具有空间与特征支持；
- residual high-frequency preservation：限制 GNN 前后局部方差被完全抹平。

### 8.5 配置接口

```python
"graph": {
    "encoder_position": "per_modality_and_post_fusion",
    "spatial_scales": [5, 10],
    "radius": None,
    "feature_graph": {
        "enabled": True,
        "k": 10,
        "mutual": True,
    },
    "num_layers": 2,
    "edge_gate": "distance_plus_mlp",
    "boundary_aware": True,
}
```

### 8.6 预期修改文件

- `model/model_component.py`：多层、多图、edge gate；
- `model/utils.py`：空间/feature 图构建和缓存；
- `model/stage_model.py`：per-modality 与 post-fusion 编排；
- `model/configure.py`：删除无效字段或让其真正生效；
- `tests/test_multiscale_graph.py`：图结构、权重归一化、缓存和梯度测试。

### 8.7 消融顺序

| ID | 模态内图 | 融合后图 | 多尺度 | feature graph |
| --- | --- | --- | --- | --- |
| G0 | 无 | 当前单层 | 否 | 否 |
| G1 | 无 | 2 层 residual | 否 | 否 |
| G2 | 无 | 2 层 residual | 是 | 否 |
| G3 | 空间图 | 1 层 | 是 | 否 |
| G4 | 空间+feature | 1 层 | 是 | 是 |

### 8.8 验收

- CRC/MISAR/HLN/Spleen/Thymus 空间连续性至少在多数数据集改善；
- 不能只提升空间连续性而使 ARI/Label ASW 或有效秩明显下降；
- 边界碎片数、Moran's I 等新增指标同步报告；
- `num_layers`、feature graph、edge gate 配置确实改变模型；
- 图缓存不会跨 section 或错误配置复用。

---

## 9. Phase 7 / P1：可拒配、可学习的跨切片 OT

### 9.1 当前限制

- UOT prior 在 `torch.no_grad()` 下构建；
- prior 只影响 attention 候选和 log-bias；
- 没有 loss 要求匹配 spot 在最终空间一致；
- 每 20 epoch 从含旧 OT 消息的 final embedding 重建 prior，存在错误自增强；
- confidence 只是 top-k coupling mass coverage，不表示 top-k 内部是否明确；
- 多 section 只连接相邻列表项；
- 中间 section 接收两个更新后简单平均，未按 pair 质量加权；
- 没有 unmatched/dustbin，可能强迫 section-specific 生物状态配对。

### 9.2 推荐架构

#### 9.2.1 Teacher embedding 构建候选

候选来源优先级：

1. EMA teacher 的 pre-OT GraphSAGE embedding；
2. 当前模型的 pre-OT GraphSAGE embedding；
3. 不推荐继续默认使用 final embedding。

EMA 更新：

```text
theta_teacher = m * theta_teacher + (1-m) * theta_student
```

初始 `m=0.99`，搜索 0.99/0.995/0.999。

#### 9.2.2 Coupling momentum

不要每 20 epoch 硬替换。对于共享候选边：

```text
P_t = momentum * P_{t-1} + (1-momentum) * P_new
```

候选集合变化时需要重新对齐边索引；如果实现复杂，可先只对 top-k weight 做 EMA，并把硬替换作为基线。

#### 9.2.3 组合置信度

推荐：

```text
confidence = coverage
           * (1 - normalized_entropy)
           * reciprocal_indicator_or_score
           * cycle_consistency_score
```

至少记录而不一定全部相乘：

- top-k coverage；
- normalized entropy；
- top1-top2 margin；
- reciprocal rate；
- cycle distance；
- local density correction；
- target hubness penalty。

#### 9.2.4 可拒配机制

选择之一：

- partial OT：只运输指定质量比例；
- dustbin/unmatched 节点；
- confidence threshold：低置信 row 不做 attention update；
- top-k 全部低于阈值时保留 source 原表示。

推荐先实现 confidence threshold + keep-source，再实现 partial OT。

#### 9.2.5 可学习 OT consistency loss

对高置信匹配：

```text
L_ot_pair = sum_ij stopgrad(P_ij) * distance(z_i, z_j)
```

并加入对比负样本或 margin，避免所有表示直接坍塌：

```text
L_ot_contrast = weighted InfoNCE / triplet on high-confidence pairs
```

双向 cycle：

```text
i → j → i'，约束 i 与 i' 的 embedding 或空间/标签一致
```

初始阶段对 `P` stop-gradient 即可；等稳定后再考虑可微 Sinkhorn。

### 9.3 Section graph

多 section 不再直接使用 `zip(section_order[:-1], section_order[1:])`。

构建 section graph 的选择：

1. 用户/数据元信息指定；
2. 根据 shared modality 全局分布距离；
3. 完全图后保留 top-r section neighbors；
4. 发育序列数据保留有向相邻关系，但明确由 stage 元信息决定。

每条 section edge 保存：

- 生物意义；
- 是否有向；
- 初始相似度；
- OT confidence；
- 更新权重。

同一 section 接收多个更新时，按 pair confidence 加权，不做简单平均。

### 9.4 Attention 强度调度

现有 beta schedule 配置需要真正启用：

```text
训练早期 beta=0
→ teacher/encoder 稳定后 beta 线性升高
→ 低 confidence pair 保持小 beta
```

gate 初始化应偏向保留 source：

- gate 最后一层 bias 初始化为负值；
- 初始平均 gate 约 0.05–0.1；
- 随训练和 confidence 增加。

### 9.5 配置接口

```python
"uot": {
    "candidate_source": "ema_pre_ot",
    "teacher_momentum": 0.995,
    "coupling_momentum": 0.9,
    "section_graph": "metadata_or_similarity",
    "allow_unmatched": True,
    "confidence": {
        "use_coverage": True,
        "use_entropy": True,
        "use_reciprocal": True,
        "use_cycle": True,
        "min_confidence": 0.2,
    },
    "consistency_loss_weight": 0.1,
    "contrastive_loss_weight": 0.1,
    "start_step": 0.7,
}
```

`start_step` 应最终使用绝对 step 或总 step 比例，避免误解为 epoch。

### 9.6 预期修改文件

- `model/sparse_uot.py`：新增 entropy、reciprocal、cycle、unmatched 诊断/机制；
- `model/stage_model.py`：teacher/pre-OT、pair-weighted updates、OT loss；
- `model/model_component.py`：gate 初始化和 confidence-aware attention；
- `model/configure.py`：OT 配置；
- `scripts/run_*.py`：section graph 元信息、日志和保存；
- `tests/test_ot_confidence.py`；
- `tests/test_section_graph.py`；
- `tests/test_ot_cycle_consistency.py`。

### 9.7 消融矩阵

| ID | Candidate source | Reject | OT loss | Section graph |
| --- | --- | --- | --- | --- |
| O0 | final | 否 | 否 | chain |
| O1 | pre-OT | 否 | 否 | chain |
| O2 | EMA pre-OT | 否 | 否 | chain |
| O3 | EMA pre-OT | confidence threshold | 否 | chain |
| O4 | EMA pre-OT | threshold | weighted contrastive | chain |
| O5 | EMA pre-OT | partial/dustbin | weighted contrastive+cycle | metadata/similarity |

### 9.8 重点数据集

- Thymus：首要检查 section effect、低置信匹配、target usage 和 loss 突刺；
- MISAR：section graph 应体现发育 stage，而不是任意列表顺序；
- CRC：关注两方向 spot 数不平衡和 hubness；
- MouseBrain：防止更强 OT 破坏现有区域标签。

### 9.9 验收

- final-based 自举不再是默认；
- 低置信 spot 可以不更新；
- 多 section 配对不由列表顺序隐式决定；
- OT entropy、reciprocal、cycle、target usage 被记录；
- Thymus batch 指标和 section diagnostic 改善；
- MouseBrain 生物标签回归通过；
- OT loss 不引发表征坍塌。

---

## 10. Phase 8 / P1-P2：模态化重构目标

### 10.1 当前限制

当前从 final embedding 使用 MSE 重构 PCA/Harmony 特征。该目标：

- 无法直接保留 RNA count 分布；
- 无法表达 ATAC 稀疏二项特性；
- 可能迫使 shared embedding 重构模态私有噪声；
- 所有模态默认权重相同。

### 10.2 推荐渐进路线

#### R1：保留低维目标，但按模态使用合理损失

- PCA/LSI/CLR feature：Huber 或标准化 MSE；
- 每个模态 loss 先除以维度和方差尺度；
- 使用 learned uncertainty weighting 或固定经验证权重。

这是最低风险方案，优先实施。

#### R2：shared/private decoder

decoder 输入 `[shared, private_m]`，避免 shared 表示承担全部模态私有信号。

#### R3：原始/近原始数据似然

- RNA：NB、deviance 或 Pearson residual reconstruction；
- ATAC：Bernoulli 或 TF-IDF/LSI reconstruction；
- ADT：CLR/DSB 后 Gaussian/Huber；
- Metabolite：TIC/median normalization 后 robust loss；
- HE：UNI feature projector/reconstruction。

R3 工程量较大，只有 R1/R2 验证重构确实限制几何后再实施。

### 10.3 Loss 自动平衡

优先顺序：

1. 各模态标准化后固定权重；
2. uncertainty weighting；
3. GradNorm，作为后续消融。

必须记录每个模态的 raw/weighted loss 和 gradient norm。

### 10.4 验收

- 单一强模态不长期支配重构梯度；
- shared embedding 内部几何改善；
- private latent 能提高模态重构但不降低 shared 生物指标；
- 复杂原始似然只有在可证明优于低维 Huber/MSE 时保留。

---

## 11. Phase 9 / P2：训练策略与可重复性

### 11.1 按 optimizer steps 对齐

当前 fullspot 每 epoch 只有一次更新。后续报告必须同时给出：

- epochs；
- optimizer steps；
- processed spot-pairs；
- contrastive batches；
- OT refresh 次数。

不同方法比较时优先对齐 optimizer steps 或总计算预算，而不是只写 epoch。

### 11.2 推荐优化器

初始方案：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
gradient_clip_norm = 5.0
warmup = total_steps 的 5%
cosine decay 到 1e-6
```

搜索范围：

- lr：1e-4、3e-4、1e-3；
- weight decay：0、1e-4、3e-4、1e-3；
- clip：1、5、10；
- warmup：0%、5%、10%。

不应在 P0 loss 未稳定时进行大规模搜索。

### 11.3 Checkpoint 选择

有真值数据集：

- 训练时原则上不直接用测试标签选 checkpoint；
- 如果当前实验属于方法开发，可明确标记为 oracle analysis；
- 正式结果使用无监督 validation proxy 或 held-out section/spot。

无真值数据集：

- 使用内部几何、空间、batch、retrieval 的 composite；
- 保存 Pareto 最优 checkpoint，不只保存单一加权分数最优。

### 11.4 Early stopping

推荐：

- 每固定 steps 做 validation；
- patience 按 validation 次数而不是 epoch；
- 同时检查 loss finite、effective rank、gradient norm；
- 表示坍塌时立即标记失败，不继续耗费训练预算。

### 11.5 多 seed

开发期：3 seeds；最终报告：至少 5 seeds。

建议：

```text
42, 123, 2026, 3407, 8888
```

每个模型 seed 下，聚类再运行 5 个 clustering seeds。报告：

- mean；
- std；
- 95% bootstrap CI；
- 每个 seed 原始值；
- paired comparison，尽量使用同一 seed 配对。

### 11.6 确定性与环境

统一设置：

- Python random；
- NumPy；
- Torch CPU/CUDA；
- FAISS seed；
- KMeans seed；
- DataLoader worker seed；
- deterministic 算法状态；
- CUDA/cuDNN/FAISS 版本。

如果某组件无法完全确定，必须在 `run_summary.json` 中注明。

---

## 12. 配置清理计划

以下字段当前存在但需要确认是否真正生效：

- `encoder.residual`；
- `graph.use_feature_graph`；
- `graphsage.num_layers`；
- `uot.use_momentum`；
- `uot.normalize_total_mass`；
- `uot.cost`；
- `loss.use_ot_loss`；
- `loss.use_spatial_smooth_loss`；
- `loss.use_gate_regularization`。

每个字段只能选择：

1. 实现并增加测试；
2. 从配置删除；
3. 标记为 `reserved_not_implemented`，加载时如果用户设为 True 则直接报错。

禁止静默忽略。

建议新增配置验证：

```python
def validate_model_config(config: Mapping[str, Any]) -> None:
    ...
```

验证内容：

- 不支持字段；
- 相互冲突设置；
- loss 权重与对应模块；
- latent/projection/fusion 维度；
- checkpoint 与 chunk 参数依赖；
- OT section graph 完整性；
- modality preprocessing 与 modality 类型一致性。

---

## 13. 数据集专项实施路线

### 13.1 MouseBrain

定位：回归保护集和生物标签主验证集。

执行顺序：

1. stable crossview loss；
2. stage-level 诊断；
3. fusion gate；
4. 多尺度空间图；
5. 最后调整 OT。

重点指标：

- RegionLoupe/annotations/celltype ARI/NMI；
- Label ASW；
- 高 k 空间连续性；
- batch kBET/PCR；
- 细粒度 Y 标签。

禁止仅凭 Cluster ASW 选择模型。

### 13.2 CRC

定位：超大规模、batch correction 强但几何弱的压力集。

执行顺序：

1. 完整共享集合上的 HVG；
2. stable sampled crossview；
3. stage-level 诊断；
4. Harmony × OT；
5. gated fusion；
6. 多尺度/feature graph；
7. partial OT。

重点指标：

- k=8/10 ASW、DBI、CH；
- bLISI、kBET、PCR；
- 分样本与联合空间连续性；
- RNA/ADT retrieval；
- OT 两方向 entropy、hubness、target coverage；
- 计算时间和显存。

### 13.3 MISAR-seq

定位：跨发育 section、RNA/ATAC 双模态验证集。

执行顺序：

1. TF-IDF/LSI；
2. stable crossview；
3. stage-level 诊断；
4. stage-aware section graph；
5. partial OT/cycle；
6. 模态内 feature graph。

重点指标：

- RNA、ATAC、Combined、Y 的 ARI/NMI；
- RNA 指标不能因 ATAC 改进而明显下降；
- Label ASW；
- batch mixing；
- 发育相邻 section 的 OT cycle consistency。

### 13.4 Human Lymph Node

定位：无真值、batch mixing 接近最优但几何弱。

执行顺序：

1. stable crossview；
2. Harmony × OT；
3. shared/private fusion；
4. 多尺度空间图；
5. marker/label transfer 补充评估。

重点避免过度整合。

### 13.5 Mouse Spleen

定位：batch correction 第一但内部几何最后的典型数据集。

执行顺序与 Lymph Node 相同。额外增加：

- marker enrichment；
- 空间域边界碎片数；
- Harmony only 与 OT only 的重点比较。

### 13.6 Mouse Thymus

定位：section effect、低局部 mixing 和 loss 突刺诊断集。

执行顺序：

1. stable crossview；
2. stage-level section classifier；
3. section graph；
4. confidence rejection/partial OT；
5. Harmony × OT；
6. 多 seed。

优先目标不是继续提高 ASW，而是降低 section-driven 几何并保持合理空间结构。

---

## 14. 测试体系规划

当前项目主要依赖 smoke/validation 脚本。建议新增：

```text
tests/
  test_crossview_loss.py
  test_atac_preprocessing.py
  test_crc_feature_selection.py
  test_evaluation_geometry.py
  test_gated_fusion.py
  test_multiscale_graph.py
  test_ot_confidence.py
  test_ot_cycle_consistency.py
  test_section_graph.py
  test_config_validation.py
  test_stage_artifacts.py
```

### 14.1 测试层级

#### Unit

小 tensor/matrix，CPU 秒级完成。

#### Integration

两个 section、每 section 100–1000 spots，覆盖完整 forward/backward/save/load。

#### GPU smoke

验证 bf16/fp16、checkpoint、chunk、FAISS candidate、OT attention。

#### Dataset smoke

每个真实数据集固定小子集，检查预处理和接口，不作为性能结论。

#### Full regression

仅对候选配置运行全量，产出完整指标。

### 14.2 兼容现有验证脚本

保留并扩展：

- `scripts/run_stage_model.py`；
- `scripts/validate_bidirectional_ot_attention.py`；
- `scripts/validate_checkpoint_ot_attention.py`。

现有 legacy 路径应继续能运行，但新默认配置必须走稳定实现。

---

## 15. 实验记录与结果选择

### 15.1 单因素原则

P0 阶段每次只改一个概念块：

- loss；
- preprocessing；
- evaluation；
- Harmony/OT。

禁止一次同时更换 loss、ATAC、fusion、graph 和 OT 后直接与旧结果比较。

### 15.2 每个实验必须回答的问题

1. 修改影响了哪个 stage？
2. 生物标签、几何、空间、batch 分别如何变化？
3. 改善是否跨 seeds 稳定？
4. 是否只是提高空间平滑或 batch mixing？
5. 是否出现 effective rank 下降或单模态支配？
6. 计算成本增加多少？
7. 是否通过 MouseBrain 回归保护？

### 15.3 推荐结果表

每个实验输出：

```text
comparison/
  config_diff.json
  metric_delta_vs_baseline.csv
  stage_metric_delta.csv
  seed_summary.csv
  pareto_front.csv
  regression_guard.json
  decision.md
```

`decision.md` 必须明确：保留、拒绝或需要进一步实验，以及原因。

---

## 16. 分支与提交建议

建议按阶段建立独立分支或至少独立提交：

```text
p0-baseline-artifacts
p0-stable-crossview
p0-atac-lsi
p0-crc-streaming-hvg
p0-stage-evaluation
p1-harmony-ot-ablation
p1-gated-fusion
p1-multiscale-graph
p1-ot-consistency
p2-training-reproducibility
```

每个提交应满足：

- 单一目的；
- 有测试；
- 有配置迁移说明；
- 不提交大型结果/embedding；
- 不覆盖 legacy 路径；
- 文档更新与代码同提交。

---

## 17. 风险与回退策略

### 17.1 Stable contrastive loss 导致 MouseBrain ARI 下降

回退：

- 保留 projection head，聚类使用 encoder latent；
- 降低 crossview 权重；
- 延长 reconstruction-only warmup；
- 使用 corrected COSIE 作为过渡；
- 检查 hard negatives 是否错误地把同区域 spot 推远。

### 17.2 Batch mixing 下降但生物几何上升

不立即判失败。查看 Pareto front，并区分：

- 恢复了真实生物结构；
- 重新引入纯技术 section effect。

需用 marker、标签或 held-out section 验证。

### 17.3 多尺度图造成过平滑

回退：

- 降低图层数；
- 加 residual/high-frequency preservation；
- 降低 medium/radius graph 权重；
- 启用 boundary-aware negative；
- 将 GNN 只用于部分 latent channels。

### 17.4 OT consistency 引发表征坍塌

回退：

- OT coupling stop-gradient；
- 降低 OT loss；
- 加 negatives/margin；
- 延迟 OT 启用；
- 提高 confidence threshold；
- 使用 unmatched/dustbin。

### 17.5 CRC 计算资源过高

回退：

- streaming HVG；
- sampled crossview；
- neighbor/subgraph batch；
- float16 artifact；
- 只对候选配置运行 fullspot；
- 所有子采样必须保存索引，不得使用不可追踪的临时随机子集。

---

## 18. 最终验收门槛

只有满足以下条件，才可宣称 spa_mo_model 得到整体性能提升：

### 正确性

- crossview 概率/对比目标定义正确；
- loss 与梯度稳定；
- 配置字段不再静默失效；
- ATAC/CRC 特征选择符合数据类型；
- 聚类与指标使用相同几何。

### 生物保持

- MouseBrain 主要 ARI/NMI 通过回归保护；
- MISAR RNA 指标不因 ATAC 改进显著下降；
- 无真值数据集增加 marker/label-transfer 证据。

### 内部几何

- CRC、MISAR、HLN、Spleen 中至少三个数据集的 ASW/DBI/CH 概念块平均名次改善；
- 改善不能完全由 k=2 或 section 分离贡献；
- effective rank 不坍塌。

### 空间

- 多数数据集空间连续性提高；
- 同时报告并通过边界/过平滑诊断。

### Batch

- CRC、HLN、Spleen 保留可接受 batch mixing；
- MISAR、Thymus 的局部 mixing 有改善；
- 不以最大化 batch mixing 作为唯一目标。

### 稳定性

- 至少 5 seeds；
- 结果均值和 CI 支持提升；
- 无单 seed 排名反转导致结论失效；
- fullspot 可运行且资源可接受。

### 公平性

- 与 baseline 使用相同评估代码、共同 k、seed/n_init、embedding transform；
- 明确 fullspot/metacell、维度、HVG、Harmony 等配置差异；
- 不把方法专属指标混入四方法总榜。

---

## 19. 建议的近期工作包

### Work Package 1：稳定训练目标

范围：Phase 0 + Phase 1。

交付物：

- stable crossview API；
- legacy/corrected/new 三种方法切换；
- loss unit tests；
- MouseBrain/MISAR/Spleen 三数据集消融；
- loss/gradient/effective-rank 诊断。

### Work Package 2：修复输入表示

范围：MISAR ATAC + CRC HVG。

交付物：

- sparse TF-IDF/LSI；
- streaming batch-aware HVG；
- 预处理单元测试；
- 输入特征审计文件；
- 固定模型下的纯预处理比较。

### Work Package 3：定位 stage 影响

范围：Phase 3 + Harmony×OT。

交付物：

- stage artifact writer；
- 统一评估脚本；
- 六数据集 stage delta；
- Harmony×OT Pareto 分析。

### Work Package 4：结构增强

范围：gated shared/private fusion + multiscale graph。

前置条件：WP1–3 全部通过。

### Work Package 5：OT v2

范围：EMA pre-OT、confidence、reject/partial、cycle、section graph。

前置条件：先证明当前 OT 在哪些 stage/数据集导致下降。

---

## 20. 最终执行原则

1. 先修正确性，再调超参数；
2. 先做单因素和 stage 定位，再叠加结构；
3. MouseBrain 是回归保护，不是唯一优化目标；
4. CRC/Spleen 的高 batch mixing 不能掩盖几何弱；
5. COSIE 的高空间连续性不能不加检查地等同于生物准确；
6. SpaMosaic 的高 ASW/CH 提示紧致表示的重要性，但不能直接复制低 k 优势；
7. 所有实验必须可追踪、可重复、不覆盖旧结果；
8. 最终目标是在生物标签、簇几何、空间结构和 batch correction 之间得到稳定 Pareto 改善，而不是只优化单一指标。
