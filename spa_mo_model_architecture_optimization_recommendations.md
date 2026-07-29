# spa_mo_model 模型结构优化建议

> 冻结现有对比损失后的架构专项版  
> 编写日期：2026-07-28  
> 本文只做结构分析与方案设计，不包含任何代码、配置或数据修改。

## 0. 结论先行

### 0.1 本文的明确边界

1. **冻结当前 L0 / Legacy COSIE 对比损失。**  
   已完成的损失对比中，L0 在五个全版本公共数据集及包含 CRC 的六数据集可比子集上综合最好；Simulation、SPATCH 没有进入该轮损失横向比较。本文依据这些结果和本次已明确的研究边界冻结 L0：不替换 `compute_joint()`、`crossview_contrastive_Loss()`，不引入 InfoNCE、VICReg、CLIP、DGI、triplet 等替代目标，也不改变现有数据集的 L0 权重或调度。这里的“冻结”不代表 L0 的大幅负值和数值跳变问题已经解决。

2. **不讨论数据预处理优化。**  
   MISAR-seq ATAC、CRC HVG、Harmony 等预处理问题仍然重要，但不属于本次“模型结构或模块如何修改”的范围。

3. **不把单次结果误当作统计结论。**  
   当前主结果大多是单 seed，部分无真值数据集只能说明结构性风险，不能单独证明生物学信息被改善或破坏。

4. **所有新模块都应能从当前模型平滑接入。**  
   推荐采用 identity-preserving residual：新分支初始贡献为 0 或很小，使模型初始行为尽量接近当前基线。每条分支只能选择一种 identity 机制——“零门控 + 非零随机分支”或“固定门控 + 最后一层 zero-init”——不能把门控和分支同时初始化为零，否则会形成零梯度死分支。

### 0.2 一句话诊断

当前模型的主要矛盾不是“对比损失不够强”，而是：

> **同一个 128 维 shared embedding 被迫同时承担模态对齐、多模态融合、空间平滑、跨切片传递和所有模态重构；与此同时，空间图过于单层单尺度，OT 又把高熵或不可匹配的候选当作可用信息。**

这使模型很擅长 batch mixing 和保留总体信息，却经常出现：

- 空间连续性弱于 COSIE；
- batch 已经混得很好，但内部簇几何仍然松散；
- shared embedding 同时保留真实空间因子和模态私有 nuisance；
- 多 section 数据仍被 section 主轴支配；
- 粗粒度组织 compartment 较强，细粒度 cell state 不足。

### 0.3 推荐优先级

| 优先级 | 结构方向 | 首要解决的问题 | 首选验证数据集 |
|---|---|---|---|
| **P1-A** | state/context 双通道、多尺度、关系感知空间图 | 跨数据集最一致的空间连续性短板；避免只会平滑粗区域 | CRC、SPATCH、Simulation、MouseBrain |
| **P1-B** | 双头 encoder + 质量感知融合 | 冻结 L0 的同时，让 fusion 使用非归一化质量信息；去掉硬编码等权贡献 | Human Lymph Node、Spleen、Simulation、SPATCH |
| **P1-C** | shared / modality-private / section-condition 分解 + 条件 decoder | 避免 shared 表示重构模态私有噪声和 section 效应 | Simulation、MISAR、Thymus、Human Lymph Node |
| **P1-D** | topology-aware、可拒配、真实 matchability 的 UOT attention | 固定 top-10、高熵匹配、hub 和 final→prior 自反馈 | CRC、SPATCH、Thymus、Simulation |
| **P1-E** | 显式 section graph + pair-quality aggregation | 相邻链和等权平均无法表达 replicate、发育阶段与非相邻相似性 | Thymus、MISAR |
| **P2-F** | low-rank / prototype-to-spot 层次 OT | 百万 spot 下同时提高匹配质量和可扩展性 | CRC、SPATCH |
| **P2-G** | state/domain 任务化读出与 final-stage 几何头 | 一个 embedding 难以同时服务细胞状态和组织域 | MouseBrain、SPATCH、CRC |
| **P2-H** | 有生物先验的跨模态 feature adapter | RNA–ATAC 等强异质模态仅靠通用 MLP 不足 | MISAR |

如果只允许先做三项，建议顺序是：

1. **P1-A 多尺度 state/context 图；**
2. **P1-B + P1-C 质量融合与 shared/private decoder；**
3. **P1-D + P1-E 可靠 OT 与显式 section graph。**

### 0.4 与原实施计划的关系

本文不是重复原计划，而是在确认 L0 最优后，对其中真正的模型结构部分重新收敛：

| 原计划内容 | 本文处理 |
|---|---|
| Phase 1：替换跨视图损失 | **冻结并移出后续路线** |
| Phase 2：数据集预处理 | 仍有价值，但不属于本次结构建议 |
| Phase 3：stage-level 诊断 | 保留为所有结构实验的必要基础 |
| Phase 4：Harmony × OT 消融 | 属于机制诊断，不作为新结构 |
| Phase 5：quality-gated shared/private fusion | 细化为“双头 encoder 保持 L0 接口 + 稳定质量先验 + learned residual + condition/section 分解” |
| Phase 6：多尺度图 | 细化为“self-path 语义消融 + 密度可比感受野 + identity/high-frequency + state/domain 双输出 + 渐进 per-modality graph” |
| Phase 7：可拒配 OT | 加入已有 top-k 数组实证，细化为 adaptive support、分解后的 matchability、topology cost、EMA pre-OT teacher、section graph 和百万 spot 低秩路径 |
| Phase 8：模态化重构 | 从“换 loss”改为优先改变 decoder 信息通路：shared + modality-private + section/condition-conditioned residual |
| 原计划未单列 | 新增 state/domain 双任务输出、prototype/low-rank OT、MISAR feature-prior adapter |

---

## 1. 审查依据

### 1.1 本地材料

- 现有总计划：[spa_mo_model_performance_improvement_implementation_plan.md](spa_mo_model_performance_improvement_implementation_plan.md)
- 对比损失最终实验：[spa_mo_model_contrastive_loss_final_experiment_report.md](spa_mo_model_contrastive_loss_final_experiment_report.md)
- 8 数据集横向比较：[spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md](spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md)
- 8 个 `*_preprocessing_comparison_report.md`
- 各数据集主运行目录中的 `run_summary.json`、`loss_history.json`、聚类 `SUMMARY.md`、metrics CSV
- 各运行 `ot_prior_topk/` 中的 `topk_idx`、`topk_weight`、`confidence`、mass 和 metadata
- 模型代码：
  - `model/model_component.py`
  - `model/stage_model.py`
  - `model/utils.py`
  - `model/sparse_uot.py`
  - `model/loss.py`
  - `model/configure.py`

### 1.2 当前真实数据流

```text
每个 section、每个模态的 PCA/Harmony 特征 X(s,m)
  │
  ├─ ModalityMLPEncoder_m
  │    └─ L2-normalized z(s,m), 128d
  │          └─ 当前 L0 cross-view loss（只作用在这里）
  │
  ├─ concat(z_m) → FusionMLP → + mean(z_m) → LayerNorm
  │
  ├─ 固定空间 KNN → 单层 WeightedResidualGraphSAGE
  │
  ├─ no_grad UOT top-10 prior
  │      └─ scalar-gated cross-section attention
  │
  └─ final embedding
        └─ 每个模态一个 MLP decoder，以 MSE 重构 PCA/Harmony 特征
```

实际总损失是：

```text
L_total
  = lambda_reconstruction × Σ(section, modality) MSE
  + lambda_contrast × Σ(section, modality pair) L0
```

这意味着：

- L0 只直接监督 modality encoder latent，见 `model/stage_model.py:852-879`；
- Fusion、GraphSAGE 和 OT attention 没有专属的空间、拓扑或匹配监督，只能通过最终重构得到梯度，见 `model/stage_model.py:1047-1064`；
- UOT coupling 的构建与刷新都在 `no_grad` 下，见 `model/stage_model.py:542-691`；
- final embedding 的下游用途是聚类、空间域和跨 section 分析，但训练时主要被要求“重构所有模态的低维输入”。

### 1.3 关键代码事实

| 当前模块 | 代码事实 | 结构后果 |
|---|---|---|
| `ModalityMLPEncoder` | 每个模态只输出一个 L2-normalized latent，`model/model_component.py:32-73` | L0 对齐、质量判断、融合和重构共用一套表示；幅值型质量信息被归一化抹掉 |
| `FusionMLP` | concat-MLP 后硬加所有模态的等权均值，`model/model_component.py:120-153` | 即使某 spot 某模态质量差，也有固定基础贡献 |
| `WeightedResidualGraphSAGE` | 单层、单关系、固定距离权重，`model/model_component.py:156-225` | 无多尺度、无 feature relation、无边界/高频通道 |
| 空间图 | self-loop + `self_linear` + outer residual 同时存在，`model/utils.py:219-306` | self 信息有三条通路，邻域增量容易相对偏弱 |
| candidate-sparse OT confidence | `raw_topk_mass / row_mass`，`model/sparse_uot.py:261-276` | 它只是 top-10 coverage，不等于匹配可靠性 |
| dense OT confidence | `row_mass / (1 / n_source)` 后截断到 1，`model/linkage_construction.py:102-124` | 它是相对均匀源边际的运输质量，也不是校准后的匹配可靠性 |
| 多 section 更新 | section 列表相邻成链；多个方向更新直接算术平均，`model/stage_model.py:955-1000` | section 顺序被隐式当作生物关系；低质量 pair 会稀释高质量 pair |
| Decoder | final embedding 单独重构每个模态，且硬编码 MSE，`model/stage_model.py:430-484` | shared code 被迫保存模态私有、技术和 section 信息 |

---

## 2. 8 个数据集共同指向的结构问题

### 2.1 数据集概览与模块归因

| 数据集 | 规模与模态 | 当前优势 | 最明显的结构短板 | 最适合验证的模块 |
|---|---|---|---|---|
| Simulation | 6,480 spots，5 sections，RNA+ADT | spatial factor recovery `0.9833`，batch mixing 强 | RNA/ADT nuisance leakage `0.9784/0.9592`；跨切片同网格 Top-1 `0.158` | shared/private、decoder、可靠 OT、多尺度图 |
| MouseBrain | 7,866 spots，3 sections，HE+RNA+Metabolite | RegionLoupe/annotations/celltype ARI `0.765/0.697/0.674`；joint k=5 空间连续性 `0.865` | 优势集中在粗区域；细粒度 Y ARI 约 `0.422`，高 k 空间稳定性下降 | state/domain 双通道；所有新模块的回归保护集 |
| CRC Stereo-CITE-seq | 612,374 spots，2 sections，RNA+ADT | 全量可运行；batch ASW 高 | k=8 ASW raw 约 `0.04`、DBI 约 `4.0`；joint k=8 空间连续性 `0.492`，COSIE `0.897`；OT hub 明显 | 大图多尺度编码、可靠/低秩 OT、融合几何 |
| SPATCH | 1,068,962 spots，2 sections，RNA+Protein+HE | full-spot 可扩展；cell_type_common ARI `0.579` 高于 COSIE | spatial_cluster 与 codex coarse 较弱；joint k=10 `0.752`，COSIE `0.888`；细胞类型优势主要集中在少量粗簇 | 多尺度图、state/domain 双输出、层次 OT |
| Human Lymph Node | 6,843 spots，2 sections，RNA+ADT | bLISI `0.933`、kBET `0.824`，section mixing 很好 | k=5 ASW raw `0.098`、DBI `2.54`，内部几何弱 | 质量融合、shared/private decoder；OT 不是第一优先 |
| Mouse Spleen | 5,336 spots，2 sections，RNA+ADT | bLISI `0.982`、kBET `0.970`；candidate-sparse OT retained support 的 coverage/hubness 未见明显异常 | k=5 ASW raw `0.061`、DBI `3.95`；空间连续性随 k 快降 | 质量融合、shared/private decoder、多尺度图 |
| Mouse Thymus | 17,824 spots，4 sections，RNA+ADT | 部分共同 k 的 ASW 较高 | section ARI/NMI `0.307/0.383`、kBET `0.0025`；effective rank 约 `12/128`；中间 OT pair 质量差 | section graph、pair quality、section-private、秩保护 |
| MISAR-seq | 7,118 spots，4 stages，RNA+ATAC | RNA_Clusters ARI `0.307`；空间连续性通常第二；candidate-sparse OT retained support 的 coverage/hubness 未见明显异常 | section ARI `0.275`、kBET `0.033`；既有 joint/independent 数值使用了不同聚类协议，不能据其差距归因于表示或 section graph | RNA/ATAC 专属 encoder、shared/condition-private、全局 section graph |

### 2.2 空间编码是最一致的跨数据集短板

统一分析下，spa_mo_model 的 joint 空间连续性在多个真实或有真值数据集上落后于 COSIE：

| 数据集 | k | spa_mo_model | COSIE |
|---|---:|---:|---:|
| CRC | 8 | `0.492` | `0.897` |
| SPATCH | 10 | `0.752` | `0.888` |
| Human Lymph Node | 8 | `0.514` | `0.658` |
| Mouse Spleen | 8 | `0.365` | `0.542` |
| MISAR | 8 | `0.717` | `0.809` |
| Simulation | 8 | `0.406` | `0.735` |

当前只有一个 fusion 后的固定 k、单层 GraphSAGE，很难同时适应 Visium、Stereo-seq、SPATCH 等完全不同的 spot 密度与组织尺度。

但结果也不支持“简单加深图网络”：

- MouseBrain 在低 k 的粗组织结构已经很好；
- MouseBrain、SPATCH 的细粒度标签明显弱于粗 compartment；
- 继续增强单一低通平滑可能提高连续性，却进一步合并细胞状态和边界。

因此需要的是 **identity/high-frequency 与 multiscale context 并存**，不是单纯更深、更平滑。

### 2.3 强 batch mixing 与弱生物几何同时出现

Human Lymph Node 和 Mouse Spleen 是最清楚的反例：

- section 已经几乎完全混合；
- candidate-sparse OT retained support 的目标覆盖和 hubness 未见明显异常；
- 但 ASW、DBI 和空间连续性仍然弱。

Simulation 又提供了真值证据：空间因子几乎完整保留，同时 RNA/ADT nuisance 也几乎完整进入 final embedding。

这更符合以下结构解释：

1. 等权 `mean_residual` 把弱或噪声模态固定写入融合结果；
2. final embedding 必须重构所有模态，因而没有动力丢弃 modality-private nuisance；
3. shared、modality-private、section/condition-specific 信息没有分开。

### 2.4 当前 candidate-sparse OT confidence 不是真正的 matchability

已有 OT 数组的直接统计表明：

| 数据集/方向 | mean top-10 coverage（candidate-sparse confidence） | `confidence < 0.2` | retained top-k target coverage | retained top-k target hub max/mean |
|---|---:|---:|---:|---:|
| CRC 003→006 | `0.180` | `66.2%` | `97.0%` | `26.8` |
| CRC 006→003 | `0.316` | `13.7%` | `99.8%` | `9.4` |
| SPATCH section1→2（posthoc） | `0.235` | `48.0%` | `92.4%` | `12.7` |
| SPATCH section2→1（posthoc） | `0.173` | `70.1%` | `89.7%` | `24.6` |
| Thymus 中间相邻 pair | `0.22–0.24` | `60–62%` | `81–83%` | `10.7–13.3` |
| Human Lymph Node | `0.42–0.43` | `6.6–8.1%` | 约 `100%` | `1.7–2.0` |
| Mouse Spleen | `0.58–0.59` | `<1%` | `100%` | `1.7` |
| MISAR | `0.39–0.45` | `2.7–17.1%` | 约 `100%` | `2.4–3.2` |

同时，保留的 top-10 权重普遍很散：

- CRC 归一化熵约 `0.945–0.971`；
- MISAR 约 `0.929–0.948`；
- Thymus 约 `0.952–0.975`；
- Simulation 约 `0.978`；
- SPATCH 约 `0.978–0.986`。

所以：

- coverage 低可能只是固定 top-10 截断了大量 transport mass；
- coverage 高也不代表 top-k 内存在清晰对应；
- candidate-sparse confidence 同时受 spot 数量、局部密度、候选 K 和两侧规模不平衡影响；
- high-entropy、hub-dominated 或根本不应匹配的 row 仍会进入 attention。

上述 coverage/hubness 只描述**被保留 support 的占用形态**；在没有真实对应关系的真实数据集上，即使 coverage 接近 100%、hubness 较低，也不能证明候选具有生物学语义。dense 路径使用的是另一种 confidence：`row_mass / (1 / n_source)` 截断到 1，同样不能直接解释为 matchability。

### 2.5 两组派生诊断的公式与复现口径

本节的 OT 数值不是训练日志中原生汇总的指标，而是从下列 seed42 主运行的 `ot_prior_topk/` 全量数组只读计算：

```text
results/crc_stereocite/fullspot_200ep_bidirectional_ot_attention_all_checkpoint_detailmem_lc0.1_seed42/
results/spatch/fullspot_200ep_gpu_seed42/
results/mouse_thymus/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/
results/human_lymph_node/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/
results/mouse_spleen/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/
results/misar_seq/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/
results/simulation/fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42/
```

对应 `*_metadata.json` 中，CRC、Thymus、Human Lymph Node、Spleen、MISAR 和 Simulation 标记为 `run_mode=training_final_eval`；SPATCH 是例外，标记为 `run_mode=posthoc_final_embedding_recompute_no_retraining`，所以 SPATCH 数值只诊断 final embedding 上重算的 candidate prior，不能写成训练时实际消费的 prior。七者均为 `candidate_source=final`、`ot_prior_mode=candidate_sparse`、`attention_topk=10`。候选后端、candidate K、FAISS 参数和方向语义以每个 pair 自己的 metadata 为准，不能从一个数据集外推。

对每个方向、全部 source rows（**不抽样**）：

```text
w_ik = topk_weight[i,k]                         # 保存值含 stabilizer，行和不严格等于 1
p_ik = w_ik / (Σ_valid-k w_ik + eps)            # 在有效 support 内再次严格归一化
H_i  = -Σ_valid-k p_ik log(p_ik) / log(K_valid) # 0 log 0 记为 0；K_valid<=1 时 H_i=0
mean normalized entropy = mean_i(H_i)

c_j = Σ_i,k 1[topk_idx[i,k] = j]
target coverage = count_j(c_j > 0) / n_target
target hub max/mean = max_j(c_j) / mean_j(c_j)   # 均值包含未命中 target
```

表中的 entropy、coverage、低 confidence 比例和 hubness 均先按方向计算；范围表示同一数据集各方向或相邻 pair 的最小值到最大值。它们依赖当前 `K=10`，因此以后比较 adaptive support 时必须同时报告有效 `K_i` 与 captured mass，不能只比较熵。

本文提到的 effective rank 使用 raw final embedding，而不是逐维标准化后的 embedding：

```text
Xc = X - column_mean(X)
lambda = eigenvalues(cov(Xc))
p_d = lambda_d / Σ_d lambda_d
effective_rank = exp(-Σ_d p_d log(p_d))
```

- CRC：依次拼接 `final_embeddings_CRC_003.npy`、`final_embeddings_CRC_006.npy`，从 612,374 rows 中用 `numpy.random.default_rng(42)` 无放回抽取 50,000 rows，得到约 `50.30/128`。
- Thymus：依次拼接 `final_embeddings_Mouse_Thymus1.npy` 至 `...4.npy`，使用全部 17,824 rows，得到 `12.01`，文中约写为 `12/128`。

这两个 rank 值是本次审查的探索性诊断，不是既有评估流水线的已保存产物；在把它们用于正式 go/no-go 之前，必须把同一公式、row index、seed、输入路径和输出 CSV 纳入评估脚本，并对 baseline/candidate 使用完全相同的抽样。

### 2.6 当前表示偏向粗粒度 compartment

- MouseBrain 9–11 类的粗组织标签很强，但更细 Y 标签明显弱；
- SPATCH 的 16 类 `cell_type_common` 最佳结果出现在 k=3，而真实 k=16 时 ARI 仅约 `0.100`；
- MISAR 13–16 类的 joint ARI 仍只有约 `0.26–0.31`。

这说明一个 128 维 final embedding 同时承担 cell-state clustering 和 tissue-domain segmentation 时，当前模型更容易选择“粗空间组织”这个解。后续空间模块必须显式保留 state 通道，不能只优化 Moran’s I 或空间连续性。

### 2.7 证据限制

- 主结果大多只有一个 seed；
- 多数运行只保存 final embedding，尚不能严格因果定位性能损失发生在 Fusion、GraphSAGE 还是 OT 后；
- CRC、Human Lymph Node、Spleen、Thymus 缺少可靠生物真值，内部指标和 section 指标只能联合解释；
- Thymus 早期 `SUMMARY.md` 的空间指标为 NA，本文采用后续统一分析结果；
- MISAR 的既有 joint 与 independent 聚类没有共享同一标签空间/聚类协议，两者差距只能生成待验证假设，不能证明 joint 表示或 section graph 导致性能下降；
- MouseBrain 的实际 L0 权重调度与其他数据集不同，必须保留其真实 run summary 作为独立回归基线。

---

## 3. 推荐的目标架构

### 3.1 总体数据流

```text
X(s,m)
  │
  └─ ModalityTrunk_m
       ├─ z_cv(s,m) = Normalize(Head_cv_m(h))
       │      ├─ 原 L0：函数、gamma、lambda/schedule、shape/L2 norm/spot order 不变
       │      └─ 仍进入 current FusionMLP；不是“只服务 L0”
       │
       ├─ z_fuse(s,m) = z_cv + fusion adapter
       ├─ z_private(s,m)
       └─ quality_features(s,m)
       │
       ├──────────── current full path ──────────── z_legacy_final
       │       current Fusion → GraphSAGE
       │       → current OT/LN[legacy_ot_prior]
       │
       └──────────── candidate architecture
               mask-aware quality fusion
                        │
               identity/high-frequency + local/context relation bank
                        │
                z_state_preOT       z_domain_preOT
                        \              /
                         AlignReadout
                              │
                  EMA pre-OT teacher + explicit section graph
                              │
             candidate_ot_prior：topology-aware partial/UOT
                         + matchability/support mask
                              │
                  z_state_out       z_domain_out
                  (首版旁路 OT)       (接收 OT message)
                         \              /
                          JointReadout
                              │
                       z_candidate_final

z_final = IdentityPreservingMerge(z_legacy_final, z_candidate_final)

decoder_m([z_final, z_private_m, condition/section token])
```

在 legacy path 正式退出前，`legacy_ot_prior` 与 `candidate_ot_prior` 必须是两套独立的 state/buffer：前者保留当前相邻链、top-10/confidence、refresh source 与 schedule，后者才允许使用 adaptive support、EMA pre-OT source、matchability 和新 section graph。O1/O2/S1 不得原地覆盖当前唯一的 `self.ot_prior` 后再声称 legacy branch 未变。

### 3.2 冻结 L0 时的关键接口

这里必须区分“冻结损失接口”和“冻结 learned representation”。

**本文冻结的是：**

- `compute_joint()` / `crossview_contrastive_Loss()` 的公式与 `gamma`；
- 每个数据集当前 `lambda_contrast` 及其 schedule；
- L0 输入的维度、L2 normalization、spot 顺序、真实观测模态配对与 mask 语义。

**本文不承诺冻结的是：**

- `Head_cv`、shared trunk 的参数或每个 epoch 的 `z_cv` 数值；
- reconstruction、fusion 或新增分支回传到 trunk/`z_cv` 的梯度。

当前 `z_cv` 本来就同时进入 L0 和 `FusionMLP`，重构梯度会经 fusion 回到 encoder。新增 private/quality/graph 分支也可能改变 shared trunk；若要让 L0 表征和梯度完全隔离，必须使用独立 trunk 或显式 stop-gradient，那是另一项更重的结构消融，不能由“冻结 L0”自动推出。

建议把当前 encoder 拆成共享 trunk 和多个输出接口：

```text
h_m       = Trunk_m(x_m)
z_cv_m    = L2Normalize(Head_cv_m(h_m))
z_fuse_m  = z_cv_m + Adapter_m(h_m)
p_m       = PrivateHead_m(h_m)
```

其中：

- `z_cv_m` 的维度、归一化、spot 顺序和 L0 调用逻辑保持当前行为，但其训练值可以随共享 trunk 改变；
- `Adapter_m` 的最后一层 zero-init，且不再叠加一个同时为 0 的 `alpha_m`；
- `z_fuse_m` 可以保留非归一化幅值或不确定性信息，不再要求所有下游模块只能看 L2-normalized latent；
- `p_m` 只供本模态 decoder 使用，不作为默认聚类 embedding；
- 对下游正式输出，优先使用 shared/state/domain head，而不是 `[shared, private]` 全部拼接。

### 3.3 保护当前基线的残差接入

基线锚点必须是**当前完整路径**，而不是只到 graph 的中间表示：

```text
z_legacy_final
  = CurrentOTandLayerNorm(
      CurrentGraphSAGE(
        CurrentFusionMLP({z_cv_m})
      )
    )

z_candidate_final
  = CandidateJointReadout(
      CandidateStatePath,
      CandidateDomainAndOTPath
    )

z_final
  = z_legacy_final
    + Delta_replace([z_legacy_final, z_candidate_final])
```

`Delta_replace` 的最后一层可以 zero-init；此时不再额外设置零门控。这样初始 `z_final` 与当前 full path 严格相同，current OT update 也不会被意外关掉。另一种合法做法是：

```text
z_final = (1 - beta) * z_legacy_final + beta * z_candidate_final
```

此时 `beta=0` 时 candidate branch 必须是非零随机初始化，不能再次把 candidate 最后一层置零。无论使用哪一种：

- `LayerNorm` 必须包含在 legacy/candidate 各自分支内部，不能在 zero residual 外再包一层新 norm，否则“零贡献”也不等于当前基线；
- row-level `keep-source` 必须 bypass candidate path 的 post-norm，而不是令 message 为 0 后仍执行 `LayerNorm(source)`；
- `legacy_ot_prior` 与 `candidate_ot_prior` 必须独立保存、刷新和序列化；candidate 模块关闭时，任何新 section edge/support 都不能进入 legacy attention；
- 只有 candidate branch 独立通过后，才允许逐步 cross-fade 并最终移除 legacy path；
- 每个新增模块只使用一种 identity 机制，避免双重零初始化。

“同一个 seed”本身不能保证 module-off 等价：如果先构造新模块，它会消耗 RNG 并改变后续 legacy 参数初始化。B0 必须采用以下协议之一：

1. 先按原顺序构造纯 legacy 模型并保存 `state_dict`、buffer、两侧 OT prior 与 RNG/config snapshot，再 attach 新模块并严格加载 legacy keys；
2. 或为每个新增模块使用独立 `torch.Generator`，保证 legacy 构造顺序和随机流完全不变。

如果现有主结果没有可依赖的 checkpoint，就先重跑 B0 并保存，而不是只凭 seed 假设权重相同。所有“最后一层 zero-init”都同时指 **weight 和 bias**。实现后在 `eval()`、同一输入和同一 `legacy_ot_prior` 下运行 module-off identity test，逐项断言下列量在预先规定的浮点容差内一致：

```text
z_cv
z_legacy_fusion
z_legacy_graph
z_legacy_final
all reconstructions
reconstruction loss / L0 / total loss
```

FP32 默认可先用 `atol=1e-6, rtol=1e-5`；BF16/混合精度应把两边输出转 FP32 后，根据 B0 的重复前向误差在实验前另行锁定容差，不能看到结果后再放宽。

---

## 4. P1-A：state/context 双通道、多尺度、关系感知空间图

### 4.1 为什么它是第一优先级

这是 8 个数据集里最一致的结构证据：

- 多个数据集的空间连续性稳定落后于 COSIE；
- 当前空间模块只有 fusion 后的一层固定 KNN GraphSAGE；
- 同一个 `k` 在不同空间分辨率下并不代表相同物理感受野；
- self-loop、`self_linear` 和 outer residual 重复传递自身信息；
- MouseBrain 和 SPATCH 又表明，单纯增强低通平滑会损伤细粒度 cell state。

### 4.2 建议模块

#### A. 先做 self-path 语义消融

当前同时存在 adjacency self-loop、`self_linear(x)` 和 outer `x + out` residual，说明 self 信息有三条通路，但这本身还不能证明是代码 bug。去掉 self-loop 还会重新归一化其余邻边，因此必须作为结构消融，而不是无条件“修复”：

```text
G0-current:
  self-loop + self_linear + outer residual

G0-a:
  neighbor aggregate 不含 self-loop
  + self_linear + outer residual

G0-b:
  self-loop aggregate
  + outer residual；去掉独立 self_linear
```

对三者记录 self/neighbor 消息范数、邻边重归一化后的实际权重和下游结果，再决定默认语义。与此同时，让 `num_layers`、`use_distance_weight`、`use_spatial_graph` 等配置真实控制执行路径。

#### B. 从单一邻接矩阵改为 relation bank

每个 section 至少保留以下关系：

```text
A_local       ：局部空间邻居，保护局部连续性
A_context     ：更大 k 或物理 radius，表达组织域/微环境
A_feature_m   ：每模态的 mutual-KNN 特征邻居，可选
A_boundary_m  ：通过 latent similarity / 梯度抑制跨边界传播，可选
```

不要把不同边直接混成一张图；应分别聚合，再由 relation gate 融合：

```text
h_local   = G_local(h, A_local)
h_context = G_context(h, A_context)
h_feature = G_feature(h, A_feature)

w_rel = softmax(RelationGate([h, h_local, h_context, h_feature]))
h_ctx = Σ_r w_rel,r * h_r
```

#### C. 显式保留低频与高频

建议将自身状态、平滑上下文和局部差异分开：

```text
h_low,k  = A_k h
h_high,k = h - A_k h

h_state  = h + scale_high * F_high({h_high,k})
h_domain = h + Σ_k scale_low,k * F_low,k(h_low,k)
```

其中：

- `h_state` 偏 cell state、边界与局部差异；
- `h_domain` 偏组织域和多尺度微环境；
- `scale_*` 使用 LayerScale/ReZero；若 scale 从 0 开始，`F_*` 必须非零随机初始化，不能两端同时为零；
- 只需 1–2 个可学习层，配合 Jumping Knowledge/step concatenation，比盲目堆深更容易控制过平滑。

上述“低/高频神经双通道”是针对本模型结果的**迁移设计**，不是下述论文中某一篇的原样结构。

在统一目标流中，这两个输出位于 candidate OT 之前：`h_state → z_state_preOT`，`h_domain → z_domain_preOT`。**G2 单模块、O1-off 时，两条 candidate 分支都旁路 candidate OT**，即 `z_state_out=z_state_preOT`、`z_domain_out=z_domain_preOT`；legacy branch 仍单独使用当前 OT。只有打开 O1 后，才让高 matchability 的 `candidate_ot_prior` message 更新 domain，state 继续旁路。`JointReadout([z_state_out, z_domain_out])` 再进入 shared decoder，使两个 head 都通过重构获得梯度。是否也让 OT 更新 state 必须作为后续独立消融。

#### D. 让感受野具有密度可比性

不能在 Visium、Stereo-seq 和 SPATCH 上只复用固定 `k`。建议记录每个 section 的局部尺度：

```text
rho_s = median(distance to kth local neighbor)
```

再用：

- 小尺度：`1 × rho_s`；
- 中尺度：`2–3 × rho_s`；
- 或者各平台分别选择能覆盖相近物理半径的 k。

跨技术数据可像 SLAT 那样根据空间密度使用不同邻居数，但选择规则必须由坐标密度决定，不能用标签调参。

### 4.3 渐进实现顺序

1. **G0：self-path 语义消融与配置真实性。**
2. **G1：只在 fusion 后加 local + context + identity 三分支。**  
   这是对 CRC/SPATCH 最低风险、最省显存的版本。
3. **G2：增加 high-frequency / gradient branch 和 state/domain 输出。**
4. **G3：在每个模态 fusion 前加入共享 spatial graph encoder。**
5. **G4：只对确有收益的模态加入 `A_feature_m`。**  
   百万 spot 上应使用 mutual-KNN、稀疏缓存和 edge chunking。

### 4.4 预期受益与风险

预期首先改善：

- CRC、SPATCH、Simulation 的多尺度空间连续性；
- Human Lymph Node、Spleen 的局部组织几何；
- MouseBrain 的高 k 稳定性；
- 通过 `h_state` 保护 MouseBrain/SPATCH 的细粒度信息。

主要风险：

- context 分支过强导致边界与稀有状态被抹平；
- feature graph 把技术相似性当成生物邻居；
- 每模态多图使 SPATCH 显存和 I/O 成本上升；
- learned edge gate 可能把初始聚类错误自我强化。

控制措施：

- identity/high-frequency 永远保留；
- feature graph 先做 mutual-KNN，再由小幅 residual 接入；
- 不使用真实标签剪边；
- 记录各 relation 权重、边跨标签率（仅评估时）、每层 effective rank 和空间频谱变化。

### 4.5 学习来源与迁移边界

- [SpatialGlue, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02316-4)：论文真实机制是每模态分别建立 spatial-KNN 和 feature-KNN，两路 GNN 后做模态内图注意力，再做 spot-wise 模态间注意力。本文借鉴“模态内双图与自适应关系权重”；它没有解决多 section 或本文提出的高/低频双通道。
- [MENDER, Nature Communications 2024](https://www.nature.com/articles/s41467-023-44367-9)：真实机制是在多个空间范围统计 cell-state composition 并拼接成 context profile。本文把它迁移为可微的多尺度 context/soft-prototype 分支；MENDER 本身不是端到端多组学 GNN。
- [BANKSY, Nature Genetics 2024](https://www.nature.com/articles/s41588-024-01664-3)：真实机制是联合自身表达、邻域均值和局部空间梯度，并用 mixing parameter 控制 cell typing 与 domain segmentation 的空间强度。本文据此保留 identity、context 和 gradient/high-frequency；神经双 head 是本文推断，不是 BANKSY 原结构。
- [SLAT, Nature Communications 2023](https://www.nature.com/articles/s41467-023-43105-5)：真实机制是拼接 `X, AX, A²X, ...` 表达不同空间尺度，并按技术密度调整邻居数。本文借鉴多跳表示与密度可比感受野；不迁入其 adversarial loss。
- [STAGATE, Nature Communications 2022](https://www.nature.com/articles/s41467-022-29439-6)：真实机制是在空间图上学习 edge attention，并可选按预聚类修剪边界。本文只借鉴可学习边权；不建议第一版使用预聚类硬剪边。
- [SMART, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70821-5)：真实机制是每模态使用 GraphSAGE encoder/decoder 并保持可扩展性。本文借鉴模态化图编码与采样聚合；不迁入 triplet objective。

### 4.6 验收条件

至少同时满足：

- CRC/SPATCH/Simulation 的 joint 空间连续性在多个 k 上改善，而不是只改善单一 k；
- MouseBrain RegionLoupe、annotations、celltype ARI 的绝对下降均不超过约 `0.01`；
- MouseBrain 细粒度 Y、SPATCH k=16 cell type 不进一步恶化；
- SPATCH 保持 full-spot 可运行，峰值显存和 wall time 有明确记录；
- effective rank 不因图分支明显下降；
- state head 与 domain head 的收益方向符合预期，而非两者输出几乎相同。

---

## 5. P1-B：双头 encoder + 质量感知多模态融合

### 5.1 当前 FusionMLP 的关键限制

当前所有模态输出先 L2 normalize，然后：

```text
concat → MLP → + equal mean → LayerNorm
```

这有两个问题：

1. L2 normalization 抹掉了可用于质量判断的幅值信息；
2. `mean_residual` 给每个模态固定相同的基础贡献，spot-level 弱模态无法真正被拒绝。

CRC、Human Lymph Node、Spleen、Thymus 的 MOFA 方差解释也提示，不同模态的有效贡献并不相等；SPATCH 中 HE、RNA、Protein 的局部质量更不可能在每个 spot 上恒定相同。

### 5.2 建议模块

#### A. 把 L0 head 与 fusion head 解耦

```text
h_m      = Trunk_m(x_m)
z_cv_m   = Normalize(Head_cv_m(h_m))       # 同时进入 L0 与 legacy fusion
z_fuse_m = z_cv_m + Adapter_m(h_m)         # candidate fusion 可见质量/幅值信息
q_m      = QualityHead_m(h_m, local_context_m)
```

要求：

- `Head_cv_m` 保持现有 L0 的 shape、L2 norm、spot order 和调用方式，但不声称参数/梯度不变；
- `Adapter_m` 小参数，只将最后一层 zero-init，不再叠加一个零 `alpha_m`；
- `QualityHead` 不接收真实标签。

#### B. 用“稳定质量先验 + 小型 learned residual”代替纯 attention

纯 learned attention 在噪声下可能不稳定。建议先构造非参数或 stop-gradient 的局部效用先验 `r_m`：

```text
r_m 可包含：
- 同模态邻域保持度
- 跨模态邻域可预测度
- decoder reconstruction residual 的 EMA
- MC-dropout / ensemble uncertainty（可选）
- 输入 availability/QC mask
```

再只学习一个小的校正量：

```text
logit_m = log(r_m + eps) + delta_m
w_m = softmax(logit_m)
```

其中 `delta_m` 最后一层可 zero-init。若第一版令 `r_m=1`，质量权重只是在 candidate path 中等权；identity 由下述 fusion replacement 的最后一层负责，不能再让所有 candidate feature 与门控同时为零。

#### C. 区分“校正 legacy fusion”和“真正拒绝弱模态”

最低风险的 F1 只能做 legacy 校正：

```text
z_legacy = FusionMLP_current({z_cv_m})
z_quality = Σ_m w_m * z_fuse_m

z_fused
  = z_legacy
    + Delta_fuse([z_legacy, z_quality, quality_features, observed_mask])
```

`Delta_fuse` 最后一层 zero-init，且其外部不再增加新的 LayerNorm；此时严格回到当前 fusion。这个版本仍然保留了 legacy concat-MLP 和 equal-mean 内所有模态的贡献，因此：

- 它只能被称为“质量感知校正”，不能声称真正拒绝了弱模态；
- `w_m` 不能解释为某模态对最终 embedding 的总贡献。

若 F1 有稳定收益，再做 F2：

```text
z_gated = FullyGatedFusion({z_fuse_m}, observed_mask)
z_fused = IdentityPreservingMerge(z_legacy, z_gated)
```

只有逐步 cross-fade 到不再经过 legacy concat/mean 的 `z_gated`，才可能真正把某个弱模态的贡献降到零。删除当前 `mean_residual` 或 legacy path 必须是单独消融，不能与 gate 首次引入同时发生。

#### D. 防止 gate 坍塌

- modality dropout：只在 fusion/decoder 路径以小概率把某模态替换为 zero/mask token，并传入 `observed_mask`；不能删除 modality 字典 key，因为当前 `FusionMLP` 会直接 `KeyError`。L0 仍在全部真实观测的 `z_cv` 模态对上按当前逻辑计算；
- minimum weight floor 或温度退火；
- 监控 gate entropy、按模态平均权重、按空间域/section 的权重分布；
- gate regularization 只需弱约束，不能强迫所有 spot 等权；
- 对 HE/Protein/ATAC 等模态分别校准 quality feature 的尺度。

### 5.3 首选验证

- **Human Lymph Node / Spleen**：candidate-sparse OT retained support 的 coverage/hubness 未见明显异常，适合优先隔离 Fusion；这不代表语义匹配已经被验证；
- **Simulation**：看 nuisance leakage 是否下降；
- **SPATCH**：验证三模态局部质量和 full-spot 可扩展性；
- **MouseBrain**：验证三模态优势是否被保留。

### 5.4 学习来源与迁移边界

- [SpatialGlue, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02316-4)：真实机制是 spot-wise between-modality attention。本文借鉴 spot-level 模态权重，但不把 attention 权重直接解释为校准后的 QC 或因果贡献。
- [COSMOS, Nature Communications 2025](https://www.nature.com/articles/s41467-024-55204-y)：真实机制是两个独立 spatial GCN 后，用 WNN 的局部模态效用权重融合；作者也讨论了非参数 WNN 相对复杂 attention 的稳定性。本文据此采用“稳定先验 + learned residual”；不迁入 COSMOS 的 DGI loss。
- [SpaMV, Nature Communications 2026 early/unedited online version](https://www.nature.com/articles/s41467-026-74718-1)：真实机制强调 shared/private 表示和 mixture-of-experts shared fusion。本文借鉴 shared/private 与专家融合思路；该版本于 2026-07-01 online，页面明确标注尚未完成最终编辑，因此只作为最新补充证据。
- [MultiVI, Nature Methods 2023](https://www.nature.com/articles/s41592-023-01909-9)：真实机制使用模态特异 posterior，并可从后验采样得到不确定性。本文只迁移“让 gate 看见不确定性”的原则；不在第一版引入完整 VAE。

### 5.5 验收条件

- Human Lymph Node / Spleen 的 ASW 或 DBI 改善，同时 bLISI、kBET 不出现大幅退化；
- Simulation 的 RNA/ADT nuisance leakage 明显下降，spatial factor recovery 的绝对下降不超过约 `0.02`；
- gate 不长期坍塌到单一模态；
- gate 权重在不同组织区域有可重复结构，但不以“权重高”直接宣称生物因果；
- MouseBrain 三组粗标签 ARI 基本保持；
- SPATCH 仍能 full-spot 训练和推理。

---

## 6. P1-C：shared / modality-private / section-condition 分解与条件 decoder

### 6.1 为什么不能只做 gate

质量 gate 只能决定“当前 spot 更信任哪个模态”，却不能解决：

- shared embedding 为重构所有模态而保存 modality-private nuisance；
- 技术 section 与真实发育/疾病 condition 混在同一个方向；
- final embedding 既要 batch-invariant，又要重构被 batch/section 影响的输入。

Simulation 的 nuisance leakage 是最直接的真值证据；MISAR 和 Thymus 则表明 section 轴不能简单等同于纯技术 batch。

### 6.2 建议四类 latent

```text
z_shared       ：跨模态、跨可比 section 的共同生物状态
p_m            ：模态 m 的私有信息
c_condition    ：已知发育阶段/疾病条件相关生物信息（若有）
u_section      ：技术 section/batch 信息
```

Decoder 改为：

```text
x_hat_m
  = Decoder_m(
      z_final,
      p_m,
      c_condition,
      u_section or section token
    )
```

默认下游聚类输出只使用：

- 通用任务使用 `z_final`；
- cell-state / spatial-domain 专项任务分别评估 `z_state_out` / `z_domain_out`，有明确生物 condition 时可同时报告 `[z_final, c_condition]`；
- 不默认拼接 `p_m` 和 `u_section`。

### 6.3 最低风险版本

第一版不必上完整 VAE 或 adversarial training，可先使用确定性结构：

```text
p_m = small_private_head_m(h_m), dim 16–32
e_s = learned low-rank section token

x_hat_m
  = Decoder_shared_m(z_final)
  + Decoder_private_residual_m([p_m, e_s])
```

其中只把 `Decoder_private_residual_m` 的最后一层 zero-init，不再同时使用零 `alpha_private`。它允许 private 分支解释重构中的模态或 section 残差，从而减少 shared code 保存 nuisance 的压力。

之后才逐步增加：

- shared/private 低相关约束；
- condition classifier 只作用 `c_condition`；
- bounded section nuisance head 只作用 shared/joint 路径，不作用 `c_condition`；
- observed-only reconstruction 和 modality dropout。

### 6.4 section 不是天然的 nuisance

必须区分：

| 场景 | 建议 |
|---|---|
| 技术重复切片 | 可让 `u_section` 和 bounded nuisance head 吸收技术差异 |
| MISAR E11.0–E18.5 发育阶段 | stage 是生物变量，应进入 `c_condition`，不能无条件 adversarial removal |
| 肿瘤不同区域/患者 | section 可能包含真实克隆和微环境差异，应允许 condition-private |
| 同组织相邻连续切片 | 可共享大部分结构，但仍要允许 slice-unique population |

### 6.5 decoder 结构

当前输入是 PCA/Harmony 特征，第一版仍可保持原目标，不需要同时修改预处理：

```text
shared decoder branch  ：重构各模态共同可预测部分
private decoder branch ：重构本模态残差
section branch         ：解释低秩 section 偏移
```

如果以后改为原始/近原始特征，再考虑：

- RNA：NB/ZINB 或 count-aware likelihood；
- ATAC：Bernoulli/binomial；
- Protein：带 background 的 mixture；
- HE / Metabolite：Gaussian、Huber 或 uncertainty-weighted regression。

这属于后续大改，不应与第一轮结构消融同时进行。

### 6.6 学习来源与迁移边界

- [MIDAS, Nature Biotechnology 2024](https://www.nature.com/articles/s41587-023-02040-y)：真实机制把 biological state 与 technical noise 拆为两个 modality-agnostic latent，并让模块化 decoder 使用它们生成观测。本文借鉴生物/技术分离；MIDAS 是非空间单细胞方法。
- [scDisInFact, Nature Communications 2024](https://www.nature.com/articles/s41467-024-45227-w)：真实机制区分 shared biological、按 condition type 划分的 unshared biological，以及 one-hot technical batch。本文据此区分 `c_condition` 与 `u_section`；它不是空间多组学方法。
- [SpaMV, Nature Communications 2026 early/unedited online version](https://www.nature.com/articles/s41467-026-74718-1)：真实机制为每模态 shared/private encoder，并使用 shared+private 重构。本文借鉴模态私有 decoder 路径；多 section/HE/代谢组的适用性仍需本模型实验验证。
- [INSPIRE, Nature Genetics 2026](https://www.nature.com/articles/s41588-026-02579-x)：真实机制包括 section-specific reconstruction、section/gene effect，以及对 section-unique population 减弱强制对齐的 bounded discriminator。本文借鉴条件 decoder 和“只对可能共享群体去 nuisance”；INSPIRE 是多 section 转录组，不是多组学。
- [MultiVI, Nature Methods 2023](https://www.nature.com/articles/s41592-023-01909-9)：真实机制按 RNA、ATAC、Protein 的观测分布建立模态化生成模型，并只对已观测模态计算 likelihood。本文借鉴 modality-specific/observed-only decoder 原则；当前 PCA/Harmony 输入下不应直接套用其原始 likelihood。

上述“三类或四类 latent 的组合”是对这些方法和本项目结果的综合迁移，不是任何单篇论文的原样复制。

### 6.7 验收条件

- Simulation nuisance leakage 下降且 spatial factor recovery 保持；
- Thymus shared embedding 的 section 主轴减弱，同时把 effective rank 作为监控项而非单独成功条件；不能只用 section mixing 判断成功；
- MISAR 在同一 embedding、同一 K/label space 和同一聚类协议下，RNA_Clusters 等绝对指标不明显下降，发育阶段轨迹或 condition-private 分支仍可解释；
- Human Lymph Node/Spleen 的内部几何改善；
- `p_m` 不应单独比 `z_shared` 更能恢复所有生物标签，否则 private 分支可能吞掉共享生物信息；
- `u_section` 应预测 section，但不应成为默认下游 embedding。

---

## 7. P1-D：topology-aware、可拒配、真实 matchability 的 UOT attention

### 7.1 先明确：当前缺的不是“再换一种 OT 名称”

当前模型已经使用 unbalanced OT，边际可以松弛。真正的问题是：

1. UOT 的 transported mass 没有被解释为“这个 spot 是否可匹配”；
2. candidate-sparse 路径把固定 top-10 coverage 命名为 confidence；dense 路径则把相对均匀源边际的 row mass 命名为 confidence，两者都没有校准；
3. high-entropy、hub-dominated、非 reciprocal 的候选仍会进入 attention；
4. cost 主要来自 embedding cosine，没有显式保护局部空间拓扑；
5. prior 每 20 epoch 从含旧 OT message 的 final embedding 刷新，形成：

```text
old OT → final embedding → new OT prior → next final embedding
```

错误匹配可能自我强化。

### 7.2 自适应 support

固定 `topk=10` 应改为按累计质量选择：

```text
K_i
  = 达到候选 row transport mass 的 q 比例所需的最小 K

K_min ≤ K_i ≤ K_max
```

推荐输出 ragged sparse support，或在固定张量中同时保存有效 mask。这样：

- 大规模/弥散 row 不会因固定 10 个候选丢掉 70–80% 的质量；
- 集中的 row 不需要人为塞满 10 个近似等权邻居；
- top-k coverage 不再被误用为匹配置信度。

这项改动必须贯穿 attention 接口，而不只是多保存一个 mask。当前 `OTGuidedAttention.forward/compute_update_only` 接收固定 `[N,K]`，并对 `log(topk_weight.clamp_min(delta))` 做 softmax；padding 的零权重会因 clamp 获得非零注意力。未来接口必须新增 `support_mask`：

```text
has_support = support_mask.any(dim=1)
assert all(0 <= topk_idx[support_mask] < n_target)
safe_idx = where(support_mask, topk_idx, 0)       # gather 前替换为合法 in-range index

只对 has_support rows：
    target = target_h[safe_idx]
    logits = attention_logits + log(clamp(weight, min=delta))
    logits[~support_mask] = -inf
    attention = softmax(logits)

对 ~has_support rows：
    直接 keep-source；不进入 gather/softmax/post-norm
```

不能让 padded `topk_idx=-1` 静默索引最后一个 target，也不能让全 `-inf` softmax 先产生 NaN 后再 `where`。如果 `n_target=0`，整条 pair 必须在 attention 前跳过。聚合、entropy、hubness 和 unmatched 统计都只使用有效 support；`K_valid=0/1` 时 entropy 定义为 0，`K_valid<2` 时 top-margin 记为 NA。

`reciprocal_score` 与 `cycle_score` 只有在同一 pair 的双向 prior/support 都存在时才定义；缺少反向 support 时必须记为 NA，而不是默认 0。

### 7.3 将 coverage 拆成可解释诊断

每个 source spot 至少单独保存：

```text
row_mass_ratio     ：实际运输质量 / 期望或稳健参考质量
support_coverage   ：保留 support 捕获的 row mass 比例
normalized_entropy ：support 内分布是否集中
top_margin         ：top1 与 top2 的区分度
reciprocal_score   ：target 是否也把 source 视作高优先级
cycle_score        ：s→t→s 是否回到局部邻域
hubness_penalty    ：target 是否被过多 source 共同命中
stability_score    ：相邻两次 teacher refresh 是否稳定
```

不要再把这些量压成一个未经校准的名字。用于 attention 的 `matchability_i` 可以是：

```text
matchability_i
  = Calibrator(
      row_mass_ratio,
      1 - normalized_entropy,
      top_margin,
      reciprocal_score,
      cycle_score,
      hubness_penalty,
      stability_score
    )
```

第一版 `Calibrator` 应是单调、低参数或固定规则；后续如改成 learned calibration，也只能使用无标签诊断特征。

所有可能为 NA 的输入都必须同时带 `validity_mask`。固定规则只在有效项上归一化加权；learned calibrator 则同时接收数值和 missing indicator。不能把 NA 直接喂入数值层，也不能把填充值 0 当作“最低质量”的真实证据。第一次 refresh 的 `stability_score` 同样是 NA。

### 7.4 显式 unmatched / keep-source

当前 UOT 的边际松弛没有自动保证 attention 会跳过不可匹配 row。建议让“未匹配”进入神经模块：

```text
if matchability_i < threshold:
    z_i_candidate = z_i_source                 # bypass post-LN
else:
    z_i_candidate = CandidateNorm(
        z_i_source
        + matchability_i * gate_i * OT_message_i
    )
```

可以采用：

- dustbin/unmatched target；
- 基于 row mass 的 matchability；
- hard keep-source；
- 或连续的 residual gate。

对于真实 section-unique population，keep-source 不是失败，而是正确行为。

这里的逐 row `where(unmatched, source, transformed)` 必须发生在 candidate post-norm 之后；仅把 `OT_message=0` 再调用现有 `apply_update` 仍会得到 `LayerNorm(source)`，并不等于 keep-source。整个 candidate OT path 再通过第 3.3 节的 `IdentityPreservingMerge` 接入 `z_legacy_final`，才能在模块初始时保留当前完整 OT baseline。

### 7.5 在 cost 中加入局部拓扑，但避免全量 FGW

PASTE/FGW 的核心经验是：跨切片点相似还不够，还要保护各切片内部结构。对本模型可采用 candidate-limited 近似：

```text
C(i,j)
  = a * cosine_cost(z_preOT_i, z_preOT_j)
  + b * distance(context_signature_i, context_signature_j)
  + c * local_degree_or_density_mismatch
```

`context_signature` 可来自：

- P1-A 的多尺度 state composition；
- local/medium graph 的低维统计；
- prototype membership；
- 相对距离分位数或局部谱特征。

这不是完整 FGW，但能在不构造 `N²` 几何矩阵时把拓扑信息带入候选 cost。

### 7.6 先切断 candidate branch 的 final→prior 自反馈

建议使用 EMA teacher 的 **pre-OT embedding**：

```text
z_teacher_preOT
  = EMA(
      shared fusion
      + multiscale graph
      but before OT attention
    )
```

并采用：

- candidate teacher 只刷新 `candidate_ot_prior`，绝不覆盖 `legacy_ot_prior`；
- coupling momentum，而不是每次硬替换；
- refresh 前后记录 candidate overlap；
- 只有当 teacher 表示和 pair diagnostics 稳定时才刷新；
- OT message branch 使用小的 LayerScale；
- `W_O` 最后一层 zero-init，或使用从 0 开始的 ReZero gate；二者只选一种。

过渡期的通用 `z_final` 仍包含 `z_legacy_final`，因此整个模型仍保留 legacy final→prior 自反馈；只有 cross-fade 完成、legacy path/prior 正式退出后，才能宣称全模型已切断该反馈。

### 7.7 学习来源与迁移边界

- [PASTE, Nature Methods 2022](https://www.nature.com/articles/s41592-022-01459-6)：真实机制是 fused Gromov–Wasserstein 同时考虑跨切片表达不匹配和各切片内部空间距离结构。本文借鉴“cost 应包含内部拓扑”；candidate-limited signature 只是本项目的可扩展近似。
- [PASTE2, Genome Research 2023](https://genome.cshlp.org/content/33/7/1124)：真实机制是 partial FGW，只运输估计的重叠组织质量，允许局部重叠和 slice-specific cell types。本文把 partial transport 迁移为 attention 中的 unmatched/keep-source；PASTE2 没有本文的 entropy、reciprocal、cycle gate。
- [moscot, Nature 2025](https://www.nature.com/articles/s41586-024-08453-2)：真实机制统一 W/GW/FGW，支持 multimodal cost、unbalanced formulation、GPU 和 low-rank solver。本文借鉴结构成本与可扩展求解；moscot 是 solver/framework，不是 OT neural attention。
- [SLAT, Nature Communications 2023](https://www.nature.com/articles/s41467-023-43105-5)：真实机制用 adaptive clipping 和高置信 anchor 避免把异质区域强行对齐。本文借鉴“只让可靠群体传递跨切片信息”；并不迁入其 Wasserstein adversarial loss。
- [INSPIRE, Nature Genetics 2026](https://www.nature.com/articles/s41588-026-02579-x)：真实机制通过限制 discriminator logit，使 section-unique population 不再承受强制对齐梯度。本文借鉴“明确保留 unique population”的原则，而不是复制其 adversarial 模块。

### 7.8 验收条件

- Simulation 同网格 Top-1、Recall@5、FOSCTTM 明显改善；
- CRC/SPATCH 的 hub max/mean 和 entropy 降低，且不能仅通过缩小 support 假性改善；
- Thymus 中间 pair 的低质量 row 能被拒绝，不再稀释其他更新；
- Human Lymph Node/Spleen 不因继续增强对齐而损失已有 batch mixing 或内部几何；
- 输出并保存所有 matchability 分量及其 validity masks，而不是只保存单一 confidence；
- 对每个 section pair 报告保留质量、unmatched 比例和方向不对称性。

---

## 8. P1-E：显式 section graph 与 pair-quality aggregation

### 8.1 当前相邻链的隐含假设

当前 UOT 基于：

```text
zip(section_order[:-1], section_order[1:])
```

这等于默认：

- section 列表顺序就是生物关系；
- 只有相邻 section 值得连接；
- 中间 section 收到的所有 pair 更新同等可靠。

这些假设对发育阶段、不同患者、技术重复或空间距离不等的切片并不成立。

### 8.2 建议显式建模 section-level graph

```text
section node:
  metadata:
    tissue / subject / stage / technology

section edge:
  relation_type:
    technical_replicate
    adjacent_slice
    developmental_neighbor
    shared_population_similarity
```

边的建立优先级：

1. 已知 metadata；
2. 共享模态或 `z_shared_preOT` 上的双向 MNN/分布相似性；
3. 允许非相邻 edge；
4. 自动 edge 必须输出可审计的理由和 pair QC。

当前 `initialize_from_feature_dict` 要求所有 section 的 modality key 集合完全相同，因此 `bridge_modality` 不能作为当前 P1-E relation。只有完成 observed-mask、初始化、fusion、decoder 与 reconstruction 的全链路 missing-modality 改造后，才能在长期 mosaic 路线中加入 bridge-modality edge。

### 8.3 pair-quality 加权

每条 section edge 计算：

```text
q(s,t)
  = PairQuality(
      median matchability,
      reciprocal rate,
      unmatched fraction,
      entropy,
      hubness,
      distribution overlap,
      metadata compatibility
    )
```

`PairQuality` 与 spot-level calibrator 使用同一缺失值合同：每个分量都有 validity mask，只在有效项上归一化；首次 refresh 的 stability、缺反向 support 的 reciprocal/cycle 不参与该次数值聚合，并显式记录 missing indicator。若某条 edge 的必要分量全部缺失，则不自动赋予中性质量，而是停用该 edge 或要求人工 metadata 决策。

中间 section 的更新改为：

```text
Delta_s
  = Σ_t q(s,t) * Delta_(s<-t)
    / (Σ_t q(s,t) + eps)
```

而不是：

```text
mean(all incoming updates)
```

这避免一个低质量 pair 既贡献错误消息，又通过“除以邻居数”把高质量消息同步减半。

### 8.4 relation-specific adapter

不建议每个 section pair 单独训练完整 attention。可以使用：

```text
shared OT attention
+ low-rank relation adapter(relation_type)
+ pair-quality scalar
```

例如：

- technical replicate：更强调 shared biological alignment；
- developmental neighbor：允许有方向的 stage transition；
- distant but similar section：只传递高置信 shared population；

### 8.5 数据集专项

- **Thymus**：首要检查中间 section pair 的低 coverage/high hub 是否被自动降权；
- **MISAR**：不要只连 E11→E13→E15→E18；可以保留发育相邻方向，同时加入由共享群体支持的非相邻 reference edge；
- **MouseBrain**：3 个 section 当前表现好，应作为 section graph 的保守回归；
- **CRC/SPATCH**：只有两个 section，section graph 本身收益有限，重点转到 pair quality 和层次 OT。

### 8.6 学习来源与迁移边界

- [SpaMosaic, Nature Genetics 2026](https://www.nature.com/articles/s41588-026-02573-3) 及其[官方代码](https://github.com/JinmiaoChenLab/SpaMosaic)：真实机制包括切片内空间图、共享模态上的跨 batch MNN、bridge batches 和模态化重构，可连接大量 mosaic sections。本文只借鉴稀疏跨 section candidate graph 和 bridge connectivity；不迁入其 contrastive objective。其 batch–modality overlap graph 需要连通，MNN 也可能过度对齐组成不同的组织。
- [SMART/SMART-MS, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70821-5)：真实机制枚举 section pair，并在 Harmony feature space 取跨 section MNN 正样本。需要特别澄清：SMART-MS 的空间邻接是各 section 邻接的 block diagonal，并没有直接建立跨切片空间边；跨 section 作用来自 metric-learning triplets。本文只借其 all-pair MNN candidate 思路，不迁入 triplet loss。
- [MEFISTO, Nature Methods 2022](https://www.nature.com/articles/s41592-021-01343-9)：真实机制是在多组相关数据中识别空间或时间上的平滑/非平滑因子，并可用 DTW 对多 group 的一维时间轨迹做非线性对齐；它不支持跨数据集空间切片坐标对齐。本文只借鉴“sample relation 与平滑结构应显式建模”；section graph 是本文的工程推断，不能以 MEFISTO 作为空间配准依据。
- [INSPIRE, Nature Genetics 2026](https://www.nature.com/articles/s41588-026-02579-x)：真实机制使用相邻 section discriminator、section-specific effects 和 shared spatial factors。本文借鉴 shared/section-specific 分工；不建议在 MISAR 等发育数据上无条件删除 section 差异。

### 8.7 验收条件

- section graph 和所有 edge/pair QC 可导出并人工检查；
- Thymus section ARI/NMI、kBET/PCR 共同改善，但同时检查空间结构和 latent rank；
- MISAR 必须在同一 embedding、同一 K/label space、同一 KMeans seed/n_init 下比较结构改动前后的 joint 绝对指标；既有 joint/independent 差距只作为待验证假设，不作为验收目标；
- 错误调整 section 列表顺序不再改变图的生物语义；
- 去掉任意低质量 edge 后结果应稳定，高质量 edge 的作用可单独复现。

---

## 9. P2-F：CRC/SPATCH 的 low-rank 与 prototype-to-spot 层次 OT

### 9.1 为什么需要专门的大规模路径

CRC 有约 61 万 spots，SPATCH 超过 106 万 spots。当前 spot-to-spot candidate UOT 虽然能运行，但出现：

- retained support 高熵；
- target hubness 明显；
- 两个方向的 coverage 强烈不对称；
- 全部跨 section 关系都由局部 cosine candidate 承担。

其中 CRC 保存的是 training-final-eval prior，SPATCH 是 final embedding 上 posthoc 重算的 candidate prior；后者能说明候选几何和 full-scale 资源风险，但不能反推 SPATCH 训练过程中实际 attention 已消费了同一 prior。

只扩大 candidate K 会进一步增加显存和计算，并不能自动提高语义匹配质量。

### 9.2 两级方案

#### 方案 F1：low-rank UOT

优先尝试：

```text
P ≈ Q Rᵀ
```

或其他低秩 coupling/cost factorization，使内存从接近二次关系降为与 `N × rank` 相关。rank 不应只按运行速度选择，还要检查：

- rare-state recall；
- topological consistency；
- rank 增大时结果是否稳定。

低秩 coupling 不能直接塞进当前要求显式 `topk_idx/topk_weight/confidence` 的 OT attention。实现时必须选择一个闭环出口：

```text
出口 A：factorized barycentric/message aggregation

P = Q R^T，Q/R 非负且满足所选边际约束
message = Q (R^T V) / [Q (R^T 1) + eps]
row_mass = Q (R^T 1)
```

该路径不物化 `P`，直接把 factorized message 和 row-mass matchability 送入新的 residual readout，才保留低秩内存收益。

出口 A 没有显式 spot-to-spot support，因此不能假装拥有完整的 top-k coverage、top-margin、reciprocal/cycle 或 top-k hubness。它的诊断合同改为：

```text
row/column mass 与目标边际偏差
unmatched / transported mass
Q/R factor concentration 与 effective factor rank
column-load concentration
message norm 与 refresh stability
固定小样本上的 exact blockwise support audit
```

```text
出口 B1：exact blockwise top-k，仅用于小子集/ANN 召回基准

按 block 计算 q_i^T r_j
→ 提取 sparse top-k + support_mask
→ 复用第 7.2 节的 masked attention
```

exact blockwise 只降低峰值内存，计算仍为 `O(N_source × N_target × rank)`，不能作为 CRC/SPATCH 百万点 full-scale 路径。

```text
出口 B2：full-scale ANN/MIPS 或 candidate-restricted scoring

在 Q/R factor space 建 ANN/MIPS index
或只在独立生成的稀疏 candidate set 上计算 q_i^T r_j
→ sparse top-k + support_mask
→ masked attention
```

出口 B2 可以继续计算 support-based reciprocal/cycle/hubness，但必须在固定 exact-B1 小子集上报告 ANN top-k recall，并记录 candidate recall、临时内存和 wall time。不能先物化完整 `P` 再 top-k。F1 首轮实验应固定 A 或 B2 其中一个出口，不把两种接口混在同一消融中。

#### 方案 F2：prototype → local spot refinement

如果低秩仍有明显 hub 或语义混合，再使用：

```text
每个 section：
  spot multiscale graph
      → overcomplete prototypes

跨 section：
  prototype-level partial / unbalanced OT

局部：
  只在匹配 prototype 对内部构建 spot candidate
      → matchability-gated attention
```

prototype 应：

- overcomplete，而不是直接等于预期真实类别数；
- 允许一个 prototype 与多个近邻 prototype 对应；
- 有 rare residual path 和 minimum occupancy；
- 保留 dustbin/unmatched prototype；
- 使用 state 与 domain 两种 prototype bank，避免只匹配粗组织域。

### 9.3 这一步的推断性质

“prototype-to-spot OT”不是下列论文的原样模块，而是结合：

- moscot 的 low-rank/scalable OT；
- PASTE/PASTE2 的结构与 partial alignment；
- 本项目 CRC/SPATCH 的 candidate hubness；
- BANKSY/SLAT 的多尺度空间表示

得到的工程推断。

### 9.4 学习来源与迁移边界

- [moscot, Nature 2025](https://www.nature.com/articles/s41586-024-08453-2)：真实机制用 GPU、on-the-fly cost、entropic 和 low-rank formulation 扩展 OT，并展示百万级细胞应用。本文直接借鉴 low-rank 路线；prototype refinement 是额外推断。
- [PASTE, Nature Methods 2022](https://www.nature.com/articles/s41592-022-01459-6) 与 [PASTE2, Genome Research 2023](https://genome.cshlp.org/content/33/7/1124)：真实机制分别提供结构保持和 partial overlap。本文借鉴 prototype 级结构 cost 与 unmatched mass；两者的标准几何计算不适合直接搬到百万 spots。
- [BANKSY, Nature Genetics 2024](https://www.nature.com/articles/s41588-024-01664-3)：真实机制可扩到百万细胞并同时表示自身状态和微环境。本文借鉴可扩展多尺度 prototype 输入；BANKSY 不包含 OT。

### 9.5 验收条件

- SPATCH/CRC 峰值显存、OT wall time 和 candidate 文件体积下降；
- spatial_cluster / codex coarse、cell type 与空间连续性至少有一组实质提升，不能只报告速度；
- rare-state recall 不因低秩或 prototype 压缩显著下降；
- 出口 A 报告 mass/factor/message 合同及抽样 support audit；出口 B2 才要求 full-run top-k hubness、reciprocal/cycle、entropy，并报告相对 exact-B1 的 ANN recall；
- 方向不对称和 unmatched 比例在相应接口可定义的口径下得到改善；
- SPATCH 当前 `cell_type_common` ARI 的绝对下降不超过约 `0.01`。

---

## 10. P2-G：state/domain 任务化读出与 final-stage 几何头

### 10.1 为什么一个 final embedding 不够

当前评估同时要求：

- 识别细胞类型或细粒度 cell state；
- 识别连续的组织空间域；
- 跨 section 混合；
- 保持模态信息并完成重构。

这些目标并不总是共享同一最优空间平滑强度。BANKSY 的结果和本项目 MouseBrain/SPATCH 的粒度差异都说明：

- cell typing 更依赖自身分子状态和弱空间上下文；
- domain segmentation 更依赖多尺度微环境；
- 强行输出一个 embedding，容易偏向较容易的粗 compartment。

### 10.2 建议输出接口

P1-A/G2 已负责产生 state/domain 两个结构通道；本节不是再造一套互相矛盾的 head，而是把同一通道提升为可保存、可评价的正式输出，并在 P2 阶段才考虑额外 factor/geometry objective。统一接口固定谁生成 prior、谁接收 message、谁进入 decoder：

```text
z_state_preOT
  = StateHead(identity, high-frequency, weak local context)

z_domain_preOT
  = DomainHead(local + medium + global context)

z_align_preOT
  = AlignReadout([z_state_preOT, z_domain_preOT])
  → EMA teacher 生成 OT prior

z_state_out
  = z_state_preOT                       # 首版始终旁路 candidate OT

z_domain_out
  = z_domain_preOT                      # O1-off / G2 单模块
  = MatchabilityGatedCandidateOT(
      z_domain_preOT,
      candidate_ot_prior
    )                                   # 仅 O1-on

z_candidate_final
  = JointReadout([z_state_out, z_domain_out])

z_final
  = IdentityPreservingMerge(z_legacy_final, z_candidate_final)
  → shared decoder 与通用下游输出
```

要求：

- 训练和评估保存 pre-OT、两个正式 head、candidate 和 final；
- shared decoder 使用 `z_final`，JointReadout 同时连接两个 head，确保二者都能获得重构梯度；若一个 head 的梯度长期接近零，实验判为结构未闭环；
- cell-type 指标优先评估 `z_state`；
- spatial-domain 指标优先评估 `z_domain`；
- `z_final` 是通用默认输出，不能掩盖两个 head 的相反变化；
- 不使用真实标签在训练时选择 head。

只有 domain-only OT 通过后，才单独测试“高 matchability message 是否也进入 state”。不能在首次引入双 head 时同时改变两个分支的跨 section 信息流。

### 10.3 可选的非负空间 factor head

可在 `z_domain` 上增加：

```text
beta = normalize(softplus(W_factor z_domain))
```

其中 `beta` 是少量非负空间 factor，作用是：

- 提供跨 section 可比较的粗/中尺度组织因子；
- 让 domain 信息具有可解释 readout；
- 检查 final embedding 是否只是高维弥散而没有稳定空间轴。

第一版可只导出 factor，不将其拼回 `z_state_out`。只有 factor 稳定且确实改善 domain 指标后，才考虑用小 residual 反馈到 `z_candidate_final`。

### 10.4 可选 final geometry head

当前 Fusion、GraphSAGE、OT 只通过重构间接得到监督。P1 结构稳定后，可增加训练期专用、推理时丢弃的轻量 head：

1. sampled spatial/feature edge reconstruction；
2. 多分辨率 balanced prototype prediction；
3. 在已构建稀疏邻边上保持输入/latent 局部 cosine geometry。

这类 head **不替换 L0，也不属于新的跨模态 contrastive loss**，但它仍然改变总目标，所以只能列为 P2，不能与 P1 结构同时启用。

不建议第一版使用单一 K 的 DEC 式伪聚类，因为它很容易：

- 把 section 当作 cluster；
- 固化早期伪标签；
- 只强化粗粒度结构。

### 10.5 学习来源与迁移边界

- [BANKSY, Nature Genetics 2024](https://www.nature.com/articles/s41588-024-01664-3)：真实机制用不同空间 mixing 强度服务 cell typing 与 domain segmentation。本文据此提出 state/domain 双神经 head；BANKSY 本身没有该双 head。
- [INSPIRE, Nature Genetics 2026](https://www.nature.com/articles/s41588-026-02579-x)：真实机制从 integrated latent 生成可解释的非负空间 factors，并使用 geometry regularizer 保留 section 内生物结构。本文借鉴 factor head 和稀疏 geometry 保护；INSPIRE 的 shared gene loading 不能直接推广到 HE、Protein、Metabolite。
- [COSMOS, Nature Communications 2025](https://www.nature.com/articles/s41467-024-55204-y)：真实机制用 DGI 捕获 local-global 图信息。本文只借鉴“final 表示需要直接看见 global context”的结构动机，明确不迁入其 DGI contrastive discriminator。
- [SMART, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70821-5)：真实机制用 MNN triplet 调整统一 latent 几何。本文只借鉴“final geometry 需要直接校准”的结论，不迁入 triplet loss。

### 10.6 验收条件

- MouseBrain 粗标签与细粒度 Y 至少由不同 head 各自取得优势；
- SPATCH 的 cell type 和 spatial cluster 不再只能由同一个低 k 粗解折中；
- CRC 的 `z_domain` 空间连续性和 `z_state` 内部几何分别改善；
- 两个 head 不是简单线性复制：邻域重叠、空间频谱和 gate 权重应有可解释差异；
- 若加入 factor/geometry head，必须独立做 `head off/on` 消融，L0 全程不变。

---

## 11. P2-H：MISAR 的跨模态 feature-prior adapter

### 11.1 适用范围

这个方向不是通用模块，只适用于存在可靠 feature correspondence 的模态组合：

- RNA–ATAC：peak–gene genomic proximity、enhancer–gene 或已知 regulatory links；
- RNA–Protein：gene–protein correspondence或已知 pathway；
- 不建议直接用于 HE–RNA、HE–Metabolite 或未知代谢物关系。

MISAR 中 RNA 单模态结构相对好，而 ATAC 和 joint 结果较弱。除预处理外，通用 MLP 也没有利用 RNA–ATAC 的生物关系。

### 11.2 建议的轻量版本

不建议第一版建立全 feature heterogeneous transformer。可先增加稀疏 residual adapter：

```text
h_ATAC_to_RNA
  = SparsePriorAdapter(ATAC features, peak-gene graph)

h_RNA_fuse
  = h_RNA
    + alpha_prior * h_ATAC_to_RNA
```

或在 decoder 侧约束跨模态可预测部分：

```text
RNA shared decoder weights
  ← low-rank prior-guided adapter from ATAC
```

若 `alpha_prior` 从 0 开始，prior adapter 必须使用非零随机初始化，不能再把其最后一层置零；prior edge 带置信度并允许丢弃。若当前模型只输入 PCA/LSI 且已丢失 feature identity，则该 adapter 必须放在降维前的 feature encoder，或依赖保存的 loading；这会扩大实现范围，因此列为 P2。

### 11.3 学习来源与迁移边界

- [MultiGATE, Nature Communications 2025](https://www.nature.com/articles/s41467-025-63418-x)：真实机制是先在 peak–gene、gene–protein 等跨模态 feature-prior graph 上做 GAT，再在各模态 spatial graph 上传播，并使用对称/绑定的 encoder-decoder。本文借鉴稀疏 feature-prior adapter；不迁入其 CLIP-style loss。
- [scGLUE, Nature Biotechnology 2022](https://www.nature.com/articles/s41587-022-01284-4)：真实机制利用先验 regulatory graph 连接不同组学 feature，并通过模态化 autoencoder 学习。本文只借其 feature graph 思路；scGLUE 不是空间方法。

### 11.4 风险与验收

风险：

- 错误 peak–gene prior 会系统性传播偏差；
- 稀有或远端调控不在简单 proximity graph 中；
- 高维 ATAC feature graph 在大样本上开销高；
- prior adapter 可能让 RNA 主导 ATAC，而不是互补融合。

验收：

- MISAR ATAC/Combined joint 指标改善，同时 RNA_Clusters ARI 不下降；
- prior edge 随机打乱应显著破坏收益，否则收益可能并非来自生物先验；
- 不在 MouseBrain、SPATCH 等无可信 feature map 的组合上强行复用。

---

## 12. 当前不建议优先做的结构改动

### 12.1 不建议继续修改或叠加跨模态对比目标

明确排除：

- COSMOS 的 DGI discriminator；
- SpaMosaic 的 contrastive 部分；
- SMART 的 triplet loss；
- MultiGATE 的 CLIP-style loss；
- InfoNCE/VICReg 替换；
- 任何先改变 L0 再判断新结构的实验。

本文只学习这些工作的图、门控、连接、分解和重构结构。

### 12.2 不建议只增加 latent 维度

按第 2.5 节可复现口径得到的探索性诊断中，CRC effective rank 约 `50.3`，Thymus 约 `12.0`。它们提示两者的问题方向可能不同，至少没有证据支持统一把 latent 从 128 扩大到 256；但在该诊断进入正式评估流水线和多 seed 复现前，不把 rank 数值本身作为 go/no-go。

### 12.3 不建议只堆更深 GraphSAGE

更深的单一低通图会扩大过平滑风险，也不能解决：

- 模态内 feature relation；
- state/domain 粒度冲突；
- 边界；
- 不同平台的物理感受野。

### 12.4 不建议直接上完整 Transformer 或统一 heterogeneous graph

把“模态、spot、section、spatial edge、feature edge、OT edge”一次性放入大型 hetero-transformer，虽然表达能力强，但：

- 无法知道收益来自哪条 relation；
- 对 CRC/SPATCH 的显存风险大；
- 当前最明显的几个模块问题可以用轻量 residual 修复；
- 容易破坏 MouseBrain 和 SPATCH 已有优势。

可以把它作为所有增量模块完成消融后的长期重构，而不是下一步。

### 12.5 不建议第一轮上完整 raw-count VAE

MultiVI/MIDAS 等概率模型很有启发，但完整迁移意味着同时改变：

- 输入数据；
- likelihood；
- decoder；
- batch covariate；
- 训练和显存策略。

这会与“架构模块归因”冲突。第一轮先在现有 PCA/Harmony 目标上验证 shared/private/conditional decoder。

### 12.6 不建议无条件 adversarial 去 section

MISAR stage、肿瘤区域和不同个体都可能含真实生物差异。只有技术 replicate 或可靠 shared population 才适合强制 batch-invariant；其他情况应采用 condition-private、bounded alignment 或 explicit unmatched。

### 12.7 缺失模态支持暂不列为性能第一优先

当前 8 个主运行要求所有 section 具有相同观测模态集合，确实限制了未来 mosaic 数据。MultiVI、MIDAS、SpaMosaic 和 [scVAEIT, PNAS 2022](https://doi.org/10.1073/pnas.2214414119) 提供了 availability mask、observed-only reconstruction 和 modality dropout 的经验，但当前 8 数据集的主要性能瓶颈并不是 missing modality。可以在 P1-B/P1-C 接口中预留 mask，不必先重构整个训练管线。

---

## 13. 建议的结构实验顺序

### 13.1 原则

- L0 函数、`gamma`、输入 shape/L2 normalization/spot order 与当前数据集权重/调度全程冻结；shared trunk/Head_cv 参数和梯度不宣称冻结；
- 每次只改变一个结构因素；
- 新 residual 采用第 3.3 节的一种 identity 机制，禁止双重零初始化；
- 小数据完成机制筛选后，再上 CRC/SPATCH；
- 单 seed 只允许用于 shape、loss finite、显存和运行时间 smoke test，不能用于性能 go/no-go；
- 所有性能筛选都让 baseline 与 candidate 使用同一组至少 3 个 model-training seeds（建议 `42/123/2026`），并复用完全相同的预处理、spot 顺序、KMeans seeds/n_init、固定 K、抽样 index 和评价脚本；
- 保存逐 seed 配对差值；若共运行 `n≥3` 个 seeds，候选的主指标需要 median 达到目标且至少 `ceil(2n/3)` 个 seeds 同方向，回归保护则要求 candidate median 高于 floor 且至少 `ceil(2n/3)` 个 seeds 不越界；
- 不以 total loss 选择模型，因为现有 L0 的数值尺度与突刺会干扰判断；
- 以固定下游评估协议和 stage-level 表示选择结构。

### 13.2 推荐实验梯子

| 阶段 | 唯一新增变量 | 首选数据集 | 目的 |
|---|---|---|---|
| B0 | 精确复现当前基线 | 8 个数据集 | 保存可加载的 legacy state_dict/buffer/prior/RNG/config，并建立 module-off identity test 与资源基线 |
| G0 | self-path 语义消融；不加新图 | MouseBrain、Simulation、HLN | 确认当前 GraphSAGE 的 self/neighbor 比例 |
| G1 | fusion 后 local+context+identity residual | Simulation、HLN、CRC 子集 | 验证多尺度图本身 |
| G2 | high-frequency + state/domain heads | MouseBrain、SPATCH 子集 | 防过平滑和粒度冲突 |
| G3 | per-modality spatial/feature graph | MISAR、HLN | 验证 early graph 是否优于 post-fusion-only |
| F1 | 双头 encoder + quality residual gate | HLN、Spleen、Simulation | 隔离融合质量问题 |
| D1 | private + section-conditioned decoder | Simulation、Thymus、MISAR | 隔离 shared/private 与 section effect |
| O1 | adaptive support + 分解后的 matchability | Simulation、CRC 子集、Thymus | 先修正 OT 信息语义 |
| O2 | EMA pre-OT teacher + topology cost | Simulation、CRC、SPATCH 子集 | 避免自反馈并保护结构 |
| S1 | explicit section graph + pair weighting | Thymus、MISAR | 验证多 section 连接 |
| L1 | low-rank OT | CRC、SPATCH | 验证大规模速度和精度 |
| L2 | prototype-to-spot refinement | 仅当 L1 仍有 hub/高熵 | 进一步处理大规模语义匹配 |
| H1 | feature-prior adapter | MISAR | 数据集专项增益 |
| J1 | factor/geometry auxiliary head | P1 模块稳定后 | 直接塑造 final 几何 |

只有单模块通过后，才建议组合：

```text
推荐组合顺序：

G2
→ G2 + F1
→ G2 + F1 + D1
→ G2 + F1 + D1 + O1
→ + S1（多 section）
→ + L1/L2（百万 spot）
→ 最后才考虑 J1
```

### 13.3 每一级必须保存的表示

```text
z_cv_m
z_legacy_fusion
z_legacy_graph
z_legacy_final
z_quality_fusion
z_graph_local
z_graph_context
z_state_preOT
z_domain_preOT
z_state_out
z_domain_out
z_preOT_teacher
z_candidate_final
z_final
p_m
u_section / c_condition
legacy_ot_prior metadata
candidate_ot_prior metadata
```

每一级统一计算：

- 固定 k 和 best k 的 ARI/NMI；
- Label ASW、内部 ASW/CH/DBI；
- bASW、bLISI、kBET、PCR；
- 多 k 空间连续性；
- effective rank、top-PC variance、isotropy；
- section centroid / within-section RMS；
- modality/section/known nuisance 的线性可预测性；
- graph relation weights、fusion gate、private/shared 信息分配；
- sparse/candidate-B OT：coverage、entropy、margin、reciprocal、cycle、hubness、unmatched、refresh stability 及 validity masks；
- factorized outlet-A OT：row/column mass、unmatched、factor/column concentration、effective factor rank、message norm、refresh stability与固定抽样 support audit；
- time、peak GPU、CPU RAM、临时文件大小。

这一步能回答“性能在哪一 stage 开始变差”，避免再次只能从 final embedding 推断原因。

### 13.4 建议的 go/no-go 门槛

下表是**工程筛选目标**，不是论文显著性声明。括号内先给 seed42 当前值和基于它的 provisional floor；完成至少 3 个 paired baseline seeds 后，正式 floor 统一改为：

```text
越高越好：
floor = median(baseline seeds) - max(表中允许绝对下降, SD_baseline)

越低越好：
ceiling = median(baseline seeds) + max(表中允许绝对上升, SD_baseline)
```

其中 `SD_baseline` 固定为跨 model-training seeds 的 **sample standard deviation，`ddof=1`**。主改进指标要求 paired median 达到表中绝对变化，并且至少 `ceil(2n/3)` 个 seeds 同方向。所有外部 ARI 都锁定表中给出的 K，不允许重新扫描 best-k 后再挑结果；这样不会用单 seed 的 `0.01` 差异或 K 后选假装超过未知训练方差。

| 数据集 | 主改进目标 | 回归保护：seed42 参考值 → provisional floor/ceiling |
|---|---|---|
| MouseBrain | joint k=8/10 空间稳定性或细粒度 Y/state 指标改善 | 固定聚类 k=5：RegionLoupe ARI `0.7654→≥0.7554`；annotations `0.6975→≥0.6875`；celltype `0.6743→≥0.6643` |
| Simulation | RNA/ADT nuisance leakage 各至少绝对下降 `0.10`；Top-1 `0.1582→≥0.2582` | spatial factor `0.9833→≥0.9633`；bLISI `0.9767→≥0.9467`；kBET `0.9867→≥0.9567` |
| Human Lymph Node | ASW raw 至少提高 `0.03` 或 DBI 至少下降 `15%` | bLISI `0.9331→≥0.8831`；kBET `0.8241→≥0.7741` |
| Mouse Spleen | ASW raw 至少提高 `0.03` 或 DBI 至少下降 `15%`，且多 K 空间不只单点改善 | bLISI `0.9818→≥0.9318`；kBET `0.9695→≥0.9195` |
| CRC | joint k=8 空间 `0.4923→≥0.5423`；k=8 DBI `3.9999→≤3.3999`；hubness 在 captured mass 不降的前提下下降 | bASW `0.9962→≥0.9862`；bLISI `0.4636→≥0.4336`；kBET `0.3104→≥0.2804`；PCR_score `0.9901→≥0.9801`；612,374 spots 全量可运行 |
| SPATCH | joint k=10 空间 `0.7515→≥0.7815`；实验前二选一锁定：spatial_cluster ARI@k=5 `0.3301→≥0.3501`，或 codex ARI@k=3 `0.2555→≥0.2755` | cell_type_common ARI@k=3 `0.5786→≥0.5686`；bLISI `0.3595→≥0.3295`；kBET `0.1595→≥0.1295`；joint spatial k=16 `0.6983→≥0.6683`、k=20 `0.6562→≥0.6262`；1,068,962 spots 不 OOM |
| Thymus | k=5 section ARI `0.3070→≤0.2570` 与 pair hub/entropy 同时改善；不把 rank 上升单独算成功 | joint spatial k=8 `0.5826→≥0.5526`、k=12 `0.4675→≥0.4375`；condition/section-private 仍可解释；17,824 spots 全量评估 |
| MISAR | 在同一 joint 协议下实验前锁定一个主端点：RNA ARI@k=10 `0.3066→≥0.3266`、Y@k=14 `0.2848→≥0.3048` 或 ATAC@k=12 `0.2597→≥0.2797`；不再以 joint–independent gap 为目标 | 同一固定 K 下：RNA `≥0.2966`、Y `≥0.2748`、ATAC `≥0.2497`；joint spatial k=8 `0.7174→≥0.6874`、k=16 `0.6253→≥0.5953`；stage/condition 分支仍保留可解释差异 |

对于无可靠生物真值的 CRC、Human Lymph Node、Spleen、Thymus，达到内部、batch 或空间门槛仍只表示“值得继续”，不能单独宣称生物学改进。

---

## 14. 未来实现时的代码落点

> 本文没有修改以下任何代码；此表只说明未来实施范围。

| 文件 | 未来结构职责 |
|---|---|
| `model/model_component.py` | dual-head encoder、quality gate、shared/private heads、multiscale relation graph、state/domain heads、conditional decoder、matchability-gated OT readout |
| `model/stage_model.py` | 新的数据流编排、stage outputs、互不覆盖的 legacy/candidate OT prior、section graph、pair-quality aggregation、EMA pre-OT teacher、observed mask |
| `model/utils.py` | local/context/radius/mutual-feature graph、密度尺度、可显式选择的 self-path 语义 |
| `model/sparse_uot.py` | adaptive support、ragged mask、matchability 分量、dustbin/keep-source、low-rank/prototype接口 |
| `model/configure.py` | 让所有结构开关真实生效，增加明确的模块级配置 |
| `model/loss.py` | **保留现有 L0 实现不变**；若未来有 geometry/factor auxiliary objective，应与 L0 函数分离 |
| `scripts/run_*.py` | 真实配置快照、stage dump、OT/graph/gate QC、资源统计 |

### 14.1 配置真实性是结构实验前置条件

当前默认配置中，下列字段在主模型执行路径里没有真实控制对应行为，或只有名字没有实现：

- `encoder.residual`
- `graph.use_spatial_graph`
- `graph.use_feature_graph`
- `graphsage.num_layers`
- `graphsage.use_distance_weight`
- `uot.initial_from_modalities`
- `uot.update_from_final_embedding`
- `uot.use_momentum`
- `uot.normalize_total_mass`
- `uot.cost`
- `loss.use_ot_loss`
- `loss.use_spatial_smooth_loss`
- `loss.use_gate_regularization`
- `reconstruction.loss`
- `contrastive.loss_weight`

例如：

- GraphSAGE 构造没有消费 `num_layers`，见 `model/stage_model.py:194-203`；
- 空间图仍被直接构建，见 `model/stage_model.py:829-846`；
- decoder 路径硬编码 MSE，见 `model/stage_model.py:430-484`；
- 总损失硬编码为 reconstruction + L0，见 `model/stage_model.py:1064`；
- UOT prior 刷新直接覆盖，没有配置所暗示的 momentum。

其中 `uot.update_from_final_embedding` 这个配置字段本身没有控制主路径，但 dense/candidate 运行到底从初始模态表示还是 final embedding 刷新，当前仍可能由数据集脚本或 CLI 分支实际决定；因此应修复的是“配置与真实 source of truth 脱节”，不能误写为项目完全没有 final-embedding refresh 能力。

在结构实验前，每个字段必须满足三者之一：

1. 真正实现；
2. 删除；
3. 非默认值时显式报错。

否则会产生“配置改了，但结构根本没有变”的假消融。

---

## 15. 文献来源与适用范围

### 15.1 直接空间多组学来源

| 方法 | 期刊/年份 | 本文实际学习的机制 | 不迁移或需谨慎的部分 |
|---|---|---|---|
| [SpatialGlue](https://www.nature.com/articles/s41592-024-02316-4) | Nature Methods, 2024 | 每模态 spatial/feature 双图、模态内和模态间 spot-wise attention | 主要是同点 paired data；attention 不等于校准质量 |
| [COSMOS](https://www.nature.com/articles/s41467-024-55204-y) | Nature Communications, 2025 | per-modality spatial GCN、WNN 局部模态权重 | 不迁入 DGI contrastive loss；主要两模态 |
| [SMART/SMART-MS](https://www.nature.com/articles/s41467-026-70821-5) | Nature Communications, 2026 | per-modality GraphSAGE、graph decoder、all-pair cross-section MNN 候选、可扩展性 | 不迁入 triplet；其空间图是 section block diagonal |
| [SpaMosaic](https://www.nature.com/articles/s41588-026-02573-3) | Nature Genetics, 2026 | 切片内空间图、跨 batch MNN、bridge connectivity、模态化重构 | 不迁入 contrastive；MNN/bridge 需连通且可能过对齐 |
| [SpaMV](https://www.nature.com/articles/s41467-026-74718-1) | Nature Communications, 2026 | shared/private 表示与专家融合 | early/unedited online version；主要 paired vertical integration |
| [MultiGATE](https://www.nature.com/articles/s41467-025-63418-x) | Nature Communications, 2025 | feature-prior GAT 后接模态内 spatial GAT | 不迁入 CLIP-style loss；只适合可信 feature prior |

### 15.2 空间多样本、空间结构或对齐来源

| 方法 | 期刊/年份 | 本文实际学习的机制 | 边界 |
|---|---|---|---|
| [MENDER](https://www.nature.com/articles/s41467-023-44367-9) | Nature Communications, 2024 | multi-range context、multi-slice、可扩展 | 单组学、非端到端 GNN |
| [BANKSY](https://www.nature.com/articles/s41588-024-01664-3) | Nature Genetics, 2024 | own state + neighborhood mean + spatial gradient；不同空间混合服务不同任务 | 单组学；双神经 head 是本文推断 |
| [STAGATE](https://www.nature.com/articles/s41467-022-29439-6) | Nature Communications, 2022 | learned spatial edge attention、边界感知 | 单切片转录组；预聚类剪边可能自强化 |
| [SLAT](https://www.nature.com/articles/s41467-023-43105-5) | Nature Communications, 2023 | 多跳图拼接、密度适配邻居、高置信 anchor/clipping | 不迁入 adversarial loss |
| [PASTE](https://www.nature.com/articles/s41592-022-01459-6) | Nature Methods, 2022 | 表达相似性 + 内部空间结构的 FGW | 小规模注册；不是 attention |
| [PASTE2](https://genome.cshlp.org/content/33/7/1124) | Genome Research, 2023 | partial overlap、slice-specific population | partial mass 估计和二次几何成本 |
| [MEFISTO](https://www.nature.com/articles/s41592-021-01343-9) | Nature Methods, 2022 | 平滑/非平滑的空间或时间因子；多 group 一维时间轨迹的 DTW 对齐 | 因子模型，不是图神经网络；不能作为跨 section 空间配准依据 |
| [INSPIRE](https://www.nature.com/articles/s41588-026-02579-x) | Nature Genetics, 2026 | shared spatial GNN、bounded section alignment、section-specific decoder、非负 spatial factors、geometry protection | 多 section transcriptomics，不是多组学 |

### 15.3 通用 OT 与非空间多组学迁移来源

| 方法 | 期刊/年份 | 本文实际学习的机制 | 边界 |
|---|---|---|---|
| [moscot](https://www.nature.com/articles/s41586-024-08453-2) | Nature, 2025 | W/GW/FGW、multimodal cost、unbalanced、GPU、low-rank | solver/framework，不是 Spa-Mo attention |
| [MultiVI](https://www.nature.com/articles/s41592-023-01909-9) | Nature Methods, 2023 | modality-specific posterior/likelihood、observed-only likelihood、不确定性 | 非空间；完整 VAE 不是第一轮 |
| [MIDAS](https://www.nature.com/articles/s41587-023-02040-y) | Nature Biotechnology, 2024 | biological state / technical noise disentanglement、模块化 decoder | 非空间单细胞 |
| [scDisInFact](https://www.nature.com/articles/s41467-024-45227-w) | Nature Communications, 2024 | shared biology、condition-specific biology、technical batch 分离 | 非空间；condition 定义错误会删真实信号 |
| [scGLUE](https://www.nature.com/articles/s41587-022-01284-4) | Nature Biotechnology, 2022 | feature regulatory graph 和 modality-specific autoencoder | 非空间；只适合有可信 feature graph 的模态 |
| [scVAEIT](https://doi.org/10.1073/pnas.2214414119) | PNAS, 2022 | 显式 missing mask、训练期随机 mask 与 conditional reconstruction | 非空间；完整 VAE 不是当前性能第一优先 |

---

## 16. 最终建议

### 16.1 推荐的目标架构与首轮最小子集

完整目标架构是：

```text
1. 保留当前 L0 函数、gamma、输入格式与权重/调度；不声称冻结 shared trunk/head 参数；
2. 在 fusion 后增加 identity + local + context + high-frequency 图 residual；
3. 在 OT 前输出 z_state/z_domain，首版只让 domain 接收 OT，JointReadout 后进入 shared decoder；
4. encoder 增加保持 L0 调用接口的 fusion adapter；
5. 先用 identity-preserving residual 校正现有 FusionMLP，验证后再 cross-fade 到 fully gated fusion；
6. decoder 增加小维 modality-private + section-conditioned residual，并只使用一种 zero-init 机制；
7. 暂时保留 UOT solver，但改 adaptive support、attention support_mask、分解后的 matchability 和真正 bypass post-LN 的 keep-source；
8. legacy 退出前双轨保存 OT state；只有 candidate prior 改由 EMA pre-OT embedding 生成；
9. 多 section 使用显式 section graph 和 pair-quality aggregation。
```

但这 9 项**不应在同一轮同时打开**。首轮最小子集只包括：

```text
L0 损失接口冻结
+ P1-A 的 fusion 后 state/context 图
+ P1-B 的 legacy quality-correction residual
+ P1-C 的小维 private/section-conditioned decoder residual
```

这里列的是首轮**候选集合**，不是要求在一个 run 同时打开三项。OT 与 section graph 保持当前行为，只额外保存完整诊断；三个结构分别通过独立消融后，再按 `G2 → F1 → D1 → O1 → S1` 的顺序组合。

### 16.2 为什么不是一次性重写

当前模型已经有两项非常重要的资产：

- MouseBrain 上很强的生物标签结果；
- CRC/SPATCH 的 full-spot 可扩展性，尤其 SPATCH 超过 100 万 spots。

最合理的方向不是推翻现有模型，而是把现有数据流中承担过多职责的 shared 128d 表示拆开，并让每个新增模块以可关、可审计、可回退的 residual 方式进入。

### 16.3 成功标准

真正成功的结构升级不应只是“某个 ARI 更高”，而应同时表现为：

- shared embedding 的 nuisance/section 泄漏下降；
- state 与 domain 两种粒度都能被表达；
- 空间连续性提高但边界和稀有状态不被抹平；
- OT 能明确拒绝不可匹配群体；sparse/ANN-support 接口用 row mass、coverage、entropy、reciprocal/cycle、hubness 和 stability，factorized 无显式 support 接口则用 mass/factor/message 合同与抽样 support audit，而不是对两者强套同一个 confidence；
- 多 section 关系不再由文件列表顺序决定；
- 百万 spot 仍能全量运行；
- MouseBrain 和 SPATCH 的现有优势得到保护；
- 所有改善能在 stage-level 表示和独立消融中定位到具体模块。
