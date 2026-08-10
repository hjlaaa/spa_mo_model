# spa_mo_model 从 O1-a 到 O2-c0 的 OT 信息路径实验演进与效果报告

## 1. 文档目的

本文档系统记录 `spa_mo_model` 以 G0-b 为共同基线，围绕 OT 信息路径开展的六轮模型结构实验：

- O1-a：可靠度控制的 OT bypass；
- O1-b：相对可靠度与选择性 OT bypass；
- O2-a：全局/局部 OT residual 分解；
- O2-b0：detached pre-OT prior 刷新；
- O2-b1：EMA pre-OT prior 刷新；
- O2-c0：拓扑感知 OT 成本。

文档重点回答以下问题：

1. 每个版本具体修改了模型中的哪一段信息路径；
2. 为什么上一版本会过渡到下一版本；
3. 每个版本在哪些数据集和指标上具有优势；
4. 每个版本又在哪些方面出现下降、跨 seed 不稳定或收益—代价冲突；
5. 哪些实验机制已经得到验证，哪些结构方向不应继续作为当前主线；
6. 为什么六个候选版本最终都没有替换 G0-b。

这条实验路线的共同出发点是：

```text
G0-b
  │
  ├─ O1-a：根据 spot 的绝对 matchability 缩放 OT residual
  │      └─ O1-b：改成 direction 内相对百分位，只削弱低可靠尾部
  │
  ├─ O2-a：放弃可靠度打分，直接把 OT residual 分成全局和局部部分
  │
  └─ O2-b0：完整保留 G0-b OT 输出，只改变动态 prior 的刷新来源
         ├─ O2-b1：用 EMA teacher 稳定 pre-OT prior 刷新
         └─ O2-c0：回到 O2-b0 母版，在刷新成本中加入局部拓扑信息
```

需要特别注意：

- O1-b 是在 O1-a 思路上的可靠度标定改进；
- O2-a 不再沿用 O1 的 matchability，而是回到 G0-b 后重新设计 OT residual；
- O2-b0 也不是叠加在 O2-a 上，而是回到完整 G0-b OT 输出，只修改 prior 刷新闭环；
- O2-b1 直接建立在 O2-b0 上；
- O2-c0 没有叠加 O2-b1 的 EMA，而是回到 O2-b0 母版后加入拓扑成本。

## 2. 统一实验口径

### 2.1 共同基线

六轮实验都以已经确认保留的 G0-b 为结构基线：

```text
GraphSAGE：
  保留 adjacency self-loop
  旁路 self_linear
  保留 outer residual
```

G0-b 之外，以下部分原则上保持冻结：

- 已经确认的 legacy 对比损失 L0；
- Simulation 和 Human Lymph Node 的 `lambda_contrast=0.1`；
- MouseBrain 原有的 lambda schedule；
- 数据预处理；
- Fusion 主体；
- GraphSAGE 主体；
- decoder；
- UOT solver 超参数；
- OT attention top-K；
- 评价脚本及标签口径。

各数据集的局部空间图 K 保持为：

| 数据集 | 空间图 K |
|---|---:|
| Simulation | 10 |
| Human Lymph Node | 10 |
| MouseBrain | 5 |

这里的 K 是模型内部空间 KNN 图的邻居数。Simulation 和 Human Lymph Node 的候选稀疏 OT 另外使用 `candidate K=200`，OT attention 使用 `top-K=10`，两者与空间图 K 不是同一个参数。

### 2.2 正式训练与评估协议

每个候选版本均完成：

```text
3 datasets × 3 model seeds = 9 次 200-epoch 训练
9 trainings × 5 clustering seeds = 45 次评估
```

统一随机种子为：

- model-training seeds：`42、123、2026`；
- clustering seeds：`0、1、2、3、4`。

汇总时：

1. 先在每个 model seed 内对 5 个 clustering seeds 取中位数；
2. 再计算候选版本相对参考版本的逐 model-seed 配对改善量；
3. 最后报告 3 个 model seeds 的配对改善量中位数和改善 seed 数。

### 2.3 指标方向

本文中的正负号全部来自各结果目录的 `paired_deltas.csv`：

```text
improvement_delta > 0：候选版本变好
improvement_delta < 0：候选版本变差
```

因此，对于 DBI、FOSCTTM 和 nuisance leakage 等原始定义中“越低越好”的指标，汇总脚本已经统一转换方向。

文档中的：

```text
+0.0200（3/3）
```

表示配对改善量中位数为 `+0.0200`，并且 3 个 model seeds 全部改善。

### 2.4 为什么必须看配对改善量

不同实验目录可能复用既有 G0-b，也可能包含不同版本的参考结果。为了避免直接用两个目录中的 raw median 相减造成误导，本文的版本效果数字统一来自正式配对结果。

## 3. 六个版本的总体结论

| 版本 | 唯一核心修改 | Simulation | Human Lymph Node | MouseBrain | 机制 | 总体决定 |
|---|---|---:|---:|---:|---:|---|
| O1-a | 用绝对 matchability 缩放 OT residual | PASS | FAIL | PASS | PASS | NO-GO |
| O1-b | direction 内相对百分位，只削弱低可靠尾部 | FAIL | FAIL | FAIL | PASS | NO-GO |
| O2-a | 完整保留全局 OT residual，只保留 25% 局部 residual | FAIL | FAIL | FAIL | PASS | NO-GO |
| O2-b0 | dynamic prior 从 detached pre-OT embedding 刷新 | PASS | FAIL | FAIL | PASS | NO-GO |
| O2-b1 | 用 momentum=0.5 的 EMA teacher 刷新 prior | FAIL | FAIL | PASS | PASS | NO-GO |
| O2-c0 | 0.8 语义成本 + 0.2 局部拓扑成本 | FAIL | FAIL | PASS | PASS | NO-GO |

NO-GO 不表示该版本所有指标都下降，而是表示该版本没有同时满足三个筛选数据集的收益和回归保护门槛。

六轮实验中最突出的局部优势分别是：

- O1-a：MouseBrain 综合表现最强，同时显著改善 Simulation 匹配；
- O1-b：保留了一部分 MouseBrain 空间连续性，但整体性能明显退化；
- O2-a：Simulation 匹配提升幅度最大；
- O2-b0：Simulation 中“匹配、空间域、batch”三者最均衡；
- O2-b1：刷新轨迹最稳定，MouseBrain spatial-k10 提升最大；
- O2-c0：O2-b 系列中 Simulation 匹配最好，并使 MouseBrain 粗标签和 batch 指标同时改善。

## 4. 跨版本关键指标总表

以下表格均为候选版本相对 G0-b 的配对改善量中位数，括号中为改善 seed 数。

### 4.1 Simulation

| 版本 | Top1 | Recall@5 | Spatial-domain ARI | spfac recovery | bLISI | kBET |
|---|---:|---:|---:|---:|---:|---:|
| O1-a | +0.0285（2/3） | +0.0368（3/3） | -0.0930（0/3） | -0.0006（0/3） | -0.0000（1/3） | +0.0012（2/3） |
| O1-b | -0.0039（1/3） | -0.0087（0/3） | -0.0940（1/3） | -0.0061（0/3） | -0.0044（0/3） | -0.0090（1/3） |
| O2-a | +0.0802（3/3） | +0.0934（3/3） | -0.0901（0/3） | +0.0027（2/3） | +0.0023（2/3） | +0.0066（2/3） |
| O2-b0 | -0.0006（1/3） | +0.0110（2/3） | -0.0011（1/3） | +0.0057（3/3） | +0.0021（3/3） | +0.0037（2/3） |
| O2-b1 | +0.0052（2/3） | +0.0168（3/3） | -0.0277（1/3） | +0.0022（3/3） | +0.0011（3/3） | +0.0014（2/3） |
| O2-c0 | +0.0199（3/3） | +0.0403（3/3） | -0.0348（1/3） | +0.0064（3/3） | +0.0009（3/3） | +0.0025（2/3） |

最重要的规律：

- O2-a 的匹配提升最大，但空间域 ARI 系统性下降；
- O2-b0 的匹配增益较小，却几乎保护住空间域 ARI，并改善 spfac、bLISI 和 kBET；
- O2-b1 和 O2-c0 重新增加匹配强度时，空间域 ARI 又开始下降；
- Simulation 中更强的跨切片匹配并不自动等于更好的空间域恢复。

### 4.2 Human Lymph Node

| 版本 | silhouette | DBI | spatial-k10 | bLISI | kBET |
|---|---:|---:|---:|---:|---:|
| O1-a | +0.0004（2/3） | +0.0234（2/3） | +0.0125（3/3） | -0.0082（0/3） | -0.0228（0/3） |
| O1-b | +0.0016（2/3） | +0.0434（2/3） | -0.0043（1/3） | -0.0109（0/3） | -0.0291（0/3） |
| O2-a | +0.0023（2/3） | +0.0466（2/3） | +0.0355（3/3） | -0.0289（0/3） | -0.1002（0/3） |
| O2-b0 | +0.0001（2/3） | -0.0007（1/3） | +0.0040（2/3） | -0.0040（1/3） | -0.0137（1/3） |
| O2-b1 | -0.0007（1/3） | +0.0301（2/3） | +0.0109（3/3） | -0.0065（1/3） | -0.0208（1/3） |
| O2-c0 | +0.0008（2/3） | -0.0052（1/3） | +0.0094（3/3） | -0.0068（0/3） | -0.0238（0/3） |

最重要的规律：

- 除 O1-b 外，多数版本都能提高 HLN spatial-k10；
- 空间结构提升几乎总是伴随 bLISI 和 kBET 下降；
- O2-a 的 spatial-k10 提升最大，但 batch mixing 下降也最严重；
- O2-b0 是 HLN batch 代价相对最小的版本，但 kBET 仍超过保护边界；
- O2-b1 的 EMA 和 O2-c0 的拓扑成本都没有解决这个核心冲突。

### 4.3 MouseBrain

| 版本 | Region | annotations | celltype | spatial-k10 | bLISI | kBET |
|---|---:|---:|---:|---:|---:|---:|
| O1-a | +0.0395（3/3） | +0.0404（3/3） | +0.0271（3/3） | +0.0500（2/3） | +0.0468（3/3） | +0.0594（3/3） |
| O1-b | -0.0015（1/3） | -0.0051（1/3） | -0.0003（1/3） | +0.0862（2/3） | +0.0013（2/3） | -0.0024（1/3） |
| O2-a | -0.0014（1/3） | -0.0057（1/3） | -0.0023（1/3） | -0.0014（1/3） | -0.0106（1/3） | -0.0131（0/3） |
| O2-b0 | +0.0211（2/3） | +0.0202（2/3） | +0.0145（2/3） | +0.0423（2/3） | +0.0101（2/3） | -0.0137（1/3） |
| O2-b1 | +0.0168（2/3） | +0.0157（2/3） | +0.0199（2/3） | +0.1115（2/3） | +0.0030（2/3） | -0.0042（1/3） |
| O2-c0 | +0.0206（3/3） | +0.0210（3/3） | +0.0161（3/3） | +0.1068（2/3） | +0.0094（2/3） | +0.0216（2/3） |

最重要的规律：

- O1-a 是六个版本中 MouseBrain 最全面的版本，粗标签、空间结构和 batch mixing 同时提高；
- O1-b 为了选择性削弱 OT，保留了空间提升，却丢失了 O1-a 的粗标签和 batch 优势；
- O2-a 在 MouseBrain 上出现严重 seed 不稳定；
- O2-b0、O2-b1 和 O2-c0 逐渐恢复了 MouseBrain 的空间与粗标签收益；
- O2-c0 是 O2-b 系列里综合最好的 MouseBrain 版本，但仍不足以抵消 Simulation 和 HLN 的失败。

## 5. O1-a：可靠 Matchability 控制的 OT Bypass

正式结果：

- [result_O1a_reliable_ot_bypass/SUMMARY.md](result_O1a_reliable_ot_bypass/SUMMARY.md)
- [result_O1a_reliable_ot_bypass/paired_deltas.csv](result_O1a_reliable_ot_bypass/paired_deltas.csv)

### 5.1 修改动机

G0-b 对每个 spot 都完整应用 OT attention。这个设计默认所有跨切片匹配都同样可信，但实际 UOT prior 中可能存在：

- top-K 权重分散、没有明确对应 spot；
- 正向和反向候选不一致；
- row mass 较低、可匹配质量不足。

因此 O1-a 尝试让可靠匹配使用完整 OT，让不可靠匹配更多保留 pre-OT GraphSAGE 表示。

### 5.2 具体修改

O1-a 从三个停止梯度的分量计算 matchability：

1. top-K 熵集中度；
2. 正反向 reciprocal support；
3. 归一化 UOT row mass。

三个分量等权融合为 `r_i`：

```text
delta_i = legacy_OT_i - pre_OT_i

final_i = pre_OT_i + r_i × delta_i
```

其中：

- `r_i=0`：完全 bypass OT，精确保留 pre-OT 表示；
- `r_i=1`：完整恢复 G0-b legacy OT 输出；
- `0<r_i<1`：按可靠度缩放 OT residual。

O1-a 不改变 OT prior 本身，也不增加可学习参数。

### 5.3 优势

Simulation：

- Top1 `+0.0285`，2/3 seeds 改善；
- Recall@5 `+0.0368`，3/3 seeds 改善；
- FOSCTTM `+0.0012`；
- kBET `+0.0012`。

Human Lymph Node：

- silhouette `+0.0004`；
- DBI `+0.0234`；
- spatial-k10 `+0.0125`，3/3 seeds 改善。

MouseBrain：

- Region `+0.0395`，3/3 seeds 改善；
- annotations `+0.0404`，3/3 seeds 改善；
- celltype `+0.0271`，3/3 seeds 改善；
- spatial-k10 `+0.0500`；
- bLISI `+0.0468`，3/3 seeds 改善；
- kBET `+0.0594`，3/3 seeds 改善。

O1-a 证明：

- G0-b 的完整 OT residual 并非对所有 spot 都最优；
- 按 spot 保留一部分 pre-OT 表示可以明显改善匹配和 MouseBrain 生物结构；
- OT information-path 是值得优化的方向。

### 5.4 下降与问题

Simulation：

- spatial-domain ARI `-0.0930`，3 个 seeds 全部下降；
- spfac recovery `-0.0006`，3 个 seeds 全部下降；
- ADT nuisance protection 中位 `-0.0010`。

Human Lymph Node：

- bLISI `-0.0082`，3/3 seeds 下降；
- kBET `-0.0228`，3/3 seeds 下降；
- spatial 提升与 batch mixing 下降同时出现。

机制上的数据集尺度差异也很明显：

- Simulation/HLN 有双向 prior，可以使用 reciprocal evidence；
- MouseBrain 保持原单向 dense OT，reciprocal evidence 缺失并被记为 0；
- 这使不同数据集的绝对 matchability 数值不可直接比较。

### 5.5 决定

O1-a 的机制验证通过，Simulation 和 MouseBrain 通过，但 HLN 因 batch mixing 下降而失败。

最终：

```text
O1-a = NO-GO
不替换 G0-b
```

### 5.6 为什么过渡到 O1-b

O1-a 暴露的问题不是 matchability 完全无效，而是绝对分数存在数据集和 direction 尺度差异：

- MouseBrain 缺 reciprocal 后，大多数 spot 的绝对 matchability 偏低；
- 对全部 spot 连续缩放 OT residual，可能把中高可靠 spot 也削弱；
- 更合理的尝试是只处理每条 OT direction 内相对最差的尾部 spot。

因此进入 O1-b。

## 6. O1-b：相对可靠度选择性 OT Bypass

正式结果：

- [result_O1b_relative_tail_ot_bypass/SUMMARY.md](result_O1b_relative_tail_ot_bypass/SUMMARY.md)
- [result_O1b_relative_tail_ot_bypass/paired_deltas.csv](result_O1b_relative_tail_ot_bypass/paired_deltas.csv)

### 6.1 修改动机

O1-b 试图修复 O1-a 的两个问题：

1. 绝对 matchability 在不同数据集和 direction 间不可比；
2. reciprocal 缺失会系统性拉低 MouseBrain 分数。

### 6.2 具体修改

三个可靠度分量先在每条 OT direction 内转成经验百分位。

当 reciprocal 不可用时：

- 不再把 reciprocal 记为 0；
- 直接排除该分量；
- 在其余有效分量之间重新归一化。

相对分数映射为 residual 保留比例：

```text
relative_score <= 0.2：r = 0.25
relative_score >= 0.5：r = 1.0
中间区域：smoothstep 从 0.25 过渡到 1.0
```

因此：

- 约 20% 的低可靠尾部只保留 25% OT residual；
- 约 50% 的高可靠 spot 完整使用原 OT；
- 中间 spot 平滑过渡。

### 6.3 优势

O1-b 的机制工作完全符合设计：

- 三个数据集的低可靠尾部比例约为 20%；
- 完整 OT 比例约为 50%；
- MouseBrain 不再因 reciprocal 缺失被人为整体降权。

相对 G0-b 的局部优势：

- Simulation RNA nuisance `+0.0082`；
- Simulation ADT nuisance `+0.0166`，3/3 seeds 改善；
- HLN silhouette `+0.0016`、DBI `+0.0434`；
- MouseBrain spatial-k10 `+0.0862`；
- MouseBrain Y/Y.l1 分别 `+0.0421/+0.0380`。

### 6.4 下降与问题

相对 G0-b：

Simulation：

- Top1 `-0.0039`；
- Recall@5 `-0.0087`，3/3 seeds 下降；
- spatial-domain ARI `-0.0940`；
- spfac `-0.0061`；
- bLISI/kBET `-0.0044/-0.0090`。

Human Lymph Node：

- spatial-k10 `-0.0043`；
- bLISI `-0.0109`，3/3 seeds 下降；
- kBET `-0.0291`，3/3 seeds 下降。

MouseBrain：

- Region/annotations/celltype 分别 `-0.0015/-0.0051/-0.0003`；
- kBET `-0.0024`。

相对 O1-a 的退步更清晰：

- Simulation Top1 `-0.0280`；
- Simulation Recall@5 `-0.0507`；
- MouseBrain Region/annotations/celltype 分别 `-0.0171/-0.0111/-0.0066`；
- MouseBrain bLISI `-0.0428`；
- MouseBrain kBET `-0.0864`。

### 6.5 决定

O1-b 的分位数标定和选择性尾部机制通过，但三个筛选数据集全部失败。

最终：

```text
O1-b = NO-GO
明显不如 O1-a
```

### 6.6 为什么过渡到 O2-a

O1-a 和 O1-b 共同说明：

- 用 scalar reliability 决定每个 spot 应保留多少 OT residual，无法稳定区分“有益 OT 信息”和“有害局部改写”；
- 可靠度标定更精细，并没有解决空间域、batch 和生物语义之间的冲突；
- 问题可能不在 spot 是否可靠，而在 OT residual 内部混合了不同性质的信息。

因此 O2-a 放弃 matchability，改为按 residual 的全局/局部成分进行结构分解。

## 7. O2-a：全局/局部 OT Residual 分解

正式结果：

- [result_O2a_global_local_ot_residual/SUMMARY.md](result_O2a_global_local_ot_residual/SUMMARY.md)
- [result_O2a_global_local_ot_residual/paired_deltas.csv](result_O2a_global_local_ot_residual/paired_deltas.csv)

### 7.1 修改动机

O2-a 的核心假设是：

- OT residual 中跨所有 spot 共享的全局平移更可能代表有用的跨切片对齐；
- spot-specific 的局部 residual 更容易扭曲原有空间邻域；
- 因此可以完整保留全局成分，同时压缩局部成分。

### 7.2 具体修改

对每个 section：

```text
delta_i = legacy_OT_i - pre_OT_i

global_delta = mean_i(delta_i)

local_delta_i = delta_i - global_delta

final_i = pre_OT_i
        + global_delta
        + 0.25 × local_delta_i
```

O2-a：

- 完整保留 global residual；
- 只保留 25% local residual；
- 不使用 matchability；
- 不增加参数；
- OT prior 和 attention 计算保持不变。

### 7.3 优势

Simulation：

- Top1 `+0.0802`，3/3 seeds 改善；
- Recall@5 `+0.0934`，3/3 seeds 改善；
- FOSCTTM `+0.0023`，3/3 seeds 改善；
- spfac `+0.0027`；
- bLISI `+0.0023`；
- kBET `+0.0066`。

这是六个版本中最强的 Simulation 匹配提升。

Human Lymph Node：

- silhouette `+0.0023`；
- DBI `+0.0466`；
- spatial-k10 `+0.0355`，3/3 seeds 改善；
- 局部空间结构提升也是六个版本中最强。

机制诊断：

- local residual 的逐维中心接近 0；
- decomposed/legacy edge distortion ratio 约为 `0.19–0.31`；
- 说明 O2-a 确实显著降低了 OT 对局部空间边几何的直接改写。

### 7.4 下降与问题

Simulation：

- spatial-domain ARI `-0.0901`，3/3 seeds 下降；
- ADT nuisance protection `-0.0065`；
- seed 2026 的 spatial-domain ARI 下降达到 `-0.1470`。

这表明“局部边扭曲变小”并不保证下游空间域聚类更好。全局 residual 本身仍可能重排跨切片域关系。

Human Lymph Node：

- bLISI `-0.0289`，3/3 seeds 下降；
- kBET `-0.1002`，3/3 seeds 下降；
- 是六个版本中 HLN batch mixing 退步最严重的版本。

MouseBrain：

- Region/annotations/celltype 分别 `-0.0014/-0.0057/-0.0023`；
- spatial-k10 `-0.0014`；
- bLISI/kBET `-0.0106/-0.0131`。

MouseBrain 还出现明显的训练 seed 两极化：

- seed 123 的粗标签大幅提高；
- seed 2026 的 Region/annotations/celltype 分别约下降 `-0.2270/-0.2747/-0.2621`；
- 这说明 O2-a 的效果高度依赖训练轨迹。

### 7.5 决定

O2-a 分解机制通过，但三个数据集全部未通过完整验收。

最终：

```text
O2-a = NO-GO
匹配收益最大，但代价也很大
```

### 7.6 为什么过渡到 O2-b0

O1 和 O2-a 都直接修改了最终 OT 输出：

- O1 修改每个 spot 使用多少 residual；
- O2-a 修改 residual 的全局/局部组成。

两条路线都产生了明显的性能—空间—batch 冲突。

因此 O2-b0 改变问题位置：

- 不再修改最终 OT attention 输出；
- 完整保留 G0-b inference path；
- 只修改训练中动态 prior 的刷新来源；
- 目标是消除 `final embedding → prior → OT output → final embedding` 的自反馈闭环。

## 8. O2-b0：Detached Pre-OT Prior 刷新

正式结果：

- [result_O2b0_preOT_prior_refresh/SUMMARY.md](result_O2b0_preOT_prior_refresh/SUMMARY.md)
- [result_O2b0_preOT_prior_refresh/paired_deltas.csv](result_O2b0_preOT_prior_refresh/paired_deltas.csv)

### 8.1 修改动机

原始动态刷新使用：

```text
P_t = UOT(stop_gradient(final_embedding_t))
```

但 `final_embedding_t` 已经包含当前 OT prior 产生的 OT attention 输出，因此存在自举闭环：

```text
旧 prior
  → OT attention
  → final embedding
  → 用 final embedding 生成新 prior
  → 再影响下一轮 OT attention
```

O2-b0 希望让 prior 的更新依据不再直接包含旧 prior 的输出。

### 8.2 具体修改

唯一变化为：

```text
原版：
P_t = UOT(stop_gradient(final_embedding_t))

O2-b0：
P_t = UOT(stop_gradient(pre_OT_GraphSAGE_embedding_t))
```

保持不变：

- 初始 multimodal prior；
- UOT solver；
- attention top-K；
- 20-epoch 刷新间隔；
- 完整 G0-b OT attention 输出；
- 不使用 EMA；
- 不增加参数；
- O1 和 O2-a 分支关闭。

### 8.3 优势

Simulation：

- Recall@5 `+0.0110`；
- spfac `+0.0057`，3/3 seeds 改善；
- RNA nuisance `+0.0067`，3/3 seeds 改善；
- spatial-domain ARI 中位仅 `-0.0011`；
- bLISI `+0.0021`，3/3 seeds 改善；
- kBET `+0.0037`。

与 O2-a 相比：

- spatial-domain ARI 改善 `+0.0838`；
- spfac 改善 `+0.0086`；
- 虽然匹配增益下降，但恢复了 Simulation 的空间保护。

因此 O2-b0 是六个版本中 Simulation 综合权衡最好的版本，也是唯一正式通过 Simulation 完整验收的 O2-b 版本。

Human Lymph Node：

- spatial-k10 `+0.0040`；
- 相对 O2-a，bLISI 改善 `+0.0234`；
- 相对 O2-a，kBET 改善 `+0.0754`；
- 大幅修复了 O2-a 的 batch 崩溃。

MouseBrain：

- Region/annotations/celltype 分别 `+0.0211/+0.0202/+0.0145`；
- spatial-k8 `+0.0112`，3/3 seeds 改善；
- spatial-k10 `+0.0423`；
- bLISI `+0.0101`。

### 8.4 下降与问题

Simulation：

- Top1 中位 `-0.0006`；
- spatial-domain ARI 中位虽然接近 0，但 seed 42 下降 `-0.0960`；
- seed 稳定性仍不足。

Human Lymph Node：

- silhouette 基本不变；
- DBI `-0.0007`；
- bLISI `-0.0040`；
- kBET `-0.0137`；
- batch 代价比 O2-a 小得多，但仍未通过保护门槛。

MouseBrain：

- kBET `-0.0137`；
- 只有 1/3 seeds 的 kBET 改善；
- seed 42 的 coarse labels 和 kBET 同时下降。

刷新轨迹诊断显示：

- Simulation 的连续刷新 top-K Jaccard 约 `0.55–0.62`；
- HLN 约 `0.53–0.58`；
- MouseBrain 约 `0.48–0.56`；
- pre-OT 虽消除了直接自反馈，但 prior 在相邻刷新之间仍有明显波动。

### 8.5 决定

O2-b0：

- Simulation PASS；
- HLN FAIL；
- MouseBrain FAIL；
- 刷新机制 PASS。

最终：

```text
O2-b0 = NO-GO
但它是 O 系列最重要的折中版本和后续 O2-b/O2-c 母版
```

### 8.6 为什么过渡到 O2-b1

O2-b0 已经把问题从“修改最终输出”转移到了“稳定 prior 刷新”，但逐次刷新轨迹仍不稳定。

因此 O2-b1 的目标非常明确：

- 保留 O2-b0 的 detached pre-OT 刷新；
- 不改变 cost、solver 或输出；
- 只用 EMA teacher 平滑刷新表示；
- 判断 prior 波动是否是剩余性能冲突的主要原因。

## 9. O2-b1：EMA Pre-OT Prior 刷新

正式结果：

- [result_O2b1_ema_preOT_prior_refresh/SUMMARY.md](result_O2b1_ema_preOT_prior_refresh/SUMMARY.md)
- [result_O2b1_ema_preOT_prior_refresh/paired_deltas.csv](result_O2b1_ema_preOT_prior_refresh/paired_deltas.csv)

### 9.1 具体修改

epoch 20 第一次刷新：

```text
teacher_20 = stop_gradient(pre_OT_20)
```

后续刷新：

```text
teacher_t
  = 0.5 × teacher_(t-20)
  + 0.5 × stop_gradient(pre_OT_t)

P_t = UOT(teacher_t)
```

其中 EMA momentum 固定为 `0.5`。

以下保持不变：

- G0-b 完整 OT 输出；
- solver；
- cost；
- top-K；
- 20-epoch 刷新间隔；
- loss；
- 其他候选结构分支。

### 9.2 优势

EMA 稳定机制非常明确地生效：

- 9/9 正式运行的连续 prior Jaccard 都高于 O2-b0；
- 9/9 正式运行的 transport TV 都低于 O2-b0；
- Jaccard 提升范围约 `+0.0473～+0.1379`；
- TV 改善范围约 `+0.0456～+0.1171`；
- current-to-teacher cosine 约 `0.9875～0.9921`。

Human Lymph Node：

- DBI `+0.0301`；
- spatial-k8 `+0.0138`，3/3 seeds 改善；
- spatial-k10 `+0.0109`，3/3 seeds 改善。

MouseBrain：

- Region/annotations/celltype 分别 `+0.0168/+0.0157/+0.0199`；
- spatial-k8 `+0.0110`，3/3 seeds 改善；
- spatial-k10 `+0.1115`；
- 相对 O2-b0，bLISI `+0.0112`；
- 相对 O2-b0，kBET `+0.0300`。

### 9.3 下降与问题

相对 O2-b0，Simulation 多数关键指标反而变差：

- Top1 `-0.0048`；
- Recall@5 `-0.0052`；
- spatial-domain ARI `-0.0068`；
- spfac `-0.0047`；
- kBET `-0.0026`。

相对 G0-b：

- Simulation spatial-domain ARI `-0.0277`；
- seed 42 下降达到 `-0.1027`；
- ADT nuisance protection `-0.0028`，3/3 seeds 下降。

Human Lymph Node：

- silhouette `-0.0007`；
- bLISI `-0.0065`；
- kBET `-0.0208`；
- EMA 没有解决 spatial 提升与 batch mixing 下降的冲突。

MouseBrain 虽正式 PASS，但仍存在：

- Region 和 annotations 相对 O2-b0 略降；
- Y.l1 相对 O2-b0 `-0.0048`；
- spatial-k8 相对 O2-b0 `-0.0030`；
- 仍有明显的 seed 依赖。

### 9.4 决定

O2-b1：

- Simulation FAIL；
- HLN FAIL；
- MouseBrain PASS；
- EMA 机制和刷新稳定性 PASS。

最终：

```text
O2-b1 = NO-GO
EMA 解决了“刷新轨迹抖动”，但没有解决“优化目标冲突”
```

### 9.5 为什么过渡到 O2-c0

O2-b1 给出了一个重要的排除性结论：

```text
prior 更稳定
≠
下游效果必然更好
```

因此不再继续调整 EMA momentum 或做更多时间平滑，而是关注 prior 成本本身是否缺少局部空间结构。

O2-c0 的设计选择是：

- 不在 EMA 上继续叠加；
- 回到更直接的 O2-b0 detached pre-OT 刷新；
- 保留语义候选支持；
- 在 OT cost 内增加弱局部拓扑约束。

## 10. O2-c0：拓扑感知 OT 成本

正式结果：

- [result_O2c0_topology_aware_ot_cost/SUMMARY.md](result_O2c0_topology_aware_ot_cost/SUMMARY.md)
- [result_O2c0_topology_aware_ot_cost/paired_deltas.csv](result_O2c0_topology_aware_ot_cost/paired_deltas.csv)

### 10.1 修改动机

O2-b0/O2-b1 的动态 cost 只比较单个 spot 的 pre-OT 语义 embedding：

```text
C_semantic(i,j) = 1 - cosine(z_i, z_j)
```

这可能找到单点语义接近、但局部空间环境不一致的跨切片匹配。

由于不同切片没有预配准，不能直接比较绝对坐标。因此 O2-c0 不使用坐标距离，而是比较每个 spot 的局部邻域表示。

### 10.2 具体修改

对每个 spot：

1. 复用已有空间 KNN 图；
2. 删除 self-loop；
3. 对剩余邻居权重重新归一化；
4. 计算邻居 pre-OT embedding 的加权均值：

```text
context_i = weighted_mean(z_neighbor)
```

定义：

```text
C_semantic(i,j)
  = 1 - cosine(z_i, z_j)

C_context(i,j)
  = 1 - cosine(context_i, context_j)

C_final
  = 0.8 × C_semantic
  + 0.2 × C_context
```

Simulation 和 Human Lymph Node：

- 仍由语义 FAISS 产生 `candidate K=200`；
- 组合成本只在语义候选集合内部重排和求解；
- OT attention top-K 仍为 10。

MouseBrain：

- 在原 dense cost 中加入同样的 0.2 context 成本。

O2-c0：

- 不使用 EMA；
- 不增加参数；
- 初始 prior 不变；
- 完整 G0-b OT 输出不变。

### 10.3 优势

Simulation：

- Top1 `+0.0199`，3/3 seeds 改善；
- Recall@5 `+0.0403`，3/3 seeds 改善；
- FOSCTTM `+0.0023`，3/3 seeds 改善；
- spfac `+0.0064`，3/3 seeds 改善；
- bLISI `+0.0009`，3/3 seeds 改善；
- kBET `+0.0025`。

相对 O2-b1：

- Top1 `+0.0235`，3/3 seeds 改善；
- Recall@5 `+0.0345`，3/3 seeds 改善；
- spfac `+0.0023`。

Human Lymph Node：

- spatial-k5 `+0.0103`，3/3 seeds 改善；
- spatial-k8 `+0.0264`，3/3 seeds 改善；
- spatial-k10 `+0.0094`，3/3 seeds 改善；
- 相对 O2-b1，spatial-k5 `+0.0707`。

MouseBrain：

- Region `+0.0206`，3/3 seeds 改善；
- annotations `+0.0210`，3/3 seeds 改善；
- celltype `+0.0161`，3/3 seeds 改善；
- spatial-k10 `+0.1068`；
- bLISI `+0.0094`；
- kBET `+0.0216`。

相对 O2-b0：

- Region `+0.0095`；
- annotations `+0.0113`，3/3 seeds 改善；
- celltype `+0.0054`，3/3 seeds 改善；
- spatial-k10 `+0.0073`，3/3 seeds 改善；
- kBET `+0.0240`。

### 10.4 下降与问题

Simulation：

- spatial-domain ARI `-0.0348`；
- 只有 1/3 seeds 改善；
- 三个 seed 的变化为 `+0.0020/-0.0433/-0.0348`；
- 相对 O2-b0 spatial-domain ARI 进一步下降 `-0.0422`。

因此，拓扑成本增强了匹配和 spfac，却没有保护真正的 spatial-domain clustering。

Human Lymph Node：

- DBI `-0.0052`；
- bLISI `-0.0068`，3/3 seeds 下降；
- kBET `-0.0238`，3/3 seeds 下降；
- 三个 kBET seed 分别为 `-0.0238/-0.0320/-0.0164`。

这再次出现：

```text
空间邻域保持提高
但切片 batch mixing 下降
```

MouseBrain 虽正式 PASS，但仍有局部回退：

- seed 42 的 Y.l1 `-0.0146`；
- seed 42 的 kBET `-0.0112`；
- 相对 O2-b0，Y.l1 中位 `-0.0034`；
- 相对 O2-b0，spatial-k8 中位 `-0.0046`。

### 10.5 机制诊断

O2-c0 的拓扑成本确实生效：

- Simulation 语义/上下文成本相关系数约 `0.48–0.49`；
- HLN 约 `0.49–0.50`；
- MouseBrain 约 `0.84–0.86`；
- semantic-vs-combined top-K Jaccard：
  - Simulation 约 `0.54–0.56`；
  - HLN 约 `0.67–0.68`；
  - MouseBrain 约 `0.68–0.69`；
- 超过 92% 的 spot 至少有一个 top-K 候选位置发生变化。

这说明 0.2 context 权重不是“几乎没起作用”，而是对候选排序产生了明显影响。

### 10.6 决定

O2-c0：

- Simulation FAIL；
- HLN FAIL；
- MouseBrain PASS；
- 拓扑成本机制 PASS。

最终：

```text
O2-c0 = NO-GO
不替换 G0-b，也不替换 O2-b0 作为最平衡的 O 系列参考
```

## 11. 相邻版本是如何过渡的

### 11.1 O1-a → O1-b

过渡原因：

- O1-a 的绝对 matchability 在不同 direction 间不可比；
- MouseBrain 缺 reciprocal，绝对分数被系统性拉低；
- 希望只削弱相对最差的尾部 spot。

实际结果：

- 标定机制更合理；
- 但 Simulation 匹配、HLN 空间与 batch、MouseBrain 粗标签和 batch 全面退步；
- 说明问题不只是可靠度标定方式。

代表性相邻改善量：

| 数据集 | O1-b 相对 O1-a 的变化 |
|---|---|
| Simulation | Top1 `-0.0280`；Recall@5 `-0.0507`；spfac `-0.0056` |
| HLN | spatial-k10 `-0.0072`；bLISI `-0.0027`；kBET `-0.0063` |
| MouseBrain | Region `-0.0171`；bLISI `-0.0428`；kBET `-0.0864` |

结论：

```text
更精细的可靠度标定没有带来更好的下游表示。
```

### 11.2 O1-b → O2-a

过渡原因：

- 放弃 scalar matchability；
- 不再判断“哪个 spot 可靠”；
- 改为判断 OT residual 中“哪类信息应保留”。

代表性相邻改善量：

| 数据集 | O2-a 相对 O1-b 的变化 |
|---|---|
| Simulation | Top1 `+0.0841`；Recall@5 `+0.1179`；spfac `+0.0075` |
| HLN | DBI `+0.0556`；spatial-k10 `+0.0398`；但 kBET `-0.0639` |
| MouseBrain | spatial-k10 `-0.0629`；bLISI `-0.0119`；kBET `-0.0107` |

结论：

```text
global/local 分解恢复了对齐和 HLN 空间结构，
但显著放大了 batch 冲突和 MouseBrain 不稳定性。
```

### 11.3 O2-a → O2-b0

过渡原因：

- 停止直接修改 final OT output；
- 完整恢复 G0-b OT 输出；
- 只切断 final embedding 驱动下一轮 prior 的自反馈。

代表性相邻改善量：

| 数据集 | O2-b0 相对 O2-a 的变化 |
|---|---|
| Simulation | Top1 `-0.0808`；Recall@5 `-0.0986`；但 spatial ARI `+0.0838` |
| HLN | spatial-k10 `-0.0074`；bLISI `+0.0234`；kBET `+0.0754` |
| MouseBrain | spatial-k10 `+0.0651`；bLISI `+0.0024`；kBET `-0.0117` |

结论：

```text
O2-b0 主动牺牲 O2-a 的极强匹配，
换回更好的空间保护、batch 稳定性和 MouseBrain 空间表现。
```

这是整条路线中最关键的一次方向切换。

### 11.4 O2-b0 → O2-b1

过渡原因：

- O2-b0 的 prior 在连续刷新之间波动明显；
- 用 EMA teacher 检验“刷新抖动”是否是性能冲突根因。

代表性相邻改善量：

| 数据集 | O2-b1 相对 O2-b0 的变化 |
|---|---|
| Simulation | Top1 `-0.0048`；Recall@5 `-0.0052`；spatial ARI `-0.0068` |
| HLN | spatial-k10 `+0.0069`；bLISI `-0.0016`；kBET `+0.0004` |
| MouseBrain | celltype `+0.0054`；spatial-k10 `+0.0012`；kBET `+0.0300` |

结论：

```text
EMA 在机制上稳定了全部 9 个运行，
但效果上只明显帮助 MouseBrain，没有改善整体泛化。
```

### 11.5 O2-b1 → O2-c0

过渡原因：

- O2-b1 已证明时间平滑不是充分条件；
- 返回 O2-b0 母版；
- 在动态 OT cost 中加入局部空间上下文。

代表性相邻改善量：

| 数据集 | O2-c0 相对 O2-b1 的变化 |
|---|---|
| Simulation | Top1 `+0.0235`；Recall@5 `+0.0345`；但 spatial ARI `-0.0155` |
| HLN | spatial-k10 `+0.0011`；DBI `+0.0239`；kBET `-0.0031` |
| MouseBrain | Region `+0.0045`；annotations `+0.0065`；spatial-k10 `+0.0033` |

结论：

```text
拓扑成本重新增强了匹配、HLN 空间和 MouseBrain 综合表现，
但又加重 Simulation 空间域与 HLN batch mixing 的代价。
```

## 12. 六轮实验共同得到的结构认识

### 12.1 OT 信息本身是有用的

几乎所有版本都能在至少一个数据集上产生明确收益：

- O1-a 改善 Simulation 匹配和 MouseBrain；
- O2-a 大幅提高 Simulation 匹配和 HLN spatial；
- O2-b 系列持续改善 MouseBrain 空间或粗标签。

因此问题不是“应该删除 OT”，而是：

```text
如何让 OT 对齐、局部空间、生物状态和 batch mixing
不再由同一个表示改写承担全部责任。
```

### 12.2 单一 scalar gate 不足以解决冲突

O1-a 和 O1-b 都用一个标量控制 OT residual：

- O1-a 使用绝对 matchability；
- O1-b 使用相对 percentile。

两者都无法同时保留：

- Simulation 匹配；
- Simulation spatial domain；
- HLN batch mixing；
- MouseBrain 生物语义。

这说明冲突不是简单的“OT 用多了或用少了”。

### 12.3 减少局部边扭曲不等于空间域一定更好

O2-a 把局部 edge distortion 降低到原来的约 20%–31%，但：

- Simulation spatial-domain ARI 仍下降 `-0.0901`；
- HLN batch mixing 严重下降。

因此局部图几何只是空间表征的一部分，跨切片全局对齐也会改变空间域可分性。

### 12.4 切断 prior 自反馈是正确但不充分的

O2-b0 是最均衡的 O 系列版本：

- 完整保留 G0-b OT 输出；
- 只从 pre-OT embedding 刷新 prior；
- Simulation 空间域保护明显优于 O1/O2-a。

这说明切断自反馈是有价值的，但它没有解决 HLN batch 和 MouseBrain kBET。

### 12.5 刷新更稳定不等于效果更稳定

O2-b1 在所有 9 个运行中都改善 prior Jaccard 和 transport TV，但：

- Simulation 从 PASS 变为 FAIL；
- HLN 仍 FAIL；
- 只有 MouseBrain PASS。

因此继续只调 EMA momentum 的优先级较低。

### 12.6 在同一个 OT cost 中加入空间上下文仍会产生目标竞争

O2-c0 证明局部拓扑能够显著改变候选排序，并改善：

- Simulation 匹配；
- HLN spatial；
- MouseBrain 综合指标。

但同一个 cost 同时承担语义和拓扑目标时，仍然出现：

- Simulation spatial-domain ARI 下降；
- HLN bLISI/kBET 下降。

因此问题已从“是否需要拓扑”转变为“拓扑信息应当放在哪条输出路径中”。

## 13. 当前版本定位

### 13.1 当前正式基线

当前仍应保留：

```text
G0-b
```

原因：

- 六个 O 系列候选都未同时通过三个数据集；
- G0-b 的跨数据集表现最均衡；
- 所有 O 系列候选均保持默认关闭，不影响 G0-b 原始行为。

### 13.2 各候选版本的研究价值

| 版本 | 建议定位 |
|---|---|
| O1-a | MouseBrain 与 matchability 信息路径的强参考版本 |
| O1-b | 作为“相对分位 gate 仍不足”的负结果保留 |
| O2-a | Simulation 强匹配上界及 global/local residual 机制参考 |
| O2-b0 | O 系列最均衡的 prior-refresh 母版 |
| O2-b1 | EMA prior 稳定性参考，不建议继续单独调 momentum |
| O2-c0 | 拓扑感知 cost 的有效性参考，不作为当前部署版本 |

### 13.3 后续方向所需满足的条件

后续若继续优化 OT 结构，不应只是继续微调：

- bypass 阈值；
- local residual 比例；
- EMA momentum；
- topology cost 权重。

因为这些参数本质上仍在同一条 embedding/OT 路径上做强弱折中。

下一类结构更需要考虑：

```text
把跨切片 alignment 表示
与空间/生物状态表示
分成职责不同的输出或子空间，
再由下游任务选择或受控融合。
```

这种方向才可能避免：

- Simulation 匹配越强，spatial-domain ARI 越差；
- HLN spatial 越好，batch mixing 越差；
- MouseBrain 某些 seed 极好、另一些 seed 明显崩溃。

## 14. 正式结果与产物索引

### O1-a

- [正式总结](result_O1a_reliable_ot_bypass/SUMMARY.md)
- [逐 seed 配对差值](result_O1a_reliable_ot_bypass/paired_deltas.csv)
- [完整性审计](result_O1a_reliable_ot_bypass/integrity_audit.csv)

### O1-b

- [正式总结](result_O1b_relative_tail_ot_bypass/SUMMARY.md)
- [逐 seed 配对差值](result_O1b_relative_tail_ot_bypass/paired_deltas.csv)
- [完整性审计](result_O1b_relative_tail_ot_bypass/integrity_audit.csv)

### O2-a

- [正式总结](result_O2a_global_local_ot_residual/SUMMARY.md)
- [逐 seed 配对差值](result_O2a_global_local_ot_residual/paired_deltas.csv)
- [完整性审计](result_O2a_global_local_ot_residual/integrity_audit.csv)

### O2-b0

- [正式总结](result_O2b0_preOT_prior_refresh/SUMMARY.md)
- [逐 seed 配对差值](result_O2b0_preOT_prior_refresh/paired_deltas.csv)
- [完整性审计](result_O2b0_preOT_prior_refresh/integrity_audit.csv)

### O2-b1

- [正式总结](result_O2b1_ema_preOT_prior_refresh/SUMMARY.md)
- [逐 seed 配对差值](result_O2b1_ema_preOT_prior_refresh/paired_deltas.csv)
- [完整性审计](result_O2b1_ema_preOT_prior_refresh/integrity_audit.csv)

### O2-c0

- [正式总结](result_O2c0_topology_aware_ot_cost/SUMMARY.md)
- [逐 seed 配对差值](result_O2c0_topology_aware_ot_cost/paired_deltas.csv)
- [完整性审计](result_O2c0_topology_aware_ot_cost/integrity_audit.csv)

## 15. 最终总结

从 O1-a 到 O2-c0，这条路线依次检验了：

```text
按 spot 可靠度控制 OT
  → 只控制低可靠尾部
  → 分解 OT residual
  → 切断 prior 自反馈
  → 平滑 prior 时间变化
  → 在 prior cost 中加入局部拓扑
```

最核心的实验结论是：

1. OT 对齐是有效信号，不能简单删除；
2. O1 的 scalar gate 无法稳定分离有益与有害 OT 信息；
3. O2-a 的 global/local 分解能极大增强匹配，但不能保护空间域和 batch；
4. O2-b0 的 detached pre-OT 刷新提供了最好的综合折中；
5. O2-b1 证明刷新稳定性不是主要充分条件；
6. O2-c0 证明拓扑信息有用，但把语义和拓扑继续压进同一个 cost 仍会发生目标冲突；
7. 六个候选均不应替换 G0-b；
8. 后续若继续推进，应优先研究 alignment 与 spatial/cell-state 的职责分离，而不是继续在单一路径上做强弱微调。
