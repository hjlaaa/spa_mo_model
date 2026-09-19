# spa_mo_model 从 G0 到 G2-b1 的结构实验演进与效果报告

## 1. 文档目的

本文档系统记录 `spa_mo_model` 从 G0 self-path 语义消融，到 G1、G2-a、G2-b0 和 G2-b1 的结构修改、正式实验结果、优势、回退和版本过渡原因。

这几轮实验的共同目标不是修改已经确认保留的对比损失 L0，而是回答以下问题：

1. 当前 GraphSAGE 的 self-path 是否重复注入了自身信息；
2. 增加中尺度空间上下文能否改善空间结构；
3. 高频残差能否保护细粒度状态并防止过平滑；
4. 高频和低频上下文能否统一为 state/domain 双通道；
5. 自适应 router 能否缓解两个空间分支直接相加产生的冲突。

实验演进关系如下：

```text
G0-current
  ├─ G0-a：删除 adjacency self-loop                    → NO-GO
  └─ G0-b：旁路 GraphSAGE self_linear                 → GO，成为后续结构实验基线
         │
         ├─ G1：增加 rank-annulus context             → MouseBrain 有效，但 batch 退步
         │
         └─ G2-a：改为 high-frequency residual        → Simulation/HLN 有效，但跨 seed 不稳
                │
                └─ G2-b0：channel-wise 双频 state/domain 直接相加
                       │                               → 空间收益存在，但生物语义/batch 冲突
                       └─ G2-b1：逐 spot 竞争式 router
                                                       → 修复 MouseBrain 稳定性，但仍未通过
```

## 2. 统一实验口径

### 2.1 数据集与随机种子

五轮结构筛选均围绕以下三个数据集展开：

- Simulation；
- Human Lymph Node；
- MouseBrain。

正式性能实验统一使用：

- model-training seeds：`42、123、2026`；
- cluster seeds：`0、1、2、3、4`；
- 每个 model seed 训练 `200 epochs`；
- 每个 model seed 内先对 5 个 cluster seeds 取中位数；
- 再报告 3 个 model seeds 的配对中位改善量和同方向 seed 数。

### 2.2 配对比较基线

需要特别区分两类基线：

- G0 阶段：
  - G0-a 和 G0-b 均与 `G0-current` 比较；
- G1、G2-a、G2-b0、G2-b1：
  - 每轮都重新训练一组完全配对的 `G0-b`；
  - 候选结构与该轮新训练的 G0-b 比较。

因此，跨 result 目录不能直接用两张表中的 raw median 相减。本文的增减数字统一来自各轮 `paired_deltas.csv`：

```text
improvement_delta > 0：候选变好
improvement_delta < 0：候选变差
```

对 DBI、FOSCTTM 和 nuisance leakage 等“越低越好”的指标，汇总脚本已经统一转换方向。

### 2.3 固定不变的内容

从 G1 开始，以下内容均保持冻结：

- 对比损失继续使用已经验证的 legacy L0；
- Simulation/HLN 的 `lambda_contrast=0.1`；
- MouseBrain 使用原有 lambda schedule；
- 预处理、spot 顺序、OT、attention、decoder 和评价脚本不变；
- Simulation/HLN 局部空间 K 为 `10`；
- MouseBrain 局部空间 K 为 `5`；
- G1/G2 中尺度环带分别为：
  - Simulation/HLN：空间邻居排名 `11–30`；
  - MouseBrain：空间邻居排名 `6–15`。

## 3. 各版本总体结论

| 版本 | 唯一核心变化 | Simulation | Human Lymph Node | MouseBrain | 总体决定 |
|---|---|---:|---:|---:|---|
| G0-a | 删除 adjacency self-loop | 有部分提升 | 基本持平 | 粗标签明显回退 | NO-GO |
| G0-b | 旁路 `self_linear`，保留 adjacency self-loop 与 outer residual | 小幅改善 | 稳定改善 | 空间和粗标签总体改善 | GO，作为后续候选基线 |
| G1 | 增加独立 annulus context residual | FAIL | FAIL | PASS，但 batch mixing 下降 | NO-GO |
| G2-a | 增加 high-frequency residual | FAIL | PASS | FAIL | NO-GO |
| G2-b0 | channel-wise high/context 双频直接相加 | FAIL | FAIL | FAIL | NO-GO |
| G2-b1 | identity/high/context 竞争式 router | FAIL | FAIL | FAIL，但明显优于 G2-b0 | NO-GO |

这里的 GO/NO-GO 是按每轮预先固定的结构收益和回归保护门槛判断，不表示 NO-GO 版本所有指标都下降。

## 4. G0：GraphSAGE self-path 语义消融

正式结果：[result_G0_self_path_ablation/SUMMARY.md](result_G0_self_path_ablation/SUMMARY.md)

### 4.1 为什么先做 G0

原始 GraphSAGE 路径同时存在三种自身信息来源：

1. 空间邻接矩阵中的 self-loop；
2. GraphSAGE 内部的 `self_linear`；
3. GraphSAGE 外部的 residual。

这可能导致自身表示被重复注入，并掩盖真实的邻域消息贡献。因此首先不增加新模块，只消融 self-path 语义。

### 4.2 三种语义

```text
G0-current：
  adjacency self-loop
  + self_linear
  + outer residual

G0-a：
  删除 adjacency self-loop
  + 保留 self_linear
  + 保留 outer residual

G0-b：
  保留 adjacency self-loop
  + 旁路 self_linear
  + 保留 outer residual
```

### 4.3 G0-a 的优势与回退

G0-a 相对 G0-current 的代表性配对改善量：

| 数据集 | 指标 | 改善量 | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0108` | 3/3 |
| Simulation | spatial-domain ARI | `+0.1957` | 2/3 |
| MouseBrain | spatial-k8 | `+0.0743` | 3/3 |
| MouseBrain | spatial-k10 | `+0.0373` | 3/3 |
| MouseBrain | RegionLoupe ARI | `-0.0458` | 1/3 |
| MouseBrain | annotations ARI | `-0.0286` | 1/3 |
| MouseBrain | celltype ARI | `-0.0236` | 1/3 |

优势：

- 空间连续性明显增强；
- Simulation 的匹配和空间域指标都有提升；
- MouseBrain 的 batch mixing 也有一定改善。

主要问题：

- 直接删除 adjacency self-loop 后，MouseBrain 三项粗标签全部明显下降；
- 模型更依赖邻居平均，局部自身状态保护不足；
- 得到的空间平滑收益是以粗粒度生物标签回退为代价。

因此 G0-a 被判定为 NO-GO。

### 4.4 G0-b 的优势与回退

G0-b 相对 G0-current 的代表性配对改善量：

| 数据集 | 指标 | 改善量 | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0068` | 3/3 |
| Simulation | RNA nuisance leakage | `+0.0040` | 3/3 |
| Simulation | ADT nuisance leakage | `+0.0166` | 3/3 |
| Simulation | spatial-factor recovery | `-0.0054` | 0/3 |
| HLN | silhouette | `+0.0077` | 3/3 |
| HLN | DBI | `+0.1810` | 3/3 |
| HLN | spatial-k10 | `+0.0289` | 3/3 |
| MouseBrain | RegionLoupe ARI | `+0.0151` | 2/3 |
| MouseBrain | spatial-k8 | `+0.0756` | 3/3 |
| MouseBrain | spatial-k10 | `+0.0494` | 3/3 |
| MouseBrain | kBET | `-0.0165` | 1/3 |

优势：

- 保留 adjacency self-loop，使自身特征仍能通过邻接聚合进入；
- 删除额外 `self_linear`，减少重复 self-path；
- 三个数据集的整体方向最均衡；
- MouseBrain 空间连续性显著提高，同时没有出现 G0-a 的粗标签系统性回退；
- Simulation 的 nuisance leakage 和 HLN 的几何结构均有稳定改善。

主要问题：

- Simulation Top-1 只提升 `+0.0068`，仍低于实质改善目标；
- Simulation spatial-factor recovery 小幅下降；
- MouseBrain kBET 有一定回退；
- 尚不足以直接替换生产默认。

### 4.5 从 G0 到 G1 的过渡逻辑

G0 证明：

- 原始模型确实存在重复 self-path；
- G0-b 是更合理、更平衡的 GraphSAGE 语义；
- 但仅修正 self-path 仍不能解决中尺度组织结构不足。

因此后续 G1–G2 全部以 G0-b 为结构基线，不再使用 G0-current。

## 5. G1：最小多尺度 annulus context 分支

正式结果：[result_G1_minimal_multiscale_ablation/SUMMARY.md](result_G1_minimal_multiscale_ablation/SUMMARY.md)

### 5.1 结构定义

G1 在 G0-b 的局部 GraphSAGE 输出之后增加一条独立的中尺度环带分支：

```text
h_local
  = G0-b GraphSAGE(h_fused, A_local)

h_context
  = ContextBranch(h_fused, A_annulus)

h_G1
  = h_local + tanh(gate_raw) * h_context
```

其中：

- context 图与 local 图严格去重；
- context 图无 self-loop；
- gate 从 0 初始化；
- G1 不启用高频分支。

### 5.2 效果

| 数据集 | 指标 | G1 相对 G0-b | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0031` | 2/3 |
| Simulation | Recall@5 | `+0.0122` | 2/3 |
| Simulation | spatial-factor recovery | `-0.0030` | 1/3 |
| HLN | silhouette | `+0.0053` | 3/3 |
| HLN | DBI | `+0.0695` | 3/3 |
| HLN | spatial-k10 | `+0.0009` | 2/3 |
| HLN | kBET | `-0.0120` | 0/3 |
| MouseBrain | RegionLoupe ARI | `+0.0341` | 3/3 |
| MouseBrain | annotations ARI | `+0.0311` | 3/3 |
| MouseBrain | celltype ARI | `+0.0226` | 3/3 |
| MouseBrain | spatial-k10 | `+0.0363` | 3/3 |
| MouseBrain | bLISI | `-0.0282` | 0/3 |
| MouseBrain | kBET | `-0.0515` | 0/3 |

### 5.3 G1 的优势

- MouseBrain 是 G1 最明确的受益数据集：
  - 三项粗标签全部 3/3 seeds 改善；
  - spatial-k8/k10 均稳定改善；
  - 说明中尺度组织上下文对脑组织 compartment 很有效。
- HLN silhouette 和 DBI 稳定改善，说明 context 能改善局部几何。
- 环带图的无 self-loop、与局部图互斥和行归一化全部通过机制验证。

### 5.4 G1 的回退与失败原因

- Simulation Top-1 只提升 `+0.0031`，远低于目标；
- HLN 的 spatial-k10 几乎没有实质改善，kBET 越过保护线；
- MouseBrain 虽然标签和空间结构明显提升，但 bLISI/kBET 分别下降 `-0.0282/-0.0515`；
- context 让 section 内空间结构更强，同时也强化了 section-specific geometry。

机制上，G1 的单一全局标量 gate 还存在跨 seed 符号翻转：

- Simulation：`+0.0461、+0.0512、-0.0740`；
- HLN：`+0.1996、-0.1852、-0.1637`；
- MouseBrain：三个 seeds 方向一致为负。

这说明一个全局标量难以稳定表示 context 在不同数据集、不同随机种子中的作用方向。

### 5.5 从 G1 到 G2-a 的过渡逻辑

G1 表明中尺度上下文可以增强空间域，但风险是：

- 过度平滑局部状态；
- 强化 section-specific 结构；
- 无法提升 Simulation 的精确跨切片匹配。

因此下一步没有直接叠加更多 context，而是单独测试互补的高频分支，验证“局部差异与边界信息”是否比更强的平滑更重要。

## 6. G2-a：最小高频残差分支

正式结果：[result_G2a_high_frequency_ablation/SUMMARY.md](result_G2a_high_frequency_ablation/SUMMARY.md)

### 6.1 结构定义

G2-a 关闭 G1 context，只在 G0-b 上加入高频残差：

```text
high_raw
  = h_fused - A_local h_fused

high_message
  = F_high(high_raw)

h_G2a
  = h_local + tanh(gate_raw) * high_message
```

其中：

- `F_high` 是可学习投影；
- gate 从 0 初始化；
- 分支 dropout 为 0；
- 使用固定私有初始化 seed；
- G1 annulus context 完全关闭。

### 6.2 效果

| 数据集 | 指标 | G2-a 相对 G0-b | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0544` | 3/3 |
| Simulation | Recall@5 | `+0.0698` | 3/3 |
| Simulation | FOSCTTM | `+0.0023` | 3/3 |
| Simulation | RNA nuisance leakage | `+0.0040` | 2/3 |
| Simulation | spatial-domain ARI | `-0.0125` | 1/3 |
| HLN | spatial-k10 | `+0.0448` | 3/3 |
| HLN | bLISI | `+0.0159` | 3/3 |
| HLN | kBET | `+0.0275` | 2/3 |
| MouseBrain | RegionLoupe ARI | `+0.0299` | 2/3 |
| MouseBrain | annotations ARI | `+0.0312` | 2/3 |
| MouseBrain | celltype ARI | `+0.0266` | 2/3 |
| MouseBrain | spatial-k10 | `-0.0591` | 1/3 |
| MouseBrain | kBET | `-0.0323` | 0/3 |

### 6.3 G2-a 的优势

- 是目前 Simulation 跨切片精确匹配收益最强的版本：
  - Top-1 `+0.0544`；
  - Recall@5 `+0.0698`；
  - FOSCTTM 3/3 seeds 改善。
- HLN 正式通过：
  - spatial-k10 稳定提高；
  - bLISI 和 kBET 同时改善；
  - 说明高频残差在该数据集上可以增强空间结构而不破坏 batch mixing。
- MouseBrain 三项粗标签的配对中位数为正，说明高频信息确实有助于细胞状态保护。

### 6.4 G2-a 的回退与失败原因

- MouseBrain spatial-k10 明显下降 `-0.0591`；
- MouseBrain kBET 下降 `-0.0323`；
- Y.l1 细粒度指标为 `-0.0080`，0/3 seeds 改善；
- MouseBrain 三项粗标签虽然中位数为正，但 seed 2026 出现约 `-0.24～-0.29` 的大幅回退；
- Simulation spatial-domain ARI 小幅下降。

全局 gate 的符号同样不稳定：

- Simulation：`+0.1357、-0.1807、+0.1448`；
- HLN：`+0.2082、-0.2082、-0.1990`；
- MouseBrain：`-0.2304、-0.2495、+0.2488`。

因此 G2-a 的主要问题不是高频信号无效，而是：

- 一个全局 signed scalar 决定所有 spot、所有通道的高频作用；
- 不同 seed 可学习到完全相反的方向；
- 在 MouseBrain 上存在明显的随机种子崩塌。

### 6.5 从 G2-a 到 G2-b0 的过渡逻辑

G1 和 G2-a 分别证明：

- context 对 MouseBrain 的组织域有价值；
- high-frequency 对 Simulation 匹配和 HLN 有价值。

但两个版本都使用单一全局 gate，且优势分布在不同数据集。于是 G2-b0 尝试：

1. 同时保留 high/context 两种信号；
2. 用 128 维 channel-wise scale 替换不稳定的全局 scalar gate；
3. 明确输出 state/domain；
4. 去掉随机投影，减少任意旋转；
5. 从零初始化，保持与 G0-b 的精确初始等价。

## 7. G2-b0：零初始化 channel-wise 双频读出

正式结果：[result_G2b0_dual_frequency_readout/SUMMARY.md](result_G2b0_dual_frequency_readout/SUMMARY.md)

### 7.1 结构定义

```text
high_input
  = LayerNorm(h_fused - A_local h_fused)

context_input
  = LayerNorm(A_context h_fused)

delta_high
  = scale_high[channel] * high_input

delta_context
  = scale_context[channel] * context_input

state
  = h_local + delta_high

domain
  = h_local + delta_context

joint
  = h_local + delta_high + delta_context
```

其中：

- `scale_high` 和 `scale_context` 均为 128 维；
- 两组 scale 全零初始化；
- G1 和 G2-a 的旧分支关闭；
- state/domain/joint-preOT 均保存；
- 正式 go/no-go 固定使用 joint 经原 OT/decoder 后的 final embedding。

### 7.2 效果

| 数据集 | 指标 | G2-b0 相对 G0-b | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0187` | 2/3 |
| Simulation | Recall@5 | `+0.0262` | 2/3 |
| Simulation | spatial-factor recovery | `+0.0007` | 2/3 |
| Simulation | ADT nuisance leakage | `-0.0125` | 1/3 |
| HLN | silhouette | `+0.0042` | 3/3 |
| HLN | DBI | `+0.0636` | 2/3 |
| HLN | spatial-k10 | `+0.0373` | 3/3 |
| HLN | bLISI | `-0.0082` | 1/3 |
| HLN | kBET | `-0.0232` | 1/3 |
| MouseBrain | RegionLoupe ARI | `-0.0055` | 1/3 |
| MouseBrain | annotations ARI | `-0.0073` | 0/3 |
| MouseBrain | celltype ARI | `-0.0055` | 1/3 |
| MouseBrain | spatial-k10 | `+0.0173` | 3/3 |
| MouseBrain | bLISI | `-0.0482` | 1/3 |
| MouseBrain | kBET | `-0.0674` | 1/3 |

### 7.3 G2-b0 的优势

- 两个 channel-wise scale 均能稳定离开 0；
- scale 范数和残差比例比 G1/G2-a 的全局 gate 符号更稳定；
- Simulation Top-1 `+0.0187`，距离 `+0.0200` 门槛仅差 `0.0013`；
- HLN spatial-k5/k8/k10 全部 3/3 seeds 改善；
- MouseBrain spatial-k10 3/3 seeds 改善；
- 成功建立了 state/domain/joint 三套可保存的机制输出。

最终 joint 相对 local 的中位贡献比例约为：

| 数据集 | high | context | joint |
|---|---:|---:|---:|
| Simulation | `2.86%` | `1.89%` | `3.32%` |
| HLN | `8.60%` | `3.44%` | `9.24%` |
| MouseBrain | `10.69%` | `8.29%` | `13.25%` |

### 7.4 G2-b0 的回退与失败原因

- Simulation 的高频贡献比 G2-a 明显更弱，因此没有复现 G2-a 的 Top-1 大幅提升；
- Simulation ADT nuisance leakage 下降 `-0.0125`；
- HLN 虽然空间连续性提升，但 kBET 下降 `-0.0232`；
- MouseBrain 三项粗标签全部未获得稳定提升；
- MouseBrain seed 2026 三项粗标签出现约 `-0.24～-0.30` 的大幅崩塌；
- MouseBrain bLISI/kBET 回退比 G1、G2-a 更严重。

G2-b0 说明：

- channel-wise 双频参数化本身可稳定学习；
- 但 `delta_high + delta_context` 的直接相加无法自动解决目标冲突；
- 空间连续性收益可能以粗标签语义和 batch mixing 为代价。

### 7.5 从 G2-b0 到 G2-b1 的过渡逻辑

由于 G2-b0 的两个分支都有非零信号，而 joint 直接相加后出现明显冲突，G2-b1 不再修改 high/context 输入，只替换 joint 合并方式：

- state/domain 定义保持不变；
- 增加按 spot 计算的 identity/high/context 竞争式 router；
- 用 simplex 概率限制两个残差不能同时以完整强度相加；
- router 输入 stop-gradient，避免 backbone 为迎合路由而改变。

这一步只测试“直接相加是否是 G2-b0 失败的主要原因”。

## 8. G2-b1：有界逐 spot 竞争式双频 router

正式结果：[result_G2b1_competitive_router/SUMMARY.md](result_G2b1_competitive_router/SUMMARY.md)

G2-b0 对照：[result_G2b1_competitive_router/g2b1_vs_g2b0.csv](result_G2b1_competitive_router/g2b1_vs_g2b0.csv)

### 8.1 结构定义

```text
[p_identity, p_high, p_context]
  = softmax(
      Router(
        stop_gradient(LayerNorm(h_local))
      )
    )

joint
  = h_local
    + p_high * delta_high
    + p_context * delta_context
```

初始化：

```text
p_identity = 0.50
p_high     = 0.25
p_context  = 0.25
```

需要注意：

- `h_local` 始终完整保留；
- `p_identity` 的作用是占用 residual budget，而不是直接缩放 `h_local`；
- high/context scale 仍从 0 初始化；
- 初始 forward、loss 和 RNG 均与 G0-b 精确一致。

### 8.2 效果

| 数据集 | 指标 | G2-b1 相对 G0-b | 改善 seeds |
|---|---|---:|---:|
| Simulation | Top-1 | `+0.0006` | 2/3 |
| Simulation | Recall@5 | `+0.0131` | 2/3 |
| Simulation | spatial-domain ARI | `+0.1124` | 2/3 |
| Simulation | ADT nuisance leakage | `-0.0010` | 1/3 |
| HLN | silhouette | `+0.0056` | 3/3 |
| HLN | DBI | `+0.0560` | 3/3 |
| HLN | spatial-k5 | `+0.0117` | 3/3 |
| HLN | spatial-k10 | `+0.0082` | 2/3 |
| HLN | bLISI | `-0.0086` | 1/3 |
| HLN | kBET | `-0.0209` | 1/3 |
| MouseBrain | RegionLoupe ARI | `+0.0235` | 3/3 |
| MouseBrain | annotations ARI | `+0.0259` | 3/3 |
| MouseBrain | celltype ARI | `+0.0228` | 3/3 |
| MouseBrain | spatial-k10 | `+0.0314` | 2/3 |
| MouseBrain | bLISI | `-0.0135` | 0/3 |
| MouseBrain | kBET | `-0.0226` | 0/3 |

### 8.3 G2-b1 的优势

G2-b1 对 G2-b0 的最主要修复发生在 MouseBrain：

| 指标 | G2-b0 改善量 | G2-b1 改善量 | G2-b1 相对变化 |
|---|---:|---:|---:|
| RegionLoupe ARI | `-0.0055` | `+0.0235` | `+0.0290` |
| annotations ARI | `-0.0073` | `+0.0259` | `+0.0331` |
| celltype ARI | `-0.0055` | `+0.0228` | `+0.0283` |
| spatial-k10 | `+0.0173` | `+0.0314` | `+0.0140` |
| bLISI | `-0.0482` | `-0.0135` | 回退收窄 `0.0347` |
| kBET | `-0.0674` | `-0.0226` | 回退收窄 `0.0447` |

具体优势：

- MouseBrain 三项粗标签全部变为 3/3 seeds 改善；
- G2-b0 的 seed 2026 粗标签崩塌被消除；
- MouseBrain 空间连续性继续提高；
- MouseBrain batch mixing 的回退幅度明显小于 G2-b0；
- Simulation spatial-domain ARI 从 G2-b0 的 `-0.0098` 改为 `+0.1124`；
- Simulation ADT nuisance 回退从 `-0.0125` 收窄到 `-0.0010`。

### 8.4 G2-b1 的回退与失败原因

竞争式 residual budget 同时压弱了有用信号：

- Simulation Top-1 从 G2-b0 的 `+0.0187` 降为 `+0.0006`；
- Simulation Recall@5 从 `+0.0262` 降为 `+0.0131`；
- HLN spatial-k10 从 `+0.0373` 降为 `+0.0082`；
- HLN kBET 仍下降 `-0.0209`；
- MouseBrain bLISI/kBET 虽明显修复，但仍分别下降 `-0.0135/-0.0226`，未通过保护线；
- MouseBrain Y.l1 仍为 `-0.0054`。

### 8.5 Router 实际学习到的模式

跨三个 model seeds 的中位路由概率：

| 数据集 | identity | high | context | router entropy | 最大概率 >0.95 的 spot 比例 |
|---|---:|---:|---:|---:|---:|
| Simulation | `0.4163` | `0.3283` | `0.2406` | `0.5318` | `17.56%` |
| HLN | `0.0086` | `0.9811` | `0.0131` | `0.0940` | `93.71%` |
| MouseBrain | `0.0170` | `0.3142` | `0.6691` | `0.4660` | `16.16%` |

这说明：

- Simulation 中 router 相对均衡，但残差过弱，跨切片匹配收益不足；
- HLN 中 router 几乎完全坍缩到 high 路径；
- MouseBrain 中 router 明显偏向 context；
- 不同 section 之间的平均路由差异总体较小，router 更像学习了数据集/seed 级策略，而没有形成足够丰富、可复现的逐 spot 模式。

Router section 诊断：

- [router_section_diagnostics.csv](result_G2b1_competitive_router/router_section_diagnostics.csv)
- [router_section_range_summary.csv](result_G2b1_competitive_router/router_section_range_summary.csv)

### 8.6 G2-b1 的最终判断

G2-b1 不是全面退步：

- 它证明了有界合并可以修复 G2-b0 的 MouseBrain 随机种子崩塌；
- 也证明了高频/context 对不同数据集确实具有不同作用。

但 G2-b1 仍不能替换 G0-b：

- Simulation 的主要匹配收益被过度抑制；
- HLN 和 MouseBrain 的 batch mixing 仍未守住；
- router 出现明显数据集级偏置和 HLN 路径坍塌；
- 三个数据集均未通过预设门槛。

## 9. 跨版本效果对照

### 9.1 Simulation：跨切片匹配与空间域的冲突

| 版本 | Top-1 | Recall@5 | spatial-domain ARI | ADT nuisance |
|---|---:|---:|---:|---:|
| G0-a | `+0.0108` | `+0.0247` | `+0.1957` | `+0.0156` |
| G0-b | `+0.0068` | `+0.0158` | `+0.0137` | `+0.0166` |
| G1 | `+0.0031` | `+0.0122` | `-0.0002` | `-0.0057` |
| G2-a | **`+0.0544`** | **`+0.0698`** | `-0.0125` | `-0.0014` |
| G2-b0 | `+0.0187` | `+0.0262` | `-0.0098` | `-0.0125` |
| G2-b1 | `+0.0006` | `+0.0131` | **`+0.1124`** | `-0.0010` |

主要结论：

- 高频投影 G2-a 最有利于精确跨切片匹配；
- G2-b1 最有利于空间域，但牺牲了 Top-1；
- 当前结构尚未同时保留两类收益。

### 9.2 Human Lymph Node：结构改善与 batch mixing 的冲突

| 版本 | silhouette | DBI | spatial-k10 | bLISI | kBET |
|---|---:|---:|---:|---:|---:|
| G0-a | `+0.0009` | `+0.0089` | `+0.0006` | `-0.0006` | `-0.0025` |
| G0-b | `+0.0077` | **`+0.1810`** | `+0.0289` | `+0.0053` | `+0.0136` |
| G1 | `+0.0053` | `+0.0695` | `+0.0009` | `-0.0011` | `-0.0120` |
| G2-a | `-0.0036` | `+0.0264` | **`+0.0448`** | **`+0.0159`** | **`+0.0275`** |
| G2-b0 | `+0.0042` | `+0.0636` | `+0.0373` | `-0.0082` | `-0.0232` |
| G2-b1 | `+0.0056` | `+0.0560` | `+0.0082` | `-0.0086` | `-0.0209` |

主要结论：

- G0-b 本身已经是 HLN 上非常强且平衡的结构；
- G2-a 是唯一在增强 spatial-k10 的同时改善 bLISI/kBET 的新增分支；
- 双频直接相加或竞争路由都未复现 G2-a 的 batch 优势。

### 9.3 MouseBrain：粗标签、空间连续性与 batch mixing 的冲突

| 版本 | RegionLoupe | annotations | celltype | spatial-k10 | bLISI | kBET |
|---|---:|---:|---:|---:|---:|---:|
| G0-a | `-0.0458` | `-0.0286` | `-0.0236` | `+0.0373` | `+0.0140` | `+0.0465` |
| G0-b | `+0.0151` | `+0.0000` | `+0.0040` | `+0.0494` | `-0.0030` | `-0.0165` |
| G1 | **`+0.0341`** | `+0.0311` | `+0.0226` | `+0.0363` | `-0.0282` | `-0.0515` |
| G2-a | `+0.0299` | **`+0.0312`** | **`+0.0266`** | `-0.0591` | `-0.0063` | `-0.0323` |
| G2-b0 | `-0.0055` | `-0.0073` | `-0.0055` | `+0.0173` | `-0.0482` | `-0.0674` |
| G2-b1 | `+0.0235` | `+0.0259` | `+0.0228` | `+0.0314` | `-0.0135` | `-0.0226` |

主要结论：

- G1 的 context 最有利于 MouseBrain 粗标签和空间域，但 batch 代价明显；
- G2-a 的高频分支有利于粗标签，却损害 spatial-k10；
- G2-b0 没有保住两者各自优势；
- G2-b1 是目前 MouseBrain 上最均衡的 G2 版本，但 batch 指标仍未过线；
- G0-b 仍是整体风险最低的版本。

## 10. 这条实验链得到的结构认识

### 10.1 已经确认的有效经验

1. 原始 self-path 确实冗余，G0-b 比 G0-current 更合理。
2. 中尺度 context 对 MouseBrain 组织域有效。
3. 高频分支对 Simulation 跨切片匹配和 HLN 有效。
4. state/domain 双频信号均能稳定学习，不是无效分支。
5. 有界 router 可以减少极端 seed 崩塌。

### 10.2 尚未解决的核心问题

1. 空间结构增强经常伴随 batch mixing 下降。
2. 当前 reconstruction + L0 目标没有直接鼓励 router 保护 batch-invariant geometry。
3. 一个通用 joint embedding 被同时要求承担：
   - 细胞状态；
   - 组织空间域；
   - 跨 section mixing；
   - 模态融合与重构。
4. G2-b1 router 更容易学习数据集级偏好，而不是稳定的 spot-level 质量策略。
5. 简单继续调 router 温度、entropy 或 K，可能变成针对三个筛选数据集的超参数修补。

### 10.3 当前版本选择

- 后续通用基线继续使用 G0-b；
- G1、G2-a、G2-b0、G2-b1 均保留为机制实验结果；
- 不将任何 G2 版本设为当前通用默认；
- 不立即进行 G1+G2 或 G2+其他模块组合；
- G2-b1 在 MouseBrain 上的稳定化经验可以在后续 Fusion/decoder 模块单独通过后再考虑复用。

## 11. 下一阶段建议

经过连续四轮 post-fusion 空间分支实验，下一步不建议继续做 G2-b2 的 router、温度或 K 调参。

更优先的方向是从 G0-b 出发，进行 F1-a 最小质量感知 Fusion 校正：

```text
z_legacy
  = 当前 FusionMLP({z_cv_m})

z_quality
  = Σ_m quality_weight_m * z_cv_m

z_fused
  = z_legacy
    + ZeroInitDeltaFuse(
        z_legacy,
        z_quality - z_legacy,
        quality_features,
        observed_mask
      )
```

这样可以先解决当前 Fusion 中“所有模态固定等权写入 shared embedding”的问题，同时保持：

- L0 不变；
- G0-b 不变；
- 图结构不变；
- OT/decoder 不变；
- candidate 初始输出与 G0-b 精确一致。

只有 F1 单模块通过后，才应重新讨论它能否为 G2 的空间收益提供 batch 保护。

## 12. 正式结果目录

- G0：[result_G0_self_path_ablation](result_G0_self_path_ablation)
- G1：[result_G1_minimal_multiscale_ablation](result_G1_minimal_multiscale_ablation)
- G2-a：[result_G2a_high_frequency_ablation](result_G2a_high_frequency_ablation)
- G2-b0：[result_G2b0_dual_frequency_readout](result_G2b0_dual_frequency_readout)
- G2-b1：[result_G2b1_competitive_router](result_G2b1_competitive_router)

各目录均保留：

- `integrity_audit.csv`；
- `cluster_seed_metrics.csv`；
- `model_seed_metrics.csv`；
- `variant_summary.csv`；
- `paired_deltas.csv`；
- 对应的 `SUMMARY.md`。
