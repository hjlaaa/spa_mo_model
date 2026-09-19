# spa_mo_model 不同跨视图对比损失最终实验报告

> 报告日期：2026-07-20  
> 实验结果根目录：`result_v2_Contrastive_loss/`  
> 报告用途：归档 L0–L4 对比损失实验的最终结论，为代码回滚到仅保留 L0 的版本提供依据。  
> 说明：本报告只汇总和分析现有结果，没有重新训练、删除结果或修改模型代码。

## 1. 最终结论

本轮实验的核心结论如下。

1. **按目前完成的实验直接比较，L0（Legacy COSIE）仍是综合指标最好的版本。**
   - 在五个所有版本都完成了 seed42 的公共数据集上，L0 的综合平均名次为 **2.0722**，明显优于其余版本。
   - 在包含 CRC 的六数据集比较中，只有 L0、L1、L3-strict 具备完整 seed42 结果，排名仍是 **L0 > L3-strict > L1**。
   - L0 在内部聚类结构指标（Silhouette、Calinski–Harabasz、Davies–Bouldin）上优势最明显。

2. **L0 的效果好不等于它的对比损失数值正常。**
   - L0 的跨视图损失绝对值中位最大值约为 **197,687**，单次实验最大绝对值达到 **3,552,822**，并出现巨大的负值和跳变。
   - 这些数值是 Legacy COSIE 维度损失公式本身造成的；“训练没有 NaN/Inf”只能说明运行完成，不能证明目标函数在数值或统计意义上合理。

3. **如果要求一个数值稳定、公式清晰的替代方案，L3-strict 是目前最合适的候选。**
   - L3-strict 使用 **Symmetric InfoNCE + VICReg variance/covariance**，并对共享参数初始化和 L0 的损失权重日程进行匹配。
   - 它的跨视图损失最大绝对值约为 **8.49**，训练过程稳定。
   - MouseBrain 的三个 model seed（42、123、2026）成对比较中，L3-strict 的等权综合名次均优于 L0，尤其改善 batch、spatial 和部分有标签指标；但 L0 仍全面占优内部聚类结构指标。

4. **L1、L2、L4 均没有足够证据成为新的默认版本。**
   - L1 修正了 COSIE 损失的异常，但综合性能下降最多。
   - L2（仅 InfoNCE）与 L3、L4 接近，未形成稳定优势，且没有 CRC 结果。
   - L4（关闭跨视图梯度）表现与 L2/L3 接近，说明当前实验中的性能相当一部分来自重构、图结构、融合和 OT 等其他模块，而不是跨视图对比损失本身。

5. **因此，回滚为只保留 L0 的代码可以作为代码精简决定，但不能把 L0 的异常问题解释为已经解决。**
   - 应保留本报告和本轮结果目录，作为日后重新引入稳定跨视图损失的依据。
   - 若未来继续研究该问题，应从 L3-strict 重新开始，而不是从 L1 或 L3-plan 开始。

一句话总结：**当前实测性能选择 L0；当前稳定替代方案选择 L3-strict；研究结论是 L0 效果最好但损失异常仍然存在。**

## 2. 实验版本定义

| 代号 | 实验目录 | 跨视图损失定义 | 主要设置 | 定位 |
|---|---|---|---|---|
| L0 | `L0_legacy_strict_init_matched` | Legacy COSIE dimension loss | 保留有符号维度联合项；严格匹配初始化；MouseBrain 使用原始 warm-up 日程 | 原始基线 |
| L1 | `L1_corrected_cosie_strict_init_matched` | Corrected COSIE dimension loss | softmax 后构造非负分布；联合分布 clamp 并重新归一化；pair/section 取均值 | 对原 COSIE 公式的直接数值修正 |
| L2 | `L2_infonce_plan_schedule` | Symmetric InfoNCE | 同 spot 正样本、batch 内负样本；temperature=0.2；projection head；无 VICReg | 纯 InfoNCE 消融 |
| L3-plan | `L3_vicreg_plan_schedule` | Symmetric InfoNCE + VICReg | temperature=0.2；projection dim=64；variance=1；covariance=0.04；使用早期计划日程 | 第一版主方案，后被 strict 版取代 |
| L3-strict | `L3_vicreg_strict_init_matched` | Symmetric InfoNCE + VICReg | 私有 projection 初始化，保证共享参数与 L0 匹配；其余核心参数同 L3-plan；匹配 L0 日程 | 公平对照的稳定主方案 |
| L4 | `L4_no_crossview_matched` | 无跨视图训练梯度 | `lambda_contrast=0`；仍可计算 L3 诊断值，但不进入总损失 | 无跨视图损失的控制组 |

总损失的基本形式为：

```text
total_loss = reconstruction_and_other_losses
             + lambda_contrast * crossview_loss
```

L4 中第二项的实际训练权重为 0。L0/L1/L2/L3 之间只改变跨视图损失路径及其必要的 projection head，重构、图编码、融合和 OT 等主体模块不作为本轮修改对象。

### 2.1 损失权重和日程

- MouseBrain 的 L0、L1 和 L3-strict 使用匹配的 warm-up：
  `epoch 1–5: 1e-4`，`6–10: 3e-4`，`11–15: 1e-3`，`16–200: 1e-2`。
- 其他小数据集及 CRC 的 L0、L1、L3-strict 通常使用 `lambda_contrast=0.1`。
- L2 和 L3-plan 属于早期 plan schedule：MouseBrain 前期为 `0.001`，之后为 `0.1`，因此它们与 strict 路径并非完全等条件。
- 在最终方法判断中，应优先使用 **L3-strict**，L3-plan 只作为历史实验保留。

## 3. 实验覆盖范围

所有列入本报告的完整评估均包含 clustering seed `0,1,2,3,4`，表中的 seed 指模型训练 seed。

| 版本 | CRC | Human Lymph Node | MISAR-seq | Mouse Spleen | Mouse Thymus | MouseBrain |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 42 | 42 | 42 | 42 | 42 | 42、123、2026 |
| L1 | 42 | 42 | 42 | 42 | 42 | 42 |
| L2 | — | 42 | 42 | 42 | 42 | 42 |
| L3-plan | — | 42 | 42 | 42 | 42 | 42 |
| L3-strict | 42 | 42 | 42 | 42 | 42 | 42、123、2026 |
| L4 | — | 42 | 42 | 42 | 42 | 42 |

补充说明：MISAR-seq 下曾存在 L0 seed123 的训练目录，但没有完整评估结果，因此没有纳入任何排名。

这意味着：

- **五版本公平横向比较**只能使用五个公共小数据集的 seed42。
- **CRC 比较**只能比较 L0、L1、L3-strict 的 seed42。
- **多 model seed 稳定性比较**只能在 MouseBrain 上比较 L0 与 L3-strict。
- 不能把“缺少某数据集”的版本与六数据集完整版本直接平均后宣称总体更优。

## 4. 统一评估与排名方法

本报告重新读取当前结果目录中的有效评估结果，共核对 **185 个完整 evaluation 目录**；没有使用旧报告中的手工结论代替实际结果。

### 4.1 固定 K，不做结果后选

为避免在多个 K 中挑选表现最好的结果，带标签指标采用预先固定的端点：

- MouseBrain `RegionLoupe`：K=9
- MouseBrain `annotations`：K=11
- MISAR-seq `Y`：K=14

因此本报告的排名不是按每个方法各自最优 K 生成的。

### 4.2 指标方向

- 越高越好：Silhouette、Calinski–Harabasz、spatial、batch、ARI、NMI。
- 越低越好：Davies–Bouldin Index。

### 4.3 聚合层级

1. 每个固定 endpoint 对 clustering seed 0–4 求均值。
2. 在完全相同的 dataset / representation / target / K / metric 上对方法排名。
3. 先在数据集内部按指标 family 聚合。
4. 各 family 等权，再对数据集等权。

该方法可避免“某一类指标数量较多”自动获得更大权重。报告同时保留原始指标解读，防止只看单一综合分数。

## 5. 总体排名

### 5.1 五个公共数据集、seed42、五种 canonical 损失的公平排名

此表是回答“现有 L0–L4 中谁综合最好”的主要依据。L3 使用严格匹配版，不混入 L3-plan。

| 排名 | 版本 | 五数据集综合平均名次 | 数据集名次均值 | 结论 |
|---:|---|---:|---:|---|
| 1 | **L0** | **2.0722** | **1.8** | 明确第一，内部聚类结构优势最大 |
| 2 | L4 | 3.1042 | 3.2 | 与 L2/L3 基本同一梯队，batch 较好 |
| 3 | L2 | 3.1090 | 3.6 | 与 L4 仅差 0.0049，不能视为实质差异 |
| 4 | L3-strict | 3.1340 | 3.4 | 稳定且更均衡，与 L2 仅差 0.0250 |
| 5 | L1 | 3.5806 | 3.0 | 综合最后，主要受内部和空间指标拖累 |

严格按数字可写成：

```text
L0 > L4 > L2 > L3-strict > L1
```

但 L4、L2、L3-strict 的差距非常小，更稳妥的统计解释是：

```text
L0 > {L4 ≈ L2 ≈ L3-strict} > L1
```

### 5.2 六个数据集完整覆盖的 seed42 排名

只有 L0、L1、L3-strict 完成了 CRC，因此完整六数据集比较为：

| 排名 | 版本 | 六数据集综合平均名次 | 数据集名次均值 |
|---:|---|---:|---:|
| 1 | **L0** | **2.2245** | **2.0000** |
| 2 | **L3-strict** | **3.4010** | **3.6667** |
| 3 | **L1** | **3.8333** | **3.3333** |

该表中的名次是在所有可用端点的统一排名空间中计算，因此数值不要求落在 1–3；适合看相对顺序，不应与上一表的绝对值直接横比。

### 5.3 包含两种 L3 路径的历史六配置排名

为了完整归档，若把 L3-plan 作为独立配置加入五公共数据集 seed42 排名，结果为：

| 排名 | 配置 | 综合平均名次 |
|---:|---|---:|
| 1 | L0 | 2.3361 |
| 2 | L4 | 3.5903 |
| 3 | L2 | 3.6146 |
| 4 | L3-strict | 3.6229 |
| 5 | L3-plan | 3.6444 |
| 6 | L1 | 4.1917 |

L3-strict 略优于 L3-plan；考虑 strict 版的初始化和日程控制更公平，后续只需把 L3-strict 视为正式 L3。

## 6. 分数据集排名

下表使用五个 canonical 版本、model seed42 和五个公共数据集。括号内为该数据集的等 family 平均名次，越低越好。

| 数据集 | 第 1 | 第 2 | 第 3 | 第 4 | 第 5 |
|---|---|---|---|---|---|
| Human Lymph Node | **L0 (1.2083)** | L2 (3.0139) | L4 (3.0833) | L3-strict (3.2361) | L1 (4.4583) |
| MISAR-seq | **L0 (2.0000)** | L1 (2.9271) | L3-strict (3.2500) | L2 (3.2604) | L4 (3.5625) |
| Mouse Spleen | **L0 (1.0833)** | L2 (2.8889) | L3-strict (3.0139) | L4 (3.2500) | L1 (4.7639) |
| Mouse Thymus | **L1 (2.7500)** | L0 (2.9444) | L4 (3.0000) | L3-strict (3.0556) | L2 (3.2500) |
| MouseBrain | **L4 (2.6250)** | L1 (3.0035) | L3-strict (3.1146) | L0 (3.1250) | L2 (3.1319) |

分数据集观察：

- **Human Lymph Node：**L0 优势显著；L2、L4、L3 位于第二梯队；L1 最弱。
- **MISAR-seq：**L0 第一；L1 在该数据集相对较好；L2、L3 接近；L4 最后。
- **Mouse Spleen：**L0 优势最明显；L2 第二；L1 明显落后。
- **Mouse Thymus：**这是 L1 唯一取得综合第一的数据集，但五种方法总体差距较小。
- **MouseBrain seed42：**L4 综合第一，说明关闭跨视图梯度并不会立刻破坏该数据集表现；L0 因内部指标强、但 batch/空间及部分标签端点不占优而排在第四。该单 seed 结论必须结合第 8 节多 seed 结果理解。

## 7. CRC 结果

CRC 只完成 L0、L1 和 L3-strict 的 model seed42：

| 排名 | 版本 | CRC 综合平均名次 |
|---:|---|---:|
| 1 | **L0** | **1.6667** |
| 2 | L1 | 2.0417 |
| 3 | L3-strict | 2.2917 |

分 family 排名为：

| Family | 第 1 | 第 2 | 第 3 |
|---|---|---|---|
| Batch | **L1** | L3-strict | L0 |
| Internal | **L0** | L3-strict | L1 |
| Spatial | **L0** | L1 = L3-strict |

CRC 的结论不是“L0 所有指标都最好”，而是 L0 依靠内部结构和空间指标获得综合第一；L1 的 batch 指标最好。由于 CRC 只有一个 model seed，目前只能作为 seed42 下的性能比较，不能给出跨训练随机种子的稳定性结论。

## 8. MouseBrain 多 model seed 比较

MouseBrain 是唯一同时对 L0 和 L3-strict 完成 model seed 42、123、2026 的数据集，也是判断稳定替代方案最重要的证据。

### 8.1 每个 model seed 的成对综合名次

下表只在 L0 和 L3-strict 之间排名，并对 family 等权，因此 1.0 最好、2.0 最差。

| Model seed | L3-strict | L0 | 胜者 |
|---:|---:|---:|---|
| 42 | **1.4792** | 1.5208 | L3-strict |
| 123 | **1.3542** | 1.6458 | L3-strict |
| 2026 | **1.3438** | 1.6563 | L3-strict |
| 三 seed 均值 | **1.3924** | 1.6076 | **L3-strict** |

因此，在 MouseBrain 的多训练 seed 成对综合比较中，L3-strict 的优势是重复出现的，并非只来自 seed42。

### 8.2 Endpoint 获胜次数

| Family | L0 获胜数 | L3-strict 获胜数 | 解释 |
|---|---:|---:|---|
| Batch | 0 | **12** | L3 全胜 |
| Biology（ARI/NMI） | 9 | **15** | L3 多数获胜，但依赖标签体系与表示方式 |
| Internal | **108** | 0 | L0 全胜，是 L0 总体优势的主要来源 |
| Spatial | 7 | **29** | L3 明显占优 |

这揭示了两个版本的核心取舍：

```text
L0：更强的内部聚类几何结构
L3：更好的 batch 混合、空间保持和多数标签端点
```

### 8.3 固定 K 的 ARI/NMI 原始均值

以下数值先对 clustering seed 0–4 求均值，再统计三个 model seed 的均值和 model-seed 标准差。

| 表示 | 标签 / K | 指标 | L0 | L3-strict | 均值较优 |
|---|---|---|---:|---:|---|
| Independent | RegionLoupe / 9 | ARI | **0.4396 ± 0.0199** | 0.4375 ± 0.0059 | L0，差异很小 |
| Independent | RegionLoupe / 9 | NMI | **0.6222 ± 0.0114** | 0.6199 ± 0.0022 | L0，差异很小 |
| Independent | annotations / 11 | ARI | 0.3594 ± 0.0028 | **0.3739 ± 0.0008** | L3 |
| Independent | annotations / 11 | NMI | 0.6070 ± 0.0057 | **0.6134 ± 0.0058** | L3 |
| Joint | RegionLoupe / 9 | ARI | **0.4243 ± 0.0311** | 0.4099 ± 0.0185 | L0 |
| Joint | RegionLoupe / 9 | NMI | **0.6039 ± 0.0127** | 0.5982 ± 0.0077 | L0 |
| Joint | annotations / 11 | ARI | 0.3346 ± 0.0232 | **0.3817 ± 0.0156** | L3，提升明显 |
| Joint | annotations / 11 | NMI | 0.5663 ± 0.0067 | **0.5892 ± 0.0108** | L3 |

标签结果不能简化为“L3 的 ARI 全面更好”：

- RegionLoupe 标签下 L0 的均值略高。
- annotations 标签下 L3 在 independent 和 joint 表示上均更好，joint ARI 的提升最明显。
- L3 的 independent RegionLoupe ARI/NMI 跨 model seed 波动更小。
- 上述结果均使用固定 K，不是从多个 K 中挑选最优值。

## 9. 按指标 family 的总体表现

五公共数据集、五 canonical 版本、seed42 的 family 平均名次如下，越低越好：

| 版本 | Batch | Biology | Internal | Spatial |
|---|---:|---:|---:|---:|
| **L0** | 3.000 | **2.000** | **1.067** | **2.375** |
| L1 | 3.125 | 2.188 | 4.028 | 3.900 |
| L2 | 3.125 | 4.250 | 3.231 | 2.550 |
| L3-strict | 3.200 | 3.188 | 3.050 | 3.150 |
| L4 | **2.550** | 3.375 | 3.625 | 3.025 |

主要结论：

- L0 的第一名主要由 **internal** 指标的压倒性优势支撑，同时 biology 和 spatial 的总体名次也不差。
- L4 的 batch 平均名次最好，说明没有跨视图梯度并不等于 batch 指标必然恶化。
- L2 的 spatial 排名第二，但 biology 最差。
- L1 的 biology 排名第二，却在 internal 和 spatial 上明显落后。
- L3-strict 没有在 seed42 公共五数据集的某一个 family 中取得第一，但各项较均衡，且数值稳定。

## 10. 训练数值稳定性

所有纳入统计的训练均为 finite，没有因为 NaN/Inf 中断。但跨视图损失的量级差异非常明显。

| 版本 | 完整训练数 | 跨视图损失绝对值最大值的中位数 | 所有实验最大绝对值 | 最大单步跳变量级 | 判断 |
|---|---:|---:|---:|---:|---|
| **L0** | 8 | **197,687** | **3,552,822** | **3,550,997** | 明显异常 |
| L1 | 6 | 48.474 | 48.512 | 0.022 | 稳定 |
| L2 | 5 | 7.684 | 7.700 | 0.117 | 稳定 |
| L3-strict | 8 | 8.474 | 8.493 | 0.162 | 稳定 |
| L4 | 5 | 8.479（诊断值） | 8.533 | 0.022 | 训练权重为 0 |

L0 在各数据集观测到的最大绝对值包括：

| 数据集 / seed | L0 最大绝对值（约） |
|---|---:|
| CRC / 42 | 337,374 |
| Human Lymph Node / 42 | **3,552,822** |
| MISAR-seq / 42 | 151,439 |
| Mouse Spleen / 42 | 52,158 |
| Mouse Thymus / 42 | 54,444 |
| MouseBrain / 42 | 673,947 |
| MouseBrain / 123 | 126,640 |
| MouseBrain / 2026 | 243,936 |

原始 COSIE 代码的独立检查也确认了相同形式的损失路径，详细代码审计记录见 [COSIE 对比学习审计报告](docs/cosie_contrastive_learning_report.md)。因此，L0 的异常不是本项目重写时偶然引入的普通日志错误，而是继承自原始目标形式。

## 11. 为什么 L0 损失异常，却仍可能取得最好结果

目前证据支持以下组合解释，而不是单一原因。

### 11.1 异常项可能产生了很强的隐式正则化

L0 的有符号联合项并不是标准概率分布目标。它可能通过大幅度梯度强迫两个模态在某些维度统计上快速对齐，从而显著收紧聚类结构。这个机制在数学上不够规范，却可能恰好提高 Silhouette、CH 和 DBI。

### 11.2 总损失权重较小，主体网络仍由其他目标稳定

MouseBrain 对 L0 使用 warm-up，最终权重为 0.01；模型还有重构、图结构、融合、OT 等约束。因此跨视图项虽然日志值很大，实际参数更新仍受到其他模块和优化器共同限制，未必立即导致 NaN 或训练崩溃。

### 11.3 L0 更偏向优化内部几何，不一定更符合所有生物标签

MouseBrain 多 seed 结果显示：L0 在 internal endpoint 上 108:0 全胜，但在 batch 和 spatial 上明显落后 L3，annotations 的 ARI/NMI 也较差。也就是说，L0 的“综合好”主要反映它在内部结构指标上非常强，而不是所有评价维度都一致更好。

### 11.4 新损失可能与现有网络、权重和训练周期尚未充分适配

InfoNCE/VICReg 是合理目标，但合理公式并不保证直接替换后立刻最优。projection head、温度、负样本构成、损失权重、warm-up、训练轮数和主模型的重构目标之间均会影响最终权衡。本轮只进行了有限配置，没有开展系统超参数搜索。

### 11.5 L4 的竞争力说明跨视图损失不是当前性能的唯一来源

L4 与 L2、L3 的综合差距极小，意味着图结构、模态内重构、融合、空间邻接和 OT 已经能提供较强表示。新对比损失的有效信号可能与这些目标重复，或在当前权重下收益有限。

## 12. 各版本最终判定

| 版本 | 性能 | 稳定性 | 证据完整度 | 最终判定 |
|---|---|---|---|---|
| **L0** | 当前综合最好 | 差，存在巨大负值/跳变 | 六数据集 seed42；MouseBrain 三 seed | **保留为当前实测最强基线** |
| L1 | 综合最弱 | 稳定 | 六数据集 seed42 | 不作为默认方案，直接修正公式破坏了原有有效行为 |
| L2 | 第二梯队 | 稳定 | 五小数据集 seed42，无 CRC | 不作为默认方案；可作为纯 InfoNCE 消融归档 |
| **L3-strict** | 第二梯队；MouseBrain 多 seed 更均衡 | 稳定 | 六数据集 seed42；MouseBrain 三 seed | **最佳稳定替代方案** |
| L3-plan | 略逊 strict | 稳定 | 五小数据集 seed42，无 CRC | 被 strict 版取代，仅归档 |
| L4 | 第二梯队 | 无跨视图梯度 | 五小数据集 seed42，无 CRC | 重要控制组，不作为最终模型 |

综合排名有两种合理口径：

1. **只看当前实验的数值性能：**`L0 > L4 ≈ L2 ≈ L3-strict > L1`。
2. **同时考虑完整覆盖、数值稳定性和未来可开发性：**`L0 > L3-strict > L4 ≈ L2 > L1`。

第二种口径把 L3-strict 提升到第二位，并不是因为其五数据集综合数字高于 L2/L4，而是因为它完成了 CRC、完成了 MouseBrain 三 model seed、公式稳定且是公平初始化版本。

## 13. 对代码回滚的建议

如果当前目标是结束本轮对比损失探索、恢复代码简洁度，可以回滚到只保留 L0 的实现，但建议遵循以下边界。

### 13.1 回滚前必须保留

- 本报告：`spa_mo_model_contrastive_loss_final_experiment_report.md`
- 完整实验结果根目录：`result_v2_Contrastive_loss/`
- 各实验的配置、`run_summary.json`、`loss_history.json`、最终 embedding 和评估 CSV。
- L0/L3 MouseBrain 多 seed 结果及 CRC 的 L0/L1/L3 结果。
- 如使用 Git 回滚，先把本报告复制到不会被回滚覆盖的位置，或单独提交/打标签后再回滚。

### 13.2 可以从主代码路径移除、但结果应归档

- L1 corrected COSIE 分支。
- L2 InfoNCE-only 分支。
- L3-plan 和 L3-strict 的 projection head、VICReg 配置及选择逻辑。
- L4 no-crossview 实验分支。
- 仅用于这些分支的测试和实验参数入口。

是否删除这些代码属于后续回滚操作，本报告生成过程没有执行删除。

### 13.3 回滚后必须明确记录的已知问题

建议在 L0 代码或项目文档中至少保留以下说明：

```text
Legacy COSIE cross-view dimension loss is retained for empirical compatibility.
It can produce very large negative values and abrupt loss changes.
Finite training does not imply that this objective is numerically well behaved.
```

否则未来维护者可能再次把相同现象误判为新引入的 bug，或在没有对照的情况下重复本轮实验。

### 13.4 不建议仅保留挑选后的少量指标

若磁盘或 Git 压力较大，原始大体积中间文件可以另行归档，但不能只留下“每个方法最好的 K”。最低限度应保留：

- 固定 K 的完整评估指标；
- clustering seed 0–4 的逐 seed 结果；
- model seed、训练配置和损失历史；
- 支撑本报告表格的 run summary 与 evaluation CSV。

## 14. 结果目录索引与旧汇总注意事项

本轮核心目录位于：

- [`result_v2_Contrastive_loss/crc_stereocite/`](result_v2_Contrastive_loss/crc_stereocite/)
- [`result_v2_Contrastive_loss/human_lymph_node/`](result_v2_Contrastive_loss/human_lymph_node/)
- [`result_v2_Contrastive_loss/misar_seq/`](result_v2_Contrastive_loss/misar_seq/)
- [`result_v2_Contrastive_loss/mouse_spleen/`](result_v2_Contrastive_loss/mouse_spleen/)
- [`result_v2_Contrastive_loss/mouse_thymus/`](result_v2_Contrastive_loss/mouse_thymus/)
- [`result_v2_Contrastive_loss/mousebrain/`](result_v2_Contrastive_loss/mousebrain/)

目录中的主要版本名为：

- `L0_legacy_strict_init_matched`
- `L1_corrected_cosie_strict_init_matched`
- `L2_infonce_plan_schedule`
- `L3_vicreg_plan_schedule`
- `L3_vicreg_strict_init_matched`
- `L4_no_crossview_matched`

部分数据集还保留了 `L0_legacy_matched` 和 `L3_vicreg_matched` 等早期 pilot 目录。它们没有进入第 5.1 节的 canonical 公平排名；正式比较分别使用 `L0_legacy_strict_init_matched` 和 `L3_vicreg_strict_init_matched`。

已有的 `_comparison/` 是较早阶段生成的 L0/L2/L3/L4 汇总，未完整纳入后来补跑的 L1 和 CRC；`_comparison_l0_l1_l3_seed42/` 只覆盖 L0/L1/L3 的 seed42。**二者均不应再被当作本轮所有版本的最终总排名。**本报告使用当前文件系统中的完整结果重新聚合得出。

## 15. 局限性

- 除 MouseBrain 的 L0/L3-strict 外，其余比较只有一个 model seed，不能进行完整的训练随机性评估。
- L2、L3-plan、L4 没有 CRC，五版本排名只代表五个公共小数据集。
- L2/L3-plan 的日程与 strict 版本不同，它们主要是早期方案和消融，不是完全同条件的最终候选。
- 本轮没有进行系统的温度、projection 维度、VICReg 系数、batch 负样本规模或损失权重搜索。
- 综合排名依赖 family 等权这一明确选择；如果业务更重视 ARI/NMI 或更重视空间/batch，最终偏好可能发生变化。
- 本报告未做正式显著性检验；对 L2/L3/L4 这类极小名次差异，不应给出过强结论。
- MouseBrain 有两套标签体系，RegionLoupe 与 annotations 的结论并不完全一致，因此不能只报告其中一套标签。

## 16. 最终归档结论

本轮对比损失实验已经回答了三个问题：

1. **修改后的损失是否全面超过原始 L0？**没有。L0 仍是当前综合性能第一。
2. **是否找到数值正常且有竞争力的替代方案？**找到了，L3-strict 是最佳稳定候选，但尚未在总体性能上超过 L0。
3. **L0 的异常是否可以忽略为普通日志现象？**不可以。异常真实存在，只是现有主模型和其他损失使训练仍能完成，并且这种强约束在部分内部聚类指标上可能恰好有效。

因此，本轮最终决策建议为：

```text
当前生产/复现实测基线：L0 Legacy COSIE
稳定研究候选：L3-strict Symmetric InfoNCE + VICReg
不再继续投入：L1、L2、L3-plan、L4（保留为实验归档）
```

完成本报告归档后，可以回滚到最初只包含 L0 的代码版本；但应同时保留本报告、实验结果和“L0 存在已知数值异常”的明确记录。
