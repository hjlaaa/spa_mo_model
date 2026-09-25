# 邻域信息参与跨样本 Value 的首轮实验

主线模型通过 `ot_attention.interaction_neighbor_weight` 控制，无新版本类、无额外可学习参数。默认 `0.0`，首轮实验设 `0.25`，合法范围为有限的 `[0, 1]`。

数据流为当前 forward 的 pre-OT GraphSAGE 输出 `h` → 已有 spatial graph 的 self-excluded weighted mean `n` → `v=(1-rho)*h+rho*n` → 原 attention 的 `W_V(v[topk_idx])`。Q/K 继续使用原 `h`；candidate、top-k、prior、softmax、gate、residual、norm、多 section 更新均值和 post-OT GraphSAGE 均保留。

池化复用 `spatial_pool_self_excluded`，显式传 `detach=False, l2_normalize=False`。使用已有空间边和边权，去掉 self 后重新计算行权重和，并按原 edge batch size 分块聚合。孤立点、只有 self 的点或零权重邻域得到零向量，因此这类点的 value 为 `(1-rho)*h`。输出与 `h` 的 shape/dtype/device 相同；行权重和继续使用 FP32。不会使用 OT refresh 的 detached context，也不产生 dense 节点配对张量。

`rho=0` 直接跳过 pooling 和线性混合，attention 使用原 candidate embedding，保留原参数 schema 和初始化顺序。旧完整模型配置缺少此字段时亦按零处理。已有 OT refresh 的 pooling 默认仍 detach 并 L2 normalize；刷新 source 仍是 `ot_embeddings`，cost 仍为 `0.8 semantic + 0.2 context`。实验可能通过变化后的 OT embedding 间接改变后续 refresh 数值，这是原刷新数据流的自然结果，未新增邻域 refresh 分支。

## 配置和首轮对照

模型 JSON 中增加：

```json
{
  "ot_attention": {
    "interaction_neighbor_weight": 0.25
  }
}
```

通用训练入口通过原 `--model_config` 读取；MouseBrain 的原 `--config` JSON 将该对象置于 `model` 下。遵循基础默认 < dataset preset < JSON < 显式 CLI，CLI 的 `0` 也会覆盖 JSON 的非零值。其他原本仅支持 CLI 的数据集入口继续使用其原配置机制，没有扩展 JSON 加载接口。字段自动包含在已有 resolved `model_config` 和 audit 序列化中。

在同一份已确认的 v7A 命令/配置上只增加或替换以下参数，并选择不同 output_dir：

```text
baseline: --interaction_neighbor_weight 0.0  --post_ot_graphsage_scale 0.5
实验组:   --interaction_neighbor_weight 0.25 --post_ot_graphsage_scale 0.5
```

模型全局 post-OT scale 默认仍是 `1.0`，因此不要用新的基础默认覆盖既有 v7A 的 `0.5`。其余数据、seed、candidate、attention top-k、空间 K、AMP、chunk、checkpoint、训练和评估参数沿用同一份 v7A 设置。本次没有启动训练或完整消融。

所有训练入口支持 `--interaction_neighbor_weight`；批量入口 `scripts/run_experiments.py` 支持 `--interaction-neighbor-weight`，也可用已有 `--runner-args` 传入单个数据集参数。

## 验证

在已有 `cosie` 环境、仓库根目录运行：

```bash
python -m unittest tools.validation.test_interaction_neighbor -v
python -m tools.validation.validate_microenvironment_context --device cpu
python -m tools.validation.validate_checkpoint_ot_attention --device cuda
```

新增 8 项 unittest 覆盖：旧 attention 公式的逐元素输出/梯度/RNG 一致性，模型零权重 pooling bypass，手算均值/self 排除/空邻域，非候选空间邻居的梯度，Q/K 输入不变，CPU/CUDA FP32/BF16 和 autocast，chunk/checkpoint 输出与梯度对照，完整 CUDA BF16 loss-only 训练反传，运行期禁止 dense 节点配对张量，三 section 四方向 smoke，prior 不被 forward 修改，refresh 来源与 detach，以及配置覆盖和 audit。

本次验证结果：上述 8 项 unittest 全部通过（CPU + RTX 4090 CUDA）；既有 context 和 CUDA checkpoint 验证均 PASS。另直接加载修改前 `model_component.py`，在 CPU/CUDA × FP32/BF16 autocast × checkpoint 开/关的 8 种布局下核对零权重 attention 的输出、输入及参数梯度、CPU/CUDA RNG，全部逐元素相等。`git diff --check` 与改动 Python 文件编译检查通过。初次尝试的 `py310` 环境存在旧 PyTorch/NumPy 2 不兼容且缺少 scanpy；正式验证使用已有 `cosie` 环境，未安装或修改依赖。

## 修改文件

- `model/model_component.py`：pooling 的可选梯度路径、attention 的可选 value 输入及 chunk/checkpoint 传递。
- `model/stage_model.py`：pre-OT 邻域均值和 value 混合、双向传递、参数校验。
- `model/configure.py`：基础默认 `0.0`。
- `training/config.py`：resolved config 参数范围校验。
- `scripts/paired_cli.py`、`scripts/multisection_cli.py`、`scripts/run_mousebrain.py`、`scripts/run_spatch.py`、`scripts/run_human_embryo_rna_only.py`、`scripts/train_stage_model.py`：各入口参数解析与配置投影。
- `scripts/run_experiments.py`：批量入口透传与范围校验。
- `tools/validation/test_interaction_neighbor.py`：新增回归测试。
- `tools/validation/README.md`：验证入口索引。
- `docs/interaction_neighbor_value.md`：实验与验证说明。
