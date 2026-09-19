# v15C-2：三次独立重复实验

当前状态：**18次模型训练已跑完；SPATCH 三次训练末尾的摘要序列化错误已修复并从保存产物恢复摘要，三个分析任务已继续后台串行运行。** 实时进度以 `result_v15C-2/_control/status.json` 为准。

这里保存本次实验的配置和运行控制。删除整个 `experiments/v15c2/` 即可清理本次队列、配置和调度测试；`model/`、`training/`、`analysis/` 均不依赖此目录。实验结果和后台日志写到独立的 `result_v15C-2/`，删除调度代码不会删除结果。原 `result_v15C/` 和预处理缓存只读。

## 文件职责

```text
model/feature_graph.py          特征 KNN、缓存、空间/特征边组合
training/graph_refresh.py       两次 eval/no_grad 前向的顺序
training/fit.py                 唯一训练循环与刷新日程
experiments/v15c2/
  configs/suite.json            六数据集显式参数、三个种子、历史来源校验值
  configs/mousebrain.json       v15C 当时的 MouseBrain 输入配置副本
  plan.py                      任务清单与只读输入检查
  supervisor.py                串行子进程、锁、日志、状态、失败处理与续跑
  gpu.py                       独立进程 CUDA/FAISS GPU 检查与重试
  run.py                       默认仅打印计划的 CLI
  test_queue.py                使用模拟子进程的队列验证
```

## 模型设置

通过六数据集 runner 的 `--feature_graph` 显式开启（共享底层 parser 也支持该参数）；不传时仍为原有 v7A 路径。模型配置字段是 `feature_graph.enabled`，没有恢复已经退休的 `graph.use_feature_graph` 字段。

- 切片内的 `fused_embeddings.detach().float()` 先做 L2 归一化，再用 FAISS FlatIP 自排除近邻；`k_feature=(k_spatial+1)//2`。FAISS 的 feature graph 在 CUDA 表征上使用 GPU，独立于跨切片 UOT 的候选后端设置，与历史实现一致。
- 特征边追加权重为 `1/k_spatial`，与已经归一化的空间边拼接后再次按源节点归一化。重合边保留两份贡献，不去重、不降为 v15B 的半权重。
- 仅 pre-OT GraphSAGE 使用组合图；post-OT GraphSAGE 和 UOT 空间上下文继续使用原空间图。post-OT residual scale 为 0.5，前后不共享参数。
- epoch 1–99 不构建特征图。在 100/120/140/160/180/200 的参数更新后，第一次 eval/no_grad 前向取得 fused 表征并刷新图，第二次前向使用新图计算 `ot_embeddings`，随后刷新 UOT。新 prior 从下一轮训练使用；epoch 200 的刷新用于最终导出前向。
- 每个训练目录保存 `feature_graph_refresh.json`，记录实际刷新代数、时点、各切片邻居数和边数。运行摘要中的 `resolved_config` 保存模型开关和有效配置。

## 数据集与独立重复

共 6 × 3 = 18 次训练，每次完成后独立分析，共 36 个阶段。训练种子为 **42、43、44**，每次新建子进程、模型、优化器、特征图及 OT prior；不从前一次权重继续训练。每一轮 seed 按以下顺序访问六个数据集。

| 数据集 | epochs | 空间/特征 K | 训练精度 | joint 与 independent 分析 K |
|---|---:|---:|---|---|
| MouseBrain | 200 | 5 / 3 | FP32 | 9, 10, 11 |
| MISAR-seq | 200 | 5 / 3 | FP32 | 12, 14, 15 |
| Mouse Spleen | 200 | 10 / 5 | BF16 | 3, 5 |
| Mouse Thymus | 200 | 10 / 5 | BF16 | 5, 8, 10 |
| SPATCH | 200 | 10 / 5 | BF16 | 5, 8, 10, 12, 14, 16, 20 |
| Human Lymph Node | 200 | 10 / 5 | BF16 | 8, 10, 12 |

其余训练参数来自历史 `suite_status.json` 命令和各自 `run_summary.json`，包括 lr、loss 权重、FAISS 设置、chunk、checkpoint、Harmony 和基因数量限制。MouseBrain 的配置副本保留历史内容，其中原 `training.epochs=5/device=cpu` 会被显式 CLI 的 `200/cuda` 覆盖，和 v15C 原命令一致。

分析使用历史最终保留的 K 集合和 requested 协议，分别做 joint 与 independent 标准化。普通五数据集 KMeans seed=0、n_init=20、max_iter=300；SPATCH MiniBatchKMeans seed=42、n_init=20、max_iter=300、batch_size=4096。训练 seed 改变不改变分析聚类 seed。

Human Lymph Node 的 requested 协议现已接入公共分析入口（原 comparison 保留）；无生物学 truth 的数据集不生成 ARI/NMI。Mouse Thymus 使用历史 `results/mouse_thymus_preprocessing_comparison/shared_metrics/asw_sample.csv` 固定身份。SPATCH 只读取历史 v6 Harmony cache，启动前核对其 manifest SHA256，不重建预处理缓存。

SPATCH 缓存路径固定为 `/home/hujinlan/spa_mo_model/preprocessed_cache/spatch/v6_harmony_n50_hvg3000`。全部18次训练显式使用 `--device cuda`；MISAR 的 `faiss_device=cpu` 是历史跨切片候选检索配置，模型训练仍在 GPU，保留该配置不变。

## 查看计划：不会启动任务

从仓库根目录执行：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run --check
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run --status
```

默认命令只输出 36 项命令计划，不建结果目录、不加载数据、不启动子进程。`--check` 核对路径和小型历史/缓存 manifest；不读取真实数据矩阵、不探测 GPU。`--status` 未启动时显示 `not_started`。

仅检查 GPU（小型 CUDA 运算与 FAISS GPU 检索，不加载模型或启动训练/分析）：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run --check-gpu
```

GPU 检查最多进行3次，每次使用新 Python 进程，失败间隔5秒；全部失败则报错，绝不回退 CPU。

## 后续明确启动时使用（本轮未执行）

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run --execute --detach
```

该命令启动独立 session 的后台 supervisor，终端断开后继续运行；不依赖 cron、systemd 或 tmux。训练与分析串行运行，一次一个子进程，避免多个 SPATCH 任务争用 GPU/内存。子进程输出进入独立日志。启动会先执行上述多次 CUDA 和 FAISS GPU 检查；全部失败时停止并在 `supervisor.log` 报错，不自动改成 CPU。

```text
result_v15C-2/
  supervisor.log
  _control/
    status.json                 任务状态、命令、PID、退出码、产物验证
    suite.lock                  防止重复 supervisor
    suite_config.json           本次配置副本
    source/                     本次模型/训练/分析/调度源码副本
    logs/                       dataset_seed_stage_attempt.log
  mousebrain/seed_42/
    train_attempt_01/epochs_200/ 训练结果、刷新记录
    analysis_attempt_01/        本次训练独立分析
  misar_seq/seed_42/
    train_attempt_01/
    analysis_attempt_01/
  ...
```

每个分析目录保存本次指标、标准化配置、聚类结果与空间图。为写入本次新结果根目录，队列显式传入 `--writable-result-root .../result_v15C-2`。默认 `result_*` 写保护、输入/输出分离和分析 cache identity 检查继续有效。

默认某项失败后跳过它的依赖分析、继续后面的独立任务，最终状态为 `completed_with_failures` 且返回非零退出码。若希望遇错即停，启动时添加 `--stop-on-error`。退出码为 0 还必须通过产物检查，才记为 completed。

续跑命令：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.run --execute --detach --resume
```

续跑只跳过验证通过且未被改动的 completed 阶段。失败/中断阶段写新 `attempt_02` 等目录，保留失败现场；分析失败只重做分析，不重训已成功模型。源码或配置发生变化时拒绝混入旧队列。若上一次 supervisor 被强行杀死而训练子进程仍存活，也会拒绝重复运行。

需要停止时先读 `--status` 中的 `supervisor_pid`，对该 PID 发送 SIGTERM（例如 `kill -TERM <PID>`）。supervisor 转发 SIGTERM 给当前子进程组，最长等待 30 秒再终止，保存状态并不再开启后续任务。机器重启后需手动执行续跑命令。

## 已做验证与范围

验证命令（只用小型 CPU 前向和模拟子进程，不启动训练/数据分析）：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m unittest tools.validation.validate_feature_graph experiments.v15c2.test_queue -v
```

检查包括：历史邻居与边权规则、重复边、原子缓存刷新、两遍前向顺序、默认关闭路径与改前模型的初始化/RNG/前向等值、post-OT 空间图隔离、18 组配置、模拟失败与续跑、独立日志/输出、并发锁及结果目录授权边界。最初五次重复的历史证据见 [初始实施报告](../../refactor_checks/v15c2_implementation_20260919/REPORT.md)，当前三次重复、GPU/缓存约束与检测结果见 [本次更新记录](../../refactor_checks/v15c2_three_runs_20260919/REPORT.md)。

本轮未执行真实预处理、训练、聚类、指标计算或模型的 GPU 数值验证。2026-09-19 已在服务器沙箱外确认 RTX 4090 可见，PyTorch CUDA 运算和 FAISS GPU 检索均通过；沙箱内设备不可见，因此正式启动必须使用能访问服务器 GPU 的执行环境。不同种子的结果不要求一致，即便 seed=42，在当前重构代码和不同数值环境下也不承诺与历史 v15C 逐位相同。

特征图和 OT prior 是运行时状态，不在 weights-only `state_dict` 中。这里的续跑是队列阶段级续跑，不是从中断 epoch 精确恢复优化器；失败训练从新模型重新开始。

## SPATCH 摘要恢复与分析续跑（2026-09-19）

SPATCH seed42/43/44 均已完成200轮和最终CUDA前向，但保存摘要时，嵌套 runner 配置中的 `Path` 未被 `json_safe` 转为字符串。修复仅增加路径序列化，不改变模型或分析计算。

`recover_spatch.py` 已完整扫描已存 embedding 的形状、dtype、有限值，并记录SHA256；验证200轮loss、六次刷新、最终CUDA事件及与缓存一致的元数据后补齐摘要。原 `run_failure.json`、日志、训练 returncode=1 与原训练源码快照均保留。新摘要包含 `summary_recovery` 证据及 `retrained=false`，状态账本保留恢复前备份和原失败记录。

由于修复后的源码与原训练源码指纹不同，这次使用独立的、仅允许 SPATCH 分析的恢复入口，**不改写原 source_identity，也不绕过普通队列的源码检查**。后台日志为 `_control/spatch_recovery_supervisor.log`，仍更新同一个 `status.json`。恢复入口没有训练调用；失败分析如需重试可使用：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -m experiments.v15c2.recover_spatch --execute --detach
```

该命令在已有恢复校验记录基础上跳过完成的分析，对失败分析使用新目录；运行中拒绝重复取得队列锁。实现及验证见 [恢复报告](../../refactor_checks/spatch_summary_recovery_20260919/REPORT.md)。
