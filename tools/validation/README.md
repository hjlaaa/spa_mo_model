# 开发验证工具

从仓库根目录、已有 `cosie` 环境运行 `python -m tools.validation.<module>`。普通训练/analysis 使用 [scripts](../../scripts/README.md)，不依赖 tools。没有新增 sys.path 引导或旧路径 shim，也没有把这些开发诊断改造成 pytest。

| 模块 | 职责 / CLI |
|---|---|
| `validate_bidirectional_ot_attention` | CPU tiny 双向 OT，`--candidate_backend`、`--faiss_nlist`、`--faiss_nprobe` |
| `validate_checkpoint_ot_attention` | 同布局 checkpoint 对照，`--device cpu/cuda`、`--seed` |
| `validate_ot_attention_flow` | 普通 gate、OT attention 和 Stage 流程，固定 tiny 输入，无参数 |
| `validate_microenvironment_context` | self-excluded context / sparse refresh，`--device`、`--seed` |
| `validate_single_modality_mode` | RNA/Protein/HE 单模态与多模态回归，固定 tiny 输入，无参数 |
| `smoke_stage_model` | Stage smoke；保留既存错误消息断言失败，见下文 |
| `check_spatch_cache` | schema=1/C1 73 项；显式只读 reference 和新 output |
| `check_fixed_gpu` | frozen FP32/BF16 `check` / `diagnose-chunks`，不提供 record |
| `check_gpu_report` | 对 replay 退出状态及完整已知 cross-chunk FAIL 对象作精确核对 |
| `replay_model` | 上述 replay 的观测/RNG/state/prior helper，无独立 CLI |
| `noop_config_projection` / `retired_config_projection` | 已验收旧 fixture→当前合法 config 投影，仅验证使用 |

```bash
python -m tools.validation.validate_bidirectional_ot_attention --help
python -m tools.validation.validate_checkpoint_ot_attention --device cpu --seed 123
python -m tools.validation.validate_ot_attention_flow
python -m tools.validation.validate_microenvironment_context --device cpu --seed 2026
python -m tools.validation.validate_single_modality_mode
python -m tools.validation.check_spatch_cache --help
python -m tools.validation.check_fixed_gpu --help
```

C1 的 reference 包含冻结 `old_behavior.json`、`old_cases/` 和从旧 suite 固定命令保存的 `spatch_command.json`。命令是测试输入，不是第二份 dataset defaults；当前 batch CLI 检查与实际只读 cache loader 消费它；loader 参数由显式小 fixture 提供并与冻结命令核对。输出须选全新的独立目录：

```bash
python -m tools.validation.check_spatch_cache --reference-dir refactor_checks/s2c_validation_20260917/c1_reference --output-dir refactor_checks/new_cache_check
```

GPU报告路径的父目录需要先创建，且每份报告名必须是新的（例如先执行 `mkdir refactor_checks/new_gpu_check`）。GPU gate 使用已冻结的 environment、fixture SHA、参数 schema 和 expected；容差 atol/rtol=1e-5。从仓库根目录在有 CUDA 的已验收环境中执行，例如：

```bash
python -m tools.validation.check_fixed_gpu check --profile fp32 --baseline refactor_checks/p0_supplement/fp32_deterministic --report refactor_checks/new_gpu_check/fp32.json
python -m tools.validation.check_gpu_report --profile fp32 --process-exit-code 1 --report refactor_checks/new_gpu_check/fp32.json
python -m tools.validation.check_fixed_gpu check --profile spatch_bf16 --baseline refactor_checks/p0_supplement/spatch_bf16_deterministic --report refactor_checks/new_gpu_check/bf16.json
python -m tools.validation.check_gpu_report --profile spatch_bf16 --process-exit-code 1 --report refactor_checks/new_gpu_check/bf16.json
```

`check_fixed_gpu` 保持 exit 1：原 cross-chunk 仍为 **FAIL / KNOWN_NUMERICAL_SENSITIVITY**。只有 `check_gpu_report` 确认 13/13 fixed cases、same-layout 差0，以及完整 cross-configuration 对象与原 frozen replay 一致时，固定 gate 才通过。任意 exit 1 都不能直接当成已知失败。不得 record、重写 expected 或改变布局/AMP/容差。

旧 baseline/suite 文件已退休；历史来源快照和旧命令原文保存在 refactor_checks，无永久转发层。原冻结录制命令不是当前 CLI。当前验证数学与 S2b 接受的 replay 相同，批准的 schema/config 投影不修改原 fixture。

**既存 F-tests 问题**：`smoke_stage_model` 的 `invalid_single_modality` 文案断言仍期待 `at least two`，与当前 `single-modality mode is disabled` 不匹配。移动前后均在同一位置 FAIL，不报告为已修复；单模态专项通过。

当前批次自动回归及小 fixture/expected 继续在 refactor_checks，未创建 tests 或重写测试框架。S2c 的完整工具前后事件/数值/RNG对照、依赖审计与限制见 [REPORT](../../refactor_checks/s2c_validation_20260917/REPORT.md)。

## U7c 依赖边界

数值 replay 直接调用 `training.fit` 的 fit、forward、initial/refresh prior；不再借用 CRC/MouseBrain runner 导出的业务函数。SPATCH schema/identity checker 直接使用 `data_io.spatch_preparation` 的常量、参数合同和 strict loader，raw 禁入哨兵直接覆盖 `data_io.spatch_raw`。

`check_spatch_cache` 的两处 CLI 引用是**被测入口**：`main_case` 对当前 `run_spatch.main` 加 Stage/GPU 哨兵，`check_command` 调用当前 batch 的公开 `main(... --dry-run)` 并检查打印命令。它们不调用 runner parser/build_tasks 来取得业务参数；loader 使用小 fixture 的显式参数，并断言冻结命令的 n_comps/hvg/Harmony 与之相符。没有启动旧 suite、GPU 或训练。`replay_model.static_refresh_audit` 对当前 MouseBrain runner 只读 AST，验证它仍把任务交给公共 task；这不是运行时业务依赖。

仓库没有独立 `tests/`。历史 `refactor_checks` 的 oracle、fixture、expected 和旧 import 属于冻结证据，不批量改写；需要重放时使用其冻结环境/源码。本批接口回归放在新的 [U7c 目录](../../refactor_checks/u7c_validation_dependencies_20260922/REPORT.md)。历史 probe 使用的 `replay_model.crc/mouse` 模块别名不再是当前工具接口，当前使用 `replay_model.fit_runtime`；生产 runner 原公开兼容 API 未删除。

继续从仓库根目录以 `python -m tools.validation...` 运行；外部 cwd 可显式设置仓库根 `PYTHONPATH`。validation 源码不插入 scripts 路径，不引入安装/package 布局变化。`EXPECTED_HEAD` 的旧值仅是未使用的历史常量，不代表当前工作树、验收起点或运行限制。完整 GPU replay、已知 smoke 文案失败均不在本批重跑或修正范围。
