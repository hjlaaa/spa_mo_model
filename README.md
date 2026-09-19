# spa_mo_model

当前代码默认保留 **v7A** 模型基线，支持九数据集训练、显式 batch 调度和独立 analysis。v15C 的可选特征图通过 `feature_graph.enabled` / `--feature_graph` 显式启用；本次六数据集三次重复的配置与后台队列独立放在 [experiments/v15c2](experiments/v15c2/README.md)，实验已启动，实时进度见 `result_v15C-2/_control/status.json`。单数据集 runner 各解析一次配置，正式训练调用 `training/fit.py`；分析使用 `analysis/` 中的 loader、protocol、clustering、metrics 和 plotting。

## 模型合同

```text
Multimodal Encoder → Fusion → pre-OT Spatial GraphSAGE
→ bidirectional candidate-sparse UOT / OT-guided Attention
→ post-OT Spatial GraphSAGE → Decoder / final embedding
```

- 一个 Stage，默认不使用 feature graph。可选的切片内特征 KNN 与组合边权在 `model/feature_graph.py`；启用时仅影响 pre-OT GraphSAGE。v7A 研究配置的 post-OT residual scale 为 0.5，保留普通参数覆盖能力。底层/部分裸 runner 默认仍为 1.0；以下研究命令显式传入 0.5。
- 启用 OT 时只使用双向 candidate-sparse UOT；保留 FAISS IVF、Flat 和显式 blockwise backend。合法 noOT、单模态、H&E/UNI 输入仍支持。
- refresh source 为 `ot`，cost 为 0.8 semantic + 0.2 self-excluded spatial context。默认第 100 epoch 首刷，间隔 20；在 optimizer step 之后，以 eval/no_grad 单遍 forward 使用旧 prior 产生 OT embedding，再整体替换 prior。启用 feature graph 时，由 `training/graph_refresh.py` 执行两遍前向：先取 fused 表征更新特征图，再使用新图计算用于 UOT 的表征；前 99 轮特征图为空。
- dataset 的 precision、K、candidate、chunk、checkpoint、epoch 和 preprocessing 设置各自保留。MouseBrain 保留 FP32 及 runner 内 epoch0 train-mode/no_grad dry forward；MISAR 为 amp none。其余当前主链使用各自已有 BF16 设置。
- 配置优先级：基础 defaults < preset（如适用）< dataset config < 显式 CLI。权威解析工具为 `training/config.py`，dataset defaults 在各 runner 中。

**初始化与复现：** 当前 GraphSAGE 保留 v7A 初始化时的随机数消耗顺序，保证同 seed 下保留参数的初值与退休 self-path 之前一致；不恢复已删除的 self-path 模块。固定权重下计算等价不代表完整 GPU 训练逐位确定，预处理数值库线程数等环境因素也需要记录。初始化修复、真实数据对照及剩余复现限制见 [验证报告](refactor_checks/finit_v7a_rng_20260917/REPORT.md)。

## 九数据集与当前入口

下表中的训练 ID 用于 batch；analysis ID 用于 evaluate，两者在 MISAR 和 CRC 上有明确差别。

| 数据集 | 训练 ID | Canonical 单数据集脚本 | Analysis ID / 常用 protocol |
|---|---|---|---|
| MouseBrain | `mousebrain` | `scripts/run_mousebrain.py` | `mousebrain` / `requested` |
| Human Embryo | `human_embryo` | `scripts/run_human_embryo_rna_only.py` | `human_embryo` / `embryo-reference` |
| MISAR-seq | `misar` | `scripts/run_misar_seq.py` | `misar_seq` / `requested` |
| SPATCH | `spatch` | `scripts/run_spatch.py` | `spatch` / `requested` 或 `comparison` |
| CRC Stereo-CITE-seq | `crc` | `scripts/run_crc_stereocite.py` | `crc_stereocite` / `comparison` |
| Human Lymph Node | `human_lymph_node` | `scripts/run_human_lymph_node.py` | `human_lymph_node` / `requested` 或 `comparison` |
| Mouse Spleen | `mouse_spleen` | `scripts/run_mouse_spleen.py` | `mouse_spleen` / `requested` |
| Mouse Thymus | `mouse_thymus` | `scripts/run_mouse_thymus.py` | `mouse_thymus` / `requested` |
| Simulation | `simulation` | `scripts/run_simulation.py` | `simulation` / `requested` |

MouseBrain 为三 section、RNA/Metabolite/HE；Embryo 为七 section、RNA-only；MISAR 为 RNA/ATAC，训练顺序 dataset4→3→2→1；SPATCH 为两 section、RNA50/Protein15/HE50。CRC 为 CRC_003/CRC_006；HLN 为 A1/D1；Spleen 两 section；Thymus 四 section；Simulation 五 section、RNA/ADT，保留 spfac/nsfac/spatial_domain。分析 loader 保留各自原有的明确 section 顺序，不用训练顺序猜测分析行序。

## 训练

在仓库根目录、现有依赖环境中运行（本项目验证环境为 `cosie`）。以下是接口示例；先准备对应真实输入，并选择**新的**输出目录。`/path/to/...` 是需要替换的输入路径。这些示例没有在重构验收中启动真实训练。

```bash
python scripts/run_mousebrain.py --config data/configs/mousebrain_preprocess_train.json --output_dir runs/mousebrain --post_ot_graphsage_scale 0.5
python scripts/run_crc_stereocite.py --data_dir data/CRC_Stereo-CITE-seq --output_dir runs/crc --train --post_ot_graphsage_scale 0.5
python scripts/run_human_embryo_rna_only.py --train --reuse_preprocessed --preprocessed_run_dir /path/to/existing/embryo_harmony_run --require_harmony --output_dir runs/human_embryo --post_ot_graphsage_scale 0.5
python scripts/run_spatch.py --preprocessed_cache_dir /path/to/existing/spatch_model_ready_cache --output_dir runs/spatch --post_ot_graphsage_scale 0.5
```

MouseBrain JSON 中的真实输入路径须与本机数据一致。Embryo 示例读取已有外部 Harmony 预处理结果及其 manifest；仍保留 HESTA 原始数据能力。SPATCH 训练只读已有 schema=1 model-ready cache：没有自动 normalize/PCA/Harmony 或 cache build，不必读取百万 spot 原始数组来验证来源。

批量入口为 `scripts/run_experiments.py`。显式列表可选九数据集任意子集，按给定顺序分别启动独立子进程；batch 不复制 dataset defaults，也不自动运行 analysis。

```bash
python scripts/run_experiments.py --datasets crc misar --output-root runs/batch_example --post-ot-graphsage-scale 0.5 --dry-run
python scripts/run_experiments.py --datasets mousebrain spatch --output-root runs/batch_inputs --post-ot-graphsage-scale 0.5 --runner-args mousebrain '["--config","data/configs/mousebrain_preprocess_train.json"]' --runner-args spatch '["--preprocessed_cache_dir","/path/to/existing/spatch_model_ready_cache"]' --dry-run
```

`--dry-run` 只打印任务，不建输出、不启动子进程。确认输入后移除它才启动训练。运行状态写入 output root 的 `batch_status.json`；默认失败停止，`--keep-going` 可继续余下任务。跨实验 override 只在显式提供时传递。

generic 保留 bundle、`--preprocess_config`、synthetic、合法 noOT 和 weights export；weights export 不等同于精确 resume。必要预处理入口独立保留，使用前查看现有参数：

```bash
python scripts/train_stage_model.py --help
python scripts/run_preprocessing.py --help
```

## Analysis

当前入口为 `scripts/evaluate.py`。`--run-dir` 只读已有模型结果，`--data-dir` 指向必要 metadata/truth 来源，`--output-dir` 必须是独立的新分析位置；不根据 `result_*` 名称猜版本或算法。

新实验若需要将独立分析目录放在 `result_*` 根目录内，可显式传 `--writable-result-root /path/to/new/result_name` 授权该根目录。默认不授权；历史结果保护与输入/输出分离检查继续有效。v15C-2 队列只指定本次 `result_v15C-2`。

```bash
python scripts/evaluate.py --help
python scripts/evaluate.py --dataset mousebrain --run-dir runs/mousebrain --data-dir data/dataset_MouseBrain --output-dir analysis_outputs/mousebrain_requested --protocol requested --scope joint independent --joint-per-section
python scripts/evaluate.py --dataset spatch --run-dir runs/spatch --output-dir analysis_outputs/spatch_requested --protocol requested --scope joint independent
python scripts/evaluate.py --dataset human_embryo --run-dir runs/human_embryo --output-dir analysis_outputs/embryo_reference --protocol embryo-reference --scope joint independent
```

三种 scope 的含义严格分开：

| Scope | 行为 |
|---|---|
| `joint` | 合并所有 section，joint StandardScaler、聚类一次，再整体评价。 |
| `independent` | 每个 section 分别 standardize、聚类和评价。 |
| `joint_per_section` | 使用同一套已有 joint assignments 按 section 切片，与对应 truth 评价；不重新标准化或聚类。 |

`--joint-per-section` 在 joint workflow 上增加 per-section 评价。单独评价已有 assignments 时，显式给出 K 和 assignments 目录：

```bash
python scripts/evaluate.py --dataset mousebrain --run-dir runs/mousebrain --data-dir data/dataset_MouseBrain --output-dir analysis_outputs/mousebrain_joint_sections --protocol requested --scope joint_per_section --assignments-dir analysis_outputs/mousebrain_requested/standardized_embedding --k 5
```

当前 joint-per-section 参考协议只计算 **per-section ARI/NMI，不做跨 section 平均**；输出保留 dataset、scope、K、section、truth label、有效 spot 数及来源。支持范围为 MouseBrain（RegionLoupe、annotations）、MISAR、SPATCH 和 Simulation 的现有 truth 协议，以 `analysis/protocols.py` 为准；Embryo 不因有 celltype 自动扩展 supplementary 范围。CRC/HLN/Spleen/Thymus 无 ground-truth biological label，不制造 ARI/NMI。

科研协议仍各自独立：SPATCH requested K=5/8/10/12/14/16/20；comparison ALL_KS=2…20，retain=5/8/10/12/16/20，后者要求显式历史 `--batch-metrics-source` 并标记其历史来源。Mouse Thymus requested 需要已有 `--asw-sample` identity 文件。Embryo 保留 developmental section 诊断、celltype、25 叶 weighted Ward 层次结果及展示；已有 assignments 的补充展示使用 `embryo-per-section`。

B9 cache 绑定 embedding 内容/shape/dtype、section/spot 顺序、truth、必要 run/config、protocol 和 implementation identity。匹配才复用；来源不匹配或缺 manifest 时不会作为新合同有效 cache，需使用新的独立 output。历史 metrics 保留为 historical/unverified provenance，不自动覆盖或补 manifest。

跨方法比较和独立工作流现在使用职责型薄 CLI：

- `python -m scripts.comparison.compare_embeddings`：显式多方法输入，仅执行保留的 standardized clustering。
- `python -m scripts.comparison.plot_method_umaps`：D8b，共享 reference cohort 的跨方法 UMAP。
- `python -m scripts.workflows.plot_input_integration`：D8c，原模态输入 / 已有整合结果的 SpaMosaic 风格 panel；不运行 SpaMosaic 模型。
- `python -m scripts.workflows.evaluate_hln_annotations`：HLN A1 外部人工标注的五方法评估。

上述模块从仓库根目录执行，均支持 `--help`，输入路径显式提供，输出写新的独立目录。JSON格式和具体参数见 [scripts/README.md](scripts/README.md)。已有 UMAP 坐标重绘继续使用 evaluate 的 `umap-rerender`。Simulation 独有诊断继续使用 `scripts/analyze_simulation.py`。

## 代码导航

```text
model/       Stage、encoder/fusion、attention、decoder/loss、sparse UOT、FAISS、spatial_graph
              configure.py 保留已验收的模型/预处理 defaults 与字段校验
data_io/    preprocessing.py、datasets.py、image_features.py、hesta.py、common.py
              paired.py / misar.py / adapted.py 为显式数据适配
training/    config.py（解析工具）、fit.py（唯一共享训练 runtime）
analysis/    loaders / inputs、protocols、clustering、metrics / sampling / batch_metrics
              cache、plotting / umap、evaluation 及 dataset-specific 普通函数
scripts/     九 canonical runner、run_experiments、evaluate、预处理入口
              comparison/（跨方法）、workflows/（输入整合展示、外部人工标注）
tools/validation/  开发验证、smoke、C1 与只读 GPU replay（从仓库根目录用 python -m 执行）
```

SPATCH cache loader/metadata 暂留 `run_spatch.py`，各 runner 的特有 artifact 保存也保留原位，避免统一不同的文件/schema 合同。`model/`、`training/`、`data_io/` 不依赖 scripts；analysis 不依赖 training runner 或历史 suite。旧 data 模块位置不提供空转发 shim，普通调用者直接 import `data_io`；weighted spatial KNN 的唯一实现为 `model/spatial_graph.py`。

完整入口导航见 [scripts/README.md](scripts/README.md)，开发检查见 [tools/validation/README.md](tools/validation/README.md)。当前 scripts 共 **18 个 Python CLI**：根目录14个稳定入口，comparison两个，workflows两个。九dataset训练、batch、evaluate和preprocessing路径保持。

八个 raw-vs-standardized 比选实验 CLI 已退休。跨方法 reader、标准化分支和独有流程迁入 analysis/data_io，P6 clustering/metrics/plotting仍为权威实现。当前推荐CLI文件名不含实验版本号；无旧路径转发 shim。此整理未改变 batch metrics、UMAP、Human Embryo diagnostic 等科研输入空间。

S2d 证据见 [报告](refactor_checks/s2d_scripts_final_20260917/REPORT.md) 和 [最终入口审计](refactor_checks/s2d_scripts_final_20260917/S2D_FINAL_SCRIPTS_AUDIT.md)。S2c 的开发工具整理证据见 [报告](refactor_checks/s2c_validation_20260917/REPORT.md)。历史 fixture/expected/REPORT、源码快照及实验目录版本身份保持原样。

## 文档导航

历史记录与重构文档集中在 [docs/README.md](docs/README.md)，其中保留两批实验记录的来源边界：

| 位置 | 内容 |
|---|---|
| [docs/experiments_early/](docs/experiments_early/) | 原 docs 中的早期实验、方法阅读和模型验证记录，共 22 份 |
| [docs/experiments_later/](docs/experiments_later/) | 原根目录中的后续实验、跨方法比较及优化方案，共 15 份 |
| [docs/refactoring/](docs/refactoring/) | 重构审计、计划和用户决策，共 3 份 |
| [handoff.md](handoff.md) | 保留在根目录的工作交接入口 |

2026-09-17 文档归档只调整文档位置和必要导航，没有修改生产代码或重算实验。历史报告正文保持原样；旧路径与旧命令属于当时记录，当前使用方法以本 README 和 [scripts/README.md](scripts/README.md) 为准。各文件原位置、新位置以及历史相对链接的导航见 docs 总索引。`refactor_checks/` 的既有证据保持原位置，新归档核对见 [归档记录](refactor_checks/docs_archive_20260917/REPORT.md)。

## 保护与已知限制

`gene_imputation/`、`gene_imputation_spatial_smoothing/`、`gene_imputation_shared_gene_validation/` 为保护下游。`preprocessed_cache/` 和历史 `results/`、`result_*` 在重构中只读，不自动清理、重建、重算或升级 manifest。新验证只写各批独立 `refactor_checks/` 目录。

Batch 0 状态仍是 **BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY**。固定 precision/AMP/chunk/有效参数/prior/graph/fixture/RNG 下重放；FP32/BF16 固定 gate 的容差为 atol=rtol=1e-5。same-layout checkpoint 保持差0；原 cross-chunk **FAIL / KNOWN_NUMERICAL_SENSITIVITY** 不代表跨 chunk 数值等价，禁止重录 expected 或放宽容差。D2 已批准两个 self_linear state key 退休及同 seed 初始化变化；参考映射按参数名称严格核对。

以下问题仍 pending，目录整理没有修复：

| 编号 | 已知问题 |
|---|---|
| B13 | shared modality 的 target_sum 传播 |
| B14 | 自定义 section ID 与 s1/s2/... 输出 key 映射 |
| B11 | spatial graph cache identity |
| B12 | BF16 → NumPy |
| B4 | attention padding mask |
| R1 / R3 | crossview loss / Sinkhorn 的独立研究级审查 |

验收范围是小型 synthetic/sentinel 回归与冻结 GPU gate，不是九数据集长期训练或完整 analysis 的重新验收。P8 完整证据见 [REPORT.md](refactor_checks/p8_final_20260917/REPORT.md)。
