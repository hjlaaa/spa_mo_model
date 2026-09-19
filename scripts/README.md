# 可执行入口

从仓库根目录使用项目已有 Python 环境。当前共 **18 个 Python CLI**：根目录14个稳定入口，`comparison/` 两个跨方法入口，`workflows/` 两个独立工作流。开发检查位于 [tools/validation](../tools/validation/README.md)。

## Training / batch

九数据集参数与默认值由各自 runner 负责，batch 每个数据集启动独立进程。具体命令见[根 README](../README.md)。

| Dataset ID | Canonical runner |
|---|---|
| mousebrain | [run_mousebrain.py](run_mousebrain.py) |
| human_embryo | [run_human_embryo_rna_only.py](run_human_embryo_rna_only.py) |
| misar | [run_misar_seq.py](run_misar_seq.py) |
| spatch | [run_spatch.py](run_spatch.py) |
| crc | [run_crc_stereocite.py](run_crc_stereocite.py) |
| human_lymph_node | [run_human_lymph_node.py](run_human_lymph_node.py) |
| mouse_spleen | [run_mouse_spleen.py](run_mouse_spleen.py) |
| mouse_thymus | [run_mouse_thymus.py](run_mouse_thymus.py) |
| simulation | [run_simulation.py](run_simulation.py) |

- [run_experiments.py](run_experiments.py)：显式 dataset 列表、output root、preset/override。
- [train_stage_model.py](train_stage_model.py)：generic bundle / preprocess_config / synthetic / 合法 noOT / weights export。
- [run_preprocessing.py](run_preprocessing.py)：现有数据预处理 CLI，算法位于 `data_io/`。

## Evaluation

[evaluate.py](evaluate.py) 是单 run 的正式分析入口，显式指定 dataset、run_dir、output_dir、protocol 和 scope。不同数据集的科研协议保持独立。

Human Lymph Node 也支持历史 requested 协议（K=8/10/12、joint/independent、无生物学监督标签）。新实验若把独立分析目录放在 `result_*` 根目录中，需要显式指定 `--writable-result-root`；默认保持历史结果写保护。

```bash
python scripts/evaluate.py --help
python scripts/evaluate.py --dataset simulation --run-dir /path/to/run --output-dir /path/to/new_analysis --protocol requested --scope joint_per_section --k 5 8 10 12 --assignments-dir /path/to/joint_assignments
```

`joint` 合并聚类；`independent` 每 section 独立标准化和聚类；`joint_per_section` 读取已有 joint labels，按 section 计算 ARI/NMI，**不重新聚类、不跨 section 平均**。无 truth 的数据集不制造监督指标。

[analyze_simulation.py](analyze_simulation.py) 保留独有的 factor recovery、RNA/ADT nuisance leakage、相邻 section 同网格 retrieval/FOSCTTM；普通聚类用 evaluate。

```bash
python scripts/analyze_simulation.py --run-dir /path/to/simulation_run --data-dir /path/to/Simulation --output-dir /path/to/new_diagnostics
```

## Cross-method comparison

从仓库根目录用 `python -m` 执行嵌套 CLI，无额外 sys.path 设置。

| CLI | 用途 | 实现 |
|---|---|---|
| [comparison/compare_embeddings.py](comparison/compare_embeddings.py) | 对显式给出的多个方法执行保留的 standardized clustering 协议，输出各方法结果及合并表 | `analysis/method_comparison.py`、`data_io/comparison_inputs.py` 与 P6 权威函数 |
| [comparison/plot_method_umaps.py](comparison/plot_method_umaps.py) | D8b：对齐同一 reference cohort，绘制各方法 embedding / 已有 joint labels 的 UMAP | `analysis/method_umap.py`、`analysis/umap.py` |

```bash
python -m scripts.comparison.compare_embeddings --help
python -m scripts.comparison.compare_embeddings --dataset mousebrain --inputs /path/to/methods.json --output-dir /path/to/new_comparison
python -m scripts.comparison.plot_method_umaps --help
python -m scripts.comparison.plot_method_umaps --inputs /path/to/umap_inputs.json --output-dir /path/to/new_method_umaps
```

`compare_embeddings` 支持 `mousebrain / misar_seq / human_lymph_node / mouse_spleen / mouse_thymus / simulation / crc_stereocite / spatch`。Human Embryo 的专属单-run协议仍用 evaluate；其 cross-method UMAP 受下述 D8b 支持。

`methods.json` 是有序且 method 不重复的列表，例如：

```json
[
  {"method": "spa", "paths": {"run_dir": "/path/to/model_run", "data_dir": "/path/to/dataset"}},
  {"method": "cosie", "paths": {"run_dir": "/path/to/cosie_run", "data_dir": "/path/to/dataset"}}
]
```

沿用各方法现有文件格式、对齐和 cohort 检查。普通七数据集 `paths` 要求如下；`analysis_dir` 是该方法已有分析根目录，包含原格式 embedding/metadata/shared metrics 等输入。

| Method | Required paths |
|---|---|
| spa | run_dir, data_dir |
| cosie | run_dir, data_dir；HLN / Simulation 改用 analysis_dir, data_dir |
| mofa | analysis_dir, data_dir |
| spamosaic | embedding_file, data_dir, analysis_dir |

SPATCH 只保留 spa/COSIE full-spot 方法，各 entry 的 `paths` 为：run_dir、data_dir、embedding_paths（section1/section2→文件）、metadata_path、id_column、analysis_dir、batch_metrics_source。其 compare K 保持2…20、保留图K=5/8/10/12/16/20；不替换 evaluate requested K=5/8/10/12/14/16/20。

新比较入口只执行 standardized 分支，不再提供 raw-vs-standardized 比选。不同方法/数据集的 K、采样、truth、scaler scope 保持原协议；输出保留 per-method rows，不新增平均指标。标签/图目录仍按各数据集原有 retained K 保留，完整K的指标表保持；不提供维护历史输出的 prune CLI。batch metrics 沿用原共享输入文件及来源记录，不改变其输入空间或重算。缓存使用 B9 来源检查；来源不匹配时请选新的独立 output。

`umap_inputs.json` 格式如下：

```json
[
  {"method": "cosie", "dataset": "mousebrain", "reference_dir": "/path/to/reference_umap", "analysis_dir": "/path/to/cosie_analysis"}
]
```

D8b 支持 COSIE / MOFA+ / SpaMosaic 的九数据集及 Harmony 的 Human Embryo（method keys：cosie/mofa/spamosaic/harmony）。reference_dir 包含已生成的 UMAP config、sample indices、coordinates/labels；可使用下述 input-integration 输出的 `umap/`。各方法采用原格式读取和已有 joint labels。默认 UMAP n_neighbors=30、min_dist=.3、seed=42，不改变既有输入空间、采样和 panel。缺失方法输入记录 UNAVAILABLE。

## Additional workflows

| CLI | 用途 |
|---|---|
| [workflows/plot_input_integration.py](workflows/plot_input_integration.py) | D8c：九数据集输入模态 panel c 与已完成整合 embedding panel e 的 SpaMosaic 风格展示 |
| [workflows/evaluate_hln_annotations.py](workflows/evaluate_hln_annotations.py) | HLN A1 外部人工标签与五方法比较，D1不评分 |

```bash
python -m scripts.workflows.plot_input_integration --help
python -m scripts.workflows.plot_input_integration --dataset mousebrain --run-dir /path/to/run --data-dir /path/to/dataset --analysis-dir /path/to/existing_analysis --output-dir /path/to/new_panels
python -m scripts.workflows.evaluate_hln_annotations --help
python -m scripts.workflows.evaluate_hln_annotations --inputs /path/to/annotation_inputs.json --output-dir /path/to/new_annotation_metrics --dry-run
```

D8c **不运行 SpaMosaic 模型、不训练 integration**。它读取已有整合 embedding；原模态视图按原流程可能执行数据读取、PCA/Harmony 等输入准备。SPATCH 的 visualization sample feature cache 只写新的 output/input_modalities，独立于只读 model-ready `preprocessed_cache/spatch`。默认 max_samples=100000、UMAP参数同上。其 dataset key 与 compare/evaluate一致（MISAR为misar_seq，CRC为crc_stereocite）。

人工标注 JSON 示例：

```json
[
  {"method": "spa", "paths": {"run_dir": "/path/to/hln_run", "data_dir": "/path/to/hln_data"}, "analysis_dir": "/path/to/standardized_embedding", "annotation_path": "/path/to/manual_annotations.csv"}
]
```

method 支持 spa / cosie / present / mofa / spamosaic。spa 的 paths 如上；present 需要 run_dir；其余需要 embedding_file。analysis_dir 含已有 config.json、scaler_parameters.npz 和 retained labels。外部标注仍使用原 `Barcode` / `manual-anno` 列。可提供 display_name。

该工作流复用已存 scaler，以 joint A1+D1 和 independent A1 两种空间运行 K=2…12、seed=0、n_init=20、max_iter=300；检查保留K=5/8/10/12的标签重现。输出 ARI/NMI/Homogeneity/Completeness/V-measure/label-ASW/scaled label-ASW。这不改变 canonical HLN 无 ground-truth 的默认合同。`--dry-run` 会读取、计算、校验，但不写结果；它不是免计算模式。输出须全新，拒绝覆盖已有结果。

## Layout and protection

本次 v15C-2 的后台队列、固定配置与状态管理独立位于 [experiments/v15c2](../experiments/v15c2/README.md)，不放入这里的长期 CLI 或核心模块。六数据集训练入口通过共享参数支持可选 `--feature_graph`，默认关闭。

```text
scripts/
  run_*.py / train_stage_model.py  九dataset、batch、preprocessing、generic
  evaluate.py                    单run分析
  analyze_simulation.py          独有diagnostics
  comparison/
    compare_embeddings.py
    plot_method_umaps.py
  workflows/
    plot_input_integration.py
    evaluate_hln_annotations.py
```

所有输入与新输出明确分离；历史 result/cache 只读。旧 raw-vs-standardized main、版本化 UMAP / supplementary CLI 已退休，不保留转发 shim。历史证据与源码快照保留原位置。当前工具见 [tools/validation](../tools/validation/README.md)，完整迁移和验证见 [S2d REPORT](../refactor_checks/s2d_scripts_final_20260917/REPORT.md)。
