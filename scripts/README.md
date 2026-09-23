# 可执行入口

从仓库根目录使用项目已有 Python 环境。下列 /path/to 路径均为待替换模板，输出必须选择新的不存在目录；所有示例未使用overwrite。当前共 **18 个 Python CLI**：根目录14个稳定入口，`comparison/` 两个跨方法入口，`workflows/` 两个独立工作流。开发检查位于 [tools/validation](../tools/validation/README.md)。

## Training / batch

九数据集默认值由 `training/entry_defaults.py` 提供，模型配置由 `training/config.py` 解析；各 CLI 保留原参数映射。batch 每个数据集启动独立进程。具体命令见[根 README](../README.md)。

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

- [run_experiments.py](run_experiments.py)：显式 dataset 列表、output root、原样传递的 `--runner-args` 与共享 override；不调用历史 suite。
- [train_stage_model.py](train_stage_model.py)：generic bundle / preprocess_config / synthetic / 合法 noOT / weights export。
- [run_preprocessing.py](run_preprocessing.py)：现有数据预处理 CLI，算法位于 `data_io/`。

## Evaluation

[evaluate.py](evaluate.py) 是单 run 的正式分析入口，显式指定 dataset、run_dir、output_dir、protocol 和 scope。不同数据集的科研协议保持独立。

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

从仓库根目录用 `python -m` 执行嵌套 CLI，无需将 scripts 子目录加入 sys.path；外部工作目录须显式提供仓库根 PYTHONPATH。

| CLI | 用途 | 实现 |
|---|---|---|
| [comparison/compare_embeddings.py](comparison/compare_embeddings.py) | 对显式给出的多个方法执行保留的 standardized clustering 协议，输出各方法结果及合并表 | `analysis/method_comparison.py`、公共 evaluation/exact workflow 与中立 input readers |
| [comparison/plot_method_umaps.py](comparison/plot_method_umaps.py) | 对齐同一 reference cohort，绘制各方法 embedding / 已有 joint labels 的 UMAP | `analysis/method_umap.py`、公共 `analysis/umap_workflow.py` 与 `analysis/umap.py` |

```bash
python -m scripts.comparison.compare_embeddings --help
python -m scripts.comparison.compare_embeddings --dataset mousebrain --inputs /path/to/methods.json --output-dir /path/to/new_comparison
python -m scripts.comparison.plot_method_umaps --help
python -m scripts.comparison.plot_method_umaps --inputs /path/to/umap_inputs.json --output-dir /path/to/new_method_umaps
```

`compare_embeddings` 支持 `mousebrain / misar_seq / human_lymph_node / mouse_spleen / mouse_thymus / simulation / crc_stereocite / spatch`。Human Embryo 的专属单-run协议仍用 evaluate；其 cross-method UMAP 受下述method UMAP支持。

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

method UMAP支持 COSIE / MOFA+ / SpaMosaic 的九数据集及 Harmony 的 Human Embryo（method keys：cosie/mofa/spamosaic/harmony）。reference_dir 包含已生成的 UMAP config、sample indices、coordinates/labels；可使用下述 input-integration 输出的 `umap/`。各方法采用原格式读取和已有 joint labels。默认 UMAP n_neighbors=30、min_dist=.3、seed=42，不改变既有输入空间、采样和 panel。缺失方法输入记录 UNAVAILABLE。

## Additional workflows

| CLI | 用途 |
|---|---|
| [workflows/plot_input_integration.py](workflows/plot_input_integration.py) | 九数据集输入模态 panel c 与已完成整合 embedding panel e 的 SpaMosaic 风格展示 |
| [workflows/evaluate_hln_annotations.py](workflows/evaluate_hln_annotations.py) | HLN A1 外部人工标签与五方法比较，D1不评分 |

```bash
python -m scripts.workflows.plot_input_integration --help
python -m scripts.workflows.plot_input_integration --dataset mousebrain --run-dir /path/to/run --data-dir /path/to/dataset --analysis-dir /path/to/existing_analysis --output-dir /path/to/new_panels
python -m scripts.workflows.evaluate_hln_annotations --help
python -m scripts.workflows.evaluate_hln_annotations --inputs /path/to/annotation_inputs.json --output-dir /path/to/new_annotation_metrics --dry-run
```

该workflow **不运行 SpaMosaic 模型、不训练 integration**。它读取已有整合 embedding；原模态视图按原流程可能执行数据读取、PCA/Harmony 等输入准备。SPATCH 的 visualization sample feature cache 只写新的 output/input_modalities，独立于只读 model-ready `preprocessed_cache/spatch`。默认 max_samples=100000、UMAP参数同上。其 dataset key 与 compare/evaluate一致（MISAR为misar_seq，CRC为crc_stereocite）。

人工标注 JSON 示例：

```json
[
  {"method": "spa", "paths": {"run_dir": "/path/to/hln_run", "data_dir": "/path/to/hln_data"}, "analysis_dir": "/path/to/standardized_embedding", "annotation_path": "/path/to/manual_annotations.csv"}
]
```

method 支持 spa / cosie / present / mofa / spamosaic。spa 的 paths 如上；present 需要 run_dir；其余需要 embedding_file。analysis_dir 含已有 config.json、scaler_parameters.npz 和 retained labels。外部标注仍使用原 `Barcode` / `manual-anno` 列。可提供 display_name。

该工作流复用已存 scaler，以 joint A1+D1 和 independent A1 两种空间运行 K=2…12、seed=0、n_init=20、max_iter=300；检查保留K=5/8/10/12的标签重现。输出 ARI/NMI/Homogeneity/Completeness/V-measure/label-ASW/scaled label-ASW。这不改变 canonical HLN 无 ground-truth 的默认合同。`--dry-run` 会读取、计算、校验，但不写结果；它不是免计算模式。输出须全新，拒绝覆盖已有结果。

## Layout and protection

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

所有输入与新输出明确分离；历史 result/cache 只读。旧 raw-vs-standardized main、版本化 UMAP / supplementary CLI 已退休，不保留转发 shim。历史证据与源码快照保留原位置。当前工具见 [tools/validation](../tools/validation/README.md)，当前边界见[架构说明](../docs/CURRENT_ARCHITECTURE.md)，过程证据见[U系列计划](../docs/refactoring/CORE_INTERFACE_REFACTOR_PLAN.md)。


## CLI 与兼容边界

正式可执行入口仍为上列18个，不需要选择历史版本脚本。`--model-version v7A`等参数只是现有模型来源标签；带版本的历史结果目录不代表另一套CLI。

- **A 正式入口**：九数据集训练CLI、generic、batch、evaluate、standalone preprocessing，共13个。
- **D 专项入口**：Simulation diagnostics、两项comparison、两项workflows，共5个。保留其科研职责；`--dry-run`的含义以各自help为准，不能统一理解为“不计算”。
- **B 兼容API**：CRC/MISAR旧`parse_args`、配置投影及`run_*_pipeline`导入名保留。parser/投影从`paired_cli.py`、`multisection_cli.py`导出；pipeline只委托共享entry shell。它们不是另一套可执行launcher。
- **C 已退休历史CLI**：旧`run_result_v*_suite`、daemon、旧raw-vs-standardized比选、版本化UMAP/supplementary launcher与baseline录制工具继续退休。只保留历史报告/快照，不新增转发脚本、不删除冻结证据；退休明细见[U7b入口清单](../refactor_checks/u7b_cli_wrappers_20260922/REPORT.md)。

`paired_cli.py`和`multisection_cli.py`仅提供共享parser及已存在的配置投影；`paired_entry.py`和`multisection_entry.py`提供输入准备/资源所有权接缝，均无`__main__`，不是用户CLI。原始读取/预处理在data_io，Stage/prior/fit/export在公共training task，epoch循环在training.fit。正式runner之间不再互相导入parser、默认表或训练业务实现。

batch仍接受`crc`/`misar`；evaluate/comparison仍接受`crc_stereocite`/`misar_seq`。这是现有各入口的显式ID，不在本批新增别名或统一改名。batch的共享override追加在`--runner-args`之后，output-root与模式控制保持原行为；`--dry-run`只打印命令，不创建输出或启动子训练。

SPATCH `--input_mode reuse`为默认，仍是严格schema-1 model-ready cache；`raw`仍从六个已对齐H5AD（两section的RNA、Protein、已有HE/UNI）开始，不提供原图重新提取UNI。raw默认写本次output_dir/preparation_cache，--output_cache_dir只接受全新目录；overwrite不覆盖cache，invalid reuse不fallback raw。完整合同见[当前架构](../docs/CURRENT_ARCHITECTURE.md)。validation继续位于tools/validation；有些工具没有argparse，不能用`--help`假设会阻止其执行计算。未调整安装方式、包布局或运行目录规则。

当前checker直接依赖公共data_io/training/model接口。仅在测试正式CLI行为时调用其main；冻结refactor_checks旧导入保持历史原样，不能当作当前推荐API。兼容API状态与验证限制见[U7c报告](../refactor_checks/u7c_validation_dependencies_20260922/REPORT.md)。
