# 当前架构、输入合同与配置

本文描述 U 系列重构最终冻结的 v7A 纯空间图生产架构；历史实验/归档报告不是能力清单。用户运行入口见[根README](../README.md)和[scripts说明](../scripts/README.md)。冻结范围、验收证据及已知限制见本文第7节和[U系列最终退出记录](refactoring/CORE_INTERFACE_REFACTOR_PLAN.md)。

## 1. 职责边界

| 层 | 权威模块 | 职责 |
|---|---|---|
| 显式输入准备 | data_io的各族preparation模块 | reader/identity/原预处理 → PreparedDataset；不二次resolve、不训练 |
| 配置 | [entry_defaults.py](../training/entry_defaults.py)、[config.py](../training/config.py)、[configure.py](../model/configure.py) | 入口默认、层合并、模型/预处理基础默认各有权威 |
| fit外任务 | [paired_task.py](../training/paired_task.py)、[multisection_task.py](../training/multisection_task.py)、[mousebrain_task.py](../training/mousebrain_task.py)、[embryo_task.py](../training/embryo_task.py)、[spatch_task.py](../training/spatch_task.py)、[generic_task.py](../training/generic_task.py) | resolved配置+prepared → Stage/prior/前置行为/fit/export；公共runtime在task_runtime.py |
| epoch循环 | [fit.py](../training/fit.py) | 唯一shared fit/iter_fit_model，保持refresh/yield时点 |
| 通用写出 | [artifacts.py](../training/artifacts.py) | embedding/arrays/prior/weights/JSON机械写出；caller决定layout/payload/flags |
| analysis输入 | [analysis_results.py](../data_io/analysis_results.py)、[saved_assignments.py](../data_io/saved_assignments.py)及显式格式adapter | 文件格式、identity/order、来源；不决定truth派生/筛选或聚类策略 |
| analysis编排 | [evaluation_workflow.py](../analysis/evaluation_workflow.py)、[exact_kmeans_workflow.py](../analysis/exact_kmeans_workflow.py)、[comparison_readers.py](../analysis/comparison_readers.py)、[umap_workflow.py](../analysis/umap_workflow.py)、[integration_workflow.py](../analysis/integration_workflow.py) | 已对齐输入 → 显式协议/算法调用 → 保存/绘图；不另造算法 |

“统一prepare_dataset”是逻辑边界，不是一个支持全部数据的万能函数或根包dispatcher；各族显式spec与必要算法分开实现。training同样保留不同任务函数及窄policy，不根据dataset名称暗中选择训练行为。CLI保留输入选择、审计、资源所有权、GPU准入等必要外层职责，runner之间不再共享训练业务实现。

## 2. PreparedDataset

实际类型位于 [paired_preparation.py](../data_io/paired_preparation.py)，各族共享，不因文件名而限制为paired。

| 字段 | 含义与限制 |
|---|---|
| section_order | 显式训练顺序；不根据analysis展示顺序自动排序 |
| feature_dict | section→modality→模型就绪二维特征；保留原行序、dtype和对象/copy边界；派生PCA/Harmony轴不是原基因名称 |
| spatial_loc_dict | 与对应section特征逐行对应的坐标；原dtype/ownership保持 |
| processed_data_dict | 原预处理实际返回的AnnData等对象；可为None；不是Decoder输出 |
| identity | 本次实际拥有的section/spot/feature证据；只有row-order时如实记录，不伪造barcode |
| truth | 适用性及加载证据；not_loaded/not_provided不等于源数据没有truth |
| provenance | raw/model-ready/manifest/external/synthetic的实际来源，不能用shape猜身份 |
| preparation_audit | 原校验、选择、映射等记录；不为填满字段额外加载全量metadata |
| compatibility_context | 原summary/保存需要的过渡引用；不要求所有未来adapter永久提供同一批局部变量 |

这是逻辑封装，不强制统一float32或复制数组。generic bundle原有feature float32边界仍保留；其他adapter保留自身行为。HESTА/mmap对象按原所有权处理；SPATCH按实际cache loader读取方式，不宣称所有模式均使用mmap。backed资源登记/finally关闭仍由原调用层负责。

## 3. 所有现有训练输入模式

| 路径 | 显式边界 | 仍保留的专用语义 |
|---|---|---|
| CRC/HLN/Spleen | PairSpec → prepare_paired_dataset | CRC symbol/make_unique；HLN/Spleen gene_ids及既有global nonzero策略；Protein身份策略不同，不自动统一 |
| MISAR/Thymus/Simulation | [misar_preparation.py](../data_io/misar_preparation.py)的显式reader/adapter → PreparedDataset | MISAR RNA+ATAC、训练4→1；Thymus坐标/marker适配；Simulation spfac/nsfac/spatial_domain，audit保留原时点 |
| MouseBrain | MouseBrainInput → [configured_preparation.py](../data_io/configured_preparation.py) | 三section RNA+Metabolite+RNA.obsm已有UNI；gene_ids、obs校验、自定义section key恢复 |
| generic bundle | BundleInput → prepare_dataset | torch bundle中的feature_dict/spatial_loc_dict；optional processed_data_dict/section_order；不重做PCA/Harmony；原feature转float32 |
| generic raw config | RawConfigInput → prepare_dataset | 原flat COSIE schema，支持RNA/Protein/Metabolite/HE；不因MISAR存在而自动支持raw ATAC |
| generic synthetic | SyntheticInput（CLI --smoke_test） | 原随机生成与模式约束；明确synthetic身份，不作为真实adapter |
| Embryo manifest | ManifestInput → prepare_source → prepare_dataset | 既有manifest及RNA-only arrays；annotation/source-row与mmap不伪造、不全量复制 |
| Embryo external | ExternalRunInput → prepare_source → prepare_dataset | 原summary/status/manifest检查、逐section前N截断；source-row和annotation同步 |
| Embryo raw | RawHestaInput → HESTA预处理→manifest→prepare_dataset | HDF5/CSR、QC/采样/HVG、normalize/log1p、TruncatedSVD/component z-score；不改成普通COSIE/PCA/Harmony |
| SPATCH reuse | SpatchCacheInput → [spatch_preparation.py](../data_io/spatch_preparation.py) | strict schema-1/identity/参数检查，不读取raw来补造barcode |
| SPATCH raw | SpatchRawInput → [spatch_raw.py](../data_io/spatch_raw.py) →新cache→strict loader | inspect → RNA/Protein/HE sequential preprocessing；没有cache自动fallback或原图UNI提取 |

Embryo无external目录时的 `--reuse_preprocessed` 从本次output_dir/preprocessed读取；新运行通常显式给 `--preprocessed_run_dir`，不要为复用数据覆盖旧run。audit_only不进task；preprocess_only不构造Stage。require_harmony只验证记录，raw HESTA不会因此自动增加Harmony。

SPATCH raw的data_dir布局固定为：

```text
section1/adata_xenium_bin_filter.h5ad    RNA
section1/adata_codex_bin_filter.h5ad    Protein
section1/adata_he.h5ad                 已有HE/UNI
section2/adata_hd_filter.h5ad           RNA
section2/adata_codex_filter.h5ad        Protein
section2/adata_he.h5ad                 已有HE/UNI
```

六个文件必须已经对齐并满足原metadata/canonical ID/spatial/4828共同gene/DAPI-marker合同；不自动join、重排或修复。RNA按原HVG/normalize/log/scale/PCA，可选Harmony；Protein去恰好一个DAPI后CLR/scale/PCA；HE读取已有2048维X后PCA。默认n_comps=50时model-ready维数RNA50/Protein15/HE50。新cache目标必须不存在，默认output_dir/preparation_cache；invalid reuse绝不转raw。

### H&E/UNI能力的区别

通用 [datasets.py](../data_io/datasets.py) / [image_features.py](../data_io/image_features.py) 保留三种现有能力：reference AnnData的obsm特征、显式feature file、显式image+mask经UNI提取。相关字段是 `he_reference_adata_input`、`he_feature_input`、`he_input`、`he_mask_input`；权重/设备等沿用现有配置。多个来源同时提供时仍按原reference→feature→image优先级，不新增格式猜测。

当前正式MouseBrain配置走已有UNI，SPATCH raw走已有HE H5AD；二者不默认触发图像推理。通用图像路径存在不意味着本轮下载权重或验证了真实UNI推理。缺依赖/权重不能据此宣称可即刻运行。普通COSIE的obs一致性warn行为保持，不将它描述为所有adapter都严格raise或自动对齐。

## 4. 配置来源和真实默认

`model.configure`提供模型/预处理基础值；`training.entry_defaults`提供入口默认；`training.config`提供唯一字典合并/CLI缺省补齐工具：

```text
defaults/base < preset/model_config < dataset/input_config < explicit CLI
```

不是每个入口都提供全部层，也没有可按名字调用的preset registry。MouseBrain nested config和generic flat preprocess_config不互换。CLI→字段投影保留在现有CLI支撑层。None/False/0不同：非None CLI覆盖，False/0保留；JSON null仍是实际值，未承诺任意null都有效。analysis.protocols不与训练配置混用。

| 入口 | 默认epochs | 默认scale | precision / 默认动作 |
|---|---:|---:|---|
| CRC | 0 | 1.0 | amp none；train=False，训练需显式--train及正epochs |
| MISAR | 0 | 1.0 | amp none；train=False，训练需显式--train及正epochs |
| HLN / Spleen | 200 | 1.0 | BF16；默认train |
| Thymus / Simulation | 200 | 1.0 | BF16；默认train |
| MouseBrain样例 | 5 | 1.0 | 原FP32路径，样例device=cpu；配置/CLI可覆盖已有字段 |
| Embryo | 100 | 1.0 | 默认audit；训练配置BF16/RNA-only |
| SPATCH | 200 | 1.0 | BF16，正式GPU准入，默认reuse |
| generic普通 | 300 | 1.0 | 原路径；模型JSON/CLI覆盖既有字段 |

基础模型epochs300不等于所有入口300。研究用0.5要显式指定；generic没有scale CLI，通过已有 `--model_config` JSON，例如：

```json
{"graphsage": {"post_ot_graphsage_scale": 0.5}}
```

batch没有自动注入scale；evaluate的scale默认0.5是来源说明，不是训练override。generic smoke保留 `args.epochs or 3`，所以该模式显式0仍得到3；SPATCH固定执行设置、Embryo RNA-only/crossview=0等是已有模式约束，不声称任意CLI都能覆盖。旧help中Stage版本或“checkpoint”字样不代表另一模型入口或exact resume能力。

## 5. 训练和analysis的特有行为

CRC dry为train-mode/no_grad；MISAR dry为eval/no_grad。MouseBrain正常训练和dry都先执行train-mode/no_grad epoch0，保持dropout/RNG，不能删除或移入fit。Embryo noOT跳过initial prior但保留既有refresh/eval记录；SPATCH raw/reuse准备及GPU准入在task之前。generic periodic由iter_fit_model的yield触发，不新增forward，不导出optimizer/scaler/RNG状态；final eval与periodic语义不同。

analysis使用 [AlignedAnalysisInput](../data_io/analysis_results.py) 逻辑合同，表达embedding、sections、spot_ids/identity_evidence、coords、truth和provenance。格式adapter可保留legacy对象投影，不能通过缺barcode的旧NPY制造强身份。saved assignments只经中立格式/identity核读取，truth派生/过滤仍由analysis决定。

普通exact KMeans、evaluation scope、comparison输入、UMAP投影与integration输入分别有公共编排。joint_per_section只评价已有joint labels。UMAP输入dtype、reference fit/transform cohort、sampling及batch metrics空间由原协议决定，不因standardized clustering自动统一。plotting消费现成坐标；integration原模态展示预处理不是纯reader。HLN外部A1标注、Embryo层级/发育、Simulation factor、SPATCH大规模workflow保持独立。

## 6. 已保留修复、兼容与证据

B13 target_sum、B14 section key、B12 BF16→NumPy、B11 graph cache identity及Finit已保留，来源见 [U2总验收](../refactor_checks/u2_final_acceptance_20260921/REPORT.md)。B12只在NumPy边界把BF16升FP32，不把所有数据合同统一float32。当前初始化不能用不含Finit的历史快照替代。

B4/R1/R3、旧Stage smoke文案问题仍是独立事项。历史GPU记录只能称历史记录；固定precision/chunk等条件的验收不保证跨chunk逐位等价。SPATCH百万spot RAM、source SHA I/O及长期训练仍需单独规模验证。

18个正式CLI与少量薄Python API保留，历史suite已退休。当前validation直接调用公共模块；正式CLI只作为显式被测对象，不作为业务库。冻结报告、fixture、expected和known failure不重写；详见 [U7c记录](../refactor_checks/u7c_validation_dependencies_20260922/REPORT.md)。

## 7. 最终冻结范围与后续扩展

冻结的是当前源码、配置和用户文档；Git checkpoint及本地验证状态见[U8c报告](../refactor_checks/u8c_final_freeze_20260923/REPORT.md)。`refactor_checks/`、历史result/cache和临时验证输出不纳入版本，也没有删除；报告链接依赖本地保留的证据目录，纯源码clone不包含这些文件。冻结不等于向远程发布或完成全部规模实验。

- v7A纯空间图及Finit、B11/B12/B13/B14保持。U8b-1原固定FP32/BF16和same-layout验收通过；cross-chunk仍为 **KNOWN_NUMERICAL_SENSITIVITY**，不能推断跨chunk数值等价。
- CRC代表小输入的训练/artifact/reader通过；因不满足固定cohort，合格cohort的完整evaluate为 **NOT RUN**，不放宽协议。Simulation承担已通过的完整小链代表。
- SPATCH raw仍仅六个已对齐H5AD（RNA、Protein、已有HE/UNI）；不包含原图UNI重新提取。百万spot raw完整运行、生产规模RAM/耗时/source SHA I/O为 **NOT RUN**。
- 九数据集完整长期训练、完整benchmark、全量UMAP、GPU长期训练为 **NOT RUN**；现有weights不承诺exact resume。其它历史限制和失败足迹见[U8b-3状态记录](../refactor_checks/u8b3_final_state_record_20260923/REPORT.md)。

后续输入能力通过显式data_io spec/adapter交接PreparedDataset；训练差异通过公共task的明确策略接入，epoch循环继续复用fit，写出复用artifacts。analysis格式读取与identity在机械reader，truth/协议在analysis，workflow只组织已有算法。scripts继续只承担CLI和必要外层职责，不重新建立runner间业务依赖。新增实验需独立命名输出与验证目录，明确输入空间/参数/身份/随机性，不覆盖历史result/cache/fixture/expected；模型或科研协议变更另立批次并声明新的验证范围。
