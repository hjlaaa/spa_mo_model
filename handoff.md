# spa_mo_model 重构交接

更新日期：2026-09-17。仓库：`/home/hujinlan/spa_mo_model`。

**文档位置更新（2026-09-17）：** 已按用户授权完成文档归档，完整清单见 [docs/README.md](docs/README.md)。原 docs 的 22 份早期记录移至 `docs/experiments_early/`，原根目录 15 份后续实验记录移至 `docs/experiments_later/`；重构[审计](docs/refactoring/REFACTOR_AUDIT.md)、[计划](docs/refactoring/REFACTOR_PLAN.md)、[用户决策](docs/refactoring/USER_DECISIONS.md)移至 `docs/refactoring/`。本文件及根 README 保留；只更新必要导航，下文各阶段结论保留当时语义。`refactor_checks/` 既有证据未搬迁，data_io 整合意向仍未实施。

## 0. 新对话先看这里

1. P0–P8和S1/S2a/S2b/S2c已由用户验收；本轮仅按授权实施S2d，未执行correctness轨道。
2. scripts **25→18**：根14个稳定CLI，comparison/2个，workflows/2个。8个preprocessing comparison旧main退休，3个科研旧main由薄CLI替代；无版本化当前CLI、无shim。
3. 跨方法standardized比较、D8b UMAP、D8c输入/已有整合结果展示、HLN A1人工annotation、Simulation诊断保持。D8c不运行SpaMosaic模型。当前路径/JSON输入见scripts/README.md。
4. HEAD仍为`922d1738922e8b94890a54f5e42bbf6551f5ccc0`，累计成果未提交；禁止自动reset/checkout/clean/stash/stage/commit。
5. GPU FP32/BF16各13/13、same-layout差0，cross-chunk仍FAIL/KNOWN_NUMERICAL_SENSITIVITY。Stage smoke旧文案FAIL不修。B13/B14/B11/B12/B4/R1/R3均未执行。
6. S2a–S2d整理实施完成后停止；不自动进入其它工作。历史报告中的旧路径是历史事实，不要求旧CLI继续可运行。

最新证据：[S2d报告](refactor_checks/s2d_scripts_final_20260917/REPORT.md)、[最终入口审计](refactor_checks/s2d_scripts_final_20260917/S2D_FINAL_SCRIPTS_AUDIT.md)、[回归](refactor_checks/s2d_scripts_final_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/s2d_scripts_final_20260917/protection_check.json)。下列较早记录保持原阶段语义。

### 建议阅读顺序

- 本文。
- 最新S2b：[报告](refactor_checks/s2b_analysis_entries_20260917/REPORT.md)、[入口审计](refactor_checks/s2b_analysis_entries_20260917/S2B_ANALYSIS_ENTRY_AUDIT.md)、[回归](refactor_checks/s2b_analysis_entries_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/s2b_analysis_entries_20260917/protection_check.json)。当前 analysis checker 直测 P6 权威模块；旧 analyzer 路径与历史 checker 命令不承诺可执行。旧证据/expected 不改。
- 已验收S2a：[报告](refactor_checks/s2a_train_launchers_20260917/REPORT.md)、[launcher审计](refactor_checks/s2a_train_launchers_20260917/S2A_TRAIN_LAUNCHER_AUDIT.md)、[回归](refactor_checks/s2a_train_launchers_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/s2a_train_launchers_20260917/protection_check.json)。旧v4/v5/v6历史训练CLI已退休；原历史checker/REPORT保留，不承诺旧命令可直接重跑。当前checkpoint/config/handoff检查从本轮副本复制到新目录只check。
- S1：[报告](refactor_checks/s1_scripts_20260917/REPORT.md)、[脚本审计](refactor_checks/s1_scripts_20260917/SCRIPTS_AUDIT.md)、[回归](refactor_checks/s1_scripts_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/s1_scripts_20260917/protection_check.json)。工具当前位置见 tools/validation；正式runner路径不变。
- 已验收P8：[README](README.md)、[报告](refactor_checks/p8_final_20260917/REPORT.md)、[结构审计](refactor_checks/p8_final_20260917/FINAL_STRUCTURE_AUDIT.md)、[回归](refactor_checks/p8_final_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/p8_final_20260917/protection_check.json)。普通data模块旧路径已迁，不恢复shim；SPATCH loader和历史壳保留理由见审计。
- 已验收P7：[报告](refactor_checks/p7_cleanup_20260917/REPORT.md)、[DEAD审计](refactor_checks/p7_cleanup_20260917/P7_DEAD_CODE_AUDIT.md)、[回归](refactor_checks/p7_cleanup_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/p7_cleanup_20260917/protection_check.json)。有caller的兼容API/旧suite仍在，不能误当已退休。
- 已验收P6c：[报告](refactor_checks/p6c_plotting_20260916/REPORT.md)、[main审计](refactor_checks/p6c_plotting_20260916/ANALYSIS_MAIN_AUDIT.md)、[回归](refactor_checks/p6c_plotting_20260916/REGRESSION_REPORT.md)、[保护](refactor_checks/p6c_plotting_20260916/protection_check.json)。新evaluate用法和UMAP KEEP边界见报告。
- 已验收P6b：[报告](refactor_checks/p6b_protocols_20260916/REPORT.md)、[协议审计](refactor_checks/p6b_protocols_20260916/ANALYSIS_PROTOCOL_AUDIT.md)、[回归](refactor_checks/p6b_protocols_20260916/REGRESSION_REPORT.md)、[保护](refactor_checks/p6b_protocols_20260916/protection_check.json)。joint-per-section只切joint labels，ARI/NMI，无aggregation；P6c现在已完成，见最新记录。
- 已验收P6a：[报告](refactor_checks/p6a_loaders_20260916/REPORT.md)、[依赖审计](refactor_checks/p6a_loaders_20260916/ANALYSIS_DEPENDENCY_AUDIT.md)、[回归](refactor_checks/p6a_loaders_20260916/REGRESSION_REPORT.md)、[保护核对](refactor_checks/p6a_loaders_20260916/protection_check.json)。只完成loader；旧metric/plot壳范围见审计。
- 已验收B9：[报告](refactor_checks/fanalysis_b9_20260916/REPORT.md)、[缓存审计](refactor_checks/fanalysis_b9_20260916/B9_CACHE_AUDIT.md)、[回归](refactor_checks/fanalysis_b9_20260916/REGRESSION_REPORT.md)、[保护核对](refactor_checks/fanalysis_b9_20260916/protection_check.json)。新分析必须显式独立output；旧指标不补manifest、不覆盖、不冒称新合同已验收。
- [REFACTOR_PLAN.md](docs/refactoring/REFACTOR_PLAN.md)：§9后续依赖、§13固定验收合同、§21–§37最近状态；按下一批需要读对应合同。
- [USER_DECISIONS.md](docs/refactoring/USER_DECISIONS.md)：§1九数据集、§4能力决定、§12–§28最近实施状态。
- 最新命名补充：[MouseBrain入口重命名](refactor_checks/p5_mousebrain_name_20260916/REPORT.md)、[保护核对](refactor_checks/p5_mousebrain_name_20260916/protection_check.json)。当前canonical为scripts/run_mousebrain.py；旧名已删除，所有当前Python引用已迁。
- P5主体 [P5报告](refactor_checks/p5_entry_20260916/REPORT.md)、[launcher审计](refactor_checks/p5_entry_20260916/LAUNCHER_AUDIT.md)、[回归](refactor_checks/p5_entry_20260916/REGRESSION_REPORT.md)、[GPU gate](refactor_checks/p5_entry_20260916/p0_gate.json)、[保护核对](refactor_checks/p5_entry_20260916/protection_check.json)。
- generic训练合同仍见 [P4b-3报告](refactor_checks/p4b3_generic_20260916/REPORT.md)、[generic训练对照](refactor_checks/p4b3_generic_20260916/GENERIC_EQUIVALENCE.md)、[回归报告](refactor_checks/p4b3_generic_20260916/REGRESSION_REPORT.md)、[GPU gate](refactor_checks/p4b3_generic_20260916/p0_gate.json)、[保护核对](refactor_checks/p4b3_generic_20260916/protection_check.json)。
- MouseBrain epoch0合同仍见 [P4b-2报告](refactor_checks/p4b2_mousebrain_20260916/REPORT.md)。
- 共享fit提取起点仍见 [P4b-1报告](refactor_checks/p4b1_fit_20260916/REPORT.md)。
- 配置权威记录仍见 [P3b报告](refactor_checks/p3b_defaults_20260916/REPORT.md)及[默认来源与解析链](refactor_checks/p3b_defaults_20260916/DEFAULTS_AUDIT.md)。
- [P4a报告](refactor_checks/p4a_adapters_20260916/REPORT.md)和[数据适配专项](refactor_checks/p4a_adapters_20260916/ADAPTER_REPORT.md)继续定义数据身份与顺序合同。
- 上一批 [B10报告](refactor_checks/b10_parse_io_20260916/REPORT.md)中的parse-before-I/O/audit路径边界继续保持。
- 涉及配置时读 [B8模型消费者矩阵](refactor_checks/fconfig_b_20260916/model_consumer_matrix.md)、[入口消费者矩阵](refactor_checks/fconfig_b_20260916/RUNNER_CONSUMER_MATRIX.md)及 [P3a报告](refactor_checks/p3a_config_20260916/REPORT.md)。
- 涉及数值参考时读 [P0补充报告](refactor_checks/p0_supplement/REPORT.md)、[原默认GPU报告](refactor_checks/REPORT.md)和本文§7。
- `REFACTOR_AUDIT.md`只作历史线索，需要时再读，不是当前实现清单。

**文档时间层次注意：** 计划保留早期核对及各批“当时未执行后续”的记录，部分正文仍有旧措辞，例如feature字段残留、self_path_mode仍在、某批“下一批尚未授权”等。以当前源码、最新追加状态及用户新指令为准。不要根据旧段落恢复已经退休的功能，旧行号也不能机械套用。

## 1. 工作树、环境与保护范围

### 当前工作树

当前S2b：23个analysis壳删除，Simulation诊断与只读joint结果校验归位；两份冻结suite仅解除旧analysis调度。model/training/data_io、九canonical/batch/evaluate不改。下面P8及更早段落保留其历史状态。

P8：五个源文件普通移动（utils按函数拆为两处）、同步import、README/当前导航；数学body与state保持。起点7443项旧状态和34保护根按白名单核对，证据见§17。下面保留P5/B9等历史说明，不应将其“本批”措辞当作当前边界。

P5起点与P4b-3的2918项保护清单无差异。本批冻结3175项已有文件状态、34个保护根，以及508项P3b/13项旧fit/17项旧MouseBrain/58项旧generic/14项P5旧命令与交接证据。生产新增scripts/run_experiments.py，Human Embryo只增加可选scale CLI接口，删除无消费者的v7B/v7C suite壳；其余模型、data_io、training/config.py、training/fit.py与canonical runner不改。另更新三份进度文档；HEAD未变、staged为空。详见最新P5 protection_check.json中的current_hashes。

累计tracked修改主要在`model/`和`scripts/`；已经删除：

- `model/linkage_construction.py`：P2退休dense OT。
- `scripts/benchmark_microenvironment_fullspot.py`：D3退休专属大型实验。
- `scripts/run_result_v7b_suite.py`、`scripts/run_result_v7c_suite.py`：P5审计无调用者、显式scale/新batch覆盖后退休。

未tracked内容包括`REFACTOR_AUDIT.md`、`REFACTOR_PLAN.md`、`USER_DECISIONS.md`、`refactor_checks/`、两个原始冻结脚本、`training/`、`data_io/`及`handoff.md`。真实状态请先运行只读git命令核对；出现新差异就说明，不自动回退。

每批新目录通常有`before.json`、`before_sources/`、原累计patch、本批增量patch、`protection_check.json`。P4b-3历史增量继续保留；最新P5保存before.json、before_sources/、p5_only.patch及protection_check.json。P4b-3保存：

- `refactor_checks/p4b3_generic_20260916/accepted_through_p4b2.patch`
- `refactor_checks/p4b3_generic_20260916/p4b3_only.patch`

这些用于区分既有修改与新批修改，不是要求覆盖当前工作树的备份还原脚本。

### 环境

- Python：`/home/hujinlan/miniconda3/envs/cosie/bin/python`，通常加`-B`。
- Git：`/home/hujinlan/miniconda3/envs/cosie/bin/git`；不要假定普通PATH里的git可用。
- 既有验证环境：RTX4090、Torch2.4.0、CUDA12.1、FAISS1.9.0。
- 普通沙箱可能看不到GPU，既有GPU检查通过相应执行权限运行。不要据此判断没有GPU、安装依赖或擅自改为CPU验收。
- 未发现适用的AGENTS.md；若新环境有新增规则再读取。

### 严格保护

以下目录保持原位置、格式、内容，不修改、移动、覆盖、重算、清理：

```text
gene_imputation/
gene_imputation_spatial_smoothing/
gene_imputation_shared_gene_validation/
preprocessed_cache/
results/
所有现有 result_* 目录
历史 audit / config / summary / embedding / metacell 映射产物
已有 P0 至 P5 的 fixture / expected / 报告 / 失败证据
```

只按需读取小源码/小元数据，不批量读取、复制或哈希大数组/历史结果。新测试、缓存替身、依赖运行缓存、报告统一写`refactor_checks/`下**新的独立位置**。不运行真实九数据集训练、100/200轮训练、真实预处理、百万spot缓存、昂贵benchmark；不安装/升级依赖。

冻结`.pt`受现有gitignore影响，多数只在本地；普通git add不代表已经包含这些二进制。不要删除、重录或误称它们已经进入版本库。

保护指纹是明确清单和目录根stat，结合测试I/O guard，不是全量历史数组内容核验。

## 2. 完整实验范围与能力决定

### 九数据集全部KEEP

v7A是**模型/算法基线**；v7A suite只是调试子集。完整实验名单已由`result_v6`三组suite状态与九份run_summary核实，不需要再推测：

| 数据集 | 当前入口/共享链 | 必须保留的差异 |
|---|---|---|
| MouseBrain | `run_mousebrain.py` → 独立epoch0 → training.fit | 三section，RNA/Metabolite/HE，FP32，epoch0 dry forward |
| Human Embryo（HESTA RNA-only） | `run_human_embryo_rna_only.py`、`hesta_rna_utils.py` | 七section、外部Harmony记录、RNA单模态、lambda_contrast=0、合法no-OT |
| MISAR-seq | `run_misar_seq.py` → CRC shared fit | RNA/ATAC、原section顺序、AMP none |
| SPATCH | `run_spatch.py` → CRC shared fit | 既有cache、RNA50/Protein15/HE50、BF16 |
| CRC_Stereo-CITE-seq | `run_crc_stereocite.py` | 共享基因/marker对齐、RNA/Protein |
| Human_Lymph_Node | `run_human_lymph_node.py` → `data_io.paired` / CRC普通pipeline | 基因ID、跨section非零基因过滤、RNA50/Protein20 |
| Mouse_Spleen | `run_mouse_spleen.py` → HLN普通parser / `data_io.paired` / CRC | ADT marker保序、RNA50/Protein20 |
| Mouse_Thymus | `run_mouse_thymus.py.read_pair` → MISAR普通pipeline | 四section、RNA obs[x,y]坐标、ADT marker轴、RNA50/Protein15 |
| Simulation | `run_simulation.py.read_pair` → MISAR普通pipeline | 五section、原RNA/ADT（模型RNA/Protein）、spfac/nsfac/spatial_domain |

以上简写入口均在`scripts/`。将来九数据集都应能经统一框架运行当前v7A，并有明确分析入口；原聚类、标准化、seed/K/抽样/指标协议有差异就保留，不能以统一代码为由统一算法。没有在v7A suite出现不能成为删dataset support的理由。

### D1–D8与其它既定决定

- **D1**：保留普通可调`post_ot_graphsage_scale`；当前研究命令=.5，.25/.75等仍有效。B/C命名preset与专属suite最终退休，但壳尚未清理，留后续迁完调用者再做；历史版本名称/结果不改。
- **D2已实施**：legacy/no_adj_self/self_path_mode及两个self_linear退休，保留空间自环和outer residual。schema/RNG影响见§5。
- **D3已实施**：额外attention context gate、alpha反传实验开关及专属大型benchmark退休。普通gate、正常attention alpha反传、共享spatial pooling和OT topology cost保留。
- **D4已实施**：动态刷新仅ot，无source选择。fused/ot/final三个embedding阶段及必要输出仍保留。B3按路径退休关闭，不重新修复fused功能。
- **D5已实施**：本项目metacell生成、训练、还原映射退休。full-spot、旧cache/产物/映射及外部COSIE metacell结果读取保留。
- **D6 KEEP**：原始H&E+mask→UNI提取，以及已有HE/UNI特征输入；不得强迫重提取。
- **D7 KEEP**：FAISS IVF/Flat和生产CLI显式blockwise候选检索，不新增自动fallback。blockwise不是dense OT。
- **D8a KEEP**方法间指标/KMeans预处理比较；**D8b KEEP**跨方法UMAP。迁为显式输入/输出参数后再删硬编码版本壳，不改真实评估协议。
- **D8c暂KEEP**：SpaMosaic风格工作流，修改前先说明具体输入/输出/调用者及是否实际运行SpaMosaic模型，再交用户决定；不阻塞无关批次。
- dense OT和独立单向candidate-sparse OT已由P2真正删除；单模态不等于单向OT。双向内部的正反检索/索引/top-k工具必须保留。
- 保留历史result_vX不意味着永久保留run_result_vX_suite.py；但不能未经当前批次授权就整批删历史launcher。

## 3. 已完成节点索引（按实际推进顺序）

| 节点 | 关键结果/限制 | 证据 |
|---|---|---|
| 初始审计与修订计划 | 取消v15A，实际v7A为主；九数据集覆盖与保护范围明确 | [计划](docs/refactoring/REFACTOR_PLAN.md)、[能力决定](docs/refactoring/USER_DECISIONS.md)；旧AUDIT仅历史 |
| Batch 0/P0 | 固定FP32/BF16及同布局checkpoint参考接受；cross-chunk保留已知FAIL | [补充报告](refactor_checks/p0_supplement/REPORT.md)、计划§13 |
| C1 | SPATCH只读复用schema1 cache；解除整训练源码hash及无关raw路径/mtime硬耦合；缺cache/build请求明确停止，无预处理fallback | [C1](refactor_checks/c1_cache_contract_20260915/REPORT.md)，73项合同 |
| P1 | generic、九数据集、smoke保留入口真实初始化/forward/refresh迁为双向sparse；当时先留旧core | [P1](refactor_checks/p1_entry_migration_20260915/P1_REPORT.md)，原78项、后续退休适配70项 |
| P2 | 删除dense文件/Stage接口、单向solver/helper/attention分支、模式CLI/config及专属占位；保留no-OT | [P2](refactor_checks/p2_ot_retirement_20260915/P2_REPORT.md)，19项；净删1195行 |
| F-config-A | 修B1/B2：显式CLI不再被JSON盖回，未提供CLI不覆盖JSON；bool三态、目录/summary依resolved | [A](refactor_checks/fconfig_a_20260916/REPORT.md)，55项 |
| D4 | 退休fused/final刷新选择，固定ot；B3按退休关闭，阶段embedding不删 | [D4](refactor_checks/d4_ot_refresh_20260916/REPORT.md)，7项 |
| D3 | 退休额外context gate/alpha实验/大型benchmark，保留ordinary gate与OT .8/.2 context | [D3](refactor_checks/d3_context_gate_20260916/REPORT.md)，8项 |
| D5 | 退休本项目metacell生成/训练/还原；保留外部读取与cache false身份 | [D5](refactor_checks/d5_metacell_20260916/REPORT.md)，28+9项 |
| D2 | 退休额外self-path与两个self_linear；同有效参数状态等价，原同seed初始化确实改变 | [D2](refactor_checks/d2_self_path_20260916/REPORT.md)，9项；APPROVED_SCHEMA_RETIREMENT |
| F-config-B/B8 | 删23模型NO-OP、4预处理NO-OP、2无效CLI；保留65个真实模型默认字段；已知无效输入明确拒绝 | [B8](refactor_checks/fconfig_b_20260916/REPORT.md)，模型34/入口16项 |
| P3a | 新training/config.py提公共解析；Stage消费完整resolved；dataset常量和四文本包装不迁 | [P3a](refactor_checks/p3a_config_20260916/REPORT.md)，解析23组、Stage11项 |
| F-data(B10) | 两runner先parse，再audit实际data/output；help零数据/audit/运行输出副作用；原schema/adapter保持 | [B10](refactor_checks/b10_parse_io_20260916/REPORT.md)，main17/adapter12项；仅2文件+17/-16 |
| P4a | 四训练入口普通函数/显式参数，去源码replace/exec、argv/globals；基础CRC/MISAR数据/配置/训练保持 | [P4a](refactor_checks/p4a_adapters_20260916/REPORT.md)，专项12+9+6组、CPU406；10生产文件+632/-584 |
| P3b | 九数据集默认值显式化，入口单次parse/model resolution，共同消费resolved；原值/P4a/B10/C1保持 | [P3b](refactor_checks/p3b_defaults_20260916/REPORT.md)，27配置、18交接、CPU406；10生产文件+742/-323 |
| P4b-1 | CRC共享fit及13个runtime定义原样迁training/fit.py；八数据集调用，MouseBrain/generic仍原位 | [P4b-1](refactor_checks/p4b1_fit_20260916/REPORT.md)，6事件、4 CUDA一步全等、CPU406；5生产文件+473/-436 |
| P4b-2 | MouseBrain独立epoch0后调用共享fit；移除本地loop，保留FP32/schedule/梯度/history；generic仍原位 | [P4b-2](refactor_checks/p4b2_mousebrain_20260916/REPORT.md)，6事件、3 CUDA一步全等、CPU406；2生产文件+66/-137 |
| P4b-3 | generic接入同一epoch主体，普通生成器交回周期保存；bundle/preprocess/synthetic/noOT/weights保持；P4b完成 | [P4b-3](refactor_checks/p4b3_generic_20260916/REPORT.md)，12 CPU/3 CUDA专项、CPU406；2生产文件+106/-97 |
| P5 | 九canonical CLI不重解析，新非版本batch独立子进程、显式参数与status；审计后退休v7B/C壳 | [P5](refactor_checks/p5_entry_20260916/REPORT.md)，75配置/命令、9+9 handoff、9 scale、24 batch检查；4生产文件+150/-202 |

各批最初计数与后续退休后的回归计数可能不同，有明确适配记录；不要把旧专属模式的删除误报为当前KEEP覆盖缺失，也不能擅自再删失败用例。

## 4. 当前v7A数学和执行合同

权威模型只有`model/stage_model.py::StageMultiModalModel`，不建V7AModel、不恢复feature graph。

```text
feature_dict → 各模态MLP → fusion
→ pre-OT空间GraphSAGE
→ 同步双向OT-guided attention
→ post-OT空间GraphSAGE
→ decoder / final embedding
```

- pre/post均为空间图，独立参数实例；pre scale=1，post普通可调，研究命令=.5。**裸库默认post scale仍1**，不能把“v7A=.5”理解为所有默认都已改成.5。
- 每section空间KNN、k+1查询、距离权重、自环、反向边、duplicate max、row normalization、edge排序均保持。邻接自环与OT context的self-excluded pooling是不同功能。
- GraphSAGE当前无learned self_linear，仍weighted neighbor→原linear/bias/activation/dropout→`x + scale * message`→norm。保留原零张量加法/dtype提升及运算顺序，不随手“简化”；scale0也不是完全bypass。
- 双向candidate-sparse UOT：保留reverse-query、映射到(A_local,B_local)、edge去重、candidate并集裁剪及同一coupling行/列top-k生成正反prior。
- 三section各方向从同一pre状态计算update，中间section平均两个update后只做一次apply/residual/norm；不串行更新。
- 普通attention gate、confidence、attention score、dropout/norm不变；D3删除的是额外context gate，不能把全部gate删除。
- refresh固定`output['ot_embeddings'].detach()`；对同一ot阶段做self-excluded spatial context并normalize。候选检索用semantic ot embedding，当前cost=.8 semantic cosine + .2 context cosine。
- epoch100开始每20轮，optimizer step后eval/no_grad，用旧prior单遍forward，再替换prior；下一轮train。最后epoch刷新时最终eval使用新prior，不两遍refresh。
- MouseBrain保留epoch0、train mode/no_grad的dry forward及dropout RNG消耗；SPATCH等不能凭统一框架新增这一步。
- MouseBrain训练FP32；SPATCH训练CUDA BF16 autocast，参数/有效梯度FP32、不启用GradScaler；OT refresh/final eval在autocast外。MISAR显式AMP none，其他按入口真实设置保留。保存FP32 embedding不能证明全程FP32。
- loss、Adam、AMP边界、chunk/checkpoint、loss-only、graph cache与各数据集epochs/K/候选数不强行统一。合法no-OT继续直通；启用跨section OT则要求完整双向prior。

## 5. D2之后的state与初始化：必须特别注意

唯一退休key：

```text
graphsage.self_linear.weight
post_ot_graphsage.self_linear.weight
```

两者在当前v7A路径从不参与forward、梯度None、无Adam状态，但原来仍初始化并消耗RNG。删除后：

- RNA+Protein小例完整state/optimizer注册参数148→146；P0三模态162→160。不是所有数据集都固定同一key数量。
- 原128维各key为FP32 `[128,128]`，共退休32768个注册标量。
- D2同seed构造的RNG改变，小例34个保留张量初值受顺序移位影响；**不能宣称当前同seed从头训练会复现原v7A初始化。**
- 数值等价使用冻结旧有效状态，按完整名称/shape/dtype/requires_grad映射，再strict=True加载并恢复冻结forward RNG。不能strict=False、zip参数或只比loss。
- 没有dummy模块、补消耗RNG或生产checkpoint兼容框架。旧checkpoint两个额外key不再能原样strict加载。
- B8、P3a、B10、P4a、P3b、P4b-1、P4b-2、P4b-3、P5相对D2之后没有新增schema/初始化RNG变化。继续使用既有APPROVED_SCHEMA_RETIREMENT，不扩大投影。

详细映射与17条P0投影见 [D2参考适配](refactor_checks/d2_self_path_20260916/P0_APPROVED_SCHEMA_RETIREMENT.md)。

## 6. 当前配置、缓存与数据适配边界

### 配置（B1/B2 → B8 → P3a）

固定优先级：**base defaults < preset/model config < dataset/input config < 显式CLI**。None表示未提供，False/0/空值不能被当成缺省。既有generic smoke后置特例等未借P3a重写。

`training/config.py`公共函数：`load_json`、`merge_config`、`apply_explicit_overrides`、`resolve_model_config`、`resolve_option_values`、`config_source_layers`、`serialize_config`，P3b新增`parse_dataset_args`和`describe_run_config`。只管普通dict/Namespace解析及记录，不管数据、model构造、fit、refresh、artifact写入或analysis。依赖scripts→training.config→model.configure，model不可反向import training/scripts。

`model/configure.py`仍保留模型/预处理默认和已知字段拒绝。Stage提供config时只深拷贝完整resolved，不再递归补默认；`config=None`纯模型默认API保持。当前16条KEEP调用均提供完整配置；旧partial库调用需先resolve。

B8删除的23模型NO-OP包括graph两假开关、encoder.type/residual、contrastive六假开关/权重、fusion.mode/input_dim、graphsage.num_layers/use_distance_weight、uot五项假initial/momentum/cost、reconstruction.loss及loss三假开关。另删4预处理假参数（rna_var_names_source/superpixel_size/patch_size/uni_checkpoint）和CRC无效dry_run、独立smoke无效smoke_test flag。**不是删除真实图像算法参数，也不是删除generic有效smoke能力。**

拒绝已核实NO-OP/RETIRED字段，不建立任意拼写的全局schema。Mouse可用的一些JSON字段在generic无consumer，generic明确拒绝；预处理字段层级也按原入口消费者区别检查，不能统一成新行为。细表看B8两份矩阵。

P3b将九入口默认值集中为各自get_dataset_defaults和必要常量，dataset实际值没有统一。MouseBrain的JSON优先级保持；SPATCH原十个后置运行参数在数据读取前显式补入，cache身份不变。CRC/MISAR基础次模态及Thymus/Simulation显式Protein/adt语义保持。正常main单次argparse、单次模型配置解析；CRC/MISAR直接调用者仍可让pipeline在缺少run_config时生成一次，不再重复merge。

训练summary新增resolved记录，或在MouseBrain现有记录中增加数据身份/输入配置；原字段及各入口序列化差异保持，没有统一artifact schema。新记录与实际model/runner/preprocessing对象单独核对。True默认可用BooleanOptionalAction显式False覆盖，help/usage新增反向选项；旧有效命令的全字段值保持。CRC/MISAR裸调试默认仍有CPU/epochs0/AMP none，suite显式研究参数不冒充裸默认。

### SPATCH cache（C1+D5）

现有目录：`preprocessed_cache/spatch/v6_harmony_n50_hvg3000`，schema=1，历史`metacell=false`必须继续接受。

- 直接复用原格式；不删除/移动/重算/写回/升级，不normalize/PCA/Harmony，不自动build或fallback。
- 有效性不再绑定整训练脚本源码hash或无关路径/mtime；仍验证真实输入/预处理身份、文件完整性、section/spot顺序、模态、shape/dtype和必要维度。
- 默认明确使用所选cache。显式另一raw路径但身份冲突/证据不足时拒绝，不偷偷换旧数据。`input_identity_manifest`只核验来源声明，不冒充当前raw内容重新核验；记录`current_raw_files_verified=False`。
- manifest顺序RNA/Protein/HE、Stage顺序HE/RNA/Protein分别保留；实际输入RNA50/Protein15/HE50。
- 老cache缺每个feature数组与ordered barcode的内部绑定证据：记录限制，不因此判失效重算、补写manifest或伪造证明。
- metacell=true明确拒绝；false历史身份不等于恢复metacell生成。
- C1的73项是小缓存替身动态验证，**不等于真实百万spot缓存动态验证**。

### B10与P4a

Thymus/Simulation现在：普通parse_args一次 → 生成MISAR run_config一次（显式Protein/adt）→ `_write_adapter_audit(Path(args.data_dir), Path(args.output_dir))` → 将run_config显式交给原MISAR pipeline。P4a删除的源码load/replace/exec与argv注入未恢复；B10 audit函数及parse-before-I/O边界不变。

help退出0；解析失败退出2；均无data loader/audit/运行output副作用。旧问题只用spy重现，不执行旧真实audit。

audit的CSV/JSON schema、字段/顺序/统计/读模式/直接异常保持；Simulation dataset_path反映实际输入。audit仍遍历原完整SECTIONS；共享validate_args仍在原pipeline位置。不借B10修section选择/B14或文件关闭的其它问题。

原全局DATA_DIR/OUTPUT_DIR仍用于CLI默认，其中输出默认仍含result_v4；这是保持原参数值，不是audit继续忽略显式路径。无显式output的真实任务仍可能选择旧默认目录，**不要为了交接或验证直接运行它**。

HLN/Spleen显式调用原CRC pipeline；纯读取/基因/marker函数在data_io.paired。MISAR读取/shared-axis函数在data_io.misar，section常量仍在原入口。唯一必要UMAP调用者已改普通数据import/call，分析协议不变。新data_io不承载训练、通用预处理迁移或registry。

### P4b-1共享训练runtime

权威函数为`training.fit.train_small_crc_model`，保留原函数名和原函数体。`run_one_forward`、AMP/scaler、prior初始化/更新编排及必要监控/清理共13个定义一起迁移。CRC保留普通import名称，MISAR/SPATCH/Human Embryo直接import training.fit；HLN/Spleen经CRC、Thymus/Simulation经MISAR到同一函数。八条实际绑定与main spy见`fit_callers.json`。数据/CLI/resolved/Stage构造/各入口保存均留原处，fit不再解析或merge配置。P4b-1当时MouseBrain和generic源码字节不变；P4b-2随后仅接入MouseBrain，generic仍未迁移。

### P4b-2 MouseBrain接入

MouseBrain配置/数据/Stage/initial prior后，原epoch0 train-mode/no_grad完整dry forward继续留在runner原位置，随后仅一次调用training.fit.train_small_crc_model。移除本地正式epoch循环；prior三helper在真实resolved输入下比较等价后改普通import。lambda_for_epoch原样迁到fit，schedule字符串解析仍在runner；json_safe的ndarray语义与CRC不同，保留原函数。

fit新增四个明确可选参数：lambda_contrast_schedule（默认None）、clear_step_state（默认True）、record_elapsed_time（默认True）、allow_empty_epochs（默认False）。MouseBrain分别传已解析schedule、False、False、True；它原来保留step后梯度/输出、不记录耗时，并允许epochs=0空训练后final eval。CRC八入口采用原默认，事件/数值保持。无dataset名称判断、callback、额外解析或merge。MouseBrain从既有resolved执行记录构造明确fit参数，FP32/AMP none、无chunk/checkpoint/cache保持；原None块大小显式转0，表示同一不分块分支，resolved原记录不变。共享disabled GradScaler对象不启用缩放或autocast，RNG不变。

### P4b-3 generic接入

generic的parse/build_model_config/三种输入准备/Stage/initial prior及save_training_artifacts保持；正式epoch主体改用training.fit.iter_fit_model。只新增record_loss_weights=True、refresh_ot=True、yield_every=0三个参数。generic传False、resolved uot.enabled和原save_every，保留四字段history、合法noOT、step/refresh后周期保存。yield的普通tuple为(is_final, history, outputs, ot_updates)，中途outputs是当轮train forward，final是最终eval；yield发生在autocast/no_grad之外。既有train_small_crc_model作为普通最终结果函数消费默认不交回中间结果的同一主体，签名和原tuple返回不变。

generic仍FP32，无epoch0；初始epsilon_init与刷新epsilon_update分别消费，不改为同一个值。原processed_data_dict虽由loader返回但未传model，当前也保持None。普通epochs0仍只有final forward，原smoke显式0→3特例保持。周期/最终weights、文件名state/config/history/section_order/input_dims和可选embedding仍由generic保存，不增加resume。九数据集runner源码未改，MouseBrain独立epoch0/schedule/梯度/history合同保持。

## 7. 最新验证状态与如何继续使用参考

### Batch 0最终状态

**BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY。**

原冻结参考：

```text
refactor_checks/v7a_gpu_fp32_frozen/
refactor_checks/p0_supplement/fp32_deterministic/
refactor_checks/p0_supplement/spatch_bf16_deterministic/
```

固定两/三section小参考保存参数、输入、顺序、图、prior、RNG、阶段embedding、loss、梯度及一步Adam。本轮P6b fixed FP32/BF16各13/13、same-layout0、原cross-chunk FAIL保持，详见§14。下表保留P5已验收完整回归记录，不能当作P6b重新跑过全部训练测试：

| 检查 | P5已验收结果（历史） |
|---|---|
| B10 main/help | 17/17 PASS，含6个独立Python进程真实main的help/非法CLI |
| B10 audit/adapter | 12/12 PASS；两audit逐字节相同，六真实adapter合成病例全等 |
| P3b专项 | 九数据集27/27完整配置；18/18真实main/适配交接，单次parse/resolution；19项False/零值/默认隔离 |
| P4a专项 | 12/12完整old/new数据/配置/handoff、9/9异常、6/6导出（351个小文件）；本批resolved已属旧字段，全部精确比较 |
| P4b-1回归 | 原6组FP32/BF16/FP16事件及4组CUDA一步与已验收基准全等；八调用关系保持 |
| P4b-2回归 | 6组MouseBrain旧/新事件（199/200、schedule、dry、零epoch）及3组CUDA一步（2/3 section、schedule）全等 |
| P4b-3专项 | 12 CPU真实Stage用例、3 CUDA一步old/new一致；bundle/preprocess/synthetic/noOT/weights/零epoch；123项范围/默认主体AST核对 |
| P3a | 解析23/23组（含原55矩阵旧/新）及Stage11/11；另10入口默认/显式20case对P4a原记录精确一致、无投影 |
| B8至C1完整CPU | 343/343：B8 34+16、D2 9、D5 28+9、D3 8、D4 7、A 55、P2 19、P1 70、generic 15、C1 73 |
| 既有CPU回归合计 | 406/406本批run01 PASS，含B10 29、P3a 34及B8至C1 343；各批旧失败证据保留，专项另计 |
| FP32 / SPATCH BF16 fixed | 各13/13 PASS，max_abs=max_rel=0 |
| 同布局四checkpoint | 两精度PASS，输出/loss/梯度/一步参数/RNG差0 |
| cross-chunk | 仍FAIL，完整cross_configuration与原replay相同 |

跨chunk参数最大差：FP32 `6.517022848129272e-5`；BF16 `0.0019998354837298393`，原BF16更新后FP32 final embedding差约`9.3021e-3`。分类为KNOWN_NUMERICAL_SENSITIVITY，不阻塞保持布局的pure-refactor；不表示证明了具体CUDA kernel原因。原默认GPU非确定性失败及R1 NaN反例保留。

### 固定验收规则

同precision/AMP边界、同chunk layout、同checkpoint（或已验证同布局对照）、同有效参数/prior/graph/fixture、同section/spot/模态顺序及RNG。验证进程使用既有deterministic CUDA设置，**不加入生产训练配置**。

atol=rtol=1e-5；结构/索引/schema/dtype/None/RNG精确；检查输出、分项loss、梯度和一步Adam，不只比最终loss。改变chunk、loss归约、autocast或checkpoint分块必须重新审查cross-chunk，不能套用固定gate。任何新fixed-layout/resolved/state/RNG差异立即停止定位，不能归入旧敏感性。

### 当前应使用的checker

优先参考最新目录：`refactor_checks/p6b_protocols_20260916/`。P0五个checker/projection文件保持已验收版本字节；本批B9/P6a/数值专项用法见REGRESSION_REPORT.md。P5完整历史训练回归仍在p5_entry_20260916，不直接覆盖旧输出重跑。

- 数值checker：`p0_supplement_v7a_baseline.py`，依赖同目录`p0_freeze_v7a_baseline.py`和投影helper。
- 结构化门禁：`check_p0_gate.py`；命令/结果见`p0_gate.json`。
- CPU来源、命令和副本差异见`REGRESSION_REPORT.md`、`cpu_regression_summary_run01.json`、`runtime_regressions_cpu.json`、`runtime_regressions_cuda.json`；本批16个checker均run01通过。
- P4b-3的`REGRESSION_ADAPTATION.md`仍记录上一批generic接口适配。P5直接复制既有checker，无新增数值投影，所有指定回归通过；P5新主入口spy增加任意argv和MouseBrain fit边界，范围及搭建记录见本批REPORT.md。旧fixture/失败证据不改。
- 不直接重跑旧`run_p0_regressions.py`/CPU编排写既有报告；它们有固定输出名。下一批复制到新目录并保留原baseline只读引用，按真实新接口最小适配spy。
- 不直接拿根`scripts/freeze_v7a_baseline.py`、`scripts/supplement_v7a_baseline.py`重录：它们保留已退休API/schema及record路径，是原证据；部分record/旧行号解析已不适用于当前树。

未来检查示例（**本次交接不执行**）：先为获授权新批选择不存在的`refactor_checks/<新批次>/`，将MPLCONFIGDIR、XDG_CACHE_HOME、NUMBA_CACHE_DIR、CUDA_CACHE_PATH、PYTORCH_KERNEL_CACHE_PATH、TMPDIR都定向新目录。设`P0_CHECK_DIR`为其绝对路径，再从仓库根运行：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -B refactor_checks/p5_entry_20260916/p0_supplement_v7a_baseline.py check --profile fp32 --baseline refactor_checks/p0_supplement/fp32_deterministic --report "$P0_CHECK_DIR/fp32.json"
```

立即保存真实退出码。用`check_p0_gate.py`的`--profile fp32 --process-exit-code 实际退出码 --report 新报告绝对路径`核对；固定gate通过后再执行：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python -B refactor_checks/p5_entry_20260916/p0_supplement_v7a_baseline.py check --profile spatch_bf16 --baseline refactor_checks/p0_supplement/spatch_bf16_deterministic --report "$P0_CHECK_DIR/bf16.json"
```

不能只看退出码：当前两check通常为1，必须证明`same_configuration_failures=[]`、13项PASS且差0、`checkpoint_same_chunks`精确PASS、唯一跨配置失败`chunk_dropout_zero`，并且整个cross_configuration与原replay相同。不能硬编码忽略非零码或把checker改成返回0。

### 已批准的历史适配，仅测试进程使用

- P2：`P_dense=None`旧占位投影；B8新增明确拒绝时另严格核实6个旧P2输入值：tol=1e-6、check_every=10、clip=0.0/2.0、keep_dense=False、direction=forward。
- D4：旧固定ot/False source配置；D3：旧关闭的context-gate配置及空context输出；沿各自报告的严格断言范围。
- D2：只投影两self_linear及对应schema/None梯度、旧no_self_linear字段，共17条明确路径；其余按名strict检查。
- B8：只投影列明23个NO-OP输入字段，逐项断言旧值和类型，其余叶对象保持identity。
- P3a/B10/P4a/P3b没有新增config/state/RNG/数值投影；P4b-1/P4b-2/P4b-3仅调整P0静态源码定位和调用链检查，数值部分及已有投影不变。P5没有新增checker数值或schema适配。

这些不是生产兼容层；旧fixture/expected从未改写。B8首次因6个P2旧输入字段拒绝而停止的失败记录仍保留，后来仅适配测试输入，不能把它归入数值敏感性或删掉记录。

## 8. 未完成事项与下一批边界

**没有正在执行或获授权的下一批。** 用户选定后再按单批继续，不把“总体范围已决定”当成全计划实施授权。

### 计划后续依赖

P4a/P3b/P4b-1/P4b-2/P4b-3/P5已完成，B13/B14没有夹入。P5保留各runner原artifact格式，没有统一summary schema。剩余结构依赖：

1. **P6整体已完成**：B9来源绑定与九dataset真实协议保持，显式evaluate/绘图/UMAP权威已形成；剩余具有真实caller或独有能力的混合壳逐项列于P6c审计。v7A/SPATCH suite仍有冻结checker消费者，v4/v5/v6历史混合壳按P5审计暂留；旧analysis命令缺新output参数时应明确退出，不为兼容而恢复输入目录写入。
2. **P7已完成，P8待单独授权**：9个DEAD函数及22个unused import已清。processed_data_dict、load_cosie_style_data及有caller/独有能力的旧壳保持；不建设永久兼容网或大型registry。

### 独立bug批次，不能顺手修

| 问题 | 当前状态/边界 |
|---|---|
| B13 | shared modality路径漏传target_sum，默认None不触发；单独修非默认输入预期，不重算SPATCH缓存 |
| B14 | custom section_id与输出s1/s2映射不一致；单独核对feature/spatial/metadata/order，别借B10改 |
| B11 | spatial graph cache identity不足；单独正确性修复 |
| B12 | 直接BF16→numpy问题；单独转换边界修复 |
| B4 | prior padding进入softmax的support mask问题；需独立数学预期，不改confidence/方向平均等其它语义 |
| R1 | crossview signed分母可零/近零；原NaN反例保留，不擅自换loss/softmax |
| R3 | sparse solver kernel floor等稳定性；不改u/v顺序、迭代或log-domain |
| B9 | F-analysis已完成：新输出绑定embedding/order/truth/run/protocol/implementation；来源不符或无manifest明确拒绝，旧历史指标保持未核验provenance |
| R2 | 现有weights不是完整resume；推理恢复/精确resume是新功能，不顺手实现 |
| T1/T4 | 旧smoke错误文字断言、checkpoint比较及validator graph.k_neighbors假键待独立测试清理；不能改模型迎合旧测试 |
| T2 | 保护插补目录中的discovery/import问题，本计划不改其文件 |

B1/B2已修，B3随D4退休关闭，B7随P2退出；B5/B6/旧B15机制已不在当前主路径。不要重新复活它们来“修旧审计”。

### 尚未动态覆盖

没有运行真实九数据集完整训练、100/200轮、百万spot cache、真实normalize/PCA/Harmony/UNI、正式analysis/指标/聚类/绘图、生产大文件导出、三个插补目录的真实动态工作流或性能基准。现有小型替身、真实helper和静态核对不能冒充这些覆盖。

## 9. 新对话启动清单

1. 确认cwd与当前git status/staged/HEAD；对照最新P7保护清单的current_hashes及本批COMPLETION记录，保留所有累计修改。P7起点核对6635项旧状态及34个保护根；不把旧HEAD当当前成果。
2. 读本文及当前任务相关计划/报告；先识别用户是否明确授权下一批。没有新实施任务时只接手，不自行推进。
3. 为新批创建新的refactor_checks子目录，记录起点及必要小型证据。旧快照用于对照，不要拿git HEAD当最新工作树。
4. 只扫描该批相关调用者/数据合同；保留九数据集、保护对象、已批准退休与配置支持面。
5. 实施前小测试/spy，实施后指定回归和固定GPU gate；新失败停止定位，不通过重录、放宽容差、丢用例继续。
6. 报告实际增量、行为变化/不变、验证与未覆盖、下一步建议，完成本批即停止。

P7及本次进度更新已完成；不会自动进入B13/B14、P8或其它实现批次。

## 10. P5 batch入口当前用法

九个canonical CLI和完整示例见[P5报告](refactor_checks/p5_entry_20260916/REPORT.md)。scripts/run_experiments.py要求显式--datasets和--output-root，使用--runner-args DATASET JSON_ARGV透传原CLI；可显式覆盖epochs/seed/device/post scale，缺省不补科研设置。普通scale .25/.5/.75及其它合法值保持；Human Embryo新可选flag未提供时parsed/default/model完整不变。CRC/MISAR原epochs0是准备默认，训练必须显式给epochs；MouseBrain必须给config，SPATCH必须给已有cache。每dataset独立subprocess，batch无analysis、模型/data/config import。v7B/C壳已退；v7A/SPATCH有真实冻结checker调用者而暂留。

## 11. MouseBrain入口命名补充（2026-09-16）

按用户单独要求，scripts/run_mousebrain_v2.py改为scripts/run_mousebrain.py，无旧名转发壳。batch、五个保留suite及根freeze脚本的旧名import/路径全部机械替换；原freeze脚本的其它退休接口问题未修，不因此成为可直接record的入口。runner仅去掉docstring/help中的V2字样，数据/配置/Stage/prior/epoch0/fit/save代码保持。14项命名/配置/真实main-fit交接及6组旧MouseBrain事件、RNG、参数/保存检查通过；没有真实训练或GPU gate重放。P5已验收数值证据保持，不将本次静态/CPU检查冒称新GPU验收。

历史报告/fixture/expected和旧checker不改。下一次获授权回归应复制到新的独立目录；从p5_mousebrain_name_20260916取得已改当前import/source路径的mouse_test_support.py及p0_freeze_v7a_baseline.py，其余沿P5既有checker，按当前文件名适配测试路径，不能直接运行固定旧输出目录下的编排。旧快照/报告里的run_mousebrain_v2是历史路径，不应批量改写。最新保护清单3405项旧文件状态、34个保护根及17项MouseBrain旧冻结文件核对保持，HEAD/staged不变。本次仅命名补充，不进入F-analysis/P6、P7/P8或独立correctness批次。

## 12. F-analysis(B9)完成边界（2026-09-16）

仅新增analysis/cache.py来源/manifest普通函数，并在真实持久cache消费者处接入；没有P6目录迁移。覆盖SPATCH requested completion与fit、compare SPATCH raw/standardized、通用requested五数据集、Human Embryo标准化/邻居/flat labels、CRC/SPATCH历史包装外层completion，以及独立Human Embryo补图PNG缓存。HLN及其它始终重算/拒绝覆盖的分析能力保持原样，没有把requested五数据集当成完整支持范围。

新输出写analysis_source_manifest.json（Human Embryo原analysis_manifest.json仍保留）。embedding逐文件SHA256/shape/dtype/顺序、实际aligned spot/coordinate/truth、必要run/resolved投影、各自protocol、明确implementation版本参与身份。输入与输出分离；缺来源或不匹配直接拒绝并要求新独立目录；相同来源才复用。无历史metrics迁移/补manifest/重算。实现版本必须随分析数学变化更新；不会hash训练runner或整仓库。历史复制的batch指标和作为补图输入的旧label仍明确historical/unverified，不冒称重新计算。

真实tiny旧bug已复现，旧证据378文件冻结。SPATCH、通用五数据集、七section Human Embryo与compare两协议的指标/label/scaler逐值一致，补图PNG字节一致。V12十四类及额外来源/缺artifact/无副作用检查通过；C1 73/73，P5 command、九数据集+generic shared fit引用与P3b/P4a源码保护通过。B9固定GPU FP32/BF16各13/13、差0，同layout checkpoint差0；cross-chunk原完整FAIL保持。最新只check适配器位于refactor_checks/fanalysis_b9_20260916/，无新数学/schema投影。

验证限制：仅tiny合成/替身；通用requested loader保留原路径布局（P5 flat输出loader适配不是本批）；CRC外层以tiny AnnData+计算sentinel检查，CRC核心算法字节不变；未真实跑百万spot、真实九数据集完整analysis、真实预处理或cache重建。B13/B14、P6/P7/P8未执行。后续需要用户另行授权。


## 13. P6a完成边界（2026-09-16）

本批按用户单独授权仅迁analysis loader/数据读取依赖。新增analysis/loaders.py、analysis/inputs.py及data_io/adapted.py，普通函数显式接收run/data/section输入；九dataset均有明确路径。现有MethodData由原消费者保留，未建registry/Loader类或统一evaluate CLI。Thymus/Simulation六个纯data helper与ADT轴迁入data_io，runner按原名import；MISAR只迁SECTION_INFO字典。所有剩余runner函数AST不变，训练/model/fit/config和P4a有效行为保持。D8b/D8c跨方法/SpaMosaic风格UMAP继续KEEP，数据helper不再import training runner。

修改前冻结30组真实tiny loader输出与265项输入/输出证据；v6/v7A样式及旧通用格式的embedding值/dtype、section/spot/barcode、metadata/truth和来源identity逐项一致。22项requested/旧analysis壳路径handoff、9个Thymus/Simulation section的纯helper对照、D8b/D8c import与数据handoff通过。B9原43项和补充13项回归、17项实际旧loader身份命中/输入变化miss通过，implementation identity不因纯位置移动而改变。P4a12例、P3b27例、C1 73/73、P5命令/九dataset+generic shared-fit引用、B10 help无I/O通过；401个分析函数AST及223个算法调用AST保持。

FP32/BF16固定check各13/13且差0，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture覆盖、容差/layout/AMP或生产CUDA设置变化。4072项起点状态、93个源快照、265项旧loader证据和34个保护根核对通过；HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，所有成果仍在未提交工作树。

详见[P6a报告](refactor_checks/p6a_loaders_20260916/REPORT.md)、[依赖审计](refactor_checks/p6a_loaders_20260916/ANALYSIS_DEPENDENCY_AUDIT.md)、[回归](refactor_checks/p6a_loaders_20260916/REGRESSION_REPORT.md)、[保护核对](refactor_checks/p6a_loaders_20260916/protection_check.json)。旧完整analysis exec/argv壳中metric/export/plot职责明确暂留P6b/P6c；新loader已独立。没有全量迁移跨方法混合workflow、删除旧main或统一输出布局。历史format只做tiny替身验证，不读取真实大数组，不补造不存在的barcode/truth。

P6a完成，P6整体未完成。P6b未执行、P6c未执行、P7/P8未执行、B13未执行、B14未执行；clustering/metrics算法未改、plotting未重构，历史metrics未重算、SPATCH preprocessing cache未改，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成本批后停止；下一批必须另行授权。

后续验证只能在新的独立目录check。最新P0 checker/projection和check_p0_gate副本位于refactor_checks/p6a_loaders_20260916，数值逻辑与B9字节相同。B9/P4a/P3b/C1复制checker如何只读旧expected并重定向输出见本批REGRESSION_REPORT.md；不要直接复跑固定输出名覆盖现有报告。


## 14. P6b完成边界（2026-09-16）

本批仅执行P6b。新增analysis/clustering.py、metrics.py、protocols.py、sampling.py、batch_metrics.py；九dataset标准化/聚类/监督与internal指标/采样和batch数学使用普通函数，保留原dataset与requested/compare差异。23个生产文件+1626/-1355；training/model/data_io、P6a loader及canonical训练runner未修改。

完整读取supplementary参考后正式定义joint / joint_per_section / independent。joint_per_section只切已有joint assignments，零重新scaler/cluster；实际指标只有ARI/NMI，原协议没有跨section平均，aggregation=none。MouseBrain仅RegionLoupe/annotations、MISAR五truth、SPATCH仅cell_type_common、Simulation spatial_domain；CRC/HLN/Spleen/Thymus无truth不制造指标。Embryo不擅自增加原参考没有的监督范围。保留参考与requested不同的truth过滤规则。两个原requested入口通过--joint-per-section显式启用，产生逐section结果与completion状态；B9绑定scope/K/section顺序/label/metric/filter/aggregation及implementation，纯函数搬动保留原默认分析identity。

122项old/new数值exact，29项joint-per-section专项，18行/14个supplementary导出文件（仅生成时间不同）、真实requested新scope拟合次数/旧指标/缓存一致性通过；P6a30loader与22条路径、B9 43+13、P4a12、P3b27、C1 73、P5/P4b必要引用回归通过。FP32/BF16 fixed各13/13且差0、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture/tolerance/layout/AMP修改。4564项旧状态、96源快照、旧tiny基准及34保护根核对无越界差异；HEAD/staged不变。

详见[P6b报告](refactor_checks/p6b_protocols_20260916/REPORT.md)、[协议审计及joint-per-section专章](refactor_checks/p6b_protocols_20260916/ANALYSIS_PROTOCOL_AUDIT.md)、[回归](refactor_checks/p6b_protocols_20260916/REGRESSION_REPORT.md)、[保护核对](refactor_checks/p6b_protocols_20260916/protection_check.json)。P6b实施完成，P6整体仍未完成；P6c、P7/P8、B13/B14未执行，plotting/UMAP未重构。历史metrics未重算，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成后停止，下一批必须另行授权。

## 15. P6c完成边界（2026-09-16）

本批按用户单独授权完成P6c：新增明确scripts/evaluate.py和普通analysis.evaluation/requested/comparisons/spatch/embryo/plotting/umap。九dataset显式选择dataset/run-dir/output-dir/protocol/scope，直接消费P6a loader、P6b clustering/metrics/protocol。31个生产文件+5245/-4344；training/model/data_io、九canonical runner、shared fit、P5入口完全未改。

joint / independent / joint_per_section明确分开，后者只切已有joint labels，真实ARI/NMI、valid spot与section字段保持，不重新聚类、不跨section平均；无truth dataset不造值，Embryo不擅自扩参考监督范围。Embryo七section/celltype/developmental/25叶weighted Ward与层次图保留，逐section补图可显式调用。SPATCH requested与compare K独立，B9新输出来源绑定保持。

先审计34个analysis main；退休无生产caller且已完整替代的rerender_result_v6_umap_figures.py、complete_human_embryo_per_section_plots.py。真实suite/aggregate/cross-method/legacy格式与后处理caller仍有的旧壳逐项暂留。D8b跨方法和D8c原始模态准备保持KEEP，二者共用analysis.umap/plotting并写新output root；新evaluate提供已有UMAP坐标重绘，不另创统一九dataset UMAP预处理协议。D8c SPATCH可视化输入缓存隔离到新output，来源变化在预处理前拒绝，不触碰C1缓存。

21个renderer旧/新输入、panel/colors/order/size/dpi逐项相同；九workflow文件集合/CSV字段/JSON字段类型及数值相同；UMAP16组prepared input/参数/显示/重绘对照通过。九真实loader→workflow与九standalone joint_per_section+真实requested共19组，最终绘图入口和cache边界通过。P6b122 exact、joint29、requested scope7、P6a30、B9 43+13、P4a12、P3b27、C1 73及P5/P4b必要引用通过。FP32/BF16各13/13、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持，只check。

5341项旧状态、全部源快照/旧tiny基准及34保护根核对，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged为空。所有成果未提交。详见[P6c报告](refactor_checks/p6c_plotting_20260916/REPORT.md)、[main审计](refactor_checks/p6c_plotting_20260916/ANALYSIS_MAIN_AUDIT.md)、[回归](refactor_checks/p6c_plotting_20260916/REGRESSION_REPORT.md)、[保护](refactor_checks/p6c_plotting_20260916/protection_check.json)。旧历史metadata/裸label证据不足时保持原能力并明确限制，不补造barcode或来源；未读取真实大数组。

P6a/P6b/P6c组成的P6整体实施完成。P7未执行、P8未执行、B13未执行、B14未执行；未运行真实九dataset完整analysis、长期训练、真实预处理/UMAP大优化或cache重建，未修改历史result/cache。完成后立即停止，没有下一批授权。

后续check从本批目录复制当前checker，旧expected只读，输出新获授权独立目录。原Embryo补图/重绘两个已退休main的旧checker import仅在新副本中指向analysis权威；不恢复生产兼容壳。最新P0适配器与P6b字节相同，仍必须检查结构化结果而非忽略exit1。

## 16. P7完成边界（2026-09-17）

本批仅执行P7，基于已验收P6真实工作树重新扫描当前AST/import/CLI/string/dynamic/test/protected引用。修改前建立P7_DEAD_CODE_AUDIT；原9个DEAD候选全部仍有定义、无当前消费者，现删除。另删22个无消费者import绑定。7个生产文件+2/-248，净减246行；剩余全部函数/class及顶层AST仅有列明删除，整个Stage类、state/schema/constructor/forward/参数顺序保持，training/shared fit/config/data_io和所有runner字节未改。

processed_data_dict与load_cosie_style_data有真实调用者继续KEEP；旧order/epoch兼容未证明可安全移除而KEEP TEMPORARILY。v7A/SPATCH suite仍有P0/C1等冻结checker消费者；其它混合壳有真实caller或独有调度/历史输出职责，未证完整替代者暂留。本批未新增退休launcher/analysis wrapper。v7B/v7C suite、旧MouseBrain文件及P6c两个已退休main标ALREADY REMOVED，不重复计删。D8b跨方法UMAP、D8c raw modality准备/整合、single modality/noOT、H&E/UNI、FAISS/Flat/blockwise、weights export及三种analysis scope继续KEEP。

P5 batch12（含九真实canonical tiny handoff）、P4a12、P3b27、C1 73、shared fit事件6、MouseBrain事件/RNG6、generic实际tiny Stage/Adam12、P6a30、P6b数值122、joint_per_section29、requested scope7、evaluate19、B9 43及UMAP16均通过。55个模块import、14个子进程help和2359个Python源码compile通过。FP32/BF16固定各13/13 exact，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保留，只check不record。P4a首次checker命令参数误用导致parse退出2，修正调用后通过，原错误日志保留。

6635项旧状态、104个生产源快照、旧tiny expected及34保护根核对；无非白名单差异，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，成果仍未提交。保护检查不声称递归hash真实大数组。详见[P7报告](refactor_checks/p7_cleanup_20260917/REPORT.md)、[DEAD审计](refactor_checks/p7_cleanup_20260917/P7_DEAD_CODE_AUDIT.md)、[回归](refactor_checks/p7_cleanup_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/p7_cleanup_20260917/protection_check.json)。

P7完成。P8未执行，B13未执行、B14未执行，其它独立correctness问题未执行；未改变model数学或analysis协议，未运行真实长期训练/完整analysis/真实预处理，未修改历史result/cache，未重建cache。完成后停止，下一批需要另行授权。

后续check须复制到新的独立授权目录；优先使用P7已有适配器和只读expected指针。P0五个checker/projection与P6c字节相同，无新数值/schema投影。不要执行旧suite/main，也不要清理任何旧验证产物以消除失败日志。

## 17. P8完成边界（2026-09-17）

本批仅执行 P8，先基于已验收 P7 工作树建立 FINAL_STRUCTURE_AUDIT。普通 MOVE：model/data_preprocessing.py→data_io/preprocessing.py（字节相同）、model/multimodal_preprocessing.py→data_io/datasets.py、model/image_preprocessing.py→data_io/image_features.py、scripts/hesta_rna_utils.py→data_io/hesta.py（字节相同）；model/utils.py 按原函数边界拆为 data_io/common.py 与 model/spatial_graph.py。更新真实 import，model 包不再重新导出数据函数，无旧路径 shim；load_cosie_style_data 的真实调用能力仍保留。SPATCH schema=1 cache loader/metadata、dataset artifact 保存和有消费者历史壳按审计留原位，未创建新的配置/trainer/artifact/loader框架。

迁移函数/class/签名及有效可执行 AST 完全相同；唯一函数 body 字符串变化为 SPATCH 未由训练main调用的 cache writer 的源文件诊断路径，既有 manifest 和 C1 identity 不变。Stage 类/constructor/state/schema/参数顺序、preprocessing 数学、training/config.py、training/fit.py、全部 analysis 权威源码不变。44 组图/data/UNI/HESTА/state/RNG 小对照精确一致。九dataset P5 handoff、P4a/P3b/C1、shared fit、MouseBrain epoch0、generic 12小例、P6a/P6b/joint-per-section/evaluate/B9/plot/D8b/D8c通过。FP32/BF16 fixed 各13/13 exact，same-layout checkpoint差0；原cross-chunk FAIL / KNOWN_NUMERICAL_SENSITIVITY完整保留，只check不record。

新增根 README.md 和 scripts/README.md，列出九canonical训练入口、batch、evaluate及69个脚本分类。MouseBrain名称保持run_mousebrain.py；batch用misar/crc，evaluate用misar_seq/crc_stereocite，明确区分。13条README命令经真实parser/help/dry-run/sentinel检查；v7A研究scale=.5显式传入，未把底层/部分裸入口1.0默认改写成.5。joint_per_section只切joint labels，参考ARI/NMI、不聚类、不跨section平均；SPATCH双K协议与Embryo特殊分析保持。model/training/data_io无scripts依赖，analysis无training runner/历史suite依赖；迁移模块无循环依赖和活跃旧路径引用。

起点7443项旧状态、104份生产源码快照、旧fixture/expected/报告/FAIL及34保护根按白名单核对。HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，累计成果未提交。保护核对不声称递归hash真实大数组。旧checker的模块路径仅在新P8副本适配；原报告和expected不改。旧UMAP快照的嵌套import适配过程日志完整保留，不是算法/数值失败。

详见[P8报告](refactor_checks/p8_final_20260917/REPORT.md)、[结构审计](refactor_checks/p8_final_20260917/FINAL_STRUCTURE_AUDIT.md)、[回归](refactor_checks/p8_final_20260917/REGRESSION_REPORT.md)、[保护](refactor_checks/p8_final_20260917/protection_check.json)。后续如需check，从本批适配checker复制到另一个新目录；不可重跑record或覆盖本批/既有产物。

P8实施完成，当前主线重构实施整体完成（等待用户审阅本批，不冒称用户已经验收P8）。B13未执行、B14未执行、B11/B12未执行、B4/R1/R3未执行；模型/预处理/analysis科研方法未改。未运行真实长期训练或完整九dataset analysis、未重建cache、未修改历史result/cache。完成后停止，不自动开启任何独立correctness批次。


## 18. S1工程组织收尾（2026-09-17）

S1 按当前工作树审计 69 个 scripts Python 文件，完成用途/CLI/import/subprocess/dynamic/frozen/README/helper 分类。九 canonical runner 原路径 help 全部通过；原样嵌套至 scripts/train 的真实探针全部 import 失败，因此正式 CLI 保持根路径。仅普通移动 scripts/validate_bidirectional_ot_attention.py → tools/validation/validate_bidirectional_ot_attention.py，使用仓库根目录 python -m，去掉原有 6 行 import/bootstrap；全部函数/签名/main AST 相同，不新增 sys.path hack 或兼容 shim。scripts Python 数 69→68，tools/validation 1 个 Python 工具；未创建空 tests/train/comparison/workflows 目录，无 RETIRE。

其余 68 文件保留：正式路径稳定，P1/D3/D4/generic 行为 checker 消费的 validator 留原位，P0 freeze/supplement 和 v7A/SPATCH suite 留原位。supplementary joint-per-section 有 P6/P8 当前 checker import 和历史 labels/export 变体，未退休；HLN A1 外部人工 truth 的五方法流程替代证据不足而 KEEP TEMPORARILY。D8b/D8c 独立工作流保持，旧壳/implementation leak 如实记录，不宣称所有历史脚本都已变薄。

105 生产 Python compile/import、24 正式/工具 help（I/O guard）、13 根 README 命令通过。移动工具 4 组参数/Stage/input/prior/forward/输出/RNG 精确一致，仅 3 种耗时 metadata 字段变化；旧/新原始证据保留。九 dataset handoff/batch12、P4a12、P3b27、C1 73、shared fit6、MouseBrain6、generic synthetic/zero/noOT、P6a30、P6b122、joint29、evaluate19、requested scope7、B9 43、plot15、D8b/D8c UMAP16 通过。4 个保留 validator main 通过；run_stage_model 旧 smoke 文案断言原版/当前同 FAIL，未修、未冒称 PASS。

FP32/BF16 fixed 各13/13 exact，same-layout 差0，cross-chunk 完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保持。只 check，不 record、不改 fixture/expected/tolerance。model/data_io/training/analysis 及 68 个保留 scripts 字节不变；旧状态、保护根和 HEAD/staged 按本批 protection_check 核对，成果未提交。

S1 实施完成，等待用户审阅。B13/B14/B11/B12/B4/R1/R3 未执行；未运行真实长期训练、完整 analysis、真实预处理或 cache rebuild，未修改历史 result/cache。完成后停止，无下一批实施授权。

详见 [S1报告](refactor_checks/s1_scripts_20260917/REPORT.md)、[完整脚本审计](refactor_checks/s1_scripts_20260917/SCRIPTS_AUDIT.md)、[回归](refactor_checks/s1_scripts_20260917/REGRESSION_REPORT.md)、[保护核对](refactor_checks/s1_scripts_20260917/protection_check.json)。后续 checker 复制到新授权目录并只读旧 expected；不要原地重跑覆盖本批报告。


## 19. S2a历史训练壳退休（2026-09-17）

本轮只执行 S2a，按当前引用闭包成组退休 11 个 v4/v5/v6 历史训练 launcher/suite/daemon，生产 +0/-3186 行。历史训练壳 13→2；scripts Python 68→57。v4c→v4b、v4c_large→v4_large 的历史内部引用不再作为保留理由。没有当前正式 caller 或需迁移的共享 helper，不新增 shim；九 canonical runner、run_experiments/evaluate/preprocessing 路径与字节保持。

保留两项 FROZEN_PATH_EXCEPTION：run_result_v7a_suite 的 training_commands 仍为当前 P0 replay 配置来源；run_spatch_v7abc_suite 的 training_command/validate_training 仍由 P0/C1 当前 checker 调用。它们的生产调度能力已可用 batch 表达；本轮不迁移这两个冻结校验依赖，后续 S2c 是否处理须另行授权。其它 v4/v5/v6 普通 helper 没有当前 consumer，旧后台 PID/GPU等待/from-task/训练后自动分析随壳退休，未新增 scheduler。

16 份历史 check_entries.py 的 v6 import 属于旧批次配置投影证据，原样保留，不作为当前回归入口。当前使用 S1 已迁移的 run_experiments→九 runner handoff/config checker，并复制到本轮独立目录 check；不修改历史 checker/fixture/expected。所有 11 个 Git 旧路径可恢复；其中 4 个前期未提交版本与 HEAD 不同，其精确字节已存在于已验收 S1/before_sources，本轮再次留存，不能把 HEAD 当作全部成果提交。

94 个保留生产模块 compile/import、13 个 guarded/direct CLI help、11 个替代 batch dry-run 逐 token 对照通过；九 dataset 实际 tiny handoff/batch12、P3b27、P4a12、C1 73、shared fit6、MouseBrain epoch0/RNG6、generic synthetic/zero/noOT3 通过。analysis/evaluate 和 validators 必要 import 通过；无当前源码残留路径。FP32/BF16 固定各13/13 exact、same-layout差0，cross-chunk完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保留。只 check、不 record、不改数学/协议/默认值。新静态 checker 曾把 batch dry-run 的 JSON list 误当 dict，修正本轮测试读取后通过；生产无变化，诊断留存。

12206 项继承状态、34 个保护根及 HEAD/staged 按本轮白名单核对，保留生产文件字节相同。HEAD 仍 922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged 为空；累计成果未提交。历史结果/大数组没有递归哈希；证据范围以 protection_check 为准。

S2a 实施完成，等待用户审阅；完成后停止。S2b/S2c/S2d 未执行，analysis/supplementary/validation 物理整理未执行；B13/B14/B11/B12/B4/R1/R3 均未执行。未运行真实长期训练、完整 analysis、真实预处理或 cache 重建，未修改历史 result/cache。

详见 [S2a报告](refactor_checks/s2a_train_launchers_20260917/REPORT.md)、[launcher审计](refactor_checks/s2a_train_launchers_20260917/S2A_TRAIN_LAUNCHER_AUDIT.md)、[回归](refactor_checks/s2a_train_launchers_20260917/REGRESSION_REPORT.md)、[保护核对](refactor_checks/s2a_train_launchers_20260917/protection_check.json)。后续只在新授权目录使用当前 checker，不覆盖本轮或历史证据。

## S2b（2026-09-17）完成记录

23 个旧 analysis/supplementary/postprocess CLI 已退休；scripts 57→34。Simulation 保留 explicit run/data/output diagnostics CLI，数学移入 analysis/simulation_diagnostics；七个 persisted joint-label read/validation helper 移入 analysis/joint_results。CRC/SPATCH 新 generation_script provenance 指权威 analysis 模块，B9 key/version 不改。两份冻结 suite 的 training API 保持，旧 analysis 调度移除。

当前 checker 从本批目录复制到下一批独立目录，只 check；P6a loader、P6b 122 数值项、29 joint-per-section、evaluate 19、plot15、UMAP16、B9 43、requested scope7、C1 73 通过。FP32/BF16 13/13、same-layout0、cross-chunk FAIL / KNOWN_NUMERICAL_SENSITIVITY 保持。额外静态/能力迁移/保护证据见本批报告。无 S2c/S2d、correctness 或真实完整运行。
