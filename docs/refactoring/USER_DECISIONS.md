# 科研与运行能力保留决策

## S2d 当前实施状态（2026-09-17）

S2d完成scripts最终入口收敛：**25→18个Python CLI**。8个raw-vs-standardized比选入口退休；D8b、D8c、HLN A1人工标注三个旧main分别由非版本化薄CLI替代，另有一个统一的标准化跨方法comparison入口。comparison/两个、workflows/两个；Simulation独有诊断继续留根部。所有保留CLI文件名无实验版本号，无compat shim、无新sys.path hack。

D8c的真实职责是原模态输入panel c与已有整合embedding panel e的SpaMosaic风格展示；不运行外部SpaMosaic模型或integration训练。跨方法reader、必要标准化编排、人工标注及可视化输入准备已归analysis/data_io；P6科研参数、UMAP/batch输入空间、原label/figure保留K不变。九canonical runner、batch、evaluate、preprocessing、12个工具路径和原实现保持。

FP32/BF16各13/13，same-layout差0；cross-chunk保持FAIL/KNOWN_NUMERICAL_SENSITIVITY。Stage smoke既存文案断言失败不修。B13/B14/B11/B12/B4/R1/R3均未执行。未运行真实长期训练、完整analysis或cache重建，历史result/cache/fixture/expected/report未改。

最新入口见 [scripts导航](../../scripts/README.md)，证据见 [S2d报告](../../refactor_checks/s2d_scripts_final_20260917/REPORT.md) 与 [最终审计](../../refactor_checks/s2d_scripts_final_20260917/S2D_FINAL_SCRIPTS_AUDIT.md)。S2a–S2d工程整理实施完成；完成后停止，无自动进入correctness轨道的授权。以下较早章节为当时阶段记录，当前状态以本节和README为准。


## S2c 实施结果与 S2d 边界（2026-09-17）

S2c已完成实施：scripts **34→25**；5开发工具移动至tools/validation并用职责命名，两个旧record工具与两个旧suite退休。当前12个开发Python无实验版本文件名，无新增compat shim/sys.path hack。冻结证据路径不改；没有current executable需要FROZEN_PATH_EXCEPTION。

九dataset/batch/evaluate/preprocess保持；current checker已解除对8个compare preprocessing CLI的普通helper依赖。八compare原字节未改，S2d建议先迁必要跨方法能力再退休旧CLI；D8b/D8c/A1/Simulation独有能力保持。唯一保留版本化scripts为D8c generate_result_v6_spamosaic_umap，留S2d处理。

FP32/BF16 13/13、same-layout差0；cross-chunk仍KNOWN_NUMERICAL_SENSITIVITY；Stage smoke旧文案失败不修。B13/B14/B11/B12/B4/R1/R3未执行；未运行真实长期训练/完整analysis/重建cache。S2d尚未执行，完成后停止。[S2c报告](../../refactor_checks/s2c_validation_20260917/REPORT.md)。以下较早阶段段落保留当时记录，当前入口以README和本节为准。

更新：2026-09-17。P7已验收，本轮仅完成P8（§28），当前主线重构实施整体完成，待用户审阅本批。数据helper/空间图按原函数归位，README与当前入口导航完成；有消费者历史壳继续保留。科研数学与analysis协议、D2 schema和原cross-chunk FAIL保持。B13/B14/B11/B12/B4/R1/R3未执行，完成后停止，无后续批次授权。

Batch 0正式状态为 **BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY**；固定pure-refactor验收条件见 [REFACTOR_PLAN.md §13](REFACTOR_PLAN.md)。原cross-chunk FAIL保留，不作为符合固定条件的pure-refactor阻塞项；改变分块或相关数值组织须重新审查。

本文件记录科研/运行能力的用户最终决定，不逐函数审批。§2、§3、§4的`KEEP`/`RETIRE`是已确定的保留或退休范围；各批实施范围以该批用户授权及最新完成记录为准，不扩大为其它退休或配置清理。D8c暂时保留，待明确具体输入、输出、调用者及是否实际运行SpaMosaic模型后再说明，不阻塞本批。普通无调用者且有权威替代的DEAD helper不列入决策表，后续按计划核对后清理。

## 1. 完整实验覆盖九数据集全部 KEEP

**v7A代表模型/算法基线，v7A suite的数据集列表只是调试子集。完整实验必须保留result_v6实际使用的全部9个数据集。** 未出现在v7A suite不能用于判断dataset adapter、training entry、analysis loader、metric pipeline或plotting逻辑可以退休。

### 1.1 名单与运行证据

证据是以下三个已有状态JSON及九份run_summary，均先核对大小再只读。对应训练、分析任务记录completed且returncode=0：

- [主suite记录](../../result_v6/suite_status.json)：MouseBrain、Human Embryo、MISAR-seq。
- [补齐实验suite记录](../../result_v6/missing_v3_matched_suite_status.json)：Human_Lymph_Node、Mouse_Spleen、Mouse_Thymus、Simulation、CRC_Stereo-CITE-seq。
- [SPATCH独立suite记录](../../result_v6/spatch_suite_status.json)：SPATCH。主suite的excluded_datasets=[SPATCH]只描述该分组，不能用来排除完整实验中的SPATCH。

| 数据集 / 决定 | 已核对的历史run_summary | 当前训练/数据入口 | 必须保留的适配 |
|---|---|---|---|
| MouseBrain / KEEP | [MouseBrain](../../result_v6/mousebrain/v3_bidirectional_sparse_fixed_lc0.1/epochs_200/run_summary.json) | run_mousebrain.py、model/multimodal_preprocessing.py | 三section；RNA/Metabolite/已有UNI作为HE，各50维；原spot/模态顺序、FP32及epoch0预检 |
| Human Embryo（HESTA RNA-only）/ KEEP | [Human Embryo](../../result_v6/human_embryo_harmony/run_summary.json) | run_human_embryo_rna_only.py、hesta_rna_utils.py | 七section、外部Harmony manifest、RNA50单模态、lambda_contrast=0；BF16及双向sparse OT |
| MISAR-seq / KEEP | [MISAR-seq](../../result_v6/misar_seq/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_misar_seq.py | RNA/ATAC50，dataset4→3→2→1；obs/var对齐；amp_dtype=none |
| SPATCH / KEEP | [SPATCH](../../result_v6/spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_spatch.py | 两section、RNA50/Protein15/HE50、DAPI去除与canonical ID；当前直接复用既有cache，BF16 |
| CRC_Stereo-CITE-seq / KEEP | [CRC](../../result_v6/crc_stereocite/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_crc_stereocite.py | CRC_003/CRC_006；RNA50/Protein50；共享基因/marker对齐；BF16 |
| Human_Lymph_Node / KEEP | [Human Lymph Node](../../result_v6/human_lymph_node/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_human_lymph_node.py及CRC helpers | A1/D1、基因ID、跨section非零基因过滤、RNA50/Protein20；BF16 |
| Mouse_Spleen / KEEP | [Mouse Spleen](../../result_v6/mouse_spleen/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_mouse_spleen.py及HLN/CRC helpers | 两section、ADT marker保序、RNA50/Protein20；BF16 |
| Mouse_Thymus / KEEP | [Mouse Thymus](../../result_v6/mouse_thymus/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_mouse_thymus.py及MISAR helpers | 四section、RNA obs的x/y、ADT固定marker轴、RNA50/Protein15；BF16 |
| Simulation / KEEP | [Simulation](../../result_v6/simulation/bidirectional_sparse_uot_fixed_lc0.1_seed42/run_summary.json) | run_simulation.py及MISAR helpers | 五section、RNA1000/ADT100原特征与50维模型输入、spfac/nsfac及spatial_domain；BF16 |

入口简写均位于scripts/。历史记录证明实际实验覆盖，不意味着本轮运行过当前v7A的九数据集训练，也不以v6模型替代v7A。历史SPATCH summary没有当前cache字段，不能倒推历史v6训练已加载现在的缓存。

九数据集都必须能经新的统一训练框架运行当前v7A，并具有明确分析入口。统一adapter/shared fit时保留原数据和评估差异，不强行统一超参数、预处理或算法。历史result_v6/result_v7A等只作为保护产物；目录名不决定代码结构，也不要求同名suite永久存在。

### 1.2 九数据集的分析入口和协议

以下是实际历史config和当前可用源码的静态证据，不是本轮重算认证。所有历史分析CLI都可能默认写旧目录，本轮不执行。后续保留功能时须显式传历史输入和新输出，不能重写旧结果。

| 数据集 | 当前保留入口/实现（scripts/）及已核对config | 必须保留的真实协议 |
|---|---|---|
| MouseBrain | analyze_mousebrain_v4_standardized.py → compare_mousebrain_kmeans_preprocessing.py；[config](../../result_v6/mousebrain/v3_bidirectional_sparse_fixed_lc0.1/analysis/standardized_embedding/config.json) | StandardScaler+KMeans，seed0、n_init20、max_iter300，v6 K2…12；全量ASW/CH/DBI；标签RegionLoupe/annotations/celltype/Y.l1/Y/group。v7A requested的independent保留K列表另存为明确协议 |
| Human Embryo | analyze_human_embryo_reference_metrics.py；[config](../../result_v6/human_embryo_harmony/analysis/config.json) | MiniBatchKMeans，seed42、n_init3、max_iter100、batch8192；K2/3/4/10/15/20/25；metric采样5000；section-mixing从标准化joint_space抽100001，再传指标函数；developmental section是生物发育时间，其混合指标仅作诊断，不作为应消除的batch；celltype；保留25叶簇质心加权Ward嵌套层次聚类 |
| MISAR-seq | analyze_misar_result_v5_standardized.py → analyze_result_v3_standardized.py/compare_misar_seq_kmeans_preprocessing.py；[config](../../result_v6/misar_seq/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | KMeans seed0、n_init20、max_iter300，K2…16；全量指标；五类真实标签；保留绘图K5/8/10/12/14/16 |
| SPATCH | analyze_spatch_result_v6_standardized.py → analyze_result_v3_large_standardized.py::analyze_spatch → compare_spatch_kmeans_preprocessing.py；[config](../../result_v6/spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | MiniBatchKMeans seed42、n_init20、max_iter300、batch4096；v6指标K2…20，保留K5/8/10/12/16/20；ASW各scope/有效标签按RandomState(42)抽10000并保存身份，CH/DBI全量；三类真实标签。v7A requested K5/8/10/12/14/16/20另保留，不互相覆盖 |
| CRC_Stereo-CITE-seq | analyze_result_v3_large_standardized.py::analyze_crc → compare_crc_stereocite_kmeans_preprocessing.py；[config](../../result_v6/crc_stereocite/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | MiniBatchKMeans seed0、n_init20、max_iter300、batch8192；K5/10/15/20/25；固定joint ASW10000及样本文件，CH/DBI全量；无ground-truth标签 |
| Human_Lymph_Node | analyze_result_v3_standardized.py → compare_human_lymph_node_kmeans_preprocessing.py；[config](../../result_v6/human_lymph_node/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | KMeans seed0、n_init20、max_iter300，K2…12；全量ASW/CH/DBI，保留K5/8/10/12；无ground-truth标签 |
| Mouse_Spleen | analyze_result_v3_standardized.py → compare_mouse_spleen_kmeans_preprocessing.py；[config](../../result_v6/mouse_spleen/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | KMeans seed0、n_init20、max_iter300，joint/independent K2…12；全量ASW/CH/DBI；无ground-truth标签 |
| Mouse_Thymus | analyze_result_v3_standardized.py → compare_mouse_thymus_kmeans_preprocessing.py；[config](../../result_v6/mouse_thymus/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | KMeans seed0、n_init20、max_iter300，K2…12；joint ASW按section分层固定10000，绑定section/barcode，CH/DBI全量；无ground-truth标签 |
| Simulation | analyze_result_v3_standardized.py → compare_simulation_kmeans_preprocessing.py；[config](../../result_v6/simulation/bidirectional_sparse_uot_fixed_lc0.1_seed42/analysis/standardized_embedding/config.json) | KMeans seed42、n_init20、max_iter300，K5/8/10/12；全6480 spots；truth=spatial_domain，有ARI/NMI及label ASW |

joint StandardScaler在合并数据上拟合，independent按section各自拟合；保留对应标准化空间及各batch metric内部协议。ASW为轮廓系数，CH/DBI为内部聚类指标；不得给无标签数据补造生物学ARI/NMI。训练空间K与分析空间邻居K各自记录，不混用。原标签列、有效标签过滤、采样身份、UMAP/空间绘图配置等完整细节以相应config和真实消费者为准，迁移前针对性补小接口测试。

## 2. 已由用户决定的范围

| 决定 | 能力/对象 | 执行边界 |
|---|---|---|
| KEEP | 当前v7A；一个StageMultiModalModel；双向candidate-sparse UOT；FAISS candidate retrieval | 不恢复v15A/feature graph，不共享pre/post权重，不改数学 |
| KEEP | 大规模训练需要的BF16、chunk、activation checkpoint | 固定precision/AMP/layout验收；不改变生产确定性设置 |
| KEEP | §1全部九数据集的训练、适配、分析、指标与必要绘图能力 | 可替换脆弱实现方式，不能取消数据集支持 |
| KEEP | 三个gene_imputation*目录 | 整个目录不改、不移动；维持现有路径、调用与输出格式 |
| KEEP | preprocessed_cache，尤其现有SPATCH缓存 | 原路径、格式、内容保留；不重算、不写回、不升级 |
| KEEP | results/和所有现有result_* | 仅历史产物保存/只读；不清理、不重算，不因此永久保留版本suite |
| RETIRE | dense OT生产路径、独立单向OT生产路径，以及仅服务它们的配置/兼容逻辑 | 已批准退休方向；先将保留入口迁成双向sparse，再删专属实现；保留双向所需正反检索/索引/top-k。不是本轮执行授权 |

## 3. 已决定的能力范围与未来删除影响

### 阅读规则：历史结果、插补与Git恢复

已核对的下面各项旧模型/入口/分析/验证源码及v7B/C JSON都在上述HEAD的`git ls-tree -r --name-only HEAD model scripts`中，Git可恢复**这些源码**。这不保证旧实验逐位重现，也不恢复原环境、raw数据、外部UNI权重、ignored产物或未跟踪配置。Batch 0的两份新验证脚本和本地`.pt`不在该承诺内，必须继续保留。

三个gene_imputation*目录没有对model/scripts的直接导入；其runpy/importlib依赖位于三个目录内部，外部消费历史SPATCH embedding、summary、metadata和RNA文件。以下能力代码均无该类直接依赖，但任何退休都必须保持这些产物路径和格式。历史结果继续存放和直接读取数组/JSON/CSV不需要旧trainer；分析所需loader/协议则必须先迁移保留，不能仅凭Git可恢复就直接删除正在被调用的文件。

### 3.1 v4/v5/v6历史suite和daemon

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 当前v7A suites不导入这些历史训练套件；其版本命令不是当前模型运行合同 |
| 当前训练/分析依赖 | 历史套件内部互导仍在，如v4c→v4b、v4c_large→v4_large；九数据集参数/分析协议需要迁出保留，不能整组先删 |
| 三插补目录依赖 | 无上述代码直接依赖；SPATCH历史产物合同必须保留 |
| 主要文件 | scripts/run_result_v4*_suite.py、run_result_v5_*suite.py、run_result_v5_spatch_daemon.py、run_result_v6_*suite.py、run_result_v6_spatch_daemon.py |
| 历史result保存/读取与Git | 保存/读取结果不需旧suite；现有源码可从核实HEAD恢复。仅历史复现的旧启动方法可以退出现行代码树 |
| 建议及复杂度 | **RETIRE，先迁保留能力**。减少重复命令拼装、版本路径常量、daemon监督/status和硬编码参数 |
| 删除后实际影响 | 旧命令不再可直接运行；九数据集通过统一入口运行当前v7A，历史结果原物不变。若发现仍有保留caller，先迁依赖再删壳 |

### 3.2 v7B/v7C post-OT scale消融

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | v7A post scale=.5；不调用B/C的.25/.75；SPATCH v7ABC suite包含三种scale |
| 当前训练/分析依赖 | B/C导入并修改v7A suite globals；差别可由同Stage普通参数表达，结果分析读取scale元数据无需训练B/C |
| 三插补目录依赖 | 无B/C代码直接依赖 |
| 主要文件 | model/configs/v7b_post_ot_graphsage_scale_0.25.json、v7c_post_ot_graphsage_scale_0.75.json；scripts/run_result_v7b_suite.py、run_result_v7c_suite.py、run_spatch_v7abc_suite.py |
| 历史result保存/读取与Git | B/C历史embedding/summary可继续读取；源码和两份JSON可从核实HEAD恢复 |
| 最终决定及复杂度 | **D1：KEEP普通可调post-OT scale；RETIRE v7B/v7C命名preset和专属suite壳，先迁保留调用者**。默认v7A仍为.5，不增加独立模型；去globals monkeypatch、历史命名配置及重复调度 |
| 删除后实际影响 | 将来取消B/C命名快捷入口，但仍可显式设置.25/.75或其他受支持scale；不改变v7A=.5，不影响旧结果。本轮不删除preset、suite或模型参数 |

### 3.3 G0 / self_path_mode

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | **保留原no_self_linear计算**，不再有模式参数；邻接self-loop和outer residual保持 |
| 当前训练/分析依赖 | legacy/no_adj_self及其注册self_linear已删除；保留入口无依赖，analysis仅存历史标签并保持不变 |
| 三插补目录依赖 | 无模式代码直接依赖 |
| 主要文件 | model/configure.py、model/model_component.py::WeightedResidualGraphSAGE、model/stage_model.py::_get_spatial_graph |
| 历史result保存/读取与Git | 读取历史embedding不需重执行self-path；旧实现可从核实HEAD恢复。旧weights的解释与重新加载另按state合同处理 |
| 最终决定及复杂度 | **D2已实施，见§12**：仅保留原no_self_linear计算，删除mode选择和两个self_linear模块；自环/outer residual保持。完整schema与构造RNG变化单列，严格有效参数映射后等价 |
| 删除后实际影响 | 不再支持另外两种self-path实验或旧mode参数；完整schema少两个key，同seed构造RNG及后续有效初值改变。已先冻结旧状态再逐名映射、strict加载，固定gate差0；没有生产RNG/旧checkpoint兼容层 |

### 3.4 context gate及alpha反传选项

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 原context gate关闭，现已按D3实际删除；普通attention gate保留。v7A topology refresh仍用空间context，与实验gate不同 |
| 当前训练/分析依赖 | D3后CLI/config/API不再能开启；无KEEP消费者依赖该实验。小型pooling/topology和普通gate验证保留，历史analysis metadata读取不改 |
| 三插补目录依赖 | 无gate代码直接依赖 |
| 主要文件 | model/model_component.py::OTGuidedAttention、model/stage_model.py；scripts/validate_microenvironment_context.py、benchmark_microenvironment_fullspot.py |
| 历史result保存/读取与Git | 结果读取不需重执行gate；源码可从核实HEAD恢复 |
| 最终决定及复杂度 | **D3已实施，见§10**：额外gate、reliability、alpha反传实验开关与专属fullspot benchmark实际退休；不留空模块或兼容alias |
| 删除后实际影响 | 不再支持该gate消融及专属性能研究；**普通gate和0.8 semantic + 0.2 context动态cost保持**，共享spatial pooling保留。原gate-only空context_embeddings输出移除；三个embedding阶段不变 |

### 3.5 dynamic OT fused/final source

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 当前source=ot；fused/final不参与当前主路径 |
| 当前训练/分析依赖 | 历史v4/v5等命令与experiment-C/context validators可选择fused/final；分析读取source说明不等于依赖其训练计算 |
| 三插补目录依赖 | 无source算法直接依赖 |
| 主要文件 | model/stage_model.py::prepare_ot_prior_refresh、runner动态prior helpers、历史v4/v5 suite、scripts/validate_experiment_c.py |
| 历史result保存/读取与Git | 历史embedding/summary直接读取；旧source实现可从核实HEAD恢复 |
| 最终决定及复杂度 | **D4已实施，见§9：仅KEEP ot刷新源；fused/final已实际退休**。source enum/CLI/API/config选择与fused刷新context特例已删除；B3以路径退休关闭 |
| 删除后实际影响 | 不再支持其它刷新源或旧source参数；fused/ot/final阶段embedding、ot topology与刷新时序保持。B3原始证据保留，状态为RESOLVED BY RETIREMENT OF UNSUPPORTED SOURCE PATH |

### 3.6 metacell及还原映射

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 使用full-spot；SPATCH现有schema=1 cache仍读取metacell=False身份字段 |
| 当前训练/分析依赖 | D5已删除generic/MouseBrain的metacell预处理选项并拒绝显式true；两个COSIE外部结果读取器保持。九数据集没有必须调用本项目生成/还原的依赖 |
| 三插补目录依赖 | 无本项目metacell实现直接依赖，保留其现有spot/结果格式 |
| 主要文件 | model/data_preprocessing.py、model/multimodal_preprocessing.py；scripts/run_preprocessing.py、train_stage_model.py、run_mousebrain.py |
| 历史result保存/读取与Git | 既有映射/结果继续保存；读取比较产物无需重新聚合；源码可从核实HEAD恢复，映射数据不由Git恢复 |
| 最终决定及复杂度 | **D5已实施，见§11**：三个生成/还原函数、聚合分支、三个API参数及专属默认/输出字段实际删除；11个生产文件净删99行 |
| 删除后实际影响 | 本项目不能再新建/还原metacell；显式true报不支持。full-spot九数据集、已有cache和外部比较读取保持；旧false可作无行为历史输入读取，不改写旧映射/结果，不重算 |

### 3.7 single-modality

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 当前调试suite未覆盖Human Embryo；完整九数据集需要RNA-only，不能依据局部suite删除 |
| 当前训练/分析依赖 | Human Embryo入口及single-modality validator实际依赖；RNA-only仍可使用双向candidate-sparse OT |
| 三插补目录依赖 | 无该训练模式直接依赖 |
| 主要文件 | scripts/run_human_embryo_rna_only.py、validate_single_modality_mode.py；model/stage_model.py、scripts/hesta_rna_utils.py |
| 历史result保存/读取与Git | 旧embedding可直接读取；代码可从核实HEAD恢复，但它属于当前必须支持的能力，不能以可恢复为由退休 |
| 建议及复杂度 | **KEEP**。通过普通dataset adapter+shared fit收敛重复入口，保留单模态输入/零crossview合同 |
| 删除后实际影响 | 删除会破坏Human Embryo统一v7A训练，违反九数据集KEEP要求。single-modality、合法无OT与独立单向OT不是同一能力 |

### 3.8 H&E / UNI输入

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | MouseBrain从RNA obsm['uni_feature']取得HE；SPATCH从既有HE特征/cache输入，均实际使用 |
| 当前训练/分析依赖 | 九数据集必要已有HE/UNI特征读取必须保留。generic build_section_modalities另外支持raw image+mask→UNI提取，当前固定v7A输入不执行该提取链 |
| 三插补目录依赖 | 无UNI提取代码直接依赖；仍需现有SPATCH embedding/metadata |
| 主要文件 | model/multimodal_preprocessing.py、model/image_preprocessing.py；scripts/run_mousebrain.py、run_spatch.py、run_preprocessing.py |
| 历史result保存/读取与Git | 结果和已有HE特征读取不需在线UNI模型；源码可从核实HEAD恢复，外部UNI权重不由Git提供 |
| 最终决定及复杂度 | **D6：KEEP H&E/已有UNI输入和raw image+mask→UNI特征提取**。允许后续收敛普通输入接口，不退休图像切patch、mask/权重/设备处理的必要能力 |
| 删除后实际影响 | 不退休这两种输入能力；直接删除会失去原图特征提取或九数据集必要HE输入。实现整理须先迁调用者，不能借重构重录SPATCH缓存或重新提取已有UNI |

### 3.9 blockwise candidate检索（不是自动fallback）

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 主命令选择FAISS IVF/Flat；blockwise是显式选择的exact候选后端，FAISS失败不会自动切换它，也不是dense OT |
| 当前训练/分析依赖 | 双向、checkpoint、experiment-C、context validators实际使用；生产CLI也可显式选择该后端 |
| 三插补目录依赖 | 无此OT候选实现直接依赖 |
| 主要文件 | model/faiss_candidate_search.py::build_faiss_candidates、_blockwise_exact_search；scripts/validate_bidirectional_ot_attention.py、validate_checkpoint_ot_attention.py、validate_experiment_c.py、validate_microenvironment_context.py |
| 历史result保存/读取与Git | 读取结果不需候选后端；源码可从核实HEAD恢复 |
| 最终决定及复杂度 | **D7：KEEP FAISS及生产CLI显式blockwise候选选择，也保留小型验证后端；不新增自动fallback**。这些是candidate retrieval后端，不能随dense OT删除 |
| 删除后实际影响 | 不退休该后端；删除会失去无FAISS时显式选择候选构造的能力并破坏现有验证消费者。后续仅收敛重复入口与参数传递 |

### 3.10 历史standalone analysis CLI

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 当前requested分析实际导入analyze_mousebrain_v4_standardized、analyze_result_v3_standardized及compare_*；SPATCH导入compare_spatch；这些函数仍有消费者 |
| 当前训练/分析依赖 | 九数据集loader/metrics/plot必须KEEP；v6 UMAP代码还导入CRC/MISAR/Thymus/Simulation数据helpers，其他绘图脚本复用它，须先迁依赖 |
| 三插补目录依赖 | 无这些analysis CLI直接依赖；其SPATCH历史输入及内部绘图代码不动 |
| 主要文件 | scripts/analyze_result_v3*_standardized.py、analyze_mousebrain_v4_standardized.py、analyze_misar_result_v5_standardized.py、analyze_spatch_result_v6_standardized.py、analyze_human_embryo_reference_metrics.py、compare_*、generate_result_v6_spamosaic_umap.py、generate_comparison_method_umaps.py、rerender_result_v6_umap_figures.py |
| 历史result保存/读取与Git | 保存结果不需旧CLI，但读取/分析需要当前这些文件内尚未迁出的loader和协议；旧CLI源码可从核实HEAD恢复 |
| 最终决定及复杂度 | **KEEP九数据集分析/历史读取；D8a KEEP方法间KMeans预处理/指标比较，迁后RETIRE版本壳；D8b KEEP跨方法UMAP，使用显式参数；D8c SpaMosaic风格输入/整合工作流暂时KEEP**。D8c须后续说明具体I/O、调用者及是否实际运行SpaMosaic模型；不阻塞C1。去replace+exec、argv/globals、硬编码历史output及自动prune，不退科研能力 |
| 删除后实际影响 | 将来退壳会改变旧命令入口，明确的新入口保留比较和UMAP功能；直接删整文件会破坏当前分析。D8c说明完成前保持现状，不把风格绘图名称当成实际运行外部模型的证据 |

### 3.11 历史benchmark / validation scripts

| 字段 | 核对与建议 |
|---|---|
| 当前v7A是否调用 | 训练不执行旧bench/validator；它们提供部分开发验证，不能当当前模型训练路径 |
| 当前训练/分析依赖 | 双向/checkpoint/single-modality验证覆盖保留能力；context/experiment-C/fullspot benchmark部分服务已决定退休的D3/D4研究，共享必要覆盖先迁。Batch 0不导入旧validator，但尚未覆盖所有旧接口 |
| 三插补目录依赖 | 无这些validator直接依赖；三个目录自身tests全部保护 |
| 主要文件 | scripts/validate_bidirectional_ot_attention.py、validate_checkpoint_ot_attention.py、validate_single_modality_mode.py、validate_experiment_c.py、validate_microenvironment_context.py、benchmark_microenvironment_fullspot.py |
| 历史result保存/读取与Git | 不需bench代码维持旧产物；这些旧脚本可从核实HEAD恢复；Batch 0新脚本/fixture不在此恢复承诺内 |
| 最终决定及复杂度 | **KEEP保留能力的必要验证覆盖；RETIRE迁完的重复/退休算法专属壳；按D3/D4退休专属大型context性能研究**。D3/D4已迁移保留的普通gate/checkpoint与小型pooling/topology验证；专属fullspot benchmark删除。旧fixture/报告/失败证据不清理 |
| 删除后实际影响 | 只退旧壳不丢必要验证；退休大型benchmark则停止维护该fullspot性能实验。不能把Batch 0参考、原FAIL/NaN证据当历史垃圾清理，也不运行大benchmark作本轮核查 |

## 4. 用户最终决定与剩余说明项

以下决定已明确，不再逐项申请确认。本轮仅执行D2，其他退休须在相应后续批次迁完依赖并完成验证后实施；不把范围决定解释为本轮全面清理授权。仅D8c保留具体工作流说明项，它不阻塞本批。

| 编号 | 最终范围 | 实施边界/影响 |
|---|---|---|
| D1 | KEEP普通可调scale；RETIRE B/C命名preset和suite | v7A默认.5；.25/.75可由普通参数表达，先迁调用者 |
| D2 | **已实施**：保留no_self_linear计算；legacy/no_adj_self及模式选择退休 | 两个self_linear key删除，schema/RNG/optimizer逐名核对；相同有效状态等价，不宣称同seed初值不变，见§12 |
| D3 | **已实施**：额外context gate、alpha实验及专属大型benchmark退休 | 普通gate、正常alpha反传、0.8 semantic + 0.2 context动态cost及小型验证保持，见§10 |
| D4 | **已实施**：仅KEEP ot；fused/final刷新源退休 | B3以路径退休关闭；阶段embedding、topology/刷新时序不变，见§9 |
| D5 | **已实施**：本项目metacell生成/训练及还原实现退休 | KEEP full-spot、schema=1 false身份、旧产物/映射及外部结果读取；不重算cache，见§11 |
| D6 | KEEP原图+mask→UNI提取以及已有H&E/UNI特征输入 | 两种输入链都保留，不触发已有特征重提取 |
| D7 | KEEP FAISS和显式blockwise候选检索 | 保留生产CLI及小型验证能力，不增加自动fallback |
| D8a | KEEP方法间KMeans预处理/指标比较；RETIRE迁完的版本壳 | 保留各方法真实协议，改为明确输入和新输出 |
| D8b | KEEP跨方法UMAP | 使用显式参数，保留必要loader/metric/plot，不依赖suite globals |
| D8c | 暂时KEEP SpaMosaic风格输入/整合工作流 | 后续说明具体输入、输出、调用者及是否实际运行SpaMosaic模型；不阻塞C1、不先删 |

## 5. C1 已完成的实施范围与验收

### 5.1 唯一目标和保护对象

让已有SPATCH model-ready cache的有效性依赖真正影响产物解释/兼容性的输入及预处理身份，解除整个训练脚本源码hash的硬门槛。它是**加载接口/接受条件调整**，不是完全无行为变化的纯代码搬动。

唯一既有缓存为`preprocessed_cache/spatch/v6_harmony_n50_hvg3000`。schema=1，boundary=model_ready_feature_dict_after_PCA_Harmony；section1/section2各665399/403563 spots；RNA50/Protein15/HE50，spatial N×2，float32。manifest SHA仍为`4fdf8ed1a3c4beaa856eff4e91e3970639543718afce45a5527cfb159754fba7`。

既有cache不删除、不移动、不改格式、不重算、不写回、不再次执行normalization/PCA/Harmony。C1不建CacheManager、版本数据库、cache migration framework或自动升级体系；不提前搬目录，不涉及dense/单向退休、模型/fit合并或数值bug fix。

### 5.2 C1修改前的hash和接受条件

C1修改前没有独立的“预处理语义hash”。原loader分别执行整源码SHA门槛、parameters字典相等、raw stat比较和缓存内容检查；运行报告另记整个manifest文件的SHA。下表行号均指修改前核对位置。

| 当前记录/门槛 | 实际内容与源码位置 | C1保留或调整 |
|---|---|---|
| preprocessing_sources两份SHA硬比较 | run_spatch.py:359–369；对完整run_spatch.py及model/data_preprocessing.py字节计算SHA，包含CLI、训练配置/调用、保存/日志等不影响缓存解释的源码 | 删除整文件相等的接受门槛，保留历史SHA作provenance；预处理语义冲突仍拒绝，不把任意预处理算法变化自动判兼容 |
| cache_parameters整个字典相等 | run_spatch.py:177–190、353–358；sections、modalities、n_comps、hvg_num、hvg_num_by_modality、target_sum、use_harmony、metacell、memory_efficient、retain_processed、dapi_removed | 按字段解释实际身份并报告冲突；保留section/模态/order、维度及HVG/Harmony/DAPI/metacell等必要语义。memory_efficient有不同预处理concat路径，未证明等价前保留构建约定；retain_processed保留来源，不要求消费端重现保存预处理对象的动作 |
| source_files | 构建记录每section/modality的raw绝对path、size_bytes、mtime_ns、SHA（:193–209）；加载要求raw存在且path/size/mtime全等（:212–236、370–371），**加载时不重核raw SHA** | 保留记录和可用内容身份；解除路径/mtime本身的硬耦合。默认只消费所选cache，不自动打开raw；显式输入身份冲突或无法确认分别明确拒绝/报告，不能忽略请求 |
| arrays与metadata_artifacts | 数组记录relative_path、section/kind/modality、size、SHA、shape、dtype；loader检查文件/大小/SHA/shape/dtype/finite；metadata记录size/SHA，当前loader查SHA后复制到output（:374–396） | 保留一次加载的内容完整性检查；补schema/boundary/dataset、section/spot/order、跨数组N、模态/真实维度、metadata坐标/ID核对；通过必要检查后仅向新output复制必需metadata |
| 整个manifest SHA | loader:403–408仅记录来源指纹；manifest内容含上述记录、alignment、software_versions、created_at等 | 继续记录原manifest身份，不把软件版本/创建时间/训练源码变化单独作为失效理由，不改写manifest |

C1修改前两份被hash源码匹配manifest，不能宣称既有cache已经加载失败。原代码未充分检查dataset/boundary、跨数组N、metadata行序等；接手核对只读小manifest和源码，不标记真实数组/metadata动态检查已通过。C1实施与小替身验证结果单独记录于§5.6。

### 5.3 C1后的判断逻辑

1. **明确使用既有cache。** 主SPATCH入口接收指定cache；缺目录/manifest、不支持schema/dataset/boundary、请求build时明确报错。suite不追加build，main不落回inspect_full_inputs/preprocess_modalities_sequentially/save_preprocessed_cache。
2. **核对输入/预处理身份。** section身份与顺序、模态集合/顺序、n_comps/HVG/分模态HVG/target_sum/Harmony/DAPI/metacell及保留构建约定与明确请求相容。manifest记录模态顺序为RNA/Protein/HE，Stage解析顺序为HE/RNA/Protein，分别保持，不要求两者统一。训练epochs/lr/loss/OT/GraphSAGE/AMP/chunk/checkpoint不属于本缓存预处理身份，不能作为拒绝理由。
3. **正确处理raw身份。** CLI默认data_dir=None，默认以所选cache及其记录来源为输入；显式同来源data_dir仅作缓存来源选择，不冒充当前raw文件核验。可选`--input_identity_manifest`复用含source_files的JSON声明，逐section/modality核对SHA及size；它证明声明与cache来源记录相容，不读取或重新hash raw，`current_raw_files_verified=False`。仅路径或mtime不同而声明内容身份相同不判失效；显式另一data_dir缺身份凭据时拒绝，不悄悄使用旧cache。
4. **保持数组和顺序。** 按manifest原顺序读取，拒绝重复/缺失/多余section或数组项、不支持kind、模态冲突及N/维度矛盾；RNA50/Protein15/HE50，不能误把Protein要求成50。历史Protein在去DAPI后有16个marker，原PCA分支实际输出15维；feature维度与marker数分别核对。
5. **检查metadata与文件完整性。** 保留缓存文件size/checksum、shape/dtype、finite；metadata按既有section/spot行序与spatial的x/y、canonical ID及数量核对，不自动排序修补。不改float32输入、内存拷贝或显存语义，不逐epoch重新扫描。
6. **来源如实记录。** 原manifest和source_files只读；cache_info区分历史来源与当前请求，记录`current_raw_files_verified=False`和feature行身份的证据限制。新summary的`input_data_path`指向实际cache，`input_data_kind=model_ready_preprocessed_cache`；不把未经核验的args.data_dir冒充输入，不改历史summary、embedding文件名/顺序或三个插补接口。

**历史证据限制：**schema没有在每个feature数组内部绑定ordered barcode。SHA证明字节与manifest一致，metadata/spatial逐行核对能发现可观察的顺序/坐标矛盾，但不能独立证明所有feature行的生物学身份。普通合规旧cache不能仅因缺这项历史证明而被判失效；明确记录限制，不构造新证据、不补写schema、不重算。

### 5.4 已授权的精确修改文件与函数

| 文件 | 限定修改点 | 不扩大的范围 |
|---|---|---|
| scripts/run_spatch.py | 原cache_parameters字段/值与schema保持，仅补docstring；loader新增必要schema/输入身份/对齐校验，retain_processed保留来源但不作硬门槛，其余未知或冲突参数明确拒绝；parse_args/main指定cache、可选input_identity_manifest、build拒绝和无预处理回退；cache_info及summary最小来源说明 | 不改模型构造有效值、loss/AMP/chunk/checkpoint、Adam、OT刷新或fit；不修改历史预处理数学，不顺手删除预处理helpers |
| scripts/run_spatch_v7abc_suite.py | training_command的manifest缺失处理；validate_training的新运行cache-mode仅loaded，保持缺cache明确失败 | 不执行suite、不改版本/数据集/训练数值参数、不重写历史built记录、不夹带B/C退休 |
| refactor_checks/c1_cache_contract_20260915/ | 小型schema=1 substitute、测试及报告；先冻结原加载接受/拒绝机制，再测新合同 | 不指向真实cache/result，不覆盖P0参考或旧报告 |
| REFACTOR_PLAN.md、USER_DECISIONS.md | 用户最终决定、C1实际范围、检查结果和行为变化记录 | 区分小替身动态验证、固定P0重放与未动态覆盖的真实全量路径 |

`model/data_preprocessing.py`在C1中仅只读核对语义，**没有必要修改**。不新建data_io框架；后续职责提取仍按计划单独进行。

### 5.5 必须先建立的小型cache substitute验证

只在refactor_checks新目录使用两section、RNA50/Protein15/HE50、少量spot的schema=1替身；raw identity用几字节合成文件或预给digest。测试使用真实loader和入口接受逻辑的必要替身/spy，不跑训练，不复制预处理数学，不扫描真实h5ad。当前实现的拒绝记录与C1的新预期分开，不能把行为改变伪称原实现已通过。

| 用例 | 新合同预期 |
|---|---|
| 正常旧schema、数组/metadata一致 | 原值、dtype、section/spot/模态顺序保持，加载成功 |
| 仅训练源码hash、函数位置、日志或训练超参数变化 | 加载成功，证明缓存与无关训练组织解耦 |
| section缺失/多出/交换顺序、重复记录 | 明确指出section identity/order冲突 |
| 各模态/spatial/metadata spot count不一致或可观察行序错配 | 明确拒绝；不自动排序、截断或对齐修补 |
| 缺失/多余模态、维度错误（特别是Protein误写50） | 明确模态/维度冲突 |
| schema/dataset/boundary、n_comps/HVG/Harmony/DAPI/metacell等必要身份冲突 | 分项拒绝，不自动修改请求参数 |
| raw路径/mtime变化，但可信内容身份相同 | 不因路径/mtime单独拒绝 |
| 显式请求raw digest与source_files冲突 | 明确输入身份错误，不build |
| 显式请求另一raw输入（含显式另一data_dir）但缺可信身份凭据 | 停止该次加载，明确无法确认兼容，不悄悄使用旧cache |
| 文件缺失、大小/SHA/shape/dtype错误、NaN/Inf | 完整性失败，原cache不变 |
| metadata交换行、section/x/y/canonical ID冲突或重复ID | 对齐失败；不修改或重写metadata |
| 合规旧schema缺feature数组内barcode绑定 | 记录证据限制，不能以此自动判失效或生成新cache |
| suite/main缺cache或manifest，或用户传build开关 | 明确失败；预处理/构建调用为0 |
| 所有成功/失败分支的I/O事件 | normalize/PCA/Harmony/build调用为0；cache写入为0；仅成功后的必需metadata副本写独立新output |

### 5.6 C1最终结果与覆盖限制

**C1：PASSED WITH DECLARED COVERAGE LIMITS。** 实际生产修改仅为scripts/run_spatch.py和scripts/run_spatch_v7abc_suite.py；model/data_preprocessing.py未改，D1–D8退休、dense/单向删除、模型/训练/分析重构及数值bug fix均未执行。新产物仅位于refactor_checks/c1_cache_contract_20260915，完整报告见 [C1_REPORT.md](../../refactor_checks/c1_cache_contract_20260915/C1_REPORT.md)。

| 验证 | 结果与证据 |
|---|---|
| 原加载合同冻结 | [old_behavior.json](../../refactor_checks/c1_cache_contract_20260915/old_behavior.json)记录7项旧机制；保留原源码hash/raw stat拒绝等行为，没有将其改写为新合同 |
| 新cache substitute合同 | [new_behavior_run03.json](../../refactor_checks/c1_cache_contract_20260915/new_behavior_run03.json) **73/73 PASS**；此前两轮62/71项报告随覆盖扩展保留，未覆盖旧报告。成功/失败分支的预处理与build调用为0、cache写入为0 |
| deterministic CUDA FP32固定P0重放 | [p0_fp32_check.json](../../refactor_checks/c1_cache_contract_20260915/p0_fp32_check.json)：自身配置 **13/13 PASS，差0**；同layout四checkpoint对照差0 |
| SPATCH BF16固定P0重放 | [p0_bf16_check.json](../../refactor_checks/c1_cache_contract_20260915/p0_bf16_check.json)：自身配置 **13/13 PASS，差0**；同layout四checkpoint对照差0 |
| cross-chunk已知敏感性 | 仍为FAIL，整个跨布局对照对象与历史记录精确相同；FP32一步参数最大差6.5170228e-5，BF16为0.00199983548，BF16更新后FP32 final eval embedding为0.00930213928。两次check总退出码1均来自该已知FAIL，不改写为PASS |

小替身使用真实loader，并动态覆盖main到Stage构造前的sentinel边界；未读取百万spot的真实SPATCH缓存数组，未训练100+/200轮，最终生产summary/export仅静态核对。声明输入身份与历史source_files相容，不代表验证了当前raw文件；旧schema仍不能独立证明所有feature行的生物学身份。上述限制已保留，不能把小替身PASS扩写为全量训练、导出或所有下游动态认证。

现有缓存、三个插补目录、results/和所有result_*不改、不移动、不重算；P0原fixture/expected、验证脚本及PASS/FAIL/NaN记录保持，C1保护核对见 [protection_check.json](../../refactor_checks/c1_cache_contract_20260915/protection_check.json)。C1阶段到此结束；随后另行授权的P1结果记录如下。

## 6. P1实施状态（2026-09-15，P1阶段记录）

**P1已完成保留入口迁移与本批验收，未执行P2或D1–D8其他退休。** 完整九数据集、单模态、H&E/UNI、BF16/chunk/checkpoint、FAISS及显式blockwise、所有保护目录的KEEP决定不变。没有因v7A suite的调试子集缩减实验覆盖，也没有删除B/C壳、self_linear、context gate、fused/final刷新分支或metacell。

generic、MouseBrain、CRC共享链（CRC/MISAR/SPATCH/Human Embryo/HLN/Spleen/Thymus/Simulation）及保留smoke/验证入口在启用OT时显式双向sparse初始化、forward和refresh；显式dense或有效单向请求由入口拒绝。Human Embryo RNA-only及合法disable_uot场景保持。原已双向的v7A配置固定gate通过；原默认dense/单向入口按已批准模式迁移处理，不承诺旧算法数值。

入口小测试78/78、generic/smoke16/16、checkpoint validator对照通过；C1原73项回归通过；冻结FP32/BF16各13/13与同布局checkpoint对照差0。原cross-chunk FAIL、旧smoke断言问题及测试夹具纠正记录均保留。小例不代表真实九数据集训练、真实百万spot缓存加载或生产分析/导出已动态验证。

详细表与剩余引用见 [P1_REPORT.md](../../refactor_checks/p1_entry_migration_20260915/P1_REPORT.md) 和 [REFACTOR_PLAN.md §15](REFACTOR_PLAN.md)。目前已识别保留入口均已迁移，具备P2调用者前提；底层dense/独立单向和专属残留仍待P2实际删除与针对性复核。P2必须保留Stage合法无OT直通及双向共享helpers；历史suite/分析壳不因结果版本号或dense字符串就整文件删除。D8c仍暂不删除。本轮在此停止。

## 7. P2实施状态（2026-09-15）

**已实际退休dense OT与独立单向candidate-sparse生产路径。** 删除dense文件/Stage接口、单向solver/helpers/遍历、模式CLI/参数/专属配置与空输出占位；无生产兼容shim。双向共享候选检索、索引/topk/同coupling、FAISS IVF/Flat及显式blockwise、普通attention、noOT全部KEEP。完整九数据集、RNA-only、H&E/UNI、BF16/chunk/checkpoint及历史产物范围不变。

P2相对已验收P1起点20生产文件净删1195行。P2 19项、P1适配70项、generic15项、C1原73项均PASS；冻结FP32/BF16各13/13及同layout checkpoint差0，原cross-chunk完整对象一致且仍FAIL。state/有效参数/optimizer/RNG相同，self_linear未动。详细删除、非数值schema/诊断变化、验证适配和未覆盖范围见 [P2_REPORT.md](../../refactor_checks/p2_ot_retirement_20260915/P2_REPORT.md) 及 [REFACTOR_PLAN.md §16](REFACTOR_PLAN.md)。

历史v4–v6 launcher仍含旧命令tokens，新CLI自然拒绝；没有它们对被删core的真实Python依赖。B/C壳、context/fused/final、metacell与其他D1–D5实现未清理；D8c仍暂不删除。P0/C1/P1证据和保护目录不修改，真实大数组/训练/分析未执行。当前无KEEP入口依赖被退休实现，P2完成后停止；下一批仅建议按计划单独处理F-config的最小B1/B2问题，尚未执行。


## 8. F-config-A 实际状态（2026-09-16）

**B1/B2已修复并通过本批验证。** 仅MouseBrain配置构建/传参/记录发生生产修改，未改变能力保留范围。优先级为基础默认 < model/preset < dataset/input JSON < 显式CLI；None表示未提供，布尔显式False不会丢失。目录与summary采用resolved实际值，当前v7A真实命令有效配置保持。

配置矩阵55/55；P2 19/19、P1 70/70、generic 15/15、C1 73/73；固定FP32/BF16各13/13及同布局checkpoint差0。原cross-chunk仍为KNOWN_NUMERICAL_SENSITIVITY，未改PASS。证据和范围见 [F-config-A报告](../../refactor_checks/fconfig_a_20260916/REPORT.md) 与 REFACTOR_PLAN §17。

B8、P3a、D2–D5、训练合并未执行；B3/source联动与generic smoke后置配置策略只记录，不夹带修复。完整九数据集、三插补目录、缓存及历史结果保护决定不变。本批到此停止。


## 9. D4 实际状态（2026-09-16）

**D4完成；B3：RESOLVED BY RETIREMENT OF UNSUPPORTED SOURCE PATH。** 动态refresh只有ot_embeddings.detach及同一ot阶段的self-excluded context路径；source CLI/config/API选择实际删除，旧请求明确拒绝。ordinary gate与.8/.2 context cost、三阶段embedding、final导出、九数据集/单模态/noOT能力保持。

D4小测试7/7；F-config-A 55/55、P2 19/19、P1 70/70、generic 15/15、C1 73/73；FP32/BF16固定各13/13及同layout checkpoint差0，原cross-chunk FAIL仍按KNOWN_NUMERICAL_SENSITIVITY保留。旧证据和保护目录不改，真实数据/完整训练未运行。

生产13文件净删91行，没有KEEP调用者依赖退休source。11个旧launcher命令文字、7个analysis历史metadata读取保留分类；两validator迁移共享刷新覆盖，不退休D3其它能力。详见 [D4报告](../../refactor_checks/d4_ot_refresh_20260916/REPORT.md) 与 REFACTOR_PLAN §18。本批停止，未进入D3/D5/D2/B8/P3a。


## 10. D3 实际状态（2026-09-16）

**D3完成并停止。** 额外attention context gate、513维分支、reliability、alpha实验开关、专属CLI/config/空输出与大型benchmark实际删除。普通512维gate及confidence/正常softmax alpha保持；OT refresh的同源self-excluded spatial context与.8/.2cost完整保留。D2 self_linear未动，无额外废弃预注册模块需映射。

D3 8/8、D4 7/7、F-config-A 55/55、P2 19/19、P1 70/70、generic 15/15、C1 73/73；两个小型validator PASS；固定FP32/BF16各13/13与同layout checkpoint差0，原cross-chunk FAIL保持。148个state key/optimizer参数、有效参数与RNG精确一致；旧证据只读，检查进程仅投影已严格断言的退休配置/空gate输出。

13个生产文件净删774行，无KEEP依赖退休逻辑。所有分析、三插补目录、真实cache与历史results保持。详见 [D3报告](../../refactor_checks/d3_context_gate_20260916/REPORT.md) 与 REFACTOR_PLAN §19。本批不进入D5、D2、B8、P3a或其它重构。


## 11. D5 实际状态（2026-09-16）

**D5完成并停止。** 本项目metacell生成、聚合训练输入和metacell→original spot还原已实际退休：删除三个专属函数、聚合/映射分支、三个API参数、默认字段与专属summary占位。11个生产文件净删99行。显式顶层/nested metacell=true在数据读取前明确拒绝；旧false保留无行为解析，不新增兼容框架。

完整九数据集的full-spot和RNA-only能力保持，未发现KEEP流程或三插补目录依赖被删实现。SPATCH schema=1 manifest的metacell=false仍是历史身份，loader及接受条件不变，不改manifest、不重算。两个外部COSIE结果读取器继续工作；D8c保留，只移除五个无行为调用参数。历史mapping/结果、普通spot信息和OT索引不删。

D5 28/28、入口/外部读取9/9；D3 8/8、D4 7/7、F-config-A 55/55、P2 19/19、P1 70/70、generic 15/15、C1 73/73全部PASS。冻结FP32/BF16各13/13与同layout checkpoint差0，原cross-chunk FAIL完整保留。15组真实full-spot准备的小替身对照逐项一致；没有真实normalize/PCA/Harmony、百万spot加载或真实训练。

详细证据、残余分类和覆盖限制见 [D5报告](../../refactor_checks/d5_metacell_20260916/REPORT.md) 与 REFACTOR_PLAN §20。所有旧证据和保护对象保持；未进入D2、B8、P3a或其它清理。


## 12. D2 实际状态（2026-09-16）

**D2完成并停止。** legacy/no_adj_self与self_path_mode选择实际退休，只保留原no_self_linear计算。两个注册但不参与计算的self_linear删除；正常空间自环、weighted聚合、outer residual及activation/dropout/norm顺序保持。pre/post仍独立，pre scale1、v7A post scale.5且仍可调；.75和scale0行为均动态验证。

完整schema唯一少graphsage.self_linear.weight和post_ot_graphsage.self_linear.weight两个float32[128,128]key；小例148→146、有效梯度名称62个不变。删除前已记录完整state/optimizer/RNG及初始化顺序。相同seed构造RNG与34个保留张量初值改变，未加兼容RNG层；按名称/shape/dtype严格映射旧有效参数并恢复forward RNG后，计算逐位等价。旧state不能未投影直接strict加载，本批未增加永久兼容接口。

D2 9/9；D5 28+9；D3 8/8、D4 7/7、F-config-A 55/55、P2 19/19、P1 70/70、generic 15/15、C1 73/73全部PASS。固定FP32/BF16各13/13及同layout checkpoint差0，明确标记APPROVED_SCHEMA_RETIREMENT；原cross-chunk完整对象/FAIL保持。没有新fixed-layout失败、P0重录或expected改写。

4个生产文件净删36行；九数据集/插补KEEP无退休路径依赖，历史analysis标签及所有保护对象保持。详见 [D2报告](../../refactor_checks/d2_self_path_20260916/REPORT.md) 与 REFACTOR_PLAN §21。真实数据/完整训练/大cache未运行。本批未进入B8、P3a、D1旧suite清理或训练结构迁移。


## 13. F-config-B / B8 实际状态（2026-09-16）

B8完成：删除23个模型NO-OP默认字段、4个预处理NO-OP默认字段和2个未消费CLI；明确拒绝这些旧字段、已退休配置及已核实的入口错位/不消费参数。不实现假字段所声称的功能，不建立全局配置schema、alias或training/config.py。保留65个真实模型默认字段，入口/data/运行身份按消费者单列。字段表见 [模型矩阵](../../refactor_checks/fconfig_b_20260916/model_consumer_matrix.md) 和 [入口矩阵](../../refactor_checks/fconfig_b_20260916/RUNNER_CONSUMER_MATRIX.md)。

全部九数据集训练/分析范围保持，single modality、双向sparse、FAISS/blockwise、H&E+mask→UNI、已有HE特征、schema=1/metacell=false缓存身份保持。post scale仍普通可调参数，当前v7A=.5、.75真实生效；不删除B/C历史壳。真实v7A命令和完整有效resolved值不变，相对D2没有新增state/schema/初始化RNG变化。D2旧self_linear批准退休规则不变。

B8专项34/34及16/16组PASS；D2/D5/D3/D4/F-config-A/P2/P1/generic/C1共293项PASS。固定FP32/BF16各13/13、同layout checkpoint差0；原cross-chunk保持已知FAIL。原P0配置进入新拒绝边界的首次失败保留，逐项核实后仅在新checker内投影23个NO-OP及6个P2旧输入字段；expected/有效参数/数值断言未改。

7个生产文件新增90/删除41行；无用途未决的已审计字段或KEEP执行依赖。详细边界和未覆盖范围见 [B8报告](../../refactor_checks/fconfig_b_20260916/REPORT.md) 与 REFACTOR_PLAN §22。所有保护对象、旧证据保持；未执行真实训练/预处理/百万spot缓存。具备进入P3a的配置清理前提，但本轮停止，不自动进入P3a、B10或P4/P5。


## 14. P3a 实际状态（2026-09-16）

P3a完成：新增简单training/config.py，提取JSON读取、递归merge、None-aware显式覆盖、helper参数选择及来源/JSON记录工具。保持F-config-A优先级和B8支持面；所有dataset默认、字段映射、模态字面量和四replace+exec包装能力仍留原处。未新增alias、配置框架或训练层级。

Stage显式config只消费完整resolved副本，None纯模型默认API保持；当前16条KEEP调用无partial依赖。旧partial库调用需先resolve，不再由constructor补默认。模型state/schema/同seed初始化RNG及所有数值路径相对B8一致，无新增D2投影。

解析23/23组、Stage专项11/11通过；B8至C1全CPU343/343通过；FP32/BF16各13/13及同layout checkpoint差0。原cross-chunk完整FAIL保持。Mouse仅新增source_layers来源metadata，旧output/summary有效字段不变；其它artifact系统不重写。

11个生产文件新增210/删除144行，全部保护对象和旧证据保持。完整说明见 [P3a报告](../../refactor_checks/p3a_config_20260916/REPORT.md) 与 REFACTOR_PLAN §23。具备进入F-data(B10)的前提；本轮到此停止，不执行B10、P4a、P3b、P4b/P5或analysis。


## 15. F-data(B10) 实际状态（2026-09-16）

B10完成：仅Simulation/Mouse Thymus两runner，audit接受明确data_dir/output_dir，在唯一parse之后执行。help退出0且无data/audit/output副作用；显式新路径不再被旧全局路径覆盖。原默认路径、参数、replace+exec锚点、section/模态/marker/spfac/nsfac及预处理数学保持。原audit CSV/JSON schema和内容保持；输出位置允许改为当前resolved位置。

新增17/删除16行。入口专项17/17、audit/adapter12/12、P3a配置与B8至C1回归366组通过；FP32/BF16各13/13及同layout checkpoint差0，原cross-chunk保持已知FAIL。旧fixture/expected、历史结果/audit及保护对象保持，无真实训练/预处理/百万spot读取。

详见 [B10报告](../../refactor_checks/b10_parse_io_20260916/REPORT.md) 与 REFACTOR_PLAN §24。具备进入下一数据批次的前提，本轮停止，不自动执行B13、B14、P4a或其它批次。

## 16. P4a 实际状态（2026-09-16）

按用户本批授权完成四个训练入口的纯组织重构：HLN/Spleen普通调用CRC，Thymus/Simulation普通调用MISAR；不再依赖基础源码读取/replace/exec、argv注入、文本锚点或globals修改。最小data_io普通函数承载实际复用的数据读取/轴准备，唯一必要UMAP调用者仅更新import/数据调用。基础CRC/MISAR行为及模态身份保持，训练loop/数学未改，dataset默认未统一，B10 audit函数保持。

先冻结旧行为，再验证12组完整数据/身份/顺序/metadata/resolved/训练handoff精确一致；9组异常与6组完整导出对照PASS。20份resolved与B10记录一致；轻量回归406/406，固定FP32/BF16各13/13、同layout checkpoint差0。原cross-chunk FAIL/KNOWN_NUMERICAL_SENSITIVITY、D2两key退休合同保持，无新数值投影或expected改写。

10个生产文件新增632/删除584行；1024项起点状态与34个保护目录根核对、旧专项630项证据保持。范围和真实数据未动态覆盖限制详见 [P4a报告](../../refactor_checks/p4a_adapters_20260916/REPORT.md) 与 REFACTOR_PLAN §25。B13未执行、B14未执行、P3b未执行、P4b未执行、P5未执行、analysis重构未执行、未运行真实大规模训练。完成P4a后停止，下一批等待用户明确指定。

## 17. P3b 实际状态（2026-09-16）

按用户本批授权完成九数据集显式defaults及单次配置解析。普通get_dataset_defaults和少量常量保留全部真实差异；原training/config工具直接复用，新增两个小型解析/记录函数。SPATCH十个后置运行设置提前解析，CRC/SPATCH原模型配置块提普通builder；四个P4a adapter继续普通调用。训练summary新增resolved记录，原字段、训练循环和数学保持。

27组配置old/new完整一致，另20份与P4a记录一致；18组真实入口/交接、19项False/零值/独立默认检查、9组异常、6组导出通过。既有轻量回归406/406；固定FP32/BF16各13/13、同layout checkpoint差0，原cross-chunk FAIL保持。checker展示/路径迁移失败证据保留，无新数值投影或expected改写。

10个生产文件新增742/删除323行，净增419。保护核对1470项起点状态与34个根stat，508项本批旧基准保持。详细默认来源、调用链、验证与限制见 [P3b报告](../../refactor_checks/p3b_defaults_20260916/REPORT.md) 与 REFACTOR_PLAN §26。B13未执行、B14未执行、P4b未执行、P5未执行、analysis重构未执行；未运行真实大规模训练、真实预处理或cache重建。P3b完成后立即停止，无P4b授权。


## 18. P4b-1 实际状态（2026-09-16）

按用户本批授权，仅将CRC共享训练runtime迁至training/fit.py，13个原定义逐字保持，四个runner只改import。CRC/MISAR/SPATCH/Human Embryo直接消费新fit，HLN/Spleen经CRC、Thymus/Simulation经MISAR到同一函数。训练顺序、AMP/scaler、refresh/final eval、模型数学、P3b配置、P4a数据适配和各入口保存合同保持。MouseBrain及generic整文件不变。

5个生产文件新增473/删除436行，净增37。修改前已冻结6组事件和4组CUDA一步；修改后事件、loss、全部有效梯度、Adam参数/state、embedding及RNG精确一致，人工epoch只运行指定6/7轮标签，不跑真实200轮。RNA-only七section启用/禁用UOT均通过。P3b27配置/18交接、P4a12适配/9异常/6导出和既有CPU406全部PASS；固定FP32/BF16各13/13、同layout checkpoint差0，原cross-chunk FAIL保持。测试搭建失败证据保留。

保护核对1916项起点、34个根、508项P3b证据和13项旧fit基准，未修改既有expected或fixture。详见 [P4b-1报告](../../refactor_checks/p4b1_fit_20260916/REPORT.md) 和 REFACTOR_PLAN §27。MouseBrain尚未迁移，generic trainer尚未迁移；B13未执行、B14未执行、P4b-2未执行、P4b-3未执行、P5未执行、analysis未重构；未运行真实大规模训练、真实预处理或cache重建。P4b-1完成后停止，没有下一小批授权。


## 19. P4b-2 实际状态（2026-09-16）

按用户本批授权，仅接入MouseBrain正式loop。独立epoch0 train/no_grad保留原处，随后调用同一training.fit；数据、配置、模型构造、保存未搬。fit以四个明确参数保留lambda schedule、step后梯度/输出、不含耗时的history和epochs0行为，CRC默认路径不变。三个prior helper经过9组resolved输入/调用比较后复用，lambda_for_epoch原样迁出；没有callback或dataset策略层。

2个生产文件+66/-137，净删71。修改前冻结17项旧基准；六组事件、三组CUDA一步（2/3 section与schedule）、epoch0 RNG专项、完整loss/输入及参数grad/Adam/state/RNG、91项小artifact文件对照全等。P4b-1六事件/四一步及八条链保持，既有CPU406/P3b/P4a/B10/C1全部PASS；固定FP32/BF16各13/13、same-layout checkpoint差0，cross-chunk原FAIL保持。scope checker一次AST括号匹配失败证据保留，最终140项通过，无生产差异。

2362项起点、34根stat、508项P3b证据、13项P4b-1旧fit证据与17项本批基准保持，HEAD未变、staged为空。详见 [P4b-2报告](../../refactor_checks/p4b2_mousebrain_20260916/REPORT.md) 和计划§28。generic trainer尚未迁移；P4b-3未执行、P5未执行、B13未执行、B14未执行、analysis未重构；未运行真实MouseBrain/九数据集完整训练、真实预处理，未重建cache。完成本批后停止。

## 20. P4b-3 实际状态（2026-09-16）

按本批授权，仅迁generic正式loop。原数据/config/Stage/初始prior/保存留runner；唯一训练主体为training.fit.iter_fit_model，按明确yield_every交回周期保存结果并最后交回final结果，既有train_small_crc_model保留签名/tuple返回。只新增record_loss_weights、refresh_ot、yield_every三个参数，保持generic四字段history、noOT与周期weights；无callback、Trainer或artifact系统。

两个生产文件+106/-97。修改前58文件冻结；12 CPU/3 CUDA专项保持完整loss/输入及参数grad/Adam/state/RNG/返回/weights，bundle、preprocess_config替身、真实synthetic、单section/noOT、普通epochs0与smoke0→3均保持。人工refresh与final199/200时序一致。P4b-1六事件/四CUDA、P4b-2六事件/三CUDA、CPU406、P3b/P4a/B10/C1通过；固定FP32/BF16各13/13、same-layout checkpoint差0，cross-chunk原FAIL保持。123项结构核对通过；模型/data_io/其它runner未改。

2918项起点/34根stat/508项P3b/13项P4b-1/17项P4b-2/58项本批旧证据核对保持；HEAD不变、staged为空。完整差异、参数/事件/权重及覆盖限制见 [P4b-3报告](../../refactor_checks/p4b3_generic_20260916/REPORT.md) 和计划§29。P4b整体完成；P5未执行、B13未执行、B14未执行、analysis未重构；未运行真实九数据集长期训练、真实预处理，未重建cache。本批结束后停止。

## 21. P5：九数据集入口与batch收敛（2026-09-16）

按本批授权完成P5。新增scripts/run_experiments.py，以九个现有canonical runner为唯一单dataset CLI；显式dataset列表、output root、普通override和JSON argv透传，按顺序每dataset独立subprocess。batch不读取数据/config、不构造模型、不训练、不analysis；P3b defaults继续由runner负责。Human Embryo仅补现有普通scale的可选CLI，未传时原parsed/default/model字典完整不变。训练数学、shared fit、MouseBrain epoch0、generic输入/weights、P4a adapter和原artifact格式均保持。

先审计15个suite/launcher并冻结九默认/命令与fit交接，再实现。v7B/v7C无消费者且训练能力被显式scale覆盖，两个壳退休；v7A/SPATCH仍有P0/C1/B8/P1等真实checker消费者而暂留，其它历史混合壳逐一记录KEEP/OUT OF P5，未为删除它们修改analysis。生产4文件+150/-202，净删52。CRC/MISAR默认epochs0准备模式、MouseBrain config和SPATCH已有cache显式要求保持，不在batch补隐藏默认。

75项命令/完整配置对照、九old/new canonical handoff、九batch到真实main/fit sentinel handoff、额外九scale handoff通过；12项batch行为及12项CLI边界通过。P4b-1六事件/四CUDA、P4b-2六事件/三CUDA、P4b-3十二CPU/三CUDA、CPU406、P3b/P4a/B10/C1全通过；117项scope通过。FP32/BF16各13/13且差0，same-layout checkpoint差0，cross-chunk完整原FAIL保持，无record/expected/容差/数值投影更改。

起点与P4b-3的2918项状态无差异，本批3175项旧文件/34根stat/508项P3b/13旧fit/17旧MouseBrain/58旧generic/14本批旧证据保持。HEAD未变，staged为空。详见[P5报告](../../refactor_checks/p5_entry_20260916/REPORT.md)、[launcher审计](../../refactor_checks/p5_entry_20260916/LAUNCHER_AUDIT.md)、[保护核对](../../refactor_checks/p5_entry_20260916/protection_check.json)。P5整体完成；F-analysis未执行，P6未执行，B13未执行，B14未执行，P7/P8未执行，analysis未重构；未运行真实九数据集长期训练或真实预处理，未重建cache。完成本批后立即停止。

## 22. MouseBrain canonical入口去版本名（2026-09-16）

用户单独要求将run_mousebrain_v2.py改为run_mousebrain.py，已完成。同步batch、五个保留suite及验证源码import/路径引用，无兼容转发壳。仅改文件名与说明文字，数据/config/Stage/epoch0/fit/save代码不变；14项命名/CLI/配置/交接和6组MouseBrain旧事件/RNG/参数/保存对照通过。旧报告/fixture/expected保持，没有真实大训练或重建cache，本次未重复GPU gate。详见[重命名报告](../../refactor_checks/p5_mousebrain_name_20260916/REPORT.md)。不启动P6、P7/P8或其它批次。

## 23. F-analysis(B9)实施状态（2026-09-16）

用户单独授权并按范围完成B9。真实缓存消费者先审计并以tiny输入冻结旧错误命中，随后增加analysis/cache.py普通identity/manifest/reuse函数与必要显式output参数。新cache绑定实际embedding SHA256、shape/dtype、section/spot顺序、truth、有效run/resolved字段、各自protocol与implementation；同源命中，不匹配/无manifest明确拒绝并要求新独立输出，不覆盖历史。SPATCH requested K=5/8/10/12/14/16/20、compare不同K集合以及九数据集原分析协议均保持。没有P6 loader/metric/plot函数迁移或算法变化。

V12与实际entry验证通过；old/new合成指标、标签、scaler逐项一致，补图PNG字节一致。C1 73/73，P5命令与九数据集+generic shared fit引用检查通过；训练/model/config/adapter/launcher源码不变。冻结FP32/BF16各13/13且差0，同layout checkpoint差0，cross-chunk原FAIL/KNOWN_NUMERICAL_SENSITIVITY完整保持，只check不record。3434项起点状态及34个保护根核对见本批protection report，不声称递归hash真实大数组。

详情见[B9报告](../../refactor_checks/fanalysis_b9_20260916/REPORT.md)、[真实消费者审计](../../refactor_checks/fanalysis_b9_20260916/B9_CACHE_AUDIT.md)、[回归](../../refactor_checks/fanalysis_b9_20260916/REGRESSION_REPORT.md)。历史metrics不重算或补provenance；历史复制batch CSV及补图输入label的历史来源限制显式保留。P6/P7/P8、B13/B14未执行；SPATCH preprocessing cache未修改，未运行真实九数据集完整analysis。完成后停止，不把本批完成当成下一批授权。


## 24. P6a实施状态（2026-09-16）

本批按用户单独授权仅迁analysis loader/数据读取依赖。新增analysis/loaders.py、analysis/inputs.py及data_io/adapted.py，普通函数显式接收run/data/section输入；九dataset均有明确路径。现有MethodData由原消费者保留，未建registry/Loader类或统一evaluate CLI。Thymus/Simulation六个纯data helper与ADT轴迁入data_io，runner按原名import；MISAR只迁SECTION_INFO字典。所有剩余runner函数AST不变，训练/model/fit/config和P4a有效行为保持。D8b/D8c跨方法/SpaMosaic风格UMAP继续KEEP，数据helper不再import training runner。

修改前冻结30组真实tiny loader输出与265项输入/输出证据；v6/v7A样式及旧通用格式的embedding值/dtype、section/spot/barcode、metadata/truth和来源identity逐项一致。22项requested/旧analysis壳路径handoff、9个Thymus/Simulation section的纯helper对照、D8b/D8c import与数据handoff通过。B9原43项和补充13项回归、17项实际旧loader身份命中/输入变化miss通过，implementation identity不因纯位置移动而改变。P4a12例、P3b27例、C1 73/73、P5命令/九dataset+generic shared-fit引用、B10 help无I/O通过；401个分析函数AST及223个算法调用AST保持。

FP32/BF16固定check各13/13且差0，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture覆盖、容差/layout/AMP或生产CUDA设置变化。4072项起点状态、93个源快照、265项旧loader证据和34个保护根核对通过；HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，所有成果仍在未提交工作树。

详见[P6a报告](../../refactor_checks/p6a_loaders_20260916/REPORT.md)、[依赖审计](../../refactor_checks/p6a_loaders_20260916/ANALYSIS_DEPENDENCY_AUDIT.md)、[回归](../../refactor_checks/p6a_loaders_20260916/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/p6a_loaders_20260916/protection_check.json)。旧完整analysis exec/argv壳中metric/export/plot职责明确暂留P6b/P6c；新loader已独立。没有全量迁移跨方法混合workflow、删除旧main或统一输出布局。历史format只做tiny替身验证，不读取真实大数组，不补造不存在的barcode/truth。

P6a完成，P6整体未完成。P6b未执行、P6c未执行、P7/P8未执行、B13未执行、B14未执行；clustering/metrics算法未改、plotting未重构，历史metrics未重算、SPATCH preprocessing cache未改，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成本批后停止；下一批必须另行授权。


## 25. P6b实施状态（2026-09-16）

本批仅执行P6b。新增analysis/clustering.py、metrics.py、protocols.py、sampling.py、batch_metrics.py；九dataset标准化/聚类/监督与internal指标/采样和batch数学使用普通函数，保留原dataset与requested/compare差异。23个生产文件+1626/-1355；training/model/data_io、P6a loader及canonical训练runner未修改。

完整读取supplementary参考后正式定义joint / joint_per_section / independent。joint_per_section只切已有joint assignments，零重新scaler/cluster；实际指标只有ARI/NMI，原协议没有跨section平均，aggregation=none。MouseBrain仅RegionLoupe/annotations、MISAR五truth、SPATCH仅cell_type_common、Simulation spatial_domain；CRC/HLN/Spleen/Thymus无truth不制造指标。Embryo不擅自增加原参考没有的监督范围。保留参考与requested不同的truth过滤规则。两个原requested入口通过--joint-per-section显式启用，产生逐section结果与completion状态；B9绑定scope/K/section顺序/label/metric/filter/aggregation及implementation，纯函数搬动保留原默认分析identity。

122项old/new数值exact，29项joint-per-section专项，18行/14个supplementary导出文件（仅生成时间不同）、真实requested新scope拟合次数/旧指标/缓存一致性通过；P6a30loader与22条路径、B9 43+13、P4a12、P3b27、C1 73、P5/P4b必要引用回归通过。FP32/BF16 fixed各13/13且差0、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture/tolerance/layout/AMP修改。4564项旧状态、96源快照、旧tiny基准及34保护根核对无越界差异；HEAD/staged不变。

详见[P6b报告](../../refactor_checks/p6b_protocols_20260916/REPORT.md)、[协议审计及joint-per-section专章](../../refactor_checks/p6b_protocols_20260916/ANALYSIS_PROTOCOL_AUDIT.md)、[回归](../../refactor_checks/p6b_protocols_20260916/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/p6b_protocols_20260916/protection_check.json)。P6b实施完成，P6整体仍未完成；P6c、P7/P8、B13/B14未执行，plotting/UMAP未重构。历史metrics未重算，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成后停止，下一批必须另行授权。

## 26. P6c实施状态（2026-09-16）

本批按用户单独授权完成P6c：新增明确scripts/evaluate.py和普通analysis.evaluation/requested/comparisons/spatch/embryo/plotting/umap。九dataset显式选择dataset/run-dir/output-dir/protocol/scope，直接消费P6a loader、P6b clustering/metrics/protocol。31个生产文件+5245/-4344；training/model/data_io、九canonical runner、shared fit、P5入口完全未改。

joint / independent / joint_per_section明确分开，后者只切已有joint labels，真实ARI/NMI、valid spot与section字段保持，不重新聚类、不跨section平均；无truth dataset不造值，Embryo不擅自扩参考监督范围。Embryo七section/celltype/developmental/25叶weighted Ward与层次图保留，逐section补图可显式调用。SPATCH requested与compare K独立，B9新输出来源绑定保持。

先审计34个analysis main；退休无生产caller且已完整替代的rerender_result_v6_umap_figures.py、complete_human_embryo_per_section_plots.py。真实suite/aggregate/cross-method/legacy格式与后处理caller仍有的旧壳逐项暂留。D8b跨方法和D8c原始模态准备保持KEEP，二者共用analysis.umap/plotting并写新output root；新evaluate提供已有UMAP坐标重绘，不另创统一九dataset UMAP预处理协议。D8c SPATCH可视化输入缓存隔离到新output，来源变化在预处理前拒绝，不触碰C1缓存。

21个renderer旧/新输入、panel/colors/order/size/dpi逐项相同；九workflow文件集合/CSV字段/JSON字段类型及数值相同；UMAP16组prepared input/参数/显示/重绘对照通过。九真实loader→workflow与九standalone joint_per_section+真实requested共19组，最终绘图入口和cache边界通过。P6b122 exact、joint29、requested scope7、P6a30、B9 43+13、P4a12、P3b27、C1 73及P5/P4b必要引用通过。FP32/BF16各13/13、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持，只check。

5341项旧状态、全部源快照/旧tiny基准及34保护根核对，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged为空。所有成果未提交。详见[P6c报告](../../refactor_checks/p6c_plotting_20260916/REPORT.md)、[main审计](../../refactor_checks/p6c_plotting_20260916/ANALYSIS_MAIN_AUDIT.md)、[回归](../../refactor_checks/p6c_plotting_20260916/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p6c_plotting_20260916/protection_check.json)。旧历史metadata/裸label证据不足时保持原能力并明确限制，不补造barcode或来源；未读取真实大数组。

P6a/P6b/P6c组成的P6整体实施完成。P7未执行、P8未执行、B13未执行、B14未执行；未运行真实九dataset完整analysis、长期训练、真实预处理/UMAP大优化或cache重建，未修改历史result/cache。完成后立即停止，没有下一批授权。

## 27. P7实施状态（2026-09-17）

本批仅执行P7，基于已验收P6真实工作树重新扫描当前AST/import/CLI/string/dynamic/test/protected引用。修改前建立P7_DEAD_CODE_AUDIT；原9个DEAD候选全部仍有定义、无当前消费者，现删除。另删22个无消费者import绑定。7个生产文件+2/-248，净减246行；剩余全部函数/class及顶层AST仅有列明删除，整个Stage类、state/schema/constructor/forward/参数顺序保持，training/shared fit/config/data_io和所有runner字节未改。

processed_data_dict与load_cosie_style_data有真实调用者继续KEEP；旧order/epoch兼容未证明可安全移除而KEEP TEMPORARILY。v7A/SPATCH suite仍有P0/C1等冻结checker消费者；其它混合壳有真实caller或独有调度/历史输出职责，未证完整替代者暂留。本批未新增退休launcher/analysis wrapper。v7B/v7C suite、旧MouseBrain文件及P6c两个已退休main标ALREADY REMOVED，不重复计删。D8b跨方法UMAP、D8c raw modality准备/整合、single modality/noOT、H&E/UNI、FAISS/Flat/blockwise、weights export及三种analysis scope继续KEEP。

P5 batch12（含九真实canonical tiny handoff）、P4a12、P3b27、C1 73、shared fit事件6、MouseBrain事件/RNG6、generic实际tiny Stage/Adam12、P6a30、P6b数值122、joint_per_section29、requested scope7、evaluate19、B9 43及UMAP16均通过。55个模块import、14个子进程help和2359个Python源码compile通过。FP32/BF16固定各13/13 exact，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保留，只check不record。P4a首次checker命令参数误用导致parse退出2，修正调用后通过，原错误日志保留。

6635项旧状态、104个生产源快照、旧tiny expected及34保护根核对；无非白名单差异，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，成果仍未提交。保护检查不声称递归hash真实大数组。详见[P7报告](../../refactor_checks/p7_cleanup_20260917/REPORT.md)、[DEAD审计](../../refactor_checks/p7_cleanup_20260917/P7_DEAD_CODE_AUDIT.md)、[回归](../../refactor_checks/p7_cleanup_20260917/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p7_cleanup_20260917/protection_check.json)。

P7完成。P8未执行，B13未执行、B14未执行，其它独立correctness问题未执行；未改变model数学或analysis协议，未运行真实长期训练/完整analysis/真实预处理，未修改历史result/cache，未重建cache。完成后停止，下一批需要另行授权。

## 28. P8实施状态（2026-09-17）

本批仅执行 P8，先基于已验收 P7 工作树建立 FINAL_STRUCTURE_AUDIT。普通 MOVE：model/data_preprocessing.py→data_io/preprocessing.py（字节相同）、model/multimodal_preprocessing.py→data_io/datasets.py、model/image_preprocessing.py→data_io/image_features.py、scripts/hesta_rna_utils.py→data_io/hesta.py（字节相同）；model/utils.py 按原函数边界拆为 data_io/common.py 与 model/spatial_graph.py。更新真实 import，model 包不再重新导出数据函数，无旧路径 shim；load_cosie_style_data 的真实调用能力仍保留。SPATCH schema=1 cache loader/metadata、dataset artifact 保存和有消费者历史壳按审计留原位，未创建新的配置/trainer/artifact/loader框架。

迁移函数/class/签名及有效可执行 AST 完全相同；唯一函数 body 字符串变化为 SPATCH 未由训练main调用的 cache writer 的源文件诊断路径，既有 manifest 和 C1 identity 不变。Stage 类/constructor/state/schema/参数顺序、preprocessing 数学、training/config.py、training/fit.py、全部 analysis 权威源码不变。44 组图/data/UNI/HESTА/state/RNG 小对照精确一致。九dataset P5 handoff、P4a/P3b/C1、shared fit、MouseBrain epoch0、generic 12小例、P6a/P6b/joint-per-section/evaluate/B9/plot/D8b/D8c通过。FP32/BF16 fixed 各13/13 exact，same-layout checkpoint差0；原cross-chunk FAIL / KNOWN_NUMERICAL_SENSITIVITY完整保留，只check不record。

新增根 README.md 和 scripts/README.md，列出九canonical训练入口、batch、evaluate及69个脚本分类。MouseBrain名称保持run_mousebrain.py；batch用misar/crc，evaluate用misar_seq/crc_stereocite，明确区分。13条README命令经真实parser/help/dry-run/sentinel检查；v7A研究scale=.5显式传入，未把底层/部分裸入口1.0默认改写成.5。joint_per_section只切joint labels，参考ARI/NMI、不聚类、不跨section平均；SPATCH双K协议与Embryo特殊分析保持。model/training/data_io无scripts依赖，analysis无training runner/历史suite依赖；迁移模块无循环依赖和活跃旧路径引用。

起点7443项旧状态、104份生产源码快照、旧fixture/expected/报告/FAIL及34保护根按白名单核对。HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，累计成果未提交。保护核对不声称递归hash真实大数组。旧checker的模块路径仅在新P8副本适配；原报告和expected不改。旧UMAP快照的嵌套import适配过程日志完整保留，不是算法/数值失败。

详见[P8报告](../../refactor_checks/p8_final_20260917/REPORT.md)、[结构审计](../../refactor_checks/p8_final_20260917/FINAL_STRUCTURE_AUDIT.md)、[回归](../../refactor_checks/p8_final_20260917/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p8_final_20260917/protection_check.json)。后续如需check，从本批适配checker复制到另一个新目录；不可重跑record或覆盖本批/既有产物。

P8实施完成，当前主线重构实施整体完成（等待用户审阅本批，不冒称用户已经验收P8）。B13未执行、B14未执行、B11/B12未执行、B4/R1/R3未执行；模型/预处理/analysis科研方法未改。未运行真实长期训练或完整九dataset analysis、未重建cache、未修改历史result/cache。完成后停止，不自动开启任何独立correctness批次。


## 29. S1实施状态（2026-09-17）

S1 按当前工作树审计 69 个 scripts Python 文件，完成用途/CLI/import/subprocess/dynamic/frozen/README/helper 分类。九 canonical runner 原路径 help 全部通过；原样嵌套至 scripts/train 的真实探针全部 import 失败，因此正式 CLI 保持根路径。仅普通移动 scripts/validate_bidirectional_ot_attention.py → tools/validation/validate_bidirectional_ot_attention.py，使用仓库根目录 python -m，去掉原有 6 行 import/bootstrap；全部函数/签名/main AST 相同，不新增 sys.path hack 或兼容 shim。scripts Python 数 69→68，tools/validation 1 个 Python 工具；未创建空 tests/train/comparison/workflows 目录，无 RETIRE。

其余 68 文件保留：正式路径稳定，P1/D3/D4/generic 行为 checker 消费的 validator 留原位，P0 freeze/supplement 和 v7A/SPATCH suite 留原位。supplementary joint-per-section 有 P6/P8 当前 checker import 和历史 labels/export 变体，未退休；HLN A1 外部人工 truth 的五方法流程替代证据不足而 KEEP TEMPORARILY。D8b/D8c 独立工作流保持，旧壳/implementation leak 如实记录，不宣称所有历史脚本都已变薄。

105 生产 Python compile/import、24 正式/工具 help（I/O guard）、13 根 README 命令通过。移动工具 4 组参数/Stage/input/prior/forward/输出/RNG 精确一致，仅 3 种耗时 metadata 字段变化；旧/新原始证据保留。九 dataset handoff/batch12、P4a12、P3b27、C1 73、shared fit6、MouseBrain6、generic synthetic/zero/noOT、P6a30、P6b122、joint29、evaluate19、requested scope7、B9 43、plot15、D8b/D8c UMAP16 通过。4 个保留 validator main 通过；run_stage_model 旧 smoke 文案断言原版/当前同 FAIL，未修、未冒称 PASS。

FP32/BF16 fixed 各13/13 exact，same-layout 差0，cross-chunk 完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保持。只 check，不 record、不改 fixture/expected/tolerance。model/data_io/training/analysis 及 68 个保留 scripts 字节不变；旧状态、保护根和 HEAD/staged 按本批 protection_check 核对，成果未提交。

S1 实施完成，等待用户审阅。B13/B14/B11/B12/B4/R1/R3 未执行；未运行真实长期训练、完整 analysis、真实预处理或 cache rebuild，未修改历史 result/cache。完成后停止，无下一批实施授权。

详见 [S1报告](../../refactor_checks/s1_scripts_20260917/REPORT.md)、[完整脚本审计](../../refactor_checks/s1_scripts_20260917/SCRIPTS_AUDIT.md)、[回归](../../refactor_checks/s1_scripts_20260917/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/s1_scripts_20260917/protection_check.json)。后续 checker 复制到新授权目录并只读旧 expected；不要原地重跑覆盖本批报告。


## 30. S2a实施状态（2026-09-17）

本轮只执行 S2a，按当前引用闭包成组退休 11 个 v4/v5/v6 历史训练 launcher/suite/daemon，生产 +0/-3186 行。历史训练壳 13→2；scripts Python 68→57。v4c→v4b、v4c_large→v4_large 的历史内部引用不再作为保留理由。没有当前正式 caller 或需迁移的共享 helper，不新增 shim；九 canonical runner、run_experiments/evaluate/preprocessing 路径与字节保持。

保留两项 FROZEN_PATH_EXCEPTION：run_result_v7a_suite 的 training_commands 仍为当前 P0 replay 配置来源；run_spatch_v7abc_suite 的 training_command/validate_training 仍由 P0/C1 当前 checker 调用。它们的生产调度能力已可用 batch 表达；本轮不迁移这两个冻结校验依赖，后续 S2c 是否处理须另行授权。其它 v4/v5/v6 普通 helper 没有当前 consumer，旧后台 PID/GPU等待/from-task/训练后自动分析随壳退休，未新增 scheduler。

16 份历史 check_entries.py 的 v6 import 属于旧批次配置投影证据，原样保留，不作为当前回归入口。当前使用 S1 已迁移的 run_experiments→九 runner handoff/config checker，并复制到本轮独立目录 check；不修改历史 checker/fixture/expected。所有 11 个 Git 旧路径可恢复；其中 4 个前期未提交版本与 HEAD 不同，其精确字节已存在于已验收 S1/before_sources，本轮再次留存，不能把 HEAD 当作全部成果提交。

94 个保留生产模块 compile/import、13 个 guarded/direct CLI help、11 个替代 batch dry-run 逐 token 对照通过；九 dataset 实际 tiny handoff/batch12、P3b27、P4a12、C1 73、shared fit6、MouseBrain epoch0/RNG6、generic synthetic/zero/noOT3 通过。analysis/evaluate 和 validators 必要 import 通过；无当前源码残留路径。FP32/BF16 固定各13/13 exact、same-layout差0，cross-chunk完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保留。只 check、不 record、不改数学/协议/默认值。新静态 checker 曾把 batch dry-run 的 JSON list 误当 dict，修正本轮测试读取后通过；生产无变化，诊断留存。

12206 项继承状态、34 个保护根及 HEAD/staged 按本轮白名单核对，保留生产文件字节相同。HEAD 仍 922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged 为空；累计成果未提交。历史结果/大数组没有递归哈希；证据范围以 protection_check 为准。

S2a 实施完成，等待用户审阅；完成后停止。S2b/S2c/S2d 未执行，analysis/supplementary/validation 物理整理未执行；B13/B14/B11/B12/B4/R1/R3 均未执行。未运行真实长期训练、完整 analysis、真实预处理或 cache 重建，未修改历史 result/cache。

详见 [S2a报告](../../refactor_checks/s2a_train_launchers_20260917/REPORT.md)、[launcher审计](../../refactor_checks/s2a_train_launchers_20260917/S2A_TRAIN_LAUNCHER_AUDIT.md)、[回归](../../refactor_checks/s2a_train_launchers_20260917/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/s2a_train_launchers_20260917/protection_check.json)。后续只在新授权目录使用当前 checker，不覆盖本轮或历史证据。

## S2b 实施状态（2026-09-17）

S2a 已验收后本批仅执行旧 analysis 入口退休：23 文件实际删除，scripts 57→34，analysis 候选35→12。当前单 run 使用 evaluate/P6；Simulation 独有诊断函数归 analysis 并保留薄CLI；只读 persisted joint-label 校验归 analysis/joint_results。11 个跨方法/D8b/D8c/人工A1工作流明确留S2d。当前 checker 解除旧main import，历史证据不改。CRC/SPATCH informational SHA 指权威模块，B9 identity不改。两冻结suite的训练API保持，仅去旧analysis dispatch。详细证据见 [S2b REPORT](../../refactor_checks/s2b_analysis_entries_20260917/REPORT.md)。S2c/S2d及B13/B14/B11/B12/B4/R1/R3未执行。

## S2c/S2d 后续取舍更新（2026-09-17，尚未实施）

用户确认不再需要 raw-vs-standardized 聚类预处理比选实验入口，正式聚类分析采用 standardized_embedding。当前八个 compare_*_kmeans_preprocessing.py 确实同时执行这两条分支；S2d 将其作为成组退休目标，先迁真实需要的跨方法/对齐/科研函数，不因内部helper或彼此import永久保留旧壳。保留已批准的跨方法KMeans/UMAP、D8c和独有诊断能力；不把所有metric/UMAP输入空间强制改成standardized。

最终当前可执行脚本按职责命名，不带v7a/v6/v7abc等版本号。S2c先收敛验证工具、解除当前checker对旧comparison/suite的依赖，严格保留冻结数值合同；S2d再退休比选实验壳并为保留workflow采用非版本化名称。历史REPORT/fixture/expected和源码证据的版本名保持。具体边界见 [计划补充](../../refactor_checks/s2cd_plan_20260917/PLAN_UPDATE.md)。本次仅记录取舍，S2c/S2d均未实施。

## data_io 结构反馈与后续整合意向（2026-09-17，仅记录，未授权实施）

用户在逐步阅读重构后的项目结构时指出：当前结构虽然比以前精简，但仍显得杂乱；希望进一步改善代码组织和流程的整体可读性。此次只记录意见，不执行代码修改，不自动开启新的重构或 correctness 批次。

- **数据集适配集中组织。** 对当前分散在 `data_io/paired.py`、`misar.py`、`adapted.py`、`hesta.py` 等文件中的数据集适配，用户希望整合到一个较大的处理文件，便于集中理解不同数据集的读取和适配。允许各数据集保留独立函数、专用分支和各自协议，不要求九个 dataset 走完全相同的函数。具体文件名及通用预处理、HESTА 专用预处理的职责边界尚未确定。
- **分析输入准备形成整合流程。** 当前 `comparison_inputs.py`、`hln_annotations.py`、`visualization_inputs.py` 按不同分析用途分散，用户认为仍有“每次新实验加入新补丁”的组织感。后续设计应考虑将读取、身份对齐、校验及面向消费端的准备组织成清楚的整体流程；具体合并范围与接口尚未确定，不据此删除跨方法比较、人工标注或可视化能力。
- **后续设计关注点。** 集中适配代码的同时，需要让共同步骤和数据集特有步骤容易辨认，避免仅拼接文件而仍保留难以追踪的调用关系。这是待讨论的结构改进方向，不是已定稿的实施方案，也不改变已有科研协议、身份/顺序、dtype、cache 和 artifact 合同。

本次只追加本节决策记录；不移动、合并、重命名生产文件，不修改 CLI、模型、训练、预处理数学或分析协议，不运行验证/训练/分析，不改写历史报告或结果。后续继续按用户节奏讲解结构，实施整合须有后续明确指令。

## 历史文档集中归档意向（2026-09-17，仅讨论，未授权实施）

用户指出当前 `docs/` 主要是较早的实验记录与分析，暂时不用于当前工作；提出考虑将它们与项目根目录的若干记录集中放到总的 `docs/` 下，同时继续保留两批记录的分组。此处记录的是待讨论意向，不是移动文件的执行指令。

- 两批历史实验记录应保持来源边界：原 `docs/` 中的早期记录一组，根目录中的后续实验、比较及方案记录一组；不合并成一篇报告，不改写历史结论，不删除暂时不用的记录。
- 助手建议另行区分当前使用说明、工作交接与重构过程文档：根 `README.md` 和当前 `handoff.md` 可保留为入口；`REFACTOR_AUDIT.md`、`REFACTOR_PLAN.md`、`USER_DECISIONS.md` 的后续归属单独讨论，不直接混入实验记录。具体目录名和移动清单尚未确定。
- 若后续授权整理，先列出精确文件映射及引用关系，在总导航中注明历史批次、原路径和适用阶段；保留原报告内容，避免将旧实验命令或旧模型说明当作当前使用合同。现有 `refactor_checks/` 的冻结证据不纳入此次拟议搬迁。

本次只追加意向记录，没有移动、重命名或删除文档，没有修改历史报告、生产代码、cache 或 result。

## 文档归档实施记录（2026-09-17）

用户随后明确授权按上述安排移动文档并做好 README 记录。本次已将原 `docs/` 的 22 份早期记录移至 `docs/experiments_early/`，根目录 15 份后续实验与方案记录移至 `docs/experiments_later/`，三份重构文档移至 `docs/refactoring/`。根 README 与 handoff 保留，新增 [docs 总导航](../README.md)，列出原位置、新位置与历史相对链接的导览；同步更新重构计划、用户决策和 handoff 的必要导航链接。历史实验报告及重构审计正文保持原样。

本次未执行 data_io 整合，没有修改生产代码、已有验证证据、cache 或 result；未运行训练、预处理或分析，没有 stage 或 commit。归档核对见 [REPORT](../../refactor_checks/docs_archive_20260917/REPORT.md)。此前“仅讨论，未授权实施”是当时状态，本节仅确认文档归档已获授权并实施，不扩展为其它结构修改。
