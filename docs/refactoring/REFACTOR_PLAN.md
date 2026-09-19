# spa_mo_model 重构计划：以回退后 v7A 为唯一模型基线

## S2d 当前实施状态（2026-09-17）

S2d完成scripts最终入口收敛：**25→18个Python CLI**。8个raw-vs-standardized比选入口退休；D8b、D8c、HLN A1人工标注三个旧main分别由非版本化薄CLI替代，另有一个统一的标准化跨方法comparison入口。comparison/两个、workflows/两个；Simulation独有诊断继续留根部。所有保留CLI文件名无实验版本号，无compat shim、无新sys.path hack。

D8c的真实职责是原模态输入panel c与已有整合embedding panel e的SpaMosaic风格展示；不运行外部SpaMosaic模型或integration训练。跨方法reader、必要标准化编排、人工标注及可视化输入准备已归analysis/data_io；P6科研参数、UMAP/batch输入空间、原label/figure保留K不变。九canonical runner、batch、evaluate、preprocessing、12个工具路径和原实现保持。

FP32/BF16各13/13，same-layout差0；cross-chunk保持FAIL/KNOWN_NUMERICAL_SENSITIVITY。Stage smoke既存文案断言失败不修。B13/B14/B11/B12/B4/R1/R3均未执行。未运行真实长期训练、完整analysis或cache重建，历史result/cache/fixture/expected/report未改。

最新入口见 [scripts导航](../../scripts/README.md)，证据见 [S2d报告](../../refactor_checks/s2d_scripts_final_20260917/REPORT.md) 与 [最终审计](../../refactor_checks/s2d_scripts_final_20260917/S2D_FINAL_SCRIPTS_AUDIT.md)。S2a–S2d工程整理实施完成；完成后停止，无自动进入correctness轨道的授权。以下较早章节为当时阶段记录，当前状态以本节和README为准。


## S2c 当前实施状态（2026-09-17）

S2c已完成实施：scripts **34→25**；5开发工具移动至tools/validation并用职责命名，两个旧record工具与两个旧suite退休。当前12个开发Python无实验版本文件名，无新增compat shim/sys.path hack。冻结证据路径不改；没有current executable需要FROZEN_PATH_EXCEPTION。

九dataset/batch/evaluate/preprocess保持；current checker已解除对8个compare preprocessing CLI的普通helper依赖。八compare原字节未改，S2d建议先迁必要跨方法能力再退休旧CLI；D8b/D8c/A1/Simulation独有能力保持。唯一保留版本化scripts为D8c generate_result_v6_spamosaic_umap，留S2d处理。

FP32/BF16 13/13、same-layout差0；cross-chunk仍KNOWN_NUMERICAL_SENSITIVITY；Stage smoke旧文案失败不修。B13/B14/B11/B12/B4/R1/R3未执行；未运行真实长期训练/完整analysis/重建cache。S2d尚未执行，完成后停止。[S2c报告](../../refactor_checks/s2c_validation_20260917/REPORT.md)。以下较早阶段段落保留当时记录，当前入口以README和本节为准。

修订日期：2026-09-17。**Batch 0：BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY。P7已获用户验收，本轮仅完成P8（§37）；当前主线重构实施整体完成。** 只做普通目录/import归位及README导航，科研数学、D2 schema与原cross-chunk FAIL保持。B13/B14/B11/B12/B4/R1/R3仍pending，完成后停止，无后续批次授权。

本文替代以 v15A 为目标的旧方案，不恢复 v15A 或 feature graph。§10–§12 是原阶段记录，保留当时结论；当前验收状态以 §13 的用户决议为准。原审计、两份验证报告、JSON 和冻结 expected 均不改写。**v7A 是模型/算法基线，完整实验范围是 result_v6 实际使用的全部 9 个数据集；v7A suite 只是调试子集，不能作为删除数据集支持的依据。**

## 1. 当前基线与针对性复核结论

唯一模型主基线是初次接手核对到的工作树：HEAD **922d1738922e8b94890a54f5e42bbf6551f5ccc0**。初次接手时 tracked 工作树及 staged diff 均为空；未 tracked 项为 REFACTOR_AUDIT.md、REFACTOR_PLAN.md、refactor_checks/、scripts/freeze_v7a_baseline.py、scripts/supplement_v7a_baseline.py，与交接一致。未发现适用的 AGENTS.md。C1/P1随后形成的未提交修改保留，P1起点差异见§15；HEAD是冻结源码参考，不是回退指令。没有执行 reset、checkout、clean、stash 或提交。

基线行为由当前源码和实际命令确认：scripts/run_result_v7a_suite.py:104 的五数据集命令，以及 scripts/run_spatch_v7abc_suite.py:107 的 **v7A 分支**，最终调用一个 model/stage_model.py:255 的 StageMultiModalModel。没有 V7AModel，也不需要创建。

**裸 CLI/库默认尚不等于完整 v7A 研究命令。** configure.py的post scale默认1.0；model/configs/v7a_post_ot_graphsage_scale_0.5.json只覆盖scale。P1前MouseBrain/CRC裸参数是dense/单向，MISAR裸参数是sparse/单向，generic显式dense；P1完成调用者迁移，随后P2实际删除旧core和模式选择，见§15/§16。其他训练参数未强行统一，v7A命令有效值保持。

初次基线复核读取的小型记录包括result_v7A/suite_status.json（12253字节，支持调试子集的命令、scale、source及刷新记录）与6998字节的SPATCH manifest；本次补充读取result_v6的三组状态记录、九份run_summary及分析config，证据见§1.3。没有读取真实feature数组、checkpoint或大型metadata，没有运行训练/评估/正式测试。旧运行记录没有完整状态指纹，不能承诺历史数值逐元素重放；Batch 0冻结的是**当前代码与明确状态**。

### 1.1 与回退前审计/旧计划的差异

| 核对项 | 当前工作树证据与合同 | 旧报告/计划的处置 |
|---|---|---|
| 主版本 | 当前源码+v7A命令；不恢复其他提交 | 撤销v15A恢复批次，不建立后续版本默认 |
| feature graph | model/scripts仅configure.py:115残留use_feature_graph=False；无builder、cache、refresh或pre组合图实现 | 回退前fused.detach、K_F、混合边权、初始化/两遍refresh都不属于当前合同；不重新引入 |
| pre/post图 | Stage:1263–1272与1407–1422均使用section内同一空间图 | pre不再spatial+feature，post仍spatial-only |
| post residual | v7A suite显式.5→constructor:352–354→层:249；pre固定1 | .5在研究命令中生效，不能由版本名推断所有入口默认 |
| 图/OT调度 | should_update_ot:119–157；100起，每20，step后一次eval/no_grad，仅换prior | epoch1构feature、两次eval、图generation计划全部删除 |
| 初次forward | MouseBrain:576–584仍有epoch0、train模式下no_grad dry forward；其他五数据集训练分支无此步 | 保留其RNG消耗；它现在不构feature图 |
| 评估任务集合 | requested analyzer.SPECS:63–83的五数据集与当前v7A调试suite匹配 | 旧B15已消失；该局部一致性不定义完整实验范围，九数据集训练/分析支持均须保留，见§1.3 |
| 旧版本实现 | tracked树无v15A suite；当前scripts无后续feature实现 | 不为v8–v15建立preset框架、wrapper或模型类；已有结果目录不构成保留训练接口的理由 |
| 插补与缓存 | 三目录整体不改；SPATCH现有缓存必须直接读取 | 撤销旧计划中的插补测试修复、目录整理，以及会重建缓存的安排 |
| 数值bug | 当前缺陷单列；纯重构保留其现有语义 | 上轮“B4必须实施”的安排不再适用本轮授权；mask/loss/Sinkhorn修改均须另行批准 |

### 1.2 入口和数据集实际设置

以下设置来自初次核对的suite argv、runner参数注入、constructor消费者，不是照抄旧报告的默认值；P4a已将四个文本包装改为显式函数调用，实际设置保持，见§25。

| 数据集 | 准备/训练调用链 | 空间K与OT检索 | 保留的运行差异 |
|---|---|---|---|
| MouseBrain | run_mousebrain_v2.run_mousebrain → preprocess_mousebrain/build_mousebrain_sections → Stage → 本文件loop:677 | K5；IVF nlist256/nprobe32、auto | 训练前epoch0 dry forward消耗dropout RNG；无AMP/chunk/checkpoint调用 |
| MISAR | run_misar_seq → Stage:542 → CRC.train_small_crc_model:774 | K5；IVF128/32、CPU | edge50000、decoder50000、attention50000；四checkpoint、loss-only/cache；AMP none |
| Spleen | run_mouse_spleen → 显式HLN parser / data_io.paired → CRC原pipeline/loop | K10；IVF256/32、auto | edge100000、decoder2048、attention1024；四checkpoint、bf16 |
| Simulation | run_simulation.read_pair → MISAR原pipeline → CRC loop | K10；Flat检索 | 同上chunk/bf16；RNA HVG1000、Protein100等设置独立保留 |
| Thymus | run_mouse_thymus.read_pair → MISAR原pipeline → CRC loop | K10；IVF256/32、auto | 同上chunk/bf16；ADT/基因对齐顺序不改 |
| SPATCH | run_spatch加载既有cache → Stage:708 → CRC loop:726 | K10；IVF4096/64、GPU | edge100000、decoder8192、attention2048；四checkpoint、bf16；缓存已有PCA/Harmony结果直接进模型 |

当前这些v7A命令共同使用200 epochs、seed42、lambda_contrast=.1、post=.5、初始每模态candidate100、candidate_k200、attention_topk10、epsilon=.05、tau_a=tau_b=1、sparse固定100次迭代、source=ot、interval20。这里是**命令核对结果，不是以后强制统一其他数据集/实验的默认**。FAISS训练样本数、query batch、HVG/Harmony、模态集合、K、AMP和保存开关应各自记录，不能只冻结上表的部分字段。

上述初次核对时SPATCH suite在manifest缺失时会追加 --build_preprocessed_cache。C1已取消该自动重建分支，SPATCH主入口明确要求复用所选现有cache，见§14；P1保留C1两文件原字节，未调用真实suite或缓存build。

### 1.3 完整实验覆盖：result_v6 实际完成的 9 个数据集（KEEP）

完整名单由下列三个已存在的运行状态记录及各run_summary交叉核实；三组对应训练/分析任务均记录completed、returncode=0：

- [suite_status.json](../../result_v6/suite_status.json)：MouseBrain、Human Embryo、MISAR-seq。
- [missing_v3_matched_suite_status.json](../../result_v6/missing_v3_matched_suite_status.json)：Human_Lymph_Node、Mouse_Spleen、Mouse_Thymus、Simulation、CRC_Stereo-CITE-seq。
- [spatch_suite_status.json](../../result_v6/spatch_suite_status.json)：SPATCH。第一份记录的excluded_datasets=[SPATCH]只是该分组的排除项。

| 实验数据集（全部KEEP） | 当前必须保留并迁移的训练/数据入口 | 原有适配差异示例 |
|---|---|---|
| MouseBrain | scripts/run_mousebrain.py、model/multimodal_preprocessing.py | 三section；RNA/Metabolite/已有UNI作为HE；FP32与epoch0预检 |
| Human Embryo（HESTA RNA-only） | scripts/run_human_embryo_rna_only.py、scripts/hesta_rna_utils.py | 七section、外部Harmony记录、RNA单模态、lambda_contrast=0；BF16 |
| MISAR-seq | scripts/run_misar_seq.py | RNA/ATAC、dataset4→3→2→1；amp_dtype=none |
| SPATCH | scripts/run_spatch.py | 两section、RNA50/Protein15/HE50、既有cache直接复用、BF16 |
| CRC_Stereo-CITE-seq | scripts/run_crc_stereocite.py | CRC_003/CRC_006、共享基因/marker对齐、RNA/Protein；BF16 |
| Human_Lymph_Node | scripts/run_human_lymph_node.py及其CRC helpers | A1/D1、基因ID和非零基因过滤、RNA50/Protein20；BF16 |
| Mouse_Spleen | scripts/run_mouse_spleen.py及其HLN/CRC helpers | 两section、RNA50/Protein20、ADT marker保序；BF16 |
| Mouse_Thymus | scripts/run_mouse_thymus.py及其MISAR helpers | 四section、RNA obs坐标、固定ADT marker轴、RNA50/Protein15；BF16 |
| Simulation | scripts/run_simulation.py及其MISAR helpers | 五section、RNA/ADT、spfac/nsfac与spatial_domain语义；BF16 |

上述记录证明实验覆盖和历史输入/协议，**不把模型回退到v6，不代表九数据集已动态验证当前v7A**。各run_summary、分析config路径和明确分析入口见 [USER_DECISIONS.md §1](USER_DECISIONS.md#1-完整实验覆盖九数据集全部-keep)。本轮仅读取小JSON和必要源码，没有运行历史suite、预处理、训练或分析。

后续统一训练框架必须让九数据集都能运行当前v7A模型；保留各自必要的obs/var对齐、section映射、模态/feature轴及预处理设置。每个数据集都要有明确分析入口，按真实评估协议区分聚类算法、标准化作用域、seed、K、标签、抽样和绘图。尚未出现在v7A suite不是删除adapter、训练入口、analysis loader、metric pipeline或plotting逻辑的理由。

可以退休的是重复trainer、版本启动壳、源码replace+exec、argv注入和globals/monkeypatch；必须先迁出被九数据集使用的普通adapter/shared fit/分析函数，再删旧壳。result_v6、result_v7A及其他历史目录只作为保护产物，既不定义代码目录结构，也不要求永久保留同名suite。

## 2. 冻结当前模型和实验逻辑的行为合同

### 2.1 一次forward与backward

数据准备提供有序 feature_dict、spatial_loc_dict、section_order及必要spot metadata。模型输入转换 Stage:527–530 保留梯度；OT候选输入转换detach，不能把两者盲目合并。

1. 每section构建或读取空间图（Stage:1112–1143），按模态顺序经过MLP encoder（1145–1156）。hidden为Linear/LayerNorm/GELU/Dropout，末Linear后L2 normalize；不增加softmax或modality token。
2. 每section所有观察模态对计算现有COSIE crossview loss（1168–1178）。Fusion按有序concat→MLP，加模态均值residual和LayerNorm（1189–1214）；合法单模态模式使用encoder输出。
3. pre-OT WeightedResidualGraphSAGE（1263–1272）仅消费空间图，scale1。默认self_path_mode=no_self_linear，仍有邻接self-loop和外residual。pre/post是独立参数实例（328–355），不得共享权重。
4. 所有方向attention同时读取同一组pre-OT embeddings（1296–1331）。QK/sqrt(d)+beta*log(prior)，当前beta=.2；confidence、gate、padding处理保留当前实现。各section先收集方向update、按方向数平均，再一次apply_update做dropout/residual/norm（1333–1342）。三section中间节点不串行更新，不改成按每行“有效方向数”加权。
5. 该结果为ot_embeddings。post GraphSAGE只用同section空间图，得到final_embeddings（1407–1422）；层数学顺序为activation→dropout→x+scale*out→norm（model_component.py:237–251），v7A post scale=.5。scale0也不等于完全bypass。
6. decoder重构预处理特征，不是raw counts。Stage:580–669每模态MSE按元素mean、按模态权重求和，再跨section求和；crossview也按模态对及section求和；1456合成两个loss权重。chunk按SSE/总元素归约，不是各chunk mean的均值。
7. full-batch Adam。保留各入口forward/zero_grad/backward/step、GradScaler和loss schedule顺序。MouseBrain zero_grad()未显式set_to_none，CRC显式True；应冻结当前环境下语义，不把它们先假定为统一历史设置。
8. CRC共享loop只有训练forward在autocast中（802–825）；refresh和final在autocast外（918–948、959–984）。仅fp16启用GradScaler，bf16不启用。保留chunks、activation checkpoint use_reentrant=False/preserve_rng_state=True、图cache和loss-only的梯度与内存生命周期，不只检查shape。

### 2.2 空间图与双向sparse UOT

空间图权威函数是 model/utils.py:219 的 compute_spatial_knn_graph_with_weights，调用者Stage._get_spatial_graph:537。

- 每section单独sklearn KNN查k+1，要求n>k；sigma_i=max(末距离,delta)。非self原权exp(-d²/(sigma_i²+delta))，self原权1。
- 加反向边，空间重复边取最大**原权**，按(source,target)排序，再weight/(接收行sum+delta)归一化。edge[0]是接收/query、edge[1]是邻居，无跨section空间边。
- 默认保留self-loop；no_adj_self是现有实验模式。pre/post使用相同图但不同参数；不添加feature边、第二次混合归一化或feature图调度。
- spatial graph cache现有key缺坐标/spot identity，见B11。其修复单列，不在纯搬动中顺带修改。

初始prior调用链：runner.initialize_model_ot_prior → Stage.initialize_candidate_sparse_ot_prior:814 → sparse_uot.compute_initial_bidirectional_candidate_sparse_uot_prior:633。每相邻A/B、每模态做正反检索，反向结果统一映射至(A_local,B_local)，support去重，先平均模态cosine cost，再保留行/列candidate_k的**并集**，调用一次solver。candidate_k不保证最终每行度数≤该数；attention_topk是另一阶段的宽度。

权威solver是 sparse_unbalanced_sinkhorn_bidirectional_topk:402：FP32、uniform marginals、rho=tau/(tau+epsilon)、exp kernel floor、加性stabilizer、u后v、固定迭代、既有nan_to_num/总mass归一；同一p_edge由 _edges_to_sparse_topk:340 按行/列分别产生两个方向prior，confidence=top-k coverage。不能转置已经行归一的top-k表充当反向prior。solver的现有数值保护不是普通防御代码，本次不简化其公式。

检索IVF、FlatIP、blockwise是真实候选后端，不等于dense OT；保留实际消费者，不新增自动fallback。

### 2.3 初始化、刷新和最终导出

当前动态刷新固定为ot：runner update helper → Stage.prepare_ot_prior_refresh(outputs) → ot_embeddings.detach；D4已删除suite/CLI/config/API中的source选择，见§18。topology=true/weight=.2保持，context对同一ot embedding做self-excluded spatial pooling/L2 normalize；候选检索使用ot semantic embedding，cost=.8*semantic cosine+.2*context cosine。普通attention gate与此cost保留。额外attention context gate/alpha反传实验已在D3退休，共享OT spatial pooling保留，见§19；B3随fused/final刷新source退休关闭，不恢复废弃分支。fused/ot/final阶段embedding不删除。

| 时点 | 必须冻结的实际动作 |
|---|---|
| 初始化 | 在first forward前从预处理模态求双向sparse P0；不随机重排section/spot |
| MouseBrain epoch0 | model默认train，no_grad dry forward；消耗dropout RNG；无feature图 |
| epoch1/20/99 | train forward→loss/backward/step；没有额外图或OT刷新 |
| epoch100及100+n*interval | step后eval+no_grad，一遍完整forward使用旧prior；prepare取ot.detach并构造同源context；双向检索/solver完成后整体替换prior |
| 下一epoch | model.train，使用新prior；prior构造无梯度，固定prior上的attention仍正常反传 |
| 末epoch200 | step后照常刷新，然后独立final eval/no_grad，使用新prior导出；无201训练、无从final embedding额外重解OT |
| 末epoch199等非刷新点 | 保留最近prior做最终eval，不补刷新 |
| 保存状态 | train history末loss来自step前train；final loss来自step后eval；prior生成源是refresh的ot embedding，不是导出的post-GS final embedding |

刷新证据：MouseBrain:715–728；CRC:916–956；generic:313起也单遍刷新，但其dense路径将迁移/退休。最后输出证据：MouseBrain:730–755，CRC:958–987，SPATCH:725–729/815–819。

保存能力并不完全相同：MouseBrain/MISAR等已有可选prior导出；SPATCH目前只导final embeddings、summary/history，未调用prior exporter。generic/HumanEmbryo已有weights export。不能把新增SPATCH prior导出或生产resume伪称当前已存在。新运行可增加不改变模型的状态说明，但不得改变已有下游所需文件名、顺序和字段。

## 3. 保护范围、缓存复用与外部接口

### 3.1 不可改动对象

| 对象 | 本轮与后续重构边界 |
|---|---|
| gene_imputation/ | 整体原样保留：代码、配置、tests、README及产物 |
| gene_imputation_spatial_smoothing/ | 同上；其跨目录runpy/importlib路径保持 |
| gene_imputation_shared_gene_validation/ | 同上；旧sibling import测试问题仅记录，不安排修改 |
| preprocessed_cache/ | 原位置/内容/格式保留；只读加载，不删除、移动、改名、重写manifest、升级schema或自动再生成 |
| results/及所有现有result_* | 原位置/原内容保留；不跑会写指标、绘图、overwrite或prune它们的脚本 |
| 未来验证输出 | 只使用refactor_checks下新独立位置，不写上述区域；既有P0 fixture/报告保留，本轮不创建测试产物 |

核心重构不能要求修改受保护消费者来适应新接口。未来测试可用独立小fixture验证其格式合同；不运行那些默认会写入插补runs或历史result的main。

### 3.2 三个插补目录的真实外部依赖

只读源码/import/README/config核对，没有直接model或scripts import；主要相互使用三个目录内部的sys.path、importlib、runpy和固定文件路径。无需给核心新增一个“插补兼容层”。

外部依赖是 result_v6/spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42 下的：

- final_embeddings_section1.npy、final_embeddings_section2.npy：section顺序与原spot行序，现有float32数组合同。
- run_summary.json 的 alignment[section].n_spots。
- spot_metadata.csv.gz 的section/x/y与有序spot；插补run_spatch_knn_imputation.py:244–273对summary数量、metadata坐标顺序进行核对。
- 配置指定的raw h5ad/基因标识路径，以及插补目录自身的H5/CSV等产物。

保护这些旧位置即保留现有命令；新核心导出也应继续能生成同样基础文件/字段，额外resolved config可单独存放，不能以新schema代替旧必需字段。不得重排barcode、section、模态或feature轴。移动analysis或data_io不移动这些数据路径，不更改三目录内部脚本引用。

### 3.3 SPATCH现有cache的具体合同

当前实际cache：

    /home/hujinlan/spa_mo_model/preprocessed_cache/spatch/v6_harmony_n50_hvg3000

只读manifest SHA256：4fdf8ed1a3c4beaa856eff4e91e3970639543718afce45a5527cfb159754fba7。schema_version=1，cache_boundary=model_ready_feature_dict_after_PCA_Harmony。这已经是模型输入，不重新做normalize/PCA/Harmony。

| 内容 | manifest记录 |
|---|---|
| sections | section1、section2；665399、403563 spots |
| feature数组 | 各section RNA 50维、Protein 15维、HE 50维；float32 |
| spatial | 各section N×2、float32 |
| 预处理参数 | n_comps50、RNA HVG3000、其他HVG null、target_sum null、Harmony true、metacell false、memory_efficient true、retain_processed false、DAPI已移除 |
| metadata | spot_metadata.csv.gz、shared_rna_genes.csv、protein_markers_after_dapi_removal.csv，均有hash记录 |
| 来源 | preprocessing_sources记录run_spatch.py和model/data_preprocessing.py的hash；source_files记录raw path/size/mtime/hash；alignment记录原构建时空间对齐和canonical ID唯一性 |

**迁移风险C1（修改前，现已解除）：**原loader run_spatch.py:359–369把整个run_spatch.py、data_preprocessing.py源码hash与缓存构建记录比较；C1前核对时两者仍匹配。移动训练loop/loader或做纯重构就可能使其拒绝同一缓存。这不是数组必须重算的证据；C1已解除该加载门槛，保留原hash作来源记录。

新loader使用普通函数与必要的显式输入/预处理身份，先在现有run_spatch.py内整理接受条件，沿用schema=1和数组；C1不提前迁到data_io。具体文件、字段及小替身矩阵见 [USER_DECISIONS.md §5](USER_DECISIONS.md)：

1. 在加载边界确认section集合/顺序、所需模态、每section各模态N、空间N、输入维度及metadata行序。依manifest相对路径读取，不动态猜格式，不自动排序对齐。
2. 整个run_spatch.py/data_preprocessing.py源码hash及raw path/size/mtime/SHA保留为provenance；整源码相等、raw路径/mtime相等不再单独决定兼容性。CLI默认data_dir=None；默认消费所选cache，显式同来源data_dir仅作缓存来源选择，均不自动打开raw。可选--input_identity_manifest复用含source_files的JSON声明，逐source比对SHA和size，冲突拒绝；显式另一data_dir缺凭据同样拒绝。声明相容不等于验证当前raw，始终记录current_raw_files_verified=False。cache与输入维度、HVG/Harmony/DAPI/metacell等必要预处理语义冲突同样拒绝；memory_efficient构建语义保留，retain_processed仅记来源、不作硬门槛，其余未知或冲突参数拒绝。训练参数不进入cache身份。这是加载接受条件变化，不是纯函数搬动；不重新hash真实大raw或触发预处理来补身份。
3. 保留一次加载的数组/metadata完整性检查，不层层/逐epoch重复扫描。C1仅解除整源码相等及raw路径/mtime相等的硬门槛；仍按上一项处理显式输入身份冲突和证据不足。保留缓存内容完整性与对齐检查，不改变数组精度、顺序、拷贝/显存语义或数值验证。
4. metadata需要用于新运行时，只读来源，在新output保存必要副本；不批量复制cache、不反写manifest。缓存没有某项信息就报告限制，不生成“补齐后新缓存”。
5. 主SPATCH入口仅load：suite不自动追加--build_preprocessed_cache，run_spatch的parse_args/main同时处理cache缺失/build请求，不能落回raw预处理。新运行cache-mode校验对应loaded；必要来源说明区分实际cache、历史raw和显式请求，不把未核验的args.data_dir冒称实际输入。缺失/错配明确停止，不以“失效”为由重算；不动其他数据集预处理，也不借本批删除SPATCH历史预处理helpers。

**已知证据限制：**array记录没有每个feature数组内绑定的ordered barcode ID。独立spatial数组及spot_metadata有hash、alignment有历史对齐声明，但manifest本身不能证明所有feature行的生物学身份。后续加载时按原行序核对metadata的section/x/y、canonical ID规则（run_spatch:141）及N/维度，可发现错配；不宣称能从缺失ID重建历史证据。本轮未读数组/21MB metadata、未核raw stat或全量checksum，不标记这些动态检查已通过。

这是一项**加载接口/接受条件迁移**，单列于纯数学重构之外，必须在改变被hash的源文件前完成；不建立通用cache版本框架。它与B9的分析结果缓存是两回事：B9失配可在新输出重算小评估，绝不能触发SPATCH预处理重算。

## 4. 职责划分和旧文件/函数迁移表

先保留model/包和数学文件。仅在消除真实跨层依赖时提取普通函数；最终目录是职责边界，不是一次性建齐的文件清单。

- model/：Stage、层/attention、空间图、sparse UOT、检索、loss。
- data_io/：读取/对齐、通用预处理、现有cache loader；不含fit。
- training/：普通fit函数、配置解析、刷新调度、保存。
- analysis/：显式结果loader、标准化/聚类/指标/绘图；不导入训练入口。
- scripts/：薄入口；configs/：完整v7A preset与dataset设置；tests/：少量行为测试。

最终训练/分析职责都覆盖§1.3的九数据集。data_io保留九套必要适配，可将确有共性的适配放在同一文件；training提供共享fit和解析后的配置；analysis提供九数据集显式loader/协议选择及需要的绘图；scripts提供能明确选择dataset/model preset/input/output的薄入口。统一的是调用框架，各数据集的预处理和评估协议保持实际差异，不建设registry或抽象trainer。目录按迁移批次逐步形成，本轮不创建这些代码目录。

导入方向：scripts调用data_io/training/analysis；training调用model和显式数据接口；analysis读取产物，必要数据工具来自data_io；核心库不反向import scripts。不新增registry、抽象trainer、plugin/service层或一串转发模块。

| 当前真实位置/函数 | 建议权威位置/动作 | 迁移前必须保留的调用者与合同 |
|---|---|---|
| model/stage_model.py、model_component.py、loss.py | 原位保留；一个Stage、两个GS实例 | constructor/forward有效参数、计算顺序、state映射 |
| model/sparse_uot.py双向链、faiss_candidate_search.py | 原位保留；去掉已退休单向链即可 | 正反检索、共享coupling、solver数值、chunk/内存语义 |
| utils.compute_spatial_knn_graph_with_weights | 先原位，最终若utils仍混杂则整体图函数迁model/spatial_graph.py | Stage和小图测试；不改变edge排序/self/权重 |
| utils通用杂项 | 有数据/summary职责的随真正消费者迁data_io；无用者删；小tensor工具可原位 | 不造新巨大utils，也不拆一个函数一个文件 |
| configure.get_default_model_config | 暂保留纯模型默认；完整解析移training/config.py | model不导入scripts；最终constructor消费resolved模型配置，不再次递归override |
| configure预处理默认、data_preprocessing.load_cosie_style_data/load_data/preprocess_adata | data_io/preprocessing.py，按复用函数整体迁 | 先处理C1；所有runner、run_preprocessing、UMAP数据准备调用迁完再删旧别名 |
| multimodal_preprocessing的数据装载/对齐函数 | data_io/datasets.py或少数确有规模的dataset文件 | MouseBrain/generic与HE输入、section ID/obs-var/feature顺序 |
| image_preprocessing的HE/UNI输入链 | 可先原位；最终归data_io/image_features.py | raw image/已有UNI feature的实际读取；UNI checkpoint不是Stage恢复 |
| run_spatch.load_preprocessed_cache/cache_parameters及metadata装载 | data_io/spatch.py普通既有格式loader | 先C1，保持cache路径/格式；不带save_preprocessed_cache自动调用 |
| run_crc_stereocite.train_small_crc_model/run_one_forward/AMP/scaler/monitor | training/fit.py及确有复用的runtime helper | MISAR、SPATCH、HumanEmbryo、四个源码适配器；不把整个1500行runner直接搬成新巨文件 |
| MouseBrain本地loop、generic train_stage_model loop | 逐个接同一training.fit | 保留epoch0 dry、loss schedule、bundle/preprocess/synthetic、weights export差异 |
| CRC/MouseBrain sparse_prior_kwargs/initialize_model_ot_prior/update_model_ot_prior | 一组普通prior调度调用，可留fit同文件 | dense调用者先迁；sole source来自resolved；必要模型方法原位 |
| Stage.should_update_ot | 与fit迁到training调度职责，可留fit同文件 | 首次100/interval计算精确保持；先迁tests和旧runner imports |
| CRC/MISAR的数据helpers | data_io适配函数 | prepare_rna_var_names_make_unique、read_backed_pair、subset_to_memory、common_var_names、make_unique_rna_var_names有分析数据准备调用 |
| run_mouse_spleen/HLN/Simulation/Thymus 的 _load_pipeline、_adapt_pair及argv/globals注入 | 显式prepare_dataset函数和dataset设置 | 各自obs/var/Protein/ATAC语义；先消除文本替换依赖再改CRC/MISAR源文件结构 |
| scripts/hesta_rna_utils.py | data_io/hesta.py（有必要时整体迁） | HumanEmbryo RNA读取/预处理/已有manifest读取；不改变其输入算法，不将其独立缓存流程套到SPATCH |
| analyze_mousebrain_v4_standardized.load_data、analyze_result_v3_standardized的_run/_metadata_data/_misar_data/_paired_rna_adt_data/_thymus_data/_simulation_data | analysis/loaders.py或同等少量明确loader | 当前requested analyzer真实使用；参数为run_dir，去globals monkeypatch，不能整文件先删 |
| compare_*_kmeans_preprocessing的loader/metrics、batch_correction_metrics | analysis的结果读取/metrics/protocol函数 | 既有dataset标签过滤、raw/standardized空间和k/seed/n_init保持 |
| compare_spatch.fit_or_load、retained_outputs与协议常量 | analysis/spatch.py或共用分析文件 | requested analyzer自己K_VALUES不被compare入口ALL_KS覆盖；输出写新目录 |
| generate_result_v6_spamosaic_umap:45–55及generate_comparison_method_umaps | 跨方法UMAP KEEP，数据依赖改data_io，绘图函数归analysis；SpaMosaic风格输入/整合工作流暂时KEEP | 前者当前导入CRC/MISAR及Thymus/Simulation训练脚本；先迁helpers。D8c后续说明具体I/O/caller及是否实际运行SpaMosaic模型，不阻塞C1，不先删 |
| analyze_spleen/thymus/HLN/simulation的源码exec；analyze_spatch修改base globals | 参数化现有分析函数 | 保留其协议的调用者先迁；不只是重命名文件 |
| v7A suite及旧suite/daemon | 薄显式dataset列表、preset、output root；无用旧main退休 | 不运行旧launch/prune；所有被import的命令/验证函数先迁 |
| 三个gene_imputation* | **不迁移、不修改** | 维持内部路径和外部输出合同，不让其依赖新core API |

少量保留调用方确需的旧import名只能有明确名单和结束条件；不是为所有历史参数保留alias链。已受保护三目录并无core import，因此不能借其名义保留全部model兼容包装。

## 5. 已决定的保留和未来退休范围

### 5.1 保留主路径，先迁后删除dense/单向

| 删除对象（当前行号） | 迁移前置条件 |
|---|---|
| model/linkage_construction.py全文件 | Stage唯一生产import迁掉；dense initial/update/solver/zscore/topology helper没有sparse消费者 |
| model/utils.py:319 cosine_cost_matrix、:309 l2_normalize | 当前Python引用仅dense linkage与cosine helper互用，随dense退休复查后删；model_component的同名布尔参数不是对此函数的调用 |
| Stage.initialize_ot_prior:773、update_ot_prior:881 | MouseBrain/CRC dispatch、generic:269–270/refresh、smoke都显式走双向sparse |
| Stage.forward:1289–1292隐式dense初始化 | 保留入口在forward前初始化prior；启用OT却缺prior明确报错，不藏fallback；合法无OT另处理 |
| sparse_uot.compute_initial_candidate_sparse_uot_prior:796、update_candidate_sparse_uot_prior_from_embeddings:920、sparse_unbalanced_sinkhorn_topk:480 | Stage选择分支和测试迁完，再删单向独立循环 |
| 单向专属helpers | _union_candidate_indices:64、_compute_candidate_cosine_cost:85、_truncate_candidates_by_cost:111、_candidate_topk_change_diagnostics:213、_candidate_qc:571；最后再次查调用 |
| Stage sparse init/update bidirectional参数和forward:1351–1399单向循环 | 启用OT唯一双向，保留同步update收集/平均/apply_update |
| CLI ot_prior_mode、bidirectional_ot_attention及关闭别名 | 更新保留entry/suite/config/current文档命令后删除；旧dense/单向显式输入明确拒绝，不静默变算法 |
| dense专属配置tol/check_every/clip_cost_*/keep_dense | P2已删除；epsilon_init/update在P1后已被generic/smoke双向sparse实际消费，保留.08/.05不同值，不能再作为dense-only删除；其余sparse epsilon/tau/max_iter/attention_topk保持 |
| 专属tests与P_dense=None/has_dense_P输出占位 | 模态/形状/梯度覆盖迁完；下游消费核查；新输出删除占位不改旧metadata |

**不得误删：**双向initial:633、update:1075、solver:402；_candidate_edges_from_search(reverse_query):133、_deduplicate_edges:149、_compute_edge_cosine_cost:161、_prune_edges_bidirectional:305、_edges_to_sparse_topk:340、_make_direction_prior:603及双向QC/context工具。OTGuidedAttention.compute_update_only:464与apply_update:687有必要职责，checkpoint/chunk/context方法也保留。只删独立单向生产模式，不删双向中的方向计算。

B7随dense退休，不开发dense双向。迁移旧dense/单向入口属于已批准运行模式变化，不承诺旧算法结果等价；已经是双向sparse的v7A路径严格对照。迁移后必须实际删除旧分支，不能只换默认。

### 5.2 高置信度删除与迁移后退休建议

九个候选无合理当前调用者的函数：utils.compute_knn_graph、loss.compute_reconstruction_loss、utils.standard_modality_result、utils.summarize_adata、stage_model.summarize_ot_topology_cost、data_preprocessing.preprocess_rna_adata/preprocess_protein_adata/preprocess_metabolite_adata、image_preprocessing.rescale_image。实施删除前重新检查imports、CLI、tests、文档命令、exec/getattr和三插补/缓存依赖，不能仅靠“无直接调用”。不先修已DEAD的旧unweighted KNN再保留两套图。

建议在函数迁出后退休以下旧启动壳，无须因为有对应result目录而永久保留：

- run_result_v4_suite/large_suite、v4b、v4c/large、v5_delayed_ot/missing_v3_matched/spatch_daemon、v6_dual_graphsage/missing_v3_matched/spatch_daemon：旧调度/输出命名壳；迁走互相import的命令/验证helpers，保留当前确需的数据集设置，再删旧main。
- run_result_v7a_suite可最终由显式批量入口覆盖；run_spatch_v7abc_suite的cache自动build与历史输出写入逻辑不带入新入口。训练/分析/验证共享一份dataset列表。
- 旧analyze/compare/generate/postprocess/plot_and_prune入口中，仍被analysis或保留绘图使用的函数先迁。删除旧壳不删除历史图/CSV，也不执行壳内的清理动作。
- Stage.forward的processed_data_dict及旧位置参数order/epoch解析，load_cosie_style_data的入口别名，CRC --dry_run和smoke标志：先迁真实调用者再去无职责兼容；保留真正dry/smoke功能。OTGuidedAttention.forward便利组合只有待退单向caller时，可随后裁剪，不动底层两个必要方法。
- 当前没有的feature graph/v8–v15实现无需“删除批次”，更无需重新引入后再兼容。

以上旧壳在依赖清单闭合前标“需迁移后删除”，不先宣布DEAD。不为保留其路径新建无限期转发层；若发现实际用户流程只能由某壳提供，将该流程集中列为待确认，而不是保留整个历史树。

### 5.3 用户最终能力决定

以下是未来保留/退休范围，本轮仅实施C1，不同时清理模型、预处理、分析或旧suite。

| 能力 | 用户最终决定与实施边界 |
|---|---|
| D1 post-OT scale | KEEP普通可调scale，v7A默认.5；RETIRE B/C命名preset和suite壳，先迁调用者。仍可用普通参数设置.25/.75，不增独立模型或trainer |
| D2 G0/self-path | **已实施，见§21**：legacy/no_adj_self和模式选择退休；两个unused self_linear权重实际删除。保留原no_self_linear计算、自环和outer residual；schema及构造RNG变化明确记录，同有效参数映射后严格等价 |
| D3 context gate/alpha反传 | **已实施，见§19**：额外gate、alpha实验开关与专属大型benchmark退休；普通gate、正常softmax alpha反传、.8 semantic + .2 context动态cost及共享pooling保留 |
| D4 dynamic refresh source | **已完成，见§18**：仅KEEP ot，fused/final刷新源及source选择参数实际删除；B3以路径退休关闭。fused/ot/final阶段embedding和ot刷新时序保持 |
| D5 metacell | **已实施，见§20**：本项目metacell生成/训练参数/还原实现已删除；显式true拒绝。KEEP full-spot、schema=1缓存metacell=false身份、旧产物/映射及外部结果读取 |
| D6 单模态、H&E/UNI及九数据集适配 | KEEP完整九数据集所需能力，并KEEP原图+mask→UNI提取和已有HE/UNI特征两种输入。Human Embryo单模态不等于单向OT；合法无OT不随独立单向OT误删 |
| D7 candidate retrieval | KEEP FAISS和生产CLI显式blockwise以及小型验证后端，不新增自动fallback；不随dense OT误删 |
| D8a/D8b 比较与跨方法UMAP | KEEP方法间KMeans预处理/指标比较及跨方法UMAP，使用显式输入/输出/参数；RETIRE迁完的版本化壳，不退休协议、loader或必要plot |
| D8c SpaMosaic风格输入/整合工作流 | 暂时KEEP；后续说明具体输入、输出、调用者及是否实际运行SpaMosaic模型，不根据风格名称推断。不阻塞C1，说明前不先删 |
| weights export | 保留既有能力；新增推理状态恢复/精确resume不在本计划内 |
| 三插补目录 | 明确保护，不是“待确认是否删除” |

逐项证据、未来删除影响及最终决定见 [USER_DECISIONS.md](USER_DECISIONS.md)。D1–D8不再作为未决研究选项反复确认；仅D8c需要后续具体工作流说明并暂时保留。九数据集支持不是退休候选，普通DEAD helper不进入用户决策表。其他退休须在相应批次先迁依赖并验证，未纳入本轮C1；数值bug修复另行批准。

## 6. 配置、数据与分析的简洁合同

### 6.1 配置只解析一次

最终优先级：**基础defaults < v7A或明确实验preset < dataset config < 用户显式CLI**。布尔也区分未给/True/False；argparse.SUPPRESS或等价provided集合即可。一次解析/校验后，模型、trainer、refresh和save读相同resolved值，不在constructor/adapter再次递归override。

完整resolved config保存实际用值及来源、原argv、base preset/overrides、代码身份、输入/section顺序、AMP/chunk/候选设置。目录/epochs/summary source从resolved生成，不再仅复制输入JSON。模型超参数和数据集实际预处理/训练设置分开记录，不统一epoch/K/candidate等。

B1/B2的precedence缺陷和B8假字段已分别按**配置bug/接口行为变化批次**处理，见§17/§22；其后的P3a共享解析提取已完成，见§23。主v7A命令下effective值保持；A/B批中冲突输入按新优先级、已知无效字段明确拒绝，P3a不再改变这些语义。

无消费者字段如use_feature_graph、use_spatial_graph、use_distance_weight、num_layers、ot_attention.direction、contrastive.loss_weight等删除或明确拒绝，不实现新科研功能来满足它们。dense真实消费字段保留至调用者迁完，再删。D4完成后刷新直接固定ot，不再保留CLI/model config两份source选择；旧source配置只在Stage构造边界一次明确拒绝。内部不层层重复校验，不新增静默fallback/别名链。

### 6.2 一个fit，显式数据准备

数据适配可不同，但都提供有序features/spatial/metadata/section_order和实际输入维度。保留SPATCH缓存直接输入；其他dataset沿原预处理算法/顺序。用普通函数参数替换replace+exec、argv注入和suite globals，不通过动态接口猜测拼接流程。

一个主要fit保留dataset已有运行设置：MouseBrain前置train/no_grad dry forward、loss schedule；CRC链training-only autocast、checkpoint/chunk、内存清理；generic bundle/synthetic与权重保存。只参数化真实差异，不新增几十个无效boolean或可编程callback框架。processed_data_dict若只供保存/summary使用留在数据/输出侧，模型兼容参数在caller迁完后删除。

### 6.3 分析读取、协议和新输出

分析接受明确run_dir和独立output_dir，不能为读embedding import历史训练入口。用少量明确loader读现有格式；不会重建历史模型。旧结果按原格式只读，新增resolved记录不要求改写旧summary。

必须覆盖全部九数据集，并保留同一数据集的不同实际协议为显式选择。下表原v7A requested协议不覆盖v6完整实验协议；例如MouseBrain两者绘图/保留K范围不同，SPATCH requested K列表不等于v6全K分析。详细历史config证据及九个分析入口见USER_DECISIONS §1.2。

保持协议：

| 对象 | 当前协议 |
|---|---|
| 标准化 | joint在全体spot fit StandardScaler；independent每section单独fit |
| 聚类与内部/监督指标 | 在相应标准化空间；保留ARI/NMI/Homogeneity/Completeness、ASW/CH/DBI、truth缺失过滤和采样规则 |
| MouseBrain KMeans | joint k2…12，independent 5/6/8/9/10/11/12，seed0 |
| MISAR / Spleen / Thymus | MISAR k2…16，Spleen/Thymus k2…12，seed0 |
| Simulation | k5/8/10/12，seed42 |
| Human_Lymph_Node | StandardScaler+KMeans，k2…12、seed0，全量内部指标，无ground-truth标签；保留K5/8/10/12 |
| CRC_Stereo-CITE-seq | MiniBatchKMeans，k5/10/15/20/25、seed0、n_init20、max_iter300、batch8192；固定10000样本ASW、CH/DBI全量，无ground-truth标签 |
| Human Embryo | MiniBatchKMeans，k2/3/4/10/15/20/25、seed42、n_init3、max_iter100、batch8192；metric5000；section-mixing从标准化joint_space采样100001；developmental section为生物发育时间，其混合指标仅作诊断，不作为应消除的batch；celltype及25叶簇质心加权Ward嵌套层次聚类保留 |
| 上述KMeans实际值 | n_init20、max_iter300；各dataset sampling/label协议仍分别记录，不套用于Embryo MiniBatchKMeans |
| SPATCH requested分析 | MiniBatchKMeans，k5/8/10/12/14/16/20，seed42、n_init20、max_iter300、batch4096；tol0、max_no_improvement10、reassignment_ratio.01 |
| SPATCH独立compare入口 | ALL_KS2…20、RETAIN_KS5/8/10/12/16/20是另一协议，不覆盖requested的K_VALUES |
| 原v7A requested batch metrics | 对已核对调用保留raw final embedding输入和函数内部逻辑；LISI90、kBET50、PCR50、ASW10000及dataset独立max_samples/seed。不得套用于Human Embryo的标准化joint_space/发育时间诊断，也不统一九数据集的输入空间 |

B9修复只针对**新输出分析**：completion及内部fit cache需关联embedding内容/shape/dtype、有序spot和truth metadata、resolved配置及协议参数/实现身份，不能只看文件存在。可以用一个简单manifest，在导出/读取边界计算一次所需标识；不做通用缓存框架、不逐epochhash。旧无完整来源的metrics保留为历史记录，不能冒充新运行结果；本轮不验证或重算它们。任何新分析文件只写refactor_checks等独立目录，绝不自动清旧结果。

## 7. 旧审计问题的当前状态与修复边界

状态“仍存在”表示本轮源码复核到相同触发机制；不等于本轮重新跑了旧动态测试，也不等于默认v7A必触发。“已消失”只针对原缺陷，不代表整个模块无bug。未重新验证的影响不得写PASS。

| ID | 当前状态 / 证据位置 | 触发条件、建议及行为边界 |
|---|---|---|
| B1 | **F-config-A已修复，动态55项矩阵见§17** | 原显式epochs/device被nested model覆盖；现按默认→model→input JSON→显式CLI，目录与summary使用resolved。冲突输入允许按新优先级改变 |
| B2 | **F-config-A已修复，动态见§17** | override默认None，仅显式值覆盖JSON；布尔三态、scale/interval/candidate/topk/UOT/K等传到真实helper。主v7A解析值保持；B8及B3不夹带 |
| B3 | **RESOLVED BY RETIREMENT OF UNSUPPORTED SOURCE PATH**，D4见§18 | fused/final source入口、选择配置/API与fused专属context分支已删除；KEEP调用者全部固定ot，旧source请求拒绝、缺ot不fallback。未修复/重新支持fused；历史问题证据保留 |
| B4 | **仍存在，静态**；sparse_uot:358–383、model_component:618–645 | top-k补位权0被clamp后进入softmax；建议mask真实support，空行zero directional update。**待另批批准**，纯重构保留当前数值，不沿用上轮必修实施安排 |
| B5 | **已消失** | 当前无feature refresh函数/epoch1调用；不重新引入或为不存在路径造测试 |
| B6 | **已消失** | feature graph强制FAISS GPU的实现已不存在；OT后端仍需保留 |
| B7 | **随P2 dense路径退休退出** | 已删除dense初始化/更新及模式选择，不开发dense bidirectional；旧问题证据保留 |
| B8 | **F-config-B已完成，见§22及消费者矩阵** | 删除23模型/4预处理NO-OP默认字段和2无效CLI；已知退休/无消费者字段明确拒绝，保留65真实模型默认字段及入口/数据身份。有效配置/state/RNG不变，没有实现假字段声称的新功能 |
| B9 | **仍存在，静态**；analyze_spatch:184–189、compare_spatch.fit_or_load:418–426、通用analyzer:86起 | completion/fit cache只看存在或表结构，未绑定新embedding/order/protocol；按§6.3独立修新输出分析，历史结果和预处理cache不动 |
| B10 | **F-data(B10)已修复，见§24**；两个runner的main/audit | parse一次后audit显式消费data/output；help与解析失败无data/audit/output副作用。原CLI默认路径保留；显式路径不会被audit替换为旧全局目录。audit schema与adapter保持 |
| B11 | **仍存在，静态**；Stage:557–575 | 同section/n换coords或spot order可能复用旧图；建议固定输入图状态/显式identity，变更时重建或拒绝，不逐epochhash。独立cache正确性修复 |
| B12 | **仍存在，静态**；faiss_candidate_search:16–22及sparse转换 | 直接BF16 tensor先numpy报错；detach后float32再cpu/numpy。独立dtype修复，不能改变模型保梯度转换 |
| B13 | **仍存在，静态**；data_preprocessing:221–231 vs271–280 | shared模态漏target_sum；默认None不触发。独立传参修复，非默认预处理结果会改变；SPATCH已有cache不重新处理 |
| B14 | **仍存在，静态**；multimodal_preprocessing:340–389、generic:130 | section_ids=A/B与输出s1/s2不一致；出口统一feature/spatial/metadata/order映射。独立数据bug，不改变已对齐主输入顺序 |
| B15 | **已消失**；analyzer.SPECS:63–83与suite:104–165 | 当前均五数据集；仍建议未来训练/分析共用显式列表，但这是结构收敛，不再报告已知六对五bug |
| R1 | **机制仍存在；真实运行影响未重新验证**；loss:25–27/47–69 | signed latent joint分母零/近零可非有限；保留公式，反例单独记录，不softmax/换loss/打数值补丁。数学修复需批准 |
| R2 | **能力缺口仍存在；旧feature-cache反例已不适用** | weights未含OT prior/optimizer/RNG；当前无feature cache。保留weights export，不虚构现有resume/loader故障；新增推理恢复和精确resume分别另立范围 |
| R3 | **机制仍存在；真实cost分布未重新验证**；sparse_uot:431–450 | kernel floor抹平高cost差异；退休/整理严格保持floor、u/v、stabilizer/迭代/normalization，不改log-domain |
| T1 smoke契约 | **仍存在，静态**；run_stage_model:266–276、Stage:411–415 | 旧错误文本与当前单模态拒绝不符；测试修复独立，保留形状/非法输入/backward覆盖；dense用例先迁双向sparse |
| T2插补discovery | **机制仍存在，未重跑**；shared验证compute_scheme2_pcc:19 | sibling import问题在保护目录，**本计划不修其文件/测试**；不能把外部无关失败归为核心回归 |
| T3 context测试 | **D3专属断言随实验退休，见§19** | 旧513维/alpha/context-gate流程不再保留；小型共享pooling与ot topology验证迁出保留。普通gate同步双向更新由新固定状态测试核对，未修模型数学 |
| T4 checkpoint checker | **仍存在，静态**；validate_checkpoint_ot_attention:69–77 | zip忽略key/schema且max可能掩盖NaN；按有效参数名称映射、shape/None/finite检查。另bidirectional验证:86的k_neighbors是无效键 |

其他旧疑点均标**未重新验证**：AMP内float后matmul实际dtype、decoder target先量化、跨chunk dropout RNG、FAISS ties/线程确定性、memory_efficient预处理等价、带点output目录、generic bundle None支持。给出极小验证计划，不改代码让它“符合旧报告”。

C1缓存loader整源码hash耦合为此前核对发现的迁移风险，见§3.3；C1修改前缓存来源hash匹配，不能称旧加载已经失败。

## 8. 先冻结可重放基准，再安排修改

本节保留分批验证设计，已完成的P0范围和正式验收见§13；未动态覆盖的接口仍须在对应修改批次前补针对性小测试。本轮不重跑无新变化的测试。之后任何新验证输出仅放refactor_checks下独立位置；禁止完整数据集训练、昂贵benchmark、安装/升级依赖或批量复制大结果/cache。

### 8.1 基准固定内容

已冻结参考为两section 11/13 spots、三section 11/13/17；FP32 K5，SPATCH BF16 K10，另有7×11不完整support。固定非退化feature、坐标、section/spot/模态顺序、输入维度、模型有效权重、OT prior、spatial图及cache状态、mode/RNG、AMP/chunk/backend参数，使用已有fixture，不重新生成expected。**没有feature graph状态项**。

同seed不足：eager fusion模块和默认旁路self_linear仍消耗初始化RNG、占state_dict。删除unused nn.Module前记录构造顺序与key/shape，按同一可映射有效参数状态对照；不得只同seed重建，也不得zip忽略新增/删减项。

结构、索引、顺序、edge multiset、配置精确比对；输出/loss/有效梯度先验shape/dtype/finite，再比较数值。FP32和BF16参考容差均已固定为atol=rtol=1e-5，记录max absolute/relative error；已有逐位一致对照继续报告exact结果。不能放宽tolerance、忽略NaN、删失败例或重录golden宣布通过。pure-refactor严格条件见§13；现有证据以外的设备/运行设置不冒称已验证。

### 8.2 最小验证矩阵

| 编号 | 对照内容 | 必须证明 |
|---|---|---|
| V1 forward/loss/grad | 两/三section，固定相同有效权重/图/prior；latent/fused/pre/方向update/mean update/ot/final/decoder各阶段 | 每项crossview/MSE/total、有效参数及source/target输入梯度，requires_grad/grad None模式；不是shape或ARI近似 |
| V2 双向prior | 不等长矩形support/cost；candidate_k与attention_topk不同；正反检索映射 | 同一个coupling按行/列提两向prior；local/global index正确、support并集、coverage/top-k/tail定义保持 |
| V3 三section同步 | 记录B从A/C的update和apply次数 | 同一pre状态，平均后一次residual/norm；不串行更新、不重新定义方向平均分母 |
| V4 空间图 | self、reverse、重复坐标/边、n>k边界、cache固定输入 | section内边、max去重、接收行归一化、排序、自环、pre/post使用位置与scale；原基准残留feature字段不产生图，F-config后该字段按约定删除/拒绝，不保持静默兼容 |
| V5 sparse solver | 固定support/cost，低度数/高cost/floor、tie单独处理 | 原u/v顺序、fixed iter、stabilizer、mass归一和top-k；tiny显式矩阵算术oracle可用同一floor，但不保留dense生产实现 |
| V6 padding/空候选 | 不足top-k、某行无候选、全support为空 | 纯重构冻结当前行为及已知B4反例；全空support现为ValueError。若另获准修B4，再验真实support归一化/空行zero update，正边梯度；不能暗改 |
| V7 chunk/checkpoint | pure-refactor固定precision/AMP/chunk/checkpoint，或使用已验证同布局checkpoint对照；保留跨chunk诊断 | 固定布局检查输出/loss/梯度/一步参数/RNG；原cross-chunk FAIL归KNOWN_NUMERICAL_SENSITIVITY，不阻塞符合§13的批次。改变chunk、loss归约、autocast边界或checkpoint分块组织须重新审查，不能借此gate宣布等价 |
| V8 调度 | 人工驱动0/1/19/20/99/100/101/119/120/199/200，spy step、forward、mode/grad、prior generation | MouseBrain额外dry消耗、100起单遍刷新、P_old→P_new时点；不实际训练100epoch |
| V9 导出/weights | 最终199与200，极小独立输出/内存替身 | final使用的权重/prior、last refresh与metadata对应；现有文件名/dtype/spot order/summary字段保持；不声称weights=resume |
| V10 数据/cache/配置 | 小合成与既有manifest同schema的替身，源码hash改变、section错配、参数冲突、mock I/O事件 | P0先记录当前覆盖/提前I/O及cache拒绝机制，不运行真实旧help；C1后验cache只读且允许搬动，F-config/F-data后才验冲突处理与help无I/O等修正预期；不要求冻结时先修bug |
| V11 分析/外部合同 | 小embedding与metadata/truth重排、joint/independent scaler；旧输出格式synthetic fixture | 原聚类/指标输入空间、k/seed协议；三插补需要字段可读；不改三目录，不写旧结果 |
| V12 新分析cache | 内存/极小新目录改变embedding/order/config/protocol | B9修复单独验失配不复用；不触发SPATCH cache重建或旧metrics重算 |

小fixture无法跑历史20簇时，用可行小k验证算法，用参数记录替身验证真实k列表；不把缩小数据集冒充原协议已完整运行。原基准自身若出现非有限loss，停止该基准认证并记录R1；不修改loss使测试通过。当前未证明的CUDA数值和峰值显存保持明确限度，后续只在现有环境上用极小probe核查梯度与释放点，不能升级为昂贵benchmark。

## 9. 按依赖排序的小批次

P0已按§13正式收口，C1已按§14完成并验收为PASSED WITH DECLARED COVERAGE LIMITS；其他批次和§5.3已决定退休项本轮未执行。**没有“恢复v15A”或“保留F lifecycle”批次。**纯重构只对§13固定条件下的当前v7A有效状态承诺等价；修复或批准的模式迁移使用独立预期。所有迁移批次必须保留§1.3完整九数据集的能力。

| 批次 / 依赖 | 范围和类型 | 验证、停止条件 |
|---|---|---|
| P0 已收口 | BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY；无生产逻辑修改 | 两套固定配置13/13独立重放及同布局checkpoint对照通过；跨chunk保留FAIL。未动态覆盖的数据/分析/导出接口在对应批次前补小测试，不宣称V1–V11全部通过 |
| C1，已完成；PASSED WITH DECLARED COVERAGE LIMITS | 现有SPATCH cache依据真实输入/预处理身份判兼容，解除整源码hash及无关来源位置耦合，原格式只读；suite和主入口取消自动build/回退预处理；**加载接口迁移** | 7项旧机制冻结，新合同73/73小替身PASS，FP32/BF16各13/13固定P0重放及同layout checkpoint差0；原cross-chunk FAIL不变。未读真实大数组，summary/export仅静态；详见§14与refactor_checks/c1_cache_contract_20260915 |
| P1，已完成；覆盖限制见§15 | 九数据集保留入口、generic、smoke与checkpoint验证入口实际迁为双向candidate-sparse；**已批准模式迁移**，底层dense/单向保留至P2 | 入口78/78、generic/smoke16/16、validator对照及C1回归73/73 PASS；FP32/BF16固定配置各13/13及同layout checkpoint差0，原cross-chunk FAIL不变。没有未迁移的已识别保留入口；未运行九数据集真实训练/分析/导出 |
| P2，已完成；见§16 | 实际删除dense文件/Stage接口、独立单向solver/helper/attention遍历及模式选择；20生产文件净删1195行；**批准退休** | P2 19/19、P1适配70/70、generic15/15、C1原73/73 PASS；FP32/BF16各13/13及同layout checkpoint差0；原cross-chunk对象同值FAIL。0 executable blocker，参数/RNG不变；保留合法noOT及双向数学 |
| F-config，P0/P2/C1后；A/B完成见§17/§22 | A修复B1/B2；B独立删除/拒绝无效配置；**配置bug/接口变化**，没有提取共享解析 | V10冲突表、消费者矩阵和V1非触发值通过；主v7A有效值保持，旧NO-OP明确拒绝。B3已随D4源退休；P3a仍待单独授权 |
| P3a，F-config后；已完成见§23 | 通用解析提至training/config，训练入口提供完整resolved给constructor；**纯提取** | 旧有效resolved/目录/summary逐字段保持，仅Mouse新增source_layers来源metadata。四文本包装及dataset/modality常量保持；Stage显式config不再补默认，无KEEP partial依赖 |
| F-data(B10)，P3a后；已完成见§24 | 单独修parse前I/O和audit对硬编码data/output的依赖；其他数据bug不混入 | V10真实parser/独立help进程与合成audit通过；显式NEW_OUTPUT只写新位置，原CLI默认值不迁。正常数据/数值保持，路径与时机变化单列 |
| P4a，P3a/C1及B10独立修复后；已完成见§25 | 提数据适配普通函数，消除四个训练replace+exec/argv注入和CRC/MISAR文本锚点；**纯组织重构** | 四包装及CRC/MISAR共12组old/new全等，9异常/6导出对照通过；B13/B14保持，不能夹入本批 |
| P3b，P4a后；已完成见§26 | 将dataset默认显式化并完成各入口单次解析；**纯配置组织收敛** | 九数据集27组完整配置及18组实际main/交接同值；Protein/ATAC隔离、B10/C1保持 |
| P4b，已完成，见§27–§29 | CRC loop先迁training.fit，再MouseBrain/generic逐个接入；prior调度和保存共用；**三个小型纯重构批次** | V1/V7/V8/V9逐入口；保留epoch0 RNG、AMP范围、loss schedule、weights及可选导出。若需扩大callback/boolean体系先缩小拆分 |
| P5，P4b后 | 九数据集薄train/batch入口，显式dataset/preset/output root替换suite globals；先迁被依赖函数再删旧launchers | 全九项参数/任务/路径对照；统一框架运行当前v7A，不强制同超参数；旧历史输出不写，旧壳无调用者才退休 |
| F-analysis，P0后、P6前 | B9新输出cache来源关联及明确input/output分离；**独立评估correctness** | V12；相同来源可复用，变化不复用；禁止触碰历史指标/预处理cache。不以改聚类算法解决cache错误 |
| P6，P5/F-analysis后 | 迁九数据集analysis loader/metrics/plot函数并提供明确入口，消除analysis→历史训练入口及分析exec/globals；迁后退休无用旧分析main | V11逐数据集核对旧格式和真实协议，包括Embryo/CRC/HLN；不统一算法、seed、K或抽样；三个插补接口不变，不整文件盲删 |
| P7，相关调用迁完后 | 九DEAD函数、无用变量/参数/位置兼容小批清理；只对有证据的无用逻辑删除 | 引用检查+V1有效参数映射；D2已决定退休额外mode，但self_linear等注册模块不得按DEAD删除，独立验证RNG/state_dict/optimizer后再处理，不能仅靠同seed |
| P8 最后 | 必要时将预处理归data_io、图函数归model/spatial_graph，补当前导航/命名；已有合适文件不强搬 | C1在先、V1/V10/V11、所有import和受保护路径；核心无scripts依赖，不建新巨大utils/转发文件网 |

P1→P2必须完成迁移与实际退休两步；不能宣称P1改默认就已精简。C1可与不触碰被hash文件的任务独立review，但在任何修改run_spatch.py或model/data_preprocessing.py前必须完成其加载合同迁移。按小补丁逐项检查依赖，保持任何中间提交的保留入口可运行。

**其他bug修复轨道独立保留，不默认执行：**

- F-data：B10（parse/I/O）、B13（target_sum）、B14（ID）各一批；V10分别验证触发后明确预期，默认正常输入保持。即使修B13也不重算现有cache。
- F-api：B11（图cache identity）、B12（BF16转换）各一批；固定输入/FP32保持，触发例修正。
- F-numeric：B4 support mask、R1 loss、R3稳定方式分别需要批准和数学预期，不是P批次的前置优化。B4若批准，正support不受pad影响、空行zero directional update后仍正常apply_update，confidence和方向平均规则不变；不能顺带改solver。
- F-tests：只修非保护区内T1/T3/T4及无效配置测试键，单独review；P0独立checker不依赖其旧PASS。T2属于保护目录，计划中无修改批次。
- R2仅维护weights导出；生产推理恢复/精确resume为新功能，不列入以上实现。

共同停止条件：在§13固定gate内出现未解释数值/梯度/时序差异、NaN/Inf、参数映射遗漏、破坏保留caller/保护路径、触发真实cache重建或需要昂贵训练/依赖升级时停止该批，记录证据；不通过放宽容差、删用例、静默fallback继续。已记录的跨chunk数值敏感性不阻塞不改变相关组织的pure-refactor；新差异不能自动套用该分类。已知bug触发样例与正常等价样例分别判定，不能概括成“所有结果不变”。

## 10. 第二阶段完成边界（历史记录）

本轮做的是当前Git/source/小metadata差异核对与计划修订；没有恢复v15A，没有重新引入feature graph，没有修改生产代码、测试、配置或受保护目录，也没有创建refactor_checks/或训练产物。

最终仅REFACTOR_PLAN.md改写、REFACTOR_AUDIT.md追加。结束时核查148个tracked小型源码/说明/配置文件指纹、Git diff与审计原74699字节前缀；不为了完整性检查批量hash或复制大型结果/cache。第一实施批是P0固定当前v7A可重放基准，之后仍须用户启动实施阶段。

## 11. 第三阶段批次0执行状态（2026-09-15，追加记录）

本节追加于第二阶段历史计划之后。用户仅授权本批基准脚本/合成产物及本节记录，并要求使用可用GPU。当前HEAD仍为`922d1738922e8b94890a54f5e42bbf6551f5ccc0`，执行前tracked/staged diff为空，与本计划核实的v7A未发现新的生产行为差异。未开始后续清理、迁移、训练循环收敛或bug fix；原审计本批未修改。

**状态：基准记录、独立重放和失败留证已完成；P0整体验收尚未通过，不能据此启动后续生产修改。**

- 新增 `scripts/freeze_v7a_baseline.py`，用真实Stage/原CRC共享fit执行一次Adam更新；MouseBrain真实命令仅解析配置，epoch0真实forward保留其dropout RNG消耗，完整MouseBrain main不执行。
- GPU为RTX4090、Torch2.4.0/CUDA12.1，主例FP32且dropout=0.1；保存完整有效配置、原未跟踪JSON全文、参数/schema、prior/图/RNG、输出/loss/梯度/一步参数。fixture采用11/13及11/13/17两/三section，另7×11不完整support；本次尺寸调整仅为小型合成覆盖，不改变模型或真实数据配置。
- `refactor_checks/v7a_gpu_fp32_frozen/fixture.pt`已一次性冻结；record和check独立，完成两次新进程加载检查，未覆盖expected。已有`*.pt`忽略规则未改，当前fixture仅保存在工作区，后续review需保留。
- 真实双向sparse初始化/更新、同coupling两向top-k、固定support/空行/全空异常、scheduler、三section先完成所有update后各一次apply均有动态检查。完整epoch100+/最终200轮导出时序仍属静态核对；BF16/数据/分析/插补/生产保存接口未动态覆盖，应在对应改动前补充。
- 原容差始终为`atol=rtol=1e-5`，索引/键/None/RNG精确，正常用例非有限值失败。第一次独立重放两/三section主例分别有梯度/一步参数超差，第二次主例通过；chunk/checkpoint附加对照仍失败。最大一步参数差约`5.64e-5`，不能挑一次PASS或调整容差宣布稳定等价。
- 额外确定性GPU诊断（同fixture、独立构造模型两次，`CUBLAS_WORKSPACE_CONFIG=:4096:8`及deterministic_algorithms=True）逐位一致，仅支持非确定归约解释，不替代原默认GPU环境认证、不改变生产默认。R1 signed-zero-sum导致NaN的原失败证据单独保留，没有softmax、loss、mask、Sinkhorn或Adam修补。

完整命令、逐类误差、PASS/FAIL/静态/未覆盖边界、失败张量、有效配置和保护核对见 [refactor_checks/REPORT.md](../../refactor_checks/REPORT.md)。150项既有小文件指纹核对保持一致；没有读取真实大数据、执行SPATCH预处理或触碰保护目录的写入。

后续首先处理P0验收边界：是否增加明确标记的确定性GPU独立重放合同，保留默认GPU失败证据；不自动换生产算法或放宽容差。P0满足后，C1仍是可review的下一生产候选，目标为`run_spatch.load_preprocessed_cache/cache_parameters`和suite自动build入口，先用小schema替身验证既有缓存只读与无预处理调用。本批到此停止，没有实施C1或其他批次。

## 12. 批次0补充验证结果（2026-09-15，追加记录）

用户授权继续完成确定性GPU FP32重放、chunk/checkpoint定位及SPATCH BF16独立基准。已执行完毕；**两套固定配置可重放参考已建立，跨chunk检查仍FAIL，未标记全部验收通过，未进入生产重构。**

- HEAD仍为`922d1738922e8b94890a54f5e42bbf6551f5ccc0`，生产tracked/staged diff为空。仅扩展原验证脚本的可选观察参数，新增`scripts/supplement_v7a_baseline.py`、`refactor_checks/p0_supplement/`产物和本节；原审计、旧报告、旧fixture不变。
- 验证进程单独固定CUDA确定性设置，生产默认不变。FP32加载原MouseBrain fixture的同一参数/prior/RNG；SPATCH BF16按真实配置语句与小metadata冻结HE50/RNA50/Protein15输入、K10、BF16训练、四checkpoint/loss-only/cache及8192/2048主chunk，保留无epoch0预检的时序。真实数据、缓存数组与预处理未执行。
- 每种配置都有自己的expected，record/check独立进程，各13项自身重放全部PASS且max_abs/max_rel=0。同chunk、dropout=.1的四checkpoint开关在FP32/BF16两套参考中也逐位一致。原loss-only路径保持，额外阶段/loss观察不关闭checkpoint、不复制数学实现。
- 旧checkpoint-on/chunked项是与历史off/unchunked比较，没有自己的expected；本轮将自身重放和跨配置对照分开。Python float容差规则与tensor统一为atol+rtol×abs(expected)。既有失败主要在tensor，此勘误不撤销旧FAIL；atol/rtol始终1e-5，没有放宽。
- 跨chunk（dropout0，attention0→7、decoder0→5）仍FAIL：FP32一步参数最大差6.5170e-5；BF16一步参数最大差1.9998e-3，后续FP32 final eval embedding最大差9.3021e-3。另行attention-only/decoder-only隔离确认两者均可影响梯度；BF16 decoder第二个Linear输出已出现差异，attention-only可在前向相同的情况下出现反向差异。没有把这些直接认定为数学bug，也没有改Adam、loss、mask、solver或AMP。
- 真实sparse初始化/更新、source=ot的FP32 refresh helper、替换prior后的FP32 forward均独立重放通过；完整100+/200轮原loop和生产文件导出仍静态核对。默认CUDA历史失败、R1历史NaN证据、未覆盖数据/分析/插补接口均保留原边界。

完整命令、环境/有效配置、逐项误差、失败张量及保护核对见 [refactor_checks/p0_supplement/REPORT.md](../../refactor_checks/p0_supplement/REPORT.md)。本轮核对164项既有文件：仅授权验证脚本和计划变化，其余162项（含旧基准产物）一致；新旧fixture SHA保持匹配，未批量扫描/hash大型结果或缓存。

后续建议以固定precision/chunk布局的合同约束纯重构，把改变chunk布局的数值影响单列审查。此建议不等于宣布原跨chunk要求通过，也未自动批准下一生产批次。本轮到此停止。

## 13. Batch 0 正式收口与后续验收合同（2026-09-15，用户决议）

**最终状态：BASELINE FROZEN / ACCEPTED WITH KNOWN NUMERICAL SENSITIVITY。**

用户已明确接受以下证据支持的固定配置行为基准。本节更新验收决策，不修改历史实验事实；§11/§12和原报告中的“未全部通过”保留为当时记录，当前是否阻塞pure-refactor以本节为准。

| 已有证据 | 原始检查结果 | 本次验收分类 |
|---|---|---|
| deterministic CUDA FP32固定配置独立进程重放 | 13/13 PASS；max_abs/max_rel=0 | 固定配置参考接受 |
| SPATCH对应CUDA BF16固定配置独立进程重放 | 13/13 PASS；max_abs/max_rel=0 | 固定配置参考接受 |
| 相同chunk布局、dropout=.1的四类activation checkpoint开关对照 | 两种精度输出、loss、梯度、一步参数及RNG逐位一致 | 接受已有同布局对照；不推断未测开关组合/新分块组织 |
| dropout=0跨chunk布局 | FP32一步参数最大差6.5170e-5；BF16为1.9998e-3；原始结果仍FAIL | KNOWN_NUMERICAL_SENSITIVITY；不作为本节pure-refactor的阻塞项 |

BF16跨chunk更新后的FP32 final eval embedding最大差仍为9.3021e-3。该分类不等于证明具体CUDA kernel原因，不认定为已经修复的数学bug。证据见[补充报告](../../refactor_checks/p0_supplement/REPORT.md)、[FP32独立重放](../../refactor_checks/p0_supplement/fp32_deterministic/replay_report.json)、[BF16独立重放](../../refactor_checks/p0_supplement/spatch_bf16_deterministic/replay_report.json)。[原默认GPU报告](../../refactor_checks/REPORT.md)的非确定性失败、R1 NaN及全部原始记录继续保留。

### 13.1 pure-refactor 的严格 behavioral gate

比较双方必须同时满足：

1. 同precision、同AMP设置和autocast边界；MouseBrain FP32、SPATCH BF16训练/FP32 refresh及final eval分别按各自参考验证。
2. 同chunk layout（含attention、decoder及相关edge batch），同checkpoint设置；仅允许另行使用已经验证过的同布局checkpoint对照，不扩大为任意组合。
3. 同模型有效参数与完整参数映射、同prior/graph/冻结fixture、同section/spot/模态顺序与RNG。
4. 同deterministic CUDA验证环境：CUBLAS_WORKSPACE_CONFIG=:4096:8、deterministic_algorithms=True、cudnn.deterministic=True、cudnn.benchmark=False、matmul TF32=False、CPU/FAISS threads=1，并保留冻结环境/后端记录。

以上CUDA设置仅属于验证进程，**不加入生产训练配置**。数值比较保持atol=rtol=1e-5；shape/dtype/键/None模式、索引、结构及RNG精确比较；输出、分项loss、有效梯度、一步Adam后参数和相关阶段状态均须覆盖，正常用例NaN/Inf仍失败。

不改变attention chunk、decoder chunk、loss reduction、autocast边界或checkpoint分块组织的重构批次，使用上述固定布局gate。实际改变其中任何一项或其他chunk布局/分块算法的工作须独立审查现有cross-chunk FAIL并补针对性证据，不能引用本节自动宣布等价。保持参数值但改变归约/分块执行组织也不属于已验证布局。

### 13.2 使用与未覆盖范围

检查读取既有fixture并使用新的报告名；不执行record、不覆盖expected、旧报告或失败证据。当前check可能因同时报告已知cross-chunk FAIL而返回1：必须分别检查same_configuration及checkpoint结果、cross_configuration错误类别；只有已记录的跨布局差异可归入本次已知敏感性，任何新固定gate失败均不得忽略。本轮没有修改checker或重跑测试。

完整100+/200轮训练与生产文件导出仍只有静态核对；真实缓存加载、九数据集的数据/分析/聚类/指标/绘图、三个插补目录动态接口和性能尚未因本次收口获得动态认证。涉及对应职责的批次先补独立小测试，不通过重算受保护数据补齐覆盖。B4、R1等数值问题仍属独立研究/修复范围。

上述Batch 0收口轮仅更新计划和USER_DECISIONS.md；当时C1仅完成范围准备，未执行，生产代码、验证脚本、历史报告和所有受保护目录不变。后续C1授权与实施状态见§14，不追溯改写原始验证结论。

## 14. 用户范围决定及C1验收（2026-09-15）

用户已明确D1–D8最终范围，详见§5.3及USER_DECISIONS.md §4；D8c工作流暂时保留，后续补具体I/O、caller及是否实际运行SpaMosaic模型的说明，不阻塞本批。

**C1最终状态：PASSED WITH DECLARED COVERAGE LIMITS。** 实际生产修改仅为scripts/run_spatch.py的既有cache加载合同与必要入口/来源说明，以及scripts/run_spatch_v7abc_suite.py的缺cache拒绝逻辑；新验证产物位于refactor_checks/c1_cache_contract_20260915。完整范围见USER_DECISIONS.md §5，详细报告见 [C1_REPORT.md](../../refactor_checks/c1_cache_contract_20260915/C1_REPORT.md)。

cache_parameters完整旧schema的字段和值保持，仅补docstring；loader不把retain_processed作为硬门槛，保留其来源，其余未知或冲突参数拒绝。CLI默认data_dir=None；新增可选--input_identity_manifest复用source_files JSON声明，逐source核对SHA和size而不读取raw。同来源data_dir可作cache来源选择并记录current_raw_files_verified=False，另一位置无身份凭据拒绝。cache_info保存历史来源、当前请求及feature行身份限制；新summary的input_data_path为实际cache，input_data_kind=model_ready_preprocessed_cache。

### 14.1 验证与明确限制

| 检查 | 实际结果 |
|---|---|
| 修改前加载机制 | [old_behavior.json](../../refactor_checks/c1_cache_contract_20260915/old_behavior.json)冻结7项原行为；保留旧整源码hash/raw stat门槛的拒绝记录 |
| 新小型cache/schema合同 | [new_behavior_run03.json](../../refactor_checks/c1_cache_contract_20260915/new_behavior_run03.json) **73/73 PASS**；覆盖扩展前的62/71项报告均保留。预处理/build调用为0，cache写入为0 |
| 固定deterministic CUDA FP32 | [p0_fp32_check.json](../../refactor_checks/c1_cache_contract_20260915/p0_fp32_check.json)：自身配置 **13/13 PASS，差0**；同layout四checkpoint对照差0 |
| 固定SPATCH BF16 | [p0_bf16_check.json](../../refactor_checks/c1_cache_contract_20260915/p0_bf16_check.json)：自身配置 **13/13 PASS，差0**；同layout四checkpoint对照差0 |
| 原cross-chunk差异 | **仍FAIL**，完整跨布局对照对象与历史记录精确一致。FP32一步参数最大差6.5170228e-5；BF16一步参数0.00199983548，更新后FP32 final eval embedding 0.00930213928。两次check退出码1均来自该已知跨布局FAIL，按§13归KNOWN_NUMERICAL_SENSITIVITY |

小替身调用真实loader，动态覆盖main到Stage构造前的sentinel；未读取百万spot真实SPATCH数组，未运行100+/200轮训练，最终生产summary/export仅静态核对。输入身份声明只与历史source_files比较，不验证当前raw文件；schema=1缺feature数组内barcode绑定的证据限制保持，不把合法旧cache判失效或补写schema。

本轮未实施D1–D8退休、dense/单向删除、注册模块精简、训练/分析合并或数值bug fix；model/data_preprocessing.py未改。既有SPATCH cache和所有受保护目录不改、不重算，P0 fixture/expected、脚本和原始PASS/FAIL/NaN证据保留；保护记录见 [protection_check.json](../../refactor_checks/c1_cache_contract_20260915/protection_check.json)。Batch 0接受状态及§13固定pure-refactor gate不变；本次C1通过不扩展为九数据集完整训练/分析/导出的动态认证。

### 14.2 C1收尾时提出的下一批P1（当时未执行，现状见§15）

P1仅将保留入口显式迁为双向candidate-sparse UOT：generic训练入口、MouseBrain、CRC共享链及其SPATCH/MISAR/Human Embryo等保留调用者，以及仍保留的smoke/suite命令。逐入口核对初始化、刷新helper和缺prior行为；双向算法中的正反检索、索引转换和top-k helper保留，不把FAISS/blockwise误作dense路径。

完整九数据集支持、Human Embryo单模态、H&E/UNI输入、各数据集AMP/chunk/checkpoint/超参数、当前SPATCH只读cache、三插补接口及历史产物合同保持。先补对应的小接口/配置测试，再对当前已是双向sparse的路径使用§13固定P0 gate；旧dense/独立单向入口的迁移按已批准模式变化单独记录，不能冒称旧算法等价。

P2删除dense/独立单向生产链及专属配置，必须晚于P1全部保留调用者迁完和对应验证，不在P1先删；D1–D8其他退休也不夹带。C1阶段在验收后停止；随后用户另行授权的P1记录如下。

## 15. P1保留入口显式双向candidate-sparse迁移结果（2026-09-15，P1阶段记录）

**P1已完成本批迁移和验收，具备进入P2的调用者迁移前提；本轮到此停止，未执行P2。** 详细入口表、实际参数/函数、动态与静态边界、命令及剩余引用见 [P1_REPORT.md](../../refactor_checks/p1_entry_migration_20260915/P1_REPORT.md)。

起点HEAD仍为922d173；已有tracked修改仅C1的run_spatch.py与run_spatch_v7abc_suite.py，原差异保存于 [existing_c1.patch](../../refactor_checks/p1_entry_migration_20260915/existing_c1.patch)，本批未改这两文件。本批新增生产修改7文件：train_stage_model.py、run_stage_model.py、run_mousebrain_v2.py、run_crc_stereocite.py、run_misar_seq.py、run_human_embryo_rna_only.py、validate_checkpoint_ot_attention.py。没有修改model/、预处理、分析入口或包装/suite文件。

保留入口启用跨section OT时，实际执行Stage.initialize_candidate_sparse_ot_prior(bidirectional=True) → Stage forward(bidirectional_ot_attention=True) → Stage.update_candidate_sparse_ot_prior(bidirectional=True)。MouseBrain/CRC helper不再分发dense，MISAR及四包装共享CRC链，SPATCH保持C1后内部明确的双向模式。Human Embryo保持RNA-only和disable_uot；generic及合法无OT验证不强制初始化prior。显式dense或有效单向请求在入口拒绝，旧参数整体清理留P2；原dense/单向默认入口的迁移是算法模式变化，不以旧dense数值构造等价参考。

| 本批检查 | 结果与边界 |
|---|---|
| 入口/包装/九数据集helper与suite命令 | [entry_checks_run02.json](../../refactor_checks/p1_entry_migration_20260915/entry_checks_run02.json) 78/78 PASS；真实parser、exec后的包装函数和真实sparse helper；两/三section、两向prior/局部索引/共享coupling/至少一次refresh。采用tiny CPU/blockwise参数，不冒充各数据集真实训练 |
| generic/smoke | [generic_smoke_run04.json](../../refactor_checks/p1_entry_migration_20260915/generic_smoke_run04.json) 16/16 PASS；真实Stage/Adam单步和受控刷新、原config值对照、单模态/noOT。旧smoke字符串断言问题保留，不宣称完整旧suite通过 |
| checkpoint validator | [validator_entry_checks.json](../../refactor_checks/p1_entry_migration_20260915/validator_entry_checks.json) PASS；原双向case与新唯一双向case的state/output/loss/梯度一致；合法无OT的RNA/Protein/HE及多模态不初始化prior |
| C1回归 | [c1_regression_run01.json](../../refactor_checks/p1_entry_migration_20260915/c1_regression_run01.json) 原脚本字节相同的73/73 PASS；真实loader、入口spy，预处理/build和cache写入0；真实百万spot cache未动态读取 |
| deterministic CUDA固定P0 | [p0_gate.json](../../refactor_checks/p1_entry_migration_20260915/p0_gate.json)：FP32、SPATCH BF16各13/13 PASS，所有max_abs/max_rel=0；同布局四checkpoint对照差0 |
| cross-chunk | 仍FAIL；两种精度的完整cross_configuration对象与冻结replay一致，FP32一步参数6.517022848e-5、BF16为0.001999835484。两check原始退出码均1，只由该已知失败产生，未改checker/expected/容差 |

§13固定gate未扩展；deterministic CUDA只用于验证。pre/post图、scale参数、有效loss/AMP/chunk/checkpoint、拓扑cost、source=ot的v7A命令、MouseBrain epoch0 RNG和step后单遍刷新不变。通用入口新增检索参数沿Stage既有默认；各数据集原K、epochs、FAISS参数、precision和导出不统一。B1/B2未重构，B10解析前I/O由测试隔离，旧断言/数值bug未修。

P2剩余为model内部dense初始化/更新与隐式fallback、独立单向solver/attention分支和专属配置/测试残留；保留双向正反检索、索引/top-k及FAISS/blockwise。Stage当前else还承载合法无OT直通，P2不能整块删除。历史v4/v5/v6壳并非dense专属，B/C壳、context/fused/metacell等按后续各批处理；D8c继续暂不删除。没有未迁移的已识别保留生产入口，P2仍须在删除时复核其具体imports/文本调用/无OT及保护接口。

本批未读真实大数组、未预处理/build、未启动真实训练/完整suite或写历史输出。保护证据见 [protection_check.json](../../refactor_checks/p1_entry_migration_20260915/protection_check.json)：限定清单SHA、C1原字节与P0原证据保持，加上小测试I/O guard/spy；不声称批量哈希过所有受保护目录。100+/200轮完整调度、生产导出、九数据集数据/分析/指标/绘图和插补运行接口仍未动态覆盖。

## 16. P2实际退休结果（2026-09-15）

**P2完成，当前生产OT只保留双向candidate-sparse及合法noOT。** 详细删除清单、净代码规模、最终调用链、回归命令和边界见 [P2_REPORT.md](../../refactor_checks/p2_ot_retirement_20260915/P2_REPORT.md)。从已验收P1工作树继续，原9个tracked修改保留为起点；本批20生产文件新增64/删除1259行，净删1195行，没有提交或回退。

实际删除linkage_construction.py全文件、Stage dense init/update及隐式fallback、独立单向三个sparse主函数和五helper、dense-only utils cost/normalization、Stage方向选择/单向遍历、六个专属配置及CLI/调用参数/P_dense空占位。保留入口与v7A suite已去旧flags；历史v4–v6命令文字不改、自然被新CLI拒绝。没有保留生产compat shim。epsilon_init/update因已被generic/smoke sparse真实消费保留.08/.05不同值；QC中的双向固定事实标签不是模式开关。

双向17个保留函数源码不变；direction prior仅删除None占位；Stage同步更新主体与构造/初始化AST一致。RNA+Protein小例148个state key/注册参数、62个有效梯度参数、Adam组名/顺序、初始化及forward/backward RNG精确同；这是特定小例数量，MouseBrain/SPATCH配置另由冻结P0 strict load/梯度/一步参数验证。self_linear完全保留。

| 验证 | 最终结果 |
|---|---|
| P2真实小例 | [p2_contract_run04.json](../../refactor_checks/p2_ot_retirement_20260915/p2_contract_run04.json) 19/19 PASS；两/三section、FAISS IVF/Flat/blockwise、同coupling行列topk、局部索引、RNA-only、缺prior明确失败、noOT/state |
| P1入口/generic回归 | [p1_entry_regression_run01.json](../../refactor_checks/p2_ot_retirement_20260915/p1_entry_regression_run01.json) 70/70；[generic_smoke_run01.json](../../refactor_checks/p2_ot_retirement_20260915/generic_smoke_run01.json) 15/15。仅迁旧模式测试为API缺席/CLI拒绝，九数据集及优化/刷新/单模态覆盖保留 |
| C1原小缓存合同 | [c1_regression_run01.json](../../refactor_checks/p2_ot_retirement_20260915/c1_regression_run01.json) 73/73 PASS；原checker字节相同，预处理/build及cache写入0，真实大cache未读 |
| FP32/BF16冻结check | [p0_gate.json](../../refactor_checks/p2_ot_retirement_20260915/p0_gate.json)：各13/13及同layout checkpoint对照差0；原cross_chunk完整对象一致且仍FAIL，两次原退出码1 |
| compile/import与引用 | 84个Python内存compile、35个核心/runner/analysis模块import通过；限定引用扫描0 executable blocker，无KEEP或历史suite真实Python依赖未处理 |

P0/C1/P1旧证据不改。P2验证副本只迁移退休调用参数；原fixture每profile40个P_dense=None在内存显式投影，断言值None、其余容器类型/键序及所有tensor/state/RNG/标量叶子仍原对象，actual无此字段。原比较函数、strict load、finite、1e-5容差、索引/RNG和13项布局不变；不是record或新expected。详细适配diff见P2报告。

noOT明确直通且保留空字典/已有prior对象；不消费prior、不造虚拟coupling。多section启用OT但缺正反prior报错；单section无相邻边自然直通。删除旧单向/空direction分发带来的两种过时消息是明确非数值变化，小测试逐字符串核对，其他字段/数值保持。旧uot=False但attention=True且已有prior仍会消费prior的组合不作为合法noOT数值参考，P2明确禁用时直通。

残余分类见 [REFERENCE_CLASSIFICATION.md](../../refactor_checks/p2_ot_retirement_20260915/REFERENCE_CLASSIFICATION.md)：9个历史launcher旧flags/summary字符串、历史结果显示标签与文档、原P0 checker旧接口、保留QC与共享方向构件均区分；未以字符串命中为由修改历史结果或保留旧core。初版扫描glob范围错误误读150个历史小JSON/Markdown及另5个小文本，已记录并修正；无大数组读取/保护范围写入。

保护证据见 [protection_check.json](../../refactor_checks/p2_ot_retirement_20260915/protection_check.json)：限定153文件SHA及保护目录根inode/mtime、测试I/O guard与C1参考/冻结fixture指纹；不声称递归核验了所有历史产物。真实九数据集/100–200轮/百万spot cache、生产导出/分析/插补、FAISS GPU检索和性能未动态验证。B4/R1/R3、D1–D5、B10及F-config/P3等均未处理。

本轮到P2结束。下一批建议为既定F-config中的最小B1/B2优先级问题，先列冲突输入与真实消费者/resolved值再独立修改；当前不执行。


## 17. F-config-A：MouseBrain B1/B2（2026-09-16）

**F-config-A通过并停止；这是配置correctness修复，不是冲突输入上的pure-refactor等价承诺。** 用户已验收P2后授权本批，仅改 `scripts/run_mousebrain_v2.py`，相对P2新增158/删除83行、净增75。原C1/P1/P2工作树保持，详细字段、旧例、测试与边界见 [REPORT.md](../../refactor_checks/fconfig_a_20260916/REPORT.md)。

解析顺序：base defaults（含MouseBrain原epochs5/max_iter100）→既有JSON model块→输入JSON顶层training→显式CLI。model/helper override默认None，布尔False仍是显式值。候选/FAISS/stabilizer统一从resolved传给真实init/refresh helper；没有通用配置框架或training/config.py。数据预处理前解析一次，输出epochs目录与实际训练同源；train/dry-run新增resolved_config，包含模型初始配置副本、runner参数、最终输出目录、实际FP32/AMP/chunk/checkpoint及epoch0后参数设备/dtype。

模型默认scale仍1；当前v7A真实suite命令得到scale.5、epochs200/devicecuda、K5、candidate200/topk10、epsilon.05/tau1/iter100、sourceot/interval20，与P2既有model叶子/runner/helper值及P0关键冻结值相同。source选择仍沿原CLI，Stage context-source另行记录；未同步fused以顺手修复B3，fused/final、context gate、self_path_mode、metacell和B8字段保留。模型/优化器/AMP/刷新/epoch0 RNG代码未改。

| 检查 | 结果 |
|---|---|
| 真实parser/builder/resolver与边界spy | [config_matrix_run03.json](../../refactor_checks/fconfig_a_20260916/config_matrix_run03.json) 55/55 PASS；6项指定冲突、六种布尔组合、17类其它override、真实helper传参、输出目录/resolved记录、实际v7A命令 |
| P2/P1/generic | 分别19/19、70/70、15/15 PASS；旧smoke文字断言仍单列 |
| C1 | 73/73 PASS，预处理/build/cache写入0；真实大cache未读 |
| FP32/BF16固定参考 | [p0_gate.json](../../refactor_checks/fconfig_a_20260916/p0_gate.json)：各13/13及同layout checkpoint差0；两个完整cross_configuration与原replay一致，cross-chunk仍FAIL，原退出码均1 |
| 保护/编译/diff | [protection_check.json](../../refactor_checks/fconfig_a_20260916/protection_check.json)：限定204项指纹、保护目录根信息、内存compile与git diff检查；不冒充全历史产物递归校验 |

矩阵run01唯一失败来自CUDA availability spy与真实Adam检查相互影响，改为明确AdamSpy后通过；run02/03为新报告，run03补充真实执行记录断言，原证据保留。没有根据该测试问题修改生产数学。GPU checker逐字沿用P2已验收适配，仅check，不record或覆盖expected；deterministic CUDA仅验证使用。

未运行真实MouseBrain/九数据集预处理或训练、100/200轮、生产保存/分析/插补。其他runner未统一；generic smoke后置默认策略及MouseBrain B3只记录为后续问题，Stage既有设备fallback也不改。建议下一批明确B8单独授权/字段消费者与拒绝策略；本批不进入B8、P3a、D4或任何其它批次。


## 18. D4：退休 fused/final 动态刷新 source（2026-09-16）

**D4完成并通过本批验证，B3：RESOLVED BY RETIREMENT OF UNSUPPORTED SOURCE PATH。** 仅执行用户授权的D4；详细删除、入口表、测试、残余分类与命令见 [D4 REPORT.md](../../refactor_checks/d4_ot_refresh_20260916/REPORT.md)。本批13个生产文件新增69/删除160行，净删91；HEAD与已验收C1/P1/P2/F-config-A未提交修改保持。

Stage.prepare_ot_prior_refresh不再接收source，固定ot_embeddings.detach，并对同一semantic输入计算self-excluded spatial context。Stage及sparse update删除candidate_source参数；metadata固定ot。删除uot.dynamic_refresh_source/update_from_final_embedding及五parser的dynamic_candidate_source，保留suite不再传该flag；旧CLI自然拒绝，旧JSON两source键在Stage边界一次明确拒绝。没有兼容alias/静默改算法，也未清理B8其它字段。

v7A仍为.8 semantic+.2 context cost；topology enabled/weight、candidate/FAISS/blockwise/UOT/interval保留。epoch100起每20轮，optimizer之后eval/no_grad单遍旧prior forward再替换；最后刷新轮的最终eval用新prior。fused/ot/final阶段输出、Fusion、GraphSAGE、普通gate与final导出均保持。额外context gate/alpha反传的共享pooling继续保留，未执行D3。两共享validator只迁移source相关部分，B10过时断言不修。

九数据集保留入口全部核对：MouseBrain独立helper；CRC共享fit及MISAR/SPATCH/Embryo；Lymph Node/Spleen真实exec CRC包装；Thymus/Simulation真实exec MISAR再import CRC；generic及合法noOT、suite命令与smoke保留。未发现KEEP依赖退休source，文本替换锚点仍有效。

| 验证 | 结果 |
|---|---|
| D4针对性 | [d4_contract_run01.json](../../refactor_checks/d4_ot_refresh_20260916/d4_contract_run01.json) 7/7 PASS：两/三section、RNA-only、双向prior、ot与context同源、实际.8/.2cost、三阶段保留、拒绝退休source；真实fit仅执行99/100/101/120四个标签并核对更新顺序 |
| F-config-A/P2/P1/generic/C1 | 分别55/55、19/19、70/70、15/15、73/73 PASS；旧smoke文字断言仍单列；P2 state/有效参数/optimizer/RNG保持；C1预处理/build/cache写入0 |
| FP32/BF16固定参考 | [p0_gate.json](../../refactor_checks/d4_ot_refresh_20260916/p0_gate.json)：各13/13及同layout checkpoint差0；完整cross_configuration与原replay一致，跨chunk仍FAIL，原退出码均1 |
| 核心源码 | [core_retirement_check.json](../../refactor_checks/d4_ot_refresh_20260916/core_retirement_check.json)：17sparse函数不变，动态update除source metadata专门化外AST相同，Stage forward除fused刷新context条件外相同；pooling/scheduler数学不变 |
| 引用/保护 | 100个限定Python源码核对无KEEP执行阻塞；16插补源码无source依赖；[protection_check.json](../../refactor_checks/d4_ot_refresh_20260916/protection_check.json)限定243项指纹和目录根信息，不冒充全历史产物递归核验 |

P0检查副本只在内存投影旧fixture两个source配置键，分别断言旧值ot/False，其余所有叶对象和整个expected对象不变；沿用P2的P_dense=None投影。未record、覆盖expected、改容差或生产CUDA设置。F-config/P1首次副本适配错误报告保留，只修正新检查的记录路径/退役CLI token。

残余11个历史launcher为旧CLI/标签，7个analysis为历史metadata读取，不是可执行refresh选择；原冻结验证脚本、历史文档/结果与旧证据保持。Stage退役键拒绝字符串、固定ot事实标签及三个embedding阶段仍合法存在。没有真实九数据集训练、100/200轮、百万spot数组、预处理/缓存重建、历史输出/插补写入。

本批到D4停止；D3、D5、D2、B8、P3a、训练循环重构和任何其它数值修复均未执行。


## 19. D3：额外 attention context gate 退休（2026-09-16）

**D3完成并停止。** 详细删项、普通gate与topology两条调用链、状态合同、验证迁移与限制见 [D3 REPORT.md](../../refactor_checks/d3_context_gate_20260916/REPORT.md)。从已验收D4工作树继续，13个生产文件新增50/删除824行、净删774（git numstat），含整个benchmark_microenvironment_fullspot.py删除；已有修改和所有旧证据保留。

OTGuidedAttention删除context_gate_enabled/context_consistency_backprop_to_alpha参数属性、compute_context_reliability、513维分支及source_context/target_context输入。普通gate仍拼接source/message/差/积，经原MLP+sigmoid，保留confidence乘法及message/residual/dropout/norm。正常softmax alpha和反传不删除。Stage删除额外gate pooling/metadata/空context_embeddings输出；三项gate专属配置在单点明确拒绝，CLI自然拒绝旧开关。五runner/保留suite的该实验记录与override已移除，F-config-A其它优先级不变。

OT refresh仍为ot_embeddings.detach→同源self-excluded pooling/normalize→.8 semantic+.2 context cost→双向candidate-sparse UOT→替换prior。共享pooling和Stage context helper完整函数源码保持，prepare/update/scheduler AST不变；sparse_uot/faiss_candidate_search/loss字节不变。context_eps从未控制当前ot刷新（原refresh使用delta），本批没有替换其数学。fused/ot/final阶段、普通gate、topology参数、precision/chunk/checkpoint/optimizer及刷新时序保持，D2 self_linear完全不动。

关闭额外gate时没有额外预注册nn.Module；旧普通gate Linear已是4×dim。因此不需要参数映射、strict=False或遗留空模块。已验收D4源码与当前版本在两/三section、dropout.1、固定chunks、四checkpoint全关/全开逐名称对照：148个state key和optimizer参数、requires_grad、62个有效梯度参数、初始化/forward/backward RNG、loss/输出/一步Adam状态全部精确一致。四个gate参数有非零梯度；confidence=.5/0分别精确使update减半/清零；中间section两向update仍平均后只apply一次。

| 验证 | 结果 |
|---|---|
| D3针对性 | [d3_contract_run01.json](../../refactor_checks/d3_context_gate_20260916/d3_contract_run01.json) 8/8 PASS，固定状态、ordinary gate/confidence/alpha、同源context与实际.8/.2cost、退休拒绝与共享helper |
| D4/F-config-A/P2/P1/generic/C1 | 7/7、55/55、19/19、70/70、15/15、73/73 PASS；旧gate布尔用例迁为拒绝，旧smoke文字断言仍单列 |
| 小型validator | microenvironment保留pooling/孤立点/双向init/ot context flow；experiment-C保留普通gate/checkpoint/refresh；两个小CPU CLI完整PASS；大型benchmark不运行且已删除 |
| FP32/BF16冻结 | [p0_gate.json](../../refactor_checks/d3_context_gate_20260916/p0_gate.json) 各13/13及同layout四checkpoint差0；完整cross_configuration与原replay一致，cross-chunk仍FAIL、原退出码均1 |
| 核心/引用/保护 | [core_retirement_check.json](../../refactor_checks/d3_context_gate_20260916/core_retirement_check.json) 完整普通gate专门化AST对照；99份Python定向引用无KEEP阻塞；[protection_check.json](../../refactor_checks/d3_context_gate_20260916/protection_check.json)限定298项指纹/保护目录根信息及diff/compile |

P0只check：沿用P2/D4投影，再断言并移除三个旧gate配置False/False/1e-8和18个FP32、8个BF16空gate输出字典。全部其它叶对象保持identity，未删除真实refresh context或任何tensor；未record、覆盖expected、改容差/NaN规则或生产CUDA设置。P2首轮检查副本遗漏空输出迁移的18/19失败保留，只在副本补严格空字典断言/投影后通过。

专属benchmark无KEEP消费者；7个历史launcher为旧CLI/标签，7个analysis为历史metadata读取，原冻结checker/历史报告/结果不改。Stage残余键只用于拒绝；当前alpha和context_embedding_dict属于保留计算，不能按关键词清理。16份保护插补源码无D3依赖，包装替换锚点保持。

没有真实九数据集/100或200轮/百万spot cache/预处理/build/生产保存/分析或插补运行。保护证据非全量历史产物哈希。D5、D2、B8、P3a、训练迁移及B4/R1/R3等其它修复未执行，本批停止。


## 20. D5：本项目metacell生成/训练输入/还原退休（2026-09-16）

**D5完成并停止。** 详细文件、调用者分类、命令和边界见 [D5 REPORT.md](../../refactor_checks/d5_metacell_20260916/REPORT.md)。从用户验收D3的工作树继续；HEAD、staged及既有未提交修改保持，起点与D3最终指纹无新差异。11个生产文件新增8/删除107行，净删99；不含文档与独立检查产物。

实际删除data_preprocessing中的metacell_construction_optimized、construct_metacell_data_dict、reconstruct_metacell_to_original及其聚合/还原mapping代码；三个预处理API删除metacell参数，默认配置删除该项。generic/run_preprocessing转交现有config，在多模态预处理入口一次拒绝顶层或嵌套true；MouseBrain直接路径在读取数据前同样拒绝。旧false可作无行为历史输入读取，不新增alias/stub。CRC/MISAR/SPATCH调用去掉false；Embryo新summary删除metacell_used占位；D8c脚本仅移除五处false调用参数，不退休该工作流。

整个三个预处理模块AST在仅去D5代码/加入拒绝后相同，full-spot共享数学不变；B13 target_sum和B14 section命名已在测试中明确断言保持，B10不处理。H&E/UNI文件和所有Stage/attention/OT/solver/loss数学文件逐字节不变，D2 self_linear未动。九数据集没有必须使用本项目生成/还原的依赖；现有4个源码包装入口加载与锚点保持。

SPATCH cache_parameters/load_preprocessed_cache AST不变，schema=1的metacell=false继续作为历史预处理身份读取；manifest不改写/升级/重算。两个外部COSIE比较器源码不改，真实读取函数用5-spot替身验证值/dtype/barcode/顺序不变，不调用项目生成算法。历史产物、mapping、普通spot metadata和局部索引都保留。

| 验证 | 结果 |
|---|---|
| D5数据准备/cache/拒绝 | [d5_contract_run01.json](../../refactor_checks/d5_metacell_20260916/d5_contract_run01.json) 28/28 PASS；15组旧/新full-spot准备结构逐项相同，含两section、多模态、RNA-only、memory_efficient/retain_processed、Harmony开关与dry-run |
| 入口/外部读取 | [d5_callers_readers_run01.json](../../refactor_checks/d5_metacell_20260916/d5_callers_readers_run01.json) 9/9 PASS；generic/预处理/MouseBrain true请求先拒绝，SPATCH/CRC COSIE结果只读 |
| D3/D4/F-config-A/P2/P1/generic/C1 | 8/8、7/7、55/55、19/19、70/70、15/15、73/73 PASS；C1无预处理/build/cache写入 |
| FP32/BF16冻结 | [p0_gate.json](../../refactor_checks/d5_metacell_20260916/p0_gate.json) 各13/13及同layout checkpoint差0；完整cross_configuration与原replay相同，跨chunk仍FAIL，原退出码均1 |
| 核心/引用/保护 | AST对照、139份定向源码/config核对通过，无KEEP执行阻塞；[protection_check.json](../../refactor_checks/d5_metacell_20260916/protection_check.json)限定375项指纹/保护目录根信息和diff/compile |

P0副本逐字沿用D3适配，D5无新增expected投影；只check、不record。CPU汇总编排的键名错误原记录保留，仅修正编排且复用已经PASS的D3报告；没有数值失败或测试断言调整。旧generic smoke文字断言限制保持。原P0/C1/P1/P2/F-config-A/D4/D3证据及失败不改。

小数据准备使用真实adapter/拼接/对齐/切分，但替换preprocess_adata和Harmony边界；未执行真实normalize/PCA/Harmony、真实九数据集训练、百万spot缓存、生产分析/导出或插补。没有对未标注来源的tensor bundle推断是否来自外部聚合；没有声称真实大数据已动态验收。残余metacell仅为显式拒绝、缓存false身份、外部结果路径/标签及历史记录。D2、B8、P3a、目录/训练迁移和其它bug fix均未执行。


## 21. D2：GraphSAGE额外self-path退休（2026-09-16）

**D2完成；APPROVED_SCHEMA_RETIREMENT下，同有效参数状态数值等价。** 详细删除、初始化记录、schema映射、回归与限制见 [D2 REPORT.md](../../refactor_checks/d2_self_path_20260916/REPORT.md)。从用户验收D5的当前工作树继续，起点与D5最终指纹相同；HEAD/staged及已有未提交修改保持。4个生产文件新增15/删除51行，净删36。

WeightedResidualGraphSAGE删除legacy/no_adj_self、self_path_mode参数/enum/属性、两个实例的self_linear注册及三个专属diagnostics；Stage图调用固定include_self_loop=True，默认model配置去mode字段，MouseBrain两处新summary不再写mode。旧Python/CLI自然拒绝，旧JSON mode字段在Stage边界明确拒绝。当前没有独立G0 launcher/CLI可删，不清理其它suite/analysis/B8字段。

生产修改前捕获两/三section真实Stage的完整state/梯度None模式/optimizer组/Linear初始化顺序及RNG。只删除graphsage.self_linear.weight、post_ot_graphsage.self_linear.weight，当前各float32[128,128]；它们旧forward从不调用、梯度None、无Adam状态。强制其forward抛错并扰动权重仍不影响旧计算。D2小例schema与optimizer参数148→146，有效梯度参数62个保持；P0三模态schema162→160。其余key/shape/dtype/requires_grad逐名称保持。

**同seed构造不再保持原初始化。** 原顺序fusion→pre self/neigh→post self/neigh→attention→观测encoder/decoder；删除两次随机Linear初始化后，构造RNG和小例34个保留张量初值改变。没有保留空模块、补随机数或加入兼容层。按old[name]→new[name]逐名映射、strict=True加载，再恢复同一forward RNG后，两/三section、四checkpoint全关/全开，latent/fused/pre/ot/final/loss/所有有效梯度/一步Adam状态和运行RNG逐项相同。不能将此写成原完整state_dict或同seed初值不变。

唯一保留公式仍为weighted邻居聚合→neigh_linear→原input-dtype零加法+neighbor+bias→activation→dropout→x+scale*message→norm。保留零加法及其顺序以保持autocast语义。pre/post独立参数，pre scale1；post仍普通可调，v7A .5；.75动态证明确实生效，scale0仍执行norm。utils空间图构建文件逐字节不变，k+1/self权重/reverse/max去重/row normalization/edge顺序保持，缓存/非缓存每spot自环及所有edge/weight精确一致。OT self-excluded context、solver、attention/loss/AMP/chunk/optimizer不改。

| 验证 | 结果 |
|---|---|
| D2专项 | [d2_contract_run02.json](../../refactor_checks/d2_self_path_20260916/d2_contract_run02.json) 9/9 PASS；run01的8/8保留，run02新增未调用self_linear证明，没有数值失败 |
| D5/D3/D4 | 28+9、8/8、7/7 PASS |
| F-config-A/P2/P1/generic/C1 | 55/55、19/19、70/70、15/15、73/73 PASS；C1无预处理/build/cache写入 |
| 固定FP32/BF16 | [p0_gate.json](../../refactor_checks/d2_self_path_20260916/p0_gate.json) 各13/13差0，同layout checkpoint差0，标记APPROVED_SCHEMA_RETIREMENT；完整cross_configuration与原replay一致，原跨chunk FAIL/退出1保留 |
| 源码/引用/保护 | 四文件整体AST仅D2差异；139份小源码/config、99份Python compile、11个import通过，无KEEP阻塞；[protection_check.json](../../refactor_checks/d2_self_path_20260916/protection_check.json)限定445项指纹与保护目录根核对 |

P0只check、不record。每profile新增17条明确schema投影：旧no_self_linear配置、两个权重及schema、六组各两个None梯度。其余所有叶对象保持identity，strict逐名加载；不改有效tensor/expected/RNG/容差。原checker已在forward前恢复fixture RNG，无需伪造旧constructor。原fixture、expected、旧报告/失败证据保持。CPU旧schema对照同样按名称映射optimizer状态，旧diagnostics先断言no_self_linear/0/0再投影；旧generic smoke文字断言限制保持。

残余为Stage拒绝旧字段、5个analysis文件7处历史标签和历史文档/结果/验证证据；analysis不改。完整九数据集保留，无KEEP路径依赖legacy/no_adj_self。未运行真实数据/完整训练/预处理/百万spot cache/分析导出或插补，保护核对不是全量历史数组哈希。D2到此停止；未执行B8、P3a、D1壳清理、训练迁移及B4/R1/R3等其它修复。


## 22. F-config-B / B8：配置消费者收敛（2026-09-16）

**B8完成；这是配置接口变化，不新增模型计算。** 从用户验收D2后的实际工作树继续，仅改7个生产文件，新增90/删除41行，净增49。起点、增量和保护见 [B8报告](../../refactor_checks/fconfig_b_20260916/REPORT.md)。原各批未提交修改保持，无提交或回退。

[模型消费者矩阵](../../refactor_checks/fconfig_b_20260916/model_consumer_matrix.md)逐项覆盖原88个默认字段：65 LIVE保留、23 NO-OP删除并拒绝。[入口/预处理/数据矩阵](../../refactor_checks/fconfig_b_20260916/RUNNER_CONSUMER_MATRIX.md)有329项上下文记录，覆盖九数据集、generic/smoke及active suite。没有用途未决的已审计字段。

删除graph的两个假图开关，encoder.type/residual，contrastive六项假算法/权重开关，fusion.mode/input_dim，graphsage.num_layers/use_distance_weight，uot五项假初始化/momentum/cost开关，reconstruction.loss和loss三个假附加loss开关。另删4个未传给真实算法的预处理选择字段和CRC --dry_run、独立smoke入口 --smoke_test。真实generic smoke、默认dry forward、H&E+mask→UNI、已有HE输入和预处理算法均保持。active数据JSON/scale presets原本无这些NO-OP，逐字节不变。

拒绝逻辑只处理已核实字段，不建立全局schema或兼容框架。generic中12个只被MouseBrain消费的model JSON字段明确拒绝；generic的4个错层预处理字段、MouseBrain的8个错层预处理字段明确报正确位置，未改变优先级。数据身份/路径/输出/协议metadata保留。既往P2/D4/D3/D5/D2退休字段不恢复；SPATCH schema=1的metacell=false身份仍读取。

v7A真实命令、完整resolved runner参数及实际prior helper参数与D2起点一致；模型config只减少23个NO-OP，其余逐项一致。scale=.5及普通.75、blockwise、topology .2、sparse核心参数、AMP/chunk/checkpoint和optimizer保留。两/三section的146个state key、shape/dtype/requires_grad、有效梯度、optimizer参数组、同seed初始化/RNG与D2后精确相同；本批未新增schema退休。对更早P0继续使用D2两个self_linear的APPROVED_SCHEMA_RETIREMENT，不恢复历史随机消耗。

| 验证 | 结果 |
|---|---|
| B8模型 / 入口专项 | 34/34、16/16组PASS；NO-OP旧值扰动无效果，新输入拒绝；LIVE代表字段实测 |
| D2 / D5 / D3 / D4 | 9/9、28+9、8/8、7/7 PASS |
| F-config-A / P2 / P1 / generic / C1 | 55/55、19/19、70/70、15/15、73/73 PASS；CPU总293项 |
| FP32 / BF16固定配置 | 各13/13 PASS，max_abs=max_rel=0；同layout checkpoint差0 |
| cross-chunk | 原FAIL完整对象保持，KNOWN_NUMERICAL_SENSITIVITY |
| 静态范围/compile/引用 | core 88项、99个Python内存compile通过；99小源码+4active JSON残余分类，无KEEP执行阻塞 |

新checker只在内存严格断言并投影23个NO-OP config字段和6个P2退休输入字段，不改数值expected、有效state、RNG、容差或原fixture。FP32首次因旧fixture六字段被明确拒绝而停止的证据保留；定位后新报告run02通过，不能把该失败归cross-chunk。模型专项首轮缺少测试调用参数的失败也保留。详见 [回归报告](../../refactor_checks/fconfig_b_20260916/REGRESSION_REPORT.md)。

保护核对限定526个预记录hash及目录根stat，结合I/O guard；不是历史大数组全量哈希。无真实九数据集训练、预处理/UNI提取、百万spot加载、生产分析/导出/插补或性能验证。所有历史结果与旧证据不改。B8达到本批完成条件，具备进入P3a的配置清理前提；本轮到此停止，P3a、B10/B13/B14、P4/P5及训练合并均未执行。


## 23. P3a：提取通用配置解析（2026-09-16）

**P3a完成；保留入口resolved/模型/训练行为纯提取等价。** 从用户已验收B8工作树继续，新增training/__init__.py与training/config.py，修改Stage及8个runner；11个生产文件新增210/删除144行，净增66。详细范围、增量、测试见 [P3a报告](../../refactor_checks/p3a_config_20260916/REPORT.md)。

training.config使用普通dict提供load_json、merge_config、apply_explicit_overrides、resolve_model_config、resolve_option_values、config_source_layers、serialize_config。沿用默认→model/preset→dataset/input→显式CLI，None不覆盖，False/0/空值保留。只提取机制，未统一dataset默认或新增JSON支持。model.configure默认/拒绝函数保持；model不import training或scripts。

Mouse/generic的JSON/递归merge/显式覆盖迁出，Mouse helper默认/JSON/CLI选择共用小函数。CRC/MISAR/SPATCH现有参数映射使用公共resolver；Embryo/smoke共用纯默认获取，其特例留原处。预处理CLI仅迁JSON reader。各dataset默认、AMP/chunk/checkpoint、K、候选数、类型转换、ATAC字面量、section与文本锚点不移动。HLN/Spleen/Thymus/Simulation四包装文件逐字节不变。

Stage提供config时只deepcopy，不再重复merge默认；16条当前KEEP入口/validator均传完整resolved，无partial依赖。config=None保留纯模型默认，数值/RNG同旧；直接partial库调用现在应先经公共resolver，不再隐式补齐。未增加兼容层、state/schema变化或随机初始化补偿。

Mouse已有resolved_config仅新增source_layers来源metadata；旧effective字段、目录suffix、文件名和保存格式保持。不同runner已有NumPy/tensor summary序列化保持，未硬套新JSON工具；其它入口完整resolved summary留P5。generic smoke后置特例保持，不夹带新precedence修复。

| 验证 | 结果 |
|---|---|
| P3a解析等价 | 23/23组PASS；九数据集+generic默认/显式20case，类型/key顺序/值一致；原55矩阵旧新各55/55且旧details一致 |
| Stage专项 | 11/11 PASS；2/3section、RNA-only、noOT、.75 scale、实际四checkpoint，state/初始化RNG/输出/梯度/Adam/refresh精确 |
| B8至C1全CPU回归 | 343/343首次PASS：B8 34+16、D2 9、D5 28+9、D3 8、D4 7、A 55、P2 19、P1 70、generic 15、C1 73 |
| FP32/BF16冻结 | 各13/13 PASS，max_abs=max_rel=0；同layout checkpoint差0；无新增投影 |
| cross-chunk | 原FAIL完整对象保持，KNOWN_NUMERICAL_SENSITIVITY |
| 静态范围/compile | 92项PASS、101个Python内存compile；fit/导出/模型数学范围保持 |

P0 checker及B8/P2配置投影逐字节沿用已验收B8，D2 APPROVED_SCHEMA_RETIREMENT保持。只check、不record，原fixture/expected/报告不改。保护核对718个起点状态及目录根stat，另记录两个新增training文件，非历史大数组全量核验。未运行真实训练/预处理/大cache/分析或插补。

当前bidirectional validator残留graph.k_neighbors赋值没有影响实际K，单列记录而不顺手修；B10 parse前I/O仍存在。P3a具备进入下一批F-data(B10)的前提，但本轮停止，不自动执行B10、P4a、P3b、P4b/P5或analysis重构。


## 24. F-data(B10)：Simulation / Thymus parse前I/O（2026-09-16）

**B10完成，仅修audit调用时机与路径消费。** 从验收P3a工作树继续，只改scripts/run_simulation.py与scripts/run_mouse_thymus.py，新增17/删除16行、净增1行。旧流程main→全局路径audit/mkdir/read/write→load/argv注入→parse→run；新流程load/原argv注入→parse一次→audit(Path(args.data_dir),Path(args.output_dir))→原run。详细diff与证据见 [B10报告](../../refactor_checks/b10_parse_io_20260916/REPORT.md)。

两个audit不再引用DATA_DIR/OUTPUT_DIR全局路径，其余body在路径改名后AST一致。原CLI默认路径（含result_v4）仍保留，不作为P3b/P5提前迁移；正常显式NEW_DATA/NEW_OUTPUT会同时进入audit与pipeline，help/解析失败不会进入audit。共享validate_args原位置、audit全量SECTIONS策略、直接异常和文件关闭边界均保持。

main/help专项17/17通过，包括六个独立Python进程真实__main__的help/非法CLI，退出0/2且数据/audit/运行输出副作用0。旧问题只用spy记录。audit/adapter专项12/12通过：两audit的JSON/CSV逐字节一致，18次读取来自显式data_dir，六个真实adapter合成输入的shape/dtype/value/顺序一致。Simulation spfac/nsfac/spatial_domain及五section，Thymus RNA obs坐标/17-marker轴及四section不变；没有真实预处理。

P3a解析23/23与B8至C1 343/343均首次通过，合计366组；10入口默认/显式20组resolved值与验收P3a报告逐项一致。FP32/BF16固定各13/13、同layout checkpoint差0；原cross-chunk完整对象及FAIL保持。P0 checker/既有投影逐字节沿用P3a，D2 APPROVED_SCHEMA_RETIREMENT保持，B10无新增投影或expected变化。

静态107项与101个Python内存compile通过；replace+exec、argv注入和模型/训练/预处理源码除上述两点外保持。保护核对限定908个起点状态及目录根stat，没有真实数据/训练/大cache/分析/插补运行，不声称全量历史数组哈希。B10具备进入下一数据批次的前提；本轮停止，不执行B13/B14、B11/B12、B4/R1/R3、P4a/P3b/P4b/P5或analysis问题。

## 25. P4a：四个训练入口的数据适配普通函数（2026-09-16）

**P4a完成，纯组织重构。** 从已验收B10工作树继续，先实际执行旧四包装及CRC/MISAR小型合成路径并冻结12例，再修改生产代码。HLN/Spleen改为普通CRC调用，Thymus/Simulation改为普通MISAR调用；四者均不再读取/替换基础源码、exec、注入sys.argv或修改基础模块globals。parser仅增加必要显式argv/defaults/数据身份接口；dataset默认值仍留原入口，复用training/config.py，没有进入P3b。

新data_io.paired承载CRC读取/切片及HLN/Spleen基因/marker纯函数，data_io.misar承载MISAR读取/共享轴函数。Thymus/Simulation原adapt/audit/坐标/domain函数不变。唯一必要UMAP消费者只改import与数据函数调用。CRC fit及全部训练helper原位；模型/训练包、预处理数学、optimizer/backward/AMP/refresh/final eval不变。10个生产文件新增632/删除584行，净增48行；函数清单、所有必要调用者及增量见 [P4a报告](../../refactor_checks/p4a_adapters_20260916/REPORT.md)。

old/new专项12/12 PASS，完整数据值、dtype、shape、section/spot/modality/gene/marker顺序、obs/var/spatial/metadata、预处理参数、resolved及训练handoff一致；异常9/9、六入口完整后半pipeline导出6/6，共351个小文件一致。Simulation保留spfac/nsfac/spatial_domain，Thymus保留RNA obs坐标和17 marker轴。小型PCA/训练替身不冒充真实预处理或训练；HLN/Spleen真实RNA50/Protein20合同保持，专项小轴的替身输出宽度单列说明。

轻量回归406/406首次PASS：B10 17+12、P3a 23+11、B8 34+16、D2 9、D5 28+9、D3 8、D4 7、A55、P2 19、P1 70、generic15、C1 73。另10入口默认/显式20份resolved与B10精确一致、无投影。固定GPU FP32/BF16各13/13、误差0、同layout checkpoint差0；原cross-chunk FAIL及完整对象保持。P0 checker/已有投影逐字节沿用B10，无新投影/record/expected变化。

静态范围68项、CRC完整pipeline三种专门化、MISAR54项均PASS。保护核对限定1024项起点状态与34个目录根stat，并核对新data_io文件；没有读取真实大数组或声称全量历史产物hash。旧专项冻结630项证据保持。B13/B14未修，P3b/P4b/P5、analysis重构及其它已知问题未执行；没有真实大规模训练/预处理/缓存重建。本批结束，下一结构依赖为P3b，但须用户另行明确授权。

## 26. P3b：显式dataset defaults与单次配置解析（2026-09-16）

**P3b完成，纯配置组织重构。** 从已验收P4a工作树继续，先冻结九数据集27组真实parser/resolver/helper配置、12组CRC/MISAR系完整数据交接和6组MouseBrain/HESTA/SPATCH真实main边界，再修改生产。九入口各有get_dataset_defaults；MouseBrain另有明确MODEL/UOT_HELPER/TRAINING常量，SPATCH原十个后置运行设置迁为TRAINING_DEFAULTS。使用training/config.py的简单parse_dataset_args与describe_run_config；不建新配置体系。

基础defaults < preset/model < dataset/input < 显式CLI保持。None表示缺省，False/0保留；BooleanOptionalAction使True默认可被显式False覆盖，help/usage新增反向选项单列记录。CRC/MISAR裸调试默认与suite训练参数继续区分，不强行统一BF16/epochs/K/资源。CRC/SPATCH原模型块提普通builder，CRC/MISAR/HESTA/SPATCH在数据准备前生成一份配置，后续不重复merge。SPATCH保持原cache身份判断，未把训练设置加入cache身份。MouseBrain既有单次JSON/resolver路径保留。

10个生产文件新增742/删除323行，净增419行；九入口新调用链与全部默认来源见 [DEFAULTS_AUDIT.md](../../refactor_checks/p3b_defaults_20260916/DEFAULTS_AUDIT.md)，完整记录见 [P3b报告](../../refactor_checks/p3b_defaults_20260916/REPORT.md)。训练summary只增resolved记录，单独验证其与真实消费对象一致；原输出字段保持，没有统一artifacts。训练loop仍原位，模型、data_io、通用preprocessing与analysis源码不变。

配置27/27逐字段/type/key/order一致；另外20份配置与P4a记录精确一致，无投影。实际main/适配交接18/18、False/零值/默认隔离19项、异常9/9、导出6/6（351个小文件，新增resolved字段单列）PASS。保留轻量回归406/406通过；B10 help展示和旧合成路径的checker迁移过程及失败日志完整保留，不称首次全部通过。静态610项PASS。

冻结CUDA FP32/BF16各13/13、max_abs=max_rel=0，同layout checkpoint差0；原cross-chunk FAIL与完整对象保持。P0核心checker/已有投影字节不变，无record/expected/容差/schema/RNG更改。保护核对1470项起点状态与34个保护根，并保留本批508项旧专项冻结证据；不声称递归核验真实大数组。B13/B14、P4b/P5、analysis重构及其它独立修复均未执行；未运行真实大规模训练、真实预处理或cache重建。完成P3b后停止，P4b须另行明确授权。


## 27. P4b-1：仅提取CRC共享训练runtime（2026-09-16）

**P4b-1完成，纯组织迁移。** 从用户已验收P3b工作树继续，先冻结6组真实旧loop事件与4组CUDA一步数值，再移动生产源码。新增`training/fit.py`，保留`train_small_crc_model`原名；run_one_forward、AMP/scaler、prior初始化/更新编排、监控/清理及其少量helper共13个定义逐字不变。CRC/MISAR/SPATCH/Human Embryo四runner仅调整import，所有剩余函数体不变。5个生产文件新增473/删除436行，净增37；详细增量见 [P4b-1报告](../../refactor_checks/p4b1_fit_20260916/REPORT.md)。

八数据集实际绑定同一training.fit函数：CRC、MISAR、SPATCH、Human Embryo直接调用，HLN/Spleen经CRC，Thymus/Simulation经MISAR。CLI/defaults/resolved、data_io、Stage构造、dataset-specific数据与保存仍在原处；fit不parse、不merge、不读cache、不识别dataset。MouseBrain和generic整文件字节不变，未创建统一trainer/回调或artifacts层。

人工epoch99/100/101/119/120/199/200验证6组FP32/BF16/FP16事件顺序全等，包括forward先于zero_grad、backward/step、history、清理、refresh前后prior版本、final eval、scaler/monitor。199末轮不补刷；200先刷新再final eval。只有训练forward进入autocast，BF16不启用scaler，refresh/final在外。四组CUDA实际一步（FP32两section、BF16固定chunk/checkpoint三section、RNA-only七section启用/禁用UOT）loss、OT/final embedding、全部有效梯度按名称/shape/dtype/None/finite/value、Adam后参数/state及RNG完全一致。RNA-only crossview为0，原UOT开关保持。

P3b配置27/27、18交接、19默认/False/零值、20份P4a accepted配置全等；P4a 12适配、9异常、6导出（351文件，含既有resolved记录）通过。既有CPU406/406 run01通过，含B10/C1/generic；316项精确搬动/范围核对。固定CUDA FP32/BF16各13/13、max_abs=max_rel=0，同layout checkpoint差0；cross-chunk原FAIL及完整对象保持，仍为KNOWN_NUMERICAL_SENSITIVITY。P0数值checker只调整静态源码定位，原投影/expected/fixture/tolerance未改。

保护核对1916项起点状态、34个保护根、508项P3b冻结证据及13项本批旧fit基准；HEAD不变、staged为空。测试搭建中的缺少optimizer参数、JSON tuple/list表示差异均记录并保留日志，未改生产数学或旧基准。B13/B14及其它独立修复未执行；MouseBrain/generic未迁，P4b-2/P4b-3/P5/analysis未执行。没有真实大规模训练、真实预处理或cache重建。本批立即停止，下一小批须另行明确授权。


## 28. P4b-2：仅接入MouseBrain训练循环（2026-09-16）

**P4b-2完成，纯组织重构。** 从已验收P4b-1工作树继续，生产仅改run_mousebrain_v2.py与training/fit.py，新增66/删除137行、净删71。MouseBrain保留全部parse/resolve、数据准备、Stage构造、initial prior、原位置epoch0 train-mode/no_grad完整dry forward和保存；正式epoch循环仅由共享fit承担。原样迁lambda_for_epoch，三个prior helper在9组真实resolved profile/backend下比较后复用；json_safe因ndarray语义不同而不合并。

fit新增lambda_contrast_schedule、clear_step_state、record_elapsed_time、allow_empty_epochs四个明确参数，默认维持CRC。MouseBrain传已解析schedule/False/False/True，分别保留真实schedule、step后梯度/输出、无耗时history和零轮数空训练能力；没有dataset分支或callback。当前PyTorch实测zero_grad()与显式set_to_none=True等价；MouseBrain仍只在forward之后、backward之前清一次。共享FP32 loss.float为同一tensor，不改梯度；AMP none、不启用scaler，epoch0仍独立消耗dropout RNG。

修改前封存17项旧MouseBrain基准。六组实际runner事件（final199/200、对应schedule、dry、epochs0）old/new全等，指定99/100/101/119/120/199/200标签；refresh100/120/200在step后eval/no_grad旧prior单遍forward，再替换prior，199不补刷、200再final eval。三组实际CUDA一步（两section、三section、两section schedule）全阶段输出/分项loss、source/target输入grad、全部160个参数按名称的grad/None/finite/shape/dtype、Adam后参数/state和RNG完全一致；epoch0参数/prior不变且无optimizer，RNG前进与epoch1首forward一致。三section双向更新先计算，中间计数2、每section一次apply。真实小artifact文件56项加事件替身35项均一致，保存schema未改。

P4b-1六事件/四CUDA一步、八条调用关系继续保持，generic整文件不变。P3b27配置/18交接/19默认及20份P4a accepted配置、P4a12适配/9异常/6导出、既有CPU406全部通过。140项范围核对通过；scope最初AST字符串括号匹配失败改为直接节点判断，失败证据保留，未更改生产代码。FP32/BF16固定各13/13、max_abs=max_rel=0，同layout checkpoint差0；cross-chunk原FAIL完整对象保持。

保护核对2362项起点状态、34个目录根、508项P3b证据、13项P4b-1旧fit证据及17项本批旧基准，HEAD不变、staged为空。模型/data_io/config/analysis及其它runner不改。详情见 [P4b-2报告](../../refactor_checks/p4b2_mousebrain_20260916/REPORT.md)。generic尚未迁移；P4b-3/P5/B13/B14及其它独立修复未执行；无真实MouseBrain/九数据集完整训练、真实预处理或cache重建。本批完成后停止。

## 29. P4b-3：仅接入generic trainer（2026-09-16）

**P4b-3完成，P4b三个小批次整体完成。** 从已验收P4b-2继续，仅改scripts/train_stage_model.py（+38/-79）与training/fit.py（+68/-18），合计+106/-97。generic本地训练epoch循环退休，数据/config/Stage/初始prior和保存仍原位。shared fit的唯一epoch主体为普通iter_fit_model；既有train_small_crc_model签名及最终tuple返回保持，九数据集原调用者不改。generic仅遍历该函数给出的保存结果，无第二套训练循环、callback或生命周期框架。

新增三个明确参数record_loss_weights、refresh_ot、yield_every，默认True/True/0保持原共享行为；generic传False/已resolved uot.enabled/原save_every，保留四字段history、noOT不刷新和refresh后周期weights。复用P4b-2 clear_step_state=False、record_elapsed_time=False、allow_empty_epochs=True；无epoch0，FP32保持。初始epsilon_init和刷新epsilon_update分别消费，不统一；processed_data_dict原本不传Stage，仍传None。bundle、preprocess_config、真实synthetic、普通epochs0以及原smoke显式0→3、weights文件名/state/config/history/section_order/input_dims、final保存全部保持，无resume扩展。

修改前封存58个旧证据/输入文件。12个真实Stage/Adam CPU用例（含bundle顺序、预处理边界替身、synthetic、noOT/单section、零epoch、无输出、人工199/200）及3个CUDA一步用例old/new逐项一致。所有阶段、loss、输入及按名称的参数grad/None/finite/shape/dtype、Adam/state、RNG及每次保存内容比较；只对三项非负finite墙钟耗时归一，保留原键和值证据，调用kwargs按键比较。模型/config/history/section/modality顺序和数值均不放宽。原fixture建立时两次缩维不完整的失败留证，另存v3配置后在生产修改前成功冻结，没有修改生产模型或既有expected。

refresh用人工99/100/101/119/120/199/200，只在100/120/200、step后eval/no_grad完整一遍旧prior forward，再从ot更新prior；保存随后发生。final199不补刷，final200用新prior；noOT 200无初始化/刷新。没有实际跑200轮。P4b-1六事件/四CUDA、P4b-2六事件/三CUDA、十条generic/九dataset绑定、P3b27配置/18交接/19默认/20accepted、P4a12适配/9异常/6导出（351小文件）、CPU406均通过。123项scope通过，选取默认参数后共享epoch主体AST与P4b-2一致。FP32/BF16各13/13、max_abs=max_rel=0，同layout checkpoint差0，cross-chunk完整原FAIL保持。

保护核对2918项起点、34个根目录stat、508个P3b旧文件、13个P4b-1旧fit文件、17个P4b-2旧MouseBrain文件及58个本批旧证据文件。HEAD不变、staged为空；模型、数据、analysis与所有其它runner不改。详情见 [P4b-3报告](../../refactor_checks/p4b3_generic_20260916/REPORT.md)。P5/B13/B14/analysis及其它独立修复未执行；无真实九数据集长期训练、真实预处理或cache重建。本批完成后停止。

## 30. P5：九数据集入口与batch收敛（2026-09-16）

按本批授权完成P5。新增scripts/run_experiments.py，以九个现有canonical runner为唯一单dataset CLI；显式dataset列表、output root、普通override和JSON argv透传，按顺序每dataset独立subprocess。batch不读取数据/config、不构造模型、不训练、不analysis；P3b defaults继续由runner负责。Human Embryo仅补现有普通scale的可选CLI，未传时原parsed/default/model字典完整不变。训练数学、shared fit、MouseBrain epoch0、generic输入/weights、P4a adapter和原artifact格式均保持。

先审计15个suite/launcher并冻结九默认/命令与fit交接，再实现。v7B/v7C无消费者且训练能力被显式scale覆盖，两个壳退休；v7A/SPATCH仍有P0/C1/B8/P1等真实checker消费者而暂留，其它历史混合壳逐一记录KEEP/OUT OF P5，未为删除它们修改analysis。生产4文件+150/-202，净删52。CRC/MISAR默认epochs0准备模式、MouseBrain config和SPATCH已有cache显式要求保持，不在batch补隐藏默认。

75项命令/完整配置对照、九old/new canonical handoff、九batch到真实main/fit sentinel handoff、额外九scale handoff通过；12项batch行为及12项CLI边界通过。P4b-1六事件/四CUDA、P4b-2六事件/三CUDA、P4b-3十二CPU/三CUDA、CPU406、P3b/P4a/B10/C1全通过；117项scope通过。FP32/BF16各13/13且差0，same-layout checkpoint差0，cross-chunk完整原FAIL保持，无record/expected/容差/数值投影更改。

起点与P4b-3的2918项状态无差异，本批3175项旧文件/34根stat/508项P3b/13旧fit/17旧MouseBrain/58旧generic/14本批旧证据保持。HEAD未变，staged为空。详见[P5报告](../../refactor_checks/p5_entry_20260916/REPORT.md)、[launcher审计](../../refactor_checks/p5_entry_20260916/LAUNCHER_AUDIT.md)、[保护核对](../../refactor_checks/p5_entry_20260916/protection_check.json)。P5整体完成；F-analysis未执行，P6未执行，B13未执行，B14未执行，P7/P8未执行，analysis未重构；未运行真实九数据集长期训练或真实预处理，未重建cache。完成本批后立即停止。

## 31. P5后续：MouseBrain入口命名调整（2026-09-16）

用户另行授权单文件命名整理，run_mousebrain_v2.py改为run_mousebrain.py；新batch、五个保留suite、根freeze脚本仅更新import和路径。旧源码保存在新的验证目录，历史记录仍用当时文件名；无转发壳、无数学或数据/配置/训练修改。14项命名/CLI/配置及真实fit handoff检查、6组MouseBrain事件/epoch0 RNG/参数/保存回归通过；保护3405项旧状态、34根及17项旧冻结文件保持。报告见[命名补充](../../refactor_checks/p5_mousebrain_name_20260916/REPORT.md)。本次未重跑GPU gate，P5冻结数值证据不改；不扩展到P8其它整理、P6/analysis或B13/B14。

## 32. F-analysis(B9)：新分析缓存来源绑定（2026-09-16）

用户单独授权并按范围完成B9。真实缓存消费者先审计并以tiny输入冻结旧错误命中，随后增加analysis/cache.py普通identity/manifest/reuse函数与必要显式output参数。新cache绑定实际embedding SHA256、shape/dtype、section/spot顺序、truth、有效run/resolved字段、各自protocol与implementation；同源命中，不匹配/无manifest明确拒绝并要求新独立输出，不覆盖历史。SPATCH requested K=5/8/10/12/14/16/20、compare不同K集合以及九数据集原分析协议均保持。没有P6 loader/metric/plot函数迁移或算法变化。

V12与实际entry验证通过；old/new合成指标、标签、scaler逐项一致，补图PNG字节一致。C1 73/73，P5命令与九数据集+generic shared fit引用检查通过；训练/model/config/adapter/launcher源码不变。冻结FP32/BF16各13/13且差0，同layout checkpoint差0，cross-chunk原FAIL/KNOWN_NUMERICAL_SENSITIVITY完整保持，只check不record。3434项起点状态及34个保护根核对见本批protection report，不声称递归hash真实大数组。

详情见[B9报告](../../refactor_checks/fanalysis_b9_20260916/REPORT.md)、[真实消费者审计](../../refactor_checks/fanalysis_b9_20260916/B9_CACHE_AUDIT.md)、[回归](../../refactor_checks/fanalysis_b9_20260916/REGRESSION_REPORT.md)。历史metrics不重算或补provenance；历史复制batch CSV及补图输入label的历史来源限制显式保留。P6/P7/P8、B13/B14未执行；SPATCH preprocessing cache未修改，未运行真实九数据集完整analysis。完成后停止，不把本批完成当成下一批授权。


## 33. P6a：analysis loader / 数据读取依赖（2026-09-16）

本批按用户单独授权仅迁analysis loader/数据读取依赖。新增analysis/loaders.py、analysis/inputs.py及data_io/adapted.py，普通函数显式接收run/data/section输入；九dataset均有明确路径。现有MethodData由原消费者保留，未建registry/Loader类或统一evaluate CLI。Thymus/Simulation六个纯data helper与ADT轴迁入data_io，runner按原名import；MISAR只迁SECTION_INFO字典。所有剩余runner函数AST不变，训练/model/fit/config和P4a有效行为保持。D8b/D8c跨方法/SpaMosaic风格UMAP继续KEEP，数据helper不再import training runner。

修改前冻结30组真实tiny loader输出与265项输入/输出证据；v6/v7A样式及旧通用格式的embedding值/dtype、section/spot/barcode、metadata/truth和来源identity逐项一致。22项requested/旧analysis壳路径handoff、9个Thymus/Simulation section的纯helper对照、D8b/D8c import与数据handoff通过。B9原43项和补充13项回归、17项实际旧loader身份命中/输入变化miss通过，implementation identity不因纯位置移动而改变。P4a12例、P3b27例、C1 73/73、P5命令/九dataset+generic shared-fit引用、B10 help无I/O通过；401个分析函数AST及223个算法调用AST保持。

FP32/BF16固定check各13/13且差0，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture覆盖、容差/layout/AMP或生产CUDA设置变化。4072项起点状态、93个源快照、265项旧loader证据和34个保护根核对通过；HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，所有成果仍在未提交工作树。

详见[P6a报告](../../refactor_checks/p6a_loaders_20260916/REPORT.md)、[依赖审计](../../refactor_checks/p6a_loaders_20260916/ANALYSIS_DEPENDENCY_AUDIT.md)、[回归](../../refactor_checks/p6a_loaders_20260916/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/p6a_loaders_20260916/protection_check.json)。旧完整analysis exec/argv壳中metric/export/plot职责明确暂留P6b/P6c；新loader已独立。没有全量迁移跨方法混合workflow、删除旧main或统一输出布局。历史format只做tiny替身验证，不读取真实大数组，不补造不存在的barcode/truth。

P6a完成，P6整体未完成。P6b未执行、P6c未执行、P7/P8未执行、B13未执行、B14未执行；clustering/metrics算法未改、plotting未重构，历史metrics未重算、SPATCH preprocessing cache未改，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成本批后停止；下一批必须另行授权。


## 34. P6b：标准化、聚类、指标与joint-per-section协议（2026-09-16）

本批仅执行P6b。新增analysis/clustering.py、metrics.py、protocols.py、sampling.py、batch_metrics.py；九dataset标准化/聚类/监督与internal指标/采样和batch数学使用普通函数，保留原dataset与requested/compare差异。23个生产文件+1626/-1355；training/model/data_io、P6a loader及canonical训练runner未修改。

完整读取supplementary参考后正式定义joint / joint_per_section / independent。joint_per_section只切已有joint assignments，零重新scaler/cluster；实际指标只有ARI/NMI，原协议没有跨section平均，aggregation=none。MouseBrain仅RegionLoupe/annotations、MISAR五truth、SPATCH仅cell_type_common、Simulation spatial_domain；CRC/HLN/Spleen/Thymus无truth不制造指标。Embryo不擅自增加原参考没有的监督范围。保留参考与requested不同的truth过滤规则。两个原requested入口通过--joint-per-section显式启用，产生逐section结果与completion状态；B9绑定scope/K/section顺序/label/metric/filter/aggregation及implementation，纯函数搬动保留原默认分析identity。

122项old/new数值exact，29项joint-per-section专项，18行/14个supplementary导出文件（仅生成时间不同）、真实requested新scope拟合次数/旧指标/缓存一致性通过；P6a30loader与22条路径、B9 43+13、P4a12、P3b27、C1 73、P5/P4b必要引用回归通过。FP32/BF16 fixed各13/13且差0、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持。只check，无record、expected/fixture/tolerance/layout/AMP修改。4564项旧状态、96源快照、旧tiny基准及34保护根核对无越界差异；HEAD/staged不变。

详见[P6b报告](../../refactor_checks/p6b_protocols_20260916/REPORT.md)、[协议审计及joint-per-section专章](../../refactor_checks/p6b_protocols_20260916/ANALYSIS_PROTOCOL_AUDIT.md)、[回归](../../refactor_checks/p6b_protocols_20260916/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/p6b_protocols_20260916/protection_check.json)。P6b实施完成，P6整体仍未完成；P6c、P7/P8、B13/B14未执行，plotting/UMAP未重构。历史metrics未重算，未运行真实九dataset完整analysis/长期训练、真实预处理或cache重建。完成后停止，下一批必须另行授权。

## 35. P6c：绘图、UMAP与显式evaluate（2026-09-16）

本批按用户单独授权完成P6c：新增明确scripts/evaluate.py和普通analysis.evaluation/requested/comparisons/spatch/embryo/plotting/umap。九dataset显式选择dataset/run-dir/output-dir/protocol/scope，直接消费P6a loader、P6b clustering/metrics/protocol。31个生产文件+5245/-4344；training/model/data_io、九canonical runner、shared fit、P5入口完全未改。

joint / independent / joint_per_section明确分开，后者只切已有joint labels，真实ARI/NMI、valid spot与section字段保持，不重新聚类、不跨section平均；无truth dataset不造值，Embryo不擅自扩参考监督范围。Embryo七section/celltype/developmental/25叶weighted Ward与层次图保留，逐section补图可显式调用。SPATCH requested与compare K独立，B9新输出来源绑定保持。

先审计34个analysis main；退休无生产caller且已完整替代的rerender_result_v6_umap_figures.py、complete_human_embryo_per_section_plots.py。真实suite/aggregate/cross-method/legacy格式与后处理caller仍有的旧壳逐项暂留。D8b跨方法和D8c原始模态准备保持KEEP，二者共用analysis.umap/plotting并写新output root；新evaluate提供已有UMAP坐标重绘，不另创统一九dataset UMAP预处理协议。D8c SPATCH可视化输入缓存隔离到新output，来源变化在预处理前拒绝，不触碰C1缓存。

21个renderer旧/新输入、panel/colors/order/size/dpi逐项相同；九workflow文件集合/CSV字段/JSON字段类型及数值相同；UMAP16组prepared input/参数/显示/重绘对照通过。九真实loader→workflow与九standalone joint_per_section+真实requested共19组，最终绘图入口和cache边界通过。P6b122 exact、joint29、requested scope7、P6a30、B9 43+13、P4a12、P3b27、C1 73及P5/P4b必要引用通过。FP32/BF16各13/13、same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保持，只check。

5341项旧状态、全部源快照/旧tiny基准及34保护根核对，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged为空。所有成果未提交。详见[P6c报告](../../refactor_checks/p6c_plotting_20260916/REPORT.md)、[main审计](../../refactor_checks/p6c_plotting_20260916/ANALYSIS_MAIN_AUDIT.md)、[回归](../../refactor_checks/p6c_plotting_20260916/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p6c_plotting_20260916/protection_check.json)。旧历史metadata/裸label证据不足时保持原能力并明确限制，不补造barcode或来源；未读取真实大数组。

P6a/P6b/P6c组成的P6整体实施完成。P7未执行、P8未执行、B13未执行、B14未执行；未运行真实九dataset完整analysis、长期训练、真实预处理/UMAP大优化或cache重建，未修改历史result/cache。完成后立即停止，没有下一批授权。

## 36. P7：当前调用图与高置信度DEAD清理（2026-09-17）

本批仅执行P7，基于已验收P6真实工作树重新扫描当前AST/import/CLI/string/dynamic/test/protected引用。修改前建立P7_DEAD_CODE_AUDIT；原9个DEAD候选全部仍有定义、无当前消费者，现删除。另删22个无消费者import绑定。7个生产文件+2/-248，净减246行；剩余全部函数/class及顶层AST仅有列明删除，整个Stage类、state/schema/constructor/forward/参数顺序保持，training/shared fit/config/data_io和所有runner字节未改。

processed_data_dict与load_cosie_style_data有真实调用者继续KEEP；旧order/epoch兼容未证明可安全移除而KEEP TEMPORARILY。v7A/SPATCH suite仍有P0/C1等冻结checker消费者；其它混合壳有真实caller或独有调度/历史输出职责，未证完整替代者暂留。本批未新增退休launcher/analysis wrapper。v7B/v7C suite、旧MouseBrain文件及P6c两个已退休main标ALREADY REMOVED，不重复计删。D8b跨方法UMAP、D8c raw modality准备/整合、single modality/noOT、H&E/UNI、FAISS/Flat/blockwise、weights export及三种analysis scope继续KEEP。

P5 batch12（含九真实canonical tiny handoff）、P4a12、P3b27、C1 73、shared fit事件6、MouseBrain事件/RNG6、generic实际tiny Stage/Adam12、P6a30、P6b数值122、joint_per_section29、requested scope7、evaluate19、B9 43及UMAP16均通过。55个模块import、14个子进程help和2359个Python源码compile通过。FP32/BF16固定各13/13 exact，same-layout checkpoint差0；cross-chunk完整原FAIL / KNOWN_NUMERICAL_SENSITIVITY保留，只check不record。P4a首次checker命令参数误用导致parse退出2，修正调用后通过，原错误日志保留。

6635项旧状态、104个生产源快照、旧tiny expected及34保护根核对；无非白名单差异，HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，成果仍未提交。保护检查不声称递归hash真实大数组。详见[P7报告](../../refactor_checks/p7_cleanup_20260917/REPORT.md)、[DEAD审计](../../refactor_checks/p7_cleanup_20260917/P7_DEAD_CODE_AUDIT.md)、[回归](../../refactor_checks/p7_cleanup_20260917/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p7_cleanup_20260917/protection_check.json)。

P7完成。P8未执行，B13未执行、B14未执行，其它独立correctness问题未执行；未改变model数学或analysis协议，未运行真实长期训练/完整analysis/真实预处理，未修改历史result/cache，未重建cache。完成后停止，下一批需要另行授权。

## 37. P8：最终职责归位、当前导航与依赖收尾（2026-09-17）

本批仅执行 P8，先基于已验收 P7 工作树建立 FINAL_STRUCTURE_AUDIT。普通 MOVE：model/data_preprocessing.py→data_io/preprocessing.py（字节相同）、model/multimodal_preprocessing.py→data_io/datasets.py、model/image_preprocessing.py→data_io/image_features.py、scripts/hesta_rna_utils.py→data_io/hesta.py（字节相同）；model/utils.py 按原函数边界拆为 data_io/common.py 与 model/spatial_graph.py。更新真实 import，model 包不再重新导出数据函数，无旧路径 shim；load_cosie_style_data 的真实调用能力仍保留。SPATCH schema=1 cache loader/metadata、dataset artifact 保存和有消费者历史壳按审计留原位，未创建新的配置/trainer/artifact/loader框架。

迁移函数/class/签名及有效可执行 AST 完全相同；唯一函数 body 字符串变化为 SPATCH 未由训练main调用的 cache writer 的源文件诊断路径，既有 manifest 和 C1 identity 不变。Stage 类/constructor/state/schema/参数顺序、preprocessing 数学、training/config.py、training/fit.py、全部 analysis 权威源码不变。44 组图/data/UNI/HESTА/state/RNG 小对照精确一致。九dataset P5 handoff、P4a/P3b/C1、shared fit、MouseBrain epoch0、generic 12小例、P6a/P6b/joint-per-section/evaluate/B9/plot/D8b/D8c通过。FP32/BF16 fixed 各13/13 exact，same-layout checkpoint差0；原cross-chunk FAIL / KNOWN_NUMERICAL_SENSITIVITY完整保留，只check不record。

新增根 README.md 和 scripts/README.md，列出九canonical训练入口、batch、evaluate及69个脚本分类。MouseBrain名称保持run_mousebrain.py；batch用misar/crc，evaluate用misar_seq/crc_stereocite，明确区分。13条README命令经真实parser/help/dry-run/sentinel检查；v7A研究scale=.5显式传入，未把底层/部分裸入口1.0默认改写成.5。joint_per_section只切joint labels，参考ARI/NMI、不聚类、不跨section平均；SPATCH双K协议与Embryo特殊分析保持。model/training/data_io无scripts依赖，analysis无training runner/历史suite依赖；迁移模块无循环依赖和活跃旧路径引用。

起点7443项旧状态、104份生产源码快照、旧fixture/expected/报告/FAIL及34保护根按白名单核对。HEAD仍922d1738922e8b94890a54f5e42bbf6551f5ccc0，staged为空，累计成果未提交。保护核对不声称递归hash真实大数组。旧checker的模块路径仅在新P8副本适配；原报告和expected不改。旧UMAP快照的嵌套import适配过程日志完整保留，不是算法/数值失败。

详见[P8报告](../../refactor_checks/p8_final_20260917/REPORT.md)、[结构审计](../../refactor_checks/p8_final_20260917/FINAL_STRUCTURE_AUDIT.md)、[回归](../../refactor_checks/p8_final_20260917/REGRESSION_REPORT.md)、[保护](../../refactor_checks/p8_final_20260917/protection_check.json)。后续如需check，从本批适配checker复制到另一个新目录；不可重跑record或覆盖本批/既有产物。

P8实施完成，当前主线重构实施整体完成（等待用户审阅本批，不冒称用户已经验收P8）。B13未执行、B14未执行、B11/B12未执行、B4/R1/R3未执行；模型/预处理/analysis科研方法未改。未运行真实长期训练或完整九dataset analysis、未重建cache、未修改历史result/cache。完成后停止，不自动开启任何独立correctness批次。


## 38. S1：额外工程组织收尾（2026-09-17）

S1 按当前工作树审计 69 个 scripts Python 文件，完成用途/CLI/import/subprocess/dynamic/frozen/README/helper 分类。九 canonical runner 原路径 help 全部通过；原样嵌套至 scripts/train 的真实探针全部 import 失败，因此正式 CLI 保持根路径。仅普通移动 scripts/validate_bidirectional_ot_attention.py → tools/validation/validate_bidirectional_ot_attention.py，使用仓库根目录 python -m，去掉原有 6 行 import/bootstrap；全部函数/签名/main AST 相同，不新增 sys.path hack 或兼容 shim。scripts Python 数 69→68，tools/validation 1 个 Python 工具；未创建空 tests/train/comparison/workflows 目录，无 RETIRE。

其余 68 文件保留：正式路径稳定，P1/D3/D4/generic 行为 checker 消费的 validator 留原位，P0 freeze/supplement 和 v7A/SPATCH suite 留原位。supplementary joint-per-section 有 P6/P8 当前 checker import 和历史 labels/export 变体，未退休；HLN A1 外部人工 truth 的五方法流程替代证据不足而 KEEP TEMPORARILY。D8b/D8c 独立工作流保持，旧壳/implementation leak 如实记录，不宣称所有历史脚本都已变薄。

105 生产 Python compile/import、24 正式/工具 help（I/O guard）、13 根 README 命令通过。移动工具 4 组参数/Stage/input/prior/forward/输出/RNG 精确一致，仅 3 种耗时 metadata 字段变化；旧/新原始证据保留。九 dataset handoff/batch12、P4a12、P3b27、C1 73、shared fit6、MouseBrain6、generic synthetic/zero/noOT、P6a30、P6b122、joint29、evaluate19、requested scope7、B9 43、plot15、D8b/D8c UMAP16 通过。4 个保留 validator main 通过；run_stage_model 旧 smoke 文案断言原版/当前同 FAIL，未修、未冒称 PASS。

FP32/BF16 fixed 各13/13 exact，same-layout 差0，cross-chunk 完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保持。只 check，不 record、不改 fixture/expected/tolerance。model/data_io/training/analysis 及 68 个保留 scripts 字节不变；旧状态、保护根和 HEAD/staged 按本批 protection_check 核对，成果未提交。

S1 实施完成，等待用户审阅。B13/B14/B11/B12/B4/R1/R3 未执行；未运行真实长期训练、完整 analysis、真实预处理或 cache rebuild，未修改历史 result/cache。完成后停止，无下一批实施授权。

详见 [S1报告](../../refactor_checks/s1_scripts_20260917/REPORT.md)、[完整脚本审计](../../refactor_checks/s1_scripts_20260917/SCRIPTS_AUDIT.md)、[回归](../../refactor_checks/s1_scripts_20260917/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/s1_scripts_20260917/protection_check.json)。后续 checker 复制到新授权目录并只读旧 expected；不要原地重跑覆盖本批报告。


## 39. S2a：历史训练调度壳成组退休（2026-09-17）

本轮只执行 S2a，按当前引用闭包成组退休 11 个 v4/v5/v6 历史训练 launcher/suite/daemon，生产 +0/-3186 行。历史训练壳 13→2；scripts Python 68→57。v4c→v4b、v4c_large→v4_large 的历史内部引用不再作为保留理由。没有当前正式 caller 或需迁移的共享 helper，不新增 shim；九 canonical runner、run_experiments/evaluate/preprocessing 路径与字节保持。

保留两项 FROZEN_PATH_EXCEPTION：run_result_v7a_suite 的 training_commands 仍为当前 P0 replay 配置来源；run_spatch_v7abc_suite 的 training_command/validate_training 仍由 P0/C1 当前 checker 调用。它们的生产调度能力已可用 batch 表达；本轮不迁移这两个冻结校验依赖，后续 S2c 是否处理须另行授权。其它 v4/v5/v6 普通 helper 没有当前 consumer，旧后台 PID/GPU等待/from-task/训练后自动分析随壳退休，未新增 scheduler。

16 份历史 check_entries.py 的 v6 import 属于旧批次配置投影证据，原样保留，不作为当前回归入口。当前使用 S1 已迁移的 run_experiments→九 runner handoff/config checker，并复制到本轮独立目录 check；不修改历史 checker/fixture/expected。所有 11 个 Git 旧路径可恢复；其中 4 个前期未提交版本与 HEAD 不同，其精确字节已存在于已验收 S1/before_sources，本轮再次留存，不能把 HEAD 当作全部成果提交。

94 个保留生产模块 compile/import、13 个 guarded/direct CLI help、11 个替代 batch dry-run 逐 token 对照通过；九 dataset 实际 tiny handoff/batch12、P3b27、P4a12、C1 73、shared fit6、MouseBrain epoch0/RNG6、generic synthetic/zero/noOT3 通过。analysis/evaluate 和 validators 必要 import 通过；无当前源码残留路径。FP32/BF16 固定各13/13 exact、same-layout差0，cross-chunk完整原 FAIL / KNOWN_NUMERICAL_SENSITIVITY 保留。只 check、不 record、不改数学/协议/默认值。新静态 checker 曾把 batch dry-run 的 JSON list 误当 dict，修正本轮测试读取后通过；生产无变化，诊断留存。

12206 项继承状态、34 个保护根及 HEAD/staged 按本轮白名单核对，保留生产文件字节相同。HEAD 仍 922d1738922e8b94890a54f5e42bbf6551f5ccc0、staged 为空；累计成果未提交。历史结果/大数组没有递归哈希；证据范围以 protection_check 为准。

S2a 实施完成，等待用户审阅；完成后停止。S2b/S2c/S2d 未执行，analysis/supplementary/validation 物理整理未执行；B13/B14/B11/B12/B4/R1/R3 均未执行。未运行真实长期训练、完整 analysis、真实预处理或 cache 重建，未修改历史 result/cache。

详见 [S2a报告](../../refactor_checks/s2a_train_launchers_20260917/REPORT.md)、[launcher审计](../../refactor_checks/s2a_train_launchers_20260917/S2A_TRAIN_LAUNCHER_AUDIT.md)、[回归](../../refactor_checks/s2a_train_launchers_20260917/REGRESSION_REPORT.md)、[保护核对](../../refactor_checks/s2a_train_launchers_20260917/protection_check.json)。后续只在新授权目录使用当前 checker，不覆盖本轮或历史证据。

## S2b 实施状态（2026-09-17）

S2a 已验收后本批仅执行旧 analysis 入口退休：23 文件实际删除，scripts 57→34，analysis 候选35→12。当前单 run 使用 evaluate/P6；Simulation 独有诊断函数归 analysis 并保留薄CLI；只读 persisted joint-label 校验归 analysis/joint_results。11 个跨方法/D8b/D8c/人工A1工作流明确留S2d。当前 checker 解除旧main import，历史证据不改。CRC/SPATCH informational SHA 指权威模块，B9 identity不改。两冻结suite的训练API保持，仅去旧analysis dispatch。详细证据见 [S2b REPORT](../../refactor_checks/s2b_analysis_entries_20260917/REPORT.md)。S2c/S2d及B13/B14/B11/B12/B4/R1/R3未执行。

## S2c/S2d 工程收尾计划更新（2026-09-17，未执行）

按用户新决定：S2c负责当前验证工具非版本化命名及旧comparison/suite引用闭包；S2d负责八个raw-vs-standardized比选CLI成组退休、必要跨方法函数归analysis、保留workflow无版本号命名。不是把旧compare文件整组搬家，也不是取消跨方法科研能力。current CLI去版本号，历史冻结证据原名保留；不改batch metric/UMAP等既有input-space协议。细节、事实定位和验收边界见 [PLAN_UPDATE](../../refactor_checks/s2cd_plan_20260917/PLAN_UPDATE.md)。尚无本次实施动作，仍待分批授权。
