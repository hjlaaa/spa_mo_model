# U系列：公共接口与流程收敛实施计划

日期：2026-09-21。状态：**本轮仅完成U0：基线、实施合同、验证合同及U1首批方案冻结；U1–U8均未开始。** 本文中的新接口、验证和代码迁移均为后续计划，不表示已经执行。

依据：[只读审计REPORT](../../refactor_checks/v7a_core_structure_audit_20260920/REPORT.md)、[讨论摘要](../../refactor_checks/v7a_core_structure_audit_20260920/SUMMARY_FOR_REVIEW.md)及本轮用户指令。旧[REFACTOR_PLAN.md](REFACTOR_PLAN.md)、[USER_DECISIONS.md](USER_DECISIONS.md)保留历史含义，不覆盖、不续写成当前实施状态。检查未发现已有同用途U系列计划，故新增本文。函数行号以下述起始快照为准，后续按函数身份跟踪，不以旧行号作为接口。

**本轮新增的SPATCH规则具有优先级：最终必须同时支持从原始数据准备后训练，以及复用已有预处理结果训练。复用是效率选项，不能成为唯一能力。** 当前代码的cache-only入口只是迁移起点；本要求排入U2e，不在U0/U1处理，也不回写旧审计结论。

## 1．U0基线与状态确认

2026-09-21 08:33（Asia/Shanghai）必要只读核对结果：

| 项目 | 当前状态与实施依据 |
|---|---|
| HEAD | `d52829b769ebd06cae25c4ca78e4a02aba7068f6`；分支`l0-original-20260717`。 |
| 生产范围 | model/data_io/training/analysis/scripts/tools共78个文件，与指定提交的完整Git blob逐一相同；已跟踪工作树及暂存区无差异。没有额外物理生产文件（排除__pycache__），没有未跟踪生产源码。 |
| 已知ignored配置 | MouseBrain输入JSON SHA256仍为`45cf1fd40674a91f2c4e63dea73afa7ab0797cb6224f28613f41afe7bee8156a`，与审计一致；不据此承诺全部真实数据/cache历史归属。 |
| 模型/工程状态 | 采用审计已证明的纯空间图v7A生产链，保留S2d、B13/B14/B12/B11及E1后的Finit。未重复完整全仓架构审计。 |
| 快照措辞 | 当前生产Python是E1的75个相同文件＋model_component的Finit增量，**不是E1原样**；旧922d提交不是累计重构完成态。 |
| Finit锚点 | model/model_component.py SHA256=`32f8ec48e58cfbd27677eace70f69704219904370990d44398612d81b441cf9a`；`:188–191`的额外reset保持原初始化RNG消耗，不删除、不重新排序。 |
| 审计文档锚点 | REPORT SHA256=`06dcc0d69c6b2ddc2114f23cf27391bd8fd95a35b69e1250c6a0e67697f4c3d3`；SUMMARY SHA256=`87f0a5dde570c39bb3a9030a6ab4bb0393738a854ffc25bb8ddc7e92785c2c79`。 |
| 本轮写入 | 仅本文。没有执行生产import、模型构造、forward、训练、预处理、analysis、GPU gate或回退；未改README、配置、测试或历史证据。 |

本次没有发现相对于审计的生产差异。后续每小批以“本起点＋已验收U批次增量”为比较对象：开始前检查实际改动，遇到额外差异先列出归属和影响，不自动回退，也不把前一批合法改动误判为基线污染。

### U0阶段合同

| 要素 | 冻结内容 |
|---|---|
| 依赖/唯一目标 | 依赖既有审计和当前用户要求；唯一目标是形成可直接实施的分批合同。 |
| 允许/禁止 | 只读状态、U1边界与U6候选的必要核实；只新增本文。禁止生产修改、运行验证任务、重做全仓审计及启动U1。 |
| 当前→拟议接口 | 当前实际接口不变；拟议逻辑合同见第3节，U1具体接缝见第4节。 |
| caller/迁移 | 本轮无caller迁移；保留全部正式入口和历史计划。 |
| 验证 | 只读Git/blob、少量源码及文档引用检查；数值/行为/产物测试只制定设计。 |
| 停止条件 | 当前生产差异无法归属、既有同用途计划将被覆盖、关键边界证据不足；记录问题，不擅自扩大工作。 |
| 完成条件 | U0–U8逐阶段边界、U1实施及验证方案、U6首对、SPATCH双输入规则和待决CLI事项写入独立文档；文档交付后停止。 |

## 2．总体路线与全程不变量

目标是稳定数据交接、训练任务和普通分析接口，减少同一流程的多份实现。文件数量、统一CLI数量和目录树不是优先验收指标；本文出现的新函数/类型名用于冻结职责，具体物理文件仅为候选，不为匹配草图强行建文件。

| 阶段 | 唯一成果 | 前置依赖 |
|---|---|---|
| U0 | 实施及验证合同冻结 | 当前审计与状态确认 |
| U1 | HLN/Spleen成对准备试点，形成PreparedDataset | U0 |
| U2a–e | 各输入族逐批接入显式prepare_dataset边界 | U1；各批独立验收后累计 |
| U3 | 公共artifact保存核＋显式导出策略 | U1合同及U2各批输出映射；全范围完成要求U2各族已验收 |
| U4a–e | fit外公共训练任务，按caller族迁移 | 对应U2、U3批次已验收 |
| U5 | 分析输入与已保存标签合同，读取/依赖方向收敛 | U2/U3合同稳定；结合U4产物兼容记录 |
| U6a–b | 同类exact KMeans scope任务共用 | U5；先HLN/Spleen，再评估后续相近路径 |
| U7 | 正式CLI/配置别名/运行方式和文档收口 | U4/U5/U6稳定，CLI事项一次性决定 |
| U8 | 综合兼容验收，关闭迁移债务 | U1–U7各批记录齐全 |

默认按阶段顺序推进；同阶段中一次只迁一个明确caller族。后续某批若只获局部实施范围，不因本文已有后续路线而自动继续。没有为本计划创建任何新的experiments流程。

### 全阶段兼容合同

| 合同 | 必须保持的行为 | 对照方式（均为后续验证，不在U0运行） |
|---|---|---|
| 计算与初始化 | 单Stage纯空间图、no_self_linear、独立pre/post参数、双向candidate-sparse OT、既有刷新时机和输出含义；Finit/RNG次序 | 模型源码范围约束；以当前Finit起点分别比较固定权重计算与同seed初始化，不能互相替代。 |
| 数据身份 | section/spot/modality/feature原顺序、输入dtype和缺失语义 | 比较有序身份、selected indices及校验/警告轨迹，不用shape相同代替身份相同。 |
| 内存和生命周期 | tensor/array引用、原有copy点、mmap方式、backed句柄关闭时点、已有原地元数据修改 | 同进程alias/共享buffer探针及句柄事件；跨进程比较ownership种类，不比较对象地址。不额外加载大metadata。 |
| CLI/配置 | 现有正式CLI迁移期间可用；显式False/0/None、覆盖顺序、数据默认值保持 | 原argv→resolved配置逐字段对照；只记录不改变裸scale=1.0等现状。 |
| 产物 | 原文件名、字段、数组dtype、行列顺序、保存开关和时点 | 新独立目录中的小样本产物对照；普通稳定文件比较字节，压缩容器比较成员及payload，不能忽略真实内容差异。 |
| 运行差异 | dry/epoch0的train/eval状态、noOT、零epoch、周期yield、最终eval、weights/prior范围 | 使用入口行为矩阵，禁止以“统一”名义默默更正。weights导出不升级成精确resume。 |
| 分析协议 | joint/independent；joint_per_section仅切已有joint labels、不重聚类、不平均、不扩标签 | 比较fit调用次数、labels来源、truth mask、scope和指标行。既有空间连续性均值是另一个指标，不能误删。 |
| 分析空间 | batch metrics、UMAP、Simulation及HLN等专项诊断各自的输入空间/dtype/采样 | 按实际caller→space→metric链对照，不用standardized clustering改造其它任务。 |
| 缓存与来源 | B9身份与生成源码provenance分别处理；HESTА、UNI、SPATCH保留专用边界 | 数值/输入协议不变时验证B9 identity；记录新运行的源码路径/SHA变化。不得修改旧manifest让其假装命中。 |
| 冻结证据 | fixture/expected、旧报告、失败记录、历史结果/cache、三gene_imputation目录不动 | 每批生产allowlist及保护记录；新小型接口证据独立存放，不是重录旧基准。 |

每批验证前固定环境、seed、线程与测试输入，并预先声明动态字段白名单：仅新输出根路径、时间/耗时、内存观测、实际生成源码位置/SHA等必要差异可分类比较，不能笼统忽略summary或provenance整块。任何新数值容差必须在看候选结果之前确定；出现差异不靠放宽阈值完成验收。旧已知失败与本批新回归分列，不用一个总PASS覆盖。

## 3．拟议逻辑公开接口

### 3.1 PreparedDataset最小交接合同

统一的是逻辑成员与语义；U1只做最小容器，不强制dataclass框架、不可变深拷贝、统一NumPy/torch类型或新磁盘格式。

| 字段 | 最小含义 | U1取值/所有权与缺失语义 |
|---|---|---|
| section_order | 有序section ID，决定现有OT相邻配对 | 原samples顺序的两元素列表；不排序、不从dataset名字生成。 |
| feature_dict | section→modality→现有模型输入矩阵 | 直接引用现有COSIE返回并按原映射改key；不新转dtype/device，不额外copy/concat。 |
| spatial_loc_dict | 与feature同序坐标 | 原返回坐标引用；保持当前维度/dtype，已有subset_to_memory复制点保留。 |
| processed_data_dict | 现有兼容对象，可为None | U1原对象直传；不把它重新解释为原始counts或Decoder输出。 |
| identity | 分section的有序spot来源、source_rows；输入feature身份及最终特征轴的证据 | 可引用已materialize RNA的obs_names和原indices/slice；大metadata可留引用/按需访问。输入gene/marker身份与PCA/Harmony轴分开，不给派生轴伪造gene名。 |
| truth | 可选标签及适用性/缺失说明 | U1不新增生物truth读取；声明not_loaded/not_provided，不能伪造标签，也不能把“未读取”断言为“原数据不存在”。 |
| provenance | 显式input_kind、adapter/身份策略、输入路径、现有manifest或证据限制 | U1为raw paired输入；dataset_id只描述来源，不触发隐藏行为。没有原始hash/身份就标未证明，不为此读大型矩阵或构造全量DataFrame。 |
| preparation_audit | 准备校验、选择和原有summary所需的轻量记录 | 沿用rna_info、duplicate tables、alignment、gene/marker选择、suffix统计、preview、内部s1/s2映射等；字段来源见4.4。 |

`data_dict`及旧summary仍需访问的AnnData属于**过渡compatibility context**，保持原引用，可随准备结果一起交回caller，但不强制成为所有adapter永久必填公共字段。它和backed资源所有权、导出路径不能混成模型输入。U2逐族补齐逻辑字段时允许证据等级不同，不能为了让类型整齐增加读数、排序或copy。

### 3.2 准备、导出、任务、分析四个边界

| 边界 | 当前公开用法 | 拟议逻辑接口及禁止耦合 |
|---|---|---|
| 准备 | CRC/MISAR pipeline内部准备；datasets.preprocess_multisection_cosie_style；HESTА/runner cache loader；generic bundle | U1为prepare_paired_dataset(显式PairSpec, PreparationOptions, rng, resources, hooks)；U2形成prepare_dataset(显式InputSpec/adapter/input_kind, options, context)→PreparedDataset。不识别文件名/shape来猜dataset。 |
| 保存 | CRC/MouseBrain/generic各保存器和runner内summary组装 | 公共保存核接收明确payload＋ExportOptions/LegacyLayout→路径记录；数据含义由caller提供，不反查另一个runner。 |
| 完整训练任务 | 多个run_*负责Stage/prior→前置步骤→fit→save | 公共任务接收已resolved配置、PreparedDataset、ExecutionPolicy与ExportOptions；fit仍为既有iter_fit_model，不再解析argv。 |
| 分析 | dict/SimpleNamespace/多种MethodData、各scope循环 | AlignedAnalysisInput与load_saved_assignments；普通scope任务接收明确ProtocolSpec。专用格式/科研流程委托adapter或独立workflow。 |

上述名称是接口讨论约定，物理位置在对应实施批次按真实依赖选择；不预建train.py、training/run.py、artifacts.py或固定目录树。

## 4．U1：HLN/Spleen准备合同与读取/身份试点

### 4.1 阶段合同

| 要素 | U1冻结方案 |
|---|---|
| 依赖/唯一目标 | U0；只让HLN/Spleen通过显式成对准备接口取得PreparedDataset，保持后续执行合同。 |
| 允许候选 | data_io.paired的原reader/identity函数；必要的轻量结果/准备实现；CRC pipeline的准备接缝及纯校验/选行helper；HLN/Spleen main中显式注入准备策略；新增本批独立验证材料。 |
| 禁止范围 | 不搬整个run_crc_pipeline进data_io；不动model、training.fit、预处理数学、最终artifact保存实现、parser/defaults、analysis；不统一CRC gene策略；不改变既有warn/raise、错误类型/时点或自动对齐；不提前消除所有runner相互依赖。 |
| 当前→拟议接口 | 当前HLN/Spleen把read_pair、prepare_rna、filter_shared_genes传给CRC；拟议保留该调用及参数兼容，新增仅供内部使用的可选准备委托，选择prepare_paired_dataset返回合同。不是新增CLI选项。 |
| caller/顺序 | 先抽机械helper并保留CRC旧函数委托兼容，再接HLN，后接Spleen；CRC自身默认路径暂走旧准备块，U2a再迁移。MISAR及其族对CRC helper的旧导入暂保留可用。 |
| 验证 | 第4.5节：旧/新准备结果、输入身份、RNG、引用/资源、轻量artifact/异常轨迹、CRC自身与MISAR helper回归。 |
| 停止条件 | 必须改预处理算法/模型/fit或最终保存才能封装；CRC路径被暗改；dtype/alias/随机消耗或失败产物不一致；真实函数关系与此处证据不符。定位记录，不扩范围修复。 |
| 完成条件 | HLN/Spleen准备经公共接口，原CLI/Stage/fit/save位置与行为不变；CRC控制路径回归成立；所有必要字段有来源/限制说明；新增验证通过且历史证据未动。完成即停，不自动进入U2。 |

### 4.2 实际准备边界及保留点

源码证据：HLN `main:81–90`、Spleen `main:40–49`都委托CRC；`scripts/run_crc_stereocite.py::run_crc_pipeline:549–1038`包含准备、模型、训练和保存。必须按职责切开，不能以整个函数作为“数据接口”。

| CRC当前行/操作 | U1准备接口职责 | 仍由runner承担 |
|---|---|---|
| :554–609，参数验证、resolve fallback、np/torch seed、default_rng、输出目录、CudaMemoryMonitor和容器 | 不迁移；接收已确定的options和同一个rng | 所有前置顺序不变；先验证再打开数据。 |
| :611–617，read_pair→登记backed对象→prepare_rna→alignment | 按原循环顺序调用相同函数，成功返回后登记到caller提供的resources容器 | 资源容器创建及现有finally关闭，见下。 |
| :619–654，首section有序gene交集→全局非零过滤→max_shared_genes→suffix审计 | 原样提取到成对准备流程，filter明确传入 | 不改变过滤在spot抽样之前、截断在全局过滤之后的次序。 |
| :655–664，ADT唯一性、集合和顺序比较 | 原校验核；拒绝顺序不一致，不自动重排 | 错误文本/类型兼容。 |
| :666–670，duplicate CSV/gene txt/marker txt | 在原时点发出窄metadata-ready hook，提供原数据 | 现有to_csv/save_list仍在runner调用；不能延迟到预处理成功后再保存。 |
| :672–698，抽样/preview/可选indices保存 | 使用caller的同一rng，保持section调用顺序；返回原slice/array；spots-selected hook | args.save_outputs分支及save_selected_spot_indices、路径返回仍归runner。 |
| :700–724，四次subset_to_memory、gene顺序校验、data_dict | 原次序/对象/copy点；构建相同RNA/Protein/HE/Metabolite列表，None不消失 | 旧summary保留同一data_dict引用，不另算一份。 |
| :726–743，COSIE预处理、原section改名、shape/内存事件 | 直接调用现有load_cosie_style_data，传原run_config.preprocessing；结果只封装 | 原memory record/reset_peak通过窄观察hook维持，不把training依赖引入data_io。 |
| :745以后Stage/prior、train/dry、输出断言、final embedding/prior/summary/history | **全部不属于U1准备接口** | 留在CRC原pipeline；以prepared字段恢复既有局部变量。 |
| :1035–1038 finally关闭backed文件 | 不提前close，不把backed结果改成提前拷贝 | 原caller在成功/失败退出时关闭已登记资源；U1不顺带修复原reader内部尚未返回时的异常清理策略。 |

**接缝选择冻结：** CRC pipeline新增可选的内部prepare委托，默认None保留CRC原分支；HLN/Spleen显式传委托和PairSpec。不使用`if dataset_name == ...`切换；dataset_name继续用于显示/来源。prepare实现不得import scripts。U1暂有CRC旧准备块与新共用准备块并存，这是有退出日期的迁移过渡，U2a完成CRC接入后删除重复块；不能把这项临时状态宣布为全系列收敛完成。

为保留交错I/O，只设计三个窄hook职责：metadata_ready（原CSV/txt写入）、spots_selected（原可选indices写入）、observe（原内存阶段名及reset_peak）。实现可用少量函数或轻量对象，不建立通用事件总线。hook由runner绑定原保存函数和memory_monitor，准备接口提供payload；已有写入失败继续在原阶段抛出，不能吞错、重试或新建缓存。最终artifact保存不进入任何hook。

### 4.3 直接复用与显式差异

| 事项 | HLN | Spleen | U1处置 |
|---|---|---|---|
| 文件/section映射 | 原SAMPLES：A1/D1及各路径 | 原SAMPLES：Spleen1/2及各路径 | PairSpec显式携带有序(section_id, sample_dir)，不解析文件名猜dataset。 |
| reader | paired.read_lymph_pair:168，底层read_backed_pair:68 | paired.read_spleen_pair:175，同底层reader | 直接复用原reader，职责不改。 |
| RNA身份 | prepare_rna_gene_ids:103→set_gene_ids:91 | 同左 | 直接复用；不改missing/duplicate检查，不换make_unique策略。 |
| Protein身份 | reader中set_gene_ids(ADT)，保留原marker显示名 | prepare_spleen_adt:160保留现有marker并校验 | 由明确reader/身份策略表达；不能统一成一条隐式gene_id规则。 |
| RNA共同feature过滤 | filter_globally_nonzero_genes:150 | 同左 | 显式传入同一filter；保留原sum>0条件、日志、全section过滤时点和顺序。 |
| 选行/物化 | CRC.select_obs_indices:337；paired.subset_to_memory:79 | 同左 | 纯select/validate核可移入数据层；CRC旧函数保留原签名委托，使MISAR旧调用继续有效。不要另写第二个算法。 |
| 预处理 | run_config.preprocessing原值 | 同左或各自显式值 | options只承接已resolved参数；不重新取defaults，不再resolve一次。 |
| 验证/改名helper | CRC.validate_rna_adt_alignment:311、spatial_range:303、rename_section_keys:380 | 同左 | 机械核可共用，保留np.allclose与旧fallback行为；通用核要求显式映射，CRC兼容wrapper继续提供原默认映射。 |
| 默认值/训练/保存 | 仍委托现有CRC/HLN关系 | Spleen仍从HLN取defaults | U1不处理；U3/U4/U7分别收敛保存、任务、默认/解析依赖。 |

候选位置是扩展paired.py或增加一个小型成对准备模块和轻量合同定义；避免把178行reader模块变成容纳所有输入族的巨型文件。选址以无scripts反向依赖和清楚的所有权为准，不以新增几个文件为验收条件。

### 4.4 preparation_audit与现有保存信息交接

| 原变量/保存消费者 | 来源及交接方式 | 不允许的替代 |
|---|---|---|
| rna_info、duplicate_tables、alignment_summary | 原reader/prepare/validate返回值，按section原顺序放入audit；metadata-ready交原writer | 不重算以“清洗”历史命名；HLN/Spleen仍可有名字含make_unique的旧产物。 |
| shared_genes_all、selected_shared_genes、suffix列表、adt_markers及集合/顺序bool | 原步骤直接交audit；保留原统计和文本内容 | 不排序交集，不把工程后缀解释成生物ID，不只保留数量而丢掉已有writer需要的列表。 |
| obs_indices、selected_obs_names_preview、原n_obs | 保留slice/array和原preview算法；原writer按n_obs输出int64 indices | 不把slice(None)提前扩成全量数组；不将preview当作完整barcode证据。 |
| saved_selected_spot_indices及其它路径 | runner的hook仍调用原writer，并保留返回paths供原summary使用 | PreparedDataset不决定output_root、命名或开关；不增新模型产物格式。 |
| data_dict、processed_data_dict、ADT唯一性、preprocessing_generated_keys、section_key_map | 原对象/现有bool放compatibility context或audit，供原summary原位消费 | 不提前关闭backed对象后再打开；不复制大AnnData仅为摘要。 |
| spot/feature metadata | 引用本轮准备已拥有的obs/var/indices；需要时表明source-row、现有校验证据 | 不读取新truth或外部标注；不将PCA轴伪装成选中gene；无barcode只能标row-order限制。 |

验证需列出CRC summary用到的准备局部量，避免封装后从另一runner“补字段”。summary的字段名、现有说明文字和最终保存位置U1不变。成功路径和中途失败已经写出的轻量文件集合也属于兼容检查，不只比较成功时的三个数组字典。

### 4.5 U1可直接实施的验证设计

验证材料在实施U1时新建独立目录，例如`refactor_checks/u1_paired_contract_YYYYMMDD/`；**U0不创建这些材料，不运行下列检查。** 旧oracle取自指定d52829b的只读源码及当前Finit，必要时通过只读Git对象导出到独立验证位置，不checkout/reset工作树，不import历史results中的实验快照。新旧版本用隔离进程防止同名module相互污染。

| 编号/层次 | 输入/操作 | 必须比较的结果 |
|---|---|---|
| U1-V0 静态范围 | allowlist diff、caller/import链、默认配置映射；模型/fit/预处理算法哈希 | 只有准备接缝/机械helper/两wrapper注入及独立验证增量；CRC默认委托None，MISAR旧helper仍能解析。 |
| U1-V1 身份正例 | 新小型paired AnnData/H5AD：原symbols可重复但gene_ids唯一；两section feature顺序不同；共有gene含两边都为零及只一边非零；HLN的ADT gene_id与显示名不同；Spleen直接marker | 与旧准备路径同一有序ID、全局过滤结果、gene截断次序、marker/display字段及audit；CRC控制样例仍用make_unique且不自动套HLN全零过滤。 |
| U1-V2 选行/RNG | all/first/random；max_spots=None、小于N、等于/大于N；两section大小不同；固定seed；gene截断开关 | indices值/dtype及slice保留，random排序、调用次数/顺序、rng.bit_generator状态、NumPy全局状态、进入Stage前torch RNG状态一致。边界非法值保持原入口错误，不改选行helper现有独立语义。 |
| U1-V3 错误与警告 | 缺gene_ids、重复ID/marker、obs不匹配、空间shape/值不匹配、marker集合相同但顺序不同、无公共gene；hook写入失败 | 旧/新异常类型/信息和阶段、已写轻量文件、已登记backed对象close轨迹一致。generic warn策略不在本批升级；不新增自动排序。 |
| U1-V4 内存/生命周期 | 同进程用哨兵preprocessor/reader返回明确dtype与共享buffer；记录四次subset顺序和资源表；真实小backed对象 | PreparedDataset不变换返回矩阵，processed同对象；原spatial复制点保留；Stage前/后backed仍按旧生命周期，finally关闭；不因新容器额外强制to_memory/astype。mmap通用容器做透传探针，U1不假称paired真实路径使用mmap。 |
| U1-V5 小型真实准备 | 足够支持所选HVG/n_comps的小RNA/Protein数据，新旧路径在同依赖、seed/线程、相同参数下CPU准备；先无Harmony，已有环境可支持时增加小Harmony对照 | feature/spatial的shape、dtype、数值、顺序、processed obsm与必要audit；不是仅靠mock证明预处理等价。缺依赖记录未完成，不安装、不把跳过写成通过。 |
| U1-V6 保存/事件 | 旧pipeline在Stage构造前以测试哨兵停止，新路径相同；检查metadata/indices及memory事件；再用相同Stage/fit测试替身使原最终writer消费确定payload | 提前保存的文件名/字段/内容、条件开关和失败足迹一致；最终summary准备字段映射和产物名称一致。替身证据只验证编排/保存，不宣称真实训练数值通过。 |
| U1-V7 初始化与CRC回归 | 在独立小CPU验证进程中捕获新旧进入Stage的同一配置/准备输入/RNG，再按当前Finit构造；CRC自身旧默认路径、HLN、Spleen三路径对照；MISAR对旧helper做调用回归 | 全部保留参数初值及构造后RNG逐项一致；Finit不得用E1未修正初始化作oracle。模型/fit代码未变且prepared输入一致，与旧GPU数值记录分开报告；U1不启动真实训练/完整analysis/GPU gate。 |

测试哨兵只存在新增验证进程内，不进入业务模块。比较值级产物时预先映射新旧临时根目录，不忽略其它字段；压缩容器的时间戳与payload分开。接口透传原则要求精确dtype/order/identity一致，浮点真实准备若因既有库/线程无法稳定，先复核同版本重复性，记录证据并停止该验收项，不据旧报告判本批通过。

**U1验收必须同时覆盖HLN、Spleen和CRC控制路径。** 只看到公共函数可调用或两份feature形状相同不算完成。模型前向等价已有历史证据只作背景；将来U4/U8计算验证必须使用包含Finit的当前初始化基线。

## 5．U2：按输入族接入prepare_dataset

| 要素 | U2阶段合同 |
|---|---|
| 依赖/唯一目标 | U1验收；唯一目标是所有训练输入族提供同一逻辑结果边界，各族算法和存储不统一。 |
| 允许候选 | 各data_io adapter/准备函数，相关runner准备接缝，显式InputSpec分派与轻量合同；SPATCH原始路径能力按U2e单列。 |
| 禁止范围 | 不迁Stage/fit/final save，不改模型/图/OT/损失，不造全数据集大文件，不重写历史cache或更改预处理语义。 |
| 当前→拟议接口 | 多种三元组/字典/manifest/runner loader→prepare_dataset显式分派→PreparedDataset；可保留旧函数做委托，不能通过dataset名称猜文件和参数。 |
| caller/顺序 | U2a CRC→U2b MISAR/Thymus/Simulation→U2c MouseBrain/generic→U2d Embryo→U2e SPATCH；每族先旧入口适配，再消除已替代的重复准备块。 |
| 验证 | 每族复用U1身份/生命周期合同并增加自己的输入/缓存/模式检查；不把九族测试都压到U8。 |
| 停止条件 | 需要统一算法/顺序/dtype、扩大内存、伪造identity、改旧缓存才可接入；独立记录，不带病进入下族。 |
| 完成条件 | 九dataset与generic各合法模式都有显式路径及证据限制；各小批独立对照通过，准备实现不依赖scripts业务。SPATCH两条输入路径都可用才算该族完成。 |

### U2独立小批合同

| 小批/依赖与唯一目标 | 允许候选；当前→拟议接口及caller顺序 | 禁止与验证 | 停止/完成条件 |
|---|---|---|---|
| U2a／U1：CRC接入 | CRC旧:611–743准备块、paired.make_unique/reader；CRC默认准备→显式CRC PairSpec→公共prepare；先CRC本入口，回归HLN/Spleen | 不套gene_ids/全零过滤；验证symbol后缀、首section交集、ADT顺序、选行与原轻量文件 | 需要改变CRC身份则停；CRC全路径经公共准备、U1过渡重复块删除且三caller兼容即完成。 |
| U2b／U2a：MISAR族接入 | misar.common_var_names/read_backed_pair、adapted.read_thymus_pair/read_simulation_pair、MISAR准备块；先MISAR，再Thymus，再Simulation；各保留旧入口 | 不改RNA/ATAC现行算法、Protein CLR、Thymus坐标/marker规则及Simulation factors；验证训练4→1、各族section/obs metadata及audit先后 | 模态/ID策略须改变则停；三caller各有PreparedDataset与旧artifact字段映射，数据helper不再经CRC导入即完成。 |
| U2c／前批合同：MouseBrain/generic接入 | datasets.build_mousebrain_section/preprocess_multisection_cosie_style、MouseBrain.prepare、generic.load_preprocessed_inputs；依次MouseBrain→bundle→显式raw config→synthetic | B14改名、UNI来源、RNA/Metabolite、原flat/nested配置、generic float32转换所在边界保留；warn不升级；bundle/mmap不得额外复制 | 配置读取为“统一”而失真则停；所有已有模式显式分派，缺barcode/processed语义如实记载即完成。 |
| U2d／已稳定合同：Embryo接入 | hesta.preprocess_hesta_rna/load_preprocessed_manifest、runner.load_external_preprocessed_run；先已有manifest/外部复用，再原始HESTА准备适配 | 不换QC/HVG/SVD/z-score算法，不强加Harmony；不变reuse模式前置audit、require_harmony、前N截断、mmap所有权 | 无法保留分块/内存或外部身份仅凭row count被“补齐”则停；audit/preprocess-only/dry/train原准备分支各有对照即完成。 |
| U2e／前族验收：SPATCH双输入能力 | 下述U2e-1和U2e-2独立记录；当前runner.load_preprocessed_cache及原始准备helpers→两个明确adapter，共同PreparedDataset；正式SPATCH caller保持 | 不把cache-only视为最终合同；不隐式cache miss后重做预处理，不修改旧cache、旧checker历史expected或来源 | 既有cache兼容不成立或raw算法/身份不足以确定则停；显式raw可从无cache开始，显式reuse严格复用且两者交接合同验证完成才结束U2e。 |

### U2e：SPATCH规则的实施约束

1. **U2e-1先做结构迁移。** 将`run_spatch.load_preprocessed_cache:455`及相关schema/身份校验移至明确数据边界，保留既有只读cache路径、校验、dtype、metadata限制和当前CLI；验证工具的必要导入可同期迁移，但不改历史fixture/expected。迁移不能新建或覆盖用户已有cache。
2. **U2e-2再做明确功能增量。** 按本轮新增要求提供显式raw模式：从原始数据完成必要读取、身份/模态对齐及既有预处理，再训练；不要求预先有model-ready cache。可以把本次结果写入用户明确选择的新cache以便下次复用，但不能强制先运行一条独立预处理命令才允许训练。
3. raw与reuse为明确输入方式；默认及CLI参数拼写到该批冻结，既有有效cache命令保持兼容。无效cache不静默回退raw，缺原始输入不伪造数据，raw不暗读某个历史结果当作“从头准备”。
4. 审计指出runner仍有`inspect_full_inputs:626`、`preprocess_modalities_sequentially:754`、`save_preprocessed_cache:289`等函数，但正式main禁用build；**函数存在不足以证明raw已可直接启用**。到U2e再针对性核实算法、内存和身份链，不仅删除禁用判断。现阶段不认定旧helper完整可用，也不提前选新算法。
5. raw能力是已纳入路线的功能要求，不伪装成对当前cache-only主路径的纯等价移动。验证分为“旧cache路径等价”和“新增raw路径正确/可完成”两套证据；小型合法原始fixture必须走真实准备，再走新cache读取对照feature/coords/身份/参数。真实规模内存或GPU运行只在该实施范围明确包含时安排，缺失则如实列未验证范围。
6. H&E/Protein/RNA处理与去除项等具体raw语义以届时核对的既有实现/有效记录冻结；若无法从源码确认，列出缺少的输入schema或算法证据，不用通用PCA默认补猜。维度相同不能证明cache/raw语义一致。

## 6．U3：公共artifact保存核与导出选项

| 要素 | U3阶段合同 |
|---|---|
| 依赖/唯一目标 | 各族PreparedDataset合同稳定；唯一目标是消除同一保存操作多份实现，让导出差异显式。 |
| 允许候选 | CRC.save_final_embeddings/save_spatial_arrays/save_ot_prior_topk及indices；MouseBrain.save_embeddings/save_ot_prior_topk/save_run_artifacts；generic.save_training_artifacts；Embryo weights写入；已有tensor_to_numpy直接复用。 |
| 禁止范围 | 不迁epoch/Stage/prior编排，不统一文件名/schema，不强制全部dataset导出所有大数组，不实现resume，不更改BF16转换和prior数值。 |
| 当前→拟议接口 | 多套save_*→共同数组/JSON/prior/weights保存核＋显式ExportOptions/LegacyLayout；调用者仍提供历史summary payload，writer不import runner。 |
| caller/迁移顺序 | U3a先CRC保存器及其MISAR/HLN/Spleen/Embryo/SPATCH调用者，保留旧函数薄委托；U3b再MouseBrain；U3c最后generic的周期/最终保存和weights布局。各小批单独验收。 |
| 验证 | 新小payload含FP32/BF16、双向top-k及QC开关、无prior、None history、所有保存flag；比较名称/键/顺序/dtype、旧return path结构、覆盖/目录错误行为；压缩容器比较成员payload，不以新timestamp否定数值等价。 |
| 停止条件 | 必须改历史reader、输出layout或schema才能共享，或者writer开始决定训练模式/输入身份；停止并缩小保存核。 |
| 完成条件 | 原有保存功能和产物布局全部有显式policy映射，共同保存不再由某个runner提供；旧public函数若保留只委托，不复制实现。 |

ExportOptions至少区分embedding/spatial/indices/metadata、prior及candidate QC、weights、history/summary，以及周期或最终调用原因。布局字段采用现有名称，不推出新的统一结果磁盘格式；U1准备阶段轻量writer可在本阶段共用，但其原调用时点不能合并到最终save。

## 7．U4：fit之外的公共训练任务编排

| 要素 | U4阶段合同 |
|---|---|
| 依赖/唯一目标 | 对应U2/U3已验收；唯一目标是公共任务承担Stage/prior→前置步骤→fit→导出，runner只映射显式请求。 |
| 允许候选 | CRC.run_crc_pipeline、MISAR.run_misar_pipeline、MouseBrain.run_mousebrain、Embryo.run、SPATCH.main、generic.train_stage_model中的任务部分；必要训练任务边界/ExecutionPolicy。 |
| 禁止范围 | 不重写iter_fit_model数学/调度，不改模型/Finit、干跑mode、AMP/chunk、预处理、CLI默认，不把数据算法搬进训练任务。 |
| 当前→拟议接口 | 各runner任务函数→公共任务(已resolved配置, PreparedDataset, ExecutionPolicy, ExportOptions)；现有fit及保存核被调用，不二次解析/合并配置。 |
| caller/迁移顺序 | 见下表；先provider再其delegating caller，不同时迁九runner。默认/argparse相互依赖允许暂留至U7，必须与业务依赖分账。 |
| 验证 | 同seed/Finit初始化、初始prior、前置forward及train/eval轨迹、fit实参、最终输出/产物；小型固定权重数值与行为矩阵分别比较，必要checker导入迁移记录。 |
| 停止条件 | 公共任务只能通过dataset名暗分支，或必须消除历史运行差异才可复用；发现新数值/随机轨迹偏差即停，不修改loss或放宽容差。 |
| 完成条件 | 所有任务族各自验收；业务编排不再借另一个runner，shared fit唯一；原入口可用且没有重复resolve。 |

| 小批 | caller依赖与迁移顺序 | 明确保护及完成门槛 |
|---|---|---|
| U4a | CRC任务先提为公共paired任务；CRC保留薄调用，再逐个接HLN、Spleen | CRC自身/两wrapper配置和初始prior断言、train-mode dry、准备hook时点、保存和finally对照；只这三个完成后结束本批。 |
| U4b | MISAR任务→Thymus→Simulation | 先解除MISAR对CRC业务helper依赖；保留RNA/ATAC与RNA/Protein、4→1顺序、adapter audit先行、MISAR eval dry及各族metadata；三入口分别验收。 |
| U4c | MouseBrain单独迁移 | 正常训练也先train-mode/no_grad epoch0；lambda schedule、allow_empty_epochs、FP32及独立导出选项保持；不能用共同dry默认替换。 |
| U4d | Embryo与SPATCH分两个验收单元，均调用公共任务 | Embryo保留audit/preprocess-only/external reuse/单模态/noOT和weights；SPATCH通过U2e raw或reuse取得合同，任务不负责重做预处理。两者不强行合成“大数据runner”。 |
| U4e | generic最后接入 | 保留bundle/raw/synthetic、合法noOT、零epoch、周期yield保存、最终embedding选择与weights布局；周期输出是该epoch训练前向、权重已step的现状，不另做eval改语义。 |

ExecutionPolicy必须明确preflight/epoch0、dry的模型mode、fit参数、刷新enabled语义、allow_empty_epochs、yield_every及导出时点。公共任务不自动推断“某dataset应该怎样”，无独有行为时使用普通路径；确有既有差异才有窄策略。不要把任务接口变成几十个任意回调的不可审计框架。

## 8．U5：analysis输入、结果/标签读取及依赖方向

| 要素 | U5阶段合同 |
|---|---|
| 依赖/唯一目标 | 数据与产物合同稳定；唯一目标是同一输入身份只装配/对齐一次，读取层不依赖分析流程。 |
| 允许候选 | analysis.loaders/inputs的现有result/metadata读取，data_io.comparison_inputs及hln_annotations方法reader，load_saved_labels相关函数，必要轻量AnalysisInput；visualization_inputs的接口边界。 |
| 禁止范围 | 不改聚类/指标/truth过滤/采样，不强迫改历史文件；不合并整个inputs/loaders/joint_results；不把重做输入预处理的可视化工作流伪装成普通只读loader。 |
| 当前→拟议接口 | 多种dict/MethodData/tuple→AlignedAnalysisInput（embedding, sections, spot_ids/evidence, coords, optional truth, sources）；已有CSV/NPY/zip labels→load_saved_assignments(显式layout, identity)。 |
| caller/迁移顺序 | U5a先HLN/Spleen/CRC普通结果与labels；U5b再MouseBrain/MISAR/Thymus/Simulation及外部method；U5c最后SPATCH/Embryo和UMAP/外部注释的读取边界；每批保留旧reader委托。 |
| 验证 | 原文件布局、section/spot顺序与别名、重复/缺失身份的原错误策略、truth缺失、NPY仅row-order证据、同输入重复打开计数、mmap/dtype、旧saved labels精确对齐；不执行新聚类证明读取。 |
| 停止条件 | “共享读取”要求统一truth过滤/标准化、改变科学cohort、排序行或虚构barcode；出现这样的要求就保留adapter，不强行合并。 |
| 完成条件 | 当前caller统一到逻辑输入/标签合同，明确格式adapter；共享读取核不import analysis.metrics/protocols或scripts；data_io→analysis的相关知识依赖解除，必要工作流仍独立。 |

实现方向：读格式/身份核保持中立；comparison的truth policy由analysis协议层显式传入/后处理；SPATCH可视化文件映射作为显式输入配置传给reader；scale_full_then_select等展示变换仍由analysis caller组织。是否移动文件由依赖决定，不能只为了消除包双向箭头把所有读取塞进analysis。joint_results的历史pooled指标校验有不同职责，静态无当前caller不能作为删除依据。

## 9．U6：普通exact KMeans scope流程收敛

### 9.1 第一对冻结为HLN普通comparison与Spleen普通comparison

选择的是`analysis/comparisons.py::hln_run_scheme:679–896`与`analysis/method_comparison.py::mouse_spleen_run_scheme:151–212`，**不是HLN人工标注工作流，也不是Spleen requested协议**。

| 选择依据/真实差异 | 已核实源码 | U6处理 |
|---|---|---|
| 两section、无普通生物truth，exact KMeans | 两函数均joint scaler→KMeans→分section labels；independent为K外循环、section内循环 | 首批共享scope骨架，无需先解决监督truth和大规模MiniBatch差异。 |
| 同seed/n_init/max_iter和指标核 | protocols.HLN/SPLEEN均seed0、n_init20、max_iter300；HLN的metric_space/internal分别是fitted_space/internal_metrics别名；Spleen直接调同一核 | 不改变算法、dtype或默认KMeans参数。 |
| 相同空间/标签保存核可复用 | hln_save_labels:665与mouse_spleen_save_labels:138列相同；Spleen已调用hln_spatial_agreement，均section内6-NN；绘图均point_size10、dpi220 | 共享参数明确的计算/保存核；section ID和来源保持各自值。 |
| **independent默认K不同** | HLN:680–681 joint=2…12、independent=[5,8,10,12]；Spleen两个scope都=2…12 | ProtocolSpec分别传joint_ks/independent_ks；不能看到同METRIC_KS就统一。evaluate:268–276也显式保持HLN这一区别。 |
| 绘图/保留差异 | HLN independent对实际请求的每个K绘图；Spleen只对PLOT_KS绘图；外层各有retention/shared metrics | 保留请求覆盖、绘图选择和新输出内retention；不动历史结果、不混淆joint_per_section的“不平均”与既有空间连续性均值。 |
| 其它候选为何后置 | MouseBrain有多label/group诊断；MISAR有多truth；Thymus有固定ASW样本；Simulation有真实因子/监督指标；CRC/SPATCH为MiniBatch | 这些不是首批只改参数即可证明等价的两个最小目标，后续逐个比较，不整文件合并。 |

### 9.2 阶段合同

| 要素 | U6冻结内容 |
|---|---|
| 依赖/唯一目标 | U5输入/labels合同；唯一目标是相同scope流程共用一个任务核，协议和科研问题不统一。 |
| 允许候选 | 上述两run_scheme，所调用共同clustering/metrics/labels writer的窄接口，明确ProtocolSpec和各caller适配；必要protocol常量引用。 |
| 禁止范围 | 不并整个comparisons/method_comparison/requested；不改K、seed、dtype、truth mask、采样、batch/UMAP或HLN专项；不恢复raw-vs-standardized比选入口。已有合法单协议raw用法不得被抽取意外破坏。 |
| 当前→拟议接口 | 两套run_scheme→execute_exact_kmeans_scopes(AlignedAnalysisInput, ProtocolSpec, 已有输出策略)；计算labels与指标共用，科研额外诊断和跨方法汇总仍在各workflow。 |
| caller/迁移顺序 | U6a先HLN：保留evaluate和method_comparison两caller及joint/independent显式K；再Spleen：保留跨方法caller。U6b只在首对验收后逐个评估其它exact KMeans路径，requested先作为对照，不强迁。 |
| 验证 | 固定同序小embedding/coords：scaler参数、fit调用次数/先后、每K labels、所有指标行及CSV字段/顺序、plot/retention集合、B9身份；HLN默认/显式K、单scope/双scope、Spleen全K分别测。 |
| 停止条件 | 必须统一协议或增加dataset名隐藏分支；某协议差异无法由明确参数/保留的外层步骤表达；新label/metric/cache差异无法解释。 |
| 完成条件 | 首对同一scope核、两个原工作流行为对照成立；后续每迁一个单独验收；未迁专用科研流程独立合理，不以把所有dataset塞入一函数作为完成要求。 |

第一批保留原循环次序和scaler重复拟合次数，即使两者都在每个independent K重新fit相同section scaler，也不顺便优化；减少重复计算可另列优化决定。HLN和Spleen的普通comparison无reference joint_per_section适用标签，不因此生成监督truth；其它协议的joint_per_section继续切已有labels，不在这个任务核重新fit。

## 10．U7：CLI、配置/别名、运行方式与文档收口

| 要素 | U7阶段合同 |
|---|---|
| 依赖/唯一目标 | 公共数据/保存/任务/分析层稳定；唯一目标是让正式用户接口清楚、一致且兼容，不再由runner互相提供defaults/parser业务。 |
| 允许候选 | scripts九入口/generic/batch/evaluate及四薄工作流的参数映射；training.config/显式dataset配置归属；运行方式说明和README更新仅到此阶段再做；当前checker必要导入调整。 |
| 禁止范围 | 不附带改模型/协议/默认值，不因流行引入src或测试框架，不删除旧CLI而无已定兼容安排，不恢复experiments/旧比选入口。 |
| 当前→拟议接口 | 九训练入口＋generic＋batch→在下表决定A保留薄CLI或B单正式CLI并兼容旧入口；都只把argv映射到公共请求，不含业务流程。 |
| caller/迁移顺序 | 先冻结配置/ID映射与兼容矩阵→将CRC/MISAR parser/defaults与Spleen→HLN依赖从runner移到明确公共/数据配置边界→迁各入口→batch/generic/evaluate→文档/验证启动方式。 |
| 验证 | 旧argv、help/必需项、显式覆盖和错误返回、cwd/PYTHONPATH、batch子进程命令、数据ID别名、产物目录；AST确认runner之间无业务/默认/保存import。 |
| 停止条件 | 尚未决定旧命令生命周期或新的统一CLI迫使改变默认/准备语义；不靠删除旧入口完成“简化”。 |
| 完成条件 | 最终选择已记录并实现；所有旧有效命令在承诺期可用；runner不互相提供业务/默认/保存，运行方式与文档对应，未选择单CLI也可通过。 |

### 后续一次性决定的CLI兼容事项

以下集中在U7前冻结，不要求U0现在选CLI数量；SPATCH输入语义已确定，具体参数映射在U2e先冻结并带入U7，不被推翻。

| 待决定事项 | 决策需要包含的内容 | 当前保守兼容线 |
|---|---|---|
| A九薄CLI或B单CLI | 主入口、旧入口是否长期薄委托/过渡期、generic与单CLI关系 | 两者均不保留runner业务实现；公共层不依赖CLI选择。 |
| 数据ID/别名 | misar↔misar_seq、crc↔crc_stereocite的显式映射；canonical ID归属 | 现有训练/分析ID均可明确解析，不从result目录名推断。 |
| 配置schema与覆盖 | flat generic/nested MouseBrain兼容读法、None/False/0、preset与dataset/default的权威 | 旧配置不重写；不要靠迁移顺手改裸scale或epochs。 |
| 工作目录/启动方式 | python scripts/...与python -m的支持范围、相对路径基准、sys.path方案、是否另开安装支持任务 | 先保留已有根目录运行方式；不以加pyproject作为结构批次的先决条件。 |
| SPATCH raw/reuse参数 | 输入模式、旧cache参数别名、cache写入/复用选项、错误时不隐式切模式 | raw可无既有cache起步；reuse只读所选结果；旧有效cache命令不失效。 |
| 输出与导出选项 | LegacyLayout、output_dir层级、overwrite现状、weights/prior/periodic行为 | 旧文件名/字段/顺序/dtype和有效参数含义保持；不承诺新增resume。 |
| 验证工具导入 | 哪些runner helper旧导入临时保留、迁到何公共接口、取消委托的条件 | 当前checker可随实施批迁必要import；不修改旧fixture/expected/报告来适配新实现。 |

README错误示例的更正属于第12节单列问题，不能把“文档收口”当作更改epochs默认的授权；届时若修文档，须明确仅修示例还是另有行为需求。

## 11．U8：综合兼容验收

| 要素 | U8阶段合同 |
|---|---|
| 依赖/唯一目标 | 各U批记录齐全；唯一目标是证明公共层收敛后的兼容与新增SPATCH能力范围，不继续做结构设计。 |
| 允许候选 | 新综合验收材料、必要当前checker导入/公共接口适配、接口文档与尚未移除的纯兼容委托处置；任何生产修正回到对应批次重新验收。 |
| 禁止范围 | 不借验收修B4/R1/R3/旧smoke，不换冻结expected，不重跑/覆盖历史result/cache，不做新科研实验流程。 |
| 当前→拟议接口 | 检查U1–U7约定的公共接口与全部保留CLI，而非再创建一层统一入口。 |
| caller/顺序 | 先数据/配置/身份/产物→任务行为和初始化/计算→分析协议与缓存→CLI与依赖；逐数据族归档。 |
| 验证 | 九dataset＋generic合法模式；SPATCH raw/reuse；CRC/MISAR短caller族；epoch0/noOT/零epoch/周期保存；B13/B14/B12/B11；Finit同seed初始化＋固定权重计算；joint_per_section、batch/UMAP/专项输入不变；新第十dataset小adapter接入演示无需复制任务/普通分析。 |
| 停止条件 | 任一必要gate缺失/新回归、历史失败被改写成通过、源码provenance与cache identity混淆、仍靠runner业务import或复制scope循环完成接入。 |
| 完成条件 | 所有必需对照有证据及明确范围，迁移债务关闭；结构兼容、计算数值、初始化、既有已知失败、新SPATCH功能分别下结论。真实大规模/GPU未验证时明确标记，不声称已获完整数值验收。 |

计算验收应将“固定参数下前向/梯度/step/刷新一致”与“同seed创建参数一致”分成两条。原有GPU gate若纳入届时执行范围，保持原fixture/expected及历史已知cross-chunk敏感；额外当前Finit的初始化/小型接口对照是新增证据，不是替换旧基准。未获得运行范围或环境条件时保持该验收项未完成，不引用E1历史通过替代本轮结果。

第十dataset接入演示采用新小型显式配置/必要adapter和已有普通分析协议：应只增加输入适配/配置，不复制Stage/prior、fit外编排、保存或scope循环。若数据有新模态算法/独有科研问题，它们合理需要独立实现，不能以“零代码接入所有数据”作不现实承诺。

## 12．独立记录、不夹入结构批次的问题

| 当前现状/问题 | 本路线处置 |
|---|---|
| generic的obs身份检查只warn，build_section_modalities不消费False | 保留。U1不升级raise，不自动重排；合同记录证据等级。是否改严格性单独决定。 |
| 基础模型/部分裸入口post-OT scale=1.0，研究命令显式0.5 | 保留默认/覆盖链；不因统一配置改模型行为。前次核对三组历史运行均0.5也不成为改裸默认的理由。 |
| noOT/零epoch/dry/epoch0差异 | 用ExecutionPolicy忠实表达；不取消Embryo noOT的既有刷新eval记录，不修改MouseBrain epoch0，不统一所有dry为eval。 |
| generic与其它入口summary/identity artifact不同 | 用显式layout和证据限制兼容；新增统一内存合同不要求改旧summary或补历史barcode。 |
| README CRC训练示例与epochs=0默认不一致 | 记录为单独文档/行为决定；U0–U6不修，U7也不能未经说明改变epochs默认。 |
| B4/R1/R3及旧Stage smoke文案断言问题 | 本系列结构批不处理；旧失败保留，不改期望值规避，不自动开始这些修复。 |

SPATCH双输入能力不在“永久不处理”列表：它是用户本轮明确补充的未来功能要求，按U2e-2单独实施和验证。旧cache路径的等价迁移与raw新能力不能合并成含糊的“缓存重构通过”。

## 13．交付与阶段退出记录

U0产物仅本文；历史计划、两份审计报告及生产文件保持原样。本轮状态确认未发现生产源码变化，后续无须再次回退。

每个后续小批的退出记录至少包含：实际起点/源码增量、允许范围与实际修改、迁入caller/临时委托清单、数据身份及资源合同对照、产物/配置兼容、B9与新provenance分别说明、数值/初始化证据范围、保留失败、未验证事项、下一批依赖。小型新增证据使用独立目录，不覆盖原报告或历史运行。

**冻结决定：** U1仅HLN/Spleen准备试点，保留CRC控制路径和原Stage/fit/save；U2分五族且SPATCH最终具备raw/reuse双路径；U4按依赖分批；U6首对为HLN/Spleen普通comparison，保留不同independent K；CLI数量留公共层稳定后决定。本文交付即停止，U1尚未开始。


## 13．U1退出记录（2026-09-21追加，保留上述U0历史原文）

**U1已完成；U2a未执行。** 起点仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`，没有提交或回退。实施及验证报告：[U1 REPORT](../../refactor_checks/u1_paired_contract_20260921/REPORT.md)；机器可核实结果：[validation_results.json](../../refactor_checks/u1_paired_contract_20260921/validation_results.json)。

HLN/Spleen的main显式传入`PairSpec`和`prepare_paired_dataset`，公共准备返回`PreparedDataset`及过渡`compatibility_context`。CRC新增内部可选准备委托，默认None仍走原准备块；原reader/identity算法不变，机械helper归data_io.paired、CRC旧签名继续委托，MISAR旧导入/调用兼容。资源容器、rng、三个窄hook、Stage/prior、fit或dry、最终writer和finally仍由原runner承接。

本批生产变更为4个现有文件及新增`data_io/paired_preparation.py`；未改parser/defaults、模型、fit、预处理数学、analysis或最终保存实现。CRC从Stage配置到finally及文件末尾逐字不变。U1-V0～V7通过：100个入口案例加MISAR helper对照；真实小型CPU无Harmony及Harmony预处理；HLN/Spleen/CRC各146个参数的当前Finit初始化及构造后RNG精确一致；引用、身份、事件/保存失败足迹和保护检查完成。细分覆盖范围见报告，不将预期异常案例写成准备成功。

新增真实证据中两项旧行为原样保留：当前AnnData的to_memory会关闭共享backed句柄，原finally再执行close；dense H5AD同时用行/列数组切片存在旧h5py限制，旧/new均触发相同TypeError，成功抽样路径已用独立CSR-backed小输入补齐。没有把这两项修正夹入U1。

迁移债务明确登记：CRC旧准备块与公共准备块暂时并存，**在独立授权的U2a接入CRC并完成三caller回归后删除替代块**；MISAR旧helper导入迁移留给U2b；compatibility_context只承接旧data_dict与indices路径，公共保存核与完整训练任务分别仍属U3/U4。CLI和defaults依赖未在U1收口。SPATCH raw/reuse双输入要求继续保留在U2e。

未跑GPU gate，未执行模型forward、真实训练、真实大规模数据准备或完整analysis；未修改历史result/cache、冻结证据或三个gene_imputation保护目录。没有安装依赖、处理B4/R1/R3或改变scale/epochs/noOT等现状。上述U0“当时尚未执行”文字保留原义，本追加记录只说明本次U1退出状态；到此停止。


## 14．U2a退出记录（2026-09-21追加，不改写U0/U1历史状态）

**U2a完成，U1的CRC重复准备块并存债务已关闭。** 实际起点是U1已验收的工作树，HEAD仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`，U1交付SHA逐项匹配；本批以该工作树的独立源码副本作oracle，没有把原始d528或E1作为整个U2a对照。

CRC、HLN、Spleen现在都通过`data_io.paired_preparation.prepare_paired_dataset`取得U1既有`PreparedDataset`。CRC显式PairSpec保留read_backed_pair、symbol/make_unique、filter_shared_genes=None、原Protein var_names及CRC_003/CRC_006顺序；HLN/Spleen保留gene_ids、全局nonzero过滤及各自Protein策略。没有增加dataset名称dispatcher，也没有新建或改名PreparationOptions等接口。

在公共路径等价验证通过后，已删除U1 CRC文件原582–714行的旧准备业务块，删除后重新对照通过。三个窄hook及解包只去掉一级缩进；原前置步骤、Stage/Finit、prior、fit/dry、最终save和finally保持。相对U1仅CRC接缝及公共模块2行说明变化，其余75个相关源码SHA一致。三caller完成的是准备层收敛，未收敛完整训练任务或保存。

[U2a实施报告](../../refactor_checks/u2a_crc_prepared_20260921/REPORT.md)与[验证结果](../../refactor_checks/u2a_crc_prepared_20260921/validation_results.json)记录U2a-V0～V8：110个入口案例的删除前/后对照、三caller真实小型CPU无Harmony/Harmony预处理、三个入口各146个参数的Finit初值及构造后RNG、身份/资源/错误/保存时点、MISAR实际导入的六个helper调用及保护检查均通过。数值按dtype/shape/值精确比较；没有忽略summary整块。原dense双fancy-index失败及reader未返回时的清理缺口保持，成功抽样用独立CSR-backed案例验证，未夹带correctness修复。

仍保留的兼容接口及后续归属：

- CRC.select_obs_indices仍供MISAR:503调用，subset_to_memory再导出仍供MISAR:517/521调用；权威算法在data_io.paired，MISAR依赖迁移留U2b/U4b。
- CRC.validate_rna_adt_alignment、rename_section_keys仍供U1/本批引用探针调用；当前scripts/tools未发现其它正式外部caller。spatial_range也未发现剩余正式外部caller，暂保留原薄委托API；它们不是第二套准备算法，本批不顺带删除旧签名。
- ensure_dir及embedding/spatial/prior保存器仍供MISAR使用，Embryo/SPATCH仍有既有保存依赖；保存核留U3，任务编排留U4。
- HLN/Spleen仍调用CRC训练后半段，Spleen仍继承HLN defaults；公共任务与CLI/defaults分别留U4/U7。当前legacy keyword策略接口保持可用，PreparedDataset的data_dict/indices路径compatibility_context仍是过渡合同。

原U0/U1关于当时未开始或旧块并存的记载保持原文。SPATCH raw/reuse要求继续留U2e。本批未执行U2b/U3/U4，未修改MISAR/Thymus/Simulation准备，未处理B4/R1/R3。未跑GPU gate、模型forward、真实大规模数据准备、训练或完整analysis；未改模型数学、历史result/cache、冻结证据或三个gene_imputation保护目录。交付后停止。


## 15．U2b退出记录（2026-09-21追加，不改写U0/U1/U2a历史状态）

**U2b完成，MISAR family数据准备层完成收敛；U2c未执行。** 起点是U2a已验收工作树，U2a交付SHA匹配，HEAD仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`。本批oracle来自该工作树的独立源码副本，不回到原始d528或E1，不回退、不提交、不改暂存区。

实际依次迁移MISAR、Mouse Thymus、Simulation，各24/27/28个对照案例通过后才进入下一caller。三个main显式传入`data_io.misar_preparation.MultisectionSpec`和`prepare_multisection_dataset`，返回原`data_io.paired_preparation.PreparedDataset`，未移动或重命名U1/U2a接口。MISAR训练section仍4→1；RNA make_unique、RNA/ATAC首section有序交集与截断不变；Thymus obs坐标及17-marker adapter保持；Simulation通过独有truth_provider交接已加载spatial_domain/spfac/nsfac引用，没有向普通caller生成truth。

三个caller对照通过后删除MISAR runner内旧准备块，再跑全部79个案例通过。prepare只负责准备骨架，四个窄hook承接原duplicate/shared txt、indices、obs_metadata与memory观察；两个adapter audit仍在原main进入pipeline之前写入。资源容器、seed/rng、Stage/Finit、prior、fit/dry、最终save和finally归原runner，model_config至finally逐字不变。compatibility_context保留已有data_dict/AnnData及保存路径引用，不规定未来adapter永久必填。

[U2b实施报告](../../refactor_checks/u2b_misar_family_20260921/REPORT.md)和[验证结果](../../refactor_checks/u2b_misar_family_20260921/validation_results.json)记录V0～V9：身份/顺序/RNG/错误及资源/轻量artifact时点精确一致；三caller各真实小型CPU无Harmony及Harmony预处理；各62个参数的当前Finit CPU初始化及构造后RNG精确一致；原writer完整summary/文件payload/flags比较。79例包含预期异常和确定性writer替身，并非79次训练。另执行CRC/HLN/Spleen共6个小handoff、MISAR select/subset实际导入对照，paired family源码不变。184,666个既有保护文件的路径及lstat元信息清单保持；不将元信息检查冒称全量内容SHA验证。

已关闭与暂留依赖：

- **已关闭：** MISAR数据准备经CRC runner获取select/subset的业务依赖，改用data_io权威机械核；MISAR旧准备双实现退出。data_io无反向scripts依赖。
- **暂留U3：** MISAR仍从CRC导入ensure_dir及三个最终保存函数；本runner的轻量保存hook仍调用原writer。没有提取artifact核。
- **暂留U4b：** Thymus/Simulation仍调用MISAR训练后半段，fit之外完整任务编排未迁移；Stage/fit算法未改。
- **暂留U7：** parser/default/CLI依赖，以及旧runner helper API是否删除。MISAR alignment/spatial_range/rename为data_io薄委托，当前scripts/tools未发现其正式外部consumer；CRC旧select/subset API仍供历史验证/兼容使用，不因本批迁移删除历史调用位置或证据。当前tools/validation/replay_model对CRC既有导入未改，后续仅按需要迁导入。

本批修改5个现有生产文件，新增1个175行准备模块；原reader、adapter和预处理数学保持，基线其余72个Python源码SHA不变。没有强迫CRC/HLN/Spleen与MISAR族使用同一生物学策略或同一个准备函数。没有改B9 cache identity/旧summary schema，没有重建cache。SPATCH raw/reuse双输入要求继续留U2e，U2c/U3/U4未执行。

未跑GPU gate、真实模型forward、大规模真实数据准备、长期训练或完整analysis；未修改模型数学、历史result/cache、冻结证据或三个gene_imputation目录。没有处理B4/R1/R3、旧smoke、裸scale/epochs或noOT/dry差异。旧异常行为保留，初始化证据使用当前Finit，不引用历史GPU记录替代本批验证。交付后停止。


## 16．U2c退出记录（2026-09-21追加，保留此前各阶段历史状态）

**U2c完成；MouseBrain/generic准备层已收敛，U2d未执行。** 起点是U2b已验收工作树，42项U2b交付SHA匹配，HEAD仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`。oracle复制当前工作树，未回退到原始d528或旧E1，未提交或修改暂存区。

按MouseBrain→generic bundle→generic flat raw config→synthetic依次迁移，分别完成19/15/22/6个对照用例后才进入下一路径。新增`data_io.configured_preparation.prepare_dataset`及四种显式spec，复用原PreparedDataset，不改U1/U2a/U2b接口。MouseBrain subset/build/preprocess和generic float32/synthetic算法机械迁入data_io；runner旧签名只作薄委托，无两套准备算法。最终MouseBrain与generic训练入口均直接解包PreparedDataset；generic旧dict接口作为legacy view保持。

MouseBrain保留gene_ids、RNA/Metabolite及RNA.obsm已有UNI，B14原section恢复不变；输入顺序与原训练section_order/fallback分别记录。bundle保持model-ready语义、feature-only float32、spatial/processed引用及None顺序语义，不重跑预处理。raw继续原flat schema、HE显式字段及优先级，obs mismatch仍只warn，不新增ATAC reader。synthetic固定seed88、shape/分布/dtype及优先级保持。processed可为None，generic fit原来传None的行为不改。

[U2c实施报告](../../refactor_checks/u2c_mousebrain_generic_20260921/REPORT.md)和[验证结果](../../refactor_checks/u2c_mousebrain_generic_20260921/validation_results.json)记录V0～V10：最终62个入口案例精确对照；真实小型MouseBrain/raw CPU COSIE及Harmony；四路径当前Finit分别160/146/146/160个参数和构造后RNG一致；writer由确定性Stage/fit替身驱动，完整产物/summary对照。raw四模态可准备但Stage原本拒绝，保留原异常并新增合法RNA+Protein初始化用例，未扩展模型组合。

HE/image接口先用真实图像/mask机械核及UNI推理替身核对。发现本地权重及已装依赖具备条件后，按用户明确允许的小fixture例外，另执行32×32合成图像、四patch的真实CPU UNI对照，特征/坐标/PCA精确一致，无下载、无大图。没有让正式MouseBrain已有UNI路径重新推理。该验证范围补充及日志均独立记录，不放宽数值标准。

CRC/HLN/Spleen各all和CSR random共6例，MISAR/Thymus/Simulation各all共3例，以及MISAR helper导入探针全部通过。standalone run_preprocessing真实小型和dry输出/配置解释保持。B12出口函数不改，BF16最终embedding保存对照通过。两个既有数据族、datasets/image_features/preprocessing、model/fit/analysis及现有validator共76个基线Python文件SHA不变；本批只改两个runner并新增295行准备模块。MouseBrain从lambda_schedule起、generic从Stage起到文件末尾逐字不变，epoch0/noOT/zero/周期保存保持。189,442个保护文件路径及lstat元信息清单一致；不冒称全量历史内容SHA验证。

暂留与退出阶段：MouseBrain三个准备helper、generic load_preprocessed_inputs/to_float_tensors/build_synthetic_training_inputs保留薄兼容，当前scripts/tools未发现独立外部准备caller，U7决定API退出。datasets.preprocess_multisection_cosie_style仍供raw与standalone CLI实际调用；build_mousebrain_section仍供准备和既有visualization使用，继续保留。原summary collector/writer留U3，MouseBrain epoch0/完整任务留U4c，generic任务留U4e，CLI/schema统一留U7。不因bundle允许None顺序而反向改变此前两个数据族。

U2d/U3/U4未执行，SPATCH raw/reuse要求仍留U2e。未跑GPU gate、Stage真实forward、真实大规模准备、长期训练或完整analysis；未修改模型数学、历史result/cache、冻结证据或三个gene_imputation目录。B4/R1/R3、旧smoke、裸scale/epochs和generic obs严格性均未处理。原各阶段“当时未执行”结论保留，交付后停止。


## 17．U2d退出记录（2026-09-21追加，保留此前阶段的历史状态）

**U2d完成；Human Embryo准备层已收敛，U2e未执行。** 起点为U2c已验收工作树，55项U2c交付SHA匹配，HEAD仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`。独立oracle复制实际U2c源码；未回到原始d528/922d/E1，未提交或修改暂存区。

按existing manifest→external preprocessed run→raw HESTA→运行模式回归顺序执行。分别14/21/10个初始对照用例通过后才进入下一路径，最终53个唯一入口用例对照通过。新增`data_io.embryo_preparation`，使用ManifestInput、ExternalRunInput、RawHestaInput显式选择来源；prepare_source交接HestaSource，原require_harmony/preprocess_only退出后，prepare_dataset返回既有PreparedDataset。保留两时点是为了不让raw preprocess_only额外读取model-ready、不改变reuse两次加载；不要求所有数据族共用一个物理函数。

`data_io/hesta.py`整文件SHA保持`97392d6c77f56a99989809f558ec85bc98b7008005ce7e812bb201caa4546765`；CSR/QC/sampling/HVG/原total_counts归一化/log1p/SVD/z-score/annotation/memmap/manifest全部不变。原external loader逐AST机械迁入，runner旧函数仅薄委托；三模式的旧字典装配接缝已退出。只改1个现有runner（+22/-40行）并新增163行adapter，其余78个基线Python源码不变。原audit前置步骤和Stage之后逐字保持；没有提取U3/U4任务或保存实现。

RNA-only数组保持原dtype和mmap：未截断feature为COW、spatial只读；external前N仍按每section在原loader中复制，identity/truth文件引用附带同一个slice。raw仍先写manifest再加载，禁止静默复用历史cache。source row/annotation只引用manifest，不新增metadata读取；缺证据标row-order only，truth为not_loaded/not_provided，不把developmental stage当section。processed保持None。require_harmony仍只检查原metadata，raw不增加Harmony。

生命周期复核保留reuse旧`_`变量对第一次spatial mmap的持有：由HestaSource显式保留至原退出时点。弱引用探针验证四种路径的Stage前存活与退出释放一致；同进程验证COW/base/slice/own-data，HDF5打开关闭事件保持。完整53例的21个预期错误、18次Stage前哨兵和14次正常返回分别核对，不把对照中的错误计作训练成功。

[U2d实施报告](../../refactor_checks/u2d_embryo_prepared_20260921/REPORT.md)和[验证结果](../../refactor_checks/u2d_embryo_prepared_20260921/validation_results.json)记录V0～V10：七section真实小型CSR H5AD经历实际seurat_v3 HVG/SVD/z-score，原始与40行采样输出精确一致；raw→manifest→reuse闭环；四组各132参数的当前Finit CPU初值及构造后RNG一致；真实CLI audit/preprocess退出码与2/33文件payload一致；原writer用确定性Stage/fit替身验证，含非空双向prior及QC开关。最初CLI测试脚本的路径归一化假差异及修正证据独立保留，未改生产数值或放宽容差。

U2a–U2c回归覆盖paired六例、MISAR族三例、MouseBrain/generic四例及MISAR helper；原准备接口不变，无data_io→scripts反向依赖。190,568个历史保护文件路径及lstat元信息一致，不冒称全量历史内容SHA。当前计划只追加，不改写U0–U2c“当时未执行”的记录。

暂留：runner.load_external_preprocessed_run无发现的scripts/tools正式外部caller，保留薄兼容签名供旧调用及本批探针，U7决定退出；HESTА原算法接口继续保留。Embryo→CRC两保存器依赖留U3，完整任务/early modes留U4d，CLI/defaults留U7，analysis身份读取留U5。SPATCH raw/reuse双输入要求仍留U2e，本批未开始。

未跑GPU gate、真实Stage forward、真实大规模HESTА准备、长期训练或完整analysis；未修改模型数学、历史result/cache、冻结证据或gene_imputation*目录。B4/R1/R3、旧smoke、scale/epochs/noOT差异均未处理。交付后停止。


## 18．U2e-1退出记录（2026-09-21追加，仅SPATCH cache/reuse结构迁移）

**U2e-1完成；SPATCH reuse/cache路径已完成准备层结构收敛。raw能力仍未提供，U2e-2未执行，因此U2e整体尚未完成。** SPATCH必须最终支持raw/reuse双输入的既定要求保持，不把当前cache-only阶段视为该族最终完成态。

本批起点为U2d已验收工作树，145项U2d交付SHA匹配；用户尚无新checkpoint，HEAD仍`d52829b769ebd06cae25c4ca78e4a02aba7068f6`，暂存区为空。oracle复制实际累计U2d源码，未回到旧提交，未回退/清理/提交。

新增`data_io.spatch_preparation.SpatchCacheInput`及prepare_dataset，正式main直接取得既有PreparedDataset。原canonical_ids、sha256_file、cache_parameters、validate_cache_input_identity、load_preprocessed_cache逐AST迁入权威data_io实现；runner五个旧签名仅薄委托，原schema/identity/loader业务块已退出。cache schema=1、两个section和RNA/Protein/HE顺序、float32/宽度、文件size/SHA、metadata/canonical身份、source证据与metadata复制时点均不变。原cache_info和historical_preprocessing_sources保持原引用/字段，不把当前reader源文件SHA写回历史manifest。

PreparedDataset不重新预处理、不重读raw或annotation；identity为row-order＋外部metadata证据，携带文件/每section slice与原feature_row_identity_limit；truth not_loaded，processed None。当前loader本来使用np.load而非mmap，保持原8次数组加载、3次CSV读取、8次连续化、12次SHA、3次metadata复制。C-order buffer共享和Fortran原copy点不变；DataFrame不被新封装延长持有，weakref释放探针一致。

main仍拒绝build、无cache、invalid cache，--raw仍未知参数，合法cache仍要求GPU。原前置检查/seed/memory时点和Stage/prior/fit/save后半段保持；所有旧raw/preprocessing helper原body保留但main不可达，不启用、不清理、不承诺可以直接当作U2e-2正式raw实现。

[U2e-1报告](../../refactor_checks/u2e1_spatch_cache_adapter_20260921/REPORT.md)及[验证结果](../../refactor_checks/u2e1_spatch_cache_adapter_20260921/validation_results.json)记录V0～V8：55个独立loader/main用例精确对照；现有check_spatch_cache 73例在迁移前后及最终代码通过；两组n_comps=50/5的当前Finit CPU初始化各160参数及构造后RNG完全一致；15个既有数据族入口handoff＋MISAR helper探针通过。正式训练/forward未运行，GPU要求仅在隔离CPU handoff测试中替身处理。

check_spatch_cache只改新权威导入及三处loader receiver，fixture生成函数和所有assert/expected不变；check_fixed_gpu并无SPATCH loader实际调用，仅移除一个未使用runner导入，真实导入与replay导入通过，完整GPU main未执行。验证脚本最初记录序列化失败独立保留，修正类型名称序列化后重新生成before_v2，未放宽数值标准或修改生产以掩盖失败。

本批新增350行adapter，run_spatch为+28/-236；两个validator仅+4/-3和删除1行导入；其余77个基线Python源码SHA不变，无data_io→scripts依赖。192,809个既有保护文件路径及lstat元信息一致，不冒称历史大文件全量内容hash。当前U计划仅追加，不改写此前退出记录。

暂留：canonical_ids/sha256_file仍被旧raw helper使用，cache_parameters仍供resolve、旧builder及checker fixture；load_preprocessed_cache/validate_cache_input_identity旧wrapper没有scripts/tools正式数据业务caller，暂供旧API与本批兼容探针。后续U2e-2审查raw候选，U7决定旧签名退出；原embedding保存依赖留U3，完整任务留U4d。本批未进入U2e-2/U3/U4，未处理B4/R1/R3。

未跑GPU gate；未运行真实大规模SPATCH、长期训练、真实模型forward或完整analysis；未修改模型数学、历史cache/result、冻结fixture/expected/REPORT或gene_imputation*目录。交付后停止。


## 19．U2e-2退出记录（2026-09-21追加；实现已交付，验收未全部关闭）

**U2e-2整体NOT_COMPLETE，不能记为已全部验收。** 六个上游已对齐H5AD起点的SPATCH raw/reuse准备能力已实现；V0～V7、V9～V11通过，V8原writer成功写summary仍被起点已存在的Path序列化错误阻塞。保留此前U2e-1当时cache-only的历史结论，不改写为其已提供raw。

当前HEAD仍`d52829b769ebd06cae25c4ca78e4a02aba7068f6`；实际基线是U2e1累计工作树，81个U2e-2A源码SHA及计划SHA起始匹配，暂存区不变。旧raw算法oracle复制该工作树helper；Finit只使用当前模型，不恢复922d/E1。

按用户冻结决定新增`--input_mode {reuse,raw}`默认reuse；raw必须data_dir，默认新cache为output_dir/preparation_cache，可显式output_cache_dir；overwrite不覆盖cache，reuse invalid/missing绝不fallback。raw输入仅两section RNA/Protein/已有HE特征H5AD，不新增image/mask/weight/patch参数，不调用UNI图像推理。

新增data_io.spatch_raw，将已核实inspect/轻量reader/sequential/schema1 writer等机械迁移；SpatchRawInput经新cache写出、释放原准备结果、原strict loader后返回既有PreparedDataset。Linux renameat2(RENAME_NOREPLACE)提供发布时排他保护，不退回replace。原schema/identity loader逐AST不变，Stage构造至runner EOF逐字不变，模型/Finit/fit/预处理数学/analysis及其它数据族不改。

[本批报告](../../refactor_checks/u2e2_spatch_raw_20260921/REPORT.md)与[逐项验收](../../refactor_checks/u2e2_spatch_raw_20260921/validation_results.json)记录：510spot、4828共同gene、17Protein及2048HE的三组真实CPU对照（Harmony/noHarmony/关闭HVG小维数）；raw→新cache→strict reload逐值一致；三raw及两reuse各160参数当前Finit初值/RNG一致；21例错误足迹、实际参数边界、16项发布/CLI（含实际双进程竞争）、55例reuse精确对照、现有checker73例和其它族16项最小回归通过。没有跑真实训练或GPU。

V8确定性Stage/fit替身抵达原writer后，新旧reuse/raw均因`json_safe`不处理resolved_config里的Path而在summary写出抛`TypeError: Object of type PosixPath is not JSON serializable`。完整summary payload、embedding/history和失败足迹已比较一致，但**不将共同失败当成功保存**；该成功路径尚未通过，因此整批不宣布完成。最终保存器及共享json_safe本批禁止修改，未越界修复；留待单独决定，不进入U3。

本批新增437行raw模块、adapter+21/-4、runner+44/-345、validator仅import/guard+2/-1；其余78份源码SHA不变。已有wrapper退出/CLI整理仍留U7，embedding保存依赖留U3、完整任务留U4d。过程中的gc导入遗漏已被reuse回归捕获并恢复，失败证据保留；负hvg旧接受行为未升级raise。

保护范围、历史result/cache/fixture/expected/REPORT及gene_imputation*未改；本次所有fixture和cache只写新的u2e2验证目录。未测真实百万spot RAM、耗时和13GB来源SHA IO，未跑GPU gate、长期训练、真实forward或完整analysis。未处理B4/R1/R3，未修改模型数学；U3未执行。交付后停止。


## 20．U2e-2最终验收记录（2026-09-21追加；独立JSON Path修正）

**U2e-2当前状态由第19节当时的NOT_COMPLETE更新为COMPLETE。** 第19节、原REPORT正文及原验证失败证据保留，不改写历史。U3未执行。

用户授权单独correctness批次后，发现全部Path均来自既存summary的resolved_config.runner四字段（data_dir、input_identity_manifest、output_dir、preprocessed_cache_dir），并非新增raw provenance。因此只在SPATCH当前run_summary writer加入os.PathLike→os.fspath default；其它unsupported对象仍TypeError，不使用default=str，不改training.fit.json_safe或任何其它保存器。scripts/run_spatch.py +10/-1，其余81份源码SHA保持。

[独立修正报告](../../refactor_checks/u2e2_json_path_fix_20260921/REPORT.md)与[最终门槛](../../refactor_checks/u2e2_json_path_fix_20260921/validation_results.json)记录：nested Path、纯JSON-safe、unsupported类型、完整字段/类型/顺序检查通过；V8原四例及两例增补通过，四次原summary写出成功，embedding/history/metadata和Stage前输入/RNG一致。原Stage/prior/fit用测试替身验证writer，raw preparation为本次小型真实CPU流程，不视为真实训练。V0/保护检查重新执行；Stage至EOF仅此summary default有差异，其它行为保持。原V1～V7、V9～V11来源代码/证据未受影响、仍适用；原Finit CPU参数/RNG通过记录被保留，未冒称本修正重新构造验证。

SPATCH准备层raw/reuse在本轮冻结定义下完成：raw从两section的RNA、Protein、已有HE/UNI特征六H5AD开始，经sequential→全新schema1 cache→释放中间结果→strict loader→PreparedDataset。无原图UNI推理能力新增；默认reuse、无invalid fallback、无cache覆盖保持。真实百万spot资源/13GB SHA IO/GPU数值仍属后续规模验证。本批不执行真实训练、完整analysis、GPU gate或U3；模型/Finit/fit/preprocessing/analysis与历史cache/result/fixture/expected不改。暂留保存依赖U3、完整任务U4d、CLI整理U7未提前实施。完成后停止。


## 21．U2 总退出记录（2026-09-21追加；不进入 U3）

**U2 完成。** U1～U2e-2（含独立 PathLike summary 修正）完成后的当前工作树，九数据集＋generic 的15条主输入路径均有显式 spec/adapter→同一 PreparedDataset 逻辑结果：CRC/HLN/Spleen；MISAR/Thymus/Simulation；MouseBrain；generic bundle/raw/synthetic；Embryo raw/manifest/external；SPATCH reuse/raw六H5AD。统一逻辑合同，不要求建立一个全局dispatcher或把全部算法放在同一文件。

[U2总收口报告](../../refactor_checks/u2_final_acceptance_20260921/REPORT.md)、[最小handoff结果](../../refactor_checks/u2_final_acceptance_20260921/handoff_results.json)及[最终验收](../../refactor_checks/u2_final_acceptance_20260921/validation_results.json)记录23个小型接口用例＋一个MISAR helper回归通过。覆盖正式入口/显式分派、九字段、section顺序、identity/truth/provenance和引用；包含generic HE来源及bundle缺省证据扩展。Stage前哨兵阻止模型构造，COSIE/HESTА raw/sequential数值计算使用明确替身，SPATCH inspect/新cache writer/strict reload实际执行；未把接口探针当真实训练或新的预处理数值证明。各小批已通过的真实CPU/Harmony/Finit/错误/保存证据继续有效，本轮未重复重型测试。

没有生产代码新增、删除或修改。当前82份源码SHA保持；U1起点39份model/training/analysis及关键数据数学源码SHA仍相同，B13/B14/B12/B11与Finit保留。PathLike default仅在SPATCH原summary边界，未扩展为全局default=str。原U2e-2的NOT_COMPLETE与V8失败JSON仍保留为历史；第20节及独立Path批次是其最终COMPLETE的明确更新，不覆盖旧证据。

U2临时债务已关闭：CRC过渡旧准备块、MISAR旧准备块/准备helper→CRC依赖、MouseBrain/generic旧完整准备实现、Embryo旧输入映射、SPATCH旧cache/raw业务实现均已由公共权威边界替代。没有找到计划明确要求U2退出但仍残留的完整双实现，故不做额外清理。

剩余依赖明确登记：U3处理目录/轻量与最终artifact保存及MISAR/Embryo/SPATCH→CRC保存器；U4a/b/c/d/e处理原caller族任务编排、资源/early mode/epoch0等；U7决定CRC/MISAR/MouseBrain/generic/Embryo/SPATCH薄兼容API、旧SPATCH不可达validate_source_stats、parser/default依赖、公共类型注解和模块导出。当前有caller的保存/validation接口不提前删，历史探针不冒称生产caller。standalone preprocessing与专用HESTА/UNI/SPATCH算法保留；分析输入工作流不作为训练双实现清理。

PreparedDataset已统一逻辑字段，但不是统一磁盘/JSON/嵌套标签schema。generic section_order可None、MouseBrain原sorted fallback、可选processed、分族identity/truth证据等级及transitional compatibility_context均在报告逐项解释并保持。没有新增读标签/造barcode/给PCA轴伪造gene名称，未把warn升raise或改变原模型合法模态组合。

205,361个既有保护文件路径/lstat保持，历史cache/result/fixture/expected/REPORT/known failures及gene_imputation*不改；仅本计划追加且原前缀保持，HEAD/index不变。SPATCH raw仍从六个上游已对齐H5AD开始，不支持本次新增原始图片UNI推理；百万spot资源、13GB source SHA IO与GPU数值未验证。本轮未构造模型、运行训练/forward、完整analysis或GPU gate，未处理B4/R1/R3。U3未执行，U2在上述范围完成，停止。


## 22．U3a退出记录（2026-09-21追加；CRC系公共artifact）

**U3a COMPLETE；U3b/U3c未执行。** 以U2 final已验收累计工作树为起点，47项U2交付SHA匹配，HEAD仍为`d52829b769ebd06cae25c4ca78e4a02aba7068f6`、index为空；未将旧HEAD视作U2完整快照，没有回退或提交。

新增`training/artifacts.py`，机械承接CRC的ensure_dir、save_list、indexer_to_numpy、save_selected_spot_indices、save_spatial_arrays、save_final_embeddings、save_ot_prior_topk七函数。七函数AST与本批oracle一致，仅backed_rna类型注解由AnnData改Any。CRC旧签名薄委托；MISAR/Embryo/SPATCH保存导入改为公共层，MISAR三本地机械helper薄委托，原map(str)差异保留。HLN/Spleen随CRC pipeline、Thymus/Simulation随MISAR pipeline实际调用公共writer，不新增runner业务库。

文件模板、prior方向/metadata/QC、dtype/顺序、返回路径、flags、summary payload和写入/失败时点保持；不建新schema或统一summary serializer。公共embedding复用B12 tensor_to_numpy，公共prior复用原fit.json_safe，没有第二套转换。SPATCH PathLike default仅原summary边界，prior/spatial原BF16错误与原slice/list接受语义未夹带修正。

[U3a报告](../../refactor_checks/u3a_artifacts_crc_family_20260921/REPORT.md)、[验收结果](../../refactor_checks/u3a_artifacts_crc_family_20260921/validation_results.json)记录V0～V9全部通过：48个writer用例（含15个预期异常）；CRC系10、MISAR系10、Embryo4、SPATCH6个caller对照；运行时profile确认调用公共writer；现有SPATCH checker73例；九数据集/generic23个新handoff＋MISAR helper。完整值/类型/key顺序/summary比较，SPATCH gzip MTIME及其真实摘要单列核对，无整块忽略provenance。checker第一次缺PYTHONPATH的启动失败日志保留，正确路径重试通过。

本批只新增120行公共模块并修改四runner（CRC+15/-70、MISAR+8/-11、Embryo和SPATCH各+1/-1）；其余78份生产/validator源码SHA保持。除导入和薄wrapper外四runner整个AST一致，PreparedDataset/data_io、model/Finit/fit/preprocessing/analysis、CLI/defaults均不动。205,778个既存保护文件路径/lstat保持，本计划旧前缀不改；历史result/cache/fixture/expected/报告不改。

暂留CRC六个本地pipeline保存wrapper、MISAR三个准备/obs metadata机械wrapper，U4a/b迁caller时可直调，U7决定旧API退出；CRC indexer wrapper无scripts/tools正式caller，仅旧API兼容待U7。MISAR/Embryo/SPATCH已不从CRC获取保存业务。HLN/Spleen→CRC及Thymus/Simulation→MISAR的完整任务/默认值依赖仍属U4/U7；MouseBrain保存U3b、generic周期/weights/history U3c，未提前迁移。

本轮仅CPU小fixture/原writer及确定性Stage/fit/预处理替身，未构造真实模型或重跑Finit初始化；旧U2数值证据不受本批变化影响，未冒称新训练数值通过。未跑真实训练/forward、完整analysis、百万spot、UNI推理或GPU gate，未修改科研数学。SPATCH raw/reuse定义不变，raw仍为六个上游已对齐H5AD。U3a交付后停止，不进入U3b/U3c。


## 23．U3b退出记录（2026-09-21追加；MouseBrain artifact）

**U3b COMPLETE；U3c未执行。** 起点为U3a累计完成态，46项U3a交付SHA匹配；HEAD仍`d52829b769ebd06cae25c4ca78e4a02aba7068f6`，index为空，未回退或提交。oracle冻结当前83份源码，未回到旧HEAD推断重构成果。

MouseBrain的embedding/prior数组和metadata、summary/history机械写出已使用`training/artifacts.py`。公共embedding新增可选filename_template（默认保持CRC），MouseBrain继续写`final_embeddings/{section}_final_embedding.npy`。公共prior新增显式skip_incomplete/direction_meaning/note/metadata_transform/prepare_directory参数，分别保持MouseBrain缺字段KeyError、短note、不默认增加direction_meaning、原json_safe与原目录语义；无dataset隐式分派。新增save_json只负责原UTF-8/indent写出，caller选择已有transform；summary传原MouseBrain.json_safe、history不投影，不使用default=str或PathLike扩展。

MouseBrain原json_safe的ndarray→shape、部分NumPy scalar不支持行为保留，未替换为fit.json_safe的值列表语义。B12 embedding仍只用原tensor_to_numpy，BF16在NumPy边界升FP32、其它dtype不变；prior原BF16错误不夹带修正。MouseBrain仍决定payload、配置复制、路径记录、prior开关/目录及保存时点；没有新增embedding/summary/history CLI。SPATCH PathLike仅原summary边界，raw/reuse不改。

[U3b报告](../../refactor_checks/u3b_mousebrain_artifacts_20260921/REPORT.md)、[验收汇总](../../refactor_checks/u3b_mousebrain_artifacts_20260921/validation_results.json)记录V0～V9全部通过：38个MouseBrain writer用例＋8个执行哨兵（共19个预期错误）新旧精确一致；epoch0 train-mode/no_grad、fit flags、lambda schedule、零epoch及三RNG/保存事件顺序保持。U3a公共48个writer、30个caller对照重新通过，U2九数据集/generic23个新handoff＋MISAR helper通过；replay实际import及相关静态条件通过。调度器最初遗漏worker参数的启动失败日志保留，修正仅限本批新验证代码。

只改training/artifacts.py（120→142行，+32/-10）和run_mousebrain.py（718→678行，+14/-54）；无生产增删文件。其余81份源码SHA不变，run_mousebrain整个AST及非writer代码/CLI/defaults/JSON投影不变。Stage/Finit/prior计算/fit/preprocessing/PreparedDataset/analysis协议未动；通用保存不反向依赖scripts或data_io。209,910个历史保护文件路径/lstat保持，计划只追加，不改写U3a当时状态。

暂留MouseBrain.save_embeddings布局委托（caller为save_run_artifacts）、save_ot_prior_topk策略委托（caller为dry/train），无第二套保存算法，U4c迁任务时可直调、旧API退出由U7决定。save_run_artifacts继续承担当前caller的配置复制/payload/时点组织，属于U4c边界；json_safe保留其真实不同语义。当前validation未发现直接借用MouseBrain保存业务，replay只import及检查epoch0/fit。generic保存U3c未迁。

全部测试使用CPU小fixture和确定性Stage/fit/预处理替身；未构造真实Finit模型或运行真实训练/forward、完整analysis、百万spot、UNI推理、GPU gate。旧初始化记录不冒充本批新数值测试。未修改模型数学、历史result/cache/fixture/expected/报告或gene_imputation*。U3b交付后停止，不进入U3c。


## 24．U3c退出记录（2026-09-21追加；generic periodic/final artifact）

**U3c COMPLETE；U4尚未执行。** 以U3b累计完成态为起点，28项U3b交付SHA匹配，83份当前生产/validator源码作为oracle；HEAD仍`d52829b769ebd06cae25c4ca78e4a02aba7068f6`、index为空，没有回退、提交或清理。

generic的save_training_artifacts已将weights/history/training_summary/可选embedding机械写出委托training.artifacts。公共模块只增加5行save_weights(path,payload)，原样torch.save，不增加resume字段；其余原公共函数AST不变。generic仍提供原五键payload（model_state_dict/config/history/section_order/input_dims）、文件名和保存条件，state_dict的dtype/metadata/顺序保持。summary投影通过已有save_json.transform在原文件打开后求值，失败足迹不被提前求值改变；history不加json_safe，None仍写null。原generic.json_safe与B12边界保持，不推广SPATCH PathLike或default=str。

train_stage_model及全部非writer函数AST保持。periodic仍由iter_fit_model非final yield触发：epoch训练前向输出、weights已step且可能已refresh，覆盖同名weights/history/summary、不导出embedding，不另加forward/eval。final仍在迭代器耗尽后使用final_outputs，embedding按原save_embeddings或smoke决定。noOT、零epoch、allow_empty_epochs、refresh_ot、yield_every和最终eval均不变；未增加optimizer/scaler/RNG状态，不宣称weights可exact resume。

[U3c报告](../../refactor_checks/u3c_generic_artifacts_20260921/REPORT.md)、[验收汇总](../../refactor_checks/u3c_generic_artifacts_20260921/validation_results.json)记录V0～V10全部通过：generic31个writer＋12个runner替身用例（20个预期异常）完整值/类型/字段/ZIP成员/足迹对照；多yield逐次快照验证periodic和final不同输出，fit实参和三RNG/事件序列一致。U3a fresh48 writer＋30 caller、MouseBrain fresh46例、U2 fresh23 handoff＋MISAR helper回归通过。新记录器首轮不能编码故意unsupported object，失败日志/输出保留；只改本批测试编码，正式结果另存before_v2/after_v2，未修生产JSON以绕过错误。

生产仅training/artifacts.py +5/-0（142→147行）和scripts/train_stage_model.py +18/-18（319→319行）；无生产文件增删，其余81份源码SHA保持。model/Finit/fit/PreparedDataset/preprocessing/analysis/CLI/defaults及SPATCH raw/reuse不动，B13/B14/B12/B11保持；公共层不反向依赖runner。213,140个历史保护文件路径/lstat保持，计划只追加、原前缀不变，历史result/cache/fixture/expected/REPORT与gene_imputation*不改。

暂留generic.save_training_artifacts作为原periodic/final两处caller的布局/payload适配与公共核委托，无torch.save/open/json.dump/np.save第二套算法；U4e迁任务时可调整，U7决定旧API退出。当前scripts/tools未发现外部保存业务caller，不为历史probe改旧证据。json_safe/summarize_outputs是现有payload语义而非待删writer。未把periodic与final收敛为一种业务语义。

本轮只用CPU小fixture、真实保存器和Stage/iter_fit/预处理替身；fit中实际step→refresh→yield及final eval顺序另做未改源码静态核实，替身不证明真实训练/OT数值。未构造真实Stage/Finit、运行真实训练/forward、完整analysis、百万spot、UNI推理或GPU gate。U3c交付后停止，不进入U4。


## 25．U3总退出记录（2026-09-21追加；总门槛未通过，尚未退出）

**本轮总收口验收已执行；U3全局状态NOT_COMPLETE，U4未执行。** U3a/U3b/U3c此前各自验收完成的历史结论保持，本次不改写，也不把兼容回归通过等同于所有保存机械实现已收敛。

起点为当前U3c完成态，37项U3c交付SHA匹配，83份源码冻结，HEAD仍`d52829b769ebd06cae25c4ca78e4a02aba7068f6`且index为空。没有生产修改或删除，没有回退/提交。相对U3a起点，六runner仅有既知保存增量、新增artifacts，其余76源码SHA保持；六个主训练函数AST与U3a开始前相同。

[总验收报告](../../refactor_checks/u3_final_acceptance_20260921/REPORT.md)、[逐项结论](../../refactor_checks/u3_final_acceptance_20260921/validation_results.json)记录明确缺口：U3-R1 Embryo正式train分支仍直接torch.save三键model_checkpoint.pt；U3-R2 CRC/MISAR summary/history仍直接open/json.dump；U3-R3 Embryo.write_json和SPATCH history/summary/failure仍直接write_text(json.dumps)。运行时Embryo小型train替身实际写出checkpoint且未调用公共save_weights，证实不是历史死代码。这些当前可达写出不能通过删除无caller wrapper收口，也不能标为已关闭。

已迁移的embedding/spatial/indices/list/prior/QC公共实现保持；MouseBrain和generic已使用相应公共核，runner→runner保存业务import/别名调用退出，完整pipeline/defaults依赖仍属U4/U7。CRC indexer旧API无scripts/tools直接caller，但计划已指定U7决定，不能仅据仓内无caller证明删除不影响旧接口；本轮未找到可安全按U3债务直接删除的wrapper。准备audit/obs CSV与辅助JSON仍在caller，需要明确机械边界，不能把专用payload和读取整体迁入通用writer。

本轮34个选定跨族artifact旧/新对照通过（公共12、CRC族5、MISAR族3、Embryo2、SPATCH2、MouseBrain5、generic5），另有23个fresh PreparedDataset handoff＋MISAR helper通过。覆盖BF16/JSON/PathLike、flags与缺失路径、MouseBrain epoch0、generic周期/最终差异，值/类型/顺序/路径/错误/RNG精确一致。SPATCH gzip时间头及其真实关联摘要单列核验，未忽略整个summary/provenance。全部是小fixture与Stage/fit/预处理替身或真实小型writer/loader，不是训练数值新证明。

B12、SPATCH局部PathLike、MouseBrain布局/JSON投影、genericperiodic/final/noOT/零epoch保持。model/Finit/fit/data_io/PreparedDataset/analysis源码在U3过程无额外变化。217,023个历史保护文件路径/lstat保持；历史result/cache/fixture/expected/报告及gene_imputation*不改，计划旧前缀保留。没有运行真实训练/forward、完整analysis、百万spot、UNI推理或GPU gate。

后续需独立补齐保存余项后再验收：Embryo weights原payload委托；CRC/MISAR流式JSON；Embryo/SPATCH显式先序列化再写入的buffered JSON语义（保留ensure_ascii、局部PathLike、异常前是否创建/截断文件及原时点）；明确CSV/JSONL边界。当前save_json先open再transform，不能直接套给write_text(json.dumps)而改变失败足迹。不修改schema、CLI/defaults、Stage/prior/fit或U2准备。本轮不实施补充迁移、不进入U4，交付后停止。


## 26．U3d退出记录（2026-09-21追加；剩余保存边界完成）

**U3d COMPLETE。** 以U3c完成且总验收发现R1–R3的当前累计工作树为起点；17项前一交付SHA匹配，HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6、index为空。原第25节“当时未完成”保留，不改写历史。

[U3d REPORT](../../refactor_checks/u3d_artifact_residuals_20260921/REPORT.md)及[验证结果](../../refactor_checks/u3d_artifact_residuals_20260921/validation_results.json)：Embryo三键model_checkpoint.pt在原train/fit之后时点委托artifacts.save_weights；CRC/MISAR summary/history用原streaming语义；Embryo.write_json改为单次buffered委托；SPATCH history/summary/failure用显式buffered写出。save_json只增加buffered、ensure_ascii、default策略，SPATCH summary独自使用原PathLike default，不用default=str，不改变其它caller接受行为。

生产仅5文件：artifacts +26/-5、CRC +2/-4、MISAR +2/-4、Embryo +5/-4、SPATCH +6/-10；无新增/删除生产文件，其余78份源码SHA不变。四caller整模块AST在只还原writer接缝后与起点相同。B13/B14/B12/B11、Finit、model、fit、PreparedDataset、preprocessing、analysis、CLI/defaults、raw/reuse均保持。

V0～V8全部通过：78接缝（含5weights）、48公共writer、10paired、10MISAR族、4Embryo＋5early/failure、6SPATCH、46MouseBrain、43generic，共250个分组小型保存对照；另23个fresh PreparedDataset handoff＋MISAR helper。保留JSON全文/顺序/ASCII/PathLike、streaming部分文件与buffered不截断、dtype、flags、原summary路径及完整替身时序/RNG。SPATCH新fixture一次旧绝对路径导致identity拒绝，仅修本批副本并留失败证据后重跑；历史fixture/expected不动。

Embryo.write_json剩余caller为run的audit/early/history/summary及main failure，现为薄委托；U4d/U7可调整接缝。原CRC/MISAR薄委托、MouseBrain/generic布局适配继续按U4a/b/c/e及U7登记，CRC无仓内直接caller的indexer旧API仍留U7明确决定。没有新增第二套writer或删除未知外部API。未运行真实模型/训练、完整analysis、百万spot、UNI推理或GPU gate。


## 27．U3最终总退出记录（2026-09-21追加；U3完成）

**U3 COMPLETE，U4未执行。** [最终验收追加记录](../../refactor_checks/u3_final_acceptance_20260921/U3D_FINAL_ACCEPTANCE.md)承接第25节旧失败结论；旧REPORT不覆盖。R1–R3经U3d修正后重跑直接受影响的保存验证，全部通过。

模型运行artifact的embedding、spatial、indices/list、双向prior/QC、weights、history和summary通用机械核均唯一位于training/artifacts.py。caller只决定layout、payload、flag、编码/投影策略和调用时点；runner之间无保存业务import/别名调用。专用preparation audit/obs CSV、adapter audit JSON、JSONL日志、输入cache/manifest、analysis/batch状态保留各自原边界，不扩大本批两类余项为全仓存储框架，也不声称这些物理写出全已迁入training。

SPATCH PathLike仅原summary责任边界；B12 BF16仅NumPy转换边界升FP32；Embryo三键weights、MouseBrain布局/epoch0、generic五键weights与periodic/final原语义均保持。完整pipeline/defaults依赖是U4/U7任务，不冒称已全部解除runner依赖。剩余wrapper有真实caller或既定旧API退出阶段，不留未解释的U3权威writer重复。

本批218,905个历史保护文件路径/size/mtime_ns/inode一致，原历史report/result/cache/fixture/expected/gene_imputation*未改。计划仅追加；HEAD/index不变。U3累计非caller的76份原源码SHA不变，model/Finit/fit/data_io/analysis数学协议不动；既有真实CPU准备和Finit证据仍适用，但未称本轮重跑。当前总门槛通过后停止，不进入U4。


## 28．U4a退出记录（2026-09-21追加；CRC/HLN/Spleen公共训练任务完成）

**U4a COMPLETE，U4b未执行。** 起点是U3最终完成累计工作树，23项U3d交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6、index为空，不把该HEAD当作全部增量已提交。读取原第7节及U3最终追加记录后，严格按CRC→HLN→Spleen迁移，每个caller先完成17项独立对照才进入下一个。最终Namespace配置引用明确后又按同顺序完成三者最终对照。

[U4a REPORT](../../refactor_checks/u4a_crc_family_training_task_20260921/REPORT.md)、[结果](../../refactor_checks/u4a_crc_family_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4a_crc_family_training_task_20260921/static_results.json)。新增training/paired_task.py与scripts/paired_entry.py；仅修改CRC/HLN/Spleen三个runner，其余80份起点源码SHA不变。

公共 `run_paired_training_task(run_config, prepared, *, memory_monitor, protein_var_names, status_prefix)` 只接已resolved配置和PreparedDataset及旧monitor/轻量身份引用，不parse/resolve/seed/读raw。Namespace沿用run_config.runner原字典，不复制配置权威；protein_var_names仅是两个已加载Index引用，用于旧summary，不改PreparedDataset字段。三者训练行为相同，无新增ExecutionPolicy、dataset-name dispatcher、registry或trainer框架。

scripts.paired_entry承担原CLI验证、seed/default_rng、目录/monitor、准备hooks和finally资源所有权，调用既有data_io.prepare_paired_dataset后进入公共任务；避免任务读raw，也避免HLN/Spleen继续借CRC业务pipeline。旧CRC prepared解包→Stage/prior→dry/fit→导出/summary业务块已退出，公共任务对应整块AST除显式Protein Index来源外原样；验证/seed/hooks/prepare调用及finally AST保持。fit唯一epoch循环、artifacts唯一writer不动。

V0～V11全部通过：主矩阵CRC/HLN/Spleen各17例，共51例；补6例summary/history写失败、6例AMP映射/CPU非法AMP。三caller各在隔离CPU进程真实构造当前Finit，146个named parameter逐name/shape/dtype/值及构造后RNG完全一致；真实initial blockwise candidate-sparse UOT与固定权重dry前向/出口一致，dry保持train-mode+no_grad且一次forward。完整resolved配置、prepared引用/身份/section、seed/RNG、prior断言、fit实参、artifact文件/字段/顺序/flags和错误足迹保持；Namespace类型与共享runner字典引用有断言。

真实prior的candidate_search_time_sec/sparse_uot_time_sec/total_prior_time_sec逐路径列为时间观测差异，其余数值不放宽容差。初次dense fixture触发旧fancy indexing错误，仅新建合法tiny CSR fixture继续任务验收，未修预处理；旧失败证据保留。训练/AMP映射用替身，AMP测试可用性替身不代表GPU执行。fit数学/refresh/final eval源码及调用实参不变，没有本轮真实训练epoch通过声明。

另有23个fresh U2交接＋MISAR helper、48个公共writer对照、跨族28个小型artifact文件对照及current replay/check_fixed_gpu仅import/helper身份回归。没有运行长期训练、完整analysis、百万spot或GPU gate；真实CPU小型forward本轮确已执行。B13/B14/B12/B11/Finit、SPATCH raw/reuse、PathLike边界、model/fit/PreparedDataset/preprocessing/analysis/artifacts不改。

HLN/Spleen业务编排不再调用CRC；仅保留crc.parse_args/resolve_run_config、Spleen→HLN defaults至U7。CRC.run_crc_pipeline保留本main/旧签名适配，旧save/helper薄兼容及validator经CRC引用fit/forward也登记U7，不擅自删除旧API、不借本批迁MISAR等其它任务。222,604个既有保护路径/lstat保持，历史result/cache/fixture/expected/REPORT与gene_imputation*不变，计划只追加。完成后停止，不进入U4b。


## 29．U4b退出记录（2026-09-21追加；MISAR/Thymus/Simulation公共训练任务完成）

**U4b COMPLETE，U4c未执行。** 以当前U4a完成累计工作树为起点，25项U4a交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6、index为空。旧第28节“当时U4b未执行”保留。先MISAR 21例验收，再Thymus 27例，再Simulation 26例；最后补齐原finally期间模型引用生存期，并按同顺序重跑全部74例，最终收据与当前源码SHA匹配。

[U4b REPORT](../../refactor_checks/u4b_misar_family_training_task_20260921/REPORT.md)、[验证结果](../../refactor_checks/u4b_misar_family_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4b_misar_family_training_task_20260921/static_results.json)。新增training/task_runtime.py、training/multisection_task.py、scripts/multisection_entry.py；仅改U4a paired_task及三个MISAR族runner，其余81份起点源码SHA保持。

公共execute_prepared_task接已resolved配置、PreparedDataset、原monitor及5字段ExecutionPolicy；Stage/prior/dry/fit/输出检查/export只有一个执行核，paired/multisection facade分别保留旧summary布局。CRC明确paired checks/train-mode dry/详细memory recorder；MISAR族明确adjacent checks/eval-mode dry/原memory kwargs。没有dataset-name分派、第二套epoch循环、writer、callback框架或trainer体系。Namespace沿原run_config.runner字典，不二次resolve。

multisection_entry保留原validate、seed/default_rng、prepare hooks和finally；数据算法仍在U2 data_io。两个adapter的audit仍在resolve后、entry/Stage前；Thymus coords/17-marker与Simulation factor/truth/obs metadata不改。entry的普通execution_state只持有原model/prior/outputs引用直到finally结束，无复制/物化/新清理回调。MISAR原Stage/prior/fit/export业务块已退出；Thymus/Simulation main直接调用共享entry，不再借MISAR业务pipeline。

V0～V12全部通过。三caller各实际CPU构造当前Finit，62个named parameter逐name/shape/dtype/值及构造后RNG一致；真实blockwise candidate-sparse prior方向分别6/6/8，固定权重dry前向/出口一致。MISAR顺序dataset4→3→2→1保持，三者dry eval+no_grad且一次forward。全部fit实参、AMP/chunk/checkpoint/refresh/epochs映射、audit/资源/轻量与最终artifact时点及错误足迹保持。fit/AMP映射使用替身，未运行真实训练epoch或GPU。只有预先声明的prior三个时间叶子及临时输出根单列差异，不忽略整个summary/provenance，不放宽数值容差。

另有CRC/HLN/Spleen各17例fresh U4a回归（含真实Finit/prior/固定前向及train-mode dry），23个U2 handoff＋MISAR helper、48个公共writer case和28个跨族artifact文件对照。U2本轮为接口/身份/引用与最小接缝回归，预处理/UNI/HESTА部分用明确替身，不冒充重新运行真实预处理。replay/check_fixed_gpu仅import/helper身份核对。

残留依赖明确留U7：Thymus/Simulation仍用MISAR parse_args/resolve_run_config；MISAR.run_misar_pipeline只供本main及旧签名兼容，不含业务计算。旧helper/summary导入别名与data_io薄wrapper保留，权威实现分别位于entry、training、data_io；无新跨runner保存或任务业务依赖，不贸然删除未知外部API。

model/Finit、fit、artifacts、PreparedDataset/全部data_io、preprocessing、analysis、CLI/defaults、B13/B14/B12/B11及SPATCH raw/reuse/PathLike均保持。228,214个既有保护文件路径/lstat不变，历史result/cache/fixture/expected/REPORT与gene_imputation*未改，计划只追加。未运行长期训练、完整analysis、百万spot或GPU gate；真实训练/GPU综合gate留后续U4/U8。U4b交付后停止，不进入U4c。


## 30．U4c退出记录（2026-09-21追加；MouseBrain公共训练任务完成）

**U4c COMPLETE，U4d未执行。** 起点为U4b完成累计工作树，94项U4b交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，分支l0-original-20260717、index为空。旧第29节“当时U4c未执行”保留，不改历史结论。

[U4c REPORT](../../refactor_checks/u4c_mousebrain_training_task_20260921/REPORT.md)、[结果](../../refactor_checks/u4c_mousebrain_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4c_mousebrain_training_task_20260921/static_results.json)。新增training/mousebrain_task.py；只修改MouseBrain runner、training/task_runtime.py及replay_model的静态位置检查，其余85份起点源码SHA不变。

MouseBrain的Stage/prior→无条件epoch0→dry或fit→export/summary已退出runner。公共run_mousebrain_training_task接已resolved对象引用包和PreparedDataset；引用包明确区分原runner Namespace、原可变model_config、原resolved_config保存快照，不二次resolve或deepcopy。lambda schedule仍在runner原时点解析一次，原list/None直接交给fit，没有重定义schedule层。

Stage/prior使用与CRC/MISAR相同的initialize_prepared_task，MouseBrain不套额外modality断言/monitor。无条件run_epoch0_preflight显式Epoch0Policy(eval_mode=False)，在dry分支和fit之前执行：train-mode、no_grad、epoch=0，一次原model调用。不是普通dry、未移入fit、不增加eval/forward。正常训练、FP32 execution、allow_empty_epochs=True、clear_step_state=False、record_elapsed_time=False、chunk/checkpoint/refresh/final eval及U3布局保持。CRC/MISAR仍丢弃初始完整prior返回，MouseBrain保留它供原summary，不扩大引用生存期。

V0～V11全部通过：MouseBrain38个writer＋26个task用例；四个真实小型CPU案例各116 named parameter精确一致，epoch0实际34次Dropout，train/no_grad、完整输入/输出和前后Python/NumPy/Torch RNG逐项一致。真实prior四相邻方向保持；固定权重dry和训练handoff一致。零epoch实际进入shared fit并执行原final eval，没有optimizer训练步；非零epochfit为明确替身，不冒称真实训练数值通过。

另有CRC/HLN/Spleen/MISAR/Thymus/Simulation各5例fresh任务回归，含真实Finit/prior/固定前向；23个U2handoff＋MISAR helper、28个U3跨族artifact文件对照。仅预先声明的三个prior耗时叶子及临时根路径规范化，不忽略整个summary/provenance，不放宽浮点容差。首轮新probe缺confidence在旧路径就失败，修正仅本批小prior替身并重跑；旧失败证据留存、生产校验不改。

replay_model.static_refresh_audit现在沿runner→public task→epoch0 helper检查无条件调用、显式非eval policy、no_grad、epoch0及唯一shared fit；实际执行本静态检查，未跑GPU replay。数值replay/refresh算法、旧expected基线保持。原layout/json_safe/summary辅助函数机械迁入task并由runner导入兼容；writer仍唯一artifacts。旧parser/defaults/prepare/helper别名退出留U7，不新增runner业务依赖。

model/Finit、fit、artifacts、PreparedDataset/全部data_io、preprocessing、analysis、CLI/defaults、B13/B14/B12/B11及SPATCH raw/reuse/PathLike未改。243,116个既有保护文件路径/lstat一致，历史result/cache/fixture/expected/REPORT/known failure及gene_imputation*不动。未运行长期训练、完整analysis、百万spot或GPU gate，真实规模与训练GPU综合gate留后续U4/U8。完成后停止，不进入U4d。


## 31．U4d-1退出记录（2026-09-21追加；Human Embryo公共训练任务完成）

**U4d-1 COMPLETE；SPATCH未迁移，U4d-2未执行。** 起点为U4c完成累计工作树，96项U4c交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，分支l0-original-20260717、index为空。保留第30节“当时U4d未执行”。

[U4d-1 REPORT](../../refactor_checks/u4d1_embryo_training_task_20260921/REPORT.md)、[验证结果](../../refactor_checks/u4d1_embryo_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4d1_embryo_training_task_20260921/static_results.json)。新增training/embryo_task.py，仅修改Embryo runner和training/task_runtime.py，其余87份起点Python源码SHA不变。

run_embryo_training_task接已resolved配置和PreparedDataset，Namespace沿用runner字典，feature/spatial/section/manifest沿用引用。显式SingleModalityPolicy(initial_uot=not disable_uot,dry_eval=True)，无dataset-name分派、第二套fit或writer。共享initialize_prepared_task增加默认False的defer_prior窄分支，使Embryo保持Stage→RNA-only断言→monitor/start→optional initial prior的原顺序；原CRC/MISAR/MouseBrain默认路径AST保持。

runner原Stage/prior/dry/fit/export/summary业务块已退出；审计、prepare_source/require_harmony、audit_only/preprocess_only提前退出原样，二者不进入task/Stage。三种HESTА输入、mmap/截断/identity、RNA-only/crossview=0保持。noOT跳过initial prior，但原fit的refresh/eval及更新记录不优化；dry保持eval+no_grad一次forward。train仍调用同一个shared fit，原weights/history在输出断言之前写出，U3布局/flags/JSON策略不改。

V0～V12全部通过：Embryo61主例+10补例；三输入×UOT开关的6个真实CPU Finit各132个named parameter逐name/shape/dtype/bytes及构造后RNG一致；真实prior方向、固定权重dry输出/RNG一致。raw小型七sectionH5AD实际执行原HESTА预处理/manifest重载。非零epoch用fit替身核对全部实参和writer，不冒称真实训练通过。另从未改fit抽取原noOT refresh guard和final-eval语句在真实CPU模型执行：99不刷新、100仍eval/no_grad并记录[100]、prior为空，随后原final eval；不运行epoch循环或optimizer。

fresh回归：CRC/HLN/Spleen/MISAR/Thymus/Simulation各5例共30，MouseBrain64例（含train/no_grad epoch0及Dropout/RNG），23个U2handoff+MISAR helper，28个U3跨族artifact文件对照。只规范化预声明output root及四个计时叶子，summary/provenance其余字段/顺序不忽略，浮点不放宽。main错误包装AST不动，不声称源码traceback栈位置相同。

parser/defaults/CLI、external loader薄兼容、原Stage/fit/artifact导入别名保留U7兼容清单；公共任务不反向依赖runner。250,208个既有保护路径/lstat一致；model/Finit、fit、artifacts、PreparedDataset/data_io/HESTА、analysis、CLI/defaults、B13/B14/B12/B11、SPATCH raw/reuse及PathLike边界不改。未修改历史result/cache/fixture/expected/REPORT/known failure或gene_imputation*。未跑长期训练、完整analysis、百万spot或GPU gate；U4d-1交付后停止，不进入U4d-2。


## 32．U4d-2退出记录（2026-09-21追加；SPATCH公共训练任务完成）

**U4d-2 COMPLETE；U4e未执行。** 以U4d-1完成累计工作树为起点，96项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，分支l0-original-20260717、index为空。保留第31节“当时SPATCH未迁移”。

[U4d-2 REPORT](../../refactor_checks/u4d2_spatch_training_task_20260921/REPORT.md)、[验证](../../refactor_checks/u4d2_spatch_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4d2_spatch_training_task_20260921/static_results.json)。新增training/spatch_task.py；runner +6/-106，checker仅+3/-2迁哨兵拦截，其他88份起点Python源码SHA保持，task_runtime不改。

run_spatch_training_task接resolved config+PreparedDataset，GPU信息和caller持有的execution_state仅用于原summary/生命周期。Namespace沿原runner字典；feature/spatial/section/audit/cache_info原引用。显式ModalityOrderPolicy指定section1及HE/RNA/Protein校验；调用已有initialize_prepared_task(defer_prior=True)，保持Stage→原modality check→monitor→initial prior→shared fit→原export/summary。没有dataset-name隐藏分支、第二套epoch/writer或callback/trainer框架。

GPU准入仍在seed/prepare之前，raw/reuse完全留在task之前。raw仍仅六个已对齐RNA/Protein/已有HE-UNI H5AD→原sequential→全新schema1 cache→释放→strict loader；不新增图像UNI提取或fallback/cache覆盖。task动态无data_io/np.load访问，不解释历史source。原main没有dry/preflight/--dry_run，本批不新增；V6验证无fit前forward和参数拒绝，固定CPU eval出口只是测试。

原finally执行时model/outputs仍应存活。新task将原model/monitor/history/outputs/ot_updates/summary引用交回execution_state持有至caller finally结束，无复制；成功/失败GC和CUDA清理替身均核对。PathLike helper机械移至SPATCH task，runner导入兼容，只在SPATCH summary使用os.PathLike→os.fspath，其它unsupported仍TypeError；U3/B12边界不动。

V0～V12全部通过（V6按当前不存在独立dry的事实验收）。36主任务+3生命周期例；raw/reuse各160个Finit参数与构造后RNG精确，真实CPU blockwise双向prior及固定eval出口一致，默认BF16/显式FP16/chunk/checkpoint/candidate/top-k/refresh等fit参数映射保持。正式GPU拒绝分支真实执行；后续CPU测试使用明确GPU准入/monitor替身，无GPU训练声明。raw实际510spot两section/4828共同gene/17去DAPI为16marker/2048已有HE，noHarmony及Harmony预处理、全新cache和strict reload闭环通过。

另有55例reuse loader和73例current checker，CRC/HLN/Spleen/MISAR/Thymus/Simulation30例、MouseBrain64例、Embryo6例、U2 23handoff+MISAR helper、U3 28文件回归。只规范化预声明输出根/时间/gzip时间及验真的依赖digest，不忽略summary/provenance、不放宽数值容差。身份正例缺新fixture JSON与旧handoff拦截位置失效的初次失败记录保留；只修本批新测试并在独立目录重跑。后者仅小型CPU prior到FAISS GPU不可用检查，无GPU gate/训练执行。

原Stage/prior/fit/export业务块退出runner；input选择/GPU/seed/准备观察/错误finally保留；parser/defaults、旧helper导入和wrapper兼容列U7，不提前清理未知外部API。256,395个既有保护路径/lstat不变；model/Finit、fit、artifacts、PreparedDataset/data_io/SPATCH数学/schema、analysis、CLI/defaults、B13/B14/B12/B11和历史result/cache/fixture/expected/REPORT/gene_imputation*均保持。未跑百万spot、长期训练、完整analysis或固定GPU gate；GPU综合与真实规模留后续U4/U8。完成后停止，不进入U4e。


## 33．U4e退出记录（2026-09-21追加；generic公共训练任务完成）

**U4e COMPLETE，V0～V12全部通过；U5未执行。** 起点为U4d-2累计工作树，97项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，l0-original-20260717，index为空。第32节“当时U4e未执行”保留。

[U4e REPORT](../../refactor_checks/u4e_generic_training_task_20260921/REPORT.md)、[验证](../../refactor_checks/u4e_generic_training_task_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u4e_generic_training_task_20260921/static_results.json)。仅runner +11/-177、新增training/generic_task.py 206行；其它90份起点Python源码SHA不变。

run_generic_training_task接原resolved config、PreparedDataset、已解析args、effective section_order和四布尔YieldExecutionPolicy；复用task_runtime.initialize_prepared_task(defer_prior=True)，沿用generic原optional sparse prior和iter_fit_model，无二次resolve/读数据，无dataset-name隐藏分支、额外dry/forward或第二套epoch/writer。generic自己的json_safe/summarize_outputs/save_training_artifacts仅机械移至task并由runner导入兼容，U3 artifacts权威核不改。

bundle/raw/synthetic准备优先级、float32边界、引用和section顺序保持。noOT跳过initial prior且关闭refresh_ot；普通零/负epoch继续allow_empty_epochs=True并final eval，smoke的epochs=0仍经原args.epochs or 3。yield_every条件不变，periodic沿原step/可能refresh后yield的epoch payload保存权重/history/summary，不额外forward/eval、不存optimizer/scaler/RNG、不称exact resume。final yield与最终embedding选择、文件布局/flags/调用顺序保持。

43个writer/受控任务对照（20预期错误）和12个真实CPU/失败路径对照（7成功、5预期错误）通过：三输入真实handoff、逐参数Finit/RNG、真实blockwise prior、shared fit实际0或2步、yield前step状态、final eval、artifact完全一致。真实首次refresh周期未到达；受控替身验证step→refresh→yield时点，fit算法SHA不变，不声称本批数值refresh/GPU gate通过。

回归CRC/MISAR六caller30例、MouseBrain64例、Embryo6例、SPATCH4例、U2 23handoff+MISAR helper和U3 28文件。SPATCH raw仍六个已对齐H5AD→现有sequential→新cache→strict loader，不含图像UNI；PathLike边界不扩展。新raw manifest创建时间/gzip时间引发的派生SHA按U4d-2既有验真合同核对，不整块忽略summary/provenance；详见报告。

266,042个既有保护文件lstat不变，保护目录无新增文件；model/Finit/fit/artifacts/data_io/preprocessing/analysis、B13/B14/B12/B11和历史result/cache/fixture/expected/REPORT/gene_imputation*均保持。runner仅保留输入解析、CLI/config验证及兼容导入；旧训练业务块退出，兼容面留U7盘点。未跑长期训练、百万spot、完整analysis或GPU gate；后续综合验证需独立授权。本批停止，不进入U5。


## 34．U4 总退出记录（2026-09-21追加；训练任务层总验收完成）

**U4 COMPLETE；U5未执行。** 以U4e完成累计工作树为起点，103项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，l0-original-20260717，index为空。既有U4a～U4e当时的状态记录均保留。

[U4总验收REPORT](../../refactor_checks/u4_final_acceptance_20260921/REPORT.md)、[静态证据](../../refactor_checks/u4_final_acceptance_20260921/static_results.json)、[任务回归](../../refactor_checks/u4_final_acceptance_20260921/task_results.json)、[依赖/兼容清单](../../refactor_checks/u4_final_acceptance_20260921/dependency_debts.json)。本轮生产代码零修改、零删除，不重新实施U4，不提前整理U7。

九dataset＋generic正式任务分别经paired/multisection/mousebrain/embryo/spatch/generic六个中立training门面；Stage构造权威为task_runtime.initialize_prepared_task，epoch/step/refresh/final eval权威为fit.iter_fit_model，通用artifact机械write权威为training.artifacts。PreparedDataset在task前完成，task不重新读raw/resolve。统一逻辑边界不强制将真实不同任务塞入一个函数。

runner间仅5条provider import、9处parse_args/resolve_run_config/get_dataset_defaults使用，全部B类，留U7；A类Stage/prior/fit/export业务依赖为零。CRC/MISAR旧pipeline仍由自身main使用，仅映射配置/准备并委托中立entry；不是无caller。validation的CRC shared-fit/forward、MouseBrain helper、SPATCH CLI/cache别名等列C/U7，公共算法不复制。未找到可以确认为无caller且仅U4过渡的业务wrapper，因此不删除不明外部接口。

本轮42例小型task及28个artifact文件回归通过：CRC/MISAR六caller各3例，MouseBrain4，Embryo6，SPATCH4，generic受控5＋真实CPU5。覆盖resolved/PreparedDataset/section身份、Finit/prior/RNG、ExecutionPolicy、fit映射、artifact及特殊模式。CRC train-mode dry、MISAR eval dry、MouseBrain正常训练也执行train-mode+no_grad epoch0/dropout、adapter audit先行、Embryo提前退出/RNA-only/noOT、SPATCH GPU准入/raw-reuse先准备/BF16配置、generic zero/yield/periodic-final均保持。SPATCH raw仍六个对齐H5AD且HE为已有UNI特征，不新增原图推理。

92份生产/工具Python源码本轮SHA不变；51份model/Finit/fit/artifacts/data_io/preprocessing/analysis核心源码与U4a前oracle相同。271,755个既有保护文件lstat一致，保护目录无新文件；Git状态/HEAD/index保持。B13/B14/B12/B11、PathLike边界、历史result/cache/fixture/expected/REPORT及gene_imputation*不改。

小型CPU/Finit/prior/固定forward与generic零或两步shared fit实际执行；非零跨族fit多用替身。未跑长期训练、完整analysis、百万spot或GPU gate，未到正式首次动态refresh数值周期；GPU/全程refresh/真实规模仍需后续U8或独立授权验证。U4结构/兼容验收完成不等于U8数值验收完成。追加本记录后停止，不进入U5。


## 35．U5a退出记录（2026-09-21追加；CRC/HLN/Spleen analysis输入与标签机械核）

**U5a COMPLETE，V0～V12通过；U5b未执行。** 以U4总验收累计工作树为起点，99项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6，index为空。第34节“当时U5未执行”保留。

[U5a REPORT](../../refactor_checks/u5a_analysis_inputs_crc_hln_spleen_20260921/REPORT.md)、[静态/保护](../../refactor_checks/u5a_analysis_inputs_crc_hln_spleen_20260921/static_results.json)、[输入对照](../../refactor_checks/u5a_analysis_inputs_crc_hln_spleen_20260921/input_results.json)、[外部caller/跨层回归](../../refactor_checks/u5a_analysis_inputs_crc_hln_spleen_20260921/remaining_results.json)。

新增data_io.analysis_results（AlignedAnalysisInput、显式paired layout、identity/coords机械核）、paired_comparison_inputs（三族原专用layout）、saved_assignments（显式CSV/NPY/zip/分section合同）。analysis.loaders与comparison caller仅接缝迁移，legacy dict/MethodData字段/类型及source顺序保持，类旧module导出兼容。truth在新合同为None/not_loaded，旧消费者仍无truth属性；无barcode的标签只能声明row-order证据。未改变弱/强身份校验差异、cohort/section/spot顺序。

严格paired普通/comparison及CRC COSIE每次两section RNA由4次打开降至2次；实际pandas identity校验仍执行，旧第二pass异常延至原逻辑阶段抛出，保留缺文件与重复ID的失败优先级。重复打开消除后AnnData同一warning不再重复发两次，此观察差异明确登记。没有表达矩阵读取或新增embedding全量copy；选中coords临时引用保留至装配，未声称真实规模内存已验证。

116例基础输入/标签、33例外部布局/caller哨兵、mmap/alias/资源探针、真实小bundle、28个artifact文件、5个generic任务替身对照通过。输入值/dtype/顺序/errors/B9 data_identity精确对照，唯一允许独立fixture根差异；真实caller在聚类前截断。初次别名错误及object-NPY fixture的旧B9限制记录保留，修正接缝/另建fixture后独立重跑，不放宽生产校验。

新共享reader不依赖analysis.metrics/protocols/scripts；旧comparison_inputs中其他族依赖留U5b/U5c，不声称全data_io反向依赖已清零。joint_results、HLN人工标注、UMAP、visualization和无当前caller的旧CRC summary/tuple API未改。兼容wrapper/reexport用途与后续边界见报告，不删不明外部接口。

3个生产文件修改、3个新增，其余89份源码SHA不变；275,129个保护文件lstat不变、保护目录无新增。model/Finit、fit、U2 PreparedDataset/preprocessing、U3 artifacts、U4 task、CLI/defaults与科研协议保持。新运行的method_comparison源码digest变化与B9输入身份分开记录，不修改历史cache。未运行训练、真实重聚类、完整analysis或GPU gate。本批停止，不进入U5b。


## 36．U5b退出记录（2026-09-21追加；MouseBrain/MISAR/Thymus/Simulation输入）

**U5b COMPLETE，V0～V13通过；U5c未执行。** 起点为U5a累计工作树，104项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6、index为空。第35节“当时U5b未执行”保留。

[U5b REPORT](../../refactor_checks/u5b_analysis_inputs_mousebrain_misar_20260921/REPORT.md)、[验收矩阵](../../refactor_checks/u5b_analysis_inputs_mousebrain_misar_20260921/validation_results.json)、[静态/保护](../../refactor_checks/u5b_analysis_inputs_mousebrain_misar_20260921/static_results.json)、[回归](../../refactor_checks/u5b_analysis_inputs_mousebrain_misar_20260921/regression_results.json)、[实际caller](../../refactor_checks/u5b_analysis_inputs_mousebrain_misar_20260921/caller_final_results.json)。

普通入口依次MouseBrain→MISAR→Thymus→Simulation各16例通过后，才迁四族16个跨方法format reader。新增data_io.multisection_results与external_result_formats，复用U5a AlignedAnalysisInput/_align_positions；新增analysis.comparison_readers仅绑定原协议/清理/派生truth及原validator。机械核不import analysis；NPY/metadata CSV打开、身份/coords对齐只由公共实现承担。旧dict/MethodData字段/顺序/缺失语义及pickle导出地址保持。

MouseBrain raw obs/多label/group原策略保持；MISAR analysis dataset1→4，不套训练4→1；Thymus普通saved coords无truth、comparison canonical obs[x,y]严格身份保持；Simulation普通saved spatial_domain与comparison raw spatial_domain→spfac回退分开保留。未新增nsfac读取或factor字段；专项factor诊断不改。旧标签CSV/NPY/zip仍经U5a核，row-order-only证据不伪造成barcode。

最终64普通+196外部输入、32旧validator、12真实caller哨兵对照通过，值/dtype/order/truth/B9/errors精确核对。MouseBrain普通/SPA raw打开6→3，MISAR/Thymus SPA 8→4；原pandas身份校验仍执行，成功open/close一致。只保留必要metadata列数组引用至装配，不保留AnnData/X；无新embedding全量copy或dtype转换。mmap是容器alias探针，不声称实际loader新增mmap或真实规模RAM已验收。

U5a 149案例、U3 28文件、U4 5个任务替身及U2真实小bundle回归通过。未执行聚类/指标/训练；替身不能代替真实Finit/训练数值验收。MISAR漏pandas import与ownership探针两次误断言的初始证据保留，独立重跑通过，不修改历史fixture/expected或放宽生产校验。

3生产文件修改+3新增，其余92份起点源码SHA不变；43个method_comparison函数（只规范化接缝别名）、原truth清理、validator、Simulation派生公式AST保持。278,507个历史保护文件lstat不变、保护目录无新增。model/Finit/fit、U2/U3/U4、preprocessing、科研协议与CLI/defaults不改。

遗留：comparison_inputs的SPATCH input/protocol依赖及visualization_inputs的UMAP policy依赖留U5c；四族旧导出__getattr__仅兼容转发到analysis策略门面，正式caller已直接使用新门面，后续公开API/U7再决定退出。专用format adapter合理保留。不宣称全data_io反向箭头清零；新共享reader的协议知识依赖已经解除。源码SHA变化与B9输入身份分开记录，不回写历史cache/result。完成后停止，不进入U5c。


## 37．U5c退出记录（2026-09-21追加；剩余analysis读取边界）

**U5c COMPLETE，V0～V14通过；U6未执行。** 起点为U5b累计工作树，108项交付SHA匹配；HEAD仍d52829b769ebd06cae25c4ca78e4a02aba7068f6、index为空。第36节当时状态保留。

[U5c REPORT](../../refactor_checks/u5c_analysis_inputs_remaining_20260921/REPORT.md)、[验收矩阵](../../refactor_checks/u5c_analysis_inputs_remaining_20260921/validation_results.json)、[最终输入对照](../../refactor_checks/u5c_analysis_inputs_remaining_20260921/part_results.json)、[静态/保护](../../refactor_checks/u5c_analysis_inputs_remaining_20260921/static_results.json)、[跨层回归](../../refactor_checks/u5c_analysis_inputs_remaining_20260921/regression_results.json)。

依次SPATCH→Embryo→method UMAP→input integration→HLN人工标注，各部分通过才进入下一部分。新增data_io.large_results/reference_results中立机械核，复用AlignedAnalysisInput和saved_assignments；原返回tuple/LoadedData/DatasetBundle、dtype、section/spot/annotation/provenance保持。无ID的NPY及Embryo shared rows只记row-order/source-row；metadata继续原位借用，不为合同制造百万行ID或额外coords矩阵。

SPATCH严格cohort/spec/validator移到analysis.spatch_readers；原requested/comparison差异、fixed sample身份保持。五个展示预处理函数归analysis.visualization_preparation，原8个科学/机械函数body AST保持；不把标准化/预处理伪装成read-only loader。method UMAP只共享显式格式/alias/identity选择与labels读取；scaler/UMAP输入空间不改。HLN五类format adapter、A1 truth、A1/D1 joint、retained K专项语义保留，未混同joint_per_section。

最终253个输入/错误/读取时序对照、4个真实caller/cache-hit哨兵、fixed sampling/B9 source mismatch、mmap/alias/backed关闭probe通过。U5a149+U5b260输入、12caller、28artifact、5generic任务替身、真实小bundle回归通过。不声称替身等于训练/Finit/科学数值重验。

11生产文件修改+4新增，其余87份SHA不变；295,739个历史保护文件lstat不变、无保护根新增。model/Finit/fit、U2/U3/U4、preprocessing、clustering/metrics/batch/sampling/protocols/umap/cache/joint_results与CLI/defaults不改。源码path/SHA变化与B9输入身份分开，不回写历史cache/result。

正式中立reader已无analysis协议依赖；字面反向import仍有旧公开API懒兼容桥：comparison_inputs的U5b四族及SPATCH导出、visualization_inputs旧展示函数，单份实现且正式caller均已迁出，留U7公开API决定。专用格式和科研workflow继续独立，不以文件数为收敛标准。

未运行完整analysis、真实重聚类、百万spot计算、长期训练或GPU gate；真实规模RAM和科研数值未重验。完成后停止，不进入U6。


## 38．U5总验收退出记录（2026-09-22追加；未通过）

**U5 NOT_ACCEPTED，不能标记总完成；U6未执行。** 以U5c累计工作树为起点，111项交付SHA匹配，生产102份Python文件未改，HEAD/index保持。前述U5a/U5b/U5c当时验收记录不改写。

[U5总验收REPORT](../../refactor_checks/u5_final_acceptance_20260921/REPORT.md)、[条件矩阵](../../refactor_checks/u5_final_acceptance_20260921/acceptance_results.json)、[实际边界探针](../../refactor_checks/u5_final_acceptance_20260921/boundary_results.json)、[静态保护](../../refactor_checks/u5_final_acceptance_20260921/static_results.json)。目录按用户指定保留20260921，实际检查日期为2026-09-22。

发现两个总收口缺口：正式requested绘图和Embryo分节/层级/reuse等saved-label读取仍未全部接公共saved_assignments核；Simulation的aligned_derived_truth仍在data_io生命周期内调用analysis._simulation_truth派生策略。前者以真实小CSV/NPY记录shared reader调用0次；后者以5个小section记录truth回调调用者为data_io。无直接反向import不等于严格运行时职责已彻底分离。

这些均有真实caller，不是允许删除的D类临时wrapper，因此本轮未修改生产代码、未擅自修正或扩大到U6。三处旧API懒兼容桥仍明确保留到U7；未删除joint_results或独立科研模块。legacy逻辑字段可对应，但不声称所有下游收到完整identity_evidence对象。

九数据集/外部method/标签与row-only布局小型回归完成；74条pre-U5/当前对照确认值/错误/B9保持，raw打开减少4→2、6→3、8→4且身份检查仍在。SPATCH/Embryo mmap和HLN关闭探针、generic显式NPY及原无generic evaluation协议的边界、U2 bundle、28artifact、5任务替身通过。没有真实聚类/UMAP/完整analysis/百万spot/训练/GPU gate；源码与保护证据见报告。

下一步需单独明确G1机械读接缝修正与G2读取/truth策略分离的合同，不在总验收中暗自实施。验收工作到此结束，U5未总完成，停止，不进入U6。


## 39．U5d退出及U5最终通过记录（2026-09-22追加）

**U5d COMPLETE，V0～V7通过；重新进行的U5总验收通过，U5 COMPLETE；U6未执行。** 第38节及旧U5 final REPORT当时的NOT_ACCEPTED状态保持，不覆盖历史记录。

[U5d REPORT及总收口](../../refactor_checks/u5d_analysis_reader_residuals_20260922/REPORT.md)、[V0～V7](../../refactor_checks/u5d_analysis_reader_residuals_20260922/validation_results.json)、[U5最终十项条件](../../refactor_checks/u5d_analysis_reader_residuals_20260922/u5_final_acceptance.json)、[静态保护](../../refactor_checks/u5d_analysis_reader_residuals_20260922/static_results.json)。

以U5总验收后的累计工作树为起点，111项交付SHA匹配，HEAD/index保持。8份reader接缝文件修改、无新增生产模块；其余94份生产/工具SHA不变。绘图、Embryo层级/分节/reuse、CRC统计、joint_results及MouseBrain group诊断的12处标签读取均委托既有saved_assignments机械核；原策略/校验/格式/dtype/mmap/copy点不改。五个workflow文件规范化reader别名后整AST一致。

Simulation四种format reader先返回原表/数组/identity引用，由analysis adapter执行原truth派生和最终逻辑合同装配。中立open_aligned_rna_metadata只负责backed打开、identity positions和finally关闭；不接受policy回调。原spatial_domain优先/spfac回退公式AST不变；20次实际external truth调用栈验证没有data_io策略执行。旧aligned_derived_truth回调核退出，不保留双实现。

88标签案例、49 Simulation外部案例、12 truth/独立微型diagnostic对照、135 U5关键回归、mmap/alias/关闭探针通过；28 artifact、5 generic任务替身与真实小bundle保持。诊断仅30行独立CPU输入输出，未运行完整analysis/真实聚类/UMAP。新增测试初始fixture、证据容器/JSON记录和静态误匹配问题已记录，另建证据重跑，不修生产来迁就测试。

U5最终核对通过：正式reader无科研策略依赖，saved-label核单份权威；row-order-only/optional truth与legacy证据传播限制如实保留。3处旧API懒兼容桥（comparison_inputs四族/SPATCH、visualization_inputs展示函数）保留到U7；专用format adapter及joint_results保持独立。未新增通用generic科研协议。B9 input/implementation/source identity分开处理，不修改历史cache。

312,892保护文件lstat/成员保持；model/Finit、fit、U2/preprocessing、U3 artifacts、U4 task及科研协议/CLI/defaults均未改。未运行百万spot、训练、完整analysis或GPU gate，未声称本轮重跑Finit/全族数值。U5到此完成，停止，不进入U6。


## 40．U6a退出记录（2026-09-22追加；HLN/Spleen普通exact KMeans scope）

**U6a COMPLETE，V0～V10通过；U6b/U6c未执行。** 起点为U5d最终通过后的累计工作树，111项交付SHA匹配，HEAD/index保持。第9.1节首对范围保持：HLN普通comparison与Spleen普通comparison；requested、跨方法外层编排和专项workflow不迁。

[U6a REPORT](../../refactor_checks/u6a_hln_spleen_kmeans_workflow_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6a_hln_spleen_kmeans_workflow_20260922/validation_results.json)、[静态/保护](../../refactor_checks/u6a_hln_spleen_kmeans_workflow_20260922/static_results.json)、[真实PNG](../../refactor_checks/u6a_hln_spleen_kmeans_workflow_20260922/real_plot_results.json)、[正式entry/B9](../../refactor_checks/u6a_hln_spleen_kmeans_workflow_20260922/entry_results.json)。

新增analysis.exact_kmeans_workflow.execute_exact_kmeans_scopes，两个原run_scheme只传显式section/K/seed/n_init/max_iter/plot/mask分配规则及既有label/plot/spatial操作，公共核负责scope循环和原metrics/scaler输出；config/SUMMARY仍为caller原尾段。没有dataset-name分派、registry/factory/class框架。AlignedAnalysisInput仍经既有legacy投影交接，输入reader不改。

HLN先22例真实小CPU对照通过后才迁Spleen18例。HLN independent默认[5,8,10,12]与Spleen2…12保持；dtype、KMeans、metric核唯一且不改。joint每K一次fit后切片，independent仍K外/section内，重复scaler fit保留；默认HLN19/9、Spleen33/23次KMeans/scaler。ordinary无biological truth，额外truth/None仍不新增监督指标。HLN A1专项不改。

40个主例的数组/scaler/labels/指标/事件/CSV/NPZ/config/summary和错误足迹精确一致；另K=5各4张真实PNG像素一致。正式HLN comparison和Spleen requested验证真实fit及B9 hit/mismatch无额外fit；原不支持的协议组合仍拒绝。requested batch仅用哨兵，不声称重验batch数值。135条U5关键输入、28artifact、5任务替身、真实bundle及九runner导入回归保持。

2生产文件修改+1新公共workflow，其余100份SHA不变；两个owner其它函数/常量/外层及config/summary尾段AST不变。原run_scheme继续承担协议与summary职责，并非需要删除的空compat wrapper。当前源码SHA与B9输入身份分开记录，不修改历史cache。

319,167保护文件lstat/集合保持；model/Finit/fit、U2/U3/U4、preprocessing、科研算法/协议和CLI/defaults均未改。只跑新小fixture真实CPU KMeans/指标与小图验证，未运行真实大规模analysis、训练、UMAP或GPU gate。停止，不进入U6b/U6c。


## 41．U6b退出记录（2026-09-22追加；evaluation orchestration）

**U6b COMPLETE，V0～V12通过；U6c未执行。** 起点为U6a累计工作树，112项交付SHA匹配，HEAD/index保持。当前用户明确将requested纳入本批；第40节U6a当时的范围保持。

[U6b REPORT](../../refactor_checks/u6b_evaluation_orchestration_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6b_evaluation_orchestration_20260922/validation_results.json)、[静态保护](../../refactor_checks/u6b_evaluation_orchestration_20260922/static_results.json)。

新增reader-free analysis.evaluation_workflow：evaluate_standardized_scopes承接原普通requested五数据集编排，append_scope_metrics合并joint/independent重复row组装，evaluate_saved_joint只消费已对齐joint labels，evaluation_session统一B9 begin/成功finish边界。requested保留原float64/sections/truth投影和兼容入口；evaluation保留参数、显式协议和输入读取。无registry/factory/engine或dataset-name隐藏分派。

CRC专用抽样/MiniBatch、U6a comparison exact scope与requested的dtype/scaler次序/指标布局存在真实差异，保留各自单份权威实现，不强行合成同一科学流程。method_comparison只接共同成功边界，原跨method校验/cohort/shared/retention/逐method汇总保持。SPATCH/Embryo/Simulation/HLN专项不改。

25条requested/saved-joint、最终14条comparison、4条正式entry旧/新有序字段/数值/事件/产物对照通过；joint_per_section用抛错聚类哨兵证明零新fit。HLN22+Spleen18个U6a真实小CPU scope回归、135个U5输入回归、28artifact、5任务替身及真实小bundle通过。batch/CRC专用科学步骤仅核对边界实参，不冒充其数值重验。

3生产owner修改+1公共模块，其余100份原源码SHA不变；322,708历史保护文件集合/lstat保持。clustering/metrics/protocols/sampling/batch/UMAP、U2/U3/U4/model/fit/preprocessing及CLI均不改。当前源码SHA与B9输入身份分开记录，不改历史cache。测试曾有fixture缺data_dir及输出目录碰名，另建目录修正验证并保留初次证据，未修生产迁就测试。

requested.analyze_prepared仍是正式入口，save_labels旧公开名为直接re-export；U7再决定公开API兼容。各comparison run_scheme继续承担协议/layout责任，不作为死wrapper删除。没有新增迁移期双实现。未运行真实全量analysis、UMAP、训练、百万spot或GPU gate；U6b完成后停止，不进入U6c。


## 42．U6c-1退出记录（2026-09-22追加；跨方法输入机械流程）

**U6c-1 COMPLETE，V0～V8通过；U6c-2/U6c-3未执行。** 以U6b完成工作树为起点，112项交付SHA匹配，HEAD/index保持。未改写第41节当时状态。

[U6c-1 REPORT](../../refactor_checks/u6c1_method_inputs_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6c1_method_inputs_20260922/validation_results.json)、[静态保护](../../refactor_checks/u6c1_method_inputs_20260922/static_results.json)。

复用U5现有AlignedAnalysisInput/legacy投影、reference/identity核，不新建MethodAdapter框架。reference_results新增四个机械函数：table_identity、stack_embedding_files、validate_embedding_table、read_reference_metadata。paired三族及四族external adapter共19处身份字段提取、四处COSIE表格/final数组校验接公共核；四族CSV/prefix/merged_emb改用已有U5实现。method UMAP仅normalize_section/load_reference委托，benchmark/UMAP计算、saved scaler/labels、reference cohort/section顺序保持。

原格式adapter、source/shared路径、truth与强弱身份语义保留；普通comparison按前缀文件列序，UMAP按既有数字列序；comparison alias缺失与reference fallback差异不统一。SPATCH仍专用严格cohort，PRESENT仅HLN标注、Harmony仅Embryo UMAP既有读取，不新增普通benchmark支持。input integration算法和method_comparison外层源码不改。

176四族+27paired external+90reference+15边界共308条扩展输入对照通过；135 U5关键回归、U2小bundle/28artifact/5任务替身保持；U6a40及U6b43条小型workflow回归通过。值/dtype/order/metadata/source/B9、读取/关闭与错误一致。reference边界用抛错fit/metric/UMAP哨兵证明不新增科学操作。无ID仍保持row-order证据，不伪造barcode/truth。

只修改四份reader文件，无新增生产文件，其余100份SHA不变；327,572历史保护文件集合/lstat保持。静态检查曾将相对.analysis_results误当analysis依赖，修正检查器的包解析，未改生产行为。未运行完整benchmark/analysis、真实UMAP、大数据、长期训练或GPU gate。完成后停止，不进入U6c-2/U6c-3。

## 43．U6c-2退出记录（2026-09-22追加；跨方法UMAP workflow）

**U6c-2 COMPLETE，V0～V8通过；U6c-3未执行。** 以U6c-1完成工作树为起点，111项交付SHA匹配；没有用裸HEAD替代累计完成态，没有改写第42节当时结论。

[REPORT](../../refactor_checks/u6c2_method_umap_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6c2_method_umap_20260922/validation_results.json)、[真实CPU UMAP](../../refactor_checks/u6c2_method_umap_20260922/umap_real_results.json)、[静态保护](../../refactor_checks/u6c2_method_umap_20260922/static_results.json)。

新增轻量`analysis/umap_workflow.py`：project_umap_views、save_umap_coordinate_table、run_method_projection。method_umap.generate_outputs委托公共方法投影与绘图调用；umap.generate_dataset共用逐view投影和坐标表。既有fit_umap、protocol、plotting、reference/scaler/assignment readers不改，config/README/finish的AST保持。没有method-name科学dispatcher、engine或registry。

reference cohort仍由U5/U6c-1输入层确定，公共workflow不重读、不排序、不抽样、不fit scaler。外部方法继续原standardized输入和force_float32=False；本方法继续原模态顺序→integrated及原FP32边界，有panel c/indices的独有布局。每个已选view仍独立fit_transform，未增加transform、跨方法concat或重复fit。input integration算法全文件保持。

6个真实CPU场景覆盖COSIE/MOFA+/SpaMosaic/本方法及subset/section cohort，每版本8次投影的坐标逐值一致；15张实际PNG像素一致，绘图期间reader/UMAP哨兵没有触发。另25个编排/缓存/失败场景和14个alias/多方法入口/非法参数场景old/new一致。293扩展输入、135 U5、40 U6a/43 U6b、28 artifacts/5任务替身/6 bundle本批回归通过；不将替身对照说成真实UMAP数值测试。

旧method_umap.fit_umap公开薄委托保留，没有第二套算法；正式generate_outputs改走公共workflow，旧外部API删除留U7兼容审查。generate_outputs/generate_dataset继续拥有不同layout及payload，不强行合并。B9 input/implementation identity保持，当前script SHA变化单列核验，不修改历史cache。

只修改method_umap.py、umap.py并新增73行公共模块；其余102份生产/工具Python SHA保持。338,715个历史保护文件集合/lstat不变，HEAD/index不变。未运行完整analysis、真实大数据UMAP、百万spot、训练或GPU gate；未修改算法、benchmark或历史证据。完成后停止，不进入U6c-3。

## 44．U6c-3退出记录（2026-09-22追加；input integration输入组织）

**U6c-3 COMPLETE，V0～V8通过；U6总验收未执行。** 以U6c-2完成工作树为起点，115项交付SHA匹配，HEAD/index保持。未改写第43节当时结论。

[REPORT](../../refactor_checks/u6c3_input_integration_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6c3_input_integration_20260922/validation_results.json)、[输入对照](../../refactor_checks/u6c3_input_integration_20260922/integration_results.json)、[静态保护](../../refactor_checks/u6c3_input_integration_20260922/static_results.json)。

新增85行`analysis/integration_workflow.py`，公共化section manifest、CSV/NPY saved-label装配、metadata拼接、full-run模态选行，以及五处“saved embedding→选行→已有scaler→已有labels”流程。三个移入函数AST相同；`input_integration`保留原数据集adapter与公开re-export，无第二套实现、engine或隐藏method dispatcher。原DatasetBundle/U5 borrowed contract、各adapter返回payload/provenance表达式保持。

明确区分：input integration正式组织本方法九数据集的原模态与已有集成结果；外部COSIE/MOFA+/SpaMosaic沿用U5/U6c-1 comparison/reference格式核，本批未扩展支持或统一benchmark输入空间。sample/section/spot、缺失模态、metadata、颜色/标签、provenance保持，row token不伪称barcode。原visualization_preparation和sampling不改；MouseBrain原全量标准化/KMeans标签步骤原位保留，新公共核不执行聚类。

U6c-2 UMAP生产源码保持。公共输入核不计算UMAP、不重选cohort；原顶层仍在prepare后仅一次委托已有generate_dataset。SPATCH展示feature cache原算法/来源/写出时点和其它保存布局保持，无新增磁盘schema。旧section_manifest/label导出为兼容re-export，退出需U7审查。

35个expanded integration（九数据集×三个max_samples及标签边界）、56个机械装配/错误/alias/入口场景old/new一致；293跨方法/reference、135 U5输入及28 artifacts/5任务替身/6 bundle、U6c-2 25个确定性编排回归通过。真实小NPY/CSV和saved-scaler计算实际运行，重型原模态准备、MouseBrain标签计算和UMAP回归使用明确替身；不宣称本批重跑真实Harmony/KMeans/UMAP。一次验证目录命名冲突在写fixture前拒绝，改用新名字重跑通过，保留失败日志。

只修改input_integration.py并新增公共模块，其余104份生产/工具Python SHA不变；350,886个历史保护文件集合/lstat保持。未运行完整analysis、真实integration方法、百万spot、训练或GPU gate；未修改model、preprocessing、benchmark、历史result/cache或图片。完成后停止，不进入U6总验收。


## 45．U6 总收口验收记录（2026-09-22追加）

**U6 final acceptance：NOT ACCEPTED；不标记U6完成；U7未执行。**

[REPORT](../../refactor_checks/u6_final_acceptance_20260922/REPORT.md)、[验收矩阵](../../refactor_checks/u6_final_acceptance_20260922/validation_results.json)、[重复调用证据](../../refactor_checks/u6_final_acceptance_20260922/handoff_results.json)、[静态保护](../../refactor_checks/u6_final_acceptance_20260922/static_results.json)。

以U6c-3交付工作树为起点，114项交付SHA匹配。本轮不修改生产源码；666项小型矩阵对照及ownership/B9探针通过兼容性检查。106份生产/工具Python、363,397保护文件集合/lstat保持；plan仅追加。U2/U3/U4小型接口回归及静态保护保持；没有重跑全部历史真实准备/Finit。

唯一不能签署的严格条件是V10：exact_kmeans_workflow independent K-first循环在每个K内重新拟合同一section scaler。两个section、两个independent K，加一个joint K，实际5次fit对应3组输入，多2次相同section标准化。不同K的必要聚类不是重复聚类。该行为U6a明确为兼容保留，并非新数值回归；但与本轮“不重复standardization”要求不一致。保留各子阶段当时已通过结论，不改写历史。

需后续单独决定显式接受历史例外，或以窄批次消除重复并对数值、错误/保存时点及内存验证；此次不擅自修改算法/循环。中立reader、专用adapter、兼容转发及科研workflow按报告分层，U7兼容事项未提前处理。未跑完整benchmark/analysis、真实UMAP、百万spot、训练或GPU gate。停止，不进入U7。


## 46．U6d退出与U6最终验收（2026-09-22追加）

**U6d COMPLETE；U6 final acceptance COMPLETE；U7未执行。** 第45节NOT ACCEPTED是当时状态，原报告保留。本次根据用户明确授权先审计再修正，未用“协议例外”绕过V10。

[U6d报告](../../refactor_checks/u6d_independent_scaler_20260922/REPORT.md)、[修改前审计](../../refactor_checks/u6d_independent_scaler_20260922/BEHAVIOR_AUDIT.md)、[U6重新验收](../../refactor_checks/u6d_independent_scaler_20260922/U6_FINAL_ACCEPTANCE.md)、[专项对照](../../refactor_checks/u6d_independent_scaler_20260922/scaler_results.json)、[保护检查](../../refactor_checks/u6d_independent_scaler_20260922/static_results.json)。

审计确定fitted_space的StandardScaler与K无关；per-section范围是协议，每K重fit只是U6a为兼容保留的冗余。仅修改exact_kmeans_workflow.py（+10/-3）：每次workflow局部缓存section scaler，首次到达原点fit，后续K transform。保留K-first、异常/写出时点；不改joint、不跨section共享、不缓存全部转换矩阵、不改clustering/metrics/protocol。

11项真实CPUscaler场景的mean/scale/var、dtype、transforms、KMeans输入/labels、metrics、产物及错误完全一致；多K fit7→3，原阻塞例5→3、每section一次。transform次数保持，这是本次明确批准的fit once + per-K transform边界。666项U5/U6/U2/U3/U4矩阵回归及ownership/B9/handoff通过；仅批准的fit事件减少，其它事件保持。UMAP/训练使用编排替身的范围如实记录，不冒充真实大规模数值验证。

其余105份生产/工具Python及378,007历史保护文件集合/lstat保持；没有修改模型、fit、PreparedDataset、artifact、算法或历史证据。兼容转发仍留U7，不提前清理。未运行完整benchmark/analysis、全量UMAP、百万spot、训练或GPU gate。U6d与U6总验收完成后停止，不进入U7。


## 47．U7a退出记录（2026-09-22追加；默认来源与配置边界）

**U7a COMPLETE，V0～V9通过；U7b/U7c/U7d未执行。** 起点为U6d后的U6最终完成工作树，123项交付SHA匹配。

[REPORT](../../refactor_checks/u7a_config_defaults_20260922/REPORT.md)、[配置对照](../../refactor_checks/u7a_config_defaults_20260922/config_results.json)、[补充合同](../../refactor_checks/u7a_config_defaults_20260922/core_results.json)、[静态保护](../../refactor_checks/u7a_config_defaults_20260922/static_results.json)。

新增纯training.entry_defaults作为九数据集/generic入口默认权威。HLN/Spleen共用paired profile，HLN/Thymus/Simulation原相同14项执行默认共用一份；runner旧get_dataset_defaults薄委托，MouseBrain/SPATCH常量保持旧名导入别名；Spleen→HLN默认依赖退出。基础模型/预处理default仍由model.configure权威提供，analysis.protocols独立。没有dataset dispatcher或新preset框架。

resolve_model_config原合并算法保持：defaults/base < preset/model JSON < dataset/input < 非None CLI；旧无JSON能力的入口不扩展schema。generic smoke的epochs0→3、SPATCH固定执行设置、Embryo RNA-only/crossview=0属于原模式约束，明确记录并保持，不假称任意CLI都能覆盖它们。当前训练裸scale1.0不改，evaluate来源说明默认0.5不混入训练。当前样例位置data/configs，非根configs。

1,607解析场景+57核心合同逐值/类型/键序/错误一致，含九真实batch子parser；257跨U2～U6小型回归通过。model、fit、PreparedDataset、artifact、training task、analysis、preprocessing及配置JSON保持；其余95份生产/工具Python不变。392,824历史保护文件集合/lstat、HEAD/index保持。测试探针的argparse递归与路径规范化假差异保留日志，最终隔离对照通过，未改生产错误策略。

仍留CRC/MISAR parser/配置投影兼容依赖供后续U7明确批次处理；旧API不提前删除，README位置描述留文档批。未运行真实训练、完整analysis、百万spot或GPU gate。完成后停止，不进入U7b/U7c/U7d。


## 48．U7b退出记录（2026-09-22追加；CLI与兼容入口）

**U7b COMPLETE，V0～V10通过；U7c/U7d未执行。** 以U7a工作树为起点，117项交付SHA匹配；18个用户CLI全部保留。

[REPORT](../../refactor_checks/u7b_cli_wrappers_20260922/REPORT.md)、[入口清单](../../refactor_checks/u7b_cli_wrappers_20260922/entry_inventory.json)、[历史分类](../../refactor_checks/u7b_cli_wrappers_20260922/historical_inventory.json)、[help对照](../../refactor_checks/u7b_cli_wrappers_20260922/help_results.json)、[静态保护](../../refactor_checks/u7b_cli_wrappers_20260922/static_results.json)。

新增非可执行scripts.paired_cli/multisection_cli，机械归位CRC/MISAR共享parser、section校验和配置投影。HLN/Spleen/Thymus/Simulation直连provider，正式runner→runner导入清零。旧API re-export；run_*_pipeline单return委托到既有entry shell的legacy参数适配，body重命名后AST相同。9个parser/config函数AST相同，U7a默认权威不改。

A类13、D类5；4个内部支撑模块不是用户CLI。54条历史旧scripts路径按S2记录核对，3个同名validation工具已迁tools，其余退休/替换；本批无删除/恢复旧launcher。tools12模块保持，38机械兼容函数登记，冻结API不因caller少删除。

1607配置场景、57核心合同、19入口/旧API哨兵、18CLI与7validation完整help、batch print-only、12validation import及174跨层小型回归通过。scripts README同步入口/default归属/ID边界。其余99份生产/工具Python及397274历史保护文件集合/lstat保持，HEAD/index不变。

未改模型/fit/PreparedDataset/analysis/artifact/defaults或科研协议，未运行真实训练、完整analysis、百万spot或GPU gate。哨兵不冒充真实训练验证。完成后停止，不进入U7c/U7d。


## 49．U7c退出记录（2026-09-22追加；validation依赖边界）

**U7c COMPLETE；U7d未执行。** 以U7b实际工作树为起点，123项交付SHA匹配；无回退/提交/暂存区操作。

[REPORT](../../refactor_checks/u7c_validation_dependencies_20260922/REPORT.md)、[依赖图](../../refactor_checks/u7c_validation_dependencies_20260922/dependency_graph.json)、[wrapper清单](../../refactor_checks/u7c_validation_dependencies_20260922/WRAPPERS.md)、[验证结果](../../refactor_checks/u7c_validation_dependencies_20260922/validation_results.json)、[静态保护](../../refactor_checks/u7c_validation_dependencies_20260922/static_results.json)。

replay_model/check_fixed_gpu直接使用training.fit的fit、prior和forward；SPATCH checker直接使用data_io.spatch_preparation常量与loader。移除validator内部crc/mouse模块别名；生产兼容API保持。cache checker不再调用runner parser/build_tasks获取业务参数，采用公开batch print-only CLI及明确fixture options核对。保留2处局部正式CLI被测入口和MouseBrain只读AST结构断言，它们不提供业务算法。

旧新完整73项cache checker、1607配置+57核心合同、18正式CLI及7validation完整help、213跨层小型回归通过；真实小CPU双向OT诊断输出一致。12工具import不加载runner；FP32/BF16 checker以非GPU替身核对调用权威和顺序，不冒充GPU gate。最初artifact probe对已退休validator别名的失败保留，独立v2/v3探针改用公共身份验证后通过，历史脚本不改。

38个薄兼容函数逐项登记；保留真实caller和冻结API，无法证明无外部消费者的不删。后续API退役需U8兼容验收/显式决定，不自动列为U7d删除项。当前无tests目录；tools无sys.path修改；历史phase oracle路径隔离与旧suite文本是证据，全部保留。

只改3份validation Python及tools README；其余106份生产/工具Python、399649历史保护文件集合/lstat、HEAD/index保持。model/Finit/fit/PreparedDataset/config默认/analysis/artifacts/CLI均不改。未运行真实训练、完整analysis、百万spot或GPU gate；known-failure smoke和B4/R1/R3未处理。完成后停止，不进入U7d。


## 50．U7d退出记录（2026-09-22追加；当前文档与运行说明）

**U7d COMPLETE；U7总验收未执行。** 以U7c完成工作树为起点，125项交付SHA匹配。只改根README、scripts README、docs导航，新增CURRENT_ARCHITECTURE，并追加本记录。

[REPORT](../../refactor_checks/u7d_documentation_20260922/REPORT.md)、[文档检查](../../refactor_checks/u7d_documentation_20260922/documentation_results.json)、[验证结果](../../refactor_checks/u7d_documentation_20260922/validation_results.json)、[保护检查](../../refactor_checks/u7d_documentation_20260922/scope_results.json)。

文档明确v7A纯空间图、final embedding与重构的区别、PreparedDataset/公共task/fit/artifacts及analysis边界；默认权威在entry_defaults/config，不再写成runner持有业务实现。CRC示例显式正epochs和新output。训练缺省scale1.0与研究命令0.5分开。B11/B12/B13/B14/Finit改为当前保留状态，独立待决问题不夹带修复。

SPATCH默认reuse；raw仅六个上游已对齐H5AD→sequential preparation→全新schema-1 cache→strict loader，不含原图UNI。MouseBrain已有UNI和通用image+mask能力分开，不声称真实图像推理已验证。Embryo raw/manifest/external与generic各schema/mode限制写明。所有示例输出是全新路径模板、无overwrite；旧suite只作退休说明。

143本地链接、30示例parser/help、18正式CLI清单、4JSON示例通过；57配置核心+10checker边界旧新一致，18CLI/7validation完整help及batch print-only一致。109份生产/工具Python、配置及405760历史保护文件保持；docs历史SHA保持，plan只追加。未执行训练、预处理、UNI、完整analysis或GPU gate；解析通过不等于真实输入计算通过。完成后停止，不进入U7总验收。


## 51．U7最终总退出记录（2026-09-22追加；工程入口验收完成）

**U7 COMPLETE，V0～V9限定静态/小型接口验收通过；U8未执行。** 以U7d累计完成工作树为实际起点，123项交付SHA匹配；不是以裸HEAD代替历次未提交成果。本次无生产修改，只新增独立验收证据并追加本记录。

[REPORT](../../refactor_checks/u7_final_acceptance_20260922/REPORT.md)、[验证结果](../../refactor_checks/u7_final_acceptance_20260922/validation_results.json)、[入口/依赖清单](../../refactor_checks/u7_final_acceptance_20260922/entry_inventory.json)、[wrapper状态](../../refactor_checks/u7_final_acceptance_20260922/WRAPPERS.md)、[保护检查](../../refactor_checks/u7_final_acceptance_20260922/static_results.json)。

entry_defaults是入口profile权威，model.configure保持基础模型/预处理默认，analysis协议独立；resolve_model_config保持defaults/base < preset/model < dataset/input < 非None CLI。入口schema投影和已有模式约束继续显式存在，不强制所有scale=0.5、不改变epochs/AMP/OT。1607配置场景、57核心合同、10validation依赖记录隔离旧新一致。

18正式CLI及7个可安全执行的validation help一致；9数据集batch真实print-only、36个evaluate main/parser哨兵、19入口/legacy handoff通过，未启动分析或子训练。12工具import、73项SPATCH小cache checker通过；validation只从公共模块取得业务实现，保留2处局部正式CLI作为被测对象。tools无sys.path获取runner业务，当前无tests目录。

213条跨层序列化记录、9组公共合同/import/alias及3组HESTА/SPATCH小型source-handoff通过；其中bundle6条为字段记录，不冒称6种独立模式。raw预处理替身、训练/UMAP替身均明确标记；真实cache写入/strict reload仅发生在本轮新小fixture目录。未重跑各历史阶段真实数值/Finit全部测试。

38个显式兼容函数、Spleen薄默认/parser、parser re-export及data_io旧API lazy bridge均有用途/退出条件。正式runner→runner业务import为零；lazy bridge不是第二套reader科研算法，旧module identity/未知外部consumer不擅自删除。后续API退役需显式兼容决定，不自动作为U8清理任务。

143文档路径、30示例parser/help命令、4JSON示例、18CLI清单一致；v7A纯空间图、SPATCH raw仅六个已对齐H5AD/已有HE-UNI、无原图UNI、各输入边界准确。U1～U7完整退出链核对；U3/U5/U6早期失败记录由后续最终通过记录承接，不覆盖历史。

109生产/工具Python SHA、U7d其余交付内容、405908历史保护文件集合/lstat、HEAD/index保持。新增探针错误简称被原CLI拒绝，修正探针为正式ID后重跑通过，日志保留，无生产修正。未运行真实训练、完整analysis、百万spot、真实UNI或GPU gate。到此停止，不进入U8。


## 52．U8a最终工程结构审计记录（2026-09-22追加；审计完成，有遗留）

**U8a审计交付完成（COMPLETE_WITH_FINDINGS）；不作无保留最终结构通过声明；U8b未执行。** 以U7总验收累计工作树为起点，123项交付SHA匹配。只读源码/AST/Git及新报告输出，没有运行生产import、help、预处理、训练、analysis或GPU gate。

[REPORT](../../refactor_checks/u8a_final_structure_audit_20260922/REPORT.md)、[结构证据](../../refactor_checks/u8a_final_structure_audit_20260922/structure_checks.json)、[兼容复核表](../../refactor_checks/u8a_final_structure_audit_20260922/WRAPPERS.md)、[发现](../../refactor_checks/u8a_final_structure_audit_20260922/findings.json)、[保护检查](../../refactor_checks/u8a_final_structure_audit_20260922/protection_results.json)。

109份Python/952函数静态检查：data_io/model/training/analysis到scripts无反向import；生产epoch循环唯一在training.fit，模型运行数组/weights/JSON写出核在training.artifacts；分族prepare和同语义evaluation/comparison/UMAP/integration公共流程主干成立，专项科研workflow保持。18CLI、3份当前文档78个本地链接、30命令模板路径保持，未执行命令。少量SHA/json_safe/观察hook重复不属于第二套完整prepare或workflow，不自动合并。

**F1未关闭：** MouseBrain四个外部format reader通过align回调进入analysis.comparison_readers/analysis.loaders，在reader调用栈内执行section→group truth投影。没有直接protocol import不等于运行时科研职责已彻底分离。Simulation的U5d修正不能替代MouseBrain核查。数值错误未被证明，本轮不改truth/错误/读取时点；若要修正需独立授权的最小职责批次，不自动进入U8b。

**F2补登记但不自动退休：** U7逐项清单漏列generic load_preprocessed_inputs及旧SPATCH validate_source_stats等对象的终局状态；后者仍有校验body、未发现正式caller，不能称薄wrapper或raw strict校验。新表复核50个wrapper/旧helper，另登记lazy桥与import re-export；未知外部consumer不擅自删。PreparedDataset可None顺序注解、legacy identity证据传播等已知限制列F3，不夹带API行为更改。

v7A纯空间图、Finit、SPATCH raw仅六个对齐H5AD/已有HE-UNI、PreparedDataset/task入口描述主干准确；“reader不派生truth”的强表述受F1限制。没有修改既有U7/更早通过记录，也不把本次结构发现等同历史数值回归失败。

生产修改0，109源码SHA及文档（本计划仅追加）保持，410630历史保护文件集合/lstat、HEAD/index保持。历史fixture/expected/report/cache/result/gene_imputation*未改。未跑真实训练、完整analysis、百万spot或GPU gate；未重跑Finit/历史数值测试。到此停止，U8b及U8综合最终验收未执行。


## 53．U8a-fix退出与U8a结构重新验收（2026-09-23追加）

**U8a-fix完成，V0～V7通过；U8a结构重新验收通过，U8b未执行。** 目录日期沿用用户指定20260922；不改写§52及旧U8a“有遗留”的历史记录。

[REPORT](../../refactor_checks/u8a_fix_structure_residuals_20260922/REPORT.md)、[逐项兼容登记](../../refactor_checks/u8a_fix_structure_residuals_20260922/WRAPPERS.md)、[静态/AST证明](../../refactor_checks/u8a_fix_structure_residuals_20260922/static_results.json)、[重新结构审计](../../refactor_checks/u8a_fix_structure_residuals_20260922/structure_checks.json)、[保护检查](../../refactor_checks/u8a_fix_structure_residuals_20260922/protection_results.json)。

起点为U8a累计工作树，123项receipt匹配；没有用裸d52829b替代后续未提交成果。F1关闭：MouseBrain四个外部格式reader返回原数组/table/identity/source context，原truth/coords投影改在analysis侧继续；section→group派生行运行时无data_io reader frame。COSIE后续校验次序、SPA单次RNA snapshots、数组引用及旧copy点保持。52条读取/身份/错误轨迹对照、四方法实际小型KMeans/metrics/save及plot输入对照通过，源文件SHA与路径保持。比较器首次共享fixture路径归一化假差异已单独记录并以磁盘原值核实，无生产修正或容差放宽。

F2关闭：52项逐个分类A22/B4/D26，C0；另登记3条lazy桥。不假造未知外部caller，不自动删API。旧SPATCH validate_source_stats与CRC summarize_outputs的整函数机械迁至中立data_io/training层，runner同名接口仅薄委托，仍不启用为正式raw/reuse/训练路径。D类明确保留，退休须单独确认外部兼容承诺；旧参数/layout/返回视图适配保留。

生产6文件+83/-61；109源码中103份SHA不变，所有既有prepare/task和科学函数不变。13组213条跨层回归记录、1607 config/57 config-core/10依赖记录、18CLI及7安全validation help、18条兼容helper对照通过。模型/Finit、fit、PreparedDataset、artifacts、预处理和analysis科学算法未修改；410647历史保护文件集合/lstat与HEAD/index保持；计划仅追加。

重新审计109Python/954函数：主干分层成立，无core→scripts反向业务依赖；机械reader无科学协议import；fit/artifacts及同语义公共workflow仍唯一。78文档链接/30命令模板及18CLI一致。原F3类型/legacy身份传播限制和少量小工具重复为已知范围，不在本批改变。没有运行真实训练、完整analysis、全量UMAP、百万spot、UNI推理或GPU gate；本次不等同U8综合数值/规模验收。到此停止，不进入U8b。


## 54．U8b-2代表性真实流程验证（2026-09-23）

**验证执行结束，U8b-2总验收未通过；U8b-3未执行。** 起点为U8a-fix和[U8b-1固定gate完成态](../../refactor_checks/u8b1_fixed_numeric_gate_20260923/REPORT.md)，不改写§53当时“U8b未执行”的历史记录。U8b-1证据和冻结文件保持，原gate本批不重复运行。

[本批REPORT](../../refactor_checks/u8b2_representative_flows_20260923/REPORT.md)及[实际观测](../../refactor_checks/u8b2_representative_flows_20260923/observation_checks.json)记录五类风险、六次真实一epoch训练：CRC CSR、MouseBrain、Simulation、Embryo manifest/external、SPATCH reuse。真实prepare→task→Finit/prior→fit→artifact执行，Stage config/feature/RNG交接与独立CPU初始化重放通过；MouseBrain保留且精确重放一次train-mode/no_grad epoch0。Simulation正式evaluate K=2 joint/independent小链完成，truth派生在analysis层。SPATCH reuse在真实4090上BF16一轮；另有510 spots的六H5AD raw→新schema1 cache→strict reload逐值闭环，未使用替身。

**未通过项：** CRC正式comparison/evaluate固定要求2716/7284 ASW行和原cohort，小样本48/48在抽样校验失败；没有改协议、伪造cohort或跑完整benchmark。因此CRC训练/保存/reader通过不等于指定完整链通过。另保留旧dense-backed双轴索引错误、首次Simulation CPU/BF16参数冲突和补充Embryo annotation x/y假设失败。合法CSR与显式CPU FP32另案通过；Embryo已存在annotation字段/embedding检查通过，但本例annotation无x/y，完整analysis未运行、不补造metadata。

当前各小run沿裸入口scale=1.0，不自动改0.5，不声称复现研究命令科研指标。SPATCH raw仍仅六个已对齐RNA/Protein/已有HE-UNI H5AD，不含原图UNI。没有修改生产代码、默认、fixture/expected或历史证据；仅本批新目录与本计划追加。未跑长期训练、全量UMAP、完整benchmark、百万spot、固定GPU gate或U8b-3。后续需单独决定CRC小样本验收边界，本批不修复、不扩大范围。


## 55．U8b-2 CRC cohort核实与代表范围验收更新（2026-09-23）

**U8b-2代表性流程验收通过（PASS_SCOPED）；U8b-3未执行。** 本次按用户明确授权重新分配完整evaluate代表，不修改生产协议；§54和旧U8b-2 REPORT/completion/失败记录保持当时状态。

[核实报告](../../refactor_checks/u8b2_crc_cohort_resolution_20260923/REPORT.md)、[当前验收记录](../../refactor_checks/u8b2_crc_cohort_resolution_20260923/acceptance.json)、[只读复核](../../refactor_checks/u8b2_crc_cohort_resolution_20260923/verification_results.json)。CRC机械reader再次读出96×128 FP32与真实任务输出/coords一致，96个section/spot身份唯一。原cohort选择函数的拒绝仅由96≠10000触发，重复identity分支为假；正式ASW要求2716/7284行，完整/plot cohort为166279/446095，均不放宽。

CRC分类为：PreparedDataset→training→artifact PASS、analysis reader PASS、**合格cohort完整evaluate NOT RUN（验证输入不满足正式cohort合同）**。原不合格输入尝试实际RUN/FAIL/退出1保留，不能将其改写为未运行或PASS。已执行和复核范围未发现prepare/task/artifact/analysis workflow回归，不声称CRC未执行的聚类/指标已通过。

Simulation担任完整小链代表：原真实一epoch训练产物的10个analysis source SHA保持，joint240行及五section independent各48行的身份/truth一致，正式evaluate退出0并有6行supervised、6行internal及1行batch metrics。其余代表保留CRC paired、MouseBrain epoch0/RNG、SPATCH真实GPU reuse及六H5AD raw重载、Embryo manifest/external/RNA-only风险覆盖。Embryo annotation无x/y的补充断言失败继续保留，完整analysis仍NOT RUN。不是九数据集全部evaluate、完整benchmark或规模验收通过。

仅新核实目录与本计划追加，生产/默认/cohort校验不变，历史报告、fixture/expected、result/cache和失败记录不变；本轮未重新训练/聚类/完整analysis或运行GPU。停止，不进入U8b-3。


## 56．U8b-3最终环境、状态记录与冻结前审计（2026-09-23追加）

**U8b-3完成；U8c未执行，尚未建立最终Git冻结点。** 以§55接受的U8b-2累计工作树为起点，仅只读Git/源码/历史证据/环境和设备属性；没有模型计算、训练、预处理、analysis或GPU gate。

[REPORT](../../refactor_checks/u8b3_final_state_record_20260923/REPORT.md)、[Git清单](../../refactor_checks/u8b3_final_state_record_20260923/git_state.json)、[环境](../../refactor_checks/u8b3_final_state_record_20260923/environment.json)、[依赖补充](../../refactor_checks/u8b3_final_state_record_20260923/additional_dependencies.json)、[生产SHA](../../refactor_checks/u8b3_final_state_record_20260923/production_sha256.json)、[保护检查](../../refactor_checks/u8b3_final_state_record_20260923/protection_results.json)。

HEAD仍为d52829b769ebd06cae25c4ca78e4a02aba7068f6，branch=l0-original-20260717，index空。已有37份tracked修改、33份未跟踪源码、2份未跟踪文档、5729份未跟踪历史result文件；refactor_checks的104个直接子目录均被ignore。HEAD不是当前完成态的单独代表，源码/文档与ignored验收证据需要在U8c明确checkpoint/存档方式；本轮不commit/add/tag、不改变ignore、不自动纳入历史result。

实际解释器cosie Python3.9.19，Torch2.4.0/CUDA build12.1/cuDNN9.1，RTX4090/driver535.183.01，FAISS1.9.0、NumPy1.26.4。记录完整distribution和Conda CUDA元数据；区分shell base环境、实际cosie解释器、驱动CUDA12.2显示和包标签。当前只读查询进程的默认确定性/线程设置不冒充U8b-1受控gate环境，未升级依赖或修改设置。

U8b-1 FP32/BF16各13/13、same-layout、Finit和handoff证据保留；cross-chunk仍FAIL/KNOWN_NUMERICAL_SENSITIVITY。U8b-2仍为PASS_SCOPED：Simulation代表正式完整小链；CRC训练/保存/reader通过，合格cohort完整evaluate NOT RUN，原不合法输入RUN/FAIL记录保持。SPATCH raw仅六个已对齐RNA/Protein/已有HE-UNI H5AD，不含原图UNI。未跑九数据集长期训练、百万spot raw、完整benchmark、全量UMAP或GPU长期训练。

109源码与U8b-2一致；U8b-1/2/CRC-resolution的32/57/11项收据匹配，冻结14文件SHA保持。U8a-fix交付123项中122项不变，唯一计划差异证明为旧170204字节SHA前缀之后的授权追加。415991个既有保护文件成员/lstat保持，44文档仅本计划追加；model/Finit/fit/preprocessing/PreparedDataset/artifacts/analysis科学算法无变化。已知失败和兼容D类限制继续登记，不改写历史验收。到此停止，不进入U8c。


## 57．U8c最终冻结与发布状态（2026-09-23追加）

**U系列重构收口为当前v7A纯空间图生产架构的本地Git checkpoint。** 本节随最终源码/文档提交；包含本节的`refactor: finalize v7A project architecture`提交即本次冻结对象。完整commit hash、提交后状态、CLI help/文档复核与保护结果记录在[U8c REPORT](../../refactor_checks/u8c_final_freeze_20260923/REPORT.md)。没有推送远程或创建release/tag，不将本地冻结描述为远程发布。

起点为U8b-3完成态，原HEAD为d52829b769ebd06cae25c4ca78e4a02aba7068f6、分支l0-original-20260717、index空；109份生产源码及U8b-3证据收据匹配。只纳入已验收的37份tracked变更、33份新增源码和2份新增文档，共72文件；本批生产代码修改为0。最终文档改动仅本计划追加与CURRENT_ARCHITECTURE更新冻结说明、已知限制和后续扩展规则。model/Finit、fit、data contract、科研协议没有本轮未记录修改。

`refactor_checks/`及验证临时输出、全部历史result/cache不进入commit，不修改.gitignore、不删除证据。既有五个未跟踪result目录保留，因此提交后“tracked源码/文档与index干净”不等于“整个工作目录无untracked”。历史证据留在原本地目录，报告/收据通过本批独立目录保存；纯源码clone不包含这些证据，不能把本地存在当作远程已归档。

主干合同保持：显式data_io adapter→PreparedDataset；resolved config+prepared→公共training task→唯一fit→公共artifacts；analysis input/identity→显式workflow→既有算法；scripts为CLI及必要输入/资源/GPU准入边界。专项workflow和真实协议差异保留，不以目录数量作为完成标准。

最终接受范围继承U8a-fix、U8b-1及CRC resolution后的U8b-2 PASS_SCOPED：FP32/BF16固定配置和same-layout已通过，cross-chunk仍FAIL/KNOWN_NUMERICAL_SENSITIVITY；CRC小输入训练/保存/reader通过，合格cohort完整evaluate NOT RUN；Simulation覆盖完整小链。SPATCH raw仅六个已对齐RNA/Protein/已有HE-UNI H5AD，不含原图UNI；百万spot raw、全量benchmark/UMAP、九数据集长期训练和GPU长期训练NOT RUN。weights非exact resume、兼容D类与既有失败足迹继续保留。

后续新增输入使用显式adapter与身份合同，训练复用task/fit/artifacts，analysis保持reader与科研策略分离，CLI不复建runner业务依赖。新实验使用独立输出与验证目录；模型/协议变更需独立批次和相应数值验证，不修改冻结fixture/expected或覆盖历史result/cache。本批不开发功能、不重跑训练/预处理/analysis/GPU gate，最终提交和只读入口复核后停止。
