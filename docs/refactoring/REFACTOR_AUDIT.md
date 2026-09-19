# spa_mo_model 第一阶段重构审计

审计日期：2026-09-12。对象：`/home/hujinlan/spa_mo_model` 的**当前工作树**，包括原有未提交修改。本文是审计与实施方案，不是已实施的重构。没有运行真实数据训练，没有修改生产代码、测试、配置或既有结果。

## 1. 首要结论与证据边界

1. **仓库没有一个被统一声明、统一解析的正式默认训练入口。** 没有根 README、统一 train 命令、安装清单或模型版本选择器。能够证实实际运行过的主要路线是 dataset runners + suite；`run_stage_model.py` 实际是 smoke checks，不能按名称当作训练入口。
2. **实际模型只有 `model/stage_model.py:255` 的 `StageMultiModalModel`。** MouseBrain 直接构造它；其余主要数据集直接或通过源码 `replace + exec` 适配 CRC/MISAR runner，再构造同一类。没有独立的 v7A、v14A、v14B 模型类。
3. **v7A 的 0.5 确实生效，但不是裸默认。** `get_default_model_config()` 和 MouseBrain/CRC/MISAR/SPATCH 的 scale 默认均为 1.0；v7A suite 显式传 0.5。当前配置另有 `graph.use_feature_graph=True`，因此现在执行 v7A suite 也会运行已经进入共享实现的 pre-OT feature graph 流程。
4. **当前实际刷新是 epoch 100、120、140、160、180、200 的 optimizer step 后。** `should_update_ot(..., first_update_epoch=100)` 位于 `stage_model.py:119-157`，所有生产 trainer 都只传 epoch 和 interval，未把 first epoch 暴露成配置。epoch 1 只初始化 feature graph。首次刷新后的 prior 从 epoch 101 的训练 forward 开始使用。这与“约 epoch 20 首刷”的先验不同，但不是 off-by-one。
5. **历史稳定基线与当前共享实现已经混在一起。** `git show 922d173:model/configure.py` 中 feature graph 为 False；当前为 True。v7A/v15A/v15F suite 没有传一个能选择旧架构的参数。版本名/输出目录不冻结模型数学路径。
6. 真实问题主要集中在：配置解析与实验身份漂移、feature/OT refresh 的接口状态不一致、稀疏 padding 的 attention 语义、数值 loss、评估缓存与运行结果失联。不是“旧文件很多，所以都能删”。

本报告以代码中的默认值、调用者、实际保存的命令记录相互印证。只读了必要的小文件 `result_v7A/suite_status.json`、`suite.exit.json` 及 v15F 对应文件，确认两条 suite 确实完成过；没有扫描大型 embedding、h5ad、图片或 checkpoint 内容。不能由这两个记录推断用户全局的运行频率，也不能声称 v15F 已被正式批准替代 v7A。

## 2. 初始 Git 状态、说明与仓库地图

系统 PATH 无 git；使用已有 `/home/hujinlan/miniconda3/envs/cosie/bin/git` 完成只读检查。分支 `l0-original-20260717`，HEAD `c4380fccd71c388409c126ab7c885eb2710410cc`。初始状态：

```text
 M model/stage_model.py
 M scripts/analyze_result_v7a_requested_metrics.py
 M scripts/plot_and_prune_result_v7a_clusters.py
 M scripts/run_crc_stereocite.py
 M scripts/run_mousebrain_v2.py
 M scripts/run_result_v15a_suite.py
 M scripts/train_stage_model.py
?? result_v15F/
?? scripts/run_result_v15f_suite.py
```

未暂存 diff 为 7 文件、182 insertions / 79 deletions；暂存 diff 为空。逐项阅读了完整 diff：模型删除了 lazy feature graph 初始化和 prepare-refresh 内隐式刷新；三 trainer 增加 epoch1/两次 forward 刷新；suite 与评估增加 v15F/数据集选择。上述改动是用户原有工作，不是本次审计产生。

近期历史（只用来定位变更，主路径判断另有代码证据）：`c4380fc` pre-OT spatial+feature、`922d173` v7A/B/C 与 0.5 基线、`24a6eae` v6 双 GraphSAGE/延迟刷新、`a73e289` 首刷延迟100、`ceb6380/6a3152c` final-source、`dd176c9` 单模态、`4875279` fused-source、`4f3c2f0` topology/context/G0。历史备注不能替代当前代码。

仓库及父目录未发现适用于本项目的 AGENTS.md；根无 README。三个 gene-imputation 目录有 README，已阅读；`docs/` 的模型验证、OT/FAISS、预处理、单模态说明和根历史实验报告作为历史说明使用，不把其“PASS”或旧默认值当作当前保证。没有 tracked shell scripts、notebooks、pytest 配置或环境依赖清单。

| 区域 | 当前规模/作用 | 审计处置 |
|---|---|---|
| `model/` | 12 Python、6391 行；数学模型、预处理、图、OT、图像特征 | 重点审计 |
| `scripts/` | 73 Python、37315 行，含未 tracked v15F；trainer、dataset adapters、suite、评估、测试混放 | 结构精简的最大收益点 |
| 三个 `gene_imputation*` | 共16 Python、4862行；插补、平滑、PCC 与测试 | 下游独立工作流；不是 decoder 的替代实现 |
| `model/configs/` | v7A/B/C 三个小 JSON，只含 post scale | 有用 preset，但没有冻结完整基线 |
| `data/configs/mousebrain_preprocess_train.json` | 被实际 suite 引用的1547字节配置；当前未 tracked（data 被忽略） | 必须在重构时把可运行配置与数据分离 |
| `docs/`、根实验报告 | 历史设计、运行命令、比较说明 | 保留来源；迁移到 history 时更新当前导航 |
| `results/` | 2373 个 tracked 生成文件 | 不读大内容，不在代码清理中删除科研结果 |
| 其他 `result_*`、data、UNI、cache、logs | 输入、历史运行、模型权重、大规模产物 | 不作代码删除对象 |

Git 共 tracked 2525 文件：model 15、scripts 72、docs 22、根16、三个插补目录25、report_derived_metrics 2、results 2373。真正源代码远少于文件总数。审计前对192个源代码/说明/配置文件保存了 SHA256 清单，结束时复核原文件内容未变。

## 3. 默认入口与配置矩阵

### 3.1 可证实的训练入口

| 入口 | 实际用途与默认行为 | 最终构造/训练 |
|---|---|---|
| `scripts/run_result_v7a_suite.py:104` | 五数据集 suite；显式 sparse+双向、scale0.5；输出目录叫v7A | `training_commands()` → dataset runners |
| `scripts/run_spatch_v7abc_suite.py:107` | SPATCH A/B/C 对照；相同训练流程，不同scale | `run_spatch.py` → CRC共享trainer |
| `scripts/run_result_v15a_suite.py:48`、`run_result_v15f_suite.py:26,99` | 复用/改写 v7A suite globals、任务和输出目录；没有模型版本选择逻辑 | 同一批dataset runners、同一Stage类 |
| `scripts/run_mousebrain_v2.py:34,512` | 必须给 `--config`；裸默认 dense、单向、scale1；实际suite显式改为 sparse、双向、0.5、200ep | `StageMultiModalModel` → 本文件循环677 |
| `scripts/run_crc_stereocite.py:68,1025` | 默认 dry forward，`--train` 才训练；dense、单向、scale1 | Stage → `train_small_crc_model:774` |
| `scripts/run_misar_seq.py:61,361` | 默认 dry forward、epochs0/CPU，`--train` 才训练；candidate_sparse、单向、scale1；RNA/ATAC自定义模态集合 | Stage → CRC共享trainer |
| `scripts/run_mouse_spleen.py`、`run_human_lymph_node.py` | main 向 argv 前端注入200ep、sparse、双向、checkpoint等；原用户参数在后，标量通常后者优先 | 读取CRC源码→replace→exec→CRC pipeline |
| `scripts/run_simulation.py:72,146`、`run_mouse_thymus.py:119,188` | 读取MISAR源码替换ATAC/数据集，注入默认参数；Protein适配 | MISAR pipeline → CRC共享trainer |
| `scripts/run_spatch.py:623` | GPU训练入口；sparse+双向、BF16、chunks/checkpoints、默认scale1 | Stage → CRC共享trainer |
| `scripts/run_human_embryo_rna_only.py:197,230,296` | 默认audit_only；显式选择preprocess_only/dry_run/train；单模态RNA、contrast权重0、identity fusion | Stage → CRC共享trainer |
| `scripts/train_stage_model.py:262` | 通用 bundle/preprocess-config/synthetic trainer；dense初始化，无双向CLI | Stage → 本文件循环283 |
| `scripts/run_stage_model.py:377` | parse后无条件跑合成检查，`--smoke_test`兼容标志不影响行为 | 不是正式trainer |

因此“当前最常用入口”只能严谨地回答为：**有实际运行证据的主线是 suite 所启动的 dataset runners；核心复用训练循环在 `run_crc_stereocite.py:774`，MouseBrain 是另一套独立循环。仓库没有可据以宣布唯一官方默认的配置。**

### 3.2 当前 v7A/v15F MouseBrain 实际命令

下例是 suite 中的训练命令，展示路径用，**本次未执行**：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_mousebrain_v2.py \
  --config data/configs/mousebrain_preprocess_train.json \
  --epochs 200 --lambda_contrast 0.1 --device cuda --seed 42 \
  --output_dir result_v7A/mousebrain/v3_bidirectional_sparse_fixed_lc0.1 \
  --ot_prior_mode candidate_sparse --bidirectional_ot_attention \
  --candidate_backend faiss_ivf --initial_modality_candidate_k 100 \
  --candidate_k 200 --attention_topk 10 \
  --faiss_nlist 256 --faiss_nprobe 32 --faiss_device auto \
  --faiss_train_sample_size 10000 --faiss_query_batch_size 2048 \
  --dynamic_candidate_source ot --uot_epsilon 0.05 \
  --uot_tau_a 1.0 --uot_tau_b 1.0 --uot_max_iter 100 \
  --update_interval 20 --spatial_knn_k 5 \
  --post_ot_graphsage_scale 0.5 --save_ot_prior_topk
```

v15F 的 MouseBrain 命令仅改变输出根。裸执行 `train_stage_model.py --model_config model/configs/v7a_post_ot_graphsage_scale_0.5.json ...` 则仍走 dense/单向，不等于这条科研基线。

## 4. 完整真实 runtime call chain

### 4.1 数据、构造与初始 prior

MouseBrain：`main → parse_args → run_mousebrain → load_json → preprocess_mousebrain:316 → build_mousebrain_sections:293 → build_mousebrain_section(multimodal_preprocessing.py:69)`。按配置 sections 的顺序装载RNA、从RNA提取HE已有feature、对齐Metabolite/obs；随后 `build_cosie_data_dict:293 → load_cosie_style_data(data_preprocessing.py:474) → load_data:127 → preprocess_adata:52`。

shared modalities 先按section顺序合并、按模态处理RNA（启用HVG时先选HVG，再normalize_total→log1p→scale→PCA）、Protein CLR/PCA、HE PCA等，再按配置选择Harmony，按原section长度 split (`data_preprocessing.py:170-255`)。得到 `feature_dict[s][m]` 的FP32矩阵和同序空间坐标；默认 n_comps50，模型 decoder 重构的是这些预处理特征，不是原始counts/图像。CRC/MISAR/SPATCH有各自 obs/var 对齐、section重命名与模态适配，见其输出保存和附录；不要把这些适配视为可直接删除的重复训练逻辑。

`build_model_config(run_mousebrain_v2.py:339) → StageMultiModalModel(config, feature_dict):568`：

- 合并 `get_default_model_config()` 与传入配置；构造 ModuleDict（`stage_model.py:268-326`）。
- 为配置列出的每个多模态组合预建 FusionMLP；`initialize_from_feature_dict:475` 按实际模态维度建 encoder/decoder，并强制所有section使用相同 observed modality set、同模态同维度。
- pre GraphSAGE `:329` 的scale固定1；post `:345` 是不同实例、不同参数，读取scale0.5；不是共享权重。
- `OTGuidedAttention:360` 构造Q/K/V/O、scalar gate、confidence与residual/norm。
- `initialize_model_ot_prior:196 → initialize_candidate_sparse_ot_prior(stage_model.py:856)`。根据 bidirectional 选择 `compute_initial_bidirectional_candidate_sparse_uot_prior(sparse_uot.py:633)`，不是dense初始化。

初始 bidirectional prior 的数学步骤：相邻section A/B；每个模态做 A→B 与 B→A 检索（`build_faiss_candidates:229`）；所有检索结果映射到统一的 `(A_local_index, B_local_index)`、去重；候选边上计算各模态 cosine cost 后求平均；`_prune_edges_bidirectional:305` 保留行top-candidate_k与列top-candidate_k的**并集**；`sparse_unbalanced_sinkhorn_bidirectional_topk:402` 只解一次coupling；`_edges_to_sparse_topk:340` 从行、列分别提取两个方向的attention prior，保存至 `model.ot_prior`。

注意：candidate_k 是候选检索/裁剪参数，不是最终每行度数的严格上限（双向并集可超过它）；attention_topk 才是coupling截取宽度。反向prior的source是B、target是A，反向权重按原coupling列归一化，不能直接转置一个已经行归一化的topk表。

另一路dense不是同一数学的稠密存储版：`linkage_construction.py:222-281`先做跨section的模态zscore，每个模态各解UOT，再平均coupling；sparse是先平均模态cost再解一次UOT。dense confidence在`:118`为row_mass相对uniform mass的截断比值；sparse在`:384`为top-k coverage。dense的tol/check_every提前停止与sparse固定max_iter也不同。统一edge solver时必须保留这些算法差异，不能仅切一个backend就覆盖它们。

### 4.2 一次 training forward 与 backward

MouseBrain `run_mousebrain_v2.py:677-696`；其余主要数据集 `run_crc_stereocite.py:795-853 → run_one_forward:652 → StageMultiModalModel.forward:1016`：

```text
feature_dict + spatial_loc_dict + section_order
  → 校验section/spot数量/模态维度；转FP32输入并选择device
  → 每个section各自的 spatial KNN graph
  → 每个模态 ModalityMLPEncoder
      ├→ 所有模态对的 COSIE crossview loss
      └→ FusionMLP(concat + mean residual + LayerNorm) → fused z
  → spatial graph + 已缓存的 feature graph（仅pre-OT；epoch1训练时尚无feature cache）
  → pre-OT WeightedResidualGraphSAGE → graphsage_embeddings
  → 从缓存OT prior取邻居，双向OTGuidedAttention.compute_update_only
  → 每个section对收到的相邻section updates求平均，再apply_update一次
  → ot_embeddings
  → post-OT WeightedResidualGraphSAGE，仅section内spatial graph、scale0.5
  → final_embeddings
  → 每模态ModalityDecoder → MSE(preprocessed feature target)
  → total = lambda_reconstruction * sum_section,modality(weight * MSE)
            + lambda_contrast * sum_section,modality-pair(crossview)
  → zero_grad → backward（FP16时GradScaler）→ Adam.step
```

对应位置：空间图 `stage_model.py:579,1168` / `utils.py:219`；encoder `:1187-1198` / `model_component.py:32`；loss `:1210-1220` / `loss.py:74`；fusion `:1231-1256` / `model_component.py:76`；pre `:1305-1324`；双向 `:1346-1400`；post `:1457-1474`；decoder/MSE `:622-720,1489-1506` / `model_component.py:700`。

空间图按section单独调用，`edge_index[0]` 是接收消息的本section query，`edge_index[1]` 是邻居。原始权重 `exp(-d²/(sigma_i²+delta))`，self原权重1；加反向边，以max合并重复空间边，最终按接收行归一化。默认 `no_self_linear` 保留adjacency self-loop和外residual，仅旁路learned self projection。GraphSAGE实际为 `LayerNorm(x + scale * Dropout(GELU(W_neigh * weighted_neighbors + bias)))`；scale=0时仍有LayerNorm，不能当成完全bypass。

feature graph 来自每个section的 `fused.detach()`、L2 normalize、FAISS FlatIP、自排除；k_feature=(k_spatial+1)//2。feature边原追加权重1/k_spatial，和**已经归一化**的spatial边拼接后再次按行归一化（`stage_model.py:567-577`）。相同空间/feature邻居目前叠加两条贡献，并非先去重；重构不能无声改成max/均值。post不消费feature graph。

attention score = Q·K/sqrt(d_attn) + beta·log(topk_weight)，beta当前0.2；V/O形成message，gate读 `[source,message,source-message,source*message]` 的512维输入，乘confidence，再对多方向update求均值，最后仅一次residual/dropout/LayerNorm。三section中间section不是串行先左后右更新；所有方向使用同一批 pre-OT embedding。context gate默认关闭，开启时多一个context reliability输入（513维）。

这里的modality embedding就是各MLP输出的latent；未发现另外叠加一个可学习的modality token/table。不要在整理时把领域描述中的“modality embeddings”误实现成新增embedding层。

### 4.3 一次 dynamic OT refresh

生产循环共三处：MouseBrain `:715-741`、CRC共享trainer `:916-990`、generic trainer `:313-349`。均在 optimizer step 之后：

1. epoch1 或满足 `should_update_ot(epoch, interval)` 时，`model.eval()` + `torch.no_grad()`，用旧feature cache/旧prior做 `refresh_pass_1 = forward(...)`。
2. `model.refresh_feature_graph(refresh_pass_1['fused_embeddings'])`（`stage_model.py:539`）：以当前权重的fused z构建/替换section feature graph。epoch1到此结束，**不刷新OT**。
3. epoch100/120/...再做 `refresh_pass_2 = forward(...)`，其pre-OT图已是步骤2的新图，OT attention仍使用上一版prior。
4. `update_model_ot_prior`（MouseBrain `:208` / CRC `:732`）选source；主suite为ot；`prepare_ot_prior_refresh(stage_model.py:722)`取 `refresh_pass_2['ot_embeddings'].detach()`。
5. topology默认开启、weight0.2；从同一ot embedding及section spatial graph计算self-excluded、归一化local context（`:160,795`）。候选检索用ot semantic embedding；candidate cost为 `0.8*cosine_cost(ot) + 0.2*cosine_cost(local_context)`，不是用context代替检索源。
6. `update_candidate_sparse_ot_prior:958 → update_bidirectional_candidate_sparse_uot_prior_from_embeddings(sparse_uot.py:1075)`：双向FAISS→统一A/B index→去重→semantic/context cost→行列候选并集裁剪→同一个sparse UOT solver→两方向topk→整体替换 `model.ot_prior`。
7. 下一epoch开头 `model.train()`；refresh没有给encoder/attention反传梯度，后续训练对固定prior上的attention投影/输入正常反传。这是当前训练算法的明确语义。

若epochs=200，epoch200 step后的新prior直接用于最后的eval forward，未再经历一个optimizer step。这是当前状态定义，不是保存了“最后一次训练forward”的embedding；停止在199时prior来自180，仍是当前缓存状态。不要在等价重构中擅自补刷或省掉末epoch刷新。

### 4.4 最终输出、checkpoint、评估

最终 `model.eval()` + no_grad forward：MouseBrain `:743-757`，CRC `:993-1019`，generic `:363-370`。保存 final_embeddings（post-OT空间传播之后），保存当次forward实际使用的 `model.ot_prior` top-k（可选）、history及run_summary。loss_history是每次step前、train/dropout状态的loss；summary loss是末次eval loss，两者不要求相等。

MouseBrain `save_run_artifacts:490` 拷贝的是输入配置，summary补充部分CLI值，**不是完整resolved config**。CRC/MISAR/SPATCH会保存更丰富的h5ad、spot metadata、embeddings、QC和summary，行为受save flags控制；sparse prior是section内局部index，其解释依赖同序spot metadata，不能单独当全局barcode。

模型权重checkpoint只有 `train_stage_model.py:231-238` 和 `run_human_embryo_rna_only.py:315-317`；未发现生产用的Stage checkpoint loader/resume。`validate_checkpoint_ot_attention.py` 的load_state_dict用于比较**activation checkpoint**，不证明旧模型文件可恢复。`image_preprocessing.py:170` 加载UNI权重是另一对象。保存checkpoint不包含OT prior/feature cache/optimizer/RNG，因此不等于可精确重启的训练状态，见R2。

默认suite评估由 `analyze_result_v7a_requested_metrics.py:260` 驱动，使用 `analyze_mousebrain_v4_standardized`、`analyze_result_v3_standardized` 的loader和各 `compare_*_kmeans_preprocessing` 的数据/指标函数。final embedding在训练保存时不再standardize；评估对joint全体fit StandardScaler，对independent section分别fit StandardScaler，再KMeans，计算ARI/NMI与内部指标、batch指标。SPATCH由 `analyze_spatch_v7_requested_metrics.py:176` 标准化后用MiniBatchKMeans；部分旧分析入口使用raw embedding或不同指标空间。现有输出config通常已注明standardized，此差异本身不是bug，但不同入口产物不可当成同一评估协议。

## 5. 每个概念的实现位置与分类

分类按运行证据，而非版本名：ACTIVE含主suite实际使用和仍存在的入口默认分支；EXPERIMENTAL是有明确实验用途而非主suite设定；COMPATIBILITY必须说明对象；DEAD限本仓库已检查的调用范围，不代表已承诺任意外部notebook API稳定。

| 概念 | 实现与实际调用 | 分类/处理 |
|---|---|---|
| model class | `stage_model.StageMultiModalModel:255`，所有模型runner共用 | ACTIVE，保留单一实现 |
| modality encoder | `model_component.ModalityMLPEncoder:32`，`_make_encoder:453` | ACTIVE；没有另一套v7/v14 encoder |
| fusion | `FusionMLP:76`，多模态组合ModuleDict；单模态identity | ACTIVE；单模态为EXPERIMENTAL/独立dataset路径，不删 |
| spatial builder | `utils.compute_spatial_knn_graph_with_weights:219`，Stage `_get_spatial_graph:579` | ACTIVE |
| unweighted graph | `utils.compute_knn_graph:163` | DEAD候选，详见下表 |
| feature builder | `utils.compute_feature_knn_graph:309`、Stage `refresh_feature_graph:539/_get_pre_ot_graph:554` | 当前ACTIVE；科研属性是v15方向实验，需显式配置隔离 |
| GraphSAGE | `WeightedResidualGraphSAGE:156`，pre/post两实例 | ACTIVE；三个self_path_mode为同实现参数差异 |
| sparse UOT | `sparse_uot.py:402,480` 两个重复solver循环；四个initial/update方向wrapper | 双向ACTIVE；单向有CLI/API/test调用，EXPERIMENTAL；先统一solver不能先删wrapper |
| dense UOT | `linkage_construction.py:38,202,295`，Stage initialize/update | ACTIVE（裸CLI/generic默认），科研主suite外；只在入口迁移后才可列COMPATIBILITY |
| candidate retrieval | `faiss_candidate_search.build_faiss_candidates:229`，IVF/FlatIP/blockwise | IVF/Flat用于实际运行；blockwise用于测试及无FAISS替代，均非DEAD |
| OT refresh | Stage `prepare/update*` + 三trainer调用 | ACTIVE；重复的是orchestration、kwargs、配置，而非三套独立模型 |
| context pooling | `model_component.spatial_pool_self_excluded:289`；`stage_model.compute_self_excluded_spatial_context:160`在`:206`调用前者 | ACTIVE；后者是section校验/diagnostics wrapper，底层聚合已单一，不应当作重复算法 |
| OT attention | `OTGuidedAttention:375`，compute_update_only/apply_update/forward/checkpoint helper | ACTIVE；这些是分阶段职责，不是可删旧类wrapper |
| decoder | `ModalityDecoder:700` + Stage `_decode_and_reconstruct_section:622` | ACTIVE；chunk/full共用同decoder，保留grad路径 |
| reconstruction helper | `loss.compute_reconstruction_loss:120` | DEAD候选；Stage拥有实际MSE实现 |
| crossview loss | `loss.compute_joint:14 → crossview_contrastive_Loss:33 → compute_pairwise_cosie_crossview_loss:74` | ACTIVE；不能因为名字像历史COSIE就删 |
| trainer | CRC `train_small_crc_model:774`，MouseBrain `run_mousebrain:512`，generic `train_stage_model:262` | ACTIVE；应合并shared loop，保留dataset准备步骤 |
| activation checkpoint | Stage `:60,72,102`、fusion `:1241`；Attention `:561`附近 | ACTIVE（大数据suite）；不是checkpoint文件兼容层 |
| checkpoint files | generic/HumanEmbryo两个save位置；UNI另有loader | 有保存，无Stage恢复入口；不能虚构迁移loader |
| evaluation | requested_metrics → historical named loaders → compare_* + batch_correction_metrics | ACTIVE，很多旧名字实际是当前库代码 |
| preprocessing | `load_cosie_style_data:474 → load_data:127 → preprocess_adata:52`；multimodal adapters；image路径 | ACTIVE；alias是仍在使用的旧接口兼容，先迁调用再合名 |
| CLI/config | configure默认 + runner argparse/手写merge + argv注入 + suite globals | ACTIVE；配置权威分散，是优先重构对象 |

### 5.1 可高置信度删除的 DEAD 候选

逐个查过所有生产Python、tests、tracked JSON/Markdown、README命令、动态 `exec/importlib/getattr`、模型exports与checkpoint loader。无shell/notebook入口，无model registry或wildcard import提供隐藏调用。历史文档中的代码清单或对外部COSIE的描述不等于可运行调用者。

| 位置 | 为何存在/为何可删 | 删除后的替代与验证 |
|---|---|---|
| `model/utils.py:163 compute_knn_graph` | 外部COSIE迁入的旧unweighted KNN；本仓库无执行调用；文档引用主要解释COSIE旧结构 | 当前图全部用weighted builder；删除后import/graph smoke |
| `model/loss.py:120 compute_reconstruction_loss` | 老MSE helper；文档仍列出，但无Python调用/export/动态引用 | 当前Stage chunk/full MSE已权威；做loss/grad等价检查 |
| `model/utils.py:63 standard_modality_result` | 没有调用/导出/配置命名；adapter已直接生成dict | 删除不改变pipeline |
| `model/utils.py:144 summarize_adata` | 无调用；实际有各runner和multimodal summary函数 | 保留真实使用的summary schema |
| `model/stage_model.py:233 summarize_ot_topology_cost` | 无调用的旧诊断汇总；prior metadata仍由真实OT函数生成 | 删helper，不删metadata |
| `model/data_preprocessing.py:423,440,457 preprocess_rna_adata / preprocess_protein_adata / preprocess_metabolite_adata` | 三个薄wrapper只有定义，无exports或命令调用 | 活跃路径直接 `preprocess_adata`，可删三个wrapper |
| `model/image_preprocessing.py:66 rescale_image` | COSIE迁入resize helper；除定义/来源注释、旧文档清单外无调用 | 当前image extraction链未使用；可删helper，不删raw-image输入功能 |

这是9个函数的候选集合，不是批准删除整文件。`compute_knn_graph` 还有重复坐标时以最近邻首index冒充query index的缺陷（`:210`）；因DEAD不建议先投入修补，再保留一套重复图算法。

### 5.2 不能直接删除的兼容与实验代码

- `processed_data_dict` forward参数及位置参数解析 `stage_model.py:1035-1042`：兼容 `model(features, spatial, order[, epoch])` 与旧pipeline签名。当前runner仍传processed data，数学上不使用；迁完调用可删位置兼容，不能先改签名。
- `WeightedResidualGraphSAGE.self_linear` 在 `no_self_linear` 模式仍存在（`model_component.py:195-198`）：为G0消融各模式的state_dict键兼容；legacy/no_adj_self确实会使用。没有生产旧checkpoint loader并不代表可以在保留实验mode时删除参数。可在确定退休G0 modes后删；单纯默认旁路不是DEAD。
- `load_cosie_style_data` → `load_data` alias：当前所有主要预处理仍通过它，属于ACTIVE中的接口兼容；可以迁为一个名字，但不是“无调用wrapper”。
- `get_default_model_config` 中记录性旧字段见下一节；能移到preset说明的应移走，不应永久保留可写但无效的开关。
- CRC `--dry_run`和`run_stage_model --smoke_test`是仍被文档命令使用的COMPATIBILITY标志：前者真正行为由`--train`决定，后者无条件smoke。可以迁移/删冗余CLI，但并非控制数学的失效消融开关。
- v4/v4B/v4C/v5/v6的suite/daemon：仍有main、历史命令，且互相import和monkeypatch。没有用户明确退休这些实验的证据，归COMPATIBILITY（历史运行/结果协议），不能直接DEAD。若确定不再重跑，可删除其启动wrapper，但保留已被当前路径依赖的函数直到迁完。
- v7B/v7C：仍是可运行消融，EXPERIMENTAL；已经主要是scale/config差异，不需要新model class。
- v15A/v15F：EXPERIMENTAL suite；其feature实现已经污染共享ACTIVE默认。应把开关与refresh timing固定在各自preset，不能根据版本号继续增加类。
- v14A/B、v12/v13、v15B-E：当前没有同名模型/源码选择分支，只有结果目录等记录。本阶段没有恢复或推测这些模型实现；若未来仍需实验，先找对应可验证源码/配置快照，再决定graph source/aggregation参数。不能凭旧产物名字给它们编造当前call chain。
- 单模态、context gate、fused/final refresh source、self_path_modes、dense、blockwise：都有显式调用/入口/测试，留作明确preset或测试基准；不属于DEAD。
- metacell构造和 `reconstruct_metacell_to_original:409`：前者仍由config触发，后者是其潜在下游恢复接口，保守归EXPERIMENTAL/COMPATIBILITY，待是否退休metacell共同决定；不要因本轮suite未用就删除整链。
- image preprocessing/UNI：`build_section_modalities` 仍支持从raw image/mask或feature file构建HE，是数据入口功能；不是旧模型分支。
- 三个gene-imputation目录的v6配置：README提供当前可执行命令，且新模型可替换embedding路径。Annoy均值插补及可选来源空间平滑与decoder重构是不同任务。不要按v6标签删除。

## 6. 配置权威、失效字段与可参数化版本

真正控制计算的是Stage constructor/forward所读取的值和trainer实际传入参数，不是configuration里出现的所有键。

| 字段/参数 | 实际情况 |
|---|---|
| `graph.use_spatial_graph` | 未作为构图或传播开关；设False仍构图 |
| `graphsage.num_layers` | 未控制层数；pre/post各固定一个layer，设9也不改变 |
| `graphsage.use_distance_weight` | 未被读取；始终使用weighted builder |
| `encoder.type/residual` | 记录性字段；实际固定MLP，不实现encoder residual开关 |
| `fusion.mode/input_dim` | mode未选择路径；输入维度从模态数×latent_dim计算，单模态走identity |
| `contrastive.method/loss_weight/pairwise_all_observed_modalities/use_infonce/use_temperature/use_spot_positive_negative_pairs` | 计算固定为所有观察模态对COSIE公式；实际总权重是 `loss.lambda_contrast` |
| `uot.initial_from_modalities/update_from_final_embedding/use_momentum/momentum/normalize_total_mass/cost` | 大部分为旧记录/无效配置；初始化与refresh由入口显式方法选；coupling normalization在solver固定执行，momentum未执行 |
| `ot_attention.direction` | 没有驱动forward方向；实际是独立布尔 `bidirectional_ot_attention` 加 prior keys |
| `uot.dynamic_refresh_source` | API prepare默认会读、forward context条件会读，但生产trainer另从CLI或硬编码选source，两者会不一致（B3） |
| `uot.max_iter/tol/check_every/clip/epsilon_init` | dense路径消费；sparse使用独立kwargs，只固定次数迭代；不能把dense参数报告为sparse已生效 |
| `reconstruction.loss`、`loss.use_ot_loss/use_spatial_smooth_loss/use_gate_regularization` | 不选择loss分支；实际总loss只有crossview+MSE |
| `post_ot_graphsage_scale` | constructor实际读取且生效；不是无效参数，但CLI/JSON precedence有bug |
| checkpoint/chunk/training_loss_only/cache flags | 真实进入计算/显存分支，不能删为“复杂兼容参数” |

无效字段不意味着必须实现它承诺的所有功能；推荐保留已有数学，删除记录性可写配置或对不支持的非默认值给出明确错误。研究人员以为做了消融而实际上没改变计算时，这是correctness问题（B8），而不是unused变量风格问题。

| 版本/实验 | 应表达成的显式差异 | 限制 |
|---|---|---|
| v7A/B/C | post scale 0.5/0.25/0.75；同一Stage/GraphSAGE | 还须固定feature=False等完整基线；现有三个JSON只覆盖scale |
| v6→v7 | post residual系数差异可preset化 | 不能只换输出目录保证历史行为 |
| G0 variants | self_path_mode | 保留self-loop、自投影、外residual各自语义 |
| OT source实验 | refresh.source = fused / ot / final，topology_weight，first_epoch/interval | first_epoch目前硬编码；必须显式化 |
| context gate实验 | context_gate_enabled、是否对alpha反传 | state_dict尺寸有差异，需记录 |
| v15A/F方向 | pre graph source spatial或spatial+feature；feature k/weight；初始化/刷新时机与两次forward | 当前行为已经共享，历史A与F的区别不能只靠preset名称恢复 |
| 历史v14等 | 在证据恢复后用post graph source、aggregation、scale表达必要差异 | 当前无实现可确认，不预建全套“版本兼容”框架 |

## 7. 已确认 bug 与条件性 correctness 问题

P1：会使合法长训练失败、无声改变实验或结果失真；P2：非主默认组合/可恢复局部问题；P3：测试/诊断集成问题。分级结合触发范围，不按代码观感。

### B1 — 显式 MouseBrain CLI epochs/device 被 JSON 覆盖（P1）

- **位置**：`scripts/run_mousebrain_v2.py:354-366`，目录名又在`:540-543`提前由CLI计算。
- **触发**：JSON含嵌套 `model.training.epochs/device`，同时给 `--epochs` / `--device`。
- **当前/预期**：先应用CLI后merge嵌套model，实际按JSON训练，目录可能仍叫CLI的epoch数；应遵守声明为override的显式CLI优先。
- **依据与最小验证**：AST抽取原函数执行，JSON `{model:{training:{epochs:999,device:'cuda'}}}`、调用epochs=2/device=cpu，实际返回999/cuda。无需装载数据；根代理独立复现。
- **建议修复**：先合并defaults与JSON，最后仅应用用户显式CLI；从resolved config生成输出目录和summary。

### B2 — 未显式输入的 CLI 默认覆盖 JSON 模型配置（P2）

- **位置**：MouseBrain `parse_args:65-101`、`build_model_config:369-383`。
- **触发**：JSON设置scale0.5、update_interval7等，省略对应CLI参数。
- **当前/预期**：仍被函数默认scale1、interval20等强制覆盖；应保留JSON，只有显式CLI覆盖。
- **依据与最小验证**：与B1同一AST试验，实际scale=1.0/interval=20；JSON还可被覆盖topk、tau、max_iter、spatial k、context gate。
- **建议修复**：override类参数默认None/显式provided集合；统一解析顺序，保存resolved values。不要把默认修改成0.5后就当precedence已修好。

### B3 — 合法 fused refresh CLI 在首刷新时崩溃（P1）

- **位置**：`stage_model.py:1272-1288,767-775`；MouseBrain `:209-216`；CRC `:740-745`；各config builder未写回source。
- **触发**：candidate_sparse、`--dynamic_candidate_source fused`、默认context gate=False/topology=True；模型config里的source仍ot。
- **当前/预期**：eval forward不生成fused contexts，返回空dict；prepare看到Mapping就当有效cache，报 `Missing fused spatial contexts`。应从选定fused embedding计算相应context并正常更新prior。
- **依据与最小验证**：7+8或8+9 spots单次eval，无训练；默认 `context_embeddings={}`，prepare(fused)稳定KeyError；对齐内存config为fused再eval则正常。生产将到epoch100才报错，影响v4/v4C等仍可运行实验。
- **建议修复**：只保留一个resolved refresh source；prepare仅复用覆盖全部section的有效context，否则从实际source与spatial graph计算。

### B4 — sparse top-k 补位参与 attention（P2）

- **位置**：`sparse_uot.py:358-383`（双向补位）；`:544-550`（单向无效候选）；`model_component.py:618-620`。
- **触发**：某source有效candidate不足attention_topk，包括IVF返回-1、低度数行；补位权重为0。
- **当前/预期**：0被clamp为delta，softmax给虚拟候选正权重；应对不存在的边mask掉，候选不足不改变真实support上的归一化attention。
- **依据与最小验证**：两真实边mass[.5,.5]，宽3得到index[0,1,0]/weight[.5,.5,0]；令Q/K=0、V/O=identity、gate=.5、beta=.2，padding版update=2.46444559，无padding为2.5。无需训练。
- **建议修复**：传递valid mask或以真实support mask scores=-inf；全空行显式zero update。保留正mass边的log prior与confidence语义。此为数值行为修复，不能混入纯rename等价声明。

### B5 — feature graph 关闭后仍在epoch1构图（P2）

- **位置**：`stage_model.py:539-552`；三trainer epoch1 unconditional调用（MouseBrain726、CRC947、generic322）。
- **触发**：`graph.use_feature_graph=False`，OT用dense/blockwise，环境无FAISS。
- **当前/预期**：仍强制执行FAISS特征KNN，epoch1后ImportError；关闭的feature graph应无构图副作用或额外依赖。
- **依据与最小验证**：False config+8spot fused，mock `_load_faiss`抛ImportError，直接refresh仍抛错。正常有FAISS环境也做了无用查询。
- **建议修复**：refresh方法及trainer日程尊重开关；该关闭路径不再读取feature cache。勿改变开启路径的两次forward时序。

### B6 — feature graph 强制FAISS GPU，绕过可用CPU检索配置（P2）

- **位置**：`utils.py:319-325`、`faiss_candidate_search.py:49-77`。
- **触发**：Torch CUDA可用但只有faiss-cpu，OT CLI指定faiss_device=cpu/auto；feature graph开启。
- **当前/预期**：feature builder按fused.is_cuda强制gpu并报错；应有明确feature检索device策略，允许CPU FAISS处理CUDA模型产生的detached embedding。
- **依据/最小验证**：静态调用及工具分支可确定；以GPU API缺失的fake faiss/mock索引验证，不必真实训练。当前cosie环境的GPU组合未跑。
- **建议修复**：传入统一或显式feature_device=auto/cpu策略；保留FlatIP作为feature检索算法，不能顺便换IVF并声称完全等价。

### B7 — dense + bidirectional 在CRC路径静默单向（P2）

- **位置**：CRC parser `:84-85`、校验`:1025-1054`、initialize`:718-719`；Stage `:1346-1400`。
- **触发**：CRC/其适配入口选择dense同时请求bidirectional。
- **当前/预期**：只初始化正向prior，所谓双向forward遍历现有prior，末section无反向更新；应拒绝不支持的组合或真正提供反向prior。
- **依据/最小验证**：dense初始prior只含(A,B)，把两section forward的bidirectional flag设True仍没有(B,A)；MouseBrain已经在`:515-518`拒绝此组合，说明入口不一致。
- **建议修复**：统一CLI验证及prior方向完整性；保留dense数学时不要伪装为双向sparse。

### B8 — 可配置的图/方向/损失开关未影响计算（P2）

- **位置**：`configure.py:112-145,152-169,171-241`；Stage构造`:329-376`、forward`:1168,1216,1311,1346,1506`。
- **触发**：用户以 `use_spatial_graph=False`、`use_distance_weight=False`、`num_layers=9`、`ot_attention.direction`、`contrastive.loss_weight`等做实验。
- **当前/预期**：第6节字段保持默认计算，却可能被保存为实验参数；应生效或明确拒绝，不应产生“已经做过该消融”的假象。
- **依据/最小验证**：同权重8spot eval，上述三个graph字段改变后仍56条spatial边，final逐位相同；源码无对应switch。方向由另一个forward布尔决定，总loss只读lambda_contrast/reconstruction。
- **建议修复**：清除记录性字段/迁为只读描述；对不支持的配置值报错，真正保留的实验开关接到唯一计算路径。不是要求实现InfoNCE等新功能。

### B9 — 同目录重训后评估复用旧 metrics（P1）

- **位置**：`run_spatch.py:632-639`，`analyze_spatch_v7_requested_metrics.py:178-196`；通用requested analyzer `:91-138,455-462`；v15F suite `:35`自动追加overwrite。
- **触发**：run目录已有analysis completion/manifest，再训练或替换embedding，同scale；重新分析。
- **当前/预期**：SPATCH看到completion直接return，未读当前embedding，也没比较其已记录summary SHA256；通用脚本只验CSV列/有限性。应验证metrics属于当前embedding、obs顺序、resolved config和评估协议。
- **依据与最小验证**：AST执行原analyze函数，用内存FakePath模拟新run_id和旧completion，将load_embedding替换为若调用就raise，实际返回OLD，未调用loader。根代理交叉核对所有早退与overwrite分支。
- **建议修复**：用run/embedding/metadata/protocol指纹关联缓存，失配拒绝或重算；独立运行目录优先。不要通过只检查文件存在认定评估已完成。

### B10 — Simulation/Thymus 审计产物忽略用户输入输出路径（P2）

- **位置**：Simulation `:94-147`、Thymus `:142-189`，main在parse前调用 `_write_adapter_audit()`。
- **触发**：用户或suite给其他 `--output_dir` / `--data_dir`，甚至请求 `--help`。
- **当前/预期**：先读硬编码DATA_DIR并写硬编码旧result_v4目录；之后训练才解析用户路径。应从resolved args读取和保存该次运行的数据审计。
- **依据/最小验证**：静态顺序确定；mock `_write_adapter_audit` / parser记录事件和paths即可，未运行会读真实数据的help命令。
- **建议修复**：parse一次后把data_dir/output_dir明确传给adapter/audit；让help在任何数据I/O之前返回。

### B11 — spatial graph cache对相同shape的新坐标返回旧图（P2，条件性API）

- **位置**：`stage_model.py:599-620`，cache key只有section、n、k、自环等，无坐标身份/版本。
- **触发**：复用模型，cache_spatial_graphs=True，同section和spot数但坐标或spot顺序改变。
- **当前/预期**：沿用旧edge/weights；应重建或明确要求/检查输入不可变。
- **依据/最小验证**：同section/count更换小矩阵坐标，cached结果不变而uncached结果改变，CPU已证实。当前一次固定数据训练不会触发。
- **建议修复**：将graph作为固定输入显式准备一次，或给坐标/spot-order版本绑定cache；保留clear接口，避免每个epoch做昂贵全hash。

### B12 — bfloat16 candidate输入在转numpy前未转float（P2，条件性API）

- **位置**：`faiss_candidate_search.py:16-22`；sparse转换helper有相同`.numpy()`顺序。
- **触发**：直接把BF16 tensor传candidate retrieval/prior接口。
- **当前/预期**：`.cpu().numpy()`报 `Got unsupported ScalarType BFloat16`；既然接口统一为float32，应先tensor.float再numpy。
- **依据/最小验证**：2×D BF16 tensor调用转换函数即可，CPU已复现。当前生产refresh在autocast外，通常是FP32，不能说主训练必然失败。
- **建议修复**：detached tensor先转float32，再cpu/numpy；保留检索无梯度语义。

### B13 — shared modality预处理遗漏target_sum（P2）

- **位置**：`data_preprocessing.py:221-231` 对比 `:271-280`；`load_cosie_style_data:491`、MouseBrain `:325`、generic trainer `:125` 均可传入。
- **触发**：同一模态出现在多个section，配置非默认target_sum。
- **当前/预期**：shared分支调用preprocess_adata未传target_sum，unique分支会传，造成同一配置因section数量改变语义；应保持显式归一化参数一致。
- **依据/最小验证**：AST执行原load_data、stub preprocess_adata记录kwargs，target_sum=12345时shared调用缺参数、unique明确收到12345，已实测；没有运行真实AnnData归一化。当前主suite target_sum=None不受影响。
- **建议修复**：共享分支传入resolved target_sum；验证默认None输出等价及非默认值确实传达。

### B14 — 自定义section_id与通用训练预处理输出不一致（P2）

- **位置**：`multimodal_preprocessing.py:340-342,372-389`；`data_preprocessing.py:290,305`；`train_stage_model.py:130`；Stage section解析`:385`。
- **触发**：`--preprocess_config` 的section_id为A/B等，而不是s1/s2。
- **当前/预期**：预处理返回feature/spatial keys=s1/s2，section_ids=A/B；trainer把A/B传给模型，报缺section。应在一个位置保留真实section ID与顺序映射。
- **依据/最小验证**：AST执行真实preprocess_multisection和generic load_preprocessed_inputs，stub最底层load按真实合同返回s1/s2；实际得到feature keys=s1/s2、order=A/B，再执行真实Stage._resolve_section_order立即KeyError。现有MouseBrain默认s1/s2/s3不触发。
- **建议修复**：在预处理出口一致重命名feature/spatial/metadata，保存local-to-global spot mapping；不能仅修改显示标签。

### B15 — v7A suite默认分析集合超出训练集合（P2）

- **位置**：`run_result_v7a_suite.py:104-165,204-213`；`analyze_result_v7a_requested_metrics.py:81-87,444,455-462`。
- **触发**：当前工作树运行v7A suite，未显式限定analysis datasets。
- **当前/预期**：suite训练五个数据集，但analysis默认SPECS已加入Human Lymph Node成为六个；全新目录缺其输入，已有五数据集manifest也因expected datasets不一致而失败。suite分析应只消费自身训练集合。
- **依据/最小验证**：静态比较training_commands返回的数据集集合与analyzer默认SPECS；检查analysis_command没传`--datasets`。无需训练。v15F已显式传dataset列表，不能说明v7A调用也已修好。
- **建议修复**：suite显式传同一份dataset选择给训练/分析/验证；保留已有五数据集manifest协议，或明确新一轮六数据集run。

## 8. 数学与状态风险：已证实机制，真实数据影响仍待验证

### R1 — signed latent的crossview joint不是稳定概率分布（高）

**位置**：encoder `model_component.py:66-72` 末Linear仅L2 normalize；`loss.py:25-27,47-69`。**触发**：合法signed embedding使joint总和0或接近0。**当前行为**：直接除以总和；负项/无穷产生后再clamp不能恢复分布。**预期**：定义良好、有限的loss与梯度；如果是概率joint应满足非负与归一化假设。**依据**：`z=[[1,-1],[1,-1]]/sqrt(2)`，`compute_joint(z,z)`得到±inf，crossview为NaN，根代理与子代理独立CPU复现；普通signed输入也不保证概率解释。**最小验证**：上述两行、加近零和用例及backward。**修复建议**：先明确保留COSIE历史signed目标还是改用满足概率假设的投影；softmax/其他objective/分母处理都可能改变科学方法，必须单独数学修复与消融，不得夹在等价重构里。这里只证明允许输入的失败及假设问题，未证明现存真实训练已NaN；负loss本身不是bug。

### R2 — 当前checkpoint不能精确重建最终状态（高，能力缺口）

**位置**：`train_stage_model.py:231-238`、HumanEmbryo`:315-317`、Stage`:311-312` 的普通dict缓存。**触发**：试图仅load_state_dict重建保存的最终embedding或resume。**当前/预期**：OT prior、feature graph不在state_dict，optimizer/RNG也未保存；重建需要相同权重+同一实际graph/prior/spot order。**依据**：根代理8+9spot验证，相同state_dict并额外复制相同prior，单独遗漏feature cache，final最大差0.2277677。**最小验证**：两模型相同state+prior、一个refresh过feature graph一个未refresh，eval对比。**建议**：如果需要restore，增加一个直接、明确的checkpoint payload和loader，存resolved config/graph/prior/epoch/spot-order，resume再存optimizer/RNG；如果只导出weights则命名和说明明确。当前没有生产resume入口，不能把能力缺口误报为“现有loader silently变更模型”。

### R3 — Sinkhorn kernel floor显著改变高cost边的相对权重（高，数学近似）

**位置**：`sparse_uot.py:431,523`，dense `linkage_construction.py:71`。**触发**：epsilon=.05、stabilizer=1e-8，cost≥0.921034。**当前/预期**：所有这些K被拉到同一floor；精确的exp(-cost/epsilon)应保持cost差异。**依据**：2×2 cost=[[1,1.2],[1.2,1]]得到row weights [.5,.5]，对称精确kernel为约[.982014,.017986]。**最小验证**：固定全support小矩阵比较floor solver与不下限/log-domain高精度参考；记录真实候选cost落在floor以上比例。**建议**：先暴露诊断，若需要严格UOT改log-domain或经验证的稳定实现；归数学变更，不与单向/双向solver去重同时改。当前并非dense/sparse谁“更新”的问题，两个都有该近似。

### 尚未证实为主路径bug的事项

| 疑点 | 已知证据与还需验证 |
|---|---|
| AMP FP32保护不足 | `loss.py:47-48`先float不能阻止autocast内matmul重新BF16；`model_component.py:668`的bmm同理。CPU autocast已观察dtype降低；CUDA真实梯度/误差需测。建议数值敏感运算局部禁autocast，单独验效果 |
| decoder target dtype | Stage`:653`转为final dtype，随后MSE又float；若final是BF16会先量化FP32 target。当前正常GraphSAGE/residual常把final保持FP32；需按实际config记录dtype，不能直接说默认目标已错误量化 |
| chunk vs unchunk dropout | dropout0时应数学等价；dropout>0改变调用分块可改变随机mask分配，即使seed相同，也不承诺逐元素相等。固定chunk布局的checkpoint前后应同mask且grad等价 |
| FAISS近似与tie/seed | IVFFlat相对Flat/blockwise候选不必一样；同分ties、GPU scatter累加顺序可造成差异。当前seed并未启用全部确定性；需固定backend/config/线程并比候选集合 |
| feature/spatial重复边 | 当前拼接后归一化，相同neighbor贡献叠加。这是明确可见数学，未找到证据说设计必须coalesce；不得当作已确认bug擅改 |
| feature更新source | 代码明确是fused.detach，post只spatial；没发现错误使用final的主路径。v14若设计不同需单独证据 |
| mode/no_grad | 三trainer刷新均eval+no_grad，下一epoch恢复train；attention训练输入/QKV未被无意detach。MouseBrain dry_run只有no_grad未eval，故带dropout的dry产物不代表正式eval，属于诊断语义差异 |
| section/spot对齐 | 当前默认数据适配包含obs顺序/坐标一致性校验，OT index是section-local。未用真实大数据再次核验；自定义ID已定位B14，不能泛称所有section映射错误 |
| preprocessing memory_efficient | 合并copy与view、common var交集、Harmony输入需做小数据等价；不能只因节省内存的分支较长就合并 |
| metadata固定字段 | 多runner summary硬写ot_refresh_embedding_key=ot_embeddings（例如MouseBrain`:616,794`），选择final/fused时不准确。应从resolved runtime source生成；实际主suite=ot时正确 |
| 带点output路径 | `utils.ensure_dir:28-36`根据suffix猜文件/目录；名字含lc0.1等时可能只建父目录。许多runner另有mkdir，是否在具体保存入口触发需逐点验证，不能概括为所有训练无法保存 |
| generic bundle中的None模态 | `train_stage_model.py:87-96`直接as_tensor每个value，与Stage接受None表示未观察模态的部分逻辑不一致；bundle schema是否支持None尚需明确 |

## 9. 已执行检查与验证限度

所有Python以 `-B` 或 `PYTHONDONTWRITEBYTECODE=1` 运行；只用已有cosie环境，不安装依赖。没有执行dataset runner的help（某些会提前I/O）、suite、昂贵benchmark、真实500spot训练或完整训练。

| 检查 | 结果 |
|---|---|
| 101个Python源的内存compile/AST | 全通过，无pyc落盘 |
| `run_stage_model.py --smoke_test` | 前5个合成多模态forward/backward case通过；随后失败在`:266`期望错误文本`at least two`，实际为single-modality disabled。是测试契约过期（P3），不是模型拒绝单模态有bug；后面的case未执行 |
| `validate_checkpoint_ot_attention.py --device cpu` | 37/53spots，单向/双向均PASS；loss、final、参数grad maxdiff全部0，prior未变；dropout0、相同chunk布局 |
| `validate_single_modality_mode.py` | RNA/Protein/HE单模态、默认拒绝单模态、双模态回归均PASS |
| attention独立小验证 | dropout=.2、固定chunk=2，checkpoint开关output/source/target/参数grad差0；dropout0、chunk0 vs2最大差≤3.16e-7 |
| sparse solver独立小验证 | 固定相同矩形support，单向/双向solver对应结果一致；不能据此认为不同候选策略也等价 |
| 配置/异常/缓存反例 | B1/B2、B3、B4、B5、B8、B11/B12、B13/B14、R1/R2/R3已用AST或极小CPU矩阵验证；B9以无文件FakePath验证早退 |
| 插补unittest | baseline3 + smoothing3 + shared validation13 = 19 PASS（shared需要显式PYTHONPATH） |
| 默认shared test discovery | 初次9 PASS + scheme2导入ERROR，原因`compute_scheme2_pcc.py:19`绝对import sibling；加`PYTHONPATH=$PWD/gene_imputation_shared_gene_validation`后13 PASS。P3测试集成问题，README直接脚本运行路径正常 |

不能由这些检查宣称CUDA/BF16、大规模内存、epoch100完整真实刷新、全部历史入口都验证通过。现有`validate_experiment_c.py`、`validate_microenvironment_context.py`、bidirectional验证和500spot/2epoch类命令可作为下一阶段候选；尤其需先修正测试中对refresh source/architecture的旧假设。测试过期应更新有意义的行为断言，不只把预期字符串换掉后宣布架构已验证。

## 10. 推荐最终结构：保持直接、少层次

目标不是再创建一个framework，而是把实际数学与训练/数据/评估职责放到找得到的位置。可采用下面的平实结构；不要求一次全部搬动：

```text
model/
  stage_model.py             一个Stage类，直接forward、必要的prior/graph状态方法
  model_component.py         encoder/fusion/GraphSAGE/attention/decoder
  loss.py                    唯一loss数学（decoder chunk loss可留Stage，避免强拆）
  sparse_uot.py              一个edge-based solver，候选cost/topk提取
  faiss_candidate_search.py  IVF/Flat/blockwise检索及device策略
  utils.py                  小图构建与必要工具；若明显过大才单独graphs.py
  data_preprocessing.py      唯一通用模态预处理
  multimodal_preprocessing.py 数据集准备/对齐的普通函数，逐步替代源码exec
  image_preprocessing.py     可选UNI输入链
  configure.py               默认值与一次配置解析，不做plugin/registry
  training.py                一个fit loop、refresh、save；直接调用模型
configs/
  datasets/                 可tracked的数据路径模板/预处理设置
  presets/                  完整v7A及scale/context/feature/source等明确实验配置
scripts/
  train.py                  薄CLI：解析→准备数据→fit
  evaluate.py               薄CLI：读取embedding/metadata→选定评估协议
  run_experiments.py         仅遍历显式preset与dataset组合
evaluation/
  clustering.py             共享标准化/聚类/协议
  metrics.py                唯一指标实现（保留协议明确的变体）
gene_imputation*/            暂保持独立；重复均值/平滑/PCC再按数学核验后收敛
tests/                      小矩阵、forward/gradient、refresh/state/eval缓存检查
docs/history/               旧实验叙述与已退休命令
README.md                   唯一当前命令、默认preset、数据与output契约
```

如dense仍需复现，保留 `linkage_construction.py`，说明dense是不同初始prior算法；待所有调用者退休再删。不要为了“一个OT implementation”让sparse与dense偷偷统一成本预处理/置信度定义。无需service layer、动态registry、factory hierarchy、abstract trainer或多层dataclass包装。

### 10.1 必须明确但本阶段不擅自修改的实验决策

- 若希望未来默认回到历史稳定v7A，建议完整preset明确 `feature_graph=False, post_scale=.5, sparse+bidirectional, refresh.source=ot, first_epoch=100, interval=20, topology_weight=.2`，其依据是历史源码和实际命令，不是用户先验20。
- 当前工作树的feature graph/epoch1/两次forward保存在独立preset（可称当前feature实验），保留相同Stage实现。不能让“v7A”别名继续默认继承未来实验值。
- 是否退休dense、G0 self modes、metacell、旧suite，由实际实验保留范围决定。本文不将尚未退休者列DEAD，也不为它们提前实现永久迁移框架。
- R1/R3的科学数学修复与“结构整理后行为等价”是不同交付，分别记录结果与checkpoint schema。

## 11. 分批实施与每批 behavioral invariants

| 批次 | 内容 | 重构前后必须验证 |
|---|---|---|
| 0：固定可审核基准 | 保存当前resolved config、真实entry命令、git+dirty指纹、小fixture的模型state/prior/graph/spot order；历史v7A与当前feature实验分别建preset | 同一输入、参数、state下可重放；禁止用目录名判版本；0.5和feature开关可直接在实例/graph检查 |
| 1：失败与结果来源修复 | 分开修B1/B2/B3、B5/B6/B7、B9/B10/B13/B14/B15；改P3测试契约；先不做大搬家 | 未触发bug的默认case loss/final/grad不变；显式CLI优先；refresh.source/context一致；输入ID顺序一致；更换embedding后旧metrics不能早退 |
| 2：DEAD与无效配置清理 | 删9个候选函数、更新历史说明；无效字段删除/非默认拒绝；不顺带改loss/OT算法 | imports、CLI读取、已有19下游tests；主模型state keys/初始化顺序和frozen input结果不变；supported preset仍可解析 |
| 3：训练入口收敛 | CRC共享loop迁为training.py；MouseBrain/generic收敛；抽出真正dataset adapter，替换源码replace+exec；suite只传配置 | data preprocessing/obs-var顺序、模态集合、初始prior、Adam参数相同；train/eval/no_grad/AMP/zero_grad顺序相同；epoch1仅feature；100后两次forward与更新时点相同；最终eval/保存对应同状态 |
| 4：数学重复实现收敛 | 单向/双向sparse共享edge solver，保留不同support策略；保持现有单一context pooling；图/decoder只保留权威算法 | 相同support/cost→同coupling/两向topk/confidence；方向与局部index边界正确；chunk/full loss和梯度一致；空间图self/重复边/归一化保持；feature只pre、post scale保持 |
| 5：评估与产物契约收敛 | 统一loader/standardization/metrics，历史脚本薄封装或退休；run指纹与checkpoint用途明确 | joint/independent scaler scope、KMeans vs MiniBatch、k/seed/n_init、metric输入空间相同；label/spot对齐；改变run后缓存失效；恢复state后final同值 |
| 6：单独的科学修复/实验 | B4 masking、R1 loss、R3 log-domain等独立变更；根据已决定的实验范围退休兼容路径 | 不声称旧结果等价；用解析小例/数值参考、finite gradient、新旧结果对照；新数学与preset/schema明确记录 |

建议最小行为基准包括：

1. 不等长2section和3section，小型两/三模态及单模态；固定feature/coords/state，验证每阶段embedding和按模态loss。三section验证中间section双向update平均、一次residual。
2. 固定candidate edge和质量、包含0候选/不足topk/重复检索/tie；检验正反source/target范围、shared coupling行列提取。candidate_k与attention_topk独立取不同值。
3. 原空间图self-loop、反向边max去重、行和；混合feature图目前的重复贡献与权重；修改post scale只能作用post branch。
4. 同chunk checkpoint开关保持dropout RNG及参数/source/target梯度；dropout0跨chunk比较，FP32容差起点1e-5、GPU scatter/BF16容差由测量定义而不是放宽到“都通过”。decoder分块必须累计sum-squared-error/total_numel，不能平均各chunk mean。
5. 人工驱动epoch1/99/100/101/120刷新函数与小state，不需真的训练100epoch；记录pass次数、mode、grad-enabled、feature/prior版本以及喂给cost的embedding来源。
6. 用已构建的同state而非仅同seed验证精简初始化：当前预建所有fusion组合，删未用模块会改变RNG消耗；self_linear旁路仍占参数，删除会改state_dict。初始化/schema变化必须明确记录。
7. 预处理memory_efficient与常规分支的小AnnData等价、非默认target_sum、自定义section IDs、obs/var重排；再验证embedding与saved prior对应同一spot order。

第一阶段到此停止：已能够从真实命令追到一次forward、一次dynamic refresh和最终保存；已区分当前共享路径、可参数化实验、具体兼容对象与有证据的DEAD函数。后续任何生产代码修改属于下一阶段。

## 附录 A：完整可执行入口清单

以下由当前源码AST的 `if __name__ == ...` guard加顶层`runpy`代理核对，共71个scripts入口、10个下游CLI、6个unittest模块。列出的文件具有可执行入口，不代表所有命令在任意数据/环境下都能成功；已确认失败与副作用见正文。没有通过运行所有help或main来枚举，以避免触发真实数据读取、训练或历史结果清理。

`scripts/batch_correction_metrics.py` 和 `scripts/hesta_rna_utils.py` 是两个无main的库模块；前者提供batch指标，后者提供Human Embryo/HESTA数据准备。`model/`是import库，没有独立CLI。

历史suite仍有main/交叉调用，按COMPATIBILITY或EXPERIMENTAL管理；这些入口没有任何一个仅因缺直接import而被列为DEAD。analyze/compare文件中被当前评估导入的函数是ACTIVE，即使同文件main服务历史结果。

### 训练与数据预处理（11）

| 文件 | main guard行号 |
|---|---|
| `scripts/run_crc_stereocite.py` | 1543 |
| `scripts/run_human_embryo_rna_only.py` | 444 |
| `scripts/run_human_lymph_node.py` | 185 |
| `scripts/run_misar_seq.py` | 786 |
| `scripts/run_mouse_spleen.py` | 78 |
| `scripts/run_mouse_thymus.py` | 233 |
| `scripts/run_mousebrain_v2.py` | 828 |
| `scripts/run_preprocessing.py` | 148 |
| `scripts/run_simulation.py` | 186 |
| `scripts/run_spatch.py` | 834 |
| `scripts/train_stage_model.py` | 395 |

### 实验suite与daemon（17）

| 文件 | main guard行号 |
|---|---|
| `scripts/run_result_v15a_suite.py` | 261 |
| `scripts/run_result_v15f_suite.py` | 133 |
| `scripts/run_result_v4_large_suite.py` | 245 |
| `scripts/run_result_v4_suite.py` | 241 |
| `scripts/run_result_v4b_suite.py` | 200 |
| `scripts/run_result_v4c_large_suite.py` | 155 |
| `scripts/run_result_v4c_suite.py` | 109 |
| `scripts/run_result_v5_delayed_ot_suite.py` | 304 |
| `scripts/run_result_v5_missing_v3_matched_suite.py` | 490 |
| `scripts/run_result_v5_spatch_daemon.py` | 302 |
| `scripts/run_result_v6_dual_graphsage_suite.py` | 323 |
| `scripts/run_result_v6_missing_v3_matched_suite.py` | 480 |
| `scripts/run_result_v6_spatch_daemon.py` | 318 |
| `scripts/run_result_v7a_suite.py` | 350 |
| `scripts/run_result_v7b_suite.py` | 100 |
| `scripts/run_result_v7c_suite.py` | 100 |
| `scripts/run_spatch_v7abc_suite.py` | 471 |

### 小型验证与benchmark（7）

| 文件 | main guard行号 |
|---|---|
| `scripts/benchmark_microenvironment_fullspot.py` | 227 |
| `scripts/run_stage_model.py` | 382 |
| `scripts/validate_bidirectional_ot_attention.py` | 171 |
| `scripts/validate_checkpoint_ot_attention.py` | 220 |
| `scripts/validate_experiment_c.py` | 180 |
| `scripts/validate_microenvironment_context.py` | 560 |
| `scripts/validate_single_modality_mode.py` | 86 |

### clustering/evaluation入口（18）

| 文件 | main guard行号 |
|---|---|
| `scripts/analyze_crc_stereocite_clustering.py` | 634 |
| `scripts/analyze_human_embryo_reference_metrics.py` | 1257 |
| `scripts/analyze_human_lymph_node.py` | 52 |
| `scripts/analyze_misar_result_v5_standardized.py` | 50 |
| `scripts/analyze_misar_seq_clustering.py` | 711 |
| `scripts/analyze_mouse_spleen.py` | 41 |
| `scripts/analyze_mouse_thymus.py` | 75 |
| `scripts/analyze_mousebrain_clustering.py` | 452 |
| `scripts/analyze_mousebrain_v3_standardized.py` | 211 |
| `scripts/analyze_mousebrain_v4_standardized.py` | 212 |
| `scripts/analyze_result_v3_large_standardized.py` | 342 |
| `scripts/analyze_result_v3_standardized.py` | 431 |
| `scripts/analyze_result_v7a_requested_metrics.py` | 490 |
| `scripts/analyze_simulation.py` | 147 |
| `scripts/analyze_spatch.py` | 297 |
| `scripts/analyze_spatch_result_v5_standardized.py` | 54 |
| `scripts/analyze_spatch_result_v6_standardized.py` | 58 |
| `scripts/analyze_spatch_v7_requested_metrics.py` | 360 |

### 预处理/聚类协议比较（9）

| 文件 | main guard行号 |
|---|---|
| `scripts/compare_crc_stereocite_kmeans_preprocessing.py` | 1951 |
| `scripts/compare_human_lymph_node_kmeans_preprocessing.py` | 1442 |
| `scripts/compare_misar_seq_kmeans_preprocessing.py` | 1785 |
| `scripts/compare_mouse_spleen_kmeans_preprocessing.py` | 1543 |
| `scripts/compare_mouse_thymus_kmeans_preprocessing.py` | 1776 |
| `scripts/compare_mousebrain_kmeans_preprocessing.py` | 1772 |
| `scripts/compare_result_v3_v4c_by_k.py` | 376 |
| `scripts/compare_simulation_kmeans_preprocessing.py` | 1494 |
| `scripts/compare_spatch_kmeans_preprocessing.py` | 1465 |

### 指标与图件补充（9）

| 文件 | main guard行号 |
|---|---|
| `scripts/complete_human_embryo_per_section_plots.py` | 308 |
| `scripts/complete_spa_mo_analysis.py` | 561 |
| `scripts/compute_result_v7A_supplementary_joint_per_section_ari_nmi.py` | 510 |
| `scripts/generate_comparison_method_umaps.py` | 853 |
| `scripts/generate_result_v6_spamosaic_umap.py` | 1646 |
| `scripts/plot_and_prune_result_v7a_clusters.py` | 251 |
| `scripts/postprocess_result_variant.py` | 45 |
| `scripts/rerender_result_v6_umap_figures.py` | 187 |
| `scripts/supplement_human_lymph_node_a1_supervised_metrics.py` | 613 |

### 下游插补与PCC（10）

| 文件 | main guard行号 |
|---|---|
| `gene_imputation/extract_imputed_genes.py` | 77 |
| `gene_imputation/plot_imputation_results.py` | 287 |
| `gene_imputation/run_spatch_knn_imputation.py` | 894 |
| `gene_imputation_spatial_smoothing/extract_imputed_genes.py` | 8：顶层runpy代理到baseline exporter，无main guard |
| `gene_imputation_spatial_smoothing/plot_smoothed_imputation_results.py` | 280 |
| `gene_imputation_spatial_smoothing/run_spatch_smoothed_knn_imputation.py` | 647 |
| `gene_imputation_shared_gene_validation/compute_scheme1_pcc.py` | 539 |
| `gene_imputation_shared_gene_validation/compute_scheme2_pcc.py` | 556 |
| `gene_imputation_shared_gene_validation/plot_shared_gene_validation.py` | 533 |
| `gene_imputation_shared_gene_validation/run_shared_gene_validation.py` | 702 |

### 已有unittest模块（6）

| 文件 | main guard行号 |
|---|---|
| `gene_imputation/tests/test_equivalence.py` | 46 |
| `gene_imputation_spatial_smoothing/tests/test_smoothing.py` | 50 |
| `gene_imputation_shared_gene_validation/tests/test_metrics.py` | 36 |
| `gene_imputation_shared_gene_validation/tests/test_pcc_plots.py` | 48 |
| `gene_imputation_shared_gene_validation/tests/test_scheme1_pcc.py` | 67 |
| `gene_imputation_shared_gene_validation/tests/test_scheme2_pcc.py` | 73 |

## 附录 B：主要脚本重复的具体边界

| 重复/兼容形式 | 真实依赖及迁移边界 |
|---|---|
| CRC↔MouseBrain prior orchestration | `sparse_prior_kwargs/initialize_model_ot_prior/update_model_ot_prior`分别位于CRC689/712/732和MouseBrain177/196/208；统一resolved args和source后只保留一组 |
| 三个trainer | CRC774、MouseBrain512内部循环、generic262；普通函数fit足够，不需要trainer基类 |
| 源码replace+exec | Spleen22/HumanLymph22读CRC；Simulation72/Thymus119读MISAR；先把数据准备/section/模态显式传给同一pipeline，再删源码适配壳 |
| 历史suite monkeypatch | v7B/C修改v7A globals；v15A修改v7A/SPATCH globals，v15F再修改v15A与v7A；改为显式dataset列表、preset、output root；保留命令来源记录 |
| 评估历史名字复用 | requested analyzer33-40导入v3/v4 loader及compare模块；先收敛loader/metric纯函数再退休旧main |
| 两套top-k extraction | 单向matrix mask与双向edge排序组织不同；可以共享底层edge质量与提取，但需保持空行、padding、tie和输出schema |
| 不应合并的转换 | Stage输入FP32转换保留autograd，OT helper会detach；只因函数同名而合并可能截断encoder梯度 |
| 插补与平滑 | `W @ X`与`W @ ((1-alpha)X + alpha*S@X)`是同迁移加可选平滑；PCC raw/normalized是明确指标协议，不是多个模型decoder。可参数化，先保留已有19个数学测试 |

## 附录 C：第二阶段用户决策与必要勘误（2026-09-12，追加记录）

本节仅追加，不改变上文第一阶段对当时工作树的审计记录。第二阶段方案见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)。本轮没有实施生产重构、测试修改、配置修改、训练或结果清理。

### C.1 已明确的范围决定

- 用户暂定 **v15A 为当前主版本及未来默认研究路径**；v15B–F 不因此成为默认。上文“没有唯一官方默认”“v15A为实验suite”是第一阶段的状态判断；第二阶段已明确未来目标，尚未修改默认代码。
- 只保留**双向 candidate-sparse UOT**生产路径。dense、独立单向及其专属参数/兼容层/验证获准退休，先迁移仍依赖这些默认的入口，再删除分支。B7按路径退休处理，不实现完整dense双向。
- 双向内部A→B/B→A检索、统一索引、同coupling行/列提取，以及FAISS IVF/FlatIP/blockwise检索后端仍保留。上文“共享单向/双向solver、保留dense视需求”的方案已被本轮明确退休范围替代。
- v7A可运行基线、其他历史suite和v15B–F、G0、context gate、fused/final source、metacell、单模态、H&E/UNI、数据集适配、插补、weights export没有获准退休；先保留实际功能，不新建永久兼容框架。历史数据、结果、图件、指标和论文产物不是代码删除对象。

### C.2 v15A身份核实与时序补充

本轮新增只读证据：c4380fccd71c388409c126ab7c885eb2710410cc固定源码、parent 922d173差异、result_v15A的suite/summary/输入配置及必要小型metadata。六个保存训练argv与固定提交suite的纯命令构造逐token一致。由constructor/forward/trainer确认可冻结的源码合同：唯一Stage、pre spatial+feature、post严格spatial-only且suite传scale=.5、feature来自section内fused.detach、K_F=(K_S+1)//2、FlatIP自排除、feature原权1/K_S、组合图重复贡献叠加再行归一化。

**当前工作树不是该历史v15A时序，不能直接改名：**

| 时点 | c4380fc源码合同 | 第一阶段记录的当前工作树 |
|---|---|---|
| MouseBrain首次构图 | epoch0 dry forward仍为train mode，仅no_grad；fused之后/pre-GS之前lazy构图并消耗dropout RNG | epoch0不构feature，epoch1训练亦纯spatial；step后eval才构图 |
| 其余五个v15A任务首次构图 | epoch1训练forward内部lazy构图，使用各dataset既有autocast设置 | epoch1训练后eval构图 |
| epoch1结束 | 无额外feature/OT刷新 | 一遍eval换feature，不换OT |
| epoch100起周期刷新 | 一遍eval用旧G/P；prepare用这遍fused换G，新P仍从这遍旧G/P产生的ot输出求得 | pass1用旧G/P→换G→pass2用新G/旧P→换P |
| epoch200后final | 刷新仍执行，再最终eval使用新G/P；无201训练、无最终额外OT求解 | 也刷新再导出，但新P的来源因两遍forward而不同 |

两者刷新集合都为100、120、140、160、180、200；首次不是20，不是该处off-by-one。OT source=ot，topology context来自同源spatial self-excluded pooling、weight=.2；attention context gate=False。当前未提交时序与v15F明确metadata相符，但不据版本号反推B–E算法。

**证据边界仍在：**历史结果未保存Git/dirty源码指纹、完整resolved config和首次图/RNG状态；MouseBrain输入JSON副本仍写epochs5/cpu，不能当实际200/cuda的resolved记录。已核实可冻结源码基准，不等于证明既有result_v15A数值由该干净commit精确产生，也没有宣称已恢复模型。缺失历史字段没有用当前默认补写。

### C.3 bug与验证方案勘误

- 上文§11把 **B4 masking**放入“科学修复/实验”批次不恰当。本轮明确它是既有attention support-mask缺陷，必须独立修复；已复现padding反例输出会改变，不触发缺陷的基准保持。它不同于R1/R3需研究选择的数学变化。空行修后应zero directional update，之后仍执行既有residual/norm；全support为空的solver现有ValueError合同不顺带改变。
- B5的epoch1 unconditional触发属于当前工作树。固定c4380fc的prepare调用有feature开关保护、无epoch1额外刷新；恢复历史时序会去掉该日程触发，但直接refresh API与保留F关闭路径仍需检查/修复。不能把工作树问题泛称历史v15A默认必触发。
- R1暂缓，不加softmax、不换loss、不用数值补丁掩盖假设；R3暂缓，solver退休/整理不改floor、u/v、迭代或log-domain。已知NaN反例不能忽略后记为PASS。
- R2明确区分weights export、恢复推理状态、精确resume。保留现有导出；测试fixture保存完整状态不代表新增生产恢复能力。上文建议的restore/resume属于另立范围，纯重构不承诺旧weights可恢复最终embedding。
- 测试新增只读核查：validate_microenvironment_context.py:299的同步平均结果应先对ot_embeddings，不能直接对post之后final_embeddings；其513维gate断言需显式开启context gate。该测试make_config:61已显式source=fused，**fused断言本身并未过期**。validate_bidirectional_ot_attention.py:93的graph.k_neighbors是无效键；真实键为knn_neighbors_spatial。
- validate_checkpoint_ot_attention.py:69–78的zip(named_parameters())未比较名称/schema，max聚合可能掩盖NaN；不能作为删除模块后有效梯度映射的充分证据。后续按名称/shape/None/finite逐项检查；第一阶段同schema已执行记录保留，不扩张其证明范围。默认test discovery的sibling import及smoke旧字符串问题单独修。
- 评估协议以实际调用者为准：SPATCH requested analyzer的K_VALUES为5/8/10/12/14/16/20；被复用compare脚本的ALL_KS=2…20、RETAIN_KS=5/8/10/12/16/20属于另一独立比较入口，不能让共享函数迁移把两者混为同一协议。joint全体/independent分section的StandardScaler范围与raw输入batch metrics分别冻结。

### C.4 新的实施边界

以REFACTOR_PLAN.md替代上文实施建议作为下一阶段候选顺序：**P0先冻结已核实源码与可重放小基准；P1单独恢复目标v15A行为；随后独立bug修复、迁调用者再退dense/单向、再纯结构收敛。**恢复行为相对当前工作树会改变数学/随机时序，不能冒充纯等价重构。

仅对冻结v15A非bug基准及明确修正后的bug预期做逐项承诺，不保留获准退休的dense/单向数学。配置一次解析采用defaults < preset < dataset config < 显式CLI，并保存完整resolved记录；不统一dataset超参数。P3清理无效字段不能提前删除仍供dense使用的真实参数，须等调用者迁完再退休。

本轮未创建正式测试、未重新跑第一阶段测试、未运行真实数据训练。只新增计划并追加本节；第三阶段尚未开始。

## 附录 D：已回退 v7A；第二阶段范围再次修订（2026-09-15，追加记录）

本节追加在原记录之后，不改写第一阶段或附录C的历史证据。用户已取消以v15A为目标的方案，当前生效计划为 [REFACTOR_PLAN.md](REFACTOR_PLAN.md) 的v7A修订版。附录C中的v15A恢复、保留后续feature lifecycle、B4必修实施及插补测试修改安排均不再作为当前授权。

### D.1 当前Git和真实基线

本轮检查HEAD为 **922d1738922e8b94890a54f5e42bbf6551f5ccc0**；tracked工作树/staged diff均为空，仅两份重构文档未tracked。没有reset/checkout/clean/stash/提交，没有修改代码使它符合旧报告。

当前主基线由实际调用确认：run_result_v7a_suite.py:104的五数据集命令及run_spatch_v7abc_suite.py:107的v7A分支 → dataset runners → 唯一StageMultiModalModel。裸库/CLI默认尚不统一：post scale默认仍1，MouseBrain/CRC及generic仍有dense默认；suite显式sparse+bidirectional、post=.5才是本轮冻结对象。现有v7A JSON只覆盖scale，不是完整resolved config。

| 核心项 | 回退后源码核实结果 |
|---|---|
| pre/post图 | Stage:1263–1272、1407–1422均为section内spatial-only；pre/post两个独立参数实例 |
| residual | pre固定1；v7A命令post=.5，经constructor:352–354传至GraphSAGE:249 |
| 空间图 | utils:219–306；KNN k+1、距离权重、自环原权1、加reverse、重复空间边max合并、接收行weight/(sum+delta)归一化；无跨section边 |
| feature实现 | 当前无builder/cache/refresh/pre组合图，只有configure:115无消费者的use_feature_graph=False残留；旧fused/K_F/feature边和两遍时序不再适用，不重引入 |
| 初始化 | runner显式初始化双向candidate-sparse prior；同coupling行/列提两个方向top-k |
| attention | 所有方向读同一pre-OT状态，各section方向update平均后一次apply_update；后续独立post GraphSAGE |
| refresh source/context | 当前命令source=ot；topology=true/weight=.2，self-excluded spatial context来自同源ot；attention context gate=False |
| refresh时序 | should_update_ot:119–157从100起，interval20，optimizer后一次eval/no_grad，用旧prior产ot/context再替换新prior；无epoch1特例 |
| 初次forward差异 | MouseBrain:576–584仍有默认train模式no_grad epoch0 dry forward，消耗dropout RNG；CRC训练链无此前置步，不可在fit收敛时无声省掉 |
| 最终导出 | epoch200命中刷新，之后final eval消费新prior；不再从final embedding额外求OT。非刷新末epoch保留最近prior。SPATCH现有导出无prior文件，不能声称所有入口都已支持 |
| 模型数值 | loss、solver floor/u-v/stabilizer、confidence/top-k、padding、AMP/chunk/checkpoint等当前语义均作为纯重构合同；不以shape/ARI近似代替 |

本轮只读取必要小型suite/config/cache metadata、源码和Git；未运行真实训练/评估/正式测试。result_v7A/suite_status.json的五任务记录支持命令及scale/source/刷新，但无源码与完整状态指纹，不能把历史数值精确复现当作已证明事实。

### D.2 旧问题状态重分类

“仍存在”指本轮静态确认触发机制，不冒称重新执行了旧动态验证；默认主路径是否触发需区分。

| 状态 | 问题及本轮勘误 |
|---|---|
| 仍存在 | B1/B2配置覆盖；B3 source/context不一致；B4补位进入attention；B7 dense假双向；B8无效配置；B9分析cache仅按存在复用；B10提前I/O/旧输出路径；B11图cache缺输入身份；B12 BF16转numpy；B13 shared漏target_sum；B14 section ID不一致 |
| 已消失 | B5关闭feature仍构图、B6 feature强制FAISS GPU：实现已不存在；B15：requested SPECS恢复五数据集，与v7A suite一致 |
| 机制仍在、影响未重新验证 | R1 signed joint假设、R3 kernel floor；本轮不验证真实训练NaN/cost分布，不修改数学 |
| R2仍有能力缺口，证据需改适用范围 | weights仍未含OT prior/optimizer/RNG；当前没有feature cache，上文0.2277677 feature-cache对照不适用于当前v7A。weights export、推理恢复、精确resume继续分开，后两者不扩入本计划 |
| 非保护区测试机制仍在，未重跑 | T1 smoke错误文本、T3 gate/stage断言、T4参数zip/finite checker与bidirectional验证无效k_neighbors键；将来单列测试批，不为过测试改模型 |
| 保护区问题仅记录 | shared-gene scheme2 sibling import机制仍在；三个插补目录整体不可改，撤销旧计划对它的测试/代码修复 |
| 未重新验证 | AMP内部dtype、decoder target量化、GPU/tie/线程确定性、跨chunk dropout、memory_efficient预处理、带点output目录、None模态bundle、真实数据逐spot对齐等旧疑点 |

B4仍是有证据的correctness缺陷，但用户本轮明确未经批准不改attention masking等数值语义。当前只记录触发条件/修复方案，独立待批准；不沿用附录C“必须实施”安排。B7继续随dense退休，不实现dense双向；R1/R3不加softmax、不换loss/稳定算法，不在solver清理中改公式。

### D.3 新保护边界与缓存证据

三个gene_imputation目录整体原位置原内容保留，包括代码/配置/tests/说明/产物。只读检查未发现对model/scripts的直接import；它们互相通过sys.path/importlib/runpy调用，外部合同是固定result_v6的float32 final_embeddings_section1/2.npy、run_summary alignment.n_spots、spot_metadata.csv.gz的section/x/y行序及raw h5ad路径。无需为它们建立历史模型框架；必须保护原格式/路径与顺序。

preprocessed_cache/整体不可删除/移动/改名/改格式/重写/自动生成。SPATCH实际cache为preprocessed_cache/spatch/v6_harmony_n50_hvg3000，6998字节manifest SHA256：
4fdf8ed1a3c4beaa856eff4e91e3970639543718afce45a5527cfb159754fba7。
它是schema1、PCA/Harmony后的模型输入；RNA/HE50维、Protein15维、两section 665399/403563行。只读manifest，不加载数组或大型CSV。

新增迁移风险C1：run_spatch.load_preprocessed_cache:359–369比较整个run_spatch.py和model/data_preprocessing.py源码hash；当前hash匹配，纯重构后会拒绝原缓存。新loader须沿用旧schema，将源码/raw来源记录作为provenance，明确解除整文件hash和raw path/size/mtime严格相等的加载门槛，保留缓存内容完整性及加载边界对齐检查。接受条件变化单列，不能声称只有移动代码；绝不重做normalize/PCA/Harmony或改缓存manifest。

manifest没有逐feature数组绑定的ordered spot IDs；spatial与metadata有独立hash及原构建alignment声明。这个静态限制不能靠猜测补齐，也不能成为重算理由。后续加载按原行序核对section、各模态N/维度、metadata坐标；矛盾则报告并停止该加载，不触碰缓存。本轮没有核数组内容/raw stat，不能宣称端到端缓存验证通过。

results/与所有现有result_*原位置原内容保留；不启动可能overwrite/prune/重算指标的历史脚本。未来小验证仅写新独立refactor_checks/等目录，**本阶段没有创建该目录或测试产物**。B9新分析结果关联与SPATCH预处理缓存严格分开，旧结果缺provenance也不自动重算。

### D.4 生效计划顺序

先P0冻结**当前v7A**与同有效参数/状态小基准；没有恢复v15A批次。先完成缓存加载边界迁移，再迁保留调用者并实际删除dense/单向；配置bug修复与纯解析提取分开；消除源码replace后再迁受其影响的dataset默认，收敛一个fit；随后整理analysis实际依赖并退休无用启动壳，最后做必要命名/目录调整。

核心保留model/与一个Stage；data_io/training/analysis按真实职责提函数，不先大搬家。分析不能为读数据导入训练入口；CRC/MISAR/Thymus/Simulation中仍被UMAP等读取的data helpers先迁出。旧结果目录不要求永久保留同名训练接口；当前不存在的后续实现不重新引入。G0/context/source/metacell等尚有真实用途但非主路径的功能集中待确认，不自行扩大支持或构建版本框架。

P0记录当前已知缺陷的原始行为；例如配置覆盖、parse前I/O必须用mock观察，不能要求基准先达到修复后预期而顺手改生产。纯重构、批准的模式退休、缓存加载接受条件变化、bug fix、科学方法修改分别review。本轮到两份文档交付即停止，未进入实施。
