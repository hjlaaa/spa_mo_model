# spa_mo_model

空间多模态表示学习，当前生产模型为 **v7A 纯空间图基线**。支持九数据集与 generic 输入；训练、分析和跨方法展示有各自明确入口。

## 模型与数据流程

```text
多模态 Encoder → Fusion → pre-OT 空间 GraphSAGE
→ 双向 OT-guided Attention → post-OT 空间 GraphSAGE
→ final embedding
                    └→ Decoder → 各模态重构
```

两处 GraphSAGE 只用 section 内空间图，参数独立；保留邻接自环和外残差，没有额外 learned self_linear 分支。分析消费 final embedding，不把 Decoder 重构当作 embedding。启用 OT 时使用双向 candidate-sparse UOT，共享 coupling 后提取行/列 top-k；embedding 候选检索不是特征图。

动态 OT refresh 使用 detached OT embedding 与同源、自排除空间 context，cost 为 0.8 semantic + 0.2 context。默认首次刷新 epoch=100、interval=20；optimizer step 后以 eval/no_grad 单次前向取得刷新输入。Finit 初始化的 RNG 兼容修复保留。

v7A 研究命令使用 post-OT scale=0.5；**裸模型及当前训练入口的缺省通常仍为1.0**，不会因版本标签自动设为0.5。下面适用的训练示例显式传入0.5；generic通过模型JSON指定。各入口 epochs、precision 和模式不同，见[配置与运行合同](docs/CURRENT_ARCHITECTURE.md)。

```text
显式输入 spec / adapter → PreparedDataset
→ 已 resolved 配置 + 公共 training task
→ Stage / prior / 特有前置行为 → shared fit → 公共 artifact writer
```

统一合同不统一预处理算法、文件布局或科研协议。详细字段与所有输入模式见[当前架构和能力边界](docs/CURRENT_ARCHITECTURE.md)。

## 九数据集与入口

训练 ID 用于 batch；analysis ID 用于 evaluate/comparison。

| 数据集 | 训练 ID | 单数据集 CLI | Analysis ID |
|---|---|---|---|
| MouseBrain | mousebrain | [run_mousebrain.py](scripts/run_mousebrain.py) | mousebrain |
| Human Embryo | human_embryo | [run_human_embryo_rna_only.py](scripts/run_human_embryo_rna_only.py) | human_embryo |
| MISAR-seq | misar | [run_misar_seq.py](scripts/run_misar_seq.py) | misar_seq |
| SPATCH | spatch | [run_spatch.py](scripts/run_spatch.py) | spatch |
| CRC Stereo-CITE-seq | crc | [run_crc_stereocite.py](scripts/run_crc_stereocite.py) | crc_stereocite |
| Human Lymph Node | human_lymph_node | [run_human_lymph_node.py](scripts/run_human_lymph_node.py) | human_lymph_node |
| Mouse Spleen | mouse_spleen | [run_mouse_spleen.py](scripts/run_mouse_spleen.py) | mouse_spleen |
| Mouse Thymus | mouse_thymus | [run_mouse_thymus.py](scripts/run_mouse_thymus.py) | mouse_thymus |
| Simulation | simulation | [run_simulation.py](scripts/run_simulation.py) | simulation |

## 运行方式

从仓库根目录，在已具备依赖的 Python 环境运行；本项目验证使用现有 `cosie` 环境。本文不提供未经验证的安装流程。嵌套 comparison/workflow 使用 `python -m`。完整18个 CLI及其输入JSON格式见 [scripts README](scripts/README.md)。

下列是命令模板，`/path/to/...` 须替换为真实输入或**全新的、不存在的输出目录**。不要把 output 指向历史 result/cache，也不要加入 overwrite 选项。解析成功不表示真实数据、硬件和依赖已验证。部分训练默认会落到已有 results 路径，因此示例总是显式给出 output。

### 单数据集与 generic 训练

```bash
python scripts/run_mousebrain.py --config data/configs/mousebrain_preprocess_train.json --output_dir /path/to/new_mousebrain_run --epochs 5 --post_ot_graphsage_scale 0.5
python scripts/run_crc_stereocite.py --data_dir /path/to/CRC_Stereo-CITE-seq --output_dir /path/to/new_crc_run --train --epochs 200 --device cuda --save_outputs --save_embeddings --post_ot_graphsage_scale 0.5
python scripts/run_human_embryo_rna_only.py --train --data_dir /path/to/hesta_sources --reuse_preprocessed --preprocessed_run_dir /path/to/existing_embryo_run --output_dir /path/to/new_embryo_run --epochs 100 --post_ot_graphsage_scale 0.5
python scripts/train_stage_model.py --input_bundle /path/to/model_ready.pt --model_config /path/to/model_config.json --output_dir /path/to/new_generic_run --epochs 5 --save_embeddings
python scripts/train_stage_model.py --help
python scripts/run_preprocessing.py --help
```

MouseBrain样例JSON含本机绝对输入路径，使用前准备自己的配置副本；正式MouseBrain adapter读取RNA中已有的UNI特征。generic还支持 `--preprocess_config` 和显式 `--smoke_test`，各自schema与模式语义见[输入合同](docs/CURRENT_ARCHITECTURE.md)。generic周期/最终weights不是精确resume checkpoint。

Embryo默认是输入审计；`--preprocess_only`、`--dry_run`、`--train` 显式选择后续动作。raw HESTA与manifest/external reuse均保留，HESTА算法不自动加Harmony。`--require_harmony`仅校验预处理记录；只在已确认来源具备Harmony时使用。当前入口在准备前仍执行原输入审计，不能把external reuse理解成取消全部来源检查。

### SPATCH：reuse 与 raw

```bash
python scripts/run_spatch.py --input_mode reuse --preprocessed_cache_dir /path/to/existing_spatch_cache --output_dir /path/to/new_spatch_reuse_run --post_ot_graphsage_scale 0.5
python scripts/run_spatch.py --input_mode raw --data_dir /path/to/aligned_spatch_h5ad --output_dir /path/to/new_spatch_raw_run --post_ot_graphsage_scale 0.5
```

- `reuse`是默认模式：严格校验已有schema-1 model-ready cache；missing/invalid cache报错，不自动fallback raw。
- `raw`只指**两section的六个上游已对齐H5AD：RNA、Protein、已有HE/UNI feature**。逐模态预处理后写全新cache，释放中间结果，再经strict loader进入PreparedDataset和训练。
- 默认新cache位置为本次 `output_dir/preparation_cache`；可用 `--output_cache_dir` 指定另一个不存在的目录。任何cache均不可由 `--overwrite` 覆盖；旧 `--build_preprocessed_cache` 仍拒绝。
- SPATCH raw **不包含原始图片重新提取UNI**。通用image+mask→UNI能力仍在数据层，但不接入此SPATCH输入模式。SPATCH正式GPU准入保持；小型验收不代表百万spot的RAM/耗时或GPU训练已经验证。

### 批量训练

```bash
python scripts/run_experiments.py --datasets crc misar --output-root /path/to/new_batch_run --epochs 200 --post-ot-graphsage-scale 0.5 --dry-run
python scripts/run_experiments.py --datasets mousebrain spatch --output-root /path/to/new_batch_inputs --post-ot-graphsage-scale 0.5 --runner-args mousebrain '["--config","data/configs/mousebrain_preprocess_train.json"]' --runner-args spatch '["--input_mode","reuse","--preprocessed_cache_dir","/path/to/existing_spatch_cache"]' --dry-run
```

batch的 `--dry-run`只打印命令，不建输出、不启动子进程。检查输入/输出后移除它才运行；每个数据集独立进程，状态写 `batch_status.json`，默认失败停止。共享override追加在 `--runner-args` 后；batch不自动运行analysis。其他入口的dry未必免计算，不能套用batch含义。

## Analysis 与专项工作流

`evaluate`读取已有结果，在独立新目录写分析产物。普通单run聚类使用standardized embedding；batch metrics、UMAP及专项诊断继续保留各自输入空间。

```bash
python scripts/evaluate.py --help
python scripts/evaluate.py --dataset mousebrain --run-dir /path/to/mousebrain_run --data-dir /path/to/mousebrain_data --output-dir /path/to/new_requested_analysis --protocol requested --scope joint independent --joint-per-section
python scripts/evaluate.py --dataset human_embryo --run-dir /path/to/embryo_run --output-dir /path/to/new_embryo_analysis --protocol embryo-reference --scope joint independent
python scripts/evaluate.py --dataset mousebrain --run-dir /path/to/mousebrain_run --data-dir /path/to/mousebrain_data --output-dir /path/to/new_joint_section_analysis --protocol requested --scope joint_per_section --assignments-dir /path/to/existing_analysis/standardized_embedding --k 5
```

| scope | 行为 |
|---|---|
| joint | joint空间标准化，对各K生成joint labels并评价 |
| independent | 每section拟合自己的scaler，多个K复用该标准化结果，再分别聚类/评价 |
| joint_per_section | 只切已有joint labels，按section评价ARI/NMI；不重聚类、不跨section平均 |

joint_per_section范围由当前protocol明确约束：MouseBrain、MISAR、Simulation、SPATCH的现有truth协议；不因有annotation自动扩展Embryo。CRC/HLN/Spleen/Thymus普通工作流不制造生物truth，HLN外部人工标注是另一个专项工作流。Thymus requested需要已有ASW sample；SPATCH comparison需要显式历史batch metrics来源，详见scripts说明。

```bash
python -m scripts.comparison.compare_embeddings --help
python -m scripts.comparison.plot_method_umaps --help
python -m scripts.workflows.plot_input_integration --help
python -m scripts.workflows.evaluate_hln_annotations --help
python scripts/analyze_simulation.py --help
```

跨方法格式、reference cohort、saved scaler/labels由显式输入指定。input integration展示可能重新准备原模态特征，但不训练整合模型；HLN annotation的dry仍会计算。Embryo层级/发育、Simulation factor诊断、SPATCH大规模协议保持独立。B9将输入身份与实现身份/来源分别记录；源码变化不允许改写历史cache伪装命中。

## 代码与文档导航

| 边界 | 位置 |
|---|---|
| 纯空间图Stage及数学 | [model/](model/) |
| 显式adapter、PreparedDataset、mechanical analysis readers | [data_io/](data_io/) |
| 默认/解析、训练task、唯一epoch循环、通用writer | [training/](training/) |
| protocol、公共workflow、算法、专项分析 | [analysis/](analysis/) |
| 正式CLI与输入/资源接缝 | [scripts/README.md](scripts/README.md) |
| validation/checker | [tools/validation/README.md](tools/validation/README.md) |
| 当前架构、字段、配置与能力说明 | [CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) |
| U系列计划及各批退出状态 | [CORE_INTERFACE_REFACTOR_PLAN.md](docs/refactoring/CORE_INTERFACE_REFACTOR_PLAN.md) |
| 历史文档索引 | [docs/README.md](docs/README.md) |

当前命令无需选择历史版本launcher。旧版本suite和比选入口已退休；少量Python兼容API仅薄委托公共实现，不是另一套CLI。历史报告中的旧命令不是当前运行建议。

B13 target_sum、B14 section identity、B12 BF16→NumPy、B11 graph cache identity及Finit均已保留，不再列为待修复。B4/R1/R3与旧Stage smoke文案问题仍需独立处理。保护目录、历史result/cache和冻结fixture/expected不自动重写；已有数值验收有precision/chunk/RNG等边界，不能推断跨配置逐位等价或九数据集长期训练全部重跑。
