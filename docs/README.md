# 项目文档总导航

归档日期：2026-09-17。本目录集中保存实验记录和重构文档；两批实验记录按原来源分开，不将它们合并成一份结论。

## 当前阅读入口

- [根 README](../README.md)：当前项目结构、正式运行命令和使用边界。
- [scripts/README.md](../scripts/README.md)：正式 CLI 与输入格式。
- [handoff.md](../handoff.md)：工作交接入口，保留在根目录。
- [重构计划](refactoring/REFACTOR_PLAN.md)、[用户决策](refactoring/USER_DECISIONS.md)：过程合同和后续取舍，按各节记录的阶段阅读。
- [tools/validation/README.md](../tools/validation/README.md)：当前验证工具入口。

## 分组与阅读约定

| 分组 | 数量 | 来源及用途 |
|---|---:|---|
| [experiments_early/](experiments_early/) | 22 | 原 docs 中的早期模型验证、方法阅读、数据适配和初期实验记录，含 1 份 PDF |
| [experiments_later/](experiments_later/) | 15 | 原根目录中的后续比较、损失与结构实验、优化建议和实施方案 |
| [refactoring/](refactoring/) | 3 | 原根目录中的重构审计、计划和用户决策 |

上述分组保留原来两批实验文档的来源，不声称每篇文档都具有严格连续的日期。旧模型版本、命令、源码位置、数值和结论均按当时记录保留；它们不自动构成当前实现说明，也不表示其中的建议已实施。以“report”或“plan”命名的记录，仍须结合其自身状态阅读。

历史实验文档（37 份）和重构审计只移动，内容不改写。重构计划及用户决策仅调整必要的导航链接，用户决策另追加本次归档实施记录。`README.md` 与 `handoff.md` 保留在根目录并更新文档导航。没有创建旧路径转发文件或符号链接。

原 `refactor_checks/` 的历史 REPORT、fixture、expected、源码快照和已知失败证据保持原位置、原内容。历史证据中记录的旧文档路径也是当时事实，查找当前位置时使用下面的映射；不改写冻结证据以适配新目录。

## 文件清单：原位置 → 当前位置

下面的“原位置”均相对于仓库根目录；每份文档保留原文件名。

### 第一批：原 docs 记录

| 原位置 | 当前位置 |
|---|---|
| `docs/COSIE 中切片内跨模态对比学习方法总结.pdf` | [experiments_early/COSIE 中切片内跨模态对比学习方法总结.pdf](experiments_early/COSIE%20%E4%B8%AD%E5%88%87%E7%89%87%E5%86%85%E8%B7%A8%E6%A8%A1%E6%80%81%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93.pdf) |
| `docs/cosie_contrastive_learning_report.md` | [experiments_early/cosie_contrastive_learning_report.md](experiments_early/cosie_contrastive_learning_report.md) |
| `docs/cosie_preprocessing_report.md` | [experiments_early/cosie_preprocessing_report.md](experiments_early/cosie_preprocessing_report.md) |
| `docs/crc_fullspot_2epoch_gpu_stress_report.md` | [experiments_early/crc_fullspot_2epoch_gpu_stress_report.md](experiments_early/crc_fullspot_2epoch_gpu_stress_report.md) |
| `docs/crc_make_unique_integrated_pipeline_report.md` | [experiments_early/crc_make_unique_integrated_pipeline_report.md](experiments_early/crc_make_unique_integrated_pipeline_report.md) |
| `docs/crc_random1k_hvg_train_smoke_report.md` | [experiments_early/crc_random1k_hvg_train_smoke_report.md](experiments_early/crc_random1k_hvg_train_smoke_report.md) |
| `docs/crc_var_names_audit_and_fix_strategy_report.md` | [experiments_early/crc_var_names_audit_and_fix_strategy_report.md](experiments_early/crc_var_names_audit_and_fix_strategy_report.md) |
| `docs/faiss_ivf_sparse_uot_implementation_report.md` | [experiments_early/faiss_ivf_sparse_uot_implementation_report.md](experiments_early/faiss_ivf_sparse_uot_implementation_report.md) |
| `docs/human_embryo_rna_only.md` | [experiments_early/human_embryo_rna_only.md](experiments_early/human_embryo_rna_only.md) |
| `docs/model_stage_v1_verification_report.md` | [experiments_early/model_stage_v1_verification_report.md](experiments_early/model_stage_v1_verification_report.md) |
| `docs/model_stage_v2_fix_report.md` | [experiments_early/model_stage_v2_fix_report.md](experiments_early/model_stage_v2_fix_report.md) |
| `docs/model_stage_v2_verification_report.md` | [experiments_early/model_stage_v2_verification_report.md](experiments_early/model_stage_v2_verification_report.md) |
| `docs/model_validation_report.md` | [experiments_early/model_validation_report.md](experiments_early/model_validation_report.md) |
| `docs/model_validation_targeted_fix_report.md` | [experiments_early/model_validation_targeted_fix_report.md](experiments_early/model_validation_targeted_fix_report.md) |
| `docs/mousebrain_clustering_analysis_report.md` | [experiments_early/mousebrain_clustering_analysis_report.md](experiments_early/mousebrain_clustering_analysis_report.md) |
| `docs/mousebrain_fullspot_warmup_feasibility_report.md` | [experiments_early/mousebrain_fullspot_warmup_feasibility_report.md](experiments_early/mousebrain_fullspot_warmup_feasibility_report.md) |
| `docs/mousebrain_lambda_contrast_sweep_report.md` | [experiments_early/mousebrain_lambda_contrast_sweep_report.md](experiments_early/mousebrain_lambda_contrast_sweep_report.md) |
| `docs/mousebrain_loss_history_check_report.md` | [experiments_early/mousebrain_loss_history_check_report.md](experiments_early/mousebrain_loss_history_check_report.md) |
| `docs/mousebrain_v2_run_report.md` | [experiments_early/mousebrain_v2_run_report.md](experiments_early/mousebrain_v2_run_report.md) |
| `docs/ot_weight_attention_graphsage_code_audit_report.md` | [experiments_early/ot_weight_attention_graphsage_code_audit_report.md](experiments_early/ot_weight_attention_graphsage_code_audit_report.md) |
| `docs/variable_two_modality_self_validation_report.md` | [experiments_early/variable_two_modality_self_validation_report.md](experiments_early/variable_two_modality_self_validation_report.md) |
| `docs/variable_two_modality_support_report.md` | [experiments_early/variable_two_modality_support_report.md](experiments_early/variable_two_modality_support_report.md) |

### 第二批：原根目录实验记录

| 原位置 | 当前位置 |
|---|---|
| `G0_to_G2b1_model_structure_experiment_evolution_report.md` | [experiments_later/G0_to_G2b1_model_structure_experiment_evolution_report.md](experiments_later/G0_to_G2b1_model_structure_experiment_evolution_report.md) |
| `O1a_to_O2c0_ot_path_experiment_evolution_report.md` | [experiments_later/O1a_to_O2c0_ot_path_experiment_evolution_report.md](experiments_later/O1a_to_O2c0_ot_path_experiment_evolution_report.md) |
| `crc_stereocite_preprocessing_comparison_report.md` | [experiments_later/crc_stereocite_preprocessing_comparison_report.md](experiments_later/crc_stereocite_preprocessing_comparison_report.md) |
| `human_lymph_node_preprocessing_comparison_report.md` | [experiments_later/human_lymph_node_preprocessing_comparison_report.md](experiments_later/human_lymph_node_preprocessing_comparison_report.md) |
| `misar_seq_preprocessing_comparison_report.md` | [experiments_later/misar_seq_preprocessing_comparison_report.md](experiments_later/misar_seq_preprocessing_comparison_report.md) |
| `mouse_spleen_preprocessing_comparison_report.md` | [experiments_later/mouse_spleen_preprocessing_comparison_report.md](experiments_later/mouse_spleen_preprocessing_comparison_report.md) |
| `mouse_thymus_preprocessing_comparison_report.md` | [experiments_later/mouse_thymus_preprocessing_comparison_report.md](experiments_later/mouse_thymus_preprocessing_comparison_report.md) |
| `mousebrain_preprocessing_comparison_report.md` | [experiments_later/mousebrain_preprocessing_comparison_report.md](experiments_later/mousebrain_preprocessing_comparison_report.md) |
| `result_v3_v5_v6_metrics_comparison.md` | [experiments_later/result_v3_v5_v6_metrics_comparison.md](experiments_later/result_v3_v5_v6_metrics_comparison.md) |
| `simulation_preprocessing_comparison_report.md` | [experiments_later/simulation_preprocessing_comparison_report.md](experiments_later/simulation_preprocessing_comparison_report.md) |
| `spa_mo_model_architecture_optimization_recommendations.md` | [experiments_later/spa_mo_model_architecture_optimization_recommendations.md](experiments_later/spa_mo_model_architecture_optimization_recommendations.md) |
| `spa_mo_model_contrastive_loss_final_experiment_report.md` | [experiments_later/spa_mo_model_contrastive_loss_final_experiment_report.md](experiments_later/spa_mo_model_contrastive_loss_final_experiment_report.md) |
| `spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md` | [experiments_later/spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md](experiments_later/spa_mo_model_mofa_cosie_spamosaic_mousebrain_crc_report.md) |
| `spa_mo_model_performance_improvement_implementation_plan.md` | [experiments_later/spa_mo_model_performance_improvement_implementation_plan.md](experiments_later/spa_mo_model_performance_improvement_implementation_plan.md) |
| `spatch_preprocessing_comparison_report.md` | [experiments_later/spatch_preprocessing_comparison_report.md](experiments_later/spatch_preprocessing_comparison_report.md) |

### 重构过程文档

| 原位置 | 当前位置 |
|---|---|
| `REFACTOR_AUDIT.md` | [refactoring/REFACTOR_AUDIT.md](refactoring/REFACTOR_AUDIT.md) |
| `REFACTOR_PLAN.md` | [refactoring/REFACTOR_PLAN.md](refactoring/REFACTOR_PLAN.md) |
| `USER_DECISIONS.md` | [refactoring/USER_DECISIONS.md](refactoring/USER_DECISIONS.md) |

## 历史报告中的相对链接

少数后续报告用相对链接指向仓库根目录下的实验结果或原 docs 文档。为保留历史报告原文，这些链接没有在报告中改写；移动后请使用下面的导览访问其原目标。报告中仍能正常解析的同组文档链接不重复列出。代码块、命令及普通文字中的旧路径继续按历史语境理解，不保证旧命令可执行。

这些链接只提供导航，不会移动、读取重算或补建实验结果。

<details>
<summary>G0_to_G2b1_model_structure_experiment_evolution_report.md</summary>

[查看历史报告](experiments_later/G0_to_G2b1_model_structure_experiment_evolution_report.md)

| 原相对链接 | 现在的目标 |
|---|---|
| `result_G0_self_path_ablation/SUMMARY.md` | `result_G0_self_path_ablation/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_G1_minimal_multiscale_ablation/SUMMARY.md` | `result_G1_minimal_multiscale_ablation/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_G2a_high_frequency_ablation/SUMMARY.md` | `result_G2a_high_frequency_ablation/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_G2b0_dual_frequency_readout/SUMMARY.md` | `result_G2b0_dual_frequency_readout/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_G2b1_competitive_router/SUMMARY.md` | `result_G2b1_competitive_router/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_G2b1_competitive_router/g2b1_vs_g2b0.csv` | `result_G2b1_competitive_router/g2b1_vs_g2b0.csv`（迁移前已不存在，未补建） |
| `result_G2b1_competitive_router/router_section_diagnostics.csv` | `result_G2b1_competitive_router/router_section_diagnostics.csv`（迁移前已不存在，未补建） |
| `result_G2b1_competitive_router/router_section_range_summary.csv` | `result_G2b1_competitive_router/router_section_range_summary.csv`（迁移前已不存在，未补建） |
| `result_G0_self_path_ablation` | `result_G0_self_path_ablation`（迁移前已不存在，未补建） |
| `result_G1_minimal_multiscale_ablation` | `result_G1_minimal_multiscale_ablation`（迁移前已不存在，未补建） |
| `result_G2a_high_frequency_ablation` | `result_G2a_high_frequency_ablation`（迁移前已不存在，未补建） |
| `result_G2b0_dual_frequency_readout` | `result_G2b0_dual_frequency_readout`（迁移前已不存在，未补建） |
| `result_G2b1_competitive_router` | `result_G2b1_competitive_router`（迁移前已不存在，未补建） |

</details>

<details>
<summary>O1a_to_O2c0_ot_path_experiment_evolution_report.md</summary>

[查看历史报告](experiments_later/O1a_to_O2c0_ot_path_experiment_evolution_report.md)

| 原相对链接 | 现在的目标 |
|---|---|
| `result_O1a_reliable_ot_bypass/SUMMARY.md` | `result_O1a_reliable_ot_bypass/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O1a_reliable_ot_bypass/paired_deltas.csv` | `result_O1a_reliable_ot_bypass/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O1b_relative_tail_ot_bypass/SUMMARY.md` | `result_O1b_relative_tail_ot_bypass/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O1b_relative_tail_ot_bypass/paired_deltas.csv` | `result_O1b_relative_tail_ot_bypass/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O2a_global_local_ot_residual/SUMMARY.md` | `result_O2a_global_local_ot_residual/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O2a_global_local_ot_residual/paired_deltas.csv` | `result_O2a_global_local_ot_residual/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O2b0_preOT_prior_refresh/SUMMARY.md` | `result_O2b0_preOT_prior_refresh/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O2b0_preOT_prior_refresh/paired_deltas.csv` | `result_O2b0_preOT_prior_refresh/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O2b1_ema_preOT_prior_refresh/SUMMARY.md` | `result_O2b1_ema_preOT_prior_refresh/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O2b1_ema_preOT_prior_refresh/paired_deltas.csv` | `result_O2b1_ema_preOT_prior_refresh/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O2c0_topology_aware_ot_cost/SUMMARY.md` | `result_O2c0_topology_aware_ot_cost/SUMMARY.md`（迁移前已不存在，未补建） |
| `result_O2c0_topology_aware_ot_cost/paired_deltas.csv` | `result_O2c0_topology_aware_ot_cost/paired_deltas.csv`（迁移前已不存在，未补建） |
| `result_O1a_reliable_ot_bypass/integrity_audit.csv` | `result_O1a_reliable_ot_bypass/integrity_audit.csv`（迁移前已不存在，未补建） |
| `result_O1b_relative_tail_ot_bypass/integrity_audit.csv` | `result_O1b_relative_tail_ot_bypass/integrity_audit.csv`（迁移前已不存在，未补建） |
| `result_O2a_global_local_ot_residual/integrity_audit.csv` | `result_O2a_global_local_ot_residual/integrity_audit.csv`（迁移前已不存在，未补建） |
| `result_O2b0_preOT_prior_refresh/integrity_audit.csv` | `result_O2b0_preOT_prior_refresh/integrity_audit.csv`（迁移前已不存在，未补建） |
| `result_O2b1_ema_preOT_prior_refresh/integrity_audit.csv` | `result_O2b1_ema_preOT_prior_refresh/integrity_audit.csv`（迁移前已不存在，未补建） |
| `result_O2c0_topology_aware_ot_cost/integrity_audit.csv` | `result_O2c0_topology_aware_ot_cost/integrity_audit.csv`（迁移前已不存在，未补建） |

</details>

<details>
<summary>spa_mo_model_contrastive_loss_final_experiment_report.md</summary>

[查看历史报告](experiments_later/spa_mo_model_contrastive_loss_final_experiment_report.md)

| 原相对链接 | 现在的目标 |
|---|---|
| `docs/cosie_contrastive_learning_report.md` | [docs/experiments_early/cosie_contrastive_learning_report.md](../docs/experiments_early/cosie_contrastive_learning_report.md) |
| `result_v2_Contrastive_loss/crc_stereocite/` | [result_v2_Contrastive_loss/crc_stereocite](../result_v2_Contrastive_loss/crc_stereocite) |
| `result_v2_Contrastive_loss/human_lymph_node/` | [result_v2_Contrastive_loss/human_lymph_node](../result_v2_Contrastive_loss/human_lymph_node) |
| `result_v2_Contrastive_loss/misar_seq/` | [result_v2_Contrastive_loss/misar_seq](../result_v2_Contrastive_loss/misar_seq) |
| `result_v2_Contrastive_loss/mouse_spleen/` | [result_v2_Contrastive_loss/mouse_spleen](../result_v2_Contrastive_loss/mouse_spleen) |
| `result_v2_Contrastive_loss/mouse_thymus/` | [result_v2_Contrastive_loss/mouse_thymus](../result_v2_Contrastive_loss/mouse_thymus) |
| `result_v2_Contrastive_loss/mousebrain/` | [result_v2_Contrastive_loss/mousebrain](../result_v2_Contrastive_loss/mousebrain) |

</details>

## 本次归档核对

本次只移动 40 份文档并补充导航，不修改生产代码，不整理 data_io，不运行训练、预处理或完整分析，不修改历史 result/cache。保留累计未提交工作树，不 stage、不 commit。

移动清单和原文件 SHA256 见 [BEFORE.json](../refactor_checks/docs_archive_20260917/BEFORE.json)，必要导航调整见 [LINK_CHANGES.json](../refactor_checks/docs_archive_20260917/LINK_CHANGES.json)，最终核对见 [REPORT.md](../refactor_checks/docs_archive_20260917/REPORT.md) 和 [VERIFICATION.json](../refactor_checks/docs_archive_20260917/VERIFICATION.json)。这些是本次新建的归档记录，不覆盖既有验证批次。
