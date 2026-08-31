# 共享基因伪缺失验证

本流程使用spatch两个切片中均有实测值的4,828个共享基因，评价未平滑和
参考数据平滑两种插补方案。预测时将每个共享基因视为section1中的缺失基因，
section1实测表达仅作为伪真值用于评价。

两种方案使用完全相同的result_v6 embedding-KNN邻居表。唯一差别是平滑方案
在跨切片迁移前，对section2参考表达执行一次`alpha=0.25`的空间残差平滑。

## 原有评价指标

- 在全部空间点上计算raw和log1p表达的逐基因Pearson相关系数；
- 在固定抽取的20,000个空间点上计算逐基因Spearman相关系数；
- 在同一批抽样空间点上计算section1表达检出的Average Precision；
- 计算两种方案逐基因差值、改善比例、bootstrap区间和检出率分层结果。

由于Xenium与Visium HD的count尺度不同，原有评价以Spearman为主要指标。
这里属于模型训练后的伪缺失诊断，不是严格无泄漏验证：这些共享基因参与过
final embedding的训练。严格的留出基因验证需要先从section1模型输入中移除
评价基因，然后重新训练embedding。

## 运行共享基因验证与PCC正式绘图

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_shared_gene_validation/run_shared_gene_validation.py \
  --config gene_imputation_shared_gene_validation/configs/spatch_result_v6_shared4828_validation.json \
  --stage all

/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_shared_gene_validation/plot_shared_gene_validation.py \
  --config gene_imputation_shared_gene_validation/configs/spatch_result_v6_shared4828_validation.json
```

预测按每块64个基因执行，并支持断点续跑。流程只保存评价指标和6个标志基因
的全切片实测/预测结果，不会写出完整的4,828基因稠密预测矩阵。

已完成的result_v6运行位于
`runs/spatch_result_v6_shared4828_knn50_alpha0.25/`。`SUMMARY.md`记录原有
Spearman、Pearson和Average Precision补充指标；当前正式图以方案1、方案2的
`pcc_cell`和`pcc_gene`为核心，查看`figures/FIGURE_GUIDE.md`。

当前正式图包括PCC分布、逐cell/逐gene配对PCC、按section1真实检测率分层的
PCC分析，以及带逐gene PCC标注的共享marker空间图。旧版Spearman/AP图件、
旧说明和旧manifest完整保存在
`figures/legacy_pre_pcc_20260831/`，没有被删除或覆盖。

## 方案1 raw-scale PCC

指定的PCC定义已在`compute_scheme1_pcc.py`中独立实现：

- `pcc_cell`：逐空间点计算Pearson相关系数，再对有限结果取算术平均值；
- `pcc_gene`：逐基因计算Pearson相关系数，再对有限结果取算术平均值。

预测值和真值均保持raw尺度，不使用`normalize_total`、`log1p`或`scale`。

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_shared_gene_validation/compute_scheme1_pcc.py \
  --config gene_imputation_shared_gene_validation/configs/spatch_result_v6_shared4828_validation.json
```

PCC计算采用基因分块和断点保存，不会生成完整的665,399×4,828稠密预测矩阵。
正式PCC结果入口为
`runs/spatch_result_v6_shared4828_knn50_alpha0.25/SCHEME1_PCC.md`。

这些PCC只评价4,828个共享基因。13,257个source-only正式插补基因在section1
中没有实测真值，因此不能直接计算真实PCC。

## 方案2 normalize_total + log1p PCC

方案2在相同4,828个共享基因范围内，对pred和true分别按空间点执行
`normalize_total(target_sum=1e4) → log1p`，不使用`scale`，然后按与方案1
相同的定义计算`pcc_cell`和`pcc_gene`。

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_shared_gene_validation/compute_scheme2_pcc.py \
  --config gene_imputation_shared_gene_validation/configs/spatch_result_v6_shared4828_scheme2_pcc.json
```

方案2同样采用基因分块和断点保存，并复用方案1断点中完整4,828基因的raw行
总量。正式结果入口为
`runs/spatch_result_v6_shared4828_knn50_alpha0.25/SCHEME2_PCC.md`。
