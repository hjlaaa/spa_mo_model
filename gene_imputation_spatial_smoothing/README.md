# 参考数据空间平滑基因插补

本目录是`gene_imputation/`未平滑方案的独立对照。两种方案使用相同的
result_v6 final embedding、Annoy KNN设置、目标空间点和13,257个section2特有
目标基因。唯一的流程差异是：进行跨切片迁移前，先对section2参考表达执行
一次空间去噪。

## 方法

设来源表达矩阵为`X`，来源空间邻居均值矩阵为`S`，跨切片KNN迁移矩阵为`W`，
平滑强度为`alpha`：

```text
X_smooth  = (1 - alpha) X + alpha S X
X_imputed = W X_smooth
```

第一份配置使用`alpha=0.25`。`S`在section2整数网格上连接半径1.5范围内的
8个直接邻居；一个网格步长约为7.994微米。该图不会跨越更大的组织空隙。
没有局部邻居的行使用自身，因此孤立空间点保持不变。

空间平滑与插补均采用稀疏矩阵并按每块64个基因计算，然后直接写入压缩
float32 HDF5。内存中不会保存完整的平滑来源矩阵或完整稠密预测矩阵。

## 运行或断点续跑

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_spatial_smoothing/run_spatch_smoothed_knn_imputation.py \
  --config gene_imputation_spatial_smoothing/configs/spatch_result_v6_final_knn50_spatial8_alpha0.25.json \
  --stage all
```

可选阶段为`audit`、`neighbors`、`spatial_graph`、`impute`、`validate`、
`summarize`和`all`。邻居分块和基因分块均支持断点续跑。

使用后续模型时，应复制配置并修改`run_name`、`run_dir`和两个embedding路径。
进行受控比较时，应保持KNN和平滑设置不变，并且不要复用旧模型的运行目录。

## 输出内容

每次运行会保存空间图及manifest、Annoy邻居缓存、压缩插补HDF5、验证报告、
逐基因QC，以及与未平滑基线的抽样比较。应保留`gene_imputation/`的原始结果，
将其作为`alpha=0`对照。

可使用本目录中的`extract_imputed_genes.py`按基因导出指定列，无需载入完整
HDF5矩阵。

## 正式结果图

使用以下命令生成平滑方案的空间标志基因图、匹配的未平滑/平滑/差值图，
以及基因层面的平滑对照图：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation_spatial_smoothing/plot_smoothed_imputation_results.py \
  --config gene_imputation_spatial_smoothing/configs/spatch_result_v6_formal_plots.json
```

两种方案使用相同的标志基因集合、精确section1网格、表达转换和逐基因色阶。
PNG、PDF和图件manifest会写入正式运行的`figures/`目录。

两个切片均有实测值的4,828个共享基因在
`gene_imputation_shared_gene_validation/`中进行配对伪缺失准确性评价。

该平滑方案的方案1 raw-scale PCC结果保存在
`runs/spatch_result_v6_final_knn50_spatial8_alpha0.25/SCHEME1_PCC.md`。这里仅
报告共享基因的`pcc_cell`和`pcc_gene`；13,257个source-only正式预测基因没有
section1真值。

该平滑方案的方案2 normalize_total + log1p PCC结果保存在
`runs/spatch_result_v6_final_knn50_spatial8_alpha0.25/SCHEME2_PCC.md`，机器
可读结果为`scheme2_shared4828_pcc.json`。方案1与方案2使用相同预测和真值，
差别仅在PCC计算前的表达预处理尺度。
