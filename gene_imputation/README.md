# 可复用基因插补流程

本目录保存基于embedding进行基因插补所需的脚本、配置、日志、缓存和结果。
模型训练输出与原始数据保持为只读外部输入，并在每次运行中记录文件指纹。

## 方法

默认方法有意与COSIE的非参数预测保持一致：

1. 使用目标切片的final embedding作为查询；
2. 使用Annoy在来源切片中寻找`K`个近似最近邻空间点；
3. 对这些邻居的来源表达count取无权重均值，作为每个基因的预测值。

对于目标空间点`i`和基因`g`：

```text
prediction[i, g] = mean(source_expression[neighbors[i], g])
```

本实现与COSIE的差别仅在内存组织方式。邻居均值被表示为稀疏迁移矩阵`W`，
然后按基因分块计算`W @ X_source`。每个完成的基因块会立即写入压缩HDF5，
内存中不会同时保存完整的稠密来源矩阵或预测矩阵。

在邻居表固定时，该计算与COSIE逐空间点取均值的公式等价，仅可能存在
float32舍入误差。正式运行会与直接逐点实现进行核对，并将结果记录在
`validation.json`中。

## 目录结构

```text
gene_imputation/
  configs/                    不同版本的JSON配置
  runs/                       带版本号的结果和可复用KNN缓存
  tests/                      数学等价性测试
  run_spatch_knn_imputation.py
```

第一份正式配置为`configs/spatch_result_v6_final_knn50.json`，对应结果目录为
`runs/spatch_result_v6_final_knn50/`。

## 运行或断点续跑

使用与COSIE相同的Python环境：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation/run_spatch_knn_imputation.py \
  --config gene_imputation/configs/spatch_result_v6_final_knn50.json \
  --stage all
```

可选阶段为`audit`、`neighbors`、`impute`、`validate`、`summarize`和`all`。
邻居查询和基因分块均有完成标记，因此重复执行相同命令可以从中断位置继续。
兼容的现有结果会被复用；如果配置不兼容，流程会报错停止，不会覆盖旧数据。

分析后续模型时，应复制JSON配置，修改`run_name`、两个embedding路径，并指定
新的`run_dir`，然后执行相同命令。不要让修改后的模型继续写入旧运行目录。
进行受控模型比较时，应保持`K`、距离度量、目标基因和预处理方式一致。

## 已完成的result_v6结果

第一版完整运行对section1全部665,399个空间点预测了13,257个section2特有
基因。实测峰值常驻内存为2.36 GiB；Annoy查询耗时44.7秒，分块HDF5插补
耗时637.9秒。压缩结果为9.60 GiB；若保存为未压缩float32矩阵则约32.86 GiB，
因此能够在服务器62 GiB内存限制下运行。

## 结果格式

`imputed_section1_section2_only_genes.h5`是分块压缩HDF5文件：

```text
/X                              float32 [目标空间点, 目标基因]
/obs_names                      目标空间点标准ID
/var_names                      按section2顺序保存的目标基因
/obsm/spatial                   目标空间坐标
/gene_stats/source_nnz
/gene_stats/source_total
/gene_stats/predicted_mean
/gene_stats/predicted_nonzero_fraction
/gene_stats/predicted_max
/completed_gene_blocks          断点续跑标记
```

只读取少量指定基因，无需载入完整结果：

```python
import h5py

path = "gene_imputation/runs/spatch_result_v6_final_knn50/imputed_section1_section2_only_genes.h5"
with h5py.File(path, "r") as handle:
    genes = handle["var_names"].asstr()[:]
    wanted = ["IGKC", "OLFM4", "COL1A1"]  # 按HDF5列顺序递增排列
    columns = [list(genes).index(gene) for gene in wanted]
    values = handle["X"][:, columns]
```

除非机器内存足以容纳完整稠密矩阵，否则不要执行`handle["X"][:]`。

如需按基因名导出，可使用：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation/extract_imputed_genes.py \
  --input gene_imputation/runs/spatch_result_v6_final_knn50/imputed_section1_section2_only_genes.h5 \
  --genes IGKC COL1A1 OLFM4 \
  --output gene_imputation/runs/spatch_result_v6_final_knn50/exports/selected_markers.npz
```

导出文件只保存指定基因列，以及空间点ID和坐标。

## 正式结果图

使用以下命令生成可复现的空间标志基因图、基因QC图和KNN-QC图：

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  gene_imputation/plot_imputation_results.py \
  --config gene_imputation/configs/spatch_result_v6_formal_plots.json
```

PNG、PDF和图件manifest会写入正式运行的`figures/`目录。空间图会按照精确的
section1整数网格绘制全部空间点。

两个切片均有实测值的4,828个共享基因在独立配对流程
`gene_imputation_shared_gene_validation/`中进行伪缺失准确性评价。

该未平滑方案的方案1 raw-scale PCC结果保存在
`runs/spatch_result_v6_final_knn50/SCHEME1_PCC.md`。这里仅报告共享基因的
`pcc_cell`和`pcc_gene`；13,257个source-only正式预测基因没有section1真值。

该未平滑方案的方案2 normalize_total + log1p PCC结果保存在
`runs/spatch_result_v6_final_knn50/SCHEME2_PCC.md`，机器可读结果为
`scheme2_shared4828_pcc.json`。方案1与方案2使用相同预测和真值，差别仅在
PCC计算前的表达预处理尺度。
