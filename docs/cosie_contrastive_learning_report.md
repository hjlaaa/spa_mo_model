# COSIE Contrastive / Cross-modality Alignment Report

## 1. 本次阅读的文件列表

- `/home/hujinlan/cosie/COSIE/COSIE_framework.py`：COSIE 模型入口，包含 `COSIE_model`、`train_model()`、全图训练、子图训练、loss 组合、缺失模态恢复、最终 embedding 保存。
- `/home/hujinlan/cosie/COSIE/loss.py`：包含 `compute_joint()` 和 `crossview_contrastive_Loss()`，是 COSIE 中唯一显式命名为 contrastive 的 loss。
- `/home/hujinlan/cosie/COSIE/model_component.py`：包含 `GraphAutoencoder` 和 `Prediction_mlp`，分别用于各模态 GCN encoder/decoder 和跨模态 latent prediction。
- `/home/hujinlan/cosie/COSIE/linkage_construction.py`：包含 strong/weak linkage、跨 section triplet 构造、子图 linkage 构造。
- `/home/hujinlan/cosie/COSIE/utils.py`：包含 `compute_knn_graph()`、`construct_knn_graph_hnsw()`、`compute_neighborhood_embedding()` 等图构建和邻域 embedding 工具。
- `/home/hujinlan/cosie/COSIE/configure.py`：包含 GAE hidden dim、predictor hidden dim、epoch、KNN、loss 权重等默认配置。
- `/home/hujinlan/cosie/COSIE/data_preprocessing.py`：用于理解 `load_data()` 如何生成 `feature_dict`、`spatial_loc_dict`、`data_dict_processed`。
- `/home/hujinlan/cosie/Tutorials/1_Tutotial_of_SPOTS.ipynb`：SPOTS 示例，展示 RNA + Protein 输入、RNA-RNA cross-section linkage、全图训练和 `s1_embedding.npy` / `s2_embedding.npy` 保存。
- `/home/hujinlan/cosie/Tutorials/2_Tutorial_of_10x_Tonsil_integration.ipynb`：10x Tonsil 示例，展示 HE + RNA + Protein 输入、RNA-Protein cross-section linkage、metacell、子图训练和缺失模态预测。

## 2. COSIE 模型阶段总体调用链

Tutorial 中先构造 `data_dict`。SPOTS 示例为：

```python
data_dict = {
    "RNA": [adata1_rna, adata2_rna],
    "Protein": [adata1_adt, None],
}
```

10x Tonsil 示例先从 `adata*_adt.obsm["UNI_feature"]` 构造 HE AnnData，并复制 `.obsm["spatial"]`，再构造：

```python
data_dict = {
    "HE": [adata1_he, adata2_he],
    "RNA": [adata1_rna, None],
    "Protein": [None, adata2_adt],
}
```

随后调用：

```python
feature_dict, spatial_loc_dict, data_dict_processed = load_data(...)
```

`feature_dict` 是模型输入特征，结构是 `section -> modality -> Tensor[n_cells, input_dim]`。`spatial_loc_dict` 是 `section -> spatial coordinates`。`data_dict_processed` 保留预处理后的 AnnData，用于 linkage 和 metacell 还原。

模型阶段为：

```python
model = COSIE_model(config, feature_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
final_embeddings = model.train_model(
    file_path,
    config,
    optimizer,
    device,
    feature_dict,
    spatial_loc_dict,
    data_dict_processed,
    Linkage_indicator,
    ...
)
```

`COSIE_model.__init__()` 会按 modality 创建一个 `GraphAutoencoder`，并为同一 section 中共同出现的模态对创建双向 `Prediction_mlp`。`train_model()` 先构造 cross-section linkage，然后构造每个 section 的 spatial graph 和每个 section/modality 的 feature graph，再训练 GAE、cross-view loss、prediction loss 和 cross-section triplet loss。训练结束后，每个 section 的最终 embedding 保存为 `{section}_embedding.npy`。

## 3. COSIE 是否存在显式 contrastive learning

COSIE 中存在一个显式命名为 contrastive 的函数：

- 文件：`/home/hujinlan/cosie/COSIE/loss.py`
- 函数：`crossview_contrastive_Loss(view1, view2, gamma=9.0, EPS=...)`
- 调用位置：`/home/hujinlan/cosie/COSIE/COSIE_framework.py::COSIE_model.train_model()`

但它不是常见的 spot-level InfoNCE / CLIP 式对比学习：

- 未找到 `InfoNCE` 实现。
- 未找到 temperature 参数。
- 未找到基于 spot pair 的 softmax similarity matrix。
- `crossview_contrastive_Loss()` 没有显式构造 `HE_i` 与 `RNA_i` 为 positive pair、`HE_i` 与 `RNA_j` 为 negative pair。
- 该 loss 的 `compute_joint()` 对两个 view 的 shape 要求是 `[n_cells, dim]`，然后通过 `view1.unsqueeze(2) * view2.unsqueeze(1)` 得到并汇总为 `[dim, dim]` 的 feature-dimension joint probability matrix。

因此结论是：COSIE 有一个名为 cross-view contrastive 的 loss，但没有实现我们通常说的同 spot 正样本、不同 spot 负样本的 InfoNCE/CLIP-style 对比学习。实现上更接近基于两个 view 的 joint probability / marginal entropy 的 cross-view objective，再叠加 row-wise predictor MSE 和 cross-section triplet linkage。

## 4. COSIE 的跨模态对齐机制

COSIE 的跨模态对齐由多部分共同完成。

第一，modality-specific graph autoencoder。每个模态一个 `GraphAutoencoder`，默认 encoder 维度为：

```python
[input_dim, 256, 128]
```

encoder 使用 PyG `GCNConv`，输出后做 `F.normalize(x, p=2, dim=1)`。decoder 将 128 维 latent reconstruct 回原始输入维度，并用 MSE reconstruction loss 约束。

第二，同 section 内 co-present 模态之间使用 `crossview_contrastive_Loss()`。在 `train_model()` 中，每个 section 内遍历所有模态对：

```python
contrastive_loss = crossview_contrastive_Loss(
    embeddings[mod1],
    embeddings[mod2],
    config["training"]["gamma"],
)
```

这个 loss 对同一 section 内的模态 latent 起作用，但不是显式 spot-level positive/negative pair loss。

第三，dual prediction。`epoch >= start_dual_prediction` 后启用，默认从第 100 epoch 开始。对于同 section 内的模态对，COSIE 使用双向 predictor：

```python
pred_mod1_to_mod2 = predictor(embeddings[mod1])
pred_mod2_to_mod1 = predictor(embeddings[mod2])
prediction_loss = MSE(pred_mod1_to_mod2, embeddings[mod2]) + MSE(pred_mod2_to_mod1, embeddings[mod1])
```

这一项显式依赖同一 section 内模态矩阵的行顺序一致。它是更直接的 row-wise 跨模态 latent 对齐机制。

第四，missing modality prediction。训练和 evaluation 中，如果某个 section 缺少某个 modality，COSIE 会尝试用已有 modality 的 predictor 恢复缺失模态 latent。如果有多个 predictor 候选，会取平均；没有 predictor 时使用 zero tensor。

第五，cross-section triplet linkage。`epoch >= start_cross_section_integration` 后启用，默认从第 200 epoch 开始。它使用 `linkage_construction.py` 构造的 anchor/positive/negative triplets，并在 spatial-neighborhood embedding 上计算 `TripletMarginLoss`。

第六，final neighborhood smoothing。evaluation 阶段先 concat 所有模态 latent 或恢复出的 latent，再用 spatial graph 计算邻域平均 embedding，最终：

```python
bi_embedding = (concatenated_embedding + neighborhood_embedding) * 0.5
```

这一步不是 loss，而是最终 section embedding 的生成方式。

## 5. COSIE 的同切片内模态关系如何处理

COSIE 会处理同一个 section 内共同存在的模态对，但不是用户计划中严格的 spot-level pairwise InfoNCE。

- 同一个 section 内 HE/RNA/Protein 会做 pairwise `crossview_contrastive_Loss()`，前提是这些模态在该 section 同时存在。
- 同一个 spot 的不同模态没有被显式构造为 positive pair。
- 不同 spot 没有在该同切片 cross-view loss 中被显式构造为 negative pair。
- predictor MSE 会按行比较 `pred(mod1_i)` 和 `mod2_i`，因此它实际要求同一 section 内不同模态的 spot/cell 行顺序已经对齐。
- 显式 negative sample 只出现在 linkage triplet 中，且 linkage 是跨 section 的，不是同 section 内 HE/RNA/Protein 的 spot-level 对比。

所以，如果下一阶段实现“同 spot 不同模态为正样本、不同 spot 为负样本”的 InfoNCE，这会是新设计，不能称为严格复用 COSIE。

## 6. COSIE 的跨 section / linkage 机制

`linkage_construction.py` 中的 linkage 是跨 section 构造的。入口是：

```python
compute_linkages(data_dict, linkage_indicator, num_hvg=3000)
```

`linkage_indicator` 的格式类似：

```python
{
    ("s1", "s2"): [("RNA", "RNA")]
}
```

或：

```python
{
    ("s1", "s2"): [("RNA", "Protein")]
}
```

strong linkage 指同一 modality 在两个 section 间的 linkage，例如 RNA-RNA。COSIE 优先使用 `adata.obsm["{modality}_harmony"]`，若没有则使用 `adata.obsm["{modality}_pca"]`。`perform_strong_linkage_knn()` 做双向 nearest neighbor，positive 来自另一个 section，negative 从 anchor 所在 section 随机采样。

weak linkage 指不同但相关 modality 在两个 section 间的 linkage，例如 RNA-Protein。`perform_weak_linkage_knn()` 会构造共享 feature 空间：RNA-Protein 会使用 `Protein_gene_relationship.csv` 做 protein-gene mapping；其他 modality 尝试使用 `.var_names` 的交集。然后做 normalize/log/scale 的临时匹配矩阵，再双向 KNN 构造 triplet。

linkage 参与训练的方式是在 `train_model()` 中第 200 epoch 后使用 `TripletMarginLoss(margin=1.0, p=2)`。triplet loss 的输入不是单个模态 latent，而是所有模态 latent concat 之后再计算 spatial-neighborhood embedding。

因此 linkage 是跨 section 机制。它可以是跨 modality 的，例如 10x Tonsil tutorial 的 `("RNA", "Protein")`，但不是同切片内跨模态 spot pair contrastive learning。

## 7. COSIE 的空间图构建方式

空间图由 `/home/hujinlan/cosie/COSIE/utils.py::compute_knn_graph()` 构建。默认参数来自 `configure.py`：

```python
knn_neighbors_spatial = 5
```

输入是 `spatial_loc_dict[section]`，即 spatial coordinates。输出是 PyTorch `edge_index`，shape 为 `[2, n_edges]`。

但 COSIE 模型输入图不只有空间图。全图训练中，COSIE 还为每个 section/modality 的 feature tensor 构建 feature graph：

```python
knn_neighbors_feature = 30
feature_knn = compute_knn_graph(features, k_neighs_feature)
combined_knn = torch.cat([spatial_knn, feature_knn], dim=1)
```

`combined_knn` 作为该 section/modality 的 GCN `edge_index` 输入 `GraphAutoencoder.encoder()` 和 `decoder()`。子图训练中，feature graph 使用 `construct_knn_graph_hnsw()` 近似构建。

空间图还在两个地方单独使用：

- cross-section triplet loss 前，用 `compute_neighborhood_embedding()` 计算 spatial-neighborhood embedding；
- evaluation 阶段，用 spatial graph 对 concat embedding 做邻域平均，生成最终 `bi_embedding`。

因此，“只建空间图 KNN, K=5，不建 feature graph”与 COSIE 不完全一致。K=5 来自 COSIE，但 COSIE 原始训练还使用 feature graph K=30。

## 8. COSIE 的 embedding 维度和融合方式

`feature_dict` 中每个模态的输入维度来自 `load_data()` 的预处理结果。常见 tutorial 中：

- HE：`load_data(..., n_comps=50)` 后输入 50 维；
- RNA：输入 50 维；
- Protein：SPOTS 原始 21 个 ADT 最终日志显示 input 20；10x Tonsil protein 日志显示 input 20；
- 所有模态进入模型后，encoder 输出统一是 128 维。

每个模态的 encoder 默认是：

```python
[input_dim, 256, 128]
```

最终 embedding 的生成不是 attention，也不是 concat + MLP fusion。COSIE 在 evaluation 中对所有 `self.all_modalities` 按顺序取 latent。如果某个 section 缺模态，先用 predictor 恢复；恢复失败则 zero tensor。然后：

```python
concatenated_embedding = torch.cat([recovered_embeddings[mod] for mod in self.all_modalities], dim=1)
neighborhood_embedding = compute_neighborhood_embedding(spatial_graph[section], concatenated_embedding, device)
bi_embedding = (concatenated_embedding + neighborhood_embedding) * 0.5
```

所以最终维度是：

```text
128 * 全局模态数
```

SPOTS 示例全局模态为 RNA + Protein，因此 `s1_embedding.npy` / `s2_embedding.npy` 是 256 维。10x Tonsil 示例全局模态为 HE + RNA + Protein，因此最终 embedding 是 384 维。COSIE 没有把三模态融合压回 128 维，也没有保存每个模态单独的最终 latent 文件。

## 9. COSIE loss 公式与代码位置

### crossview_contrastive_Loss

- 源码：`/home/hujinlan/cosie/COSIE/loss.py`
- 函数：`crossview_contrastive_Loss(view1, view2, gamma=9.0, EPS=...)`
- 调用：`COSIE_framework.py::COSIE_model.train_model()`
- 输入：两个同 shape latent matrix，`[n_cells, 128]`
- 输出：scalar loss
- 默认参数：`gamma=5` 来自 `config["training"]["gamma"]`
- 默认权重：代码中直接加入 total loss，实际权重为 1.0

`compute_joint()` 先计算：

```python
p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
p_i_j = p_i_j.sum(dim=0)
p_i_j = (p_i_j + p_i_j.t()) / 2
p_i_j = p_i_j / p_i_j.sum()
```

loss 为：

```text
- sum p_ij * [log(p_ij) - (gamma + 1) log(p_i) - (gamma + 1) log(p_j)]
```

用途：同 section 内 co-present modality 的 cross-view latent objective。它用于跨模态对齐，但不是显式 spot-level positive/negative contrastive loss。

### reconstruction loss

- 源码：`/home/hujinlan/cosie/COSIE/COSIE_framework.py`
- 代码：`F.mse_loss(reconstructed, features)`
- 输入：GAE decoder 输出与原始 feature tensor
- 输出：scalar MSE
- 默认权重：`lambda1 = 0.1`

用途：每个 section/modality 内的 graph autoencoder reconstruction。不是跨模态 loss。

### prediction loss

- 源码：`/home/hujinlan/cosie/COSIE/COSIE_framework.py`
- 模块：`/home/hujinlan/cosie/COSIE/model_component.py::Prediction_mlp`
- 启用 epoch：`start_dual_prediction = 100`
- 默认权重：`lambda2 = 0.2`

公式：

```text
MSE(Pred_mod1_to_mod2(z_mod1), z_mod2)
+ MSE(Pred_mod2_to_mod1(z_mod2), z_mod1)
```

用途：同 section 内 co-present modality 的 row-wise latent prediction。它是 COSIE 中最直接依赖同 spot/cell 行顺序对齐的跨模态对齐 loss。

### triplet linkage loss

- 源码：`/home/hujinlan/cosie/COSIE/COSIE_framework.py`
- linkage 构造：`/home/hujinlan/cosie/COSIE/linkage_construction.py::compute_linkages()`
- loss：`torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction="mean")`
- 启用 epoch：`start_cross_section_integration = 200`
- 默认权重：`lambda3 = 1.0`

triplet 的 anchor/positive/negative 来自 cross-section linkage。训练中先 concat 所有模态 latent 或恢复 latent，再计算 spatial-neighborhood embedding，然后用 triplet index 取 anchor、positive、negative。

用途：跨 section 对齐。它可以是 same-modality strong linkage，也可以是 cross-modality weak linkage，但不是同 section 内 spot-level HE/RNA/Protein 对比。

## 10. 对我们新项目的可复用部分

可以直接复用或迁移的 COSIE 部分：

- `loss.py::compute_joint()` 和 `loss.py::crossview_contrastive_Loss()`：如果下一阶段要声称 COSIE-style cross-view objective，应使用这个 loss，而不是新写 InfoNCE。
- `model_component.py::GraphAutoencoder`：COSIE 的 modality-specific encoder/decoder，使用 GCNConv，默认 latent 128。
- `model_component.py::Prediction_mlp`：COSIE 的跨模态 latent predictor，默认 `[128, 512, 512, 128]`。
- `utils.py::compute_knn_graph()`：空间 KNN 和小规模 feature KNN 图构造。
- `utils.py::construct_knn_graph_hnsw()`：大规模 feature KNN 近似构图。
- `utils.py::compute_neighborhood_embedding()`：triplet loss 和最终 embedding smoothing 都用到。
- `linkage_construction.py::compute_linkages()`、`perform_strong_linkage_knn()`、`perform_weak_linkage_knn()`：如果要复用 COSIE 的跨 section linkage，可迁移。

不建议直接整块复用的部分：

- `COSIE_framework.py::train_model()` 是高度耦合的 monolithic 训练过程，包含构图、训练、缺失模态恢复、evaluation、保存 `.npy`、metacell 还原，直接复用不利于新项目模块化。
- `compute_linkages()` 当前对缺失模态会直接报错，需要新项目在调用前做缺失模态保护。
- 原始 `GraphAutoencoder.forward()` 没有 return，COSIE 实际直接调用 `encoder.encoder()` 和 `encoder.decoder()`；迁移时要注意这一点。
- 如果新项目只想使用空间图而不用 feature graph，这不是 COSIE 原始行为，需要在配置和文档中明确为新设计。

## 11. 对我们当前阶段计划的修正建议

对当前计划逐条判断：

1. 只建空间图 KNN, K=5：K=5 与 COSIE 默认 `knn_neighbors_spatial=5` 一致；但 COSIE 原始 GAE 输入图还 concat 了 feature graph，默认 `knn_neighbors_feature=30`。如果只用空间图，是简化版新设计，不能完全称为 COSIE 原始模型。

2. 每个模态先过 2 hidden layers MLP，输出 128：这与 COSIE 不一致。COSIE 每个模态使用 `GraphAutoencoder`，encoder 是 GCNConv，默认 `[input_dim, 256, 128]`。COSIE 的 2 hidden layer MLP 是 `Prediction_mlp`，用于跨模态预测，不是初始模态 encoder。

3. 同切片内三模态做跨模态对比学习：COSIE 会对同 section 内共同存在的模态对调用 `crossview_contrastive_Loss()`，但不是 spot-level positive/negative InfoNCE。如果实现 InfoNCE，则是新设计；如果要 COSIE-style，应迁移 `crossview_contrastive_Loss()` 和 predictor MSE。

4. 对比后 concat + MLP fusion：COSIE 不这么做。COSIE 是所有模态 128 维 latent concat，然后与 spatial-neighborhood average 做 0.5 加权平均。没有 MLP fusion、GELU、LayerNorm、dropout、mean residual。

5. 输出每个 stage 每个 spot 的 128 维 fused embedding：这与 COSIE 不一致。COSIE 输出维度是 `128 * 全局模态数`。HE + RNA + Protein 为 384 维；RNA + Protein 为 256 维；HE + RNA + Metabolite 若按三模态建模，也会是 384 维。

可以保留的部分：多 section/stage 分开构图、空间 KNN K=5、每模态统一到 128 维 latent、同 section 内模态对齐、最终按 section 输出 embedding。

需要改成 COSIE-style 的部分：modality encoder 改为 GCN graph autoencoder；对齐 loss 使用 `crossview_contrastive_Loss()` + predictor MSE；最终 embedding 默认 concat 多模态 latent 并做 spatial-neighborhood smoothing；如使用 cross-section 信息，则用 linkage triplet loss。

属于我们自己的新设计的部分：spot-level InfoNCE、纯 MLP modality encoder、concat + MLP fusion、输出固定 128 维 fused embedding、只用空间图不使用 feature graph。

## 12. 初步实现建议，但不要写代码

建议下一阶段模型代码按以下文件组织，但当前报告阶段不实现。

### `/home/hujinlan/spa_mo_model/model/graph_construction.py`

负责图构建。

- `build_spatial_knn_graph(spatial_coords, k=5) -> edge_index`
- `build_feature_knn_graph(features, k=30, method="sklearn"|"hnsw") -> edge_index`
- `build_cosie_combined_graph(spatial_edge_index, feature_edge_index) -> edge_index`

输入 shape：`spatial_coords [n_spots, 2]` 或 `[n_spots, d_coord]`，`features [n_spots, d_feature]`。输出 PyG `edge_index [2, n_edges]`。

### `/home/hujinlan/spa_mo_model/model/modality_encoder.py`

负责 COSIE-style modality-specific encoder。

- `CosieGraphAutoencoder`
- `ModalityEncoderRegistry`
- `encode_modalities(feature_dict, graph_dict) -> latent_dict`

输入：`feature_dict[section][modality] = Tensor[n_spots, d_in]`。输出：`latent_dict[section][modality] = Tensor[n_spots, 128]`。

### `/home/hujinlan/spa_mo_model/model/cosie_alignment.py`

负责对齐 loss。

- `compute_joint()`
- `crossview_contrastive_loss()`
- `CosiePredictionMLP`
- `compute_prediction_loss(latent_dict)`
- `compute_linkage_triplet_loss(latent_dict, spatial_graph_dict, linkage_results)`

如果要严格 COSIE-style，同 section 内不要替换成 InfoNCE，除非配置中明确标记为新实验。

### `/home/hujinlan/spa_mo_model/model/fusion.py`

建议同时支持两个路径：

- `CosieConcatNeighborhoodFusion`：COSIE-style，concat 所有模态 128 维 latent，缺失模态用 predictor 恢复，最后与 spatial-neighborhood embedding 平均，输出 `[n_spots, 128 * n_modalities]`。
- `ConcatMlpFusion`：新设计，concat + MLP 输出 `[n_spots, 128]`，但必须标记为非 COSIE 原始做法。

### `/home/hujinlan/spa_mo_model/model/stage_model.py`

负责训练 orchestration。

- `CosieStageModel`
- `train_one_epoch()`
- `evaluate_sections()`
- 输入：`feature_dict`、`spatial_loc_dict`、`processed_data_dict`、可选 `linkage_indicator`
- 输出：`final_embeddings[section] = np.ndarray`

建议把 COSIE 原始保存 `.npy` 的副作用改成可选配置，主返回值保留内存对象。

### `/home/hujinlan/spa_mo_model/model/model_config.py`

负责模型阶段配置。

建议配置项：

```python
latent_dim = 128
gae_hidden_dim = [256, 128]
predictor_hidden_dim = [512, 512]
knn_neighbors_spatial = 5
knn_neighbors_feature = 30
gamma = 5
lambda_reconstruction = 0.1
lambda_prediction = 0.2
lambda_triplet = 1.0
start_dual_prediction = 100
start_cross_section_integration = 200
epoch = 600
use_feature_graph = True
use_cosie_crossview_loss = True
use_linkage_triplet = True
```

## 13. 当前结论摘要

- COSIE 有显式命名为 `crossview_contrastive_Loss` 的 cross-view loss，但没有 InfoNCE/CLIP-style spot-level 正负样本对比。
- COSIE 同 section 内模态对齐主要来自 `crossview_contrastive_Loss()` 和 row-wise `Prediction_mlp` MSE。
- COSIE 的显式 positive/negative/triplet 构造来自 linkage，且 linkage 是跨 section 的。
- COSIE 每个模态使用 GCN graph autoencoder，不是普通 modality-specific MLP encoder。
- COSIE 原始图输入由 spatial KNN K=5 和 feature KNN K=30 concat 得到，不是只用空间图。
- COSIE 最终 embedding 是所有模态 128 维 latent concat 后，再与 spatial-neighborhood embedding 平均。
- COSIE 最终维度是 `128 * 模态数`，例如 RNA+Protein 为 256，HE+RNA+Protein 为 384。
- 缺失模态通过 predictor 预测 latent，预测失败时使用 zero tensor。
- 新项目可以复用 COSIE 的 GAE、prediction MLP、crossview loss、KNN graph、neighborhood embedding 和 linkage triplet。
- 如果新项目使用 InfoNCE、concat+MLP fusion、输出固定 128 维 fused embedding，应明确标记为新设计，而不是 COSIE 原始机制。
