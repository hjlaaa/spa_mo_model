# COSIE Preprocessing Report

## 1. 本次阅读的文件列表

- `/home/hujinlan/cosie/COSIE/data_preprocessing.py`：核心多模态 `AnnData` 预处理入口，包含 `preprocess_adata()`、`load_data()`、protein CLR、metacell 构造与 metacell embedding 还原。
- `/home/hujinlan/cosie/COSIE/image_preprocessing.py`：HE/H&E 原图读取、mask/superpixel 坐标生成、UNI 模型加载、patch 构造和 image embedding 提取。
- `/home/hujinlan/cosie/COSIE/configure.py`：COSIE 模型和训练默认超参数，不包含数据预处理算法。
- `/home/hujinlan/cosie/COSIE/utils.py`：随机种子、KNN/近似 KNN 图、空间邻域 embedding 等工具，服务图构建和训练。
- `/home/hujinlan/cosie/COSIE/COSIE_framework.py`：模型主体，消费 `load_data()` 产生的 `feature_dict`/`spatial_loc_dict`，训练后保存 `s1_embedding.npy` 等最终 embedding，并在训练/推理阶段补齐缺失模态 embedding。
- `/home/hujinlan/cosie/COSIE/linkage_construction.py`：额外阅读；负责跨 section linkage，包含 `Protein_gene_relationship.csv` 的真实调用路径。
- `/home/hujinlan/cosie/COSIE/downstream_analysis.py`：额外阅读；负责下游聚类、可视化、KNN 预测缺失模态表达，不是原始预处理主入口。
- `/home/hujinlan/cosie/COSIE/Protein_gene_relationship.csv`：protein 到 RNA gene 的映射表，被 linkage 构造使用。
- `/home/hujinlan/cosie/Tutorials/0_Tutorial_of_image_feature_extraction.ipynb`：演示从 HE.jpg + Image_mask.png 用 UNI 提取 `uni_embeddings.pickle`。
- `/home/hujinlan/cosie/Tutorials/1_Tutotial_of_SPOTS.ipynb`：演示 SPOTS RNA + ADT/Protein 输入、`load_data()` 预处理、训练和 Protein 缺失预测。
- `/home/hujinlan/cosie/Tutorials/2_Tutorial_of_10x_Tonsil_integration.ipynb`：演示 10x Tonsil RNA + ADT/Protein + 已有 UNI feature 的 HE 输入、metacell、subgraph 训练和缺失 RNA/Protein 预测。
- `/home/hujinlan/cosie/README.md`：项目用途、安装和依赖说明，不包含具体预处理流程。
- 只读检查的数据/产物：`data/SPOTS/*.h5ad`、`Tonsil_10x/*.h5ad`、`data/SPOTS/s*_embedding.npy`、`Tonsil_10x/s*_embedding.npy`、`Tutorials/uni_embeddings.pickle` 的字段和形状。

## 2. COSIE 项目整体代码结构理解

- `COSIE/data_preprocessing.py`：把每个模态的 `AnnData.X` 转成模型输入 embedding。RNA/非 protein omics 走 HVG、`normalize_total`、`log1p`、`scale`、PCA；Protein 走 CLR、`scale`、PCA；HE embedding 直接 PCA。`load_data()` 负责多 section 拼接、shared modality Harmony、`None` 缺失占位跳过、空间坐标一致性检查。
- `COSIE/image_preprocessing.py`：负责 HE 图像原始特征提取。它能加载原图和 mask，生成 superpixel/patch 坐标，用 UNI ViT 提取 2048 维特征，并保存 pickle。
- `COSIE/configure.py`：返回默认训练配置，包括 GAE hidden dim `[256, 128]`、预测 MLP hidden dim `[512, 512]`、epoch、KNN 数等。
- `COSIE/COSIE_framework.py`：COSIE 模型定义和训练。它不读取原始 h5ad 或图像，而是使用已预处理的 `feature_dict` 和 `spatial_loc_dict` 构图、训练、保存最终 section embedding。
- `COSIE/utils.py`：图和 KNN 工具，例如 `compute_knn_graph()`、`construct_knn_graph_hnsw()`、`compute_neighborhood_embedding()`。
- `Tutorials/0_Tutorial_of_image_feature_extraction.ipynb`：HE 原图特征抽取教程。
- `Tutorials/1_Tutotial_of_SPOTS.ipynb`：SPOTS 数据上 RNA + Protein/ADT 的整合示例；故意设定 section2 缺 Protein。
- `Tutorials/2_Tutorial_of_10x_Tonsil_integration.ipynb`：10x Tonsil 数据上 HE + RNA + Protein/ADT 的整合示例；HE 使用 h5ad 中已有 `UNI_feature`，不是现场从原图提取。
- `data/SPOTS`：包含 section1/section2 的 RNA h5ad、ADT h5ad，以及训练后或预生成的 `s1_embedding.npy`、`s2_embedding.npy`。
- `Tonsil_10x`：包含两个 section 的 RNA h5ad、ADT h5ad，以及训练后或预生成的 `s1_embedding.npy`、`s2_embedding.npy`；ADT h5ad 里还包含 `obsm['UNI_feature']`。

## 3. HE / H&E 图像模态预处理流程

COSIE 支持从原始 HE/H&E 图像提取特征，具体实现位于 `/home/hujinlan/cosie/COSIE/image_preprocessing.py`：

- 原图读取：`load_image(filename)` 使用 PIL 读取 `.png/.jpg/.tif` 等图像，转成 NumPy array；如果有 alpha channel 会去掉。教程示例 `HE.jpg` 输出形状为 `(4000, 4000, 3)`。
- 可选缩放：`rescale_image(img, scale)` 存在，但 notebook 只说明示例图像分辨率为 `0.5 microns per pixel`，没有实际调用缩放；没有 magnification 参数。
- mask 使用：`0_Tutorial` 读取 `Image_mask.png`，mask 输出形状 `(4000, 4000)`。教程说明只保留 mask 中白色的 `16x16` superpixel。
- superpixel 坐标：`get_white_superpixel_centers(image_path, superpixel_size=16)` 扫描整张 mask，每个完整白色 `16x16` block 取中心点。教程中找到 `29398` 个 superpixels。
- 坐标变换：`get_white_superpixel_centers()` 返回 `(x, y)`，教程随后执行 `centers = centers[:, [1, 0]]` 变成 `(y, x)`，并用 `spatial_location = (centers - 8) // 16` 得到空间网格坐标。
- patch 切割：`PatchDataset` 对每个中心点提取 `224x224` patch，即中心上下左右各 `112` pixels；边界不足处用 0 padding。没有 stride 参数；真实采样位置由 mask 白色 `16x16` superpixel 中心决定。
- 图像 transform：`transforms.ToTensor()` 后使用 ImageNet mean/std normalize，mean `(0.485, 0.456, 0.406)`，std `(0.229, 0.224, 0.225)`。
- UNI 模型：`create_model(local_dir)` 使用 `timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, global_pool='')`，然后从 `os.path.join(local_dir, "pytorch_model.bin")` 加载权重。教程中 `uni_local_dir = '/home/hujinlan/cosie/UNI'`，实际权重为 `/home/hujinlan/cosie/UNI/pytorch_model.bin`。
- embedding 提取：`extract_features()` 返回 global/token embedding 和 local patch embedding。教程输出第一批 `feature_emb` 形状为 `[128, 197, 1024]`，`patch_emb` 形状为 `[128, 1024, 14, 14]`。`image_feature_extraction()` 对每个 patch 取 `feature_emb[idx, 0]` 作为 1024 维 CLS/全局特征，取 `patch_emb[idx, :, 7, 7]` 作为 1024 维中心局部特征，然后 concat 成 2048 维。
- 输出格式：`image_feature_extraction()` 将 list of NumPy arrays 保存为当前工作目录下硬编码文件名 `uni_embeddings.pickle`。函数参数里有 `path=None`，但源码没有使用该参数。notebook 文本写的是 `uni_embeddings.pkl`，源码和实际输出是 `uni_embeddings.pickle`。
- 输出形状：`Tutorials/uni_embeddings.pickle` 实际为 list，长度 `29398`，转成 array 后形状 `(29398, 2048)`。

HE 在整合 tutorial 中有两种真实路径：

- `0_Tutorial`：从 `HE.jpg` + `Image_mask.png` 直接生成 `uni_embeddings.pickle`，再手动 `PCA(n_components=50)`，构造 `sc.AnnData(X=image_feature_pca)`，并写入 `adata_img.obsm["spatial"] = spatial_location` 用于聚类可视化。
- `2_Tutorial`：不从原图提取；直接从 ADT h5ad 中读取已存在的 `adata1_adt.obsm['UNI_feature']` 和 `adata2_adt.obsm['UNI_feature']`，构造 `adata1_he = sc.AnnData(X=...)`、`adata2_he = sc.AnnData(X=...)`，再复制 ADT 的 `obsm['spatial']`。两个 HE AnnData 形状分别为 `(234983, 2048)` 和 `(184956, 2048)`。之后由 `load_data()` 对 shared HE 做 PCA 50 维并默认 Harmony。

`s1_embedding.npy` / `s2_embedding.npy` 不是 HE 原图预处理输出，而是 `COSIE_model.train_model()` 的最终整合 embedding 输出，源码中通过 `np.save(os.path.join(file_path, f"{section}_embedding.npy"), bi_embedding_numpy)` 保存。只读检查到：

- `data/SPOTS/s1_embedding.npy`：`(2568, 256)`，`float32`。
- `data/SPOTS/s2_embedding.npy`：`(2759, 256)`，`float32`。
- `Tonsil_10x/s1_embedding.npy`：`(234983, 384)`，`float64`。
- `Tonsil_10x/s2_embedding.npy`：`(184956, 384)`，`float64`。

维度可由模型逻辑解释：最终 embedding 是所有模态 128 维 latent 的拼接，SPOTS 中 `RNA + Protein = 256`，Tonsil 中 `HE + RNA + Protein = 384`。

## 4. RNA 模态预处理流程

- 输入格式：tutorial 直接用 `scanpy.read_h5ad()` 读取 h5ad，输入对象是 `AnnData`。
- 使用字段：预处理使用 `.X` 作为 RNA 表达矩阵，使用 `.var_names` 做 feature 对齐/HVG，使用 `.obsm['spatial']` 做空间坐标；未看到 `.layers` 用于 RNA 预处理，检查到示例 h5ad 的 `layers` 为空。
- filtering：`preprocess_adata()` 中没有 cell/gene filtering。
- HVG：当 `hvg_num` 不为 `None` 且 `len(adata.var_names) > hvg_num` 时，RNA/RNA_panel2 使用 `sc.pp.highly_variable_genes(..., n_top_genes=hvg_num, flavor="seurat_v3", batch_key='batch' if use_batch else None)`。默认 `hvg_num=3000`。
- normalize：RNA 走 `sc.pp.normalize_total(adata, target_sum=target_sum)`；如果 `target_sum is None`，使用 Scanpy 默认。
- log transform：RNA 走 `sc.pp.log1p(adata)`。
- scale：RNA 走 `sc.pp.scale(adata)`。
- PCA：RNA 走 `sc.tl.pca(adata, n_comps=n_comps)`，默认 `n_comps=50`。
- Harmony / batch correction：只在 `load_data()` 判断某模态出现在多个 section 时发生。shared modality 会先取共同 var_names、concat、写入临时 `obs['batch']`，再 `preprocess_adata()`，默认 `use_harmony=True` 时调用 `sc.external.pp.harmony_integrate(adata_combined, key='batch')`，输出使用 `obsm['X_pca_harmony']`。
- 是否直接使用已有 embedding：RNA 预处理没有直接使用已有 RNA embedding；用的是 `.X` 现算 PCA/Harmony。
- 输出位置：shared RNA 默认写回每个原始 AnnData 的 `.obsm['RNA_harmony']`，并放入 `feature_dict['sN']['RNA']`；若 `use_harmony=False` 则写 `.obsm['RNA_pca']`。unique RNA 写 `.obsm['RNA_pca']`。
- 输出维度：默认 50。SPOTS tutorial 输出日志显示 RNA encoder input 为 50。10x Tonsil 中 RNA 只放在 section1，因此作为 unique modality PCA 到 50。
- 多样本组织：`data_dict = {'RNA': [adata_section1, adata_section2, ...]}`，缺失位置用 `None`。`load_data()` 用 list index 映射到 `s1/s2/...`。

示例数据检查：

- SPOTS section1 RNA：`(2568, 32285)`，`obsm['spatial']` 为 `(2568, 2)`。
- SPOTS section2 RNA：`(2759, 32285)`，`obsm` 有 `location1/location2/spatial`，预处理只使用 `spatial`。
- Tonsil section1 RNA：`(234983, 3000)`，`obsm['spatial']`。
- Tonsil section2 RNA：`(184956, 3000)`，`obsm['spatial']`。

## 5. protein / ADT 模态预处理流程

- 输入格式：tutorial 读取 ADT h5ad，但在 `data_dict` 中命名为 `'Protein'`。COSIE 源码识别的是 modality string `"Protein"`，不是 `"ADT"`。
- 使用字段：Protein/ADT 预处理使用 AnnData `.X`，空间对齐使用 `.obsm['spatial']`。示例 h5ad `layers` 为空。
- CLR normalization：`preprocess_adata()` 中 `modality == "Protein"` 时调用 `clr_normalize_each_cell()`。其 per-cell 公式是：对每个 cell 取正值 `x[x > 0]` 的 `sum(log1p())`，除以 feature 数得到指数基准，再返回 `log1p(x / exp)`。
- scale：CLR 后调用 `sc.pp.scale(adata)`。
- PCA：Protein PCA 维度按 feature 数决定：`n_proteins >= n_comps` 用 `n_comps`；`n_proteins >= 20` 用 20；否则用 15。SPOTS ADT 21 个 feature，Tonsil section2 ADT 39 个 feature，tutorial 输出 Protein encoder input 均为 20。
- normalize_total/log1p：Protein 主预处理不走 `normalize_total` 和普通 `log1p`，只走 CLR 里的 `log1p`。
- batch correction：如果 Protein 在多个 section 都存在，则作为 shared modality 进入 `load_data()` 的 concat + Harmony 流程；SPOTS tutorial 设定 Protein 只在 section1，Tonsil tutorial 设定 Protein 只在 section2，因此 tutorial 中 Protein 都是 unique modality，不做 Harmony。
- 是否和 RNA 共享 obs_names：只读检查示例 h5ad 中同一 section 的 RNA 与 ADT `obs_names` 顺序相等，`obsm['spatial']` 也完全相等。但 COSIE 代码没有显式按 `obs_names` reindex，只依赖输入行顺序以及空间坐标一致性检查。
- 是否使用 `Protein_gene_relationship.csv`：Protein 主预处理不使用该文件。它在 `/home/hujinlan/cosie/COSIE/linkage_construction.py` 的 `load_protein_gene_mapping()` 中被读取，路径为 `Path(__file__).parent / 'Protein_gene_relationship.csv'`。
- `Protein_gene_relationship.csv` 调用场景：`perform_weak_linkage_knn()` 在 `RNA`/`RNA_panel2` 与 `Protein` 跨模态 linkage 时调用该映射，把 protein feature 名映射到 RNA gene 名，再在重叠 feature 空间中做 KNN linkage。该 linkage 内部会对用于匹配的临时子矩阵执行 `normalize_total`、`log1p`、`scale`，但这是 linkage 构造，不是 `load_data()` 的 Protein feature 预处理。
- 输出：`.obsm['Protein_pca']` 或 shared 时 `.obsm['Protein_harmony']`，同时放入 `feature_dict['sN']['Protein']` 作为 `torch.FloatTensor`。

示例数据检查：

- SPOTS section1 ADT：`(2568, 21)`，`obsm['spatial']`。
- SPOTS section2 ADT：`(2759, 21)`，`obsm['spatial']`，但 tutorial 中故意设为缺失，不进入 `data_dict['Protein'][1]`。
- Tonsil section1 ADT：`(234983, 35)`，`obsm['UNI_feature']` 为 `(234983, 2048)`，`obsm['spatial']`；tutorial 中它只为 HE 提供 UNI feature 和后续对照，不作为 observed Protein 输入。
- Tonsil section2 ADT：`(184956, 39)`，`obsm['UNI_feature']` 为 `(184956, 2048)`，`obsm['spatial']`；tutorial 中作为 observed Protein 输入。

## 6. metabolite / metabolomics 模态预处理流程

检索结果显示：COSIE 中未找到独立、专用、可运行教程级别的 metabolite/metabolomics 预处理实现；没有 metabolomics 数据目录、h5ad 示例、notebook 调用或单独 IO 逻辑。

但源码里存在一个通用 `Metabolite` 字符串分支：

- `/home/hujinlan/cosie/COSIE/data_preprocessing.py` 的 docstring 声明支持 `'Metabolite'`。
- `preprocess_adata()` 在 HVG 判断中把 `"Metabolite"` 放进非 Protein 的 omics 集合。
- 如果用户传入 `modality == "Metabolite"`，实际流程会走普通非 HE、非 Protein 分支：可选 HVG、`normalize_total`、`log1p`、`scale`、PCA。
- 输入格式只能从通用逻辑推断为 `AnnData`，且 `.X` 为矩阵、`.obsm['spatial']` 用于空间坐标；源码没有 metabolite 专属字段、文件格式、log transform 特例、scale 特例或空间对齐特例。
- 输出会与其他 unique/shared modality 一样写入 `.obsm['Metabolite_pca']` 或 `.obsm['Metabolite_harmony']`，并进入 `feature_dict`。

因此，严格结论是：COSIE 中未找到明确的 metabolite/metabolomics 专用预处理实现；仅找到一个 generic `Metabolite` 分支，算法等同于 RNA/其他非 Protein omics 的通用分支。

## 7. 多模态样本对齐方式

- 组织方式：`data_dict` 是 `{modality: [section1_adata_or_None, section2_adata_or_None, ...]}`。section list 的位置决定 section 编号，最终改名为 `s1/s2/...`。
- spot/cell 对齐：同一 section 内，COSIE 不按 `obs_names` 做 join 或 reindex，而是假设各模态 AnnData 行顺序已经对齐。
- 空间坐标检查：`load_data()` 会收集同一 section 中所有非缺失模态的 `.obsm['spatial']`，如果多个模态存在，则要求所有 spatial 数组 `np.array_equal()`；否则抛出 `ValueError("inconsistent spatial information")`。
- `obs_names`：shared modality 跨 section concat 前，COSIE 会给 `obs_names` 加上 `_{section_idx}` 防止重复；这不是同 section 模态对齐机制。只读检查发现示例 RNA/ADT h5ad 在同一 section 内 `obs_names` 顺序相等，但代码没有依赖它。
- HE 对齐：10x tutorial 中 `adata_he = sc.AnnData(X=adata_adt.obsm['UNI_feature'])`，再复制 `adata_adt.obsm['spatial']`。这个 HE AnnData 没有继承 ADT 的 `obs`/`obs_names`，真实对齐依赖 `UNI_feature` 与 ADT 行顺序一致以及 spatial 复制。
- 空间坐标：用于一致性检查、空间 KNN 图构建、subgraph 切分和可视化。
- 样本顺序：非常关键。`data_dict['RNA'][0]` 就是 `s1`，`data_dict['RNA'][1]` 就是 `s2`。`Linkage_indicator` 的 `('s1','s2')` 也依赖这个顺序。
- section1/section2：`load_data()` 对每个 section 单独输出 `feature_dict['s1']`、`feature_dict['s2']`。shared modality 可跨 section joint preprocess + Harmony；unique modality 单 section 独立 preprocess。
- sample_id/batch：原始示例 h5ad 的 `obs` 没有显式 sample_id/batch 列。`load_data()` 会为 shared modality concat 后临时创建 `obs['batch'] = batch_0/batch_1/...` 用于 Harmony。

## 8. 缺失模态处理方式

COSIE 原项目部分支持缺失模态，但需要按它的约定组织：

- `load_data()` 明确支持在 `data_dict` list 中用 `None` 表示某个 section 缺失某模态；循环中会跳过 `None`。
- 如果缺少 HE embedding：SPOTS tutorial 完全不包含 HE，能正常跑 RNA/Protein。若某 section 的 `HE` 为 `None`，`load_data()` 可跳过；但模型是否能补 HE 取决于训练中是否存在可用 predictor。
- 如果缺少 ADT/Protein：tutorial 明确支持。SPOTS 中 `Protein: [adata1_adt, None]`，模型评估阶段输出 “Missing modality [Protein] in Section [s2]” 并用 `RNA -> Protein` predictor 补 embedding。Tonsil 中 `Protein: [None, adata2_adt]`，用 `HE -> Protein` predictor 补 section1。
- 如果缺少 RNA：Tonsil tutorial 中 `RNA: [adata1_rna, None]`，模型评估阶段用 `HE -> RNA` predictor 补 section2。
- 如果缺少 metabolite：没有专门逻辑。它只能作为普通 modality 走同一套 `None` 占位、predictor 补 embedding、下游 KNN 预测逻辑。
- 限制：`compute_linkages()` 如果 `Linkage_indicator` 指向缺失的 modality/section，会直接抛错；因此 linkage 配置必须只引用真实存在的源数据。模型中如果没有任何 predictor 可用于补某缺失模态，会使用 zero tensor；若一个 section 完全没有任何模态，源码中存在不完整的 `raise Val` 路径，会失败。

迁移到新项目时，最薄兼容层建议：

- 保留 COSIE 原算法不改，只在 wrapper 中把未提供的模态填成 `None`。
- 在调用 `load_data()` 前验证每个 section 至少有一个非缺失模态和可用 `obsm['spatial']`。
- 自动剔除 `Linkage_indicator` 中引用 `None` 的 pair，或者在配置校验时报出清晰错误。
- 对外 API 返回结构中，缺失的 raw/preprocessed modality 保持 `None`；如果训练后 COSIE 预测了 embedding，再单独放在 `predicted_embeddings` 或 `final_embeddings` 中，避免把预测 embedding 冒充原始预处理结果。

## 9. 可直接复用的函数 / 类 / 代码块

| 原始路径 | 函数/类 | 输入参数 | 输出 | 可直接复用 | 是否需要 adapter |
|---|---|---|---|---|---|
| `COSIE/data_preprocessing.py` | `preprocess_adata` | `adata_raw, modality, hvg_num=3000, n_comps=50, target_sum=None` | 含 `obsm['X_pca']` 的 AnnData copy | 可复用 | 需要；统一 modality 名称、控制 target_sum/HVG |
| `COSIE/data_preprocessing.py` | `load_data` | `data_dict, n_comps=50, hvg_num=3000, target_sum=None, use_harmony=True, metacell=False` | `feature_dict, spatial_loc_dict, data_dict` | 可复用 | 需要；构造 `data_dict`、处理缺失、校验 spatial |
| `COSIE/data_preprocessing.py` | `clr_normalize_each_cell` | `adata, inplace=True` | CLR-normalized AnnData | 可复用 | 基本不需要 |
| `COSIE/data_preprocessing.py` | `metacell_construction_optimized` | `adata` with `.X` and `.obsm['spatial']` | metacell AnnData | 可复用 | 需要；只适合规则网格且空间步长有效的数据 |
| `COSIE/data_preprocessing.py` | `construct_metacell_data_dict` | multimodal `data_dict` | metacell 后的 `data_dict` | 可复用 | 需要；配置是否启用 |
| `COSIE/data_preprocessing.py` | `reconstruct_metacell_to_original` | `adata_metacell, metacell_embedding` | original cell-level embedding | 可复用 | 基本不需要 |
| `COSIE/image_preprocessing.py` | `load_image` | `filename, verbose=True` | image ndarray | 可复用 | 需要；路径配置化 |
| `COSIE/image_preprocessing.py` | `rescale_image` | `img, scale` | rescaled ndarray | 可复用 | 需要；把 scale 放入 config |
| `COSIE/image_preprocessing.py` | `get_white_superpixel_centers` | `image_path, superpixel_size=16` | center list | 可复用 | 需要；明确坐标顺序和 superpixel_size |
| `COSIE/image_preprocessing.py` | `PatchDataset` | `image, location` | torch Dataset | 可复用 | 需要；patch size 224 写死 |
| `COSIE/image_preprocessing.py` | `create_model` | `local_dir` | timm UNI ViT model | 可复用 | 需要；UNI 路径配置化 |
| `COSIE/image_preprocessing.py` | `extract_features` | `model, batch` | global/local embeddings | 可复用 | 基本不需要 |
| `COSIE/image_preprocessing.py` | `image_feature_extraction` | `he_image, uni_local_dir, cell_location, device, batch_size, num_workers, path=None` | 当前目录 `uni_embeddings.pickle` | 不建议直接裸用 | 需要；修正输出路径/文件名、避免 CWD 副作用 |
| `COSIE/linkage_construction.py` | `load_protein_gene_mapping` | 无 | protein-to-gene dict | 可复用 | 需要；允许自定义 mapping path |
| `COSIE/linkage_construction.py` | `compute_linkages` | `data_dict, linkage_indicator, num_hvg=3000` | section pair triplets | 可复用 | 需要；校验缺失模态 |
| `COSIE/configure.py` | `get_default_config` | 无 | config dict | 可复用 | 可选；新项目可扩展路径/预处理配置 |
| `COSIE/COSIE_framework.py` | `COSIE_model` | `config, feature_dict` | torch model | 可复用 | 需要；保存路径、缺失模态策略、错误信息 |

## 10. 不能直接复用、需要改造的部分

- 路径写死：notebook 中写死 `/home/hujinlan/cosie/data/SPOTS`、`/home/hujinlan/cosie/Tonsil_10x`、`/home/hujinlan/cosie/UNI`。
- 输出路径绑定：`image_feature_extraction()` 忽略 `path` 参数，硬编码保存到当前工作目录 `uni_embeddings.pickle`。
- 数据集名称写死：notebook 直接引用 SPOTS 和 Tonsil_10x 文件名。
- 只支持示例组织方式：tutorial 只展示 SPOTS/Tonsil 的 `data_dict` 组织，脚本化时需要通用 config。
- UNI 路径绑定：`create_model()` 只接受 local dir，但 tutorial 写死 `/home/hujinlan/cosie/UNI`。
- h5ad 字段名绑定：空间坐标写死使用 `.obsm['spatial']`；10x HE 写死使用 ADT h5ad 的 `.obsm['UNI_feature']`。
- 模态名绑定：ADT 在 COSIE 中必须映射成 `'Protein'`；`'ADT'` 不是源码识别的特殊分支。
- 输出命名绑定：训练输出固定是 `{file_path}/s1_embedding.npy`、`{file_path}/s2_embedding.npy` 等。
- 不支持自动 obs_names 对齐：代码不按 `obs_names` merge/reindex，只做 spatial 数组一致性检查。
- 缺失模态不完全稳健：`None` 支持存在，但 `Linkage_indicator` 引用缺失数据会报错；完全空 section 或无 predictor 的路径有风险。
- notebook 流程无法直接脚本化：多个步骤手动构造 AnnData、手动复制 spatial、手动指定 missing modality 和 linkage。
- Protein-gene 映射路径固定在包目录下：不便于用户替换 mapping。
- `metacell_construction_optimized()` 假设规则空间网格，非规则坐标需要适配或关闭。
- subgraph strong linkage 中 `compute_linkages_per_subgraph()` 对同模态只查找 `'{modality}_harmony'`，若关闭 Harmony 可能不兼容。

## 11. 建议的新项目结构

推荐 `/home/hujinlan/spa_mo_model` 结构：

```text
/home/hujinlan/spa_mo_model/
├── model/
│   ├── __init__.py
│   ├── configure.py
│   ├── data_preprocessing.py
│   ├── image_preprocessing.py
│   ├── multimodal_preprocessing.py
│   └── utils.py
├── UNI/
│   └── pytorch_model.bin
├── data/
├── docs/
│   └── cosie_preprocessing_report.md
└── scripts/
    └── run_preprocessing.py
```

职责建议：

- `model/__init__.py`：暴露新项目稳定 API。
- `model/configure.py`：在 COSIE 默认训练配置基础上增加路径、模态、h5ad 字段、UNI、输出目录等配置。
- `model/data_preprocessing.py`：封装/迁移 COSIE 的 `preprocess_adata()`、`load_data()`、CLR、metacell；保持算法一致。
- `model/image_preprocessing.py`：封装/迁移 HE 原图到 UNI embedding 的逻辑；把 patch size、superpixel size、UNI path、pickle/npy 输出路径配置化。
- `model/multimodal_preprocessing.py`：新加薄 wrapper，负责读取用户 config、构造 `data_dict`、处理 `None`、校验 spatial/obs_names、调用 COSIE 风格函数。
- `model/utils.py`：放 KNN、seed、路径检查、AnnData 字段校验等通用工具。
- `UNI/pytorch_model.bin`：本地 UNI 权重位置，路径由 config 指向。
- `data/`：新项目输入数据与预处理产物根目录，不写死具体数据集名。
- `docs/cosie_preprocessing_report.md`：当前报告。
- `scripts/run_preprocessing.py`：第二阶段再实现的命令行入口，仅调用 wrapper，不内嵌 COSIE 算法细节。

## 12. 第二阶段实现建议

当前不要实现。下一阶段建议按最小迁移做：

- wrapper 优先：先 wrapper `preprocess_adata()`、`load_data()`、`clr_normalize_each_cell()`、`image_feature_extraction()`，不要重写 scanpy 流程。
- 需要复制/迁移的代码：`data_preprocessing.py` 中主函数、`image_preprocessing.py` 中 HE 提取函数、必要的 `utils.py` KNN/seed 工具、`Protein_gene_relationship.csv` 及其加载函数。
- 路径配置化：`file_path`、h5ad 文件、`spatial_key`、`uni_feature_key`、UNI 权重目录、HE 原图路径、mask 路径、输出 embedding 路径、protein-gene mapping 路径都放进 config。
- HE 缺失：如果没有原图或 `UNI_feature`，wrapper 返回 `HE: None`；如果有原图则调用 HE extractor；如果 h5ad 已有 `UNI_feature` 则直接构造 HE AnnData。
- RNA 缺失：对应 section 放 `None`，但 wrapper 需确保 linkage 不引用缺失 RNA。
- Protein/ADT 缺失：用户输入名可以是 ADT，但内部映射成 `'Protein'`；缺失放 `None`。
- Metabolite 缺失：默认 `None`；若未来提供 AnnData，则用 COSIE generic `Metabolite` 分支，不增加额外算法。
- 行为一致：保留 COSIE 的 HVG、normalize/log/scale/PCA、CLR、Harmony 默认值；不要添加新的 filtering、layers 逻辑或 CLR 变体。
- 最小测试命令建议第二阶段创建后再跑，例如：
  - `python scripts/run_preprocessing.py --config configs/spots_preprocessing.yaml --dry-run`
  - `python scripts/run_preprocessing.py --config configs/tonsil_preprocessing.yaml --dry-run`
  - `pytest tests/test_cosie_preprocessing_wrappers.py -q`
  - 对 HE extractor 单测只检查小图/少量坐标的输出 shape 和输出路径，不跑完整 29k patch。

## 13. 当前结论摘要

- COSIE 支持从原始 HE/H&E 图像提特征，函数在 `COSIE/image_preprocessing.py`，使用 UNI ViT-L/16。
- 真实整合 tutorial 中，10x Tonsil 没有现场从原图提 HE，而是直接使用 ADT h5ad 里的 `obsm['UNI_feature']`。
- HE 原图教程输出 `uni_embeddings.pickle`，示例形状 `(29398, 2048)`；之后手动 PCA 到 50 维。
- RNA 输入为 h5ad/AnnData，使用 `.X`，标准流程是 HVG、`normalize_total`、`log1p`、`scale`、PCA；shared RNA 默认再 Harmony。
- Protein/ADT 输入为 h5ad/AnnData，内部模态名是 `'Protein'`，使用 `.X`，流程是 CLR、`scale`、PCA。
- `Protein_gene_relationship.csv` 不参与 Protein 主预处理，只在 RNA-Protein weak linkage 构造中使用。
- COSIE 中未找到明确的 metabolite/metabolomics 专用预处理实现；仅有 generic `Metabolite` 分支，走普通非 Protein omics 流程。
- 多模态 spot/cell 对齐主要依赖输入行顺序和 `.obsm['spatial']` 完全一致，不按 `obs_names` 自动 reindex。
- 缺失模态用 `None` 占位有原生支持，但 linkage 配置和完全缺失 section 仍需 wrapper 保护。
- 新项目最小迁移方案是保留 COSIE 原算法，增加配置化路径、字段适配、ADT 到 Protein 命名适配、缺失模态 `None` 兼容层。
