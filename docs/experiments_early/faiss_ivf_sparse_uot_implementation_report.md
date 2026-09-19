# FAISS IVF Sparse UOT Implementation Report

## 1. Scope

This report documents the new scalable OT prior path:

```text
FAISS candidate search
    -> candidate-restricted sparse UOT
    -> top-k OT prior for the existing OT-guided attention
```

The dense UOT path is kept as the default. The new path is enabled only when
`scripts/run_crc_stereocite.py` is run with:

```text
--ot_prior_mode candidate_sparse
```

No processed h5ad files are saved, raw CRC h5ad files are not overwritten, HE is
not fabricated, and ADT remains mapped to the `Protein` modality.

## 2. Files Modified

New files:

- `model/faiss_candidate_search.py`
- `model/sparse_uot.py`

Modified files:

- `model/stage_model.py`
- `scripts/run_crc_stereocite.py`

Files intentionally not modified:

- `model/loss.py`
- `model/data_preprocessing.py`
- `model/multimodal_preprocessing.py`
- `model/model_component.py`
- `scripts/run_mousebrain_v2.py`

## 3. Dense UOT Path

The original dense UOT path remains available and remains the default:

```text
--ot_prior_mode dense
```

The existing MouseBrain script was not changed. Dense UOT initialization and
dynamic update methods are still present in `StageMultiModalModel`.

## 4. New CLI Parameters

`scripts/run_crc_stereocite.py` now supports:

```text
--ot_prior_mode {dense,candidate_sparse}
--candidate_backend {faiss_ivf,faiss_flat,blockwise}
--initial_modality_candidate_k
--candidate_k
--attention_topk
--faiss_nlist
--faiss_nprobe
--faiss_device {auto,cpu,gpu}
--faiss_train_sample_size
--dynamic_candidate_source {fused,final}
--uot_epsilon
--uot_tau_a
--uot_tau_b
--uot_stabilizer
--save_candidate_qc
```

Default behavior remains dense. The previous full-spot stress-test options such
as `--full_spots`, `--force_full_uot`, and dense UOT memory preflight logic were
not reintroduced.

## 5. FAISS Candidate Search

Implemented in `model/faiss_candidate_search.py`.

Function:

```python
build_faiss_candidates(
    source_embeddings,
    target_embeddings,
    candidate_k,
    backend="faiss_ivf",
    nlist=4096,
    nprobe=64,
    faiss_device="auto",
    train_sample_size=100000,
    seed=42,
)
```

Input embeddings are converted to float32 and L2-normalized. Cosine similarity
is implemented as inner product in FAISS.

For IVF-Flat:

```python
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(
    quantizer,
    dim,
    effective_nlist,
    faiss.METRIC_INNER_PRODUCT,
)
```

Small target sets automatically reduce IVF parameters:

```text
effective_nlist = min(requested_nlist, max(1, int(sqrt(n_target)) * 4))
effective_nlist = min(effective_nlist, n_target)
effective_nprobe = min(requested_nprobe, effective_nlist)
```

If the target set is large, FAISS training samples at most
`faiss_train_sample_size` target embeddings.

Supported candidate backends:

- `faiss_ivf`: approximate IVF-Flat candidate search.
- `faiss_flat`: exact FAISS inner-product search.
- `blockwise`: exact blockwise top-k without FAISS, useful when FAISS is not
  available.

FAISS is imported lazily, so environments without FAISS can still import and
compile the project. If FAISS is requested but unavailable, the error is raised
at runtime with a clear message.

## 6. Initial Candidate Logic

Initial sparse OT prior uses preprocessed single-modality features.

For each adjacent section pair:

```text
source section -> target section
```

the implementation:

1. Detects the real shared modalities.
2. Runs candidate search separately per modality.
3. Uses `initial_modality_candidate_k` candidates per source spot per modality.
4. Takes a per-source union of candidate target spots across modalities.
5. Recomputes exact multimodal candidate cost on the union.
6. Truncates to `candidate_k` lowest-cost targets if the union is larger.
7. Runs one candidate-restricted sparse UOT, not one UOT per modality.

For CRC RNA+Protein:

```text
C_ij = 0.5 * (1 - cosine(RNA_i, RNA_j))
     + 0.5 * (1 - cosine(Protein_i, Protein_j))
```

For MouseBrain-style HE+RNA+Metabolite:

```text
C_ij = mean(C_HE, C_RNA, C_Metabolite)
```

Missing modalities are not fabricated and do not enter either candidate search
or cost calculation.

## 7. Dynamic Candidate Update

Dynamic sparse OT update is available through
`StageMultiModalModel.update_candidate_sparse_ot_prior()`.

In the CRC runner:

```text
epoch 0:
    initial candidate union over real modalities

epoch update_interval, 2 * update_interval, ...:
    candidate search from current fused or final embeddings
```

The default is:

```text
--dynamic_candidate_source final
```

With this default, candidate-sparse dynamic OT updates use `final_embeddings`.
For source sections that have a next section, these embeddings are after
OT-guided attention. The explicit `--dynamic_candidate_source fused` option is
kept for ablation/debug runs.

The dynamic update cost is:

```text
C_ij = 1 - cosine(embedding_i, embedding_j)
```

For dynamic updates, initial single-modality candidates are not mixed back in.

## 8. Sparse UOT Formula

Implemented in `model/sparse_uot.py`.

Candidate matrices:

```text
candidate_idx: [N_source, K]
candidate_mask: [N_source, K]
candidate_cost: [N_source, K]
```

are converted to sparse edges:

```text
edge_src: [E]
edge_tgt: [E]
edge_cost: [E]
```

where:

```text
E <= N_source * K
```

Sparse unbalanced Sinkhorn:

```text
K_edge = exp(-edge_cost / epsilon)
a_i = 1 / n_source
b_j = 1 / n_target

rho_a = tau_a / (tau_a + epsilon)
rho_b = tau_b / (tau_b + epsilon)
```

Initialize:

```text
u = ones(n_source)
v = ones(n_target)
```

Iterate:

```text
Kv_i = sum_{edges i->j} K_ij * v_j
u_i = (a_i / (Kv_i + stabilizer)) ** rho_a

KTu_j = sum_{edges i->j} K_ij * u_i
v_j = (b_j / (KTu_j + stabilizer)) ** rho_b
```

Final sparse coupling:

```text
P_edge = u[edge_src] * K_edge * v[edge_tgt]
P_edge = P_edge / (sum(P_edge) + stabilizer)
```

Top-k attention prior per source row:

```text
topk_raw_weight = top attention_topk P_edge values in the row
topk_weight = topk_raw_weight / (sum(topk_raw_weight) + stabilizer)
row_mass = sum all candidate P_edge in the row
raw_topk_mass = sum top-k raw P_edge in the row
topk_coverage = raw_topk_mass / (row_mass + stabilizer)
tail_mass = row_mass - raw_topk_mass
confidence = topk_coverage
```

The implementation uses native `torch.scatter_add_`; no new `torch_scatter`
dependency is introduced.

## 9. Saved OT Top-k and QC

When `--save_ot_prior_topk` is enabled, the output format remains compatible
with existing OT-guided attention QC:

```text
ot_prior_topk/CRC_003_to_CRC_006_topk_idx.npy
ot_prior_topk/CRC_003_to_CRC_006_topk_weight.npy
ot_prior_topk/CRC_003_to_CRC_006_confidence.npy
ot_prior_topk/CRC_003_to_CRC_006_row_mass.npy
ot_prior_topk/CRC_003_to_CRC_006_metadata.json
```

When `--save_candidate_qc` is enabled, additional files are saved:

```text
ot_prior_topk/CRC_003_to_CRC_006_raw_topk_mass.npy
ot_prior_topk/CRC_003_to_CRC_006_topk_coverage.npy
ot_prior_topk/CRC_003_to_CRC_006_tail_mass.npy
ot_prior_topk/CRC_003_to_CRC_006_target_hit_count.npy
```

Dense `P` is not saved.

## 10. FAISS Availability

Checked in the `cosie` environment:

```text
faiss version: 1.9.0
faiss gpu count: 0
```

FAISS CPU is available. FAISS GPU is not available in this environment.

## 11. Validation Commands

Compile command:

```bash
cd /home/hujinlan/spa_mo_model

/home/hujinlan/miniconda3/envs/cosie/bin/python -m py_compile \
    model/configure.py \
    model/data_preprocessing.py \
    model/multimodal_preprocessing.py \
    model/image_preprocessing.py \
    model/utils.py \
    model/loss.py \
    model/model_component.py \
    model/linkage_construction.py \
    model/stage_model.py \
    scripts/run_crc_stereocite.py \
    model/faiss_candidate_search.py \
    model/sparse_uot.py
```

Result:

```text
PASS
```

Required candidate-sparse smoke train:

```bash
cd /home/hujinlan/spa_mo_model

/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_crc_stereocite.py \
    --data_dir /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq \
    --max_spots_per_section 500 \
    --max_shared_genes 3000 \
    --spot_sampling random \
    --train \
    --epochs 2 \
    --lambda_contrast 0.1 \
    --device cpu \
    --seed 42 \
    --save_outputs \
    --save_embeddings \
    --save_ot_prior_topk \
    --save_candidate_qc \
    --ot_prior_mode candidate_sparse \
    --candidate_backend faiss_ivf \
    --initial_modality_candidate_k 50 \
    --candidate_k 100 \
    --attention_topk 10 \
    --faiss_nlist 128 \
    --faiss_nprobe 16 \
    --faiss_device cpu \
    --uot_epsilon 0.05 \
    --uot_tau_a 1.0 \
    --uot_tau_b 1.0 \
    --uot_max_iter 50 \
    --update_interval 20 \
    --output_dir /home/hujinlan/spa_mo_model/results/crc_stereocite/faiss_sparse_uot_smoke_500spot_2ep
```

Result:

```text
PASS
```

This 2-epoch smoke run used `update_interval=20`, so it validated initial
candidate-sparse OT prior construction and training, but did not trigger a
dynamic OT update.

An additional tiny dynamic-update smoke was run with `update_interval=1`:

```text
output_dir = /home/hujinlan/spa_mo_model/results/crc_stereocite/faiss_sparse_uot_dynamic_update_smoke_100spot_2ep
```

Result:

```text
PASS
```

It recorded:

```text
ot_updates = [1, 2]
final ot_prior modalities_used = ["fused_embedding"]
candidate_source = fused
```

## 12. Smoke Test Results

Required 500 spot candidate-sparse run:

```text
final_embeddings_CRC_003.npy shape = [500, 128]
final_embeddings_CRC_006.npy shape = [500, 128]
total_loss_finite = true
ot_prior_mode = candidate_sparse
ot_prior modalities_used = ["RNA", "Protein"]
candidate_source = initial_modalities
```

Final losses after the smoke run:

```text
total_loss = -203.42971801757812
crossview_loss = -2190.042236328125
reconstruction_loss = 15.574503898620605
```

Saved OT prior arrays:

```text
topk_idx shape = [500, 10], dtype int64
topk_weight shape = [500, 10], dtype float32
confidence shape = [500], dtype float32
row_mass shape = [500], dtype float32
raw_topk_mass shape = [500], dtype float32
topk_coverage shape = [500], dtype float32
tail_mass shape = [500], dtype float32
target_hit_count shape = [500], dtype int64
```

Candidate metadata for the 500 spot run:

```text
backend = faiss_ivf
requested_nlist = 128
effective_nlist = 88
requested_nprobe = 16
effective_nprobe = 16
faiss_device_used = cpu
candidate_edge_count = 47207
candidate_target_unique_coverage = 1.0
```

No processed h5ad files were written in the smoke output directory.

## 13. Formal 10k / 200 Epoch Command

Do not run this from Codex unless explicitly requested. The recommended command
for the next controlled CRC run is:

```bash
cd /home/hujinlan/spa_mo_model

/home/hujinlan/miniconda3/envs/cosie/bin/python scripts/run_crc_stereocite.py \
    --data_dir /home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq \
    --max_spots_per_section 10000 \
    --max_shared_genes 10000 \
    --spot_sampling random \
    --train \
    --epochs 200 \
    --lambda_contrast 0.1 \
    --device cuda \
    --seed 42 \
    --save_outputs \
    --save_embeddings \
    --save_ot_prior_topk \
    --save_candidate_qc \
    --ot_prior_mode candidate_sparse \
    --candidate_backend faiss_ivf \
    --initial_modality_candidate_k 100 \
    --candidate_k 200 \
    --attention_topk 10 \
    --faiss_nlist 4096 \
    --faiss_nprobe 64 \
    --faiss_device auto \
    --uot_epsilon 0.05 \
    --uot_tau_a 1.0 \
    --uot_tau_b 1.0 \
    --uot_max_iter 100 \
    --update_interval 20 \
    --output_dir /home/hujinlan/spa_mo_model/results/crc_stereocite/train_random10k_faiss_ivf_sparse_uot_200ep_lc0.1
```

## 14. Final Conclusion

```text
FAISS_IVF_SPARSE_UOT_IMPLEMENTATION: PASS
```

The new candidate-sparse OT prior path is implemented and smoke-tested. The
dense UOT path remains the default. MouseBrain scripts, loss functions,
preprocessing modules, GraphSAGE, decoder, and OT-guided attention formulas were
not changed.
