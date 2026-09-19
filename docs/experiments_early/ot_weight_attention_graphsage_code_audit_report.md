# OT Weight, OT-Guided Attention, and GraphSAGE Code Audit Report

## 1. Audit scope

This report directly inspected the current implementation in:

- `model/linkage_construction.py`
- `model/sparse_uot.py`
- `model/model_component.py`
- `model/stage_model.py`
- `model/utils.py`
- `model/configure.py`
- `scripts/run_crc_stereocite.py`

The goal was to answer three implementation questions:

1. Whether OT weights are actually costs, and whether the project has used them in the wrong direction.
2. How OT-guided attention uses OT weights, and whether there is self-attention.
3. How GraphSAGE is implemented, and whether its core is MLP or attention.

No model code was changed for this audit.

## 2. Are OT weights costs or similarity-like coupling mass?

### 2.1 Dense UOT path

The dense UOT path computes a cosine distance cost first:

```python
cost = 1.0 - x_norm @ y_norm.t()
cost = cost.clamp(clip_min, clip_max)
```

This is implemented in `model/utils.py::cosine_cost_matrix()`. Here, lower cost means higher cosine similarity.

Then `model/linkage_construction.py::unbalanced_sinkhorn()` converts the cost to an entropic kernel:

```python
kernel = torch.exp(-cost / epsilon).clamp_min(delta)
```

This means lower cost produces larger kernel value. The final dense OT coupling is:

```python
coupling = u[:, None] * kernel * v[None, :]
```

So the returned OT matrix `P` is not the cost matrix. It is a transport coupling / transport mass matrix. Larger `P_ij` means the UOT solver assigned more transport mass to source spot `i` and target spot `j`, after considering cost, entropy, and unbalanced mass constraints.

The top-k prior is then selected by coupling mass:

```python
values, indices = torch.topk(P, k=effective_topk, dim=1)
topk_weight = values / (row_topk_mass + delta)
```

Therefore, in the current dense implementation, `topk_weight` is a row-normalized top-k transport mass, not a transport cost. Treating larger `topk_weight` as a stronger matching prior is directionally correct.

### 2.2 Candidate-sparse UOT path

The candidate-sparse path has the same sign logic.

Candidates are found by cosine similarity, then exact candidate cost is recomputed as:

```python
cost = (1.0 - sim).clamp(0.0, 2.0)
```

Sparse Sinkhorn then computes:

```python
kernel_edge = torch.exp(-edge_cost / epsilon)
p_edge = u[edge_src] * kernel_edge * v[edge_tgt]
p_edge = p_edge / (p_edge.sum() + stabilizer)
```

The top-k attention prior is selected from `p_edge` mass:

```python
topk_raw, topk_pos = torch.topk(p_matrix, k=effective_topk, dim=1)
topk_weight = topk_raw / (raw_topk_mass[:, None] + stabilizer)
```

So candidate-sparse `topk_weight` is also coupling mass, not cost.

### 2.3 Conclusion on possible reversal

The implementation is not reversed. The code does not feed raw transport cost as "larger is more similar". It feeds row-normalized transport coupling mass. Because the solver forms `exp(-cost / epsilon)`, low-cost pairs become higher-probability / higher-mass candidates, all else equal.

Important nuance: OT `topk_weight` is not pure biological similarity or pure cosine similarity. It is a transport prior shaped by cosine cost, entropy, marginal relaxation, candidate restrictions, and row top-k normalization.

## 3. How OT-guided attention uses OT weights

`model/model_component.py::OTGuidedAttention.forward()` implements single-direction top-k cross-stage attention.

For a source section and the next target section:

```python
q = W_Q(source_h)
candidate_h = target_h[topk_idx]
k = W_K(candidate_h)
v = W_V(candidate_h)
```

The attention score is:

```python
scores = (q.unsqueeze(1) * k).sum(dim=-1) / sqrt(d_attn)
scores = scores + beta * log(topk_weight + delta)
alpha = softmax(scores, dim=1)
```

Therefore, OT weight is not used as the attention distribution directly. It is used as an additive log-prior / log-bias on top of the learned Q-K content score.

The message is:

```python
message = sum_j alpha_ij * v_ij
message_bar = W_O(message)
```

Then a scalar gate is computed from content features:

```python
gate = sigmoid(MLP([source_h, message, source_h - message, source_h * message]))
```

The final update is:

```python
update = Dropout(confidence * gate * message_bar)
h_tilde = LayerNorm(source_h + update)
```

So the OT prior enters in two places:

1. `topk_weight` biases the attention logits through `beta * log(topk_weight)`.
2. `confidence` scales the residual update if `use_confidence=True`.

The OT weight is not concatenated into the gate input. The gate input concatenates source and attention message features, not the OT weight itself.

## 4. Is there self-attention?

No general self-attention module is implemented.

The search found no `MultiheadAttention`, `Transformer`, or self-attention block in the model. The only attention module is `OTGuidedAttention`, and it is cross-section:

```python
source_h = graphsage_embeddings[source_section]
target_h = graphsage_embeddings[target_section]
target candidates = target_h[topk_idx]
```

In `StageMultiModalModel.forward()`, attention is applied only for adjacent pairs:

```python
for source_section, target_section in zip(resolved_order[:-1], resolved_order[1:]):
    final_embeddings[source_section] = self.ot_attention(...)
```

The last section has no next section to attend to:

```python
final_embeddings[last_section] = graphsage_embeddings[last_section]
```

Thus for two CRC sections, `CRC_003` can attend to `CRC_006`, while `CRC_006` remains the GraphSAGE output. This is by design of the current single-direction attention implementation.

## 5. How dynamic OT update chooses embeddings

Dense mode in `scripts/run_crc_stereocite.py` updates OT from:

```python
model.update_ot_prior(eval_outputs["final_embeddings"], section_order=section_order)
```

Candidate-sparse mode uses:

```python
embeddings = eval_outputs["fused_embeddings"] if args.dynamic_candidate_source == "fused" else eval_outputs["final_embeddings"]
```

The current CRC script default is:

```python
--dynamic_candidate_source final
```

So candidate-sparse mode now defaults to attention-after-final embeddings for dynamic candidate search and sparse UOT. The caveat above still applies: the last section's `final_embeddings` equal its GraphSAGE embeddings because there is no forward target section after it.

Both dense and sparse OT update functions are decorated with `@torch.no_grad()` and/or detach tensors internally, so OT prior construction does not backpropagate through the matching computation.

## 6. How GraphSAGE is implemented

The spatial graph is built in `model/utils.py::compute_spatial_knn_graph_with_weights()`:

- KNN uses spatial coordinates.
- Default `k=5`.
- Self-loops are included.
- Reverse edges are added when `undirected=True`.
- Raw non-self weights are:

```python
w_raw = exp(-dist^2 / (sigma_source^2 + delta))
```

- Self-loop raw weight is `1.0`.
- Weights are normalized by source row:

```python
w_ij = w_raw_ij / sum_j w_raw_ij
```

GraphSAGE itself is implemented in `model/model_component.py::WeightedResidualGraphSAGE`.

Aggregation:

```python
neigh_i = sum_{edges i->j} edge_weight_ij * x_j
```

The current code performs the same `index_add_` aggregation in edge batches to reduce memory:

```python
msg_b = x[target_b] * weight_b.unsqueeze(-1)
neigh.index_add_(0, source_b, msg_b)
```

Then it applies two learned linear maps:

```python
out = GELU(W_self x + W_neigh neigh + bias)
out = Dropout(out)
out = x + out
out = LayerNorm(out)
```

This is not attention. There are no learned attention coefficients over neighbors, no Q/K/V, and no softmax in GraphSAGE. It is weighted mean aggregation using fixed spatial edge weights, followed by learned linear projections, GELU, dropout, residual, and LayerNorm.

Calling it "MLP" is also not quite exact: the GraphSAGE core is one weighted aggregation plus two linear transformations. It is closer to a one-layer weighted residual GraphSAGE block than a multi-layer MLP or attention block.

## 7. Final answers to the three questions

### Question 1: Are OT weights used backward?

No. The implementation computes cost first, but the saved and used `topk_weight` is not cost. It is normalized transport coupling mass. Larger `topk_weight` means the OT solver assigned more mass to that source-target edge. Using larger weight as stronger prior is consistent with the code.

### Question 2: Is OT-guided attention directly equal to OT weights?

No. OT weights are added as a log-prior to learned Q-K attention scores:

```text
score_ij = q_i^T k_j / sqrt(d) + beta * log(w_ij)
```

Then softmax is applied. So OT prior biases attention, but the learned content score still matters. Confidence also scales the residual update. There is no general self-attention; only adjacent cross-section OT-guided attention is implemented.

### Question 3: Is GraphSAGE MLP or attention?

GraphSAGE is not attention. It uses fixed spatial edge weights for weighted mean aggregation, then applies learned linear transforms plus GELU, dropout, residual, and LayerNorm. It is a one-layer weighted residual GraphSAGE block.

