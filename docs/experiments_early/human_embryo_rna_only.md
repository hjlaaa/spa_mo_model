# SpaMO single-modality mode and HESTA RNA-only workflow

## Model switch

Single-modality support is opt-in. Existing multimodal configurations remain
unchanged.

```python
config["model"]["single_modality_mode"] = {
    "enabled": True,
    "modality": "RNA",  # RNA, Protein, HE, or Metabolite
}
```

With exactly one observed modality, the model:

- runs that modality's MLP encoder and decoder;
- uses identity fusion because the encoder already produces a 128-D latent;
- returns an exactly zero `crossview_loss` and reports
  `contrastive_skipped=true`;
- retains the spatial KNN graph, residual GraphSAGE, reconstruction loss and
  adjacent-section OT-guided attention;
- never fabricates a missing modality.

If the switch is disabled, a single input modality still raises an error. If
`modality` is specified, supplying another modality or multiple modalities also
raises an error, which catches miswired input adapters early.

Run the fast regression checks with:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/validate_single_modality_mode.py
```

## HESTA preprocessing decisions

The seven files contain 1,030,013 spots and RNA only. The adapter reads raw
`layers/counts`, applies light QC, selects shared HVGs using a balanced sample
from every developmental section, performs library-size normalization and
`log1p`, and fits a balanced-sample TruncatedSVD. It does not use Harmony:
developmental stage is biological signal rather than a nuisance batch here.

Raw counts are read directly from HDF5 CSR row blocks because AnnData backed
mode can materialize `layers/counts`. The resulting 50-D arrays are cached as
memory-mapped `.npy` files. The original h5ad files are never changed.

Adjacent sections follow this order:

`CS12-13_E2S1 -> CS14-15_E1S1 -> CS17_E1S1 -> CS18_E1S1 -> CS19_E1S1 -> CS20_E1S1 -> CS23_E1S1`.

Only adjacent-stage candidate-sparse OT is used. Dense OT is not suitable for
these section sizes.

## Recommended execution sequence

Audit only (the default and very fast):

```bash
cd /home/hujinlan/spa_mo_model
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/run_human_embryo_rna_only.py
```

GPU pilot with at most 10,000 spots per section:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/run_human_embryo_rna_only.py \
  --dry_run \
  --max_spots_per_section 10000 \
  --output_dir results/human_embryo_rna_only_pilot
```

Build the full reusable preprocessing cache:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/run_human_embryo_rna_only.py \
  --preprocess_only \
  --output_dir results/human_embryo_rna_only_full
```

Train from that cache (defaults: bidirectional candidate-sparse OT, bf16,
loss-only train forwards, chunking and activation checkpointing):

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/run_human_embryo_rna_only.py \
  --train --epochs 100 --reuse_preprocessed \
  --output_dir results/human_embryo_rna_only_full
```

Analyze the trained embeddings:

```bash
/home/hujinlan/miniconda3/envs/cosie/bin/python \
  scripts/analyze_human_embryo_reference_metrics.py \
  --input-dir results/human_embryo_rna_only_full
```

The analysis uses joint and per-section MiniBatchKMeans with the retained
cluster counts 5, 8, 10, 12, 14 and 16. It exports standardized-embedding
internal metrics, cell-type ARI/NMI/homogeneity/completeness/V-measure,
exact within-section spatial-neighbor agreement, developmental-section
diagnostics, OT summaries, training diagnostics, label arrays and spatial
plots. Stage separation should be interpreted as developmental biology, not
automatically penalized as batch leakage.
