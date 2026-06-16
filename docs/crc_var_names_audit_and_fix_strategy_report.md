# CRC RNA var_names Audit and Fix Strategy Report

## 1. Scope

This audit checked RNA `var_names` duplication and repair options for:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq
/home/hujinlan/spa_mo_model/data/dataset_MouseBrain
```

Files and code inspected:

- `CRC_003_bin20/adata_RNA.h5ad`
- `CRC_006_bin20/adata_RNA.h5ad`
- `CRC_003_bin20/adata_ADT.h5ad`
- `CRC_006_bin20/adata_ADT.h5ad`
- MouseBrain SectionA/B/C `adata_RNA.h5ad`
- `scripts/run_mousebrain_v2.py`
- `model/data_preprocessing.py`
- `model/multimodal_preprocessing.py`
- `data/configs/mousebrain_preprocess_train.json`

This was a read-only audit of dataset metadata and AnnData `var` fields. No
formal h5ad files were modified, no model training was run, and no full CRC UOT
matrix was constructed.

## 2. CRC_003 RNA var_names audit

File:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq/CRC_003_bin20/adata_RNA.h5ad
```

Summary:

| Item | Value |
|---|---:|
| RNA shape | `(166279, 28592)` |
| `adata.var_names.is_unique` | `False` |
| total RNA columns | `28592` |
| unique gene symbols / unique `var_names` | `28573` |
| duplicated gene symbol groups | `19` |
| extra duplicated columns, `sum(count - 1)` | `19` |
| columns belonging to duplicated groups | `38` |

`adata.var` columns:

```text
real_gene_name
```

Candidate ID columns checked:

```text
gene_id, gene_ids, gene_name, gene_symbol, real_gene_name,
real_gene_id, ensembl_id
```

Only `real_gene_name` exists. It has no missing/empty values, but it is not
unique and it is exactly the same as current `var_names`. Therefore it is a gene
symbol/name column, not a usable unique gene ID column.

Top duplicated gene symbols:

| gene symbol | count |
|---|---:|
| ABCF2 | 2 |
| ATXN7 | 2 |
| CCDC39 | 2 |
| COG8 | 2 |
| CYB561D2 | 2 |
| DIABLO | 2 |
| EMG1 | 2 |
| HSPA14 | 2 |
| IGF2 | 2 |
| LINC01238 | 2 |
| MATR3 | 2 |
| PDE11A | 2 |
| PINX1 | 2 |
| POLR2J3 | 2 |
| PRSS50 | 2 |
| RGS5 | 2 |
| SCO2 | 2 |
| SOD2 | 2 |
| TBCE | 2 |

Required known duplicated genes:

| gene | count | duplicated |
|---|---:|---|
| MATR3 | 2 | yes |
| ABCF2 | 2 | yes |
| SOD2 | 2 | yes |
| CYB561D2 | 2 | yes |
| EMG1 | 2 | yes |
| PDE11A | 2 | yes |
| SCO2 | 2 | yes |
| RGS5 | 2 | yes |
| IGF2 | 2 | yes |
| TBCE | 2 | yes |

Conclusion for CRC_003 RNA: current `var_names` are not safe for formal
cross-sample preprocessing because duplicate gene symbols represent multiple
matrix columns with the same name.

## 3. CRC_006 RNA var_names audit

File:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq/CRC_006_bin20/adata_RNA.h5ad
```

Summary:

| Item | Value |
|---|---:|
| RNA shape | `(446095, 31809)` |
| `adata.var_names.is_unique` | `False` |
| total RNA columns | `31809` |
| unique gene symbols / unique `var_names` | `31789` |
| duplicated gene symbol groups | `20` |
| extra duplicated columns, `sum(count - 1)` | `20` |
| columns belonging to duplicated groups | `40` |

`adata.var` columns:

```text
real_gene_name
```

Only `real_gene_name` exists among the candidate ID columns. It has no
missing/empty values, but it is not unique and exactly equals current
`var_names`. It is not suitable as a unique formal `var_names` column.

Top 20 duplicated gene symbols:

| gene symbol | count |
|---|---:|
| ABCF2 | 2 |
| ATXN7 | 2 |
| CCDC39 | 2 |
| COG8 | 2 |
| CYB561D2 | 2 |
| DIABLO | 2 |
| EMG1 | 2 |
| HSPA14 | 2 |
| IGF2 | 2 |
| LINC01238 | 2 |
| MATR3 | 2 |
| PDE11A | 2 |
| PINX1 | 2 |
| POLR2J3 | 2 |
| PRSS50 | 2 |
| RGS5 | 2 |
| SCO2 | 2 |
| SOD2 | 2 |
| TBCE | 2 |
| TMSB15B | 2 |

Required known duplicated genes:

| gene | count | duplicated |
|---|---:|---|
| MATR3 | 2 | yes |
| ABCF2 | 2 | yes |
| SOD2 | 2 | yes |
| CYB561D2 | 2 | yes |
| EMG1 | 2 | yes |
| PDE11A | 2 | yes |
| SCO2 | 2 | yes |
| RGS5 | 2 | yes |
| IGF2 | 2 | yes |
| TBCE | 2 | yes |

Conclusion for CRC_006 RNA: current `var_names` are not safe for formal
cross-sample preprocessing. CRC_006 has the same duplicated genes as CRC_003
plus `TMSB15B`.

## 4. CRC RNA overlap audit

Current gene-symbol overlap:

| Item | Value |
|---|---:|
| CRC_003 unique gene symbols | `28573` |
| CRC_006 unique gene symbols | `31789` |
| unique symbol intersection | `28411` |
| unique symbol union | `31951` |
| exact original var order identical | `False` |
| first position where var order differs | `130` |

The direct current `var_names` intersection by unique strings is `28411`.
After duplicate-symbol aggregation, the theoretical shared gene-symbol set is
also `28411`, but with one important difference: each shared gene symbol would
then correspond to exactly one matrix column per sample.

Duplicate-symbol consistency:

- CRC_003 duplicated symbols: `19`
- CRC_006 duplicated symbols: `20`
- common duplicated symbols: `19`
- duplicated only in CRC_003: none
- duplicated only in CRC_006: `TMSB15B`

Interpretation: duplicates are mostly consistent across samples, but not
perfectly identical. This is one reason plain suffixing with
`var_names_make_unique()` is risky for formal cross-sample work.

## 5. CRC ADT / Protein marker audit

Files:

```text
CRC_003_bin20/adata_ADT.h5ad
CRC_006_bin20/adata_ADT.h5ad
```

Summary:

| Sample | ADT shape | `var_names.is_unique` | marker count |
|---|---:|---|---:|
| CRC_003 | `(166279, 163)` | `True` | `163` |
| CRC_006 | `(446095, 163)` | `True` | `163` |

Both ADT files have `var["real_gene_name"]`, and marker names are already
unique.

Cross-sample marker comparison:

| Item | Value |
|---|---|
| same marker set | `True` |
| same marker order | `True` |
| marker intersection | `163` |

First 20 markers:

```text
CD101, CD103, CD107a, CD109, CD112, CD115, CD116, CD119,
CD11a, CD11b, CD11c, CD122, CD123, CD127, CD13, CD131,
CD134, CD137, CD141, CD142
```

Conclusion: CRC ADT should be used as the `Protein` modality. It should not be
treated as `Metabolite`.

## 6. MouseBrain RNA var_names handling

MouseBrain config:

```json
"rna_gene_id_key": "gene_ids"
```

Code path:

- `scripts/run_mousebrain_v2.py` reads the config key and calls
  `build_mousebrain_section(...)`.
- `model/multimodal_preprocessing.py::build_mousebrain_section()` requires
  `rna.var["gene_ids"]`, checks missing values, checks uniqueness, saves
  original gene symbols, then switches `var_names` to gene IDs:

```python
gene_ids = rna.var[rna_gene_id_key].astype(str)
if not gene_ids.is_unique:
    raise ValueError(...)
rna.var["gene_symbol"] = rna.var_names.astype(str)
rna.var_names = gene_ids
rna.var_names_make_unique()
```

`model/data_preprocessing.py::preprocess_adata()` also calls
`adata_obj.var_names_make_unique()` on its copy before HVG/PCA. In the
MouseBrain path this is a defensive no-op for RNA because `gene_ids` are already
unique after `build_mousebrain_section()`.

MouseBrain raw RNA audit:

| Section | raw shape | raw `var_names.is_unique` | raw unique symbols | duplicate symbol groups | extra duplicate columns |
|---|---:|---|---:|---:|---:|
| SectionA | `(2384, 32285)` | `False` | `32245` | `38` | `40` |
| SectionB | `(2820, 32285)` | `False` | `32245` | `38` | `40` |
| SectionC | `(2662, 32285)` | `False` | `32245` | `38` | `40` |

MouseBrain raw RNA has duplicated gene symbols, including genes with three
copies such as `Ccl27a` and `Il11ra2`.

MouseBrain `gene_ids` audit:

| Section | `gene_ids` present | missing/empty | `gene_ids.is_unique` | `gene_ids` count | equals raw symbols |
|---|---|---:|---|---:|---|
| SectionA | yes | `0` | `True` | `32285` | no |
| SectionB | yes | `0` | `True` | `32285` | no |
| SectionC | yes | `0` | `True` | `32285` | no |

After the MouseBrain adapter:

| Section | adapted `var_names.is_unique` | adapted `var_names` name | `var["gene_symbol"]` present |
|---|---|---|---|
| SectionA | `True` | `gene_ids` | yes |
| SectionB | `True` | `gene_ids` | yes |
| SectionC | `True` | `gene_ids` | yes |

MouseBrain overlap:

| Item | Value |
|---|---:|
| raw symbol intersection across 3 sections | `32245` |
| gene ID intersection across 3 sections | `32285` |

Conclusion: MouseBrain solves duplicate RNA gene symbols by using a true unique
ID column, `var["gene_ids"]`, as formal `var_names`, while preserving original
gene symbols in `var["gene_symbol"]`.

## 7. CRC vs MouseBrain differences

Key difference:

- MouseBrain RNA has a unique `gene_ids` column.
- CRC RNA h5ad files do not have `gene_ids`, `gene_id`, `ensembl_id`,
  `real_gene_id`, or any other audited unique ID column.

CRC only has:

```text
var["real_gene_name"]
```

That column is identical to the current duplicated gene-symbol `var_names`.
Therefore the MouseBrain strategy cannot be directly applied to current CRC
h5ad files.

If a unique ID can be recovered from the original GEF files or external
annotation, then a MouseBrain-like ID strategy would become preferable. Based
only on the current CRC h5ad files, that information is not present.

## 8. Fix strategy comparison

### Strategy A: use a unique gene ID column

Proposed pattern:

```python
adata.var["gene_symbol"] = adata.var_names.astype(str)
adata.var_names = adata.var["unique_id_column"].astype(str)
adata.var_names_make_unique()
```

Pros:

- Best formal solution when a real stable gene ID exists.
- Supports stable cross-sample gene intersection by biological ID.
- Keeps gene symbols available for marker interpretation.
- Mirrors the current MouseBrain approach.

Cons for current CRC h5ad:

- No suitable unique ID column is present.
- `real_gene_name` is not unique and equals current `var_names`.
- Cannot be applied safely unless unique IDs are recovered from upstream GEF or
  external annotation.

CRC suitability:

- Suitable only if a real unique ID column can be recovered.
- Not applicable to the current h5ad files as they stand.

### Strategy B: aggregate duplicate gene-symbol columns by counts sum

Proposed behavior:

```text
For each duplicated gene symbol, sum all columns with that symbol into one
column. Non-duplicated symbols pass through unchanged.
```

Pros:

- Appropriate for raw count matrices because counts from duplicate entries can
  be summed while preserving total counts.
- Produces one column per gene symbol.
- Makes HVG, PCA, Harmony, gene intersection, and marker interpretation
  deterministic.
- Avoids unstable suffix names such as `MATR3-1`.
- Works with the current CRC h5ad files because gene symbols are available.

Required checks:

- Total RNA counts before and after aggregation should be equal per sample.
- `adata.var_names.is_unique` should be `True` after aggregation.
- `adata.var["gene_symbol"]` should preserve the final unique symbol.
- A duplicate summary table should record which original columns were merged.

Sparse CSR implementation notes:

- Do not densify full CRC matrices.
- Use sparse matrix operations.
- A practical approach is to construct a sparse column-grouping matrix
  `G` with shape `[n_original_vars, n_unique_symbols]`, where each original
  column maps to its gene-symbol group, then compute:

```text
X_aggregated = X_original @ G
```

- Keep output as CSR/CSC sparse matrix.
- Verify:

```text
X_original.sum() == X_aggregated.sum()
```

within exact integer arithmetic or a small numerical tolerance if conversion is
needed.

CRC suitability:

- Recommended for the current CRC h5ad files if no true unique ID can be
  recovered.

### Strategy C: use `adata.var_names_make_unique()` directly

Pros:

- Fast and convenient.
- Acceptable for tiny smoke tests when no biological interpretation or formal
  cross-sample comparison is needed.
- Avoids AnnData errors caused by duplicate index values.

Cons:

- It does not merge duplicate biological genes.
- It creates artificial names such as `MATR3`, `MATR3-1`, etc.
- Suffix assignment depends on original column order and duplicate pattern.
- CRC_003 and CRC_006 do not have identical gene lists/order and have slightly
  different duplicate sets; for example `TMSB15B` is duplicated only in
  CRC_006.
- Cross-sample intersection can treat suffix-generated names inconsistently.
- Marker interpretation becomes fragile because `MATR3-1` is not a real gene
  symbol or stable gene ID.

CRC suitability:

- Acceptable only for smoke tests.
- Not recommended for formal CRC preprocessing or model training.

## 9. Final recommendation

For the current CRC h5ad files, the recommended formal strategy is:

```text
Strategy B: aggregate duplicated RNA gene-symbol columns by counts sum.
```

Reason:

- Current CRC RNA h5ad files do not contain a usable unique gene ID column.
- `real_gene_name` is duplicated and equals current `var_names`.
- Direct suffixing with `var_names_make_unique()` would be unstable and
  biologically awkward for cross-sample work.
- Count-sum aggregation preserves total counts and gives one stable feature per
  gene symbol.

Important qualification:

```text
If unique gene IDs can be recovered from the original GEF files or another
trusted annotation source, use Strategy A instead.
```

But based on the actual h5ad files audited here, Strategy A cannot be applied
today.

## 10. Recommended CRC preprocessing plan

Add one small CRC-specific preprocessing script rather than modifying generic
model code. Suggested name:

```text
scripts/preprocess_crc_stereocite.py
```

Recommended output directory:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq_processed
```

Do not overwrite:

```text
/home/hujinlan/spa_mo_model/data/CRC_Stereo-CITE-seq
```

Recommended outputs:

```text
CRC_Stereo-CITE-seq_processed/
├── CRC_003_bin20/
│   ├── adata_RNA_gene_symbol_aggregated.h5ad
│   ├── adata_ADT.h5ad
│   ├── duplicate_gene_summary.csv
│   └── gene_symbol_aggregation_map.csv
├── CRC_006_bin20/
│   ├── adata_RNA_gene_symbol_aggregated.h5ad
│   ├── adata_ADT.h5ad
│   ├── duplicate_gene_summary.csv
│   └── gene_symbol_aggregation_map.csv
├── shared_gene_symbols_CRC_003_CRC_006.txt
├── adt_marker_list.txt
└── preprocessing_summary.json
```

Recommended metadata to preserve:

- original gene symbol for every original RNA column;
- final unique RNA `var_names`;
- duplicate gene summary;
- original-to-aggregated gene mapping;
- shared gene intersection list;
- ADT marker list;
- total counts before and after RNA aggregation;
- original file paths and processing timestamp.

After RNA aggregation:

- set `adata.var_names` to unique gene symbols;
- set `adata.var["gene_symbol"] = adata.var_names.astype(str)`;
- optionally set `adata.var["aggregation_n_original_columns"]`;
- keep `adata.var_names.is_unique == True`;
- keep `adata.X` sparse;
- verify total counts are preserved.

## 11. Operations that must wait until after var_names repair

Run these only after CRC RNA `var_names` are repaired:

- cross-sample gene intersection;
- HVG selection;
- PCA;
- Harmony;
- model training;
- marker analysis;
- any comparison of RNA features across CRC_003 and CRC_006.

ADT/Protein marker intersection does not need this RNA repair because ADT marker
names are already unique and aligned.

## 12. Unresolved / unable to determine

Unresolved from current h5ad files:

- Whether the original GEF files contain Ensembl IDs or other stable gene IDs.
- Whether an external annotation table can map every CRC `real_gene_name` to a
  unique stable ID without ambiguity.

No formal data repair was performed in this task.

## 13. Final conclusion

The CRC RNA h5ad files have real duplicate gene-symbol `var_names` and do not
contain a usable unique gene ID column. MouseBrain previously avoided the same
class of problem by switching RNA `var_names` to unique `var["gene_ids"]` while
preserving gene symbols. CRC cannot directly copy that exact solution from the
current h5ad files.

Recommended formal CRC fix:

```text
Aggregate duplicated RNA gene-symbol columns by counts sum, save repaired data
to a new processed directory, and only then run gene intersection, HVG, PCA,
Harmony, and model training.
```

Do not use `var_names_make_unique()` alone for formal CRC experiments.
