# spa_mo_model: standardized_embedding

- spots: 5336; dimensions: 128
- exact KMeans: seed=0, n_init=20, max_iter=300
- joint/independent metrics: K=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- retained clustering directories and full labels: K=[5, 8, 10, 12]
- metrics for K=[2, 3, 4, 6, 7, 9, 11] remain in CSV, but their historical labels_path targets were removed
- spatial figures: K=[5, 8, 10, 12]; point size=10
- Cluster ASW, CH and DBI: full samples in each scope
- Label ASW: not applicable because no ground-truth label exists
- spatial continuity: exact 6-NN per section
- retained-K metrics and every figure reuse saved full label CSVs
