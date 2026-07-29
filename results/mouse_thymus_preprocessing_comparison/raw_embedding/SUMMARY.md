# spa_mo_model: raw_embedding

- spots: 17824; dimensions: 128
- exact KMeans: seed=0, n_init=20, max_iter=300
- joint/independent metrics: K=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- retained clustering directories and full labels: K=[5, 8, 10, 12]
- metrics for K=[2, 3, 4, 6, 7, 9, 11] remain in CSV, but their historical labels_path targets were removed
- spatial figures: K=[5, 8, 10, 12]; point sizes={'Mouse_Thymus1': 8.0, 'Mouse_Thymus2': 5.0, 'Mouse_Thymus3': 5.0, 'Mouse_Thymus4': 5.0}
- Cluster ASW: fixed section-stratified 10000-spot sample; independent quotas={'Mouse_Thymus1': 2635, 'Mouse_Thymus2': 2386, 'Mouse_Thymus3': 2607, 'Mouse_Thymus4': 2372}
- CH and DBI: full samples in each scope
- Label ASW: not applicable because no ground-truth label exists
- spatial continuity: exact 6-NN per section
- retained-K metrics and every figure reuse saved full label CSVs
