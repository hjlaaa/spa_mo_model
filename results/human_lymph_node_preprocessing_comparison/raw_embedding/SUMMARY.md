# spa_mo_model: raw_embedding

- spots: 6843; dimensions: 128
- KMeans: sklearn exact, seed=0, n_init=20, max_iter=300
- joint metric K: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; independent metric K: [5, 8, 10, 12]
- retained clustering directories/labels/plots K: [5, 8, 10, 12]
- deleted joint clustering directory K: [2, 3, 4, 6, 7, 9, 11]; metric CSV rows remain, so their historical labels_path values no longer resolve
- metrics: full sample in the same space used by KMeans
- spatial continuity: exact 6-NN within each section
- for retained K, labels are shared by figures and metrics
