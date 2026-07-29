# spa_mo_model: standardized_embedding

- spots: 7866; dimensions: 128
- KMeans: sklearn exact, seed=0, n_init=20, max_iter=300
- joint metric K: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; independent metric K: [5, 6, 8, 10]
- retained clustering directories/labels/plots K: [5, 6, 8, 10]
- deleted joint clustering directory K: [2, 3, 4, 7, 9, 11, 12]; metric CSV rows remain, so their historical labels_path values no longer resolve
- external labels: five biological annotations plus group, where group is the aligned s1/s2/s3 section-source diagnostic
- internal and label ASW: full valid samples in KMeans metric space
- spatial continuity: exact 6-NN per section
- for retained K, figures and metrics reuse the saved label CSV files
