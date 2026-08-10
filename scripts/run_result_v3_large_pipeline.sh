#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/hujinlan/spa_mo_model
PYTHON_BIN=/home/hujinlan/miniconda3/envs/cosie/bin/python
LOG_DIR="$PROJECT_ROOT/result_v3/logs"
CRC_OUT="$PROJECT_ROOT/result_v3/crc_stereocite/bidirectional_sparse_uot_fixed_lc0.1_seed42"
SPATCH_OUT="$PROJECT_ROOT/result_v3/spatch/bidirectional_sparse_uot_fixed_lc0.1_seed42"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 97

printf 'running\n' > "$LOG_DIR/large_pipeline.status"
printf '%s\n' "$(date --iso-8601=seconds) CRC training started"

"$PYTHON_BIN" scripts/run_crc_stereocite.py \
  --train \
  --epochs 200 \
  --lambda_contrast 0.1 \
  --device cuda \
  --ot_prior_mode candidate_sparse \
  --bidirectional_ot_attention \
  --candidate_backend faiss_ivf \
  --initial_modality_candidate_k 100 \
  --candidate_k 200 \
  --attention_topk 10 \
  --faiss_nlist 4096 \
  --faiss_nprobe 64 \
  --faiss_device auto \
  --faiss_train_sample_size 150000 \
  --faiss_query_batch_size 2048 \
  --dynamic_candidate_source final \
  --uot_epsilon 0.05 \
  --uot_tau_a 1.0 \
  --uot_tau_b 1.0 \
  --uot_max_iter 100 \
  --update_interval 20 \
  --spatial_knn_k 5 \
  --graphsage_edge_batch_size 100000 \
  --training_loss_only \
  --decoder_chunk_size 50000 \
  --ot_attention_source_chunk_size 50000 \
  --checkpoint_ot_attention \
  --checkpoint_encoder_fusion \
  --checkpoint_decoder_chunks \
  --checkpoint_graph_encoder \
  --amp_dtype bf16 \
  --cache_spatial_graphs \
  --save_candidate_qc \
  --save_outputs \
  --save_embeddings \
  --save_ot_prior_topk \
  --log_cuda_memory \
  --log_cuda_memory_detail \
  --seed 42 \
  --output_dir "$CRC_OUT"
crc_rc=$?
printf '%s\n' "$crc_rc" > "$LOG_DIR/crc_stereocite_seed42.exit"
if (( crc_rc != 0 )); then
  printf 'failed_crc:%s\n' "$crc_rc" > "$LOG_DIR/large_pipeline.status"
  exit "$crc_rc"
fi

printf '%s\n' "$(date --iso-8601=seconds) spatch training started"
"$PYTHON_BIN" scripts/run_spatch.py \
  --output_dir "$SPATCH_OUT" \
  --epochs 200 \
  --seed 42 \
  --device cuda \
  --lambda_contrast 0.1 \
  --update_interval 20 \
  --uot_max_iter 100 \
  --candidate_backend faiss_ivf \
  --initial_modality_candidate_k 100 \
  --candidate_k 200 \
  --attention_topk 10 \
  --faiss_nlist 4096 \
  --faiss_nprobe 64 \
  --faiss_device gpu \
  --faiss_train_sample_size 100000 \
  --faiss_query_batch_size 2048 \
  --dynamic_candidate_source final \
  --uot_epsilon 0.05 \
  --uot_tau_a 1.0 \
  --uot_tau_b 1.0 \
  --spatial_knn_k 10 \
  --graphsage_edge_batch_size 100000 \
  --decoder_chunk_size 8192 \
  --ot_attention_source_chunk_size 2048 \
  --amp_dtype bf16
spatch_rc=$?
printf '%s\n' "$spatch_rc" > "$LOG_DIR/spatch_seed42.exit"
if (( spatch_rc != 0 )); then
  printf 'failed_spatch:%s\n' "$spatch_rc" > "$LOG_DIR/large_pipeline.status"
  exit "$spatch_rc"
fi

printf '%s\n' "$(date --iso-8601=seconds) standardized analyses started"
"$PYTHON_BIN" scripts/analyze_result_v3_large_standardized.py
analysis_rc=$?
printf '%s\n' "$analysis_rc" > "$LOG_DIR/large_standardized_analysis.exit"
if (( analysis_rc != 0 )); then
  printf 'failed_analysis:%s\n' "$analysis_rc" > "$LOG_DIR/large_pipeline.status"
  exit "$analysis_rc"
fi

printf 'complete\n' > "$LOG_DIR/large_pipeline.status"
printf '%s\n' "$(date --iso-8601=seconds) large result_v3 pipeline completed"
