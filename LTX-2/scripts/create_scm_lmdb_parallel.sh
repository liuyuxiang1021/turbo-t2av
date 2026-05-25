#!/bin/bash
# 8-GPU parallel SCM latent LMDB creation.
# Usage: bash create_scm_lmdb_parallel.sh <mapping_csv> <video_dir> <checkpoint_path> <output_lmdb>
set -e
MAPPING="${1:?Usage: $0 <mapping_csv> <video_dir> <checkpoint_path> <output_lmdb>}"
VIDEO_DIR="${2:?}"
CKPT="${3:?}"
OUTPUT="${4:?}"
NUM_GPUS="${NUM_GPUS:-8}"
WORKERS="${NUM_WORKERS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"


mkdir -p "$OUTPUT"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i \
  pixi run python -m ltx_distillation.tools.create_scm_latent_lmdb \
    --mapping_csv "$MAPPING" \
    --video_dir "$VIDEO_DIR" \
    --checkpoint_path "$CKPT" \
    --output_lmdb "$OUTPUT" \
    --num_shards "$NUM_GPUS" \
    --shard_id "$i" \
    --num_workers "$WORKERS" \
    --max_samples "$MAX_SAMPLES" \
    > "${OUTPUT}/shard_${i}.log" 2>&1 &
done
wait
echo "All $NUM_GPUS shards complete."
