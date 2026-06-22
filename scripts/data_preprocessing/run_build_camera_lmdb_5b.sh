#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding + WorldPlayGen camera-trajectory parsing
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

INPUT_JSON=${INPUT_JSON:-./dataset/LongLive/CameraSFT/raw/clips.json}
OUTPUT_DIR=${OUTPUT_DIR:-./dataset/LongLive/CameraSFT}
VIDEO_DIR=${VIDEO_DIR:-}
TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-77}
NPROC=${NPROC:-8}
# Resume controls:
#   By default the job resumes automatically: the merged data/ LMDB records which
#   clips are already done (by video_path), so a re-run skips them and appends
#   only the new ones. The per-rank shards are just a transient parallel-write
#   buffer and are deleted after the merge.
#   Set NO_RESUME=1 to wipe data/ + shards and reprocess from scratch.
#   Set KEEP_SHARDS=1 to keep the per-rank shards after merging (debugging).
KEEP_SHARDS=${KEEP_SHARDS:-0}
NO_RESUME=${NO_RESUME:-0}

EXTRA_ARGS=()
if [[ "${KEEP_SHARDS}" == "1" ]]; then
    EXTRA_ARGS+=(--keep_shards)
fi
if [[ "${NO_RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--no_resume)
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b.py \
    --input_json   "${INPUT_JSON}" \
    --video_dir    "${VIDEO_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --target_h     "${TARGET_H}" \
    --target_w     "${TARGET_W}" \
    --max_frames   "${MAX_FRAMES}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
