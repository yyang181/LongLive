#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding + WorldPlayGen camera-trajectory parsing
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

INPUT_JSON=${INPUT_JSON:-./dataset/LongLive/CameraSFT/raw/clips.json}
OUTPUT_DIR=${OUTPUT_DIR:-./dataset/LongLive/CameraSFT}
TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-77}
NPROC=${NPROC:-8}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    scripts/data_preprocessing/build_camera_lmdb_5b.py \
    --input_json   ${INPUT_JSON} \
    --output_dir   ${OUTPUT_DIR} \
    --target_h     ${TARGET_H} \
    --target_w     ${TARGET_W} \
    --max_frames   ${MAX_FRAMES}
