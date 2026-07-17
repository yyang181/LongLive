#!/usr/bin/env bash
# Build camera-aware LMDB from DynamicMem metadata JSON (metadata_0710.json).
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
#
# Usage:
#   INPUT_JSON=/nfs/yinhanzhang/DynamicMem/datasets/metadata_0710.json \
#   OUTPUT_DIR=./data/train/dynmem-data/ \
#   TARGET_H=704 TARGET_W=1280 MAX_FRAMES=77 NPROC=8 \
#   bash scripts/data_preprocessing/run_build_camera_lmdb_5b_dynmem.sh
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"

INPUT_JSON=${INPUT_JSON:-/nfs/yinhanzhang/DynamicMem/datasets/metadata_0710.json}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/dynmem-data/}
VIDEO_DIR=${VIDEO_DIR:-}
TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-77}
NPROC=${NPROC:-8}
# Resume controls (identical to run_build_camera_lmdb_5b.sh):
KEEP_SHARDS=${KEEP_SHARDS:-0}
NO_RESUME=${NO_RESUME:-0}

EXTRA_ARGS=()
if [[ "${KEEP_SHARDS}" == "1" ]]; then
    EXTRA_ARGS+=(--keep_shards)
fi
if [[ "${NO_RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--no_resume)
fi

if ! [[ "${TIMER_TOTAL_ITEMS:-}" =~ ^[1-9][0-9]*$ ]]; then
    TIMER_TOTAL_ITEMS=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d,list) else len(d.get("clips", d)))' "${INPUT_JSON}" 2>/dev/null || echo 0)
fi
TIMER_OUTPUT_DIR="${OUTPUT_DIR}"
run_with_timing torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b_dynmem.py \
    --input_json   "${INPUT_JSON}" \
    --video_dir    "${VIDEO_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --target_h     "${TARGET_H}" \
    --target_w     "${TARGET_W}" \
    --max_frames   "${MAX_FRAMES}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
