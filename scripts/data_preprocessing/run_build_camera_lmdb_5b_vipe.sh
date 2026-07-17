#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the Sekai dataset using ViPE camera estimates.
#
# Camera parameters come from ViPE (NVIDIA VIdeo Pose Estimation), stored as
# per-video <clip_id>.npz in $CAMERA_DIR (the 'pose' subdirectory of ViPE
# results).  Intrinsics are auto-derived from $CAMERA_DIR by replacing 'pose'
# with 'intrinsics' (or override with INTRINSICS_DIR).
#
# ViPE poses are c2w (camera-to-world) at interval=1 (every frame).  The build
# script subsamples every 4 frames to match the Wan2.2 VAE 4× temporal stride,
# converts c2w → w2c, and stores as (F_lat, 7) [tx,ty,tz, qx,qy,qz,qw] —
# identical format to build_camera_lmdb_5b.py (minWM string-based).
#
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai/video}
CAMERA_DIR=${CAMERA_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai/vipe_results/pose}
INTRINSICS_DIR=${INTRINSICS_DIR:-}
CAPTION_CSV=${CAPTION_CSV:-"/nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Game.csv /nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Real-HQ.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/sekai_vipe/}

TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-957}
NPROC=${NPROC:-4}

# Resume controls:
KEEP_SHARDS=${KEEP_SHARDS:-0}
NO_RESUME=${NO_RESUME:-0}

EXTRA_ARGS=()
if [[ "${KEEP_SHARDS}" == "1" ]]; then
    EXTRA_ARGS+=(--keep_shards)
fi
if [[ "${NO_RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--no_resume)
fi

# Expand CAPTION_CSV (space- or comma-separated) into an explicit argv list.
read -r -a CAPTION_CSV_LIST <<< "$(echo "${CAPTION_CSV}" | tr ',' ' ')"
if [[ ${#CAPTION_CSV_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no CAPTION_CSV specified" >&2; exit 1
fi
for f in "${CAPTION_CSV_LIST[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: caption file not found: ${f}" >&2; exit 1
    fi
done

# Use parent directory IDs only for nested layouts such as <clip_id>/gen.mp4.
if [[ -z "${USE_PARENT_AS_CLIP_ID:-}" ]]; then
    USE_PARENT_AS_CLIP_ID=0
    if find "${VIDEO_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*.mp4' -print -quit | grep -q .; then
        USE_PARENT_AS_CLIP_ID=1
    fi
fi
if [[ "${USE_PARENT_AS_CLIP_ID}" == "1" ]]; then
    EXTRA_ARGS+=(--use_parent_as_clip_id)
fi

# Build --intrinsics_dir if set.
INTRINSICS_ARG=()
if [[ -n "${INTRINSICS_DIR}" ]]; then
    INTRINSICS_ARG=(--intrinsics_dir "${INTRINSICS_DIR}")
fi

if ! [[ "${TIMER_TOTAL_ITEMS:-}" =~ ^[1-9][0-9]*$ ]]; then
    TIMER_TOTAL_ITEMS=$(find "${VIDEO_DIR}" -type f -name '*.mp4' 2>/dev/null | wc -l)
fi
TIMER_OUTPUT_DIR="${OUTPUT_DIR}"
run_with_timing torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b_vipe.py \
    --video_dir     "${VIDEO_DIR}" \
    --camera_dir    "${CAMERA_DIR}" \
    "${INTRINSICS_ARG[@]+"${INTRINSICS_ARG[@]}"}" \
    --caption_csv   "${CAPTION_CSV_LIST[@]}" \
    --output_dir    "${OUTPUT_DIR}" \
    --target_h      "${TARGET_H}" \
    --target_w      "${TARGET_W}" \
    --max_frames    "${MAX_FRAMES}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
