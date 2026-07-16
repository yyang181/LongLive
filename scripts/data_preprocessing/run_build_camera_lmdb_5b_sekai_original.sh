#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the *official* Sekai (Leegen/Sekai-Project)
# per-video camera NPZ release.
#
# Key differences from run_build_camera_lmdb_5b_sekai.sh (VGGT-Omega):
#   - Camera NPZs come from the official Sekai-Project HF dataset. Each
#     <clip_id>.npz has:
#         intrinsic : (3, 3)   float   ALREADY normalized by image W/H
#         extrinsic : (T, 4, 4) float  c2w, one pose per RAW video frame
#                                       (T = 300 or 1800, NOT strided).
#   - Because the trajectory is per-raw-frame (not 4x-strided like the
#     VGGT-Omega output), the Python script internally uses
#     ``vae_time_stride=4`` and ``cam_sample_strategy='last'`` — identical to
#     run_build_camera_lmdb_5b_sekai_game.sh's convention.
#   - Intrinsics are already normalized, so no --orig_w / --orig_h arg.
#   - Captions come from CSV files (same schema as
#     run_build_camera_lmdb_5b_sekai.sh: columns ``videoFile`` and ``caption``).
#
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai/video}
CAMERA_DIR=${CAMERA_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai-Project/camera}
CAPTION_CSV=${CAPTION_CSV:-"/nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Game.csv /nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Real-HQ.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/sekai_original/}

TARGET_H=${TARGET_H:-448}
TARGET_W=${TARGET_W:-832}
MAX_FRAMES=${MAX_FRAMES:-157}
NPROC=${NPROC:-4}
CAM_SAMPLE_STRATEGY=${CAM_SAMPLE_STRATEGY:-last}

# Resume controls (same semantics as run_build_camera_lmdb_5b_sekai.sh):
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
        echo "ERROR: caption CSV not found: ${f}" >&2; exit 1
    fi
done

if [[ -z "${TIMER_TOTAL_ITEMS:-}" ]]; then
    TIMER_TOTAL_ITEMS=$(find "${VIDEO_DIR}" -type f -name '*.mp4' 2>/dev/null | wc -l)
fi
TIMER_OUTPUT_DIR="${OUTPUT_DIR}"
run_with_timing torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b_sekai_original.py \
    --video_dir           "${VIDEO_DIR}" \
    --camera_dir          "${CAMERA_DIR}" \
    --caption_csv         "${CAPTION_CSV_LIST[@]}" \
    --output_dir          "${OUTPUT_DIR}" \
    --target_h            "${TARGET_H}" \
    --target_w            "${TARGET_W}" \
    --max_frames          "${MAX_FRAMES}" \
    --cam_sample_strategy "${CAM_SAMPLE_STRATEGY}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
