#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the **MIND** dataset (CSU-JPG/MIND).
#
# MIND stores per-clip data in ``data-<id>/`` sub-folders, each containing:
#   - video.mp4     (1920x1080, 24 fps)
#   - action.json   (ws/ad/ud/lr controls; optional pose fields are ignored)
#
# Key differences from run_build_camera_lmdb_5b_sekai_original.sh:
#   - Camera trajectories are generated from the official MIND 0/1/2
#     ws/ad/ud/lr controls using the same local OpenCV convention and default
#     0.08 m / 3 degree per-frame speeds as build_camera_lmdb_5b.py.
#   - The root can contain both 1st_data/train and 3rd_data/train; the builder
#     keeps relative paths so data-* names cannot collide and excludes test.
#   - Intrinsics are synthesized from a default horizontal FOV (90 degrees)
#     because MIND does not ship camera intrinsics.
#   - Captions: action.json files may not contain captions; a generic gameplay
#     prompt is used unless optional caption files are provided.
#
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/MIND}
CAMERA_DIR=${CAMERA_DIR:-/nfs/yixinyang/code/LongLive/data/MIND}
CAPTION_CSV=${CAPTION_CSV:-""}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/MIND/}

TARGET_H=${TARGET_H:-448}
TARGET_W=${TARGET_W:-832}
MAX_FRAMES=${MAX_FRAMES:-157}
NPROC=${NPROC:-8}
CAM_SAMPLE_STRATEGY=${CAM_SAMPLE_STRATEGY:-last}

# MIND-specific options
DEFAULT_CAPTION=${DEFAULT_CAPTION:-"A high-quality gameplay video with detailed 3D environments, smooth character animation, and dynamic camera movement."}
CAMERA_FOV=${CAMERA_FOV:-90}
ORIG_W=${ORIG_W:-1920}
ORIG_H=${ORIG_H:-1080}
SPLIT=${SPLIT:-train}
FORWARD_SPEED=${FORWARD_SPEED:-0.08}
YAW_SPEED_DEG=${YAW_SPEED_DEG:-3.0}
PITCH_SPEED_DEG=${PITCH_SPEED_DEG:-3.0}

# Resume controls (same semantics as the Sekai scripts):
KEEP_SHARDS=${KEEP_SHARDS:-0}
NO_RESUME=${NO_RESUME:-0}

EXTRA_ARGS=()
if [[ "${KEEP_SHARDS}" == "1" ]]; then
    EXTRA_ARGS+=(--keep_shards)
fi
if [[ "${NO_RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--no_resume)
fi

# Build the torchrun command.  --caption_csv is only passed when at least one
# valid CSV file is provided; otherwise the Python script uses --default_caption.
CMD=(
    scripts/data_preprocessing/build_camera_lmdb_5b_mind.py
    --video_dir           "${VIDEO_DIR}"
    --camera_dir          "${CAMERA_DIR}"
    --output_dir          "${OUTPUT_DIR}"
    --target_h            "${TARGET_H}"
    --target_w            "${TARGET_W}"
    --max_frames          "${MAX_FRAMES}"
    --cam_sample_strategy "${CAM_SAMPLE_STRATEGY}"
    --default_caption     "${DEFAULT_CAPTION}"
    --camera_fov          "${CAMERA_FOV}"
    --orig_w              "${ORIG_W}"
    --orig_h              "${ORIG_H}"
    --split               "${SPLIT}"
    --forward_speed       "${FORWARD_SPEED}"
    --yaw_speed_deg       "${YAW_SPEED_DEG}"
    --pitch_speed_deg     "${PITCH_SPEED_DEG}"
)

# Expand CAPTION_CSV (space- or comma-separated) into an explicit argv list.
# Only pass files that actually exist (CAPTION_CSV may point to a directory or
# be empty, in which case we fall back to the default caption).
CAPTION_ARGS=()
if [[ -n "${CAPTION_CSV}" ]]; then
    read -r -a CAPTION_CSV_LIST <<< "$(echo "${CAPTION_CSV}" | tr ',' ' ')"
    for f in "${CAPTION_CSV_LIST[@]}"; do
        if [[ -f "${f}" ]]; then
            CAPTION_ARGS+=("${f}")
        fi
    done
fi
if [[ ${#CAPTION_ARGS[@]} -gt 0 ]]; then
    CMD+=(--caption_csv "${CAPTION_ARGS[@]}")
fi

CMD+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if [[ -z "${TIMER_TOTAL_ITEMS:-}" ]]; then
    TIMER_TOTAL_ITEMS=$(find "${CAMERA_DIR}" -type f -name 'action.json' 2>/dev/null | wc -l)
fi
TIMER_OUTPUT_DIR="${OUTPUT_DIR}"
run_with_timing torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" "${CMD[@]}"
