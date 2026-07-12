#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the **MIND** dataset (CSU-JPG/MIND).
#
# MIND stores per-clip data in ``data-<id>/`` sub-folders, each containing:
#   - video.mp4     (1920x1080, 24 fps)
#   - action.json   (camera_pos/camera_rpy per frame, NO caption, NO intrinsics)
#
# Key differences from run_build_camera_lmdb_5b_sekai_original.sh:
#   - Camera poses are parsed from action.json (camera_pos + camera_rpy Euler
#     angles), not from NPZ files.
#   - Intrinsics are synthesized from a default horizontal FOV (UE5 default
#     90 deg) since MIND does not ship camera intrinsics.
#   - Captions: action.json has no ``caption`` field in the released files;
#     a high-quality default prompt is used instead.  Optional CSV captions
#     can be provided via CAPTION_CSV (same schema as Sekai: videoFile + caption).
#   - UE5 positions are in centimetres; POSE_SCALE (default 0.01) converts
#     to metres.
#
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/MIND/3rd_data/train}
CAMERA_DIR=${CAMERA_DIR:-/nfs/yixinyang/code/LongLive/data/MIND/3rd_data/train}
CAPTION_CSV=${CAPTION_CSV:-""}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/MIND/}

TARGET_H=${TARGET_H:-448}
TARGET_W=${TARGET_W:-832}
MAX_FRAMES=${MAX_FRAMES:-157}
NPROC=${NPROC:-4}
CAM_SAMPLE_STRATEGY=${CAM_SAMPLE_STRATEGY:-last}

# MIND-specific options
DEFAULT_CAPTION=${DEFAULT_CAPTION:-"A high-quality third-person gameplay video with detailed 3D environments, smooth character animation, and dynamic camera movement."}
CAMERA_FOV=${CAMERA_FOV:-90}
ORIG_W=${ORIG_W:-1920}
ORIG_H=${ORIG_H:-1080}
POSE_SCALE=${POSE_SCALE:-0.01}

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
    --pose_scale          "${POSE_SCALE}"
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

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" "${CMD[@]}"
