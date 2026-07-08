#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the Sekai (Leegen/Sekai) dataset using
# VGGT-Omega camera estimates.
#
# Key difference from run_build_camera_lmdb_5b_sekai_game.sh:
#   - Camera parameters come from VGGT-Omega (batch_vggt_omega.py), stored as
#     per-video <clip_id>.npz in $CAMERA_DIR.
#   - Cameras are ALREADY 4×-strided (--frame_stride 4 in VGGT), so each camera
#     pose maps 1-to-1 to a Wan2.2 latent frame. The build script uses
#     vae_time_stride=1 internally (no additional subsampling).
#   - Captions come from CSV files ($CAPTION_CSV) with 'videoFile' and
#     'caption' columns, not JSON.
#
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai/video}
CAMERA_DIR=${CAMERA_DIR:-/nfs/yixinyang/code/LongLive/data/Sekai/vggt_omega_results}
CAPTION_CSV=${CAPTION_CSV:-"/nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Game.csv /nfs/yixinyang/code/LongLive/data/Sekai/data/train/Sekai-Real-HQ.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/sekai/}

TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-157}
NPROC=${NPROC:-8}

# Resume controls (same semantics as run_build_camera_lmdb_5b_sekai_game.sh):
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

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b_sekai.py \
    --video_dir     "${VIDEO_DIR}" \
    --camera_dir    "${CAMERA_DIR}" \
    --caption_csv   "${CAPTION_CSV_LIST[@]}" \
    --output_dir    "${OUTPUT_DIR}" \
    --target_h      "${TARGET_H}" \
    --target_w      "${TARGET_W}" \
    --max_frames    "${MAX_FRAMES}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
