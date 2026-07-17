#!/usr/bin/env bash
# Wan2.2-TI2V-5B VAE encoding for the Sekai-Game release.
#   - Reads videos from $VIDEO_DIR/*.mp4
#   - Reads per-frame c2w + intrinsics from one or more $CAMERA_NPZ shards
#   - Reads captions from one or more $CAPTION_JSON files
# Produces: $OUTPUT_DIR/data/  (LMDB consumed by CameraLatentLMDBDataset)
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"

VIDEO_DIR=${VIDEO_DIR:-/local-ssd/code/LongLive/data/sekai_game_train_961frames_16fps_ovl640/video}
CAMERA_NPZ=${CAMERA_NPZ:-"/local-ssd/code/LongLive/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_*_camera.npz"}
CAPTION_JSON=${CAPTION_JSON:-"/local-ssd/code/LongLive/data/sekai_game_train_961frames_16fps_ovl640/sekai_game_train_*_LongSceneStaticCaption-Qwen3-VL-30B-A3B-Instruct.json"}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/sekai_game_961frames_16fps_ovl640/}

TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
ORIG_H=${ORIG_H:-1080}
ORIG_W=${ORIG_W:-1920}
MAX_FRAMES=${MAX_FRAMES:-77}
NPROC=${NPROC:-8}
# Per-latent-frame anchor selection — mirrors SANA's
# SanaWMZipLatentDataset.cam_sample_strategy. 'last' (default) matches
# configs/sana_wm/stage1/v0_First_chunk.yaml; 'first' is also supported.
CAM_SAMPLE_STRATEGY=${CAM_SAMPLE_STRATEGY:-last}

# Resume controls (same semantics as run_build_camera_lmdb_5b.sh):
#   By default the job resumes automatically: the merged data/ LMDB records
#   which clips are already done (by video_path), so a re-run skips them and
#   appends only the new ones. The per-rank shards are just a transient
#   parallel-write buffer and are deleted after the merge.
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

# Expand glob-y CAMERA_NPZ / CAPTION_JSON into explicit lists so torchrun sees
# multiple --camera_npz / --caption_json argv entries (the python script also
# tolerates a single quoted glob pattern via _expand_globs, but expanding here
# fails fast if a path is mistyped).
shopt -s nullglob
read -r -a CAMERA_NPZ_LIST <<< "$(printf '%s\n' ${CAMERA_NPZ} | xargs)"
read -r -a CAPTION_JSON_LIST <<< "$(printf '%s\n' ${CAPTION_JSON} | xargs)"
if [[ ${#CAMERA_NPZ_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no CAMERA_NPZ matched: ${CAMERA_NPZ}" >&2; exit 1
fi
if [[ ${#CAPTION_JSON_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no CAPTION_JSON matched: ${CAPTION_JSON}" >&2; exit 1
fi
shopt -u nullglob

if ! [[ "${TIMER_TOTAL_ITEMS:-}" =~ ^[1-9][0-9]*$ ]]; then
    TIMER_TOTAL_ITEMS=$(find "${VIDEO_DIR}" -type f -name '*.mp4' 2>/dev/null | wc -l)
fi
TIMER_OUTPUT_DIR="${OUTPUT_DIR}"
run_with_timing torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    scripts/data_preprocessing/build_camera_lmdb_5b_sekai_game.py \
    --video_dir     "${VIDEO_DIR}" \
    --camera_npz    "${CAMERA_NPZ_LIST[@]}" \
    --caption_json  "${CAPTION_JSON_LIST[@]}" \
    --output_dir    "${OUTPUT_DIR}" \
    --target_h      "${TARGET_H}" \
    --target_w      "${TARGET_W}" \
    --orig_h        "${ORIG_H}" \
    --orig_w        "${ORIG_W}" \
    --max_frames    "${MAX_FRAMES}" \
    --cam_sample_strategy "${CAM_SAMPLE_STRATEGY}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
