#!/usr/bin/env bash
# Build a camera-aware LMDB from extracted P2P-full recordings.
# P2P stores <game>/<uuid>.mp4 + <uuid>.proto; the MIND builder expects
# <sample>/video.mp4 + action.json.  Convert first, then reuse the standard
# builder so the final LMDB contract remains unchanged.
set -euxo pipefail

VIDEO_DIR=${VIDEO_DIR:-/nfs/yixinyang/code/LongLive/data/p2pfull}
OUTPUT_DIR=${OUTPUT_DIR:-./data/train/p2pfull/}
TARGET_H=${TARGET_H:-704}
TARGET_W=${TARGET_W:-1280}
MAX_FRAMES=${MAX_FRAMES:-593}
NPROC=${NPROC:-8}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/_run_with_timing.sh"
ADAPTER_DIR=${P2P_MIND_ADAPTER_DIR:-${OUTPUT_DIR%/}/.p2p_mind_input}

PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}:${PYTHONPATH:-}" \
python "${SCRIPT_DIR}/convert_p2p_to_mind.py" \
    --p2p_path "${VIDEO_DIR}" \
    --output_dir "${ADAPTER_DIR}"

VIDEO_DIR="${ADAPTER_DIR}" \
CAMERA_DIR="${ADAPTER_DIR}" \
CAPTION_CSV="" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TARGET_H="${TARGET_H}" TARGET_W="${TARGET_W}" \
MAX_FRAMES="${MAX_FRAMES}" NPROC="${NPROC}" \
TIMER_TOTAL_ITEMS="${TIMER_TOTAL_ITEMS:-$(find "${VIDEO_DIR}" -type f -name '*.mp4' 2>/dev/null | wc -l)}" \
TIMER_OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/run_build_camera_lmdb_5b_mind.sh"
