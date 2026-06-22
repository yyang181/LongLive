#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bidirectional + camera (PRoPE) inference launcher (single GPU is enough).
#
# Override env vars to point at a trained checkpoint:
#     CKPT=logs/train_bidir_camera/checkpoint_model_005000/model.pt \
#         bash scripts/inference/run_infer_bidirectional_camera.sh

set -euo pipefail

CONFIG=${CONFIG:-configs/infer_bidir_camera.yaml}
CKPT=${CKPT:-}
OUT=${OUT:-videos/camera_bidir}
GPU=${GPU:-0}
SEED=${SEED:-42}

mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/inference/inference_bidir_camera.py \
    --config_path "${CONFIG}" \
    --generator_ckpt "${CKPT}" \
    --output_dir "${OUT}" \
    --seed "${SEED}"
