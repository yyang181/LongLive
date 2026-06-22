#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage-0: Bidirectional + camera (PRoPE) SFT for Wan2.2-TI2V-5B.
#
# Trains the bidirectional WanModel with zero-init PRoPE residual blocks on
# the LMDB built by scripts/data_preprocessing/run_build_camera_lmdb_5b.sh.
#
# Override defaults via environment variables, e.g.
#     NPROC=4 CONFIG=configs/train_bidir_camera.yaml LOGDIR=logs/run1 \
#         bash scripts/training/run_stage0_bidirectional_camera.sh

set -euo pipefail

CONFIG=${CONFIG:-configs/train_bidir_camera.yaml}
LOGDIR=${LOGDIR:-logs/train_bidir_camera}
WANDB_DIR=${WANDB_DIR:-wandb}
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-29501}

mkdir -p "${LOGDIR}" "${WANDB_DIR}"

torchrun \
    --standalone \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    train.py \
        --config_path "${CONFIG}" \
        --logdir "${LOGDIR}" \
        --wandb-save-dir "${WANDB_DIR}" \
        --disable-wandb
