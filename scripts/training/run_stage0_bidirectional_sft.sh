#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage-0: plain bidirectional SFT for Wan2.2-TI2V-5B (no camera / no PRoPE).
#
# Override defaults via environment variables, e.g.
#     NPROC=4 LOGDIR=logs/train_bidir_sft \
#         bash scripts/training/run_stage0_bidirectional_sft.sh

set -euo pipefail

CONFIG=${CONFIG:-configs/train_bidir_sft.yaml}
LOGDIR=${LOGDIR:-logs/train_bidir_sft}
WANDB_DIR=${WANDB_DIR:-wandb}
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-29502}

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
