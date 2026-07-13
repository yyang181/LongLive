# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import utils.tv_io_patch  # noqa: F401 — patch torchvision.io before anything imports it
import torch.distributed as dist
from omegaconf import OmegaConf
import wandb

from trainer import (
    ScoreDistillationTrainer,
    DiffusionTrainer,
    CameraBidirectionalDiffusionTrainer,
    DreamXInfMemStreamingDiffusionTrainer,
)
from utils.config import normalize_config


def _is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _validate_no_placeholder_paths(config, config_path):
    for key in ("data_path", "eval_data_path"):
        value = config.get(key, None)
        if isinstance(value, str) and "/path/to/longlive2" in value:
            raise ValueError(
                f"{config_path} still contains placeholder {key}={value!r}. "
                "Point it at a real dataset path, or use "
                "configs/train_bidir_camera.yaml / configs/train_bidir_sft.yaml "
                "for LMDB-based bidirectional SFT."
            )


def _resolve_relative_paths(config, config_path):
    """Resolve relative dataset paths to absolute paths.

    Relative ``data_path`` / ``eval_data_path`` entries are resolved against
    the repository root (the directory containing this ``train.py`` file) so
    that training works regardless of the cwd from which ``torchrun`` was
    launched.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for key in ("data_path", "eval_data_path"):
        value = config.get(key, None)
        if not isinstance(value, str) or not value:
            continue
        if os.path.isabs(value):
            continue
        resolved = os.path.normpath(os.path.join(repo_root, value))
        config[key] = resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--generate-before-train", action="store_true", help="Run one evaluation inference before training starts")

    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config_path)
    # Allow CLI overrides like: infra.sequence_parallel_size=2 batch_size=4
    if unknown:
        cli_conf = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(config, cli_conf)
        if _is_rank0():
            print(f"[train.py] CLI overrides applied: {unknown}")
    config = normalize_config(config)
    _validate_no_placeholder_paths(config, args.config_path)
    _resolve_relative_paths(config, args.config_path)
    if _is_rank0():
        print(
            f"[train.py] config={args.config_path} "
            f"trainer={config.get('trainer', None)} "
            f"data_path={config.get('data_path', None)}"
        )
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.generate_before_train = args.generate_before_train

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "camera_bidirectional_diffusion":
        trainer = CameraBidirectionalDiffusionTrainer(config)
    elif config.trainer == "dreamx_infmem_streaming_diffusion":
        trainer = DreamXInfMemStreamingDiffusionTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
