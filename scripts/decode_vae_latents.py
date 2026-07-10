# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decode saved latent files (.pt) to pixel video and write .mp4 using the specified VAE.

Supports multi-GPU via torchrun:
    torchrun --nproc_per_node=8 scripts/decode_vae_latents.py --input_dir /path/to/latents
Single-GPU also works:
    python scripts/decode_vae_latents.py --input_dir /path/to/latents

Uses the Wan2.2-TI2V-5B VAE.
"""
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.tv_io_patch  # noqa: F401 — patch torchvision.io before importing write_video
import torch
import torch.distributed as dist
from torchvision.io import write_video
from tqdm import tqdm


def init_distributed():
    """Initialize distributed process group if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def get_vae():
    """Return the Wan2.2-TI2V-5B VAE wrapper."""
    from utils.wan_5b_wrapper import WanVAEWrapper
    return WanVAEWrapper()


def decode_latent_to_video(vae, latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Decode latent to pixel video; same logic as pipeline/causal_diffusion_inference.py.

    Args:
        vae: Wan2.2-TI2V-5B VAE wrapper
        latent: shape (batch, T, C, H, W) or (T, C, H, W)
        device, dtype: device and dtype for computation

    Returns:
        video: shape (batch, T, C, H, W), range [0, 1]
    """
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(device=device, dtype=dtype)
    video = vae.decode_to_pixel(latent)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video


_DEFAULT_LOGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "train_ar")
)
_DEFAULT_VIS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "train_ar", "decoded_videos")
)


def _find_latest_generated_video_dir(logs_dir: str) -> str:
    """Find the latest `generated_video_*` subdirectory under logs_dir."""
    candidates = sorted(glob.glob(os.path.join(logs_dir, "generated_video_*")))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        raise FileNotFoundError(
            f"No 'generated_video_*' directories found under {logs_dir}. "
            f"Please pass --input_dir explicitly."
        )
    return candidates[-1]


def _latent_to_mp4_name(pt_path: str) -> str:
    basename = os.path.splitext(os.path.basename(pt_path))[0]
    if basename.startswith("latents_"):
        return basename.replace("latents_", "video_", 1) + ".mp4"
    return f"video_{basename}.mp4"


def _collect_tasks(input_dir: str, pattern: str, output_dir_arg, vis_root: str,
                   skip_existing: bool = True):
    """
    Build a list of (latent_path, out_dir) tasks.

    Supports:
      1) input_dir directly contains <pattern> files.
      2) input_dir is a parent dir containing 'generated_video_*' subdirs.

    Returns:
        tasks, out_dirs, skipped_count
    """
    def _filter_existing(files, sub_out):
        kept, skipped = [], 0
        for p in files:
            mp4 = os.path.join(sub_out, _latent_to_mp4_name(p))
            if skip_existing and os.path.exists(mp4) and os.path.getsize(mp4) > 0:
                skipped += 1
                continue
            kept.append(p)
        return kept, skipped

    # Case 1: input_dir itself has latent files.
    direct = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if direct:
        out_dir = output_dir_arg or os.path.join(
            vis_root, os.path.basename(os.path.normpath(input_dir))
        )
        kept, skipped = _filter_existing(direct, out_dir)
        return [(p, out_dir) for p in kept], [out_dir], skipped

    # Case 2: scan generated_video_* subdirs.
    sub_dirs = sorted(
        d for d in glob.glob(os.path.join(input_dir, "generated_video_*"))
        if os.path.isdir(d)
    )
    tasks = []
    out_dirs = []
    total_skipped = 0
    for sub in sub_dirs:
        sub_files = sorted(glob.glob(os.path.join(sub, pattern)))
        if not sub_files:
            continue
        if output_dir_arg:
            sub_out = os.path.join(output_dir_arg, os.path.basename(sub))
        else:
            sub_out = os.path.join(vis_root, os.path.basename(sub))
        kept, skipped = _filter_existing(sub_files, sub_out)
        total_skipped += skipped
        if not kept:
            # Fully decoded already; no need to (re)create dir or add tasks.
            continue
        out_dirs.append(sub_out)
        for p in kept:
            tasks.append((p, sub_out))
    return tasks, out_dirs, total_skipped


def main():
    parser = argparse.ArgumentParser(description="Decode saved latents to video.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=(
            "Directory containing .pt latent files (e.g. latents_rank00_idx000000.pt). "
            f"Default: latest 'generated_video_*' under {_DEFAULT_LOGS_DIR}."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save decoded .mp4 videos. "
            f"Default: {_DEFAULT_VIS_ROOT}/<input_dir basename>"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS for output video (default: 24).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="latents_*.pt",
        help="Glob pattern for latent files (default: latents_*.pt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-decode and overwrite even if the target .mp4 already exists.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)

    if args.input_dir is None:
        args.input_dir = _find_latest_generated_video_dir(_DEFAULT_LOGS_DIR)
        if is_main:
            print(f"[Auto] --input_dir not given, using latest: {args.input_dir}")

    tasks, out_dirs, skipped = _collect_tasks(
        args.input_dir, args.pattern, args.output_dir, _DEFAULT_VIS_ROOT,
        skip_existing=(not args.overwrite),
    )
    if is_main and skipped > 0:
        print(f"[Skip] {skipped} latent(s) already decoded (mp4 exists). Use --overwrite to redo.")
    if not tasks:
        if is_main:
            if skipped > 0:
                print("All latents already decoded. Nothing to do.")
            else:
                print(
                    f"No files matching '{args.pattern}' found under "
                    f"'{args.input_dir}' (also scanned its 'generated_video_*' subdirs). Exiting."
                )
        return

    if is_main:
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    vae = get_vae()
    vae = vae.to(device=device, dtype=dtype).eval()

    # Shard tasks across ranks
    tasks_local = tasks[rank::world_size]
    if is_main:
        print(
            f"Found {len(tasks)} latent file(s) to decode across {len(out_dirs)} output dir(s), "
            f"{world_size} GPU(s), ~{len(tasks_local)} per GPU."
        )

    pbar = tqdm(tasks_local, desc=f"[Rank {rank}] Decoding", disable=(not is_main))
    for pt_path, out_dir in pbar:
        out_path = os.path.join(out_dir, _latent_to_mp4_name(pt_path))

        # Race-safe re-check (another rank/run might have produced it).
        if (not args.overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            if isinstance(data, torch.Tensor):
                latent = data
            elif isinstance(data, dict):
                latent = next(iter(data.values()))
                if not isinstance(latent, torch.Tensor):
                    print(f"[Rank {rank}] Skip {pt_path}: dict value is not a tensor.")
                    continue
            else:
                print(f"[Rank {rank}] Skip {pt_path}: unsupported type {type(data)}.")
                continue
        except Exception as e:
            print(f"[Rank {rank}] Failed to load {pt_path}: {e}")
            continue

        video = decode_latent_to_video(vae, latent, device, dtype)
        video_frames = video[0].cpu()
        video_uint8 = (video_frames * 255.0).clamp(0, 255).to(torch.uint8)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        write_video(out_path, video_uint8, fps=args.fps)

    if world_size > 1:
        dist.barrier()
    if is_main:
        print(f"Done. Decoded {len(tasks)} latent(s) into {len(out_dirs)} dir(s) under "
              f"{args.output_dir or _DEFAULT_VIS_ROOT}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
