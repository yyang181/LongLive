#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional + camera (PRoPE) inference for Wan2.2-TI2V-5B.

Loads a Camera-PRoPE Wan2.2-TI2V-5B generator checkpoint, parses one or more
WorldPlayGen camera-trajectory strings, and runs 50-step UniPC/Euler-style
flow-matching denoising with PRoPE conditioning, then VAE-decodes to mp4.

Inputs (paths inside `configs/infer_bidir_camera.yaml`):
    inference.prompt_path      : one prompt per line
    inference.trajectory_path  : one WorldPlayGen pose string per line
                                 (must align 1-to-1 with prompts)

Run:
    python scripts/inference/inference_bidir_camera.py \
        --config_path  configs/infer_bidir_camera.yaml \
        --generator_ckpt logs/train_bidir_camera/checkpoint_model_005000/model.pt \
        --output_dir videos/camera_bidir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure repository root is importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the camera-trajectory parsing from the data-prep script.
from scripts.data_preprocessing.build_camera_lmdb_5b import poses_from_pose_str  # noqa: E402
from utils.config import normalize_config  # noqa: E402
from utils.inference_utils import save_video  # noqa: E402
from utils.wan_5b_camera_wrapper import CameraWanDiffusionWrapper  # noqa: E402
from utils.wan_5b_wrapper import WanTextEncoder, WanVAEWrapper  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    """[x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


def build_viewmats_and_Ks(
    intrinsics_norm: np.ndarray,
    poses: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (intrinsics_norm[4], poses[F,7]) to (viewmats[F,4,4], Ks[F,3,3]).

    Mirrors `utils.camera_dataset.build_viewmats_and_Ks` so that training and
    inference share an identical PRoPE input convention.
    """
    F = poses.shape[0]
    fx_n, fy_n, cx_n, cy_n = intrinsics_norm.tolist()
    fx, fy = fx_n * target_w, fy_n * target_h
    cx, cy = cx_n * target_w, cy_n * target_h

    Ks = np.zeros((F, 3, 3), dtype=np.float32)
    Ks[:, 0, 0] = fx
    Ks[:, 1, 1] = fy
    Ks[:, 0, 2] = cx
    Ks[:, 1, 2] = cy
    Ks[:, 2, 2] = 1.0

    viewmats = np.zeros((F, 4, 4), dtype=np.float32)
    for i in range(F):
        t = poses[i, :3]
        q = poses[i, 3:]
        viewmats[i, :3, :3] = _quat_to_R(q)
        viewmats[i, :3, 3] = t
        viewmats[i, 3, 3] = 1.0

    # Normalize to first-frame camera coordinates (so PRoPE is translation-
    # invariant within a clip).
    inv0 = np.linalg.inv(viewmats[0])
    viewmats = np.einsum("ij,fjk->fik", inv0, viewmats)
    return torch.from_numpy(viewmats), torch.from_numpy(Ks)


def load_generator(config, ckpt_path: str, device: torch.device) -> CameraWanDiffusionWrapper:
    mk = config.get("model_kwargs", {})
    gen = CameraWanDiffusionWrapper(
        model_name=mk.get("model_name", "Wan2.2-TI2V-5B"),
        timestep_shift=float(mk.get("timestep_shift", 5.0)),
        is_causal=False,
        use_camera=bool(mk.get("use_camera", True)),
    )
    if ckpt_path:
        print(f"[infer] loading generator weights from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        # Strip optional "model." prefix dumped by some FSDP routines.
        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in sd.items()
        }
        missing, unexpected = gen.model.load_state_dict(cleaned, strict=False)
        print(f"[infer] missing={len(missing)}  unexpected={len(unexpected)}")
        if len(unexpected):
            print("  unexpected (first 5):", unexpected[:5])
    gen = gen.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return gen


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--generator_ckpt", default="",
                        help="Path to model.pt (optional; falls back to "
                             "pretrained Wan2.2-TI2V-5B with zero-init PRoPE).")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_clips", type=int, default=-1,
                        help="Cap number of (prompt, trajectory) pairs to render.")
    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    inf_cfg = config["inference"]
    out_dir = Path(args.output_dir or config["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # ---------- text + VAE ----------
    text_encoder = WanTextEncoder().eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    # ---------- generator ----------
    ckpt = args.generator_ckpt or config.get("checkpoints", {}).get("generator_ckpt", "")
    generator = load_generator(config, ckpt, device)

    # ---------- prompts and trajectories ----------
    with open(inf_cfg["prompt_path"]) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    with open(inf_cfg["trajectory_path"]) as f:
        trajectories = [ln.strip() for ln in f if ln.strip()]
    assert len(prompts) == len(trajectories), \
        f"prompts ({len(prompts)}) and trajectories ({len(trajectories)}) must align"
    if args.max_clips > 0:
        prompts = prompts[: args.max_clips]
        trajectories = trajectories[: args.max_clips]

    # ---------- shapes ----------
    F_lat = int(inf_cfg["num_latent_frames"])
    H_lat = int(inf_cfg["height_latent"])
    W_lat = int(inf_cfg["width_latent"])
    C_lat = int(inf_cfg.get("num_channels", 48))
    target_h = H_lat * 16
    target_w = W_lat * 16
    fps = int(inf_cfg.get("fps", 24))
    intrinsics_norm = np.array([
        float(inf_cfg["fx_norm"]), float(inf_cfg["fy_norm"]),
        float(inf_cfg["cx_norm"]), float(inf_cfg["cy_norm"]),
    ], dtype=np.float32)

    # ---------- scheduler ----------
    scheduler = generator.get_scheduler()
    sampling_steps = int(inf_cfg.get("sampling_steps", 50))
    scheduler.set_timesteps(sampling_steps, training=False)
    timesteps = scheduler.timesteps.to(device)

    # ---------- generate ----------
    for clip_idx, (prompt, traj) in enumerate(zip(prompts, trajectories)):
        gen = torch.Generator(device=device).manual_seed(args.seed + clip_idx)

        # Pose -> viewmats / Ks for this trajectory.
        intrinsics_clip, poses_clip = poses_from_pose_str(
            traj, F_lat, target_h, target_w)
        # The data-prep script returns intrinsics already normalized; we
        # override with config-provided defaults here so inference can vary
        # focal/principal-point at test time.
        viewmats, Ks = build_viewmats_and_Ks(
            intrinsics_norm, poses_clip, target_h, target_w)
        viewmats = viewmats[None].to(device=device, dtype=torch.float32)
        Ks = Ks[None].to(device=device, dtype=torch.float32)

        # Text conditioning (CFG).
        cond = text_encoder([prompt])
        uncond = text_encoder([""])

        # Initial noise: [B=1, F_lat, C, H_lat, W_lat] (matches WanDiffusionWrapper input).
        x = torch.randn(
            (1, F_lat, C_lat, H_lat, W_lat),
            device=device, dtype=torch.bfloat16, generator=gen,
        )

        cfg_scale = float(inf_cfg.get("guidance_scale", 5.0))
        for ti, t_scalar in enumerate(timesteps):
            t = t_scalar.expand(1, F_lat).to(device)
            flow_c, _ = generator(
                noisy_image_or_video=x,
                conditional_dict=cond,
                timestep=t,
                viewmats=viewmats,
                Ks=Ks,
            )
            flow_u, _ = generator(
                noisy_image_or_video=x,
                conditional_dict=uncond,
                timestep=t,
                viewmats=viewmats,
                Ks=Ks,
            )
            flow_pred = flow_u + cfg_scale * (flow_c - flow_u)

            # FlowMatchScheduler.step expects 4-D tensors; flatten F.
            x_flat = x.flatten(0, 1)              # (F, C, H, W)
            f_flat = flow_pred.flatten(0, 1)
            x_flat = scheduler.step(f_flat, t.flatten(0, 1), x_flat)
            x = x_flat.unflatten(0, x.shape[:2]).to(torch.bfloat16)

            if (ti + 1) % 10 == 0:
                print(f"[clip {clip_idx}] step {ti + 1}/{sampling_steps}")

        # Decode latent to pixel: vae expects [B, F_lat, C, H, W].
        with torch.no_grad():
            video = vae.decode_to_pixel(x.to(torch.bfloat16))   # in [-1, 1]
        video = ((video.float()[0] + 1.0) * 0.5).clamp(0, 1)    # (F_raw, C, H, W) in [0,1]

        out_path = out_dir / f"clip_{clip_idx:03d}.mp4"
        save_video(video, str(out_path), fps=fps)
        print(f"[clip {clip_idx}] saved -> {out_path}")

        # also persist the prompt / trajectory next to the video
        with open(out_path.with_suffix(".txt"), "w") as f:
            f.write(f"prompt:     {prompt}\n")
            f.write(f"trajectory: {traj}\n")

    print(f"\nAll clips written to: {out_dir}")


if __name__ == "__main__":
    main()
