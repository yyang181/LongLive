#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pretrained Wan2.2-TI2V-5B Image-to-Video inference (no fine-tune, no PRoPE).

This script runs **vanilla I2V** with the pretrained Wan2.2-TI2V-5B model
shipped under ``wan_models/Wan2.2-TI2V-5B/``. It does NOT load any
fine-tuned checkpoint and does NOT use camera/PRoPE conditioning.

I2V recipe (matches Wan2.2 native behavior):
  1. VAE-encode the source image (single frame) -> 1 latent frame.
  2. Initialize x ~ N(0, I) of shape [1, F_lat, C_lat, H_lat, W_lat] and
     overwrite x[:, :1] with the encoded image latent.
  3. At every denoising step, pass per-frame timesteps with t[:, 0] = 0
     (frame 0 is clean) and t[:, 1:] = current scalar timestep.
  4. Run CFG with positive + negative text prompt, take a flow-matching
     scheduler step, then re-pin x[:, :1] back to the clean image latent.
  5. VAE-decode the final latent -> mp4.

Usage:
    python scripts/inference/inference_pretrained_i2v.py \
        --image  /path/to/source.png \
        --prompt "a cinematic dolly-in shot of the scene" \
        --output_path videos/pretrained_i2v/clip_000.mp4

You can also pass aligned text files instead of single inputs:
    python scripts/inference/inference_pretrained_i2v.py \
        --image_list  prompts/images.txt \
        --prompt_list prompts/prompts.txt \
        --output_dir  videos/pretrained_i2v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure repository root is importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import utils.tv_io_patch  # noqa: E402, F401 — patch torchvision.io before anything imports it
from utils.inference_utils import save_video  # noqa: E402
from utils.scheduler import FlowMatchScheduler  # noqa: E402
from utils.wan_5b_camera_wrapper import CameraWanDiffusionWrapper  # noqa: E402
from utils.wan_5b_wrapper import (  # noqa: E402
    WanTextEncoder,
    WanVAEWrapper,
)


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_image_as_pixel(
    image_path: str,
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a [1, 3, 1, target_h, target_w] tensor in [-1, 1] for VAE encode."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_w, target_h), Image.BICUBIC)
    arr = torch.from_numpy(np.array(img, dtype=np.float32))      # (H, W, 3)
    arr = (arr / 127.5) - 1.0
    pixel = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)        # [1, 3, 1, H, W]
    return pixel.to(device=device, dtype=dtype)


def read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def to_bf16(d: dict) -> dict:
    return {k: (v.to(torch.bfloat16) if torch.is_tensor(v) else v) for k, v in d.items()}


# --------------------------------------------------------------------------
# Main I2V inference loop
# --------------------------------------------------------------------------
@torch.no_grad()
def run_i2v(
    *,
    generator: CameraWanDiffusionWrapper,
    text_encoder: WanTextEncoder,
    vae: WanVAEWrapper,
    image_path: str,
    prompt: str,
    negative_prompt: str,
    out_path: Path,
    F_lat: int,
    H_lat: int,
    W_lat: int,
    C_lat: int,
    sampling_steps: int,
    guidance_scale: float,
    fps: int,
    seed: int,
    device: torch.device,
) -> None:
    target_h = H_lat * 16
    target_w = W_lat * 16
    dtype = torch.bfloat16

    gen_rng = torch.Generator(device=device).manual_seed(seed)

    # ---- text conditioning (CFG) ----
    cond = to_bf16(text_encoder([prompt]))
    uncond = to_bf16(text_encoder([negative_prompt]))

    # ---- encode source image -> first latent frame ----
    img_pixel = load_image_as_pixel(image_path, target_h, target_w, device, dtype)
    image_latent = vae.encode_to_latent(img_pixel).to(dtype=dtype)  # [1, 1, C, H_lat, W_lat]
    assert image_latent.shape[1] == 1, (
        f"Expected single-frame image latent, got shape={tuple(image_latent.shape)}"
    )
    assert image_latent.shape[2:] == (C_lat, H_lat, W_lat), (
        f"image latent {tuple(image_latent.shape[2:])} does not match "
        f"(C={C_lat}, H={H_lat}, W={W_lat})"
    )

    # ---- initialize x = noise, then pin frame 0 ----
    x = torch.randn(
        (1, F_lat, C_lat, H_lat, W_lat),
        device=device, dtype=dtype, generator=gen_rng,
    )
    x[:, :1] = image_latent

    # ---- scheduler ----
    # NOTE: WanDiffusionWrapper internally builds a FlowMatchScheduler with
    # `sigma_min=0.0, extra_one_step=True` for **training** -- if we reuse it
    # at inference, the last sigma is `1/N` (e.g. 0.02 for 50 steps), and
    # after shift=5 the final timestep stays at t~=92, which leaves ~9% of
    # the noise unsolved and shows up as a still-noisy / broken tail in the
    # rendered video while frame 0 remains clean (because we hard-pin it).
    #
    # For pretrained Wan2.2-TI2V-5B inference we follow the official recipe:
    #   sigma_min = 0.003 / 1.002, extra_one_step = False
    # which makes the last shift=5 timestep ~= 14.8, i.e. essentially clean.
    scheduler = FlowMatchScheduler(
        shift=float(generator.scheduler.shift),
        sigma_min=0.003 / 1.002,
        extra_one_step=False,
    )
    scheduler.set_timesteps(sampling_steps, training=False)
    timesteps = scheduler.timesteps.to(device)

    print(f"[i2v] image='{image_path}'")
    print(f"[i2v] prompt='{prompt}'")
    print(f"[i2v] F_lat={F_lat} H_lat={H_lat} W_lat={W_lat} (pixel {target_h}x{target_w})")
    print(f"[i2v] sampling_steps={sampling_steps}  cfg={guidance_scale}  shift={scheduler.shift}")

    # ---- denoising loop (mirrors official Wan2.2 WanTI2V.i2v) ----
    #
    # Official Wan2.2 recipe (wan/textimage2video.py, WanTI2V.i2v):
    #
    #     latent = (1. - mask2) * z[0] + mask2 * latent           # init
    #     for t in timesteps:
    #         # per-token timestep: frame-0 patch tokens get t=0, the rest get t
    #         temp_ts = (mask2[0][:, ::2, ::2] * t).flatten()
    #         temp_ts = cat([temp_ts, ones(seq_len - len(temp_ts)) * t])
    #         noise_pred_c = model(latent, t=temp_ts.unsqueeze(0), context=c)
    #         noise_pred_u = model(latent, t=temp_ts.unsqueeze(0), context=u)
    #         noise_pred  = noise_pred_u + cfg * (noise_pred_c - noise_pred_u)
    #         latent      = scheduler.step(noise_pred, t, latent)   # scalar t
    #         latent      = (1. - mask2) * z[0] + mask2 * latent    # re-pin
    #
    # Two non-obvious points:
    #   * The scheduler step uses the **scalar** denoising timestep `t`, not
    #     the per-frame timestep. Per-frame `t[:, 0]=0` is ONLY for telling
    #     the model "frame 0 is clean"; it must NOT be fed to the scheduler.
    #     LongLive's FlowMatchScheduler.step uses argmin to map t -> sigma,
    #     and t=0 maps to the last sigma (sigma_min) which makes
    #     `(timestep_id + 1 >= len).any()` fire and forces sigma_next=0 for
    #     the whole batch -- a single step then overshoots from the current
    #     sigma all the way to 0, destroying the noisy frames (-> all black).
    #   * Frame 0 is hard-pinned back to `image_latent` AFTER each step,
    #     exactly like the official `(1.-mask2)*z[0] + mask2*latent` line.
    for ti, t_scalar in enumerate(timesteps):
        # Per-frame timestep ONLY for the model: frame 0 = 0 (clean), rest = t.
        t_per_frame = t_scalar.expand(1, F_lat).to(device).clone()
        t_per_frame[:, 0] = 0

        # CFG: cond + uncond model evaluations.
        flow_c, _ = generator(
            noisy_image_or_video=x,
            conditional_dict=cond,
            timestep=t_per_frame,
        )
        flow_u, _ = generator(
            noisy_image_or_video=x,
            conditional_dict=uncond,
            timestep=t_per_frame,
        )
        flow_pred = flow_u + guidance_scale * (flow_c - flow_u)

        # Scheduler step uses the SCALAR denoising timestep, broadcast to
        # the full latent. This matches official Wan2.2 i2v exactly.
        B = x.shape[0]
        t_scalar_b = t_scalar.expand(B).to(device)
        x_flat = x.flatten(0, 1)
        f_flat = flow_pred.flatten(0, 1)
        # Broadcast the scalar timestep to every (B*F) row of the flat batch
        # so FlowMatchScheduler.step maps every frame to the same sigma.
        t_flat = t_scalar.to(device).expand(x_flat.shape[0])
        x_flat = scheduler.step(f_flat, t_flat, x_flat)
        x = x_flat.unflatten(0, (1, F_lat)).to(dtype)

        # Re-pin the clean image latent on frame 0.
        x[:, :1] = image_latent

        if (ti + 1) % 10 == 0 or ti == len(timesteps) - 1:
            print(f"[i2v] step {ti + 1}/{sampling_steps}  t={float(t_scalar):.2f}")

    # ---- decode ----
    video = vae.decode_to_pixel(x.to(dtype))                 # [1, F_raw, 3, H, W] in [-1, 1]
    video = ((video.float()[0] + 1.0) * 0.5).clamp(0, 1)     # [F_raw, 3, H, W] in [0, 1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(out_path), fps=fps)
    print(f"[i2v] saved -> {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrained Wan2.2-TI2V-5B Image-to-Video inference."
    )
    # Single-clip mode
    parser.add_argument("--image", default=None, help="Path to a source image.")
    parser.add_argument("--prompt", default=None, help="Text prompt for the clip.")
    parser.add_argument("--output_path", default=None,
                        help="Output mp4 path (single-clip mode).")

    # Batch mode
    parser.add_argument("--image_list", default=None,
                        help="Text file of image paths (one per line).")
    parser.add_argument("--prompt_list", default=None,
                        help="Text file of prompts (one per line, aligned with --image_list).")
    parser.add_argument("--output_dir", default="videos/pretrained_i2v",
                        help="Output directory (batch mode).")

    # Negative prompt + sampling
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--timestep_shift", type=float, default=5.0)

    # Latent shape (defaults match the camera-bidir configs: 704x1280, 77 frames).
    parser.add_argument("--num_latent_frames", type=int, default=20)
    parser.add_argument("--height_latent", type=int, default=44)
    parser.add_argument("--width_latent", type=int, default=80)
    parser.add_argument("--num_channels", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_clips", type=int, default=-1,
                        help="Cap number of clips to render in batch mode.")

    args = parser.parse_args()

    # ---- resolve job list ----
    if args.image_list and args.prompt_list:
        images = read_lines(args.image_list)
        prompts = read_lines(args.prompt_list)
        assert len(images) == len(prompts), (
            f"image_list ({len(images)}) and prompt_list ({len(prompts)}) must align"
        )
        if args.max_clips > 0:
            images = images[: args.max_clips]
            prompts = prompts[: args.max_clips]
        out_dir = Path(args.output_dir)
        out_paths = [out_dir / f"clip_{i:03d}.mp4" for i in range(len(images))]
    elif args.image and args.prompt:
        images = [args.image]
        prompts = [args.prompt]
        if args.output_path:
            out_paths = [Path(args.output_path)]
        else:
            out_paths = [Path(args.output_dir) / "clip_000.mp4"]
    else:
        parser.error(
            "Either provide BOTH --image and --prompt, or BOTH --image_list and "
            "--prompt_list."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print("[i2v] loading text encoder ...")
    text_encoder = WanTextEncoder().eval().requires_grad_(False)

    print("[i2v] loading VAE ...")
    vae = (
        WanVAEWrapper()
        .to(device=device, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )

    print("[i2v] loading pretrained Wan2.2-TI2V-5B generator ...")
    # NOTE: We use CameraWanDiffusionWrapper with use_camera=False instead of
    # the base WanDiffusionWrapper because:
    #
    #   1. CameraWanDiffusionWrapper.forward correctly handles per-frame
    #      timesteps for the I2V protocol: when t[:, 0] = 0 differs from the
    #      other frames, it expands `timestep` to a per-token tensor of shape
    #      [B, seq_len] and feeds that to WanModel. Each patch token in
    #      frame `f` then carries `timestep[:, f]` -- so frame 0 gets t=0
    #      (clean) and the rest get the current denoising timestep.
    #
    #   2. The base WanDiffusionWrapper has `uniform_timestep=True` for
    #      non-causal models and silently collapses `timestep -> timestep[:, 0]`,
    #      which makes the entire video appear clean to the model, the flow
    #      prediction collapses to ~0, only frame 0 (which we hard-pin) stays
    #      sane and every other frame stays at its initial Gaussian noise.
    #      That manifests as: frame 0 is the source image, all later frames
    #      are completely broken.
    #
    # Setting `use_camera=False` skips PRoPE parameter registration, so the
    # backbone is byte-identical to the pretrained Wan2.2-TI2V-5B weights;
    # we only inherit the per-frame-timestep fix from CameraWanDiffusionWrapper.
    generator = CameraWanDiffusionWrapper(
        model_name="Wan2.2-TI2V-5B",
        timestep_shift=float(args.timestep_shift),
        is_causal=False,
        use_camera=False,
    )
    generator = (
        generator.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    )

    for clip_idx, (img_p, prm) in enumerate(zip(images, prompts)):
        out_path = out_paths[clip_idx]
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[i2v] clip {clip_idx}: skip (already exists -> {out_path})")
            continue

        run_i2v(
            generator=generator,
            text_encoder=text_encoder,
            vae=vae,
            image_path=img_p,
            prompt=prm,
            negative_prompt=args.negative_prompt,
            out_path=out_path,
            F_lat=args.num_latent_frames,
            H_lat=args.height_latent,
            W_lat=args.width_latent,
            C_lat=args.num_channels,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
            seed=args.seed + clip_idx,
            device=device,
        )

        # persist the prompt next to the video
        meta_path = out_path.with_suffix(".txt")
        with open(meta_path, "w") as f:
            f.write(f"prompt: {prm}\n")
            f.write(f"image:  {img_p}\n")


if __name__ == "__main__":
    main()
