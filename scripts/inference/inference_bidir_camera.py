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
import re
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
from utils.camera_dataset import build_viewmats_and_Ks  # noqa: E402
from utils.config import normalize_config  # noqa: E402
from utils.inference_utils import save_video  # noqa: E402
from utils.wan_5b_camera_wrapper import CameraWanDiffusionWrapper  # noqa: E402
from utils.wan_5b_wrapper import WanTextEncoder, WanVAEWrapper  # noqa: E402

try:  # PIL is only needed for the I2V branch.
    from PIL import Image  # noqa: E402
except ImportError:  # pragma: no cover - PIL is in requirements but be defensive.
    Image = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
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
        # Selection priority: env var USE_EMA=1 forces EMA; otherwise prefer
        # the nominal `generator` weights, because EMA(decay=0.99) needs
        # ~1/(1-decay)=100s of steps before it represents a sane model — in
        # an early training run picking EMA gives a mixture of pretrained +
        # training weights that produces all-black / all-white videos.
        use_ema = os.environ.get("USE_EMA", "0") == "1"
        if isinstance(sd, dict):
            picked = None
            if use_ema and "generator_ema" in sd and isinstance(sd["generator_ema"], dict):
                picked = "generator_ema"; sd = sd["generator_ema"]
            elif "generator" in sd and isinstance(sd["generator"], dict):
                picked = "generator"; sd = sd["generator"]
            elif "generator_ema" in sd and isinstance(sd["generator_ema"], dict):
                picked = "generator_ema"; sd = sd["generator_ema"]
            elif "model" in sd and isinstance(sd["model"], dict):
                picked = "model"; sd = sd["model"]
            print(f"[infer] using checkpoint section: {picked!r} "
                  f"(set USE_EMA=1 to force EMA)")
        # Strip optional "model." prefix dumped by some FSDP routines.
        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in sd.items()
        }
        cleaned = {
            k.replace("_fsdp_wrapped_module.", "")
             .replace("_checkpoint_wrapped_module.", "")
             .replace("_orig_mod.", ""): v
            for k, v in cleaned.items()
        }
        missing, unexpected = gen.model.load_state_dict(cleaned, strict=False)
        print(f"[infer] missing={len(missing)}  unexpected={len(unexpected)}")
        if len(unexpected):
            print("  unexpected (first 5):", unexpected[:5])
    gen = gen.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return gen


def _load_image_as_pixel(
    image_path: str,
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load an image and return a [1, 3, 1, target_h, target_w] tensor in [-1, 1]
    suitable for ``WanVAEWrapper.encode_to_latent``.
    """
    if Image is None:
        raise RuntimeError(
            "PIL is required for I2V inference; please ``pip install pillow``."
        )
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_w, target_h), Image.BICUBIC)
    arr = torch.from_numpy(np.array(img, dtype=np.float32))  # (H, W, 3)
    arr = (arr / 127.5) - 1.0
    pixel = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    return pixel.to(device=device, dtype=dtype)


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

    # ---------- resolve generator ckpt ----------
    # Allow passing a checkpoint *directory* (e.g. ``logs/.../checkpoint_model_000200``);
    # auto-append ``/model.pt`` so users don't have to spell it out every time.
    ckpt = args.generator_ckpt or config.get("checkpoints", {}).get("generator_ckpt", "")
    if ckpt and not ckpt.endswith(".pt") and not ckpt.endswith(".pth"):
        candidate = os.path.join(ckpt, "model.pt")
        if os.path.isdir(ckpt) or os.path.exists(candidate):
            ckpt = candidate

    # ---------- resolve output dir ----------
    # If the ckpt path encodes a training step (e.g. ``checkpoint_model_000200``),
    # append that suffix to the user-supplied output_dir so different checkpoints
    # don't clobber each other (e.g. ``videos/camera_bidir`` -> ``videos/camera_bidir_000200``).
    base_out_dir = args.output_dir or config["logging"]["output_dir"]
    step_suffix = ""
    if ckpt:
        m = re.search(r"checkpoint_model_(\d+)", ckpt)
        if m:
            step_suffix = m.group(1)
    if step_suffix:
        out_dir = Path(base_out_dir) / step_suffix
    else:
        out_dir = Path(base_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # ---------- text + VAE ----------
    text_encoder = WanTextEncoder().eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    # ---------- generator ----------
    generator = load_generator(config, ckpt, device)

    # ---------- Optional ablation: disable the PRoPE residual branch ----
    # Set NO_PROPE=1 to zero out every block's `prope_o` and skip the PRoPE
    # path. This isolates whether the all-black/white outputs come from the
    # PRoPE residual (mis-conditioned cameras) or from the RoPE backbone
    # itself. If the video looks fine with NO_PROPE=1, the PRoPE branch is
    # the culprit; if it still looks broken, the issue is elsewhere
    # (e.g. checkpoint mismatch or backbone divergence).
    if os.environ.get("NO_PROPE", "0") == "1":
        n_disabled = 0
        for blk in generator.model.blocks:
            sa = blk.self_attn
            if hasattr(sa, "prope_o") and sa.prope_o is not None:
                sa.prope_o = None
                n_disabled += 1
        print(f"[infer] NO_PROPE=1: disabled prope_o on {n_disabled} blocks")

    # ---------- prompts and trajectories ----------
    with open(inf_cfg["prompt_path"]) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    with open(inf_cfg["trajectory_path"]) as f:
        trajectories = [ln.strip() for ln in f if ln.strip()]
    assert len(prompts) == len(trajectories), \
        f"prompts ({len(prompts)}) and trajectories ({len(trajectories)}) must align"

    # ---------- I2V switch -----------------------------------------------
    # ``algorithm.i2v: true`` (or top-level ``i2v: true`` after
    # normalize_config flattens it) turns this into image-to-video inference:
    # the source image is VAE-encoded into the first latent frame, that
    # frame is pinned across all denoising steps, and only frames >=1 are
    # sampled.  When False (default), behaves exactly like the original T2V
    # bidirectional + camera (PRoPE) inference.
    is_i2v = bool(
        config.get("i2v", False)
        or (config.get("algorithm", {}) or {}).get("i2v", False)
    )
    image_paths: list[str] = []
    if is_i2v:
        img_path_cfg = inf_cfg.get("image_path", None)
        assert img_path_cfg, (
            "I2V inference requires inference.image_path to point to a text "
            "file containing one source image path per line (aligned with "
            "prompts/trajectories)."
        )
        with open(img_path_cfg) as f:
            image_paths = [ln.strip() for ln in f if ln.strip()]
        assert len(image_paths) == len(prompts), (
            f"image_paths ({len(image_paths)}) must align with prompts "
            f"({len(prompts)}) for I2V inference."
        )
        print(f"[infer] I2V mode enabled, {len(image_paths)} source images "
              f"loaded from {img_path_cfg}")
    if args.max_clips > 0:
        prompts = prompts[: args.max_clips]
        trajectories = trajectories[: args.max_clips]
        if is_i2v:
            image_paths = image_paths[: args.max_clips]

    # ---------- shapes ----------
    F_lat = int(inf_cfg["num_latent_frames"])
    H_lat = int(inf_cfg["height_latent"])
    W_lat = int(inf_cfg["width_latent"])
    C_lat = int(inf_cfg.get("num_channels", 48))
    target_h = H_lat * 16
    target_w = W_lat * 16
    fps = int(inf_cfg.get("fps", 24))
    intrinsics_norm = torch.tensor([
        float(inf_cfg["fx_norm"]), float(inf_cfg["fy_norm"]),
        float(inf_cfg["cx_norm"]), float(inf_cfg["cy_norm"]),
    ], dtype=torch.float32).numpy()

    # ---------- scheduler ----------
    scheduler = generator.get_scheduler()
    sampling_steps = int(inf_cfg.get("sampling_steps", 50))
    scheduler.set_timesteps(sampling_steps, training=False)
    timesteps = scheduler.timesteps.to(device)

    # CFG negative prompt: use the configured negative prompt (matches minWM's
    # BidirectionalDiffusionInferencePipeline, which encodes
    # self.args.negative_prompt rather than an empty string). normalize_config
    # injects DEFAULT_NEGATIVE_PROMPT when none is provided in the config.
    neg_prompt = config.get("negative_prompt", "") or ""
    uncond = text_encoder([neg_prompt])
    # Generator runs in bfloat16; cast text embeddings to match its dtype to
    # avoid `mat1 and mat2 must have the same dtype` in text_embedding.
    uncond = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) else v)
              for k, v in uncond.items()}

    # ---------- generate ----------
    for clip_idx, (prompt, traj) in enumerate(zip(prompts, trajectories)):
        out_path = out_dir / f"clip_{clip_idx:03d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[clip {clip_idx}] skip (already exists -> {out_path})")
            continue

        gen = torch.Generator(device=device).manual_seed(args.seed + clip_idx)

        # Pose -> viewmats / Ks for this trajectory.
        intrinsics_clip, poses_clip = poses_from_pose_str(
            traj, F_lat, target_h, target_w)
        # The data-prep script returns intrinsics already normalized; we
        # override with config-provided defaults here so inference can vary
        # focal/principal-point at test time.
        viewmats, Ks = build_viewmats_and_Ks(intrinsics_norm, poses_clip)
        viewmats = torch.from_numpy(viewmats)[None].to(device=device, dtype=torch.float32)
        Ks = torch.from_numpy(Ks)[None].to(device=device, dtype=torch.float32)

        # Text conditioning (CFG). The unconditional embedding (uncond) is
        # prompt-independent and was computed once above.
        cond = text_encoder([prompt])
        cond = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) else v)
                for k, v in cond.items()}

        # Initial noise: [B=1, F_lat, C, H_lat, W_lat] (matches WanDiffusionWrapper input).
        x = torch.randn(
            (1, F_lat, C_lat, H_lat, W_lat),
            device=device, dtype=torch.bfloat16, generator=gen,
        )

        if os.environ.get("DEBUG_INFER", "0") == "1":
            print(f"[clip {clip_idx}] shapes: "
                  f"x={tuple(x.shape)} "
                  f"viewmats={tuple(viewmats.shape)} "
                  f"Ks={tuple(Ks.shape)} "
                  f"cond.keys={list(cond.keys())} "
                  f"is_i2v={is_i2v} cfg_scale={float(inf_cfg.get('guidance_scale',5.0))} "
                  f"shift={generator.scheduler.shift if hasattr(generator,'scheduler') else 'n/a'}")
            for k, v in cond.items():
                if torch.is_tensor(v):
                    print(f"    cond[{k}]: shape={tuple(v.shape)} dtype={v.dtype} "
                          f"|v|.mean={v.float().abs().mean().item():.3e} "
                          f"|v|.max={v.float().abs().max().item():.3e}")

        # ---- I2V: encode source image and pin its latent into x[:, :1] ----
        image_latent = None
        if is_i2v:
            img_pixel = _load_image_as_pixel(
                image_paths[clip_idx],
                target_h=target_h, target_w=target_w,
                device=device, dtype=torch.bfloat16,
            )
            image_latent = vae.encode_to_latent(img_pixel)  # [1, 1, C, H_lat, W_lat]
            assert image_latent.shape[1] == 1, (
                f"expected single-frame image latent, got "
                f"shape={tuple(image_latent.shape)}"
            )
            assert image_latent.shape[2:] == x.shape[2:], (
                f"image latent shape {tuple(image_latent.shape[2:])} does not "
                f"match noise latent shape {tuple(x.shape[2:])}"
            )
            image_latent = image_latent.to(dtype=torch.bfloat16)
            x[:, :1] = image_latent

        cfg_scale = float(inf_cfg.get("guidance_scale", 5.0))
        for ti, t_scalar in enumerate(timesteps):
            # Per-frame timestep ONLY for the model: in I2V mode, frame 0=0
            # tells the backbone "frame 0 is clean" (mirrors training-time
            # ``model/diffusion.py`` which zeroes timestep[:, :1] for the
            # context frame). The scheduler must NEVER see this t[:,0]=0
            # (see comment below).
            t_per_frame = t_scalar.expand(1, F_lat).to(device)
            if is_i2v:
                t_per_frame = t_per_frame.clone()
                t_per_frame[:, 0] = 0

            flow_c, _ = generator(
                noisy_image_or_video=x,
                conditional_dict=cond,
                timestep=t_per_frame,
                viewmats=viewmats,
                Ks=Ks,
            )
            flow_u, _ = generator(
                noisy_image_or_video=x,
                conditional_dict=uncond,
                timestep=t_per_frame,
                viewmats=viewmats,
                Ks=Ks,
            )
            flow_pred = flow_u + cfg_scale * (flow_c - flow_u)

            # FlowMatchScheduler.step expects 4-D tensors; flatten F.
            #
            # CRITICAL: feed the SCALAR denoising timestep `t_scalar` (broadcast
            # to every flattened row) into the scheduler, NOT the per-frame
            # timestep. ``FlowMatchScheduler.step`` uses argmin to map t -> sigma:
            #
            #     timestep_id = argmin(|self.timesteps - t|, dim=1)
            #     if (timestep_id + 1 >= len(self.timesteps)).any():
            #         sigma_ = 0
            #
            # In I2V mode frame 0 has t=0, which argmin maps to the LAST sigma
            # (sigma_min). The ``.any()`` check then forces ``sigma_=0`` for
            # the WHOLE batch, so every frame is single-stepped from the
            # current sigma straight to 0. The model's flow was estimated at
            # the current t (not at the final step), so this overshoots and
            # blows up the noisy frames -> VAE decodes them as all-black /
            # all-white. Mirrors official Wan2.2 ``WanTI2V.i2v``, which also
            # uses a scalar t for ``sample_scheduler.step``.
            x_flat = x.flatten(0, 1)              # (F, C, H, W)
            f_flat = flow_pred.flatten(0, 1)
            t_flat = t_scalar.to(device).expand(x_flat.shape[0])
            x_flat = scheduler.step(f_flat, t_flat, x_flat)
            x = x_flat.unflatten(0, x.shape[:2]).to(torch.bfloat16)

            # Re-pin the I2V context latent: scheduler.step still updates the
            # first frame (we now use the same scalar t for all frames), so
            # we must restore the encoded source image. Mirrors the official
            # Wan2.2 line ``latent = (1.-mask2)*z[0] + mask2*latent``.
            if is_i2v and image_latent is not None:
                x[:, :1] = image_latent

            # ---- Diagnostic: track flow / x stats per frame ------------
            # Train-vs-infer mismatch typically shows up as flow_pred or x
            # blowing up on non-context frames after a few denoising steps.
            # Frame 0 is pinned, so its stats stay flat; if non-zero frames
            # diverge wildly the cause is in the model output, not the
            # scheduler. Set DEBUG_INFER=1 to enable.
            if os.environ.get("DEBUG_INFER", "0") == "1" and ti % 5 == 0:
                with torch.no_grad():
                    fp = flow_pred.float()
                    xf = x.float()
                    f0 = fp[0, 0].abs().mean().item()
                    f1 = fp[0, 1:].abs().mean().item()
                    fmax = fp[0, 1:].abs().max().item()
                    x1 = xf[0, 1:].abs().mean().item()
                    xmax = xf[0, 1:].abs().max().item()
                print(f"[clip {clip_idx}] step {ti+1}/{sampling_steps} "
                      f"t={float(t_scalar):.2f} "
                      f"|flow_pred[f=0]|={f0:.3e} "
                      f"|flow_pred[f>=1]|={f1:.3e} (max {fmax:.3e}) "
                      f"|x[f>=1]|={x1:.3e} (max {xmax:.3e})")
            elif (ti + 1) % 10 == 0:
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
            if is_i2v:
                f.write(f"image:      {image_paths[clip_idx]}\n")

    print(f"\nAll clips written to: {out_dir}")


if __name__ == "__main__":
    main()
