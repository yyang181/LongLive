#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visualize a camera-aware sample with action overlay.

Supports two input modes (auto-detected from ``--lmdb_path``):

1. **LMDB mode** (default): reads VAE latents + intrinsics + poses from a
   camera-aware LMDB, decodes the Wan2.2 VAE latents back to pixel video,
   then composites a Genie-3-style WASD + joystick action overlay.

2. **ViPE results mode**: when the path points to a ViPE output directory
   (contains ``rgb/``, ``pose/``, ``intrinsics/`` sub-directories), reads
   the raw video and camera info directly — no VAE decoding needed.

Usage (from repo root):

    # LMDB mode:
    python scripts/visualize_lmdb_sample.py \
        --lmdb_path /nfs/yixinyang/code/LongLive/data/train/minWM-data/data

    # ViPE results mode (auto-detected):
    python scripts/visualize_lmdb_sample.py \
        --lmdb_path /nfs/yixinyang/code/LongLive/data/Sekai/vipe_results

    # Pick a specific sample instead of random:
    python scripts/visualize_lmdb_sample.py --lmdb_path <path> --sample_idx 42

    # Visualize multiple samples:
    python scripts/visualize_lmdb_sample.py --lmdb_path <path> --num_samples 5

LMDB schema (per sample ``<idx>``):
    latents_{idx}_data     : float16  (F_lat, 48, H_lat, W_lat)  — VAE latents
    prompts_{idx}_data     : utf-8 str                             — caption
    intrinsics_{idx}_data  : float32  (4,)    [fx_norm, fy_norm, cx_norm, cy_norm]
    poses_{idx}_data       : float32  (F_lat, 7)  [tx,ty,tz, qx,qy,qz,qw] w2c
    paths_{idx}_data       : utf-8 str         — original video path

ViPE results layout:
    <vipe_dir>/rgb/<clip_id>.mp4          — raw video
    <vipe_dir>/pose/<clip_id>.npz         — 'data': (T, 4, 4) c2w, 'inds': (T,)
    <vipe_dir>/intrinsics/<clip_id>.npz   — 'data': (T, 4) [fx,fy,cx,cy] px
"""

import argparse
import os
import random
import sys

# Make repo root importable so ``utils.*`` resolves regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lmdb
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

import utils.tv_io_patch  # noqa: F401 — patches torchvision.io.write_video if missing
from torchvision.io import write_video

from utils.camera_dataset import _get_array_shape, _retrieve_row, build_viewmats_and_Ks
from utils.action_overlay import apply_overlay


# ---------------------------------------------------------------------------
# LMDB reading
# ---------------------------------------------------------------------------
def read_lmdb_sample(lmdb_path: str, idx: int) -> dict:
    """Read a single sample (latents, prompts, intrinsics, poses, path) from LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    latents_shape = _get_array_shape(env, "latents")
    intrinsics_shape = _get_array_shape(env, "intrinsics")
    poses_shape = _get_array_shape(env, "poses")
    count = latents_shape[0]
    if idx >= count:
        raise ValueError(f"Sample index {idx} out of range (count={count})")

    latents = _retrieve_row(env, "latents", np.float16, idx, latents_shape[1:])
    prompts = _retrieve_row(env, "prompts", str, idx)
    intrinsics = _retrieve_row(env, "intrinsics", np.float32, idx, intrinsics_shape[1:])
    poses = _retrieve_row(env, "poses", np.float32, idx, poses_shape[1:])

    with env.begin() as txn:
        path_raw = txn.get(f"paths_{idx}_data".encode())
    path = path_raw.decode("utf-8") if path_raw is not None else "N/A"
    env.close()

    return {
        "latents": latents,
        "prompts": prompts,
        "intrinsics": intrinsics,
        "poses": poses,
        "path": path,
        "count": count,
    }


def get_lmdb_count(lmdb_path: str) -> int:
    """Return the number of samples in the LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    shape = _get_array_shape(env, "latents")
    env.close()
    return shape[0]


# ---------------------------------------------------------------------------
# ViPE results directory reading
# ---------------------------------------------------------------------------
def is_vipe_dir(path: str) -> bool:
    """Return True if *path* looks like a ViPE output directory."""
    return all(os.path.isdir(os.path.join(path, sub))
               for sub in ("rgb", "pose", "intrinsics"))


def list_vipe_clips(vipe_dir: str) -> list:
    """Return a sorted list of clip_ids (stems) available in a ViPE dir."""
    rgb_dir = os.path.join(vipe_dir, "rgb")
    clips = []
    for f in sorted(os.listdir(rgb_dir)):
        if f.endswith(".mp4"):
            clips.append(os.path.splitext(f)[0])
    return clips


def load_video_frames_uint8(video_path: str, max_frames: int = 0) -> np.ndarray:
    """Load video frames as (T, H, W, 3) uint8 array.

    Uses decord if available, otherwise falls back to OpenCV.
    """
    try:
        import decord
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        n = len(vr)
        if max_frames > 0:
            n = min(n, max_frames)
        frames = vr.get_batch(list(range(n)))  # (T, H, W, 3) uint8
        if isinstance(frames, np.ndarray):
            return frames
        return frames.numpy()
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        cap.release()
        return np.stack(frames)


def read_vipe_sample(vipe_dir: str, clip_id: str,
                     max_frames: int = 0) -> dict:
    """Read a single sample (video, intrinsics, poses) from a ViPE dir.

    Returns a dict with keys:
        video_hwc : (T, H, W, 3) uint8  — raw RGB frames
        intrinsics: (4,) float32        — [fx_norm, fy_norm, cx_norm, cy_norm]
        c2w       : (T, 4, 4) float32   — camera-to-world, frame-0 aligned
        path      : str                  — video path
        clip_id   : str
    """
    video_path = os.path.join(vipe_dir, "rgb", f"{clip_id}.mp4")
    pose_path = os.path.join(vipe_dir, "pose", f"{clip_id}.npz")
    intr_path = os.path.join(vipe_dir, "intrinsics", f"{clip_id}.npz")

    # ---- Load video ----
    video_hwc = load_video_frames_uint8(video_path, max_frames=max_frames)
    T, H, W = video_hwc.shape[:3]

    # ---- Load poses (c2w, 4×4) ----
    pz = np.load(pose_path, allow_pickle=True)
    c2w_all = np.asarray(pz["data"], dtype=np.float64)  # (T_total, 4, 4)
    if max_frames > 0:
        c2w_all = c2w_all[:max_frames]
    T_pose = c2w_all.shape[0]
    if T_pose < T:
        # Pad with last pose if video has more frames than poses
        c2w_all = np.concatenate(
            [c2w_all, np.tile(c2w_all[-1:], (T - T_pose, 1, 1))], axis=0)
    c2w_all = c2w_all[:T]

    # Frame-0 align: normalize so frame 0 is identity
    c2w_aligned = np.array(
        [np.linalg.inv(c2w_all[0]) @ c for c in c2w_all],
        dtype=np.float32)

    # ---- Load intrinsics [fx, fy, cx, cy] in pixel units ----
    iz = np.load(intr_path, allow_pickle=True)
    intr_all = np.asarray(iz["data"], dtype=np.float32)  # (T_total, 4)
    fx, fy, cx, cy = intr_all[0]  # first frame
    intrinsics = np.array(
        [fx / W, fy / H, cx / W, cy / H], dtype=np.float32)

    return {
        "video_hwc": video_hwc,
        "intrinsics": intrinsics,
        "c2w": c2w_aligned,
        "path": video_path,
        "clip_id": clip_id,
    }


# ---------------------------------------------------------------------------
# VAE decoding
# ---------------------------------------------------------------------------
def decode_latents_to_video(latents: np.ndarray, device: torch.device,
                            dtype: torch.dtype) -> np.ndarray:
    """Decode Wan2.2 VAE latents to a uint8 HWC video array.

    Args:
        latents: (F_lat, C, H_lat, W_lat) float16 numpy array.
        device, dtype: torch device / dtype for the VAE.

    Returns:
        (T, H, W, 3) uint8 numpy array.
    """
    from utils.wan_5b_wrapper import WanVAEWrapper

    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval()

    # (F, C, H, W) -> (1, F, C, H, W)
    latent_t = torch.from_numpy(latents.copy()).unsqueeze(0).to(
        device=device, dtype=dtype)

    with torch.no_grad():
        video = vae.decode_to_pixel(latent_t)  # (1, T, 3, H, W) in [-1, 1]

    video = (video * 0.5 + 0.5).clamp(0, 1)       # -> [0, 1]
    video = video[0].cpu()                         # (T, 3, H, W)
    video_uint8 = (video * 255.0).to(torch.uint8)
    video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, H, W, 3)
    return video_uint8.numpy()


# ---------------------------------------------------------------------------
# Pose interpolation (latent-frame → video-frame resolution)
# ---------------------------------------------------------------------------
VAE_TEMPORAL_STRIDE = 4  # Wan2.2 VAE: 4× temporal compression


def interpolate_c2w_poses(c2w: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate c2w poses from latent-frame resolution to video-frame resolution.

    The Wan2.2 VAE does 4× temporal upsampling, so latent frame *i* corresponds
    to video frame *i × 4*.  Poses stored in the LMDB are at latent resolution
    (``F_lat`` frames), but the decoded video has ``(F_lat - 1) × 4 + 1`` frames.
    Without interpolation the action overlay would only cover the first
    ``F_lat`` video frames, leaving the rest stuck on the last pose.

    Interpolation:
        * Translation — linear (``np.interp``).
        * Rotation    — SLERP (``scipy.spatial.transform.Slerp``).

    Args:
        c2w:        (F_lat, 4, 4) camera-to-world matrices.
        target_len: Desired number of output frames (decoded video length).

    Returns:
        (target_len, 4, 4) interpolated c2w matrices.
    """
    F_lat = c2w.shape[0]
    if target_len <= F_lat:
        return c2w[:target_len].copy()

    # Source timestamps in latent-frame units: [0, 1, 2, ..., F_lat-1].
    # Destination timestamps: linspace(0, F_lat-1, target_len) which places
    # samples at video-frame intervals (every 0.25 in latent units = every 1
    # video frame when stride = 4).
    src_times = np.arange(F_lat, dtype=np.float64)
    dst_times = np.linspace(0, F_lat - 1, target_len, dtype=np.float64)

    # --- Rotation: SLERP ---
    rotations = Rotation.from_matrix(c2w[:, :3, :3])
    slerp = Slerp(src_times, rotations)
    interp_rot_mats = slerp(dst_times).as_matrix()  # (target_len, 3, 3)

    # --- Translation: linear ---
    translations = c2w[:, :3, 3]  # (F_lat, 3)
    interp_trans = np.zeros((target_len, 3), dtype=c2w.dtype)
    for d in range(3):
        interp_trans[:, d] = np.interp(dst_times, src_times, translations[:, d])

    out = np.zeros((target_len, 4, 4), dtype=c2w.dtype)
    out[:, :3, :3] = interp_rot_mats
    out[:, :3, 3] = interp_trans
    out[:, 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Pose convention conversion (legacy c2w → w2c)
# ---------------------------------------------------------------------------
def invert_poses_c2w_to_w2c(poses_c2w: np.ndarray) -> np.ndarray:
    """Invert (T, 7) poses from c2w [tx,ty,tz, qx,qy,qz,qw] to w2c.

    Legacy VGGT-Omega LMDBs (built before the extrinsics convention fix)
    stored c2w poses mislabeled as w2c.  This function inverts each pose
    so ``build_viewmats_and_Ks`` receives the correct w2c input.
    """
    T = len(poses_c2w)
    poses_w2c = np.zeros_like(poses_c2w)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses_c2w[i]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w[:3, 3] = [tx, ty, tz]
        w2c = np.linalg.inv(c2w)
        poses_w2c[i, :3] = w2c[:3, 3]
        poses_w2c[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    return poses_w2c.astype(np.float32)


# ---------------------------------------------------------------------------
# Info text overlay
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_info_text(video_hwc: np.ndarray, prompt: str, intrinsics: np.ndarray,
                  path: str, sample_idx: int, n_poses: int) -> np.ndarray:
    """Stamp prompt / intrinsics / path text on the top of every frame."""
    T, H, W, _ = video_hwc.shape
    font = _load_font(max(14, int(H * 0.022)))
    line_h = int(H * 0.032)

    fx, fy, cx, cy = intrinsics
    info_lines = [
        f"Sample #{sample_idx}  |  Poses: {n_poses} frames",
        f"Prompt: {prompt[:150]}",
        f"Intrinsics (normalized): fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}",
        f"Path: {path[:150]}",
    ]

    out = video_hwc.copy()
    for t in range(T):
        frame = Image.fromarray(video_hwc[t])
        draw = ImageDraw.Draw(frame)
        y = 8
        for line in info_lines:
            # Shadow + text for readability over any background.
            draw.text((12, y + 1), line, fill=(0, 0, 0), font=font)
            draw.text((10, y), line, fill=(255, 255, 255), font=font)
            y += line_h
        out[t] = np.asarray(frame)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize a random LMDB sample with camera action overlay.")
    p.add_argument("--lmdb_path", type=str,
                   default="/nfs/yixinyang/code/LongLive/data/train/minWM-data/data",
                   help="Path to LMDB directory (contains data.mdb) OR a ViPE "
                        "results directory (contains rgb/, pose/, intrinsics/). "
                        "Auto-detected.")
    p.add_argument("--output_dir", type=str, default="./lmdb_vis",
                   help="Output directory for the visualization .mp4")
    p.add_argument("--sample_idx", type=int, default=None,
                   help="Specific sample index (random if omitted). "
                        "When --num_samples > 1, this is the starting index.")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of samples to visualize (default: 1). "
                        "When > 1 and --sample_idx is set, visualizes "
                        "[sample_idx, sample_idx+1, ...]. When > 1 and "
                        "--sample_idx is unset, picks that many random indices.")
    p.add_argument("--fps", type=int, default=24,
                   help="FPS for output video (default: 24)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Max frames to load from video in ViPE mode "
                        "(0 = all frames). Ignored in LMDB mode.")
    p.add_argument("--corner", type=str, default="bottom-left",
                   choices=["bottom-left", "bottom-right", "top-left", "top-right"],
                   help="Corner placement for the action overlay panel")
    p.add_argument("--no_overlay", action="store_true",
                   help="Skip action overlay (decode video only)")
    p.add_argument("--no_text", action="store_true",
                   help="Skip info text overlay")
    p.add_argument("--pose_convention", type=str, default="w2c",
                   choices=["w2c", "c2w"],
                   help="Convention of stored poses in the LMDB: 'w2c' "
                        "(world-to-camera, default, correct) or 'c2w' "
                        "(camera-to-world, legacy buggy VGGT-Omega LMDBs "
                        "built before the extrinsics fix). When 'c2w', "
                        "poses are inverted to w2c before visualization.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducible sample selection")
    return p.parse_args()


def visualize_one_sample(
    lmdb_path: str, output_dir: str, idx: int, args, device, dtype
) -> str:
    """Visualize a single LMDB sample and return the output video path."""
    print(f"\n{'='*60}")
    print(f"Sample index: {idx}")
    print(f"{'='*60}")

    # ---- Read sample ----
    print(f"Reading from {lmdb_path} ...")
    sample = read_lmdb_sample(lmdb_path, idx)
    print(f"  Latents shape : {sample['latents'].shape}  dtype={sample['latents'].dtype}")
    print(f"  Intrinsics    : {sample['intrinsics']}")
    print(f"  Poses shape   : {sample['poses'].shape}")
    print(f"  Prompt        : {sample['prompts'][:200]}")
    print(f"  Path          : {sample['path']}")

    # ---- Convert poses if legacy c2w convention ----
    poses = sample["poses"]
    if args.pose_convention == "c2w":
        print("  Pose convention: c2w (legacy) -> inverting to w2c")
        poses = invert_poses_c2w_to_w2c(poses)
    else:
        print("  Pose convention: w2c (default)")

    # ---- Decode latents ----
    print("Loading Wan2.2 VAE and decoding latents ...")
    video_hwc = decode_latents_to_video(sample["latents"], device, dtype)
    print(f"  Decoded video : {video_hwc.shape}  dtype={video_hwc.dtype}")

    # ---- Compute c2w from stored w2c poses ----
    viewmats, Ks = build_viewmats_and_Ks(sample["intrinsics"], poses)
    c2w = np.linalg.inv(viewmats)  # (F_lat, 4, 4) camera-to-world, frame-0 aligned

    # ---- Interpolate poses to match decoded video length ----
    video_len = video_hwc.shape[0]
    c2w_interp = interpolate_c2w_poses(c2w, video_len)
    print(f"  Interpolated poses: {c2w.shape[0]} -> {c2w_interp.shape[0]} "
          f"(stride={VAE_TEMPORAL_STRIDE})")

    # ---- Action overlay ----
    if not args.no_overlay:
        print("Applying action overlay ...")
        video_hwc = apply_overlay(video_hwc, c2w_interp, corner=args.corner)

    # ---- Info text ----
    if not args.no_text:
        print("Adding info text ...")
        video_hwc = add_info_text(
            video_hwc, sample["prompts"], sample["intrinsics"],
            sample["path"], idx, n_poses=len(sample["poses"]))

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"lmdb_sample_{idx}.mp4"
    out_path = os.path.join(output_dir, out_name)
    print(f"Writing video to {out_path} ...")
    video_tensor = torch.from_numpy(np.ascontiguousarray(video_hwc))
    write_video(out_path, video_tensor, fps=args.fps)

    print(f"  Output : {out_path}")
    print(f"  Shape  : {video_hwc.shape}")
    print(f"  FPS    : {args.fps}")
    print(f"  Frames : {video_hwc.shape[0]}")
    return out_path


def visualize_one_vipe_sample(
    vipe_dir: str, output_dir: str, clip_id: str, args
) -> str:
    """Visualize a single ViPE results sample and return the output path."""
    print(f"\n{'='*60}")
    print(f"ViPE clip: {clip_id}")
    print(f"{'='*60}")

    # ---- Read sample ----
    sample = read_vipe_sample(vipe_dir, clip_id, max_frames=args.max_frames)
    video_hwc = sample["video_hwc"]
    c2w = sample["c2w"]
    print(f"  Video shape  : {video_hwc.shape}  dtype={video_hwc.dtype}")
    print(f"  Intrinsics   : {sample['intrinsics']}")
    print(f"  c2w shape    : {c2w.shape}")
    print(f"  Path         : {sample['path']}")

    # ---- Action overlay ----
    # ViPE poses are c2w at video-frame resolution — no interpolation needed.
    if not args.no_overlay:
        print("Applying action overlay ...")
        video_hwc = apply_overlay(video_hwc, c2w, corner=args.corner)

    # ---- Info text ----
    if not args.no_text:
        print("Adding info text ...")
        video_hwc = add_info_text(
            video_hwc, f"[ViPE] {clip_id}", sample["intrinsics"],
            sample["path"], 0, n_poses=len(c2w))

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"vipe_{clip_id}.mp4"
    out_path = os.path.join(output_dir, out_name)
    print(f"Writing video to {out_path} ...")
    video_tensor = torch.from_numpy(np.ascontiguousarray(video_hwc))
    write_video(out_path, video_tensor, fps=args.fps)

    print(f"  Output : {out_path}")
    print(f"  Shape  : {video_hwc.shape}")
    print(f"  FPS    : {args.fps}")
    print(f"  Frames : {video_hwc.shape[0]}")
    return out_path


def main():
    args = parse_args()

    # Resolve to absolute *before* any chdir so paths stay valid.
    input_path = os.path.abspath(args.lmdb_path)
    output_dir = os.path.abspath(args.output_dir)

    if args.seed is not None:
        random.seed(args.seed)

    # ---- Auto-detect ViPE vs LMDB ----
    if is_vipe_dir(input_path):
        # ============ ViPE results mode ============
        print(f"[ViPE mode] Input: {input_path}")
        clips = list_vipe_clips(input_path)
        if not clips:
            print("ERROR: no .mp4 files found in rgb/", file=sys.stderr)
            sys.exit(1)
        count = len(clips)
        n = min(args.num_samples, count)
        if args.sample_idx is not None:
            indices = [args.sample_idx + i for i in range(n)
                       if args.sample_idx + i < count]
        else:
            indices = sorted(random.sample(range(count), n))
        selected = [clips[i] for i in indices]
        print(f"Total clips: {count}")
        print(f"Visualizing {len(selected)} clip(s): {selected}")

        output_paths = []
        for clip_id in selected:
            out_path = visualize_one_vipe_sample(
                input_path, output_dir, clip_id, args)
            output_paths.append(out_path)

        print(f"\n{'='*60}")
        print(f"Done! {len(output_paths)} video(s) written to {output_dir}:")
        for p in output_paths:
            print(f"  {p}")
        return

    # ============ LMDB mode (default) ============
    if not os.path.isfile(os.path.join(input_path, "data.mdb")):
        print(f"ERROR: not a ViPE dir or LMDB dir: {input_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[LMDB mode] Input: {input_path}")
    # The VAE loads weights from a relative path (wan_models/...), so we must
    # run from the repo root.
    os.chdir(_REPO_ROOT)

    # ---- Select sample indices ----
    count = get_lmdb_count(input_path)
    n = min(args.num_samples, count)
    if args.sample_idx is not None:
        indices = [args.sample_idx + i for i in range(n)
                   if args.sample_idx + i < count]
    else:
        indices = random.sample(range(count), n)
    print(f"Total samples: {count}")
    print(f"Visualizing {len(indices)} sample(s): {indices}")

    # ---- Shared VAE device/dtype ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    output_paths = []
    for idx in indices:
        out_path = visualize_one_sample(
            input_path, output_dir, idx, args, device, dtype)
        output_paths.append(out_path)
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Done! {len(output_paths)} video(s) written to {output_dir}:")
    for p in output_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
