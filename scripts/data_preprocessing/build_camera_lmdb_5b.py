#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B PRoPE Bidirectional SFT.

For each input video clip, this script:
  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w>.
  2. Encodes them with the Wan2.2 VAE (4× temporal, 16× spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Parses a WorldPlayGen-style camera-trajectory string into a list of
     per-frame c2w SE(3) matrices, inverts them to w2c, and stores them as
     (F_lat, 7) [tx, ty, tz, qx, qy, qz, qw].
  4. Stores normalized intrinsics [fx/W, fy/H, cx/W, cy/H] (4,) float32.
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/`` so that the final layout is what
     ``utils.camera_dataset.CameraLatentLMDBDataset`` expects.

Input JSON format (single list of dicts, one per video):
    [
      {
        "video_path": "/abs/or/rel/path/to/clip.mp4",
        "caption":    "text prompt",
        "pose_str":   "w-4, a-7, up-3, ..."          # WorldPlayGen grammar
      },
      ...
    ]

Wan2.2 VAE has ratio (4× temporal, 16× spatial), so for ``F_lat = 20``:
    raw_frames  = (F_lat - 1) * 4 + 1 = 77
    latent_h    = target_h / 16
    latent_w    = target_w / 16

Usage (8-GPU example):
    torchrun --nproc_per_node=8 \
        scripts/data_preprocessing/build_camera_lmdb_5b.py \
        --input_json   /path/to/clips.json \
        --output_dir   ./dataset/LongLive/CameraSFT \
        --target_h 704 --target_w 1280 --max_frames 77
"""

import argparse
import json
import os
import shutil
import time

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm


try:
    import decord; decord.bridge.set_bridge("torch"); _USE_DECORD = True
except ImportError:
    _USE_DECORD = False
    import cv2

# --------------------------------------------------------------------------
# WorldPlayGen-style camera-trajectory parsing (kept identical to minWM)
# --------------------------------------------------------------------------
def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _generate_camera_trajectory_local(motions):
    poses = [np.eye(4)]
    T = np.eye(4)
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward != 0:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right != 0:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            T = T @ (np.linalg.inv(c_origin) @ C)
        poses.append(T.copy())
    return poses


def _parse_pose_string(pose_string,
                       forward_speed=0.08,
                       yaw_speed_deg=3.0,
                       pitch_speed_deg=3.0):
    yaw_speed = np.deg2rad(yaw_speed_deg)
    pitch_speed = np.deg2rad(pitch_speed_deg)
    motions = []
    for cmd in [c.strip() for c in pose_string.split(",")]:
        if not cmd:
            continue
        action, num = cmd.split("-")
        action = action.strip()
        n = int(float(num.strip()))
        for _ in range(n):
            if action == "w":
                motions.append({"forward": forward_speed})
            elif action == "s":
                motions.append({"forward": -forward_speed})
            elif action == "a":
                motions.append({"right": -forward_speed})
            elif action == "d":
                motions.append({"right": forward_speed})
            elif action == "up":
                motions.append({"pitch": pitch_speed})
            elif action == "down":
                motions.append({"pitch": -pitch_speed})
            elif action == "left":
                motions.append({"yaw": -yaw_speed})
            elif action == "right":
                motions.append({"yaw": yaw_speed})
            else:
                raise ValueError(f"Unknown camera action: {action}")
    return motions


def poses_from_pose_str(pose_str, n_latent, target_h, target_w):
    """Returns (intrinsics(4,), poses(F_lat, 7)) for a single clip."""
    motions = _parse_pose_string(pose_str)
    c2w_list = _generate_camera_trajectory_local(motions)
    if len(c2w_list) < n_latent:
        # pad by repeating the last pose
        c2w_list = c2w_list + [c2w_list[-1]] * (n_latent - len(c2w_list))
    # Default WorldPlayGen-style intrinsics for 1920x1080 capture (f≈969.7).
    fx = 969.6969696969696
    fy = 969.6969696969696
    cx = target_w / 2.0
    cy = target_h / 2.0
    intrinsics = np.array(
        [fx / target_w, fy / target_h, cx / target_w, cy / target_h],
        dtype=np.float32,
    )
    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i in range(n_latent):
        c2w = np.array(c2w_list[i])
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    return intrinsics, poses


# --------------------------------------------------------------------------
# Frame loading
# --------------------------------------------------------------------------
def load_video_frames(video_path, max_frames, target_h, target_w):
    if _USE_DECORD:
        vr = decord.VideoReader(video_path)
        if len(vr) < max_frames:
            return None
        frames = vr.get_batch(list(range(max_frames)))
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)  # C,F,H,W
    else:
        cap = cv2.VideoCapture(video_path)
        buf = []
        for _ in range(max_frames):
            ok, fr = cap.read()
            if not ok:
                cap.release()
                return None
            buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        tensor = torch.from_numpy(np.stack(buf)).float().permute(3, 0, 1, 2)
    if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(tensor.shape[1], target_h, target_w),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
    tensor = (tensor / 255.0 - 0.5) * 2.0
    return tensor


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True,
                   help="JSON list of {video_path, caption, pose_str}")
    p.add_argument("--video_dir", default="",
                   help="Optional root used to resolve relative video_path values")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=704)
    p.add_argument("--target_w", type=int, default=1280)
    p.add_argument("--max_frames", type=int, default=77,
                   help="raw frame count (must give integer F_lat under 4x temporal)")
    return p.parse_args()


def _resolve_video_path(video_path: str, video_dir: str, input_json: str) -> str:
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path
    candidates = []
    if video_dir:
        candidates.append(os.path.join(video_dir, video_path))
    candidates.append(video_path)
    candidates.append(os.path.join(os.path.dirname(input_json), video_path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else video_path


def main():
    args = parse_args()
    if (args.max_frames - 1) % 4 != 0:
        raise ValueError(
            f"max_frames={args.max_frames} is invalid for Wan2.2 VAE; "
            "expected 4*k+1 raw frames so camera poses match latent frames."
        )
    if args.target_h % 16 != 0 or args.target_w % 16 != 0:
        raise ValueError(
            f"target_h/target_w must be divisible by 16, got "
            f"{args.target_h}x{args.target_w}."
        )
    n_latent = (args.max_frames - 1) // 4 + 1   # Wan2.2 VAE: 4x temporal
    h_lat = args.target_h // 16
    w_lat = args.target_w // 16
    per_sample_bytes = (
        n_latent * 48 * h_lat * w_lat * 2  # latent fp16
        + 4 * 4                            # intrinsics
        + n_latent * 7 * 4                 # poses
        + 4096                             # caption + overhead
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        import datetime
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=4))
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0
    device = torch.device(f"cuda:{local_rank}")

    # Load entries
    with open(args.input_json) as f:
        data_list = json.load(f)
    if global_rank == 0:
        print(f"Loaded {len(data_list)} clips from {args.input_json}")
        print(f"F_lat={n_latent}  H_lat={h_lat}  W_lat={w_lat}")

    # Validate and resolve paths
    valid = []
    for entry in data_list:
        resolved = _resolve_video_path(
            entry["video_path"], args.video_dir, args.input_json)
        if os.path.exists(resolved):
            item = dict(entry)
            item["video_path"] = resolved
            valid.append(item)
    if global_rank == 0:
        print(f"Valid videos: {len(valid)}/{len(data_list)}")

    # Shard
    if world_size > 1:
        per_gpu = (len(valid) + world_size - 1) // world_size
        shard = valid[global_rank * per_gpu:(global_rank + 1) * per_gpu]
    else:
        shard = valid

    # Init Wan2.2 VAE
    from utils.wan_5b_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    rank_dir = os.path.join(args.output_dir, f".rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)
    rank_map = int(len(shard) * per_sample_bytes * 1.3) + 100_000_000
    rank_env = lmdb.open(rank_dir, map_size=rank_map, subdir=True)

    count = 0
    errors = 0
    first_shape = None
    t0 = time.time()
    pbar = tqdm(shard, desc=f"GPU{local_rank}", disable=(global_rank != 0),
                dynamic_ncols=True)
    for idx, item in enumerate(pbar):
        try:
            frames = load_video_frames(
                item["video_path"], args.max_frames,
                args.target_h, args.target_w)
            if frames is None:
                errors += 1; continue
            pixel = frames.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode_to_latent(pixel)  # (1, F_lat, 48, h, w)
            latent_np = latent[0].float().cpu().numpy().astype(np.float16)
            intrinsics, poses = poses_from_pose_str(
                item["pose_str"], n_latent, args.target_h, args.target_w)
        except Exception as e:
            print(f"[GPU{local_rank}] failed {item.get('video_path')}: {e}")
            errors += 1; continue

        with rank_env.begin(write=True) as txn:
            txn.put(f"latents_{count}_data".encode(), latent_np.tobytes())
            txn.put(f"prompts_{count}_data".encode(),
                    item["caption"].encode("utf-8"))
            txn.put(f"intrinsics_{count}_data".encode(), intrinsics.tobytes())
            txn.put(f"poses_{count}_data".encode(), poses.tobytes())

        if first_shape is None:
            first_shape = (latent_np.shape, intrinsics.shape, poses.shape)
            if global_rank == 0:
                print(f"\nFirst sample: latent={latent_np.shape}, "
                      f"intrinsics={intrinsics}, poses={poses.shape}")

        del latent, latent_np, intrinsics, poses, pixel, frames
        torch.cuda.empty_cache()
        count += 1
        elapsed = time.time() - t0
        speed = (idx + 1) / max(elapsed, 1e-3)
        pbar.set_postfix(ok=count, err=errors, speed=f"{speed:.1f}it/s")

    with rank_env.begin(write=True) as txn:
        txn.put(b"__count__", str(count).encode())
        if first_shape:
            txn.put(b"__lat_shape__", " ".join(map(str, first_shape[0])).encode())
            txn.put(b"__intr_shape__", " ".join(map(str, first_shape[1])).encode())
            txn.put(b"__poses_shape__", " ".join(map(str, first_shape[2])).encode())
    rank_env.sync(); rank_env.close()
    print(f"GPU{local_rank}: {count} OK, {errors} errors, {time.time()-t0:.0f}s")

    # ---- Phase 2: rank-0 merges all shards ----
    if world_size > 1:
        torch.distributed.barrier()

    if global_rank == 0:
        total = 0
        lat_shape = intr_shape = poses_shape = None
        for r in range(world_size):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            if not os.path.exists(rd):
                continue
            renv = lmdb.open(rd, readonly=True, lock=False)
            with renv.begin() as txn:
                total += int(txn.get(b"__count__").decode())
                if lat_shape is None:
                    ls = txn.get(b"__lat_shape__")
                    if ls:
                        lat_shape = tuple(map(int, ls.decode().split()))
                        intr_shape = tuple(map(int,
                            txn.get(b"__intr_shape__").decode().split()))
                        poses_shape = tuple(map(int,
                            txn.get(b"__poses_shape__").decode().split()))
            renv.close()

        if total == 0:
            print("No valid samples produced.")
            return

        print(f"Merging {total} samples from {world_size} ranks ...")
        final_dir = os.path.join(args.output_dir, "data")
        os.makedirs(final_dir, exist_ok=True)
        fmap = int(total * per_sample_bytes * 1.3) + 1_000_000_000
        env = lmdb.open(final_dir, map_size=fmap, subdir=True)
        gi = 0
        for r in tqdm(range(world_size), desc="Merge ranks"):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            if not os.path.exists(rd):
                continue
            renv = lmdb.open(rd, readonly=True, lock=False)
            with renv.begin() as rtxn:
                rc = int(rtxn.get(b"__count__").decode())
                for j in range(rc):
                    lat = rtxn.get(f"latents_{j}_data".encode())
                    cap = rtxn.get(f"prompts_{j}_data".encode())
                    intr = rtxn.get(f"intrinsics_{j}_data".encode())
                    pos = rtxn.get(f"poses_{j}_data".encode())
                    with env.begin(write=True) as wtxn:
                        wtxn.put(f"latents_{gi}_data".encode(), lat)
                        wtxn.put(f"prompts_{gi}_data".encode(), cap)
                        wtxn.put(f"intrinsics_{gi}_data".encode(), intr)
                        wtxn.put(f"poses_{gi}_data".encode(), pos)
                    gi += 1
            renv.close()
            shutil.rmtree(rd)

        with env.begin(write=True) as txn:
            txn.put(b"latents_shape",
                    f"{total} {' '.join(map(str, lat_shape))}".encode())
            txn.put(b"prompts_shape", f"{total}".encode())
            txn.put(b"intrinsics_shape",
                    f"{total} {' '.join(map(str, intr_shape))}".encode())
            txn.put(b"poses_shape",
                    f"{total} {' '.join(map(str, poses_shape))}".encode())
        env.sync(); env.close()
        print(f"Done! {total} samples -> {final_dir}")

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
