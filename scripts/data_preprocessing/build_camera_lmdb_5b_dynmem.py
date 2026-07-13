#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B PRoPE Bidirectional SFT.

This variant processes DynamicMem-style metadata JSON (metadata_0710.json) where
each entry provides pre-computed camera intrinsics (.npz) and poses (.npz)
instead of WorldPlayGen-style pose strings.

For each input video clip, this script:
  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w>.
  2. Encodes them with the Wan2.2 VAE (4x temporal, 16x spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Loads camera intrinsics [fx, fy, cx, cy] from .npz, normalizes by the
     original video resolution: [fx/W_orig, fy/H_orig, cx/W_orig, cy/H_orig].
  4. Loads c2w (camera-to-world) 4x4 matrices from .npz, inverts to w2c,
     and stores them as (F_lat, 7) [tx, ty, tz, qx, qy, qz, qw].
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/`` so that the final layout is what
     ``utils.camera_dataset.CameraLatentLMDBDataset`` expects.

Input JSON format (single list of dicts, one per video):
    [
      {
        "file_path":      "/abs/or/rel/path/to/clip.mp4",
        "intrinsic_path": "/abs/or/rel/path/to/intrinsics.npz",
        "pose_path":      "/abs/or/rel/path/to/pose.npz",
        "text":           "caption text",
        "type":           "video"
      },
      ...
    ]

NPZ format (both intrinsic and pose):
  - key "data": array of camera parameters, indexed by "inds"
  - intrinsic data: shape (N, 4) = [fx, fy, cx, cy] in pixel space
  - pose data:     shape (N, 4, 4) = c2w (camera-to-world) 4x4 matrices

Camera poses are sub-sampled to match VAE latent frames:
  latent i <-> raw frame i * 4   (4x temporal downsampling)

Wan2.2 VAE has ratio (4x temporal, 16x spatial), so for ``F_lat = 20``:
    raw_frames  = (F_lat - 1) * 4 + 1 = 77
    latent_h    = target_h / 16
    latent_w    = target_w / 16

Usage (8-GPU example):
    torchrun --nproc_per_node=8 \
        scripts/data_preprocessing/build_camera_lmdb_5b_dynmem.py \
        --input_json   /path/to/metadata_0710.json \
        --output_dir   ./dataset/LongLive/CameraSFT \
        --target_h 704 --target_w 1280 --max_frames 77
"""

import argparse
import json
import os
import shutil
import sys
import time

# Make the repo root importable so ``utils.*`` resolves regardless of the
# directory torchrun is launched from (this file lives two levels below root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
# Camera parameter loading from DynamicMem npz files
# --------------------------------------------------------------------------
def poses_from_npz(intrinsic_path, pose_path, n_latent, orig_h, orig_w):
    """Load camera params from DynamicMem npz files and convert to LMDB format.

    Args:
        intrinsic_path: path to intrinsics .npz file
        pose_path:      path to pose .npz file
        n_latent:       number of latent frames (== number of poses to return)
        orig_h:         original video height (for intrinsic normalization)
        orig_w:         original video width  (for intrinsic normalization)

    Returns:
        intrinsics: (4,) float32 [fx/orig_w, fy/orig_h, cx/orig_w, cy/orig_h]
        poses:      (n_latent, 7) float32 [tx, ty, tz, qx, qy, qz, qw] w2c
    """
    # Load npz files
    intr_npz = np.load(intrinsic_path)
    pose_npz = np.load(pose_path)

    intr_data = intr_npz["data"]   # (N, 4) = [fx, fy, cx, cy]
    pose_data = pose_npz["data"]  # (N, 4, 4) = c2w

    # Select frame indices matching VAE latent frames.
    # VAE 4x temporal downsampling: latent i <-> raw frame i * 4
    frame_indices = np.arange(n_latent) * 4
    # Clamp to valid range (shouldn't be needed if video has >= max_frames)
    frame_indices = np.clip(frame_indices, 0, len(pose_data) - 1)

    # --- Intrinsics (use first selected frame; usually constant across video) ---
    fx, fy, cx, cy = intr_data[frame_indices[0]]
    intrinsics = np.array(
        [fx / orig_w, fy / orig_h, cx / orig_w, cy / orig_h],
        dtype=np.float32,
    )

    # --- Poses: c2w -> w2c -> [tx, ty, tz, qx, qy, qz, qw] ---
    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i, fi in enumerate(frame_indices):
        c2w = np.array(pose_data[fi], dtype=np.float64)  # use float64 for stable inv
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3].astype(np.float32)
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat().astype(np.float32)

    return intrinsics, poses


# --------------------------------------------------------------------------
# Frame loading (identical to build_camera_lmdb_5b.py)
# --------------------------------------------------------------------------
def load_video_frames(video_path, max_frames, target_h, target_w):
    """Load up to ``max_frames`` frames, resize to target_h x target_w.

    Returns:
        tensor: (C, F, H, W) float32 in [-1, 1] range, or None if too short.
        orig_h: original video height
        orig_w: original video width
    """
    if _USE_DECORD:
        vr = decord.VideoReader(video_path)
        if len(vr) < max_frames:
            return None, None, None
        frames = vr.get_batch(list(range(max_frames)))
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        orig_h, orig_w = frames.shape[1], frames.shape[2]
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)  # C,F,H,W
    else:
        cap = cv2.VideoCapture(video_path)
        buf = []
        for _ in range(max_frames):
            ok, fr = cap.read()
            if not ok:
                cap.release()
                return None, None, None
            buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.stack(buf)
        orig_h, orig_w = frames.shape[1], frames.shape[2]
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(tensor.shape[1], target_h, target_w),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
    tensor = (tensor / 255.0 - 0.5) * 2.0
    return tensor, orig_h, orig_w


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True,
                   help="JSON list of {file_path, intrinsic_path, pose_path, text, type}")
    p.add_argument("--video_dir", default="",
                   help="Optional root used to resolve relative file_path values")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=704)
    p.add_argument("--target_w", type=int, default=1280)
    p.add_argument("--max_frames", type=int, default=77,
                   help="raw frame count (must give integer F_lat under 4x temporal)")
    p.add_argument("--keep_shards", action="store_true",
                   help="Keep the transient per-rank shard dirs after merge "
                        "(debugging only). By default they are deleted; resume "
                        "relies on the merged 'data/' LMDB, not on the shards.")
    p.add_argument("--no_resume", action="store_true",
                   help="Wipe any existing output ('data/' + shards) and "
                        "reprocess everything from scratch.")
    return p.parse_args()


def _read_paths_from_lmdb(lmdb_dir):
    """Return (count, set_of_video_paths) stored in an existing LMDB.

    Works for both the merged ``data/`` dir and a per-rank shard dir. Returns
    ``(0, set())`` if the dir has no ``data.mdb`` yet.
    """
    if not os.path.isfile(os.path.join(lmdb_dir, "data.mdb")):
        return 0, set()
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    paths = set()
    count = 0
    with env.begin() as txn:
        cnt_raw = txn.get(b"__count__")
        if cnt_raw is not None:
            count = int(cnt_raw.decode())
        else:
            ls = txn.get(b"latents_shape")
            if ls is not None:
                count = int(ls.decode().split()[0])
        for j in range(count):
            p = txn.get(f"paths_{j}_data".encode())
            if p is not None:
                paths.add(p.decode("utf-8"))
    env.close()
    return count, paths


def _flush_pending_shards_into_final(output_dir, world_size, lat_shape_det,
                                     intr_shape_det, poses_shape_det,
                                     per_sample_bytes):
    """Merge any leftover ``.rank_*`` shards from a previous interrupted run
    into ``data/`` and delete them, so the current run can repartition the
    remaining clips cleanly.

    Only invoked on rank-0 before processing starts. Idempotent: if no shards
    exist, this is a no-op.
    """
    final_dir = os.path.join(output_dir, "data")
    pending = []
    for r in range(max(world_size, 1)):
        rd = os.path.join(output_dir, f".rank_{r}")
        if os.path.isfile(os.path.join(rd, "data.mdb")):
            cnt, _ = _read_paths_from_lmdb(rd)
            if cnt > 0:
                pending.append((rd, cnt))
    # Also catch shards from a previous run that used a different world_size.
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            if not name.startswith(".rank_"):
                continue
            rd = os.path.join(output_dir, name)
            if any(rd == p[0] for p in pending):
                continue
            if os.path.isfile(os.path.join(rd, "data.mdb")):
                cnt, _ = _read_paths_from_lmdb(rd)
                if cnt > 0:
                    pending.append((rd, cnt))

    if not pending:
        # Still clean up any empty leftover shard dirs so a fresh run starts tidy.
        if os.path.isdir(output_dir):
            for name in sorted(os.listdir(output_dir)):
                if name.startswith(".rank_"):
                    rd = os.path.join(output_dir, name)
                    shutil.rmtree(rd, ignore_errors=True)
        return

    new_total = sum(c for _, c in pending)
    base_count, existing_paths = _read_paths_from_lmdb(final_dir)
    print(f"[pre-merge] found {len(pending)} leftover rank shard(s) with "
          f"{new_total} samples; merging into {final_dir} (existing: {base_count}) "
          f"before resuming.")

    os.makedirs(final_dir, exist_ok=True)
    fmap = int((base_count + new_total) * per_sample_bytes * 1.3) + 1_000_000_000
    env = lmdb.open(final_dir, map_size=fmap, subdir=True)
    gi = base_count
    for rd, _ in tqdm(pending, desc="Pre-merge ranks"):
        renv = lmdb.open(rd, readonly=True, lock=False)
        with renv.begin() as rtxn:
            rc_raw = rtxn.get(b"__count__")
            rc = int(rc_raw.decode()) if rc_raw is not None else 0
            for j in range(rc):
                path = rtxn.get(f"paths_{j}_data".encode())
                path = path.decode("utf-8") if path is not None else None
                if path is not None and path in existing_paths:
                    continue
                lat = rtxn.get(f"latents_{j}_data".encode())
                cap = rtxn.get(f"prompts_{j}_data".encode())
                intr = rtxn.get(f"intrinsics_{j}_data".encode())
                pos = rtxn.get(f"poses_{j}_data".encode())
                if lat is None or cap is None or intr is None or pos is None:
                    # Partial / corrupt record from a hard kill — drop it.
                    continue
                with env.begin(write=True) as wtxn:
                    wtxn.put(f"latents_{gi}_data".encode(), lat)
                    wtxn.put(f"prompts_{gi}_data".encode(), cap)
                    wtxn.put(f"intrinsics_{gi}_data".encode(), intr)
                    wtxn.put(f"poses_{gi}_data".encode(), pos)
                    if path is not None:
                        wtxn.put(f"paths_{gi}_data".encode(), path.encode("utf-8"))
                if path is not None:
                    existing_paths.add(path)
                gi += 1
        renv.close()

    total = gi
    with env.begin(write=True) as txn:
        txn.put(b"__count__", str(total).encode())
        txn.put(b"latents_shape",
                f"{total} {' '.join(map(str, lat_shape_det))}".encode())
        txn.put(b"prompts_shape", f"{total}".encode())
        txn.put(b"intrinsics_shape",
                f"{total} {' '.join(map(str, intr_shape_det))}".encode())
        txn.put(b"poses_shape",
                f"{total} {' '.join(map(str, poses_shape_det))}".encode())
    env.sync(); env.close()
    print(f"[pre-merge] data/ now has {total} samples "
          f"({total - base_count} added from leftover shards).")

    # Delete every .rank_* dir in output_dir (including empties / mismatched ws).
    for name in sorted(os.listdir(output_dir)):
        if name.startswith(".rank_"):
            rd = os.path.join(output_dir, name)
            shutil.rmtree(rd, ignore_errors=True)
    print("[pre-merge] removed leftover rank shard dirs.")


def _resolve_path(path_str: str, video_dir: str, input_json: str) -> str:
    """Resolve a file path from the JSON entry (works for video, intrinsic, pose)."""
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    candidates = []
    if video_dir:
        candidates.append(os.path.join(video_dir, path_str))
    candidates.append(path_str)
    candidates.append(os.path.join(os.path.dirname(input_json), path_str))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else path_str


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
    lat_shape_det = (n_latent, 48, h_lat, w_lat)
    intr_shape_det = (4,)
    poses_shape_det = (n_latent, 7)
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
        resolved_video = _resolve_path(
            entry["file_path"], args.video_dir, args.input_json)
        resolved_intr = _resolve_path(
            entry["intrinsic_path"], args.video_dir, args.input_json)
        resolved_pose = _resolve_path(
            entry["pose_path"], args.video_dir, args.input_json)
        if (os.path.exists(resolved_video)
                and os.path.exists(resolved_intr)
                and os.path.exists(resolved_pose)):
            item = dict(entry)
            item["file_path"] = resolved_video
            item["intrinsic_path"] = resolved_intr
            item["pose_path"] = resolved_pose
            valid.append(item)
    if global_rank == 0:
        print(f"Valid videos (with intrinsics + poses): "
              f"{len(valid)}/{len(data_list)}")

    # Init Wan2.2 VAE
    from utils.wan_5b_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "data")
    rank_dir = os.path.join(args.output_dir, f".rank_{global_rank}")

    # ---- Optional clean slate ----
    if args.no_resume:
        if global_rank == 0 and os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        if os.path.exists(rank_dir):
            shutil.rmtree(rank_dir)
        if world_size > 1:
            torch.distributed.barrier()
    else:
        # ---- Pre-merge any leftover .rank_* shards from a previous run ----
        if global_rank == 0:
            _flush_pending_shards_into_final(
                args.output_dir, world_size,
                lat_shape_det, intr_shape_det, poses_shape_det,
                per_sample_bytes,
            )
        if world_size > 1:
            torch.distributed.barrier()

    # ---- Resume: read the merged-done set (identical on every rank) ----
    done_paths = set()
    if not args.no_resume:
        _, merged_paths = _read_paths_from_lmdb(final_dir)
        done_paths |= merged_paths

    # ---- Filter out already-done clips BEFORE sharding ----
    remaining = [it for it in valid if it["file_path"] not in done_paths]
    if global_rank == 0:
        print(f"[resume] {len(done_paths)} clips already in merged data/; "
              f"{len(remaining)} clips still to process "
              f"(re-partitioning across {world_size} GPU(s)).")

    # Shard the *remaining* work.
    if world_size > 1:
        per_gpu = (len(remaining) + world_size - 1) // world_size
        shard = remaining[global_rank * per_gpu:(global_rank + 1) * per_gpu]
    else:
        shard = remaining

    os.makedirs(rank_dir, exist_ok=True)
    rank_map = int(max(len(shard), 1) * per_sample_bytes * 1.3) + 100_000_000
    rank_env = lmdb.open(rank_dir, map_size=rank_map, subdir=True)

    # Defensive: if for any reason this rank's freshly-created shard already
    # has entries (should not happen because pre-merge wipes all shards),
    # continue appending instead of overwriting.
    count = 0
    with rank_env.begin() as txn:
        cnt_raw = txn.get(b"__count__")
        if cnt_raw is not None:
            count = int(cnt_raw.decode())
            for j in range(count):
                p = txn.get(f"paths_{j}_data".encode())
                if p is not None:
                    done_paths.add(p.decode("utf-8"))

    errors = 0
    first_shape = None
    t0 = time.time()
    pbar = tqdm(shard, desc=f"GPU{local_rank}", disable=(global_rank != 0),
                dynamic_ncols=True)
    for idx, item in enumerate(pbar):
        if item["file_path"] in done_paths:
            pbar.set_postfix(ok=count, err=errors, skip=len(done_paths))
            continue
        try:
            frames, orig_h, orig_w = load_video_frames(
                item["file_path"], args.max_frames,
                args.target_h, args.target_w)
            if frames is None:
                errors += 1; continue
            pixel = frames.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode_to_latent(pixel)  # (1, F_lat, 48, h, w)
            latent_np = latent[0].float().cpu().numpy().astype(np.float16)
            intrinsics, poses = poses_from_npz(
                item["intrinsic_path"], item["pose_path"],
                n_latent, orig_h, orig_w)
        except Exception as e:
            print(f"[GPU{local_rank}] failed {item.get('file_path')}: {e}")
            errors += 1; continue

        with rank_env.begin(write=True) as txn:
            txn.put(f"latents_{count}_data".encode(), latent_np.tobytes())
            txn.put(f"prompts_{count}_data".encode(),
                    item["text"].encode("utf-8"))
            txn.put(f"intrinsics_{count}_data".encode(), intrinsics.tobytes())
            txn.put(f"poses_{count}_data".encode(), poses.tobytes())
            # Stored so a future run knows this clip is already processed.
            txn.put(f"paths_{count}_data".encode(),
                    item["file_path"].encode("utf-8"))
            # Persist running count immediately so an interrupted run is resumable.
            txn.put(b"__count__", str(count + 1).encode())

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
        txn.put(b"__lat_shape__", " ".join(map(str, lat_shape_det)).encode())
        txn.put(b"__intr_shape__", " ".join(map(str, intr_shape_det)).encode())
        txn.put(b"__poses_shape__", " ".join(map(str, poses_shape_det)).encode())
    rank_env.sync(); rank_env.close()
    print(f"GPU{local_rank}: {count} OK, {errors} errors, {time.time()-t0:.0f}s")

    # ---- Phase 2: rank-0 appends new shard samples into the merged data/ ----
    if world_size > 1:
        torch.distributed.barrier()

    if global_rank == 0:
        # New samples produced across all shards this run.
        new_total = 0
        for r in range(world_size):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            nc, _ = _read_paths_from_lmdb(rd)
            new_total += nc
        lat_shape, intr_shape, poses_shape = (
            lat_shape_det, intr_shape_det, poses_shape_det)

        # Existing merged records (authoritative across runs) — we append after.
        base_count, existing_paths = _read_paths_from_lmdb(final_dir)

        if new_total == 0:
            print(f"No new samples; merged data/ already has {base_count}.")
            if not args.keep_shards:
                for r in range(world_size):
                    rd = os.path.join(args.output_dir, f".rank_{r}")
                    if os.path.exists(rd):
                        shutil.rmtree(rd)
            if world_size > 1:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            return

        print(f"Appending up to {new_total} new samples to "
              f"{base_count} existing ones ...")
        os.makedirs(final_dir, exist_ok=True)
        fmap = int((base_count + new_total) * per_sample_bytes * 1.3) + 1_000_000_000
        env = lmdb.open(final_dir, map_size=fmap, subdir=True)
        gi = base_count
        for r in tqdm(range(world_size), desc="Merge ranks"):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            if not os.path.exists(rd):
                continue
            renv = lmdb.open(rd, readonly=True, lock=False)
            with renv.begin() as rtxn:
                rc_raw = rtxn.get(b"__count__")
                rc = int(rc_raw.decode()) if rc_raw is not None else 0
                for j in range(rc):
                    path = rtxn.get(f"paths_{j}_data".encode())
                    path = path.decode("utf-8") if path is not None else None
                    # Skip clips already present in data/ (idempotent append).
                    if path is not None and path in existing_paths:
                        continue
                    lat = rtxn.get(f"latents_{j}_data".encode())
                    cap = rtxn.get(f"prompts_{j}_data".encode())
                    intr = rtxn.get(f"intrinsics_{j}_data".encode())
                    pos = rtxn.get(f"poses_{j}_data".encode())
                    with env.begin(write=True) as wtxn:
                        wtxn.put(f"latents_{gi}_data".encode(), lat)
                        wtxn.put(f"prompts_{gi}_data".encode(), cap)
                        wtxn.put(f"intrinsics_{gi}_data".encode(), intr)
                        wtxn.put(f"poses_{gi}_data".encode(), pos)
                        if path is not None:
                            wtxn.put(f"paths_{gi}_data".encode(), path.encode("utf-8"))
                    if path is not None:
                        existing_paths.add(path)
                    gi += 1
            renv.close()
            if not args.keep_shards:
                shutil.rmtree(rd)

        total = gi  # base_count + actually appended
        with env.begin(write=True) as txn:
            txn.put(b"__count__", str(total).encode())
            txn.put(b"latents_shape",
                    f"{total} {' '.join(map(str, lat_shape))}".encode())
            txn.put(b"prompts_shape", f"{total}".encode())
            txn.put(b"intrinsics_shape",
                    f"{total} {' '.join(map(str, intr_shape))}".encode())
            txn.put(b"poses_shape",
                    f"{total} {' '.join(map(str, poses_shape))}".encode())
        env.sync(); env.close()
        print(f"Done! {total} samples ({total - base_count} new) -> {final_dir}")

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
