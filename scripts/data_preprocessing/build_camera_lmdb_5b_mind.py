#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B PRoPE Bidirectional SFT
from the **MIND** dataset (CSU-JPG/MIND) action.json release.

MIND stores per-clip metadata in ``action.json`` (one per ``data-<id>/``
directory) with the following schema (3rd-person mode):

    {
        "total_time":  [int]   total frames of the ground-truth video
        "mark_time":   [int]   divider between memory context and prediction
        "data": [
            {
                "time":       [int]   frame index
                "ws/ad/ud/lr":[int]   action flags
                "actor_pos":  {x, y, z}        character world position
                "actor_rpy":  {x, y, z}        character Euler angles (deg)
                "camera_pos": {x, y, z}        camera world position  (3rd-person)
                "camera_rpy": {x, y, z}        camera Euler angles (deg)  (3rd-person)
            },
            ...
        ]
    }

Key differences vs. ``build_camera_lmdb_5b_sekai_original.py``:
  * **No NPZ camera files.** Camera extrinsics are derived from the per-frame
    ``camera_pos`` / ``camera_rpy`` fields in action.json (3rd-person) or
    ``actor_pos`` / ``actor_rpy`` (1st-person fallback).
  * **No intrinsics provided.** MIND is captured in Unreal Engine 5 at
    1920x1080; we synthesize normalized intrinsics from a configurable
    horizontal FOV (default 90 deg, UE5's standard).
  * **No caption field.** The action.json schema lists ``caption`` in the
    README but the released files do not contain one.  A high-quality default
    prompt is used instead (configurable via ``--default_caption``).
  * **UE5 units (cm).** Positions are in centimetres; ``--pose_scale``
    (default 0.01) converts to metres so the translation magnitude is
    comparable to other datasets.

For each clip, this script:
  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w>.
  2. Encodes them with the Wan2.2 VAE (4x temporal, 16x spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Subsamples the per-frame c2w trajectory to F_lat poses using
     ``cam_sample_strategy='last'`` (raw indices [0, 4, ..., (F_lat-1)*4]),
     inverts to w2c, and packs as (F_lat, 7) [tx, ty, tz, qx, qy, qz, qw].
  4. Stores normalized intrinsics (4,) float32 [fx/W, fy/H, cx/W, cy/H].
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/``.

Wan2.2 VAE has ratio (4x temporal, 16x spatial), so for ``F_lat = 40``:
    raw_frames  = (F_lat - 1) * 4 + 1 = 157
    latent_h    = target_h / 16
    latent_w    = target_w / 16

Usage:
    torchrun --nproc_per_node=4 \
        scripts/data_preprocessing/build_camera_lmdb_5b_mind.py \
        --video_dir       /path/to/MIND/3rd_data/train \
        --camera_dir      /path/to/MIND/3rd_data/train \
        --output_dir      ./data/train/MIND/ \
        --target_h 448 --target_w 832 --max_frames 157
"""

import argparse
import glob
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
# MIND action.json loading
# --------------------------------------------------------------------------
def load_mind_camera_dir(camera_dir, pose_scale=0.01):
    """Scan *camera_dir* (recursively) for per-clip ``data-<id>/action.json``
    files in the MIND layout.

    Each action.json has a ``data`` array of per-frame dicts with:
        camera_pos : {x, y, z}  (3rd-person; absent in 1st-person)
        camera_rpy : {x, y, z}  Euler angles in degrees (3rd-person)
        actor_pos  : {x, y, z}  (always present)
        actor_rpy  : {x, y, z}  (always present)

    For 3rd-person clips we use ``camera_pos`` / ``camera_rpy``; for
    1st-person clips (no camera_* fields) we fall back to ``actor_pos`` /
    ``actor_rpy``.

    Returns a dict ``clip_id -> {pose: (T, 4, 4) c2w float32,
                                 total_time: int,
                                 has_camera: bool}``.
    """
    out = {}
    skipped = 0
    pattern = os.path.join(camera_dir, "**", "action.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        clip_id = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r") as f:
                meta = json.load(f)
            frames = meta.get("data", [])
            total_time = int(meta.get("total_time", len(frames)))
            if not frames:
                skipped += 1
                continue

            # Detect whether this clip has 3rd-person camera fields.
            has_camera = ("camera_pos" in frames[0]
                          and "camera_rpy" in frames[0])

            T = len(frames)
            c2w_seq = np.zeros((T, 4, 4), dtype=np.float32)
            for i, fr in enumerate(frames):
                if has_camera:
                    pos = fr["camera_pos"]
                    rpy = fr["camera_rpy"]
                else:
                    pos = fr["actor_pos"]
                    rpy = fr["actor_rpy"]

                tx = float(pos["x"]) * pose_scale
                ty = float(pos["y"]) * pose_scale
                tz = float(pos["z"]) * pose_scale

                roll  = float(rpy["x"])
                pitch = float(rpy["y"])
                yaw   = float(rpy["z"])

                # UE5 intrinsic rotation order: Yaw(Z) -> Pitch(Y) -> Roll(X),
                # which is equivalent to extrinsic 'xyz' with [roll, pitch, yaw].
                R = Rotation.from_euler(
                    "xyz", [roll, pitch, yaw], degrees=True
                ).as_matrix().astype(np.float32)

                c2w_seq[i, :3, :3] = R
                c2w_seq[i, :3, 3] = [tx, ty, tz]
                c2w_seq[i, 3, 3] = 1.0

            out[clip_id] = {
                "pose": c2w_seq,
                "total_time": total_time,
                "has_camera": has_camera,
            }
        except Exception as e:
            print(f"[WARN] skipping action.json {clip_id}: {e}", flush=True)
            skipped += 1
            continue

    if skipped:
        print(f"[WARN] {skipped} action.json file(s) skipped.", flush=True)
    return out


def load_caption_csvs(csv_paths):
    """Merge one or more CSV files with ``videoFile`` and ``caption`` columns.

    Returns a dict ``clip_id -> caption`` (clip_id = videoFile without
    extension).
    """
    import csv as csv_mod
    out = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                vf = row.get("videoFile", "").strip()
                if not vf:
                    continue
                clip_id = os.path.splitext(vf)[0]
                cap = row.get("caption", "").strip()
                if not cap:
                    continue
                out[clip_id] = cap
    return out


# --------------------------------------------------------------------------
# Default intrinsics from FOV
# --------------------------------------------------------------------------
def default_intrinsics_from_fov(fov_deg, orig_w=1920, orig_h=1080):
    """Compute normalized intrinsics [fx/W, fy/H, cx/W, cy/H] from a
    horizontal FOV (degrees) assuming square pixels and a centered principal
    point.

    For fov_h = 90 deg at 1920x1080:
        fx_norm = 1 / (2 * tan(45)) = 0.5
        fy_norm = fx_norm * (W/H)   = 0.5 * 1920/1080 = 0.889
        cx_norm = 0.5
        cy_norm = 0.5
    """
    fx_norm = 1.0 / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy_norm = fx_norm * (float(orig_w) / float(orig_h))
    cx_norm = 0.5
    cy_norm = 0.5
    return np.array([fx_norm, fy_norm, cx_norm, cy_norm], dtype=np.float32)


# --------------------------------------------------------------------------
# Camera subsampling (vae_time_stride=4, per-raw-frame c2w from action.json)
# --------------------------------------------------------------------------
def _build_time_indices(n_latent, vae_time_stride, strategy):
    """Mirror SANA's ``SanaWMZipLatentDataset.cam_sample_strategy`` on the
    Wan2.2 4x temporal grid."""
    if strategy == "last":
        idxs = [i * vae_time_stride for i in range(n_latent)]
    elif strategy == "first":
        idxs = [0] + [i * vae_time_stride - vae_time_stride + 1
                      for i in range(1, n_latent)]
    else:
        raise ValueError(f"Invalid cam_sample_strategy: {strategy!r} "
                         f"(expected 'first' or 'last').")
    return idxs


def poses_from_c2w_array(c2w_seq, n_latent, intrinsics,
                         vae_time_stride=4, cam_sample_strategy="last"):
    """Subsample a (L, 4, 4) c2w trajectory to ``n_latent`` w2c poses and
    return the already-normalized intrinsics as a (4,) vector.

    Args:
        c2w_seq:              (L, 4, 4) float32, per-raw-frame c2w.
        n_latent:             number of latent frames after the 4x temporal VAE.
        intrinsics:           (4,) float32, already-normalized
                              [fx/W, fy/H, cx/W, cy/H].
        vae_time_stride:      temporal compression factor of the VAE (4).
        cam_sample_strategy:  'last' (default) or 'first'.

    Returns:
        intrinsics: (4,) float32, [fx/W, fy/H, cx/W, cy/H].
        poses:      (n_latent, 7) float32, w2c [tx, ty, tz, qx, qy, qz, qw].
    """
    L = int(c2w_seq.shape[0])
    idxs = _build_time_indices(n_latent, int(vae_time_stride),
                               cam_sample_strategy)
    # Clamp every index into the available trajectory length.
    idxs = [max(0, min(int(fi), L - 1)) for fi in idxs]

    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i, fi in enumerate(idxs):
        c2w = np.asarray(c2w_seq[fi], dtype=np.float64)
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
    p.add_argument("--video_dir", required=True,
                   help="Root directory containing data-<id>/ sub-folders "
                        "with video.mp4 files (searched recursively).")
    p.add_argument("--camera_dir", required=True,
                   help="Root directory containing data-<id>/ sub-folders "
                        "with action.json files (searched recursively). "
                        "Usually the same as --video_dir.")
    p.add_argument("--caption_csv", nargs="*", default=None,
                   help="Optional CSV file(s) with 'videoFile' and 'caption' "
                        "columns. When omitted or a clip is missing, "
                        "--default_caption is used.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=448)
    p.add_argument("--target_w", type=int, default=832)
    p.add_argument("--max_frames", type=int, default=157,
                   help="raw frame count (must give integer F_lat under 4x temporal)")
    p.add_argument("--cam_sample_strategy", choices=("first", "last"),
                   default="last",
                   help="Per-latent-frame anchor selection strategy. "
                        "'last' (default) matches sekai_game/sekai_original.")
    p.add_argument("--default_caption", type=str,
                   default=("A high-quality third-person gameplay video "
                            "with detailed 3D environments, smooth character "
                            "animation, and dynamic camera movement."),
                   help="Caption used when no CSV caption is available.")
    p.add_argument("--camera_fov", type=float, default=90.0,
                   help="Horizontal field of view in degrees (UE5 default 90). "
                        "Used to synthesize normalized intrinsics.")
    p.add_argument("--orig_w", type=int, default=1920,
                   help="Original capture width for intrinsic normalization.")
    p.add_argument("--orig_h", type=int, default=1080,
                   help="Original capture height for intrinsic normalization.")
    p.add_argument("--pose_scale", type=float, default=0.01,
                   help="Scale factor applied to camera positions (UE5 cm -> m).")
    p.add_argument("--keep_shards", action="store_true",
                   help="Keep the transient per-rank shard dirs after merge.")
    p.add_argument("--no_resume", action="store_true",
                   help="Wipe any existing output and reprocess from scratch.")
    return p.parse_args()


def _read_paths_from_lmdb(lmdb_dir):
    """Return (count, set_of_video_paths) stored in an existing LMDB."""
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
            pt = txn.get(f"paths_{j}_data".encode())
            if pt is not None:
                paths.add(pt.decode("utf-8"))
    env.close()
    return count, paths


def _flush_pending_shards_into_final(output_dir, world_size, lat_shape_det,
                                     intr_shape_det, poses_shape_det,
                                     per_sample_bytes):
    """Merge any leftover ``.rank_*`` shards from a previous interrupted run
    into ``data/`` and delete them."""
    final_dir = os.path.join(output_dir, "data")
    pending = []
    for r in range(max(world_size, 1)):
        rd = os.path.join(output_dir, f".rank_{r}")
        if os.path.isfile(os.path.join(rd, "data.mdb")):
            cnt, _ = _read_paths_from_lmdb(rd)
            if cnt > 0:
                pending.append((rd, cnt))
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

    for name in sorted(os.listdir(output_dir)):
        if name.startswith(".rank_"):
            rd = os.path.join(output_dir, name)
            shutil.rmtree(rd, ignore_errors=True)
    print("[pre-merge] removed leftover rank shard dirs.")


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

    # ---- Load camera metadata from action.json files ----
    if global_rank == 0:
        print(f"Loading MIND action.json cameras from: {args.camera_dir}")
        print(f"  pose_scale={args.pose_scale} (UE5 cm -> m)")
    cam_table = load_mind_camera_dir(args.camera_dir, pose_scale=args.pose_scale)
    if global_rank == 0:
        n_3rd = sum(1 for v in cam_table.values() if v["has_camera"])
        n_1st = len(cam_table) - n_3rd
        print(f"Loaded {len(cam_table)} action.json files "
              f"({n_3rd} 3rd-person, {n_1st} 1st-person fallback).")

    # ---- Load optional captions ----
    cap_table = {}
    if args.caption_csv:
        cap_table = load_caption_csvs(args.caption_csv)
        if global_rank == 0:
            print(f"Loaded {len(cap_table)} captions from "
                  f"{len(args.caption_csv)} CSV file(s).")
    if global_rank == 0:
        if not cap_table:
            print(f"No caption CSV provided; using default caption: "
                  f"\"{args.default_caption[:80]}...\"")

    # ---- Compute default normalized intrinsics ----
    intrinsics_default = default_intrinsics_from_fov(
        args.camera_fov, args.orig_w, args.orig_h)
    if global_rank == 0:
        print(f"Default normalized intrinsics (fov={args.camera_fov}deg, "
              f"orig={args.orig_w}x{args.orig_h}): {intrinsics_default}")

    # ---- Build the valid clip list (video ∩ camera) ----
    raw_frames_needed = args.max_frames
    valid = []
    missing_camera = 0
    too_short = 0
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"),
                                   recursive=True))
    for vp in video_files:
        clip_id = os.path.basename(os.path.dirname(vp))
        if clip_id not in cam_table:
            missing_camera += 1
            continue
        cam = cam_table[clip_id]
        # Need at least raw_frames_needed poses so cam_sample_strategy='last'
        # index (F_lat-1)*4 = max_frames - 1 is in range.
        if cam["pose"].shape[0] < raw_frames_needed:
            too_short += 1
            continue
        # Determine caption: CSV if available, else default.
        caption = cap_table.get(clip_id, args.default_caption)
        valid.append({
            "clip_id":    clip_id,
            "video_path": vp,
            "caption":    caption,
            "c2w":        cam["pose"],
        })

    if global_rank == 0:
        print(f"Valid clips (video ∩ camera): {len(valid)}")
        print(f"  missing camera: {missing_camera}")
        print(f"  too short (< {raw_frames_needed} frames): {too_short}")
        print(f"F_lat={n_latent}  H_lat={h_lat}  W_lat={w_lat}")

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
    remaining = [it for it in valid if it["video_path"] not in done_paths]
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

    # Defensive: continue appending if shard already has entries.
    count = 0
    with rank_env.begin() as txn:
        cnt_raw = txn.get(b"__count__")
        if cnt_raw is not None:
            count = int(cnt_raw.decode())
            for j in range(count):
                pt = txn.get(f"paths_{j}_data".encode())
                if pt is not None:
                    done_paths.add(pt.decode("utf-8"))

    errors = 0
    first_shape = None
    t0 = time.time()
    pbar = tqdm(shard, desc=f"GPU{local_rank}", disable=(global_rank != 0),
                dynamic_ncols=True)
    for idx, item in enumerate(pbar):
        if item["video_path"] in done_paths:
            pbar.set_postfix(ok=count, err=errors, skip=len(done_paths))
            continue
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
            intrinsics, poses = poses_from_c2w_array(
                item["c2w"], n_latent, intrinsics_default,
                vae_time_stride=4,
                cam_sample_strategy=args.cam_sample_strategy)
        except Exception as e:
            print(f"[GPU{local_rank}] failed {item.get('video_path')}: {e}")
            errors += 1; continue

        with rank_env.begin(write=True) as txn:
            txn.put(f"latents_{count}_data".encode(), latent_np.tobytes())
            txn.put(f"prompts_{count}_data".encode(),
                    item["caption"].encode("utf-8"))
            txn.put(f"intrinsics_{count}_data".encode(), intrinsics.tobytes())
            txn.put(f"poses_{count}_data".encode(), poses.tobytes())
            txn.put(f"paths_{count}_data".encode(),
                    item["video_path"].encode("utf-8"))
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
        new_total = 0
        for r in range(world_size):
            rd = os.path.join(args.output_dir, f".rank_{r}")
            nc, _ = _read_paths_from_lmdb(rd)
            new_total += nc
        lat_shape, intr_shape, poses_shape = (
            lat_shape_det, intr_shape_det, poses_shape_det)

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

        total = gi
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
