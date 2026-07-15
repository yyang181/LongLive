#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB from MIND action.json controls.

MIND action.json stores one ``ws/ad/ud/lr`` control state per raw video frame.
This builder converts those controls to the same local OpenCV camera trajectory
used by ``build_camera_lmdb_5b.py`` (W/S along +/-Z, A/D along +/-X, look
controls as +/- yaw/pitch), then subsamples the trajectory at the Wan VAE's
4-frame temporal stride. ``camera_pos/camera_rpy`` and ``actor_pos/actor_rpy``
are intentionally not used, allowing first- and third-person clips to share
the exact same control-to-camera representation.

MIND action values follow the official evaluator: 0=no-op, 1=forward/left/up,
2=backward/right/down (the direction depends on the field).
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
# MIND action.json loading and action -> camera trajectory
# --------------------------------------------------------------------------

def _rot_x(theta):
    c, sn = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]], dtype=np.float64)


def _rot_y(theta):
    c, sn = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]], dtype=np.float64)


def _generate_camera_trajectory_local(motions):
    """Mirror the generic builder's local OpenCV trajectory integration."""
    poses = [np.eye(4, dtype=np.float64)]
    T = np.eye(4, dtype=np.float64)
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0.0, 0.0])
        poses.append(T.copy())
    return poses


def _action_to_motion(frame, forward_speed, yaw_speed, pitch_speed):
    """Decode one MIND frame using the official 0/1/2 action encoding."""
    values = {name: int(frame[name]) for name in ("ws", "ad", "ud", "lr")}
    invalid = {name: value for name, value in values.items()
               if value not in (0, 1, 2)}
    if invalid:
        raise ValueError(f"Invalid MIND action values {invalid}; expected 0, 1, or 2")

    move = {}
    if values["ws"] == 1:
        move["forward"] = forward_speed
    elif values["ws"] == 2:
        move["forward"] = -forward_speed
    if values["ad"] == 1:
        move["right"] = -forward_speed
    elif values["ad"] == 2:
        move["right"] = forward_speed
    if values["ud"] == 1:
        move["pitch"] = pitch_speed
    elif values["ud"] == 2:
        move["pitch"] = -pitch_speed
    if values["lr"] == 1:
        move["yaw"] = -yaw_speed
    elif values["lr"] == 2:
        move["yaw"] = yaw_speed
    return move


def poses_from_mind_actions(frames, forward_speed=0.08,
                            yaw_speed_deg=3.0, pitch_speed_deg=3.0):
    """Return one local c2w pose per raw frame from MIND controls.

    As in the generic pose-string builder, action at frame *t* advances the
    camera from pose *t* to pose *t+1*. The first pose is identity.
    """
    yaw_speed = np.deg2rad(yaw_speed_deg)
    pitch_speed = np.deg2rad(pitch_speed_deg)
    motions = [
        _action_to_motion(frame, forward_speed, yaw_speed, pitch_speed)
        for frame in frames
    ]
    poses = _generate_camera_trajectory_local(motions)
    return np.stack(poses[:len(frames)], axis=0).astype(np.float32)


def _path_matches_split(path, split):
    """Keep only MIND ``1st_data/<split>`` and ``3rd_data/<split>`` files."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    for perspective in ("1st_data", "3rd_data"):
        if perspective in parts:
            i = parts.index(perspective)
            return i + 1 < len(parts) and parts[i + 1] == split
    # Also support a direct camera_dir such as .../3rd_data/train.
    return True


def _sample_key(sample_dir, root):
    return os.path.normpath(os.path.relpath(sample_dir, root))


def load_mind_action_dir(camera_dir, split="train", forward_speed=0.08,
                          yaw_speed_deg=3.0, pitch_speed_deg=3.0):
    """Scan MIND action files and create action-derived local c2w trajectories.

    Keys are paths relative to ``camera_dir`` (e.g. ``1st_data/train/data-75``),
    preventing collisions between first- and third-person directories.
    """
    root = os.path.abspath(camera_dir)
    pattern = os.path.join(root, "**", "action.json")
    out = {}
    skipped = 0
    for action_path in sorted(glob.glob(pattern, recursive=True)):
        if not _path_matches_split(action_path, split):
            continue
        sample_dir = os.path.dirname(action_path)
        if os.path.basename(sample_dir) == "":
            continue
        key = _sample_key(sample_dir, root)
        try:
            with open(action_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            frames = meta.get("data", [])
            if not frames:
                raise ValueError("empty data array")
            for i, frame in enumerate(frames):
                missing = {"ws", "ad", "ud", "lr"}.difference(frame)
                if missing:
                    raise ValueError(f"frame {i} missing {sorted(missing)}")
            out[key] = {
                "pose": poses_from_mind_actions(
                    frames, forward_speed, yaw_speed_deg, pitch_speed_deg),
                "total_time": int(meta.get("total_time", len(frames))),
                "clip_id": os.path.basename(sample_dir),
                "perspective": "1st" if "1st_data" in key.split(os.sep) else "3rd",
                "caption": meta.get("caption", ""),
            }
        except Exception as e:
            print(f"[WARN] skipping action.json {action_path}: {e}", flush=True)
            skipped += 1
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
                   default=("A high-quality gameplay video with detailed 3D "
                            "environments, smooth character animation, and "
                            "dynamic camera movement."),
                   help="Caption used when no CSV caption is available.")
    p.add_argument("--camera_fov", type=float, default=90.0,
                   help="Horizontal field of view in degrees (UE5 default 90). "
                        "Used to synthesize normalized intrinsics.")
    p.add_argument("--orig_w", type=int, default=1920,
                   help="Original capture width for intrinsic normalization.")
    p.add_argument("--orig_h", type=int, default=1080,
                   help="Original capture height for intrinsic normalization.")
    p.add_argument("--split", default="train",
                   help="MIND split to include when scanning a root (default: train).")
    p.add_argument("--forward_speed", type=float, default=0.08,
                   help="Camera translation per active MIND frame in metres.")
    p.add_argument("--yaw_speed_deg", type=float, default=3.0,
                   help="Camera yaw change per active MIND frame in degrees.")
    p.add_argument("--pitch_speed_deg", type=float, default=3.0,
                   help="Camera pitch change per active MIND frame in degrees.")
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

    # ---- Load action-derived camera metadata ----
    if global_rank == 0:
        print(f"Loading MIND ws/ad/ud/lr actions from: {args.camera_dir}")
        print(f"  split={args.split}, forward={args.forward_speed}m/frame, "
              f"yaw={args.yaw_speed_deg}deg/frame, pitch={args.pitch_speed_deg}deg/frame")
    cam_table = load_mind_action_dir(
        args.camera_dir, split=args.split,
        forward_speed=args.forward_speed,
        yaw_speed_deg=args.yaw_speed_deg,
        pitch_speed_deg=args.pitch_speed_deg,
    )
    if global_rank == 0:
        n_3rd = sum(1 for v in cam_table.values() if v["perspective"] == "3rd")
        n_1st = len(cam_table) - n_3rd
        print(f"Loaded {len(cam_table)} action.json files "
              f"({n_3rd} 3rd-person, {n_1st} 1st-person).")

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

    # ---- Build valid clips using relative sample keys ----
    raw_frames_needed = args.max_frames
    valid = []
    missing_camera = 0
    too_short = 0
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "**", "video.mp4"),
                                   recursive=True))
    for vp in video_files:
        sample_dir = os.path.dirname(vp)
        if not _path_matches_split(vp, args.split):
            continue
        key = _sample_key(sample_dir, os.path.abspath(args.video_dir))
        if key not in cam_table:
            missing_camera += 1
            continue
        cam = cam_table[key]
        if cam["pose"].shape[0] < raw_frames_needed:
            too_short += 1
            continue
        caption = cap_table.get(cam["clip_id"], cam.get("caption") or args.default_caption)
        valid.append({
            "clip_id": key,
            "video_path": vp,
            "caption": caption,
            "c2w": cam["pose"],
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
