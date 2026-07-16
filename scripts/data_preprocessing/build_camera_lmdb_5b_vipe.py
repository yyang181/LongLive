#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B from ViPE results.

ViPE (NVIDIA VIdeo Pose Estimation) outputs per-frame camera poses and
intrinsics for each video.  This script:

  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w> from each video.
  2. Encodes them with the Wan2.2 VAE (4× temporal, 16× spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Reads ViPE camera poses (c2w, per-frame) and subsamples every 4 frames
     to match the VAE temporal stride.  Converts c2w → w2c and stores as
     (F_lat, 7) [tx,ty,tz, qx,qy,qz,qw] — **identical format** to
     ``build_camera_lmdb_5b.py`` (minWM string-based builder).
  4. Reads ViPE intrinsics [fx,fy,cx,cy] (pixel units), normalizes by the
     original video resolution, and stores as (4,) float32.
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/``.

ViPE results layout::

    <vipe_dir>/pose/<clip_id>.npz        — 'data': (T, 4, 4) c2w, 'inds': (T,)
    <vipe_dir>/intrinsics/<clip_id>.npz  — 'data': (T, 4) [fx,fy,cx,cy] px
    <vipe_dir>/rgb/<clip_id>.mp4         — (optional, if --video_dir is the ViPE rgb dir)

Pose convention (CRITICAL):
  * ViPE ``pose/data`` stores **c2w** (camera-to-world) 4×4 matrices.
  * The LMDB must store **w2c** (world-to-camera) [tx,ty,tz, qx,qy,qz,qw],
    matching ``build_camera_lmdb_5b.py`` and ``build_viewmats_and_Ks``.
  * Conversion: ``w2c = inv(c2w)``, then extract translation + quaternion
    from the w2c matrix.

Usage::
    torchrun --nproc_per_node=4 \\
        scripts/data_preprocessing/build_camera_lmdb_5b_vipe.py \\
        --video_dir    /path/to/Sekai/video \\
        --camera_dir   /path/to/Sekai/vipe_results/pose \\
        --caption_csv  /path/to/Sekai-Game.csv /path/to/Sekai-Real-HQ.csv \\
        --output_dir   ./data/train/sekai_vipe/ \\
        --target_h 704 --target_w 1280 --max_frames 957
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

# Make the repo root importable so ``utils.*`` resolves regardless of the
# directory torchrun is launched from.
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
# ViPE camera loading
# --------------------------------------------------------------------------
def load_vipe_cameras(camera_dir, intrinsics_dir=None):
    """Scan *camera_dir* for ViPE ``<clip_id>.npz`` pose files.

    Each pose NPZ has:
        data  : (T, 4, 4) float32 — c2w (camera-to-world)
        inds  : (T,) int64        — frame indices

    Each intrinsics NPZ (in *intrinsics_dir*) has:
        data  : (T, 4) float32 — [fx, fy, cx, cy] in pixel units
        inds  : (T,) int64

    Args:
        camera_dir:     Directory containing ``<clip_id>.npz`` pose files.
        intrinsics_dir: Directory containing ``<clip_id>.npz`` intrinsics
                        files.  If None, auto-derived from *camera_dir* by
                        replacing the last ``pose`` component with
                        ``intrinsics``.

    Returns:
        dict ``clip_id -> {c2w: (T, 4, 4), intrinsics: (T, 4)}``.
    """
    # Auto-derive intrinsics dir if not given.
    if intrinsics_dir is None:
        if os.path.basename(camera_dir.rstrip("/")) == "pose":
            intrinsics_dir = os.path.join(
                os.path.dirname(camera_dir.rstrip("/")), "intrinsics")
        else:
            intrinsics_dir = camera_dir  # fallback: same dir

    out = {}
    skipped = 0
    pose_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    for path in pose_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            pz = np.load(path, allow_pickle=True)
            c2w = np.asarray(pz["data"], dtype=np.float32)  # (T, 4, 4) c2w
        except Exception as e:
            print(f"[WARN] skipping corrupted pose NPZ {stem}: {e}",
                  flush=True)
            skipped += 1
            continue

        # Load intrinsics (optional — fall back to None if missing).
        intr_path = os.path.join(intrinsics_dir, f"{stem}.npz")
        intr = None
        if os.path.isfile(intr_path):
            try:
                iz = np.load(intr_path, allow_pickle=True)
                intr = np.asarray(iz["data"], dtype=np.float32)  # (T, 4)
            except Exception:
                pass

        out[stem] = {
            "c2w": c2w,
            "intrinsics": intr,
        }

    if skipped:
        print(f"[WARN] {skipped} corrupted pose NPZ file(s) skipped.",
              flush=True)
    return out


# --------------------------------------------------------------------------
# Camera subsampling (c2w → w2c, 4× temporal stride)
# --------------------------------------------------------------------------
def poses_from_vipe_c2w(c2w_seq, n_latent, intrinsics_row,
                         orig_w, orig_h):
    """Subsample a ViPE c2w trajectory to ``n_latent`` w2c poses.

    ViPE was run at interval=1 (every frame), so we subsample every 4
    frames (``vae_time_stride=4``) to match the Wan2.2 VAE 4× temporal
    compression.  Pose indices: ``[0, 4, 8, ..., 4*(n_latent-1)]``.

    **Convention**: ViPE ``data`` is c2w (camera-to-world).  We invert to
    w2c (world-to-camera) and store as ``[tx,ty,tz, qx,qy,qz,qw]`` —
    identical to ``build_camera_lmdb_5b.py`` (minWM string-based).

    Args:
        c2w_seq:        (T, 4, 4) float32, per-frame c2w (camera-to-world).
        n_latent:       number of latent frames (= number of camera poses).
        intrinsics_row: (4,) [fx, fy, cx, cy] in pixel units at orig resolution.
        orig_w/orig_h:  original video resolution (used to normalize intrinsics).

    Returns:
        intrinsics: (4,) float32, normalized [fx/W, fy/H, cx/W, cy/H].
        poses:      (n_latent, 7) float32, w2c [tx,ty,tz, qx,qy,qz,qw].
    """
    L = int(c2w_seq.shape[0])
    vae_time_stride = 4  # Wan2.2 VAE: 4× temporal compression
    idxs = [min(i * vae_time_stride, L - 1) for i in range(n_latent)]

    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i, fi in enumerate(idxs):
        c2w = np.asarray(c2w_seq[fi], dtype=np.float64)
        w2c = np.linalg.inv(c2w)  # c2w → w2c
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()

    fx, fy, cx, cy = intrinsics_row
    intrinsics = np.array(
        [fx / orig_w, fy / orig_h, cx / orig_w, cy / orig_h],
        dtype=np.float32,
    )
    return intrinsics, poses


# --------------------------------------------------------------------------
# Caption loading (CSV or JSON)
# --------------------------------------------------------------------------
def load_caption_csvs(csv_paths, use_parent_as_clip_id=False):
    """Load captions from CSV/JSON files with flexible dataset schemas."""
    def text(value):
        if value is None: return ""
        if isinstance(value, (list, tuple)): value = " ".join(str(x) for x in value if x is not None)
        return str(value).strip()
    def clip_id(value):
        value = text(value).replace("\\", "/")
        if not value: return ""
        base = os.path.basename(value.rstrip("/"))
        # minWM captions point to <clip_id>/gen.mp4 even when the processed
        # videos are flattened to <clip_id>.mp4; recognize that convention
        # independently of the video directory layout.
        if (use_parent_as_clip_id or os.path.splitext(base)[0].lower() == "gen") and os.path.splitext(base)[1]:
            return os.path.basename(os.path.dirname(value.rstrip("/")))
        return os.path.splitext(base)[0]
    out = {}
    def consume(entry, key_hint=""):
        if not isinstance(entry, dict): return
        path = (entry.get("video_path") or entry.get("videoFile") or entry.get("video") or entry.get("path") or entry.get("file") or entry.get("filename") or key_hint)
        cap = entry.get("caption") if "caption" in entry else entry.get("text", entry.get("prompt", entry.get("description", "")))
        cid, cap = clip_id(path), text(cap)
        if cid and cap: out[cid] = cap
    for path in csv_paths:
        if os.path.splitext(path)[1].lower() == ".json":
            with open(path, "r") as f: data = json.load(f)
            if isinstance(data, list):
                for entry in data: consume(entry)
            elif isinstance(data, dict):
                wrapped = next((data[k] for k in ("data", "items", "annotations", "clips") if isinstance(data.get(k), list)), None)
                if wrapped is not None:
                    for entry in wrapped: consume(entry)
                else:
                    for key, value in data.items():
                        if isinstance(value, dict): consume(value, key)
                        else:
                            cid, cap = clip_id(key), text(value)
                            if cid and cap: out[cid] = cap
        else:
            with open(path, newline="") as f:
                for row in csv.DictReader(f): consume(row, row.get("videoFile", ""))
    return out


# --------------------------------------------------------------------------
# Frame loading
# --------------------------------------------------------------------------
def load_video_frames(video_path, max_frames, target_h, target_w):
    """Load video frames, resize to target_h x target_w, return (C,F,H,W) tensor.

    Returns None if the video has fewer than *max_frames* frames.
    Also returns the original (H, W) before resizing as the second element.
    """
    if _USE_DECORD:
        vr = decord.VideoReader(video_path)
        if len(vr) < max_frames:
            return None, None
        orig_h, orig_w = vr[0].shape[:2]
        frames = vr.get_batch(list(range(max_frames)))
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    else:
        cap = cv2.VideoCapture(video_path)
        buf = []
        orig_h = orig_w = None
        for _ in range(max_frames):
            ok, fr = cap.read()
            if not ok:
                cap.release()
                if len(buf) < max_frames:
                    return None, None
                break
            if orig_h is None:
                orig_h, orig_w = fr.shape[:2]
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
    return tensor, (orig_h, orig_w)


# --------------------------------------------------------------------------
# LMDB helpers
# --------------------------------------------------------------------------
def _read_paths_from_lmdb(lmdb_dir):
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
                    shutil.rmtree(os.path.join(output_dir, name),
                                  ignore_errors=True)
        return

    new_total = sum(c for _, c in pending)
    base_count, existing_paths = _read_paths_from_lmdb(final_dir)
    print(f"[pre-merge] found {len(pending)} leftover rank shard(s) with "
          f"{new_total} samples; merging into {final_dir} (existing: "
          f"{base_count}) before resuming.")

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
                        wtxn.put(f"paths_{gi}_data".encode(),
                                  path.encode("utf-8"))
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
            shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)
    print("[pre-merge] removed leftover rank shard dirs.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", required=True,
                   help="Directory containing <clip_id>.mp4 video files.")
    p.add_argument("--camera_dir", required=True,
                   help="Directory of ViPE <clip_id>.npz pose files "
                        "(c2w, per-frame).")
    p.add_argument("--intrinsics_dir", default=None,
                   help="Directory of ViPE <clip_id>.npz intrinsics files. "
                        "Auto-derived from --camera_dir by replacing 'pose' "
                        "with 'intrinsics' if omitted.")
    p.add_argument("--caption_csv", required=True, nargs="+",
                   help="One or more caption files. CSV files must have "
                        "'videoFile' and 'caption' columns. JSON files can "
                        "be a list of {video_path, caption} objects.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=704)
    p.add_argument("--target_w", type=int, default=1280)
    p.add_argument("--max_frames", type=int, default=957,
                   help="raw frame count (must be 4*k+1 for Wan2.2 VAE).")
    p.add_argument("--keep_shards", action="store_true")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--use_parent_as_clip_id", action="store_true",
                   help="Use parent directory name as clip_id (for minWM-data "
                        "where videos are <clip_id>/gen.mp4).")
    return p.parse_args()


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
    n_latent = (args.max_frames - 1) // 4 + 1
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

    # ---- Load ViPE cameras + captions ----
    if global_rank == 0:
        print(f"Loading ViPE cameras from: {args.camera_dir}")
        if args.intrinsics_dir:
            print(f"Intrinsics dir: {args.intrinsics_dir}")
        else:
            print("Intrinsics dir: (auto-derived)")
    cam_table = load_vipe_cameras(args.camera_dir, args.intrinsics_dir)
    if global_rank == 0:
        print(f"Loaded {len(cam_table)} ViPE camera NPZ files.")

    cap_table = load_caption_csvs(
        args.caption_csv, use_parent_as_clip_id=args.use_parent_as_clip_id)
    if global_rank == 0:
        print(f"Loaded {len(cap_table)} captions from "
              f"{len(args.caption_csv)} file(s).")

    # ---- Build the valid clip list (video ∩ camera ∩ caption) ----
    valid = []
    missing_video = missing_camera = missing_caption = 0
    if args.use_parent_as_clip_id:
        video_files = sorted(glob.glob(
            os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    else:
        video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    for vp in video_files:
        if args.use_parent_as_clip_id:
            clip_id = os.path.basename(os.path.dirname(vp))
        else:
            clip_id = os.path.splitext(os.path.basename(vp))[0]
        if clip_id not in cam_table:
            missing_camera += 1
            continue
        if clip_id not in cap_table:
            missing_caption += 1
            continue
        cam = cam_table[clip_id]
        # Need at least max_frames camera poses (interval=1)
        if cam["c2w"].shape[0] < args.max_frames:
            missing_camera += 1
            continue
        valid.append({
            "clip_id":    clip_id,
            "video_path": vp,
            "caption":    cap_table[clip_id],
            "c2w":        cam["c2w"],            # (T, 4, 4) c2w
            "intr":       cam["intrinsics"],      # (T, 4) or None
        })

    if global_rank == 0:
        print(f"Valid clips (video ∩ camera ∩ caption): {len(valid)}")
        print(f"  missing camera (or too short): {missing_camera}")
        print(f"  missing caption: {missing_caption}")
        print(f"F_lat={n_latent}  H_lat={h_lat}  W_lat={w_lat}")

    # Init Wan2.2 VAE
    from utils.wan_5b_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "data")
    rank_dir = os.path.join(args.output_dir, f".rank_{global_rank}")

    # ---- Clean slate or pre-merge ----
    if args.no_resume:
        if global_rank == 0 and os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        if os.path.exists(rank_dir):
            shutil.rmtree(rank_dir)
        if world_size > 1:
            torch.distributed.barrier()
    else:
        if global_rank == 0:
            _flush_pending_shards_into_final(
                args.output_dir, world_size,
                lat_shape_det, intr_shape_det, poses_shape_det,
                per_sample_bytes,
            )
        if world_size > 1:
            torch.distributed.barrier()

    # ---- Resume ----
    done_paths = set()
    if not args.no_resume:
        _, merged_paths = _read_paths_from_lmdb(final_dir)
        done_paths |= merged_paths

    remaining = [it for it in valid if it["video_path"] not in done_paths]
    if global_rank == 0:
        print(f"[resume] {len(done_paths)} clips already in merged data/; "
              f"{len(remaining)} clips still to process "
              f"(re-partitioning across {world_size} GPU(s)).")

    if world_size > 1:
        per_gpu = (len(remaining) + world_size - 1) // world_size
        shard = remaining[global_rank * per_gpu:(global_rank + 1) * per_gpu]
    else:
        shard = remaining

    os.makedirs(rank_dir, exist_ok=True)
    rank_map = int(max(len(shard), 1) * per_sample_bytes * 1.3) + 100_000_000
    rank_env = lmdb.open(rank_dir, map_size=rank_map, subdir=True)

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
        if item["video_path"] in done_paths:
            pbar.set_postfix(ok=count, err=errors, skip=len(done_paths))
            continue
        try:
            frames, orig_hw = load_video_frames(
                item["video_path"], args.max_frames,
                args.target_h, args.target_w)
            if frames is None or orig_hw is None:
                errors += 1; continue
            orig_h, orig_w = orig_hw
            pixel = frames.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode_to_latent(pixel)
            latent_np = latent[0].float().cpu().numpy().astype(np.float16)

            # Get intrinsics row [fx, fy, cx, cy] in pixel units.
            if item["intr"] is not None:
                intr_row = item["intr"][0]  # first frame
            else:
                # Fallback: assume principal point at center, fov ~60°
                intr_row = np.array(
                    [orig_w * 0.75, orig_h * 0.75,
                     orig_w / 2.0, orig_h / 2.0], dtype=np.float32)

            # Convert ViPE c2w → w2c poses (4× temporal subsampling).
            intrinsics, poses = poses_from_vipe_c2w(
                item["c2w"], n_latent, intr_row,
                orig_w=orig_w, orig_h=orig_h)
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

    # ---- Phase 2: rank-0 merges ----
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
                            wtxn.put(f"paths_{gi}_data".encode(),
                                      path.encode("utf-8"))
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
