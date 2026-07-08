#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B PRoPE Bidirectional SFT
from the *Sekai* (Leegen/Sekai) dataset using VGGT-Omega camera estimates.

Differences from ``build_camera_lmdb_5b_sekai_game.py``:
  * Camera parameters come from VGGT-Omega (``batch_vggt_omega.py``) instead of
    the Sekai-Game sharded NPZ.  Each video has its own ``<clip_id>.npz`` in
    ``--camera_dir`` with keys:
      - ``extrinsics``  : (1, T, 3, 4)  float32, c2w (camera-to-world)
      - ``intrinsics``  : (1, T, 3, 3)  float32, pinhole [fx 0 cx; 0 fy cy; 0 0 1]
      - ``num_frames``  : int64 = T
      - ``height``      : int64  (VGGT processing resolution)
      - ``width``       : int64
  * **Cameras are already 4×-strided**: VGGT-Omega was run with
    ``--frame_stride 4``, so T poses correspond to original video frames
    ``[0, 4, 8, ..., 4*(T-1)]``.  This means each camera pose already aligns
    1-to-1 with a Wan2.2 latent frame (4× temporal VAE).  We therefore use
    ``vae_time_stride=1`` when subsampling the trajectory — there is no
    additional 4× subsampling to do.
  * Captions are read from CSV files (``--caption_csv``) with a ``videoFile``
    column (e.g. ``clip_id.mp4``) and a ``caption`` column, instead of JSON.
  * Intrinsics are normalized by the VGGT processing resolution
    (``height`` / ``width`` from the NPZ), not the original capture resolution.

For each clip, this script:
  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w>.
  2. Encodes them with the Wan2.2 VAE (4× temporal, 16× spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Selects F_lat camera poses from the VGGT trajectory using
     ``vae_time_stride=1`` (indices ``[0, 1, ..., F_lat-1]``).  Each VGGT pose
     i corresponds to original frame 4*i, which is the *last* frame of the
     i-th 4-frame VAE chunk — matching the ``cam_sample_strategy='last'``
     convention from SANA / Sekai-Game.
  4. Stores normalized intrinsics [fx/W_vggt, fy/H_vggt, cx/W_vggt, cy/H_vggt]
     (4,) float32.
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/``.

Wan2.2 VAE has ratio (4× temporal, 16× spatial), so for ``F_lat = 40``:
    raw_frames  = (F_lat - 1) * 4 + 1 = 157
    latent_h    = target_h / 16
    latent_w    = target_w / 16

Usage:
    torchrun --nproc_per_node=2 \
        scripts/data_preprocessing/build_camera_lmdb_5b_sekai.py \
        --video_dir    /path/to/Sekai/video \
        --camera_dir   /path/to/Sekai/vggt_omega_results \
        --caption_csv  /path/to/Sekai-Game.csv /path/to/Sekai-Real-HQ.csv \
        --output_dir   ./data/train/sekai/ \
        --target_h 704 --target_w 1280 --max_frames 157
"""

import argparse
import csv
import glob
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
# VGGT-Omega per-video NPZ loading
# --------------------------------------------------------------------------
def load_vggt_camera_dir(camera_dir):
    """Scan *camera_dir* for ``<clip_id>.npz`` files produced by
    ``batch_vggt_omega.py --cameras_only``.

    Each NPZ has:
        extrinsics  : (1, T, 3, 4) float32, c2w
        intrinsics  : (1, T, 3, 3) float32
        num_frames  : int64 = T
        height      : int64  (VGGT processing resolution)
        width       : int64

    Returns a dict ``clip_id -> {pose: (T, 4, 4) c2w, intrinsics: (T, 3, 3),
                                height, width}``.
    """
    out = {}
    skipped = 0
    for path in sorted(glob.glob(os.path.join(camera_dir, "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            z = np.load(path, allow_pickle=True)
            # Only read the small arrays we actually need; never touch
            # 'depth' (large, and can be corrupted in old-format NPZs).
            ext = z["extrinsics"]          # (1, T, 3, 4)
            intr = z["intrinsics"]         # (1, T, 3, 3)
        except Exception as e:
            print(f"[WARN] skipping corrupted NPZ {stem}: {e}", flush=True)
            skipped += 1
            continue
        # Squeeze batch dim
        ext = np.asarray(ext[0], dtype=np.float32)    # (T, 3, 4)
        intr = np.asarray(intr[0], dtype=np.float32)  # (T, 3, 3)
        T = ext.shape[0]
        # Pad (T, 3, 4) -> (T, 4, 4) c2w by appending [0, 0, 0, 1]
        c2w = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        c2w[:, :3, :] = ext
        # Resolution: prefer explicit height/width keys; otherwise derive
        # from the intrinsics principal point (cx = W/2, cy = H/2).
        keys = set(z.files)
        if "height" in keys and "width" in keys:
            h = int(z["height"])
            w = int(z["width"])
        else:
            cx = float(intr[0, 0, 2])
            cy = float(intr[0, 1, 2])
            w = int(round(cx * 2))
            h = int(round(cy * 2))
            if w <= 0 or h <= 0:
                w, h = 688, 384  # safe fallback
        out[stem] = {
            "pose": c2w,
            "intrinsics": intr,
            "height": h,
            "width": w,
        }
    if skipped:
        print(f"[WARN] {skipped} corrupted NPZ file(s) skipped.", flush=True)
    return out


def load_caption_csvs(csv_paths):
    """Merge one or more CSV files with ``videoFile`` and ``caption`` columns.

    Returns a dict ``clip_id -> caption`` (clip_id = videoFile without
    extension).
    """
    out = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
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
# Camera subsampling (vae_time_stride=1 because VGGT already strided by 4)
# --------------------------------------------------------------------------
def poses_from_vggt_c2w(c2w_seq, n_latent, intrinsics_row,
                         orig_w, orig_h):
    """Subsample a VGGT-Omega c2w trajectory to ``n_latent`` w2c poses.

    VGGT-Omega was run with ``--frame_stride 4``, so the trajectory already
    has one pose per 4-frame VAE chunk.  We therefore take poses
    ``[0, 1, ..., n_latent-1]`` directly (``vae_time_stride=1``).

    Args:
        c2w_seq:        (T, 4, 4) float32, per-strided-frame c2w.
        n_latent:       number of latent frames (= number of camera poses to
                        select).
        intrinsics_row: (3, 3) float32 pinhole matrix at the VGGT resolution.
        orig_w/orig_h:  VGGT processing resolution (used to normalize fx, cx /
                        fy, cy).

    Returns:
        intrinsics: (4,) float32, normalized [fx/W, fy/H, cx/W, cy/H].
        poses:     (n_latent, 7) float32, w2c [tx, ty, tz, qx, qy, qz, qw].
    """
    L = int(c2w_seq.shape[0])
    # With vae_time_stride=1, indices are simply [0, 1, ..., n_latent-1].
    idxs = [min(i, L - 1) for i in range(n_latent)]

    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i, fi in enumerate(idxs):
        c2w = np.asarray(c2w_seq[fi], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()

    fx = float(intrinsics_row[0, 0])
    fy = float(intrinsics_row[1, 1])
    cx = float(intrinsics_row[0, 2])
    cy = float(intrinsics_row[1, 2])
    intrinsics = np.array(
        [fx / orig_w, fy / orig_h, cx / orig_w, cy / orig_h],
        dtype=np.float32,
    )
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
                   help="Directory containing <clip_id>.mp4 files")
    p.add_argument("--camera_dir", required=True,
                   help="Directory of VGGT-Omega <clip_id>.npz files "
                        "(produced by batch_vggt_omega.py --cameras_only)")
    p.add_argument("--caption_csv", required=True, nargs="+",
                   help="One or more CSV files with 'videoFile' and 'caption' "
                        "columns (e.g. Sekai-Game.csv, Sekai-Real-HQ.csv)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=704)
    p.add_argument("--target_w", type=int, default=1280)
    p.add_argument("--max_frames", type=int, default=157,
                   help="raw frame count (must give integer F_lat under 4x temporal)")
    p.add_argument("--keep_shards", action="store_true",
                   help="Keep the transient per-rank shard dirs after merge "
                        "(debugging only).")
    p.add_argument("--no_resume", action="store_true",
                   help="Wipe any existing output ('data/' + shards) and "
                        "reprocess everything from scratch.")
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

    # ---- Load camera + caption metadata ----
    if global_rank == 0:
        print(f"Loading VGGT-Omega cameras from: {args.camera_dir}")
    cam_table = load_vggt_camera_dir(args.camera_dir)
    if global_rank == 0:
        print(f"Loaded {len(cam_table)} VGGT camera NPZ files.")

    cap_table = load_caption_csvs(args.caption_csv)
    if global_rank == 0:
        print(f"Loaded {len(cap_table)} captions from {len(args.caption_csv)} CSV file(s).")

    # ---- Build the valid clip list (video ∩ camera ∩ caption) ----
    valid = []
    missing_video = missing_camera = missing_caption = 0
    # Iterate over video files that exist on disk
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    for vp in video_files:
        clip_id = os.path.splitext(os.path.basename(vp))[0]
        if clip_id not in cam_table:
            missing_camera += 1
            continue
        if clip_id not in cap_table:
            missing_caption += 1
            continue
        cam = cam_table[clip_id]
        # Need at least n_latent camera poses
        if cam["pose"].shape[0] < n_latent:
            missing_camera += 1
            continue
        valid.append({
            "clip_id":    clip_id,
            "video_path": vp,
            "caption":    cap_table[clip_id],
            "c2w":        cam["pose"],
            "intr_mat":   cam["intrinsics"][0],   # (3, 3) first frame
            "vggt_h":     cam["height"],
            "vggt_w":     cam["width"],
        })

    if global_rank == 0:
        print(f"Valid clips (video ∩ camera ∩ caption): {len(valid)}")
        print(f"  missing camera (or too short): {missing_camera}")
        print(f"  missing caption: {missing_caption}")
        print(f"F_lat={n_latent}  H_lat={h_lat}  W_lat={w_lat}")

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
        if global_rank == 0:
            _flush_pending_shards_into_final(
                args.output_dir, world_size,
                lat_shape_det, intr_shape_det, poses_shape_det,
                per_sample_bytes,
            )
        if world_size > 1:
            torch.distributed.barrier()

    os.makedirs(rank_dir, exist_ok=True)
    rank_map = int(len(shard) * per_sample_bytes * 1.3) + 100_000_000
    rank_env = lmdb.open(rank_dir, map_size=rank_map, subdir=True)

    # ---- Resume ----
    done_paths = set()
    if not args.no_resume:
        _, merged_paths = _read_paths_from_lmdb(final_dir)
        done_paths |= merged_paths
    count = 0
    with rank_env.begin() as txn:
        cnt_raw = txn.get(b"__count__")
        if cnt_raw is not None:
            count = int(cnt_raw.decode())
            for j in range(count):
                pt = txn.get(f"paths_{j}_data".encode())
                if pt is not None:
                    done_paths.add(pt.decode("utf-8"))
    if global_rank == 0 and done_paths:
        print(f"[resume] {len(done_paths)} clips already processed "
              f"(merged + shards); they will be skipped.")

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
            # vae_time_stride=1: VGGT cameras are already 4×-strided, so each
            # camera pose i corresponds to latent frame i directly.
            intrinsics, poses = poses_from_vggt_c2w(
                item["c2w"], n_latent, item["intr_mat"],
                orig_w=item["vggt_w"], orig_h=item["vggt_h"])
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
