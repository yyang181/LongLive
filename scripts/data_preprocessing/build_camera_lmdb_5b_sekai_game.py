#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a camera-aware LMDB dataset for Wan2.2-TI2V-5B PRoPE Bidirectional SFT
from the *Sekai-Game* layout.

Differences from ``build_camera_lmdb_5b.py``:
  * Inputs are no longer a single ``clips.json``. Instead the script consumes:
      - ``--video_dir``     : directory of ``<clip_id>.mp4`` files.
      - ``--camera_npz``    : one (or more) ``*_camera.npz`` shard with keys
                              ``ids (N,)``, ``ranges (N, 2) = [start, length]``,
                              ``pose (T, 4, 4)`` (c2w),
                              ``intrinsics (T, 4) = [fx, fy, cx, cy]``
                              referenced to the *original capture* resolution.
      - ``--caption_json``  : one (or more) JSON file mapping
                              ``{clip_id: {"prompt": "..."}}``.
    A clip is included only if it appears in *all three* (video / npz / caption).
  * Real per-frame c2w trajectories are read from the NPZ instead of being
    synthesized from a WorldPlayGen pose-string. They are converted to
    per-latent-frame [tx, ty, tz, qx, qy, qz, qw] (w2c) exactly like the
    original script.
  * Intrinsics are read from the NPZ (first frame of each clip — they are
    constant per clip in the Sekai-Game release) and normalized by the
    *original capture* resolution, matching the convention used by
    ``utils.camera_dataset.CameraLatentLMDBDataset`` (resolution-independent).

For each clip, this script:
  1. Loads <MAX_FRAMES> RGB frames at <target_h x target_w>.
  2. Encodes them with the Wan2.2 VAE (4× temporal, 16× spatial, 48 channels)
     into a (F_lat, 48, H/16, W/16) fp16 latent.
  3. Subsamples the per-frame c2w trajectory to F_lat poses, mirroring SANA's
     ``SanaWMZipLatentDataset.cam_sample_strategy`` (configs/sana_wm/stage1/
     v0_First_chunk.yaml uses ``cam_sample_strategy: last``). Concretely:
        - ``last``  (default, matches SANA yaml):  raw indices = ``[0, 4, 8,
          ..., (F_lat-1)*4]`` — each latent token's anchor is the *last* frame
          of its 4-frame chunk (SANA does the equivalent on 8-frame chunks
          since LTX2VAE has stride 8; we use 4 for Wan2.2-TI2V-5B).
        - ``first``: raw indices = ``[0, 1, 5, 9, ..., (F_lat-2)*4 + 1]`` —
          first chunk degenerates to frame 0, then anchor = first frame of
          each subsequent chunk.
     Anchors are then inverted to w2c and stored as
     (F_lat, 7) [tx, ty, tz, qx, qy, qz, qw].
  4. Stores normalized intrinsics [fx/W_orig, fy/H_orig, cx/W_orig, cy/H_orig]
     (4,) float32. NOTE: ``ASPECT_RATIO_VIDEO_720_MS_DIV32`` from SANA is *not*
     used here — that table is only a multi-aspect-ratio bucket grid for
     SANA's data sampler; for this single-resolution (704x1280, key '0.57' in
     the bucket table) preprocessing pipeline the only thing that matters is
     that ``orig_w/orig_h`` correctly reflect the capture resolution that the
     NPZ intrinsics reference (1920x1080 for Sekai-Game).
  5. Streams each rank to its own LMDB shard, then rank-0 merges into
     ``<output_dir>/data/`` so that the final layout is what
     ``utils.camera_dataset.CameraLatentLMDBDataset`` expects.

Wan2.2 VAE has ratio (4× temporal, 16× spatial), so for ``F_lat = 20``:
    raw_frames  = (F_lat - 1) * 4 + 1 = 77
    latent_h    = target_h / 16
    latent_w    = target_w / 16

Usage (8-GPU example):
    torchrun --nproc_per_node=8 \
        scripts/data_preprocessing/build_camera_lmdb_5b_sekai_game.py \
        --video_dir    /path/to/sekai_game_.../video \
        --camera_npz   /path/to/sekai_game_train_00000000_camera.npz \
        --caption_json /path/to/sekai_game_train_00000000_LongSceneStaticCaption-Qwen3-VL-30B-A3B-Instruct.json \
        --output_dir   ./data/train/sekai_game/ \
        --target_h 704 --target_w 1280 --max_frames 77
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
# Sekai-Game NPZ + caption-json loading
# --------------------------------------------------------------------------
def load_sekai_camera_shards(npz_paths):
    """Concatenate one or more Sekai-Game ``*_camera.npz`` shards.

    Each NPZ has:
        ids        : (N,) string clip ids
        ranges     : (N, 2) int = [start, length] into the per-frame arrays
        pose       : (T, 4, 4) float32, c2w SE(3)
        intrinsics : (T, 4) float32 = [fx, fy, cx, cy] in original-pixel units

    Returns a dict ``clip_id -> {pose: (L, 4, 4), intrinsics: (L, 4)}``.
    """
    out = {}
    for path in npz_paths:
        z = np.load(path, allow_pickle=True)
        ids = z["ids"]
        ranges = z["ranges"]
        pose = z["pose"]
        intr = z["intrinsics"]
        for cid, (start, length) in zip(ids.tolist(), ranges.tolist()):
            if length <= 0:
                continue
            end = start + length
            if end > pose.shape[0] or end > intr.shape[0]:
                # Defensive: drop malformed entries.
                continue
            out[str(cid)] = {
                "pose": np.asarray(pose[start:end], dtype=np.float32),
                "intrinsics": np.asarray(intr[start:end], dtype=np.float32),
            }
    return out


def load_caption_jsons(json_paths):
    """Merge one or more ``{clip_id: {prompt: ...}}`` mappings."""
    out = {}
    for path in json_paths:
        with open(path) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError(f"Caption JSON {path} is not a dict")
        for cid, val in d.items():
            if isinstance(val, dict) and "prompt" in val:
                cap = val["prompt"]
            elif isinstance(val, str):
                cap = val
            else:
                continue
            if not isinstance(cap, str) or not cap.strip():
                continue
            out[str(cid)] = cap
    return out


def _build_time_indices(n_latent, vae_time_stride, raw_frames, strategy):
    """Build the list of *raw-frame indices* used as the per-latent-frame
    camera anchor, mirroring SANA's ``cam_sample_strategy`` semantics.

    SANA's ``SanaWMZipLatentDataset`` (LTX2 VAE, 8× temporal) samples:
        - "last":  ``arange(0, T_pose, 8)`` -> ``[0, 8, 16, ...]`` —
                   each latent token's chunk window is ``[idx-7, idx]`` (the
                   anchor is the *last* frame of the 8-frame chunk; the first
                   chunk is clamped+padded).
        - "first": ``arange(0, T_pose, 8) - 7`` then index 0 forced to 0 ->
                   ``[0, 1, 9, 17, ...]`` (first chunk is single-frame, then
                   each anchor is the *first* frame of subsequent chunks).

    Wan2.2-TI2V-5B VAE has 4× temporal compression, so we substitute
    ``vae_time_stride = 4`` and produce the equivalent indices.
    """
    if strategy == "last":
        idxs = [i * vae_time_stride for i in range(n_latent)]
    elif strategy == "first":
        # First chunk: single frame at 0; subsequent chunks: the chunk's
        # first frame, i.e. ``i*stride - stride + 1`` for i >= 1.
        idxs = [0] + [i * vae_time_stride - vae_time_stride + 1
                      for i in range(1, n_latent)]
    else:
        raise ValueError(f"Invalid cam_sample_strategy: {strategy!r} "
                         f"(expected 'first' or 'last').")
    # Note: indices are *not* clamped here; the caller is responsible for
    # clamping into the actual trajectory length (which may be shorter than
    # ``raw_frames`` for malformed clips).
    return idxs


def poses_from_c2w_array(c2w_seq, n_latent, intrinsics_row,
                         orig_w=1920.0, orig_h=1080.0,
                         vae_time_stride=4, cam_sample_strategy="last"):
    """Subsample a (L, 4, 4) c2w trajectory to ``n_latent`` w2c poses and
    return the normalized intrinsics for the clip.

    Args:
        c2w_seq:        (L, 4, 4) float32, per-raw-frame c2w (camera-to-world).
        n_latent:       number of latent frames after the Wan2.2 4× temporal VAE.
        intrinsics_row: (4,) float32 = [fx, fy, cx, cy] in original-pixel
                        units (we use the first frame — Sekai-Game's intrinsics
                        are constant per clip).
        orig_w/orig_h:  capture resolution that ``intrinsics_row`` references
                        (Sekai-Game default = 1920x1080).
        vae_time_stride: temporal compression factor of the VAE (4 for
                        Wan2.2-TI2V-5B; SANA uses 8 for LTX2VAE).
        cam_sample_strategy: 'last' (default) or 'first' — see
                        ``_build_time_indices`` for the exact semantics; this
                        mirrors SANA's ``SanaWMZipLatentDataset`` knob of the
                        same name. The default ('last') matches SANA's
                        ``v0_First_chunk.yaml`` config.

    Returns:
        intrinsics: (4,) float32, normalized to [0, 1] by capture resolution.
        poses:     (n_latent, 7) float32, w2c [tx, ty, tz, qx, qy, qz, qw].
    """
    L = int(c2w_seq.shape[0])
    raw_frames = (n_latent - 1) * int(vae_time_stride) + 1
    idxs = _build_time_indices(n_latent, int(vae_time_stride),
                               raw_frames, cam_sample_strategy)
    # Clamp every index into the available trajectory length.
    idxs = [max(0, min(int(fi), L - 1)) for fi in idxs]

    poses = np.zeros((n_latent, 7), dtype=np.float32)
    for i, fi in enumerate(idxs):
        c2w = np.asarray(c2w_seq[fi], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        poses[i, :3] = w2c[:3, 3]
        poses[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()

    fx, fy, cx, cy = (float(v) for v in intrinsics_row[:4])
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
    p.add_argument("--camera_npz", required=True, nargs="+",
                   help="One or more Sekai-Game *_camera.npz shards "
                        "(supports glob, e.g. '/path/*_camera.npz')")
    p.add_argument("--caption_json", required=True, nargs="+",
                   help="One or more {clip_id: {prompt:...}} JSON files "
                        "(supports glob)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_h", type=int, default=704)
    p.add_argument("--target_w", type=int, default=1280)
    p.add_argument("--orig_w", type=float, default=1920.0,
                   help="Capture width that the NPZ intrinsics reference "
                        "(used to normalize fx, cx).")
    p.add_argument("--orig_h", type=float, default=1080.0,
                   help="Capture height that the NPZ intrinsics reference "
                        "(used to normalize fy, cy).")
    p.add_argument("--max_frames", type=int, default=77,
                   help="raw frame count (must give integer F_lat under 4x temporal)")
    p.add_argument("--cam_sample_strategy", choices=("first", "last"),
                   default="last",
                   help="Per-latent-frame anchor selection strategy, mirroring "
                        "SANA's SanaWMZipLatentDataset.cam_sample_strategy. "
                        "'last' (default) anchors each latent token to the "
                        "*last* raw frame of its temporal chunk and matches "
                        "configs/sana_wm/stage1/v0_First_chunk.yaml.")
    p.add_argument("--keep_shards", action="store_true",
                   help="Keep the transient per-rank shard dirs after merge "
                        "(debugging only). By default they are deleted; resume "
                        "relies on the merged 'data/' LMDB, not on the shards.")
    p.add_argument("--no_resume", action="store_true",
                   help="Wipe any existing output ('data/' + shards) and "
                        "reprocess everything from scratch.")
    return p.parse_args()


def _expand_globs(patterns):
    out = []
    seen = set()
    for pat in patterns:
        matched = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not matched and not any(c in pat for c in "*?["):
            matched = [pat]
        for m in matched:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
    return out


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
    # Soft compatibility check with SANA's ASPECT_RATIO_VIDEO_720_MS_DIV32
    # buckets (all entries are multiples of 32). This is informational —
    # Wan2.2-TI2V-5B itself only requires multiples of 16.
    if args.target_h % 32 != 0 or args.target_w % 32 != 0:
        print(
            f"[warn] target_h/target_w = {args.target_h}x{args.target_w} are not "
            "multiples of 32; this dataset is still valid for Wan2.2-TI2V-5B "
            "but will not align with SANA's ASPECT_RATIO_VIDEO_720_MS_DIV32 "
            "bucket grid (all entries there are 32-divisible).",
            flush=True,
        )
    n_latent = (args.max_frames - 1) // 4 + 1   # Wan2.2 VAE: 4x temporal
    h_lat = args.target_h // 16
    w_lat = args.target_w // 16
    # Output tensor shapes are fully determined by the args, so we can rely on
    # them for merge metadata even when a resumed run adds zero new samples.
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

    # ---- Resolve input shards (NPZ + caption JSON, supports globs) ----
    npz_paths = _expand_globs(args.camera_npz)
    cap_paths = _expand_globs(args.caption_json)
    if global_rank == 0:
        print(f"Camera NPZ shards ({len(npz_paths)}): {npz_paths}")
        print(f"Caption JSON shards ({len(cap_paths)}): {cap_paths}")

    # ---- Load camera + caption metadata (small enough to keep on every rank) ----
    cam_table = load_sekai_camera_shards(npz_paths)
    cap_table = load_caption_jsons(cap_paths)
    if global_rank == 0:
        print(f"Loaded {len(cam_table)} clip cameras and "
              f"{len(cap_table)} captions.")

    # ---- Build the (clip_id, video_path, caption, c2w_seq, intrinsics_row) list ----
    valid = []
    missing_video = missing_camera = missing_caption = 0
    for cid in sorted(cam_table.keys()):
        if cid not in cap_table:
            missing_caption += 1
            continue
        vp = os.path.join(args.video_dir, f"{cid}.mp4")
        if not os.path.exists(vp):
            missing_video += 1
            continue
        valid.append({
            "clip_id":    cid,
            "video_path": vp,
            "caption":    cap_table[cid],
            "c2w":        cam_table[cid]["pose"],
            "intr_row":   cam_table[cid]["intrinsics"][0],
        })
    # Also account for caption ids that have no NPZ entry.
    for cid in cap_table:
        if cid not in cam_table:
            missing_camera += 1

    if global_rank == 0:
        print(f"Valid clips (video ∩ camera ∩ caption): {len(valid)}")
        print(f"  missing video:   {missing_video}")
        print(f"  missing camera:  {missing_camera}")
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
        # ---- Pre-merge any leftover .rank_* shards from a previous run ----
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
    # Defensive: if for any reason this rank's shard already has entries,
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
            intrinsics, poses = poses_from_c2w_array(
                item["c2w"], n_latent, item["intr_row"],
                orig_w=args.orig_w, orig_h=args.orig_h,
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
            # Stored so a future run knows this clip is already processed.
            txn.put(f"paths_{count}_data".encode(),
                    item["video_path"].encode("utf-8"))
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
