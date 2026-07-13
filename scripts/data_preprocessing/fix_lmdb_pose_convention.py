#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix camera pose convention in legacy VGGT-Omega LMDBs.

Legacy VGGT-Omega LMDBs (built before the extrinsics convention fix in
``build_camera_lmdb_5b_sekai.py``) stored **c2w** (camera-to-world) poses
mislabeled as **w2c** (world-to-camera), because VGGT-Omega's ``extrinsics``
output is w2c but was incorrectly treated as c2w and inverted.

This script inverts each stored pose **in-place** so the LMDB ends up with
the correct w2c convention, matching what ``build_viewmats_and_Ks`` and
the training pipeline expect.

Only ``poses_{idx}_data`` is modified; ``latents``, ``prompts``,
``intrinsics``, and ``paths`` are untouched.

Usage::

    # Preview (dry run, no writes):
    python scripts/data_preprocessing/fix_lmdb_pose_convention.py \
        --lmdb_path /path/to/lmdb/data --dry_run

    # Fix in-place (creates data.mdb.bak backup by default):
    python scripts/data_preprocessing/fix_lmdb_pose_convention.py \
        --lmdb_path /path/to/lmdb/data

    # Skip backup:
    python scripts/data_preprocessing/fix_lmdb_pose_convention.py \
        --lmdb_path /path/to/lmdb/data --no_backup

Works with both a single LMDB directory (has ``data.mdb`` directly) and a
sharded layout (parent dir of sub-directories each containing ``data.mdb``).
"""

import argparse
import os
import shutil
import sys

import lmdb
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Pose inversion (c2w → w2c)
# ---------------------------------------------------------------------------
def invert_poses_c2w_to_w2c(poses_c2w: np.ndarray) -> np.ndarray:
    """Invert (T, 7) poses from c2w [tx,ty,tz, qx,qy,qz,qw] to w2c.

    Each row is a c2w pose: translation = camera position in world,
    rotation = camera-to-world rotation.  The inverse gives the w2c pose
    that ``build_viewmats_and_Ks`` expects.
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
# LMDB helpers
# ---------------------------------------------------------------------------
def _get_lmdb_count(env) -> int:
    """Return the number of samples stored in *env*."""
    with env.begin() as txn:
        cnt_raw = txn.get(b"__count__")
        if cnt_raw is not None:
            return int(cnt_raw.decode())
        ls = txn.get(b"latents_shape")
        if ls is not None:
            return int(ls.decode().split()[0])
    return 0


def _get_poses_shape(env):
    """Return the per-sample poses shape, e.g. (F_lat, 7)."""
    with env.begin() as txn:
        raw = txn.get(b"poses_shape")
    if raw is None:
        return None
    parts = raw.decode().split()
    # First element is the count; remaining are the per-sample shape.
    return tuple(int(x) for x in parts[1:])


def _discover_lmdb_dirs(data_path: str) -> list:
    """Return a list of directories that each contain a ``data.mdb``.

    Handles both:
      * Single LMDB:  ``data_path/data.mdb``  →  [data_path]
      * Sharded:      ``data_path/<shard>/data.mdb``  →  [data_path/<shard>, ...]
    """
    if os.path.isfile(os.path.join(data_path, "data.mdb")):
        return [data_path]
    dirs = []
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub = os.path.join(data_path, name)
            if os.path.isfile(os.path.join(sub, "data.mdb")):
                dirs.append(sub)
    return dirs


# ---------------------------------------------------------------------------
# Core fix logic
# ---------------------------------------------------------------------------
def fix_one_lmdb(lmdb_dir: str, dry_run: bool, no_backup: bool) -> dict:
    """Fix poses in a single LMDB directory. Returns a stats dict."""
    stats = {"dir": lmdb_dir, "count": 0, "fixed": 0, "skipped": 0, "error": None}

    # ---- Read-only pass: get count + poses shape ----
    env_ro = lmdb.open(lmdb_dir, readonly=True, lock=False,
                       readahead=False, meminit=False)
    count = _get_lmdb_count(env_ro)
    poses_shape = _get_poses_shape(env_ro)
    env_ro.close()

    if count == 0:
        stats["error"] = "empty LMDB (count=0)"
        return stats
    if poses_shape is None or len(poses_shape) < 2:
        stats["error"] = f"cannot determine poses shape: {poses_shape}"
        return stats

    stats["count"] = count
    per_sample_shape = poses_shape  # (F_lat, 7)
    print(f"  [{lmdb_dir}] {count} samples, poses shape={per_sample_shape}")

    # ---- Dry run: just preview the first sample ----
    if dry_run:
        env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            raw = txn.get(b"poses_0_data")
        env.close()
        if raw is None:
            stats["error"] = "poses_0_data not found"
            return stats
        poses0 = np.frombuffer(raw, dtype=np.float32).reshape(per_sample_shape)
        fixed0 = invert_poses_c2w_to_w2c(poses0)
        print(f"    [DRY RUN] Sample 0 first pose (before): {poses0[0]}")
        print(f"    [DRY RUN] Sample 0 first pose (after) : {fixed0[0]}")
        print(f"    [DRY RUN] No writes performed.")
        stats["skipped"] = count
        return stats

    # ---- Backup ----
    if not no_backup:
        src = os.path.join(lmdb_dir, "data.mdb")
        bak = os.path.join(lmdb_dir, "data.mdb.bak")
        if os.path.exists(bak):
            print(f"    Backup already exists, skipping backup: {bak}")
        else:
            print(f"    Creating backup: {bak}")
            shutil.copy2(src, bak)

    # ---- In-place fix ----
    # Open with a map_size large enough for the existing data.
    mdb_size = os.path.getsize(os.path.join(lmdb_dir, "data.mdb"))
    map_size = max(mdb_size * 2, 1 << 30)  # at least 1 GB
    env = lmdb.open(lmdb_dir, map_size=map_size, lock=False,
                    readahead=False, meminit=False)

    for idx in range(count):
        key = f"poses_{idx}_data".encode()
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            print(f"    [SKIP] {key} not found")
            stats["skipped"] += 1
            continue

        poses = np.frombuffer(raw, dtype=np.float32).reshape(per_sample_shape)
        fixed = invert_poses_c2w_to_w2c(poses)

        with env.begin(write=True) as txn:
            txn.put(key, fixed.tobytes())

        stats["fixed"] += 1
        if (idx + 1) % 500 == 0 or idx + 1 == count:
            print(f"    [{idx+1}/{count}] fixed")

    env.sync()
    env.close()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fix camera pose convention (c2w → w2c) in legacy "
                    "VGGT-Omega LMDBs.")
    p.add_argument("--lmdb_path", required=True,
                   help="Path to a single LMDB directory (contains data.mdb) "
                        "or a parent directory of sharded LMDBs.")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview the fix without writing anything.")
    p.add_argument("--no_backup", action="store_true",
                   help="Skip creating data.mdb.bak before modifying.")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = os.path.abspath(args.lmdb_path)

    if not os.path.isdir(data_path):
        print(f"ERROR: not a directory: {data_path}", file=sys.stderr)
        sys.exit(1)

    lmdb_dirs = _discover_lmdb_dirs(data_path)
    if not lmdb_dirs:
        print(f"ERROR: no LMDB (data.mdb) found under: {data_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(lmdb_dirs)} LMDB directory(ies) to process.")
    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")
    elif not args.no_backup:
        print("Backups will be created (data.mdb.bak).\n")
    else:
        print("WARNING: --no_backup set, no backups will be created.\n")

    all_stats = []
    for lmdb_dir in lmdb_dirs:
        print(f"Processing: {lmdb_dir}")
        stats = fix_one_lmdb(
            lmdb_dir,
            dry_run=args.dry_run,
            no_backup=args.no_backup,
        )
        all_stats.append(stats)
        if stats["error"]:
            print(f"  ERROR: {stats['error']}")
        else:
            print(f"  Done: {stats['fixed']} fixed, {stats['skipped']} skipped "
                  f"out of {stats['count']}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Summary:")
    total_count = sum(s["count"] for s in all_stats)
    total_fixed = sum(s["fixed"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)
    print(f"  Total samples : {total_count}")
    print(f"  Fixed         : {total_fixed}")
    print(f"  Skipped       : {total_skipped}")
    if all_stats:
        for s in all_stats:
            if s["error"]:
                print(f"  ERROR in {s['dir']}: {s['error']}")

    if args.dry_run:
        print("\n[DRY RUN] Re-run without --dry_run to apply the fix.")


if __name__ == "__main__":
    main()
