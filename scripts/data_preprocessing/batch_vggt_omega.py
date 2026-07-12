#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch-run VGGT-Omega inference on all videos under a directory tree.

Recursively finds every video file under ``--input_dir``, decodes its frames,
runs VGGT-Omega inference, and saves the predictions (extrinsics, intrinsics,
depth, depth_conf, camera/register tokens, pose_enc) into ``--output_dir``,
preserving the relative folder structure.

Features
--------
* **Resume / skip-existing** (default on): videos whose ``.npz`` output already
  exists are skipped automatically. Disable with ``--no_resume``.
* **Multi-GPU parallelism**: pass ``--num_gpus N`` (N > 1) to split videos
  evenly across N GPUs. Each GPU runs in its own worker process with
  ``CUDA_VISIBLE_DEVICES`` set to a single physical device.
* **Frame sampling**: control via ``--max_frames`` and ``--frame_stride``.

Example
-------
    CUDA_VISIBLE_DEVICES=0,3 python batch_vggt_omega.py \
        --input_dir   /nfs/yixinyang/code/LongLive/data/Sekai/video \
        --output_dir  /nfs/yixinyang/code/LongLive/data/Sekai/vggt_omega_results \
        --checkpoint  /path/to/vggt_omega_1b_512.pt \
        --image_resolution 512 \
        --num_gpus    2
"""

import argparse
import multiprocessing as mp
import os
import queue as _queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}
PARQUET_EXTS = {".parquet"}


# ---------------------------------------------------------------------------
# Source abstraction: either a real video file on disk, or a row in a
# parquet archive produced by parquet_util.py.
# ---------------------------------------------------------------------------
@dataclass
class VideoSource:
    """A logical handle for one video to be processed.

    Exactly one of (file_path, parquet_path) is set:
      - file_path     : real video on disk.
      - parquet_path  : parquet archive; row_index identifies the row, and
                        relpath is the path-inside-parquet (used for
                        mirroring the output directory layout).
    """
    rel_parent: Path     # relative dir under --output_dir to write into
    stem: str            # basename without extension (used as output name)
    suffix: str          # original extension, e.g. ".mp4"
    file_path: Optional[Path] = None
    parquet_path: Optional[Path] = None
    row_index: Optional[int] = None
    relpath: Optional[str] = None  # relative path inside the parquet

    @property
    def is_parquet(self) -> bool:
        return self.parquet_path is not None

    @property
    def display(self) -> str:
        if self.is_parquet:
            return f"{self.parquet_path.name}::{self.relpath}"
        return str(self.file_path)


def is_parquet_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in PARQUET_EXTS


def find_videos(input_dir: Path):
    """Yield (video_path, relative_parent) for every video under *input_dir*."""
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                video_path = Path(root) / f
                rel_parent = video_path.parent.relative_to(input_dir)
                yield video_path, rel_parent


def discover_sources(input_path: Path,
                     use_parent_as_stem: bool = False) -> List[VideoSource]:
    """Build the list of VideoSource objects from either a directory or a
    parquet file produced by parquet_util.py.

    For a directory: mirrors the directory layout exactly as before.
    For a parquet file: each row becomes one VideoSource, with
    ``rel_parent`` taken from the ``relpath`` column inside the parquet.

    When *use_parent_as_stem* is True, the output stem is the immediate
    parent directory name instead of the video filename (useful for
    minWM-data where every video is ``gen.mp4`` inside a uniquely-named
    subdirectory).
    """
    sources: List[VideoSource] = []

    if is_parquet_path(input_path):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(input_path))
        # Read only metadata columns to avoid loading video bytes here.
        tbl = pf.read(columns=["relpath"])
        relpaths = tbl.column("relpath").to_pylist()
        for row_idx, rel in enumerate(relpaths):
            rel_posix = PurePosixPath(rel)
            stem = rel_posix.stem
            suffix = rel_posix.suffix
            if suffix.lower() not in VIDEO_EXTS:
                # Skip non-video rows (if any) defensively.
                continue
            rel_parent = Path(*rel_posix.parts[:-1]) if len(rel_posix.parts) > 1 \
                else Path("")
            sources.append(VideoSource(
                rel_parent=rel_parent,
                stem=stem,
                suffix=suffix,
                parquet_path=input_path,
                row_index=row_idx,
                relpath=rel,
            ))
        return sources

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"input_dir is neither a directory nor a .parquet file: "
            f"{input_path}"
        )

    for video_path, rel_parent in find_videos(input_path):
        if use_parent_as_stem:
            # Use the immediate parent directory name as the output stem,
            # and set rel_parent to the grandparent relative to input_path.
            # e.g. input_dir/000000_right8a11/gen.mp4 -> stem="000000_right8a11"
            parent_dir = video_path.parent
            stem = parent_dir.name
            try:
                rel_parent = parent_dir.parent.relative_to(input_path)
            except ValueError:
                rel_parent = Path("")
        else:
            stem = video_path.stem
        sources.append(VideoSource(
            rel_parent=rel_parent,
            stem=stem,
            suffix=video_path.suffix,
            file_path=video_path,
        ))
    return sources


def output_path_for(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}.npz"


def is_already_done(out_dir: Path, stem: str) -> bool:
    return output_path_for(out_dir, stem).exists()


# ---------------------------------------------------------------------------
# Parquet row -> bytes
# ---------------------------------------------------------------------------
# Per-process cache of opened ParquetFile handles, so a worker doesn't
# re-open the file for every video.
_PARQUET_CACHE: dict = {}


def _get_parquet_file(path: Path):
    import pyarrow.parquet as pq
    key = str(path)
    pf = _PARQUET_CACHE.get(key)
    if pf is None:
        pf = pq.ParquetFile(key)
        _PARQUET_CACHE[key] = pf
    return pf


def read_video_bytes_from_parquet(parquet_path: Path, row_index: int) -> bytes:
    """Read the raw bytes of a single row's ``data`` column from parquet.

    Reads only the row group that contains *row_index* — not the whole file.
    """
    pf = _get_parquet_file(parquet_path)
    md = pf.metadata
    # Locate the row group that contains *row_index*.
    cum = 0
    target_rg = 0
    local_offset = row_index
    for rg in range(md.num_row_groups):
        n = md.row_group(rg).num_rows
        if row_index < cum + n:
            target_rg = rg
            local_offset = row_index - cum
            break
        cum += n
    else:
        raise IndexError(
            f"row_index {row_index} out of range for {parquet_path} "
            f"(num_rows={md.num_rows})"
        )

    tbl = pf.read_row_group(target_rg, columns=["data"])
    data = tbl.column("data")[local_offset].as_py()
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    return data


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
def extract_frames_to_dir(video_path: Path, tmp_dir: Path,
                          max_frames: int, frame_stride: int):
    """Decode *video_path* to PNG frames in *tmp_dir*. Returns sorted list of
    frame file paths actually used (after stride / max_frames sampling)."""
    import cv2  # local import to keep top-level import light

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_stride > 1 and (idx % frame_stride) != 0:
            idx += 1
            continue
        out_path = tmp_dir / f"frame_{kept:06d}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
        kept += 1
        idx += 1
        if max_frames > 0 and kept >= max_frames:
            break
    cap.release()
    return saved


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Per-video inference
# ---------------------------------------------------------------------------
def run_one(model, source: VideoSource, out_dir: Path, image_resolution: int,
            max_frames: int, frame_stride: int, dry_run: bool,
            cameras_only: bool = False, tag: str = ""):
    """Run VGGT-Omega inference for a single video.

    *source* may be a real file on disk or a row inside a parquet file. In the
    parquet case, the bytes are first materialized to a temp file so that
    OpenCV's VideoCapture can decode it.

    Returns (display_name, rc, elapsed_seconds).
    """
    import shutil
    import tempfile
    import torch

    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    prefix = f"[{tag}] " if tag else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path_for(out_dir, source.stem)

    if dry_run:
        print(f"{prefix}[DRY-RUN] {source.display} -> {out_file}")
        return source.display, 0, 0.0

    print(f"{prefix}[RUN] {source.display}", flush=True)
    t_start = time.time()

    tmp_root = Path(tempfile.mkdtemp(prefix="vggt_omega_frames_"))
    parquet_tmp_video: Optional[Path] = None
    try:
        # Resolve the actual on-disk video path. For parquet rows we dump
        # the raw bytes to a temporary file (since OpenCV needs a path).
        if source.is_parquet:
            data = read_video_bytes_from_parquet(
                source.parquet_path, source.row_index)
            parquet_tmp_video = tmp_root / f"{source.stem}{source.suffix}"
            with open(parquet_tmp_video, "wb") as fp:
                fp.write(data)
            video_path = parquet_tmp_video
        else:
            video_path = source.file_path

        frames = extract_frames_to_dir(
            video_path, tmp_root, max_frames=max_frames,
            frame_stride=frame_stride,
        )
        if len(frames) == 0:
            print(f"{prefix}[SKIP] {source.display} (no frames decoded)",
                  flush=True)
            return source.display, -1, time.time() - t_start

        # images = load_and_preprocess_images(
        #     [str(p) for p in frames], image_resolution=image_resolution, mode="max_size"
        # ).to("cuda")
        images = load_and_preprocess_images(
            [str(p) for p in frames], image_resolution=image_resolution
        ).to("cuda")

        with torch.inference_mode():
            predictions = model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        camera_and_register_tokens = predictions["camera_and_register_tokens"]
        camera_tokens = camera_and_register_tokens[:, :, :1]
        registers = camera_and_register_tokens[:, :, 1:]

        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        np.savez_compressed(
            out_file,
            extrinsics=_to_numpy(extrinsics),
            intrinsics=_to_numpy(intrinsics),
            depth=_to_numpy(predictions["depth"]),
            depth_conf=_to_numpy(predictions["depth_conf"]),
            pose_enc=_to_numpy(predictions["pose_enc"]),
            camera_tokens=_to_numpy(camera_tokens),
            registers=_to_numpy(registers),
            num_frames=np.int64(len(frames)),
        )

        if cameras_only:
            # Re-save the npz keeping only camera params + resolution derived
            # from depth, and drop everything else to save disk space.
            with np.load(out_file) as _saved:
                _ext = _saved["extrinsics"]
                _intr = _saved["intrinsics"]
                _depth_shape = _saved["depth"].shape  # (B, T, H, W, 1)
                _num = _saved["num_frames"]
            # depth shape is (B, T, H, W, 1) -> resolution = (H, W)
            _height = np.int64(_depth_shape[-3])
            _width = np.int64(_depth_shape[-2])
            # NOTE: numpy.savez auto-appends ".npz" if the path doesn't end
            # with it, so we must keep the ".npz" suffix on the tmp file.
            tmp_out = out_file.with_name(out_file.stem + ".tmp.npz")
            np.savez_compressed(
                tmp_out,
                extrinsics=_ext,
                intrinsics=_intr,
                num_frames=_num,
                height=_height,
                width=_width,
            )
            os.replace(tmp_out, out_file)

        elapsed_one = time.time() - t_start
        print(f"{prefix}[ OK ] {source.display} -> {out_file} "
              f"(took {_format_duration(elapsed_one)})", flush=True)
        return source.display, 0, elapsed_one
    except Exception as e:
        import traceback
        print(f"{prefix}[FAIL] {source.display}: {e}", flush=True)
        traceback.print_exc()
        return source.display, -1, time.time() - t_start
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def gpu_worker(worker_idx, gpu_id, gpu_tasks, checkpoint, image_resolution,
               max_frames, frame_stride, dry_run, cameras_only,
               result_queue, progress_queue=None):
    """Process all tasks assigned to one GPU. Sends results back via queue."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tag = f"GPU{gpu_id}" if gpu_id is not None else "GPU"
    n = len(gpu_tasks)
    print(f"[{tag}] Worker {worker_idx} started, "
          f"{n} video(s) assigned.", flush=True)

    failed = []
    model = None
    if not dry_run:
        import torch
        from vggt_omega.models import VGGTOmega
        print(f"[{tag}] Loading checkpoint: {checkpoint}", flush=True)
        model = VGGTOmega().to("cuda").eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    for i, (source, out_dir) in enumerate(gpu_tasks, 1):
        print(f"[{tag}] --- [{i}/{n}] {source.stem}{source.suffix} ---",
              flush=True)
        _, rc, elapsed_one = run_one(
            model, source, out_dir, image_resolution,
            max_frames, frame_stride, dry_run,
            cameras_only=cameras_only, tag=tag,
        )
        if rc != 0:
            failed.append(source.display)
        if progress_queue is not None:
            progress_queue.put((gpu_id, source.display, rc, elapsed_one))

    print(f"[{tag}] Worker finished. "
          f"Success: {n - len(failed)} / {n}", flush=True)
    result_queue.put((gpu_id, failed))


def parse_visible_devices():
    val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if val is None or val.strip() == "":
        return None
    return [x.strip() for x in val.split(",") if x.strip() != ""]


def auto_detect_gpu_ids() -> Optional[List[str]]:
    """Detect all GPU device IDs on the system using nvidia-smi.

    Returns a list of GPU index strings (e.g. ['0', '1', ..., '7']) or
    None if detection fails.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            ids = [line.strip() for line in result.stdout.strip().split("\n")
                   if line.strip()]
            if ids:
                return ids
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch-run VGGT-Omega inference on all videos in a "
                    "directory tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Either (a) a root directory containing videos (searched "
             "recursively), or (b) a .parquet file produced by "
             "parquet_util.py — videos will be read directly from it.",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Root output directory (mirrors input directory layout).",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to VGGT-Omega .pt checkpoint.",
    )
    parser.add_argument(
        "--image_resolution", type=int, default=512,
        help="Image resolution passed to load_and_preprocess_images "
             "(default: 512).",
    )
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="Max frames to use per video (0 = use all). Default: 0.",
    )
    parser.add_argument(
        "--frame_stride", type=int, default=1,
        help="Use every Nth frame from the decoded stream (default: 1).",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use in parallel. If >1, videos are split "
             "evenly across GPUs visible via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable auto-resume: do NOT skip videos whose .npz output "
             "already exists.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be processed without running inference.",
    )
    parser.add_argument(
        "--cameras_only", action="store_true",
        help="After inference, keep ONLY extrinsics/intrinsics/num_frames "
             "plus the (height, width) resolution derived from the depth "
             "tensor; drop depth/depth_conf/pose_enc/camera_tokens/registers "
             "to save disk space.",
    )
    parser.add_argument(
        "--use_parent_as_stem", action="store_true",
        help="Use the immediate parent directory name as the output stem "
             "instead of the video filename. Useful for datasets like "
             "minWM-data where every video is named gen.mp4 inside a "
             "uniquely-named subdirectory (e.g. videos/000000_right8a11/"
             "gen.mp4 -> output_dir/000000_right8a11.npz).",
    )
    args = parser.parse_args()

    # Accept either a directory or a .parquet file as the input source.
    if not (args.input_dir.is_dir() or is_parquet_path(args.input_dir)):
        print(
            f"Error: --input_dir must be an existing directory or a "
            f".parquet file: {args.input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    resume = not args.no_resume

    # ---- Discover videos & apply resume filter ----------------------------
    if is_parquet_path(args.input_dir):
        print(f"[SCAN] Reading video list from parquet: {args.input_dir}")
    else:
        print(f"[SCAN] Scanning videos under: {args.input_dir}")
    if args.use_parent_as_stem:
        print(f"[SCAN] Using parent directory name as output stem "
              f"(--use_parent_as_stem)")
    all_sources = discover_sources(args.input_dir,
                                   use_parent_as_stem=args.use_parent_as_stem)
    print(f"[SCAN] Found {len(all_sources)} video(s) total.")

    tasks: List[Tuple[VideoSource, Path]] = []
    skipped = 0
    for src in all_sources:
        out_dir = args.output_dir / src.rel_parent
        if resume and is_already_done(out_dir, src.stem):
            skipped += 1
            continue
        tasks.append((src, out_dir))

    total = len(tasks)
    if resume:
        print(f"[RESUME] Skipped {skipped} already-completed video(s); "
              f"{total} remaining.")
    else:
        print(f"[RESUME] Resume disabled; {total} video(s) to process.")

    if total == 0:
        print("Nothing to do.")
        return

    # ---- Decide GPU layout ------------------------------------------------
    visible = parse_visible_devices()
    if args.num_gpus <= 1:
        gpu_layout = [None]
        print(f"[GPU] Running sequentially on default GPU "
              f"(CUDA_VISIBLE_DEVICES={visible if visible else 'unset'}).")
    else:
        # If CUDA_VISIBLE_DEVICES is not set or has fewer GPUs than
        # requested, auto-detect all GPUs on the system via nvidia-smi.
        if visible is None or len(visible) < args.num_gpus:
            auto_ids = auto_detect_gpu_ids()
            if auto_ids is not None and len(auto_ids) > 0:
                if visible is None:
                    print(f"[GPU] CUDA_VISIBLE_DEVICES not set; auto-detected "
                          f"{len(auto_ids)} GPU(s): {auto_ids}")
                else:
                    print(f"[GPU] CUDA_VISIBLE_DEVICES has {len(visible)} "
                          f"GPU(s) but --num_gpus={args.num_gpus}; "
                          f"auto-detected {len(auto_ids)} GPU(s): {auto_ids}")
                visible = auto_ids
        if visible is None:
            print("Error: --num_gpus > 1 but no GPUs detected. Set "
                  "CUDA_VISIBLE_DEVICES or ensure nvidia-smi is available.",
                  file=sys.stderr)
            sys.exit(1)
        if args.num_gpus > len(visible):
            print(f"[GPU] Warning: --num_gpus={args.num_gpus} but only "
                  f"{len(visible)} GPU(s) available; using all "
                  f"{len(visible)}.", file=sys.stderr)
            args.num_gpus = len(visible)
        gpu_layout = visible[: args.num_gpus]
        print(f"[GPU] Using {len(gpu_layout)} GPU(s): {gpu_layout}")

    n_workers = len(gpu_layout)

    # ---- Split tasks evenly across GPUs ----------------------------------
    buckets = [[] for _ in range(n_workers)]
    for i, t in enumerate(tasks):
        buckets[i % n_workers].append(t)

    print("[SPLIT] Task distribution:")
    for w_idx, gpu_id in enumerate(gpu_layout):
        gid = gpu_id if gpu_id is not None else "default"
        print(f"  - worker {w_idx} (GPU {gid}): {len(buckets[w_idx])} video(s)")

    # ---- Run --------------------------------------------------------------
    t0 = time.time()
    failed = []
    completed = 0  # number of finished videos (success + fail)
    sum_elapsed = 0.0  # cumulative per-video elapsed seconds (wall-clock per video)

    def _log_progress(elapsed_one: float):
        """Update progress counters and log a per-video ETA line."""
        nonlocal completed, sum_elapsed
        completed += 1
        sum_elapsed += elapsed_one
        avg = sum_elapsed / completed
        remaining = total - completed
        # Effective parallelism speeds up wall-clock vs sum of per-video times.
        eta_wall = remaining * avg / max(n_workers, 1)
        wall_so_far = time.time() - t0
        eta_total_wall = wall_so_far + eta_wall
        print(
            f"[PROGRESS] {completed}/{total}  "
            f"last={_format_duration(elapsed_one)}  "
            f"avg/video={_format_duration(avg)}  "
            f"wall={_format_duration(wall_so_far)}  "
            f"ETA={_format_duration(eta_wall)}  "
            f"total≈{_format_duration(eta_total_wall)}",
            flush=True,
        )

    if n_workers == 1:
        # Single worker: run inline.
        gpu_id = gpu_layout[0]
        n = len(buckets[0])
        tag = f"GPU{gpu_id}" if gpu_id is not None else "GPU"

        model = None
        if not args.dry_run:
            import torch
            from vggt_omega.models import VGGTOmega
            print(f"[{tag}] Loading checkpoint: {args.checkpoint}", flush=True)
            model = VGGTOmega().to("cuda").eval()
            model.load_state_dict(
                torch.load(args.checkpoint, map_location="cpu"))

        for i, (source, out_dir) in enumerate(buckets[0], 1):
            print(f"\n[{tag}] --- [{i}/{n}] {source.stem}{source.suffix} ---",
                  flush=True)
            _, rc, elapsed_one = run_one(
                model, source, out_dir, args.image_resolution,
                args.max_frames, args.frame_stride, args.dry_run,
                cameras_only=args.cameras_only, tag=tag,
            )
            if rc != 0:
                failed.append(source.display)
            _log_progress(elapsed_one)
    else:
        # Multi-GPU: one worker process per GPU.
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        progress_queue = ctx.Queue()
        procs = []
        for w_idx, gpu_id in enumerate(gpu_layout):
            p = ctx.Process(
                target=gpu_worker,
                args=(w_idx, gpu_id, buckets[w_idx], args.checkpoint,
                      args.image_resolution, args.max_frames,
                      args.frame_stride, args.dry_run, args.cameras_only,
                      result_queue, progress_queue),
                daemon=False,
            )
            p.start()
            procs.append(p)
            print(f"[SPAWN] Launched worker pid={p.pid} on GPU {gpu_id}.")

        finished_workers = 0
        while finished_workers < n_workers:
            # Drain any pending progress events first (non-blocking).
            while True:
                try:
                    _gpu_id, _vp, _rc, elapsed_one = progress_queue.get_nowait()
                except _queue.Empty:
                    break
                _log_progress(elapsed_one)

            # Then check for a worker-completion event with a short timeout
            # so we keep draining progress promptly.
            try:
                gpu_id, w_failed = result_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            failed.extend(w_failed)
            finished_workers += 1
            print(f"[DONE] GPU {gpu_id} worker reported. "
                  f"({finished_workers}/{n_workers} workers finished, "
                  f"{len(w_failed)} failed on this GPU.)")

        # Final drain of progress queue (in case workers pushed progress
        # right before reporting completion).
        while True:
            try:
                _gpu_id, _vp, _rc, elapsed_one = progress_queue.get_nowait()
            except _queue.Empty:
                break
            _log_progress(elapsed_one)

        for p in procs:
            p.join()

    # ---- Summary ----------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total: {total}  |  Success: {total - len(failed)}  "
          f"|  Failed: {len(failed)}  |  Time: {_format_duration(elapsed)} "
          f"({elapsed:.1f}s)")
    if failed:
        print("Failed videos:")
        for v in failed:
            print(f"  - {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
