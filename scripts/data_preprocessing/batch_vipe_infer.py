#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch-run VIPE inference on all videos under a directory tree.

Recursively finds every video file under ``--input_dir`` and runs the VIPE
inference command for each one. Results are written into ``--output_dir`` using
VIPE's default sub-folder layout (``pose/``, ``depth/``, ``rgb/``,
``intrinsics/``, ``mask/``, ``vipe/``).

Features
--------
* **Resume / skip-existing** (default on): videos whose VIPE outputs already
  exist are skipped automatically. Disable with ``--no_resume``.
* **Multi-GPU parallelism**: pass ``--num_gpus N`` (N > 1) to split the
  remaining videos evenly across N GPUs. Each GPU runs in its own worker
  process with ``CUDA_VISIBLE_DEVICES`` set to a single physical device.
* **Per-video timing & ETA**: each video's processing time is printed, and an
  estimated remaining time is shown based on the average per-video duration.

Example
-------
    CUDA_VISIBLE_DEVICES=0,3 python batch_vipe_infer.py \\
        --input_dir  /nfs/yixinyang/code/LongLive/data/Sekai/video \\
        --output_dir /nfs/yixinyang/code/LongLive/data/Sekai/vipe_results \\
        --vipe_bin   /nfs/yixinyang/miniconda3/envs/vipe/bin/vipe \\
        --preset     lyra \\
        --num_gpus   2
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}

# VIPE writes outputs into these sub-folders. We require all of these to exist
# (with the corresponding suffix) before we consider a video "done".
RESUME_CHECK_FILES = [
    ("pose",       ".npz"),
    ("depth",      ".zip"),
    ("intrinsics", ".npz"),
]


def find_videos(input_dir: Path):
    """Yield (video_path, relative_parent) for every video under *input_dir*."""
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                video_path = Path(root) / f
                rel_parent = video_path.parent.relative_to(input_dir)
                yield video_path, rel_parent


def is_already_done(out_dir: Path, stem: str) -> bool:
    """Return True if VIPE outputs for *stem* already exist under *out_dir*."""
    for sub, ext in RESUME_CHECK_FILES:
        if not (out_dir / sub / f"{stem}{ext}").exists():
            return False
    return True


def run_one(vipe_bin, preset, video_path, out_dir, extra_args, dry_run,
            gpu_id=None, tag=""):
    """Run VIPE inference for a single video.

    Returns ``(video_path, returncode, elapsed_seconds)``.
    """
    cmd = [vipe_bin, "infer", "-p", preset, "-o", str(out_dir), str(video_path)]
    if extra_args:
        cmd.extend(extra_args)

    prefix = f"[{tag}] " if tag else ""

    if dry_run:
        env_str = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        print(f"{prefix}[DRY-RUN] {video_path}")
        print(f"{prefix}          {env_str}{' '.join(cmd)}")
        return video_path, 0, 0.0

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"{prefix}[RUN] {video_path}", flush=True)
    t_start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        elapsed = time.time() - t_start
        if result.returncode != 0:
            print(f"{prefix}[FAIL] {video_path} (exit {result.returncode}, "
                  f"{elapsed:.1f}s)", flush=True)
            if result.stderr:
                print(result.stderr[-2000:], flush=True)
        else:
            print(f"{prefix}[ OK ] {video_path} ({elapsed:.1f}s)", flush=True)
        return video_path, result.returncode, elapsed
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"{prefix}[ERR ] {video_path}: {e} ({elapsed:.1f}s)", flush=True)
        return video_path, -1, elapsed


def gpu_worker(worker_idx, gpu_id, gpu_tasks, vipe_bin, preset, extra_args,
               dry_run, result_queue, total_tasks, done_counter, done_lock):
    """Process all tasks assigned to one GPU. Sends results back via queue.

    *total_tasks* is the grand total across all workers; *done_counter* is a
    shared ``mp.Value`` for tracking how many videos have finished globally;
    *done_lock* guards updates to the counter. These are used to compute an
    ETA across all GPUs.
    """
    tag = f"GPU{gpu_id}"
    n = len(gpu_tasks)
    print(f"[{tag}] Worker {worker_idx} started, "
          f"{n} video(s) assigned.", flush=True)

    failed = []
    for i, (video_path, out_dir) in enumerate(gpu_tasks, 1):
        t0 = time.time()
        print(f"[{tag}] --- [{i}/{n}] {video_path.name} ---", flush=True)
        _, rc, elapsed = run_one(vipe_bin, preset, video_path, out_dir,
                                 extra_args, dry_run, gpu_id=gpu_id, tag=tag)

        # Update global progress counter and compute ETA.
        with done_lock:
            done_counter.value += 1
            done = done_counter.value
        remaining_global = total_tasks - done
        # Use exponential moving average for smoother ETA.
        if done == 1:
            avg = elapsed
        else:
            # We don't have history of all prior videos here; approximate
            # using the current video's time as a rough average.
            # The main process prints a better summary based on wall-clock.
            avg = elapsed  # fallback; main process has more info
        eta_str = _format_eta(avg * remaining_global)

        print(f"[{tag}] Done [{i}/{n}] | "
              f"Global: {done}/{total_tasks} | "
              f"This video: {elapsed:.1f}s | "
              f"ETA: {eta_str}",
              flush=True)

        if rc != 0:
            failed.append(str(video_path))

    print(f"[{tag}] Worker finished. "
          f"Success: {n - len(failed)} / {n}", flush=True)
    result_queue.put((gpu_id, failed))


def parse_visible_devices():
    """Return the list of GPU ids visible to this process via
    ``CUDA_VISIBLE_DEVICES``. If not set, returns None (=> use default cuda)."""
    val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if val is None or val.strip() == "":
        return None
    return [x.strip() for x in val.split(",") if x.strip() != ""]


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like ``1h23m45s``."""
    if seconds < 0:
        return "N/A"
    td = timedelta(seconds=int(seconds))
    total_sec = int(td.total_seconds())
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_eta(seconds: float) -> str:
    """Alias for ``_fmt_duration`` used in ETA context."""
    return _fmt_duration(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run VIPE inference on all videos in a directory tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Root directory containing videos (searched recursively).",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Root output directory (VIPE default subfolder layout).",
    )
    parser.add_argument(
        "--vipe_bin", default="/nfs/yixinyang/miniconda3/envs/vipe/bin/vipe",
        help="Path to the vipe executable.",
    )
    parser.add_argument(
        "--preset", "-p", default="lyra",
        help="VIPE preset name (default: lyra).",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use in parallel. If >1, the remaining videos "
             "are split evenly across GPUs visible to this process via "
             "CUDA_VISIBLE_DEVICES (default: 1 = sequential on one GPU).",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable auto-resume: do NOT skip videos whose outputs already "
             "exist. By default resume is ON.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra_args", nargs=argparse.REMAINDER, default=[],
        help="Extra arguments appended to every vipe infer command "
             "(e.g. --extra_args --some_flag value).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input_dir does not exist: {args.input_dir}",
              file=sys.stderr)
        sys.exit(1)

    resume = not args.no_resume

    # ---- Discover videos & apply resume filter ----------------------------
    print(f"[SCAN] Scanning videos under: {args.input_dir}")
    all_videos = list(find_videos(args.input_dir))
    print(f"[SCAN] Found {len(all_videos)} video file(s) total.")

    tasks = []   # list of (video_path, out_dir)
    skipped = 0
    for video_path, rel_parent in all_videos:
        out_dir = args.output_dir / rel_parent
        if resume and is_already_done(out_dir, video_path.stem):
            skipped += 1
            continue
        tasks.append((video_path, out_dir))

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
        # Single-GPU / sequential mode.
        # We don't override CUDA_VISIBLE_DEVICES; vipe picks whatever is set.
        gpu_layout = [None]
        print(f"[GPU] Running sequentially on default GPU "
              f"(CUDA_VISIBLE_DEVICES={visible if visible else 'unset'}).")
    else:
        if visible is None:
            print("Error: --num_gpus > 1 requires CUDA_VISIBLE_DEVICES to be "
                  "set (e.g. CUDA_VISIBLE_DEVICES=0,3).", file=sys.stderr)
            sys.exit(1)
        if args.num_gpus > len(visible):
            print(f"Error: --num_gpus={args.num_gpus} but only "
                  f"{len(visible)} GPU(s) visible: {visible}",
                  file=sys.stderr)
            sys.exit(1)
        gpu_layout = visible[: args.num_gpus]
        print(f"[GPU] Using {len(gpu_layout)} GPU(s): {gpu_layout}")

    n_workers = len(gpu_layout)

    # ---- Split tasks evenly across GPUs ----------------------------------
    # Round-robin assignment so any size mismatch differs by at most 1.
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
    total_done = 0  # global counter for single-GPU mode

    if n_workers == 1:
        # Single worker: run inline (no subprocess spawn) for cleaner logs.
        gpu_id = gpu_layout[0]
        n = len(buckets[0])
        tag = f"GPU{gpu_id}" if gpu_id is not None else "GPU"
        for i, (video_path, out_dir) in enumerate(buckets[0], 1):
            print(f"\n[{tag}] --- [{i}/{n}] {video_path.name} ---", flush=True)
            t_video_start = time.time()
            _, rc, elapsed = run_one(
                args.vipe_bin, args.preset, video_path, out_dir,
                args.extra_args, args.dry_run,
                gpu_id=gpu_id, tag=tag)
            t_video_end = time.time()

            total_done += 1
            remaining = total - total_done
            # Compute running average based on wall-clock time.
            wall_elapsed = t_video_end - t0
            avg_per_video = wall_elapsed / total_done
            eta = avg_per_video * remaining

            print(
                f"[{tag}] Progress: {total_done}/{total} | "
                f"This video: {elapsed:.1f}s | "
                f"Avg: {avg_per_video:.1f}s | "
                f"Elapsed: {_fmt_duration(wall_elapsed)} | "
                f"ETA: {_fmt_duration(eta)}",
                flush=True)

            if rc != 0:
                failed.append(str(video_path))
    else:
        # Multi-GPU: one worker process per GPU.
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        # Shared counter for global progress across workers.
        done_counter = ctx.Value("i", 0)
        done_lock = ctx.Lock()

        procs = []
        for w_idx, gpu_id in enumerate(gpu_layout):
            p = ctx.Process(
                target=gpu_worker,
                args=(w_idx, gpu_id, buckets[w_idx], args.vipe_bin,
                      args.preset, args.extra_args, args.dry_run,
                      result_queue, total, done_counter, done_lock),
                daemon=False,
            )
            p.start()
            procs.append(p)
            print(f"[SPAWN] Launched worker pid={p.pid} on GPU {gpu_id}.")

        # Spawn a lightweight monitor thread to print periodic ETA summaries.
        import threading

        stop_monitor = threading.Event()

        def _monitor():
            while not stop_monitor.is_set():
                time.sleep(30)
                if stop_monitor.is_set():
                    break
                with done_lock:
                    done = done_counter.value
                wall = time.time() - t0
                if done > 0:
                    avg = wall / done
                    eta = avg * (total - done)
                    print(
                        f"[MONITOR] Progress: {done}/{total} "
                        f"({done/total*100:.1f}%) | "
                        f"Elapsed: {_fmt_duration(wall)} | "
                        f"Avg/video: {avg:.1f}s | "
                        f"ETA: {_fmt_duration(eta)}",
                        flush=True)

        mon_thread = threading.Thread(target=_monitor, daemon=True)
        mon_thread.start()

        # Collect results.
        finished = 0
        while finished < n_workers:
            gpu_id, w_failed = result_queue.get()
            failed.extend(w_failed)
            finished += 1
            with done_lock:
                done = done_counter.value
            wall = time.time() - t0
            if done > 0:
                avg = wall / done
                eta = avg * (total - done)
            else:
                avg = 0
                eta = 0
            print(f"[DONE] GPU {gpu_id} worker reported. "
                  f"({finished}/{n_workers} workers finished, "
                  f"{len(w_failed)} failed on this GPU.) "
                  f"Global progress: {done}/{total} | "
                  f"Elapsed: {_fmt_duration(wall)} | "
                  f"ETA: {_fmt_duration(eta)}",
                  flush=True)

        stop_monitor.set()
        mon_thread.join(timeout=1)

        for p in procs:
            p.join()

    # ---- Summary ----------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total: {total}  |  Success: {total - len(failed)}  "
          f"|  Failed: {len(failed)}  |  "
          f"Time: {elapsed:.1f}s ({_fmt_duration(elapsed)})")
    if failed:
        print("Failed videos:")
        for v in failed:
            print(f"  - {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
