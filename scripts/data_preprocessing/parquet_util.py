#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack videos to & from a Parquet file (lossless, byte-exact).

This utility provides two operations:

* ``pack``   — recursively scan an input directory for video files, read each
  one as raw bytes, and write all of them into a single ``.parquet`` file.
* ``unpack`` — read a ``.parquet`` file produced by ``pack`` and write every
  video back to disk under an output directory, preserving the original
  relative path, file name, extension, and exact byte content.

Because we store the raw bytes verbatim (no re-encoding, no compression of
the per-row binary payload by default), round-tripping is **byte-for-byte
lossless**: ``sha256(original) == sha256(unpacked)``.

Schema of the produced Parquet file
-----------------------------------
Each row corresponds to one video file:

    relpath : string   # path relative to --input_dir, using forward slashes
    name    : string   # basename of the file (e.g. "clip_001.mp4")
    size    : int64    # original file size in bytes
    sha256  : string   # hex digest of the original bytes
    data    : binary   # raw file bytes

Examples
--------
Pack every video under a folder into one parquet file::

    python parquet_util.py pack \
        --input_dir  /path/to/videos \
        --output     /path/to/videos.parquet

Unpack the parquet file back to a directory tree::

    python parquet_util.py unpack \
        --input      /path/to/videos.parquet \
        --output_dir /path/to/restored_videos

Verify that round-tripping is lossless (compare hashes)::

    python parquet_util.py unpack \
        --input      /path/to/videos.parquet \
        --output_dir /path/to/restored_videos \
        --verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm", ".m4v", ".ts",
              ".mpg", ".mpeg", ".wmv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f}{u}" if u != "B" else f"{int(f)}{u}"
        f /= 1024.0
    return f"{n}B"


def find_videos(input_dir: Path) -> List[Tuple[Path, str]]:
    """Return [(abs_path, rel_path_using_forward_slashes), ...] sorted."""
    items: List[Tuple[Path, str]] = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in VIDEO_EXTS:
                rel = p.relative_to(input_dir).as_posix()
                items.append((p, rel))
    items.sort(key=lambda x: x[1])
    return items


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = pa.schema([
    pa.field("relpath", pa.string()),
    pa.field("name",    pa.string()),
    pa.field("size",    pa.int64()),
    pa.field("sha256",  pa.string()),
    pa.field("data",    pa.large_binary()),  # large_binary supports >2GB total
])


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------
def _iter_row_batches(items: List[Tuple[Path, str]],
                      rows_per_batch: int) -> Iterator[pa.RecordBatch]:
    """Yield Arrow RecordBatches of at most *rows_per_batch* rows each.

    Reads files lazily so we never hold more than one batch in memory.
    """
    relpaths: List[str] = []
    names: List[str] = []
    sizes: List[int] = []
    digests: List[str] = []
    blobs: List[bytes] = []

    def _flush() -> pa.RecordBatch:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(relpaths, type=pa.string()),
                pa.array(names,    type=pa.string()),
                pa.array(sizes,    type=pa.int64()),
                pa.array(digests,  type=pa.string()),
                pa.array(blobs,    type=pa.large_binary()),
            ],
            schema=SCHEMA,
        )
        relpaths.clear(); names.clear(); sizes.clear()
        digests.clear(); blobs.clear()
        return batch

    for abs_path, rel in items:
        with open(abs_path, "rb") as fp:
            data = fp.read()
        relpaths.append(rel)
        names.append(abs_path.name)
        sizes.append(len(data))
        digests.append(_sha256_bytes(data))
        blobs.append(data)

        if len(relpaths) >= rows_per_batch:
            yield _flush()

    if relpaths:
        yield _flush()


def pack(input_dir: Path, output: Path, *,
         rows_per_batch: int = 8,
         compression: str = "none") -> None:
    """Pack every video under *input_dir* into a single Parquet at *output*.

    Parameters
    ----------
    input_dir : Path
        Root directory to scan recursively for video files.
    output : Path
        Output ``.parquet`` file path. Parent directories are created.
    rows_per_batch : int
        Number of videos per Arrow RecordBatch / Parquet row group.
        Smaller = less memory at write time; larger = potentially better I/O.
    compression : str
        Parquet column compression: one of ``none``, ``snappy``, ``gzip``,
        ``brotli``, ``zstd``, ``lz4``. Lossless regardless of choice — this
        only affects on-disk size, not byte-correctness of the round-trip.
        Default ``none`` because video files are usually already compressed.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    items = find_videos(input_dir)
    if not items:
        print(f"[PACK] No videos found under {input_dir}.")
        return

    total_bytes = sum(p.stat().st_size for p, _ in items)
    print(f"[PACK] Found {len(items)} video(s), "
          f"total {_format_bytes(total_bytes)}.")
    print(f"[PACK] Writing -> {output}  "
          f"(compression={compression}, rows_per_batch={rows_per_batch})")

    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    written = 0
    written_bytes = 0
    writer = pq.ParquetWriter(
        str(output), SCHEMA,
        compression=None if compression == "none" else compression,
    )
    try:
        for batch in _iter_row_batches(items, rows_per_batch=rows_per_batch):
            writer.write_batch(batch)
            written += batch.num_rows
            written_bytes += int(
                pa.compute.sum(batch.column("size")).as_py() or 0
            )
            print(f"[PACK]   {written}/{len(items)} packed  "
                  f"({_format_bytes(written_bytes)} so far, "
                  f"elapsed {_format_duration(time.time() - t0)})",
                  flush=True)
    finally:
        writer.close()

    out_size = output.stat().st_size
    print(f"[PACK] Done. {written} video(s), "
          f"raw={_format_bytes(total_bytes)}, "
          f"parquet={_format_bytes(out_size)}, "
          f"time={_format_duration(time.time() - t0)}.")


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------
def unpack(input_parquet: Path, output_dir: Path, *,
           verify: bool = False, overwrite: bool = False) -> None:
    """Restore every video from *input_parquet* into *output_dir*.

    File names, extensions, relative directory layout, and bytes are all
    preserved exactly.

    Parameters
    ----------
    verify : bool
        If True, recompute SHA-256 of the restored bytes and compare against
        the digest stored in the parquet file. Aborts on mismatch.
    overwrite : bool
        If True, existing files are overwritten. If False (default), existing
        files are skipped.
    """
    if not input_parquet.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {input_parquet}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(input_parquet))
    total_rows = pf.metadata.num_rows
    print(f"[UNPACK] {input_parquet} -> {output_dir}  "
          f"(rows={total_rows}, verify={verify}, overwrite={overwrite})")

    t0 = time.time()
    done = 0
    skipped = 0
    written_bytes = 0
    for batch in pf.iter_batches(batch_size=8):
        relpaths = batch.column("relpath").to_pylist()
        sizes    = batch.column("size").to_pylist()
        digests  = batch.column("sha256").to_pylist()
        datas    = batch.column("data").to_pylist()

        for relpath, size, digest, data in zip(relpaths, sizes,
                                               digests, datas):
            out_path = output_dir / relpath
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not overwrite:
                skipped += 1
                done += 1
                continue

            if not isinstance(data, (bytes, bytearray)):
                # pyarrow returns bytes for binary/large_binary; be defensive.
                data = bytes(data)

            if size is not None and len(data) != size:
                raise RuntimeError(
                    f"Size mismatch for {relpath}: "
                    f"stored size={size}, actual={len(data)}")

            if verify:
                got = _sha256_bytes(data)
                if got != digest:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {relpath}: "
                        f"expected {digest}, got {got}")

            # Atomic-ish write: write to .tmp then rename.
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp_path, "wb") as fp:
                fp.write(data)
            os.replace(tmp_path, out_path)

            written_bytes += len(data)
            done += 1

        print(f"[UNPACK]   {done}/{total_rows} restored  "
              f"({_format_bytes(written_bytes)} written, "
              f"skipped={skipped}, "
              f"elapsed {_format_duration(time.time() - t0)})",
              flush=True)

    print(f"[UNPACK] Done. {done} row(s), "
          f"written={_format_bytes(written_bytes)}, "
          f"skipped(existing)={skipped}, "
          f"time={_format_duration(time.time() - t0)}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lossless pack/unpack of a folder of videos to/from a "
                    "single Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="Pack a directory of videos into "
                                         "a parquet file.")
    p_pack.add_argument("--input_dir", required=True, type=Path,
                        help="Root directory to scan for videos.")
    p_pack.add_argument("--output", required=True, type=Path,
                        help="Output parquet file path.")
    p_pack.add_argument("--rows_per_batch", type=int, default=8,
                        help="Videos per row group / batch (default: 8). "
                             "Lower this if you hit memory pressure.")
    p_pack.add_argument("--compression", type=str, default="none",
                        choices=["none", "snappy", "gzip", "brotli",
                                 "zstd", "lz4"],
                        help="Parquet column compression (default: none). "
                             "All choices are lossless.")

    p_unp = sub.add_parser("unpack", help="Unpack a parquet file back into "
                                          "a directory of videos.")
    p_unp.add_argument("--input", required=True, type=Path,
                       help="Input parquet file path.")
    p_unp.add_argument("--output_dir", required=True, type=Path,
                       help="Output root directory.")
    p_unp.add_argument("--verify", action="store_true",
                       help="Recompute SHA-256 of restored bytes and assert "
                            "it matches the digest stored at pack time.")
    p_unp.add_argument("--overwrite", action="store_true",
                       help="Overwrite files that already exist in the "
                            "output directory (default: skip).")

    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "pack":
            pack(args.input_dir, args.output,
                 rows_per_batch=args.rows_per_batch,
                 compression=args.compression)
        elif args.cmd == "unpack":
            unpack(args.input, args.output_dir,
                   verify=args.verify, overwrite=args.overwrite)
        else:
            print(f"Unknown command: {args.cmd}", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
