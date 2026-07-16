#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract selected Roblox games directly from P2P ``batch_*.tar.gz`` files.

The selection is based on the authoritative ``metadata.env`` fields in each
``annotation.proto``.  By default it extracts all recordings normalized as
``be-a-tornado``, ``be-a-shark``, or ``be-a-snake`` and writes the original
video and annotation directly under each game directory:

    <output_dir>/<game>/<uuid>.mp4
    <output_dir>/<game>/<uuid>.proto

The extraction is resumable: a sample that already has both files is skipped.
Archives are streamed, so the P2P dataset need not first be unpacked in its
entirety. The auxiliary ``192x192.mp4`` is deliberately not extracted.

    python scripts/extract_p2p_roblox_games.py \
        --p2p_path /nfs/hongfenglai/p2p-full-data \
        --output_dir /nfs/yixinyang/code/LongLive/data/p2pfull
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse the P2P protobuf decoder and Roblox subtype normalization maintained by
# the direct visualizer.  This keeps extraction and visualization labels
# identical, including known spelling variants in the source metadata.
from visualize_p2p_sample import (  # noqa: E402
    is_roblox_environment,
    normalize_game_name,
    parse_annotation,
)


DEFAULT_GAMES = ("be-a-tornado", "be-a-shark", "be-a-snake")
REQUIRED_SUFFIXES = (".mp4", ".proto")


def list_archives(p2p_path: Path) -> list[Path]:
    dataset_dir = p2p_path / "dataset" if (p2p_path / "dataset").is_dir() else p2p_path
    archives = sorted(dataset_dir.glob("batch_*.tar.gz"))
    if not archives:
        raise RuntimeError(f"No batch_*.tar.gz archives found under {dataset_dir}")
    return archives


def is_complete(game_dir: Path, sample_id: str) -> bool:
    return all((game_dir / f"{sample_id}{suffix}").is_file() for suffix in REQUIRED_SUFFIXES)


def copy_member(handle: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    source = handle.extractfile(member)
    if source is None:
        raise RuntimeError(f"Could not extract {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def extract_archive(
    archive: Path, output_dir: Path, games: set[str], *, overwrite: bool,
) -> tuple[Counter, int]:
    """Extract matching records from one archive in a single stream pass."""
    extracted: Counter = Counter()
    skipped_complete = 0
    active: dict[str, tuple[Path, str]] = {}

    with tarfile.open(archive, mode="r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            sample_id, filename = parts[-2], parts[-1]

            if filename == "annotation.proto":
                source = handle.extractfile(member)
                if source is None:
                    continue
                annotation_bytes = source.read()
                annotation = parse_annotation(annotation_bytes, f"{archive.name}:{member.name}")
                if not is_roblox_environment(annotation.metadata.env.env):
                    continue
                game = normalize_game_name(annotation.metadata.env.env_subtype)
                if game not in games:
                    continue

                game_dir = output_dir / game
                if not overwrite and is_complete(game_dir, sample_id):
                    skipped_complete += 1
                    continue
                game_dir.mkdir(parents=True, exist_ok=True)
                annotation_path = game_dir / f"{sample_id}.proto"
                if overwrite or not annotation_path.is_file():
                    annotation_path.write_bytes(annotation_bytes)
                active[sample_id] = (game_dir, game)
                continue

            if sample_id not in active or filename != "video.mp4":
                continue
            game_dir, game = active[sample_id]
            destination = game_dir / f"{sample_id}.mp4"
            if overwrite or not destination.is_file():
                copy_member(handle, member, destination)
            if is_complete(game_dir, sample_id):
                extracted[game] += 1
            active.pop(sample_id, None)

    # A malformed archive might omit one required member.  Preserve partial
    # output so a subsequent resumable run can fill it, but make it visible.
    for sample_id, (game_dir, game) in active.items():
        missing = [
            f"{sample_id}{suffix}"
            for suffix in REQUIRED_SUFFIXES
            if not (game_dir / f"{sample_id}{suffix}").is_file()
        ]
        print(f"[WARN] {archive.name}:{sample_id} ({game}) is incomplete; missing {missing}")
    return extracted, skipped_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected Roblox P2P samples from tar archives."
    )
    parser.add_argument(
        "--p2p_path", type=Path,
        default=Path("/nfs/hongfenglai/p2p-full-data"),
        help="P2P root containing dataset/batch_*.tar.gz.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("/nfs/yixinyang/code/LongLive/data/p2pfull"),
        help="Directory receiving <game>/<uuid>.{mp4,proto}.",
    )
    parser.add_argument(
        "--games", nargs="+", default=list(DEFAULT_GAMES),
        help="Normalized Roblox game names to extract.",
    )
    parser.add_argument(
        "--max_archives", type=int, default=0,
        help="Process only this many archives (0 means every available archive).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rewrite existing sample files instead of resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_archives < 0:
        raise ValueError("--max_archives must be non-negative")
    args.p2p_path = args.p2p_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    games = {normalize_game_name(name) for name in args.games}
    archives = list_archives(args.p2p_path)
    if args.max_archives:
        archives = archives[: args.max_archives]

    print(
        f"Processing {len(archives)} archive(s); games={sorted(games)}; "
        f"output={args.output_dir}; overwrite={args.overwrite}"
    )
    totals: Counter = Counter()
    skipped_complete = 0
    for index, archive in enumerate(archives, start=1):
        print(f"[{index}/{len(archives)}] {archive.name}", flush=True)
        try:
            extracted, skipped = extract_archive(
                archive, args.output_dir, games, overwrite=args.overwrite
            )
        except (OSError, tarfile.TarError, ValueError, RuntimeError) as exc:
            print(f"[WARN] Failed {archive.name}: {exc}")
            continue
        totals.update(extracted)
        skipped_complete += skipped

    print(
        "Done. Newly completed samples: "
        + ", ".join(f"{game}={totals[game]}" for game in sorted(games))
        + f"; already complete and skipped={skipped_complete}"
    )


if __name__ == "__main__":
    main()
