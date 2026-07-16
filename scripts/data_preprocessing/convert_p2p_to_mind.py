#!/usr/bin/env python3
"""Convert extracted P2P ``mp4/proto`` pairs to the MIND input layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_p2p_sample import parse_annotation, select_action


def _encode(keys: set[str], positive: str, negative: str) -> int:
    pos, neg = positive in keys, negative in keys
    if pos == neg:
        return 0
    return 1 if pos else 2


def _convert_frame(frame_annotation) -> dict[str, int]:
    action, _ = select_action(frame_annotation)
    keys = {str(key).strip().lower() for key in action.keyboard.keys}
    return {
        "ws": _encode(keys, "w", "s"),
        "ad": _encode(keys, "a", "d"),
        "ud": _encode(keys, "uparrow", "downarrow"),
        "lr": _encode(keys, "leftarrow", "rightarrow"),
    }


def convert(video_path: Path, output_root: Path) -> None:
    proto_path = video_path.with_suffix(".proto")
    annotation = parse_annotation(proto_path.read_bytes(), str(proto_path))
    if not annotation.frame_annotations:
        raise ValueError("empty frame_annotations")

    sample_dir = output_root / video_path.parent.name / video_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    target_video = sample_dir / "video.mp4"
    if target_video.exists() or target_video.is_symlink():
        target_video.unlink()
    target_video.symlink_to(video_path.resolve())

    frames = [_convert_frame(frame) for frame in annotation.frame_annotations]
    env = getattr(annotation.metadata.env, "env_subtype", "")
    caption = f"A high-quality gameplay video in {env or video_path.parent.name}."
    (sample_dir / "action.json").write_text(
        json.dumps(
            {"data": frames, "total_time": len(frames), "caption": caption},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted P2P mp4/proto pairs to MIND layout."
    )
    parser.add_argument("--p2p_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    converted = skipped = 0
    for video_path in sorted(args.p2p_path.rglob("*.mp4")):
        if not video_path.with_suffix(".proto").is_file():
            skipped += 1
            continue
        try:
            convert(video_path, args.output_dir)
            converted += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] skipped {video_path}: {exc}")
    print(f"Converted {converted} P2P samples to {args.output_dir}")
    if skipped:
        print(f"Skipped {skipped} files")


if __name__ == "__main__":
    main()
