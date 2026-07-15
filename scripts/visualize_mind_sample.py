#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visualize raw MIND clips with their discrete action controls.

Each MIND sample is expected to contain ``video.mp4`` and ``action.json``.
The overlay is driven directly by the per-frame ``ws/ad/ud/lr`` values;
camera and actor poses are deliberately not used as control signals.

MIND's official three-state encoding is:

* 0: no operation;
* 1: forward / left / look up / look left;
* 2: backward / right / look down / look right.

Frames are decoded and encoded one at a time, so full-length 1080p clips do
not have to be held in memory.

Example (from the LongLive repository root)::

    python scripts/visualize_mind_sample.py \
        --mind_path data/MIND/3rd_data/train \
        --output_dir ./lmdb_vis/MIND_3rd \
        --num_samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Make the repository root importable regardless of the current directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.action_overlay import ActionOverlayRenderer  # noqa: E402


_SAMPLE_RE = re.compile(r"^data-(\d+)$")
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def list_mind_samples(mind_path: Path) -> list[tuple[int, Path]]:
    """Return numerically sorted ``(sample_id, sample_dir)`` pairs."""
    samples = []
    if not mind_path.is_dir():
        raise FileNotFoundError(f"MIND directory does not exist: {mind_path}")
    for child in mind_path.iterdir():
        match = _SAMPLE_RE.match(child.name)
        if not match or not child.is_dir():
            continue
        if (child / "video.mp4").is_file() and (child / "action.json").is_file():
            samples.append((int(match.group(1)), child))
    samples.sort(key=lambda item: item[0])
    return samples


def load_action_json(action_path: Path) -> dict:
    """Load MIND metadata and validate the per-frame action fields."""
    with action_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    frames = metadata.get("data", [])
    if not frames:
        raise ValueError(f"No frames in {action_path}")
    required = {"ws", "ad", "ud", "lr"}
    for i, frame in enumerate(frames):
        missing = required.difference(frame)
        if missing:
            raise ValueError(f"Frame {i} in {action_path} is missing {sorted(missing)}")
    return metadata


def decode_mind_action(frame: dict) -> tuple[list[str], float, float, str]:
    """Decode one official MIND action into overlay inputs and a label."""
    values = {name: int(frame[name]) for name in ("ws", "ad", "ud", "lr")}
    invalid = {name: value for name, value in values.items()
               if value not in (0, 1, 2)}
    if invalid:
        raise ValueError(f"Invalid MIND actions {invalid}; expected values 0, 1, or 2")

    keys: list[str] = []
    names: list[str] = []
    if values["ws"] == 1:
        keys.append("W")
        names.append("forward")
    elif values["ws"] == 2:
        keys.append("S")
        names.append("backward")

    if values["ad"] == 1:
        keys.append("A")
        names.append("left")
    elif values["ad"] == 2:
        keys.append("D")
        names.append("right")

    # The renderer uses negative displacement for left/up and positive
    # displacement for right/down.
    yaw = -1.0 if values["lr"] == 1 else 1.0 if values["lr"] == 2 else 0.0
    pitch = -1.0 if values["ud"] == 1 else 1.0 if values["ud"] == 2 else 0.0
    if values["ud"] == 1:
        names.append("look_up")
    elif values["ud"] == 2:
        names.append("look_down")
    if values["lr"] == 1:
        names.append("look_left")
    elif values["lr"] == 2:
        names.append("look_right")

    return keys, yaw, pitch, "+".join(names) if names else "no_op"


def _select_samples(
    samples: list[tuple[int, Path]], sample_id: int | None,
    num_samples: int, seed: int,
) -> list[tuple[int, Path]]:
    if num_samples < 1:
        raise ValueError("--num_samples must be at least 1")
    if sample_id is not None:
        start = next((i for i, item in enumerate(samples) if item[0] == sample_id), None)
        if start is None:
            raise ValueError(f"data-{sample_id} was not found")
        return samples[start : start + num_samples]
    rng = random.Random(seed)
    return sorted(
        rng.sample(samples, min(num_samples, len(samples))), key=lambda item: item[0]
    )


def _draw_information(
    image: Image.Image,
    *,
    sample_name: str,
    source_frame: int,
    action_frame: dict,
    action_name: str,
    phase: str,
) -> None:
    """Draw the decoded action, raw values, frame index, and phase."""
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(max(14, int(height * 0.021)))
    line_height = max(20, int(height * 0.029))
    raw_action = "  ".join(
        f"{key.upper()}={action_frame[key]}" for key in ("ws", "ad", "ud", "lr")
    )
    lines = [
        f"{sample_name} | frame {source_frame} | {phase}",
        f"Action: {action_name}",
        f"Raw: {raw_action}",
    ]
    panel_width = min(width - 24, max(560, int(width * 0.43)))
    panel_height = line_height * len(lines) + 18
    draw.rounded_rectangle(
        (10, 10, 10 + panel_width, 10 + panel_height),
        radius=10,
        fill=(0, 0, 0, 142),
        outline=(255, 255, 255, 55),
        width=1,
    )
    y = 18
    for line in lines:
        draw.text((20, y), line, font=font, fill=(255, 255, 255, 235))
        y += line_height


def visualize_sample(sample_id: int, sample_dir: Path, output_dir: Path, args) -> Path:
    """Stream one MIND clip to an annotated MP4 and return its path."""
    metadata = load_action_json(sample_dir / "action.json")
    action_frames = metadata["data"]
    start = args.start_frame
    if start < 0 or start >= len(action_frames):
        raise ValueError(
            f"--start_frame {start} is outside action.json range [0, {len(action_frames) - 1}]"
        )

    capture = cv2.VideoCapture(str(sample_dir / "video.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {sample_dir / 'video.mp4'}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    available = min(video_frames, len(action_frames)) - start
    frame_count = available if args.max_frames == 0 else min(available, args.max_frames)
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"No aligned video/action frames available for {sample_dir.name}")

    end = start + frame_count
    renderer = ActionOverlayRenderer(width=width, height=height)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mind_data-{sample_id}_f{start}-{end - 1}.mp4"
    output_fps = source_fps if args.fps <= 0 else float(args.fps)
    # Match the repository's other video-saving paths: H.264/yuv420p.
    # Avoid imageio's default macro-block resizing from 1080 to 1088.
    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=output_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
    except Exception:
        capture.release()
        raise

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    mark_time = int(metadata.get("mark_time", -1))
    print(
        f"[{sample_dir.name}] video={video_frames} action={len(action_frames)} "
        f"range=[{start}, {end}) size={width}x{height} fps={output_fps:g}"
    )
    try:
        for local_index in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                print(f"[WARN] {sample_dir.name}: video ended at frame {start + local_index}")
                break
            source_index = start + local_index
            action_frame = action_frames[source_index]
            pressed_keys, yaw, pitch, action_name = decode_mind_action(action_frame)
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            if not args.no_overlay:
                panel = renderer.render_panel(
                    pressed_keys=pressed_keys,
                    yaw=yaw,
                    pitch=pitch,
                    corner=args.corner,
                )
                image.alpha_composite(panel)

            if not args.no_text:
                phase = "memory/context" if mark_time >= 0 and source_index < mark_time else "prediction"
                _draw_information(
                    image,
                    sample_name=sample_dir.name,
                    source_frame=source_index,
                    action_frame=action_frame,
                    action_name=action_name,
                    phase=phase,
                )
            # imageio expects uint8 RGB frames.
            writer.append_data(np.asarray(image.convert("RGB")))
    finally:
        capture.release()
        writer.close()

    print(f"  -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw MIND video.mp4 with ws/ad/ud/lr controls."
    )
    parser.add_argument(
        "--mind_path", type=Path,
        default=Path("/nfs/yixinyang/code/LongLive/data/MIND/3rd_data/train"),
        help="Directory containing data-*/video.mp4 and data-*/action.json.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./mind_vis"))
    parser.add_argument(
        "--sample_id", type=int, default=None,
        help="Visualize data-<ID>; random samples are chosen when omitted.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of samples (consecutive from --sample_id, otherwise random).",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--max_frames", type=int, default=157,
        help="Maximum output frames per clip (default: 157; 0 means all).",
    )
    parser.add_argument(
        "--fps", type=float, default=0.0,
        help="Output FPS (default: 0, preserve source FPS).",
    )
    parser.add_argument(
        "--corner", default="bottom-left", choices=ActionOverlayRenderer.CORNER_CHOICES,
        help="Corner for the ws/ad/ud/lr action overlay.",
    )
    parser.add_argument("--no_overlay", action="store_true",
                        help="Disable the WASD/joystick action panel.")
    parser.add_argument("--no_text", action="store_true",
                        help="Disable frame/action text.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mind_path = args.mind_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative")

    samples = list_mind_samples(args.mind_path)
    if not samples:
        raise RuntimeError(f"No valid data-* samples found under {args.mind_path}")
    selected = _select_samples(samples, args.sample_id, args.num_samples, args.seed)
    print(f"Found {len(samples)} samples; visualizing {[sid for sid, _ in selected]}")

    outputs = [
        visualize_sample(sample_id, sample_dir, args.output_dir, args)
        for sample_id, sample_dir in selected
    ]
    print(f"Done. Wrote {len(outputs)} video(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
