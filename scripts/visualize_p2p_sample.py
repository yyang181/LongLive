#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visualize Open Pixel2Play (P2P) videos with keyboard/mouse actions.

The full P2P dataset stores 200 recordings per ``batch_*.tar.gz``.  Every
recording contains ``video.mp4``, ``192x192.mp4`` and a binary
``annotation.proto``.  This script reads the archives directly and extracts
only one temporary recording from each selected batch; the dataset does not
need to be unpacked first.

P2P actions are more general than MIND's ``ws/ad/ud/lr`` controls:

* held keyboard keys include W/A/S/D (plus jump, sprint, weapon keys, etc.);
* ``mouse_delta_px.x`` is analogous to MIND ``lr``;
* ``mouse_delta_px.y`` is analogous to MIND ``ud``;
* mouse buttons and scroll are also available.

The official P2P behavior-cloning loader prioritizes ``system_action`` when
known and otherwise uses ``user_action``.  The visualizer follows that rule.

Example (from the LongLive repository root)::

    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_p2p_sample.py \
        --mind_path /nfs/hongfenglai/p2p-full-data \
        --output_dir ./lmdb_vis/p2pfull \
        --min_frames 593 \
        --num_samples 10

``--mind_path`` is accepted as a compatibility alias for ``--p2p_path``.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise ImportError(
        "visualize_p2p_sample.py requires protobuf (pip install protobuf)."
    ) from exc


# Make the repository root importable regardless of the current directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.action_overlay import ActionOverlayRenderer  # noqa: E402


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


def _build_video_annotation_class():
    """Build a minimal wire-compatible subset of P2P's protobuf schema.

    Protobuf safely preserves/skips unknown fields, so the large task state
    and text-embedding messages do not need to be vendored merely to inspect
    keyboard, mouse, environment and instruction fields.
    """
    file_desc = descriptor_pb2.FileDescriptorProto(
        name="p2p_visualizer.proto", package="p2p_visualizer", syntax="proto3"
    )
    field_proto = descriptor_pb2.FieldDescriptorProto

    def add_message(name: str):
        message = file_desc.message_type.add()
        message.name = name
        return message

    def add_field(
        message, name: str, number: int, field_type: int, *,
        repeated: bool = False, type_name: str = "",
    ) -> None:
        field = message.field.add()
        field.name = name
        field.number = number
        field.type = field_type
        field.label = (
            field_proto.LABEL_REPEATED if repeated else field_proto.LABEL_OPTIONAL
        )
        if type_name:
            field.type_name = type_name

    vec2_int = add_message("Vec2Int")
    add_field(vec2_int, "x", 1, field_proto.TYPE_INT32)
    add_field(vec2_int, "y", 2, field_proto.TYPE_INT32)

    environment = add_message("VideoAnnotationEnv")
    add_field(environment, "env", 1, field_proto.TYPE_STRING)
    add_field(environment, "env_subtype", 2, field_proto.TYPE_STRING)
    add_field(environment, "env_version", 3, field_proto.TYPE_STRING)

    metadata = add_message("VideoAnnotationMetadata")
    add_field(metadata, "id", 1, field_proto.TYPE_STRING)
    add_field(metadata, "frames_per_second", 5, field_proto.TYPE_FLOAT)
    add_field(
        metadata, "env", 6, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.VideoAnnotationEnv",
    )

    keyboard = add_message("KeyboardAction")
    add_field(keyboard, "keys", 1, field_proto.TYPE_STRING, repeated=True)

    mouse = add_message("MouseAction")
    add_field(
        mouse, "mouse_absolute_px", 1, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.Vec2Int",
    )
    add_field(
        mouse, "mouse_delta_px", 3, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.Vec2Int",
    )
    add_field(
        mouse, "scroll_delta_px", 4, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.Vec2Int",
    )
    add_field(mouse, "buttons_down", 9, field_proto.TYPE_STRING, repeated=True)

    action = add_message("LowLevelAction")
    add_field(
        action, "keyboard", 2, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.KeyboardAction",
    )
    add_field(action, "is_known", 3, field_proto.TYPE_BOOL)
    add_field(
        action, "mouse", 4, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.MouseAction",
    )

    annotator = add_message("FrameTextAnnotator")
    add_field(annotator, "provider", 1, field_proto.TYPE_STRING)
    add_field(annotator, "version", 2, field_proto.TYPE_STRING)

    text_annotation = add_message("FrameTextAnnotation")
    add_field(text_annotation, "instruction", 1, field_proto.TYPE_STRING)
    add_field(
        text_annotation, "frame_text_annotator", 2, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.FrameTextAnnotator",
    )
    add_field(text_annotation, "duration", 3, field_proto.TYPE_FLOAT)

    frame = add_message("FrameAnnotation")
    add_field(
        frame, "user_action", 1, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.LowLevelAction",
    )
    add_field(
        frame, "system_action", 6, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.LowLevelAction",
    )
    add_field(
        frame, "frame_text_annotation", 7, field_proto.TYPE_MESSAGE,
        repeated=True, type_name=".p2p_visualizer.FrameTextAnnotation",
    )
    add_field(frame, "frame_time", 8, field_proto.TYPE_UINT64)

    video = add_message("VideoAnnotation")
    add_field(
        video, "metadata", 2, field_proto.TYPE_MESSAGE,
        type_name=".p2p_visualizer.VideoAnnotationMetadata",
    )
    add_field(
        video, "frame_annotations", 3, field_proto.TYPE_MESSAGE,
        repeated=True, type_name=".p2p_visualizer.FrameAnnotation",
    )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    descriptor = pool.FindMessageTypeByName("p2p_visualizer.VideoAnnotation")
    # protobuf <=4 exposes GetPrototype; newer versions expose GetMessageClass.
    factory = message_factory.MessageFactory(pool)
    if hasattr(factory, "GetPrototype"):
        return factory.GetPrototype(descriptor)
    return message_factory.GetMessageClass(descriptor)


VideoAnnotation = _build_video_annotation_class()


def list_archives(p2p_path: Path) -> list[Path]:
    dataset_dir = p2p_path / "dataset" if (p2p_path / "dataset").is_dir() else p2p_path
    archives = sorted(dataset_dir.glob("batch_*.tar.gz"))
    if not archives:
        raise RuntimeError(f"No batch_*.tar.gz archives found under {dataset_dir}")
    return archives


def parse_annotation(data: bytes, source: str):
    annotation = VideoAnnotation()
    try:
        annotation.ParseFromString(data)
    except Exception as exc:
        raise ValueError(f"Failed to parse {source} as P2P annotation.proto") from exc
    if not annotation.frame_annotations:
        raise ValueError(f"No frame annotations in {source}")
    return annotation


def extract_first_sample(
    archive: Path, temp_dir: Path, video_name: str, min_frames: int = 0,
) -> tuple[str, Path, object]:
    """Stream the first recording with more than ``min_frames`` frames."""
    selected_id = None
    annotation = None
    # Stream mode avoids indexing/decompressing the complete, potentially huge archive.
    with tarfile.open(archive, mode="r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            sample_id, filename = parts[-2], parts[-1]
            if selected_id is None and filename == "annotation.proto":
                fileobj = handle.extractfile(member)
                if fileobj is None:
                    continue
                candidate = parse_annotation(
                    fileobj.read(), f"{archive.name}:{member.name}"
                )
                if len(candidate.frame_annotations) <= min_frames:
                    continue
                selected_id = sample_id
                annotation = candidate
                continue
            if selected_id == sample_id and filename == video_name:
                output_path = temp_dir / video_name
                fileobj = handle.extractfile(member)
                if fileobj is None:
                    raise RuntimeError(f"Could not read {archive.name}:{member.name}")
                with output_path.open("wb") as output:
                    shutil.copyfileobj(fileobj, output, length=8 * 1024 * 1024)
                return selected_id, output_path, annotation
    if selected_id is None:
        raise RuntimeError(
            f"No recording with more than {min_frames} frames found in {archive}"
        )
    raise RuntimeError(f"No {video_name} found for {selected_id} in {archive}")


def select_action(frame_annotation):
    """Match the official P2P priority: known system action, then user action."""
    if frame_annotation.system_action.is_known:
        return frame_annotation.system_action, "system"
    if frame_annotation.user_action.is_known:
        return frame_annotation.user_action, "user"
    return frame_annotation.user_action, "unknown"


def action_inputs(action, mouse_scale: float) -> tuple[list[str], float, float]:
    keys = [key.upper() for key in action.keyboard.keys]
    pressed_keys = [key for key in keys if key in {"W", "A", "S", "D"}]
    dx = int(action.mouse.mouse_delta_px.x)
    dy = int(action.mouse.mouse_delta_px.y)
    yaw = float(np.clip(dx / mouse_scale, -1.0, 1.0))
    pitch = float(np.clip(dy / mouse_scale, -1.0, 1.0))
    return pressed_keys, yaw, pitch


def instruction_timeline(annotation, end_frame: int) -> list[str]:
    """Expand sparse P2P instructions over their declared duration."""
    fps = float(annotation.metadata.frames_per_second) or 20.0
    result = [""] * max(0, end_frame)
    active = ""
    remaining = 0
    limit = min(len(annotation.frame_annotations), end_frame)
    for index in range(limit):
        frame = annotation.frame_annotations[index]
        candidates = [item for item in frame.frame_text_annotation if item.instruction]
        if candidates:
            item = candidates[0]
            active = item.instruction.strip()
            remaining = max(1, int(round(float(item.duration) * fps)))
        if remaining > 0:
            result[index] = active
            remaining -= 1
        else:
            active = ""
    return result


def _draw_information(
    image: Image.Image, *, sample_id: str, archive_name: str,
    video_frame: int, action_frame: int, action, action_source: str,
    instruction: str,
) -> None:
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(max(12, int(height * 0.021)))
    line_height = max(18, int(height * 0.029))
    keys = "+".join(str(key).upper() for key in action.keyboard.keys) or "none"
    buttons = "+".join(str(button) for button in action.mouse.buttons_down) or "none"
    dx = int(action.mouse.mouse_delta_px.x)
    dy = int(action.mouse.mouse_delta_px.y)
    lines = [
        f"{archive_name} | {sample_id}",
        f"video={video_frame} annotation={action_frame} source={action_source}",
        f"Keys: {keys} | Mouse delta: ({dx}, {dy}) | Buttons: {buttons}",
    ]
    if instruction:
        compact = " ".join(instruction.split())
        lines.append(f"Instruction: {compact[:140]}")
    panel_width = min(width - 20, max(420, int(width * 0.66)))
    panel_height = line_height * len(lines) + 16
    draw.rounded_rectangle(
        (8, 8, 8 + panel_width, 8 + panel_height), radius=9,
        fill=(0, 0, 0, 150), outline=(255, 255, 255, 60), width=1,
    )
    y = 15
    for line in lines:
        draw.text((16, y), line, font=font, fill=(255, 255, 255, 238))
        y += line_height


def visualize_sample(
    archive: Path, sample_id: str, video_path: Path, annotation,
    output_dir: Path, args,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open temporary video {video_path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
    action_frames = annotation.frame_annotations
    if video_frames <= args.min_frames:
        capture.release()
        raise ValueError(
            f"Video {sample_id} has {video_frames} frames; --min_frames "
            f"requires more than {args.min_frames}"
        )
    start = args.start_frame
    first_action = start + args.action_offset
    if start < 0 or start >= video_frames:
        capture.release()
        raise ValueError(f"--start_frame {start} is outside video range [0, {video_frames - 1}]")
    if first_action < 0 or first_action >= len(action_frames):
        capture.release()
        raise ValueError(
            f"First annotation {first_action} is outside range [0, {len(action_frames) - 1}]"
        )
    available = min(video_frames - start, len(action_frames) - first_action)
    frame_count = available if args.max_frames == 0 else min(available, args.max_frames)
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"No aligned frames for {sample_id}")

    end = start + frame_count
    instructions = instruction_timeline(annotation, first_action + frame_count)
    renderer = ActionOverlayRenderer(width=width, height=height)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"p2p_{archive.name.removesuffix('.tar.gz')}_{sample_id}_f{start}-{end - 1}.mp4"
    )
    output_fps = source_fps if args.fps <= 0 else float(args.fps)
    writer = imageio.get_writer(
        str(output_path), fps=output_fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=None,
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    metadata = annotation.metadata
    print(
        f"[{archive.name}:{sample_id}] env={metadata.env.env}/"
        f"{metadata.env.env_subtype} video={video_frames} annotation={len(action_frames)} "
        f"range=[{start}, {end}) size={width}x{height} fps={output_fps:g}"
    )
    try:
        for local_index in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                print(f"[WARN] {sample_id}: video ended at frame {start + local_index}")
                break
            video_index = start + local_index
            action_index = first_action + local_index
            action, source = select_action(action_frames[action_index])
            pressed_keys, yaw, pitch = action_inputs(action, args.mouse_scale)
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
            if not args.no_overlay:
                panel = renderer.render_panel(
                    pressed_keys=pressed_keys, yaw=yaw, pitch=pitch, corner=args.corner
                )
                image.alpha_composite(panel)
            if not args.no_text:
                _draw_information(
                    image,
                    sample_id=sample_id,
                    archive_name=archive.name,
                    video_frame=video_index,
                    action_frame=action_index,
                    action=action,
                    action_source=source,
                    instruction=instructions[action_index],
                )
            writer.append_data(np.asarray(image.convert("RGB")))
    finally:
        capture.release()
        writer.close()
    print(f"  -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize P2P batch archives with keyboard and mouse controls."
    )
    parser.add_argument(
        "--p2p_path", "--mind_path", dest="p2p_path", type=Path,
        default=Path("/nfs/hongfenglai/p2p-full-data"),
        help="P2P root containing data_metadata.parquet and dataset/batch_*.tar.gz.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./p2p_vis"))
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--min_frames", type=int, default=0,
        help=(
            "Exclusive minimum source-video length. For example, 593 keeps "
            "only videos with more than 593 frames (default: 0)."
        ),
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--max_frames", type=int, default=157,
        help="Maximum output frames per recording (default: 157; 0 means all).",
    )
    parser.add_argument(
        "--video_name", choices=("video.mp4", "192x192.mp4"), default="video.mp4",
        help="Video member to visualize (default: original-resolution video.mp4).",
    )
    parser.add_argument(
        "--action_offset", type=int, default=0,
        help=(
            "Annotation index minus displayed video index. Use 0 for direct overlay; "
            "the official behavior-cloning input-to-next-action alignment uses 1."
        ),
    )
    parser.add_argument(
        "--mouse_scale", type=float, default=20.0,
        help="Mouse pixels mapped to full joystick deflection (default: 20).",
    )
    parser.add_argument("--fps", type=float, default=0.0,
                        help="Output FPS (default: preserve source FPS).")
    parser.add_argument(
        "--corner", default="bottom-left", choices=ActionOverlayRenderer.CORNER_CHOICES
    )
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.p2p_path = args.p2p_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative")
    if args.min_frames < 0:
        raise ValueError("--min_frames must be non-negative")
    if args.mouse_scale <= 0:
        raise ValueError("--mouse_scale must be positive")

    archives = list_archives(args.p2p_path)
    rng = random.Random(args.seed)
    candidates = archives.copy()
    rng.shuffle(candidates)
    print(
        f"Found {len(archives)} archives; selecting up to {args.num_samples} "
        f"recordings (seed={args.seed}, video={args.video_name})"
    )
    outputs = []
    failures = 0
    for archive in candidates:
        if len(outputs) >= args.num_samples:
            break
        try:
            with tempfile.TemporaryDirectory(prefix="p2p_visualize_") as tmp:
                sample_id, video_path, annotation = extract_first_sample(
                    archive, Path(tmp), args.video_name, args.min_frames
                )
                outputs.append(
                    visualize_sample(
                        archive, sample_id, video_path, annotation, args.output_dir, args
                    )
                )
        except (OSError, tarfile.TarError, ValueError, RuntimeError) as exc:
            failures += 1
            print(f"[WARN] Skipping {archive.name}: {exc}")
    if len(outputs) < args.num_samples:
        raise RuntimeError(
            f"Only produced {len(outputs)}/{args.num_samples} videos; "
            f"{failures} archive(s) failed."
        )
    print(f"Done. Wrote {len(outputs)} video(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
