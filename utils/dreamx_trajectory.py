# SPDX-License-Identifier: MIT
# DreamX-World style trajectory parser ported into LongLive.
#
# Mirrors:
#   DreamX-World/utils/inference_utils.py
#   DreamX-World/utils/pose_utils.py
#
# Public API:
#   - action_to_viewmats_Ks(
#         action_seq: List[str],
#         action_speed_list: List[float],
#         duration: int,
#         target_length: int,
#         h: int,
#         w: int,
#     ) -> (viewmats: torch.Tensor[T_lat,4,4], Ks: torch.Tensor[T_lat,3,3])
#
#   - action_to_raw_c2w(
#         action_seq: List[str],
#         action_speed_list: List[float],
#         duration: int,
#         target_length: int,
#     ) -> np.ndarray[target_length, 4, 4]
#     Raw (pre-VAE-stride) per-frame camera-to-world matrices, anchored so
#     that frame 0 is identity. Used to drive the action overlay at the
#     video's native frame rate.
#
#   - parse_trajectory_string(
#         traj: str,                   # e.g. "w-4,a-4,w-4,d-4"
#     ) -> (action_seq: List[str], action_speed_list: List[float])
#     Convenience parser so the same line-aligned ``trajectory_path`` text
#     file can host worldplaygen / sana_dsl / dreamx_action_dsl strings.
#
# Action DSL (DreamX-World convention):
#   "w" forward, "s" backward, "a" left, "d" right
#   "j" left_rot, "l" right_rot, "i" up_rot, "k" down_rot
# Composite keys ("wl") combine motions in one segment.
#
# Units (matching DreamX defaults):
#   TRANSLATION_BASE_UNIT = 1.0   (meters per speed unit)
#   ROTATION_BASE_UNIT    = 10.0  (degrees per speed unit)
#
# Output convention:
#   viewmats : (T_lat, 4, 4) world-to-camera (w2c) matrices, OpenCV-like
#              (camera +Z is the viewing direction).
#   Ks       : (T_lat, 3, 3) intrinsics in *normalized* form
#              (cx=cy=0, fx/fy normalized by 2*cx, 2*cy of a default 1080p
#               camera) — this matches DreamX's GetPoseEmbedsFromPosesPrope
#               output and the shape PRoPE expects.
#
# T_lat is the number of *latent* camera frames after VAE temporal
# downsampling (1 + (raw_frames - 1) // 4), aligning with Wan2.2-TI2V-5B.

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Action DSL
# ---------------------------------------------------------------------------

ACTION_DICT = {
    "w": "forward", "s": "backward", "a": "left", "d": "right",
    "j": "left_rot", "l": "right_rot", "i": "up_rot", "k": "down_rot",
}

TRANSLATION_BASE_UNIT = 1.0   # meters per speed unit
ROTATION_BASE_UNIT = 10.0     # degrees per speed unit


def _compute_translation_step(motion_type, current_pose, translation_value, duration):
    """World-space per-frame translation for a single primitive."""
    if motion_type in ("forward", "backward"):
        yaw_rad = np.radians(current_pose["rotation"][1])
        pitch_rad = np.radians(current_pose["rotation"][0])
        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
        ])
        direction = 1 if motion_type == "forward" else -1
        return forward_vec * translation_value * direction / duration
    if motion_type in ("left", "right"):
        yaw_rad = np.radians(current_pose["rotation"][1])
        right_vec = np.array([math.cos(yaw_rad), 0.0, math.sin(yaw_rad)])
        direction = -1 if motion_type == "left" else 1
        return right_vec * translation_value * direction / duration
    return np.zeros(3)


def _compute_rotation_step(motion_type, rotation_value, duration):
    """Per-frame [pitch, yaw, roll] step in degrees."""
    if not motion_type.endswith("rot"):
        return np.zeros(3)
    axis = motion_type.split("_")[0]
    rot = np.zeros(3)
    if axis == "left":
        rot[1] = rotation_value
    elif axis == "right":
        rot[1] = -rotation_value
    elif axis == "up":
        rot[0] = -rotation_value
    elif axis == "down":
        rot[0] = rotation_value
    return rot / duration


def _generate_composite_motion_segment(current_pose, motion_types,
                                       translation_value, rotation_value,
                                       duration):
    if isinstance(motion_types, str):
        motion_types = [motion_types]
    positions, rotations = [], []
    t_step = np.zeros(3)
    r_step = np.zeros(3)
    for mt in motion_types:
        t_step += _compute_translation_step(mt, current_pose, translation_value, duration)
        r_step += _compute_rotation_step(mt, rotation_value, duration)
    for i in range(1, duration + 1):
        positions.append((current_pose["position"] + t_step * i).copy())
        rotations.append((current_pose["rotation"] + r_step * i).copy())
    current_pose["position"] = positions[-1].copy()
    current_pose["rotation"] = rotations[-1].copy()
    return positions, rotations, current_pose


def _euler_to_quaternion(angles_deg):
    """[pitch, yaw, roll] (deg, ZYX intrinsic) -> [qw, qx, qy, qz]."""
    pitch, yaw, roll = np.radians(angles_deg)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * sp * cr + sy * cp * sr
    qy = sy * cp * cr - cy * sp * sr
    qz = cy * cp * sr - sy * sp * cr
    return [qw, qx, qy, qz]


def _quat_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ])


def _action_seq_to_w2c(
    action_seq: Sequence[str],
    action_speed_list: Sequence[float],
    duration: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk the action DSL, return (R_w2c, t_w2c) arrays of shape (N,3,3) and (N,3).

    N == 1 + len(action_seq) * duration (frame 0 is the identity start pose).
    """
    assert len(action_seq) == len(action_speed_list), (
        f"action_seq has {len(action_seq)} entries but action_speed_list has "
        f"{len(action_speed_list)}.")

    current = {
        "position": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0]),  # [pitch, yaw, roll]
    }
    all_positions: List[np.ndarray] = []
    all_rotations: List[np.ndarray] = []
    for action, speed in zip(action_seq, action_speed_list):
        keys = list(action) if isinstance(action, str) else list(action)
        motion_types = [ACTION_DICT[k] for k in keys]
        positions, rotations, current = _generate_composite_motion_segment(
            current, motion_types,
            translation_value=speed * TRANSLATION_BASE_UNIT,
            rotation_value=speed * ROTATION_BASE_UNIT,
            duration=duration,
        )
        all_positions.extend(positions); all_rotations.extend(rotations)

    # Frame 0: identity pose at the origin.
    R_list = [np.eye(3)]
    t_list = [np.zeros(3)]
    for pos, rot in zip(all_positions, all_rotations):
        q = _euler_to_quaternion(rot)
        R = _quat_to_R(q)
        t = -R @ pos                # world -> camera translation
        R_list.append(R); t_list.append(t)

    return np.stack(R_list), np.stack(t_list)


# ---------------------------------------------------------------------------
# Pose interpolation (raw frames -> latent frames)
# ---------------------------------------------------------------------------

def _interpolate_w2c(
    R_src: np.ndarray,            # (N, 3, 3)
    t_src: np.ndarray,             # (N, 3)
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """SLERP rotations + linear-interpolate translations onto ``tgt_indices``.

    Verbatim port of DreamX-World/utils/pose_utils.py::interpolate_camera_poses,
    minus the Camera-object packaging (we keep R/t as numpy arrays).
    """
    src_indices = np.asarray(src_indices, dtype=np.float64)
    tgt_indices = np.asarray(tgt_indices, dtype=np.float64)

    # Left-handed-coord guard (matches DreamX).
    dets = np.linalg.det(R_src)
    flip = dets.size > 0 and np.median(dets) < 0.0
    if flip:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(R_src.dtype)
        R_src = R_src @ flip_mat

    # Translation: linear.
    f_trans = interp1d(
        src_indices, t_src, axis=0, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    t_tgt = f_trans(tgt_indices)

    # Rotation: SLERP.
    src_rot = Rotation.from_matrix(R_src)
    quats = src_rot.as_quat().copy()  # (N, 4) xyzw
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_rot = Rotation.from_quat(quats)
    slerp = Slerp(src_indices, src_rot)
    R_tgt = slerp(tgt_indices).as_matrix()

    if flip:
        R_tgt = R_tgt @ flip_mat

    return R_tgt, t_tgt


def _build_relative_w2c(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Re-anchor the trajectory so that frame 0's c2w is identity, return w2c.

    Mirrors DreamX's ``get_relative_pose`` followed by ``_invert_SE3``.
    Specifically: c2w_rel[0] = I, c2w_rel[i] = (target_c2w @ w2c[0]) @ c2w[i].
    Since target_c2w = I and w2c[0] = I (we always start at identity), this
    simplifies to c2w_rel[i] = c2w[i]. We keep the explicit form so that the
    code is robust to non-identity start poses.
    """
    n = R_w2c.shape[0]
    w2c = np.zeros((n, 4, 4), dtype=np.float64)
    w2c[:, :3, :3] = R_w2c
    w2c[:, :3, 3] = t_w2c
    w2c[:, 3, 3] = 1.0

    c2w = np.linalg.inv(w2c)
    target_c2w = np.eye(4)
    abs2rel = target_c2w @ w2c[0]
    c2w_rel = np.zeros_like(c2w)
    c2w_rel[0] = target_c2w
    for i in range(1, n):
        c2w_rel[i] = abs2rel @ c2w[i]

    w2c_rel = np.linalg.inv(c2w_rel)
    return w2c_rel.astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def action_to_viewmats_Ks(
    action_seq: Sequence[str],
    action_speed_list: Sequence[float],
    duration: int,
    target_length: int,
    h: int = 704,
    w: int = 1280,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a DreamX-style ``(action_seq, action_speed_list)`` to PRoPE
    ``(viewmats, Ks)`` ready to be passed into LongLive's WanModel.

    Args:
        action_seq:        e.g. ``["w","a","w","d"]`` or composite ``["wl"]``.
        action_speed_list: same length as ``action_seq``.
        duration:          per-segment frame count (DreamX uses 33 by default
                           when there are 4 segments → 1 + 4*33 == 133 raw
                           frames; for video_length=121 use ``duration=30``,
                           or pre-trim externally).
        target_length:     number of *raw* frames to consume from the head of
                           the trajectory before VAE-stride downsampling.
                           Must satisfy ``target_length <= 1 + len(action_seq)*duration``.
        h, w:              image height / width — used only when overriding
                           default normalized intrinsics; current build uses
                           the same DreamX defaults regardless of resolution.
        dtype, device:     output tensor dtype/device.

    Returns:
        viewmats (T_lat, 4, 4) world-to-camera matrices.
        Ks       (T_lat, 3, 3) normalized intrinsics (cx=cy=0).

    ``T_lat = 1 + (target_length - 1) // 4`` matches Wan2.2 VAE temporal
    stride and the (F, H, W) grid PRoPE expects.
    """
    R_src, t_src = _action_seq_to_w2c(action_seq, action_speed_list, duration)
    n_raw = R_src.shape[0]
    assert target_length <= n_raw, (
        f"target_length={target_length} exceeds raw trajectory length {n_raw}; "
        f"increase ``duration`` or shorten ``target_length``.")

    # Take the first ``target_length`` raw frames (DreamX uses start_index=0).
    R_src = R_src[:target_length]
    t_src = t_src[:target_length]

    # Latent-frame alignment: 1 + (N-1)//4.
    n_frames = target_length
    latent_frame_count = 1 + (n_frames - 1) // 4
    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, latent_frame_count)
    R_tgt, t_tgt = _interpolate_w2c(R_src, t_src, src_indices, tgt_indices)

    # Re-anchor to identity start (no-op when the trajectory already starts
    # at identity, but keeps the function correct for arbitrary inputs).
    viewmats_np = _build_relative_w2c(R_tgt, t_tgt)  # (T_lat, 4, 4) w2c
    viewmats = torch.as_tensor(viewmats_np, dtype=dtype, device=device)

    # DreamX intrinsic defaults (1080p reference image, normalized to [-1,1]):
    #   fx_norm = 969.6969.../1920 = 0.50505...
    #   fy_norm = 969.6969.../1080 = 0.89786...
    default_intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]
    fx_norm = default_intrinsic[0][0] / (default_intrinsic[0][2] * 2)
    fy_norm = default_intrinsic[1][1] / (default_intrinsic[1][2] * 2)
    T_lat = viewmats.shape[0]
    Ks = torch.zeros((T_lat, 3, 3), dtype=dtype, device=device)
    Ks[:, 0, 0] = fx_norm
    Ks[:, 1, 1] = fy_norm
    Ks[:, 2, 2] = 1.0
    return viewmats, Ks


# ---------------------------------------------------------------------------
# Raw (pre-VAE-stride) c2w trajectory — used to drive the action overlay.
# ---------------------------------------------------------------------------

def action_to_raw_c2w(
    action_seq: Sequence[str],
    action_speed_list: Sequence[float],
    duration: int,
    target_length: int,
) -> np.ndarray:
    """Return the *raw* per-frame camera-to-world trajectory.

    Mirrors the head of :func:`action_to_viewmats_Ks`, but stops *before* the
    VAE-stride latent-frame interpolation. The returned poses are anchored so
    that frame 0 is the identity (matching the convention used by the PRoPE
    branch and ``utils.action_overlay.apply_overlay``).

    Args:
        action_seq, action_speed_list, duration: same as
            :func:`action_to_viewmats_Ks`.
        target_length: number of raw frames to return. Must satisfy
            ``target_length <= 1 + len(action_seq) * duration``.

    Returns:
        ``(target_length, 4, 4)`` ``np.float32`` c2w matrices in OpenCV
        convention (camera +Z is the viewing direction).
    """
    R_src, t_src = _action_seq_to_w2c(action_seq, action_speed_list, duration)
    n_raw = R_src.shape[0]
    assert target_length <= n_raw, (
        f"target_length={target_length} exceeds raw trajectory length {n_raw}; "
        f"increase ``duration`` or shorten ``target_length``.")

    R_src = R_src[:target_length]
    t_src = t_src[:target_length]

    # Re-anchor so frame 0 is identity, mirroring _build_relative_w2c but
    # skipping the SLERP/LERP step (we want the raw, native-frame-rate poses
    # for the overlay so fast key presses don't get under-sampled).
    n = R_src.shape[0]
    w2c = np.zeros((n, 4, 4), dtype=np.float64)
    w2c[:, :3, :3] = R_src
    w2c[:, :3, 3] = t_src
    w2c[:, 3, 3] = 1.0

    c2w = np.linalg.inv(w2c)
    abs2rel = w2c[0]                   # target_c2w (= I) @ w2c[0]
    c2w_rel = np.zeros_like(c2w)
    c2w_rel[0] = np.eye(4)
    for i in range(1, n):
        c2w_rel[i] = abs2rel @ c2w[i]

    return c2w_rel.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory string parser (line-aligned text-file format).
# ---------------------------------------------------------------------------

def parse_trajectory_string(traj: str) -> Tuple[List[str], List[float]]:
    """Parse a single line of the DreamX action DSL into ``(action_seq, action_speed_list)``.

    Grammar mirrors WorldPlayGen's ``"<token>-<speed>"`` segment style so a
    single ``inference.trajectory_path`` text file can host any of the three
    DSLs supported by this script:

        ``"w-4,a-4,w-4,d-4"`` -> (["w","a","w","d"], [4.0, 4.0, 4.0, 4.0])

    The token may be a single primitive key from ``ACTION_DICT`` or a
    composite (e.g. ``"wl"`` for forward + right_rot), matching the upstream
    DreamX convention. Whitespace inside a segment is tolerated; empty
    segments are skipped.

    Args:
        traj: trajectory string for a single clip.

    Returns:
        ``(action_seq, action_speed_list)`` ready to feed into
        :func:`action_to_viewmats_Ks` / :func:`action_to_raw_c2w`.
    """
    action_seq: List[str] = []
    action_speed_list: List[float] = []
    for raw_seg in traj.split(","):
        seg = raw_seg.strip()
        if not seg:
            continue
        if "-" not in seg:
            raise ValueError(
                f"dreamx_action_dsl segment must be 'token-speed', got {seg!r} "
                f"(in trajectory {traj!r})")
        token, _, speed_str = seg.partition("-")
        token = token.strip().lower()
        speed_str = speed_str.strip()
        if not token:
            raise ValueError(
                f"empty action token in segment {seg!r} of trajectory {traj!r}")
        for ch in token:
            if ch not in ACTION_DICT:
                raise ValueError(
                    f"unknown action key {ch!r} in token {token!r}; valid keys "
                    f"are {sorted(ACTION_DICT)}")
        try:
            speed = float(speed_str)
        except ValueError as exc:
            raise ValueError(
                f"invalid speed {speed_str!r} in segment {seg!r}") from exc
        action_seq.append(token)
        action_speed_list.append(speed)
    if not action_seq:
        raise ValueError(f"trajectory string is empty: {traj!r}")
    return action_seq, action_speed_list
