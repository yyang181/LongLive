# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0
"""Sana-WM camera-control trajectory generator (vendored).

A faithful port of ``inference_video_scripts/wm/camera_control.py`` plus the
``action_string_to_c2w`` rollout from sana's ``inference_sana_wm.py``. Used as
an *alternative* trajectory source for LongLive inference, so we can reuse the
sana DSL (with combo keys, ``none``, smooth velocity, FPS-camera-style yaw vs.
self-frame pitch, ground-plane-locked translation, pitch clamp). After
generating ``(N+1, 4, 4)`` camera-to-world matrices we convert them to the
``(T, 7)`` w2c quaternion+translation format that
``utils.camera_dataset.build_viewmats_and_Ks`` consumes, so downstream
PRoPE conditioning matches LongLive's existing path bit-for-bit.

Coordinate convention: OpenCV (+X right, +Y down, +Z forward); poses start
from identity.

Control scheme (unified with the sana interactive demo):
    w / s            forward / back translation (along heading)
    a / d            yaw left / right rotation
    i / k            pitch up / down rotation
    j / l            strafe left / right translation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Defaults — kept identical to sana so DSL strings are portable.
# ---------------------------------------------------------------------------
FPS = 16

DEFAULT_TRANSLATION_SPEED = 0.025
DEFAULT_ROTATION_SPEED_DEG = 0.6
DEFAULT_PITCH_LIMIT_DEG = 60.0

# Exponential-smoothing time constants (seconds): faster ramp on press, slower
# coast on release so motion eases to a stop instead of snapping.
TAU_PRESS = 0.45
TAU_COAST = 1.0

# Canonical control tokens.
CONTROL_TOKENS = frozenset(
    {"forward", "back", "strafe_left", "strafe_right",
     "yaw_left", "yaw_right", "pitch_up", "pitch_down"}
)

# DSL letters → canonical controls (sana's mapping; must NOT be edited).
DSL_KEY_TO_CONTROL: dict = {
    "w": "forward",
    "s": "back",
    "a": "yaw_left",
    "d": "yaw_right",
    "i": "pitch_up",
    "k": "pitch_down",
    "j": "strafe_left",
    "l": "strafe_right",
}
ALLOWED_ACTION_KEYS = frozenset(DSL_KEY_TO_CONTROL.keys())


# ---------------------------------------------------------------------------
# Rotation primitives.
# ---------------------------------------------------------------------------
def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[ c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,  c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Velocity model + integrator (shared with the sana interactive demo).
# ---------------------------------------------------------------------------
@dataclass
class VelocityState:
    """Per-frame velocity: translation (tx forward, sx strafe-right) + rotation
    (yaw right+, pitch up+), in the same units as the motion magnitudes."""

    tx: float = 0.0
    sx: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def snap_to(self, target: "VelocityState") -> None:
        self.tx, self.sx, self.yaw, self.pitch = target.tx, target.sx, target.yaw, target.pitch

    def step_toward(self, target: "VelocityState", dt: float) -> None:
        for attr in ("tx", "sx", "yaw", "pitch"):
            cur = getattr(self, attr)
            tgt = getattr(target, attr)
            tau = TAU_PRESS if abs(tgt) > 1e-12 else TAU_COAST
            alpha = 1.0 - math.exp(-dt / tau)
            setattr(self, attr, cur + alpha * (tgt - cur))


def controls_to_target_velocity(
    controls: set,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_rad: float = None,
) -> VelocityState:
    """Map a set of canonical control tokens to a target velocity."""
    if rotation_speed_rad is None:
        rotation_speed_rad = math.radians(DEFAULT_ROTATION_SPEED_DEG)
    fwd = (1.0 if "forward" in controls else 0.0) - (1.0 if "back" in controls else 0.0)
    strafe = (1.0 if "strafe_right" in controls else 0.0) - (1.0 if "strafe_left" in controls else 0.0)
    yaw = (1.0 if "yaw_right" in controls else 0.0) - (1.0 if "yaw_left" in controls else 0.0)
    pit = (1.0 if "pitch_up" in controls else 0.0) - (1.0 if "pitch_down" in controls else 0.0)
    return VelocityState(
        tx=fwd * translation_speed,
        sx=strafe * translation_speed,
        yaw=yaw * rotation_speed_rad,
        pitch=pit * rotation_speed_rad,
    )


class CameraPoseIntegrator:
    """Integrate per-frame velocity into a camera-to-world pose.

    ``rot_y(yaw) @ R @ rot_x(pitch)`` for rotation; translation is on the
    horizontal (y=0) plane along the projected forward / right axes. Pitch
    saturates at ``±pitch_limit``. Avoids roll drift and "burrowing into the
    ground" when looking up/down.
    """

    def __init__(self, pitch_limit_rad: float = math.radians(DEFAULT_PITCH_LIMIT_DEG)) -> None:
        self.pose = np.eye(4, dtype=np.float64)
        self.pitch = 0.0
        self.pitch_limit = float(pitch_limit_rad)

    def step(self, v: VelocityState) -> np.ndarray:
        new_pitch = max(-self.pitch_limit, min(self.pitch_limit, self.pitch + v.pitch))
        pitch_step = new_pitch - self.pitch
        self.pitch = new_pitch

        R = self.pose[:3, :3]
        R_new = rot_y(v.yaw) @ R @ rot_x(pitch_step)

        fwd = R_new[:, 2].copy()
        fwd[1] = 0.0
        rgt = R_new[:, 0].copy()
        rgt[1] = 0.0
        fn = float(np.linalg.norm(fwd))
        rn = float(np.linalg.norm(rgt))
        if fn > 0:
            fwd /= fn + 1e-6
        if rn > 0:
            rgt /= rn + 1e-6
        T_ = self.pose[:3, 3] + fwd * v.tx + rgt * v.sx

        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, :3] = R_new
        self.pose[:3, 3] = T_
        return self.pose.copy()


# ---------------------------------------------------------------------------
# DSL parser + rollout.
# ---------------------------------------------------------------------------
def parse_action_string(action: str) -> list:
    """``"w-10,iw-5,none-3"`` → list of per-frame held-key lists.

    Each comma-separated segment is ``"<keys>-<duration>"``; ``<keys>`` may be
    multiple letters (combo keys held simultaneously) or the literal
    ``"none"``.
    """
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(f"Invalid action segment {segment!r}: expected '<keys>-<duration>'.")
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(f"Action segment {segment!r} has a non-positive duration {dur_str!r}.")
        n = int(dur_str)
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list = []
        else:
            bad = sorted({c for c in keys_lower if c not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(n)])
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
    smooth: bool = True,
) -> np.ndarray:
    """Roll out a ``(N+1, 4, 4)`` camera-to-world trajectory from an action string.

    The DSL groups segments as ``<keys>-<frames>`` joined by commas; ``"none"``
    means no keys held. Letters map to the unified control scheme:

        w / s   forward / back translation        a / d   yaw left / right
        i / k   pitch up / down                    j / l   strafe left / right

    With ``smooth=True`` the sana exponential velocity model is applied
    (instant on a new key press, gentle coast on release). Coordinate
    convention: OpenCV (``+X right, +Y down, +Z forward``); poses start at
    identity.
    """
    per_frame = parse_action_string(action)
    rotation_speed_rad = math.radians(rotation_speed_deg)
    integrator = CameraPoseIntegrator(math.radians(pitch_limit_deg))
    velocity = VelocityState()
    poses = [integrator.pose.copy()]
    last_controls: set = set()
    dt = 1.0 / FPS

    for keys in per_frame:
        controls = {DSL_KEY_TO_CONTROL[c] for c in keys if c in DSL_KEY_TO_CONTROL}
        target = controls_to_target_velocity(
            controls,
            translation_speed=translation_speed,
            rotation_speed_rad=rotation_speed_rad,
        )
        if smooth:
            # Snap on a fresh press so a new key takes effect immediately;
            # otherwise ease toward the target (gentle coast on release).
            if controls - last_controls:
                velocity.snap_to(target)
            else:
                velocity.step_toward(target, dt)
            last_controls = controls
        else:
            velocity = target
        poses.append(integrator.step(velocity))

    return np.stack(poses, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter to LongLive's storage convention.
# ---------------------------------------------------------------------------
def c2w_to_w2c_quat_t(c2w: np.ndarray) -> np.ndarray:
    """Convert ``(F, 4, 4)`` camera-to-world matrices to LongLive's
    ``(F, 7)`` ``[tx, ty, tz, qx, qy, qz, qw]`` w2c representation.

    Matches the format produced by
    ``scripts/data_preprocessing/build_camera_lmdb_5b.poses_from_pose_str`` so
    the result can be fed straight into ``build_viewmats_and_Ks``.
    """
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected (F, 4, 4) c2w, got shape {c2w.shape}")
    F = c2w.shape[0]
    out = np.zeros((F, 7), dtype=np.float32)
    for i in range(F):
        w2c = np.linalg.inv(c2w[i])
        out[i, :3] = w2c[:3, 3]
        out[i, 3:] = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    return out


def poses_from_action_string(
    action: str,
    n_latent: int,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
    smooth: bool = True,
) -> np.ndarray:
    """Roll out a sana-DSL action string and return ``(n_latent, 7)`` w2c poses
    in LongLive's storage format.

    The full per-frame c2w trajectory has length ``≈ Σ duration + 1``. We
    sub-sample it to ``n_latent`` frames at a stride matching the VAE
    temporal compression: each Wan2.2 latent frame corresponds to 4 raw
    frames (latent_T = (raw_T - 1) // 4 + 1), so we pick raw indices
    ``[0, 4, 8, ...]``. If the rollout is shorter than required we pad by
    repeating the last pose; if longer we truncate.
    """
    c2w_full = action_string_to_c2w(
        action,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_limit_deg=pitch_limit_deg,
        smooth=smooth,
    )
    # Wan2.2 VAE temporal stride is 4 raw frames per latent frame.
    raw_indices = [min(i * 4, c2w_full.shape[0] - 1) for i in range(n_latent)]
    c2w_lat = c2w_full[raw_indices]
    return c2w_to_w2c_quat_t(c2w_lat)
