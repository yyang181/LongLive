# SPDX-License-Identifier: Apache-2.0
"""Pi3X-based intrinsics estimation helper for LongLive inference.

Ported from sana's ``inference_video_scripts/wm/inference_sana_wm.py``: when
the user does not supply intrinsics for an I2V source image, we estimate
``[fx, fy, cx, cy]`` (in pixel units) from the first frame using Pi3X, then
sanity-check the recovered FOV. Caller code is expected to convert the result
to the normalized ``[fx/W, fy/H, cx/W, cy/H]`` format LongLive's
``build_viewmats_and_Ks`` consumes.

Pi3X must be importable; we expect ``pi3`` to be on ``PYTHONPATH`` or
installed (the LongLive workspace already vendors it under ``code/Pi3``).
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# Pi3X intrinsics sanity check. Outside this range we refuse to proceed —
# matches sana's hard-stop behaviour (very small/very large FOV almost always
# means estimation failed).
MIN_FOV_DEG = 30.0
MAX_FOV_DEG = 130.0


def _resolve_pi3x():
    """Import Pi3X, attempting to add the vendored repo to sys.path on demand."""
    try:
        from pi3.models.pi3x import Pi3X
        from pi3.utils.geometry import recover_intrinsic_from_rays_d
        return Pi3X, recover_intrinsic_from_rays_d
    except ImportError:
        # Fallback: try the canonical workspace layout.
        import sys
        candidate = Path(__file__).resolve().parents[2] / "Pi3"
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        from pi3.models.pi3x import Pi3X  # noqa: E402
        from pi3.utils.geometry import recover_intrinsic_from_rays_d  # noqa: E402
        return Pi3X, recover_intrinsic_from_rays_d


def estimate_intrinsics_with_pi3x(
    image: Image.Image,
    device: torch.device,
    *,
    logger: logging.Logger | None = None,
    pi3x_repo_id: str = "yyfz233/Pi3X",
) -> np.ndarray:
    """Estimate ``(fx, fy, cx, cy)`` for ``image`` using Pi3X.

    The image is internally resized to a Pi3X-friendly shape (multiple of 14,
    pixel budget ≤ 255k), and the returned intrinsics are scaled back to
    ``image.size``. We assert ``MIN_FOV_DEG < horizontal_fov < MAX_FOV_DEG``
    and abort otherwise so the user knows to provide intrinsics manually.

    Args:
        image: PIL.Image in original resolution (do NOT pre-resize).
        device: target torch device for Pi3X inference.
        logger: optional logger (defaults to a module logger).
        pi3x_repo_id: Hugging Face repo id for the Pi3X checkpoint.

    Returns:
        ``np.ndarray`` shape ``(4,)``: ``[fx, fy, cx, cy]`` in *pixel* units
        of the original image (NOT normalized).
    """
    if logger is None:
        logger = logging.getLogger("longlive.intrinsics")

    Pi3X, recover_intrinsic_from_rays_d = _resolve_pi3x()

    logger.warning(
        "Intrinsics not provided — estimating with Pi3X from the input image. "
        "Estimation errors propagate into the generated camera geometry; "
        "supply explicit intrinsics in the YAML when accurate values exist."
    )

    W_orig, H_orig = image.size
    pixel_limit = 255_000
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    W_t, H_t = W_orig * scale, H_orig * scale
    k, m = max(1, round(W_t / 14)), max(1, round(H_t / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_t / H_t:
            k -= 1
        else:
            m -= 1
    W_model, H_model = max(1, k) * 14, max(1, m) * 14
    resized = image.resize((W_model, H_model), Image.Resampling.LANCZOS)
    tensor = T.ToTensor()(resized).unsqueeze(0).unsqueeze(0).to(device)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    model = Pi3X.from_pretrained(pi3x_repo_id).to(device).eval()
    if hasattr(model, "disable_multimodal"):
        model.disable_multimodal()
    model.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model(imgs=tensor)
    rays_d = torch.nn.functional.normalize(out["local_points"], dim=-1)
    K_model = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    K_model_np = K_model[0, 0].detach().cpu().float().numpy()

    sx, sy = W_orig / W_model, H_orig / H_model
    fx, fy = float(K_model_np[0, 0] * sx), float(K_model_np[1, 1] * sy)
    cx, cy = float(K_model_np[0, 2] * sx), float(K_model_np[1, 2] * sy)

    fov_x = math.degrees(2.0 * math.atan(W_orig / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(H_orig / (2.0 * fy)))
    logger.info(
        f"Pi3X intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
        f"(FOV: H={fov_x:.1f}° V={fov_y:.1f}°)"
    )
    if not (MIN_FOV_DEG < fov_x < MAX_FOV_DEG and MIN_FOV_DEG < fov_y < MAX_FOV_DEG):
        raise SystemExit(
            f"Pi3X-estimated FOV (H={fov_x:.1f}°, V={fov_y:.1f}°) falls outside "
            f"[{MIN_FOV_DEG}°, {MAX_FOV_DEG}°]. Estimation likely failed; "
            f"set inference.fx_norm/fy_norm/cx_norm/cy_norm in the YAML."
        )

    # Free Pi3X memory before downstream models load — this is the biggest
    # peak-VRAM saver in practice.
    del model, out, K_model, rays_d, tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def transform_intrinsics_for_resize(
    intrinsics_pix: np.ndarray,
    src_size: tuple,
    target_size: tuple,
) -> np.ndarray:
    """Scale ``[fx, fy, cx, cy]`` for a pure resize (no crop).

    LongLive's I2V loader uses ``Image.resize((target_w, target_h), BICUBIC)``
    without aspect-preserving / center-crop, so the only transformation
    needed is a per-axis scale.

    Args:
        intrinsics_pix: ``(4,)`` ``[fx, fy, cx, cy]`` in pixels of ``src_size``.
        src_size: ``(W_src, H_src)``.
        target_size: ``(W_tgt, H_tgt)``.

    Returns:
        ``(4,)`` ``[fx, fy, cx, cy]`` in pixels of ``target_size``.
    """
    src_w, src_h = src_size
    tgt_w, tgt_h = target_size
    sx, sy = tgt_w / src_w, tgt_h / src_h
    out = intrinsics_pix.astype(np.float32, copy=True)
    out[0] *= sx
    out[2] *= sx
    out[1] *= sy
    out[3] *= sy
    return out


def normalize_intrinsics(
    intrinsics_pix: np.ndarray,
    target_size: tuple,
) -> np.ndarray:
    """Convert pixel-space ``[fx, fy, cx, cy]`` → LongLive's normalized form
    ``[fx/W, fy/H, cx/W, cy/H]`` (a float32 ``(4,)`` array).
    """
    tgt_w, tgt_h = target_size
    out = intrinsics_pix.astype(np.float32, copy=True)
    out[0] /= float(tgt_w)
    out[2] /= float(tgt_w)
    out[1] /= float(tgt_h)
    out[3] /= float(tgt_h)
    return out
