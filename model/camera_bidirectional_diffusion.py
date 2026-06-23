# SPDX-License-Identifier: Apache-2.0
"""Camera-controlled bidirectional diffusion model for Wan2.2-TI2V-5B.

Implements the Phase-1 SFT objective described in
``minWM/training_wan.md §1.1``:
  - Non-causal Wan2.2-TI2V-5B backbone with PRoPE positional encoding.
  - Flow-matching loss with a single timestep shared across all frames.
  - Conditional-only (no DMD, no critic).
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from utils.i2v_conditioning import (
    _get_i2v_context_frames,
    _overwrite_i2v_context,
)
from utils.wan_5b_camera_wrapper import CameraWanDiffusionWrapper
from utils.wan_5b_wrapper import WanTextEncoder, WanVAEWrapper
from model.base import BaseModel


def _flow_matching_loss(pred: torch.Tensor, target: torch.Tensor,
                        weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted MSE used by Wan flow matching (matches minWM)."""
    diff = (pred.float() - target.float()) ** 2
    if weight is not None:
        diff = diff * weight.float()
    return diff.mean()


class CameraBidirectionalDiffusion(BaseModel):
    """BaseModel variant that owns a single bidirectional camera-aware
    generator. Mirrors `minWM/.../camera_bidirectional_diffusion.py`."""

    def _initialize_models(self, args, device):
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        # Force the camera wrapper to the bidirectional path.
        model_kwargs["use_camera"] = bool(model_kwargs.get("use_camera", True))
        # Drop is_causal / num_frame_per_block defaults coming from AR cfg.
        model_kwargs.pop("is_causal", None)

        model_name = model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(
                f"CameraBidirectionalDiffusion only supports Wan2.2-TI2V-5B, "
                f"got {model_name}"
            )
        # Cache the i2v switch on the model itself so generator_loss can read it
        # without going back to ``self.args``. Defaults to False (pure T2V).
        self.i2v = bool(getattr(args, "i2v", False))
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[CameraBidirectionalDiffusion] backbone={model_name} "
                  f"use_camera={model_kwargs['use_camera']} i2v={self.i2v}")

        self.generator = CameraWanDiffusionWrapper(is_causal=False, **model_kwargs)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        # Used by base._get_timestep; bidirectional mode applies a single
        # shared timestep, but we still mimic block-uniform in case of future
        # ablations.
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        index = self._get_timestep(
            0, self.scheduler.num_train_timesteps,
            batch_size, num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        timestep = self.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device,
        )

        # ---- I2V conditioning -------------------------------------------------
        # The first ``context_frames`` latents are kept clean (no noise, no
        # loss), exactly mirroring how Causal I2V handles the conditioning
        # frame in ``model/diffusion.py`` and ``utils/i2v_conditioning.py``.
        #
        # Under sequence parallelism, ``clean_latent`` is the LOCAL chunk of
        # the latent frame dimension. The trainer is responsible for passing
        # ``initial_latent=None`` on the SP ranks that do NOT own global
        # frame 0; here we just gate on ``self.i2v`` and the truthiness of
        # ``initial_latent``.
        context_frames = 0
        if self.i2v and initial_latent is not None:
            context_frames = _get_i2v_context_frames(clean_latent, initial_latent)
            if context_frames > 0:
                # Force the conditioning frames to t=0 in the schedule so the
                # model never sees them noised at training time.
                timestep[:, :context_frames] = 0
        # ----------------------------------------------------------------------

        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep
        )

        # Pin the first ``context_frames`` latents to the clean image latent
        # both in the noisy input and the target.
        if context_frames > 0:
            noisy_latents = _overwrite_i2v_context(
                noisy_latents, initial_latent, context_frames
            )
            training_target[:, :context_frames] = 0

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            viewmats=viewmats,
            Ks=Ks,
        )

        weight = self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frame)
        )
        weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Per-element MSE; broadcast the schedule weight, then optionally mask
        # out the I2V context frames before reducing.
        diff = (flow_pred.float() - training_target.float()) ** 2
        diff = diff * weight.float()
        if context_frames > 0:
            mask = torch.ones_like(diff)
            mask[:, :context_frames] = 0.0
            denom = mask.sum().clamp(min=1.0)
            loss = (diff * mask).sum() / denom
        else:
            loss = diff.mean()

        return loss, {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
            "i2v_context_frames": context_frames,
        }
