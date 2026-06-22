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
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[CameraBidirectionalDiffusion] backbone={model_name} "
                  f"use_camera={model_kwargs['use_camera']}")

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
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep
        )

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
        loss = _flow_matching_loss(flow_pred, training_target, weight=weight)

        return loss, {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
        }
