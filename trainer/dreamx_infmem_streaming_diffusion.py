"""Independent Echo-Infinity-style streaming trainer for DreamX Camera I2V AR + InfMem.

This path intentionally lives outside ``trainer/diffusion.py`` so the existing
DreamX Camera AR and teacher-forcing InfMem experiments remain unchanged.

The key difference from ``CausalDiffusion.generator_loss`` is graph lifetime:
we backward each streaming chunk before mutating the clean/context KV cache for
future chunks. This mirrors Echo-Infinity's streaming organization and avoids
keeping a full-clip mutable-cache graph alive until the end of the step.
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.distributed as dist
import wandb

from utils.distributed import EMA_FSDP
from .diffusion import Trainer as DiffusionTrainer


class Trainer(DiffusionTrainer):
    """DreamX Camera I2V AR + InfMem streaming trainer.

    Requirements:
      * CameraLatentLMDBDataset input (precomputed latents + viewmats/Ks).
      * ``DreamXInfMemWanDiffusionWrapper`` with an attached QueryMemoryEncoder.
      * ``sequence_parallel_size == 1`` for now; InfMem KV-cache training is not
        compatible with LongLive's SP attention override.
    """

    def __init__(self, config):
        super().__init__(config)
        if not self.use_camera_lmdb:
            raise ValueError(
                "dreamx_infmem_streaming_diffusion requires CameraLatentLMDBDataset "
                "input with precomputed camera latents."
            )
        if self.sequence_parallel_size != 1:
            raise ValueError(
                "dreamx_infmem_streaming_diffusion currently requires "
                "sequence_parallel_size=1."
            )
        if self.infmem_optimizer is None:
            raise ValueError(
                "dreamx_infmem_streaming_diffusion requires an attached "
                "QueryMemoryEncoder; check model_kwargs.wrapper_cls and memory_kwargs."
            )
        if bool(getattr(config, "gradient_checkpointing", False)) and self.is_main_process:
            print(
                "[DreamXInfMemStreaming] gradient_checkpointing is ignored for "
                "mutable KV-cache streaming training."
            )

    def _prepare_camera_batch(self, batch):
        self.model.generator.train()
        text_prompts = batch["prompts"]
        if (
            isinstance(text_prompts, list)
            and len(text_prompts) > 0
            and isinstance(text_prompts[0], list)
        ):
            text_prompts = [p for sublist in text_prompts for p in sublist]

        clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        viewmats = batch["viewmats"].to(device=self.device, dtype=self.dtype)
        Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)
        batch_size = len(text_prompts)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict

        return clean_latent, viewmats, Ks, conditional_dict

    def _train_one_step_camera(self, batch, accumulation_step=0, accumulation_steps=1):
        """Chunk-wise streaming supervised loss with Echo-style cache updates.

        For each chunk:
          1. Run noisy prediction with grad, reading the current clean cache.
          2. Backward that chunk loss immediately.
          3. Run clean/context cache + memory update under no_grad.

        This ensures no backward pass ever recomputes or references a KV cache
        after it has been mutated by later chunks.
        """
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        clean_latent, viewmats, Ks, conditional_dict = self._prepare_camera_batch(batch)
        batch_size, num_frame, _, height, width = clean_latent.shape
        frame_seq_length = (height * width) // 4
        context_frames = 1 if getattr(self.config, "i2v", False) else 0
        initial_latent = clean_latent[:, :context_frames] if context_frames > 0 else None

        if num_frame % self.model.num_frame_per_block != 0:
            raise ValueError(
                f"num_frame={num_frame} must be divisible by "
                f"num_frame_per_block={self.model.num_frame_per_block}."
            )

        index = self.model._get_timestep(
            0,
            self.model.scheduler.num_train_timesteps,
            batch_size,
            num_frame,
            self.model.num_frame_per_block,
            uniform_timestep=False,
        )
        timestep = self.model.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device
        )

        noise = torch.randn_like(clean_latent)
        noisy_latents = self.model.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.model.scheduler.training_target(clean_latent, noise, timestep)

        if context_frames > 0:
            noisy_latents[:, :context_frames] = initial_latent.to(
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )
            training_target[:, :context_frames] = 0
            timestep[:, :context_frames] = 0

        clean_latent_aug = clean_latent
        timestep_clean_aug = torch.zeros_like(timestep)

        kv_cache, crossattn_cache = self.model._build_streaming_caches(
            batch_size, clean_latent.dtype, clean_latent.device, frame_seq_length
        )
        from utils.infinity_memory_hooks import reset_infmem, maybe_detach_infmem

        if not reset_infmem(
            self.model.generator,
            batch_size=batch_size,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        ):
            raise RuntimeError(
                "Streaming InfMem trainer requires QueryMemoryEncoder, but reset failed."
            )

        if accumulation_step == 0:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.infmem_optimizer.zero_grad(set_to_none=True)

        valid_count = float(batch_size * max(num_frame - context_frames, 1))
        total_loss_value = torch.zeros([], device=self.device, dtype=torch.float32)
        chunk_count = 0

        for block_index, start in enumerate(range(0, num_frame, self.model.num_frame_per_block)):
            end = start + self.model.num_frame_per_block
            block_cond = self.model._conditional_for_streaming_block(
                conditional_dict, batch_size, block_index
            )
            if block_cond is not conditional_dict:
                for cache in crossattn_cache:
                    cache["is_init"] = False

            chunk_viewmats = viewmats[:, start:end]
            chunk_Ks = Ks[:, start:end]
            current_start = start * frame_seq_length

            flow_pred, _ = self.model.generator(
                noisy_image_or_video=noisy_latents[:, start:end],
                conditional_dict=block_cond,
                timestep=timestep[:, start:end],
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                defer_cache_updates=True,
                update_memory=False,
                apply_cache_updates=False,
                viewmats=chunk_viewmats,
                Ks=chunk_Ks,
            )

            per_frame = torch.nn.functional.mse_loss(
                flow_pred.float(),
                training_target[:, start:end].float(),
                reduction="none",
            ).mean(dim=(2, 3, 4))
            weight = self.model.scheduler.training_weight(
                timestep[:, start:end]
            ).unflatten(0, (batch_size, end - start)).float()
            per_frame = per_frame * weight
            if context_frames > 0 and start < context_frames:
                per_frame[:, : max(0, min(end, context_frames) - start)] = 0
            chunk_loss = per_frame.sum() / valid_count
            (chunk_loss / accumulation_steps).backward()
            total_loss_value = total_loss_value + chunk_loss.detach()

            # Echo-Infinity-style context/cache advancement: state update only.
            with torch.no_grad():
                self.model.generator(
                    noisy_image_or_video=clean_latent_aug[:, start:end],
                    conditional_dict=block_cond,
                    timestep=timestep_clean_aug[:, start:end],
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    update_memory=True,
                    viewmats=chunk_viewmats,
                    Ks=chunk_Ks,
                )

            chunk_count += 1
            maybe_detach_infmem(
                self.model.generator,
                chunk_count,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
            )

        if accumulation_step != accumulation_steps - 1:
            return None

        from utils.infinity_memory_hooks import sync_infmem_gradients, clip_infmem_grad_norm

        _sync_group = self.dp_group if self.dp_group is not None else None
        sync_infmem_gradients(self.model.generator, group=_sync_group, average=True)
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        _infmem_max_norm = getattr(self.config, "infmem_max_grad_norm", self.max_grad_norm)
        infmem_grad_norm = clip_infmem_grad_norm(self.model.generator, _infmem_max_norm)

        self.generator_optimizer.step()
        self.infmem_optimizer.step()

        self.step += 1

        if (
            (self.step >= self.config.ema_start_step)
            and (self.generator_ema is None)
            and (getattr(self.config, "ema_weight", None) is not None)
            and (self.config.ema_weight > 0)
        ):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        if self.generator_ema is not None and self.step >= self.config.ema_start_step:
            self.generator_ema.update(self.model.generator)

        wandb_loss_dict = {
            "generator_loss": total_loss_value.item(),
            "generator_grad_norm": generator_grad_norm.item(),
            "infmem_grad_norm": infmem_grad_norm.item(),
            "infmem_lr": self.infmem_optimizer.param_groups[0]["lr"],
        }
        infmem_log_str = self._log_infmem_eviction_diagnostics(wandb_loss_dict)
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(
                f"[stream-step {self.step:07d}] "
                f"generator_loss={wandb_loss_dict['generator_loss']:.6f}, "
                f"generator_grad_norm={wandb_loss_dict['generator_grad_norm']:.6f}"
                f"{infmem_log_str}"
            )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
