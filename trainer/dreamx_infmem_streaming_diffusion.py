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
import random
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import wandb

from utils.config import section_get
from utils.distributed import EMA_FSDP, barrier
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
        self.streaming_activation_checkpointing = bool(
            getattr(config, "streaming_activation_checkpointing", True)
        )
        self.streaming_fsdp_no_sync = bool(
            getattr(config, "streaming_fsdp_no_sync", True)
        )
        self.streaming_cache_source = str(
            getattr(config, "streaming_cache_source", "prediction")
        ).strip().lower()
        if self.streaming_cache_source != "prediction":
            raise ValueError(
                "DreamX InfMem long-rollout training requires "
                "streaming_cache_source='prediction'. Feeding GT latents into "
                "the clean recache path creates a train/inference mismatch and "
                f"can hide rollout drift; got {self.streaming_cache_source!r}."
            )
        self.streaming_bounded_window_size = int(
            getattr(config, "streaming_bounded_window_size", 0) or 0
        )
        self._streaming_sequence_state = None
        if self.streaming_bounded_window_size > 0:
            block_size = int(self.model.num_frame_per_block)
            if self.streaming_bounded_window_size % block_size != 0:
                raise ValueError(
                    "streaming_bounded_window_size must be divisible by "
                    f"num_frame_per_block={block_size}; got "
                    f"{self.streaming_bounded_window_size}."
                )
            if self.streaming_bounded_window_size <= int(
                config.model_kwargs.local_attn_size
            ):
                raise ValueError(
                    "streaming_bounded_window_size must exceed local_attn_size "
                    "so every full window exercises physical KV eviction."
                )
            if int(getattr(config, "gradient_accumulation_steps", 1)) != 1:
                raise ValueError(
                    "Persistent bounded-window rollout currently requires "
                    "gradient_accumulation_steps=1."
                )
            evaluation_interval = section_get(
                config, "evaluation", "interval", 0
            )
            if int(evaluation_interval or 0) != 0:
                raise ValueError(
                    "Persistent bounded-window rollout requires evaluation.interval=0; "
                    "inference would reset the live training cache/memory state."
                )
        if self.streaming_fsdp_no_sync and not hasattr(
            self.model.generator, "no_sync"
        ):
            raise TypeError(
                "streaming_fsdp_no_sync requires the generator to be the "
                "root FSDP module."
            )
        if self.streaming_activation_checkpointing and self.is_main_process:
            print(
                "[DreamXInfMemStreaming] block activation checkpointing enabled "
                "for immediate-backward prediction forwards."
            )
        if self.streaming_fsdp_no_sync and self.is_main_process:
            print(
                "[DreamXInfMemStreaming] FSDP gradient synchronization is deferred "
                "to the final chunk of the final accumulation micro-step."
            )
        if self.is_main_process:
            print(
                "[DreamXInfMemStreaming] clean cache/memory advancement uses "
                f"{self.streaming_cache_source!r} latents."
            )
            if self.streaming_bounded_window_size > 0:
                print(
                    "[DreamXInfMemStreaming] persistent long rollout enabled: "
                    f"bounded_window={self.streaming_bounded_window_size} frames."
                )

    def _fsdp_gradient_sync_context(self, sync_gradients):
        """Return root-FSDP no_sync for intermediate streaming backwards."""
        if (self.generator_optimizer is None
                or sync_gradients
                or not self.streaming_fsdp_no_sync):
            return nullcontext()
        return self.model.generator.no_sync()

    def _select_streaming_cache_base(self, x0_pred):
        """Detach the model prediction used to advance persistent KV/memory."""
        if self.streaming_cache_source != "prediction":
            raise RuntimeError(
                "Persistent recache must use x0_pred, never the GT latent."
            )
        return x0_pred.detach().clone()

    @staticmethod
    def _next_bounded_window(cursor, num_frame, window_size):
        """Return the next globally aligned half-open rollout window."""
        cursor = int(cursor)
        num_frame = int(num_frame)
        window_size = int(window_size)
        if cursor < 0 or cursor >= num_frame:
            raise ValueError(
                f"cursor={cursor} must satisfy 0 <= cursor < {num_frame}."
            )
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        return cursor, min(cursor + window_size, num_frame)

    def _start_bounded_sequence(self, batch):
        """Initialize one GT trajectory while preserving self-fed runtime state."""
        if batch is None:
            raise RuntimeError("A new bounded rollout requires a camera batch.")
        clean_latent, viewmats, Ks, conditional_dict = self._prepare_camera_batch(
            batch
        )
        batch_size, num_frame, _, height, width = clean_latent.shape
        block_size = int(self.model.num_frame_per_block)
        if num_frame % block_size != 0:
            raise ValueError(
                f"num_frame={num_frame} must be divisible by block_size={block_size}."
            )
        if num_frame % self.streaming_bounded_window_size != 0:
            raise ValueError(
                f"num_frame={num_frame} must be divisible by "
                f"streaming_bounded_window_size={self.streaming_bounded_window_size}."
            )

        frame_seq_length = (height * width) // 4
        context_frames = 1 if getattr(self.config, "i2v", False) else 0
        initial_latent = (
            clean_latent[:, :context_frames].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            if context_frames > 0
            else None
        )
        kv_cache, crossattn_cache = self.model._build_streaming_caches(
            batch_size, self.dtype, self.device, frame_seq_length
        )
        from utils.infinity_memory_hooks import reset_infmem

        if not reset_infmem(
            self.model.generator,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        ):
            raise RuntimeError(
                "Persistent bounded rollout requires QueryMemoryEncoder."
            )
        self._streaming_sequence_state = {
            "clean_latent": clean_latent,
            "viewmats": viewmats,
            "Ks": Ks,
            "conditional_dict": conditional_dict,
            "batch_size": batch_size,
            "num_frame": num_frame,
            "frame_seq_length": frame_seq_length,
            "context_frames": context_frames,
            "initial_latent": initial_latent,
            "kv_cache": kv_cache,
            "crossattn_cache": crossattn_cache,
            "cursor": 0,
            "chunk_count": 0,
        }
        if self.is_main_process:
            print(
                "[DreamXInfMemStreaming] new persistent GT trajectory: "
                f"frames={num_frame}, bounded_window="
                f"{self.streaming_bounded_window_size}."
            )
        return self._streaming_sequence_state

    def _prepare_camera_batch(self, batch):
        """Prepare metadata while keeping the full video tensors on CPU.

        The streaming trainer only needs one model block on the GPU at a
        time. Moving the tensors here used to materialize the complete clip
        on every rank and torch.randn_like(clean_latent) later allocated a
        second full-clip tensor. Consequently GPU memory still grew linearly
        with image_or_video_shape[1] even though autograd was chunked.

        CameraLatentLMDBDataset and its collate function return CPU tensors.
        Keep those tensors as the backing store and transfer temporal slices
        in _train_one_step_camera instead.
        """
        self.model.generator.train()
        text_prompts = batch["prompts"]
        if (
            isinstance(text_prompts, list)
            and len(text_prompts) > 0
            and isinstance(text_prompts[0], list)
        ):
            text_prompts = [p for sublist in text_prompts for p in sublist]

        clean_latent = batch["clean_latent"]
        viewmats = batch["viewmats"]
        Ks = batch["Ks"]
        for name, tensor in (
            ("clean_latent", clean_latent),
            ("viewmats", viewmats),
            ("Ks", Ks),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"batch[{name!r}] must be a tensor, got {type(tensor)!r}."
                )
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"Streaming camera batch tensor {name!r} must remain on CPU; "
                    f"got device={tensor.device}. Moving the full clip to CUDA "
                    "makes memory scale with image_or_video_shape[1]."
                )
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

    def _streaming_inject_error_block(
        self,
        tensor,
        buffer,
        index_chunk,
        block_index,
        *,
        sample_any_t=False,
    ):
        if buffer is None or buffer.is_empty():
            return tensor, False
        if (
            getattr(self.model, "er_skip_block_0", False)
            and (getattr(self.model, "er_block_offset", 0) + block_index) == 0
        ):
            return tensor, False

        result = tensor.clone()
        injected = False
        for batch_idx in range(result.shape[0]):
            if sample_any_t:
                if getattr(buffer, "num_blocks", 0) > 0:
                    err = buffer.sample_pos_any_t(
                        block_index, device=result.device, dtype=result.dtype
                    )
                else:
                    err = buffer.sample_global(device=result.device, dtype=result.dtype)
            else:
                block_pos = block_index if getattr(buffer, "num_blocks", 0) > 0 else None
                err = buffer.sample(
                    index_chunk[batch_idx, 0].item(),
                    device=result.device,
                    dtype=result.dtype,
                    block_pos=block_pos,
                )
            if err is not None:
                result[batch_idx] = result[batch_idx] + err[: result.shape[1]]
                injected = True
        return result, injected

    def _streaming_collect_error_items(self, buffer, error_chunk, index_chunk, block_index):
        if buffer is None:
            return []

        use_distributed = self.step <= getattr(self.model, "er_buffer_warmup_iter", 0)
        if not use_distributed or not dist.is_initialized() or dist.get_world_size() <= 1:
            err_list = [error_chunk.detach().contiguous()]
            idx_list = [index_chunk.detach().contiguous()]
        else:
            if getattr(buffer, "num_blocks", 0) > 0 and self.dp_group is not None:
                comm_group = self.dp_group
                comm_size = dist.get_world_size(comm_group)
            else:
                comm_group = None
                comm_size = dist.get_world_size()
            err_local = error_chunk.detach().contiguous()
            idx_local = index_chunk.detach().contiguous()
            err_list = [torch.empty_like(err_local) for _ in range(comm_size)]
            idx_list = [torch.empty_like(idx_local) for _ in range(comm_size)]
            if comm_group is None:
                dist.all_gather(err_list, err_local)
                dist.all_gather(idx_list, idx_local)
            else:
                dist.all_gather(err_list, err_local, group=comm_group)
                dist.all_gather(idx_list, idx_local, group=comm_group)

        block_pos = block_index if getattr(buffer, "num_blocks", 0) > 0 else None
        items = []
        for err_rank, idx_rank in zip(err_list, idx_list):
            for batch_idx in range(err_rank.shape[0]):
                items.append((err_rank[batch_idx], idx_rank[batch_idx, 0].item(), block_pos))
        return items

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

        from utils.infinity_memory_hooks import maybe_detach_infmem, reset_infmem

        persistent_window = self.streaming_bounded_window_size > 0
        if persistent_window:
            state = self._streaming_sequence_state
            if state is None:
                state = self._start_bounded_sequence(batch)
            clean_latent_cpu = state["clean_latent"]
            viewmats_cpu = state["viewmats"]
            Ks_cpu = state["Ks"]
            conditional_dict = state["conditional_dict"]
            batch_size = state["batch_size"]
            num_frame = state["num_frame"]
            frame_seq_length = state["frame_seq_length"]
            context_frames = state["context_frames"]
            initial_latent = state["initial_latent"]
            kv_cache = state["kv_cache"]
            crossattn_cache = state["crossattn_cache"]
            window_start, window_end = self._next_bounded_window(
                state["cursor"], num_frame, self.streaming_bounded_window_size
            )
            chunk_count = int(state["chunk_count"])
        else:
            clean_latent_cpu, viewmats_cpu, Ks_cpu, conditional_dict = (
                self._prepare_camera_batch(batch)
            )
            batch_size, num_frame, _, height, width = clean_latent_cpu.shape
            frame_seq_length = (height * width) // 4
            context_frames = 1 if getattr(self.config, "i2v", False) else 0
            initial_latent = (
                clean_latent_cpu[:, :context_frames].to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
                if context_frames > 0
                else None
            )
            if num_frame % self.model.num_frame_per_block != 0:
                raise ValueError(
                    f"num_frame={num_frame} must be divisible by "
                    f"num_frame_per_block={self.model.num_frame_per_block}."
                )
            kv_cache, crossattn_cache = self.model._build_streaming_caches(
                batch_size, self.dtype, self.device, frame_seq_length
            )
            if not reset_infmem(
                self.model.generator,
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            ):
                raise RuntimeError(
                    "Streaming InfMem trainer requires QueryMemoryEncoder, "
                    "but reset failed."
                )
            window_start = 0
            window_end = num_frame
            chunk_count = 0

        if accumulation_step == 0:
            if self.generator_optimizer is not None:
                self.generator_optimizer.zero_grad(set_to_none=True)
            self.infmem_optimizer.zero_grad(set_to_none=True)

        er_ready = (
            self.model.error_buffer is not None
            and self.model.noise_error_buffer is not None
            and self.step >= getattr(self.model, "er_start_step", 0)
        )
        er_use_clean = (
            er_ready
            and getattr(self.model, "er_clean_prob", 0.0) > 0
            and random.random() < self.model.er_clean_prob
        )
        er_context_injected = False
        er_latent_injected = False
        er_noise_injected = False

        context_in_window = max(
            0, min(window_end, context_frames) - window_start
        )
        valid_count = float(
            batch_size * max(window_end - window_start - context_in_window, 1)
        )
        total_loss_value = torch.zeros([], device=self.device, dtype=torch.float32)
        num_chunks = (
            (window_end - window_start) // self.model.num_frame_per_block
        )
        if (
            accumulation_step == 0
            and self.is_main_process
            and not getattr(self, "_fsdp_sync_schedule_logged", False)
        ):
            sync_backwards = (
                1 if self.streaming_fsdp_no_sync
                else num_chunks * accumulation_steps
            )
            print(
                "[DreamXInfMemStreaming] FSDP backward sync schedule: "
                f"chunks={num_chunks}, accumulation_steps={accumulation_steps}, "
                f"synchronized_backwards_per_optimizer_step={sync_backwards}."
            )
            self._fsdp_sync_schedule_logged = True

        for start in range(
            window_start, window_end, self.model.num_frame_per_block
        ):
            block_index = start // self.model.num_frame_per_block
            end = start + self.model.num_frame_per_block
            block_cond = self.model._conditional_for_streaming_block(
                conditional_dict, batch_size, block_index
            )
            if block_cond is not conditional_dict:
                for cache in crossattn_cache:
                    cache["is_init"] = False

            # Keep the complete clip in host memory. Only this temporal block
            # and its noise tensor live on CUDA.
            clean_chunk = clean_latent_cpu[:, start:end].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            chunk_viewmats = viewmats_cpu[:, start:end].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            chunk_Ks = Ks_cpu[:, start:end].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            current_start = start * frame_seq_length
            chunk_frames = end - start
            index_chunk = self.model._get_timestep(
                0,
                self.model.scheduler.num_train_timesteps,
                batch_size,
                chunk_frames,
                self.model.num_frame_per_block,
                uniform_timestep=False,
            )
            timestep_chunk = self.model.scheduler.timesteps[index_chunk].to(
                dtype=self.dtype, device=self.device
            )
            if context_frames > 0 and start < context_frames:
                pinned = min(chunk_frames, context_frames - start)
                timestep_chunk[:, :pinned] = 0

            clean_chunk_for_noise = clean_chunk
            noise_chunk_for_train = torch.randn_like(clean_chunk)
            if er_ready and not er_use_clean:
                if (
                    self.model.er_noise_inject_prob > 0
                    and not self.model.noise_error_buffer.is_empty()
                    and random.random() < self.model.er_noise_inject_prob
                ):
                    noise_chunk_for_train, injected = self._streaming_inject_error_block(
                        noise_chunk_for_train,
                        self.model.noise_error_buffer,
                        index_chunk,
                        block_index,
                    )
                    er_noise_injected = er_noise_injected or injected
                if (
                    self.model.er_latent_inject_prob > 0
                    and not self.model.error_buffer.is_empty()
                    and random.random() < self.model.er_latent_inject_prob
                ):
                    clean_chunk_for_noise, injected = self._streaming_inject_error_block(
                        clean_chunk_for_noise,
                        self.model.error_buffer,
                        index_chunk,
                        block_index,
                    )
                    er_latent_injected = er_latent_injected or injected

            noisy_chunk = self.model.scheduler.add_noise(
                clean_chunk_for_noise.flatten(0, 1),
                noise_chunk_for_train.flatten(0, 1),
                timestep_chunk.flatten(0, 1),
            ).unflatten(0, (batch_size, end - start))
            training_target_chunk = self.model.scheduler.training_target(
                clean_chunk, noise_chunk_for_train, timestep_chunk
            )
            if context_frames > 0 and start < context_frames:
                pinned = max(0, min(end, context_frames) - start)
                noisy_chunk[:, :pinned] = initial_latent[:, start:start + pinned].to(
                    device=noisy_chunk.device, dtype=noisy_chunk.dtype
                )
                training_target_chunk[:, :pinned] = 0

            reached_sync_boundary = (
                end == window_end if persistent_window else end == num_frame
            )
            sync_gradients = (
                reached_sync_boundary
                and accumulation_step == accumulation_steps - 1
            )
            with self._fsdp_gradient_sync_context(sync_gradients):
                flow_pred, x0_pred = self.model.generator(
                    noisy_image_or_video=noisy_chunk,
                    conditional_dict=block_cond,
                    timestep=timestep_chunk,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    defer_cache_updates=True,
                    update_memory=False,
                    apply_cache_updates=False,
                    checkpoint_blocks=self.streaming_activation_checkpointing,
                    viewmats=chunk_viewmats,
                    Ks=chunk_Ks,
                )

                per_frame = torch.nn.functional.mse_loss(
                    flow_pred.float(),
                    training_target_chunk.float(),
                    reduction="none",
                ).mean(dim=(2, 3, 4))
                weight = self.model.scheduler.training_weight(
                    timestep_chunk
                ).unflatten(0, (batch_size, end - start)).float()
                per_frame = per_frame * weight
                if context_frames > 0 and start < context_frames:
                    per_frame[:, : max(0, min(end, context_frames) - start)] = 0
                chunk_loss = per_frame.sum() / valid_count
                scaled_chunk_loss = chunk_loss / accumulation_steps
                if scaled_chunk_loss.requires_grad:
                    scaled_chunk_loss.backward()
                elif not bool(getattr(self.config, "train_memory_only", False)):
                    raise RuntimeError(
                        "Streaming chunk loss has no gradient path."
                    )
            total_loss_value = total_loss_value + chunk_loss.detach()

            # Let the loss consume the prior query state before truncating
            # BPTT; the newly updated state must survive into the next block.
            chunk_count += 1
            maybe_detach_infmem(
                self.model.generator,
                chunk_count,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
            )

            # Advance persistent KV/memory only from the model prediction.
            # GT remains the supervised flow-matching target above and never
            # enters generated-frame recache. The sole exception is the fixed
            # first I2V conditioning frame, exactly as at inference.
            with torch.no_grad():
                context_base = self._select_streaming_cache_base(x0_pred)
                if er_ready and not er_use_clean:
                    if (
                        getattr(self.model, "er_context_inject_on_self_feed", False)
                        and
                        self.model.er_context_inject_prob > 0
                        and not self.model.error_buffer.is_empty()
                        and random.random() < self.model.er_context_inject_prob
                    ):
                        context_base, injected = self._streaming_inject_error_block(
                            context_base,
                            self.model.error_buffer,
                            index_chunk,
                            block_index,
                            sample_any_t=True,
                        )
                        er_context_injected = er_context_injected or injected

                context_timestep = torch.full_like(
                    timestep_chunk,
                    float(getattr(self.config, "context_noise", 0)),
                )
                context_noise = torch.randn_like(context_base)
                if context_frames > 0 and start < context_frames:
                    pinned = max(0, min(end, context_frames) - start)
                    context_base[:, :pinned] = initial_latent[:, start:start + pinned].to(
                        device=context_base.device, dtype=context_base.dtype
                    )
                    context_noise[:, :pinned] = 0
                    context_timestep[:, :pinned] = 0
                context_noisy = self.model.scheduler.add_noise(
                    context_base.flatten(0, 1),
                    context_noise.flatten(0, 1),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, (batch_size, end - start))

                self.model.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    update_memory=True,
                    memory_update_with_grad=True,
                    viewmats=chunk_viewmats,
                    Ks=chunk_Ks,
                )

                if er_ready:
                    latent_err = x0_pred.detach() - clean_chunk.detach()
                    sigma = self.model.scheduler.sigmas.to(flow_pred.device)[index_chunk].reshape(
                        batch_size, end - start, 1, 1, 1
                    ).to(flow_pred.dtype)
                    noise_err = (
                        flow_pred.detach() - training_target_chunk.detach()
                    ) * (1.0 - sigma)

                    lat_items = self._streaming_collect_error_items(
                        self.model.error_buffer, latent_err, index_chunk, block_index
                    )
                    noise_items = self._streaming_collect_error_items(
                        self.model.noise_error_buffer, noise_err, index_chunk, block_index
                    )
                    should_update = True
                    if (
                        er_use_clean
                        and random.random() >= self.model.er_clean_buffer_update_prob
                    ):
                        should_update = False
                    if should_update:
                        self.model._apply_gathered_items(self.model.error_buffer, lat_items)
                        self.model._apply_gathered_items(
                            self.model.noise_error_buffer, noise_items
                        )

            # Drop the final references to this block's CUDA tensors before
            # loading the next one. The caching allocator can reuse the same
            # storage, keeping peak memory independent of clip length.
            del (
                clean_chunk,
                clean_chunk_for_noise,
                noise_chunk_for_train,
                noisy_chunk,
                training_target_chunk,
                flow_pred,
                x0_pred,
                per_frame,
                weight,
                chunk_loss,
                scaled_chunk_loss,
                chunk_viewmats,
                chunk_Ks,
                index_chunk,
                timestep_chunk,
                context_base,
                context_noise,
                context_timestep,
                context_noisy,
            )
            if er_ready:
                del latent_err, sigma, noise_err, lat_items, noise_items

        sequence_complete = bool(persistent_window and window_end >= num_frame)
        if persistent_window:
            state["cursor"] = window_end
            state["chunk_count"] = chunk_count
            # The final recache update created the query state for the next
            # window. Cut its old-parameter graph before optimizer.step while
            # keeping the state values and physical KV cache alive.
            from utils.infinity_memory_hooks import get_infmem_encoder

            encoder = get_infmem_encoder(self.model.generator)
            if encoder is None:
                raise RuntimeError(
                    "QueryMemoryEncoder disappeared during persistent rollout."
                )
            encoder.detach_state()

        if accumulation_step != accumulation_steps - 1:
            return None

        from utils.infinity_memory_hooks import sync_infmem_gradients, clip_infmem_grad_norm

        _sync_group = self.dp_group if self.dp_group is not None else None
        sync_infmem_gradients(self.model.generator, group=_sync_group, average=True)
        if self.generator_optimizer is not None:
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)
        _infmem_max_norm = getattr(self.config, "infmem_max_grad_norm", self.max_grad_norm)
        infmem_grad_norm = clip_infmem_grad_norm(self.model.generator, _infmem_max_norm)

        if self.generator_optimizer is not None:
            self.generator_optimizer.step()
        self.infmem_optimizer.step()

        self.step += 1

        ema_enabled = (
            self.step >= self.config.ema_start_step
            and getattr(self.config, "ema_weight", None) is not None
            and self.config.ema_weight > 0
        )
        if ema_enabled:
            if (
                self.generator_optimizer is not None
                and not self.is_lora_enabled
                and self.generator_ema is None
            ):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            if self.infmem_ema is None:
                from utils.infinity_memory_hooks import InfMemEMA
                self.infmem_ema = InfMemEMA(self.model.generator, decay=self.config.ema_weight)
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            if self.infmem_ema is not None:
                self.infmem_ema.update(self.model.generator)

        wandb_loss_dict = {
            "generator_loss": total_loss_value.item(),
            "generator_grad_norm": generator_grad_norm.item(),
            "infmem_grad_norm": infmem_grad_norm.item(),
            "infmem_lr": self.infmem_optimizer.param_groups[0]["lr"],
        }
        if persistent_window:
            wandb_loss_dict.update({
                "rollout_window_start": window_start,
                "rollout_window_end": window_end,
                "rollout_total_frames": num_frame,
            })
        if self.model.error_buffer is not None:
            buf_stats = self.model.error_buffer.stats()
            noise_buf_stats = self.model.noise_error_buffer.stats()
            wandb_loss_dict.update({
                "er_total_added": buf_stats["total_added"],
                "er_filled_buckets": buf_stats["filled_buckets"],
                "er_total_entries": buf_stats["total_entries"],
                "er_noise_total_entries": noise_buf_stats["total_entries"],
                "er_injected": er_context_injected,
                "er_latent_injected": er_latent_injected,
                "er_noise_injected": er_noise_injected,
            })
        infmem_log_str = self._log_infmem_eviction_diagnostics(wandb_loss_dict)
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(
                f"[stream-step {self.step:07d}] "
                f"generator_loss={wandb_loss_dict['generator_loss']:.6f}, "
                f"generator_grad_norm={wandb_loss_dict['generator_grad_norm']:.6f}"
                + (
                    f", rollout_window={window_start}:{window_end}/{num_frame}"
                    if persistent_window else ""
                )
                + f"{infmem_log_str}"
            )

        if sequence_complete:
            self._streaming_sequence_state = None

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        """Consume one dataset sample per persistent long rollout, not per window."""
        if self.streaming_bounded_window_size <= 0:
            return super().train()

        if getattr(self.config, "generate_before_train", False):
            if self.is_main_process:
                print(
                    "[generate_before_train] Running evaluation inference "
                    "before training starts..."
                )
            self._run_evaluation_inference()
            barrier()
            return

        while True:
            batch = (
                next(self.dataloader)
                if self._streaming_sequence_state is None
                else None
            )
            self.train_one_step(batch, accumulation_step=0, accumulation_steps=1)

            if (
                not self.config.no_save
                and self.step % self.config.log_iters == 0
            ):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log(
                            {"per iteration time": current_time - self.previous_time},
                            step=self.step,
                        )
                    self.previous_time = current_time

            if self.step >= self.config.max_iters:
                break
