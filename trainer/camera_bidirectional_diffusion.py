# SPDX-License-Identifier: Apache-2.0
"""Camera (PRoPE) Bidirectional SFT trainer for Wan2.2-TI2V-5B.

Mirrors `minWM/Wan21/wan_trainer/camera_bidirectional_diffusion.py` but is
self-contained for the LongLive infrastructure (no SP, no multi-shot,
no error-buffer). Reuses LongLive's:
    - utils.distributed.fsdp_wrap / EMA_FSDP / launch_distributed_job
    - CameraBidirectionalDiffusion model
    - CameraLatentLMDBDataset

Saves checkpoints to ``{logdir}/checkpoint_model_{step:06d}/`` exactly like
the standard LongLive trainer, but with a much smaller payload (only the
generator + EMA + optimizer states are needed).
"""

import gc
import logging
import os
import re
import time
import types

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from model.camera_bidirectional_diffusion import CameraBidirectionalDiffusion
from utils.camera_dataset import CameraLatentLMDBDataset
from utils.dataset import RepeatDataset, cycle
from utils.config import wan_default_config
from utils.distributed import (
    EMA_FSDP, barrier, fsdp_state_dict, fsdp_wrap, launch_distributed_job,
)
from utils.misc import set_seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # ---- Sequence Parallel (SP) ----
        # The 5B backbone + bidirectional camera-PRoPE SFT does not fit in
        # memory for the full latent on a single GPU, so we shard the latent
        # *frame* dimension across an SP group. ``world_size = sp_size * dp_size``.
        self.sequence_parallel_size = int(getattr(config, "sequence_parallel_size", 1) or 1)
        self.data_parallel_size = (
            world_size // self.sequence_parallel_size
            if self.sequence_parallel_size > 1 else world_size
        )
        self.sp_group = None
        self.dp_group = None
        self.sp_rank = 0
        self.dp_rank = global_rank

        if self.sequence_parallel_size > 1:
            from wan_5b.distributed.sp_training import (
                set_data_parallel_group,
                set_sequence_parallel_group,
            )
            model_name = getattr(
                getattr(config, "model_kwargs", None), "model_name",
                getattr(config, "model_name", "")) or ""
            assert "Wan2.2-TI2V-5B" in model_name, (
                f"sequence_parallel_size>1 is only supported for "
                f"Wan2.2-TI2V-5B, got model_name={model_name!r}")
            assert world_size % self.sequence_parallel_size == 0, (
                f"world_size ({world_size}) must be divisible by "
                f"sequence_parallel_size ({self.sequence_parallel_size})")
            # The sharded frame count must divide evenly across the SP group and
            # the head count must be divisible by sp_size (Ulysses head split).
            num_latent_frames = int(list(config.image_or_video_shape)[1])
            num_heads = int(wan_default_config[model_name]["num_heads"])
            assert num_latent_frames % self.sequence_parallel_size == 0, (
                f"latent frames ({num_latent_frames}) must be divisible by "
                f"sequence_parallel_size ({self.sequence_parallel_size})")
            assert num_heads % self.sequence_parallel_size == 0, (
                f"num_heads ({num_heads}) must be divisible by "
                f"sequence_parallel_size ({self.sequence_parallel_size})")

            sp_size = self.sequence_parallel_size
            dp_size = self.data_parallel_size
            # SP groups: contiguous ranks [g*sp, (g+1)*sp). DP groups: strided
            # [k, sp+k, ...] so peers with the same SP rank own the same chunk
            # position across DP replicas.
            sp_groups = [dist.new_group(ranks=list(range(g * sp_size, (g + 1) * sp_size)))
                         for g in range(dp_size)]
            self.sp_group = sp_groups[global_rank // sp_size]
            set_sequence_parallel_group(self.sp_group)

            dp_groups = [dist.new_group(ranks=[g * sp_size + k for g in range(dp_size)])
                         for k in range(sp_size)]
            self.dp_group = dp_groups[global_rank % sp_size]
            set_data_parallel_group(self.dp_group)

            self.sp_rank = global_rank % sp_size
            self.dp_rank = global_rank // sp_size
            if self.is_main_process:
                print(f"[CameraBiDiff][SP] enabled sp_size={sp_size} "
                      f"dp_size={dp_size} world_size={world_size}")

        total_batch_size = getattr(config, "total_batch_size", None)
        micro_global_batch = int(config.batch_size) * int(self.data_parallel_size)
        configured_accumulation = getattr(config, "gradient_accumulation_steps", None)
        if configured_accumulation is None:
            if total_batch_size is not None:
                if int(total_batch_size) % micro_global_batch != 0:
                    raise ValueError(
                        f"total_batch_size={total_batch_size} must be divisible "
                        f"by batch_size*world_size={micro_global_batch}."
                    )
                self.gradient_accumulation_steps = max(
                    1, int(total_batch_size) // micro_global_batch)
            else:
                self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = int(configured_accumulation)
            if total_batch_size is not None:
                effective = micro_global_batch * self.gradient_accumulation_steps
                if int(total_batch_size) != effective:
                    raise ValueError(
                        f"total_batch_size={total_batch_size} but "
                        f"batch_size*world_size*gradient_accumulation_steps={effective}."
                    )
        if self.is_main_process and self.gradient_accumulation_steps > 1:
            eff_batch = micro_global_batch * self.gradient_accumulation_steps
            print(
                f"[CameraBiDiff] gradient_accumulation_steps="
                f"{self.gradient_accumulation_steps}, effective batch size={eff_batch}"
            )

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        # SP peers (same dp_rank, different sp_rank) MUST share the RNG seed so
        # that the per-sample uniform timestep drawn inside generator_loss is
        # identical across the group; otherwise the frame chunks would be
        # denoised at inconsistent noise levels.
        set_seed(config.seed + self.dp_rank)

        if self.is_main_process and not self.disable_wandb:
            if getattr(config, "wandb_key", None):
                wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=getattr(config, "wandb_entity", None),
                project=getattr(config, "wandb_project", "LongLive-CameraSFT"),
                dir=config.wandb_save_dir,
            )

        self.output_path = config.logdir
        if self.is_main_process and self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

        # ---- Model ----
        self.model = CameraBidirectionalDiffusion(config, device=self.device)
        # Move VAE to device (kept unsharded; only used under no_grad).
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        # Bind the SP self-attention before FSDP wrapping. The rest of
        # ``WanModel._forward`` already runs on the local frame chunk, so only
        # the self-attention needs an Ulysses all-to-all around flash_attention.
        if self.sequence_parallel_size > 1:
            from wan_5b.distributed.sequence_parallel_camera import (
                sp_camera_attn_forward,
            )
            wan_model = self.model.generator.model
            self._sp_attn_blocks = []
            for block in wan_model.blocks:
                sa = block.self_attn
                if not hasattr(sa, "_orig_forward"):
                    sa._orig_forward = sa.forward
                sa.forward = types.MethodType(sp_camera_attn_forward, sa)
                self._sp_attn_blocks.append(sa)
            if self.is_main_process:
                print("[CameraBiDiff][SP] sp_camera_attn_forward enabled on "
                      f"{len(self._sp_attn_blocks)} self-attention blocks")

        # FSDP-wrap the generator (the only trainable module).
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=getattr(config, "sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "generator_fsdp_wrap_strategy", "size"),
        )

        # FSDP-wrap the (frozen) text encoder, mirroring minWM. This shards the
        # large umt5-xxl weights across ranks instead of replicating a full copy
        # per GPU, which matters for the umt5-xxl + 5B memory footprint. The
        # module is only ever called under torch.no_grad() during training.
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=getattr(config, "sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "text_encoder_fsdp_wrap_strategy", "size"),
        )

        # Optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.95)),
            weight_decay=getattr(config, "weight_decay", 0.01),
        )

        # EMA
        ema_weight = getattr(config, "ema_weight", 0.0)
        self.generator_ema = None
        if ema_weight is not None and ema_weight > 0.0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        self.ema_start_step = getattr(config, "ema_start_step", 0)

        # ---- Data ----
        # Under SP, all ranks in an SP group must load the SAME sample (they
        # process different frame chunks of it), so sampling is indexed by the
        # DP rank/size rather than the global rank/world size.
        configured_shape = list(config.image_or_video_shape)
        target_num_frames = int(configured_shape[1])
        expected_latent_shape = tuple(int(v) for v in configured_shape[2:])
        base_dataset = CameraLatentLMDBDataset(
            config.data_path,
            max_pair=int(1e8),
            target_num_frames=target_num_frames,
            expected_latent_shape=expected_latent_shape,
        )
        dataset_repeat = getattr(config, "dataset_repeat", None)
        # ``repeat``/``repeat_dataset`` are compatibility aliases.  They must
        # override the config default (dataset_repeat=1) when supplied via CLI.
        alias_repeat = getattr(config, "repeat_dataset", None)
        if alias_repeat is None:
            alias_repeat = getattr(config, "repeat", None)
        if alias_repeat is not None:
            if dataset_repeat not in (None, 1) and int(dataset_repeat) != int(alias_repeat):
                raise ValueError(
                    "Conflicting dataset repeat values: "
                    f"dataset_repeat={dataset_repeat}, alias={alias_repeat}."
                )
            dataset_repeat = alias_repeat
        if dataset_repeat is None:
            dataset_repeat = 1
        dataset_repeat = int(dataset_repeat)
        if dataset_repeat < 1:
            raise ValueError(f"dataset_repeat must be >= 1, got {dataset_repeat}.")
        dataset = (
            RepeatDataset(base_dataset, dataset_repeat)
            if dataset_repeat > 1 else base_dataset
        )
        sampler = DistributedSampler(dataset, num_replicas=self.data_parallel_size,
                                     rank=self.dp_rank, shuffle=True, drop_last=True)
        self.dataset = dataset
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        if self.is_main_process:
            if dataset_repeat > 1:
                print(f"[CameraBiDiff] dataset size = {len(dataset)} "
                      f"(base_size={len(base_dataset)}, repeat={dataset_repeat}, "
                      f"target_frames={target_num_frames})")
            else:
                print(f"[CameraBiDiff] dataset size = {len(dataset)} "
                      f"(target_frames={target_num_frames})")
        self.dataloader = cycle(loader)

        # ---- Optional resume / initialization ----
        auto_resume = getattr(config, "auto_resume", True)
        resume_ckpt = self._find_latest_checkpoint(self.output_path) if auto_resume else None
        init_ckpt = getattr(config, "generator_ckpt", None)
        if resume_ckpt is not None:
            self._load_checkpoint(resume_ckpt, resume_training=True)
        elif init_ckpt and os.path.exists(init_ckpt):
            self._load_checkpoint(init_ckpt, resume_training=False)
        elif init_ckpt and self.is_main_process:
            print(f"[CameraBiDiff] generator_ckpt not found: {init_ckpt}")

        self.unconditional_dict = None
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None
        self.gc_interval = getattr(config, "gc_interval", 100)
        self.log_interval = getattr(config, "log_interval", 10)
        # ---- I2V switch -----------------------------------------------------
        # ``algorithm.i2v: true`` (flattened by normalize_config to ``i2v``)
        # turns the same Bidirectional + camera (PRoPE) SFT loop into an I2V
        # SFT that pins the first latent frame to the source image's clean
        # latent and masks it out of the flow-matching loss. Defaults to T2V
        # (False) so existing configs keep their previous semantics.
        self.i2v = bool(getattr(config, "i2v", False))
        if self.is_main_process:
            print(f"[CameraBiDiff] i2v={self.i2v}")
        self.save_interval = getattr(config, "save_interval", 1000)
        self.max_iters = getattr(config, "max_iters", 100000)
        # 默认只保留最新的 N 个 checkpoint，<=0 表示不清理（保留全部）
        self.keep_last_n_checkpoints = int(getattr(config, "keep_last_n_checkpoints", 3))


    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_state_key(key: str) -> str:
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if key.startswith("model."):
            # Unwrapped inference checkpoints may store the inner Wan model only.
            # FSDP-wrapped CameraWanDiffusionWrapper checkpoints keep this prefix.
            pass
        return (key.replace("_fsdp_wrapped_module.", "")
                   .replace("_checkpoint_wrapped_module.", "")
                   .replace("_orig_mod.", ""))

    @classmethod
    def _extract_generator_state_dict(cls, raw_state):
        if isinstance(raw_state, dict):
            if "generator" in raw_state and isinstance(raw_state["generator"], dict):
                raw_state = raw_state["generator"]
            elif "model" in raw_state and isinstance(raw_state["model"], dict):
                raw_state = raw_state["model"]
        if not isinstance(raw_state, dict):
            raise TypeError("checkpoint does not contain a state dict")
        return {cls._clean_state_key(k): v for k, v in raw_state.items()}

    @staticmethod
    def _find_latest_checkpoint(logdir):
        if not logdir or not os.path.isdir(logdir):
            return None
        candidates = []
        pat = re.compile(r"checkpoint_model_(\d+)$")
        for name in os.listdir(logdir):
            match = pat.match(name)
            if not match:
                continue
            path = os.path.join(logdir, name, "model.pt")
            if os.path.isfile(path):
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _step_from_checkpoint_path(path):
        match = re.search(r"checkpoint_model_(\d+)", path or "")
        return int(match.group(1)) if match else 0

    def _load_checkpoint(self, ckpt_path, resume_training: bool):
        if self.is_main_process:
            mode = "resuming" if resume_training else "initializing"
            print(f"[CameraBiDiff] {mode} from {ckpt_path}")
        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gen_sd = self._extract_generator_state_dict(raw_state)
        missing, unexpected = self.model.generator.load_state_dict(gen_sd, strict=False)
        if self.is_main_process:
            print(f"[CameraBiDiff] load generator: missing={len(missing)} unexpected={len(unexpected)}")
            if unexpected:
                print(f"[CameraBiDiff] unexpected keys (first 5): {unexpected[:5]}")

        if not resume_training:
            return

        self.step = int(raw_state.get("step", self._step_from_checkpoint_path(ckpt_path)))
        if "generator_optimizer" in raw_state:
            optim_sd = FSDP.optim_state_dict_to_load(
                self.model.generator, self.generator_optimizer,
                raw_state["generator_optimizer"],
            )
            self.generator_optimizer.load_state_dict(optim_sd)
            if self.is_main_process:
                print("[CameraBiDiff] resumed generator optimizer")
        elif self.is_main_process:
            print("[CameraBiDiff] optimizer state missing; continuing with a fresh optimizer")

        ema_state = raw_state.get("generator_ema", None)
        ema_path = os.path.join(os.path.dirname(ckpt_path), "model_ema.pt")
        if ema_state is None and os.path.exists(ema_path):
            ema_raw = torch.load(ema_path, map_location="cpu", weights_only=False)
            ema_state = ema_raw.get("generator_ema", ema_raw)
        if self.generator_ema is not None and ema_state is not None:
            self.generator_ema.load_state_dict(ema_state)
            if self.is_main_process:
                print("[CameraBiDiff] resumed generator EMA")

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(self):
        while self.step < self.max_iters:
            t0 = time.time()
            log = None
            for accumulation_step in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                log = self.train_one_step(
                    batch, accumulation_step=accumulation_step,
                    accumulation_steps=self.gradient_accumulation_steps)
            elapsed = time.time() - t0

            if (self.step % self.log_interval == 0) and self.is_main_process:
                now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                msg = (f"[step {self.step}] loss={log['generator_loss']:.6e} "
                       f"grad={log['generator_grad_norm']:.3e} "
                       f"t/it={elapsed:.2f}s "
                       f"time={now_str}")
                print(msg, flush=True)
                if not self.disable_wandb:
                    wandb.log(log, step=self.step)

            if (self.step > 0 and self.step % self.save_interval == 0
                    and not getattr(self.config, "no_save", False)):
                self.save()

            if self.step % self.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if not getattr(self.config, "no_save", False):
            self.save()

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=1):
        self.model.generator.train()

        text_prompts = batch["prompts"]
        if isinstance(text_prompts, list):
            pass
        else:
            # tensor of strings is unusual; ensure list[str]
            text_prompts = list(text_prompts)

        clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        viewmats = batch["viewmats"].to(device=self.device, dtype=self.dtype)
        Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)

        # ---- Sequence Parallel: shard the latent frame dimension ----
        # Each SP rank keeps frames [sp_rank*F_local, (sp_rank+1)*F_local). The
        # per-frame view matrices / intrinsics are sharded the same way so the
        # local PRoPE cameras stay aligned with the local token sequence. The
        # generator then computes noise / target / loss purely on local frames
        # (local ``.mean()``), and FSDP-over-world averaging reproduces the
        # exact data-parallel gradient.
        if self.sequence_parallel_size > 1:
            sp = self.sequence_parallel_size
            assert clean_latent.shape[1] % sp == 0, (
                f"latent frames ({clean_latent.shape[1]}) must be divisible "
                f"by sequence_parallel_size ({sp})")
            clean_latent = clean_latent.chunk(sp, dim=1)[self.sp_rank].contiguous()
            viewmats = viewmats.chunk(sp, dim=1)[self.sp_rank].contiguous()
            Ks = Ks.chunk(sp, dim=1)[self.sp_rank].contiguous()

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        # Local frame count (== full count when SP is disabled).
        image_or_video_shape[1] = clean_latent.shape[1]

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if self.unconditional_dict is None:
                ud = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                self.unconditional_dict = {k: v.detach() for k, v in ud.items()}

        loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=self.unconditional_dict,
            clean_latent=clean_latent,
            # I2V: pin the global first latent on the SP rank that actually
            # owns it (sp_rank == 0). On all other SP ranks, leave
            # initial_latent=None so the loss treats their local frame 0 as a
            # normal training frame. T2V mode never sets a context frame.
            initial_latent=(
                clean_latent[:, 0:1]
                if self.i2v and self.sp_rank == 0
                else None
            ),
            viewmats=viewmats,
            Ks=Ks,
        )

        if accumulation_step == 0:
            self.generator_optimizer.zero_grad(set_to_none=True)
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if accumulation_step != accumulation_steps - 1:
            return None

        grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        if self.generator_ema is not None and self.step >= self.ema_start_step:
            self.generator_ema.update(self.model.generator)

        self.step += 1
        return {
            "generator_loss": float(loss.detach().cpu()),
            "generator_grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self):
        save_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        barrier()
        gen_sd = fsdp_state_dict(self.model.generator)
        gen_optim_sd = FSDP.optim_state_dict(
            self.model.generator, self.generator_optimizer)
        ema_sd = (self.generator_ema.state_dict()
                  if self.generator_ema is not None else None)
        if self.is_main_process:
            payload = {
                "step": self.step,
                "generator": gen_sd,
                "generator_optimizer": gen_optim_sd,
            }
            if ema_sd is not None:
                payload["generator_ema"] = ema_sd
            torch.save(payload, os.path.join(save_dir, "model.pt"))
            if ema_sd is not None:
                torch.save({"generator_ema": ema_sd},
                           os.path.join(save_dir, "model_ema.pt"))
            print(f"[CameraBiDiff] saved checkpoint to {save_dir}")
            self._prune_old_checkpoints()
        barrier()

    def _prune_old_checkpoints(self):
        """只保留最新的 ``keep_last_n_checkpoints`` 个 checkpoint 目录，删除更旧的。

        仅 main process 调用。``keep_last_n_checkpoints <= 0`` 时不做清理。
        """
        keep = int(getattr(self, "keep_last_n_checkpoints", 0) or 0)
        if keep <= 0:
            return
        logdir = self.output_path
        if not logdir or not os.path.isdir(logdir):
            return
        pat = re.compile(r"checkpoint_model_(\d+)$")
        candidates = []
        for name in os.listdir(logdir):
            match = pat.match(name)
            if not match:
                continue
            full = os.path.join(logdir, name)
            if not os.path.isdir(full):
                continue
            candidates.append((int(match.group(1)), full))
        if len(candidates) <= keep:
            return
        # 按 step 升序排序，删除最早的那些
        candidates.sort(key=lambda item: item[0])
        to_remove = candidates[:-keep]
        import shutil
        for step, path in to_remove:
            try:
                shutil.rmtree(path)
                print(f"[CameraBiDiff] pruned old checkpoint: {path}")
            except Exception as e:
                print(f"[CameraBiDiff] failed to prune {path}: {e}")
