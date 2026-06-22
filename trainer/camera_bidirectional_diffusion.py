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
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from model.camera_bidirectional_diffusion import CameraBidirectionalDiffusion
from utils.camera_dataset import CameraLatentLMDBDataset, cycle
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

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

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
        # Move VAE / TextEncoder modules
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        self.model.text_encoder = self.model.text_encoder.to(device=self.device)

        # FSDP-wrap the generator (the only trainable module).
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=getattr(config, "sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "generator_fsdp_wrap_strategy", "size"),
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
        dataset = CameraLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=global_rank, shuffle=True, drop_last=True)
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
            print(f"[CameraBiDiff] dataset size = {len(dataset)}")
        self.dataloader = cycle(loader)

        # ---- Optional resume ----
        ckpt_path = getattr(config, "generator_ckpt", None)
        if ckpt_path and os.path.exists(ckpt_path):
            if self.is_main_process:
                print(f"[CameraBiDiff] loading generator from {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("generator", sd.get("model", sd))
            fixed = {}
            for k, v in sd.items():
                if k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                fixed[k] = v
            try:
                self.model.generator.load_state_dict(fixed, strict=False)
            except Exception as e:
                if self.is_main_process:
                    print(f"[CameraBiDiff] resume warning: {e}")

        self.unconditional_dict = None
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None
        self.gc_interval = getattr(config, "gc_interval", 100)
        self.log_interval = getattr(config, "log_interval", 10)
        self.save_interval = getattr(config, "save_interval", 1000)
        self.max_iters = getattr(config, "max_iters", 100000)

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(self):
        while self.step < self.max_iters:
            batch = next(self.dataloader)
            t0 = time.time()
            log = self.train_one_step(batch)
            elapsed = time.time() - t0

            if (self.step % self.log_interval == 0) and self.is_main_process:
                msg = (f"[step {self.step}] loss={log['generator_loss']:.4f} "
                       f"grad={log['generator_grad_norm']:.3f} "
                       f"t/it={elapsed:.2f}s")
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

    def train_one_step(self, batch):
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

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

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
            initial_latent=clean_latent[:, 0:1],
            viewmats=viewmats,
            Ks=Ks,
        )

        self.generator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        if self.is_main_process:
            torch.save({"generator": gen_sd}, os.path.join(save_dir, "model.pt"))
            if self.generator_ema is not None:
                torch.save({"generator_ema": self.generator_ema.state_dict()},
                           os.path.join(save_dir, "model_ema.pt"))
            print(f"[CameraBiDiff] saved checkpoint to {save_dir}")
        barrier()
