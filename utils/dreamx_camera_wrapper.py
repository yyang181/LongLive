# SPDX-License-Identifier: Apache-2.0
"""DreamX-World style camera-aware Wan2.2-TI2V-5B diffusion wrapper.

Differs from ``CameraWanDiffusionWrapper`` (the legacy LongLive PRoPE-residual
path) in exactly two places:

1. The PRoPE branch is materialized as a *full* ``cam_self_attn`` submodule
   per WanAttentionBlock (DreamX layout), not a single zero-init ``prope_o``.
   Submodule keys match ``blocks.{i}.cam_self_attn.{q_proj,k_proj,v_proj,
   out_proj}.{weight,bias}`` so the official ``DreamX-World-5B-Cam``
   safetensors load 1:1.

2. Optional helper to load DreamX's pretrained safetensors checkpoint and to
   freeze the backbone for "train-camera-only" fine-tuning (matching the
   DreamX recipe).

Forward pass and timestep handling are inherited verbatim from
``CameraWanDiffusionWrapper``; the ``viewmats`` / ``Ks`` plumbing through
``WanModel.forward`` already produces the ``prope_meta`` dict that the
patched ``WanAttentionBlock.forward`` consumes via the ``cam_self_attn``
branch.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Optional

import torch

from utils.wan_5b_camera_wrapper import CameraWanDiffusionWrapper
from utils.wan_5b_wrapper import WanDiffusionWrapper


_BACKBONE_FREEZE_PREFIXES_KEEP = ("cam_self_attn",)


class DreamXCameraWanDiffusionWrapper(CameraWanDiffusionWrapper):
    """Camera-aware wrapper that uses DreamX's full ``PropeSelfAttention``
    branch instead of the LongLive ``prope_o`` residual.

    Args (only DreamX-specific ones documented; rest see base class):
        attn_compress: int. The DreamX 5B-Cam config uses ``attn_compress=1``
            (cam attn dim equals model dim). Use ``2``/``4`` to mirror their
            ablations.
        qk_norm:       Whether to RMSNorm Q/K inside ``cam_self_attn``
            (DreamX default: True).
        dreamx_ckpt:   Optional path to a DreamX safetensors directory
            (e.g. ``/path/to/DreamX-World-5B-Cam``). If given, both the
            backbone and ``cam_self_attn`` weights are loaded from it,
            replacing the Wan2.2-TI2V-5B base weights.
        freeze_backbone_for_train: When True (typical DreamX fine-tune
            recipe), set ``requires_grad=False`` on every parameter except
            the ``cam_self_attn`` submodules. The forward pass is unaffected.
    """

    def __init__(
        self,
        model_name: str = "Wan2.2-TI2V-5B",
        timestep_shift: float = 5.0,
        is_causal: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        t_scale: float = 1.0,
        rope_method: str = "linear",
        original_seq_len=None,
        attn_compress: int = 1,
        qk_norm: bool = True,
        dreamx_ckpt: Optional[str] = None,
        freeze_backbone_for_train: bool = False,
    ):
        # Bidirectional DreamX uses CameraWanDiffusionWrapper with the legacy
        # PRoPE registration disabled; causal AR must bypass that wrapper
        # because it intentionally asserts ``is_causal=False``.
        common_kwargs = dict(
            model_name=model_name,
            timestep_shift=timestep_shift,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            t_scale=t_scale,
            rope_method=rope_method,
            original_seq_len=original_seq_len,
        )
        if is_causal:
            WanDiffusionWrapper.__init__(self, **common_kwargs)
        else:
            # Skip CameraWanDiffusionWrapper's ``add_prope_parameters`` call by
            # passing ``use_camera=False``; DreamX installs cam_self_attn below.
            CameraWanDiffusionWrapper.__init__(self, **common_kwargs, use_camera=False)
        self.use_camera = True
        self.dreamx_camera = True
        self._is_causal = is_causal

        # Resolve attn_dim / num_heads from the model itself so that we don't
        # rely on any parameter not present in the LongLive WanModel config.
        first_block = self.model.blocks[0]
        block_dim = first_block.self_attn.dim
        block_heads = first_block.self_attn.num_heads
        assert block_dim % attn_compress == 0, (
            f"attn_compress={attn_compress} must divide model dim={block_dim}"
        )
        assert block_heads % attn_compress == 0, (
            f"attn_compress={attn_compress} must divide num_heads={block_heads}"
        )
        attn_dim = block_dim // attn_compress
        n_heads = block_heads // attn_compress

        from wan_5b.modules.dreamx_camera import add_dreamx_cam_self_attn
        n_added = add_dreamx_cam_self_attn(
            self.model,
            attn_dim=attn_dim,
            num_heads=n_heads,
            qk_norm=qk_norm,
        )
        self._log(
            f"[DreamXCameraWanDiffusionWrapper] cam_self_attn added to "
            f"{n_added} blocks (attn_dim={attn_dim}, num_heads={n_heads})."
        )

        if dreamx_ckpt is not None:
            self.load_dreamx_checkpoint(dreamx_ckpt)

        if freeze_backbone_for_train:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------
    # When is_causal=True (AR path), we must use WanDiffusionWrapper.forward
    # (which supports teacher forcing / clean_x / kv_cache). The bidirectional
    # CameraWanDiffusionWrapper.forward asserts clean_x is None and is only
    # correct for the non-causal path.
    _is_causal = False

    def forward(self, *args, **kwargs):
        if self._is_causal:
            return WanDiffusionWrapper.forward(self, *args, **kwargs)
        return CameraWanDiffusionWrapper.forward(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log(msg: str) -> None:
        try:
            import torch.distributed as _dist
            if (not _dist.is_initialized()) or _dist.get_rank() == 0:
                print(msg)
        except Exception:
            print(msg)

    def load_dreamx_checkpoint(self, ckpt_dir: str) -> None:
        """Load all safetensors shards under ``ckpt_dir`` into ``self.model``.

        ``ckpt_dir`` is expected to look like the official
        ``DreamX-World-5B-Cam`` snapshot:

            ckpt_dir/
                config.json
                diffusion_pytorch_model-00001-of-00003.safetensors
                diffusion_pytorch_model-00002-of-00003.safetensors
                diffusion_pytorch_model-00003-of-00003.safetensors
                diffusion_pytorch_model.safetensors.index.json

        Keys are bare (``blocks.0.cam_self_attn.q_proj.weight`` etc.), so they
        load directly into ``self.model.state_dict()`` once
        ``add_dreamx_cam_self_attn`` has run.
        """
        from safetensors.torch import load_file as _load_st

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"DreamX checkpoint directory not found: {ckpt_dir}"
            )
        shard_files = sorted(glob(os.path.join(ckpt_dir, "*.safetensors")))
        if not shard_files:
            # Fall back to a single-file checkpoint.
            single = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")
            if os.path.isfile(single):
                shard_files = [single]
        if not shard_files:
            raise FileNotFoundError(
                f"No .safetensors shards found in {ckpt_dir}"
            )

        merged: dict = {}
        for shard in shard_files:
            sd = _load_st(shard, device="cpu")
            merged.update(sd)

        missing, unexpected = self.model.load_state_dict(merged, strict=False)
        # Keys in the LongLive WanModel that are *not* in the DreamX ckpt
        # (e.g. transient buffers like rotary frequency tables). These are
        # expected and harmless — log only the structural surprises.
        struct_missing = [
            k for k in missing
            if not k.endswith(".freqs") and "rope" not in k.lower()
        ]
        if struct_missing:
            self._log(
                f"[DreamXCameraWanDiffusionWrapper] missing keys "
                f"({len(struct_missing)} of {len(missing)}): "
                f"{struct_missing[:5]}{'...' if len(struct_missing) > 5 else ''}"
            )
        if unexpected:
            self._log(
                f"[DreamXCameraWanDiffusionWrapper] unexpected keys "
                f"({len(unexpected)}): "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        self._log(
            f"[DreamXCameraWanDiffusionWrapper] loaded DreamX checkpoint "
            f"from {ckpt_dir} ({len(shard_files)} shard(s), "
            f"{len(merged)} tensors)."
        )

    def freeze_backbone(
        self,
        keep_prefixes: Iterable[str] = _BACKBONE_FREEZE_PREFIXES_KEEP,
    ) -> int:
        """Freeze every parameter whose name does not contain any of
        ``keep_prefixes``. Matches the DreamX fine-tune recipe (only
        ``cam_self_attn`` is trainable).

        Returns the number of trainable parameters left.
        """
        n_trainable = 0
        for name, p in self.model.named_parameters():
            keep = any(token in name for token in keep_prefixes)
            p.requires_grad = bool(keep)
            if keep:
                n_trainable += p.numel()
        self._log(
            f"[DreamXCameraWanDiffusionWrapper] backbone frozen, "
            f"{n_trainable / 1e6:.2f}M trainable params left "
            f"(keep prefixes: {tuple(keep_prefixes)})."
        )
        return n_trainable

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Convenience: returns the list of currently trainable parameters
        (post ``freeze_backbone`` if it was called)."""
        return [p for p in self.model.parameters() if p.requires_grad]
