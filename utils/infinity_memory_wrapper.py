"""
InfMemWanDiffusionWrapper
=========================

A drop-in replacement for :class:`utils.wan_5b_wrapper.WanDiffusionWrapper` that
plugs Echo-Infinity's ``QueryMemoryEncoder`` (memory KV distilled from evicted
KV pages) and Relative-RoPE self-attention into the LongLive Wan2.2-TI2V-5B
causal transformer.

Design goals
------------
1. Do **not** touch the upstream ``wan_5b/modules/causal_model.py`` code path.
   The Relative-RoPE + memory-cross-attn behaviour is added by monkey-patching
   each block's ``forward`` in :func:`wan_5b.modules.infinity_memory.attach_infmem`,
   so the base I2V AR training/inference code paths remain bit-exact when the
   feature is disabled.
2. The ``QueryMemoryEncoder`` is attached with ``object.__setattr__`` so that
   FSDP flatten-params logic does **not** treat it as a submodule of the wrapped
   Wan model. The encoder is optimised via a separate param group by the
   trainer.
3. The wrapper exposes ``self.model.query_memory_encoder`` (the encoder module)
   and ``self.encoder`` (alias) for the trainer / inference pipeline to reach
   without having to walk through FSDP wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from utils.wan_5b_wrapper import WanDiffusionWrapper
from wan_5b.modules.infinity_memory import attach_infmem
from model.query_memory import QueryMemoryEncoder


class _EncoderConfig:
    """Lightweight attribute-bag config accepted by ``QueryMemoryEncoder``.

    The Echo-Infinity encoder reads its hyperparameters via ``getattr`` on a
    config object, so we mirror the same shape here rather than requiring the
    trainer to pass an OmegaConf node."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_encoder_config(memory_kwargs: Dict[str, Any]) -> _EncoderConfig:
    """Build the encoder-side config with LongLive-5B safe defaults.

    Wan2.2-TI2V-5B has 30 blocks, dim=3072 and each frame corresponds to
    ``grid_h * grid_w = 22 * 40 = 880`` visual tokens (1280x704 latent at
    16x16 patch size). These override Echo-Infinity's Wan2.1 (1560) defaults."""

    defaults: Dict[str, Any] = {
        # --- structural (must match Wan2.2-TI2V-5B) ---
        "hidden_dim": 3072,
        "num_heads": 24,
        "n_encoder_layers": 30,
        "head_dim": 128,
        "tokens_per_frame": 880,          # 22 * 40 for 1280x704 latent
        # --- memory sizing ---
        "Q_frames": 8,
        "M_tokens_per_frame": 32,
        "num_query_groups": 1,
        # --- optimisation ---
        "bptt_clips": 2,
        "encoder_lr_multiplier": 5.0,
        # --- feature toggles ---
        "use_sink_anchor": True,
        "use_batch_update": False,
        "use_residual_update": True,
        "use_post_norm": True,
        "use_vib": False,
        "vib_kl_weight": 1.0e-4,
        # --- init ---
        "initializer_range": 0.02,
    }
    memory_kwargs = dict(memory_kwargs or {})
    # Echo-Infinity configs often use these names. QueryMemoryEncoder reads
    # the Wan-style names below, so normalize them before constructing it.
    aliases = {
        "dim": "hidden_dim",
        "num_layers": "n_encoder_layers",
        "init_scale": "initializer_range",
    }
    for src, dst in aliases.items():
        if src in memory_kwargs and dst not in memory_kwargs:
            memory_kwargs[dst] = memory_kwargs[src]
        memory_kwargs.pop(src, None)
    defaults.update(memory_kwargs)
    return _EncoderConfig(**defaults)


def _attach_infmem_to_wrapper(
    wrapper: Any,
    *,
    enable_relative_rope: bool = False,
    relative_rope_pmax: int = 24,
    memory_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach InfMem monkey-patches + QueryMemoryEncoder to *wrapper*.

    This is the shared init logic used by both:
      * :class:`InfMemWanDiffusionWrapper` (plain Wan I2V AR)
      * :class:`DreamXInfMemWanDiffusionWrapper` (DreamX Camera I2V AR)

    It must be called AFTER the base wrapper has created ``wrapper.model``
    (including any ``cam_self_attn`` submodules), so the monkey-patch is
    applied on top of the final model structure.

    The function is idempotent and safe to call on models that already have
    ``cam_self_attn`` (DreamX) — the InfMem patch only replaces
    ``block.forward`` / ``self_attn.forward`` / ``model._forward_inference``,
    and the patched block forward checks ``getattr(self, "cam_self_attn", None)``
    at runtime.
    """
    # Idempotent monkey-patch: block/self-attn/model.forward_inference
    # gain a memory_kv-aware branch. When enable_relative_rope=False and
    # query_memory_encoder is None, the patched forward falls back to the
    # original code path.
    attach_infmem(
        wrapper.model,
        relative_rope=enable_relative_rope,
        relative_rope_pmax=int(relative_rope_pmax),
        num_frame_per_block_attr=int(getattr(wrapper.model, "num_frame_per_block", 8)),
    )

    # Encoder: mounted via object.__setattr__ so FSDP does not pull the
    # encoder parameters into the wrapped model's flat_param buffer. The
    # trainer must recognise this attribute and build a separate
    # optimizer param group; see ``model.diffusion._initialize_models``.
    if memory_kwargs is not None:
        enc_cfg = _make_encoder_config(memory_kwargs)
        encoder = QueryMemoryEncoder(enc_cfg)
        object.__setattr__(wrapper.model, "query_memory_encoder", encoder)
        object.__setattr__(wrapper, "query_memory_encoder", encoder)
    else:
        object.__setattr__(wrapper.model, "query_memory_encoder", None)
        object.__setattr__(wrapper, "query_memory_encoder", None)

    # Reset the per-window bookkeeping used by the Relative-RoPE
    # heuristic in ``_self_attn_infmem_forward``.
    object.__setattr__(wrapper.model, "_ei_prev_window_start", None)


class InfMemWanDiffusionWrapper(WanDiffusionWrapper):
    """Wan2.2-TI2V-5B wrapper with Echo-Infinity memory + Relative RoPE.

    Extra kwargs (all optional, default-off):
        enable_relative_rope: bool, default False.
            When True, self-attention within each chunk applies the Relative
            RoPE layout (sink | memory | local | Q collapsed onto
            ``[0, pmax-1]``). When False, the wrapper still owns the encoder
            (if ``memory_kwargs`` is provided) but the attention path is a
            no-op relative to the base wrapper, so training/inference is
            numerically identical to :class:`WanDiffusionWrapper`.
        relative_rope_pmax: int, default 24.
            The window depth ``pmax`` used by ``_compute_relative_positions``.
            Must be >= number of frames spanned by ``sink + local + Q``.
        memory_kwargs: dict, default None.
            When provided, instantiates a ``QueryMemoryEncoder`` with these
            hyperparameters and attaches it as ``self.model.query_memory_encoder``.
            When None, no encoder is created (the block ``forward`` will treat
            ``memory_kv = None`` and skip the memory-cross-attn contribution).
    """

    def __init__(
        self,
        *args: Any,
        enable_relative_rope: bool = False,
        relative_rope_pmax: int = 24,
        memory_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        _attach_infmem_to_wrapper(
            self,
            enable_relative_rope=enable_relative_rope,
            relative_rope_pmax=relative_rope_pmax,
            memory_kwargs=memory_kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def encoder(self) -> Optional[torch.nn.Module]:
        """Return the ``QueryMemoryEncoder`` or ``None`` if disabled."""
        return getattr(self.model, "query_memory_encoder", None)

    def reset_memory(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Convenience alias for tests / notebooks; equivalent to calling
        :func:`utils.infinity_memory_hooks.reset_infmem` on ``self.model``."""
        enc = self.encoder
        if enc is None:
            return
        enc.reset(batch_size=batch_size, device=device, dtype=dtype)
        object.__setattr__(self.model, "_ei_prev_window_start", None)
