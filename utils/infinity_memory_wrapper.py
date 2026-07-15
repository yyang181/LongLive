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


def _normalize_memory_kwargs(memory_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize deprecated Echo/LongLive aliases before encoder creation."""
    normalized = dict(memory_kwargs or {})
    aliases = {
        "dim": "hidden_dim",
        "num_layers": "n_encoder_layers",
        "init_scale": "initializer_range",
    }
    for src, dst in aliases.items():
        if src in normalized and dst in normalized:
            if normalized[src] != normalized[dst]:
                raise ValueError(
                    f"memory_kwargs contains both deprecated '{src}'={normalized[src]} "
                    f"and '{dst}'={normalized[dst]} with different values. "
                    f"Remove '{src}' and keep '{dst}'."
                )
        elif src in normalized:
            normalized[dst] = normalized[src]
        normalized.pop(src, None)
    return normalized


def _make_encoder_config(memory_kwargs: Dict[str, Any]) -> _EncoderConfig:
    """Build the encoder-side config with Echo-Infinity-like safe defaults.

    Wan2.2-TI2V-5B uses 880 visual tokens per latent frame at 1280x704
    (22 * 40). The memory encoder defaults follow Echo-Infinity's public
    configuration scale (1536 hidden / 12 heads / 2 layers) rather than the
    full Wan-5B hidden size, with KV head adaptation handled at runtime.
    """

    defaults: Dict[str, Any] = {
        # --- structural (Echo-Infinity public scale; adapted to Wan-5B KV) ---
        "hidden_dim": 1536,
        "num_heads": 12,
        "n_encoder_layers": 2,
        "head_dim": 128,
        "tokens_per_frame": 880,          # 22 * 40 for 1280x704 latent
        # --- memory sizing ---
        "Q_frames": 3,
        "M_tokens_per_frame": 880,
        "num_query_groups": 1,
        # --- optimisation ---
        "bptt_clips": 1,
        "encoder_lr_multiplier": 5.0,
        # --- feature toggles ---
        "use_sink_anchor": False,
        "use_batch_update": False,
        "use_residual_update": True,
        "use_post_norm": True,
        "use_vib": False,
        "vib_kl_weight": 1.0e-4,
        # --- init ---
        "initializer_range": 0.014,
    }
    normalized = _normalize_memory_kwargs(memory_kwargs)
    defaults.update(normalized)
    if "M_tokens_per_frame" not in normalized:
        defaults["M_tokens_per_frame"] = defaults["tokens_per_frame"]
    return _EncoderConfig(**defaults)


def _estimate_query_memory_encoder_params(cfg: _EncoderConfig) -> int:
    """Estimate QueryMemoryEncoder parameter count before instantiation."""
    hidden_dim = int(getattr(cfg, "hidden_dim"))
    num_heads = int(getattr(cfg, "num_heads"))
    head_dim = int(getattr(cfg, "head_dim"))
    n_encoder_layers = int(getattr(cfg, "n_encoder_layers"))
    q_frames = int(getattr(cfg, "Q_frames"))
    m_tokens_per_frame = int(getattr(cfg, "M_tokens_per_frame"))
    num_query_groups = int(getattr(cfg, "num_query_groups", 1))
    qk_norm = bool(getattr(cfg, "qk_norm", True))
    use_post_norm = bool(getattr(cfg, "use_post_norm", False))
    use_vib = bool(getattr(cfg, "use_vib", False))
    normalize_memory_k = bool(getattr(cfg, "normalize_memory_k", False))

    memory_tokens = q_frames * m_tokens_per_frame
    kv_out_dim = num_heads * head_dim
    ffn_dim = hidden_dim * 4

    # MemoryCrossAttentionLayer: q/o, q/norm1/norm2, and 2-layer FFN.
    layer_params = 0
    layer_params += hidden_dim * hidden_dim + hidden_dim  # q
    layer_params += hidden_dim * hidden_dim + hidden_dim  # o
    layer_params += hidden_dim if qk_norm else 0          # norm_q
    layer_params += hidden_dim                            # norm1
    layer_params += hidden_dim                            # norm2
    layer_params += hidden_dim * ffn_dim + ffn_dim        # ffn.0
    layer_params += ffn_dim * hidden_dim + hidden_dim     # ffn.2
    total = n_encoder_layers * layer_params

    # Per query group parameters.
    per_group = 0
    per_group += memory_tokens * hidden_dim               # query_init
    per_group += hidden_dim * hidden_dim + hidden_dim     # connector_proj.0
    per_group += hidden_dim * hidden_dim + hidden_dim     # connector_proj.2
    per_group += hidden_dim                               # connector_proj RMSNorm
    per_group += (hidden_dim * 2) * hidden_dim + hidden_dim  # gate_linear
    per_group += hidden_dim * kv_out_dim + kv_out_dim     # to_k
    per_group += hidden_dim * kv_out_dim + kv_out_dim     # to_v
    per_group += kv_out_dim if normalize_memory_k else 0
    total += num_query_groups * per_group

    if use_post_norm:
        total += hidden_dim
    if use_vib:
        total += 2 * (hidden_dim * hidden_dim + hidden_dim)
    return int(total)


def _format_encoder_hparams(cfg: _EncoderConfig) -> str:
    keys = (
        "hidden_dim", "num_heads", "head_dim", "n_encoder_layers",
        "tokens_per_frame", "Q_frames", "M_tokens_per_frame",
        "num_query_groups", "bptt_clips", "encoder_lr_multiplier",
        "use_sink_anchor", "use_batch_update", "use_residual_update",
        "use_post_norm", "use_vib", "initializer_range",
    )
    return ", ".join(f"{k}={getattr(cfg, k, None)}" for k in keys)


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
        estimated_params = _estimate_query_memory_encoder_params(enc_cfg)
        _MAX_ENCODER_PARAMS = 300_000_000
        hparam_str = _format_encoder_hparams(enc_cfg)
        if estimated_params > _MAX_ENCODER_PARAMS:
            raise ValueError(
                f"Estimated QueryMemoryEncoder parameter count "
                f"({estimated_params:,}) exceeds the maximum allowed "
                f"({_MAX_ENCODER_PARAMS:,}) before construction. "
                f"memory_kwargs: {hparam_str}"
            )

        print(
            f"[InfMem] QueryMemoryEncoder config accepted: "
            f"estimated_params={estimated_params:,} "
            f"({estimated_params / 1e6:.2f}M), {hparam_str}",
            flush=True,
        )
        encoder = QueryMemoryEncoder(enc_cfg)
        n_params = sum(p.numel() for p in encoder.parameters())
        print(
            f"[InfMem] QueryMemoryEncoder created: "
            f"params={n_params:,} ({n_params / 1e6:.2f}M), "
            f"hidden_dim={getattr(enc_cfg, 'hidden_dim', None)}, "
            f"n_encoder_layers={getattr(enc_cfg, 'n_encoder_layers', None)}, "
            f"Q_frames={getattr(enc_cfg, 'Q_frames', None)}, "
            f"M_tokens_per_frame={getattr(enc_cfg, 'M_tokens_per_frame', None)}",
            flush=True,
        )

        # Keep the encoder outside FSDP, but do not force FP32 here. The
        # trainer moves/casts it to the runtime dtype following Echo-Infinity.
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
