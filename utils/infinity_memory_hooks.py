"""
Utility hooks for Echo-Infinity memory in LongLive Wan2.2-TI2V-5B.

These are the *only* places downstream code (trainer / inference pipeline) is
expected to touch the encoder directly. Every helper is a no-op when the
encoder is absent, so callers can invoke them unconditionally.

The FSDP-aware model resolution mirrors
``Echo-Infinity/model/streaming_training.py::reset_state``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


# ----------------------------------------------------------------------
# FSDP-aware inner-model resolution
# ----------------------------------------------------------------------
def resolve_inner_wan_model(generator: Any) -> Optional[torch.nn.Module]:
    """Return the underlying Wan transformer regardless of FSDP wrapping.

    ``generator`` is expected to be either an
    :class:`InfMemWanDiffusionWrapper` (with ``.model``) or an FSDP-wrapped
    version thereof (``._fsdp_wrapped_module.model``). Returns ``None`` if no
    inner model is reachable.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP  # type: ignore
    except Exception:  # pragma: no cover - torch<2 or no dist
        _FSDP = tuple()

    inner: Optional[torch.nn.Module]
    if hasattr(generator, "_fsdp_wrapped_module"):
        # Outer FSDP wraps the wrapper (which owns .model).
        wrapper = generator._fsdp_wrapped_module
        m = getattr(wrapper, "model", None)
        if m is None:
            return None
        # In case FSDP was applied at model-level too.
        inner = m._fsdp_wrapped_module if isinstance(m, _FSDP) else m
    elif hasattr(generator, "model"):
        inner = generator.model
    else:
        return None

    # Some ablations put base_model.model one level deeper.
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
    return inner


# ----------------------------------------------------------------------
# Public helpers used by trainer / inference / config validators
# ----------------------------------------------------------------------
def get_infmem_encoder(generator: Any) -> Optional[torch.nn.Module]:
    """Return the ``QueryMemoryEncoder`` attached to ``generator`` or None."""
    inner = resolve_inner_wan_model(generator)
    if inner is not None:
        encoder = getattr(inner, "query_memory_encoder", None)
        if encoder is not None:
            return encoder
    wrapper = getattr(generator, "_fsdp_wrapped_module", generator)
    return getattr(wrapper, "query_memory_encoder", None)


def move_infmem_encoder(
    generator: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    """Move the out-of-FSDP memory encoder to the runtime device/dtype."""
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return False
    encoder.to(device=device, dtype=dtype)
    return True


def infmem_state_dict(generator: Any) -> Optional[dict]:
    """Return a CPU state_dict for the out-of-FSDP memory encoder."""
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return None
    return {k: v.detach().cpu() for k, v in encoder.state_dict().items()}


def load_infmem_state_dict(generator: Any, state: Any, strict: bool = True) -> bool:
    """Load the out-of-FSDP memory encoder state_dict if one is attached."""
    encoder = get_infmem_encoder(generator)
    if encoder is None or state is None:
        return False
    encoder.load_state_dict(state, strict=strict)
    return True


def reset_infmem(
    generator: Any,
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    """Reset the memory encoder state at the start of a new video/window.

    Returns True if a reset happened, False if no encoder is attached (so the
    caller can log the situation but does not have to branch).
    """
    inner = resolve_inner_wan_model(generator)
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return False
    encoder.reset(batch_size=batch_size, device=device, dtype=dtype)
    # ``_ei_prev_window_start`` gates the "did we roll the window?" logic in
    # ``_self_attn_infmem_forward``; None means "this is chunk 0".
    if inner is not None:
        object.__setattr__(inner, "_ei_prev_window_start", None)
    return True


def maybe_detach_infmem(generator: Any, chunk_count: int) -> bool:
    """Truncated BPTT: detach encoder state every ``bptt_clips`` chunks.

    ``chunk_count`` is the caller-maintained running count of chunks generated
    since the last :func:`reset_infmem`. When ``chunk_count`` is a positive
    multiple of ``encoder.bptt_clips``, the encoder's hidden state is
    detached and this function returns True.
    """
    if chunk_count <= 0:
        return False
    inner = resolve_inner_wan_model(generator)
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return False
    bptt_clips = int(getattr(encoder, "bptt_clips", 1) or 1)
    if bptt_clips <= 0 or chunk_count % bptt_clips != 0:
        return False
    encoder.detach_state()
    return True


def infmem_extra_param_groups(
    generator: Any,
    base_lr: float,
) -> list:
    """Build the optimizer param group(s) for the memory encoder.

    Returns a (possibly empty) list of dicts ready to be appended to the main
    ``[{"params": ...}]`` list passed to ``torch.optim.AdamW``. The encoder
    uses its own multiplied LR (``encoder.encoder_lr_multiplier * base_lr``).
    """
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return []
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    if not trainable:
        return []
    lr_mult = float(getattr(encoder, "encoder_lr_multiplier", 1.0) or 1.0)
    return [
        {
            "params": trainable,
            "lr": base_lr * lr_mult,
            # A name is handy for logging / lr schedulers that inspect groups.
            "name": "query_memory_encoder",
        }
    ]
