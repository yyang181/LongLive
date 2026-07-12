"""
Utility hooks for Echo-Infinity memory in LongLive Wan2.2-TI2V-5B.

These are the *only* places downstream code (trainer / inference pipeline) is
expected to touch the encoder directly. Every helper is a no-op when the
encoder is absent, so callers can invoke them unconditionally.

The FSDP-aware model resolution mirrors
``Echo-Infinity/model/streaming_training.py::reset_state``.
"""

from __future__ import annotations

from typing import Any, Optional, List

import torch
import torch.distributed as dist


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
    force_cast: bool = False,
) -> bool:
    """Move the out-of-FSDP memory encoder to the runtime device.

    By default the encoder stays in FP32 (only device is moved). Pass
    ``force_cast=True`` to also cast dtypes — this should only be used in
    inference paths where the entire model is cast to bf16.
    """
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return False
    if device is not None:
        encoder.to(device=device)
    if force_cast and dtype is not None:
        encoder.to(dtype=dtype)
    elif dtype is not None and not force_cast:
        # Even without force_cast, move to device but keep FP32 params.
        # The caller may pass dtype for autocast context but we don't cast.
        pass
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
    # Encoder parameters stay FP32; only the runtime state (query_state) is
    # cast to the requested dtype for compute.
    encoder.reset(batch_size=batch_size, device=device, dtype=dtype)
    # ``_ei_prev_window_start`` gates the "did we roll the window?" logic in
    # ``_self_attn_infmem_forward``; None means "this is chunk 0".
    if inner is not None:
        object.__setattr__(inner, "_ei_prev_window_start", None)
        object.__setattr__(inner, "_ei_total_evicted_frames", 0)
    return True


def maybe_detach_infmem(
    generator: Any,
    chunk_count: int,
    kv_cache: Optional[List[dict]] = None,
    crossattn_cache: Optional[List[dict]] = None,
) -> bool:
    """Truncated BPTT: detach encoder state every ``bptt_clips`` chunks.

    ``chunk_count`` is the caller-maintained running count of chunks generated
    since the last :func:`reset_infmem`. When ``chunk_count`` is a positive
    multiple of ``encoder.bptt_clips``, the encoder's hidden state AND the KV
    cache tensors are detached to cut the autograd graph.

    KV cache integer position metadata (``global_end_index``, ``local_end_index``,
    ``pinned_start``, ``pinned_len``) are NOT detached — they are plain
    integer tensors that don't carry autograd history.
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
    # Detach KV cache tensors to cut the autograd graph — only the actual
    # k/v data tensors, not the scalar index tensors.
    if kv_cache is not None:
        for cache in kv_cache:
            k = cache.get("k", None)
            v = cache.get("v", None)
            if k is not None and isinstance(k, torch.Tensor):
                cache["k"] = k.detach()
            if v is not None and isinstance(v, torch.Tensor):
                cache["v"] = v.detach()
    if crossattn_cache is not None:
        for cache in crossattn_cache:
            k = cache.get("k", None)
            v = cache.get("v", None)
            if k is not None and isinstance(k, torch.Tensor):
                cache["k"] = k.detach()
            if v is not None and isinstance(v, torch.Tensor):
                cache["v"] = v.detach()
    return True


def sync_infmem_gradients(
    generator: Any,
    dp_world_size: int = 1,
) -> None:
    """All-reduce encoder gradients across data-parallel ranks.

    Must be called ONLY on the last micro-step of gradient accumulation.

    Handles ``grad is None`` by creating zero gradients on ranks that didn't
    receive any. Detects NaN/Inf and raises RuntimeError.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return

    for p in encoder.parameters():
        if p.requires_grad:
            if p.grad is None:
                # Create a zero gradient so all ranks can participate in the
                # all-reduce.
                p.grad = torch.zeros_like(p)
            else:
                # Clone to avoid in-place issues after all-reduce.
                p.grad = p.grad.clone()

            # All-reduce (SUM).
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

            # Average.
            if dp_world_size > 1:
                p.grad.div_(dp_world_size)

            # Check for NaN / Inf.
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                raise RuntimeError(
                    f"NaN/Inf detected in encoder gradient after sync. "
                    f"Parameter shape={tuple(p.shape)}"
                )


def clip_infmem_grad_norm(
    generator: Any,
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients of the encoder parameters only.

    Returns the total norm (before clipping) as a scalar tensor.
    """
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return torch.tensor(0.0)

    params = [p for p in encoder.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params]),
        2,
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in params:
        p.grad.detach().mul_(clip_coef_clamped)
    return total_norm


def broadcast_infmem_params(generator: Any) -> None:
    """Broadcast encoder parameters + buffers from rank 0 to all ranks.

    Also compares checksums across ranks after broadcast.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return

    # Broadcast all parameters.
    for p in encoder.parameters():
        dist.broadcast(p.data, src=0)

    # Broadcast all buffers.
    for buf in encoder.buffers():
        dist.broadcast(buf.data, src=0)

    # Checksum comparison.
    rank = dist.get_rank()
    checksum = sum(p.data.sum().item() for p in encoder.parameters())
    checksum_tensor = torch.tensor(checksum, device=f"cuda:{rank}")
    dist.all_reduce(checksum_tensor, op=dist.ReduceOp.SUM)
    # If any rank disagrees, all_reduce(sum) would differ from
    # world_size * local_checksum. We can't directly compare, but we log.
    expected = checksum * dist.get_world_size()
    if abs(checksum_tensor.item() - expected) > 1.0:
        print(
            f"[InfMem][WARN] rank {rank}: encoder param checksum mismatch "
            f"after broadcast (local={checksum:.2f}, "
            f"all_reduce_sum={checksum_tensor.item():.2f}, "
            f"expected={expected:.2f})",
            flush=True,
        )


def infmem_extra_param_groups(
    generator: Any,
    base_lr: float,
) -> list:
    """Build the optimizer param group(s) for the memory encoder.

    .. deprecated::
        This function is kept for backward compatibility with the old
        ``InfMemWanDiffusionWrapper`` path that put encoder params into the
        generator optimizer. The combined DreamX+InfMem path uses a
        **separate** optimizer instead. This function returns an empty list
        when called by the combined trainer.

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
            "name": "query_memory_encoder",
        }
    ]
