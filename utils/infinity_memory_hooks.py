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
    *,
    group=None,
    average: bool = True,
    dp_world_size: Optional[int] = None,  # deprecated, kept for backward compat
) -> None:
    """All-reduce encoder gradients across data-parallel ranks.

    Must be called ONLY on the last micro-step of gradient accumulation.

    Two-phase protocol to avoid creating spurious zero gradients:
      * Phase 1: collect has-grad flags for every trainable parameter via a
        single all-reduce.
      * Phase 2: for each parameter, only all-reduce when at least one rank
        has a gradient. Parameters with no gradient on ANY rank keep
        ``p.grad is None`` so AdamW does not apply weight decay to them.
    """
    if not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return

    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return

    group_world_size = dist.get_world_size(group)
    # Backward-compat: if the caller still passes dp_world_size and no group,
    # honor it as the divisor. Otherwise use the actual group size.
    divisor = group_world_size
    if group is None and dp_world_size is not None and dp_world_size > 0:
        divisor = dp_world_size

    params = [p for p in encoder.parameters() if p.requires_grad]
    if not params:
        return

    # Determine device for the has-grad flag tensor. Use the encoder's own
    # parameter device (NOT cuda:{global_rank}) so multi-node is correct.
    ref_device = next(encoder.parameters()).device

    # Phase 1: collect has-grad flags.
    local_has_grad = torch.tensor(
        [1 if p.grad is not None else 0 for p in params],
        device=ref_device,
        dtype=torch.int32,
    )
    dist.all_reduce(local_has_grad, op=dist.ReduceOp.SUM, group=group)

    # Phase 2: per-parameter all-reduce.
    for index, p in enumerate(params):
        grad_rank_count = int(local_has_grad[index].item())
        if grad_rank_count == 0:
            # No rank has a gradient for this parameter — keep grad=None so
            # AdamW does NOT apply weight decay.
            p.grad = None
            continue
        if p.grad is None:
            # Some rank(s) have a gradient but this rank doesn't — contribute zero.
            p.grad = torch.zeros_like(p)
        else:
            p.grad = p.grad.clone()
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        if average and divisor > 1:
            p.grad.div_(divisor)
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError(
                f"NaN/Inf detected in encoder gradient after sync. "
                f"Parameter shape={tuple(p.shape)}"
            )


def clip_infmem_grad_norm(
    generator: Any,
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients of the encoder parameters only.

    Uses :func:`torch.nn.utils.clip_grad_norm_` for parity with the generator
    path. Returns the total norm (before clipping) as a scalar tensor.
    """
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return torch.tensor(0.0)

    params = [p for p in encoder.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    return total_norm


def recursive_to_cpu(obj):
    """Recursively detach + CPU-ize every tensor in a nested structure."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    return obj


def state_dict_stats(state_dict) -> dict:
    """Compute a stable multi-statistic summary of a state_dict for verification."""
    if state_dict is None:
        return {"n_tensors": 0, "total_sum": 0.0, "total_sq": 0.0, "max_abs": 0.0}
    keys = sorted(state_dict.keys())
    total_sum = 0.0
    total_sq = 0.0
    max_abs = 0.0
    n_tensors = 0
    for k in keys:
        v = state_dict[k]
        if torch.is_tensor(v):
            vf = v.detach().float()
            total_sum += vf.sum().item()
            total_sq += vf.pow(2).sum().item()
            max_abs = max(max_abs, vf.abs().max().item())
            n_tensors += 1
    return {
        "n_tensors": n_tensors,
        "total_sum": total_sum,
        "total_sq": total_sq,
        "max_abs": max_abs,
        "n_keys": len(keys),
    }


def verify_infmem_params(
    generator: Any,
    *,
    group=None,
) -> bool:
    """Verify all ranks have identical encoder parameters + buffers.

    Uses multi-statistic all-gather (sum, square-sum, max-abs, count) and
    raises RuntimeError if any rank disagrees.
    """
    if not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return True
    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return True

    ref_device = next(encoder.parameters()).device
    params = list(encoder.parameters())
    buffers = list(encoder.buffers())
    all_tensors = params + buffers
    if not all_tensors:
        return True

    total_sum = 0.0
    total_sq = 0.0
    max_abs = 0.0
    num = 0
    for t in all_tensors:
        tf = t.detach().float()
        total_sum += tf.sum().item()
        total_sq += tf.pow(2).sum().item()
        max_abs = max(max_abs, tf.abs().max().item())
        num += t.numel()
    local_stats = torch.tensor(
        [total_sum, total_sq, max_abs, float(num)],
        device=ref_device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local_stats) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local_stats, group=group)
    for r, st in enumerate(gathered):
        if not torch.allclose(st, local_stats):
            raise RuntimeError(
                f"QueryMemoryEncoder parameters differ across ranks after "
                f"broadcast/verify: rank0 stats={local_stats.tolist()} "
                f"rank{r} stats={st.tolist()}"
            )
    return True


def broadcast_infmem_params(
    generator: Any,
    *,
    src: int = 0,
    group=None,
    verify: bool = True,
) -> bool:
    """Broadcast encoder parameters + buffers from ``src`` rank to all ranks.

    Uses the encoder's own parameter device (NOT ``cuda:{global_rank}``) so
    multi-node is correct. When ``verify=True``, after broadcast all ranks
    are checked for consistency via :func:`verify_infmem_params`.
    """
    if not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return True

    encoder = get_infmem_encoder(generator)
    if encoder is None:
        return True

    # Broadcast all parameters.
    for p in encoder.parameters():
        dist.broadcast(p.data, src=src, group=group)

    # Broadcast all buffers.
    for buf in encoder.buffers():
        dist.broadcast(buf.data, src=src, group=group)

    if verify:
        verify_infmem_params(generator, group=group)

    return True


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
