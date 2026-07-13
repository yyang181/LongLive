"""Echo-Infinity's Relative-RoPE + learnable-memory forward, adapted for
LongLive's Wan2.2-TI2V-5B ``CausalWanSelfAttention`` / ``CausalWanAttentionBlock``.

Design goals
------------
1) **Zero-overhead when disabled.** The wrapper only calls :func:`attach_infmem`
   when the config asks for it; unpatched forwards behave bit-exact like the
   stock LongLive I2V AR path.

2) **Additive KV branch.** When memory is enabled, self-attn attends over
   ``concat([sink, memory, local])`` where the memory K/V are produced by an
   external :class:`QueryMemoryEncoder` (kept **outside** FSDP via
   ``object.__setattr__``, see ``utils/infinity_memory_wrapper.py``).

3) **Relative RoPE.** The rope layout used inside this file pins Q to
   ``[q_last-B+1 .. q_last]`` where ``q_last = min(current_start_frame+B-1,
   pmax-1)``. All other segments (sink, memory, local KV) are placed to the
   LEFT of Q inside the same ``[0, pmax-1]`` window. This keeps the RoPE
   support bounded regardless of true video length -- the key idea from
   Echo-Infinity.

4) **Compatible with LongLive's KV cache path.** We reuse LongLive's
   ``_apply_cache_updates`` (unchanged), including the ``pinned_shift`` /
   multi-shot sink protocol. The only path we override is:
     * ``CausalWanSelfAttention.forward``  -> :func:`_self_attn_infmem_forward`
     * ``CausalWanAttentionBlock.forward`` -> :func:`_block_infmem_forward`
   All extra state (``relative_rope_pmax``, ``num_frame_per_block_attr``,
   ``_layer_id``, and the ``memory_kv`` kwarg) is added purely as attributes
   / kwargs, so nothing else in the codebase needs to change.

5) **5B-specific adaptations vs Echo-Infinity's 1.3B reference:**
     * ``frame_seqlen`` is derived from ``_CURRENT_GRID_META`` (dynamic,
       ``= h * w``) rather than hard-coded to 1560.
     * ``num_frame_per_block`` defaults to 8 (I2V AR) rather than 3.
     * We forward LongLive's rope extras (``t_scale``, ``method``,
       ``original_seq_len``, ``temporal_offset``) into ``causal_rope_apply``.
     * The multi-shot sink / global_sink / pinned protocol is respected by
       treating ``effective_sink`` as the "left-boundary" of the local window
       (identical to LongLive's stock formula).

The mechanism is orthogonal to LongLive's ``use_relative_rope``: when infmem
is attached, the layer ignores ``use_relative_rope`` and uses infmem's own
relative-window layout instead. To use LongLive's own relative rope, simply
do NOT attach infmem.
"""

from __future__ import annotations

import math
import types
import torch
import torch.distributed as dist
from contextlib import nullcontext

from wan_5b.modules.attention import attention
from wan_5b.modules.causal_model import (
    causal_rope_apply,
    _CURRENT_GRID_META,
    _TRITON_ADALN_ENABLED,
)


# Log flags mirror Echo-Infinity for parity with training logs.
_FIRST_LONG_LOGGED = {"flag": False}
_FIRST_BULK_LOGGED = {"flag": False}


def _reset_log_flags() -> None:
    _FIRST_LONG_LOGGED["flag"] = False
    _FIRST_BULK_LOGGED["flag"] = False


def _infmem_autocast_context(reference: torch.Tensor):
    """Return an autocast context matching the reference tensor's dtype.

    Only enables CUDA autocast when the reference tensor is a CUDA half /
    bfloat16 tensor. This keeps the FP32 encoder parameters numerically
    correct while letting the runtime BF16 query_state flow through the
    encoder's Linear layers without dtype-mismatch errors. CPU / FP32
    tensors return a no-op context so unit tests and FP32 training work.
    """
    if (
        reference.is_cuda
        and reference.dtype in (torch.float16, torch.bfloat16)
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=reference.dtype,
        )
    return nullcontext()


def _compute_relative_positions(
    current_start_frame: int,
    B: int,
    R: int,
    N_Q: int,
    N_S: int,
    pmax: int,
    num_frame_per_block: int,
):
    """1:1 with Echo-Infinity's `CausalWanSelfAttention._compute_relative_positions`.

    Returns a dict describing the relative frame windows used for RoPE:
      * sink_start = 0 (always)
      * memory : [mem_start, mem_end] (only when use_memory is True)
      * local  : [local_start, local_end] (only when R > 0)
      * Q      : [q_start, q_last]

    ``q_last`` is pinned at ``pmax - 1`` when the AR has advanced past the
    training rope horizon; earlier chunks use their true position.
    """
    is_bulk_forward = B > num_frame_per_block
    q_last = min(current_start_frame + B - 1, pmax - 1)
    q_start = q_last - B + 1
    local_end = q_last
    local_start = local_end - R + 1 if R > 0 else q_last
    if is_bulk_forward or N_Q == 0:
        use_memory = False
        mem_start = -1
        mem_end = -1
    else:
        use_memory = True
        mem_end = local_start - 1
        mem_start = mem_end - N_Q + 1
    return dict(
        is_bulk_forward=is_bulk_forward,
        use_memory=use_memory,
        q_start=q_start,
        q_last=q_last,
        local_start=local_start,
        local_end=local_end,
        mem_start=mem_start,
        mem_end=mem_end,
        sink_start=0,
    )


def _self_attn_infmem_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
    # LongLive-only rope extras; accepted so the stock block forward can pass
    # them through without knowing about infmem.
    t_scale=1.0,
    use_relative_rope=False,
    method="linear",
    original_seq_len=None,
    temporal_offset=0.0,
    # infmem-only kwargs
    memory_kv=None,
):
    """Drop-in replacement for :meth:`CausalWanSelfAttention.forward`.

    Differences vs the stock forward:
      * When ``kv_cache is None`` we fall back to the original forward path
        so that teacher-forcing / non-KV-cache training keeps working.
      * When ``kv_cache is not None``, we always run the RELATIVE-RoPE layout
        below, regardless of ``use_relative_rope`` (infmem's rope is stricter).
      * If ``memory_kv`` is provided AND ``B <= num_frame_per_block`` AND
        history exists, an additional segment ``mem_k_roped`` is spliced in
        between sink and local.
    """
    # -------- Fallback for training (no KV cache): reuse stock forward ---
    if kv_cache is None:
        # Restore the original forward for this one call. This path is used
        # by teacher-forcing training (Wan2.2 flex_attention), which does not
        # benefit from infmem/relative-rope (the whole clip is materialized).
        orig = getattr(self, "_original_forward", None)
        if orig is None:
            raise RuntimeError(
                "_self_attn_infmem_forward called without kv_cache and no "
                "_original_forward is bound; did attach_infmem run?"
            )
        return orig(
            x, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache=None, current_start=current_start, cache_start=cache_start,
            t_scale=t_scale, use_relative_rope=use_relative_rope,
            method=method, original_seq_len=original_seq_len,
            temporal_offset=temporal_offset,
        )

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if cache_start is None:
        cache_start = current_start

    def qkv_fn(_x):
        q = self.norm_q(self.q(_x)).view(b, s, n, d)
        k = self.norm_k(self.k(_x)).view(b, s, n, d)
        v = self.v(_x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # -------- geometry pulled from LongLive's _CURRENT_GRID_META -----------
    if _CURRENT_GRID_META:
        frame_seqlen = _CURRENT_GRID_META["frame_seqlen"]
        num_new_frames = _CURRENT_GRID_META["num_new_frames"]
        h = _CURRENT_GRID_META["h"]
        w = _CURRENT_GRID_META["w"]
    else:  # unit-test fallback
        frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
        num_new_frames = int(grid_sizes[0][0].item())
        h = int(grid_sizes[0][1].item()); w = int(grid_sizes[0][2].item())

    num_new_tokens = q.shape[1]
    current_end = current_start + num_new_tokens

    # Sink / global-sink / pinned handling (identical to LongLive stock).
    sink_tokens = self.sink_size * frame_seqlen
    global_sink_tokens = getattr(self, "global_sink_size", 0) * frame_seqlen
    if _CURRENT_GRID_META and "pinned_start" in _CURRENT_GRID_META:
        pinned_start_val = _CURRENT_GRID_META["pinned_start"]
        pinned_len_val = _CURRENT_GRID_META["pinned_len"]
    else:
        _ps = kv_cache.get("pinned_start", None)
        if _ps is not None and hasattr(_ps, "item"):
            pinned_start_val = _ps.item()
            pinned_len_val = kv_cache["pinned_len"].item()
        else:
            pinned_start_val = -1
            pinned_len_val = 0
    has_pinned = pinned_start_val >= 0 and pinned_len_val > 0
    if has_pinned and pinned_start_val == global_sink_tokens:
        effective_sink = global_sink_tokens + pinned_len_val
    elif has_pinned:
        effective_sink = global_sink_tokens
    else:
        effective_sink = max(global_sink_tokens, sink_tokens)

    if _CURRENT_GRID_META and "global_end_index" in _CURRENT_GRID_META:
        _cache_global_end = _CURRENT_GRID_META["global_end_index"]
        _cache_local_end = _CURRENT_GRID_META["local_end_index"]
    else:
        _cache_global_end = kv_cache["global_end_index"].item()
        _cache_local_end = kv_cache["local_end_index"].item()

    kv_cache_size = kv_cache["k"].shape[1]
    is_recompute = current_end <= _cache_global_end and current_start > 0

    # -------- KV write path (mirrors stock LongLive; not quantized) --------
    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]

    # Build only the actually attended sink/local segments instead of cloning
    # the full KV cache. A full clone costs ~130MB/layer for 12 cached frames
    # at Wan-5B shape and easily causes OOM when gradient checkpointing is off.
    k_sink_raw = v_sink_raw = None
    k_local_raw = v_local_raw = None

    if (
        self.local_attn_size != -1
        and current_end > _cache_global_end
        and num_new_tokens + _cache_local_end > kv_cache_size
    ):
        num_evicted_tokens = num_new_tokens + _cache_local_end - kv_cache_size
        num_rolled_tokens = _cache_local_end - num_evicted_tokens - effective_sink
        local_end_index = (
            _cache_local_end + current_end - _cache_global_end - num_evicted_tokens
        )
        local_start_index = local_end_index - num_new_tokens

        write_start_index = max(local_start_index, effective_sink) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        new_k = k[:, roped_offset:roped_offset + write_len]
        new_v = v[:, roped_offset:roped_offset + write_len]

        pinned_shift = num_evicted_tokens if (has_pinned and pinned_start_val >= effective_sink) else 0
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": effective_sink,
            "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "new_k": new_k,
            "new_v": new_v,
            "current_end": current_end,
            "pinned_shift": pinned_shift,
        }

        if effective_sink > 0:
            k_sink_raw = cache_k[:, :effective_sink]
            v_sink_raw = cache_v[:, :effective_sink]

        local_k_parts, local_v_parts = [], []
        if num_rolled_tokens > 0:
            roll_start = effective_sink + num_evicted_tokens
            roll_end = roll_start + num_rolled_tokens
            local_k_parts.append(cache_k[:, roll_start:roll_end])
            local_v_parts.append(cache_v[:, roll_start:roll_end])
        if write_len > 0:
            local_k_parts.append(new_k)
            local_v_parts.append(new_v)
        if local_k_parts:
            k_local_raw = torch.cat(local_k_parts, dim=1) if len(local_k_parts) > 1 else local_k_parts[0]
            v_local_raw = torch.cat(local_v_parts, dim=1) if len(local_v_parts) > 1 else local_v_parts[0]
    else:
        local_end_index = _cache_local_end + current_end - _cache_global_end
        local_start_index = local_end_index - num_new_tokens

        write_start_index = max(local_start_index, effective_sink) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        new_k = k[:, roped_offset:roped_offset + write_len]
        new_v = v[:, roped_offset:roped_offset + write_len]

        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "new_k": new_k,
            "new_v": new_v,
            "current_end": current_end,
            "pinned_shift": 0,
        }

        if effective_sink > 0:
            if write_len > 0 and write_start_index < effective_sink:
                sink_k_parts, sink_v_parts = [], []
                if write_start_index > 0:
                    sink_k_parts.append(cache_k[:, :write_start_index])
                    sink_v_parts.append(cache_v[:, :write_start_index])
                overlap_end = min(local_end_index, effective_sink)
                overlap_len = max(0, overlap_end - write_start_index)
                if overlap_len > 0:
                    sink_k_parts.append(k[:, roped_offset:roped_offset + overlap_len])
                    sink_v_parts.append(v[:, roped_offset:roped_offset + overlap_len])
                if overlap_end < effective_sink:
                    sink_k_parts.append(cache_k[:, overlap_end:effective_sink])
                    sink_v_parts.append(cache_v[:, overlap_end:effective_sink])
                k_sink_raw = torch.cat(sink_k_parts, dim=1) if len(sink_k_parts) > 1 else sink_k_parts[0]
                v_sink_raw = torch.cat(sink_v_parts, dim=1) if len(sink_v_parts) > 1 else sink_v_parts[0]
            else:
                k_sink_raw = cache_k[:, :effective_sink]
                v_sink_raw = cache_v[:, :effective_sink]

        local_k_parts, local_v_parts = [], []
        prefix_end = max(effective_sink, min(write_start_index, local_end_index))
        if prefix_end > effective_sink:
            local_k_parts.append(cache_k[:, effective_sink:prefix_end])
            local_v_parts.append(cache_v[:, effective_sink:prefix_end])
        insert_start = max(write_start_index, effective_sink)
        if write_len > 0 and local_end_index > insert_start:
            insert_offset = roped_offset + (insert_start - write_start_index)
            insert_len = local_end_index - insert_start
            local_k_parts.append(k[:, insert_offset:insert_offset + insert_len])
            local_v_parts.append(v[:, insert_offset:insert_offset + insert_len])
        if local_k_parts:
            k_local_raw = torch.cat(local_k_parts, dim=1) if len(local_k_parts) > 1 else local_k_parts[0]
            v_local_raw = torch.cat(local_v_parts, dim=1) if len(local_v_parts) > 1 else local_v_parts[0]

    # -------- Relative RoPE layout --------------------------------------
    num_cache_frames = local_end_index // frame_seqlen
    sink_size_frames = effective_sink // frame_seqlen
    R_active = num_cache_frames - sink_size_frames
    N_Q_rr = int(_CURRENT_GRID_META.get("memory_frames", 0)) if memory_kv is not None else 0
    current_start_frame = current_start // frame_seqlen
    num_frame_per_block_attr = getattr(self, "num_frame_per_block_attr", 8)
    relative_rope_pmax = getattr(self, "relative_rope_pmax", 24)

    rr_pos = _compute_relative_positions(
        current_start_frame=current_start_frame,
        B=num_new_frames,
        R=R_active,
        N_Q=N_Q_rr,
        N_S=sink_size_frames,
        pmax=relative_rope_pmax,
        num_frame_per_block=num_frame_per_block_attr,
    )

    _diag = (
        f"[InfMem-diag] layer={getattr(self, '_layer_id', -1)} "
        f"cur_start_f={current_start_frame} B={num_new_frames} R={R_active} "
        f"N_Q={N_Q_rr} N_S={sink_size_frames} pmax={relative_rope_pmax} "
        f"is_bulk={rr_pos['is_bulk_forward']} use_mem={rr_pos['use_memory']} "
        f"sink_s={rr_pos['sink_start']} mem_s={rr_pos['mem_start']} "
        f"mem_e={rr_pos['mem_end']} local_s={rr_pos['local_start']} "
        f"local_e={rr_pos['local_end']} q_s={rr_pos['q_start']} q_l={rr_pos['q_last']}"
    )
    assert rr_pos["q_last"] <= relative_rope_pmax - 1, f"InfMem overflow: q_last > pmax-1 | {_diag}"
    assert rr_pos["q_start"] >= 0, f"InfMem underflow: q_start < 0 | {_diag}"
    if rr_pos["use_memory"]:
        assert rr_pos["mem_start"] >= sink_size_frames, f"InfMem mem overlaps sink | {_diag}"
        assert rr_pos["mem_end"] + 1 == rr_pos["local_start"], f"InfMem mem-local not contiguous | {_diag}"
    if R_active > 0 and (not rr_pos["is_bulk_forward"]):
        assert rr_pos["q_last"] == rr_pos["local_end"], f"InfMem Q-local tail mismatch | {_diag}"

    # ---- Apply RoPE per-segment (sink | memory | local | Q) --------------
    grid_py = [(num_new_frames, h, w)] * b

    if effective_sink > 0 and k_sink_raw is not None:
        v_sink = v_sink_raw
        sink_grid = [(sink_size_frames, h, w)] * b
        k_sink = causal_rope_apply(
            k_sink_raw, sink_grid, freqs, start_frame=0,
            t_scale=t_scale, method=method,
            original_seq_len=original_seq_len,
        ).type_as(v)
    else:
        k_sink = None
        v_sink = None

    if R_active > 0 and k_local_raw is not None:
        v_local = v_local_raw
        local_grid = [(R_active, h, w)] * b
        k_local = causal_rope_apply(
            k_local_raw, local_grid, freqs,
            start_frame=rr_pos["local_start"],
            t_scale=t_scale, method=method,
            original_seq_len=original_seq_len,
        ).type_as(v)
    else:
        k_local = None
        v_local = None

    roped_query = causal_rope_apply(
        q, grid_py, freqs,
        start_frame=rr_pos["q_start"],
        t_scale=t_scale, method=method,
        original_seq_len=original_seq_len,
    ).type_as(v)

    mem_k_roped = None
    mem_v = None
    if memory_kv is not None and rr_pos["use_memory"] and (not rr_pos["is_bulk_forward"]):
        mem_k_raw, mem_v_raw = memory_kv
        N_Q = max(1, int(_CURRENT_GRID_META.get("memory_frames", 1)))
        mem_tpf = max(1, mem_k_raw.shape[1] // N_Q)
        mem_grid = [(N_Q, 1, mem_tpf)] * b
        mem_k_roped = causal_rope_apply(
            mem_k_raw, mem_grid, freqs,
            start_frame=rr_pos["mem_start"],
            t_scale=t_scale, method=method,
            original_seq_len=original_seq_len,
        ).type_as(v)
        mem_v = mem_v_raw

    parts_k, parts_v = [], []
    if k_sink is not None:
        parts_k.append(k_sink); parts_v.append(v_sink)
    if mem_k_roped is not None:
        parts_k.append(mem_k_roped); parts_v.append(mem_v)
    if k_local is not None:
        parts_k.append(k_local); parts_v.append(v_local)
    k_cat = torch.cat(parts_k, dim=1)
    v_cat = torch.cat(parts_v, dim=1)

    # ---- One-shot diagnostic prints (layer 0 only, rank 0 only) ----------
    _layer_id = getattr(self, "_layer_id", -1)
    if _layer_id == 0:
        if (
            not rr_pos["is_bulk_forward"]
            and rr_pos["q_last"] == relative_rope_pmax - 1
            and not _FIRST_LONG_LOGGED["flag"]
        ):
            mem_str = (
                f"mem[{rr_pos['mem_start']}..{rr_pos['mem_end']}]"
                if rr_pos["use_memory"] else "mem=inactive"
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"[InfMem] FIRST LONG PHASE @ t={current_start_frame}: "
                    f"sink[0..{sink_size_frames - 1}] | {mem_str} | "
                    f"local[{rr_pos['local_start']}..{rr_pos['local_end']}] | "
                    f"Q[{rr_pos['q_start']}..{rr_pos['q_last']}]",
                    flush=True,
                )
            _FIRST_LONG_LOGGED["flag"] = True
        if rr_pos["is_bulk_forward"] and not _FIRST_BULK_LOGGED["flag"]:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"[InfMem] FIRST BULK FORWARD @ t={current_start_frame}, "
                    f"B={num_new_frames}, R={R_active}: "
                    f"sink[0..{sink_size_frames - 1}] | gap | "
                    f"local[{rr_pos['local_start']}..{rr_pos['local_end']}] | "
                    f"Q[{rr_pos['q_start']}..{rr_pos['q_last']}] (memory skipped)",
                    flush=True,
                )
            _FIRST_BULK_LOGGED["flag"] = True

    # ---- Attention + output --------------------------------------------
    x_out = attention(roped_query, k_cat, v_cat)
    x_out = x_out.flatten(2)
    x_out = self.o(x_out)

    return x_out, (current_end, local_end_index, cache_update_info)


def _block_infmem_forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    block_mask,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    cache_start=None,
    t_scale=1.0,
    use_relative_rope=False,
    method="linear",
    original_seq_len=None,
    temporal_offset=0.0,
    # infmem-only kwarg
    memory_kv=None,
    # DreamX camera kwarg
    prope_meta=None,
):
    """Mirror of :meth:`CausalWanAttentionBlock.forward`, but plumbs
    ``memory_kv`` down to the (patched) self-attn forward AND executes the
    DreamX ``cam_self_attn`` parallel camera branch when ``prope_meta`` is
    provided.
    """
    num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
    e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    use_triton_adaln = _TRITON_ADALN_ENABLED and not torch.is_grad_enabled()

    if use_triton_adaln:
        from utils.adaln_triton import adaln_modulate_triton
        modulated_x = adaln_modulate_triton(
            x, e_mod[1], e_mod[0], frame_seqlen,
            eps=self.norm1.eps, add_one_to_scale=True,
        )
    else:
        modulated_x = (
            self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e_mod[1]) + e_mod[0]
        ).flatten(1, 2)

    self_attn_result = self.self_attn(
        modulated_x, seq_lens, grid_sizes, freqs, block_mask,
        kv_cache=kv_cache, current_start=current_start, cache_start=cache_start,
        t_scale=t_scale, use_relative_rope=use_relative_rope,
        method=method, original_seq_len=original_seq_len,
        temporal_offset=temporal_offset,
        memory_kv=memory_kv,
    )

    if kv_cache is not None:
        y, cache_update_info = self_attn_result
    else:
        y = self_attn_result
        cache_update_info = None

    x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[2]).flatten(1, 2)

    # ----- Optional DreamX-style parallel PRoPE self-attn -----
    # Active iff (a) ``add_dreamx_cam_self_attn`` was called on this model
    # and (b) the caller supplied viewmats/Ks via ``prope_meta``.
    # Uses the SAME modulated_x and the SAME AdaLN gate e_mod[2] as the
    # native CausalWanAttentionBlock — no new untrained gates introduced.
    if (getattr(self, "cam_self_attn", None) is not None
            and prope_meta is not None):
        cam_emb = {
            "viewmats": prope_meta["viewmats"],
            "K": prope_meta.get("Ks", None),
        }
        y_cam = self.cam_self_attn(
            modulated_x, cam_emb, seq_lens=seq_lens, block_mask=block_mask
        )
        x = x + (y_cam.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[2]).flatten(1, 2)

    # cross-attention + FFN (identical to stock)
    if _CURRENT_GRID_META and "frame_seqlen" in _CURRENT_GRID_META:
        seq_len_py = _CURRENT_GRID_META["frame_seqlen"] * _CURRENT_GRID_META["num_new_frames"]
        is_tf = (x.shape[1] == seq_len_py * 2)
    else:
        is_tf = (x.shape[1] == seq_lens[0].item() * 2)

    x = x + self.cross_attn(
        self.norm3(x), context, context_lens,
        crossattn_cache=crossattn_cache, is_teacher_forcing=is_tf,
    )
    if use_triton_adaln:
        from utils.adaln_triton import adaln_modulate_triton
        ffn_in = adaln_modulate_triton(
            x, e_mod[4], e_mod[3], frame_seqlen,
            eps=self.norm2.eps, add_one_to_scale=True,
        )
    else:
        ffn_in = (
            self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e_mod[4]) + e_mod[3]
        ).flatten(1, 2)
    y = self.ffn(ffn_in)
    x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[5]).flatten(1, 2)

    if cache_update_info is not None:
        return x, cache_update_info
    return x


# ---------------------------------------------------------------------------
# Model-level orchestration
# ---------------------------------------------------------------------------

def _model_forward_inference_infmem(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    cache_start=0,
    defer_cache_updates=False,
    update_memory=True,
    viewmats=None,
    Ks=None,
):
    """Replacement for :meth:`CausalWanModel._forward_inference` that:
      1. Fetches memory KV from the (external) ``query_memory_encoder`` and
         passes it down to each block.
      2. AFTER all blocks have run, snapshots the KV slice that just got
         evicted from the local window and feeds it into
         ``query_memory_encoder.update(...)`` so that future chunks can
         attend over it via the memory branch.
    """
    from wan_5b.modules.model import sinusoidal_embedding_1d

    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

    # Publish per-chunk grid metadata identical to the stock forward.
    first_shape = tuple(x[0].shape[2:])
    _CURRENT_GRID_META["frame_seqlen"] = int(first_shape[1] * first_shape[2])
    _CURRENT_GRID_META["num_new_frames"] = int(first_shape[0])
    _CURRENT_GRID_META["h"] = int(first_shape[1])
    _CURRENT_GRID_META["w"] = int(first_shape[2])

    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    x = torch.cat(x)

    # Time embeddings.
    if t.dim() == 1:
        raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")
    bt = t.size(0)
    t_len = t.size(1)
    t = t.flatten()
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, t_len)).type_as(x)
    )
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # Context.
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    # ---- Build PRoPE meta (if camera info provided) ----
    # Mirrors CausalWanModel._forward_inference exactly so DreamX cam_self_attn
    # receives the same prope_meta dict structure it expects.
    prope_meta = None
    if viewmats is not None:
        viewmats = viewmats.to(device=device)
        if Ks is not None:
            Ks = Ks.to(device=device)
        f_lat = int(grid_sizes[0, 0].item())
        h_lat = int(grid_sizes[0, 1].item())
        w_lat = int(grid_sizes[0, 2].item())
        assert viewmats.shape[1] == f_lat, (
            f"viewmats has {viewmats.shape[1]} camera frames but latent grid has "
            f"{f_lat} frames; camera and VAE latent time axes must match."
        )
        if Ks is not None:
            assert Ks.shape[1] == f_lat, (
                f"Ks has {Ks.shape[1]} camera frames but latent grid has {f_lat}."
            )
        prope_meta = {
            "viewmats": viewmats,
            "Ks": Ks,
            "F": f_lat,
            "H": h_lat,
            "W": w_lat,
        }

    # ---------- INFMEM: pull memory KV once per chunk ---------------------
    frame_seqlen = _CURRENT_GRID_META["frame_seqlen"]
    enc = getattr(self, "query_memory_encoder", None)
    num_blocks = len(self.blocks)

    def _adapt_memory_kv_to_model_heads(kv_pair):
        if kv_pair is None:
            return None
        mk, mv = kv_pair
        model_heads = int(getattr(self, "num_heads"))
        model_head_dim = int(getattr(self, "dim")) // model_heads
        if mk.shape[-1] != model_head_dim or mv.shape[-1] != model_head_dim:
            raise ValueError(
                f"Memory KV head_dim must match model head_dim={model_head_dim}, "
                f"got k={tuple(mk.shape)}, v={tuple(mv.shape)}."
            )
        src_heads = int(mk.shape[2])
        if src_heads == model_heads:
            return mk, mv
        if model_heads % src_heads == 0:
            repeat = model_heads // src_heads
            return mk.repeat_interleave(repeat, dim=2), mv.repeat_interleave(repeat, dim=2)
        if src_heads % model_heads == 0:
            group = src_heads // model_heads
            new_shape = (mk.shape[0], mk.shape[1], model_heads, group, mk.shape[3])
            return mk.reshape(new_shape).mean(dim=3), mv.reshape(new_shape).mean(dim=3)
        raise ValueError(
            f"Cannot adapt memory KV heads from {src_heads} to model heads "
            f"{model_heads}; one must divide the other."
        )

    if enc is not None and getattr(enc, "has_history", False):
        num_query_groups = getattr(enc, "num_query_groups", 1)
        # Validate query-group configuration before use.
        if num_query_groups < 1:
            raise ValueError(
                f"num_query_groups must be >= 1, got {num_query_groups}"
            )
        if num_query_groups > num_blocks:
            raise ValueError(
                f"num_query_groups ({num_query_groups}) > num_blocks "
                f"({num_blocks})"
            )
        if num_blocks % num_query_groups != 0:
            raise ValueError(
                f"num_blocks ({num_blocks}) must be divisible by "
                f"num_query_groups ({num_query_groups})"
            )
        # Reference hidden tensor for autocast dtype selection. The external
        # encoder runs in the training dtype, and we still keep this context so
        # mixed-precision calls stay aligned with the hidden states.
        _autocast_ref = x
        with _infmem_autocast_context(_autocast_ref):
            if num_query_groups > 1:
                blocks_per_group = num_blocks // num_query_groups
                group_kvs = [
                    enc.get_kv(group_index=g)
                    for g in range(num_query_groups)
                ]
                _CURRENT_GRID_META["memory_frames"] = int(getattr(enc, "Q_frames", 0) or 0)
                _CURRENT_GRID_META["memory_tokens_per_frame"] = int(getattr(enc, "M_tokens_per_frame", frame_seqlen) or frame_seqlen)
                memory_kv_list = []
                for i in range(num_blocks):
                    g = min(i // blocks_per_group, num_query_groups - 1)
                    kv_pair = _adapt_memory_kv_to_model_heads(group_kvs[g])
                    if kv_pair is not None:
                        mk, mv = kv_pair
                        mk = mk.to(device=x.device, dtype=x.dtype)
                        mv = mv.to(device=x.device, dtype=x.dtype)
                        kv_pair = (mk, mv)
                    memory_kv_list.append(kv_pair)
            else:
                kv = enc.get_kv()
                _CURRENT_GRID_META["memory_frames"] = int(getattr(enc, "Q_frames", 0) or 0)
                _CURRENT_GRID_META["memory_tokens_per_frame"] = int(getattr(enc, "M_tokens_per_frame", frame_seqlen) or frame_seqlen)
                kv = _adapt_memory_kv_to_model_heads(kv)
                if kv is not None:
                    mk, mv = kv
                    mk = mk.to(device=x.device, dtype=x.dtype)
                    mv = mv.to(device=x.device, dtype=x.dtype)
                    memory_kv_list = [(mk, mv)] * num_blocks
                else:
                    memory_kv_list = [None] * num_blocks
    else:
        memory_kv_list = [None] * num_blocks

    # Build kwargs common to every block.
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=self.block_mask,
        t_scale=self.t_scale,
        use_relative_rope=self.use_relative_rope,
        method=self.rope_method,
        original_seq_len=self.original_seq_len,
        temporal_offset=self.rope_temporal_offset,
        prope_meta=prope_meta,
    )

    def create_custom_forward(module):
        def custom_forward(*inputs, **_kwargs):
            return module(*inputs, **_kwargs)
        return custom_forward

    cache_update_infos = []
    # Do not use torch.utils.checkpoint on the InfMem KV-cache path: backward
    # recomputation would see the cache after subsequent clean/context updates,
    # so saved/recomputed tensor metadata can differ.
    use_block_checkpoint = False
    for block_index, block in enumerate(self.blocks):
        kwargs["memory_kv"] = memory_kv_list[block_index]
        if use_block_checkpoint:
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
            })
            result = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block), x, **kwargs, use_reentrant=False,
            )
        else:
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
            })
            result = block(x, **kwargs)
        if kv_cache is not None and isinstance(result, tuple):
            x, block_cache_update_info = result
            cache_update_infos.append((block_index, block_cache_update_info))
        else:
            x = result

    # ---------- INFMEM: capture EVICTED KV → encoder.update() ------------
    # We do this BEFORE `_apply_cache_updates` so that the eviction snapshot
    # is taken from the pre-shift KV cache.
    #
    # The ONLY source of truth for how many tokens were evicted is the
    # ``cache_update_info`` dict returned by the self-attention forward
    # (``num_evicted_tokens`` + ``sink_tokens``). We MUST NOT re-derive the
    # evicted-frame count from a local cursor like ``_ei_prev_window_start``
    # because multi-shot pinned-sink layouts shift the effective sink, which
    # would make a static ``sink_tok`` capture the wrong slice.
    if (
        kv_cache is not None
        and cache_update_infos
        and enc is not None
        and update_memory
    ):
        last_block_idx = cache_update_infos[-1][0]
        last_update_info = cache_update_infos[-1][1]
        update_dict = last_update_info[2] if len(last_update_info) > 2 else None

        # Canonical sink (stable global sink) used for the sink anchor.
        canonical_sink_frames = self.blocks[0].self_attn.sink_size
        canonical_sink_tokens = canonical_sink_frames * frame_seqlen

        # Only perform a memory update when the self-attention actually
        # reported a roll-and-insert eviction with a positive token count.
        do_memory_update = (
            update_dict is not None
            and update_dict.get("action") == "roll_and_insert"
            and int(update_dict.get("num_evicted_tokens", 0)) > 0
        )

        if do_memory_update:
            effective_sink_tokens = int(update_dict["sink_tokens"])
            num_evicted_tokens = int(update_dict["num_evicted_tokens"])
            assert effective_sink_tokens >= 0, (
                f"effective_sink_tokens must be >= 0, got {effective_sink_tokens}"
            )
            assert num_evicted_tokens > 0, (
                f"num_evicted_tokens must be > 0, got {num_evicted_tokens}"
            )
            assert num_evicted_tokens % frame_seqlen == 0, (
                f"num_evicted_tokens ({num_evicted_tokens}) must be a multiple "
                f"of frame_seqlen ({frame_seqlen})"
            )

            # Capture from the pre-shift cache using the REAL effective sink
            # (which includes pinned shot sinks under multi-shot). Do NOT use
            # the static ``sink_tok`` here.
            cache_blk = kv_cache[last_block_idx]
            capture_start = effective_sink_tokens
            capture_end = capture_start + num_evicted_tokens
            exited_k = cache_blk["k"][:, capture_start:capture_end].clone()
            exited_v = cache_blk["v"][:, capture_start:capture_end].clone()

            # Sink anchor uses the CANONICAL (stable global) sink, NOT the
            # effective (pinned-inclusive) sink, so temporary pinned shot
            # sinks are not folded into long-term memory.
            if canonical_sink_tokens > 0:
                sink_k = cache_blk["k"][:, :canonical_sink_tokens].clone()
                sink_v = cache_blk["v"][:, :canonical_sink_tokens].clone()
            else:
                sink_k = None
                sink_v = None

            num_evicted_frames = num_evicted_tokens // frame_seqlen
            prev_total = getattr(self, "_ei_total_evicted_frames", 0)
            object.__setattr__(self, "_ei_total_evicted_frames", prev_total + num_evicted_frames)
            # Track for diagnostics only — must NOT drive the capture.
            object.__setattr__(self, "_ei_last_evicted_frames", num_evicted_frames)
            object.__setattr__(self, "_ei_last_evicted_tokens", num_evicted_tokens)

            strict_update = getattr(self, "_ei_strict_update", True)
            try:
                with _infmem_autocast_context(exited_k):
                    enc.update(exited_k, exited_v, sink_k, sink_v)
            except Exception as exc:
                if strict_update:
                    raise RuntimeError(
                        f"[InfMem] encoder.update() failed (strict_update=True): {exc}"
                    ) from exc
                # Non-strict: log once and continue.
                if not getattr(self, "_ei_update_error_logged", False):
                    print(f"[InfMem][warn] encoder.update failed: {exc}", flush=True)
                    self._ei_update_error_logged = True
        else:
            # No eviction this chunk — update the diagnostics cursor so
            # downstream logging still has a value, but do NOT capture.
            object.__setattr__(self, "_ei_last_evicted_frames", 0)
            object.__setattr__(self, "_ei_last_evicted_tokens", 0)

    if kv_cache is not None and cache_update_infos and not defer_cache_updates:
        self._apply_cache_updates(kv_cache, cache_update_infos)

    x = self.head(x, e.unsqueeze(2))
    x = self.unpatchify(x, grid_sizes)
    output = torch.stack(x)
    if kv_cache is not None and defer_cache_updates:
        return output, cache_update_infos
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_infmem(
    model,
    *,
    relative_rope: bool = True,
    relative_rope_pmax: int = 24,
    num_frame_per_block_attr: int = 8,
):
    """Attach Echo-Infinity's Relative-RoPE + memory forward paths to a
    LongLive :class:`CausalWanModel` **in place**.

    Idempotent: re-invoking simply overwrites the bound methods again.

    Parameters
    ----------
    model : CausalWanModel
        Un-wrapped model (call this BEFORE FSDP wrapping).
    relative_rope : bool
        Currently must be True. Kept as a knob for future ablations.
    relative_rope_pmax : int
        Upper bound of the relative rope window. Must satisfy
        ``pmax >= sink_frames + N_Q_frames + num_frame_per_block``.
    num_frame_per_block_attr : int
        Chunk size in latent frames (Wan2.2-TI2V-5B I2V AR = 8).
    """
    # Model-level knobs.
    model.relative_rope = bool(relative_rope)
    model.relative_rope_pmax = int(relative_rope_pmax)
    model.num_frame_per_block_attr = int(num_frame_per_block_attr)

    # Per-block patches.
    for i, block in enumerate(model.blocks):
        sa = block.self_attn
        sa.relative_rope = bool(relative_rope)
        sa.relative_rope_pmax = int(relative_rope_pmax)
        sa.num_frame_per_block_attr = int(num_frame_per_block_attr)
        sa._layer_id = i
        if not hasattr(sa, "_original_forward"):
            sa._original_forward = sa.forward
        sa.forward = types.MethodType(_self_attn_infmem_forward, sa)
        if not hasattr(block, "_original_forward"):
            block._original_forward = block.forward
        block.forward = types.MethodType(_block_infmem_forward, block)

    # Model-level forward override (memory reset/get/update orchestration).
    if not hasattr(model, "_original_forward_inference"):
        model._original_forward_inference = model._forward_inference
    model._forward_inference = types.MethodType(_model_forward_inference_infmem, model)

    # Placeholders. The actual encoder is attached by the wrapper via
    # object.__setattr__ so FSDP does not flatten it.
    if not hasattr(model, "query_memory_encoder"):
        object.__setattr__(model, "query_memory_encoder", None)
    if not hasattr(model, "_ei_prev_window_start"):
        model._ei_prev_window_start = None
    if not hasattr(model, "_ei_total_evicted_frames"):
        model._ei_total_evicted_frames = 0
    # Diagnostics populated from the real cache_update_info metadata.
    if not hasattr(model, "_ei_last_evicted_frames"):
        model._ei_last_evicted_frames = 0
    if not hasattr(model, "_ei_last_evicted_tokens"):
        model._ei_last_evicted_tokens = 0
    model._ei_strict_update = True

    _reset_log_flags()


def detach_infmem(model):
    """Restore original forwards. Safe to call even if never attached."""
    for block in getattr(model, "blocks", []):
        sa = block.self_attn
        if hasattr(sa, "_original_forward"):
            sa.forward = sa._original_forward
            del sa._original_forward
        if hasattr(block, "_original_forward"):
            block.forward = block._original_forward
            del block._original_forward
    if hasattr(model, "_original_forward_inference"):
        model._forward_inference = model._original_forward_inference
        del model._original_forward_inference
