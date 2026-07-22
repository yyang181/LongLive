# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel (Ulysses) attention for camera-conditioned Wan.

Why this exists
---------------
The base model here is Wan2.2-TI2V-5B (vs. minWM's 1.3B). A full bidirectional
camera-PRoPE SFT forward over all latent frames does not fit in memory on a
single GPU, so we shard the latent **frame** dimension across the SP group:
frame ``i`` is owned by SP rank ``i // F_local``. Because tokens are laid out
``(F, H, W)``, a frame chunk is exactly a contiguous token chunk, which keeps
PRoPE's per-frame view matrices aligned with the local token sequence.

Design
------
Only ``WanSelfAttention.forward`` needs SP-awareness. The rest of
``WanModel._forward`` already operates purely on the **local** frame chunk
(patch-embed, per-token time modulation ``e``/``e0``, ``head``, ``unpatchify``)
once the trainer feeds it a frame-sharded latent + frame-sharded
``viewmats``/``Ks``. So we monkey-patch just the self-attention to run an
Ulysses head-dim all-to-all around ``flash_attention``:

    local [B, S_local, n, d]
      --(all_to_all scatter heads=2, gather seq=1)-->  [B, S_full, n/sp, d]
      --flash_attention over the full sequence-->
      --(all_to_all scatter seq=1, gather heads=2)-->  [B, S_local, n, d]

The forward keeps the output **sharded by frame** (no gather before the head).
The trainer computes a local ``.mean()`` flow-matching loss; combined with
FSDP-over-world this yields exactly the data-parallel gradient (the ``sp``
factor in the local mean cancels the ``1/sp`` from world-averaging), so no
all-gather-before-head and no extra SP gradient all-reduce are required.

Both attention residuals run through the same distributed attention:
  * RoPE residual: ``sp_rope_apply`` already offsets positions by
    ``sp_rank * F_local``, so the local q/k receive the correct global RoPE.
  * PRoPE residual: ``prope_qkv`` is applied to the **local** q/k/v with the
    **local** view matrices *before* the all-to-all, and its output correction
    (``apply_fn_o``) is applied to the **local** attention output *after* the
    return all-to-all. Both are per-token (per-frame) maps, so applying them
    locally then gathering is identical to the full non-SP computation.

The patched attention is also a no-op-correct fallback when ``sp_size == 1``
(``all_to_all`` short-circuits and ``full_seq_lens == seq_lens``).

DreamX E-PRoPE
---------------
DreamX adds a separate ``cam_self_attn`` branch instead of reusing the
backbone Q/K/V.  :func:`sp_dreamx_camera_attn_forward` gives that branch the
same Ulysses treatment.  PRoPE is applied to each rank's local camera frames
before all-to-all and inverted after the return all-to-all.  For causal
teacher forcing, the local ``[clean, noisy]`` layout is intentionally left
unchanged: all-to-all then produces the natural balanced-SP layout
``[r0 clean, r0 noisy, r1 clean, r1 noisy, ...]`` expected by the shared
global block mask.
"""

import torch

from ..modules.attention import flash_attention
from .sp_training import (
    all_to_all_with_grad,
    distributed_flex_attention,
    get_sp_world_size,
)
from .sequence_parallel import sp_rope_apply


def _distributed_attention_with_grad(q, k, v, seq_lens, window_size=(-1, -1)):
    """Ulysses attention with autograd-safe all-to-all (training path).

    Args:
        q, k, v: ``[B, S_local, n, d]`` local (frame-sharded) tensors.
        seq_lens: ``[B]`` *full* key lengths (== local token count * sp_size).
        window_size: passed through to ``flash_attention``.

    Returns:
        ``[B, S_local, n, d]`` attention output for the local frame chunk.
    """
    q = all_to_all_with_grad(q, scatter_dim=2, gather_dim=1)
    k = all_to_all_with_grad(k, scatter_dim=2, gather_dim=1)
    v = all_to_all_with_grad(v, scatter_dim=2, gather_dim=1)
    x = flash_attention(q, k, v, k_lens=seq_lens, window_size=window_size)
    return all_to_all_with_grad(x, scatter_dim=1, gather_dim=2)


def sp_camera_attn_forward(self, x, seq_lens, grid_sizes, freqs, prope_meta=None):
    """SP replacement for ``WanSelfAttention.forward`` (camera-PRoPE bidir).

    ``x`` carries the LOCAL frame chunk ``[B, S_local, C]``. ``grid_sizes`` is
    ``(F_local, H, W)`` and ``prope_meta['viewmats']`` is ``(B, F_local, 4, 4)``
    -- both already frame-sharded by the trainer.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    sp_size = get_sp_world_size()
    # After the head-dim all-to-all the local sequence is gathered into the
    # full sequence, so flash_attention must see the full key length.
    full_seq_lens = seq_lens * sp_size if seq_lens is not None else None

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # ----- Default RoPE path (always runs) -----
    rq = sp_rope_apply(q, grid_sizes, freqs).type_as(v)
    rk = sp_rope_apply(k, grid_sizes, freqs).type_as(v)
    x_rope = _distributed_attention_with_grad(
        rq, rk, v, full_seq_lens, window_size=self.window_size)
    x_rope = x_rope.flatten(2)
    out = self.o(x_rope)

    # ----- Optional PRoPE residual path -----
    if prope_meta is not None and getattr(self, "prope_o", None) is not None:
        from ..modules.prope import prope_qkv
        viewmats = prope_meta["viewmats"]   # (B, F_local, 4, 4)
        Ks = prope_meta.get("Ks", None)     # (B, F_local, 3, 3) or None
        num_cams = viewmats.shape[1]
        # Self-attn input is (B, S_local, n, d); PRoPE expects (B, n, S_local, d).
        qp = q.permute(0, 2, 1, 3).contiguous()
        kp = k.permute(0, 2, 1, 3).contiguous()
        vp = v.permute(0, 2, 1, 3).contiguous()
        # Local tokens keep the (F_local, H, W) layout, so the local sequence
        # partitions exactly into ``num_cams == F_local`` cameras.
        assert qp.shape[2] % num_cams == 0, (
            f"local seqlen={qp.shape[2]} not divisible by local num_cams="
            f"{num_cams}; PRoPE expects token order (F, H, W).")
        qp, kp, vp, apply_fn_o = prope_qkv(
            qp, kp, vp,
            viewmats=viewmats.to(dtype=qp.dtype),
            Ks=(Ks.to(dtype=qp.dtype) if Ks is not None else None),
        )
        # Back to flash_attention layout: (B, S_local, n, d).
        qf = qp.permute(0, 2, 1, 3).contiguous()
        kf = kp.permute(0, 2, 1, 3).contiguous()
        vf = vp.permute(0, 2, 1, 3).contiguous()
        x_prope = _distributed_attention_with_grad(
            qf, kf, vf, full_seq_lens, window_size=self.window_size)
        # The return all-to-all leaves the LOCAL frames on this rank, so the
        # per-camera output correction uses the LOCAL view matrices.
        x_prope = x_prope.permute(0, 2, 1, 3).contiguous()  # (B, n, S_local, d)
        x_prope = apply_fn_o(x_prope)
        x_prope = x_prope.permute(0, 2, 1, 3).contiguous().flatten(2)
        out = out + self.prope_o(x_prope)

    return out


def sp_dreamx_camera_attn_forward(
    self,
    x,
    cam_emb,
    seq_lens=None,
    block_mask=None,
):
    """Balanced-SP forward for DreamX ``PropeSelfAttention``.

    ``x`` and ``cam_emb`` contain only the frames owned by the current SP
    rank.  The function keeps PRoPE's camera-dependent transforms local while
    making the actual attention global with an autograd-safe Ulysses
    all-to-all.  ``block_mask=None`` is the bidirectional SFT path; a block
    mask selects the causal AR path.

    This function is monkey-patched only when ``sequence_parallel_size > 1``;
    the original DreamX implementation remains untouched for SP=1.
    """
    from ..modules.prope import prope_qkv

    b, s, _ = x.shape
    n, d = self.num_heads, self.head_dim
    sp_size = get_sp_world_size()
    if n % sp_size != 0:
        raise ValueError(
            f"DreamX camera num_heads ({n}) must be divisible by "
            f"sequence_parallel_size ({sp_size})."
        )

    q = self.norm_q(self.q_proj(x)).view(b, s, n, d)
    k = self.norm_k(self.k_proj(x)).view(b, s, n, d)
    v = self.v_proj(x).view(b, s, n, d)

    viewmats = cam_emb["viewmats"]
    Ks = cam_emb.get("K", cam_emb.get("Ks", None))
    num_cams = int(viewmats.shape[1])
    if num_cams <= 0:
        raise ValueError("SP cam_self_attn received an empty camera sequence")

    if seq_lens is not None:
        base_seq_len = int(seq_lens[0].item())
    else:
        if s % num_cams != 0:
            raise ValueError(
                f"SP cam_self_attn sequence length {s} is not divisible by "
                f"local camera frames {num_cams}."
            )
        base_seq_len = s
    is_teacher_forcing = s == base_seq_len * 2

    def apply_local_prope(q_local, k_local, v_local):
        qp = q_local.transpose(1, 2).contiguous()
        kp = k_local.transpose(1, 2).contiguous()
        vp = v_local.transpose(1, 2).contiguous()
        if qp.shape[2] % num_cams != 0:
            raise ValueError(
                f"SP cam_self_attn tokens {qp.shape[2]} are not divisible by "
                f"local camera frames {num_cams}."
            )
        qp, kp, vp, inverse = prope_qkv(
            qp,
            kp,
            vp,
            viewmats=viewmats.to(dtype=qp.dtype),
            Ks=Ks.to(dtype=qp.dtype) if Ks is not None else None,
        )
        return (
            qp.transpose(1, 2).contiguous(),
            kp.transpose(1, 2).contiguous(),
            vp.transpose(1, 2).contiguous(),
            inverse,
        )

    if is_teacher_forcing:
        q_clean, q_noisy = q.split(base_seq_len, dim=1)
        k_clean, k_noisy = k.split(base_seq_len, dim=1)
        v_clean, v_noisy = v.split(base_seq_len, dim=1)
        qp_clean, kp_clean, vp_clean, inverse = apply_local_prope(
            q_clean, k_clean, v_clean
        )
        qp_noisy, kp_noisy, vp_noisy, _ = apply_local_prope(
            q_noisy, k_noisy, v_noisy
        )
        qp = torch.cat([qp_clean, qp_noisy], dim=1)
        kp = torch.cat([kp_clean, kp_noisy], dim=1)
        vp = torch.cat([vp_clean, vp_noisy], dim=1)
    else:
        qp, kp, vp, inverse = apply_local_prope(q, k, v)

    if block_mask is None:
        full_seq_lens = seq_lens * sp_size if seq_lens is not None else None
        out = _distributed_attention_with_grad(
            qp,
            kp,
            vp,
            full_seq_lens,
            window_size=(-1, -1),
        )
    else:
        # distributed_flex_attention preserves each rank's local
        # [clean, noisy] ordering across the return all-to-all.
        out = distributed_flex_attention(qp, kp, vp, block_mask)

    if is_teacher_forcing:
        out_clean, out_noisy = out.split(base_seq_len, dim=1)
        out_clean = inverse(out_clean.transpose(1, 2)).transpose(1, 2)
        out_noisy = inverse(out_noisy.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([out_clean, out_noisy], dim=1)
    else:
        out = inverse(out.transpose(1, 2)).transpose(1, 2)

    return self.out_proj(out.flatten(2))
