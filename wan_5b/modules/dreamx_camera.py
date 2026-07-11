# SPDX-License-Identifier: MIT
# DreamX-World style camera module ported into LongLive.
#
# Mirrors DreamX-World/models/wan_transformer3d.py::PropeSelfAttention so that
# we can load the official `DreamX-World-5B-Cam` checkpoint by name. Each
# WanAttentionBlock gets a parallel `cam_self_attn` branch:
#
#     y = self.self_attn(temp_x, ...)               # RoPE branch (existing)
#     if hasattr(self, "cam_self_attn") and cam_emb is not None:
#         y = y + self.cam_self_attn(temp_x, cam_emb, seq_lens=seq_lens)
#     x = x + y * e[2]
#
# The math inside `cam_self_attn` is the same PRoPE formulation already used
# by LongLive's `prope_o` residual path (`wan_5b/modules/prope.py::prope_qkv`).
# The structural difference is that DreamX puts a *full* QKV+out projection
# block alongside the RoPE self-attn (rather than reusing the RoPE Q/K/V and
# only adding a zero-init `o`).
#
# Public API:
#   - PropeSelfAttention(dim, attn_dim, num_heads, qk_norm=True, eps=1e-6)
#       Module shape (must match the DreamX checkpoint layout exactly):
#           q_proj/k_proj/v_proj : Linear(dim, attn_dim)
#           out_proj             : Linear(attn_dim, dim) [zero-init]
#           norm_q/norm_k        : WanRMSNorm(attn_dim)  [if qk_norm]
#   - add_dreamx_cam_self_attn(model, attn_dim=None, num_heads=None,
#                              qk_norm=True, eps=1e-6) -> int
#       Attaches one PropeSelfAttention to every WanAttentionBlock as
#       `block.cam_self_attn`. Returns the number of blocks patched.
#
# Why this lives next to (not inside) `prope.py`:
#   prope.py implements the lightweight `prope_o` residual that LongLive
#   trained from scratch. The DreamX-World-5B-Cam checkpoint instead expects
#   a full `cam_self_attn` submodule per block. The two are mutually
#   exclusive and selected via the yaml flag `algorithm.dreamx_camera`.

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention import attention
from .prope import prope_qkv

try:
    from torch.nn.attention.flex_attention import flex_attention
except ModuleNotFoundError:  # pragma: no cover - depends on torch build
    flex_attention = None


# Local alias so we don't import from model.py (would create a cycle if any
# helper here is imported during model construction).
class _WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match wan_5b/modules/model.py::WanRMSNorm exactly.
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class PropeSelfAttention(nn.Module):
    """Parallel PRoPE self-attention branch (DreamX-World layout).

    This is intentionally a near-verbatim port of
    DreamX-World/models/wan_transformer3d.py::PropeSelfAttention so that the
    state-dict keys
        blocks.{i}.cam_self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
        blocks.{i}.cam_self_attn.{norm_q,norm_k}.weight
    load 1:1 from the `DreamX-World-5B-Cam` checkpoint.

    Args:
        dim:        Model hidden dim (3072 for Wan2.2-TI2V-5B).
        attn_dim:   PRoPE branch hidden dim (defaults to ``dim``; the 5B-Cam
                    checkpoint uses ``attn_compress=1`` → ``attn_dim == dim``).
        num_heads:  PRoPE branch num heads. ``head_dim = attn_dim // num_heads``.
        qk_norm:    Whether to RMSNorm Q and K (DreamX default: True).
        eps:        RMSNorm eps.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0, (
            f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}"
        )
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = _WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = _WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        # Zero-init `out_proj` so that a freshly added module is a no-op
        # residual. The pretrained checkpoint will overwrite these on load.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                      # (B, S, D)
        cam_emb: Dict[str, torch.Tensor],     # {"viewmats": (B,T,4,4), "K"/"Ks": (B,T,3,3)}
        seq_lens: Optional[torch.Tensor] = None,
        block_mask=None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim

        q = self.norm_q(self.q_proj(x)).view(b, s, n, d)
        k = self.norm_k(self.k_proj(x)).view(b, s, n, d)
        v = self.v_proj(x).view(b, s, n, d)

        viewmats = cam_emb["viewmats"]
        # DreamX uses key "K"; LongLive's prope_meta uses "Ks". Accept both.
        Ks = cam_emb.get("K", cam_emb.get("Ks", None))
        num_cams = int(viewmats.shape[1])
        if num_cams <= 0:
            raise ValueError("cam_self_attn received an empty camera sequence")
        if seq_lens is not None:
            base_seq_len = int(seq_lens[0].item())
        else:
            if s % num_cams != 0:
                raise ValueError(
                    f"cam_self_attn sequence length {s} is not divisible by "
                    f"camera frames {num_cams}"
                )
            base_seq_len = s
        is_tf = (s == base_seq_len * 2)

        def _apply_prope(q_in, k_in, v_in):
            # `prope_qkv` expects (B, N, S, D); attention uses (B, S, N, D).
            qp = q_in.transpose(1, 2).contiguous()
            kp = k_in.transpose(1, 2).contiguous()
            vp = v_in.transpose(1, 2).contiguous()
            if qp.shape[2] % num_cams != 0:
                raise ValueError(
                    f"cam_self_attn tokens {qp.shape[2]} are not divisible by "
                    f"camera frames {num_cams}; check clean/noisy camera layout"
                )
            qp, kp, vp, apply_fn_o = prope_qkv(
                qp, kp, vp,
                viewmats=viewmats.to(dtype=qp.dtype),
                Ks=(Ks.to(dtype=qp.dtype) if Ks is not None else None),
            )
            return (
                qp.transpose(1, 2).contiguous(),
                kp.transpose(1, 2).contiguous(),
                vp.transpose(1, 2).contiguous(),
                apply_fn_o,
            )

        apply_fn_o = None
        if is_tf:
            q_clean, q_noisy = q.split(base_seq_len, dim=1)
            k_clean, k_noisy = k.split(base_seq_len, dim=1)
            v_clean, v_noisy = v.split(base_seq_len, dim=1)
            qp_clean, kp_clean, vp_clean, apply_fn_o = _apply_prope(q_clean, k_clean, v_clean)
            qp_noisy, kp_noisy, vp_noisy, _ = _apply_prope(q_noisy, k_noisy, v_noisy)
            qp = torch.cat([qp_clean, qp_noisy], dim=1)
            kp = torch.cat([kp_clean, kp_noisy], dim=1)
            vp = torch.cat([vp_clean, vp_noisy], dim=1)
        else:
            qp, kp, vp, apply_fn_o = _apply_prope(q, k, v)

        if block_mask is None:
            out = attention(qp, kp, v=vp, k_lens=seq_lens)
        else:
            if flex_attention is None:
                raise ModuleNotFoundError(
                    "torch.nn.attention.flex_attention is required for causal "
                    "DreamX camera attention"
                )
            padded_length = ((qp.shape[1] + 127) // 128) * 128 - qp.shape[1]
            if padded_length > 0:
                pad_shape = (b, padded_length, n, d)
                qp = torch.cat([qp, torch.zeros(pad_shape, device=qp.device, dtype=vp.dtype)], dim=1)
                kp = torch.cat([kp, torch.zeros(pad_shape, device=kp.device, dtype=vp.dtype)], dim=1)
                vp = torch.cat([vp, torch.zeros(pad_shape, device=vp.device, dtype=vp.dtype)], dim=1)
            out = flex_attention(
                query=qp.transpose(2, 1),
                key=kp.transpose(2, 1),
                value=vp.transpose(2, 1),
                block_mask=block_mask,
            )
            if padded_length > 0:
                out = out[:, :, :-padded_length]
            out = out.transpose(2, 1)

        # Inverse PRoPE projection on (B, N, S, D), then back to (B, S, N, D).
        if is_tf:
            out_clean, out_noisy = out.split(base_seq_len, dim=1)
            out_clean = apply_fn_o(out_clean.transpose(1, 2)).transpose(1, 2)
            out_noisy = apply_fn_o(out_noisy.transpose(1, 2)).transpose(1, 2)
            out = torch.cat([out_clean, out_noisy], dim=1)
        else:
            out = apply_fn_o(out.transpose(1, 2)).transpose(1, 2)
        out = out.flatten(2)         # (B, S, attn_dim)
        out = self.out_proj(out)     # (B, S, dim)
        return out


def add_dreamx_cam_self_attn(
    model: nn.Module,
    attn_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    qk_norm: bool = True,
    eps: float = 1e-6,
) -> int:
    """Attach a DreamX-style ``cam_self_attn`` to every WanAttentionBlock.

    The submodule name MUST be ``cam_self_attn`` to match the DreamX
    checkpoint state-dict (``blocks.{i}.cam_self_attn.*``).

    Args:
        model:     A ``WanModel`` (LongLive Wan2.2-TI2V-5B variant).
        attn_dim:  PRoPE attn dim. Defaults to each block's ``self.dim``
                   (i.e. ``attn_compress=1``, matching the 5B-Cam config).
        num_heads: PRoPE attn heads. Defaults to each block's ``self.num_heads``.
        qk_norm:   Whether to RMSNorm Q/K (DreamX default: True).
        eps:       RMSNorm eps.

    Returns:
        Number of blocks patched.
    """
    n_added = 0
    for block in getattr(model, "blocks", []):
        # Support both bidirectional WanAttentionBlock and causal
        # CausalWanAttentionBlock. Both expose ``self_attn.dim`` and
        # ``self_attn.num_heads``; the causal AR path currently keeps this
        # module for checkpoint compatibility with DreamX/bidirectional SFT.
        if hasattr(block, "cam_self_attn"):
            continue
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "dim") or not hasattr(self_attn, "num_heads"):
            continue
        block_dim = self_attn.dim
        block_heads = self_attn.num_heads
        a_dim = attn_dim if attn_dim is not None else block_dim
        n_heads = num_heads if num_heads is not None else block_heads
        cam = PropeSelfAttention(
            dim=block_dim,
            attn_dim=a_dim,
            num_heads=n_heads,
            qk_norm=qk_norm,
            eps=eps,
        )
        # Match parameter device/dtype to the host block.
        ref = block.self_attn.o.weight
        cam = cam.to(device=ref.device, dtype=ref.dtype)
        block.cam_self_attn = cam
        n_added += 1
    return n_added
