# SPDX-License-Identifier: MIT
# Ported to LongLive (Wan2.2-TI2V-5B). Math is identical to:
#   minWM/Wan21/wan/modules/prope.py
# (PRoPE: Projective Positional Encoding for Multiview Transformers)
#
# Public API used by LongLive:
#   - prope_qkv(q, k, v, viewmats=, Ks=) -> (q', k', v', apply_fn_o)
#   - add_prope_parameters(model, zero_init=True): attaches a zero-init
#     `prope_o = nn.Linear(dim, dim)` to every WanSelfAttention block.

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


def prope_qkv(
    q: torch.Tensor,        # (B, H, S, D)
    k: torch.Tensor,        # (B, H, S, D)
    v: torch.Tensor,        # (B, H, S, D)
    *,
    viewmats: torch.Tensor, # (B, C, 4, 4)
    Ks: Optional[torch.Tensor],  # (B, C, 3, 3) or None
):
    """Apply PRoPE projective transforms to q/k/v.

    Returns (q', k', v', apply_fn_o). The caller must call
    ``apply_fn_o(attn_out)`` after the attention to project the output back.
    Self-attention only (q/k/v same shape).
    """
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim, viewmats=viewmats, Ks=Ks,
    )
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o


def _prepare_apply_fns_all_dim(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    (batch, cameras, _, _) = viewmats.shape

    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)

        # P = lift(K) @ viewmats : image<-world
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        # GTA fallback: P is `camera<-world`.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"

    transforms_q  = [(partial(_apply_tiled_projmat, matrix=P_T),  head_dim)]
    transforms_kv = [(partial(_apply_tiled_projmat, matrix=P_inv), head_dim)]
    transforms_o  = [(partial(_apply_tiled_projmat, matrix=P),    head_dim)]

    apply_fn_q  = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o  = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,    # (B, H, S, F)
    matrix: torch.Tensor,   # (B, C, D, D)
) -> torch.Tensor:
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0, (
        f"seqlen={seqlen} must be a multiple of cameras={cameras}"
    )
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([f(x) for f, x in zip(funcs, x_blocks)], dim=-1)
    assert out.shape == feats.shape
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out.to(dtype=Ks.dtype)


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def add_prope_parameters(model, zero_init: bool = True):
    """Attach a per-block zero-init `prope_o = nn.Linear(dim, dim)` to every
    WanSelfAttention module inside `model`.

    The PRoPE attention path then computes its output as:
        out = self.o(rope_attn) + self.prope_o(prope_attn)
    so the model output is unchanged at init (since prope_o is zero).
    """
    try:
        from .model import WanSelfAttention, WanCrossAttention
    except ImportError:
        WanSelfAttention = None
        WanCrossAttention = None

    cross_attn_types: tuple = ()
    if WanCrossAttention is not None:
        cross_attn_types = (WanCrossAttention,)

    attn_types = tuple(t for t in (WanSelfAttention,) if t is not None)
    if not attn_types:
        return

    n_added = 0
    for _name, module in model.named_modules():
        if not isinstance(module, attn_types):
            continue
        if isinstance(module, cross_attn_types):
            continue  # only self-attention
        if hasattr(module, "prope_o"):
            continue
        dim = module.o.out_features
        prope_o = nn.Linear(dim, dim)
        if zero_init:
            nn.init.zeros_(prope_o.weight)
            nn.init.zeros_(prope_o.bias)
        prope_o = prope_o.to(
            device=module.o.weight.device, dtype=module.o.weight.dtype,
        )
        module.prope_o = prope_o
        n_added += 1
    return n_added
