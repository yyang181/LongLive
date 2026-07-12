#!/usr/bin/env python
"""Distributed sync tests for the InfMem encoder.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/run_infmem_distributed_sync.py --mode all

Modes:
  broadcast  — Test 9:  two ranks with different seeds → params match after broadcast
  grad_sync   — Test 10: different inputs → gradients match after sync
  no_grad     — Test 11: all ranks have no grad → grad stays None
  partial_grad — Test 12: rank 0 has grad, rank 1 doesn't → synced
  all         — run all of the above
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.query_memory import QueryMemoryEncoder
from utils.infinity_memory_hooks import (
    broadcast_infmem_params,
    sync_infmem_gradients,
    verify_infmem_params,
)


def _make_encoder_config():
    class _Cfg:
        Q_frames = 2
        tokens_per_frame = 4
        M_tokens_per_frame = 4
        n_encoder_layers = 2
        hidden_dim = 32
        num_heads = 4
        head_dim = 8
        gate_init_bias = 2.0
        qk_norm = True
        use_batch_update = False
        use_sink_anchor = True
        use_vib = False
        bptt_clips = 1
        encoder_lr_multiplier = 5.0
        normalize_memory_k = False
        use_residual_update = True
        use_post_norm = True
        num_query_groups = 1
        initializer_range = 0.02

    return _Cfg()


def _build_encoder(rank):
    """Build an encoder with a DIFFERENT random init per rank."""
    torch.manual_seed(1000 + rank)
    enc = QueryMemoryEncoder(_make_encoder_config()).float()
    enc.requires_grad_(True)
    return enc


class _FakeInner:
    def __init__(self, encoder):
        self.query_memory_encoder = encoder
        self._ei_prev_window_start = None
        self._ei_total_evicted_frames = 0
        self._ei_last_evicted_frames = 0
        self._ei_strict_update = True


class _FakeGenerator:
    def __init__(self, encoder):
        self.model = _FakeInner(encoder)


def _checksum(enc):
    return sum(p.data.sum().item() for p in enc.parameters())


def test_broadcast():
    rank = dist.get_rank()
    world = dist.get_world_size()
    enc = _build_encoder(rank)
    gen = _FakeGenerator(enc)

    before = _checksum(enc)
    # Verify params differ before broadcast.
    before_tensor = torch.tensor([before], device="cuda", dtype=torch.float64)
    gathered_before = [torch.empty_like(before_tensor) for _ in range(world)]
    dist.all_gather(gathered_before, before_tensor)
    # At least one rank should differ (different seeds).
    all_same = all(torch.allclose(g, gathered_before[0]) for g in gathered_before)
    if all_same:
        print(f"[rank {rank}] WARNING: params identical before broadcast (unexpected)")

    broadcast_infmem_params(gen, src=0, group=None, verify=True)
    after = _checksum(enc)
    after_tensor = torch.tensor([after], device="cuda", dtype=torch.float64)
    gathered_after = [torch.empty_like(after_tensor) for _ in range(world)]
    dist.all_gather(gathered_after, after_tensor)
    all_match = all(torch.allclose(g, gathered_after[0]) for g in gathered_after)
    print(f"[rank {rank}] broadcast: before={before:.4f} after={after:.4f} match={all_match}")
    assert all_match, f"Params differ across ranks after broadcast (rank {rank})"
    # Per-tensor comparison.
    for p in enc.parameters():
        t = p.data.clone()
        gathered = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(gathered, t)
        for r2, g2 in enumerate(gathered):
            assert torch.allclose(g2, gathered[0]), f"param mismatch rank {rank} vs {r2}"
    print(f"[rank {rank}] BROADCAST TEST PASSED")


def test_grad_sync():
    rank = dist.get_rank()
    world = dist.get_world_size()
    enc = _build_encoder(rank)
    gen = _FakeGenerator(enc)
    broadcast_infmem_params(gen, src=0, group=None, verify=False)
    dist.barrier()

    enc.reset(batch_size=1, device=torch.device("cuda"), dtype=torch.float32)
    b, m = 1, enc.M
    # Different input per rank.
    torch.manual_seed(2000 + rank)
    fake_k = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device="cuda")
    fake_v = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device="cuda")
    enc.update(fake_k, fake_v)
    kv = enc.get_kv()
    loss = kv[0].pow(2).sum() + kv[1].pow(2).sum()
    loss.backward()

    sync_infmem_gradients(gen, group=None, average=True)

    # Verify gradients match across ranks.
    for p in enc.parameters():
        if p.grad is None:
            continue
        g = p.grad.clone()
        gathered = [torch.empty_like(g) for _ in range(world)]
        dist.all_gather(gathered, g)
        for r2, g2 in enumerate(gathered):
            assert torch.allclose(g2, gathered[0], rtol=1e-4, atol=1e-5), \
                f"grad mismatch rank {rank} vs {r2}"
    print(f"[rank {rank}] GRAD SYNC TEST PASSED")


def test_no_grad():
    """All ranks have no gradient for a parameter → grad stays None."""
    rank = dist.get_rank()
    enc = _build_encoder(rank)
    gen = _FakeGenerator(enc)
    broadcast_infmem_params(gen, src=0, group=None, verify=False)
    dist.barrier()

    # Pick a parameter that no rank uses in forward.
    # We don't call update() at all → no parameter has a grad.
    sync_infmem_gradients(gen, group=None, average=True)

    for p in enc.parameters():
        assert p.grad is None, \
            f"rank {rank}: grad should be None (no rank had gradient), got {p.grad}"
    print(f"[rank {rank}] NO GRAD TEST PASSED")


def test_partial_grad():
    """Rank 0 has a gradient, rank 1 doesn't → synced, no deadlock."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        print(f"[rank {rank}] PARTIAL GRAD TEST SKIPPED (world<2)")
        return
    enc = _build_encoder(rank)
    gen = _FakeGenerator(enc)
    broadcast_infmem_params(gen, src=0, group=None, verify=False)
    dist.barrier()

    enc.reset(batch_size=1, device=torch.device("cuda"), dtype=torch.float32)
    if rank == 0:
        b, m = 1, enc.M
        torch.manual_seed(3000)
        fake_k = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device="cuda")
        fake_v = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device="cuda")
        enc.update(fake_k, fake_v)
        kv = enc.get_kv()
        loss = kv[0].pow(2).sum() + kv[1].pow(2).sum()
        loss.backward()
    # rank 1: no backward at all.

    sync_infmem_gradients(gen, group=None, average=True)

    # Verify rank 0's gradient equals rank 1's (rank 1 contributed zero).
    for p in enc.parameters():
        if p.grad is None:
            continue
        g = p.grad.clone()
        gathered = [torch.empty_like(g) for _ in range(world)]
        dist.all_gather(gathered, g)
        for r2, g2 in enumerate(gathered):
            assert torch.allclose(g2, gathered[0], rtol=1e-4, atol=1e-5), \
                f"partial grad mismatch rank {rank} vs {r2}"
    print(f"[rank {rank}] PARTIAL GRAD TEST PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    mode = args.mode
    if mode == "all":
        test_broadcast()
        dist.barrier()
        test_grad_sync()
        dist.barrier()
        test_no_grad()
        dist.barrier()
        test_partial_grad()
    elif mode == "broadcast":
        test_broadcast()
    elif mode == "grad_sync":
        test_grad_sync()
    elif mode == "no_grad":
        test_no_grad()
    elif mode == "partial_grad":
        test_partial_grad()
    else:
        print(f"Unknown mode: {mode}")

    dist.barrier()
    if rank == 0:
        print("ALL DISTRIBUTED TESTS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
