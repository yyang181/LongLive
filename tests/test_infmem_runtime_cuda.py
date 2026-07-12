"""CUDA runtime tests for the InfMem QueryMemoryEncoder.

These tests require a CUDA GPU and verify:
  * Test 3: ``get_kv()`` works with FP32 encoder params + BF16 query_state
    (autocast prevents dtype mismatch).
  * Test 4: ``update()`` works with BF16 evicted K/V + FP32 encoder params
    (forward + backward + FP32 grads + finite loss).
  * Test 5: 20-frame streaming memory update produces real evictions.

Run with:
    python -m pytest -q tests/test_infmem_runtime_cuda.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.query_memory import QueryMemoryEncoder
from wan_5b.modules.infinity_memory import _infmem_autocast_context

_HAS_CUDA = torch.cuda.is_available()


def _make_small_encoder_config():
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
        batch_update_interval = 1
        use_sink_anchor = True
        use_vib = False
        bptt_clips = 1
        encoder_lr_multiplier = 5.0
        normalize_memory_k = False
        use_residual_update = True
        use_post_norm = True
        memory_recache = False
        num_query_groups = 1
        initializer_range = 0.02

    return _Cfg()


def _build_encoder(device):
    cfg = _make_small_encoder_config()
    enc = QueryMemoryEncoder(cfg).float().to(device)
    enc.requires_grad_(True)
    return enc


@unittest.skipUnless(_HAS_CUDA, "CUDA required for these tests")
class TestGetKvAutocast(unittest.TestCase):
    """Test 3: get_kv() with FP32 encoder + BF16 query_state."""

    def test_get_kv_bf16_query_state_no_dtype_mismatch(self):
        device = torch.device("cuda")
        enc = _build_encoder(device)
        # Verify encoder params are FP32.
        self.assertTrue(all(p.dtype == torch.float32 for p in enc.parameters()))
        # Reset with BF16 query_state.
        enc.reset(batch_size=1, device=device, dtype=torch.bfloat16)
        self.assertEqual(enc.query_state.dtype, torch.bfloat16)

        # Perform one update so has_history becomes True.
        b, m = 1, enc.M
        fake_k = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        fake_v = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        sink_k = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        sink_v = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        # The autocast helper must let the BF16 input flow through FP32 Linear.
        with _infmem_autocast_context(fake_k):
            enc.update(fake_k, fake_v, sink_k, sink_v)
        self.assertTrue(enc.has_history)

        # get_kv under autocast — must not raise dtype mismatch.
        with _infmem_autocast_context(enc.query_state):
            kv = enc.get_kv()
        self.assertIsNotNone(kv)
        memory_k, memory_v = kv
        # Explicitly cast to the target dtype (as the model forward does).
        target_dtype = torch.bfloat16
        memory_k = memory_k.to(device=device, dtype=target_dtype)
        memory_v = memory_v.to(device=device, dtype=target_dtype)
        self.assertEqual(memory_k.dtype, torch.bfloat16)
        self.assertEqual(memory_v.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(memory_k).all())
        self.assertTrue(torch.isfinite(memory_v).all())

    def test_autocast_context_cpu_noop(self):
        """CPU reference tensor returns a no-op context."""
        ref = torch.randn(2, 2)
        ctx = _infmem_autocast_context(ref)
        with ctx:
            x = torch.randn(2, 2)
        self.assertEqual(x.dtype, torch.float32)

    def test_autocast_context_fp32_cuda_noop(self):
        """FP32 CUDA tensor returns a no-op context."""
        device = torch.device("cuda")
        ref = torch.randn(2, 2, device=device, dtype=torch.float32)
        ctx = _infmem_autocast_context(ref)
        with ctx:
            x = torch.randn(2, 2, device=device)
        self.assertEqual(x.dtype, torch.float32)


@unittest.skipUnless(_HAS_CUDA, "CUDA required for these tests")
class TestUpdateAutocast(unittest.TestCase):
    """Test 4: update() with BF16 K/V + FP32 encoder params (backward)."""

    def test_update_forward_backward_fp32_grad(self):
        device = torch.device("cuda")
        enc = _build_encoder(device)
        enc.reset(batch_size=1, device=device, dtype=torch.bfloat16)

        b, m = 1, enc.M
        fake_k = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        fake_v = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        sink_k = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)
        sink_v = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device, dtype=torch.bfloat16)

        with _infmem_autocast_context(fake_k):
            enc.update(fake_k, fake_v, sink_k, sink_v)
        kv = enc.get_kv()
        loss = kv[0].float().pow(2).sum() + kv[1].float().pow(2).sum()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        # All encoder parameter grads must be FP32.
        for p in enc.parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.dtype, torch.float32,
                                 f"grad dtype {p.grad.dtype} != float32")
        # At least one parameter must have a non-zero grad norm.
        grad_norm = sum(
            p.grad.detach().pow(2).sum().item()
            for p in enc.parameters() if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


@unittest.skipUnless(_HAS_CUDA, "CUDA required for these tests")
class TestStreamingMemoryUpdate(unittest.TestCase):
    """Test 5: 20-frame streaming teacher-forcing produces real evictions."""

    def test_20_frame_streaming_eviction(self):
        """Simulate a 20-frame clip with block=4, local=12, sink=4.

        After the first chunk (4 frames) fills the cache, subsequent chunks
        should trigger roll-and-insert evictions once the window exceeds
        sink+local. We verify the encoder update count and has_history.
        """
        device = torch.device("cuda")
        enc = _build_encoder(device)
        frame_seqlen = enc.num_heads * enc.head_dim  # tokens per frame

        # 20 frames, block=4 -> 5 chunks.
        num_frames = 20
        block = 4
        sink_frames = 4
        local_frames = 12
        # KV cache size = (sink + local) * frame_seqlen tokens.
        kv_cache_size = (sink_frames + local_frames) * frame_seqlen

        enc.reset(batch_size=1, device=device, dtype=torch.bfloat16)
        total_evicted = 0
        chunk_count = 0

        for chunk_idx in range(num_frames // block):
            current_start_frame = chunk_idx * block
            # Simulate the cache metadata that self-attention would return.
            cache_local_end = (current_start_frame + block) * frame_seqlen
            if cache_local_end > kv_cache_size:
                # Eviction happens.
                num_evicted_tokens = cache_local_end - kv_cache_size
                effective_sink = sink_frames * frame_seqlen
                if num_evicted_tokens > 0:
                    # Capture evicted slice (fake data).
                    exited_k = torch.randn(
                        1, num_evicted_tokens, enc.num_heads, enc.head_dim,
                        device=device, dtype=torch.bfloat16,
                    )
                    exited_v = torch.randn(
                        1, num_evicted_tokens, enc.num_heads, enc.head_dim,
                        device=device, dtype=torch.bfloat16,
                    )
                    sink_k = torch.randn(
                        1, effective_sink, enc.num_heads, enc.head_dim,
                        device=device, dtype=torch.bfloat16,
                    )
                    sink_v = torch.randn(
                        1, effective_sink, enc.num_heads, enc.head_dim,
                        device=device, dtype=torch.bfloat16,
                    )
                    with _infmem_autocast_context(exited_k):
                        enc.update(exited_k, exited_v, sink_k, sink_v)
                    total_evicted += num_evicted_tokens // frame_seqlen
            chunk_count += 1
            print(
                f"chunk={chunk_idx} start_f={current_start_frame} "
                f"update_count={enc._update_count} "
                f"has_history={enc.has_history} "
                f"total_evicted_frames={total_evicted}"
            )

        self.assertGreater(enc._update_count, 0)
        self.assertTrue(enc.has_history)
        self.assertGreater(total_evicted, 0)


if __name__ == "__main__":
    unittest.main()
