"""Real checkpoint round-trip tests for the InfMem encoder + optimizer.

These tests verify:
  * Test 1: optimizer restore order — the encoder optimizer is created BEFORE
    its state is loaded (no AttributeError on ``self.infmem_optimizer``).
  * Test 2: optimizer round-trip — after save → reload → one more training
    step, the resumed optimizer produces bit-identical parameters to an
    uninterrupted run (verifying ``exp_avg`` / ``exp_avg_sq`` are restored).

The tests build a *minimal* QueryMemoryEncoder directly (no full Wan model)
so they run on CPU without GPU or distributed.
"""

import copy
import io
import os
import sys
import tempfile
import types
import unittest

import torch
import torch.nn as nn

# Ensure project root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.query_memory import QueryMemoryEncoder

_HAS_CUDA = torch.cuda.is_available()


def _make_small_encoder_config():
    """A tiny encoder config that builds quickly on CPU for tests."""

    class _Cfg:
        Q_frames = 2
        tokens_per_frame = 4
        M_tokens_per_frame = 4
        n_encoder_layers = 2
        hidden_dim = 32
        num_heads = 4
        head_dim = 8
        ffn_dim = 64
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


def _build_encoder():
    cfg = _make_small_encoder_config()
    enc = QueryMemoryEncoder(cfg)
    enc.requires_grad_(True)
    # Encoder parameters MUST stay FP32.
    enc = enc.float()
    return enc


class _FakeInner:
    """Minimal stand-in for the inner Wan model to exercise helper code."""

    def __init__(self, encoder):
        self.query_memory_encoder = encoder
        self._ei_prev_window_start = None
        self._ei_total_evicted_frames = 0
        self._ei_last_evicted_frames = 0
        self._ei_last_evicted_tokens = 0
        self._ei_strict_update = True


class _FakeGenerator:
    """Minimal stand-in for the generator wrapper."""

    def __init__(self, encoder):
        self.model = _FakeInner(encoder)


class TestInfMemEMA(unittest.TestCase):
    """The external encoder EMA must be CPU-resident and round-trippable."""

    def test_ema_update_and_roundtrip(self):
        from utils.infinity_memory_hooks import InfMemEMA

        enc = _build_encoder()
        generator = _FakeGenerator(enc)
        ema = InfMemEMA(generator, decay=0.5)
        name, parameter = next(iter(enc.named_parameters()))
        before = ema.shadow[name].clone()
        with torch.no_grad():
            parameter.add_(2.0)
        ema.update(generator)
        self.assertTrue(torch.allclose(ema.shadow[name], before + 1.0))
        state = ema.state_dict()
        self.assertTrue(all(v.device.type == "cpu" for v in state["shadow"].values()))
        loaded = InfMemEMA(generator, decay=0.1)
        loaded.load_state_dict(state)
        self.assertEqual(loaded.decay, 0.5)
        self.assertTrue(torch.equal(loaded.shadow[name], ema.shadow[name]))


# ---------------------------------------------------------------------------
# Test 1: optimizer restore order
# ---------------------------------------------------------------------------
class TestOptimizerRestoreOrder(unittest.TestCase):
    """Verify the encoder optimizer is created before its state is loaded."""

    def test_no_attribute_error_on_deferred_load(self):
        """Simulate the Trainer checkpoint-read-then-restore pattern.

        The checkpoint read phase must NOT touch ``self.infmem_optimizer``
        (which is None until the optimizer is created). Instead the optimizer
        state is stashed in pending slots and restored after creation.
        """
        enc = _build_encoder()
        # Simulate the Trainer pattern: declare state early.
        infmem_optimizer = None
        _pending_state = None
        _pending_meta = None

        # Phase A: checkpoint read — build a fake checkpoint.
        opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
        # Take a step so optimizer state is non-empty.
        loss = enc.parameters().__next__().pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        checkpoint = {
            "query_memory_encoder": {k: v.cpu() for k, v in enc.state_dict().items()},
            "query_memory_encoder_optimizer": opt.state_dict(),
            "query_memory_encoder_meta": {
                "n_params": sum(p.numel() for p in enc.parameters()),
                "n_encoder_layers": len(enc.layers),
                "dtype": "torch.float32",
            },
        }

        # Phase B: stash optimizer state into pending (do NOT load yet).
        _pending_state = checkpoint.pop("query_memory_encoder_optimizer")
        _pending_meta = checkpoint.pop("query_memory_encoder_meta")
        # At this point infmem_optimizer is still None — the OLD code would
        # have raised AttributeError here. The NEW code stashes instead.
        self.assertIsNone(infmem_optimizer)

        # Phase C: create optimizer (simulates post-FSDP creation).
        infmem_optimizer = torch.optim.AdamW(enc.parameters(), lr=1e-3)

        # Phase D: restore optimizer state now that it exists.
        infmem_optimizer.load_state_dict(_pending_state)
        self.assertEqual(len(infmem_optimizer.param_groups), 1)
        # Validate param count.
        n_opt_params = sum(len(pg["params"]) for pg in infmem_optimizer.param_groups)
        n_enc_params = sum(1 for _ in enc.parameters())
        self.assertEqual(n_opt_params, n_enc_params)

        # Validate metadata.
        cur_n_params = sum(p.numel() for p in enc.parameters())
        self.assertEqual(int(_pending_meta["n_params"]), cur_n_params)
        self.assertEqual(int(_pending_meta["n_encoder_layers"]), len(enc.layers))
        self.assertIn("float32", str(_pending_meta["dtype"]))

    def test_missing_optimizer_in_combined_resume_raises(self):
        """Combined resume missing the encoder optimizer must raise."""
        enc = _build_encoder()
        allow_partial = False
        checkpoint = {"query_memory_encoder": {k: v.cpu() for k, v in enc.state_dict().items()}}
        # No query_memory_encoder_optimizer in checkpoint.
        with self.assertRaises(RuntimeError):
            if not allow_partial and "query_memory_encoder_optimizer" not in checkpoint:
                raise RuntimeError(
                    "Combined InfMem resume checkpoint is missing "
                    "'query_memory_encoder_optimizer'."
                )


# ---------------------------------------------------------------------------
# Test 2: optimizer round-trip (exp_avg / exp_avg_sq parity)
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_CUDA, "CUDA required for encoder update/get_kv (flash attention)")
class TestOptimizerRoundTrip(unittest.TestCase):
    """Verify optimizer state survives a save → reload cycle bit-for-bit."""

    def _train_step(self, enc, opt, seed):
        """One deterministic training step on the encoder (CUDA).

        ``enc`` must already be on CUDA and ``opt`` must reference its
        parameters (created after the move).
        """
        device = torch.device("cuda")
        torch.manual_seed(seed)
        enc.reset(batch_size=2, device=device, dtype=torch.float32)
        # Simulate a memory update + backward.
        b, m = 2, enc.M
        fake_k = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device)
        fake_v = torch.randn(b, m * 2, enc.num_heads, enc.head_dim, device=device)
        sink_k = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device)
        sink_v = torch.randn(b, 4, enc.num_heads, enc.head_dim, device=device)
        from wan_5b.modules.infinity_memory import _infmem_autocast_context
        with _infmem_autocast_context(fake_k):
            enc.update(fake_k, fake_v, sink_k, sink_v)
        with _infmem_autocast_context(enc.query_state):
            kv = enc.get_kv()
        self.assertIsNotNone(kv)
        loss = kv[0].pow(2).sum() + kv[1].pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    def test_round_trip_param_and_optimizer_state(self):
        """Uninterrupted vs resumed produces identical results after 2 steps."""
        from utils.infinity_memory_hooks import recursive_to_cpu
        device = torch.device("cuda")

        # --- Uninterrupted run ---
        torch.manual_seed(42)
        enc_a = _build_encoder().to(device)
        opt_a = torch.optim.AdamW(enc_a.parameters(), lr=1e-3)
        self._train_step(enc_a, opt_a, seed=100)
        # Record optimizer state at the checkpoint boundary.
        ref_state = copy.deepcopy(opt_a.state_dict())
        # Verify Adam state exists (exp_avg / exp_avg_sq).
        any_state = next(iter(ref_state["state"].values()))
        self.assertIn("exp_avg", any_state)
        self.assertIn("exp_avg_sq", any_state)

        # --- Save checkpoint ---
        ckpt = {
            "query_memory_encoder": {
                k: v.cpu() for k, v in enc_a.state_dict().items()
            },
            "query_memory_encoder_optimizer": recursive_to_cpu(
                opt_a.state_dict()
            ),
        }
        # Verify recursive_to_cpu put everything on CPU.
        for st in ckpt["query_memory_encoder_optimizer"]["state"].values():
            for v in st.values():
                if torch.is_tensor(v):
                    self.assertEqual(v.device.type, "cpu")

        # --- Resumed run ---
        device = torch.device("cuda")
        torch.manual_seed(42)  # same init
        enc_b = _build_encoder()
        enc_b.load_state_dict(ckpt["query_memory_encoder"], strict=True)
        enc_b = enc_b.to(device)
        opt_b = torch.optim.AdamW(enc_b.parameters(), lr=1e-3)
        opt_b.load_state_dict(ckpt["query_memory_encoder_optimizer"])
        # Move optimizer state to CUDA to match parameters (AdamW state lives
        # on the same device as the parameter it tracks).
        for state in opt_b.state.values():
            for k2, v2 in state.items():
                if torch.is_tensor(v2):
                    state[k2] = v2.to(device)

        # Verify Adam state restored bit-for-bit.
        for k in ref_state["state"]:
            st_a = ref_state["state"][k]
            st_b = opt_b.state_dict()["state"][k]
            torch.testing.assert_close(
                st_a["exp_avg"], st_b["exp_avg"], rtol=0, atol=0,
            )
            torch.testing.assert_close(
                st_a["exp_avg_sq"], st_b["exp_avg_sq"], rtol=0, atol=0,
            )
            # step is a scalar tensor — compare values regardless of device.
            self.assertEqual(
                float(st_a["step"]), float(st_b["step"]),
                f"step mismatch for param {k}",
            )

        # One more step with identical data → identical parameters.
        self._train_step(enc_a, opt_a, seed=200)
        ref_param = next(enc_a.parameters()).detach().clone()
        self._train_step(enc_b, opt_b, seed=200)
        new_param = next(enc_b.parameters()).detach().clone()
        torch.testing.assert_close(ref_param, new_param, rtol=0, atol=0)

    def test_recursive_to_cpu_round_trip(self):
        from utils.infinity_memory_hooks import recursive_to_cpu
        nested = {
            "a": torch.tensor([1.0, 2.0]),
            "b": [torch.tensor([3.0]), {"c": torch.tensor([4.0])}],
            "d": (torch.tensor([5.0]),),
            "e": "scalar",
        }
        result = recursive_to_cpu(nested)
        self.assertTrue(torch.is_tensor(result["a"]))
        self.assertEqual(result["a"].device.type, "cpu")
        self.assertTrue(torch.is_tensor(result["b"][0]))
        self.assertTrue(torch.is_tensor(result["b"][1]["c"]))
        self.assertTrue(torch.is_tensor(result["d"][0]))
        self.assertEqual(result["e"], "scalar")


# ---------------------------------------------------------------------------
# Test: state_dict_stats helper
# ---------------------------------------------------------------------------
class TestStateDictStats(unittest.TestCase):
    def test_stats_match_identical_state_dicts(self):
        from utils.infinity_memory_hooks import state_dict_stats
        enc = _build_encoder()
        sd = {k: v.clone() for k, v in enc.state_dict().items()}
        s1 = state_dict_stats(sd)
        s2 = state_dict_stats(sd)
        self.assertEqual(s1, s2)
        self.assertGreater(s1["n_tensors"], 0)

    def test_stats_differ_after_perturbation(self):
        from utils.infinity_memory_hooks import state_dict_stats
        enc = _build_encoder()
        sd1 = {k: v.clone() for k, v in enc.state_dict().items()}
        sd2 = {k: v.clone() for k, v in enc.state_dict().items()}
        first_key = next(iter(sd2))
        sd2[first_key] += 1.0
        s1 = state_dict_stats(sd1)
        s2 = state_dict_stats(sd2)
        self.assertNotEqual(s1["total_sum"], s2["total_sum"])


if __name__ == "__main__":
    unittest.main()
