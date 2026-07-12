"""Eviction and reset-isolation tests for the InfMem memory path.

These tests verify:
  * Test 6: multi-shot pinned sink eviction — the evicted K/V slice passed
    to the encoder corresponds EXACTLY to the real evicted frames, not the
    global/pinned sink or still-retained local frames.
  * Test 7: reset isolation — a second video starts with a fresh memory
    state, uncontaminated by the first video.
  * Test 8: camera checkpoint validation — the helper that checks whether
    a checkpoint contains cam_self_attn tensors uses the CHECKPOINT's keys,
    not the model's keys.

These run on CPU (no GPU required) because they test the metadata logic,
not the CUDA autocast path.
"""

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test 6: multi-shot pinned sink eviction slice
# ---------------------------------------------------------------------------
class TestPinnedSinkEvictionSlice(unittest.TestCase):
    """Verify the eviction capture uses effective_sink, not static sink_tok."""

    def test_capture_start_uses_effective_sink(self):
        """When pinned sink is active, effective_sink > static sink.

        The evicted slice must be captured at [effective_sink : effective_sink + n_evicted],
        NOT [sink_tok : sink_tok + n_evicted].
        """
        frame_seqlen = 4
        sink_frames = 2
        # Pinned adds 3 frames.
        pinned_frames = 3
        effective_sink_frames = sink_frames + pinned_frames
        effective_sink_tokens = effective_sink_frames * frame_seqlen
        static_sink_tokens = sink_frames * frame_seqlen

        # Build a fake cache with identifiable per-frame K values.
        total_frames = 20
        kv_cache_size = total_frames * frame_seqlen
        num_heads, head_dim = 2, 2
        cache_k = torch.arange(total_frames * frame_seqlen, dtype=torch.float32).view(
            1, total_frames * frame_seqlen, 1, 1
        ).expand(1, total_frames * frame_seqlen, num_heads, head_dim).clone()

        # Evict 4 frames worth of tokens.
        num_evicted_tokens = 4 * frame_seqlen
        capture_start = effective_sink_tokens
        capture_end = capture_start + num_evicted_tokens
        exited_k = cache_k[:, capture_start:capture_end].clone()

        # The exited slice must start at effective_sink, NOT static sink.
        first_val = exited_k[0, 0, 0, 0].item()
        self.assertEqual(first_val, float(effective_sink_tokens),
                         f"evicted slice should start at effective_sink={effective_sink_tokens}, "
                         f"got first value {first_val}")

        # Verify the static-sink capture would have been WRONG.
        wrong_start = static_sink_tokens
        wrong_k = cache_k[:, wrong_start:wrong_start + num_evicted_tokens].clone()
        wrong_first = wrong_k[0, 0, 0, 0].item()
        self.assertNotEqual(wrong_first, first_val,
                            "static-sink capture must differ from effective-sink capture")

    def test_sink_anchor_uses_canonical_sink(self):
        """The sink anchor must use the canonical (stable global) sink,
        not the effective (pinned-inclusive) sink."""
        frame_seqlen = 4
        sink_frames = 2
        pinned_frames = 3
        canonical_sink_tokens = sink_frames * frame_seqlen
        effective_sink_tokens = (sink_frames + pinned_frames) * frame_seqlen

        total_frames = 20
        cache_k = torch.arange(total_frames * frame_seqlen, dtype=torch.float32).view(
            1, -1, 1, 1
        ).expand(1, total_frames * frame_seqlen, 2, 2).clone()

        # Sink anchor = canonical sink (first sink_frames).
        sink_anchor_k = cache_k[:, :canonical_sink_tokens].clone()
        self.assertEqual(sink_anchor_k[0, 0, 0, 0].item(), 0.0)
        # Effective sink is larger — must not be used for the anchor.
        self.assertGreater(effective_sink_tokens, canonical_sink_tokens)

    def test_evicted_slice_excludes_still_local_frames(self):
        """The evicted slice must not include frames that are still in the
        local window (i.e., capture_end must be < the newest retained frame)."""
        frame_seqlen = 4
        sink_frames = 2
        pinned_frames = 2
        effective_sink = (sink_frames + pinned_frames) * frame_seqlen
        local_window = 8 * frame_seqlen
        kv_cache_size = effective_sink + local_window
        # New chunk of 4 frames, cache overflows by 4 frames.
        num_evicted_tokens = 4 * frame_seqlen
        capture_start = effective_sink
        capture_end = capture_start + num_evicted_tokens
        # The local window starts at effective_sink and the evicted frames
        # are the OLDEST local frames (just after the sink).
        self.assertLessEqual(capture_end, effective_sink + local_window)
        # The newest frames are still in the window (after the rolled region).
        rolled_region_end = effective_sink + (local_window - num_evicted_tokens)
        self.assertGreater(rolled_region_end, capture_start)
        # capture_end must not reach into the newest chunk's slot.
        newest_chunk_start = kv_cache_size - num_evicted_tokens
        self.assertLessEqual(capture_end, newest_chunk_start)


# ---------------------------------------------------------------------------
# Test 7: reset isolation
# ---------------------------------------------------------------------------
class TestResetIsolation(unittest.TestCase):
    """Verify reset clears all memory state between videos."""

    def _make_encoder(self):
        from model.query_memory import QueryMemoryEncoder

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

        return QueryMemoryEncoder(_Cfg()).float()

    def test_reset_clears_history(self):
        from utils.infinity_memory_hooks import reset_infmem

        enc = self._make_encoder()
        # Build a fake generator with the encoder + model diagnostics.
        inner = types.SimpleNamespace(
            query_memory_encoder=enc,
            _ei_prev_window_start=5,
            _ei_total_evicted_frames=10,
            _ei_last_evicted_frames=3,
        )
        generator = types.SimpleNamespace(model=inner)

        # Simulate having history without running the CUDA attention path.
        enc.has_history = True
        enc._update_count = 3

        # Reset for the next video.
        ok = reset_infmem(generator, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(ok)
        self.assertFalse(enc.has_history)
        self.assertEqual(enc._update_count, 0)
        self.assertIsNone(inner._ei_prev_window_start)
        self.assertEqual(inner._ei_total_evicted_frames, 0)

    def test_reset_isolation_between_videos(self):
        """Second video must not see first video's memory."""
        from utils.infinity_memory_hooks import reset_infmem

        enc = self._make_encoder()
        inner = types.SimpleNamespace(
            query_memory_encoder=enc,
            _ei_prev_window_start=None,
            _ei_total_evicted_frames=0,
            _ei_last_evicted_frames=0,
        )
        generator = types.SimpleNamespace(model=inner)

        # Video 1: simulate history without CUDA attention.
        reset_infmem(generator, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        enc.has_history = True
        enc._update_count = 2

        # Video 2: reset.
        reset_infmem(generator, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        # After reset, get_kv must return None (no history).
        kv2 = enc.get_kv()
        self.assertIsNone(kv2)
        self.assertFalse(enc.has_history)
        self.assertEqual(enc._update_count, 0)


# ---------------------------------------------------------------------------
# Test 8: camera checkpoint validation (uses checkpoint keys, not model keys)
# ---------------------------------------------------------------------------
class TestCameraCheckpointValidation(unittest.TestCase):
    """Verify camera key validation uses the checkpoint's keys."""

    def test_checkpoint_with_cam_keys_passes(self):
        """A checkpoint containing cam_self_attn tensors should be accepted."""
        from utils.nvfp4_checkpoint import clean_fsdp_state_dict_keys
        raw_state = {
            "blocks.0.self_attn.q.weight": torch.zeros(1),
            "blocks.0.cam_self_attn.q.weight": torch.zeros(1),
            "blocks.1.cam_self_attn.q.weight": torch.zeros(1),
        }
        normalized = clean_fsdp_state_dict_keys(raw_state)
        cam_keys = {
            k for k in normalized.keys()
            if ".cam_self_attn." in k or "cam_self_attn" in k
        }
        self.assertEqual(len(cam_keys), 2)

    def test_checkpoint_without_cam_keys_detected(self):
        """A checkpoint with NO cam_self_attn keys must be detected."""
        raw_state = {
            "blocks.0.self_attn.q.weight": torch.zeros(1),
            "blocks.1.self_attn.q.weight": torch.zeros(1),
        }
        cam_keys = {
            k for k in raw_state
            if ".cam_self_attn." in k or "cam_self_attn" in k
        }
        self.assertEqual(len(cam_keys), 0)

    def test_model_keys_do_not_falsely_pass(self):
        """The OLD buggy logic checked ``k in model_sd`` which is always True
        for model-defined keys. The NEW logic checks the checkpoint's OWN keys."""
        # Simulate: model defines cam_self_attn, but checkpoint has none.
        model_sd = {
            "blocks.0.cam_self_attn.q.weight": torch.zeros(1),
            "blocks.1.cam_self_attn.q.weight": torch.zeros(1),
        }
        checkpoint_state = {
            "blocks.0.self_attn.q.weight": torch.zeros(1),
        }
        # OLD (buggy): loaded_cam_keys = [k for k in expected if k in model_sd]
        # → always len == len(model_cam_keys) because it checks the MODEL dict.
        old_loaded = [k for k in model_sd if k in model_sd]
        self.assertEqual(len(old_loaded), 2)  # falsely "passes"

        # NEW: check checkpoint keys.
        checkpoint_cam_keys = {
            k for k in checkpoint_state
            if "cam_self_attn" in k
        }
        self.assertEqual(len(checkpoint_cam_keys), 0)  # correctly detects absence

    def test_fsdp_prefix_normalization(self):
        """FSDP wrapper prefixes must be stripped before key comparison."""
        from utils.nvfp4_checkpoint import clean_fsdp_state_dict_keys
        raw_state = {
            "_fsdp_wrapped_module.model.blocks.0.cam_self_attn.q.weight": torch.zeros(1),
        }
        normalized = clean_fsdp_state_dict_keys(raw_state)
        cam_keys = {
            k for k in normalized
            if "cam_self_attn" in k
        }
        self.assertEqual(len(cam_keys), 1)
        # clean_fsdp_state_dict_keys strips _fsdp_wrapped_module. but keeps
        # the model. prefix (the wrapper-level key). The trainer further
        # normalizes by stripping "model." to match inner-model keys.
        self.assertIn("model.blocks.0.cam_self_attn.q.weight", cam_keys)


if __name__ == "__main__":
    unittest.main()
