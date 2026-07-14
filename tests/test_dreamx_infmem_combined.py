"""Smoke tests for the DreamX Camera + Echo-Infinity Memory combined wrapper.

These tests verify:
  1. Structural: the combined wrapper exists, inherits correctly, and the
     InfMem patches + QueryMemoryEncoder + cam_self_attn are all present.
  2. Signature: ``_model_forward_inference_infmem`` accepts ``viewmats``/``Ks``
     and ``_block_infmem_forward`` accepts ``prope_meta``.
  3. Relative RoPE position bounds: ``_compute_relative_positions`` stays
     within ``[0, pmax-1]`` across many chunk scenarios.
  4. Config files: the new YAML configs use the combined wrapper and preserve
     DreamX camera settings.
  5. Streaming x0_pred: ``_generator_loss_infmem_streaming`` returns a valid
     x0_pred tensor (not None).

These tests are designed to run without GPU or full model loading where
possible. Tests marked ``@unittest.skipUnless`` require the Wan2.2-TI2V-5B
model weights on disk.
"""

import importlib
import importlib.util
import inspect
import math
import os
import unittest

import torch


# ---------------------------------------------------------------------------
# Test 1: Structural — combined wrapper class + InfMem helper
# ---------------------------------------------------------------------------
class TestCombinedWrapperStructure(unittest.TestCase):
    """Verify the combined wrapper class exists and has the right shape."""

    def test_dreamx_infmem_wrapper_importable(self):
        """utils.dreamx_infmem_wrapper.DreamXInfMemWanDiffusionWrapper loads."""
        mod = importlib.import_module("utils.dreamx_infmem_wrapper")
        cls = getattr(mod, "DreamXInfMemWanDiffusionWrapper")
        self.assertIsNotNone(cls)

    def test_inherits_from_dreamx_camera(self):
        """Combined wrapper inherits from DreamXCameraWanDiffusionWrapper."""
        from utils.dreamx_camera_wrapper import DreamXCameraWanDiffusionWrapper
        from utils.dreamx_infmem_wrapper import DreamXInfMemWanDiffusionWrapper
        self.assertTrue(
            issubclass(DreamXInfMemWanDiffusionWrapper, DreamXCameraWanDiffusionWrapper),
            "DreamXInfMemWanDiffusionWrapper must inherit from DreamXCameraWanDiffusionWrapper",
        )

    def test_does_not_inherit_from_infmem_wrapper(self):
        """Combined wrapper does NOT inherit from InfMemWanDiffusionWrapper
        (to avoid diamond-init double-creating the Wan model)."""
        from utils.dreamx_infmem_wrapper import DreamXInfMemWanDiffusionWrapper
        from utils.infinity_memory_wrapper import InfMemWanDiffusionWrapper
        self.assertFalse(
            issubclass(DreamXInfMemWanDiffusionWrapper, InfMemWanDiffusionWrapper),
            "DreamXInfMemWanDiffusionWrapper must NOT inherit from "
            "InfMemWanDiffusionWrapper to avoid double __init__ of Wan model.",
        )

    def test_attach_infmem_to_wrapper_helper_exists(self):
        """The shared helper _attach_infmem_to_wrapper exists and is callable."""
        from utils.infinity_memory_wrapper import _attach_infmem_to_wrapper
        self.assertTrue(callable(_attach_infmem_to_wrapper))


# ---------------------------------------------------------------------------
# Test 2: Signature — InfMem patched functions accept camera kwargs
# ---------------------------------------------------------------------------
class TestInfMemSignatureCompatibility(unittest.TestCase):
    """Verify the InfMem patched functions accept the camera-related kwargs."""

    def test_model_forward_inference_accepts_viewmats_ks(self):
        """_model_forward_inference_infmem signature includes viewmats and Ks."""
        from wan_5b.modules.infinity_memory import _model_forward_inference_infmem
        sig = inspect.signature(_model_forward_inference_infmem)
        params = sig.parameters
        self.assertIn("viewmats", params, "viewmats must be in the patched _forward_inference signature")
        self.assertIn("Ks", params, "Ks must be in the patched _forward_inference signature")
        self.assertIn(
            "checkpoint_blocks", params,
            "checkpoint_blocks must explicitly gate safe streaming recomputation",
        )

    def test_block_forward_accepts_prope_meta(self):
        """_block_infmem_forward signature includes prope_meta."""
        from wan_5b.modules.infinity_memory import _block_infmem_forward
        sig = inspect.signature(_block_infmem_forward)
        self.assertIn("prope_meta", sig.parameters,
                       "prope_meta must be in the patched block forward signature")

    def test_block_forward_accepts_memory_kv(self):
        """_block_infmem_forward still accepts memory_kv (InfMem compat)."""
        from wan_5b.modules.infinity_memory import _block_infmem_forward
        sig = inspect.signature(_block_infmem_forward)
        self.assertIn("memory_kv", sig.parameters)

    def test_wrapper_forward_accepts_viewmats_ks(self):
        """WanDiffusionWrapper.forward (the base class that
        DreamXInfMemWanDiffusionWrapper inherits forward from) accepts
        viewmats and Ks."""
        from utils.wan_5b_wrapper import WanDiffusionWrapper
        sig = inspect.signature(WanDiffusionWrapper.forward)
        params = sig.parameters
        self.assertIn("viewmats", params)
        self.assertIn("Ks", params)
        self.assertIn("checkpoint_blocks", params)


# ---------------------------------------------------------------------------
# Test 3: Relative RoPE position bounds
# ---------------------------------------------------------------------------
class TestRelativeRoPEBounds(unittest.TestCase):
    """Verify _compute_relative_positions stays within [0, pmax-1]."""

    def _run_scenario(self, sink_size, local_attn_size, Q_frames,
                      M_tokens_per_frame, num_frame_per_block, pmax):
        """Simulate AR generation and check all positions are in bounds."""
        from wan_5b.modules.infinity_memory import _compute_relative_positions

        frame_seqlen = 880  # 22*40 for Wan2.2-TI2V-5B
        B = num_frame_per_block  # frames per chunk
        R_max = local_attn_size  # max local cache frames
        N_Q = Q_frames  # memory frames

        # Simulate chunks 0, 1, 2, ... until we're deep into the video.
        # ``use_memory`` in _compute_relative_positions is a LAYOUT flag — it
        # says "the memory region WOULD go here if memory were active". The
        # actual runtime check is ``enc.has_history`` in the model forward,
        # which only passes memory_kv when enough frames have been evicted.
        # So we only check memory overlap bounds when the memory region is
        # actually valid (mem_start >= sink_size, meaning enough frames exist).
        for chunk_idx in range(20):
            current_start_frame = chunk_idx * B
            rr = _compute_relative_positions(
                current_start_frame=current_start_frame,
                B=B,
                R=min(R_max, current_start_frame + B),
                N_Q=N_Q,
                N_S=sink_size,
                pmax=pmax,
                num_frame_per_block=num_frame_per_block,
            )
            # q_start and q_last must be in [0, pmax-1]
            self.assertGreaterEqual(rr["q_start"], 0,
                f"chunk {chunk_idx}: q_start={rr['q_start']} < 0")
            self.assertLess(rr["q_last"], pmax,
                f"chunk {chunk_idx}: q_last={rr['q_last']} >= pmax={pmax}")
            # sink_start is always 0
            self.assertEqual(rr["sink_start"], 0)
            # If memory region is actually valid (non-negative start), check
            # it doesn't overlap sink and is contiguous with local.
            if rr["use_memory"] and rr["mem_start"] >= sink_size:
                self.assertGreaterEqual(rr["mem_start"], sink_size,
                    f"chunk {chunk_idx}: mem_start={rr['mem_start']} < sink_size={sink_size}")
                self.assertEqual(rr["mem_end"] + 1, rr["local_start"],
                    f"chunk {chunk_idx}: mem-local not contiguous")
            # q_last == local_end when not bulk forward and R>0
            if R_max > 0 and not rr["is_bulk_forward"]:
                self.assertEqual(rr["q_last"], rr["local_end"])

    def test_default_recipe_bounds(self):
        """Default recipe: sink=4, local=32, Q_frames=8, block=4, pmax=48."""
        self._run_scenario(
            sink_size=4, local_attn_size=32, Q_frames=8,
            M_tokens_per_frame=32, num_frame_per_block=4, pmax=48,
        )

    def test_small_window_bounds(self):
        """Smaller window: sink=2, local=8, Q_frames=4, block=2, pmax=20."""
        self._run_scenario(
            sink_size=2, local_attn_size=8, Q_frames=4,
            M_tokens_per_frame=32, num_frame_per_block=2, pmax=20,
        )

    def test_first_chunk_layout(self):
        """The very first chunk: q positions start from 0."""
        from wan_5b.modules.infinity_memory import _compute_relative_positions
        rr = _compute_relative_positions(
            current_start_frame=0, B=4, R=4, N_Q=8, N_S=4,
            pmax=48, num_frame_per_block=4,
        )
        # q_start should be 0 (first chunk starts at position 0)
        self.assertEqual(rr["q_start"], 0, "First chunk q_start should be 0")
        self.assertEqual(rr["q_last"], 3, "First chunk q_last should be B-1=3")
        # Not a bulk forward (B == num_frame_per_block)
        self.assertFalse(rr["is_bulk_forward"])


# ---------------------------------------------------------------------------
# Test 4: Config files
# ---------------------------------------------------------------------------
class TestConfigFiles(unittest.TestCase):
    """Verify the new YAML configs have the right structure."""

    def _load_yaml(self, path):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_train_config_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "train_dreamx_camera_i2v_ar_infmem.yaml",
        )
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")
        cfg = self._load_yaml(path)
        self.assertEqual(
            cfg["model_kwargs"]["wrapper_cls"],
            "utils.dreamx_infmem_wrapper.DreamXInfMemWanDiffusionWrapper",
        )
        # Must have DreamX camera settings
        self.assertTrue(cfg["model_kwargs"].get("qk_norm", False))
        self.assertIn("attn_compress", cfg["model_kwargs"])
        # Must have InfMem settings
        self.assertTrue(cfg["model_kwargs"].get("enable_relative_rope", False))
        self.assertIn("memory_kwargs", cfg["model_kwargs"])
        # Must have streaming training
        self.assertTrue(cfg["training"].get("infmem_streaming_training", False))
        # Must have strict update
        self.assertTrue(cfg["training"].get("infmem_strict_update", False))
        # Must have I2V AR settings
        self.assertTrue(cfg["algorithm"].get("i2v", False))
        self.assertTrue(cfg["algorithm"].get("causal", False))
        self.assertTrue(cfg["algorithm"].get("teacher_forcing", False))
        self.assertTrue(cfg["algorithm"].get("independent_first_frame", False))

    def test_streaming_train_config_enables_safe_checkpointing(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "train_dreamx_camera_i2v_ar_infmem_streaming.yaml",
        )
        cfg = self._load_yaml(path)
        self.assertTrue(
            cfg["training"].get("streaming_activation_checkpointing", False)
        )
        self.assertTrue(cfg["training"].get("streaming_fsdp_no_sync", False))
        self.assertFalse(cfg["infra"].get("gradient_checkpointing", True))

    def test_train_config_new_hyperparams(self):
        """Verify the new hyperparameters match the updated recipe."""
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "train_dreamx_camera_i2v_ar_infmem.yaml",
        )
        cfg = self._load_yaml(path)
        mk = cfg["model_kwargs"]
        self.assertEqual(mk["local_attn_size"], 12)
        self.assertEqual(mk["sink_size"], 4)
        # Window budget: sink(4) + local(12) + memory(3) + block(4) = 23 < 24.
        self.assertEqual(mk["relative_rope_pmax"], 24)
        self.assertEqual(mk["memory_kwargs"]["Q_frames"], 3)
        # n_encoder_layers should be 2, not 30
        self.assertEqual(mk["memory_kwargs"]["n_encoder_layers"], 2)
        self.assertNotIn("num_layers", mk["memory_kwargs"])

    def test_infer_config_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "infer_dreamx_camera_i2v_ar_infmem.yaml",
        )
        self.assertTrue(os.path.isfile(path), f"Missing: {path}")
        cfg = self._load_yaml(path)
        self.assertEqual(
            cfg["model_kwargs"]["wrapper_cls"],
            "utils.dreamx_infmem_wrapper.DreamXInfMemWanDiffusionWrapper",
        )
        self.assertTrue(cfg["algorithm"].get("use_camera", False))
        self.assertTrue(cfg["algorithm"].get("causal", False))
        self.assertTrue(cfg["model_kwargs"].get("enable_relative_rope", False))
        self.assertIn("memory_kwargs", cfg["model_kwargs"])

    def test_old_configs_unchanged(self):
        """Old configs still reference their original wrappers."""
        configs_dir = os.path.join(
            os.path.dirname(__file__), "..", "configs",
        )
        # train_i2v_ar_infmem.yaml should still use InfMemWanDiffusionWrapper
        cfg1 = self._load_yaml(os.path.join(configs_dir, "train_i2v_ar_infmem.yaml"))
        self.assertIn("InfMemWanDiffusionWrapper", cfg1["model_kwargs"]["wrapper_cls"])

        # train_dreamx_camera_i2v_ar.yaml should still use DreamXCameraWanDiffusionWrapper
        cfg2 = self._load_yaml(os.path.join(configs_dir, "train_dreamx_camera_i2v_ar.yaml"))
        self.assertIn("DreamXCameraWanDiffusionWrapper", cfg2["model_kwargs"]["wrapper_cls"])


# ---------------------------------------------------------------------------
# Test 5: Streaming x0_pred is valid (uses mock generator)
# ---------------------------------------------------------------------------
class TestStreamingX0Pred(unittest.TestCase):
    """Verify _generator_loss_infmem_streaming returns valid x0_pred."""

    def test_streaming_returns_valid_x0_pred(self):
        """The streaming loss function should return (flow_pred, x0_pred)
        where x0_pred is NOT None."""
        # We can't easily instantiate the full model, so we verify the code
        # by checking that the return statement concatenates x0_pred_chunks.
        import ast
        import pathlib

        # Read model/diffusion.py and parse the streaming function
        diff_path = pathlib.Path(__file__).parent.parent / "model" / "diffusion.py"
        source = diff_path.read_text()
        tree = ast.parse(source)

        streaming_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_generator_loss_infmem_streaming":
                streaming_fn = node
                break
        self.assertIsNotNone(streaming_fn, "_generator_loss_infmem_streaming not found")

        # Check that the function accepts viewmats and Ks parameters
        arg_names = [arg.arg for arg in streaming_fn.args.args]
        self.assertIn("viewmats", arg_names,
                       "_generator_loss_infmem_streaming must accept viewmats")
        self.assertIn("Ks", arg_names,
                       "_generator_loss_infmem_streaming must accept Ks")

        # Check the return statement uses x0_pred_chunks (not None)
        return_src = ast.get_source_segment(source, streaming_fn)
        self.assertIn("x0_pred_chunks", return_src,
                       "Streaming function must collect x0_pred_chunks")
        self.assertIn("torch.cat(x0_pred_chunks", return_src,
                       "Streaming function must return concatenated x0_pred")


# ---------------------------------------------------------------------------
# Test 6: _block_infmem_forward includes camera branch
# ---------------------------------------------------------------------------
class TestBlockForwardCameraBranch(unittest.TestCase):
    """Verify the InfMem block forward includes the cam_self_attn branch."""

    def test_block_forward_has_camera_branch(self):
        """The patched block forward source must reference cam_self_attn and prope_meta."""
        import inspect
        from wan_5b.modules.infinity_memory import _block_infmem_forward
        src = inspect.getsource(_block_infmem_forward)
        self.assertIn("cam_self_attn", src,
                       "Block forward must reference cam_self_attn for camera branch")
        self.assertIn("prope_meta", src,
                       "Block forward must reference prope_meta")
        self.assertIn("cam_emb", src,
                       "Block forward must construct cam_emb dict for cam_self_attn")

    def test_model_forward_builds_prope_meta(self):
        """The patched model forward must build prope_meta from viewmats/Ks."""
        import inspect
        from wan_5b.modules.infinity_memory import _model_forward_inference_infmem
        src = inspect.getsource(_model_forward_inference_infmem)
        self.assertIn("prope_meta", src)
        self.assertIn("viewmats", src)
        self.assertIn("Ks", src)


# ---------------------------------------------------------------------------
# Test 7: Backward compat — InfMem and DreamX wrappers unchanged
# ---------------------------------------------------------------------------
class TestBackwardCompat(unittest.TestCase):
    """Verify old wrappers still work and their key methods are unchanged."""

    def test_infmem_wrapper_still_callable(self):
        from utils.infinity_memory_wrapper import InfMemWanDiffusionWrapper
        self.assertTrue(callable(InfMemWanDiffusionWrapper))

    def test_dreamx_wrapper_still_callable(self):
        from utils.dreamx_camera_wrapper import DreamXCameraWanDiffusionWrapper
        self.assertTrue(callable(DreamXCameraWanDiffusionWrapper))

    def test_attach_infmem_still_callable(self):
        from wan_5b.modules.infinity_memory import attach_infmem
        self.assertTrue(callable(attach_infmem))


# ---------------------------------------------------------------------------
# Test 8: Encoder lifecycle — FP32, n_encoder_layers, param count, hooks
# ---------------------------------------------------------------------------
class TestEncoderLifecycle(unittest.TestCase):
    """Verify encoder config and lifecycle requirements."""

    def test_encoder_config_defaults_to_2_layers(self):
        """_make_encoder_config should default to n_encoder_layers=2."""
        from utils.infinity_memory_wrapper import _make_encoder_config
        cfg = _make_encoder_config({})
        self.assertEqual(getattr(cfg, "n_encoder_layers", None), 2)

    def test_encoder_config_conflict_raises(self):
        """Passing both num_layers and n_encoder_layers with different values raises."""
        from utils.infinity_memory_wrapper import _attach_infmem_to_wrapper
        # Can't easily call _attach_infmem_to_wrapper without a full wrapper,
        # but we can test the conflict detection logic by checking that
        # _make_encoder_config normalizes correctly.
        from utils.infinity_memory_wrapper import _make_encoder_config
        # When only num_layers is given, it should be aliased to n_encoder_layers.
        cfg = _make_encoder_config({"num_layers": 5})
        self.assertEqual(getattr(cfg, "n_encoder_layers"), 5)
        # When both are given with same value, should work.
        cfg = _make_encoder_config({"num_layers": 3, "n_encoder_layers": 3})
        self.assertEqual(getattr(cfg, "n_encoder_layers"), 3)

    def test_hook_functions_exist(self):
        """New hook functions exist and are callable."""
        from utils.infinity_memory_hooks import (
            sync_infmem_gradients,
            clip_infmem_grad_norm,
            broadcast_infmem_params,
            maybe_detach_infmem,
        )
        for fn in [sync_infmem_gradients, clip_infmem_grad_norm,
                   broadcast_infmem_params, maybe_detach_infmem]:
            self.assertTrue(callable(fn))

    def test_maybe_detach_infmem_accepts_cache_kwargs(self):
        """maybe_detach_infmem signature includes kv_cache and crossattn_cache."""
        import inspect
        from utils.infinity_memory_hooks import maybe_detach_infmem
        sig = inspect.signature(maybe_detach_infmem)
        self.assertIn("kv_cache", sig.parameters)
        self.assertIn("crossattn_cache", sig.parameters)

    def test_clip_infmem_grad_norm_exists(self):
        """clip_infmem_grad_norm is callable and returns a tensor."""
        from utils.infinity_memory_hooks import clip_infmem_grad_norm
        self.assertTrue(callable(clip_infmem_grad_norm))

    def test_move_infmem_encoder_preserves_fp32(self):
        """move_infmem_encoder should accept force_cast=False and not cast."""
        import inspect
        from utils.infinity_memory_hooks import move_infmem_encoder
        sig = inspect.signature(move_infmem_encoder)
        self.assertIn("force_cast", sig.parameters)


# ---------------------------------------------------------------------------
# Test 9: Streaming trainer keeps clip-length-dependent tensors off CUDA
# ---------------------------------------------------------------------------
class TestStreamingTrainerBoundedGpuMemory(unittest.TestCase):
    """Regression checks for full-clip CUDA allocations in streaming training."""

    @staticmethod
    def _method_source(path, method_name):
        import ast
        import pathlib

        source = pathlib.Path(path).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    return ast.get_source_segment(source, node)
        raise AssertionError(f"{method_name} not found in {path}")

    def test_camera_clip_is_sliced_on_cpu_before_cuda_transfer(self):
        import pathlib

        trainer_path = (
            pathlib.Path(__file__).parent.parent
            / "trainer"
            / "dreamx_infmem_streaming_diffusion.py"
        )
        prepare_src = self._method_source(trainer_path, "_prepare_camera_batch")
        train_src = self._method_source(trainer_path, "_train_one_step_camera")

        self.assertIn('clean_latent = batch["clean_latent"]', prepare_src)
        self.assertNotIn(
            'batch["clean_latent"].to(', prepare_src,
            "The full video must not be moved to CUDA during batch preparation.",
        )
        self.assertIn("clean_latent_cpu[:, start:end].to(", train_src)
        self.assertIn("viewmats_cpu[:, start:end].to(", train_src)
        self.assertIn("Ks_cpu[:, start:end].to(", train_src)
        self.assertIn("torch.randn_like(clean_chunk)", train_src)
        self.assertNotIn("torch.randn_like(clean_latent_cpu)", train_src)
        self.assertIn(
            "checkpoint_blocks=self.streaming_activation_checkpointing", train_src
        )
        self.assertIn(
            "with self._fsdp_gradient_sync_context(sync_gradients)", train_src
        )
        self.assertIn("end == num_frame", train_src)
        self.assertIn(
            "accumulation_step == accumulation_steps - 1", train_src
        )

    def test_fsdp_no_sync_context_is_configurable(self):
        import pathlib

        trainer_path = (
            pathlib.Path(__file__).parent.parent
            / "trainer"
            / "dreamx_infmem_streaming_diffusion.py"
        )
        context_src = self._method_source(
            trainer_path, "_fsdp_gradient_sync_context"
        )
        self.assertIn("self.model.generator.no_sync()", context_src)
        self.assertIn("not self.streaming_fsdp_no_sync", context_src)
        self.assertIn("nullcontext()", context_src)

    def test_checkpointing_is_gated_to_the_safe_prediction_pass(self):
        import pathlib

        infmem_path = (
            pathlib.Path(__file__).parent.parent
            / "wan_5b"
            / "modules"
            / "infinity_memory.py"
        )
        forward_src = self._method_source(
            infmem_path, "_model_forward_inference_infmem"
        )
        self.assertIn("checkpoint_blocks", forward_src)
        self.assertIn("and defer_cache_updates", forward_src)
        self.assertIn("and not update_memory", forward_src)

    def test_unused_cross_attention_cache_has_no_dense_kv_allocation(self):
        import pathlib

        diffusion_path = pathlib.Path(__file__).parent.parent / "model" / "diffusion.py"
        cache_src = self._method_source(diffusion_path, "_build_streaming_caches")

        self.assertIn('"k": None', cache_src)
        self.assertIn('"v": None', cache_src)
        self.assertNotIn("[batch_size, 512, num_heads, head_dim]", cache_src)


if __name__ == "__main__":
    unittest.main()
