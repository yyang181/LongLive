import inspect
import os
import types
import unittest

import torch
from omegaconf import OmegaConf

from trainer.dreamx_infmem_streaming_diffusion import Trainer
from utils.config import normalize_config


class TestARLongInfMemConfig(unittest.TestCase):
    def _config(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            "train_dreamx_camera_i2v_ar_long_infmem.yaml",
        )
        return normalize_config(OmegaConf.load(path))

    def test_recipe_is_supervised_ar_not_dmd(self):
        config = self._config()
        self.assertEqual(
            config.trainer, "dreamx_infmem_streaming_diffusion"
        )
        self.assertTrue(config.teacher_forcing)
        self.assertNotIn("distribution_loss", config)
        self.assertNotIn("real_model_kwargs", config)
        self.assertNotIn("fake_model_kwargs", config)

    def test_recipe_uses_prediction_recache_and_rank_256(self):
        config = self._config()
        self.assertEqual(config.streaming_cache_source, "prediction")
        self.assertEqual(config.streaming_bounded_window_size, 20)
        self.assertEqual(config.adapter.rank, 256)
        self.assertEqual(config.adapter.alpha, 256)
        self.assertEqual(config.model_kwargs.local_attn_size, 12)
        self.assertEqual(config.model_kwargs.sink_size, 4)
        self.assertTrue(config.model_kwargs.enable_relative_rope)
        self.assertFalse(config.streaming_multistep_supervision)
        self.assertIsNone(config.streaming_multistep_sampling_steps)

    def test_cache_base_is_x0_prediction(self):
        trainer = Trainer.__new__(Trainer)
        trainer.streaming_cache_source = "prediction"
        x0_pred = torch.randn(1, 4, 2, 1, 1, requires_grad=True)
        cache_base = trainer._select_streaming_cache_base(x0_pred)
        self.assertTrue(torch.equal(cache_base, x0_pred.detach()))
        self.assertFalse(cache_base.requires_grad)
        self.assertNotEqual(cache_base.data_ptr(), x0_pred.data_ptr())

    def test_bounded_windows_follow_global_rollout_position(self):
        windows = []
        cursor = 0
        while cursor < 240:
            start, end = Trainer._next_bounded_window(cursor, 240, 20)
            windows.append((start, end))
            cursor = end
        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0], (0, 20))
        self.assertEqual(windows[-1], (220, 240))

    def test_gt_is_target_but_not_clean_recache_source(self):
        source = inspect.getsource(Trainer._train_one_step_camera)
        self.assertIn("self.model.scheduler.training_target(", source)
        self.assertIn("self._streaming_multistep_flow_target(", source)
        self.assertIn("_select_streaming_cache_base(x0_pred)", source)
        self.assertNotIn("_select_streaming_cache_base(clean_chunk", source)

    def test_multistep_mode_uses_solver_latent_for_recache(self):
        source = inspect.getsource(Trainer._train_one_step_camera)
        self.assertIn("if self.streaming_multistep_supervision:", source)
        self.assertIn("sample_scheduler.step(", source)
        self.assertIn("x0_pred = noisy_chunk", source)
        self.assertIn("accumulation_steps * num_denoising_steps", source)
        self.assertIn('getattr(encoder, "has_history", False)', source)

    def test_inference_timestep_maps_to_training_grid(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(
                timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0])
            )
        )
        indices = trainer._nearest_training_timestep_index(
            torch.tensor(510.0), batch_size=2, num_frames=4
        )
        self.assertEqual(indices.shape, (2, 4))
        self.assertTrue(torch.equal(indices, torch.full((2, 4), 2)))

    def test_multistep_target_rectifies_current_solver_state(self):
        clean = torch.tensor([[[[[2.0]]]]])
        latent = torch.tensor([[[[[5.0]]]]], requires_grad=True)
        target = Trainer._streaming_multistep_flow_target(
            latent, clean, torch.tensor(500.0), 1000
        )
        self.assertTrue(torch.equal(target, torch.tensor([[[[[6.0]]]]])))
        self.assertFalse(target.requires_grad)


if __name__ == "__main__":
    unittest.main()
