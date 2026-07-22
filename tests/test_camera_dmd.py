import unittest

import torch
from omegaconf import OmegaConf

import utils.tv_io_patch  # noqa: F401 - restore torchvision.io compatibility
from model.dmd import DMD
from pipeline.self_forcing_training import SelfForcingTrainingPipeline
from utils.config import normalize_config


class _RecordingScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy_image_or_video, **kwargs):
        self.calls.append(kwargs)
        return None, torch.zeros_like(noisy_image_or_video)


class CameraDMDTest(unittest.TestCase):
    def test_training_config_shares_ar_checkpoint_across_camera_roles(self):
        config = OmegaConf.merge(
            OmegaConf.load("configs/train_dreamx_camera_i2v_dmd.yaml"),
            OmegaConf.from_dotlist(["checkpoints.generator_ckpt=/tmp/camera-ar.pt"]),
        )
        config = normalize_config(config)

        self.assertEqual(config.generator_ckpt, "/tmp/camera-ar.pt")
        self.assertEqual(config.real_score_ckpt, config.generator_ckpt)
        self.assertEqual(config.fake_score_ckpt, config.generator_ckpt)
        self.assertTrue(config.backward_simulation)
        self.assertFalse(config.inherit_base_checkpoint_step)
        self.assertTrue(config.i2v)
        self.assertEqual(config.model_kwargs.wrapper_cls,
                         "utils.dreamx_camera_wrapper.DreamXCameraWanDiffusionWrapper")

    def test_camera_chunk_slicing_stays_frame_aligned(self):
        viewmats = torch.arange(2 * 8 * 4 * 4).reshape(2, 8, 4, 4)
        Ks = torch.arange(2 * 8 * 3 * 3).reshape(2, 8, 3, 3)

        camera = SelfForcingTrainingPipeline._camera_kwargs(
            viewmats, Ks, start_frame=4, num_frames=4
        )

        self.assertTrue(torch.equal(camera["viewmats"], viewmats[:, 4:8]))
        self.assertTrue(torch.equal(camera["Ks"], Ks[:, 4:8]))
        with self.assertRaises(ValueError):
            SelfForcingTrainingPipeline._camera_kwargs(
                viewmats, Ks[:, :6], start_frame=4, num_frames=4
            )

    def test_teacher_and_critic_receive_same_camera_inputs(self):
        dmd = object.__new__(DMD)
        torch.nn.Module.__init__(dmd)
        dmd.fake_score = _RecordingScore()
        dmd.real_score = _RecordingScore()
        dmd.fake_guidance_scale = 0.0
        dmd.real_guidance_scale = 3.0

        latent = torch.randn(1, 4, 2, 2, 2)
        viewmats = torch.randn(1, 4, 4, 4)
        Ks = torch.randn(1, 4, 3, 3)
        timestep = torch.ones(1, 4)
        conditional = {"prompt_embeds": torch.randn(1, 3, 4)}

        dmd._compute_kl_grad(
            noisy_image_or_video=latent,
            estimated_clean_image_or_video=latent,
            timestep=timestep,
            conditional_dict=conditional,
            unconditional_dict=conditional,
            normalization=False,
            viewmats=viewmats,
            Ks=Ks,
        )

        calls = dmd.fake_score.calls + dmd.real_score.calls
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertIs(call["viewmats"], viewmats)
            self.assertIs(call["Ks"], Ks)


if __name__ == "__main__":
    unittest.main()
