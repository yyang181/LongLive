import pathlib
import unittest
from unittest import mock

import torch

from wan_5b.distributed import sequence_parallel_camera as sp_camera


class _IdentityDreamXAttention(torch.nn.Module):
    def __init__(self, dim=8, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        with torch.no_grad():
            for layer in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.out_proj,
            ):
                layer.weight.copy_(torch.eye(dim))


def _identity_prope(q, k, v, **kwargs):
    return q, k, v, lambda out: out


class DreamXEPropeBalancedSPTest(unittest.TestCase):
    def setUp(self):
        self.attn = _IdentityDreamXAttention()
        self.viewmats = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        self.Ks = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    def test_teacher_forcing_uses_distributed_flex_attention(self):
        # Local natural layout is [clean(6 tokens), noisy(6 tokens)].
        x = torch.randn(1, 12, 8)
        mask = object()
        with (
            mock.patch.object(sp_camera, "get_sp_world_size", return_value=2),
            mock.patch(
                "wan_5b.modules.prope.prope_qkv",
                side_effect=_identity_prope,
            ) as prope,
            mock.patch.object(
                sp_camera,
                "distributed_flex_attention",
                side_effect=lambda q, k, v, block_mask: v,
            ) as distributed,
        ):
            out = sp_camera.sp_dreamx_camera_attn_forward(
                self.attn,
                x,
                {"viewmats": self.viewmats, "K": self.Ks},
                seq_lens=torch.tensor([6]),
                block_mask=mask,
            )

        torch.testing.assert_close(out, x)
        self.assertEqual(prope.call_count, 2)
        distributed.assert_called_once()
        q_arg, _, _, mask_arg = distributed.call_args.args
        self.assertEqual(tuple(q_arg.shape), (1, 12, 4, 2))
        self.assertIs(mask_arg, mask)

    def test_bidirectional_path_uses_global_ulysses_attention(self):
        x = torch.randn(1, 6, 8)
        with (
            mock.patch.object(sp_camera, "get_sp_world_size", return_value=2),
            mock.patch(
                "wan_5b.modules.prope.prope_qkv",
                side_effect=_identity_prope,
            ) as prope,
            mock.patch.object(
                sp_camera,
                "_distributed_attention_with_grad",
                side_effect=lambda q, k, v, seq_lens, window_size: v,
            ) as distributed,
        ):
            out = sp_camera.sp_dreamx_camera_attn_forward(
                self.attn,
                x,
                {"viewmats": self.viewmats, "Ks": self.Ks},
                seq_lens=torch.tensor([6]),
            )

        torch.testing.assert_close(out, x)
        self.assertEqual(prope.call_count, 1)
        distributed.assert_called_once()
        self.assertEqual(distributed.call_args.args[3].tolist(), [12])

    def test_camera_heads_must_divide_sp_size(self):
        attn = _IdentityDreamXAttention(dim=6, num_heads=3)
        with mock.patch.object(sp_camera, "get_sp_world_size", return_value=2):
            with self.assertRaisesRegex(ValueError, "num_heads"):
                sp_camera.sp_dreamx_camera_attn_forward(
                    attn,
                    torch.randn(1, 6, 6),
                    {"viewmats": self.viewmats},
                    seq_lens=torch.tensor([6]),
                )

    def test_sp1_default_config_and_forward_remain_unpatched(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        config_text = (repo / "configs/train_dreamx_camera_i2v_ar.yaml").read_text()
        trainer_text = (repo / "trainer/diffusion.py").read_text()

        self.assertIn("sequence_parallel_size: 1", config_text)
        self.assertIn("if self.sequence_parallel_size > 1:", trainer_text)
        self.assertIn("sp_dreamx_camera_attn_forward", trainer_text)


if __name__ == "__main__":
    unittest.main()
