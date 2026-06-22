# SPDX-License-Identifier: Apache-2.0
"""Camera-aware Wan2.2-TI2V-5B diffusion wrapper for bidirectional + PRoPE SFT.
Wraps WanModel and:
  1) registers per-block zero-init `prope_o` parameters via add_prope_parameters
  2) forwards optional ``viewmats`` / ``Ks`` to the model

Strictly additive: when ``viewmats`` is None, behaves identically to the
plain bidirectional WanDiffusionWrapper.
"""

from typing import List, Optional

import torch

from utils.wan_5b_wrapper import WanDiffusionWrapper


class CameraWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(
        self,
        model_name: str = "Wan2.2-TI2V-5B",
        timestep_shift: float = 5.0,
        is_causal: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        t_scale: float = 1.0,
        rope_method: str = "linear",
        original_seq_len=None,
        use_camera: bool = True,
    ):
        # Camera-PRoPE bidirectional SFT requires a non-causal backbone.
        assert not is_causal, (
            "CameraWanDiffusionWrapper only supports the bidirectional "
            "(non-causal) WanModel; got is_causal=True."
        )
        super().__init__(
            model_name=model_name,
            timestep_shift=timestep_shift,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            t_scale=t_scale,
            rope_method=rope_method,
            original_seq_len=original_seq_len,
        )
        self.use_camera = use_camera
        if self.use_camera:
            from wan_5b.modules.prope import add_prope_parameters
            n_added = add_prope_parameters(self.model, zero_init=True)
            if n_added is not None:
                # Print only on rank 0 if torch.distributed is up.
                try:
                    import torch.distributed as _dist
                    if (not _dist.is_initialized()) or _dist.get_rank() == 0:
                        print(f"[CameraWanDiffusionWrapper] PRoPE parameters "
                              f"added to {n_added} self-attention blocks.")
                except Exception:
                    print(f"[CameraWanDiffusionWrapper] PRoPE parameters "
                          f"added to {n_added} self-attention blocks.")

    def forward(  # type: ignore[override]
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        rope_temporal_offset: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bidirectional forward with optional PRoPE camera conditioning."""
        prompt_embeds = conditional_dict["prompt_embeds"]

        # All frames share the same timestep in bidirectional mode.
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        rope_offset_was_set = (
            rope_temporal_offset is not None
            and hasattr(self.model, "rope_temporal_offset")
        )
        if rope_offset_was_set:
            prev_rope_temporal_offset = self.model.rope_temporal_offset
            self.model.rope_temporal_offset = rope_temporal_offset

        # Camera-bidirectional path: only the simplest forward is supported.
        # No KV cache, no teacher forcing, no classifier branch.
        assert kv_cache is None, "kv_cache is not supported with PRoPE bidir."
        assert clean_x is None, "clean_x (TF) is not supported with PRoPE bidir."
        assert not classify_mode, "classify_mode not supported with PRoPE bidir."

        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),  # [B, C, F, H, W]
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self._compute_seq_len(noisy_image_or_video),
            viewmats=viewmats,
            Ks=Ks,
        ).permute(0, 2, 1, 3, 4)

        if rope_offset_was_set:
            self.model.rope_temporal_offset = prev_rope_temporal_offset

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    @staticmethod
    def _compute_seq_len(noisy_image_or_video: torch.Tensor) -> int:
        # noisy_image_or_video: [B, F, C, H, W]; patch_size=(1,2,2)
        _, F, _, H, W = noisy_image_or_video.shape
        return F * (H // 2) * (W // 2)
