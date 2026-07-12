"""DreamX Camera + Echo-Infinity Memory combined wrapper for Wan2.2-TI2V-5B.

This wrapper combines:
  1. DreamX-World style camera control (``cam_self_attn`` with E-PRoPE per block)
  2. Echo-Infinity streaming memory (QueryMemoryEncoder + Relative RoPE + KV sink)

Initialization order (see section III of the design doc):
  1. ``DreamXCameraWanDiffusionWrapper.__init__`` creates the causal Wan model
     and attaches ``cam_self_attn`` to every block.
  2. ``_attach_infmem_to_wrapper`` monkey-patches the block / self-attn /
     model._forward_inference with InfMem's memory-KV-aware variants.
  3. ``QueryMemoryEncoder`` is created and mounted via ``object.__setattr__``.

The patched ``_block_infmem_forward`` (in ``wan_5b/modules/infinity_memory.py``)
checks ``getattr(self, "cam_self_attn", None)`` and ``prope_meta`` at runtime,
so both the content self-attention (with memory KV + Relative RoPE) and the
camera self-attention (with E-PRoPE) run in parallel within each block.

Backward compatibility:
  * When ``memory_kwargs=None``, the wrapper degrades to DreamX Camera AR
    (InfMem patches are still applied but ``memory_kv=None`` → memory branch
    is a no-op).
  * When ``viewmats/Ks`` are not passed, the camera branch is skipped.
  * Old wrappers (``InfMemWanDiffusionWrapper``, ``DreamXCameraWanDiffusionWrapper``)
    are unaffected because they do not use this class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.dreamx_camera_wrapper import DreamXCameraWanDiffusionWrapper
from utils.infinity_memory_wrapper import _attach_infmem_to_wrapper


class DreamXInfMemWanDiffusionWrapper(DreamXCameraWanDiffusionWrapper):
    """Combined DreamX Camera I2V AR + Echo-Infinity Memory wrapper.

    Inherits from :class:`DreamXCameraWanDiffusionWrapper` (NOT from
    :class:`InfMemWanDiffusionWrapper`) to avoid double-initializing the Wan
    model via diamond inheritance. The InfMem monkey-patch + encoder creation
    is applied AFTER the DreamX camera submodules are in place.

    Extra kwargs (all optional, default-off):
        enable_relative_rope: bool, default False.
        relative_rope_pmax: int, default 24.
        memory_kwargs: dict, default None.
    """

    def __init__(
        self,
        *args: Any,
        enable_relative_rope: bool = False,
        relative_rope_pmax: int = 24,
        memory_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Step 1-4: DreamXCameraWanDiffusionWrapper.__init__ creates the causal
        # Wan model, adds cam_self_attn to every block, and (optionally) loads
        # the DreamX checkpoint.
        super().__init__(*args, **kwargs)

        # Step 5: Attach InfMem monkey-patches (block/self-attn/model.forward).
        # Step 6: Create and mount QueryMemoryEncoder.
        _attach_infmem_to_wrapper(
            self,
            enable_relative_rope=enable_relative_rope,
            relative_rope_pmax=relative_rope_pmax,
            memory_kwargs=memory_kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience accessors (mirror InfMemWanDiffusionWrapper)
    # ------------------------------------------------------------------
    @property
    def encoder(self):
        """Return the ``QueryMemoryEncoder`` or ``None`` if disabled."""
        return getattr(self.model, "query_memory_encoder", None)

    def reset_memory(
        self,
        batch_size: int,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        """Reset the memory encoder state at the start of a new video/window."""
        enc = self.encoder
        if enc is None:
            return
        enc.reset(batch_size=batch_size, device=device, dtype=dtype)
        object.__setattr__(self.model, "_ei_prev_window_start", None)
