"""Patch torchvision.io.write_video/read_video if missing (torchvision >= 0.27).

Importing this module is a no-op when the functions already exist.
"""
import torchvision.io as _tv_io

if not hasattr(_tv_io, "write_video"):
    import imageio.v2 as _imageio_v2
    import numpy as _np

    def _shim_write_video(filename, video_array, fps, **_unused):
        if hasattr(video_array, "detach"):
            video_array = video_array.detach().cpu().numpy()
        _imageio_v2.mimwrite(filename, video_array, fps=fps, codec="libx264", quality=8)

    _tv_io.write_video = _shim_write_video

if not hasattr(_tv_io, "read_video"):
    import imageio.v3 as _imageio_v3
    import torch as _torch

    def _shim_read_video(filename, pts_unit="sec", output_format="THWC", **_unused):
        frames = _imageio_v3.imread(filename, plugin="pyav")
        tensor = _torch.from_numpy(frames)
        if output_format == "TCHW":
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor, None, {}

    _tv_io.read_video = _shim_read_video
