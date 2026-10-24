"""CPU contracts for native LTXV image preprocessing."""

from __future__ import annotations

import torch

from dinkster_nodes_comfy.ltx_video_preprocess import LTXVPreprocess


def _frames(dtype: torch.dtype) -> torch.Tensor:
    frame0 = torch.arange(5 * 7 * 3, dtype=dtype).reshape(5, 7, 3) / 104
    frame1 = torch.flip(frame0, (0, 1))
    return torch.stack((frame0, frame1))


def test_compression_zero_preserves_all_frames_identity_and_values() -> None:
    image = _frames(torch.float64)
    output = LTXVPreprocess.execute(image=image, img_compression=0)["IMAGE"]
    assert output.dtype is torch.float64
    assert output.device == image.device
    assert output.shape == image.shape
    assert torch.equal(output, image)


def test_h264_preprocess_trims_odd_dimensions_preserves_dtype_and_processes_each_frame() -> None:
    image = _frames(torch.float32)
    output = LTXVPreprocess.execute(image=image, img_compression=35)["IMAGE"]
    assert output.shape == (2, 4, 6, 3)
    assert output.dtype is torch.float32
    assert output.device == image.device
    assert torch.all((output >= 0) & (output <= 1))
    assert not torch.equal(output[0], output[1])
    assert torch.equal(image, _frames(torch.float32))


def test_h264_preprocess_preserves_float64_dtype() -> None:
    output = LTXVPreprocess.execute(image=_frames(torch.float64), img_compression=23)["IMAGE"]
    assert output.dtype is torch.float64
    assert output.shape == (2, 4, 6, 3)
