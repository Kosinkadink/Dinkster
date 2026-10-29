from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as torch_functional
from dinkster_inference import LTXLatentUpsamplerConfig
from dinkster_inference_torch import LTXLatentUpsampler, LTXLatentUpsamplerResBlock, PixelShuffle2D


def _tiny_upscaler() -> LTXLatentUpsampler:
    model = LTXLatentUpsampler(
        LTXLatentUpsamplerConfig(
            in_channels=2,
            mid_channels=32,
            num_blocks_per_stage=1,
        )
    )
    generator = torch.Generator().manual_seed(19)
    for parameter in model.parameters():
        parameter.data.normal_(generator=generator)
    return model


def _reference_block(block: LTXLatentUpsamplerResBlock, value: torch.Tensor) -> torch.Tensor:
    residual = value
    value = torch_functional.silu(block.norm1(block.conv1(value)))
    value = block.norm2(block.conv2(value))
    return torch_functional.silu(value + residual)


def _source_derived_forward(model: LTXLatentUpsampler, latent: torch.Tensor) -> torch.Tensor:
    batch, _channels, frames, _height, _width = latent.shape
    value = torch_functional.silu(model.initial_norm(model.initial_conv(latent)))
    for block in model.res_blocks:
        value = _reference_block(cast("LTXLatentUpsamplerResBlock", block), value)
    channels, height, width = value.shape[1], value.shape[3], value.shape[4]
    value = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    convolution = cast("torch.nn.Conv2d", model.upsampler[0])
    value = torch_functional.conv2d(
        value,
        convolution.weight,
        convolution.bias,
        padding=convolution.padding,
    )
    value = torch_functional.pixel_shuffle(value, 2)
    height, width = value.shape[2], value.shape[3]
    value = value.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    for block in model.post_upsample_res_blocks:
        value = _reference_block(cast("LTXLatentUpsamplerResBlock", block), value)
    return model.final_conv(value)


def test_pixel_shuffle_matches_torch_channel_order() -> None:
    value = torch.arange(2 * 12 * 2 * 3, dtype=torch.float32).reshape(2, 12, 2, 3)

    actual = PixelShuffle2D(2)(value)

    assert torch.equal(actual, torch_functional.pixel_shuffle(value, 2))


def test_ltx_latent_upscaler_matches_source_derived_forward_and_autograd() -> None:
    model = _tiny_upscaler()
    latent = torch.randn(1, 2, 2, 3, 4, requires_grad=True)

    actual = model(latent)
    expected = _source_derived_forward(model, latent)

    assert actual.shape == (1, 2, 2, 6, 8)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
