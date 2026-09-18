"""Focused Wan 2.2 core diffusion model tests."""

from __future__ import annotations

import torch
from dinkster_inference import WAN22_I2V_14B, WAN22_TI2V_5B, Wan21Config, wan21_layout
from dinkster_inference_torch.wan21_model import Wan21Model

SMALL_TI2V = Wan21Config(
    model_type="ti2v",
    in_channels=48,
    hidden_size=24,
    ffn_hidden_size=32,
    num_heads=3,
    num_layers=1,
    text_dim=12,
    time_freq_dim=8,
    out_channels=48,
)


def _small_model() -> Wan21Model:
    model = Wan21Model(SMALL_TI2V)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17 - 8) * 0.001)
    return model


def test_official_state_layout_matches_header_contract() -> None:
    with torch.device("meta"):
        model = Wan21Model(WAN22_TI2V_5B)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == wan21_layout(WAN22_TI2V_5B)


def test_official_14b_i2v_uses_concat_channels_without_vision_modules() -> None:
    with torch.device("meta"):
        model = Wan21Model(WAN22_I2V_14B)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == wan21_layout(WAN22_I2V_14B)
    assert actual["patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert model.img_emb is None
    assert not any("k_img" in key or "v_img" in key for key in actual)


def test_ti2v_forward_uses_48_channel_native_latents() -> None:
    model = _small_model()
    x = torch.linspace(-0.2, 0.2, 1 * 48 * 2 * 3 * 4).reshape(1, 48, 2, 3, 4)
    context = torch.linspace(-0.1, 0.1, 1 * 3 * 12).reshape(1, 3, 12)
    output = model(x, torch.tensor([500.0]), context)
    assert output.shape == x.shape
    assert output.dtype is torch.float32
    assert torch.isfinite(output).all()


def test_per_frame_timesteps_match_scalar_when_rows_are_equal() -> None:
    model = _small_model()
    x = torch.linspace(-0.2, 0.2, 1 * 48 * 2 * 3 * 4).reshape(1, 48, 2, 3, 4)
    context = torch.linspace(-0.1, 0.1, 1 * 3 * 12).reshape(1, 3, 12)
    scalar = model(x, torch.tensor([500.0]), context)
    per_frame = model(x, torch.tensor([[500.0, 500.0]]), context)
    assert torch.equal(per_frame, scalar)


def test_per_frame_timesteps_drive_temporal_masking_path() -> None:
    model = _small_model()
    x = torch.linspace(-0.2, 0.2, 1 * 48 * 2 * 3 * 4).reshape(1, 48, 2, 3, 4)
    context = torch.linspace(-0.1, 0.1, 1 * 3 * 12).reshape(1, 3, 12)
    uniform = model(x, torch.tensor([[500.0, 500.0]]), context)
    masked = model(x, torch.tensor([[0.0, 500.0]]), context)
    assert not torch.equal(masked[:, :, 0], uniform[:, :, 0])


def test_per_frame_timesteps_require_one_row_per_latent_frame() -> None:
    model = _small_model()
    x = torch.empty(1, 48, 2, 3, 4)
    context = torch.empty(1, 3, 12)
    try:
        model(x, torch.empty(1, 3), context)
    except ValueError as error:
        assert "timesteps must have shape (1,) or (1, 2)" in str(error)
    else:
        raise AssertionError("mismatched per-frame timesteps were accepted")
