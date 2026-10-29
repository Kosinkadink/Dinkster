from __future__ import annotations

import math
from typing import Any

import pytest
from dinkster_inference.devices import FLOAT16, FLOAT32, INT64
from dinkster_inference.wan22_vae import (
    WAN22_VAE_CONFIG,
    Wan22VAEConfig,
    Wan22VAEHeaderError,
    validate_wan22_vae_header,
    wan22_vae_layout,
)
from dinkster_inference.weights import TensorGeometry


def _official_header() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT16) for key, shape in wan22_vae_layout().items()}


def test_default_config_is_exact_wan22_geometry() -> None:
    config = WAN22_VAE_CONFIG
    assert config.dim == 160
    assert config.decoder_dim == 256
    assert config.z_dim == 48
    assert config.dim_mult == (1, 2, 4, 4)
    assert config.num_res_blocks == 2
    assert config.attn_scales == ()
    assert config.temporal_downsample == (False, True, True)
    assert config.image_channels == config.conv_out_channels == 3
    assert config.patch_size == 2
    assert config.dropout == 0.0
    assert config.spatial_ratio == 16
    assert config.temporal_ratio == 4


@pytest.mark.parametrize(
    "changes",
    (
        {"dim": True},
        {"decoder_dim": 0},
        {"z_dim": 0},
        {"num_res_blocks": 0},
        {"dim_mult": (1, 2, 4)},
        {"temporal_downsample": (True, True, False)},
        {"attn_scales": (1.0,)},
        {"image_channels": 4},
        {"conv_out_channels": 4},
        {"patch_size": 1},
        {"dropout": float("nan")},
    ),
)
def test_config_refuses_non_wan22_topology(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Wan22VAEConfig(**changes)


def test_layout_pins_complete_official_state_geometry() -> None:
    layout = wan22_vae_layout()
    assert len(layout) == 196
    assert sum(math.prod(shape) * 2 for shape in layout.values()) == 1_409_377_336
    assert layout["encoder.conv1.weight"] == (160, 12, 3, 3, 3)
    assert layout["encoder.downsamples.2.downsamples.2.time_conv.weight"] == (
        640,
        640,
        3,
        1,
        1,
    )
    assert layout["encoder.head.2.weight"] == (96, 640, 3, 3, 3)
    assert layout["conv1.weight"] == (96, 96, 1, 1, 1)
    assert layout["conv2.weight"] == (48, 48, 1, 1, 1)
    assert layout["decoder.conv1.weight"] == (1024, 48, 3, 3, 3)
    assert layout["decoder.upsamples.0.upsamples.3.time_conv.weight"] == (
        2048,
        1024,
        3,
        1,
        1,
    )
    assert "encoder.downsamples.0.downsamples.2.time_conv.weight" not in layout
    assert "decoder.upsamples.2.upsamples.3.time_conv.weight" not in layout
    assert layout["decoder.head.2.weight"] == (12, 256, 3, 3, 3)


def test_layout_tracks_reduced_widths_without_changing_topology() -> None:
    config = Wan22VAEConfig(dim=2, decoder_dim=3, z_dim=2, num_res_blocks=1)
    layout = wan22_vae_layout(config)
    assert layout["encoder.conv1.weight"] == (2, 12, 3, 3, 3)
    assert layout["encoder.downsamples.1.downsamples.0.shortcut.weight"] == (
        4,
        2,
        1,
        1,
        1,
    )
    assert layout["encoder.head.2.weight"] == (4, 8, 3, 3, 3)
    assert layout["decoder.conv1.weight"] == (12, 2, 3, 3, 3)
    assert layout["decoder.upsamples.2.upsamples.0.shortcut.weight"] == (
        6,
        12,
        1,
        1,
        1,
    )
    assert layout["decoder.head.2.weight"] == (12, 3, 3, 3, 3)


def test_official_header_contract_accepts_exact_geometry_with_floating_storage() -> None:
    header = _official_header()
    assert validate_wan22_vae_header(header) is WAN22_VAE_CONFIG

    float32 = {key: TensorGeometry(value.shape, FLOAT32) for key, value in header.items()}
    assert validate_wan22_vae_header(float32) is WAN22_VAE_CONFIG

    missing = dict(header)
    missing.pop("conv1.bias")
    with pytest.raises(Wan22VAEHeaderError, match="missing required"):
        validate_wan22_vae_header(missing)

    extra = dict(header)
    extra["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    with pytest.raises(Wan22VAEHeaderError, match="foreign"):
        validate_wan22_vae_header(extra)

    wrong_shape = dict(header)
    wrong_shape["conv2.weight"] = TensorGeometry((48, 47, 1, 1, 1), FLOAT16)
    with pytest.raises(Wan22VAEHeaderError, match="geometry mismatch"):
        validate_wan22_vae_header(wrong_shape)

    wrong_dtype = dict(header)
    wrong_dtype["conv2.weight"] = TensorGeometry((48, 48, 1, 1, 1), INT64)
    with pytest.raises(Wan22VAEHeaderError, match="storage dtype mismatch"):
        validate_wan22_vae_header(wrong_dtype)


def test_layout_rejects_foreign_config() -> None:
    with pytest.raises(TypeError, match="Wan22VAEConfig"):
        wan22_vae_layout(object())  # type: ignore[arg-type]
