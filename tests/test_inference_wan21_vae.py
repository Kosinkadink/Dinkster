from __future__ import annotations

from typing import Any

import pytest
from dinkster_inference.wan21_vae import (
    WAN21_FLOW_RVS_VAE_CONFIG,
    WAN21_VAE_CONFIG,
    Wan21VAEConfig,
    wan21_vae_layout,
)


def test_default_config_is_exact_wan21_geometry() -> None:
    config = WAN21_VAE_CONFIG
    assert config.dim == 96
    assert config.z_dim == 16
    assert config.dim_mult == (1, 2, 4, 4)
    assert config.num_res_blocks == 2
    assert config.attn_scales == ()
    assert config.temporal_downsample == (False, True, True)
    assert config.image_channels == config.conv_out_channels == 3
    assert config.dropout == 0.0
    assert config.spatial_ratio == 8


def test_flow_rvs_config_is_the_exact_rgb_to_mask_variant() -> None:
    config = WAN21_FLOW_RVS_VAE_CONFIG
    assert config.image_channels == 3
    assert config.conv_out_channels == 1
    assert config == Wan21VAEConfig(conv_out_channels=1)
    assert config != WAN21_VAE_CONFIG


@pytest.mark.parametrize(
    "changes",
    (
        {"z_dim": 0},
        {"dim_mult": (1,)},
        {"temporal_downsample": (True,)},
        {"temporal_downsample": (False, False, False)},
        {"image_channels": 4},
        {"conv_out_channels": 4},
        {"dropout": float("nan")},
        {"attn_scales": (0.0,)},
    ),
)
def test_config_refuses_non_wan21_geometry(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Wan21VAEConfig(**changes)


def test_layout_pins_complete_official_state_geometry() -> None:
    layout = wan21_vae_layout()
    assert len(layout) == 194
    assert layout["encoder.conv1.weight"] == (96, 3, 3, 3, 3)
    assert layout["encoder.downsamples.2.resample.1.weight"] == (96, 96, 3, 3)
    assert layout["encoder.downsamples.5.time_conv.weight"] == (192, 192, 3, 1, 1)
    assert layout["encoder.head.2.weight"] == (32, 384, 3, 3, 3)
    assert layout["decoder.conv1.weight"] == (384, 16, 3, 3, 3)
    assert layout["decoder.upsamples.3.time_conv.weight"] == (768, 384, 3, 1, 1)
    assert layout["decoder.upsamples.12.residual.6.weight"] == (96, 96, 3, 3, 3)
    assert layout["decoder.head.2.weight"] == (3, 96, 3, 3, 3)


def test_flow_rvs_layout_changes_only_the_decoder_output_channels() -> None:
    base = wan21_vae_layout(WAN21_VAE_CONFIG)
    flow_rvs = wan21_vae_layout(WAN21_FLOW_RVS_VAE_CONFIG)
    changed = {key for key in base if base[key] != flow_rvs[key]}

    assert changed == {"decoder.head.2.bias", "decoder.head.2.weight"}
    assert flow_rvs["encoder.conv1.weight"] == (96, 3, 3, 3, 3)
    assert flow_rvs["decoder.head.2.bias"] == (1,)
    assert flow_rvs["decoder.head.2.weight"] == (1, 96, 3, 3, 3)


def test_layout_tracks_reduced_geometry_and_attention_indices() -> None:
    config = Wan21VAEConfig(
        dim=2,
        z_dim=2,
        dim_mult=(1, 1, 1),
        num_res_blocks=1,
        attn_scales=(1.0,),
        temporal_downsample=(True, True),
    )
    layout = wan21_vae_layout(config)
    assert layout["encoder.downsamples.1.norm.gamma"] == (2, 1, 1)
    assert layout["encoder.downsamples.2.resample.1.weight"] == (2, 2, 3, 3)
    assert layout["decoder.upsamples.4.norm.gamma"] == (2, 1, 1)
    assert layout["decoder.upsamples.5.residual.0.gamma"] == (2, 1, 1, 1)


def test_layout_rejects_foreign_config() -> None:
    with pytest.raises(TypeError, match="Wan21VAEConfig"):
        wan21_vae_layout(object())  # type: ignore[arg-type]
