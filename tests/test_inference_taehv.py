from __future__ import annotations

import pytest
from dinkster_inference import (
    FLOAT32,
    TAEHVConfig,
    TAEHVDetectError,
    TensorGeometry,
    detect_taehv_decoder_config,
    taehv_decoder_layout,
)


def geometries(config: TAEHVConfig, *, prefix: str = "") -> dict[str, TensorGeometry]:
    return {
        prefix + key: TensorGeometry(shape, FLOAT32)
        for key, shape in taehv_decoder_layout(config).items()
    }


def test_detects_both_wan_variants_by_geometry() -> None:
    assert detect_taehv_decoder_config(geometries(TAEHVConfig(16, 1))) == TAEHVConfig(16, 1)
    assert detect_taehv_decoder_config(geometries(TAEHVConfig(48, 2))) == TAEHVConfig(48, 2)


def test_detection_strips_the_decoder_prefix_and_ignores_the_encoder_half() -> None:
    data = geometries(TAEHVConfig(16, 1), prefix="decoder.")
    data["encoder.1.weight"] = TensorGeometry((64, 12, 3, 3), FLOAT32)
    assert detect_taehv_decoder_config(data) == TAEHVConfig(16, 1)


def test_detection_is_strict() -> None:
    truncated = geometries(TAEHVConfig(16, 1))
    truncated.pop("22.bias")
    wrong_width = geometries(TAEHVConfig(16, 1))
    wrong_width["1.weight"] = TensorGeometry((256, 12, 3, 3), FLOAT32)
    extra = geometries(TAEHVConfig(16, 1))
    extra["23.weight"] = TensorGeometry((3, 64, 3, 3), FLOAT32)
    for data in (truncated, wrong_width, extra):
        with pytest.raises(TAEHVDetectError, match="wrong tensor geometry"):
            detect_taehv_decoder_config(data)


def test_unsupported_geometry_refuses() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        TAEHVConfig(32, 2)


def test_config_derives_upscales_and_trim() -> None:
    config = TAEHVConfig(48, 2)
    assert config.spatial_upscale == 16
    assert config.temporal_upscale == 4
    assert config.frames_to_trim == 3
    assert TAEHVConfig(16, 1).spatial_upscale == 8


def test_layout_pins_architecture_listing() -> None:
    layout = taehv_decoder_layout(TAEHVConfig(16, 1))
    assert len(layout) == 64
    assert layout["1.weight"] == (256, 16, 3, 3)
    assert layout["7.conv.weight"] == (256, 256, 1, 1)
    assert "7.conv.bias" not in layout
    assert taehv_decoder_layout(TAEHVConfig(48, 2))["22.weight"] == (12, 64, 3, 3)
