"""Torch-free detection for the TencentARC SD1.5 full T2I Adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .weights import TensorGeometry


class T2IAdapterDetectError(ValueError):
    """A header is not the supported full SD1.5 adapter layout."""


@dataclass(frozen=True)
class SD15T2IAdapterConfig:
    channels: tuple[int, ...] = (320, 640, 1280, 1280)
    num_res_blocks: int = 2
    input_channels: int = 64
    kernel_size: int = 1
    skip: bool = True
    use_conv_downsample: bool = False
    pixel_unshuffle: int = 8

    def __post_init__(self) -> None:
        if (
            self.channels,
            self.num_res_blocks,
            self.input_channels,
            self.kernel_size,
            self.skip,
            self.use_conv_downsample,
            self.pixel_unshuffle,
        ) != ((320, 640, 1280, 1280), 2, 64, 1, True, False, 8):
            raise ValueError("only the TencentARC SD1.5 full adapter v2 geometry is supported")


SD15_T2I_ADAPTER_CONFIG = SD15T2IAdapterConfig()


def sd15_t2i_adapter_layout(
    config: SD15T2IAdapterConfig = SD15_T2I_ADAPTER_CONFIG,
) -> Mapping[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {
        "conv_in.weight": (320, 64, 3, 3),
        "conv_in.bias": (320,),
    }
    for index in range(8):
        level = index // 2
        channels = config.channels[level]
        previous = config.channels[level - 1] if level else channels
        if index in (2, 4):
            keys[f"body.{index}.in_conv.weight"] = (channels, previous, 1, 1)
            keys[f"body.{index}.in_conv.bias"] = (channels,)
        keys[f"body.{index}.block1.weight"] = (channels, channels, 3, 3)
        keys[f"body.{index}.block1.bias"] = (channels,)
        keys[f"body.{index}.block2.weight"] = (channels, channels, 1, 1)
        keys[f"body.{index}.block2.bias"] = (channels,)
    return MappingProxyType(keys)


def normalize_sd15_t2i_adapter(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[SD15T2IAdapterConfig, dict[str, str]]:
    if not geometries:
        raise T2IAdapterDetectError("empty state dict header")
    config = SD15T2IAdapterConfig()
    layout = sd15_t2i_adapter_layout(config)
    keys = set(geometries)
    expected = set(layout)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        detail = f"missing {missing[:3]}" if missing else f"leftover {extra[:3]}"
        raise T2IAdapterDetectError(f"incomplete or foreign T2I Adapter keys: {detail}")
    for key, expected_shape in layout.items():
        geometry = geometries[key]
        if geometry.dtype.kind != "float":
            raise T2IAdapterDetectError(f"T2I Adapter storage must be floating for {key}")
        if geometry.shape != expected_shape:
            raise T2IAdapterDetectError(
                f"geometry mismatch for {key}: got {geometry.shape}, expected {expected_shape}"
            )
    return config, {key: key for key in layout}


def detect_sd15_t2i_adapter(
    geometries: Mapping[str, TensorGeometry],
) -> SD15T2IAdapterConfig:
    return normalize_sd15_t2i_adapter(geometries)[0]


__all__ = [
    "SD15T2IAdapterConfig",
    "T2IAdapterDetectError",
    "detect_sd15_t2i_adapter",
    "normalize_sd15_t2i_adapter",
    "sd15_t2i_adapter_layout",
]
