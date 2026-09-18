"""TencentARC SD1.5 full T2I Adapter execution provider."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import SD15T2IAdapterConfig

from .controlnet import (
    ControlResourceBindingError,
    SDControlResiduals,
    _matches_sealed_resource_tensor,  # pyright: ignore[reportPrivateUsage]
    _resource_tensor_seal,  # pyright: ignore[reportPrivateUsage]
    _resource_tensors,  # pyright: ignore[reportPrivateUsage]
    _ResourceTensorSeal,  # pyright: ignore[reportPrivateUsage]
)
from .operations import INITLESS, Operations

_CONFIG = SD15T2IAdapterConfig()


class _Block(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, *, down: bool, operations: Operations
    ) -> None:
        super().__init__()
        self.down = down
        self.in_conv = (
            operations.conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.block1 = operations.conv2d(out_channels, out_channels, 3, padding=1)
        self.block2 = operations.conv2d(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = F.avg_pool2d(x, 2, 2, padding=(x.shape[-2] % 2, x.shape[-1] % 2))
        if self.in_conv is not None:
            x = self.in_conv(x)
        return self.block2(F.relu(self.block1(x))) + x


class SD15T2IAdapter(torch.nn.Module):
    """Full non-XL Adapter from ComfyUI adapter.py at de6b062f."""

    def __init__(
        self,
        config: SD15T2IAdapterConfig = _CONFIG,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        self.unshuffle = torch.nn.PixelUnshuffle(8)
        self.conv_in = operations.conv2d(64, 320, 3, padding=1)
        blocks: list[torch.nn.Module] = []
        for index in range(8):
            level = index // 2
            channels = config.channels[level]
            previous = config.channels[level - 1] if level and index % 2 == 0 else channels
            blocks.append(
                _Block(previous, channels, down=index in (2, 4, 6), operations=operations)
            )
        self.body = torch.nn.ModuleList(blocks)

    def forward(self, hint: torch.Tensor) -> SDControlResiduals:
        if hint.ndim != 4 or hint.shape[1] != 1 or not hint.is_floating_point():
            raise ValueError("T2I Adapter hint must be floating [batch x 1 x H x W]")
        if hint.shape[-2] % 64 or hint.shape[-1] % 64:
            raise ValueError("T2I Adapter hint dimensions must be divisible by 64")
        x = self.conv_in(self.unshuffle(hint))
        features: list[torch.Tensor] = []
        for index, block in enumerate(self.body):
            x = block(x)
            if index % 2 == 1:
                features.append(x)
        batch, _, height, width = features[0].shape
        channels = (320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280)
        scales = (1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8)
        down = tuple(
            features[(2, 5, 8, 11).index(index)]
            if index in (2, 5, 8, 11)
            else features[0].new_zeros(batch, channel, height // scale, width // scale)
            for index, (channel, scale) in enumerate(zip(channels, scales, strict=True))
        )
        middle = features[0].new_zeros(batch, 1280, height // 8, width // 8)
        return SDControlResiduals(down, middle)

    @property
    def resource_digest(self) -> str | None:
        seal = _SEALS.get(self)
        return None if seal is None else seal.digest


@dataclass(frozen=True)
class _Seal:
    digest: str
    tensors: tuple[_ResourceTensorSeal, ...]


_SEALS: weakref.WeakKeyDictionary[SD15T2IAdapter, _Seal] = weakref.WeakKeyDictionary()


def sd15_t2i_adapter_resource_digest(asset_digest: str, dtype: torch.dtype) -> str:
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("T2I Adapter source digest must be a canonical blake3 asset digest")
    if not dtype.is_floating_point:
        raise TypeError("T2I Adapter compute dtype must be floating")
    return blake3(f"dinkster.sd15_t2i_adapter.v1\0{asset_digest}\0{dtype}".encode()).hexdigest()


def _bind_sd15_t2i_adapter_resource(  # pyright: ignore[reportUnusedFunction]
    model: SD15T2IAdapter, digest: str
) -> None:
    if type(model) is not SD15T2IAdapter:
        raise TypeError("T2I Adapter resource binding requires an exact SD15T2IAdapter")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("T2I Adapter resource digest must be a lowercase BLAKE3 digest")
    if model in _SEALS:
        raise ValueError("T2I Adapter resource is already bound")
    _SEALS[model] = _Seal(
        digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def validate_sd15_t2i_adapter_resource(model: SD15T2IAdapter, digest: str) -> None:
    seal = _SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            not _matches_sealed_resource_tensor(model, name, tensor, expected)
            for (name, tensor), expected in zip(current, seal.tensors if seal else (), strict=False)
        )
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: T2I Adapter provenance is not sealed"
        )


__all__ = [
    "SD15T2IAdapter",
    "sd15_t2i_adapter_resource_digest",
    "validate_sd15_t2i_adapter_resource",
]
