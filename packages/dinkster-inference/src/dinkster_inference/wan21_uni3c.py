"""Torch-free admission for the maintained Wan 2.1 Uni3C patch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import FLOAT16, FLOAT32, DType
from .weights import WeightSource


@dataclass(frozen=True, slots=True)
class Wan21Uni3CConfig:
    """The exact published Uni3C architecture maintained by Dinkster."""

    input_channels: int = 36
    patch_width: int = 5120
    hidden_width: int = 1024
    ffn_width: int = 4096
    attention_heads: int = 16
    layers: int = 20
    timestep_width: int = 5120
    output_width: int = 5120
    mask_channels: int = 7
    mask_hidden_width: int = 256

    def __post_init__(self) -> None:
        expected = (36, 5120, 1024, 4096, 16, 20, 5120, 5120, 7, 256)
        actual = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if actual != expected:
            raise ValueError("unsupported Wan 2.1 Uni3C configuration")


WAN21_UNI3C = Wan21Uni3CConfig()


def wan21_uni3c_layout(
    config: Wan21Uni3CConfig = WAN21_UNI3C,
) -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 490-entry published checkpoint layout."""
    if config is not WAN21_UNI3C:
        raise ValueError("config must be the exact maintained Wan 2.1 Uni3C profile")
    hidden = config.hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "controlnet_patch_embedding.weight": (
            config.patch_width,
            config.input_channels,
            1,
            2,
            2,
        ),
        "controlnet_patch_embedding.bias": (config.patch_width,),
        "controlnet_mask_embedding.mask_proj.0.weight": (
            config.mask_hidden_width,
            config.mask_channels,
            4,
            8,
            8,
        ),
        "controlnet_mask_embedding.mask_proj.0.bias": (config.mask_hidden_width,),
        "controlnet_mask_embedding.mask_proj.1.weight": (config.mask_hidden_width,),
        "controlnet_mask_embedding.mask_proj.1.bias": (config.mask_hidden_width,),
        "controlnet_mask_embedding.mask_zero_proj.weight": (
            config.patch_width,
            config.mask_hidden_width,
            1,
            2,
            2,
        ),
        "controlnet_mask_embedding.mask_zero_proj.bias": (config.patch_width,),
        "proj_in.weight": (hidden, config.patch_width),
        "proj_in.bias": (hidden,),
    }
    block = {
        "norm1.linear.weight": (3 * hidden, config.timestep_width),
        "norm1.linear.bias": (3 * hidden,),
        "norm1.norm.weight": (hidden,),
        "norm1.norm.bias": (hidden,),
        "self_attn.to_q.weight": (hidden, hidden),
        "self_attn.to_q.bias": (hidden,),
        "self_attn.to_k.weight": (hidden, hidden),
        "self_attn.to_k.bias": (hidden,),
        "self_attn.to_v.weight": (hidden, hidden),
        "self_attn.to_v.bias": (hidden,),
        "self_attn.to_out.0.weight": (hidden, hidden),
        "self_attn.to_out.0.bias": (hidden,),
        "self_attn.norm_q.weight": (hidden,),
        "self_attn.norm_k.weight": (hidden,),
        "norm2.linear.weight": (3 * hidden, config.timestep_width),
        "norm2.linear.bias": (3 * hidden,),
        "norm2.norm.weight": (hidden,),
        "norm2.norm.bias": (hidden,),
        "ffn.0.weight": (config.ffn_width, hidden),
        "ffn.0.bias": (config.ffn_width,),
        "ffn.2.weight": (hidden, config.ffn_width),
        "ffn.2.bias": (hidden,),
    }
    for index in range(config.layers):
        layout.update({f"controlnet_blocks.{index}.{key}": shape for key, shape in block.items()})
        layout[f"proj_out.{index}.weight"] = (config.output_width, hidden)
        layout[f"proj_out.{index}.bias"] = (config.output_width,)
    return MappingProxyType(layout)


_ATTENTION_RENAMES = (
    (".self_attn.to_q.", ".self_attn.q."),
    (".self_attn.to_k.", ".self_attn.k."),
    (".self_attn.to_v.", ".self_attn.v."),
    (".self_attn.to_out.0.", ".self_attn.o."),
)


def normalize_wan21_uni3c_key(key: str) -> str:
    """Convert the published diffusers attention spelling to the native model spelling."""
    normalized = key
    for source, target in _ATTENTION_RENAMES:
        normalized = normalized.replace(source, target)
    return normalized


def wan21_uni3c_model_layout(
    config: Wan21Uni3CConfig = WAN21_UNI3C,
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete native state layout after key normalization."""
    return MappingProxyType(
        {normalize_wan21_uni3c_key(key): shape for key, shape in wan21_uni3c_layout(config).items()}
    )


def wan21_uni3c_dtype(key: str) -> DType:
    """Return the storage dtype pinned by the maintained artifact."""
    if key not in wan21_uni3c_layout():
        raise KeyError(key)
    return FLOAT32 if key.startswith("controlnet_patch_embedding.") else FLOAT16


def detect_wan21_uni3c(source: WeightSource) -> Wan21Uni3CConfig | None:
    """Fail closed unless every source key, shape, and dtype matches."""
    layout = wan21_uni3c_layout()
    if frozenset(source.keys()) != frozenset(layout):
        return None
    for key, shape in layout.items():
        geometry = source.entry(key).geometry
        if geometry.shape != shape or geometry.dtype != wan21_uni3c_dtype(key):
            return None
    return WAN21_UNI3C


def require_wan21_uni3c_layout(
    source: WeightSource,
) -> tuple[Wan21Uni3CConfig, Mapping[str, tuple[int, ...]]]:
    """Return exact Uni3C detection and layout or raise at planning."""
    config = detect_wan21_uni3c(source)
    if config is None:
        raise ValueError("source is not the exact maintained Wan 2.1 Uni3C checkpoint")
    return config, wan21_uni3c_layout(config)


__all__ = [
    "WAN21_UNI3C",
    "Wan21Uni3CConfig",
    "detect_wan21_uni3c",
    "normalize_wan21_uni3c_key",
    "require_wan21_uni3c_layout",
    "wan21_uni3c_dtype",
    "wan21_uni3c_layout",
    "wan21_uni3c_model_layout",
]
