"""Torch-free admission for the maintained Wan 2.1 MultiTalk patch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT8_E4M3FNUZ,
    FLOAT8_E5M2,
    FLOAT8_E5M2FNUZ,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    DType,
)
from .weights import WeightSource


@dataclass(frozen=True, slots=True)
class Wan21MultiTalkConfig:
    """The exact native InfiniteTalk/MultiTalk patch architecture."""

    audio_window: int = 5
    latter_audio_window: int = 8
    audio_encoder_blocks: int = 12
    audio_input_width: int = 768
    audio_hidden_width: int = 512
    audio_context_width: int = 768
    audio_context_tokens: int = 32
    patch_width: int = 5120
    attention_heads: int = 40
    layers: int = 40
    speaker_class_range: int = 24
    speaker_class_interval: int = 4

    def __post_init__(self) -> None:
        expected = (5, 8, 12, 768, 512, 768, 32, 5120, 40, 40, 24, 4)
        actual = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if actual != expected:
            raise ValueError("unsupported Wan 2.1 MultiTalk configuration")


WAN21_MULTITALK = Wan21MultiTalkConfig()


def wan21_multitalk_layout(
    config: Wan21MultiTalkConfig = WAN21_MULTITALK,
) -> Mapping[str, tuple[int, ...]]:
    """Return the immutable complete 330-entry checkpoint layout."""
    if config is not WAN21_MULTITALK:
        raise ValueError("config must be the exact maintained Wan 2.1 MultiTalk profile")
    hidden = config.audio_hidden_width
    context = config.audio_context_width
    patch = config.patch_width
    layout: dict[str, tuple[int, ...]] = {
        "audio_proj.proj1.weight": (
            hidden,
            config.audio_window * config.audio_encoder_blocks * config.audio_input_width,
        ),
        "audio_proj.proj1.bias": (hidden,),
        "audio_proj.proj1_vf.weight": (
            hidden,
            config.latter_audio_window * config.audio_encoder_blocks * config.audio_input_width,
        ),
        "audio_proj.proj1_vf.bias": (hidden,),
        "audio_proj.proj2.weight": (hidden, hidden),
        "audio_proj.proj2.bias": (hidden,),
        "audio_proj.proj3.weight": (config.audio_context_tokens * context, hidden),
        "audio_proj.proj3.bias": (config.audio_context_tokens * context,),
        "audio_proj.norm.weight": (context,),
        "audio_proj.norm.bias": (context,),
    }
    block = {
        "audio_cross_attn.q_linear.weight": (patch, patch),
        "audio_cross_attn.q_linear.bias": (patch,),
        "audio_cross_attn.kv_linear.weight": (2 * patch, context),
        "audio_cross_attn.kv_linear.bias": (2 * patch,),
        "audio_cross_attn.proj.weight": (patch, patch),
        "audio_cross_attn.proj.bias": (patch,),
        "norm_x.weight": (patch,),
        "norm_x.bias": (patch,),
    }
    for index in range(config.layers):
        layout.update({f"blocks.{index}.{key}": shape for key, shape in block.items()})
    return MappingProxyType(layout)


def wan21_multitalk_model_layout(
    config: Wan21MultiTalkConfig = WAN21_MULTITALK,
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete native state layout after normalization."""
    return MappingProxyType(dict(wan21_multitalk_layout(config)))


_LOADABLE_FLOAT_DTYPES = frozenset(
    (
        FLOAT64,
        FLOAT32,
        FLOAT16,
        BFLOAT16,
        FLOAT8_E4M3,
        FLOAT8_E4M3FNUZ,
        FLOAT8_E5M2,
        FLOAT8_E5M2FNUZ,
    )
)


def _is_loadable_float(dtype: DType) -> bool:
    return dtype.kind == "float" and dtype in _LOADABLE_FLOAT_DTYPES


def detect_wan21_multitalk(source: WeightSource) -> Wan21MultiTalkConfig | None:
    """Fail closed unless every source key and shape is loadable floating state."""
    layout = wan21_multitalk_layout()
    if frozenset(source.keys()) != frozenset(layout):
        return None
    for key, shape in layout.items():
        geometry = source.entry(key).geometry
        if geometry.shape != shape or not _is_loadable_float(geometry.dtype):
            return None
    return WAN21_MULTITALK


def require_wan21_multitalk_layout(
    source: WeightSource,
) -> tuple[Wan21MultiTalkConfig, Mapping[str, tuple[int, ...]]]:
    """Return exact MultiTalk detection and layout or raise at planning."""
    config = detect_wan21_multitalk(source)
    if config is None:
        raise ValueError("source is not an exact maintained Wan 2.1 MultiTalk checkpoint")
    return config, wan21_multitalk_layout(config)


__all__ = [
    "WAN21_MULTITALK",
    "Wan21MultiTalkConfig",
    "detect_wan21_multitalk",
    "require_wan21_multitalk_layout",
    "wan21_multitalk_layout",
    "wan21_multitalk_model_layout",
]
