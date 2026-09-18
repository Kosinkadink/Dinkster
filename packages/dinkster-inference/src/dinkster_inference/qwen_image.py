"""Torch-free Qwen Image profiles and fail-closed header detection.

The profiles follow ComfyUI 76135e557da1ec7dcb270160f01e597565e3e003.
They remain an inert source foundation until the native family route is registered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .weights import TensorGeometry, WeightSource

_PREFIXES = ("model.diffusion_model.", "")
_INDEX_TIMESTEP_ZERO = "__index_timestep_zero__"
_ADDITIONAL_T_COND = "time_text_embed.addition_t_embedding.weight"
_LINEAR_OUTPUTS: Mapping[str, int] = MappingProxyType(
    {
        "img_in.weight": 3072,
        "txt_in.weight": 3072,
        "proj_out.weight": 64,
        "time_text_embed.timestep_embedder.linear_2.weight": 3072,
    }
)
_BLOCK_INDICES = frozenset(range(60))
_BLOCK_INDEX_TEXT = frozenset(str(index) for index in _BLOCK_INDICES)


@dataclass(frozen=True)
class QwenImageConfig:
    """One exact, non-runnable Qwen Image architecture profile."""

    family_id: str = "dinkster.qwen_image"
    transformer_blocks: int = 60
    hidden_width: int = 3072
    attention_heads: int = 24
    attention_head_dim: int = 128
    text_width: int = 3584
    pooled_width: int = 768
    patchified_input_channels: int = 64
    output_latent_channels: int = 16
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (16, 56, 56)
    latent_id: str = "Wan21"
    latent_channels: int = 16
    latent_dimensions: int = 3
    temporal_downscale: int = 4
    sampling_multiplier: float = 1.0
    sampling_shift: float = 1.15
    inference_dtypes: tuple[DType, DType] = (BFLOAT16, FLOAT32)
    memory_factor: float = 1.8
    text_encoder_id: str = "Qwen2.5-VL-7B"
    default_ref_method: str = "index"
    use_additional_t_cond: bool = False

    def __post_init__(self) -> None:
        actual = (
            self.family_id,
            self.transformer_blocks,
            self.hidden_width,
            self.attention_heads,
            self.attention_head_dim,
            self.text_width,
            self.pooled_width,
            self.patchified_input_channels,
            self.output_latent_channels,
            self.patch,
            self.rope_axes,
            self.latent_id,
            self.latent_channels,
            self.latent_dimensions,
            self.temporal_downscale,
            self.sampling_multiplier,
            self.sampling_shift,
            self.inference_dtypes,
            self.memory_factor,
            self.text_encoder_id,
        )
        expected = (
            "dinkster.qwen_image",
            60,
            3072,
            24,
            128,
            3584,
            768,
            64,
            16,
            (2, 2),
            (16, 56, 56),
            "Wan21",
            16,
            3,
            4,
            1.0,
            1.15,
            (BFLOAT16, FLOAT32),
            1.8,
            "Qwen2.5-VL-7B",
        )
        variant = (self.default_ref_method, self.use_additional_t_cond)
        if (
            any(
                type(value) is not type(required) or value != required
                for value, required in zip(actual, expected, strict=True)
            )
            or any(type(value) is not int for value in self.patch)
            or any(type(value) is not int for value in self.rope_axes)
            or variant
            not in (("index", False), ("index_timestep_zero", False), ("negative_index", True))
            or type(self.default_ref_method) is not str
            or type(self.use_additional_t_cond) is not bool
        ):
            raise ValueError("QwenImageConfig only represents an exact supported profile")


QWEN_IMAGE_CONFIG = QwenImageConfig()
QWEN_IMAGE_EDIT_2511_CONFIG = QwenImageConfig(default_ref_method="index_timestep_zero")
QWEN_IMAGE_LAYERED_CONFIG = QwenImageConfig(
    default_ref_method="negative_index", use_additional_t_cond=True
)


@dataclass(frozen=True)
class QwenImageEvidence:
    """Immutable deterministic evidence for one exact Qwen Image profile."""

    config: QwenImageConfig
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def _entry_geometry(source: WeightSource, keys: frozenset[str], key: str) -> TensorGeometry | None:
    if key not in keys:
        return None
    try:
        return source.entry(key).geometry
    except KeyError:
        return None


def _block_indices(keys: frozenset[str], prefix: str) -> frozenset[int] | None:
    root = prefix + "transformer_blocks."
    indices: set[int] = set()
    for key in keys:
        if not key.startswith(root):
            continue
        text, separator, remainder = key[len(root) :].partition(".")
        if (
            not separator
            or not remainder
            or not text.isascii()
            or not text.isdecimal()
            or len(text) > 1
            and text.startswith("0")
            or text not in _BLOCK_INDEX_TEXT
        ):
            return None
        indices.add(int(text))
    return frozenset(indices)


def _matches_prefix(
    source: WeightSource, keys: frozenset[str], prefix: str
) -> tuple[str, ...] | None:
    if _block_indices(keys, prefix) != _BLOCK_INDICES:
        return None

    matched: list[str] = []
    txt_norm_key = prefix + "txt_norm.weight"
    txt_norm = _entry_geometry(source, keys, txt_norm_key)
    if txt_norm is None or txt_norm.shape != (3584,):
        return None
    matched.append(txt_norm_key)

    for suffix, output_dimension in _LINEAR_OUTPUTS.items():
        key = prefix + suffix
        geometry = _entry_geometry(source, keys, key)
        if geometry is None or len(geometry.shape) != 2 or geometry.shape[0] != output_dimension:
            return None
        matched.append(key)

    for index in range(60):
        key = prefix + f"transformer_blocks.{index}.attn.norm_q.weight"
        geometry = _entry_geometry(source, keys, key)
        if geometry is None or geometry.shape != (128,):
            return None
        matched.append(key)

    index_timestep_zero = prefix + _INDEX_TIMESTEP_ZERO
    if index_timestep_zero in keys:
        geometry = _entry_geometry(source, keys, index_timestep_zero)
        if geometry is None or geometry.shape != (0,):
            return None
        matched.append(index_timestep_zero)
    additional_t_cond = prefix + _ADDITIONAL_T_COND
    if additional_t_cond in keys:
        geometry = _entry_geometry(source, keys, additional_t_cond)
        if geometry is None or geometry.shape != (2, 3072):
            return None
        matched.append(additional_t_cond)
    return tuple(sorted(matched))


def detect_qwen_image(source: WeightSource) -> QwenImageEvidence | None:
    """Detect only the exact base Qwen Image header, or fail closed."""

    keys = frozenset(source.keys())
    for prefix in _PREFIXES:
        matched_keys = _matches_prefix(source, keys, prefix)
        if matched_keys is None:
            continue
        if prefix + _ADDITIONAL_T_COND in keys:
            config = QWEN_IMAGE_LAYERED_CONFIG
        elif prefix + _INDEX_TIMESTEP_ZERO in keys:
            config = QWEN_IMAGE_EDIT_2511_CONFIG
        else:
            config = QWEN_IMAGE_CONFIG
        return QwenImageEvidence(
            config=config,
            key_prefix=prefix,
            matched_keys=matched_keys,
            fields={
                "attention_head_dim": config.attention_head_dim,
                "attention_heads": config.attention_heads,
                "default_ref_method": config.default_ref_method,
                "hidden_width": config.hidden_width,
                "key_prefix": prefix,
                "output_latent_channels": config.output_latent_channels,
                "patch": "2x2",
                "patchified_input_channels": config.patchified_input_channels,
                "pooled_width": config.pooled_width,
                "text_width": config.text_width,
                "transformer_blocks": config.transformer_blocks,
                "use_additional_t_cond": config.use_additional_t_cond,
            },
        )
    return None


__all__ = [
    "QWEN_IMAGE_CONFIG",
    "QWEN_IMAGE_EDIT_2511_CONFIG",
    "QWEN_IMAGE_LAYERED_CONFIG",
    "QwenImageConfig",
    "QwenImageEvidence",
    "detect_qwen_image",
]
