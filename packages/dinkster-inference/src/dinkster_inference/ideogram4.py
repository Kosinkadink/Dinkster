"""Exact Ideogram 4 diffusion profile and fail-closed detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .quantization import QuantizationError, split_quantization
from .spaces import FlowSigmas
from .weights import ConfigurationPayloadSource, TensorGeometry, WeightSource


class Ideogram4DetectError(ValueError):
    """A header is not the published Ideogram 4 diffusion model."""


_PREFIXES = ("model.diffusion_model.", "")


@dataclass(frozen=True)
class Ideogram4Config:
    family_id: str = "dinkster.ideogram4"
    hidden_size: int = 4608
    layers: int = 34
    attention_heads: int = 18
    attention_head_dim: int = 256
    intermediate_size: int = 12288
    adaln_dim: int = 512
    latent_channels: int = 128
    ae_channels: int = 32
    patch: tuple[int, int] = (2, 2)
    text_width: int = 53248
    rope_theta: float = 5_000_000.0
    rope_dims: tuple[int, int, int] = (24, 20, 20)
    norm_eps: float = 1e-5
    sampling_shift: float = 1.0
    inference_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT32)
    memory_factor: float = 11.6
    text_encoder_id: str = "Qwen3-VL-8B"

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "dinkster.ideogram4",
            4608,
            34,
            18,
            256,
            12288,
            512,
            128,
            32,
            (2, 2),
            53248,
            5_000_000.0,
            (24, 20, 20),
            1e-5,
            1.0,
            (BFLOAT16, FLOAT32),
            11.6,
            "Qwen3-VL-8B",
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("Ideogram4Config only represents the published Ideogram 4 model")


IDEOGRAM4_CONFIG = Ideogram4Config()
IDEOGRAM4_SIGMAS = FlowSigmas(shift=IDEOGRAM4_CONFIG.sampling_shift)


def ideogram4_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 458-entry Ideogram 4 diffusion layout."""

    config = IDEOGRAM4_CONFIG
    hidden = config.hidden_size
    layout: dict[str, tuple[int, ...]] = {
        "input_proj.weight": (hidden, config.latent_channels),
        "input_proj.bias": (hidden,),
        "llm_cond_norm.weight": (config.text_width,),
        "llm_cond_proj.weight": (hidden, config.text_width),
        "llm_cond_proj.bias": (hidden,),
        "t_embedding.mlp_in.weight": (hidden, hidden),
        "t_embedding.mlp_in.bias": (hidden,),
        "t_embedding.mlp_out.weight": (hidden, hidden),
        "t_embedding.mlp_out.bias": (hidden,),
        "adaln_proj.weight": (config.adaln_dim, hidden),
        "adaln_proj.bias": (config.adaln_dim,),
        "embed_image_indicator.weight": (2, hidden),
        "final_layer.linear.weight": (config.latent_channels, hidden),
        "final_layer.linear.bias": (config.latent_channels,),
        "final_layer.adaln_modulation.weight": (hidden, config.adaln_dim),
        "final_layer.adaln_modulation.bias": (hidden,),
    }
    block = {
        "attention.qkv.weight": (hidden * 3, hidden),
        "attention.norm_q.weight": (config.attention_head_dim,),
        "attention.norm_k.weight": (config.attention_head_dim,),
        "attention.o.weight": (hidden, hidden),
        "feed_forward.w1.weight": (config.intermediate_size, hidden),
        "feed_forward.w2.weight": (hidden, config.intermediate_size),
        "feed_forward.w3.weight": (config.intermediate_size, hidden),
        "attention_norm1.weight": (hidden,),
        "attention_norm2.weight": (hidden,),
        "ffn_norm1.weight": (hidden,),
        "ffn_norm2.weight": (hidden,),
        "adaln_modulation.weight": (hidden * 4, config.adaln_dim),
        "adaln_modulation.bias": (hidden * 4,),
    }
    for index in range(config.layers):
        layout.update({f"layers.{index}.{key}": shape for key, shape in block.items()})
    return MappingProxyType(layout)


def detect_ideogram4_config(geometries: Mapping[str, TensorGeometry]) -> Ideogram4Config:
    if "embed_image_indicator.weight" not in geometries:
        raise Ideogram4DetectError("not Ideogram 4 (no embed_image_indicator.weight)")
    layout = ideogram4_layout()
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise Ideogram4DetectError(f"geometry does not match Ideogram 4: {shown}")
    return IDEOGRAM4_CONFIG


@dataclass(frozen=True)
class Ideogram4Evidence:
    config: Ideogram4Config
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_ideogram4(source: WeightSource) -> Ideogram4Evidence | None:
    all_keys = tuple(source.keys())
    all_geometries = {key: source.entry(key).geometry for key in all_keys}
    for prefix in _PREFIXES:
        try:
            split = split_quantization(
                all_geometries,
                source.metadata(),
                prefix=prefix,
                payload_reader=(
                    source.read_uint8_configuration
                    if isinstance(source, ConfigurationPayloadSource)
                    else None
                ),
            )
            config = detect_ideogram4_config(split.architecture)
        except (Ideogram4DetectError, QuantizationError):
            continue
        return Ideogram4Evidence(
            config,
            prefix,
            tuple(sorted(key for key in all_keys if key.startswith(prefix))),
            {
                "attention_heads": config.attention_heads,
                "hidden_size": config.hidden_size,
                "key_prefix": prefix,
                "layers": config.layers,
                "text_width": config.text_width,
            },
        )
    return None


__all__ = [
    "IDEOGRAM4_CONFIG",
    "IDEOGRAM4_SIGMAS",
    "Ideogram4Config",
    "Ideogram4DetectError",
    "Ideogram4Evidence",
    "detect_ideogram4",
    "detect_ideogram4_config",
    "ideogram4_layout",
]
