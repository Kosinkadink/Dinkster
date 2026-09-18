"""Exact Lumina Image 2.0 profile and fail-closed header detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .spaces import FlowSigmas
from .weights import WeightSource

_PREFIXES = ("model.diffusion_model.", "")


@dataclass(frozen=True)
class Lumina2Config:
    family_id: str = "dinkster.lumina2"
    hidden_width: int = 2304
    caption_width: int = 2304
    main_blocks: int = 26
    noise_refiner_blocks: int = 2
    context_refiner_blocks: int = 2
    attention_heads: int = 24
    kv_heads: int = 8
    attention_head_dim: int = 96
    ffn_width: int = 9216
    latent_channels: int = 16
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (32, 32, 32)
    rope_theta: float = 10000.0
    qk_norm_eps: float = 1e-5
    timestep_embedding_width: int = 256
    modulation_width: int = 1024
    timestep_multiplier: float = 1.0
    block_modulation_silu: bool = True
    pad_tokens_multiple: int = 1
    learned_padding: bool = False
    sampling_shift: float = 6.0
    inference_dtypes: tuple[DType, DType] = (BFLOAT16, FLOAT32)
    memory_factor: float = 1.4

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "dinkster.lumina2",
            2304,
            2304,
            26,
            2,
            2,
            24,
            8,
            96,
            9216,
            16,
            (2, 2),
            (32, 32, 32),
            10000.0,
            1e-5,
            256,
            1024,
            1.0,
            True,
            1,
            False,
            6.0,
            (BFLOAT16, FLOAT32),
            1.4,
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("Lumina2Config only represents exact Lumina Image 2.0")


LUMINA2_CONFIG = Lumina2Config()
LUMINA2_SIGMAS = FlowSigmas(shift=LUMINA2_CONFIG.sampling_shift)


def _block_layout(config: Lumina2Config, *, modulated: bool) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_width
    query = config.attention_heads * config.attention_head_dim
    kv = config.kv_heads * config.attention_head_dim
    layout = {
        "attention.qkv.weight": (query + 2 * kv, hidden),
        "attention.out.weight": (hidden, query),
        "attention.q_norm.weight": (config.attention_head_dim,),
        "attention.k_norm.weight": (config.attention_head_dim,),
        "attention_norm1.weight": (hidden,),
        "attention_norm2.weight": (hidden,),
        "ffn_norm1.weight": (hidden,),
        "ffn_norm2.weight": (hidden,),
        "feed_forward.w1.weight": (config.ffn_width, hidden),
        "feed_forward.w2.weight": (hidden, config.ffn_width),
        "feed_forward.w3.weight": (config.ffn_width, hidden),
    }
    if modulated:
        layout["adaLN_modulation.1.weight"] = (4 * hidden, config.modulation_width)
        layout["adaLN_modulation.1.bias"] = (4 * hidden,)
    return layout


def lumina2_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 400-entry Lumina Image 2.0 diffusion layout."""
    config = LUMINA2_CONFIG
    hidden = config.hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "cap_embedder.0.weight": (config.caption_width,),
        "cap_embedder.1.weight": (hidden, config.caption_width),
        "cap_embedder.1.bias": (hidden,),
        "x_embedder.weight": (hidden, config.latent_channels * 4),
        "x_embedder.bias": (hidden,),
        "t_embedder.mlp.0.weight": (1024, config.timestep_embedding_width),
        "t_embedder.mlp.0.bias": (1024,),
        "t_embedder.mlp.2.weight": (config.modulation_width, 1024),
        "t_embedder.mlp.2.bias": (config.modulation_width,),
        "final_layer.adaLN_modulation.1.weight": (hidden, config.modulation_width),
        "final_layer.adaLN_modulation.1.bias": (hidden,),
        "final_layer.linear.weight": (config.latent_channels * 4, hidden),
        "final_layer.linear.bias": (config.latent_channels * 4,),
        # Present in official and Neta checkpoints but unused by the reference model.
        "norm_final.weight": (hidden,),
    }
    for root, count, modulated in (
        ("context_refiner", config.context_refiner_blocks, False),
        ("noise_refiner", config.noise_refiner_blocks, True),
        ("layers", config.main_blocks, True),
    ):
        block = _block_layout(config, modulated=modulated)
        for index in range(count):
            layout.update({f"{root}.{index}.{key}": shape for key, shape in block.items()})
    return MappingProxyType(layout)


@dataclass(frozen=True)
class Lumina2Evidence:
    config: Lumina2Config
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_lumina2(source: WeightSource) -> Lumina2Evidence | None:
    """Admit only the complete Lumina Image 2.0 diffusion geometry."""
    keys = frozenset(source.keys())
    config = LUMINA2_CONFIG
    layout = lumina2_layout()
    for prefix in _PREFIXES:
        expected_keys = frozenset(prefix + key for key in layout)
        candidate_keys = (
            keys if not prefix else frozenset(key for key in keys if key.startswith(prefix))
        )
        if candidate_keys != expected_keys:
            continue
        matched: list[str] = []
        for suffix, shape in layout.items():
            key = prefix + suffix
            try:
                geometry = source.entry(key).geometry
            except KeyError:
                break
            if geometry.shape != shape or geometry.dtype.kind != "float":
                break
            matched.append(key)
        else:
            return Lumina2Evidence(
                config,
                prefix,
                tuple(sorted(matched)),
                {
                    "caption_width": config.caption_width,
                    "hidden_width": config.hidden_width,
                    "key_prefix": prefix,
                    "main_blocks": config.main_blocks,
                    "sampling_shift": config.sampling_shift,
                },
            )
    return None


__all__ = [
    "LUMINA2_CONFIG",
    "LUMINA2_SIGMAS",
    "Lumina2Config",
    "Lumina2Evidence",
    "detect_lumina2",
    "lumina2_layout",
]
