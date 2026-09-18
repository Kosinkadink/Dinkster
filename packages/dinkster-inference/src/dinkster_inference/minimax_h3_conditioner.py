"""Exact torch-free Qwen3-VL-32B conditioner layout for MiniMax H3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .weights import TensorGeometry


class MiniMaxH3ConditionerDetectError(ValueError):
    """A header is not the exact truncated H3 conditioner profile."""


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConditionerConfig:
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    layers: int = 50
    attention_heads: int = 64
    key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    rope_dimensions: tuple[int, int, int] = (24, 20, 20)
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    final_norm: bool = False
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_layers: int = 27
    vision_heads: int = 16
    vision_patch_size: int = 16
    vision_temporal_patch: int = 2
    vision_merge_size: int = 2
    vision_position_embeddings: int = 2304
    deepstack_layers: tuple[int, int, int] = (8, 16, 24)

    def __post_init__(self) -> None:
        values = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.layers,
            self.attention_heads,
            self.key_value_heads,
            self.head_dim,
            self.rope_theta,
            self.rope_dimensions,
            self.max_position_embeddings,
            self.rms_norm_eps,
            self.final_norm,
            self.vision_hidden_size,
            self.vision_intermediate_size,
            self.vision_layers,
            self.vision_heads,
            self.vision_patch_size,
            self.vision_temporal_patch,
            self.vision_merge_size,
            self.vision_position_embeddings,
            self.deepstack_layers,
        )
        if values != (
            151936,
            5120,
            25600,
            50,
            64,
            8,
            128,
            5_000_000.0,
            (24, 20, 20),
            262144,
            1e-6,
            False,
            1152,
            4304,
            27,
            16,
            16,
            2,
            2,
            2304,
            (8, 16, 24),
        ):
            raise ValueError("only the exact MiniMax H3 conditioner profile is supported")


MINIMAX_H3_CONDITIONER_CONFIG = MiniMaxH3ConditionerConfig()


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConditionerLayout:
    config: MiniMaxH3ConditionerConfig
    keys: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.config != MINIMAX_H3_CONDITIONER_CONFIG:
            raise ValueError("layout requires the exact H3 conditioner profile")
        snapshot = dict(self.keys)
        if snapshot != _conditioner_keys(self.config):
            raise ValueError("conditioner layout keys or shapes are not exact")
        object.__setattr__(self, "keys", MappingProxyType(snapshot))


def _conditioner_keys(config: MiniMaxH3ConditionerConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    kv = config.key_value_heads * config.head_dim
    keys: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (config.vocab_size, hidden)}
    for index in range(config.layers):
        prefix = f"model.layers.{index}."
        keys[f"{prefix}input_layernorm.weight"] = (hidden,)
        keys[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        keys[f"{prefix}self_attn.q_proj.weight"] = (
            config.attention_heads * config.head_dim,
            hidden,
        )
        keys[f"{prefix}self_attn.k_proj.weight"] = (kv, hidden)
        keys[f"{prefix}self_attn.v_proj.weight"] = (kv, hidden)
        keys[f"{prefix}self_attn.o_proj.weight"] = (
            hidden,
            config.attention_heads * config.head_dim,
        )
        keys[f"{prefix}self_attn.q_norm.weight"] = (config.head_dim,)
        keys[f"{prefix}self_attn.k_norm.weight"] = (config.head_dim,)
        keys[f"{prefix}mlp.gate_proj.weight"] = (config.intermediate_size, hidden)
        keys[f"{prefix}mlp.up_proj.weight"] = (config.intermediate_size, hidden)
        keys[f"{prefix}mlp.down_proj.weight"] = (hidden, config.intermediate_size)

    vision = config.vision_hidden_size
    vision_intermediate = config.vision_intermediate_size
    merge = vision * config.vision_merge_size**2
    keys["visual.patch_embed.proj.weight"] = (
        vision,
        3,
        config.vision_temporal_patch,
        config.vision_patch_size,
        config.vision_patch_size,
    )
    keys["visual.patch_embed.proj.bias"] = (vision,)
    keys["visual.pos_embed.weight"] = (config.vision_position_embeddings, vision)
    for index in range(config.vision_layers):
        prefix = f"visual.blocks.{index}."
        for norm in ("norm1", "norm2"):
            keys[f"{prefix}{norm}.weight"] = (vision,)
            keys[f"{prefix}{norm}.bias"] = (vision,)
        keys[f"{prefix}attn.qkv.weight"] = (vision * 3, vision)
        keys[f"{prefix}attn.qkv.bias"] = (vision * 3,)
        keys[f"{prefix}attn.proj.weight"] = (vision, vision)
        keys[f"{prefix}attn.proj.bias"] = (vision,)
        keys[f"{prefix}mlp.linear_fc1.weight"] = (vision_intermediate, vision)
        keys[f"{prefix}mlp.linear_fc1.bias"] = (vision_intermediate,)
        keys[f"{prefix}mlp.linear_fc2.weight"] = (vision, vision_intermediate)
        keys[f"{prefix}mlp.linear_fc2.bias"] = (vision,)
    keys["visual.merger.norm.weight"] = (vision,)
    keys["visual.merger.norm.bias"] = (vision,)
    keys["visual.merger.linear_fc1.weight"] = (merge, merge)
    keys["visual.merger.linear_fc1.bias"] = (merge,)
    keys["visual.merger.linear_fc2.weight"] = (hidden, merge)
    keys["visual.merger.linear_fc2.bias"] = (hidden,)
    for index in range(len(config.deepstack_layers)):
        prefix = f"visual.deepstack_merger_list.{index}."
        keys[f"{prefix}norm.weight"] = (merge,)
        keys[f"{prefix}norm.bias"] = (merge,)
        keys[f"{prefix}linear_fc1.weight"] = (merge, merge)
        keys[f"{prefix}linear_fc1.bias"] = (merge,)
        keys[f"{prefix}linear_fc2.weight"] = (hidden, merge)
        keys[f"{prefix}linear_fc2.bias"] = (hidden,)
    return keys


def minimax_h3_conditioner_layout() -> MiniMaxH3ConditionerLayout:
    return MiniMaxH3ConditionerLayout(
        MINIMAX_H3_CONDITIONER_CONFIG,
        _conditioner_keys(MINIMAX_H3_CONDITIONER_CONFIG),
    )


def detect_minimax_h3_conditioner(
    geometries: Mapping[str, TensorGeometry],
) -> MiniMaxH3ConditionerConfig:
    """Accept only the complete exact model-scoped H3 conditioner header."""
    expected = minimax_h3_conditioner_layout().keys
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected {shape}, found {found.shape}")
    problems.extend(f"unexpected {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        detail = "; ".join(problems[:6])
        if len(problems) > 6:
            detail += f"; and {len(problems) - 6} more"
        raise MiniMaxH3ConditionerDetectError("conditioner geometry is not exact: " + detail)
    return MINIMAX_H3_CONDITIONER_CONFIG


__all__ = [
    "MINIMAX_H3_CONDITIONER_CONFIG",
    "MiniMaxH3ConditionerConfig",
    "MiniMaxH3ConditionerDetectError",
    "MiniMaxH3ConditionerLayout",
    "detect_minimax_h3_conditioner",
    "minimax_h3_conditioner_layout",
]
