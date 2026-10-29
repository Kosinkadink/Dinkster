"""Exact Qwen3-30B-A3B sparse-MoE profile and checkpoint layout.

The accepted profile is Qwen/Qwen3-30B-A3B at Hugging Face revision
``d47d535f78ec44bd57128f8e8aeba17eeb0285ea``. Every decoder layer is sparse:
128 independent SwiGLU experts, eight selected per token, and no shared expert.
Detection accepts only the original unpacked safetensors layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .devices import BFLOAT16
from .weights import TensorGeometry


class Qwen3MoeDetectError(ValueError):
    """A checkpoint header is not the accepted Qwen3-30B-A3B profile."""


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """Construction facts needed for Qwen3 sparse-MoE causal inference."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    norm_topk_prob: bool
    qkv_bias: bool
    qk_norm: bool
    tie_word_embeddings: bool
    eos_token_id: int

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "num_experts",
            "num_experts_per_tok",
            "decoder_sparse_step",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("RMS epsilon and RoPE theta must be positive")
        if type(self.mlp_only_layers) is not tuple or any(
            type(index) is not int or not 0 <= index < self.num_hidden_layers
            for index in self.mlp_only_layers
        ):
            raise ValueError("mlp_only_layers must be a tuple of decoder-layer indices")
        if len(set(self.mlp_only_layers)) != len(self.mlp_only_layers):
            raise ValueError("mlp_only_layers must not contain duplicates")
        if self.decoder_sparse_step != 1 or self.mlp_only_layers:
            raise ValueError("only all-sparse decoder layers are supported")
        if type(self.eos_token_id) is not int or not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be within the vocabulary")


QWEN3_30B_A3B_CONFIG = Qwen3MoeConfig(
    architecture="qwen3_30b_a3b",
    vocab_size=151936,
    hidden_size=2048,
    intermediate_size=6144,
    moe_intermediate_size=768,
    num_hidden_layers=48,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    num_experts=128,
    num_experts_per_tok=8,
    decoder_sparse_step=1,
    mlp_only_layers=(),
    norm_topk_prob=True,
    qkv_bias=False,
    qk_norm=True,
    tie_word_embeddings=False,
    eos_token_id=151645,
)


def qwen3_moe_layout(config: Qwen3MoeConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact original ``Qwen3MoeForCausalLM`` state layout."""
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    expert_hidden = config.moe_intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        layout[f"{prefix}input_layernorm.weight"] = (hidden,)
        layout[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        layout[f"{prefix}self_attn.q_proj.weight"] = (query, hidden)
        layout[f"{prefix}self_attn.k_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.v_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.o_proj.weight"] = (hidden, query)
        if config.qkv_bias:
            layout[f"{prefix}self_attn.q_proj.bias"] = (query,)
            layout[f"{prefix}self_attn.k_proj.bias"] = (kv,)
            layout[f"{prefix}self_attn.v_proj.bias"] = (kv,)
        if config.qk_norm:
            layout[f"{prefix}self_attn.q_norm.weight"] = (config.head_dim,)
            layout[f"{prefix}self_attn.k_norm.weight"] = (config.head_dim,)
        layout[f"{prefix}mlp.gate.weight"] = (config.num_experts, hidden)
        for expert in range(config.num_experts):
            expert_prefix = f"{prefix}mlp.experts.{expert}."
            layout[f"{expert_prefix}gate_proj.weight"] = (expert_hidden, hidden)
            layout[f"{expert_prefix}up_proj.weight"] = (expert_hidden, hidden)
            layout[f"{expert_prefix}down_proj.weight"] = (hidden, expert_hidden)
    return layout


def detect_qwen3_moe_config(
    geometries: Mapping[str, TensorGeometry],
) -> Qwen3MoeConfig:
    """Accept only the complete original Qwen3-30B-A3B checkpoint layout."""
    if not geometries:
        raise Qwen3MoeDetectError("empty state dict header")
    expected = qwen3_moe_layout(QWEN3_30B_A3B_CONFIG)
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
        elif found.dtype != BFLOAT16:
            problems.append(f"{key}: expected dtype bfloat16, found {found.dtype.name}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if not problems:
        return QWEN3_30B_A3B_CONFIG
    shown = "; ".join(problems[:6])
    if len(problems) > 6:
        shown += f"; and {len(problems) - 6} more"
    raise Qwen3MoeDetectError("geometry does not match qwen3_30b_a3b: " + shown)


__all__ = [
    "QWEN3_30B_A3B_CONFIG",
    "Qwen3MoeConfig",
    "Qwen3MoeDetectError",
    "detect_qwen3_moe_config",
    "qwen3_moe_layout",
]
