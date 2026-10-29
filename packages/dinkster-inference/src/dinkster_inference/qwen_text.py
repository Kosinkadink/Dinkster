"""Llama-style text towers shared by Ovis, Z-Image, Flux2, and Anima.

Each accepted profile mirrors one config from ComfyUI's
``comfy/text_encoders/llama.py``: Ovis 2.5/Qwen3-2B
(``Ovis25_2BConfig``), Z-Image Qwen3-4B (``Qwen3_4BConfig``), Flux2
Klein Qwen3-8B (``Qwen3_8BConfig``), and Flux2 dev Mistral3-Small 24B
(``Mistral3Small24BConfig``, full and layer-pruned) @ b78cec87, and
Anima Qwen3-0.6B (``Qwen3_06BConfig``) @ 82f839f5. Checkpoints may
carry non-tower siblings outside ``model.*`` (Ovis vision modules,
Qwen3-8B ``lm_head.weight``, Mistral3 ``vision_tower.*``,
``multi_modal_projector.*``, and ``tekken_model``); the caller extracts
only ``model.*`` and decides what to do with the rest.

Detection is exact. Attention head counts, KV head counts, RoPE policy,
and q/k normalization are not derivable from arbitrary tensor geometry,
so the detector accepts only a full known layout and refuses every
missing, extra, or wrong-width model key. Profiles that share a byte
layout (Z-Image and Klein both run the Qwen3-4B tower) cannot be told
apart by detection; the detector returns the canonical tower config and
the consuming family substitutes its prompt-policy variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .weights import TensorGeometry


class QwenTextDetectError(ValueError):
    """A model-scoped header is not an accepted Qwen text profile."""


@dataclass(frozen=True)
class QwenTextConfig:
    """Construction and prompt-policy facts for one Qwen text tower."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    qkv_bias: bool
    qk_norm: bool
    prompt_template: str
    min_tokens: int
    pad_token_id: int
    # Field order and repr visibility down to zero_masked are part of the
    # runtime structural identity (identity.py hashes repr(config)).
    slice_marker_id: int | None = None
    slice_marker_suffix_id: int | None = None
    zero_masked: bool = False
    attention_head_dim: int | None = field(default=None, repr=False)
    output_hidden_layer: int | None = field(default=None, repr=False)
    output_hidden_layers: tuple[int, ...] | None = field(default=None, repr=False)
    layer_norm_hidden_state: bool = field(default=True, repr=False)
    final_norm: bool = field(default=True, repr=False)
    merged_qkv: bool = field(default=False, repr=False)
    merged_mlp: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "min_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.attention_head_dim is None and self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.attention_head_dim is not None and self.attention_head_dim < 1:
            raise ValueError("attention_head_dim must be positive or None")
        if self.output_hidden_layer is not None and not (
            -self.num_hidden_layers <= self.output_hidden_layer < self.num_hidden_layers
        ):
            raise ValueError("output_hidden_layer must address a model layer or be None")
        if self.output_hidden_layers is not None:
            if self.output_hidden_layer is not None:
                raise ValueError("output_hidden_layer and output_hidden_layers are exclusive")
            if not self.output_hidden_layers:
                raise ValueError("output_hidden_layers must name at least one layer or be None")
            if any(not 0 <= index < self.num_hidden_layers for index in self.output_hidden_layers):
                raise ValueError("output_hidden_layers entries must address model layers")
            if any(
                later <= earlier
                for earlier, later in zip(
                    self.output_hidden_layers, self.output_hidden_layers[1:], strict=False
                )
            ):
                raise ValueError("output_hidden_layers must be strictly increasing")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("RMS epsilon and RoPE theta must be positive")
        for name in ("pad_token_id", "slice_marker_id", "slice_marker_suffix_id"):
            value = getattr(self, name)
            if value is not None and not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} {value} is outside the vocabulary")
        if (self.slice_marker_id is None) != (self.slice_marker_suffix_id is None):
            raise ValueError("slice marker tokens must be set together or not at all")
        if self.prompt_template.count("{}") != 1:
            raise ValueError("prompt_template must carry exactly one text slot")

    @property
    def head_dim(self) -> int:
        return (
            self.hidden_size // self.num_attention_heads
            if self.attention_head_dim is None
            else self.attention_head_dim
        )


OVIS_QWEN3_2B_CONFIG = QwenTextConfig(
    architecture="ovis_qwen3_2b",
    vocab_size=151936,
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    qkv_bias=False,
    qk_norm=True,
    prompt_template=(
        "<|im_start|>user\nDescribe the image by detailing the color, quantity,"
        " text, shape, size, texture, spatial relationships of the objects and"
        " background: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    min_tokens=284,
    pad_token_id=151643,
    slice_marker_id=4004,
    slice_marker_suffix_id=25,
    zero_masked=True,
)

# Anima runs the raw prompt through this tower with no chat template;
# the emphasis grammar is parsed but its weights are discarded on this
# side (comfy/text_encoders/anima.py @ 82f839f5). head_dim 128 is an
# explicit override: query width 2048 exceeds the 1024 hidden width.
ANIMA_QWEN3_06B_CONFIG = QwenTextConfig(
    architecture="anima_qwen3_06b",
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=32768,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    qkv_bias=False,
    qk_norm=True,
    prompt_template="{}",
    min_tokens=1,
    pad_token_id=151643,
    zero_masked=False,
    attention_head_dim=128,
    layer_norm_hidden_state=False,
)

Z_IMAGE_QWEN3_4B_CONFIG = QwenTextConfig(
    architecture="z_image_qwen3_4b",
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    qkv_bias=False,
    qk_norm=True,
    prompt_template="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    min_tokens=1,
    pad_token_id=151643,
    slice_marker_id=151644,
    slice_marker_suffix_id=151645,
    zero_masked=False,
    attention_head_dim=128,
    output_hidden_layer=-2,
    layer_norm_hidden_state=False,
)

_KLEIN_PROMPT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# Same Qwen3-4B tower bytes as Z-Image; only the Flux2 Klein prompt policy
# differs, so this profile stays out of detection (see module docstring).
KLEIN_QWEN3_4B_CONFIG = replace(
    Z_IMAGE_QWEN3_4B_CONFIG,
    architecture="klein_qwen3_4b",
    prompt_template=_KLEIN_PROMPT_TEMPLATE,
    min_tokens=512,
    slice_marker_id=None,
    slice_marker_suffix_id=None,
    output_hidden_layer=None,
    output_hidden_layers=(8, 17, 26),
)

KLEIN_QWEN3_8B_CONFIG = QwenTextConfig(
    architecture="klein_qwen3_8b",
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    qkv_bias=False,
    qk_norm=True,
    prompt_template=_KLEIN_PROMPT_TEMPLATE,
    min_tokens=512,
    pad_token_id=151643,
    zero_masked=False,
    attention_head_dim=128,
    output_hidden_layers=(8, 17, 26),
    layer_norm_hidden_state=False,
)

_MISTRAL3_PROMPT_TEMPLATE = (
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions."
    " You give structured responses focusing on object relationships, object\n"
    "attribution and actions without speculation.[/SYSTEM_PROMPT][INST]{}[/INST]"
)

MISTRAL3_24B_CONFIG = QwenTextConfig(
    architecture="mistral3_24b",
    vocab_size=131072,
    hidden_size=5120,
    intermediate_size=32768,
    num_hidden_layers=40,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=1_000_000_000.0,
    qkv_bias=False,
    qk_norm=False,
    prompt_template=_MISTRAL3_PROMPT_TEMPLATE,
    min_tokens=1,
    pad_token_id=11,
    zero_masked=False,
    attention_head_dim=128,
    output_hidden_layers=(9, 19, 29),
    layer_norm_hidden_state=False,
)

# The official flux2-dev release ships this layer-pruned variant (layers
# 30..39 removed). Its checkpoint still carries model.norm.weight, but the
# reference constructs the model with final_norm=False, so the tensor loads
# and is never applied.
MISTRAL3_24B_PRUNED_CONFIG = replace(
    MISTRAL3_24B_CONFIG,
    architecture="mistral3_24b_pruned",
    num_hidden_layers=30,
    final_norm=False,
)

# ACE uses the Qwen tokenizer's exact vocabulary on the conditioner and
# an extended audio-code vocabulary on the tied-head language models.
ACE15_QWEN3_06B_CONFIG = replace(
    ANIMA_QWEN3_06B_CONFIG,
    architecture="qwen3_06b_ace15",
    vocab_size=151669,
)
ACE15_QWEN3_2B_CONFIG = replace(
    ACE15_QWEN3_06B_CONFIG,
    architecture="qwen3_2b_ace15_lm",
    vocab_size=217204,
    hidden_size=2048,
    intermediate_size=6144,
    max_position_embeddings=40960,
)
ACE15_QWEN3_4B_CONFIG = replace(
    ACE15_QWEN3_2B_CONFIG,
    architecture="qwen3_4b_ace15_lm",
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
)

_KNOWN_QWEN_TEXT_CONFIGS = (
    OVIS_QWEN3_2B_CONFIG,
    ANIMA_QWEN3_06B_CONFIG,
    ACE15_QWEN3_06B_CONFIG,
    ACE15_QWEN3_2B_CONFIG,
    ACE15_QWEN3_4B_CONFIG,
    Z_IMAGE_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
)


def qwen_text_layout(config: QwenTextConfig) -> dict[str, tuple[int, ...]]:
    """Exact ``model.*``-stripped state dictionary for ``config``."""
    hidden = config.hidden_size
    head = config.head_dim
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (config.vocab_size, hidden),
        "norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}."
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
            layout[f"{prefix}self_attn.q_norm.weight"] = (head,)
            layout[f"{prefix}self_attn.k_norm.weight"] = (head,)
        layout[f"{prefix}mlp.gate_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.up_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.down_proj.weight"] = (hidden, inter)
    return layout


def detect_qwen_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> QwenTextConfig:
    """Accept one complete known Qwen text-model subtree."""
    if not geometries:
        raise QwenTextDetectError("empty state dict header")
    embedding = geometries.get("embed_tokens.weight")
    if embedding is None:
        raise QwenTextDetectError("not a Qwen text-model role (missing model.embed_tokens.weight)")
    if len(embedding.shape) != 2:
        raise QwenTextDetectError(
            f"embed_tokens.weight has rank {len(embedding.shape)}, expected 2"
        )
    candidates = [
        candidate
        for candidate in _KNOWN_QWEN_TEXT_CONFIGS
        if embedding.shape == (candidate.vocab_size, candidate.hidden_size)
    ]
    if not candidates:
        accepted = ", ".join(
            f"{candidate.architecture} ({candidate.vocab_size}, {candidate.hidden_size})"
            for candidate in _KNOWN_QWEN_TEXT_CONFIGS
        )
        raise QwenTextDetectError(
            "unknown Qwen text width: token embedding"
            f" {embedding.shape}; accepted profiles are {accepted}"
        )

    best: tuple[QwenTextConfig, list[str]] | None = None
    for config in candidates:
        layout = qwen_text_layout(config)
        problems: list[str] = []
        for key, shape in layout.items():
            found = geometries.get(key)
            if found is None:
                problems.append(f"missing {key}")
            elif found.shape != shape:
                problems.append(f"{key}: expected shape {shape}, found {found.shape}")
        problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
        if not problems:
            return config
        if best is None or len(problems) < len(best[1]):
            best = (config, problems)
    assert best is not None
    config, problems = best
    shown = "; ".join(problems[:6])
    if len(problems) > 6:
        shown += f"; and {len(problems) - 6} more"
    raise QwenTextDetectError(
        f"geometry does not match the {config.architecture} text layout: " + shown
    )


__all__ = [
    "ACE15_QWEN3_06B_CONFIG",
    "ACE15_QWEN3_2B_CONFIG",
    "ACE15_QWEN3_4B_CONFIG",
    "ANIMA_QWEN3_06B_CONFIG",
    "KLEIN_QWEN3_4B_CONFIG",
    "KLEIN_QWEN3_8B_CONFIG",
    "MISTRAL3_24B_CONFIG",
    "MISTRAL3_24B_PRUNED_CONFIG",
    "OVIS_QWEN3_2B_CONFIG",
    "Z_IMAGE_QWEN3_4B_CONFIG",
    "QwenTextConfig",
    "QwenTextDetectError",
    "detect_qwen_text_config",
    "qwen_text_layout",
]
