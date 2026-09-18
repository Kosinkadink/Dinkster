"""Torch-free Ideogram 4 Qwen3-VL-8B text contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .weights import TensorGeometry


class Ideogram4TextDetectError(ValueError):
    """A header is not the official Ideogram 4 Qwen3-VL-8B text role."""


IDEOGRAM4_TAP_LAYERS = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36)
_PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class Ideogram4TextConfig:
    architecture: str = "ideogram4_qwen3vl_8b"
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    rope_dims: tuple[int, int, int] = (24, 20, 20)
    interleaved_mrope: bool = True
    qkv_bias: bool = False
    qk_norm: bool = True
    final_norm: bool = True
    tap_layers: tuple[int, ...] = IDEOGRAM4_TAP_LAYERS
    prompt_template: str = _PROMPT_TEMPLATE
    pad_token_id: int = 151643
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_layers: int = 27
    vision_heads: int = 16
    vision_patch: tuple[int, int, int] = (2, 16, 16)
    vision_merge_size: int = 2
    vision_position_embeddings: int = 2304
    deepstack_layers: tuple[int, int, int] = (8, 16, 24)

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "ideogram4_qwen3vl_8b",
            151936,
            4096,
            12288,
            36,
            32,
            8,
            128,
            262144,
            1e-6,
            5_000_000.0,
            (24, 20, 20),
            True,
            False,
            True,
            True,
            IDEOGRAM4_TAP_LAYERS,
            _PROMPT_TEMPLATE,
            151643,
            1152,
            4304,
            27,
            16,
            (2, 16, 16),
            2,
            2304,
            (8, 16, 24),
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("Ideogram4TextConfig only represents Qwen3-VL-8B")

    @property
    def stack_width(self) -> int:
        return len(self.tap_layers) * self.hidden_size


IDEOGRAM4_TEXT_CONFIG = Ideogram4TextConfig()
IDEOGRAM4_LANGUAGE_SUBTREE = "model."
IDEOGRAM4_VISION_SUBTREE = "model.visual."


def ideogram4_text_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the official 750-entry full Qwen3-VL-8B layout."""

    config = IDEOGRAM4_TEXT_CONFIG
    hidden = config.hidden_size
    head = config.head_dim
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "lm_head.weight": (config.vocab_size, hidden),
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        layout[f"{prefix}input_layernorm.weight"] = (hidden,)
        layout[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        layout[f"{prefix}self_attn.q_proj.weight"] = (query, hidden)
        layout[f"{prefix}self_attn.k_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.v_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.o_proj.weight"] = (hidden, query)
        layout[f"{prefix}self_attn.q_norm.weight"] = (head,)
        layout[f"{prefix}self_attn.k_norm.weight"] = (head,)
        layout[f"{prefix}mlp.gate_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.up_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.down_proj.weight"] = (hidden, inter)

    vision = config.vision_hidden_size
    vision_inter = config.vision_intermediate_size
    merged = vision * config.vision_merge_size**2
    layout["model.visual.patch_embed.proj.weight"] = (vision, 3, *config.vision_patch)
    layout["model.visual.patch_embed.proj.bias"] = (vision,)
    layout["model.visual.pos_embed.weight"] = (config.vision_position_embeddings, vision)
    for index in range(config.vision_layers):
        prefix = f"model.visual.blocks.{index}."
        for norm in ("norm1", "norm2"):
            layout[f"{prefix}{norm}.weight"] = (vision,)
            layout[f"{prefix}{norm}.bias"] = (vision,)
        layout[f"{prefix}attn.qkv.weight"] = (vision * 3, vision)
        layout[f"{prefix}attn.qkv.bias"] = (vision * 3,)
        layout[f"{prefix}attn.proj.weight"] = (vision, vision)
        layout[f"{prefix}attn.proj.bias"] = (vision,)
        layout[f"{prefix}mlp.linear_fc1.weight"] = (vision_inter, vision)
        layout[f"{prefix}mlp.linear_fc1.bias"] = (vision_inter,)
        layout[f"{prefix}mlp.linear_fc2.weight"] = (vision, vision_inter)
        layout[f"{prefix}mlp.linear_fc2.bias"] = (vision,)
    layout["model.visual.merger.norm.weight"] = (vision,)
    layout["model.visual.merger.norm.bias"] = (vision,)
    layout["model.visual.merger.linear_fc1.weight"] = (merged, merged)
    layout["model.visual.merger.linear_fc1.bias"] = (merged,)
    layout["model.visual.merger.linear_fc2.weight"] = (hidden, merged)
    layout["model.visual.merger.linear_fc2.bias"] = (hidden,)
    for index in range(len(config.deepstack_layers)):
        prefix = f"model.visual.deepstack_merger_list.{index}."
        layout[f"{prefix}norm.weight"] = (merged,)
        layout[f"{prefix}norm.bias"] = (merged,)
        layout[f"{prefix}linear_fc1.weight"] = (merged, merged)
        layout[f"{prefix}linear_fc1.bias"] = (merged,)
        layout[f"{prefix}linear_fc2.weight"] = (hidden, merged)
        layout[f"{prefix}linear_fc2.bias"] = (hidden,)
    return MappingProxyType(layout)


def ideogram4_language_layout() -> Mapping[str, tuple[int, ...]]:
    return MappingProxyType(
        {
            key.removeprefix(IDEOGRAM4_LANGUAGE_SUBTREE): shape
            for key, shape in ideogram4_text_layout().items()
            if key.startswith(IDEOGRAM4_LANGUAGE_SUBTREE)
            and not key.startswith(IDEOGRAM4_VISION_SUBTREE)
        }
    )


def detect_ideogram4_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> Ideogram4TextConfig:
    anchors = {
        "model.embed_tokens.weight",
        "model.visual.deepstack_merger_list.0.norm.weight",
        "lm_head.weight",
    }
    if not anchors.issubset(geometries):
        raise Ideogram4TextDetectError("not the full Ideogram 4 Qwen3-VL-8B text role")
    layout = ideogram4_text_layout()
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
        raise Ideogram4TextDetectError(f"header does not match Qwen3-VL-8B: {shown}")
    return IDEOGRAM4_TEXT_CONFIG


@dataclass(frozen=True)
class Ideogram4Prompt:
    text: str
    preformatted: bool


def format_ideogram4_prompt(text: str) -> Ideogram4Prompt:
    if text.startswith("<|im_start|>"):
        return Ideogram4Prompt(text, True)
    return Ideogram4Prompt(IDEOGRAM4_TEXT_CONFIG.prompt_template.format(text), False)


__all__ = [
    "IDEOGRAM4_LANGUAGE_SUBTREE",
    "IDEOGRAM4_TAP_LAYERS",
    "IDEOGRAM4_TEXT_CONFIG",
    "IDEOGRAM4_VISION_SUBTREE",
    "Ideogram4Prompt",
    "Ideogram4TextConfig",
    "Ideogram4TextDetectError",
    "detect_ideogram4_text_config",
    "format_ideogram4_prompt",
    "ideogram4_language_layout",
    "ideogram4_text_layout",
]
