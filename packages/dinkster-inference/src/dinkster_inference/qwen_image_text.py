"""Torch-free Qwen2.5-VL-7B text contract for base Qwen Image.

The exact profile, state layout, templates, placeholder ordering, and output
selection follow ComfyUI 2a68ce33b4c9ea6ee4283e618a74560cefb32694. This
module describes an inert contract and does not provide a tokenizer, model, or
runtime registration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType

from .weights import TensorGeometry


class QwenImageTextDetectError(ValueError):
    """A header is not the exact Qwen Image Qwen2.5-VL-7B text role."""


_TEXT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_IMAGE_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}"
    "<|im_end|>\n<|im_start|>assistant\n"
)
_EDIT_PLUS_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
    "size, texture, objects, background), then explain how the user's text instruction "
    "should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class QwenImageTextConfig:
    """The exact base Qwen Image Qwen2.5-VL-7B profile."""

    architecture: str = "qwen2.5_vl_7b_qwen_image"
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True
    rope_dims: tuple[int, int, int] = (16, 24, 24)
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    image_token_id: int = 151655
    user_token_id: int = 872
    newline_token_id: int = 198
    text_template: str = _TEXT_TEMPLATE
    image_template: str = _IMAGE_TEMPLATE
    vision_hidden_size: int = 1280
    vision_output_size: int = 3584
    vision_intermediate_size: int = 3420
    vision_heads: int = 16
    vision_layers: int = 32
    vision_patch: tuple[int, int, int] = (2, 14, 14)
    vision_spatial_merge: int = 2

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "qwen2.5_vl_7b_qwen_image",
            152064,
            3584,
            18944,
            28,
            28,
            4,
            128000,
            1e-6,
            1_000_000.0,
            True,
            (16, 24, 24),
            151643,
            151644,
            151655,
            872,
            198,
            _TEXT_TEMPLATE,
            _IMAGE_TEMPLATE,
            1280,
            3584,
            3420,
            16,
            32,
            (2, 14, 14),
            2,
        )
        if (
            any(
                type(value) is not type(required) or value != required
                for value, required in zip(actual, expected, strict=True)
            )
            or any(type(value) is not int for value in self.rope_dims)
            or any(type(value) is not int for value in self.vision_patch)
        ):
            raise ValueError("QwenImageTextConfig only represents the exact staged profile")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


QWEN_IMAGE_TEXT_CONFIG = QwenImageTextConfig()


def qwen_image_text_layout() -> Mapping[str, tuple[int, ...]]:
    """Return the immutable 728-entry encoder state layout."""

    config = QWEN_IMAGE_TEXT_CONFIG
    hidden = config.hidden_size
    kv = config.num_key_value_heads * config.head_dim
    intermediate = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        layout[f"{prefix}input_layernorm.weight"] = (hidden,)
        layout[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        layout[f"{prefix}self_attn.q_proj.weight"] = (hidden, hidden)
        layout[f"{prefix}self_attn.q_proj.bias"] = (hidden,)
        layout[f"{prefix}self_attn.k_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.k_proj.bias"] = (kv,)
        layout[f"{prefix}self_attn.v_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.v_proj.bias"] = (kv,)
        layout[f"{prefix}self_attn.o_proj.weight"] = (hidden, hidden)
        layout[f"{prefix}mlp.gate_proj.weight"] = (intermediate, hidden)
        layout[f"{prefix}mlp.up_proj.weight"] = (intermediate, hidden)
        layout[f"{prefix}mlp.down_proj.weight"] = (hidden, intermediate)

    vision = config.vision_hidden_size
    vision_intermediate = config.vision_intermediate_size
    layout["visual.patch_embed.proj.weight"] = (
        vision,
        3,
        *config.vision_patch,
    )
    for index in range(config.vision_layers):
        prefix = f"visual.blocks.{index}."
        layout[f"{prefix}norm1.weight"] = (vision,)
        layout[f"{prefix}norm2.weight"] = (vision,)
        layout[f"{prefix}attn.qkv.weight"] = (vision * 3, vision)
        layout[f"{prefix}attn.qkv.bias"] = (vision * 3,)
        layout[f"{prefix}attn.proj.weight"] = (vision, vision)
        layout[f"{prefix}attn.proj.bias"] = (vision,)
        layout[f"{prefix}mlp.gate_proj.weight"] = (vision_intermediate, vision)
        layout[f"{prefix}mlp.gate_proj.bias"] = (vision_intermediate,)
        layout[f"{prefix}mlp.up_proj.weight"] = (vision_intermediate, vision)
        layout[f"{prefix}mlp.up_proj.bias"] = (vision_intermediate,)
        layout[f"{prefix}mlp.down_proj.weight"] = (vision, vision_intermediate)
        layout[f"{prefix}mlp.down_proj.bias"] = (vision,)

    merged = vision * config.vision_spatial_merge**2
    layout["visual.merger.ln_q.weight"] = (vision,)
    layout["visual.merger.mlp.0.weight"] = (merged, merged)
    layout["visual.merger.mlp.0.bias"] = (merged,)
    layout["visual.merger.mlp.2.weight"] = (config.vision_output_size, merged)
    layout["visual.merger.mlp.2.bias"] = (config.vision_output_size,)
    return MappingProxyType(layout)


def detect_qwen_image_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> QwenImageTextConfig:
    """Accept only the complete exact Qwen Image text encoder header."""

    role_anchors = {
        "model.embed_tokens.weight",
        "visual.patch_embed.proj.weight",
    }
    if not role_anchors.issubset(geometries):
        raise QwenImageTextDetectError(
            "not a Qwen Image text role (missing language or vision anchor)"
        )
    layout = qwen_image_text_layout()
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
        raise QwenImageTextDetectError(
            "header does not match the Qwen Image Qwen2.5-VL-7B layout: " + shown
        )
    return QWEN_IMAGE_TEXT_CONFIG


@dataclass(frozen=True)
class QwenImagePrompt:
    """A formatted prompt and its expected image cardinality."""

    text: str
    image_count: int
    preformatted: bool

    def __post_init__(self) -> None:
        if type(self.image_count) is not int or not 0 <= self.image_count <= 3:
            raise ValueError("Qwen Image prompts accept zero to three images")
        if type(self.preformatted) is not bool:
            raise TypeError("preformatted must be a bool")
        if self.text.count("<|image_pad|>") != self.image_count:
            raise ValueError("image cardinality must match image placeholders")


def format_qwen_image_prompt(
    text: str,
    *,
    image_count: int = 0,
    template: str | None = None,
    edit_plus: bool = False,
    image_slots: Sequence[int] | None = None,
) -> QwenImagePrompt:
    """Apply the exact base or edit template unless already formatted."""

    if type(image_count) is not int or not 0 <= image_count <= 3:
        raise ValueError("Qwen Image prompts accept zero to three images")
    if type(edit_plus) is not bool:
        raise TypeError("edit_plus must be a bool")
    slots = tuple(range(1, image_count + 1)) if image_slots is None else tuple(image_slots)
    if (
        len(slots) != image_count
        or any(type(slot) is not int or not 1 <= slot <= 3 for slot in slots)
        or tuple(sorted(set(slots))) != slots
    ):
        raise ValueError("Qwen Image slots must be unique ascending integers in [1, 3]")
    preformatted = text.startswith(("<|im_start|>", "<|start_header_id|>"))
    if preformatted:
        return QwenImagePrompt(text, image_count, True)
    selected = template
    if selected is None:
        if edit_plus or image_count > 1:
            pictures = "".join(
                f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>" for index in slots
            )
            selected = _EDIT_PLUS_TEMPLATE
            text = pictures + text
        else:
            selected = (
                QWEN_IMAGE_TEXT_CONFIG.image_template
                if image_count
                else QWEN_IMAGE_TEXT_CONFIG.text_template
            )
    try:
        fields = tuple(
            (field_name, format_spec, conversion)
            for _, field_name, format_spec, conversion in Formatter().parse(selected)
            if field_name is not None
        )
    except ValueError as error:
        raise ValueError("Qwen Image template must carry exactly one text slot") from error
    if fields != (("", "", None),):
        raise ValueError("Qwen Image template must carry exactly one text slot")
    return QwenImagePrompt(selected.format(text), image_count, False)


def _snapshot_token_rows(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    snapshot = tuple(tuple(row) for row in rows)
    if len(snapshot) != 1 or not snapshot[0]:
        raise ValueError("Qwen Image requires exactly one non-empty token row")
    for row in snapshot:
        for token_id in row:
            if type(token_id) is not int:
                raise ValueError("token rows must contain integer token IDs")
            if not 0 <= token_id < QWEN_IMAGE_TEXT_CONFIG.vocab_size:
                raise ValueError("token IDs must be inside the vocabulary")
    return snapshot


@dataclass(frozen=True)
class QwenImageTokenPlan:
    """Immutable token rows and image-slot substitution order."""

    rows: tuple[tuple[int, ...], ...]
    image_slots: tuple[tuple[int, int], ...]
    image_count: int

    def __post_init__(self) -> None:
        rows = _snapshot_token_rows(self.rows)
        slots = tuple(tuple(slot) for slot in self.image_slots)
        if any(
            len(slot) != 2
            or any(type(value) is not int for value in slot)
            or slot[0] != 0
            or not 0 <= slot[1] < len(rows[0])
            for slot in slots
        ):
            raise ValueError("image slots must identify valid positions in the token row")
        if type(self.image_count) is not int or not 0 <= self.image_count <= 3:
            raise ValueError("Qwen Image token plans accept zero to three images")
        expected_slots = tuple(
            (row_index, token_index)
            for row_index, row in enumerate(rows)
            for token_index, token_id in enumerate(row)
            if token_id == QWEN_IMAGE_TEXT_CONFIG.image_token_id
        )
        if slots != expected_slots or len(slots) != self.image_count:
            raise ValueError("image slots must exactly match placeholders and image cardinality")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "image_slots", slots)


def plan_qwen_image_token_rows(
    rows: Sequence[Sequence[int]], *, image_count: int
) -> QwenImageTokenPlan:
    """Record left-to-right image-placeholder substitution without payloads."""

    snapshot = _snapshot_token_rows(rows)
    if type(image_count) is not int or not 0 <= image_count <= 3:
        raise ValueError("Qwen Image token plans accept zero to three images")
    image_token = QWEN_IMAGE_TEXT_CONFIG.image_token_id
    slots = tuple(
        (row_index, token_index)
        for row_index, row in enumerate(snapshot)
        for token_index, token_id in enumerate(row)
        if token_id == image_token
    )
    if len(slots) != image_count:
        raise ValueError(
            f"image cardinality {image_count} does not match {len(slots)} token placeholders"
        )
    return QwenImageTokenPlan(snapshot, slots, image_count)


@dataclass(frozen=True)
class QwenImageOutputSelection:
    """The selected encoder rows and optional retained attention mask."""

    token_rows: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...] | None
    slice_start: int

    def __post_init__(self) -> None:
        rows = _snapshot_token_rows(self.token_rows)
        if type(self.slice_start) is not int or self.slice_start < 0:
            raise ValueError("slice_start must be a non-negative integer")
        object.__setattr__(self, "token_rows", rows)
        if self.attention_mask is not None:
            mask = tuple(tuple(row) for row in self.attention_mask)
            if len(mask) != 1 or len(mask[0]) != len(rows[0]):
                raise ValueError("attention mask shape must match token rows")
            if any(type(value) is not int or value not in (0, 1) for value in mask[0]):
                raise ValueError("attention mask values must be binary integers")
            object.__setattr__(self, "attention_mask", None if all(mask[0]) else mask)


def select_qwen_image_output(
    token_rows: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]] | None,
    *,
    template_end: int = -1,
) -> QwenImageOutputSelection:
    """Select post-template states and retain a mask only when it masks tokens."""

    rows = _snapshot_token_rows(token_rows)
    row = rows[0]
    if type(template_end) is not int or template_end < -1 or template_end >= len(row):
        raise ValueError("template_end must be -1 or an index inside the token row")
    slice_start = template_end
    if slice_start == -1:
        markers = tuple(
            index
            for index, token_id in enumerate(row)
            if token_id == QWEN_IMAGE_TEXT_CONFIG.im_start_token_id
        )
        if len(markers) < 2:
            raise ValueError("Qwen Image template must contain at least two im-start tokens")
        slice_start = markers[1]
        suffix = row[slice_start + 1 : slice_start + 3]
        expected = (
            QWEN_IMAGE_TEXT_CONFIG.user_token_id,
            QWEN_IMAGE_TEXT_CONFIG.newline_token_id,
        )
        if suffix == expected:
            slice_start += 3
        elif any(token_id in expected for token_id in suffix):
            raise ValueError("ambiguous user boundary after the second im-start token")
    if slice_start >= len(row):
        raise ValueError("template boundary selects an empty token row")

    selected_rows = (row[slice_start:],)
    selected_mask: tuple[tuple[int, ...], ...] | None = None
    if attention_mask is not None:
        mask = tuple(tuple(mask_row) for mask_row in attention_mask)
        if len(mask) != 1 or len(mask[0]) != len(row):
            raise ValueError("attention mask shape must match token rows")
        if any(type(value) is not int or value not in (0, 1) for value in mask[0]):
            raise ValueError("attention mask values must be binary integers")
        sliced_mask = (mask[0][slice_start:],)
        if not all(sliced_mask[0]):
            selected_mask = sliced_mask
    return QwenImageOutputSelection(selected_rows, selected_mask, slice_start)


__all__ = [
    "QWEN_IMAGE_TEXT_CONFIG",
    "QwenImageOutputSelection",
    "QwenImagePrompt",
    "QwenImageTextConfig",
    "QwenImageTextDetectError",
    "QwenImageTokenPlan",
    "detect_qwen_image_text_config",
    "format_qwen_image_prompt",
    "plan_qwen_image_token_rows",
    "qwen_image_text_layout",
    "select_qwen_image_output",
]
