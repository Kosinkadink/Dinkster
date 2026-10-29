"""Torch-free Krea 2 Qwen3-VL-4B text contract.

The exact profile, checkpoint layout, template, tap policy, and output
selection follow ComfyUI b78cec879b9460d5cb25228a83a942fb78d2cd24
(``comfy/text_encoders/krea2.py``, ``comfy/text_encoders/qwen3vl.py``,
and ``Qwen3VL_4BConfig`` in ``comfy/text_encoders/llama.py``). This
module describes an inert contract and does not provide a tokenizer,
model, or runtime registration.

Krea 2 conditions on twelve tapped hidden states of the Qwen3-VL-4B
language tower. Tap ``k`` is the residual stream entering decoder layer
``k`` (the reference's ``hidden_states[k]``), so no tap passes through
the final RMS norm. The reference carries the stack as
``(batch, 12, seq, 2560)`` and flattens it to ``(batch, seq, 30720)``
for conditioning; the DiT's TextFusion module unpacks it.

The standalone checkpoint keeps the Hugging Face Qwen3-VL naming:
``model.language_model.*`` for the text tower and ``model.visual.*``
for the vision tower (ComfyUI remaps those prefixes at load). Text-only
conditioning never executes the vision tower, but the layout pins every
key so detection refuses partial or near-miss files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .weights import TensorGeometry


class Krea2TextDetectError(ValueError):
    """A header is not the exact Krea 2 Qwen3-VL-4B text-encoder role."""


# hidden_states indices tapped by the reference; tap k == the input to
# decoder layer k (comfy/text_encoders/krea2.py KREA2_TAP_LAYERS).
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Identical system prompt to Qwen Image; Krea 2 strips the system and
# user-opening prefix from the encoded sequence.
_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class Krea2TextConfig:
    """The exact Krea 2 Qwen3-VL-4B profile."""

    architecture: str = "krea2_qwen3vl_4b"
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
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
    tap_layers: tuple[int, ...] = KREA2_TAP_LAYERS
    prompt_template: str = _PROMPT_TEMPLATE
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    user_token_id: int = 872
    newline_token_id: int = 198
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_layers: int = 24
    vision_heads: int = 16
    vision_patch: tuple[int, int, int] = (2, 16, 16)
    vision_merge_size: int = 2
    vision_position_embeddings: int = 2304
    deepstack_layers: tuple[int, int, int] = (5, 11, 17)

    def __post_init__(self) -> None:
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        expected = (
            "krea2_qwen3vl_4b",
            151936,
            2560,
            9728,
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
            KREA2_TAP_LAYERS,
            _PROMPT_TEMPLATE,
            151643,
            151644,
            872,
            198,
            1024,
            4096,
            24,
            16,
            (2, 16, 16),
            2,
            2304,
            (5, 11, 17),
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("Krea2TextConfig only represents the exact Krea 2 profile")

    @property
    def stack_width(self) -> int:
        """Flattened conditioning width: taps times hidden size."""
        return len(self.tap_layers) * self.hidden_size


KREA2_TEXT_CONFIG = Krea2TextConfig()

KREA2_LANGUAGE_SUBTREE = "model.language_model."
KREA2_VISION_SUBTREE = "model.visual."


def krea2_text_layout() -> Mapping[str, tuple[int, ...]]:
    """The immutable 713-entry checkpoint layout (HF Qwen3-VL naming)."""

    config = KREA2_TEXT_CONFIG
    hidden = config.hidden_size
    head = config.head_dim
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        f"{KREA2_LANGUAGE_SUBTREE}embed_tokens.weight": (config.vocab_size, hidden),
        f"{KREA2_LANGUAGE_SUBTREE}norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"{KREA2_LANGUAGE_SUBTREE}layers.{index}."
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
    layout[f"{KREA2_VISION_SUBTREE}patch_embed.proj.weight"] = (vision, 3, *config.vision_patch)
    layout[f"{KREA2_VISION_SUBTREE}patch_embed.proj.bias"] = (vision,)
    layout[f"{KREA2_VISION_SUBTREE}pos_embed.weight"] = (config.vision_position_embeddings, vision)
    for index in range(config.vision_layers):
        prefix = f"{KREA2_VISION_SUBTREE}blocks.{index}."
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
    layout[f"{KREA2_VISION_SUBTREE}merger.norm.weight"] = (vision,)
    layout[f"{KREA2_VISION_SUBTREE}merger.norm.bias"] = (vision,)
    layout[f"{KREA2_VISION_SUBTREE}merger.linear_fc1.weight"] = (merged, merged)
    layout[f"{KREA2_VISION_SUBTREE}merger.linear_fc1.bias"] = (merged,)
    layout[f"{KREA2_VISION_SUBTREE}merger.linear_fc2.weight"] = (hidden, merged)
    layout[f"{KREA2_VISION_SUBTREE}merger.linear_fc2.bias"] = (hidden,)
    for index in range(len(config.deepstack_layers)):
        prefix = f"{KREA2_VISION_SUBTREE}deepstack_merger_list.{index}."
        layout[f"{prefix}norm.weight"] = (merged,)
        layout[f"{prefix}norm.bias"] = (merged,)
        layout[f"{prefix}linear_fc1.weight"] = (merged, merged)
        layout[f"{prefix}linear_fc1.bias"] = (merged,)
        layout[f"{prefix}linear_fc2.weight"] = (hidden, merged)
        layout[f"{prefix}linear_fc2.bias"] = (hidden,)
    return MappingProxyType(layout)


def krea2_language_layout() -> Mapping[str, tuple[int, ...]]:
    """The text-tower subset of the layout, stripped of its prefix."""

    return MappingProxyType(
        {
            key.removeprefix(KREA2_LANGUAGE_SUBTREE): shape
            for key, shape in krea2_text_layout().items()
            if key.startswith(KREA2_LANGUAGE_SUBTREE)
        }
    )


def detect_krea2_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> Krea2TextConfig:
    """Accept only the complete exact Krea 2 text-encoder header."""

    role_anchors = {
        f"{KREA2_LANGUAGE_SUBTREE}embed_tokens.weight",
        # DeepStack mergers are unique to Qwen3-VL and separate this
        # role from text-only Qwen3 towers of identical geometry.
        f"{KREA2_VISION_SUBTREE}deepstack_merger_list.0.norm.weight",
    }
    if not role_anchors.issubset(geometries):
        raise Krea2TextDetectError(
            "not a Krea 2 Qwen3-VL-4B text role (missing language or DeepStack anchor)"
        )
    layout = krea2_text_layout()
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
        raise Krea2TextDetectError("header does not match the Krea 2 Qwen3-VL-4B layout: " + shown)
    return KREA2_TEXT_CONFIG


@dataclass(frozen=True)
class Krea2Prompt:
    """A formatted Krea 2 prompt."""

    text: str
    preformatted: bool

    def __post_init__(self) -> None:
        if type(self.preformatted) is not bool:
            raise TypeError("preformatted must be a bool")


def format_krea2_prompt(text: str) -> Krea2Prompt:
    """Apply the fixed template unless the prompt is already formatted.

    The reference skips templating when the prompt starts with
    ``<|im_start|>`` and never appends a think block (Krea 2 tokenizes
    with ``thinking=True``, which suppresses the empty ``<think>``
    suffix the shared Qwen3-VL tokenizer would otherwise add).
    """

    if text.startswith("<|im_start|>"):
        return Krea2Prompt(text, True)
    return Krea2Prompt(KREA2_TEXT_CONFIG.prompt_template.format(text), False)


def _snapshot_token_rows(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    snapshot = tuple(tuple(row) for row in rows)
    if len(snapshot) != 1 or not snapshot[0]:
        raise ValueError("Krea 2 requires exactly one non-empty token row")
    for row in snapshot:
        for token_id in row:
            if type(token_id) is not int:
                raise ValueError("token rows must contain integer token IDs")
            if not 0 <= token_id < KREA2_TEXT_CONFIG.vocab_size:
                raise ValueError("token IDs must be inside the vocabulary")
    return snapshot


@dataclass(frozen=True)
class Krea2OutputSelection:
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


def select_krea2_output(
    token_rows: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]] | None = None,
    *,
    template_end: int = -1,
) -> Krea2OutputSelection:
    """Select post-template states and retain a mask only when it masks.

    Mirrors the reference strip: the boundary is the second
    ``<|im_start|>`` token, advanced past ``user\\n`` when those two
    tokens follow it. The mask is sliced identically and dropped when
    fully attended.
    """

    rows = _snapshot_token_rows(token_rows)
    row = rows[0]
    if type(template_end) is not int or template_end < -1 or template_end >= len(row):
        raise ValueError("template_end must be -1 or an index inside the token row")
    slice_start = template_end
    if slice_start == -1:
        markers = tuple(
            index
            for index, token_id in enumerate(row)
            if token_id == KREA2_TEXT_CONFIG.im_start_token_id
        )
        if len(markers) < 2:
            raise ValueError("Krea 2 template must contain at least two im-start tokens")
        slice_start = markers[1]
        suffix = row[slice_start + 1 : slice_start + 3]
        expected = (
            KREA2_TEXT_CONFIG.user_token_id,
            KREA2_TEXT_CONFIG.newline_token_id,
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
    return Krea2OutputSelection(selected_rows, selected_mask, slice_start)


__all__ = [
    "KREA2_LANGUAGE_SUBTREE",
    "KREA2_TAP_LAYERS",
    "KREA2_TEXT_CONFIG",
    "KREA2_VISION_SUBTREE",
    "Krea2OutputSelection",
    "Krea2Prompt",
    "Krea2TextConfig",
    "Krea2TextDetectError",
    "detect_krea2_text_config",
    "format_krea2_prompt",
    "krea2_language_layout",
    "krea2_text_layout",
    "select_krea2_output",
]
