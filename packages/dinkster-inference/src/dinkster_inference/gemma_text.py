"""Gemma 3 12B text tower and LTX-2 prompt policy - torch-free.

Mirrors ComfyUI's ``Gemma3_12B_Config`` (comfy/text_encoders/llama.py)
and the LTXAV tokenizer and projection policy (comfy/text_encoders/lt.py)
@ b78cec87. The checkpoint carries SigLIP ``vision_model.*`` and
``multi_modal_projector.*`` siblings outside ``model.*``; the caller
extracts only ``model.*`` and decides what to do with the rest. The
image-token path is out of scope: the prompt policy takes no images
and a literal ``<image_soft_token>`` in text is the plain token id
262144, exactly what the reference produces with no image attached
(segment-leading occurrences fall to the slice described below).

The prompt policy replays the reference SDTokenizer with
``disable_weights``: the emphasis grammar is inert (only ``\\(`` and
``\\)`` unescape), segments split on whitespace-preceded
``embedding:`` markers, and every segment loses its first token to the
reference's start-token slice - a plain segment's dropped token is the
SentencePiece BOS it just added, but a segment routed through the
special-token splitter never gained one, so its first CONTENT token is
dropped (templated text loses the leading ``<start_of_turn>``). Rows
gain start token 2, no end token, and pad LEFT with 0 to 1024. The
chat template is applied only on request: at the pin no LTXAV
conditioning node passes ``skip_template=False``, so inference
encodes the raw prompt; text already starting with
``<start_of_turn>`` always skips the template.

The attention-mask rule duplicates ``anima_text._reference_mask``
(kept private there on purpose): an opening pad run is masked without
ending attention, and the first pad after it masks the rest.

The projection detector accepts the ``text_embedding_projection.*``
subtree of a combined checkpoint: the released single-projection
files spell the weight ``aggregate_embed.weight`` or plain
``weight`` (one 3840 x 188160 Linear, no bias), and the dual layout
adds separate video (4096) and audio (2048) Linears with biases.

Combined checkpoints (the released ltx-2-19b-dev) additionally carry
two ``model.diffusion_model.*_embeddings_connector`` transformer
towers that refine the single projection's output into concatenated
video and audio text embeddings. :func:`detect_ltx_text_connectors`
ports the reference's exact activation condition for that stage (a
two-block audio tower whose ``blocks.0.attn1.to_q.bias`` is 3840 wide,
with no third block) and then requires the full 58-key layout under
both prefixes; connector keys that do not satisfy the reference
condition are refused rather than silently ignored.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .weights import TensorGeometry


class GemmaTextDetectError(ValueError):
    """A header is not an accepted Gemma text tower or LTX projection."""


@dataclass(frozen=True)
class GemmaTextConfig:
    """Construction and prompt-policy facts for one Gemma text tower."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta_global: float
    rope_theta_local: float
    rope_scale_global: float
    rope_scale_local: float
    sliding_window: int
    sliding_pattern: tuple[bool, ...]
    prompt_template: str
    min_tokens: int
    pad_token_id: int
    bos_token_id: int
    end_of_turn_token_id: int
    image_soft_token_id: int | None
    global_head_dim: int | None = None
    num_global_key_value_heads: int | None = None
    global_k_eq_v: bool = False
    rms_norm_add: bool = True
    value_rms_norm: bool = False
    layer_scalar: bool = False
    global_partial_rotary_factor: float = 1.0
    attention_scale: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "sliding_window",
            "min_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.global_head_dim is not None and self.global_head_dim < 1:
            raise ValueError("global_head_dim must be positive")
        if self.num_global_key_value_heads is not None:
            if self.num_global_key_value_heads < 1:
                raise ValueError("num_global_key_value_heads must be positive")
            if self.num_attention_heads % self.num_global_key_value_heads:
                raise ValueError(
                    "num_attention_heads must be divisible by num_global_key_value_heads"
                )
        if self.rms_norm_eps <= 0:
            raise ValueError("RMS epsilon must be positive")
        for name in ("rope_theta_global", "rope_theta_local"):
            if getattr(self, name) <= 0:
                raise ValueError("RoPE thetas must be positive")
        for name in ("rope_scale_global", "rope_scale_local"):
            if getattr(self, name) <= 0:
                raise ValueError("RoPE scales must be positive")
        if not 0.0 < self.global_partial_rotary_factor <= 1.0:
            raise ValueError("global_partial_rotary_factor must be in (0, 1]")
        if self.attention_scale is not None and self.attention_scale <= 0.0:
            raise ValueError("attention_scale must be positive")
        if not self.sliding_pattern:
            raise ValueError("sliding_pattern must name at least one layer kind")
        for name in ("pad_token_id", "bos_token_id", "end_of_turn_token_id"):
            value = getattr(self, name)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} {value} is outside the vocabulary")
        if self.image_soft_token_id is not None and not (
            0 <= self.image_soft_token_id < self.vocab_size
        ):
            raise ValueError(
                f"image_soft_token_id {self.image_soft_token_id} is outside the vocabulary"
            )
        if self.prompt_template.count("{}") != 1:
            raise ValueError("prompt_template must carry exactly one text slot")


LTX_GEMMA_PROMPT_TEMPLATE = (
    "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n"
    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
)

GEMMA3_LTX_12B_CONFIG = GemmaTextConfig(
    architecture="gemma3_ltx_12b",
    vocab_size=262208,
    hidden_size=3840,
    intermediate_size=15360,
    num_hidden_layers=48,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=256,
    rms_norm_eps=1e-6,
    rope_theta_global=1_000_000.0,
    rope_theta_local=10_000.0,
    rope_scale_global=8.0,
    rope_scale_local=1.0,
    sliding_window=1024,
    sliding_pattern=(True, True, True, True, True, False),
    prompt_template=LTX_GEMMA_PROMPT_TEMPLATE,
    min_tokens=1024,
    pad_token_id=0,
    bos_token_id=2,
    end_of_turn_token_id=106,
    image_soft_token_id=262144,
)

GEMMA3_NEWBIE_4B_CONFIG = GemmaTextConfig(
    architecture="gemma3_newbie_4b",
    vocab_size=262208,
    hidden_size=2560,
    intermediate_size=10240,
    num_hidden_layers=34,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    rms_norm_eps=1e-6,
    rope_theta_global=1_000_000.0,
    rope_theta_local=10_000.0,
    rope_scale_global=8.0,
    rope_scale_local=1.0,
    sliding_window=1024,
    sliding_pattern=(True, True, True, True, True, False),
    prompt_template="{}",
    min_tokens=1,
    pad_token_id=0,
    bos_token_id=2,
    end_of_turn_token_id=106,
    image_soft_token_id=262144,
)

GEMMA4_LTX_12B_CONFIG = GemmaTextConfig(
    architecture="gemma4_ltx_12b",
    vocab_size=262144,
    hidden_size=3840,
    intermediate_size=15360,
    num_hidden_layers=48,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=256,
    rms_norm_eps=1e-6,
    rope_theta_global=1_000_000.0,
    rope_theta_local=10_000.0,
    rope_scale_global=1.0,
    rope_scale_local=1.0,
    sliding_window=1024,
    sliding_pattern=(True, True, True, True, True, False),
    prompt_template=LTX_GEMMA_PROMPT_TEMPLATE,
    min_tokens=1024,
    pad_token_id=0,
    bos_token_id=2,
    end_of_turn_token_id=1,
    image_soft_token_id=0,
    global_head_dim=512,
    num_global_key_value_heads=1,
    global_k_eq_v=True,
    rms_norm_add=False,
    value_rms_norm=True,
    layer_scalar=True,
    global_partial_rotary_factor=0.25,
    attention_scale=1.0,
)

GEMMA2_LUMINA_2B_CONFIG = GemmaTextConfig(
    architecture="gemma2_lumina_2b",
    vocab_size=256000,
    hidden_size=2304,
    intermediate_size=9216,
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    rms_norm_eps=1e-6,
    rope_theta_global=10_000.0,
    rope_theta_local=10_000.0,
    rope_scale_global=1.0,
    rope_scale_local=1.0,
    sliding_window=8192,
    sliding_pattern=(False,),
    prompt_template="{}",
    min_tokens=1,
    pad_token_id=0,
    bos_token_id=2,
    end_of_turn_token_id=107,
    image_soft_token_id=None,
)

_KNOWN_GEMMA_TEXT_CONFIGS = (
    GEMMA3_LTX_12B_CONFIG,
    GEMMA3_NEWBIE_4B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    GEMMA2_LUMINA_2B_CONFIG,
)


def gemma_text_layout(config: GemmaTextConfig) -> dict[str, tuple[int, ...]]:
    """Exact ``model.*``-stripped state dictionary for ``config``."""
    hidden = config.hidden_size
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (config.vocab_size, hidden),
        "norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        sliding = config.sliding_pattern[index % len(config.sliding_pattern)]
        head = config.head_dim if sliding else config.global_head_dim or config.head_dim
        kv_heads = (
            config.num_key_value_heads
            if sliding
            else config.num_global_key_value_heads or config.num_key_value_heads
        )
        query = config.num_attention_heads * head
        kv = kv_heads * head
        prefix = f"layers.{index}."
        layout[f"{prefix}input_layernorm.weight"] = (hidden,)
        layout[f"{prefix}post_attention_layernorm.weight"] = (hidden,)
        layout[f"{prefix}pre_feedforward_layernorm.weight"] = (hidden,)
        layout[f"{prefix}post_feedforward_layernorm.weight"] = (hidden,)
        layout[f"{prefix}self_attn.q_proj.weight"] = (query, hidden)
        layout[f"{prefix}self_attn.k_proj.weight"] = (kv, hidden)
        if not (config.global_k_eq_v and not sliding):
            layout[f"{prefix}self_attn.v_proj.weight"] = (kv, hidden)
        layout[f"{prefix}self_attn.o_proj.weight"] = (hidden, query)
        if config.architecture != "gemma2_lumina_2b":
            layout[f"{prefix}self_attn.q_norm.weight"] = (head,)
            layout[f"{prefix}self_attn.k_norm.weight"] = (head,)
        layout[f"{prefix}mlp.gate_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.up_proj.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.down_proj.weight"] = (hidden, inter)
        if config.layer_scalar:
            layout[f"{prefix}layer_scalar"] = (1,)
    return layout


def detect_gemma_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> GemmaTextConfig:
    """Accept one complete known Gemma text-model subtree."""
    if not geometries:
        raise GemmaTextDetectError("empty state dict header")
    embedding = geometries.get("embed_tokens.weight")
    if embedding is None:
        raise GemmaTextDetectError(
            "not a Gemma text-model role (missing model.embed_tokens.weight)"
        )
    if len(embedding.shape) != 2:
        raise GemmaTextDetectError(
            f"embed_tokens.weight has rank {len(embedding.shape)}, expected 2"
        )
    candidates = [
        candidate
        for candidate in _KNOWN_GEMMA_TEXT_CONFIGS
        if embedding.shape == (candidate.vocab_size, candidate.hidden_size)
    ]
    if not candidates:
        accepted = ", ".join(
            f"{candidate.architecture} ({candidate.vocab_size}, {candidate.hidden_size})"
            for candidate in _KNOWN_GEMMA_TEXT_CONFIGS
        )
        raise GemmaTextDetectError(
            "unknown Gemma text width: token embedding"
            f" {embedding.shape}; accepted profiles are {accepted}"
        )

    best: tuple[GemmaTextConfig, list[str]] | None = None
    for config in candidates:
        layout = gemma_text_layout(config)
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
    raise GemmaTextDetectError(
        f"geometry does not match the {config.architecture} text layout: " + shown
    )


#: Layer stack width the projections consume: the embedding output plus
#: all 48 layer outputs of the 12B tower.
LTX_TEXT_STACK_DEPTH = 49
LTX_TEXT_STACK_FEATURES = 3840 * LTX_TEXT_STACK_DEPTH

LTX_SINGLE_PROJECTION_LAYOUT: dict[str, tuple[int, ...]] = {
    "weight": (3840, LTX_TEXT_STACK_FEATURES),
}
LTX_DUAL_PROJECTION_LAYOUT: dict[str, tuple[int, ...]] = {
    "audio_aggregate_embed.bias": (2048,),
    "audio_aggregate_embed.weight": (2048, LTX_TEXT_STACK_FEATURES),
    "video_aggregate_embed.bias": (4096,),
    "video_aggregate_embed.weight": (4096, LTX_TEXT_STACK_FEATURES),
}

LtxTextProjectionKind = Literal["single_linear", "dual_linear", "dual_linear_gemma4"]


def ltx_text_projection_layout(kind: LtxTextProjectionKind) -> dict[str, tuple[int, ...]]:
    """Exact ``text_embedding_projection.*``-stripped layout for ``kind``."""
    if kind == "single_linear":
        return dict(LTX_SINGLE_PROJECTION_LAYOUT)
    return dict(LTX_DUAL_PROJECTION_LAYOUT)


def detect_ltx_text_projection(
    geometries: Mapping[str, TensorGeometry],
) -> LtxTextProjectionKind:
    """Accept one complete LTX text-projection subtree.

    Takes the ``text_embedding_projection.*``-stripped keys. The
    released single-projection checkpoints spell the weight
    ``aggregate_embed.weight``; it is the same tensor as ``weight``
    (the reference renames it on load) and both spellings are
    accepted, never together.
    """
    if not geometries:
        raise GemmaTextDetectError("empty text-projection header")
    canonical = dict(geometries)
    aggregate = canonical.pop("aggregate_embed.weight", None)
    if aggregate is not None:
        if "weight" in canonical:
            raise GemmaTextDetectError(
                "text projection carries both aggregate_embed.weight and weight"
            )
        canonical["weight"] = aggregate

    kind: LtxTextProjectionKind = (
        "dual_linear" if "audio_aggregate_embed.bias" in canonical else "single_linear"
    )
    layout = ltx_text_projection_layout(kind)
    problems: list[str] = []
    for key, shape in layout.items():
        found = canonical.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(canonical) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise GemmaTextDetectError(
            f"geometry does not match the LTX {kind} text-projection layout: " + shown
        )
    return kind


@dataclass(frozen=True)
class LtxConnectorConfig:
    """Geometry of one LTX text-embedding connector tower."""

    num_attention_heads: int
    attention_head_dim: int
    num_layers: int
    num_learnable_registers: int
    positional_embedding_theta: float
    positional_embedding_max_pos: int
    gated_attention: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "num_learnable_registers",
            "positional_embedding_max_pos",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even (split RoPE halves each head)")
        if self.positional_embedding_theta <= 0:
            raise ValueError("positional_embedding_theta must be positive")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def ffn_dim(self) -> int:
        """The reference FeedForward hard-codes mult=4."""
        return 4 * self.inner_dim


LTX_TEXT_CONNECTOR_CONFIG = LtxConnectorConfig(
    num_attention_heads=30,
    attention_head_dim=128,
    num_layers=2,
    num_learnable_registers=128,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=4096,
)

LTXAV_22B_V23_VIDEO_CONNECTOR_CONFIG = LtxConnectorConfig(
    num_attention_heads=32,
    attention_head_dim=128,
    num_layers=8,
    num_learnable_registers=128,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=4096,
    gated_attention=True,
)

LTXAV_22B_V23_AUDIO_CONNECTOR_CONFIG = LtxConnectorConfig(
    num_attention_heads=32,
    attention_head_dim=64,
    num_layers=8,
    num_learnable_registers=128,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=4096,
    gated_attention=True,
)

LTX_VIDEO_CONNECTOR_PREFIX = "model.diffusion_model.video_embeddings_connector."
LTX_AUDIO_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."

_CONNECTOR_ABSENT_PROBE = f"{LTX_AUDIO_CONNECTOR_PREFIX}transformer_1d_blocks.2.attn1.to_q.bias"
_CONNECTOR_TRIGGER = f"{LTX_AUDIO_CONNECTOR_PREFIX}transformer_1d_blocks.0.attn1.to_q.bias"


def ltx_connector_layout(config: LtxConnectorConfig) -> dict[str, tuple[int, ...]]:
    """Exact prefix-stripped state dictionary of one connector tower."""
    inner = config.inner_dim
    ffn = config.ffn_dim
    layout: dict[str, tuple[int, ...]] = {
        "learnable_registers": (config.num_learnable_registers, inner),
    }
    for index in range(config.num_layers):
        prefix = f"transformer_1d_blocks.{index}."
        layout[f"{prefix}attn1.q_norm.weight"] = (inner,)
        layout[f"{prefix}attn1.k_norm.weight"] = (inner,)
        for projection in ("to_q", "to_k", "to_v"):
            layout[f"{prefix}attn1.{projection}.weight"] = (inner, inner)
            layout[f"{prefix}attn1.{projection}.bias"] = (inner,)
        if config.gated_attention:
            layout[f"{prefix}attn1.to_gate_logits.weight"] = (
                config.num_attention_heads,
                inner,
            )
            layout[f"{prefix}attn1.to_gate_logits.bias"] = (config.num_attention_heads,)
        layout[f"{prefix}attn1.to_out.0.weight"] = (inner, inner)
        layout[f"{prefix}attn1.to_out.0.bias"] = (inner,)
        layout[f"{prefix}ff.net.0.proj.weight"] = (ffn, inner)
        layout[f"{prefix}ff.net.0.proj.bias"] = (ffn,)
        layout[f"{prefix}ff.net.2.weight"] = (inner, ffn)
        layout[f"{prefix}ff.net.2.bias"] = (inner,)
    return layout


def detect_ltx_text_connectors(
    geometries: Mapping[str, TensorGeometry],
) -> LtxConnectorConfig | None:
    """Detect the LTX text-embedding connector stage in a full header.

    Takes the FULL checkpoint header. The reference activates its
    connector stage when the audio tower has no third block and its
    ``blocks.0.attn1.to_q.bias`` is 3840 wide; that exact condition is
    ported here. When it holds, the video and audio subtrees must both
    match the complete connector layout. Connector keys present without
    satisfying the condition are refused (the reference silently
    ignores them, which would drop checkpoint tensors on the floor).
    Returns None when the checkpoint carries no connector keys.
    """
    scoped = {
        key
        for key in geometries
        if key.startswith((LTX_VIDEO_CONNECTOR_PREFIX, LTX_AUDIO_CONNECTOR_PREFIX))
    }
    trigger = geometries.get(_CONNECTOR_TRIGGER)
    fires = (
        _CONNECTOR_ABSENT_PROBE not in geometries
        and trigger is not None
        and trigger.shape[:1] == (LTX_TEXT_CONNECTOR_CONFIG.inner_dim,)
    )
    if not fires:
        if not scoped:
            return None
        raise GemmaTextDetectError(
            "unsupported *_embeddings_connector variant:"
            f" {len(scoped)} connector keys present but the reference"
            " activation condition does not hold (requires"
            f" {_CONNECTOR_TRIGGER} with width"
            f" {LTX_TEXT_CONNECTOR_CONFIG.inner_dim} and no"
            f" {_CONNECTOR_ABSENT_PROBE})"
        )
    layout = ltx_connector_layout(LTX_TEXT_CONNECTOR_CONFIG)
    expected = {
        f"{prefix}{key}": shape
        for prefix in (LTX_VIDEO_CONNECTOR_PREFIX, LTX_AUDIO_CONNECTOR_PREFIX)
        for key, shape in layout.items()
    }
    problems: list[str] = []
    for key, shape in sorted(expected.items()):
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(scoped - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise GemmaTextDetectError(
            "geometry does not match the LTX text-embedding connector layout: " + shown
        )
    return LTX_TEXT_CONNECTOR_CONFIG


@dataclass(frozen=True)
class LtxGemmaPromptTokens:
    """One prompt's left-padded Gemma row for the LTX text encoder."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    word_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.attention_mask):
            raise ValueError("LTX Gemma ids and attention mask must be non-empty and equal")
        if len(self.ids) != len(self.word_ids):
            raise ValueError("LTX Gemma ids and word ids must be equal length")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ValueError("LTX Gemma attention mask values must be binary integers")


_GEMMA_SPECIAL_TOKEN_IDS = {"<image_soft_token>": 262144, "<end_of_turn>": 106}
_GEMMA_SPECIAL_PATTERN = re.compile(
    "|".join(re.escape(token) for token in _GEMMA_SPECIAL_TOKEN_IDS)
)
_GEMMA_SPECIAL_SPLIT = re.compile(
    "(" + "|".join(re.escape(token) for token in _GEMMA_SPECIAL_TOKEN_IDS) + ")"
)
_EMBEDDING_SPLIT = re.compile(r"(?<=\s)embedding:")


def _segment_ids(segment: str, encode: Callable[[str], Sequence[int]]) -> list[int]:
    """One segment's ids after the reference start-token slice.

    ``encode`` must tokenize WITHOUT adding BOS or EOS. A segment with
    no special token gains BOS in the reference and immediately loses
    it to the [1:] slice, so plain encoding is returned as-is; a
    segment routed through the special-token splitter never gained a
    BOS, so the slice drops its first content token.
    """
    if _GEMMA_SPECIAL_PATTERN.search(segment):
        ids: list[int] = []
        for part in _GEMMA_SPECIAL_SPLIT.split(segment):
            if not part:
                continue
            special = _GEMMA_SPECIAL_TOKEN_IDS.get(part)
            if special is not None:
                ids.append(special)
            else:
                ids.extend(encode(part))
        return ids[1:]
    return list(encode(segment))


def tokenize_ltx_gemma_prompt(
    text: str,
    *,
    encode: Callable[[str], Sequence[int]],
    apply_template: bool = False,
    config: GemmaTextConfig = GEMMA3_LTX_12B_CONFIG,
) -> LtxGemmaPromptTokens:
    """Apply the exact ComfyUI LTXAV Gemma prompt policy.

    ``encode`` is the raw SentencePiece encode with no BOS or EOS
    (see ``GemmaSentencePieceTokenizer`` in dinkster-inference-torch).
    """
    if text.startswith("<start_of_turn>"):
        apply_template = False
    if apply_template:
        text = LTX_GEMMA_PROMPT_TEMPLATE.format(text)
    text = text.replace("\\)", ")").replace("\\(", "(")

    split = _EMBEDDING_SPLIT.split(text)
    segments = [split[0]] + [f"embedding:{part}" for part in split[1:]]
    segments = [segment for segment in segments if segment != ""]

    ids = [config.bos_token_id]
    word_ids = [0]
    for index, segment in enumerate(segments):
        segment_ids = (
            _segment_ids(segment, encode)
            if config.architecture == "gemma3_ltx_12b"
            else list(encode(segment))
        )
        ids.extend(segment_ids)
        word_ids.extend([index + 1] * len(segment_ids))
    padding = max(0, config.min_tokens - len(ids))
    ids = [config.pad_token_id] * padding + ids
    word_ids = [0] * padding + word_ids
    return LtxGemmaPromptTokens(
        tuple(ids), _reference_mask(tuple(ids), config.pad_token_id), tuple(word_ids)
    )


def _reference_mask(ids: tuple[int, ...], pad: int) -> tuple[int, ...]:
    """The reference process_tokens masking for a row with no end
    token (duplicated from anima_text, private there): an opening pad
    run is masked without ending attention, and the first pad past it
    is masked along with everything after."""
    mask: list[int] = []
    left_pad = False
    for index, token in enumerate(ids):
        if index == 0 and token == pad:
            left_pad = True
        if left_pad:
            if token == pad:
                mask.append(0)
                continue
            left_pad = False
        if token == pad:
            mask.extend([0] * (len(ids) - index))
            break
        mask.append(1)
    return tuple(mask)


__all__ = [
    "GEMMA2_LUMINA_2B_CONFIG",
    "GEMMA3_LTX_12B_CONFIG",
    "GEMMA4_LTX_12B_CONFIG",
    "LTXAV_22B_V23_AUDIO_CONNECTOR_CONFIG",
    "LTXAV_22B_V23_VIDEO_CONNECTOR_CONFIG",
    "LTX_AUDIO_CONNECTOR_PREFIX",
    "LTX_DUAL_PROJECTION_LAYOUT",
    "LTX_GEMMA_PROMPT_TEMPLATE",
    "LTX_SINGLE_PROJECTION_LAYOUT",
    "LTX_TEXT_CONNECTOR_CONFIG",
    "LTX_TEXT_STACK_DEPTH",
    "LTX_TEXT_STACK_FEATURES",
    "LTX_VIDEO_CONNECTOR_PREFIX",
    "GemmaTextConfig",
    "GemmaTextDetectError",
    "LtxConnectorConfig",
    "LtxGemmaPromptTokens",
    "LtxTextProjectionKind",
    "detect_gemma_text_config",
    "detect_ltx_text_connectors",
    "detect_ltx_text_projection",
    "gemma_text_layout",
    "ltx_connector_layout",
    "ltx_text_projection_layout",
    "tokenize_ltx_gemma_prompt",
]
