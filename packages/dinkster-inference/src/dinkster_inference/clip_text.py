"""SD1/SDXL CLIP text model: torch-free config, layout, detection.

The reference builds its text model from transformers-style config
JSON (comfy/clip_model.py CLIPTextModel_ @ b78cec87, reading
comfy/sd1_clip_config.json for CLIP-L and comfy/clip_config_bigg.json
for CLIP-G). :class:`ClipTextConfig` carries exactly the fields that
construction consumes - the JSONs' dropout/initializer/bos fields are
inert there and are not modeled.

Two reference behaviors are pinned as facts of the layout:

- ``vocab_size`` is HARDCODED to 49408 in the reference
  (CLIPEmbeddings' default is never overridden by the config dict),
  so both known configs carry it explicitly here.
- ``text_projection`` is always constructed as hidden->hidden
  (``projection_dim`` in the JSON is ignored by the reference), and
  SD1 checkpoints do not ship the key (the reference loads with
  strict=False); detection treats it as the single optional key.

Detection REJECTS rather than guesses: attention-head count and
activation are not derivable from tensor shapes, so only the two
known geometries (CLIP-L: 768x12, CLIP-G: 1280x32) are accepted, by
full layout comparison against :func:`clip_text_layout`. OpenCLIP-
format checkpoints (SDXL's conditioner.embedders.1.model.*, converted
upstream by comfy/utils.py transformers_convert), long-context
position tables, and every other text-encoder family refuse with a
:class:`ClipTextDetectError` naming what was found; deferred variants
are ledgered in ROADMAP.md ("Native inference").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .weights import TensorGeometry

#: The activations comfy/clip_model.py ACTIVATIONS defines minus
#: gelu_pytorch_tanh, which no supported SD1/SDXL text model uses.
CLIP_TEXT_ACTIVATIONS = ("quick_gelu", "gelu")

#: The reference loads SD1 CLIP-L checkpoints with strict=False; this
#: is the only key legitimately absent from a supported checkpoint.
CLIP_TEXT_OPTIONAL_KEYS = frozenset({"text_projection.weight"})


class ClipTextDetectError(ValueError):
    """The geometry mapping is not a supported CLIP text-model
    checkpoint; the message names what was found instead."""


@dataclass(frozen=True)
class ClipTextConfig:
    """The construction-relevant subset of the reference config JSON.

    ``layer_norm_eps`` is nominally configurable but the reference
    never reads it from the JSON (operations.LayerNorm keeps torch's
    1e-5 default); it is kept explicit so the pin is visible.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    vocab_size: int = 49408
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 49407

    def __post_init__(self) -> None:
        for field in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by"
                f" num_attention_heads {self.num_attention_heads}"
            )
        if self.hidden_act not in CLIP_TEXT_ACTIVATIONS:
            raise ValueError(
                f"hidden_act must be one of {CLIP_TEXT_ACTIVATIONS}, got {self.hidden_act!r}"
            )
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError(
                f"eos_token_id {self.eos_token_id} is outside the vocabulary of {self.vocab_size}"
            )
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")


#: comfy/sd1_clip_config.json @ b78cec87 (openai/clip-vit-large-patch14).
CLIP_L_TEXT_CONFIG = ClipTextConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="quick_gelu",
)

#: comfy/clip_config_bigg.json @ b78cec87 (OpenCLIP bigG, converted).
CLIP_G_TEXT_CONFIG = ClipTextConfig(
    hidden_size=1280,
    num_hidden_layers=32,
    num_attention_heads=20,
    intermediate_size=5120,
    hidden_act="gelu",
)

#: The geometries detection accepts; hidden sizes are distinct, so
#: the token-embedding shape routes to at most one candidate.
KNOWN_CLIP_TEXT_CONFIGS = (CLIP_L_TEXT_CONFIG, CLIP_G_TEXT_CONFIG)


def clip_text_layout(config: ClipTextConfig) -> dict[str, tuple[int, ...]]:
    """The exact state-dict key/shape listing the reference
    CLIPTextModel produces for ``config`` - the single source both
    detection (compare a header against it) and the torch tests
    (compare the constructed module against the executed reference)
    consume."""
    hidden = config.hidden_size
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "text_model.embeddings.token_embedding.weight": (
            config.vocab_size,
            hidden,
        ),
        "text_model.embeddings.position_embedding.weight": (
            config.max_position_embeddings,
            hidden,
        ),
        "text_model.final_layer_norm.weight": (hidden,),
        "text_model.final_layer_norm.bias": (hidden,),
        "text_projection.weight": (hidden, hidden),
    }
    for i in range(config.num_hidden_layers):
        prefix = f"text_model.encoder.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layout[f"{prefix}self_attn.{proj}.weight"] = (hidden, hidden)
            layout[f"{prefix}self_attn.{proj}.bias"] = (hidden,)
        for norm in ("layer_norm1", "layer_norm2"):
            layout[f"{prefix}{norm}.weight"] = (hidden,)
            layout[f"{prefix}{norm}.bias"] = (hidden,)
        layout[f"{prefix}mlp.fc1.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.fc1.bias"] = (inter,)
        layout[f"{prefix}mlp.fc2.weight"] = (hidden, inter)
        layout[f"{prefix}mlp.fc2.bias"] = (hidden,)
    return layout


def _config_name(config: ClipTextConfig) -> str:
    if config == CLIP_L_TEXT_CONFIG:
        return "CLIP-L"
    if config == CLIP_G_TEXT_CONFIG:
        return "CLIP-G"
    return f"CLIP text model ({config.hidden_size}x{config.num_hidden_layers})"


def detect_clip_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> ClipTextConfig:
    """Classify a text-model-scoped header (``text_model.*`` keys, any
    checkpoint prefix already stripped) as CLIP-L or CLIP-G, or refuse
    loudly. Dtypes are ignored - checkpoints legitimately ship fp16."""
    if not geometries:
        raise ClipTextDetectError("empty state dict header")
    if any(key.startswith("transformer.resblocks.") for key in geometries):
        raise ClipTextDetectError(
            "OpenCLIP-format text encoder (transformer.resblocks.*);"
            " transformers-format conversion is not ported yet"
            " (ROADMAP: Native inference)"
        )
    token_key = "text_model.embeddings.token_embedding.weight"
    token = geometries.get(token_key)
    if token is None:
        raise ClipTextDetectError(f"not a transformers-format CLIP text model (no {token_key})")
    if len(token.shape) != 2:
        raise ClipTextDetectError(f"{token_key} has rank {len(token.shape)}, expected 2")
    vocab, hidden = token.shape
    config = next(
        (
            candidate
            for candidate in KNOWN_CLIP_TEXT_CONFIGS
            if candidate.hidden_size == hidden and candidate.vocab_size == vocab
        ),
        None,
    )
    if config is None:
        raise ClipTextDetectError(
            f"unknown CLIP text geometry: token embedding {vocab}x{hidden};"
            " attention-head count is not derivable from shapes, so only"
            " CLIP-L (49408x768) and CLIP-G (49408x1280) are accepted"
        )

    layout = clip_text_layout(config)
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            if key not in CLIP_TEXT_OPTIONAL_KEYS:
                problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise ClipTextDetectError(
            f"geometry does not match the {_config_name(config)} layout: {shown}"
        )
    return config


__all__ = [
    "CLIP_G_TEXT_CONFIG",
    "CLIP_L_TEXT_CONFIG",
    "CLIP_TEXT_ACTIVATIONS",
    "CLIP_TEXT_OPTIONAL_KEYS",
    "KNOWN_CLIP_TEXT_CONFIGS",
    "ClipTextConfig",
    "ClipTextDetectError",
    "clip_text_layout",
    "detect_clip_text_config",
]
