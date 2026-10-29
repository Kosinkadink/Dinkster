"""OpenCLIP-format text encoders: torch-free layout and conversion.

Raw SDXL checkpoints ship CLIP-G in the OpenCLIP layout
(``conditioner.embedders.1.model.*`` on SDXL base,
``conditioner.embedders.0.model.*`` on the refiner). The reference
converts that layout to its transformers-format text model at load
time (comfy/utils.py clip_text_transformers_convert @ b78cec87):
plain renames for norms/MLP/embeddings, a 3-way row split of the
fused ``attn.in_proj_*`` tensors into q/k/v, and a transpose of the
``text_projection`` matrix into the Linear layout.

This module is the header-level port: :func:`convert_openclip_text`
maps an OpenCLIP-scoped header to the transformers-format MODEL keys,
recording for each one its source key, the derived geometry, and the
:class:`~.weights.TensorTransform` (if any) the executor must apply.
Detection stays REJECT-loud: the input must reproduce the exact
OpenCLIP layout of a known CLIP config (:func:`openclip_text_layout`)
- width routes to at most one candidate, and any drift refuses with
the differences. Inert siblings the reference leaves for its
strict=False load to drop (``logit_scale``, OpenCLIP's ``attn_mask``
buffer) are returned as ignored, never silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .clip_text import (
    KNOWN_CLIP_TEXT_CONFIGS,
    ClipTextConfig,
)
from .weights import (
    RowChunk,
    TensorGeometry,
    TensorTransform,
    Transpose2D,
    transformed_geometry,
)

#: Sibling keys the reference's strict=False load drops: CLIP's
#: contrastive-training temperature and OpenCLIP's causal-mask
#: buffer. Recorded as ignored by the conversion, never errors.
OPENCLIP_TEXT_INERT_KEYS = frozenset({"logit_scale", "attn_mask"})

#: Plain renames inside one residual block (comfy/utils.py
#: transformers_convert resblock_to_replace @ b78cec87).
_RESBLOCK_RENAMES = MappingProxyType(
    {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }
)

#: The q/k/v order of the fused in_proj rows (torch
#: MultiheadAttention packing, consumed in that order by
#: transformers_convert's chunked split).
_IN_PROJ_SPLIT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


class OpenClipTextDetectError(ValueError):
    """The geometry mapping is not a supported OpenCLIP-format text
    encoder; the message names what was found instead."""


@dataclass(frozen=True)
class ConvertedKey:
    """One converted tensor: where it comes from (stripped SOURCE
    key), what the model sees (derived geometry), and how to get
    there (``transform`` None = plain rename/read-through)."""

    source: str
    geometry: TensorGeometry
    transform: TensorTransform | None = None


@dataclass(frozen=True)
class OpenClipTextConversion:
    """The full conversion of one OpenCLIP text-encoder slice:
    transformers-format MODEL key -> :class:`ConvertedKey`, plus the
    inert source keys deliberately left unread."""

    config: ClipTextConfig
    keys: Mapping[str, ConvertedKey]
    ignored: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))


def is_openclip_text(geometries: Mapping[str, TensorGeometry]) -> bool:
    """Whether a text-encoder-scoped header is the OpenCLIP layout
    (the reference routes on the same resblocks spelling)."""
    return any(key.startswith("transformer.resblocks.") for key in geometries)


def openclip_text_layout(config: ClipTextConfig) -> dict[str, tuple[int, ...]]:
    """The exact OpenCLIP state-dict key/shape listing for ``config``
    - the source-side counterpart of clip_text_layout, verified
    against the real sd_xl_base_1.0 header. ``text_projection`` is
    the ``x @ W`` matrix (hidden x hidden - transposed relative to
    the Linear layout, invisible in shape because it is square)."""
    hidden = config.hidden_size
    inter = config.intermediate_size
    layout: dict[str, tuple[int, ...]] = {
        "positional_embedding": (config.max_position_embeddings, hidden),
        "token_embedding.weight": (config.vocab_size, hidden),
        "ln_final.weight": (hidden,),
        "ln_final.bias": (hidden,),
        "text_projection": (hidden, hidden),
    }
    for i in range(config.num_hidden_layers):
        prefix = f"transformer.resblocks.{i}."
        layout[f"{prefix}attn.in_proj_weight"] = (3 * hidden, hidden)
        layout[f"{prefix}attn.in_proj_bias"] = (3 * hidden,)
        layout[f"{prefix}attn.out_proj.weight"] = (hidden, hidden)
        layout[f"{prefix}attn.out_proj.bias"] = (hidden,)
        for norm in ("ln_1", "ln_2"):
            layout[f"{prefix}{norm}.weight"] = (hidden,)
            layout[f"{prefix}{norm}.bias"] = (hidden,)
        layout[f"{prefix}mlp.c_fc.weight"] = (inter, hidden)
        layout[f"{prefix}mlp.c_fc.bias"] = (inter,)
        layout[f"{prefix}mlp.c_proj.weight"] = (hidden, inter)
        layout[f"{prefix}mlp.c_proj.bias"] = (hidden,)
    return layout


def _model_key(stripped: str) -> tuple[str, TensorTransform | None] | None:
    """The transformers-format model key (and transform) one OpenCLIP
    source key feeds - None for the fused in_proj tensors, which fan
    out to three model keys (handled by the caller)."""
    fixed = {
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
    }
    if stripped in fixed:
        return fixed[stripped], None
    if not stripped.startswith("transformer.resblocks."):
        return None
    rest = stripped[len("transformer.resblocks.") :]
    index, _, param = rest.partition(".")
    if not index.isdigit():
        return None
    for old, new in _RESBLOCK_RENAMES.items():
        for suffix in (".weight", ".bias"):
            if param == old + suffix:
                return f"text_model.encoder.layers.{index}.{new}{suffix}", None
    return None


def convert_openclip_text(
    geometries: Mapping[str, TensorGeometry],
) -> OpenClipTextConversion:
    """Convert an OpenCLIP-scoped header (any checkpoint prefix
    already stripped) to transformers-format model keys, or refuse
    loudly. Dtypes ride through unchanged - transforms never cast."""
    if not geometries:
        raise OpenClipTextDetectError("empty state dict header")
    token = geometries.get("token_embedding.weight")
    if token is None:
        raise OpenClipTextDetectError("not an OpenCLIP text encoder (no token_embedding.weight)")
    if len(token.shape) != 2:
        raise OpenClipTextDetectError(
            f"token_embedding.weight has rank {len(token.shape)}, expected 2"
        )
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
        raise OpenClipTextDetectError(
            f"unknown OpenCLIP text geometry: token embedding"
            f" {vocab}x{hidden}; only the known CLIP configs are accepted"
        )

    layout = openclip_text_layout(config)
    problems: list[str] = []
    ignored: list[str] = []
    keys: dict[str, ConvertedKey] = {}

    # The projection carries two reference-accepted spellings
    # (comfy/utils.py clip_text_transformers_convert @ b78cec87):
    # bare ``text_projection`` is the OpenCLIP x @ W matrix and gets
    # transposed into the Linear layout; ``text_projection.weight``
    # is already the Linear layout and is renamed read-through.
    # Exactly one must be present.
    bare_projection = geometries.get("text_projection")
    weight_projection = geometries.get("text_projection.weight")
    if bare_projection is not None and weight_projection is not None:
        problems.append("both text_projection and text_projection.weight present")
    elif bare_projection is None and weight_projection is None:
        problems.append("missing text_projection")
    else:
        found = bare_projection if bare_projection is not None else weight_projection
        assert found is not None
        projection_source = (
            "text_projection" if bare_projection is not None else "text_projection.weight"
        )
        expected = layout["text_projection"]
        if found.shape != expected:
            problems.append(f"{projection_source}: expected shape {expected}, found {found.shape}")
        else:
            transform = Transpose2D() if bare_projection is not None else None
            keys["text_projection.weight"] = ConvertedKey(
                source=projection_source,
                geometry=(transformed_geometry(found, transform) if transform else found),
                transform=transform,
            )

    for source, expected in layout.items():
        if source == "text_projection":
            continue  # both spellings handled above
        found = geometries.get(source)
        if found is None:
            problems.append(f"missing {source}")
            continue
        if found.shape != expected:
            problems.append(f"{source}: expected shape {expected}, found {found.shape}")
            continue
        if source.endswith((".attn.in_proj_weight", ".attn.in_proj_bias")):
            index = source[len("transformer.resblocks.") :].partition(".")[0]
            suffix = ".weight" if source.endswith("_weight") else ".bias"
            for part, proj in enumerate(_IN_PROJ_SPLIT):
                transform = RowChunk(part=part, parts=len(_IN_PROJ_SPLIT))
                keys[f"text_model.encoder.layers.{index}.{proj}{suffix}"] = ConvertedKey(
                    source=source,
                    geometry=transformed_geometry(found, transform),
                    transform=transform,
                )
            continue
        target = _model_key(source)
        assert target is not None  # every layout key maps
        model_key, transform = target
        geometry = transformed_geometry(found, transform) if transform else found
        keys[model_key] = ConvertedKey(source=source, geometry=geometry, transform=transform)
    projection_spellings = {"text_projection", "text_projection.weight"}
    for source in sorted(set(geometries) - set(layout) - projection_spellings):
        if source in OPENCLIP_TEXT_INERT_KEYS:
            ignored.append(source)
        else:
            problems.append(f"unexpected key {source}")
    if problems:
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise OpenClipTextDetectError(
            f"geometry does not match the OpenCLIP"
            f" ({config.hidden_size}x{config.num_hidden_layers}) text"
            f" layout: {shown}"
        )
    return OpenClipTextConversion(config=config, keys=keys, ignored=tuple(ignored))


__all__ = [
    "OPENCLIP_TEXT_INERT_KEYS",
    "ConvertedKey",
    "OpenClipTextConversion",
    "OpenClipTextDetectError",
    "convert_openclip_text",
    "is_openclip_text",
    "openclip_text_layout",
]
