"""Typed source-mask declarations and immutable compiled effect fields."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

from .conditioning import PayloadDescriptor
from .token_layout import ModelTokenLayout, RowValue, TokenGridTransform, TokenRowTable

__all__ = [
    "CompiledEffectMaskField",
    "EffectMaskInput",
    "MaskInputError",
    "MaskMediaPlacement",
    "SourceMaskInput",
    "compile_effect_mask_field",
]

_ASSET_DIGEST = re.compile(r"^blake3:[0-9a-f]{64}$")
_FIELD_DOMAIN = "dinkster.effect-mask-field.v1"
_INPUT_DOMAIN = "dinkster.effect-mask-input.v1"


class MaskInputError(ValueError):
    pass


class MaskMediaPlacement(StrEnum):
    FULL_DOMAIN = "full-domain"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _validate_payload(
    payload: PayloadDescriptor,
    source_digest: str,
    axis_identity: tuple[str, ...],
    media_placement: MaskMediaPlacement,
) -> None:
    if type(payload) is not PayloadDescriptor:
        raise MaskInputError("mask payload must be an exact PayloadDescriptor")
    if type(source_digest) is not str or _ASSET_DIGEST.fullmatch(source_digest) is None:
        raise MaskInputError("mask source digest must be a canonical BLAKE3 asset digest")
    if payload.reference.id != source_digest:
        raise MaskInputError("mask payload reference must equal its BLAKE3 source digest")
    if type(axis_identity) is not tuple or len(axis_identity) != len(payload.shape):
        raise MaskInputError("mask axis identity must name every source dimension")
    if any(type(axis) is not str or not axis or axis != axis.strip() for axis in axis_identity):
        raise MaskInputError("mask axis identities must be non-empty trimmed strings")
    if len(set(axis_identity)) != len(axis_identity):
        raise MaskInputError("mask axis identities must be unique")
    if type(media_placement) is not MaskMediaPlacement:
        raise MaskInputError("mask media placement must be an exact MaskMediaPlacement")


@dataclass(frozen=True, slots=True)
class SourceMaskInput:
    """A mask consumed by an auxiliary source or conditioning transform."""

    payload: PayloadDescriptor
    source_digest: str
    axis_identity: tuple[str, ...]
    media_placement: MaskMediaPlacement

    def __post_init__(self) -> None:
        _validate_payload(
            self.payload, self.source_digest, self.axis_identity, self.media_placement
        )


@dataclass(frozen=True, slots=True)
class EffectMaskInput:
    """One content-addressed mask declaration for an applying site."""

    payload: PayloadDescriptor
    source_digest: str
    axis_identity: tuple[str, ...]
    media_placement: MaskMediaPlacement
    target_segment: str
    transform: str

    def __post_init__(self) -> None:
        _validate_payload(
            self.payload, self.source_digest, self.axis_identity, self.media_placement
        )
        for name, value in (
            ("target segment", self.target_segment),
            ("transform", self.transform),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise MaskInputError(f"effect-mask {name} must be a non-empty trimmed string")

    @property
    def canonical_preimage(self) -> str:
        facts = {
            "axis_identity": list(self.axis_identity),
            "dtype": self.payload.dtype,
            "media_placement": self.media_placement.value,
            "shape": list(self.payload.shape),
            "source_digest": self.source_digest,
            "space": self.payload.space,
            "target_segment": self.target_segment,
            "transform": self.transform,
        }
        return _canonical_json((_INPUT_DOMAIN, facts))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_preimage.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class CompiledEffectMaskField:
    """A source declaration compiled onto its target segment's semantic rows."""

    input_digest: str
    source_digest: str
    layout_digest: str
    transform_digest: str
    target_segment: str
    table: TokenRowTable
    canonical_preimage: str
    digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("compiled effect-mask fields are created by compile_effect_mask_field")


def _mask_value_facts(value: RowValue) -> object:
    if type(value) is float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise MaskInputError("effect-mask values must be finite exact floats in [0, 1]")
        return 0.0 if value == 0.0 else value
    if type(value) is tuple and value:
        return [_mask_value_facts(item) for item in value]
    raise MaskInputError("effect-mask rows must contain exact floats or non-empty tuples")


def compile_effect_mask_field(
    source: EffectMaskInput,
    layout: ModelTokenLayout,
    transform: TokenGridTransform,
    table: TokenRowTable,
) -> CompiledEffectMaskField:
    """Validate and freeze a family-transformed mask over semantic rows."""

    if type(source) is not EffectMaskInput:
        raise MaskInputError("source must be an exact EffectMaskInput")
    if type(layout) is not ModelTokenLayout:
        raise MaskInputError("layout must be an exact ModelTokenLayout")
    if type(transform) is not TokenGridTransform:
        raise MaskInputError("transform must be an exact TokenGridTransform")
    if type(table) is not TokenRowTable:
        raise MaskInputError("table must be an exact TokenRowTable")
    try:
        target = layout.by_identity(source.target_segment)
    except KeyError:
        raise MaskInputError("mask_layout_mismatch: target segment is absent") from None
    if transform.transform != source.transform:
        raise MaskInputError("missing_token_grid_transform: transform identity does not match")
    if transform.segment_identity != target.identity or transform.modality != target.modality:
        raise MaskInputError("mask_layout_mismatch: transform does not bind the target segment")
    media_geometry = tuple(
        size
        for size, axis in zip(source.payload.shape, source.axis_identity, strict=True)
        if axis != "batch"
    )
    if transform.source_geometry != media_geometry:
        raise MaskInputError("mask_layout_mismatch: transform source geometry does not match")
    if table.rows != target.rows:
        raise MaskInputError(
            "mask_layout_mismatch: row table must cover exactly the target segment's semantic rows"
        )
    values = [_mask_value_facts(value) for value in table.values]
    semantic_layout_digest = ModelTokenLayout(layout.segments, 0).digest
    facts = {
        "input_digest": source.digest,
        "layout_digest": semantic_layout_digest,
        "source_digest": source.source_digest,
        "table": values,
        "target_segment": source.target_segment,
        "transform_digest": transform.digest,
    }
    preimage = _canonical_json((_FIELD_DOMAIN, facts))
    field = object.__new__(CompiledEffectMaskField)
    object.__setattr__(field, "input_digest", source.digest)
    object.__setattr__(field, "source_digest", source.source_digest)
    object.__setattr__(field, "layout_digest", semantic_layout_digest)
    object.__setattr__(field, "transform_digest", transform.digest)
    object.__setattr__(field, "target_segment", source.target_segment)
    object.__setattr__(field, "table", table)
    object.__setattr__(field, "canonical_preimage", preimage)
    object.__setattr__(field, "digest", hashlib.sha256(preimage.encode("utf-8")).hexdigest())
    return field
