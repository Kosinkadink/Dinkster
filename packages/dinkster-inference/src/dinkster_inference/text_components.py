"""Standalone classic text architecture plans, independent of encoding recipes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .assembly import (
    ComponentPlan,
    _clip_text_plan,  # pyright: ignore[reportPrivateUsage]
    _extract,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from .clip_text import CLIP_G_TEXT_CONFIG
from .quantization import QuantizationError, quantization_error_cause
from .t5_text import BYT5_SMALL_GLYPH_CONFIG, T5_TEXT_OPTIONAL_KEYS, detect_t5_config
from .weights import AssetIdentifiedSource, WeightSource


def plan_text_components(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, ComponentPlan[Any]], ...]:
    """Retain source keys and quantization while resolving CLIP and T5 architecture."""
    candidates = (
        ("", "clip"),
        ("clip_l.", "clip"),
        ("clip_g.", "clip"),
        ("text_encoders.clip_l.transformer.", "clip"),
        ("cond_stage_model.transformer.", "clip"),
        ("conditioner.embedders.0.transformer.", "clip"),
        ("conditioner.embedders.1.model.", "clip"),
        ("conditioner.embedders.0.model.", "clip"),
        ("", "t5"),
        ("t5xxl.transformer.", "t5"),
        ("text_encoders.t5xxl.transformer.", "t5"),
    )
    found: list[tuple[str, ComponentPlan[Any]]] = []
    quantization_error: QuantizationError | None = None
    keys = source.keys()
    for prefix, architecture in candidates:
        if prefix and not any(key.startswith(prefix) for key in keys):
            continue
        try:
            extracted = _extract(source, path, architecture, prefix)
            if architecture == "clip":
                planned = _clip_text_plan("clip", extracted)
                role = "clip_g" if planned.config == CLIP_G_TEXT_CONFIG else "clip_l"
            else:
                config = detect_t5_config(extracted.geometries)
                role = (
                    "umt5xxl"
                    if config.model_type == "umt5"
                    else "byt5_small"
                    if config == BYT5_SMALL_GLYPH_CONFIG
                    else "t5xxl"
                )
                planned = _plan(role, extracted, config, drop=T5_TEXT_OPTIONAL_KEYS)
        except ValueError as error:
            quantization_error = quantization_error or quantization_error_cause(error)
            continue
        facts = planned.identity_facts
        if bind_asset_identity and isinstance(source, AssetIdentifiedSource):
            facts += (f"asset_digest={source.asset_digest}", f"asset_size={source.asset_size}")
        found.append((role, replace(planned, component=role, identity_facts=facts)))
    if not found and quantization_error is not None:
        raise quantization_error
    return tuple(found)
