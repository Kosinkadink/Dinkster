"""Header-derived output profiles for the generic model loader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

from .assembly import ComponentPlan
from .component_registry import component_plans
from .identity import runtime_component_identity
from .refusal import NativeRefusalError
from .registries import builtin_registries
from .runtime import plan_native
from .sources import (
    SafetensorsSource,
    load_safetensors_header_from_file,
)

MODEL_OUTPUT_PROFILE_REVISION = "1"
ModelProfileKind = Literal["checkpoint", "model"]


@dataclass(frozen=True)
class ModelOutputProfile:
    kind: ModelProfileKind
    document: Mapping[str, object]

    def to_json(self) -> str:
        return json.dumps(self.document, sort_keys=True, separators=(",", ":"))


def _sha256_document(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _components_document(
    family_id: str, components: tuple[ComponentPlan[Any] | None, ...]
) -> dict[str, object]:
    present = tuple(component for component in components if component is not None)
    return {
        "family": family_id,
        "plans": [
            {
                "component": component.component,
                "identity": _sha256_document(runtime_component_identity(family_id, (component,))),
                "mappingDigest": _sha256_document(
                    [[key, component.keys[key]] for key in sorted(component.keys)]
                ),
            }
            for component in present
        ],
    }


def _shape_document(source: SafetensorsSource, components: Mapping[str, object]) -> object:
    return {
        "components": components,
        "tensors": [
            [key, source.entry(key).geometry.dtype.name, list(source.entry(key).geometry.shape)]
            for key in sorted(source.keys())
        ],
    }


def _shape_digest(source: SafetensorsSource, components: Mapping[str, object]) -> str:
    return _sha256_document(_shape_document(source, components))


def probe_model_output_profile(source: SafetensorsSource) -> ModelOutputProfile:
    """Classify one verified safetensors source without loading tensor payloads."""
    if source.asset_digest is None:
        raise ValueError("model output profiles require a content-addressed asset source")

    diagnostics: list[str] = []
    try:
        plan = plan_native(checkpoint=source)
    except (ValueError, NativeRefusalError):
        try:
            registry = builtin_registries().components
            matches = registry.detect(source, source.path)
            descriptor, _role, planned = registry.select_detected(matches, "model")
        except ValueError:
            kind: ModelProfileKind = "model"
            components: dict[str, object] = {"family": None, "plans": []}
            diagnostics.append(
                "No supported component plan was detected; execution will attempt MODEL loading."
            )
        else:
            kind = "model"
            family_id = descriptor.family_for(planned)
            components = _components_document(family_id, component_plans(planned))
    else:
        kind = "checkpoint"
        components = _components_document(plan.family.id, plan.identity_components)

    entries = (
        [
            {"id": "model", "name": "MODEL", "type": "model"},
            {"id": "clip", "name": "CLIP", "type": "clip"},
            {"id": "vae", "name": "VAE", "type": "vae"},
        ]
        if kind == "checkpoint"
        else [{"id": "model", "name": "MODEL", "type": "model"}]
    )
    document: dict[str, object] = {
        "entries": entries,
        "assetDigest": source.asset_digest,
        "detectorRevision": MODEL_OUTPUT_PROFILE_REVISION,
        "shapeDigest": _shape_digest(source, components),
        "components": components,
        "diagnostics": diagnostics,
    }
    return ModelOutputProfile(kind=kind, document=document)


def load_model_output_profile(
    path: Path,
    *,
    asset_digest: str,
    asset_size: int,
    stored: str | None = None,
    handle: BinaryIO | None = None,
) -> ModelOutputProfile:
    if handle is None:
        with path.open("rb") as opened:
            return load_model_output_profile(
                path,
                asset_digest=asset_digest,
                asset_size=asset_size,
                stored=stored,
                handle=opened,
            )
    source = replace(
        load_safetensors_header_from_file(
            handle,
            path=path,
            asset_digest=asset_digest,
            asset_size=asset_size,
        ),
        configuration_file=handle,
    )
    profile = probe_model_output_profile(source)
    if stored is None:
        return profile
    try:
        supplied: object = json.loads(stored)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("model output profile must be valid JSON") from error
    if supplied != profile.document:
        raise ValueError("stored model output profile does not match current asset metadata")
    return profile


__all__ = [
    "MODEL_OUTPUT_PROFILE_REVISION",
    "ModelOutputProfile",
    "load_model_output_profile",
    "probe_model_output_profile",
]
