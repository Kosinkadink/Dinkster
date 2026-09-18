"""Canonical artifact and execution identity for encoded GGUF storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from dinkster_schema.names import canonical_name

from .gguf import (
    GGUFComponentMap,
    GGUFMetadataValue,
    GGUFResidencySelection,
    GGUFSource,
    GGUFValueType,
    GGUFWeightSource,
    load_gguf,
    map_gguf_component,
    open_gguf_artifact_file,
)
from .registry import validate_registry_id

_MANIFEST_SCHEMA = "dinkster.gguf.parsed-manifest.v1"
_ENDIANNESS_POLICY = "little"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")


class GGUFIdentityError(ValueError):
    """The source bytes and parsed GGUF interpretation are not one stable artifact."""


def _require_sha256(name: str, value: str) -> None:
    raw = cast("object", value)
    if not isinstance(raw, str) or _SHA256_RE.fullmatch(raw) is None:
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")


def _require_token(name: str, value: str) -> None:
    raw = cast("object", value)
    if not isinstance(raw, str) or _TOKEN_RE.fullmatch(raw) is None:
        raise ValueError(f"{name} must be a canonical lowercase token")


def _scalar(value_type: GGUFValueType, value: object) -> object:
    if value_type is GGUFValueType.FLOAT32:
        return {"bits": struct.pack("<f", cast("float", value)).hex()}
    if value_type is GGUFValueType.FLOAT64:
        return {"bits": struct.pack("<d", cast("float", value)).hex()}
    return value


def _metadata_value(field: GGUFMetadataValue) -> dict[str, object]:
    if field.value_type is GGUFValueType.ARRAY:
        assert field.element_type is not None
        assert isinstance(field.value, tuple)
        return {
            "elementType": field.element_type.name,
            "type": field.value_type.name,
            "value": [_scalar(field.element_type, value) for value in field.value],
        }
    return {
        "type": field.value_type.name,
        "value": _scalar(field.value_type, field.value),
    }


def _component_manifest(component: GGUFComponentMap) -> dict[str, object]:
    return {
        "architecture": component.architecture,
        "component": component.component,
        "familyId": component.family_id,
        "mapperId": component.mapper_id,
        "tensorPrefix": component.tensor_prefix,
        "tensors": [
            {
                "ggmlType": tensor.ggml_type.name,
                "logicalShape": list(tensor.logical_shape),
                "modelKey": tensor.model_key,
                "nbytes": tensor.nbytes,
                "offset": tensor.offset,
                "sourceName": tensor.source_name,
            }
            for _, tensor in sorted(component.tensors.items())
        ],
    }


def _parsed_manifest(source: GGUFSource, component: GGUFComponentMap) -> bytes:
    quantization = source.metadata_values.get("general.quantization_version")
    document = {
        "alignment": source.alignment,
        "componentMap": _component_manifest(component),
        "dataOffset": source.data_offset,
        "endianness": _ENDIANNESS_POLICY,
        "metadata": [
            {"key": key, **_metadata_value(field)}
            for key, field in sorted(source.metadata_values.items())
        ],
        "quantizationVersion": quantization.value if quantization is not None else None,
        "schema": _MANIFEST_SCHEMA,
        "tensors": [
            {
                "ggmlType": {
                    "blockBytes": tensor.ggml_type.block_bytes,
                    "blockElements": tensor.ggml_type.block_elements,
                    "code": tensor.ggml_type.code,
                    "name": tensor.ggml_type.name,
                    "quantized": tensor.ggml_type.quantized,
                },
                "name": tensor.name,
                "nbytes": tensor.nbytes,
                "offset": tensor.offset,
                "relativeOffset": tensor.relative_offset,
                "shape": list(tensor.shape),
                "wireShape": list(tensor.wire_shape),
            }
            for _, tensor in sorted(source.tensors.items())
        ],
        "version": source.version,
    }
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def gguf_manifest_sha256(source: GGUFSource, component: GGUFComponentMap) -> str:
    """Digest one canonical parsed and mapped GGUF interpretation."""

    return hashlib.sha256(_parsed_manifest(source, component)).hexdigest()


@dataclass(frozen=True)
class GGUFArtifactIdentity:
    file_sha256: str
    manifest_sha256: str
    mapper_id: str
    architecture: str
    family_id: str
    component: str

    def __post_init__(self) -> None:
        _require_sha256("GGUF file digest", self.file_sha256)
        _require_sha256("GGUF manifest digest", self.manifest_sha256)
        for name, value in (
            ("mapper_id", self.mapper_id),
            ("architecture", self.architecture),
            ("family_id", self.family_id),
            ("component", self.component),
        ):
            raw = cast("object", value)
            if not isinstance(raw, str) or not raw or "\n" in raw:
                raise ValueError(f"GGUF artifact {name} must be non-empty and newline-free")

    @property
    def facts(self) -> tuple[str, ...]:
        return (
            f"gguf.artifact.file_sha256={self.file_sha256}",
            f"gguf.artifact.manifest_sha256={self.manifest_sha256}",
            f"gguf.artifact.mapper_id={self.mapper_id}",
            f"gguf.artifact.architecture={self.architecture}",
            f"gguf.artifact.family_id={self.family_id}",
            f"gguf.artifact.component={self.component}",
        )


def identify_gguf_artifact(source: GGUFSource) -> GGUFArtifactIdentity:
    """Bind complete bytes to one canonical parsed and mapped interpretation."""

    digest = hashlib.sha256()
    with open_gguf_artifact_file(source.path) as file:
        before = os.fstat(file.fileno())
        current = load_gguf(source.path, _artifact_file=file)
        file.seek(0)
        while chunk := file.read(8 * 1024**2):
            digest.update(chunk)
        after = os.fstat(file.fileno())
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_after != fingerprint_before:
        raise GGUFIdentityError("GGUF artifact changed during identity verification")
    source_component = map_gguf_component(source)
    current_component = map_gguf_component(current)
    source_manifest = _parsed_manifest(source, source_component)
    current_manifest = _parsed_manifest(current, current_component)
    if current_manifest != source_manifest:
        raise GGUFIdentityError("GGUF source facts changed after parsing")
    manifest_sha256 = gguf_manifest_sha256(current, current_component)
    return GGUFArtifactIdentity(
        file_sha256=digest.hexdigest(),
        manifest_sha256=manifest_sha256,
        mapper_id=current_component.mapper_id,
        architecture=current_component.architecture,
        family_id=current_component.family_id,
        component=current_component.component,
    )


class GGUFExecutionKind(StrEnum):
    REFERENCE_DECODE = "reference-decode"
    BOUNDED_DECODE = "bounded-decode"
    CACHED_DECODE = "cached-decode"
    FUSED = "fused"


@dataclass(frozen=True)
class GGUFExecutionRoute:
    provider_id: str
    implementation_version: str
    kind: GGUFExecutionKind
    device_kind: str
    device_capability: str
    compute_dtype: str
    accumulation_dtype: str
    decoded_cache: str | None = None
    fused_matmul: str | None = None

    def __post_init__(self) -> None:
        validate_registry_id(self.provider_id)
        for name, value in (
            ("implementation_version", self.implementation_version),
            ("device_kind", self.device_kind),
            ("device_capability", self.device_capability),
            ("compute_dtype", self.compute_dtype),
            ("accumulation_dtype", self.accumulation_dtype),
        ):
            _require_token(name, value)
        if not isinstance(cast("object", self.kind), GGUFExecutionKind):
            raise TypeError("GGUF execution kind must be GGUFExecutionKind")
        if self.decoded_cache is not None:
            _require_token("decoded_cache", self.decoded_cache)
        if self.fused_matmul is not None:
            _require_token("fused_matmul", self.fused_matmul)

    @property
    def facts(self) -> tuple[str, ...]:
        facts = (
            f"gguf.route.provider_key={canonical_name(self.provider_id)}",
            f"gguf.route.implementation_version={self.implementation_version}",
            f"gguf.route.kind={self.kind.value}",
            f"gguf.route.device_kind={self.device_kind}",
            f"gguf.route.device_capability={self.device_capability}",
            f"gguf.route.compute_dtype={self.compute_dtype}",
            f"gguf.route.accumulation_dtype={self.accumulation_dtype}",
        )
        if self.decoded_cache is not None:
            facts = (*facts, f"gguf.route.decoded_cache={self.decoded_cache}")
        if self.fused_matmul is not None:
            facts = (*facts, f"gguf.route.fused_matmul={self.fused_matmul}")
        return facts


def gguf_runtime_identity_facts(
    artifact: GGUFArtifactIdentity,
    route: GGUFExecutionRoute,
) -> tuple[str, ...]:
    """Return canonical facts for the existing runtime identity builder."""

    return (*artifact.facts, *route.facts)


def load_gguf_weight_source(
    path: Path,
    *,
    residency_mode: GGUFResidencySelection = "auto",
    decoded_cache_budget: int | None = None,
) -> GGUFWeightSource:
    """Parse, map, authenticate, and bind the selected execution route."""

    return GGUFWeightSource(
        path,
        residency_mode=residency_mode,
        decoded_cache_budget=decoded_cache_budget,
    )


__all__ = [
    "GGUFArtifactIdentity",
    "GGUFExecutionKind",
    "GGUFExecutionRoute",
    "GGUFIdentityError",
    "gguf_runtime_identity_facts",
    "identify_gguf_artifact",
    "load_gguf_weight_source",
]
