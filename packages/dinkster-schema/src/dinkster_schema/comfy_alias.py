"""Maintained ComfyUI-to-native translation registry data.

Aliases are import metadata, not executable node schemas. Each record carries
an exact shared ``ReplacementRule`` and an import-only source schema snapshot
so positional LiteGraph data can be decoded without a ComfyUI backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from .model import NodeSchema
from .names import validate_name
from .replace import (
    ReplacementRule,
    rule_from_wire,
    rule_to_wire,
    validate_replacement_references,
)
from .wire import (
    SCHEMA_WIRE_VERSION,
    schema_from_wire,
    schema_to_wire,
)

COMFY_ALIAS_FORMAT = "dinkster-comfy-alias/1"
COMFY_CORE_REVISION = "b78cec87"
COMFY_CORE_REVISIONS = frozenset(
    (
        COMFY_CORE_REVISION,
        "8a33128f",
        "c67885b1",
        "15eb748b",
        "95539f56",
        "f00bfd610cb001381603669e2cc01160ae37aaf3",
    )
)

ComfyAliasMappingKind = Literal["op", "family"]
ComfyAliasConfidenceTier = Literal["exact", "parametric", "equivalent", "grouped"]
ToleranceOperator = Literal["<=", ">="]

_MAPPING_KINDS = frozenset({"op", "family"})
_CONFIDENCE_TIERS = frozenset({"exact", "parametric", "equivalent", "grouped"})
_TOLERANCE_OPERATORS = frozenset({"<=", ">="})


@dataclass(frozen=True)
class ComfyAliasSource:
    pack: str
    node_class: str
    node_type: str
    revision: str

    def __post_init__(self) -> None:
        if validate_name(self.pack) is not None:
            raise ValueError(f"invalid comfy alias source pack: {self.pack!r}")
        if not self.node_class:
            raise ValueError("comfy alias source nodeClass must be non-empty")
        if not self.node_type:
            raise ValueError("comfy alias source nodeType must be non-empty")
        if not self.revision:
            raise ValueError("comfy alias source revision must be non-empty")
        if self.pack == "comfy-core" and self.revision not in COMFY_CORE_REVISIONS:
            revisions = ", ".join(sorted(COMFY_CORE_REVISIONS))
            raise ValueError(f"comfy-core aliases must use one of revisions: {revisions}")


@dataclass(frozen=True)
class ComfyAliasFamily:
    id: str
    provider: str = ""

    def __post_init__(self) -> None:
        if validate_name(self.id) is not None:
            raise ValueError(f"invalid comfy alias family id: {self.id!r}")
        if self.provider and validate_name(self.provider) is not None:
            raise ValueError(f"invalid comfy alias family provider: {self.provider!r}")


@dataclass(frozen=True)
class ComfyAliasTolerance:
    metric: str
    operator: ToleranceOperator
    value: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("comfy alias tolerance metric must be non-empty")
        if self.operator not in _TOLERANCE_OPERATORS:
            raise ValueError(f"unknown comfy alias tolerance operator: {self.operator!r}")
        try:
            finite = type(self.value) in (int, float) and math.isfinite(self.value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("comfy alias tolerance value must be a finite number")


@dataclass(frozen=True)
class ComfyAliasConfidence:
    tier: ComfyAliasConfidenceTier
    evidence: tuple[str, ...]
    tolerances: tuple[ComfyAliasTolerance, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in _CONFIDENCE_TIERS:
            raise ValueError(f"unknown comfy alias confidence tier: {self.tier!r}")
        if not self.evidence or any(not item for item in self.evidence):
            raise ValueError("comfy alias confidence requires non-empty evidence strings")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("comfy alias confidence has duplicate evidence")
        if self.tier == "equivalent" and not self.tolerances:
            raise ValueError("equivalent comfy alias confidence requires tolerances")
        if self.tier == "exact" and self.tolerances:
            raise ValueError("exact comfy alias confidence forbids tolerances")


@dataclass(frozen=True)
class ComfyAliasRecord:
    id: str
    mapping_kind: ComfyAliasMappingKind
    carrier: str
    source: ComfyAliasSource
    replacement: ReplacementRule
    confidence: ComfyAliasConfidence
    family: ComfyAliasFamily | None = None

    def __post_init__(self) -> None:
        expected_id = f"comfy_alias:{self.source.pack}/{self.source.node_class}"
        if self.id != expected_id:
            raise ValueError(f"comfy alias id must be {expected_id!r}")
        if self.mapping_kind not in _MAPPING_KINDS:
            raise ValueError(f"unknown comfy alias mapping kind: {self.mapping_kind!r}")
        if not self.carrier:
            raise ValueError("comfy alias carrier must be non-empty")
        if self.replacement.from_type != self.source.node_type:
            raise ValueError("comfy alias replacement.from must equal source.nodeType")
        if self.carrier not in {case.to for case in self.replacement.cases}:
            raise ValueError("comfy alias carrier must be one of replacement.cases[].to")
        if (self.mapping_kind == "family") != (self.family is not None):
            raise ValueError("family mappings require family data and op mappings forbid it")


@dataclass(frozen=True)
class ComfyAliasSourceSchema:
    """One import-only source schema."""

    schema: NodeSchema
    wire_version: int

    def __post_init__(self) -> None:
        if self.wire_version != SCHEMA_WIRE_VERSION:
            raise ValueError(f"unsupported schemaVersion: {self.wire_version!r}")
        wire = schema_to_wire(self.schema)
        try:
            decoded = schema_from_wire(cast("dict[str, Any]", wire))
        except ValueError as exc:
            raise ValueError(
                f"comfy alias source schema must use a decodable wire version: {exc}"
            ) from None
        if decoded != self.schema:
            raise ValueError("comfy alias source schema must round-trip without losing fields")


@dataclass(frozen=True)
class ComfyAliasRegistry:
    source_schemas: tuple[ComfyAliasSourceSchema, ...]
    records: tuple[ComfyAliasRecord, ...]
    format: str = COMFY_ALIAS_FORMAT

    def __post_init__(self) -> None:
        if self.format != COMFY_ALIAS_FORMAT:
            raise ValueError(f"unsupported comfy alias format: {self.format!r}")
        node_types = [snapshot.schema.node_type for snapshot in self.source_schemas]
        if len(set(node_types)) != len(node_types):
            raise ValueError("comfy alias registry has duplicate source schema nodeType values")
        if any(snapshot.schema.replacements for snapshot in self.source_schemas):
            raise ValueError("comfy alias source schemas must not carry replacements")
        record_ids = [record.id for record in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("comfy alias registry has duplicate record ids")
        source_keys = [(record.source.pack, record.source.node_class) for record in self.records]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("comfy alias registry has duplicate source pack/nodeClass keys")
        used_types = [record.source.node_type for record in self.records]
        if len(set(used_types)) != len(used_types):
            raise ValueError("comfy alias registry has duplicate source nodeType records")
        unknown = sorted(set(used_types) - set(node_types))
        if unknown:
            raise ValueError(f"comfy alias records lack source schemas: {', '.join(unknown)}")
        unused = sorted(set(node_types) - set(used_types))
        if unused:
            raise ValueError(f"comfy alias registry has unused source schemas: {', '.join(unused)}")


def _expect_object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object with string keys")
    raw = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where} must be an object with string keys")
    return cast("dict[str, Any]", raw)


def _expect_list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return cast("list[Any]", value)


def _expect_string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def _expect_fields(
    value: object, where: str, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    obj = _expect_object(value, where)
    optional = optional or set()
    missing = sorted(required - set(obj))
    unknown = sorted(set(obj) - required - optional)
    if missing:
        raise ValueError(f"{where} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where} has unknown fields: {', '.join(unknown)}")
    return obj


def comfy_alias_source_from_wire(value: object, where: str) -> ComfyAliasSource:
    obj = _expect_fields(value, where, {"pack", "nodeClass", "nodeType", "revision"})
    return ComfyAliasSource(
        pack=_expect_string(obj["pack"], f"{where}.pack"),
        node_class=_expect_string(obj["nodeClass"], f"{where}.nodeClass"),
        node_type=_expect_string(obj["nodeType"], f"{where}.nodeType"),
        revision=_expect_string(obj["revision"], f"{where}.revision"),
    )


def comfy_alias_family_from_wire(value: object, where: str) -> ComfyAliasFamily:
    obj = _expect_fields(value, where, {"id"}, {"provider"})
    return ComfyAliasFamily(
        id=_expect_string(obj["id"], f"{where}.id"),
        provider=_expect_string(obj.get("provider", ""), f"{where}.provider"),
    )


def _tolerance_from_wire(value: object, where: str) -> ComfyAliasTolerance:
    obj = _expect_fields(value, where, {"metric", "operator", "value"})
    number = obj["value"]
    if type(number) not in (int, float):
        raise ValueError(f"{where}.value must be a finite number")
    return ComfyAliasTolerance(
        metric=_expect_string(obj["metric"], f"{where}.metric"),
        operator=cast("ToleranceOperator", _expect_string(obj["operator"], f"{where}.operator")),
        value=cast("float", number),
    )


def comfy_alias_confidence_from_wire(value: object, where: str) -> ComfyAliasConfidence:
    obj = _expect_fields(value, where, {"tier", "evidence"}, {"tolerances"})
    tier = _expect_string(obj["tier"], f"{where}.tier")
    evidence = tuple(
        _expect_string(item, f"{where}.evidence[{index}]")
        for index, item in enumerate(_expect_list(obj["evidence"], f"{where}.evidence"))
    )
    has_tolerances = "tolerances" in obj
    tolerances = tuple(
        _tolerance_from_wire(item, f"{where}.tolerances[{index}]")
        for index, item in enumerate(_expect_list(obj.get("tolerances", []), f"{where}.tolerances"))
    )
    if tier == "exact" and has_tolerances:
        raise ValueError(f"{where}.tolerances is forbidden for exact confidence")
    return ComfyAliasConfidence(
        tier=cast("ComfyAliasConfidenceTier", tier),
        evidence=evidence,
        tolerances=tolerances,
    )


def _record_from_wire(value: object, where: str) -> ComfyAliasRecord:
    obj = _expect_fields(
        value,
        where,
        {"id", "mappingKind", "carrier", "source", "replacement", "confidence"},
        {"family"},
    )
    mapping_kind = _expect_string(obj["mappingKind"], f"{where}.mappingKind")
    replacement_wire = _expect_object(obj["replacement"], f"{where}.replacement")
    try:
        replacement = rule_from_wire(replacement_wire)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}.replacement is invalid: {exc}") from None
    if rule_to_wire(replacement) != replacement_wire:
        raise ValueError(f"{where}.replacement is not canonical ReplacementRule wire data")
    family = (
        comfy_alias_family_from_wire(obj["family"], f"{where}.family") if "family" in obj else None
    )
    return ComfyAliasRecord(
        id=_expect_string(obj["id"], f"{where}.id"),
        mapping_kind=cast("ComfyAliasMappingKind", mapping_kind),
        carrier=_expect_string(obj["carrier"], f"{where}.carrier"),
        source=comfy_alias_source_from_wire(obj["source"], f"{where}.source"),
        replacement=replacement,
        confidence=comfy_alias_confidence_from_wire(obj["confidence"], f"{where}.confidence"),
        family=family,
    )


def comfy_alias_registry_from_wire(value: object) -> ComfyAliasRegistry:
    obj = _expect_fields(value, "comfy alias registry", {"format", "sourceSchemas", "records"})
    schemas: list[ComfyAliasSourceSchema] = []
    for index, item in enumerate(
        _expect_list(obj["sourceSchemas"], "comfy alias registry.sourceSchemas")
    ):
        where = f"comfy alias registry.sourceSchemas[{index}]"
        schema_wire = _expect_object(item, where)
        try:
            schema = schema_from_wire(schema_wire)
            canonical = schema_to_wire(schema)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{where} is invalid: {exc}") from None
        if canonical != schema_wire:
            raise ValueError(f"{where} is not canonical schema_to_wire data")
        schemas.append(
            ComfyAliasSourceSchema(
                schema=schema,
                wire_version=cast("int", schema_wire["schemaVersion"]),
            )
        )
    records = tuple(
        _record_from_wire(item, f"comfy alias registry.records[{index}]")
        for index, item in enumerate(_expect_list(obj["records"], "comfy alias registry.records"))
    )
    registry = ComfyAliasRegistry(
        format=_expect_string(obj["format"], "comfy alias registry.format"),
        source_schemas=tuple(schemas),
        records=records,
    )
    if comfy_alias_registry_to_wire(registry) != obj:
        raise ValueError("comfy alias registry is not canonical wire data")
    return registry


def comfy_alias_source_to_wire(source: ComfyAliasSource) -> dict[str, object]:
    return {
        "pack": source.pack,
        "nodeClass": source.node_class,
        "nodeType": source.node_type,
        "revision": source.revision,
    }


def comfy_alias_family_to_wire(family: ComfyAliasFamily) -> dict[str, object]:
    wire: dict[str, object] = {"id": family.id}
    if family.provider:
        wire["provider"] = family.provider
    return wire


def comfy_alias_confidence_to_wire(confidence: ComfyAliasConfidence) -> dict[str, object]:
    wire: dict[str, object] = {
        "tier": confidence.tier,
        "evidence": list(confidence.evidence),
    }
    if confidence.tolerances:
        wire["tolerances"] = [
            {
                "metric": tolerance.metric,
                "operator": tolerance.operator,
                "value": tolerance.value,
            }
            for tolerance in confidence.tolerances
        ]
    return wire


def _record_to_wire(record: ComfyAliasRecord) -> dict[str, object]:
    wire: dict[str, object] = {
        "id": record.id,
        "mappingKind": record.mapping_kind,
        "carrier": record.carrier,
        "source": comfy_alias_source_to_wire(record.source),
        "replacement": rule_to_wire(record.replacement),
        "confidence": comfy_alias_confidence_to_wire(record.confidence),
    }
    if record.family is not None:
        wire["family"] = comfy_alias_family_to_wire(record.family)
    return wire


def comfy_alias_registry_to_wire(
    registry: ComfyAliasRegistry,
    *,
    schemas: Mapping[str, NodeSchema] | None = None,
) -> dict[str, object]:
    records = registry.records
    used_source_types = frozenset(record.source.node_type for record in records)
    return {
        "format": registry.format,
        "sourceSchemas": [
            schema_to_wire(snapshot.schema)
            for snapshot in registry.source_schemas
            if snapshot.schema.node_type in used_source_types
        ],
        "records": [_record_to_wire(record) for record in records],
    }


def comfy_alias_registry_problems(
    registry: ComfyAliasRegistry,
    schemas: Mapping[str, NodeSchema],
    *,
    ignore_unknown_carriers: bool = False,
) -> tuple[str, ...]:
    """Validate native carriers and shared replacement references.

    ``ignore_unknown_carriers`` skips records whose carrier is absent from
    ``schemas`` instead of reporting them, for callers that validate a
    partially published surface and re-validate once the carrier publishes.
    """
    problems: list[str] = []
    augmented = dict(schemas)
    for snapshot in registry.source_schemas:
        source = snapshot.schema
        if source.node_type in augmented:
            problems.append(f"source schema {source.node_type!r} collides with a native schema")
        else:
            augmented[source.node_type] = source

    records_by_carrier: dict[str, list[ComfyAliasRecord]] = {}
    for record in registry.records:
        carrier = schemas.get(record.carrier)
        if carrier is None:
            if not ignore_unknown_carriers:
                problems.append(f"record {record.id!r} names unknown carrier {record.carrier!r}")
            continue
        records_by_carrier.setdefault(record.carrier, []).append(record)

    alias_pairs: set[tuple[str, str]] = set()
    for carrier_type, records in records_by_carrier.items():
        carrier = augmented[carrier_type]
        existing_from = {rule.from_type for rule in carrier.replacements}
        for record in records:
            if record.source.node_type in existing_from:
                problems.append(
                    f"record {record.id!r} duplicates a native replacement on {carrier_type!r}"
                )
            alias_pairs.add((carrier_type, record.source.node_type))
        augmented[carrier_type] = replace(
            carrier,
            replacements=carrier.replacements + tuple(record.replacement for record in records),
        )

    for problem in validate_replacement_references(augmented):
        if (problem.carrier, problem.from_type) in alias_pairs:
            problems.append(problem.message)
    return tuple(problems)


def comfy_alias_collision_problems(
    registries: Mapping[str, ComfyAliasRegistry],
) -> tuple[str, ...]:
    """Find ambiguous alias identities across a complete pack table."""
    problems: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_sources: dict[tuple[str, str], str] = {}
    seen_types: dict[str, str] = {}
    for pack_id in sorted(registries):
        registry = registries[pack_id]
        for record in registry.records:
            owner = seen_ids.setdefault(record.id, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy alias "
                    f"record id {record.id!r}"
                )
            source_key = (record.source.pack, record.source.node_class)
            owner = seen_sources.setdefault(source_key, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy alias "
                    f"source key {source_key!r}"
                )
            owner = seen_types.setdefault(record.source.node_type, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy alias "
                    f"source nodeType {record.source.node_type!r}"
                )
    return tuple(problems)
