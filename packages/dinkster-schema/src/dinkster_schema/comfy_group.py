"""Declarative ComfyUI node-group translation patterns.

Group records describe an exact foreign subgraph and collapse it to one
import-only source schema. The shared ``ReplacementRule`` then translates that
source shape to native nodes. Pattern schemas and group schemas are data only;
they never join the executable node surface.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from .comfy_alias import (
    COMFY_CORE_REVISIONS,
    ComfyAliasConfidence,
    ComfyAliasFamily,
    ComfyAliasSource,
    comfy_alias_confidence_from_wire,
    comfy_alias_confidence_to_wire,
    comfy_alias_family_from_wire,
    comfy_alias_family_to_wire,
    comfy_alias_source_from_wire,
    comfy_alias_source_to_wire,
)
from .model import InputSpec, NodeSchema, OutputSpec
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

COMFY_GROUP_FORMAT = "dinkster-comfy-group/1"
COMFY_GROUP_MAX_NODES = 16
COMFY_GROUP_MAX_EDGES = 64

ComfyGroupMappingKind = Literal["op", "family"]
ComfyGroupMode = Literal["active", "muted", "bypassed"]

_MAPPING_KINDS = frozenset({"op", "family"})
_MODES = frozenset({"active", "muted", "bypassed"})
_STRUCTURAL_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ComfyGroupSource:
    pack: str
    name: str
    revision: str

    def __post_init__(self) -> None:
        if validate_name(self.pack) is not None:
            raise ValueError(f"invalid comfy group source pack: {self.pack!r}")
        if validate_name(self.name) is not None:
            raise ValueError(f"invalid comfy group source name: {self.name!r}")
        if not self.revision:
            raise ValueError("comfy group source revision must be non-empty")
        if self.pack == "comfy-core" and self.revision not in COMFY_CORE_REVISIONS:
            expected = ", ".join(sorted(COMFY_CORE_REVISIONS))
            raise ValueError(f"comfy-core groups must use a maintained revision: {expected}")


@dataclass(frozen=True)
class ComfyGroupNode:
    source: ComfyAliasSource
    mode: ComfyGroupMode

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"unknown comfy group node mode: {self.mode!r}")


@dataclass(frozen=True)
class ComfyGroupEdge:
    from_address: str
    to: str


@dataclass(frozen=True)
class ComfyGroupPattern:
    group_type: str
    anchor: str
    nodes: tuple[tuple[str, ComfyGroupNode], ...]
    edges: tuple[ComfyGroupEdge, ...]
    inputs: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, object], ...]
    outputs: tuple[tuple[str, str], ...]
    disconnected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if validate_name(self.group_type) is not None:
            raise ValueError(f"invalid comfy group source type: {self.group_type!r}")
        if not 2 <= len(self.nodes) <= COMFY_GROUP_MAX_NODES:
            raise ValueError(f"comfy group pattern requires 2 to {COMFY_GROUP_MAX_NODES} nodes")
        local_ids = [local_id for local_id, _ in self.nodes]
        if len(set(local_ids)) != len(local_ids):
            raise ValueError("comfy group pattern has duplicate local node ids")
        if any(_STRUCTURAL_ID.fullmatch(local_id) is None for local_id in local_ids):
            raise ValueError("comfy group pattern has an invalid local node id")
        if self.anchor not in set(local_ids):
            raise ValueError("comfy group pattern anchor must name a local node")
        if len(self.edges) > COMFY_GROUP_MAX_EDGES:
            raise ValueError(f"comfy group pattern exceeds the {COMFY_GROUP_MAX_EDGES}-edge cap")
        if len({node.mode for _, node in self.nodes}) != 1:
            raise ValueError("comfy group pattern nodes must declare one uniform mode")
        for label, entries in (
            ("inputs", self.inputs),
            ("parameters", self.parameters),
            ("constants", self.constants),
            ("outputs", self.outputs),
        ):
            keys = [key for key, _ in entries]
            if len(set(keys)) != len(keys):
                raise ValueError(f"comfy group pattern has duplicate {label} keys")
        for key, value in self.constants:
            _require_json(value, f"comfy group constant {key!r}")
        if len(set(self.disconnected)) != len(self.disconnected):
            raise ValueError("comfy group pattern has duplicate disconnected inputs")


@dataclass(frozen=True)
class ComfyGroupSourceSchema:
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
                f"comfy group schema must use a decodable wire version: {exc}"
            ) from None
        if decoded != self.schema:
            raise ValueError("comfy group schema must round-trip without losing fields")


@dataclass(frozen=True)
class ComfyGroupRecord:
    id: str
    mapping_kind: ComfyGroupMappingKind
    carrier: str
    source: ComfyGroupSource
    pattern: ComfyGroupPattern
    replacement: ReplacementRule
    confidence: ComfyAliasConfidence
    family: ComfyAliasFamily | None = None

    def __post_init__(self) -> None:
        expected_id = f"comfy_group:{self.source.pack}/{self.source.name}"
        if self.id != expected_id:
            raise ValueError(f"comfy group id must be {expected_id!r}")
        expected_type = f"comfy-group.{self.source.pack}.{self.source.name}"
        if self.pattern.group_type != expected_type:
            raise ValueError(f"comfy group pattern groupType must be {expected_type!r}")
        if self.mapping_kind not in _MAPPING_KINDS:
            raise ValueError(f"unknown comfy group mapping kind: {self.mapping_kind!r}")
        if not self.carrier:
            raise ValueError("comfy group carrier must be non-empty")
        if self.replacement.from_type != self.pattern.group_type:
            raise ValueError("comfy group replacement.from must equal pattern.groupType")
        if self.carrier not in {case.to for case in self.replacement.cases}:
            raise ValueError("comfy group carrier must be one of replacement.cases[].to")
        if self.confidence.tier != "grouped":
            raise ValueError("comfy group confidence tier must be 'grouped'")
        if (self.mapping_kind == "family") != (self.family is not None):
            raise ValueError("family mappings require family data and op mappings forbid it")


@dataclass(frozen=True)
class ComfyGroupRegistry:
    source_schemas: tuple[ComfyGroupSourceSchema, ...]
    group_schemas: tuple[ComfyGroupSourceSchema, ...]
    records: tuple[ComfyGroupRecord, ...]
    format: str = COMFY_GROUP_FORMAT

    def __post_init__(self) -> None:
        if self.format != COMFY_GROUP_FORMAT:
            raise ValueError(f"unsupported comfy group format: {self.format!r}")
        source_types = [snapshot.schema.node_type for snapshot in self.source_schemas]
        group_types = [snapshot.schema.node_type for snapshot in self.group_schemas]
        if len(set(source_types)) != len(source_types):
            raise ValueError("comfy group registry has duplicate source schema nodeType values")
        if len(set(group_types)) != len(group_types):
            raise ValueError("comfy group registry has duplicate group schema nodeType values")
        if set(source_types) & set(group_types):
            raise ValueError("comfy group source and group schema nodeType values overlap")
        if any(
            snapshot.schema.replacements for snapshot in (*self.source_schemas, *self.group_schemas)
        ):
            raise ValueError("comfy group schemas must not carry replacements")
        record_ids = [record.id for record in self.records]
        source_keys = [(record.source.pack, record.source.name) for record in self.records]
        record_group_types = [record.pattern.group_type for record in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("comfy group registry has duplicate record ids")
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("comfy group registry has duplicate source pack/name keys")
        if len(set(record_group_types)) != len(record_group_types):
            raise ValueError("comfy group registry has duplicate groupType records")
        used_source_types = {
            node.source.node_type for record in self.records for _, node in record.pattern.nodes
        }
        missing_sources = sorted(used_source_types - set(source_types))
        if missing_sources:
            raise ValueError(
                f"comfy group records lack source schemas: {', '.join(missing_sources)}"
            )
        unused_sources = sorted(set(source_types) - used_source_types)
        if unused_sources:
            raise ValueError(
                f"comfy group registry has unused source schemas: {', '.join(unused_sources)}"
            )
        missing_groups = sorted(set(record_group_types) - set(group_types))
        if missing_groups:
            raise ValueError(f"comfy group records lack group schemas: {', '.join(missing_groups)}")
        unused_groups = sorted(set(group_types) - set(record_group_types))
        if unused_groups:
            raise ValueError(
                f"comfy group registry has unused group schemas: {', '.join(unused_groups)}"
            )


def _require_json(value: object, where: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be JSON-serializable data") from None


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


def _schema_from_wire(value: object, where: str) -> ComfyGroupSourceSchema:
    schema_wire = _expect_object(value, where)
    try:
        schema = schema_from_wire(schema_wire)
        wire_version = cast("int", schema_wire.get("schemaVersion"))
        canonical = schema_to_wire(schema)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{where} is invalid: {exc}") from None
    if canonical != schema_wire:
        raise ValueError(f"{where} is not canonical schema_to_wire data")
    return ComfyGroupSourceSchema(schema=schema, wire_version=wire_version)


def _group_source_from_wire(value: object, where: str) -> ComfyGroupSource:
    obj = _expect_fields(value, where, {"pack", "name", "revision"})
    return ComfyGroupSource(
        pack=_expect_string(obj["pack"], f"{where}.pack"),
        name=_expect_string(obj["name"], f"{where}.name"),
        revision=_expect_string(obj["revision"], f"{where}.revision"),
    )


def _node_from_wire(value: object, where: str) -> ComfyGroupNode:
    obj = _expect_fields(value, where, {"source", "mode"})
    return ComfyGroupNode(
        source=comfy_alias_source_from_wire(obj["source"], f"{where}.source"),
        mode=cast("ComfyGroupMode", _expect_string(obj["mode"], f"{where}.mode")),
    )


def _edge_from_wire(value: object, where: str) -> ComfyGroupEdge:
    obj = _expect_fields(value, where, {"from", "to"})
    return ComfyGroupEdge(
        from_address=_expect_string(obj["from"], f"{where}.from"),
        to=_expect_string(obj["to"], f"{where}.to"),
    )


def _string_map(value: object, where: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, _expect_string(item, f"{where}.{key}"))
        for key, item in _expect_object(value, where).items()
    )


def _pattern_from_wire(value: object, where: str) -> ComfyGroupPattern:
    obj = _expect_fields(
        value,
        where,
        {
            "groupType",
            "anchor",
            "nodes",
            "edges",
            "boundary",
            "parameters",
            "constants",
        },
        {"disconnected"},
    )
    nodes = tuple(
        (local_id, _node_from_wire(node, f"{where}.nodes.{local_id}"))
        for local_id, node in _expect_object(obj["nodes"], f"{where}.nodes").items()
    )
    edges = tuple(
        _edge_from_wire(edge, f"{where}.edges[{index}]")
        for index, edge in enumerate(_expect_list(obj["edges"], f"{where}.edges"))
    )
    boundary = _expect_fields(obj["boundary"], f"{where}.boundary", {"inputs", "outputs"})
    constants = tuple(_expect_object(obj["constants"], f"{where}.constants").items())
    return ComfyGroupPattern(
        group_type=_expect_string(obj["groupType"], f"{where}.groupType"),
        anchor=_expect_string(obj["anchor"], f"{where}.anchor"),
        nodes=nodes,
        edges=edges,
        inputs=_string_map(boundary["inputs"], f"{where}.boundary.inputs"),
        parameters=_string_map(obj["parameters"], f"{where}.parameters"),
        constants=constants,
        outputs=_string_map(boundary["outputs"], f"{where}.boundary.outputs"),
        disconnected=tuple(
            _expect_string(item, f"{where}.disconnected[{index}]")
            for index, item in enumerate(
                _expect_list(obj.get("disconnected", []), f"{where}.disconnected")
            )
        ),
    )


def _record_from_wire(value: object, where: str) -> ComfyGroupRecord:
    obj = _expect_fields(
        value,
        where,
        {"id", "mappingKind", "carrier", "source", "pattern", "replacement", "confidence"},
        {"family"},
    )
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
    return ComfyGroupRecord(
        id=_expect_string(obj["id"], f"{where}.id"),
        mapping_kind=cast(
            "ComfyGroupMappingKind",
            _expect_string(obj["mappingKind"], f"{where}.mappingKind"),
        ),
        carrier=_expect_string(obj["carrier"], f"{where}.carrier"),
        source=_group_source_from_wire(obj["source"], f"{where}.source"),
        pattern=_pattern_from_wire(obj["pattern"], f"{where}.pattern"),
        replacement=replacement,
        confidence=comfy_alias_confidence_from_wire(obj["confidence"], f"{where}.confidence"),
        family=family,
    )


def comfy_group_registry_from_wire(value: object) -> ComfyGroupRegistry:
    obj = _expect_fields(
        value,
        "comfy group registry",
        {"format", "sourceSchemas", "groupSchemas", "records"},
    )
    record_wires = _expect_list(obj["records"], "comfy group registry.records")
    source_schemas = tuple(
        _schema_from_wire(item, f"comfy group registry.sourceSchemas[{index}]")
        for index, item in enumerate(
            _expect_list(obj["sourceSchemas"], "comfy group registry.sourceSchemas")
        )
    )
    group_schemas = tuple(
        _schema_from_wire(item, f"comfy group registry.groupSchemas[{index}]")
        for index, item in enumerate(
            _expect_list(obj["groupSchemas"], "comfy group registry.groupSchemas")
        )
    )
    records = tuple(
        _record_from_wire(item, f"comfy group registry.records[{index}]")
        for index, item in enumerate(record_wires)
    )
    registry = ComfyGroupRegistry(
        format=_expect_string(obj["format"], "comfy group registry.format"),
        source_schemas=source_schemas,
        group_schemas=group_schemas,
        records=records,
    )
    normalized_records: list[dict[str, object]] = []
    for index, record in enumerate(record_wires):
        record_obj = _expect_object(record, f"comfy group registry.records[{index}]")
        pattern_obj = _expect_object(
            record_obj["pattern"], f"comfy group registry.records[{index}].pattern"
        )
        normalized_records.append({**record_obj, "pattern": {"disconnected": [], **pattern_obj}})
    normalized_obj = dict(obj)
    normalized_obj["records"] = normalized_records
    if comfy_group_registry_to_wire(registry) != normalized_obj:
        raise ValueError("comfy group registry is not canonical wire data")
    return registry


def _node_to_wire(node: ComfyGroupNode) -> dict[str, object]:
    return {"source": comfy_alias_source_to_wire(node.source), "mode": node.mode}


def _pattern_to_wire(pattern: ComfyGroupPattern) -> dict[str, object]:
    return {
        "groupType": pattern.group_type,
        "anchor": pattern.anchor,
        "nodes": {local_id: _node_to_wire(node) for local_id, node in pattern.nodes},
        "edges": [{"from": edge.from_address, "to": edge.to} for edge in pattern.edges],
        "boundary": {
            "inputs": dict(pattern.inputs),
            "outputs": dict(pattern.outputs),
        },
        "parameters": dict(pattern.parameters),
        "constants": dict(pattern.constants),
        "disconnected": list(pattern.disconnected),
    }


def _record_to_wire(record: ComfyGroupRecord) -> dict[str, object]:
    wire: dict[str, object] = {
        "id": record.id,
        "mappingKind": record.mapping_kind,
        "carrier": record.carrier,
        "source": {
            "pack": record.source.pack,
            "name": record.source.name,
            "revision": record.source.revision,
        },
        "pattern": _pattern_to_wire(record.pattern),
        "replacement": rule_to_wire(record.replacement),
        "confidence": comfy_alias_confidence_to_wire(record.confidence),
    }
    if record.family is not None:
        wire["family"] = comfy_alias_family_to_wire(record.family)
    return wire


def comfy_group_registry_to_wire(
    registry: ComfyGroupRegistry,
    *,
    schemas: Mapping[str, NodeSchema] | None = None,
) -> dict[str, object]:
    records = registry.records
    used_group_types = frozenset(record.pattern.group_type for record in records)
    used_source_types = frozenset(
        node.source.node_type for record in records for _local_id, node in record.pattern.nodes
    )
    return {
        "format": registry.format,
        "sourceSchemas": [
            schema_to_wire(snapshot.schema)
            for snapshot in registry.source_schemas
            if snapshot.schema.node_type in used_source_types
        ],
        "groupSchemas": [
            schema_to_wire(snapshot.schema)
            for snapshot in registry.group_schemas
            if snapshot.schema.node_type in used_group_types
        ],
        "records": [_record_to_wire(record) for record in records],
    }


def _address_parts(address: str, local_ids: frozenset[str], where: str) -> tuple[str, str]:
    parts = address.split(":")
    if (
        len(parts) != 2
        or _STRUCTURAL_ID.fullmatch(parts[0]) is None
        or _STRUCTURAL_ID.fullmatch(parts[1]) is None
        or parts[0] not in local_ids
    ):
        raise ValueError(f"{where} has invalid pattern address {address!r}")
    return parts[0], parts[1]


def _static_schema_ports(
    schema: NodeSchema, where: str
) -> tuple[dict[str, InputSpec], dict[str, OutputSpec]]:
    if not schema.is_static:
        raise ValueError(f"{where} must have a static interface")
    return (
        {item.id: item for item in schema.inputs},
        {item.id: item for item in schema.outputs},
    )


def comfy_group_pattern_problems(
    record: ComfyGroupRecord,
    source_schemas: Mapping[str, NodeSchema],
    group_schema: NodeSchema,
) -> tuple[str, ...]:
    """Validate one exact pattern against its foreign and collapsed schemas."""
    try:
        group_inputs, group_outputs = _static_schema_ports(group_schema, "group schema")
        nodes = dict(record.pattern.nodes)
        local_ids = frozenset(nodes)
        source_inputs: dict[str, InputSpec] = {}
        source_outputs: dict[str, OutputSpec] = {}
        for local_id, node in nodes.items():
            schema = source_schemas.get(node.source.node_type)
            if schema is None:
                raise ValueError(
                    f"pattern node {local_id!r} names unknown source schema "
                    f"{node.source.node_type!r}"
                )
            inputs, outputs = _static_schema_ports(schema, f"pattern node {local_id!r}")
            source_inputs.update((f"{local_id}:{port}", spec) for port, spec in inputs.items())
            source_outputs.update((f"{local_id}:{port}", spec) for port, spec in outputs.items())

        internal_targets: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()
        adjacent: dict[str, set[str]] = {local_id: set() for local_id in local_ids}
        for edge in record.pattern.edges:
            source_node, _ = _address_parts(edge.from_address, local_ids, "pattern edge.from")
            target_node, _ = _address_parts(edge.to, local_ids, "pattern edge.to")
            if edge.from_address not in source_outputs:
                raise ValueError(f"pattern edge source {edge.from_address!r} is not an output")
            if edge.to not in source_inputs:
                raise ValueError(f"pattern edge target {edge.to!r} is not an input")
            if source_outputs[edge.from_address].type != source_inputs[edge.to].type:
                raise ValueError(
                    f"pattern edge {edge.from_address!r} -> {edge.to!r} has mismatched types"
                )
            if edge.to in internal_targets:
                raise ValueError(f"pattern input {edge.to!r} has multiple internal feeders")
            edge_key = (edge.from_address, edge.to)
            if edge_key in seen_edges:
                raise ValueError(f"pattern has duplicate edge {edge_key!r}")
            seen_edges.add(edge_key)
            internal_targets.add(edge.to)
            adjacent[source_node].add(target_node)
            adjacent[target_node].add(source_node)

        connected = {record.pattern.anchor}
        pending = [record.pattern.anchor]
        while pending:
            for neighbor in adjacent[pending.pop()] - connected:
                connected.add(neighbor)
                pending.append(neighbor)
        if connected != set(local_ids):
            raise ValueError("comfy group pattern nodes must be connected by internal edges")

        role_addresses: dict[str, str] = {}
        group_input_keys: set[str] = set()
        for role, entries in (
            ("boundary input", record.pattern.inputs),
            ("parameter", record.pattern.parameters),
        ):
            for group_input, address in entries:
                _address_parts(address, local_ids, role)
                if group_input not in group_inputs:
                    raise ValueError(f"{role} {group_input!r} is not a group schema input")
                if group_input in group_input_keys:
                    raise ValueError(
                        f"group schema input {group_input!r} has multiple pattern mappings"
                    )
                if address not in source_inputs:
                    raise ValueError(f"{role} address {address!r} is not a source input")
                if address in role_addresses or address in internal_targets:
                    raise ValueError(f"source input {address!r} has multiple pattern roles")
                if source_inputs[address].type != group_inputs[group_input].type:
                    raise ValueError(
                        f"{role} {group_input!r} type differs from source input {address!r}"
                    )
                if role == "parameter" and (
                    source_inputs[address].widget is None
                    or group_inputs[group_input].widget is None
                ):
                    raise ValueError(f"parameter {group_input!r} must map widget inputs")
                role_addresses[address] = role
                group_input_keys.add(group_input)

        for address, value in record.pattern.constants:
            _address_parts(address, local_ids, "constant")
            if address not in source_inputs:
                raise ValueError(f"constant address {address!r} is not a source input")
            source_input = source_inputs[address]
            if source_input.widget is None and not (not source_input.required and value is None):
                raise ValueError(f"constant address {address!r} is not a widget input")
            if address in role_addresses or address in internal_targets:
                raise ValueError(f"source input {address!r} has multiple pattern roles")
            role_addresses[address] = "constant"

        for address in record.pattern.disconnected:
            _address_parts(address, local_ids, "disconnected input")
            if address not in source_inputs:
                raise ValueError(f"disconnected address {address!r} is not a source input")
            source_input = source_inputs[address]
            if source_input.required:
                raise ValueError(f"disconnected source input {address!r} must be optional")
            if source_input.widget is not None and not source_input.force_input:
                raise ValueError(
                    f"disconnected source input {address!r} must be a socket, not a widget"
                )
            if address in role_addresses or address in internal_targets:
                raise ValueError(f"source input {address!r} has multiple pattern roles")
            role_addresses[address] = "disconnected"

        classified_inputs = internal_targets | set(role_addresses)
        if classified_inputs != set(source_inputs):
            missing = sorted(set(source_inputs) - classified_inputs)
            extra = sorted(classified_inputs - set(source_inputs))
            detail = missing or extra
            raise ValueError(
                "every source input must have exactly one internal, boundary, parameter, "
                f"or constant role; offending inputs: {', '.join(detail)}"
            )
        if group_input_keys != set(group_inputs):
            missing = sorted(set(group_inputs) - group_input_keys)
            raise ValueError(f"group schema inputs lack pattern mappings: {', '.join(missing)}")

        group_output_keys: set[str] = set()
        source_boundary_outputs: set[str] = set()
        for group_output, address in record.pattern.outputs:
            _address_parts(address, local_ids, "boundary output")
            if group_output not in group_outputs:
                raise ValueError(f"boundary output {group_output!r} is not a group schema output")
            if address not in source_outputs:
                raise ValueError(f"boundary output address {address!r} is not a source output")
            if source_outputs[address].type != group_outputs[group_output].type:
                raise ValueError(
                    f"boundary output {group_output!r} type differs from source output {address!r}"
                )
            if address in source_boundary_outputs:
                raise ValueError(f"source output {address!r} maps to multiple group outputs")
            group_output_keys.add(group_output)
            source_boundary_outputs.add(address)
        if group_output_keys != set(group_outputs):
            missing = sorted(set(group_outputs) - group_output_keys)
            raise ValueError(f"group schema outputs lack pattern mappings: {', '.join(missing)}")
    except ValueError as exc:
        return (str(exc),)
    return ()


def comfy_group_registry_problems(
    registry: ComfyGroupRegistry,
    schemas: Mapping[str, NodeSchema],
    *,
    ignore_unknown_carriers: bool = False,
) -> tuple[str, ...]:
    """Validate pattern schemas, native carriers, and replacement references.

    ``ignore_unknown_carriers`` skips the unknown-carrier report for records
    whose carrier is absent from ``schemas``, for callers that validate a
    partially published surface and re-validate once the carrier publishes.
    Pattern problems still surface: they depend only on the registry's own
    schemas, not on carrier publication.
    """
    problems: list[str] = []
    source_schemas = {
        snapshot.schema.node_type: snapshot.schema for snapshot in registry.source_schemas
    }
    group_schemas = {
        snapshot.schema.node_type: snapshot.schema for snapshot in registry.group_schemas
    }
    for node_type in (*source_schemas, *group_schemas):
        if node_type in schemas:
            problems.append(f"comfy group schema {node_type!r} collides with a native schema")

    augmented = {**schemas, **group_schemas}
    records_by_carrier: dict[str, list[ComfyGroupRecord]] = {}
    for record in registry.records:
        group_schema = group_schemas.get(record.pattern.group_type)
        if group_schema is not None:
            for problem in comfy_group_pattern_problems(record, source_schemas, group_schema):
                problems.append(f"record {record.id!r}: {problem}")
        carrier = schemas.get(record.carrier)
        if carrier is None:
            if not ignore_unknown_carriers:
                problems.append(f"record {record.id!r} names unknown carrier {record.carrier!r}")
            continue
        records_by_carrier.setdefault(record.carrier, []).append(record)

    group_pairs: set[tuple[str, str]] = set()
    for carrier_type, records in records_by_carrier.items():
        carrier = augmented[carrier_type]
        existing_from = {rule.from_type for rule in carrier.replacements}
        for record in records:
            if record.pattern.group_type in existing_from:
                problems.append(
                    f"record {record.id!r} duplicates a native replacement on {carrier_type!r}"
                )
            group_pairs.add((carrier_type, record.pattern.group_type))
        augmented[carrier_type] = replace(
            carrier,
            replacements=carrier.replacements + tuple(record.replacement for record in records),
        )

    for problem in validate_replacement_references(augmented):
        if (problem.carrier, problem.from_type) in group_pairs:
            problems.append(problem.message)
    return tuple(problems)


def comfy_group_collision_problems(
    registries: Mapping[str, ComfyGroupRegistry],
) -> tuple[str, ...]:
    """Find ambiguous group identities across a complete pack table."""
    problems: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_sources: dict[tuple[str, str], str] = {}
    seen_types: dict[str, str] = {}
    for pack_id in sorted(registries):
        for record in registries[pack_id].records:
            owner = seen_ids.setdefault(record.id, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy group "
                    f"record id {record.id!r}"
                )
            source_key = (record.source.pack, record.source.name)
            owner = seen_sources.setdefault(source_key, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy group "
                    f"source key {source_key!r}"
                )
            owner = seen_types.setdefault(record.pattern.group_type, pack_id)
            if owner != pack_id:
                problems.append(
                    f"packs {owner!r} and {pack_id!r} collide on comfy group "
                    f"groupType {record.pattern.group_type!r}"
                )
    return tuple(problems)
