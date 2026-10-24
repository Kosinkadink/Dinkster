"""Return-unit documents shared by worker transport and persistent caches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from dinkster_values import Value

from .result_algebra import (
    BlockedOutput,
    CurrentNodeRef,
    CurrentOutputRef,
    DirectReturn,
    ExpandedReturn,
    ExternalNodeRef,
    ExternalOutputRef,
    InputBinding,
    JsonLiteral,
    JsonObject,
    LiteralInput,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    NodeMetadataRef,
    OutputBinding,
    PresentOutput,
    ReturnUnit,
)


def _ref_to_document(
    value: CurrentOutputRef | LocalOutputRef | ExternalOutputRef,
) -> dict[str, object]:
    if isinstance(value, CurrentOutputRef):
        return {"kind": "outputRef", "scope": "current", "outputId": value.output_id}
    if isinstance(value, ExternalOutputRef):
        return {
            "kind": "outputRef",
            "scope": "external",
            "nodeId": value.node_id,
            "outputId": value.output_id,
        }
    return {
        "kind": "outputRef",
        "scope": "local",
        "localNodeId": value.local_node_id,
        "outputId": value.output_id,
    }


def _metadata_to_document(value: NodeMetadataRef | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, CurrentNodeRef):
        return {"kind": "nodeRef", "scope": "current"}
    if isinstance(value, ExternalNodeRef):
        return {"kind": "nodeRef", "scope": "external", "nodeId": value.node_id}
    return {"kind": "nodeRef", "scope": "local", "localNodeId": value.local_node_id}


def _literal_to_json(value: JsonLiteral) -> object:
    if isinstance(value, JsonObject):
        return {key: _literal_to_json(item) for key, item in value.fields}
    if isinstance(value, tuple):
        return [_literal_to_json(item) for item in value]
    return value


def return_unit_to_document(
    unit: ReturnUnit, encode_present: Callable[[Value], int]
) -> dict[str, object]:
    def binding(value: OutputBinding) -> dict[str, object]:
        if isinstance(value, PresentOutput):
            return {"kind": "present", "valueIndex": encode_present(value.value)}
        if isinstance(value, BlockedOutput):
            return {"kind": "blocked", "message": value.message}
        return _ref_to_document(value)

    def input_value(value: InputBinding) -> dict[str, object]:
        if isinstance(value, LiteralInput):
            return {"kind": "literal", "value": _literal_to_json(value.value)}
        return _ref_to_document(value)

    result: dict[str, object] = {
        "kind": "direct",
        "bindings": [{"id": key, "value": binding(value)} for key, value in unit.bindings],
    }
    if isinstance(unit, ExpandedReturn):
        result["kind"] = "expanded"
        result["expansion"] = {
            "nodes": [
                {
                    "display": _metadata_to_document(node.display),
                    "inputs": [
                        {"id": key, "value": input_value(value)} for key, value in node.inputs
                    ],
                    "localId": node.local_id,
                    "nodeType": node.node_type,
                    "parent": _metadata_to_document(node.parent),
                }
                for node in unit.expansion.nodes
            ]
        }
    return result


def _exact(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(cast("dict[object, object]", value)) != fields:
        raise ValueError("unexpected fields")
    return cast("dict[str, Any]", value)


def _ref_from_document(raw: object) -> CurrentOutputRef | LocalOutputRef | ExternalOutputRef:
    if not isinstance(raw, dict):
        raise ValueError("reference must be an object")
    item = cast("dict[str, Any]", raw)
    if item.get("kind") != "outputRef":
        raise ValueError("unknown reference")
    if item.get("scope") == "current":
        item = _exact(item, {"kind", "scope", "outputId"})
        return CurrentOutputRef(item["outputId"])
    if item.get("scope") == "local":
        item = _exact(item, {"kind", "scope", "localNodeId", "outputId"})
        return LocalOutputRef(item["localNodeId"], item["outputId"])
    if item.get("scope") == "external":
        item = _exact(item, {"kind", "scope", "nodeId", "outputId"})
        return ExternalOutputRef(item["nodeId"], item["outputId"])
    raise ValueError("unknown reference")


def _metadata_from_document(raw: object) -> NodeMetadataRef | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("metadata reference must be an object")
    item = cast("dict[str, Any]", raw)
    if item.get("kind") != "nodeRef":
        raise ValueError("unknown metadata reference")
    if item.get("scope") == "current":
        _exact(item, {"kind", "scope"})
        return CurrentNodeRef()
    if item.get("scope") == "external":
        item = _exact(item, {"kind", "scope", "nodeId"})
        return ExternalNodeRef(item["nodeId"])
    item = _exact(item, {"kind", "scope", "localNodeId"})
    if item["scope"] != "local":
        raise ValueError("unknown metadata scope")
    return LocalNodeRef(item["localNodeId"])


def _json_to_literal(value: object) -> JsonLiteral:
    if isinstance(value, list):
        return tuple(_json_to_literal(item) for item in cast("list[object]", value))
    if isinstance(value, dict):
        fields = cast("dict[object, object]", value)
        if not all(isinstance(key, str) for key in fields):
            raise ValueError("literal object keys must be strings")
        return JsonObject(
            tuple((cast(str, key), _json_to_literal(item)) for key, item in sorted(fields.items()))
        )
    return cast("JsonLiteral", value)


def return_unit_from_document(
    raw: object, decode_present: Callable[[object, str], Value]
) -> ReturnUnit:
    if not isinstance(raw, dict):
        raise ValueError("return unit must be an object")
    kind = cast("dict[str, Any]", raw).get("kind")
    expected = {"kind", "bindings"} if kind == "direct" else {"kind", "bindings", "expansion"}
    unit = _exact(cast(object, raw), expected)
    if type(unit["bindings"]) is not list:
        raise ValueError("bindings must be an array")
    bindings: list[tuple[str, OutputBinding]] = []
    for raw_binding in cast("list[object]", unit["bindings"]):
        entry = _exact(raw_binding, {"id", "value"})
        value = entry["value"]
        if not isinstance(value, dict):
            raise ValueError("binding must be an object")
        value = cast("dict[str, Any]", value)
        if value.get("kind") == "present":
            value = _exact(value, {"kind", "valueIndex"})
            binding: OutputBinding = PresentOutput(decode_present(value["valueIndex"], entry["id"]))
        elif value.get("kind") == "blocked":
            binding = BlockedOutput(_exact(value, {"kind", "message"})["message"])
        else:
            binding = _ref_from_document(value)
        bindings.append((entry["id"], binding))
    if kind == "direct":
        return DirectReturn(tuple(bindings))
    if kind != "expanded":
        raise ValueError("unknown return kind")
    expansion = _exact(unit["expansion"], {"nodes"})
    if type(expansion["nodes"]) is not list:
        raise ValueError("nodes must be an array")
    nodes: list[LocalNode] = []
    for raw_node in cast("list[object]", expansion["nodes"]):
        node = _exact(raw_node, {"display", "inputs", "localId", "nodeType", "parent"})
        if type(node["inputs"]) is not list:
            raise ValueError("inputs must be an array")
        inputs: list[tuple[str, InputBinding]] = []
        for raw_input in cast("list[object]", node["inputs"]):
            entry = _exact(raw_input, {"id", "value"})
            value: object = entry["value"]
            if isinstance(value, dict) and cast("dict[str, Any]", value).get("kind") == "literal":
                literal = _exact(cast(object, value), {"kind", "value"})
                decoded: InputBinding = LiteralInput(_json_to_literal(literal["value"]))
            else:
                decoded = _ref_from_document(cast(object, value))
            inputs.append((entry["id"], decoded))
        nodes.append(
            LocalNode(
                node["localId"],
                node["nodeType"],
                tuple(inputs),
                _metadata_from_document(node["parent"]),
                _metadata_from_document(node["display"]),
            )
        )
    return ExpandedReturn(LocalExpansion(tuple(nodes)), tuple(bindings))
