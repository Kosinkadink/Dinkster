"""Graph wire format: the JSON shape of a graph document.

The document a frontend stores/submits is exactly this shape - node ids
mapping to {nodeType, inputs, outputMembers}. Links are distinguished from
literal values by the reserved ``$link`` marker::

    {"$link": {"node": "loader", "output": "model"}}

A literal may carry an explicit runtime type id via the reserved
``$typed`` marker (joint contract with the frontend, 2026-07-25: the
widget-tap lowering case, where a value is inlined into a
wildcard/union/variable input the engine could not otherwise wrap)::

    {"$typed": {"type": "core.int", "value": 7}}

A core.int literal outside the JSON-double-safe range uses the reserved
``$int`` marker so JavaScript clients can submit it without rounding::

    {"$int": "18446744073709551615"}

``$link``, ``$typed``, and ``$int`` are reserved: a literal dict input
carrying any of these keys cannot be expressed (decode raises rather than guessing).
Everything else passes through untouched - literal identity belongs to
the value boundary, not the graph model.

Wire values are typed ``Any`` deliberately: this is the untrusted-JSON
boundary, and every shape assumption is checked at runtime before use.
"""

from __future__ import annotations

import re
from typing import Any, Literal, cast

from dinkster_schema.wire import (
    SCHEMA_WIRE_SERVE_VERSIONS,
    type_expr_from_wire,
    type_expr_to_wire,
)

from .model import (
    NODE_ID_FORBIDDEN_CHARS,
    AnyNode,
    BindingMode,
    Graph,
    GraphNode,
    Link,
    RegionKind,
    RegionNode,
    RegionOutput,
    TypedLiteral,
)

LINK_KEY = "$link"
TYPED_KEY = "$typed"
INT_KEY = "$int"

RESERVED_INPUT_KEYS: frozenset[str] = frozenset({LINK_KEY, TYPED_KEY, INT_KEY})

_JSON_SAFE_INT = 2**53 - 1
_DECIMAL_WIRE_INT_MIN = -(2**63)
_DECIMAL_WIRE_INT_MAX = 2**64 - 1
_CANONICAL_DECIMAL_INT = re.compile(r"^-?(?:0|[1-9][0-9]*)$")

REGION_KINDS: frozenset[str] = frozenset({"map", "fold", "while"})
BINDING_MODES: frozenset[str] = frozenset({"zip", "cross", "broadcast"})
OUTPUT_MODES: frozenset[str] = frozenset({"gather", "compact", "state", "flatten"})

JsonObject = dict[str, Any]
_TYPE_EXPR_WIRE_VERSION = max(SCHEMA_WIRE_SERVE_VERSIONS)


class GraphWireError(ValueError):
    """The submitted document is not a valid graph wire shape."""


def _refuse_nested_unsafe_ints(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is int and abs(item) > _JSON_SAFE_INT:
            raise GraphWireError(
                "unsafe integer nested inside a literal is not representable on the graph wire"
            )
        if isinstance(item, list | tuple):
            pending.extend(cast("list[object] | tuple[object, ...]", item))
        elif isinstance(item, dict):
            pending.extend(cast(JsonObject, item).values())


def _as_object(value: Any, what: str) -> JsonObject:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[Any, Any], value)
    ):
        raise GraphWireError(f"{what} must be an object")
    return cast(JsonObject, value)


def _input_to_wire(value: object) -> Any:
    if isinstance(value, Link):
        return {LINK_KEY: {"node": value.node_id, "output": value.output_id}}
    if isinstance(value, TypedLiteral):
        _refuse_nested_unsafe_ints(value.value)
        return {TYPED_KEY: {"type": value.type_id, "value": value.value}}
    if type(value) is int and abs(value) > _JSON_SAFE_INT:
        if not _DECIMAL_WIRE_INT_MIN <= value <= _DECIMAL_WIRE_INT_MAX:
            raise GraphWireError(f"integer literal is outside the supported range: {value}")
        return {INT_KEY: str(value)}
    if isinstance(value, dict):
        reserved = RESERVED_INPUT_KEYS & cast(JsonObject, value).keys()
        if reserved:
            raise GraphWireError(
                f"literal inputs must not contain the reserved key(s) {sorted(reserved)!r}"
            )
    _refuse_nested_unsafe_ints(cast(object, value))
    return cast(Any, value)


def _input_from_wire(value: Any) -> object:
    if type(value) is int and abs(value) > _JSON_SAFE_INT:
        raise GraphWireError("unsafe integer literal must use the decimal integer marker")
    if not isinstance(value, dict):
        _refuse_nested_unsafe_ints(cast(object, value))
        return cast(object, value)
    obj = cast(JsonObject, value)
    reserved = RESERVED_INPUT_KEYS & obj.keys()
    if not reserved:
        _refuse_nested_unsafe_ints(cast(object, obj))
        return cast(object, value)
    # A marker input is EXACTLY the marker object: two markers, or extra
    # keys riding alongside one, decode as an error rather than a guess
    # (the encoder can never produce either shape).
    if len(obj) > 1:
        raise GraphWireError(
            f"a marker input must be exactly one reserved key, got {sorted(obj)!r}"
        )
    if LINK_KEY in obj:
        link: Any = obj[LINK_KEY]
        if (
            not isinstance(link, dict)
            or not isinstance(cast(JsonObject, link).get("node"), str)
            or not isinstance(cast(JsonObject, link).get("output"), str)
        ):
            raise GraphWireError(f"malformed link: {value!r}")
        linked = cast(JsonObject, link)
        return Link(node_id=linked["node"], output_id=linked["output"])
    if TYPED_KEY in obj:
        typed: Any = obj[TYPED_KEY]
        if (
            not isinstance(typed, dict)
            or not isinstance(cast(JsonObject, typed).get("type"), str)
            or not cast(JsonObject, typed)["type"]
            or "value" not in cast(JsonObject, typed)
        ):
            raise GraphWireError(
                f"malformed typed literal (needs a non-empty string 'type' "
                f"and a 'value'): {value!r}"
            )
        stamped = cast(JsonObject, typed)
        _refuse_nested_unsafe_ints(stamped["value"])
        return TypedLiteral(type_id=stamped["type"], value=stamped["value"])
    if INT_KEY in obj:
        decimal: Any = obj[INT_KEY]
        if not isinstance(decimal, str) or _CANONICAL_DECIMAL_INT.fullmatch(decimal) is None:
            raise GraphWireError(f"malformed decimal integer literal: {value!r}")
        decoded = int(decimal)
        if abs(decoded) <= _JSON_SAFE_INT:
            raise GraphWireError(
                "decimal integer literal must be outside the JSON-double-safe range"
            )
        if not _DECIMAL_WIRE_INT_MIN <= decoded <= _DECIMAL_WIRE_INT_MAX:
            raise GraphWireError("decimal integer literal is outside the supported range")
        return decoded
    return cast(object, value)


def _link_to_wire(link: Link) -> JsonObject:
    return {"node": link.node_id, "output": link.output_id}


def _region_to_wire(node: RegionNode) -> JsonObject:
    region: JsonObject = {
        "kind": node.kind,
        "ports": {
            port: type_expr_to_wire(expr, wire_version=_TYPE_EXPR_WIRE_VERSION)
            for port, expr in node.ports.items()
        },
        "inputs": {input_id: _input_to_wire(value) for input_id, value in node.inputs.items()},
        "body": graph_to_wire(node.body),
        "outputs": {
            out_id: {"source": _link_to_wire(out.source), "mode": out.mode}
            for out_id, out in node.outputs.items()
        },
    }
    if node.element_ports:
        region["elementPorts"] = list(node.element_ports)
    if node.state_ports:
        region["statePorts"] = list(node.state_ports)
    if node.binding != "zip":
        region["binding"] = node.binding
    if node.max_iterations is not None:
        region["maxIterations"] = node.max_iterations
    if node.continue_source is not None:
        region["continueSource"] = _link_to_wire(node.continue_source)
    return {"region": region}


def graph_to_wire(graph: Graph) -> JsonObject:
    nodes: JsonObject = {}
    for node_id, node in graph.nodes.items():
        if isinstance(node, RegionNode):
            nodes[node_id] = _region_to_wire(node)
            continue
        entry: JsonObject = {
            "nodeType": node.node_type,
            "inputs": {input_id: _input_to_wire(value) for input_id, value in node.inputs.items()},
        }
        if node.output_members:
            entry["outputMembers"] = {
                fam: list(members) for fam, members in node.output_members.items()
            }
        if node.slot_variants:
            entry["slotVariants"] = dict(node.slot_variants)
        nodes[node_id] = entry
    return {"nodes": nodes}


def _link_from_wire(value: Any, what: str) -> Link:
    obj = _as_object(value, what)
    if not isinstance(obj.get("node"), str) or not isinstance(obj.get("output"), str):
        raise GraphWireError(f"{what} must carry string 'node' and 'output'")
    return Link(node_id=obj["node"], output_id=obj["output"])


def _str_list(value: Any, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in cast(list[Any], value)
    ):
        raise GraphWireError(f"{what} must be a list of strings")
    return tuple(cast(list[str], value))


def _region_from_wire(node_id: str, wire: JsonObject) -> RegionNode:
    where = f"region {node_id}"
    kind = wire.get("kind")
    if kind not in REGION_KINDS:
        raise GraphWireError(f"{where}: unknown kind {kind!r}")
    binding = wire.get("binding", "zip")
    if binding not in BINDING_MODES:
        raise GraphWireError(f"{where}: unknown binding {binding!r}")
    ports_wire = _as_object(wire.get("ports", {}), f"{where}: 'ports'")
    try:
        ports = {
            str(port): type_expr_from_wire(
                _as_object(expr, f"{where}: port {port}"),
                wire_version=_TYPE_EXPR_WIRE_VERSION,
            )
            for port, expr in ports_wire.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphWireError(f"{where}: malformed port type: {exc}") from exc
    inputs_wire = _as_object(wire.get("inputs", {}), f"{where}: 'inputs'")
    outputs_wire = _as_object(wire.get("outputs", {}), f"{where}: 'outputs'")
    outputs: dict[str, RegionOutput] = {}
    for out_id, raw in outputs_wire.items():
        out_obj = _as_object(raw, f"{where}: output {out_id}")
        mode = out_obj.get("mode", "gather")
        if mode not in OUTPUT_MODES:
            raise GraphWireError(f"{where}: output {out_id}: unknown mode {mode!r}")
        outputs[str(out_id)] = RegionOutput(
            source=_link_from_wire(out_obj.get("source"), f"{where}: output {out_id} source"),
            mode=cast('Literal["gather", "compact", "state", "flatten"]', mode),
        )
    max_iterations = wire.get("maxIterations")
    if max_iterations is not None and (
        isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
    ):
        raise GraphWireError(f"{where}: maxIterations must be an integer")
    continue_wire = wire.get("continueSource")
    return RegionNode(
        kind=cast(RegionKind, kind),
        body=graph_from_wire(wire.get("body")),
        ports=ports,
        inputs={str(input_id): _input_from_wire(value) for input_id, value in inputs_wire.items()},
        element_ports=_str_list(wire.get("elementPorts", []), f"{where}: 'elementPorts'"),
        state_ports=_str_list(wire.get("statePorts", []), f"{where}: 'statePorts'"),
        outputs=outputs,
        binding=cast(BindingMode, binding),
        max_iterations=max_iterations,
        continue_source=(
            None
            if continue_wire is None
            else _link_from_wire(continue_wire, f"{where}: 'continueSource'")
        ),
    )


def graph_from_wire(wire: Any) -> Graph:
    document = _as_object(wire, "graph wire")
    if "nodes" not in document:
        raise GraphWireError("graph wire must be an object with a 'nodes' object")
    nodes_wire = _as_object(document["nodes"], "'nodes'")
    nodes: dict[str, AnyNode] = {}
    for node_id, raw_entry in nodes_wire.items():
        # Keys are strings by _as_object's check. Reject the reserved path
        # characters here too (region bodies recurse through this function,
        # so this covers every nesting level) - see NODE_ID_FORBIDDEN_CHARS.
        if not node_id or any(ch in node_id for ch in NODE_ID_FORBIDDEN_CHARS):
            raise GraphWireError(
                f"node ids must be non-empty and may not contain '/', '[' or ']': {node_id!r}"
            )
        entry = _as_object(raw_entry, f"node {node_id}: entry")
        if "region" in entry:
            nodes[node_id] = _region_from_wire(
                node_id, _as_object(entry["region"], f"region {node_id}")
            )
            continue
        if not isinstance(entry.get("nodeType"), str):
            raise GraphWireError(f"node {node_id}: entry must be an object with 'nodeType'")
        inputs_wire = _as_object(entry.get("inputs", {}), f"node {node_id}: 'inputs'")
        members_wire = _as_object(
            entry.get("outputMembers", {}), f"node {node_id}: 'outputMembers'"
        )
        output_members: dict[str, tuple[str, ...]] = {}
        for fam, members in members_wire.items():
            if not isinstance(members, list) or not all(
                isinstance(m, str) for m in cast(list[Any], members)
            ):
                raise GraphWireError(
                    f"node {node_id}: output members for {fam!r} must be a list of strings"
                )
            output_members[str(fam)] = tuple(cast(list[str], members))
        variants_wire = _as_object(entry.get("slotVariants", {}), f"node {node_id}: 'slotVariants'")
        slot_variants: dict[str, str] = {}
        for slot_id, key in variants_wire.items():
            if not isinstance(key, str):
                raise GraphWireError(
                    f"node {node_id}: slot variant for {slot_id!r} must be a string"
                )
            slot_variants[str(slot_id)] = key
        nodes[node_id] = GraphNode(
            node_type=entry["nodeType"],
            inputs={
                str(input_id): _input_from_wire(value) for input_id, value in inputs_wire.items()
            },
            output_members=output_members,
            slot_variants=slot_variants,
        )
    return Graph(nodes=nodes)
