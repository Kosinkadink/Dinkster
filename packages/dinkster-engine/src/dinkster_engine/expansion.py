"""Run-local graph admission for worker-returned expansion declarations."""

from __future__ import annotations

from collections.abc import Container, Mapping

from dinkster_graph import Graph, GraphNode, Link, elaborate_graph, snapshot_graph, validate
from dinkster_protocol import (
    CachedResult,
    CurrentNodeRef,
    CurrentOutputRef,
    JsonObject,
    LiteralInput,
    LocalOutputRef,
)
from dinkster_protocol.result_algebra import JsonLiteral, NodeMetadataRef
from dinkster_schema import NodeSchema, TypeExpr


def _literal(value: JsonLiteral) -> object:
    if isinstance(value, JsonObject):
        return {key: _literal(item) for key, item in value.fields}
    if isinstance(value, tuple):
        return [_literal(item) for item in value]
    return value


def local_output_id(binding: LocalOutputRef, schema: NodeSchema) -> str:
    if isinstance(binding.output_id, int):
        if binding.output_id >= len(schema.outputs):
            raise ValueError(f"expansion output index {binding.output_id} is out of range")
        return schema.outputs[binding.output_id].id
    return binding.output_id


def admit_expansion(
    graph: Graph,
    parent_id: str,
    call_index: int,
    entry: CachedResult,
    schemas: Mapping[str, NodeSchema],
    known_types: Container[str],
    ports: Mapping[str, TypeExpr] | None = None,
) -> tuple[Graph, dict[str, NodeSchema], dict[str, str], dict[str, tuple[str, str]]]:
    """Validate the overlay before exposing any node or changing the scheduler.

    Current-output edges read the parent's already-pinned present values.
    They do not demand the parent's unresolved child-backed output bindings.
    """
    expansion = entry.expansion
    assert expansion is not None
    ids = {
        node.local_id: f"{parent_id}.{call_index}.0.{node.local_id}"
        for node in expansion.expansion.nodes
    }
    if set(ids.values()) & set(graph.nodes):
        raise ValueError("expansion node id collides with the live graph")

    def metadata(ref: NodeMetadataRef | None) -> str:
        if ref is None or isinstance(ref, CurrentNodeRef):
            return parent_id
        return ids[ref.local_node_id]

    nodes = dict(graph.nodes)
    origins: dict[str, tuple[str, str]] = {}
    for node in expansion.expansion.nodes:
        inputs: dict[str, object] = {}
        for name, binding in node.inputs:
            if isinstance(binding, LiteralInput):
                inputs[name] = _literal(binding.value)
            elif isinstance(binding, CurrentOutputRef):
                inputs[name] = Link(parent_id, binding.output_id)
            else:
                inputs[name] = Link(ids[binding.local_node_id], str(binding.output_id))
        node_id = ids[node.local_id]
        nodes[node_id] = GraphNode(node.node_type, inputs)
        origins[node_id] = (metadata(node.parent), metadata(node.display))
    overlay = snapshot_graph(Graph(nodes))
    effective, diagnostics = elaborate_graph(overlay, schemas)
    for node in expansion.expansion.nodes:
        node_id = ids[node.local_id]
        inputs = dict(nodes[node_id].inputs)
        for name, binding in node.inputs:
            if isinstance(binding, LocalOutputRef) and isinstance(binding.output_id, int):
                source = ids[binding.local_node_id]
                inputs[name] = Link(source, local_output_id(binding, effective[source]))
        nodes[node_id] = GraphNode(node.node_type, inputs)
    overlay = snapshot_graph(Graph(nodes))
    diagnostics += validate(overlay, schemas, (), effective, known_types=known_types, ports=ports)
    added = set(ids.values())
    failures = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity == "error"
        or (diagnostic.node_id in added and diagnostic.code == "type-mismatch")
    ]
    if failures:
        raise ValueError("; ".join(f"{d.node_id}: {d.message}" for d in failures))
    parent_schema = effective[parent_id]
    for name, binding in expansion.bindings:
        if isinstance(binding, LocalOutputRef):
            child = effective[ids[binding.local_node_id]]
            output = child.output(local_output_id(binding, child))
            if output is None:
                raise ValueError(f"expansion output '{name}' references a missing child output")
            expected = parent_schema.output(name)
            assert expected is not None
            runtime = output.type.runtime_type_id()
            if runtime is not None and not expected.type.accepts_concrete(runtime):
                raise ValueError(f"expansion output '{name}' rejects child type {runtime}")
    return overlay, effective, ids, origins
