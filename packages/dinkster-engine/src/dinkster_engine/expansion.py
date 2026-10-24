"""Run-local graph admission for worker-returned expansion declarations."""

from __future__ import annotations

from collections.abc import Container, Mapping

from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    LoopPair,
    RegionNode,
    elaborate_graph,
    region_interface,
    snapshot_graph,
    validate,
)
from dinkster_protocol import (
    CachedResult,
    CurrentNodeRef,
    CurrentOutputRef,
    ExternalNodeRef,
    ExternalOutputRef,
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


def referenced_output(
    binding: LocalOutputRef | ExternalOutputRef,
    source_id: str,
    graph: Graph,
    effective: Mapping[str, NodeSchema],
    schemas: Mapping[str, NodeSchema],
) -> tuple[str, TypeExpr | None]:
    source = graph.nodes[source_id]
    interface = (
        region_interface(source, schemas)
        if isinstance(source, RegionNode)
        else {output.id: output.type for output in effective[source_id].outputs}
    )
    if isinstance(binding.output_id, int):
        if binding.output_id >= len(interface):
            raise ValueError(f"expansion output index {binding.output_id} is out of range")
        name = tuple(interface)[binding.output_id]
    else:
        name = binding.output_id
    if name not in interface:
        raise ValueError(f"expansion references missing output '{source_id}/{name}'")
    return name, interface[name]


def admit_expansion(
    graph: Graph,
    parent_id: str,
    call_index: int,
    entry: CachedResult,
    schemas: Mapping[str, NodeSchema],
    known_types: Container[str],
    ports: Mapping[str, TypeExpr] | None = None,
    expansion_demands: Mapping[str, set[str]] | None = None,
    loop_pairs: Mapping[str, LoopPair] | None = None,
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
        if isinstance(ref, ExternalNodeRef):
            if ref.node_id not in graph.nodes:
                raise ValueError(f"external display/parent node '{ref.node_id}' is missing")
            return ref.node_id
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
            elif isinstance(binding, ExternalOutputRef):
                if binding.node_id not in graph.nodes:
                    raise ValueError(f"external node '{binding.node_id}' is missing")
                inputs[name] = Link(binding.node_id, str(binding.output_id))
            else:
                inputs[name] = Link(ids[binding.local_node_id], str(binding.output_id))
        node_id = ids[node.local_id]
        nodes[node_id] = GraphNode(
            node.node_type,
            inputs,
            dict(node.output_members),
            dict(node.slot_variants),
        )
        origins[node_id] = (metadata(node.parent), metadata(node.display))
    overlay = snapshot_graph(Graph(nodes))
    effective, _ = elaborate_graph(overlay, schemas)
    for node in expansion.expansion.nodes:
        node_id = ids[node.local_id]
        inputs = dict(nodes[node_id].inputs)
        for name, binding in node.inputs:
            if isinstance(binding, (LocalOutputRef, ExternalOutputRef)) and isinstance(
                binding.output_id, int
            ):
                source = (
                    ids[binding.local_node_id]
                    if isinstance(binding, LocalOutputRef)
                    else binding.node_id
                )
                output_id, _ = referenced_output(binding, source, overlay, effective, schemas)
                inputs[name] = Link(source, output_id)
        nodes[node_id] = GraphNode(
            node.node_type,
            inputs,
            dict(node.output_members),
            dict(node.slot_variants),
        )
    overlay = snapshot_graph(Graph(nodes))
    effective, diagnostics = elaborate_graph(overlay, schemas)
    diagnostics += validate(
        overlay,
        schemas,
        (),
        effective,
        known_types=known_types,
        ports=ports,
        loop_pairs=loop_pairs,
    )
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
        if isinstance(binding, (LocalOutputRef, ExternalOutputRef)):
            source = (
                ids[binding.local_node_id]
                if isinstance(binding, LocalOutputRef)
                else binding.node_id
            )
            _, output_type = referenced_output(binding, source, overlay, effective, schemas)
            expected = parent_schema.output(name)
            assert expected is not None
            runtime = output_type.runtime_type_id() if output_type is not None else None
            if runtime is not None and not expected.type.accepts_concrete(runtime):
                raise ValueError(f"expansion output '{name}' rejects child type {runtime}")

    # Parent returns and generated output nodes are also demand edges. Ordinary
    # graph validation cannot see them, and a task awaiting itself never finishes.
    demand = {
        node_id: {value.node_id for value in node.inputs.values() if isinstance(value, Link)}
        for node_id, node in nodes.items()
    }
    if expansion_demands is not None:
        demand.update(
            (node_id, set(dependencies)) for node_id, dependencies in expansion_demands.items()
        )
    for node in expansion.expansion.nodes:
        demand[ids[node.local_id]] = {
            ids[value.local_node_id] if isinstance(value, LocalOutputRef) else value.node_id
            for _, value in node.inputs
            if isinstance(value, (LocalOutputRef, ExternalOutputRef))
        }
    demand[parent_id].update(
        ids[value.local_node_id] if isinstance(value, LocalOutputRef) else value.node_id
        for _, value in expansion.bindings
        if isinstance(value, (LocalOutputRef, ExternalOutputRef))
    )
    demand[parent_id].update(node_id for node_id in ids.values() if effective[node_id].output_node)
    pending: list[tuple[str, tuple[str, ...]]] = [
        (child, (parent_id,)) for child in demand[parent_id]
    ]
    visited: set[str] = set()
    while pending:
        node_id, path = pending.pop()
        if node_id == parent_id:
            raise ValueError("dynamic cycle: " + " -> ".join((*path, node_id)))
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend((child, (*path, node_id)) for child in demand.get(node_id, ()))
    return overlay, effective, ids, origins
