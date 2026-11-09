"""Lower ComfyUI's implicit list mapping into explicit Dinkster map regions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dinkster_graph import (
    Graph,
    GraphCycleError,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    elaborate_graph,
    plan,
)
from dinkster_graph.model import PORTS_NODE_ID
from dinkster_schema import NodeSchema, TypeExpr, plan_asset_coercion


@dataclass(frozen=True)
class ListMapProblem:
    code: str
    message: str
    node_id: str
    output_id: str = ""


@dataclass(frozen=True)
class ListMapResult:
    graph: Graph
    problems: tuple[ListMapProblem, ...] = ()


def _element_is_compatible(element: TypeExpr, destination: TypeExpr) -> bool:
    runtime_type = element.runtime_type_id()
    if runtime_type is None:
        return True
    if destination.accepts_concrete(runtime_type):
        return True
    return plan_asset_coercion(runtime_type, destination) is not None


def lower_implicit_list_maps(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
) -> ListMapResult:
    """Replace implicitly mapped compat nodes with explicit map regions.

    Effective list cardinality propagates through mapped nodes in one
    topological pass. Edges whose element type does not fit the scalar socket
    stay untouched so ordinary graph validation reports the existing type and
    cardinality diagnostics. One element port uses zip; multiple element
    ports use repeat-final-element broadcast.
    """
    effective, _ = elaborate_graph(graph, schemas)
    effective_outputs: dict[tuple[str, str], TypeExpr] = {
        (node_id, output.id): output.type
        for node_id, schema in effective.items()
        for output in schema.outputs
    }
    rewritten = dict(graph.nodes)

    try:
        order = plan(graph, list(graph.nodes))
    except GraphCycleError:
        return ListMapResult(graph)

    for node_id in order:
        node = graph.nodes[node_id]
        if not isinstance(node, GraphNode):
            continue
        schema = effective.get(node_id)
        if schema is None:
            continue

        mapped_inputs: dict[str, tuple[Link, TypeExpr]] = {}
        linked_inputs: dict[str, Link] = {}
        for input_id, value in node.inputs.items():
            if not isinstance(value, Link):
                continue
            input_spec = schema.input(input_id)
            source_type = effective_outputs.get((value.node_id, value.output_id))
            if input_spec is None or source_type is None:
                continue
            if input_spec.type.cardinality() != "scalar":
                linked_inputs[input_id] = value
                continue
            if source_type.cardinality() != "list":
                linked_inputs[input_id] = value
                continue
            assert source_type.element is not None
            if not _element_is_compatible(source_type.element, input_spec.type):
                linked_inputs[input_id] = value
                continue
            mapped_inputs[input_id] = (value, source_type.element)

        if not mapped_inputs:
            continue

        ports: dict[str, TypeExpr] = {}
        region_inputs: dict[str, Link | object] = {}
        body_inputs = dict(node.inputs)
        element_ports: list[str] = []
        for input_id, (outer_link, element_type) in mapped_inputs.items():
            ports[input_id] = element_type
            region_inputs[input_id] = outer_link
            body_inputs[input_id] = Link(PORTS_NODE_ID, input_id)
            element_ports.append(input_id)
        for input_id, outer_link in linked_inputs.items():
            input_spec = schema.input(input_id)
            if input_spec is None:
                continue
            ports[input_id] = input_spec.type
            region_inputs[input_id] = outer_link
            body_inputs[input_id] = Link(PORTS_NODE_ID, input_id)

        body_node = GraphNode(
            node_type=node.node_type,
            inputs=body_inputs,
            output_members=node.output_members,
            slot_variants=node.slot_variants,
        )
        rewritten[node_id] = RegionNode(
            kind="map",
            binding="zip" if len(element_ports) == 1 else "broadcast",
            body=Graph(nodes={node_id: body_node}),
            ports=ports,
            inputs=region_inputs,
            element_ports=tuple(element_ports),
            outputs={
                output.id: RegionOutput(
                    Link(node_id, output.id),
                    mode=("flatten" if output.type.cardinality() == "list" else "gather"),
                )
                for output in schema.outputs
            },
        )
        for output in schema.outputs:
            effective_outputs[(node_id, output.id)] = (
                output.type
                if output.type.cardinality() == "list"
                else TypeExpr.list_of(output.type)
            )

    return ListMapResult(Graph(nodes=rewritten))


__all__ = ["ListMapProblem", "ListMapResult", "lower_implicit_list_maps"]
