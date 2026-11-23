from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from dinkster_schema import ElaborationError, NodeSchema, elaborate
from dinkster_values import CORE_INT

from .model import Graph, GraphNode, Link, RegionNode, TypedLiteral, is_link, literal_value


@dataclass(frozen=True)
class LoweringProblem:
    code: str
    message: str
    node_id: str
    input_id: str = ""


@dataclass(frozen=True)
class LoweringResult:
    graph: Graph
    targets: tuple[str, ...]
    problems: tuple[LoweringProblem, ...] = ()


def _cone(nodes: Mapping[str, GraphNode | RegionNode], source: Link) -> set[str]:
    found: set[str] = set()
    stack = [source.node_id]
    while stack:
        node_id = stack.pop()
        if node_id in found or node_id not in nodes:
            continue
        found.add(node_id)
        stack.extend(v.node_id for v in nodes[node_id].inputs.values() if isinstance(v, Link))
    return found


def _prune(
    nodes: dict[str, GraphNode | RegionNode], candidates: set[str], targets: set[str]
) -> None:
    while True:
        referenced = {
            value.node_id
            for node in nodes.values()
            for value in node.inputs.values()
            if isinstance(value, Link)
        }
        removable = candidates & nodes.keys() - targets - referenced
        if not removable:
            return
        for node_id in removable:
            del nodes[node_id]


def lower_selectors(
    graph: Graph, targets: Sequence[str], schemas: Mapping[str, NodeSchema]
) -> LoweringResult:
    """Resolve every top-level selector whose pick is stored.

    Consumers of the selector are rewired to the picked member's value, the
    selector is dropped, and the other members' exclusive cones are pruned.
    A picked member with nothing plugged in keeps the selector as one empty
    member picked by position, so it passes nothing on at run time. A wired
    pick, and a node whose members do not elaborate, stays for the engine."""
    target_tuple = tuple(targets)
    nodes: dict[str, GraphNode | RegionNode] = dict(graph.nodes)
    target_set = set(target_tuple)
    candidates: set[str] = set()
    retained: set[str] = set()
    while True:
        selector_id = next(
            (
                node_id
                for node_id, node in nodes.items()
                if isinstance(node, GraphNode)
                and (schema := schemas.get(node.node_type)) is not None
                and schema.selector is not None
                and node_id not in retained
            ),
            None,
        )
        if selector_id is None:
            _prune(nodes, candidates, target_set)
            return LoweringResult(Graph(nodes), target_tuple)
        node = nodes[selector_id]
        assert isinstance(node, GraphNode)
        schema = schemas[node.node_type]
        selector = schema.selector
        assert selector is not None
        selected = node.inputs.get(selector.input)
        if isinstance(selected, Link):
            retained.add(selector_id)
            continue
        try:
            effective = elaborate(
                schema,
                node.inputs,
                node.output_members,
                node.slot_variants,
                linked=is_link,
                literal=literal_value,
            )
        except ElaborationError:
            retained.add(selector_id)
            continue
        if selector_id in target_set:
            problem = LoweringProblem(
                "prompt.selector_is_target",
                "a selector node cannot be a submission target",
                selector_id,
            )
            return LoweringResult(graph, target_tuple, (problem,))
        members = selector.members(effective.inputs)
        try:
            picked = selector.pick(members, None if selected is None else literal_value(selected))
        except ValueError as error:
            problem = LoweringProblem(
                "prompt.bad_selector_value", str(error), selector_id, selector.input
            )
            return LoweringResult(graph, target_tuple, (problem,))
        unpicked = [node.inputs.get(member) for member in members if member != picked]
        for value in unpicked:
            if isinstance(value, Link):
                candidates.update(_cone(nodes, value))
        if picked not in node.inputs:
            retained.add(selector_id)
            nodes[selector_id] = replace(
                node,
                inputs={
                    **{k: v for k, v in node.inputs.items() if k not in members},
                    selector.input: TypedLiteral(CORE_INT, 0),
                },
                output_members={
                    **node.output_members,
                    selector.group: (picked.removeprefix(f"{selector.group}."),),
                },
            )
            _prune(nodes, candidates, target_set)
            continue
        value = node.inputs[picked]
        output = Link(selector_id, schema.outputs[0].id)
        del nodes[selector_id]
        for consumer_id, consumer in tuple(nodes.items()):
            inputs = {
                key: value if current == output else current
                for key, current in consumer.inputs.items()
            }
            nodes[consumer_id] = replace(consumer, inputs=inputs)
        _prune(nodes, candidates, target_set)


__all__ = ["LoweringProblem", "LoweringResult", "lower_selectors"]
