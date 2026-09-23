from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from dinkster_schema import NodeSchema

from .model import Graph, GraphNode, Link, RegionNode


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
    target_tuple = tuple(targets)
    nodes: dict[str, GraphNode | RegionNode] = dict(graph.nodes)
    target_set = set(target_tuple)
    candidates: set[str] = set()
    runtime_selectors: set[str] = set()
    while True:
        selector_id = next(
            (
                node_id
                for node_id, node in nodes.items()
                if isinstance(node, GraphNode)
                and (schema := schemas.get(node.node_type)) is not None
                and schema.selector is not None
                and node_id not in runtime_selectors
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
            runtime_selectors.add(selector_id)
            continue
        if selector_id in target_set:
            problem = LoweringProblem(
                "prompt.selector_is_target",
                "a selector node cannot be a submission target",
                selector_id,
            )
            return LoweringResult(graph, target_tuple, (problem,))
        if type(selected) is not bool:
            problem = LoweringProblem(
                "prompt.bad_selector_value",
                "selector input must be a JSON boolean",
                selector_id,
                selector.input,
            )
            return LoweringResult(graph, target_tuple, (problem,))
        branch_id = selector.branches["true" if selected else "false"]
        if branch_id not in node.inputs:
            branch = schema.input(branch_id)
            if branch is not None and not branch.required:
                runtime_selectors.add(selector_id)
                continue
            problem = LoweringProblem(
                "prompt.missing_branch",
                f"selected branch input {branch_id!r} is absent",
                selector_id,
                branch_id,
            )
            return LoweringResult(graph, target_tuple, (problem,))
        value = node.inputs[branch_id]
        output = Link(selector_id, schema.outputs[0].id)
        inactive = node.inputs.get(selector.branches["false" if selected else "true"])
        del nodes[selector_id]
        for consumer_id, consumer in tuple(nodes.items()):
            inputs = {
                key: value if current == output else current
                for key, current in consumer.inputs.items()
            }
            nodes[consumer_id] = replace(consumer, inputs=inputs)
        if isinstance(inactive, Link):
            candidates.update(_cone(nodes, inactive))
        _prune(nodes, candidates, target_set)


__all__ = ["LoweringProblem", "LoweringResult", "lower_selectors"]
