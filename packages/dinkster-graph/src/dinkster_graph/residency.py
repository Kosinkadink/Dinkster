"""Pre-execution checks for graph edges that can carry process-resident values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dinkster_schema import NodeSchema, TypeExpr

from .model import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode
from .validate import Diagnostic, graph_output_types


@dataclass(frozen=True)
class ResidentRoute:
    process: str
    pack: str
    worker: str = "local"


@dataclass(frozen=True)
class ResidentTopology:
    types: frozenset[str]
    routes: Mapping[str, tuple[ResidentRoute, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "routes",
            MappingProxyType({key: tuple(value) for key, value in self.routes.items()}),
        )

    def validate(
        self,
        graph: Graph,
        schemas: Mapping[str, NodeSchema],
        placement: Callable[[str], str | None] | None = None,
    ) -> list[Diagnostic]:
        nodes: dict[str, GraphNode] = {}
        outputs: dict[tuple[str, str], TypeExpr] = {}
        aliases: dict[tuple[str, str], list[tuple[str, str]]] = {}
        inputs: list[tuple[str, str, tuple[str, str]]] = []

        def collect(current: Graph, prefix: str = "", region: RegionNode | None = None) -> None:
            def path(node_id: str) -> str:
                return f"{prefix}/{node_id}" if prefix else node_id

            declared = graph_output_types(current, schemas, {} if region is None else region.ports)
            for node_id, ports in declared.items():
                for output_id, expr in ports.items():
                    outputs[path(node_id), output_id] = expr
            if region is not None:
                for port, expr in region.ports.items():
                    outputs[path(PORTS_NODE_ID), port] = expr
            for node_id, node in current.nodes.items():
                node_path = path(node_id)
                if isinstance(node, GraphNode):
                    nodes[node_path] = node
                    inputs.extend(
                        (node_path, input_id, (path(source.node_id), source.output_id))
                        for input_id, source in node.inputs.items()
                        if isinstance(source, Link)
                    )
                    continue
                for port in node.ports:
                    sources: list[tuple[str, str]] = []
                    source = node.inputs.get(port)
                    if isinstance(source, Link):
                        sources.append((path(source.node_id), source.output_id))
                    if port in node.state_ports and (state := node.outputs.get(port)) is not None:
                        sources.append(
                            (f"{node_path}/{state.source.node_id}", state.source.output_id)
                        )
                    aliases[f"{node_path}/{PORTS_NODE_ID}", port] = sources
                for output_id, output in node.outputs.items():
                    sources = [(f"{node_path}/{output.source.node_id}", output.source.output_id)]
                    if output.mode == "state":
                        source = node.inputs.get(output_id)
                        if isinstance(source, Link):
                            sources.append((path(source.node_id), source.output_id))
                    aliases[node_path, output_id] = sources
                collect(node.body, node_path, node)

        collect(graph)

        def producers(
            source: tuple[str, str], seen: frozenset[tuple[str, str]] = frozenset()
        ) -> set[str]:
            if source in seen:
                return set()
            if source in aliases:
                return {
                    producer
                    for parent in aliases[source]
                    for producer in producers(parent, seen | {source})
                }
            return {source[0]} if source[0] in nodes else set()

        candidates: dict[str, tuple[ResidentRoute, ...]] = {}
        for path, node in nodes.items():
            routes = self.routes.get(node.node_type, ())
            preferred = None if placement is None else placement(path)
            candidates[path] = tuple(
                route for route in routes if preferred is None or route.worker == preferred
            )
        components = {path: {path} for path in nodes}
        problems: list[Diagnostic] = []
        for consumer, input_id, source in inputs:
            expr = outputs.get(source)
            if expr is None or not self._resident(expr):
                continue
            for producer in sorted(producers(source)):
                left = components[producer]
                right = components[consumer]
                if left is right or not candidates[producer] or not candidates[consumer]:
                    continue
                members = left | right
                processes: set[str] | None = None
                for member in sorted(members):
                    choices = {route.process for route in candidates[member]}
                    processes = choices if processes is None else processes & choices
                if not processes:
                    problems.append(
                        Diagnostic(
                            "error",
                            "resident-process-boundary",
                            f"Pack '{candidates[consumer][0].pack}' "
                            "cannot receive a resident value "
                            f"from pack '{candidates[producer][0].pack}' "
                            "in another process; admit the pack to the shared inference process "
                            "or use the owner's node through an engine-run node on a reference.",
                            consumer,
                            input_id,
                        )
                    )
                else:
                    for member in members:
                        components[member] = members
        return problems

    def _resident(self, expr: TypeExpr) -> bool:
        if expr.kind == "asset":
            return False
        return (
            bool(self.types.intersection(expr.types))
            or (expr.element is not None and self._resident(expr.element))
            or any(self._resident(element) for element in expr.elements)
        )
