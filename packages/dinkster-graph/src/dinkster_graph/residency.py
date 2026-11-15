"""Pre-execution checks for graph edges that can carry process-resident values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from dinkster_schema import NodeSchema, TypeExpr

from .model import Graph, GraphNode, Link
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

    def validate(
        self,
        graph: Graph,
        schemas: Mapping[str, NodeSchema],
        placement: Callable[[str], str | None] | None = None,
    ) -> list[Diagnostic]:
        outputs = graph_output_types(graph, schemas)
        problems: list[Diagnostic] = []
        for node_id, node in graph.nodes.items():
            if not isinstance(node, GraphNode):
                continue
            consumers = self.routes.get(node.node_type, ())
            preferred = None if placement is None else placement(node_id)
            if preferred is not None:
                consumers = tuple(route for route in consumers if route.worker == preferred)
            for input_id, source in node.inputs.items():
                if not isinstance(source, Link):
                    continue
                producer = graph.nodes.get(source.node_id)
                expr = outputs.get(source.node_id, {}).get(source.output_id)
                if not isinstance(producer, GraphNode) or expr is None or not self._resident(expr):
                    continue
                producers = self.routes.get(producer.node_type, ())
                preferred = None if placement is None else placement(source.node_id)
                if preferred is not None:
                    producers = tuple(route for route in producers if route.worker == preferred)
                if (
                    producers
                    and consumers
                    and not any(
                        left.process == right.process for left in producers for right in consumers
                    )
                ):
                    owner = producers[0].pack
                    consumer = consumers[0].pack
                    problems.append(
                        Diagnostic(
                            "error",
                            "resident-process-boundary",
                            f"Pack '{consumer}' cannot receive a resident value "
                            f"from pack '{owner}' "
                            "in another process; admit the pack to the shared inference process "
                            "or use the owner's node through an engine-run node on a reference.",
                            node_id,
                            input_id,
                        )
                    )
        return problems

    def _resident(self, expr: TypeExpr) -> bool:
        if expr.kind == "asset":
            return False
        return (
            bool(self.types.intersection(expr.types))
            or (expr.element is not None and self._resident(expr.element))
            or any(self._resident(element) for element in expr.elements)
        )
