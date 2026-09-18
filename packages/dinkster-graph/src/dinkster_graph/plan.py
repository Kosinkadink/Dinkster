"""Execution planning: output-driven reachability + deterministic topo order."""

from __future__ import annotations

from collections.abc import Sequence

from .model import Graph, Link


class GraphCycleError(Exception):
    pass


def _dependencies(graph: Graph, node_id: str) -> set[str]:
    deps: set[str] = set()
    for value in graph.nodes[node_id].inputs.values():
        if isinstance(value, Link) and value.node_id in graph.nodes:
            deps.add(value.node_id)
    return deps


def reachable_from(graph: Graph, targets: Sequence[str]) -> set[str]:
    """Nodes needed to produce the targets (output-driven execution)."""
    needed: set[str] = set()
    stack = [t for t in targets if t in graph.nodes]
    while stack:
        node_id = stack.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        stack.extend(_dependencies(graph, node_id))
    return needed


def dependency_map(graph: Graph, targets: Sequence[str]) -> dict[str, set[str]]:
    """Dependencies of every reachable node, restricted to the reachable set.

    This is the plan's real shape: a DAG, not a line. The engine's ready-set
    scheduler dispatches every node whose dependency set is satisfied, so
    independent branches execute concurrently (hazard H12); plan() remains
    the deterministic linearization used for events and diagnostics.
    """
    needed = reachable_from(graph, targets)
    return {n: _dependencies(graph, n) & needed for n in needed}


def plan(graph: Graph, targets: Sequence[str]) -> list[str]:
    """Deterministic topological order over the reachable subgraph.

    Kahn's algorithm with a sorted ready set, so equal plans are identical
    across runs and machines.
    """
    needed = reachable_from(graph, targets)
    remaining_deps = {n: _dependencies(graph, n) & needed for n in needed}
    order: list[str] = []
    ready = sorted(n for n, deps in remaining_deps.items() if not deps)
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        newly_ready: list[str] = []
        for other, deps in remaining_deps.items():
            if node_id in deps:
                deps.discard(node_id)
                if not deps and other not in order:
                    newly_ready.append(other)
        ready = sorted(set(ready) | set(newly_ready))
    if len(order) != len(needed):
        raise GraphCycleError("cannot order graph: cycle among " + ", ".join(sorted(needed)))
    return order


def find_cycle(graph: Graph) -> list[str] | None:
    """Return one cycle as a node-id path, or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph.nodes, WHITE)
    path: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        color[node_id] = GRAY
        path.append(node_id)
        for dep in sorted(_dependencies(graph, node_id)):
            if color[dep] == GRAY:
                return path[path.index(dep) :] + [dep]
            if color[dep] == WHITE:
                found = visit(dep)
                if found:
                    return found
        path.pop()
        color[node_id] = BLACK
        return None

    for node_id in sorted(graph.nodes):
        if color[node_id] == WHITE:
            found = visit(node_id)
            if found:
                return found
    return None
