"""The queue-time step for automatic values.

Before a run is compiled, admitted or looked up in any cache, every field in
the run that declares ``automatic`` and has nothing stored and no wire is
worked out again here, and the value used is written into the run's graph as
an ordinary stored value. From then on the run treats it like any value the
user typed, so it is in the cache key as any field value is. The record says
for each such field which value was used and whether it was worked out or
was the field's default. An automatic value never blocks a run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from dinkster_graph import Graph, GraphNode, RegionNode
from dinkster_graph.model import is_link, literal_value
from dinkster_graph.plan import output_node_targets, reachable_from
from dinkster_schema import Input, NodeSchema, core_logger
from dinkster_schema.automatic import (
    AutomaticOutcome,
    AutomaticReason,
    LookupAnswer,
    LookupCall,
    work_out_automatic,
)
from dinkster_schema.body import all_rows
from dinkster_schema.formula import Failed

LOOKUP_TIME_LIMIT = 10.0
"""Seconds the step waits for the lookups of one queued run, all together."""

MAX_LOOKUP_ROUNDS = 8
"""Rounds of calls: a lookup's arguments may read another lookup's answer."""

CallLookup = Callable[[str, str, Sequence[object]], Awaitable[object]]
"""Calls a node type's declared lookup by its wire name with worked-out
arguments and returns its checked answer, or raises."""


@dataclass(frozen=True)
class AutomaticRecord:
    """What one automatic field of the run used."""

    node_id: str
    """The node's id in the run; inside a region, ``region/node``."""
    input_id: str
    value: object
    worked_out: bool
    reason: AutomaticReason | None = None

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "nodeId": self.node_id,
            "inputId": self.input_id,
            "value": self.value,
            "source": "worked-out" if self.worked_out else "default",
        }
        if self.reason is not None:
            wire["reason"] = self.reason
        return wire

    @staticmethod
    def from_wire(data: Mapping[str, object]) -> AutomaticRecord:
        reason = data.get("reason")
        return AutomaticRecord(
            node_id=str(data["nodeId"]),
            input_id=str(data["inputId"]),
            value=data.get("value"),
            worked_out=data.get("source") == "worked-out",
            reason=reason if isinstance(reason, str) else None,  # type: ignore[arg-type]
        )


_log = core_logger("engine.automatic")


def _declares_automatic(schema: NodeSchema) -> bool:
    return any(
        isinstance(row, Input) and row.automatic is not None for row in all_rows(schema.body)
    )


@dataclass
class _Node:
    path: str
    node: GraphNode
    schema: NodeSchema
    answers: dict[LookupCall, LookupAnswer]
    outcomes: dict[str, AutomaticOutcome]

    def work_out(self) -> None:
        self.outcomes = work_out_automatic(
            self.schema.body,
            self.node.inputs,
            self.node.output_members,
            self.node.slot_variants,
            tables=self.schema.tables,
            derived=self.schema.derived,
            linked=is_link,
            literal=literal_value,
            answers=self.answers,
        )


def _nodes(
    graph: Graph, schemas: Mapping[str, NodeSchema], targets: Sequence[str], prefix: str
) -> list[_Node]:
    """The nodes of the run that declare an automatic field: those the
    targets need, and every node inside a region the targets need."""
    found: list[_Node] = []
    wanted = targets or output_node_targets(graph, schemas)
    for node_id in sorted(reachable_from(graph, wanted)):
        node = graph.nodes[node_id]
        path = f"{prefix}{node_id}"
        if isinstance(node, RegionNode):
            found.extend(_nodes(node.body, schemas, (), f"{path}/"))
            continue
        schema = schemas.get(node.node_type)
        if schema is not None and not schema.effective and _declares_automatic(schema):
            found.append(_Node(path, node, schema, {}, {}))
    return found


async def _answer(call_lookup: CallLookup | None, node_type: str, call: LookupCall) -> LookupAnswer:
    if call_lookup is None:
        return Failed("lookup-failed")
    try:
        return await call_lookup(node_type, call.function, call.args)
    except TimeoutError:
        return Failed("lookup-timeout")
    except Exception:  # noqa: BLE001 - a failing lookup gives the default, never a failed run
        return Failed("lookup-failed")


def _with_values(
    graph: Graph, values: Mapping[str, Mapping[str, object]], prefix: str = ""
) -> Graph:
    nodes: dict[str, GraphNode | RegionNode] = {}
    for node_id, node in graph.nodes.items():
        path = f"{prefix}{node_id}"
        if isinstance(node, RegionNode):
            body = _with_values(node.body, values, f"{path}/")
            nodes[node_id] = node if body is node.body else replace(node, body=body)
        elif path in values:
            nodes[node_id] = replace(node, inputs={**node.inputs, **values[path]})
        else:
            nodes[node_id] = node
    if all(nodes[node_id] is node for node_id, node in graph.nodes.items()):
        return graph
    return replace(graph, nodes=nodes)


async def resolve_automatic(
    graph: Graph,
    targets: Sequence[str],
    schemas: Mapping[str, NodeSchema],
    *,
    call_lookup: CallLookup | None = None,
    time_limit: float = LOOKUP_TIME_LIMIT,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Graph, tuple[AutomaticRecord, ...]]:
    """Work out every automatic value the run uses and write it into the graph.

    The lookups of the run are called side by side; the step waits for them
    at most ``time_limit`` seconds in all, and a lookup without an answer by
    then gives the field's default."""
    nodes = _nodes(graph, schemas, targets, "")
    if not nodes:
        return graph, ()
    deadline = clock() + time_limit
    for node in nodes:
        node.work_out()
    for round_number in range(MAX_LOOKUP_ROUNDS):
        pending = [
            (node, outcome.call)
            for node in nodes
            for outcome in node.outcomes.values()
            if outcome.pending and outcome.call is not None and outcome.call not in node.answers
        ]
        if not pending:
            break
        remaining = deadline - clock() if round_number < MAX_LOOKUP_ROUNDS - 1 else 0.0
        calls = {
            (node.schema.node_type, call): asyncio.ensure_future(
                _answer(call_lookup, node.schema.node_type, call)
            )
            for node, call in pending
            if remaining > 0
        }
        if calls:
            done, late = await asyncio.wait(calls.values(), timeout=max(remaining, 0))
            for task in late:
                task.cancel()
            del done
        for node, call in pending:
            task = calls.get((node.schema.node_type, call))
            node.answers[call] = (
                task.result()
                if task is not None and task.done() and not task.cancelled()
                else Failed("lookup-timeout")
            )
        for node in nodes:
            node.work_out()
    values: dict[str, dict[str, object]] = {}
    records: list[AutomaticRecord] = []
    for node in nodes:
        for input_id, outcome in sorted(node.outcomes.items()):
            values.setdefault(node.path, {})[input_id] = outcome.value
            reason = "lookup-timeout" if outcome.pending else outcome.reason
            if reason == "wrong-type":
                _log.warning(
                    "%s node %s: the automatic value of %s is not a value of its field; "
                    "using the default, %r",
                    node.schema.node_type,
                    node.path,
                    input_id,
                    outcome.value,
                )
            records.append(
                AutomaticRecord(node.path, input_id, outcome.value, outcome.worked_out, reason)
            )
    return _with_values(graph, values), tuple(records)
