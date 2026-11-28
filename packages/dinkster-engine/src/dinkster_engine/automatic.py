"""Queue-time values for automatic fields and unanswered numeric fixes.

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
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
from typing import cast

from dinkster_graph import Graph, GraphNode, RegionNode
from dinkster_graph.model import TypedLiteral, is_link, literal_value
from dinkster_graph.plan import output_node_targets, reachable_from
from dinkster_schema import Input, NodeSchema, NumberWidget, WidgetRepresentations, core_logger
from dinkster_schema.automatic import (
    AutomaticOutcome,
    AutomaticReason,
    LookupAnswer,
    LookupCall,
    work_out_automatic,
)
from dinkster_schema.body import Formula, Lookup, all_rows, function_name
from dinkster_schema.elaborate import failed_rules
from dinkster_schema.flatten import (
    NOT_A_FILE,
    FlattenError,
    InputSite,
    flatten_body,
    site_evaluator,
)
from dinkster_schema.formula import UNKNOWN, Failed, FormulaRefused

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


@dataclass(frozen=True)
class CorrectedRecord:
    """A stored numeric field's effective value for this run only."""

    node_id: str
    input_id: str
    value: int | float

    def to_wire(self) -> dict[str, object]:
        return {
            "nodeId": self.node_id,
            "inputId": self.input_id,
            "value": {"$int": str(self.value)}
            if isinstance(self.value, int) and abs(self.value) > 2**53 - 1
            else self.value,
            "source": "number-outside-limits",
        }


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
    """The declared nodes the targets need, including those inside regions."""
    found: list[_Node] = []
    wanted = targets or output_node_targets(graph, schemas)
    for node_id in sorted(reachable_from(graph, wanted)):
        node = graph.nodes[node_id]
        path = f"{prefix}{node_id}"
        if isinstance(node, RegionNode):
            found.extend(_nodes(node.body, schemas, (), f"{path}/"))
            continue
        schema = schemas.get(node.node_type)
        if schema is not None and not schema.effective:
            found.append(_Node(path, node, schema, {}, {}))
    return found


async def _answer(call_lookup: CallLookup | None, node_type: str, call: LookupCall) -> LookupAnswer:
    if call_lookup is None:
        return Failed("lookup-failed")
    try:
        return await call_lookup(node_type, call.function, call.args)
    except TimeoutError:
        return Failed("lookup-timeout")
    except Exception as error:  # noqa: BLE001 - a failing lookup gives the default, never a failed run
        # The host's lookup error names a timeout by its code.
        timed_out = getattr(error, "code", None) == "lookup-timeout"
        return Failed("lookup-timeout" if timed_out else "lookup-failed")


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
    nodes = [
        node for node in _nodes(graph, schemas, targets, "") if _declares_automatic(node.schema)
    ]
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
        # Equal calls of the run share one call.
        calls: dict[tuple[str, LookupCall], asyncio.Future[LookupAnswer]] = {}
        for node, call in pending if remaining > 0 else ():
            key = (node.schema.node_type, call)
            if key not in calls:
                calls[key] = asyncio.ensure_future(_answer(call_lookup, *key))
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


def _correct_number(
    value: int | float, whole: bool, bounds: Mapping[str, int | float | None], widget: NumberWidget
) -> int | float | None:
    if value in widget.special:
        return value
    low, high = bounds["min"], bounds["max"]
    if low is not None and high is not None and low > high:
        return None
    if not whole:
        return min(
            high if high is not None else math.inf,
            max(low if low is not None else -math.inf, value),
        )
    if any(bound is not None and type(bound) is not int for bound in bounds.values()):
        return None
    minimum = low if isinstance(low, int) else -(2**63)
    maximum = high if isinstance(high, int) else 2**64 - 1
    declared_step = bounds["step"]
    step = declared_step if isinstance(declared_step, int) else 1
    if step <= 0:
        return None
    offset = bounds["offset"]
    start = offset if isinstance(offset, int) else minimum if low is not None else 0
    first = start - ((start - minimum) // step) * step
    last = start + ((maximum - start) // step) * step
    if first > last:
        return None
    lower = start + ((math.floor(value) - start) // step) * step
    upper = lower + step
    nearer = lower if value - lower <= upper - value else upper
    return min(last, max(first, nearer))


def _number_widget(spec: Input) -> NumberWidget | None:
    widgets = (
        tuple(item.widget for item in spec.widget.representations)
        if isinstance(spec.widget, WidgetRepresentations)
        else (spec.widget,)
    )
    return next((widget for widget in widgets if isinstance(widget, NumberWidget)), None)


def _correction_order(
    graph: Graph,
    targets: Sequence[str],
    schemas: Mapping[str, NodeSchema],
    automatic: Sequence[AutomaticRecord],
) -> TopologicalSorter[tuple[str, str]]:
    order: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for node in _nodes(graph, schemas, targets, ""):
        in_use = {
            record.input_id: record.value for record in automatic if record.node_id == node.path
        }
        if any(
            not rule.fix
            for rule in failed_rules(
                node.schema,
                {**node.node.inputs, **in_use},
                node.node.output_members,
                node.node.slot_variants,
                linked=is_link,
                literal=literal_value,
            )
        ):
            continue
        try:
            flat = flatten_body(
                node.schema.body or node.schema.inputs,
                node.node.inputs,
                node.node.output_members,
                node.node.slot_variants,
                tables=node.schema.tables,
                derived=node.schema.derived,
                linked=is_link,
                literal=literal_value,
                in_use=in_use,
            )
        except FlattenError:
            continue
        evaluator = site_evaluator(
            node.node.inputs,
            node.node.output_members,
            node.node.slot_variants,
            dict(flat.choices),
            tables=node.schema.tables,
            derived=node.schema.derived,
            linked=is_link,
            literal=literal_value,
            in_use=in_use,
        )
        reads: dict[str, set[str]] = {}
        numbers: set[str] = set()
        for site in flat.sites:
            spec = site.input
            expressions: Sequence[object] = ()
            if spec.id not in node.node.inputs and spec.automatic is not None:
                expressions = (spec.automatic,)
            elif (
                not is_link(node.node.inputs.get(spec.id))
                and type(literal_value(node.node.inputs.get(spec.id))) in (int, float)
                and (widget := _number_widget(spec)) is not None
            ):
                numbers.add(spec.id)
                expressions = tuple(
                    getattr(widget, name) for name in ("min", "max", "step", "offset")
                )
            reads[spec.id] = set()
            for expression in expressions:
                arguments = expression.args if isinstance(expression, Lookup) else (expression,)
                for argument in arguments:
                    if isinstance(argument, (Formula, str)):
                        try:
                            reads[spec.id].update(evaluator.dependencies(argument, site))
                        except FormulaRefused:
                            pass
        dependencies: dict[str, set[str]] = {}
        for input_id in numbers:
            found: set[str] = set()
            pending = list(reads[input_id])
            while pending:
                dependency = pending.pop()
                if dependency not in found:
                    found.add(dependency)
                    pending.extend(reads.get(dependency, ()))
            # A circular limit is unknown, not a reason to refuse or hold a run.
            if input_id not in found:
                dependencies[input_id] = found & numbers
        for input_id, found in dependencies.items():
            order[node.path, input_id] = {
                (node.path, dependency) for dependency in found if dependency in dependencies
            }
    return TopologicalSorter(order)


async def _resolve_numeric_corrections(
    graph: Graph,
    targets: Sequence[str],
    schemas: Mapping[str, NodeSchema],
    *,
    ready: set[tuple[str, str]],
    call_lookup: CallLookup | None = None,
    automatic: Sequence[AutomaticRecord] = (),
) -> tuple[Graph, tuple[CorrectedRecord, ...]]:
    """Apply unanswered numeric fixes only to the run, never its source document.

    Wires and absent fields stay unchanged. Unknown limits offer no correction;
    failed hard rules still see the submitted values and cannot be repaired away.
    """
    nodes = _nodes(graph, schemas, targets, "")
    candidates: list[tuple[_Node, InputSite, NumberWidget, dict[str, object]]] = []
    calls: dict[tuple[str, LookupCall], asyncio.Future[LookupAnswer]] = {}
    for node in nodes:
        in_use = {
            record.input_id: record.value for record in automatic if record.node_id == node.path
        }
        try:
            flat = flatten_body(
                node.schema.body or node.schema.inputs,
                node.node.inputs,
                node.node.output_members,
                node.node.slot_variants,
                tables=node.schema.tables,
                derived=node.schema.derived,
                linked=is_link,
                literal=literal_value,
                in_use=in_use,
            )
        except FlattenError:
            continue
        evaluator = site_evaluator(
            node.node.inputs,
            node.node.output_members,
            node.node.slot_variants,
            dict(flat.choices),
            tables=node.schema.tables,
            derived=node.schema.derived,
            linked=is_link,
            literal=literal_value,
            in_use=in_use,
        )
        for site in flat.sites:
            if (node.path, site.input.id) not in ready:
                continue
            value = node.node.inputs.get(site.input.id)
            if is_link(value) or type(literal_value(value)) not in (int, float):
                continue
            widget = _number_widget(site.input)
            if widget is None:
                continue
            bounds: dict[str, object] = {}
            for name in ("min", "max", "step", "offset"):
                bound = getattr(widget, name)
                try:
                    if isinstance(bound, Formula):
                        bound = evaluator.formula(bound, site)
                    elif isinstance(bound, Lookup):
                        arguments: list[object] = []
                        for argument in bound.args:
                            found = (
                                evaluator.file(argument, site)
                                if isinstance(argument, str)
                                else NOT_A_FILE
                            )
                            if found is NOT_A_FILE:
                                found = evaluator.formula(
                                    Formula(argument) if isinstance(argument, str) else argument,
                                    site,
                                )
                            if found is UNKNOWN or found is None or isinstance(found, Failed):
                                break
                            arguments.append(
                                list(cast("tuple[object, ...]", found))
                                if isinstance(found, tuple)
                                else found
                            )
                        else:
                            call = LookupCall(function_name(bound.function), tuple(arguments))
                            key = (node.schema.node_type, call)
                            if key not in calls:
                                calls[key] = asyncio.ensure_future(_answer(call_lookup, *key))
                            bound = calls[key]
                        if isinstance(bound, Lookup):
                            bound = UNKNOWN
                except FormulaRefused:
                    bound = UNKNOWN
                bounds[name] = bound
            candidates.append((node, site, widget, bounds))
    if calls:
        _, late = await asyncio.wait(calls.values(), timeout=LOOKUP_TIME_LIMIT)
        for task in late:
            task.cancel()
    values: dict[str, dict[str, object]] = {}
    records: list[CorrectedRecord] = []
    for node, site, widget, bounds in candidates:
        for name, bound in bounds.items():
            if isinstance(bound, asyncio.Future):
                bounds[name] = bound.result() if bound.done() and not bound.cancelled() else UNKNOWN
        if any(
            bound is not None
            and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or (isinstance(bound, float) and not math.isfinite(bound))
            )
            for bound in bounds.values()
        ):
            continue
        value = literal_value(node.node.inputs[site.input.id])
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        whole = site.input.type.runtime_type_id() == "INT"
        if (whole and type(value) is not int) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            continue
        corrected = _correct_number(
            value,
            whole,
            cast("dict[str, int | float | None]", bounds),
            widget,
        )
        if corrected is not None and corrected != value:
            stored = node.node.inputs[site.input.id]
            values.setdefault(node.path, {})[site.input.id] = (
                replace(stored, value=corrected) if isinstance(stored, TypedLiteral) else corrected
            )
            records.append(CorrectedRecord(node.path, site.input.id, corrected))
    return _with_values(graph, values), tuple(records)


async def resolve_run_values(
    graph: Graph,
    targets: Sequence[str],
    schemas: Mapping[str, NodeSchema],
    *,
    call_lookup: CallLookup | None = None,
) -> tuple[Graph, tuple[AutomaticRecord, ...], tuple[CorrectedRecord, ...]]:
    """Settle numeric dependencies before correcting a value, at most once.

    Circular limits stay unknown. Automatic fields read the settled values;
    lookups share one answer and one deadline for the whole run.
    """
    answers: dict[tuple[str, LookupCall], asyncio.Task[LookupAnswer]] = {}
    deadline = time.monotonic() + LOOKUP_TIME_LIMIT

    async def answer(node_type: str, function: str, args: Sequence[object]) -> object:
        call = LookupCall(function, tuple(args))
        key = (node_type, call)
        if key not in answers:
            answers[key] = asyncio.create_task(_answer(call_lookup, node_type, call))
        task = answers[key]
        if task.cancelled() or task.cancelling():
            return Failed("lookup-timeout")
        if task.done():
            return task.result()
        try:
            return await asyncio.wait_for(task, max(0, deadline - time.monotonic()))
        except TimeoutError:
            return Failed("lookup-timeout")

    _, preview = await resolve_automatic(graph, targets, schemas, call_lookup=answer)
    order = _correction_order(graph, targets, schemas, preview)
    order.prepare()
    corrected: list[CorrectedRecord] = []
    while order.is_active():
        ready = order.get_ready()
        graph, records = await _resolve_numeric_corrections(
            graph, targets, schemas, ready=set(ready), call_lookup=answer, automatic=preview
        )
        corrected.extend(records)
        order.done(*ready)
        if order.is_active():
            _, preview = await resolve_automatic(graph, targets, schemas, call_lookup=answer)
    graph, automatic = await resolve_automatic(graph, targets, schemas, call_lookup=answer)
    return (
        graph,
        automatic,
        tuple(sorted(corrected, key=lambda record: (record.node_id, record.input_id))),
    )
