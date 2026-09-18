"""Generate the real-engine preview/partial-execution conformance contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError, RunResult
from dinkster_graph import (
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    graph_to_wire,
)
from dinkster_nodes_dev import DEV_NODES, register_dev_types
from dinkster_schema import TypeExpr, build_node_types, build_schemas
from dinkster_server import value_descriptor
from dinkster_values import TypeRegistry, list_children, register_core_types
from dinkster_workers import InProcessWorker

GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "goldens" / "preview_partial_execution.json"
)
INT = TypeExpr.concrete("core.int")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _graph_digest(graph: Graph) -> str:
    return "sha256:" + hashlib.sha256(_canonical(graph_to_wire(graph))).hexdigest()


def _engine(events: list[EngineEvent], *, cache_entries: int = 128) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_dev_types(registry)
    return Engine(
        schemas=build_schemas(DEV_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(DEV_NODES), registry),
        cache=MemoryLRUCache(cache_entries),
        on_event=events.append,
        resource_capacities={"conformance-resource": 1},
    )


def _inspection(
    selections: Sequence[Mapping[str, str]],
    result: RunResult,
    engine: Engine,
) -> list[dict[str, object]]:
    inspections: list[dict[str, object]] = []
    for selection in selections:
        node_id = selection["nodeId"]
        output_id = selection["outputId"]
        descriptor = None
        if node_id in result.outputs and output_id in result.outputs[node_id]:
            descriptor = value_descriptor(result.outputs[node_id][output_id], engine.registry)
        inspections.append(
            {
                "nodeId": node_id,
                "outputId": output_id,
                "descriptor": descriptor,
            }
        )
    return inspections


def _outputs(result: RunResult, engine: Engine) -> dict[str, object]:
    described: dict[str, object] = {}
    for node_id, outputs in result.outputs.items():
        described[node_id] = {
            output_id: value_descriptor(value, engine.registry)
            for output_id, value in outputs.items()
        }
    return described


def _phase(
    result: RunResult,
    engine: Engine,
    events: Sequence[EngineEvent],
    selections: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    kinds = [event.kind for event in events]
    assert kinds and kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    for node_id in result.executed:
        assert (
            sum(event.kind == "node_finished" and event.node_id == node_id for event in events) == 1
        )
    return {
        "terminalState": "completed",
        "expectedExecuted": sorted(result.executed),
        "expectedCached": sorted(result.cached),
        "expectedSkipped": sorted(result.skipped),
        "outputs": _outputs(result, engine),
        "postRunInspection": _inspection(selections, result, engine),
        "eventConstraints": [
            "run_started precedes node terminal events",
            "each executed node has one terminal node_finished event",
            "run_finished is terminal",
        ],
    }


def _case(
    case_id: str,
    graph: Graph,
    targets: Sequence[str],
    selections: Sequence[Mapping[str, str]],
    phases: Mapping[str, object],
) -> dict[str, object]:
    return {
        "id": case_id,
        "graph": graph_to_wire(graph),
        "graphDigest": _graph_digest(graph),
        "targetNodeIds": list(targets),
        "inspectSelections": list(selections),
        "phases": dict(phases),
    }


def _target_graph(value: int) -> Graph:
    return Graph(
        {
            "source": GraphNode("dev.conformance.source", {"value": value}),
            "split": GraphNode("dev.conformance.split", {"value": Link("source", "value")}),
            "left": GraphNode(
                "dev.conformance.add",
                {"value": Link("split", "left"), "amount": 10},
            ),
            "right": GraphNode(
                "dev.conformance.add",
                {"value": Link("split", "right"), "amount": 20},
            ),
            "downstream": GraphNode("dev.conformance.add", {"value": Link("left", "value")}),
            "unselectedEffect": GraphNode(
                "dev.conformance.effect", {"value": Link("source", "value")}
            ),
        }
    )


async def _targets_case() -> dict[str, object]:
    events: list[EngineEvent] = []
    engine = _engine(events)
    graph = _target_graph(3)
    targets = ["split", "left", "right"]
    selections = [
        {"nodeId": "split", "outputId": "left"},
        {"nodeId": "split", "outputId": "right"},
        {"nodeId": "left", "outputId": "value"},
        {"nodeId": "right", "outputId": "value"},
    ]
    cold = await engine.run(graph, targets)
    cold_events = tuple(events)
    events.clear()
    warm = await engine.run(graph, targets)
    warm_events = tuple(events)
    mutated_graph = _target_graph(4)
    events.clear()
    mutated = await engine.run(mutated_graph, targets)
    mutated_events = tuple(events)
    return _case(
        "selected-target-closure",
        graph,
        targets,
        selections,
        {
            "cold": _phase(cold, engine, cold_events, selections),
            "warm": _phase(warm, engine, warm_events, selections),
            "inputMutated": {
                **_phase(mutated, engine, mutated_events, selections),
                "graph": graph_to_wire(mutated_graph),
                "graphDigest": _graph_digest(mutated_graph),
            },
        },
    )


async def _effect_case() -> dict[str, object]:
    events: list[EngineEvent] = []
    engine = _engine(events)
    graph = Graph(
        {
            "source": GraphNode("dev.conformance.source", {"value": 5}),
            "effect": GraphNode("dev.conformance.effect", {"value": Link("source", "value")}),
        }
    )
    targets = ["effect"]
    selections = [{"nodeId": "effect", "outputId": "value"}]
    first = await engine.run(graph, targets)
    first_events = tuple(events)
    events.clear()
    second = await engine.run(graph, targets)
    return _case(
        "effectful-rerun",
        graph,
        targets,
        selections,
        {
            "first": _phase(first, engine, first_events, selections),
            "second": _phase(second, engine, tuple(events), selections),
        },
    )


async def _absence_case() -> dict[str, object]:
    events: list[EngineEvent] = []
    engine = _engine(events)
    graph = Graph(
        {
            "maybe": GraphNode("dev.conformance.maybe", {"present": False}),
            "skip": GraphNode("dev.conformance.add", {"value": Link("maybe", "value")}),
            "omit": GraphNode(
                "dev.conformance.omit",
                {"base": 11, "optional": Link("maybe", "value")},
            ),
            "fail": GraphNode("dev.conformance.fail-absent", {"value": Link("maybe", "value")}),
        }
    )
    selections = [
        {"nodeId": "maybe", "outputId": "value"},
        {"nodeId": "skip", "outputId": "value"},
        {"nodeId": "omit", "outputId": "value"},
    ]
    successful_targets = ["maybe", "skip", "omit"]
    successful = await engine.run(graph, successful_targets)
    successful_events = tuple(events)
    events.clear()
    failure: dict[str, object]
    try:
        await engine.run(graph, ["fail"])
    except ExecutionError as exc:
        assert not any(event.kind == "node_started" and event.node_id == "fail" for event in events)
        failure = {
            "terminalState": "failed",
            "error": exc.error.message,
            "expectedExecuted": sorted(
                event.node_id
                for event in events
                if event.kind == "node_finished" and event.node_id is not None
            ),
            "expectedCached": sorted(
                event.node_id
                for event in events
                if event.kind == "node_cached" and event.node_id is not None
            ),
            "expectedSkipped": [],
            "inspectSelections": [],
            "postRunInspection": [],
            "eventConstraints": ["absent fail policy rejects before consumer invocation"],
        }
    else:  # pragma: no cover - the conformance node is intentionally strict
        raise AssertionError("fail-on-absent phase unexpectedly completed")
    return _case(
        "absence-policies",
        graph,
        successful_targets,
        selections,
        {
            "defaultSkipAndOmit": _phase(successful, engine, successful_events, selections),
            "fail": {**failure, "targetNodeIds": ["fail"]},
        },
    )


async def _region_case() -> dict[str, object]:
    events: list[EngineEvent] = []
    engine = _engine(events)
    region = RegionNode(
        kind="map",
        body=Graph(
            {
                "add": GraphNode(
                    "dev.conformance.add",
                    {"value": Link(PORTS_NODE_ID, "item"), "amount": 1},
                )
            }
        ),
        ports={"item": INT},
        inputs={"item": [2, 4, 8]},
        element_ports=("item",),
        outputs={"values": RegionOutput(Link("add", "value"))},
    )
    graph = Graph({"mapped": region})
    selections = [{"nodeId": "mapped", "outputId": "values"}]
    result = await engine.run(graph, ["mapped"])
    value = result.outputs["mapped"]["values"]
    children = list_children(value)
    assert children is not None
    phase = _phase(result, engine, tuple(events), selections)
    phase["exactElementInspection"] = {
        "nodeId": "mapped",
        "outputId": "values",
        "element": 1,
        "value": children[1].resolve(),
    }
    expanded_index = next(i for i, event in enumerate(events) if event.kind == "region_expanded")
    finished_index = next(i for i, event in enumerate(events) if event.kind == "region_finished")
    assert expanded_index < finished_index
    iteration_node_ids = {
        event.node_id
        for event in events
        if event.kind == "node_started" and event.node_id is not None
    }
    assert iteration_node_ids
    assert iteration_node_ids.isdisjoint({"mapped"})
    phase["eventConstraints"] = [
        "region_expanded precedes region_finished",
        "iteration identities are observation-only and never target ids",
        "run_finished is terminal",
    ]
    return _case("region-list-inspection", graph, ["mapped"], selections, {"run": phase})


async def _preview_terminal_case() -> dict[str, object]:
    events: list[EngineEvent] = []
    engine = _engine(events)
    preview_graph = Graph({"preview": GraphNode("dev.conformance.preview", {"seed": 9})})
    selections = [
        {"nodeId": "preview", "outputId": "image"},
        {"nodeId": "preview", "outputId": "fallback"},
    ]
    result = await engine.run(preview_graph, ["preview"])
    preview_events = tuple(events)
    phase = _phase(result, engine, preview_events, selections)
    reports = [
        event
        for event in preview_events
        if event.kind == "node_event"
        and event.node_id == "preview"
        and event.detail.get("name") == "preview"
    ]
    assert reports and all(event.run_id == result.run_id for event in reports)
    phase["runtimePreviews"] = [
        {
            "nodeId": event.node_id,
            "data": dict(event.detail["data"]),
            "blobSha256": hashlib.sha256(event.detail["blob"]).hexdigest(),
        }
        for event in reports
    ]
    phase["renditions"] = [
        {"kind": spec.kind, "mime": spec.mime, "default": spec.default}
        for spec in engine.registry.renditions_of("dev.image")
    ]
    rendition = engine.registry.render(result.outputs["preview"]["image"], "png")
    phase["finalRenditionSha256"] = hashlib.sha256(rendition.data).hexdigest()
    phase["fallbackRenditions"] = [
        {"kind": spec.kind, "mime": spec.mime, "default": spec.default}
        for spec in engine.registry.renditions_of("core.int")
    ]

    events.clear()
    failure_graph = Graph({"failure": GraphNode("dev.conformance.failure")})
    try:
        await engine.run(failure_graph, ["failure"])
    except ExecutionError as exc:
        assert events[-1].kind == "node_failed"
        assert events[-1].node_id == "failure"
        assert events[-1].detail["message"] == exc.error.message
        failed: dict[str, object] = {
            "terminalState": "failed",
            "error": exc.error.message,
            "expectedExecuted": [],
            "expectedCached": [],
            "expectedSkipped": [],
            "inspectSelections": [],
            "postRunInspection": [],
            "eventConstraints": ["node_failed carries stable failure text"],
            "graph": graph_to_wire(failure_graph),
            "graphDigest": _graph_digest(failure_graph),
        }
    else:  # pragma: no cover
        raise AssertionError("failure phase unexpectedly completed")

    events.clear()
    cancel_graph = Graph({"cancel": GraphNode("dev.conformance.cancellable")})
    run = asyncio.create_task(engine.run(cancel_graph, ["cancel"]))
    for _ in range(1000):
        if any(event.kind == "node_event" and event.node_id == "cancel" for event in events):
            break
        await asyncio.sleep(0)
    assert any(
        event.kind == "node_event"
        and event.node_id == "cancel"
        and event.detail.get("name") == "progress"
        for event in events
    )
    in_use = engine.resource_status()["conformance-resource"]["executionInUse"]
    run.cancel()
    try:
        await run
    except asyncio.CancelledError:
        pass
    after_cancel = engine.resource_status()["conformance-resource"]["executionInUse"]
    assert in_use == 1
    assert after_cancel == 0
    cancelled = {
        "terminalState": "cancelled",
        "expectedExecuted": [],
        "expectedCached": [],
        "expectedSkipped": [],
        "inspectSelections": [],
        "postRunInspection": [],
        "resourceInUseBeforeCancel": in_use,
        "resourceInUseAfterCancel": after_cancel,
        "eventConstraints": [
            "progress precedes cancellation",
            "resource admission is released after cancellation",
        ],
        "targetNodeIds": ["cancel"],
        "graph": graph_to_wire(cancel_graph),
        "graphDigest": _graph_digest(cancel_graph),
    }
    return _case(
        "preview-and-terminal-paths",
        preview_graph,
        ["preview"],
        selections,
        {
            "preview": phase,
            "stableFailure": {**failed, "targetNodeIds": ["failure"]},
            "cancelled": cancelled,
        },
    )


async def build_golden_async() -> dict[str, object]:
    cases = [
        await _targets_case(),
        await _effect_case(),
        await _absence_case(),
        await _region_case(),
        await _preview_terminal_case(),
    ]
    return {
        "formatVersion": 1,
        "semantics": {
            "targets": "top-level node ids only",
            "inspection": "post-run (nodeId, outputId), never planner or cache identity",
            "events": "causal constraints only; progress and node chatter may coalesce or drop",
        },
        "cases": cases,
    }


def build_golden() -> dict[str, object]:
    return asyncio.run(build_golden_async())


def render_golden() -> bytes:
    return (json.dumps(build_golden(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def main() -> None:
    content = render_golden()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_bytes(content)
    print(f"golden sha256: {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
