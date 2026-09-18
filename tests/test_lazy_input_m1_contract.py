"""Executable oracle for the frozen M1 lazy-input contract.

The committed pinned-ComfyUI fixture is reproducible, complete, internally
consistent, and executed through the real scheduler. Unsupported surfaces
remain covered by explicit refusal tests and roadmap boundaries.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import translate_mappings
from dinkster_engine import Engine, EngineEvent, ExecutionError, GraphValidationError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
    register_core_types,
)
from dinkster_workers import InProcessWorker

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "goldens" / "lazy_input_m1.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_lazy_input_m1_fixtures",
        REPO / "tools" / "gen_lazy_input_m1_fixtures.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN.read_text(encoding="ascii")))


def _visible(inputs: Sequence[Mapping[str, object]], demanded: set[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in inputs:
        input_id = cast(str, item["id"])
        if item["binding"] == "omitted":
            continue
        if item["lazy"] and item["binding"] == "link" and input_id not in demanded:
            result[input_id] = None
        else:
            result[input_id] = item.get("value")
    return result


def _normalize(
    request: Sequence[object], inputs: Sequence[Mapping[str, object]]
) -> tuple[list[str], str | None]:
    by_id = {cast(str, item["id"]): item for item in inputs}
    for item in request:
        if not isinstance(item, str):
            return [], "lazy-request-invalid"
        if item not in by_id:
            return [], "lazy-request-unknown-input"
        declared = by_id[item]
        if not declared["lazy"]:
            return [], "lazy-request-non-lazy"
        if declared["binding"] == "literal":
            return [], "lazy-request-literal"
        if declared["binding"] in {"omitted", "default"}:
            return [], "lazy-request-unconnected"
    names = set(cast("Sequence[str]", request))
    return [cast(str, item["id"]) for item in inputs if item["id"] in names], None


def _replay(case: Mapping[str, object]) -> dict[str, object]:
    if "lazy" not in cast("list[str]", case["targets"]):
        return {
            "outcome": "not-reached",
            "demandedInputs": [],
            "executedProducers": [],
            "reusedProducers": [],
            "hookViews": [],
            "demandEvents": [],
        }
    if cast("list[str]", case["staticCycle"]):
        return {
            "outcome": "validation-error",
            "errorCode": "cycle",
            "errorNode": None,
            "demandedInputs": [],
            "executedProducers": [],
            "reusedProducers": [],
            "hookViews": [],
            "demandEvents": [],
        }

    inputs = cast("list[dict[str, object]]", case["inputs"])
    by_id = {cast(str, item["id"]): item for item in inputs}
    demanded: set[str] = set()
    executed: list[str] = []
    reused: list[str] = []
    views: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    outcome = "execute"
    error_code: str | None = None
    error_node: str | None = None

    for round_number, hook_result in enumerate(cast("list[object]", case["hookRounds"]), 1):
        views.append(_visible(inputs, demanded))
        if isinstance(hook_result, Mapping):
            outcome, error_code, error_node = "error", "lazy-hook-failed", "lazy"
            break
        requested, problem = _normalize(cast("Sequence[object]", hook_result), inputs)
        if problem is not None:
            outcome, error_code, error_node = "error", problem, "lazy"
            break
        new = [name for name in requested if name not in demanded]
        demanded.update(new)
        producers = sorted(
            {cast(str, by_id[name]["producer"]) for name in new if "producer" in by_id[name]}
        )
        events.append(
            {
                "name": "lazy_demand",
                "data": {
                    "round": round_number,
                    "status": "waiting" if new else "ready",
                    "requestedInputs": requested,
                    "newInputs": new,
                    "demandedInputs": [
                        cast(str, item["id"]) for item in inputs if item["id"] in demanded
                    ],
                    "producerNodes": producers,
                },
            }
        )
        if not new:
            break
        for name in new:
            item = by_id[name]
            producer = cast(str, item["producer"])
            state = item.get("producerState", "cold")
            if state == "error":
                outcome, error_code, error_node = "error", "producer-failed", producer
                break
            if state == "cancel":
                outcome = "cancelled"
                break
            target = reused if state in {"warm", "incidental"} else executed
            if producer not in target:
                target.append(producer)
        if outcome != "execute":
            break

    result: dict[str, object] = {
        "outcome": outcome,
        "demandedInputs": [cast(str, item["id"]) for item in inputs if item["id"] in demanded],
        "executedProducers": executed,
        "reusedProducers": reused,
        "hookViews": views,
        "demandEvents": events,
    }
    if error_code is not None:
        result["errorCode"] = error_code
        result["errorNode"] = error_node
    return result


def _type_id(value: object) -> str:
    if type(value) is bool:
        return CORE_BOOLEAN
    if type(value) is int:
        return CORE_INT
    return CORE_STRING


async def _run_real_case(case: Mapping[str, object]) -> dict[str, object]:
    """Execute one pinned trace through the real planner, worker, and cache."""
    case_id = cast(str, case["id"])
    input_rows = cast("list[dict[str, object]]", case["inputs"])
    hook_rounds = list(cast("list[object]", case["hookRounds"]))
    hook_views: list[dict[str, object]] = []
    requested_inputs: set[str] = set()
    deferred_by_producer: dict[str, set[str]] = {}
    producer_order: list[str] = []
    executed_after_demand: set[str] = set()
    reused_producers: set[str] = set()
    cancel_started = asyncio.Event()
    demand_waiting = asyncio.Event()
    events: list[EngineEvent] = []
    node_types: list[type[Node]] = []
    nodes: dict[str, GraphNode] = {}

    for row in input_rows:
        if row["binding"] != "link" or not cast(bool, row["lazy"]):
            continue
        producer = cast(str, row["producer"])
        deferred_by_producer.setdefault(producer, set()).add(cast(str, row["id"]))
        if producer not in producer_order:
            producer_order.append(producer)

    def on_event(event: EngineEvent) -> None:
        events.append(event)
        if (
            event.kind == "node_event"
            and event.node_id == "lazy"
            and event.detail.get("name") == "lazy_demand"
        ):
            data = cast("Mapping[str, object]", event.detail["data"])
            if data["status"] == "waiting":
                demand_waiting.set()

    def make_source(
        node_id: str,
        value: object,
        state: str,
        *,
        cycle_back: bool = False,
    ) -> type[Node]:
        node_type = f"test.lazy-contract.{case_id}.{node_id}"
        value_type = TypeExpr.concrete(_type_id(value))

        class Source(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                inputs = [InputSpec("value", value_type)]
                if cycle_back:
                    inputs.append(InputSpec("back", TypeExpr.concrete(CORE_STRING)))
                return NodeSchema(
                    node_type=node_type,
                    inputs=tuple(inputs),
                    outputs=(OutputSpec("value", value_type),),
                )

            @classmethod
            async def execute(cls, value: object, **_inputs: object) -> Mapping[str, object]:
                if state == "error":
                    raise RuntimeError("producer failed")
                if state == "cancel":
                    cancel_started.set()
                    await asyncio.Event().wait()
                demanded = deferred_by_producer.get(node_id, set())
                if demanded:
                    if demanded & requested_inputs:
                        executed_after_demand.add(node_id)
                    else:
                        reused_producers.add(node_id)
                return cls.outputs(value=value)

        return Source

    for row in input_rows:
        if row["binding"] != "link":
            continue
        producer = cast(str, row["producer"])
        if producer in nodes:
            continue
        source = make_source(
            producer,
            row.get("value"),
            cast(str, row.get("producerState", "cold")),
            cycle_back=bool(case["staticCycle"]),
        )
        node_types.append(source)
        source_inputs: dict[str, object] = {"value": row.get("value")}
        if bool(case["staticCycle"]):
            source_inputs["back"] = Link("lazy", "value")
        nodes[producer] = GraphNode(source.define_schema().node_type, source_inputs)

    lazy_specs: list[InputSpec] = []
    lazy_bindings: dict[str, object] = {}
    for row in input_rows:
        input_id = cast(str, row["id"])
        binding = cast(str, row["binding"])
        value = row.get("value")
        required = binding not in {"omitted", "default"}
        lazy_specs.append(
            InputSpec(
                input_id,
                TypeExpr.concrete(_type_id(value)),
                required=required,
                default=value if binding == "default" else None,
                lazy=cast(bool, row["lazy"]),
            )
        )
        if binding == "link":
            lazy_bindings[input_id] = Link(cast(str, row["producer"]), "value")
        elif binding == "literal":
            lazy_bindings[input_id] = value
    if not any(spec.lazy for spec in lazy_specs):
        # The non-lazy-request oracle isolates that one validation rule. A
        # private omitted lazy socket activates the hook without changing its
        # visible kwargs or the committed trace vocabulary.
        lazy_specs.append(
            InputSpec(
                "__hook_trigger",
                TypeExpr.concrete(CORE_STRING),
                required=False,
                lazy=True,
            )
        )

    if any(row.get("producerState") == "incidental" for row in input_rows):
        incidental = next(row for row in input_rows if row.get("producerState") == "incidental")
        producer = cast(str, incidental["producer"])
        gate_type = f"test.lazy-contract.{case_id}.selector-gate"

        class SelectorGate(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    node_type=gate_type,
                    inputs=(InputSpec("dependency", TypeExpr.concrete(CORE_STRING)),),
                    outputs=(OutputSpec("value", TypeExpr.concrete(CORE_BOOLEAN)),),
                )

            @classmethod
            def execute(cls, dependency: object) -> Mapping[str, object]:
                del dependency
                return cls.outputs(value=False)

        node_types.append(SelectorGate)
        nodes["selector_gate"] = GraphNode(gate_type, {"dependency": Link(producer, "value")})
        lazy_bindings["selector"] = Link("selector_gate", "value")

    lazy_type = f"test.lazy-contract.{case_id}.consumer"

    class LazyConsumer(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type=lazy_type,
                inputs=tuple(lazy_specs),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def check_lazy_status(cls, **inputs: object) -> object:
            del cls
            hook_views.append(dict(inputs))
            if not hook_rounds:
                raise AssertionError("engine invoked an unexpected lazy round")
            result = hook_rounds.pop(0)
            if isinstance(result, Mapping):
                raise RuntimeError(str(result.get("error", "hook failed")))
            for item in cast("Sequence[object]", result):
                if isinstance(item, str):
                    requested_inputs.add(item)
            return result

        @classmethod
        def execute(cls, **_inputs: object) -> Mapping[str, object]:
            return cls.outputs(value="done")

    node_types.append(LazyConsumer)
    nodes["lazy"] = GraphNode(lazy_type, lazy_bindings)

    other_type = f"test.lazy-contract.{case_id}.other"

    class Other(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type=other_type,
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(value="other")

    node_types.append(Other)
    nodes["other"] = GraphNode(other_type)

    registry = TypeRegistry()
    register_core_types(registry)
    schemas = build_schemas(node_types)
    engine = Engine(
        schemas=schemas,
        registry=registry,
        worker=InProcessWorker(build_node_types(node_types), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )
    graph = Graph(nodes)

    warm = [cast(str, row["producer"]) for row in input_rows if row.get("producerState") == "warm"]
    for producer in warm:
        await engine.run(graph, [producer])
    events.clear()
    executed_after_demand.clear()
    reused_producers.clear()

    outcome = "not-reached" if "lazy" not in cast("list[str]", case["targets"]) else "execute"
    error_code: str | None = None
    error_node: str | None = None
    run_result = None
    try:
        if any(row.get("producerState") == "cancel" for row in input_rows):
            task = asyncio.create_task(engine.run(graph, cast("list[str]", case["targets"])))
            await asyncio.wait_for(demand_waiting.wait(), 2)
            await asyncio.wait_for(cancel_started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            outcome = "cancelled"
            assert not any(event.kind in {"node_failed", "run_finished"} for event in events)
        else:
            run_result = await engine.run(graph, cast("list[str]", case["targets"]))
    except GraphValidationError as exc:
        outcome = "validation-error"
        error_code = next(diag.code for diag in exc.diagnostics if diag.severity == "error")
    except ExecutionError as exc:
        outcome = "error"
        error_node = exc.error.node_id
        if error_node in deferred_by_producer:
            error_code = "producer-failed"
        else:
            error_code = exc.error.message.split(":", 1)[0]

    demand_events = [
        {
            "name": event.detail["name"],
            "data": dict(cast("Mapping[str, object]", event.detail["data"])),
        }
        for event in events
        if event.kind == "node_event"
        and event.node_id == "lazy"
        and event.detail.get("name") == "lazy_demand"
    ]
    demanded_inputs = (
        list(cast("list[str]", demand_events[-1]["data"]["demandedInputs"]))
        if demand_events
        else []
    )
    if run_result is not None:
        reused_producers.update(
            producer
            for producer in producer_order
            if producer in run_result.cached and deferred_by_producer[producer] & requested_inputs
        )

    actual: dict[str, object] = {
        "outcome": outcome,
        "demandedInputs": demanded_inputs,
        "executedProducers": [
            producer for producer in producer_order if producer in executed_after_demand
        ],
        "reusedProducers": [
            producer for producer in producer_order if producer in reused_producers
        ],
        "hookViews": hook_views,
        "demandEvents": demand_events,
    }
    if error_code is not None:
        actual["errorCode"] = error_code
        actual["errorNode"] = error_node
    return actual


def test_fixture_is_generator_authored_and_pinned() -> None:
    generator = _load_generator()
    assert GOLDEN.read_bytes() == generator.render_fixture()
    fixture = _fixture()
    meta = cast("dict[str, object]", fixture["_meta"])
    assert meta["referenceCommit"] == "947c2749dd04c51ef0e21b069544d8b0b4f9b411"
    assert set(cast("dict[str, str]", meta["referenceFiles"])) == {
        "comfy_execution/graph.py",
        "execution.py",
        "tests/execution/testing_nodes/testing-pack/specific_tests.py",
        "tests/execution/test_execution.py",
        "comfy_extras/nodes_logic.py",
    }
    assert all(
        len(digest) == 64 for digest in cast("dict[str, str]", meta["referenceFiles"]).values()
    )
    assert meta["wholeListReferenceCommit"] == "f4b99bc62389af315013dda85f24f2bbd262b686"
    assert set(cast("dict[str, str]", meta["wholeListReferenceFiles"])) == {
        "execution.py",
        "comfy_api/latest/_io.py",
    }
    assert all(
        len(digest) == 64
        for digest in cast("dict[str, str]", meta["wholeListReferenceFiles"]).values()
    )


def test_every_contract_case_replays_to_its_expected_trace() -> None:
    cases = cast("list[dict[str, object]]", _fixture()["cases"])
    assert len(cases) == 24
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert _replay(case) == case["expected"], case["id"]


def test_every_contract_case_runs_through_the_real_engine() -> None:
    async def scenario() -> None:
        cases = cast("list[dict[str, object]]", _fixture()["cases"])
        assert len(cases) == 24
        for case in cases:
            assert await _run_real_case(case) == case["expected"], case["id"]

    asyncio.run(scenario())


def test_shared_producer_is_deduplicated_in_demand_event() -> None:
    generator = _load_generator()
    case = next(
        case
        for case in cast("list[dict[str, object]]", _fixture()["cases"])
        if case["id"] == "demand-all"
    )
    inputs = cast("list[dict[str, object]]", case["inputs"])
    for item in inputs:
        if item["id"] in {"left", "right"}:
            item["producer"] = "shared_source"
    replayed = _replay(case)
    assert replayed == generator._oracle(case)
    events = cast("list[dict[str, object]]", replayed["demandEvents"])
    first_data = cast("dict[str, object]", events[0]["data"])
    assert first_data["producerNodes"] == ["shared_source"]
    real = asyncio.run(_run_real_case(case))
    real_events = cast("list[dict[str, object]]", real["demandEvents"])
    real_first = cast("dict[str, object]", real_events[0]["data"])
    assert real_first["producerNodes"] == ["shared_source"]


def test_fixture_matrix_covers_every_authorized_m1_case() -> None:
    fixture = _fixture()
    cases = cast("list[dict[str, object]]", fixture["cases"])
    covered = {label for case in cases for label in cast("list[str]", case["covers"])}
    assert {
        "all",
        "none",
        "some",
        "multi-round",
        "repeated",
        "satisfied",
        "invalid",
        "unknown",
        "literal",
        "unconnected",
        "non-lazy",
        "computed-selector",
        "deterministic-visibility",
        "serial",
        "parallel",
        "cold-cache",
        "warm-cache",
        "tri-state",
        "static-cycle",
        "hook-error",
        "producer-error",
        "cancellation",
        "partial-target",
    } <= covered
    deferred = {
        item["id"]: item["milestone"] for item in cast("list[dict[str, str]]", fixture["deferred"])
    }
    assert deferred == {
        "async-hook": "M1-follow-up",
        "list-mapping": "M3",
        "blocker-in-list": "M3",
        "is-changed": "M2",
        "raw-link": "M3",
        "accept-all": "M3",
        "dynamic-expansion": "M4",
        "native-control-flow-prediction-speculation": "M5",
    }


def test_visibility_is_independent_of_schedule_and_cache_incidents() -> None:
    cases = {
        case["id"]: case
        for case in cast("list[dict[str, object]]", _fixture()["cases"])
        if cast(str, case["id"]).startswith("determinism-")
    }
    assert set(cases) == {
        "determinism-serial-cold",
        "determinism-parallel-cold",
        "determinism-serial-warm",
        "determinism-parallel-warm",
    }
    initial_views = {
        json.dumps(cast("dict[str, object]", case["expected"])["hookViews"])
        for case in cases.values()
    }
    demanded = {
        tuple(cast("list[str]", cast("dict[str, object]", case["expected"])["demandedInputs"]))
        for case in cases.values()
    }
    assert len(initial_views) == 1
    assert demanded == {("left",)}


def test_current_runtime_executes_sync_scalar_lazy_inputs() -> None:
    class FutureLazy:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {"lazy": True})}}

        def run(self, value):  # noqa: ANN001, ANN201
            return (value,)

        @classmethod
        def check_lazy_status(cls, value):  # noqa: ANN001, ANN206
            return [] if value is not None else ["value"]

    runtime = cast("dict[str, object]", _fixture()["currentRuntime"])
    translation = translate_mappings({"FutureLazy": FutureLazy})
    assert runtime["engineImplementation"] is True
    assert "FutureLazy" not in translation.skipped
    (node_class,) = translation.node_classes
    value = node_class.define_schema().input("value")
    assert value is not None and value.lazy


def test_current_comfy_whole_list_projection_is_source_pinned() -> None:
    runtime = cast("dict[str, object]", _fixture()["currentRuntime"])
    projection = cast("dict[str, object]", runtime["wholeListProjection"])
    assert projection == {
        "connectedUndemanded": [None],
        "demanded": ["FIRST", "SECOND"],
        "unconnected": "omitted",
    }
