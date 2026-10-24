"""Native execution-semantic gate ported from ComfyUI's test_execution.py.

Each test names its source function. The Dinkster job API and event log replace
ComfyUI's prompt, websocket, and history endpoints. Tests run with cache both
enabled and disabled where the source checks cache-sensitive behavior.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionError, GraphValidationError
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_nodes_dev.execution import EXECUTION_NODES
from dinkster_schema import build_node_types, build_schemas
from dinkster_server import create_app
from dinkster_values import TypeRegistry, is_absent, register_core_types
from dinkster_workers import InProcessWorker

SCHEMAS = build_schemas(EXECUTION_NODES)


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(EXECUTION_NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


async def make_client() -> TestClient:
    client = TestClient(TestServer(create_app(make_engine, SCHEMAS)))
    await client.start_server()
    return client


async def wait_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(500):
        response = await client.get(f"/api/jobs/client/{job_id}")
        assert response.status == 200
        status = await response.json()
        if status["state"] in ("completed", "failed", "cancelled"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def submit_body(
    graph: Graph, targets: list[str], job_id: str, *, cache_enabled: bool
) -> dict[str, object]:
    return {
        "clientId": "client",
        "jobId": job_id,
        "graph": graph_to_wire(graph),
        "targets": targets,
        "cacheEnabled": cache_enabled,
    }


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_lazy_input(cache_enabled: bool) -> None:
    """Port of ComfyUI test_lazy_input."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {
                "left": GraphNode("dev.execution.value", {"value": 11}),
                "right": GraphNode("dev.execution.value", {"value": 29}),
                "select": GraphNode(
                    "dev.execution.lazy_select",
                    {
                        "choose_right": True,
                        "left": Link("left", "value"),
                        "right": Link("right", "value"),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["select"], cache_enabled=cache_enabled)
        assert result.outputs["select"]["value"].resolve() == 29
        assert "left" not in result.executed
        assert "right" in result.executed

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_full_cache(cache_enabled: bool) -> None:
    """Port of ComfyUI test_full_cache."""

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph({"value": GraphNode("dev.execution.value", {"value": 7})})
        first = await engine.run(graph, ["value"], cache_enabled=cache_enabled)
        second = await engine.run(graph, ["value"], cache_enabled=cache_enabled)
        assert first.outputs["value"]["value"].resolve() == 7
        assert ("value" in second.cached) is cache_enabled
        assert ("value" in second.executed) is not cache_enabled

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_cache(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_cache."""

    async def scenario() -> None:
        engine = make_engine()
        first = Graph(
            {
                "source": GraphNode("dev.execution.value", {"value": 7}),
                "add": GraphNode(
                    "dev.execution.add", {"value": Link("source", "value"), "amount": 1}
                ),
            }
        )
        second = Graph(
            {
                "source": GraphNode("dev.execution.value", {"value": 7}),
                "add": GraphNode(
                    "dev.execution.add", {"value": Link("source", "value"), "amount": 2}
                ),
            }
        )
        await engine.run(first, ["add"], cache_enabled=cache_enabled)
        result = await engine.run(second, ["add"], cache_enabled=cache_enabled)
        assert result.outputs["add"]["value"].resolve() == 9
        assert ("source" in result.cached) is cache_enabled
        assert "add" in result.executed

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_cycle_error(cache_enabled: bool) -> None:
    """Port of ComfyUI test_cycle_error."""

    async def scenario() -> None:
        graph = Graph(
            {
                "left": GraphNode("dev.execution.add", {"value": Link("right", "value")}),
                "right": GraphNode("dev.execution.add", {"value": Link("left", "value")}),
            }
        )
        with pytest.raises(GraphValidationError, match="cycle"):
            await make_engine().run(graph, ["left"], cache_enabled=cache_enabled)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_missing_node_error(cache_enabled: bool) -> None:
    """Port of ComfyUI test_missing_node_error."""

    async def scenario() -> None:
        graph = Graph({"missing": GraphNode("dev.execution.missing")})
        with pytest.raises(GraphValidationError, match="unknown node type"):
            await make_engine().run(graph, ["missing"], cache_enabled=cache_enabled)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_undeclared_inputs(cache_enabled: bool) -> None:
    """Port of ComfyUI test_undeclared_inputs through native graph admission."""

    async def scenario() -> None:
        graph = Graph({"value": GraphNode("dev.execution.value", {"value": 3, "extra": 5})})
        with pytest.raises(GraphValidationError, match="unknown-input"):
            await make_engine().run(graph, ["value"], cache_enabled=cache_enabled)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_mixed_expansion_returns(cache_enabled: bool) -> None:
    """Port of ComfyUI test_mixed_expansion_returns."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {
                "parent": GraphNode("dev.execution.expand", {"value": 4}),
                "output": GraphNode("dev.execution.output", {"value": Link("parent", "value")}),
                "literal": GraphNode("dev.execution.output", {"value": Link("parent", "literal")}),
            }
        )
        for index in range(2):
            events.clear()
            result = await engine.run(graph, ["output", "literal"], cache_enabled=cache_enabled)
            assert result.outputs["output"]["value"].resolve() == 5
            assert result.outputs["literal"]["value"].resolve() == 99
            assert any(
                event.kind == "node_expanded" and event.node_id == "parent" for event in events
            )
            if index and cache_enabled:
                assert "parent" in result.cached

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_dynamic_cycle_error(cache_enabled: bool) -> None:
    """Port of ComfyUI test_dynamic_cycle_error."""

    async def scenario() -> None:
        with pytest.raises(ExecutionError, match="cycle"):
            await make_engine().run(
                Graph({"cycle": GraphNode("dev.execution.dynamic_cycle")}),
                ["cycle"],
                cache_enabled=cache_enabled,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_error(cache_enabled: bool) -> None:
    """Port of ComfyUI test_error."""

    async def scenario() -> None:
        with pytest.raises(ExecutionError, match="execution fixture failure"):
            await make_engine().run(
                Graph({"error": GraphNode("dev.execution.fail")}),
                ["error"],
                cache_enabled=cache_enabled,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_output_reuse(cache_enabled: bool) -> None:
    """Port of ComfyUI test_output_reuse."""

    async def scenario() -> None:
        graph = Graph(
            {
                "source": GraphNode("dev.execution.value", {"value": 41}),
                "left": GraphNode("dev.execution.output", {"value": Link("source", "value")}),
                "right": GraphNode("dev.execution.output", {"value": Link("source", "value")}),
            }
        )
        result = await make_engine().run(graph, ["left", "right"], cache_enabled=cache_enabled)
        assert result.outputs["left"]["value"].resolve() == 41
        assert result.outputs["right"]["value"].resolve() == 41
        assert result.executed.count("source") == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_custom_is_changed(cache_enabled: bool) -> None:
    """Port of ComfyUI test_custom_is_changed through native non-idempotency."""

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph({"value": GraphNode("dev.execution.volatile", {"value": 41})})
        first = await engine.run(graph, ["value"], cache_enabled=cache_enabled)
        second = await engine.run(graph, ["value"], cache_enabled=cache_enabled)
        assert first.outputs["value"]["value"].resolve() == 41
        assert second.outputs["value"]["value"].resolve() == 41
        assert second.cached == ()
        assert second.executed == ("value",)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_blocked_output_reports_once(cache_enabled: bool) -> None:
    """Native blocked-output coverage; per-element list broadcast is excluded."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {
                "block": GraphNode(
                    "dev.execution.block",
                    {"value": 4, "block": True, "message": "Blocked Execution"},
                ),
                "consumer": GraphNode("dev.execution.add", {"value": Link("block", "value")}),
            }
        )
        result = await engine.run(graph, ["consumer"], cache_enabled=cache_enabled)
        assert is_absent(result.outputs["consumer"]["value"])
        assert result.skipped == ("consumer",)
        notices = [event for event in events if event.kind == "node_blocked"]
        assert len(notices) == 1
        assert notices[0].detail["message"] == "Blocked Execution"

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_included_outputs(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_included_outputs."""

    async def scenario() -> None:
        graph = Graph(
            {
                "left_source": GraphNode("dev.execution.value", {"value": 3}),
                "right_source": GraphNode("dev.execution.value", {"value": 5}),
                "left": GraphNode("dev.execution.output", {"value": Link("left_source", "value")}),
                "right": GraphNode(
                    "dev.execution.output", {"value": Link("right_source", "value")}
                ),
            }
        )
        result = await make_engine().run(graph, ["left"], cache_enabled=cache_enabled)
        assert result.outputs["left"]["value"].resolve() == 3
        assert "right" not in result.outputs
        assert "right_source" not in result.executed

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_excluded_outputs(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_excluded_outputs."""

    async def scenario() -> None:
        graph = Graph(
            {
                "first": GraphNode("dev.execution.output", {"value": 3}),
                "second": GraphNode("dev.execution.output", {"value": 5}),
                "third": GraphNode("dev.execution.output", {"value": 7}),
            }
        )
        result = await make_engine().run(graph, ["first", "third"], cache_enabled=cache_enabled)
        assert result.outputs["first"]["value"].resolve() == 3
        assert result.outputs["third"]["value"].resolve() == 7
        assert "second" not in result.outputs
        assert "second" not in result.executed

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_dependencies(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_dependencies."""

    async def scenario() -> None:
        graph = Graph(
            {
                "source": GraphNode("dev.execution.value", {"value": 3}),
                "output": GraphNode("dev.execution.output", {"value": Link("source", "value")}),
                "final": GraphNode(
                    "dev.execution.add", {"value": Link("output", "value"), "amount": 2}
                ),
            }
        )
        result = await make_engine().run(graph, ["final"], cache_enabled=cache_enabled)
        assert result.outputs["final"]["value"].resolve() == 5
        assert {"source", "output", "final"}.issubset(result.executed)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_multiple_output_nodes(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_multiple_output_nodes."""

    async def scenario() -> None:
        graph = Graph(
            {
                "source": GraphNode("dev.execution.value", {"value": 3}),
                "first": GraphNode("dev.execution.output", {"value": Link("source", "value")}),
                "second": GraphNode("dev.execution.output", {"value": Link("first", "value")}),
                "other": GraphNode("dev.execution.output", {"value": 7}),
            }
        )
        result = await make_engine().run(graph, ["second"], cache_enabled=cache_enabled)
        assert result.outputs["second"]["value"].resolve() == 3
        assert "first" in result.executed
        assert "other" not in result.outputs

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_empty_list(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_empty_list."""

    async def scenario() -> None:
        result = await make_engine().run(
            Graph({"output": GraphNode("dev.execution.output", {"value": 3})}),
            [],
            cache_enabled=cache_enabled,
        )
        assert result.outputs == {}
        assert result.executed == ()
        assert result.cached == ()

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_partial_execution_with_lazy_nodes(cache_enabled: bool) -> None:
    """Port of ComfyUI test_partial_execution_with_lazy_nodes."""

    async def scenario() -> None:
        graph = Graph(
            {
                "selected": GraphNode("dev.execution.value", {"value": 3}),
                "unselected": GraphNode("dev.execution.value", {"value": 5}),
                "other": GraphNode("dev.execution.value", {"value": 7}),
                "select": GraphNode(
                    "dev.execution.lazy_select",
                    {
                        "choose_right": False,
                        "left": Link("selected", "value"),
                        "right": Link("unselected", "value"),
                    },
                ),
                "output": GraphNode("dev.execution.output", {"value": Link("select", "value")}),
                "other_output": GraphNode(
                    "dev.execution.output", {"value": Link("other", "value")}
                ),
            }
        )
        result = await make_engine().run(graph, ["output"], cache_enabled=cache_enabled)
        assert result.outputs["output"]["value"].resolve() == 3
        assert "unselected" not in result.executed
        assert "other" not in result.executed
        assert "other_output" not in result.outputs

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_for_loop(cache_enabled: bool) -> None:
    """Port of ComfyUI test_for_loop using native expansion, not raw links."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {"loop": GraphNode("dev.execution.test_for_loop_open", {"initial": 3, "iterations": 4})}
        )
        for _ in range(2):
            events.clear()
            result = await engine.run(graph, ["loop"], cache_enabled=cache_enabled)
            assert result.outputs["loop"]["value"].resolve() == 7
            expanded = [event for event in events if event.kind == "node_expanded"]
            assert len(expanded) == 5

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_parallel_sleep_nodes(cache_enabled: bool) -> None:
    """Port of ComfyUI test_parallel_sleep_nodes."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {
                "left": GraphNode("dev.execution.sleep", {"value": 3, "seconds": 0.1}),
                "right": GraphNode("dev.execution.sleep", {"value": 5, "seconds": 0.1}),
            }
        )
        started = time.monotonic()
        result = await engine.run(graph, ["left", "right"], cache_enabled=cache_enabled)
        assert time.monotonic() - started < 0.18
        assert result.outputs["left"]["value"].resolve() == 3
        assert result.outputs["right"]["value"].resolve() == 5

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_parallel_sleep_expansion(cache_enabled: bool) -> None:
    """Port of ComfyUI test_parallel_sleep_expansion."""

    async def scenario() -> None:
        events = []
        engine = make_engine(events.append)
        graph = Graph(
            {
                "parallel": GraphNode(
                    "dev.execution.parallel_sleep",
                    {"left": 3, "middle": 5, "right": 7, "seconds": 0.1},
                )
            }
        )
        started = time.monotonic()
        result = await engine.run(graph, ["parallel"], cache_enabled=cache_enabled)
        assert time.monotonic() - started < 0.18
        assert result.outputs["parallel"]["value"].resolve() == 15
        assert any(event.kind == "node_expanded" for event in events)

    asyncio.run(scenario())


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_cached_outputs_in_job_without_client_id(cache_enabled: bool) -> None:
    """Port of ComfyUI test_cached_outputs_in_job_without_client_id via /api/jobs events."""

    async def scenario() -> None:
        client = await make_client()
        try:
            graph = Graph({"output": GraphNode("dev.execution.output", {"value": 41})})
            first = await client.post(
                "/api/jobs", json=submit_body(graph, ["output"], "one", cache_enabled=cache_enabled)
            )
            assert first.status == 202
            first_ref = (await first.json())["jobRef"]
            assert (await wait_terminal(client, "one"))["state"] == "completed"
            second = await client.post(
                "/api/jobs", json=submit_body(graph, ["output"], "two", cache_enabled=cache_enabled)
            )
            assert second.status == 202
            second_ref = (await second.json())["jobRef"]
            assert (await wait_terminal(client, "two"))["state"] == "completed"
            events = await (
                await client.get(f"/api/jobs/by-ref/{second_ref}/events?after=0")
            ).json()
            kinds = [event["type"] for event in events["events"]]
            assert ("node_cached" in kinds) is cache_enabled
            assert first_ref != second_ref
        finally:
            await client.close()

    asyncio.run(scenario())
