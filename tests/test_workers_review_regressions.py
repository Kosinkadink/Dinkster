from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Mapping

import pytest
from dinkster_api.v1 import current_export_snapshot
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_protocol import (
    Invocation,
    LazyStatusInvocation,
)
from dinkster_schema import (
    Input,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_event,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
    register_core_types,
)
from dinkster_workers import (
    InProcessWorker,
    current_execution_context,
)

from tests.async_scenarios import run_scenario


class SlowSyncNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="review.slow",
            display_name="Slow",
            category="test",
            body=(Output("value", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        time.sleep(0.1)
        report_event("review.thread", {"ok": True})
        return cls.outputs(value="done")


class ComboListAdmissionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        combo_list = TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
        return NodeSchema(
            node_type="review.combo_list_admission",
            body=(Input("values", combo_list), Output("values", combo_list)),
        )

    @classmethod
    def execute(cls, *, values: list[str]) -> Mapping[str, object]:
        return cls.outputs(values=values)


class ExecutionContextProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="review.execution_context",
            body=(Output("value", string),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        context = current_execution_context()
        assert context is not None
        snapshot = current_export_snapshot()
        assert snapshot == context.export_snapshot
        if snapshot is not None:
            assert context.export_snapshot is not None
            assert snapshot is not context.export_snapshot
            assert snapshot.prompt is not context.export_snapshot.prompt
        return cls.outputs(
            value="|".join(
                (
                    context.node_id or "",
                    context.arm or "",
                    context.expected_execution_identity or "",
                    context.extension_snapshot_digest or "",
                    ("none" if snapshot is None else str(snapshot.prompt)),
                    str(context.fp8_matmul),
                    str(context.cancelled()),
                )
            )
        )


def test_sync_node_does_not_block_loop_and_reports_from_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original_execute = SlowSyncNode.execute

        def pause_execution(_cls: type[SlowSyncNode]) -> Mapping[str, object]:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            return original_execute()

        monkeypatch.setattr(SlowSyncNode, "execute", classmethod(pause_execution))
        events: list[EngineEvent] = []
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas([SlowSyncNode]),
            registry=registry,
            worker=InProcessWorker(build_node_types([SlowSyncNode]), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        run = asyncio.create_task(
            engine.run(Graph(nodes={"slow": GraphNode("review.slow", {})}), ["slow"])
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert not run.done()
        finally:
            release.set()
            await run
        assert any(
            event.kind == "node_event" and event.detail["name"] == "review.thread"
            for event in events
        )

    run_scenario(scenario())


def test_worker_preserves_combo_list_order_duplicates_and_fingerprint() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = ComboListAdmissionNode.schema()
        worker = InProcessWorker(build_node_types((ComboListAdmissionNode,)), registry)
        value = registry.wrap("list<COMBO>", ["b", "a", "b"])
        result = await worker.invoke(
            Invocation(
                invocation_id="combo-list",
                node_id="n",
                node_type=schema.node_type,
                inputs={"values": value},
                effective_schema=schema,
            )
        )
        assert result.error is None
        assert result.outputs is not None
        output = result.outputs["values"]
        assert output.resolve() == ["b", "a", "b"]
        assert output.fingerprint == value.fingerprint

    run_scenario(scenario())


def test_secret_names_cannot_be_wired_or_executed_without_a_trusted_resolver() -> None:
    class CredentialConsumer(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.credential_consumer",
                body=(
                    Input("credential", TypeExpr.concrete(CORE_STRING), secret=True, pin=False),
                    Output("value", TypeExpr.concrete(CORE_BOOLEAN)),
                ),
            )

        @classmethod
        def check_lazy_status(cls, *, credential: str) -> list[str]:
            return [credential]

        @classmethod
        def execute(cls, *, credential: str) -> Mapping[str, object]:
            pytest.fail("An unresolved credential name reached pack code")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = schema_from_wire(schema_to_wire(CredentialConsumer.schema()))
        schemas = {schema.node_type: schema, "review.slow": SlowSyncNode.schema()}
        for invalid in (
            Link("source", "value"),
            TypedLiteral(CORE_STRING, "service_key"),
            "Bearer sensitive value",
            ["service_key"],
        ):
            graph = Graph(
                nodes={
                    "source": GraphNode("review.slow"),
                    "consumer": GraphNode(schema.node_type, {"credential": invalid}),
                }
            )
            diagnostics = validate(graph, schemas, ["consumer"])
            assert any(
                d.code == "invalid-secret-name"
                and d.node_id == "consumer"
                and d.input_id == "credential"
                for d in diagnostics
            )
        graph = Graph(
            nodes={"consumer": GraphNode(schema.node_type, {"credential": "service_key"})}
        )
        assert not validate(graph, schemas, ["consumer"])
        invoking = False

        @contextlib.contextmanager
        def pack_context():
            if invoking:
                raise RuntimeError("synthetic credential in pack setup failure")
            yield

        worker = InProcessWorker(
            build_node_types((CredentialConsumer,)), registry, pack_context=pack_context
        )
        invoking = True
        result = await worker.invoke(
            Invocation(
                invocation_id="credential",
                node_id="consumer",
                node_type=schema.node_type,
                inputs={"credential": registry.wrap(CORE_STRING, "service_key")},
                effective_schema=schema,
            )
        )
        assert result.outputs is None
        assert result.error is not None
        assert result.error.node_id == "consumer"
        assert result.error.message == "Secret inputs require a trusted credential resolver"
        assert "service_key" not in result.error.message
        for pack_failure in (False, True):
            invoking = pack_failure
            lazy_result = await worker.check_lazy_status(
                LazyStatusInvocation(
                    request_id="credential-lazy",
                    node_id="consumer",
                    node_type=schema.node_type,
                    available_inputs={"credential": registry.wrap(CORE_STRING, "service_key")},
                    connected_undemanded_inputs=(),
                    effective_schema=schema,
                )
            )
            assert lazy_result.requested_inputs is None
            assert lazy_result.error is not None
            assert lazy_result.error.node_id == "consumer"
            assert lazy_result.error.message == (
                "Secret inputs require a trusted credential resolver"
            )
            assert not lazy_result.error.traceback

    run_scenario(scenario())


def test_private_node_returns_refuse_literal_secrets_before_wrapping() -> None:
    from dinkster_assets.privacy import protect_secret_outputs

    class PrivateReturn(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.private_return",
                body=(
                    Input("text", TypeExpr.concrete(CORE_STRING)),
                    Output("text", TypeExpr.concrete(CORE_STRING)),
                ),
            )

        @classmethod
        def execute(cls, *, text: str) -> Mapping[str, object]:
            return cls.outputs(text=text)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types((PrivateReturn,)), registry)
        secret = "synthetic-private-return-credential"
        for text in ("safe response", f"prefix {secret} suffix"):
            invocation = Invocation(
                invocation_id="private-return",
                node_id="consumer",
                node_type=PrivateReturn.schema().node_type,
                inputs={"text": registry.wrap(CORE_STRING, text)},
                effective_schema=PrivateReturn.schema(),
            )
            with protect_secret_outputs((secret,)):
                result = await worker.invoke(invocation)
            if secret in text:
                assert result.outputs is None
                assert result.error is not None
                assert result.error.message == "Secret-bearing content cannot be saved"
                assert not result.error.traceback
            else:
                assert result.error is None
                assert result.outputs is not None
                assert result.outputs["text"].resolve() == text

    run_scenario(scenario())


def test_async_lazy_hooks_await_in_pack_and_execution_contexts() -> None:
    pack_active = False

    @contextlib.contextmanager
    def pack_context():  # noqa: ANN202 - local worker context fixture
        nonlocal pack_active
        assert not pack_active
        pack_active = True
        try:
            yield
        finally:
            pack_active = False

    class AsyncLazyNode(Node):
        behavior = "request"
        seen: list[tuple[bool, str | None]] = []

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.async-lazy",
                body=(
                    Input("selector", TypeExpr.concrete(CORE_BOOLEAN)),
                    Input("value", TypeExpr.concrete(CORE_INT), lazy=True),
                    Output("value", TypeExpr.concrete(CORE_INT)),
                ),
            )

        @classmethod
        async def check_lazy_status(cls, *, selector: bool, value: int | None) -> object:
            context = current_execution_context()
            cls.seen.append((pack_active, context.node_id if context else None))
            await asyncio.sleep(0)
            context = current_execution_context()
            cls.seen.append((pack_active, context.node_id if context else None))
            if cls.behavior == "error":
                raise RuntimeError("async lazy failure")
            if cls.behavior == "malformed":
                return "value"
            if selector and value is None:
                return ["value"]
            return None

        @classmethod
        def execute(cls, *, selector: bool, value: int | None) -> Mapping[str, object]:
            del selector
            return cls.outputs(value=value)

    registry = TypeRegistry()
    register_core_types(registry)
    worker = InProcessWorker(
        build_node_types((AsyncLazyNode,)), registry, pack_context=pack_context
    )
    schema = AsyncLazyNode.schema()

    def invocation(request_id: str, *, with_value: bool = False) -> LazyStatusInvocation:
        available = {"selector": registry.wrap(CORE_BOOLEAN, True)}
        connected = ("value",)
        if with_value:
            available["value"] = registry.wrap(CORE_INT, 7)
            connected = ()
        return LazyStatusInvocation(
            request_id=request_id,
            node_id="async-lazy",
            node_type=schema.node_type,
            available_inputs=available,
            connected_undemanded_inputs=connected,
            effective_schema=schema,
        )

    first = run_scenario(worker.check_lazy_status(invocation("round-1")))
    second = run_scenario(worker.check_lazy_status(invocation("round-2", with_value=True)))
    assert first.requested_inputs == ("value",)
    assert second.requested_inputs == ()
    assert AsyncLazyNode.seen == [(True, "async-lazy")] * 4

    AsyncLazyNode.behavior = "error"
    failed = run_scenario(worker.check_lazy_status(invocation("error")))
    assert failed.error is not None
    assert failed.error.message == "lazy-hook-failed: async lazy failure"
    assert "RuntimeError: async lazy failure" in failed.error.traceback

    AsyncLazyNode.behavior = "malformed"
    malformed = run_scenario(worker.check_lazy_status(invocation("malformed")))
    assert malformed.error is not None
    assert malformed.error.message == (
        "lazy-hook-failed: check_lazy_status must return a list or tuple"
    )
