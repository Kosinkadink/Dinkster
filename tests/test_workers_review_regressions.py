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
    InvocationEvent,
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


def test_trusted_secret_invocations_keep_values_out_of_results_and_reports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster_api.v1 import SecretRejected
    from dinkster_schema import report_log, report_progress
    from dinkster_values import Value, ValueMeta
    from dinkster_workers.boundary import ValueCodec, encode_result, error_from_wire

    secret = "synthetic-invocation-credential"
    mode = "safe"
    snapshot = {"text": "safe snapshot"}

    class DeferredText:
        transport = "pyobj"

        def load(self) -> object:
            return secret if mode == "deferred" else snapshot["text"]

    class SecretConsumer(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            string = TypeExpr.concrete(CORE_STRING)
            return NodeSchema(
                node_type="test.secret_consumer",
                input_values=True,
                body=(
                    Input("credential", string, secret=True, pin=False),
                    Input("duplicate", string, secret=True, pin=False),
                    Input("optional", string, secret=True, pin=False, required=False),
                    Output("text", string),
                ),
            )

        @classmethod
        def execute(
            cls,
            *,
            credential: str,
            duplicate: str,
            input_values: Mapping[str, Value],
            optional: str = "unset",
        ) -> Mapping[str, object]:
            print(credential)
            report_log("warning", credential)
            report_progress(2, 7, text=credential)
            if mode == "throw":
                raise RuntimeError(credential)
            if mode in ("rejected", "bad-rejected"):
                raise SecretRejected("credential" if mode == "rejected" else secret)
            if mode == "echo":
                return cls.outputs(text=credential)
            if mode in ("deferred", "snapshot"):
                return cls.outputs(text=Value(CORE_STRING, "opaque", ValueMeta(), DeferredText()))
            authorized = (
                credential == duplicate == secret
                and optional == "unset"
                and input_values["credential"].resolve() == "service_key"
            )
            return cls.outputs(text="authorized" if authorized else "wrong credential snapshot")

        @classmethod
        def check_lazy_status(cls, **kwargs: object) -> list[str]:
            credential = kwargs["credential"]
            if mode == "throw":
                raise RuntimeError(credential)
            if mode in ("rejected", "bad-rejected"):
                raise SecretRejected("credential" if mode == "rejected" else secret)
            return [str(credential)] if mode == "echo" else []

    async def scenario() -> None:
        nonlocal mode
        registry = TypeRegistry()
        register_core_types(registry)

        @contextlib.contextmanager
        def pack_context():
            if mode == "setup":
                raise RuntimeError(secret)
            yield
            if mode == "cleanup":
                raise RuntimeError(secret)

        worker = InProcessWorker(
            build_node_types((SecretConsumer,)), registry, pack_context=pack_context
        )
        schema = SecretConsumer.schema()
        inputs = {
            name: registry.wrap(CORE_STRING, "service_key" if name != "optional" else "missing")
            for name in ("credential", "duplicate", "optional")
        }
        requests: dict[str, int] = {}

        def resolve(name: str) -> str | None:
            requests[name] = requests.get(name, 0) + 1
            if name == "missing" or mode == "missing":
                return None
            return secret if requests[name] == 1 else "rotated-after-first-resolution"

        for mode in (
            "safe",
            "throw",
            "rejected",
            "bad-rejected",
            "echo",
            "deferred",
            "snapshot",
            "missing",
            "setup",
            "cleanup",
        ):
            snapshot["text"] = "safe snapshot"
            requests.clear()
            events: list[InvocationEvent] = []
            result = await worker.invoke(
                Invocation(
                    invocation_id="secret-run",
                    node_id="consumer",
                    node_type=schema.node_type,
                    inputs=inputs,
                    effective_schema=schema,
                    resolve_secret=resolve,
                ),
                events.append,
            )
            assert secret not in repr(result)
            assert secret not in repr(events)
            if mode in ("safe", "snapshot"):
                assert result.error is None
                assert result.outputs is not None
                snapshot["text"] = secret
                assert result.outputs["text"].resolve() == (
                    "authorized" if mode == "safe" else "safe snapshot"
                )
                assert [(e.name, e.data) for e in events] == [("progress", {"step": 2, "total": 7})]
            else:
                assert result.outputs is None
                assert result.error is not None
                assert result.error.message == (
                    "The service refused this secret."
                    if mode == "rejected"
                    else "Secret execution failed"
                )
                assert [(hint.code, hint.input_id) for hint in result.error.hints] == (
                    [("secret-rejected", "credential")] if mode == "rejected" else []
                )
                assert not result.error.traceback
                codec = ValueCodec(registry, use_shm=False)
                header, blobs, segments = encode_result(codec, result, "refused", 0)
                error_wire = header["error"]
                assert isinstance(error_wire, dict)
                assert error_from_wire(error_wire) == result.error
                assert not blobs and not segments
            requests.clear()
            lazy = await worker.check_lazy_status(
                LazyStatusInvocation(
                    request_id="secret-lazy",
                    node_id="consumer",
                    node_type=schema.node_type,
                    available_inputs=inputs,
                    connected_undemanded_inputs=(),
                    effective_schema=schema,
                    resolve_secret=resolve,
                )
            )
            assert secret not in repr(lazy)
            if mode in ("safe", "deferred", "snapshot"):
                assert lazy.error is None
                assert lazy.requested_inputs == ()
            else:
                assert lazy.error is not None
                assert lazy.error.message == (
                    "The service refused this secret."
                    if mode == "rejected"
                    else "Secret execution failed"
                )
                assert [(hint.code, hint.input_id) for hint in lazy.error.hints] == (
                    [("secret-rejected", "credential")] if mode == "rejected" else []
                )
                assert not lazy.error.traceback
        assert inputs["credential"].resolve() == "service_key"

    run_scenario(scenario())
    print("public-after-secret-invocations")
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "public-after-secret-invocations" in captured.out


def test_public_node_cannot_request_secret_replacement_for_arbitrary_inputs() -> None:
    from dinkster_api.v1 import SecretRejected

    class PublicConsumer(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            string = TypeExpr.concrete(CORE_STRING)
            return NodeSchema(
                node_type="test.public_secret_rejection",
                body=(Input("text", string), Output("text", string)),
            )

        @classmethod
        def execute(cls, *, text: str) -> Mapping[str, object]:
            raise SecretRejected(text)

        @classmethod
        def check_lazy_status(cls, *, text: str) -> list[str]:
            raise SecretRejected(text)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types((PublicConsumer,)), registry)
        schema = PublicConsumer.schema()
        for rejected_id in ("text", "/home/user/private/path"):
            inputs = {"text": registry.wrap(CORE_STRING, rejected_id)}
            result = await worker.invoke(
                Invocation(
                    invocation_id="public-rejection",
                    node_id="consumer",
                    node_type=schema.node_type,
                    inputs=inputs,
                    effective_schema=schema,
                )
            )
            assert result.error is not None
            assert not result.error.hints
            lazy = await worker.check_lazy_status(
                LazyStatusInvocation(
                    request_id="public-rejection-lazy",
                    node_id="consumer",
                    node_type=schema.node_type,
                    available_inputs=inputs,
                    connected_undemanded_inputs=(),
                    effective_schema=schema,
                )
            )
            assert lazy.error is not None
            assert not lazy.error.hints

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
