"""dinkster-server: queue policy above the engine plus the native HTTP/WS
protocol (DESIGN 3.5, 3.10). Job identity is (clientId, jobId); events are
typed; node progress is a state map, never a single cursor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_collab import SessionService, add_session_routes
from dinkster_engine import (
    CompiledGraph,
    Engine,
    EngineEvent,
    EventListener,
    ExecutionRuntime,
    GraphCompileError,
    ProviderResolutionError,
)
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_memory import (
    ConsumerItem,
    MeasuredMemory,
    MemoryGovernor,
    PageMap,
    PressureSignal,
    Shedder,
)
from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    GRAPH_COMPILE_ERROR_TIMEOUT,
    GRAPH_COMPILERS_SURFACE,
    AttentionPolicyConfig,
    CompatGateDiagnostic,
    ExportSnapshot,
    ExtensionSnapshot,
    GraphCompilerRegistrySnapshot,
    KeyedContribution,
    PackSettingsSchema,
)
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComfyAliasConfidence,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    InputSpec,
    MappingSource,
    Node,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    SelectorSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_preview,
    report_progress,
    schema_signature,
    schema_to_wire,
)
from dinkster_server import (
    BINARY_BLOB_KEY,
    STATE_KEY,
    AuthError,
    EventHub,
    Job,
    JobQueue,
    PackFrontendAsset,
    PackIconAsset,
    PackInfo,
    Principal,
    Subscription,
    WorkerInfo,
    create_app,
    encode_binary_event,
    load_authenticator,
    principal_for,
    resolve_scope,
)
from dinkster_server.app import handle_submit
from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY
from dinkster_values import (
    CURVE_TYPE,
    PNG_CONTAINER_VERSION,
    Curve,
    TypeRegistry,
    register_core_types,
    register_curve_type,
)
from dinkster_values.model import PyObjPayload, Value, ValueMeta
from dinkster_workers import InProcessWorker
from scaffold_nodes import (
    SCAFFOLD_NODES,
    register_scaffold_types,
    scaffold_choices,
    scaffold_lazy_choices,
)

STRING = TypeExpr.concrete("core.string")
FLOAT = TypeExpr.concrete("core.float")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


class Shout(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.shout",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text.upper())


class Sleeper(Node):
    """Waits on a class-level gate so tests control when jobs finish."""

    gate: asyncio.Event
    entered: asyncio.Event

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.sleeper",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,  # never coalesced/cached across tests
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        Sleeper.entered.set()
        await Sleeper.gate.wait()
        return cls.outputs(out=tag)


class Boom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.boom",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        raise ValueError(f"boom: {tag}")


class ShapeBoom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.shape_boom",
            inputs=(InputSpec("layout", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, layout: str) -> Mapping[str, object]:
        if layout == "permutation":
            raise RuntimeError(
                "size mismatch for input: got torch.Size([1, 64, 64, 4]) "
                "and expected torch.Size([1, 4, 64, 64])"
            )
        raise RuntimeError(
            "The size of tensor a (64) must match the size of tensor b (32) "
            "at non-singleton dimension 2"
        )


class Previewer(Node):
    """Reports progress and a binary preview while running (DESIGN 3.5)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.previewer",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,  # never cached: events must fire every run
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        report_progress(1, 2, text="rendering")
        report_preview(b"\x89png-bytes", mime="image/png", width=2, height=2)
        return cls.outputs(out=text)


class Splitter(Node):
    """List output: descriptors must carry runtime length (DESIGN 3.13)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.splitter",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("words", TypeExpr.list_of(STRING)),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(words=text.split())


NODES: list[type[Node]] = [Echo, Shout, Sleeper, Boom, ShapeBoom, Previewer, Splitter]
SCHEMAS = build_schemas(NODES)


def alias_registry(*, carrier: str = "test.echo", target_input: str = "text") -> ComfyAliasRegistry:
    source = ComfyAliasSource("comfy-core", "Echo", "comfy.Echo", "b78cec87")
    return ComfyAliasRegistry(
        source_schemas=(
            ComfyAliasSourceSchema(
                NodeSchema(
                    source.node_type,
                    inputs=(InputSpec("text", STRING),),
                    outputs=(OutputSpec("out", STRING),),
                ),
                SCHEMA_WIRE_VERSION,
            ),
        ),
        records=(
            ComfyAliasRecord(
                id="comfy_alias:comfy-core/Echo",
                mapping_kind="op",
                carrier=carrier,
                source=source,
                replacement=ReplacementRule(
                    from_type=source.node_type,
                    cases=(
                        ReplacementCase.build(
                            carrier,
                            inputs={target_input: MappingSource.copy("text")},
                            outputs={"out": "out"},
                        ),
                    ),
                ),
                confidence=ComfyAliasConfidence("exact", ("tests/test_server.py",)),
            ),
        ),
    )


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


def compiler_runtime(engine: Engine) -> ExecutionRuntime:
    async def unused_transport(*_args: object) -> Mapping[str, object]:
        raise AssertionError("tests replace compile_for_execution before transport")

    contribution = KeyedContribution(
        surface_id=GRAPH_COMPILERS_SURFACE,
        id="test.compiler",
        behavior_metadata=(("contractVersion", 1), ("order", 0)),
    )
    return replace(
        engine.pin_execution(),
        graph_compiler_registry=GraphCompilerRegistrySnapshot((contribution,)),
        graph_compile_transport=unused_transport,
    )


def generated_echo_compiled(runtime: ExecutionRuntime, text: str = "compiled") -> CompiledGraph:
    node_id = "$gen-test-compiled"
    return CompiledGraph(
        Graph({node_id: GraphNode("test.echo", {"text": text})}),
        (node_id,),
        runtime.extension_snapshot_digest,
        {
            node_id: {
                "nodeId": node_id,
                "compilerId": "test.compiler",
                "passIndex": 0,
                "sources": ("e",),
                "localKey": "compiled",
            }
        },
    )


def echo_graph(text: str = "hello") -> Graph:
    return Graph(
        nodes={
            "e": GraphNode("test.echo", {"text": text}),
            "s": GraphNode("test.shout", {"text": Link("e", "out")}),
        }
    )


def reset_sleeper() -> None:
    Sleeper.gate = asyncio.Event()
    Sleeper.entered = asyncio.Event()


async def wait_for_state(job: Job, *states: str, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while job.state not in states:
            await asyncio.sleep(0.005)


# -- queue policy (no HTTP) ---------------------------------------------------


def test_queue_runs_job_to_completion() -> None:
    async def scenario() -> None:
        transitions: list[tuple[str, str]] = []
        queue = JobQueue(
            make_engine(), on_job_event=lambda j: transitions.append((str(j.key), j.state))
        )
        queue.start()
        job = queue.submit("c1", "j1", echo_graph(), ["s"])
        await wait_for_state(job, "completed")
        assert job.result is not None
        assert job.result.outputs["s"]["out"].resolve() == "HELLO"
        assert transitions == [("c1/j1", "queued"), ("c1/j1", "running"), ("c1/j1", "completed")]
        await queue.close()

    asyncio.run(scenario())


def test_queue_passes_job_attempt_to_engine_invocations(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        engine = make_engine()
        attempts: list[int] = []
        invoke = engine._worker.invoke

        async def recording_invoke(invocation, on_event=None):
            attempts.append(invocation.attempt_id)
            return await invoke(invocation, on_event)

        monkeypatch.setattr(engine._worker, "invoke", recording_invoke)
        queue = JobQueue(engine)
        job = queue.submit("c1", "j1", echo_graph(), ["s"])
        job.attempt = 5
        queue.start()
        await wait_for_state(job, "completed")
        assert attempts == [5, 5]
        await queue.close()

    asyncio.run(scenario())


def test_queue_idempotent_active_submit_and_different_content_conflict() -> None:
    async def scenario() -> None:
        reset_sleeper()
        transitions: list[str] = []
        queue = JobQueue(make_engine(), on_job_event=lambda job: transitions.append(job.state))
        queue.start()
        graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
        first = queue.submit("c1", "j1", graph, ["n"], fingerprint="same")
        duplicate = queue.submit("c1", "j1", graph, ["n"], fingerprint="same")
        assert duplicate is first
        assert transitions == ["queued"]
        with pytest.raises(ValueError, match="different content"):
            queue.submit("c1", "j1", graph, ["n"], fingerprint="different")
        # A different client may reuse the same jobId: identity is the pair.
        queue.submit("c2", "j1", echo_graph(), ["s"])
        await queue.close()

    asyncio.run(scenario())


def test_queue_export_snapshot_changes_default_fingerprint_and_resubmission() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine())
        graph = echo_graph()
        targets = ["s"]
        legacy_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "graph": graph_to_wire(graph),
                    "targets": targets,
                    "priority": 0,
                    "sourceDocument": "",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        without_snapshot = queue.submit("c", "legacy", graph, targets)
        assert without_snapshot.fingerprint == legacy_fingerprint
        default_attention_config = AttentionPolicyConfig()
        with_default_attention = queue.submit(
            "c",
            "default-attention",
            graph,
            targets,
            attention_config=default_attention_config,
        )
        assert with_default_attention.fingerprint == legacy_fingerprint
        assert with_default_attention.attention_config is default_attention_config
        attention_config = AttentionPolicyConfig("flash")
        with_attention = queue.submit(
            "c",
            "attention",
            graph,
            targets,
            attention_config=attention_config,
        )
        assert with_attention.fingerprint != without_snapshot.fingerprint
        assert with_attention.attention_config is attention_config
        with pytest.raises(TypeError, match="attention_config"):
            queue.submit(
                "c",
                "bad-attention",
                graph,
                targets,
                attention_config="flash",  # type: ignore[arg-type]
            )

        first_prompt = {"save": {"inputs": {"seed": 1}}}
        first_snapshot = ExportSnapshot(prompt=first_prompt)
        second_snapshot = ExportSnapshot(prompt={"save": {"inputs": {"seed": 2}}})
        first = queue.submit("c", "snapshot", graph, targets, export_snapshot=first_snapshot)
        assert first.fingerprint != without_snapshot.fingerprint
        first_prompt["save"]["inputs"]["seed"] = 99
        assert first.export_snapshot is not None
        assert first.export_snapshot.prompt["save"] == {"inputs": {"seed": 1}}
        with pytest.raises(ValueError, match="different content"):
            queue.submit("c", "snapshot", graph, targets, export_snapshot=second_snapshot)

        queue.cancel("c", "snapshot")
        replacement = queue.submit("c", "snapshot", graph, targets, export_snapshot=second_snapshot)
        assert replacement is not first
        assert replacement.fingerprint != first.fingerprint
        assert replacement.export_snapshot == second_snapshot
        assert replacement.export_snapshot is not second_snapshot
        assert queue.job_for_run(first.run_id) is None
        await queue.close()

    asyncio.run(scenario())


def test_queue_priority_order_with_fifo_tiebreak() -> None:
    async def scenario() -> None:
        reset_sleeper()
        order: list[str] = []

        def on_event(job: Job) -> None:
            if job.state == "running":
                order.append(job.key.job_id)

        queue = JobQueue(make_engine(), max_running_jobs=1, on_job_event=on_event)
        queue.start()
        blocker = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "block"})})
        first = queue.submit("c", "blocker", blocker, ["n"])
        await Sleeper.entered.wait()  # occupies the single running slot
        queue.submit("c", "low-a", echo_graph("a"), ["s"], priority=0)
        queue.submit("c", "low-b", echo_graph("b"), ["s"], priority=0)
        high = queue.submit("c", "high", echo_graph("c"), ["s"], priority=10)
        Sleeper.gate.set()
        await wait_for_state(first, "completed")
        await wait_for_state(high, "completed")
        low_b = queue.get("c", "low-b")
        assert low_b is not None
        await wait_for_state(low_b, "completed")
        assert order == ["blocker", "high", "low-a", "low-b"]
        await queue.close()

    asyncio.run(scenario())


def test_queue_set_max_running_validates_and_kicks_dispatch_on_raise() -> None:
    async def scenario() -> None:
        reset_sleeper()
        queue = JobQueue(make_engine(), max_running_jobs=1)
        queue.start()
        first = queue.submit(
            "c",
            "first",
            Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "first"})}),
            ["n"],
        )
        await Sleeper.entered.wait()
        second = queue.submit("c", "second", echo_graph(), ["s"])
        await asyncio.sleep(0)
        assert second.state == "queued"

        with pytest.raises(ValueError, match=">= 1"):
            queue.set_max_running(0)
        assert queue.status()["maxRunningJobs"] == 1

        queue.set_max_running(2)
        await wait_for_state(second, "running", "completed")
        assert first.state == "running"
        assert queue.status()["maxRunningJobs"] == 2
        Sleeper.gate.set()
        await wait_for_state(second, "completed")
        await wait_for_state(first, "completed")
        await queue.close()

    asyncio.run(scenario())


def test_queue_cancel_queued_and_running() -> None:
    async def scenario() -> None:
        reset_sleeper()
        queue = JobQueue(make_engine(), max_running_jobs=1)
        queue.start()
        running = queue.submit(
            "c", "running", Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "r"})}), ["n"]
        )
        await Sleeper.entered.wait()
        queued = queue.submit("c", "queued", echo_graph(), ["s"])
        assert queued.state == "queued"

        queue.cancel("c", "queued")
        assert queued.state == "cancelled"

        queue.cancel("c", "running")
        await wait_for_state(running, "cancelled")
        # Cancelling a terminal job is a no-op, not an error.
        recancelled = queue.cancel("c", "queued")
        assert recancelled is not None and recancelled.state == "cancelled"
        await queue.close()

    asyncio.run(scenario())


def test_queue_pause_holds_dispatch_and_resume_releases() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine())
        queue.start()
        assert not queue.paused
        queue.pause()
        job = queue.submit("c", "j1", echo_graph(), ["s"])
        await asyncio.sleep(0.05)  # dispatcher had every chance to misbehave
        assert job.state == "queued"
        assert queue.status()["paused"] is True
        queue.resume()
        await wait_for_state(job, "completed")
        # Cancel still works while paused: queued jobs are not stuck.
        queue.pause()
        held = queue.submit("c", "j2", echo_graph(), ["s"])
        queue.cancel("c", "j2")
        assert held.state == "cancelled"
        await queue.close()

    asyncio.run(scenario())


def test_queue_clear_cancels_queued_never_running() -> None:
    async def scenario() -> None:
        reset_sleeper()
        queue = JobQueue(make_engine(), max_running_jobs=1)
        queue.start()
        running = queue.submit(
            "c1", "run", Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "r"})}), ["n"]
        )
        await Sleeper.entered.wait()
        a = queue.submit("c1", "a", echo_graph(), ["s"])
        b = queue.submit("c2", "b", echo_graph(), ["s"])
        c = queue.submit("c1", "c", echo_graph(), ["s"])

        cleared = queue.clear("c1")  # one client's queued jobs only
        assert {j.key.job_id for j in cleared} == {"a", "c"}
        assert a.state == "cancelled" and c.state == "cancelled"
        assert b.state == "queued"
        assert running.state == "running"  # clearing never touches hardware

        cleared = queue.clear()  # everything queued
        assert {j.key.job_id for j in cleared} == {"b"}
        assert b.state == "cancelled"
        assert running.state == "running"
        Sleeper.gate.set()
        await wait_for_state(running, "completed")
        await queue.close()

    asyncio.run(scenario())


def test_queue_history_forgets_oldest_terminal_jobs() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine(), history_limit=2)
        queue.start()
        jobs = [queue.submit("c", f"j{i}", echo_graph(f"t{i}"), ["s"]) for i in range(4)]
        for job in jobs:
            await wait_for_state(job, "completed")
        # Only the two most recent terminal jobs remain queryable.
        assert queue.get("c", "j0") is None
        assert queue.get("c", "j1") is None
        assert queue.get("c", "j2") is not None
        assert queue.get("c", "j3") is not None
        assert queue.job_for_run(jobs[0].run_id) is None
        assert queue.job_for_run(jobs[3].run_id) is jobs[3]
        await queue.close()

    asyncio.run(scenario())


def test_new_job_ref_globally_unique_and_time_ordered() -> None:
    """jobRef contract (platform plan): server-assigned, globally unique,
    opaque; lexical order approximates submission order (ms-timestamp
    prefix + 80 random bits)."""
    from dinkster_server.queue import new_job_ref

    refs = [new_job_ref() for _ in range(1000)]
    assert len(set(refs)) == len(refs)
    assert all(len(r) == 32 and int(r, 16) >= 0 for r in refs)
    # Refs minted across a real clock tick sort after earlier ones.
    early = new_job_ref()
    time.sleep(0.002)
    assert new_job_ref() > early


def test_queue_resubmit_survives_predecessor_eviction() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine(), history_limit=1)
        queue.start()
        first = queue.submit("c", "same", echo_graph("one"), ["s"])
        await wait_for_state(first, "completed")
        first_run = first.run_id
        # Rerun under the same key: predecessor is retired outright, and its
        # stale history slot must not evict the live job later.
        second = queue.submit("c", "same", echo_graph("two"), ["s"])
        assert second.run_id != first_run
        assert queue.job_for_run(first_run) is None
        await wait_for_state(second, "completed")
        filler = queue.submit("c", "filler", echo_graph("f"), ["s"])
        await wait_for_state(filler, "completed")
        assert queue.job_for_run(first_run) is None
        assert queue.get("c", "filler") is filler
        assert queue.get("c", "same") is None  # evicted by filler, as the oldest
        await queue.close()

    asyncio.run(scenario())


def test_queue_failure_shapes() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine())
        queue.start()

        bad_type = Graph(nodes={"n": GraphNode("test.nope", {})})
        invalid = queue.submit("c", "invalid", bad_type, ["n"])
        await wait_for_state(invalid, "failed")
        assert invalid.error is not None
        assert invalid.error["kind"] == "validation"
        assert invalid.error["diagnostics"][0]["code"] == "unknown-node-type"

        boom = queue.submit(
            "c", "boom", Graph(nodes={"n": GraphNode("test.boom", {"tag": "t"})}), ["n"]
        )
        await wait_for_state(boom, "failed")
        assert boom.error is not None
        assert boom.error["kind"] == "execution"
        assert boom.error["nodeId"] == "n"
        assert "boom: t" in boom.error["message"]
        # Public error bodies are stable {kind, type, message, ...}. The
        # traceback rides along by DEFAULT (user directive 2026-07-25) as a
        # separate optional field, so clients can always render the message
        # alone. By default the payload is path-redacted: filesystem paths
        # in the traceback collapse to stable tokens, never absolute paths.
        raising_file = str(Path(__file__).resolve())
        assert boom.error["type"]
        assert "traceback" in boom.error
        assert raising_file not in boom.error["traceback"]
        assert "test_server.py" in boom.error["traceback"]
        assert "hints" not in boom.error

        shape_boom = queue.submit(
            "c",
            "shape-boom",
            Graph(nodes={"n": GraphNode("test.shape_boom", {"layout": "plain"})}),
            ["n"],
        )
        await wait_for_state(shape_boom, "failed")
        assert shape_boom.error is not None
        assert shape_boom.error["hints"] == [
            {
                "code": "tensor-shape-mismatch",
                "message": "Tensor sizes 64 and 32 do not match at dimension 2.",
            }
        ]
        assert "traceback" in shape_boom.error
        await queue.close()

        # debug_errors=True is the local-debugging escape hatch: error
        # payloads go out raw, absolute paths included.
        debug_queue = JobQueue(make_engine(), debug_errors=True)
        debug_queue.start()
        debug_boom = debug_queue.submit(
            "c", "boom", Graph(nodes={"n": GraphNode("test.boom", {"tag": "t"})}), ["n"]
        )
        await wait_for_state(debug_boom, "failed")
        assert debug_boom.error is not None
        assert debug_boom.error["kind"] == "execution"
        assert debug_boom.error["type"]
        assert "boom: t" in debug_boom.error["message"]
        assert "traceback" in debug_boom.error
        assert raising_file in debug_boom.error["traceback"]

        debug_shape_boom = debug_queue.submit(
            "c",
            "shape-boom",
            Graph(nodes={"n": GraphNode("test.shape_boom", {"layout": "permutation"})}),
            ["n"],
        )
        await wait_for_state(debug_shape_boom, "failed")
        assert debug_shape_boom.error is not None
        assert debug_shape_boom.error["hints"][0]["suggestion"].startswith(
            "Check for a channels-first versus channels-last"
        )
        assert "traceback" in debug_shape_boom.error
        await debug_queue.close()

    asyncio.run(scenario())


def test_queue_compiled_dispatch_digest_failure_and_raw_fingerprint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = make_engine()
        runtime = compiler_runtime(engine)
        compiled = CompiledGraph(echo_graph(), ("s",), runtime.extension_snapshot_digest, {})
        compile_calls = 0

        async def forbidden_compile(*_args: object, **_kwargs: object) -> None:
            nonlocal compile_calls
            compile_calls += 1
            raise AssertionError("queued compiled execution must never recompile")

        monkeypatch.setattr(engine, "compile_for_execution", forbidden_compile)
        queue = JobQueue(engine)
        queue.start()
        compiled_job = queue.submit(
            "c",
            "compiled",
            echo_graph(),
            ["s"],
            execution=runtime,
            compiled_graph=compiled,
        )
        await wait_for_state(compiled_job, "completed")
        assert compiled_job.result is not None
        assert compiled_job.result.outputs["s"]["out"].resolve() == "HELLO"

        drift = CompiledGraph(
            echo_graph(),
            ("s",),
            "sha256:" + "f" * 64,
            {},
        )
        drifted = queue.submit(
            "c",
            "drift",
            echo_graph(),
            ["s"],
            execution=runtime,
            compiled_graph=drift,
        )
        await wait_for_state(drifted, "failed")
        assert drifted.error == {
            "kind": "compile",
            "code": GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
            "message": "compiled graph generation does not match the execution runtime",
        }
        assert compile_calls == 0

        raw_engine = make_engine()
        raw_queue = JobQueue(raw_engine)
        raw_graph = echo_graph("raw")
        raw_targets = ["s"]
        raw = raw_queue.submit("c", "raw", raw_graph, raw_targets)
        expected = hashlib.sha256(
            json.dumps(
                {
                    "graph": graph_to_wire(raw_graph),
                    "targets": raw_targets,
                    "priority": 0,
                    "sourceDocument": "",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert raw.fingerprint == expected
        await raw_queue.close()
        await queue.close()

    asyncio.run(scenario())


def test_queue_close_cancels_pending_and_running() -> None:
    async def scenario() -> None:
        reset_sleeper()
        queue = JobQueue(make_engine(), max_running_jobs=1)
        queue.start()
        running = queue.submit(
            "c", "r", Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "r"})}), ["n"]
        )
        await Sleeper.entered.wait()
        pending = queue.submit("c", "p", echo_graph(), ["s"])
        await queue.close()
        assert running.state == "cancelled"
        assert pending.state == "cancelled"
        with pytest.raises(RuntimeError, match="closed"):
            queue.submit("c", "late", echo_graph(), ["s"])

    asyncio.run(scenario())


# -- HTTP/WS protocol ---------------------------------------------------------


async def make_client() -> TestClient:
    app = create_app(make_engine, SCHEMAS)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def wait_for_http_state(
    client: TestClient,
    client_id: str,
    job_id: str,
    expected: str,
) -> dict[str, Any]:
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/{client_id}/{job_id}")).json()
            if status["state"] == expected:
                return cast(dict[str, Any], status)
            if status["state"] in ("completed", "failed", "cancelled"):
                raise AssertionError(f"job reached {status['state']}, expected {expected}")
            await asyncio.sleep(0.005)


def test_health_endpoint() -> None:
    """GET /api/health answers 200 the moment the app exists: create_app
    runs after composition (it takes the merged schemas), so a 200 here IS
    readiness - the supervisor's engine protocol v1 probes exactly this."""

    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.get("/api/health")
            assert resp.status == 200
            assert await resp.json() == {"ok": True}
        finally:
            await client.close()

    asyncio.run(scenario())


class StubAuthenticator:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal
        self.tokens: list[str] = []

    async def authenticate(self, token: str) -> Principal | None:
        self.tokens.append(token)
        return self.principal if token == "accepted" else None


class MappingAuthenticator:
    def __init__(self, principals: Mapping[str, Principal]) -> None:
        self.principals = dict(principals)

    async def authenticate(self, token: str) -> Principal | None:
        return self.principals.get(token)


def test_auth_off_attaches_implicit_local_superuser_without_changing_responses() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)

        async def whoami(request: web.Request) -> web.Response:
            principal = principal_for(request)
            return web.json_response(
                {
                    "principalId": principal.principal_id,
                    "jobsSubmit": principal.allows("jobs:submit"),
                }
            )

        app.router.add_get("/api/whoami", whoami)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            health = await client.get("/api/health")
            assert health.status == 200
            assert await health.json() == {"ok": True}
            identity = await client.get("/api/whoami")
            assert identity.status == 200
            assert await identity.json() == {
                "principalId": "local",
                "jobsSubmit": True,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pluggable_authenticator_attaches_only_its_request_principal() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(
            Principal("alice", {"workspace-a": frozenset({"history:read"})})
        )
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)

        async def whoami(request: web.Request) -> web.Response:
            return web.json_response({"principalId": principal_for(request).principal_id})

        app.router.add_get("/api/history/whoami", whoami)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get(
                "/api/history/whoami", headers={"Authorization": "Bearer accepted"}
            )
            assert response.status == 200
            assert await response.json() == {"principalId": "alice"}
            assert authenticator.tokens == ["accepted"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_authentication_failures_are_structured_and_never_echo_tokens() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("alice", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers_cases = (
                {},
                {"Authorization": "Basic opaque"},
                {"Authorization": "Bearer unknown-secret"},
                {"Authorization": f"Bearer nonascii-{chr(233)}"},
            )
            for headers in headers_cases:
                response = await client.get("/api/nodes", headers=headers)
                assert response.status == 401
                assert response.headers["WWW-Authenticate"] == "Bearer"
                body = await response.json()
                assert body == {
                    "error": "authentication-required",
                    "message": "a valid Bearer credential is required",
                }
                assert "unknown-secret" not in await response.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_auth_policy_fails_closed_only_for_matched_unclassified_routes() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("alice", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)

        async def unclassified(_request: web.Request) -> web.Response:
            return web.json_response({"unsafe": True})

        app.router.add_get("/api/unclassified", unclassified)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer accepted"}
            response = await client.get("/api/unclassified", headers=headers)
            assert response.status == 403
            assert await response.json() == {
                "error": "authorization-policy-missing",
                "message": "this route is not classified by the capability policy",
            }
            assert (await client.get("/api/not-a-route", headers=headers)).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_missing_capability_is_403_for_every_route_family() -> None:
    routes = (
        ("POST", "/api/jobs", "jobs:submit"),
        ("GET", "/api/jobs", "jobs:read"),
        ("DELETE", "/api/jobs/client/job", "jobs:cancel"),
        ("GET", "/api/history", "history:read"),
        ("GET", "/api/assets/digest", "assets:read"),
        ("POST", "/api/assets", "assets:write"),
        ("GET", "/api/settings", "settings:read"),
        ("PUT", "/api/settings/jobs", "settings:write"),
        ("PUT", "/api/packs/missing/settings", "settings:write"),
        ("GET", "/api/p2p/status", "settings:read"),
        ("POST", f"/api/p2p/transfers/blake3:{'a' * 64}/pause", "settings:write"),
        ("GET", "/api/sessions", "sessions:read"),
        ("POST", "/api/sessions", "sessions:write"),
        ("GET", "/api/queue", "queue:control"),
        ("GET", "/memory/status", "memory:read"),
        ("GET", "/cache/entry/key", "cache:read"),
    )

    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for method, path, capability in routes:
                response = await client.request(
                    method, path, headers={"Authorization": "Bearer accepted"}
                )
                assert response.status == 403, (method, path, await response.text())
                assert await response.json() == {
                    "error": "capability-required",
                    "capability": capability,
                    "message": f"this route requires {capability}",
                }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_generation_routes_require_their_declared_capabilities() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)

        async def ok(_: web.Request) -> web.Response:
            return web.Response(status=204)

        app.router.add_get("/api/generation/models", ok)
        app.router.add_get("/v1/models", ok)
        protected = (
            ("POST", "/api/generation", "jobs:submit"),
            ("POST", "/api/generation/models/load", "settings:write"),
            ("POST", "/api/generation/models/unload", "settings:write"),
            ("DELETE", "/api/generation/sessions/session", "sessions:write"),
            ("POST", "/v1/completions", "jobs:submit"),
            ("POST", "/v1/chat/completions", "jobs:submit"),
            ("POST", "/v1/responses", "jobs:submit"),
        )
        for method, path, _ in protected:
            app.router.add_route(method, path, ok)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer accepted"}
            assert (await client.get("/api/generation/models", headers=headers)).status == 204
            assert (await client.get("/v1/models", headers=headers)).status == 204
            assert (await client.get("/v1/models")).status == 401
            for method, path, capability in protected:
                response = await client.request(method, path, headers=headers)
                assert response.status == 403
                assert (await response.json())["capability"] == capability
        finally:
            await client.close()

    asyncio.run(scenario())


def test_late_mounted_session_routes_use_read_and_write_capabilities() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(
            Principal("reader", {"scope": frozenset({"sessions:read"})})
        )
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        add_session_routes(
            app,
            SessionService(),
            principal_for=principal_for,
            resolve_scope=resolve_scope,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer accepted"}
            assert (await client.get("/api/sessions?scope=local", headers=headers)).status == 200
            assert (await client.head("/api/sessions?scope=local", headers=headers)).status == 200
            response = await client.post("/api/sessions", json={}, headers=headers)
            assert response.status == 403
            assert (await response.json())["capability"] == "sessions:write"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_catalog_is_open_to_any_authenticated_principal_and_health_is_public() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("catalog", {"scope": frozenset()}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            health = await client.get("/api/health")
            assert health.status == 200
            for path in (
                "/api/nodes",
                "/api/composition",
                "/api/diagnostics",
                "/api/choices/missing",
                "/api/templates",
                "/api/packs/missing/icon",
                "/api/packs/missing/settings",
                "/packs/missing/static/theme.css",
            ):
                response = await client.get(path, headers={"Authorization": "Bearer accepted"})
                assert response.status not in (401, 403), (path, await response.text())
        finally:
            await client.close()

    asyncio.run(scenario())


def test_websocket_authentication_is_rejected_before_upgrade() -> None:
    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset({"jobs:read"})}))
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as missing:
                await client.ws_connect("/api/events")
            assert missing.value.status == 401
            ws = await client.ws_connect(
                "/api/events", headers={"Authorization": "Bearer accepted"}
            )
            assert not ws.closed
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_ws_ticket_is_single_use_expiring_and_never_authenticates_plain_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        principal = Principal("alice", {"a": frozenset({"jobs:read"})})
        authenticator = MappingAuthenticator({"alice-token": principal})
        app = create_app(make_engine, SCHEMAS, authenticator=authenticator)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer alice-token"}
            minted = await client.post("/api/auth/ws-ticket", headers=headers)
            assert minted.status == 200
            payload = await minted.json()
            assert payload["expiresInSeconds"] == 30
            ticket = payload["ticket"]

            # Tickets are ignored by plain routes and are not consumed there.
            assert (await client.get(f"/api/nodes?ticket={ticket}")).status == 401
            ws = await client.ws_connect(f"/api/events?ticket={ticket}")
            await ws.close()
            with pytest.raises(aiohttp.WSServerHandshakeError) as reused:
                await client.ws_connect(f"/api/events?ticket={ticket}")
            assert reused.value.status == 401
            with pytest.raises(aiohttp.WSServerHandshakeError) as unknown:
                await client.ws_connect("/api/events?ticket=unknown")
            assert unknown.value.status == 401

            near_miss_ticket = await (
                await client.post("/api/auth/ws-ticket", headers=headers)
            ).json()
            with pytest.raises(aiohttp.WSServerHandshakeError) as near_miss:
                await client.ws_connect(
                    f"/api/sessions/a/extra/events?ticket={near_miss_ticket['ticket']}"
                )
            assert near_miss.value.status == 401
            ws = await client.ws_connect(f"/api/events?ticket={near_miss_ticket['ticket']}")
            await ws.close()

            monkeypatch.setattr("dinkster_server.auth.time.monotonic", lambda: 100.0)
            expiring = await (await client.post("/api/auth/ws-ticket", headers=headers)).json()
            monkeypatch.setattr("dinkster_server.auth.time.monotonic", lambda: 131.0)
            with pytest.raises(aiohttp.WSServerHandshakeError) as expired:
                await client.ws_connect(f"/api/events?ticket={expiring['ticket']}")
            assert expired.value.status == 401
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_scope_resolution_stamps_jobs_and_lists_readable_scope_union() -> None:
    async def scenario() -> None:
        submitter = Principal(
            "alice",
            {
                "a": frozenset({"jobs:submit", "jobs:read"}),
                "b": frozenset({"jobs:submit"}),
                "c": frozenset({"jobs:read"}),
            },
            kind="agent",
        )
        unique = Principal("unique", {"a": frozenset({"jobs:submit", "jobs:read"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"token": submitter, "unique": unique}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer token"}
            ambiguous = await client.post(
                "/api/jobs", json=submit_body(echo_graph(), ["s"]), headers=headers
            )
            assert ambiguous.status == 400
            assert (await ambiguous.json())["error"] == "scope-required"

            denied = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"], scope="c"),
                headers=headers,
            )
            assert denied.status == 403
            assert await denied.json() == {
                "error": "capability-required",
                "capability": "jobs:submit",
                "scope": "c",
                "message": "scope c requires jobs:submit",
            }

            accepted = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"], scope="a"),
                headers=headers,
            )
            wire = await accepted.json()
            assert accepted.status == 202
            assert wire["scope"] == "a"
            assert wire["submittedBy"] == {"principalId": "alice", "kind": "agent"}
            state = app[STATE_KEY]
            job = state.queue.get("c1", "j1")
            assert job is not None
            assert (job.scope, job.principal_id, job.principal_kind) == ("a", "alice", "agent")
            unique_response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph("unique"), ["s"], clientId="unique", jobId="j2"),
                headers={"Authorization": "Bearer unique"},
            )
            assert unique_response.status == 202
            assert (await unique_response.json())["scope"] == "a"
            state.queue.submit("c2", "hidden", echo_graph("hidden"), ["s"], scope="b")
            state.queue.submit("c3", "visible", echo_graph("visible"), ["s"], scope="c")
            listed = await (await client.get("/api/jobs", headers=headers)).json()
            assert {(job["clientId"], job["scope"]) for job in listed["jobs"]} == {
                ("c1", "a"),
                ("c3", "c"),
                ("unique", "a"),
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_auth_off_submission_stays_local_and_collab_keeps_explicit_scopes() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        add_session_routes(
            app,
            SessionService(),
            principal_for=principal_for,
            resolve_scope=resolve_scope,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            submitted = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"], scope="ignored"),
            )
            assert submitted.status == 202
            submitted_wire = await submitted.json()
            assert submitted_wire["scope"] == "local"
            assert submitted_wire["submittedBy"] == {
                "principalId": "local",
                "kind": "human",
            }

            created = await client.post(
                "/api/sessions",
                json={"scope": "shared", "documentId": "doc", "snapshot": None},
            )
            session = await created.json()
            assert created.status == 201
            assert session["scope"] == "shared"
            assert (await client.get(f"/api/sessions/{session['sessionId']}")).status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cross_scope_job_resources_are_all_hidden_as_existing_404_shapes() -> None:
    async def scenario() -> None:
        principal = Principal("alice", {"a": frozenset({"jobs:read", "jobs:cancel"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"token": principal}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            hidden = app[STATE_KEY].queue.submit(
                "other", "job", echo_graph(), ["s"], scope="b", principal_id="bob"
            )
            headers = {"Authorization": "Bearer token"}
            assert (await client.get("/api/jobs/other/job", headers=headers)).status == 404
            assert (
                await client.get(f"/api/jobs/by-ref/{hidden.run_id}", headers=headers)
            ).status == 404
            assert (
                await client.get(f"/api/jobs/by-ref/{hidden.run_id}/events", headers=headers)
            ).status == 404
            value = await client.get(
                "/api/values?clientId=other&jobId=job&nodeId=s&outputId=out",
                headers=headers,
            )
            assert value.status == 404
            assert (await value.json())["reason"] == "unknown-job"
            assert (await client.delete("/api/jobs/other/job", headers=headers)).status == 404
            assert (
                await client.delete(f"/api/jobs/by-ref/{hidden.run_id}", headers=headers)
            ).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_websocket_events_filter_job_scopes_but_broadcast_non_job_frames() -> None:
    async def receive_probes(
        websocket: aiohttp.ClientWebSocketResponse, own: str, hidden: str
    ) -> list[str]:
        received: list[str] = []
        while len(received) < 2:
            event_type = (await websocket.receive_json())["type"]
            assert event_type != hidden
            if event_type in {own, "global-probe"}:
                received.append(event_type)
        return received

    async def scenario() -> None:
        alice = Principal("alice", {"a": frozenset({"jobs:read"})})
        bob = Principal("bob", {"b": frozenset({"jobs:read"})})
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"alice": alice, "bob": bob}),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            job_a = app[STATE_KEY].queue.submit("ca", "ja", echo_graph("a"), ["s"], scope="a")
            job_b = app[STATE_KEY].queue.submit("cb", "jb", echo_graph("b"), ["s"], scope="b")
            ws_a = await client.ws_connect("/api/events", headers={"Authorization": "Bearer alice"})
            ws_b = await client.ws_connect("/api/events", headers={"Authorization": "Bearer bob"})
            app[STATE_KEY].hub.publish(
                {"type": "probe-a", "jobRef": job_a.run_id},
                client_id="ca",
                droppable=False,
            )
            app[STATE_KEY].hub.publish(
                {"type": "probe-b", "jobRef": job_b.run_id},
                client_id="cb",
                droppable=False,
            )
            app[STATE_KEY].hub.publish({"type": "global-probe"}, client_id=None, droppable=False)
            async with asyncio.timeout(2):
                assert await receive_probes(ws_a, "probe-a", "probe-b") == [
                    "probe-a",
                    "global-probe",
                ]
                assert await receive_probes(ws_b, "probe-b", "probe-a") == [
                    "probe-b",
                    "global-probe",
                ]
            await ws_a.close()
            await ws_b.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_session_scope_binding_and_ticket_authenticated_event_stream() -> None:
    async def scenario() -> None:
        alice = Principal("alice", {"a": frozenset({"sessions:read", "sessions:write"})})
        service = SessionService()
        hidden = service.create(scope="b", document_id="hidden", snapshot=None)
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=MappingAuthenticator({"alice": alice}),
        )
        add_session_routes(
            app,
            service,
            principal_for=principal_for,
            resolve_scope=resolve_scope,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer alice"}
            denied = await client.post(
                "/api/sessions",
                json={"scope": "b", "documentId": "denied", "snapshot": None},
                headers=headers,
            )
            assert denied.status == 403
            created_response = await client.post(
                "/api/sessions",
                json={"documentId": "visible", "snapshot": None},
                headers=headers,
            )
            created = await created_response.json()
            assert created_response.status == 201
            assert created["scope"] == "a"
            listed = await (await client.get("/api/sessions", headers=headers)).json()
            assert [session["sessionId"] for session in listed["sessions"]] == [
                created["sessionId"]
            ]
            for method, suffix in (
                ("get", ""),
                ("get", "/ops?after=0"),
                ("get", "/snapshot"),
                ("delete", ""),
            ):
                response = await getattr(client, method)(
                    f"/api/sessions/{hidden.session_id}{suffix}", headers=headers
                )
                assert response.status == 404

            ticket = await (await client.post("/api/auth/ws-ticket", headers=headers)).json()
            ws = await client.ws_connect(
                f"/api/sessions/{created['sessionId']}/events?ticket={ticket['ticket']}"
            )
            hello = await ws.receive_json()
            assert hello["type"] == "session"
            await ws.close()
            with pytest.raises(aiohttp.WSServerHandshakeError) as hidden_ws:
                await client.ws_connect(
                    f"/api/sessions/{hidden.session_id}/events", headers=headers
                )
            assert hidden_ws.value.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_static_auth_file_loads_frozen_scoped_grants(tmp_path: Path) -> None:
    path = tmp_path / "auth.toml"
    path.write_text(
        """version = 1

[[tokens]]
token = "operator-secret"
principalId = "alice"
[tokens.grants]
workspace = ["jobs:read", "assets:read"]
personal = ["jobs:submit"]
""",
        encoding="utf-8",
    )
    authenticator = load_authenticator(path)

    async def scenario() -> None:
        principal = await authenticator.authenticate("operator-secret")
        assert principal is not None
        assert principal.principal_id == "alice"
        assert principal.allows("jobs:submit")
        assert principal.allows("jobs:read")
        assert await authenticator.authenticate("wrong") is None
        with pytest.raises(TypeError):
            principal.grants["other"] = frozenset()  # type: ignore[index]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    (
        "not toml =",
        "version = 2\ntokens = []\n",
        "version = true\ntokens = []\n",
        "version = 1\ntokens = []\n",
        'version = 1\n[[tokens]]\ntoken = " "\nprincipalId = "p"\n[tokens.grants]\nscope = []\n',
        f'version = 1\n[[tokens]]\ntoken = "{chr(233)}"\nprincipalId = "p"\n'
        "[tokens.grants]\nscope = []\n",
        'version = 1\n[[tokens]]\ntoken = "x"\nprincipalId = "p"\n'
        '[tokens.grants]\nscope = ["not:a-capability"]\n',
        'version = 1\n[[tokens]]\ntoken = "x"\nprincipalId = "p"\n'
        '[tokens.grants]\nscope = ["jobs:read", "jobs:read"]\n',
    ),
)
def test_static_auth_file_refuses_malformed_config(tmp_path: Path, body: str) -> None:
    path = tmp_path / "auth.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(AuthError, match="auth.toml"):
        load_authenticator(path)


def test_cors_disabled_by_default() -> None:
    """No --allow-origin means NO CORS headers, ever: a random website's
    scripts must not be able to read responses from a local instance."""

    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.get("/api/health", headers={"Origin": "https://evil.example"})
            assert resp.status == 200
            assert "Access-Control-Allow-Origin" not in resp.headers
            # Preflights fall through to normal routing (no OPTIONS route).
            resp = await client.options(
                "/api/jobs",
                headers={
                    "Origin": "https://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "Access-Control-Allow-Origin" not in resp.headers
        finally:
            await client.close()

    asyncio.run(scenario())


def test_browser_boundary_accepts_loopback_and_rejects_foreign_host() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_hosts=["ui.example", "::1"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            accepted = await client.get("/api/health", headers={"Host": "127.0.0.2:3639"})
            assert accepted.status == 200
            configured = await client.get("/api/health", headers={"Host": "ui.example:3639"})
            assert configured.status == 200

            rejected = await client.get("/api/health", headers={"Host": "attacker.example"})
            assert rejected.status == 421
            assert await rejected.json() == {
                "error": "host-not-allowed",
                "message": "the request Host is not allowed by this server",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_browser_boundary_preserves_same_origin_development_proxy() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            response = await client.post(
                "/api/jobs",
                json={},
                headers={
                    "Origin": "http://dev-machine:5199",
                    "Sec-Fetch-Site": "same-origin",
                },
            )
            assert response.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_browser_boundary_rejects_foreign_origin_on_state_change() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_origins=["https://ui.example"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            rejected = await client.post(
                "/api/jobs",
                json={},
                headers={"Origin": "https://attacker.example"},
            )
            assert rejected.status == 403
            assert await rejected.json() == {
                "error": "origin-not-allowed",
                "message": "the request Origin is not allowed for this operation",
            }

            allowed = await client.post(
                "/api/jobs",
                json={},
                headers={"Origin": "https://ui.example"},
            )
            assert allowed.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_browser_boundary_rejects_websocket_upgrade_with_foreign_host() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as rejected:
                await client.ws_connect("/api/events", headers={"Host": "attacker.example"})
            assert rejected.value.status == 421
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cors_exact_origin() -> None:
    """A configured exact origin is echoed with Vary: Origin and ETag
    exposed; reads from other origins get no CORS response headers."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_origins=["https://ui.example"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/api/health", headers={"Origin": "https://ui.example"})
            assert resp.headers["Access-Control-Allow-Origin"] == "https://ui.example"
            assert resp.headers["Access-Control-Expose-Headers"] == "ETag"
            assert "Origin" in resp.headers.getall("Vary", [])

            resp = await client.get("/api/health", headers={"Origin": "https://other.example"})
            assert resp.status == 200
            assert "Access-Control-Allow-Origin" not in resp.headers

            # No Origin header -> no CORS headers (same-origin/cli traffic).
            resp = await client.get("/api/health")
            assert "Access-Control-Allow-Origin" not in resp.headers

            # Preflight: answered locally, 204, permissions echoed.
            resp = await client.options(
                "/api/jobs",
                headers={
                    "Origin": "https://ui.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == "https://ui.example"
            assert resp.headers["Access-Control-Allow-Methods"] == "POST"
            assert resp.headers["Access-Control-Allow-Headers"] == "content-type"

            # Preflight from an unconfigured origin is not granted.
            resp = await client.options(
                "/api/jobs",
                headers={
                    "Origin": "https://other.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "Access-Control-Allow-Origin" not in resp.headers
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cors_wildcard() -> None:
    """'*' allows any origin; the literal '*' is sent (no origin echo, no
    Vary: Origin needed)."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, allow_origins=["*"])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/api/health", headers={"Origin": "https://anything.example"})
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
            assert "Origin" not in resp.headers.getall("Vary", [])
        finally:
            await client.close()

    asyncio.run(scenario())


def submit_body(graph: Graph, targets: list[str], **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": targets,
    }
    body.update(overrides)
    return body


def test_nodes_endpoint_serves_native_schema_wire() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.get("/api/nodes")
            assert resp.status == 200
            data = await resp.json()
            # Top-level schemaVersion is the API SURFACE version (currently
            # 1) - NOT the schema wire version, which is served as
            # dinkster.schemaWire plus per node entry. Frontend decodes
            # dinkster.schemaWire and validates per-entry, so these three
            # assertions pin the shipped shape.
            assert data["schemaVersion"] == 1
            # Surface epoch: monotonic node-surface generation, constant 1
            # until progressive announcement/hot-reload bump it. Distinct
            # from both version fields by frontend contract.
            assert data["epoch"] == 1
            snapshot = await (await client.get("/api/extensions/snapshot")).json()
            assert snapshot["frontendApi"] == "1.0.0"
            assert snapshot["extensions"] == []
            canonical = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"))
            assert data["extensionSnapshotDigest"] == (
                "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            )
            # Environment header: authoritative runtime facts for workflow
            # environment stamping (advisory record, never identity).
            assert isinstance(data["dinkster"]["version"], str)
            assert data["dinkster"]["version"]
            assert data["dinkster"]["schemaWire"] == SCHEMA_WIRE_VERSION
            assert SCHEMA_WIRE_VERSION == 1
            # Graph/job DOCUMENT wire capabilities, additive feature list
            # decoupled from the schema wire (joint contract 2026-07-25):
            # clients gate optional emission forms on membership.
            assert data["dinkster"]["graphFeatures"] == [
                "typedLiteral",
                "decimalInt",
                "regions",
                "placement",
            ]
            # mergeableTypes (joint contract 2026-07-26): ALWAYS present,
            # the sorted atom ids with a registered batch-merge provider.
            # This app's registry registers none, so the list is empty -
            # empty means "no scalar-T merge arm anywhere", never omitted.
            assert data["dinkster"]["mergeableTypes"] == []
            # Every node schema entry carries the wire version itself.
            for entry in data["nodes"].values():
                assert entry["schemaVersion"] == SCHEMA_WIRE_VERSION
            echo = data["nodes"]["test.echo"]
            assert echo["nodeType"] == "test.echo"
            # Provenance defaults: the packs table always carries the
            # reserved "core" entry, and the per-node pack key is always
            # present (frontend contract: never omitted, core included).
            assert data["packs"]["core"]["displayName"] == "Dinkster Core"
            assert all(entry["pack"] == "core" for entry in data["nodes"].values())
            # Interface identity: EVERY node entry carries an always-present
            # signature (stamping never has holes), equal to the schema's
            # authoritative signature and separate from the schema wire
            # itself (transport metadata, never a schema field).
            for type_id, entry in data["nodes"].items():
                assert entry["signature"] == schema_signature(SCHEMAS[type_id])
            assert "signature" not in schema_to_wire(SCHEMAS["test.echo"])
        finally:
            await client.close()

    asyncio.run(scenario())


def test_nodes_endpoint_does_not_negotiate_schema_wire_versions() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            for query in ("", "?wire=1", "?wire=invalid"):
                response = await client.get(f"/api/nodes{query}")
                assert response.status == 200
                data = await response.json()
                assert data["dinkster"]["schemaWire"] == 1
                assert all(entry["schemaVersion"] == 1 for entry in data["nodes"].values())
        finally:
            await client.close()

    asyncio.run(scenario())


def test_workers_endpoint_shape_and_authentication() -> None:
    workers = (
        WorkerInfo("local", "connected", ("test.echo",), ()),
        WorkerInfo("gpu-box", "configured", ("test.shout",), ("@gpu-box",)),
    )

    async def scenario() -> None:
        authenticator = StubAuthenticator(Principal("reader", {"scope": frozenset()}))
        app = create_app(
            make_engine,
            SCHEMAS,
            authenticator=authenticator,
            workers=lambda: workers,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.get("/api/workers")).status == 401
            response = await client.get(
                "/api/workers", headers={"Authorization": "Bearer accepted"}
            )
            assert response.status == 200
            assert await response.json() == {
                "workers": [
                    {
                        "name": "local",
                        "status": "connected",
                        "routedNodeTypes": ["test.echo"],
                        "deviceQualifiers": [],
                    },
                    {
                        "name": "gpu-box",
                        "status": "configured",
                        "routedNodeTypes": ["test.shout"],
                        "deviceQualifiers": ["@gpu-box"],
                    },
                ]
            }
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("placement", "message"),
    (
        ({"e": "missing"}, "placement hint 'e' names unknown worker 'missing'"),
        ({"missing": "local"}, "placement hint 'missing' names unknown top-level node id"),
        (
            {"region/body": "local"},
            "placement hint 'region/body' is invalid: keys must be top-level node ids without '/'",
        ),
        (
            {"e": "gpu-box"},
            "placement hint 'e' names worker 'gpu-box', which cannot execute node type 'test.echo'",
        ),
    ),
)
def test_job_placement_validation_errors(
    placement: dict[str, str],
    message: str,
) -> None:
    async def scenario() -> None:
        workers = (
            WorkerInfo("local", "connected", tuple(sorted(SCHEMAS))),
            WorkerInfo("gpu-box", "connected", ("test.shout",), ("@gpu-box",)),
        )
        app = create_app(make_engine, SCHEMAS, workers=lambda: workers)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hello"})})
            response = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["e"], placement=placement),
            )
            assert response.status == 400
            assert await response.json() == {"error": message}
            assert app[STATE_KEY].queue.jobs() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_submission_migrates_retired_node_type_before_placement() -> None:
    retired_type = "comfy.CreateHookKeyframe"
    canonical = replace(
        SCHEMAS["test.echo"],
        replacements=(
            ReplacementRule(
                from_type=retired_type,
                cases=(
                    ReplacementCase.build(
                        "test.echo",
                        inputs={"text": MappingSource.copy("text")},
                        outputs={"out": "out"},
                    ),
                ),
            ),
        ),
    )
    schemas = {**SCHEMAS, canonical.node_type: canonical}

    def make_replacement_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(make_replacement_engine, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            stored = Graph({"keyframe": GraphNode(retired_type, {"text": "legacy"})})
            response = await client.post(
                "/api/jobs",
                json=submit_body(
                    stored,
                    ["keyframe"],
                    placement={"keyframe": "local"},
                ),
            )

            assert response.status == 202
            await wait_for_http_state(client, "c1", "j1", "completed")
            job = app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None
            assert job.graph.nodes["keyframe"] == GraphNode("test.echo", {"text": "legacy"})
            assert job.result is not None
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("node_id", "message"),
    (
        (
            "pick",
            "placement hint 'pick' names a selector node removed by selector lowering",
        ),
        (
            "inactive",
            "placement hint 'inactive' names a node that was pruned by selector lowering",
        ),
    ),
)
def test_job_placement_rejects_nodes_removed_by_selector_lowering(
    node_id: str,
    message: str,
) -> None:
    selector_schema = NodeSchema(
        "test.placement_selector",
        inputs=(
            InputSpec("switch", TypeExpr.concrete("core.boolean")),
            InputSpec("off", STRING),
            InputSpec("on", STRING),
        ),
        outputs=(OutputSpec("out", STRING),),
        selector=SelectorSpec("switch", {"false": "off", "true": "on"}),
    )
    schemas = {**SCHEMAS, selector_schema.node_type: selector_schema}

    def make_selector_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(make_selector_engine, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(
                nodes={
                    "inactive": GraphNode("test.echo", {"text": "unused"}),
                    "pick": GraphNode(
                        selector_schema.node_type,
                        {
                            "switch": True,
                            "off": Link("inactive", "out"),
                            "on": "selected",
                        },
                    ),
                    "sink": GraphNode("test.shout", {"text": Link("pick", "out")}),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["sink"], placement={node_id: "local"}),
            )
            assert response.status == 400
            assert await response.json() == {"error": message}
            assert app[STATE_KEY].queue.jobs() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_placement_rejects_remote_lazy_node() -> None:
    lazy_schema = NodeSchema(
        "test.lazy_placement",
        inputs=(InputSpec("value", STRING, lazy=True),),
        outputs=(OutputSpec("out", STRING),),
    )
    schemas = {**SCHEMAS, lazy_schema.node_type: lazy_schema}
    workers = (
        WorkerInfo("local", "connected", tuple(sorted(schemas))),
        WorkerInfo("gpu-box", "connected", (lazy_schema.node_type,), ("@gpu-box",)),
    )

    async def scenario() -> None:
        app = create_app(make_engine, schemas, workers=lambda: workers)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"lazy": GraphNode(lazy_schema.node_type, {"value": "ready"})})
            response = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["lazy"], placement={"lazy": "gpu-box"}),
            )
            assert response.status == 400
            assert await response.json() == {
                "error": (
                    "placement hint 'lazy' names worker 'gpu-box', but remote placement "
                    "does not support lazy node type 'test.lazy_placement'"
                )
            }
            assert app[STATE_KEY].queue.jobs() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_local_placement_attribution_and_unhinted_compatibility() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hello"})})
            placed = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["e"], placement={"e": "local"}),
            )
            assert placed.status == 202
            placed_wire = await placed.json()
            await wait_for_http_state(client, "c1", "j1", "completed")
            placed_job = app[STATE_KEY].queue.get("c1", "j1")
            assert placed_job is not None
            assert placed_job.node_receipts["e"]["worker"] == "local"
            replay = await (
                await client.get(f"/api/jobs/by-ref/{placed_wire['jobRef']}/events?after=0")
            ).json()
            finished = next(event for event in replay["events"] if event["type"] == "node_finished")
            assert finished["detail"]["worker"] == "local"

            unhinted = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["e"],
                    clientId="c2",
                    jobId="j2",
                ),
            )
            assert unhinted.status == 202
            await wait_for_http_state(client, "c2", "j2", "completed")
            unhinted_job = app[STATE_KEY].queue.get("c2", "j2")
            assert unhinted_job is not None
            assert "worker" not in unhinted_job.node_receipts["e"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_provider_resolution_failure_is_actionable_before_queue() -> None:
    def make_resolving_engine(on_event: EventListener | None = None) -> Engine:
        engine = make_engine(on_event)

        def resolve(_graph: Graph) -> Graph:
            raise ValueError(
                "test.vision has no compatible live vision implementation for an automatic "
                "model; install or connect a compatible vision pack"
            )

        runtime = replace(engine.pin_execution(), resolve_providers=resolve)
        engine._pin_execution = lambda: runtime
        return engine

    async def scenario() -> None:
        app = create_app(make_resolving_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"]),
            )
            assert response.status == 400
            assert await response.json() == {
                "error": "vision-provider-unavailable",
                "message": (
                    "test.vision has no compatible live vision implementation for an "
                    "automatic model; install or connect a compatible vision pack"
                ),
            }
            assert app[STATE_KEY].queue.jobs() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_provider_resolution_failure_identifies_the_responsible_node() -> None:
    def make_resolving_engine(on_event: EventListener | None = None) -> Engine:
        engine = make_engine(on_event)

        def resolve(_graph: Graph) -> Graph:
            raise ProviderResolutionError(
                node_id="realistic-line-art",
                node_type="dinkster.preprocess.lineart_realistic",
                title="Preprocess Realistic Line Art",
                capability="a compatible vision-processing implementation",
                remedy="Install or reconnect the standard vision components, then retry.",
            )

        runtime = replace(engine.pin_execution(), resolve_providers=resolve)
        engine._pin_execution = lambda: runtime
        return engine

    async def scenario() -> None:
        app = create_app(make_resolving_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"]),
            )
            assert response.status == 400
            assert await response.json() == {
                "error": "capability-unavailable",
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "capability-unavailable",
                        "message": (
                            "Preprocess Realistic Line Art cannot run because a compatible "
                            "vision-processing implementation is unavailable on this server. "
                            "Install or reconnect the standard vision components, then retry."
                        ),
                        "nodeId": "realistic-line-art",
                        "nodeType": "dinkster.preprocess.lineart_realistic",
                        "title": "Preprocess Realistic Line Art",
                        "capability": "a compatible vision-processing implementation",
                        "remedy": (
                            "Install or reconnect the standard vision components, then retry."
                        ),
                    }
                ],
            }
            assert app[STATE_KEY].queue.jobs() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_provider_resolution_is_transient_in_queued_graph() -> None:
    def make_resolving_engine(on_event: EventListener | None = None) -> Engine:
        engine = make_engine(on_event)

        def resolve(graph: Graph) -> Graph:
            echo = cast("GraphNode", graph.nodes["e"])
            return Graph(
                {
                    **graph.nodes,
                    "e": replace(echo, inputs={**echo.inputs, "text": "resolved"}),
                }
            )

        runtime = replace(engine.pin_execution(), resolve_providers=resolve)
        engine._pin_execution = lambda: runtime
        return engine

    async def scenario() -> None:
        app = create_app(make_resolving_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph(), ["s"]),
            )
            assert response.status == 202
            await wait_for_http_state(client, "c1", "j1", "completed")
            job = app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None and job.result is not None
            echo = cast("GraphNode", job.graph.nodes["e"])
            assert echo.inputs["text"] == "hello"
            assert job.result.outputs["s"]["out"].resolve() == "RESOLVED"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_placement_participates_in_job_idempotency_only() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            app[STATE_KEY].queue.pause()
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hello"})})
            first = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["e"], placement={"e": "local"}),
            )
            assert first.status == 202
            duplicate = await client.post(
                "/api/jobs",
                json=submit_body(graph, ["e"], placement={"e": "local"}),
            )
            assert duplicate.status == 202
            assert (await duplicate.json())["duplicate"] is True
            changed = await client.post("/api/jobs", json=submit_body(graph, ["e"]))
            assert changed.status == 409
        finally:
            await client.close()

    asyncio.run(scenario())


def test_nodes_envelope_lists_mergeable_types_sorted() -> None:
    """mergeableTypes derives from the composed registry at ENVELOPE BUILD
    time, not server construction (joint contract 2026-07-26): every atom
    with a registered batch-merge provider appears, sorted; atoms without
    one do not. Registrations landing after startup (how progressive
    composition adds compat-pack providers - the live counterpart is
    comfy.IMAGE via dinkster.image-batch-merge@1) must show up on the next
    fetch, so the test registers between two fetches."""

    registry = TypeRegistry()
    register_core_types(registry)

    def merging_engine(on_event: EventListener | None = None) -> Engine:
        return Engine(
            schemas=SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(merging_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/api/nodes")
            assert resp.status == 200
            data = await resp.json()
            assert data["dinkster"]["mergeableTypes"] == []
            # Post-startup registration, deliberately in non-sorted order
            # to pin the sort.
            registry.register_batch_merge(
                "core.string",
                provider_id="test.string-merge@1",
                merge=lambda vs: vs[0],
            )
            registry.register_batch_merge(
                "core.int", provider_id="test.int-merge@1", merge=lambda vs: vs[0]
            )
            resp = await client.get("/api/nodes")
            assert resp.status == 200
            data = await resp.json()
            assert data["dinkster"]["mergeableTypes"] == ["core.int", "core.string"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_nodes_endpoint_pack_provenance() -> None:
    """Attribution is attached by the host at collection (a schema never
    claims its own pack); presentation is author-declared badge data,
    optional fields omitted from the wire when undeclared."""

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={
                "ade": PackInfo(
                    display_name="AnimateDiff-Evolved",
                    abbr="ADE",
                    mark="\N{PERFORMING ARTS}",
                    color="#8844ff",
                ),
                "plain": PackInfo(display_name="Plain Pack"),
            },
            node_packs={"test.echo": "ade", "test.shout": "plain"},
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            data = await (await client.get("/api/nodes")).json()
            assert data["packs"]["ade"] == {
                "displayName": "AnimateDiff-Evolved",
                "abbr": "ADE",
                "mark": "\N{PERFORMING ARTS}",
                "color": "#8844ff",
            }
            assert data["packs"]["plain"] == {"displayName": "Plain Pack"}
            assert data["nodes"]["test.echo"]["pack"] == "ade"
            assert data["nodes"]["test.shout"]["pack"] == "plain"
            assert data["nodes"]["test.sleeper"]["pack"] == "core"
            # Provenance is publication metadata, never schema identity: the
            # announced schema wire itself carries no pack claim.
            assert "pack" not in schema_to_wire(SCHEMAS["test.echo"])
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_provenance_pins_on_the_wire() -> None:
    """Install provenance rides the packs table with exact camelCase names;
    every unknown field is omitted, never null - omission MEANS unpinned
    (frontend renders an explicit 'unpinned' badge from absence)."""
    pinned = PackInfo(
        display_name="Pinned",
        version="1.2.3",
        artifact_digest="sha256:" + "ab" * 32,
        source="registry",
        publisher="alice",
    )
    assert pinned.to_wire() == {
        "displayName": "Pinned",
        "version": "1.2.3",
        "artifactDigest": "sha256:" + "ab" * 32,
        "source": "registry",
        "publisher": "alice",
    }
    unpinned = PackInfo(display_name="Dev", source="local:/somewhere/pack")
    assert unpinned.to_wire() == {
        "displayName": "Dev",
        "source": "local:/somewhere/pack",
    }

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={"pinned": pinned, "dev": unpinned},
            node_packs={"test.echo": "pinned", "test.shout": "dev"},
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            data = await (await client.get("/api/nodes")).json()
            assert data["packs"]["pinned"]["version"] == "1.2.3"
            assert data["packs"]["pinned"]["artifactDigest"] == "sha256:" + "ab" * 32
            assert data["packs"]["pinned"]["publisher"] == "alice"
            assert "version" not in data["packs"]["dev"]
            assert "artifactDigest" not in data["packs"]["dev"]
            # None never appears: absence is the only unpinned encoding.
            assert None not in data["packs"]["dev"].values()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_comfy_alias_registry_is_dedicated_wire_metadata() -> None:
    registry = alias_registry()
    info = PackInfo(display_name="Native", comfy_aliases=registry)
    pack_wire = info.to_wire()
    assert pack_wire["comfyAliases"]["format"] == "dinkster-comfy-alias/1"  # type: ignore[index]
    assert pack_wire["comfyAliases"]["records"][0]["carrier"] == "test.echo"  # type: ignore[index]
    assert info.to_wire(schemas=SCHEMAS) == pack_wire
    assert SCHEMAS["test.echo"].replacements == ()

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={"native": info},
            node_packs={"test.echo": "native"},
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            wire = await (await client.get("/api/nodes")).json()
            assert wire["packs"]["native"]["comfyAliases"] == pack_wire["comfyAliases"]
            assert "replacements" not in wire["nodes"]["test.echo"]
            assert "comfy.Echo" not in wire["nodes"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_comfy_group_registry_is_dedicated_wire_metadata() -> None:
    from test_comfy_group_registry import TARGET, registry

    group_registry = registry()
    schemas = {**SCHEMAS, TARGET.node_type: TARGET}
    info = PackInfo(display_name="Native", comfy_groups=group_registry)
    pack_wire = info.to_wire()
    assert pack_wire["comfyGroups"]["format"] == "dinkster-comfy-group/1"  # type: ignore[index]
    assert pack_wire["comfyGroups"]["records"][0]["carrier"] == TARGET.node_type  # type: ignore[index]
    assert info.to_wire(schemas=schemas) == pack_wire
    assert TARGET.replacements == ()

    app = create_app(
        make_engine,
        schemas,
        packs={"native": info},
        node_packs={TARGET.node_type: "native"},
    )
    state = app[STATE_KEY]
    assert state.packs["native"].comfy_groups == group_registry
    assert group_registry.records[0].pattern.group_type not in state.schemas
    for snapshot in group_registry.source_schemas:
        assert snapshot.schema.node_type not in state.schemas


def test_server_rejects_comfy_alias_ownership_reference_and_cross_pack_collisions() -> None:
    with pytest.raises(ValueError, match="is not owned by pack"):
        create_app(
            make_engine,
            SCHEMAS,
            packs={
                "native": PackInfo("Native", comfy_aliases=alias_registry()),
                "other": PackInfo("Other"),
            },
            node_packs={"test.echo": "other"},
        )

    with pytest.raises(ValueError, match="not a static input id"):
        create_app(
            make_engine,
            SCHEMAS,
            packs={
                "native": PackInfo(
                    "Native",
                    comfy_aliases=alias_registry(target_input="missing"),
                )
            },
            node_packs={"test.echo": "native"},
        )

    duplicated = PackInfo("Duplicate", comfy_aliases=alias_registry())
    with pytest.raises(ValueError, match="collide on comfy alias record id"):
        create_app(
            make_engine,
            SCHEMAS,
            packs={"native-a": duplicated, "native-b": duplicated},
            node_packs={"test.echo": "native-a"},
        )


def test_server_rejects_comfy_group_ownership_reference_and_cross_pack_collisions() -> None:
    from test_comfy_group_registry import TARGET, registry

    with pytest.raises(ValueError, match="is not owned by pack"):
        create_app(
            make_engine,
            {**SCHEMAS, TARGET.node_type: TARGET},
            packs={
                "native": PackInfo("Native", comfy_groups=registry()),
                "other": PackInfo("Other"),
            },
            node_packs={TARGET.node_type: "other"},
        )

    duplicated = PackInfo("Duplicate", comfy_groups=registry())
    with pytest.raises(ValueError, match="collide on comfy group record id"):
        create_app(
            make_engine,
            {**SCHEMAS, TARGET.node_type: TARGET},
            packs={"native-a": duplicated, "native-b": duplicated},
            node_packs={TARGET.node_type: "native-a"},
        )


def test_replace_defers_alias_carrier_checks_until_publication() -> None:
    """A pack's alias registry may land before its provider-gated schema-only
    carriers publish (the generation schema owner composes before the compat
    worker that executes its nodes). Ownership is enforced on the replace
    that publishes the carrier."""
    aliases = alias_registry(carrier="late.echo")
    app = create_app(make_engine, SCHEMAS)
    state = app[STATE_KEY]
    assert state.replace((), (), {}, {"native": PackInfo("Native", comfy_aliases=aliases)}, {}) == 2
    late = build_schemas([Late])
    with pytest.raises(ValueError, match="is not owned by pack"):
        state.replace((), (), late, {"other": PackInfo("Other")}, {"late.echo": "other"})
    assert "late.echo" not in state.schemas
    assert state.replace((), (), late, {}, {"late.echo": "native"}) == 3
    assert state.schemas["late.echo"] is late["late.echo"]


@pytest.mark.parametrize("publication", ["replace", "generation"])
def test_prepared_surface_validation_keeps_requests_responsive(
    monkeypatch: pytest.MonkeyPatch, publication: str
) -> None:
    import threading

    import dinkster_server.app as app_module

    original_validate = app_module._validate_comfy_registry_packs
    validation_started = threading.Event()
    release_validation = threading.Event()
    app = create_app(make_engine, SCHEMAS)
    state = app[STATE_KEY]
    state.seed_composition(["native"])
    state.narrate_composition(0, 1)
    late = build_schemas([Late])

    def pause_validation(
        packs: Mapping[str, PackInfo],
        schemas: Mapping[str, NodeSchema],
        node_packs: Mapping[str, str],
        *,
        subject: str,
        defer_unpublished_carriers: bool = False,
    ) -> None:
        validation_started.set()
        release_validation.wait()
        original_validate(
            packs,
            schemas,
            node_packs,
            subject=subject,
            defer_unpublished_carriers=defer_unpublished_carriers,
        )

    monkeypatch.setattr(app_module, "_validate_comfy_registry_packs", pause_validation)

    async def scenario() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        if publication == "replace":
            validation = asyncio.create_task(
                state.prepare_replace(
                    (),
                    (),
                    late,
                    {"native": PackInfo("Native")},
                    {"late.echo": "native"},
                )
            )
        else:
            validation = asyncio.create_task(
                state.prepare_generation(
                    {**SCHEMAS, **late},
                    {"native": PackInfo("Native")},
                    {"late.echo": "native"},
                )
            )
        fallback_release = threading.Timer(2, release_validation.set)
        fallback_release.start()
        try:
            assert await asyncio.to_thread(validation_started.wait)
            started = time.monotonic()
            health_response, job_response, nodes_response = await asyncio.gather(
                client.get("/api/health"),
                client.get("/api/jobs/client/missing"),
                client.get("/api/nodes"),
            )
            elapsed = time.monotonic() - started
            assert elapsed < 1
            assert health_response.status == 200
            assert job_response.status == 404
            assert "late.echo" not in (await nodes_response.json())["nodes"]
            health_response.close()
            job_response.close()
            nodes_response.close()

            release_validation.set()
            prepared = await validation
            if publication == "replace":
                epoch = state.replace(
                    (),
                    (),
                    late,
                    {"native": PackInfo("Native")},
                    {"late.echo": "native"},
                    _validation=prepared,
                )
            else:
                epoch = state.publish_generation(
                    {**SCHEMAS, **late},
                    {"native": PackInfo("Native")},
                    {"late.echo": "native"},
                    _validation=prepared,
                )
            assert epoch == 2
            assert "late.echo" in (await (await client.get("/api/nodes")).json())["nodes"]
        finally:
            release_validation.set()
            fallback_release.cancel()
            if not validation.done():
                await validation
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("publication", ["replace", "generation"])
def test_prepared_surface_validation_rejects_a_stale_epoch(publication: str) -> None:
    app = create_app(make_engine, SCHEMAS)
    state = app[STATE_KEY]
    late = build_schemas([Late])
    packs = {"native": PackInfo("Native")}
    node_packs = {"late.echo": "native"}
    if publication == "replace":
        prepared = asyncio.run(state.prepare_replace((), (), late, packs, node_packs))
    else:
        prepared = asyncio.run(state.prepare_generation({**SCHEMAS, **late}, packs, node_packs))

    assert state.replace((), (), {}, {}, {}) == 2
    before_schemas = dict(state.schemas)
    if publication == "replace":
        with pytest.raises(RuntimeError, match="prepared replacement no longer matches"):
            state.replace((), (), late, packs, node_packs, _validation=prepared)
    else:
        with pytest.raises(RuntimeError, match="prepared generation no longer matches"):
            state.publish_generation({**SCHEMAS, **late}, packs, node_packs, _validation=prepared)
    assert state.schema_epoch == 2
    assert state.schemas == before_schemas


def test_announce_defers_group_carrier_checks_until_publication() -> None:
    from test_comfy_group_registry import TARGET, registry

    app = create_app(make_engine, SCHEMAS)
    state = app[STATE_KEY]
    assert state.announce({}, {"native": PackInfo("Native", comfy_groups=registry())}, {}) == 2
    assert state.announce({TARGET.node_type: TARGET}, {}, {TARGET.node_type: "native"}) == 3
    assert TARGET.node_type in state.schemas

    stranger = create_app(make_engine, SCHEMAS)
    stranger_state = stranger[STATE_KEY]
    assert (
        stranger_state.announce({}, {"native": PackInfo("Native", comfy_groups=registry())}, {})
        == 2
    )
    with pytest.raises(ValueError, match="is not owned by pack"):
        stranger_state.announce(
            {TARGET.node_type: TARGET},
            {"other": PackInfo("Other")},
            {TARGET.node_type: "other"},
        )


def test_complete_surfaces_still_reject_unknown_registry_carriers() -> None:
    from test_comfy_group_registry import registry

    with pytest.raises(ValueError, match="is not owned by pack"):
        create_app(
            make_engine,
            SCHEMAS,
            packs={"native": PackInfo("Native", comfy_aliases=alias_registry(carrier="late.echo"))},
        )
    with pytest.raises(ValueError, match="is not owned by pack"):
        create_app(
            make_engine,
            SCHEMAS,
            packs={"native": PackInfo("Native", comfy_groups=registry())},
        )


def test_published_registries_follow_carrier_removal_and_republication() -> None:
    from test_comfy_group_registry import TARGET, registry

    async def scenario() -> None:
        aliases = alias_registry(carrier="late.echo")
        groups = registry()
        info = PackInfo("Native", comfy_aliases=aliases, comfy_groups=groups)
        app = create_app(make_engine, SCHEMAS)
        state = app[STATE_KEY]
        state.announce({}, {"native": info}, {})
        carriers = {**build_schemas([Late]), TARGET.node_type: TARGET}
        client = TestClient(TestServer(app))
        await client.start_server()
        try:

            async def published() -> dict:
                wire = await (await client.get("/api/nodes")).json()
                return wire["packs"]["native"]

            absent = await published()
            assert absent["comfyAliases"]["records"] == []
            assert absent["comfyAliases"]["sourceSchemas"] == []
            assert absent["comfyGroups"]["records"] == []
            assert absent["comfyGroups"]["sourceSchemas"] == []
            assert absent["comfyGroups"]["groupSchemas"] == []

            state.announce(carriers, {}, dict.fromkeys(carriers, "native"))
            present = await published()
            assert present == info.to_wire(schemas=state.schemas)
            assert len(present["comfyAliases"]["records"]) == 1
            assert len(present["comfyGroups"]["records"]) == 1

            state.replace(tuple(carriers), (), {}, {}, {})
            assert await published() == absent
            assert state.packs["native"].comfy_aliases is aliases
            assert state.packs["native"].comfy_groups is groups

            state.replace((), (), carriers, {}, dict.fromkeys(carriers, "native"))
            assert await published() == present
        finally:
            await client.close()

    asyncio.run(scenario())


def test_replace_allows_live_comfy_sources_beside_import_registries() -> None:
    from test_comfy_group_registry import TARGET
    from test_comfy_group_registry import registry as group_registry

    aliases = alias_registry()
    groups = group_registry()
    native_schemas = {**SCHEMAS, TARGET.node_type: TARGET}
    app = create_app(
        make_engine,
        native_schemas,
        packs={
            "native": PackInfo(
                "Native",
                comfy_aliases=aliases,
                comfy_groups=groups,
            )
        },
        node_packs={"test.echo": "native", TARGET.node_type: "native"},
    )
    state = app[STATE_KEY]
    live_sources = {
        snapshot.schema.node_type: snapshot.schema
        for snapshot in (*aliases.source_schemas, *groups.source_schemas)
    }

    assert (
        state.replace(
            (),
            (),
            live_sources,
            {"comfy": PackInfo("ComfyUI Compat")},
            dict.fromkeys(live_sources, "comfy"),
        )
        == 2
    )
    assert live_sources.keys() <= state.schemas.keys()

    unrelated_app = create_app(
        make_engine,
        SCHEMAS,
        packs={"native": PackInfo("Native", comfy_aliases=aliases)},
        node_packs={"test.echo": "native"},
    )
    unrelated_state = unrelated_app[STATE_KEY]
    alias_source = aliases.source_schemas[0].schema
    with pytest.raises(ValueError, match="collides with a native schema"):
        unrelated_state.replace(
            (),
            (),
            {alias_source.node_type: alias_source},
            {"unrelated": PackInfo("Unrelated")},
            {alias_source.node_type: "unrelated"},
        )


def test_pack_provenance_misconfiguration_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown pack"):
        create_app(make_engine, SCHEMAS, node_packs={"test.echo": "nope"})
    with pytest.raises(ValueError, match="unknown node type"):
        create_app(make_engine, SCHEMAS, node_packs={"test.missing": "core"})


def test_pack_icon_endpoint() -> None:
    """The packs table advertises {digest, mediaType} and the bytes ride
    GET /api/packs/{packId}/icon with rendition-style immutable caching:
    quoted digest ETag, If-None-Match -> 304, forever cache lifetime.
    Packs without icons (and unknown packs) are 404 - the descriptor's
    presence in /api/nodes is the client's only signal."""
    from icon_bytes import png_bytes

    data = png_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={
                "ade": PackInfo(
                    display_name="AnimateDiff-Evolved",
                    icon=PackIconAsset(digest=digest, media_type="image/png", data=data),
                ),
                "plain": PackInfo(display_name="Plain Pack"),
            },
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Descriptor on the packs table: digest + mediaType, no bytes.
            nodes = await (await client.get("/api/nodes")).json()
            assert nodes["packs"]["ade"]["icon"] == {
                "digest": digest,
                "mediaType": "image/png",
            }
            assert "icon" not in nodes["packs"]["plain"]

            resp = await client.get("/api/packs/ade/icon")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert resp.headers["ETag"] == f'"{digest}"'
            assert resp.headers["Cache-Control"] == ("private, max-age=31536000, immutable")
            assert await resp.read() == data

            # Conditional revalidation: the digest never changes, so a
            # matching If-None-Match is always 304.
            resp = await client.get("/api/packs/ade/icon", headers={"If-None-Match": f'"{digest}"'})
            assert resp.status == 304
            assert resp.headers["ETag"] == f'"{digest}"'

            assert (await client.get("/api/packs/plain/icon")).status == 404
            assert (await client.get("/api/packs/nope/icon")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_static_assets_and_settings_end_to_end(tmp_path: Path) -> None:
    schema = PackSettingsSchema.from_wire(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean", "title": "Enabled", "default": True},
                "strength": {
                    "type": "number",
                    "title": "Strength",
                    "description": "Rendering strength",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["enabled", "strength"],
        }
    )
    other_schema = PackSettingsSchema.from_wire(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"label": {"type": "string", "title": "Label", "default": "other"}},
            "required": ["label"],
        }
    )
    packs = {
        "visual-pack": PackInfo(
            "Visual Pack",
            frontend_assets=(
                PackFrontendAsset("badge.png", "image/png", b"\x89PNG\r\nfixture"),
                PackFrontendAsset("styles/theme.css", "text/css", b".pack { color: #123; }\n"),
            ),
            settings_schema=schema,
        ),
        "other-pack": PackInfo("Other Pack", settings_schema=other_schema),
    }
    settings_root = tmp_path / "pack-settings"

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs=packs,
            pack_settings_root=settings_root,
        )
        async with TestClient(TestServer(app)) as client:
            image = await client.get("/packs/visual-pack/static/badge.png")
            assert image.status == 200
            assert image.headers["Content-Type"] == "image/png"
            assert await image.read() == b"\x89PNG\r\nfixture"
            css = await client.get("/packs/visual-pack/static/styles/theme.css")
            assert css.status == 200
            assert css.headers["Content-Type"].startswith("text/css")
            assert await css.read() == b".pack { color: #123; }\n"
            for path in (
                "/packs/visual-pack/static/missing.css",
                "/packs/visual-pack/static/styles/",
                "/packs/visual-pack/static/%2e%2e/secret",
                "/packs/visual-pack/static/styles/%2e%2e/%2e%2e/secret",
                "/packs/visual-pack/static/styles%5C..%5Csecret",
                "/packs/other-pack/static/badge.png",
            ):
                assert (await client.get(path)).status == 404

            initial = await client.get("/api/packs/visual-pack/settings")
            assert initial.status == 200
            initial_wire = await initial.json()
            assert initial_wire["packId"] == "visual-pack"
            assert initial_wire["displayName"] == "Visual Pack"
            assert initial_wire["schema"] == schema.to_wire()
            assert initial_wire["values"] == {"enabled": True, "strength": 0.5}
            nodes = await (await client.get("/api/nodes")).json()
            assert nodes["packs"]["visual-pack"]["settings"] is True
            assert "settings" not in nodes["packs"]["core"]

            updated = await client.put(
                "/api/packs/visual-pack/settings",
                json={"enabled": False, "strength": 0.75},
            )
            assert updated.status == 200
            assert (await updated.json())["values"] == {"enabled": False, "strength": 0.75}
            invalid = await client.put(
                "/api/packs/visual-pack/settings",
                json={"enabled": False, "strength": 2},
            )
            assert invalid.status == 400
            assert (await (await client.get("/api/packs/visual-pack/settings")).json())[
                "values"
            ] == {"enabled": False, "strength": 0.75}
            assert (await client.get("/api/packs/missing/settings")).status == 404
            assert (await (await client.get("/api/packs/other-pack/settings")).json())[
                "values"
            ] == {"label": "other"}

        restarted = create_app(
            make_engine,
            SCHEMAS,
            packs=packs,
            pack_settings_root=settings_root,
        )
        async with TestClient(TestServer(restarted)) as client:
            persisted = await client.get("/api/packs/visual-pack/settings")
            assert (await persisted.json())["values"] == {
                "enabled": False,
                "strength": 0.75,
            }
            assert (await (await client.get("/api/packs/other-pack/settings")).json())[
                "values"
            ] == {"label": "other"}

    asyncio.run(scenario())


def test_pack_blueprint_endpoint() -> None:
    """The packs table advertises full blueprint descriptors inline
    ({id, name, description?, tags?, digest} - never the body bytes) and
    the body rides GET /api/packs/{packId}/blueprints/{id} with the icon
    endpoint's exact contract: application/json, quoted digest ETag,
    If-None-Match -> 304, forever cache lifetime. Unknown packs and
    unknown blueprint ids are 404 - the descriptor's presence in
    /api/nodes is the client's only signal. Blueprints never touch node
    signatures (they are pack data, not schema identity)."""
    from dinkster_server import PackBlueprintAsset

    data = b'{"graphs": {"main": {}}}'
    digest = "sha256:" + hashlib.sha256(data).hexdigest()

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={
                "ade": PackInfo(
                    display_name="AnimateDiff-Evolved",
                    blueprints=(
                        PackBlueprintAsset(
                            id="animate",
                            name="Animate",
                            digest=digest,
                            description="Starter animation",
                            tags=("video",),
                            boundary_inputs=("core.image",),
                            boundary_outputs=("core.image",),
                            data=data,
                        ),
                        PackBlueprintAsset(id="minimal", name="Minimal", digest=digest, data=data),
                    ),
                ),
                "plain": PackInfo(display_name="Plain Pack"),
            },
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Descriptors ride the packs table inline; optional fields are
            # omitted when absent, and body bytes never appear.
            nodes = await (await client.get("/api/nodes")).json()
            assert nodes["packs"]["ade"]["blueprints"] == [
                {
                    "id": "animate",
                    "name": "Animate",
                    "description": "Starter animation",
                    "tags": ["video"],
                    "boundaryInputs": ["core.image"],
                    "boundaryOutputs": ["core.image"],
                    "digest": digest,
                },
                {"id": "minimal", "name": "Minimal", "digest": digest},
            ]
            assert "blueprints" not in nodes["packs"]["plain"]
            # Blueprints are pack data, never schema identity: node
            # signatures are identical with and without them.
            for type_id, entry in nodes["nodes"].items():
                assert entry["signature"] == schema_signature(SCHEMAS[type_id])

            resp = await client.get("/api/packs/ade/blueprints/animate")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            assert resp.headers["ETag"] == f'"{digest}"'
            assert resp.headers["Cache-Control"] == ("private, max-age=31536000, immutable")
            assert await resp.read() == data

            # Conditional revalidation: the digest never changes, so a
            # matching If-None-Match is always 304.
            resp = await client.get(
                "/api/packs/ade/blueprints/animate",
                headers={"If-None-Match": f'"{digest}"'},
            )
            assert resp.status == 304
            assert resp.headers["ETag"] == f'"{digest}"'

            assert (await client.get("/api/packs/ade/blueprints/nope")).status == 404
            assert (await client.get("/api/packs/plain/blueprints/x")).status == 404
            assert (await client.get("/api/packs/nope/blueprints/x")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_template_endpoints() -> None:
    """Templates are a browse surface, not schema data: descriptors ride
    the query-first paged /api/templates index (NEVER the /api/nodes packs
    table), and bodies ride GET /api/packs/{packId}/templates/{id} with
    the blueprint endpoint's exact immutable-caching contract. The index
    owns its query (q/tag/pack), pages with a cursor BOUND to that query
    (mismatch -> 400), and lists asset requirements as pack-local ids for
    clients to join against the packs table."""
    from dinkster_server import PackIconAsset, PackTemplateAsset

    data = b'{"graphs": {"main": {}}}'
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    thumbnail_data = b"thumbnail"
    thumbnail_digest = "sha256:" + hashlib.sha256(thumbnail_data).hexdigest()

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={
                "ade": PackInfo(
                    display_name="AnimateDiff-Evolved",
                    templates=(
                        PackTemplateAsset(
                            id="animate",
                            name="Animate",
                            digest=digest,
                            description="Starter animation",
                            tags=("video",),
                            family="dinkster.wan22",
                            models=("wan.safetensors",),
                            assets=("motion-model",),
                            thumbnail=PackIconAsset(
                                digest=thumbnail_digest,
                                media_type="image/png",
                                data=thumbnail_data,
                            ),
                            data=data,
                        ),
                        PackTemplateAsset(id="minimal", name="Minimal", digest=digest, data=data),
                    ),
                ),
                "upscalers": PackInfo(
                    display_name="Upscalers",
                    templates=(
                        PackTemplateAsset(
                            id="upscale",
                            name="Upscale",
                            digest=digest,
                            tags=("image",),
                            data=data,
                        ),
                    ),
                ),
                "plain": PackInfo(display_name="Plain Pack"),
            },
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Templates never ride the packs table - the index is the
            # browse surface, so /api/nodes stays lean.
            nodes = await (await client.get("/api/nodes")).json()
            assert "templates" not in nodes["packs"]["ade"]

            # Unfiltered index: deterministic (pack, id) order, full
            # descriptors, optional fields omitted, no body bytes.
            listing = await (await client.get("/api/templates")).json()
            assert listing["templates"] == [
                {
                    "pack": "ade",
                    "id": "animate",
                    "name": "Animate",
                    "description": "Starter animation",
                    "tags": ["video"],
                    "family": "dinkster.wan22",
                    "models": ["wan.safetensors"],
                    "assets": ["motion-model"],
                    "thumbnail": {"digest": thumbnail_digest, "mediaType": "image/png"},
                    "digest": digest,
                },
                {"pack": "ade", "id": "minimal", "name": "Minimal", "digest": digest},
                {
                    "pack": "upscalers",
                    "id": "upscale",
                    "name": "Upscale",
                    "tags": ["image"],
                    "digest": digest,
                },
            ]
            assert "cursor" not in listing

            # q= matches id/name/description/tags, case-insensitive.
            hits = await (await client.get("/api/templates?q=ANIM")).json()
            assert [t["id"] for t in hits["templates"]] == ["animate"]
            # tag= is exact; pack= scopes to one pack.
            hits = await (await client.get("/api/templates?tag=image")).json()
            assert [t["id"] for t in hits["templates"]] == ["upscale"]
            hits = await (await client.get("/api/templates?pack=ade")).json()
            assert [t["id"] for t in hits["templates"]] == ["animate", "minimal"]

            # Paging: limit=1 yields a cursor; following it walks the
            # collection without skips or repeats.
            seen: list[str] = []
            cursor = ""
            while True:
                url = "/api/templates?limit=1" + (f"&cursor={cursor}" if cursor else "")
                page = await (await client.get(url)).json()
                seen.extend(f"{t['pack']}/{t['id']}" for t in page["templates"])
                if "cursor" not in page:
                    break
                cursor = page["cursor"]
            assert seen == ["ade/animate", "ade/minimal", "upscalers/upscale"]

            # The cursor BINDS its query: replaying it under a different
            # query is a loud 400, never silently wrong results.
            first = await (await client.get("/api/templates?limit=1")).json()
            resp = await client.get(f"/api/templates?limit=1&q=animate&cursor={first['cursor']}")
            assert resp.status == 400
            assert (await client.get("/api/templates?cursor=garbage")).status == 400
            assert (await client.get("/api/templates?limit=x")).status == 400

            # Body endpoint: blueprint contract verbatim.
            resp = await client.get("/api/packs/ade/templates/animate")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            assert resp.headers["ETag"] == f'"{digest}"'
            assert resp.headers["Cache-Control"] == ("private, max-age=31536000, immutable")
            assert await resp.read() == data

            resp = await client.get(
                "/api/packs/ade/templates/animate",
                headers={"If-None-Match": f'"{digest}"'},
            )
            assert resp.status == 304
            assert resp.headers["ETag"] == f'"{digest}"'

            resp = await client.get("/api/packs/ade/templates/animate/thumbnail")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert resp.headers["ETag"] == f'"{thumbnail_digest}"'
            assert resp.headers["Cache-Control"] == ("private, max-age=31536000, immutable")
            assert await resp.read() == thumbnail_data
            resp = await client.get(
                "/api/packs/ade/templates/animate/thumbnail",
                headers={"If-None-Match": f'"{thumbnail_digest}"'},
            )
            assert resp.status == 304
            assert (await client.get("/api/packs/ade/templates/minimal/thumbnail")).status == 404

            assert (await client.get("/api/packs/ade/templates/nope")).status == 404
            assert (await client.get("/api/packs/plain/templates/x")).status == 404
            assert (await client.get("/api/packs/nope/templates/x")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_pack_docs_index_marker_and_immutable_routes() -> None:
    from dinkster_server import PackDocAsset, PackDocPageAsset, PackDocsAsset

    page_data = b"# Echo\n\nThis body has no front matter.\n"
    guide_data = b"# Guide\n\nThis body is unchanged.\n"
    unknown_data = b"# Unknown\n"
    image_data = b"image"
    page_digest = "sha256:" + hashlib.sha256(page_data).hexdigest()
    guide_digest = "sha256:" + hashlib.sha256(guide_data).hexdigest()
    unknown_digest = "sha256:" + hashlib.sha256(unknown_data).hexdigest()
    image_digest = "sha256:" + hashlib.sha256(image_data).hexdigest()
    image = PackDocAsset(
        source="assets/preview.webp",
        digest=image_digest,
        media_type="image/webp",
        data=image_data,
    )
    docs = PackDocsAsset(
        default_locale="en",
        pages=(
            PackDocPageAsset(
                kind="node",
                id="test.echo",
                locale="en",
                title="Echo",
                summary="Echoes a value.",
                schema_version=1,
                digest=page_digest,
                assets=(image,),
                data=page_data,
            ),
            PackDocPageAsset(
                kind="node",
                id="test.unknown",
                locale="en",
                title="Unknown",
                summary="Not published.",
                digest=unknown_digest,
                data=unknown_data,
            ),
            PackDocPageAsset(
                kind="guide",
                id="getting-started",
                locale="en",
                title="Getting started",
                summary="Learn the basics.",
                digest=guide_digest,
                order=10,
                tags=("basics",),
                data=guide_data,
            ),
        ),
    )

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={"docs-pack": PackInfo("Docs Pack", docs=docs)},
            node_packs={"test.echo": "docs-pack"},
        )
        async with TestClient(TestServer(app)) as client:
            current = await (await client.get("/api/nodes")).json()
            assert current["dinkster"]["schemaWire"] == 1
            assert current["nodes"]["test.echo"]["hasDocs"] is True
            assert "docs" not in current["packs"]["docs-pack"]

            listing = await (await client.get("/api/docs")).json()
            assert [item["id"] for item in listing["docs"]] == [
                "getting-started",
                "test.echo",
            ]
            assert "cursor" not in listing
            node = listing["docs"][1]
            assert node == {
                "pack": "docs-pack",
                "kind": "node",
                "id": "test.echo",
                "defaultLocale": "en",
                "locales": {
                    "en": {
                        "title": "Echo",
                        "summary": "Echoes a value.",
                        "schemaVersion": 1,
                        "digest": page_digest,
                        "assets": {
                            "assets/preview.webp": {
                                "digest": image_digest,
                                "mediaType": "image/webp",
                            }
                        },
                    }
                },
            }
            hits = await (await client.get("/api/docs?kind=node&id=test.echo")).json()
            assert [item["id"] for item in hits["docs"]] == ["test.echo"]
            first = await (await client.get("/api/docs?limit=1")).json()
            assert "cursor" in first
            second = await (await client.get(f"/api/docs?limit=1&cursor={first['cursor']}")).json()
            assert first["docs"][0]["id"] != second["docs"][0]["id"]
            assert "cursor" not in second
            assert (await client.get(f"/api/docs?kind=node&cursor={first['cursor']}")).status == 400

            guide = await client.get(f"/api/packs/docs-pack/docs/pages/{guide_digest}")
            assert guide.status == 200
            assert await guide.read() == guide_data

            page = await client.get(f"/api/packs/docs-pack/docs/pages/{page_digest}")
            assert page.status == 200
            assert page.headers["Content-Type"] == "text/markdown"
            assert page.headers["ETag"] == f'"{page_digest}"'
            assert page.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            assert await page.read() == page_data
            page = await client.get(
                f"/api/packs/docs-pack/docs/pages/{page_digest}",
                headers={"If-None-Match": f'"{page_digest}"'},
            )
            assert page.status == 304

            asset = await client.get(f"/api/packs/docs-pack/docs/assets/{image_digest}")
            assert asset.status == 200
            assert asset.headers["Content-Type"] == "image/webp"
            assert await asset.read() == image_data
            asset = await client.get(
                f"/api/packs/docs-pack/docs/assets/{image_digest}",
                headers={"If-None-Match": f'"{image_digest}"'},
            )
            assert asset.status == 304
            assert (await client.get(f"/api/packs/other/docs/pages/{page_digest}")).status == 404
            assert (await client.get("/api/packs/docs-pack/docs/pages/sha256:nope")).status == 404
            assert (await client.get("/api/packs/docs-pack/docs/assets/sha256:nope")).status == 404
            unknown = await client.get(f"/api/packs/docs-pack/docs/pages/{unknown_digest}")
            assert unknown.status == 404

    asyncio.run(scenario())


def test_pack_doc_front_matter_is_descriptor_only(tmp_path: Path) -> None:
    from dinkster_workers import load_manifest

    from dinkster.packs import pack_info_from_manifest

    body = b"# Echo\n\nThis is the presentation body.\n"
    docs = tmp_path / "docs" / "nodes" / "test.echo"
    docs.mkdir(parents=True)
    (docs / "en.md").write_bytes(
        b'+++\ntitle = "Echo"\nsummary = "Echoes a value."\nschema_version = 3\n+++\n' + body
    )
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "docs-pack"\n[pack.entry]\nnodes = "m:N"\n'
        '[pack.docs]\ndir = "docs"\ndefault_locale = "en"\n'
    )
    info = pack_info_from_manifest(load_manifest(manifest_path))
    digest = "sha256:" + hashlib.sha256(body).hexdigest()

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={"docs-pack": info},
            node_packs={"test.echo": "docs-pack"},
        )
        async with TestClient(TestServer(app)) as client:
            listing = await (await client.get("/api/docs")).json()
            locale = listing["docs"][0]["locales"]["en"]
            assert (
                locale.items()
                >= {
                    "title": "Echo",
                    "summary": "Echoes a value.",
                    "schemaVersion": 3,
                    "digest": digest,
                }.items()
            )
            response = await client.get(f"/api/packs/docs-pack/docs/pages/{digest}")
            assert response.status == 200
            assert await response.read() == body

    asyncio.run(scenario())


def test_pack_locale_catalog_wire_and_immutable_route() -> None:
    from dinkster_server import PackLocaleCatalogAsset

    en_data = b'{"nodes":{"test.echo":{"displayName":"Echo"}}}\n'
    pt_data = b'{\n  "nodes": {"test.echo": {"displayName": "Eco"}}\n}\n'
    hidden_data = b'{"nodes":{"test.missing":{"displayName":"Missing"}}}\n'
    en_digest = "sha256:" + hashlib.sha256(en_data).hexdigest()
    pt_digest = "sha256:" + hashlib.sha256(pt_data).hexdigest()
    hidden_digest = "sha256:" + hashlib.sha256(hidden_data).hexdigest()
    catalogs = (
        PackLocaleCatalogAsset("pt-br", pt_digest, ("test.echo",), pt_data),
        PackLocaleCatalogAsset("en", en_digest, ("test.echo",), en_data),
        PackLocaleCatalogAsset("zh", hidden_digest, ("test.missing",), hidden_data),
    )

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            packs={"catalog-pack": PackInfo("Catalog Pack", locale_catalogs=catalogs)},
            node_packs={"test.echo": "catalog-pack"},
        )
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/api/nodes")
            assert response.status == 200
            wire = await response.json()
            assert wire["dinkster"]["schemaWire"] == 1
            assert wire["packs"]["catalog-pack"]["locales"] == {
                "en": en_digest,
                "pt-br": pt_digest,
            }

            catalog = await client.get(f"/api/packs/catalog-pack/locales/{pt_digest}")
            assert catalog.status == 200
            assert catalog.headers["Content-Type"] == "application/json"
            assert catalog.headers["ETag"] == f'"{pt_digest}"'
            assert catalog.headers["Cache-Control"] == ("private, max-age=31536000, immutable")
            assert await catalog.read() == pt_data
            catalog = await client.get(
                f"/api/packs/catalog-pack/locales/{pt_digest}",
                headers={"If-None-Match": f'"{pt_digest}"'},
            )
            assert catalog.status == 304
            assert catalog.headers["ETag"] == f'"{pt_digest}"'

            assert (await client.get(f"/api/packs/other/locales/{pt_digest}")).status == 404
            assert (
                await client.get(f"/api/packs/catalog-pack/locales/{hidden_digest}")
            ).status == 404
            assert (await client.get("/api/packs/catalog-pack/locales/sha256:nope")).status == 404

    asyncio.run(scenario())


def test_diagnostics_endpoint_reports_cross_pack_replacement_problems() -> None:
    """The instance sees the COMPLETE schema mapping, so cross-pack
    replacement-rule references are verified here (doctor only sees one pack
    at a time). Advisory: a bad rule is a diagnostic, never a failed start."""
    import dataclasses

    from dinkster_schema import MappingSource, ReplacementCase, ReplacementRule

    async def scenario() -> None:
        clean = await make_client()
        try:
            resp = await clean.get("/api/diagnostics")
            assert resp.status == 200
            assert await resp.json() == {
                "replacementProblems": [],
                "compatSkips": [],
            }
        finally:
            await clean.close()

        bad_rule = ReplacementRule(
            from_type="test.echo",  # exists in SCHEMAS -> references checked
            cases=(
                ReplacementCase.build(
                    "test.shout",
                    inputs={"text": MappingSource.copy("no_such_input")},
                ),
            ),
        )
        successor = dataclasses.replace(SCHEMAS["test.shout"], replacements=(bad_rule,))
        dirty = dict(SCHEMAS)
        dirty["test.shout"] = successor
        app = create_app(make_engine, dirty)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/api/diagnostics")
            assert resp.status == 200
            problems = (await resp.json())["replacementProblems"]
            assert len(problems) == 1
            # Structured anchors (agreed shape with Dinkster-Frontend) plus the
            # human message alongside.
            assert problems[0]["carrier"] == "test.shout"
            assert problems[0]["from"] == "test.echo"
            assert problems[0]["caseIndex"] == 0
            assert problems[0]["ref"] == "no_such_input"
            assert problems[0]["refKind"] == "input"
            assert problems[0]["target"] == "test.echo"
            assert "'no_such_input' is not a static input id of test.echo" in problems[0]["message"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_diagnostics_surfaces_sorted_core_and_legacy_compat_skips() -> None:
    """The additive wire shape preserves core and derived legacy pack ids,
    and sorts deterministically by (packId, nodeId)."""

    async def scenario() -> None:
        def diagnostic(source: str, reason: str) -> CompatGateDiagnostic:
            return CompatGateDiagnostic(
                code="compat.dynamic.unsupported",
                source_node=source,
                reason=reason,
                source_generation="v3",
                path_kind="dynamic-family",
                input_id="member",
                input_path=("options", "member"),
                lazy=None,
                input_is_list=False,
                output_is_list=False,
                raw_link=None,
                accept_all=False,
            )

        app = create_app(
            make_engine,
            SCHEMAS,
            compat_skips={
                "comfy.zeta": {"Later": diagnostic("zeta.Later", "legacy reason")},
                "comfy": {
                    "Nested": diagnostic("Nested", "nested dynamic marker"),
                    "Alpha": diagnostic("Alpha", "unsupported names form"),
                },
            },
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/diagnostics")
            assert response.status == 200
            skips = (await response.json())["compatSkips"]
            assert [(item["packId"], item["nodeId"]) for item in skips] == [
                ("comfy", "Alpha"),
                ("comfy", "Nested"),
                ("comfy.zeta", "Later"),
            ]
            assert [item["reason"] for item in skips] == [
                "unsupported names form",
                "nested dynamic marker",
                "legacy reason",
            ]
            assert all(item["schemaEpoch"] == 1 for item in skips)
            assert all(
                item["extensionSnapshotDigest"] == app[STATE_KEY].engine.extension_snapshot_digest
                for item in skips
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_source_document_passthrough() -> None:
    """sourceDocument (frontend contract 2026-07) is an execution-OPAQUE
    provenance reference: a canonical asset digest ("blake3:<hex>") of the
    workflow document the job was compiled from. Syntax-checked at
    submission, echoed on the job wire and history, omitted when unset -
    and never anything more (existence is not checked, execution is not
    affected)."""
    digest = "blake3:" + "ab" * 32

    async def scenario() -> None:
        client = await make_client()
        try:
            # Valid digest: accepted, echoed immediately and on status.
            body = submit_body(echo_graph(), ["s"], sourceDocument=digest)
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202
            assert (await resp.json())["sourceDocument"] == digest
            status = await (await client.get("/api/jobs/c1/j1")).json()
            assert status["sourceDocument"] == digest

            # Unset: omitted from the wire, never null.
            body = submit_body(echo_graph(), ["s"], jobId="j2")
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202
            assert "sourceDocument" not in await resp.json()

            # Anything but a canonical digest string is a 400: sha256:,
            # bare hex, and non-strings never silently enter provenance.
            for bad in ("sha256:" + "ab" * 32, "ab" * 32, 7, "blake3:short"):
                body = submit_body(echo_graph(), ["s"], jobId="j3", sourceDocument=bad)
                resp = await client.post("/api/jobs", json=body)
                assert resp.status == 400
                assert "sourceDocument" in (await resp.json())["error"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_lifecycle_over_http() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            accepted = await resp.json()
            assert accepted["state"] == "queued"
            assert accepted["clientId"] == "c1"
            assert accepted["jobId"] == "j1"
            assert accepted["attemptId"] == 1
            assert accepted["latestSeq"] == 1
            nodes = await (await client.get("/api/nodes")).json()
            assert accepted["extensionSnapshotDigest"] == nodes["extensionSnapshotDigest"]
            # jobRef: server-assigned global job reference (platform plan),
            # same identity as the legacy runId alias.
            first_ref = accepted["jobRef"]
            assert first_ref and first_ref == accepted["runId"]

            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.01)
            assert status["state"] == "completed"
            assert status["extensionSnapshotDigest"] == accepted["extensionSnapshotDigest"]
            # Results are value descriptors: envelope facts, not raw payloads.
            out = status["outputs"]["s"]["out"]
            assert out["typeId"] == "core.string"
            assert out["fingerprint"]
            # Node progress is a state map.
            assert status["nodeStates"] == {"e": "completed", "s": "completed"}

            # Duplicate active/terminal identity handling: resubmitting a
            # finished job key is allowed (a rerun), an active one conflicts.
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            # A rerun is a new execution: same (clientId, jobId), new jobRef.
            assert (await resp.json())["jobRef"] != first_ref
        finally:
            await client.close()

    asyncio.run(scenario())


def test_list_output_descriptor_reports_length() -> None:
    """List descriptors carry runtime "length" (DESIGN 3.13): the element
    count is data, never schema, and rides the execution-scoped job result
    (keyed by client/job/node/output) - no global side channel."""

    async def scenario() -> None:
        client = await make_client()
        try:
            graph = Graph(nodes={"sp": GraphNode("test.splitter", {"text": "a b c"})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["sp"]))
            assert resp.status == 202
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.01)
            assert status["state"] == "completed"
            words = status["outputs"]["sp"]["words"]
            assert words["typeId"] == "list<core.string>"
            assert words["length"] == 3
            # Child descriptors recurse (bounded): per-element structure is
            # remotely interrogable, not just the top-level count.
            assert [e["typeId"] for e in words["elements"]] == ["core.string"] * 3
            assert "elementsTruncated" not in words
        finally:
            await client.close()

    asyncio.run(scenario())


def test_value_descriptor_recurses_and_truncates() -> None:
    """Nested lists expose shape level by level; per-level children are
    capped at DESCRIPTOR_ELEMENT_CAP with explicit truncation, while length
    stays the full count."""
    from dinkster_server.app import DESCRIPTOR_ELEMENT_CAP, value_descriptor
    from dinkster_values import make_list_value

    registry = TypeRegistry()
    register_core_types(registry)
    inner1 = registry.wrap("list<core.int>", [1, 2])
    inner2 = registry.wrap("list<core.int>", [3])
    outer = make_list_value("list<core.int>", (inner1, inner2))

    descriptor = value_descriptor(outer)
    assert descriptor["typeId"] == "list<list<core.int>>"
    assert descriptor["length"] == 2
    elements = descriptor["elements"]
    assert isinstance(elements, list)
    assert [e["length"] for e in elements] == [2, 1]  # ragged shape visible
    assert elements[0]["elements"][0]["typeId"] == "core.int"

    big = registry.wrap("list<core.int>", list(range(DESCRIPTOR_ELEMENT_CAP + 5)))
    descriptor = value_descriptor(big)
    assert descriptor["length"] == DESCRIPTOR_ELEMENT_CAP + 5
    big_elements = descriptor["elements"]
    assert isinstance(big_elements, list)
    assert len(big_elements) == DESCRIPTOR_ELEMENT_CAP
    assert descriptor["elementsTruncated"] is True


def test_scalar_output_descriptor_has_no_length() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.01)
            assert status["state"] == "completed"
            assert "length" not in status["outputs"]["s"]["out"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_validation_diagnostics_carry_input_id() -> None:
    """Validation failures serialize the structured anchor: nodeId plus
    inputId, so frontends map errors onto per-port affordances without
    parsing messages."""

    async def scenario() -> None:
        client = await make_client()
        try:
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi", "bogus": 1})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["e"]))
            assert resp.status == 202
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        break
                    await asyncio.sleep(0.01)
            assert status["state"] == "failed"
            error = status["error"]
            assert error["kind"] == "validation"
            diag = next(d for d in error["diagnostics"] if d["code"] == "unknown-input")
            assert diag["nodeId"] == "e"
            assert diag["inputId"] == "bogus"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_validation_errors() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            cases: list[dict[str, object]] = [
                submit_body(echo_graph(), ["s"], clientId=""),
                submit_body(echo_graph(), ["s"], jobId=42),
                submit_body(echo_graph(), []),
                submit_body(echo_graph(), ["s"], priority="high"),
                {**submit_body(echo_graph(), ["s"]), "graph": {"nodes": []}},
            ]
            for body in cases:
                resp = await client.post("/api/jobs", json=body)
                assert resp.status == 400, await resp.text()

            resp = await client.post("/api/jobs", data="not json")
            assert resp.status == 400

            assert (await client.get("/api/jobs/c1/nope")).status == 404
            assert (await client.delete("/api/jobs/c1/nope")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_attention_submit_validation_and_idempotency() -> None:
    async def paused_client(*, default: str = "auto") -> TestClient:
        app = create_app(
            make_engine,
            SCHEMAS,
            attention_policy=default,  # type: ignore[arg-type]
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        assert (await client.post("/api/queue/pause")).status == 200
        return client

    async def scenario() -> None:
        graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "attention"})})

        client = await paused_client()
        try:
            invalid = submit_body(
                graph,
                ["n"],
                attention={
                    "requestedPolicy": "auto",
                    "requestedRolePolicies": [
                        ["flux", "flash"],
                        ["flux", "sage"],
                    ],
                },
            )
            response = await client.post("/api/jobs", json=invalid)
            assert response.status == 400
            assert client.app[STATE_KEY].queue.jobs() == []

            first = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert first.status == 202
            equal_default = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["n"],
                    attention={
                        "requestedPolicy": "auto",
                        "requestedRolePolicies": [],
                    },
                ),
            )
            assert equal_default.status == 202
            assert (await equal_default.json())["duplicate"] is True
            job = client.app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None
            assert job.attention_config == AttentionPolicyConfig()
        finally:
            await client.close()

        client = await paused_client(default="dinkster_kitchen_int8")
        try:
            first = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert first.status == 202
            equal_default = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["n"],
                    attention={
                        "requestedPolicy": "dinkster_kitchen_int8",
                        "requestedRolePolicies": [],
                    },
                ),
            )
            assert equal_default.status == 202
            assert (await equal_default.json())["duplicate"] is True
            different_default = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["n"],
                    attention={
                        "requestedPolicy": "auto",
                        "requestedRolePolicies": [],
                    },
                ),
            )
            assert different_default.status == 409
        finally:
            await client.close()

        client = await paused_client()
        try:
            role_pairs = [
                ["qwen", "flash"],
                ["flux", "dinkster_kitchen_int8"],
            ]
            first = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["n"],
                    attention={
                        "requestedPolicy": "auto",
                        "requestedRolePolicies": role_pairs,
                    },
                ),
            )
            assert first.status == 202
            permuted = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    ["n"],
                    attention={
                        "requestedPolicy": "auto",
                        "requestedRolePolicies": list(reversed(role_pairs)),
                    },
                ),
            )
            assert permuted.status == 202
            assert (await permuted.json())["duplicate"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


def test_active_submit_idempotency_over_http() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            first_response = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert first_response.status == 202
            first = await first_response.json()
            duplicate_response = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert duplicate_response.status == 202
            duplicate = await duplicate_response.json()
            assert duplicate["duplicate"] is True
            assert duplicate["jobRef"] == first["jobRef"]
            assert duplicate["latestSeq"] == first["latestSeq"] == 1
            replay = await (await client.get(f"/api/jobs/by-ref/{first['jobRef']}/events")).json()
            assert [event["state"] for event in replay["events"]] == ["queued"]

            different_submissions = (
                submit_body(graph, ["n"], priority=1),
                submit_body(graph, ["other"]),
                submit_body(
                    Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "y"})}),
                    ["n"],
                ),
                submit_body(
                    graph,
                    ["n"],
                    sourceDocument="blake3:" + "ab" * 32,
                ),
            )
            for different in different_submissions:
                conflict_response = await client.post("/api/jobs", json=different)
                assert conflict_response.status == 409
                conflict = await conflict_response.json()
                assert conflict == {
                    "error": "job-key-in-use",
                    "message": ("job key c1/j1 is in use by queued job with different content"),
                    "clientId": "c1",
                    "jobId": "j1",
                }
            # Cancel over HTTP; job reaches a terminal state.
            resp = await client.delete("/api/jobs/c1/j1")
            assert resp.status == 200
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] == "cancelled":
                        break
                    await asyncio.sleep(0.01)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_empty_registry_server_skips_compile_and_preserves_generated_id_event_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        graph = Graph({"$gen-submitted": GraphNode("test.echo", {"text": "raw"})})
        direct_events: list[EngineEvent] = []
        direct_engine = make_engine(direct_events.append)
        await direct_engine.run(graph, ["$gen-submitted"], run_id="direct")

        server_events: list[EngineEvent] = []
        engine_box: list[Engine] = []

        def factory(on_event: EventListener) -> Engine:
            def record(event: EngineEvent) -> None:
                server_events.append(event)
                on_event(event)

            engine = make_engine(record)
            engine_box.append(engine)
            return engine

        app = create_app(factory, SCHEMAS)
        engine = engine_box[0]

        async def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("empty compiler registry must not compile")

        monkeypatch.setattr(engine, "compile_for_execution", forbidden)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/api/jobs", json=submit_body(graph, ["$gen-submitted"]))
            assert response.status == 202
            status = await wait_for_http_state(client, "c1", "j1", "completed")
            assert set(status["outputs"]) == {"$gen-submitted"}
            job = app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None and job.compiled_graph is None

            def stable_event_values(events: list[EngineEvent]) -> list[tuple[object, ...]]:
                values: list[tuple[object, ...]] = []
                for event in events:
                    detail = dict(event.detail)
                    detail.pop("duration_ms", None)
                    values.append((event.kind, event.node_id, detail))
                return values

            assert stable_event_values(server_events) == stable_event_values(direct_events)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_server_compile_once_preserves_raw_fingerprint_and_pinned_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = make_engine()
        resolved_graphs: list[frozenset[str]] = []

        def resolve_providers(graph: Graph) -> Graph:
            resolved_graphs.append(
                frozenset(
                    node.node_type for node in graph.nodes.values() if isinstance(node, GraphNode)
                )
            )
            generated = graph.nodes.get("$gen-test-compiled")
            if generated is None:
                return graph
            return Graph(
                {
                    **graph.nodes,
                    "$gen-test-compiled": replace(
                        generated,
                        inputs={**generated.inputs, "text": "resolved"},
                    ),
                }
            )

        old_runtime = replace(
            compiler_runtime(engine),
            resolve_providers=resolve_providers,
        )
        runtime_box = {"current": old_runtime}
        engine._pin_execution = lambda: runtime_box["current"]
        calls: list[tuple[Graph, tuple[str, ...], ExecutionRuntime]] = []
        compile_entered = asyncio.Event()
        compile_release = asyncio.Event()

        async def compile_once(
            graph: Graph,
            targets: list[str],
            *,
            execution: ExecutionRuntime,
        ) -> CompiledGraph:
            calls.append((graph, tuple(targets), execution))
            compile_entered.set()
            await compile_release.wait()
            return generated_echo_compiled(execution)

        monkeypatch.setattr(engine, "compile_for_execution", compile_once)

        def factory(on_event: EventListener) -> Engine:
            engine._on_event = on_event
            return engine

        app = create_app(factory, SCHEMAS)
        app[STATE_KEY].queue.pause()
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            body = submit_body(echo_graph(), ["s"])
            first_task = asyncio.create_task(client.post("/api/jobs", json=body))
            await compile_entered.wait()
            duplicate_task = asyncio.create_task(client.post("/api/jobs", json=body))
            await asyncio.sleep(0.01)
            assert len(calls) == 1
            compile_release.set()
            first_response, duplicate_response = await asyncio.gather(
                first_task,
                duplicate_task,
            )
            assert first_response.status == 202
            first = await first_response.json()
            assert duplicate_response.status == 202
            duplicate = await duplicate_response.json()
            assert duplicate["duplicate"] is True
            assert duplicate["jobRef"] == first["jobRef"]
            assert len(calls) == 1
            assert calls[0][2] is old_runtime

            job = app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None and job.compiled_graph is not None
            assert graph_to_wire(job.graph) == body["graph"]
            assert job.targets == ("$gen-test-compiled",)
            assert job.compiled_graph.graph.nodes["$gen-test-compiled"].inputs["text"] == (
                "resolved"
            )
            expected_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "graph": body["graph"],
                        "targets": ["s"],
                        "priority": 0,
                        "scope": "local",
                        "sourceDocument": "",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            assert job.fingerprint == expected_fingerprint

            runtime_box["current"] = replace(
                old_runtime,
                extension_snapshot=ExtensionSnapshot(frontend_api="1.0.1"),
            )
            app[STATE_KEY].queue.resume()
            status = await wait_for_http_state(client, "c1", "j1", "completed")
            assert set(status["outputs"]) == {"$gen-test-compiled"}
            assert len(calls) == 1
            assert job.execution is old_runtime
            assert job.compiled_graph.extension_snapshot_digest == (
                old_runtime.extension_snapshot_digest
            )
            assert resolved_graphs == [
                frozenset({"test.echo", "test.shout"}),
                frozenset({"test.echo"}),
                frozenset({"test.echo"}),
            ]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_server_compile_errors_and_cancellation_queue_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = make_engine()
        runtime = compiler_runtime(engine)
        engine._pin_execution = lambda: runtime

        def factory(on_event: EventListener) -> Engine:
            engine._on_event = on_event
            return engine

        app = create_app(factory, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for job_id, code, message in (
                ("malformed", GRAPH_COMPILE_ERROR_COMPILER_FAILURE, "remote malformed reply"),
                ("timeout", GRAPH_COMPILE_ERROR_TIMEOUT, "graph compilation timed out"),
            ):

                async def refuse(
                    *_args: object,
                    code: str = code,
                    message: str = message,
                    **_kwargs: object,
                ) -> None:
                    raise GraphCompileError(code, message)

                monkeypatch.setattr(engine, "compile_for_execution", refuse)
                response = await client.post(
                    "/api/jobs",
                    json=submit_body(echo_graph(), ["s"], jobId=job_id),
                )
                assert response.status == 400
                assert await response.json() == {"error": code, "message": message}
                assert app[STATE_KEY].queue.get("c1", job_id) is None

            entered = asyncio.Event()
            cancelled = asyncio.Event()

            async def wait_forever(*_args: object, **_kwargs: object) -> None:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            monkeypatch.setattr(engine, "compile_for_execution", wait_forever)

            class FakeRequest:
                def __init__(self) -> None:
                    self.app = app

                def __getitem__(self, key: object) -> object:
                    if key is PRINCIPAL_KEY:
                        return LOCAL_PRINCIPAL
                    raise KeyError(key)

                async def json(self) -> object:
                    return submit_body(echo_graph(), ["s"], jobId="cancelled")

            task = asyncio.create_task(handle_submit(cast(web.Request, FakeRequest())))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
            assert app[STATE_KEY].queue.get("c1", "cancelled") is None
            assert app[STATE_KEY]._job_admission_locks == {}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_event_sequences_replay_and_by_ref_routes() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            first_response = await client.post(
                "/api/jobs", json=submit_body(echo_graph("one"), ["s"])
            )
            second_response = await client.post(
                "/api/jobs",
                json=submit_body(echo_graph("two"), ["s"], clientId="c2", jobId="j2"),
            )
            first = await first_response.json()
            second = await second_response.json()
            refs = (first["jobRef"], second["jobRef"])

            for job_ref in refs:
                async with asyncio.timeout(5):
                    while True:
                        response = await client.get(f"/api/jobs/by-ref/{job_ref}")
                        assert response.status == 200
                        status = await response.json()
                        if status["state"] == "completed":
                            break
                        await asyncio.sleep(0.01)
                keyed = await (
                    await client.get(f"/api/jobs/{status['clientId']}/{status['jobId']}")
                ).json()
                assert status == keyed
                assert status["attemptId"] == 1
                replay_response = await client.get(f"/api/jobs/by-ref/{job_ref}/events")
                assert replay_response.status == 200
                replay = await replay_response.json()
                sequences = [event["seq"] for event in replay["events"]]
                assert sequences == list(range(1, replay["latestSeq"] + 1))
                assert replay["latestSeq"] == status["latestSeq"]
                assert all(event["jobRef"] == job_ref for event in replay["events"])
                job_states = [event for event in replay["events"] if event["type"] == "job_state"]
                assert all(event["attemptId"] == 1 for event in job_states)
                after = sequences[len(sequences) // 2]
                resumed = await (
                    await client.get(f"/api/jobs/by-ref/{job_ref}/events?after={after}")
                ).json()
                assert [event["seq"] for event in resumed["events"]] == [
                    seq for seq in sequences if seq > after
                ]
                assert resumed["latestSeq"] == replay["latestSeq"]

            for bad in ("-1", "nope", "1.0", "+1"):
                response = await client.get(f"/api/jobs/by-ref/{refs[0]}/events?after={bad}")
                assert response.status == 400
            assert (await client.get("/api/jobs/by-ref/missing")).status == 404
            assert (await client.get("/api/jobs/by-ref/missing/events")).status == 404
            assert (await client.delete("/api/jobs/by-ref/missing")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_cancel_by_ref_matches_keyed_cancel() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            accepted = await (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).json()
            response = await client.delete(f"/api/jobs/by-ref/{accepted['jobRef']}")
            assert response.status == 200
            by_ref = await response.json()
            keyed = await (await client.get("/api/jobs/c1/j1")).json()
            assert by_ref == keyed
            assert by_ref["state"] == "cancelled"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_by_ref_routes_preserve_legacy_by_ref_client_id() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            body = submit_body(echo_graph(), ["s"], clientId="by-ref", jobId="legacy")
            assert (await client.post("/api/jobs", json=body)).status == 202
            response = await client.get("/api/jobs/by-ref/legacy")
            assert response.status == 200
            assert (await response.json())["clientId"] == "by-ref"
            response = await client.delete("/api/jobs/by-ref/legacy")
            assert response.status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_replay_pressure_blob_omission_and_retention() -> None:
    from dinkster_server.app import JOB_EVENT_BUFFER, _JobReplay

    replay = _JobReplay(limit=2)
    replay.append({"type": "job_state", "state": "queued", "seq": 1}, droppable=False)
    replay.append(
        {"type": "node_event", "seq": 2, BINARY_BLOB_KEY: b"preview"},
        droppable=True,
    )
    replay.append({"type": "job_state", "state": "running", "seq": 3}, droppable=False)
    assert replay.after(1) is None
    retained = replay.after(2)
    assert retained is not None
    assert [(event["type"], event["seq"]) for event in retained] == [("job_state", 3)]

    blob_replay = _JobReplay(limit=2)
    blob_replay.append(
        {"type": "node_event", "seq": 1, BINARY_BLOB_KEY: b"preview"},
        droppable=True,
    )
    blob = blob_replay.after(0)
    assert blob == [{"type": "node_event", "seq": 1, "blobOmitted": True}]

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, history_limit=1)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            accepted = await (
                await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            ).json()
            first_ref = accepted["jobRef"]
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)

            second = await client.post(
                "/api/jobs", json=submit_body(echo_graph(), ["s"], jobId="j2")
            )
            second_ref = (await second.json())["jobRef"]
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j2")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)
            assert first_ref not in state._job_events
            assert (await client.get(f"/api/jobs/by-ref/{first_ref}/events")).status == 404

            job = state.queue.job_for_run(second_ref)
            assert job is not None
            for _ in range(JOB_EVENT_BUFFER + 1):
                state._on_engine_event(EngineEvent("node_started", second_ref, "s"))
            response = await client.get(f"/api/jobs/by-ref/{second_ref}/events?after=0")
            assert response.status == 410
            assert await response.json() == {"error": "resync-required"}
            buffered = state._job_events[second_ref]
            assert any(event["type"] == "job_state" for event, _droppable in buffered._events)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_job_replay_evicts_oldest_droppable_for_new_droppable() -> None:
    from dinkster_server.app import _JobReplay

    replay = _JobReplay(limit=2)
    replay.append({"type": "node_event", "seq": 1}, droppable=True)
    replay.append({"type": "node_event", "seq": 2}, droppable=True)
    replay.append({"type": "node_event", "seq": 3}, droppable=True)
    # The oldest droppable was evicted; the newest survives (a final
    # progress update has no later superseder). The floor rises only to
    # the evicted seq, so replay from the survivors still works.
    assert replay.floor == 1
    assert replay.after(0) is None
    assert [event["seq"] for event in replay.after(1) or []] == [2, 3]


def test_job_replay_sheds_new_droppable_when_only_protected_queued() -> None:
    from dinkster_server.app import _JobReplay

    replay = _JobReplay(limit=2)
    replay.append({"type": "job_state", "state": "queued", "seq": 1}, droppable=False)
    replay.append({"type": "job_state", "state": "running", "seq": 2}, droppable=False)
    replay.append({"type": "node_event", "seq": 3}, droppable=True)
    assert replay.floor == 3
    assert replay.after(2) is None
    assert replay.after(3) == []
    # Protected events exceed the soft bound rather than disappear.
    replay.append({"type": "job_state", "state": "completed", "seq": 4}, droppable=False)
    assert [event["seq"] for event, _droppable in replay._events] == [1, 2, 4]


def test_subscription_evicts_oldest_droppable_when_full() -> None:
    sub = Subscription(None, 2)
    sub.push({"type": "node_event", "seq": 1}, droppable=True)
    sub.push({"type": "job_state", "seq": 2}, droppable=False)
    sub.push({"type": "node_event", "seq": 3}, droppable=True)
    assert [event["seq"] for event, _droppable in sub._events] == [2, 3]
    # A protected event under pressure also evicts the oldest droppable.
    sub.push({"type": "job_state", "seq": 4}, droppable=False)
    assert [event["seq"] for event, _droppable in sub._events] == [2, 4]


def test_subscription_sheds_new_droppable_when_all_protected() -> None:
    sub = Subscription(None, 2)
    sub.push({"type": "job_state", "seq": 1}, droppable=False)
    sub.push({"type": "job_state", "seq": 2}, droppable=False)
    sub.push({"type": "node_event", "seq": 3}, droppable=True)
    assert [event["seq"] for event, _droppable in sub._events] == [1, 2]
    # Protected events exceed the soft bound rather than disappear.
    sub.push({"type": "job_state", "seq": 4}, droppable=False)
    assert [event["seq"] for event, _droppable in sub._events] == [1, 2, 4]


def test_subscription_droppability_is_the_publishers_flag_not_the_type() -> None:
    # memory_status is published droppable=True but is not an engine event
    # kind: eviction must honor the flag the publisher supplied, not a
    # type-based reclassification.
    sub = Subscription(None, 2)
    sub.push({"type": "memory_status", "seq": 1}, droppable=True)
    sub.push({"type": "memory_status", "seq": 2}, droppable=True)
    sub.push({"type": "memory_status", "seq": 3}, droppable=True)
    assert [event["seq"] for event, _droppable in sub._events] == [2, 3]


def test_progress_throttle_keeps_first_final_and_paced_updates() -> None:
    from dinkster_server.events import ProgressThrottle

    now = [0.0]
    throttle = ProgressThrottle(min_interval=0.1, min_delta=0.005, clock=lambda: now[0])
    assert throttle.admit("r", "n", {"step": 1, "total": 1000})
    # Sub-interval, sub-delta chatter is suppressed.
    now[0] += 0.01
    assert not throttle.admit("r", "n", {"step": 2, "total": 1000})
    # The interval alone is not enough without the fraction delta.
    now[0] += 0.2
    assert not throttle.admit("r", "n", {"step": 3, "total": 1000})
    # Interval and delta both satisfied: kept.
    assert throttle.admit("r", "n", {"step": 500, "total": 1000})
    # A big jump without the interval is still suppressed ...
    assert not throttle.admit("r", "n", {"step": 990, "total": 1000})
    # ... but the first update to reach the total is always kept.
    assert throttle.admit("r", "n", {"step": 1000, "total": 1000})
    # A repeated final is chatter, no matter how much later it arrives.
    now[0] += 5.0
    assert not throttle.admit("r", "n", {"step": 1000, "total": 1000})
    # Other nodes and runs are throttled independently.
    assert throttle.admit("r", "n2", {"step": 1, "total": 1000})
    assert throttle.admit("r2", "n", {"step": 1, "total": 1000})


def test_progress_throttle_keeps_phase_and_total_changes() -> None:
    from dinkster_server.events import ProgressThrottle

    now = [0.0]
    throttle = ProgressThrottle(clock=lambda: now[0])
    assert throttle.admit("r", "n", {"step": 1, "total": 100})
    assert not throttle.admit("r", "n", {"step": 1, "total": 100})
    # Phase-text changes are never chatter.
    assert throttle.admit("r", "n", {"step": 1, "total": 100, "text": "decoding"})
    assert not throttle.admit("r", "n", {"step": 1, "total": 100, "text": "decoding"})
    # A total change restates the scale and is kept.
    assert throttle.admit("r", "n", {"step": 1, "total": 200, "text": "decoding"})


def test_progress_throttle_passes_malformed_and_prunes_finished_runs() -> None:
    from dinkster_server.events import ProgressThrottle

    now = [0.0]
    throttle = ProgressThrottle(clock=lambda: now[0])
    # Missing or non-positive totals are indeterminate, not this policy's
    # call: pass them through untouched.
    assert throttle.admit("r", "n", {"step": 1})
    assert throttle.admit("r", "n", {"step": 1, "total": 0})
    assert throttle.admit("r", "n", {"step": "x", "total": "y"})
    assert throttle.admit("r", "n", {"step": float("nan"), "total": float("nan")})
    assert throttle.admit("r", "n", {"step": 1, "total": float("inf")})
    assert throttle.admit("r", "n", {"step": 1, "total": 100})
    assert not throttle.admit("r", "n", {"step": 1, "total": 100})
    throttle.retain(set())
    # State was dropped: the same update counts as the node's first again.
    assert throttle.admit("r", "n", {"step": 1, "total": 100})


def test_progress_events_throttled_at_ingestion_final_survives_in_replay() -> None:
    # Feeds synthetic progress node_events straight into the engine-event
    # listener: the throttle must collapse the flood before the replay
    # buffer sees it, while the first and final updates always survive.
    # (Replay-buffer pressure itself is covered by the _JobReplay units.)
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            accepted = await (
                await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            ).json()
            run_ref = accepted["jobRef"]
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)
            for step in range(1, 2001):
                state._on_engine_event(
                    EngineEvent(
                        "node_event",
                        run_ref,
                        "s",
                        {"name": "progress", "data": {"step": step, "total": 2000}},
                    )
                )
            response = await client.get(f"/api/jobs/by-ref/{run_ref}/events?after=0")
            assert response.status == 200
            events = (await response.json())["events"]
            progress = [event for event in events if event.get("event") == "progress"]
            # The tight flood collapses to a handful of kept updates, and the
            # final one survives in the replay buffer.
            assert progress
            assert progress[0]["data"]["step"] == 1
            assert progress[-1]["data"]["step"] == 2000
            assert len(progress) < 50
        finally:
            await client.close()

    asyncio.run(scenario())


def test_queue_endpoints_over_http() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            # Baseline shape: nothing queued, nothing running, not paused.
            status = await (await client.get("/api/queue")).json()
            assert status == {
                "queued": [],
                "running": [],
                "maxRunningJobs": 1,
                "paused": False,
            }

            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            assert (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).status == 202
            await asyncio.sleep(0.05)
            status = await (await client.get("/api/queue")).json()
            assert status["paused"] is True
            assert [j["jobId"] for j in status["queued"]] == ["j1"]
            assert status["running"] == []

            assert (await client.post("/api/queue/resume")).status == 200
            await Sleeper.entered.wait()
            status = await (await client.get("/api/queue")).json()
            assert status["paused"] is False
            assert status["queued"] == []
            assert [j["jobId"] for j in status["running"]] == ["j1"]

            Sleeper.gate.set()
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)

            # Job listing: every client's, then filtered to one.
            body = submit_body(echo_graph(), ["s"], clientId="c2", jobId="other")
            assert (await client.post("/api/jobs", json=body)).status == 202
            jobs = (await (await client.get("/api/jobs")).json())["jobs"]
            assert {(j["clientId"], j["jobId"]) for j in jobs} == {("c1", "j1"), ("c2", "other")}
            jobs = (await (await client.get("/api/jobs?clientId=c2")).json())["jobs"]
            assert [(j["clientId"], j["jobId"]) for j in jobs] == [("c2", "other")]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_queue_clear_over_http_and_queue_state_broadcast() -> None:
    async def scenario() -> None:
        reset_sleeper()
        client = await make_client()
        try:
            async with client.ws_connect("/api/events?clientId=someone-else") as ws:
                # Occupy the single slot, then queue two more jobs.
                graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
                assert (
                    await client.post("/api/jobs", json=submit_body(graph, ["n"]))
                ).status == 202
                await Sleeper.entered.wait()
                for job_id in ("q1", "q2"):
                    body = submit_body(echo_graph(), ["s"], jobId=job_id)
                    assert (await client.post("/api/jobs", json=body)).status == 202

                resp = await client.post("/api/queue/clear", json={"clientId": "c1"})
                assert resp.status == 200
                cleared = (await resp.json())["cleared"]
                assert {c["jobId"] for c in cleared} == {"q1", "q2"}
                # The running job survived the clear.
                status = await (await client.get("/api/queue")).json()
                assert [j["jobId"] for j in status["running"]] == ["j1"]

                # Pause/resume state changes are instance-wide broadcasts:
                # even a subscriber filtered to an unrelated client sees them.
                assert (await client.post("/api/queue/pause")).status == 200
                async with asyncio.timeout(5):
                    while True:
                        event = json.loads((await ws.receive()).data)
                        if event["type"] == "queue_state":
                            break
                assert event["paused"] is True
                assert (await client.post("/api/queue/resume")).status == 200

                # Validation: clientId must be a non-empty string when given.
                resp = await client.post("/api/queue/clear", json={"clientId": 42})
                assert resp.status == 400

                Sleeper.gate.set()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_websocket_event_stream() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            ws = await client.ws_connect("/api/events?clientId=c1")
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202

            events: list[dict[str, object]] = []
            async with asyncio.timeout(5):
                while True:
                    event = await ws.receive_json()
                    events.append(event)
                    if event["type"] == "job_state" and event["state"] == "completed":
                        break

            kinds = [e["type"] for e in events]
            assert kinds[0] == "job_state"  # queued
            assert "run_finished" in kinds
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
            node_events = [e for e in events if e["type"] == "node_finished"]
            assert {e["nodeId"] for e in node_events} == {"e", "s"}
            # Every event carries the job identity for client-side routing.
            assert all(e.get("clientId") == "c1" for e in events)
            assert all(e.get("jobId") == "j1" for e in events)
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_websocket_filters_other_clients() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            ws = await client.ws_connect("/api/events?clientId=other")
            resp = await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            assert resp.status == 202
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] == "completed":
                        break
                    await asyncio.sleep(0.01)
            # Nothing was routed to the other client's subscription.
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.1):
                    await ws.receive_json()
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_node_events_over_websocket_with_binary_preview() -> None:
    """A reporting node's progress arrives as a typed JSON node_event; its
    preview arrives as one self-describing binary frame (length-prefixed
    JSON header + raw payload), never base64 in JSON."""

    async def scenario() -> None:
        client = await make_client()
        try:
            ws = await client.ws_connect("/api/events?clientId=c1")
            graph = Graph(nodes={"p": GraphNode("test.previewer", {"text": "go"})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["p"]))
            assert resp.status == 202

            progress: dict[str, object] | None = None
            preview_header: dict[str, object] | None = None
            preview_payload: bytes | None = None
            async with asyncio.timeout(5):
                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        raw = msg.data
                        header_len = int.from_bytes(raw[:4], "big")
                        preview_header = json.loads(raw[4 : 4 + header_len])
                        preview_payload = raw[4 + header_len :]
                        continue
                    event = json.loads(msg.data)
                    if event["type"] == "node_event" and event["event"] == "progress":
                        progress = event
                    if event["type"] == "job_state" and event["state"] == "completed":
                        break

            assert progress is not None
            assert progress["nodeId"] == "p"
            assert progress["clientId"] == "c1"
            assert progress["data"] == {"step": 1, "total": 2, "text": "rendering"}
            assert "worker" not in progress
            assert "executionArm" not in progress

            assert preview_header is not None and preview_payload is not None
            assert preview_header["type"] == "node_event"
            assert preview_header["event"] == "preview"
            assert preview_header["nodeId"] == "p"
            assert preview_header["data"] == {
                "mime": "image/png",
                "width": 2,
                "height": 2,
            }
            assert BINARY_BLOB_KEY not in preview_header
            assert "worker" not in preview_header
            assert "executionArm" not in preview_header
            assert preview_payload == b"\x89png-bytes"
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_encode_binary_event_frame_layout() -> None:
    frame = encode_binary_event(
        {"type": "node_event", "event": "preview", BINARY_BLOB_KEY: b"\x00\x01\xff"}
    )
    header_len = int.from_bytes(frame[:4], "big")
    header = json.loads(frame[4 : 4 + header_len])
    assert header == {"type": "node_event", "event": "preview"}
    assert frame[4 + header_len :] == b"\x00\x01\xff"


def test_hub_broadcast_reaches_client_filtered_subscribers() -> None:
    async def scenario() -> None:
        hub = EventHub()
        filtered = hub.subscribe("c1")
        # Client-scoped events for another client stay invisible ...
        hub.publish({"type": "x"}, client_id="other", droppable=False)
        # ... but instance-scoped broadcasts (client_id=None) reach everyone.
        hub.publish({"type": "memory_status"}, client_id=None, droppable=True)
        event = await filtered.get()
        assert event is not None
        assert event == {"type": "memory_status"}
        assert "seq" not in event
        hub.close()

    asyncio.run(scenario())


def test_memory_status_events_broadcast_when_governed() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        app = create_app(make_engine, SCHEMAS, governor=governor, memory_status_interval=0.02)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # A client-filtered subscriber still sees instance telemetry.
            ws = await client.ws_connect("/api/events?clientId=c1")
            async with asyncio.timeout(5):
                while True:
                    event = await ws.receive_json()
                    if event["type"] == "memory_status":
                        break
            assert event["memoryGovernor"]["ram"]["budgetBytes"] == 100
            assert "devices" in event and "queue" in event
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_status_endpoint() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.get("/memory/status")
            assert resp.status == 200
            data = await resp.json()
            assert data["devices"] == {}  # lanes appear once first used
            # No governor supplied: null is honest, never a fake empty report.
            assert data["memoryGovernor"] is None
            assert data["queue"]["maxRunningJobs"] == 1

            await client.post("/api/jobs", json=submit_body(echo_graph(), ["s"]))
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] == "completed":
                        break
                    await asyncio.sleep(0.01)
            data = await (await client.get("/memory/status")).json()
            assert data["devices"]["compute"]["executionInUse"] == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_status_reports_governor_when_supplied() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        app = create_app(make_engine, SCHEMAS, governor=governor)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            async with governor.reserve("ram", 60):
                data = await (await client.get("/memory/status")).json()
                report = data["memoryGovernor"]["ram"]
                assert report["budgetBytes"] == 100
                assert report["reservedBytes"] == 60
                assert report["availableBytes"] == 40
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["ram"]["reservedBytes"] == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_status_distinguishes_live_and_restart_bound_accelerator_budgets() -> None:
    async def scenario() -> None:
        gib = 1024**3
        device = "vram:cuda:0"
        applied = {"dinkster-compat-comfy": {device: 24 * gib}}
        governor = MemoryGovernor(
            {device: 24 * gib},
            telemetry=lambda _device: MeasuredMemory(12 * gib, 32 * gib),
        )
        app = create_app(
            make_engine,
            SCHEMAS,
            governor=governor,
            residency_memory_budgets=lambda: applied,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            data = await (await client.get("/memory/status")).json()
            policy = data["acceleratorPolicy"]
            assert policy["physicalHeadroomBytes"] == 256 * 1024**2
            assert policy["inferenceReserveBytes"] == int(0.8 * gib)
            assert policy["minimumFreeBytes"] == 256 * 1024**2 + int(0.8 * gib)
            assert policy["aimdoSimpleHeadroomBaseBytes"] == 256 * 1024**2
            cuda = policy["devices"][device]
            assert cuda["governorAdmission"]["budgetBytes"] == 24 * gib
            assert cuda["governorAdmission"]["budgetHeadroomBytes"] == 8 * gib
            assert cuda["residencyApplied"]["budgetBytes"] == 24 * gib
            assert cuda["residencyApplied"]["budgetsByWorker"] == {
                "dinkster-compat-comfy": 24 * gib
            }
            assert not cuda["residencyBudgetStale"]

            app[STATE_KEY].settings.update("memory-budgets", {device: 20 * gib})
            data = await (await client.get("/memory/status")).json()
            cuda = data["acceleratorPolicy"]["devices"][device]
            assert cuda["governorAdmission"]["budgetBytes"] == 20 * gib
            assert cuda["residencyApplied"]["budgetBytes"] == 24 * gib
            assert cuda["residencyBudgetStale"]

            applied["dinkster-compat-comfy"] = {device: 20 * gib}
            data = await (await client.get("/memory/status")).json()
            cuda = data["acceleratorPolicy"]["devices"][device]
            assert cuda["residencyApplied"]["effectiveBudgetBytes"] == 20 * gib
            assert not cuda["residencyBudgetStale"]

            applied["legacy-comfy"] = {device: 24 * gib}
            data = await (await client.get("/memory/status")).json()
            cuda = data["acceleratorPolicy"]["devices"][device]
            assert cuda["residencyApplied"]["budgetBytes"] is None
            assert not cuda["residencyApplied"]["consistent"]
            assert cuda["residencyBudgetStale"]
        finally:
            await client.close()

    asyncio.run(scenario())


# -- cross-instance memory coordination (DESIGN 3.10) --------------------------


class TrimmableCache(Shedder):
    """A governed consumer holding sheddable bytes on one device."""

    def __init__(self, device: str, holding: int) -> None:
        self.device = device
        self.holding = holding

    def footprint(self, device: str) -> int:
        return self.holding if device == self.device else 0

    async def shed(self, pressure: PressureSignal) -> int:
        freed = min(self.holding, pressure.bytes_needed)
        self.holding -= freed
        return freed


async def make_governed_client(
    governor: MemoryGovernor,
) -> TestClient:
    app = create_app(make_engine, SCHEMAS, governor=governor)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_coordination_endpoints_answer_409_when_ungoverned() -> None:
    async def scenario() -> None:
        client = await make_client()  # no governor
        try:
            for method, path in [
                ("post", "/memory/shed"),
                ("post", "/memory/reserve"),
                ("post", "/memory/reserve/xyz/renew"),
                ("delete", "/memory/reserve/xyz"),
                ("post", "/cache/trim"),
            ]:
                resp = await getattr(client, method)(path, json={})
                assert resp.status == 409, (method, path)
                assert "ungoverned" in (await resp.json())["error"]
            # Status still answers, with honest nulls.
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"] is None
            assert data["leases"] is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_reserve_status_renew_release_round_trip() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        client = await make_governed_client(governor)
        try:
            resp = await client.post(
                "/memory/reserve",
                json={"device": "vram:cuda:0", "bytes": 60, "ttlSeconds": 30},
            )
            assert resp.status == 201
            lease = await resp.json()
            assert lease["device"] == "vram:cuda:0"
            assert lease["bytes"] == 60
            assert 29 < lease["expiresInSeconds"] <= 30
            lease_id = lease["reservationId"]

            # The lease is visible in status: peers see what is pinned.
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["vram:cuda:0"]["reservedBytes"] == 60
            assert [entry["reservationId"] for entry in data["leases"]] == [lease_id]

            resp = await client.post(f"/memory/reserve/{lease_id}/renew", json={"ttlSeconds": 45})
            assert resp.status == 200
            renewed = await resp.json()
            assert 44 < renewed["expiresInSeconds"] <= 45

            resp = await client.delete(f"/memory/reserve/{lease_id}")
            assert resp.status == 204
            assert governor.reserved("vram:cuda:0") == 0
            # Gone means gone: release and renew are now misses.
            assert (await client.delete(f"/memory/reserve/{lease_id}")).status == 404
            resp = await client.post(f"/memory/reserve/{lease_id}/renew", json={"ttlSeconds": 5})
            assert resp.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_reserve_maps_governor_verdicts_to_507_and_503() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        try:
            # It can never fit: 507 Insufficient Storage.
            resp = await client.post("/memory/reserve", json={"device": "ram", "bytes": 200})
            assert resp.status == 507

            # It did not fit in time: 503, try later.
            first = await (
                await client.post("/memory/reserve", json={"device": "ram", "bytes": 100})
            ).json()
            resp = await client.post(
                "/memory/reserve",
                json={"device": "ram", "bytes": 50, "timeoutSeconds": 0.05},
            )
            assert resp.status == 503
            assert governor.reserved("ram") == 100  # no residue from failures
            await client.delete(f"/memory/reserve/{first['reservationId']}")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_lease_ttl_expiry_frees_reservation_over_http() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        try:
            resp = await client.post(
                "/memory/reserve",
                json={"device": "ram", "bytes": 60, "ttlSeconds": 0.05},
            )
            assert resp.status == 201
            # The holder goes silent - a crashed peer. TTL is the cleanup.
            async with asyncio.timeout(2):
                while governor.reserved("ram") > 0:
                    await asyncio.sleep(0.01)
            data = await (await client.get("/memory/status")).json()
            assert data["leases"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_shed_and_cache_trim() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        cache = TrimmableCache("ram", 40)
        pool = TrimmableCache("ram", 30)
        governor.register_shedder(cache, name="cache", priority=0)
        governor.register_shedder(pool, name="pool", priority=1)
        client = await make_governed_client(governor)
        try:
            # Consumer names are discoverable through status.
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["ram"]["consumers"] == {
                "cache": 40,
                "pool": 30,
            }

            # Shed asks everyone in priority order; freed is honest.
            resp = await client.post("/memory/shed", json={"device": "ram", "bytes": 50})
            assert resp.status == 200
            body = await resp.json()
            assert body["freedBytes"] == 50
            assert (cache.holding, pool.holding) == (0, 20)

            # Trim targets named consumers only; the pool is untouched.
            cache.holding = 40
            resp = await client.post(
                "/cache/trim",
                json={"device": "ram", "consumers": ["cache"]},
            )
            body = await resp.json()
            assert body["freedBytes"] == 40
            assert (cache.holding, pool.holding) == (0, 20)

            # Bytes omitted means full maintenance: everything sheddable goes.
            cache.holding = 10
            resp = await client.post("/cache/trim", json={"device": "ram"})
            body = await resp.json()
            assert body["freedBytes"] == 30
            assert (cache.holding, pool.holding) == (0, 0)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_free_requires_and_preserves_paused_idle_queue() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def full_free(request_id: str) -> list[dict[str, object]]:
            assert request_id == "maintenance-1"
            entered.set()
            await finish.wait()
            return [
                {
                    "worker": "models-a",
                    "workerInstance": "instance-a",
                    "deviceMap": {"mapping": {"cuda:0": "cuda:1"}, "qualifier": None},
                    "status": "complete",
                    "consumers": [
                        {"consumer": "models", "status": "complete"},
                        {"consumer": "zero-cost", "status": "complete"},
                    ],
                },
                {
                    "worker": "models-b",
                    "workerInstance": "instance-b",
                    "deviceMap": {"mapping": {}, "qualifier": "cpu"},
                    "status": "complete",
                    "consumers": [
                        {"consumer": "models", "status": "complete"},
                        {"consumer": "zero-cost", "status": "complete"},
                    ],
                },
            ]

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/memory/free", json={"requestId": "maintenance-1"})
            assert response.status == 409
            assert (await response.json())["queuePaused"] is False

            assert (await client.post("/api/queue/pause")).status == 200
            operation = asyncio.create_task(
                client.post("/memory/free", json={"requestId": "maintenance-1"})
            )
            await entered.wait()
            assert (await client.post("/api/queue/resume")).status == 409
            duplicate = await client.post(
                "/memory/free", json={"requestId": "maintenance-duplicate"}
            )
            assert duplicate.status == 409
            finish.set()

            response = await operation
            assert response.status == 200
            body = await response.json()
            assert body["requestId"] == "maintenance-1"
            assert body["completed"] is True
            assert body["queuePaused"] is True
            assert len(body["workers"]) == 2
            assert body["workers"][0]["deviceMap"]["mapping"] == {"cuda:0": "cuda:1"}
            assert app[STATE_KEY].queue.paused
            assert (await client.post("/api/queue/resume")).status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_free_reports_incomplete_participant_and_validates_nonce() -> None:
    async def scenario() -> None:
        async def full_free(_request_id: str) -> list[dict[str, object]]:
            return [
                {
                    "worker": "models",
                    "workerInstance": "instance-a",
                    "deviceMap": {"mapping": {}, "qualifier": None},
                    "status": "unsupported",
                    "consumers": [{"consumer": "legacy-cache", "status": "unsupported"}],
                }
            ]

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            for body in ({}, {"requestId": ""}, {"requestId": 1}):
                assert (await client.post("/memory/free", json=body)).status == 400
            response = await client.post("/memory/free", json={"requestId": "maintenance-2"})
            assert response.status == 200
            body = await response.json()
            assert body["completed"] is False
            assert body["queuePaused"] is True
            assert body["workers"][0]["status"] == "unsupported"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_free_checks_declared_consumer_statuses_for_completion() -> None:
    async def scenario() -> None:
        async def full_free(_request_id: str) -> list[dict[str, object]]:
            return [
                {
                    "worker": "models",
                    "workerInstance": "instance-a",
                    "deviceMap": {"mapping": {}, "qualifier": None},
                    "status": "complete",
                    "consumers": [{"consumer": "cache", "status": "busy"}],
                }
            ]

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            response = await client.post("/memory/free", json={"requestId": "maintenance-3"})
            assert response.status == 200
            body = await response.json()
            assert body["completed"] is False
            assert body["queuePaused"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


def test_memory_free_refuses_running_queue_without_cancelling_job() -> None:
    async def scenario() -> None:
        reset_sleeper()
        called = False

        async def full_free(_request_id: str) -> list[dict[str, object]]:
            nonlocal called
            called = True
            return []

        app = create_app(make_engine, SCHEMAS, full_free=full_free)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            assert (await client.post("/api/jobs", json=submit_body(graph, ["n"]))).status == 202
            await Sleeper.entered.wait()
            assert (await client.post("/api/queue/pause")).status == 200
            response = await client.post("/memory/free", json={"requestId": "while-running"})
            assert response.status == 409
            assert called is False
            assert (await response.json())["queuePaused"] is True
            assert (await (await client.get("/api/jobs/c1/j1")).json())["state"] == "running"
            Sleeper.gate.set()
            async with asyncio.timeout(5):
                while (await (await client.get("/api/jobs/c1/j1")).json())["state"] != "completed":
                    await asyncio.sleep(0.01)
            assert app[STATE_KEY].queue.paused
        finally:
            Sleeper.gate.set()
            await client.close()

    asyncio.run(scenario())


def test_coordination_request_validation() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        try:
            cases = [
                ("/memory/reserve", {"bytes": 10}),  # device missing
                ("/memory/reserve", {"device": "ram"}),  # bytes missing
                ("/memory/reserve", {"device": "ram", "bytes": True}),  # bool
                ("/memory/reserve", {"device": "ram", "bytes": -1}),
                ("/memory/reserve", {"device": "ram", "bytes": 1, "ttlSeconds": 0}),
                (
                    "/memory/reserve",
                    {"device": "ram", "bytes": 1, "timeoutSeconds": -2},
                ),
                ("/memory/shed", {"device": "", "bytes": 10}),
                ("/cache/trim", {"device": "ram", "consumers": "cache"}),
                ("/cache/trim", {"device": "ram", "consumers": [1, 2]}),
            ]
            for path, body in cases:
                resp = await client.post(path, json=body)
                assert resp.status == 400, (path, body)
            resp = await client.post(
                "/memory/reserve",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            assert governor.reserved("ram") == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_app_cleanup_releases_active_leases() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        client = await make_governed_client(governor)
        resp = await client.post(
            "/memory/reserve", json={"device": "ram", "bytes": 80, "ttlSeconds": 300}
        )
        assert resp.status == 201
        assert governor.reserved("ram") == 80
        await client.close()  # app cleanup closes the broker
        assert governor.reserved("ram") == 0

    asyncio.run(scenario())


# -- consumer detail contract over the wire (DESIGN 3.10 observability) --------


class DetailedPool(TrimmableCache):
    """A consumer that also names what it holds: items with stable IDs."""

    def __init__(self, device: str, models: dict[str, int]) -> None:
        super().__init__(device, sum(models.values()))
        self.models = dict(models)

    async def shed(self, pressure: PressureSignal) -> int:
        if pressure.items is None:
            return await super().shed(pressure)
        freed = 0
        for item_id in pressure.items:
            nbytes = self.models.pop(item_id, 0)
            freed += nbytes
            self.holding -= nbytes
        return freed

    def details(self) -> list[ConsumerItem]:
        return [
            ConsumerItem(
                item_id=item_id,
                display_name=f"checkpoint {item_id}.safetensors",
                bytes_by_residency={self.device: nbytes},
                pages=PageMap(page_bytes=32, flags=(1, 1, 0)),
            )
            for item_id, nbytes in sorted(self.models.items())
        ]


def test_status_details_serves_items_with_geometry() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        pool = DetailedPool("vram:cuda:0", {"ckpt-a": 40})
        plain = TrimmableCache("vram:cuda:0", 10)
        governor.register_shedder(pool, name="models")
        governor.register_shedder(plain, name="plain")
        client = await make_governed_client(governor)
        try:
            # The cheap default: no item lists.
            data = await (await client.get("/memory/status")).json()
            assert "consumerDetails" not in data

            data = await (await client.get("/memory/status?details=1")).json()
            # Only detail-contract consumers are itemized; the plain one
            # keeps its footprint number in the governor view.
            assert set(data["consumerDetails"]) == {"models"}
            (item,) = data["consumerDetails"]["models"]
            assert item == {
                "itemId": "ckpt-a",
                "displayName": "checkpoint ckpt-a.safetensors",
                "bytesByResidency": {"vram:cuda:0": 40},
                # Geometry travels with the data: page size and count are
                # the server's facts, never a client-side constant.
                "pages": {"pageBytes": 32, "pageCount": 3, "flags": [1, 1, 0]},
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_trim_items_unloads_one_model_by_stable_id() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"vram:cuda:0": 100})
        pool = DetailedPool("vram:cuda:0", {"ckpt-a": 40, "ckpt-b": 30})
        governor.register_shedder(pool, name="models")
        client = await make_governed_client(governor)
        try:
            # "Unload ckpt-a", as a frontend button would express it.
            resp = await client.post(
                "/cache/trim",
                json={
                    "device": "vram:cuda:0",
                    "consumers": ["models"],
                    "items": ["ckpt-a"],
                },
            )
            assert resp.status == 200
            assert (await resp.json())["freedBytes"] == 40
            assert set(pool.models) == {"ckpt-b"}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_trim_items_validation() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        governor.register_shedder(DetailedPool("ram", {"a": 10}), name="models")
        client = await make_governed_client(governor)
        try:
            for body in [
                {"device": "ram", "items": ["a"]},  # no consumer
                {  # two consumers: whose namespace is "a" in?
                    "device": "ram",
                    "consumers": ["models", "other"],
                    "items": ["a"],
                },
                {"device": "ram", "consumers": ["models"], "items": [1]},
            ]:
                resp = await client.post("/cache/trim", json=body)
                assert resp.status == 400, body
        finally:
            await client.close()

    asyncio.run(scenario())


def test_status_reports_measured_telemetry_over_the_wire() -> None:
    async def scenario() -> None:
        governor = MemoryGovernor(
            {"vram:cuda:0": 100},
            telemetry=lambda d: MeasuredMemory(free_bytes=7, total_bytes=24),
        )
        client = await make_governed_client(governor)
        try:
            data = await (await client.get("/memory/status")).json()
            assert data["memoryGovernor"]["vram:cuda:0"]["measured"] == {
                "freeBytes": 7,
                "totalBytes": 24,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


# -- frozen-run value peeking (/api/values, DESIGN 3.5) -------------------------


class CurveOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.curve_output",
            outputs=(OutputSpec("curve", TypeExpr.concrete(CURVE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            curve=Curve(((0.0, 0.125), (1.25, 0.875), (4.0, 0.25)), "monotone_cubic")
        )


VALUES_NODES: list[type[Node]] = NODES + SCAFFOLD_NODES + [CurveOutput]
VALUES_SCHEMAS = build_schemas(VALUES_NODES)


async def make_values_client(cache_entries: int = 1024) -> TestClient:
    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        register_curve_type(registry)
        return Engine(
            schemas=VALUES_SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(VALUES_NODES), registry),
            cache=MemoryLRUCache(max_entries=cache_entries),
            on_event=on_event,
        )

    app = create_app(
        factory,
        VALUES_SCHEMAS,
        choices=scaffold_choices(),
        lazy_choices=scaffold_lazy_choices(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def finish_job(
    client: TestClient, graph: Graph, targets: list[str], job_id: str = "j1"
) -> Any:
    resp = await client.post("/api/jobs", json=submit_body(graph, targets, jobId=job_id))
    assert resp.status == 202
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed", "cancelled"):
                assert status["state"] == "completed"
                return status
            await asyncio.sleep(0.01)


def values_url(node_id: str, output_id: str = "out", job_id: str = "j1", **extra: str) -> str:
    params = {
        "clientId": "c1",
        "jobId": job_id,
        "nodeId": node_id,
        "outputId": output_id,
        **extra,
    }
    return "/api/values?" + "&".join(f"{k}={v}" for k, v in params.items())


def test_job_result_descriptors_inline_small_scalars() -> None:
    """Completed-job descriptors carry declared-safe inline scalars: peeking
    an int/short string needs no second round trip. Omission means "not
    inline", never absence."""

    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"sp": GraphNode("test.splitter", {"text": "a b c"})})
            status = await finish_job(client, graph, ["sp"])
            words = status["outputs"]["sp"]["words"]
            assert "value" not in words  # lists never inline at the top level
            assert [e["value"] for e in words["elements"]] == ["a", "b", "c"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_serves_target_and_intermediate() -> None:
    """Lookup authority is execution-scoped identity: targets resolve from
    the retained job result, intermediates through the exact cache key the
    run's own events reported."""

    async def scenario() -> None:
        client = await make_values_client()
        try:
            await finish_job(client, echo_graph(), ["s"])
            resp = await client.get(values_url("s"))
            assert resp.status == 200
            data = await resp.json()
            assert data["available"] is True
            assert data["descriptor"]["typeId"] == "core.string"
            assert data["descriptor"]["value"] == "HELLO"
            assert data["descriptor"]["fingerprint"]
            assert data["renditions"] == []  # strings have no rich rendition

            resp = await client.get(values_url("e"))  # intermediate, retained
            data = await resp.json()
            assert (resp.status, data["available"]) == (200, True)
            assert data["descriptor"]["value"] == "hello"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_reports_eviction_never_substitutes() -> None:
    """An evicted intermediate is a structured refusal (410), never a
    silently different value - peek's whole safety story."""

    async def scenario() -> None:
        client = await make_values_client(cache_entries=1)
        try:
            await finish_job(client, echo_graph(), ["s"])
            # "s" evicted "e" from the single-entry cache; the target output
            # itself is retained on the job result regardless.
            resp = await client.get(values_url("e"))
            data = await resp.json()
            assert (resp.status, data["available"]) == (410, False)
            assert data["reason"] == "evicted"

            resp = await client.get(values_url("s"))
            assert (await resp.json())["descriptor"]["value"] == "HELLO"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_identity_refusals() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            await finish_job(client, echo_graph(), ["s"])

            resp = await client.get(values_url("s", job_id="nope"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "unknown-job")

            resp = await client.get(values_url("s", output_id="nope"))
            assert (resp.status, (await resp.json())["reason"]) == (
                404,
                "unknown-output",
            )

            resp = await client.get(values_url("ghost"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "not-retained")

            resp = await client.get("/api/values?clientId=c1&jobId=j1")
            assert resp.status == 400  # missing nodeId/outputId

            resp = await client.get(values_url("s", element="0"))
            assert (resp.status, (await resp.json())["reason"]) == (404, "bad-element")

            resp = await client.get(values_url("s", element="zero"))
            assert resp.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_refuses_running_job() -> None:
    """Frozen-view contract: values are served once the job is terminal;
    live values ride the event stream."""

    async def scenario() -> None:
        reset_sleeper()
        client = await make_values_client()
        try:
            graph = Graph(nodes={"n": GraphNode("test.sleeper", {"tag": "x"})})
            resp = await client.post("/api/jobs", json=submit_body(graph, ["n"]))
            assert resp.status == 202
            await Sleeper.entered.wait()
            resp = await client.get(values_url("n"))
            data = await resp.json()
            assert (resp.status, data["reason"]) == (409, "not-complete")
            Sleeper.gate.set()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_element_traversal() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"sp": GraphNode("test.splitter", {"text": "a b c"})})
            await finish_job(client, graph, ["sp"])

            resp = await client.get(values_url("sp", output_id="words", element="1"))
            data = await resp.json()
            assert data["descriptor"]["typeId"] == "core.string"
            assert data["descriptor"]["value"] == "b"

            resp = await client.get(values_url("sp", output_id="words", element="5"))
            data = await resp.json()
            assert (resp.status, data["reason"]) == (404, "bad-element")
            assert "length 3" in data["error"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_curve_points_rendition_is_authoritative_and_bounded() -> None:
    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"curve": GraphNode("test.curve_output", {})})
            status = await finish_job(client, graph, ["curve"])
            descriptor = status["outputs"]["curve"]["curve"]
            assert set(descriptor) == {"typeId", "fingerprint", "meta"}
            assert descriptor["typeId"] == CURVE_TYPE
            assert descriptor["meta"] == {"pointCount": 3, "start": 0.0, "end": 4.0}

            url = values_url("curve", output_id="curve")
            response = await client.get(url)
            discovery = await response.json()
            assert discovery["descriptor"] == descriptor
            assert discovery["renditions"] == [
                {
                    "kind": "curve-points",
                    "mime": "application/json",
                    "default": True,
                    "cacheKey": "curve-points",
                    "limits": {"points": 4096},
                }
            ]

            response = await client.get(url + "&rendition=curve-points")
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/json"
            assert await response.json() == {
                "interpolation": "monotone_cubic",
                "points": [
                    {"position": 0.0, "value": 0.125},
                    {"position": 1.25, "value": 0.875},
                    {"position": 4.0, "value": 0.25},
                ],
            }

            job = cast(TestServer, client.server).app[STATE_KEY].queue.get("c1", "j1")
            assert job is not None and job.result is not None
            outputs = cast(dict[str, dict[str, Value]], job.result.outputs)
            oversized = {"points": [{"position": index, "value": index} for index in range(4097)]}
            outputs["curve"]["curve"] = Value(
                type_id=CURVE_TYPE,
                fingerprint="oversized-curve",
                meta=ValueMeta({"pointCount": 4097, "start": 0.0, "end": 4096.0}),
                payload=PyObjPayload(oversized),
            )
            response = await client.get(url + "&rendition=curve-points")
            refusal = await response.json()
            assert (response.status, refusal["available"], refusal["reason"]) == (
                404,
                False,
                "rendition_unavailable",
            )
            assert "between 1 and 4096" in refusal["error"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_image_png_rendition() -> None:
    """Rich types return negotiated browser-renderable bytes: renditions are
    declared in the descriptor response, fetched by kind, and cached by the
    fingerprint-derived ETag (identity looks up, fingerprint validates)."""

    async def scenario() -> None:
        client = await make_values_client()
        try:
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 4, "height": 2})})
            await finish_job(client, graph, ["g"])
            url = values_url("g", output_id="image")

            resp = await client.get(url)
            data = await resp.json()
            assert data["descriptor"]["typeId"] == "dev.image"
            assert "value" not in data["descriptor"]  # tensors never inline
            assert data["renditions"] == [
                {
                    "kind": "png",
                    "mime": "image/png",
                    "default": True,
                    "cacheKey": f"png/{PNG_CONTAINER_VERSION}",
                }
            ]

            legacy_url = url + "&rendition=png"
            immutable_url = url + f"&rendition=png/{PNG_CONTAINER_VERSION}"
            assert immutable_url != legacy_url
            resp = await client.get(immutable_url)
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert resp.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            assert resp.headers["X-Dinkster-Type-Id"] == "dev.image"
            assert resp.headers["X-Dinkster-Rendition"] == "png"
            etag = resp.headers["ETag"]
            assert resp.headers["X-Dinkster-Fingerprint"] in etag
            legacy_etag = f'"{resp.headers["X-Dinkster-Fingerprint"]}/png"'
            assert etag != legacy_etag
            body = await resp.read()
            assert body[:8] == b"\x89PNG\r\n\x1a\n"

            # The unversioned compatibility URL always revalidates.
            resp = await client.get(legacy_url, headers={"If-None-Match": legacy_etag})
            assert resp.status == 200
            assert resp.headers["Cache-Control"] == "private, no-cache"
            assert resp.headers["ETag"] == etag
            assert await resp.read() == body
            resp = await client.get(legacy_url, headers={"If-None-Match": etag})
            assert resp.status == 304
            assert resp.headers["Cache-Control"] == "private, no-cache"

            # "default" remains a revalidating compatibility alias.
            resp = await client.get(url + "&rendition=default")
            assert resp.headers["X-Dinkster-Rendition"] == "png"
            assert resp.headers["Cache-Control"] == "private, no-cache"
            resp = await client.get(immutable_url, headers={"If-None-Match": etag})
            assert resp.status == 304

            resp = await client.get(url + "&rendition=png/unknown")
            assert resp.status == 406

            resp = await client.get(url + "&rendition=bogus")
            data = await resp.json()
            assert (resp.status, data["reason"]) == (406, "no-rendition")
            assert data["renditions"] == ["png"]
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("type_id", ["dinkster.layers", "comfy.LAYERS"])
def test_values_endpoint_layer_document_rendition(
    type_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    from dinkster_graph import TypedLiteral
    from dinkster_image_document.document import ImageDocument
    from dinkster_image_document.format import IMAGE_DOCUMENT_MEDIA_TYPE
    from dinkster_nodes_image import AddLayer, register_image_types

    from dinkster.comfy_compose import register_comfy_host_types

    pixels = np.array([[[[1, 0, 0, 1], [0, 1, 0, 0.5]]]], dtype=np.float32)
    expected = cast(ImageDocument, AddLayer.execute(image=pixels)["layers"]).data

    class CompatLayers(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.compat_layers",
                outputs=(OutputSpec("layers", TypeExpr.concrete("comfy.LAYERS")),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return AddLayer.execute(image=pixels)

    nodes: list[type[Node]] = [AddLayer] if type_id == "dinkster.layers" else [CompatLayers]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        if type_id == "dinkster.layers":
            register_image_types(registry)
        else:
            register_comfy_host_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    def forbid_raster(*args, **kwargs):
        pytest.fail("document retrieval must not render or read raster resources")

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas)))
        await client.start_server()
        try:
            node = (
                GraphNode(
                    "dinkster.layers.add",
                    {"image": TypedLiteral("dinkster.image", pixels.tolist())},
                )
                if type_id == "dinkster.layers"
                else GraphNode("test.compat_layers", {})
            )
            await finish_job(client, Graph(nodes={"layers": node}), ["layers"])
            monkeypatch.setattr(ImageDocument, "render", forbid_raster)
            monkeypatch.setattr(ImageDocument, "read_resource", forbid_raster)
            url = values_url("layers", output_id="layers")
            resp = await client.get(url)
            assert resp.status == 200
            data = await resp.json()
            assert data["available"] is True
            assert data["descriptor"]["typeId"] == type_id
            assert "value" not in data["descriptor"]
            renditions = [
                {
                    "kind": "image",
                    "mime": "image/png",
                    "default": True,
                    "cacheKey": "image",
                }
            ]
            assert data["renditions"] == (renditions if type_id == "dinkster.layers" else []) + [
                {
                    "kind": "document",
                    "mime": IMAGE_DOCUMENT_MEDIA_TYPE,
                    "default": type_id == "comfy.LAYERS",
                    "cacheKey": "document",
                }
            ]

            resp = await client.get(url + "&rendition=document")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == IMAGE_DOCUMENT_MEDIA_TYPE
            assert resp.headers["X-Dinkster-Type-Id"] == type_id
            assert resp.headers["X-Dinkster-Rendition"] == "document"
            fingerprint = data["descriptor"]["fingerprint"]
            assert resp.headers["X-Dinkster-Fingerprint"] == fingerprint
            etag = resp.headers["ETag"]
            assert etag == f'"{fingerprint}/document"'
            assert await resp.read() == expected

            monkeypatch.setattr(TypeRegistry, "render", forbid_raster)
            resp = await client.get(url + "&rendition=document", headers={"If-None-Match": etag})
            assert resp.status == 304
            assert resp.headers["ETag"] == etag
            assert await resp.read() == b""
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_comfy_image_png_rendition() -> None:
    """The compat host registration at the endpoint: a comfy.IMAGE value
    (numpy BHWC floats - exactly the form the compat npy codec decodes to
    host-side) DECLARES the png rendition on peek and serves valid PNG
    bytes of the first batch element. This is the contract the frontend
    negotiates against; it never hardcodes png."""
    import struct

    import numpy as np

    from dinkster.comfy_compose import COMFY_IMAGE_TYPE, register_comfy_host_types

    class CompatImage(Node):
        """Stands in for the compat worker: emits the host runtime form."""

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.compat_image",
                outputs=(OutputSpec("image", TypeExpr.concrete(COMFY_IMAGE_TYPE)),),
            )

        @classmethod
        async def execute(cls) -> Mapping[str, object]:
            batch = np.linspace(0.0, 1.0, 2 * 4 * 3 * 3, dtype=np.float32)
            return cls.outputs(image=batch.reshape(2, 4, 3, 3))

    nodes: list[type[Node]] = [CompatImage]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(factory, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"i": GraphNode("test.compat_image", {})})
            await finish_job(client, graph, ["i"])
            url = values_url("i", output_id="image")

            resp = await client.get(url)
            data = await resp.json()
            assert data["available"] is True
            assert data["descriptor"]["typeId"] == COMFY_IMAGE_TYPE
            assert "value" not in data["descriptor"]  # tensors never inline
            assert data["renditions"] == [
                {
                    "kind": "png",
                    "mime": "image/png",
                    "default": True,
                    "cacheKey": f"png/{PNG_CONTAINER_VERSION}",
                }
            ]

            resp = await client.get(url + "&rendition=png")
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/png"
            assert resp.headers["X-Dinkster-Type-Id"] == COMFY_IMAGE_TYPE
            body = await resp.read()
            assert body[:8] == b"\x89PNG\r\n\x1a\n"
            # IHDR carries the FIRST batch element's dimensions (H=4, W=3).
            width, height = struct.unpack(">II", body[16:24])
            assert (width, height) == (3, 4)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_omits_original_rendition_for_component_video() -> None:
    import numpy as np
    from dinkster_nodes_media_io.video_ops import AssembleVideo

    from dinkster.comfy_compose import COMFY_VIDEO_TYPE, register_comfy_host_types

    class ComponentVideo(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.component_video",
                outputs=(OutputSpec("video", TypeExpr.concrete(COMFY_VIDEO_TYPE)),),
            )

        @classmethod
        async def execute(cls) -> Mapping[str, object]:
            frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
            return cls.outputs(video=AssembleVideo.execute(images=frames, fps=4.0)["video"])

    nodes: list[type[Node]] = [ComponentVideo]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(factory, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"v": GraphNode("test.component_video", {})})
            await finish_job(client, graph, ["v"])
            url = values_url("v", output_id="video")
            response = await client.get(url)
            data = await response.json()
            assert response.status == 200
            assert data["descriptor"]["typeId"] == COMFY_VIDEO_TYPE
            assert data["renditions"] == []

            response = await client.get(url + "&rendition=original")
            data = await response.json()
            assert (response.status, data["reason"]) == (406, "no-rendition")
            assert data["renditions"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_values_endpoint_av_renditions_share_discovery_and_response_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AV discovery and bytes resolve MIME from the same value metadata."""
    import struct

    import numpy as np

    from dinkster.comfy_compose import (
        COMFY_AUDIO_TYPE,
        COMFY_VIDEO_TYPE,
        register_comfy_host_types,
    )
    from tests.test_video_probe import _encode

    mp4 = _encode("mp4", "libx264", "yuv420p")
    webm = _encode("webm", "libvpx-vp9", "yuv420p")
    waveform = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)

    class AVOutputs(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.av_outputs",
                outputs=(
                    OutputSpec("audio", TypeExpr.concrete(COMFY_AUDIO_TYPE), preview=True),
                    OutputSpec("mp4", TypeExpr.concrete(COMFY_VIDEO_TYPE), preview=True),
                    OutputSpec("webm", TypeExpr.concrete(COMFY_VIDEO_TYPE), preview=True),
                ),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(
                audio={"waveform": waveform, "sample_rate": 8_000},
                mp4={"container": "mp4", "bytes": mp4},
                webm={"container": "webm", "bytes": webm},
            )

    nodes: list[type[Node]] = [AVOutputs]
    schemas = build_schemas(nodes)
    expected_registry = TypeRegistry()
    register_core_types(expected_registry)
    register_comfy_host_types(expected_registry)
    expected_fingerprints = {
        "audio": expected_registry.wrap(
            COMFY_AUDIO_TYPE, {"waveform": waveform, "sample_rate": 8_000}
        ).fingerprint,
        "mp4": expected_registry.wrap(
            COMFY_VIDEO_TYPE, {"container": "mp4", "bytes": mp4}
        ).fingerprint,
        "webm": expected_registry.wrap(
            COMFY_VIDEO_TYPE, {"container": "webm", "bytes": webm}
        ).fingerprint,
    }

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas)))
        await client.start_server()
        try:
            await finish_job(client, Graph(nodes={"av": GraphNode("test.av_outputs", {})}), ["av"])
            cases = (
                ("audio", "wav", "audio/wav", None),
                ("mp4", "original", "video/mp4", mp4),
                ("webm", "original", "video/webm", webm),
            )
            for output_id, kind, mime, expected_body in cases:
                url = values_url("av", output_id=output_id)
                discovery = await (await client.get(url)).json()
                assert discovery["descriptor"]["fingerprint"] == expected_fingerprints[output_id]
                expected_renditions = [
                    {"kind": kind, "mime": mime, "default": True, "cacheKey": kind}
                ]
                if output_id == "audio":
                    expected_renditions.extend(
                        [
                            {
                                "kind": "waveform",
                                "mime": "image/png",
                                "default": False,
                                "cacheKey": "waveform/rgba-v1",
                                "version": "rgba-v1",
                                "parameters": ["batch", "waveform"],
                                "defaults": {"batch": "0"},
                                "limits": {"width": 2048, "height": 512, "pixels": 262144},
                            },
                            {
                                "kind": "window",
                                "mime": "audio/wav",
                                "default": False,
                                "cacheKey": "window/pcm16-v1",
                                "version": "pcm16-v1",
                                "parameters": ["batch", "window"],
                                "defaults": {"batch": "0"},
                                "limits": {"durationSeconds": 30, "sampleValues": 8388608},
                            },
                        ]
                    )
                assert discovery["renditions"] == expected_renditions

                response = await client.get(url + "&rendition=default")
                assert response.status == 200
                assert response.headers["Content-Type"] == mime
                assert (
                    response.headers["X-Dinkster-Fingerprint"] == expected_fingerprints[output_id]
                )
                assert response.headers["ETag"] == (f'"{expected_fingerprints[output_id]}/{kind}"')
                body = await response.read()
                if expected_body is None:
                    assert body[:12] == b"RIFF" + struct.pack("<I", 42) + b"WAVE"
                    assert struct.unpack("<3h", body[44:]) == (-32768, 0, 32767)
                else:
                    assert body == expected_body

            audio_url = values_url("av", output_id="audio")
            response = await client.get(
                audio_url + "&rendition=waveform/rgba-v1&batch=0&waveform=3x5"
            )
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            waveform_etag = response.headers["ETag"]
            assert waveform_etag == (
                f'"{expected_fingerprints["audio"]}/waveform/rgba-v1?batch=0&waveform=3x5"'
            )
            assert (await response.read()).startswith(b"\x89PNG\r\n\x1a\n")

            response = await client.get(audio_url + "&rendition=waveform&waveform=3x5")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "private, no-cache"
            assert response.headers["ETag"] == waveform_etag

            response = await client.get(
                audio_url + "&rendition=waveform/rgba-v1&batch=00&waveform=3x5",
                headers={"If-None-Match": waveform_etag},
            )
            assert response.status == 304
            assert response.headers["ETag"] == waveform_etag

            response = await client.get(
                audio_url + "&rendition=window/pcm16-v1&batch=0&window=0,0.00025"
            )
            assert response.status == 200
            assert response.headers["Content-Type"] == "audio/wav"
            assert response.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            assert response.headers["ETag"] == (
                f'"{expected_fingerprints["audio"]}/window/pcm16-v1?batch=0&window=0%2C0.00025"'
            )
            assert struct.unpack("<2h", (await response.read())[44:]) == (-32768, 0)

            for query in (
                "&rendition=waveform&waveform=3x5&waveform=4x5",
                "&rendition=waveform&window=0,1&waveform=3x5",
                "&rendition=waveform&batch=1&waveform=3x5",
                "&rendition=window&batch=0",
                f"&rendition=waveform&waveform={'1' * 5000}x1",
                f"&rendition=waveform&batch={'1' * 5000}&waveform=3x5",
                f"&rendition=window&window=0,{'1' * 5000}",
            ):
                response = await client.get(audio_url + query)
                data = await response.json()
                assert response.status == 400
                assert response.headers["Content-Type"].startswith("application/problem+json")
                assert data["reason"] == "invalid_rendition_request"

            import dinkster_values.audio_codec as codec
            from dinkster_values import RenditionUnavailable

            class UnavailableReader:
                def __init__(self, _obj: object) -> None:
                    pass

                def __enter__(self) -> UnavailableReader:
                    raise RenditionUnavailable("audio source is unavailable")

                def __exit__(self, *args: object) -> None:
                    pass

            monkeypatch.setattr(codec, "AudioWindowReader", UnavailableReader)
            response = await client.get(audio_url + "&rendition=waveform&waveform=3x5")
            data = await response.json()
            assert response.status == 404
            assert response.headers["Content-Type"].startswith("application/problem+json")
            assert data["reason"] == "rendition_unavailable"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_compat_save_target_reaches_execution_with_default_and_explicit_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compat host registry must wrap the SAVE_TARGET literal before a
    job reaches its worker. This is the no-ComfyUI regression for the real
    dinkster.save_image input contract: both its advertised default and an
    explicit graph value arrive as validated SaveTarget values and produce
    digest-addressed asset receipts."""
    from dinkster_assets import (
        ASSET_TYPE,
        SAVE_TARGET_TYPE,
        AssetError,
        SaveTarget,
        digest_bytes,
    )
    from dinkster_compat_comfy.native import mount_writer
    from dinkster_nodes_foundation import StringPrimitive
    from dinkster_nodes_media_io import SetSaveTargetPrefix, register_media_types

    from dinkster.comfy_compose import register_comfy_host_types

    output_root = tmp_path / "output"
    output_root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(output_root),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    save_target = TypeExpr.concrete(SAVE_TARGET_TYPE)

    class CompatSave(Node):
        seen: list[SaveTarget] = []

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="dinkster.save_image",
                inputs=(
                    InputSpec(
                        "target",
                        save_target,
                        required=False,
                        default={"mount": "comfy-output", "prefix": "ComfyUI"},
                    ),
                ),
                outputs=(OutputSpec("receipt", TypeExpr.concrete(ASSET_TYPE)),),
                idempotent=False,
                output_node=True,
            )

        @classmethod
        async def execute(cls, *, target: SaveTarget) -> Mapping[str, object]:
            if not isinstance(target, SaveTarget):
                raise AssetError("save target input must be a SaveTarget")
            receipt = mount_writer().save_bytes(
                target,
                f"saved:{target.prefix}".encode(),
                suffix=".bin",
                media_type="application/octet-stream",
            )
            cls.seen.append(target)
            return cls.outputs(receipt=receipt)

    nodes: list[type[Node]] = [StringPrimitive, SetSaveTargetPrefix, CompatSave]
    schemas = build_schemas(nodes)

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        register_comfy_host_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(factory, schemas)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            default_graph = Graph(nodes={"s": GraphNode("dinkster.save_image", {})})
            default_status = await finish_job(client, default_graph, ["s"])
            default_meta = default_status["outputs"]["s"]["receipt"]["meta"]
            default_payload = b"saved:ComfyUI"
            assert default_meta["digest"] == digest_bytes(default_payload)
            assert default_meta["virtualPath"].endswith("/ComfyUI_00001.bin")
            assert (output_root / "ComfyUI_00001.bin").read_bytes() == default_payload

            explicit = {"mount": "comfy-output", "prefix": "renders/scene"}
            explicit_graph = Graph(
                nodes={"s": GraphNode("dinkster.save_image", {"target": explicit})}
            )
            explicit_status = await finish_job(client, explicit_graph, ["s"], job_id="j2")
            explicit_meta = explicit_status["outputs"]["s"]["receipt"]["meta"]
            explicit_payload = b"saved:renders/scene"
            assert explicit_meta["digest"] == digest_bytes(explicit_payload)
            assert explicit_meta["virtualPath"].endswith("/renders/scene_00001.bin")
            assert (output_root / "renders" / "scene_00001.bin").read_bytes() == (explicit_payload)

            computed_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "computed/live-scene"}),
                    "t": GraphNode(
                        "dinkster.set_save_target_prefix",
                        {
                            "target": {
                                "mount": "comfy-output",
                                "prefix": "manual/fallback",
                            },
                            "prefix": Link("p", "value"),
                        },
                    ),
                    "s": GraphNode(
                        "dinkster.save_image",
                        {"target": Link("t", "save_target")},
                    ),
                }
            )
            computed_status = await finish_job(client, computed_graph, ["s"], job_id="j3")
            computed_meta = computed_status["outputs"]["s"]["receipt"]["meta"]
            computed_payload = b"saved:computed/live-scene"
            assert computed_meta["digest"] == digest_bytes(computed_payload)
            assert computed_meta["virtualPath"].endswith("/computed/live-scene_00001.bin")
            assert (
                output_root / "computed" / "live-scene_00001.bin"
            ).read_bytes() == computed_payload

            raw_string_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "raw/string"}),
                    "s": GraphNode("dinkster.save_image", {"target": Link("p", "value")}),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(raw_string_graph, ["s"], jobId="j4"),
            )
            assert response.status == 202
            raw_status = await poll_job(client, "j4")
            assert raw_status["state"] == "failed"
            assert "must be a SaveTarget" in json.dumps(raw_status)

            snapshot.write_text(
                json.dumps(
                    {
                        "mounts": [
                            {
                                "id": "comfy-output",
                                "root": str(output_root),
                                "mode": "read",
                            }
                        ]
                    }
                ),
                "utf-8",
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(computed_graph, ["s"], jobId="j5"),
            )
            assert response.status == 202
            read_only_status = await poll_job(client, "j5")
            assert read_only_status["state"] == "failed"
            assert "read-only" in json.dumps(read_only_status)

            unknown_mount_graph = Graph(
                nodes={
                    "p": GraphNode("dinkster.string", {"value": "computed/unknown"}),
                    "t": GraphNode(
                        "dinkster.set_save_target_prefix",
                        {
                            "target": {
                                "mount": "missing-output",
                                "prefix": "manual/fallback",
                            },
                            "prefix": Link("p", "value"),
                        },
                    ),
                    "s": GraphNode(
                        "dinkster.save_image",
                        {"target": Link("t", "save_target")},
                    ),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(unknown_mount_graph, ["s"], jobId="j6"),
            )
            assert response.status == 202
            unknown_status = await poll_job(client, "j6")
            assert unknown_status["state"] == "failed"
            assert "no ready mount 'missing-output'" in json.dumps(unknown_status)

            snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
            response = await client.post(
                "/api/jobs",
                json=submit_body(computed_graph, ["s"], jobId="j7"),
            )
            assert response.status == 202
            revoked_status = await poll_job(client, "j7")
            assert revoked_status["state"] == "failed"
            assert "no ready mount 'comfy-output'" in json.dumps(revoked_status)

            assert CompatSave.seen == [
                SaveTarget(mount="comfy-output", prefix="ComfyUI"),
                SaveTarget(**explicit),
                SaveTarget(mount="comfy-output", prefix="computed/live-scene"),
            ]
        finally:
            await client.close()

    asyncio.run(scenario())


# -- progressive schema announcement -------------------------------------------


class Late(Node):
    """A node type that does not exist at create_app time: announced onto
    the live surface mid-test, the progressive-startup shape."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="late.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


async def poll_job(client: TestClient, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(5):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed"):
                return status
            await asyncio.sleep(0.01)


def test_announce_grows_surface_bumps_epoch_and_orders_event() -> None:
    """ServerState.announce (progressive startup; hot reload rides the same
    seam): the served surface grows additively, the epoch bumps, and the
    schema_changed event fires only after /api/nodes serves the new
    surface - the settled frontend ordering contract. The announced type
    executes immediately: engine and routes grew before the event."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        # The worker already knows how to RUN late.echo (in dinkster-serve the
        # RoutingWorker gains the route before announce); the SERVED surface
        # is what grows here.
        worker = InProcessWorker(build_node_types([*NODES, Late]), registry)

        def engine_factory(on_event: EventListener) -> Engine:
            return Engine(
                schemas=SCHEMAS,
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=on_event,
            )

        app = create_app(engine_factory, SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == 1
            assert "late.echo" not in data["nodes"]

            # Before announcement the type is unknown to the engine: a job
            # naming it fails validation instead of executing.
            graph = Graph(nodes={"l": GraphNode("late.echo", {"text": "hi"})})
            body = submit_body(graph, ["l"], jobId="early")
            assert (await client.post("/api/jobs", json=body)).status == 202
            assert (await poll_job(client, "early"))["state"] == "failed"

            ws = await client.ws_connect("/api/events?clientId=c1")
            epoch = state.announce(
                build_schemas([Late]),
                {"late-pack": PackInfo(display_name="Late Pack")},
                {"late.echo": "late-pack"},
                execution_arms={"late.echo": ("native", "comfyui")},
            )
            assert epoch == 2

            # The event names the epoch; the surface it announces is
            # already served (fetch-on-event sees >= the event's epoch).
            async with asyncio.timeout(5):
                while True:
                    event = await ws.receive_json()
                    if event["type"] == "schema_changed":
                        break
            assert event == {"type": "schema_changed", "epoch": 2}
            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == 2
            assert data["nodes"]["late.echo"]["pack"] == "late-pack"
            assert data["nodes"]["late.echo"]["executionArms"] == ["native", "comfyui"]
            assert data["packs"]["late-pack"] == {"displayName": "Late Pack"}

            # And the announced type executes through the live queue.
            body = submit_body(graph, ["l"], jobId="after")
            assert (await client.post("/api/jobs", json=body)).status == 202
            status = await poll_job(client, "after")
            assert status["state"] == "completed", status

            assert (
                state.announce(
                    {},
                    {},
                    {},
                    execution_arms={"late.echo": ("comfyui",)},
                )
                == 3
            )
            data = await (await client.get("/api/nodes")).json()
            assert data["nodes"]["late.echo"]["executionArms"] == ["comfyui"]
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_announce_refuses_bad_deltas_without_mutating() -> None:
    """Announcement validation mirrors construction: node-type collisions,
    unknown-pack attribution, and attribution outside the delta all raise
    - and a refused delta changes NOTHING (epoch, surface, packs table),
    so a miswired host cannot leave the server half-announced."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        state = app[STATE_KEY]
        late = build_schemas([Late])

        with pytest.raises(ValueError, match="redefine"):
            state.announce(build_schemas([Echo]), {}, {})
        with pytest.raises(ValueError, match="unknown pack"):
            state.announce(late, {}, {"late.echo": "nope"})
        with pytest.raises(ValueError, match="not in"):
            state.announce(
                late,
                {"p": PackInfo(display_name="P")},
                {"test.echo": "p"},
            )
        assert state.schema_epoch == 1
        assert "late.echo" not in state.schemas
        assert "p" not in state.packs
        assert "late.echo" not in state.node_packs

        # A successful announce; then re-declaring its pack entry is legal
        # only when identical (the shared compat-entry shape).
        assert state.announce(late, {"p": PackInfo(display_name="P")}, {"late.echo": "p"}) == 2
        with pytest.raises(ValueError, match="different info"):
            state.announce({}, {"p": PackInfo(display_name="Different")}, {})
        assert state.announce({}, {"p": PackInfo(display_name="P")}, {}) == 3
        assert state.schemas["late.echo"] is late["late.echo"]

    asyncio.run(scenario())


def test_announce_rejects_comfy_alias_collisions_without_mutating() -> None:
    app = create_app(
        make_engine,
        SCHEMAS,
        packs={"native": PackInfo("Native", comfy_aliases=alias_registry())},
        node_packs={"test.echo": "native"},
    )
    state = app[STATE_KEY]
    late = build_schemas([Late])

    with pytest.raises(ValueError, match="collide on comfy alias record id"):
        state.announce(
            late,
            {
                "late-pack": PackInfo(
                    "Late",
                    comfy_aliases=alias_registry(carrier="late.echo"),
                )
            },
            {"late.echo": "late-pack"},
        )

    assert state.schema_epoch == 1
    assert "late.echo" not in state.schemas
    assert "late-pack" not in state.packs


def test_composition_narration_health_nodes_flag_and_events() -> None:
    """One source of truth, three read paths. While the host narrates
    composition: /api/health carries "composition" (the supervisor mirror),
    /api/nodes carries "composing": true (this table is not final), and
    subscribers see composition_progress chatter. complete_composition is
    the positive finish: a non-droppable composition_complete event naming
    the FINAL epoch, health back to bare ok, the nodes flag gone - and a
    no-op when nothing was composing (zero-pack hosts have no loading to
    finish, so they never emit a phantom completion)."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        state = app[STATE_KEY]
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            data = await (await client.get("/api/nodes")).json()
            assert "composing" not in data

            ws = await client.ws_connect("/api/events?clientId=c1")
            state.narrate_composition(1, 3)
            assert await (await client.get("/api/health")).json() == {
                "ok": True,
                "composition": {"done": 1, "total": 3, "phase": "packs"},
            }
            data = await (await client.get("/api/nodes")).json()
            assert data["composing"] is True
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event == {
                "type": "composition_progress",
                "done": 1,
                "total": 3,
                "phase": "packs",
            }

            state.complete_composition()
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event == {
                "type": "composition_complete",
                "epoch": state.schema_epoch,
            }
            assert await (await client.get("/api/health")).json() == {"ok": True}
            data = await (await client.get("/api/nodes")).json()
            assert "composing" not in data

            # Idempotent finish: nothing composing, no phantom event - the
            # next thing the subscriber sees is unrelated (queue_state from
            # a pause, proving no completion event was queued between).
            state.complete_composition()
            assert (await client.post("/api/queue/pause")).status == 200
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event["type"] == "queue_state"
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_composition_report_pack_failures_and_endpoint() -> None:
    """The per-pack composition report: seeded pending, marked announced or
    failed, served on /api/composition for the process lifetime. A failed
    pack broadcasts non-droppable pack_failed with the cached error,
    completion names the failures in an additive "failed" list, and the
    report survives completion - the startup-error cache that lets clients
    say "pack X failed to load: <why>" instead of "unknown node"."""

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS)
        state = app[STATE_KEY]
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Std-only host: complete at first fetch, empty report.
            data = await (await client.get("/api/composition")).json()
            assert data == {"epoch": 1, "packs": {}}

            # Duplicate labels get distinct rows: the second spec's failure
            # must not clobber the first's record.
            keys = state.seed_composition(["good", "bad", "good"])
            assert keys == ["good", "bad", "good#2"]
            state.narrate_composition(0, 3)
            data = await (await client.get("/api/composition")).json()
            assert data["composing"] is True
            assert data["progress"] == {"done": 0, "total": 3, "phase": "packs"}
            assert data["packs"] == {
                "good": {"state": "pending"},
                "bad": {"state": "pending"},
                "good#2": {"state": "pending"},
            }

            ws = await client.ws_connect("/api/events?clientId=c1")
            state.mark_pack_announced("good", 2)
            state.mark_pack_failed("bad", "worker exploded")
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event == {
                "type": "pack_failed",
                "pack": "bad",
                "error": "worker exploded",
            }
            state.mark_pack_failed("good#2", "duplicate pack name 'good'")
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event["pack"] == "good#2"

            # Failure text is path-redacted once at entry, so the cached
            # /api/composition row and the broadcast event agree.
            state.seed_composition(["leaky"])
            home_path = str(Path.home() / "packs" / "leaky" / "pack.toml")
            state.mark_pack_failed("leaky", f"cannot parse {home_path}")
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            assert event["pack"] == "leaky"
            assert home_path not in event["error"]
            assert "pack.toml" in event["error"]
            event_error = event["error"]

            # Completion names the failures - and the report outlives it.
            state.complete_composition()
            async with asyncio.timeout(5):
                event = await ws.receive_json()
            while event["type"] == "composition_progress":
                async with asyncio.timeout(5):
                    event = await ws.receive_json()
            assert event == {
                "type": "composition_complete",
                "epoch": state.schema_epoch,
                "failed": ["bad", "good#2", "leaky"],
            }
            data = await (await client.get("/api/composition")).json()
            assert "composing" not in data
            assert "progress" not in data
            leaky_row = data["packs"].pop("leaky")
            assert leaky_row["state"] == "failed"
            assert home_path not in leaky_row["error"]
            assert leaky_row["error"] == event_error
            assert data["packs"] == {
                "good": {"state": "announced", "epoch": 2},
                "bad": {"state": "failed", "error": "worker exploded"},
                "good#2": {"state": "failed", "error": "duplicate pack name 'good'"},
            }
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())
