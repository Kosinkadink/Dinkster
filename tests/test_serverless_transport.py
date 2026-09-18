from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dinkster_caches import DiskCAS, MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_protocol import (
    ErrorHint,
    ExportSnapshot,
    Invocation,
    InvocationResult,
    NodeError,
    SavedArtifactCandidate,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputSpec,
    TypeExpr,
    build_node_types,
    report_progress,
)
from dinkster_values import (
    EncodedLatentTensor,
    ResidencyTable,
    TypeRegistry,
    ValueMeta,
    default_encode,
    encode_latent,
    make_list_value,
    register_core_types,
    register_resident_type,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import BoundaryError, encode_invocation

from dinkster.comfy_compose import register_comfy_host_types
from tools.serverless_transport import (
    DiskObjectStore,
    Envelope,
    InvocationHandler,
    Limits,
    OnDemandWorker,
    Transport,
    UnknownOutcome,
    UnsupportedCapability,
)

SOURCE = "test-deployment-source-and-lock-digest"
IMAGE = TypeExpr.concrete("comfy.IMAGE")
LATENT = TypeExpr.concrete("comfy.LATENT")


class Generate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.generate",
            outputs=(OutputSpec("image", IMAGE), OutputSpec("latent", LATENT)),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            image=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 12,
            latent={"samples": EncodedLatentTensor("float32", (1, 1, 2, 2), b"\0" * 16)},
        )


class Transform(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.transform",
            inputs=(InputSpec("image", IMAGE), InputSpec("latent", LATENT)),
            outputs=(OutputSpec("image", IMAGE), OutputSpec("latent", LATENT)),
        )

    @classmethod
    def execute(cls, image: Any, latent: Any) -> Mapping[str, object]:
        return cls.outputs(image=1 - image, latent={**latent, "metadata": {"processed": True}})


class CountingStore(DiskObjectStore):
    def __init__(self, cas: DiskCAS) -> None:
        super().__init__(cas)
        self.reads: list[str] = []

    async def get(self, digest: str, *, max_bytes: int) -> bytes:
        self.reads.append(digest)
        return await super().get(digest, max_bytes=max_bytes)


class Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = CountingStore(DiskCAS(root / "objects"))
        self.client_transport = self.transport("client")
        self.registry = self.client_transport.registry
        self.handler = self.new_handler("handler")
        self.requests: list[Envelope] = []
        self.responses: list[Envelope] = []
        self.worker = OnDemandWorker(
            self.client_transport, self.handler.worker.schemas, self.dispatch
        )

    def transport(self, name: str, *, limits: Limits | None = None) -> Transport:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)
        return Transport(
            registry,
            self.store,
            DiskCAS(self.root / name),
            source_identity=SOURCE,
            limits=limits or Limits(),
        )

    def new_handler(self, name: str) -> InvocationHandler:
        transport = self.transport(name)
        return InvocationHandler(
            transport, InProcessWorker(build_node_types((Generate, Transform)), transport.registry)
        )

    async def dispatch(self, envelope: Envelope) -> Envelope:
        self.requests.append(json.loads(json.dumps(envelope, allow_nan=False)))
        response = await self.handler(self.requests[-1])
        self.responses.append(json.loads(json.dumps(response, allow_nan=False)))
        return self.responses[-1]

    def invocation(self) -> Invocation:
        return Invocation(
            "invocation-1",
            "node-1",
            "test.transform",
            {
                key: self.registry.wrap("comfy.IMAGE" if key == "image" else "comfy.LATENT", value)
                for key, value in Generate.execute().items()
            },
            Transform.schema(),
            job_ref="job-1",
            attempt_id=3,
        )

    async def request(self) -> Envelope:
        invocation = self.invocation()
        header, blobs, _, _ = encode_invocation(self.client_transport.codec(), invocation)
        correlation = {
            key: header[key]
            for key in ("invocationId", "jobRef", "attemptId", "nodeId", "nodeType")
        }
        return await self.client_transport.pack(header, blobs, correlation)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_schema_transfer_uses_latest_served_wire(harness: Harness) -> None:
    async def scenario() -> None:
        request = await harness.request()
        assert request["schemaVersion"] == 45
        assert request["frame"]["effectiveSchema"]["schemaVersion"] == 45

    asyncio.run(scenario())


def test_prepare_and_catalog_do_not_dispatch(harness: Harness) -> None:
    async def scenario() -> None:
        assert harness.worker.schemas["test.generate"] == Generate.schema()
        await harness.worker.prepare(["test.generate", "test.transform"])
        with pytest.raises(UnsupportedCapability, match="unknown node"):
            await harness.worker.prepare(["missing"])
        assert harness.requests == []
        assert harness.store.cas.digests() == []

    asyncio.run(scenario())


def test_dynamic_output_descriptors_are_not_dispatched(harness: Harness) -> None:
    schema = NodeSchema(
        node_type="test.descriptors",
        inputs=(InputSpec("description", TypeExpr.concrete("core.string")),),
        output_descriptors=OutputDescriptorsSpec(
            "description", (OutputSpec("image", IMAGE),), max_entries=1
        ),
    )
    worker = OnDemandWorker(harness.client_transport, {schema.node_type: schema}, harness.dispatch)
    invocation = replace(harness.invocation(), node_type=schema.node_type, effective_schema=schema)
    with pytest.raises(UnsupportedCapability, match="dynamic interfaces"):
        asyncio.run(worker.prepare([schema.node_type]))
    with pytest.raises(UnsupportedCapability, match="dynamic interfaces"):
        asyncio.run(worker.invoke(invocation))
    assert not harness.requests
    assert not harness.store.cas.digests()


def test_real_engine_image_and_latent_graph(harness: Harness) -> None:
    async def scenario() -> None:
        engine = Engine(
            schemas=harness.worker.schemas,
            registry=harness.registry,
            worker=harness.worker,
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "generate": GraphNode("test.generate", {}),
                "transform": GraphNode(
                    "test.transform",
                    {
                        "image": Link("generate", "image"),
                        "latent": Link("generate", "latent"),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["transform"])
        assert set(result.executed) == {"generate", "transform"}
        expected = Transform.execute(**Generate.execute())
        np.testing.assert_array_equal(
            result.outputs["transform"]["image"].resolve(), expected["image"]
        )
        assert encode_latent(result.outputs["transform"]["latent"].resolve()) == encode_latent(
            expected["latent"]
        )
        assert len(harness.requests) == 2
        assert harness.store.reads
        assert (
            len(
                {
                    harness.store.cas.root,
                    harness.client_transport.cache.cas.root,
                    harness.handler.transport.cache.cas.root,
                }
            )
            == 3
        )
        cached = await engine.run(graph, ["transform"])
        assert not cached.executed
        assert len(harness.requests) == 2
        for envelope in harness.requests + harness.responses:
            for ref in envelope["blobs"]:
                blob = harness.store.cas.get(ref["digest"])
                assert blob is not None and len(blob) == ref["size"]
                assert set(ref) == {"digest", "size"}
            assert "numpy" not in json.dumps(envelope)

    asyncio.run(scenario())


@pytest.mark.parametrize("reuse_cache", [False, True])
def test_fresh_handlers_and_durable_cas(harness: Harness, reuse_cache: bool) -> None:
    async def scenario() -> None:
        request = await harness.request()
        first = await harness.handler(request)
        reads = len(harness.store.reads)
        count = len(harness.store.cas.digests())
        assert reads > 0
        second_handler = harness.new_handler("handler" if reuse_cache else "cold-handler")
        second = await second_handler(request)
        assert first["blobs"] == second["blobs"]
        assert len(harness.store.cas.digests()) == count
        assert (len(harness.store.reads) == reads) == reuse_cache
        # A new client with an empty cache can consume the persisted result after turnover.
        fresh_client = OnDemandWorker(
            harness.transport("fresh-client"),
            harness.worker.schemas,
            lambda _: asyncio.sleep(0, result=second),
        )
        result = await fresh_client.invoke(harness.invocation())
        assert result.outputs is not None
        expected = harness.registry.wrap(
            "comfy.IMAGE", Transform.execute(**Generate.execute())["image"]
        )
        assert result.outputs["image"].fingerprint == expected.fingerprint
        assert default_encode(dict(result.outputs["image"].meta.entries)) == default_encode(
            dict(expected.meta.entries)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["missing", "corrupt", "oversize", "wrong-size"])
def test_blob_integrity_before_worker_execution(
    harness: Harness, damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        request = await harness.request()
        ref = request["blobs"][1]
        hexpart = ref["digest"].split(":")[1]
        path = harness.store.cas.root / hexpart[:2] / hexpart
        if damage == "missing":
            path.unlink()
        elif damage == "corrupt":
            path.write_bytes(b"x" * ref["size"])
        elif damage == "oversize":
            path.write_bytes(b"x" * (ref["size"] + 1))
        else:
            ref["size"] += 1

        def forbidden(*args: Any, **kwargs: Any) -> None:
            pytest.fail("worker must not execute after invalid storage response")

        monkeypatch.setattr(harness.handler.worker, "invoke", forbidden)
        with pytest.raises((BoundaryError, FileNotFoundError)):
            await harness.handler(request)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "protocolVersion",
        "schemaVersion",
        "sourceIdentity",
        "invocationId",
        "jobRef",
        "attemptId",
        "nodeId",
        "nodeType",
    ],
)
def test_mismatched_result_identity(harness: Harness, field: str) -> None:
    async def scenario() -> None:
        async def dispatch(request: Envelope) -> Envelope:
            response = await harness.dispatch(request)
            target = response if field in response else response["correlation"]
            target[field] = target[field] + 1 if type(target[field]) is int else "wrong"
            return response

        harness.worker.dispatch = dispatch
        with pytest.raises(BoundaryError, match="mismatch"):
            await harness.worker.invoke(harness.invocation())
        assert len(harness.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "limits",
    [
        Limits(envelope_bytes=100),
        Limits(blob_bytes=1),
        Limits(payload_bytes=1),
        Limits(blob_count=1),
    ],
)
def test_send_limits_prevent_dispatch(harness: Harness, limits: Limits) -> None:
    harness.client_transport.limits = limits
    with pytest.raises(BoundaryError, match="limit|count"):
        asyncio.run(harness.worker.invoke(harness.invocation()))
    assert not harness.requests
    assert not harness.store.cas.digests()


@pytest.mark.parametrize(
    "limits",
    [
        Limits(envelope_bytes=100),
        Limits(blob_bytes=1),
        Limits(payload_bytes=1),
        Limits(blob_count=1),
    ],
)
def test_receive_limits_prevent_reads(harness: Harness, limits: Limits) -> None:
    async def scenario() -> None:
        request = await harness.request()
        harness.handler.transport.limits = limits
        with pytest.raises(BoundaryError, match="limit|count"):
            await harness.handler(request)
        assert not harness.store.reads

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"executor": "gpu"},
        {"arm": "gpu"},
        {"expected_execution_identity": "execution"},
        {"extension_snapshot_digest": "sha256:" + "a" * 64},
        {"export_snapshot": ExportSnapshot({})},
        {"preview_mode": "cheap"},
        {"connected_undemanded_inputs": ("image",)},
    ],
)
def test_unsupported_invocation_is_not_dispatched(
    harness: Harness, changes: dict[str, Any]
) -> None:
    with pytest.raises(UnsupportedCapability):
        asyncio.run(harness.worker.invoke(replace(harness.invocation(), **changes)))
    assert not harness.requests


@pytest.mark.parametrize(
    "type_id",
    [
        "dinkster.model",
        "dinkster.clip",
        "dinkster.vae",
        "core.resource",
        "comfy.MODEL",
        "comfy.CLIP",
        "comfy.VAE",
        "custom.resident",
    ],
)
def test_resident_types_are_not_dispatched(harness: Harness, type_id: str) -> None:
    invocation = harness.invocation()
    resident = replace(invocation.inputs["image"], type_id=type_id)
    with pytest.raises(UnsupportedCapability, match="resident"):
        asyncio.run(harness.worker.invoke(replace(invocation, inputs={"image": resident})))
    assert not harness.requests


def test_resident_metadata_is_not_dispatched(harness: Harness) -> None:
    invocation = harness.invocation()
    resident = replace(invocation.inputs["image"], meta=ValueMeta({"resourceOwner": "dead-worker"}))
    with pytest.raises(UnsupportedCapability, match="resident"):
        asyncio.run(harness.worker.invoke(replace(invocation, inputs={"image": resident})))
    assert not harness.requests


def test_preserves_node_errors_and_hints(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = NodeError(
        "node-1",
        "test.transform",
        "failure",
        "traceback",
        (ErrorHint("code", "message", "suggestion"),),
    )

    async def invoke(*args: Any, **kwargs: Any) -> InvocationResult:
        return InvocationResult(error=expected)

    monkeypatch.setattr(harness.handler.worker, "invoke", invoke)
    result = asyncio.run(harness.worker.invoke(harness.invocation()))
    assert result == InvocationResult(error=expected)


@pytest.mark.parametrize("capability", ["artifact", "event", "continuation", "resident"])
def test_unsupported_worker_results_are_explicit(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    async def invoke(invocation: Invocation, on_event: Any) -> Any:
        if capability == "event":
            from dinkster_protocol import InvocationEvent

            on_event(InvocationEvent("progress", {"value": 1}, b"preview"))
        if capability == "continuation":
            return object()
        if capability == "resident":
            value = replace(invocation.inputs["image"], type_id="dinkster.model")
            return InvocationResult(outputs={"image": value})
        return InvocationResult(
            outputs={},
            artifact_candidates=(
                (SavedArtifactCandidate("node-1", "file.png", "", "output"),)
                if capability == "artifact"
                else ()
            ),
        )

    monkeypatch.setattr(harness.handler.worker, "invoke", invoke)
    with pytest.raises(UnknownOutcome) as exc:
        asyncio.run(harness.worker.invoke(harness.invocation()))
    assert isinstance(exc.value.__cause__, UnsupportedCapability)
    assert len(harness.requests) == 1


def test_unknown_outcome_does_not_retry(harness: Harness) -> None:
    async def scenario() -> None:
        async def dispatch(request: Envelope) -> Envelope:
            await harness.dispatch(request)
            raise ConnectionError("response lost after execution")

        harness.worker.dispatch = dispatch
        with pytest.raises(UnknownOutcome, match="do not retry"):
            await harness.worker.invoke(harness.invocation())
        assert len(harness.responses) == 1
        assert len(harness.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_during", ["dispatch", "result_read"])
def test_cancelled_engine_owner_does_not_redispatch_coalesced_work(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, cancel_during: str
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        coalesced = asyncio.Event()
        release = asyncio.Event()
        providers: list[asyncio.Task[Envelope]] = []
        engine = Engine(
            schemas=harness.worker.schemas,
            registry=harness.registry,
            worker=harness.worker,
            cache=MemoryLRUCache(),
        )
        graph = Graph(nodes={"generate": GraphNode("test.generate", {})})
        shield = asyncio.shield

        def observe_shield(future: Any) -> Any:
            if any(future is inflight for inflight in engine._inflight.values()):
                coalesced.set()
            return shield(future)

        async def provider(request: Envelope) -> Envelope:
            if cancel_during == "dispatch":
                started.set()
                await release.wait()
            return await harness.dispatch(request)

        async def dispatch(request: Envelope) -> Envelope:
            task = asyncio.create_task(provider(request))
            providers.append(task)
            return await asyncio.shield(task)

        unpack = harness.client_transport.unpack

        async def read_result(
            envelope: Envelope, kind: str, expected: Envelope | None = None
        ) -> tuple[Envelope, list[bytes], Envelope]:
            started.set()
            await release.wait()
            return await unpack(envelope, kind, expected)

        if cancel_during == "result_read":
            monkeypatch.setattr(harness.client_transport, "unpack", read_result)
        monkeypatch.setattr(asyncio, "shield", observe_shield)
        harness.worker.dispatch = dispatch
        owner = asyncio.create_task(engine.run(graph, ["generate"]))
        runs = [owner]
        try:
            await asyncio.wait_for(started.wait(), 5)
            waiter = asyncio.create_task(engine.run(graph, ["generate"]))
            runs.append(waiter)
            await asyncio.wait_for(coalesced.wait(), 5)
            assert len(providers) == 1
            owner.cancel()
            results = await asyncio.wait_for(asyncio.gather(*runs, return_exceptions=True), 5)
            for result in results:
                assert isinstance(result, ExceptionGroup)
                assert len(result.exceptions) == 1
                assert isinstance(result.exceptions[0], UnknownOutcome)
                assert isinstance(result.exceptions[0].__cause__, asyncio.CancelledError)
            assert len(providers) == 1
            assert providers[0].done() == (cancel_during == "result_read")
        finally:
            release.set()
            await asyncio.gather(*providers)
            for run in runs:
                if not run.done():
                    run.cancel()
            await asyncio.gather(*runs, return_exceptions=True)
        assert len(harness.requests) == len(harness.responses) == 1

    asyncio.run(scenario())


def test_cancellation_does_not_claim_remote_stop_or_retry(harness: Harness) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        tasks: list[asyncio.Task[Envelope]] = []

        async def provider(request: Envelope) -> Envelope:
            started.set()
            await release.wait()
            return await harness.dispatch(request)

        async def dispatch(request: Envelope) -> Envelope:
            task = asyncio.create_task(provider(request))
            tasks.append(task)
            return await asyncio.shield(task)

        harness.worker.dispatch = dispatch
        call = asyncio.create_task(harness.worker.invoke(harness.invocation()))
        await started.wait()
        call.cancel()
        with pytest.raises(UnknownOutcome, match="outcome unknown.*do not retry") as exc:
            await call
        assert isinstance(exc.value.__cause__, asyncio.CancelledError)
        assert not tasks[0].done()
        release.set()
        await tasks[0]
        assert len(tasks) == len(harness.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["executor", "continuation", "arm", "mediaSources", "events"])
def test_raw_unsupported_invocation_fields_fail_before_reads(harness: Harness, field: str) -> None:
    async def scenario() -> None:
        request = await harness.request()
        request["frame"][field] = "unsupported"
        with pytest.raises(UnsupportedCapability):
            await harness.handler(request)
        assert not harness.store.reads

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["artifacts", "artifactCandidates", "continuation", "events"])
def test_raw_unsupported_result_fields_are_not_dropped(harness: Harness, field: str) -> None:
    async def scenario() -> None:
        async def dispatch(request: Envelope) -> Envelope:
            response = await harness.dispatch(request)
            response["frame"][field] = []
            return response

        harness.worker.dispatch = dispatch
        with pytest.raises(UnsupportedCapability):
            await harness.worker.invoke(harness.invocation())
        assert len(harness.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["source", "schema", "lazy", "shm", "index", "binary", "nan"])
def test_rejects_raw_request_tampering(harness: Harness, damage: str) -> None:
    async def scenario() -> None:
        request = await harness.request()
        if damage == "source":
            request["sourceIdentity"] = "different-source"
        elif damage == "schema":
            request["frame"]["effectiveSchema"]["unknownCapability"] = True
        elif damage == "lazy":
            request["frame"]["effectiveSchema"]["interface"][0]["lazy"] = True
        elif damage == "shm":
            request["frame"]["inputs"]["image"]["payload"] = {
                "transport": "shm",
                "segment": "do-not-open",
                "size": 1,
            }
        elif damage == "index":
            request["frame"]["inputs"]["image"]["metaBlob"] = -1
        elif damage == "binary":
            request["frame"]["nodeId"] = b"binary-in-rpc"
        else:
            request["frame"]["nodeId"] = float("nan")
        with pytest.raises(BoundaryError):
            await harness.handler(request)

    asyncio.run(scenario())


def test_revalidates_noncompliant_store_bytes(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        request = await harness.request()

        async def bad_get(digest: str, *, max_bytes: int) -> bytes:
            return b"bad" * (max_bytes + 1)

        monkeypatch.setattr(harness.store, "get", bad_get)
        with pytest.raises(BoundaryError, match="fetched blob digest/size mismatch"):
            await harness.handler(request)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["error", "event"])
def test_real_inprocess_error_and_event_paths(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    def execute(**kwargs: Any) -> Mapping[str, object]:
        if mode == "error":
            raise RuntimeError("real node failure")
        report_progress(1, 1)
        return kwargs

    monkeypatch.setattr(Transform, "execute", execute)
    if mode == "error":
        result = asyncio.run(harness.worker.invoke(harness.invocation()))
        assert result.error is not None
        assert result.error.message == "real node failure"
        assert "RuntimeError: real node failure" in result.error.traceback
    else:
        with pytest.raises(UnknownOutcome) as exc:
            asyncio.run(harness.worker.invoke(harness.invocation()))
        assert isinstance(exc.value.__cause__, UnsupportedCapability)
        assert "events" in str(exc.value.__cause__)


def test_recursive_list_values_and_resident_child(harness: Harness) -> None:
    async def scenario() -> None:
        inner = make_list_value("core.int", (harness.registry.wrap("core.int", 7),))
        values = {"nested": make_list_value(inner.type_id, (inner,))}
        invocation = replace(harness.invocation(), inputs=values)
        header, blobs, _, _ = encode_invocation(harness.client_transport.codec(), invocation)
        request = await harness.request()
        envelope = await harness.client_transport.pack(header, blobs, request["correlation"])
        received, fetched, _ = await harness.handler.transport.unpack(envelope, "invoke")
        from dinkster_workers.boundary import decode_invocation

        decoded = decode_invocation(harness.handler.transport.codec(), received, fetched, [])
        assert decoded.inputs["nested"].resolve() == [[7]]
        assert decoded.inputs["nested"].fingerprint == values["nested"].fingerprint
        register_resident_type(harness.registry, "custom.resident", table=ResidencyTable())
        resident = harness.registry.wrap("custom.resident", object())
        # Even a mislabeled list cannot hide a child with owner-resolved state.
        nested = replace(
            values["nested"], payload=make_list_value("custom.resident", (resident,)).payload
        )
        with pytest.raises(UnsupportedCapability, match="resident"):
            await harness.worker.invoke(replace(invocation, inputs={"nested": nested}))
        assert not harness.requests

    asyncio.run(scenario())
