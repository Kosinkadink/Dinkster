"""Pack diagnostics use persistent output metadata, not transient events."""

import asyncio
from contextvars import copy_context
from pathlib import Path
from types import MappingProxyType
from typing import cast

import dinkster_api.v1 as api
import dinkster_schema as schema
import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_protocol import Invocation
from dinkster_schema.reporting import capture_value_diagnostics
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker

from tests.test_media_contract import IMAGE, MASK, SemanticSource, register_semantic_types

STRING = api.TypeExpr.concrete(api.CORE_STRING)


def test_value_diagnostic_public_api() -> None:
    assert api.report_value_diagnostic is schema.report_value_diagnostic
    api.report_value_diagnostic("media_format_fallback", {"reason": "preserve_alpha"})


@pytest.mark.parametrize(
    ("code", "data", "message"),
    [
        ("", None, "non-empty string"),
        ("  ", None, "non-empty string"),
        (None, None, "non-empty string"),
        ("pack_code", {"code": "spoof"}, "reserved"),
        ("pack_code", {"nodeId": "spoof"}, "reserved"),
        ("pack_code", [], "mapping"),
        ("pack_code", {"bad": object()}, "JSON-representable"),
        ("pack_code", {"bad": float("nan")}, "JSON-representable"),
        ("pack_code", {"bad": float("inf")}, "JSON-representable"),
    ],
)
def test_invalid_diagnostics_are_programmer_errors(code, data, message) -> None:
    with pytest.raises(ValueError, match=message):
        api.report_value_diagnostic(code, data)


def test_diagnostic_snapshot_and_closed_nested_contexts() -> None:
    requested = {"codec": "h264", "channels": ["L", "R"]}
    data = MappingProxyType({"requested": requested})
    stale = None
    with capture_value_diagnostics() as outer:
        api.report_value_diagnostic("arbitrary_pack_code", data)
        requested["channels"].append("C")
        requested["codec"] = "changed"
        with pytest.raises(RuntimeError), capture_value_diagnostics() as failed:
            api.report_value_diagnostic("failed")
            stale = copy_context()
            raise RuntimeError("node failure")
        assert stale is not None
        stale.run(api.report_value_diagnostic, "late")
        api.report_value_diagnostic("outer")
    api.report_value_diagnostic("outside")
    assert failed == [{"code": "failed"}]
    assert outer == [
        {"code": "arbitrary_pack_code", "requested": {"codec": "h264", "channels": ["L", "R"]}},
        {"code": "outer"},
    ]


class DiagnosticString(api.Node):
    @classmethod
    def define_schema(cls):
        return api.NodeSchema(
            node_type="diagnostic.string",
            inputs=(api.InputSpec("value", STRING),),
            outputs=(api.OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, value):
        api.report_value_diagnostic("pack_specific_code", {"value": value})
        api.report_value_diagnostic("second")
        return cls.outputs(value=value)


def string_worker(nodes=(DiagnosticString,)):
    registry = TypeRegistry()
    register_core_types(registry)
    return registry, InProcessWorker(schema.build_node_types(nodes), registry)


def invocation(cls, registry, value="ok"):
    return Invocation(
        invocation_id=value,
        node_id=value,
        node_type=cls.schema().node_type,
        effective_schema=cls.schema(),
        inputs={"value": registry.wrap(api.CORE_STRING, value)},
    )


@pytest.mark.parametrize("mode", ["sync", "async", "to_thread"])
def test_diagnostics_without_observer_and_without_source_metadata_mutation(
    mode, monkeypatch
) -> None:
    async def scenario():
        registry, _ = string_worker()
        source = registry.wrap(api.CORE_STRING, "ok")
        wrapped = []
        wrap = registry.wrap

        def capture_wrap(type_id, obj):
            value = wrap(type_id, obj)
            wrapped.append(value)
            return value

        monkeypatch.setattr(registry, "wrap", capture_wrap)

        class ReturnValue(DiagnosticString):
            @classmethod
            def execute(cls, value):
                super().execute(value)
                return cls.outputs(value=source.resolve())

        class AsyncReturn(api.Node):
            @classmethod
            def define_schema(cls):
                return ReturnValue.schema()

            @classmethod
            async def execute(cls, value):
                await asyncio.sleep(0)
                if mode == "to_thread":
                    return await asyncio.to_thread(ReturnValue.execute, value)
                return ReturnValue.execute(value)

        node = ReturnValue if mode == "sync" else AsyncReturn
        worker = InProcessWorker(schema.build_node_types((node,)), registry)
        result = await worker.invoke(invocation(node, registry))
        assert result.error is None
        assert result.outputs is not None
        output = result.outputs["value"]
        assert output.resolve() == "ok"
        assert output.meta.get("valueDiagnostics") == [
            {"code": "pack_specific_code", "value": "ok"},
            {"code": "second"},
        ]
        assert source.meta.get("valueDiagnostics") is None
        assert output is not source
        assert wrapped[-1].meta.get("valueDiagnostics") is None
        assert output is not wrapped[-1]
        assert output.fingerprint == source.fingerprint

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["success", "failure", "cancel"])
def test_concurrent_failed_and_late_invocations_are_isolated(termination) -> None:
    async def scenario():
        entered = asyncio.Event()
        finish = asyncio.Event()
        late = asyncio.Event()
        tasks = []

        async def report_late():
            await late.wait()
            await asyncio.to_thread(api.report_value_diagnostic, "late")

        class Overlap(api.Node):
            @classmethod
            def define_schema(cls):
                return DiagnosticString.schema()

            @classmethod
            async def execute(cls, value):
                api.report_value_diagnostic("entered", {"value": value})
                if value == "first":
                    tasks.append(asyncio.create_task(report_late()))
                    entered.set()
                    await finish.wait()
                    if termination == "failure":
                        raise RuntimeError("failed invocation")
                else:
                    finish.set()
                return cls.outputs(value=value)

        registry, worker = string_worker((Overlap,))
        first = asyncio.create_task(worker.invoke(invocation(Overlap, registry, "first")))
        await entered.wait()
        if termination == "cancel":
            first.cancel()
        second = await worker.invoke(invocation(Overlap, registry, "second"))
        first_result = None
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            first_result = await first
        late.set()
        await asyncio.gather(*tasks)
        api.report_value_diagnostic("outside")
        third = await worker.invoke(invocation(Overlap, registry, "third"))
        for value, result in (("second", second), ("third", third)):
            assert result.outputs is not None
            assert result.outputs["value"].meta.get("valueDiagnostics") == [
                {"code": "entered", "value": value}
            ]
        if first_result is not None:
            if termination == "failure":
                assert first_result.error is not None and first_result.outputs is None
            else:
                assert first_result.outputs is not None
                assert first_result.outputs["value"].meta.get("valueDiagnostics") == [
                    {"code": "entered", "value": "first"}
                ]

    asyncio.run(scenario())


FALLBACK = {
    "code": "media_format_fallback",
    "outputId": "video",
    "requested": {
        "container": "mp4",
        "codec": "h264",
        "pixelFormat": "yuv420p",
        "channelLayout": "5.1",
    },
    "effective": {
        "container": "matroska",
        "codec": "ffv1",
        "pixelFormat": "gbrap16le",
        "channelLayout": "5.1",
    },
    "reason": "preserve_alpha_and_precision",
}


class PreservingFallback(api.Node):
    """Simulate a pack's format decision without encoding or selecting a format."""

    @classmethod
    def define_schema(cls):
        return api.NodeSchema(
            node_type="diagnostic.fallback",
            inputs=(api.InputSpec("image", IMAGE), api.InputSpec("mask", MASK)),
            outputs=(api.OutputSpec("video", IMAGE), api.OutputSpec("receipt", STRING)),
        )

    @classmethod
    def execute(cls, image, mask):
        api.report_value_diagnostic(
            "media_format_fallback",
            {key: value for key, value in FALLBACK.items() if key != "code"},
        )
        api.report_value_diagnostic("pack_specific_code")
        return cls.outputs(video=image, receipt="preserved")


class DroppingFallback(PreservingFallback):
    @classmethod
    def define_schema(cls):
        return api.NodeSchema(
            node_type="diagnostic.drop",
            inputs=(
                api.InputSpec("image", IMAGE),
                api.InputSpec("mask", MASK, mask_polarity="coverage"),
            ),
            outputs=(api.OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, image, mask):
        api.report_value_diagnostic("pack_specific_code")
        return cls.outputs(image=image[..., :3])


DIAGNOSTIC_NODES = (SemanticSource, PreservingFallback, DroppingFallback, DiagnosticString)


@pytest.mark.parametrize("boundary_kind", ["in_process", "isolated_shm", "remote_loopback"])
@pytest.mark.parametrize("observe_first", [False, True])
def test_persistent_diagnostics_transport_and_cache_replay(
    tmp_path, boundary_kind, observe_first
) -> None:
    from tests.test_remote import remote_worker, start_service, stop_service

    async def scenario():
        registry = TypeRegistry()
        register_core_types(registry)
        register_semantic_types(registry)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "diagnostic"\n[pack.entry]\n'
            'nodes = "tests.test_value_diagnostics:DIAGNOSTIC_NODES"\n'
            'types = "tests.test_media_contract:register_semantic_types"\n'
        )
        root = Path(__file__).resolve().parents[1]
        transports: list[BoundaryDiagnostic] = []
        proc = None
        if boundary_kind == "isolated_shm":
            worker = IsolatedWorker(
                manifest,
                registry,
                shm_threshold=1,
                on_diagnostic=transports.append,
                extra_env={"PYTHONPATH": str(root)},
            )
        elif boundary_kind == "remote_loopback":
            proc, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
            worker = remote_worker(host, port, registry, on_diagnostic=transports.append)
        else:
            worker = InProcessWorker(schema.build_node_types(DIAGNOSTIC_NODES), registry)
        try:
            await worker.start()
            cache = MemoryLRUCache()
            events: list[EngineEvent] = []
            for cached, node_id in ((False, "first"), (True, "current")):
                events.clear()
                engine = Engine(
                    schemas=worker.schemas,
                    registry=registry,
                    worker=worker,
                    cache=cache,
                    on_event=events.append if cached or observe_first else None,
                )
                graph = Graph(
                    nodes={
                        "source": GraphNode("test.semantic-source"),
                        node_id: GraphNode(
                            "diagnostic.fallback",
                            {
                                "image": Link("source", "image"),
                                "mask": Link("source", "mask"),
                            },
                        ),
                        "drop": GraphNode(
                            "diagnostic.drop",
                            {
                                "image": Link("source", "image"),
                                "mask": Link("source", "mask"),
                            },
                        ),
                        "string": GraphNode("diagnostic.string", {"value": "ok"}),
                    }
                )
                result = await engine.run(graph, [node_id, "drop", "string"])
                assert not result.diagnostics
                assert (node_id in result.cached) == cached
                assert ("string" in result.cached) == cached
                output = result.outputs[node_id]["video"]
                expected = SemanticSource.execute()["image"]
                np.testing.assert_array_equal(output.resolve(), expected)
                assert api.encode_image_array(output.resolve()) == api.encode_image_array(expected)
                expected_records = [FALLBACK, {"code": "pack_specific_code"}]
                for value in result.outputs[node_id].values():
                    assert value.meta.get("valueDiagnostics") == expected_records
                assert result.outputs["drop"]["image"].meta.get("valueDiagnostics") == [
                    {
                        "code": "mask_polarity_mismatch",
                        "inputId": "mask",
                        "expected": "coverage",
                        "actual": "transparency",
                    },
                    {"code": "alpha_dropped", "outputId": "image", "inputIds": ["image"]},
                    {"code": "pack_specific_code"},
                ]
                if cached or observe_first:
                    by_node = {
                        event.node_id: event.detail["diagnostics"]
                        for event in events
                        if event.kind == "value_diagnostics"
                    }
                    assert by_node[node_id] == [
                        {**item, "nodeId": node_id} for item in expected_records
                    ]
                    assert by_node["string"] == [
                        {"code": "pack_specific_code", "value": "ok", "nodeId": "string"},
                        {"code": "second", "nodeId": "string"},
                    ]
                    assert not any(event.kind == "node_event" for event in events)
                    record = cast("list[dict[str, object]]", by_node[node_id])[0]
                    cast("dict[str, object]", record["requested"])["codec"] = "observer-edit"
                    for value in result.outputs[node_id].values():
                        assert value.meta.get("valueDiagnostics") == expected_records
                else:
                    assert events == []
            if boundary_kind == "isolated_shm":
                fallback = next(d for d in transports if d.node_type == "diagnostic.fallback")
                assert next(s for s in fallback.outputs if s.edge_id == "video").transport == "shm"
        finally:
            await worker.close()
            if proc is not None:
                await stop_service(proc)

    asyncio.run(scenario())
