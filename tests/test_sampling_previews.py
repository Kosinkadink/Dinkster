"""Sampling-preview plumbing: the run-scoped PreviewPolicy resolves per
node and rides the invocation across the worker boundary as execution
data (never cache identity), the submit surface parses and fingerprints
the previews field, providers resolve declaratively, and the native-arm
emitter stays throttled, droppable, and never fatal to sampling."""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import tomllib
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy.preview_emit import (
    PREVIEW_DECODE_FAILURE_LIMIT,
    MultiStreamPreviewEmitter,
    SamplingPreviewEmitter,
    comfy_sampling_preview_emitter,
    multistream_sampling_preview_emitter,
    preview_stage,
    sampling_preview_emitter,
)
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_inference import (
    ANIMA,
    KREA2,
    LATENT2RGB_ANIMATION_PROVIDER,
    LATENT2RGB_PROVIDER,
    LATENT2RGB_WEBP_PROVIDER,
    LATENT2WAVEFORM_PROVIDER,
    TAEHV_PROVIDER,
    TAESD_PROVIDER,
    TRIPOSPLAT_SPLAT_PROVIDER,
    EncodedPreviewAnimation,
    LatentDescriptor,
    MultiStreamLatent,
    MultiStreamLatentDescriptor,
    PreviewClip,
    PreviewFrame,
    PreviewProviderRegistry,
    PreviewProviderSpec,
    builtin_preview_registry,
)
from dinkster_protocol import (
    Invocation,
    InvocationResult,
    OnInvocationEvent,
    PreviewPolicy,
    validate_preview_animation,
    validate_preview_mode,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import JobQueue, create_app
from dinkster_server.app import _parse_previews
from dinkster_values import TypeRegistry, Value, register_core_types
from dinkster_workers import ExecutionContext, InProcessWorker, current_execution_context
from dinkster_workers.boundary import (
    BoundaryError,
    ValueCodec,
    decode_invocation,
    encode_invocation,
)
from dinkster_workers.execution import use_execution_context

STRING = TypeExpr.concrete("core.string")

_REGISTRY = TypeRegistry()
register_core_types(_REGISTRY)


class Probe(Node):
    """Echoes the execution context's preview mode and animation transport
    as its output."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.previewprobe",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
            emits_previews=True,
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        context = current_execution_context()
        mode = context.preview_mode if context is not None else "missing"
        animation = context.preview_animation if context is not None else "missing"
        return cls.outputs(out=f"{tag}:{mode}:{animation}")


class UnflaggedProbe(Probe):
    """Same probe body, but the schema never declares emits_previews: the
    engine must resolve its effective preview mode to off regardless of
    policy."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.previewprobe-unflagged",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )


SCHEMAS = build_schemas([Probe, UnflaggedProbe])
SCHEMA = SCHEMAS["test.previewprobe"]


def _string(text: str) -> Value:
    return _REGISTRY.wrap("core.string", text)


def _invocation() -> Invocation:
    return Invocation(
        invocation_id="inv-1",
        node_id="p",
        node_type="test.previewprobe",
        inputs={"tag": _string("x")},
        effective_schema=SCHEMA,
        output_members=(("out", ()),),
        executor=None,
    )


def _graph(tag: str = "x") -> Graph:
    return Graph(nodes={"p": GraphNode("test.previewprobe", {"tag": tag})})


class RecordingWorker:
    """Fake worker recording every invocation it receives."""

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []

    async def prepare(self, node_types: object) -> None:
        pass

    async def invoke(
        self, invocation: Invocation, on_event: OnInvocationEvent | None = None
    ) -> InvocationResult:
        self.invocations.append(invocation)
        return InvocationResult(outputs={"out": _string("done")})


def _recording_engine(worker: RecordingWorker) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
    )


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types([Probe, UnflaggedProbe]), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


# -- protocol -----------------------------------------------------------------


def test_preview_mode_validation() -> None:
    for mode in ("off", "cheap", "quality", "auto"):
        assert validate_preview_mode(mode) == mode
    for bad in ("", "OFF", "fast", 3, None):
        with pytest.raises(ValueError, match="unsupported preview mode"):
            validate_preview_mode(bad)


def test_preview_policy_resolves_node_overrides_and_stays_immutable() -> None:
    policy = PreviewPolicy(mode="cheap", node_modes={"sampler": "quality"})
    assert policy.resolve("sampler") == "quality"
    assert policy.resolve("other") == "cheap"
    assert PreviewPolicy().resolve("anything") == "off"
    with pytest.raises(TypeError):
        cast("Any", policy.node_modes)["sampler"] = "off"


def test_preview_policy_rejects_malformed_content() -> None:
    with pytest.raises(ValueError, match="unsupported preview mode"):
        PreviewPolicy(mode=cast("Any", "fast"))
    with pytest.raises(ValueError, match="unsupported preview mode"):
        PreviewPolicy(node_modes=cast("Any", {"n": "fast"}))
    with pytest.raises(ValueError, match="non-empty strings"):
        PreviewPolicy(node_modes=cast("Any", {"": "off"}))
    with pytest.raises(ValueError, match="must be a mapping"):
        PreviewPolicy(node_modes=cast("Any", [("n", "off")]))


def test_invocation_validates_preview_mode() -> None:
    assert _invocation().preview_mode == "off"
    assert replace(_invocation(), preview_mode="quality").preview_mode == "quality"
    with pytest.raises(ValueError, match="unsupported preview mode"):
        replace(_invocation(), preview_mode=cast("Any", "fast"))


def test_execution_context_validates_preview_mode() -> None:
    context = ExecutionContext(arm=None, expected_execution_identity=None)
    assert context.preview_mode == "off"
    with pytest.raises(ValueError, match="unsupported preview mode"):
        ExecutionContext(
            arm=None,
            expected_execution_identity=None,
            preview_mode=cast("Any", "fast"),
        )


def test_preview_animation_validation() -> None:
    for animation in ("ring", "encoded"):
        assert validate_preview_animation(animation) == animation
    for bad in ("", "RING", "webp", 3, None):
        with pytest.raises(ValueError, match="unsupported preview animation"):
            validate_preview_animation(bad)


def test_preview_policy_carries_the_animation_transport() -> None:
    assert PreviewPolicy().animation == "ring"
    assert PreviewPolicy(mode="cheap", animation="encoded").animation == "encoded"
    with pytest.raises(ValueError, match="unsupported preview animation"):
        PreviewPolicy(animation=cast("Any", "webp"))


def test_invocation_validates_preview_animation() -> None:
    assert _invocation().preview_animation == "ring"
    assert replace(_invocation(), preview_animation="encoded").preview_animation == "encoded"
    with pytest.raises(ValueError, match="unsupported preview animation"):
        replace(_invocation(), preview_animation=cast("Any", "webp"))


def test_execution_context_validates_preview_animation() -> None:
    context = ExecutionContext(arm=None, expected_execution_identity=None)
    assert context.preview_animation == "ring"
    with pytest.raises(ValueError, match="unsupported preview animation"):
        ExecutionContext(
            arm=None,
            expected_execution_identity=None,
            preview_animation=cast("Any", "webp"),
        )


# -- worker boundary wire -----------------------------------------------------


def test_wire_omits_preview_mode_when_off_and_carries_it_otherwise() -> None:
    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation())
    try:
        assert "previewMode" not in header
        assert decode_invocation(ValueCodec(_REGISTRY), header, blobs, []).preview_mode == "off"
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()

    header, blobs, segments, _stats = encode_invocation(
        codec, replace(_invocation(), preview_mode="cheap")
    )
    try:
        assert header["previewMode"] == "cheap"
        assert decode_invocation(ValueCodec(_REGISTRY), header, blobs, []).preview_mode == "cheap"
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


@pytest.mark.parametrize("value", (3, "", "fast"))
def test_malformed_preview_mode_wire_field_is_loud(value: object) -> None:
    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation())
    header["previewMode"] = value
    try:
        with pytest.raises(BoundaryError, match="preview mode"):
            decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def test_wire_carries_preview_animation_only_when_previews_run_encoded() -> None:
    codec = ValueCodec(_REGISTRY)
    # Off previews leave the transport dormant; the ring default is implied.
    for invocation in (
        replace(_invocation(), preview_animation="encoded"),
        replace(_invocation(), preview_mode="cheap"),
    ):
        header, blobs, segments, _stats = encode_invocation(codec, invocation)
        try:
            assert "previewAnimation" not in header
            decoded = decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
            assert decoded.preview_animation == "ring"
        finally:
            for segment in segments:
                segment.close()
                segment.unlink()

    header, blobs, segments, _stats = encode_invocation(
        codec, replace(_invocation(), preview_mode="cheap", preview_animation="encoded")
    )
    try:
        assert header["previewAnimation"] == "encoded"
        decoded = decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
        assert decoded.preview_mode == "cheap"
        assert decoded.preview_animation == "encoded"
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


@pytest.mark.parametrize("value", (3, "", "webp"))
def test_malformed_preview_animation_wire_field_is_loud(value: object) -> None:
    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation())
    header["previewAnimation"] = value
    try:
        with pytest.raises(BoundaryError, match="preview animation"):
            decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


# -- engine -------------------------------------------------------------------


def test_policy_resolves_per_invocation_without_changing_cache_identity() -> None:
    async def scenario() -> None:
        worker = RecordingWorker()
        engine = _recording_engine(worker)
        policy = PreviewPolicy(mode="cheap", node_modes={"p": "quality"})
        await engine.run(_graph("a"), ["p"], preview_policy=policy)
        await engine.run(_graph("b"), ["p"], preview_policy=PreviewPolicy(mode="cheap"))
        await engine.run(_graph("c"), ["p"])
        assert [inv.preview_mode for inv in worker.invocations] == ["quality", "cheap", "off"]
        # A policy is execution data, never cache identity: the same graph
        # resubmitted without one is a plain cache hit.
        await engine.run(_graph("a"), ["p"])
        assert len(worker.invocations) == 3

    asyncio.run(scenario())


def test_animation_transport_rides_every_invocation_of_the_run() -> None:
    async def scenario() -> None:
        worker = RecordingWorker()
        engine = _recording_engine(worker)
        await engine.run(
            _graph("a"), ["p"], preview_policy=PreviewPolicy(mode="cheap", animation="encoded")
        )
        await engine.run(_graph("b"), ["p"], preview_policy=PreviewPolicy(mode="cheap"))
        await engine.run(_graph("c"), ["p"])
        assert [inv.preview_animation for inv in worker.invocations] == [
            "encoded",
            "ring",
            "ring",
        ]

    asyncio.run(scenario())


def test_malformed_preview_policy_does_not_poison_run_state() -> None:
    async def scenario() -> None:
        engine = _recording_engine(RecordingWorker())
        with pytest.raises(ValueError, match="PreviewPolicy"):
            await engine.run(
                _graph(),
                ["p"],
                run_id="reusable",
                preview_policy=cast("Any", {"mode": "cheap"}),
            )
        result = await engine.run(_graph(), ["p"], run_id="reusable")
        assert result.run_id == "reusable"

    asyncio.run(scenario())


def test_preview_mode_reaches_the_executing_node() -> None:
    async def scenario() -> None:
        engine = make_engine()
        with_policy = await engine.run(
            _graph("a"), ["p"], preview_policy=PreviewPolicy(mode="quality")
        )
        without = await engine.run(_graph("b"), ["p"])
        encoded = await engine.run(
            _graph("c"), ["p"], preview_policy=PreviewPolicy(mode="quality", animation="encoded")
        )
        assert with_policy.outputs["p"]["out"].resolve() == "a:quality:ring"
        assert without.outputs["p"]["out"].resolve() == "b:off:ring"
        assert encoded.outputs["p"]["out"].resolve() == "c:quality:encoded"

    asyncio.run(scenario())


def test_unflagged_schema_forces_preview_mode_off() -> None:
    """A node type whose schema never declares emits_previews executes with
    preview mode off no matter what the policy says - even an explicit
    per-node override. The declaration is the one gate both the frontend
    menu and the executing side read."""

    def _unflagged_graph(tag: str) -> Graph:
        return Graph(nodes={"p": GraphNode("test.previewprobe-unflagged", {"tag": tag})})

    async def scenario() -> None:
        worker = RecordingWorker()
        engine = _recording_engine(worker)
        await engine.run(
            _unflagged_graph("a"),
            ["p"],
            preview_policy=PreviewPolicy(mode="quality", node_modes={"p": "quality"}),
        )
        await engine.run(_graph("b"), ["p"], preview_policy=PreviewPolicy(mode="quality"))
        assert [inv.preview_mode for inv in worker.invocations] == ["off", "quality"]

    asyncio.run(scenario())


# -- server queue and submit surface ------------------------------------------


def test_queue_previews_ride_the_job_and_partition_idempotency() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_engine())
        policy = PreviewPolicy(mode="cheap")
        job = queue.submit("c1", "j1", _graph(), ["p"], previews=policy)
        assert job.preview_policy == policy
        duplicate = queue.submit("c1", "j1", _graph(), ["p"], previews=policy)
        assert duplicate is job
        with pytest.raises(ValueError, match="different content"):
            queue.submit("c1", "j1", _graph(), ["p"], previews=PreviewPolicy(mode="quality"))
        with pytest.raises(ValueError, match="different content"):
            queue.submit("c1", "j1", _graph(), ["p"])
        plain = queue.submit("c2", "j1", _graph(), ["p"])
        assert plain.preview_policy is None
        await queue.close()

    asyncio.run(scenario())


def test_parse_previews_defaults_and_validation() -> None:
    assert _parse_previews(None, default_mode="off") is None
    assert _parse_previews(None, default_mode="cheap") == PreviewPolicy(mode="cheap")
    parsed = _parse_previews({"mode": "quality", "nodes": {"s": "off"}}, default_mode="cheap")
    assert parsed == PreviewPolicy(mode="quality", node_modes={"s": "off"})
    assert _parse_previews({}, default_mode="cheap") == PreviewPolicy(mode="off")
    for malformed in (
        "cheap",
        {"mode": "fast"},
        {"mode": "cheap", "extra": True},
        {"mode": "cheap", "nodes": ["s"]},
        {"nodes": {"": "off"}},
        {"nodes": {"s": "fast"}},
        {"nodes": {3: "off"}},
    ):
        assert isinstance(_parse_previews(malformed, default_mode="cheap"), str)
    assert _parse_previews("cheap", default_mode="cheap") == (
        "'previews' must be an object with 'mode', optional 'nodes', and optional 'animation'"
    )
    assert _parse_previews({"mode": "cheap", "extra": True}, default_mode="cheap") == (
        "'previews' accepts only 'mode', 'nodes', and 'animation'"
    )


def test_parse_previews_animation_transport() -> None:
    # The server default transport rides both the implied and the explicit policy.
    assert _parse_previews(None, default_mode="cheap", default_animation="encoded") == (
        PreviewPolicy(mode="cheap", animation="encoded")
    )
    assert _parse_previews({"mode": "cheap"}, default_mode="off", default_animation="encoded") == (
        PreviewPolicy(mode="cheap", animation="encoded")
    )
    # An explicit transport wins over the default.
    assert _parse_previews(
        {"mode": "cheap", "animation": "ring"}, default_mode="off", default_animation="encoded"
    ) == PreviewPolicy(mode="cheap")
    assert _parse_previews({"mode": "cheap", "animation": "encoded"}, default_mode="off") == (
        PreviewPolicy(mode="cheap", animation="encoded")
    )
    for malformed in ({"animation": "webp"}, {"animation": 3}, {"mode": "cheap", "animation": ""}):
        assert isinstance(_parse_previews(malformed, default_mode="cheap"), str)


async def _make_client(**app_kwargs: object) -> TestClient:
    app = create_app(make_engine, SCHEMAS, **cast("Any", app_kwargs))
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _submit_body(tag: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": f"j-{tag}",
        "graph": graph_to_wire(_graph(tag)),
        "targets": ["p"],
    }
    body.update(overrides)
    return body


async def _completed_output(client: TestClient, body: Mapping[str, object]) -> str:
    resp = await client.post("/api/jobs", json=body)
    assert resp.status == 202, await resp.text()
    async with asyncio.timeout(5):
        while True:
            status = await (
                await client.get(f"/api/jobs/{body['clientId']}/{body['jobId']}")
            ).json()
            if status["state"] == "completed":
                break
            assert status["state"] not in ("failed", "cancelled"), status
            await asyncio.sleep(0.005)
    return cast("str", status["outputs"]["p"]["out"]["value"])


def test_submit_previews_default_field_and_override() -> None:
    async def scenario() -> None:
        client = await _make_client()
        try:
            # The server default (cheap) applies when the field is absent...
            assert await _completed_output(client, _submit_body("a")) == "a:cheap:ring"
            # ...and an explicit policy overrides it, per node included.
            body = _submit_body("b", previews={"mode": "off"})
            assert await _completed_output(client, body) == "b:off:ring"
            body = _submit_body("c", previews={"mode": "off", "nodes": {"p": "quality"}})
            assert await _completed_output(client, body) == "c:quality:ring"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_previews_off_default_disables_plumbing() -> None:
    async def scenario() -> None:
        client = await _make_client(preview_default="off")
        try:
            assert await _completed_output(client, _submit_body("a")) == "a:off:ring"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_previews_validation_and_idempotency_conflict() -> None:
    async def scenario() -> None:
        client = await _make_client()
        try:
            for previews in (
                None,
                "cheap",
                {"mode": "fast"},
                {"nodes": {"p": 3}},
                {"mode": "cheap", "animation": "webp"},
            ):
                resp = await client.post("/api/jobs", json=_submit_body("bad", previews=previews))
                assert resp.status == 400, await resp.text()

            null_resp = await client.post("/api/jobs", json=_submit_body("bad", previews=None))
            assert null_resp.status == 400
            assert (await null_resp.json())["error"] == (
                "'previews' must be an object with 'mode', optional 'nodes',"
                " and optional 'animation'"
            )

            assert (await client.post("/api/queue/pause")).status == 200
            body = _submit_body("held", previews={"mode": "cheap"})
            assert (await client.post("/api/jobs", json=body)).status == 202
            duplicate = await client.post("/api/jobs", json=body)
            assert duplicate.status == 202
            assert (await duplicate.json())["duplicate"] is True
            conflict = await client.post(
                "/api/jobs", json=_submit_body("held", previews={"mode": "quality"})
            )
            assert conflict.status == 409
            # The animation transport joins the fingerprint too.
            conflict = await client.post(
                "/api/jobs",
                json=_submit_body("held", previews={"mode": "cheap", "animation": "encoded"}),
            )
            assert conflict.status == 409
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_previews_animation_default_and_override() -> None:
    async def scenario() -> None:
        client = await _make_client(preview_animation="encoded")
        try:
            # The server default transport applies to the implied policy...
            assert await _completed_output(client, _submit_body("a")) == "a:cheap:encoded"
            # ...and to an explicit policy that leaves the transport unset...
            body = _submit_body("b", previews={"mode": "quality"})
            assert await _completed_output(client, body) == "b:quality:encoded"
            # ...while an explicit transport overrides it.
            body = _submit_body("c", previews={"mode": "quality", "animation": "ring"})
            assert await _completed_output(client, body) == "c:quality:ring"
        finally:
            await client.close()

    asyncio.run(scenario())


# -- provider registry ---------------------------------------------------------


def _descriptor(**overrides: object) -> LatentDescriptor:
    fields: dict[str, Any] = {"channels": 2}
    fields.update(overrides)
    return LatentDescriptor(**fields)


TAE_SPEC = PreviewProviderSpec(
    id="dinkster.tae", kind="image", cost="model", requires="taesd_decoder"
)
FAMILY_TAE_SPEC = PreviewProviderSpec(
    id="acme.tae", kind="image", cost="model", requires="taesd_decoder", family_id="acme"
)
CODEC_SPEC = PreviewProviderSpec(
    id="dinkster.vae", kind="animation", cost="model", requires="family_codec"
)


def _registry(*specs: PreviewProviderSpec) -> PreviewProviderRegistry:
    registry = PreviewProviderRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def test_registry_resolution_orders_by_mode_family_and_id() -> None:
    registry = _registry(LATENT2RGB_PROVIDER, TAE_SPEC, FAMILY_TAE_SPEC)
    descriptor = _descriptor(rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), taesd_decoder="tae_x")
    resolve = lambda mode, family: registry.resolve(  # noqa: E731
        descriptor, family_id=family, mode=mode
    )
    assert resolve("off", "acme") == ()
    assert resolve("cheap", "acme") == (LATENT2RGB_PROVIDER,)
    # Model-cost first, family-pinned before generic, cheap fallback last.
    assert resolve("quality", "acme") == (FAMILY_TAE_SPEC, TAE_SPEC, LATENT2RGB_PROVIDER)
    assert resolve("auto", "other") == (TAE_SPEC, LATENT2RGB_PROVIDER)


def test_registry_matches_descriptor_requirements() -> None:
    registry = _registry(LATENT2RGB_PROVIDER, TAE_SPEC, CODEC_SPEC)
    bare = _descriptor()
    assert registry.resolve(bare, family_id="f", mode="quality") == ()
    assert registry.resolve(_descriptor(taesd_decoder="tae_x"), family_id="f", mode="quality") == (
        TAE_SPEC,
    )
    # family_codec providers apply only when the caller vouches for a codec,
    # and non-image kinds only when asked for.
    assert registry.resolve(bare, family_id="f", mode="quality", family_codec=True) == ()
    assert registry.resolve(
        bare,
        family_id="f",
        mode="quality",
        kinds=("image", "animation"),
        family_codec=True,
    ) == (CODEC_SPEC,)


def test_registry_registration_is_idempotent_but_conflicts_are_loud() -> None:
    registry = _registry(TAE_SPEC)
    registry.register(TAE_SPEC)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(replace(TAE_SPEC, cost="cheap"))
    assert _registry(TAE_SPEC, LATENT2RGB_PROVIDER).specs() == (
        LATENT2RGB_PROVIDER,
        TAE_SPEC,
    )


def test_builtin_registry_serves_latent2rgb_only_when_factors_exist() -> None:
    registry = builtin_preview_registry()
    with_factors = _descriptor(rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert registry.resolve(with_factors, family_id="f", mode="cheap") == (LATENT2RGB_PROVIDER,)
    # A Wan-like descriptor (TAE decoder, no factors) has no cheap provider.
    assert registry.resolve(_descriptor(taesd_decoder="tae_x"), family_id="f", mode="cheap") == ()


def test_builtin_registry_orders_taesd_before_latent2rgb() -> None:
    registry = builtin_preview_registry()
    descriptor = _descriptor(rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), taesd_decoder="tae_x")
    quality = (TAESD_PROVIDER, LATENT2RGB_PROVIDER)
    assert registry.resolve(descriptor, family_id="f", mode="quality") == quality
    assert registry.resolve(descriptor, family_id="f", mode="auto") == quality
    assert registry.resolve(descriptor, family_id="f", mode="cheap") == (LATENT2RGB_PROVIDER,)


def test_builtin_registry_resolves_animation_specs_for_video_spaces() -> None:
    registry = builtin_preview_registry()
    video = _descriptor(
        dimensions=3,
        temporal_downscale=4,
        rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        taesd_decoder="tae_v",
    )
    assert registry.resolve(video, family_id="f", mode="quality", kinds=("animation",)) == (
        TAEHV_PROVIDER,
        LATENT2RGB_ANIMATION_PROVIDER,
    )
    assert registry.resolve(video, family_id="f", mode="cheap", kinds=("animation",)) == (
        LATENT2RGB_ANIMATION_PROVIDER,
    )
    # The dimensions pin keeps animation specs away from still-image spaces.
    still = _descriptor(rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), taesd_decoder="tae_x")
    assert registry.resolve(still, family_id="f", mode="quality", kinds=("animation",)) == ()


def test_builtin_registry_resolves_the_waveform_for_audio_spaces() -> None:
    registry = builtin_preview_registry()
    audio = _descriptor(dimensions=1)
    waveform = (LATENT2WAVEFORM_PROVIDER,)
    assert registry.resolve(audio, family_id="f", mode="cheap", kinds=("audio",)) == waveform
    assert registry.resolve(audio, family_id="f", mode="quality", kinds=("audio",)) == waveform
    # The dimensions pin keeps the waveform away from image/video spaces,
    # and audio providers serve only the audio kind.
    assert registry.resolve(_descriptor(), family_id="f", mode="cheap", kinds=("audio",)) == ()
    assert registry.resolve(audio, family_id="f", mode="cheap") == ()


def test_builtin_registry_resolves_the_splat_provider_for_triposplat() -> None:
    registry = builtin_preview_registry()
    tokens = _descriptor(channels=16, dimensions=1)
    family = "dinkster.triposplat"
    splat = (TRIPOSPLAT_SPLAT_PROVIDER,)
    assert registry.resolve(tokens, family_id=family, mode="quality", family_codec=True) == splat
    assert registry.resolve(tokens, family_id=family, mode="auto", family_codec=True) == splat
    # The decoder is the family's own model, so cheap mode excludes it.
    assert registry.resolve(tokens, family_id=family, mode="cheap", family_codec=True) == ()
    # Family-pinned: no other family resolves it, codec or not.
    assert registry.resolve(tokens, family_id="f", mode="quality", family_codec=True) == ()
    # The family_codec requirement gates resolution.
    assert registry.resolve(tokens, family_id=family, mode="quality") == ()
    # The dimensions pin keeps it away from spatial latent grids.
    grid = _descriptor(channels=16)
    assert registry.resolve(grid, family_id=family, mode="quality", family_codec=True) == ()


def test_encoded_animation_specs_resolve_only_when_asked_for() -> None:
    spec = PreviewProviderSpec(
        id="acme.webp", kind="encoded_animation", cost="cheap", requires="latent", dimensions=3
    )
    registry = _registry(spec, LATENT2RGB_ANIMATION_PROVIDER)
    video = _descriptor(dimensions=3, rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert registry.resolve(video, family_id="f", mode="cheap", kinds=("encoded_animation",)) == (
        spec,
    )
    assert registry.resolve(
        video, family_id="f", mode="cheap", kinds=("animation", "encoded_animation")
    ) == (spec, LATENT2RGB_ANIMATION_PROVIDER)
    assert registry.resolve(video, family_id="f", mode="cheap", kinds=("animation",)) == (
        LATENT2RGB_ANIMATION_PROVIDER,
    )


def test_encoded_preview_animation_validates_its_payload() -> None:
    EncodedPreviewAnimation(data=b"webp", mime="image/webp", width=2, height=2)
    with pytest.raises(ValueError, match="non-empty"):
        EncodedPreviewAnimation(data=b"", mime="image/webp", width=2, height=2)
    with pytest.raises(ValueError, match="mime"):
        EncodedPreviewAnimation(data=b"x", mime="text/plain", width=2, height=2)
    with pytest.raises(ValueError, match="dimensions"):
        EncodedPreviewAnimation(data=b"x", mime="video/mp4", width=0, height=2)


def test_preview_clip_validates_its_ring_addressing() -> None:
    frame = PreviewFrame(rgb=b"x", width=1, height=1)
    clip = PreviewClip(frames=(frame,), frame_indices=(1,), frame_count=2, fps=4.0)
    assert clip.frame_count == 2
    with pytest.raises(ValueError, match="at least one frame"):
        PreviewClip(frames=(), frame_indices=(), frame_count=1)
    with pytest.raises(ValueError, match="must align"):
        PreviewClip(frames=(frame,), frame_indices=(0, 1), frame_count=2)
    with pytest.raises(ValueError, match="frame_indices must lie"):
        PreviewClip(frames=(frame,), frame_indices=(2,), frame_count=2)
    with pytest.raises(ValueError, match="fps must be positive"):
        PreviewClip(frames=(frame,), frame_indices=(0,), frame_count=1, fps=0.0)


def test_pack_manifest_declares_the_catalog_preview_decoder_assets() -> None:
    from dinkster_inference import SD15, SDXL, WAN21, WAN22

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "dinkster-compat-comfy"
        / "dinkster-pack.toml"
    )
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assets = {entry["id"]: entry for entry in manifest["pack"]["assets"]}
    decoders = set()
    for family in (SD15, SDXL, WAN21, WAN22):
        descriptor = family.latent
        assert isinstance(descriptor, LatentDescriptor)
        assert descriptor.taesd_decoder is not None
        decoders.add(descriptor.taesd_decoder)
    assert set(assets) == decoders | {"triposplat_vae_decoder"}
    for asset_id, entry in assets.items():
        # TAE decoders are approximations; TripoSplat previews rasterize
        # through the family's real gaussian decoder.
        if asset_id == "triposplat_vae_decoder":
            assert entry["kind"] == "model/vae"
        else:
            assert entry["kind"] == "model/vae-approx"
        assert entry["digest"].startswith("blake3:")
        assert all(url.startswith("https://huggingface.co/") for url in entry["urls"])
        assert entry["size"] > 0
        # Previews degrade when the asset is absent; no node requires it.
        assert "nodes" not in entry


def test_preview_frame_rejects_empty_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        PreviewFrame(rgb=b"", width=0, height=1)


# -- native-arm emitter ---------------------------------------------------------


@dataclass
class FakeStateEvent:
    denoised: object = None
    current: object = None


class EmitterHarness:
    """Drives one SamplingPreviewEmitter with a manual clock and captures
    what it decodes and emits; ``wait`` joins the background encode."""

    def __init__(self, *, stream_role: str | None = None, blocking_encode: bool = False) -> None:
        self.now = 0.0
        self.decoded: list[object] = []
        self.emitted: list[tuple[bytes, str, int, int]] = []
        self.meta: list[dict[str, object]] = []
        self.gate = threading.Event()
        self._delivered = threading.Semaphore(0)
        if not blocking_encode:
            self.gate.set()

        def decode(state: object) -> PreviewFrame | PreviewClip:
            self.decoded.append(state)
            if isinstance(state, Exception):
                raise state
            if type(state) is PreviewClip:
                return state
            return PreviewFrame(rgb=state, width=3, height=2)

        def encode(frame: PreviewFrame) -> bytes:
            assert self.gate.wait(timeout=5)
            return b"jpeg:" + str(frame.rgb).encode()

        def emit(data: bytes, mime: str, width: int, height: int, **meta: object) -> None:
            self.emitted.append((data, mime, width, height))
            self.meta.append(meta)
            self._delivered.release()

        self.emitter = SamplingPreviewEmitter(
            decode, stream_role=stream_role, encode=encode, emit=emit, clock=lambda: self.now
        )

    def state(self, payload: object, *, at: float | None = None) -> None:
        if at is not None:
            self.now = at
        self.emitter.on_state(FakeStateEvent(denoised=payload))

    def wait(self, count: int) -> None:
        for _ in range(count):
            assert self._delivered.acquire(timeout=5)


def test_emitter_sends_first_frame_then_throttles() -> None:
    harness = EmitterHarness()
    harness.state("s1", at=0.0)
    harness.wait(1)
    harness.state("s2", at=0.1)  # inside the 0.2s window: dropped
    harness.state("s3", at=0.25)
    harness.wait(1)
    assert harness.decoded == ["s1", "s3"]
    assert [item[0] for item in harness.emitted] == [b"jpeg:s1", b"jpeg:s3"]
    assert harness.emitted[0][1:] == ("image/jpeg", 3, 2)


def test_emitter_drops_frames_while_the_encode_slot_is_busy() -> None:
    harness = EmitterHarness(blocking_encode=True)
    harness.state("s1", at=0.0)
    harness.state("s2", at=1.0)  # throttle satisfied, slot busy: dropped
    assert harness.decoded == ["s1"]
    harness.gate.set()
    harness.wait(1)
    harness.state("s3", at=2.0)
    harness.wait(1)
    assert harness.decoded == ["s1", "s3"]


def test_emitter_selects_the_named_stream_and_skips_missing_roles() -> None:
    streams = MultiStreamLatent.from_pairs([("video", "v-latent"), ("audio", "a-latent")])
    harness = EmitterHarness(stream_role="video")
    harness.state(streams, at=0.0)
    harness.wait(1)
    assert harness.decoded == ["v-latent"]

    unnamed = EmitterHarness()  # multi-stream state without a role: skipped
    unnamed.state(streams, at=0.0)
    missing = EmitterHarness(stream_role="depth")
    missing.state(streams, at=0.0)
    assert unnamed.decoded == [] and missing.decoded == []
    # The skip released the slot: a later matching event still emits.
    missing.state(
        MultiStreamLatent.from_pairs([("depth", "d-latent"), ("audio", "a-latent")]), at=1.0
    )
    missing.wait(1)
    assert missing.decoded == ["d-latent"]


def test_emitter_prefers_denoised_and_falls_back_to_current() -> None:
    harness = EmitterHarness()
    harness.emitter.on_state(FakeStateEvent(denoised=None, current="raw"))
    harness.wait(1)
    assert harness.decoded == ["raw"]


def test_emitter_survives_a_thread_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_compat_comfy.preview_emit as preview_emit_module

    class FailingThread(threading.Thread):
        def start(self) -> None:
            raise RuntimeError("no threads")

    harness = EmitterHarness()
    with monkeypatch.context() as patch:
        patch.setattr(preview_emit_module, "threading", SimpleNamespace(Thread=FailingThread))
        harness.state("s1", at=0.0)
    assert harness.decoded == ["s1"] and harness.emitted == []
    # The failed handoff released the slot and never counted as sent.
    harness.state("s2", at=0.0)
    harness.wait(1)
    assert [item[0] for item in harness.emitted] == [b"jpeg:s2"]


def test_emitter_swallows_decode_failures_and_recovers() -> None:
    harness = EmitterHarness()
    harness.state(RuntimeError("decoder exploded"), at=0.0)
    assert harness.emitted == []
    harness.state("good", at=0.0)  # failed frame never counted as sent
    harness.wait(1)
    assert [item[0] for item in harness.emitted] == [b"jpeg:good"]


def test_emitter_ships_encoded_animations_without_reencoding() -> None:
    emitted: list[tuple[bytes, str, int, int]] = []
    meta: list[dict[str, object]] = []
    delivered = threading.Semaphore(0)

    def emit(data: bytes, mime: str, width: int, height: int, **fields: object) -> None:
        emitted.append((data, mime, width, height))
        meta.append(fields)
        delivered.release()

    def encode(frame: PreviewFrame) -> bytes:
        raise AssertionError("an encoded animation must ship its bytes as-is")

    animation = EncodedPreviewAnimation(data=b"animated", mime="image/webp", width=4, height=2)
    emitter = SamplingPreviewEmitter(
        lambda state: animation, stream_role="video", encode=encode, emit=emit
    )
    emitter.on_state(FakeStateEvent(denoised="s1"))
    assert delivered.acquire(timeout=5)
    assert emitted == [(b"animated", "image/webp", 4, 2)]
    assert meta == [{"stream": "video", "frame_index": None, "frame_count": None, "fps": None}]


def test_emitter_converts_clips_with_the_animation_hook() -> None:
    emitted: list[tuple[bytes, str, int, int]] = []
    meta: list[dict[str, object]] = []
    delivered = threading.Semaphore(0)

    def emit(data: bytes, mime: str, width: int, height: int, **fields: object) -> None:
        emitted.append((data, mime, width, height))
        meta.append(fields)
        delivered.release()

    def encode(frame: PreviewFrame) -> bytes:
        raise AssertionError("a clip must take the animation hook, not per-frame stills")

    def encode_animation(clip: PreviewClip) -> EncodedPreviewAnimation:
        return EncodedPreviewAnimation(
            data=b"anim:%d" % clip.frame_count, mime="image/webp", width=9, height=7
        )

    frame = PreviewFrame(rgb=b"x", width=1, height=1)
    clip = PreviewClip(frames=(frame, frame), frame_indices=(0, 1), frame_count=2, fps=2.0)
    emitter = SamplingPreviewEmitter(
        lambda state: clip,
        stream_role="video",
        encode=encode,
        encode_animation=encode_animation,
        emit=emit,
    )
    emitter.on_state(FakeStateEvent(denoised="s1"))
    assert delivered.acquire(timeout=5)
    assert emitted == [(b"anim:2", "image/webp", 9, 7)]
    assert meta == [{"stream": "video", "frame_index": None, "frame_count": None, "fps": None}]


def test_webp_animation_encoder_produces_a_looping_container() -> None:
    import numpy as np
    from dinkster_compat_comfy.preview_emit import _encode_webp_animation
    from PIL import Image

    def frame(value: int) -> PreviewFrame:
        return PreviewFrame(rgb=np.full((2, 3, 3), value, dtype=np.uint8), width=3, height=2)

    clip = PreviewClip(
        frames=(frame(0), frame(128), frame(255)),
        frame_indices=(0, 1, 2),
        frame_count=3,
        fps=4.0,
    )
    animation = _encode_webp_animation(clip)
    assert animation.mime == "image/webp"
    assert animation.data[0:4] == b"RIFF" and animation.data[8:12] == b"WEBP"
    assert (animation.width, animation.height) == (3, 2)
    with Image.open(io.BytesIO(animation.data)) as image:
        assert getattr(image, "n_frames", 1) == 3
        assert image.size == (3, 2)
        assert image.info["loop"] == 0
        image.load()  # per-frame metadata appears once a frame decodes
        # duration derives from the clip's fps (1000ms / 4fps = 250ms per frame)
        assert image.info["duration"] == 250


def test_emitter_for_encoded_provider_ships_animated_webp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    frame = PreviewFrame(rgb=np.zeros((2, 3, 3), dtype=np.uint8), width=3, height=2)
    clip = PreviewClip(frames=(frame, frame), frame_indices=(0, 1), frame_count=2, fps=8.0)
    _fake_torch_backend(
        monkeypatch, {LATENT2RGB_WEBP_PROVIDER.id: lambda descriptor: lambda state: clip}
    )
    with use_execution_context(_context("cheap", "encoded")):
        emitter = sampling_preview_emitter(_handle(VIDEO_RGB_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)

    emitted: list[tuple[bytes, str, int, int]] = []
    delivered = threading.Semaphore(0)

    def emit(data: bytes, mime: str, width: int, height: int, **fields: object) -> None:
        emitted.append((data, mime, width, height))
        delivered.release()

    emitter._emit = emit  # noqa: SLF001
    emitter.on_state(FakeStateEvent(denoised="s1"))
    assert delivered.acquire(timeout=5)
    data, mime, width, height = emitted[0]
    assert mime == "image/webp"
    assert data[0:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert (width, height) == (3, 2)


def test_sampling_preview_emitter_is_none_without_execution_context() -> None:
    assert sampling_preview_emitter(cast("Any", object())) is None


def _handle(latent: object, family_id: str = "fam") -> Any:
    family = SimpleNamespace(id=family_id, latent=latent)
    return SimpleNamespace(runtime=SimpleNamespace(family=family))


def _fake_torch_backend(
    monkeypatch: pytest.MonkeyPatch, builders: Mapping[str, Any]
) -> list[LatentDescriptor]:
    built: list[LatentDescriptor] = []

    def wrap(builder: Any) -> Any:
        def build(descriptor: LatentDescriptor) -> Any:
            built.append(descriptor)
            return builder(descriptor)

        return build

    module = SimpleNamespace(
        preview_decoder_builders=lambda: {
            spec_id: wrap(builder) for spec_id, builder in builders.items()
        }
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", cast("Any", module))
    return built


RGB_DESCRIPTOR = LatentDescriptor(channels=2, rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def _context(mode: str, animation: str = "ring") -> ExecutionContext:
    return ExecutionContext(
        arm=None,
        expected_execution_identity=None,
        preview_mode=cast("Any", mode),
        preview_animation=cast("Any", animation),
    )


def test_sampling_preview_emitter_builds_a_decoder_for_the_resolved_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: lambda descriptor: decoder})
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(RGB_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [RGB_DESCRIPTOR]


def test_sampling_preview_emitter_selects_the_named_multistream_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _fake_torch_backend(
        monkeypatch, {LATENT2RGB_PROVIDER.id: lambda descriptor: lambda state: None}
    )
    plain = LatentDescriptor(channels=4)
    latent = MultiStreamLatentDescriptor(streams=(("audio", plain), ("video", RGB_DESCRIPTOR)))
    with use_execution_context(_context("cheap")):
        assert sampling_preview_emitter(_handle(latent), stream_role="video") is not None
        assert built == [RGB_DESCRIPTOR]
        # No role, an unknown role, or a stream with no usable provider:
        # previews silently stay off.
        assert sampling_preview_emitter(_handle(latent)) is None
        assert sampling_preview_emitter(_handle(latent), stream_role="depth") is None
        assert sampling_preview_emitter(_handle(latent), stream_role="audio") is None


def test_sampling_preview_emitter_skips_cheaply_when_nothing_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with use_execution_context(_context("off")):
        assert sampling_preview_emitter(_handle(RGB_DESCRIPTOR)) is None
    with use_execution_context(_context("cheap")):
        # No family on the runtime, or nothing a provider can use.
        assert sampling_preview_emitter(cast("Any", SimpleNamespace(runtime=object()))) is None
        assert sampling_preview_emitter(_handle(LatentDescriptor(channels=4))) is None
        # A resolved spec without a backend decoder builder degrades to None.
        _fake_torch_backend(monkeypatch, {})
        assert sampling_preview_emitter(_handle(RGB_DESCRIPTOR)) is None


VIDEO_RGB_DESCRIPTOR = LatentDescriptor(
    channels=2, dimensions=3, rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
)


def test_animation_transport_picks_the_provider_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen: list[str] = []

    def tagged(tag: str) -> Any:
        def build(descriptor: LatentDescriptor) -> Any:
            chosen.append(tag)
            return lambda state: PreviewFrame(rgb=state, width=1, height=1)

        return build

    _fake_torch_backend(
        monkeypatch,
        {
            LATENT2RGB_ANIMATION_PROVIDER.id: tagged("ring"),
            LATENT2RGB_WEBP_PROVIDER.id: tagged("encoded"),
        },
    )
    handle = _handle(VIDEO_RGB_DESCRIPTOR)
    # The encoded transport puts self-contained animation providers first...
    with use_execution_context(_context("cheap", "encoded")):
        assert sampling_preview_emitter(handle) is not None
    assert chosen == ["encoded"]
    # ...while the default ring transport keeps ring providers winning.
    chosen.clear()
    with use_execution_context(_context("cheap")):
        assert sampling_preview_emitter(handle) is not None
    assert chosen == ["ring"]


def test_encoded_transport_falls_back_to_ring_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen: list[str] = []

    def build(descriptor: LatentDescriptor) -> Any:
        chosen.append("ring")
        return lambda state: PreviewFrame(rgb=state, width=1, height=1)

    _fake_torch_backend(monkeypatch, {LATENT2RGB_ANIMATION_PROVIDER.id: build})
    with use_execution_context(_context("cheap", "encoded")):
        assert sampling_preview_emitter(_handle(VIDEO_RGB_DESCRIPTOR)) is not None
    assert chosen == ["ring"]


H3_LIKE_LATENT = MultiStreamLatentDescriptor(
    streams=(
        ("video", LatentDescriptor(channels=4, dimensions=3)),
        ("audio", LatentDescriptor(channels=2, dimensions=1)),
    )
)


def test_multistream_emitter_previews_the_audio_stream_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded: list[object] = []

    def decode(state: object) -> PreviewFrame:
        decoded.append(state)
        raise RuntimeError("decode inspected; stop before the encode slot")

    built = _fake_torch_backend(
        monkeypatch, {LATENT2WAVEFORM_PROVIDER.id: lambda descriptor: decode}
    )
    with use_execution_context(_context("cheap")):
        emitter = multistream_sampling_preview_emitter(_handle(H3_LIKE_LATENT))
    # Only the audio stream resolves a provider, so no fan-out wraps it.
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [H3_LIKE_LATENT.streams[1][1]]
    streams = MultiStreamLatent.from_pairs([("video", "v-latent"), ("audio", "a-latent")])
    emitter.on_state(FakeStateEvent(denoised=streams))
    assert decoded == ["a-latent"]


def test_multistream_emitter_fans_out_to_every_previewable_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded: list[tuple[str, object]] = []

    def builder_for(tag: str) -> Any:
        def build(descriptor: LatentDescriptor) -> Any:
            def decode(state: object) -> PreviewFrame:
                decoded.append((tag, state))
                raise RuntimeError("decode inspected; stop before the encode slot")

            return decode

        return build

    latent = MultiStreamLatentDescriptor(
        streams=(
            (
                "video",
                LatentDescriptor(
                    channels=2, dimensions=3, rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
                ),
            ),
            ("audio", LatentDescriptor(channels=2, dimensions=1)),
        )
    )
    _fake_torch_backend(
        monkeypatch,
        {
            LATENT2RGB_ANIMATION_PROVIDER.id: builder_for("video"),
            LATENT2WAVEFORM_PROVIDER.id: builder_for("audio"),
        },
    )
    with use_execution_context(_context("cheap")):
        emitter = multistream_sampling_preview_emitter(_handle(latent))
    assert isinstance(emitter, MultiStreamPreviewEmitter)
    streams = MultiStreamLatent.from_pairs([("video", "v-latent"), ("audio", "a-latent")])
    emitter.on_state(FakeStateEvent(denoised=streams))
    assert sorted(decoded) == [("audio", "a-latent"), ("video", "v-latent")]


def test_multistream_emitter_delegates_plain_descriptors_and_skips_cheaply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch_backend(
        monkeypatch, {LATENT2RGB_PROVIDER.id: lambda descriptor: lambda state: None}
    )
    with use_execution_context(_context("cheap")):
        emitter = multistream_sampling_preview_emitter(_handle(RGB_DESCRIPTOR))
        assert isinstance(emitter, SamplingPreviewEmitter)
        assert (
            multistream_sampling_preview_emitter(cast("Any", SimpleNamespace(runtime=object())))
            is None
        )
    with use_execution_context(_context("off")):
        assert multistream_sampling_preview_emitter(_handle(H3_LIKE_LATENT)) is None
    assert multistream_sampling_preview_emitter(cast("Any", object())) is None


def test_multistream_preview_emitter_stage_enters_every_child() -> None:
    entered: list[str] = []
    exited: list[str] = []

    def child(tag: str) -> SamplingPreviewEmitter:
        @contextmanager
        def stage() -> Any:
            entered.append(tag)
            try:
                yield None
            finally:
                exited.append(tag)

        return SamplingPreviewEmitter(
            lambda state: PreviewFrame(rgb=b"x", width=1, height=1), stage=stage
        )

    composite = MultiStreamPreviewEmitter((child("a"), child("b")))
    with preview_stage(composite):
        assert entered == ["a", "b"]
        assert exited == []
    assert exited == ["b", "a"]
    with pytest.raises(ValueError, match="at least two"):
        MultiStreamPreviewEmitter((child("solo"),))


def _comfy_model(fmt: object, load_device: object = "cpu") -> Any:
    return SimpleNamespace(model=SimpleNamespace(latent_format=fmt), load_device=load_device)


def _comfy_format(name: str, **attrs: Any) -> Any:
    """An instance of a freshly minted class named like a comfy latent format."""
    return type(name, (), attrs)()


class _TensorLike:
    """Just enough of a torch tensor: coercion happens through tolist()."""

    def __init__(self, listed: object) -> None:
        self._listed = listed

    def tolist(self) -> object:
        return self._listed


def test_comfy_emitter_is_none_without_context_or_when_off() -> None:
    fmt = _comfy_format("SD15", latent_channels=4)
    assert comfy_sampling_preview_emitter(_comfy_model(fmt)) is None
    with use_execution_context(_context("off")):
        assert comfy_sampling_preview_emitter(_comfy_model(fmt)) is None


def test_comfy_emitter_is_none_without_a_latent_format() -> None:
    with use_execution_context(_context("cheap")):
        assert comfy_sampling_preview_emitter(cast("Any", object())) is None
        assert comfy_sampling_preview_emitter(_comfy_model(None)) is None


def test_comfy_emitter_maps_known_formats_to_catalog_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: None  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: lambda descriptor: decoder})
    # The comfy instance carries no rgb factors; the catalog descriptor
    # (recognized by class name) supplies them.
    fmt = _comfy_format("SD15", latent_channels=4, latent_rgb_factors=None)
    with use_execution_context(_context("cheap")):
        emitter = comfy_sampling_preview_emitter(_comfy_model(fmt))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built[0].taesd_decoder == "taesd_decoder"
    assert built[0].scale_factor == 0.18215
    assert built[0].rgb_factors is not None


def test_comfy_emitter_maps_wan_to_the_catalog_video_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: None  # noqa: E731
    built = _fake_torch_backend(
        monkeypatch,
        {LATENT2RGB_ANIMATION_PROVIDER.id: lambda descriptor: decoder},
    )
    fmt = _comfy_format("Wan21", latent_channels=16, latent_dimensions=3)
    with use_execution_context(_context("cheap")):
        emitter = comfy_sampling_preview_emitter(_comfy_model(fmt))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built[0].dimensions == 3
    assert built[0].content_fps == 16.0
    assert built[0].taesd_decoder == "lighttaew2_1"


def test_comfy_emitter_builds_a_generic_descriptor_for_unknown_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: None  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: lambda descriptor: decoder})
    fmt = _comfy_format(
        "SomethingNew",
        latent_channels=2,
        scale_factor=0.5,
        latent_rgb_factors=_TensorLike([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        latent_rgb_factors_bias=_TensorLike([0.1, 0.2, 0.3]),
    )
    with use_execution_context(_context("cheap")):
        emitter = comfy_sampling_preview_emitter(_comfy_model(fmt))
    assert isinstance(emitter, SamplingPreviewEmitter)
    descriptor = built[0]
    assert descriptor.channels == 2
    assert descriptor.scale_factor == 0.5
    assert descriptor.rgb_factors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert descriptor.rgb_bias == (0.1, 0.2, 0.3)
    assert descriptor.taesd_decoder is None


def test_comfy_emitter_degrades_on_malformed_format_attributes() -> None:
    with use_execution_context(_context("cheap")):
        # Factors that do not coerce to one triple per channel: no provider
        # can serve the space, so previews silently stay off.
        wrong_shape = _comfy_format(
            "Weird",
            latent_channels=2,
            latent_rgb_factors=_TensorLike([[1.0, 0.0], [0.0, 1.0]]),
        )
        assert comfy_sampling_preview_emitter(_comfy_model(wrong_shape)) is None
        wrong_count = _comfy_format(
            "Weird",
            latent_channels=3,
            latent_rgb_factors=[[1.0, 0.0, 0.0]],
        )
        assert comfy_sampling_preview_emitter(_comfy_model(wrong_count)) is None
        # Valid factors with a bias that does not coerce: a silently
        # dropped bias would tint every frame, so previews stay off.
        bad_bias = _comfy_format(
            "Weird",
            latent_channels=1,
            latent_rgb_factors=[[1.0, 0.0, 0.0]],
            latent_rgb_factors_bias=_TensorLike([0.1, 0.2]),
        )
        assert comfy_sampling_preview_emitter(_comfy_model(bad_bias)) is None
        # Attributes that do not describe a latent space at all.
        no_channels = _comfy_format("Weird", latent_rgb_factors=[[1.0, 0.0, 0.0]])
        assert comfy_sampling_preview_emitter(_comfy_model(no_channels)) is None
        bad_dimensions = _comfy_format(
            "Weird",
            latent_channels=1,
            latent_dimensions=4,
            latent_rgb_factors=[[1.0, 0.0, 0.0]],
        )
        assert comfy_sampling_preview_emitter(_comfy_model(bad_dimensions)) is None


def test_emitter_stops_decoding_after_repeated_failures() -> None:
    harness = EmitterHarness()
    for _ in range(PREVIEW_DECODE_FAILURE_LIMIT):
        harness.state(RuntimeError("decoder exploded"), at=0.0)
    assert len(harness.decoded) == PREVIEW_DECODE_FAILURE_LIMIT
    harness.state("good", at=100.0)  # the emitter disabled itself
    assert len(harness.decoded) == PREVIEW_DECODE_FAILURE_LIMIT and harness.emitted == []


def test_preview_stage_and_stage_factory_failures_are_no_ops() -> None:
    with preview_stage(None):
        pass

    def broken_stage() -> Any:
        raise RuntimeError("stage exploded")

    frame = PreviewFrame(rgb=b"x", width=1, height=1)
    emitter = SamplingPreviewEmitter(lambda state: frame, stage=broken_stage)
    with preview_stage(emitter):  # staging is best-effort, never fatal
        pass


def test_preview_stage_swallows_exit_failures_but_not_body_exceptions() -> None:
    exits: list[str] = []

    @contextmanager
    def exploding_exit() -> Any:
        try:
            yield
        finally:
            exits.append("exited")
            raise RuntimeError("stage teardown exploded")

    frame = PreviewFrame(rgb=b"x", width=1, height=1)
    emitter = SamplingPreviewEmitter(lambda state: frame, stage=exploding_exit)
    with preview_stage(emitter):  # a teardown failure never fails the node
        pass
    assert exits == ["exited"]

    with pytest.raises(ValueError, match="sampling failed"):
        with preview_stage(emitter):  # the sampler's own failure survives
            raise ValueError("sampling failed")
    assert exits == ["exited", "exited"]


def _capture_report_log(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, Any]]:
    import dinkster_compat_comfy.preview_emit as preview_emit_module

    logs: list[tuple[str, str, Any]] = []
    monkeypatch.setattr(
        preview_emit_module,
        "report_log",
        lambda level, message, data=None: logs.append((level, message, data)),
    )
    return logs


TAESD_DESCRIPTOR = _descriptor(
    rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), taesd_decoder="tae_x"
)


def test_taesd_unavailability_logs_once_in_quality_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = _capture_report_log(monkeypatch)
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: lambda d: decoder})
    with use_execution_context(_context("quality")):
        # The handle's family "fam" has no supported TAESD architecture.
        emitter = sampling_preview_emitter(_handle(TAESD_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [TAESD_DESCRIPTOR]  # fell through to latent2rgb
    assert [(level, data["assetId"]) for level, _, data in logs] == [("info", "tae_x")]
    assert logs[0][2]["previewProvider"] == TAESD_PROVIDER.id


def test_taesd_unavailability_degrades_silently_outside_quality_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = _capture_report_log(monkeypatch)
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: lambda d: decoder})
    with use_execution_context(_context("auto")):
        emitter = sampling_preview_emitter(_handle(TAESD_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [TAESD_DESCRIPTOR] and logs == []


def test_quality_taesd_decoder_serves_the_emitter_and_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_compat_comfy.preview_emit as preview_emit_module

    _fake_torch_backend(monkeypatch, {})
    staged: list[str] = []

    @contextmanager
    def stage(*, observer_stage: str) -> Any:
        staged.append(observer_stage)
        yield

    entry = SimpleNamespace(
        handle=SimpleNamespace(released=False, stage=stage),
        decode=lambda state: PreviewFrame(rgb=b"x", width=1, height=1),
    )
    resolved: list[tuple[str, str, object]] = []

    def resolve(asset_id: str, family: str, load_device: object) -> Any:
        resolved.append((asset_id, family, load_device))
        return entry

    monkeypatch.setattr(preview_emit_module, "_resolve_taesd_decoder", resolve)
    handle = _handle(TAESD_DESCRIPTOR, family_id="dinkster.sd15")
    handle.load_device = "cuda:0"
    with use_execution_context(_context("quality")):
        emitter = sampling_preview_emitter(handle)
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert resolved == [("tae_x", "sd15", "cuda:0")]
    with preview_stage(emitter):
        assert staged == ["sample"]


TRIPOSPLAT_LIKE_LATENT = MultiStreamLatentDescriptor(
    streams=(
        ("latent", LatentDescriptor(channels=16, dimensions=1)),
        ("camera", LatentDescriptor(channels=5, dimensions=1)),
    )
)


def test_triposplat_emitter_serves_the_shape_code_stream_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_compat_comfy.preview_emit as preview_emit_module

    _fake_torch_backend(monkeypatch, {})
    staged: list[str] = []

    @contextmanager
    def stage(*, observer_stage: str) -> Any:
        staged.append(observer_stage)
        yield

    decoded: list[object] = []

    def decode(state: object) -> PreviewFrame:
        decoded.append(state)
        raise RuntimeError("decode inspected; stop before the encode slot")

    entry = SimpleNamespace(handle=SimpleNamespace(released=False, stage=stage), decode=decode)
    resolved: list[tuple[str, object]] = []

    def resolve(asset_id: str, load_device: object) -> Any:
        resolved.append((asset_id, load_device))
        return entry

    monkeypatch.setattr(preview_emit_module, "_resolve_triposplat_decoder", resolve)
    handle = _handle(TRIPOSPLAT_LIKE_LATENT, family_id="dinkster.triposplat")
    handle.load_device = "cuda:0"
    with use_execution_context(_context("quality")):
        emitter = multistream_sampling_preview_emitter(handle)
    # Only the 16-channel shape-code stream resolves a decoder; the camera
    # stream declines structurally, so no fan-out wraps the emitter.
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert resolved == [("triposplat_vae_decoder", "cuda:0")]
    streams = MultiStreamLatent.from_pairs([("latent", "shape-codes"), ("camera", "cams")])
    emitter.on_state(FakeStateEvent(denoised=streams))
    assert decoded == ["shape-codes"]
    with preview_stage(emitter):
        assert staged == ["sample"]


def test_triposplat_unavailability_logs_once_in_quality_and_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_compat_comfy.preview_emit as preview_emit_module

    logs = _capture_report_log(monkeypatch)
    _fake_torch_backend(monkeypatch, {})

    def resolve(asset_id: str, load_device: object) -> Any:
        raise RuntimeError("the decoder asset is not materializable")

    monkeypatch.setattr(preview_emit_module, "_resolve_triposplat_decoder", resolve)
    handle = _handle(TRIPOSPLAT_LIKE_LATENT, family_id="dinkster.triposplat")
    handle.load_device = "cuda:0"
    with use_execution_context(_context("quality")):
        assert multistream_sampling_preview_emitter(handle) is None
    assert [(level, data["assetId"]) for level, _, data in logs] == [
        ("info", "triposplat_vae_decoder")
    ]
    assert logs[0][2]["previewProvider"] == TRIPOSPLAT_SPLAT_PROVIDER.id
    logs.clear()
    with use_execution_context(_context("auto")):
        assert multistream_sampling_preview_emitter(handle) is None
    assert logs == []


def test_triposplat_decoder_build_binds_asset_identity_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoder build carries the declared asset's digest and size into
    planning, pins the loader to the planned identity, and pool-governs
    the enrolled module exactly like the TAE decoder builds."""
    import dinkster_compat_comfy.preview_emit as preview_emit_module
    from dinkster_inference import BFLOAT16

    calls: dict[str, Any] = {}
    path = Path("/vault/triposplat_vae_decoder.safetensors")
    ref = SimpleNamespace(
        digest="blake3:abc",
        size=42,
        name="TripoSplat VAE decoder",
        local_path=lambda: path,
    )

    def header(header_path: Path, *, asset_digest: str, asset_size: int) -> str:
        calls["header"] = (header_path, asset_digest, asset_size)
        return "source"

    def plan(source: str, *, role: str, path: Path) -> str:
        calls["plan"] = (source, role, path)
        return "planned"

    def identity(planned: str, dtype: object) -> str:
        calls["identity"] = (planned, dtype)
        return "expected-identity"

    def load(load_path: Path, **kwargs: Any) -> Any:
        calls["load"] = (load_path, kwargs)
        return SimpleNamespace(module="module")

    def enroll(module: str, *, load_device: object, offload_device: object) -> str:
        calls["enroll"] = (module, load_device, offload_device)
        return "mechanism"

    def preview_decoder(module: str) -> Any:
        calls["decoder"] = module
        return lambda state: PreviewFrame(rgb=b"x", width=1, height=1)

    class FakeHandle:
        def __init__(
            self,
            module: str,
            mechanism: str,
            load_device: object,
            *,
            resource_identity: str,
            coordinator: object,
        ) -> None:
            calls["handle"] = (module, mechanism, load_device, resource_identity, coordinator)
            self.released = False

        def attach_pool(self, pool: object) -> None:
            calls["attached"] = pool

    class FakeCoordinator:
        @staticmethod
        def enroll_component(
            module: object,
            *,
            load_device: object,
            offload_device: object,
            enroller: Any,
        ) -> object:
            return enroller(
                module,
                load_device=load_device,
                offload_device=offload_device,
            )

    pool = SimpleNamespace(
        register_invalidator=lambda drop: None,
        label=lambda handle, name: calls.setdefault("label", (handle, name)) and 7,
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch",
        cast(
            "Any",
            SimpleNamespace(
                load_triposplat_component=load,
                enroll_component=enroll,
                triposplat_preview_decoder=preview_decoder,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        cast("Any", SimpleNamespace(bfloat16="bf16", device=lambda name: f"device:{name}")),
    )
    monkeypatch.setattr(preview_emit_module, "declared_asset", lambda asset_id: ref)
    monkeypatch.setattr(preview_emit_module, "load_safetensors_header", header)
    monkeypatch.setattr(preview_emit_module, "plan_triposplat_split_component", plan)
    monkeypatch.setattr(preview_emit_module, "triposplat_component_runtime_identity", identity)
    monkeypatch.setattr(preview_emit_module, "NativeComponentHandle", FakeHandle)
    coordinator = FakeCoordinator()
    monkeypatch.setattr(preview_emit_module, "default_native_residency", lambda: coordinator)
    monkeypatch.setattr(preview_emit_module, "default_pool", lambda: pool)
    monkeypatch.setattr(preview_emit_module, "resident_resource_id", lambda rid: f"resident:{rid}")
    monkeypatch.setattr(preview_emit_module, "_triposplat_cache", {})

    entry = preview_emit_module._resolve_triposplat_decoder("triposplat_vae_decoder", "cuda:0")
    assert calls["header"] == (path, "blake3:abc", 42)
    assert calls["plan"] == ("source", "gaussian-decoder", path)
    assert calls["identity"] == ("planned", BFLOAT16)
    assert calls["load"] == (
        path,
        {
            "asset": ref,
            "expected_role": "gaussian-decoder",
            "expected_identity": "expected-identity",
            "compute_dtype": "bf16",
        },
    )
    assert calls["enroll"] == ("module", "cuda:0", "device:cpu")
    assert calls["handle"] == (
        "module",
        "mechanism",
        "cuda:0",
        "triposplat-preview:blake3:abc",
        coordinator,
    )
    assert calls["label"] == (calls["attached"] and entry.handle, ref.name)
    assert calls["attached"] is pool
    assert calls["decoder"] == "module"
    assert entry.resource_id == "resident:7"
    # A second resolve for the same key serves the cached entry.
    assert preview_emit_module._resolve_triposplat_decoder("triposplat_vae_decoder", "cuda:0") is (
        entry
    )


# -- animated previews ---------------------------------------------------------


def test_emitter_fans_a_clip_into_per_frame_events_with_ring_metadata() -> None:
    harness = EmitterHarness(stream_role="video")
    frames = tuple(PreviewFrame(rgb=f"f{i}".encode(), width=3, height=2) for i in (0, 1))
    clip = PreviewClip(frames=frames, frame_indices=(4, 5), frame_count=8, fps=4.0)
    harness.state(clip, at=0.0)
    harness.wait(2)
    assert [item[0] for item in harness.emitted] == [b"jpeg:b'f0'", b"jpeg:b'f1'"]
    assert harness.meta == [
        {"stream": "video", "frame_index": 4, "frame_count": 8, "fps": 4.0},
        {"stream": "video", "frame_index": 5, "frame_count": 8, "fps": 4.0},
    ]


def test_emitter_stills_carry_the_stream_role_without_ring_metadata() -> None:
    harness = EmitterHarness(stream_role="video")
    streams = MultiStreamLatent.from_pairs([("video", "v"), ("audio", "a")])
    harness.state(streams, at=0.0)
    harness.wait(1)
    assert harness.meta == [
        {"stream": "video", "frame_index": None, "frame_count": None, "fps": None}
    ]


VIDEO_RGB_DESCRIPTOR = LatentDescriptor(
    channels=2,
    dimensions=3,
    temporal_downscale=4,
    rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
)


def test_sampling_preview_emitter_prefers_animation_providers_for_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen: list[str] = []

    def make_builder(tag: str) -> Any:
        def build(descriptor: LatentDescriptor) -> Any:
            chosen.append(tag)
            return lambda state: PreviewFrame(rgb=b"x", width=1, height=1)

        return build

    _fake_torch_backend(
        monkeypatch,
        {
            LATENT2RGB_PROVIDER.id: make_builder("image"),
            LATENT2RGB_ANIMATION_PROVIDER.id: make_builder("animation"),
        },
    )
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(VIDEO_RGB_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert chosen == ["animation"]

    # Without an animation-capable backend the still-image pass still serves.
    chosen.clear()
    _fake_torch_backend(monkeypatch, {LATENT2RGB_PROVIDER.id: make_builder("image")})
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(VIDEO_RGB_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert chosen == ["image"]


def test_anima_resolves_the_shared_wan_preview_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(
        monkeypatch,
        {LATENT2RGB_ANIMATION_PROVIDER.id: lambda _descriptor: decoder},
    )
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(ANIMA.latent, family_id=ANIMA.id))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [ANIMA.latent]


def test_krea2_resolves_the_shared_wan_preview_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(
        monkeypatch,
        {LATENT2RGB_ANIMATION_PROVIDER.id: lambda _descriptor: decoder},
    )
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(KREA2.latent, family_id=KREA2.id))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [KREA2.latent]


def test_plain_descriptor_families_keep_the_stream_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Wan's family latent is a plain descriptor, but its runtime wraps
    # every sampling state in a one-role MultiStreamLatent: the caller's
    # role must survive resolution so the emitter can unwrap those states.
    decoded: list[object] = []

    def decode(state: object) -> PreviewFrame:
        decoded.append(state)
        return PreviewFrame(rgb=b"x", width=1, height=1)

    _fake_torch_backend(monkeypatch, {LATENT2RGB_ANIMATION_PROVIDER.id: lambda d: decode})
    with use_execution_context(_context("cheap")):
        emitter = sampling_preview_emitter(_handle(VIDEO_RGB_DESCRIPTOR), stream_role="video")
    assert isinstance(emitter, SamplingPreviewEmitter)
    streams = MultiStreamLatent.from_pairs([("video", "v-latent")])
    emitter.on_state(FakeStateEvent(denoised=streams))
    assert decoded == ["v-latent"]


TAEHV_DESCRIPTOR = LatentDescriptor(
    channels=2,
    dimensions=3,
    temporal_downscale=4,
    rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    taesd_decoder="tae_v",
)


def test_taehv_unavailability_logs_once_in_quality_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = _capture_report_log(monkeypatch)
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_ANIMATION_PROVIDER.id: lambda d: decoder})
    with use_execution_context(_context("quality")):
        # The handle's family "fam" has no TAEHV-covered video decoder.
        emitter = sampling_preview_emitter(_handle(TAEHV_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [TAEHV_DESCRIPTOR]  # fell through to latent2rgb.animation
    assert [(level, data["assetId"]) for level, _, data in logs] == [("info", "tae_v")]
    assert logs[0][2]["previewProvider"] == TAEHV_PROVIDER.id


def test_taehv_unavailability_degrades_silently_outside_quality_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = _capture_report_log(monkeypatch)
    decoder = lambda state: PreviewFrame(rgb=state, width=1, height=1)  # noqa: E731
    built = _fake_torch_backend(monkeypatch, {LATENT2RGB_ANIMATION_PROVIDER.id: lambda d: decoder})
    with use_execution_context(_context("auto")):
        emitter = sampling_preview_emitter(_handle(TAEHV_DESCRIPTOR))
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [TAEHV_DESCRIPTOR] and logs == []


@pytest.mark.parametrize("family_id", ("dinkster.wan21", "dinkster.anima", "dinkster.krea2"))
def test_quality_taehv_decoder_serves_the_emitter_and_stage(
    monkeypatch: pytest.MonkeyPatch, family_id: str
) -> None:
    import dinkster_compat_comfy.preview_emit as preview_emit_module
    from dinkster_inference import WAN21

    module = SimpleNamespace(
        preview_decoder_builders=lambda: {},
        taehv_preview_decoder=lambda decoder, config, descriptor: (
            lambda state: PreviewFrame(rgb=b"x", width=1, height=1)
        ),
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", cast("Any", module))
    staged: list[str] = []

    @contextmanager
    def stage(*, observer_stage: str) -> Any:
        staged.append(observer_stage)
        yield

    entry = SimpleNamespace(
        resource_id="resident:test",
        handle=SimpleNamespace(released=False, stage=stage),
        decoder=object(),
        config=object(),
    )
    built: list[tuple[str, object, object]] = []

    def build(asset_id: str, descriptor: LatentDescriptor, load_device: object) -> Any:
        built.append((asset_id, descriptor, load_device))
        return entry

    monkeypatch.setattr(preview_emit_module, "_taehv_cache", {})
    monkeypatch.setattr(preview_emit_module, "_build_taehv_entry", build)
    latent = {
        "dinkster.wan21": WAN21.latent,
        "dinkster.anima": ANIMA.latent,
        "dinkster.krea2": KREA2.latent,
    }[family_id]
    handle = _handle(latent, family_id=family_id)
    handle.load_device = "cuda:0"
    with use_execution_context(_context("quality")):
        emitter = sampling_preview_emitter(handle)
    assert isinstance(emitter, SamplingPreviewEmitter)
    assert built == [("lighttaew2_1", latent, "cuda:0")]
    with preview_stage(emitter):
        assert staged == ["sample"]
