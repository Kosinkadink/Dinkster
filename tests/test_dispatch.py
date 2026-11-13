"""Stage-6 execution dispatch, worker side: DispatchWorker follows the
engine's selection (never decides), enforces the residency-ownership
contract around it, the executor field never crosses the boundary wire,
owner provenance is producer-stamped and relay-preserved, and manifest
``[pack] executes`` claims parse as claims-not-grants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from dinkster_protocol import (
    Invocation,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    OnInvocationEvent,
)
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_values import (
    RESOURCE_HANDLE_TYPE,
    RESOURCE_PRODUCER_ARM_META_KEY,
    EncodedLatentTensor,
    ResourceHandle,
    TypeRegistry,
    Value,
    ValueMeta,
    register_core_types,
    register_resource_handle_type,
    stamp_resource_producer_arm,
    value_resource_refs,
)
from dinkster_workers import DispatchWorker
from dinkster_workers.boundary import (
    ValueCodec,
    decode_invocation,
    encode_invocation,
    release_segment,
)

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")

SCHEMA = NodeSchema(
    node_type="test.produce",
    inputs=(InputSpec("tag", STRING),),
    outputs=(OutputSpec("out", STRING),),
)
LAZY_SCHEMA = replace(
    SCHEMA,
    inputs=(InputSpec("tag", STRING, lazy=True),),
)

_REGISTRY = TypeRegistry()
register_core_types(_REGISTRY)
register_resource_handle_type(_REGISTRY)


def _string(text: str) -> Value:
    return _REGISTRY.wrap("STRING", text)


def _owned(rid: str, owner: str | None, *, as_list: bool = False) -> Value:
    """A resource-referencing Value with (optionally) stamped owner
    provenance; ``as_list`` nests it inside a list tree."""
    handle = ResourceHandle(resource_id=rid, kind="model", owner=owner)
    if as_list:
        return _REGISTRY.wrap(f"list<{RESOURCE_HANDLE_TYPE}>", [handle])
    return _REGISTRY.wrap(RESOURCE_HANDLE_TYPE, handle)


def _produced(rid: str, owner: str, producer: object) -> Value:
    value = _owned(rid, owner)
    entries = dict(value.meta.entries)
    entries[RESOURCE_PRODUCER_ARM_META_KEY] = producer
    return replace(value, meta=ValueMeta(entries))


def _invocation(inputs: Mapping[str, Value] | None = None, *, executor: str | None) -> Invocation:
    return Invocation(
        invocation_id="inv-1",
        node_id="p",
        node_type="test.produce",
        inputs=inputs if inputs is not None else {"tag": _string("x")},
        effective_schema=SCHEMA,
        output_members=(("out", ()),),
        executor=executor,
    )


class Arm:
    """Fake dispatch arm: records invocations, replays canned outputs,
    forwards one event when asked."""

    def __init__(self, outputs: Mapping[str, Value] | None = None) -> None:
        self.prepared: list[Sequence[str]] = []
        self.invocations: list[Invocation] = []
        self.lazy_invocations: list[LazyStatusInvocation] = []
        self.outputs = outputs if outputs is not None else {"out": _string("done")}

    async def prepare(self, node_types: Sequence[str]) -> None:
        self.prepared.append(tuple(node_types))

    async def invoke(
        self, invocation: Invocation, on_event: OnInvocationEvent | None = None
    ) -> InvocationResult:
        self.invocations.append(invocation)
        if on_event is not None:
            on_event(InvocationEvent("progress", {"value": 1.0}))
        return InvocationResult(outputs=self.outputs)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        del on_event
        self.lazy_invocations.append(invocation)
        return LazyStatusResult(requested_inputs=())


def _dispatcher(arms: Mapping[str, Arm], owners: Mapping[str, str] | None = None) -> DispatchWorker:
    domains = {name: object() for name in arms}
    table = {token: domains[arm] for token, arm in (owners or {}).items()}
    return DispatchWorker(
        arms,
        resolve_owner=table.get,
        arm_domains=domains,
        default_arms={arm: arm for arm in domains},
    )


def test_legacy_invocation_header_defaults_fp8_matmul_off() -> None:
    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation(executor=None))
    del header["fp8Matmul"]
    try:
        decoded = decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
        assert decoded.fp8_matmul is False
        assert decoded.export_snapshot is None
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def _manifest(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "p"\n{extra}\n[pack.entry]\nnodes = "m:attr"\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("use_shm", [False, True])
def test_sampling_settings_cross_domains_but_guider_remains_affine(
    use_shm: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_inference_wire import register_inference_types
    from dinkster_inference_wire.sampling_wire import (
        NoiseSelection,
        SamplerSelection,
        SigmaSchedule,
    )

    # Both codecs share the creator's resource tracker in this process.
    monkeypatch.setattr(
        "dinkster_workers.boundary._attach_segment", lambda name: SharedMemory(name=name)
    )

    async def scenario() -> None:
        registry = TypeRegistry()
        register_inference_types(registry)
        sender = ValueCodec(registry, shm_threshold=1, use_shm=use_shm, accept_shm=use_shm)
        receiver = ValueCodec(registry, use_shm=use_shm, accept_shm=use_shm)
        inputs = {"guider": _produced("model-guider", "native-session", "native")}
        for name, type_id, raw in (
            ("noise", "NOISE", NoiseSelection(7)),
            ("sampler", "SAMPLER", SamplerSelection("dinkster.euler", ())),
            ("sigmas", "SIGMAS", SigmaSchedule((1.0, 0.0))),
        ):
            value = stamp_resource_producer_arm(
                registry.wrap(type_id, raw), "default-session", "default"
            )
            blobs: list[bytes] = []
            segments: list[SharedMemory] = []
            try:
                wire, sent = sender.encode(value, blobs, segments)
                inputs[name], received = receiver.decode(wire, blobs, [])
                assert inputs[name].resolve() == raw
                assert sent.transport == received.transport == ("shm" if use_shm else "inline")
            finally:
                for segment in segments:
                    release_segment(segment)
            assert not tuple(value_resource_refs(inputs[name]))
        default, native = Arm(), Arm()
        worker = _dispatcher(
            {"default": default, "native": native},
            {"default-session": "default", "native-session": "native"},
        )
        invocation = _invocation(inputs, executor="native")
        assert (await worker.invoke(invocation)).error is None
        assert native.invocations == [invocation]
        assert default.invocations == []
        rejected = await worker.invoke(replace(invocation, executor="default"))
        assert rejected.error is not None
        assert "cannot cross residency domains" in rejected.error.message

    run_scenario(scenario())


@pytest.mark.parametrize("use_shm", [False, True])
def test_sigma_tensor_crosses_domains_as_an_encoded_tensor(
    use_shm: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_inference_wire import register_inference_types

    monkeypatch.setattr(
        "dinkster_workers.boundary._attach_segment", lambda name: SharedMemory(name=name)
    )
    raw = EncodedLatentTensor("float64", (2,), b"\x00" * 8 + b"\x00\x00\x00\x00\x00\x00\xf0?")
    registry = TypeRegistry()
    register_inference_types(registry)
    sender = ValueCodec(registry, shm_threshold=1, use_shm=use_shm, accept_shm=use_shm)
    receiver = ValueCodec(registry, use_shm=use_shm, accept_shm=use_shm)
    value = registry.wrap("SIGMAS", raw)
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    try:
        wire, sent = sender.encode(value, blobs, segments)
        received, received_stat = receiver.decode(wire, blobs, [])
        assert received.resolve() == raw
        assert received.fingerprint == value.fingerprint
        assert sent.transport == received_stat.transport == ("shm" if use_shm else "inline")
    finally:
        for segment in segments:
            release_segment(segment)
