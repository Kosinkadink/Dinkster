"""Stage-6 execution dispatch, worker side: DispatchWorker follows the
engine's selection (never decides), enforces the residency-ownership
contract around it, the executor field never crosses the boundary wire,
owner provenance is producer-stamped and relay-preserved, and manifest
``[pack] executes`` claims parse as claims-not-grants."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from dinkster_protocol import (
    ATTENTION_ROLES,
    AttentionRoute,
    AttentionRouteToken,
    ExportSnapshot,
    Invocation,
    InvocationEvent,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    MediaSourceAuthority,
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
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_PRODUCER_ARM_META_KEY,
    RESOURCE_REFS_META_KEY,
    ResourceHandle,
    TypeRegistry,
    Value,
    ValueMeta,
    list_children,
    process_instance_token,
    register_core_types,
    register_resource_handle_type,
    stamp_resource_producer_arm,
    value_resource_refs,
)
from dinkster_workers import ArmWorker, DispatchWorker
from dinkster_workers.boundary import (
    BoundaryError,
    ValueCodec,
    decode_invocation,
    encode_invocation,
    release_segment,
)
from dinkster_workers.host import AttentionRouteDiscoveryError, load_pack
from dinkster_workers.manifest import ManifestError, load_manifest

STRING = TypeExpr.concrete("core.string")

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
    return _REGISTRY.wrap("core.string", text)


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


def test_dispatch_worker_requires_arms() -> None:
    with pytest.raises(ValueError):
        DispatchWorker(
            {},
            resolve_owner=lambda token: None,
            arm_domains={},
            default_arms={},
        )


def test_prepare_fans_out_to_every_arm() -> None:
    async def scenario() -> None:
        compat, native = Arm(), Arm()
        worker = _dispatcher({"compat": compat, "native": native})
        await worker.prepare(["test.produce"])
        # Residency can pin an invocation to any arm, so all must be ready.
        assert compat.prepared == [("test.produce",)]
        assert native.prepared == [("test.produce",)]

    asyncio.run(scenario())


def test_lazy_hook_uses_owner_arm_and_alternate_execution_succeeds() -> None:
    async def scenario() -> None:
        owner, alternate = Arm(), Arm()
        worker = _dispatcher({"owner": owner, "alternate": alternate})
        lazy = LazyStatusInvocation(
            request_id="lazy-1",
            node_id="p",
            node_type="test.produce",
            available_inputs={"tag": _string("x")},
            connected_undemanded_inputs=(),
            effective_schema=LAZY_SCHEMA,
        )

        status = await worker.check_lazy_status(lazy)
        assert status.requested_inputs == ()
        assert owner.lazy_invocations == [lazy]
        assert alternate.lazy_invocations == []

        owner_result = await worker.invoke(
            replace(_invocation(executor="owner"), effective_schema=LAZY_SCHEMA)
        )
        assert owner_result.error is None
        alternate_result = await worker.invoke(
            replace(_invocation(executor="alternate"), effective_schema=LAZY_SCHEMA)
        )
        assert alternate_result.error is None
        assert len(alternate.invocations) == 1

    asyncio.run(scenario())


def test_lazy_hook_accepts_resident_input_from_same_session_alternate_arm() -> None:
    async def scenario() -> None:
        owner, alternate = Arm(), Arm()
        domain = object()
        worker = DispatchWorker(
            {"owner": owner, "alternate": alternate},
            resolve_owner={"session": domain}.get,
            arm_domains={"owner": domain, "alternate": domain},
            default_arms={"owner": "owner", "alternate": "owner"},
        )
        lazy = LazyStatusInvocation(
            request_id="lazy-resident",
            node_id="p",
            node_type="test.produce",
            available_inputs={"tag": _produced("clip", "session", "alternate")},
            connected_undemanded_inputs=(),
            effective_schema=LAZY_SCHEMA,
        )

        status = await worker.check_lazy_status(lazy)

        assert status.error is None
        assert owner.lazy_invocations == [lazy]

    asyncio.run(scenario())


def test_lazy_hook_rejects_resident_input_from_another_residency_domain() -> None:
    async def scenario() -> None:
        owner, alternate = Arm(), Arm()
        worker = _dispatcher(
            {"owner": owner, "alternate": alternate},
            owners={"alternate-session": "alternate"},
        )
        lazy = LazyStatusInvocation(
            request_id="lazy-foreign-resident",
            node_id="p",
            node_type="test.produce",
            available_inputs={"tag": _produced("clip", "alternate-session", "alternate")},
            connected_undemanded_inputs=(),
            effective_schema=LAZY_SCHEMA,
        )

        status = await worker.check_lazy_status(lazy)

        assert status.error is not None
        assert "cannot cross residency domains" in status.error.message
        assert owner.lazy_invocations == []

    asyncio.run(scenario())


def test_selected_arm_alone_is_invoked_and_events_forward() -> None:
    async def scenario() -> None:
        compat, native = Arm(), Arm()
        worker = _dispatcher({"compat": compat, "native": native})
        events: list[InvocationEvent] = []
        result = await worker.invoke(
            _invocation(executor="native"),
            on_event=events.append,
        )
        assert result.error is None
        assert compat.invocations == []
        assert [inv.invocation_id for inv in native.invocations] == ["inv-1"]
        assert [(e.name, dict(e.data)) for e in events] == [("progress", {"value": 1.0})]

    asyncio.run(scenario())


def test_missing_or_unknown_selection_is_a_loud_wiring_error() -> None:
    async def scenario() -> None:
        native = Arm()
        worker = _dispatcher({"native": native})

        result = await worker.invoke(_invocation(executor=None))
        assert result.error is not None
        assert "without an execution selection" in result.error.message

        result = await worker.invoke(_invocation(executor="compat"))
        assert result.error is not None
        assert "unknown arm 'compat'" in result.error.message
        assert native.invocations == []  # never fell through to a guess

    asyncio.run(scenario())


@pytest.mark.parametrize("as_list", [False, True])
def test_resident_inputs_must_resolve_to_the_selected_arm(as_list: bool) -> None:
    async def scenario() -> None:
        compat, native = Arm(), Arm()
        worker = _dispatcher(
            {"compat": compat, "native": native},
            owners={"life-compat": "compat", "life-native": "native"},
        )

        def inputs(owner: str | None) -> Mapping[str, Value]:
            return {"tag": _owned("res:model", owner, as_list=as_list)}

        # Unstamped resident reference: dispatched types require provenance.
        result = await worker.invoke(_invocation(inputs(None), executor="native"))
        assert result.error is not None
        assert "no owner token" in result.error.message

        # Dead owner: no live arm holds the token.
        result = await worker.invoke(_invocation(inputs("life-old"), executor="native"))
        assert result.error is not None
        assert "no live arm holds" in result.error.message

        # Live, but pinned to the OTHER arm: resident state cannot cross.
        result = await worker.invoke(_invocation(inputs("life-compat"), executor="native"))
        assert result.error is not None
        assert "cannot cross residency domains" in result.error.message
        assert native.invocations == []

        # Owner matches the selection: dispatched.
        result = await worker.invoke(_invocation(inputs("life-native"), executor="native"))
        assert result.error is None
        assert len(native.invocations) == 1

    asyncio.run(scenario())


def test_outputs_owned_by_another_arm_are_refused() -> None:
    async def scenario() -> None:
        # The arm that ran claims residency owned by the OTHER arm: the
        # result would poison the cache entry the selection's tag names.
        native = Arm(outputs={"out": _owned("res:model", "life-compat")})
        compat = Arm()
        worker = _dispatcher(
            {"compat": compat, "native": native},
            owners={"life-compat": "compat", "life-native": "native"},
        )
        result = await worker.invoke(_invocation(executor="native"))
        assert result.error is not None
        assert "cannot cross residency domains" in result.error.message

        # Correctly self-owned residency passes through untouched.
        good = Arm(outputs={"out": _owned("res:model", "life-native")})
        worker = _dispatcher({"native": good}, owners={"life-native": "native"})
        result = await worker.invoke(_invocation(executor="native"))
        assert result.error is None
        assert result.outputs is not None

    asyncio.run(scenario())


def test_executor_stays_host_only_while_body_fields_cross_the_wire() -> None:
    codec = ValueCodec(_REGISTRY)
    attention_token = AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
        (("torch", "2.13.0"),),
        "dinkster.attention-kernel.v1",
        "cpu",
        None,
        "2.13.0",
        "auto",
    )
    invocation = replace(
        _invocation(executor="native"),
        arm="fast",
        expected_execution_identity="body-v2",
        fp8_matmul=True,
        diffusion_dtype="bfloat16",
        text_dtype="float16",
        vae_dtype="float32",
        attention_route_token=attention_token,
        extension_snapshot_digest="sha256:" + "a" * 64,
        export_snapshot=ExportSnapshot(
            prompt={"save": {"class_type": "Save", "inputs": {}}},
            extra_pnginfo={"workflow": {"nodes": [2, 1]}},
        ),
        media_sources=(
            MediaSourceAuthority(
                "blake3:" + "b" * 64,
                "media/image",
                "image/png",
                "png",
                17,
            ),
        ),
    )
    header, blobs, segments, _stats = encode_invocation(codec, invocation)
    try:
        assert header["jobRef"] == invocation.invocation_id
        assert header["attemptId"] == 1
        # The frame carries no trace of the host's selection...
        assert "executor" not in header
        assert header["arm"] == "fast"
        assert header["expectedExecutionIdentity"] == "body-v2"
        assert header["fp8Matmul"] is True
        assert header["componentDtypes"] == {
            "diffusion": "bfloat16",
            "textEncoder": "float16",
            "vae": "float32",
        }
        assert header["attentionPolicy"] == "auto"
        assert "attentionRouteToken" in header
        assert "attentionCapabilities" not in header
        assert header["extensionSnapshotDigest"] == "sha256:" + "a" * 64
        assert header["exportSnapshot"] == {
            "prompt": {"save": {"class_type": "Save", "inputs": {}}},
            "extraPnginfo": {"workflow": {"nodes": [2, 1]}},
        }
        decoded = decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
        # ...so the child executes whatever arrives, selection stays host-side.
        assert decoded.executor is None
        assert decoded.arm == "fast"
        assert decoded.expected_execution_identity == "body-v2"
        assert decoded.fp8_matmul is True
        assert decoded.diffusion_dtype == "bfloat16"
        assert decoded.text_dtype == "float16"
        assert decoded.vae_dtype == "float32"
        assert decoded.attention_policy == "auto"
        assert decoded.attention_route_token == attention_token
        assert not hasattr(decoded, "attention_capabilities")
        assert decoded.extension_snapshot_digest == "sha256:" + "a" * 64
        assert decoded.export_snapshot == invocation.export_snapshot
        assert decoded.media_sources == invocation.media_sources
        assert decoded.job_ref == invocation.invocation_id
        assert decoded.attempt_id == 1
        assert decoded.node_type == invocation.node_type
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("jobRef", "", "jobRef"),
        ("jobRef", 7, "jobRef"),
        ("attemptId", 0, "attemptId"),
        ("attemptId", True, "attemptId"),
        ("attemptId", "2", "attemptId"),
    ),
)
def test_invocation_identity_wire_fields_are_strict(
    field: str, value: object, message: str
) -> None:
    header, blobs, segments, _stats = encode_invocation(
        ValueCodec(_REGISTRY), replace(_invocation(executor=None), job_ref="job-7", attempt_id=3)
    )
    header[field] = value
    try:
        with pytest.raises(BoundaryError, match=message):
            decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("arm", 3),
        ("arm", ""),
        ("expectedExecutionIdentity", 3),
        ("expectedExecutionIdentity", ""),
        ("fp8Matmul", "true"),
        ("componentDtypes", {"diffusion": "float16"}),
        ("componentDtypes", {"diffusion": "float16", "textEncoder": 3, "vae": "float32"}),
        ("attentionPolicy", "sdpa"),
        ("extensionSnapshotDigest", "sha256:short"),
        ("extensionSnapshotDigest", "sha256:" + "A" * 64),
        ("extensionSnapshotDigest", "sha256:" + "z" * 64),
        ("exportSnapshot", None),
        ("exportSnapshot", {}),
        ("exportSnapshot", {"prompt": [], "extraPnginfo": None}),
        ("exportSnapshot", {"prompt": {}, "extraPnginfo": []}),
        ("mediaSources", {}),
        ("mediaSources", [{"digest": "blake3:bad"}]),
    ),
)
def test_malformed_body_wire_fields_are_loud(field: str, value: object) -> None:
    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation(executor=None))
    header[field] = value
    try:
        with pytest.raises(BoundaryError):
            decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


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


def test_default_invocation_wire_omits_export_snapshot() -> None:
    codec = ValueCodec(_REGISTRY)
    header, _blobs, segments, _stats = encode_invocation(codec, _invocation(executor=None))
    try:
        assert set(header) == {
            "type",
            "invocationId",
            "jobRef",
            "attemptId",
            "nodeId",
            "nodeType",
            "effectiveSchema",
            "outputMembers",
            "inputs",
            "fp8Matmul",
            "attentionPolicy",
        }
        assert "exportSnapshot" not in header
        assert "mediaSources" not in header
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def test_fp8_matmul_cannot_cross_without_expected_identity() -> None:
    with pytest.raises(ValueError, match="requires expected_execution_identity"):
        replace(_invocation(executor=None), fp8_matmul=True)

    codec = ValueCodec(_REGISTRY)
    header, blobs, segments, _stats = encode_invocation(codec, _invocation(executor=None))
    header["fp8Matmul"] = True
    try:
        with pytest.raises(BoundaryError, match="requires expectedExecutionIdentity"):
            decode_invocation(ValueCodec(_REGISTRY), header, blobs, [])
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


def test_media_source_authorities_are_exact_and_unique() -> None:
    authority = MediaSourceAuthority(
        "blake3:" + "a" * 64,
        "media/video",
        "video/mp4",
        "mp4",
        42,
    )
    with pytest.raises(ValueError, match="unique digests"):
        replace(_invocation(executor=None), media_sources=(authority, authority))
    with pytest.raises(ValueError, match="canonical"):
        replace(authority, media_type="video/webm")
    with pytest.raises(ValueError, match="blake3"):
        replace(authority, digest="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="non-negative"):
        replace(authority, byte_size=-1)


def test_process_token_is_stable_and_stamps_local_handles() -> None:
    token = process_instance_token()
    assert token and token == process_instance_token()

    # A LOCAL handle is owned here: its envelope carries this lifetime's
    # token, whatever (stale) owner field the handle itself carries.
    local = ResourceHandle(resource_id="res:model", kind="model", obj=object(), owner="life-stale")
    value = _REGISTRY.wrap(RESOURCE_HANDLE_TYPE, local)
    assert value.meta.get(RESOURCE_OWNER_META_KEY) == token
    assert value_resource_refs(value) == (("res:model", token),)

    # An unstamped reference reads as (rid, None) - judged by pin/release
    # layers, not owner admission.
    unstamped = _REGISTRY.wrap(
        RESOURCE_HANDLE_TYPE, ResourceHandle(resource_id="res:x", kind="model")
    )
    assert value_resource_refs(unstamped) == (("res:x", None),)


def test_owner_provenance_survives_a_wire_relay() -> None:
    # Producer side: a local handle crosses the boundary; the wire form
    # carries the producer's lifetime token.
    producer_codec = ValueCodec(_REGISTRY)
    local = ResourceHandle(resource_id="res:model", kind="model", obj=object())
    blobs: list[bytes] = []
    wire, _stat = producer_codec.encode(_REGISTRY.wrap(RESOURCE_HANDLE_TYPE, local), blobs, [])

    # Relay side: decoding yields a NON-local handle still stamped with the
    # producer's token - and re-encoding (the relay hop) preserves it
    # rather than rewriting provenance this process did not create.
    relay_codec = ValueCodec(_REGISTRY)
    relayed, _ = relay_codec.decode(wire, blobs, [])
    handle = relayed.resolve()
    assert isinstance(handle, ResourceHandle)
    assert not handle.is_local
    assert handle.owner == process_instance_token()

    rewrapped = _REGISTRY.wrap(RESOURCE_HANDLE_TYPE, handle)
    assert rewrapped.meta.get(RESOURCE_OWNER_META_KEY) == process_instance_token()
    assert rewrapped.meta.get(RESOURCE_ID_META_KEY) == "res:model"

    # In-process the producer's and the relay's tokens coincide, which
    # cannot distinguish "preserved" from "rewritten": relay a handle
    # stamped by a FOREIGN lifetime and prove the token survives untouched.
    foreign = ResourceHandle(resource_id="res:far", kind="model", owner="life-far")
    blobs = []
    wire, _stat = ValueCodec(_REGISTRY).encode(
        _REGISTRY.wrap(RESOURCE_HANDLE_TYPE, foreign), blobs, []
    )
    relayed, _ = ValueCodec(_REGISTRY).decode(wire, blobs, [])
    handle = relayed.resolve()
    assert isinstance(handle, ResourceHandle)
    assert handle.owner == "life-far" != process_instance_token()
    rewrapped = _REGISTRY.wrap(RESOURCE_HANDLE_TYPE, handle)
    assert rewrapped.meta.get(RESOURCE_OWNER_META_KEY) == "life-far"


def _manifest(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        f'[pack]\nname = "p"\n{extra}\n[pack.entry]\nnodes = "m:attr"\n',
        encoding="utf-8",
    )
    return path


def test_manifest_executes_parses_verbatim_claims(tmp_path: Path) -> None:
    # Absent means none - the common pack claims no alternative executions.
    assert load_manifest(_manifest(tmp_path, "")).executes == ()
    # Node types pass through VERBATIM, order preserved: legacy uppercase
    # tails are not grammar-valid names but are real node types.
    assert load_manifest(
        _manifest(tmp_path, 'executes = ["comfy.KSampler", "dinkster.load_checkpoint"]\n')
    ).executes == ("comfy.KSampler", "dinkster.load_checkpoint")


def test_manifest_executes_rejects_malformed_lists(tmp_path: Path) -> None:
    for bad in (
        'executes = "comfy.KSampler"\n',  # not a list
        "executes = []\n",  # empty list: omit instead
        "executes = [3]\n",  # not a string
        'executes = [""]\n',  # empty entry
        'executes = ["a", "a"]\n',  # duplicate claim
    ):
        with pytest.raises(ManifestError):
            load_manifest(_manifest(tmp_path, bad))


def test_manifest_body_arms_parse_and_validate(tmp_path: Path) -> None:
    valid = _manifest(
        tmp_path,
        '[pack.arms]\nalt = ["p.echo"]\n\n',
    )
    valid.write_text(
        valid.read_text().replace(
            'nodes = "m:attr"',
            'nodes = "m:attr"\narm_nodes = "m:arms"',
        )
    )
    manifest = load_manifest(valid)
    assert manifest.arms == (("alt", ("p.echo",)),)
    assert manifest.arm_nodes_entry == "m:arms"

    bad_cases = (
        ('arms = "bad"\n', "malformed table"),
        ("[pack.arms]\nalt = []\n", "empty node list"),
        ('[pack.arms]\nalt = ["other.echo"]\n', "unowned node type"),
        ('[pack.arms]\n"bad.arm" = ["p.echo"]\n', "bad arm name"),
        ('[pack.arms]\n__proto__ = ["p.echo"]\n', "reserved arm name"),
    )
    for index, (body, _label) in enumerate(bad_cases):
        root = tmp_path / str(index)
        root.mkdir()
        with pytest.raises(ManifestError):
            load_manifest(_manifest(root, body))

    arms_only = tmp_path / "arms-only"
    arms_only.mkdir()
    with pytest.raises(ManifestError, match="both be present"):
        load_manifest(_manifest(arms_only, '[pack.arms]\nalt = ["p.echo"]\n'))
    entry_root = tmp_path / "entry-only"
    entry_root.mkdir()
    entry_only = _manifest(entry_root, "")
    entry_only.write_text(
        entry_only.read_text().replace('nodes = "m:attr"', 'nodes = "m:attr"\narm_nodes = "m:arms"')
    )
    with pytest.raises(ManifestError, match="both be present"):
        load_manifest(entry_only)


def test_dispatch_enforces_domain_and_producer_arm_affinity() -> None:
    async def scenario() -> None:
        shared = object()
        other = object()
        owner, alt = Arm(), Arm()
        worker = DispatchWorker(
            {"pack": owner, "pack@alt": alt},
            resolve_owner={"same": shared, "other": other}.get,
            arm_domains={"pack": shared, "pack@alt": shared},
            default_arms={"pack": "pack", "pack@alt": "pack"},
        )

        async def result(value: Value, target: str) -> InvocationResult:
            return await worker.invoke(_invocation({"tag": value}, executor=target))

        assert (await result(_produced("r1", "same", "pack@alt"), "pack@alt")).error is None
        assert (await result(_owned("r2", "same"), "pack")).error is None
        for value, target, message in (
            (_produced("r3", "same", "pack"), "pack@alt", "affinity"),
            (_produced("r4", "same", "pack@alt"), "pack", "affinity"),
            (_owned("r5", "same"), "pack@alt", "affinity"),
            (_produced("r6", "other", "pack@alt"), "pack@alt", "domains"),
            (_produced("r7", "same", 3), "pack@alt", "malformed"),
            (_produced("r8", "same", "unknown"), "pack@alt", "unknown"),
        ):
            outcome = await result(value, target)
            assert outcome.error is not None and message in outcome.error.message

    asyncio.run(scenario())


def test_native_model_enters_native_load_lora_body() -> None:
    async def scenario() -> None:
        shared = object()
        default, native = Arm(), Arm()
        worker = DispatchWorker(
            {"dinkster-compat-comfy": default, "dinkster-compat-comfy@native": native},
            resolve_owner=lambda token: shared if token == "session" else None,
            arm_domains={
                "dinkster-compat-comfy": shared,
                "dinkster-compat-comfy@native": shared,
            },
            default_arms={
                "dinkster-compat-comfy": "dinkster-compat-comfy",
                "dinkster-compat-comfy@native": "dinkster-compat-comfy",
            },
        )
        invocation = replace(
            _invocation(
                {"model": _produced("native-model", "session", "dinkster-compat-comfy@native")},
                executor="dinkster-compat-comfy@native",
            ),
            node_type="dinkster.load_lora",
        )

        outcome = await worker.invoke(invocation)

        assert outcome.error is None
        assert default.invocations == []
        assert native.invocations == [invocation]

    asyncio.run(scenario())


@pytest.mark.parametrize("use_shm", [False, True])
def test_sampling_settings_cross_domains_but_guider_remains_affine(
    use_shm: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_inference import register_inference_types
    from dinkster_inference.sampling_wire import NoiseSelection, SamplerSelection, SigmaSchedule

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
            ("noise", "dinkster.noise", NoiseSelection(7)),
            ("sampler", "dinkster.sampler", SamplerSelection("dinkster.euler", ())),
            ("sigmas", "dinkster.sigmas", SigmaSchedule((1.0, 0.0))),
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

    asyncio.run(scenario())


def test_residency_domains_use_identity_not_value_equality() -> None:
    class EqualDomain:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, EqualDomain)

        __hash__ = object.__hash__

    async def scenario() -> None:
        first, second = EqualDomain(), EqualDomain()
        worker = DispatchWorker(
            {"first": Arm(), "second": Arm()},
            resolve_owner=lambda token: second,
            arm_domains={"first": first, "second": second},
            default_arms={"first": "first", "second": "second"},
        )
        outcome = await worker.invoke(
            _invocation({"tag": _owned("equal", "token")}, executor="first")
        )
        assert outcome.error is not None
        assert "cannot cross residency domains" in outcome.error.message

    asyncio.run(scenario())


def test_producer_stamping_walks_lists_and_preserves_foreign_provenance() -> None:
    local = _REGISTRY.wrap(
        f"list<{RESOURCE_HANDLE_TYPE}>",
        [ResourceHandle(resource_id="nested", kind="model", owner="session")],
    )
    stamped = stamp_resource_producer_arm(local, "session", "pack@alt")
    children = list_children(stamped)
    assert children is not None
    child = children[0]
    assert child.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "pack@alt"

    foreign = _produced("foreign", "other", "other@arm")
    unchanged = stamp_resource_producer_arm(foreign, "session", "pack@alt")
    assert unchanged.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "other@arm"

    composite = _owned("base", "session")
    entries = dict(composite.meta.entries)
    entries[RESOURCE_REFS_META_KEY] = ("control",)
    stamped_composite = stamp_resource_producer_arm(
        replace(composite, meta=ValueMeta(entries)),
        "session",
        "pack@alt",
    )
    assert stamped_composite.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "pack@alt"
    assert value_resource_refs(stamped_composite) == (
        ("base", "session"),
        ("control", "session"),
    )


def test_arm_worker_stamps_exact_replica_arm_on_nested_resident_outputs() -> None:
    local = _REGISTRY.wrap(
        f"list<{RESOURCE_HANDLE_TYPE}>",
        [ResourceHandle(resource_id="nested", kind="model", owner="session")],
    )
    foreign = _produced("foreign", "other", "other@arm")

    class ResidentArm(Arm):
        instance_token = "session"

    async def scenario() -> InvocationResult:
        return await ArmWorker(
            ResidentArm({"local": local, "foreign": foreign}),
            "native",
            "pack@native:cuda:4",
        ).invoke(_invocation(executor="pack@native:cuda:4"))

    result = asyncio.run(scenario())
    assert result.outputs is not None
    children = list_children(result.outputs["local"])
    assert children is not None
    assert children[0].meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "pack@native:cuda:4"
    assert result.outputs["foreign"].meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == "other@arm"


@pytest.mark.parametrize("variant", ["missing", "extra", "mismatch"])
def test_worker_startup_rejects_invalid_arm_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    module = f"arm_registration_{variant}"
    extra_input = ', InputSpec("extra", STRING, default="")' if variant == "mismatch" else ""
    registrations = {
        "missing": 'ARM_NODES = {"alt": []}',
        "extra": 'ARM_NODES = {"alt": [Alt, Extra]}',
        "mismatch": 'ARM_NODES = {"alt": [Alt]}',
    }[variant]
    (tmp_path / f"{module}.py").write_text(
        f"""from collections.abc import Mapping
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
STRING = TypeExpr.concrete("core.string")
class Base(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="p.echo",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )
    @classmethod
    def execute(cls, *, value):
        return cls.outputs(value=value)
class Alt(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="p.echo",
            inputs=(InputSpec("value", STRING){extra_input},),
            outputs=(OutputSpec("value", STRING),),
        )
    @classmethod
    def execute(cls, *, value, **kwargs):
        return cls.outputs(value=value)
class Extra(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type="p.extra", inputs=(), outputs=())
    @classmethod
    def execute(cls):
        return {{}}
NODES = [Base]
{registrations}
"""
    )
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        f'''[pack]
name = "p"
[pack.arms]
alt = ["p.echo"]
[pack.entry]
nodes = "{module}:NODES"
arm_nodes = "{module}:ARM_NODES"
'''
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ManifestError):
        load_pack(load_manifest(manifest_path))


def test_native_provider_pack_keeps_attention_discovery_failure_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        """[pack]
name = "provider"
[pack.arms]
native = ["provider.node"]
[pack.entry]
nodes = "provider_nodes:NODES"
arm_nodes = "provider_nodes:ARM_NODES"
""",
        encoding="utf-8",
    )

    def broken_provider() -> None:
        raise AttentionRouteDiscoveryError("attention runtime nested import failed")

    monkeypatch.setattr("dinkster_workers.host._discover_attention_routing", broken_provider)
    with pytest.raises(AttentionRouteDiscoveryError, match="nested import"):
        load_pack(load_manifest(manifest_path))


def test_native_provider_pack_refuses_absent_evidence_before_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        """[pack]
name = "provider"
[pack.arms]
native = ["provider.node"]
[pack.entry]
nodes = "entry_must_not_import:NODES"
arm_nodes = "entry_must_not_import:ARM_NODES"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("dinkster_workers.host._discover_attention_routing", lambda: (None, None))

    with pytest.raises(AttentionRouteDiscoveryError, match="requires authenticated"):
        load_pack(load_manifest(manifest_path))
    assert "entry_must_not_import" not in sys.modules
