from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
from dinkster_protocol import (
    RESULT_ALGEBRA_CAPABILITY,
    BlockedOutput,
    CurrentNodeRef,
    CurrentOutputRef,
    DirectReturn,
    ErrorHint,
    ExpandedReturn,
    ExportSnapshot,
    Invocation,
    InvocationFailure,
    InvocationResult,
    InvocationReturn,
    JsonObject,
    LiteralInput,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    NodeError,
    PresentOutput,
    ReturnBatch,
    negotiate_result_capabilities,
)
from dinkster_schema import NodeSchema
from dinkster_values import ListPayload, TypeRegistry, Value, make_list_value, register_core_types
from dinkster_workers import boundary
from dinkster_workers.boundary import (
    BoundaryError,
    ValueCodec,
    decode_invocation,
    decode_invocation_outcome,
    encode_invocation_outcome,
    encode_result,
    read_frame,
    write_frame,
)
from dinkster_workers.session import BoundarySession, WorkerDied

CAPABILITIES = frozenset({RESULT_ALGEBRA_CAPABILITY})


def registry() -> TypeRegistry:
    result = TypeRegistry()
    register_core_types(result)
    return result


def direct_outcome() -> InvocationReturn:
    value = registry().wrap("core.string", "payload")
    return InvocationReturn(
        ReturnBatch(
            "scalar",
            (
                DirectReturn(
                    (
                        ("a", PresentOutput(value)),
                        ("b", BlockedOutput(None)),
                        ("c", CurrentOutputRef("a")),
                    )
                ),
            ),
        )
    )


def expanded_outcome() -> InvocationReturn:
    value = registry().wrap("core.int", 7)
    expansion = LocalExpansion(
        (
            LocalNode(
                "child",
                "Comfy.Node",
                (
                    (
                        "literal",
                        LiteralInput(JsonObject((("a", 1), ("b", (True, None))))),
                    ),
                    ("source", CurrentOutputRef("seed")),
                ),
                parent=CurrentNodeRef(),
                display=LocalNodeRef("display"),
            ),
            LocalNode("display", "Comfy.Display"),
        )
    )
    return InvocationReturn(
        ReturnBatch(
            "mapped",
            (
                ExpandedReturn(
                    expansion,
                    (
                        ("result", LocalOutputRef("child", "out")),
                        ("seed", PresentOutput(value)),
                    ),
                ),
            ),
        )
    )


def encode(outcome: InvocationReturn | InvocationFailure) -> tuple[dict[str, object], list[bytes]]:
    header, blobs, segments = encode_invocation_outcome(
        ValueCodec(registry(), use_shm=False),
        outcome,
        "inv-1",
        1.5,
        negotiated_capabilities=CAPABILITIES,
    )
    assert segments == []
    return header, blobs


def decode(header: dict[str, object], blobs: list[bytes]) -> object:
    return decode_invocation_outcome(
        ValueCodec(registry(), use_shm=False),
        header,
        blobs,
        [],
        negotiated_capabilities=CAPABILITIES,
    )[0]


def algebra(header: dict[str, object]) -> dict[str, Any]:
    return cast("dict[str, Any]", header["resultAlgebra"])


def replace_document(header: dict[str, object], document: object) -> dict[str, object]:
    changed = copy.deepcopy(header)
    algebra(changed)["document"] = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return changed


def test_construction_is_frozen_and_preserves_silent_blocker_state() -> None:
    silent = BlockedOutput(None)
    empty = BlockedOutput("")
    assert silent != empty
    with pytest.raises(FrozenInstanceError):
        silent.message = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="RPC-clean"):
        LiteralInput([1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="RPC-clean"):
        LiteralInput(lambda: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        LiteralInput(float("nan"))
    with pytest.raises(ValueError, match="negative zero"):
        LiteralInput(-0.0)
    with pytest.raises(ValueError, match="64-bit"):
        LiteralInput(2**63)
    with pytest.raises(ValueError, match="Unicode"):
        LiteralInput("\ud800")
    with pytest.raises(ValueError, match="uniquely sorted"):
        JsonObject((("b", 1), ("a", 2)))
    with pytest.raises(ValueError, match="uniquely sorted"):
        JsonObject((("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="exact ErrorHint"):
        InvocationFailure(NodeError("n", "T", "bad", hints=(object(),)))  # type: ignore[arg-type]
    assert InvocationFailure(NodeError("n", "T", "bad", hints=(ErrorHint("c", "m"),)))


@pytest.mark.parametrize(
    "literal",
    [
        (),
        JsonObject(()),
        (("a", "b"),),
        JsonObject((("array", (JsonObject(()), (), (1, 2))),)),
    ],
)
def test_json_array_and_object_literals_round_trip_without_ambiguity(literal: object) -> None:
    outcome = InvocationReturn(
        ReturnBatch(
            "scalar",
            (
                ExpandedReturn(
                    LocalExpansion(
                        (
                            LocalNode(
                                "child", "Node", (("value", LiteralInput(cast(Any, literal))),)
                            ),
                        )
                    ),
                    (),
                ),
            ),
        )
    )
    header, blobs = encode(outcome)
    decoded = cast(InvocationReturn, decode(header, blobs))
    unit = cast(ExpandedReturn, decoded.batch.units[0])
    assert cast(LiteralInput, unit.expansion.nodes[0].inputs[0][1]).value == literal


def test_batch_cardinality_order_and_sorted_fields() -> None:
    unit = DirectReturn(())
    assert ReturnBatch("mapped", ()).units == ()
    assert ReturnBatch("mapped", (unit, unit)).units == (unit, unit)
    with pytest.raises(ValueError, match="exactly one"):
        ReturnBatch("scalar", ())
    with pytest.raises(ValueError, match="exactly one"):
        ReturnBatch("scalar", (unit, unit))
    with pytest.raises(ValueError, match="uniquely sorted"):
        DirectReturn((("b", BlockedOutput()), ("a", BlockedOutput())))
    with pytest.raises(ValueError, match="uniquely sorted"):
        DirectReturn((("a", BlockedOutput()), ("a", BlockedOutput())))


def test_local_references_are_closed_and_acyclic() -> None:
    with pytest.raises(ValueError, match="escapes"):
        LocalExpansion((LocalNode("a", "Node", (("x", LocalOutputRef("b", "out")),)),))
    with pytest.raises(ValueError, match="cycle"):
        LocalExpansion(
            (
                LocalNode("a", "Node", (("x", LocalOutputRef("b", "out")),)),
                LocalNode("b", "Node", (("x", LocalOutputRef("a", "out")),)),
            )
        )
    with pytest.raises(ValueError, match="same-unit"):
        ExpandedReturn(
            LocalExpansion((LocalNode("a", "Node", (("x", CurrentOutputRef("missing")),)),)),
            (("out", BlockedOutput()),),
        )
    with pytest.raises(ValueError, match="same-unit"):
        DirectReturn((("out", CurrentOutputRef("missing")),))


def test_capability_negotiation_is_exact_and_strict() -> None:
    assert negotiate_result_capabilities(None) == frozenset()
    assert negotiate_result_capabilities(["unknown"]) == frozenset()
    assert negotiate_result_capabilities([RESULT_ALGEBRA_CAPABILITY]) == CAPABILITIES
    with pytest.raises(ValueError, match="unique strings"):
        negotiate_result_capabilities([RESULT_ALGEBRA_CAPABILITY] * 2)
    with pytest.raises(ValueError, match="array or set"):
        negotiate_result_capabilities(RESULT_ALGEBRA_CAPABILITY)


def test_direct_canonical_wire_and_value_blob_round_trip() -> None:
    header, blobs = encode(direct_outcome())
    assert set(header) == {"type", "invocationId", "executeMs", "resultAlgebra"}
    assert len(blobs) == 2
    document = algebra(header)["document"]
    assert document == (
        '{"batch":{"mode":"scalar","units":[{"bindings":['
        '{"id":"a","value":{"kind":"present","valueIndex":0}},'
        '{"id":"b","value":{"kind":"blocked","message":null}},'
        '{"id":"c","value":{"kind":"outputRef","outputId":"a","scope":"current"}}'
        '],"kind":"direct"}]},"kind":"return"}'
    )
    decoded = cast(InvocationReturn, decode(header, blobs))
    unit = cast(DirectReturn, decoded.batch.units[0])
    assert cast(PresentOutput, unit.bindings[0][1]).value.resolve() == "payload"
    assert unit.bindings[1][1] == BlockedOutput(None)


def test_expanded_and_error_variants_round_trip() -> None:
    header, blobs = encode(expanded_outcome())
    decoded = cast(InvocationReturn, decode(header, blobs))
    unit = cast(ExpandedReturn, decoded.batch.units[0])
    assert unit.expansion.nodes[0].parent == CurrentNodeRef()
    assert unit.expansion.nodes[0].display == LocalNodeRef("display")
    assert cast(PresentOutput, unit.bindings[1][1]).value.resolve() == 7

    failure = InvocationFailure(NodeError("node", "Type", "failed"))
    error_header, error_blobs = encode(failure)
    assert error_blobs == []
    assert decode(error_header, error_blobs) == failure


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda h: algebra(h).update(version=2), "capability or version"),
        (lambda h: algebra(h).update(capability="wrong"), "capability or version"),
        (lambda h: algebra(h).update(extra=True), "fields"),
        (lambda h: h.update(extra=True), "typed result frame"),
    ],
)
def test_envelope_refuses_wrong_fields_capability_and_version(mutation: object, match: str) -> None:
    header, blobs = encode(direct_outcome())
    cast(Any, mutation)(header)
    with pytest.raises(BoundaryError, match=match):
        decode(header, blobs)


def test_codec_requires_negotiated_capability() -> None:
    with pytest.raises(BoundaryError, match="not negotiated"):
        encode_invocation_outcome(
            ValueCodec(registry()),
            direct_outcome(),
            "inv",
            0.0,
            negotiated_capabilities=frozenset(),
        )
    header, blobs = encode(direct_outcome())
    with pytest.raises(BoundaryError, match="not negotiated"):
        decode_invocation_outcome(
            ValueCodec(registry()),
            header,
            blobs,
            [],
            negotiated_capabilities=frozenset(),
        )


@pytest.mark.parametrize(
    "execute_ms",
    [float("nan"), float("inf"), -0.0, pytest.param(10**10000, id="huge-int")],
)
def test_encode_refuses_noncanonical_execute_time(execute_ms: object) -> None:
    with pytest.raises(BoundaryError):
        encode_invocation_outcome(
            ValueCodec(registry()),
            direct_outcome(),
            "inv",
            cast(Any, execute_ms),
            negotiated_capabilities=CAPABILITIES,
        )


def test_value_tree_preflight_refuses_noncanonical_or_excessive_lists() -> None:
    reg = registry()
    scalar = reg.wrap("core.int", 1)
    string = reg.wrap("core.string", "wrong")
    invalid_payload = Value("list<core.int>", "bad", scalar.meta, scalar.payload)
    invalid_child = Value("list<core.int>", "bad", scalar.meta, ListPayload((string,)))
    too_wide = make_list_value("core.int", (scalar,) * 4097)
    for value in (invalid_payload, invalid_child, too_wide):
        outcome = InvocationReturn(
            ReturnBatch("scalar", (DirectReturn((("out", PresentOutput(value)),)),))
        )
        with pytest.raises(BoundaryError):
            encode_invocation_outcome(
                ValueCodec(reg, use_shm=False),
                outcome,
                "inv",
                0.0,
                negotiated_capabilities=CAPABILITIES,
            )


def test_typed_encode_rolls_back_segments_on_later_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = registry()
    large = reg.wrap("core.string", "x" * 32)
    scalar = reg.wrap("core.int", 1)
    invalid = Value("list<core.int>", "bad", scalar.meta, scalar.payload)
    outcome = InvocationReturn(
        ReturnBatch(
            "scalar",
            (DirectReturn((("a", PresentOutput(large)), ("b", PresentOutput(invalid)))),),
        )
    )
    released: list[str] = []
    original = boundary.release_segment

    def release(segment: object) -> None:
        released.append(cast(Any, segment).name)
        original(cast(Any, segment))

    monkeypatch.setattr(boundary, "release_segment", release)
    with pytest.raises(BoundaryError):
        encode_invocation_outcome(
            ValueCodec(reg, shm_threshold=1),
            outcome,
            "inv",
            0.0,
            negotiated_capabilities=CAPABILITIES,
        )
    assert len(released) == 1


def test_typed_encode_rolls_back_cas_retention_on_later_failure() -> None:
    reg = registry()
    first = reg.wrap("core.string", "retained")
    scalar = reg.wrap("core.int", 1)
    invalid = Value("list<core.int>", "bad", scalar.meta, scalar.payload)
    codec = ValueCodec(reg, use_shm=False, cas_threshold=1)
    codec.enable_cas()
    failed = InvocationReturn(
        ReturnBatch(
            "scalar",
            (DirectReturn((("a", PresentOutput(first)), ("b", PresentOutput(invalid)))),),
        )
    )
    with pytest.raises(BoundaryError):
        encode_invocation_outcome(codec, failed, "inv", 0.0, negotiated_capabilities=CAPABILITIES)
    retry = InvocationReturn(ReturnBatch("scalar", (DirectReturn((("a", PresentOutput(first)),)),)))
    header, blobs, _ = encode_invocation_outcome(
        codec, retry, "retry", 0.0, negotiated_capabilities=CAPABILITIES
    )
    wire = cast(dict[str, Any], algebra(header)["values"][0])
    assert "blob" in wire["payload"]
    peer = ValueCodec(reg, use_shm=False, cas_threshold=1)
    peer.enable_cas()
    decoded = cast(
        InvocationReturn,
        decode_invocation_outcome(peer, header, blobs, [], negotiated_capabilities=CAPABILITIES)[0],
    )
    unit = cast(DirectReturn, decoded.batch.units[0])
    assert cast(PresentOutput, unit.bindings[0][1]).value.resolve() == "retained"


@pytest.mark.parametrize(
    "text",
    [
        '{"kind":"return","kind":"return"}',
        '{ "kind":"return"}',
        '{"kind":"return","batch":{"units":[],"mode":"mapped"}}',
        '{"batch":{"mode":"mapped","units":[]},"kind":"return"}\n',
        '{"batch":{"mode":1e0,"units":[]},"kind":"return"}',
        '{"batch":{"mode":NaN,"units":[]},"kind":"return"}',
        '{"batch":{"mode":-0.0,"units":[]},"kind":"return"}',
    ],
)
def test_noncanonical_or_malformed_json_is_refused(text: str) -> None:
    header, blobs = encode(direct_outcome())
    algebra(header)["document"] = text
    with pytest.raises(BoundaryError):
        decode(header, blobs)


@pytest.mark.parametrize(
    "header_text",
    [
        '{"type":"result","type":"invoke"}',
        '{"type":NaN}',
        '{"type":Infinity}',
        '{"blobs":[true]}',
    ],
)
def test_frame_reader_refuses_non_strict_json(header_text: str) -> None:
    async def read() -> object:
        reader = asyncio.StreamReader()
        encoded = header_text.encode()
        reader.feed_data(len(encoded).to_bytes(4, "big") + encoded)
        reader.feed_eof()
        return await read_frame(reader)

    with pytest.raises(BoundaryError):
        asyncio.run(read())


def test_frame_reader_normalizes_invalid_utf8() -> None:
    async def read() -> object:
        reader = asyncio.StreamReader()
        reader.feed_data((1).to_bytes(4, "big") + b"\xff")
        reader.feed_eof()
        return await read_frame(reader)

    with pytest.raises(BoundaryError):
        asyncio.run(read())


def test_frame_writer_refuses_non_strict_json_before_writing() -> None:
    class Writer:
        writes: list[bytes]

        def __init__(self) -> None:
            self.writes = []

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        async def drain(self) -> None:
            return None

    async def write() -> Writer:
        writer = Writer()
        with pytest.raises(BoundaryError, match="strict JSON"):
            await write_frame(cast(Any, writer), {"data": float("nan")}, [])
        return writer

    writer = asyncio.run(write())
    assert writer.writes == []


def test_frame_writer_refuses_oversized_header_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Writer:
        writes: list[bytes]

        def __init__(self) -> None:
            self.writes = []

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        async def drain(self) -> None:
            return None

    async def write() -> Writer:
        writer = Writer()
        with pytest.raises(BoundaryError, match="header too large"):
            await write_frame(cast(Any, writer), {"data": "large"}, [])
        return writer

    monkeypatch.setattr(boundary, "_MAX_HEADER_BYTES", 1)
    writer = asyncio.run(write())
    assert writer.writes == []


def test_non_strict_invocation_header_is_a_node_error_without_session_death() -> None:
    reg = registry()
    session = BoundarySession(
        reg,
        role="test worker",
        pack="test",
        codec=ValueCodec(reg, use_shm=False),
    )
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    session._writer = cast(Any, object())  # noqa: SLF001 - serialization fails first
    invocation = Invocation(
        invocation_id="invalid-json",
        node_id="node",
        node_type="Node",
        inputs={},
        effective_schema=NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )

    async def scenario() -> object:
        result = await session.invoke(invocation)
        assert session.alive
        return result

    result = cast(InvocationResult, asyncio.run(scenario()))
    assert result.error is not None
    assert result.error.message == "frame header is not strict JSON"


def test_failed_invocation_write_rolls_back_cas_for_retry() -> None:
    reg = registry()
    codec = ValueCodec(reg, use_shm=False, cas_threshold=1)
    codec.enable_cas()
    session = BoundarySession(reg, role="test worker", pack="test", codec=codec)
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    session._writer = cast(Any, object())  # noqa: SLF001 - serialization fails first
    value = reg.wrap("core.string", "retained")
    invalid = Invocation(
        "invalid-json",
        "node",
        "Node",
        {"value": value},
        NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )

    result = asyncio.run(session.invoke(invalid))
    assert result.error is not None
    retry = Invocation("retry", "node", "Node", {"value": value}, NodeSchema("Node"))
    header, blobs, _, _ = boundary.encode_invocation(codec, retry)
    wire = cast(dict[str, Any], cast(dict[str, object], header["inputs"])["value"])
    assert "blob" in cast(dict[str, object], wire["payload"])
    peer = ValueCodec(reg, use_shm=False, cas_threshold=1)
    peer.enable_cas()
    decoded = decode_invocation(peer, header, blobs, [])
    assert decoded.inputs["value"].resolve() == "retained"


def test_failed_invocation_write_releases_registered_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = registry()
    session = BoundarySession(
        reg,
        role="test worker",
        pack="test",
        codec=ValueCodec(reg, shm_threshold=1),
    )
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    writes: list[bytes] = []

    class Writer:
        def write(self, data: bytes) -> None:
            writes.append(data)

        async def drain(self) -> None:
            return None

    session._writer = cast(Any, Writer())  # noqa: SLF001
    released: list[str] = []
    original_release = boundary.release_segment

    def release(segment: Any) -> None:
        released.append(segment.name)
        original_release(segment)

    monkeypatch.setattr("dinkster_workers.session.release_segment", release)
    invalid = Invocation(
        "invalid-json",
        "node",
        "Node",
        {"value": reg.wrap("core.string", "retained")},
        NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )

    with pytest.raises(BoundaryError, match="strict JSON"):
        asyncio.run(session._send_invocation(invalid))  # noqa: SLF001
    assert writes == []
    assert len(released) == 1
    assert session._sent_segments == {}  # noqa: SLF001


def test_concurrent_invocation_write_rollback_keeps_cas_peer_coherent() -> None:
    reg = registry()
    codec = ValueCodec(reg, use_shm=False, cas_threshold=1)
    codec.enable_cas()
    session = BoundarySession(reg, role="test worker", pack="test", codec=codec)
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    writes: list[bytes] = []

    class Writer:
        def write(self, data: bytes) -> None:
            writes.append(data)

        async def drain(self) -> None:
            return None

    session._writer = cast(Any, Writer())  # noqa: SLF001
    value = reg.wrap("core.string", "retained")
    invalid = Invocation(
        "invalid",
        "node",
        "Node",
        {"value": value},
        NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )
    valid = Invocation("valid", "node", "Node", {"value": value}, NodeSchema("Node"))

    async def scenario() -> tuple[object, object]:
        await session._send_lock.acquire()  # noqa: SLF001 - force both frames to queue
        try:
            invalid_task = asyncio.create_task(session._send_invocation(invalid))  # noqa: SLF001
            valid_task = asyncio.create_task(session._send_invocation(valid))  # noqa: SLF001
            await asyncio.sleep(0)
        finally:
            session._send_lock.release()  # noqa: SLF001
        return await asyncio.gather(invalid_task, valid_task, return_exceptions=True)

    results = asyncio.run(scenario())
    assert isinstance(results[0], BoundaryError)
    assert isinstance(results[1], dict)
    header = cast(dict[str, Any], json.loads(writes[1]))
    wire = cast(dict[str, Any], header["inputs"]["value"])
    assert "blob" in wire["payload"]
    peer = ValueCodec(reg, use_shm=False, cas_threshold=1)
    peer.enable_cas()
    decoded = decode_invocation(peer, header, writes[2:], [])
    assert decoded.inputs["value"].resolve() == "retained"


def test_queued_invocation_refuses_after_session_close_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = registry()
    codec = ValueCodec(reg, shm_threshold=64, cas_threshold=1)
    codec.enable_cas()
    session = BoundarySession(reg, role="test worker", pack="test", codec=codec)
    session._alive = True  # noqa: SLF001 - exercise a closing adopted session
    writes: list[bytes] = []

    class Writer:
        def write(self, data: bytes) -> None:
            writes.append(data)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    writer = Writer()
    session._writer = cast(Any, writer)  # noqa: SLF001
    created_segments: list[object] = []
    released_segments: list[object] = []

    class Segment:
        name = "queued-close-segment"

    def create_segment(_data: bytes | memoryview) -> Any:
        segment = Segment()
        created_segments.append(segment)
        return segment

    monkeypatch.setattr(boundary, "_create_segment", create_segment)
    monkeypatch.setattr("dinkster_workers.session.release_segment", released_segments.append)
    invocation = Invocation(
        "queued-close",
        "node",
        "Node",
        {
            "cas": reg.wrap("core.string", "retained"),
            "shm": reg.wrap("core.string", "x" * 256),
        },
        NodeSchema("Node"),
    )

    async def scenario() -> None:
        close_started = asyncio.Event()
        finish_close = asyncio.Event()

        async def pause_close() -> None:
            close_started.set()
            await finish_close.wait()

        monkeypatch.setattr(session, "_unwind_leases", pause_close)
        await session._send_lock.acquire()  # noqa: SLF001 - queue the invocation first
        send_task = asyncio.create_task(session._send_invocation(invocation))  # noqa: SLF001
        await asyncio.sleep(0)
        close_task = asyncio.create_task(session.close())
        await close_started.wait()
        assert not session.alive
        assert session._writer is writer  # noqa: SLF001 - close has not reached the writer
        session._send_lock.release()  # noqa: SLF001
        try:
            with pytest.raises(WorkerDied):
                await send_task
        finally:
            finish_close.set()
            await close_task

    asyncio.run(scenario())
    assert codec.conversation_checkpoint() == ({}, 0)
    assert created_segments == []
    assert released_segments == []
    assert session._sent_segments == {}  # noqa: SLF001
    assert writes == []


def test_queued_generic_send_releases_prebuilt_segment_after_close_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = registry()
    session = BoundarySession(
        reg,
        role="test worker",
        pack="test",
        codec=ValueCodec(reg, use_shm=False),
    )
    session._alive = True  # noqa: SLF001 - exercise an adopted session
    writes: list[bytes] = []
    cleanup_reached = asyncio.Event()
    finish_close = asyncio.Event()

    class Writer:
        def write(self, data: bytes) -> None:
            writes.append(data)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            cleanup_reached.set()
            await finish_close.wait()

    class Segment:
        name = "generic-send-close-segment"

    writer = Writer()
    segment = Segment()
    released: list[object] = []
    session._writer = cast(Any, writer)  # noqa: SLF001
    monkeypatch.setattr("dinkster_workers.session.release_segment", released.append)

    async def scenario() -> None:
        await session._send_lock.acquire()  # noqa: SLF001 - queue the send before close
        send_task = asyncio.create_task(session.send({"type": "test"}, [], [cast(Any, segment)]))
        await asyncio.sleep(0)
        close_task = asyncio.create_task(session.close())
        await cleanup_reached.wait()
        assert not session.alive
        assert session._writer is writer  # noqa: SLF001 - close has not nulled the writer
        assert session._sent_segments == {}  # noqa: SLF001 - cleanup already passed
        session._send_lock.release()  # noqa: SLF001
        try:
            with pytest.raises(WorkerDied):
                await send_task
            assert session._sent_segments == {}  # noqa: SLF001 - no late registration
        finally:
            finish_close.set()
            await close_task

    asyncio.run(scenario())
    assert writes == []
    assert released == [segment]


def test_invocation_encode_releases_partial_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = registry()
    codec = ValueCodec(reg, shm_threshold=1)
    value = reg.wrap("core.string", "large")
    invocation = Invocation(
        "partial",
        "node",
        "Node",
        {"a": value, "b": value},
        NodeSchema("Node"),
    )
    original_encode = codec.encode
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BoundaryError("second input failed")
        return original_encode(*args, **kwargs)

    released: list[str] = []
    original_release = boundary.release_segment

    def release(segment: Any) -> None:
        released.append(segment.name)
        original_release(segment)

    monkeypatch.setattr(codec, "encode", fail_second)
    monkeypatch.setattr(boundary, "release_segment", release)
    with pytest.raises(BoundaryError, match="second input"):
        boundary.encode_invocation(codec, invocation)
    assert len(released) == 1


@pytest.mark.parametrize(
    ("reply_id", "execute_ms"),
    [("first", 0.0), ("unknown", 0.0), ("first", "malformed")],
)
def test_unexpected_typed_frame_is_session_fatal_before_execute_time(
    reply_id: str, execute_ms: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = registry()
    session = BoundarySession(
        reg,
        role="test worker",
        pack="test",
        codec=ValueCodec(reg, use_shm=False),
    )
    closed: list[bool] = []
    released: list[object] = []

    class Writer:
        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(boundary, "release_segment", released.append)
    monkeypatch.setattr("dinkster_workers.session.release_segment", released.append)
    session._writer = cast(Any, Writer())  # noqa: SLF001
    session._alive = True  # noqa: SLF001
    sent_segment = object()
    session._sent_segments["sent"] = cast(Any, sent_segment)  # noqa: SLF001
    first = Invocation("first", "a", "Node", {}, NodeSchema("Node"))
    second = Invocation("second", "b", "Node", {}, NodeSchema("Node"))

    async def scenario() -> tuple[InvocationResult, InvocationResult]:
        reader = asyncio.StreamReader()
        session._reader = reader  # noqa: SLF001 - drive the production receive path
        session._reader_task = asyncio.create_task(session._read_loop())  # noqa: SLF001
        first_task = asyncio.create_task(session.invoke(first))
        second_task = asyncio.create_task(session.invoke(second))
        await asyncio.sleep(0)
        header = {
            "type": "result",
            "invocationId": reply_id,
            "executeMs": execute_ms,
            "resultAlgebra": {},
            "blobs": [],
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        reader.feed_data(len(encoded).to_bytes(4, "big") + encoded)
        await cast(asyncio.Task[None], session._reader_task)
        return await first_task, await second_task

    first_result, second_result = asyncio.run(scenario())
    assert not session.alive
    assert closed == [True]
    assert released == [sent_segment]
    assert first_result.error is not None
    assert "is not running" in first_result.error.message
    assert second_result.error is not None
    assert "is not running" in second_result.error.message


def test_unknown_missing_and_escaping_document_fields_are_refused() -> None:
    header, blobs = encode(direct_outcome())
    document = json.loads(algebra(header)["document"])
    document["extra"] = True
    with pytest.raises(BoundaryError, match="malformed"):
        decode(replace_document(header, document), blobs)

    del document["extra"]
    del document["batch"]
    with pytest.raises(BoundaryError, match="malformed"):
        decode(replace_document(header, document), blobs)

    expanded_header, expanded_blobs = encode(expanded_outcome())
    expanded = json.loads(algebra(expanded_header)["document"])
    expanded["batch"]["units"][0]["bindings"][0]["value"]["localNodeId"] = "escape"
    with pytest.raises(BoundaryError, match="malformed"):
        decode(replace_document(expanded_header, expanded), expanded_blobs)


def test_descriptor_indexes_must_be_contiguous_unique_and_used() -> None:
    header, blobs = encode(direct_outcome())
    document = json.loads(algebra(header)["document"])
    document["batch"]["units"][0]["bindings"][0]["value"]["valueIndex"] = 1
    with pytest.raises(BoundaryError, match="malformed"):
        decode(replace_document(header, document), blobs)

    header, blobs = encode(direct_outcome())
    algebra(header)["values"].append(copy.deepcopy(algebra(header)["values"][0]))
    algebra(header)["valueStats"].append(copy.deepcopy(algebra(header)["valueStats"][0]))
    with pytest.raises(BoundaryError, match="duplicate"):
        decode(header, blobs)

    header, blobs = encode(direct_outcome())
    cast(dict[str, object], algebra(header)["values"][0])["extra"] = True
    with pytest.raises(BoundaryError, match="descriptor fields"):
        decode(header, blobs)

    header, blobs = encode(direct_outcome())
    cast(dict[str, object], algebra(header)["values"][0])["metaBlob"] = 1
    with pytest.raises(BoundaryError, match="duplicate blob"):
        decode(header, blobs)

    header, blobs = encode(direct_outcome())
    blobs.append(b"trailing")
    with pytest.raises(BoundaryError, match="unused or duplicate blobs"):
        decode(header, blobs)

    header, blobs = encode(direct_outcome())
    cast(dict[str, object], algebra(header)["valueStats"][0])["codecMs"] = float("nan")
    with pytest.raises(BoundaryError, match="transfer stat"):
        decode(header, blobs)


def test_excessive_nesting_is_refused_before_typed_construction() -> None:
    header, blobs = encode(direct_outcome())
    nested: object = None
    for _ in range(70):
        nested = [nested]
    algebra(header)["document"] = json.dumps(nested, separators=(",", ":"))
    with pytest.raises(BoundaryError, match="JSON"):
        decode(header, blobs)


def test_legacy_result_headers_remain_exact_and_algebra_free() -> None:
    reg = registry()
    codec = ValueCodec(reg, use_shm=False)
    value = reg.wrap("core.int", 3)
    success, _, _ = encode_result(codec, InvocationResult(outputs={"out": value}), "legacy", 2.0)
    assert success["type"] == "result"
    assert set(success) == {"type", "invocationId", "executeMs", "outputs", "outputStats"}
    assert "resultAlgebra" not in success

    error, blobs, segments = encode_result(
        codec,
        InvocationResult(error=NodeError("node", "Type", "bad")),
        "legacy",
        2.0,
    )
    assert blobs == [] and segments == []
    assert error == {
        "type": "result",
        "invocationId": "legacy",
        "executeMs": 2.0,
        "error": {
            "nodeId": "node",
            "nodeType": "Type",
            "message": "bad",
            "traceback": "",
            "hints": [],
        },
    }
