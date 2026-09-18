from __future__ import annotations

import asyncio
from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory

import dinkster_workers.resume as resume_module
import pytest
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import BoundaryError, ValueCodec
from dinkster_workers.resume import InvocationKey, ResumableConversation
from dinkster_workers.session import BoundarySession


def _key(invocation_id: str = "invoke-1") -> InvocationKey:
    return InvocationKey("job-1", 2, invocation_id)


def _entry(key: InvocationKey, *, cancelled: bool = False, sequence: int = 0) -> dict[str, object]:
    return {
        **key.to_wire(),
        "lastEventSeq": sequence,
        "cancelled": cancelled,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("jobRef", ""),
        ("jobRef", 1),
        ("attemptId", 0),
        ("attemptId", True),
        ("invocationId", ""),
        ("invocationId", None),
    ),
)
def test_invocation_key_wire_is_strict(field: str, value: object) -> None:
    wire = _key().to_wire()
    wire[field] = value
    with pytest.raises(BoundaryError, match="invocation identity"):
        InvocationKey.from_header(wire)


def test_rebind_reports_running_completed_missing_and_fences_epochs() -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        running = _key("running")
        completed = _key("completed")
        missing = _key("missing")
        conversation.register(running.to_wire())
        conversation.register(completed.to_wire())
        await conversation.send(
            {"type": "result", "invocationId": completed.invocation_id},
            [b"result"],
        )
        response, requested, cancelled = conversation.plan_rebind(
            {
                "ownerEpoch": 0,
                "invocations": [_entry(running), _entry(completed), _entry(missing)],
            }
        )
        assert response["invocations"] == [
            {**running.to_wire(), "status": "running"},
            {**completed.to_wire(), "status": "completed"},
            {**missing.to_wire(), "status": "missing"},
        ]
        assert requested == (running, completed)
        assert cancelled == ()
        with pytest.raises(BoundaryError, match="epoch"):
            conversation.plan_rebind({"ownerEpoch": 1, "invocations": []})
        with pytest.raises(BoundaryError, match="epoch"):
            conversation.plan_rebind({"ownerEpoch": False, "invocations": []})
        with pytest.raises(BoundaryError, match="omitted"):
            conversation.plan_rebind({"ownerEpoch": 0, "invocations": [_entry(running)]})

    asyncio.run(scenario())


def test_disconnected_events_keep_only_the_latest_data_snapshot() -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        key = conversation.register(_key().to_wire())
        await conversation.send(
            {"type": "invocationEvent", "invocationId": key.invocation_id, "data": {"step": 1}}
        )
        await conversation.send(
            {"type": "invocationEvent", "invocationId": key.invocation_id, "data": {"step": 2}},
            [b"preview"],
        )
        await conversation.send(
            {"type": "invocationEvent", "invocationId": key.invocation_id, "data": {"step": 3}}
        )
        record = conversation._records[key]
        assert record.event_seq == 3
        assert record.latest_event is not None
        assert record.latest_event[0]["eventSeq"] == 3
        assert record.latest_event[0]["data"] == {"step": 3}
        assert record.latest_event[1] == []

    asyncio.run(scenario())


def test_event_snapshot_retention_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        key = conversation.register(_key().to_wire())
        await conversation.send(
            {"type": "invocationEvent", "invocationId": key.invocation_id, "data": {"step": 1}}
        )
        assert conversation._records[key].latest_event is not None
        monkeypatch.setattr(resume_module, "MAX_RETAINED_EVENT_BYTES", 1)
        await conversation.send(
            {"type": "invocationEvent", "invocationId": key.invocation_id, "data": {"step": 2}}
        )
        assert conversation._records[key].latest_event is None
        assert conversation._retained_event_bytes == 0

    asyncio.run(scenario())


def test_result_is_retained_until_exact_acknowledgement() -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        key = conversation.register(_key().to_wire())
        await conversation.send({"type": "result", "invocationId": key.invocation_id}, [b"final"])
        assert conversation.has_records
        assert conversation._records[key].result_bytes > len(b"final")
        with pytest.raises(BoundaryError, match="acknowledgement"):
            conversation.acknowledge_result(
                InvocationKey(key.job_ref, key.attempt_id + 1, key.invocation_id).to_wire()
            )
        assert conversation.has_records
        conversation.acknowledge_result(key.to_wire())
        assert not conversation.has_records

    asyncio.run(scenario())


def test_result_acknowledgement_cannot_overtake_memory_release() -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        key = conversation.register(_key().to_wire())
        await conversation.send(
            {
                "type": "memoryReserve",
                "requestId": key.invocation_id,
                "requests": [{"residency": "ram", "nbytes": 1}],
            }
        )
        await conversation.send({"type": "result", "invocationId": key.invocation_id})
        with pytest.raises(BoundaryError, match="memory release"):
            conversation.acknowledge_result(key.to_wire())
        await conversation.send({"type": "memoryRelease", "requestId": key.invocation_id})
        conversation.acknowledge_result(key.to_wire())
        assert not conversation.has_records

    asyncio.run(scenario())


def test_result_retention_exhaustion_closes_the_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        conversation = ResumableConversation("engine-1", "engine", "worker-1")
        key = conversation.register(_key().to_wire())
        monkeypatch.setattr(resume_module, "MAX_RETAINED_RESULT_BYTES", 1)
        with pytest.raises(BoundaryError, match="retention capacity"):
            await conversation.send(
                {"type": "result", "invocationId": key.invocation_id}, [b"result"]
            )
        assert await conversation.read() is None

    asyncio.run(scenario())


def test_disconnected_conversation_expires_after_its_grace() -> None:
    async def scenario() -> None:
        closed = asyncio.Event()

        async def on_close(_conversation: ResumableConversation) -> None:
            closed.set()

        conversation = ResumableConversation(
            "engine-1",
            "engine",
            "worker-1",
            grace=0.01,
            on_close=on_close,
        )
        conversation.register(_key().to_wire())
        await conversation.detach(conversation.owner_epoch)
        await asyncio.wait_for(closed.wait(), 1.0)
        assert await conversation.read() is None

    asyncio.run(scenario())


def test_replayed_memory_reserve_keeps_one_client_lease() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        session = BoundarySession(
            registry,
            role="remote worker",
            pack="pack",
            codec=ValueCodec(registry, use_shm=False),
            resumable=True,
        )
        reserve = {
            "type": "memoryReserve",
            "requestId": "invoke-1",
            "requests": [{"residency": "ram", "nbytes": 10}],
        }
        session._start_lease(reserve)  # noqa: SLF001 - exercise replay deduplication
        await asyncio.sleep(0)
        first = session._leases["invoke-1"]  # noqa: SLF001
        session._start_lease(reserve)  # noqa: SLF001 - same reserve replayed on rebind
        assert session._leases["invoke-1"] is first  # noqa: SLF001
        first[1].set()
        await first[0]
        assert session._leases == {}  # noqa: SLF001

    asyncio.run(scenario())


def test_replayed_memory_reserve_resends_the_retained_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        session = BoundarySession(
            registry,
            role="remote worker",
            pack="pack",
            codec=ValueCodec(registry, use_shm=False),
            resumable=True,
        )
        sent: list[dict[str, object]] = []

        async def record_send(
            header: dict[str, object],
            blobs: Sequence[bytes],
            segments: Sequence[SharedMemory] = (),
        ) -> None:
            del blobs, segments
            sent.append(header)

        monkeypatch.setattr(session, "send", record_send)
        session._lease_decisions["invoke-1"] = {
            "type": "memoryGrant",
            "requestId": "invoke-1",
        }
        session._start_lease(
            {
                "type": "memoryReserve",
                "requestId": "invoke-1",
                "requests": [{"residency": "ram", "nbytes": 10}],
            }
        )
        await asyncio.sleep(0)
        assert sent == [{"type": "memoryGrant", "requestId": "invoke-1"}]

    asyncio.run(scenario())
