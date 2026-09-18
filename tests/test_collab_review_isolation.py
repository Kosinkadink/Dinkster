import asyncio
from typing import Any, cast

import pytest
from dinkster_collab import SessionService
from dinkster_collab.routes import _SUBSCRIBERS_KEY, _broadcast, _Subscriber


def test_append_copies_nested_caller_data_and_returned_ops() -> None:
    service = SessionService()
    session = service.create(scope="local", document_id="doc", snapshot={})
    value = {"nested": [1]}
    patch = [{"op": "add", "path": ["x"], "value": value}]
    returned, _ = service.append(
        session.session_id,
        op_id="op",
        actor_id="actor",
        base_revision=0,
        patch=patch,
    )
    value["nested"].append(2)
    cast("dict[str, Any]", returned.patch[0]["value"])["nested"].append(3)
    stored = service.ops_after(session.session_id, 0)[0]
    assert stored.patch[0]["value"] == {"nested": [1]}


def test_snapshots_and_session_accessors_are_isolated() -> None:
    service = SessionService()
    source = {"nested": [1]}
    created = service.create(scope="local", document_id="doc", snapshot=source)
    source["nested"].append(2)
    cast("dict[str, list[int]]", created.snapshot)["nested"].append(3)
    created.ops.append(cast(Any, "poison"))
    stored = service.get(created.session_id)
    assert stored.snapshot == {"nested": [1]}
    assert stored.ops == []
    cast("dict[str, list[int]]", stored.snapshot)["nested"].append(4)
    assert service.get(created.session_id).snapshot == {"nested": [1]}


def test_service_rejects_non_json_values() -> None:
    service = SessionService()
    with pytest.raises(ValueError, match="JSON-shaped"):
        service.create(scope="local", document_id="doc", snapshot={"bad": object()})


def test_broadcast_does_not_wait_for_slow_or_failing_sockets() -> None:
    class Socket:
        async def close(self) -> None:
            return None

    async def scenario() -> None:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        await queue.put("already full")
        sleeper = asyncio.create_task(asyncio.sleep(60))
        subscriber = _Subscriber(cast(Any, Socket()), queue, sleeper, "local", lambda: True)
        subscribers = {subscriber}
        request = cast(Any, type("Request", (), {"app": {}})())
        request.app[_SUBSCRIBERS_KEY] = {"sid": subscribers}
        await asyncio.wait_for(_broadcast(request, "sid", {"type": "op"}), timeout=0.1)
        assert subscriber not in subscribers
        await asyncio.sleep(0)

    asyncio.run(scenario())
