from __future__ import annotations

import asyncio

from dinkster_workers.resume import InvocationKey, ResumableConversation

from tests.async_scenarios import run_scenario


def _key(invocation_id: str = "invoke-1") -> InvocationKey:
    return InvocationKey("job-1", 2, invocation_id)


def _entry(key: InvocationKey, *, cancelled: bool = False, sequence: int = 0) -> dict[str, object]:
    return {
        **key.to_wire(),
        "lastEventSeq": sequence,
        "cancelled": cancelled,
    }


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

    run_scenario(scenario())
