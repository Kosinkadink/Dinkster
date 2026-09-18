from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from dinkster_compat_comfy.workgroup import SingleJobWorkGroupHandler
from dinkster_memory import ReservationRequest
from dinkster_protocol import (
    MAX_REASON_BYTES,
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    BeginWorkGroup,
    CommitWorkGroup,
    DeviceResourceId,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaId,
    ReplicaRecipeId,
    RunWorkUnit,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupId,
    WorkGroupMessage,
    WorkGroupState,
    WorkUnitFailed,
    WorkUnitId,
)
from dinkster_workers import (
    WorkGroupCoordinator,
    WorkGroupLaneCandidate,
    build_workgroup_configuration,
)


class _Reservations:
    @asynccontextmanager
    async def reserve(self, requests: tuple[ReservationRequest, ...]):
        assert requests == ()
        yield


def test_native_invocations_are_gated_by_the_workgroup_lifecycle() -> None:
    async def scenario() -> None:
        handlers = (SingleJobWorkGroupHandler(), SingleJobWorkGroupHandler())
        queues = (asyncio.Queue[WorkGroupMessage](), asyncio.Queue[WorkGroupMessage]())
        recipe = ReplicaRecipeId("sha256:" + "1" * 64)

        def candidate(rank: int) -> WorkGroupLaneCandidate:
            async def send(message: WorkGroupMessage) -> None:
                for reply in await handlers[rank](message):
                    await queues[rank].put(reply)

            return WorkGroupLaneCandidate(
                ReplicaId(f"rank-{rank}"),
                WorkerInstanceId(f"worker-{rank}"),
                DeviceResourceId(f"cuda-{rank}"),
                recipe,
                WorkUnitId(f"sample-{rank}"),
                SemanticSlot.SINGLE,
                frozenset({WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY}),
                send,
                queues[rank].get,
            )

        definition, endpoints = build_workgroup_configuration(
            WorkGroupId("single-job"),
            WorkGroupAttempt(1),
            (candidate(0), candidate(1)),
        )
        coordinator = asyncio.create_task(
            WorkGroupCoordinator(_Reservations()).execute(definition, endpoints)  # type: ignore[arg-type]
        )

        async def invoke(rank: int) -> None:
            await handlers[rank].before_invocation(f"invoke-{rank}")
            await handlers[rank].after_invocation(f"invoke-{rank}", None)

        await asyncio.gather(*(invoke(rank) for rank in range(2)))
        lifecycle = await coordinator

        assert lifecycle.state is WorkGroupState.SUCCEEDED
        assert lifecycle.completed == (WorkUnitId("sample-0"), WorkUnitId("sample-1"))
        assert not handlers[0]._attempts  # pyright: ignore[reportPrivateUsage]
        assert not handlers[1]._attempts  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_rank_failure_reason_is_protocol_bounded() -> None:
    async def scenario() -> None:
        handler = SingleJobWorkGroupHandler()
        common = {
            "worker": WorkerInstanceId("worker-0"),
            "replica": ReplicaId("rank-0"),
            "group": WorkGroupId("single-job"),
            "attempt": WorkGroupAttempt(1),
            "device": DeviceResourceId("cuda-0"),
        }
        prepare = PrepareReplica(
            **common,  # type: ignore[arg-type]
            recipe=ReplicaRecipeId("sha256:" + "1" * 64),
        )
        await handler(prepare)
        await handler(BeginWorkGroup(**common))  # type: ignore[arg-type]
        await handler(CommitWorkGroup(**common))  # type: ignore[arg-type]
        run = asyncio.create_task(
            handler(
                RunWorkUnit(  # type: ignore[arg-type]
                    **common,
                    unit=WorkUnitId("sample-0"),
                    slot=SemanticSlot.SINGLE,
                )
            )
        )
        await handler.before_invocation("invoke-0")
        await handler.after_invocation("invoke-0", "ValueError: failure " + "x" * 5000)

        (reply,) = await run
        assert isinstance(reply, WorkUnitFailed)
        assert reply.reason.startswith("ValueError: failure")
        assert len(reply.reason.encode("utf-8")) <= MAX_REASON_BYTES
        await handler(ReleaseWorkGroup(**common))  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_debug_flag_logs_pinned_storage_at_invocation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[str] = []
    monkeypatch.setenv("DINKSTER_PINNED_STORAGE_DEBUG", "1")
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch.pinned_host",
        SimpleNamespace(log_storage_ledger=phases.append),
    )

    asyncio.run(SingleJobWorkGroupHandler().after_invocation("decode", None))

    assert phases == ["decode"]
