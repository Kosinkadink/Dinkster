from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from dinkster_native import attention  # noqa: E402
from dinkster_native.workgroup import SingleJobWorkGroupHandler  # noqa: E402
from dinkster_protocol import (  # noqa: E402
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
    WorkUnitId,
    WorkUnitResult,
)


def test_attention_runtime_owns_an_isolated_named_registry() -> None:
    first = attention.create_attention_runtime()
    second = attention.create_attention_runtime()

    def marker() -> str:
        return "first"

    first.registry["worker"] = marker

    assert first.registry is not second.registry
    assert first.registry["worker"] is marker
    assert "worker" not in second.registry
    assert first.resolve(first.route_token) is first.registry["pytorch"]


@pytest.mark.parametrize("skip_output_reshape", (False, True))
def test_distributed_attention_partitions_heads_and_gathers_in_order(
    monkeypatch: pytest.MonkeyPatch, skip_output_reshape: bool
) -> None:
    config = attention._DistributedConfig(1, 2, "sequence", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    q = torch.arange(1 * 4 * 3 * 2, dtype=torch.float32).reshape(1, 4, 3, 2)
    k = q + 100
    v = q + 200
    calls: list[tuple[Any, Any, Any, int]] = []

    def selected(
        local_q: Any,
        local_k: Any,
        local_v: Any,
        heads: int,
        **_kwargs: object,
    ) -> Any:
        calls.append((local_q, local_k, local_v, heads))
        if skip_output_reshape:
            return local_v
        return local_v.transpose(1, 2).reshape(1, 3, -1)

    def all_gather(outputs: list[Any], value: Any) -> None:
        if value.dtype == torch.int64:
            for output in outputs:
                output.copy_(value)
            return
        rank_zero = v[:, :2]
        if not skip_output_reshape:
            rank_zero = rank_zero.transpose(1, 2).reshape(1, 3, -1)
        outputs[0].copy_(rank_zero)
        outputs[1].copy_(value)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _value, **_kwargs: None)

    distributed = attention._DistributedAttention(selected)  # pyright: ignore[reportPrivateUsage]
    output = distributed(
        q,
        k,
        v,
        4,
        skip_reshape=True,
        skip_output_reshape=skip_output_reshape,
    )

    expected = v if skip_output_reshape else v.transpose(1, 2).reshape(1, 3, -1)
    torch.testing.assert_close(output, expected)
    actual_q, actual_k, actual_v, actual_heads = calls[0]
    assert actual_heads == 4
    torch.testing.assert_close(actual_q, q)
    torch.testing.assert_close(actual_k, k)
    torch.testing.assert_close(actual_v, v)


def test_workgroup_attempt_gates_one_invocation_and_releases_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "dinkster_native.workgroup.activate_distributed_attention",
        lambda group, attempt: events.append(("activate", group, attempt)),
    )
    monkeypatch.setattr(
        "dinkster_native.workgroup.release_distributed_attention",
        lambda group, attempt: events.append(("release", group, attempt)),
    )

    async def scenario() -> None:
        handler = SingleJobWorkGroupHandler()
        common = {
            "worker": WorkerInstanceId("worker-0"),
            "replica": ReplicaId("rank-0"),
            "group": WorkGroupId("single-job"),
            "attempt": WorkGroupAttempt(1),
            "device": DeviceResourceId("cuda-0"),
        }
        await handler(
            PrepareReplica(  # type: ignore[arg-type]
                **common,
                recipe=ReplicaRecipeId("sha256:" + "1" * 64),
            )
        )
        await handler(BeginWorkGroup(**common))  # type: ignore[arg-type]
        await handler(CommitWorkGroup(**common))  # type: ignore[arg-type]
        running = asyncio.create_task(
            handler(
                RunWorkUnit(  # type: ignore[arg-type]
                    **common,
                    unit=WorkUnitId("sample-0"),
                    slot=SemanticSlot.SINGLE,
                )
            )
        )
        await handler.before_invocation("invoke-0")
        await handler.after_invocation("invoke-0", None)
        assert isinstance((await running)[0], WorkUnitResult)
        await handler(ReleaseWorkGroup(**common))  # type: ignore[arg-type]

    asyncio.run(scenario())
    assert events == [("activate", "single-job", 1), ("release", "single-job", 1)]
