from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
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


def test_attention_runtime_starts_on_cpu_without_comfy_arguments() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dinkster_native.attention import create_attention_runtime; "
            "runtime = create_attention_runtime(); "
            "import dinkster_inference.model_management as memory; "
            "assert memory.get_torch_device().type == 'cpu'; "
            "assert 'sdpa' in runtime.capabilities.available_policies",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_attention_runtime_resolves_upstream_named_attention() -> None:
    runtime = attention.create_attention_runtime()
    assert "sdpa" in runtime.capabilities.available_policies
    selected = runtime.resolve(runtime.route_token)
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10
    k = q + 0.2
    v = q - 0.1
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q.reshape(1, 3, 2, 4).transpose(1, 2),
            k.reshape(1, 3, 2, 4).transpose(1, 2),
            v.reshape(1, 3, 2, 4).transpose(1, 2),
        )
        .transpose(1, 2)
        .reshape(1, 3, 8)
    )
    torch.testing.assert_close(selected(q, k, v, 2), expected)


def test_native_sampling_choices_match_the_installed_upstream_engine() -> None:
    from dinkster_native.fork_nodes import GenerationKSampler
    from dinkster_schema import ComboOption, ComboWidget

    samplers = importlib.import_module("dinkster_inference.samplers")
    schema = GenerationKSampler.schema()
    for name, expected in (
        ("sampler_name", samplers.KSampler.SAMPLERS),
        ("scheduler", samplers.KSampler.SCHEDULERS),
    ):
        spec = schema.input(name)
        assert spec is not None and isinstance(spec.widget, ComboWidget)
        assert all(isinstance(option, ComboOption) for option in spec.widget.options)
        assert [cast("ComboOption", option).value for option in spec.widget.options] == [
            f"dinkster.{value}" for value in expected
        ]


@pytest.mark.parametrize("skip_output_reshape", (False, True))
def test_ulysses_attention_exchanges_local_sequences_and_heads(
    monkeypatch: pytest.MonkeyPatch, skip_output_reshape: bool
) -> None:
    config = attention._DistributedConfig(1, 2, "sequence", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    sequence_group = object()
    monkeypatch.setattr(attention, "_sequence_process_group", sequence_group)
    peer_q = torch.arange(1 * 4 * 3 * 2, dtype=torch.float32).reshape(1, 4, 3, 2)
    q = peer_q + 1000
    k = q + 100
    v = q + 200
    calls: list[tuple[Any, Any, Any, int]] = []
    masks: list[Any] = []

    def selected(
        local_q: Any,
        local_k: Any,
        local_v: Any,
        heads: int,
        **kwargs: object,
    ) -> Any:
        calls.append((local_q, local_k, local_v, heads))
        masks.append(kwargs["mask"])
        return local_v

    def all_gather(outputs: list[Any], value: Any, **kwargs: object) -> None:
        assert kwargs["group"] is sequence_group
        if value.dtype == torch.int64:
            for output in outputs:
                output.copy_(value)

    exchanges = 0

    def all_to_all(outputs: list[Any], inputs: list[Any], **kwargs: object) -> None:
        nonlocal exchanges
        assert kwargs["group"] is sequence_group
        if exchanges < 3:
            peer = (peer_q, peer_q + 100, peer_q + 200)[exchanges]
            outputs[0].copy_(peer[:, 2:4])
            outputs[1].copy_(inputs[1])
        else:
            outputs[0].copy_(v[:, :2])
            outputs[1].copy_(inputs[1])
        exchanges += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "all_to_all", all_to_all)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _value, **_kwargs: None)

    distributed = attention._UlyssesAttention(selected)  # pyright: ignore[reportPrivateUsage]
    output = distributed(
        q,
        k,
        v,
        4,
        skip_reshape=True,
        skip_output_reshape=skip_output_reshape,
        transformer_options={
            "dinkster_sequence_sharded": True,
            "dinkster_sequence_widths": (3, 3),
        },
    )

    expected = v if skip_output_reshape else v.transpose(1, 2).reshape(1, 3, -1)
    torch.testing.assert_close(output, expected)
    actual_q, actual_k, actual_v, actual_heads = calls[0]
    assert actual_heads == 2
    torch.testing.assert_close(actual_q, torch.cat((peer_q[:, 2:4], q[:, 2:4]), dim=2))
    torch.testing.assert_close(actual_k, torch.cat((peer_q[:, 2:4] + 100, k[:, 2:4]), dim=2))
    torch.testing.assert_close(actual_v, torch.cat((peer_q[:, 2:4] + 200, v[:, 2:4]), dim=2))
    assert masks == [None]
    assert exchanges == 4


def test_ulysses_attention_preserves_four_rank_head_and_sequence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(2, 4, "sequence", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    sequence_widths = (3, 2, 2, 2)
    peers = [
        torch.arange(1 * 8 * width * 2, dtype=torch.float32).reshape(1, 8, width, 2) + rank * 1000
        for rank, width in enumerate(sequence_widths)
    ]
    q = peers[2]
    k = q + 100
    v = q + 200
    calls: list[tuple[Any, Any, Any, int]] = []

    def selected(local_q: Any, local_k: Any, local_v: Any, heads: int, **_kwargs: object) -> Any:
        calls.append((local_q, local_k, local_v, heads))
        return local_v

    controls: list[Any] = []

    def all_gather(outputs: list[Any], value: Any, **_kwargs: object) -> None:
        if value.dtype == torch.int64:
            controls.append(value.clone())
            for output in outputs:
                output.copy_(value)

    exchanges = 0

    def all_to_all(outputs: list[Any], inputs: list[Any], **_kwargs: object) -> None:
        nonlocal exchanges
        if exchanges < 3:
            offset = exchanges * 100
            for source, output in enumerate(outputs):
                output.copy_(peers[source][:, 4:6] + offset)
        else:
            for source, output in enumerate(outputs):
                output.copy_(v[:, source * 2 : (source + 1) * 2])
        exchanges += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "all_to_all", all_to_all)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _value, **_kwargs: None)

    output = attention._UlyssesAttention(selected)(  # pyright: ignore[reportPrivateUsage]
        q,
        k,
        v,
        8,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options={
            "dinkster_sequence_sharded": True,
            "dinkster_sequence_widths": sequence_widths,
        },
    )

    torch.testing.assert_close(output, v)
    local_q, local_k, local_v, local_heads = calls[0]
    assert local_heads == 2
    torch.testing.assert_close(local_q, torch.cat([peer[:, 4:6] for peer in peers], dim=2))
    torch.testing.assert_close(local_k, local_q + 100)
    torch.testing.assert_close(local_v, local_q + 200)
    torch.testing.assert_close(controls[0][1:], torch.tensor([8, 1, 9, 2]))
    assert exchanges == 4


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
