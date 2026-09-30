from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from dinkster_native import attention, multigpu  # noqa: E402


def test_guidance_wrapper_assigns_only_model_lanes_and_reduces_canonical_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(1, 2, "guidance", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    calls: list[list[Any]] = []

    def evaluate(_model: object, conds: list[Any], *_args: object) -> list[Any]:
        calls.append(conds)
        return [torch.tensor(0.0), torch.tensor(20.0), torch.tensor(0.0)]

    output_index = 0

    def all_reduce(value: Any, **_kwargs: object) -> None:
        nonlocal output_index
        if value.dtype == torch.int32:
            return
        if output_index == 0:
            value.add_(10.0)
        output_index += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    conds = [[{"lane": "positive"}], [{"lane": "negative"}], None]

    outputs = multigpu._guidance_wrapper(  # pyright: ignore[reportPrivateUsage]
        evaluate, object(), conds, torch.zeros(1), torch.ones(1), {}
    )

    assert calls == [[None, conds[1], None]]
    assert [float(value) for value in outputs] == [10.0, 20.0, 0.0]


def test_window_wrapper_uses_contiguous_four_rank_assignment_and_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(2, 4, "window", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    visited: list[int] = []

    class WindowExecutor:
        plan = SimpleNamespace(joint_windows=tuple(range(6)))

        def evaluate_window(self, index: int, *_args: object) -> list[Any]:
            visited.append(index)
            return [torch.tensor(float(index + 1))]

        def window_latent(self, _index: int, _template: Any) -> Any:
            return torch.empty(())

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda value, **_kwargs: None,
    )
    gather_calls = 0

    def all_gather(outputs: list[Any], value: Any) -> None:
        nonlocal gather_calls
        assert tuple(value.shape) == (3,)
        for rank, packed in enumerate(
            ((0.0, 1.0, 2.0), (0.0, 3.0, 4.0), (0.0, 5.0, -1.0), (0.0, 6.0, -1.0))
        ):
            outputs[rank].copy_(torch.tensor(packed))
        gather_calls += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    executor = SimpleNamespace(class_obj=WindowExecutor())

    outputs = multigpu._window_wrapper(  # pyright: ignore[reportPrivateUsage]
        executor,
        object(),
        object(),
        [[{}]],
        torch.zeros(()),
        torch.ones(1),
        {},
        False,
    )

    assert visited == [4]
    assert gather_calls == 1
    assert [float(value) for value in outputs[0]] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_window_wrapper_reports_peer_failure_through_the_output_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(0, 2, "window", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)

    class WindowExecutor:
        plan = SimpleNamespace(joint_windows=(0, 1))

        def evaluate_window(self, index: int, *_args: object) -> list[Any]:
            return [torch.tensor(float(index + 1))]

        def window_latent(self, _index: int, _template: Any) -> Any:
            return torch.empty(())

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)

    def all_gather(outputs: list[Any], _value: Any) -> None:
        outputs[0].copy_(torch.tensor((0.0, 1.0)))
        outputs[1].copy_(torch.tensor((1.0, 0.0)))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    with pytest.raises(RuntimeError, match="peer distributed execution rank failed"):
        multigpu._window_wrapper(  # pyright: ignore[reportPrivateUsage]
            SimpleNamespace(class_obj=WindowExecutor()),
            object(),
            object(),
            [[{}]],
            torch.zeros(()),
            torch.ones(1),
            {},
            False,
        )


def test_window_wrapper_preserves_local_failure_after_the_output_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(0, 2, "window", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)

    class WindowExecutor:
        plan = SimpleNamespace(joint_windows=(0, 1))

        def evaluate_window(self, _index: int, *_args: object) -> list[Any]:
            raise ValueError("local window failed")

        def window_latent(self, _index: int, _template: Any) -> Any:
            return torch.empty(())

    def all_gather(outputs: list[Any], value: Any) -> None:
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    with pytest.raises(ValueError, match="local window failed"):
        multigpu._window_wrapper(  # pyright: ignore[reportPrivateUsage]
            SimpleNamespace(class_obj=WindowExecutor()),
            object(),
            object(),
            [[{}]],
            torch.zeros(()),
            torch.ones(1),
            {},
            False,
        )


def test_window_gather_preserves_nested_stream_dtypes_shapes_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = importlib.import_module("dinkster_inference.nested_tensor").NestedTensor

    expected = [
        nested((torch.empty(1, dtype=torch.float32), torch.empty(2, dtype=torch.float16)))
        for _ in range(2)
    ]
    local = {0: [nested((torch.tensor([1.0]), torch.tensor([10.0, 11.0], dtype=torch.float16)))]}
    gathers = []

    def all_gather(outputs: list[Any], value: Any) -> None:
        gathers.append(value.dtype)
        if value.dtype is torch.float32:
            assert torch.equal(value, torch.tensor([0.0, 1.0]))
            outputs[0].copy_(torch.tensor([0.0, 1.0]))
            outputs[1].copy_(torch.tensor([0.0, 2.0]))
        else:
            assert torch.equal(value, torch.tensor([0.0, 10.0, 11.0], dtype=torch.float16))
            outputs[0].copy_(torch.tensor([0.0, 10.0, 11.0], dtype=torch.float16))
            outputs[1].copy_(torch.tensor([0.0, 20.0, 21.0], dtype=torch.float16))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    outputs = multigpu._gather_window_outputs(  # pyright: ignore[reportPrivateUsage]
        local,
        expected,
        1,
        SimpleNamespace(rank=0, world_size=2),
        None,
    )

    assert gathers == [torch.float32, torch.float16]
    assert [part.tolist() for part in outputs[0][0].unbind()] == [[1.0], [10.0, 11.0]]
    assert [part.tolist() for part in outputs[0][1].unbind()] == [[2.0], [20.0, 21.0]]


def test_sequence_block_patch_shards_modulation_and_gathers_last_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(1, 2, "sequence", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    hidden = torch.arange(12.0).reshape(6, 2)
    rope = torch.arange(6.0).reshape(1, 6, 1)
    rows = torch.tensor([20, 21, 22, 23])
    options: dict[str, object] = {}

    def original(args: dict[str, Any]) -> dict[str, Any]:
        assert options["dinkster_sequence_sharded"] is True
        assert options["dinkster_sequence_valid"] == 6
        torch.testing.assert_close(args["img"], hidden[3:6])
        torch.testing.assert_close(args["rope_freqs"], rope[:, 3:6])
        assert len(args["mod_segments"]) == 1
        start, stop, row = args["mod_segments"][0]
        assert (start, stop) == (0, 3)
        torch.testing.assert_close(row, torch.tensor([21, 22, 23]))
        return {"img": args["img"] + 1}

    def all_gather(outputs: list[Any], value: Any) -> None:
        outputs[0].fill_(-1)
        outputs[1].copy_(value)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    patch = multigpu._SequenceBlockPatch(0, 1)  # pyright: ignore[reportPrivateUsage]

    result = patch(
        {
            "img": hidden,
            "rope_freqs": rope,
            "mod_segments": [(0, 2, 0), (2, 6, rows)],
            "transformer_options": options,
        },
        {"original_block": original},
    )["img"]

    assert "dinkster_sequence_sharded" not in options
    assert "dinkster_sequence_valid" not in options
    torch.testing.assert_close(result[:3], torch.full((3, 2), -1.0))
    torch.testing.assert_close(result[3:], hidden[3:6] + 1)
