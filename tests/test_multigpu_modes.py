from __future__ import annotations

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


def test_sequence_block_patch_shards_modulation_and_gathers_last_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = attention._DistributedConfig(2, 4, "sequence", "file:///unused", "token")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention, "_ensure_process_group", lambda: config)
    hidden = torch.arange(14.0).reshape(7, 2)
    rope = torch.arange(7.0).reshape(1, 7, 1)
    rows = torch.tensor([20, 21, 22, 23, 24])
    options: dict[str, object] = {}

    def original(args: dict[str, Any]) -> dict[str, Any]:
        assert options["dinkster_sequence_sharded"] is True
        assert options["dinkster_sequence_widths"] == (2, 2, 2, 1)
        torch.testing.assert_close(args["img"], hidden[4:6])
        torch.testing.assert_close(args["rope_freqs"], rope[:, 4:6])
        assert len(args["mod_segments"]) == 1
        start, stop, row = args["mod_segments"][0]
        assert (start, stop) == (0, 2)
        torch.testing.assert_close(row, torch.tensor([22, 23]))
        return {"img": args["img"] + 1}

    def all_gather(outputs: list[Any], value: Any, **_kwargs: object) -> None:
        for rank, output in enumerate(outputs):
            output.fill_(rank)
        outputs[2].copy_(value)

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
    assert "dinkster_sequence_widths" not in options
    torch.testing.assert_close(result[:2], torch.zeros((2, 2)))
    torch.testing.assert_close(result[2:4], torch.ones((2, 2)))
    torch.testing.assert_close(result[4:6], hidden[4:6] + 1)
    torch.testing.assert_close(result[6:], torch.full((1, 2), 3.0))
