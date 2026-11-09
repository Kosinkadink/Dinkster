from __future__ import annotations

import importlib
from typing import Any

import torch
from dinkster_api.v1 import (
    materialize_clip_vision_output,
)
from dinkster_sd.clip import unCLIPConditioning


def _vision_output(value: float) -> Any:
    return materialize_clip_vision_output(
        {
            "image_embeds": torch.tensor([[value, value + 0.5]], dtype=torch.float32),
            "image_sizes": [(224, 336, 3)],
        }
    )


def _vision_values(conditioning: list[list[Any]]) -> list[list[dict[str, object]]]:
    return [entry[1]["unclip_conditioning"] for entry in conditioning]


def _assert_record(
    record: dict[str, object],
    value: float,
    image_sizes: list[tuple[int, int, int]],
    strength: float | None,
    noise: float | None,
) -> None:
    output = record["clip_vision_output"]
    assert type(output) is importlib.import_module("dinkster_inference.clip_vision").Output
    assert torch.equal(
        output.image_embeds, torch.tensor([[value, value + 0.5]], dtype=torch.float32)
    )
    assert output.image_sizes == image_sizes
    if strength is None:
        assert "strength" not in record
    else:
        assert record["strength"] == strength
    if noise is None:
        assert "noise_augmentation" not in record
    else:
        assert record["noise_augmentation"] == noise


def test_unclip_conditioning_appends_without_mutating_ordered_metadata() -> None:
    earlier = _vision_output(1.0)
    earlier_second = _vision_output(2.0)
    conditioning = [
        [
            torch.tensor([[3.0]]),
            {"tag": "first", "unclip_conditioning": [{"clip_vision_output": earlier}]},
        ],
        [
            torch.tensor([[4.0]]),
            {
                "tag": "second",
                "unclip_conditioning": [
                    {"clip_vision_output": earlier},
                    {"clip_vision_output": earlier_second},
                ],
            },
        ],
        [torch.tensor([[6.0]]), {"tag": "third", "unclip_conditioning": None}],
    ]
    source_metadata = [entry[1].copy() for entry in conditioning]

    result: Any = unCLIPConditioning.execute(
        conditioning=conditioning,
        clip_vision_output={
            "image_embeds": torch.tensor([[5.0, 5.5]], dtype=torch.float32),
            "image_sizes": [(448, 224, 3)],
        },
        strength=-2.5,
        noise_augmentation=0.75,
    )["CONDITIONING"]

    assert [entry[0] is source[0] for entry, source in zip(result, conditioning, strict=True)] == [
        True,
        True,
        True,
    ]
    assert [entry[1]["tag"] for entry in result] == ["first", "second", "third"]
    assert [len(records) for records in _vision_values(result)] == [2, 3, 1]
    _assert_record(_vision_values(result)[0][0], 1.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[0][1], 5.0, [(448, 224, 3)], -2.5, 0.75)
    _assert_record(_vision_values(result)[1][0], 1.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[1][1], 2.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[1][2], 5.0, [(448, 224, 3)], -2.5, 0.75)
    _assert_record(_vision_values(result)[2][0], 5.0, [(448, 224, 3)], -2.5, 0.75)
    assert [entry[1] for entry in conditioning] == source_metadata


def test_unclip_conditioning_strength_zero_returns_source() -> None:
    conditioning = [[torch.tensor([[3.0]]), {"tag": "source"}]]

    result: Any = unCLIPConditioning.execute(
        conditioning=conditioning,
        clip_vision_output={"image_embeds": torch.tensor([[5.0]])},
        strength=0.0,
        noise_augmentation=0.25,
    )["CONDITIONING"]

    assert result is conditioning
