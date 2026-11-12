"""Style-model application preserves ComfyUI conditioning semantics."""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
    materialize_clip_vision_output,
)
from dinkster_conditioning.style_model import StyleModelApply


def _model() -> Any:
    initialize_inference_process()
    projection = torch.nn.Linear(3, 2)
    projection.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]]))
    projection.bias = torch.nn.Parameter(torch.tensor([0.75, -0.5]))
    return importlib.import_module("dinkster_inference.sd").StyleModel(projection)


def _vision() -> dict[str, object]:
    return {
        "last_hidden_state": torch.tensor(
            [[[2.0, 3.0, -1.0], [-4.0, 0.5, 5.0]]], dtype=torch.float32
        ),
        "image_embeds": {"reference": torch.tensor([[5.0, -3.0]])},
        "image_sizes": [(448, 224, 3)],
    }


@pytest.mark.parametrize("strength", (0.0, 1.0, 2.5))
def test_multiply_strength_and_masks_preserve_inputs(strength: float) -> None:
    style = _model()
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    boolean_mask = torch.tensor(
        [
            [
                [True, False, True, True, False, True, True, False],
                [False, True, False, True, True, False, True, True],
                [True, True, False, False, True, True, False, True],
                [True, False, True, True, False, False, True, True],
                [False, True, True, False, True, True, False, False],
                [True, True, False, True, True, False, True, False],
                [False, True, True, False, True, False, True, True],
                [True, False, True, True, False, True, False, True],
            ]
        ],
        dtype=torch.bool,
    )
    additive_mask = torch.arange(8 * 8, dtype=torch.float16).reshape(1, 8, 8)
    source = [
        [text, {"attention_mask": boolean_mask, "attention_mask_img_shape": (2, 3), "tag": "one"}],
        [
            text + 10,
            {"attention_mask": additive_mask, "attention_mask_img_shape": (2, 3), "tag": "two"},
        ],
    ]
    original_text = [record[0].clone() for record in source]
    original_masks = [record[1]["attention_mask"].clone() for record in source]

    output = cast(
        list[list[Any]],
        StyleModelApply.execute(
            conditioning=source,
            style_model=style,
            clip_vision_output=_vision(),
            strength=strength,
            strength_type="multiply",
        )["CONDITIONING"],
    )

    expected = style.get_cond(materialize_clip_vision_output(_vision())).flatten(0, 1).unsqueeze(0)
    assert [record[1]["tag"] for record in output] == ["one", "two"]
    for index, (result, original, mask) in enumerate(
        zip(output, original_text, original_masks, strict=True)
    ):
        assert torch.equal(result[0][:, :2], original)
        assert torch.equal(result[0][:, 2:], expected * strength)
        new_mask = result[1]["attention_mask"]
        assert new_mask.shape == (1, 10, 10)
        assert result[1]["attention_mask_img_shape"] == (2, 3)
        expected_mask = torch.log(mask.to(torch.float16)) if index == 0 else mask
        assert torch.equal(new_mask[:, :2, :2], expected_mask[:, :2, :2])
        assert torch.equal(new_mask[:, :2, 4:], expected_mask[:, :2, 2:])
        assert torch.equal(new_mask[:, 4:, :2], expected_mask[:, 2:, :2])
        assert torch.equal(new_mask[:, 4:, 4:], expected_mask[:, 2:, 2:])
        assert torch.equal(new_mask[:, :2, 2:4], torch.zeros((1, 2, 2), dtype=torch.float16))
    for record, original, mask in zip(source, original_text, original_masks, strict=True):
        assert torch.equal(record[0], original)
        assert torch.equal(record[1]["attention_mask"], mask)


@pytest.mark.parametrize("strength", (0.0, 1.0, 0.25))
def test_attention_bias_strength_adds_mask_without_changing_tokens(strength: float) -> None:
    style = _model()
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    source = [[text, {"tag": "only"}]]

    output = cast(
        list[list[Any]],
        StyleModelApply.execute(
            conditioning=source,
            style_model=style,
            clip_vision_output=_vision(),
            strength=strength,
            strength_type="attn_bias",
        )["CONDITIONING"],
    )[0]
    expected = style.get_cond(materialize_clip_vision_output(_vision())).flatten(0, 1).unsqueeze(0)
    assert torch.equal(output[0][:, :2], text)
    assert torch.equal(output[0][:, 2:], expected)
    if strength == 1.0:
        assert "attention_mask" not in output[1]
    else:
        mask = output[1]["attention_mask"]
        assert mask.shape == (1, 5, 5)
        bias = torch.log(torch.tensor(strength, dtype=mask.dtype))
        assert torch.equal(mask[:, :2, 2:4], torch.full((1, 2, 2), bias, dtype=mask.dtype))
        assert torch.equal(mask[:, 4:, 2:4], torch.full((1, 1, 2), bias, dtype=mask.dtype))
    assert torch.equal(source[0][0], text)
    assert source[0][1] == {"tag": "only"}
