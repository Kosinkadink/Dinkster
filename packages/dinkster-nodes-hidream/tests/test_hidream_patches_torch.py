from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_hidream.hidream_patches import (
    HiDreamO1PatchSeamSmoothing,
    HiDreamO1ReferenceImages,
)


class Sampling:
    multiplier = 3.0

    def percent_to_sigma(self, percent: float) -> float:
        return (1.0 - percent) * 10.0


class Model:
    def __init__(self) -> None:
        self.wrappers: dict[tuple[str, str], Any] = {}

    def clone(self) -> Model:
        result = Model()
        result.wrappers = self.wrappers.copy()
        return result

    def get_model_object(self, name: str) -> Sampling:
        assert name == "model_sampling"
        return Sampling()

    def add_wrapper_with_key(self, point: str, name: str, wrapper: Any) -> None:
        self.wrappers[point, name] = wrapper


def test_reference_order_append_and_shallow_identity() -> None:
    token, negative_token, marker = object(), object(), object()
    prior, negative_prior = torch.tensor([-7.0]), torch.tensor([-13.0])
    positive = [[token, {"reference_latents": [prior], "marker": marker}]]
    negative = [[negative_token, {"reference_latents": [negative_prior], "prompt_type": "old"}]]
    images = {
        name: torch.full((1, 2, 3, 3), float(index))
        for name, index in (
            ("extra", 201),
            ("image_100", 100),
            ("image_10", 10),
            ("image_2", 2),
            ("tail", 301),
        )
    }
    images["image_1"] = None
    result = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, images=images)
    for output, original, previous in (
        ("positive", positive, prior),
        ("negative", negative, negative_prior),
    ):
        entry = result[output][0]
        assert entry[0] is original[0][0] and entry[1] is not original[0][1]
        refs = entry[1]["reference_latents"]
        assert refs[0] is previous and refs[1] is None
        assert all(
            actual is images[name]
            for actual, name in zip(
                refs[2:], ("image_2", "image_10", "image_100", "extra", "tail"), strict=True
            )
        )
        assert len(original[0][1]["reference_latents"]) == 1
    assert result["positive"][0][1]["marker"] is marker
    assert result["negative"][0][1]["prompt_type"] == "negative"
    assert negative[0][1]["prompt_type"] == "old"
    for empty in (None, {}):
        unchanged = HiDreamO1ReferenceImages.execute(
            positive=positive, negative=negative, images=empty
        )
        assert unchanged["positive"] is positive and unchanged["negative"] is negative


def test_shifted_grid_average_preserves_edges_and_scales_strength() -> None:
    source = Model()
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=source,
        start_percent=0.2,
        end_percent=0.8,
        pattern="single_shift",
        passes="2",
        blend="average",
        strength=0.625,
    )["MODEL"]
    assert isinstance(patched, Model)
    assert patched is not source and source.wrappers == {}
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    bias = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 283

    def predict(value: torch.Tensor, sigma: torch.Tensor, *, offset: float) -> torch.Tensor:
        return value.square() + bias + offset

    sigma = torch.tensor([15.0])
    natural = predict(image, sigma, offset=0.75)
    shifted = image.square() + torch.roll(bias, (-16, -16), (-2, -1)) + 0.75
    expected_center = natural * 0.375 + ((natural + shifted) / 2) * 0.625
    result = wrapper(predict, image, sigma, offset=0.75)
    torch.testing.assert_close(
        result[..., 36:60, 36:92], expected_center[..., 36:60, 36:92], rtol=0, atol=0
    )
    assert torch.equal(result[..., :33, :], natural[..., :33, :])
    assert torch.equal(result[..., -33:, :], natural[..., -33:, :])
    assert torch.equal(result[..., :, :33], natural[..., :, :33])
    assert torch.equal(result[..., :, -33:], natural[..., :, -33:])


@pytest.mark.parametrize("pattern", ("single_shift", "symmetric"))
@pytest.mark.parametrize("blend", ("average", "window", "median"))
def test_ramped_smoothing_window_boundaries_and_pass_counts(pattern: str, blend: str) -> None:
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=Model(),
        start_percent=0.2,
        end_percent=0.8,
        pattern=pattern,
        passes="ramp_2_4_8",
        blend=blend,
        strength=1.0,
    )["MODEL"]
    assert isinstance(patched, Model)
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    calls: list[torch.Tensor] = []

    def predict(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return value.square()

    for timestep, level in (
        (24.1, 0),
        (24.0, 2),
        (18.1, 2),
        (18.0, 4),
        (12.1, 4),
        (12.0, 8),
        (6.0, 8),
        (5.9, 0),
    ):
        calls.clear()
        actual = wrapper(predict, image, torch.tensor([timestep]))
        expected_calls = max(level, 1) + (pattern == "symmetric" and level > 0)
        assert len(calls) == expected_calls
        torch.testing.assert_close(actual, image.square(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "strength,start,end", ((0.0, 0.2, 0.8), (-0.5, 0.2, 0.8), (0.5, 0.7, 0.7), (0.5, 0.8, 0.2))
)
def test_disabled_smoothing_preserves_model_identity(
    strength: float, start: float, end: float
) -> None:
    model = Model()
    assert (
        HiDreamO1PatchSeamSmoothing.execute(
            model=model,
            start_percent=start,
            end_percent=end,
            pattern="single_shift",
            passes="2",
            blend="average",
            strength=strength,
        )["MODEL"]
        is model
    )
