"""Core latent node interfaces and metadata behavior stay exact."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_latent.latent import (
    LatentBlend,
    LatentComposite,
    LatentCrop,
    LatentFromBatch,
    LatentUpscale,
    RepeatLatentBatch,
    SetLatentNoiseMask,
)


class ArrayTensor:
    """CPU array fixture for tensor-only slicing, cloning and repetition."""

    def __init__(self, values: object) -> None:
        self.array = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def __getitem__(self, key: object) -> ArrayTensor:
        return ArrayTensor(self.array[cast(Any, key)])

    def __setitem__(self, key: object, value: ArrayTensor) -> None:
        self.array[cast(Any, key)] = value.array

    def __mul__(self, value: float) -> ArrayTensor:
        return ArrayTensor(self.array * value)

    def __add__(self, value: ArrayTensor) -> ArrayTensor:
        return ArrayTensor(self.array + value.array)

    def clone(self) -> ArrayTensor:
        return ArrayTensor(self.array.copy())

    def repeat(self, *shape: Any) -> ArrayTensor:
        dimensions = shape[0] if len(shape) == 1 else shape
        return ArrayTensor(np.tile(self.array, dimensions))

    def reshape(self, shape: tuple[int, ...]) -> ArrayTensor:
        return ArrayTensor(self.array.reshape(shape))

    def detach(self) -> ArrayTensor:
        return self


@pytest.mark.parametrize(("index", "expected"), [(-2, [20, 30]), (-99, [10, 20]), (99, [30])])
def test_batch_selection_clamps_indices_tiles_masks_and_copies(
    index: int, expected: list[int]
) -> None:
    source = ArrayTensor(np.array([10, 20, 30]).reshape(3, 1, 1, 1))
    masks = ArrayTensor(np.array([2, 7]).reshape(2, 1, 1, 1))
    latent = {"samples": source, "noise_mask": masks, "batch_index": [11, 19, 23], "note": "keep"}
    result = cast(
        dict[str, Any],
        LatentFromBatch.execute(samples=latent, batch_index=index, length=2)["LATENT"],
    )
    assert result["samples"].array.flatten().tolist() == expected
    expected_indices = {10: 11, 20: 19, 30: 23}
    assert result["batch_index"] == [expected_indices[value] for value in expected]
    expected_masks = {10: 2, 20: 7, 30: 2}
    assert result["noise_mask"].array.flatten().tolist() == [
        expected_masks[value] for value in expected
    ]
    result["samples"].array.fill(-1)
    result["noise_mask"].array.fill(-1)
    assert source.array.flatten().tolist() == [10, 20, 30]
    assert masks.array.flatten().tolist() == [2, 7]
    assert result["note"] == "keep"


def test_repeat_offsets_nonconsecutive_indices_and_repeats_original_masks() -> None:
    source = ArrayTensor(np.array([10, 20, 30]).reshape(3, 1, 1, 1))
    masks = ArrayTensor(np.array([2, 7]).reshape(2, 1, 1, 1))
    latent = {"samples": source, "noise_mask": masks, "batch_index": [4, 9, 6]}
    result = cast(dict[str, Any], RepeatLatentBatch.execute(samples=latent, amount=3)["LATENT"])
    assert result["samples"].array.flatten().tolist() == [10, 20, 30] * 3
    assert result["noise_mask"].array.flatten().tolist() == [2, 7] * 3
    assert result["batch_index"] == [4, 9, 6, 10, 15, 12, 16, 21, 18]
    assert latent["batch_index"] == [4, 9, 6]


def test_mask_shape_is_batch_channel_height_width_without_changing_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=ArrayTensor))
    samples = object()
    mask = np.arange(24).reshape(2, 3, 4)
    latent = {"samples": samples, "note": "preserved"}
    result = cast(dict[str, Any], SetLatentNoiseMask.execute(samples=latent, mask=mask)["LATENT"])
    assert result["samples"] is samples
    assert result["note"] == "preserved"
    assert isinstance(result["noise_mask"], ArrayTensor)
    np.testing.assert_array_equal(result["noise_mask"].array, mask[:, None, :, :])
    assert "noise_mask" not in latent


def test_zero_size_upscale_returns_the_existing_latent_unchanged() -> None:
    latent = {"samples": object(), "note": "preserved"}
    assert (
        LatentUpscale.execute(
            samples=latent, upscale_method="bislerp", width=0, height=0, crop="center"
        )["LATENT"]
        is latent
    )


def test_latent_composite_clips_and_copies_while_blend_weights_the_first_input() -> None:
    target = ArrayTensor(np.arange(20).reshape(1, 1, 4, 5).astype(np.float32))
    source = ArrayTensor(np.full((1, 1, 3, 4), -3.0, dtype=np.float32))
    result = cast(
        dict[str, Any],
        LatentComposite.execute(
            samples_to={"samples": target, "note": "keep"},
            samples_from={"samples": source},
            x=24,
            y=16,
        )["LATENT"],
    )
    np.testing.assert_array_equal(
        result["samples"].array,
        np.array(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, -3, -3], [15, 16, 17, -3, -3]]
        ).reshape(1, 1, 4, 5),
    )
    assert result["note"] == "keep"
    np.testing.assert_array_equal(target.array.flatten(), np.arange(20))
    blend = cast(
        dict[str, Any],
        LatentBlend.execute(
            samples1={"samples": target, "note": "keep"},
            samples2={"samples": ArrayTensor(np.full((1, 1, 4, 5), 40.0))},
            blend_factor=0.25,
        )["LATENT"],
    )
    np.testing.assert_array_equal(blend["samples"].array.flatten(), 30 + np.arange(20) / 4)
    assert blend["note"] == "keep"


def test_latent_crop_clamps_offset_to_leave_eight_cells() -> None:
    source = ArrayTensor(np.arange(130).reshape(1, 1, 10, 13))
    result = cast(
        dict[str, Any],
        LatentCrop.execute(
            samples={"samples": source, "note": "keep"}, width=64, height=64, x=800, y=800
        )["LATENT"],
    )
    np.testing.assert_array_equal(
        result["samples"].array, np.arange(130).reshape(1, 1, 10, 13)[:, :, 2:10, 5:13]
    )
    assert result["note"] == "keep"
