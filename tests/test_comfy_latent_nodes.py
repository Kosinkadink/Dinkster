"""Core latent node interfaces and metadata behavior stay exact."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_comfy.latent import (
    LATENT_NODES,
    LatentFromBatch,
    LatentUpscale,
    RepeatLatentBatch,
    SetLatentNoiseMask,
)


@pytest.mark.parametrize(
    ("node_id", "display", "category", "inputs"),
    [
        ("EmptyLatentImage", "Empty Latent Image", "", ("width", "height", "batch_size")),
        (
            "LatentFromBatch",
            "Get Latent From Batch",
            "/batch",
            ("samples", "batch_index", "length"),
        ),
        ("RepeatLatentBatch", "Repeat Latent Batch", "/batch", ("samples", "amount")),
        (
            "LatentUpscale",
            "Upscale Latent",
            "",
            ("samples", "upscale_method", "width", "height", "crop"),
        ),
        ("LatentUpscaleBy", "Upscale Latent By", "", ("samples", "upscale_method", "scale_by")),
        ("LatentRotate", "Rotate Latent", "/transform", ("samples", "rotation")),
        ("LatentFlip", "Flip Latent", "/transform", ("samples", "flip_method")),
        ("SetLatentNoiseMask", "Set Latent Noise Mask", "", ("samples", "mask")),
    ],
)
def test_latent_schemas_use_exact_core_interfaces(
    node_id: str, display: str, category: str, inputs: tuple[str, ...]
) -> None:
    node = next(node for node in LATENT_NODES if node.schema().node_type == node_id)
    schema = node.schema()
    assert schema.display_name == display
    assert schema.category == "model/latent" + category
    assert tuple(spec.id for spec in schema.inputs) == inputs
    assert tuple((spec.id, spec.type.types) for spec in schema.outputs) == (
        ("LATENT", ("LATENT",)),
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

    def clone(self) -> ArrayTensor:
        return ArrayTensor(self.array.copy())

    def repeat(self, *shape: Any) -> ArrayTensor:
        dimensions = shape[0] if len(shape) == 1 else shape
        return ArrayTensor(np.tile(self.array, dimensions))


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


def test_mask_shape_is_batch_channel_height_width_without_changing_samples() -> None:
    samples = object()
    mask = np.arange(24).reshape(2, 3, 4)
    latent = {"samples": samples, "note": "preserved"}
    result = cast(dict[str, Any], SetLatentNoiseMask.execute(samples=latent, mask=mask)["LATENT"])
    assert result["samples"] is samples
    assert result["note"] == "preserved"
    np.testing.assert_array_equal(result["noise_mask"], mask[:, None, :, :])
    assert "noise_mask" not in latent


def test_zero_size_upscale_returns_the_existing_latent_unchanged() -> None:
    latent = {"samples": object(), "note": "preserved"}
    assert (
        LatentUpscale.execute(
            samples=latent, upscale_method="bislerp", width=0, height=0, crop="center"
        )["LATENT"]
        is latent
    )
