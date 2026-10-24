"""CPU behavior for native dataset collection nodes."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.dataset_collections import (
    ImageDeduplication,
    ImageGrid,
    MergeTextLists,
    ShuffleDataset,
    ShuffleImageTextDataset,
)


def _images() -> list[torch.Tensor]:
    return [torch.full((2, 3, 4, 3), value / 10, dtype=torch.float32) for value in (1, 2, 3)]


def test_shuffle_flattens_batches_with_reference_numpy_order() -> None:
    images = _images()
    result = ShuffleDataset.execute(images=images, seed=[0])["images"]
    assert [float(image[0, 0, 0, 0]) for image in result] == [
        0.30000001192092896,
        0.20000000298023224,
        0.10000000149011612,
        0.20000000298023224,
        0.10000000149011612,
        0.30000001192092896,
    ]


def test_shuffle_image_text_preserves_pairing() -> None:
    images = [torch.full((1, 2, 2, 3), value, dtype=torch.float32) for value in (1, 2, 3)]
    result = ShuffleImageTextDataset.execute(images=images, texts=["one", "two", "three"], seed=[0])
    assert [float(image[0, 0, 0, 0]) for image in result["images"]] == [3, 2, 1]
    assert result["texts"] == ["three", "two", "one"]


def test_deduplication_uses_the_reference_threshold_boundary() -> None:
    image = torch.arange(3 * 9 * 11, dtype=torch.float32).reshape(1, 9, 11, 3) / (3 * 9 * 11)
    altered = image.clone()
    altered[..., 0] = 1.0 - altered[..., 0]
    result = ImageDeduplication.execute(
        images=[image, image.clone(), altered], similarity_threshold=[0.95]
    )
    assert len(result["images"]) == 1
    assert ImageDeduplication.execute(images=[], similarity_threshold=[0.95]) == {"images": []}


def test_image_grid_quantizes_and_pads_like_the_reference() -> None:
    red = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    blue = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    result = ImageGrid.execute(
        images=[red, blue], columns=[2], cell_width=[2], cell_height=[2], padding=[1]
    )["images"]
    assert tuple(result.shape) == (1, 2, 5, 3)
    assert torch.equal(result[0, :, :2], torch.tensor([[[1.0, 0.0, 0.0]]]).expand(2, 2, 3))
    assert torch.equal(result[0, :, 2], torch.zeros((2, 3)))
    assert torch.equal(result[0, :, 3:], torch.tensor([[[0.0, 0.0, 1.0]]]).expand(2, 2, 3))
    try:
        ImageGrid.execute(
            images=[torch.empty((0, 1, 1, 3))],
            columns=[2],
            cell_width=[2],
            cell_height=[2],
            padding=[1],
        )
    except ValueError as error:
        assert str(error) == "Cannot create grid from empty image list"
    else:
        raise AssertionError("Empty image batches must not produce a grid")


def test_merge_text_lists_preserves_the_reference_nested_payload() -> None:
    assert MergeTextLists.execute(texts=["one", "two"]) == {"texts": [["one", "two"]]}
