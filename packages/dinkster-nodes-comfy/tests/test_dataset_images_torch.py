"""CPU behavior for native ComfyUI dataset image transform nodes."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.dataset_images import (
    AdjustBrightness,
    AdjustContrast,
    CenterCropImages,
    NormalizeImages,
    RandomCropImages,
    ResizeImagesByLongerEdge,
    ResizeImagesByShorterEdge,
)


def test_spatial_dataset_transforms_quantize_and_process_each_frame() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 5 * 4).reshape(1, 3, 5, 4)
    images = image.repeat((2, 1, 1, 1))
    before = images.clone()

    shorter = ResizeImagesByShorterEdge.execute(images=images, shorter_edge=4)["images"]
    longer = ResizeImagesByLongerEdge.execute(images=images, longer_edge=4)["images"]
    center = CenterCropImages.execute(images=images, width=9, height=2)["images"]
    random = RandomCropImages.execute(images=images, width=3, height=2, seed=511)["images"]

    assert shorter.shape == (2, 4, 6, 4)
    assert longer.shape == (2, 2, 4, 4)
    assert center.shape == (2, 2, 5, 4)
    assert random.shape == (2, 2, 3, 4)
    torch.testing.assert_close(random[0], random[1])
    torch.testing.assert_close(images, before)


def test_tensor_dataset_transforms_preserve_alpha_without_mutating_input() -> None:
    images = torch.tensor([[[[0.2, 0.4, 0.6, 0.3], [0.8, 0.1, 0.5, 0.9]]]], dtype=torch.float32)
    before = images.clone()

    normalized = NormalizeImages.execute(images=images, mean=0.5, std=0.5)["images"]
    brightness = AdjustBrightness.execute(images=images, factor=0.5)["images"]
    contrast = AdjustContrast.execute(images=images, factor=2.0)["images"]

    torch.testing.assert_close(normalized[..., :3], (images[..., :3] - 0.5) / 0.5)
    torch.testing.assert_close(brightness[..., :3], images[..., :3] * 0.5)
    torch.testing.assert_close(
        contrast[..., :3], ((images[..., :3] - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0)
    )
    for result in (normalized, brightness, contrast):
        torch.testing.assert_close(result[..., 3], images[..., 3])
    torch.testing.assert_close(images, before)
