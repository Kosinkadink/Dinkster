"""Image filter schemas and CPU behavior match the fixed ComfyUI reference."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dinkster_image.ported.image_filters import (
    ImageBlend,
    ImageBlur,
    ImageSharpen,
)


def test_zero_radius_returns_original_live_value() -> None:
    image = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    assert ImageBlur.execute(image=image, blur_radius=0, sigma=1.0)["IMAGE"] is image
    assert (
        ImageSharpen.execute(image=image, sharpen_radius=0, sigma=1.0, alpha=1.0)["IMAGE"] is image
    )


class ArrayTensor(np.ndarray[Any, Any]):
    storage_device = "accelerator"

    def detach(self) -> ArrayTensor:
        return self

    def to(self, device: str | None = None, dtype: Any = None) -> ArrayTensor:
        result = self.astype(dtype or self.dtype).view(ArrayTensor)
        result.storage_device = device or self.storage_device
        return result

    def numpy(self) -> np.ndarray[Any, Any]:
        if self.storage_device != "cpu":
            raise RuntimeError("NumPy requires CPU storage")
        return np.asarray(self)

    def float(self) -> ArrayTensor:
        return self.astype(np.float32).view(ArrayTensor)


@pytest.fixture
def array_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = SimpleNamespace(
        as_tensor=lambda value: np.asarray(value).view(ArrayTensor),
        zeros_like=np.zeros_like,
        tensor=lambda value: np.asarray(value).view(ArrayTensor),
        uint8=np.uint8,
        cat=lambda values, dim: np.concatenate(values, axis=dim).view(ArrayTensor),
        where=np.where,
        sqrt=np.sqrt,
        clamp=np.clip,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "dinkster_inference.utils", SimpleNamespace())


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("normal", [0.7, 0.3, 0.6]),
        ("multiply", [0.14, 0.24, 0.24]),
        ("screen", [0.76, 0.86, 0.76]),
        ("overlay", [0.28, 0.72, 0.48]),
        ("difference", [0.0, 0.5, 0.0]),
    ],
)
def test_blend_pixels_preserve_destination_alpha(
    array_runtime: None, mode: str, expected: list[float]
) -> None:
    image1 = np.array([[[[0.2, 0.8, 0.4, 0.25]]]], dtype=np.float32)
    image2 = np.array([[[[0.7, 0.3, 0.6, 0.9]]]], dtype=np.float32)
    original1, original2 = image1.copy(), image2.copy()
    result: Any = ImageBlend.execute(
        image1=image1, image2=image2, blend_factor=1.0, blend_mode=mode
    )["IMAGE"]
    np.testing.assert_allclose(result, [[[[*expected, 0.25]]]], atol=1e-7)
    np.testing.assert_array_equal(image1, original1)
    np.testing.assert_array_equal(image2, original2)
