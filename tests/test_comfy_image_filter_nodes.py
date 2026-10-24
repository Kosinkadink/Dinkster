"""Image filter schemas and CPU behavior match the fixed ComfyUI reference."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.image_filters import (
    IMAGE_FILTER_NODES,
    ImageBlend,
    ImageBlur,
    ImageQuantize,
    ImageScaleToTotalPixels,
    ImageSharpen,
)


def test_image_filter_schemas_match_the_fixed_reference_contract() -> None:
    expected = (
        (
            "ImageBlend",
            "Blend Images",
            "image/filters",
            ("image1", "image2", "blend_factor", "blend_mode"),
        ),
        ("ImageBlur", "Blur Image", "image/filters", ("image", "blur_radius", "sigma")),
        ("ImageQuantize", "Quantize Image", "image/filters", ("image", "colors", "dither")),
        (
            "ImageSharpen",
            "Sharpen Image",
            "image/filters",
            ("image", "sharpen_radius", "sigma", "alpha"),
        ),
        (
            "ImageScaleToTotalPixels",
            "Scale Image to Total Pixels",
            "image/upscaling",
            ("image", "upscale_method", "megapixels", "resolution_steps"),
        ),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            node.schema().category,
            tuple(port.id for port in node.schema().inputs),
        )
        for node in IMAGE_FILTER_NODES
    )
    assert actual == expected
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_FILTER_NODES}
    assert schemas["ImageBlend"].inputs[2].default == 0.5
    assert schemas["ImageSharpen"].inputs[1].advanced is True
    assert schemas["ImageScaleToTotalPixels"].inputs[3].advanced is True
    for schema in schemas.values():
        assert schema_to_wire(schema)


def test_zero_radius_returns_original_live_value() -> None:
    image = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    assert ImageBlur.execute(image=image, blur_radius=0, sigma=1.0)["IMAGE"] is image
    assert (
        ImageSharpen.execute(image=image, sharpen_radius=0, sigma=1.0, alpha=1.0)["IMAGE"] is image
    )


@pytest.fixture
def torch_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    torch: Any = pytest.importorskip("torch")
    utils = SimpleNamespace(
        common_upscale=lambda samples, width, height, method, crop: torch.nn.functional.interpolate(
            samples, size=(height, width), mode="nearest"
        )
    )
    management = SimpleNamespace(get_torch_device=lambda: "cpu", intermediate_device=lambda: "cpu")
    monkeypatch.setitem(sys.modules, "dinkster_inference.utils", utils)
    monkeypatch.setitem(sys.modules, "dinkster_inference.model_management", management)
    return torch


def test_blend_modes_preserve_destination_alpha_and_source_values(torch_runtime: Any) -> None:
    torch: Any = torch_runtime
    image1 = torch.tensor([[[[0.2, 0.8, 0.4, 0.25]]]], dtype=torch.float32)
    image2 = torch.tensor([[[[0.7, 0.3, 0.6]]]], dtype=torch.float32)
    original1, original2 = image1.clone(), image2.clone()
    for mode in ("normal", "multiply", "screen", "overlay", "soft_light", "difference"):
        result: Any = ImageBlend.execute(
            image1=image1, image2=image2, blend_factor=1.0, blend_mode=mode
        )["IMAGE"]
        assert result.shape == image1.shape
        assert result[..., 3].equal(image1[..., 3])
    assert image1.equal(original1)
    assert image2.equal(original2)


def test_blur_sharpen_quantize_and_scaling_cpu_outputs(torch_runtime: Any) -> None:
    torch: Any = torch_runtime
    image = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.9, 0.1], [0.3, 0.4, 0.5, 0.2], [0.8, 0.6, 0.1, 0.3]],
                [[0.1, 0.9, 0.3, 0.4], [0.5, 0.7, 0.2, 0.5], [0.9, 0.0, 0.8, 0.6]],
                [[0.7, 0.1, 0.4, 0.7], [0.2, 0.8, 0.6, 0.8], [0.4, 0.3, 0.5, 0.9]],
            ]
        ],
        dtype=torch.float32,
    )
    original = image.clone()
    blurred: Any = ImageBlur.execute(image=image, blur_radius=1, sigma=1.0)["IMAGE"]
    sharpened: Any = ImageSharpen.execute(image=image, sharpen_radius=1, sigma=1.0, alpha=1.0)[
        "IMAGE"
    ]
    assert blurred.shape == image.shape
    assert sharpened.shape == image.shape
    for dither in ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"):
        result: Any = ImageQuantize.execute(image=image, colors=4, dither=dither)["IMAGE"]
        assert result.dtype == image.dtype
        assert result.device == image.device
        assert result[..., 3].equal(image[..., 3])
    scaled: Any = ImageScaleToTotalPixels.execute(
        image=image, upscale_method="nearest-exact", megapixels=0.01, resolution_steps=8
    )["IMAGE"]
    assert scaled.shape[1:3] == (104, 104)
    assert image.equal(original)
