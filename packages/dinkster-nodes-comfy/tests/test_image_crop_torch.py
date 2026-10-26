"""Public CPU behavior for native image crop primitives."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.image_crop import ImageCropV2, PrimitiveBoundingBox
from dinkster_schema import use_reporter


def _image() -> torch.Tensor:
    return torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5, 1) / 30


def test_image_crop_v2_uses_asymmetric_bounds_and_reports_the_cropped_preview() -> None:
    image = _image()
    reports: list[tuple[str, dict[str, object], bytes | None]] = []

    with use_reporter(lambda name, data, blob: reports.append((name, dict(data), blob))):
        result = ImageCropV2.execute(
            image=image, crop_region={"x": 3, "y": 1, "width": 2, "height": 1}
        )

    assert torch.equal(result["IMAGE"], image[:, 1:2, 3:5, :])
    assert len(reports) == 1
    name, data, blob = reports[0]
    assert (name, data) == ("preview", {"mime": "image/png", "width": 2, "height": 1})
    assert blob is not None and blob.startswith(b"\x89PNG\r\n\x1a\n")


def test_image_crop_v2_uses_region_defaults_and_clamps_only_the_origin() -> None:
    image = _image()

    defaults = ImageCropV2.execute(image=image, crop_region={})["IMAGE"]
    clamped = ImageCropV2.execute(
        image=image, crop_region={"x": 99, "y": 99, "width": 10, "height": 10}
    )["IMAGE"]

    assert torch.equal(defaults, image)
    assert torch.equal(clamped, image[:, 2:3, 4:5, :])


def test_primitive_bounding_box_returns_input_values_without_normalization() -> None:
    assert PrimitiveBoundingBox.execute(x=3, y=1, width=2, height=1) == {
        "BOUNDING_BOX": {"x": 3, "y": 1, "width": 2, "height": 1}
    }
