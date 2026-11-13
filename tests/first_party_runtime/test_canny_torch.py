"""CPU pixels and ordinary-manifest transport for native Canny edges."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_image.ported.image_filters import Canny


def image() -> np.ndarray[Any, Any]:
    pixels = np.zeros((2, 13, 15, 4), dtype=np.float64)
    pixels[0, :, 7:, :3] = 1.0
    pixels[1, 5:, :, :3] = 1.0
    pixels[..., 3] = np.arange(13 * 15).reshape(13, 15) % 2
    return pixels


def expected() -> torch.Tensor:
    pixels = torch.zeros((2, 13, 15, 3), dtype=torch.float32)
    pixels[0, :, 6, :] = 1.0
    pixels[1, 4, :, :] = 1.0
    return pixels


def test_canny_discards_alpha_converts_dtype_and_preserves_source() -> None:
    pixels = torch.from_numpy(image())
    before = pixels.clone()
    output = Canny.execute(image=pixels, low_threshold=0.2, high_threshold=0.6)["IMAGE"]
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert torch.equal(output, expected())
    assert torch.equal(pixels, before)
    constant_rgb = pixels.clone()
    constant_rgb[..., :3] = 0
    assert (
        torch.count_nonzero(
            Canny.execute(image=constant_rgb, low_threshold=0.1, high_threshold=0.9)["IMAGE"]
        )
        == 0
    )


class ImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="canny-fixture.image",
            inputs=(),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(IMAGE=image())


class Result(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="canny-fixture.result",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("pixels", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        pixels = materialize_image(image)
        assert pixels.dtype == torch.float32
        assert torch.equal(pixels, expected())
        return cls.outputs(pixels=int(torch.count_nonzero(pixels)))


FIXTURE_NODES = (ImageSource, Result)
