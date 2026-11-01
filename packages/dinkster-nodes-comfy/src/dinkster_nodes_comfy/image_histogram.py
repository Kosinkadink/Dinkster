"""Native first-frame RGB and luminance histogram computation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    HISTOGRAM_TYPE,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)


class ImageHistogram(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageHistogram",
            display_name="Image Histogram",
            category="utilities",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(HISTOGRAM_TYPE))
                for name in ("rgb", "luminance", "red", "green", "blue")
            ),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        np = importlib.import_module("numpy")
        img = materialize_image(image)[0].cpu().numpy()
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

        def bincount(data: Any) -> Any:
            return np.bincount(data.ravel(), minlength=256)[:256]

        red = bincount(img_uint8[:, :, 0])
        green = bincount(img_uint8[:, :, 1])
        blue = bincount(img_uint8[:, :, 2])
        rgb = ((red + green + blue) // 3).tolist()
        luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        return cls.outputs(
            rgb=rgb,
            luminance=bincount(np.clip(luminance * 255, 0, 255).astype(np.uint8)).tolist(),
            red=red.tolist(),
            green=green.tolist(),
            blue=blue.tolist(),
        )
