"""Native image morphology and RGB/YUV conversion nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_INT,
    IMAGE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_OPERATIONS = ("erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat")


class Morphology(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Morphology",
            display_name="Apply Morphology",
            category="image/filters",
            search_terms=("erode", "dilate"),
            inputs=(
                InputSpec("image", _IMAGE),
                InputSpec(
                    "operation",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(tuple(ComboOption(value) for value in _OPERATIONS)),
                ),
                InputSpec(
                    "kernel_size",
                    TypeExpr.concrete(CORE_INT),
                    default=3,
                    widget=NumberWidget(min=3, max=999, step=1),
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: Any, operation: str, kernel_size: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        morphology = importlib.import_module("kornia.morphology")
        management = importlib.import_module("dinkster_inference.model_management")
        image = materialize_image(image)
        device = management.get_torch_device()
        kernel = torch.ones(kernel_size, kernel_size, device=device)
        image_k = image.to(device).movedim(-1, 1)
        functions = {
            "erode": morphology.erosion,
            "dilate": morphology.dilation,
            "open": morphology.opening,
            "close": morphology.closing,
            "gradient": morphology.gradient,
            "top_hat": morphology.top_hat,
            "bottom_hat": morphology.bottom_hat,
        }
        if operation not in functions:
            raise ValueError(
                f"Invalid operation {operation} for morphology. Must be one of 'erode', "
                "'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'"
            )
        output = functions[operation](image_k, kernel)
        return cls.outputs(IMAGE=output.to(management.intermediate_device()).movedim(1, -1))


class ImageRGBToYUV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageRGBToYUV",
            display_name="Image RGB to YUV",
            category="image/color",
            search_terms=("color space conversion",),
            inputs=(InputSpec("image", _IMAGE),),
            outputs=tuple(OutputSpec(name, _IMAGE) for name in ("Y", "U", "V")),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        color = importlib.import_module("kornia.color")
        image = materialize_image(image)[..., :3]
        out = color.rgb_to_ycbcr(image.movedim(-1, 1)).movedim(1, -1)
        return cls.outputs(
            Y=out[..., 0:1].expand_as(image),
            U=out[..., 1:2].expand_as(image),
            V=out[..., 2:3].expand_as(image),
        )


class ImageYUVToRGB(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageYUVToRGB",
            display_name="Image YUV to RGB",
            category="image/color",
            search_terms=("color space conversion",),
            inputs=tuple(InputSpec(name, _IMAGE) for name in ("Y", "U", "V")),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, Y: Any, U: Any, V: Any) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        color = importlib.import_module("kornia.color")
        image = torch.cat(
            [
                torch.mean(materialize_image(channel)[..., :3], dim=-1, keepdim=True)
                for channel in (Y, U, V)
            ],
            dim=-1,
        )
        out = color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1)
        return cls.outputs(IMAGE=out)


MORPHOLOGY_NODES = (Morphology, ImageRGBToYUV, ImageYUVToRGB)
