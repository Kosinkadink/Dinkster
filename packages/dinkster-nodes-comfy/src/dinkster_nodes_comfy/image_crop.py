"""Native ComfyUI image crop and bounding-box primitive nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    render_image_png,
    report_preview,
)

_BOUNDING_BOX = TypeExpr.concrete(BOUNDING_BOX_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 16_384


class ImageCropV2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageCropV2",
            display_name="Crop Image",
            category="image/transform",
            description="Crop an image to the specified dimensions.",
            inputs=(
                InputSpec("image", _IMAGE),
                InputSpec(
                    "crop_region",
                    _BOUNDING_BOX,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                    widget=CustomWidgetDescriptor("BOUNDING_BOX", {"component": "ImageCrop"}),
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE, preview=True),),
            emits_previews=True,
            search_terms=("crop", "cut", "trim"),
        )

    @classmethod
    def execute(cls, *, image: Any, crop_region: Mapping[str, int]) -> Mapping[str, object]:
        x = min(crop_region.get("x", 0), image.shape[2] - 1)
        y = min(crop_region.get("y", 0), image.shape[1] - 1)
        result = image[
            :, y : y + crop_region.get("height", 512), x : x + crop_region.get("width", 512), :
        ]
        report_preview(
            render_image_png(result),
            mime="image/png",
            width=int(result.shape[2]),
            height=int(result.shape[1]),
        )
        return cls.outputs(IMAGE=result)


class PrimitiveBoundingBox(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PrimitiveBoundingBox",
            display_name="Bounding Box",
            category="utilities/primitive",
            inputs=(
                InputSpec("x", _INT, default=0, widget=NumberWidget(min=0, max=_MAX_RESOLUTION)),
                InputSpec("y", _INT, default=0, widget=NumberWidget(min=0, max=_MAX_RESOLUTION)),
                InputSpec(
                    "width", _INT, default=512, widget=NumberWidget(min=1, max=_MAX_RESOLUTION)
                ),
                InputSpec(
                    "height", _INT, default=512, widget=NumberWidget(min=1, max=_MAX_RESOLUTION)
                ),
            ),
            outputs=(OutputSpec("BOUNDING_BOX", _BOUNDING_BOX),),
        )

    @classmethod
    def execute(cls, *, x: int, y: int, width: int, height: int) -> Mapping[str, object]:
        return cls.outputs(BOUNDING_BOX={"x": x, "y": y, "width": width, "height": height})
