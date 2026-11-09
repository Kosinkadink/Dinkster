"""Native ComfyUI image crop and bounding-box primitive nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    COLOR_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MASK_TYPE,
    ColorWidget,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    render_image_png,
    report_preview,
)

_BOUNDING_BOX = TypeExpr.concrete(BOUNDING_BOX_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 16_384


class ImageCropToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageCropToMask",
            display_name="Crop Image to Mask",
            category="image/transform",
            search_terms=(
                "crop to mask",
                "mask crop",
                "crop mask",
                "mask crop resize",
                "crop mask resize",
                "trellis2",
                "pixal3d",
            ),
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec("masks", TypeExpr.concrete(MASK_TYPE)),
                InputSpec(
                    "width",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=64, max=4096, step=8),
                    doc="Output width in pixels.",
                ),
                InputSpec(
                    "height",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=64, max=4096, step=8),
                    doc="Output height in pixels.",
                ),
                InputSpec(
                    "pad_factor",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=1.0, max=2.0, step=0.01),
                    doc="Extra margin around the mask bounding box as a multiplier.",
                ),
                InputSpec(
                    "grow_mask",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=-32, max=32, step=1),
                    doc="Grow or shrink the mask by this many pixels before cropping.",
                ),
                InputSpec(
                    "background",
                    TypeExpr.concrete(COLOR_TYPE),
                    default="#000000",
                    widget=ColorWidget(),
                    doc="Background color behind the masked subject.",
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE, display_name="images"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        masks: Any,
        width: int,
        height: int,
        pad_factor: float,
        grow_mask: int,
        background: str,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        model_management = importlib.import_module("dinkster_inference.model_management")
        image_crop = importlib.import_module("dinkster_native.model3d.image_crop")

        result = image_crop.crop_images_to_masks(
            materialize_image(images),
            materialize_image(masks),
            width,
            height,
            pad_factor,
            grow_mask,
            background,
        ).to(
            device=model_management.intermediate_device(),
            dtype=model_management.intermediate_dtype(),
        )
        return cls.outputs(IMAGE=result)


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
