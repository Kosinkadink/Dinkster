"""Native ComfyUI image crop and bounding-box primitive nodes."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
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


def _crop_image_with_mask(
    image: Any,
    mask: Any,
    max_size: int,
    pad_factor: float,
    mask_offset: int,
    background: tuple[float, float, float],
    aspect_ratio: float,
) -> Any:
    torch = importlib.import_module("torch")
    utils = importlib.import_module("dinkster_inference.utils")
    image = image.permute(2, 0, 1).unsqueeze(0).cpu().float().clamp(0, 1)
    mask = mask.unsqueeze(0).unsqueeze(0).cpu().float().clamp(0, 1)
    plane = mask[0, 0]
    height, width = plane.shape
    border = torch.cat([plane[0, :], plane[-1, :], plane[:, 0], plane[:, -1]])
    center = plane[height // 4 : height - height // 4, width // 4 : width - width // 4]
    if float(border.mean()) > 0.5 and float(center.mean()) < 0.5:
        mask = 1.0 - mask
    if mask_offset > 0:
        mask = torch.nn.functional.max_pool2d(
            mask, kernel_size=2 * mask_offset + 1, stride=1, padding=mask_offset
        )
    elif mask_offset < 0:
        radius = -mask_offset
        mask = 1.0 - torch.nn.functional.max_pool2d(
            1.0 - mask, kernel_size=2 * radius + 1, stride=1, padding=radius
        )
    mask = torch.where(mask < 0.05, torch.zeros_like(mask), mask)
    height, width = image.shape[-2:]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        width, height = int(width * scale), int(height * scale)
        image = utils.common_upscale(image, width, height, "lanczos", "disabled")
        mask = utils.common_upscale(mask, width, height, "lanczos", "disabled")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    alpha = (mask[0, 0].clamp(0, 1) * 255.0).to(torch.uint8)
    foreground = (alpha > 204).nonzero()
    if foreground.numel() == 0:
        inverted = ((255 - alpha) > 204).nonzero()
        if inverted.numel() > 0:
            logging.info("Mask bounding box is empty; using the inverted mask.")
            mask = 1.0 - mask
            foreground = inverted
    if foreground.numel() > 0:
        y_min, x_min = foreground.min(dim=0).values.tolist()
        y_max, x_max = foreground.max(dim=0).values.tolist()
        center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
        box_width, box_height = x_max - x_min + 1, y_max - y_min + 1
        if box_width / max(box_height, 1) >= aspect_ratio:
            crop_width = int(box_width * pad_factor)
            crop_height = int(box_width / aspect_ratio * pad_factor)
        else:
            crop_height = int(box_height * pad_factor)
            crop_width = int(box_height * aspect_ratio * pad_factor)
        half_width, half_height = math.ceil(crop_width / 2), math.ceil(crop_height / 2)
        x1, y1 = int(center_x - half_width), int(center_y - half_height)
        x2, y2 = x1 + 2 * half_width, y1 + 2 * half_height
    else:
        logging.warning(
            "Mask for the image is empty; a clean foreground mask is required for best quality."
        )
        x1, y1, x2, y2 = 0, 0, width, height
    left, top = max(0, -x1), max(0, -y1)
    right, bottom = max(0, x2 - width), max(0, y2 - height)
    if left or top or right or bottom:
        image = torch.nn.functional.pad(image, (left, right, top, bottom), value=0.0)
        mask = torch.nn.functional.pad(mask, (left, right, top, bottom), value=0.0)
        x1, x2, y1, y2 = x1 + left, x2 + left, y1 + top, y2 + top
    image, mask = image[..., y1:y2, x1:x2], mask[..., y1:y2, x1:x2]
    fill = torch.tensor(background, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image * mask + fill * (1.0 - mask)).clamp(0, 1)


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
                    TypeExpr.concrete(CORE_STRING),
                    default="#000000",
                    widget=ColorWidget(),
                    doc="Background color behind the masked subject.",
                ),
            ),
            outputs=(OutputSpec("images", _IMAGE),),
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
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        color = background.lstrip("#")
        rgb = (
            tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
            if len(color) == 6
            else (0.0, 0.0, 0.0)
        )
        images, masks = materialize_image(images)[..., :3], materialize_image(masks)
        batch = images.shape[0]
        if masks.shape[0] == 1 and batch > 1:
            masks = masks.expand(batch, -1, -1)
        elif masks.shape[0] != batch:
            raise ValueError(f"Mask batch {masks.shape[0]} does not match image batch {batch}")
        if masks.shape[-2:] != images.shape[1:3]:
            masks = utils.common_upscale(
                masks.unsqueeze(1).float(), images.shape[2], images.shape[1], "bilinear", "disabled"
            ).squeeze(1)
        output: list[Any] = []
        for index in range(batch):
            composite = _crop_image_with_mask(
                images[index],
                masks[index],
                max(width, height),
                pad_factor,
                grow_mask,
                (rgb[0], rgb[1], rgb[2]),
                width / height,
            )
            composite = utils.common_upscale(composite, width, height, "lanczos", "disabled")
            output.append(composite.movedim(-3, -1))
        return cls.outputs(
            images=torch.cat(output, dim=0).to(
                device=management.intermediate_device(), dtype=management.intermediate_dtype()
            )
        )
