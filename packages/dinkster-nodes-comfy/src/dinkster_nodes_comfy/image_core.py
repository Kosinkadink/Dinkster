"""Core image creation, resizing, batching, color and outpainting nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MASK_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")


def _integer(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    step: int | None = None,
    *,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_INT),
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo(name: str, options: tuple[str, ...]) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_COMBO),
        widget=ComboWidget(tuple(ComboOption(value) for value in options)),
    )


def _schema(
    node_type: str, display_name: str, category: str, inputs: tuple[InputSpec, ...]
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=(OutputSpec("IMAGE", _IMAGE),),
    )


class ImageScale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageScale",
            "Upscale Image",
            "image/upscaling",
            (
                InputSpec("image", _IMAGE),
                _combo("upscale_method", _UPSCALE_METHODS),
                _integer("width", 512, 0, 16384, 1),
                _integer("height", 512, 0, 16384, 1),
                _combo("crop", ("disabled", "center")),
            ),
        )

    @classmethod
    def execute(
        cls, *, image: Any, upscale_method: str, width: int, height: int, crop: str
    ) -> Mapping[str, object]:
        if width == 0 and height == 0:
            return cls.outputs(IMAGE=image)
        utils = importlib.import_module("dinkster_inference.utils")
        samples = materialize_image(image).movedim(-1, 1)
        if width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))
        output = utils.common_upscale(samples, width, height, upscale_method, crop).movedim(1, -1)
        return cls.outputs(IMAGE=output)


class ImageScaleBy(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageScaleBy",
            "Upscale Image By",
            "image/upscaling",
            (
                InputSpec("image", _IMAGE),
                _combo("upscale_method", _UPSCALE_METHODS),
                InputSpec(
                    "scale_by",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.01, max=8.0, step=0.01),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, upscale_method: str, scale_by: float) -> Mapping[str, object]:
        utils = importlib.import_module("dinkster_inference.utils")
        samples = materialize_image(image).movedim(-1, 1)
        width, height = round(samples.shape[3] * scale_by), round(samples.shape[2] * scale_by)
        output = utils.common_upscale(samples, width, height, upscale_method, "disabled").movedim(
            1, -1
        )
        return cls.outputs(IMAGE=output)


class ImageInvert(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageInvert", "Invert Image Colors", "image/color", (InputSpec("image", _IMAGE),)
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        output = 1.0 - image
        if image.shape[-1] == 4:
            output[..., 3] = image[..., 3]
        return cls.outputs(IMAGE=output)


class ImageBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageBatch",
            "Batch Images (DEPRECATED)",
            "image/batch",
            (InputSpec("image1", _IMAGE), InputSpec("image2", _IMAGE)),
        )

    @classmethod
    def execute(cls, *, image1: Any, image2: Any) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        image1, image2 = materialize_image(image1), materialize_image(image2)
        if image1.shape[-1] != image2.shape[-1]:
            if image1.shape[-1] > image2.shape[-1]:
                image2 = torch.nn.functional.pad(image2, (0, 1), mode="constant", value=1.0)
            else:
                image1 = torch.nn.functional.pad(image1, (0, 1), mode="constant", value=1.0)
        if image1.shape[1:] != image2.shape[1:]:
            image2 = utils.common_upscale(
                image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center"
            ).movedim(1, -1)
        return cls.outputs(IMAGE=torch.cat((image1, image2), dim=0))


class EmptyImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyImage",
            "Empty Image",
            "image",
            (
                _integer("width", 512, 1, 16384, 1),
                _integer("height", 512, 1, 16384, 1),
                _integer("batch_size", 1, 1, 4096),
                InputSpec(
                    "color",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=0xFFFFFF, step=1, display="color"),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, batch_size: int = 1, color: int = 0
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        dtype, device = management.intermediate_dtype(), management.intermediate_device()
        channels = [
            torch.full(
                [batch_size, height, width, 1],
                ((color >> shift) & 0xFF) / 0xFF,
                device=device,
                dtype=dtype,
            )
            for shift in (16, 8, 0)
        ]
        return cls.outputs(IMAGE=torch.cat(channels, dim=-1))


class ImagePadForOutpaint(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImagePadForOutpaint",
            display_name="Pad Image for Outpainting",
            category="image/transform",
            inputs=(
                InputSpec("image", _IMAGE),
                *(_integer(side, 0, 0, 16384, 8) for side in ("left", "top", "right", "bottom")),
                _integer("feathering", 40, 0, 16384, 1, advanced=True),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE), OutputSpec("MASK", TypeExpr.concrete(MASK_TYPE))),
        )

    @classmethod
    def execute(
        cls, *, image: Any, left: int, top: int, right: int, bottom: int, feathering: int
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        batch, height, width, channels = image.size()
        output = (
            torch.ones(
                (batch, height + top + bottom, width + left + right, channels), dtype=torch.float32
            )
            * 0.5
        )
        output[:, top : top + height, left : left + width, :] = image
        mask = torch.ones((height + top + bottom, width + left + right), dtype=torch.float32)
        interior = torch.zeros((height, width), dtype=torch.float32)
        if feathering > 0 and feathering * 2 < height and feathering * 2 < width:
            for y in range(height):
                for x in range(width):
                    distance = min(
                        y if top else height,
                        height - y if bottom else height,
                        x if left else width,
                        width - x if right else width,
                    )
                    if distance < feathering:
                        interior[y, x] = ((feathering - distance) / feathering) ** 2
        mask[top : top + height, left : left + width] = interior
        return cls.outputs(IMAGE=output, MASK=mask.unsqueeze(0))


IMAGE_CORE_NODES = (
    ImageScale,
    ImageScaleBy,
    ImageInvert,
    ImageBatch,
    EmptyImage,
    ImagePadForOutpaint,
)
