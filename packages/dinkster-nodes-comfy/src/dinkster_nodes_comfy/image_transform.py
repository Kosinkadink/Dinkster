"""Native ComfyUI image transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
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

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 16384


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
        _INT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo(
    name: str, options: tuple[str, ...], *, default: str | None = None, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        advanced=advanced,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
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


class ImageCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageCrop",
            "Crop Image (DEPRECATED)",
            "image/transform",
            (
                InputSpec("image", _IMAGE),
                _integer("width", 512, 1, _MAX_RESOLUTION, 1),
                _integer("height", 512, 1, _MAX_RESOLUTION, 1),
                _integer("x", 0, 0, _MAX_RESOLUTION, 1),
                _integer("y", 0, 0, _MAX_RESOLUTION, 1),
            ),
        )

    @classmethod
    def execute(
        cls, *, image: Any, width: int, height: int, x: int, y: int
    ) -> Mapping[str, object]:
        image = materialize_image(image)
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        return cls.outputs(IMAGE=image[:, y : y + height, x : x + width, :])


class RepeatImageBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "RepeatImageBatch",
            "Repeat Image Batch",
            "image/batch",
            (InputSpec("image", _IMAGE), _integer("amount", 1, 1, 4096)),
        )

    @classmethod
    def execute(cls, *, image: Any, amount: int) -> Mapping[str, object]:
        image = materialize_image(image)
        return cls.outputs(IMAGE=image.repeat((amount, 1, 1, 1)))


class ImageFromBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageFromBatch",
            "Get Image from Batch",
            "image/batch",
            (
                InputSpec("image", _IMAGE),
                _integer("batch_index", 0, -_MAX_RESOLUTION, _MAX_RESOLUTION),
                _integer("length", 1, 1, 4096),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, batch_index: int, length: int) -> Mapping[str, object]:
        image = materialize_image(image)
        if batch_index < 0:
            batch_index += image.shape[0]
        batch_index = max(0, min(image.shape[0] - 1, batch_index))
        length = min(image.shape[0] - batch_index, length)
        return cls.outputs(IMAGE=image[batch_index : batch_index + length].clone())


class ImageAddNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageAddNoise",
            "Add Noise to Image",
            "image/filters",
            (
                InputSpec("image", _IMAGE),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
                    ),
                ),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, seed: int, strength: float) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        generator = torch.manual_seed(seed)
        output = torch.clip(
            image
            + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image),
            min=0.0,
            max=1.0,
        )
        if image.shape[-1] == 4:
            output[..., 3] = image[..., 3]
        return cls.outputs(IMAGE=output)


class ImageStitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageStitch",
            "Stitch Images",
            "image/transform",
            (
                InputSpec("image1", _IMAGE),
                _combo("direction", ("right", "down", "left", "up"), default="right"),
                InputSpec("match_image_size", _BOOLEAN, default=True),
                _integer("spacing_width", 0, 0, 1024, 2, advanced=True),
                _combo(
                    "spacing_color",
                    ("white", "black", "red", "green", "blue"),
                    default="white",
                    advanced=True,
                ),
                InputSpec("image2", _IMAGE, required=False),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image1: Any,
        direction: str,
        match_image_size: bool,
        spacing_width: int,
        spacing_color: str,
        image2: Any = None,
    ) -> Mapping[str, object]:
        if image2 is None:
            return cls.outputs(IMAGE=image1)
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        image1 = materialize_image(image1)
        image2 = materialize_image(image2)
        if image1.shape[0] != image2.shape[0]:
            batch_size = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < batch_size:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(batch_size - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < batch_size:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(batch_size - image2.shape[0], 1, 1, 1)]
                )
        if match_image_size:
            height1, width1 = image1.shape[1:3]
            height2, width2 = image2.shape[1:3]
            aspect_ratio = width2 / height2
            if direction in ("left", "right"):
                target_height, target_width = height1, int(height1 * aspect_ratio)
            else:
                target_width, target_height = width1, int(width1 / aspect_ratio)
            image2 = utils.common_upscale(
                image2.movedim(-1, 1), target_width, target_height, "lanczos", "disabled"
            ).movedim(1, -1)
        colors: dict[str, float | tuple[float, float, float]] = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }
        color = colors[spacing_color]
        if not match_image_size:
            height1, width1 = image1.shape[1:3]
            height2, width2 = image2.shape[1:3]
            padding = color if isinstance(color, float) else 0.0
            if direction in ("left", "right") and height1 != height2:
                target_height = max(height1, height2)
                if height1 < target_height:
                    before = (target_height - height1) // 2
                    image1 = torch.nn.functional.pad(
                        image1,
                        (0, 0, 0, 0, before, target_height - height1 - before),
                        value=padding,
                    )
                if height2 < target_height:
                    before = (target_height - height2) // 2
                    image2 = torch.nn.functional.pad(
                        image2,
                        (0, 0, 0, 0, before, target_height - height2 - before),
                        value=padding,
                    )
            elif direction in ("up", "down") and width1 != width2:
                target_width = max(width1, width2)
                if width1 < target_width:
                    before = (target_width - width1) // 2
                    image1 = torch.nn.functional.pad(
                        image1, (0, 0, before, target_width - width1 - before), value=padding
                    )
                if width2 < target_width:
                    before = (target_width - width2) // 2
                    image2 = torch.nn.functional.pad(
                        image2, (0, 0, before, target_width - width2 - before), value=padding
                    )
        if image1.shape[-1] != image2.shape[-1]:
            channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1], channels - image1.shape[-1], device=image1.device
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1], channels - image2.shape[-1], device=image2.device
                        ),
                    ],
                    dim=-1,
                )
        images = [image2, image1] if direction in ("left", "up") else [image1, image2]
        if spacing_width > 0:
            spacing_width += spacing_width % 2
            shape = (
                image1.shape[0],
                max(image1.shape[1], image2.shape[1])
                if direction in ("left", "right")
                else spacing_width,
                spacing_width
                if direction in ("left", "right")
                else max(image1.shape[2], image2.shape[2]),
                image1.shape[-1],
            )
            spacing = torch.full(shape, 0.0, device=image1.device)
            if isinstance(color, tuple):
                for index, value in enumerate(color):
                    if index < spacing.shape[-1]:
                        spacing[..., index] = value
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0
            images.insert(1, spacing)
        return cls.outputs(IMAGE=torch.cat(images, dim=2 if direction in ("left", "right") else 1))


class ResizeAndPadImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ResizeAndPadImage",
            "Resize And Pad Image",
            "image/transform",
            (
                InputSpec("image", _IMAGE),
                _integer("target_width", 512, 1, _MAX_RESOLUTION, 1),
                _integer("target_height", 512, 1, _MAX_RESOLUTION, 1),
                _combo("padding_color", ("white", "black"), advanced=True),
                _combo(
                    "interpolation",
                    ("area", "bicubic", "nearest-exact", "bilinear", "lanczos"),
                    advanced=True,
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: Any,
        target_width: int,
        target_height: int,
        padding_color: str,
        interpolation: str,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(image)
        batch, height, width, channels = image.shape
        scale = min(target_width / width, target_height / height)
        width, height = int(width * scale), int(height * scale)
        resized = utils.common_upscale(
            image.permute(0, 3, 1, 2), width, height, interpolation, "disabled"
        )
        padded = torch.full(
            (batch, channels, target_height, target_width),
            0.0 if padding_color == "black" else 1.0,
            dtype=image.dtype,
            device=image.device,
        )
        top, left = (target_height - height) // 2, (target_width - width) // 2
        padded[:, :, top : top + height, left : left + width] = resized
        return cls.outputs(IMAGE=padded.permute(0, 2, 3, 1))


class ImageRotate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageRotate",
            "Rotate Image",
            "image/transform",
            (
                InputSpec("image", _IMAGE),
                _combo("rotation", ("none", "90 degrees", "180 degrees", "270 degrees")),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, rotation: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        turns = (
            1
            if rotation.startswith("90")
            else 2
            if rotation.startswith("180")
            else 3
            if rotation.startswith("270")
            else 0
        )
        return cls.outputs(IMAGE=torch.rot90(image, k=turns, dims=[2, 1]))


class ImageFlip(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageFlip",
            "Flip Image",
            "image/transform",
            (
                InputSpec("image", _IMAGE),
                _combo("flip_method", ("x-axis: vertically", "y-axis: horizontally")),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, flip_method: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        return cls.outputs(IMAGE=torch.flip(image, dims=[1 if flip_method.startswith("x") else 2]))


class ImageScaleToMaxDimension(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageScaleToMaxDimension",
            "Scale Image to Max Dimension",
            "image/upscaling",
            (
                InputSpec("image", _IMAGE),
                _combo(
                    "upscale_method",
                    ("area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"),
                ),
                _integer("largest_size", 512, 0, _MAX_RESOLUTION, 1),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, upscale_method: str, largest_size: int) -> Mapping[str, object]:
        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(image)
        height, width = image.shape[1:3]
        if height > width:
            width, height = round(width / height * largest_size), largest_size
        elif width > height:
            height, width = round(height / width * largest_size), largest_size
        else:
            height = width = largest_size
        return cls.outputs(
            IMAGE=utils.common_upscale(
                image.movedim(-1, 1), width, height, upscale_method, "disabled"
            ).movedim(1, -1)
        )


IMAGE_TRANSFORM_NODES = (
    ImageCrop,
    RepeatImageBatch,
    ImageFromBatch,
    ImageAddNoise,
    ImageStitch,
    ResizeAndPadImage,
    ImageRotate,
    ImageFlip,
    ImageScaleToMaxDimension,
)
