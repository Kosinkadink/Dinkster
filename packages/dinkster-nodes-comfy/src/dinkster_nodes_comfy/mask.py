"""Native implementations of ComfyUI's core mask and compositing nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)

_BOOLEAN = TypeExpr.concrete("BOOLEAN")
_COMBO = TypeExpr.concrete("COMBO")
_FLOAT = TypeExpr.concrete("FLOAT")
_IMAGE = TypeExpr.concrete("IMAGE")
_INT = TypeExpr.concrete("INT")
_LATENT = TypeExpr.concrete("LATENT")
_MASK = TypeExpr.concrete("MASK")
_MAX_RESOLUTION = 16384
_MASK_CATEGORY = "image/mask"


def _integer(name: str, default: int, minimum: int, maximum: int, step: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _mask_schema(node_type: str, display_name: str, inputs: tuple[InputSpec, ...]) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=_MASK_CATEGORY,
        inputs=inputs,
        outputs=(OutputSpec("MASK", _MASK),),
    )


def _repeat_to_batch_size(tensor: Any, batch_size: int) -> Any:
    if tensor.shape[0] > batch_size:
        return tensor.narrow(0, 0, batch_size)
    if tensor.shape[0] < batch_size:
        return tensor.repeat(
            [int(-(-batch_size // tensor.shape[0]))] + [1] * (len(tensor.shape) - 1)
        ).narrow(0, 0, batch_size)
    return tensor


def composite(
    destination: Any,
    source: Any,
    x: int,
    y: int,
    mask: Any | None = None,
    multiplier: int = 8,
    resize_source: bool = False,
) -> Any:
    torch = importlib.import_module("torch")
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )
    source = _repeat_to_batch_size(source, destination.shape[0])
    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))
    left, top = x // multiplier, y // multiplier
    right, bottom = left + source.shape[-1], top + source.shape[-2]
    if mask is None:
        resolved_mask: Any = torch.ones_like(source)
    else:
        resolved_mask = mask.to(destination.device, copy=True)
        resolved_mask = torch.nn.functional.interpolate(
            resolved_mask.reshape((-1, 1, resolved_mask.shape[-2], resolved_mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        resolved_mask = _repeat_to_batch_size(resolved_mask, source.shape[0])
    visible_width = destination.shape[-1] - left + min(0, x)
    visible_height = destination.shape[-2] - top + min(0, y)
    resolved_mask = resolved_mask[:, :, :visible_height, :visible_width]
    if resolved_mask.ndim < source.ndim:
        resolved_mask = resolved_mask.unsqueeze(1)
    source_portion: Any = resolved_mask * source[..., :visible_height, :visible_width]
    destination_portion: Any = (torch.ones_like(resolved_mask) - resolved_mask) * destination[
        ..., top:bottom, left:right
    ]
    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination


def _image_alpha_fix(destination: Any, source: Any) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., : destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = torch.nn.functional.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source


class LatentCompositeMasked(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentCompositeMasked",
            display_name="Latent Composite Masked",
            category="model/latent",
            inputs=(
                InputSpec("destination", _LATENT),
                InputSpec("source", _LATENT),
                _integer("x", 0, 0, _MAX_RESOLUTION, 8),
                _integer("y", 0, 0, _MAX_RESOLUTION, 8),
                InputSpec("resize_source", _BOOLEAN, default=False),
                InputSpec("mask", _MASK, required=False),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        destination: dict[str, Any],
        source: dict[str, Any],
        x: int,
        y: int,
        resize_source: bool,
        mask: Any | None = None,
    ) -> Mapping[str, object]:
        destination = materialize_latent(destination)
        source = materialize_latent(source)
        output = destination.copy()
        output["samples"] = composite(
            destination["samples"].clone(), source["samples"], x, y, mask, 8, resize_source
        )
        return cls.outputs(LATENT=output)


class ImageCompositeMasked(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageCompositeMasked",
            display_name="Image Composite Masked",
            category="image/compositing",
            inputs=(
                InputSpec("destination", _IMAGE),
                InputSpec("source", _IMAGE),
                _integer("x", 0, 0, _MAX_RESOLUTION, 1),
                _integer("y", 0, 0, _MAX_RESOLUTION, 1),
                InputSpec("resize_source", _BOOLEAN, default=False),
                InputSpec("mask", _MASK, required=False),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(
        cls,
        *,
        destination: Any,
        source: Any,
        x: int,
        y: int,
        resize_source: bool,
        mask: Any | None = None,
    ) -> Mapping[str, object]:
        destination = materialize_image(destination)
        source = materialize_image(source)
        destination, source = _image_alpha_fix(destination, source)
        output = composite(
            destination.clone().movedim(-1, 1),
            source.movedim(-1, 1),
            x,
            y,
            mask,
            1,
            resize_source,
        ).movedim(1, -1)
        return cls.outputs(IMAGE=output)


class MaskToImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MaskToImage",
            display_name="Convert Mask to Image",
            category=_MASK_CATEGORY,
            inputs=(InputSpec("mask", _MASK),),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, mask: Any) -> Mapping[str, object]:
        mask = materialize_image(mask)
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1)
        return cls.outputs(IMAGE=result.expand(-1, -1, -1, 3))


class ImageToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "ImageToMask",
            "Convert Image to Mask",
            (
                InputSpec("image", _IMAGE),
                InputSpec(
                    "channel",
                    _COMBO,
                    widget=ComboWidget(
                        tuple(ComboOption(value) for value in ("red", "green", "blue", "alpha"))
                    ),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, channel: str) -> Mapping[str, object]:
        image = materialize_image(image)
        return cls.outputs(MASK=image[:, :, :, ("red", "green", "blue", "alpha").index(channel)])


class ImageColorToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "ImageColorToMask",
            "Convert Image Color to Mask",
            (
                InputSpec("image", _IMAGE),
                InputSpec(
                    "color",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=0xFFFFFF, step=1, display="number"),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, color: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        pixels = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        pixels = (
            torch.bitwise_left_shift(pixels[:, :, :, 0], 16)
            + torch.bitwise_left_shift(pixels[:, :, :, 1], 8)
            + pixels[:, :, :, 2]
        )
        return cls.outputs(MASK=torch.where(pixels == color, 1.0, 0).float())


class SolidMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "SolidMask",
            "Create Solid Mask",
            (
                InputSpec(
                    "value", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
                _integer("width", 512, 1, _MAX_RESOLUTION, 1),
                _integer("height", 512, 1, _MAX_RESOLUTION, 1),
            ),
        )

    @classmethod
    def execute(cls, *, value: float, width: int, height: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        return cls.outputs(
            MASK=torch.full(
                (1, height, width),
                value,
                dtype=torch.float32,
                device=management.intermediate_device(),
            )
        )


class InvertMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema("InvertMask", "Invert Mask", (InputSpec("mask", _MASK),))

    @classmethod
    def execute(cls, *, mask: Any) -> Mapping[str, object]:
        return cls.outputs(MASK=1.0 - materialize_image(mask))


class CropMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "CropMask",
            "Crop Mask",
            (
                InputSpec("mask", _MASK),
                _integer("x", 0, 0, _MAX_RESOLUTION, 1),
                _integer("y", 0, 0, _MAX_RESOLUTION, 1),
                _integer("width", 512, 1, _MAX_RESOLUTION, 1),
                _integer("height", 512, 1, _MAX_RESOLUTION, 1),
            ),
        )

    @classmethod
    def execute(cls, *, mask: Any, x: int, y: int, width: int, height: int) -> Mapping[str, object]:
        mask = materialize_image(mask).reshape((-1, mask.shape[-2], mask.shape[-1]))
        return cls.outputs(MASK=mask[:, y : y + height, x : x + width])


class MaskComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "MaskComposite",
            "Combine Masks",
            (
                InputSpec("destination", _MASK),
                InputSpec("source", _MASK),
                _integer("x", 0, 0, _MAX_RESOLUTION, 1),
                _integer("y", 0, 0, _MAX_RESOLUTION, 1),
                InputSpec(
                    "operation",
                    _COMBO,
                    widget=ComboWidget(
                        tuple(
                            ComboOption(value)
                            for value in ("multiply", "add", "subtract", "and", "or", "xor")
                        )
                    ),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, destination: Any, source: Any, x: int, y: int, operation: str
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        destination = materialize_image(destination)
        source = materialize_image(source)
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1])).to(output.device)
        right = min(x + source.shape[-1], destination.shape[-1])
        bottom = min(y + source.shape[-2], destination.shape[-2])
        visible_width, visible_height = right - x, bottom - y
        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = output[:, y:bottom, x:right]
        if operation == "multiply":
            output[:, y:bottom, x:right] = destination_portion * source_portion
        elif operation == "add":
            output[:, y:bottom, x:right] = destination_portion + source_portion
        elif operation == "subtract":
            output[:, y:bottom, x:right] = destination_portion - source_portion
        elif operation == "and":
            output[:, y:bottom, x:right] = torch.bitwise_and(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()
        elif operation == "or":
            output[:, y:bottom, x:right] = torch.bitwise_or(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()
        elif operation == "xor":
            output[:, y:bottom, x:right] = torch.bitwise_xor(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()
        return cls.outputs(MASK=torch.clamp(output, 0.0, 1.0))


class FeatherMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "FeatherMask",
            "Feather Mask",
            (
                InputSpec("mask", _MASK),
                _integer("left", 0, 0, _MAX_RESOLUTION, 1),
                _integer("top", 0, 0, _MAX_RESOLUTION, 1),
                _integer("right", 0, 0, _MAX_RESOLUTION, 1),
                _integer("bottom", 0, 0, _MAX_RESOLUTION, 1),
            ),
        )

    @classmethod
    def execute(
        cls, *, mask: Any, left: int, top: int, right: int, bottom: int
    ) -> Mapping[str, object]:
        mask = materialize_image(mask)
        output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
        left, right = min(left, output.shape[-1]), min(right, output.shape[-1])
        top, bottom = min(top, output.shape[-2]), min(bottom, output.shape[-2])
        for x in range(left):
            output[:, :, x] *= (x + 1.0) / left
        for x in range(right):
            output[:, :, -(x + 1)] *= (x + 1) / right
        for y in range(top):
            output[:, y, :] *= (y + 1) / top
        for y in range(bottom):
            output[:, -(y + 1), :] *= (y + 1) / bottom
        return cls.outputs(MASK=output)


class GrowMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "GrowMask",
            "Grow Mask",
            (
                InputSpec("mask", _MASK),
                _integer("expand", 0, -_MAX_RESOLUTION, _MAX_RESOLUTION, 1),
                InputSpec("tapered_corners", _BOOLEAN, default=True, advanced=True),
            ),
        )

    @classmethod
    def execute(cls, *, mask: Any, expand: int, tapered_corners: bool) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        scipy = importlib.import_module("scipy.ndimage")
        torch = importlib.import_module("torch")
        corner = 0 if tapered_corners else 1
        kernel = numpy.array(((corner, 1, corner), (1, 1, 1), (corner, 1, corner)))
        mask = materialize_image(mask)
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        output: list[Any] = []
        for item in mask:
            result = item.numpy()
            for _ in range(abs(expand)):
                result = (
                    scipy.grey_erosion(result, footprint=kernel)
                    if expand < 0
                    else scipy.grey_dilation(result, footprint=kernel)
                )
            output.append(torch.from_numpy(result))
        return cls.outputs(MASK=torch.stack(output, dim=0))


class ThresholdMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _mask_schema(
            "ThresholdMask",
            "Threshold Mask",
            (
                InputSpec("mask", _MASK),
                InputSpec(
                    "value", _FLOAT, default=0.5, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
            ),
        )

    @classmethod
    def execute(cls, *, mask: Any, value: float) -> Mapping[str, object]:
        return cls.outputs(MASK=(materialize_image(mask) > value).float())


MASK_NODES = (
    LatentCompositeMasked,
    ImageCompositeMasked,
    MaskToImage,
    ImageToMask,
    ImageColorToMask,
    SolidMask,
    InvertMask,
    CropMask,
    MaskComposite,
    FeatherMask,
    GrowMask,
    ThresholdMask,
)
