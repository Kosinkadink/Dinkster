"""Native ComfyUI Porter-Duff and alpha-channel nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from enum import Enum
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    IMAGE_TYPE,
    MASK_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)


class PorterDuffMode(Enum):
    ADD = 0
    CLEAR = 1
    DARKEN = 2
    DST = 3
    DST_ATOP = 4
    DST_IN = 5
    DST_OUT = 6
    DST_OVER = 7
    LIGHTEN = 8
    MULTIPLY = 9
    OVERLAY = 10
    SCREEN = 11
    SRC = 12
    SRC_ATOP = 13
    SRC_IN = 14
    SRC_OUT = 15
    SRC_OVER = 16
    XOR = 17


def _resize_mask(mask: Any, shape: tuple[int, int]) -> Any:
    torch = importlib.import_module("torch")
    return torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(shape[0], shape[1]),
        mode="bilinear",
    ).squeeze(1)


def _porter_duff_composite(
    source_image: Any,
    source_alpha: Any,
    destination_image: Any,
    destination_alpha: Any,
    mode: PorterDuffMode,
) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    source_alpha = 1 - source_alpha
    destination_alpha = 1 - destination_alpha
    source_image = source_image * source_alpha
    destination_image = destination_image * destination_alpha

    if mode == PorterDuffMode.ADD:
        output_alpha = torch.clamp(source_alpha + destination_alpha, 0, 1)
        output_image = torch.clamp(source_image + destination_image, 0, 1)
    elif mode == PorterDuffMode.CLEAR:
        output_alpha = torch.zeros_like(destination_alpha)
        output_image = torch.zeros_like(destination_image)
    elif mode == PorterDuffMode.DARKEN:
        output_alpha = source_alpha + destination_alpha - source_alpha * destination_alpha
        output_image = (
            (1 - destination_alpha) * source_image
            + (1 - source_alpha) * destination_image
            + torch.min(source_image * destination_alpha, destination_image * source_alpha)
        )
    elif mode == PorterDuffMode.DST:
        output_alpha = destination_alpha
        output_image = destination_image
    elif mode == PorterDuffMode.DST_ATOP:
        output_alpha = source_alpha
        output_image = source_alpha * destination_image + (1 - destination_alpha) * source_image
    elif mode == PorterDuffMode.DST_IN:
        output_alpha = source_alpha * destination_alpha
        output_image = destination_image * source_alpha
    elif mode == PorterDuffMode.DST_OUT:
        output_alpha = (1 - source_alpha) * destination_alpha
        output_image = (1 - source_alpha) * destination_image
    elif mode == PorterDuffMode.DST_OVER:
        output_alpha = destination_alpha + (1 - destination_alpha) * source_alpha
        output_image = destination_image + (1 - destination_alpha) * source_image
    elif mode == PorterDuffMode.LIGHTEN:
        output_alpha = source_alpha + destination_alpha - source_alpha * destination_alpha
        output_image = (
            (1 - destination_alpha) * source_image
            + (1 - source_alpha) * destination_image
            + torch.max(source_image * destination_alpha, destination_image * source_alpha)
        )
    elif mode == PorterDuffMode.MULTIPLY:
        output_alpha = source_alpha + destination_alpha - source_alpha * destination_alpha
        output_image = (
            (1 - destination_alpha) * source_image
            + (1 - source_alpha) * destination_image
            + source_image * destination_image
        )
    elif mode == PorterDuffMode.OVERLAY:
        output_alpha = source_alpha + destination_alpha - source_alpha * destination_alpha
        overlap = torch.where(
            2 * destination_image < destination_alpha,
            2 * source_image * destination_image,
            source_alpha * destination_alpha
            - 2 * (source_alpha - source_image) * (destination_alpha - destination_image),
        )
        output_image = (
            (1 - destination_alpha) * source_image
            + (1 - source_alpha) * destination_image
            + overlap
        )
    elif mode == PorterDuffMode.SCREEN:
        output_alpha = source_alpha + destination_alpha - source_alpha * destination_alpha
        output_image = source_image + destination_image - source_image * destination_image
    elif mode == PorterDuffMode.SRC:
        output_alpha = source_alpha
        output_image = source_image
    elif mode == PorterDuffMode.SRC_ATOP:
        output_alpha = destination_alpha
        output_image = destination_alpha * source_image + (1 - source_alpha) * destination_image
    elif mode == PorterDuffMode.SRC_IN:
        output_alpha = source_alpha * destination_alpha
        output_image = source_image * destination_alpha
    elif mode == PorterDuffMode.SRC_OUT:
        output_alpha = (1 - destination_alpha) * source_alpha
        output_image = (1 - destination_alpha) * source_image
    elif mode == PorterDuffMode.SRC_OVER:
        output_alpha = source_alpha + (1 - source_alpha) * destination_alpha
        output_image = source_image + (1 - source_alpha) * destination_image
    elif mode == PorterDuffMode.XOR:
        output_alpha = (1 - destination_alpha) * source_alpha + (
            1 - source_alpha
        ) * destination_alpha
        output_image = (1 - destination_alpha) * source_image + (
            1 - source_alpha
        ) * destination_image
    else:
        raise ValueError(f"Unsupported Porter-Duff mode: {mode}")

    output_image = torch.where(
        output_alpha > 1e-5, output_image / output_alpha, torch.zeros_like(output_image)
    )
    return torch.clamp(output_image, 0, 1), 1 - output_alpha


class PorterDuffImageComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PorterDuffImageComposite",
            display_name="Porter-Duff Image Composite",
            category="image/compositing",
            inputs=(
                InputSpec("source", _IMAGE),
                InputSpec("source_alpha", _MASK),
                InputSpec("destination", _IMAGE),
                InputSpec("destination_alpha", _MASK),
                InputSpec(
                    "mode",
                    _COMBO,
                    default=PorterDuffMode.DST.name,
                    widget=ComboWidget(tuple(ComboOption(mode.name) for mode in PorterDuffMode)),
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE), OutputSpec("MASK", _MASK)),
        )

    @classmethod
    def execute(
        cls,
        *,
        source: Any,
        source_alpha: Any,
        destination: Any,
        destination_alpha: Any,
        mode: str,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        source, destination = materialize_image(source), materialize_image(destination)
        batch_size = min(len(source), len(source_alpha), len(destination), len(destination_alpha))
        output_images = []
        output_alphas = []

        for index in range(batch_size):
            source_image = source[index]
            destination_image = destination[index]
            if source_image.shape[2] != destination_image.shape[2]:
                raise AssertionError("source and destination must have the same number of channels")

            source_mask = source_alpha[index].unsqueeze(2)
            destination_mask = destination_alpha[index].unsqueeze(2)
            if destination_mask.shape[:2] != destination_image.shape[:2]:
                destination_mask = (
                    utils.common_upscale(
                        destination_mask.unsqueeze(0).permute(0, 3, 1, 2),
                        destination_image.shape[1],
                        destination_image.shape[0],
                        "bicubic",
                        "center",
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
            if source_image.shape != destination_image.shape:
                source_image = (
                    utils.common_upscale(
                        source_image.unsqueeze(0).permute(0, 3, 1, 2),
                        destination_image.shape[1],
                        destination_image.shape[0],
                        "bicubic",
                        "center",
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
            if source_mask.shape != destination_mask.shape:
                source_mask = (
                    utils.common_upscale(
                        source_mask.unsqueeze(0).permute(0, 3, 1, 2),
                        destination_mask.shape[1],
                        destination_mask.shape[0],
                        "bicubic",
                        "center",
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )

            output_image, output_alpha = _porter_duff_composite(
                source_image,
                source_mask,
                destination_image,
                destination_mask,
                PorterDuffMode[mode],
            )
            output_images.append(output_image)
            output_alphas.append(output_alpha.squeeze(2))

        return cls.outputs(IMAGE=torch.stack(output_images), MASK=torch.stack(output_alphas))


class SplitImageWithAlpha(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplitImageWithAlpha",
            display_name="Split Image with Alpha",
            category="image/compositing",
            inputs=(InputSpec("image", _IMAGE),),
            outputs=(OutputSpec("IMAGE", _IMAGE), OutputSpec("MASK", _MASK)),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        images = [item[:, :, :3] for item in image]
        alphas = [
            item[:, :, 3] if item.shape[2] > 3 else torch.ones_like(item[:, :, 0]) for item in image
        ]
        return cls.outputs(IMAGE=torch.stack(images), MASK=1.0 - torch.stack(alphas))


class JoinImageWithAlpha(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="JoinImageWithAlpha",
            display_name="Join Image with Alpha",
            category="image/compositing",
            inputs=(InputSpec("image", _IMAGE), InputSpec("alpha", _MASK)),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: Any, alpha: Any) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(image)
        batch_size = max(len(image), len(alpha))
        alpha = 1.0 - _resize_mask(alpha.to(image), image.shape[1:3])
        alpha = utils.repeat_to_batch_size(alpha, batch_size)
        image = utils.repeat_to_batch_size(image, batch_size)
        return cls.outputs(IMAGE=torch.cat((image[..., :3], alpha.unsqueeze(-1)), dim=-1))


COMPOSITING_NODES = (PorterDuffImageComposite, SplitImageWithAlpha, JoinImageWithAlpha)
